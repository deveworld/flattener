"""Independent pixel-edge, segmentation-gate, and page-coverage checks."""

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from flattener.contracts import FlattenerError, Status
from flattener.regions.geometry import (
    PageRegion,
    RegionConfig,
    detect_region,
    ensure_page_coverage,
    page_coverage,
    region_map_to_source,
    virtual_page_box,
)


def _rectangle_map(left, top, right, bottom, *, source_size=(100, 100), shape=(7, 9)):
    """Explicit normalized source coordinates, independent of project helpers."""
    height, width = source_size
    x = np.linspace(left, right, shape[1], dtype=np.float64)
    y = np.linspace(top, bottom, shape[0], dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return np.stack((2 * xx / width - 1, 2 * yy / height - 1), axis=-1).astype(np.float32)


def _assert_error(status, callback):
    with pytest.raises(FlattenerError) as caught:
        callback()
    assert caught.value.status == status
    assert caught.value.reason_code


def test_region_edges_scale_from_non_square_probability_grid_to_source_pixels():
    probability = np.zeros((10, 12), dtype=np.float64)
    probability[2:6, 3:9] = 0.9
    original = probability.copy()
    estimate = detect_region(probability, (100, 240), config=RegionConfig(padding_fraction=0))
    assert estimate.region == PageRegion(left=60, top=20, right=180, bottom=60)
    np.testing.assert_array_equal(estimate.mask, probability >= 0.5)
    assert estimate.mask.dtype == np.bool_
    assert isinstance(estimate.metadata, dict)
    np.testing.assert_array_equal(probability, original)
    assert not np.shares_memory(estimate.mask, probability)


def test_padding_is_relative_to_the_detected_width_and_height_and_clamped():
    probability = np.zeros((10, 12), np.float32)
    probability[2:6, 3:9] = 0.9
    estimate = detect_region(probability, (100, 240))
    actual = estimate.region
    np.testing.assert_allclose(
        [actual.left, actual.top, actual.right, actual.bottom],
        [55.2, 18.4, 184.8, 61.6],
        rtol=0,
        atol=1e-6,
    )
    edge_probability = np.zeros((10, 10), np.float32)
    edge_probability[:5, :4] = 1
    edge = detect_region(edge_probability, (100, 200)).region
    assert edge.left == edge.top == 0
    assert edge.right == pytest.approx(83.2)
    assert edge.bottom == pytest.approx(52)


def test_diagonally_touching_foreground_uses_eight_connectivity():
    probability = np.zeros((10, 10), np.float32)
    probability[np.arange(2, 6), np.arange(2, 6)] = 1
    estimate = detect_region(probability, (100, 100), config=RegionConfig(padding_fraction=0))
    assert estimate.mask.sum() == 4
    assert estimate.region == PageRegion(20, 20, 60, 60)


def test_dominant_component_excludes_separate_small_foreground_from_mask_and_box():
    probability = np.zeros((20, 20), np.float32)
    probability[4:12, 5:13] = 0.95
    probability[17:19, 17:19] = 1
    estimate = detect_region(probability, (200, 400), config=RegionConfig(padding_fraction=0))
    assert estimate.mask.sum() == 64
    assert not estimate.mask[17:19, 17:19].any()
    assert estimate.region == PageRegion(100, 40, 260, 120)


@pytest.mark.parametrize("case", ["empty", "tiny", "whole_image", "uncertain", "ambiguous"])
def test_detection_refuses_unusable_masks_instead_of_silently_returning_full_image(case):
    probability = np.zeros((20, 20), np.float32)
    if case == "tiny":
        probability[7, 7] = 1
    elif case == "whole_image":
        probability[:] = 1
    elif case == "uncertain":
        probability[4:14, 4:14] = 0.6
    elif case == "ambiguous":
        probability[2:7, 2:7] = 0.9
        probability[12:17, 12:17] = 0.9
    _assert_error(Status.MODEL_ERROR, lambda: detect_region(probability, (200, 200)))


def test_confidence_gate_measures_selected_component_not_the_background():
    probability = np.full((20, 20), 0.49, np.float32)
    probability[5:15, 5:15] = 0.9
    estimate = detect_region(probability, (40, 60))
    assert estimate.mask.sum() == 100


@pytest.mark.parametrize(
    "probability",
    [
        [[0.0, 1.0]],
        np.full((10, 10), np.nan, np.float32),
        np.full((10, 10), 1.01, np.float32),
    ],
)
def test_invalid_probability_arguments_raise_input_error(probability):
    _assert_error(Status.INPUT_ERROR, lambda: detect_region(probability, (100, 100)))


@pytest.mark.parametrize("source_size", [(0, 10), (10, True)])
def test_detector_rejects_invalid_source_size(source_size):
    _assert_error(
        Status.INPUT_ERROR,
        lambda: detect_region(np.zeros((10, 10), np.float32), source_size),
    )


def test_numpy_integer_source_dimensions_cannot_overflow_the_resource_limit():
    probability = np.zeros((10, 10), np.float32)
    probability[2:8, 2:8] = 0.9
    with np.errstate(over="raise"):
        _assert_error(
            Status.INPUT_ERROR,
            lambda: detect_region(probability, (np.uint16(5000), np.uint16(5000))),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -0.1},
        {"min_area_fraction": 0.8, "max_area_fraction": 0.2},
        {"max_boundary_expansion": np.inf},
    ],
)
def test_invalid_config_is_rejected_before_detection(kwargs):
    def detect():
        config = RegionConfig(**kwargs)
        return detect_region(np.zeros((10, 10), np.float32), (100, 100), config=config)

    _assert_error(Status.INPUT_ERROR, detect)


def test_huge_numeric_region_inputs_raise_structured_errors_without_type_or_cast_leaks():
    huge = 10**1000
    _assert_error(Status.INPUT_ERROR, lambda: RegionConfig(threshold=huge))
    _assert_error(Status.INPUT_ERROR, lambda: PageRegion(0, 0, huge, 10))
    _assert_error(Status.INPUT_ERROR, lambda: virtual_page_box(PageRegion(0, 0, 10, 10), huge))


def test_finite_edges_with_overflowing_extent_are_rejected():
    _assert_error(Status.INPUT_ERROR, lambda: PageRegion(-1e308, 0, 1e308, 10))


def test_supported_real_config_scalars_are_canonical_and_metadata_is_strict_json():
    config = RegionConfig(threshold=np.float32(0.5), min_dominance=Fraction(9, 10))
    probability = np.zeros((10, 10), np.float32)
    probability[2:8, 2:8] = 0.9
    estimate = detect_region(probability, (100, 100), config=config)
    assert type(config.threshold) is float
    assert type(config.min_dominance) is float
    json.dumps(estimate.metadata, allow_nan=False)


def test_virtual_box_is_centered_and_extends_beyond_the_source_without_clamping():
    box = virtual_page_box(PageRegion(0, 0, 80, 40), 0.5)
    assert box == PageRegion(-40, -20, 120, 60)
    assert virtual_page_box(PageRegion(20, 30, 80, 70), 1) == PageRegion(20, 30, 80, 70)


@pytest.mark.parametrize("occupancy", [0, True])
def test_virtual_box_rejects_invalid_occupancy(occupancy):
    _assert_error(
        Status.INPUT_ERROR, lambda: virtual_page_box(PageRegion(10, 20, 90, 80), occupancy)
    )


@pytest.mark.parametrize("edges", [(5, 2, 4, 8), (0, 0, np.inf, 9)])
def test_reversed_degenerate_or_nonfinite_boxes_are_rejected(edges):
    _assert_error(Status.INPUT_ERROR, lambda: virtual_page_box(PageRegion(*edges), 0.68))
    _assert_error(
        Status.INPUT_ERROR,
        lambda: region_map_to_source(np.zeros((2, 3, 2), np.float32), PageRegion(*edges), (20, 30)),
    )


def test_map_conversion_uses_pixel_edges_and_xy_order_for_non_square_source_and_box():
    mapping = np.array(
        [[[-1, -1], [0, 0], [1, 1]], [[-0.5, 0.5], [0.5, -0.5], [2, -2]]],
        dtype=np.float32,
    )
    original = mapping.copy()
    converted = region_map_to_source(mapping, PageRegion(20, 10, 100, 70), (80, 160))
    expected = np.array(
        [[[-0.75, -0.75], [-0.25, 0], [0.25, 0.75]], [[-0.5, 0.375], [0, -0.375], [0.75, -1.5]]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(converted, expected)
    assert converted.dtype == np.float32
    assert not np.shares_memory(converted, mapping)
    np.testing.assert_array_equal(mapping, original)


def test_identity_virtual_box_preserves_maps_and_translation_roundtrips():
    mapping = np.array([[[-0.8, -0.4], [0.6, -0.4]], [[-0.8, 0.7], [0.6, 0.7]]], np.float32)
    source_size = (120, 200)
    identity = region_map_to_source(mapping, PageRegion(0, 0, 200, 120), source_size)
    np.testing.assert_array_equal(identity, mapping)
    translated = region_map_to_source(mapping, PageRegion(30, -12, 230, 108), source_size)
    np.testing.assert_allclose(translated, mapping + [0.3, -0.2], rtol=0, atol=1e-7)
    recovered = region_map_to_source(translated, PageRegion(-30, 12, 170, 132), source_size)
    np.testing.assert_allclose(recovered, mapping, rtol=0, atol=1e-7)


def test_virtual_box_map_is_not_clipped_even_when_whole_box_exceeds_source():
    mapping = np.array([[[-1, -1], [1, -1]], [[-1, 1], [1, 1]]], np.float32)
    converted = region_map_to_source(mapping, PageRegion(-50, -20, 150, 120), (100, 100))
    np.testing.assert_array_equal(
        converted,
        np.array([[[-2, -1.4], [2, -1.4]], [[-2, 1.4], [2, 1.4]]], np.float32),
    )


@pytest.mark.parametrize(
    "mapping",
    [
        np.zeros((3, 4, 2), np.float64),
        np.full((3, 4, 2), np.nan, np.float32),
    ],
)
def test_map_conversion_rejects_invalid_map_arguments(mapping):
    _assert_error(
        Status.INPUT_ERROR,
        lambda: region_map_to_source(mapping, PageRegion(0, 0, 100, 100), (100, 100)),
    )


def test_coverage_measures_mask_content_in_polygon_not_rectangle_area_or_mask_mean():
    mask = np.zeros((100, 100), np.bool_)
    mask[40:60, 40:60] = True
    assert page_coverage(_rectangle_map(20, 20, 80, 80), mask) == 1
    assert page_coverage(_rectangle_map(0, 0, 20, 20), mask) == 0
    assert page_coverage(_rectangle_map(0, 0, 50, 100), mask) == pytest.approx(0.5)


def test_identity_pixel_center_boundary_includes_last_row_and_column():
    mask = np.ones((10, 20), np.bool_)
    mapping = _rectangle_map(0.5, 0.5, 19.5, 9.5, source_size=(10, 20), shape=(10, 20))
    assert page_coverage(mapping, mask) == 1


def test_coverage_uses_all_boundary_samples_not_just_the_four_corners():
    mask = np.zeros((100, 100), np.bool_)
    mask[21:28, 46:54] = True
    mapping = _rectangle_map(20, 20, 80, 80, shape=(7, 9))
    mapping[0, 3:6, 1] = 2 * 40 / 100 - 1
    assert page_coverage(mapping, mask) == 0


def test_concave_polygon_coverage_excludes_large_open_notch():
    vertices = np.array(
        [[10, 10], [30, 10], [30, 70], [70, 70], [70, 10], [90, 10], [90, 90], [10, 90]],
        dtype=np.float32,
    )
    # This 2x4 grid represents the eight-point boundary in outer-perimeter order.
    # It is a coverage-only fixture: the renderer separately validates local folds.
    mapping = np.stack((vertices[:4], vertices[4:][::-1]), axis=0) / 50 - 1
    mask = np.ones((100, 100), np.bool_)
    assert page_coverage(mapping, mask) == pytest.approx((80 * 80 - 40 * 60) / 10000)
    mask[:] = False
    mask[20:60, 40:60] = True
    assert page_coverage(mapping, mask) == 0


@pytest.mark.parametrize("row", [0, 7])
def test_nonhorizontal_polygon_vertex_on_mask_center_is_included_at_y_extrema(row):
    mapping = np.array([[[0, -0.875], [0.875, 0]], [[-0.875, 0], [0, 0.875]]], np.float32)
    mask = np.zeros((8, 9), np.bool_)
    mask[row, 4] = True
    assert page_coverage(mapping, mask) == 1


def test_coverage_is_source_resolution_independent_for_same_normalized_mask_regions():
    mapping = _rectangle_map(0, 0, 50, 100)
    coarse_mask = np.zeros((10, 10), np.bool_)
    coarse_mask[4:6, 4:6] = True
    fine_mask = np.repeat(np.repeat(coarse_mask, 10, axis=0), 10, axis=1)
    assert page_coverage(mapping, coarse_mask) == pytest.approx(0.5)
    assert page_coverage(mapping, fine_mask) == pytest.approx(0.5)


@pytest.mark.parametrize("rectangle", [(50.1, 0, 80.1, 100), (19.9, 0, 49.9, 100)])
def test_coverage_counts_pixel_centers_without_integer_rasterizer_edge_bias(rectangle):
    mask = np.zeros((100, 100), np.bool_)
    mask[40:60, 40:60] = True
    # Pixel centers 50.5..59.5 or 40.5..49.5: exactly ten of twenty columns/rows.
    assert page_coverage(_rectangle_map(*rectangle), mask) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "mapping",
    [
        np.zeros((1, 10, 2), np.float32),
        np.full((10, 10, 2), np.nan, np.float32),
    ],
)
def test_coverage_rejects_invalid_maps(mapping):
    _assert_error(Status.INPUT_ERROR, lambda: page_coverage(mapping, np.ones((10, 10), bool)))


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((10, 10), np.bool_),
        np.ones((10, 10), np.float32),
    ],
)
def test_coverage_rejects_invalid_or_empty_masks(mask):
    _assert_error(Status.INPUT_ERROR, lambda: page_coverage(_rectangle_map(0, 0, 100, 100), mask))


def test_already_covered_map_is_copied_without_change_or_aliasing():
    mapping = _rectangle_map(20, 20, 80, 80)
    mask = np.zeros((100, 100), np.bool_)
    mask[30:70, 30:70] = True
    adjusted, metadata = ensure_page_coverage(mapping, mask, config=RegionConfig())
    np.testing.assert_array_equal(adjusted, mapping)
    assert not np.shares_memory(adjusted, mapping)
    assert isinstance(metadata, dict)
    assert metadata["coverage_is_ground_truth"] is False
    assert metadata["initial_estimated_page_coverage"] == 1
    assert metadata["final_estimated_page_coverage"] == 1
    assert metadata["boundary_expansion"] == 1


def test_coverage_expansion_is_bounded_centered_uniform_and_does_not_fold():
    mapping = _rectangle_map(30, 30, 70, 70, shape=(9, 11))
    original = mapping.copy()
    mask = np.zeros((100, 100), np.bool_)
    mask[28:72, 28:72] = True
    before = page_coverage(mapping, mask)
    assert before < 0.98
    adjusted, metadata = ensure_page_coverage(mapping, mask, config=RegionConfig())
    assert page_coverage(adjusted, mask) >= 0.98
    assert isinstance(metadata, dict)
    assert adjusted.dtype == np.float32
    assert adjusted.shape == mapping.shape
    assert not np.shares_memory(adjusted, mapping)
    center = mapping.mean(axis=(0, 1), dtype=np.float64)
    np.testing.assert_allclose(adjusted.mean(axis=(0, 1)), center, atol=1e-7, rtol=0)
    factor = float((adjusted[0, 0, 0] - center[0]) / (mapping[0, 0, 0] - center[0]))
    assert 1 < factor <= 1.200001
    np.testing.assert_allclose(adjusted, center + factor * (mapping - center), atol=2e-7)
    assert (np.diff(adjusted[..., 0], axis=1) > 0).all()
    assert (np.diff(adjusted[..., 1], axis=0) > 0).all()
    np.testing.assert_array_equal(mapping, original)


def test_uncoverable_page_fails_instead_of_exceeding_the_expansion_budget():
    mapping = _rectangle_map(40, 40, 60, 60)
    original = mapping.copy()
    mask = np.zeros((100, 100), np.bool_)
    mask[20:80, 20:80] = True
    _assert_error(
        Status.MODEL_ERROR,
        lambda: ensure_page_coverage(mapping, mask, config=RegionConfig()),
    )
    np.testing.assert_array_equal(mapping, original)


def test_coverage_requirement_is_an_explicit_configurable_gate():
    mask = np.zeros((100, 100), np.bool_)
    mask[40:60, 40:60] = True
    mapping = _rectangle_map(0, 0, 50, 100)
    config = replace(RegionConfig(), required_coverage=0.5, max_boundary_expansion=1)
    adjusted, _ = ensure_page_coverage(mapping, mask, config=config)
    np.testing.assert_array_equal(adjusted, mapping)
