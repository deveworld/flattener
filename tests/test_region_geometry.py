"""Independent pixel-edge, segmentation-gate, and page-coverage checks."""

import numpy as np
import pytest

from flattener.contracts import FlattenerError, Status
from flattener.regions.geometry import (
    PageRegion,
    RegionConfig,
    detect_region,
    page_coverage,
    region_map_to_source,
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


def test_dominant_component_excludes_separate_small_foreground_from_mask_and_box():
    probability = np.zeros((20, 20), np.float32)
    probability[4:12, 5:13] = 0.95
    probability[17:19, 17:19] = 1
    estimate = detect_region(probability, (200, 400), config=RegionConfig(padding_fraction=0))
    assert estimate.mask.sum() == 64
    assert not estimate.mask[17:19, 17:19].any()
    assert estimate.region == PageRegion(100, 40, 260, 120)


@pytest.mark.parametrize("case", ["tiny", "uncertain"])
def test_detection_refuses_unusable_masks_instead_of_silently_returning_full_image(case):
    probability = np.zeros((20, 20), np.float32)
    if case == "tiny":
        probability[7, 7] = 1
    else:  # uncertain
        probability[4:14, 4:14] = 0.6
    _assert_error(Status.MODEL_ERROR, lambda: detect_region(probability, (200, 200)))


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


def test_coverage_measures_mask_content_in_polygon_not_rectangle_area_or_mask_mean():
    mask = np.zeros((100, 100), np.bool_)
    mask[40:60, 40:60] = True
    assert page_coverage(_rectangle_map(20, 20, 80, 80), mask) == 1
    assert page_coverage(_rectangle_map(0, 0, 20, 20), mask) == 0
    assert page_coverage(_rectangle_map(0, 0, 50, 100), mask) == pytest.approx(0.5)


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


def test_coverage_counts_pixel_centers_without_integer_rasterizer_edge_bias():
    rectangle = (50.1, 0, 80.1, 100)
    mask = np.zeros((100, 100), np.bool_)
    mask[40:60, 40:60] = True
    # Pixel centers 50.5..59.5 or 40.5..49.5: exactly ten of twenty columns/rows.
    assert page_coverage(_rectangle_map(*rectangle), mask) == pytest.approx(0.5)
