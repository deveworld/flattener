"""Independent pixel-edge, region-detection and page-coverage checks."""

import numpy as np
import pytest

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


def test_detected_region_scales_to_source_pixels_pads_and_keeps_the_dominant_part():
    probability = np.zeros((10, 12), dtype=np.float64)
    probability[2:6, 3:9] = 0.9
    original = probability.copy()
    estimate = detect_region(probability, (100, 240), config=RegionConfig(padding_fraction=0))
    assert estimate.region == PageRegion(left=60, top=20, right=180, bottom=60)
    assert estimate.mask.dtype == np.bool_
    np.testing.assert_array_equal(estimate.mask, probability >= 0.5)
    np.testing.assert_array_equal(probability, original)

    # Padding is a share of the detected width and height, clamped at the image edge.
    padded = detect_region(probability, (100, 240)).region
    np.testing.assert_allclose(
        [padded.left, padded.top, padded.right, padded.bottom],
        [55.2, 18.4, 184.8, 61.6],
        rtol=0,
        atol=1e-6,
    )
    corner = np.zeros((10, 10), np.float32)
    corner[:5, :4] = 1
    edge = detect_region(corner, (100, 200)).region
    assert edge.left == edge.top == 0
    assert edge.right == pytest.approx(83.2)
    assert edge.bottom == pytest.approx(52)

    # A separate small blob stays out of the mask and the box.
    two = np.zeros((20, 20), np.float32)
    two[4:12, 5:13] = 0.95
    two[17:19, 17:19] = 1
    estimate = detect_region(two, (200, 400), config=RegionConfig(padding_fraction=0))
    assert estimate.mask.sum() == 64
    assert not estimate.mask[17:19, 17:19].any()
    assert estimate.region == PageRegion(100, 40, 260, 120)


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
    np.testing.assert_array_equal(mapping, original)


def test_coverage_is_the_share_of_mask_pixel_centers_inside_the_outline():
    mask = np.zeros((100, 100), np.bool_)
    mask[40:60, 40:60] = True
    assert page_coverage(_rectangle_map(20, 20, 80, 80), mask) == 1
    assert page_coverage(_rectangle_map(0, 0, 20, 20), mask) == 0
    assert page_coverage(_rectangle_map(0, 0, 50, 100), mask) == pytest.approx(0.5)
    # Pixel centers 50.5..59.5: exactly ten of twenty columns, with no rasterizer edge bias.
    assert page_coverage(_rectangle_map(50.1, 0, 80.1, 100), mask) == pytest.approx(0.5)


def test_coverage_follows_every_boundary_sample_of_the_outline():
    # A dent in the top edge between the corners leaves the small mask outside.
    mask = np.zeros((100, 100), np.bool_)
    mask[21:28, 46:54] = True
    mapping = _rectangle_map(20, 20, 80, 80, shape=(7, 9))
    mapping[0, 3:6, 1] = 2 * 40 / 100 - 1
    assert page_coverage(mapping, mask) == 0

    # A 2x4 grid holding an eight-point concave outline in perimeter order.
    vertices = np.array(
        [[10, 10], [30, 10], [30, 70], [70, 70], [70, 10], [90, 10], [90, 90], [10, 90]],
        dtype=np.float32,
    )
    notched = np.stack((vertices[:4], vertices[4:][::-1]), axis=0) / 50 - 1
    full = np.ones((100, 100), np.bool_)
    assert page_coverage(notched, full) == pytest.approx((80 * 80 - 40 * 60) / 10000)
    inside_notch = np.zeros((100, 100), np.bool_)
    inside_notch[20:60, 40:60] = True
    assert page_coverage(notched, inside_notch) == 0
