"""Coordinate contracts checked against explicit and analytic expectations."""

import numpy as np
import pytest

from flattener.contracts import FlattenerError, RenderConfig, Status
from flattener.coordinates import (
    identity_map,
    normalized_to_pixel,
    pixel_to_normalized,
    resize_backward_map,
)


def test_width_four_centers_and_physical_boundaries():
    pixels = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [-0.5, -0.5], [3.5, 0.5]], np.float32)
    normalized = np.array(
        [[-0.75, 0], [-0.25, 0], [0.25, 0], [0.75, 0], [-1, -1], [1, 1]], np.float32
    )
    np.testing.assert_array_equal(pixel_to_normalized(pixels, (1, 4)), normalized)
    np.testing.assert_array_equal(normalized_to_pixel(normalized, (1, 4)), pixels)


def test_coordinate_order_and_non_square_source():
    pixels = np.array([[[0, 0], [4, 2]], [[2, 1], [-0.5, 2.5]]], np.float32)
    normalized = np.array([[[-0.8, -2 / 3], [0.8, 2 / 3]], [[0, 0], [-1, 1]]], np.float32)
    np.testing.assert_allclose(pixel_to_normalized(pixels, (3, 5)), normalized, rtol=0, atol=1e-6)
    np.testing.assert_allclose(normalized_to_pixel(normalized, (3, 5)), pixels, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1, 1), [[[0, 0]]]),
        ((1, 4), [[[-0.75, 0], [-0.25, 0], [0.25, 0], [0.75, 0]]]),
        ((4, 1), [[[0, -0.75]], [[0, -0.25]], [[0, 0.25]], [[0, 0.75]]]),
        (
            (2, 3),
            [[[-2 / 3, -0.5], [0, -0.5], [2 / 3, -0.5]], [[-2 / 3, 0.5], [0, 0.5], [2 / 3, 0.5]]],
        ),
    ],
)
def test_identity_explicit_values_and_singleton_axes(size, expected):
    result = identity_map(size)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.array(expected), rtol=0, atol=1e-6)


@pytest.mark.parametrize("conversion", [pixel_to_normalized, normalized_to_pixel])
def test_conversion_accepts_single_point_and_does_not_mutate_views(conversion):
    base = np.arange(24, dtype=np.float32).reshape(6, 4)
    points = base[::2, ::2]
    before = base.copy()
    result = conversion(points, (7, 11))
    assert result.dtype == np.float32
    assert result.shape == points.shape
    assert not np.shares_memory(result, points)
    np.testing.assert_array_equal(base, before)
    assert conversion(np.array([0, 0], np.float32), (1, 1)).shape == (2,)


@pytest.mark.parametrize("shift", [(0, 0), (0.125, -0.25)])
def test_coarse_identity_and_translation_preserve_all_dense_centers(shift):
    coarse = np.array(
        [[[-2 / 3, -0.5], [0, -0.5], [2 / 3, -0.5]], [[-2 / 3, 0.5], [0, 0.5], [2 / 3, 0.5]]],
        np.float32,
    )
    coarse += np.array(shift, np.float32)
    before = coarse.copy()
    result = resize_backward_map(coarse, (7, 11))
    expected = np.array(
        [
            [[(2 * j + 1) / 11 - 1 + shift[0], (2 * i + 1) / 7 - 1 + shift[1]] for j in range(11)]
            for i in range(7)
        ]
    )
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(coarse, before)


def test_affine_expansion_has_analytic_interior_and_extended_residual_boundary():
    # F(x, y) = (1.2*x + 0.1*y + 0.125, -0.2*x + 0.8*y - 0.25).
    coarse = np.array(
        [[[-0.525, -0.55], [0.675, -0.75]], [[-0.425, 0.25], [0.775, 0.05]]],
        np.float32,
    )
    result = resize_backward_map(coarse, (4, 6))
    expected = np.empty((4, 6, 2), np.float64)
    full_affine = np.empty_like(expected)
    for i, y in enumerate([-0.75, -0.25, 0.25, 0.75]):
        for j, x in enumerate([-5 / 6, -0.5, -1 / 6, 1 / 6, 0.5, 5 / 6]):
            residual_x = min(0.5, max(-0.5, x))
            residual_y = min(0.5, max(-0.5, y))
            expected[i, j] = (
                x + 0.2 * residual_x + 0.1 * residual_y + 0.125,
                y - 0.2 * residual_x - 0.2 * residual_y - 0.25,
            )
            full_affine[i, j] = (1.2 * x + 0.1 * y + 0.125, -0.2 * x + 0.8 * y - 0.25)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result[1:3, 1:5], full_affine[1:3, 1:5], rtol=0, atol=1e-6)
    assert np.max(np.abs(result[0] - full_affine[0])) > 0.05


def test_same_size_map_is_an_exact_independent_copy_even_for_a_noncontiguous_view():
    coarse = np.linspace(-2, 2, 48, dtype=np.float32).reshape(4, 6, 2)[::2, ::2]
    coarse.flags.writeable = False
    result = resize_backward_map(coarse, (2, 3))
    np.testing.assert_array_equal(result, coarse)
    assert not np.shares_memory(result, coarse)


@pytest.mark.parametrize("bad_size", [(0, 2), (2,)])
def test_identity_rejects_invalid_sizes(bad_size):
    with pytest.raises(FlattenerError) as caught:
        identity_map(bad_size)
    assert caught.value.status == Status.INPUT_ERROR


def test_resource_limits_are_checked_before_dense_allocation():
    config = RenderConfig(max_output_pixels=10, max_side=8)
    with pytest.raises(FlattenerError) as caught:
        identity_map((3, 4), config=config)
    assert caught.value.reason_code == "size_limit_exceeded"
    with pytest.raises(FlattenerError) as caught:
        resize_backward_map(np.zeros((2, 2, 2), np.float32), (3, 4), config=config)
    assert caught.value.reason_code == "size_limit_exceeded"
    with pytest.raises(FlattenerError) as caught:
        identity_map((1, 8193))
    assert caught.value.reason_code == "size_limit_exceeded"
    assert identity_map((1, 8193), config=RenderConfig(max_side=8193)).shape == (1, 8193, 2)


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        ([[0, 0]], "invalid_type"),
        (np.zeros((2, 2), np.float64), "invalid_dtype"),
        (np.zeros((3,), np.float32), "invalid_shape"),
        (np.array([np.nan, 0], np.float32), "non_finite_coordinates"),
    ],
)
def test_conversion_rejects_invalid_points(points, reason):
    with pytest.raises(FlattenerError) as caught:
        pixel_to_normalized(points, (3, 5))
    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.reason_code == reason


def test_large_finite_conversion_avoids_intermediate_float32_overflow():
    point = np.array([np.finfo(np.float32).max / 2, 0], np.float32)
    result = pixel_to_normalized(point, (2, 2))
    assert np.isfinite(result).all()
    assert result[0] == point[0]
    with pytest.raises(FlattenerError) as caught:
        normalized_to_pixel(np.array([np.finfo(np.float32).max, 0], np.float32), (2, 4))
    assert caught.value.reason_code == "coordinate_overflow"


@pytest.mark.parametrize(
    ("coarse", "size", "reason"),
    [
        ([[[0, 0]]], (2, 2), "invalid_type"),
        (np.zeros((2, 2, 2), np.float64), (4, 4), "invalid_dtype"),
        (np.zeros((1, 2, 2), np.float32), (4, 4), "invalid_shape"),
        (np.zeros((3, 2, 2), np.float32), (2, 4), "invalid_size"),
    ],
)
def test_coarse_map_input_contract(coarse, size, reason):
    with pytest.raises(FlattenerError) as caught:
        resize_backward_map(coarse, size)
    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.reason_code == reason


@pytest.mark.parametrize("value,size", [(np.nan, (2, 2)), (-np.inf, (4, 4))])
def test_nonfinite_coarse_map_is_always_rejected(value, size):
    coarse = np.zeros((2, 2, 2), np.float32)
    coarse[0, 0, 0] = value
    before = coarse.copy()
    with pytest.raises(FlattenerError) as caught:
        resize_backward_map(coarse, size)
    assert caught.value.status == Status.INVALID_TRANSFORM
    assert caught.value.reason_code == "non_finite_grid"
    np.testing.assert_array_equal(coarse, before)
