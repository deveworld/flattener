"""Coordinate contracts checked against explicit and analytic expectations."""

import numpy as np
import pytest

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
