"""Analytical checks for input contracts and bilinear-cell geometry."""

import numpy as np
import pytest

from flattener.contracts import FlattenerError, RenderConfig, Status
from flattener.validation import (
    validate_config,
    validate_geometry,
    validate_size,
)


def assert_error(function, *args, reason, status=Status.INPUT_ERROR, **kwargs):
    with pytest.raises(FlattenerError) as captured:
        function(*args, **kwargs)
    assert captured.value.status == status
    assert captured.value.reason_code == reason
    return captured.value


def pixel_center_grid(height, width):
    """Construct the analytical normalized pixel centers without library helpers."""
    rows, columns = np.indices((height, width), dtype=np.float64)
    return np.stack(
        (2 * (columns + 0.5) / width - 1, 2 * (rows + 0.5) / height - 1),
        axis=-1,
    ).astype(np.float32)


def test_default_and_explicit_config():
    assert validate_config(None) == RenderConfig()
    config = RenderConfig(fill_rgb=(0, 127, 255), on_error="original")
    assert validate_config(config) == config


@pytest.mark.parametrize("size,max_pixels,max_side", [((2, 3), 5, 3), ((1, 4), 100, 3)])
def test_size_checks_pixel_count_and_each_side(size, max_pixels, max_side):
    assert_error(
        validate_size,
        size,
        max_pixels=max_pixels,
        max_side=max_side,
        name="output",
        reason="size_limit_exceeded",
    )


def test_affine_area_is_the_determinant_for_every_cell():
    grid = pixel_center_grid(4, 5)
    u, v = grid[..., 0].copy(), grid[..., 1].copy()
    grid[..., 0] = 0.5 * u + 0.125 * v
    grid[..., 1] = 0.25 * v
    source, valid, stats = validate_geometry(grid, np.ones((4, 5), bool), RenderConfig())
    assert source.all() and valid.all()
    assert stats["target_pixels"] == stats["valid_pixels"] == 20
    assert stats["out_of_bounds_pixels"] == 0
    assert stats["out_of_bounds_fraction"] == 0
    assert stats["fold_check"] == "passed"
    assert stats["checked_cells"] == 12
    assert stats["excluded_cells"] == stats["failed_cells"] == 0
    assert stats["min_jacobian_ratio"] == pytest.approx(0.5 * 0.25, abs=1e-7)


def test_fold_at_bottom_right_corner_is_detected():
    # J(s,t) = 1 - 0.75s - 0.75t: three corners are positive, one is -0.5.
    grid = np.array([[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [-0.25, -0.25]]], dtype=np.float32)
    error = assert_error(
        validate_geometry,
        grid,
        np.ones((2, 2), bool),
        RenderConfig(),
        reason="fold_or_degenerate",
        status=Status.INVALID_TRANSFORM,
    )
    stats = error.metadata["validation"]
    assert stats["fold_check"] == "failed"
    assert stats["checked_cells"] == stats["failed_cells"] == 1
    assert stats["min_jacobian_ratio"] == pytest.approx(-0.5)


@pytest.mark.parametrize("scale,expected_ratio", [(-1, -1), (0, 0)])
def test_reflection_and_collapsed_axis_fail(scale, expected_ratio):
    grid = pixel_center_grid(2, 2)
    grid[..., 0] *= scale
    error = assert_error(
        validate_geometry,
        grid,
        np.ones((2, 2), bool),
        RenderConfig(),
        reason="fold_or_degenerate",
        status=Status.INVALID_TRANSFORM,
    )
    assert error.metadata["validation"]["min_jacobian_ratio"] == expected_ratio


def test_jacobian_threshold_is_exclusive():
    assert_error(
        validate_geometry,
        pixel_center_grid(2, 2),
        np.ones((2, 2), bool),
        RenderConfig(min_jacobian_ratio=1),
        reason="fold_or_degenerate",
        status=Status.INVALID_TRANSFORM,
    )


def test_source_bounds_are_inclusive_and_oob_limit_is_inclusive():
    grid = np.array([[[-1, -1], [1, 1], [1.25, 0], [0, 0]]], dtype=np.float32)
    source, valid, stats = validate_geometry(grid, np.ones((1, 4), bool), RenderConfig())
    np.testing.assert_array_equal(source, [[True, True, False, True]])
    np.testing.assert_array_equal(valid, source)
    assert stats["out_of_bounds_fraction"] == 0.25
    assert stats["out_of_bounds_pixels"] == 1
    assert stats["valid_pixels"] == 3
    assert stats["fold_check"] == "not_applicable"
    assert stats["checked_cells"] == stats["excluded_cells"] == 0
    assert stats["min_jacobian_ratio"] is None


def test_oob_fraction_uses_only_target_pixels():
    grid = np.array([[[0, 0], [2, 0], [0, 0], [0, 0]]], dtype=np.float32)
    target = np.array([[True, True, False, False]])
    error = assert_error(
        validate_geometry,
        grid,
        target,
        RenderConfig(),
        reason="excessive_out_of_bounds",
        status=Status.INVALID_TRANSFORM,
    )
    stats = error.metadata["validation"]
    assert stats["target_pixels"] == 2
    assert stats["out_of_bounds_pixels"] == 1
    assert stats["out_of_bounds_fraction"] == 0.5


def test_active_nonfinite_coordinates_fail_before_out_of_bounds():
    grid = np.array([[[np.nan, 0], [2, 0]]], dtype=np.float32)
    assert_error(
        validate_geometry,
        grid,
        np.ones((1, 2), bool),
        RenderConfig(),
        reason="non_finite_grid",
        status=Status.INVALID_TRANSFORM,
    )


def test_empty_valid_region_fails_after_permitted_out_of_bounds():
    assert_error(
        validate_geometry,
        np.array([[[2, 0], [3, 0]]], np.float32),
        np.ones((1, 2), bool),
        RenderConfig(max_out_of_bounds_fraction=1),
        reason="empty_valid_region",
        status=Status.INVALID_TRANSFORM,
    )
