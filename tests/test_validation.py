"""Analytical checks for input contracts and bilinear-cell geometry."""

from dataclasses import replace

import numpy as np
import pytest

from flattener.contracts import FlattenerError, RenderConfig, Status
from flattener.validation import (
    validate_config,
    validate_geometry,
    validate_image,
    validate_map,
    validate_size,
    validate_target_mask,
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


@pytest.mark.parametrize("config", [{}, "default", 1, False])
def test_config_requires_render_config(config):
    assert_error(validate_config, config, reason="invalid_config")


@pytest.mark.parametrize("name", ["max_input_pixels", "max_output_pixels", "max_side"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(True)])
def test_config_resource_limits_are_positive_integers(name, value):
    assert_error(validate_config, replace(RenderConfig(), **{name: value}), reason="invalid_config")


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf, "0.2"])
def test_config_out_of_bounds_fraction_is_finite_and_bounded(value):
    assert_error(
        validate_config,
        RenderConfig(max_out_of_bounds_fraction=value),
        reason="invalid_config",
    )


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, "0.001"])
def test_config_jacobian_threshold_is_positive_and_finite(value):
    assert_error(validate_config, RenderConfig(min_jacobian_ratio=value), reason="invalid_config")


@pytest.mark.parametrize(
    "value",
    [
        (0, 0),
        (0, 0, 0, 0),
        [0, 0, 0],
        (256, 0, 0),
        (-1, 0, 0),
        (0.0, 0, 0),
        (True, 0, 0),
        (np.bool_(False), 0, 0),
        None,
    ],
)
def test_config_fill_color_is_rgb_integer_tuple(value):
    assert_error(validate_config, RenderConfig(fill_rgb=value), reason="invalid_config")


@pytest.mark.parametrize("value", ["ignore", "", None, True])
def test_config_error_policy_is_explicit(value):
    assert_error(validate_config, RenderConfig(on_error=value), reason="invalid_config")


@pytest.mark.parametrize("fraction", [0, 1])
@pytest.mark.parametrize("policy", ["raise", "return", "original"])
def test_config_allows_fraction_endpoints_and_each_error_policy(fraction, policy):
    config = RenderConfig(max_out_of_bounds_fraction=fraction, on_error=policy)
    assert validate_config(config) == config


@pytest.mark.parametrize("size", [(2, 3), [2, 3], (np.int32(2), np.int64(3))])
def test_size_accepts_positive_integral_height_width(size):
    assert validate_size(size, max_pixels=6, max_side=3, name="output") == (2, 3)


@pytest.mark.parametrize(
    "size",
    [None, "23", (), (2,), (2, 3, 4), (0, 3), (-1, 3), (2.0, 3), (True, 3), (np.bool_(True), 3)],
)
def test_size_rejects_malformed_or_nonintegral_dimensions(size):
    assert_error(
        validate_size,
        size,
        max_pixels=100,
        max_side=10,
        name="output",
        reason="invalid_size",
    )


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


@pytest.mark.parametrize(
    "function,value,reason",
    [
        (validate_image, [[[0, 0, 0]]], "invalid_type"),
        (validate_image, np.zeros((2, 3, 3), dtype=np.float32), "invalid_dtype"),
        (validate_image, np.zeros((2, 3), dtype=np.uint8), "invalid_shape"),
        (validate_image, np.zeros((2, 3, 4), dtype=np.uint8), "invalid_shape"),
        (validate_image, np.zeros((0, 3, 3), dtype=np.uint8), "invalid_size"),
        (validate_map, [[[0, 0]]], "invalid_type"),
        (validate_map, np.zeros((2, 3, 2), dtype=np.float64), "invalid_dtype"),
        (validate_map, np.zeros((2, 3), dtype=np.float32), "invalid_shape"),
        (validate_map, np.zeros((2, 3, 3), dtype=np.float32), "invalid_shape"),
        (validate_map, np.zeros((2, 0, 2), dtype=np.float32), "invalid_size"),
    ],
)
def test_array_input_contracts(function, value, reason):
    assert_error(function, value, RenderConfig(), reason=reason)


def test_array_limits_are_separate_for_input_and_output():
    config = RenderConfig(max_input_pixels=5, max_output_pixels=7)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    grid = pixel_center_grid(2, 3)
    assert_error(validate_image, image, config, reason="size_limit_exceeded")
    validate_map(grid, config)
    assert_error(validate_map, pixel_center_grid(2, 4), config, reason="size_limit_exceeded")


def test_noncontiguous_arrays_are_accepted_without_mutation():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)[:, ::2]
    grid = pixel_center_grid(4, 6)[:, ::2]
    original_image, original_grid = image.copy(), grid.copy()
    validate_image(image, RenderConfig())
    validate_map(grid, RenderConfig())
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(grid, original_grid)


def test_target_mask_default_and_copy_ownership():
    default = validate_target_mask(None, (2, 3))
    assert default.shape == (2, 3)
    assert default.dtype == np.bool_
    assert default.all()
    supplied = np.array([[True, False, True], [False, True, False]])
    actual = validate_target_mask(supplied, (2, 3))
    np.testing.assert_array_equal(actual, supplied)
    assert not np.shares_memory(actual, supplied)


@pytest.mark.parametrize(
    "mask,reason",
    [
        ([[True]], "invalid_type"),
        (np.ones((2, 3), dtype=np.uint8), "invalid_dtype"),
        (np.ones((3, 2), dtype=bool), "invalid_shape"),
        (np.ones((2, 3, 1), dtype=bool), "invalid_shape"),
    ],
)
def test_target_mask_contract(mask, reason):
    assert_error(validate_target_mask, mask, (2, 3), reason=reason)


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


def test_sparse_target_has_no_checkable_cells():
    target = np.eye(3, dtype=bool)
    source, valid, stats = validate_geometry(pixel_center_grid(3, 3), target, RenderConfig())
    assert source.all()
    np.testing.assert_array_equal(valid, target)
    assert stats["target_pixels"] == stats["valid_pixels"] == 3
    assert stats["fold_check"] == "not_applicable"
    assert stats["checked_cells"] == 0
    assert stats["excluded_cells"] == 4
    assert stats["min_jacobian_ratio"] is None


def test_masked_nonfinite_values_do_not_enter_jacobians_or_mutate_inputs():
    grid = np.full((3, 3, 2), np.inf, dtype=np.float32)
    grid[0, 2] = np.nan
    grid[:2, :2] = pixel_center_grid(3, 3)[:2, :2]
    target = np.zeros((3, 3), bool)
    target[:2, :2] = True
    before = grid.copy()
    with np.errstate(all="raise"):
        source, valid, stats = validate_geometry(grid, target, RenderConfig())
    np.testing.assert_array_equal(source, target)
    np.testing.assert_array_equal(valid, target)
    np.testing.assert_array_equal(grid, before)
    assert stats["checked_cells"] == 1
    assert stats["excluded_cells"] == 3
    assert stats["fold_check"] == "passed"
    assert stats["min_jacobian_ratio"] == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_active_nonfinite_coordinates_fail_before_out_of_bounds(bad_value):
    grid = np.array([[[bad_value, 0], [2, 0]]], dtype=np.float32)
    assert_error(
        validate_geometry,
        grid,
        np.ones((1, 2), bool),
        RenderConfig(),
        reason="non_finite_grid",
        status=Status.INVALID_TRANSFORM,
    )


def test_empty_target_fails_even_when_undefined_coordinates_are_masked():
    assert_error(
        validate_geometry,
        np.full((2, 2, 2), np.nan, np.float32),
        np.zeros((2, 2), bool),
        RenderConfig(),
        reason="empty_target",
        status=Status.INVALID_TRANSFORM,
    )


def test_oob_cells_still_undergo_fold_check_before_empty_valid_check():
    grid = pixel_center_grid(2, 2)
    grid[..., 0] = 2 - grid[..., 0]
    assert_error(
        validate_geometry,
        grid,
        np.ones((2, 2), bool),
        RenderConfig(),
        reason="excessive_out_of_bounds",
        status=Status.INVALID_TRANSFORM,
    )
    error = assert_error(
        validate_geometry,
        grid,
        np.ones((2, 2), bool),
        RenderConfig(max_out_of_bounds_fraction=1),
        reason="fold_or_degenerate",
        status=Status.INVALID_TRANSFORM,
    )
    assert error.metadata["validation"]["checked_cells"] == 1


def test_empty_valid_region_fails_after_permitted_out_of_bounds():
    assert_error(
        validate_geometry,
        np.array([[[2, 0], [3, 0]]], np.float32),
        np.ones((1, 2), bool),
        RenderConfig(max_out_of_bounds_fraction=1),
        reason="empty_valid_region",
        status=Status.INVALID_TRANSFORM,
    )
