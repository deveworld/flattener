"""Render contract of the pinned renderer, validation and coordinate code.

The sources are pinned byte for byte, so these checks guard against drift in torch,
numpy and Pillow. Expectations come from scalar formulas written out here.
"""

import json
import math

import numpy as np
import pytest

from flattener import (
    FlattenerError,
    RenderConfig,
    Status,
    identity_map,
    normalized_to_pixel,
    pixel_to_normalized,
    render,
    resize_backward_map,
)


def _grid(height, width):
    """Normalized pixel centers, derived without the library's coordinate helpers."""
    rows, columns = np.indices((height, width), dtype=np.float64)
    return np.stack(
        (2 * (columns + 0.5) / width - 1, 2 * (rows + 0.5) / height - 1), axis=-1
    ).astype(np.float32)


def _pattern(height, width):
    rows, columns = np.indices((height, width))
    return np.stack(
        [
            (37 * rows + 13 * columns) % 256,
            (19 * rows + 41 * columns + 17) % 256,
            (11 * rows + 7 * columns + 139) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def _scalar_expected(image, coordinates):
    """Scalar bilinear sampling with border clamping; no torch or production helpers."""
    height, width = image.shape[:2]
    output = np.empty((*coordinates.shape[:2], 3), dtype=np.uint8)
    for row, column in np.ndindex(coordinates.shape[:2]):
        gx, gy = map(float, coordinates[row, column])
        x = min(max((gx + 1) * width / 2 - 0.5, 0), width - 1)
        y = min(max((gy + 1) * height / 2 - 0.5, 0), height - 1)
        x0, y0 = math.floor(x), math.floor(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        ax, ay = x - x0, y - y0
        for channel in range(3):
            value = (
                (1 - ax) * (1 - ay) * int(image[y0, x0, channel])
                + ax * (1 - ay) * int(image[y0, x1, channel])
                + (1 - ax) * ay * int(image[y1, x0, channel])
                + ax * ay * int(image[y1, x1, channel])
            )
            output[row, column, channel] = math.floor(value + 0.5)
    return output


def _checked(coordinates, **config):
    """Render with the return policy; give the reason and the validation record."""
    result = render(_pattern(3, 5), coordinates, config=RenderConfig(on_error="return", **config))
    return result.reason_code, result.metadata["validation"]


def test_identity_keeps_every_pixel_and_records_a_passing_check():
    image = _pattern(3, 5)
    result = render(image, _grid(3, 5))

    assert result.status == Status.SUCCESS and result.reason_code is None
    assert result.used_fallback is False
    assert result.image.dtype == np.uint8
    np.testing.assert_array_equal(result.image, image)
    for mask in (result.source_mask, result.target_mask, result.valid_mask):
        assert mask.dtype == np.bool_ and mask.shape == (3, 5) and mask.all()
    metadata = result.metadata
    assert metadata["source_size"] == metadata["actual_output_size"] == [3, 5]
    validation = metadata["validation"]
    assert validation["target_pixels"] == validation["valid_pixels"] == 15
    assert validation["out_of_bounds_fraction"] == 0
    assert validation["fold_check"] == "passed" and validation["checked_cells"] == 8
    assert validation["min_jacobian_ratio"] == pytest.approx(1, abs=1e-6)
    for stage in ("prepare_input", "process_coordinates", "render", "total"):
        assert math.isfinite(metadata["timings"][stage])
    json.dumps(metadata, allow_nan=False)


def test_integer_shift_moves_content_the_other_way_and_fills_the_rest():
    image = _pattern(4, 4)
    fill = (9, 23, 81)
    # One pixel is 0.5 in normalized units here. Sampling one pixel to the right moves
    # the content left; sampling one row up moves it down.
    for shift, inside, source in (
        ((0.5, 0), np.s_[:, :3], np.s_[:, 1:]),
        ((0, -0.5), np.s_[1:], np.s_[:3]),
    ):
        expected = np.full_like(image, fill)
        expected[inside] = image[source]
        valid = np.zeros((4, 4), bool)
        valid[inside] = True
        coordinates = _grid(4, 4) + np.array(shift, np.float32)
        result = render(image, coordinates, config=RenderConfig(fill_rgb=fill))
        assert result.status == Status.SUCCESS
        np.testing.assert_array_equal(result.image, expected)
        np.testing.assert_array_equal(result.valid_mask, valid)
        assert result.metadata["validation"]["out_of_bounds_fraction"] == 0.25


def test_bilinear_sampling_matches_a_scalar_oracle():
    # Half-pixel shifts give exact means of the neighbours, clamped at the border.
    rows, columns = np.indices((4, 4))
    base = 16 * rows + 4 * columns
    image = np.stack([base, base + 64, base + 128], axis=-1).astype(np.uint8)
    coordinates = _grid(4, 4) + np.array([0.25, 0.25], np.float32)
    result = render(image, coordinates)
    assert result.valid_mask.all()
    np.testing.assert_array_equal(result.image, _scalar_expected(image, coordinates))
    # Enlarging and shrinking stay within one level of the float64 oracle.
    for source_size, output_size in (((3, 5), (7, 8)), ((7, 8), (3, 5))):
        image = _pattern(*source_size)
        coordinates = _grid(*output_size)
        result = render(image, coordinates)
        assert result.status == Status.SUCCESS and result.image.shape == (*output_size, 3)
        expected = _scalar_expected(image, coordinates).astype(np.int16)
        assert np.abs(result.image.astype(np.int16) - expected).max() <= 1


def test_quantization_rounds_exact_half_values_up():
    image = np.array(
        [[[0, 2, 250], [1, 3, 249], [4, 6, 244], [9, 11, 235]]],
        dtype=np.uint8,
    )
    coordinates = np.array([[[-0.5, 0], [0, 0], [0.5, 0], [1, 0]]], dtype=np.float32)
    expected = np.array(
        [[[1, 3, 250], [3, 5, 247], [7, 9, 240], [9, 11, 235]]],
        dtype=np.uint8,
    )

    result = render(image, coordinates)

    np.testing.assert_array_equal(result.image, expected)


def test_physical_boundaries_are_inclusive_and_outside_coordinates_are_filled():
    image = np.array([[[10, 20, 30], [200, 210, 220]]], dtype=np.uint8)
    coordinates = np.array(
        [[[-1.0001, 0], [-1, 0], [1, 0], [1.0001, 0], [0, -1], [0, 1], [0, 1.0001]]],
        dtype=np.float32,
    )
    fill = (1, 2, 3)

    result = render(
        image,
        coordinates,
        config=RenderConfig(max_out_of_bounds_fraction=1, fill_rgb=fill),
    )

    expected_mask = [[False, True, True, False, True, True, False]]
    np.testing.assert_array_equal(result.source_mask, expected_mask)
    np.testing.assert_array_equal(result.valid_mask, expected_mask)
    np.testing.assert_array_equal(
        result.image,
        [[fill, [10, 20, 30], [200, 210, 220], fill, [105, 115, 125], [105, 115, 125], fill]],
    )


def test_source_target_and_valid_masks_are_separate_and_denominator_is_target_count():
    image = _pattern(2, 2)
    coordinates = np.array([[[-0.5, -0.5], [2, 0], [0.5, 0.5], [2, 2]]], dtype=np.float32)
    target = np.array([[True, True, False, False]])
    fill = (4, 5, 6)
    result = render(
        image,
        coordinates,
        target_mask=target,
        config=RenderConfig(max_out_of_bounds_fraction=0.5, fill_rgb=fill),
    )

    np.testing.assert_array_equal(result.source_mask, [[True, False, True, False]])
    np.testing.assert_array_equal(result.target_mask, target)
    np.testing.assert_array_equal(result.valid_mask, [[True, False, False, False]])
    np.testing.assert_array_equal(result.image, [[image[0, 0], fill, fill, fill]])
    validation = result.metadata["validation"]
    assert validation["target_pixels"] == 2
    assert validation["valid_pixels"] == 1
    assert validation["out_of_bounds_pixels"] == 1
    assert validation["out_of_bounds_fraction"] == 0.5
    assert validation["fold_check"] == "not_applicable"

    strict = RenderConfig(max_out_of_bounds_fraction=0.49, on_error="return")
    failure = render(image, coordinates, target_mask=target, config=strict)
    assert failure.status == Status.INVALID_TRANSFORM
    assert failure.reason_code == "excessive_out_of_bounds"
    # A permitted share of out-of-bounds pixels still needs one valid pixel.
    outside = np.full((1, 2, 2), 2, np.float32)
    allowed = RenderConfig(max_out_of_bounds_fraction=1, on_error="return")
    assert render(image, outside, config=allowed).reason_code == "empty_valid_region"


def test_failure_policies_raise_return_nothing_or_give_back_the_original():
    image = _pattern(2, 3)
    coordinates = _grid(4, 5)
    coordinates[1, 2, 0] = np.nan
    before_image, before_coordinates = image.copy(), coordinates.copy()

    with pytest.raises(FlattenerError) as caught:
        render(image, coordinates, config=RenderConfig(on_error="raise"))
    assert caught.value.status == Status.INVALID_TRANSFORM
    assert caught.value.reason_code == "non_finite_grid"
    assert caught.value.metadata["requested_output_size"] == [4, 5]

    returned = render(image, coordinates, config=RenderConfig(on_error="return"))
    original = render(image, coordinates, config=RenderConfig(on_error="original"))
    for result in (returned, original):
        assert result.status == Status.INVALID_TRANSFORM
        assert result.reason_code == "non_finite_grid"
        assert result.source_mask is result.target_mask is result.valid_mask is None
        assert result.metadata["requested_output_size"] == [4, 5]
        assert result.metadata["timings"]["render"] is None
    assert returned.image is None and returned.used_fallback is False
    assert returned.metadata["actual_output_size"] is None
    np.testing.assert_array_equal(original.image, image)
    assert original.used_fallback is True
    assert original.metadata["actual_output_size"] == [2, 3]
    assert not np.shares_memory(original.image, image)
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(coordinates, before_coordinates)

    # Without a valid policy there is no way to choose, so it always raises.
    with pytest.raises(FlattenerError) as caught:
        render(image, coordinates, config=RenderConfig(on_error="ignore"))
    assert caught.value.status == Status.INPUT_ERROR


def test_fold_check_tests_every_corner_of_every_cell():
    # An affine map passes; its smallest Jacobian ratio is the determinant.
    grid = _grid(4, 5)
    u, v = grid[..., 0].copy(), grid[..., 1].copy()
    grid[..., 0] = 0.5 * u + 0.125 * v
    grid[..., 1] = 0.25 * v
    reason, stats = _checked(grid)
    assert reason is None and stats["fold_check"] == "passed"
    assert stats["checked_cells"] == 12 and stats["failed_cells"] == 0
    assert stats["min_jacobian_ratio"] == pytest.approx(0.5 * 0.25, abs=1e-7)

    # J(s,t) = 1 - 0.75s - 0.75t: three corners are positive, one is -0.5.
    corner = np.array([[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [-0.25, -0.25]]], np.float32)
    reason, stats = _checked(corner)
    assert reason == "fold_or_degenerate" and stats["fold_check"] == "failed"
    assert stats["checked_cells"] == stats["failed_cells"] == 1
    assert stats["min_jacobian_ratio"] == pytest.approx(-0.5)

    collapsed = _grid(2, 2)
    collapsed[..., 0] = 0
    reason, stats = _checked(collapsed)
    assert reason == "fold_or_degenerate" and stats["min_jacobian_ratio"] == 0
    # The threshold is exclusive: an identity cell at ratio 1 fails a limit of 1.
    assert _checked(_grid(2, 2), min_jacobian_ratio=1)[0] == "fold_or_degenerate"


def test_fold_failure_remains_failure_in_original_fallback():
    image = _pattern(3, 5)
    reflected = _grid(2, 4)
    reflected[..., 0] *= -1
    result = render(image, reflected, config=RenderConfig(on_error="original"))

    assert result.status == Status.INVALID_TRANSFORM
    assert result.reason_code == "fold_or_degenerate"
    assert result.used_fallback is True
    assert result.metadata["validation"]["fold_check"] == "failed"
    assert result.metadata["validation"]["failed_cells"] == 3
    assert result.metadata["requested_output_size"] == [2, 4]
    assert result.metadata["actual_output_size"] == [3, 5]
    np.testing.assert_array_equal(result.image, image)


def test_pixel_and_normalized_coordinates_meet_at_pixel_centers_in_xy_order():
    # A source 3 high and 5 wide; (-0.5, 2.5) is its bottom-left physical corner.
    pixels = np.array([[[0, 0], [4, 2]], [[2, 1], [-0.5, 2.5]]], np.float32)
    normalized = np.array([[[-0.8, -2 / 3], [0.8, 2 / 3]], [[0, 0], [-1, 1]]], np.float32)
    np.testing.assert_allclose(pixel_to_normalized(pixels, (3, 5)), normalized, rtol=0, atol=1e-6)
    np.testing.assert_allclose(normalized_to_pixel(normalized, (3, 5)), pixels, rtol=0, atol=1e-6)
    identity = identity_map((2, 3))
    assert identity.dtype == np.float32
    np.testing.assert_allclose(
        identity,
        [[[-2 / 3, -0.5], [0, -0.5], [2 / 3, -0.5]], [[-2 / 3, 0.5], [0, 0.5], [2 / 3, 0.5]]],
        rtol=0,
        atol=1e-6,
    )


def test_affine_expansion_has_analytic_interior_and_extended_residual_boundary():
    # F(x, y) = (1.2*x + 0.1*y + 0.125, -0.2*x + 0.8*y - 0.25).
    coarse = np.array(
        [[[-0.525, -0.55], [0.675, -0.75]], [[-0.425, 0.25], [0.775, 0.05]]],
        np.float32,
    )
    before = coarse.copy()
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
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result[1:3, 1:5], full_affine[1:3, 1:5], rtol=0, atol=1e-6)
    assert np.max(np.abs(result[0] - full_affine[0])) > 0.05
    np.testing.assert_array_equal(coarse, before)
