"""Analytic correspondence, visibility, and invertibility of synthetic warps."""

import json
from dataclasses import FrozenInstanceError, replace
from math import cos, exp, pi, sin

import numpy as np
import pytest

from flattener.contracts import FlattenerError, RenderConfig, Status
from flattener.synthetic.geometry import (
    FAMILIES,
    WarpParameters,
    flat_to_photo,
    make_geometry,
    photo_to_flat,
    sample_warp,
)
from flattener.validation import validate_geometry


def test_identity_and_singleton_maps_have_explicit_pixel_center_values():
    geometry = make_geometry((1, 4), (3, 1), WarpParameters())
    np.testing.assert_array_equal(
        geometry.backward_map,
        np.array([[[-0.75, 0], [-0.25, 0], [0.25, 0], [0.75, 0]]], np.float32),
    )
    np.testing.assert_array_equal(
        geometry.synthesis_map,
        np.array([[[0, -2 / 3]], [[0, 0]], [[0, 2 / 3]]], np.float32),
    )
    assert geometry.valid_mask.all()
    assert geometry.photo_page_mask.all()
    assert geometry.backward_map.dtype == np.float32
    assert geometry.synthesis_map.dtype == np.float32
    assert geometry.valid_mask.dtype == np.bool_
    assert geometry.photo_page_mask.dtype == np.bool_


def test_non_square_resize_does_not_change_normalized_direction():
    geometry = make_geometry((2, 3), (3, 5), WarpParameters(scale_x=0.5, scale_y=0.25))
    expected = np.array(
        [
            [[-1 / 3, -0.125], [0, -0.125], [1 / 3, -0.125]],
            [[-1 / 3, 0.125], [0, 0.125], [1 / 3, 0.125]],
        ],
        np.float32,
    )
    np.testing.assert_allclose(geometry.backward_map, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(
        geometry.synthesis_map[1, 1:4], np.array([[-0.8, 0], [0, 0], [0.8, 0]], np.float32)
    )
    assert np.isnan(geometry.synthesis_map[[0, 2]]).all()


def test_affine_transform_matches_independent_exact_points():
    params = WarpParameters(scale_x=0.5, scale_y=0.25, translate_x=0.125, translate_y=-0.25)
    points = np.array([[-1, -1], [0, 0], [1, 1]], np.float32)
    expected = np.array([[-0.375, -0.5], [0.125, -0.25], [0.625, 0]], np.float32)
    np.testing.assert_array_equal(flat_to_photo(points, params), expected)
    np.testing.assert_array_equal(photo_to_flat(expected, params), points)


def test_rotation_order_scales_axes_before_rotating():
    params = WarpParameters(scale_x=0.5, scale_y=0.25, rotation=pi / 2)
    points = np.array([[1, 0], [0, 1], [-1, -1]], np.float64)
    expected = np.array([[0, 0.5], [-0.25, 0], [0.25, -0.5]], np.float32)
    np.testing.assert_allclose(flat_to_photo(points, params), expected, rtol=0, atol=1e-7)


def test_perspective_matches_independently_solved_rational_map():
    # x = 0.5u / (1 + 0.25u); y = 0.75v / (1 + 0.25u).
    params = WarpParameters(scale_x=0.5, scale_y=0.75, perspective_x=0.25)
    points = np.array([[-1, -1], [0, 1], [1, 0.5]], np.float64)
    expected = np.array([[-2 / 3, -1], [0, 0.75], [0.4, 0.3]], np.float32)
    np.testing.assert_allclose(flat_to_photo(points, params), expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(photo_to_flat(expected, params), points, rtol=0, atol=1e-7)


@pytest.mark.parametrize("family", ["gentle", "strong"])
def test_sine_shears_have_known_axis_and_boundary_displacements(family):
    params = WarpParameters(family=family, shear_x=0.25, bend_y=0.5)
    # Horizontal shear vanishes at top/bottom, peaks on v=0. The second
    # shear is evaluated at the already displaced x, not the original u.
    points = np.array([[0, -1], [0, 1], [-0.25, 0], [0, 0]], np.float64)
    expected = np.array([[0, -0.5], [0, 1.5], [0, 0.5], [0.25, 0.5 * sin(5 * pi / 8)]], np.float32)
    np.testing.assert_allclose(flat_to_photo(points, params), expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(photo_to_flat(expected, params), points, rtol=0, atol=1e-7)


def test_gaussian_center_and_one_width_have_known_displacements():
    params = WarpParameters(family="local", bend_y=0.4, local_center=0.2, local_width=0.3)
    points = np.array([[0.2, 0], [0.5, -0.2], [-0.1, 0.2]], np.float64)
    expected = np.array(
        [[0.2, 0.4], [0.5, -0.2 + 0.4 * exp(-0.5)], [-0.1, 0.2 + 0.4 * exp(-0.5)]],
        np.float32,
    )
    np.testing.assert_allclose(flat_to_photo(points, params), expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(photo_to_flat(expected, params), points, rtol=0, atol=1e-7)


@pytest.mark.parametrize("family", FAMILIES)
def test_bidirectional_roundtrip_including_dense_boundary_and_outside_points(family):
    params = sample_warp(family, 3405)
    rng = np.random.default_rng(12)
    edge = np.linspace(-1, 1, 103)
    points = np.concatenate(
        [
            rng.uniform(-1.5, 1.5, (300, 2)),
            np.column_stack((edge, np.ones_like(edge))),
            np.column_stack((edge, -np.ones_like(edge))),
            np.column_stack((np.ones_like(edge), edge)),
            np.column_stack((-np.ones_like(edge), edge)),
        ]
    )
    restored = photo_to_flat(flat_to_photo(points, params), params)
    np.testing.assert_allclose(restored, points, rtol=0, atol=4e-7)
    photo = rng.uniform(-1, 1, (7, 9, 2))
    np.testing.assert_allclose(
        flat_to_photo(photo_to_flat(photo, params), params), photo, rtol=0, atol=3e-7
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_sampled_ranges_prove_full_continuous_page_visibility(family):
    # This is an upper bound over ALL page points, including extrema on
    # curved edges; testing only page corners would miss those extrema.
    for seed in range(40):
        params = sample_warp(family, seed)
        cx, cy = 1 + abs(params.shear_x), 1 + abs(params.bend_y)
        denominator = 1 - abs(params.perspective_x) * cx - abs(params.perspective_y) * cy
        c, s = abs(cos(params.rotation)), abs(sin(params.rotation))
        x_maximum = (
            params.scale_x * c * cx + params.scale_y * s * cy + abs(params.translate_x)
        ) / denominator
        y_maximum = (
            params.scale_x * s * cx + params.scale_y * c * cy + abs(params.translate_y)
        ) / denominator
        assert denominator > 0
        assert x_maximum < 1
        assert y_maximum < 1


@pytest.mark.parametrize("family", FAMILIES)
def test_dense_sampled_maps_pass_renderer_orientation_checks(family):
    for seed in (0, 8, 19):
        geometry = make_geometry((43, 61), (49, 67), sample_warp(family, seed))
        assert geometry.valid_mask.all()
        assert geometry.photo_page_mask.any()
        assert not geometry.photo_page_mask.all()
        for coordinate_map, mask in (
            (geometry.backward_map, geometry.valid_mask),
            (geometry.synthesis_map, geometry.photo_page_mask),
        ):
            _, _, stats = validate_geometry(coordinate_map, mask, RenderConfig())
            assert stats["fold_check"] == "passed"
            assert np.isfinite(coordinate_map[mask]).all()
            assert np.isnan(coordinate_map[~mask]).all()
        assert geometry.metadata["inverse_error_pixels_max"] < 1e-4


@pytest.mark.parametrize("family", ["gentle", "strong", "local"])
def test_numeric_jacobian_matches_closed_form_determinant(family):
    params = sample_warp(family, 23)
    point = np.array([0.17, -0.24], np.float64)
    epsilon = 1e-3  # Large enough that float32 return quantization is negligible.
    dx = np.array([epsilon, 0])
    dy = np.array([0, epsilon])
    derivative_x = (
        flat_to_photo(point + dx, params).astype(np.float64)
        - flat_to_photo(point - dx, params).astype(np.float64)
    ) / (2 * epsilon)
    derivative_y = (
        flat_to_photo(point + dy, params).astype(np.float64)
        - flat_to_photo(point - dy, params).astype(np.float64)
    ) / (2 * epsilon)
    numeric = np.linalg.det(np.column_stack((derivative_x, derivative_y)))
    x1 = point[0] + params.shear_x * sin(pi * (point[1] + 1) / 2)
    profile = (
        exp(-0.5 * ((x1 - params.local_center) / params.local_width) ** 2)
        if family == "local"
        else sin(pi * (x1 + 1) / 2)
    )
    y1 = point[1] + params.bend_y * profile
    c, s = cos(params.rotation), sin(params.rotation)
    # det(H) = sx*sy*(1 - p dot inverse(R diag(sx,sy)) t).
    inverse_translation_x = (c * params.translate_x + s * params.translate_y) / params.scale_x
    inverse_translation_y = (-s * params.translate_x + c * params.translate_y) / params.scale_y
    determinant = (
        params.scale_x
        * params.scale_y
        * (
            1
            - params.perspective_x * inverse_translation_x
            - params.perspective_y * inverse_translation_y
        )
    )
    expected = determinant / (1 + params.perspective_x * x1 + params.perspective_y * y1) ** 3
    assert expected > 0
    assert numeric == pytest.approx(expected, rel=5e-4)


def test_crop_masks_missing_labels_in_both_directions_without_clipping():
    params = WarpParameters(translate_x=1)
    geometry = make_geometry((1, 4), (1, 4), params)
    np.testing.assert_array_equal(geometry.valid_mask, [[True, True, False, False]])
    np.testing.assert_array_equal(geometry.photo_page_mask, [[False, False, True, True]])
    np.testing.assert_array_equal(geometry.backward_map[0, :2], [[0.25, 0], [0.75, 0]])
    np.testing.assert_array_equal(geometry.synthesis_map[0, 2:], [[-0.75, 0], [-0.25, 0]])
    assert np.isnan(geometry.backward_map[0, 2:]).all()
    assert np.isnan(geometry.synthesis_map[0, :2]).all()
    assert flat_to_photo(np.array([1, 0]), params)[0] == 2  # Point API does not clip.
    assert geometry.metadata["flat_visible_fraction"] == 0.5


def test_visibility_uses_physical_boundary_and_precedes_float32_rounding():
    exact = make_geometry((1, 2), (1, 2), WarpParameters(translate_x=0.5))
    assert exact.valid_mask.all()
    assert exact.backward_map[0, 1, 0] == 1
    outside = make_geometry((1, 2), (1, 2), WarpParameters(translate_x=0.5000000001))
    assert not outside.valid_mask[0, 1]
    assert np.isnan(outside.backward_map[0, 1]).all()


def test_no_correspondence_is_explicit_and_diagnostics_are_not_fake_zero():
    geometry = make_geometry((3, 4), (4, 5), WarpParameters(translate_x=3))
    assert not geometry.valid_mask.any()
    assert not geometry.photo_page_mask.any()
    assert np.isnan(geometry.backward_map).all()
    assert np.isnan(geometry.synthesis_map).all()
    assert geometry.metadata["inverse_error_pixels_max"] is None
    assert geometry.metadata["inverse_error_flat_pixels_max"] is None
    assert geometry.metadata["inverse_error_photo_pixels_max"] is None


def test_projective_poles_are_nan_pairs_and_inactive_in_synthesis():
    params = WarpParameters(scale_x=0.375, perspective_x=0.5)
    np.testing.assert_array_equal(photo_to_flat(np.array([0.75, 0]), params), [np.nan, np.nan])
    np.testing.assert_array_equal(flat_to_photo(np.array([-2, 0]), params), [np.nan, np.nan])
    geometry = make_geometry((3, 5), (3, 4), params)
    assert not geometry.photo_page_mask[:, -1].any()
    assert np.isnan(geometry.synthesis_map[:, -1]).all()


@pytest.mark.parametrize("transform", [flat_to_photo, photo_to_flat])
def test_point_functions_propagate_missing_pairs_without_modifying_input(transform):
    points = np.array([[0.0, 0], [np.nan, 1], [0.0, np.inf], [-np.inf, 0]], np.float64)
    before = points.copy()
    result = transform(points, WarpParameters())
    np.testing.assert_array_equal(result[0], [0, 0])
    assert np.isnan(result[1:]).all()
    np.testing.assert_array_equal(points, before)
    assert not np.shares_memory(points, result)
    assert result.dtype == np.float32
    assert transform(np.empty((0, 2)), WarpParameters()).shape == (0, 2)
    assert transform(np.array([0, 0]), WarpParameters()).shape == (2,)


@pytest.mark.parametrize("family", FAMILIES)
def test_reproducible_fresh_parameters_and_complete_json_metadata(family):
    params = sample_warp(family, np.int64(981))
    assert params == sample_warp(family, 981)
    assert params != sample_warp(family, 982)
    with pytest.raises(FrozenInstanceError):
        params.scale_x = 1
    first = make_geometry((9, 13), (11, 15), params)
    second = make_geometry((9, 13), (11, 15), params)
    np.testing.assert_array_equal(first.backward_map, second.backward_map)
    np.testing.assert_array_equal(first.synthesis_map, second.synthesis_map)
    assert not np.shares_memory(first.backward_map, second.backward_map)
    assert first.metadata == second.metadata
    assert json.loads(json.dumps(first.metadata, allow_nan=False)) == first.metadata
    assert first.metadata["backward_map_direction"] == "flat_output_to_photo_source"
    assert first.metadata["synthesis_map_direction"] == "photo_output_to_flat_source"
    assert first.metadata["occlusion_model"] == "none_bijective_2d_deformation"
    assert first.metadata["folds"] is False


@pytest.mark.parametrize("bad_seed", [True, np.bool_(False), -1, 2**63, 1.5, "1", None])
def test_invalid_seed_rejected(bad_seed):
    with pytest.raises(FlattenerError, match="seed"):
        sample_warp("gentle", bad_seed)


@pytest.mark.parametrize("bad_family", ["", "folded", None, 5, ["local"]])
def test_invalid_family_rejected(bad_family):
    with pytest.raises(FlattenerError, match="family"):
        sample_warp(bad_family, 0)
    with pytest.raises(FlattenerError, match="family"):
        WarpParameters(family=bad_family)


@pytest.mark.parametrize(
    "changes",
    [
        {"scale_x": 0},
        {"scale_y": -1},
        {"scale_x": True},
        {"rotation": np.nan},
        {"translate_y": np.inf},
        {"shear_x": "1"},
        {"local_width": 0},
        {"family": "perspective", "bend_y": 0.1},
        {"perspective_x": 1.0},
        {"perspective_x": 0.5, "translate_x": 3},  # Positive page q but negative det H.
        {"perspective_x": 0.5, "translate_x": 2},  # Singular H.
        {"scale_x": 1e-15},  # Near-singular H.
        {"scale_x": 10**1000},
    ],
)
def test_invalid_warps_reject_poles_folds_degeneracy_and_non_numeric_parameters(changes):
    with pytest.raises(FlattenerError) as caught:
        WarpParameters(**changes)
    assert caught.value.status == Status.INPUT_ERROR


def test_parameter_metadata_canonicalizes_numpy_scalars():
    params = replace(WarpParameters(), scale_x=np.float32(0.5), translate_y=np.int64(0))
    assert type(params.scale_x) is float
    assert type(params.translate_y) is float
    json.dumps(make_geometry((2, 3), (2, 3), params).metadata, allow_nan=False)


@pytest.mark.parametrize(
    "bad_size", [(0, 2), (2, -1), (True, 2), (2.5, 3), (1,), (8193, 1), (5000, 5000)]
)
@pytest.mark.parametrize("argument", ["flat_size", "photo_size"])
def test_size_limits_are_checked_before_any_large_allocation(bad_size, argument):
    sizes = {"flat_size": (2, 3), "photo_size": (3, 4), argument: bad_size}
    with pytest.raises(FlattenerError):
        make_geometry(**sizes, params=WarpParameters())


@pytest.mark.parametrize("transform", [flat_to_photo, photo_to_flat])
@pytest.mark.parametrize(
    "bad_points",
    [[[0, 0]], np.array(1), np.zeros((2, 3)), np.zeros((1, 2), complex), np.zeros((1, 2), bool)],
)
def test_bad_point_storage_rejected(transform, bad_points):
    with pytest.raises(FlattenerError):
        transform(bad_points, WarpParameters())


def test_wrong_parameter_object_rejected():
    with pytest.raises(FlattenerError, match="WarpParameters"):
        make_geometry((2, 3), (2, 3), {})
    with pytest.raises(FlattenerError, match="WarpParameters"):
        flat_to_photo(np.zeros((2, 2)), None)
