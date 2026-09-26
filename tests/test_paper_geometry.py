"""Independent cylinder projection, invertibility, and appearance contracts."""

import json
import random
from dataclasses import FrozenInstanceError
from math import cos, sin

import numpy as np
import pytest

from flattener.contracts import FlattenerError, RenderConfig
from flattener.synthetic.paper import (
    PaperParameters,
    flat_to_photo,
    generate_paper_sample,
    make_capture_geometry,
    make_geometry,
    paper_parameters_for_attempt,
    photo_to_flat,
    sample_paper_geometry,
    sample_paper_parameters,
)
from flattener.synthetic.pipeline import PhotometricConfig
from flattener.validation import validate_geometry


def _reference_projection(points, params):
    """Closed form, independent of the implementation's LUT and homography."""
    u, v = np.asarray(points, dtype=np.float64).T
    along = u if params.axis == "x" else v
    k = params.curvature
    curved = np.sin(k * along) / k if k else along
    depth = (1 - np.cos(k * along)) / k if k else np.zeros_like(along)
    magnification = params.camera_distance / (params.camera_distance - depth)
    x = (curved if params.axis == "x" else u) * magnification
    y = (v if params.axis == "x" else curved) * magnification
    denominator = 1 + params.perspective_x * x + params.perspective_y * y
    c, s = cos(params.rotation), sin(params.rotation)
    return np.column_stack(
        (
            (params.scale_x * c * x - params.scale_y * s * y + params.translate_x) / denominator,
            (params.scale_x * s * x + params.scale_y * c * y + params.translate_y) / denominator,
        )
    ).astype(np.float32)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("curvature", [-1.2, -0.4, 0.0, 0.4, 1.2])
def test_forward_matches_independent_projected_cylinder(axis, curvature):
    params = PaperParameters(
        axis=axis,
        curvature=curvature,
        camera_distance=2.5,
        scale_x=0.61,
        scale_y=0.47,
        rotation=0.17,
        translate_x=0.08,
        translate_y=-0.12,
        perspective_x=0.04,
        perspective_y=-0.03,
    )
    points = np.array([[-1, -1], [-0.7, 0.4], [0, 0], [0.3, -0.2], [1, 1]])
    actual = flat_to_photo(points, params)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, _reference_projection(points, params), rtol=0, atol=1e-7)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("curvature", [-1.2, 0.0, 1.2])
def test_roundtrip_random_points_and_all_four_page_edges(axis, curvature):
    params = PaperParameters(
        axis=axis,
        curvature=curvature,
        camera_distance=2.5,
        scale_x=0.54,
        scale_y=0.71,
        rotation=-0.18,
        translate_x=0.03,
        translate_y=-0.05,
        perspective_x=-0.03,
        perspective_y=0.02,
    )
    edge = np.linspace(-1, 1, 257)
    points = np.concatenate(
        [np.random.default_rng(83).uniform(-1, 1, (2000, 2))]
        + [np.column_stack((edge, np.full_like(edge, sign))) for sign in (-1, 1)]
        + [np.column_stack((np.full_like(edge, sign), edge)) for sign in (-1, 1)]
    )
    restored = photo_to_flat(flat_to_photo(points, params), params)
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, points, rtol=0, atol=1e-6)


def test_zero_curvature_identity_uses_pixel_centers_and_xy_order():
    geometry = make_geometry((2, 4), (128, 128), PaperParameters(scale_x=1, scale_y=1))
    expected = np.array(
        [
            [[-0.75, -0.5], [-0.25, -0.5], [0.25, -0.5], [0.75, -0.5]],
            [[-0.75, 0.5], [-0.25, 0.5], [0.25, 0.5], [0.75, 0.5]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(geometry.backward_map, expected)
    y, x = np.mgrid[:128, :128]
    synthesis = np.stack(((x + 0.5) / 64 - 1, (y + 0.5) / 64 - 1), axis=-1)
    np.testing.assert_array_equal(geometry.synthesis_map, synthesis.astype(np.float32))
    assert geometry.valid_mask.all() and geometry.photo_page_mask.all()


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("curvature", [-1.2, 1.2])
def test_geometry_is_positive_orientation_with_explicit_nan_masks(axis, curvature):
    params = PaperParameters(axis=axis, curvature=curvature, scale_x=0.55, scale_y=0.55)
    geometry = make_geometry((43, 61), (128, 144), params)
    assert geometry.valid_mask.all()
    assert geometry.photo_page_mask.any() and not geometry.photo_page_mask.all()
    for coordinate_map, mask in (
        (geometry.backward_map, geometry.valid_mask),
        (geometry.synthesis_map, geometry.photo_page_mask),
    ):
        assert coordinate_map.dtype == np.float32 and mask.dtype == np.bool_
        assert np.isfinite(coordinate_map[mask]).all()
        assert np.isnan(coordinate_map[~mask]).all()
        _, _, stats = validate_geometry(coordinate_map, mask, RenderConfig())
        assert stats["fold_check"] == "passed"
    json.dumps(geometry.metadata, allow_nan=False)


def test_crop_masks_are_not_clipped_or_clamped_to_page_edges():
    params = PaperParameters(scale_x=1, scale_y=1, translate_x=1)
    geometry = make_geometry((2, 4), (128, 128), params)
    np.testing.assert_array_equal(geometry.valid_mask, [[True, True, False, False]] * 2)
    assert not geometry.photo_page_mask[:, :64].any()
    assert geometry.photo_page_mask[:, 64:].all()
    assert np.isnan(geometry.backward_map[:, 2:]).all()
    assert np.isnan(geometry.synthesis_map[:, :64]).all()
    assert flat_to_photo(np.array([1, 0]), params)[0] == 2
    curved = PaperParameters(curvature=1.2)
    # Points beyond the finite inverse domain must not become a page edge.
    assert np.isnan(photo_to_flat(np.array([[100.0, 0], [-100.0, 0]]), curved)).all()


@pytest.mark.parametrize("axis", ["x", "y"])
def test_inverse_continuation_preserves_outside_points_instead_of_edge_clamping(axis):
    params = PaperParameters(axis=axis, curvature=-1.2, camera_distance=2.5)
    points = np.array([[1.005, 0], [-1.005, 0]])
    if axis == "y":
        points = points[:, ::-1]
    restored = photo_to_flat(flat_to_photo(points, params), params)
    np.testing.assert_allclose(restored, points, rtol=0, atol=1e-6)
    assert (np.max(np.abs(restored), axis=-1) > 1).all()


@pytest.mark.parametrize("transform", [flat_to_photo, photo_to_flat])
def test_missing_point_pairs_propagate_without_input_mutation(transform):
    points = np.array([[0.0, 0], [np.nan, 1], [1, np.inf]])
    before = points.copy()
    actual = transform(points, PaperParameters())
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(actual[0], [0, 0])
    assert np.isnan(actual[1:]).all() and not np.shares_memory(points, actual)
    assert transform(np.empty((0, 2)), PaperParameters()).shape == (0, 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"axis": "z"},
        {"axis": None},
        {"curvature": 1.2001},
        {"curvature": -1.2001},
        {"curvature": True},
        {"curvature": np.nan},
        {"camera_distance": 2.49},
        {"camera_distance": 8.01},
        {"scale_x": 0},
        {"scale_y": -1},
        {"rotation": np.inf},
        {"perspective_x": 2},
    ],
)
def test_invalid_physical_parameters_are_rejected(changes):
    with pytest.raises(FlattenerError):
        PaperParameters(**changes)


@pytest.mark.parametrize(
    "output_size,photo_size",
    [
        ((1, 2), (128, 128)),
        ((2, 513), (128, 128)),
        ((2, 2), (127, 128)),
        ((2, 2), (128, 513)),
        ((True, 4), (128, 128)),
        ((2, 2), [128, 128]),
    ],
)
def test_geometry_rejects_invalid_sizes(output_size, photo_size):
    with pytest.raises(FlattenerError):
        make_geometry(output_size, photo_size, PaperParameters())


@pytest.mark.parametrize("family", ["flat", "gentle", "strong"])
@pytest.mark.parametrize(
    "size_bin,bounds",
    [
        ("small", (0, 0.25)),
        ("medium", (0.25, 0.5)),
        ("large", (0.5, 1.0)),
    ],
)
def test_sampler_is_reproducible_full_page_and_in_requested_size_bin(family, size_bin, bounds):
    for seed in (0, 12):
        options = {"size_bin": size_bin, "photo_size": (128, 144), "output_size": (31, 41)}
        params, geometry, sampling = sample_paper_geometry(family, seed, **options)
        repeated, other, repeated_sampling = sample_paper_geometry(family, seed, **options)
        assert params == repeated and sampling == repeated_sampling
        assert params == sample_paper_parameters(
            family, seed, size_bin=size_bin, photo_size=(128, 144)
        )
        assert params == paper_parameters_for_attempt(
            family, seed, size_bin=size_bin, attempt=sampling["attempts"]
        )
        assert geometry.valid_mask.all()
        fraction = float(geometry.photo_page_mask.mean())
        assert bounds[0] <= fraction < bounds[1]
        assert sampling["actual_size_bin"] == sampling["requested_size_bin"] == size_bin
        assert 1 <= sampling["attempts"] <= 256
        np.testing.assert_array_equal(geometry.backward_map, other.backward_map)
        np.testing.assert_array_equal(geometry.synthesis_map, other.synthesis_map)
        edge = np.linspace(-1, 1, 1001)
        perimeter = np.concatenate(
            [np.column_stack((edge, np.full_like(edge, sign))) for sign in (-1, 1)]
            + [np.column_stack((np.full_like(edge, sign), edge)) for sign in (-1, 1)]
        )
        assert (np.abs(flat_to_photo(perimeter, params)) <= 1).all()
        assert params.curvature == 0 if family == "flat" else params.curvature != 0
        with pytest.raises(FrozenInstanceError):
            params.scale_x = 1


@pytest.mark.parametrize(
    "options",
    [
        {"family": "local"},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**63},
        {"size_bin": "huge"},
        {"attempt": 0},
        {"attempt": 257},
    ],
)
def test_sampler_rejects_invalid_arguments(options):
    arguments = {"family": "gentle", "seed": 1, **options}
    with pytest.raises(FlattenerError):
        paper_parameters_for_attempt(**arguments)


def test_disabled_appearance_preserves_constant_page_and_background():
    flat = np.full((32, 48, 3), [80, 160, 220], dtype=np.uint8)
    background = np.full((128, 128, 3), [16, 32, 48], dtype=np.uint8)
    before, before_background = flat.copy(), background.copy()
    sample = generate_paper_sample(
        flat,
        53,
        PaperParameters(scale_x=0.5, scale_y=0.5),
        photo_size=(128, 128),
        output_size=(24, 32),
        background=background,
        photometric=PhotometricConfig(enabled=False),
        white_balance_strength=0,
        jpeg_quality=None,
    )
    assert sample.photo.dtype == np.uint8 and sample.photo.shape == (128, 128, 3)
    np.testing.assert_array_equal(
        sample.photo[48:80, 48:80], np.broadcast_to(flat[0, 0], (32, 32, 3))
    )
    np.testing.assert_array_equal(sample.photo[:16], background[:16])
    np.testing.assert_array_equal(sample.image, sample.photo)
    np.testing.assert_array_equal(sample.mask, sample.geometry.photo_page_mask)
    np.testing.assert_array_equal(flat, before)
    np.testing.assert_array_equal(background, before_background)
    assert not np.shares_memory(sample.photo, flat)
    json.dumps(sample.metadata, allow_nan=False)


def test_appearance_changes_only_pixels_and_preserves_global_random_states():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    flat = np.random.default_rng(8).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    params = sample_paper_parameters("strong", 5, photo_size=(128, 128))
    options = {"photo_size": (128, 128), "output_size": (24, 32)}
    plain = generate_paper_sample(
        flat,
        7,
        params,
        **options,
        photometric=PhotometricConfig(enabled=False),
        white_balance_strength=0,
        jpeg_quality=None,
    )
    augmented = generate_paper_sample(flat, 7, params, **options)
    repeated = generate_paper_sample(flat, 7, params, **options)
    assert not np.array_equal(plain.photo, augmented.photo)
    np.testing.assert_array_equal(augmented.photo, repeated.photo)
    assert augmented.metadata == repeated.metadata
    for name in ("backward_map", "synthesis_map", "valid_mask", "photo_page_mask"):
        np.testing.assert_array_equal(
            getattr(plain.geometry, name), getattr(augmented.geometry, name)
        )
    assert plain.geometry.metadata == augmented.geometry.metadata
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_photo_mask_and_synthesis_do_not_depend_on_flat_output_resolution():
    params = PaperParameters(axis="y", curvature=1.1, scale_x=0.7, scale_y=0.6, rotation=0.12)
    flat = np.full((23, 41, 3), [70, 130, 190], dtype=np.uint8)
    first = generate_paper_sample(flat, 61, params, photo_size=(128, 144), output_size=(11, 17))
    second = generate_paper_sample(flat, 61, params, photo_size=(128, 144), output_size=(73, 29))
    assert first.geometry.backward_map.shape == (11, 17, 2)
    assert second.geometry.backward_map.shape == (73, 29, 2)
    assert first.mask.shape == second.mask.shape == (128, 144)
    assert first.mask.any() and not first.mask.all()
    np.testing.assert_array_equal(first.mask, second.mask)
    np.testing.assert_array_equal(first.geometry.synthesis_map, second.geometry.synthesis_map)
    np.testing.assert_array_equal(first.photo, second.photo)


def test_identity_synthesis_preserves_asymmetric_xy_rgb_ramp():
    # Non-square dimensions and distinct channel slopes expose x/y swaps,
    # mirrored axes, and RGB permutations that constant images cannot reveal.
    y, x = np.mgrid[:128, :192]
    flat = np.stack((x, 2 * y, (3 * x + y) // 4), axis=-1).astype(np.uint8)
    sample = generate_paper_sample(
        flat,
        7,
        PaperParameters(scale_x=1, scale_y=1),
        photo_size=(128, 192),
        output_size=(19, 31),
        photometric=PhotometricConfig(enabled=False),
        white_balance_strength=0,
        jpeg_quality=None,
    )
    assert sample.mask.all()
    assert sample.photo.shape == flat.shape
    # The 2x bilinear render rounds uint8 subpixels before averaging; allow one
    # intensity level, excluding the border where border-padding acts.
    np.testing.assert_allclose(
        sample.photo[1:-1, 1:-1].astype(np.int16),
        flat[1:-1, 1:-1].astype(np.int16),
        rtol=0,
        atol=1,
    )


@pytest.mark.parametrize(
    "flat",
    [
        np.zeros((1, 8, 3), np.uint8),
        np.zeros((8, 8), np.uint8),
        np.zeros((8, 8, 4), np.uint8),
        np.zeros((8, 8, 3), np.float32),
        np.zeros((4097, 2, 3), np.uint8),
    ],
)
def test_generator_rejects_invalid_flat_templates(flat):
    with pytest.raises(FlattenerError):
        generate_paper_sample(flat, 1, PaperParameters(), photo_size=(128, 128))


def test_capture_full_photo_quad_uses_original_pixel_centers_after_resize():
    corners = [[0, 0], [319, 0], [319, 239], [0, 239]]
    geometry = make_capture_geometry(corners, (240, 320), (17, 23), (128, 192))
    y, x = np.mgrid[:17, :23]
    expected = np.stack(
        ((2 * (x + 0.5) / 23 - 1) * 319 / 320, (2 * (y + 0.5) / 17 - 1) * 239 / 240),
        axis=-1,
    ).astype(np.float32)
    np.testing.assert_allclose(geometry.backward_map, expected, rtol=0, atol=1e-7)
    assert geometry.valid_mask.all() and geometry.photo_page_mask.all()
    assert geometry.synthesis_map.shape == (128, 192, 2)
    json.dumps(geometry.metadata, allow_nan=False)


def test_capture_nonrectangular_quad_matches_independent_rational_homography():
    # x = .5u / (1+.2u), y = .6v / (1+.2u); construct manual corners exactly.
    flat_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    projected = flat_corners * [0.5, 0.6] / (1 + 0.2 * flat_corners[:, :1])
    corners = (projected + 1) * [320, 240] / 2 - 0.5
    geometry = make_capture_geometry(corners, (240, 320), (13, 17), (128, 144))
    row, col = np.mgrid[:13, :17]
    u, v = 2 * (col + 0.5) / 17 - 1, 2 * (row + 0.5) / 13 - 1
    expected = np.stack((0.5 * u / (1 + 0.2 * u), 0.6 * v / (1 + 0.2 * u)), axis=-1)
    np.testing.assert_allclose(geometry.backward_map, expected, rtol=0, atol=1e-7)
    row, col = np.mgrid[:128, :144]
    x, y = 2 * (col + 0.5) / 144 - 1, 2 * (row + 0.5) / 128 - 1
    u = x / (0.5 - 0.2 * x)
    v = y * (1 + 0.2 * u) / 0.6
    mask = (np.abs(u) <= 1) & (np.abs(v) <= 1)
    inverse = np.stack((u, v), axis=-1)
    np.testing.assert_array_equal(geometry.photo_page_mask, mask)
    np.testing.assert_allclose(geometry.synthesis_map[mask], inverse[mask], rtol=0, atol=1e-7)
    assert np.isnan(geometry.synthesis_map[~mask]).all()


@pytest.mark.parametrize(
    "corners",
    [
        [[0, 0], [127, 127], [127, 0], [0, 127]],  # Crossed order.
        [[0, 0], [0, 127], [127, 127], [127, 0]],  # Reversed orientation.
        [[0, 0], [127, 0], [127, 0], [0, 127]],  # Degenerate edge.
        [[0, 0], [127, 0], [40, 40], [0, 127]],  # Concave quad.
        [[-1, 0], [127, 0], [127, 127], [0, 127]],  # Outside physical image.
        [[np.nan, 0], [127, 0], [127, 127], [0, 127]],
        [[0, 0], [127, 0], [127, 127]],
    ],
)
def test_capture_rejects_invalid_or_wrongly_ordered_quad(corners):
    with pytest.raises(FlattenerError):
        make_capture_geometry(corners, (128, 128), (13, 17), (128, 128))
