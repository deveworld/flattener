"""Grid conventions: value conversions, node positions and render round trips."""

import numpy as np
import torch
import torch.nn.functional as F

from flattener import RenderConfig, identity_map, render
from flattener.dewarp.coords import (
    align_corners_to_flattener,
    edge_positions,
    flattener_to_align_corners,
    flattener_to_pixels,
    flattener_to_syntheticdoc,
    grid_to_map,
    grid_to_map_torch,
    map_to_grid,
    pixels_to_flattener,
    resize_grid,
    syntheticdoc_to_flattener,
    unwarp_torch,
    uvdoc_dense_map,
)

PERMISSIVE = RenderConfig(max_out_of_bounds_fraction=1.0)


def _syntheticdoc_generator_map(page_to_pixel, map_size, render_size):
    """Replicates upstream ``uv_to_backward_map.py`` for a page whose UV->pixel map is
    affine (linear interpolation of the triangulation is then exact): node (i, j)
    samples u = (j + 0.5) / W, v' = (i + 0.5) / H (after the V flip) and stores
    the render pixel-center index, normalized as 2x / size - 1."""
    height, width = map_size
    u, v = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
    pixels = page_to_pixel(np.stack([u, v], axis=-1))
    return (pixels / np.array(render_size[::-1]) * 2 - 1).astype(np.float32)


def test_value_conversions_invert_and_agree_on_pixel_meaning():
    rng = np.random.default_rng(0)
    values = rng.uniform(-1.2, 1.2, (5, 7, 2)).astype(np.float32)
    size = (37, 23)
    for forward, backward in (
        (align_corners_to_flattener, flattener_to_align_corners),
        (syntheticdoc_to_flattener, flattener_to_syntheticdoc),
        (pixels_to_flattener, flattener_to_pixels),
    ):
        np.testing.assert_allclose(backward(forward(values, size), size), values, atol=1e-6)
    # align_corners=True: -1/+1 are the outer pixel *centers* 0 and n-1.
    corners = align_corners_to_flattener(np.array([[-1.0, -1.0], [1.0, 1.0]]), size)
    np.testing.assert_allclose(flattener_to_pixels(corners, size), [[0, 0], [22, 36]], atol=1e-5)
    # SyntheticDoc stores 2x/n - 1 for pixel-center index x.
    index = np.array([[0.0, 0.0], [22.0, 36.0]])
    stored = index / np.array([23, 37]) * 2 - 1
    np.testing.assert_allclose(
        flattener_to_pixels(syntheticdoc_to_flattener(stored, size), size), index, atol=1e-5
    )


def test_syntheticdoc_identity_page_renders_the_source_exactly():
    rng = np.random.default_rng(1)
    source = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
    size = source.shape[:2]
    stored = _syntheticdoc_generator_map(lambda uv: uv * [40, 30] - 0.5, size, size)
    converted = syntheticdoc_to_flattener(stored, size)
    np.testing.assert_allclose(converted, identity_map(size), atol=1e-6)
    np.testing.assert_array_equal(render(source, converted).image, source)
    # Reading the stored values as align_corners=True would shift the page.
    wrong = align_corners_to_flattener(stored, size)
    assert not np.array_equal(render(source, wrong, config=PERMISSIVE).image, source)


def test_syntheticdoc_affine_chain_is_exact_through_grid_and_dense_map():
    render_size, map_size = (144, 102), (18, 12)
    matrix = np.array([[70.0, 12.0], [-9.0, 100.0]])
    offset = np.array([15.0, 20.0])

    def page_to_pixel(uv):
        return uv @ matrix.T + offset

    stored = _syntheticdoc_generator_map(page_to_pixel, map_size, render_size)
    grid = map_to_grid(syntheticdoc_to_flattener(stored, render_size), (9, 7))
    for output_size in ((60, 44), (9, 7)):
        centers = (identity_map(output_size).astype(np.float64) + 1) / 2  # page (u, v')
        expected = pixels_to_flattener(page_to_pixel(centers), render_size)
        np.testing.assert_allclose(grid_to_map(grid, output_size), expected, atol=2e-6)
    corners = pixels_to_flattener(page_to_pixel(np.array([[0.0, 0.0], [1.0, 1.0]])), render_size)
    np.testing.assert_allclose(grid[[0, -1], [0, -1]], corners, atol=2e-6)


def test_uvdoc_dense_map_reproduces_upstream_bilinear_unwarping():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (50, 40, 3), dtype=np.uint8)
    base = np.stack(np.meshgrid(np.linspace(-0.8, 0.8, 7), np.linspace(-0.85, 0.8, 9)), -1)
    grid = (base + rng.normal(0, 0.02, base.shape)).astype(np.float32)
    output_size = (46, 33)
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].double() / 255
    upsampled = F.interpolate(
        torch.from_numpy(grid).permute(2, 0, 1)[None].double(),
        size=output_size,
        mode="bilinear",
        align_corners=True,
    )
    upstream = F.grid_sample(tensor, upsampled.permute(0, 2, 3, 1), align_corners=True)
    expected = np.floor(upstream[0].permute(1, 2, 0).numpy() * 255 + 0.5)
    ours = render(image, uvdoc_dense_map(grid, output_size, image.shape[:2])).image
    assert np.abs(ours.astype(int) - expected).max() <= 1


def test_torch_and_numpy_densification_agree_and_unwarp_identity():
    rng = np.random.default_rng(3)
    grid = rng.uniform(-1, 1, (6, 5, 2)).astype(np.float32)
    tensor = torch.from_numpy(grid).permute(2, 0, 1)[None].double()
    dense = grid_to_map_torch(tensor, (23, 17))[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(dense, grid_to_map(grid, (23, 17)), atol=1e-6)
    edges = np.stack(np.meshgrid(edge_positions(5), edge_positions(6)), -1).astype(np.float32)
    image = torch.rand(1, 3, 12, 10)
    identity = torch.from_numpy(edges).permute(2, 0, 1)[None]
    torch.testing.assert_close(unwarp_torch(image, identity, (12, 10)), image)


def test_resize_grid_matches_upstream_subsampling_and_nan_stays_local():
    rng = np.random.default_rng(4)
    grid = rng.normal(size=(89, 61, 2)).astype(np.float32)
    np.testing.assert_array_equal(resize_grid(grid, (45, 31)), grid[::2, ::2])
    centered = identity_map((40, 30))
    centered[20, 15] = np.nan
    coarse = map_to_grid(centered, (9, 7))
    invalid = ~np.isfinite(coarse).all(-1)
    assert 0 < invalid.sum() <= 4 and not invalid[0].any()
