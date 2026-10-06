"""Coordinate conventions for dewarp grids, derived from each producer's code.

Values (where a coordinate points in the *source* image of n pixels per axis):

* flattener: ``align_corners=False``; +-1 are the outer pixel edges and pixel
  center i is at ``(2i + 1) / n - 1``. Resolution independent.
* align_corners=True (UVDoc model output, applied with ``grid_sample(...,
  align_corners=True)``): center i is at ``2i / (n - 1) - 1``. Equating the
  pixel index gives ``q = u (n - 1) / n``.
* SyntheticDoc ``backward_map.npy``: ``uv_to_backward_map.py`` interpolates
  render pixel-center *indices* x and stores ``b = 2x / n - 1`` (divided by the
  render size n). Hence ``q = b + 1 / n``, i.e. ``q = b + (1/W, 1/H)``.
* raw pixel grids (UVDoc/Doc3D ``.mat``): pixel centers at integers, so
  ``q = (2x + 1) / n - 1``.

Positions (where a node sits in the *output* page):

* dewarp grid (Gh, Gw, 2), used for all targets and ``"flattener"`` models:
  edge-aligned, node (i, j) at ``(2j / (Gw - 1) - 1, 2i / (Gh - 1) - 1)``; the
  corner nodes are the page corners. Resolution independent.
* flattener backward map (H, W, 2) and SyntheticDoc maps: center-aligned,
  value (i, j) at output pixel center ``((2j + 1) / W - 1, (2i + 1) / H - 1)``.
  (SyntheticDoc shifts its triangulation by half a map pixel for this.)
* UVDoc upstream densification ``F.interpolate(align_corners=True)`` puts
  node j on output pixel center ``j (Wo - 1) / (Gw - 1)``, which depends on the
  output size; ``uvdoc_dense_map`` reproduces it exactly.

All array helpers take (..., 2) arrays in (x, y) order and sizes as
(height, width), compute in float64, and return float32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

Size = tuple[int, int]


def _size(size: Size, name: str = "size", minimum: int = 1) -> tuple[int, int]:
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in size)
        or min(size) < minimum
    ):
        raise ValueError(f"{name} must be a (height, width) pair of integers >= {minimum}")
    return int(size[0]), int(size[1])


def _xy(values: NDArray, name: str = "values") -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 2 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating (..., 2) array in (x, y) order")
    return array.astype(np.float64)


def _wh(size: Size, minimum: int = 1) -> NDArray[np.float64]:
    height, width = _size(size, minimum=minimum)
    return np.array([width, height], dtype=np.float64)


def align_corners_to_flattener(values: NDArray, size: Size) -> NDArray[np.float32]:
    """``align_corners=True`` normalized values -> flattener values; size of the source."""
    n = _wh(size, minimum=2)
    return (_xy(values) * (n - 1) / n).astype(np.float32)


def flattener_to_align_corners(values: NDArray, size: Size) -> NDArray[np.float32]:
    n = _wh(size, minimum=2)
    return (_xy(values) * n / (n - 1)).astype(np.float32)


def syntheticdoc_to_flattener(values: NDArray, render_size: Size) -> NDArray[np.float32]:
    """SyntheticDoc ``backward_map.npy`` values -> flattener; ``render_size`` = upstream render."""
    return (_xy(values) + 1.0 / _wh(render_size)).astype(np.float32)


def flattener_to_syntheticdoc(values: NDArray, render_size: Size) -> NDArray[np.float32]:
    return (_xy(values) - 1.0 / _wh(render_size)).astype(np.float32)


def pixels_to_flattener(points: NDArray, size: Size) -> NDArray[np.float32]:
    """Pixel coordinates (centers at integers) -> flattener normalized values."""
    return ((2.0 * _xy(points, "points") + 1.0) / _wh(size) - 1.0).astype(np.float32)


def flattener_to_pixels(values: NDArray, size: Size) -> NDArray[np.float32]:
    return (((_xy(values) + 1.0) * _wh(size) - 1.0) / 2.0).astype(np.float32)


def edge_positions(count: int) -> NDArray[np.float64]:
    """Normalized output positions of ``count`` edge-aligned grid nodes."""
    return np.linspace(-1.0, 1.0, count, dtype=np.float64)


def center_positions(count: int) -> NDArray[np.float64]:
    """Normalized positions of ``count`` pixel centers (flattener convention)."""
    return (2.0 * np.arange(count, dtype=np.float64) + 1.0) / count - 1.0


def interpolation_matrix(index: NDArray, n: int, *, extrapolate: bool) -> NDArray[np.float64]:
    """Rows of 1-D linear interpolation weights at fractional node ``index`` in [0, n-1].

    Outside that range the end segment is extended linearly (``extrapolate``)
    or the end value is held.
    """
    t = np.asarray(index, dtype=np.float64)
    matrix = np.zeros((t.size, n), dtype=np.float64)
    if n == 1:
        matrix[:] = 1.0
        return matrix
    if not extrapolate:
        t = np.clip(t, 0.0, n - 1.0)
    left = np.clip(np.floor(t), 0, n - 2).astype(np.int64)
    weight = t - left
    rows = np.arange(t.size)
    matrix[rows, left] = 1.0 - weight
    matrix[rows, left + 1] += weight
    return matrix


def _apply(rows: NDArray, grid: NDArray, cols: NDArray) -> NDArray[np.float32]:
    """rows @ grid[..., c] @ cols.T per channel; NaN nodes poison only their support."""
    values = np.asarray(grid, dtype=np.float64)
    bad = ~np.isfinite(values).all(axis=-1)
    clean = np.where(bad[..., None], 0.0, values)
    out = np.einsum("hi,ijc,wj->hwc", rows, clean, cols, optimize=True)
    if bad.any():
        reach = np.abs(rows) @ bad.astype(np.float64) @ np.abs(cols).T
        out[reach > 0] = np.nan
    return out.astype(np.float32)


def _grid(values: NDArray, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.floating) or min(array.shape[:2]) < 2:
        raise ValueError(f"{name} must be a floating (height >= 2, width >= 2, C) array")
    return array


def map_to_grid(backward_map: NDArray, grid_size: Size) -> NDArray[np.float32]:
    """Center-aligned map (H, W, C) -> edge-aligned grid, exact for affine maps.

    The outer nodes lie half a map pixel beyond the outer centers; they are
    obtained by linear extrapolation of the bilinear interpolant.
    """
    array = _grid(backward_map, "backward_map")
    gh, gw = _size(grid_size, "grid_size", minimum=2)
    height, width = array.shape[:2]
    rows = interpolation_matrix(
        (edge_positions(gh) + 1) / 2 * height - 0.5, height, extrapolate=True
    )
    cols = interpolation_matrix((edge_positions(gw) + 1) / 2 * width - 0.5, width, extrapolate=True)
    return _apply(rows, array, cols)


def grid_to_map(grid: NDArray, output_size: Size) -> NDArray[np.float32]:
    """Edge-aligned grid (Gh, Gw, C) -> dense center-aligned map at ``output_size``."""
    array = _grid(grid, "grid")
    height, width = _size(output_size, "output_size")
    gh, gw = array.shape[:2]
    rows = interpolation_matrix(
        (center_positions(height) + 1) / 2 * (gh - 1), gh, extrapolate=False
    )
    cols = interpolation_matrix((center_positions(width) + 1) / 2 * (gw - 1), gw, extrapolate=False)
    return _apply(rows, array, cols)


def resize_grid(grid: NDArray, grid_size: Size) -> NDArray[np.float32]:
    """Resample an edge-aligned grid to another node count (align_corners=True resize).

    Integer node ratios reproduce strided subsampling, e.g. 89x61 -> 45x31 == [::2, ::2].
    """
    array = _grid(grid, "grid")
    gh, gw = _size(grid_size, "grid_size", minimum=2)
    rows = interpolation_matrix(
        np.linspace(0, array.shape[0] - 1, gh), array.shape[0], extrapolate=False
    )
    cols = interpolation_matrix(
        np.linspace(0, array.shape[1] - 1, gw), array.shape[1], extrapolate=False
    )
    return _apply(rows, array, cols)


def uvdoc_dense_map(grid: NDArray, output_size: Size, source_size: Size) -> NDArray[np.float32]:
    """Exact flattener equivalent of upstream UVDoc ``bilinear_unwarping``.

    ``grid`` holds align_corners=True values (a raw UVDoc model output, as
    (Gh, Gw, 2)); ``source_size`` is the image it is applied to. The result
    rendered by ``flattener.render`` samples the same source points as
    ``grid_sample(interpolate(grid, output_size, align_corners=True),
    align_corners=True)``; outside the source it uses ``fill_rgb`` where upstream pads
    with zeros.
    """
    array = _grid(grid, "grid")
    height, width = _size(output_size, "output_size")
    gh, gw = array.shape[:2]
    rows = interpolation_matrix(np.linspace(0, gh - 1, height), gh, extrapolate=False)
    cols = interpolation_matrix(np.linspace(0, gw - 1, width), gw, extrapolate=False)
    return align_corners_to_flattener(_apply(rows, array, cols), source_size)


def _torch_matrix(count: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import torch

    index = (center_positions(count) + 1) / 2 * (n - 1)
    matrix = interpolation_matrix(index, n, extrapolate=False)
    return torch.from_numpy(matrix).to(device=device, dtype=dtype)


def grid_to_map_torch(grid: torch.Tensor, output_size: Size) -> torch.Tensor:
    """Differentiable ``grid_to_map`` for (B, C, Gh, Gw) -> (B, C, H, W)."""
    if grid.ndim != 4 or min(grid.shape[-2:]) < 2:
        raise ValueError("grid must have shape (B, C, Gh >= 2, Gw >= 2)")
    height, width = _size(output_size, "output_size")
    rows = _torch_matrix(height, grid.shape[-2], grid.device, grid.dtype)
    cols = _torch_matrix(width, grid.shape[-1], grid.device, grid.dtype)
    return rows @ grid @ cols.T


def unwarp_torch(image: torch.Tensor, grid: torch.Tensor, output_size: Size) -> torch.Tensor:
    """Sample (B, 3, H, W) images with flattener-convention grids (B, 2, Gh, Gw)."""
    import torch

    dense = grid_to_map_torch(grid, output_size).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        image, dense.to(image.dtype), mode="bilinear", padding_mode="border", align_corners=False
    )
