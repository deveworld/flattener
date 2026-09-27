"""Opt-in fast check of a grid's dense map, for the render stage.

``flattener.render`` checks a dense map cell by cell: the out-of-bounds fraction of the
output, then the four corner determinants of every cell between neighbouring pixel
centers, ``det(dP/ds, dP/dt) / (4 / (W H))`` in float64 (``validation._check_cells``). At
12 MP that check costs about two thirds of the render. ``fast_check`` reaches the same
decisions for a map that densifies a grid (``scan.placement.dense_map``) with less work:

* Jacobian: the dense map is bilinear in the output coordinate inside each grid cell
  (pieces split at the grid nodes that the output pixels cross; the edge cells extend
  linearly for a border). For a bilinear patch the renderer's four determinants of a cell
  are the exact Jacobian determinant at the cell's corners, and that determinant is
  affine in each local coordinate. So over the cells inside one piece the renderer's
  minimum is the determinant at the piece's extreme pixel centers, which
  ``grid_jacobian`` evaluates from the grid nodes. The few cells whose pixels straddle a
  grid node mix two pieces; for those the renderer's own formula runs on the
  full-resolution map (about 2 % of the cells at 12 MP). The result is the renderer's
  minimum and fold decision, computed from float64 grid values (the renderer reads the
  float32 map).
  ``continuous_min_jacobian_ratio`` adds the minimum over the whole span between pixel
  centers: a value at or below the threshold with a passing check marks a fold narrower
  than a pixel, which neither check rejects.
* Bounds: the out-of-bounds fraction and the valid pixels come from the full-resolution
  map, which the full-resolution sampling needs anyway, so they are exact.
* Proxy: ``flattener.render`` runs once on the same grid densified at no more than
  ``PROXY_PIXELS`` (2 MP) and no more than ``PROXY_FRACTION`` (1/16) of the output's
  pixels, sampled from a box-reduced source level, so it costs a small part of a full
  render. Its status, validation record and image are kept (``FastCheck.proxy``); the
  decision is the one above, since the proxy's coarser pixel lattice can land in a fold
  narrower than a full-resolution pixel.
* Sampling: ``FastCheck.sample`` samples the full-resolution map with
  ``dewarp.predict.sample_rgb``, ``flattener.render``'s sampling (the same image for
  ``round``).

``nodes`` places the grid nodes in the output as ``dense_map`` does (``flattener`` or
``uvdoc``).

Failure reasons and their order are the renderer's: ``non_finite_grid``,
``excessive_out_of_bounds``, ``fold_or_degenerate``, ``empty_valid_region``, each with
``Status.INVALID_TRANSFORM``. ``metadata["validation"]`` has the renderer's keys; its
cell counts are grid pieces plus straddling cells, and ``check_resolution`` says what was
checked where.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import FlattenerError, RenderConfig, RenderResult, Status
from ..dewarp.coords import center_positions
from ..dewarp.predict import Inset, resolve_inset, sample_rgb, source_inside
from ..renderer import render
from ..validation import validate_config, validate_image, validate_size
from .placement import CONVENTIONS as NODE_CONVENTIONS
from .placement import dense_map

PROXY_PIXELS = 2_000_000
PROXY_FRACTION = 1 / 16  # of the output's pixels: a quarter of each side
LEVEL_FACTORS = (1, 2, 4, 8)
Levels = Callable[[int], NDArray[np.uint8]]


def _positions(count: int, nodes: int, shrink: float, convention: str) -> NDArray[np.float64]:
    """Fractional node index of each output pixel center on one axis (``dense_map``)."""
    if convention == "uvdoc":
        base = np.linspace(-1.0, 1.0, count, dtype=np.float64)
    else:
        base = center_positions(count)
    return (shrink * base + 1.0) / 2.0 * (nodes - 1)


def _pixel_span(count: int, convention: str) -> float:
    """``count`` times the node-index step between neighbouring output pixels, over
    ``shrink * (nodes - 1)``: 1 for edge-aligned nodes, ``count / (count - 1)`` for
    ``uvdoc`` nodes."""
    return count / (count - 1) if convention == "uvdoc" else 1.0


@dataclass(frozen=True, slots=True)
class _Axis:
    """The pieces of one output axis: grid cell, span (node index units) and the first
    and last pixel inside each piece; and the pixel cells that straddle a grid node."""

    cells: NDArray[np.int64]
    starts: NDArray[np.float64]
    stops: NDArray[np.float64]
    first: NDArray[np.int64]
    last: NDArray[np.int64]
    positions: NDArray[np.float64]
    straddling: NDArray[np.int64]


def _axis(count: int, nodes: int, shrink: float, convention: str) -> _Axis:
    index = _positions(count, nodes, shrink, convention)
    low, high = float(index[0]), float(index[-1])
    inner = np.arange(1, nodes - 1, dtype=np.float64)
    breaks = inner[(inner > low) & (inner < high)]
    edges = np.concatenate([[low], breaks, [high]])
    starts, stops = edges[:-1], edges[1:]
    cells = np.clip(np.floor((starts + stops) / 2.0), 0, nodes - 2).astype(np.int64)
    first = np.searchsorted(index, starts, side="left")
    last = np.searchsorted(index, stops, side="right") - 1
    # Pixel cell (j, j + 1) straddles node k when t_j < k < t_(j+1) for an interior node.
    following = np.maximum(np.floor(index[:-1]) + 1.0, 1.0)
    straddling = np.flatnonzero((following < index[1:]) & (following <= nodes - 2))
    return _Axis(cells, starts, stops, first, last, index, straddling)


def _cross(u: NDArray, v: NDArray) -> NDArray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _piece_minimum(
    array: NDArray[np.float64],
    xs: _Axis,
    ys: _Axis,
    sx: tuple[NDArray, NDArray],
    sy: tuple[NDArray, NDArray],
) -> NDArray[np.float64]:
    """Minimum Jacobian determinant (node index units) of each piece over the corners
    ``sx`` x ``sy`` (local cell coordinates)."""
    rows, cols = ys.cells[:, None], xs.cells[None, :]
    base = array[rows, cols]
    a = array[rows, cols + 1] - base  # along x (columns)
    c = array[rows + 1, cols] - base  # along y (rows)
    d = array[rows + 1, cols + 1] - array[rows + 1, cols] - array[rows, cols + 1] + base
    values = []
    for s in sx:
        for r in sy:
            values.append(_cross(a + r[:, None, None] * d, c + s[None, :, None] * d))
    return np.minimum.reduce(values)


def _straddle_minimum(
    backward_map: NDArray[np.float32], xs: _Axis, ys: _Axis
) -> tuple[NDArray[np.float64], int]:
    """The renderer's cell minima (renderer units) of the cells that straddle a node."""
    height, width = backward_map.shape[:2]
    scale = height * width / 4.0
    minima = []

    def cells(p00, p10, p01, p11) -> NDArray[np.float64]:
        p00, p10, p01, p11 = (p.astype(np.float64) for p in (p00, p10, p01, p11))
        top, bottom, left, right = p10 - p00, p11 - p01, p01 - p00, p11 - p10
        value = _cross(top, left)
        for u, v in ((top, right), (bottom, left), (bottom, right)):
            np.minimum(value, _cross(u, v), out=value)
        return value * scale

    columns = xs.straddling
    if columns.size:
        minima.append(
            cells(
                backward_map[:-1, columns],
                backward_map[:-1, columns + 1],
                backward_map[1:, columns],
                backward_map[1:, columns + 1],
            ).ravel()
        )
    rows = ys.straddling
    if rows.size:
        minima.append(
            cells(
                backward_map[rows, :-1],
                backward_map[rows, 1:],
                backward_map[rows + 1, :-1],
                backward_map[rows + 1, 1:],
            ).ravel()
        )
    if not minima:
        return np.empty(0), 0
    values = np.concatenate(minima)
    return values, int(values.size)


def grid_jacobian(
    grid: NDArray,
    output_size: tuple[int, int],
    *,
    inset: Inset = None,
    border: float = 0.0,
    minimum: float = RenderConfig().min_jacobian_ratio,
    backward_map: NDArray[np.float32] | None = None,
    nodes: str = "flattener",
) -> dict[str, Any]:
    """The renderer's fold check of ``dense_map(grid, output_size, inset, border, nodes)``
    from the grid (see the module docstring).

    ``backward_map`` is that dense map, needed for the cells that straddle a grid node
    (computed when not given). Returns ``min_jacobian_ratio`` (renderer units; ``None``
    with no cells), ``continuous_min_jacobian_ratio``, ``checked_cells`` and
    ``failed_cells`` (pieces plus straddling cells), ``straddling_cells`` and
    ``fold_check`` (``passed``, ``failed`` or ``not_applicable`` for an output one pixel
    wide or high)."""
    array = np.asarray(grid, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 2 or min(array.shape[:2]) < 2:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_grid", "grid must be (height >= 2, width >= 2, 2)"
        )
    height, width = output_size
    stats: dict[str, Any] = {
        "fold_check": "not_applicable",
        "checked_cells": 0,
        "failed_cells": 0,
        "straddling_cells": 0,
        "min_jacobian_ratio": None,
        "continuous_min_jacobian_ratio": None,
    }
    if height < 2 or width < 2:
        return stats
    if nodes not in NODE_CONVENTIONS:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_grid", f"nodes must be one of {NODE_CONVENTIONS}"
        )
    ex, ey = resolve_inset(inset)
    gh, gw = array.shape[:2]
    shrink_x, shrink_y = 1.0 - ex + border, 1.0 - ey + border
    xs, ys = _axis(width, gw, shrink_x, nodes), _axis(height, gh, shrink_y, nodes)
    span = _pixel_span(width, nodes) * _pixel_span(height, nodes)
    scale = shrink_x * (gw - 1) * shrink_y * (gh - 1) * span / 4.0
    continuous = _piece_minimum(
        array,
        xs,
        ys,
        (xs.starts - xs.cells, xs.stops - xs.cells),
        (ys.starts - ys.cells, ys.stops - ys.cells),
    )
    # Pieces holding at least two pixel centers on both axes contain whole cells.
    usable_x = xs.last - xs.first >= 1
    usable_y = ys.last - ys.first >= 1
    values = []
    if usable_x.any() and usable_y.any():
        px = _Axis(
            *(part[usable_x] for part in (xs.cells, xs.starts, xs.stops, xs.first, xs.last)),
            xs.positions,
            xs.straddling,
        )
        py = _Axis(
            *(part[usable_y] for part in (ys.cells, ys.starts, ys.stops, ys.first, ys.last)),
            ys.positions,
            ys.straddling,
        )
        sx = (xs.positions[px.first] - px.cells, xs.positions[px.last] - px.cells)
        sy = (ys.positions[py.first] - py.cells, ys.positions[py.last] - py.cells)
        values.append((_piece_minimum(array, px, py, sx, sy) * scale).ravel())
    if xs.straddling.size or ys.straddling.size:
        if backward_map is None:
            backward_map = dense_map(
                array, (height, width), inset=inset, border=border, nodes=nodes
            )
        straddle, count = _straddle_minimum(backward_map, xs, ys)
        values.append(straddle)
        stats["straddling_cells"] = count
    checked = np.concatenate(values) if values else np.empty(0)
    failed = int(np.count_nonzero(checked <= minimum))
    stats.update(
        fold_check="failed" if failed else "passed",
        checked_cells=int(checked.size),
        failed_cells=failed,
        min_jacobian_ratio=float(checked.min()) if checked.size else None,
        continuous_min_jacobian_ratio=float(continuous.min() * scale),
    )
    return stats


def proxy_size(output_size: tuple[int, int], pixels: int = PROXY_PIXELS) -> tuple[int, int]:
    """The output size scaled down to at most ``pixels`` and ``PROXY_FRACTION`` of its
    own pixels (each side at least 2 and at most the output's)."""
    height, width = output_size
    scale = min(1.0, math.sqrt(min(pixels, PROXY_FRACTION * height * width) / (height * width)))
    size = (max(2, math.floor(height * scale)), max(2, math.floor(width * scale)))
    return (min(size[0], height), min(size[1], width))


@dataclass(eq=False)
class FastCheck:
    """The decision of ``fast_check`` and what the full-resolution sampling needs."""

    status: Status
    reason_code: str | None
    message: str
    backward_map: NDArray[np.float32] | None
    valid_mask: NDArray[np.bool_] | None
    validation: dict[str, Any]
    proxy: RenderResult | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == Status.SUCCESS

    def sample(
        self, image: NDArray[np.uint8], fill_rgb: tuple[int, int, int], quantize: str = "round"
    ) -> NDArray[np.uint8]:
        """The full-resolution image (``flattener.render``'s sampling)."""
        if not self.ok:
            raise FlattenerError(self.status, self.reason_code or "failed", self.message)
        return sample_rgb(image, self.backward_map, self.valid_mask, fill_rgb, quantize)

    def result(
        self,
        image: NDArray[np.uint8] | None = None,
        fill_rgb: tuple[int, int, int] = (255, 255, 255),
        quantize: str = "round",
    ) -> RenderResult:
        """A ``RenderResult`` of the check; with ``image`` (the source) it is sampled."""
        output = None if image is None or not self.ok else self.sample(image, fill_rgb, quantize)
        target = None
        if self.valid_mask is not None and self.ok:
            target = np.ones(self.valid_mask.shape, dtype=bool)
        return RenderResult(
            status=self.status,
            reason_code=self.reason_code,
            message=self.message,
            image=output,
            source_mask=self.valid_mask if self.ok else None,
            target_mask=target,
            valid_mask=self.valid_mask if self.ok else None,
            metadata={**self.metadata, "validation": self.validation},
        )


def _level(image: NDArray[np.uint8], levels: Levels | None, density: float) -> NDArray[np.uint8]:
    factor = max(f for f in LEVEL_FACTORS if f == 1 or f <= density)
    if factor == 1 or levels is None:
        return image
    return levels(factor)


def fast_check(
    image: NDArray[np.uint8],
    grid: NDArray,
    output_size: tuple[int, int],
    *,
    inset: Inset = None,
    border: float = 0.0,
    config: RenderConfig | None = None,
    proxy_pixels: int | None = PROXY_PIXELS,
    levels: Levels | None = None,
    nodes: str = "flattener",
) -> FastCheck:
    """Check ``dense_map(grid, output_size, inset, border, nodes)`` for ``image`` (see the
    module docstring). ``levels(f)`` gives the source reduced by f for the proxy render of
    at most ``proxy_pixels`` (``None``: no proxy)."""
    started = time.perf_counter()
    config = validate_config(config)
    validate_image(image, config)
    height, width = validate_size(
        output_size, max_pixels=config.max_output_pixels, max_side=config.max_side, name="output"
    )
    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "check": "fast_check",
        "check_resolution": {"bounds": [height, width], "jacobian": "grid_pieces", "proxy": None},
        "timings": timings,
    }
    array = np.asarray(grid, dtype=np.float64)
    target_pixels = height * width
    validation: dict[str, Any] = {
        "target_pixels": target_pixels,
        "valid_pixels": 0,
        "out_of_bounds_pixels": None,
        "out_of_bounds_fraction": None,
        "fold_check": None,
        "checked_cells": 0,
        "excluded_cells": 0,
        "failed_cells": 0,
        "min_jacobian_ratio": None,
    }

    def fail(reason: str, message: str) -> FastCheck:
        timings["total"] = time.perf_counter() - started
        return FastCheck(
            Status.INVALID_TRANSFORM, reason, message, None, None, validation, None, metadata
        )

    if not np.isfinite(array).all():
        return fail("non_finite_grid", "backward_map contains non-finite target coordinates")
    step = time.perf_counter()
    backward = dense_map(array, (height, width), inset=inset, border=border, nodes=nodes)
    if not np.isfinite(backward).all():
        return fail("non_finite_grid", "backward_map contains non-finite target coordinates")
    valid = source_inside(backward)
    timings["bounds"] = time.perf_counter() - step
    valid_pixels = int(np.count_nonzero(valid))
    outside = target_pixels - valid_pixels
    validation.update(
        valid_pixels=valid_pixels,
        out_of_bounds_pixels=outside,
        out_of_bounds_fraction=outside / target_pixels,
    )
    if validation["out_of_bounds_fraction"] > config.max_out_of_bounds_fraction:
        return fail("excessive_out_of_bounds", "too many target coordinates lie outside the source")
    step = time.perf_counter()
    jacobian = grid_jacobian(
        array,
        (height, width),
        inset=inset,
        border=border,
        minimum=config.min_jacobian_ratio,
        backward_map=backward,
        nodes=nodes,
    )
    timings["jacobian"] = time.perf_counter() - step
    validation.update(jacobian)
    if jacobian["failed_cells"]:
        return fail("fold_or_degenerate", "a target cell has a reversed or degenerate Jacobian")
    if not valid_pixels:
        return fail("empty_valid_region", "no target coordinates lie inside the source")
    proxy = None
    if proxy_pixels is not None:
        step = time.perf_counter()
        small = proxy_size((height, width), proxy_pixels)
        source_height, source_width = image.shape[:2]
        density = math.sqrt(source_height * source_width / (small[0] * small[1]))
        proxy_map = dense_map(array, small, inset=inset, border=border, nodes=nodes)
        level = _level(image, levels, density / 1.5)
        proxy = render(level, proxy_map, config=replace(config, on_error="return"))
        timings["proxy"] = time.perf_counter() - step
        metadata["check_resolution"]["proxy"] = list(small)
        metadata["proxy"] = {
            "status": str(proxy.status),
            "reason_code": proxy.reason_code,
            "validation": proxy.metadata.get("validation"),
            "source_size": list(level.shape[:2]),
        }
    timings["total"] = time.perf_counter() - started
    return FastCheck(Status.SUCCESS, None, "", backward, valid, validation, proxy, metadata)
