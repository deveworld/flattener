"""Output size and aspect of a scanned page (design section 1.4, default ``"auto"``).

The input is the page grid G: ``(Gh, Gw, 2)`` flattener values in source coordinates
(the ``flattener.dewarp`` grid convention: edge-aligned nodes, corner nodes on the page
corners), rows and columns in upright page order. Lengths are in pixels of the source
image the grid refers to.

``"auto"``:

* Row arc length ``L_row(i) = sum_j |P(i, j+1) - P(i, j)|`` over the source pixel
  positions P of grid row i, and ``L_col(j)`` likewise down column j.
* Aspect ``a = mean(L_row) / mean(L_col)`` (width / height).
* Width ``W = round(max L_row)``: the row spanning the most source pixels (usually the
  edge nearest the camera) gets one output pixel per source pixel, so the output never
  has a finer pixel pitch than the best-sampled part of the page. Height
  ``H = round(W / a)``. Rounding is half up.
* An aspect within ``SNAP_TOLERANCE`` (1.5 %) of a standard page ratio
  (``STANDARD_ASPECTS``, or its reciprocal for a landscape page) takes that ratio.
* A size above ``max_side`` (8192) or ``max_pixels`` (16 MP) shrinks uniformly.

``OutputPlan.aspect_source`` records where the aspect came from: ``estimated``,
``snapped``, ``paper`` (a named paper size), ``user`` (a number) or ``output_size`` (an
explicit output size, used as given, e.g. a benchmark's crop size).

The arc-length ratio measures the page as the photo shows it. A page tilted away from
the camera by an angle t is foreshortened by about ``cos t`` along the tilt, so the
estimate stays within 1 % up to a tilt of about 8 degrees. Given the focal length in
source pixels (``focal_px``, e.g. from the EXIF 35 mm equivalent), the aspect of the
corner quad is recovered for a pinhole camera with the principal point at the source
center (the rectangle-from-homography construction) and then multiplied by the ratio of
the grid's arc-length aspect to that of the flat page through the same corners, which
carries the curl the photo shows. Curl toward the camera shortens rows in the photo and
is only partly measured. Pass ``focal_px`` only for whole camera frames, since a crop
moves the principal point.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import FlattenerError, Status

SNAP_TOLERANCE = 0.015  # relative difference to a standard ratio
STANDARD_ASPECTS: dict[str, float] = {  # width / height of the portrait page
    "iso": 1 / math.sqrt(2),  # 0.707: ISO A and B series, 국판, 크라운판
    "singukpan": 152 / 225,  # 0.676: 신국판 152 x 225 mm
    "4x6": 128 / 188,  # 0.681: 4x6판 128 x 188 mm
    "letter": 8.5 / 11,  # 0.773: US Letter
}
PAPER_NAMES: dict[str, str] = {
    **{name: name for name in STANDARD_ASPECTS},
    **{name: "iso" for name in ("a3", "a4", "a5", "a6", "b4", "b5", "b6", "gukpan", "crown")},
}
ASPECT_MODES = ("auto", "estimate", "snap")
ASPECT_SOURCES = ("estimated", "snapped", "paper", "user", "output_size")
ASPECT_METHODS = ("arc_length", "perspective")
SIZE_SOURCES = ("auto", "long_side", "output_size")
MAX_SIDE = 8192  # RenderConfig.max_side
MAX_PIXELS = 16_000_000  # RenderConfig.max_output_pixels
DIAGONAL_35MM = math.hypot(36.0, 24.0)  # mm; the reference of 35 mm equivalent focal lengths

Size = tuple[int, int]


def _option_error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_output_option", message)


def _degenerate(message: str) -> FlattenerError:
    return FlattenerError(Status.INVALID_TRANSFORM, "degenerate_grid", message)


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _positive_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 1


def _size(size: Any, name: str = "source_size") -> tuple[int, int]:
    if not isinstance(size, (tuple, list)) or len(size) != 2 or not all(map(_positive_int, size)):
        raise _option_error(f"{name} must be a (height, width) pair of positive integers")
    return int(size[0]), int(size[1])


def _inset(inset: Any) -> tuple[float, float]:
    try:
        ex, ey = (float(value) for value in inset)
    except (TypeError, ValueError) as exc:
        raise _option_error("inset must be a pair of numbers") from exc
    if not (0.0 <= ex < 1.0 and 0.0 <= ey < 1.0):
        raise _option_error("inset values must be in [0, 1)")
    return ex, ey


def source_points(grid: NDArray, source_size: Size) -> NDArray[np.float64]:
    """Grid of flattener values -> source pixel-edge positions (0 and W are the outer edges)."""
    array = np.asarray(grid)
    if (
        array.ndim != 3
        or array.shape[-1] != 2
        or min(array.shape[:2]) < 2
        or not np.issubdtype(array.dtype, np.floating)
    ):
        raise FlattenerError(
            Status.INPUT_ERROR,
            "invalid_grid",
            "grid must be a floating (height >= 2, width >= 2, 2) array",
        )
    if not np.isfinite(array).all():
        raise _degenerate("grid has non-finite values")
    height, width = _size(source_size)
    scale = np.array([width, height], dtype=np.float64)
    return (array.astype(np.float64) + 1.0) / 2.0 * scale


def _lengths(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows = np.hypot(*np.moveaxis(np.diff(points, axis=1), -1, 0)).sum(axis=1)
    cols = np.hypot(*np.moveaxis(np.diff(points, axis=0), -1, 0)).sum(axis=0)
    return rows, cols


def arc_lengths(
    grid: NDArray, source_size: Size
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(L_row, L_col)``: arc length in source pixels of every grid row and column."""
    return _lengths(source_points(grid, source_size))


def _arc_ratio(points: NDArray[np.float64]) -> float:
    rows, cols = _lengths(points)
    if not rows.mean() > 0 or not cols.mean() > 0:
        raise _degenerate("grid rows or columns have zero length")
    return float(rows.mean() / cols.mean())


def arc_aspect(grid: NDArray, source_size: Size) -> float:
    """``mean(L_row) / mean(L_col)``: the page aspect (width / height) as the photo shows it."""
    return _arc_ratio(source_points(grid, source_size))


def _unit_square_homography(corners: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """H with ``H (u, v, 1) ~ corner`` for (u, v) = (0, 0), (1, 0), (1, 1), (0, 1)."""
    rows, values = [], []
    for (u, v), (x, y) in zip(((0, 0), (1, 0), (1, 1), (0, 1)), corners, strict=True):
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y])
        values.extend((x, y))
    try:
        solution = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(values))
    except np.linalg.LinAlgError:
        return None
    matrix = np.append(solution, 1.0).reshape(3, 3)
    return matrix if np.isfinite(matrix).all() else None


def perspective_aspect(grid: NDArray, source_size: Size, focal_px: float) -> float | None:
    """Page aspect (width / height) with the perspective of a pinhole camera removed.

    ``focal_px`` is the focal length in source pixels; the principal point is the source
    center. The corner nodes give the homography H from the unit square; with camera
    coordinates ``(x - c) / f`` its first two columns are the page's width and height
    directions scaled by the page size, so their length ratio is the flat page aspect.
    That aspect is multiplied by the arc-length aspect of the grid over the arc-length
    aspect of the flat page ``H(u, v)`` sampled at the grid's nodes (1 for a flat page).
    ``None`` when the corners do not form a proper quad in front of the camera.
    """
    if not (isinstance(focal_px, (int, float)) and math.isfinite(focal_px) and focal_px > 0):
        raise _option_error("focal_px must be a positive finite number")
    height, width = _size(source_size)
    points = source_points(grid, source_size)
    camera = (points - np.array([width / 2.0, height / 2.0])) / float(focal_px)
    corners = np.array([camera[0, 0], camera[0, -1], camera[-1, -1], camera[-1, 0]])
    edges = np.roll(corners, -1, axis=0) - corners
    turns = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    scale = float(np.square(edges).sum())
    if not ((turns > 1e-9 * scale).all() or (turns < -1e-9 * scale).all()):
        return None  # a projected rectangle is a convex quad
    matrix = _unit_square_homography(corners)
    if matrix is None:
        return None
    gh, gw = camera.shape[:2]
    u, v = np.meshgrid(np.linspace(0, 1, gw), np.linspace(0, 1, gh))
    projected = np.stack((u, v, np.ones_like(u)), axis=-1) @ matrix.T
    depth = projected[..., 2]
    if not (depth > 0).all() and not (depth < 0).all():
        return None
    flat = projected[..., :2] / depth[..., None]
    rect = float(np.linalg.norm(matrix[:, 0]) / np.linalg.norm(matrix[:, 1]))
    try:
        aspect = rect * _arc_ratio(camera) / _arc_ratio(flat)
    except FlattenerError:
        return None
    return aspect if math.isfinite(aspect) and aspect > 0 else None


def estimate_aspect(
    grid: NDArray, source_size: Size, *, focal_px: float | None = None
) -> tuple[float, str]:
    """``(aspect, method)``: ``perspective_aspect`` when ``focal_px`` is given and the
    quad is usable, else ``arc_aspect`` (method ``arc_length``)."""
    if focal_px is not None:
        aspect = perspective_aspect(grid, source_size, focal_px)
        if aspect is not None:
            return aspect, "perspective"
    return arc_aspect(grid, source_size), "arc_length"


def focal_px_from_35mm(focal_35mm: float, size: Size) -> float:
    """Focal length in pixels of an image of ``size`` from its 35 mm equivalent."""
    height, width = _size(size, "size")
    return float(focal_35mm) * math.hypot(width, height) / DIAGONAL_35MM


def _candidates() -> list[tuple[str, float]]:
    pairs = []
    for name, ratio in STANDARD_ASPECTS.items():
        pairs += [(name, ratio), (f"{name}-landscape", 1.0 / ratio)]
    return pairs


def snap_aspect(aspect: float, *, tolerance: float = SNAP_TOLERANCE) -> tuple[float, str] | None:
    """The nearest standard ratio within ``tolerance`` (relative) as ``(ratio, name)``."""
    best = None
    for name, ratio in _candidates():
        error = abs(aspect - ratio) / ratio
        if error <= tolerance and (best is None or error < best[2]):
            best = (ratio, name, error)
    return None if best is None else (best[0], best[1])


def resolve_aspect(aspect: str | float, estimate: float | None) -> tuple[float, str, str | None]:
    """``(aspect, aspect_source, paper)`` for an aspect option.

    ``"auto"`` snaps the estimate within ``SNAP_TOLERANCE``; ``"estimate"`` keeps it;
    ``"snap"`` takes the nearest standard ratio; a paper name (``PAPER_NAMES``) takes
    that ratio, landscape when the estimate is above 1; a number is used as given.
    """
    if isinstance(aspect, (int, float, np.floating)) and not isinstance(aspect, bool):
        value = float(aspect)
        if not (math.isfinite(value) and value > 0):
            raise _option_error("aspect must be a positive finite number")
        return value, "user", None
    if not isinstance(aspect, str):
        raise _option_error("aspect must be a mode, a paper name or a number")
    key = aspect.lower()
    if key not in ASPECT_MODES and key not in PAPER_NAMES:
        raise _option_error(f"aspect must be one of {ASPECT_MODES}, a paper name or a number")
    if key in PAPER_NAMES:
        ratio = STANDARD_ASPECTS[PAPER_NAMES[key]]
        if estimate is not None and estimate > 1.0:
            return 1.0 / ratio, "paper", f"{PAPER_NAMES[key]}-landscape"
        return ratio, "paper", PAPER_NAMES[key]
    if estimate is None or not (math.isfinite(estimate) and estimate > 0):
        raise _option_error(f"aspect {aspect!r} needs a positive aspect estimate")
    if key == "estimate":
        return float(estimate), "estimated", None
    snapped = snap_aspect(estimate, tolerance=math.inf if key == "snap" else SNAP_TOLERANCE)
    if snapped is None:
        return float(estimate), "estimated", None
    return snapped[0], "snapped", snapped[1]


def sampling_density(
    grid: NDArray,
    source_size: Size,
    output_size: Size,
    *,
    inset: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """``(min, median)`` source pixels per output pixel over the grid cells.

    Per cell: the square root of its source area (the quad through its four nodes) over
    its output area. Below 1 the output enlarges the source there.
    """
    points = source_points(grid, source_size)
    height, width = _size(output_size, "output_size")
    ex, ey = _inset(inset)
    first = points[1:, 1:] - points[:-1, :-1]
    second = points[1:, :-1] - points[:-1, 1:]
    area = 0.5 * np.abs(first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
    gh, gw = points.shape[:2]
    cell = (width / ((gw - 1) * (1.0 - ex))) * (height / ((gh - 1) * (1.0 - ey)))
    density = np.sqrt(area / cell)
    return float(density.min()), float(np.median(density))


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Output size of one scan and how it was chosen (see the module docstring)."""

    height: int
    width: int
    aspect: float  # page aspect (width / height) the size was made from
    aspect_source: str
    size_source: str
    aspect_estimate: float | None = None  # estimate from the grid
    aspect_method: str | None = None  # arc_length or perspective
    paper: str | None = None  # standard ratio name when snapped or named
    density_width: float | None = None  # max L_row in source pixels
    scale: float = 1.0  # uniform factor applied for the size limits
    limited_by: str | None = None  # max_side or max_pixels
    density_min: float | None = None  # source pixels per output pixel
    density_median: float | None = None

    def __post_init__(self) -> None:
        if not (_positive_int(self.height) and _positive_int(self.width)):
            raise _option_error("output height and width must be positive integers")
        if self.aspect_source not in ASPECT_SOURCES or self.size_source not in SIZE_SOURCES:
            raise _option_error("unknown aspect_source or size_source")
        if self.aspect_method is not None and self.aspect_method not in ASPECT_METHODS:
            raise _option_error("unknown aspect_method")
        for name in ("height", "width"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in (
            "aspect",
            "aspect_estimate",
            "density_width",
            "scale",
            "density_min",
            "density_median",
        ):
            value = getattr(self, name)
            if value is not None:
                if not math.isfinite(float(value)):
                    raise _option_error(f"{name} must be finite")
                object.__setattr__(self, name, float(value))

    @property
    def size(self) -> tuple[int, int]:
        return self.height, self.width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputPlan:
        return cls(**data)


def _fit(
    width: float, height: float, aspect: float, anchor: str, max_side: int, max_pixels: int
) -> tuple[int, int, float, str | None]:
    """Whole-pixel ``(height, width, scale, limited_by)`` for a continuous size."""

    def other(side: int) -> tuple[int, int]:
        if anchor == "width":
            return max(1, round_half_up(side / aspect)), side
        return side, max(1, round_half_up(side * aspect))

    def fits(size: tuple[int, int]) -> bool:
        return max(size) <= max_side and size[0] * size[1] <= max_pixels

    anchor_value = width if anchor == "width" else height
    size = other(max(1, round_half_up(anchor_value)))
    if fits(size):
        return size[0], size[1], 1.0, None
    side_scale = max_side / max(width, height)
    pixel_scale = math.sqrt(max_pixels / (width * height))
    limited_by = "max_side" if side_scale <= pixel_scale else "max_pixels"
    side = max(1, math.floor(anchor_value * min(1.0, side_scale, pixel_scale)))
    size = other(side)
    while side > 1 and not fits(size):
        side -= 1
        size = other(side)
    if not fits(size):
        raise _option_error("no output size within the limits keeps this aspect")
    return size[0], size[1], side / anchor_value, limited_by


def plan_output(
    grid: NDArray,
    source_size: Size,
    *,
    output: str | int | Size = "auto",
    aspect: str | float = "auto",
    focal_px: float | None = None,
    inset: tuple[float, float] = (0.0, 0.0),
    max_side: int = MAX_SIDE,
    max_pixels: int = MAX_PIXELS,
) -> OutputPlan:
    """Output size and aspect for ``grid`` (see the module docstring).

    ``output`` is ``"auto"``, a long side in pixels or an explicit ``(height, width)``.
    ``inset`` is the grid-level inset of ``dewarp.predict.dense_backward_map``: the output
    shows ``1 - e`` of the page extent per axis, which the size follows.
    """
    if not (_positive_int(max_side) and _positive_int(max_pixels)):
        raise _option_error("max_side and max_pixels must be positive integers")
    ex, ey = _inset(inset)
    rows, cols = arc_lengths(grid, source_size)
    if not rows.max() > 0 or not cols.mean() > 0:
        raise _degenerate("grid rows or columns have zero length")
    estimate, method = estimate_aspect(grid, source_size, focal_px=focal_px)
    record = {
        "aspect_estimate": estimate,
        "aspect_method": method,
        "density_width": float(rows.max()),
    }
    if isinstance(output, (tuple, list)):
        height, width = _size(output, "output")
        density = sampling_density(grid, source_size, (height, width), inset=(ex, ey))
        return OutputPlan(
            height=height,
            width=width,
            aspect=width / height,
            aspect_source="output_size",
            size_source="output_size",
            density_min=density[0],
            density_median=density[1],
            **record,
        )
    page_aspect, aspect_source, paper = resolve_aspect(aspect, estimate)
    visible = page_aspect * (1.0 - ex) / (1.0 - ey)
    if isinstance(output, str) and output == "auto":
        size_source, anchor = "auto", "width"
        width_c = (1.0 - ex) * float(rows.max())
        height_c = width_c / visible
    elif _positive_int(output):
        size_source = "long_side"
        if visible <= 1.0:
            anchor, height_c, width_c = "height", float(output), float(output) * visible
        else:
            anchor, width_c, height_c = "width", float(output), float(output) / visible
    else:
        raise _option_error("output must be 'auto', a long side or a (height, width) pair")
    height, width, scale, limited_by = _fit(
        width_c, height_c, visible, anchor, max_side, max_pixels
    )
    density = sampling_density(grid, source_size, (height, width), inset=(ex, ey))
    return OutputPlan(
        height=height,
        width=width,
        aspect=page_aspect,
        aspect_source=aspect_source,
        size_source=size_source,
        paper=paper,
        scale=scale,
        limited_by=limited_by,
        density_min=density[0],
        density_median=density[1],
        **record,
    )
