"""Where each model view lies in the source image, and quarter turns of result grids.

Values are flattener values of an image (``docs/INTERFACE_SPEC.md``): +-1 are its outer
pixel edges, and value q of the source is the pixel-edge point
``p = ((q_x + 1) W / 2, (q_y + 1) H / 2)``. A view is the image a model sees before it is
resized to the model input. A ``Placement`` maps view values q' to source values q:

    q = A q' + c
    A = diag(2/W, 2/H) R(phi) diag(w_c / 2, h_c / 2)
    c = 2 center / (W, H) - 1        (for a crop box: ((l + r) / W - 1, (t + b) / H - 1))

R(phi) turns x toward y, so a positive angle is a clockwise turn on screen, the sign of
``dewarp.predict.estimate_page_angle``; the view shows the source turned by -phi, and a page
turned clockwise by phi appears upright in it. (h_c, w_c) is the view size in source pixels:
``canvas_size`` of the crop box (the whole box fits), or the page's own extent in the turned
frame (``Placement.from_outline``, the tight view). ``split_angle`` writes phi as
theta + 90 k with theta in [-45, 45]; the k quarter turns are exact axis swaps and sign
changes of the view values, and an odd k swaps the view's height and width. For
|phi| <= 45 a crop placement is ``canvas_to_source`` itself, so it places a pass exactly as
``crop_pass`` does.

Each pass is placed in the source directly, ``G = compose(g, placement)``, in float64 per
grid node, and the next pass takes its box and angle from G; the final map samples the
original image once. Densifying a grid (``dense_map``) takes affine combinations of the
nodes (the weights sum to 1), so composing the nodes and composing the dense map agree.

``Placement.turned(k)`` turns a view before the model sees it; a 90 or 270 degree page is
set upright that way with one more prediction. A result that shows its page turned k
quarter turns clockwise is set upright after the prediction by ``quarter_turn_grid(G, k)``:
the rendered image becomes ``np.rot90`` of the old one. A 180-degree result is set upright
by ``flip_grid(G)`` (``G[::-1, ::-1]``). Both only re-index nodes.

The conventions stay apart. A legacy ``LoadedPredictor`` map is expanded in view values
first (``coordinates.resize_backward_map``) and composed after; ``Placement.from_box`` is
``regions.geometry.region_map_to_source`` bit for bit. A UVDoc-convention grid is converted
to flattener values with the view size (``view_values``), as ``crop_pass`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import FlattenerError, Status
from ..dewarp.coords import _apply, _grid, _size, center_positions, interpolation_matrix
from ..dewarp.predict import (
    RERUN_MARGIN,
    Inset,
    _flattener_values,
    _grid_index,
    canvas_size,
    canvas_to_source,
    resolve_inset,
)
from ..regions.geometry import PageRegion

KINDS = ("whole", "box", "canvas", "outline")
CONVENTIONS = ("flattener", "uvdoc")

# R(90 k): view value q' of the turned view is R(90 k) q' in the view before the turn.
_QUARTER_MATRICES = (
    np.array([[1.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, -1.0], [1.0, 0.0]]),
    np.array([[-1.0, 0.0], [0.0, -1.0]]),
    np.array([[0.0, 1.0], [-1.0, 0.0]]),
)


def _error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_placement", message)


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.floating)):
        raise _error(f"{name} must be a finite real number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise _error(f"{name} must be a finite real number, got {value!r}")
    return number


def _pixels(size: Any, name: str) -> tuple[int, int]:
    try:
        return _size(size, name)
    except ValueError as error:
        raise _error(str(error)) from None


def _quarter(k: Any) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise _error(f"quarter turns must be an integer, got {k!r}")
    return int(k) % 4


def _box(box: Any) -> tuple[float, float, float, float]:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise _error(f"box must be (left, top, right, bottom), got {box!r}")
    left, top, right, bottom = (_real(v, "box edge") for v in box)
    if not (right > left and bottom > top):
        raise _error(f"box must have positive width and height, got {box!r}")
    return left, top, right, bottom


def _extent(size: Any, name: str) -> tuple[float, float]:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise _error(f"{name} must be (height, width), got {size!r}")
    height, width = (_real(v, name) for v in size)
    if not (height > 0 and width > 0):
        raise _error(f"{name} must be positive, got {size!r}")
    return height, width


def _values(values: Any, name: str = "values") -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 2 or not np.issubdtype(array.dtype, np.floating):
        raise _error(f"{name} must be a floating (..., 2) array in (x, y) order")
    return array.astype(np.float64, copy=False)


def _turn_values(values: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """``R(90 k) q'`` for (..., 2) values: swaps and sign changes only, so exact."""
    if k == 0:
        return values
    x, y = values[..., 0], values[..., 1]
    turned = ((-y, x), (-x, -y), (y, -x))[k - 1]
    return np.stack(turned, axis=-1)


def _rotation(angle_deg: float) -> NDArray[np.float64]:
    """R(angle), built as ``canvas_to_source`` builds it."""
    theta = np.radians(angle_deg)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def split_angle(angle_deg: float) -> tuple[float, int]:
    """``(theta, k)`` with ``angle = theta + 90 k`` (mod 360), theta in [-45, 45] and k in
    0..3. An angle of at most 45 degrees either way comes back unchanged with k = 0."""
    angle = _real(angle_deg, "angle_deg")
    turns = round(angle / 90.0)
    if turns == 0:
        return angle, 0
    return angle - 90.0 * turns, turns % 4


def turned_size(size: tuple[Any, Any], k: int) -> tuple[Any, Any]:
    """``size`` (height, width) after k quarter turns: swapped for odd k."""
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise _error(f"size must be (height, width), got {size!r}")
    height, width = size
    return (width, height) if _quarter(k) % 2 else (height, width)


def view_canvas_size(box: tuple[float, float, float, float], angle_deg: float) -> tuple:
    """``canvas_size(box, angle_deg)`` with exact quarter turns: the canvas of theta turned
    by k (``split_angle``). Up to 45 degrees it is ``canvas_size`` itself."""
    theta, k = split_angle(angle_deg)
    return turned_size(canvas_size(_box(box), theta), k)


@dataclass(frozen=True, eq=False)
class Placement:
    """The affine map of one view into the source (see the module docstring).

    ``base_matrix`` and ``offset`` are A and c of the view before its ``quarter`` turn k,
    whose angle is ``theta_deg``; ``matrix`` and ``angle_deg`` include the turn.
    ``view_size`` is the (height, width) of the turned view in source pixels. A box
    placement keeps its ``edges`` (left, top, right, bottom) and is applied edge first,
    as ``region_map_to_source``; the others as ``crop_pass``, ``q' A^T + c``.
    """

    kind: str
    source_size: tuple[int, int]
    view_size: tuple[float, float]
    theta_deg: float
    base_matrix: NDArray[np.float64]
    offset: NDArray[np.float64]
    quarter: int = 0
    edges: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        for name, shape in (("base_matrix", (2, 2)), ("offset", (2,))):
            array = np.array(getattr(self, name), dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise _error(f"{name} must be finite with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.kind not in KINDS:
            raise _error(f"kind must be one of {KINDS}, got {self.kind!r}")
        if (self.kind == "box") != (self.edges is not None):
            raise _error("edges belong to box placements only")
        if abs(np.linalg.det(self.base_matrix)) == 0.0:
            raise _error("placement matrix is singular")

    @classmethod
    def whole(cls, source_size: tuple[int, int]) -> Placement:
        """The whole source as the view: ``compose`` returns values equal to its input."""
        height, width = _pixels(source_size, "source_size")
        return cls("whole", (height, width), (float(height), float(width)), 0.0, np.eye(2), (0, 0))

    @classmethod
    def from_box(cls, box: PageRegion | tuple, source_size: tuple[int, int]) -> Placement:
        """The axis-aligned ``box`` (a ``PageRegion`` or left, top, right, bottom in source
        pixel edges; it may extend past the image) as the view: view value -1 is its left or
        top edge and +1 its right or bottom edge. ``compose`` gives what
        ``region_map_to_source`` gives, bit for bit."""
        region = box if isinstance(box, PageRegion) else PageRegion(*_box(box))
        height, width = _pixels(source_size, "source_size")
        matrix = np.diag([region.width / width, region.height / height])
        offset = (
            (region.left + region.right) / width - 1,
            (region.top + region.bottom) / height - 1,
        )
        edges = (region.left, region.top, region.right, region.bottom)
        view = (region.height, region.width)
        return cls("box", (height, width), view, 0.0, matrix, offset, 0, edges)

    @classmethod
    def from_canvas(
        cls,
        box: tuple[float, float, float, float],
        angle_deg: float,
        source_size: tuple[int, int],
        *,
        canvas: tuple[float, float] | None = None,
    ) -> Placement:
        """The crop ``box`` (left, top, right, bottom in source pixels) turned by
        -``angle_deg`` on a canvas that holds all of it, centered on the box: the view of a
        ``crop_pass`` pass. ``canvas`` is the (height, width) of the turned view in source
        pixels, by default ``view_canvas_size(box, angle_deg)``. The placement is
        ``canvas_to_source`` at theta turned by k (``split_angle``); for |angle| <= 45 it is
        ``canvas_to_source(box, angle_deg, canvas, source_size)`` itself."""
        size = _pixels(source_size, "source_size")
        box = _box(box)
        theta, k = split_angle(angle_deg)
        view = view_canvas_size(box, angle_deg) if canvas is None else _extent(canvas, "canvas")
        matrix, offset = canvas_to_source(box, theta, turned_size(view, k), size)
        return cls("canvas", size, view, theta, matrix, offset, k)

    @classmethod
    def from_outline(
        cls,
        points: NDArray,
        angle_deg: float,
        source_size: tuple[int, int],
        *,
        margin: float = RERUN_MARGIN,
    ) -> Placement:
        """The tight view of outline ``points`` (N, 2 source pixel-edge points, e.g.
        ``outline_points`` of a grid) at ``angle_deg``: the points' bounds in the frame
        turned by theta, widened by ``margin`` of their extent per side, then turned by k
        (``split_angle``). Its center and extent are real numbers, and it may reach past the
        image (view sampling repeats border pixels there)."""
        size = _pixels(source_size, "source_size")
        points = _values(points, "points")
        if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise _error("points must be finite (N >= 2, 2) source pixel-edge points")
        margin = _real(margin, "margin")
        if margin < 0:
            raise _error(f"margin must be >= 0, got {margin}")
        theta, k = split_angle(angle_deg)
        rotation = _rotation(theta)
        frame = points @ rotation  # R^T p: coordinates along the axes of the view
        low, high = frame.min(axis=0), frame.max(axis=0)
        extent = high - low
        if not (extent > 0).all():
            raise _error("outline points span no area in the view frame")
        half = extent * (0.5 + margin)
        center = rotation @ ((low + high) / 2)
        height, width = size
        matrix = np.diag([2.0 / width, 2.0 / height]) @ rotation @ np.diag(half)
        offset = 2.0 * center / np.array([width, height], dtype=np.float64) - 1.0
        view = turned_size((float(2 * half[1]), float(2 * half[0])), k)
        return cls("outline", size, view, theta, matrix, offset, k)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """A of the turned view: ``base_matrix @ R(90 k)``."""
        return self.base_matrix @ _QUARTER_MATRICES[self.quarter]

    @property
    def angle_deg(self) -> float:
        """phi = theta + 90 k."""
        return self.theta_deg + 90.0 * self.quarter

    def turned(self, k: int) -> Placement:
        """This view turned k more quarter turns: it shows the source turned a further
        -90 k degrees (counterclockwise on screen for k = 1)."""
        k = _quarter(k)
        return replace(
            self, view_size=turned_size(self.view_size, k), quarter=(self.quarter + k) % 4
        )

    def to_view(self, values: NDArray) -> NDArray[np.float64]:
        """View values (float64) of source ``values``: the inverse of ``compose``."""
        local = (_values(values) - self.offset) @ np.linalg.inv(self.base_matrix).T
        return _turn_values(local, (-self.quarter) % 4)

    def view_map(self, size: tuple[int, int]) -> NDArray[np.float32]:
        """Backward map (height, width, 2) float32 of the view sampled at ``size`` pixels:
        the source values of the view's pixel centers. Sampling the source (or any
        resolution of it) with this map gives the view, e.g. the model input."""
        height, width = _pixels(size, "size")
        positions = np.stack(np.meshgrid(center_positions(width), center_positions(height)), -1)
        return compose(positions, self).astype(np.float32)

    def record(self) -> dict[str, Any]:
        """JSON-ready description; ``Placement.from_record`` rebuilds it exactly."""
        return {
            "kind": self.kind,
            "source_size": list(self.source_size),
            "view_size": [float(v) for v in self.view_size],
            "theta_deg": float(self.theta_deg),
            "quarter": self.quarter,
            "angle_deg": float(self.angle_deg),
            "edges": None if self.edges is None else [float(v) for v in self.edges],
            "base_matrix": self.base_matrix.tolist(),
            "offset": self.offset.tolist(),
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Placement:
        """The placement of a ``record``; ``matrix`` and ``angle_deg`` there are derived."""
        try:
            edges = record["edges"]
            return cls(
                record["kind"],
                _pixels(tuple(record["source_size"]), "source_size"),
                _extent(record["view_size"], "view_size"),
                _real(record["theta_deg"], "theta_deg"),
                np.asarray(record["base_matrix"], dtype=np.float64),
                np.asarray(record["offset"], dtype=np.float64),
                _quarter(record["quarter"]),
                None if edges is None else _box(edges),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise _error(f"invalid placement record: {error}") from None


def compose(values: NDArray, placement: Placement) -> NDArray[np.float64]:
    """Source values (float64) of view ``values`` (..., 2): ``q = A q' + c``.

    Grid nodes and dense maps alike; the model grid of a pass placed in the source is
    ``compose(view_values(raw, convention, placement), placement)``."""
    if not isinstance(placement, Placement):
        raise _error("placement must be a Placement")
    local = _turn_values(_values(values), placement.quarter)
    if placement.edges is None:
        return local @ placement.base_matrix.T + placement.offset
    # region_map_to_source's arithmetic, operation for operation.
    left, top, right, bottom = placement.edges
    height, width = placement.source_size
    result = local.copy()
    result[..., 0] = 2 * (left + (result[..., 0] + 1) * (right - left) / 2) / width - 1
    result[..., 1] = 2 * (top + (result[..., 1] + 1) * (bottom - top) / 2) / height - 1
    return result


def view_values(raw: NDArray, convention: str, placement: Placement) -> NDArray[np.float64]:
    """Flattener values (float64) of the view for a raw model grid: ``uvdoc``
    (align_corners=True) values are converted with the view size in source pixels, as
    ``crop_pass`` converts them."""
    if convention not in CONVENTIONS:
        raise _error(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    if not isinstance(placement, Placement):
        raise _error("placement must be a Placement")
    return _flattener_values(_values(raw, "raw"), convention, placement.view_size)


def quarter_turn_grid(grid: NDArray, k: int) -> NDArray:
    """``grid`` (Gh, Gw, C) with its nodes re-indexed by ``np.rot90(grid, k)``. Rendered at
    ``turned_size(output_size, k)``, it gives ``np.rot90`` of the image of ``grid``: turned
    counterclockwise by k quarter turns on screen, which sets upright a page that image
    shows turned k quarter turns clockwise."""
    return np.rot90(_grid(grid, "grid"), _quarter(k), axes=(0, 1)).copy()


def flip_grid(grid: NDArray) -> NDArray:
    """``grid`` turned by 180 degrees, ``G[::-1, ::-1]``: ``quarter_turn_grid(grid, 2)``."""
    return _grid(grid, "grid")[::-1, ::-1].copy()


def _border(border: float | tuple[float, float]) -> tuple[float, float]:
    values = border if isinstance(border, (tuple, list)) else (border, border)
    if len(values) != 2:
        raise _error(f"border must be a number or (bx, by), got {border!r}")
    bx, by = (_real(v, "border") for v in values)
    if not (0 <= bx < 1 and 0 <= by < 1):
        raise _error(f"border values must be in [0, 1), got {(bx, by)}")
    return bx, by


def dense_map(
    grid: NDArray,
    output_size: tuple[int, int],
    *,
    inset: Inset = None,
    border: float | tuple[float, float] = 0.0,
) -> NDArray[np.float32]:
    """Dense backward map (float32) at ``output_size`` of a grid of source values on
    edge-aligned nodes (a composed grid).

    Output coordinate u in [-1, 1] shows grid position (1 - e + b) u on each axis.
    ``inset`` e (``resolve_inset``) moves each side in by e / 2 of the page extent;
    ``border`` b (one value or (bx, by), each in [0, 1)) moves each side out by b / 2,
    extending the edge cells linearly. Without a border it equals
    ``dense_backward_map(grid, "flattener", output_size, source_size, inset,
    flattener_values=True)`` bit for bit."""
    array = _grid(grid, "grid")
    height, width = _size(output_size, "output_size")
    ex, ey = resolve_inset(inset)
    bx, by = _border(border)
    gh, gw = array.shape[:2]
    rows = interpolation_matrix(
        _grid_index(height, gh, ey - by, "flattener"), gh, extrapolate=by > 0
    )
    cols = interpolation_matrix(
        _grid_index(width, gw, ex - bx, "flattener"), gw, extrapolate=bx > 0
    )
    return _apply(rows, array, cols)
