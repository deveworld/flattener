"""Document scanning with page detection, orientation and dewarping.

Scanner.load reads the model paths in a bundle.
Scanner.plan estimates page geometry; Scanner.render applies a stored plan.
Scanner.scan combines both operations, and Scanner.scan_pages handles book spreads.

ScanOptions controls output size, refinement, preview and fallback behavior.
ScanOverrides supplies a manual region, quad or orientation.
The stage cache retains model outputs for recent images so edits can reuse inference.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..contracts import FlattenerError, RenderConfig, RenderResult, Status
from ..dewarp.predict import (
    DESKEW_MIN_AGREEMENT,
    RERUN_MARGIN,
    TTA_MODES,
    Dewarper,
    Inset,
    estimate_page_angle,
    inset_record,
    load_dewarper,
    outline_points,
    render_unchecked,
    rerun_box,
    resolve_inset,
    sample_rgb,
    source_inside,
    with_tta,
)
from ..regions.context import GateConfig, LoadedContextSegmenter, load_context_segmenter
from ..renderer import render as flattener_render
from ..validation import validate_image
from . import locate as locating
from . import look
from . import sharpen as sharpening
from .download import download_bundle
from .fast_check import FastCheck, fast_check
from .io import SourceImage, read_source
from .orient import Orienter, OrientResult, ViewGeometry
from .output_policy import (
    MAX_PIXELS,
    MAX_SIDE,
    OutputPlan,
    plan_output,
    resolve_aspect,
    round_half_up,
)
from .placement import Placement, dense_map, flip_grid, quarter_turn_grid, split_angle
from .records import (
    Outcome,
    Reason,
    ScanPlan,
    ScanResult,
    Severity,
    SourceInfo,
    collect_guidance,
    decide_outcome,
    json_plain,
    reason,
)
from .timing import StageClock
from .views import VIEW_SOURCES, crop_placement, run_view

BUNDLE_SCHEMA = "flattener-scan-bundle-1"
BUNDLE_KEYS = {
    "dewarp": {"path", "tta"},
    "gate": {"path"},
    "orient": {"path", "threshold"},
}
PIPELINE_NAME = "flattener.scan"
PIPELINE_VERSION = 3
AUTO_PASSES = 1
PASS_CHOICES = ("auto", 0, 1, 2)
ORIENT_MODES = ("model", "none", "oracle")
VIEW_EXTENTS = ("canvas", "tight")
FALLBACKS = ("original", "perspective")
ON_ERROR = ("original", "return", "raise")
BACKENDS = ("torch",)
QUANTIZE_MODES = ("round", "truncate")
VERIFY: dict[str, Any] = {
    "min_coverage": 0.95,  # below this the gate mask reaches past the page: a warning
    "fragment_side_excess": 0.05,  # mask past every side above this: a piece of the page
    # A gate page filling the frame with a grid holding less of it than this: a close-up
    # whose page runs out of the photo (the grid is a piece of it).
    "close_up_min_coverage": 0.45,
    # A plan with more of its output outside the photo than this: the page runs out of
    # the photo there.
    "cut_off_outside": 0.05,
    # Such a page is flattened anyway, with a warning, when at most this share of the
    # output lies outside and its orientation is settled twice (confident before the
    # dewarp, upright on the flat preview); otherwise the original comes back. On cut-off
    # close-ups, orientation alone can have high confidence with the wrong result, and
    # its flat-preview check disagreed.
    "cut_off_flatten_outside": 0.25,
    # The gate's page_cut_off warning (mask along a photo side) stays only for the sides
    # the accepted page comes within this share of the side of: a neighbouring page cut by
    # the frame leaves the mask along a side while the flattened page lies well inside
    "cut_off_edge_gap": 0.01,
    # A grid the gate mask reaches past must look like paper (scan.look.paper).
    "page_max_saturation": 0.28,
    "page_max_dark": 0.35,
    # A zoomed piece of the page (scan.look.text): fewer lines across it than this, lines
    # at least this regular, and ink in the outer band of at least this share of two edges.
    # A pitch longer than 1/8 of the page is a layout (columns, boxes), not text lines.
    "fragment_min_lines": 8,
    "fragment_max_lines": 20,
    "fragment_min_periodicity": 0.4,
    "fragment_min_edges": 0.5,
    # The first crop cut the page: blank paper of the page's colour runs along a crop edge,
    # and the whole-photo pass holds the gate mask and adds at most the page's own area,
    # mostly page-coloured (scan.look on a copy of the photo with this long side).
    "crop_edge_paper": 0.5,
    "continue_max_area": 1.0,
    "continue_min_paper": 0.5,
    "look_photo_long_side": 768,
    "max_drift": 0.04,  # median node drift between passes, fraction of the page diagonal
    # A spread whose preview shows no blank gutter (a picture runs to the fold) is found by
    # its outline: the top or bottom edge bends at the spine by at least this many
    # degrees over two grid columns on each side.
    "fold_min_bend_deg": 8.0,
    "low_resolution_density": 0.5,  # source pixels per output pixel below this warn
    "post_preview_long_side": 768,
}
CACHE_SIZE = 4
PAGE_CHOICES = ("auto", 1, 2)
PAGE_ORDERS = ("ltr", "rtl")


def _option_error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_scan_option", message)


# ----------------------------------------------------------------------------- options


@dataclass(frozen=True)
class ScanOptions:
    """How a scan runs.

    ``passes``: crop passes after initial prediction (``"auto"`` is 1, plus a settle pass when it
    drifts). ``output``: ``"auto"``, a long side or (height, width). ``aspect``:
    ``"auto"``, ``"estimate"``, ``"snap"``, a paper name or width / height. ``inset``: as
    ``dewarp.predict --inset``; ``border`` moves each side out by border / 2 of the page
    extent. ``fast_check`` (the default): check dewarp plans with ``scan.fast_check`` (the
    renderer's decisions from the grid, a small proxy render) and sample the full size in
    ``render``; ``False`` runs the renderer's checks at full size. ``preview``: long side
    of a preview image sampled from the same map (``None``: no preview). ``chain``: try
    the other candidates after ``default`` fails.
    ``use_focal``: take the EXIF focal length into the aspect estimate. ``fill_rgb``: the
    color of output pixels outside the source; the plan keeps it with ``quantize`` and
    ``sharpen``, so a stored plan renders the same image. ``sharpen``: the amount of
    ``scan.sharpen`` on the flattened image (0 turns it off). ``pages`` (for
    ``Scanner.split``): ``"auto"`` splits an open-book spread at its fold (``look.fold``,
    else the bend of its outline), ``2`` always splits (at the middle when no fold is
    found), ``1`` keeps one image.
    ``page_order``: ``"ltr"`` numbers the left page first, ``"rtl"`` the right one.
    """

    passes: str | int = "auto"
    deskew: bool = True
    orient: str = "model"
    output: str | int | tuple[int, int] = "auto"
    aspect: str | float = "auto"
    inset: Inset = None
    border: float = 0.0
    quantize: str = "round"
    fast_check: bool = True
    view_source: str = "original"
    view_extent: str = "canvas"
    fallback: str = "original"
    on_error: str = "original"
    preview: int | None = None
    chain: bool = True
    use_focal: bool = True
    fill_rgb: tuple[int, int, int] = (255, 255, 255)
    sharpen: float = sharpening.DEFAULT_AMOUNT
    pages: str | int = "auto"
    page_order: str = "ltr"

    def __post_init__(self) -> None:
        if self.passes not in PASS_CHOICES or isinstance(self.passes, bool):
            raise _option_error(f"passes must be one of {PASS_CHOICES}, got {self.passes!r}")
        for name, allowed in (
            ("orient", ORIENT_MODES),
            ("quantize", QUANTIZE_MODES),
            ("view_source", VIEW_SOURCES),
            ("view_extent", VIEW_EXTENTS),
            ("fallback", FALLBACKS),
            ("on_error", ON_ERROR),
        ):
            if getattr(self, name) not in allowed:
                raise _option_error(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        for name in ("deskew", "fast_check", "chain", "use_focal"):
            if not isinstance(getattr(self, name), bool):
                raise _option_error(f"{name} must be a bool")
        try:
            sharpening.check_amount(self.sharpen)
        except ValueError as error:
            raise _option_error(str(error)) from None
        try:
            resolve_inset(self.inset)
        except FlattenerError as error:
            raise _option_error(error.message) from None
        if isinstance(self.border, bool) or not isinstance(self.border, (int, float)):
            raise _option_error("border must be a number in [0, 1)")
        if not 0 <= float(self.border) < 1:
            raise _option_error("border must be a number in [0, 1)")
        if self.preview is not None and (
            isinstance(self.preview, bool) or not isinstance(self.preview, int) or self.preview < 2
        ):
            raise _option_error("preview must be a long side of at least 2 pixels or None")
        fill = self.fill_rgb
        if (
            not isinstance(fill, (tuple, list))
            or len(fill) != 3
            or any(
                isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 255
                for v in fill
            )
        ):
            raise _option_error("fill_rgb must be three integers in 0..255")
        _check_output(self.output)
        _check_aspect(self.aspect)
        if isinstance(self.pages, bool) or self.pages not in PAGE_CHOICES:
            raise _option_error(f"pages must be one of {PAGE_CHOICES}, got {self.pages!r}")
        if self.page_order not in PAGE_ORDERS:
            raise _option_error(f"page_order must be one of {PAGE_ORDERS}")

    @property
    def rerun(self) -> int:
        return AUTO_PASSES if self.passes == "auto" else int(self.passes)

    def record(self) -> dict[str, Any]:
        data = asdict(self)
        data["inset"] = list(resolve_inset(self.inset))
        return json_plain(data)


def _check_output(output: Any) -> None:
    """``output`` is ``"auto"``, a long side of at least 2 or a (height, width) of at least
    2 per side within the renderer's limits (``MAX_SIDE``, ``MAX_PIXELS``)."""
    if isinstance(output, str) and output == "auto":
        return
    if isinstance(output, (int, np.integer)) and not isinstance(output, bool) and output >= 2:
        return
    if (
        isinstance(output, (tuple, list))
        and len(output) == 2
        and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in output)
        and min(output) >= 2
    ):
        height, width = (int(v) for v in output)
        if max(height, width) > MAX_SIDE or height * width > MAX_PIXELS:
            raise _option_error(
                f"output {height}x{width} is above the limits ({MAX_SIDE} pixels a side, "
                f"{MAX_PIXELS} pixels)"
            )
        return
    raise _option_error("output must be 'auto', a long side >= 2 or (height, width)")


def _check_aspect(aspect: Any) -> None:
    """``aspect`` is a mode, a paper name or a positive number (``resolve_aspect``)."""
    try:
        resolve_aspect(aspect, 1.0)
    except FlattenerError as error:
        raise _option_error(error.message) from None


@dataclass(frozen=True)
class ScanOverrides:
    """User corrections, each recorded as ``manual`` provenance.

    ``region``: a ``PageRegion`` or (left, top, right, bottom) in source pixels; ``quad``:
    four (x, y) page corners in reading order (top-left, top-right, bottom-right,
    bottom-left); ``quarter``: clockwise quarter turns of the page's text-up direction;
    ``skip_dewarp``: flatten the region or quad by a homography only; ``output_size`` and
    ``aspect``: as ``ScanOptions.output`` and ``aspect`` (``"auto"`` is no correction).
    """

    region: Any = None
    quad: Any = None
    quarter: int | None = None
    skip_dewarp: bool = False
    output_size: str | int | tuple[int, int] | None = None
    aspect: str | float | None = None

    def __post_init__(self) -> None:
        if self.region is not None and self.quad is not None:
            raise _option_error("give either a region or a quad")
        if self.quarter is not None and (
            isinstance(self.quarter, bool)
            or not isinstance(self.quarter, (int, np.integer))
            or not 0 <= self.quarter <= 3
        ):
            raise _option_error("quarter must be an integer in 0..3")
        if not isinstance(self.skip_dewarp, bool):
            raise _option_error("skip_dewarp must be a bool")
        for name in ("output_size", "aspect"):
            value = getattr(self, name)
            if isinstance(value, str) and value == "auto":
                object.__setattr__(self, name, None)
        if self.output_size is not None:
            _check_output(self.output_size)
        if self.aspect is not None:
            _check_aspect(self.aspect)

    def record(self) -> dict[str, Any]:
        region = self.region
        if region is not None and not isinstance(region, (tuple, list)):
            region = [region.left, region.top, region.right, region.bottom]
        quad = None if self.quad is None else np.asarray(self.quad, dtype=np.float64).tolist()
        return json_plain(
            {
                "region": region,
                "quad": quad,
                "quarter": self.quarter,
                "skip_dewarp": self.skip_dewarp,
                "output_size": self.output_size,
                "aspect": self.aspect,
            }
        )


# ----------------------------------------------------------------------------- cache


class StageCache:
    """Model outputs for recent images, compared directly by their pixels."""

    def __init__(self, size: int = CACHE_SIZE) -> None:
        self.size = max(0, int(size))
        self._entries: list[tuple[NDArray, dict[Any, Any]]] = []

    def entry(self, image: NDArray) -> dict[Any, Any]:
        for index, (stored, entry) in enumerate(self._entries):
            if np.array_equal(stored, image):
                self._entries.append(self._entries.pop(index))
                return entry
        entry: dict[Any, Any] = {}
        if self.size:
            self._entries.append((image.copy(), entry))
            self._entries = self._entries[-self.size :]
        return entry

    def clear(self) -> None:
        self._entries.clear()


class _Memo:
    """Lookups in one input's cache entry, counted per stage."""

    def __init__(self, entry: dict[Any, Any]) -> None:
        self.entry = entry
        self.hits: dict[str, int] = {}
        self.misses: dict[str, int] = {}

    def get(self, stage: str, key: Any, compute):
        full = (stage, key)
        if full in self.entry:
            self.hits[stage] = self.hits.get(stage, 0) + 1
            return self.entry[full]
        self.misses[stage] = self.misses.get(stage, 0) + 1
        value = compute()
        self.entry[full] = value
        return value

    def record(self) -> dict[str, Any]:
        return {"hits": dict(self.hits), "misses": dict(self.misses)}


# ----------------------------------------------------------------------------- records


@dataclass(eq=False)
class _Checked:
    """The check of one plan: a ``RenderResult`` of ``flattener.render`` (or the fast
    check), and the map it checked."""

    plan: ScanPlan
    result: RenderResult
    fast: FastCheck | None = None

    @property
    def ok(self) -> bool:
        return self.result.status == Status.SUCCESS


@dataclass(eq=False)
class PlannedScan:
    """What ``Scanner.plan`` found: the plan (``None`` when the scan falls back to the
    original), its preview, the record so far and the checked render it can reuse.
    ``plan, preview = scanner.plan(...)`` unpacks the first two."""

    plan: ScanPlan | None
    preview: NDArray[np.uint8] | None
    source: SourceImage
    record: ScanResult
    options: ScanOptions
    overrides: ScanOverrides = field(default_factory=lambda: ScanOverrides())
    checked: _Checked | None = None
    rejected: list[ScanPlan] = field(default_factory=list)
    clock: StageClock | None = None

    def __iter__(self) -> Iterator[Any]:
        yield self.plan
        yield self.preview


@dataclass(eq=False)
class _Orientation:
    quarter: int
    method: str  # model, prior, none, oracle, manual, quad
    provenance: str
    result: OrientResult | None = None
    reasons: list[Reason] = field(default_factory=list)
    post: dict[str, Any] | None = None
    settled_by: str | None = None  # the candidate whose post check accepted the quarter

    def record(self) -> dict[str, Any]:
        return {
            "quarter": self.quarter,
            "method": self.method,
            "provenance": self.provenance,
            "pre": None if self.result is None else self.result.to_dict(),
            "post": self.post,
            "settled_by": self.settled_by,
        }


def _post_accepted(post: dict[str, Any] | None) -> bool:
    return bool(((post or {}).get("result") or {}).get("accepted"))


def _orientation_confirmed(orientation: _Orientation) -> bool:
    """OrientNet settled the quarter twice: accepted before the dewarp and the flat
    preview of the result accepted as upright (no turn left)."""
    pre = orientation.result
    post = (orientation.post or {}).get("result") or {}
    return pre is not None and pre.accepted and bool(post.get("accepted")) and post.get("k") == 0


def _edge_gaps(grid: NDArray) -> dict[str, float]:
    """How far the grid stays inside each image side, as a share of that side (negative:
    the page reaches past it)."""
    x, y = grid[..., 0], grid[..., 1]
    return {
        "top": float(y.min() + 1) / 2,
        "right": float(1 - x.max()) / 2,
        "bottom": float(1 - y.max()) / 2,
        "left": float(x.min() + 1) / 2,
    }


def _cut_off_on_page(reasons: list[Reason], grid: NDArray) -> list[Reason]:
    """``reasons`` with the gate's ``page_cut_off`` warning (evidence ``mask`` or
    ``mask_and_outline``) narrowed to the sides the accepted page reaches
    (``VERIFY["cut_off_edge_gap"]``; evidence gains ``_and_page`` and the gaps). A warning
    with no such side is dropped: the page itself is whole. A split page narrows its
    spread's warning again with its own grid."""
    gaps = _edge_gaps(grid)
    kept = []
    for item in reasons:
        evidence = str(item.detail.get("evidence")).removesuffix("_and_page")
        if item.code != "page_cut_off" or evidence not in ("mask", "mask_and_outline"):
            kept.append(item)
            continue
        sides = [s for s in item.detail.get("sides", []) if gaps[s] <= VERIFY["cut_off_edge_gap"]]
        if sides:
            gap = {side: round(gaps[side], 4) for side in sides}
            kept.append(
                reason("page_cut_off", sides=sides, evidence=f"{evidence}_and_page", gap=gap)
            )
    return kept


def _outside_sides(grid: NDArray) -> list[str]:
    """Image sides past which the grid's outline reaches (the page runs out of the photo)."""
    beyond = {
        "top": grid[..., 1].min() < -1,
        "right": grid[..., 0].max() > 1,
        "bottom": grid[..., 1].max() > 1,
        "left": grid[..., 0].min() < -1,
    }
    return [side for side, out in beyond.items() if out]


@dataclass(eq=False)
class _Candidate:
    name: str
    box: tuple[int, int, int, int] | None  # None: the whole photo
    theta: float
    deskew: bool
    orient: bool
    method: str
    rerun: int
    # How the quarter turn k is applied: "view" turns the view before the model sees it;
    # "nodes" re-indexes the grids of the unturned view (quarter_turn_grid).
    turn: str = "view"

    def key(self, quarter: int, source_size: tuple[int, int]) -> tuple:
        whole = (0, 0, source_size[1], source_size[0])
        box = None if self.box is None or tuple(self.box) == whole else tuple(self.box)
        theta = self.theta if self.deskew else 0.0
        turn = quarter if self.orient else 0
        mode = self.turn if turn else None
        return (box, theta, self.deskew if self.rerun else None, turn, self.rerun, mode)

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "box": None if self.box is None else list(self.box),
            "angle_deg": self.theta,
            "deskew": self.deskew,
            "orient": self.orient,
            "turn": self.turn,
            "method": self.method,
            "passes": self.rerun,
        }


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def as_source(source: Any, *, config: RenderConfig | None = None) -> SourceImage:
    """Load a path, SourceImage or RGB array."""
    if isinstance(source, SourceImage):
        return source
    if isinstance(source, (str, os.PathLike)):
        return read_source(source, config=config)
    if isinstance(source, np.ndarray):
        validate_image(source, config or RenderConfig())
        info = SourceInfo(size=tuple(int(v) for v in source.shape[:2]))
        return SourceImage(image=source, info=info)
    raise FlattenerError(
        Status.INPUT_ERROR, "invalid_image", "source must be a path, a SourceImage or an array"
    )


def view_geometry(placement: Placement) -> ViewGeometry:
    """OrientNet's ``ViewGeometry`` of a placed view, one view pixel per source pixel."""
    height, width = placement.source_size
    view_height, view_width = placement.view_size
    scale = np.diag([width / 2.0, height / 2.0])
    matrix = scale @ placement.matrix @ np.diag([2.0 / view_width, 2.0 / view_height])
    offset = scale @ (placement.offset + 1.0 - placement.matrix @ np.ones(2))
    return ViewGeometry((float(view_height), float(view_width)), np.column_stack([matrix, offset]))


def _drifted(passes: list[tuple[NDArray, dict[str, Any]]], source_size: tuple[int, int]) -> bool:
    """Whether the last crop pass moved the page by more than ``VERIFY["max_drift"]``."""
    if len(passes) < 2:
        return False
    first, second = passes[-2][0], passes[-1][0]
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        return False
    return median_drift(first, second, source_size) > VERIFY["max_drift"]


def median_drift(first: NDArray, second: NDArray, source_size: tuple[int, int]) -> float:
    """Median node distance between two grids over the second grid's page diagonal."""
    height, width = source_size
    scale = np.array([width / 2.0, height / 2.0])
    distance = np.hypot(*np.moveaxis((np.asarray(first) - np.asarray(second)) * scale, -1, 0))
    points = outline_points(second, source_size)
    diagonal = float(np.hypot(*(points.max(axis=0) - points.min(axis=0))))
    return float(np.median(distance)) / diagonal if diagonal > 0 else math.inf


def quad_homography(quad: NDArray[np.float64], source_size: tuple[int, int]) -> NDArray:
    """3x3 H with ``H (u, v, 1)`` proportional to the source flattener value of the
    output position (u, v) in [-1, 1]; the corners (-1, -1), (1, -1), (1, 1), (-1, 1)
    go to the quad's top-left, top-right, bottom-right and bottom-left."""
    height, width = source_size
    corners = np.asarray(quad, dtype=np.float64) / np.array([width, height]) * 2.0 - 1.0
    rows, values = [], []
    for (u, v), (x, y) in zip(((-1, -1), (1, -1), (1, 1), (-1, 1)), corners, strict=True):
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y])
        values.extend((x, y))
    try:
        solution = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(values))
    except np.linalg.LinAlgError:
        raise FlattenerError(
            Status.INVALID_TRANSFORM, "degenerate_quad", "the page quad has no homography"
        ) from None
    return np.append(solution, 1.0).reshape(3, 3)


def homography_values(
    homography: NDArray, u: NDArray[np.float64], v: NDArray[np.float64]
) -> NDArray[np.float64]:
    points = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.asarray(homography).T
    return points[..., :2] / points[..., 2:3]


def homography_grid(homography: NDArray, grid_size: tuple[int, int] = (45, 31)) -> NDArray:
    """The homography sampled at edge-aligned grid nodes (for the output policy)."""
    rows, cols = grid_size
    u, v = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    return homography_values(homography, u, v)


def homography_map(
    homography: NDArray,
    output_size: tuple[int, int],
    inset: tuple[float, float] = (0.0, 0.0),
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Dense backward map (float32) of a perspective plan at output pixel centers. As in
    ``placement.dense_map``, output coordinate u shows page position (1 - e + b) u on
    each axis for inset e and border b."""
    height, width = output_size
    ex, ey = inset
    u = ((2.0 * np.arange(width) + 1.0) / width - 1.0) * (1.0 - ex + border)
    v = ((2.0 * np.arange(height) + 1.0) / height - 1.0) * (1.0 - ey + border)
    uu, vv = np.meshgrid(u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        return homography_values(homography, uu, vv).astype(np.float32)


def rect_quad(points: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """Corners (top-left, top-right, bottom-right, bottom-left) of the rectangle at angle
    ``theta`` around ``points`` (source pixels)."""
    angle = math.radians(theta)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    frame = points @ rotation
    low, high = frame.min(axis=0), frame.max(axis=0)
    corners = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
    return corners @ rotation.T


def turn_quad(quad: NDArray[np.float64], quarter: int) -> NDArray[np.float64]:
    """Reading-order corners of a quad whose page text-up turns ``quarter`` quarter turns
    clockwise from the quad's first edge."""
    return np.roll(np.asarray(quad), -int(quarter) % 4, axis=0)


def _dense(plan: ScanPlan, size: tuple[int, int] | None = None) -> NDArray[np.float32]:
    size = plan.output.size if size is None else size
    if plan.kind == "perspective":
        return homography_map(plan.homography, size, plan.inset, plan.border)
    return dense_map(plan.grid, size, inset=plan.inset, border=plan.border, nodes=plan.nodes)


def split_grid(grid: NDArray, low: float, high: float) -> NDArray[np.float64]:
    """The part of an edge-aligned grid between column positions ``low`` and ``high``
    (-1 to 1), resampled linearly onto as many columns."""
    array = np.asarray(grid, dtype=np.float64)
    columns = array.shape[1]
    t = (np.linspace(low, high, columns) + 1.0) / 2.0 * (columns - 1)
    left = np.clip(np.floor(t).astype(int), 0, columns - 2)
    weight = (t - left)[None, :, None]
    return array[:, left] * (1.0 - weight) + array[:, left + 1] * weight


def _outline_fold(plan: ScanPlan, source_size: tuple[int, int]) -> dict[str, Any] | None:
    """The fold of a spread by its outline, for a preview with no blank gutter: the plan is
    wider than tall (``look.FOLD_MIN_ASPECT``) and its top edge bends down or its bottom
    edge up by at least ``VERIFY["fold_min_bend_deg"]`` at a grid column in
    ``look.FOLD_BAND``. ``u`` is the fold's share of the preview width, as ``look.fold``."""
    aspect = plan.output.width / plan.output.height
    if aspect < look.FOLD_MIN_ASPECT:
        return None
    height, width = source_size
    nodes = (np.asarray(plan.grid, dtype=np.float64) + 1.0) / 2.0 * np.array([width, height])
    columns, step = nodes.shape[1], 2
    low, high = (round(share * (columns - 1)) for share in look.FOLD_BAND)
    best = (-math.inf, 0)
    for edge, inward in ((nodes[0], -1.0), (nodes[-1], 1.0)):  # source y runs down
        for j in range(max(low, step), min(high, columns - 1 - step) + 1):
            a, b = edge[j] - edge[j - step], edge[j + step] - edge[j]
            bend = inward * math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a @ b))
            best = max(best, (bend, j))
    if best[0] < VERIFY["fold_min_bend_deg"]:
        return None
    extent = 1.0 - plan.inset[0] + plan.border  # page extent the preview width shows
    x = 2.0 * best[1] / (columns - 1) - 1.0
    return {
        "u": (x / extent + 1.0) / 2.0,
        "aspect": aspect,
        "text_ratio": None,
        "bend_deg": round(best[0], 2),
        "evidence": "outline",
    }


def preview_size(output_size: tuple[int, int], long_side: int) -> tuple[int, int]:
    height, width = output_size
    scale = min(1.0, long_side / max(height, width))
    return max(2, round_half_up(height * scale)), max(2, round_half_up(width * scale))


def _preview(source: SourceImage, plan: ScanPlan, long_side: int) -> NDArray[np.uint8]:
    """The plan's image at ``long_side`` sampled once from a box-reduced source level, with
    the plan's fill color and quantization."""
    size = preview_size(plan.output.size, long_side)
    backward = _dense(plan, size)
    shrink = math.sqrt(plan.output.height * plan.output.width / (size[0] * size[1]))
    density = (plan.output.density_median or 1.0) * shrink  # source pixels per preview pixel
    factor = max(f for f in (1, 2, 4, 8) if f == 1 or f <= density)
    level = source.level(factor)
    return sample_rgb(level, backward, source_inside(backward), plan.fill_rgb, plan.quantize)


def _with_border(
    output: OutputPlan, inset: tuple[float, float], border: float, request: Any
) -> OutputPlan:
    """The output plan with each side moved out by ``border / 2`` of the page extent:
    an auto size grows so the page keeps its pixels, a long side keeps its length, an
    explicit size is kept."""
    if border == 0:
        return output
    ex, ey = inset
    grow_x, grow_y = (1 - ex + border) / (1 - ex), (1 - ey + border) / (1 - ey)
    height, width = output.height, output.width
    if isinstance(request, (tuple, list)):
        new_height, new_width = height, width
    else:
        new_height, new_width = height * grow_y, width * grow_x
        if request != "auto":
            factor = float(request) / max(new_height, new_width)
            new_height, new_width = new_height * factor, new_width * factor
        limit = min(
            1.0,
            MAX_SIDE / max(new_height, new_width),
            math.sqrt(MAX_PIXELS / (new_height * new_width)),
        )
        new_height = max(1, math.floor(new_height * limit))
        new_width = max(1, math.floor(new_width * limit))
    shrink = math.sqrt(height * width / (new_height * new_width) * grow_x * grow_y)

    def scaled(value: float | None) -> float | None:
        return None if value is None else value * shrink

    return replace(
        output,
        height=int(new_height),
        width=int(new_width),
        density_min=scaled(output.density_min),
        density_median=scaled(output.density_median),
    )


def _gate_record(gate: LoadedContextSegmenter) -> dict[str, Any]:
    return {
        "role": "gate",
        "path": gate.model_id,
        "model_version": gate.metadata.get("model_version"),
        "parameters": gate.metadata.get("parameter_count"),
    }


def _dewarp_record(dewarper: Dewarper) -> dict[str, Any]:
    return {
        "role": "dewarp",
        "path": dewarper.info.get("path"),
        "kind": dewarper.info.get("kind"),
        "step": dewarper.info.get("step"),
        "parameters": dewarper.info.get("parameters"),
        "grid_convention": dewarper.config.grid_convention,
        "tta": dewarper.tta,
    }


def _orient_record(orienter: Orienter) -> dict[str, Any]:
    return {
        "role": "orient",
        "path": orienter.info.get("path"),
        "model_id": orienter.info.get("model_id"),
        "temperature": orienter.temperature,
        "threshold": orienter.threshold,
    }


# ----------------------------------------------------------------------------- scanner


class Scanner:
    """A loaded scanner (see the module docstring)."""

    def __init__(
        self,
        dewarper: Dewarper,
        *,
        gate: LoadedContextSegmenter | None = None,
        orienter: Orienter | None = None,
        gate_config: GateConfig | None = None,
        cache_size: int = CACHE_SIZE,
        info: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(dewarper, Dewarper):
            raise _option_error("dewarper must be a loaded Dewarper")
        self.dewarper = dewarper
        self.gate = gate
        self.orienter = orienter
        self.gate_config = gate_config or GateConfig()
        self.cache = StageCache(cache_size)
        self.info = dict(info or {})
        self.models = [_dewarp_record(dewarper)]
        if gate is not None:
            self.models.append(_gate_record(gate))
        if orienter is not None:
            self.models.append(_orient_record(orienter))

    # -- loading

    @classmethod
    def load(
        cls,
        bundle: str | os.PathLike[str] | dict[str, Any] | None = None,
        *,
        device: str = "cpu",
        backend: str = "torch",
        cache_size: int = CACHE_SIZE,
        orient_threshold: float | None = None,
    ) -> Scanner:
        """Load a local bundle or download the default models from Hugging Face.

        A bundle file or dict names ``dewarp`` and optional ``gate`` and ``orient``
        weights. File paths are relative to the bundle. The ``dewarp`` entry can set
        ``tta``; ``orient`` can set ``threshold``, replaced by ``orient_threshold``.
        Downloaded models are cached for later use, including offline scans.
        """
        if backend not in BACKENDS:
            raise FlattenerError(
                Status.INPUT_ERROR,
                "unsupported_backend",
                f"backend must be one of {BACKENDS}; the scanner runs on torch",
            )
        if bundle is None:
            bundle = download_bundle()
        if isinstance(bundle, dict):
            data, base = bundle, Path.cwd()
        else:
            path = Path(bundle).expanduser()
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise FlattenerError(
                    Status.IO_ERROR, "unreadable_bundle", f"{path}: {error}"
                ) from error
            base = path.parent
        if not isinstance(data, dict) or data.get("schema") != BUNDLE_SCHEMA:
            raise FlattenerError(
                Status.INPUT_ERROR, "invalid_bundle", f"bundle schema must be {BUNDLE_SCHEMA}"
            )
        unknown = sorted(set(data) - {"schema", *BUNDLE_KEYS})
        if unknown:
            raise FlattenerError(
                Status.INPUT_ERROR, "invalid_bundle", f"unknown bundle keys: {', '.join(unknown)}"
            )
        paths, entries = {}, {}
        for role, allowed in BUNDLE_KEYS.items():
            entry = data.get(role)
            if entry is None:
                continue
            if isinstance(entry, str):
                entry = {"path": entry}
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise FlattenerError(
                    Status.INPUT_ERROR, "invalid_bundle", f"{role} must be a path or {{path}}"
                )
            extra = sorted(set(entry) - allowed)
            if extra:
                raise FlattenerError(
                    Status.INPUT_ERROR,
                    "invalid_bundle",
                    f"unknown keys in {role}: {', '.join(extra)}",
                )
            entries[role] = entry
            resolved = Path(entry["path"]).expanduser()
            resolved = resolved if resolved.is_absolute() else base / resolved
            paths[role] = resolved
        if "dewarp" not in paths:
            raise FlattenerError(Status.INPUT_ERROR, "invalid_bundle", "bundle needs dewarp")
        if orient_threshold is None:
            orient_threshold = entries.get("orient", {}).get("threshold")
        tta = entries["dewarp"].get("tta", "none")
        if tta not in TTA_MODES:
            raise FlattenerError(
                Status.INPUT_ERROR, "invalid_bundle", f"dewarp tta must be one of {TTA_MODES}"
            )
        return cls.from_files(
            paths["dewarp"],
            gate=paths.get("gate"),
            orient=paths.get("orient"),
            orient_threshold=orient_threshold,
            device=device,
            cache_size=cache_size,
            info={"bundle": None if isinstance(bundle, dict) else str(bundle)},
            tta=tta,
        )

    @classmethod
    def from_files(
        cls,
        weights: str | os.PathLike[str],
        *,
        gate: str | os.PathLike[str] | None = None,
        orient: str | os.PathLike[str] | None = None,
        orient_threshold: float | None = None,
        device: str = "cpu",
        cache_size: int = CACHE_SIZE,
        info: dict[str, Any] | None = None,
        tta: str = "none",
    ) -> Scanner:
        """Validate and load the dewarper, page detector and orientation model.

        ``tta`` selects test-time augmentation; ``orient_threshold`` sets the minimum
        confidence for orientation correction.
        """
        started = time.perf_counter()
        dewarper = with_tta(load_dewarper(weights, device=device), tta)
        segmenter = None
        if gate is not None:
            segmenter = load_context_segmenter(gate, device=device)
        orienter = None
        if orient is None and orient_threshold is not None:
            raise FlattenerError(
                Status.INPUT_ERROR,
                "missing_orient_weights",
                "an OrientNet threshold needs OrientNet weights",
            )
        if orient is not None:
            if orient_threshold is not None and (
                isinstance(orient_threshold, bool) or not isinstance(orient_threshold, (int, float))
            ):
                raise FlattenerError(
                    Status.INPUT_ERROR, "invalid_bundle", "the orient threshold must be a number"
                )
            orienter = Orienter.load(orient, device=device, threshold=orient_threshold)
        load = {"seconds": time.perf_counter() - started, "device": device}
        return cls(
            dewarper,
            gate=segmenter,
            orienter=orienter,
            cache_size=cache_size,
            info={**(info or {}), "load": load},
        )

    def read(self, path: str | os.PathLike[str]) -> SourceImage:
        """Decode a file once for repeated ``plan`` / ``render`` calls."""
        return read_source(path)

    def pipeline_record(self, options: ScanOptions | None = None) -> dict[str, Any]:
        """The ``predictions.json`` pipeline identity of scans with ``options``. Its
        ``protocol`` holds the quantization, inset and passes that ``rectify_batch``
        records as the labels of the folder."""
        options = options or ScanOptions()
        settings = options.record()
        for name in ("on_error", "preview"):
            settings.pop(name, None)
        passes = options.rerun
        protocol = {
            "quantize": options.quantize,
            "inset": inset_record(options.inset),
            "rerun": {
                "passes": passes,
                "deskew": options.deskew,
                "margin": RERUN_MARGIN if passes else None,
                "max_angle_deg": None,  # the quarter turn is in the view, so no limit
                "min_agreement": DESKEW_MIN_AGREEMENT if options.deskew and passes else None,
                "settle": options.passes == "auto" and passes > 0,
            },
        }
        return {
            "name": PIPELINE_NAME,
            "version": PIPELINE_VERSION,
            "options": settings,
            "protocol": protocol,
            "models": [
                {key: m[key] for key in ("role", "path", "tta") if key in m} for m in self.models
            ],
        }

    def batch_kwargs(
        self, options: ScanOptions | None = None, *, force_render: bool = False
    ) -> dict[str, Any]:
        """``rectify_batch`` keyword arguments for this scanner:
        ``rectify_batch(scanner.dewarper, inputs, out, **scanner.batch_kwargs(options))``.
        The recorded quantization, inset and passes are those of ``options``."""
        options = options or ScanOptions()
        record = self.pipeline_record(options)
        rerun = record["protocol"]["rerun"]
        return {
            "rectifier": self.rectifier(options, force_render=force_render),
            "pipeline": record,
            "force_render": force_render,
            "quantize": options.quantize,
            "inset": options.inset,
            "rerun": rerun["passes"],
            "deskew": rerun["deskew"],
        }

    # -- public stages

    def plan(
        self,
        source: Any,
        options: ScanOptions | None = None,
        overrides: ScanOverrides | None = None,
    ) -> PlannedScan:
        """Run the stages up to the checked plan (see the module docstring)."""
        options = options or ScanOptions()
        overrides = overrides or ScanOverrides()
        clock = StageClock(cuda_sync=self.dewarper.device.type == "cuda")
        with clock.stage("read"):
            src = as_source(source)
        return self._plan(src, options, overrides, clock)

    def render(
        self,
        source: Any,
        planned: PlannedScan | ScanPlan,
        options: ScanOptions | None = None,
    ) -> ScanResult:
        """The result of a plan: the checked render of ``Scanner.plan`` when it is at
        hand, else one ``flattener.render`` of the plan. A ``ScanPlan`` read back from
        JSON must belong to this source (a plan made from a file renders from that file),
        and the result carries the plan's model records, ship verdict and evidence."""
        if isinstance(planned, ScanPlan):
            options = options or ScanOptions()
            clock = StageClock(cuda_sync=self.dewarper.device.type == "cuda")
            with clock.stage("read"):
                src = as_source(source)
            planned = self._planned_from_plan(src, planned, options, clock)
        elif not isinstance(planned, PlannedScan):
            raise _option_error("planned must be a PlannedScan or a ScanPlan")
        return self._finish(planned)

    def scan(
        self,
        source: Any,
        options: ScanOptions | None = None,
        overrides: ScanOverrides | None = None,
    ) -> ScanResult:
        """``render(plan(...))``."""
        planned = self.plan(source, options, overrides)
        return self._finish(planned)

    def scan_pages(
        self,
        source: Any,
        options: ScanOptions | None = None,
        overrides: ScanOverrides | None = None,
    ) -> list[ScanResult]:
        """The results of ``split(plan(...))``: one, or two for an open-book spread."""
        return [self._finish(page) for page in self.split(self.plan(source, options, overrides))]

    def split(self, planned: PlannedScan) -> list[PlannedScan]:
        """The pages of a planned scan (``ScanOptions.pages``). A spread splits at its fold
        on the flat preview (``look.fold``), or, when the preview shows no blank gutter, at
        the bend of its outline (``_outline_fold``), into two plans over the two halves of
        the grid, each sized, checked and recorded on its own (reason ``two_pages``); a
        page whose check fails falls back alone. Other scans, perspective plans and grids
        with ``uvdoc`` nodes stay one."""
        options, plan = planned.options, planned.plan
        if options.pages == 1 or plan is None or plan.kind != "dewarp":
            return [planned]
        if plan.nodes != "flattener":
            return [planned]
        src, clock = planned.source, planned.clock
        with clock.stage("verify"):
            found = look.fold(_preview(src, plan, look.SIZE))
            if found is None:
                found = _outline_fold(plan, src.info.size)
        if found is None:
            if options.pages == "auto":
                return [planned]
            found = {"u": 0.5, "aspect": None, "text_ratio": None}
        cut = (1.0 - plan.inset[0] + plan.border) * (2.0 * found["u"] - 1.0)
        halves = [(-1.0, cut), (cut, 1.0)]
        if options.page_order == "rtl":
            halves.reverse()
        return [
            self._page(planned, split_grid(plan.grid, low, high), number, found)
            for number, (low, high) in enumerate(halves, start=1)
        ]

    def _reflatten(
        self, planned: PlannedScan, grid: NDArray, number: int
    ) -> tuple[NDArray | None, dict[str, Any]]:
        """A split page run through the model again on a crop of its own, so the fold's
        curl is flattened as the page's edge. The new grid counts only while its median
        node drift from the split grid stays within ``VERIFY["max_drift"]``: a grid that
        runs into the facing page moves a whole side."""
        src, options, clock, plan = planned.source, planned.options, planned.clock, planned.plan
        size = src.info.size
        box = locating.outline_box(grid, size)
        if box is None:
            return None, {"status": "no_box"}
        candidate = _Candidate(f"page_{number}", box, plan.angle_deg, options.deskew, True,
                               "split", 0)  # fmt: skip
        with clock.stage("infer"):
            passes = self._infer(src, self._memo(src), candidate, plan.quarter, options, clock)
        refined = passes[-1][0]
        if not np.isfinite(refined).all():
            return None, {"status": "non_finite"}
        drift = median_drift(grid, refined, size)
        if drift > VERIFY["max_drift"]:
            return None, {"status": "kept_split_grid", "drift": round(drift, 4)}
        return refined, {"status": "used", "drift": round(drift, 4)}

    def _page(
        self, planned: PlannedScan, grid: NDArray, number: int, found: dict[str, Any]
    ) -> PlannedScan:
        """One page of a split spread: its plan, checked render and record. The page is
        flattened again on its own crop (``_reflatten``) when that stays on the page."""
        src, options, overrides, clock = (
            planned.source, planned.options, planned.overrides, planned.clock
        )  # fmt: skip
        plan = planned.plan
        refined, reflatten = self._reflatten(planned, grid, number)
        if refined is not None:
            grid = refined
        request = overrides.output_size if overrides.output_size is not None else options.output
        aspect = overrides.aspect if overrides.aspect is not None else options.aspect
        focal = src.info.focal_px() if options.use_focal else None
        with clock.stage("plan"):
            output = plan_output(
                grid, src.info.size, output=request, aspect=aspect, focal_px=focal,
                inset=plan.inset,
            )  # fmt: skip
            output = _with_border(output, plan.inset, plan.border, request)
            page_plan = replace(
                plan, grid=grid, output=output, region={**plan.region, "page": number}
            )
        record = copy.copy(planned.record)
        record.metadata = dict(planned.record.metadata)
        detail = {"page": number, "of": 2, "fold": round(float(found["u"]), 4)}
        record.metadata["page"] = {
            **detail,
            "order": options.page_order,
            "fold_found": found,
            "reflatten": reflatten,
        }
        record.reasons = _cut_off_on_page(
            [*planned.record.reasons, reason("two_pages", **detail)], grid
        )
        record.guidance = collect_guidance(record.reasons)
        record.outcome = decide_outcome(page_plan.kind, record.reasons)
        record.plan = page_plan
        page = PlannedScan(page_plan, None, src, record, options, overrides, clock=clock)
        with clock.stage("render"):
            checked = self._check(src, page_plan, options, clock)
        if checked.ok:
            page.checked = checked
            if options.preview is not None:
                page.preview = _preview(src, page_plan, options.preview)
            return page
        failure = reason(
            checked.result.reason_code or "invalid_transform", status=checked.result.status
        )
        record.reasons.append(failure)
        record.guidance = collect_guidance(record.reasons)
        record.status, record.reason_code = failure.status, failure.code
        record.message = f"returned the original: {failure.code}"
        record.outcome = Outcome.ORIGINAL_RETURNED
        record.plan = None
        page.plan = None
        page.rejected.append(page_plan)
        return page

    def rectifier(self, options: ScanOptions | None = None, *, force_render: bool = False):
        """``rectify_batch``'s ``rectifier`` for this scanner (pair it with
        ``pipeline_record(options)``): the output size is the batch's (the input size by
        default), a failed scan has no image unless ``force_render`` renders its first
        rejected dewarp plan unchecked (``forced_render``)."""
        options = replace(options or ScanOptions(), on_error="return", preview=None)

        def rectify_one(image: NDArray[np.uint8], output_size) -> RenderResult:
            size = tuple(image.shape[:2]) if output_size is None else tuple(output_size)
            planned = self.plan(image, replace(options, output=size))
            result = self._finish(planned)
            passed = result.status == Status.SUCCESS
            rendered = RenderResult(
                status=result.status,
                reason_code=result.reason_code,
                message=result.message,
                image=result.image if passed else None,
                valid_mask=result.valid_mask if passed else None,
                used_fallback=result.used_fallback,
            )
            if not passed and force_render:
                if result.outcome == Outcome.PERSPECTIVE and result.image is not None:
                    rendered.image = result.image
                elif planned.rejected:
                    plan = planned.rejected[0]
                    rendered.image = render_unchecked(
                        planned.source.image, _dense(plan), plan.fill_rgb, plan.quantize
                    )
                rendered.metadata["forced_render"] = rendered.image is not None
            views = [] if result.plan is None else list(result.plan.views)
            rendered.metadata.update(
                validation=(result.metadata.get("renderer") or {}).get("validation"),
                dewarp={
                    "output_size": list(size),
                    "crops": views,
                    "seconds": result.timings.get("total"),
                },
                batch_item={
                    "outcome": result.outcome.value,
                    "reasons": [item.code for item in result.reasons],
                },
            )
            return rendered

        return rectify_one

    # -- stages

    def _memo(self, src: SourceImage) -> _Memo:
        return _Memo(self.cache.entry(src.image))

    def _pass(
        self,
        src: SourceImage,
        memo: _Memo,
        placement: Placement,
        box: tuple[int, int, int, int] | None,
        options: ScanOptions,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        key = (
            self.dewarper.info.get("path"),
            json.dumps(placement.record(), sort_keys=True),
            box,
            options.view_source,
        )

        def compute():
            return run_view(
                self.dewarper,
                src.image,
                placement,
                crop_box=box,
                view_source=options.view_source,
                levels=src.level,
            )

        grid, record = memo.get("infer", key, compute)
        return grid.copy(), copy.deepcopy(record)

    def _pass0(self, src: SourceImage, memo: _Memo, options: ScanOptions):
        def run() -> NDArray[np.float64]:
            return self._pass(src, memo, Placement.whole(src.info.size), None, options)[0]

        return run

    def _probability(self, src: SourceImage, memo: _Memo) -> NDArray | None:
        if self.gate is None:
            return None
        return memo.get(
            "detect",
            ("gate", self.gate.model_id),
            lambda: self.gate.predict_probability(src.image),
        )

    def _locate(
        self, src: SourceImage, memo: _Memo, options: ScanOptions, overrides: ScanOverrides
    ) -> locating.Localization:
        size = src.info.size
        if overrides.quad is not None:
            shape = (192, 144)
            if self.gate is not None:
                shape = tuple(self.gate.config.input_size)
            return locating.manual_quad(overrides.quad, size, shape)
        if overrides.region is not None:
            return locating.manual_region(overrides.region, size)
        return locating.locate_page(
            size,
            probability=self._probability(src, memo),
            pass0=self._pass0(src, memo, options),
            gate_config=self.gate_config,
            full_size=src.info.full_size,
        )

    def _orient_view(self, src: SourceImage, memo: _Memo, placement: Placement) -> OrientResult:
        key = (self.orienter.info.get("path"), json.dumps(placement.record(), sort_keys=True))
        return memo.get(
            "orient", key, lambda: self.orienter.predict(src.image, view_geometry(placement))
        )

    def _orient_pre(
        self,
        src: SourceImage,
        memo: _Memo,
        located: locating.Localization,
        options: ScanOptions,
        overrides: ScanOverrides,
    ) -> _Orientation:
        if overrides.quarter is not None:
            method = "oracle" if options.orient == "oracle" else "manual"
            return _Orientation(int(overrides.quarter), method, "manual")
        if options.orient == "oracle":
            raise _option_error("orient='oracle' needs the true quarter in overrides.quarter")
        if located.quarter is not None:
            return _Orientation(int(located.quarter), "quad", "manual")
        if options.orient == "none":
            return _Orientation(0, "none", "auto")
        if self.orienter is None:
            uncertain = reason(
                "orientation_uncertain", method="geometric_prior", why="no_orientation_model"
            )
            return _Orientation(0, "prior", "auto", reasons=[uncertain])
        theta = located.angle_deg if options.deskew else 0.0
        placement = crop_placement(located.box, theta, src.info.size)
        result = self._orient_view(src, memo, placement)
        if result.accepted:
            return _Orientation(result.k, "model", "auto", result)
        uncertain = reason(
            "orientation_uncertain",
            method="geometric_prior",
            probabilities=result.probabilities,
            threshold=result.threshold,
        )
        return _Orientation(0, "model", "auto", result, [uncertain])

    def _chain(
        self,
        src: SourceImage,
        memo: _Memo,
        located: locating.Localization,
        options: ScanOptions,
    ) -> Iterator[_Candidate]:
        """Try each candidate in order.

        Look up the other box after ``default`` and ``no_deskew`` fail.
        This may require a whole-photo prediction.
        """
        rerun = options.rerun
        yield _Candidate(
            "default", located.box, located.angle_deg, options.deskew, True, located.method, rerun
        )
        if not options.chain:
            return
        if options.deskew:
            yield _Candidate("no_deskew", located.box, 0.0, False, True, located.method, rerun)
        if located.provenance == "manual":
            return
        size = src.info.size
        other = locating.alternative_box(located, size, self._pass0(src, memo, options))
        if other is not None:
            box, theta, method = other
            yield _Candidate("other_box", box, theta, options.deskew, True, method, rerun)
        yield _Candidate("whole_photo", None, 0.0, False, True, "rectify", 0, turn="nodes")

    def _view(
        self,
        src: SourceImage,
        box: tuple[int, int, int, int] | None,
        angle: float,
        options: ScanOptions,
        outline: NDArray | None = None,
        *,
        first: bool = False,
    ) -> tuple[Placement, tuple[int, int, int, int] | None]:
        """The placement of a pass (and its whole-pixel crop box). Initial prediction on the whole
        image at no angle is ``Placement.whole``, as ``rectify``'s first prediction; crop
        passes keep their box so they match ``crop_pass``."""
        size = src.info.size
        if box is None:
            return Placement.whole(size), None
        height, width = size
        if first and tuple(box) == (0, 0, width, height) and angle == 0.0:
            return Placement.whole(size), None
        if options.view_extent == "tight" and outline is not None:
            return Placement.from_outline(outline, angle, size), None
        return crop_placement(box, angle, size), tuple(box)

    def _infer(
        self,
        src: SourceImage,
        memo: _Memo,
        candidate: _Candidate,
        quarter: int,
        options: ScanOptions,
        clock: StageClock,
    ) -> list[tuple[NDArray[np.float64], dict[str, Any]]]:
        """Initial prediction and the crop passes of one candidate, each placed in the source. With
        ``passes="auto"`` a crop pass that moved the page by more than ``max_drift`` gets
        one more crop pass (``settle``): when that one stays within the limit, the page has
        settled on the crop passes (initial prediction saw more than the page, such as the
        neighbouring page). A ``nodes`` candidate runs on the unturned view and re-indexes
        every grid by the quarter turn (``quarter_turn_grid``)."""
        size = src.info.size
        theta = candidate.theta if candidate.deskew else 0.0
        in_view = candidate.orient and candidate.turn == "view"
        angle = theta + 90.0 * quarter if in_view else theta
        placement, box = self._view(src, candidate.box, angle, options, first=True)
        with clock.stage("infer_pass_initial"):
            grid, record = self._pass(src, memo, placement, box, options)
        passes = [(grid, {"pass": "initial", **record})]
        settle = options.passes == "auto" and candidate.rerun > 0
        for number in range(1, candidate.rerun + 1 + int(settle)):
            if number > candidate.rerun and not _drifted(passes, size):
                break
            previous = passes[-1][0]
            view_angle = passes[-1][1]["placement"]["angle_deg"]
            next_box, estimate, crop = rerun_box(
                previous, size, deskew=candidate.deskew, max_angle_deg=None
            )
            if next_box is None:
                passes.append((previous, {"pass": number, **crop}))
                break
            if candidate.deskew and crop["deskew_skipped"] is None:
                if _angle_difference(estimate, view_angle) > 45.0:
                    crop["deskew_skipped"] = "angle_disagrees_with_view"
                    estimate = 90.0 * split_angle(view_angle)[1] if in_view else 0.0
            else:
                estimate = 90.0 * split_angle(view_angle)[1] if in_view else 0.0
            outline = outline_points(previous, size)
            placement, crop_box = self._view(src, next_box, estimate, options, outline)
            with clock.stage(f"infer_pass_{number}"):
                grid, record = self._pass(src, memo, placement, crop_box, options)
            crop["rotated"] = placement.angle_deg != 0.0
            crop["canvas_size"] = [float(v) for v in placement.view_size]
            if number > candidate.rerun:
                crop["settle"] = True
            passes.append((grid, {"pass": number, **crop, **record}))
        turns = quarter % 4 if candidate.orient and candidate.turn == "nodes" else 0
        if turns:
            passes = [
                (quarter_turn_grid(grid, turns), {**record, "quarter_turn": turns})
                for grid, record in passes
            ]
        return passes

    def _flat_preview(
        self, src: SourceImage, grid: NDArray, long_side: int | None = None
    ) -> NDArray[np.uint8]:
        size = src.info.size
        output = plan_output(grid, size, output=long_side or VERIFY["post_preview_long_side"])
        backward = dense_map(grid, output.size, nodes=self.dewarper.config.grid_convention)
        density = output.density_median or 1.0
        factor = max(f for f in (1, 2, 4, 8) if f == 1 or f <= density)
        level = src.level(factor)
        return sample_rgb(level, backward, source_inside(backward), (255, 255, 255), "round")

    def _orient_post(
        self, src: SourceImage, grid: NDArray, orientation: _Orientation, options: ScanOptions
    ) -> tuple[int | None, dict[str, Any] | None]:
        """(k_post, record): the preview's quarter turn when OrientNet accepts one."""
        if self.orienter is None or options.orient != "model" or orientation.provenance == "manual":
            return None, None
        try:
            preview = self._flat_preview(src, grid)
        except FlattenerError as error:
            return None, {"skipped": error.reason_code}
        result = self.orienter.predict(preview)
        record = {"result": result.to_dict(), "action": None}
        return (result.k if result.accepted else None), record

    def _judge_look(
        self, src: SourceImage, grid: NDArray, checks: dict[str, Any], mask_past: bool
    ) -> Reason | None:
        """The failure a grid's small flat preview shows (``scan.look``), or ``None``. A
        grid the gate mask reaches past (``mask_past``) must look like paper; any grid
        that shows a zoomed piece of text cut by its edges is a page cut off by the photo.
        The measures go to ``checks["paper"]`` and ``checks["text"]``."""
        try:
            preview = self._flat_preview(src, grid, look.SIZE)
        except FlattenerError:
            return None  # the plan and render checks judge a grid that gives no preview
        if mask_past:
            checks["paper"] = paper = look.paper(preview)
            if (
                paper["saturation"] > VERIFY["page_max_saturation"]
                or paper["dark"] > VERIFY["page_max_dark"]
            ):
                return reason("page_not_found", check="paper", **checks)
        checks["text"] = measured = look.text(preview)
        edges = sorted(measured["edges"].values())
        if (
            measured["lines"] is not None
            and VERIFY["fragment_min_lines"] <= measured["lines"] < VERIFY["fragment_max_lines"]
            and measured["periodicity"] >= VERIFY["fragment_min_periodicity"]
            and edges[-2] >= VERIFY["fragment_min_edges"]
        ):
            return reason(
                "page_cut_off",
                status=Status.MODEL_ERROR,
                severity=Severity.FAILURE,
                check="fragment",
                **checks,
            )
        return None

    def _small_photo(self, src: SourceImage) -> NDArray[np.uint8]:
        """The photo with a long side of ``VERIFY["look_photo_long_side"]`` pixels (or
        less for a smaller photo), from the smallest pyramid level that is large enough."""
        long_side = VERIFY["look_photo_long_side"]
        longest = max(src.info.size)
        factor = max(f for f in (1, 2, 4, 8) if f == 1 or longest / f >= long_side)
        with Image.fromarray(src.level(factor)) as image:
            image.thumbnail((long_side, long_side))
            return np.asarray(image)

    def _page_continues(
        self,
        src: SourceImage,
        memo: _Memo,
        options: ScanOptions,
        located: locating.Localization,
        box: tuple[int, int, int, int],
        grid: NDArray,
        checks: dict[str, Any],
    ) -> Reason | None:
        """For a grid the gate mask reaches past: the failure when the first crop ``box``
        cut the page, else ``None``. The crop cut the page when blank paper of the grid's
        colour fills at least ``crop_edge_paper`` of a strip along a box edge inside the
        photo, and the whole-photo prediction holds the gate mask (coverage at least
        ``min_coverage``) and adds at most the grid's own area, at least
        ``continue_min_paper`` of it page-coloured (as when the gate misses white page
        margins). The chain then goes on to the candidates whose box is the whole-photo outline or
        the whole photo. The measures go to ``checks["beyond"]``."""
        photo = self._small_photo(src)
        shape = photo.shape[:2]
        inside = locating.rasterize_polygon(locating.outline_polygon(grid), shape)
        colour = look.paper_colour(photo, inside)
        if colour is None:
            return None
        coloured = look.page_coloured(photo, colour)
        beyond = {"crop_edge": look.crop_edge_share(coloured, box, src.info.size)}
        checks["beyond"] = beyond
        if beyond["crop_edge"] < VERIFY["crop_edge_paper"]:
            return None
        whole = self._pass0(src, memo, options)()
        if not np.isfinite(whole).all():
            return None
        try:
            beyond["whole_coverage"] = locating.grid_coverage(whole, located.mask)
        except FlattenerError:
            return None
        added = locating.rasterize_polygon(locating.outline_polygon(whole), shape) & ~inside
        beyond["added_area"] = float(added.sum() / max(1, inside.sum()))
        beyond["added_paper"] = float(coloured[added].mean()) if added.sum() >= 50 else 0.0
        if (
            beyond["whole_coverage"] >= VERIFY["min_coverage"]
            and beyond["added_area"] <= VERIFY["continue_max_area"]
            and beyond["added_paper"] >= VERIFY["continue_min_paper"]
        ):
            return reason("unstable_geometry", check="page_continues", **checks)
        return None

    def _plan_for(
        self,
        src: SourceImage,
        grid: NDArray,
        passes: list[dict[str, Any]],
        located: locating.Localization,
        orientation: _Orientation,
        candidate: _Candidate,
        options: ScanOptions,
        overrides: ScanOverrides,
    ) -> ScanPlan:
        size = src.info.size
        inset = resolve_inset(options.inset)
        request = overrides.output_size if overrides.output_size is not None else options.output
        aspect = overrides.aspect if overrides.aspect is not None else options.aspect
        focal = src.info.focal_px() if options.use_focal else None
        output = plan_output(grid, size, output=request, aspect=aspect, focal_px=focal, inset=inset)
        output = _with_border(output, inset, float(options.border), request)
        angle, _ = estimate_page_angle(grid, size)
        if angle is None:  # the angle of the last pass that ran
            placed = [view["placement"] for view in passes if "placement" in view]
            angle = placed[-1]["angle_deg"]
        probs = None
        if orientation.result is not None:
            probs = tuple(orientation.result.probabilities)
        return ScanPlan(
            source=src.info,
            output=output,
            kind="dewarp",
            grid=grid,
            region={
                "candidate": candidate.name,
                "method": candidate.method,
                "box": None if candidate.box is None else list(candidate.box),
                "localization": located.method,
            },
            angle_deg=float(angle),
            quarter=orientation.quarter if candidate.orient else 0,
            orientation_probs=probs,
            views=tuple(passes),
            inset=inset,
            border=float(options.border),
            quantize=options.quantize,
            nodes=self.dewarper.config.grid_convention,
            fill_rgb=tuple(options.fill_rgb),
            sharpen=float(options.sharpen),
            models=tuple(self.models),
            thresholds={"locate": locating.THRESHOLDS, "verify": VERIFY},
            provenance=self._provenance(located, orientation, overrides),
        )

    def _provenance(
        self,
        located: locating.Localization,
        orientation: _Orientation | None,
        overrides: ScanOverrides,
    ) -> dict[str, str]:
        return {
            "region": located.provenance,
            "orientation": "auto" if orientation is None else orientation.provenance,
            "output_size": "manual" if overrides.output_size is not None else "auto",
            "aspect": "manual" if overrides.aspect is not None else "auto",
            "dewarp": "manual" if overrides.skip_dewarp else "auto",
        }

    def _check(
        self,
        src: SourceImage,
        plan: ScanPlan,
        options: ScanOptions,
        clock: StageClock | None = None,
    ) -> _Checked:
        """``flattener.render`` of the plan's map, once (the renderer's checks); with
        ``fast_check`` a dewarp plan gets ``scan.fast_check`` and no image yet. The map,
        the checks and the sampling are timed as ``render_map``, ``render_check`` and
        ``render_sample`` (the last two from the renderer's own timings)."""
        clock = clock or StageClock()
        config = RenderConfig(on_error="return", fill_rgb=tuple(plan.fill_rgb))
        if options.fast_check and plan.kind == "dewarp":
            fast = fast_check(
                src.image,
                plan.grid,
                plan.output.size,
                inset=plan.inset,
                border=plan.border,
                config=config,
                levels=src.level,
                nodes=plan.nodes,
            )
            clock.add("render_check", fast.metadata["timings"]["total"])
            return _Checked(plan, fast.result(), fast)
        with clock.stage("render_map"):
            backward = _dense(plan)
        result = flattener_render(src.image, backward, config=config)
        timings = result.metadata.get("timings") or {}
        checks = [timings.get(name) for name in ("prepare_input", "process_coordinates")]
        clock.add("render_check", sum(value for value in checks if value is not None))
        if timings.get("render") is not None:
            clock.add("render_sample", timings["render"])
        if result.status == Status.SUCCESS and plan.quantize != "round":
            with clock.stage("render_sample"):
                result.image = sample_rgb(
                    src.image, backward, result.valid_mask, config.fill_rgb, plan.quantize
                )
        return _Checked(plan, result)

    def _perspective_plan(
        self,
        src: SourceImage,
        quad: NDArray[np.float64],
        located: locating.Localization,
        orientation: _Orientation | None,
        options: ScanOptions,
        overrides: ScanOverrides,
        method: str,
    ) -> ScanPlan:
        size = src.info.size
        homography = quad_homography(quad, size)
        grid = homography_grid(homography)
        inset = resolve_inset(options.inset)
        request = overrides.output_size if overrides.output_size is not None else options.output
        aspect = overrides.aspect if overrides.aspect is not None else options.aspect
        focal = src.info.focal_px() if options.use_focal else None
        output = plan_output(grid, size, output=request, aspect=aspect, focal_px=focal, inset=inset)
        output = _with_border(output, inset, float(options.border), request)
        return ScanPlan(
            source=src.info,
            output=output,
            kind="perspective",
            homography=homography,
            region={"method": method, "quad": quad.tolist(), "localization": located.method},
            angle_deg=locating.quad_angle(quad),
            quarter=0 if orientation is None else orientation.quarter,
            inset=inset,
            border=float(options.border),
            quantize=options.quantize,
            fill_rgb=tuple(options.fill_rgb),
            sharpen=float(options.sharpen),
            models=tuple(self.models),
            thresholds={"locate": locating.THRESHOLDS, "verify": VERIFY},
            provenance=self._provenance(located, orientation, overrides),
        )

    def _certain_quad(
        self,
        located: locating.Localization,
        first_grid: NDArray | None,
        source_size: tuple[int, int],
        quarter: int | None = None,
    ) -> NDArray[np.float64] | None:
        """Return the page quad in source pixels and reading order.

        Use a manual quad, reordered by a ``quarter`` override, or the initial
        prediction's corners when the gate mask agrees with their coverage and precision.
        """
        if located.quad is not None:
            if quarter is None or located.quarter is None:
                return located.quad
            return turn_quad(located.quad, quarter - located.quarter)
        if first_grid is None or located.mask is None:
            return None
        grid = np.asarray(first_grid)
        corners = np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
        quad_grid = np.array([[corners[0], corners[1]], [corners[3], corners[2]]])
        try:
            coverage = locating.grid_coverage(quad_grid, located.mask)
        except FlattenerError:
            return None
        precision = locating.outline_mask_precision(quad_grid, located.mask)
        thresholds = locating.THRESHOLDS
        if (
            coverage < thresholds["consensus_coverage"]
            or precision < thresholds["consensus_precision"]
        ):
            return None
        height, width = source_size
        return (corners + 1.0) / 2.0 * np.array([width, height])

    def _plan(
        self,
        src: SourceImage,
        options: ScanOptions,
        overrides: ScanOverrides,
        clock: StageClock,
    ) -> PlannedScan:
        memo = self._memo(src)
        planned = self._stages(src, memo, options, overrides, clock)
        planned.record.metadata["cache"] = memo.record()
        return planned

    def _stages(
        self,
        src: SourceImage,
        memo: _Memo,
        options: ScanOptions,
        overrides: ScanOverrides,
        clock: StageClock,
    ) -> PlannedScan:
        size = src.info.size
        reasons: list[Reason] = list(src.reasons)
        metadata: dict[str, Any] = {
            "pipeline": self.pipeline_record(options),
            "options": options.record(),
            "overrides": overrides.record(),
        }
        record = ScanResult(
            status=Status.SUCCESS,
            outcome=Outcome.RECTIFIED,
            source=src.info,
            models=list(self.models),
            input=src.path,
            metadata=metadata,
        )
        planned = PlannedScan(None, None, src, record, options, overrides, clock=clock)
        with clock.stage("detect"):
            located = self._locate(src, memo, options, overrides)
        reasons.extend(located.reasons)
        metadata["localization"] = located.record()
        shown = locating.THRESHOLDS["component_min_area_fraction"]
        record.candidates["regions"] = [
            item.to_dict()["box"] for item in located.components if item.area_fraction >= shown
        ]
        if located.failure is not None:
            return self._fallback(planned, memo, reasons, located.failure, located, None, None)
        if overrides.skip_dewarp:
            return self._skip_dewarp(planned, memo, reasons, located, options, overrides)
        with clock.stage("orient"):
            orientation = self._orient_pre(src, memo, located, options, overrides)
        reasons.extend(orientation.reasons)
        if orientation.result is not None:
            order = np.argsort(-np.asarray(orientation.result.probabilities), kind="stable")[:2]
            record.candidates["orientations"] = [
                {"quarter": int(k), "probability": orientation.result.probabilities[int(k)]}
                for k in order
            ]
        elif orientation.method == "prior":
            box = located.box
            landscape = box is not None and (box[2] - box[0]) > (box[3] - box[1])
            record.candidates["orientations"] = [
                {"quarter": k, "probability": None} for k in ((0, 1, 3) if landscape else (0, 2))
            ]
        tried: list[dict[str, Any]] = []
        metadata["chain"] = tried
        seen: set[tuple] = set()
        first_grid, first_quarter = None, 0
        stopped = False
        first_failure: Reason | None = None
        failed: list[str] = []
        chain = self._chain(src, memo, located, options)
        while True:
            with clock.stage("detect"):  # the other box may need whole-photo prediction
                candidate = next(chain, None)
            if candidate is None:
                break
            quarter = orientation.quarter if candidate.orient else 0
            key = candidate.key(quarter, size)
            if key in seen:
                tried.append({**candidate.record(), "skipped": "duplicate"})
                continue
            seen.add(key)
            outcome = self._try(
                planned, memo, candidate, located, orientation, options, overrides, reasons
            )
            tried.append(outcome["record"])
            if first_grid is None and outcome.get("first_grid") is not None:
                first_grid, first_quarter = outcome["first_grid"], outcome["first_quarter"]
            if outcome.get("checked") is not None:
                if first_failure is not None:
                    reasons.append(
                        reason(
                            "candidate_fallback",
                            candidate=candidate.name,
                            failed=failed,
                            failure=first_failure.to_dict(),
                        )
                    )
                return self._accept(planned, outcome, reasons, outcome["orientation"])
            failed.append(candidate.name)
            if first_failure is None or outcome.get("stop"):
                first_failure = outcome["failure"]
            if outcome.get("stop"):
                stopped = True
                break  # the page runs out of the photo; no candidate can bring it back
            if outcome["orientation"].settled_by is not None:
                # A quarter turn settled on a flat preview holds for the later candidates.
                orientation = outcome["orientation"]
        failure = first_failure or reason("unstable_geometry", why="no_candidate")
        quad = None
        if not stopped:  # a cut-off page has no corners to flatten
            quad = self._certain_quad(located, first_grid, size, overrides.quarter)
        if quad is not None and located.quad is None:
            # The grid's nodes follow its own quarter turn; the plan records the latest one.
            quad = turn_quad(quad, orientation.quarter - first_quarter)
        return self._fallback(planned, memo, reasons, failure, located, orientation, quad)

    def _try(
        self,
        planned: PlannedScan,
        memo: _Memo,
        candidate: _Candidate,
        located: locating.Localization,
        orientation: _Orientation,
        options: ScanOptions,
        overrides: ScanOverrides,
        reasons: list[Reason],
    ) -> dict[str, Any]:
        """Infer, verify, plan and check one candidate. Returns ``checked`` when a grid
        passes (the last pass, else initial prediction), otherwise ``failure`` (the first one). A
        ``FlattenerError`` from the plan or the check is that grid's failure."""
        src, clock = planned.source, planned.clock
        size = src.info.size
        quarter = orientation.quarter if candidate.orient else 0
        entry: dict[str, Any] = {**candidate.record(), "quarter": quarter}
        with clock.stage("infer"):
            passes = self._infer(src, memo, candidate, quarter, options, clock)
        if candidate.orient:
            with clock.stage("orient_post"):
                k_post, post = self._orient_post(src, passes[-1][0], orientation, options)
            if k_post == 2:
                passes = [(flip_grid(grid), {**rec, "flipped": True}) for grid, rec in passes]
                quarter = (quarter + 2) % 4
                post["action"] = "flip_180"
            elif k_post in (1, 3):
                quarter = (quarter + k_post) % 4
                with clock.stage("infer"):
                    passes = self._infer(src, memo, candidate, quarter, options, clock)
                verb = "reindex" if candidate.turn == "nodes" else "reinfer"
                post["action"] = f"{verb}_quarter_{quarter}"
            if post is not None:
                settled = candidate.name if _post_accepted(post) else orientation.settled_by
                orientation = replace(orientation, quarter=quarter, post=post, settled_by=settled)
            entry["orientation_post"] = post
        entry["passes"] = [record for _, record in passes]
        grids = [("final", passes[-1][0], len(passes))]
        if len(passes) > 1:
            grids.append(
                ("initial", passes[0][0], 1)
            )  # initial prediction when the crop passes fail
        result: dict[str, Any] = {
            "first_grid": passes[0][0],
            "first_quarter": quarter,
            "orientation": orientation,
        }
        failures: list[tuple[Reason, dict[str, Any]]] = []
        for label, grid, count in grids:
            checks: dict[str, Any] = {}
            bad = None
            with clock.stage("verify"):
                if not np.isfinite(grid).all():
                    bad = reason("non_finite_grid", status=Status.INVALID_TRANSFORM)
                else:
                    if located.mask is not None:
                        checks["coverage"] = locating.grid_coverage(grid, located.mask)
                        if checks["coverage"] < VERIFY["min_coverage"]:
                            checks["excess"] = locating.region_excess(grid, located.mask, size)
                    if label == "final" and len(passes) > 1:
                        checks["drift"] = median_drift(passes[-2][0], grid, size)
                    excess = checks.get("excess")
                    # A manual quad is the page; only the gate's mask is a hint.
                    strict = located.provenance == "manual"
                    if (
                        located.method == "fills_frame"
                        and checks.get("coverage", 1.0) < VERIFY["close_up_min_coverage"]
                    ):
                        bad = reason(
                            "page_cut_off",
                            status=Status.MODEL_ERROR,
                            severity=Severity.FAILURE,
                            check="close_up",
                            **checks,
                        )
                    elif excess and (
                        strict or min(excess.values()) > VERIFY["fragment_side_excess"]
                    ):
                        bad = reason("unstable_geometry", check="coverage", **checks)
                    elif checks.get("drift", 0.0) > VERIFY["max_drift"]:
                        bad = reason("unstable_geometry", check="drift", **checks)
                    elif not strict:
                        bad = self._judge_look(src, grid, checks, bool(excess))
                        if bad is None and excess and candidate.box is not None:
                            bad = self._page_continues(
                                src, memo, options, located, candidate.box, grid, checks
                            )
            if bad is not None:
                failures.append((bad, {"grid": label, "checks": checks}))
                if bad.code == "page_cut_off":
                    # All candidates see the same cropped photo.
                    break
                continue
            views = [record for _, record in passes[:count]]
            try:
                with clock.stage("plan"):
                    plan = self._plan_for(
                        src, grid, views, located, orientation, candidate, options, overrides
                    )
                with clock.stage("render"):
                    checked = self._check(src, plan, options, clock)
            except FlattenerError as error:
                bad = reason(
                    error.reason_code or "invalid_transform",
                    status=error.status,
                    message=error.message,
                )
                failures.append((bad, {"grid": label, "checks": checks}))
                continue
            if not checked.ok:
                planned.rejected.append(plan)
                validation = checked.result.metadata.get("validation")
                bad = reason(
                    checked.result.reason_code or "invalid_transform",
                    status=checked.result.status,
                    validation=validation,
                )
                failures.append((bad, {"grid": label, "checks": checks}))
                continue
            outside = (checked.result.metadata.get("validation") or {}).get(
                "out_of_bounds_fraction"
            )
            cut_off = not strict and outside is not None and outside > VERIFY["cut_off_outside"]
            if (
                cut_off
                and outside <= VERIFY["cut_off_flatten_outside"]
                and _orientation_confirmed(orientation)
            ):
                checks["cut_off"] = {"outside": outside, "sides": _outside_sides(grid)}
                cut_off = False
            if cut_off:
                bad = reason(
                    "page_cut_off",
                    status=Status.MODEL_ERROR,
                    severity=Severity.FAILURE,
                    check="outside",
                    outside=outside,
                    **checks,
                )
                failures.append((bad, {"grid": label, "checks": checks}))
                break  # the other candidates see the same cut page
            entry.update(accepted=label, checks=checks)
            entry["failures"] = [{**where, "failure": item.to_dict()} for item, where in failures]
            result.update(
                checked=checked,
                checks=checks,
                rerun_discarded=label != "final",
                failures=[item.code for item, _ in failures],
                record=entry,
            )
            return result
        entry["accepted"] = None
        entry["failures"] = [{**where, "failure": item.to_dict()} for item, where in failures]
        first = failures[0][0] if failures else reason("unstable_geometry", why="no_grid")
        cut_off = failures[-1][0].code == "page_cut_off" if failures else False
        if cut_off:
            first = failures[-1][0]
        detail = {**first.detail, "candidate": candidate.name}
        result.update(
            checked=None,
            failure=Reason(first.code, first.severity, first.status, detail),
            record=entry,
            stop=cut_off,
        )
        return result

    def _accept(
        self,
        planned: PlannedScan,
        outcome: dict[str, Any],
        reasons: list[Reason],
        orientation: _Orientation,
    ) -> PlannedScan:
        checked: _Checked = outcome["checked"]
        plan = checked.plan
        record = planned.record
        if outcome["rerun_discarded"]:
            reasons.append(reason("rerun_discarded", failures=outcome["failures"]))
        if plan.kind == "dewarp":
            reasons[:] = _cut_off_on_page(reasons, plan.grid)
        cut = outcome["checks"].get("cut_off")
        if cut is not None:
            detail = {"sides": cut["sides"], "evidence": "outside", "outside": cut["outside"]}
            reasons.append(reason("page_cut_off", **detail))
        coverage = outcome["checks"].get("coverage")
        if coverage is not None and coverage < VERIFY["min_coverage"]:
            excess = outcome["checks"].get("excess")
            reasons.append(reason("page_region_larger", coverage=coverage, excess=excess))
        if orientation.settled_by is not None:
            # OrientNet decided on a flat preview; the pre-dewarp doubt is settled.
            reasons[:] = [item for item in reasons if item.code != "orientation_uncertain"]
        density_min = plan.output.density_min
        if density_min is not None and density_min < VERIFY["low_resolution_density"]:
            reasons.append(reason("low_resolution", density_min=density_min))
        record.reasons = reasons
        record.guidance = collect_guidance(reasons)
        record.outcome = decide_outcome(plan.kind, reasons)
        record.plan = plan
        orientation_qa = orientation.record()
        if orientation.result is not None:
            orientation_qa["margin"] = orientation.result.margin
        record.qa = {
            "candidate": plan.region.get("candidate"),
            "coverage": outcome["checks"].get("coverage"),
            "region_excess": outcome["checks"].get("excess"),
            "paper": outcome["checks"].get("paper"),
            "text": outcome["checks"].get("text"),
            "beyond": outcome["checks"].get("beyond"),
            "drift": outcome["checks"].get("drift"),
            "drift_limit": VERIFY["max_drift"],
            "orientation": orientation_qa,
            "density_min": plan.output.density_min,
            "density_median": plan.output.density_median,
            "rerun_discarded": outcome["rerun_discarded"],
        }
        record.metadata["orientation"] = orientation.record()
        planned.plan = plan
        planned.checked = checked
        self._make_preview(planned, plan)
        return planned

    def _make_preview(self, planned: PlannedScan, plan: ScanPlan) -> None:
        if planned.options.preview is not None:
            with planned.clock.stage("preview"):
                planned.preview = _preview(planned.source, plan, planned.options.preview)

    def _skip_dewarp(
        self,
        planned: PlannedScan,
        memo: _Memo,
        reasons: list[Reason],
        located: locating.Localization,
        options: ScanOptions,
        overrides: ScanOverrides,
    ) -> PlannedScan:
        """A requested homography-only plan from the quad, the region, the gate's page box or,
        without a gate, the corners of the whole-photo outline. The quarter turn comes from
        the overrides or the quad, else from OrientNet as on the dewarp path
        (``orientation_uncertain`` when it is unsure or missing)."""
        src, clock = planned.source, planned.clock
        with clock.stage("orient"):
            orientation = self._orient_pre(src, memo, located, options, overrides)
        reasons.extend(orientation.reasons)
        if located.quad is not None:
            quad = self._certain_quad(located, None, src.info.size, overrides.quarter)
        else:
            if overrides.region is not None:
                left, top, right, bottom = locating.region_bounds(overrides.region, src.info.size)
                quad = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], float)
            elif located.mask is not None and located.method != "fills_frame":
                points = locating.cell_corners(located.mask, src.info.size)
                quad = rect_quad(points, located.angle_deg)
            elif located.mask is None and located.pass0 is not None:
                grid = np.asarray(located.pass0)
                corners = np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
                height, width = src.info.size
                quad = (corners + 1.0) / 2.0 * np.array([width, height])
            else:
                height, width = src.info.size
                quad = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
            quad = turn_quad(quad, orientation.quarter)
        with clock.stage("plan"):
            plan = self._perspective_plan(
                src, quad, located, orientation, options, overrides, "skip_dewarp"
            )
        with clock.stage("render"):
            checked = self._check(src, plan, options, clock)
        record = planned.record
        record.metadata["chain"] = [{"name": "skip_dewarp", "status": str(checked.result.status)}]
        record.metadata["orientation"] = orientation.record()
        if not checked.ok:
            failure = reason(
                checked.result.reason_code or "invalid_transform", status=checked.result.status
            )
            planned.rejected.append(plan)
            return self._fallback(planned, memo, reasons, failure, located, orientation, None)
        record.reasons = reasons
        record.guidance = collect_guidance(reasons)
        record.outcome = Outcome.PERSPECTIVE
        record.plan = plan
        record.qa = {
            "candidate": "skip_dewarp",
            "orientation": orientation.record(),
            "density_min": plan.output.density_min,
            "density_median": plan.output.density_median,
        }
        planned.plan, planned.checked = plan, checked
        self._make_preview(planned, plan)
        return planned

    def _fallback(
        self,
        planned: PlannedScan,
        memo: _Memo,
        reasons: list[Reason],
        failure: Reason,
        located: locating.Localization,
        orientation: _Orientation | None,
        quad: NDArray[np.float64] | None,
    ) -> PlannedScan:
        """The perspective fallback (when asked for and the quad is sure) or the original."""
        record, options, src = planned.record, planned.options, planned.source
        reasons = [*reasons, failure]
        if options.fallback == "perspective" and quad is not None:
            with planned.clock.stage("plan"):
                plan = self._perspective_plan(
                    src,
                    quad,
                    located,
                    orientation,
                    options,
                    planned.overrides,
                    "perspective_fallback",
                )
            with planned.clock.stage("render"):
                checked = self._check(src, plan, options, planned.clock)
            record.metadata.setdefault("chain", []).append(
                {
                    "name": "perspective_fallback",
                    "status": str(checked.result.status),
                    "reason": checked.result.reason_code,
                }
            )
            if checked.ok:
                reasons.append(reason("curvature_not_corrected"))
                record.reasons = reasons
                record.guidance = collect_guidance(reasons)
                record.status = failure.status
                record.reason_code = failure.code
                record.message = f"fell back to the page quad: {failure.code}"
                record.used_fallback = True
                record.outcome = Outcome.PERSPECTIVE
                record.plan = plan
                planned.plan, planned.checked = plan, checked
                self._make_preview(planned, plan)
                return planned
            planned.rejected.append(plan)
        record.reasons = reasons
        record.guidance = collect_guidance(reasons)
        record.status = failure.status
        record.reason_code = failure.code
        record.message = f"returned the original: {failure.code}"
        record.outcome = Outcome.ORIGINAL_RETURNED
        planned.plan = None
        return planned

    def _check_source(self, src: SourceImage, plan: ScanPlan) -> None:
        """A stored geometry must match the input dimensions."""
        if tuple(src.info.size) != tuple(plan.source.size):
            raise FlattenerError(
                Status.INPUT_ERROR,
                "source_mismatch",
                "the plan belongs to a source of another size",
            )

    def _planned_from_plan(
        self, src: SourceImage, plan: ScanPlan, options: ScanOptions, clock: StageClock
    ) -> PlannedScan:
        """A ``PlannedScan`` of a stored plan. The result carries the plan's own model
        records (the weights that made its geometry), their model settings."""
        self._check_source(src, plan)
        models = [dict(model) for model in plan.models]
        record = ScanResult(
            status=Status.SUCCESS,
            outcome=decide_outcome(plan.kind),
            source=src.info,
            models=models,
            input=src.path,
            plan=plan,
            metadata={"pipeline": self.pipeline_record(options), "options": options.record()},
        )
        planned = PlannedScan(plan, None, src, record, options, clock=clock)
        with clock.stage("render"):
            checked = self._check(src, plan, options, clock)
        if checked.ok:
            planned.checked = checked
            return planned
        failure = reason(
            checked.result.reason_code or "invalid_transform", status=checked.result.status
        )
        record.reasons = [failure]
        record.guidance = collect_guidance(record.reasons)
        record.status, record.reason_code = failure.status, failure.code
        record.message = f"returned the original: {failure.code}"
        record.outcome = Outcome.ORIGINAL_RETURNED
        planned.plan = None
        planned.rejected.append(plan)
        return planned

    def _finish(self, planned: PlannedScan) -> ScanResult:
        """The final result of a planned scan, with the failure policy applied."""
        record = copy.copy(planned.record)
        record.reasons = list(planned.record.reasons)
        record.metadata = dict(planned.record.metadata)
        options, src, clock = planned.options, planned.source, planned.clock
        if planned.plan is not None and planned.checked is not None:
            result = planned.checked.result
            fast = planned.checked.fast
            plan = planned.plan
            if fast is not None and result.image is None:
                with clock.stage("render"), clock.stage("render_sample"):
                    result.image = fast.sample(src.image, plan.fill_rgb, plan.quantize)
            record.image = result.image
            if plan.sharpen > 0 and result.image is not None:
                with clock.stage("render"), clock.stage("render_sharpen"):
                    record.image = sharpening.sharpen(result.image, plan.sharpen, result.valid_mask)
                record.metadata["sharpen"] = sharpening.record(plan.sharpen)
            record.valid_mask = result.valid_mask
            record.plan = planned.plan
            record.metadata["renderer"] = result.metadata
            record.preview = planned.preview
        else:
            record.image = None
            if options.on_error == "original":
                record.image = src.image.copy()
                record.used_fallback = True
            elif options.on_error == "raise":
                _stamp_timings(record, clock)
                raise FlattenerError(
                    record.status,
                    record.reason_code or "scan_failed",
                    record.message,
                    metadata={"scan": record.to_dict()},
                )
            else:
                record.outcome = Outcome.FAILED
        _stamp_timings(record, clock)
        return record


def _timings(clock: StageClock | None) -> dict[str, float]:
    if clock is None:
        return {}
    return {**clock.seconds(), "total": clock.elapsed()}


def _stamp_timings(record: ScanResult, clock: StageClock | None) -> None:
    """Per-stage seconds in ``timings`` and every stage record in the metadata."""
    record.timings = _timings(clock)
    if clock is not None:
        record.metadata["timing_stages"] = clock.to_dict()["stages"]
