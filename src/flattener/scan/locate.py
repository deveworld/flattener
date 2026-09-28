"""Where the page is: the detect stage of the scanner.

Inputs are the page gate's probability grid P (``regions.context``, whole image extent)
and, when needed, the pass-0 grid G0: one dewarp pass on the whole photo, in flattener
values of the source. ``locate_page`` returns a ``Localization``: the whole-pixel crop box
B of the first view, its angle theta in (-45, 45] (clockwise positive, as
``dewarp.predict.estimate_page_angle``), the page mask M on the gate grid, the gate's
components and the reasons found on the way.

Rules (``THRESHOLDS``, the design's starting values; they are recorded with every scan):

* The gate runs ``detect_region`` with an area cap of 1.0. ``page_not_found`` ends the
  scan. A gate split into pieces (``ambiguous_page_region``, or ``two_page_candidate``
  when the two largest components lie side by side) asks G0: its outline is the page
  (box and angle, method ``outline``) and the pieces holding at least half of their
  cells inside it are the mask; the code stays as a warning with the component boxes.
  When G0 has no outline, holds no piece, or the held pieces cover less than half of its
  outline (``outline_mask_precision``; a desk or a keyboard the gate cut into pieces, or a
  close-up whose page runs out of the photo), the split ends the scan. A page under 1 % of
  the photo or with a long side under 600 full-resolution pixels is ``page_too_small``.
* A mask over 90 % of the frame or touching three borders fills the frame
  (``page_fills_frame``, informational): B is the whole image and theta is 0.
* A confident gate (dominance >= 0.95, mean probability >= 0.85) gives B = the mask's
  bounds plus 4 % of their extent per side, and theta from the mask's minimum-area
  rectangle.
* Otherwise (and when the gate scores are weak) G0 decides with the consensus rule on
  c = ``page_coverage(G0, M)`` and p = ``outline_mask_precision(G0, M)``: c >= 0.95 and
  p >= 0.85 take the outline's box (``crop_box``) and angle; p < 0.85 takes the mask box
  with the angle of the mask's principal axis; c < 0.95 takes the union of the outline
  box and the mask box with the outline's angle.
* Without a gate the outline's box and angle are used with ``unverified_region``.
* Warnings: ``page_cut_off`` names each image side along which the mask runs for more than
  15 % of the side (and, when G0 is at hand, G0 reaches past that side);
  ``possible_occlusion`` when the mask's solidity (of a split gate, its largest held
  piece) is below 0.90. Other components that touch the border are left out with
  ``neighbor_page_partial`` (informational).
* ``region_excess`` measures, per page side, the mask left outside a grid's outline; the
  scanner's verify stage uses it to tell a neighbouring page, the book edge or the paper
  under the page (mask past some sides) from a grid that holds only a piece of the page
  (mask past every side).

``manual_region`` and ``manual_quad`` turn user overrides into the same record; a quad in
reading order (top-left, top-right, bottom-right, bottom-left) also gives the page's
quarter turn.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import FlattenerError, Status
from ..dewarp.predict import RERUN_MARGIN, crop_box, estimate_page_angle, outline_points
from ..regions.context import GateConfig, gate_page_region
from ..regions.geometry import PageRegion, page_coverage
from .placement import split_angle
from .records import Reason, Severity, reason

THRESHOLDS: dict[str, Any] = {
    "version": "m1-design-start",
    "consensus_coverage": 0.95,
    "consensus_precision": 0.85,
    "cut_off_side_fraction": 0.15,
    "occlusion_solidity": 0.90,
    "min_page_area_fraction": 0.01,
    "min_page_long_side_px": 600,
    "box_margin": RERUN_MARGIN,
    "two_page_min_ratio": 0.5,
    "two_page_min_overlap": 0.5,
    "component_min_area_fraction": 0.01,
    "split_min_inside": 0.5,
    "split_min_precision": 0.5,
}
SPLIT_CODES = ("ambiguous_page_region", "two_page_candidate")
SIDES = ("top", "right", "bottom", "left")
Box = tuple[int, int, int, int]
Bounds = tuple[float, float, float, float]


def _input_error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_region_input", message)


# ----------------------------------------------------------------------------- masks


def label_components(binary: NDArray[np.bool_]) -> tuple[NDArray[np.int32], list[int]]:
    """8-connected components of a boolean grid, labelled 1.. in row-major order of their
    first cell; returns the labels and each component's cell count."""
    height, width = binary.shape
    labels = np.zeros(binary.shape, dtype=np.int32)
    sizes: list[int] = []
    for y, x in zip(*np.nonzero(binary), strict=True):
        if labels[y, x]:
            continue
        label = len(sizes) + 1
        labels[y, x] = label
        queue = deque([(int(y), int(x))])
        count = 0
        while queue:
            yy, xx = queue.popleft()
            count += 1
            for ny in range(max(0, yy - 1), min(height, yy + 2)):
                for nx in range(max(0, xx - 1), min(width, xx + 2)):
                    if binary[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = label
                        queue.append((ny, nx))
        sizes.append(count)
    return labels, sizes


def mask_bounds(mask: NDArray[np.bool_], source_size: tuple[int, int]) -> Bounds:
    """Source pixel edges (left, top, right, bottom) of the cells a grid mask covers."""
    height, width = source_size
    rows, cols = mask.shape
    yy, xx = np.nonzero(mask)
    return (
        float(xx.min()) / cols * width,
        float(yy.min()) / rows * height,
        (float(xx.max()) + 1) / cols * width,
        (float(yy.max()) + 1) / rows * height,
    )


def touched_sides(mask: NDArray[np.bool_]) -> tuple[str, ...]:
    edges = (mask[0], mask[:, -1], mask[-1], mask[:, 0])
    return tuple(side for side, edge in zip(SIDES, edges, strict=True) if edge.any())


@dataclass(frozen=True, slots=True)
class Component:
    """One connected component of the gate mask."""

    bounds: Bounds  # source pixel edges
    cells: int
    area_fraction: float
    touches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [float(v) for v in self.bounds],
            "cells": self.cells,
            "area_fraction": self.area_fraction,
            "touches": list(self.touches),
        }


def components(
    binary: NDArray[np.bool_], source_size: tuple[int, int]
) -> tuple[NDArray[np.int32], list[Component]]:
    """Labels and components of a boolean grid, largest first (ties in label order)."""
    labels, sizes = label_components(binary)
    items = []
    for index, size in enumerate(sizes, start=1):
        cells = labels == index
        items.append(
            (
                -size,
                index,
                Component(
                    mask_bounds(cells, source_size),
                    size,
                    size / binary.size,
                    touched_sides(cells),
                ),
            )
        )
    items.sort(key=lambda item: item[:2])
    order = [index for _, index, _ in items]
    relabelled = np.zeros_like(labels)
    for rank, index in enumerate(order, start=1):
        relabelled[labels == index] = rank
    return relabelled, [item[2] for item in items]


def expand_box(bounds: Bounds, source_size: tuple[int, int], margin: float = RERUN_MARGIN) -> Box:
    """Whole-pixel crop box around ``bounds`` plus ``margin`` of their extent per side,
    clipped to the image (``dewarp.predict.crop_box`` of the bounds' corners)."""
    left, top, right, bottom = bounds
    corners = np.array([[left, top], [right, bottom]], dtype=np.float64)
    return crop_box(corners, source_size, margin)


def union_box(first: Box, second: Box) -> Box:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def cell_centers(mask: NDArray[np.bool_], source_size: tuple[int, int]) -> NDArray[np.float64]:
    """Source pixel positions (N, 2) of the centers of the mask's cells."""
    height, width = source_size
    rows, cols = mask.shape
    yy, xx = np.nonzero(mask)
    return np.stack([(xx + 0.5) / cols * width, (yy + 0.5) / rows * height], axis=-1)


def cell_corners(mask: NDArray[np.bool_], source_size: tuple[int, int]) -> NDArray[np.float64]:
    """Source pixel positions of the corners of the mask's cells (no duplicates)."""
    height, width = source_size
    rows, cols = mask.shape
    grown = np.zeros((rows + 1, cols + 1), dtype=bool)
    grown[:-1, :-1] |= mask
    grown[1:, :-1] |= mask
    grown[:-1, 1:] |= mask
    grown[1:, 1:] |= mask
    yy, xx = np.nonzero(grown)
    return np.stack([xx / cols * width, yy / rows * height], axis=-1)


def convex_hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Counter-clockwise (in x right, y up terms) hull vertices of (N, 2) points."""
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) < 3:
        return unique

    def cross(o: NDArray, a: NDArray, b: NDArray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[NDArray] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[NDArray] = []
    for point in unique[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(polygon: NDArray[np.float64]) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def fold_angle(angle_deg: float) -> float:
    """``angle_deg`` modulo 90 into (-45, 45]."""
    folded = (float(angle_deg) + 45.0) % 90.0 - 45.0
    return 45.0 if folded == -45.0 else folded


def min_area_rect_angle(points: NDArray[np.float64]) -> float:
    """Angle in (-45, 45] of the minimum-area rectangle around ``points`` (source pixels,
    y down, so positive turns clockwise on screen); 0 for fewer than 3 hull points."""
    hull = convex_hull(points)
    if len(hull) < 3:
        return 0.0
    best: tuple[float, float] | None = None
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        edge = end - start
        length = float(np.hypot(*edge))
        if length == 0:
            continue
        unit = edge / length
        normal = np.array([-unit[1], unit[0]])
        along, across = hull @ unit, hull @ normal
        area = float((along.max() - along.min()) * (across.max() - across.min()))
        angle = float(np.degrees(np.arctan2(unit[1], unit[0])))
        if best is None or area < best[0] - 1e-9 * max(area, 1.0):
            best = (area, angle)
    return 0.0 if best is None else fold_angle(best[1])


def pca_angle(points: NDArray[np.float64]) -> float:
    """Angle in (-45, 45] of the principal axis of ``points``; 0 when it is undefined."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    if not np.isfinite(values).all() or values[-1] <= 0:
        return 0.0
    axis = vectors[:, -1]
    return fold_angle(float(np.degrees(np.arctan2(axis[1], axis[0]))))


def solidity(mask: NDArray[np.bool_], source_size: tuple[int, int]) -> float:
    """Mask area over the area of its convex hull (cell corners), in source pixels."""
    height, width = source_size
    rows, cols = mask.shape
    area = float(mask.sum()) * (height / rows) * (width / cols)
    hull = convex_hull(cell_corners(mask, source_size))
    hull_area = polygon_area(hull) if len(hull) >= 3 else 0.0
    return 1.0 if hull_area <= 0 else min(1.0, area / hull_area)


def outline_polygon(grid: NDArray) -> NDArray[np.float64]:
    """Boundary nodes of a grid in order around the page (flattener values)."""
    array = np.asarray(grid, dtype=np.float64)
    return np.concatenate((array[0], array[1:, -1], array[-1, -2::-1], array[-2:0:-1, 0]))


def rasterize_polygon(polygon: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Cells of a (rows, cols) grid over the whole image whose centers lie inside the
    polygon (flattener values, even-odd rule)."""
    rows, cols = shape
    x = (2.0 * (np.arange(cols) + 0.5) / cols - 1.0)[None, :]
    y = (2.0 * (np.arange(rows) + 0.5) / rows - 1.0)[:, None]
    inside = np.zeros(shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if y0 == y1:
            continue
        crosses = (y0 > y) != (y1 > y)
        at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < at)
    return inside


def outline_mask_precision(grid: NDArray, mask: NDArray[np.bool_]) -> float:
    """Share of the grid's outline polygon (rasterized on the mask grid) inside the mask;
    0 when the polygon covers no cell center."""
    polygon = rasterize_polygon(outline_polygon(grid), mask.shape)
    total = int(polygon.sum())
    return 0.0 if total == 0 else float((polygon & mask).sum() / total)


def grid_coverage(grid: NDArray, mask: NDArray[np.bool_]) -> float:
    """``regions.geometry.page_coverage``: share of mask cells inside the grid's outline."""
    return page_coverage(np.asarray(grid, dtype=np.float32), mask)


def region_excess(
    grid: NDArray, mask: NDArray[np.bool_], source_size: tuple[int, int]
) -> dict[str, float]:
    """Mask cells outside the grid's outline per page side (the grid's top row, last
    column, bottom row and first column), each cell counted at the side of its nearest
    boundary node in source pixels, as a share of the outline's cells. A neighbouring page
    or the paper under the page adds to some sides; a grid that holds only a piece of the
    page leaves mask on every side."""
    height, width = source_size
    rows, cols = mask.shape
    inside = rasterize_polygon(outline_polygon(grid), mask.shape)
    area = int(inside.sum())
    if area == 0:
        return dict.fromkeys(SIDES, 1.0)
    yy, xx = np.nonzero(mask & ~inside)
    if len(yy) == 0:
        return dict.fromkeys(SIDES, 0.0)
    array = np.asarray(grid, dtype=np.float64)
    edges = (array[0], array[:, -1], array[-1], array[:, 0])  # top, right, bottom, left
    scale = np.array([width / 2.0, height / 2.0])
    nodes = np.concatenate(edges) * scale
    side = np.concatenate([np.full(len(edge), index) for index, edge in enumerate(edges)])
    cells = np.stack([2.0 * (xx + 0.5) / cols - 1.0, 2.0 * (yy + 0.5) / rows - 1.0], -1) * scale
    counts = np.zeros(len(SIDES))
    for start in range(0, len(cells), 4096):
        chunk = cells[start : start + 4096]
        distance = ((chunk[:, None, :] - nodes[None]) ** 2).sum(-1)
        counts += np.bincount(side[distance.argmin(axis=1)], minlength=len(SIDES))
    return {name: float(count / area) for name, count in zip(SIDES, counts, strict=True)}


def grid_angle(grid: NDArray, source_size: tuple[int, int]) -> tuple[float, int, str | None]:
    """(theta, k, reason): the page angle of a grid split as theta + 90 k; (0, 0, reason)
    when ``estimate_page_angle`` finds it unusable."""
    angle, why = estimate_page_angle(grid, source_size)
    if angle is None:
        return 0.0, 0, why
    theta, k = split_angle(angle)
    return theta, k, None


def outline_box(grid: NDArray, source_size: tuple[int, int]) -> Box | None:
    """``crop_box`` of the grid's outline; ``None`` when it is not finite or empty."""
    points = outline_points(grid, source_size)
    if not np.isfinite(points).all():
        return None
    box = crop_box(points, source_size)
    return box if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 else None


def cut_off_sides(
    mask: NDArray[np.bool_],
    pass0: NDArray | None = None,
    fraction: float = THRESHOLDS["cut_off_side_fraction"],
) -> list[str]:
    """Image sides along which the mask runs for more than ``fraction`` of the side and,
    when ``pass0`` is given, past which its outline reaches."""
    edges = {"top": mask[0], "right": mask[:, -1], "bottom": mask[-1], "left": mask[:, 0]}
    sides = [side for side, edge in edges.items() if edge.mean() > fraction]
    if pass0 is None or not sides:
        return sides
    outline = outline_polygon(pass0)
    beyond = {
        "top": (outline[:, 1] < -1).any(),
        "right": (outline[:, 0] > 1).any(),
        "bottom": (outline[:, 1] > 1).any(),
        "left": (outline[:, 0] < -1).any(),
    }
    return [side for side in sides if beyond[side]]


def two_page_gutter(items: list[Component]) -> list[float] | None:
    """The gap (left, right source pixels) between the two largest components when they
    lie side by side with similar size; ``None`` otherwise."""
    if len(items) < 2:
        return None
    first, second = items[0], items[1]
    if second.cells < THRESHOLDS["two_page_min_ratio"] * first.cells:
        return None
    left, right = sorted((first, second), key=lambda item: item.bounds[0])
    overlap = min(left.bounds[3], right.bounds[3]) - max(left.bounds[1], right.bounds[1])
    shorter = min(left.bounds[3] - left.bounds[1], right.bounds[3] - right.bounds[1])
    if shorter <= 0 or overlap < THRESHOLDS["two_page_min_overlap"] * shorter:
        return None
    if right.bounds[0] < left.bounds[2]:
        return None
    return [float(left.bounds[2]), float(right.bounds[0])]


# ----------------------------------------------------------------------------- result


@dataclass(eq=False)
class Localization:
    """The detect stage's answer (see the module docstring).

    ``box`` is ``None`` when ``failure`` ends the scan. ``method`` names where the box came
    from: ``gate``, ``fills_frame``, ``outline``, ``union``, ``mask`` (the consensus
    branches), ``outline_unverified`` (no gate), ``manual_region`` or ``manual_quad``.
    ``quarter`` is set by a manual quad. ``quad`` holds the page corners (source pixels,
    reading order) when they are known for sure (a manual quad).
    """

    box: Box | None
    angle_deg: float
    method: str
    mask: NDArray[np.bool_] | None = None
    reasons: list[Reason] = field(default_factory=list)
    failure: Reason | None = None
    components: list[Component] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    pass0: NDArray[np.float64] | None = None
    quarter: int | None = None
    quad: NDArray[np.float64] | None = None
    provenance: str = "auto"

    def record(self) -> dict[str, Any]:
        return {
            "box": None if self.box is None else list(self.box),
            "angle_deg": self.angle_deg,
            "method": self.method,
            "quarter": self.quarter,
            "quad": None if self.quad is None else self.quad.tolist(),
            "provenance": self.provenance,
            "components": [item.to_dict() for item in self.components],
            "metrics": self.metrics,
            "reasons": [item.to_dict() for item in self.reasons],
            "failure": None if self.failure is None else self.failure.to_dict(),
        }


def _full_scale(source_size: tuple[int, int], full_size: tuple[int, int] | None) -> float:
    if full_size is None:
        return 1.0
    return max(full_size[0] / source_size[0], full_size[1] / source_size[1])


def _failure(code: str, candidates: list[Component], **detail: Any) -> Reason:
    boxes = [item.to_dict()["box"] for item in candidates]
    return reason(code, boxes=boxes, **detail)


def _held_pieces(
    grid: NDArray,
    labels: NDArray[np.int32],
    items: list[Component],
    source_size: tuple[int, int],
) -> list[int]:
    """Labels of the gate pieces (at least ``component_min_area_fraction`` of the photo)
    with at least ``split_min_inside`` of their cells inside the outline of ``grid``;
    none when the outline is not usable."""
    if outline_box(grid, source_size) is None:
        return []
    inside = rasterize_polygon(outline_polygon(grid), labels.shape)
    held = []
    for index, item in enumerate(items, start=1):
        if item.area_fraction < THRESHOLDS["component_min_area_fraction"]:
            continue
        share = float((inside & (labels == index)).sum()) / item.cells
        if share >= THRESHOLDS["split_min_inside"]:
            held.append(index)
    return held


def _gate_failure(
    error: FlattenerError,
    probability: NDArray,
    source_size: tuple[int, int],
    config: GateConfig,
) -> Localization | None:
    """The localization for a gate rejection, or ``None`` for weak scores (the consensus
    rule decides those)."""
    binary = probability >= config.region.threshold
    _, items = components(binary, source_size)
    shown = [
        item for item in items if item.area_fraction >= THRESHOLDS["component_min_area_fraction"]
    ]
    metrics = {"gate_error": error.reason_code, "detection": error.metadata}
    if error.reason_code == "page_not_found":
        failure = _failure("page_not_found", shown)
    elif error.reason_code == "page_area_rejected":
        failure = _failure("page_too_small", shown, area_fraction=items[0].area_fraction)
    elif error.reason_code == "ambiguous_page_region":
        gutter = two_page_gutter(items)
        if gutter is not None:
            failure = _failure("two_page_candidate", shown, gutter=gutter)
        else:
            failure = _failure("ambiguous_page_region", shown)
    else:
        return None
    return Localization(None, 0.0, "gate", None, [], failure, items, metrics)


def locate_page(
    source_size: tuple[int, int],
    *,
    probability: NDArray | None = None,
    pass0: Callable[[], NDArray[np.float64]] | None = None,
    gate_config: GateConfig | None = None,
    full_size: tuple[int, int] | None = None,
) -> Localization:
    """Locate the page (see the module docstring).

    ``probability`` is the gate's grid over the whole image (``None``: no gate).
    ``pass0()`` returns G0 and is called only when a rule needs it. ``full_size`` is the
    full-resolution size of a draft-decoded source, for the long-side rule."""
    config = gate_config or GateConfig()
    height, width = source_size
    scale = _full_scale(source_size, full_size)
    if probability is None:
        if pass0 is None:
            raise _input_error("locate_page needs a gate probability or a pass-0 grid")
        grid = pass0()
        box = outline_box(grid, source_size)
        theta, k, why = grid_angle(grid, source_size)
        metrics = {"outline_quarter": k, "outline_angle_unusable": why}
        if box is None:
            failure = reason("page_not_found", boxes=[])
            return Localization(None, 0.0, "outline_unverified", None, [], failure, [], metrics)
        reasons = [reason("unverified_region", method="outline")]
        return Localization(
            box, theta, "outline_unverified", None, reasons, None, [], metrics, grid
        )
    binary = probability >= config.region.threshold
    labels, items = components(binary, source_size)
    split: Reason | None = None  # the gate split the page into pieces; G0 decides
    grid = None
    held: list[int] = []
    precision = 0.0
    try:
        gate = gate_page_region(probability, source_size, config=config)
    except FlattenerError as error:
        if error.status != Status.MODEL_ERROR:
            raise
        rejected = _gate_failure(error, probability, source_size, config)
        gate = None
        if rejected is not None:
            if rejected.failure.code not in SPLIT_CODES or pass0 is None:
                return rejected
            grid = pass0()
            held = _held_pieces(grid, labels, items, source_size)
            precision = outline_mask_precision(grid, np.isin(labels, held)) if held else 0.0
            rejected.metrics["split_precision"] = precision
            if precision < THRESHOLDS["split_min_precision"]:
                rejected.pass0 = grid
                return rejected
            split = rejected.failure
    if gate is not None:
        mask = gate.estimate.mask
        detection = dict(gate.metadata)
    elif split is not None:
        mask = np.isin(labels, held)
        detection = {"gate_error": split.code, "held_pieces": held, "split_precision": precision}
    else:
        mask = labels == 1
        detection = {"gate_error": "uncertain_page_region"}
    area = float(mask.mean())
    bounds = mask_bounds(mask, source_size)
    long_side = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * scale
    if split is not None:
        selected = [items[index - 1] for index in held]
    else:
        count = int(mask.sum())
        selected = [item for item in items if item.cells == count and item.bounds == bounds]
    others = [item for item in items if not any(item is chosen for chosen in selected)]
    metrics: dict[str, Any] = {
        "area_fraction": area,
        "page_long_side_px": long_side,
        "dominance": detection.get("detection", {}).get("dominance"),
        "mean_probability": detection.get("detection", {}).get("mean_probability"),
        "confident": bool(gate is not None and gate.confident),
        "gate": detection,
    }
    shown = [
        item for item in items if item.area_fraction >= THRESHOLDS["component_min_area_fraction"]
    ]
    if area < THRESHOLDS["min_page_area_fraction"] or (
        long_side < THRESHOLDS["min_page_long_side_px"]
        and not (gate is not None and gate.fills_frame)
    ):
        failure = _failure("page_too_small", shown, area_fraction=area, long_side_px=long_side)
        return Localization(None, 0.0, "gate", mask, [], failure, items, metrics, grid)
    reasons: list[Reason] = []
    if split is not None:
        # The scan goes on with G0's outline; the split stays as a warning.
        reasons.append(
            reason(split.code, status=Status.SUCCESS, severity=Severity.WARNING, **split.detail)
        )
    neighbors = [item for item in others if item.touches]
    if neighbors:
        reasons.append(
            reason("neighbor_page_partial", boxes=[item.to_dict()["box"] for item in neighbors])
        )
    if gate is not None and gate.fills_frame:
        reasons.insert(0, reason("page_fills_frame", touched=list(gate.touched_borders)))
        return Localization(
            (0, 0, width, height), 0.0, "fills_frame", mask, reasons, None, items, metrics
        )
    mask_box = expand_box(bounds, source_size)
    if split is not None:
        theta, k, why = grid_angle(grid, source_size)
        metrics.update(outline_quarter=k, outline_angle_unusable=why)
        box, method = outline_box(grid, source_size), "outline"
    elif gate is not None and gate.confident:
        box, theta, method = mask_box, min_area_rect_angle(cell_centers(mask, source_size)), "gate"
    else:
        if pass0 is None:
            box, theta, method = mask_box, pca_angle(cell_centers(mask, source_size)), "mask"
        else:
            grid = pass0()
            coverage = grid_coverage(grid, mask)
            precision = outline_mask_precision(grid, mask)
            outline = outline_box(grid, source_size)
            theta0, k0, why = grid_angle(grid, source_size)
            metrics.update(
                consensus_coverage=coverage,
                consensus_precision=precision,
                outline_quarter=k0,
                outline_angle_unusable=why,
            )
            good_coverage = coverage >= THRESHOLDS["consensus_coverage"]
            good_precision = precision >= THRESHOLDS["consensus_precision"]
            if outline is None or not good_precision:
                box, theta, method = mask_box, pca_angle(cell_centers(mask, source_size)), "mask"
            elif good_coverage:
                box, theta, method = outline, theta0, "outline"
            else:
                box, theta, method = union_box(outline, mask_box), theta0, "union"
    sides = cut_off_sides(mask, grid)
    if sides:
        reasons.append(
            reason(
                "page_cut_off", sides=sides, evidence="mask" if grid is None else "mask_and_outline"
            )
        )
    # The gap between the pieces of a split gate is no occlusion: its largest piece counts.
    page_solidity = solidity(mask if split is None else labels == held[0], source_size)
    metrics["solidity"] = page_solidity
    if page_solidity < THRESHOLDS["occlusion_solidity"]:
        reasons.append(reason("possible_occlusion", solidity=page_solidity))
    return Localization(box, theta, method, mask, reasons, None, items, metrics, grid)


def alternative_box(
    located: Localization,
    source_size: tuple[int, int],
    pass0: Callable[[], NDArray[np.float64]] | None,
) -> tuple[Box, float, str] | None:
    """The other localizer's box and angle (the ``other_box`` candidate): the outline for a
    gate or mask box, the mask for an outline or union box; ``None`` when there is no other
    one."""
    if located.box is None or located.provenance == "manual":
        return None
    if located.method in ("gate", "fills_frame", "mask") and pass0 is not None:
        grid = located.pass0 if located.pass0 is not None else pass0()
        box = outline_box(grid, source_size)
        if box is None:
            return None
        theta, _, _ = grid_angle(grid, source_size)
        return box, theta, "outline"
    if located.method in ("outline", "union") and located.mask is not None:
        bounds = mask_bounds(located.mask, source_size)
        theta = min_area_rect_angle(cell_centers(located.mask, source_size))
        return expand_box(bounds, source_size), theta, "gate"
    return None


# ----------------------------------------------------------------------------- overrides


def region_bounds(region: Any, source_size: tuple[int, int]) -> Bounds:
    """A manual page region (``PageRegion`` or left, top, right, bottom) inside the image."""
    if isinstance(region, PageRegion):
        bounds = (region.left, region.top, region.right, region.bottom)
    elif isinstance(region, (tuple, list)) and len(region) == 4:
        try:
            bounds = tuple(float(v) for v in region)
        except (TypeError, ValueError):
            raise _input_error(f"region must be four numbers, got {region!r}") from None
    else:
        raise _input_error(
            f"region must be a PageRegion or (left, top, right, bottom), got {region!r}"
        )
    height, width = source_size
    left, top, right, bottom = bounds
    if not all(np.isfinite(bounds)) or not (
        0 <= left < right <= width and 0 <= top < bottom <= height
    ):
        raise _input_error(f"region {bounds} must lie inside the image ({width} x {height})")
    return left, top, right, bottom


def check_quad(quad: Any, source_size: tuple[int, int]) -> NDArray[np.float64]:
    """A manual quad as (4, 2) source pixels in reading order: finite, convex, not
    self-crossing, clockwise on screen and at least partly inside the image. The same
    corners given counter-clockwise describe a mirrored page, so they are refused."""
    array = np.asarray(quad, dtype=np.float64)
    if array.shape == (8,):
        array = array.reshape(4, 2)
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise _input_error("quad must be four finite (x, y) points")
    edges = np.roll(array, -1, axis=0) - array
    turns = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    if (turns < 0).all():
        raise _input_error(
            "quad goes counter-clockwise; give top-left, top-right, bottom-right, bottom-left"
        )
    if not (turns > 0).all():
        raise _input_error("quad must be convex, in order around the page")
    height, width = source_size
    low, high = array.min(axis=0), array.max(axis=0)
    if high[0] <= 0 or high[1] <= 0 or low[0] >= width or low[1] >= height:
        raise _input_error("quad lies outside the image")
    return array


def quad_angle(quad: NDArray[np.float64]) -> float:
    """Direction in degrees of the page's top and bottom edges (top-left to top-right,
    bottom-left to bottom-right), clockwise positive."""
    top = quad[1] - quad[0]
    bottom = quad[2] - quad[3]
    units = [edge / np.hypot(*edge) for edge in (top, bottom) if np.hypot(*edge) > 0]
    if not units:
        return 0.0
    mean = np.mean(units, axis=0)
    return float(np.degrees(np.arctan2(mean[1], mean[0])))


def manual_region(region: Any, source_size: tuple[int, int]) -> Localization:
    """A user's page region: the box is the region plus ``RERUN_MARGIN`` per side."""
    bounds = region_bounds(region, source_size)
    box = expand_box(bounds, source_size)
    metrics = {"region": list(bounds)}
    return Localization(box, 0.0, "manual_region", metrics=metrics, provenance="manual")


def manual_quad(
    quad: Any, source_size: tuple[int, int], mask_shape: tuple[int, int] = (192, 144)
) -> Localization:
    """A user's page quad in reading order: box, angle theta and quarter k from its edges,
    and the quad rasterized on a ``mask_shape`` grid for the coverage check."""
    array = check_quad(quad, source_size)
    height, width = source_size
    low, high = array.min(axis=0), array.max(axis=0)
    bounds = (
        float(max(0.0, low[0])),
        float(max(0.0, low[1])),
        float(min(width, high[0])),
        float(min(height, high[1])),
    )
    box = expand_box(bounds, source_size)
    theta, k = split_angle(quad_angle(array))
    values = array / np.array([width, height]) * 2.0 - 1.0
    mask = rasterize_polygon(values, mask_shape)
    metrics = {"quad_angle_deg": quad_angle(array)}
    return Localization(
        box,
        theta,
        "manual_quad",
        mask if mask.any() else None,
        metrics=metrics,
        quarter=k,
        quad=array,
        provenance="manual",
    )
