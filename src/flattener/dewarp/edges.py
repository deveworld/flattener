"""Border snapping: move a grid's outline onto the page edge seen in the photo.

The model places the outline from a 712x488 view, so on a large photo its border can sit
a few source pixels off the paper edge; outside the page that leaves a rim of background
in the output. ``snap_border`` looks at the photo itself: for every boundary node it reads
a profile along the outward normal (``PROFILE_REACH`` of the page diagonal either way,
averaged over ``PROFILE_WIDTH`` pixels along the border), scores each sample by its colour
distance from the paper colour of that side (the median of the profiles' inner part), and
puts the edge at the strongest rise of that distance. A side whose rises are weak (less
than ``MIN_CONTRAST`` of the paper colour's scale) or scattered keeps its nodes. Offsets
are median-filtered along the side and bounded by the reach; the corner nodes take the
offsets of both sides. The displacement of the border is carried into the interior by
transfinite (Coons) interpolation, so the grid stays smooth and keeps its interior shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

PROFILE_REACH = 0.01  # of the page diagonal, each way along the normal (SmartDoc dev)
PROFILE_SAMPLES = 41
PROFILE_WIDTH = 5  # samples across, along the border
MIN_CONTRAST = 0.06  # least rise of colour distance (0-1 RGB) at the edge
MEDIAN_WINDOW = 5
SIDES = ("top", "bottom", "left", "right")


def _bilinear(image: NDArray[np.float32], points: NDArray[np.float64]) -> NDArray[np.float32]:
    """RGB at pixel-edge points (x, y) of ``image``, bilinear, clamped at the border."""
    height, width = image.shape[:2]
    x = np.clip(points[..., 0] - 0.5, 0, width - 1)
    y = np.clip(points[..., 1] - 0.5, 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _side(grid_px: NDArray[np.float64], side: str) -> tuple[NDArray, NDArray]:
    """The side's nodes and the nodes one step inside it."""
    if side == "top":
        return grid_px[0], grid_px[1]
    if side == "bottom":
        return grid_px[-1], grid_px[-2]
    if side == "left":
        return grid_px[:, 0], grid_px[:, 1]
    return grid_px[:, -1], grid_px[:, -2]


def _median_filter(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def side_offsets(
    image: NDArray[np.float32], nodes: NDArray, inner: NDArray, reach: float
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Offsets (source pixels, outward positive) of one side's nodes to the paper edge."""
    tangent = np.gradient(nodes, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    outward = nodes - inner
    flip = np.sign(np.sum(normal * outward, axis=1, keepdims=True))
    normal *= np.where(flip == 0, 1, flip)
    steps = np.linspace(-reach, reach, PROFILE_SAMPLES)
    across = np.linspace(-(PROFILE_WIDTH - 1) / 2, (PROFILE_WIDTH - 1) / 2, PROFILE_WIDTH)
    points = (
        nodes[:, None, None, :]
        + steps[None, :, None, None] * normal[:, None, None, :]
        + across[None, None, :, None] * tangent[:, None, None, :]
    )
    profiles = _bilinear(image, points).mean(axis=2)  # (nodes, samples, 3)
    inner_part = profiles[:, : PROFILE_SAMPLES // 4]
    paper = np.median(inner_part.reshape(-1, 3), axis=0)
    distance = np.linalg.norm(profiles - paper, axis=2)  # (nodes, samples)
    rise = np.diff(distance, axis=1)
    kernel = np.ones(3) / 3
    rise = np.stack([np.convolve(row, kernel, mode="same") for row in rise])
    best = np.argmax(rise, axis=1)
    strength = rise[np.arange(len(best)), best] * 3  # over three samples
    offsets = (steps[best] + steps[best + 1]) / 2
    strong = strength >= MIN_CONTRAST
    info = {
        "strong_share": float(strong.mean()),
        "median_strength": float(np.median(strength)),
        "paper": [round(float(v), 3) for v in paper],
    }
    if strong.mean() < 0.5:
        info["kept"] = "weak_edge"
        return np.zeros(len(nodes)), info
    offsets = np.where(strong, offsets, np.nan)
    fill = np.nanmedian(offsets)
    offsets = np.where(np.isnan(offsets), fill, offsets)
    offsets = _median_filter(offsets, MEDIAN_WINDOW)
    info["median_offset_px"] = float(np.median(offsets))
    return np.clip(offsets, -reach, reach) * 1.0, info | {"normal": normal}


def coons(top, bottom, left, right) -> NDArray[np.float64]:
    """Transfinite interpolation of four boundary curves (Gw,2), (Gw,2), (Gh,2), (Gh,2)."""
    rows, cols = len(left), len(top)
    v = np.linspace(0, 1, rows)[:, None, None]
    u = np.linspace(0, 1, cols)[None, :, None]
    ruled = (1 - v) * top[None] + v * bottom[None] + (1 - u) * left[:, None] + u * right[:, None]
    corners = (
        (1 - u) * (1 - v) * top[0] + u * (1 - v) * top[-1]
        + (1 - u) * v * bottom[0] + u * v * bottom[-1]
    )  # fmt: skip
    return ruled - corners


def snap_border(image: NDArray[np.uint8], grid: NDArray) -> tuple[NDArray[np.float64], dict]:
    """``grid`` (flattener values of ``image``) with its border on the photo's paper edge,
    and a record per side (see the module docstring)."""
    height, width = image.shape[:2]
    scale = np.array([width, height], dtype=np.float64)
    grid_px = (np.asarray(grid, dtype=np.float64) + 1.0) / 2.0 * scale
    record: dict[str, Any] = {}
    if not np.isfinite(grid_px).all():
        return np.asarray(grid, dtype=np.float64), {"skipped": "non_finite_grid"}
    diagonal = float(np.hypot(*(grid_px[-1, -1] - grid_px[0, 0])))
    reach = PROFILE_REACH * diagonal
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    moves = {}
    for side in SIDES:
        nodes, inner = _side(grid_px, side)
        offsets, info = side_offsets(pixels, nodes, inner, reach)
        normal = info.pop("normal", None)
        moves[side] = np.zeros_like(nodes) if normal is None else offsets[:, None] * normal
        record[side] = info
    # A corner node belongs to two sides and moves by both: each side's curve takes the
    # neighbouring sides' end moves, blended linearly along it, so the four curves meet.
    top, bottom, left, right = (moves[side] for side in SIDES)
    u = np.linspace(0, 1, len(top))[:, None]
    v = np.linspace(0, 1, len(left))[:, None]
    top_c = top + (1 - u) * left[0] + u * right[0]
    bottom_c = bottom + (1 - u) * left[-1] + u * right[-1]
    left_c = left + (1 - v) * top[0] + v * bottom[0]
    right_c = right + (1 - v) * top[-1] + v * bottom[-1]
    snapped = grid_px + coons(top_c, bottom_c, left_c, right_c)
    record["reach_px"] = round(reach, 2)
    return 2.0 * snapped / scale - 1.0, record
