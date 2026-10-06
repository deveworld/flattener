"""OrientNet-1: the quarter-turn classifier of the single-page scanner.

Label ``k`` counts the clockwise quarter turns of the text-up direction in the input
view. ``numpy.rot90(view, k)`` turns the page upright, and the scanner's view angle is
``phi = theta + 90 k`` (clockwise positive). Pages without readable text train toward a
uniform distribution, so a low top probability means the page gives no evidence.

Inputs (``page_views``), all grayscale:

* the global view: the page view letterboxed into ``global_size`` squared and padded
  with ``pad_value``;
* ``patches`` patches of ``patch_size`` squared, cut from the view scaled to a long side
  of ``page_long_side``. Their centres are the centres of the ``energy_grid`` squared
  cells with the highest text energy (standard deviation of the gradient magnitude on
  the global view); the outer ring of cells is left out. A quarter turn of the view
  permutes the cells, so the selection turns with the page.

Every view pixel is one bilinear lookup in the source (or in a box-reduced copy of it)
through one map, in the flattener convention (``align_corners=False``, +-1 on the
outer pixel edges). ``ViewGeometry`` places the view in the source: the whole image,
or a turned box when the scanner samples its deskewed crop from the photo.

Network: one depthwise-separable trunk (hardswish, squeeze-excitation in the last
blocks) shared by all views. A 1x1 saliency logit weights a softmax pool over every
position of every view, and an MLP gives 4 logits. About 0.44M parameters and
0.34 GFLOPs per page (``model_cost``).

Decision (``decide``): probabilities are ``softmax(logits / temperature)``; ``k`` is
accepted when its probability reaches ``threshold`` (0.9 until one is calibrated on a
real-photo dev split, ``operating_threshold``), and the result carries
``orientation_uncertain`` otherwise.

CLI: ``python -m flattener.scan.orient predict|export|cost``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .. import ops
from ..contracts import FlattenerError, Status
from ..dewarp.checkpoint import atomic_write, code_version, read_checkpoint
from ..runtime import OnnxModel

MODEL_ID = "orientnet-1"
CHECKPOINT_SCHEMA = "flattener-orient-checkpoint-1"
EXPORT_SCHEMA = "flattener-orient-onnx-1"
VIEWS_VERSION = "orient-views-1"
CLASSES = 4
DEFAULT_TEMPERATURE = 1.0
DEFAULT_THRESHOLD = 0.9
UNCERTAIN = "orientation_uncertain"
# (out channels, stride, depthwise kernel, squeeze-excitation) after a 3x3 stride-2 stem.
BLOCKS = (
    (32, 1, 3, False),
    (48, 2, 3, False),
    (48, 1, 3, False),
    (96, 2, 3, False),
    (96, 1, 3, False),
    (192, 2, 5, False),
    (192, 1, 5, False),
    (192, 1, 5, True),
    (320, 2, 5, True),
    (320, 1, 5, True),
)


def _error(status: Status, reason: str, message: str) -> FlattenerError:
    return FlattenerError(status, reason, message)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True, slots=True)
class OrientConfig:
    """Input views and network widths."""

    global_size: int = 256
    patch_size: int = 128
    patches: int = 4
    page_long_side: int = 1024
    energy_grid: int = 8
    pad_value: int = 128
    stem: int = 16
    blocks: tuple[tuple[int, int, int, bool], ...] = BLOCKS
    hidden: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        blocks = tuple(tuple(block) for block in self.blocks)
        object.__setattr__(self, "blocks", blocks)
        sizes = ("global_size", "patch_size", "page_long_side", "energy_grid", "stem", "hidden")
        if not all(_positive_int(getattr(self, name)) for name in sizes):
            raise ValueError(f"{', '.join(sizes)} must be positive integers")
        if isinstance(self.patches, bool) or not isinstance(self.patches, int) or self.patches < 0:
            raise ValueError("patches must be an integer >= 0")
        if self.energy_grid < 3 or self.patches > (self.energy_grid - 2) ** 2:
            raise ValueError("energy_grid must be >= 3 with patches <= (energy_grid - 2)^2")
        if not 0 <= self.pad_value <= 255 or not 0 <= self.dropout < 1:
            raise ValueError("pad_value must be in [0, 255] and dropout in [0, 1)")
        for block in blocks:
            if (
                len(block) != 4
                or not all(_positive_int(v) for v in block[:3])
                or block[1] not in (1, 2)
                or block[2] % 2 == 0
                or not isinstance(block[3], bool)
            ):
                raise ValueError(f"bad block {block}: (channels, stride 1|2, odd kernel, se)")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["blocks"] = [list(block) for block in self.blocks]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrientConfig:
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown OrientConfig fields {sorted(unknown)}")
        values = dict(data)
        if "blocks" in values:
            values["blocks"] = tuple(tuple(block) for block in values["blocks"])
        return cls(**values)


# ----------------------------------------------------------------------------- network


# ----------------------------------------------------------------------------- views


def clockwise(degrees: float) -> NDArray[np.float64]:
    """Rotation that turns a vector clockwise on screen (x right, y down)."""
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


@dataclass(frozen=True, slots=True)
class ViewGeometry:
    """Where a page view samples its source.

    ``size`` is the view's (height, width) in view pixels. ``matrix`` (2 x 3) maps a view
    point (x, y), in pixels from the view's top-left edge, to source pixels from the
    source's top-left edge. ``warp`` optionally adds a smooth displacement field
    (2, h, w) in normalized source units, looked up at the normalized source point
    (bilinear, ``align_corners=True`` over [-1, 1]).
    """

    size: tuple[float, float]
    matrix: NDArray[np.float64]
    warp: NDArray[np.float32] | None = None

    @classmethod
    def identity(cls, height: int, width: int) -> ViewGeometry:
        return cls((float(height), float(width)), np.array([[1.0, 0, 0], [0, 1.0, 0]]))

    @classmethod
    def turned(
        cls,
        center: tuple[float, float],
        box_size: tuple[float, float],
        quarter: int = 0,
        skew: float = 0.0,
        warp: NDArray[np.float32] | None = None,
    ) -> ViewGeometry:
        """A box of ``box_size`` (height, width) source pixels around ``center`` (x, y),
        shown turned clockwise by ``90 * quarter + skew`` degrees: its text-up direction
        points ``quarter`` quarter turns (plus ``skew``) clockwise of screen up."""
        height, width = float(box_size[0]), float(box_size[1])
        size = (height, width) if quarter % 2 == 0 else (width, height)
        inverse = clockwise(90.0 * quarter + skew).T  # view offset -> source offset
        half = np.array([size[1] / 2.0, size[0] / 2.0])
        offset = np.asarray(center, dtype=np.float64) - inverse @ half
        return cls(size, np.column_stack([inverse, offset]), warp)

    @property
    def scale(self) -> float:
        """Source pixels per view pixel (geometric mean of the two axes)."""
        return math.sqrt(abs(float(np.linalg.det(self.matrix[:, :2]))))


class ViewRenderer(Protocol):
    def __call__(self, points: NDArray[np.float64], density: float) -> NDArray[np.float32]:
        """Gray values (h, w) in [0, 255] at view points (h, w, 2); ``density`` is the
        number of view pixels per output pixel."""
        ...


def to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """(H, W) or (H, W, 3) uint8 -> (H, W) uint8 luma (ITU-R 601, as PIL's ``L`` mode)."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise _error(Status.INPUT_ERROR, "invalid_image", "image must be uint8 (H, W[, 3])")
    if array.ndim == 2:
        return array
    if array.shape[2] != 3:
        raise _error(Status.INPUT_ERROR, "invalid_image", "color images must have 3 channels")
    return np.asarray(Image.fromarray(np.ascontiguousarray(array)).convert("L"))


class SourceSampler:
    """Bilinear lookups in a gray source through a ``ViewGeometry``.

    A lookup at ``density`` source pixels per output pixel reads a copy reduced by an
    integer box filter of ``max(1, int(density / 1.5))``, so a view far smaller than
    its source does not alias. The reduced copy covers the source cropped to a multiple
    of the factor, and lookups account for that extent.
    """

    def __init__(self, gray: NDArray[np.uint8], geometry: ViewGeometry) -> None:
        if gray.ndim != 2 or min(gray.shape) < 2:
            raise _error(Status.INPUT_ERROR, "invalid_image", "source must be (H >= 2, W >= 2)")
        self.height, self.width = gray.shape
        self.geometry = geometry
        self._levels = {1: np.array(gray, dtype=np.float32)}
        self._warp = None
        if geometry.warp is not None:  # (2, h, w) shifts, channels last for sampling
            warp = np.asarray(geometry.warp, np.float32)
            self._warp = np.ascontiguousarray(warp.transpose(1, 2, 0))

    def level(self, factor: int) -> tuple[NDArray[np.float32], int]:
        factor = max(1, min(int(factor), self.height // 2, self.width // 2))
        if factor not in self._levels:
            self._levels[factor] = ops.avg_pool(self._levels[1], factor)
        return self._levels[factor], factor

    def normalized(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """View points (..., 2) -> normalized source points (flattener convention)."""
        m = self.geometry.matrix * (2.0 / np.array([[self.width], [self.height]]))
        x, y = points[..., 0], points[..., 1]
        normalized = np.stack(
            [
                m[0, 0] * x + m[0, 1] * y + (m[0, 2] - 1.0),
                m[1, 0] * x + m[1, 1] * y + (m[1, 2] - 1.0),
            ],
            axis=-1,
        )
        if self._warp is not None:
            grid = normalized.astype(np.float32).reshape(-1, 1, 2)
            shift = ops.grid_sample(self._warp, grid, align_corners=True)
            normalized = normalized + shift.reshape(normalized.shape)
        return normalized

    def lookup(self, normalized: NDArray[np.float64], density: float) -> NDArray[np.float32]:
        """Gray values at normalized source points (h, w, 2); border values outside."""
        image, factor = self.level(max(1, int(density / 1.5)))
        rows, cols = image.shape
        extent = np.array([cols * factor / self.width, rows * factor / self.height])
        grid = (normalized + 1.0) / extent - 1.0
        return ops.grid_sample(image[..., None], grid.astype(np.float32))[..., 0]

    def __call__(self, points: NDArray[np.float64], density: float) -> NDArray[np.float32]:
        return self.lookup(self.normalized(points), density * self.geometry.scale)


@dataclass(slots=True)
class PageViews:
    """The model inputs of one page and how they were chosen."""

    global_view: NDArray[np.uint8]  # (S, S)
    patches: NDArray[np.uint8]  # (P, s, s)
    cells: NDArray[np.int64]  # (P, 2) row, column of the chosen energy cells
    energy: NDArray[np.float32]  # (G, G) text energy per cell
    content: tuple[float, float, float, float]  # page rectangle in the global view (x0, y0, x1, y1)


def quantize(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    return np.floor(np.clip(values, 0.0, 255.0) + 0.5).astype(np.uint8)


def _inside(points: NDArray[np.float64], size: tuple[float, float]) -> NDArray[np.bool_]:
    """Points on or inside the view rectangle (symmetric under quarter turns)."""
    tolerance = 1e-9
    x, y = points[..., 0], points[..., 1]
    return (
        (x >= -tolerance)
        & (x <= size[1] + tolerance)
        & (y >= -tolerance)
        & (y <= size[0] + tolerance)
    )


def _cell_energy(
    values: NDArray[np.float32],
    inside: NDArray[np.bool_],
    points: NDArray[np.float64],
    size: tuple[float, float],
    grid: int,
) -> NDArray[np.float32]:
    """Standard deviation of the gradient magnitude per cell of a ``grid`` x ``grid`` split
    of the view; pixels whose 4-neighbours leave the view are left out."""
    v = values.astype(np.float64)
    gx = np.zeros_like(v)
    gy = np.zeros_like(v)
    gx[:, 1:-1] = v[:, 2:] - v[:, :-2]
    gy[1:-1, :] = v[2:, :] - v[:-2, :]
    usable = inside.copy()
    usable[:, 1:-1] &= inside[:, 2:] & inside[:, :-2]
    usable[1:-1, :] &= inside[2:, :] & inside[:-2, :]
    usable[[0, -1], :] = False
    usable[:, [0, -1]] = False
    magnitude = np.sqrt(gx * gx + gy * gy)[usable]
    column = np.clip(np.floor(points[..., 0][usable] / size[1] * grid), 0, grid - 1)
    row = np.clip(np.floor(points[..., 1][usable] / size[0] * grid), 0, grid - 1)
    index = (row * grid + column).astype(np.int64)
    count = np.bincount(index, minlength=grid * grid)
    total = np.bincount(index, magnitude, minlength=grid * grid)
    squares = np.bincount(index, magnitude * magnitude, minlength=grid * grid)
    mean = total / np.maximum(count, 1)
    variance = np.maximum(squares / np.maximum(count, 1) - mean * mean, 0.0)
    energy = np.where(count >= 4, np.sqrt(variance), 0.0)
    return energy.reshape(grid, grid).astype(np.float32)


def select_cells(energy: NDArray[np.float32], count: int) -> NDArray[np.int64]:
    """(row, column) of the ``count`` highest-energy cells, outer ring excluded."""
    grid = energy.shape[0]
    inner = [(r, c) for r in range(1, grid - 1) for c in range(1, grid - 1)]
    scores = np.array([energy[r, c] for r, c in inner], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")[:count]
    return np.array([inner[i] for i in order], dtype=np.int64).reshape(-1, 2)


def page_views(
    render: ViewRenderer,
    size: tuple[float, float],
    config: OrientConfig | None = None,
    *,
    finish=None,
) -> PageViews:
    """Sample the global view and the patches of a view of ``size`` (height, width).

    ``render`` gives gray values at view points. ``finish(view, role)`` may change each
    uint8 view before use (``role`` is ``"global"`` or ``"patch"``; the training
    augmentation hooks in here). Pixels outside the view keep ``pad_value``, and patch
    selection reads the finished global view.
    """
    config = config or OrientConfig()
    height, width = float(size[0]), float(size[1])
    if not height > 0 or not width > 0:
        raise _error(Status.INPUT_ERROR, "invalid_view", "view size must be positive")
    size = (height, width)
    side = config.global_size
    scale = side / max(height, width)  # global-view pixels per view pixel
    left, top = (side - width * scale) / 2.0, (side - height * scale) / 2.0
    centers = np.arange(side, dtype=np.float64) + 0.5
    points = np.stack(np.meshgrid((centers - left) / scale, (centers - top) / scale), -1)
    inside = _inside(points, size)
    values = np.where(inside, render(points, 1.0 / scale), float(config.pad_value))
    global_view = quantize(values)
    if finish is not None:
        global_view = np.where(inside, finish(global_view, "global"), global_view)
    grid = config.energy_grid
    energy = _cell_energy(global_view.astype(np.float32), inside, points, size, grid)
    cells = select_cells(energy, config.patches)
    patch = config.patch_size
    step = max(height, width) / config.page_long_side  # view pixels per patch pixel
    offsets = (np.arange(patch, dtype=np.float64) + 0.5 - patch / 2.0) * step
    patches = np.empty((len(cells), patch, patch), np.uint8)
    for index, (row, column) in enumerate(cells):
        center = ((column + 0.5) * width / grid, (row + 0.5) * height / grid)
        grid_points = np.stack(np.meshgrid(center[0] + offsets, center[1] + offsets), -1)
        within = _inside(grid_points, size)
        sampled = np.where(within, render(grid_points, step), float(config.pad_value))
        patches[index] = quantize(sampled)
        if finish is not None:
            patches[index] = np.where(within, finish(patches[index], "patch"), patches[index])
    content = (left, top, left + width * scale, top + height * scale)
    return PageViews(global_view, patches, cells, energy, content)


def prepare_views(
    image: NDArray[np.uint8],
    config: OrientConfig | None = None,
    geometry: ViewGeometry | None = None,
) -> PageViews:
    """Model inputs for a page crop (the whole image) or for ``geometry`` in ``image``."""
    gray = to_gray(image)
    geometry = geometry or ViewGeometry.identity(*gray.shape)
    return page_views(SourceSampler(gray, geometry), geometry.size, config)


def views_to_arrays(views: list[PageViews]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Float32 (N, 1, S, S) and (N, P, 1, s, s) batches from ``PageViews``."""
    global_view = np.stack([v.global_view for v in views])[:, None].astype(np.float32)
    patches = np.stack([v.patches for v in views])[:, :, None].astype(np.float32)
    return global_view, patches


# ----------------------------------------------------------------------------- decision


def softmax(logits: NDArray, temperature: float = 1.0) -> NDArray[np.float64]:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass(slots=True)
class OrientResult:
    """``k``: clockwise quarter turns of the text-up direction (``np.rot90(view, k)``
    makes it upright). ``accepted`` is false below the threshold; ``reason_code`` is then
    ``orientation_uncertain`` and callers keep their geometric prior."""

    k: int
    probabilities: list[float]
    confidence: float
    margin: float
    accepted: bool
    reason_code: str | None
    temperature: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(logits: NDArray, temperature: float, threshold: float) -> OrientResult:
    """The decision for one page's 4 logits."""
    probabilities = softmax(np.asarray(logits, dtype=np.float64).reshape(CLASSES), temperature)
    order = np.argsort(-probabilities, kind="stable")
    confidence = float(probabilities[order[0]])
    accepted = confidence >= threshold
    return OrientResult(
        k=int(order[0]),
        probabilities=[float(p) for p in probabilities],
        confidence=confidence,
        margin=confidence - float(probabilities[order[1]]),
        accepted=accepted,
        reason_code=None if accepted else UNCERTAIN,
        temperature=float(temperature),
        threshold=float(threshold),
    )


# ----------------------------------------------------------------------------- calibration


def negative_log_likelihood(logits: NDArray, labels: NDArray, temperature: float = 1.0) -> float:
    labels = np.asarray(labels)
    keep = labels >= 0
    if not keep.any():
        return float("nan")
    probabilities = softmax(np.asarray(logits)[keep], temperature)
    picked = probabilities[np.arange(int(keep.sum())), labels[keep]]
    return float(-np.log(np.maximum(picked, 1e-12)).mean())


def fit_temperature(logits: NDArray, labels: NDArray) -> float:
    """Temperature in [0.05, 20] minimizing the NLL of the labelled rows (labels >= 0):
    a log-spaced search, then golden-section refinement."""
    if not (np.asarray(labels) >= 0).any():
        return DEFAULT_TEMPERATURE

    def loss(log_t: float) -> float:
        return negative_log_likelihood(logits, labels, math.exp(log_t))

    grid = np.linspace(math.log(0.05), math.log(20.0), 121)
    best = int(np.argmin([loss(v) for v in grid]))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        if loss(a) <= loss(b):
            high = b
        else:
            low = a
    return float(math.exp((low + high) / 2))


def choose_threshold(
    probabilities: NDArray, labels: NDArray, max_silent_error: float = 0.005, floor: float = 0.5
) -> dict[str, Any]:
    """Lowest threshold (so the highest coverage) whose silent error rate, accepted wrong
    decisions over all labelled rows, stays within ``max_silent_error``.

    Rows with ``labels < 0`` (text-free pages) do not count. The threshold is at least
    ``floor``. When no threshold meets the bound, ``feasible`` is false and the threshold
    is 1.
    """
    labels = np.asarray(labels)
    keep = labels >= 0
    probabilities = np.asarray(probabilities, dtype=np.float64)[keep]
    labels = labels[keep]
    count = len(labels)
    if count == 0:
        return {"threshold": DEFAULT_THRESHOLD, "feasible": False, "samples": 0}
    confidence = probabilities.max(axis=1)
    wrong = probabilities.argmax(axis=1) != labels
    order = np.argsort(-confidence, kind="stable")
    confidence, wrong = confidence[order], wrong[order]
    errors = np.cumsum(wrong)
    # Accepting every row with confidence >= confidence[i] takes rows up to the last tie.
    last = np.searchsorted(-confidence, -confidence, side="right") - 1
    ok = errors[last] <= max_silent_error * count
    if not ok.any():
        threshold, feasible = 1.0, False
    else:
        threshold, feasible = max(float(confidence[np.nonzero(ok)[0].max()]), floor), True
    accepted = confidence >= threshold
    return {
        "threshold": threshold,
        "feasible": feasible,
        "samples": count,
        "max_silent_error": max_silent_error,
        "coverage": float(accepted.mean()),
        "silent_error": float((accepted & wrong).sum() / count),
    }


def calibration_error(probabilities: NDArray, labels: NDArray, bins: int = 15) -> float:
    """Expected calibration error of the top probability over the labelled rows."""
    labels = np.asarray(labels)
    keep = labels >= 0
    if not keep.any():
        return float("nan")
    probabilities = np.asarray(probabilities, dtype=np.float64)[keep]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels[keep]
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            total += members.sum() * abs(correct[members].mean() - confidence[members].mean())
    return float(total / len(confidence))


def orientation_metrics(
    logits: NDArray,
    labels: NDArray,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    threshold: float = DEFAULT_THRESHOLD,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """Accuracy (overall, macro, per class, per group), 90 and 180 degree confusions,
    NLL, ECE and the operating point at ``threshold``; text-free rows (label -1) report
    how often they would be accepted."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    probabilities = softmax(logits, temperature)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    keep = labels >= 0
    result: dict[str, Any] = {"samples": int(keep.sum()), "text_free_samples": int((~keep).sum())}
    if keep.any():
        truth, guess = labels[keep], predicted[keep]
        confusion = np.zeros((CLASSES, CLASSES), np.int64)
        np.add.at(confusion, (truth, guess), 1)
        recall = [
            float(confusion[c, c] / confusion[c].sum()) if confusion[c].sum() else None
            for c in range(CLASSES)
        ]
        turn = (guess - truth) % CLASSES
        accepted = confidence[keep] >= threshold
        wrong = guess != truth
        result.update(
            accuracy=float((~wrong).mean()),
            macro_accuracy=float(np.mean([r for r in recall if r is not None])),
            recall=recall,
            confusion=confusion.tolist(),
            quarter_errors=int(((turn == 1) | (turn == 3)).sum()),
            half_errors=int((turn == 2).sum()),
            nll=negative_log_likelihood(logits, labels, temperature),
            ece=calibration_error(probabilities, labels),
            coverage=float(accepted.mean()),
            silent_error=float((accepted & wrong).mean()),
        )
        if groups is not None:
            names = np.asarray(groups)[keep]
            result["groups"] = {
                name: {
                    "samples": int((names == name).sum()),
                    "accuracy": float((~wrong[names == name]).mean()),
                }
                for name in sorted(set(names.tolist()))
            }
    if (~keep).any():
        result["text_free_confidence"] = float(confidence[~keep].mean())
        result["text_free_accepted"] = float((confidence[~keep] >= threshold).mean())
    return result


# ----------------------------------------------------------------------------- lineage


def data_spec_problems(spec: Any) -> list[str]:
    """Why a ``data_spec`` is not release data. OrientNet records carry a dewarp-schema
    spec (``flattener-dewarp-data-1``) naming every source that fed the build, so the
    same ``require_track(spec, "release")`` rule applies as for dewarp weights."""
    from ..dewarp.data import DataSpec, require_track

    try:
        require_track(DataSpec.from_dict(spec), "release")
    except FlattenerError as error:
        return [f"data spec: {error.message}"]
    except (TypeError, ValueError) as error:
        return [f"data spec: {error}"]
    return []


def data_sources_problems(sources: Any) -> list[str]:
    """Why a license manifest (``{split: [source, ...]}``) is not release data: every
    source needs track ``release`` and license classes on ``RELEASE_LICENSE_KEYS``."""
    from ..dewarp.data import release_license_ok

    if not isinstance(sources, dict) or not sources:
        return ["license manifest missing"]
    problems = []
    for split, listed in sources.items():
        for source in listed if isinstance(listed, list) else [None]:
            name = source.get("name") if isinstance(source, dict) else None
            classes = source.get("license_classes") if isinstance(source, dict) else None
            if not isinstance(source, dict) or source.get("track") != "release":
                problems.append(f"{split} source {name!r} is not release-track")
            elif not isinstance(classes, dict) or not all(
                isinstance(c, str) and release_license_ok(c) for c in classes
            ):
                problems.append(f"{split} source {name!r} has license classes {classes}")
    return problems


# ----------------------------------------------------------------------------- checkpoints


def code_record() -> dict[str, Any]:
    return code_version()


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Save model and training state atomically."""
    import torch

    path = Path(path)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise _error(Status.IO_ERROR, "invalid_checkpoint", f"schema must be {CHECKPOINT_SCHEMA}")
    atomic_write(path, lambda stream: torch.save(payload, stream))
    return {"file": path.name, "step": payload.get("step")}


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = read_checkpoint(Path(path))
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise _error(Status.IO_ERROR, "invalid_checkpoint", f"{path} is not {CHECKPOINT_SCHEMA}")
    return payload


def model_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {"model_id": MODEL_ID, "kind": "checkpoint", "step": payload.get("step")}


def operating_threshold(calibration: dict[str, Any] | None) -> float:
    """The threshold a loaded model uses: a calibrated one when it was measured on real
    photos (any ``source`` other than ``synthetic_val``), else the starting value 0.9."""
    calibration = calibration or {}
    if calibration.get("source") in (None, "synthetic_val") or not calibration.get("threshold"):
        return DEFAULT_THRESHOLD
    return float(calibration["threshold"])


class Orienter:
    """A loaded OrientNet with its calibration (temperature and threshold): a PyTorch
    ``OrientNet`` on ``device``, or an ``OnnxModel`` (output ``logits``) with ``config``."""

    def __init__(
        self,
        model: Any,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        threshold: float = DEFAULT_THRESHOLD,
        device: Any = "cpu",
        info: dict[str, Any] | None = None,
        config: OrientConfig | None = None,
    ) -> None:
        if not temperature > 0 or not 0 <= threshold <= 1:
            raise _error(
                Status.INPUT_ERROR, "invalid_calibration", "temperature > 0, threshold in [0, 1]"
            )
        if isinstance(model, OnnxModel):
            if config is None:
                raise _error(Status.INPUT_ERROR, "invalid_config", "an ONNX model needs a config")
            self.device, self.model, self.config = "cpu", model, config
        else:
            import torch

            self.device = torch.device(device)
            self.model = model.to(self.device).eval()
            self.config = model.config
        self.temperature, self.threshold = float(temperature), float(threshold)
        self.info = dict(info or {})
        self.info.update(temperature=self.temperature, threshold=self.threshold)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: Any = "cpu",
        temperature: float | None = None,
        threshold: float | None = None,
    ) -> Orienter:
        """Load weights and calibration from a checkpoint (PyTorch)."""
        from .orient_net import OrientNet

        path = Path(path)
        payload = load_checkpoint(path)
        info = model_record(payload)
        model = OrientNet(OrientConfig.from_dict(payload["model_config"]))
        model.load_state_dict(payload["model_state"])
        calibration = payload.get("calibration") or {}
        if temperature is None:
            temperature = calibration.get("temperature") or DEFAULT_TEMPERATURE
        info.update(
            path=str(path.resolve()), calibration=calibration, threshold_given=threshold is not None
        )
        return cls(
            model,
            temperature=temperature,
            threshold=operating_threshold(calibration) if threshold is None else threshold,
            device=device,
            info=info,
        )

    @classmethod
    def load_onnx(
        cls,
        path: str | Path,
        config: OrientConfig,
        *,
        temperature: float,
        threshold: float,
        info: dict[str, Any] | None = None,
    ) -> Orienter:
        """An exported ONNX model (inputs ``global_view``, ``patches``; output ``logits``)."""
        model = OnnxModel(path)
        record = {"model_id": MODEL_ID, "kind": "onnx", **(info or {})}
        record["path"] = str(Path(path).resolve())
        return cls(model, temperature=temperature, threshold=threshold, info=record, config=config)

    def logits(self, views: list[PageViews]) -> NDArray[np.float32]:
        global_view, patches = views_to_arrays(views)
        if isinstance(self.model, OnnxModel):
            return np.concatenate(
                [
                    self.model.output("logits", one[None], many[None])
                    for one, many in zip(global_view, patches, strict=True)
                ]
            ).astype(np.float32)
        import torch

        with torch.no_grad():
            out = self.model(
                torch.from_numpy(global_view).to(self.device),
                torch.from_numpy(patches).to(self.device),
            )
        return out.float().cpu().numpy()

    def predict(
        self, image: NDArray[np.uint8], geometry: ViewGeometry | None = None
    ) -> OrientResult:
        """Orientation of a page crop (or of ``geometry`` in ``image``)."""
        views = prepare_views(image, self.config, geometry)
        return decide(self.logits([views])[0], self.temperature, self.threshold)


# The PyTorch network and its export load on first use (``orient_net``).
_NET_EXPORTS = (
    "OrientNet",
    "count_parameters",
    "example_inputs",
    "exportable",
    "export_onnx",
    "model_cost",
    "views_to_tensors",
)


def __getattr__(name: str):
    if name in _NET_EXPORTS:
        from . import orient_net

        return getattr(orient_net, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ----------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m flattener.scan.orient")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="orientation of page crops (JSON lines)")
    predict.add_argument("images", nargs="+")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--device", default="cpu")
    export = commands.add_parser("export", help="ONNX export with a JSON sidecar")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--opset", type=int, default=18)
    export.add_argument("--temperature", type=float, default=None)
    export.add_argument("--threshold", type=float, default=None)
    commands.add_parser("cost", help="parameters and FLOPs of the default model")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 every image processed, 1 some failed, 2 usage or loading error."""
    args = _parser().parse_args(argv)
    from .orient_net import export_onnx, model_cost

    if args.command == "cost":
        print(json.dumps(model_cost()))
        return 0
    if args.command == "export":
        try:
            record = export_onnx(
                args.checkpoint,
                args.output,
                opset=args.opset,
                temperature=args.temperature,
                threshold=args.threshold,
            )
        except FlattenerError as error:
            print(
                json.dumps(
                    {
                        "status": error.status,
                        "reason_code": error.reason_code,
                        "message": error.message,
                    }
                )
            )
            return 2
        print(json.dumps({k: record[k] for k in ("file", "parity")}))
        return 0
    from ..image_io import load_image

    try:
        orienter = Orienter.load(args.checkpoint, device=args.device)
    except FlattenerError as error:
        print(
            json.dumps(
                {"status": error.status, "reason_code": error.reason_code, "message": error.message}
            )
        )
        return 2
    failed = 0
    for name in args.images:
        try:
            result = orienter.predict(load_image(name))
            print(json.dumps({"image": name, "status": Status.SUCCESS, **result.to_dict()}))
        except FlattenerError as error:
            failed += 1
            print(
                json.dumps(
                    {
                        "image": name,
                        "status": error.status,
                        "reason_code": error.reason_code,
                        "message": error.message,
                    }
                )
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
