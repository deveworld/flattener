"""Rectify RGB images with a dewarp checkpoint or the upstream UVDoc MIT weights.

``python -m flattener.dewarp.predict --weights W image --input a.jpg --output b.png``
``python -m flattener.dewarp.predict --weights W docunet --input-dir crop --output-dir out``

Input files are read with their EXIF orientation applied. The image is resized to
the model input (bilinear, align_corners=False, no antialias: upstream's
``cv2.resize`` INTER_LINEAR), the 2-D grid is predicted,
converted to a dense flattener backward map at the requested output size
(default: the input size) and rendered by ``flattener.render``. Upstream UVDoc
weights (``grid_convention="uvdoc"``) use the exact upstream densification.
The model runs in full float32 precision (TF32 off) so CUDA and CPU predictions agree.

Rendering runs the renderer's checks (source bounds, coverage, fold/Jacobian). An
image that fails them keeps its status (``invalid_transform`` and a reason code) in
the printed JSON and in ``predictions.json``, and no image is written for it; the
``image`` command also removes an earlier file at ``--output``. ``--force-render``
writes the image anyway (published methods are scored that way) and still records
the failure. The exit status is 0 when every image passed and 1 when any image
failed, with or without ``--force-render``.

``--quantize`` sets how the sampled float image becomes uint8: ``round`` (default) keeps
the image of ``flattener.render``, ``floor(clip(x, 0, 1) * 255 + 0.5)``; ``truncate``
samples the same float image and stores ``floor(clip(x, 0, 1) * 255)``, as published
DocUNet outputs were saved (``(x * 255).astype(uint8)``, the paper protocol). Checks,
failure statuses, ``--force-render`` and the exit status are the same in both modes. The
mode is recorded in the printed JSON, in ``predictions.json`` and in their evidence
block, and an output of one mode is never kept by a run of the other.

``--inset`` moves the page outline inward before rendering: the output shows the page
region ``[-(1 - e), 1 - e]`` of the predicted grid on each axis, that is ``e / 2`` of
the page extent less on each side, scaled to the full output (see ``INSET_PRESETS``
and ``dense_backward_map``). It is applied to the backward map, so the renderer's
checks, failure statuses, ``--force-render`` and the exit status are the same as
without it. It takes a preset name, one value for both axes or ``EX,EY``; the default
0 keeps the map unchanged. The inset is recorded like the quantization, and an output
of one inset is never kept by a run with another.

``--rerun R`` adds R crop passes after the whole-image prediction (default 0: the
whole-image prediction alone, unchanged). Each pass takes the page outline of the current
map (its boundary grid nodes, in source pixels), crops the outline's axis-aligned box
plus ``RERUN_MARGIN`` of the page extent per side (whole pixels, clipped to the image),
predicts again on the crop and composes that grid with the crop's placement into
coordinates of the original image, so the final map samples the original image once.
``--deskew`` (with ``--rerun``) also estimates the page angle from the outline's top and
bottom edges and, when the estimate is usable and at most ``DESKEW_MAX_DEG``, turns the
crop upright on a larger canvas before the prediction (``crop_pass``). The final map
goes through the same checks, failure statuses, ``--force-render`` and exit status. The
settings are recorded as ``rerun`` (``rerun_record``) in the printed JSON,
``predictions.json`` and their evidence block, and every pass (crop box, angle, whether
it turned the crop or why it was skipped) as ``crops``; an output of one setting is never
kept by a run with another.

``rectify_batch`` also takes another pipeline (``rectifier``, for example the scanner
of ``flattener.scan``) with its identity record (``pipeline``, recorded in
``predictions.json`` and the evidence block). A pipeline record with a ``protocol``
(``quantize``, ``inset``, ``rerun`` records) sets those labels of the folder. A directory
that holds predictions of another pipeline is refused, as one of other weights is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from torch.nn import functional as F

from ..contracts import FlattenerError, RenderConfig, RenderResult, Status
from ..evidence import evidence
from ..renderer import _handle_failure, _new_metadata, render
from ..validation import validate_config, validate_size
from .checkpoint import CHECKPOINT_SCHEMA, atomic_write, read_checkpoint, utc_now, write_json
from .coords import (
    _apply,
    _grid,
    _size,
    align_corners_to_flattener,
    center_positions,
    grid_to_map,
    interpolation_matrix,
    uvdoc_dense_map,
)
from .model import (
    UVDOC_CONFIG,
    UVDOC_REVISION,
    UVDOC_WEIGHTS_SHA256,
    DewarpModelConfig,
    UVDocNet,
    count_parameters,
    sha256_file,
)

DOCUNET_INPUT = re.compile(r"^(\d+)_(\d+) copy\.png$")
PREDICTIONS_SCHEMA = "flattener-dewarp-predictions-2"
QUANTIZE_MODES = ("round", "truncate")  # round: flattener.render; truncate: paper protocol
EXIT_FAILED = 1  # some image failed its checks (also when it was force-rendered)
DEFAULT_PIPELINE: dict[str, Any] = {"name": "rectify"}  # rectify_batch without a rectifier

# Grid-level insets (ex, ey) in normalized output units: output coordinate u in [-1, 1]
# shows the page position (1 - e) u of the predicted grid, so each side moves in by
# e / 2 of the page extent.
#
# syntheticdoc_half_cell: the inset of SyntheticDoc's own inference (repository
# training/model.py and training/inference.py at aabd9cf). Its model predicts a
# backward map of bm_out_size (121, 177), 121 rows (y) by 177 columns (x); the
# training loader resamples the released 177-row by 121-column maps into that shape.
# The map is center aligned: value (i, j) belongs to the page position
# ((2j + 1) / 177 - 1, (2i + 1) / 121 - 1). Inference resizes it to the image with
# F.interpolate(align_corners=True), which puts the outer values on the outer output
# pixel centers, so output coordinate u shows the page position
# (1 - 1/n) u Wo / (Wo - 1) for n map cells and Wo output pixels on that axis, and
# (1 - 1/n) u for a large output. Hence e = 1/n, half a map cell per side:
# x 1/177 (1/354 = 0.282% of the page width per side), y 1/121 (1/242 = 0.413% of the
# page height per side). The framing analysis took the map as 121 columns by 177 rows
# (the released orientation) and so had the two axes swapped (0.41% x, 0.28% y).
# The outermost column and row of SyntheticDoc's training targets come from PIL's
# resize of the released maps (edge value held in x, a clipped filter in y) and sit a
# little further in (x 1/121, y about 0.0097); the preset follows the resize geometry.
SYNTHETICDOC_MAP_SIZE = (121, 177)  # bm_out_size: rows (y), columns (x)
INSET_PRESETS: dict[str, tuple[float, float]] = {
    # half a map cell per side on each axis: (ex, ey) = (1 / columns, 1 / rows)
    "syntheticdoc_half_cell": (1 / SYNTHETICDOC_MAP_SIZE[1], 1 / SYNTHETICDOC_MAP_SIZE[0]),
}
NO_INSET = (0.0, 0.0)
Inset = str | float | tuple[float, float] | None

# Crop-and-rerun defaults, fixed before any benchmark result was looked at.
RERUN_MARGIN = 0.04  # per side, fraction of the page extent: the middle of training's 0-8%
DESKEW_MAX_DEG = 45.0  # a page estimated to turn further is cropped without turning
DESKEW_MIN_AGREEMENT = 0.5  # least mean resultant length of the edge directions
MIN_CROP_PIXELS = 2  # a crop box narrower or lower than this skips the pass
# Test-time augmentation of every model call: "flip" averages in the grid of the mirror
# image, turned back.
TTA_MODES = ("none", "flip")


@dataclass(slots=True)
class Dewarper:
    model: UVDocNet
    config: DewarpModelConfig
    device: torch.device
    info: dict[str, Any] = field(default_factory=dict)
    tta: str = "none"  # TTA_MODES; see predict_input


def with_tta(dewarper: Dewarper, tta: str) -> Dewarper:
    """``dewarper`` with test-time augmentation ``tta`` (``TTA_MODES``)."""
    if tta not in TTA_MODES:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_config", f"tta must be one of {TTA_MODES}, got {tta!r}"
        )
    return replace(dewarper, tta=tta)


def load_dewarper(path: str | Path, *, device: str = "cpu", verify: bool = True) -> Dewarper:
    """Load a dewarp checkpoint (sidecar-verified) or an upstream ``best_model.pkl``."""
    path = Path(path)
    payload = read_checkpoint(path, verify=verify)
    digest = sha256_file(path)
    if isinstance(payload, dict) and payload.get("schema") == CHECKPOINT_SCHEMA:
        config = DewarpModelConfig.from_dict(payload["model_config"])
        state = payload["model_state"]
        info = {
            "kind": "checkpoint",
            "step": payload["step"],
            "track": payload["track"],
            "lineage": payload["lineage"],
            "data_spec_sha256": payload["data_spec_sha256"],
            "code_sha256": payload["code"]["sha256"],
        }
    elif isinstance(payload, dict) and set(payload) == {"model_state"}:
        config, state = UVDOC_CONFIG, payload["model_state"]
        info = {
            "kind": "uvdoc",
            "track": "external",
            "license": "MIT" if digest == UVDOC_WEIGHTS_SHA256 else "unverified",
            "sha256_matches_upstream": digest == UVDOC_WEIGHTS_SHA256,
            "source": f"https://github.com/tanguymagne/UVDoc@{UVDOC_REVISION}",
        }
    else:
        raise FlattenerError(Status.MODEL_ERROR, "invalid_checkpoint", f"{path}: unknown format")
    model = UVDocNet(config)
    model.load_state_dict(state)
    target = torch.device(device)
    info.update(
        path=str(path),
        sha256=digest,
        parameters=count_parameters(model)["total"],
        model_config=config.to_dict(),
    )
    return Dewarper(model.to(target).eval(), config, target, info)


def _check_image(image: NDArray[np.uint8]) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_image", "image must be RGB uint8 HxWx3")


def _legacy_matmul_precision() -> str | None:
    try:
        return torch.get_float32_matmul_precision()
    except RuntimeError:  # legacy and new TF32 settings were mixed; keep the new value only
        return None


@contextmanager
def ieee_float32() -> Iterator[None]:
    """Run CUDA float32 convolutions and matmuls without TF32; restore the settings on exit.

    Also selects cuDNN algorithms by heuristic (no benchmark) and deterministically.
    """
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    saved_algorithms = cudnn.benchmark, cudnn.deterministic
    new_api = hasattr(cudnn, "conv")  # torch >= 2.9 has per-op fp32_precision flags
    if new_api:
        flags = (cudnn.conv, matmul, torch.backends.mkldnn.matmul)
        saved = [flag.fp32_precision for flag in flags], _legacy_matmul_precision()
        torch.set_float32_matmul_precision("highest")  # keeps legacy and new matmul flags in sync
        cudnn.conv.fp32_precision = "ieee"
    else:
        saved = cudnn.allow_tf32, matmul.allow_tf32
        cudnn.allow_tf32 = matmul.allow_tf32 = False
    cudnn.benchmark, cudnn.deterministic = False, True
    try:
        yield
    finally:
        cudnn.benchmark, cudnn.deterministic = saved_algorithms
        if new_api:
            values, legacy = saved
            if legacy is not None:
                torch.set_float32_matmul_precision(legacy)
            for flag, value in zip(flags, values, strict=True):
                flag.fp32_precision = value
        else:
            cudnn.allow_tf32, matmul.allow_tf32 = saved


def model_input(dewarper: Dewarper, image: NDArray[np.uint8]) -> torch.Tensor:
    """``image`` as the (1, 3, H, W) float model input: resized to the input size
    (bilinear, align_corners=False, no antialias)."""
    _check_image(image)
    tensor = torch.tensor(image).to(dewarper.device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255)
    return F.interpolate(
        tensor,
        size=dewarper.config.input_size,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


# ``ieee_float32`` sets process-wide flags, so model calls from batch threads take turns.
_MODEL_LOCK = threading.Lock()


def predict_input(dewarper: Dewarper, tensor: torch.Tensor) -> NDArray[np.float32]:
    """Raw 2-D model output for a model input, as (Gh, Gw, 2). With ``tta="flip"`` the
    input and its mirror image run as one batch; the mirror's grid, turned back (columns
    reversed, x negated: both grid conventions are symmetric about the image centre), is
    averaged with the input's."""
    flip = dewarper.tta == "flip"
    batch = torch.cat([tensor, tensor.flip(-1)]) if flip else tensor
    with _MODEL_LOCK, torch.inference_mode(), ieee_float32():
        grid2d, _ = dewarper.model(batch)
    grid = grid2d.float()
    if flip:
        mirror = grid[1].flip(-1)
        mirror[0] = -mirror[0]
        return ((grid[0] + mirror) / 2).permute(1, 2, 0).cpu().numpy()
    return grid[0].permute(1, 2, 0).cpu().numpy()


def predict_grid(dewarper: Dewarper, image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Raw 2-D model output as (Gh, Gw, 2), in the model's grid convention."""
    return predict_input(dewarper, model_input(dewarper, image))


def _grid_index(count: int, nodes: int, inset: float, convention: str) -> NDArray[np.float64]:
    """Fractional grid-node index that each of ``count`` output pixels samples on one axis.

    ``flattener``: pixel center u = (2k + 1) / count - 1 on the edge-aligned grid, as
    ``grid_to_map``. ``uvdoc``: pixel k on node k (nodes - 1) / (count - 1), as
    upstream's align_corners=True interpolation. The inset scales the position about
    the page center by ``1 - inset`` before it is looked up."""
    if convention == "uvdoc":
        position = np.linspace(-1.0, 1.0, count, dtype=np.float64)
    else:
        position = center_positions(count)
    return ((1.0 - inset) * position + 1.0) / 2.0 * (nodes - 1)


def dense_backward_map(
    grid: NDArray[np.float32],
    convention: str,
    output_size: tuple[int, int],
    source_size: tuple[int, int],
    inset: Inset = None,
    *,
    flattener_values: bool = False,
) -> NDArray[np.float32]:
    """Flattener backward map at ``output_size`` for a grid in ``convention``.

    With an inset ``(ex, ey)`` (``resolve_inset``), output coordinate u = (ux, uy) in
    [-1, 1] gets the value that the map without inset has at ((1 - ex) ux, (1 - ey) uy):
    the grid is sampled on the page region [-(1 - e), 1 - e] and the outline moves in by
    e / 2 of the page extent per side. No inset returns the plain map unchanged.

    ``flattener_values``: the grid already holds flattener values of the source (a
    rerun's composed grid, ``rerun_grid``); ``convention`` then only places the nodes."""
    ex, ey = resolve_inset(inset)
    if (ex, ey) == NO_INSET and not (convention == "uvdoc" and flattener_values):
        if convention == "uvdoc":
            return uvdoc_dense_map(grid, output_size, source_size)
        return grid_to_map(grid, output_size)
    array = _grid(grid, "grid")  # the checks of grid_to_map and uvdoc_dense_map
    height, width = _size(output_size, "output_size")
    gh, gw = array.shape[:2]
    rows = interpolation_matrix(_grid_index(height, gh, ey, convention), gh, extrapolate=False)
    cols = interpolation_matrix(_grid_index(width, gw, ex, convention), gw, extrapolate=False)
    dense = _apply(rows, array, cols)
    if convention == "uvdoc" and not flattener_values:
        return align_corners_to_flattener(dense, source_size)
    return dense


def check_quantize(quantize: str) -> None:
    if quantize not in QUANTIZE_MODES:
        raise FlattenerError(
            Status.INPUT_ERROR,
            "invalid_config",
            f"quantize must be one of {QUANTIZE_MODES}, got {quantize!r}",
        )


def _inset_error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_config", message)


def resolve_inset(inset: Inset) -> tuple[float, float]:
    """``(ex, ey)`` of an inset: ``None`` (no inset), a name of ``INSET_PRESETS``, one
    value for both axes or an ``(ex, ey)`` pair, each in ``[0, 1)``."""
    if inset is None:
        return NO_INSET
    if isinstance(inset, str):
        if inset not in INSET_PRESETS:
            raise _inset_error(f"inset preset must be one of {tuple(INSET_PRESETS)}, got {inset!r}")
        return INSET_PRESETS[inset]
    values = (inset, inset) if not isinstance(inset, (tuple, list)) else tuple(inset)
    if len(values) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float, np.floating)) for v in values
    ):
        raise _inset_error(f"inset must be a preset name, a number or (ex, ey), got {inset!r}")
    ex, ey = float(values[0]), float(values[1])
    if not (0 <= ex < 1 and 0 <= ey < 1):
        raise _inset_error(f"inset values must be in [0, 1), got {(ex, ey)}")
    return ex, ey


def inset_record(inset: Inset) -> dict[str, Any]:
    """How an inset is recorded: ``{"x": ex, "y": ey, "preset": name or None}``."""
    ex, ey = resolve_inset(inset)
    return {"x": ex, "y": ey, "preset": inset if isinstance(inset, str) else None}


def recorded_inset(record: dict[str, Any] | None) -> tuple[float, float]:
    """``(ex, ey)`` of a recorded inset; a record without one predates the option."""
    if not isinstance(record, dict):
        return NO_INSET
    return float(record.get("x", 0.0)), float(record.get("y", 0.0))


def parse_inset(text: str) -> str | float | tuple[float, float]:
    """``--inset``: a preset name, ``E`` (both axes) or ``EX,EY``."""
    if text in INSET_PRESETS:
        return text
    try:
        values = tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected one of {', '.join(INSET_PRESETS)}, E or EX,EY; got {text!r}"
        ) from None
    if len(values) not in (1, 2):
        raise argparse.ArgumentTypeError(f"expected E or EX,EY; got {text!r}")
    inset = values[0] if len(values) == 1 else (values[0], values[1])
    try:
        resolve_inset(inset)
    except FlattenerError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return inset


def inset_label(inset: Inset) -> str:
    """Short name of an inset for file names: the preset name, ``E`` or ``EXxEY``;
    empty for no inset."""
    ex, ey = resolve_inset(inset)
    if isinstance(inset, str):
        return inset
    if (ex, ey) == NO_INSET:
        return ""
    return f"{ex:g}" if ex == ey else f"{ex:g}x{ey:g}"


def _rerun_error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_config", message)


def rerun_record(
    rerun: int = 0,
    deskew: bool = False,
    perspective: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
) -> dict[str, Any]:
    """How crop-and-rerun settings are recorded: ``passes``, ``deskew`` and the rules that
    apply (``margin`` with passes, ``max_angle_deg`` and ``min_agreement`` with deskew;
    ``None`` otherwise), ``perspective: true`` for perspective passes
    (``homography_pass``) and ``snap: true`` when the final border is snapped to the photo's
    page edges (``edges.snap_border``); ``rect_margins`` for turned tight-crop passes
    (``rect_pass``); ``refine: true`` for an interior pass on the flat output
    (``refine_pass``). Refuses a negative or non-integer ``rerun``, and deskew, perspective or
    rect without passes or together (each pass is one kind)."""
    if isinstance(rerun, bool) or not isinstance(rerun, (int, np.integer)) or rerun < 0:
        raise _rerun_error(f"rerun must be an integer >= 0, got {rerun!r}")
    if not all(isinstance(flag, bool) for flag in (deskew, perspective, snap, rect, refine)):
        raise _rerun_error(f"deskew and perspective must be bools, got {deskew!r}, {perspective!r}")
    if (deskew or perspective or rect) and rerun == 0:
        raise _rerun_error("deskew, perspective and rect need rerun >= 1")
    if deskew + perspective + rect > 1:
        raise _rerun_error("choose one pass kind: deskew, perspective or rect")
    record = {
        "passes": int(rerun),
        "deskew": deskew,
        "margin": RERUN_MARGIN if rerun else None,
        "max_angle_deg": DESKEW_MAX_DEG if deskew else None,
        "min_agreement": DESKEW_MIN_AGREEMENT if deskew else None,
    }
    if perspective:
        record["perspective"] = True
    if snap:
        record["snap"] = True
    if refine:
        record["refine"] = True
    if rect:
        record["rect_margins"] = [RECT_MARGINS[min(i, len(RECT_MARGINS) - 1)] for i in range(rerun)]
    return record


def recorded_rerun(record: dict[str, Any] | None) -> dict[str, Any]:
    """A recorded rerun setting; a record without one predates the option (no passes)."""
    return record if isinstance(record, dict) else rerun_record()


def parse_rerun(text: str) -> int:
    """``--rerun``: the number of crop passes, an integer >= 0."""
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {text!r}")
    return value


def _source_points(values: NDArray, source_size: tuple[int, int]) -> NDArray[np.float64]:
    """Flattener values -> source pixel-edge coordinates (0 and W are the outer edges)."""
    height, width = source_size
    scale = np.array([width, height], dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) + 1.0) / 2.0 * scale


def _flattener_values(raw: NDArray, convention: str, size: tuple[float, float]) -> NDArray:
    """Raw model values -> float64 flattener values of the (height, width) ``size`` image
    the model saw (``uvdoc``: align_corners=True values of that image)."""
    values = np.asarray(raw, dtype=np.float64)
    if convention == "uvdoc":
        n = np.array([size[1], size[0]], dtype=np.float64)
        values = values * (n - 1.0) / n
    return values


def outline_points(grid: NDArray, source_size: tuple[int, int]) -> NDArray[np.float64]:
    """Boundary nodes of a grid of flattener values, as source pixel-edge points (N, 2)."""
    points = _source_points(grid, source_size)
    return np.concatenate([points[0], points[-1], points[1:-1, 0], points[1:-1, -1]])


def crop_box(
    outline: NDArray, source_size: tuple[int, int], margin: float = RERUN_MARGIN
) -> tuple[int, int, int, int]:
    """Whole-pixel box ``(left, top, right, bottom)`` around the ``outline`` points: their
    axis-aligned bounds plus ``margin`` of their extent per side, clipped to the image."""
    height, width = source_size
    low, high = outline.min(axis=0), outline.max(axis=0)
    extent = high - low
    limit = np.array([width, height], dtype=np.float64)
    low = np.floor(low - margin * extent).clip(0, limit)
    high = np.ceil(high + margin * extent).clip(0, limit)
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def estimate_page_angle(
    grid: NDArray, source_size: tuple[int, int]
) -> tuple[float | None, str | None]:
    """Page angle in degrees from the top and bottom rows of a grid of flattener values,
    or ``None`` and why the estimate is unusable.

    The angle is the direction of the two edges in source pixels (x right, y down, so a
    positive angle turns the page clockwise on screen); both edges run along increasing
    grid columns. It is a robust mean of the directions of the edge segments: their
    circular mean, corrected by the mean deviation of the middle half of the segments
    (``n // 4`` dropped at each end). Unusable: ``degenerate_outline`` when an edge is not
    finite or its end nodes lie less than ``MIN_CROP_PIXELS`` apart, and
    ``inconsistent_edges`` when the mean resultant length of the unit directions is below
    ``DESKEW_MIN_AGREEMENT``."""
    points = _source_points(grid, source_size)
    units = []
    for edge in (points[0], points[-1]):
        if not np.isfinite(edge).all() or np.hypot(*(edge[-1] - edge[0])) < MIN_CROP_PIXELS:
            return None, "degenerate_outline"
        steps = np.diff(edge, axis=0)
        lengths = np.hypot(steps[:, 0], steps[:, 1])
        units.append(steps[lengths > 0] / lengths[lengths > 0, None])
    directions = np.concatenate(units)
    mean = directions.mean(axis=0)
    if np.hypot(*mean) < DESKEW_MIN_AGREEMENT:
        return None, "inconsistent_edges"
    center = np.arctan2(mean[1], mean[0])
    deviation = np.sort(np.angle(np.exp(1j * (np.arctan2(*directions.T[::-1]) - center))))
    drop = deviation.size // 4
    angle = np.angle(np.exp(1j * (center + deviation[drop : deviation.size - drop].mean())))
    return float(np.degrees(angle)), None


def canvas_size(box: tuple[int, int, int, int], angle_deg: float) -> tuple[float, float]:
    """(height, width) in source pixels of the canvas that holds the whole crop ``box``
    turned by ``-angle_deg``; the crop's own size for no angle."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if angle_deg == 0.0:
        return float(height), float(width)
    c, s = abs(np.cos(np.radians(angle_deg))), abs(np.sin(np.radians(angle_deg)))
    return float(height * c + width * s), float(width * c + height * s)


def canvas_to_source(
    box: tuple[int, int, int, int],
    angle_deg: float,
    canvas: tuple[float, float],
    source_size: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Affine map ``q = M q' + t`` from flattener values q' of the canvas to flattener
    values q of the source.

    The canvas (``canvas_size``, centered on the crop ``box``) shows the crop turned by
    ``-angle_deg``: canvas point q' lies at ``R(angle) (q' * canvas / 2)`` source pixels
    from the box center, with R turning x toward y. For no angle the canvas is the box."""
    left, top, right, bottom = box
    height, width = source_size
    theta = np.radians(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    matrix = np.diag([2.0 / width, 2.0 / height]) @ rotation @ np.diag(np.divide(canvas[::-1], 2))
    offset = np.array([(left + right) / width - 1.0, (top + bottom) / height - 1.0])
    return matrix, offset


def canvas_input(
    dewarper: Dewarper, crop: NDArray[np.uint8], angle_deg: float, canvas: tuple[float, float]
) -> torch.Tensor:
    """Model input of ``crop`` turned by ``-angle_deg`` on its ``canvas``, sampled once at
    the input's pixel centers (bilinear, align_corners=False); canvas pixels outside the
    crop repeat its border pixels."""
    _check_image(crop)
    height, width = crop.shape[:2]
    matrix, offset = canvas_to_source((0, 0, width, height), angle_deg, canvas, (height, width))
    in_height, in_width = dewarper.config.input_size
    positions = np.stack(np.meshgrid(center_positions(in_width), center_positions(in_height)), -1)
    grid = torch.from_numpy((positions @ matrix.T + offset).astype(np.float32))[None]
    source = torch.tensor(crop).to(dewarper.device)
    source = source.permute(2, 0, 1).unsqueeze(0).float().div_(255)
    return F.grid_sample(
        source,
        grid.to(dewarper.device),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def rerun_box(
    grid: NDArray,
    source_size: tuple[int, int],
    *,
    deskew: bool = False,
    max_angle_deg: float | None = DESKEW_MAX_DEG,
) -> tuple[tuple[int, int, int, int] | None, float, dict[str, Any]]:
    """Crop box and angle of the next rerun pass from ``grid`` (flattener values of a
    ``source_size`` image), and the pass record so far.

    The box is ``crop_box`` of the grid's outline (``outline_points``); it is ``None``
    when the outline is not finite (``skipped: non_finite_outline``) or the box is under
    ``MIN_CROP_PIXELS`` on an axis (``skipped: empty_crop``). With ``deskew`` the angle is
    ``estimate_page_angle`` when that estimate is usable and at most ``max_angle_deg``
    either way (``None``: no limit); otherwise it is 0 and ``deskew_skipped`` says why
    (the estimate's reason or ``angle_above_limit``). See ``crop_pass`` for the record."""
    record: dict[str, Any] = {
        "outline_box": None,
        "box": None,
        "angle_deg": None,
        "deskew_skipped": None,
        "rotated": False,
        "canvas_size": None,
        "skipped": None,
    }
    outline = outline_points(grid, source_size)
    if not np.isfinite(outline).all():
        record["skipped"] = "non_finite_outline"
        return None, 0.0, record
    record["outline_box"] = [*outline.min(axis=0).tolist(), *outline.max(axis=0).tolist()]
    box = crop_box(outline, source_size)
    left, top, right, bottom = box
    record["box"] = list(box)
    if min(right - left, bottom - top) < MIN_CROP_PIXELS:
        record["skipped"] = "empty_crop"
        return None, 0.0, record
    angle = 0.0
    if deskew:
        estimate, reason = estimate_page_angle(grid, source_size)
        if reason is None and max_angle_deg is not None and abs(estimate) > max_angle_deg:
            reason = "angle_above_limit"
        record.update(angle_deg=estimate, deskew_skipped=reason)
        angle = estimate if reason is None else 0.0
    return box, angle, record


def crop_pass_box(
    dewarper: Dewarper,
    image: NDArray[np.uint8],
    box: tuple[int, int, int, int],
    angle_deg: float = 0.0,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """One pass on the whole-pixel crop ``box`` (left, top, right, bottom) of ``image``:
    the predicted grid in flattener values of ``image`` and ``{"rotated", "canvas_size"}``.

    For no angle the model sees the crop resized like a whole image (``predict_grid``).
    Otherwise it sees the crop turned by ``-angle_deg`` on the canvas that keeps all of it
    (``canvas_size``, ``canvas_input``). The prediction is placed in the source with
    ``canvas_to_source``. Angles past 45 degrees either way turn the canvas as they are;
    ``flattener.scan.placement`` places exact quarter turns."""
    left, top, right, bottom = box
    crop = image[top:bottom, left:right]
    canvas = canvas_size(box, angle_deg)
    if angle_deg == 0.0:
        raw = predict_grid(dewarper, crop)
    else:
        raw = predict_input(dewarper, canvas_input(dewarper, crop, angle_deg, canvas))
    local = _flattener_values(raw, dewarper.config.grid_convention, canvas)
    matrix, offset = canvas_to_source(box, angle_deg, canvas, image.shape[:2])
    record = {"rotated": angle_deg != 0.0, "canvas_size": list(canvas)}
    return local @ matrix.T + offset, record


def crop_pass(
    dewarper: Dewarper, image: NDArray[np.uint8], grid: NDArray, deskew: bool = False
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """One rerun pass on ``image`` from ``grid`` (flattener values of ``image``): the new
    grid in flattener values of ``image`` and the pass record.

    The box and angle come from ``rerun_box`` (the outline's ``crop_box``; with
    ``deskew`` a usable ``estimate_page_angle`` of at most ``DESKEW_MAX_DEG``), and the
    prediction from ``crop_pass_box``. A pass whose outline is not finite
    (``non_finite_outline``) or whose box is under ``MIN_CROP_PIXELS`` on an axis
    (``empty_crop``) keeps ``grid``.

    The record holds ``outline_box`` (the outline's bounds, source pixels), ``box``
    (left, top, right, bottom), ``angle_deg`` (the estimate; ``None`` without deskew or
    when unusable), ``deskew_skipped`` (why the crop was not turned: the estimate's
    reason or ``angle_above_limit``), ``rotated``, ``canvas_size`` (height, width in
    source pixels) and ``skipped`` (why the pass kept ``grid``)."""
    box, angle, record = rerun_box(grid, image.shape[:2], deskew=deskew)
    if box is None:
        return np.asarray(grid, dtype=np.float64), record
    composed, placed = crop_pass_box(dewarper, image, box, angle)
    record.update(placed)
    return composed, record


def _clockwise_convex(quad: NDArray[np.float64]) -> bool:
    edges = np.roll(quad, -1, axis=0) - quad
    turns = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    return bool((turns > 0).all())  # y points down: clockwise on screen


def homography_pass(
    dewarper: Dewarper, image: NDArray[np.uint8], grid: NDArray
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """One rerun pass on ``image`` from ``grid`` through a perspective view: the new grid in
    flattener values of ``image`` and the pass record.

    A homography takes the grid's four corner nodes (TL, TR, BR, BL, source pixels) to an
    upright rectangle inside a view of the model input size, ``RERUN_MARGIN`` of the page
    extent from the view edge on every side, so the model sees the page without the
    camera's perspective. The view samples the source once (``composite.warp_homography``,
    box-reduced first when it shrinks the source). The prediction is placed back node by
    node through the inverse homography. A pass whose corners are not finite
    (``non_finite_outline``) or not a clockwise convex quad (``not_convex``) keeps
    ``grid``. The record holds ``corners`` (source pixels) and ``view_size`` and
    ``skipped``."""
    from .composite import apply_homography, corner_homography, polygon_area, warp_homography

    height, width = image.shape[:2]
    record: dict[str, Any] = {"corners": None, "view_size": None, "skipped": None}
    points = _source_points(grid, (height, width))
    corners = np.array([points[0, 0], points[0, -1], points[-1, -1], points[-1, 0]])
    if not np.isfinite(corners).all():
        record["skipped"] = "non_finite_outline"
        return np.asarray(grid, dtype=np.float64), record
    if not _clockwise_convex(corners):
        record["skipped"] = "not_convex"
        return np.asarray(grid, dtype=np.float64), record
    view_h, view_w = dewarper.config.input_size
    low, high = RERUN_MARGIN / (1 + 2 * RERUN_MARGIN), (1 + RERUN_MARGIN) / (1 + 2 * RERUN_MARGIN)
    rectangle = np.array(
        [[low * view_w, low * view_h], [high * view_w, low * view_h],
         [high * view_w, high * view_h], [low * view_w, high * view_h]]
    )  # fmt: skip
    transform = corner_homography(corners, rectangle)
    step = np.sqrt(abs(polygon_area(corners)) / abs(polygon_area(rectangle)))
    view = warp_homography(
        Image.fromarray(image), transform, (view_h, view_w), step=step, edge=True
    )
    raw = predict_grid(dewarper, np.ascontiguousarray(view))
    local = _flattener_values(raw, dewarper.config.grid_convention, (view_h, view_w))
    view_points = (np.asarray(local, dtype=np.float64) + 1.0) / 2.0 * np.array([view_w, view_h])
    source = apply_homography(np.linalg.inv(transform), view_points)
    record.update(corners=np.round(corners, 2).tolist(), view_size=[view_h, view_w])
    return 2.0 * source / np.array([width, height], dtype=np.float64) - 1.0, record


def refine_pass(
    dewarper: Dewarper, image: NDArray[np.uint8], grid: NDArray
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Interior correction from the model's view of its own flat output: ``grid``
    (flattener values of ``image``, edge-aligned nodes) renders a flat page at the model
    input size; the model's grid on it minus the identity is the leftover warp. Its border
    part (the Coons patch of the border residual) is dropped, so the outline stays, and
    the rest moves the interior nodes: the new grid is ``grid`` sampled (bilinear over
    its nodes) at the corrected positions. The record holds the mean interior move in
    view pixels (``interior_px``)."""
    from .edges import coons

    view_h, view_w = dewarper.config.input_size
    array = np.asarray(grid, dtype=np.float64)
    dense = torch.from_numpy(grid_to_map(array, (view_h, view_w)).astype(np.float32))[None]
    source = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float()
    flat = F.grid_sample(source, dense, mode="bilinear", padding_mode="border", align_corners=False)
    view = flat[0].permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).numpy()
    raw = predict_grid(dewarper, np.ascontiguousarray(view))
    local = np.asarray(
        _flattener_values(raw, dewarper.config.grid_convention, (view_h, view_w)), np.float64
    )
    rows, cols = array.shape[:2]
    ys, xs = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij")
    identity = np.stack([xs, ys], axis=-1)
    residual = local - identity
    residual -= coons(residual[0], residual[-1], residual[:, 0], residual[:, -1])
    positions = torch.from_numpy((identity + residual).astype(np.float32))[None]
    nodes = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1)[None]
    refined = F.grid_sample(nodes, positions, mode="bilinear", padding_mode="border",
                            align_corners=True)  # fmt: skip
    scale = np.array([view_w, view_h]) / 2.0
    record = {"interior_px": round(float(np.abs(residual * scale).mean()), 3)}
    return refined[0].permute(1, 2, 0).numpy().astype(np.float64), record


RECT_MARGINS = (0.01, 0.03)  # per pass, AADD's DIR300 expansion ratios; the last repeats


def _min_area_rectangle(points: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Corners (TL, TR, BR, BL on screen after turning upright) of the minimum-area
    rectangle around ``points``, turned by at most 45 degrees either way."""
    from ..scan.locate import min_area_rect_angle

    angle = np.radians(min_area_rect_angle(points))
    cos, sin = np.cos(angle), np.sin(angle)
    axis_u, axis_v = np.array([cos, sin]), np.array([-sin, cos])
    along, across = points @ axis_u, points @ axis_v
    if np.ptp(along) < MIN_CROP_PIXELS or np.ptp(across) < MIN_CROP_PIXELS:
        return None
    return np.array(
        [axis_u * a + axis_v * b for a, b in (
            (along.min(), across.min()), (along.max(), across.min()),
            (along.max(), across.max()), (along.min(), across.max()))]
    )  # fmt: skip


def rect_pass(
    dewarper: Dewarper, image: NDArray[np.uint8], grid: NDArray, margin: float
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """One rerun pass through a turned, tight crop (AADD's axis-aligned preprocessing):
    the minimum-area rectangle around the grid's outline, grown by ``margin`` of its size
    per side, is turned upright and resized to the model input in one sampling
    (``composite.warp_homography``); the prediction is placed back through the inverse
    transform. The record holds ``rectangle`` (source pixels), ``margin`` and
    ``skipped`` (``non_finite_outline`` or ``empty_crop``)."""
    from .composite import apply_homography, corner_homography, polygon_area, warp_homography

    height, width = image.shape[:2]
    record: dict[str, Any] = {"rectangle": None, "margin": margin, "skipped": None}
    points = outline_points(grid, (height, width))
    if not np.isfinite(points).all():
        record["skipped"] = "non_finite_outline"
        return np.asarray(grid, dtype=np.float64), record
    rectangle = _min_area_rectangle(points)
    if rectangle is None:
        record["skipped"] = "empty_crop"
        return np.asarray(grid, dtype=np.float64), record
    center = rectangle.mean(axis=0)
    rectangle = center + (rectangle - center) * (1.0 + 2.0 * margin)
    view_h, view_w = dewarper.config.input_size
    frame = np.array([[0, 0], [view_w, 0], [view_w, view_h], [0, view_h]], dtype=np.float64)
    transform = corner_homography(rectangle, frame)
    step = np.sqrt(abs(polygon_area(rectangle)) / abs(polygon_area(frame)))
    view = warp_homography(
        Image.fromarray(image), transform, (view_h, view_w), step=step, edge=True
    )
    raw = predict_grid(dewarper, np.ascontiguousarray(view))
    local = _flattener_values(raw, dewarper.config.grid_convention, (view_h, view_w))
    view_points = (np.asarray(local, dtype=np.float64) + 1.0) / 2.0 * np.array([view_w, view_h])
    source = apply_homography(np.linalg.inv(transform), view_points)
    record["rectangle"] = np.round(rectangle, 2).tolist()
    return 2.0 * source / np.array([width, height], dtype=np.float64) - 1.0, record


def rerun_grid(
    dewarper: Dewarper,
    image: NDArray[np.uint8],
    rerun: int,
    deskew: bool = False,
    perspective: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Grid of flattener values of ``image`` after the whole-image prediction and
    ``rerun`` passes, crop passes (``crop_pass``) or with ``perspective`` perspective
    passes (``homography_pass``), with one record per pass (``pass`` from 1); with
    ``snap`` its border then moves onto the page edge (``edges.snap_border``, recorded as
    a last entry with ``snap``)."""
    rerun_record(rerun, deskew, perspective, snap, rect, refine)
    grid = _flattener_values(
        predict_grid(dewarper, image), dewarper.config.grid_convention, image.shape[:2]
    )
    passes = []
    for number in range(1, rerun + 1):
        if perspective:
            grid, record = homography_pass(dewarper, image, grid)
        elif rect:
            margin = RECT_MARGINS[min(number - 1, len(RECT_MARGINS) - 1)]
            grid, record = rect_pass(dewarper, image, grid, margin)
        else:
            grid, record = crop_pass(dewarper, image, grid, deskew)
        passes.append({"pass": number, **record})
    if refine:
        grid, record = refine_pass(dewarper, image, grid)
        passes.append({"refine": record})
    if snap:
        from .edges import snap_border

        grid, record = snap_border(image, grid)
        passes.append({"snap": record})
    return grid, passes


def quantize_rgb(values: NDArray[np.float32], quantize: str) -> NDArray[np.uint8]:
    """Sampled float RGB as uint8, computed in float32 like the renderer.

    ``round``: ``floor(clip(x, 0, 1) * 255 + 0.5)``, exactly ``flattener.render``.
    ``truncate``: ``floor(clip(x, 0, 1) * 255)``, which is ``(x * 255).astype(uint8)``
    for sampled values (the way published DocUNet outputs were saved).
    """
    check_quantize(quantize)
    scaled = np.clip(values, 0, 1).astype(np.float32, copy=False) * np.float32(255)
    if quantize == "round":
        scaled += np.float32(0.5)
    return np.floor(scaled).astype(np.uint8)


def source_inside(backward_map: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Output pixels whose finite map position lies in the source: the renderer's
    valid pixels when no target mask is given."""
    finite = np.isfinite(backward_map).all(axis=-1)
    return finite & (np.abs(np.where(finite[..., None], backward_map, 0)) <= 1).all(axis=-1)


def sample_rgb(
    image: NDArray[np.uint8],
    backward_map: NDArray[np.float32],
    valid: NDArray[np.bool_],
    fill_rgb: tuple[int, int, int],
    quantize: str,
) -> NDArray[np.uint8]:
    """``flattener.render``'s sampling (bilinear, border padding, align_corners=False,
    positions outside ``valid`` read at 0 and filled with ``fill_rgb``), quantized by
    ``quantize``. With ``round`` and the renderer's valid mask it equals its image."""
    grid = torch.from_numpy(np.where(valid[..., None], backward_map, 0).astype(np.float32))
    source = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.inference_mode():
        sampled = F.grid_sample(
            source, grid.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=False
        )
    output = quantize_rgb(sampled[0].permute(1, 2, 0).numpy(), quantize)
    output[~valid] = fill_rgb
    return output


def render_unchecked(
    image: NDArray[np.uint8],
    backward_map: NDArray[np.float32],
    fill_rgb: tuple[int, int, int],
    quantize: str = "round",
) -> NDArray[np.uint8]:
    """``flattener.render`` sampling without its fold/coverage rejection."""
    return sample_rgb(image, backward_map, source_inside(backward_map), fill_rgb, quantize)


def rectify(
    dewarper: Dewarper,
    image: NDArray[np.uint8],
    output_size: tuple[int, int] | None = None,
    *,
    render_config: RenderConfig | None = None,
    allow_unchecked: bool = False,
    quantize: str = "round",
    inset: Inset = None,
    rerun: int = 0,
    deskew: bool = False,
    perspective: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
) -> RenderResult:
    """Predict and render through ``flattener.render`` and its checks.

    A failed check follows ``render_config.on_error`` (default ``RenderConfig()``:
    raise). With ``allow_unchecked``, a map that fails a geometry check
    (``invalid_transform``) is sampled anyway: the result keeps the failed status and
    reason code, carries the image and sets ``used_fallback`` and
    ``metadata["forced_render"]``. Out-of-source output pixels get ``fill_rgb``.

    ``quantize`` (``QUANTIZE_MODES``) only sets how the sampled image becomes uint8:
    ``round`` keeps ``flattener.render``'s image; ``truncate`` samples the same float
    image over the same valid pixels and stores ``floor(clip(x, 0, 1) * 255)``. Checks,
    statuses and the failure policy are the same; ``metadata["dewarp"]["quantize"]``
    records the mode.

    ``inset`` (``resolve_inset``) moves the outline inward on the backward map before it
    is rendered (``dense_backward_map``), so every check runs on the inset map;
    ``metadata["dewarp"]["inset"]`` records it (``inset_record``).

    ``rerun`` crop passes (``rerun_grid``, with ``deskew``) refine the grid before it is
    densified; the checks run on the final composed map, which samples ``image`` once.
    ``metadata["dewarp"]["rerun"]`` records the settings (``rerun_record``) and
    ``metadata["dewarp"]["crops"]`` the passes (empty for no passes). ``rerun=0`` is the
    whole-image prediction alone, unchanged.
    """
    started = time.perf_counter()
    config = RenderConfig() if render_config is None else render_config
    if not isinstance(config, RenderConfig) or config.on_error not in (
        "raise",
        "return",
        "original",
    ):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_config", "render_config must be RenderConfig"
        )
    check_quantize(quantize)
    inset_info = inset_record(inset)
    rerun_info = rerun_record(rerun, deskew, perspective, snap, rect, refine)
    _check_image(image)
    size = tuple(image.shape[:2]) if output_size is None else tuple(output_size)
    convention = dewarper.config.grid_convention
    grid, crops = None, []
    checked_config = replace(config, on_error="return")
    try:
        validate_config(config)
        size = validate_size(
            size, max_pixels=config.max_output_pixels, max_side=config.max_side, name="output size"
        )
    except FlattenerError as error:
        metadata = _new_metadata(checked_config)
        metadata.update(
            source_size=list(image.shape[:2]),
            requested_output_size=list(size),
            failure_stage="prepare_input",
        )
        metadata["timings"]["prepare_input"] = time.perf_counter() - started
        metadata["timings"]["total"] = time.perf_counter() - started
        result = _handle_failure(error, checked_config, image=None, metadata=metadata)
    else:
        if rerun == 0 and not snap and not refine:
            grid = predict_grid(dewarper, image)
            backward_map = dense_backward_map(grid, convention, size, image.shape[:2], inset)
        else:
            grid, crops = rerun_grid(
                dewarper, image, rerun, deskew, perspective, snap, rect, refine
            )
            backward_map = dense_backward_map(
                grid, convention, size, image.shape[:2], inset, flattener_values=True
            )
        result = render(image, backward_map, config=checked_config)
    if result.status != Status.SUCCESS:
        if allow_unchecked and result.status == Status.INVALID_TRANSFORM:
            result.image = render_unchecked(image, backward_map, config.fill_rgb, quantize)
            result.used_fallback = True
            result.metadata["forced_render"] = True
        elif config.on_error == "raise":
            raise FlattenerError(
                result.status, result.reason_code, result.message, metadata=result.metadata
            )
        elif config.on_error == "original":
            result.image, result.used_fallback = image.copy(), True
    elif quantize != "round":  # the checked image's samples, quantized the other way
        result.image = sample_rgb(image, backward_map, result.valid_mask, config.fill_rgb, quantize)
    result.metadata["dewarp"] = {
        "model_sha256": dewarper.info.get("sha256"),
        "grid_convention": dewarper.config.grid_convention,
        "grid_size": None if grid is None else list(grid.shape[:2]),
        "output_size": list(size),
        "quantize": quantize,
        "inset": inset_info,
        "rerun": rerun_info,
        "crops": crops,
        "seconds": time.perf_counter() - started,
    }
    return result


def load_rgb(path: str | Path) -> NDArray[np.uint8]:
    """The image at ``path`` as RGB uint8, turned by its EXIF orientation."""
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))


# rectifier(image, output_size) -> RenderResult: one image of a batch through a pipeline.
Rectifier = Callable[[NDArray[np.uint8], "tuple[int, int] | None"], RenderResult]


@dataclass(frozen=True, slots=True)
class BatchInput:
    """One image of a batch: output file name, input label and a loader."""

    output: str
    source: str
    load: Callable[[], NDArray[np.uint8]]
    output_size: tuple[int, int] | None = None  # None: the input size


def _validation_summary(result: RenderResult) -> dict[str, Any] | None:
    stats = result.metadata.get("validation")
    if not isinstance(stats, dict):
        return None
    keys = ("out_of_bounds_fraction", "failed_cells", "checked_cells", "min_jacobian_ratio")
    return {key: stats.get(key) for key in keys}


def _count(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "items": len(items),
        "written": sum(item["action"] == "written" for item in items),
        "kept": sum(item["action"] == "kept" for item in items),
        "failed": sum(item["status"] != Status.SUCCESS for item in items),
        "forced": sum(bool(item.get("forced")) for item in items),
    }


def _plain(value: Any) -> Any:
    """``value`` as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(value))


def _input_sha256(image: NDArray[np.uint8]) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _still_matches(
    previous: dict[str, Any],
    task: BatchInput,
    image: NDArray[np.uint8],
    digest: str,
    force_render: bool,
) -> bool:
    """Whether an earlier item was made from this very input: same label, pixels, size
    and output size. A forced image is kept only by a run that also forces."""
    if previous.get("action") not in ("written", "kept"):
        return False
    size = list(image.shape[:2])
    output_size = size if task.output_size is None else list(task.output_size)
    return (
        previous.get("input") == task.source
        and previous.get("input_sha256") == digest
        and previous.get("size") == size
        and previous.get("output_size") == output_size
        and (force_render or not previous.get("forced"))
    )


def _batch_item(
    task: BatchInput,
    target: Path,
    previous: dict[str, Any] | None,
    force_render: bool,
    rectify_one: Rectifier,
) -> dict[str, Any]:
    """Keep, render or refuse one input; the file at ``target`` always matches the item.

    ``previous`` comes only from a record of the same pipeline, render settings,
    quantization, inset and rerun settings. ``rectify_one`` renders the image."""
    item: dict[str, Any] = {"input": task.source, "output": task.output}
    try:
        image = task.load()
        digest = _input_sha256(image)
        if previous is not None and target.is_file():
            if _still_matches(previous, task, image, digest, force_render):
                return {**previous, "action": "kept"}
        result = rectify_one(image, task.output_size)
    except (FlattenerError, OSError) as error:
        target.unlink(missing_ok=True)  # an older image no longer matches this input
        status = error.status if isinstance(error, FlattenerError) else Status.IO_ERROR
        reason = error.reason_code if isinstance(error, FlattenerError) else "unreadable_input"
        item.update(action="not_written", status=str(status), reason=reason, forced=False)
        item["message"] = str(error)
        return item
    if result.image is not None:
        atomic_write(
            target,
            lambda stream, pixels=result.image: Image.fromarray(pixels).save(
                stream, "PNG", compress_level=1
            ),
        )
    else:
        target.unlink(missing_ok=True)  # an older image no longer matches the record
    dewarp = result.metadata.get("dewarp") or {}
    size = task.output_size
    output_size = dewarp.get("output_size") or list(image.shape[:2] if size is None else size)
    item.update(
        action="written" if result.image is not None else "not_written",
        status=str(result.status),
        reason=result.reason_code,
        forced=bool(result.metadata.get("forced_render")),
        size=list(image.shape[:2]),
        input_sha256=digest,
        output_size=list(output_size),
        validation=_validation_summary(result),
        crops=dewarp.get("crops", []),
        seconds=dewarp.get("seconds"),
    )
    extra = result.metadata.get("batch_item")
    if isinstance(extra, dict):
        item.update({key: value for key, value in extra.items() if key not in item})
    return item


def prediction_evidence(
    dewarper: Dewarper,
    quantize: str,
    inset: Inset = None,
    rerun: int = 0,
    deskew: bool = False,
    pipeline: dict[str, Any] | None = None,
    perspective: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
) -> dict[str, Any]:
    """The evidence block of a prediction record, with its output quantization, inset,
    rerun settings and pipeline."""
    return _evidence_block(
        dewarper,
        quantize,
        inset_record(inset),
        rerun_record(rerun, deskew, perspective, snap),
        pipeline,
    )


def _evidence_block(
    dewarper: Dewarper,
    quantize: str,
    inset_info: dict[str, Any],
    rerun_info: dict[str, Any],
    pipeline: dict[str, Any] | None,
) -> dict[str, Any]:
    block = {
        **evidence("predictions", dewarper.info),
        "quantize": quantize,
        "inset": inset_info,
        "rerun": rerun_info,
        "pipeline": pipeline_record(pipeline),
    }
    if dewarper.tta != "none":  # records made without it keep their shape
        block["tta"] = dewarper.tta
    return block


def _protocol_settings(
    protocol: Any, quantize: str, inset: Inset, rerun: int, deskew: bool
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """``(quantize, inset record, rerun record)`` of a pipeline that states its
    ``protocol``. Arguments left at their defaults take the protocol's values; others must
    agree with it."""
    if (
        not isinstance(protocol, dict)
        or protocol.get("quantize") not in QUANTIZE_MODES
        or not isinstance(protocol.get("inset"), dict)
        or not isinstance(protocol.get("rerun"), dict)
    ):
        raise _rerun_error("a pipeline protocol needs quantize, inset and rerun records")
    stated = protocol["rerun"]
    disagree = []
    if quantize != "round" and quantize != protocol["quantize"]:
        disagree.append(f"quantize {quantize!r}")
    if resolve_inset(inset) not in (NO_INSET, recorded_inset(protocol["inset"])):
        disagree.append(f"inset {resolve_inset(inset)}")
    if (rerun, deskew) not in ((0, False), (stated.get("passes"), stated.get("deskew"))):
        disagree.append(f"rerun {rerun} (deskew {deskew})")
    if disagree:
        raise _rerun_error(
            f"{', '.join(disagree)} disagree with the pipeline's protocol {protocol}"
        )
    return protocol["quantize"], dict(protocol["inset"]), dict(stated)


def pipeline_record(pipeline: dict[str, Any] | None = None) -> dict[str, Any]:
    """The pipeline identity of a predictions record: ``DEFAULT_PIPELINE`` (``rectify``)
    for ``None``, else a JSON dict with a non-empty string ``name`` and the settings
    that make its outputs differ from other pipelines' (checked for JSON only)."""
    if pipeline is None:
        return dict(DEFAULT_PIPELINE)
    if not isinstance(pipeline, dict) or not isinstance(pipeline.get("name"), str):
        raise _rerun_error("pipeline must be a dict with a string 'name'")
    if not pipeline["name"]:
        raise _rerun_error("pipeline name must not be empty")
    try:
        return _plain(pipeline)
    except (TypeError, ValueError) as error:
        raise _rerun_error(f"pipeline must be JSON data: {error}") from None


def recorded_pipeline(record: dict[str, Any] | None) -> dict[str, Any]:
    """The pipeline of a predictions record; a record without one predates the option
    and was made by ``rectify``."""
    value = record.get("pipeline") if isinstance(record, dict) else None
    return value if isinstance(value, dict) else dict(DEFAULT_PIPELINE)


class _Serial:
    """``ThreadPoolExecutor``'s ``map`` in the calling thread."""

    def __enter__(self) -> _Serial:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def map(self, function: Callable[..., Any], items: Any) -> Iterator[Any]:
        return map(function, items)


def rectify_batch(
    dewarper: Dewarper,
    inputs: list[BatchInput],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    force_render: bool = False,
    render_config: RenderConfig | None = None,
    fields: dict[str, Any] | None = None,
    quantize: str = "round",
    inset: Inset = None,
    rerun: int = 0,
    deskew: bool = False,
    rectifier: Rectifier | None = None,
    pipeline: dict[str, Any] | None = None,
    perspective: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Rectify ``inputs`` into ``output_dir`` and record every item in ``predictions.json``.

    Each item records the check status and reason code and the sha256 of the decoded
    input. An image that fails a check is written only with ``force_render``
    (``forced: true``). Without ``overwrite``, an output that ``predictions.json`` lists
    as written is kept with its recorded status when the weights, pipeline, render
    settings, quantization, inset, rerun settings, input label, input pixels and sizes are
    unchanged (a forced image only by a forced run); every other input is rendered again.
    A record without ``quantize`` predates the option and holds rounded images; one
    without ``inset`` holds images without inset; one without ``rerun`` holds whole-image
    predictions; one without ``pipeline`` was made by ``rectify``. A directory holding
    predictions of other weights, of another pipeline or of an older format, or outputs
    without a ``predictions.json``, is refused. The report names the current weights,
    pipeline, quantization, inset and rerun settings before any image is written, and says
    ``complete: false`` until the last input is done; images are written atomically. Each
    item records its crop passes (``crops``, see ``crop_pass``).

    ``rectifier(image, output_size)`` replaces ``rectify`` for another pipeline and needs
    a ``pipeline`` record (``pipeline_record``); it returns a ``RenderResult`` that
    follows the batch rules itself: no image for a failed check unless it forces one
    (``metadata["forced_render"]``). ``metadata["dewarp"]`` may give ``output_size``,
    ``crops`` and ``seconds``, and ``metadata["batch_item"]`` extra item fields. The
    settings arguments are then only recorded. When the pipeline record has a
    ``protocol``, the recorded quantization, inset and rerun settings are its own;
    arguments that are not the defaults must agree with it.

    ``workers`` above 1 rectifies that many inputs at once in threads (``rectify`` only;
    model calls take turns); items are recorded in input order and every image is the
    one a single worker makes. Images are PNG at zlib level 1.
    """
    check_quantize(quantize)
    if workers < 1 or (workers > 1 and rectifier is not None):
        raise _rerun_error("workers must be 1, or more with the default rectify")
    if rectifier is not None and pipeline is None:
        raise _rerun_error("a rectifier needs a pipeline record")
    if rectifier is None and pipeline is not None and pipeline_record(pipeline) != DEFAULT_PIPELINE:
        raise _rerun_error("a pipeline other than rectify needs its rectifier")
    pipeline_info = pipeline_record(pipeline)
    if rectifier is not None and "protocol" in pipeline_info:
        quantize, inset_info, rerun_info = _protocol_settings(
            pipeline_info["protocol"], quantize, inset, rerun, deskew
        )
    else:
        inset_info = inset_record(inset)
        rerun_info = rerun_record(rerun, deskew, perspective, snap, rect, refine)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "predictions.json"
    config = RenderConfig() if render_config is None else render_config
    config = replace(config, on_error="return")
    previous_items: dict[str, dict[str, Any]] = {}
    if not overwrite and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("model", {}).get("sha256") != dewarper.info["sha256"]:
            raise FlattenerError(
                Status.IO_ERROR, "output_exists", f"{output_dir} holds predictions of other weights"
            )
        if previous.get("schema") != PREDICTIONS_SCHEMA:
            raise FlattenerError(
                Status.IO_ERROR,
                "output_exists",
                f"{report_path} has an older format; use --overwrite",
            )
        if recorded_pipeline(previous) != pipeline_info:
            raise FlattenerError(
                Status.IO_ERROR,
                "output_exists",
                f"{output_dir} holds predictions of another pipeline "
                f"({recorded_pipeline(previous).get('name')!r}); use --overwrite",
            )
        same_quantize = previous.get("quantize", "round") == quantize
        same_inset = recorded_inset(previous.get("inset")) == recorded_inset(inset_info)
        same_rerun = recorded_rerun(previous.get("rerun")) == rerun_info
        same_render = previous.get("render_config") == _plain(asdict(config))
        if same_render and same_quantize and same_inset and same_rerun:
            previous_items = {item["output"]: item for item in previous.get("items", [])}
    elif not overwrite and any((output_dir / item.output).is_file() for item in inputs):
        raise FlattenerError(
            Status.IO_ERROR,
            "output_exists",
            f"{output_dir} holds outputs without predictions.json; use --overwrite",
        )
    if rectifier is None:

        def rectifier(image: NDArray[np.uint8], size: tuple[int, int] | None) -> RenderResult:
            return rectify(
                dewarper,
                image,
                size,
                render_config=config,
                allow_unchecked=force_render,
                quantize=quantize,
                inset=inset,
                rerun=rerun,
                deskew=deskew,
                perspective=perspective,
                snap=snap,
                rect=rect,
                refine=refine,
            )

    report = {
        "schema": PREDICTIONS_SCHEMA,
        "created_utc": utc_now(),
        "model": dewarper.info,
        "evidence": _evidence_block(dewarper, quantize, inset_info, rerun_info, pipeline),
        **(fields or {}),
        "force_render": force_render,
        "quantize": quantize,
        "inset": inset_info,
        "rerun": rerun_info,
        "pipeline": pipeline_info,
        "render_config": asdict(config),
        "complete": False,
        "counts": _count([]),
        "items": [],
    }
    write_json(report_path, report)

    def one(task: BatchInput) -> dict[str, Any]:
        previous = previous_items.get(task.output)
        return _batch_item(task, output_dir / task.output, previous, force_render, rectifier)

    with ThreadPoolExecutor(workers) if workers > 1 else _Serial() as pool:
        for item in pool.map(one, inputs):
            report["items"].append(item)
            report["counts"] = _count(report["items"])
            write_json(report_path, report)
    report["complete"] = True
    write_json(report_path, report)
    return report


def docunet_inputs(input_dir: str | Path) -> list[BatchInput]:
    """``{k}_{m} copy.png`` crops of ``input_dir`` as batch inputs for ``{k}_{m}.png``."""
    input_dir = Path(input_dir)
    names = []
    for path in input_dir.iterdir():
        match = DOCUNET_INPUT.match(path.name)
        if match:
            names.append((int(match[1]), int(match[2]), path))
    if not names:
        raise FlattenerError(
            Status.INPUT_ERROR, "no_inputs", f"no '{{k}}_{{m}} copy.png' in {input_dir}"
        )
    return [
        BatchInput(f"{k}_{m}.png", path.name, lambda path=path: load_rgb(path))
        for k, m, path in sorted(names)
    ]


def rectify_docunet(
    dewarper: Dewarper,
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    force_render: bool = False,
    quantize: str = "round",
    inset: Inset = None,
    rerun: int = 0,
    deskew: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
) -> dict[str, Any]:
    """Rectify every ``{k}_{m} copy.png`` into ``{k}_{m}.png`` at the crop's own size.

    See ``rectify_batch`` for what is kept, refused and recorded.
    """
    return rectify_batch(
        dewarper,
        docunet_inputs(input_dir),
        output_dir,
        overwrite=overwrite,
        force_render=force_render,
        fields={"input_dir": str(Path(input_dir)), "inputs": "docunet_crop"},
        quantize=quantize,
        inset=inset,
        rerun=rerun,
        deskew=deskew,
        snap=snap,
        rect=rect,
        refine=refine,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weights", required=True, help="dewarp checkpoint or UVDoc best_model.pkl"
    )
    parser.add_argument("--device", default="cpu")
    commands = parser.add_subparsers(dest="command", required=True)
    force_help = (
        "write the image even when the map fails the geometry checks; the failure is "
        "still recorded and the exit status is 1"
    )
    quantize_help = (
        "uint8 conversion of the sampled image: round (default, flattener.render) or "
        "truncate (floor(x*255), the paper protocol); checks and exit status are the same"
    )
    inset_help = (
        "move the page outline inward before rendering: a preset ("
        + ", ".join(INSET_PRESETS)
        + "), E for both axes or EX,EY, in normalized units (each side moves in by E/2 of "
        "the page extent); checks and exit status are the same; default 0"
    )
    rerun_help = (
        "crop passes after the whole-image prediction: crop the predicted outline's box "
        f"plus {RERUN_MARGIN:.0%}% of the page extent per side, predict again and compose; "
        "checks and exit status are the same; default 0"
    )
    deskew_help = (
        "with --rerun: turn each crop upright when the page angle estimated from the "
        f"outline's top and bottom edges is usable and at most {DESKEW_MAX_DEG:g} degrees"
    )
    single = commands.add_parser("image")
    single.add_argument("--input", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--output-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    single.add_argument("--force-render", action="store_true", help=force_help)
    single.add_argument("--quantize", choices=QUANTIZE_MODES, default="round", help=quantize_help)
    single.add_argument("--inset", type=parse_inset, default=None, help=inset_help)
    single.add_argument("--rerun", type=parse_rerun, default=0, help=rerun_help)
    single.add_argument("--deskew", action="store_true", help=deskew_help)
    batch = commands.add_parser("docunet")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--overwrite", action="store_true")
    batch.add_argument("--force-render", action="store_true", help=force_help)
    batch.add_argument("--quantize", choices=QUANTIZE_MODES, default="round", help=quantize_help)
    batch.add_argument("--inset", type=parse_inset, default=None, help=inset_help)
    batch.add_argument("--rerun", type=parse_rerun, default=0, help=rerun_help)
    batch.add_argument("--deskew", action="store_true", help=deskew_help)
    snap_help = "snap the final border to the photo's page edges (edges.snap_border)"
    single.add_argument("--snap-border", action="store_true", help=snap_help)
    batch.add_argument("--snap-border", action="store_true", help=snap_help)
    rect_help = "with --rerun: turned tight-crop passes (rect_pass, margins RECT_MARGINS)"
    single.add_argument("--rect", action="store_true", help=rect_help)
    batch.add_argument("--rect", action="store_true", help=rect_help)
    refine_help = "an interior pass on the flat output that keeps the outline (refine_pass)"
    single.add_argument("--refine", action="store_true", help=refine_help)
    batch.add_argument("--refine", action="store_true", help=refine_help)
    tta_help = "test-time augmentation of every model call: flip averages in the mirror image"
    single.add_argument("--tta", choices=TTA_MODES, default="none", help=tta_help)
    batch.add_argument("--tta", choices=TTA_MODES, default="none", help=tta_help)
    args = parser.parse_args(argv)
    if args.deskew and args.rerun == 0:
        parser.error("--deskew needs --rerun 1 or more")
    dewarper = with_tta(load_dewarper(args.weights, device=args.device), args.tta)
    if args.command == "image":
        size = None if args.output_size is None else tuple(args.output_size)
        result = rectify(
            dewarper,
            load_rgb(args.input),
            size,
            render_config=RenderConfig(on_error="return"),
            allow_unchecked=args.force_render,
            quantize=args.quantize,
            inset=args.inset,
            rerun=args.rerun,
            deskew=args.deskew,
            snap=args.snap_border,
            rect=args.rect,
            refine=args.refine,
        )
        output = Path(args.output)
        removed = False
        if result.image is not None:
            Image.fromarray(result.image).save(output)
        elif output.is_file():  # an earlier image there is not a result of this run
            output.unlink()
            removed = True
        summary = {
            "output": None if result.image is None else args.output,
            "removed_earlier_output": removed,
            "status": str(result.status),
            "reason": result.reason_code,
            "forced": bool(result.metadata.get("forced_render")),
            "quantize": args.quantize,
            "inset": inset_record(args.inset),
            "rerun": rerun_record(
                args.rerun, args.deskew, snap=args.snap_border, rect=args.rect, refine=args.refine
            ),
            "crops": result.metadata["dewarp"]["crops"],
            "evidence": prediction_evidence(
                dewarper,
                args.quantize,
                args.inset,
                args.rerun,
                args.deskew,
                snap=args.snap_border,
                rect=args.rect,
                refine=args.refine,
            ),
        }
        print(json.dumps(summary))
        return 0 if result.status == Status.SUCCESS else EXIT_FAILED
    report = rectify_docunet(
        dewarper,
        args.input_dir,
        args.output_dir,
        overwrite=args.overwrite,
        force_render=args.force_render,
        quantize=args.quantize,
        inset=args.inset,
        rerun=args.rerun,
        deskew=args.deskew,
        snap=args.snap_border,
        rect=args.rect,
        refine=args.refine,
    )
    print(json.dumps(report["counts"]))
    return 0 if report["counts"]["failed"] == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
