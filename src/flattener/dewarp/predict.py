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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Iterator
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
from ..renderer import render
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


@dataclass(slots=True)
class Dewarper:
    model: UVDocNet
    config: DewarpModelConfig
    device: torch.device
    info: dict[str, Any] = field(default_factory=dict)


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


def predict_grid(dewarper: Dewarper, image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Raw 2-D model output as (Gh, Gw, 2), in the model's grid convention."""
    _check_image(image)
    tensor = torch.tensor(image).to(dewarper.device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255)
    tensor = F.interpolate(
        tensor,
        size=dewarper.config.input_size,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    with torch.inference_mode(), ieee_float32():
        grid2d, _ = dewarper.model(tensor)
    return grid2d[0].permute(1, 2, 0).float().cpu().numpy()


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
) -> NDArray[np.float32]:
    """Flattener backward map at ``output_size`` for a grid in ``convention``.

    With an inset ``(ex, ey)`` (``resolve_inset``), output coordinate u = (ux, uy) in
    [-1, 1] gets the value that the map without inset has at ((1 - ex) ux, (1 - ey) uy):
    the grid is sampled on the page region [-(1 - e), 1 - e] and the outline moves in by
    e / 2 of the page extent per side. No inset returns the plain map unchanged."""
    ex, ey = resolve_inset(inset)
    if (ex, ey) == NO_INSET:
        if convention == "uvdoc":
            return uvdoc_dense_map(grid, output_size, source_size)
        return grid_to_map(grid, output_size)
    array = _grid(grid, "grid")  # the checks of grid_to_map and uvdoc_dense_map
    height, width = _size(output_size, "output_size")
    gh, gw = array.shape[:2]
    rows = interpolation_matrix(_grid_index(height, gh, ey, convention), gh, extrapolate=False)
    cols = interpolation_matrix(_grid_index(width, gw, ex, convention), gw, extrapolate=False)
    dense = _apply(rows, array, cols)
    if convention == "uvdoc":
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
    """
    started = time.perf_counter()
    config = RenderConfig() if render_config is None else render_config
    if not isinstance(config, RenderConfig):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_config", "render_config must be RenderConfig"
        )
    check_quantize(quantize)
    inset_info = inset_record(inset)
    size = tuple(image.shape[:2]) if output_size is None else tuple(output_size)
    grid = predict_grid(dewarper, image)
    convention = dewarper.config.grid_convention
    backward_map = dense_backward_map(grid, convention, size, image.shape[:2], inset)
    result = render(image, backward_map, config=replace(config, on_error="return"))
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
        "grid_size": list(grid.shape[:2]),
        "output_size": list(size),
        "quantize": quantize,
        "inset": inset_info,
        "seconds": time.perf_counter() - started,
    }
    return result


def load_rgb(path: str | Path) -> NDArray[np.uint8]:
    """The image at ``path`` as RGB uint8, turned by its EXIF orientation."""
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))


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
    dewarper: Dewarper,
    task: BatchInput,
    target: Path,
    previous: dict[str, Any] | None,
    config: RenderConfig,
    force_render: bool,
    quantize: str,
    inset: Inset = None,
) -> dict[str, Any]:
    """Keep, render or refuse one input; the file at ``target`` always matches the item.

    ``previous`` comes only from a record of the same render settings, quantization and
    inset."""
    item: dict[str, Any] = {"input": task.source, "output": task.output}
    try:
        image = task.load()
        digest = _input_sha256(image)
        if previous is not None and target.is_file():
            if _still_matches(previous, task, image, digest, force_render):
                return {**previous, "action": "kept"}
        result = rectify(
            dewarper,
            image,
            task.output_size,
            render_config=config,
            allow_unchecked=force_render,
            quantize=quantize,
            inset=inset,
        )
    except (FlattenerError, OSError) as error:
        target.unlink(missing_ok=True)  # an older image no longer matches this input
        status = error.status if isinstance(error, FlattenerError) else Status.IO_ERROR
        reason = error.reason_code if isinstance(error, FlattenerError) else "unreadable_input"
        item.update(action="not_written", status=str(status), reason=reason, forced=False)
        item["message"] = str(error)
        return item
    if result.image is not None:
        atomic_write(
            target, lambda stream, pixels=result.image: Image.fromarray(pixels).save(stream, "PNG")
        )
    else:
        target.unlink(missing_ok=True)  # an older image no longer matches the record
    item.update(
        action="written" if result.image is not None else "not_written",
        status=str(result.status),
        reason=result.reason_code,
        forced=bool(result.metadata.get("forced_render")),
        size=list(image.shape[:2]),
        input_sha256=digest,
        output_size=result.metadata["dewarp"]["output_size"],
        validation=_validation_summary(result),
        seconds=result.metadata["dewarp"]["seconds"],
    )
    return item


def prediction_evidence(dewarper: Dewarper, quantize: str, inset: Inset = None) -> dict[str, Any]:
    """The evidence block of a prediction record, with its output quantization and inset."""
    return {
        **evidence("predictions", dewarper.info),
        "quantize": quantize,
        "inset": inset_record(inset),
    }


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
) -> dict[str, Any]:
    """Rectify ``inputs`` into ``output_dir`` and record every item in ``predictions.json``.

    Each item records the check status and reason code and the sha256 of the decoded
    input. An image that fails a check is written only with ``force_render``
    (``forced: true``). Without ``overwrite``, an output that ``predictions.json`` lists
    as written is kept with its recorded status when the weights, render settings,
    quantization, inset, input label, input pixels and sizes are unchanged (a forced image
    only by a forced run); every other input is rendered again. A record without
    ``quantize`` predates the option and holds rounded images; one without ``inset`` holds
    images without inset. A directory holding predictions of other weights or of an older
    format, or outputs without a ``predictions.json``, is refused. The report names the
    current weights, quantization and inset before any image is written, and says
    ``complete: false`` until the last input is done; images are written atomically.
    """
    check_quantize(quantize)
    inset_info = inset_record(inset)
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
        same_quantize = previous.get("quantize", "round") == quantize
        same_inset = recorded_inset(previous.get("inset")) == resolve_inset(inset)
        if previous.get("render_config") == _plain(asdict(config)) and same_quantize and same_inset:
            previous_items = {item["output"]: item for item in previous.get("items", [])}
    elif not overwrite and any((output_dir / item.output).is_file() for item in inputs):
        raise FlattenerError(
            Status.IO_ERROR,
            "output_exists",
            f"{output_dir} holds outputs without predictions.json; use --overwrite",
        )
    report = {
        "schema": PREDICTIONS_SCHEMA,
        "created_utc": utc_now(),
        "model": dewarper.info,
        "evidence": prediction_evidence(dewarper, quantize, inset),
        **(fields or {}),
        "force_render": force_render,
        "quantize": quantize,
        "inset": inset_info,
        "render_config": asdict(config),
        "complete": False,
        "counts": _count([]),
        "items": [],
    }
    write_json(report_path, report)
    for task in inputs:
        target = output_dir / task.output
        previous = previous_items.get(task.output)
        item = _batch_item(dewarper, task, target, previous, config, force_render, quantize, inset)
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
    single = commands.add_parser("image")
    single.add_argument("--input", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--output-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    single.add_argument("--force-render", action="store_true", help=force_help)
    single.add_argument("--quantize", choices=QUANTIZE_MODES, default="round", help=quantize_help)
    single.add_argument("--inset", type=parse_inset, default=None, help=inset_help)
    batch = commands.add_parser("docunet")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--overwrite", action="store_true")
    batch.add_argument("--force-render", action="store_true", help=force_help)
    batch.add_argument("--quantize", choices=QUANTIZE_MODES, default="round", help=quantize_help)
    batch.add_argument("--inset", type=parse_inset, default=None, help=inset_help)
    args = parser.parse_args(argv)
    dewarper = load_dewarper(args.weights, device=args.device)
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
            "evidence": prediction_evidence(dewarper, args.quantize, args.inset),
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
    )
    print(json.dumps(report["counts"]))
    return 0 if report["counts"]["failed"] == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
