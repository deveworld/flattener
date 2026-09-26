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
from .coords import grid_to_map, uvdoc_dense_map
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
EXIT_FAILED = 1  # some image failed its checks (also when it was force-rendered)


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


def dense_backward_map(
    grid: NDArray[np.float32],
    convention: str,
    output_size: tuple[int, int],
    source_size: tuple[int, int],
) -> NDArray[np.float32]:
    """Flattener backward map at ``output_size`` for a grid in ``convention``."""
    if convention == "uvdoc":
        return uvdoc_dense_map(grid, output_size, source_size)
    return grid_to_map(grid, output_size)


def render_unchecked(
    image: NDArray[np.uint8], backward_map: NDArray[np.float32], fill_rgb: tuple[int, int, int]
) -> NDArray[np.uint8]:
    """``flattener.render`` sampling without its fold/coverage rejection."""
    finite = np.isfinite(backward_map).all(axis=-1)
    inside = finite & (np.abs(np.where(finite[..., None], backward_map, 0)) <= 1).all(axis=-1)
    grid = torch.from_numpy(np.where(inside[..., None], backward_map, 0).astype(np.float32))
    source = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.inference_mode():
        sampled = F.grid_sample(
            source, grid.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=False
        )
    values = np.floor(np.clip(sampled[0].permute(1, 2, 0).numpy(), 0, 1) * 255 + 0.5)
    output = values.astype(np.uint8)
    output[~inside] = fill_rgb
    return output


def rectify(
    dewarper: Dewarper,
    image: NDArray[np.uint8],
    output_size: tuple[int, int] | None = None,
    *,
    render_config: RenderConfig | None = None,
    allow_unchecked: bool = False,
) -> RenderResult:
    """Predict and render through ``flattener.render`` and its checks.

    A failed check follows ``render_config.on_error`` (default ``RenderConfig()``:
    raise). With ``allow_unchecked``, a map that fails a geometry check
    (``invalid_transform``) is sampled anyway: the result keeps the failed status and
    reason code, carries the image and sets ``used_fallback`` and
    ``metadata["forced_render"]``. Out-of-source output pixels get ``fill_rgb``.
    """
    started = time.perf_counter()
    config = RenderConfig() if render_config is None else render_config
    if not isinstance(config, RenderConfig):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_config", "render_config must be RenderConfig"
        )
    size = tuple(image.shape[:2]) if output_size is None else tuple(output_size)
    grid = predict_grid(dewarper, image)
    backward_map = dense_backward_map(grid, dewarper.config.grid_convention, size, image.shape[:2])
    result = render(image, backward_map, config=replace(config, on_error="return"))
    if result.status != Status.SUCCESS:
        if allow_unchecked and result.status == Status.INVALID_TRANSFORM:
            result.image = render_unchecked(image, backward_map, config.fill_rgb)
            result.used_fallback = True
            result.metadata["forced_render"] = True
        elif config.on_error == "raise":
            raise FlattenerError(
                result.status, result.reason_code, result.message, metadata=result.metadata
            )
        elif config.on_error == "original":
            result.image, result.used_fallback = image.copy(), True
    result.metadata["dewarp"] = {
        "model_sha256": dewarper.info.get("sha256"),
        "grid_convention": dewarper.config.grid_convention,
        "grid_size": list(grid.shape[:2]),
        "output_size": list(size),
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
) -> dict[str, Any]:
    """Keep, render or refuse one input; the file at ``target`` always matches the item."""
    item: dict[str, Any] = {"input": task.source, "output": task.output}
    try:
        image = task.load()
        digest = _input_sha256(image)
        if previous is not None and target.is_file():
            if _still_matches(previous, task, image, digest, force_render):
                return {**previous, "action": "kept"}
        result = rectify(
            dewarper, image, task.output_size, render_config=config, allow_unchecked=force_render
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


def rectify_batch(
    dewarper: Dewarper,
    inputs: list[BatchInput],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    force_render: bool = False,
    render_config: RenderConfig | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rectify ``inputs`` into ``output_dir`` and record every item in ``predictions.json``.

    Each item records the check status and reason code and the sha256 of the decoded
    input. An image that fails a check is written only with ``force_render``
    (``forced: true``). Without ``overwrite``, an output that ``predictions.json`` lists
    as written is kept with its recorded status when the weights, render settings, input
    label, input pixels and sizes are unchanged (a forced image only by a forced run);
    every other input is rendered again. A directory holding predictions of other weights
    or of an older format, or outputs without a ``predictions.json``, is refused. The
    report names the current weights before any image is written, and says
    ``complete: false`` until the last input is done; images are written atomically.
    """
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
        if previous.get("render_config") == _plain(asdict(config)):
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
        "evidence": evidence("predictions", dewarper.info),
        **(fields or {}),
        "force_render": force_render,
        "render_config": asdict(config),
        "complete": False,
        "counts": _count([]),
        "items": [],
    }
    write_json(report_path, report)
    for task in inputs:
        target = output_dir / task.output
        previous = previous_items.get(task.output)
        report["items"].append(_batch_item(dewarper, task, target, previous, config, force_render))
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
    single = commands.add_parser("image")
    single.add_argument("--input", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--output-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    single.add_argument("--force-render", action="store_true", help=force_help)
    batch = commands.add_parser("docunet")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--overwrite", action="store_true")
    batch.add_argument("--force-render", action="store_true", help=force_help)
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
            "evidence": evidence("predictions", dewarper.info),
        }
        print(json.dumps(summary))
        return 0 if result.status == Status.SUCCESS else EXIT_FAILED
    report = rectify_docunet(
        dewarper,
        args.input_dir,
        args.output_dir,
        overwrite=args.overwrite,
        force_render=args.force_render,
    )
    print(json.dumps(report["counts"]))
    return 0 if report["counts"]["failed"] == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
