"""Rectify RGB images with a dewarp checkpoint or the upstream UVDoc MIT weights.

``python -m flattener.dewarp.predict --weights W image --input a.jpg --output b.png``
``python -m flattener.dewarp.predict --weights W docunet --input-dir crop --output-dir out``

The image is resized to the model input (bilinear, align_corners=False, no
antialias: upstream's ``cv2.resize`` INTER_LINEAR), the 2-D grid is predicted,
converted to a dense flattener backward map at the requested output size
(default: the input size) and rendered by ``flattener.render``. Upstream UVDoc
weights (``grid_convention="uvdoc"``) use the exact upstream densification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.nn import functional as F

from ..contracts import FlattenerError, RenderConfig, RenderResult, Status
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
    with torch.inference_mode():
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
    """Predict and render. Out-of-source output pixels get ``fill_rgb``.

    A fold/degenerate map raises (per ``render_config.on_error``) unless
    ``allow_unchecked``, which samples anyway and sets ``used_fallback``.
    """
    started = time.perf_counter()
    size = tuple(image.shape[:2]) if output_size is None else tuple(output_size)
    grid = predict_grid(dewarper, image)
    backward_map = dense_backward_map(grid, dewarper.config.grid_convention, size, image.shape[:2])
    config = (
        RenderConfig(max_out_of_bounds_fraction=1.0) if render_config is None else render_config
    )
    try:
        result = render(image, backward_map, config=config)
    except FlattenerError as error:
        if not allow_unchecked or error.status != Status.INVALID_TRANSFORM:
            raise
        output = render_unchecked(image, backward_map, config.fill_rgb)
        result = RenderResult(
            Status.SUCCESS,
            image=output,
            used_fallback=True,
            metadata={"render_fallback": error.reason_code},
        )
    result.metadata["dewarp"] = {
        "model_sha256": dewarper.info.get("sha256"),
        "grid_convention": dewarper.config.grid_convention,
        "grid_size": list(grid.shape[:2]),
        "output_size": list(size),
        "seconds": time.perf_counter() - started,
    }
    return result


def load_rgb(path: str | Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def rectify_docunet(
    dewarper: Dewarper, input_dir: str | Path, output_dir: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Rectify every ``{k}_{m} copy.png`` into ``{k}_{m}.png`` at the crop's own size.

    Without ``overwrite``, outputs that ``predictions.json`` lists for the same
    weights are kept (with their recorded details) and other outputs are
    re-rendered; a directory holding predictions of other weights, or outputs
    without a ``predictions.json``, is refused. The report names the current
    weights before any image is written, and images are written atomically.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    names = []
    for path in input_dir.iterdir():
        match = DOCUNET_INPUT.match(path.name)
        if match:
            names.append((int(match[1]), int(match[2]), path))
    if not names:
        raise FlattenerError(
            Status.INPUT_ERROR, "no_inputs", f"no '{{k}}_{{m}} copy.png' in {input_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "predictions.json"
    previous_items: dict[str, dict[str, Any]] = {}
    if not overwrite and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("model", {}).get("sha256") != dewarper.info["sha256"]:
            raise FlattenerError(
                Status.IO_ERROR, "output_exists", f"{output_dir} holds predictions of other weights"
            )
        previous_items = {item["output"]: item for item in previous.get("items", [])}
    elif not overwrite and any((output_dir / f"{k}_{m}.png").is_file() for k, m, _ in names):
        raise FlattenerError(
            Status.IO_ERROR,
            "output_exists",
            f"{output_dir} holds outputs without predictions.json; use --overwrite",
        )
    report = {
        "schema": "flattener-dewarp-docunet-1",
        "created_utc": utc_now(),
        "model": dewarper.info,
        "input_dir": str(input_dir),
        "items": [],
    }
    write_json(report_path, report)
    for k, m, path in sorted(names):
        target = output_dir / f"{k}_{m}.png"
        item = {"input": path.name, "output": target.name}
        if target.is_file() and target.name in previous_items:
            item = {**previous_items[target.name], **item, "status": "kept"}
        else:
            image = load_rgb(path)
            result = rectify(dewarper, image, allow_unchecked=True)
            atomic_write(
                target,
                lambda stream, pixels=result.image: Image.fromarray(pixels).save(stream, "PNG"),
            )
            item.update(
                status="written",
                size=list(image.shape[:2]),
                fallback=result.used_fallback,
                seconds=result.metadata["dewarp"]["seconds"],
            )
        report["items"].append(item)
        write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weights", required=True, help="dewarp checkpoint or UVDoc best_model.pkl"
    )
    parser.add_argument("--device", default="cpu")
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("image")
    single.add_argument("--input", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--output-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    batch = commands.add_parser("docunet")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    dewarper = load_dewarper(args.weights, device=args.device)
    if args.command == "image":
        size = None if args.output_size is None else tuple(args.output_size)
        result = rectify(dewarper, load_rgb(args.input), size, allow_unchecked=True)
        Image.fromarray(result.image).save(args.output)
        print(json.dumps({"output": args.output, "fallback": result.used_fallback}))
    else:
        report = rectify_docunet(
            dewarper, args.input_dir, args.output_dir, overwrite=args.overwrite
        )
        written = sum(item["status"] == "written" for item in report["items"])
        print(json.dumps({"items": len(report["items"]), "written": written}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
