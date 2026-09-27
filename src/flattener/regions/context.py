"""Region-context U-Net page gate (``region-context-unet-2.0``).

The architecture is ported from the locator study (``outputs/models/locator.zip``,
``source.tar:src/flattener/regions/context_segmentation.py``). Preprocessing is the pinned
``regions.segmentation`` function: the full image extent, bilinear with antialiasing, as
float32 in [0, 1]. The model returns one page-mask logit map at ``config.input_size``.

Checkpoints need a ``<file>.sha256`` sidecar. Two schemas load:

* ``independent-locator-pilot-1.0``: the study's ``best.pt`` as it was saved.
* ``region-context-1``: weights, config, preprocessing version and a provenance record,
  written by ``save_context_checkpoint``.

``gate_page_region`` turns a probability grid into a page box. It runs ``detect_region``
with ``max_area_fraction=1.0``. A page whose mask covers more than 90% of the frame, or
touches at least three image borders, fills the frame: the box is the whole image and the
reason ``page_fills_frame`` is informational. Rejections from ``detect_region`` keep their
status and reason code.

``smartdoc_exposure`` reads which SmartDoc templates the v0 checkpoint saw in training
(``locator_v0_smartdoc.json``); evaluations of v0 exclude them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import FlattenerError, Status
from .geometry import PageRegion, RegionConfig, RegionEstimate, detect_region
from .segmentation import PREPROCESS_VERSION, SegmenterConfig
from .segmentation import preprocess_image as _preprocess_image
from .segmentation import preprocess_mask as _preprocess_mask

MODEL_VERSION = "region-context-unet-2.0"
MAX_PARAMETERS = 1_000_000
PILOT_SCHEMA = "independent-locator-pilot-1.0"
CHECKPOINT_SCHEMA = "region-context-1"
CHECKPOINT_SCHEMAS = (PILOT_SCHEMA, CHECKPOINT_SCHEMA)
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
GATE_VERSION = "page-gate-1"
SMARTDOC_EXPOSURE_FILE = "locator_v0_smartdoc.json"
BORDERS = ("top", "right", "bottom", "left")


def _parameter_count(channels):
    def block(in_channels, out_channels):
        return out_channels * (9 * in_channels + 1) + out_channels * (9 * out_channels + 1)

    a, b, c, d = channels
    return (
        block(3, a)
        + block(a, b)
        + block(b, c)
        + block(c, d)
        + block(2 * d + c, c)
        + block(c + b, b)
        + block(b + a, a)
        + a
        + 1  # the 1x1 mask head and its bias
    )


@dataclass(frozen=True, slots=True)
class ContextSegmenterConfig:
    """Input size (height, width), four encoder widths and the model version."""

    input_size: tuple[int, int] = (128, 96)
    channels: tuple[int, int, int, int] = (8, 16, 32, 64)
    model_version: str = MODEL_VERSION

    def __post_init__(self):
        for name, count, low, high in (("input_size", 2, 8, 512), ("channels", 4, 1, 128)):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != count
                or any(type(item) is not int or not low <= item <= high for item in value)
            ):
                raise ValueError(f"{name} must be a tuple of {count} integers in [{low}, {high}]")
        if self.model_version != MODEL_VERSION:
            raise ValueError(f"model_version must be {MODEL_VERSION!r}")
        if _parameter_count(self.channels) > MAX_PARAMETERS:
            raise ValueError(f"context model exceeds the {MAX_PARAMETERS:,}-parameter CPU limit")


def _config(config):
    if config is None:
        return ContextSegmenterConfig()
    if not isinstance(config, ContextSegmenterConfig):
        raise TypeError("config must be ContextSegmenterConfig or None")
    return config


def parameter_count(config: ContextSegmenterConfig | None = None) -> int:
    """Trainable parameter count, computed without building the model."""
    return _parameter_count(_config(config).channels)


def preprocess_image(
    image: np.ndarray, config: ContextSegmenterConfig | None = None
) -> torch.Tensor:
    """The pinned v1 RGB preprocessing at ``config.input_size`` (float32 CHW, CPU)."""
    config = _config(config)
    return _preprocess_image(image, SegmenterConfig(input_size=config.input_size))


def preprocess_mask(mask: np.ndarray, config: ContextSegmenterConfig | None = None) -> torch.Tensor:
    """The pinned v1 binary target preprocessing (nearest-exact) at ``config.input_size``."""
    config = _config(config)
    return _preprocess_mask(mask, SegmenterConfig(input_size=config.input_size))


def _block(in_channels, out_channels):
    # Built on CPU in float32 whatever the caller's default dtype or device is.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, device="cpu", dtype=torch.float32),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, device="cpu", dtype=torch.float32),
        nn.ReLU(),
    )


class ContextPageSegmenter(nn.Module):
    """Three-level U-Net whose first decoder also sees an image-wide context vector.

    The bottleneck is average-pooled to N×D×1×1 and broadcast over the first decoder
    grid, so every output position depends on the whole bottleneck. Skip connections
    keep boundary detail. The output is one N×1×H×W logit map.

    The model is built on CPU in float32; ``model.to(device)`` moves it to CUDA or XPU.
    Inputs must be float32 on the model's device.
    """

    def __init__(self, config: ContextSegmenterConfig | None = None):
        super().__init__()
        self.config = _config(config)
        a, b, c, d = self.config.channels
        self.encoder1 = _block(3, a)
        self.encoder2 = _block(a, b)
        self.encoder3 = _block(b, c)
        self.bottom = _block(c, d)
        self.scene_pool = nn.AdaptiveAvgPool2d(1)
        self.decoder3 = _block(2 * d + c, c)
        self.decoder2 = _block(c + b, b)
        self.decoder1 = _block(b + a, a)
        self.head = nn.Conv2d(a, 1, 1, device="cpu", dtype=torch.float32)

    def _validate_input_metadata(self, images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a tensor")
        device = self.head.weight.device
        if device.type not in {"cpu", "cuda", "xpu"}:
            raise TypeError("model device must be CPU, CUDA, or XPU")
        if images.dtype != torch.float32 or images.device != device:
            raise TypeError("images must be float32 on the model device")
        if any(
            parameter.device != device or parameter.dtype != torch.float32
            for parameter in self.parameters()
        ):
            raise TypeError("all model parameters must be float32 on the same device")
        if (
            images.ndim != 4
            or images.shape[0] < 1
            or tuple(images.shape[1:]) != (3, *self.config.input_size)
        ):
            raise ValueError("images must have shape (N, 3, *config.input_size), N >= 1")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Check shape, device, dtype, finiteness and the [0, 1] range, then run."""
        self._validate_input_metadata(images)
        if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
            raise ValueError("images must be finite and in [0, 1]")
        return self._forward_trusted(images)

    def _forward_trusted(self, images: torch.Tensor) -> torch.Tensor:
        """Training path: checks tensor metadata only and reads no values on the host.

        The caller checks values before a batch comes here. The operations and
        outputs are the same as ``forward``.
        """
        self._validate_input_metadata(images)
        first = self.encoder1(images)
        second = self.encoder2(F.avg_pool2d(first, 2))
        third = self.encoder3(F.avg_pool2d(second, 2))
        bottom = self.bottom(F.avg_pool2d(third, 2))
        up = F.interpolate(bottom, size=third.shape[-2:], mode="bilinear", align_corners=False)
        context = self.scene_pool(bottom).expand(-1, -1, *third.shape[-2:])
        up = self.decoder3(torch.cat((up, third, context), dim=1))
        up = F.interpolate(up, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up = self.decoder2(torch.cat((up, second), dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decoder1(torch.cat((up, first), dim=1)))


# Checkpoints


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.MODEL_ERROR, reason, message)


def sha256(path: str | os.PathLike[str]) -> str:
    """Hex sha256 of a file."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sidecar_path(path: str | os.PathLike[str]) -> Path:
    """``<file>.sha256``: 64 hex digits and a newline."""
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def write_sidecar(path: str | os.PathLike[str]) -> str:
    """Write the sha256 sidecar of ``path`` and return the digest."""
    digest = sha256(Path(path))
    sidecar_path(path).write_text(digest + "\n", encoding="ascii")
    return digest


def _json_like(value) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_json_like(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_like(item) for key, item in value.items())
    return False


def _model_config(raw) -> ContextSegmenterConfig:
    if not isinstance(raw, dict) or set(raw) - {"input_size", "channels", "model_version"}:
        raise ValueError("model_config must hold input_size, channels and model_version")
    return ContextSegmenterConfig(
        input_size=tuple(raw["input_size"]),
        channels=tuple(raw["channels"]),
        model_version=raw.get("model_version", MODEL_VERSION),
    )


def read_context_checkpoint(path: str | os.PathLike[str]) -> dict:
    """Read a context checkpoint after checking its sha256 sidecar.

    A sidecar checks integrity only. ``weights_only=True`` loads tensors and plain data
    and runs no pickled code. Load files from a trusted source.
    """
    path = Path(path)
    try:
        if not path.is_file():
            raise _error("missing_checkpoint", f"{path} does not exist")
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise _error("checkpoint_size_limit", "context checkpoint exceeds 64 MiB")
        sidecar = sidecar_path(path)
        if not sidecar.is_file():
            raise _error("missing_sidecar", f"{sidecar} is missing")
        digest = sidecar.read_text(encoding="ascii").strip()
        actual = sha256(path)
        if len(digest) != 64 or actual != digest:
            raise _error("checkpoint_hash_mismatch", "checkpoint differs from its sha256 sidecar")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") not in CHECKPOINT_SCHEMAS:
            raise _error("invalid_checkpoint", f"schema must be one of {CHECKPOINT_SCHEMAS}")
        config = _model_config(checkpoint.get("model_config"))
        state = checkpoint.get("model_state")
        if not isinstance(state, dict) or not state:
            raise _error("invalid_checkpoint", "model_state must be a nonempty dict")
        for value in state.values():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())
            ):
                raise _error("invalid_checkpoint", "model_state must hold finite float32 tensors")
        if checkpoint["schema"] == PILOT_SCHEMA:
            epoch, best = checkpoint.get("epoch"), checkpoint.get("best_epoch")
            if type(epoch) is not int or type(best) is not int or not 0 <= best <= epoch:
                raise _error("invalid_checkpoint", "pilot checkpoint epochs are invalid")
        else:
            if checkpoint.get("preprocess_version") != PREPROCESS_VERSION:
                raise _error("invalid_checkpoint", "preprocess_version differs")
            if not isinstance(checkpoint.get("provenance"), dict) or not _json_like(
                checkpoint["provenance"]
            ):
                raise _error("invalid_checkpoint", "provenance must be a JSON-like dict")
        checkpoint["_config"] = config
        checkpoint["_sha256"] = actual
        return checkpoint
    except FlattenerError:
        raise
    except (OSError, ValueError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError) as exc:
        raise _error("invalid_checkpoint", f"Cannot read context checkpoint: {exc}") from exc


def save_context_checkpoint(
    path: str | os.PathLike[str],
    model_state: dict[str, torch.Tensor],
    config: ContextSegmenterConfig,
    provenance: dict[str, Any],
) -> str:
    """Write a ``region-context-1`` checkpoint and its sidecar; return its sha256.

    ``provenance`` is a JSON-like record: where the weights came from, how they were
    trained (``lineage``, ``initialization``, ``track``) and on which data.
    """
    config = _config(config)
    if not isinstance(provenance, dict) or not _json_like(provenance):
        raise ValueError("provenance must be a JSON-like dict")
    random_state = torch.get_rng_state()
    try:
        # Check the state against the architecture before writing anything.
        ContextPageSegmenter(config).load_state_dict(model_state, strict=True)
    finally:
        torch.set_rng_state(random_state)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": {
            "input_size": list(config.input_size),
            "channels": list(config.channels),
            "model_version": config.model_version,
        },
        "model_state": {
            key: value.detach().to("cpu", torch.float32).contiguous().clone()
            for key, value in model_state.items()
        },
        "preprocess_version": PREPROCESS_VERSION,
        "provenance": json.loads(json.dumps(provenance, allow_nan=False)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return write_sidecar(path)


def _pilot_provenance(checkpoint: dict) -> dict:
    protocol = checkpoint.get("protocol")
    protocol = protocol if isinstance(protocol, dict) else {}
    keys = (
        "seed",
        "epochs",
        "batch_size",
        "torch_version",
        "train_counts",
        "validation_counts",
        "selection",
        "limitations",
        "no_page_negatives_evaluated",
        "test_evaluated",
    )
    summary = {key: protocol[key] for key in keys if key in protocol and _json_like(protocol[key])}
    device = protocol.get("device")
    if isinstance(device, dict) and _json_like(device):
        summary["device"] = device
    # The pilot file itself records no lineage; region-context-1 exports carry one.
    summary.update(initialization="unknown", source_schema=PILOT_SCHEMA)
    return summary


@dataclass(slots=True)
class LoadedContextSegmenter:
    """A loaded page gate. Duck-types ``regions.segmentation.LoadedSegmenter``."""

    model: ContextPageSegmenter
    config: ContextSegmenterConfig
    metadata: dict
    load_seconds: float
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def model_sha256(self) -> str:
        return self.metadata["checkpoint_sha256"]

    @property
    def model_id(self) -> str:
        return f"{MODEL_VERSION}:{self.model_sha256[:12]}"

    def _logits(self, image: np.ndarray) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            tensor = preprocess_image(image, self.config)[None].to(self.device)
            logits = self.model(tensor)[0, 0].to("cpu")
        if not bool(torch.isfinite(logits).all()):
            raise _error("nonfinite_prediction", "page gate produced nonfinite logits")
        return logits

    def predict_logits(self, image: np.ndarray) -> np.ndarray:
        """float32 H×W page logits at ``config.input_size`` over the whole image."""
        return self._logits(image).numpy().copy()

    def predict_probability(self, image: np.ndarray) -> np.ndarray:
        """float32 H×W page probabilities at ``config.input_size`` over the whole image.

        No threshold, component choice or crop is applied here; ``gate_page_region``
        does that.
        """
        return self._logits(image).sigmoid().numpy().copy()


def _device(device) -> torch.device:
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise FlattenerError(Status.INPUT_ERROR, "invalid_device", str(exc)) from exc
    if device.type not in ("cpu", "cuda", "xpu"):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_device", "device must be cpu, cuda or xpu"
        )
    backend = getattr(torch, device.type, None)
    if device.type != "cpu" and (backend is None or not backend.is_available()):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_device", f"{device} is unavailable")
    return device


def load_context_segmenter(
    path: str | os.PathLike[str], *, device: str | torch.device = "cpu"
) -> LoadedContextSegmenter:
    """Load a context checkpoint (either schema) with its sidecar onto ``device``."""
    started = time.perf_counter()
    device = _device(device)
    checkpoint = read_context_checkpoint(path)
    config = checkpoint["_config"]
    try:
        # Module construction draws from the CPU generator even though every weight is
        # replaced; keep the caller's generator state.
        random_state = torch.get_rng_state()
        try:
            model = ContextPageSegmenter(config)
            model.load_state_dict(checkpoint["model_state"], strict=True)
        finally:
            torch.set_rng_state(random_state)
        model.eval().requires_grad_(False)
        model.to(device)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise _error("invalid_checkpoint", f"Cannot build the context segmenter: {exc}") from exc
    schema = checkpoint["schema"]
    provenance = (
        checkpoint["provenance"] if schema == CHECKPOINT_SCHEMA else _pilot_provenance(checkpoint)
    )
    metadata = {
        "checkpoint_sha256": checkpoint["_sha256"],
        "checkpoint_schema": schema,
        "model_version": MODEL_VERSION,
        "model_config": asdict(config),
        "parameter_count": parameter_count(config),
        "preprocess_version": PREPROCESS_VERSION,
        "provenance": provenance,
        "device": str(device),
        "probability_grid": "whole original image extent at model_config.input_size",
    }
    if schema == PILOT_SCHEMA:
        metadata.update(epoch=checkpoint["epoch"], best_epoch=checkpoint["best_epoch"])
    return LoadedContextSegmenter(model, config, metadata, time.perf_counter() - started, device)


# Page gate decision


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Region rules of the page gate.

    ``region`` goes to ``detect_region``; its area cap is 1.0 so that close-ups pass.
    ``fills_frame_area`` and ``fills_frame_borders`` decide ``page_fills_frame``.
    ``confident_*`` mark a detection whose box needs no cross-check by a dewarp pass.
    """

    region: RegionConfig = field(default_factory=lambda: RegionConfig(max_area_fraction=1.0))
    fills_frame_area: float = 0.90
    fills_frame_borders: int = 3
    confident_dominance: float = 0.95
    confident_mean_probability: float = 0.85

    def __post_init__(self):
        if not isinstance(self.region, RegionConfig):
            raise TypeError("region must be a RegionConfig")
        for name in ("fills_frame_area", "confident_dominance", "confident_mean_probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number in [0, 1]")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number in [0, 1]")
        borders = self.fills_frame_borders
        if type(borders) is not int or not 1 <= borders <= 4:
            raise ValueError("fills_frame_borders must be an integer in [1, 4]")


@dataclass(slots=True)
class PageGate:
    """The gate's page box ``region`` in source pixels, and why.

    ``estimate`` is the ``detect_region`` result (mask on the probability grid).
    With ``fills_frame`` the region is the whole image and ``reasons`` holds
    ``page_fills_frame``.
    """

    region: PageRegion
    estimate: RegionEstimate
    fills_frame: bool
    touched_borders: tuple[str, ...]
    area_fraction: float
    confident: bool
    reasons: tuple[str, ...]
    metadata: dict


def touched_borders(mask: np.ndarray) -> tuple[str, ...]:
    """Image borders (top, right, bottom, left) that a boolean grid mask reaches."""
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 2 or not mask.size:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_region_input", "mask must be a nonempty bool H×W array"
        )
    edges = (mask[0], mask[:, -1], mask[-1], mask[:, 0])
    return tuple(side for side, edge in zip(BORDERS, edges, strict=True) if edge.any())


def _gate_config(config):
    if config is None:
        return GateConfig()
    if not isinstance(config, GateConfig):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_region_input", "config must be GateConfig"
        )
    return config


def _decide(estimate: RegionEstimate, source_size, config: GateConfig) -> PageGate:
    height, width = (int(value) for value in source_size)
    sides = touched_borders(estimate.mask)
    area = float(estimate.metadata["area_fraction"])
    fills = area > config.fills_frame_area or len(sides) >= config.fills_frame_borders
    region = PageRegion(0, 0, width, height) if fills else estimate.region
    confident = (
        float(estimate.metadata["dominance"]) >= config.confident_dominance
        and float(estimate.metadata["mean_probability"]) >= config.confident_mean_probability
    )
    reasons = ("page_fills_frame",) if fills else ()
    metadata = {
        "gate_version": GATE_VERSION,
        "gate_config": {
            "fills_frame_area": config.fills_frame_area,
            "fills_frame_borders": config.fills_frame_borders,
            "confident_dominance": config.confident_dominance,
            "confident_mean_probability": config.confident_mean_probability,
        },
        "region": asdict(region),
        "detected_region": asdict(estimate.region),
        "fills_frame": fills,
        "touched_borders": list(sides),
        "area_fraction": area,
        "confident": confident,
        "reasons": list(reasons),
        "detection": estimate.metadata,
    }
    return PageGate(region, estimate, fills, sides, area, confident, reasons, metadata)


def gate_page_region(probability, source_size, *, config: GateConfig | None = None) -> PageGate:
    """Page box from a gate probability grid covering the whole source image.

    Raises ``FlattenerError`` (``model_error``) with the ``detect_region`` reasons:
    ``page_not_found``, ``page_area_rejected`` (under the minimum area),
    ``ambiguous_page_region`` and ``uncertain_page_region``.
    """
    config = _gate_config(config)
    estimate = detect_region(probability, source_size, config=config.region)
    return _decide(estimate, source_size, config)


def gate_page(image: np.ndarray, segmenter, *, config: GateConfig | None = None) -> PageGate:
    """Run ``segmenter`` on ``image`` and decide its page box (see ``gate_page_region``).

    Error metadata and the result's detection metadata name the segmenter and its sha256.
    """
    from .inference import estimate_page_region  # inference may import this module later

    config = _gate_config(config)
    estimate = estimate_page_region(image, segmenter, config=config.region)
    return _decide(estimate, image.shape[:2], config)


# SmartDoc exposure of the v0 checkpoint


@cache
def _smartdoc_exposure_text() -> str:
    return resources.files(__package__).joinpath(SMARTDOC_EXPOSURE_FILE).read_text("utf-8")


def smartdoc_exposure() -> dict:
    """The v0 gate's SmartDoc template exposure record (a fresh copy per call).

    ``exclude_from_evaluation`` lists the SmartDoc model names (for example
    ``datasheet001``) whose frames trained v0. ``selection_exposed`` chose its epoch, and
    ``unexposed`` never reached training or selection.
    """
    return json.loads(_smartdoc_exposure_text())


__all__ = [
    "BORDERS",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMAS",
    "ContextPageSegmenter",
    "ContextSegmenterConfig",
    "GATE_VERSION",
    "GateConfig",
    "LoadedContextSegmenter",
    "MAX_PARAMETERS",
    "MODEL_VERSION",
    "PILOT_SCHEMA",
    "PREPROCESS_VERSION",
    "PageGate",
    "gate_page",
    "gate_page_region",
    "load_context_segmenter",
    "parameter_count",
    "preprocess_image",
    "preprocess_mask",
    "read_context_checkpoint",
    "save_context_checkpoint",
    "sidecar_path",
    "smartdoc_exposure",
    "touched_borders",
    "write_sidecar",
]
