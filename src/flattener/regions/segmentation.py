"""Compact CPU page segmentation; no geometric rectification is inferred here.

All sizes are (height, width). The probability grid covers the entire input
image extent and is not a crop, thresholded region, or confidence guarantee.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import FlattenerError, Status

CHECKPOINT_SCHEMA = "region-segmentation-0.1"
MODEL_VERSION = "region-unet-1.0"
PREPROCESS_VERSION = "rgb-full-extent-bilinear-antialias-1.0"


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    input_size: tuple[int, int] = (128, 96)
    channels: tuple[int, int, int] = (8, 16, 32)

    def __post_init__(self):
        for name, count, low, high in (("input_size", 2, 8, 512), ("channels", 3, 1, 128)):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != count
                or any(type(item) is not int or not low <= item <= high for item in value)
            ):
                raise ValueError(f"{name} must be a tuple of {count} integers in [{low}, {high}]")


def _config(config: SegmenterConfig | None) -> SegmenterConfig:
    if config is None:
        return SegmenterConfig()
    if not isinstance(config, SegmenterConfig):
        raise TypeError("config must be a SegmenterConfig or None")
    return config


def preprocess_image(image: np.ndarray, config: SegmenterConfig | None = None) -> torch.Tensor:
    """Return an independent float32 CPU CHW preview, without crop or padding."""
    config = _config(config)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy RGB array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have a nonempty (height, width, 3) shape")
    tensor = torch.from_numpy(image.copy(order="C")).permute(2, 0, 1).float().div_(255)
    return (
        F.interpolate(
            tensor[None],
            size=config.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        .clamp_(0, 1)
        .contiguous()
    )


def preprocess_mask(mask: np.ndarray, config: SegmenterConfig | None = None) -> torch.Tensor:
    """Return a binary 1HW target using nearest-exact center sampling."""
    config = _config(config)
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("mask must be a boolean NumPy array")
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("mask must have a nonempty (height, width) shape")
    tensor = torch.from_numpy(mask.copy(order="C")).float()[None, None]
    return F.interpolate(tensor, size=config.input_size, mode="nearest-exact")[0]


def _block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.ReLU(),
    )


class PageSegmenter(nn.Module):
    """Two-level U-Net with skip connections, no batch state or dropout."""

    def __init__(self, config: SegmenterConfig | None = None):
        super().__init__()
        self.config = _config(config)
        a, b, c = self.config.channels
        self.encoder1 = _block(3, a)
        self.encoder2 = _block(a, b)
        self.bottom = _block(b, c)
        self.decoder2 = _block(c + b, b)
        self.decoder1 = _block(b + a, a)
        self.head = nn.Conv2d(a, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a tensor")
        if images.device.type != "cpu" or images.dtype != torch.float32:
            raise TypeError("images must be CPU float32")
        if (
            images.ndim != 4
            or images.shape[0] < 1
            or tuple(images.shape[1:]) != (3, *self.config.input_size)
        ):
            raise ValueError("images must have shape (N, 3, *config.input_size), N >= 1")
        if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
            raise ValueError("images must be finite and in [0, 1]")
        first = self.encoder1(images)
        second = self.encoder2(F.avg_pool2d(first, 2))
        bottom = self.bottom(F.avg_pool2d(second, 2))
        up = F.interpolate(bottom, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up = self.decoder2(torch.cat((up, second), dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decoder1(torch.cat((up, first), dim=1)))


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.MODEL_ERROR, reason, message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path: str | os.PathLike[str]) -> dict:
    """Read tensor-only CPU state after checking a required integrity sidecar.

    A sidecar is not an authenticity signature. Only use checkpoints from a
    trusted source; weights_only avoids arbitrary Python object unpickling.
    """
    path = Path(path)
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise _error("checkpoint_size_limit", "region checkpoint exceeds 64 MiB")
        digest = path.with_suffix(path.suffix + ".sha256").read_text().strip()
        if len(digest) != 64 or sha256(path) != digest:
            raise _error("checkpoint_hash_mismatch", "checkpoint differs from SHA-256 sidecar")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "schema_version",
            "config",
            "model_state",
            "optimizer_state",
            "epoch",
            "history",
            "initial_validation",
            "best_validation",
            "best_epoch",
            "manifest_sha256",
            "provenance",
            "torch_rng_state",
            "shuffle_rng_state",
            "dataset_counts",
        }
        if (
            not isinstance(checkpoint, dict)
            or not required.issubset(checkpoint)
            or checkpoint["schema_version"] != CHECKPOINT_SCHEMA
            or not isinstance(checkpoint["config"], dict)
            or not isinstance(checkpoint["config"].get("model_config"), dict)
            or not isinstance(checkpoint["model_state"], dict)
            or type(checkpoint["epoch"]) is not int
            or checkpoint["epoch"] < 0
            or not isinstance(checkpoint["history"], list)
            or len(checkpoint["history"]) != checkpoint["epoch"]
            or type(checkpoint["best_epoch"]) is not int
            or not 0 <= checkpoint["best_epoch"] <= checkpoint["epoch"]
            or not isinstance(checkpoint["best_validation"], dict)
            or not math.isfinite(checkpoint["best_validation"].get("loss", math.nan))
        ):
            raise _error("invalid_checkpoint", "invalid region checkpoint schema or metadata")
        for value in checkpoint["model_state"].values():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise _error("invalid_checkpoint", "nonfinite or non-tensor model state")
        return checkpoint
    except (OSError, ValueError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError) as exc:
        raise _error("invalid_checkpoint", f"Cannot read region checkpoint: {exc}") from exc


@dataclass(slots=True)
class LoadedSegmenter:
    model: PageSegmenter
    config: SegmenterConfig
    metadata: dict
    load_seconds: float

    @property
    def model_sha256(self) -> str:
        return self.metadata["checkpoint_sha256"]

    @property
    def model_id(self) -> str:
        return f"{MODEL_VERSION}:{self.model_sha256[:12]}"

    def predict_probability(self, image: np.ndarray) -> np.ndarray:
        """Return independent float32 HW probabilities at config.input_size.

        The grid covers all of the original image. This intentionally does not
        choose a threshold, connected component, bbox, or output aspect ratio.
        """
        self.model.eval()
        with torch.inference_mode():
            probability = self.model(preprocess_image(image, self.config)[None]).sigmoid()[0, 0]
        if not bool(torch.isfinite(probability).all()):
            raise _error("nonfinite_prediction", "segmenter produced nonfinite probabilities")
        return probability.numpy().copy()


def load_segmenter(path: str | os.PathLike[str]) -> LoadedSegmenter:
    started = time.perf_counter()
    checkpoint = read_checkpoint(path)
    try:
        raw = checkpoint["config"]["model_config"]
        config = SegmenterConfig(
            input_size=tuple(raw["input_size"]), channels=tuple(raw["channels"])
        )
        # Module initialization consumes RNG even though all weights are replaced.
        # Preserve the caller's CPU generator, including failed state loading.
        random_state = torch.get_rng_state()
        try:
            model = PageSegmenter(config)
            model.load_state_dict(checkpoint["model_state"], strict=True)
        finally:
            torch.set_rng_state(random_state)
        model.eval()
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise _error("invalid_checkpoint", f"Cannot construct region segmenter: {exc}") from exc
    return LoadedSegmenter(
        model,
        config,
        {
            "checkpoint_sha256": sha256(Path(path)),
            "model_version": MODEL_VERSION,
            "preprocess_version": PREPROCESS_VERSION,
            "epoch": checkpoint["epoch"],
            "manifest_sha256": checkpoint["manifest_sha256"],
            "best_validation": checkpoint["best_validation"],
            "probability_grid": "full original image extent; no threshold or crop applied",
        },
        time.perf_counter() - started,
    )
