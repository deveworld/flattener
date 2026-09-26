"""Experimental wider-context page segmenter, separate from the frozen v1 model.

Only the architecture changes: three pooling levels and an image-wide average
bottleneck context feed one mask-logit decoder. This module adds no presence
label, auxiliary loss, threshold, component selection, or quality guarantee.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .segmentation import PREPROCESS_VERSION, SegmenterConfig
from .segmentation import preprocess_image as _preprocess_image
from .segmentation import preprocess_mask as _preprocess_mask

MODEL_VERSION = "region-context-unet-2.0"
MAX_PARAMETERS = 1_000_000


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
        + 1  # one 1x1 mask head, including bias
    )


@dataclass(frozen=True, slots=True)
class ContextSegmenterConfig:
    """Full-extent input and four channels, serialized with an explicit version."""

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
    """Return the exact trainable parameter count without allocating a model."""
    return _parameter_count(_config(config).channels)


def preprocess_image(
    image: np.ndarray, config: ContextSegmenterConfig | None = None
) -> torch.Tensor:
    """Delegate unchanged v1 RGB preprocessing: full extent, float32, antialias."""
    config = _config(config)
    return _preprocess_image(image, SegmenterConfig(input_size=config.input_size))


def preprocess_mask(mask: np.ndarray, config: ContextSegmenterConfig | None = None) -> torch.Tensor:
    """Delegate unchanged v1 binary nearest-exact target preprocessing."""
    config = _config(config)
    return _preprocess_mask(mask, SegmenterConfig(input_size=config.input_size))


def _block(in_channels, out_channels):
    # Explicit CPU/float32 keeps this model's contract independent of the
    # caller's default dtype/device without changing those process globals.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, device="cpu", dtype=torch.float32),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, device="cpu", dtype=torch.float32),
        nn.ReLU(),
    )


class ContextPageSegmenter(nn.Module):
    """Three-level U-Net whose first decoder also sees image-wide context.

    Bottleneck average pooling yields N,D,1,1. Broadcasting that vector across
    the first decoder grid makes every decoded position depend on the entire
    bottleneck, instead of only its local receptive field. Skip connections
    retain local boundary detail. There is still exactly one N,1,H,W logit map.

    Construction remains CPU/float32. Explicit ``model.to(device)`` supports
    CPU, CUDA, and XPU; inputs must be float32 on that same device.
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
        """Validate public inputs, including finite values and the RGB range."""
        self._validate_input_metadata(images)
        if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
            raise ValueError("images must be finite and in [0, 1]")
        return self._forward_trusted(images)

    def _forward_trusted(self, images: torch.Tensor) -> torch.Tensor:
        """Training path for previously validated finite, [0, 1] RGB tensors.

        Only tensor metadata is inspected here; no device values are read by
        the host. The caller owns value validation before admitting a batch.
        Architecture, parameters, and outputs match the public forward path.
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


__all__ = [
    "ContextPageSegmenter",
    "ContextSegmenterConfig",
    "MAX_PARAMETERS",
    "MODEL_VERSION",
    "PREPROCESS_VERSION",
    "parameter_count",
    "preprocess_image",
    "preprocess_mask",
]
