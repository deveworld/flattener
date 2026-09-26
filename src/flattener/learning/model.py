"""Small CPU image-conditioned predictor for normalized backward-map residuals.

All sizes are (height, width). A fixed full-extent resize prepares the input;
there is no crop, padding, aspect-ratio inference, or coordinate clipping.
The model's zero-initialized final layer starts at the identity map. Its
unbounded output permits the caller to inspect invalid transforms explicitly.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

MODEL_VERSION = "1.0"
PREPROCESS_VERSION = "rgb-float32-bilinear-antialias-full-extent-1.1"


def _size(value: tuple[int, int], name: str, minimum: int = 1) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in value)
    ):
        raise ValueError(f"{name} must be a (height, width) integer tuple with sides >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Provisional CPU architecture; dimensions refer to the whole image extent."""

    input_size: tuple[int, int] = (96, 72)
    grid_size: tuple[int, int] = (17, 13)
    channels: tuple[int, int, int] = (16, 32, 48)
    hidden: int = 128

    def __post_init__(self) -> None:
        _size(self.input_size, "input_size")
        _size(self.grid_size, "grid_size", minimum=2)
        if (
            not isinstance(self.channels, tuple)
            or len(self.channels) != 3
            or any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in self.channels)
        ):
            raise ValueError("channels must be a tuple of three positive integers")
        if isinstance(self.hidden, bool) or not isinstance(self.hidden, int) or self.hidden <= 0:
            raise ValueError("hidden must be a positive integer")


def _config(config: ModelConfig | None) -> ModelConfig:
    if config is None:
        return ModelConfig()
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig or None")
    return config


def _float_cpu_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be a CPU float32 tensor")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def coarse_identity(size: tuple[int, int]) -> torch.Tensor:
    """Return a fresh CPU float32 (H, W, 2) half-pixel reference in (x, y) order.

    The same formula applies to coarse and dense grids, including singleton
    dimensions. Normalized coordinates describe the source's physical image
    extent, regardless of the source resolution.
    """
    height, width = _size(size, "size")
    # Evaluate centers in float64 before the final cast to agree with the
    # renderer at float32 precision, including non-power-of-two dimensions.
    x = (2.0 * (torch.arange(width, dtype=torch.float64) + 0.5) / width - 1).float()
    y = (2.0 * (torch.arange(height, dtype=torch.float64) + 0.5) / height - 1).float()
    return torch.stack((x.expand(height, width), y[:, None].expand(height, width)), dim=-1)


def dense_maps(residual: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Differentiably expand N,2,Gh,Gw residuals into N,H,W,2 absolute maps.

    Both coarse axes must be at least two and output axes may not shrink.
    Residuals use normalized source units, not pixels of the resized network
    input. The resize extends residual boundary values and preserves identity
    and constant shifts. Returned maps are not clipped or detached.
    """
    _float_cpu_tensor(residual, "residual")
    if residual.ndim != 4 or residual.shape[0] < 1 or residual.shape[1] != 2:
        raise ValueError("residual must have shape (N, 2, Gh, Gw), with N >= 1")
    coarse_size = tuple(residual.shape[-2:])
    _size(coarse_size, "residual grid", minimum=2)
    output_size = _size(output_size, "output_size")
    if any(out < coarse for out, coarse in zip(output_size, coarse_size, strict=True)):
        raise ValueError("output_size may not shrink either residual grid axis")
    expanded = F.interpolate(
        residual, size=output_size, mode="bilinear", align_corners=False, antialias=False
    )
    return expanded.permute(0, 2, 3, 1) + coarse_identity(output_size).unsqueeze(0)


def preprocess_image(
    image: NDArray[np.uint8], config: ModelConfig | None = None
) -> torch.Tensor:
    """Copy RGB uint8 HWC pixels into a resized CPU float32 CHW tensor in [0,1].

    Resize both full image axes directly to ``config.input_size`` with bilinear
    antialiasing and align_corners=False. This changes the preview aspect ratio
    if necessary; original source dimensions remain the caller's responsibility
    when converting predicted normalized coordinates to source pixels. Caller
    memory is never modified or shared with the result. Positive bilinear
    weights preserve [0,1] analytically; clamp only the resized RGB values to
    remove floating-point summation overshoot (not predicted coordinates).
    """
    config = _config(config)
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have nonempty shape (height, width, 3)")
    tensor = torch.from_numpy(image.copy(order="C")).permute(2, 0, 1).float().div_(255.0)
    return F.interpolate(
        tensor.unsqueeze(0),
        size=config.input_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0).clamp_(0.0, 1.0).contiguous()


class CoordinatePredictor(nn.Module):
    """Predict unbounded normalized residuals from actual RGB image tensors."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = _config(config)
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in self.config.channels:
            layers.extend(
                [nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1), nn.ReLU()]
            )
            input_channels = output_channels
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((6, 5)))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.channels[-1] * 6 * 5, self.config.hidden),
            nn.ReLU(),
            nn.Linear(self.config.hidden, 2 * self.config.grid_size[0] * self.config.grid_size[1]),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return N,2,Gh,Gw for CPU float32 images N,3,Hin,Win in [0,1]."""
        _float_cpu_tensor(images, "images")
        expected = (3, *self.config.input_size)
        if images.ndim != 4 or images.shape[0] < 1 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"images must have shape (N, {expected}), with N >= 1")
        if (images < 0).any().item() or (images > 1).any().item():
            raise ValueError("images must have values in [0, 1]")
        return self.head(self.encoder(images)).reshape(images.shape[0], 2, *self.config.grid_size)
