"""Prepare whole-image RGB views and binary masks for the page detector.

All sizes are (height, width).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .. import ops

if TYPE_CHECKING:
    import torch

PREPROCESS_VERSION = "rgb-full-extent-bilinear-antialias-1.0"


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    input_size: tuple[int, int] = (192, 144)

    def __post_init__(self):
        if (
            not isinstance(self.input_size, tuple)
            or len(self.input_size) != 2
            or any(type(value) is not int or not 8 <= value <= 512 for value in self.input_size)
        ):
            raise ValueError("input_size must be a tuple of two integers in [8, 512]")


def _config(config: SegmenterConfig | None) -> SegmenterConfig:
    if config is None:
        return SegmenterConfig()
    if not isinstance(config, SegmenterConfig):
        raise TypeError("config must be a SegmenterConfig or None")
    return config


def preprocess_array(image: np.ndarray, config: SegmenterConfig | None = None) -> NDArray:
    """Return an independent float32 CHW preview, without crop or padding: bilinear
    with antialiasing over the full image extent, in [0, 1]."""
    config = _config(config)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy RGB array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have a nonempty (height, width, 3) shape")
    values = ops.resize(image, config.input_size, antialias=True)
    np.clip(values, 0, 1, out=values)
    return np.ascontiguousarray(values.transpose(2, 0, 1))


def preprocess_image(image: np.ndarray, config: SegmenterConfig | None = None) -> torch.Tensor:
    """``preprocess_array`` as a CPU tensor."""
    import torch

    return torch.from_numpy(preprocess_array(image, config))


def preprocess_mask(mask: np.ndarray, config: SegmenterConfig | None = None) -> torch.Tensor:
    """Return a binary 1HW target using nearest-exact center sampling."""
    import torch
    from torch.nn import functional as F

    config = _config(config)
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("mask must be a boolean NumPy array")
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("mask must have a nonempty (height, width) shape")
    tensor = torch.from_numpy(mask.copy(order="C")).float()[None, None]
    return F.interpolate(tensor, size=config.input_size, mode="nearest-exact")[0]
