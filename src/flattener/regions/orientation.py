"""Independent four-way direction classification of a physical square page crop.

Class ``q`` is the current clockwise quarter turn from upright. Correct a page
with ``rot90(q)`` counterclockwise; counterclockwise augmentation by ``k`` gives
the target ``(q - k) % 4``. Localization and ground-truth labels are external.
"""

from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .context_segmentation import _block

MODEL_VERSION = "page-crop-orientation-1.0"
PREPROCESS_VERSION = "physical-square-antialias-gray-1.0"
MAX_PARAMETERS = 2_000_000
MAX_IMAGE_SIDE = 16_384
MAX_IMAGE_PIXELS = 64_000_000


def _parameter_count(channels, hidden):
    def block(in_channels, out_channels):
        return out_channels * (9 * in_channels + 1) + out_channels * (9 * out_channels + 1)

    a, b, c, d = channels
    return (
        block(3, a)
        + block(a, b)
        + block(b, c)
        + block(c, d)
        + (16 * d + 1) * hidden
        + (hidden + 1) * 4
    )


@dataclass(frozen=True, slots=True)
class OrientationConfig:
    """Bounded square RGB input and four encoder widths."""

    input_size: int = 160
    channels: tuple[int, int, int, int] = (8, 16, 32, 64)
    hidden: int = 128

    def __post_init__(self):
        if type(self.input_size) is not int or not 32 <= self.input_size <= 512:
            raise ValueError("input_size must be an integer in [32, 512]")
        if (
            not isinstance(self.channels, tuple)
            or len(self.channels) != 4
            or any(type(value) is not int or not 1 <= value <= 128 for value in self.channels)
        ):
            raise ValueError("channels must be a tuple of four integers in [1, 128]")
        if type(self.hidden) is not int or not 4 <= self.hidden <= 1024:
            raise ValueError("hidden must be an integer in [4, 1024]")
        if _parameter_count(self.channels, self.hidden) > MAX_PARAMETERS:
            raise ValueError(f"orientation model exceeds the {MAX_PARAMETERS:,}-parameter limit")


def _config(config):
    if config is None:
        return OrientationConfig()
    if not isinstance(config, OrientationConfig):
        raise TypeError("config must be OrientationConfig or None")
    return config


def parameter_count(config: OrientationConfig | None = None) -> int:
    """Return the exact trainable parameter count without allocating a model."""
    config = _config(config)
    return _parameter_count(config.channels, config.hidden)


def _finite_real(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


class PageOrientationClassifier(nn.Module):
    """Context-segmenter-compatible encoder with a spatial direction head.

    The encoder1/encoder2/encoder3/bottom state keys and shapes match a context
    page segmenter with equal channels. Copying those weights is an explicit
    caller action; construction initializes a fresh CPU float32 model. A 4x4
    pooled feature grid retains spatial information for four-way classification.
    """

    def __init__(self, config: OrientationConfig | None = None):
        super().__init__()
        self.config = _config(config)
        a, b, c, d = self.config.channels
        self.encoder1 = _block(3, a)
        self.encoder2 = _block(a, b)
        self.encoder3 = _block(b, c)
        self.bottom = _block(c, d)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(16 * d, self.config.hidden, device="cpu", dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(self.config.hidden, 4, device="cpu", dtype=torch.float32),
        )

    def _validate_input_metadata(self, images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a tensor")
        device = self.classifier[-1].weight.device
        if device.type not in {"cpu", "cuda", "xpu"}:
            raise TypeError("model device must be CPU, CUDA, or XPU")
        if images.dtype != torch.float32 or images.device != device:
            raise TypeError("images must be float32 on the model device")
        if any(
            parameter.device != device or parameter.dtype != torch.float32
            for parameter in self.parameters()
        ):
            raise TypeError("all model parameters must be float32 on the same device")
        size = self.config.input_size
        if images.ndim != 4 or images.shape[0] < 1 or tuple(images.shape[1:]) != (3, size, size):
            raise ValueError("images must have shape (N, 3, input_size, input_size), N >= 1")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Validate public float32 RGB inputs and return N,4 direction logits."""
        self._validate_input_metadata(images)
        if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
            raise ValueError("images must be finite and in [0, 1]")
        return self._forward_trusted(images)

    def _forward_trusted(self, images: torch.Tensor) -> torch.Tensor:
        """Inspect metadata only; callers must already have admitted RGB values.

        No tensor values are copied to the host, so the training path introduces
        no per-batch device synchronization for range or finite checks.
        """
        self._validate_input_metadata(images)
        first = self.encoder1(images)
        second = self.encoder2(F.avg_pool2d(first, 2))
        third = self.encoder3(F.avg_pool2d(second, 2))
        bottom = self.bottom(F.avg_pool2d(third, 2))
        return self.classifier(self.pool(bottom).flatten(1))

    def rotation_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """Average four canonicalized softmax distributions as float32 N,4.

        Each exact counterclockwise quarter turn has equal weight. Up to
        floating-point rounding, ``P(rot90(x, k)) = roll(P(x), -k)``. Tied argmax
        decisions need not satisfy that relation. RGB values are checked once.
        """
        self._validate_input_metadata(images)
        if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
            raise ValueError("images must be finite and in [0, 1]")
        return self._rotation_probabilities_trusted(images)

    def _rotation_probabilities_trusted(self, images: torch.Tensor) -> torch.Tensor:
        """Use admitted RGB values with metadata-only checks and no host reads."""
        self._validate_input_metadata(images)
        canonical = []
        for quarter in range(4):
            logits = self._forward_trusted(torch.rot90(images, quarter, dims=(-2, -1)))
            probabilities = F.softmax(logits.float(), dim=-1)
            canonical.append(torch.roll(probabilities, shifts=quarter, dims=-1))
        return torch.stack(canonical).mean(dim=0)


def square_page_view(image: np.ndarray, box, size: int = 160) -> torch.Tensor:
    """Return an independent CPU float32 CHW physical square crop in [0, 1].

    ``box=(left, top, right, bottom)`` contains finite, already-padded cell-edge
    coordinates within the image. Its center is ``(cx, cy)`` and the virtual
    square side is ``L=max(right-left, bottom-top)``. The square retains its full
    extent when it exceeds the image boundary; fully outside samples are 0.5.

    For output pixel-center coordinate ``u=2*(column+.5)/size-1``, the original
    source pixel index is ``cx+L*u/2-.5``. Thus align_corners=False normalization
    is ``2*cx/W-1+L*u/W``; y follows the same physical scale. The original image
    is first antialiased to a sufficient working resolution before grid sampling.
    At source boundaries bilinear support blends RGB with constant 0.5 padding.
    No labels, direction predictions, thresholding, or extra margins enter here.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy RGB array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have a nonempty (height, width, 3) shape")
    height, width = image.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE or height * width > MAX_IMAGE_PIXELS:
        raise ValueError("image exceeds the square crop size limits")
    if type(size) is not int or not 2 <= size <= 1024:
        raise ValueError("size must be an integer in [2, 1024]")
    if not isinstance(box, (tuple, list, np.ndarray)):
        raise TypeError("box must be a tuple, list, or NumPy array")
    if isinstance(box, np.ndarray) and box.shape != (4,):
        raise ValueError("box must contain four finite cell-edge coordinates")
    if len(box) != 4 or any(not _finite_real(value) for value in box):
        raise ValueError("box must contain four finite cell-edge coordinates")
    left, top, right, bottom = map(float, box)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("box must have positive extent within the image")
    side = max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    # Avoid overflow for a valid but arbitrarily tiny subpixel box. Such a crop
    # magnifies the source and needs no preliminary downsampling.
    work_size = (height, width)
    if side > size:
        work_size = (max(1, ceil(height * size / side)), max(1, ceil(width * size / side)))
    axis = (np.arange(size, dtype=np.float64) + 0.5) * (2.0 / size) - 1.0
    grid = np.empty((size, size, 2), dtype=np.float32)
    grid[..., 0] = 2 * cx / width - 1 + side * axis[None, :] / width
    grid[..., 1] = 2 * cy / height - 1 + side * axis[:, None] / height
    # no_grad (not inference_mode): callers can use the returned tensor as an
    # ordinary training input whose activations must be saved for backward.
    with torch.no_grad():
        source = torch.from_numpy(image.copy(order="C")).permute(2, 0, 1)[None]
        source = source.to(dtype=torch.float32).div_(255)
        if work_size != (height, width):
            source = F.interpolate(
                source, size=work_size, mode="bilinear", align_corners=False, antialias=True
            )
        sampled = F.grid_sample(
            source - 0.5,
            torch.from_numpy(grid)[None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled[0].add_(0.5).clamp_(0, 1).contiguous()


__all__ = [
    "MAX_PARAMETERS",
    "MODEL_VERSION",
    "OrientationConfig",
    "PREPROCESS_VERSION",
    "PageOrientationClassifier",
    "parameter_count",
    "square_page_view",
]
