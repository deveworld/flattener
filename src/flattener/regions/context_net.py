"""The page detector network (``region-context-unet-2.0``) in PyTorch.

``regions.context`` loads it; ONNX Runtime inference runs the exported model instead.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .context import ContextSegmenterConfig, _config


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
