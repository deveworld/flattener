"""UVDocnet: grid-regression document dewarping network with configurable sizes.

Derived from UVDoc (https://github.com/tanguymagne/UVDoc, ``model.py`` at
commit 4c9b82b537057aff2526e6dd118a847cdd072e82). Module and parameter names
match upstream so that ``model/best_model.pkl`` loads unchanged, and the
default configuration reproduces the upstream forward pass exactly. Changes:
configurable input/grid sizes (the bridge features are bilinearly resized when
the grid differs from the backbone's natural size), an optional 3-D head, and
compact construction helpers.

The derived portions are distributed under the upstream MIT license:

MIT License

Copyright (c) 2023 Tanguy MAGNE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import (  # noqa: F401  (re-exported)
    ARCHITECTURE,
    GRID_CONVENTIONS,
    UVDOC_CONFIG,
    UVDOC_REVISION,
    DewarpModelConfig,
    _pair,
    natural_grid_size,
)

_BRIDGE_DILATIONS = ((1,), (2,), (5,), (8, 3, 2), (12, 7, 4), (18, 12, 6))


def _conv(cin: int, cout: int, k: int, stride: int = 1, dilation: int = 1, **kw: Any) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, k, stride=stride, padding=dilation * (k // 2), dilation=dilation, **kw
    )


def _conv_bn_relu(channels: int, dilation: int) -> nn.Sequential:
    """Upstream ``dilated_conv_bn_act``: 3x3 dilated conv, BN, ReLU."""
    return nn.Sequential(
        _conv(channels, channels, 3, dilation=dilation, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


class ResidualBlockWithDilation(nn.Module):
    """Plain convolutions for the first block of a layer, dilation 3 otherwise."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        is_top: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        if stride != 1 or is_top:
            self.conv1 = _conv(cin, cout, k, stride=stride)
            self.conv2 = _conv(cout, cout, k)
        else:  # upstream wraps dilated convolutions in a Sequential (state-dict key ".0.")
            self.conv1 = nn.Sequential(_conv(cin, cout, k, dilation=3))
            self.conv2 = nn.Sequential(_conv(cout, cout, k, dilation=3))
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        out += residual
        return self.relu(out)


class ResnetStraight(nn.Module):
    def __init__(self, num_filter: int, k: int) -> None:
        super().__init__()
        self.in_channels = num_filter
        self.layer1 = self._layer(num_filter, 3, k, stride=1)
        self.layer2 = self._layer(num_filter * 2, 4, k, stride=2)
        self.layer3 = self._layer(num_filter * 4, 6, k, stride=2)

    def _layer(self, cout: int, blocks: int, k: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != cout:
            downsample = nn.Sequential(
                _conv(self.in_channels, cout, k, stride=stride), nn.BatchNorm2d(cout)
            )
        layers = [ResidualBlockWithDilation(self.in_channels, cout, k, stride, downsample, True)]
        self.in_channels = cout
        layers += [ResidualBlockWithDilation(cout, cout, k) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(x)))


def _head(cin: int, mid: int, cout: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        _conv(cin, mid, k, bias=False, padding_mode="reflect"),
        nn.BatchNorm2d(mid),
        nn.PReLU(),
        _conv(mid, cout, k, padding_mode="reflect"),
    )


class UVDocNet(nn.Module):
    """Returns ``(grid2d, grid3d)`` with shapes (B, 2, Gh, Gw) and (B, 3, Gh, Gw).

    ``grid3d`` is ``None`` when the configuration disables the 3-D head. The
    model is built NCHW: channels_last measured ~4x slower on the target RTX
    3060 Ti. The trainer's ``channels_last`` option converts it for training.
    """

    def __init__(self, config: DewarpModelConfig | None = None) -> None:
        super().__init__()
        self.config = config = UVDOC_CONFIG if config is None else config
        if not isinstance(config, DewarpModelConfig):
            raise TypeError("config must be a DewarpModelConfig")
        nf, k = config.num_filter, config.kernel_size
        self.resnet_head = nn.Sequential(
            _conv(3, nf, k, stride=2, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            _conv(nf, nf, k, stride=2, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
        )
        self.resnet_down = ResnetStraight(nf, k)
        width = nf * 4
        for index, dilations in enumerate(_BRIDGE_DILATIONS, start=1):
            setattr(
                self,
                f"bridge_{index}",
                nn.Sequential(*(_conv_bn_relu(width, d) for d in dilations)),
            )
        self.bridge_concat = nn.Sequential(
            nn.Conv2d(width * 6, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)
        )
        self.out_point_positions2D = _head(width, nf, 2, k)
        self.out_point_positions3D = _head(width, nf, 3, k) if config.with_3d else None
        self._resize = config.grid_size != config.natural_grid_size
        for module in self.modules():  # upstream initialization
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight, gain=0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.resnet_down(self.resnet_head(x))
        bridges = [getattr(self, f"bridge_{i}")(features) for i in range(1, 7)]
        bridge = self.bridge_concat(torch.cat(bridges, dim=1))
        if self._resize:  # corner features stay on the edge-aligned corner nodes
            bridge = F.interpolate(
                bridge, size=self.config.grid_size, mode="bilinear", align_corners=True
            )
        grid3d = None if self.out_point_positions3D is None else self.out_point_positions3D(bridge)
        return self.out_point_positions2D(bridge), grid3d


def count_parameters(model: nn.Module) -> dict[str, int]:
    params = list(model.parameters())
    return {
        "total": sum(p.numel() for p in params),
        "trainable": sum(p.numel() for p in params if p.requires_grad),
    }


def load_uvdoc_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read upstream ``best_model.pkl`` safely: exactly ``{"model_state": state_dict}``.

    Anything else (e.g. a dewarp checkpoint, which also has ``model_state``)
    is refused, so only upstream-format weights load as upstream.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
    if (
        set(checkpoint if isinstance(checkpoint, dict) else ()) != {"model_state"}
        or not isinstance(state, dict)
        or not all(isinstance(v, torch.Tensor) for v in state.values())
    ):
        raise ValueError(f"{path} is not an upstream UVDoc file holding only a 'model_state' dict")
    return state


def load_uvdoc_weights(model: UVDocNet, path: str | Path) -> dict[str, Any]:
    """Load weights in the upstream UVDoc format."""
    state = load_uvdoc_state_dict(path)
    if model.out_point_positions3D is None:
        state = {k: v for k, v in state.items() if not k.startswith("out_point_positions3D.")}
    model.load_state_dict(state, strict=True)
    return {"kind": "uvdoc", "path": str(path)}


def uvdoc_model(path: str | Path) -> tuple[UVDocNet, dict[str, Any]]:
    """Upstream-equivalent model (UVDOC_CONFIG) with the MIT weights, in eval mode."""
    model = UVDocNet(UVDOC_CONFIG)
    lineage = load_uvdoc_weights(model, path)
    return model.eval(), lineage
