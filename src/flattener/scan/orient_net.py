"""OrientNet-1 in PyTorch: the network, its cost and its ONNX export.

``scan.orient`` holds the views, the decision and the loader; ONNX Runtime inference
runs the exported model without this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import Status
from ..dewarp.checkpoint import environment, utc_now, write_json
from .orient import (
    CLASSES,
    EXPORT_SCHEMA,
    MODEL_ID,
    VIEWS_VERSION,
    OrientConfig,
    Orienter,
    PageViews,
    _error,
    load_checkpoint,
    views_to_arrays,
)


class _ConvBN(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.Hardswish(),
        )


class _SqueezeExcite(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 4)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.expand(F.relu(self.reduce(x.mean((2, 3), keepdim=True))))
        return x * F.hardsigmoid(scale)


class _Block(nn.Module):
    """Depthwise k x k (with the stride), optional squeeze-excitation, pointwise 1 x 1."""

    def __init__(self, cin: int, cout: int, stride: int, kernel: int, se: bool) -> None:
        super().__init__()
        self.depthwise = _ConvBN(cin, cin, kernel, stride, groups=cin)
        self.se = _SqueezeExcite(cin) if se else nn.Identity()
        self.pointwise = _ConvBN(cin, cout, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.se(self.depthwise(x)))


class OrientNet(nn.Module):
    """Shared trunk over the global view and the patches, saliency-weighted pooling, 4 logits.

    Inputs are float gray values in [0, 255]: ``global_view`` (N, 1, S, S) and
    ``patches`` (N, P, 1, s, s) or ``None`` (global view only).
    """

    def __init__(self, config: OrientConfig | None = None) -> None:
        super().__init__()
        self.config = config or OrientConfig()
        layers: list[nn.Module] = [_ConvBN(1, self.config.stem, 3, 2)]
        channels = self.config.stem
        for out, stride, kernel, se in self.config.blocks:
            layers.append(_Block(channels, out, stride, kernel, se))
            channels = out
        self.trunk = nn.Sequential(*layers)
        self.saliency = nn.Conv2d(channels, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(channels, self.config.hidden),
            nn.Hardswish(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden, CLASSES),
        )

    def _tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(M, 1, H, W) -> features (M, L, C) and saliency logits (M, L)."""
        features = self.trunk(images / 127.5 - 1.0)
        scores = self.saliency(features)
        return features.flatten(2).transpose(1, 2), scores.flatten(1)

    def _pool(self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(scores.float(), dim=1).to(tokens.dtype)
        return self.head((weights.unsqueeze(-1) * tokens).sum(1))

    def forward_views(
        self, global_view: torch.Tensor, patches: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Logits pooled over all views, over the global view alone and over the patches."""
        global_tokens, global_scores = self._tokens(global_view)
        alone = self._pool(global_tokens, global_scores)
        if patches is None or patches.shape[1] == 0:
            return alone, alone, None
        batch, channels = global_view.shape[0], global_tokens.shape[-1]
        patch_tokens, patch_scores = self._tokens(patches.flatten(0, 1))
        patch_tokens = patch_tokens.reshape(batch, -1, channels)
        patch_scores = patch_scores.reshape(batch, -1)
        both = self._pool(
            torch.cat([global_tokens, patch_tokens], 1), torch.cat([global_scores, patch_scores], 1)
        )
        return both, alone, self._pool(patch_tokens, patch_scores)

    def forward(self, global_view: torch.Tensor, patches: torch.Tensor | None = None):
        return self.forward_views(global_view, patches)[0]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_cost(config: OrientConfig | None = None) -> dict[str, Any]:
    """Parameters and forward FLOPs for one page (global view plus patches)."""
    from torch.utils.flop_counter import FlopCounterMode

    config = config or OrientConfig()
    model = OrientNet(config).eval()
    views = example_inputs(config, 1)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(*views)
    return {
        "parameters": count_parameters(model),
        "flops": int(counter.get_total_flops()),
        "inputs": {
            "global_view": list(views[0].shape),
            "patches": list(views[1].shape),
        },
    }


def example_inputs(config: OrientConfig, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Mid-grey inputs of the model's shapes."""
    size, patch = config.global_size, config.patch_size
    return (
        torch.full((batch, 1, size, size), float(config.pad_value)),
        torch.full((batch, config.patches, 1, patch, patch), float(config.pad_value)),
    )


def views_to_tensors(views: list[PageViews]) -> tuple[torch.Tensor, torch.Tensor]:
    """Float (N, 1, S, S) and (N, P, 1, s, s) batches from ``PageViews``."""
    global_view, patches = views_to_arrays(views)
    return torch.from_numpy(global_view), torch.from_numpy(patches)


class _Exported(nn.Module):
    def __init__(self, model: OrientNet, temperature: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("inverse_temperature", torch.tensor(1.0 / temperature))

    def forward(self, global_view: torch.Tensor, patches: torch.Tensor):
        logits = self.model(global_view, patches)
        return logits, torch.softmax(logits * self.inverse_temperature, dim=1)


def exportable(orienter: Orienter) -> nn.Module:
    """The module an export traces: (global_view, patches) -> (logits, probabilities)."""
    return _Exported(orienter.model.cpu().eval(), orienter.temperature).eval()


def export_onnx(
    checkpoint: str | Path,
    output: str | Path,
    *,
    opset: int = 18,
    temperature: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Write ``output`` (ONNX, dynamic batch) and ``output.json`` describing inputs,
    preprocessing and calibration. Needs the ``onnx`` and
    ``onnxscript`` packages; with ``onnxruntime`` installed the export is checked
    against the torch model."""
    orienter = Orienter.load(checkpoint, temperature=temperature, threshold=threshold)
    module = exportable(orienter)
    config = orienter.config
    inputs = example_inputs(config, 2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch", min=1, max=1024)
    try:
        program = torch.onnx.export(
            module,
            inputs,
            dynamo=True,
            input_names=["global_view", "patches"],
            output_names=["logits", "probabilities"],
            dynamic_shapes={"global_view": {0: batch}, "patches": {0: batch}},
            opset_version=opset,
            external_data=False,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise _error(
            Status.INPUT_ERROR,
            "missing_dependency",
            f"ONNX export needs the onnx and onnxscript packages ({error})",
        ) from error
    temporary = output.with_name(output.name + ".tmp")
    try:
        program.save(temporary, external_data=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    # Random gray views: a constant input would hide a spatial layout error in the export.
    generator = torch.Generator().manual_seed(0)
    probe = tuple(torch.randint(0, 256, x.shape, generator=generator).float() for x in inputs)
    parity = _onnx_parity(output, module, probe)
    payload = load_checkpoint(checkpoint)
    record = {
        "schema": EXPORT_SCHEMA,
        "model_id": MODEL_ID,
        "file": output.name,
        "bytes": output.stat().st_size,
        "opset": opset,
        "checkpoint": {**orienter.info, "path": str(checkpoint)},
        "temperature": orienter.temperature,
        "threshold": orienter.threshold,
        "calibration": payload.get("calibration"),
        "model_config": config.to_dict(),
        "cost": model_cost(config),
        "inputs": {
            "global_view": {
                "shape": ["batch", 1, config.global_size, config.global_size],
                "dtype": "float32",
                "range": "gray values 0-255",
            },
            "patches": {
                "shape": ["batch", config.patches, 1, config.patch_size, config.patch_size],
                "dtype": "float32",
                "range": "gray values 0-255",
            },
        },
        "outputs": {
            "logits": "raw logits (batch, 4)",
            "probabilities": "softmax(logits / temperature) (batch, 4)",
        },
        "labels": "k = clockwise quarter turns of the text-up direction; rot90(view, k) is upright",
        "preprocessing": {
            "version": VIEWS_VERSION,
            "gray": "ITU-R 601 luma (PIL L)",
            "global_view": "page view letterboxed to global_size, centred, pad_value outside",
            "patches": "view scaled to page_long_side; centres of the energy_grid cells with the "
            "highest std of gradient magnitude on the global view, outer ring excluded",
            "sampling": "bilinear, align_corners=False, integer box reduction by "
            "max(1, int(density / 1.5)); quantize floor(clip(v, 0, 255) + 0.5)",
        },
        "parity": parity,
        "environment": environment(),
        "created_utc": utc_now(),
    }
    write_json(output.with_name(output.name + ".json"), record)
    return record


def _onnx_parity(path: Path, module: nn.Module, inputs) -> dict[str, Any] | None:
    try:
        import onnxruntime
    except ImportError:
        return None
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {"global_view": inputs[0].numpy(), "patches": inputs[1].numpy()}
    got = session.run(["logits"], feed)[0]
    with torch.no_grad():
        want = module(*inputs)[0].numpy()
    return {
        "max_abs_logit_difference": float(np.abs(got - want).max()),
        "inputs": "uniform random gray values 0-255, seed 0",
    }
