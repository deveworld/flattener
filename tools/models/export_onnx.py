"""Export a scanner bundle to static, single-file ONNX models for the web.

Run with ``uv run --group web-export python tools/models/export_onnx.py``.
Models use batch size 1; flip TTA runs sequentially to limit browser memory.
Every export is checked against PyTorch on spatially varying inputs before the
manifest is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from flattener._version import __version__
from flattener.scan.orient import exportable
from flattener.scan.pipeline import Scanner


class DewarpExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)[0]


class GateExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model._forward_trusted(image).sigmoid()


def export_model(module, inputs, path, names, outputs):
    module = module.eval()
    with torch.inference_mode():
        # Fixed shapes avoid dynamic-shape overhead and unsupported shape ops on WebGPU.
        torch.onnx.export(
            module,
            inputs,
            path,
            dynamo=False,
            input_names=names,
            output_names=outputs,
            opset_version=18,
            external_data=False,
        )
        expected = module(*inputs)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, torch.get_num_threads())
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = session.run(outputs, {name: value.numpy() for name, value in zip(names, inputs)})
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    for name, reference, value in zip(outputs, expected, actual):
        reference = reference.numpy()
        np.testing.assert_allclose(value, reference, atol=2e-4, rtol=2e-4)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "inputs": {name: list(value.shape) for name, value in zip(names, inputs)},
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, help="local bundle; default: download from Hugging Face"
    )
    parser.add_argument("--output", type=Path, default=Path("web/public/models"))
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    scanner = Scanner.load(args.bundle)
    if scanner.gate is None or scanner.orienter is None:
        parser.error("the browser bundle requires dewarp, gate, and orient models")
    if scanner.dewarper.config.grid_convention != "flattener":
        parser.error("the browser currently supports the flattener grid convention only")
    args.output.mkdir(parents=True, exist_ok=True)
    # Remove a previous manifest so interrupted exports cannot advertise mixed weights.
    manifest_path = args.output / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    generator = torch.Generator().manual_seed(42)

    def probe(shape, scale=1):
        return torch.rand(shape, generator=generator) * scale

    dewarp = export_model(
        DewarpExport(scanner.dewarper.model),
        (probe((1, 3, *scanner.dewarper.config.input_size)),),
        args.output / "dewarp.onnx",
        ["image"],
        ["grid"],
    )
    gate = export_model(
        GateExport(scanner.gate.model),
        (probe((1, 3, *scanner.gate.config.input_size)),),
        args.output / "gate.onnx",
        ["image"],
        ["probability"],
    )
    config = scanner.orienter.config
    orient = export_model(
        exportable(scanner.orienter),
        (
            probe((1, 1, config.global_size, config.global_size), 255),
            probe((1, config.patches, 1, config.patch_size, config.patch_size), 255),
        ),
        args.output / "orient.onnx",
        ["global_view", "patches"],
        ["logits", "probabilities"],
    )
    dewarp.update(
        input_size=list(scanner.dewarper.config.input_size),
        grid_size=list(scanner.dewarper.config.grid_size),
        grid_convention=scanner.dewarper.config.grid_convention,
        tta=scanner.dewarper.tta,
    )
    gate.update(
        input_size=list(scanner.gate.config.input_size),
    )
    orient.update(
        config=config.to_dict(),
        temperature=scanner.orienter.temperature,
        threshold=scanner.orienter.threshold,
    )
    manifest = {
        "schema": "flattener-web-bundle-1",
        "version": __version__,
        "precision": "float32",
        "opset": 18,
        "models": {"dewarp": dewarp, "gate": gate, "orient": orient},
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Exported browser models to {args.output}")


if __name__ == "__main__":
    main()
