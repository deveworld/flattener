"""ONNX Runtime scans: the plans of the PyTorch models, without importing PyTorch."""

import json
import subprocess
import sys
import warnings
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image
from scan_fakes import Page, ramp_scene
from torch import nn

from flattener import FlattenerError, runtime
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import Dewarper
from flattener.regions.context import ContextSegmenterConfig, LoadedContextSegmenter
from flattener.regions.context_net import ContextPageSegmenter
from flattener.scan import locate
from flattener.scan.orient import OrientConfig, Orienter
from flattener.scan.orient_net import OrientNet, exportable
from flattener.scan.pipeline import Scanner

SIZE = (240, 320)
PAGE = Page((170, 125), (150, 190), 7.0)
DEWARP = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
GATE = ContextSegmenterConfig(input_size=(17, 13), channels=(2, 3, 4, 5))
ORIENT = OrientConfig(
    global_size=32,
    patch_size=16,
    patches=1,
    page_long_side=128,
    energy_grid=3,
    stem=4,
    blocks=((8, 2, 3, False), (8, 1, 3, True)),
    hidden=8,
)


class _Grid(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)[0]


class _Probability(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model._forward_trusted(image).sigmoid()


def _export(module, inputs, path, names, outputs):
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module.eval(),
            inputs,
            path,
            dynamo=False,
            input_names=names,
            output_names=outputs,
            opset_version=18,
            external_data=False,
        )


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A PyTorch scanner of random small models and the ONNX bundle exported from it."""
    torch.manual_seed(0)
    dewarp = UVDocNet(DEWARP).eval()
    gate = ContextPageSegmenter(GATE).eval()
    orient = OrientNet(ORIENT).eval()
    with torch.no_grad():  # a gate that sees the page: larger weights than the init
        for parameter in gate.parameters():
            parameter.mul_(3)
    dewarper = Dewarper(dewarp, DEWARP, torch.device("cpu"), {"path": "dewarp"}, "flip")
    segmenter = LoadedContextSegmenter(gate, GATE, {"path": "gate"}, 0.0, torch.device("cpu"))
    orienter = Orienter(orient, temperature=1.5, threshold=0.5)
    scanner = Scanner(dewarper, gate=segmenter, orienter=orienter)
    directory = tmp_path_factory.mktemp("onnx")
    _export(_Grid(dewarp), (torch.rand(1, 3, *DEWARP.input_size),), directory / "dewarp.onnx",
            ["image"], ["grid"])  # fmt: skip
    _export(_Probability(gate), (torch.rand(1, 3, *GATE.input_size),), directory / "gate.onnx",
            ["image"], ["probability"])  # fmt: skip
    views = (torch.rand(1, 1, 32, 32) * 255, torch.rand(1, 1, 1, 16, 16) * 255)
    _export(exportable(orienter), views, directory / "orient.onnx", ["global_view", "patches"],
            ["logits", "probabilities"])  # fmt: skip
    manifest = {
        "schema": "flattener-web-bundle-1",
        "version": "test",
        "models": {
            "dewarp": {"file": "dewarp.onnx", "tta": "flip", "model_config": DEWARP.to_dict()},
            "gate": {"file": "gate.onnx", "model_config": asdict(GATE)},
            "orient": {
                "file": "orient.onnx",
                "config": ORIENT.to_dict(),
                "temperature": 1.5,
                "threshold": 0.5,
            },
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return scanner, path


@pytest.fixture(autouse=True)
def small_pages(monkeypatch):
    monkeypatch.setitem(locate.THRESHOLDS, "min_page_long_side_px", 50)


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.png"
    Image.fromarray(ramp_scene(SIZE, PAGE)).save(path)
    return path


def test_onnx_models_give_the_plans_of_the_torch_models(models, photo):
    torch_scanner, manifest = models
    onnx_scanner = Scanner.load(manifest)
    assert (onnx_scanner.backend, torch_scanner.backend) == ("onnx", "torch")
    assert onnx_scanner.dewarper.tta == "flip" and onnx_scanner.orienter.threshold == 0.5
    image = np.asarray(Image.open(photo).convert("RGB"))
    np.testing.assert_allclose(
        onnx_scanner.gate.predict_probability(image),
        torch_scanner.gate.predict_probability(image),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        onnx_scanner.gate.predict_logits(image), torch_scanner.gate.predict_logits(image), atol=1e-3
    )
    want, got = torch_scanner.scan(photo), onnx_scanner.scan(photo)
    assert got.outcome == want.outcome and got.reason_code == want.reason_code
    if want.plan is not None:
        np.testing.assert_allclose(got.plan.grid, want.plan.grid, atol=1e-4)
    assert got.image.shape == want.image.shape
    assert np.abs(got.image.astype(int) - want.image.astype(int)).mean() < 0.5


def test_onnx_scans_run_without_pytorch(models, photo, tmp_path):
    _, manifest = models
    output = tmp_path / "scan.png"
    script = f"""
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("PyTorch is blocked")

sys.meta_path.insert(0, Block())
from flattener.scan import locate
from flattener.scan.__main__ import main

locate.THRESHOLDS["min_page_long_side_px"] = 50
code = main(["-i", {str(photo)!r}, {str(output)!r}, "--bundle", {str(manifest)!r}])
assert "torch" not in sys.modules
sys.exit(code)
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode in (0, 1), done.stderr
    assert "scan.png" in done.stderr and output.is_file()


def test_backend_choice(monkeypatch):
    monkeypatch.setattr(runtime, "torch_installed", lambda: True)
    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    assert runtime.resolve() == ("onnx", "cpu")
    assert runtime.resolve("torch") == ("torch", "cpu")
    with pytest.raises(FlattenerError, match="CUDA"):
        runtime.resolve("auto", "cuda")
    monkeypatch.setattr(runtime, "cuda_available", lambda: True)
    assert runtime.resolve() == ("torch", "cuda")
    assert runtime.resolve("onnx") == ("onnx", "cpu")
    assert runtime.resolve("auto", "cpu") == ("onnx", "cpu")
    assert runtime.resolve("auto", "cuda:1") == ("torch", "cuda:1")
    with pytest.raises(FlattenerError, match="cpu"):
        runtime.resolve("onnx", "cuda")
    monkeypatch.setattr(runtime, "torch_installed", lambda: False)
    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    assert runtime.resolve() == ("onnx", "cpu")
    with pytest.raises(FlattenerError, match="PyTorch is not installed"):
        runtime.resolve("torch")
    with pytest.raises(FlattenerError, match="device must be"):
        runtime.resolve("auto", "gpu")
