"""Dewarp prediction: grid conventions, checkpoint loading and the DocUNet batch CLI."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from flattener import FlattenerError
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import Dewarper, load_dewarper, main, predict_grid, rectify

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
DEFAULT_UVDOC_REPO = Path.home() / ".cache/flattener-sota-20260926/repos/UVDoc"


class _IdentityGrid(torch.nn.Module):
    def forward(self, x):
        grid = np.stack(np.meshgrid(edge_positions(4), edge_positions(5)), 0)
        return torch.from_numpy(grid).float()[None].expand(x.shape[0], -1, -1, -1), None


def _checkpoint(path, seed=0):
    torch.manual_seed(seed)
    model = UVDocNet(TINY).eval()
    save_checkpoint(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": 3,
            "track": "release",
            "model_config": TINY.to_dict(),
            "lineage": [{"kind": "scratch"}],
            "data_spec_sha256": "0" * 64,
            "code": {"sha256": "1" * 64},
            "model_state": model.state_dict(),
        },
    )
    return model


def test_identity_grid_reproduces_the_input_and_resizes():
    image = np.random.default_rng(0).integers(0, 256, (37, 29, 3), dtype=np.uint8)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    result = rectify(dewarper, image)
    np.testing.assert_array_equal(result.image, image)
    assert not result.used_fallback and result.metadata["dewarp"]["grid_size"] == [5, 4]
    assert rectify(dewarper, image, (50, 20)).image.shape == (50, 20, 3)
    with pytest.raises(FlattenerError):
        rectify(dewarper, image.astype(np.float32))


def test_checkpoint_loads_with_verification(tmp_path):
    path = tmp_path / "model.pt"
    model = _checkpoint(path)
    dewarper = load_dewarper(path)
    assert dewarper.info["track"] == "release" and dewarper.info["kind"] == "checkpoint"
    image = np.random.default_rng(1).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    tensor = F.interpolate(
        torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255,
        size=(64, 48),
        mode="bilinear",
        align_corners=False,
    )
    with torch.no_grad():
        expected = model(tensor)[0][0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(predict_grid(dewarper, image), expected, atol=1e-6)
    data = bytearray(path.read_bytes())
    data[-5] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(FlattenerError, match="sha256"):
        load_dewarper(path)


def test_docunet_batch_writes_crop_sized_outputs(tmp_path, capsys):
    crops, output = tmp_path / "crop", tmp_path / "out"
    crops.mkdir()
    rng = np.random.default_rng(2)
    sizes = {"1_1": (45, 33), "10_2": (30, 52)}
    for name, size in sizes.items():
        Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(
            crops / f"{name} copy.png"
        )
    (crops / "notes.txt").write_text("ignored")
    weights = tmp_path / "model.pt"
    _checkpoint(weights)
    arguments = [
        "--weights",
        str(weights),
        "docunet",
        "--input-dir",
        str(crops),
        "--output-dir",
        str(output),
    ]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"items": 2, "written": 2}
    for name, size in sizes.items():
        with Image.open(output / f"{name}.png") as written:
            assert written.size == size[::-1]
    report = json.loads((output / "predictions.json").read_text())
    assert [item["output"] for item in report["items"]] == ["1_1.png", "10_2.png"]
    assert report["model"]["track"] == "release"
    assert main(arguments) == 0 and json.loads(capsys.readouterr().out)["written"] == 0
    kept = json.loads((output / "predictions.json").read_text())["items"]
    assert [(item["status"], item["size"]) for item in kept] == [
        ("kept", [45, 33]),
        ("kept", [30, 52]),
    ]
    other = tmp_path / "other.pt"
    _checkpoint(other, seed=1)
    with pytest.raises(FlattenerError, match="other weights"):
        main(["--weights", str(other), *arguments[2:]])
    # A file the same-weights report does not list is re-rendered, not kept.
    report = json.loads((output / "predictions.json").read_text())
    report["items"] = report["items"][:1]
    (output / "predictions.json").write_text(json.dumps(report))
    assert main(arguments) == 0 and json.loads(capsys.readouterr().out)["written"] == 1
    # Outputs of unknown provenance (no report) are refused unless --overwrite.
    (output / "predictions.json").unlink()
    with pytest.raises(FlattenerError, match="without predictions.json"):
        main(arguments)
    assert main([*arguments, "--overwrite"]) == 0
    assert json.loads(capsys.readouterr().out)["written"] == 2


def test_upstream_weights_match_the_upstream_unwarping():
    repo = Path(os.environ.get("FLATTENER_UVDOC_REPO", DEFAULT_UVDOC_REPO))
    weights = repo / "model/best_model.pkl"
    if not weights.is_file():
        pytest.skip(f"UVDoc weights not found at {weights}")
    dewarper = load_dewarper(weights)
    assert dewarper.info["kind"] == "uvdoc" and dewarper.config.grid_convention == "uvdoc"
    yy, xx = np.mgrid[0:120, 0:90]
    image = np.full((120, 90, 3), 40, np.uint8)
    image[10:110, 12:80] = 235
    image[(yy // 8 + xx // 8) % 2 == 0] //= 2
    result = rectify(dewarper, image, allow_unchecked=True)
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
    small = F.interpolate(tensor, size=(712, 488), mode="bilinear", align_corners=False)
    with torch.no_grad():
        grid = dewarper.model(small)[0]
    dense = F.interpolate(grid, size=(120, 90), mode="bilinear", align_corners=True)
    upstream = F.grid_sample(tensor, dense.permute(0, 2, 3, 1), align_corners=True)
    expected = np.floor(upstream[0].permute(1, 2, 0).numpy() * 255 + 0.5)
    inside = (dense[0].abs() <= 0.999).all(0).numpy()
    assert inside.mean() > 0.5
    assert np.abs(result.image[inside].astype(int) - expected[inside]).max() <= 1
