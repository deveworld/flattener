"""Dewarp prediction: grid conventions, checkpoint loading and the DocUNet batch CLI."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from flattener import FlattenerError, RenderConfig
from flattener.benchmark.reporting import prediction_record, report_evidence
from flattener.contracts import Status
from flattener.dewarp import predict
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import (
    Dewarper,
    ieee_float32,
    load_dewarper,
    main,
    predict_grid,
    quantize_rgb,
    rectify,
    sample_rgb,
)
from flattener.renderer import render

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)


class _IdentityGrid(torch.nn.Module):
    def forward(self, x):
        grid = np.stack(np.meshgrid(edge_positions(4), edge_positions(5)), 0)
        return torch.from_numpy(grid).float()[None].expand(x.shape[0], -1, -1, -1), None


class _RecordFlags(_IdentityGrid):
    def forward(self, x):
        self.seen = _flags()
        return super().forward(x)


def _flags():
    backends = torch.backends
    return (
        backends.cudnn.conv.fp32_precision,
        backends.cuda.matmul.fp32_precision,
        backends.mkldnn.matmul.fp32_precision,
        torch.get_float32_matmul_precision(),
        backends.cudnn.benchmark,
    )


def _set_flags(conv, cuda_matmul, mkldnn_matmul, legacy, benchmark):
    torch.set_float32_matmul_precision(legacy)
    backends = torch.backends
    backends.cudnn.conv.fp32_precision = conv
    backends.cuda.matmul.fp32_precision = cuda_matmul
    backends.mkldnn.matmul.fp32_precision = mkldnn_matmul
    backends.cudnn.benchmark = benchmark


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


def test_prediction_runs_without_tf32_and_restores_the_flags():
    original = _flags()
    model = _RecordFlags()
    dewarper = Dewarper(model, TINY, torch.device("cpu"))
    try:
        _set_flags("tf32", "tf32", "tf32", "high", True)
        predict_grid(dewarper, np.zeros((8, 6, 3), np.uint8))
        assert model.seen == ("ieee", "ieee", "ieee", "highest", False)
        assert _flags() == ("tf32", "tf32", "tf32", "high", True)
        with pytest.raises(ValueError), ieee_float32():
            raise ValueError
        assert _flags() == ("tf32", "tf32", "tf32", "high", True)
    finally:
        _set_flags(*original)
    assert _flags() == original


def _upstream_weights():
    if not os.environ.get("FLATTENER_UVDOC_REPO"):
        pytest.skip("set FLATTENER_UVDOC_REPO to an upstream UVDoc checkout")
    repo = Path(os.environ["FLATTENER_UVDOC_REPO"])
    weights = repo / "model/best_model.pkl"
    if not weights.is_file():
        pytest.skip(f"UVDoc weights not found at {weights}")
    return weights


def _checkerboard_page():
    yy, xx = np.mgrid[0:120, 0:90]
    image = np.full((120, 90, 3), 40, np.uint8)
    image[10:110, 12:80] = 235
    image[(yy // 8 + xx // 8) % 2 == 0] //= 2
    return image


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_prediction_matches_cpu():
    # With TF32 convolutions the UVDoc grid drifts by ~1e-4 (0.1 px on a DocUNet crop).
    weights, image = _upstream_weights(), _checkerboard_page()
    cpu = predict_grid(load_dewarper(weights), image)
    cuda = predict_grid(load_dewarper(weights, device="cuda"), image)
    np.testing.assert_allclose(cuda, cpu, rtol=0, atol=1e-5)


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


def _identity_grid(dewarper, image):
    height, width = dewarper.config.grid_size
    return np.stack(np.meshgrid(edge_positions(width), edge_positions(height)), -1).astype(
        np.float32
    )


def test_docunet_batch_writes_crop_sized_outputs(tmp_path, capsys, monkeypatch):
    # A random tiny model folds its map; an identity grid passes the render checks.
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
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
    counts = json.loads(capsys.readouterr().out)
    assert counts == {"items": 2, "written": 2, "kept": 0, "failed": 0, "forced": 0}
    for name, size in sizes.items():
        with Image.open(output / f"{name}.png") as written:
            assert written.size == size[::-1]
    report = json.loads((output / "predictions.json").read_text())
    assert [item["output"] for item in report["items"]] == ["1_1.png", "10_2.png"]
    assert report["model"]["track"] == "release" and report["inputs"] == "docunet_crop"
    assert report["evidence"]["evidence_kind"] == "smoke_training"  # a 3-step checkpoint
    assert report["evidence"]["measurement"] == "predictions"
    assert main(arguments) == 0 and json.loads(capsys.readouterr().out)["written"] == 0
    kept = json.loads((output / "predictions.json").read_text())["items"]
    assert [(item["action"], item["status"], item["size"]) for item in kept] == [
        ("kept", "success", [45, 33]),
        ("kept", "success", [30, 52]),
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
    dewarper = load_dewarper(_upstream_weights())
    assert dewarper.info["kind"] == "uvdoc" and dewarper.config.grid_convention == "uvdoc"
    image = _checkerboard_page()
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


def test_truncate_is_floor_of_x_times_255_on_a_known_sample():
    # Source columns a and b; a 4-wide output samples 1/4 and 3/4 of the way between them.
    image = np.zeros((4, 2, 3), np.uint8)
    image[:, 0], image[:, 1] = (0, 10, 200), (101, 31, 253)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    exact = np.array([[0, 10, 200], [25.25, 15.25, 213.25], [75.75, 25.75, 239.75], image[0, 1]])
    truncated = rectify(dewarper, image, (4, 4), quantize="truncate")
    rounded = rectify(dewarper, image, (4, 4))
    assert truncated.status == rounded.status == Status.SUCCESS
    np.testing.assert_array_equal(truncated.image, np.broadcast_to(np.floor(exact), (4, 4, 3)))
    np.testing.assert_array_equal(rounded.image, np.broadcast_to(np.floor(exact + 0.5), (4, 4, 3)))
    assert truncated.metadata["dewarp"]["quantize"] == "truncate"
    assert rounded.metadata["dewarp"]["quantize"] == "round"
    values = np.array([0, 0.1, 0.5, 0.7, 0.999, 1, 1.25, -0.5], np.float32)
    assert quantize_rgb(values, "truncate").tolist() == [0, 25, 127, 178, 254, 255, 255, 0]
    assert quantize_rgb(values, "round").tolist() == [0, 26, 128, 179, 255, 255, 255, 0]
    inside = values[(values >= 0) & (values <= 1)]  # the paper's (x * 255).astype(uint8)
    np.testing.assert_array_equal(quantize_rgb(inside, "truncate"), (inside * 255).astype(np.uint8))


def test_truncate_samples_what_flattener_render_rounds():
    image = np.random.default_rng(4).integers(0, 256, (30, 20, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(np.linspace(-1.1, 0.9, 25), np.linspace(-0.8, 1.05, 18), indexing="ij")
    backward_map = np.stack([xx, yy], -1).astype(np.float32)  # some pixels outside the source
    fill = (1, 2, 3)
    checked = render(image, backward_map, config=RenderConfig(fill_rgb=fill))
    assert checked.status == Status.SUCCESS and not checked.valid_mask.all()
    rounded = sample_rgb(image, backward_map, checked.valid_mask, fill, "round")
    np.testing.assert_array_equal(rounded, checked.image)
    truncated = sample_rgb(image, backward_map, checked.valid_mask, fill, "truncate")
    gap = checked.image.astype(int) - truncated
    assert gap.min() == 0 and gap.max() == 1  # truncation is at most one level lower
    assert (truncated[~checked.valid_mask] == fill).all()
    with pytest.raises(FlattenerError, match="quantize"):
        quantize_rgb(np.zeros(3, np.float32), "nearest")


def test_truncate_differs_from_round_by_at_most_one_level():
    image = np.random.default_rng(5).integers(0, 256, (37, 29, 3), dtype=np.uint8)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    rounded = rectify(dewarper, image, (50, 41))
    truncated = rectify(dewarper, image, (50, 41), quantize="truncate")
    gap = rounded.image.astype(int) - truncated.image
    assert gap.min() == 0 and gap.max() == 1
    assert truncated.metadata["validation"] == rounded.metadata["validation"]


def test_docunet_batch_never_keeps_outputs_of_another_quantization(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
    crops, output = tmp_path / "crop", tmp_path / "out"
    crops.mkdir()
    pixels = np.random.default_rng(6).integers(0, 256, (45, 33, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(crops / "1_1 copy.png")
    weights = tmp_path / "model.pt"
    _checkpoint(weights)
    record_path = output / "predictions.json"
    arguments = ["--weights", str(weights), "docunet", "--input-dir", str(crops)]
    arguments += ["--output-dir", str(output)]

    def run(quantize):
        assert main([*arguments, "--quantize", quantize]) == 0
        counts = json.loads(capsys.readouterr().out)
        record = json.loads(record_path.read_text())
        assert record["quantize"] == record["evidence"]["quantize"] == quantize
        return counts["written"], counts["kept"]

    assert run("round") == (1, 0)
    assert run("round") == (0, 1)
    assert run("truncate") == (1, 0)  # a rounded image is never kept for truncation
    assert run("truncate") == (0, 1)
    found = prediction_record(output)
    assert found["quantize"] == report_evidence(found)["quantize"] == "truncate"
    assert run("round") == (1, 0)
    # A record without the field predates the option; its images were rounded.
    record = json.loads(record_path.read_text())
    del record["quantize"]
    record_path.write_text(json.dumps(record))
    assert prediction_record(output)["quantize"] == "round"
    assert run("round") == (0, 1)
    record = json.loads(record_path.read_text())
    del record["quantize"]
    record_path.write_text(json.dumps(record))
    assert run("truncate") == (1, 0)
