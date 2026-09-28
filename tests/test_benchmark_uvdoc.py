"""UVDoc benchmark: archive preparation, prediction, scoring and the command line."""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from flattener.benchmark import __main__ as cli
from flattener.benchmark import metrics, uvdoc
from flattener.benchmark.reporting import prediction_record, report_evidence
from flattener.dewarp import predict
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.evidence import SMOKE_STEPS

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
FOLDERS = ("img", "warped_textures", "texture_sample")


def _tiny_images(root):
    """50 small images per folder; textures and photos differ in size."""
    rng = np.random.default_rng(3)
    shapes = {"img": (26, 20), "warped_textures": (22, 17), "texture_sample": (24, 18)}
    for folder in FOLDERS:
        (root / folder).mkdir(parents=True)
        for sample in uvdoc.SAMPLES:
            pixels = rng.integers(0, 256, (*shapes[folder], 3), np.uint8)
            Image.fromarray(pixels).save(root / folder / f"{sample}.png")


def _pin_sets(root, monkeypatch):
    sets = {
        folder: uvdoc.set_sha256(uvdoc.hash_folder(root, folder, verify=False))
        for folder in FOLDERS
    }
    monkeypatch.setattr(uvdoc, "SET_SHA256", sets)


@pytest.fixture
def tiny_root(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRIC_AREA", 300)
    root = tmp_path / "bench"
    _tiny_images(root)
    _pin_sets(root, monkeypatch)
    return root


def _identity_grid(dewarper, image):
    height, width = dewarper.config.grid_size
    grid = np.meshgrid(edge_positions(width), edge_positions(height))
    return np.stack(grid, -1).astype(np.float32)


def _tiny_weights(tmp_path):
    torch.manual_seed(0)
    weights = tmp_path / "model.pt"
    save_checkpoint(
        weights,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": SMOKE_STEPS,
            "track": "release",
            "model_config": TINY.to_dict(),
            "lineage": [{"kind": "scratch"}],
            "data_spec_sha256": "0" * 64,
            "code": {"sha256": "1" * 64},
            "model_state": UVDocNet(TINY).state_dict(),
        },
    )
    return weights


def _upstream_unwarping(image, grid):
    """uvdocBenchmark_pred.py: bilinear_unwarping, then (x * 255).astype(uint8)."""
    warped = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float().unsqueeze(0)
    points = torch.from_numpy(grid.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    size = (image.shape[1], image.shape[0])
    upsampled = F.interpolate(points, size=(size[1], size[0]), mode="bilinear", align_corners=True)
    unwarped = F.grid_sample(warped, upsampled.transpose(1, 2).transpose(2, 3), align_corners=True)
    return (unwarped[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def test_unwarp_follows_upstream_and_both_conventions_agree():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (23, 17, 3), np.uint8)
    grid = rng.uniform(-1.2, 1.2, (5, 4, 2)).astype(np.float32)
    assert np.array_equal(uvdoc.unwarp(image, grid, "uvdoc"), _upstream_unwarping(image, grid))
    # Identity grids return the image in either convention; truncation of the float
    # interpolation loses at most one level.
    xs, ys = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 5))
    identity = np.stack([xs, ys], -1).astype(np.float32)
    for convention in ("uvdoc", "flattener"):
        difference = uvdoc.unwarp(image, identity, convention).astype(int) - image
        assert difference.min() >= -1 and difference.max() == 0
        assert np.mean(difference == 0) > 0.5
    # Outside the source both pad with zeros (black).
    shifted = identity + np.float32([3.0, 0.0])
    assert not uvdoc.unwarp(image, shifted, "uvdoc").any()
    assert not uvdoc.unwarp(image, shifted, "flattener").any()
    with pytest.raises(ValueError):
        uvdoc.unwarp(image, grid, "other")


def test_predict_writes_textures_and_photos_and_applies_the_inset(tiny_root, tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
    dewarper = predict.load_dewarper(_tiny_weights(tmp_path))
    grid = _identity_grid(dewarper, None)
    output = tmp_path / "out"
    record = uvdoc.predict_uvdoc(dewarper, output, root=tiny_root)
    assert record["complete"] and record["counts"]["items"] == 100
    assert record["counts"]["failed"] == 0 and record["quantize"] == "truncate"
    assert record["inset"] == {"x": 0.0, "y": 0.0, "preset": None}
    stored = prediction_record(output)
    assert stored["inputs"] == "uvdoc_benchmark" and stored["failures"] == {}
    assert "uwp_texture/00049.png" in stored["outputs"]
    assert report_evidence(stored)["evidence_kind"] == "new_model_real_photo"
    preset = "syntheticdoc_half_cell"
    inset = uvdoc.predict_uvdoc(dewarper, tmp_path / "inset", root=tiny_root, inset=preset)
    assert inset["inset"] == inset["evidence"]["inset"] == predict.inset_record(preset)
    stored = prediction_record(tmp_path / "inset")
    assert report_evidence(stored)["inset"] == predict.inset_record(preset)
    for pred_type, folder in (("uwp_texture", "warped_textures"), ("uwp_img", "img")):
        source = np.asarray(Image.open(tiny_root / folder / "00007.png"))
        plain = np.asarray(Image.open(output / pred_type / "00007.png"))
        assert np.array_equal(plain, uvdoc.unwarp(source, grid, "flattener"))
        inset_image = np.asarray(Image.open(tmp_path / "inset" / pred_type / "00007.png"))
        assert np.array_equal(inset_image, uvdoc.unwarp(source, grid, "flattener", preset))
        assert not np.array_equal(inset_image, plain)
    with pytest.raises(predict.FlattenerError, match="inset"):
        uvdoc.predict_uvdoc(dewarper, tmp_path / "bad", root=tiny_root, inset="other")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(FileExistsError):
        uvdoc.predict_uvdoc(dewarper, output, root=tiny_root)
    (tiny_root / "img" / "00003.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        uvdoc.predict_uvdoc(dewarper, output, root=tiny_root, overwrite=True)


def _predictions(root, output):
    """Predictions: the textures themselves, a few flipped."""
    (output / "uwp_texture").mkdir(parents=True)
    for index, sample in enumerate(uvdoc.SAMPLES):
        pixels = np.asarray(Image.open(root / "warped_textures" / f"{sample}.png"))
        if index % 7 == 0:
            pixels = pixels[::-1]
        Image.fromarray(pixels).save(output / "uwp_texture" / f"{sample}.png")
    return output


def test_evaluate_scores_all_samples_against_the_flat_textures(tiny_root, tmp_path):
    predictions = _predictions(tiny_root, tmp_path / "pred")
    report = uvdoc.evaluate_uvdoc(predictions, build=None, root=tiny_root, workers=1)
    rows = report["rows"]
    assert [row["id"] for row in rows] == list(uvdoc.SAMPLES)
    reference = metrics.read_image(str(tiny_root / "texture_sample" / "00014.png"))
    expected = metrics.score_pair(
        metrics.read_image(str(predictions / "uwp_texture/00014.png")), reference
    )
    assert rows[14]["ms_ssim"] == expected.ms_ssim and rows[14]["ld"] is None
    summary = report["summary"]
    assert summary["n"] == 50 and summary["ad"] is None
    assert summary["ms_ssim"] == pytest.approx(np.mean([row["ms_ssim"] for row in rows]))
    assert summary["prediction_failures"] is None  # no predictions.json
    assert report["comparable"] is False and report["config"]["pred_type"] == "uwp_texture"
    assert report["evidence"]["measurement"] == "uvdoc_benchmark"
    assert report["published"]["values"]["uvdoc"] == {"ms_ssim": 0.785, "ad": 0.119, "ld": 6.810}
    with pytest.raises(FileNotFoundError, match="uwp_img"):
        uvdoc.evaluate_uvdoc(predictions, build=None, root=tiny_root, pred_type="uwp_img")
    (predictions / "uwp_texture" / "00049.png").unlink()
    with pytest.raises(FileNotFoundError, match="1 predictions missing"):
        uvdoc.evaluate_uvdoc(predictions, build=None, root=tiny_root)
    (tiny_root / "texture_sample" / "00000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        uvdoc.evaluate_uvdoc(predictions, build=None, root=tiny_root)


def test_cli_predicts_and_scores(tiny_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
    output, report_path = tmp_path / "out", tmp_path / "uvdoc.json"
    weights = str(_tiny_weights(tmp_path))
    base = ["--root", str(tiny_root), "--quiet"]
    predict_command = ["uvdoc-predict", "--weights", weights, "--output-dir", str(output)]
    assert cli.main([*predict_command, *base]) == 0
    assert json.loads(capsys.readouterr().out)["written"] == 100
    command = [
        "uvdoc", str(output), "--output", str(report_path), "--ms-ssim-only", "--workers", "1",
        *base,
    ]  # fmt: skip
    assert cli.main(command) == 0
    report = json.loads(report_path.read_text())
    assert report["summary"]["prediction_failures"] == 0 and report["summary"]["n"] == 50
    assert "ms_ssim=" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exists"):
        cli.main(command)


def test_process_pool_runs_a_custom_score_function(tmp_path):
    """docunet.run_tasks spawns workers that call the given module-level score function."""
    from flattener.benchmark import docunet

    rng = np.random.default_rng(9)
    tasks = []
    for sample in uvdoc.SAMPLES[:2]:
        for kind in ("ref", "pred"):
            pixels = rng.integers(0, 256, (12, 9, 3), np.uint8)
            Image.fromarray(pixels).save(tmp_path / f"{kind}{sample}.png")
        tasks.append(
            uvdoc.SampleTask(
                sample,
                str(tmp_path / f"ref{sample}.png"),
                str(tmp_path / f"pred{sample}.png"),
                metrics.DEFAULT_AD_SOLVER,
            )
        )
    seen = []
    pooled = docunet.run_tasks(
        tasks,
        None,
        workers=2,
        score=uvdoc.score_sample,
        total=2,
        progress=lambda row, done, total: seen.append((done, total)),
    )
    serial = docunet.run_tasks(tasks, None, workers=1, score=uvdoc.score_sample, total=2)
    key = "ms_ssim"
    assert sorted(row[key] for row in pooled) == sorted(row[key] for row in serial)
    assert seen == [(1, 2), (2, 2)]
