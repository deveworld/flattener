"""DocUNet stress variants: inputs, predictions, scores and the summary."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from flattener.benchmark import __main__ as cli
from flattener.benchmark import stress
from flattener.dewarp import predict
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
CROP = (28, 20)


def _identity_grid(dewarper, image):
    height, width = dewarper.config.grid_size
    grid = np.meshgrid(edge_positions(width), edge_positions(height))
    return np.stack(grid, -1).astype(np.float32)


@pytest.fixture
def stress_inputs(tiny_docunet, tmp_path):
    """Tiny crops and photos for the 130 DocUNet images and a tiny 3-step checkpoint."""
    scans, _ = tiny_docunet
    crops, photos = tmp_path / "crops", tmp_path / "photos"
    crops.mkdir()
    photos.mkdir()
    rng = np.random.default_rng(1)
    for k in range(1, 66):
        for m in (1, 2):
            crop = rng.integers(0, 256, (*CROP, 3), np.uint8)
            Image.fromarray(crop).save(crops / f"{k}_{m} copy.png")
            photo = rng.integers(0, 256, (60, 45, 3), np.uint8)
            Image.fromarray(photo).save(photos / f"docunet-photo-{k}_{m}.jpg")
    torch.manual_seed(0)
    weights = tmp_path / "model.pt"
    save_checkpoint(
        weights,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": 3,
            "track": "release",
            "model_config": TINY.to_dict(),
            "lineage": [{"kind": "scratch"}],
            "data_spec_sha256": "0" * 64,
            "code": {"sha256": "1" * 64},
            "model_state": UVDocNet(TINY).state_dict(),
        },
    )
    return scans, crops, photos, weights


def test_variants_turn_the_crop_and_shrink_it_onto_a_canvas(stress_inputs):
    _, crops, photos, _ = stress_inputs
    with Image.open(crops / "3_1 copy.png") as opened:
        crop = np.asarray(opened)
    image = stress.variant_image
    np.testing.assert_array_equal(image("crop", 3, 1, crops, photos), crop)
    np.testing.assert_array_equal(image("rot90", 3, 1, crops, photos), np.rot90(crop))
    np.testing.assert_array_equal(image("rot270", 3, 1, crops, photos), np.rot90(crop, -1))
    assert image("full_frame", 3, 1, crops, photos).shape == (60, 45, 3)
    small = image("small_page", 3, 1, crops, photos)
    assert 0.09 < crop.shape[0] * crop.shape[1] / (small.shape[0] * small.shape[1]) < 0.27
    np.testing.assert_array_equal(small, image("small_page", 3, 1, crops, photos))
    windows = np.lib.stride_tricks.sliding_window_view(small, crop.shape)[..., 0, :, :, :]
    assert (windows == crop).all(axis=(-3, -2, -1)).sum() == 1  # the crop, pasted once
    with pytest.raises(ValueError):
        image("rot45", 3, 1, crops, photos)


def _argv(stress_inputs, output, *extra):
    scans, crops, photos, weights = stress_inputs
    return [
        "docunet-stress",
        "--weights", str(weights), "--output-dir", str(output),
        "--crop-dir", str(crops), "--photo-dir", str(photos),
        "--references-dir", str(scans), "--no-verify-references",
        "--ms-ssim-only", "--workers", "1", "--quiet", "--convention", "128_excl64",
        *extra,
    ]  # fmt: skip


def test_stress_run_scores_every_variant_and_labels_it(stress_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
    output = tmp_path / "stress"
    assert cli.main(_argv(stress_inputs, output)) == 0
    summary = json.loads((output / "stress.json").read_text())
    assert summary["evidence"]["measurement"] == "docunet_stress"
    assert summary["evidence"]["evidence_kind"] == "smoke_training"  # a 3-step checkpoint
    assert set(summary["variants"]) == set(stress.VARIANTS)  # the JSON keys are sorted
    for name, entry in summary["variants"].items():
        assert entry["status"] == "scored" and entry["predictions"]["failed"] == 0, name
        assert entry["summary"]["128_excl64"]["n"] == 128
        assert entry["summary"]["128_excl64"]["prediction_failures"] == 0
        assert entry["evidence"]["measurement"] == "docunet_stress"
        record = json.loads((output / name / "predictions.json").read_text())
        assert record["inputs"] == f"docunet_stress:{name}"
        with Image.open(output / name / "7_2.png") as written:
            assert written.size == CROP[::-1]  # rendered at the upright crop's size
    before = {name: entry["report_sha256"] for name, entry in summary["variants"].items()}
    assert cli.main(_argv(stress_inputs, output, "--variant", "rot90")) == 0
    again = json.loads((output / "stress.json").read_text())["variants"]["rot90"]
    assert again["predictions"]["kept"] == 130 and again["report_sha256"] == before["rot90"]


def test_failed_predictions_leave_a_variant_unscored_unless_forced(stress_inputs, tmp_path):
    output = tmp_path / "stress"  # the random tiny model folds every map
    assert cli.main(_argv(stress_inputs, output, "--variant", "rot90")) == 1
    entry = json.loads((output / "stress.json").read_text())["variants"]["rot90"]
    assert entry["status"] == "not_scored" and entry["predictions"]["failed"] == 130
    assert "failed checks" in entry["error"]
    assert cli.main(_argv(stress_inputs, output, "--variant", "rot90", "--force-render")) == 1
    entry = json.loads((output / "stress.json").read_text())["variants"]["rot90"]
    assert entry["status"] == "scored" and entry["predictions"]["forced"] == 130
    assert entry["summary"]["128_excl64"]["prediction_failures"] == 128
    assert len(entry["failed_ids"]) == 130
