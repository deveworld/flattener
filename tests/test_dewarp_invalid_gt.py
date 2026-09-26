"""Codex review F2: validation samples without valid ground truth cannot be evaluated.

A sample whose target grid has no valid node has no endpoint error. Validation
leaves it out of every mean, counts it per view as ``not_evaluable`` and a view
where no sample is evaluable has no EPE, so it never becomes the best checkpoint.
"""

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from flattener.dewarp.data import SAMPLE_SCHEMA, SPEC_SCHEMA
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.train import TrainConfig, _Logger, improves, train, validate

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)


def _config(**changes):
    values = dict(data_spec="s", output_dir="o", track="research", model=TINY, batch_size=2)
    values.update(changes)
    return TrainConfig(**values)


def _batch(valid_samples, sources):
    """One collated evaluation batch; ``valid_samples[i]`` says whether sample i has GT."""
    views = 2  # DEFAULT_VIEWS
    valid = torch.zeros(len(sources), views, 1, *TINY.grid_size)
    for index, has_target in enumerate(valid_samples):
        valid[index] = float(has_target)
    image = torch.stack(  # sample i looks the same in every batch
        [
            torch.randint(0, 256, (views, 3, *TINY.input_size), generator=_seeded(index))
            for index in range(len(sources))
        ]
    ).to(torch.uint8)
    return {
        "image": image,
        "grid2d": torch.zeros(len(sources), views, 2, *TINY.grid_size),
        "valid": valid,
        "source": list(sources),
    }


def _seeded(seed):
    return torch.Generator().manual_seed(seed)


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _model():
    torch.manual_seed(0)
    return UVDocNet(TINY)


def test_validation_without_valid_ground_truth_has_no_epe_and_no_best():
    config = _config()
    metrics = validate(_model(), [_batch([False, False], ["a", "a"])], config, torch.device("cpu"))
    assert _missing(metrics["epe_px"])
    for name, view in metrics["views"].items():
        assert _missing(view["epe_px"]), name
        assert view["not_evaluable"] == 2, name
        assert _missing(view["per_source_epe_px"]["a"]), name
    assert not improves(None, metrics["epe_px"])


def test_samples_without_valid_ground_truth_are_counted_and_left_out():
    config = _config()
    model = _model()
    device = torch.device("cpu")
    evaluable = validate(model, [_batch([True], ["a"])], config, device)
    mixed = validate(model, [_batch([True, False], ["a", "b"])], config, device)
    for name, view in mixed["views"].items():
        assert view["epe_px"] == pytest.approx(evaluable["views"][name]["epe_px"]), name
        assert view["not_evaluable"] == 1, name
        assert view["per_source_epe_px"]["a"] == pytest.approx(view["epe_px"]), name
        assert _missing(view["per_source_epe_px"]["b"]), name  # source b has no valid GT


def _spec_with_unlabelled_val(tmp_path):
    """Four train samples with maps and two val samples whose maps are all NaN."""
    root = tmp_path / "data"
    root.mkdir()
    base = np.stack(np.meshgrid(np.linspace(-0.7, 0.7, 12), np.linspace(-0.8, 0.8, 18)), -1)
    rng = np.random.default_rng(0)
    lines = []
    for index in range(6):
        split = "val" if index >= 4 else "train"
        Image.fromarray(rng.integers(0, 256, (60, 44, 3), dtype=np.uint8)).save(
            root / f"{index}.png"
        )
        values = base if split == "train" else np.full_like(base, np.nan)
        np.save(root / f"{index}.npy", values.astype(np.float16))
        entry = {
            "schema": SAMPLE_SCHEMA,
            "sample_id": str(index),
            "split": split,
            "image": f"{index}.png",
            "backward_map": f"{index}.npy",
            "track": "release",
            "license_class": "clean",
        }
        lines.append(json.dumps(entry) + "\n")
    (root / "manifest.jsonl").write_text("".join(lines))
    sources = [
        {
            "name": f"own-{split}",
            "kind": "converted",
            "root": "data",
            "track": "release",
            "license": "Apache-2.0",
            "split": split,
        }
        for split in ("train", "val")
    ]
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps({"schema": SPEC_SCHEMA, "name": "tiny", "track": "release", "sources": sources})
    )
    return path


@pytest.mark.training
def test_training_with_unevaluable_validation_selects_no_best(tmp_path):
    spec = _spec_with_unlabelled_val(tmp_path)
    config = _config(
        data_spec=str(spec),
        output_dir=str(tmp_path / "run"),
        track="release",
        total_steps=2,
        warmup_steps=1,
        num_workers=0,
        device="cpu",
        log_every=1,
        val_every=1,
        checkpoint_every=0,
    )
    result = train(config)
    assert result["state"] == "completed"
    assert result["best"] is None
    assert not (tmp_path / "run/checkpoints/best.json").exists()
    log = (tmp_path / "run/train_log.jsonl").read_text().splitlines()
    validations = [json.loads(line) for line in log if json.loads(line)["kind"] == "val"]
    assert validations and all(record["not_evaluable"] == 2 for record in validations)


def test_the_csv_log_keeps_the_header_it_was_started_with(tmp_path):
    (tmp_path / "train_log.csv").write_text("kind,step,epe_px\n")
    log = _Logger(tmp_path, "release")
    log({"kind": "val", "step": 3, "epe_px": None, "not_evaluable": 2})
    lines = (tmp_path / "train_log.csv").read_text().splitlines()
    assert lines == ["kind,step,epe_px", "val,3,"]
    record = json.loads((tmp_path / "train_log.jsonl").read_text())
    assert record["epe_px"] is None and record["not_evaluable"] == 2
