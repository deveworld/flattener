"""Independent CPU study fitting, continuation, state isolation and rejection tests."""

import hashlib
import math
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError
from flattener.regions import experiment_training as study
from flattener.regions.checkpoints import read_region_checkpoint
from flattener.regions.experiment_training import StudyTrainConfig, train_region_study
from flattener.synthetic.scenes import SceneConfig, generate_scene_dataset

pytestmark = pytest.mark.training


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _equal(left, right)
    else:
        assert actual == expected


def _runtime():
    return {
        "torch_rng": torch.get_rng_state().clone(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "default_dtype": torch.get_default_dtype(),
    }


def _hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _rewrite_checkpoint(path, value):
    """Rehash deliberate semantic corruptions so tests reach checkpoint validation."""
    torch.save(value, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )


@pytest.fixture(scope="module")
def scene_v1(tmp_path_factory):
    directory = tmp_path_factory.mktemp("region-study-data") / "scenes"
    manifest = generate_scene_dataset(
        directory,
        SceneConfig(
            seed=927,
            train_documents=1,
            validation_documents=1,
            test_documents=1,
            variants_per_document=2,
            negative_fraction=0.5,
            image_size=(128, 128),
        ),
    )
    return directory, manifest


@pytest.fixture(scope="module")
def base_config():
    return StudyTrainConfig(seed=391, epochs=1, batch_size=3, learning_rate=0.001, threads=1)


@pytest.fixture(scope="module")
def one_epoch(scene_v1, base_config, tmp_path_factory):
    directory = tmp_path_factory.mktemp("region-study-checkpoints")
    result = {}
    for architecture in ("local", "context"):
        output = directory / architecture
        report = train_region_study(
            scene_v1[0], output, replace(base_config, architecture=architecture)
        )
        result[architecture] = (output, report)
    return result


@pytest.fixture
def copied_local(one_epoch, tmp_path):
    output = tmp_path / "resumed-study"
    shutil.copytree(one_epoch["local"][0], output)
    return output


@pytest.mark.parametrize("architecture", ["local", "context"])
def test_real_scene_one_epoch_has_explicit_architecture_and_historical_objective(
    one_epoch, architecture, scene_v1, base_config
):
    output, report = one_epoch[architecture]
    saved = read_region_checkpoint(output / "last.pt")
    best = read_region_checkpoint(output / "best.pt")
    assert saved["schema_version"] == report["schema_version"] == "region-study-0.2"
    assert saved["model_spec"]["architecture"] == architecture
    assert saved["model_spec"]["model_config"]["input_size"] == (128, 96)
    assert saved["config"] == asdict(replace(base_config, architecture=architecture))
    assert report["epochs_completed"] == saved["epoch"] == 1
    assert len(saved["history"]) == 1
    assert saved["dataset_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert (
        saved["manifest_sha256"]
        == hashlib.sha256((scene_v1[0] / "manifest.json").read_bytes()).hexdigest()
    )
    assert saved["provenance"]["dataset_schema"] == "scene-0.1"
    loss_protocol = saved["provenance"]["loss"]
    assert loss_protocol["bce_weight"] == loss_protocol["dice_weight"] == 1
    assert loss_protocol["negative_sample_weight"] == 2
    assert loss_protocol["dice_smoothing_pixels"] == 1
    assert saved["provenance"]["selection"]["direction"] == "minimize"
    assert saved["provenance"]["selection"]["threshold"] == 0.5
    metrics = [saved["initial_validation"], saved["history"][0]["validation"]]
    index = min(range(2), key=lambda position: metrics[position]["loss"])
    assert saved["best_epoch"] == best["epoch"] == index
    assert saved["best_validation"] == best["best_validation"] == metrics[index]
    for row in metrics:
        assert row["positive_images"] == row["negative_images"] == 2
        assert math.isfinite(row["loss"])
    assert math.isfinite(saved["history"][0]["train_loss"])
    assert saved["optimizer_state"]["state"]
    assert all(float(value["step"]) == 2 for value in saved["optimizer_state"]["state"].values())
    for name in ("last.pt", "best.pt"):
        assert (
            report["checkpoints"][name] == hashlib.sha256((output / name).read_bytes()).hexdigest()
        )
        assert (output / (name + ".sha256")).read_text().strip() == report["checkpoints"][name]
    assert not (output / ".training.lock").exists()


def test_three_epochs_repeat_and_one_plus_resume_are_exact(
    scene_v1, base_config, copied_local, tmp_path
):
    full = replace(base_config, epochs=3)
    first, repeated = tmp_path / "uninterrupted", tmp_path / "repeated"
    train_region_study(scene_v1[0], first, full)
    train_region_study(scene_v1[0], repeated, full)
    before_resume = _runtime()
    train_region_study(scene_v1[0], copied_local, full, resume=copied_local / "last.pt")
    _equal(_runtime(), before_resume)
    expected = read_region_checkpoint(first / "last.pt")
    keys = (
        "model_state",
        "optimizer_state",
        "history",
        "torch_rng_state",
        "shuffle_rng_state",
        "initial_validation",
        "best_validation",
        "best_epoch",
        "epoch",
        "config",
        "model_spec",
    )
    for output in (repeated, copied_local):
        actual = read_region_checkpoint(output / "last.pt")
        for key in keys:
            _equal(actual[key], expected[key])
        actual_best = read_region_checkpoint(output / "best.pt")
        expected_best = read_region_checkpoint(first / "best.pt")
        for key in ("model_state", "optimizer_state", "epoch", "history", "best_validation"):
            _equal(actual_best[key], expected_best[key])


@pytest.mark.parametrize("architecture", ["local", "context"])
def test_each_architecture_uses_independently_computed_weighted_bce_dice(
    scene_v1, base_config, tmp_path, monkeypatch, architecture
):
    original = study.segmentation_loss
    batches = []

    def inspected_loss(logits, targets):
        scores = logits.detach().numpy().astype(np.float64)
        truth = targets.numpy().astype(np.float64)
        axes = (1, 2, 3)
        probability = 1 / (1 + np.exp(-scores))
        bce = (np.maximum(scores, 0) - scores * truth + np.log1p(np.exp(-np.abs(scores)))).mean(
            axes
        )
        target_pixels = truth.sum(axes)
        dice = 1 - (2 * (probability * truth).sum(axes) + 1) / (
            probability.sum(axes) + target_pixels + 1
        )
        weights = np.where(target_pixels == 0, 2.0, 1.0)
        expected = float(np.dot(bce + dice, weights) / weights.sum())
        actual = original(logits, targets)
        assert float(actual.detach()) == pytest.approx(expected, abs=1e-6)
        batches.append(
            (float(actual.detach()), float(weights.sum()), int((target_pixels == 0).sum()))
        )
        return actual

    monkeypatch.setattr(study, "segmentation_loss", inspected_loss)
    report = train_region_study(
        scene_v1[0], tmp_path / "objective", replace(base_config, architecture=architecture)
    )
    assert len(batches) == 2  # Unequal batch sizes: 3 then 1, with two negative images total.
    assert sum(batch[2] for batch in batches) == 2
    assert sum(batch[1] for batch in batches) == 6  # 2 positives + 2 doubled negatives.
    weighted_batch_mean = sum(loss * weight for loss, weight, _ in batches) / 6
    assert report["history"][0]["train_loss"] == pytest.approx(weighted_batch_mean, abs=1e-12)


def test_no_test_sample_or_mask_is_decoded(scene_v1, base_config, tmp_path, monkeypatch):
    directory, manifest = scene_v1
    loaded, opened = [], []
    original_factory, original_open = study.scene_loader, Image.open
    resolved = []
    forbidden = {
        (directory / relative).resolve()
        for record in manifest["samples"]
        if record["split"] == "test"
        for name, relative in record["paths"].items()
        if name.endswith(".png")
    }

    def factory(directory):
        resolved.append(directory)
        original_loader = original_factory(directory)

        def load(path, record):
            assert record["split"] != "test", "held-out sample decoded during learning"
            loaded.append(record["id"])
            return original_loader(path, record)

        return load

    def image_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            assert resolved not in forbidden, "held-out image or GT mask decoded"
            opened.append(resolved)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(study, "scene_loader", factory)
    monkeypatch.setattr(Image, "open", image_open)
    train_region_study(directory, tmp_path / "isolated", base_config)
    expected = [record["id"] for record in manifest["samples"] if record["split"] != "test"]
    assert sorted(loaded) == sorted(expected)
    assert resolved == [directory.resolve()]
    assert len(opened) == 16


@pytest.mark.parametrize("architecture", ["local", "context"])
def test_training_preserves_all_process_rng_and_torch_options(
    scene_v1, base_config, tmp_path, architecture
):
    before = _runtime()
    train_region_study(
        scene_v1[0], tmp_path / "runtime", replace(base_config, architecture=architecture)
    )
    _equal(_runtime(), before)


def test_loader_error_restores_runtime_before_output_exists(
    scene_v1, base_config, tmp_path, monkeypatch
):
    before = _runtime()
    output = tmp_path / "loader-failure"

    def broken(*args):
        raise RuntimeError("deliberate scene loading failure")

    monkeypatch.setattr(study, "scene_loader", lambda directory: broken)
    with pytest.raises(RuntimeError, match="deliberate"):
        train_region_study(scene_v1[0], output, base_config)
    _equal(_runtime(), before)
    assert not output.exists()


def test_nonfinite_training_restores_runtime_and_removes_owned_lock(
    scene_v1, base_config, tmp_path, monkeypatch
):
    original_loss = study.segmentation_loss
    before = _runtime()

    def nonfinite(logits, targets):
        return original_loss(logits, targets) * float("nan")

    monkeypatch.setattr(study, "segmentation_loss", nonfinite)
    output = tmp_path / "nonfinite"
    with pytest.raises(FlattenerError) as error:
        train_region_study(scene_v1[0], output, base_config)
    assert error.value.reason_code == "nonfinite_training"
    assert not (output / ".training.lock").exists()
    _equal(_runtime(), before)


def test_existing_output_is_untouched(scene_v1, base_config, copied_local):
    before = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(scene_v1[0], copied_local, base_config)
    assert error.value.reason_code == "training_output_exists"
    assert _hashes(copied_local) == before


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 392},
        {"batch_size": 2},
        {"learning_rate": 0.002},
        {"threads": 2},
        {"architecture": "context"},
    ],
)
def test_resume_rejects_any_changed_learning_configuration(
    scene_v1, base_config, copied_local, change
):
    before = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(
            scene_v1[0],
            copied_local,
            replace(base_config, epochs=3, **change),
            resume=copied_local / "last.pt",
        )
    assert error.value.reason_code == "resume_config_mismatch"
    assert _hashes(copied_local) == before


@pytest.mark.parametrize("path", ["best.pt", "elsewhere.pt"])
def test_resume_requires_output_last_path(scene_v1, base_config, copied_local, path):
    with pytest.raises(FlattenerError) as error:
        train_region_study(
            scene_v1[0], copied_local, replace(base_config, epochs=3), resume=copied_local / path
        )
    assert error.value.reason_code == "invalid_resume_path"


def test_resume_requires_larger_epoch_budget(scene_v1, base_config, copied_local):
    with pytest.raises(FlattenerError) as error:
        train_region_study(scene_v1[0], copied_local, base_config, resume=copied_local / "last.pt")
    assert error.value.reason_code == "resume_epoch_budget"


def test_active_resume_lock_is_not_removed(scene_v1, base_config, copied_local):
    lock = copied_local / ".training.lock"
    lock.write_text("another live trainer owns this lock")
    before = _runtime()
    hashes = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(
            scene_v1[0],
            copied_local,
            replace(base_config, epochs=3),
            resume=copied_local / "last.pt",
        )
    assert error.value.reason_code == "training_in_progress"
    assert _hashes(copied_local) == hashes
    _equal(_runtime(), before)


def test_resume_rejects_changed_valid_manifest(scene_v1, base_config, copied_local, tmp_path):
    dataset = tmp_path / "different-manifest"
    shutil.copytree(scene_v1[0], dataset)
    path = dataset / "manifest.json"
    path.write_text(path.read_text() + "\n")
    before = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(
            dataset, copied_local, replace(base_config, epochs=3), resume=copied_local / "last.pt"
        )
    assert error.value.reason_code == "resume_dataset_mismatch"
    assert _hashes(copied_local) == before


def test_corrupt_dataset_artifact_rejected_before_output(scene_v1, base_config, tmp_path):
    dataset = tmp_path / "corrupt-dataset"
    shutil.copytree(scene_v1[0], dataset)
    record = next(row for row in scene_v1[1]["samples"] if row["split"] == "train")
    image = dataset / record["paths"]["image.png"]
    image.write_bytes(image.read_bytes() + b"corrupted after generation")
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as error:
        train_region_study(dataset, output, base_config)
    assert error.value.reason_code == "artifact_hash_mismatch"
    assert not output.exists()


@pytest.mark.parametrize("filename", ["last.pt", "best.pt"])
def test_corrupt_checkpoint_hash_is_rejected(scene_v1, base_config, copied_local, filename):
    sidecar = copied_local / (filename + ".sha256")
    sidecar.write_text("0" * 64 + "\n")
    before = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(
            scene_v1[0],
            copied_local,
            replace(base_config, epochs=3),
            resume=copied_local / "last.pt",
        )
    assert error.value.reason_code == "checkpoint_hash_mismatch"
    assert _hashes(copied_local) == before


@pytest.mark.parametrize("corruption", ["nan_weight", "bad_rng", "bad_optimizer", "bad_history"])
def test_rehashed_corrupt_checkpoint_is_rejected(scene_v1, base_config, copied_local, corruption):
    path = copied_local / "last.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if corruption == "nan_weight":
        next(iter(state["model_state"].values())).view(-1)[0] = float("nan")
    elif corruption == "bad_rng":
        state["shuffle_rng_state"] = torch.zeros(3, dtype=torch.uint8)
    elif corruption == "bad_optimizer":
        next(iter(state["optimizer_state"]["state"].values()))["exp_avg"] = torch.zeros(1)
    else:
        state["history"][0]["epoch"] = 9
    _rewrite_checkpoint(path, state)
    before = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(scene_v1[0], copied_local, replace(base_config, epochs=3), resume=path)
    assert error.value.reason_code == "invalid_checkpoint"
    assert _hashes(copied_local) == before


def test_resume_rejects_changed_learner_source_provenance(scene_v1, base_config, copied_local):
    path = copied_local / "last.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["provenance"]["source_code_sha256"]["regions/experiment_training.py"] = "0" * 64
    _rewrite_checkpoint(path, state)
    with pytest.raises(FlattenerError) as error:
        train_region_study(scene_v1[0], copied_local, replace(base_config, epochs=3), resume=path)
    assert error.value.reason_code == "resume_environment_mismatch"


def test_resume_rejects_best_checkpoint_from_different_seed(scene_v1, base_config, copied_local):
    path = copied_local / "best.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["config"]["seed"] += 1
    _rewrite_checkpoint(path, state)
    before = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(
            scene_v1[0],
            copied_local,
            replace(base_config, epochs=2),
            resume=copied_local / "last.pt",
        )
    assert error.value.reason_code in {"invalid_checkpoint", "resume_config_mismatch"}
    assert _hashes(copied_local) == before


@pytest.mark.parametrize("change", ["learner_source", "loss"])
def test_resume_rejects_best_checkpoint_with_different_computation_provenance(
    scene_v1, base_config, copied_local, change
):
    path = copied_local / "best.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if change == "learner_source":
        state["provenance"]["source_code_sha256"]["regions/experiment_training.py"] = "0" * 64
    else:
        state["provenance"]["loss"]["negative_sample_weight"] = 3
    _rewrite_checkpoint(path, state)
    before = _hashes(copied_local)
    with pytest.raises(FlattenerError) as error:
        train_region_study(
            scene_v1[0],
            copied_local,
            replace(base_config, epochs=2),
            resume=copied_local / "last.pt",
        )
    assert error.value.reason_code in {"invalid_checkpoint", "resume_environment_mismatch"}
    assert _hashes(copied_local) == before


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"seed": -1},
        {"epochs": 0},
        {"epochs": 1.5},
        {"batch_size": 0},
        {"threads": 0},
        {"threads": 129},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": True},
        {"architecture": "unknown"},
        {"architecture": ["local"]},
    ],
)
def test_invalid_configuration_is_rejected_at_construction(change):
    with pytest.raises(FlattenerError) as error:
        StudyTrainConfig(**change)
    assert error.value.reason_code == "invalid_config"


def test_wrong_configuration_object_is_rejected_without_output(scene_v1, tmp_path):
    output = tmp_path / "wrong-config"
    with pytest.raises(FlattenerError) as error:
        train_region_study(scene_v1[0], output, {"architecture": "local"})
    assert error.value.reason_code == "invalid_config"
    assert not output.exists()
