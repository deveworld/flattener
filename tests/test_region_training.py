"""Analytic region losses, genuine CPU fitting, determinism, and split isolation."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flattener import FlattenerError
from flattener.regions.segmentation import SegmenterConfig, load_segmenter, read_checkpoint
from flattener.regions.training import (
    RegionTrainConfig,
    segmentation_loss,
    train_regions,
    validation_metrics,
)
from flattener.synthetic import scenes

pytestmark = pytest.mark.training


def test_zero_logits_loss_matches_analytic_weighted_bce_plus_dice():
    logits = torch.zeros(2, 1, 2, 2)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0] = 1
    positive_dice = 1 - (2 * 1 + 1) / (2 + 2 + 1)
    negative_dice = 1 - 1 / (2 + 1)
    expected = math.log(2) + (positive_dice + 2 * negative_dice) / 3
    assert float(segmentation_loss(logits, targets)) == pytest.approx(expected, abs=1e-7)


def test_all_background_loss_is_finite_and_penalizes_false_probabilities():
    logits = torch.zeros(2, 1, 3, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = segmentation_loss(logits, target)
    loss.backward()
    assert bool(torch.isfinite(logits.grad).all()) and bool((logits.grad > 0).all())
    assert segmentation_loss(torch.full_like(logits, -12), target) < 1e-3
    assert segmentation_loss(torch.full_like(logits, 12), target) > loss


def test_all_foreground_loss_is_defined_and_improves_with_correct_probability():
    target = torch.ones(2, 1, 3, 4)
    assert segmentation_loss(torch.full_like(target, 12), target) < 1e-3
    assert segmentation_loss(torch.full_like(target, -12), target) > 10


@pytest.mark.parametrize("kind", ["nonfinite", "nonbinary", "double", "shape", "empty"])
def test_loss_rejects_invalid_batch(kind):
    logits, targets = torch.zeros(2, 1, 3, 4), torch.zeros(2, 1, 3, 4)
    if kind == "nonfinite":
        logits[0, 0, 0, 0] = float("inf")
    elif kind == "nonbinary":
        targets[0, 0, 0, 0] = 0.5
    elif kind == "double":
        targets = targets.double()
    elif kind == "shape":
        targets = targets[:, :, :2]
    elif kind == "empty":
        logits, targets = logits[:0], targets[:0]
    with pytest.raises(FlattenerError):
        segmentation_loss(logits, targets)


def test_validation_separates_positive_iou_and_negative_false_positives():
    target = torch.zeros(3, 1, 2, 2)
    target[0, 0, 0] = 1
    logits = torch.full_like(target, -1)
    logits[0, 0, 0, 0] = 1
    logits[1, 0, 0, 0] = 1
    metrics = validation_metrics(logits, target)
    assert metrics["foreground_iou"] == 0.5
    assert metrics["negative_pixel_false_positive_rate"] == 0.125
    assert metrics["negative_image_false_positive_rate"] == 0.5
    assert metrics["positive_images"] == 1 and metrics["negative_images"] == 2
    assert metrics["loss"] == float(segmentation_loss(logits, target))


@pytest.mark.parametrize("present", [True, False])
def test_validation_requires_positive_and_negative_examples(present):
    with pytest.raises(FlattenerError):
        validation_metrics(torch.zeros(2, 1, 2, 2), torch.full((2, 1, 2, 2), float(present)))


@pytest.fixture
def toy_dataset(tmp_path, monkeypatch):
    """Independent visible rectangle/no-page scenes, including a forbidden test set."""
    directory = tmp_path / "toy-scenes"
    directory.mkdir()
    records = [
        {"id": f"{split}-{i}", "split": split, "has_page": i % 2 == 0}
        for split in ("train", "validation", "test")
        for i in range(4)
    ]
    manifest = {"samples": records}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    loaded = []

    def load_scene(path, record):
        assert path == directory.resolve()
        assert record["split"] != "test", "test samples must never be loaded by training"
        loaded.append(record["id"])
        image = np.full((16, 16, 3), 20, np.uint8)
        mask = np.zeros((16, 16), bool)
        if record["has_page"]:
            mask[3:13, 4:12] = True
            image[mask] = 240
        return SimpleNamespace(image=image, mask=mask)

    monkeypatch.setattr(scenes, "load_scene", load_scene)
    monkeypatch.setattr(
        scenes,
        "validate_scene_dataset",
        lambda path: {
            "manifest": manifest,
            "sample_counts": {"train": 4, "validation": 4, "test": 4},
        },
    )
    return directory, loaded


@pytest.fixture
def config():
    return RegionTrainConfig(
        seed=71,
        epochs=5,
        batch_size=4,
        learning_rate=0.01,
        threads=1,
        model_config=SegmenterConfig(input_size=(16, 16), channels=(4, 8, 8)),
    )


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            assert_equal(first, second)
    else:
        assert actual == expected


def test_small_actual_learning_and_selection_protocol(tmp_path, toy_dataset, config):
    output = tmp_path / "fit"
    report = train_regions(toy_dataset[0], output, config)
    assert report["best_validation"]["loss"] < report["initial_validation"]["loss"]
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"]
    losses = [report["initial_validation"]["loss"]] + [
        row["validation"]["loss"] for row in report["history"]
    ]
    assert report["best_epoch"] == losses.index(min(losses))
    assert report["best_validation"]["loss"] == min(losses)
    assert len(toy_dataset[1]) == 8
    assert all(not record.startswith("test") for record in toy_dataset[1])
    best = read_checkpoint(output / "best.pt")
    assert best["epoch"] == report["best_epoch"]
    assert load_segmenter(output / "best.pt").config == config.model_config
    assert report["provenance"]["selection"]["direction"] == "minimize"
    assert report["provenance"]["loss"]["negative_sample_weight"] == 2


def test_same_seed_and_epoch_resume_are_exact(tmp_path, toy_dataset, config):
    paths = [tmp_path / name for name in ("first", "repeat", "resume")]
    train_regions(toy_dataset[0], paths[0], config)
    train_regions(toy_dataset[0], paths[1], config)
    train_regions(toy_dataset[0], paths[2], replace(config, epochs=2))
    train_regions(toy_dataset[0], paths[2], config, resume=paths[2] / "last.pt")
    expected = read_checkpoint(paths[0] / "last.pt")
    for path in paths[1:]:
        actual = read_checkpoint(path / "last.pt")
        for key in (
            "model_state",
            "optimizer_state",
            "history",
            "torch_rng_state",
            "shuffle_rng_state",
            "initial_validation",
            "best_validation",
            "best_epoch",
        ):
            assert_equal(actual[key], expected[key])


def test_training_restores_rng_threads_and_determinism(tmp_path, toy_dataset, config):
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    train_regions(toy_dataset[0], tmp_path / "restored", replace(config, epochs=1))
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_training_restores_global_state_when_loader_raises(
    tmp_path, toy_dataset, config, monkeypatch
):
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def broken_loader(*args):
        raise RuntimeError("deliberate loading failure")

    monkeypatch.setattr(scenes, "load_scene", broken_loader)
    output = tmp_path / "never-created"
    with pytest.raises(RuntimeError, match="deliberate"):
        train_regions(toy_dataset[0], output, config)
    assert not output.exists()
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_resume_rejects_changed_manifest_and_active_lock(tmp_path, toy_dataset, config):
    output = tmp_path / "locked"
    train_regions(toy_dataset[0], output, replace(config, epochs=1))
    lock = output / ".training.lock"
    lock.write_text("active")
    with pytest.raises(FlattenerError, match="active training lock"):
        train_regions(toy_dataset[0], output, config, output / "last.pt")
    assert lock.read_text() == "active"
    lock.unlink()
    manifest = toy_dataset[0] / "manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(FlattenerError, match="dataset manifest differs"):
        train_regions(toy_dataset[0], output, config, output / "last.pt")


def test_training_rejects_missing_negatives_before_output(
    tmp_path, toy_dataset, config, monkeypatch
):
    original = scenes.load_scene

    def positive_only(path, record):
        return original(path, dict(record, has_page=True))

    monkeypatch.setattr(scenes, "load_scene", positive_only)
    output = tmp_path / "no-negatives"
    with pytest.raises(FlattenerError, match="needs page and no-page"):
        train_regions(toy_dataset[0], output, config)
    assert not output.exists()


def test_output_no_overwrite_and_resume_constraints(tmp_path, toy_dataset, config):
    output = tmp_path / "resume"
    train_regions(toy_dataset[0], output, replace(config, epochs=1))
    with pytest.raises(FlattenerError, match="must not exist"):
        train_regions(toy_dataset[0], output, config)
    with pytest.raises(FlattenerError, match="only the total epoch"):
        train_regions(toy_dataset[0], output, replace(config, seed=72), output / "last.pt")
    with pytest.raises(FlattenerError, match="output_dir/last.pt"):
        train_regions(toy_dataset[0], output, config, output / "best.pt")
    with pytest.raises(FlattenerError, match="epochs must exceed"):
        train_regions(toy_dataset[0], output, replace(config, epochs=1), output / "last.pt")


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"epochs": 0},
        {"threads": True},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"model_config": None},
    ],
)
def test_bad_training_config_is_rejected_before_output(tmp_path, toy_dataset, config, change):
    output = tmp_path / "bad"
    with pytest.raises(FlattenerError):
        train_regions(toy_dataset[0], output, replace(config, **change))
    assert not output.exists()


def test_real_scene_generator_loader_integration(tmp_path):
    dataset = tmp_path / "real-scenes"
    scenes.generate_scene_dataset(
        dataset,
        scenes.SceneConfig(
            seed=927,
            train_documents=1,
            validation_documents=1,
            test_documents=1,
            variants_per_document=2,
            negative_fraction=0.5,
            image_size=(128, 128),
        ),
    )
    config = RegionTrainConfig(
        epochs=1,
        batch_size=2,
        model_config=SegmenterConfig(input_size=(16, 16), channels=(2, 4, 4)),
    )
    report = train_regions(dataset, tmp_path / "real-fit", config)
    assert report["dataset_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert math.isfinite(report["last_validation"]["loss"])
