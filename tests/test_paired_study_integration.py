"""Small, independent scene-0.3 routing and training regressions."""

import hashlib
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener.contracts import FlattenerError
from flattener.regions import experiment_training as training
from flattener.regions.checkpoints import read_region_checkpoint
from flattener.regions.experiment_training import StudyTrainConfig, train_region_study
from flattener.synthetic import paired_scenes, scene_dataset
from flattener.synthetic.paired_scenes import PairedSceneConfig, generate_paired_scene_dataset
from flattener.synthetic.scenes import SceneSample

pytestmark = pytest.mark.training


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
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
        "dtype": torch.get_default_dtype(),
    }


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-integration-data")
    outputs = {}
    config = PairedSceneConfig(
        seed=927,
        train_documents=1,
        validation_documents=1,
        test_documents=1,
        variants_per_document=4,
        image_size=(128, 128),
    )
    for mode in ("matched", "independent"):
        directory = root / mode
        manifest = generate_paired_scene_dataset(directory, replace(config, training_panels=mode))
        outputs[mode] = (directory, manifest)
    return outputs


@pytest.fixture(scope="module")
def trained(datasets, tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-integration-models")
    config = StudyTrainConfig(seed=1729, epochs=1, batch_size=3, threads=1, architecture="context")
    results = {}
    for mode, (directory, manifest) in datasets.items():
        forbidden = {
            (directory / relative).resolve()
            for record in manifest["samples"]
            if record["split"] == "test"
            for name, relative in record["paths"].items()
            if name.endswith(".png")
        }
        loaded, decoded = [], []
        original_factory, original_open = training.scene_loader, Image.open

        def loader_factory(root):
            implementation = original_factory(root)

            def load(path, record):
                assert record["split"] in ("train", "validation"), "held-out sample loaded"
                loaded.append((record["split"], record["id"]))
                return implementation(path, record)

            return load

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                resolved = Path(path).resolve()
                assert resolved not in forbidden, "test RGB/mask decoded during fitting"
                decoded.append(resolved)
            return original_open(path, *args, **kwargs)

        before = _runtime()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(training, "scene_loader", loader_factory)
            patch.setattr(Image, "open", guarded_open)
            report = train_region_study(directory, root / mode, config)
        _equal(_runtime(), before)
        results[mode] = {
            "output": root / mode,
            "report": report,
            "loaded": loaded,
            "decoded": decoded,
            "config": config,
        }
    return results


@pytest.mark.parametrize("mode", ["matched", "independent"])
def test_schema_three_router_validates_without_any_image_decoding(datasets, monkeypatch, mode):
    directory, manifest = datasets[mode]
    monkeypatch.setattr(
        Image, "open", lambda *a, **kw: pytest.fail("integrity check decoded image")
    )
    assert scene_dataset.scene_backend(directory) is paired_scenes
    checked = scene_dataset.validate_scene_dataset(directory)
    assert checked["manifest"]["schema_version"] == "scene-0.3"
    assert checked["sample_counts"] == {"train": 6, "validation": 6, "test": 6}
    assert checked["page_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert checked["negative_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert checked["manifest"]["samples"] == manifest["samples"]


def test_both_modes_preserve_validation_and_test_artifacts_and_positive_training_pixels(datasets):
    matched, independent = (datasets[name][1] for name in ("matched", "independent"))
    left = {record["id"]: record for record in matched["samples"]}
    right = {record["id"]: record for record in independent["samples"]}
    assert left.keys() == right.keys()
    changed_images = []
    for identifier, record in left.items():
        other = right[identifier]
        assert record["sha256"]["mask.png"] == other["sha256"]["mask.png"]
        if record["split"] != "train":
            assert record == other
        elif record["sha256"]["image.png"] != other["sha256"]["image.png"]:
            assert not record["has_page"] and record["negative_type"] == "paper_panel"
            changed_images.append(identifier)
    assert (
        len(changed_images) == 1
    )  # One toy source document, rather than the full 96-document run.


@pytest.mark.parametrize("mode", ["matched", "independent"])
def test_router_returns_scene_sample_with_blank_carrier_separate_from_printed_gt(datasets, mode):
    directory, manifest = datasets[mode]
    loader = scene_dataset.scene_loader(directory)
    for record in (record for record in manifest["samples"] if record["split"] == "validation"):
        sample = loader(directory, record)
        assert isinstance(sample, SceneSample)
        assert sample.image.dtype == np.uint8 and sample.image.shape == (128, 128, 3)
        assert sample.mask.dtype == np.bool_ and sample.mask.shape == (128, 128)
        assert bool(sample.mask.any()) == record["has_page"]
        if record["has_page"]:
            assert sample.metadata["geometry"] == sample.metadata["physical_page"]["geometry"]
        else:
            assert sample.metadata["geometry"] is None
            if record["negative_type"] == "paper_panel":
                assert sample.metadata["physical_page"]["geometry"] is not None
                assert sample.metadata["physical_page"]["foreground_fraction"] > 0
                assert sample.metadata["panel"]["is_target"] is False


@pytest.mark.parametrize("mode", ["matched", "independent"])
def test_context_one_epoch_records_schema_three_and_only_decodes_train_validation(
    datasets, trained, mode
):
    directory, manifest = datasets[mode]
    result = trained[mode]
    report = result["report"]
    checkpoint = read_region_checkpoint(result["output"] / "last.pt")
    assert checkpoint["epoch"] == report["epochs_completed"] == 1
    assert checkpoint["config"] == asdict(result["config"])
    assert checkpoint["model_spec"]["architecture"] == "context"
    assert checkpoint["dataset_counts"] == {"train": 6, "validation": 6, "test": 6}
    assert (
        report["manifest_sha256"]
        == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    )
    provenance = checkpoint["provenance"]
    assert provenance["dataset_schema"] == "scene-0.3"
    source = Path(paired_scenes.__file__)
    assert (
        provenance["source_code_sha256"]["synthetic/paired_scenes.py"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert provenance["loss"]["negative_sample_weight"] == 2
    assert provenance["loss"]["dice_smoothing_pixels"] == 1
    expected = {(row["split"], row["id"]) for row in manifest["samples"] if row["split"] != "test"}
    assert len(result["loaded"]) == 12 and set(result["loaded"]) == expected
    assert len(result["decoded"]) == 24  # RGB and GT mask for each selected sample.
    assert checkpoint["initial_validation"]["positive_images"] == 4
    assert checkpoint["initial_validation"]["negative_images"] == 2
    assert checkpoint["optimizer_state"]["state"]
    assert not (result["output"] / ".training.lock").exists()


def test_same_seed_and_identical_validation_produce_identical_initial_metrics_across_modes(trained):
    first, second = (trained[mode]["report"] for mode in ("matched", "independent"))
    assert first["initial_validation"] == second["initial_validation"]
    assert first["manifest_sha256"] != second["manifest_sha256"]
    assert first["history"][0]["train_loss"] != second["history"][0]["train_loss"]


def test_resume_refuses_changed_paired_generator_provenance_before_training(
    datasets, trained, tmp_path, monkeypatch
):
    output = tmp_path / "resume"
    shutil.copytree(trained["matched"]["output"], output)
    original = training._provenance

    def changed_source():
        provenance = original()
        provenance["source_code_sha256"]["synthetic/paired_scenes.py"] = "0" * 64
        return provenance

    monkeypatch.setattr(training, "_provenance", changed_source)
    monkeypatch.setattr(
        training, "_splits", lambda *a, **kw: pytest.fail("invalid resume decoded data")
    )
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    runtime = _runtime()
    with pytest.raises(FlattenerError) as caught:
        train_region_study(
            datasets["matched"][0],
            output,
            replace(trained["matched"]["config"], epochs=2),
            resume=output / "last.pt",
        )
    assert caught.value.reason_code == "resume_environment_mismatch"
    assert "paired_scenes.py" in caught.value.message
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    _equal(_runtime(), runtime)
