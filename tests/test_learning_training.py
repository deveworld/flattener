"""CPU training must be repeatable, resumable, and isolated from test data."""

import hashlib
import importlib
import json
import shutil
from dataclasses import replace

import pytest
import torch

from flattener import FlattenerError
from flattener.learning import coordinate_study as study
from flattener.learning import training
from flattener.learning.model import ModelConfig
from flattener.learning.training import TrainConfig, coordinate_loss, train
from flattener.synthetic.paired_scenes import PairedSceneConfig, generate_paired_scene_dataset
from flattener.synthetic.pipeline import DatasetConfig, PhotometricConfig, generate_dataset

pytestmark = pytest.mark.training


@pytest.fixture(scope="module")
def coordinate_training_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("coordinate-training") / "scenes"
    generate_paired_scene_dataset(
        directory,
        PairedSceneConfig(
            seed=509,
            train_documents=1,
            validation_documents=1,
            test_documents=1,
            image_size=(128, 128),
        ),
    )
    return directory


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_coordinate_cpu_resume_matches_uninterrupted_training(
    coordinate_training_data, tmp_path, monkeypatch, architecture
):
    config = study.StudyConfig(
        device="cpu",
        architecture=architecture,
        epochs=2,
        batch_size=2,
        output_size=(32, 32),
        page_weight=5.0 if architecture == "page" else 0.0,
        boundary_weight=0.5 if architecture == "page" else 0.0,
        differential_weight=0.2 if architecture == "page" else 0.0,
        topology_weight=50.0 if architecture == "page" else 0.0,
        gradient_clip_norm=5.0 if architecture == "page" else 0.0,
        model_config=ModelConfig(
            input_size=(16, 16), grid_size=(5, 5), channels=(2, 4, 4), hidden=8
        ),
    )
    source = study.SceneCoordinateDataset
    opened_splits = []

    def training_data_only(directory, split, *args, **kwargs):
        assert split in ("train", "validation")
        opened_splits.append(split)
        return source(directory, split, *args, **kwargs)

    def no_cuda(*args, **kwargs):
        pytest.fail("explicit CPU training attempted CUDA access")

    # Adam lazily registers torch functions; initialize those before replacing APIs.
    importlib.import_module("torch._dynamo")
    monkeypatch.setattr(study, "SceneCoordinateDataset", training_data_only)
    for name in ("is_available", "manual_seed", "get_rng_state", "set_rng_state", "device"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    uninterrupted = tmp_path / "full"
    staged = tmp_path / "staged"
    full = study.train_coordinate_study(coordinate_training_data, uninterrupted, config)
    short = study.train_coordinate_study(
        coordinate_training_data, staged, config, stop_after_epoch=1
    )
    assert not short["complete"] and short["epochs_completed"] == 1
    assert short["stop_reason"] == "stop_after_epoch"
    resumed = study.train_coordinate_study(
        coordinate_training_data,
        staged,
        config,
        resume=staged / "last.pt",
    )
    assert full["complete"] and resumed["complete"] and resumed["epochs_completed"] == 2
    expected = study.coordinate_checkpoint(uninterrupted / "last.pt")
    actual = study.coordinate_checkpoint(staged / "last.pt")
    for key in (
        "model_state",
        "optimizer_state",
        "history",
        "epoch",
        "best_epoch",
        "best_validation_epe_pixels",
        "torch_rng_state",
        "shuffle_rng_state",
        "cuda_rng_state",
    ):
        _assert_equal(actual[key], expected[key])
    assert actual["cuda_rng_state"] is None and actual["device_info"]["device"] == "cpu"
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert opened_splits == ["train", "validation"] * 3


def test_coordinate_cpu_budget_keeps_last_complete_epoch(
    coordinate_training_data, tmp_path, monkeypatch
):
    config = study.StudyConfig(
        device="cpu",
        architecture="spatial",
        epochs=2,
        batch_size=2,
        output_size=(32, 32),
        model_config=ModelConfig(
            input_size=(16, 16), grid_size=(5, 5), channels=(2, 4, 4), hidden=8
        ),
    )
    clock = [0.0]
    monkeypatch.setattr(study, "perf_counter", lambda: clock[0])
    step = study._training_step
    calls = []

    def one_step_then_deadline(*args, **kwargs):
        result = step(*args, **kwargs)
        calls.append(1)
        clock[0] = 100.0
        return result

    monkeypatch.setattr(study, "_training_step", one_step_then_deadline)
    output = tmp_path / "timed"
    report = study.train_coordinate_study(coordinate_training_data, output, config, max_seconds=10)
    assert calls == [1]
    assert report["stop_reason"] == "max_seconds" and report["epochs_completed"] == 0
    assert not report["complete"]
    checkpoint = study.coordinate_checkpoint(output / "last.pt")
    assert checkpoint["epoch"] == 0 and checkpoint["optimizer_state"]["state"] == {}
    assert not (output / ".training.lock").exists()


@pytest.fixture(scope="module")
def learning_dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("learning-data") / "dataset"
    generate_dataset(
        directory,
        config=DatasetConfig(
            seed=401,
            train_documents=1,
            validation_documents=1,
            test_documents=1,
            variants_per_document=2,
            output_size=(128, 128),
            photo_size=(128, 128),
            photometric=PhotometricConfig(enabled=False),
        ),
    )
    return directory


@pytest.fixture(scope="module")
def learning_config():
    return TrainConfig(
        seed=29,
        epochs=3,
        batch_size=2,
        learning_rate=0.003,
        threads=1,
        model_config=ModelConfig(
            input_size=(16, 16), grid_size=(5, 5), channels=(4, 8, 8), hidden=16
        ),
    )


@pytest.fixture(scope="module")
def repeat_runs(tmp_path_factory, learning_dataset, learning_config):
    root = tmp_path_factory.mktemp("training-repeatability")
    reports = []
    for name in ("first", "second"):
        reports.append(train(learning_dataset, root / name, config=learning_config))
    return root, reports


def _checkpoint(directory, name="last.pt"):
    return torch.load(directory / name, map_location="cpu", weights_only=True)


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_training_repeats_exactly_and_saves_safe_complete_checkpoints(repeat_runs):
    root, reports = repeat_runs
    first = _checkpoint(root / "first")
    second = _checkpoint(root / "second")
    for key in (
        "model_state",
        "optimizer_state",
        "history",
        "torch_rng_state",
        "shuffle_rng_state",
    ):
        _assert_equal(first[key], second[key])
    required = {
        "config",
        "model_state",
        "optimizer_state",
        "epoch",
        "history",
        "best_validation_epe_pixels",
        "initial_validation_epe_pixels",
        "best_epoch",
        "torch_rng_state",
        "shuffle_rng_state",
        "manifest_sha256",
        "provenance",
        "train_mean_residual",
        "dataset_output_size",
        "dataset_photo_size",
        "dataset_counts",
    }
    assert required <= first.keys()
    assert first["epoch"] == 3
    assert len(first["history"]) == 3
    assert first["dataset_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert first["history"][-1]["train_loss"] < first["history"][0]["train_loss"]
    for epoch, item in enumerate(first["history"], start=1):
        assert item["epoch"] == epoch
        assert item["train_loss"] >= 0
        assert item["validation_epe_pixels"] >= 0
    best = _checkpoint(root / "first", "best.pt")
    assert best["epoch"] == first["best_epoch"]
    assert best["best_validation_epe_pixels"] == first["best_validation_epe_pixels"]
    assert all(isinstance(report, dict) and report for report in reports)


def test_resume_matches_uninterrupted_training_exactly(
    tmp_path, learning_dataset, learning_config, repeat_runs
):
    interrupted = tmp_path / "interrupted"
    train(learning_dataset, interrupted, config=replace(learning_config, epochs=1))
    assert _checkpoint(interrupted)["epoch"] == 1
    train(
        learning_dataset,
        interrupted,
        config=learning_config,
        resume=interrupted / "last.pt",
    )
    expected = _checkpoint(repeat_runs[0] / "first")
    actual = _checkpoint(interrupted)
    for key in (
        "model_state",
        "optimizer_state",
        "history",
        "epoch",
        "best_epoch",
        "best_validation_epe_pixels",
        "initial_validation_epe_pixels",
        "torch_rng_state",
        "shuffle_rng_state",
        "train_mean_residual",
    ):
        _assert_equal(actual[key], expected[key])


def test_training_restores_caller_rng_threads_and_deterministic_settings(
    tmp_path, learning_dataset, learning_config
):
    rng_state = torch.get_rng_state().clone()
    thread_count = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.manual_seed(731)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(False, warn_only=True)
        expected_rng = torch.get_rng_state().clone()
        train(learning_dataset, tmp_path / "trained", config=replace(learning_config, epochs=1))
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.get_num_threads() == 2
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        torch.set_rng_state(rng_state)
        torch.set_num_threads(thread_count)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": -1},
        {"seed": True},
        {"epochs": 0},
        {"epochs": True},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"threads": 0},
    ],
)
def test_invalid_training_settings_fail_before_creating_output(
    tmp_path, learning_dataset, learning_config, changes
):
    destination = tmp_path / "must-not-exist"
    with pytest.raises(FlattenerError):
        train(learning_dataset, destination, config=replace(learning_config, **changes))
    assert not destination.exists()


@pytest.mark.parametrize("existing_kind", ["file", "directory"])
def test_training_refuses_existing_output_without_clobbering(
    tmp_path, learning_dataset, learning_config, existing_kind
):
    destination = tmp_path / "existing"
    if existing_kind == "directory":
        destination.mkdir()
        sentinel = destination / "user.txt"
    else:
        sentinel = destination
    sentinel.write_bytes(b"preserve this user data")
    with pytest.raises(FlattenerError):
        train(learning_dataset, destination, config=learning_config)
    assert sentinel.read_bytes() == b"preserve this user data"
    if existing_kind == "directory":
        assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("corruption", ["artifact", "split-leakage"])
def test_corrupt_dataset_is_rejected_before_loading_or_creating_output(
    tmp_path, learning_dataset, learning_config, monkeypatch, corruption
):
    directory = tmp_path / "dataset"
    shutil.copytree(learning_dataset, directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "artifact":
        photo = directory / manifest["samples"][0]["paths"]["photo.png"]
        photo.write_bytes(photo.read_bytes() + b"tampered")
        reason = "artifact_hash_mismatch"
    else:
        manifest["samples"][2]["document_group"] = manifest["samples"][0]["document_group"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reason = "split_leakage"

    def fail_if_loaded(*args, **kwargs):
        pytest.fail("training loaded a sample before dataset validation finished")

    monkeypatch.setattr(training, "load_sample", fail_if_loaded)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(FlattenerError) as caught:
        train(directory, destination, config=learning_config)
    assert caught.value.reason_code == reason
    assert not destination.exists()


def test_training_and_checkpoint_selection_never_load_test_split(
    tmp_path, learning_dataset, learning_config, monkeypatch
):
    original = training.load_sample
    seen = set()

    def only_training_and_validation(directory, record):
        assert record["split"] in {"train", "validation"}
        seen.add(record["split"])
        return original(directory, record)

    monkeypatch.setattr(training, "load_sample", only_training_and_validation)
    train(learning_dataset, tmp_path / "trained", config=replace(learning_config, epochs=1))
    assert seen == {"train", "validation"}


def test_coordinate_loss_uses_source_pixel_scale_and_averages_valid_coordinates():
    # For an 8×4 source, normalized deltas (1, 0.5) mean pixel errors (2, 2).
    # Smooth L1 with beta=1 gives 1.5 per coordinate, averaged across x and y.
    predicted = torch.tensor([[[[1.0, 0.5], [100.0, 100.0]]]], requires_grad=True)
    target = torch.zeros_like(predicted)
    valid = torch.tensor([[[True, False]]])
    loss = coordinate_loss(predicted, target, valid, photo_size=(8, 4))
    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    torch.testing.assert_close(
        predicted.grad, torch.tensor([[[[1.0, 2.0], [0.0, 0.0]]]]), rtol=0, atol=0
    )


def test_coordinate_loss_ignores_nan_ground_truth_without_nan_gradients():
    predicted = torch.tensor([[[[0.5, -0.5], [0.25, 0.75]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 0.0], [float("nan"), float("nan")]]]])
    valid = torch.tensor([[[True, False]]])
    loss = coordinate_loss(predicted, target, valid, photo_size=(8, 4))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.equal(predicted.grad[0, 0, 1], torch.zeros(2))
    assert torch.count_nonzero(predicted.grad[0, 0, 0]) == 2


def test_coordinate_loss_rejects_an_empty_ground_truth_mask():
    predicted = torch.zeros(1, 2, 3, 2, requires_grad=True)
    with pytest.raises(FlattenerError) as caught:
        coordinate_loss(
            predicted, predicted.detach(), torch.zeros(1, 2, 3, dtype=torch.bool), (8, 4)
        )
    assert caught.value.reason_code == "invalid_training_batch"


def test_training_failure_restores_process_state_and_releases_run_lock(
    tmp_path, learning_dataset, learning_config, monkeypatch
):
    random_state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail_during_loss(*args, **kwargs):
        torch.rand(3)
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(training, "coordinate_loss", fail_during_loss)
    directory = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="injected training failure"):
        train(learning_dataset, directory, config=learning_config)
    assert not (directory / ".training.lock").exists()
    assert torch.equal(torch.get_rng_state(), random_state)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("last-checkpoint", "checkpoint_hash_mismatch"),
        ("best-checkpoint", "checkpoint_hash_mismatch"),
        ("invalid-checkpoint-format", "invalid_checkpoint"),
        ("config", "resume_config_mismatch"),
        ("dataset-manifest", "resume_dataset_mismatch"),
        ("epoch-budget", "resume_epoch_budget"),
        ("output-directory", "invalid_resume_path"),
    ],
)
def test_invalid_resume_is_rejected_before_loading_data_or_changing_run_files(
    tmp_path, learning_dataset, learning_config, repeat_runs, monkeypatch, change, reason
):
    run = tmp_path / "run"
    shutil.copytree(repeat_runs[0] / "first", run)
    dataset = learning_dataset
    destination = run
    config = replace(learning_config, epochs=4)
    if change in {"last-checkpoint", "best-checkpoint"}:
        path = run / ("last.pt" if change == "last-checkpoint" else "best.pt")
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change == "invalid-checkpoint-format":
        payload = b"This is not a torch checkpoint."
        (run / "last.pt").write_bytes(payload)
        (run / "last.pt.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n")
    elif change == "config":
        config = replace(config, learning_rate=0.01)
    elif change == "dataset-manifest":
        dataset = tmp_path / "changed-dataset"
        shutil.copytree(learning_dataset, dataset)
        manifest = dataset / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    elif change == "epoch-budget":
        config = learning_config
    elif change == "output-directory":
        destination = tmp_path / "different-output"

    def fail_if_loaded(*args, **kwargs):
        pytest.fail("invalid resume loaded training data")

    monkeypatch.setattr(training, "load_sample", fail_if_loaded)
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    with pytest.raises(FlattenerError) as caught:
        train(dataset, destination, config=config, resume=run / "last.pt")
    assert caught.value.reason_code == reason
    after = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    assert before == after
    if destination != run:
        assert not destination.exists()
