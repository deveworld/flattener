"""Architecture-explicit region-study checkpoint integrity and inference contracts."""

import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from flattener.contracts import FlattenerError, Status
from flattener.regions import checkpoints as module
from flattener.regions.checkpoints import (
    STUDY_SCHEMA,
    LoadedRegionModel,
    create_region_model,
    load_region_model,
    model_spec,
    read_region_checkpoint,
)
from flattener.regions.segmentation import (
    CHECKPOINT_SCHEMA as LEGACY_SCHEMA,
)
from flattener.regions.segmentation import (
    LoadedSegmenter,
    PageSegmenter,
    SegmenterConfig,
    preprocess_image,
)

pytestmark = pytest.mark.training


def _write(path, state):
    torch.save(state, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    return path


def _state(architecture="local"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        model = create_region_model(architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return {
        "schema_version": STUDY_SCHEMA,
        "config": {
            "seed": 51,
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.001,
            "threads": 1,
            "architecture": architecture,
        },
        "model_spec": model_spec(architecture),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": 0,
        "history": [],
        "initial_validation": {"loss": 1.0},
        "best_validation": {"loss": 1.0},
        "best_epoch": 0,
        "manifest_sha256": "a" * 64,
        "dataset_counts": {"train": 4, "validation": 2, "test": 2},
        "provenance": {
            "device": "cpu",
            "versions": {"torch": str(torch.__version__)},
            "selection": "minimum validation loss; earliest tie",
        },
        "torch_rng_state": torch.get_rng_state().clone(),
        "shuffle_rng_state": torch.Generator().manual_seed(51).get_state(),
    }


@pytest.fixture
def checkpoint(tmp_path):
    state = _state()
    return _write(tmp_path / "local.pt", state), state


@pytest.mark.parametrize("architecture", ["local", "context"])
def test_factory_fixed_model_spec_and_normal_initialization_rng(architecture):
    before = torch.get_rng_state().clone()
    specification = model_spec(architecture)
    assert torch.equal(before, torch.get_rng_state())
    model = create_region_model(architecture)
    assert not torch.equal(before, torch.get_rng_state())
    assert specification["architecture"] == architecture
    assert specification["model_config"] == asdict(model.config)
    assert sum(parameter.numel() for parameter in model.parameters()) <= 1_000_000
    assert model.training
    specification["model_config"]["input_size"] = (9, 9)
    assert model_spec(architecture)["model_config"]["input_size"] == (128, 96)


@pytest.mark.parametrize("architecture", [None, "", "unknown", 1, [], {}])
def test_architecture_names_are_explicit_and_type_safe(architecture):
    for helper in (create_region_model, model_spec):
        with pytest.raises(FlattenerError) as caught:
            helper(architecture)
        assert caught.value.status == Status.INPUT_ERROR


@pytest.mark.parametrize("architecture", ["local", "context"])
def test_read_load_and_prediction_are_strict_original_extent_and_preserve_globals(
    tmp_path, architecture
):
    path = _write(tmp_path / "model.pt", _state(architecture))
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    checkpoint = read_region_checkpoint(path)
    loaded = load_region_model(path)
    assert isinstance(loaded, LoadedRegionModel)
    assert checkpoint["config"]["architecture"] == architecture
    yy, xx = np.mgrid[:31, :47]
    image = np.stack((xx * 5, yy * 7, (xx + yy) % 2 * 255), axis=-1).astype(np.uint8)
    image = image[::-1]
    image.setflags(write=False)
    original = image.copy()
    probability = loaded.predict_probability(image)
    with torch.inference_mode():
        preview = preprocess_image(image, SegmenterConfig(input_size=loaded.config.input_size))
        expected = loaded.model(preview[None]).sigmoid()[0, 0].numpy()
    np.testing.assert_array_equal(probability, expected)
    assert probability.dtype == np.float32 and probability.shape == (128, 96)
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    np.testing.assert_array_equal(image, original)
    probability[:] = 0
    assert loaded.predict_probability(image).max() > 0
    assert not loaded.model.training and not any(p.requires_grad for p in loaded.model.parameters())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.model_sha256 == digest
    assert loaded.model_id == f"{model_spec(architecture)['model_version']}:{digest[:12]}"
    assert loaded.metadata["manifest_sha256"] == "a" * 64
    assert loaded.metadata["model_spec"] == model_spec(architecture)
    assert loaded.metadata["epoch"] == 0 and loaded.load_seconds >= 0
    json.dumps(loaded.metadata, allow_nan=False)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_required_hash_mismatch_and_missing_sidecar_fail_before_unpickling(checkpoint, monkeypatch):
    path, _ = checkpoint
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("unverified data unpickled"))
    path.with_suffix(".pt.sha256").write_text("0" * 64)
    with pytest.raises(FlattenerError) as caught:
        read_region_checkpoint(path)
    assert caught.value.reason_code == "checkpoint_hash_mismatch"
    path.with_suffix(".pt.sha256").unlink()
    with pytest.raises(FlattenerError):
        load_region_model(path)


def test_byte_bound_is_enforced_before_load(checkpoint, monkeypatch):
    path, _ = checkpoint
    monkeypatch.setattr(module, "MAX_CHECKPOINT_BYTES", 128)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("oversized file unpickled"))
    with pytest.raises(FlattenerError) as caught:
        load_region_model(path)
    assert caught.value.reason_code == "checkpoint_size_limit"


def test_torch_load_uses_only_verified_bytes_and_cpu_weights_only(checkpoint, monkeypatch):
    path, _ = checkpoint
    original = torch.load
    calls = []

    def traced(source, **kwargs):
        calls.append(kwargs)
        assert source.read(1)
        source.seek(0)
        return original(source, **kwargs)

    monkeypatch.setattr(torch, "load", traced)
    load_region_model(path)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_root",
        "missing_root",
        "schema",
        "config_extra",
        "seed_bool",
        "epochs_zero",
        "learning_rate_nan",
        "learning_rate_huge",
        "architecture",
        "spec_version",
        "spec_channels",
        "spec_tensor",
        "epoch_bool",
        "epoch_exceeds_budget",
        "history_length",
        "best_epoch",
        "best_loss_nan",
        "initial_loss_string",
        "manifest",
        "counts_bool",
        "counts_missing",
        "provenance_tensor",
        "provenance_nan",
        "rng_float",
        "rng_bad_bytes",
        "shuffle_shape",
        "state_missing",
        "state_extra",
        "state_shape",
        "state_dtype",
        "state_nan",
        "state_sparse",
        "optimizer_empty",
        "optimizer_duplicates",
        "optimizer_nan",
        "optimizer_bad_state",
    ],
)
def test_invalid_checkpoint_contracts_are_structured_and_rng_safe(checkpoint, mutation):
    path, original = checkpoint
    state = copy.deepcopy(original)
    key = next(iter(state["model_state"]))
    if mutation == "extra_root":
        state["unknown"] = 1
    elif mutation == "missing_root":
        del state["dataset_counts"]
    elif mutation == "schema":
        state["schema_version"] = "future"
    elif mutation == "config_extra":
        state["config"]["unknown"] = 1
    elif mutation == "seed_bool":
        state["config"]["seed"] = True
    elif mutation == "epochs_zero":
        state["config"]["epochs"] = 0
    elif mutation == "learning_rate_nan":
        state["config"]["learning_rate"] = float("nan")
    elif mutation == "learning_rate_huge":
        state["config"]["learning_rate"] = 10**1000
    elif mutation == "architecture":
        state["config"]["architecture"] = ["local"]
    elif mutation == "spec_version":
        state["model_spec"]["model_version"] = "wrong"
    elif mutation == "spec_channels":
        state["model_spec"]["model_config"]["channels"] = (1, 2, 3)
    elif mutation == "spec_tensor":
        state["model_spec"]["architecture"] = torch.zeros(3)
    elif mutation == "epoch_bool":
        state["epoch"] = False
    elif mutation == "epoch_exceeds_budget":
        state["epoch"] = 4
    elif mutation == "history_length":
        state["history"] = [{}]
    elif mutation == "best_epoch":
        state["best_epoch"] = 1
    elif mutation == "best_loss_nan":
        state["best_validation"]["loss"] = float("nan")
    elif mutation == "initial_loss_string":
        state["initial_validation"]["loss"] = "1"
    elif mutation == "manifest":
        state["manifest_sha256"] = "z" * 64
    elif mutation == "counts_bool":
        state["dataset_counts"]["test"] = True
    elif mutation == "counts_missing":
        del state["dataset_counts"]["validation"]
    elif mutation == "provenance_tensor":
        state["provenance"]["bad"] = torch.zeros(2)
    elif mutation == "provenance_nan":
        state["provenance"]["bad"] = float("nan")
    elif mutation == "rng_float":
        state["torch_rng_state"] = torch.zeros(5056)
    elif mutation == "rng_bad_bytes":
        state["torch_rng_state"] = torch.zeros(12, dtype=torch.uint8)
    elif mutation == "shuffle_shape":
        state["shuffle_rng_state"] = torch.zeros((2, 2), dtype=torch.uint8)
    elif mutation == "state_missing":
        del state["model_state"][key]
    elif mutation == "state_extra":
        state["model_state"]["unexpected"] = torch.zeros(1)
    elif mutation == "state_shape":
        state["model_state"][key] = torch.zeros(1)
    elif mutation == "state_dtype":
        state["model_state"][key] = state["model_state"][key].double()
    elif mutation == "state_nan":
        state["model_state"][key].fill_(float("nan"))
    elif mutation == "state_sparse":
        state["model_state"][key] = state["model_state"][key].to_sparse()
    elif mutation == "optimizer_empty":
        state["optimizer_state"] = {}
    elif mutation == "optimizer_duplicates":
        state["optimizer_state"]["param_groups"][0]["params"][1] = 0
    elif mutation == "optimizer_nan":
        state["optimizer_state"]["param_groups"][0]["lr"] = float("nan")
    elif mutation == "optimizer_bad_state":
        state["optimizer_state"]["state"][0] = {"step": "invalid"}
    _write(path, state)
    before = torch.get_rng_state().clone()
    with pytest.raises(FlattenerError) as caught:
        read_region_checkpoint(path)
    assert caught.value.status == Status.MODEL_ERROR
    assert torch.equal(before, torch.get_rng_state())


def test_oversized_model_state_is_rejected(checkpoint):
    path, state = checkpoint
    state["model_state"]["huge"] = torch.zeros(1_000_001)
    _write(path, state)
    with pytest.raises(FlattenerError) as caught:
        load_region_model(path)
    assert caught.value.reason_code == "model_resource_limit"


def test_real_adam_moment_state_and_earliest_best_history_validate(checkpoint):
    path, state = checkpoint
    with torch.random.fork_rng(devices=[]):
        model = create_region_model("local")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state["model_state"] = model.state_dict()
    state["optimizer_state"] = optimizer.state_dict()
    state["epoch"] = 2
    state["history"] = [
        {"epoch": 1, "train_loss": 0.9, "validation": {"loss": 0.8}},
        {"epoch": 2, "train_loss": 0.8, "validation": {"loss": 0.8}},
    ]
    state["best_epoch"] = 1
    state["best_validation"] = {"loss": 0.8}
    _write(path, state)
    assert read_region_checkpoint(path)["best_epoch"] == 1
    state["best_epoch"] = 2
    _write(path, state)
    with pytest.raises(FlattenerError, match="earliest"):
        read_region_checkpoint(path)
    state["best_epoch"] = 1
    state["optimizer_state"]["state"][0]["exp_avg"] = torch.zeros(2)
    _write(path, state)
    with pytest.raises(FlattenerError, match="moment"):
        read_region_checkpoint(path)


@pytest.mark.parametrize(
    "name,value",
    [
        ("lr", -0.001),
        ("lr", 0.002),
        ("betas", (0.9, 1)),
        ("eps", 0),
        ("weight_decay", -1),
        ("amsgrad", 1),
        ("capturable", "yes"),
        ("fused", 1),
    ],
)
def test_invalid_adam_hyperparameters_do_not_pass_resume_validation(checkpoint, name, value):
    path, state = checkpoint
    state["optimizer_state"]["param_groups"][0][name] = value
    _write(path, state)
    with pytest.raises(FlattenerError):
        read_region_checkpoint(path)


def test_checkpoint_allocation_errors_are_not_disguised_as_expected_model_errors(
    checkpoint, monkeypatch
):
    def out_of_memory(*args, **kwargs):
        raise RuntimeError("can't allocate memory")

    monkeypatch.setattr(torch, "load", out_of_memory)
    with pytest.raises(MemoryError):
        load_region_model(checkpoint[0])


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 3), np.float32),
        np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (4001, 4001, 3)),
    ],
)
def test_bad_or_oversized_images_fail_before_allocation_and_model_execution(
    checkpoint, monkeypatch, image
):
    loaded = load_region_model(checkpoint[0])
    monkeypatch.setattr(
        loaded.model, "forward", lambda _: pytest.fail("invalid image reached model")
    )
    with pytest.raises(FlattenerError) as caught:
        loaded.predict_probability(image)
    assert caught.value.status == Status.INPUT_ERROR


@pytest.mark.parametrize("bad_output", ["nan", "inf", "shape", "dtype", "runtime"])
def test_invalid_model_outputs_cannot_become_probabilities(checkpoint, monkeypatch, bad_output):
    loaded = load_region_model(checkpoint[0])

    def forward(_):
        if bad_output == "runtime":
            raise RuntimeError("test failure")
        if bad_output == "shape":
            return torch.zeros((1, 2, 128, 96))
        if bad_output == "dtype":
            return torch.zeros((1, 1, 128, 96), dtype=torch.float64)
        return torch.full((1, 1, 128, 96), float(bad_output))

    monkeypatch.setattr(loaded.model, "forward", forward)
    with pytest.raises(FlattenerError) as caught:
        loaded.predict_probability(np.zeros((31, 47, 3), np.uint8))
    assert caught.value.status == Status.MODEL_ERROR


def test_legacy_checkpoints_delegate_without_rewriting_or_numerical_changes(tmp_path, monkeypatch):
    config = SegmenterConfig(input_size=(17, 13), channels=(2, 4, 6))
    with torch.random.fork_rng(devices=[]):
        model = PageSegmenter(config)
    state = _state()
    state.pop("model_spec")
    state["schema_version"] = LEGACY_SCHEMA
    state["config"] = {"model_config": asdict(config)}
    state["model_state"] = model.state_dict()
    state["optimizer_state"] = {}
    path = _write(tmp_path / "legacy.pt", state)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    original = module.load_segmenter

    def delegate(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, "load_segmenter", delegate)
    loaded = load_region_model(path)
    assert isinstance(loaded, LoadedSegmenter) and calls == [path]
    assert read_region_checkpoint(path)["schema_version"] == LEGACY_SCHEMA
    image = np.full((31, 47, 3), 127, np.uint8)
    with torch.inference_mode():
        expected = model(preprocess_image(image, config)[None]).sigmoid()[0, 0].numpy()
    np.testing.assert_array_equal(loaded.predict_probability(image), expected)
    assert loaded.model_sha256 == digest == hashlib.sha256(path.read_bytes()).hexdigest()
