"""Admission checks plus explicitly marked tiny initialization/resume tests."""

import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import test_coordinate_scene_data as coordinate_fixtures
import torch
from PIL import Image

from flattener import FlattenerError
from flattener.learning import coordinate_study as study
from flattener.learning import scene_data
from flattener.learning.model import ModelConfig, coarse_identity, dense_maps

# Reuse the authored, tiny fixture chain without another dataset-building script.
source = coordinate_fixtures.source
paper_source = coordinate_fixtures.paper_source
mutable_paper = coordinate_fixtures.mutable_paper


def _oriented_config(**changes):
    return replace(study.StudyConfig(
        architecture="oriented_surface", device="cpu", map_mode="analytic",
        page_weight=1, direction_weight=1, corner_weight=1,
        output_size=(16, 12), batch_size=2, epochs=2,
        model_config=ModelConfig(input_size=(16, 12), grid_size=(3, 3),
                                 channels=(2, 4, 4), hidden=8)), **changes)


def _semantic_batch():
    corners = torch.tensor([[[-0.7, -0.8], [0.7, -0.8], [0.7, 0.8], [-0.7, 0.8]]])
    return {"corner_target": corners, "corner_valid": torch.ones(1, 4, dtype=torch.bool),
            "direction_target": torch.zeros(1, dtype=torch.int64),
            "direction_valid": torch.ones(1, dtype=torch.bool)}


@pytest.mark.parametrize("mode", [None, True, "", "Teacher", "argmax", 1])
def test_direction_training_mode_rejects_invalid_values_before_device_or_data(
        mode, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_resolve_device", lambda *args, **kwargs:
                        pytest.fail("invalid config reached device resolution"))
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(
            tmp_path / "missing-data", output, _oriented_config(direction_training=mode))
    assert caught.value.reason_code == "invalid_config"
    assert not output.exists()


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "projective", "surface"])
def test_teacher_direction_requires_oriented_architecture(architecture):
    config = study.StudyConfig(architecture=architecture, direction_training="teacher",
                              map_mode="analytic" if architecture in ("projective", "surface")
                              else "legacy")
    with pytest.raises(FlattenerError, match="teacher direction requires oriented_surface"):
        study._config(config)


def test_teacher_geometry_uses_fixed_valid_classes_but_cross_entropy_keeps_predicted_logits():
    config = _oriented_config(direction_training="teacher", curvature_weight=1)
    model = study._new_model(config).train()
    image = torch.zeros(4, 3, *config.model_config.input_size)
    classes = torch.tensor([1, 2, 3, 999], dtype=torch.int64)
    valid = torch.tensor([True, True, True, False])
    with torch.no_grad():
        model.orientation_head.bias[0] = 4.0
        _, baseline, base_aux = study._forward_maps(model, image, config, photo_size=(32, 24))
        checks = []
        residual, dense, auxiliary = study._forward_maps(
            model, image, config, photo_size=(32, 24), _checks=checks,
            teacher_direction=classes, teacher_valid=valid)
    flat = coarse_identity(config.output_size)
    x, y = flat.unbind(-1)
    expected = torch.stack((torch.stack((-y * (32 / 24), x * (24 / 32)), -1),
                            -flat, torch.stack((y * (32 / 24), -x * (24 / 32)), -1), flat))
    torch.testing.assert_close(dense, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(baseline, flat.expand_as(baseline), rtol=0, atol=0)
    assert residual.shape == (4, 2, *config.model_config.grid_size)
    assert bool(torch.stack(checks).all())
    torch.testing.assert_close(auxiliary["orientation_logits"], base_aux["orientation_logits"],
                               rtol=0, atol=0)
    labels = {"direction_target": classes, "direction_valid": valid,
              "corner_target": torch.full((4, 4, 2), float("nan")),
              "corner_valid": torch.zeros(4, 4, dtype=torch.bool)}
    direction, corner = study._semantic_page_losses(auxiliary, labels, (32, 24))
    expected_ce = torch.nn.functional.cross_entropy(
        auxiliary["orientation_logits"][:3], classes[:3])
    torch.testing.assert_close(direction, expected_ce)
    assert direction > 4 and corner == 0  # Geometry forcing never substitutes GT logits.
    assert torch.equal(classes, torch.tensor([1, 2, 3, 999]))
    assert torch.equal(valid, torch.tensor([True, True, True, False]))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_teacher_invalid_eligible_class_is_safely_rejected_before_backward():
    config = _oriented_config(direction_training="teacher")
    model = study._new_model(config).train()
    checks = []
    with torch.no_grad():
        _, dense, auxiliary = study._forward_maps(
            model, torch.zeros(1, 3, 16, 12), config, photo_size=(32, 24), _checks=checks,
            teacher_direction=torch.tensor([2**62]), teacher_valid=torch.tensor([True]))
    assert not bool(torch.stack(checks).all())
    assert torch.isfinite(dense).all() and torch.isfinite(auxiliary["corners"]).all()
    torch.testing.assert_close(dense[0], coarse_identity(config.output_size), rtol=0, atol=0)


def test_teacher_override_is_private_checked_training_only():
    config = _oriented_config(direction_training="teacher")
    model = study._new_model(config)
    images = torch.zeros(1, 3, 16, 12)
    override = {"teacher_direction": torch.tensor([2]), "teacher_valid": torch.tensor([True])}
    with pytest.raises(FlattenerError, match="checked training path"):
        study._forward_maps(model, images, config, photo_size=(32, 24), **override)
    with pytest.raises(FlattenerError, match="checked training path"):
        study._forward_maps(model, images, replace(config, direction_training="predicted"),
                            photo_size=(32, 24), _checks=[], **override)
    for method in (model.forward_dense, model.forward_with_geometry):
        with pytest.raises(TypeError):
            method(images, config.output_size, photo_size=(32, 24), **override)
    model.eval()
    with pytest.raises(ValueError, match="only in training mode"):
        model._forward_with_geometry_validated(
            images, config.output_size, photo_size=(32, 24), **override)
    with pytest.raises(FlattenerError, match="checked training path"):
        study._forward_maps(model, images, config, photo_size=(32, 24), _checks=[], **override)


@pytest.mark.parametrize("mode", ["predicted", "teacher"])
@pytest.mark.parametrize("augment", [False, True])
def test_teacher_training_routing_uses_postaugmentation_labels_without_optimization(
        monkeypatch, mode, augment):
    config = _oriented_config(direction_training=mode, rotation_mode="per_image_fit",
                              rotation_probability=1.0 if augment else 0.0)
    model = study._new_model(config).train()
    batch = {**_semantic_batch(), "image": torch.zeros(1, 3, 16, 12),
             "target": coarse_identity(config.output_size)[None],
             "valid": torch.ones(1, *config.output_size, dtype=torch.bool),
             "source_mask": torch.ones(1, 32, 24, dtype=torch.bool)}
    labels = {"direction_target": torch.tensor([3]), "direction_valid": torch.tensor([False])}
    calls = []

    class ForwardReached(Exception):
        pass

    class AdmissionOnlyOptimizer:
        param_groups = [{"params": list(model.parameters())}]

        def zero_grad(self, *, set_to_none):
            assert set_to_none

        def step(self):
            pytest.fail("routing fixture must never optimize")

    def augmentation(value, **kwargs):
        calls.append("augment")
        return {**value, **labels}

    def forward(model, images, config, *, _checks, photo_size, **kwargs):
        calls.append("forward")
        assert photo_size == (32, 24)
        expected = labels if augment else batch
        if mode == "teacher":
            assert kwargs.keys() == {"teacher_direction", "teacher_valid"}
            assert torch.equal(kwargs["teacher_direction"], expected["direction_target"])
            assert torch.equal(kwargs["teacher_valid"], expected["direction_valid"])
        else:
            assert not kwargs  # Default path does not even pass optional teacher keywords.
        raise ForwardReached

    monkeypatch.setattr(study, "augment_training_batch", augmentation)
    monkeypatch.setattr(study, "_forward_maps", forward)
    with pytest.raises(ForwardReached):
        study._training_step(model, AdmissionOnlyOptimizer(), batch, config, (32, 24), "cpu")
    assert calls == (["augment", "forward"] if augment else ["forward"])
    assert batch["direction_target"].tolist() == [0] and batch["direction_valid"].tolist() == [True]
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("name", ["page_weight", "direction_weight", "corner_weight"])
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_oriented_supervision_weights_are_explicit_and_positive(name, value):
    with pytest.raises(FlattenerError):
        study._config(_oriented_config(**{name: value}))


def test_oriented_rotation_requires_physical_fit_and_other_models_reject_new_losses():
    config = _oriented_config(rotation_probability=1, rotation_mode="per_image_fit")
    assert study._config(config) is config
    with pytest.raises(FlattenerError):
        study._config(replace(config, rotation_mode="batch"))
    for name in ("direction_weight", "corner_weight"):
        with pytest.raises(FlattenerError):
            study._config(replace(study.StudyConfig(), **{name: 1}))


def test_semantic_losses_observe_shrinkage_and_use_fixed_corner_mask():
    batch = _semantic_batch()
    auxiliary = {"orientation_logits": torch.zeros(1, 4), "corners": batch["corner_target"]}
    direction, corner = study._semantic_page_losses(auxiliary, batch, (384, 288))
    assert direction == pytest.approx(math.log(4)) and corner == 0
    auxiliary["corners"] = 0.4 * batch["corner_target"]
    assert study._semantic_page_losses(auxiliary, batch, (384, 288))[1] > 0.1
    batch["corner_valid"][:, 3] = False
    batch["corner_target"][:, 3] = float("nan")
    auxiliary["corners"] = torch.nan_to_num(batch["corner_target"], nan=1000)
    assert study._semantic_page_losses(auxiliary, batch, (384, 288))[1] == 0


def test_semantic_labels_empty_mask_is_finite_and_out_of_range_class_fails_closed():
    batch = _semantic_batch()
    auxiliary = {"orientation_logits": torch.zeros(1, 4), "corners": batch["corner_target"].clone()}
    batch["corner_valid"].zero_()
    batch["corner_target"].fill_(float("nan"))
    batch["direction_valid"].zero_()
    batch["direction_target"].fill_(-1)
    assert all(value == 0 for value in study._semantic_page_losses(auxiliary, batch, (384, 288)))
    batch["direction_valid"].fill_(True)
    batch["direction_target"].fill_(4)
    with pytest.raises(FlattenerError):
        study._semantic_page_losses(auxiliary, batch, (384, 288))
    checks = []
    study._semantic_page_losses(auxiliary, batch, (384, 288), _checks=checks)
    assert len(checks) == 1 and not checks[0]


@pytest.mark.parametrize("direction_training", ["predicted", "teacher"])
def test_oriented_validation_diagnostics_keep_domains_documents_and_missing_counts(
        direction_training):
    config = _oriented_config(direction_training=direction_training)
    model = study._new_model(config)

    class ToyValidation(torch.utils.data.Dataset):
        records = [{"document_group": "a"}, {"document_group": "a"},
                   {"document_group": "b", "source": "paper", "kind": "capture"}]

        def __len__(self):
            return 3

        def __getitem__(self, index):
            labels = {key: value[0] for key, value in _semantic_batch().items()}
            if index == 2:
                labels["direction_target"] = torch.tensor(2)
            return {**labels, "image": torch.zeros(3, 16, 12),
                    "target": coarse_identity(config.output_size),
                    "valid": torch.ones(config.output_size, dtype=torch.bool)}

    diagnostics = {}
    value = study._validation_epe(model, torch.utils.data.DataLoader(ToyValidation(), batch_size=2),
                                  config, (32, 24), torch.device("cpu"), diagnostics=diagnostics)
    assert value == 0
    assert diagnostics["images"] == 3
    assert diagnostics["by_domain"]["authored"]["documents"] == 1
    assert diagnostics["by_domain"]["authored"]["images"] == 2
    assert diagnostics["by_domain"]["capture"]["frame_mean_epe_pixels"] == 0
    assert diagnostics["semantic"]["direction_confusion_gt_rows"] == [
        [2, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    assert diagnostics["semantic"]["direction_accuracy"] == pytest.approx(2 / 3)
    assert diagnostics["semantic"]["corner_eligible_images"] == 3
    assert diagnostics["semantic"]["corner_mean_epe_pixels"] > 0
    assert diagnostics["quality_gates_evaluated"] is False


def test_oriented_backbone_initialization_keeps_geometry_and_direction_heads_fresh(
        initial_checkpoint):
    path, source_config, payload, _ = initial_checkpoint
    config = replace(source_config, architecture="oriented_surface", map_mode="analytic",
                     page_weight=1, direction_weight=1, corner_weight=1,
                     initialization_mode="backbone")
    bound, state = study._initialization(config, path, None)
    assert bound.initial_checkpoint_sha256 == study._sha256(path)
    for key, value in payload["model_state"].items():
        if key.startswith(("encoder.", "head.1.")):
            assert torch.equal(state[key], value)
    assert not state["head.3.weight"].count_nonzero()
    assert not state["orientation_head.weight"].count_nonzero()
    assert not state["orientation_head.bias"].count_nonzero()
    assert any(key.startswith("page_head.") for key in state)


def test_oriented_backbone_fresh_page_head_uses_run_seed_and_preserves_caller_rng(
        initial_checkpoint, no_cuda):
    path, source_config, _, _ = initial_checkpoint
    config = replace(source_config, architecture="oriented_surface", map_mode="analytic",
                     page_weight=1, direction_weight=1, corner_weight=1,
                     initialization_mode="backbone", seed=314)
    states = []
    with torch.random.fork_rng(devices=[]):
        for ambient in (100, 200):
            torch.random.default_generator.manual_seed(ambient)
            before = torch.get_rng_state().clone()
            _, state = study._initialization(config, path, None)
            assert torch.equal(before, torch.get_rng_state())
            states.append(state)
        assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
        torch.random.default_generator.manual_seed(config.seed)
        fresh = study._new_model(config).state_dict()
    assert all(torch.equal(states[0][key], fresh[key]) for key in fresh
               if key.startswith("page_head."))


@pytest.mark.training
def test_semantic_direction_loss_has_gradient_for_opposite_direction():
    batch = _semantic_batch()
    logits = torch.tensor([[0.0, 0.0, 4.0, 0.0]], requires_grad=True)
    auxiliary = {"orientation_logits": logits, "corners": batch["corner_target"].clone()}
    direction, _ = study._semantic_page_losses(auxiliary, batch, (384, 288))
    direction.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < -0.9 and logits.grad[0, 2] > 0.9


@pytest.mark.training
@pytest.mark.parametrize("augment", [False, True])
def test_oriented_training_step_updates_all_heads_with_toy_batch(augment):
    config = _oriented_config(
        rotation_probability=1.0 if augment else 0.0, rotation_mode="per_image_fit",
        background_probability=0.5 if augment else 0.0)
    model = study._new_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 foreach=False, fused=False)
    batch = {**_semantic_batch(), "image": torch.full((1, 3, 16, 12), 0.5),
             "target": coarse_identity(config.output_size)[None],
             "valid": torch.ones(1, *config.output_size, dtype=torch.bool),
             "source_mask": torch.ones(1, 32, 24, dtype=torch.bool)}
    loss, pixels = study._training_step(model, optimizer, batch, config, (32, 24), "cpu")
    assert math.isfinite(loss) and pixels == 16 * 12
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert len(optimizer.state) == len(list(model.parameters()))


@pytest.mark.training
@pytest.mark.parametrize("direction_training", ["predicted", "teacher"])
def test_oriented_labels_cached_loader_train_and_resume_tiny_authored_fixture(
        source, tmp_path, no_cuda, direction_training):
    config = _oriented_config(rotation_probability=0.5, rotation_mode="per_image_fit",
                              direction_training=direction_training)
    caches = study.prepare_coordinate_data(source, config, cache_max_bytes=32 * 1024**2)
    train = caches["train"]
    loader = study._data_loader(train, config, torch.Generator().manual_seed(4), False, True)
    batch = next(iter(loader))
    assert batch["corner_target"].shape == (2, 4, 2)
    assert batch["direction_target"].dtype == torch.int64
    output = tmp_path / "oriented-tiny"
    report = study.train_coordinate_study(
        source, output, config, prepared_data=caches, cache_max_bytes=32 * 1024**2,
        stop_after_epoch=1)
    assert report["epochs_completed"] == 1
    first = study.coordinate_checkpoint(output / "last.pt")
    assert first["config"]["direction_training"] == direction_training
    opposite = "teacher" if direction_training == "predicted" else "predicted"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(
            source, output, replace(config, direction_training=opposite),
            prepared_data=caches, cache_max_bytes=32 * 1024**2, resume=output / "last.pt")
    assert caught.value.reason_code == "invalid_resume"
    report = study.train_coordinate_study(
        source, output, config, prepared_data=caches, cache_max_bytes=32 * 1024**2,
        resume=output / "last.pt")
    assert report["complete"] and report["epochs_completed"] == 2
    assert report["validation_diagnostics"]["epoch"] == 2
    assert report["validation_diagnostics"]["images"] == len(caches["validation"])
    assert report["validation_diagnostics"]["semantic"]["direction_eligible"] > 0
    assert report["quality_selection"] == {"status": "not_evaluated", "best_eligible": None}
    saved = study.coordinate_checkpoint(output / "last.pt")
    assert saved["config"]["architecture"] == "oriented_surface"
    assert saved["config"]["direction_training"] == direction_training
    assert saved["schema_version"] == study.CHECKPOINT_SCHEMA
    assert not report["test_evaluated"]


@pytest.mark.training
@pytest.mark.parametrize("direction_training", ["predicted", "teacher"])
def test_xpu_oriented_checkpoint_save_resume_preserves_rng(
        source, tmp_path, no_cuda, direction_training):
    """Opt-in authored-data XPU smoke; never runs during non-training tests."""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("explicit XPU smoke requires an available XPU")
    config = _oriented_config(device="xpu:0", rotation_probability=0.5,
                              rotation_mode="per_image_fit", background_probability=0.5,
                              direction_training=direction_training)
    budget = 32 * 1024**2
    prepared = study.prepare_coordinate_data(source, config, cache_max_bytes=budget)
    controls = {"prepared_data": prepared, "cache_max_bytes": budget}
    original_device = torch.xpu.current_device()
    selected_rng = torch.xpu.get_rng_state(0).clone()
    other_rng = (torch.xpu.get_rng_state(original_device).clone()
                 if original_device != 0 else None)
    cpu_rng = torch.get_rng_state().clone()

    def assert_caller_restored():
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.xpu.current_device() == original_device
        assert torch.equal(torch.xpu.get_rng_state(0), selected_rng)
        if other_rng is not None:
            assert torch.equal(torch.xpu.get_rng_state(original_device), other_rng)

    reference = tmp_path / "xpu-reference"
    report = study.train_coordinate_study(source, reference, config, **controls)
    assert report["complete"] and not report["test_evaluated"]
    assert_caller_restored()
    expected = study.coordinate_checkpoint(reference / "last.pt")
    output = tmp_path / "xpu-resumed"
    first = study.train_coordinate_study(source, output, config, stop_after_epoch=1, **controls)
    assert first["epochs_completed"] == 1 and not first["complete"]
    assert_caller_restored()
    report = study.train_coordinate_study(
        source, output, config, resume=output / "last.pt", **controls)
    assert report["complete"] and report["epochs_completed"] == 2
    assert_caller_restored()
    actual = study.coordinate_checkpoint(output / "last.pt")
    for checkpoint in (expected, actual):
        assert checkpoint["schema_version"] == study.CHECKPOINT_SCHEMA
        assert checkpoint["config"]["device"] == "xpu:0"
        assert checkpoint["config"]["direction_training"] == direction_training
        assert checkpoint["device_info"]["device"] == "xpu:0"
        assert checkpoint["device_info"]["capability"] is None
        assert checkpoint["cuda_rng_state"] is None
        rng = checkpoint["xpu_rng_state"]
        assert rng.device.type == "cpu" and rng.dtype == torch.uint8 and rng.ndim == 1
        assert 16 <= rng.numel() <= 65_536
    for key in ("torch_rng_state", "shuffle_rng_state", "xpu_rng_state"):
        assert torch.equal(actual[key], expected[key]), key
    # Accelerator arithmetic is not promised bit-exact; RNG/epoch admission is.
    for key in ("model_state", "optimizer_state", "history"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=2e-6)
    model = study.model_from_checkpoint(actual)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    with torch.no_grad():
        residual, dense = model.forward_dense(torch.zeros(1, 3, *config.model_config.input_size),
                                             config.output_size,
                                             photo_size=actual["dataset_photo_size"])
    assert residual.shape == (1, 2, *config.model_config.grid_size)
    assert dense.shape == (1, *config.output_size, 2) and torch.isfinite(dense).all()
    assert_caller_restored()


@pytest.fixture(autouse=True)
def never_train(monkeypatch, request):
    def forbidden(*args, **kwargs):
        pytest.fail("a non-training test attempted training or dataset construction")

    if request.node.get_closest_marker("training") is None:
        monkeypatch.setattr(torch.Tensor, "backward", forbidden)
        monkeypatch.setattr(torch.autograd, "backward", forbidden)
        monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
        monkeypatch.setattr(study, "SceneCoordinateDataset", forbidden)


@pytest.fixture
def no_cuda(monkeypatch):
    for name in (
        "is_available",
        "current_device",
        "device_count",
        "get_device_name",
        "get_device_capability",
        "get_rng_state",
        "get_rng_state_all",
        "set_rng_state",
        "set_rng_state_all",
        "manual_seed",
        "manual_seed_all",
        "device",
        "_lazy_init",
    ):
        # Dynamo may lazily inspect these APIs when Adam is first constructed.
        # Each API needs its own object to avoid conflicting tracing rules.
        def forbidden(*args, _name=name, **kwargs):
            pytest.fail(f"explicit non-CUDA admission must not access torch.cuda.{_name}")

        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_cuda_remains_default_and_cpu_requires_explicit_request():
    config = study.StudyConfig()
    assert config.device == "cuda"
    assert config.epochs == 40
    assert config.architecture == "global"
    assert config.model_config.input_size == (96, 72)
    assert config.model_config.grid_size == (17, 13)
    assert config.topology_weight == 0
    assert config.curvature_weight == 0
    assert config.gradient_clip_norm == 0
    assert config.page_weight == 0
    assert config.boundary_weight == 0
    assert config.differential_weight == 0
    assert config.page_relative_weight == 0
    assert config.orientation_weight == 0
    assert config.rotation_probability == 0
    assert config.background_probability == 0
    assert config.rotation_mode == "batch"
    assert config.initial_model_config is None
    assert config.include_paper_augmentation is False
    assert config.initial_checkpoint_sha256 is None
    assert config.direction_training == "predicted"


@pytest.mark.parametrize("field,value", [
    ("include_paper_augmentation", 0), ("include_paper_augmentation", 1),
    ("include_paper_augmentation", "true"), ("include_paper_augmentation", None),
    ("initial_checkpoint_sha256", True), ("initial_checkpoint_sha256", ""),
    ("initial_checkpoint_sha256", "a" * 63), ("initial_checkpoint_sha256", "g" * 64),
    ("initial_checkpoint_sha256", "A" * 64), ("initial_checkpoint_sha256", "a" * 64 + "\n"),
])
def test_initialization_config_rejects_invalid_fields_before_device_or_data(
    field, value, monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid initialization config reached device resolution")

    monkeypatch.setattr(study, "_resolve_device", forbidden)
    config = replace(study.StudyConfig(device="cpu"), **{field: value})
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(tmp_path / "missing-data", output, config)
    assert caught.value.reason_code == "invalid_config"
    assert not output.exists()


@pytest.fixture
def initial_checkpoint(tmp_path):
    config = study.StudyConfig(
        device="cpu", epochs=2, batch_size=2, output_size=(16, 16),
        model_config=ModelConfig(input_size=(16, 16), grid_size=(3, 3),
                                 channels=(2, 4, 4), hidden=8),
    )
    with torch.random.fork_rng(devices=[]):
        model = study._new_model(config)
    payload = {
        "schema_version": study.CHECKPOINT_SCHEMA, "checkpoint_kind": "best",
        "config": asdict(config), "model_spec": study._model_spec(config),
        "model_state": model.state_dict(),
        "optimizer_state": {
            "state": {}, "param_groups": [study._adam_group(config, len(list(model.parameters())))],
        },
        "epoch": 0, "best_epoch": 0, "history": [],
        "initial_validation_epe_pixels": 1.0, "best_validation_epe_pixels": 1.0,
        "manifest_sha256": "0" * 64, "provenance": study._provenance(),
        "dataset_output_size": config.output_size, "dataset_photo_size": (16, 16),
        "dataset_counts": {"train": 2, "validation": 2, "test": 2},
        "torch_rng_state": torch.get_rng_state(),
        "shuffle_rng_state": torch.Generator().manual_seed(config.seed).get_state(),
        "cuda_rng_state": None,
        "xpu_rng_state": None,
        "device_info": {"device": "cpu", "name": "fixture", "capability": None},
        "best_checkpoint_sha256": None,
    }
    path = tmp_path / "initial.pt"
    digest = study._save_checkpoint(path, payload)
    return path, config, payload, digest


def _xpu_checkpoint(payload, *, device="xpu:2", length=32):
    payload = deepcopy(payload)
    payload["config"]["device"] = device
    payload["device_info"] = {"device": "xpu:2", "name": "mock XPU", "capability": None}
    payload["xpu_rng_state"] = torch.zeros(length, dtype=torch.uint8)
    return payload


@pytest.mark.parametrize("device", ["xpu", "xpu:2"])
@pytest.mark.parametrize("length", [16, 32, 65_536])
def test_latest_xpu_checkpoint_validation_is_cpu_only_and_preserves_rng(
        initial_checkpoint, mocked_xpu, no_cuda, monkeypatch, device, length):
    for name in vars(mocked_xpu[0]):
        monkeypatch.setattr(mocked_xpu[0], name, lambda *args, **kwargs:
                            pytest.fail("checkpoint validation accessed an XPU API"))
    payload = _xpu_checkpoint(initial_checkpoint[2], device=device, length=length)
    before = torch.get_rng_state().clone()
    assert study._validate_saved(payload).device == device
    assert torch.equal(torch.get_rng_state(), before)
    assert payload["schema_version"] == study.CHECKPOINT_SCHEMA
    assert payload["cuda_rng_state"] is None


@pytest.mark.parametrize("bad", [
    "missing", "none", "float", "rank", "short", "long", "meta", "sparse", "unused_cuda",
    "wrong_device", "wrong_index", "capability", "version_bool", "version_empty", "protocol",
])
def test_latest_xpu_checkpoint_rejects_invalid_state_or_provenance(
        initial_checkpoint, mocked_xpu, no_cuda, bad):
    payload = _xpu_checkpoint(initial_checkpoint[2])
    if bad == "missing":
        payload.pop("xpu_rng_state")
    elif bad == "none":
        payload["xpu_rng_state"] = None
    elif bad == "float":
        payload["xpu_rng_state"] = torch.zeros(32)
    elif bad == "rank":
        payload["xpu_rng_state"] = torch.zeros((2, 16), dtype=torch.uint8)
    elif bad in ("short", "long"):
        payload["xpu_rng_state"] = torch.zeros(15 if bad == "short" else 65_537, dtype=torch.uint8)
    elif bad == "meta":
        payload["xpu_rng_state"] = torch.empty(32, dtype=torch.uint8, device="meta")
    elif bad == "sparse":
        payload["xpu_rng_state"] = torch.zeros(32, dtype=torch.uint8).to_sparse()
    elif bad == "unused_cuda":
        payload["cuda_rng_state"] = torch.zeros(32, dtype=torch.uint8)
    elif bad in ("wrong_device", "wrong_index"):
        payload["device_info"]["device"] = "cpu" if bad == "wrong_device" else "xpu:1"
    elif bad == "capability":
        payload["device_info"]["capability"] = (8, 0)
    elif bad.startswith("version"):
        payload["provenance"]["versions"]["torch_xpu"] = True if bad == "version_bool" else ""
    else:
        payload["provenance"]["execution_protocol"]["devices"] = "automatic fallback"
    with pytest.raises(FlattenerError) as caught:
        study._validate_saved(payload)
    assert caught.value.reason_code == "invalid_checkpoint"
    assert mocked_xpu[2] == []


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_latest_other_backends_require_none_xpu_state(initial_checkpoint, device):
    payload = deepcopy(initial_checkpoint[2])
    payload["config"]["device"] = device
    if device.startswith("cuda"):
        payload["device_info"] = {"device": device, "name": "mock CUDA", "capability": (8, 0)}
        payload["cuda_rng_state"] = torch.zeros(32, dtype=torch.uint8)
    assert study._validate_saved(payload).device == device
    payload["xpu_rng_state"] = torch.zeros(32, dtype=torch.uint8)
    with pytest.raises(FlattenerError) as caught:
        study._validate_saved(payload)
    assert caught.value.reason_code == "invalid_checkpoint"


@pytest.mark.parametrize("minor", range(1, 11))
def test_legacy_checkpoint_fields_remain_strict_without_xpu_state(initial_checkpoint, minor):
    payload = deepcopy(initial_checkpoint[2])
    schema = f"coordinate-study-0.{minor}"
    payload["schema_version"] = schema
    payload.pop("xpu_rng_state")
    payload["provenance"] = study._provenance(schema)
    additions = {
        2: ("topology_weight", "curvature_weight", "gradient_clip_norm"),
        3: ("page_weight", "boundary_weight", "differential_weight"),
        4: ("include_paper_augmentation", "initial_checkpoint_sha256"),
        5: ("map_mode", "topology_mode", "initialization_mode"),
        6: ("page_relative_weight",), 7: ("orientation_weight", "rotation_probability"),
        9: ("background_probability", "rotation_mode", "initial_model_config"),
        10: ("direction_weight", "corner_weight"),
        12: ("direction_training",),
    }
    for introduced, names in additions.items():
        if minor < introduced:
            for name in names:
                payload["config"].pop(name)
    assert "torch_xpu" not in payload["provenance"]["versions"]
    assert "execution_protocol" not in payload["provenance"]
    assert study._validate_saved(payload).device == "cpu"
    payload["xpu_rng_state"] = None
    with pytest.raises(FlattenerError) as caught:
        study._validate_saved(payload)
    assert caught.value.reason_code == "invalid_checkpoint"
    payload.pop("xpu_rng_state")
    payload["config"]["device"] = "xpu:2"
    with pytest.raises(FlattenerError) as caught:
        study._validate_saved(payload)
    assert caught.value.reason_code == "invalid_checkpoint"


@pytest.mark.parametrize("declare_sha", [False, True])
def test_initialization_binds_verified_model_only_and_preserves_rng(
    initial_checkpoint, declare_sha
):
    path, config, payload, digest = initial_checkpoint
    config = replace(config, seed=config.seed + 1, learning_rate=0.002,
                     include_paper_augmentation=True,
                     initial_checkpoint_sha256=digest if declare_sha else None)
    before = torch.get_rng_state().clone()
    actual_config, state = study._initialization(config, path, None)
    assert actual_config == replace(config, initial_checkpoint_sha256=digest)
    assert state.keys() == payload["model_state"].keys()
    assert all(torch.equal(state[key], payload["model_state"][key]) for key in state)
    assert "optimizer_state" not in state
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("mismatch", ["sha", "architecture", "model_config"])
def test_initialization_mismatch_fails_before_data_or_output(
    initial_checkpoint, mismatch, tmp_path
):
    path, config, _, _ = initial_checkpoint
    if mismatch == "sha":
        config = replace(config, initial_checkpoint_sha256="0" * 64)
    elif mismatch == "architecture":
        config = replace(config, architecture="spatial")
    else:
        config = replace(config, model_config=replace(config.model_config, grid_size=(4, 4)))
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(tmp_path / "missing-data", output, config,
                                     initial_checkpoint=path)
    assert caught.value.reason_code == "initial_checkpoint_mismatch"
    assert not output.exists()


def test_initialization_strictly_rejects_rehashed_nonfinite_weights(initial_checkpoint):
    path, config, payload, _ = initial_checkpoint
    next(iter(payload["model_state"].values())).reshape(-1)[0] = float("nan")
    study._save_checkpoint(path, payload)
    with pytest.raises(FlattenerError) as caught:
        study._initialization(config, path, None)
    assert caught.value.reason_code == "invalid_checkpoint"


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "projective", "surface"])
@pytest.mark.parametrize("declare_source", [False, True])
def test_resolution_initialization_copies_full_state_binds_source_and_runs_high_resolution(
        initial_checkpoint, architecture, declare_source, no_cuda):
    path, source_config, payload, _ = initial_checkpoint
    source_config = replace(
        source_config, architecture=architecture,
        map_mode="analytic" if architecture in ("projective", "surface") else "legacy")
    with torch.random.fork_rng(devices=[]):
        source_model = study._new_model(source_config)
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.add_(0.01)  # Verify that final heads, not only the backbone, transfer.
    payload.update(config=asdict(source_config), model_spec=study._model_spec(source_config),
                   model_state=source_model.state_dict(), optimizer_state={
                       "state": {}, "param_groups": [study._adam_group(
                           source_config, len(list(source_model.parameters())))]})
    digest = study._save_checkpoint(path, payload)
    config = replace(
        source_config, initialization_mode="resolution", seed=source_config.seed + 1,
        model_config=replace(source_config.model_config, input_size=(384, 288)),
        initial_model_config=source_config.model_config if declare_source else None,
        initial_checkpoint_sha256=digest if declare_source else None)
    assert study._config(config) is config
    before = torch.get_rng_state().clone()
    bound, state = study._initialization(config, path, None)
    assert torch.equal(torch.get_rng_state(), before)
    assert bound.initial_model_config == source_config.model_config
    assert bound.initial_checkpoint_sha256 == digest
    assert bound.model_config.input_size == (384, 288)
    assert bound.seed == config.seed and bound.learning_rate == config.learning_rate
    assert state.keys() == payload["model_state"].keys() and "optimizer_state" not in state
    assert all(torch.equal(value, payload["model_state"][key]) for key, value in state.items())
    with torch.random.fork_rng(devices=[]):
        destination = study._new_model(bound)
    destination.load_state_dict(state, strict=True)
    with torch.no_grad():
        _, dense, _ = study._forward_maps(destination.eval(), torch.zeros(1, 3, 384, 288), bound)
    assert dense.shape == (1, *config.output_size, 2) and torch.isfinite(dense).all()
    assert all(parameter.grad is None for parameter in destination.parameters())
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("mismatch", ["grid", "channels", "hidden", "architecture", "sha",
                                      "declared_source"])
def test_resolution_initialization_rejects_every_non_input_change(initial_checkpoint, mismatch):
    path, source_config, _, _ = initial_checkpoint
    model_config = replace(source_config.model_config, input_size=(32, 24))
    config = replace(source_config, initialization_mode="resolution", model_config=model_config)
    if mismatch in ("grid", "channels", "hidden"):
        changes = {"grid": {"grid_size": (4, 3)}, "channels": {"channels": (2, 4, 5)},
                   "hidden": {"hidden": 9}}[mismatch]
        config = replace(config, model_config=replace(model_config, **changes))
    elif mismatch == "architecture":
        config = replace(config, architecture="spatial")
    elif mismatch == "sha":
        config = replace(config, initial_checkpoint_sha256="0" * 64)
    else:
        config = replace(config, initial_model_config=replace(
            source_config.model_config, input_size=(15, 16)))
    before = torch.get_rng_state().clone()
    with pytest.raises(FlattenerError) as caught:
        study._initialization(config, path, None)
    assert caught.value.reason_code == "initial_checkpoint_mismatch"
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("field,value", [
    ("initial_model_config", True), ("initial_model_config", {}),
    ("initial_model_config", "model"), ("rotation_mode", "per_image"),
    ("rotation_mode", None), ("rotation_mode", True),
])
def test_quality_config_rejects_invalid_types_before_device_or_dataset(
        field, value, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_resolve_device", lambda *args: pytest.fail("device admission"))
    config = replace(study.StudyConfig(initialization_mode="resolution"), **{field: value})
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(tmp_path / "missing", tmp_path / "no-output", config)
    assert caught.value.reason_code == "invalid_config"
    assert not (tmp_path / "no-output").exists()


@pytest.mark.parametrize("mode", ["full", "backbone"])
def test_initial_model_config_is_exclusive_to_resolution_mode(mode):
    config = study.StudyConfig(architecture="surface", map_mode="analytic",
                              initialization_mode=mode, initial_model_config=ModelConfig())
    with pytest.raises(FlattenerError) as caught:
        study._config(config)
    assert caught.value.reason_code == "invalid_config"


@pytest.mark.parametrize("resume", [None, "last.pt"])
@pytest.mark.parametrize("bound", ["none", "sha", "config", "both"])
def test_resolution_initialization_requires_source_path_or_bound_resume(
        initial_checkpoint, resume, bound):
    _, source_config, _, digest = initial_checkpoint
    config = replace(source_config, initialization_mode="resolution",
                     model_config=replace(source_config.model_config, input_size=(32, 24)),
                     initial_model_config=(source_config.model_config
                                           if bound in ("config", "both") else None),
                     initial_checkpoint_sha256=digest if bound in ("sha", "both") else None)
    if resume is not None and bound == "both":
        actual, state = study._initialization(config, None, resume)
        assert actual == config and state is None
    else:
        with pytest.raises(FlattenerError) as caught:
            study._initialization(config, None, resume)
        assert caught.value.reason_code == "invalid_initialization"


@pytest.mark.parametrize("resume", [None, "last.pt"])
def test_initialization_path_requirements_fail_before_data_or_output(tmp_path, resume):
    config = study.StudyConfig(device="cpu", initial_checkpoint_sha256="1" * 64)
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(
            tmp_path / "missing", output, config, resume=resume,
            initial_checkpoint=None if resume is None else tmp_path / "initial.pt",
        )
    assert caught.value.reason_code == "invalid_initialization"
    assert not output.exists()


def test_cuda_admission_precedes_initial_checkpoint_read(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("unavailable CUDA attempted initial checkpoint read")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(study, "_read_checkpoint", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(tmp_path / "missing", output,
                                     initial_checkpoint=tmp_path / "initial.pt")
    assert caught.value.reason_code == "cuda_unavailable"
    assert not output.exists()


@pytest.mark.parametrize(
    "field",
    [
        "topology_weight", "curvature_weight", "gradient_clip_norm", "page_weight",
        "boundary_weight", "differential_weight", "page_relative_weight", "orientation_weight",
    ],
)
@pytest.mark.parametrize("value", [-1, True, "1", None, float("nan"), float("inf")])
def test_invalid_geometry_controls_fail_before_device_or_data(field, value, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid geometry controls reached device resolution")

    monkeypatch.setattr(study, "_resolve_device", forbidden)
    config = replace(study.StudyConfig(device="cpu"), **{field: value})
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(tmp_path / "missing-data", output, config)
    assert caught.value.reason_code == "invalid_config"
    assert not output.exists()


@pytest.mark.parametrize("field", ["rotation_probability", "background_probability"])
@pytest.mark.parametrize("value", [-0.01, 1.01, 2, True, False, None, "0.5",
                                    float("nan"), float("inf")])
def test_invalid_rotation_probability_fails_before_device_or_data(
        field, value, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_resolve_device", lambda *args: pytest.fail("device admission"))
    config = study.StudyConfig(**{field: value})
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(tmp_path / "missing", tmp_path / "no-output", config)
    assert caught.value.reason_code == "invalid_config"
    assert not (tmp_path / "no-output").exists()


@pytest.mark.parametrize("value", [0, 0.25, 1])
def test_rotation_probability_accepts_numeric_boundaries_without_cuda(value, no_cuda):
    config = study.StudyConfig(device="cpu", rotation_probability=value, orientation_weight=value,
                              background_probability=value)
    assert study._config(config) is config
    assert config.rotation_probability == config.orientation_weight == value
    assert config.background_probability == value


@pytest.mark.parametrize("value", [0, 0.25, 1])
def test_explicit_geometry_controls_are_preserved_without_device_access(value, no_cuda):
    config = study.StudyConfig(
        device="cpu", topology_weight=value, curvature_weight=value, gradient_clip_norm=value
    )
    assert study._config(config) is config
    assert (config.topology_weight, config.curvature_weight, config.gradient_clip_norm) == (
        value, value, value
    )


@pytest.mark.parametrize("value", [0, 0.25, 1])
def test_page_controls_are_preserved_without_device_access(value, no_cuda):
    config = study.StudyConfig(
        device="cpu", architecture="page", page_weight=value, boundary_weight=value,
        differential_weight=value,
    )
    assert study._config(config) is config
    assert (config.page_weight, config.boundary_weight, config.differential_weight) == (
        value, value, value
    )


@pytest.mark.parametrize("architecture", ["global", "spatial"])
def test_page_supervision_requires_page_model_before_device_data_or_output(
    architecture, monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        pytest.fail("page/model mismatch reached device resolution")

    monkeypatch.setattr(study, "_resolve_device", forbidden)
    config = study.StudyConfig(device="cpu", architecture=architecture, page_weight=0.25)
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(tmp_path / "missing-data", output, config)
    assert caught.value.reason_code == "invalid_config"
    assert not output.exists()


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "projective", "surface"])
def test_coordinate_supervision_controls_accept_all_architectures(architecture, no_cuda):
    config = study.StudyConfig(
        device="cpu", architecture=architecture, page_weight=0,
        boundary_weight=0.5, differential_weight=0.25, page_relative_weight=0.75,
        orientation_weight=0.125, rotation_probability=0.5,
        map_mode="analytic" if architecture in ("projective", "surface") else "legacy",
    )
    assert study._config(config) is config


def test_explicit_cpu_resolution_and_metadata_never_access_cuda(no_cuda):
    config = study.StudyConfig(device="cpu")
    assert study._config(config) is config
    device = study._resolve_device(config.device)
    assert device == torch.device("cpu")
    info = study._device_info(device)
    assert info["device"] == "cpu"
    assert isinstance(info["name"], str) and info["name"]
    assert info["capability"] is None


def test_check_device_cli_accepts_explicit_cpu_without_cuda(no_cuda, capsys):
    assert study.main(["check-device", "--device", "cpu"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["device"] == "cpu"
    assert result["capability"] is None


@pytest.mark.parametrize(
    "device", ["cpu:0", "mps", "auto", "cuda:-1", "cuda:x", "", None, True,
               "xpu:-1", "xpu:x", "xpu:01", "xpu:10000", "xpu:", "xpu:0\n"]
)
def test_unsupported_devices_never_fall_back(device, tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises((FlattenerError, ValueError, TypeError)):
        config = study.StudyConfig(device=device)
        study.train_coordinate_study(tmp_path / "missing-data", output, config)
    assert not output.exists()


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_unavailable_cuda_fails_before_data_or_output(architecture, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(
            tmp_path / "nonexistent-dataset",
            output,
            study.StudyConfig(architecture=architecture),
        )
    assert caught.value.reason_code == "cuda_unavailable"
    assert not output.exists()


@pytest.mark.parametrize("device", ["cuda:1", "cuda:9999"])
def test_absent_device_index_fails_before_data(device, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(FlattenerError):
        study.train_coordinate_study(
            tmp_path, tmp_path / "no-output", study.StudyConfig(device=device)
        )
    assert not (tmp_path / "no-output").exists()


@pytest.fixture
def mocked_xpu(monkeypatch):
    """Never consult a real accelerator, including fork_rng's backend discovery."""
    events = []
    state = SimpleNamespace(current=1, available=True, count=3,
                            rng={i: torch.full((32,), i, dtype=torch.uint8) for i in range(3)})

    def forbidden(*args, **kwargs):
        pytest.fail("mock XPU test attempted an all-device or real accelerator operation")

    @contextmanager
    def device(selected):
        previous = state.current
        state.current = torch.device(selected).index
        events.append(("enter", state.current))
        try:
            yield
        finally:
            events.append(("exit", state.current))
            state.current = previous

    def get_rng_state(index):
        events.append(("get", index))
        return state.rng[index].clone()

    def set_rng_state(value, index):
        events.append(("set", index))
        state.rng[index] = value.clone()

    def manual_seed(seed):
        events.append(("seed", state.current, seed))
        state.rng[state.current].fill_(seed % 256)

    backend = SimpleNamespace(
        is_available=lambda: state.available, device_count=lambda: state.count,
        current_device=lambda: state.current, get_device_name=lambda selected: f"mock {selected}",
        device=device, get_rng_state=get_rng_state, set_rng_state=set_rng_state,
        manual_seed=manual_seed, manual_seed_all=forbidden, get_rng_state_all=forbidden,
        set_rng_state_all=forbidden, _lazy_init=forbidden)
    original_module = torch.get_device_module
    monkeypatch.setattr(torch, "get_device_module",
                        lambda selected: backend if str(selected).split(":")[0] == "xpu"
                        else original_module(selected))
    monkeypatch.setattr(torch, "xpu", backend)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    return backend, state, events


@pytest.mark.parametrize("requested,index", [("xpu", 1), ("xpu:0", 0), ("xpu:2", 2)])
def test_explicit_xpu_config_resolution_and_metadata_use_only_mocked_selected_backend(
        requested, index, mocked_xpu, no_cuda):
    config = study.StudyConfig(device=requested)
    assert study._config(config) is config
    selected = study._resolve_device(requested)
    assert selected == torch.device("xpu", index)
    assert study._device_info(selected) == {
        "device": f"xpu:{index}", "name": f"mock xpu:{index}", "capability": None}
    assert mocked_xpu[2] == []


@pytest.mark.parametrize("requested", ["cpu", "cuda", "mps", "xpu:-1", "xpu:01", None, True])
def test_xpu_specific_resolver_rejects_other_or_malformed_devices_before_backend(
        requested, monkeypatch, no_cuda):
    monkeypatch.setattr(torch, "xpu", None)
    with pytest.raises(FlattenerError) as caught:
        study._require_xpu_device(requested)
    assert caught.value.reason_code == "xpu_required"


@pytest.mark.parametrize("failure", ["unavailable", "missing_backend", "absent_index", "runtime"])
def test_xpu_admission_fails_before_data_checkpoint_or_output(
        failure, mocked_xpu, no_cuda, monkeypatch, tmp_path):
    backend, state, _ = mocked_xpu
    if failure == "unavailable":
        state.available = False
    elif failure == "missing_backend":
        monkeypatch.setattr(torch, "xpu", None)
    elif failure == "absent_index":
        state.count = 2
    else:
        def unavailable():
            raise RuntimeError("mock driver unavailable")
        backend.is_available = unavailable
    monkeypatch.setattr(study, "_read_checkpoint", lambda *args, **kwargs:
                        pytest.fail("XPU admission read an initialization checkpoint"))
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(
            tmp_path / "missing", output, study.StudyConfig(device="xpu:2"),
            initial_checkpoint=tmp_path / "missing.pt")
    assert caught.value.reason_code == "xpu_unavailable"
    assert not output.exists()


@pytest.mark.parametrize("available", [True, False])
def test_xpu_check_device_cli_uses_only_mock_backend(mocked_xpu, no_cuda, capsys, available):
    mocked_xpu[1].available = available
    assert study.main(["check-device", "--device", "xpu:2"]) == (0 if available else 2)
    result = json.loads(capsys.readouterr().out)
    if available:
        assert result == {"device": "xpu:2", "name": "mock xpu:2", "capability": None}
    else:
        assert result["error"] == "xpu_unavailable" and result["cpu_fallback"] is False


def test_resume_cli_checks_xpu_before_checkpoint_read(mocked_xpu, no_cuda, monkeypatch,
                                                    tmp_path, capsys):
    mocked_xpu[1].available = False
    monkeypatch.setattr(study, "coordinate_checkpoint", lambda *args, **kwargs:
                        pytest.fail("unavailable XPU attempted a resume checkpoint read"))
    output = tmp_path / "not-created"
    assert study.main(["train", str(tmp_path / "missing"), str(output), "--device", "xpu:0",
                       "--resume", str(tmp_path / "last.pt")]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["reason_code"] == "xpu_unavailable" and result["cpu_fallback"] is False
    assert not output.exists()


@pytest.mark.parametrize("exceptional", [False, True])
def test_xpu_training_rng_forks_only_selected_mock_state_and_restores_cpu_and_device(
        mocked_xpu, no_cuda, monkeypatch, exceptional):
    _, state, events = mocked_xpu
    original_fork = torch.random.fork_rng
    fork_calls = []

    def recorded_fork(*args, **kwargs):
        fork_calls.append((args, kwargs))
        return original_fork(*args, **kwargs)

    monkeypatch.setattr(torch.random, "fork_rng", recorded_fork)
    cpu_before = torch.get_rng_state().clone()
    xpu_before = {index: value.clone() for index, value in state.rng.items()}
    expected = torch.rand(5, generator=torch.Generator(device="cpu").manual_seed(213))

    class Sentinel(Exception):
        pass

    try:
        with study._training_rng(torch.device("xpu:2"), 213):
            assert state.current == 2
            assert torch.equal(torch.rand(5), expected)
            assert torch.equal(state.rng[2], torch.full((32,), 213, dtype=torch.uint8))
            assert torch.equal(state.rng[0], xpu_before[0])
            assert torch.equal(state.rng[1], xpu_before[1])
            state.rng[2].zero_()
            if exceptional:
                raise Sentinel
    except Sentinel:
        assert exceptional
    assert fork_calls == [((), {"devices": [2], "device_type": "xpu"})]
    assert events == [("enter", 2), ("get", 2), ("seed", 2, 213), ("set", 2), ("exit", 2)]
    assert state.current == 1
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert all(torch.equal(state.rng[index], value) for index, value in xpu_before.items())


def test_explicit_cpu_rng_and_resolution_do_not_consult_xpu(mocked_xpu, no_cuda, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit CPU path consulted XPU")

    for name in vars(mocked_xpu[0]):
        monkeypatch.setattr(mocked_xpu[0], name, forbidden)
    before = torch.get_rng_state().clone()
    expected = torch.rand(5, generator=torch.Generator(device="cpu").manual_seed(27))
    selected = study._resolve_device("cpu")
    assert study._device_info(selected)["device"] == "cpu"
    with study._training_rng(selected, 27):
        assert torch.equal(torch.rand(5), expected)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("execution", ["cpu", "cuda:0", "xpu:0", "xpu:2"])
def test_xpu_backward_guard_rejects_cpu_model_before_batch_optimizer_or_backend(
        execution, mocked_xpu, no_cuda):
    model = torch.nn.Linear(2, 2)
    with pytest.raises(FlattenerError) as caught:
        study._training_step(model, object(), None, study.StudyConfig(device="xpu:2"),
                             (16, 16), execution)
    assert caught.value.reason_code == "xpu_required"
    assert mocked_xpu[2] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("epochs", True),
        ("batch_size", 0),
        ("seed", -1),
        ("learning_rate", float("nan")),
        ("learning_rate", 0),
        ("architecture", "unknown"),
    ],
)
def test_config_rejects_invalid_fields(field, value, tmp_path):
    with pytest.raises((FlattenerError, TypeError, ValueError)):
        config = replace(study.StudyConfig(), **{field: value})
        study.train_coordinate_study(tmp_path, tmp_path / "no-output", config)
    assert not (tmp_path / "no-output").exists()


def test_check_device_cli_is_read_only(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert study.main(["check-device"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "cuda_unavailable"
    assert result["cpu_fallback"] is False


@pytest.mark.parametrize(
    "controls",
    [
        {"stop_after_epoch": -1},
        {"stop_after_epoch": True},
        {"stop_after_epoch": 1.0},
        {"stop_after_epoch": "1"},
        {"stop_after_epoch": 41},
        {"max_seconds": 0},
        {"max_seconds": -1},
        {"max_seconds": True},
        {"max_seconds": float("nan")},
        {"max_seconds": float("inf")},
        {"max_seconds": -float("inf")},
        {"max_seconds": "1"},
        {"cache_max_bytes": -1},
        {"cache_max_bytes": True},
        {"cache_max_bytes": 1.0},
        {"cache_max_bytes": "1"},
        {"cache_max_bytes": None},
    ],
)
def test_invalid_run_controls_fail_before_device_data_or_output(controls, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid run controls reached device resolution")

    monkeypatch.setattr(study, "_resolve_device", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(FlattenerError) as caught:
        study.train_coordinate_study(
            tmp_path / "missing-data", output, study.StudyConfig(device="cpu"), **controls
        )
    assert caught.value.reason_code == "invalid_run_control"
    assert not output.exists()


@pytest.mark.parametrize(
    "controls",
    [
        {"stop_after_epoch": None, "max_seconds": None},
        {"stop_after_epoch": 0},
        {"stop_after_epoch": 1},
        {"stop_after_epoch": 40},
        {"max_seconds": 1},
        {"max_seconds": 0.001},
        {"stop_after_epoch": 2, "max_seconds": 1.5},
        {"cache_max_bytes": 0},
        {"cache_max_bytes": 1024 * 1024},
    ],
)
def test_valid_run_controls_reach_only_device_admission(controls, monkeypatch, tmp_path):
    class AdmissionReached(Exception):
        pass

    def stop_at_admission(device):
        assert device == "cpu"
        raise AdmissionReached

    monkeypatch.setattr(study, "_resolve_device", stop_at_admission)
    output = tmp_path / "not-created"
    with pytest.raises(AdmissionReached):
        study.train_coordinate_study(
            tmp_path / "missing-data", output, study.StudyConfig(device="cpu"), **controls
        )
    assert not output.exists()


def test_train_cli_refuses_automatic_device_without_creating_output(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flattener.learning.coordinate_study",
            "train",
            str(tmp_path / "missing"),
            str(output),
            "--device",
            "auto",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not output.exists()
    assert json.loads(result.stdout)["cpu_fallback"] is False


def test_cli_cache_units_and_geometry_options_reach_only_mocked_trainer(
    monkeypatch, tmp_path, capsys
):
    calls = []
    expected_summary = {
        "epochs_completed": 0,
        "configured_epochs": 40,
        "complete": False,
        "best_epoch": 0,
        "best_validation_epe_pixels": 1.0,
        "run_elapsed_seconds": 0.0,
        "stop_reason": "mocked_admission",
        "checkpoint_available": False,
        "discarded_epoch": None,
        "device": "cpu",
        "include_paper_augmentation": False,
        "initial_checkpoint_sha256": None,
        "initialization_mode": "full", "map_mode": "legacy", "topology_mode": "absolute",
    }

    def no_training(dataset, output, config, **controls):
        calls.append((dataset, output, config, controls))
        return {**expected_summary, "history": []}

    monkeypatch.setattr(study, "train_coordinate_study", no_training)
    output = tmp_path / "not-created"
    assert study.main(
        [
            "train", str(tmp_path / "missing-data"), str(output), "--device", "cpu",
            "--cache-mib", "3", "--topology-weight", "0.5", "--curvature-weight", "0.25",
            "--gradient-clip-norm", "2",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == expected_summary
    assert len(calls) == 1
    _, _, config, controls = calls[0]
    assert config.device == "cpu"
    assert (config.topology_weight, config.curvature_weight, config.gradient_clip_norm) == (
        0.5, 0.25, 2
    )
    assert controls["cache_max_bytes"] == 3 * 1024**2
    assert not output.exists()


def test_cli_page_options_reach_only_mocked_trainer(monkeypatch, tmp_path, capsys):
    calls = []

    def no_training(dataset, output, config, **controls):
        calls.append((dataset, output, config, controls))
        return {
            "history": [], "epochs_completed": 0, "configured_epochs": config.epochs,
            "complete": False, "best_epoch": 0, "best_validation_epe_pixels": 1.0,
            "run_elapsed_seconds": 0.0, "stop_reason": "mocked_admission",
            "checkpoint_available": False, "discarded_epoch": None, "device": config.device,
            "include_paper_augmentation": config.include_paper_augmentation,
            "initial_checkpoint_sha256": config.initial_checkpoint_sha256,
            "initialization_mode": config.initialization_mode,
            "map_mode": config.map_mode, "topology_mode": config.topology_mode,
        }

    monkeypatch.setattr(study, "train_coordinate_study", no_training)
    output = tmp_path / "not-created"
    assert study.main(
        [
            "train", str(tmp_path / "missing-data"), str(output), "--device", "cpu",
            "--architecture", "page", "--page-weight", "0.5", "--boundary-weight", "0.25",
            "--differential-weight", "0.125", "--page-relative-weight", "0.75",
            "--orientation-weight", "0.375", "--rotation-probability", "0.625",
            "--background-probability", "0.375", "--rotation-mode", "per_image_fit",
            "--initialization-mode", "resolution",
            "--include-paper-augmentation",
            "--initial-checkpoint", str(tmp_path / "initial.pt"),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "mocked_admission"
    assert len(calls) == 1
    config = calls[0][2]
    assert config.architecture == "page"
    assert (config.page_weight, config.boundary_weight, config.differential_weight) == (
        0.5, 0.25, 0.125
    )
    assert config.page_relative_weight == 0.75
    assert config.orientation_weight == 0.375 and config.rotation_probability == 0.625
    assert config.background_probability == 0.375 and config.rotation_mode == "per_image_fit"
    assert config.initialization_mode == "resolution" and config.initial_model_config is None
    assert config.include_paper_augmentation is True
    assert calls[0][3]["initial_checkpoint"] == str(tmp_path / "initial.pt")
    assert not output.exists()


def test_resume_cli_recovers_only_declared_initial_sha_without_source_file(
    initial_checkpoint, monkeypatch, tmp_path, capsys
):
    path, _, payload, _ = initial_checkpoint
    payload["config"]["initial_checkpoint_sha256"] = "1" * 64
    study._save_checkpoint(path, payload)
    calls = []

    def no_training(dataset, output, config, **controls):
        calls.append((config, controls))
        return {
            "epochs_completed": 0, "configured_epochs": config.epochs, "complete": False,
            "best_epoch": 0, "best_validation_epe_pixels": 1.0, "run_elapsed_seconds": 0.0,
            "stop_reason": "mocked_admission", "checkpoint_available": False,
            "discarded_epoch": None, "device": config.device,
            "include_paper_augmentation": config.include_paper_augmentation,
            "initial_checkpoint_sha256": config.initial_checkpoint_sha256,
            "initialization_mode": config.initialization_mode,
            "map_mode": config.map_mode, "topology_mode": config.topology_mode,
        }

    monkeypatch.setattr(study, "train_coordinate_study", no_training)
    output = tmp_path / "not-created"
    assert study.main([
        "train", str(tmp_path / "missing"), str(output), "--device", "cpu",
        "--resume", str(path), "--include-paper-augmentation",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "mocked_admission"
    assert len(calls) == 1
    config, controls = calls[0]
    assert config.initial_checkpoint_sha256 == "1" * 64
    assert config.include_paper_augmentation is True
    assert config.model_config == ModelConfig()  # Other options stay explicit.
    assert controls["initial_checkpoint"] is None
    assert controls["resume"] == str(path)
    assert not output.exists()


def test_resume_cli_checks_cuda_before_reading_checkpoint(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("unavailable CUDA attempted resume checkpoint read")

    monkeypatch.setattr(study, "coordinate_checkpoint", forbidden)
    assert study.main([
        "train", str(tmp_path / "missing"), str(tmp_path / "output"),
        "--resume", str(tmp_path / "last.pt"),
    ]) == 2
    assert json.loads(capsys.readouterr().out)["reason_code"] == "cuda_unavailable"


def test_backward_entry_rejects_mixed_devices_before_touching_batch():
    model = torch.nn.Linear(2, 2)
    # The optimizer and batch are deliberately unusable: the guard must run first.
    with pytest.raises(FlattenerError) as caught:
        study._training_step(
            model, object(), None, study.StudyConfig(), (128, 128), "cuda:0"
        )
    assert caught.value.reason_code == "cuda_required"


def test_dense_expansion_matches_existing_unclipped_half_pixel_contract():
    residual = torch.linspace(-3, 3, 2 * 2 * 9 * 7).reshape(2, 2, 9, 7)
    actual = study._dense_maps(residual, (128, 136))
    torch.testing.assert_close(actual, dense_maps(residual, (128, 136)), rtol=0, atol=0)
    assert actual.abs().max() > 1


def test_validation_is_per_image_mean_and_uses_original_source_pixel_scale(monkeypatch):
    config = study.StudyConfig(
        model_config=ModelConfig(input_size=(4, 4), grid_size=(2, 2)), output_size=(2, 2),
        rotation_probability=1, orientation_weight=0.5, background_probability=1,
        rotation_mode="per_image_fit",
    )
    monkeypatch.setattr(study, "_rotate_training_batch",
                        lambda *args: pytest.fail("validation must never augment"))
    monkeypatch.setattr(study, "augment_training_batch",
                        lambda *args, **kwargs: pytest.fail("validation must never augment"))

    class FixedPredictor(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(images.shape[0], 2, 2, 2)

    class Loader(list):
        dataset = range(2)

    target = dense_maps(torch.zeros(2, 2, 2, 2), (2, 2))
    # Non-square source W=200, H=100: shifts are 1px and 9px respectively.
    target[0, ..., 0] += 1 / 100
    target[1, ..., 1] += 9 / 50
    valid = torch.ones(2, 2, 2, dtype=torch.bool)
    valid[0].fill_(False)
    valid[0, 0, 0] = True
    target[~valid] = float("nan")
    batch = {"image": torch.zeros(2, 3, 4, 4), "target": target, "valid": valid}
    metric = study._validation_epe(
        FixedPredictor(), Loader([batch]), config, (100, 200), torch.device("cpu")
    )
    assert metric == pytest.approx(5.0, abs=2e-6)  # Not the pixel-weighted 7.4px.


@pytest.mark.parametrize("translation", [(0, 0), (3, -2)])
def test_topology_loss_preserves_identity_orientation_without_clipping(translation):
    predicted = coarse_identity((7, 5))[None] + torch.tensor(translation)
    valid = torch.ones(1, 7, 5, dtype=torch.bool)
    assert study._topology_loss(predicted, valid).item() == 0


def test_topology_loss_has_independent_reflection_and_per_image_mean_answer():
    predicted = coarse_identity((7, 5))[None].repeat(2, 1, 1, 1)
    predicted[1, ..., 0].neg_()
    valid = torch.ones(2, 7, 5, dtype=torch.bool)
    # det=-1 has (1e-3 - -1)^2 hinge, mean+worst per image, then image mean.
    assert study._topology_loss(predicted, valid).item() == pytest.approx(1.001**2, rel=1e-6)
    assert study._topology_loss(predicted[1:], valid[1:]).item() == pytest.approx(
        2 * 1.001**2, rel=1e-6
    )


def test_topology_loss_keeps_worst_cell_and_uses_only_fixed_gt_corners():
    predicted = coarse_identity((2, 3))[None]
    predicted[..., -1, 0] = predicted[..., 0, 0]
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    # One identity cell and one reversed cell: mean hinge + worst hinge.
    assert study._topology_loss(predicted, valid).item() == pytest.approx(
        1.5 * 1.001**2, rel=1e-6
    )
    valid[..., -1] = False
    assert study._topology_loss(predicted, valid).item() == 0
    valid.zero_()
    assert study._topology_loss(predicted, valid).item() == 0


@pytest.mark.parametrize("shape", [(2, 2), (5, 7)])
def test_curvature_loss_is_zero_for_constant_and_affine_residuals(shape):
    residual = coarse_identity(shape).permute(2, 0, 1)[None]
    assert study._curvature_loss(torch.ones_like(residual) * 4, (80, 200)).item() == 0
    assert study._curvature_loss(residual, (80, 200)).item() == pytest.approx(0, abs=1e-9)


def test_curvature_loss_scales_xy_in_original_source_pixels():
    x_bend = torch.zeros(1, 2, 2, 3)
    x_bend[:, 0, :, 1] = 0.01
    y_bend = torch.zeros_like(x_bend)
    y_bend[:, 1, :, 1] = 0.02
    # W=200,H=100: either bend is one source pixel; second difference is -2px.
    # SmoothL1(-2)=1.5, averaged with the zero second coordinate -> 0.75.
    assert study._curvature_loss(x_bend, (100, 200)).item() == pytest.approx(0.75)
    assert study._curvature_loss(y_bend, (100, 200)).item() == pytest.approx(0.75)


@pytest.mark.parametrize("foreground", [False, True])
def test_page_loss_handles_absent_classes_with_independent_dice_answer(foreground):
    logits = torch.zeros(1, 1, 2, 2)
    source_mask = torch.full((1, 2, 2), foreground, dtype=torch.bool)
    # At probability 1/2, BCE=log(2). Dice sums include a +1 numerator/denominator.
    dice = 2 / 7 if foreground else 2 / 3
    assert study._page_loss(logits, source_mask).item() == pytest.approx(math.log(2) + dice)


def test_page_loss_balances_classes_instead_of_pixel_counts():
    logits = torch.tensor([math.log(3), -math.log(3), -math.log(3), math.log(3)])
    logits = logits.reshape(1, 1, 2, 2)
    source_mask = torch.tensor([[[True, False], [False, False]]])
    positive = math.log(4 / 3)
    negative = (2 * math.log(4 / 3) + math.log(4)) / 3
    # Probabilities are .75,.25,.25,.75; Dice loss=1-(2*.75+1)/(2+1+1).
    expected = (positive + negative) / 2 + 3 / 8
    assert study._page_loss(logits, source_mask).item() == pytest.approx(expected)


def test_page_loss_area_resizes_source_mask_to_fractional_targets():
    source_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    source_mask[:, 0] = True
    # A 1x1 logit receives soft GT=.25, with p=.5 and Dice loss=2/7.
    assert study._page_loss(torch.zeros(1, 1, 1, 1), source_mask).item() == pytest.approx(
        math.log(2) + 2 / 7
    )


def test_page_loss_is_mean_of_per_image_class_and_dice_losses():
    source_mask = torch.zeros(2, 2, 2, dtype=torch.bool)
    source_mask[1] = True
    expected = math.log(2) + (2 / 3 + 2 / 7) / 2
    assert study._page_loss(torch.zeros(2, 1, 2, 2), source_mask).item() == pytest.approx(
        expected
    )


def test_page_loss_keeps_extreme_finite_logits_numerically_stable():
    source_mask = torch.tensor([[[True, False], [False, True]]])
    logits = torch.where(source_mask[:, None], 80.0, -80.0)
    loss = study._page_loss(logits, source_mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0, abs=1e-6)
    wrong = study._page_loss(-logits, source_mask)
    assert torch.isfinite(wrong)
    assert wrong.item() == pytest.approx(80.8, abs=1e-5)


@pytest.mark.parametrize("invalid", ["logit_dtype", "mask_dtype", "channels", "batch_size",
                                      "mask_rank", "nonfinite_logits", "empty_mask"])
def test_page_loss_rejects_invalid_tensor_contracts(invalid):
    logits = torch.zeros(1, 1, 2, 2)
    source_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    if invalid == "logit_dtype":
        logits = logits.double()
    elif invalid == "mask_dtype":
        source_mask = source_mask.float()
    elif invalid == "channels":
        logits = logits.expand(1, 2, 2, 2)
    elif invalid == "batch_size":
        source_mask = source_mask.expand(2, 4, 4)
    elif invalid == "mask_rank":
        source_mask = source_mask[:, None]
    elif invalid == "nonfinite_logits":
        logits[0, 0, 0, 0] = float("nan")
    else:
        source_mask = source_mask[:, :0]
    with pytest.raises(FlattenerError) as caught:
        study._page_loss(logits, source_mask)
    assert caught.value.reason_code == "invalid_training_batch"


@pytest.mark.parametrize("box_size,multiplier", [
    ((100, 200), 0), ((50, 100), 1), ((25, 50), 3), ((5, 10), 19),
])
@pytest.mark.parametrize("axis,coordinate_mean", [(0, 0.75), (1, 0.25)])
def test_page_relative_loss_has_exact_physical_bbox_and_original_xy_scale(
        box_size, multiplier, axis, coordinate_mean):
    predicted = torch.zeros(1, 3, 4, 2)
    predicted[..., axis] = 0.02
    target = torch.zeros_like(predicted)
    valid = torch.ones(1, 3, 4, dtype=torch.bool)
    source_mask = torch.zeros(1, 100, 200, dtype=torch.bool)
    source_mask[:, :box_size[0], :box_size[1]] = True
    # Same normalized .02 offset means x=2px or y=1px; SmoothL1 divided by xy.
    # Physical bbox dimensions use max-min+1, so half-size means w-1=1.
    result = study._page_relative_loss(predicted, target, valid, source_mask, (100, 200))
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(multiplier * coordinate_mean, abs=3e-6)


@pytest.mark.parametrize("photo_size,extra_weight", [
    ((10, 20), math.hypot(10, 20) / math.hypot(2, 2) - 1),
    ((384, 288), 19),
])
def test_page_relative_smallest_valid_bbox_has_physical_support_and_capped_weight(
        photo_size, extra_weight):
    predicted = torch.zeros(1, 2, 2, 2)
    predicted[..., 0] = 4 / photo_size[1]  # Two original source pixels.
    target = torch.zeros_like(predicted)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    source_mask = torch.zeros(1, *photo_size, dtype=torch.bool)
    source_mask[:, -2:, -2:] = True
    assert study._page_relative_loss(
        predicted, target, valid, source_mask, photo_size
    ).item() == pytest.approx(0.75 * extra_weight, abs=3e-6)


def test_page_relative_loss_uses_bbox_not_foreground_area_or_position():
    predicted = torch.zeros(2, 2, 3, 2)
    predicted[..., 0] = 0.01  # One source pixel; per-xy SmoothL1 mean=.25.
    target = torch.zeros_like(predicted)
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    source_mask = torch.zeros(2, 200, 100, dtype=torch.bool).transpose(1, 2)
    assert not source_mask.is_contiguous()
    source_mask[0, 10:60, 30:130] = True
    source_mask[1, 30, 80] = source_mask[1, 79, 179] = True
    # Both masks have a 50x100 bbox despite different positions and occupancy.
    assert study._page_relative_loss(
        predicted, target, valid, source_mask, (100, 200)
    ).item() == pytest.approx(0.25, abs=1e-6)


def test_page_relative_loss_is_macro_not_pixel_micro_and_ignores_invalid_gt_nan():
    predicted = torch.zeros(2, 2, 3, 2)
    target = torch.full_like(predicted, float("nan"))
    valid = torch.zeros(2, 2, 3, dtype=torch.bool)
    valid[0, 0, 0] = True
    valid[1, 0, :3] = True
    target[0, 0, 0] = torch.tensor([0.01, 0])
    target[1, 0, :3] = torch.tensor([0.03, 0])
    predicted[~valid] = 100  # Invalid GT cannot enlarge or shrink either denominator.
    source_mask = torch.zeros(2, 100, 200, dtype=torch.bool)
    source_mask[0, :50, :100] = True
    source_mask[1, :25, :50] = True
    result = study._page_relative_loss(predicted, target, valid, source_mask, (100, 200))
    # (1*.25 + 3*1.25)/2 = 2, not the valid-pixel micro result 2.875.
    assert result.item() == pytest.approx(2.0, abs=1e-6)
    pieces = [study._page_relative_loss(
        predicted[i:i + 1], target[i:i + 1], valid[i:i + 1], source_mask[i:i + 1],
        (100, 200)) for i in range(2)]
    assert result.item() == pytest.approx(torch.stack(pieces).mean().item(), abs=1e-6)


@pytest.mark.parametrize("invalid", [
    "prediction_dtype", "target_dtype", "valid_dtype", "source_dtype", "source_sparse",
    "source_device", "source_shape", "source_batch", "source_rank", "source_empty",
    "source_one_pixel", "source_one_row", "source_one_column", "one_image_empty_gt",
    "prediction_nan", "prediction_inf_outside_gt", "valid_target_nan", "target_shape",
    "invalid_photo_size",
])
def test_page_relative_loss_rejects_invalid_inputs_and_per_image_degeneracy(invalid):
    predicted = torch.zeros(2, 3, 4, 2)
    target = torch.zeros_like(predicted)
    valid = torch.ones(2, 3, 4, dtype=torch.bool)
    source_mask = torch.ones(2, 10, 20, dtype=torch.bool)
    photo_size = (10, 20)
    if invalid == "prediction_dtype":
        predicted = predicted.double()
    elif invalid == "target_dtype":
        target = target.double()
    elif invalid == "valid_dtype":
        valid = valid.float()
    elif invalid == "source_dtype":
        source_mask = source_mask.float()
    elif invalid == "source_sparse":
        source_mask = source_mask.to_sparse()
    elif invalid == "source_device":
        source_mask = source_mask.to("meta")
    elif invalid == "source_shape":
        source_mask = source_mask[:, :, :-1]
    elif invalid == "source_batch":
        source_mask = source_mask[:1]
    elif invalid == "source_rank":
        source_mask = source_mask[:, None]
    elif invalid.startswith("source_"):
        source_mask[0] = False
        if invalid == "source_one_pixel":
            source_mask[0, 3, 4] = True
        elif invalid == "source_one_row":
            source_mask[0, 3] = True
        elif invalid == "source_one_column":
            source_mask[0, :, 4] = True
    elif invalid == "one_image_empty_gt":
        valid[0] = False
        target[0] = float("nan")
    elif invalid == "prediction_nan":
        predicted[0, 0, 0, 0] = float("nan")
    elif invalid == "prediction_inf_outside_gt":
        valid[0, 0, 0] = False
        predicted[0, 0, 0, 0] = float("inf")
    elif invalid == "valid_target_nan":
        target[0, 0, 0, 0] = float("nan")
    elif invalid == "target_shape":
        target = target[:, :, :-1]
    else:
        photo_size = (0, 20)
    with pytest.raises(FlattenerError) as caught:
        study._page_relative_loss(predicted, target, valid, source_mask, photo_size)
    assert caught.value.reason_code == "invalid_training_batch"


def test_disabled_page_relative_weight_preserves_exact_loss_without_source_mask(monkeypatch):
    config = study.StudyConfig(
        device="cpu", output_size=(4, 5), boundary_weight=0.25, differential_weight=0.125,
        model_config=ModelConfig(input_size=(4, 4), grid_size=(2, 2),
                                 channels=(2, 2, 2), hidden=4),
    )
    with torch.random.fork_rng(devices=[]):
        model = study._new_model(config)
    with torch.no_grad():
        model.head[-1].bias.fill_(0.01)
    images = torch.zeros(2, 3, 4, 4)
    target = coarse_identity((4, 5))[None].repeat(2, 1, 1, 1) + torch.tensor([0.02, -0.04])
    valid = torch.ones(2, 4, 5, dtype=torch.bool)
    valid[0, 0, 0] = False
    target[~valid] = float("nan")
    batch = {"image": images, "target": target, "valid": valid}
    with torch.no_grad():
        _, predicted, _ = study._forward_maps(model, images, config)
        expected = study._coordinate_loss(predicted, target, valid, (100, 200))
        expected += 0.25 * study._boundary_loss(predicted, target, valid, (100, 200))
        expected += 0.125 * study._differential_loss(
            predicted, target, valid, (100, 200), config.model_config.grid_size)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    observed, zero_calls = [], []

    class StopBeforeBackward(Exception):
        pass

    class AdmissionOnlyOptimizer:
        param_groups = [{"params": list(model.parameters())}]

        def zero_grad(self, *, set_to_none):
            zero_calls.append(set_to_none)

        def step(self):
            pytest.fail("disabled-loss admission test must never optimize")

    def stop_before_backward(loss, *args, **kwargs):
        observed.append(loss.detach().clone())
        raise StopBeforeBackward

    def forbidden(*args, **kwargs):
        pytest.fail("zero weight must not evaluate page-relative supervision")

    monkeypatch.setattr(study, "_page_relative_loss", forbidden)
    monkeypatch.setattr(study, "_orientation_loss", forbidden)
    monkeypatch.setattr(torch, "rand", forbidden)
    monkeypatch.setattr(torch, "randint", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", stop_before_backward)
    with pytest.raises(StopBeforeBackward):
        study._training_step(model, AdmissionOnlyOptimizer(), batch, config, (100, 200), "cpu")
    assert zero_calls == [True]
    assert len(observed) == 1 and torch.equal(observed[0], expected)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.training
def test_page_relative_loss_has_finite_fixed_mask_gradients_without_optimizer():
    predicted = torch.zeros(2, 2, 3, 2)
    predicted[..., 0] = 0.02
    predicted.requires_grad_()
    target = torch.zeros_like(predicted)
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[0, 0, 0] = False
    target[~valid] = float("nan")
    source_mask = torch.zeros(2, 100, 200, dtype=torch.bool)
    source_mask[0, :50, :100] = True
    source_mask[1] = True  # Full-photo auxiliary weight is exactly zero.
    study._page_relative_loss(predicted, target, valid, source_mask, (100, 200)).backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, ..., 0][valid[0]].abs().min() > 0
    assert not predicted.grad[~valid].count_nonzero()
    assert not predicted.grad[1].count_nonzero()


@pytest.mark.parametrize("scale,expected", [(0.0, 1.0), (0.125, 0.5), (0.5, 0.0),
                                           (2.0, 0.0), (-1.0, 2.0)])
def test_orientation_loss_has_independent_collapse_scale_and_reversal_answers(scale, expected):
    target = coarse_identity((2, 2))[None]  # Each edge is 4px for an 8x8 source.
    predicted = target * scale
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    assert study._orientation_loss(predicted, target, valid, (8, 8), (2, 2)).item() == (
        pytest.approx(expected, abs=1e-6))


def test_orientation_loss_uses_original_source_xy_metric_not_normalized_angles():
    target = torch.zeros(1, 2, 3, 2)
    target[:] = (torch.arange(3) * 0.02)[None, :, None]
    predicted = target.clone()
    predicted[..., 1].neg_()
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    # Source-pixel vectors (2,1) and (2,-1): cosine=3/5, loss=2/5.
    assert study._orientation_loss(predicted, target, valid, (100, 200), (2, 3)).item() == (
        pytest.approx(0.4, abs=1e-6))


def test_orientation_loss_uses_fixed_gt_one_pixel_threshold_not_prediction_visibility():
    target = torch.zeros(1, 2, 3, 2)
    target[..., 0] = torch.tensor([0.0, 0.125, 0.375])
    predicted = torch.zeros_like(target)
    predicted[..., 0] = torch.tensor([0.0, 100.0, 99.0])
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    # The first GT edge is .5px (ignored), second exactly 1px (eligible).
    # Out-of-photo predictions remain eligible and point in the opposite direction.
    assert study._orientation_loss(predicted, target, valid, (8, 8), (2, 3)).item() == 2


def test_orientation_loss_is_macro_over_eligible_images_not_stencil_counts():
    target = torch.zeros(3, 2, 3, 2)
    target[0, ..., 0] = torch.tensor([0.0, 0.5, 1.0])
    target[1, ..., 0] = torch.tensor([0.0, 0.5, 0.5])
    predicted = target.clone()
    predicted[1].neg_()
    valid = torch.ones(3, 2, 3, dtype=torch.bool)
    valid[1, 1] = False
    target[~valid] = float("nan")
    # Image 0: four aligned edges. Image 1: one reversed edge. Image 2: no GT edges.
    assert study._orientation_loss(predicted, target, valid, (8, 8), (2, 3)).item() == (
        pytest.approx(1.0, abs=1e-6))  # Not stencil-micro .4 or three-image mean 2/3.


def test_orientation_loss_excludes_fractionally_invalid_bilinear_coarse_endpoints():
    target = coarse_identity((4, 4))[None]
    predicted = target.clone()
    predicted[:, :2, :2] = 100
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[0, 0, 0] = False
    target[~valid] = float("nan")
    # Coarse top-left validity=.75, so both incident stencils must disappear.
    # Other stencils retain exact GT correspondence, despite the invalid GT NaN.
    assert study._orientation_loss(predicted, target, valid, (100, 200), (2, 2)).item() == (
        pytest.approx(0.0, abs=1e-6))


@pytest.mark.parametrize("invalid", ["prediction_dtype", "target_dtype", "valid_dtype",
                                    "prediction_nan", "valid_target_nan", "large_grid"])
def test_orientation_loss_rejects_invalid_tensor_and_grid_contracts(invalid):
    target = coarse_identity((2, 3))[None]
    predicted = target.clone()
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    grid = (2, 3)
    if invalid == "prediction_dtype":
        predicted = predicted.double()
    elif invalid == "target_dtype":
        target = target.double()
    elif invalid == "valid_dtype":
        valid = valid.float()
    elif invalid == "prediction_nan":
        predicted[0, 0, 0] = float("nan")
    elif invalid == "valid_target_nan":
        target[0, 0, 0] = float("nan")
    else:
        grid = (3, 4)
    with pytest.raises(FlattenerError) as caught:
        study._orientation_loss(predicted, target, valid, (100, 200), grid)
    assert caught.value.reason_code == "invalid_training_batch"


@pytest.mark.training
@pytest.mark.parametrize("case", ["quarter_turn", "half_turn", "no_eligible"])
def test_orientation_gradients_are_finite_with_expected_half_turn_stationarity(case):
    target = coarse_identity((2, 2))[None]
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    predicted = torch.stack((target[..., 1], -target[..., 0]), dim=-1)
    if case == "half_turn":
        predicted = -target
    elif case == "no_eligible":
        valid[:] = False
        target[:] = float("nan")
    predicted.requires_grad_()
    loss = study._orientation_loss(predicted, target, valid, (8, 8), (2, 2))
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    if case == "quarter_turn":
        assert loss.item() == pytest.approx(1.0)
        assert predicted.grad.abs().max() > 0
    elif case == "half_turn":
        assert loss.item() == pytest.approx(2.0)
        assert predicted.grad.abs().max().item() == pytest.approx(0, abs=1e-7)
    else:
        assert loss.item() == 0 and not predicted.grad.count_nonzero()


def _quarter_turn_batch():
    y, x = torch.meshgrid(torch.arange(2), torch.arange(4), indexing="ij")
    image = torch.stack((0.1 + 0.1 * x, 0.2 + 0.3 * y, 0.01 + 0.02 * x + 0.04 * y))
    target = coarse_identity((2, 3))[None].repeat(2, 1, 1, 1)
    target += torch.tensor([0.2, -0.1])
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[0, 0, 0] = False
    target[~valid] = float("nan")
    mask = torch.arange(60).reshape(6, 10).remainder(7) < 2
    return {"image": torch.stack((image, 1 - image)), "target": target, "valid": valid,
            "source_mask": torch.stack((mask, ~mask)), "id": ["left", "right"]}


@pytest.mark.parametrize("rotation_mode", ["batch", "per_image_fit"])
@pytest.mark.parametrize("rotation_probability,background_probability", [
    (0, 0), (0.5, 0), (0, 0.75), (0.5, 0.75),
])
def test_quality_augmentation_dispatch_order_mask_routing_and_zero_probability_identity(
        monkeypatch, rotation_mode, rotation_probability, background_probability):
    config = study.StudyConfig(
        device="cpu", output_size=(2, 3), rotation_mode=rotation_mode,
        rotation_probability=rotation_probability, background_probability=background_probability,
        model_config=ModelConfig(input_size=(2, 4), grid_size=(2, 2),
                                 channels=(2, 2, 2), hidden=4))
    batch = _quarter_turn_batch()
    original = deepcopy(batch)
    model = torch.nn.Linear(1, 1)
    calls, observed, forwarded = [], [], []
    modern = bool(background_probability
                  or rotation_probability and rotation_mode == "per_image_fit")
    legacy = bool(rotation_probability and rotation_mode == "batch")

    def augment(value, *, photo_size, rotation_probability, background_probability,
                _validated=False, _checks=None):
        assert _validated and photo_size == (6, 10) and isinstance(_checks, list)
        assert ("source_mask" in value) == bool(background_probability)
        if background_probability:
            assert torch.equal(value["source_mask"], original["source_mask"])
            assert value["source_mask"].device == value["image"].device
        calls.append(("modern", rotation_probability, background_probability))
        _checks.append(torch.tensor(True))
        return {**value, "image": value["image"] * 0.75, "target": value["target"] + 0.01}

    def rotate(value, probability, *, _validated=False, _checks=None):
        assert _validated and isinstance(_checks, list)
        assert ("source_mask" in value) == bool(background_probability)
        if modern:
            assert calls == [("modern", 0.0, background_probability)]
        calls.append(("legacy", probability))
        _checks.append(torch.tensor(True))
        return {**value, "image": value["image"] * 0.5, "target": value["target"] + 0.02}

    def forward(model, images, config, *, _checks=None):
        forwarded.append(images.clone())
        _checks.append(torch.tensor(True))
        return None, torch.zeros_like(batch["target"]), None

    class StopBeforeBackward(Exception):
        pass

    class AdmissionOnlyOptimizer:
        param_groups = [{"params": list(model.parameters())}]

        def zero_grad(self, *, set_to_none):
            assert set_to_none

        def step(self):
            pytest.fail("augmentation dispatch must not optimize")

    def stop(loss, *args, **kwargs):
        observed.append(loss.detach().clone())
        raise StopBeforeBackward

    monkeypatch.setattr(study, "augment_training_batch", augment)
    monkeypatch.setattr(study, "_rotate_training_batch", rotate)
    monkeypatch.setattr(study, "_forward_maps", forward)
    monkeypatch.setattr(torch.Tensor, "backward", stop)
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: pytest.fail("unexpected dispatch RNG"))
    monkeypatch.setattr(torch, "randint", lambda *a, **kw: pytest.fail("unexpected dispatch RNG"))
    rng = torch.get_rng_state().clone()
    with pytest.raises(StopBeforeBackward):
        study._training_step(model, AdmissionOnlyOptimizer(), batch, config, (6, 10), "cpu")
    expected_calls = []
    target, image = original["target"], original["image"]
    if modern:
        expected_calls.append(("modern", rotation_probability if rotation_mode == "per_image_fit"
                               else 0.0, background_probability))
        target, image = target + 0.01, image * 0.75
    if legacy:
        expected_calls.append(("legacy", rotation_probability))
        target, image = target + 0.02, image * 0.5
    assert calls == expected_calls
    assert len(forwarded) == len(observed) == 1 and torch.equal(forwarded[0], image)
    expected = study._coordinate_loss(torch.zeros_like(target), target, batch["valid"], (6, 10))
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    for key in ("image", "target", "valid", "source_mask"):
        torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0, equal_nan=True)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("rotation_mode", ["batch", "per_image_fit"])
@pytest.mark.parametrize("fault", ["missing", "none", "dtype", "shape", "failed_check"])
def test_background_without_page_loss_requires_mask_and_rejects_deferred_failure(
        monkeypatch, rotation_mode, fault):
    config = study.StudyConfig(
        device="cpu", output_size=(2, 3), background_probability=0.5,
        rotation_mode=rotation_mode,
        model_config=ModelConfig(input_size=(2, 4), grid_size=(2, 2),
                                 channels=(2, 2, 2), hidden=4))
    assert config.page_weight == config.page_relative_weight == 0
    batch, calls = _quarter_turn_batch(), []
    if fault == "missing":
        del batch["source_mask"]
    elif fault == "none":
        batch["source_mask"] = None
    elif fault == "dtype":
        batch["source_mask"] = batch["source_mask"].float()
    elif fault == "shape":
        batch["source_mask"] = batch["source_mask"][:, :-1]
    model = torch.nn.Linear(1, 1)

    def augment(value, *, _checks=None, **kwargs):
        assert fault == "failed_check", "invalid source mask reached augmentation"
        calls.append("augment")
        _checks.append(torch.tensor(False))
        return value

    def forward(model, images, config, *, _checks=None):
        calls.append("forward")
        _checks.append(torch.tensor(True))
        return None, torch.zeros_like(batch["target"]), None

    class AdmissionOnlyOptimizer:
        param_groups = [{"params": list(model.parameters())}]

        def zero_grad(self, *, set_to_none):
            assert set_to_none

        def step(self):
            pytest.fail("invalid augmentation must not optimize")

    monkeypatch.setattr(study, "augment_training_batch", augment)
    monkeypatch.setattr(study, "_forward_maps", forward)
    with pytest.raises(FlattenerError) as caught:
        study._training_step(model, AdmissionOnlyOptimizer(), batch, config, (6, 10), "cpu")
    assert caught.value.reason_code == ("nonfinite_training" if fault == "failed_check"
                                        else "invalid_training_batch")
    assert calls == (["augment", "forward"] if fault == "failed_check" else [])
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
@pytest.mark.parametrize("include_mask", [False, True])
def test_forced_quarter_turn_preserves_xy_correspondence_and_masks_without_mutation(
        monkeypatch, quarter_turns, include_mask):
    batch = _quarter_turn_batch()
    if not include_mask:
        del batch["source_mask"]
    before = {key: value.clone() if isinstance(value, torch.Tensor) else value
              for key, value in batch.items()}
    calls = []

    def decision(*args, **kwargs):
        assert str(kwargs.get("device", "cpu")) == "cpu"
        calls.append("decision")
        return torch.tensor(0.25)

    def rotation(low, high, size, **kwargs):
        assert (low, high, size) == (1, 4, ())
        assert str(kwargs.get("device", "cpu")) == "cpu"
        calls.append("rotation")
        return torch.tensor(quarter_turns)

    monkeypatch.setattr(torch, "rand", decision)
    monkeypatch.setattr(torch, "randint", rotation)
    actual = study._rotate_training_batch(batch, 0.5)

    def rotated(value):
        if quarter_turns == 1:
            return value.transpose(-2, -1).flip(-2)
        if quarter_turns == 2:
            return value.flip((-2, -1))
        return value.transpose(-2, -1).flip(-1)

    expected_image = torch.nn.functional.interpolate(
        rotated(before["image"]), size=(2, 4), mode="bilinear", align_corners=False)
    x, y = before["target"].unbind(-1)
    expected_target = torch.stack({1: (y, -x), 2: (-x, -y), 3: (-y, x)}[quarter_turns], dim=-1)
    assert actual is not batch and actual.keys() == batch.keys()
    assert calls == ["decision", "rotation"]  # One common rotation, not one draw per image.
    torch.testing.assert_close(actual["image"], expected_image, rtol=0, atol=0)
    torch.testing.assert_close(actual["target"], expected_target, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(actual["valid"], before["valid"])
    assert actual["id"] == before["id"]
    if include_mask:
        expected_mask = torch.nn.functional.interpolate(
            rotated(before["source_mask"][:, None].float()),
            size=(6, 10), mode="nearest-exact")[:, 0].bool()
        assert actual["source_mask"].dtype == torch.bool
        assert torch.equal(actual["source_mask"], expected_mask)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(batch[key], value, rtol=0, atol=0, equal_nan=True)


def test_zero_probability_preserves_batch_identity_without_rng_or_source_mask_access(monkeypatch):
    class GuardedBatch(dict):
        def get(self, key, *args):
            if key == "source_mask":
                pytest.fail("zero probability accessed source_mask")
            return super().get(key, *args)

    batch = GuardedBatch(_quarter_turn_batch())
    monkeypatch.setattr(torch, "rand", lambda *args, **kw: pytest.fail("unexpected RNG"))
    monkeypatch.setattr(torch, "randint", lambda *args, **kw: pytest.fail("unexpected RNG"))
    assert study._rotate_training_batch(batch, 0) is batch


def test_unselected_rotation_consumes_only_decision_and_returns_independent_tensors(monkeypatch):
    batch = _quarter_turn_batch()
    calls = []

    def decision(*args, **kwargs):
        calls.append(True)
        return torch.tensor(0.5)

    monkeypatch.setattr(torch, "rand", decision)
    monkeypatch.setattr(torch, "randint", lambda *args, **kw: pytest.fail("unselected rotation"))
    actual = study._rotate_training_batch(batch, 0.5)
    assert actual is not batch and actual.keys() == batch.keys()
    for key in ("image", "target", "valid", "source_mask"):
        torch.testing.assert_close(actual[key], batch[key], rtol=0, atol=0, equal_nan=True)
        assert actual[key].data_ptr() != batch[key].data_ptr()
    assert calls == [True]


def test_rotation_uses_reproducible_cpu_global_rng_draws():
    batch = _quarter_turn_batch()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        before = torch.get_rng_state().clone()
        torch.rand((), device="cpu")
        torch.randint(1, 4, (), device="cpu")
        expected_after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        first = study._rotate_training_batch(batch, 1.0)
        assert torch.equal(torch.get_rng_state(), expected_after)
        torch.set_rng_state(before)
        repeated = study._rotate_training_batch(batch, 1.0)
        for key in ("image", "target", "valid", "source_mask"):
            torch.testing.assert_close(first[key], repeated[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("quarter_turns,first_expected", [
    (1, [[False, False, True, True], [False, False, False, False]]),
    (3, [[True, True, True, True], [False, False, True, True]]),
])
def test_rotation_mask_uses_independent_pixel_center_nearest_indices(
        monkeypatch, quarter_turns, first_expected):
    batch = _quarter_turn_batch()
    mask = torch.tensor([[False, True, False, True], [False, True, True, False]])
    batch["source_mask"] = torch.stack((mask, ~mask))
    before = batch["source_mask"].clone()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(quarter_turns))
    actual = study._rotate_training_batch(batch, 1.0)["source_mask"]
    # Rotate with indexing, then derive resize indices independently from pixel centers:
    # source_index = floor((output_index + .5) * input_size / output_size).
    rotated = before.transpose(-2, -1).flip(-2 if quarter_turns == 1 else -1)
    indices = []
    for incoming, outgoing in zip(rotated.shape[-2:], before.shape[-2:], strict=True):
        index = torch.floor((torch.arange(outgoing, dtype=torch.float64) + 0.5)
                            * incoming / outgoing).long().clamp(0, incoming - 1)
        indices.append(index)
    expected = rotated.index_select(-2, indices[0]).index_select(-1, indices[1])
    assert torch.equal(expected[0], torch.tensor(first_expected))
    assert torch.equal(actual, expected) and actual.dtype == torch.bool
    assert torch.equal(batch["source_mask"], before)
    legacy = torch.nn.functional.interpolate(rotated[:, None].float(),
                                             size=(2, 4), mode="nearest")[:, 0].bool()
    assert not torch.equal(legacy, expected)  # This asymmetric boundary exposes the old offset.


@pytest.mark.parametrize("invalid", ["image_dtype", "image_nan", "image_range", "target_dtype",
                                    "target_none", "target_list", "target_nan",
                                    "valid_dtype", "empty_gt", "source_none",
                                    "source_dtype", "source_device", "source_shape"])
def test_rotation_rejects_bad_inputs_before_sampling_rng(monkeypatch, invalid):
    batch = _quarter_turn_batch()
    if invalid == "image_dtype":
        batch["image"] = batch["image"].double()
    elif invalid == "image_nan":
        batch["image"][0, 0, 0, 0] = float("nan")
    elif invalid == "image_range":
        batch["image"][0, 0, 0, 0] = 1.01
    elif invalid == "target_dtype":
        batch["target"] = batch["target"].double()
    elif invalid == "target_none":
        batch["target"] = None
    elif invalid == "target_list":
        batch["target"] = []
    elif invalid == "target_nan":
        batch["target"][1, 0, 0] = float("nan")
    elif invalid == "valid_dtype":
        batch["valid"] = batch["valid"].float()
    elif invalid == "empty_gt":
        batch["valid"][0] = False
    elif invalid == "source_none":
        batch["source_mask"] = None
    elif invalid == "source_dtype":
        batch["source_mask"] = batch["source_mask"].float()
    elif invalid == "source_device":
        batch["source_mask"] = batch["source_mask"].to("meta")
    else:
        batch["source_mask"] = batch["source_mask"][:, None]
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: pytest.fail("invalid input RNG"))
    with pytest.raises(FlattenerError) as caught:
        study._rotate_training_batch(batch, 0.5)
    assert caught.value.reason_code == "invalid_training_batch"


def test_rotation_clamps_only_rgb_interpolation_roundoff_not_target_coordinates(monkeypatch):
    batch = _quarter_turn_batch()
    batch["image"] = torch.ones(2, 3, 23, 17)
    batch["target"][batch["valid"]] += 5
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(1))
    actual = study._rotate_training_batch(batch, 1)
    assert actual["image"].min() >= 0 and actual["image"].max() <= 1
    torch.testing.assert_close(actual["image"], torch.ones_like(actual["image"]),
                               rtol=0, atol=8 * torch.finfo(torch.float32).eps)
    assert actual["target"][actual["valid"]].abs().min() > 1


@pytest.mark.parametrize("axis", [0, 1])
def test_boundary_loss_uses_original_xy_pixel_scale(axis):
    predicted = torch.zeros(1, 20, 20, 2)
    target = torch.zeros_like(predicted)
    predicted[..., axis] = 2 / (100 if axis == 0 else 50)
    valid = torch.ones(1, 20, 20, dtype=torch.bool)
    # A 2px error has SmoothL1=1.5; the second coordinate contributes zero.
    assert study._boundary_loss(predicted, target, valid, (100, 200)).item() == pytest.approx(
        0.75, abs=1e-6
    )


@pytest.mark.parametrize("pixel", [(7, 8), (8, 7), (12, 8), (8, 12)])
def test_boundary_loss_uses_outer_eight_rows_and_columns_only(pixel):
    predicted = torch.zeros(1, 20, 20, 2)
    target = torch.full_like(predicted, float("nan"))
    valid = torch.zeros(1, 20, 20, dtype=torch.bool)
    # The central 4x4 square is outside the boundary band, even with valid GT.
    valid[:, 8:12, 8:12] = True
    target[valid] = 100
    assert study._boundary_loss(predicted, target, valid, (100, 200)).item() == 0
    row, col = pixel
    valid[0, row, col] = True
    target[0, row, col] = torch.tensor([0.02, 0])
    assert study._boundary_loss(predicted, target, valid, (100, 200)).item() == pytest.approx(
        0.75, abs=1e-6
    )


def test_boundary_loss_averages_all_fixed_gt_coordinate_elements():
    predicted = torch.zeros(2, 20, 20, 2)
    target = torch.full_like(predicted, float("nan"))
    valid = torch.zeros(2, 20, 20, dtype=torch.bool)
    valid[0, 0, 0] = True
    target[0, 0, 0] = torch.tensor([0.01, 0])
    valid[1, 0, :3] = True
    target[1, 0, :3] = torch.tensor([0.03, 0])
    # One 1px error and three 3px errors: (.5 + 3*2.5) / (4 pixels * 2 axes).
    assert study._boundary_loss(predicted, target, valid, (100, 200)).item() == pytest.approx(
        1.0, abs=1e-6
    )


def test_boundary_loss_empty_fixed_gt_returns_finite_zero():
    predicted = torch.zeros(1, 3, 4, 2)
    target = torch.full_like(predicted, float("nan"))
    valid = torch.zeros(1, 3, 4, dtype=torch.bool)
    assert study._boundary_loss(predicted, target, valid, (100, 200)).item() == 0


def test_differential_loss_is_zero_for_constant_coordinate_error():
    target = coarse_identity((3, 4))[None]
    predicted = target + torch.tensor([0.5, -0.25])
    valid = torch.ones(1, 3, 4, dtype=torch.bool)
    assert study._differential_loss(
        predicted, target, valid, (100, 200), (3, 4)
    ).item() == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("axis", [0, 1])
def test_differential_loss_has_independent_first_and_second_stencil_denominator(axis):
    predicted = torch.zeros(1, 3, 4, 2)
    if axis == 0:
        predicted[..., 0] = torch.arange(4) / 100
    else:
        predicted[..., 1] = torch.arange(3)[:, None] / 50
    target = torch.zeros_like(predicted)
    valid = torch.ones(1, 3, 4, dtype=torch.bool)
    # Eligible stencil counts: x1=9,y1=8,x2=6,y2=4, each with two coordinates.
    # The nonzero 1px first differences contribute .5 each; second differences=0.
    expected = (9 if axis == 0 else 8) * 0.5 / 54
    assert study._differential_loss(
        predicted, target, valid, (100, 200), (3, 4)
    ).item() == pytest.approx(expected, abs=1e-6)


def test_differential_loss_penalizes_two_axis_curvature_with_independent_answer():
    predicted = torch.zeros(1, 3, 3, 2)
    predicted[0, 1, 1, 0] = 0.02
    target = torch.zeros_like(predicted)
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    # Four first differences of magnitude 2px and two second differences of 4px.
    # Sum=4*1.5+2*3.5=13 over (6+6+3+3)*2=36 coordinate elements.
    assert study._differential_loss(
        predicted, target, valid, (100, 200), (3, 3)
    ).item() == pytest.approx(13 / 36, abs=1e-6)


def test_differential_loss_removes_stencils_crossing_invalid_gt_holes():
    predicted = torch.zeros(1, 3, 3, 2)
    predicted[0, 1, 1] = 10
    target = torch.zeros_like(predicted)
    target[0, 1, 1] = float("nan")
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    valid[0, 1, 1] = False
    assert study._differential_loss(predicted, target, valid, (100, 200), (3, 3)).item() == 0


def test_differential_loss_requires_fully_valid_bilinear_coarse_samples():
    predicted = torch.zeros(1, 4, 4, 2)
    predicted[:, :2, :2, 0] = 10
    target = torch.zeros_like(predicted)
    target[0, 0, 1] = float("nan")
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[0, 0, 1] = False
    # The top-left coarse sample has GT validity .75 and contributes no stencils.
    assert study._differential_loss(predicted, target, valid, (100, 200), (2, 2)).item() == 0


def test_differential_loss_samples_dense_maps_on_requested_coarse_grid():
    predicted = torch.zeros(1, 6, 8, 2)
    predicted[..., 0] = torch.arange(8) / 200
    target = torch.zeros_like(predicted)
    valid = torch.ones(1, 6, 8, dtype=torch.bool)
    # Half-pixel resize makes coarse adjacent x errors differ by one source pixel.
    assert study._differential_loss(
        predicted, target, valid, (100, 200), (3, 4)
    ).item() == pytest.approx(1 / 12, abs=1e-6)


@pytest.mark.parametrize("one_valid_pixel", [False, True])
def test_differential_loss_without_valid_stencils_returns_finite_zero(one_valid_pixel):
    predicted = torch.zeros(1, 3, 3, 2)
    target = torch.full_like(predicted, float("nan"))
    valid = torch.zeros(1, 3, 3, dtype=torch.bool)
    if one_valid_pixel:
        valid[0, 0, 0] = True
        target[0, 0, 0] = 0
    assert study._differential_loss(predicted, target, valid, (100, 200), (3, 3)).item() == 0


@pytest.mark.parametrize("helper", ["_coordinate_loss", "_boundary_loss", "_differential_loss"])
@pytest.mark.parametrize("invalid", ["prediction_dtype", "target_dtype", "mask_dtype",
                                      "shape", "prediction_nan", "valid_target_nan"])
def test_coordinate_supervision_rejects_invalid_tensor_contracts(helper, invalid):
    predicted = torch.zeros(1, 3, 3, 2)
    target = torch.zeros_like(predicted)
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    if invalid == "prediction_dtype":
        predicted = predicted.double()
    elif invalid == "target_dtype":
        target = target.double()
    elif invalid == "mask_dtype":
        valid = valid.float()
    elif invalid == "shape":
        target = target[..., :1]
    elif invalid == "prediction_nan":
        predicted[0, 0, 0, 0] = float("nan")
    else:
        target[0, 0, 0, 0] = float("nan")
    arguments = [predicted, target, valid, (100, 200)]
    if helper == "_differential_loss":
        arguments.append((3, 3))
    with pytest.raises(FlattenerError) as caught:
        getattr(study, helper)(*arguments)
    assert caught.value.reason_code == "invalid_training_batch"


def _fixed_shape_loss_fixture():
    target = coarse_identity((7, 5))[None].repeat(2, 1, 1, 1).transpose(1, 2)
    predicted = target + torch.linspace(-0.04, 0.08, target.numel()).reshape(target.shape)
    valid = torch.ones(2, 7, 5, dtype=torch.bool).transpose(1, 2)
    valid[0, 1::2, 1::2] = False
    valid[1, 2:, 2:] = False
    target[~valid] = float("nan")
    return predicted, target, valid


def _boolean_smooth_l1_mean(predicted, target, valid, photo_size):
    delta = (predicted[valid] - target[valid]) * predicted.new_tensor(
        (photo_size[1] / 2, photo_size[0] / 2))
    absolute = delta.abs()
    penalty = torch.where(absolute < 1, 0.5 * delta.square(), absolute - 0.5)
    return penalty.sum() / max(delta.numel(), 1)


@pytest.mark.parametrize("helper", ["_coordinate_loss", "_boundary_loss"])
@pytest.mark.parametrize("padding", [float("nan"), float("inf"), -float("inf")])
def test_fixed_shape_coordinate_losses_match_boolean_oracle_with_nonfinite_gt_padding(
        helper, padding):
    predicted, target, valid = _fixed_shape_loss_fixture()
    target[~valid] = padding
    predicted[~valid] = torch.finfo(torch.float32).max
    assert not valid.is_contiguous() and not target.is_contiguous()
    expected = _boolean_smooth_l1_mean(predicted, target, valid, (100, 200))
    result = getattr(study, helper)(predicted, target, valid, (100, 200))
    torch.testing.assert_close(result, expected, rtol=2e-6, atol=1e-6)
    assert torch.isfinite(result)


def test_fixed_shape_coordinate_loss_uses_valid_xy_micro_denominator():
    predicted = torch.zeros(2, 2, 3, 2)
    target = torch.full_like(predicted, float("nan"))
    valid = torch.zeros(2, 2, 3, dtype=torch.bool)
    valid[0, 0, 0] = True
    valid[1, 0] = True
    target[0, 0, 0] = torch.tensor([0.01, 0.0])
    target[1, 0] = torch.tensor([0.03, 0.0])
    # Per-xy means .25 and 1.25, with 1 and 3 valid pixels: micro=1, macro=.75.
    assert study._coordinate_loss(predicted, target, valid, (100, 200)).item() == (
        pytest.approx(1.0, abs=1e-6))
    with pytest.raises(FlattenerError) as caught:
        study._coordinate_loss(predicted, target, torch.zeros_like(valid), (100, 200))
    assert caught.value.reason_code == "invalid_training_batch"


@pytest.mark.parametrize("grid", [(5, 7), (3, 4), (1, 1)])
def test_fixed_shape_differential_loss_matches_boolean_stencil_reference(grid):
    predicted, target, valid = _fixed_shape_loss_fixture()
    truth = torch.where(valid[..., None], target, 0.0)
    actual, truth = [torch.nn.functional.interpolate(
        value.permute(0, 3, 1, 2), size=grid, mode="bilinear", align_corners=False)
        for value in (predicted, truth)]
    mask = torch.nn.functional.interpolate(valid[:, None].float(), size=grid,
                                          mode="bilinear", align_corners=False)[:, 0] >= 1 - 1e-6
    error = ((actual - truth) * torch.tensor([100.0, 50.0])[None, :, None, None])
    error = error.permute(0, 2, 3, 1)
    penalties = []
    for axis in (1, 2):
        current, eligible = error, mask
        for order in (1, 2):
            if grid[axis - 1] <= order:
                continue
            current = torch.diff(current, dim=axis)
            eligible = eligible.narrow(axis, 0, eligible.shape[axis] - 1) & eligible.narrow(
                axis, 1, eligible.shape[axis] - 1)
            selected = current[eligible]
            penalties.append(torch.where(selected.abs() < 1, 0.5 * selected.square(),
                                         selected.abs() - 0.5).flatten())
    expected = (torch.cat(penalties).sum() / max(sum(p.numel() for p in penalties), 1)
                if penalties else torch.tensor(0.0))
    actual = study._differential_loss(predicted, target, valid, (100, 200), grid)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("helper", [
    "_coordinate_loss", "_boundary_loss", "_differential_loss", "_orientation_loss",
    "_relative_topology_loss", "_topology_loss", "_curvature_loss", "_page_loss",
    "_page_relative_loss",
])
def test_loss_public_and_admitted_fast_paths_agree_and_public_checks_all_predictions(helper):
    predicted, target, valid = _fixed_shape_loss_fixture()
    source_mask = torch.zeros(2, 20, 30, dtype=torch.bool)
    source_mask[:, :10, :15] = True
    if helper == "_topology_loss":
        arguments = [predicted, valid]
    elif helper == "_relative_topology_loss":
        arguments = [predicted, target, valid]
    elif helper == "_curvature_loss":
        arguments = [predicted.permute(0, 3, 1, 2), (20, 30)]
    elif helper == "_page_loss":
        arguments = [predicted[..., :1].permute(0, 3, 1, 2), source_mask]
    else:
        arguments = [predicted, target, valid]
        if helper == "_page_relative_loss":
            arguments.append(source_mask)
        arguments.append((20, 30))
        if helper in ("_differential_loss", "_orientation_loss"):
            arguments.append((3, 4))
    function = getattr(study, helper)
    torch.testing.assert_close(function(*arguments), function(*arguments, _validated=True),
                               rtol=0, atol=0)
    # The corrupted location is outside fixed GT for every coordinate helper.
    for nonfinite in (float("nan"), float("inf")):
        corrupted = arguments[0].clone()
        if helper in ("_curvature_loss", "_page_loss"):
            corrupted[0, 0, 1, 1] = nonfinite
        else:
            corrupted[0, 1, 1, 0] = nonfinite
        with pytest.raises(FlattenerError) as caught:
            function(corrupted, *arguments[1:])
        assert caught.value.reason_code == "invalid_training_batch"


@pytest.mark.training
@pytest.mark.parametrize("helper", ["_coordinate_loss", "_boundary_loss",
                                    "_differential_loss", "_orientation_loss"])
def test_fixed_shape_losses_have_finite_gradients_and_zero_invalid_gt_gradient(helper):
    target = coarse_identity((4, 5))[None]
    predicted = target + torch.tensor([0.03, -0.01])
    valid = torch.ones(1, 4, 5, dtype=torch.bool)
    valid[:, 0, 0] = valid[:, -1, -1] = False
    target[~valid] = float("nan")
    predicted[~valid] = 1e30
    predicted.requires_grad_()
    arguments = [predicted, target, valid, (100, 200)]
    if helper in ("_differential_loss", "_orientation_loss"):
        arguments.append((4, 5))
    loss = getattr(study, helper)(*arguments)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(predicted.grad).all()
    assert torch.count_nonzero(predicted.grad[~valid]) == 0


@pytest.mark.parametrize("partitions", [(3,), (2, 1), (1, 1, 1)])
def test_vectorized_validation_is_macro_and_independent_of_batch_partition(monkeypatch, partitions):
    config = study.StudyConfig(device="cpu", output_size=(2, 3))
    predicted = torch.zeros(3, 2, 3, 2)
    target = torch.zeros_like(predicted)
    target[..., 0] = torch.tensor([0.01, 0.09, 0.03])[:, None, None]
    valid = torch.zeros(3, 3, 2, dtype=torch.bool).transpose(1, 2)
    valid[0, 0, 0] = True
    valid[1] = True
    valid[2, 0] = True
    target[~valid] = float("nan")
    predicted[~valid] = 1e30

    def forward(model, images, config, *, _checks=None):
        selected = predicted[(images[:, 0, 0, 0] * 2).long()]
        _checks.append(torch.isfinite(selected).all())
        return None, selected, None

    monkeypatch.setattr(study, "_forward_maps", forward)

    class Loader(list):
        dataset = range(3)

    loader, start = Loader(), 0
    for size in partitions:
        stop = start + size
        loader.append({"image": torch.arange(start, stop).float()[:, None, None, None]
                       .expand(-1, 3, 2, 2) / 2,
                       "target": target[start:stop], "valid": valid[start:stop]})
        start = stop
    result = study._validation_epe(torch.nn.Identity(), loader, config, (100, 200),
                                   torch.device("cpu"))
    assert result == pytest.approx(13 / 3, abs=1e-6)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("include_source_mask", [False, True])
@pytest.mark.parametrize("cuda", [False, True])
def test_cached_batch_loader_preserves_sampling_rng_and_fetch_contract(
        monkeypatch, training, include_source_mask, cuda):
    records = [{"image": torch.full((3, 2, 2), index / 5),
                "target": torch.full((2, 2, 2), float(index)),
                "valid": torch.ones(2, 2, dtype=torch.bool),
                "source_mask": torch.ones(4, 6, dtype=torch.bool)} for index in range(5)]

    class BatchedData:
        split = "train" if training else "validation"

        def __init__(self):
            self.calls, self.outputs = [], []

        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            pytest.fail("cached loading must use fetch_batch, not individual item clones")

        def fetch_batch(self, indices, *, include_source_mask, pin_memory):
            self.calls.append((list(indices), include_source_mask, pin_memory))
            batch = torch.utils.data.default_collate([
                {key: value for key, value in records[index].items()
                 if include_source_mask or key != "source_mask"} for index in indices])
            self.outputs.append(batch)
            return batch

    monkeypatch.setattr(study, "CachedSceneCoordinateDataset", BatchedData)
    config = study.StudyConfig(device="cpu", batch_size=2)
    fast_rng, reference_rng = (torch.Generator().manual_seed(321) for _ in range(2))
    data = BatchedData()
    fast = study._data_loader(data, config, fast_rng, cuda, include_source_mask)
    reference = torch.utils.data.DataLoader(records, batch_size=2, shuffle=training,
                                           generator=reference_rng, num_workers=0)
    global_rng = torch.get_rng_state().clone()
    for _ in range(2):
        actual, expected = list(fast), list(reference)
        assert [len(batch["image"]) for batch in actual] == [2, 2, 1]
        for batch, oracle in zip(actual, expected, strict=True):
            assert any(batch is output for output in data.outputs)
            assert ("source_mask" in batch) == include_source_mask
            for key, value in batch.items():
                assert torch.equal(value, oracle[key])
        assert torch.equal(fast_rng.get_state(), reference_rng.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert len(data.calls) == 6
    assert all(mask == include_source_mask and pin == cuda for _, mask, pin in data.calls)


def _factory_config(include_paper):
    return study.StudyConfig(
        device="cpu", output_size=(16, 20), include_paper_augmentation=include_paper,
        model_config=ModelConfig(input_size=(12, 10), grid_size=(3, 3),
                                 channels=(2, 2, 2), hidden=4))


@pytest.mark.parametrize("include_paper", [False, True])
def test_coordinate_dataset_factory_matches_public_metadata_with_one_admission(
        mutable_paper, monkeypatch, include_paper):
    config = _factory_config(include_paper)
    path = mutable_paper / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    paper_records = manifest["collection"]["paper_augmentation"]["records"]
    paper_records.reverse()  # Preserve source order, not sorted IDs or kind order.
    for record in paper_records:
        if record["kind"] == "simulated":
            del record["kind"]  # The historical omitted-kind default remains valid.
    path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("metadata admission may not decode any split or construct maps")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(scene_data, "make_geometry", forbidden)
    expected = {split: scene_data.SceneCoordinateDataset(
        mutable_paper, split, config.model_config, config.output_size,
        include_paper_augmentation=include_paper) for split in ("train", "validation")}
    admitted = []

    def constructor(*args, **kwargs):
        admitted.append((args, kwargs))
        return scene_data.SceneCoordinateDataset(*args, **kwargs)

    monkeypatch.setattr(study, "SceneCoordinateDataset", constructor)
    actual = study._coordinate_datasets(mutable_paper, config)
    assert list(actual) == ["train", "validation"]
    assert len(admitted) == 1 and admitted[0][0][1] == "train"
    assert scene_data._cache_identity(actual) == scene_data._cache_identity(expected)
    for split, dataset in actual.items():
        assert type(dataset) is scene_data.SceneCoordinateDataset
        assert vars(dataset) == vars(expected[split])
        assert all(record["split"] == split and record["has_page"] is True
                   for record in dataset.records)
        if include_paper:
            extra = [record for record in dataset.records if record.get("source") == "paper"]
            assert [record["id"] for record in extra] == [
                record["id"] for record in paper_records if record["split"] == split]
            assert [record["source_kind"] for record in extra] == ["capture", "simulated"]
            assert "kind" not in extra[1]
        else:
            assert all(record.get("source", "authored") == "authored"
                       for record in dataset.records)

    train, validation = actual["train"], actual["validation"]
    untouched = deepcopy(vars(train))
    assert validation.model_config is not train.model_config
    object.__setattr__(validation.model_config, "input_size", (14, 12))
    validation._records[0]["sha256"]["image.png"] = "record-mutation"
    assert all(record["sha256"]["image.png"] != "record-mutation"
               for record in validation._manifest["samples"])
    validation._manifest["samples"][0]["paths"]["image.png"] = "manifest-mutation"
    validation._source_code_sha256.clear()
    validation.sample_counts["train"] = -1
    validation.page_counts["train"] = -1
    validation.source_page_counts["authored"]["train"] = -1
    validation._verified_template_contents.add("verification-mutation")
    if include_paper:
        next(iter(validation._templates.values()))["sha256"] = "template-mutation"
        next(iter(validation._captures.values()))["corners_xy"][0][0] = -100
    assert vars(train) == untouched


@pytest.mark.parametrize("artifact", ["native_image", "paper_template"])
def test_coordinate_dataset_factory_still_rejects_held_out_source_corruption(
        mutable_paper, monkeypatch, artifact):
    manifest = json.loads((mutable_paper / "manifest.json").read_text(encoding="utf-8"))
    if artifact == "native_image":
        record = next(record for record in manifest["samples"] if record["split"] == "test")
        relative = record["paths"]["image.png"]
    else:
        template = next(template for template in
                        manifest["collection"]["paper_augmentation"]["templates"]
                        if template["split"] == "test")
        relative = template["path"]
    path = mutable_paper / relative
    encoded = path.read_bytes()
    path.write_bytes(encoded[:-1] + bytes([encoded[-1] ^ 1]))
    monkeypatch.setattr(study, "SceneCoordinateDataset", scene_data.SceneCoordinateDataset)
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("held-out decode"))
    with pytest.raises(FlattenerError) as caught:
        study._coordinate_datasets(mutable_paper, _factory_config(True))
    assert caught.value.reason_code == ("artifact_hash_mismatch" if artifact == "native_image"
                                        else "invalid_coordinate_scene_dataset")


def test_preparation_factory_preserves_cache_bit_exactness_full_checks_and_no_test_decode(
        mutable_paper, monkeypatch):
    config = _factory_config(True)
    expected = {split: scene_data.SceneCoordinateDataset(
        mutable_paper, split, config.model_config, config.output_size,
        include_paper_augmentation=True) for split in ("train", "validation")}
    allowed = set()
    for dataset in expected.values():
        for record in dataset.records:
            allowed.update(mutable_paper / record["paths"][name]
                           for name in ("image.png", "mask.png"))
            if record.get("source") == "paper":
                allowed.add(mutable_paper / dataset._templates[record["template_id"]]["path"])
                if record.get("kind") == "capture":
                    allowed.add(mutable_paper / dataset._captures[record["capture_id"]]["path"])
    opened, admissions, full_checks = [], [], []
    original_open = Image.open
    original_check = scene_data.CachedSceneCoordinateDataset.verify_integrity

    def open_selected(path, *args, **kwargs):
        assert path in allowed, f"unexpected or held-out pixel decode: {path}"
        opened.append(path)
        return original_open(path, *args, **kwargs)

    def constructor(*args, **kwargs):
        admissions.append(args[1])
        return scene_data.SceneCoordinateDataset(*args, **kwargs)

    def checked(cache, *, full=True, _hash_session=None):
        full_checks.append((cache.split, full))
        return original_check(cache, full=full, _hash_session=_hash_session)

    monkeypatch.setattr(Image, "open", open_selected)
    monkeypatch.setattr(study, "SceneCoordinateDataset", constructor)
    monkeypatch.setattr(scene_data.CachedSceneCoordinateDataset, "verify_integrity", checked)
    caches = study.prepare_coordinate_data(mutable_paper, config, cache_max_bytes=2 * 1024**2)
    assert admissions == ["train"]
    assert opened and all(full for _, full in full_checks)
    assert all(sum(name == split for name, _ in full_checks) >= 3 for split in expected)
    for split, dataset in expected.items():
        cache = caches[split]
        assert cache.records == dataset.records
        assert len(cache) == len(dataset)
        for index in range(len(dataset)):
            sample, cached = dataset[index], cache[index]
            for key in ("image", "target", "valid", "source_mask"):
                torch.testing.assert_close(cached[key], sample[key], rtol=0, atol=0, equal_nan=True)

    # The held-out template was hashed but never decoded; full pair verification
    # must still detect same-size replacement with the original mtime restored.
    held_out = next(value for value in expected["train"]._templates.values()
                    if value["split"] == "test")
    path = mutable_paper / held_out["path"]
    assert path not in opened
    before, encoded = path.stat(), path.read_bytes()
    path.write_bytes(encoded[:-1] + bytes([encoded[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("integrity check decoded"))
    with pytest.raises(FlattenerError) as caught:
        study.verify_scene_caches(caches)
    assert caught.value.reason_code == "invalid_coordinate_scene_dataset"


@pytest.mark.parametrize("validated", [False, True])
def test_relative_topology_overflow_rejects_or_records_failed_deferred_check(validated):
    target = coarse_identity((3, 3))[None]
    predicted = target * 1e20
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    assert torch.isfinite(predicted).all()
    with pytest.raises(FlattenerError) as caught:
        study._relative_topology_loss(predicted, target, valid, _validated=validated)
    assert caught.value.reason_code == "invalid_training_batch"
    checks = []
    study._relative_topology_loss(predicted, target, valid, _validated=validated, _checks=checks)
    assert len(checks) == 1 and checks[0].dtype == torch.bool and checks[0].ndim == 0
    assert not checks[0].item()


def test_training_admission_rejects_deferred_topology_overflow_before_backward(monkeypatch):
    config = study.StudyConfig(
        device="cpu", output_size=(3, 3), topology_weight=1, topology_mode="relative",
        model_config=ModelConfig(input_size=(4, 4), grid_size=(2, 2),
                                 channels=(2, 2, 2), hidden=4))
    model = torch.nn.Linear(1, 1)
    target = coarse_identity((3, 3))[None]
    predicted = target * 1e20

    def forward(model, images, config, *, _checks=None):
        _checks.append(torch.isfinite(predicted).all())
        return torch.zeros(1, 2, 2, 2), predicted, None

    monkeypatch.setattr(study, "_forward_maps", forward)

    class AdmissionOnlyOptimizer:
        param_groups = [{"params": list(model.parameters())}]

        def zero_grad(self, *, set_to_none):
            assert set_to_none

        def step(self):
            pytest.fail("overflow admission must not optimize")

    batch = {"image": torch.zeros(1, 3, 4, 4), "target": target,
             "valid": torch.ones(1, 3, 3, dtype=torch.bool)}
    with pytest.raises(FlattenerError) as caught:
        study._training_step(model, AdmissionOnlyOptimizer(), batch, config, (100, 200), "cpu")
    assert caught.value.reason_code == "nonfinite_training"
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.training
@pytest.mark.parametrize("architecture", ["global", "page", "projective", "surface"])
@pytest.mark.parametrize("initialization_mode", ["full", "resolution"])
@pytest.mark.parametrize("portable", [
    pytest.param(False, marks=pytest.mark.skipif(
        os.name == "nt", reason="nonportable checkpoints require POSIX symlinks")),
    True,
])
def test_finetune_resets_training_state_and_resumes_without_initial_file(
    tmp_path, monkeypatch, architecture, portable, initialization_mode
):
    monkeypatch.setattr(study, "_PORTABLE_CHECKPOINTS", portable)
    directory = tmp_path / "tiny-data"
    directory.mkdir()
    (directory / "manifest.json").write_text('{"fixture":"two-to-four-positive-records"}\n')
    opened = []

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, root, split, model_config, output_size,
                     *, include_paper_augmentation=False):
            assert split in ("train", "validation")
            opened.append((split, include_paper_augmentation))
            self.model_config, self.output_size = model_config, output_size
            self.photo_size = (16, 16)
            self.manifest_sha256 = study._sha256(root / "manifest.json")
            self.count = 4 if include_paper_augmentation else 2
            self.page_counts = dict.fromkeys(("train", "validation", "test"), self.count)

        def __len__(self):
            return self.count

        def __getitem__(self, index):
            return {
                "image": torch.full((3, *self.model_config.input_size), index / self.count),
                "target": coarse_identity(self.output_size) + torch.tensor([0.05, -0.03]),
                "valid": torch.ones(self.output_size, dtype=torch.bool),
                "source_mask": torch.ones(self.photo_size, dtype=torch.bool),
            }

    monkeypatch.setattr(study, "SceneCoordinateDataset", TinyDataset)
    config = study.StudyConfig(
        architecture=architecture, device="cpu", epochs=1, batch_size=2, seed=123,
        map_mode="analytic" if architecture in ("projective", "surface") else "legacy",
        output_size=(16, 16),
        model_config=ModelConfig(input_size=(16, 16), grid_size=(3, 3),
                                 channels=(2, 4, 4), hidden=8),
    )
    source_output = tmp_path / "source"
    study.train_coordinate_study(directory, source_output, config)
    source, source_sha = study._read_checkpoint(source_output / "last.pt")
    assert source["epoch"] == 1 and source["optimizer_state"]["state"]
    config = replace(config, epochs=2, seed=456, learning_rate=0.002,
                     include_paper_augmentation=True, initialization_mode=initialization_mode,
                     model_config=(replace(config.model_config, input_size=(24, 20))
                                   if initialization_mode == "resolution" else config.model_config))
    output = tmp_path / "finetune"
    before_rng = torch.get_rng_state().clone()
    report = study.train_coordinate_study(
        directory, output, config, initial_checkpoint=source_output / "last.pt",
        stop_after_epoch=0,
    )
    initialized = study.coordinate_checkpoint(output / "last.pt")
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert report["initial_checkpoint_sha256"] == source_sha
    assert report["include_paper_augmentation"] is True
    assert initialized["config"]["initial_checkpoint_sha256"] == source_sha
    assert initialized["config"]["initial_model_config"] == (
        source["config"]["model_config"] if initialization_mode == "resolution" else None)
    assert initialized["dataset_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert initialized["epoch"] == 0 and initialized["history"] == []
    assert initialized["optimizer_state"]["state"] == {}
    assert initialized["optimizer_state"]["param_groups"][0]["lr"] == 0.002
    assert study._same_state(initialized["model_state"], source["model_state"])
    assert not torch.equal(initialized["torch_rng_state"], source["torch_rng_state"])
    bound_config = replace(
        config, initial_checkpoint_sha256=source_sha,
        initial_model_config=(ModelConfig(**source["config"]["model_config"])
                              if initialization_mode == "resolution" else None))
    for changed, reason in (
        (config, "invalid_initialization" if initialization_mode == "resolution"
         else "invalid_resume"),
        (replace(bound_config, include_paper_augmentation=False), "invalid_resume"),
    ):
        with pytest.raises(FlattenerError) as caught:
            study.train_coordinate_study(directory, output, changed, resume=output / "last.pt")
        assert caught.value.reason_code == reason
        assert not (output / ".training.lock").exists()
    source_output.rename(tmp_path / "retired-source")
    resumed = study.train_coordinate_study(
        directory, output, bound_config, resume=output / "last.pt", stop_after_epoch=1,
    )
    saved = study.coordinate_checkpoint(output / "last.pt")
    assert resumed["epochs_completed"] == 1
    assert resumed["initial_checkpoint_sha256"] == source_sha
    assert len(saved["history"]) == 1 and saved["optimizer_state"]["state"]
    assert all(entry["step"].item() == 2 for entry in saved["optimizer_state"]["state"].values())
    assert ("train", True) in opened and ("validation", True) in opened
    assert all(split in ("train", "validation") for split, _ in opened)


@pytest.mark.training
@pytest.mark.parametrize(
    "page_relative_weight,orientation_weight,rotation_probability,rotation_mode,background_probability",
    [
    pytest.param(0, 0, 0, "batch", 0, id="0"),
    pytest.param(0.25, 0, 0, "batch", 0, id="0.25"),
    pytest.param(0.25, 0.2, 1.0, "batch", 0, id="orientation-rotation"),
    pytest.param(0.25, 0.2, 1.0, "per_image_fit", 1.0, id="quality-augmentation"),
])
def test_prepared_cache_reuse_preserves_training_and_resume(
        tmp_path, monkeypatch, page_relative_weight, orientation_weight, rotation_probability,
        rotation_mode, background_probability):
    from flattener.synthetic.paired_scenes import PairedSceneConfig, generate_paired_scene_dataset

    directory = tmp_path / "data"
    generate_paired_scene_dataset(directory, PairedSceneConfig(
        train_documents=1, validation_documents=1, test_documents=1,
        image_size=(128, 128), seed=397))
    config = study.StudyConfig(
        device="cpu", epochs=2, batch_size=2, output_size=(16, 16),
        page_relative_weight=page_relative_weight,
        orientation_weight=orientation_weight, rotation_probability=rotation_probability,
        rotation_mode=rotation_mode, background_probability=background_probability,
        model_config=ModelConfig(input_size=(16, 16), grid_size=(3, 3),
                                 channels=(2, 4, 4), hidden=8))
    budget = 1024**2
    prepared = study.prepare_coordinate_data(directory, config, cache_max_bytes=budget)
    assert set(prepared) == {"train", "validation"}
    reference = tmp_path / "reference"
    study.train_coordinate_study(directory, reference, config, cache_max_bytes=budget)
    expected = study.coordinate_checkpoint(reference / "last.pt")

    def no_rebuild(*args, **kwargs):
        pytest.fail("prepared cache must not reconstruct or decode the dataset")

    monkeypatch.setattr(study, "SceneCoordinateDataset", no_rebuild)
    output = tmp_path / "reused"
    controls = {"cache_max_bytes": budget, "prepared_data": prepared}
    first = study.train_coordinate_study(directory, output, config, stop_after_epoch=1, **controls)
    assert first["prepared_cache_reused"] and first["final_integrity_seconds"] >= 0
    report = study.train_coordinate_study(
        directory, output, config, resume=output / "last.pt", **controls)
    actual = study.coordinate_checkpoint(output / "last.pt")
    assert report["complete"] and report["prepared_cache_reused"]
    assert report["cache_integrity"] == "full_at_run_boundaries_stat_each_epoch"
    for key in ("model_state", "optimizer_state", "history",
                "torch_rng_state", "shuffle_rng_state"):
        assert study._same_state(actual[key], expected[key]), key
    checkpoint_before = study._checkpoint_location(output / "last.pt").read_bytes()
    for changes in ({"page_relative_weight": page_relative_weight + 0.25},
                    {"orientation_weight": orientation_weight + 0.25},
                    {"rotation_probability": 0.0 if rotation_probability else 1.0},
                    {"rotation_mode": "batch" if rotation_mode == "per_image_fit"
                     else "per_image_fit"},
                    {"background_probability": 0.0 if background_probability else 1.0}):
        with pytest.raises(FlattenerError) as caught:
            study.train_coordinate_study(
                directory, output, replace(config, **changes),
                resume=output / "last.pt", **controls)
        assert caught.value.reason_code == "invalid_resume"
        assert study._checkpoint_location(output / "last.pt").read_bytes() == checkpoint_before
    with pytest.raises(FlattenerError, match="preprocessing differs"):
        study.train_coordinate_study(directory, tmp_path / "wrong", replace(
            config, output_size=(20, 20)), **controls)
    with pytest.raises(FlattenerError, match="budget"):
        study.train_coordinate_study(directory, tmp_path / "small", config,
                                     prepared_data=prepared, cache_max_bytes=1)
    assert not (tmp_path / "wrong").exists() and not (tmp_path / "small").exists()


@pytest.mark.parametrize("architecture", ["projective", "surface"])
def test_backbone_initialization_transfers_only_compatible_layers(initial_checkpoint, architecture):
    path, config, source, digest = initial_checkpoint
    config = replace(config, architecture=architecture, map_mode="analytic",
                     initialization_mode="backbone")
    before = torch.get_rng_state().clone()
    actual, state = study._initialization(config, path, None)
    assert actual.initial_checkpoint_sha256 == digest
    assert torch.equal(before, torch.get_rng_state())
    for key, value in state.items():
        if key.startswith(("encoder.", "head.1.")):
            assert torch.equal(value, source["model_state"][key])
        else:
            assert key in {"head.3.weight", "head.3.bias"} and not value.count_nonzero()
    assert state["head.3.bias"].numel() == (8 if architecture == "projective" else 48)
    with pytest.raises(FlattenerError, match="checkpoint path"):
        study._initialization(config, None, None)


@pytest.mark.parametrize("scale", [1.0, 0.1, 0.02])
def test_relative_topology_is_scale_relative_and_masks_whole_invalid_cells(scale):
    target = coarse_identity((5, 7))[None] * scale
    valid = torch.ones(1, 5, 7, dtype=torch.bool)
    predicted = target.clone()
    assert study._relative_topology_loss(predicted, target, valid) == 0
    predicted[..., 0].neg_()
    assert study._relative_topology_loss(predicted, target, valid).item() == pytest.approx(2.42)
    assert study._relative_topology_loss(predicted, target, ~valid) == 0
    reflected_target = target.clone()
    reflected_target[..., 0].neg_()
    assert study._relative_topology_loss(predicted, reflected_target, valid) == 0


@pytest.mark.training
def test_relative_topology_fold_penalty_has_finite_gradients():
    target = coarse_identity((5, 7))[None] * .05
    predicted = target.clone()
    predicted[..., 0].neg_()
    predicted.requires_grad_()
    valid = torch.ones(1, 5, 7, dtype=torch.bool)
    study._relative_topology_loss(predicted, target, valid).backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().max() > 0


@pytest.mark.training
def test_geometry_regularizers_have_finite_nonzero_gradients_without_optimizer():
    predicted = coarse_identity((5, 7))[None].clone()
    predicted[..., 0].neg_()
    predicted.requires_grad_(True)
    topology = study._topology_loss(predicted, torch.ones(1, 5, 7, dtype=torch.bool))
    topology.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().max() > 0
    residual = torch.zeros(1, 2, 3, 4)
    residual[0, 0, 1, 2] = 0.02
    residual.requires_grad_(True)
    curvature = study._curvature_loss(residual, (100, 200))
    curvature.backward()
    assert torch.isfinite(residual.grad).all() and residual.grad.abs().max() > 0


@pytest.mark.training
def test_page_and_gt_differential_supervision_have_gradients_without_optimizer():
    config = study.StudyConfig(
        architecture="page", device="cpu", page_weight=1,
        model_config=ModelConfig(input_size=(16, 12), grid_size=(5, 4),
                                 channels=(2, 4, 4), hidden=8),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(153)
        model = study._new_model(config)
        images = torch.rand(2, 3, 16, 12)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.fill_(0.05)
                layer.bias.fill_(0.01)
    source_mask = torch.zeros(2, 24, 18, dtype=torch.bool)
    source_mask[:, 5:20, 3:15] = True
    _, logits = model.forward_with_page(images)
    study._page_loss(logits, source_mask).backward()
    for parameter in (model.encoder[0][0].weight, model.page_head[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0
    model.zero_grad(set_to_none=True)
    predicted = study._dense_maps(model(images), (16, 12))
    target = coarse_identity((16, 12))[None].repeat(2, 1, 1, 1)
    target[..., 0] += 0.03 * target[..., 1].square() + 0.02
    valid = torch.ones(2, 16, 12, dtype=torch.bool)
    objective = study._boundary_loss(predicted, target, valid, (24, 18))
    objective += study._differential_loss(predicted, target, valid, (24, 18), (5, 4))
    objective.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head[-1].weight.grad.abs().max() > 0
