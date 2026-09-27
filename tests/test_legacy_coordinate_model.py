"""The old coordinate model (``flattener.learning``) behind ``selected.pt``.

Loading, inference, the photo example and the dense-map expansion. No test here trains.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError, RenderConfig, Status, identity_map, render
from flattener.coordinates import resize_backward_map
from flattener.learning import coordinate_study as study
from flattener.learning.coordinate_study import coordinate_checkpoint, load_coordinate_predictor
from flattener.learning.inference import LoadedPredictor, load_predictor, unwarp
from flattener.learning.model import (
    CoordinatePredictor,
    ModelConfig,
    coarse_identity,
    dense_maps,
    preprocess_image,
)
from flattener.learning.spatial import (
    DeviceCoordinatePredictor,
    expand_coordinate_residual,
    make_coordinate_model,
)

ROOT = Path(__file__).resolve().parents[1]
SMALL = ModelConfig(input_size=(24, 18), grid_size=(5, 3), channels=(4, 6, 8), hidden=16)
SELECTED = ROOT / "outputs" / "coordinate-fold" / "selected.pt"
SELECTED_SHA256 = "ee7e2cf7a82bc238d1054cb2de44d15dcd007ea215e8c44efa86a4f9da877ea8"


@pytest.fixture(autouse=True)
def isolate_torch_rng_and_forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("these tests must never run backward or an optimizer")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture(scope="module")
def study_models(tmp_path_factory):
    """An untrained ``global`` study checkpoint with its sha256 sidecar."""
    root = tmp_path_factory.mktemp("untrained-coordinate-fixtures")
    config = study.StudyConfig(
        architecture="global",
        map_mode="legacy",
        page_weight=0.0,
        direction_weight=0.0,
        corner_weight=0.0,
        model_config=ModelConfig(
            input_size=(32, 24), grid_size=(9, 7), channels=(4, 8, 12), hidden=16
        ),
        output_size=(128, 128),
    )
    with torch.random.fork_rng(devices=[]):
        model = make_coordinate_model("global", config.model_config)
    payload = {
        "schema_version": study.CHECKPOINT_SCHEMA,
        "checkpoint_kind": "best",
        "config": asdict(config),
        "model_spec": study._model_spec(config),
        "model_state": model.state_dict(),
        "optimizer_state": {
            "state": {},
            "param_groups": [study._adam_group(config, len(list(model.parameters())))],
        },
        "epoch": 0,
        "best_epoch": 0,
        "history": [],
        "initial_validation_epe_pixels": 1.0,
        "best_validation_epe_pixels": 1.0,
        "manifest_sha256": "0" * 64,
        "provenance": study._provenance(),
        "dataset_output_size": config.output_size,
        "dataset_photo_size": (128, 128),
        "dataset_counts": {"train": 6, "validation": 6, "test": 6},
        "torch_rng_state": torch.get_rng_state(),
        "shuffle_rng_state": torch.Generator().manual_seed(config.seed).get_state(),
        # Serialized shape fixture only, never passed to any CUDA RNG API.
        "cuda_rng_state": torch.zeros(16, dtype=torch.uint8),
        "xpu_rng_state": None,
        "device_info": {"device": "cuda:0", "name": "fixture", "capability": (8, 0)},
        "best_checkpoint_sha256": None,
    }
    directory = root / "global"
    directory.mkdir()
    study._save_checkpoint(directory / "best.pt", payload)
    return root


@pytest.fixture
def checkpoint(tmp_path):
    """A plain ``learning-0.1`` checkpoint with an untrained model."""
    config = ModelConfig(input_size=(16, 12), grid_size=(5, 3), channels=(4, 6, 8), hidden=16)
    with torch.random.fork_rng(devices=[]):
        model = CoordinatePredictor(config)
    saved = {
        "schema_version": "learning-0.1",
        "config": {"model_config": asdict(config)},
        "model_state": model.state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    path.with_suffix(".pt.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return path


@pytest.fixture
def image():
    return np.arange(29 * 37 * 3, dtype=np.uint8).reshape(29, 37, 3)


# Loading and inference


def test_model_load_and_existing_renderer_exact(study_models):
    checkpoint = study_models / "global" / "best.pt"
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    predictor = load_coordinate_predictor(checkpoint)
    assert isinstance(predictor, LoadedPredictor)
    assert predictor.checksum_verified
    assert "global" in predictor.model_id
    assert predictor.model_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert predictor.checkpoint_path == str(checkpoint.resolve())
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic
    assert all(
        p.device.type == "cpu" and not p.requires_grad for p in predictor._model.parameters()
    )
    image = np.random.default_rng(111).integers(0, 256, (129, 151, 3), dtype=np.uint8)
    original = image.copy()
    grid = predictor.predict_map(image, (128, 136))
    np.testing.assert_array_equal(grid, identity_map((128, 136)))
    direct = render(image, grid, config=RenderConfig(on_error="return"))
    public = unwarp(image, predictor, (128, 136), config=RenderConfig(on_error="return"))
    assert public.status == direct.status and public.reason_code == direct.reason_code
    for key in ("image", "source_mask", "target_mask", "valid_mask"):
        np.testing.assert_array_equal(getattr(public, key), getattr(direct, key))
    exact = unwarp(image, predictor, image.shape[:2])
    np.testing.assert_array_equal(exact.image, image)
    np.testing.assert_array_equal(image, original)
    assert public.metadata["model_id"] == predictor.model_id
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_photo_cli_exit_status_and_saved_results(study_models, tmp_path):
    """Run examples/unwarp_photo.py in a subprocess with hand-set weights."""
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    checkpoint = tmp_path / "model.pt"
    study._save_checkpoint(checkpoint, saved)
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    input_path = tmp_path / "input.png"
    Image.fromarray(image).save(input_path)
    output = tmp_path / "output"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "unwarp_photo.py"),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--output-dir",
            str(output),
            "--height",
            "32",
            "--width",
            "32",
            "--on-error",
            "return",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert process.returncode == 0, process.stderr
    record = json.loads((output / "metadata.json").read_text())
    assert record["status"] == record["metadata"]["processing_status"] == "success"
    assert record["metadata"]["quality_status"] == "not_evaluated"
    assert record["used_fallback"] is False
    assert record["reason_code"] is None
    assert "success" in process.stdout
    with Image.open(output / "image.png") as decoded:
        np.testing.assert_array_equal(np.asarray(decoded), image)


def test_changed_checkpoint_is_rejected(study_models, tmp_path):
    source = study_models / "global" / "best.pt"
    destination = tmp_path / "bad.pt"
    destination.write_bytes(source.read_bytes() + b"tampered")
    Path(str(destination) + ".sha256").write_text(
        hashlib.sha256(source.read_bytes()).hexdigest() + "\n"
    )
    with pytest.raises(FlattenerError):
        load_coordinate_predictor(destination)


def test_checksum_sidecar_is_mandatory(study_models, tmp_path):
    path = tmp_path / "no-sidecar.pt"
    path.write_bytes((study_models / "global" / "best.pt").read_bytes())
    with pytest.raises(FlattenerError):
        study.coordinate_checkpoint(path)


def test_predict_map_coordinates_are_independent_of_preview_resize(checkpoint, image):
    predictor = load_predictor(checkpoint)
    original = image.copy()
    grid = predictor.predict_map(image[:, ::-1], (9, 7))
    assert grid.dtype == np.float32
    assert grid.shape == (9, 7, 2)
    y, x = np.mgrid[:9, :7]
    expected = np.stack((2 * (x + 0.5) / 7 - 1, 2 * (y + 0.5) / 9 - 1), axis=-1)
    np.testing.assert_allclose(grid, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "policy,bad_output",
    [("raise", "nan"), ("return", "oob"), ("original", "shape")],
)
def test_model_and_geometry_failures_follow_policy(
    checkpoint, image, monkeypatch, policy, bad_output
):
    predictor = load_predictor(checkpoint)

    def forward(images):
        if bad_output == "shape":
            return torch.zeros(1, 2, 3, 3)
        return torch.full((1, 2, 5, 3), float("nan") if bad_output == "nan" else 5.0)

    monkeypatch.setattr(predictor._model, "forward", forward)
    status = Status.INVALID_TRANSFORM if bad_output == "oob" else Status.MODEL_ERROR
    if policy == "raise":
        with pytest.raises(FlattenerError) as caught:
            unwarp(image, predictor, (29, 37))
        assert caught.value.status == status
        assert caught.value.metadata["processing_status"] == str(status)
        assert caught.value.metadata["quality_status"] == "not_evaluated"
        assert caught.value.metadata["model_sha256"] == predictor.model_sha256
    else:
        result = unwarp(image, predictor, (29, 37), config=RenderConfig(on_error=policy))
        assert result.status == status
        assert result.metadata["processing_status"] == str(status)
        assert result.metadata["quality_status"] == "not_evaluated"
        assert result.metadata["failure_stage"] in {"predict", "process_coordinates"}
        assert result.used_fallback == (policy == "original")
        if policy == "original":
            np.testing.assert_array_equal(result.image, image)
            assert not np.shares_memory(result.image, image)
        else:
            assert result.image is None


@pytest.mark.skipif(not SELECTED.is_file(), reason="selected.pt is not present")
def test_selected_model_loads_and_unwarps():
    assert hashlib.sha256(SELECTED.read_bytes()).hexdigest() == SELECTED_SHA256
    saved = coordinate_checkpoint(SELECTED)
    assert saved["schema_version"] == "coordinate-study-0.3"
    assert saved["config"]["architecture"] == "global"

    predictor = load_coordinate_predictor(SELECTED)
    assert predictor.model_sha256 == SELECTED_SHA256
    assert predictor.map_mode == "legacy"

    rows, cols = np.mgrid[0:288, 0:384]
    image = np.stack(
        [(rows * 255) // 287, (cols * 255) // 383, ((rows + cols) * 7) % 256], axis=-1
    ).astype(np.uint8)
    grid = predictor.predict_map(image, (192, 144))
    assert grid.shape == (192, 144, 2)
    assert np.isfinite(grid).all()

    result = unwarp(image, predictor, (192, 144))
    assert result.status == Status.SUCCESS
    assert result.image.shape == (192, 144, 3)


# Model, preprocessing and dense-map expansion


def test_initial_model_is_identity_for_different_images_and_preserves_all_dense_centers():
    model = CoordinatePredictor()
    images = torch.stack((torch.zeros(3, 96, 72), torch.ones(3, 96, 72)))
    with torch.inference_mode():
        residual = model(images)
        actual = dense_maps(residual, (256, 192))
    assert residual.shape == (2, 2, 17, 13)
    assert torch.count_nonzero(residual).item() == 0
    y, x = np.mgrid[:256, :192]
    expected = np.stack(((x + 0.5) / 96 - 1, (y + 0.5) / 128 - 1), axis=-1)
    np.testing.assert_allclose(actual[0].numpy(), expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual[0], actual[1], rtol=0, atol=0)


def test_dense_maps_matches_renderer_expansion_for_varying_residuals():
    residual = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 60 - 0.2
    coarse = coarse_identity((3, 4)) + residual[0].permute(1, 2, 0)
    expected = resize_backward_map(coarse.numpy(), (7, 11))
    np.testing.assert_allclose(
        dense_maps(residual, (7, 11))[0].numpy(), expected, rtol=0, atol=2e-7
    )


def test_preprocessing_fixed_resize_uses_entire_extent_and_half_pixel_centers():
    image = np.zeros((2, 3, 3), np.uint8)
    image[:, :, 0] = [0, 120, 240]
    image[1, :, 1] = 240
    result = preprocess_image(image, ModelConfig(input_size=(4, 6)))
    expected_x = torch.tensor([0, 30, 90, 150, 210, 240]) / 255
    expected_y = torch.tensor([0, 60, 180, 240]) / 255
    torch.testing.assert_close(result[0], expected_x.expand(4, 6), rtol=0, atol=1e-6)
    torch.testing.assert_close(result[1], expected_y[:, None].expand(4, 6), rtol=0, atol=1e-6)
    assert torch.count_nonzero(result[2]) == 0


def test_global_subclass_matches_legacy_keys_initialization_and_forward_exactly():
    torch.manual_seed(87)
    legacy = CoordinatePredictor(SMALL)
    legacy_rng_after = torch.get_rng_state().clone()
    torch.manual_seed(87)
    model = make_coordinate_model("global", SMALL)
    assert isinstance(model, DeviceCoordinatePredictor)
    assert isinstance(model, CoordinatePredictor)
    assert torch.equal(torch.get_rng_state(), legacy_rng_after)
    assert model.state_dict().keys() == legacy.state_dict().keys()
    for key, value in legacy.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
    # Set a nonzero final layer so both models produce nonzero predictions to compare.
    with torch.no_grad():
        legacy.head[-1].weight.fill_(0.031)
        legacy.head[-1].bias.fill_(-0.012)
    model.load_state_dict(legacy.state_dict(), strict=True)
    images = torch.rand(2, 3, *SMALL.input_size)
    with torch.inference_mode():
        assert torch.equal(model(images), legacy(images))


@pytest.mark.parametrize("size", [(5, 3), (17, 23), (24, 18)])
def test_legacy_expansion_is_bitwise_identical_to_original_dense_maps(size):
    residual = torch.randn(2, 2, *SMALL.grid_size)
    before = residual.clone()
    actual = expand_coordinate_residual(residual, size, mode="legacy")
    assert torch.equal(actual, dense_maps(residual, size))
    assert torch.equal(actual, expand_coordinate_residual(residual, size))
    assert torch.equal(residual, before)
    assert actual.shape == (2, *size, 2) and actual.dtype == torch.float32
