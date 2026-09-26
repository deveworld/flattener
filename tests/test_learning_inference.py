"""Loaded-model contracts, failure propagation, and identity photo integration."""

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from flattener import FlattenerError, RenderConfig, Status
from flattener.learning.inference import load_predictor, unwarp
from flattener.learning.model import CoordinatePredictor, ModelConfig
from flattener.learning.spatial import expand_coordinate_residual, make_coordinate_model


@pytest.fixture
def checkpoint(tmp_path):
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


def test_zero_checkpoint_produces_identity_and_metadata_without_reloading(
    checkpoint, image, monkeypatch
):
    predictor = load_predictor(checkpoint)
    assert predictor.checksum_verified
    assert predictor.model_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("model reloaded"))
    for _ in range(2):
        result = unwarp(image, predictor, image.shape[:2])
        assert result.status == Status.SUCCESS
        np.testing.assert_array_equal(result.image, image)
        assert result.valid_mask.all()
        metadata = result.metadata
        assert metadata["processing_status"] == "success"
        assert metadata["quality_status"] == "not_evaluated"
        assert metadata["model_sha256"] == predictor.model_sha256
        assert metadata["model_id"] == predictor.model_id
        assert metadata["source_size"] == [29, 37]
        assert metadata["requested_output_size"] == [29, 37]
        assert metadata["actual_output_size"] == [29, 37]
        assert metadata["coarse_grid_size"] == [5, 3]
        assert metadata["model_input_size"] == [16, 12]
        assert metadata["model_load_seconds"] >= 0
        assert metadata["timings"]["load_model"] == 0
        for stage in ("prepare_input", "predict", "process_coordinates", "render", "total"):
            assert metadata["timings"][stage] >= 0
        json.dumps(metadata, allow_nan=False)


def test_loading_and_inference_preserve_rng_threads_and_flags(checkpoint, image):
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    predictor = load_predictor(checkpoint)
    unwarp(image, predictor, (29, 37))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn


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


def test_sidecar_optional_but_mismatch_rejected(checkpoint):
    checkpoint.with_suffix(".pt.sha256").unlink()
    assert not load_predictor(checkpoint).checksum_verified
    checkpoint.with_suffix(".pt.sha256").write_text("0" * 64)
    with pytest.raises(FlattenerError) as caught:
        load_predictor(checkpoint)
    assert caught.value.reason_code == "checkpoint_hash_mismatch"


@pytest.mark.parametrize("change", ["schema", "config", "shape", "keys", "dtype", "nan", "huge"])
def test_invalid_checkpoints_are_rejected_strictly(checkpoint, change):
    saved = torch.load(checkpoint, weights_only=True)
    key = next(iter(saved["model_state"]))
    if change == "schema":
        saved["schema_version"] = "unknown"
    elif change == "config":
        saved["config"]["model_config"]["unexpected"] = 1
    elif change == "shape":
        saved["model_state"][key] = torch.zeros(1)
    elif change == "keys":
        saved["model_state"].pop(key)
    elif change == "dtype":
        saved["model_state"][key] = saved["model_state"][key].double()
    elif change == "nan":
        saved["model_state"][key].fill_(float("nan"))
    else:
        saved["config"]["model_config"]["hidden"] = 1_000_000
    torch.save(saved, checkpoint)
    checkpoint.with_suffix(".pt.sha256").unlink()
    with pytest.raises(FlattenerError) as caught:
        load_predictor(checkpoint)
    assert caught.value.status == Status.MODEL_ERROR


def test_malformed_file_and_missing_file_have_explicit_errors(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FlattenerError) as caught:
        load_predictor(missing)
    assert caught.value.status == Status.IO_ERROR
    missing.write_bytes(b"not a torch checkpoint")
    with pytest.raises(FlattenerError) as caught:
        load_predictor(missing)
    assert caught.value.status == Status.MODEL_ERROR


@pytest.mark.parametrize("policy", ["raise", "return", "original"])
@pytest.mark.parametrize("bad_output", ["nan", "shape", "runtime", "oob"])
def test_model_and_geometry_failures_follow_policy(
    checkpoint, image, monkeypatch, policy, bad_output
):
    predictor = load_predictor(checkpoint)

    def forward(images):
        if bad_output == "runtime":
            raise RuntimeError("test inference failure")
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


@pytest.mark.parametrize("output_size", [(0, 37), (29, True), (4, 37), (29, 2), (100_000, 37)])
def test_bad_output_dimensions_fail_before_prediction(checkpoint, image, monkeypatch, output_size):
    predictor = load_predictor(checkpoint)
    monkeypatch.setattr(predictor._model, "forward", lambda x: pytest.fail("prediction ran"))
    result = unwarp(image, predictor, output_size, config=RenderConfig(on_error="original"))
    assert result.status == Status.INPUT_ERROR
    assert result.used_fallback


def test_invalid_image_never_becomes_original_fallback(checkpoint):
    result = unwarp(
        np.zeros((29, 37, 3), dtype=np.float32),
        load_predictor(checkpoint),
        (29, 37),
        config=RenderConfig(on_error="original"),
    )
    assert result.status == Status.INPUT_ERROR
    assert not result.used_fallback
    assert result.image is None
    assert result.metadata["processing_status"] == "input_error"
    assert result.metadata["quality_status"] == "not_evaluated"


def test_invalid_failure_policy_reports_processing_error_not_quality(checkpoint, image):
    with pytest.raises(FlattenerError) as caught:
        unwarp(image, load_predictor(checkpoint), (29, 37), config={})
    assert caught.value.metadata["processing_status"] == "input_error"
    assert caught.value.metadata["quality_status"] == "not_evaluated"


def test_prediction_error_cannot_claim_processing_success_or_evaluated_quality(
    checkpoint, image, monkeypatch
):
    predictor = load_predictor(checkpoint)

    def fail(*args, **kwargs):
        raise FlattenerError(
            Status.MODEL_ERROR,
            "prediction_failed",
            "upstream error carried stale status",
            metadata={"processing_status": "success", "quality_status": "passed"},
        )

    monkeypatch.setattr(predictor._model, "forward", fail)
    result = unwarp(image, predictor, (29, 37), config=RenderConfig(on_error="return"))
    assert result.metadata["processing_status"] == "model_error"
    assert result.metadata["quality_status"] == "not_evaluated"


def test_allocation_errors_are_not_hidden_by_fallback(checkpoint, image, monkeypatch):
    predictor = load_predictor(checkpoint)

    def allocation_failure(_):
        raise RuntimeError("can't allocate memory")

    monkeypatch.setattr(predictor._model, "forward", allocation_failure)
    with pytest.raises(MemoryError):
        unwarp(image, predictor, (29, 37), config=RenderConfig(on_error="original"))


@pytest.mark.parametrize("architecture", ["projective", "surface"])
def test_analytic_inference_uses_direct_dense_coordinates(checkpoint, image, architecture):
    legacy = load_predictor(checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = make_coordinate_model(architecture, legacy.model_config).eval()
    with torch.no_grad():
        model.head[-1].bias[:8].copy_(torch.tensor([.05, -.08, .12, -.2, -.1, .03, .5, -.2]))
    predictor = replace(legacy, _model=model.requires_grad_(False), map_mode="analytic")
    from flattener.learning.model import preprocess_image
    with torch.inference_mode():
        residual, direct = model.forward_dense(
            preprocess_image(image, model.config)[None], (29, 37)
        )
    actual = predictor.predict_map(image, (29, 37))
    np.testing.assert_array_equal(actual, direct[0].numpy())
    assert not np.allclose(actual, expand_coordinate_residual(residual, (29, 37))[0].numpy())
    result = unwarp(image, predictor, (29, 37), config=RenderConfig(on_error="return"))
    assert result.metadata["model_map_mode"] == "analytic"


def test_linear_inference_preserves_affine_boundary(checkpoint, image):
    from flattener.learning.model import coarse_identity
    legacy = load_predictor(checkpoint)
    grid = coarse_identity(legacy.model_config.grid_size)
    residual = -2 * grid
    with torch.no_grad():
        legacy._model.head[-1].bias.copy_(residual.permute(2, 0, 1).flatten())
    predictor = replace(legacy, map_mode="linear")
    actual = predictor.predict_map(image, image.shape[:2])
    np.testing.assert_allclose(actual, -coarse_identity(image.shape[:2]).numpy(), atol=3e-7)
    assert unwarp(image, predictor, image.shape[:2]).status == Status.SUCCESS
    assert unwarp(image, legacy, image.shape[:2], config=RenderConfig(on_error="return")).status \
        == Status.INVALID_TRANSFORM


@pytest.mark.parametrize("failure", ["pair", "shape", "dtype", "nan", "coarse", "missing"])
def test_invalid_analytic_outputs_follow_failure_policy(checkpoint, image, monkeypatch, failure):
    legacy = load_predictor(checkpoint)
    predictor = replace(legacy, map_mode="analytic")
    coarse = torch.zeros(1, 2, *legacy.model_config.grid_size)
    dense = torch.zeros(1, *image.shape[:2], 2)
    if failure == "pair":
        output = dense
    elif failure == "shape":
        output = coarse, dense[..., 0]
    elif failure == "dtype":
        output = coarse, dense.double()
    elif failure == "nan":
        output = coarse, dense.fill_(float("nan"))
    elif failure == "coarse":
        output = coarse[:, 0], dense
    else:
        output = coarse, dense
    if failure != "missing":
        monkeypatch.setattr(predictor._model, "forward_dense", lambda *args: output, raising=False)
    result = unwarp(image, predictor, image.shape[:2], config=RenderConfig(on_error="original"))
    assert result.status == Status.MODEL_ERROR and result.used_fallback
    assert result.metadata["quality_status"] == "not_evaluated"
    np.testing.assert_array_equal(result.image, image)
