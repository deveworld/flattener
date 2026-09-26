"""Region segmentation contracts, binary-target preprocessing, and safe loading."""

import hashlib
from dataclasses import asdict

import numpy as np
import pytest
import torch

from flattener import FlattenerError
from flattener.regions.segmentation import (
    CHECKPOINT_SCHEMA,
    PageSegmenter,
    SegmenterConfig,
    load_segmenter,
    preprocess_image,
    preprocess_mask,
    read_checkpoint,
)


def _save_checkpoint(path, state):
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")


@pytest.fixture
def model_config():
    return SegmenterConfig(input_size=(17, 13), channels=(2, 4, 6))


@pytest.mark.parametrize(
    "changes",
    [
        {"input_size": [128, 96]},
        {"channels": (1, 2)},
    ],
)
def test_invalid_model_config(changes):
    with pytest.raises(ValueError):
        SegmenterConfig(**changes)


@pytest.mark.training
def test_forward_handles_nonmultiple_of_four_dimensions_and_backprop(model_config):
    model = PageSegmenter(model_config)
    image = torch.rand(2, 3, *model_config.input_size, requires_grad=True)
    output = model(image)
    assert output.shape == (2, 1, *model_config.input_size)
    assert output.dtype == torch.float32
    output.sum().backward()
    assert bool(torch.isfinite(image.grad).all())
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(1, 3, 128, 96, dtype=torch.float64),
        torch.full((1, 3, 128, 96), float("nan")),
    ],
)
def test_model_rejects_invalid_input(image):
    with pytest.raises((TypeError, ValueError)):
        PageSegmenter()(image)


def test_preprocess_preserves_whole_extent_and_caller_memory(model_config):
    image = np.zeros((40, 20, 3), np.uint8)
    image[:20, :, 0] = 255
    image[20:, :, 2] = 255
    original = image.copy()
    output = preprocess_image(image[::-1], model_config)
    assert output.shape == (3, 17, 13)
    assert output.dtype == torch.float32 and output.device.type == "cpu"
    assert output[2, 0].mean() > 0.99 and output[0, -1].mean() > 0.99
    output.fill_(0.7)
    np.testing.assert_array_equal(image, original)


def test_mask_preprocess_is_binary_with_exact_center_sampling(model_config):
    mask = np.zeros((34, 26), bool)
    mask[3::2, 3::2] = True
    output = preprocess_mask(mask, model_config)
    np.testing.assert_array_equal(output.numpy()[0], mask[1::2, 1::2])
    output.zero_()
    assert mask.any()


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), np.uint8),
        None,
    ],
)
def test_preprocess_rejects_bad_images(image):
    with pytest.raises((TypeError, ValueError)):
        preprocess_image(image)


def test_preprocess_rejects_bad_masks():
    with pytest.raises((TypeError, ValueError)):
        preprocess_mask(np.zeros((10, 10), np.uint8))


@pytest.fixture
def checkpoint(tmp_path, model_config):
    model = PageSegmenter(model_config)
    path = tmp_path / "region.pt"
    state = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": {"model_config": asdict(model_config)},
        "model_state": model.state_dict(),
        "optimizer_state": {},
        "epoch": 0,
        "history": [],
        "initial_validation": {"loss": 1.0},
        "best_validation": {"loss": 1.0},
        "best_epoch": 0,
        "manifest_sha256": "a" * 64,
        "provenance": {},
        "torch_rng_state": torch.get_rng_state(),
        "shuffle_rng_state": torch.Generator().get_state(),
        "dataset_counts": {"train": 2, "validation": 2, "test": 2},
    }
    _save_checkpoint(path, state)
    return path, state


def test_load_probability_contract_preserves_rng_and_matches_model(checkpoint):
    path, _ = checkpoint
    state = torch.get_rng_state().clone()
    loaded = load_segmenter(path)
    assert torch.equal(state, torch.get_rng_state())
    image = np.full((50, 40, 3), 180, np.uint8)
    actual = loaded.predict_probability(image)
    assert actual.dtype == np.float32 and actual.shape == loaded.config.input_size
    assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
    with torch.inference_mode():
        expected = loaded.model(preprocess_image(image, loaded.config)[None]).sigmoid()[0, 0]
    np.testing.assert_array_equal(actual, expected.numpy())
    assert len(loaded.model_sha256) == 64 and loaded.model_sha256[:12] in loaded.model_id
    assert loaded.load_seconds >= 0 and not loaded.model.training
    assert torch.equal(state, torch.get_rng_state())


def test_requires_hash_sidecar_and_rejects_mismatch(checkpoint):
    path, _ = checkpoint
    sidecar = path.with_suffix(".pt.sha256")
    sidecar.write_text("0" * 64)
    with pytest.raises(FlattenerError, match="SHA-256"):
        read_checkpoint(path)
    sidecar.unlink()
    with pytest.raises(FlattenerError):
        read_checkpoint(path)


@pytest.mark.parametrize("mutation", ["schema", "shape"])
def test_rejects_invalid_checkpoint_content(checkpoint, mutation):
    path, state = checkpoint
    if mutation == "schema":
        state["schema_version"] = "future-0.1"
    else:
        state["model_state"]["head.bias"] = torch.zeros(3)
    _save_checkpoint(path, state)
    before = torch.get_rng_state().clone()
    with pytest.raises(FlattenerError):
        load_segmenter(path)
    assert torch.equal(before, torch.get_rng_state())
