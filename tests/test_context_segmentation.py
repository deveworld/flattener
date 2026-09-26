"""Context architecture, explicit devices, and frozen-state compatibility."""

from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from flattener.regions.context_segmentation import (
    MAX_PARAMETERS,
    MODEL_VERSION,
    PREPROCESS_VERSION,
    ContextPageSegmenter,
    ContextSegmenterConfig,
    parameter_count,
    preprocess_image,
    preprocess_mask,
)
from flattener.regions.segmentation import (
    PREPROCESS_VERSION as V1_PREPROCESS_VERSION,
)
from flattener.regions.segmentation import PageSegmenter, SegmenterConfig
from flattener.regions.segmentation import preprocess_image as v1_preprocess_image
from flattener.regions.segmentation import preprocess_mask as v1_preprocess_mask

pytestmark = pytest.mark.training


@pytest.mark.parametrize("channels", [(8, 16, 32, 64), (1, 2, 3, 4), (16, 24, 40, 72)])
def test_config_version_and_exact_parameter_count(channels):
    config = ContextSegmenterConfig(channels=channels)
    model = ContextPageSegmenter(config)
    assert asdict(config)["model_version"] == MODEL_VERSION == "region-context-unet-2.0"
    assert parameter_count(config) == sum(item.numel() for item in model.parameters())
    assert parameter_count(config) <= MAX_PARAMETERS
    assert all(
        item.device.type == "cpu" and item.dtype == torch.float32 for item in model.parameters()
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"input_size": (7, 96)},
        {"input_size": (513, 96)},
        {"input_size": [128, 96]},
        {"input_size": (True, 96)},
        {"channels": (8, 16, 32)},
        {"channels": [8, 16, 32, 64]},
        {"channels": (0, 1, 2, 3)},
        {"channels": (1, 2, 3, True)},
        {"channels": (1, 2, 3, 129)},
        {"channels": (128, 128, 128, 128)},
        {"model_version": "region-unet-1.0"},
        {"model_version": None},
    ],
)
def test_config_rejects_invalid_sizes_architecture_and_resource_limit(changes):
    with pytest.raises(ValueError):
        ContextSegmenterConfig(**changes)


@pytest.mark.parametrize("config", [SegmenterConfig(), {}, "context"])
def test_factory_rejects_noncontext_configs(config):
    with pytest.raises(TypeError):
        ContextPageSegmenter(config)


@pytest.mark.parametrize("size", [(8, 8), (17, 13), (128, 96)])
def test_single_output_shape_gradient_and_forward_does_not_touch_rng(size):
    config = ContextSegmenterConfig(input_size=size, channels=(2, 3, 4, 5))
    model = ContextPageSegmenter(config)
    images = torch.rand(2, 3, *size, requires_grad=True)
    rng = torch.get_rng_state().clone()
    logits = model(images)
    assert logits.shape == (2, 1, *size) and logits.dtype == torch.float32
    assert bool(torch.isfinite(logits).all())
    logits.sum().backward()
    assert bool(torch.isfinite(images.grad).all())
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(isinstance(module, (nn.Dropout, nn.BatchNorm2d)) for module in model.modules())


def test_scene_context_is_broadcast_and_connects_distant_input_to_output():
    config = ContextSegmenterConfig(channels=(1, 1, 1, 1))
    model = ContextPageSegmenter(config)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.fill_(0.05)
                module.bias.fill_(0.1)
        # Isolate the global context input of decoder3: channels are local
        # bottleneck, local skip, pooled context, one channel each in this test.
        model.decoder3[0].weight[:, :2].zero_()
    observed = []
    hook = model.decoder3.register_forward_pre_hook(
        lambda module, values: observed.append(values[0])
    )
    image = torch.full((1, 3, 128, 96), 0.5, requires_grad=True)
    try:
        output = model(image)
    finally:
        hook.remove()
    context = observed[0][:, -1:]
    torch.testing.assert_close(context, context[:, :, :1, :1].expand_as(context), rtol=0, atol=0)
    output[0, 0, 0, 0].backward()
    assert float(image.grad[0, :, -1, -1].sum()) > 0


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.zeros(1, 3, 128, 96, dtype=torch.float64),
        torch.zeros(0, 3, 128, 96),
        torch.zeros(1, 1, 128, 96),
        torch.zeros(1, 3, 127, 96),
        torch.full((1, 3, 128, 96), float("nan")),
        torch.full((1, 3, 128, 96), float("inf")),
        torch.full((1, 3, 128, 96), -0.01),
        torch.full((1, 3, 128, 96), 1.01),
    ],
)
def test_invalid_forward_input_rejected(invalid):
    with pytest.raises((TypeError, ValueError)):
        ContextPageSegmenter()(invalid)


def test_preprocessing_is_bit_exact_v1_and_does_not_share_caller_memory():
    config = ContextSegmenterConfig(input_size=(17, 13))
    old_config = SegmenterConfig(input_size=config.input_size)
    image = np.random.default_rng(31).integers(0, 256, (60, 80, 3), dtype=np.uint8)[::-1]
    mask = image[:, :, 0] > 128
    image_before, mask_before = image.copy(), mask.copy()
    new_image, new_mask = preprocess_image(image, config), preprocess_mask(mask, config)
    assert PREPROCESS_VERSION == V1_PREPROCESS_VERSION
    assert torch.equal(new_image, v1_preprocess_image(image, old_config))
    assert torch.equal(new_mask, v1_preprocess_mask(mask, old_config))
    new_image.zero_()
    new_mask.zero_()
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_context_construction_and_forward_do_not_change_existing_v1_state_or_numerics():
    old = PageSegmenter(SegmenterConfig(input_size=(16, 16), channels=(2, 4, 4))).eval()
    image = torch.full((1, 3, 16, 16), 0.4)
    before = {key: value.clone() for key, value in old.state_dict().items()}
    with torch.inference_mode():
        original_prediction = old(image).clone()
        context = ContextPageSegmenter(
            ContextSegmenterConfig(input_size=(16, 16), channels=(2, 4, 4, 8))
        ).eval()
        context(image)
        repeated_prediction = old(image)
    assert torch.equal(original_prediction, repeated_prediction)
    assert all(torch.equal(value, old.state_dict()[key]) for key, value in before.items())
    with pytest.raises(RuntimeError):
        context.load_state_dict(old.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        old.load_state_dict(context.state_dict(), strict=True)


def test_default_torch_initialization_is_reproducible_not_zero_head():
    rng = torch.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(410)
        first = ContextPageSegmenter()
        torch.random.default_generator.manual_seed(410)
        second = ContextPageSegmenter()
    finally:
        torch.set_rng_state(rng)
    assert any(bool(torch.count_nonzero(parameter)) for parameter in first.head.parameters())
    assert all(
        torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items()
    )


def test_model_parameters_stay_float32_without_changing_callers_default_dtype():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = ContextPageSegmenter()
        assert torch.get_default_dtype() == torch.float64
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    finally:
        torch.set_default_dtype(original)


def test_public_and_trusted_cpu_paths_match_historical_forward_and_state_dict():
    config = ContextSegmenterConfig(input_size=(17, 13), channels=(2, 3, 4, 5))
    model = ContextPageSegmenter(config).to("cpu")
    image = torch.rand(2, 3, *config.input_size)
    expected_keys = {
        f"{block}.{index}.{name}"
        for block in (
            "encoder1", "encoder2", "encoder3", "bottom",
            "decoder3", "decoder2", "decoder1",
        )
        for index in (0, 2)
        for name in ("weight", "bias")
    } | {"head.weight", "head.bias"}
    state = {key: value.clone() for key, value in model.state_dict().items()}
    assert set(state) == expected_keys
    restored = ContextPageSegmenter(config)
    restored.load_state_dict(state, strict=True)
    # Keep the historical operation order explicit: this checks more than two
    # entry points delegating to the same implementation.
    with torch.inference_mode():
        first = model.encoder1(image)
        second = model.encoder2(F.avg_pool2d(first, 2))
        third = model.encoder3(F.avg_pool2d(second, 2))
        bottom = model.bottom(F.avg_pool2d(third, 2))
        up = F.interpolate(bottom, size=third.shape[-2:], mode="bilinear", align_corners=False)
        context = model.scene_pool(bottom).expand(-1, -1, *third.shape[-2:])
        up = model.decoder3(torch.cat((up, third, context), dim=1))
        up = F.interpolate(up, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up = model.decoder2(torch.cat((up, second), dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        legacy = model.head(model.decoder1(torch.cat((up, first), dim=1)))
        assert torch.equal(model(image), legacy)
        assert torch.equal(model._forward_trusted(image), legacy)
        assert torch.equal(restored(image), legacy)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())


def test_trusted_forward_has_no_host_scalar_reads_or_value_validation(monkeypatch):
    model = ContextPageSegmenter(
        ContextSegmenterConfig(input_size=(8, 8), channels=(1, 1, 1, 1))
    )
    image = torch.full((1, 3, 8, 8), 0.5)

    def forbidden(*args, **kwargs):
        pytest.fail("trusted forward must not read tensor values on the host")

    with monkeypatch.context() as patch:
        for name in ("__bool__", "item", "cpu", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch, "isfinite", forbidden)
        output = model._forward_trusted(image)
    assert output.shape == (1, 1, 8, 8)


@pytest.mark.parametrize("method", ["forward", "_forward_trusted"])
@pytest.mark.parametrize("invalid", ["device", "dtype", "shape", "type", "parameters"])
def test_public_and_trusted_paths_reject_invalid_metadata_without_accelerator(method, invalid):
    model = ContextPageSegmenter(
        ContextSegmenterConfig(input_size=(8, 8), channels=(1, 1, 1, 1))
    )
    image = torch.zeros(1, 3, 8, 8)
    if invalid == "device":
        image = torch.empty(1, 3, 8, 8, device="meta")
    elif invalid == "dtype":
        image = image.double()
    elif invalid == "shape":
        image = image[:, :, :7]
    elif invalid == "type":
        image = None
    elif invalid == "parameters":
        model.encoder1.double()
    with pytest.raises((TypeError, ValueError)):
        getattr(model, method)(image)


def test_unsupported_model_device_is_rejected_before_tensor_values_are_read():
    model = ContextPageSegmenter(
        ContextSegmenterConfig(input_size=(8, 8), channels=(1, 1, 1, 1))
    ).to("meta")
    with pytest.raises(TypeError, match="model device must be CPU, CUDA, or XPU"):
        model(torch.empty(1, 3, 8, 8, device="meta"))


@pytest.mark.parametrize("device", ["cuda:0", "xpu:0"])
def test_explicit_accelerator_public_and_trusted_forward_backward(device):
    """Opt-in tiny device smoke; does not fit a dataset or run an optimizer."""
    backend = getattr(torch, device.split(":")[0], None)
    if backend is None or not backend.is_available():
        pytest.skip(f"{device} is unavailable")
    with torch.random.fork_rng(devices=[], device_type="cpu"):
        model = ContextPageSegmenter(
            ContextSegmenterConfig(input_size=(9, 11), channels=(1, 2, 3, 4))
        ).to(device)
    image = torch.full((1, 3, 9, 11), 0.5, device=device, requires_grad=True)
    public = model(image)
    trusted = model._forward_trusted(image)
    torch.testing.assert_close(public, trusted, rtol=0, atol=0)
    trusted.square().mean().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert image.grad is not None and bool(torch.isfinite(image.grad).all())
    with pytest.raises(TypeError, match="model device"):
        model(image.detach().cpu())
    for value in (float("nan"), float("inf"), -0.01, 1.01):
        with pytest.raises(ValueError, match="finite and in"):
            model(torch.full_like(image, value))
