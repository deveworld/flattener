"""Forward-only contracts: no optimizer steps, backward calls, or CPU training."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from flattener.learning.model import CoordinatePredictor, ModelConfig, coarse_identity
from flattener.learning.spatial import (
    MODEL_VERSION,
    PAGE_MODEL_VERSION,
    DeviceCoordinatePredictor,
    PageCoordinatePredictor,
    SpatialCoordinatePredictor,
    make_coordinate_model,
    parameter_count,
)

SMALL = ModelConfig(input_size=(24, 18), grid_size=(5, 3), channels=(4, 6, 8), hidden=16)


@pytest.fixture(autouse=True)
def isolate_torch_rng_and_forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("this forward-only suite must never run backward or an optimizer")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


def test_default_spatial_architecture_and_parameter_count():
    model = SpatialCoordinatePredictor()
    assert model.config == ModelConfig()
    assert MODEL_VERSION == "spatial-coordinate-1.0"
    assert parameter_count(model) == 205_762
    assert len(model.encoder) == 3
    assert model.decoder[0].in_channels == 146
    assert "xy" not in model.state_dict()
    assert model.xy.dtype == torch.float32
    assert model.xy.device.type == "cpu"
    model.requires_grad_(False)
    assert parameter_count(model) == 205_762


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_zero_head_is_identity_for_distinct_images_without_mutation(architecture):
    model = make_coordinate_model(architecture, SMALL).eval()
    images = torch.stack((torch.zeros(3, 24, 18), torch.ones(3, 24, 18)))
    original = images.clone()
    with torch.inference_mode():
        residual = model(images)
    assert residual.shape == (2, 2, 5, 3)
    assert residual.dtype == torch.float32
    assert residual.device == images.device
    assert torch.count_nonzero(residual).item() == 0
    assert torch.equal(images, original)
    assert all(parameter.grad is None for parameter in model.parameters())


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
    # A nonzero final layer exercises the inherited computation rather than
    # merely comparing two all-zero initial predictions.
    with torch.no_grad():
        legacy.head[-1].weight.fill_(0.031)
        legacy.head[-1].bias.fill_(-0.012)
    model.load_state_dict(legacy.state_dict(), strict=True)
    images = torch.rand(2, 3, *SMALL.input_size)
    with torch.inference_mode():
        assert torch.equal(model(images), legacy(images))


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_output_is_unbounded_and_deterministic_without_training(architecture):
    model = make_coordinate_model(architecture, SMALL).eval()
    output = model.decoder[-1] if architecture == "spatial" else model.head[-1]
    with torch.no_grad():
        output.bias.fill_(2.5)
    images = torch.full((1, 3, *SMALL.input_size), 0.25)
    before_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        first = model(images)
        second = model(images)
    assert torch.equal(first, second)
    assert torch.equal(first, torch.full_like(first, 2.5))
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_spatial_decoder_receives_all_levels_half_pixel_xy_and_global_context():
    model = SpatialCoordinatePredictor(SMALL).eval()
    images = torch.linspace(0, 1, 2 * 3 * 24 * 18).reshape(2, 3, 24, 18)
    captured = []
    handle = model.decoder[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].clone())
    )
    try:
        with torch.inference_mode():
            model(images)
            encoded = images
            expected = []
            for block in model.encoder:
                encoded = block(encoded)
                expected.append(
                    F.interpolate(encoded, size=(5, 3), mode="bilinear", align_corners=False)
                )
            xy = coarse_identity((5, 3)).permute(2, 0, 1).unsqueeze(0).expand(2, -1, -1, -1)
            context = encoded.mean((-2, -1), keepdim=True).expand(-1, -1, 5, 3)
            expected = torch.cat((*expected, xy, context), dim=1)
    finally:
        handle.remove()
    assert len(captured) == 1
    assert torch.equal(captured[0], expected)
    assert captured[0].shape == (2, 28, 5, 3)
    torch.testing.assert_close(xy[0, 0, 0], torch.tensor([-2 / 3, 0.0, 2 / 3]))
    torch.testing.assert_close(xy[0, 1, :, 0], torch.tensor([-0.8, -0.4, 0, 0.4, 0.8]))


def test_spatial_encoder_can_condition_output_on_pixels_without_optimization():
    model = SpatialCoordinatePredictor(SMALL).eval()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.fill_(0.01)
                layer.bias.zero_()
    images = torch.stack((torch.zeros(3, 24, 18), torch.ones(3, 24, 18)))
    with torch.inference_mode():
        output = model(images)
    assert not torch.equal(output[0], output[1])


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
@pytest.mark.parametrize(
    "images,error",
    [
        ([], TypeError),
        (torch.zeros(1, 3, 24, 18, dtype=torch.float64), TypeError),
        (torch.zeros(1, 3, 24, 18, dtype=torch.int64), TypeError),
        (torch.zeros(1, 3, 24, 18, device="meta"), TypeError),
        (torch.zeros(1, 3, 24, 18).to_sparse(), TypeError),
        (torch.zeros(3, 24, 18), ValueError),
        (torch.zeros(0, 3, 24, 18), ValueError),
        (torch.zeros(1, 4, 24, 18), ValueError),
        (torch.zeros(1, 3, 18, 24), ValueError),
        (torch.full((1, 3, 24, 18), float("nan")), ValueError),
        (torch.full((1, 3, 24, 18), float("inf")), ValueError),
        (torch.full((1, 3, 24, 18), -0.01), ValueError),
        (torch.full((1, 3, 24, 18), 1.01), ValueError),
    ],
)
def test_invalid_input_contracts(architecture, images, error):
    with pytest.raises(error):
        make_coordinate_model(architecture, SMALL)(images)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_model_dtype_and_device_are_not_silently_changed(architecture):
    images = torch.zeros(1, 3, *SMALL.input_size)
    model = make_coordinate_model(architecture, SMALL).double()
    with pytest.raises(TypeError, match="model parameters and buffers"):
        model(images)
    assert next(model.parameters()).dtype == torch.float64
    model = make_coordinate_model(architecture, SMALL).to("meta")
    with pytest.raises(ValueError, match="same CPU, CUDA or XPU device"):
        model(images)
    assert next(model.parameters()).device.type == "meta"


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_noncontiguous_strided_input_is_valid(architecture):
    images = torch.zeros(1, 3, 18, 24).transpose(-1, -2)
    assert not images.is_contiguous()
    with torch.inference_mode():
        output = make_coordinate_model(architecture, SMALL)(images)
    assert output.shape == (1, 2, 5, 3)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
@pytest.mark.parametrize(
    "changes",
    [
        {"input_size": (2049, 2)},
        {"input_size": (1025, 1024)},
        {"grid_size": (257, 2)},
        {"grid_size": (129, 128)},
        {"channels": (257, 2, 3)},
        {"hidden": 2049},
        {"channels": (256, 256, 256), "hidden": 2048, "grid_size": (128, 128)},
    ],
)
def test_bounded_configuration_rejected_before_rng_consumption(architecture, changes):
    config = replace(SMALL, **changes)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="limit"):
        make_coordinate_model(architecture, config)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_configuration_type_and_corrupted_frozen_config_are_rejected(architecture):
    with pytest.raises(TypeError, match="ModelConfig"):
        make_coordinate_model(architecture, {})
    corrupt = replace(SMALL)
    object.__setattr__(corrupt, "hidden", False)
    with pytest.raises(ValueError, match="hidden"):
        make_coordinate_model(architecture, corrupt)


@pytest.mark.parametrize("architecture", ["unknown", "GLOBAL", "", None, 1, ["global"]])
def test_architecture_does_not_fall_back(architecture):
    with pytest.raises(ValueError, match="architecture"):
        make_coordinate_model(architecture, SMALL)


def test_parameter_count_requires_module():
    with pytest.raises(TypeError, match="Module"):
        parameter_count({})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_optional_cuda_forward_and_explicit_device_contract(architecture):
    model = make_coordinate_model(architecture, SMALL).to("cuda").eval()
    images = torch.zeros(2, 3, *SMALL.input_size, device="cuda")
    with torch.inference_mode():
        output = model(images)
    assert output.device == images.device
    assert output.dtype == torch.float32
    assert torch.count_nonzero(output).item() == 0
    with pytest.raises(ValueError, match="same CPU, CUDA or XPU device"):
        model(images.cpu())


def test_xpu_metadata_guard_without_initializing_or_running_an_accelerator(monkeypatch):
    from types import SimpleNamespace

    from torch._subclasses import fake_tensor

    from flattener.learning import spatial

    monkeypatch.setattr(fake_tensor, "init_gpu_context", lambda device: None)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "_lazy_init", lambda: pytest.fail("real XPU initialization"))
    mode = fake_tensor.FakeTensorMode()

    def tensor(shape, device="xpu:0"):
        return fake_tensor.FakeTensor(mode, torch.empty(shape, device="meta"), torch.device(device))

    images, parameter = tensor((1, 3, *SMALL.input_size)), tensor((1,))
    model = SimpleNamespace(parameters=lambda: iter((parameter,)), buffers=lambda: iter(()))
    spatial._validate_images(model, SMALL, images, check_values=False)
    parameter = tensor((1,), "xpu:1")
    with pytest.raises(ValueError, match="same CPU, CUDA or XPU device"):
        spatial._validate_images(model, SMALL, images, check_values=False)
    # A malformed XPU residual must reach the ordinary shape guard, not fail
    # merely because XPU is absent from the backend allowlist.
    with pytest.raises(ValueError, match="bounded shape"):
        spatial.expand_coordinate_residual(tensor((1, 2, 1, 1)), (5, 3))


def test_page_default_parameter_count_and_odd_input_logit_shape():
    model = make_coordinate_model("page")
    assert isinstance(model, PageCoordinatePredictor)
    assert PAGE_MODEL_VERSION == "page-coordinate-1.0"
    assert parameter_count(model) == 278_123
    assert model.page_head[0].in_channels == sum(ModelConfig().channels)
    odd = PageCoordinatePredictor(replace(SMALL, input_size=(25, 19)))
    with torch.inference_mode():
        residual, logits = odd.forward_with_page(torch.ones(2, 3, 25, 19))
    assert residual.shape == (2, 2, 5, 3)
    assert logits.shape == (2, 1, 7, 5)
    assert torch.count_nonzero(residual) == 0
    assert torch.isfinite(logits).all()


def test_page_attention_uses_one_encoder_pass_and_preserves_background_context():
    model = PageCoordinatePredictor(SMALL).eval()
    counts = [0, 0, 0]
    encoded, pooling = [], []

    def encoder_hook(index):
        def hook(module, inputs, result):
            counts[index] += 1
            if index == 2:
                encoded.append(result.clone())
        return hook

    handles = [block.register_forward_hook(encoder_hook(i))
               for i, block in enumerate(model.encoder)]
    handles.append(model.pool.register_forward_pre_hook(
        lambda module, inputs: pooling.append(inputs[0].clone())
    ))
    try:
        images = torch.rand(2, 3, *SMALL.input_size)
        with torch.inference_mode():
            residual, logits = model.forward_with_page(images)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == [1, 1, 1]
    attention = F.interpolate(logits.sigmoid(), size=encoded[0].shape[-2:],
                              mode="bilinear", align_corners=False)
    torch.testing.assert_close(pooling[0][:, :-1], encoded[0] * (0.5 + attention))
    torch.testing.assert_close(pooling[0][:, -1:], attention)
    assert torch.all((0 <= attention) & (attention <= 1))
    with torch.inference_mode():
        assert torch.equal(model(images), residual)


def test_page_logits_and_coordinates_can_depend_on_pixels_without_optimization():
    model = PageCoordinatePredictor(SMALL).eval()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                layer.weight.fill_(0.01)
                layer.bias.zero_()
    images = torch.stack((torch.zeros(3, 24, 18), torch.ones(3, 24, 18)))
    with torch.inference_mode():
        residual, logits = model.forward_with_page(images)
    assert not torch.equal(logits[0], logits[1])
    assert not torch.equal(residual[0], residual[1])
