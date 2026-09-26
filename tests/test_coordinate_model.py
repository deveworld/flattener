"""Coordinate model contracts: configuration, preprocessing, expansion and forward passes.

Tests marked ``training`` call backward or an optimizer; every other test runs forward only.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from flattener.coordinates import resize_backward_map
from flattener.learning.model import (
    CoordinatePredictor,
    ModelConfig,
    coarse_identity,
    dense_maps,
    preprocess_image,
)
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
def isolate_torch_rng_and_forbid_training(request, monkeypatch):
    if request.node.get_closest_marker("training") is None:

        def forbidden(*args, **kwargs):
            pytest.fail("forward-only tests must never run backward or an optimizer")

        monkeypatch.setattr(torch.Tensor, "backward", forbidden)
        monkeypatch.setattr(torch.autograd, "backward", forbidden)
        monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": (0, 72)},
        {"input_size": [96, 72]},
        {"hidden": 128.0},
    ],
)
def test_invalid_configuration_is_rejected_before_parameter_allocation(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_configuration_requires_the_public_type():
    with pytest.raises(TypeError, match="ModelConfig"):
        CoordinatePredictor({"hidden": 128})


def test_coarse_identity_explicit_non_square_centers_and_singleton_axes():
    expected = torch.tensor(
        [[[-2 / 3, -0.5], [0, -0.5], [2 / 3, -0.5]], [[-2 / 3, 0.5], [0, 0.5], [2 / 3, 0.5]]]
    )
    actual = coarse_identity((2, 3))
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(coarse_identity((1, 1)), torch.zeros(1, 1, 2))
    torch.testing.assert_close(
        coarse_identity((1, 4)),
        torch.tensor([[[-0.75, 0], [-0.25, 0], [0.25, 0], [0.75, 0]]]),
    )
    actual.zero_()
    torch.testing.assert_close(coarse_identity((2, 3)), expected, rtol=0, atol=1e-7)


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


def test_dense_maps_preserve_non_square_source_pixel_shifts_and_do_not_clip():
    residual = torch.zeros(2, 2, 2, 3)
    # Source height=20,width=10: +3 px in x and -4 px in y.
    residual[0, 0] = 0.6
    residual[0, 1] = -0.4
    residual[1, 0] = 2.0
    original = residual.clone()
    result = dense_maps(residual, (7, 11))
    yy, xx = np.mgrid[:7, :11]
    expected = np.stack((2 * (xx + 0.5) / 11 - 1 + 0.6, 2 * (yy + 0.5) / 7 - 1 - 0.4), axis=-1)
    np.testing.assert_allclose(result[0].numpy(), expected, rtol=0, atol=2e-7)
    assert result[1, ..., 0].max() > 1
    torch.testing.assert_close(residual, original, rtol=0, atol=0)


def test_dense_maps_matches_renderer_expansion_for_varying_residuals():
    residual = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 60 - 0.2
    coarse = coarse_identity((3, 4)) + residual[0].permute(1, 2, 0)
    expected = resize_backward_map(coarse.numpy(), (7, 11))
    np.testing.assert_allclose(
        dense_maps(residual, (7, 11))[0].numpy(), expected, rtol=0, atol=2e-7
    )


@pytest.mark.training
def test_expansion_preserves_autograd_with_known_constant_gradient():
    residual = torch.zeros(1, 2, 2, 3, requires_grad=True)
    dense_maps(residual, (4, 6)).sum().backward()
    torch.testing.assert_close(residual.grad, torch.full_like(residual, 4.0), rtol=0, atol=1e-6)


@pytest.mark.training
def test_model_learns_through_image_encoder_after_initial_identity_step():
    config = ModelConfig(input_size=(24, 18), grid_size=(3, 2), channels=(4, 6, 8), hidden=16)
    with torch.random.fork_rng():
        torch.manual_seed(47)
        model = CoordinatePredictor(config)
        images = torch.rand(2, 3, 24, 18)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target = torch.full((2, 2, 3, 2), 0.125)
    initial_loss = None
    for step in range(3):
        optimizer.zero_grad()
        loss = (model(images) - target).square().mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step == 1:
            assert torch.count_nonzero(model.encoder[0].weight.grad) > 0
        optimizer.step()
    assert loss.item() < initial_loss
    with torch.no_grad():
        predictions = model(torch.stack((images[0], torch.zeros_like(images[0]))))
    assert not torch.equal(predictions[0], predictions[1])


def test_white_image_downsampling_stays_in_unit_range_despite_roundoff():
    # Analytic white remains white. This nonintegral reduction reproduced a
    # real-photo preprocessing rejection without bundling any external image.
    image = np.full((129, 177, 3), 255, dtype=np.uint8)
    original = image.copy()
    preview = preprocess_image(image)
    assert bool((preview >= 0).all()) and bool((preview <= 1).all())
    torch.testing.assert_close(preview, torch.ones_like(preview), rtol=0, atol=1e-6)
    with torch.inference_mode():
        assert torch.isfinite(CoordinatePredictor()(preview[None])).all()
    np.testing.assert_array_equal(image, original)


def test_preprocessing_preserves_rgb_and_constant_values_without_aliasing():
    image = np.empty((13, 7, 3), dtype=np.uint8)
    image[:] = (0, 128, 255)
    original = image.copy()
    image.flags.writeable = False
    config = ModelConfig(input_size=(9, 11))
    result = preprocess_image(image[:, ::-1], config)
    assert result.shape == (3, 9, 11)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    torch.testing.assert_close(
        result, torch.tensor([0, 128 / 255, 1])[:, None, None].expand_as(result)
    )
    result.zero_()
    np.testing.assert_array_equal(image, original)


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


@pytest.mark.parametrize(
    "image,error",
    [
        ([], TypeError),
        (np.zeros((0, 3, 3), np.uint8), ValueError),
    ],
)
def test_preprocessing_rejects_invalid_arrays(image, error):
    with pytest.raises(error):
        preprocess_image(image)


@pytest.mark.parametrize(
    "images,error",
    [
        (np.zeros((1, 3, 96, 72), np.float32), TypeError),
        (torch.full((1, 3, 96, 72), 1.01), ValueError),
    ],
)
def test_model_rejects_invalid_tensor_contract(images, error):
    with pytest.raises(error):
        CoordinatePredictor()(images)


@pytest.mark.parametrize(
    "residual,size,error",
    [
        (np.zeros((1, 2, 2, 3), np.float32), (7, 11), TypeError),
        (torch.full((1, 2, 2, 3), float("inf")), (7, 11), ValueError),
    ],
)
def test_dense_maps_rejects_invalid_residuals_or_sizes(residual, size, error):
    with pytest.raises(error):
        dense_maps(residual, size)


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


@pytest.mark.parametrize("architecture", ["global", "spatial"])
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


def test_output_is_unbounded_and_deterministic_without_training():
    model = make_coordinate_model("global", SMALL).eval()
    output = model.head[-1]
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


@pytest.mark.parametrize(
    "images,error",
    [
        ([], TypeError),
        (torch.zeros(1, 3, 24, 18, dtype=torch.float64), TypeError),
        (torch.zeros(1, 3, 24, 18, device="meta"), TypeError),
        (torch.zeros(3, 24, 18), ValueError),
        (torch.full((1, 3, 24, 18), float("nan")), ValueError),
        (torch.full((1, 3, 24, 18), 1.01), ValueError),
    ],
)
def test_invalid_input_contracts(images, error):
    with pytest.raises(error):
        make_coordinate_model("global", SMALL)(images)


def test_model_dtype_and_device_are_not_silently_changed():
    images = torch.zeros(1, 3, *SMALL.input_size)
    model = make_coordinate_model("global", SMALL).double()
    with pytest.raises(TypeError, match="model parameters and buffers"):
        model(images)
    assert next(model.parameters()).dtype == torch.float64
    model = make_coordinate_model("global", SMALL).to("meta")
    with pytest.raises(ValueError, match="same CPU, CUDA or XPU device"):
        model(images)
    assert next(model.parameters()).device.type == "meta"


def test_noncontiguous_strided_input_is_valid():
    images = torch.zeros(1, 3, 18, 24).transpose(-1, -2)
    assert not images.is_contiguous()
    with torch.inference_mode():
        output = make_coordinate_model("global", SMALL)(images)
    assert output.shape == (1, 2, 5, 3)


@pytest.mark.parametrize(
    "changes",
    [
        {"input_size": (2049, 2)},
        {"hidden": 2049},
        {"channels": (256, 256, 256), "hidden": 2048, "grid_size": (128, 128)},
    ],
)
def test_bounded_configuration_rejected_before_rng_consumption(changes):
    config = replace(SMALL, **changes)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="limit"):
        make_coordinate_model("global", config)
    assert torch.equal(before, torch.get_rng_state())


def test_configuration_type_and_corrupted_frozen_config_are_rejected():
    with pytest.raises(TypeError, match="ModelConfig"):
        make_coordinate_model("global", {})
    corrupt = replace(SMALL)
    object.__setattr__(corrupt, "hidden", False)
    with pytest.raises(ValueError, match="hidden"):
        make_coordinate_model("global", corrupt)


@pytest.mark.parametrize("architecture", ["unknown", None])
def test_architecture_does_not_fall_back(architecture):
    with pytest.raises(ValueError, match="architecture"):
        make_coordinate_model(architecture, SMALL)


def test_parameter_count_requires_module():
    with pytest.raises(TypeError, match="Module"):
        parameter_count({})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_optional_cuda_forward_and_explicit_device_contract():
    model = make_coordinate_model("global", SMALL).to("cuda").eval()
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

    handles = [
        block.register_forward_hook(encoder_hook(i)) for i, block in enumerate(model.encoder)
    ]
    handles.append(
        model.pool.register_forward_pre_hook(
            lambda module, inputs: pooling.append(inputs[0].clone())
        )
    )
    try:
        images = torch.rand(2, 3, *SMALL.input_size)
        with torch.inference_mode():
            residual, logits = model.forward_with_page(images)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == [1, 1, 1]
    attention = F.interpolate(
        logits.sigmoid(), size=encoded[0].shape[-2:], mode="bilinear", align_corners=False
    )
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
