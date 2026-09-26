"""Independent crop direction: physical square views and classifier contracts."""

from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from flattener.regions.context_segmentation import ContextPageSegmenter, ContextSegmenterConfig
from flattener.regions.orientation import (
    MAX_PARAMETERS,
    OrientationConfig,
    PageOrientationClassifier,
    parameter_count,
    square_page_view,
)


@pytest.fixture
def tiny_config():
    return OrientationConfig(input_size=32, channels=(2, 3, 4, 5), hidden=8)


def _gradient_image(height=15, width=21):
    rows, columns = np.mgrid[:height, :width]
    return np.stack((10 + 8 * columns, 20 + 10 * rows, 5 + columns + rows), axis=-1).astype(
        np.uint8
    )


def test_config_defaults_are_serializable_and_immutable():
    config = OrientationConfig()
    assert asdict(config) == {
        "input_size": 160,
        "channels": (8, 16, 32, 64),
        "hidden": 128,
    }
    with pytest.raises(FrozenInstanceError):
        config.hidden = 64


@pytest.mark.parametrize(
    "changes",
    [
        {"input_size": 31},
        {"input_size": 513},
        {"input_size": True},
        {"input_size": 160.0},
        {"input_size": (160, 160)},
        {"channels": [8, 16, 32, 64]},
        {"channels": (8, 16, 32)},
        {"channels": (0, 1, 2, 3)},
        {"channels": (1, 2, 3, 129)},
        {"channels": (1, 2, 3, True)},
        {"channels": (1, 2, 3, 4.0)},
        {"hidden": 3},
        {"hidden": 1025},
        {"hidden": True},
        {"hidden": 128.0},
        {"channels": (128, 128, 128, 128), "hidden": 1024},
    ],
)
def test_config_rejects_invalid_fields_before_model_allocation(changes):
    with pytest.raises(ValueError):
        OrientationConfig(**changes)


@pytest.mark.parametrize("config", [{}, "orientation", ContextSegmenterConfig()])
def test_model_rejects_wrong_configuration_type(config):
    with pytest.raises(TypeError):
        PageOrientationClassifier(config)


@pytest.mark.parametrize("size", [32, 33, 40])
def test_four_logits_for_all_supported_pooling_parities(size):
    config = OrientationConfig(input_size=size, channels=(1, 2, 3, 4), hidden=4)
    model = PageOrientationClassifier(config)
    assert parameter_count(config) == sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count(config) <= MAX_PARAMETERS
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, size, size))
    assert output.shape == (1, 4) and bool(torch.isfinite(output).all())


def test_physical_square_crop_uses_pixel_centers_without_aspect_stretch():
    image = _gradient_image()
    # An 8 x 6 box expands vertically by one source pixel at either edge.
    actual = square_page_view(image, (3.0, 2.0, 11.0, 8.0), size=8)
    expected = torch.from_numpy(image[1:9, 3:11].copy()).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
    assert actual.shape == (3, 8, 8)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert actual.is_contiguous() and not actual.requires_grad
    assert not actual.is_inference()


def test_fractional_box_obeys_exact_physical_center_mapping():
    image = _gradient_image()
    box = (3.25, 2.75, 9.75, 7.25)
    size = 13
    actual = square_page_view(image, box, size=size).numpy()
    # Square origin is (3.25, 1.75), side 6.5. A source pixel's physical
    # center is (column + 0.5, row + 0.5); these ramps interpolate exactly.
    y, x = np.mgrid[:size, :size]
    columns = 3.25 + (x + 0.5) * 6.5 / size - 0.5
    rows = 1.75 + (y + 0.5) * 6.5 / size - 0.5
    expected = np.stack((10 + 8 * columns, 20 + 10 * rows, 5 + columns + rows)) / 255
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


def test_virtual_square_preserves_extent_and_fills_outside_source_with_gray():
    image = np.zeros((5, 11, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    actual = square_page_view(image, (0.0, 0.0, 11.0, 5.0), size=11)
    torch.testing.assert_close(actual[:, :3], torch.full((3, 3, 11), 0.5), rtol=0, atol=2e-7)
    torch.testing.assert_close(actual[:, 8:], torch.full((3, 3, 11), 0.5), rtol=0, atol=2e-7)
    expected_middle = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(actual[:, 3:8], expected_middle, rtol=0, atol=2e-7)


def test_half_cell_boundary_blends_constant_gray_without_clamping_square():
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    actual = square_page_view(image, (0, 0, 5, 3), size=10)
    expected = torch.tensor([0.5, 0.375, 0.125, 0, 0, 0, 0, 0.125, 0.375, 0.5])
    torch.testing.assert_close(actual[0, :, 5], expected, rtol=0, atol=2e-7)


@pytest.mark.parametrize("quarters", [0, 1, 2, 3])
def test_square_view_commutes_with_quarter_rotations_of_nonsquare_image(quarters):
    image = _gradient_image()
    box = (3.25, 1.5, 19.75, 13.25)
    baseline = square_page_view(image, box, size=32)
    rotated = image
    rotated_box = box
    for _ in range(quarters):
        left, top, right, bottom = rotated_box
        width = rotated.shape[1]
        rotated_box = (top, width - right, bottom, width - left)
        rotated = np.rot90(rotated)
    actual = square_page_view(rotated, rotated_box, size=32)
    expected = torch.rot90(baseline, quarters, dims=(-2, -1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)


def test_downsampling_antialiases_high_frequency_content():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, ::3] = 255
    actual = square_page_view(image, (0, 0, 96, 96), size=16)
    # Direct bilinear sampling at this scale aliases these stripes to 0.5.
    # The antialiased interior retains their one-third spatial average.
    torch.testing.assert_close(
        actual[:, 2:-2, 2:-2], torch.full((3, 12, 12), 1 / 3), rtol=0, atol=0.005
    )


@pytest.mark.parametrize("container", [tuple, list, np.array])
def test_crop_accepts_numeric_box_containers_and_never_aliases_input(container):
    image = _gradient_image()[::-1]
    before = image.copy()
    actual = square_page_view(image, container([2.0, 2.0, 10.0, 10.0]), size=8)
    actual.zero_()
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize(
    "image",
    [
        None,
        [[[0, 0, 0]]],
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((0, 8, 3), dtype=np.uint8),
    ],
)
def test_crop_rejects_invalid_rgb_images(image):
    with pytest.raises((TypeError, ValueError)):
        square_page_view(image, (0, 0, 8, 8))


@pytest.mark.parametrize(
    "box",
    [
        None,
        "0,0,8,8",
        (0, 0, 8),
        np.zeros((2, 2)),
        (0, 0, 8, float("nan")),
        (0, 0, float("inf"), 8),
        (False, 0, 8, 8),
        (0, 0, np.bool_(True), 8),
        ("0", 0, 8, 8),
        (-0.01, 0, 8, 8),
        (0, -0.01, 8, 8),
        (0, 0, 8.01, 8),
        (0, 0, 8, 8.01),
        (2, 2, 2, 6),
        (2, 6, 6, 2),
    ],
)
def test_crop_rejects_invalid_or_out_of_image_boxes(box):
    with pytest.raises((TypeError, ValueError)):
        square_page_view(np.zeros((8, 8, 3), dtype=np.uint8), box)


@pytest.mark.parametrize("size", [1, 1025, True, 160.0, (160, 160)])
def test_crop_rejects_invalid_output_size(size):
    with pytest.raises(ValueError):
        square_page_view(np.zeros((8, 8, 3), dtype=np.uint8), (0, 0, 8, 8), size=size)


def test_encoder_state_keys_and_shapes_support_context_checkpoint_initialization(tiny_config):
    model = PageOrientationClassifier(tiny_config)
    context = ContextPageSegmenter(
        ContextSegmenterConfig(input_size=(32, 32), channels=tiny_config.channels)
    )
    prefixes = ("encoder1.", "encoder2.", "encoder3.", "bottom.")
    source = {key: value for key, value in context.state_dict().items() if key.startswith(prefixes)}
    destination = {
        key: value for key, value in model.state_dict().items() if key.startswith(prefixes)
    }
    assert set(source) == set(destination)
    assert len(source) == 16
    assert all(source[key].shape == destination[key].shape for key in source)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = model.load_state_dict(source, strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == set(before) - set(source)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in source.items())
    assert all(
        torch.equal(value, model.state_dict()[key])
        for key, value in before.items()
        if key not in source
    )
    assert not any(key.startswith(("decoder", "scene_pool")) for key in model.state_dict())


def test_public_and_trusted_cpu_outputs_match_and_preserve_rng_and_state(tiny_config):
    model = PageOrientationClassifier(tiny_config).eval()
    image = torch.rand(2, 3, 32, 32)
    rng = torch.get_rng_state().clone()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        public = model(image)
        trusted = model._forward_trusted(image)
    assert public.shape == (2, 4) and public.dtype == torch.float32
    assert bool(torch.isfinite(public).all())
    torch.testing.assert_close(public, trusted, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert not any(isinstance(module, (nn.Dropout, nn.BatchNorm2d)) for module in model.modules())


def test_trusted_path_never_reads_host_tensor_values(tiny_config, monkeypatch):
    model = PageOrientationClassifier(tiny_config)
    image = torch.full((1, 3, 32, 32), 0.5)

    def forbidden(*args, **kwargs):
        pytest.fail("trusted forward may only inspect tensor metadata")

    with monkeypatch.context() as patch:
        for name in ("__bool__", "item", "cpu", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch, "isfinite", forbidden)
        output = model._forward_trusted(image)
    assert output.shape == (1, 4)


@pytest.mark.parametrize("method", ["forward", "_forward_trusted"])
@pytest.mark.parametrize("invalid", ["type", "dtype", "device", "shape", "empty", "parameters"])
def test_forward_paths_reject_invalid_metadata(tiny_config, method, invalid):
    model = PageOrientationClassifier(tiny_config)
    image = torch.zeros(1, 3, 32, 32)
    if invalid == "type":
        image = None
    elif invalid == "dtype":
        image = image.double()
    elif invalid == "device":
        image = torch.empty(1, 3, 32, 32, device="meta")
    elif invalid == "shape":
        image = image[:, :, :31]
    elif invalid == "empty":
        image = image[:0]
    else:
        model.encoder1.double()
    with pytest.raises((TypeError, ValueError)):
        getattr(model, method)(image)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_public_forward_rejects_nonfinite_or_out_of_range_rgb(tiny_config, value):
    with pytest.raises(ValueError):
        PageOrientationClassifier(tiny_config)(torch.full((1, 3, 32, 32), value))


def test_unsupported_model_device_is_rejected_before_value_access(tiny_config):
    model = PageOrientationClassifier(tiny_config).to("meta")
    with pytest.raises(TypeError, match="CPU, CUDA, or XPU"):
        model(torch.empty(1, 3, 32, 32, device="meta"))


def test_construction_does_not_depend_on_default_dtype(tiny_config):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = PageOrientationClassifier(tiny_config)
        assert torch.get_default_dtype() == torch.float64
        assert all(
            parameter.device.type == "cpu" and parameter.dtype == torch.float32
            for parameter in model.parameters()
        )
    finally:
        torch.set_default_dtype(original)


@pytest.mark.training
def test_cross_entropy_backpropagates_to_all_parameters_and_square_view(tiny_config):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(462)
        model = PageOrientationClassifier(tiny_config)
    view = square_page_view(_gradient_image(), (1, 1, 20, 14), size=32)
    images = torch.stack([torch.rot90(view, k, (-2, -1)) for k in range(4)]).requires_grad_()
    # Starting from an upright fixture, CCW k augmentation has clockwise
    # direction label -k mod 4, matching the classifier's public convention.
    targets = (-torch.arange(4)) % 4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    loss = F.cross_entropy(model._forward_trusted(images), targets)
    assert bool(torch.isfinite(loss)) and float(loss.detach()) > 0
    loss.backward()
    assert images.grad is not None and bool(torch.isfinite(images.grad).all())
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    optimizer.step()
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())


@pytest.mark.training
@pytest.mark.parametrize("device", ["cuda:0", "xpu:0"])
def test_accelerator_three_adam_steps_use_safe_foreach_false(tiny_config, device):
    """Tiny synthetic fixture only; callers can deselect this to reserve the GPU."""
    backend = getattr(torch, device.split(":")[0], None)
    if backend is None or not backend.is_available():
        pytest.skip(f"{device} is unavailable")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(928)
        model = PageOrientationClassifier(tiny_config).to(device)
    images = torch.linspace(0, 1, 4 * 3 * 32 * 32, device=device).reshape(4, 3, 32, 32)
    targets = torch.arange(4, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    with torch.no_grad():
        torch.testing.assert_close(model(images), model._forward_trusted(images), rtol=0, atol=0)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model._forward_trusted(images), targets)
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        optimizer.step()
    backend.synchronize()
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, model.parameters(), strict=True)
    )


def _asymmetric_orientation_inputs(size=32):
    axis = torch.linspace(0, 1, size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    first = torch.stack((0.1 + 0.7 * x.square(), 0.2 + 0.6 * y, 0.1 + 0.4 * x * y))
    second = torch.stack((0.8 - 0.5 * y.square(), 0.1 + 0.3 * x, 0.2 + 0.6 * x * y))
    return torch.stack((first, second))


def test_rotation_probabilities_match_independent_softmax_and_class_index_oracle(
    tiny_config, monkeypatch
):
    model = PageOrientationClassifier(tiny_config).eval()
    images = _asymmetric_orientation_inputs()

    def asymmetric_logits(batch):
        value = batch[:, 0, 0, 0]
        return torch.stack(
            (9 * value, 3 - 4 * value, 7 * (1 - value).square(), 1 + (6 * value).sin()),
            dim=1,
        )

    monkeypatch.setattr(model, "_forward_trusted", asymmetric_logits)
    expected = np.zeros((2, 4), dtype=np.float64)
    native = []
    for k in range(4):
        logits = asymmetric_logits(torch.rot90(images, k, (-2, -1))).double().numpy()
        exponential = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability = exponential / exponential.sum(axis=1, keepdims=True)
        native.append(probability)
        # CCW k makes native class q-k. Canonical class q therefore reads
        # index (q-k) from that view, independent of the implementation's roll.
        for q in range(4):
            expected[:, q] += probability[:, (q - k) % 4] / 4
    with torch.inference_mode():
        actual = model.rotation_probabilities(images)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-7)
    assert actual.dtype == torch.float32
    assert not np.allclose(expected, np.mean(native, axis=0), rtol=0, atol=0.01)
    assert not np.allclose(expected, 0.25, rtol=0, atol=0.01)


@pytest.mark.parametrize("quarters", [0, 1, 2, 3])
def test_rotation_probability_distribution_is_equivariant_for_asymmetric_model_and_input(
    tiny_config, quarters
):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1403)
        model = PageOrientationClassifier(tiny_config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(mean=0, std=0.4)
    images = _asymmetric_orientation_inputs()
    with torch.inference_mode():
        canonical = model.rotation_probabilities(images)
        actual = model.rotation_probabilities(torch.rot90(images, quarters, (-2, -1)))
        native = model(images)
        rotated_native = model(torch.rot90(images, 1, (-2, -1)))
    assert not torch.allclose(native, rotated_native, rtol=0, atol=1e-4)
    assert not torch.allclose(canonical, torch.full_like(canonical, 0.25), rtol=0, atol=1e-5)
    # The physical clockwise class decreases by the applied CCW quarter.
    expected = canonical[:, [(q + quarters) % 4 for q in range(4)]]
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
    assert bool(torch.isfinite(actual).all()) and bool((actual >= 0).all())
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(2), rtol=0, atol=2e-7)


def test_rotation_probabilities_preserve_state_rng_device_and_inference_contract(tiny_config):
    model = PageOrientationClassifier(tiny_config).eval()
    images = _asymmetric_orientation_inputs()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        public = model.rotation_probabilities(images)
        trusted = model._rotation_probabilities_trusted(images)
    assert public.shape == (2, 4) and public.device.type == "cpu"
    assert public.dtype == torch.float32 and not public.requires_grad
    torch.testing.assert_close(public, trusted, rtol=0, atol=0)
    assert not model.training
    assert all(parameter.grad is None for parameter in parameters)
    assert tuple(model.parameters()) == parameters
    assert len(state) == 20 and set(model.state_dict()) == set(state)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert torch.equal(rng, torch.get_rng_state())
    restored = PageOrientationClassifier(tiny_config).eval()
    restored.load_state_dict(state, strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored.rotation_probabilities(images), public, rtol=0, atol=0)


def test_rotation_probability_trusted_path_never_reads_host_tensor_values(tiny_config, monkeypatch):
    model = PageOrientationClassifier(tiny_config).eval()
    images = _asymmetric_orientation_inputs()

    def forbidden(*args, **kwargs):
        pytest.fail("trusted rotation probabilities may only inspect tensor metadata")

    with monkeypatch.context() as patch, torch.inference_mode():
        for name in ("__bool__", "item", "cpu", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch, "isfinite", forbidden)
        actual = model._rotation_probabilities_trusted(images)
    assert actual.shape == (2, 4)


@pytest.mark.parametrize("method", ["rotation_probabilities", "_rotation_probabilities_trusted"])
@pytest.mark.parametrize("invalid", ["type", "dtype", "device", "shape", "empty", "parameters"])
def test_rotation_probability_paths_reject_invalid_metadata(tiny_config, method, invalid):
    model = PageOrientationClassifier(tiny_config)
    images = torch.zeros(1, 3, 32, 32)
    if invalid == "type":
        images = None
    elif invalid == "dtype":
        images = images.double()
    elif invalid == "device":
        images = torch.empty(1, 3, 32, 32, device="meta")
    elif invalid == "shape":
        images = images[:, :, :31]
    elif invalid == "empty":
        images = images[:0]
    else:
        model.encoder1.double()
    with pytest.raises((TypeError, ValueError)):
        getattr(model, method)(images)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_public_rotation_probabilities_reject_bad_values_before_network(
    tiny_config, value, monkeypatch
):
    model = PageOrientationClassifier(tiny_config)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid public RGB input must be rejected before network evaluation")

    monkeypatch.setattr(model, "_forward_trusted", forbidden)
    with pytest.raises(ValueError):
        model.rotation_probabilities(torch.full((1, 3, 32, 32), value))


def test_rotation_probabilities_ignore_process_default_float64(tiny_config):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = PageOrientationClassifier(tiny_config).eval()
        images = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
        with torch.inference_mode():
            probabilities = model.rotation_probabilities(images)
        assert probabilities.device.type == "cpu" and probabilities.dtype == torch.float32
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original)


def test_xpu_rotation_probabilities_match_cpu_and_preserve_equivariance(tiny_config):
    """Inference-only parity check; no backward pass or optimizer is run."""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("xpu:0 is unavailable")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(731)
        cpu_model = PageOrientationClassifier(tiny_config).eval()
    xpu_model = PageOrientationClassifier(tiny_config).to("xpu:0").eval()
    xpu_model.load_state_dict(cpu_model.state_dict(), strict=True)
    cpu_images = _asymmetric_orientation_inputs()
    xpu_images = cpu_images.to("xpu:0")
    with torch.inference_mode():
        expected = cpu_model.rotation_probabilities(cpu_images)
        actual = xpu_model.rotation_probabilities(xpu_images)
        trusted = xpu_model._rotation_probabilities_trusted(xpu_images)
        for k in range(4):
            rotated = xpu_model.rotation_probabilities(torch.rot90(xpu_images, k, (-2, -1)))
            torch.testing.assert_close(rotated, actual.roll(-k, dims=-1), rtol=0, atol=2e-7)
    torch.xpu.synchronize()
    assert actual.device.type == "xpu" and actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, trusted, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in xpu_model.parameters())
