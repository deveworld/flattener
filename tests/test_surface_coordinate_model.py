"""Independent low-DOF maps and boundary-safe expansion; training stays opt-in."""

from dataclasses import replace
from math import comb

import numpy as np
import pytest
import torch

from flattener.learning import spatial
from flattener.learning.model import CoordinatePredictor, ModelConfig, coarse_identity, dense_maps
from flattener.learning.spatial import (
    ProjectiveCoordinatePredictor,
    SurfaceCoordinatePredictor,
    expand_coordinate_residual,
    make_coordinate_model,
    parameter_count,
)

SMALL = ModelConfig(input_size=(24, 18), grid_size=(5, 3), channels=(4, 6, 8), hidden=16)
ARCHITECTURES = ("projective", "surface")


@pytest.fixture(autouse=True)
def isolate_rng_and_forbid_unmarked_training(request, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("optimizers and unmarked backward calls are forbidden")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    if request.node.get_closest_marker("training") is None:
        monkeypatch.setattr(torch.Tensor, "backward", forbidden)
        monkeypatch.setattr(torch.autograd, "backward", forbidden)
        monkeypatch.setattr(torch.autograd, "grad", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


def _centers(size):
    height, width = size
    y, x = np.mgrid[:height, :width]
    return np.stack((2 * (x + 0.5) / width - 1, 2 * (y + 0.5) / height - 1), axis=-1)


def _reference_map(raw, size):
    """NumPy closed form, never calling the production decoder or torch resize."""
    raw = np.asarray(raw, dtype=np.float64)
    grid = _centers(size)
    if raw.size == 48:
        coefficients = 0.2 * np.tanh(raw[8:]).reshape(2, 4, 5)
        lx = 2 * np.linalg.norm(np.diff(coefficients, axis=2), axis=0).max()
        ly = 1.5 * np.linalg.norm(np.diff(coefficients, axis=1), axis=0).max()
        bound = np.hypot(lx, ly)
        coefficients *= min(1, 0.7 / bound) if bound else 1
        u, v = (grid[..., 0] + 1) / 2, (grid[..., 1] + 1) / 2
        offset = np.zeros_like(grid)
        for row in range(4):
            by = comb(3, row) * v**row * (1 - v) ** (3 - row)
            for col in range(5):
                bx = comb(4, col) * u**col * (1 - u) ** (4 - col)
                offset += (by * bx)[..., None] * coefficients[:, row, col]
        grid = grid + offset
    t = np.tanh(raw[:2])
    theta = np.pi * np.tanh(raw[2])
    sx, sy = np.exp(3 * np.tanh(raw[3:5]))
    shear = 2 * np.tanh(raw[5])
    perspective = np.tanh(raw[6:8])
    perspective *= 0.6 / (1 + np.abs(perspective).sum())
    x, y = sx * grid[..., 0] + shear * grid[..., 1], sy * grid[..., 1]
    rotated = np.stack(
        (np.cos(theta) * x - np.sin(theta) * y, np.sin(theta) * x + np.cos(theta) * y),
        axis=-1,
    )
    return (t + rotated / (1 + grid @ perspective)[..., None]).astype(np.float32)


def _minimum_cell_determinant(coordinates):
    a, b = coordinates[:-1, :-1], coordinates[:-1, 1:]
    c, d = coordinates[1:, :-1], coordinates[1:, 1:]
    determinants = []
    for dx in (b - a, d - c):
        for dy in (c - a, d - b):
            determinants.append(dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0])
    return torch.stack(determinants).min().item()


@pytest.mark.parametrize("size", [(5, 3), (17, 23), (24, 18)])
def test_legacy_expansion_is_bitwise_identical_to_original_dense_maps(size):
    residual = torch.randn(2, 2, *SMALL.grid_size)
    before = residual.clone()
    actual = expand_coordinate_residual(residual, size, mode="legacy")
    assert torch.equal(actual, dense_maps(residual, size))
    assert torch.equal(actual, expand_coordinate_residual(residual, size))
    assert torch.equal(residual, before)
    assert actual.shape == (2, *size, 2) and actual.dtype == torch.float32


@pytest.mark.parametrize(
    "matrix,translation",
    [
        ([[1, 0], [0, 1]], [0, 0]),
        ([[-1, 0], [0, -1]], [0, 0]),  # 180-degree rotation, including boundary strips.
        ([[0.6, 0.3], [-0.2, 1.4]], [2.5, -3.0]),
    ],
)
def test_linear_expansion_preserves_affine_maps_to_physical_image_edges(matrix, translation):
    matrix, translation = np.asarray(matrix), np.asarray(translation)
    coarse = _centers(SMALL.grid_size)
    residual = torch.from_numpy((coarse @ matrix.T + translation - coarse).astype(np.float32))
    residual = residual.permute(2, 0, 1).unsqueeze(0)
    size = (23, 17)
    actual = expand_coordinate_residual(residual, size, mode="linear")
    expected = torch.from_numpy((_centers(size) @ matrix.T + translation).astype(np.float32))
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=2e-6)
    assert _minimum_cell_determinant(actual[0]) > 0
    if translation[0] > 1:
        assert actual[..., 0].min() > 1  # No source-bound clipping.


def test_linear_extrapolation_matches_independent_nonaffine_four_neighbor_formula():
    values = np.random.default_rng(14).normal(size=(2, 3, 4)).astype(np.float32)
    residual = torch.from_numpy(values[None])
    size = (7, 9)
    expected = _centers(size)
    for y in range(size[0]):
        cy = (y + 0.5) * 3 / size[0] - 0.5
        iy = int(np.clip(np.floor(cy), 0, 1))
        fy = cy - iy
        for x in range(size[1]):
            cx = (x + 0.5) * 4 / size[1] - 0.5
            ix = int(np.clip(np.floor(cx), 0, 2))
            fx = cx - ix
            expected[y, x] += (
                values[:, iy, ix] * (1 - fx) * (1 - fy)
                + values[:, iy, ix + 1] * fx * (1 - fy)
                + values[:, iy + 1, ix] * (1 - fx) * fy
                + values[:, iy + 1, ix + 1] * fx * fy
            )
    actual = expand_coordinate_residual(residual, size, mode="linear")
    torch.testing.assert_close(
        actual[0], torch.from_numpy(expected.astype(np.float32)), rtol=0, atol=2e-6
    )


@pytest.mark.parametrize("mode", ["legacy", "linear"])
def test_expansion_accepts_noncontiguous_strided_residual(mode):
    residual = torch.randn(2, 2, 3, 5).transpose(-1, -2)
    assert not residual.is_contiguous()
    actual = expand_coordinate_residual(residual, (17, 11), mode=mode)
    assert torch.equal(
        actual, expand_coordinate_residual(residual.contiguous(), (17, 11), mode=mode)
    )


@pytest.mark.parametrize(
    "residual",
    [
        [],
        torch.zeros(1, 2, 2, 2, dtype=torch.float64),
        torch.zeros(1, 2, 2, 2, dtype=torch.int64),
        torch.zeros(1, 2, 2, 2, device="meta"),
        torch.zeros(1, 2, 2, 2).to_sparse(),
        torch.zeros(2, 2, 2),
        torch.zeros(0, 2, 2, 2),
        torch.zeros(1, 3, 2, 2),
        torch.zeros(1, 2, 1, 2),
        torch.zeros(1, 2, 257, 2),
        torch.zeros(1, 2, 129, 128),
        torch.full((1, 2, 2, 2), float("nan")),
        torch.full((1, 2, 2, 2), float("inf")),
    ],
)
def test_expansion_rejects_invalid_tensor_contracts(residual):
    with pytest.raises((TypeError, ValueError)):
        expand_coordinate_residual(residual, (256, 256), mode="linear")


@pytest.mark.parametrize(
    "size", [(1, 2), (2, 1), (2, 8193), (8192, 8192), (True, 2), (2.0, 2), [2, 2], (2,), None]
)
def test_expansion_rejects_output_shape_or_resource_overflow(size):
    with pytest.raises((TypeError, ValueError)):
        expand_coordinate_residual(torch.zeros(1, 2, 2, 2), size, mode="linear")


def test_expansion_output_budget_includes_batch():
    with pytest.raises(ValueError):
        expand_coordinate_residual(torch.zeros(17, 2, 2, 2), (1024, 1024), mode="linear")


@pytest.mark.parametrize("mode", ["", "cubic", None, 1, ["linear"]])
def test_expansion_never_silently_falls_back_for_unknown_mode(mode):
    with pytest.raises((TypeError, ValueError)):
        expand_coordinate_residual(torch.zeros(1, 2, 2, 2), (3, 3), mode=mode)


@pytest.mark.parametrize(
    "architecture,model_type,outputs,count",
    [
        ("projective", ProjectiveCoordinatePredictor, 8, 204_440),
        ("surface", SurfaceCoordinatePredictor, 48, 209_600),
    ],
)
def test_parameter_counts_and_state_keys_keep_the_shared_global_trunk(
    architecture, model_type, outputs, count
):
    model = make_coordinate_model(architecture)
    assert isinstance(model, model_type)
    assert parameter_count(model) == count
    assert model.head[3].out_features == outputs
    torch.manual_seed(73)
    legacy = CoordinatePredictor(SMALL)
    torch.manual_seed(73)
    candidate = make_coordinate_model(architecture, SMALL)
    assert candidate.state_dict().keys() == legacy.state_dict().keys()
    for name, value in legacy.state_dict().items():
        if not name.startswith("head.3."):
            assert torch.equal(candidate.state_dict()[name], value), name


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_zero_head_is_exact_identity_in_both_public_forward_paths(architecture):
    model = make_coordinate_model(architecture, SMALL).eval()
    images = torch.stack((torch.zeros(3, 24, 18), torch.ones(3, 24, 18)))
    before = images.clone()
    with torch.inference_mode():
        residual, dense = model.forward_dense(images, (19, 23))
        coarse = model(images)
    assert residual.shape == coarse.shape == (2, 2, 5, 3)
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(coarse, residual)
    expected = torch.from_numpy(_centers((19, 23)).astype(np.float32)).expand(2, -1, -1, -1)
    assert torch.equal(dense, expected)
    assert torch.equal(images, before)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("extreme", [False, True])
def test_direct_coarse_and_dense_predictions_match_independent_bounded_formula(
    architecture, extreme
):
    model = make_coordinate_model(architecture, SMALL).eval()
    raw = np.random.default_rng(28).normal(size=model.head[3].out_features).astype(np.float32)
    if extreme:
        raw = np.where(raw > 0, 12, -12).astype(np.float32)
    with torch.no_grad():
        model.head[3].bias.copy_(torch.from_numpy(raw))
    images = torch.zeros(1, 3, *SMALL.input_size)
    with torch.inference_mode():
        residual, dense = model.forward_dense(images, (47, 59))
        same_residual, grid = model.forward_dense(images, SMALL.grid_size)
    coarse_reference = _reference_map(raw, SMALL.grid_size) - _centers(SMALL.grid_size)
    torch.testing.assert_close(
        residual[0].permute(1, 2, 0),
        torch.from_numpy(coarse_reference.astype(np.float32)),
        rtol=0,
        atol=2e-5,
    )
    torch.testing.assert_close(
        dense[0], torch.from_numpy(_reference_map(raw, (47, 59))), rtol=0, atol=2e-5
    )
    assert torch.equal(residual, same_residual)
    torch.testing.assert_close(
        grid[0],
        residual[0].permute(1, 2, 0)
        + torch.from_numpy(_centers(SMALL.grid_size).astype(np.float32)),
        rtol=0,
        atol=2e-6,
    )
    assert _minimum_cell_determinant(dense[0]) > 0
    assert torch.isfinite(dense).all()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_manual_nonzero_weights_condition_direct_prediction_on_image_pixels(architecture):
    model = make_coordinate_model(architecture, SMALL).eval()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                layer.weight.fill_(0.01)
                layer.bias.zero_()
    images = torch.stack((torch.zeros(3, 24, 18), torch.ones(3, 24, 18)))
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        coarse, dense = model.forward_dense(images, (19, 23))
        repeated = model.forward_dense(images, (19, 23))
    assert not torch.equal(dense[0], dense[1])
    assert torch.equal(coarse, repeated[0]) and torch.equal(dense, repeated[1])
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_coarse_and_dense_paths_share_one_encoder_evaluation(architecture):
    model = make_coordinate_model(architecture, SMALL).eval()
    calls = []
    hook = model.encoder.register_forward_hook(lambda *args: calls.append(True))
    try:
        with torch.inference_mode():
            model.forward_dense(torch.zeros(1, 3, *SMALL.input_size), (19, 23))
    finally:
        hook.remove()
    assert len(calls) == 1


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_direct_models_reject_nonfinite_parameter_predictions(architecture, invalid):
    model = make_coordinate_model(architecture, SMALL)
    with torch.no_grad():
        model.head[3].bias[0] = invalid
    with pytest.raises(ValueError, match="finite"):
        model.forward_dense(torch.zeros(1, 3, *SMALL.input_size), (19, 23))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(1, 3, 24, 18, dtype=torch.float64),
        torch.zeros(1, 3, 24, 18).to_sparse(),
        torch.zeros(1, 3, 18, 24),
        torch.zeros(0, 3, 24, 18),
        torch.full((1, 3, 24, 18), float("nan")),
        torch.full((1, 3, 24, 18), 1.01),
    ],
)
def test_direct_models_validate_inputs(architecture, images):
    model = make_coordinate_model(architecture, SMALL)
    with pytest.raises((TypeError, ValueError)):
        model.forward_dense(images, (19, 23))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("size", [(4, 3), (5, 2), (5, 8193), (8192, 8192), [5, 3]])
def test_direct_models_validate_requested_output_size(architecture, size):
    model = make_coordinate_model(architecture, SMALL)
    with pytest.raises((TypeError, ValueError)):
        model.forward_dense(torch.zeros(1, 3, *SMALL.input_size), size)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_model_precision_and_device_are_never_silently_changed(architecture):
    images = torch.zeros(1, 3, *SMALL.input_size)
    with pytest.raises(TypeError):
        make_coordinate_model(architecture, SMALL).double().forward_dense(images, (19, 23))
    with pytest.raises(ValueError):
        make_coordinate_model(architecture, SMALL).to("meta").forward_dense(images, (19, 23))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize(
    "changes",
    [
        {"grid_size": (257, 2)},
        {"grid_size": (129, 128)},
        {"input_size": (2049, 2)},
        {"hidden": 2049},
    ],
)
def test_model_allocation_limits_are_checked_before_rng_consumption(architecture, changes):
    config = replace(SMALL, **changes)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        make_coordinate_model(architecture, config)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_cuda_forward_is_explicit_and_matches_cpu(architecture):
    cpu = make_coordinate_model(architecture, SMALL).eval()
    gpu = make_coordinate_model(architecture, SMALL).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    image = torch.rand(1, 3, *SMALL.input_size)
    with torch.inference_mode():
        expected = cpu.forward_dense(image, (19, 23))[1]
        actual = gpu.forward_dense(image.cuda(), (19, 23))[1]
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.training
@pytest.mark.parametrize("zero_head", [True, False])
def test_small_surface_and_linear_expansion_have_finite_gradients_without_optimizer(zero_head):
    model = make_coordinate_model("surface", SMALL)
    if not zero_head:
        with torch.no_grad():
            model.head[3].bias.copy_(torch.linspace(-0.15, 0.15, 48))
    _, dense = model.forward_dense(torch.full((1, 3, 24, 18), 0.5), (11, 9))
    dense.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head[3].bias.grad[:8].abs().sum() > 0
    assert model.head[3].bias.grad[8:].abs().sum() > 0
    residual = torch.randn(1, 2, 3, 4, requires_grad=True)
    expand_coordinate_residual(residual, (7, 9), mode="linear").square().mean().backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert residual.grad.abs().sum() > 0


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_decoder_constants_reuse_original_values_without_batch_or_checkpoint_state(
    architecture, monkeypatch
):
    model = make_coordinate_model(architecture, SMALL).eval()
    keys = tuple(model.state_dict())
    calls = []
    original = spatial.coarse_identity

    def counted(size):
        calls.append(size)
        return original(size)

    monkeypatch.setattr(spatial, "coarse_identity", counted)
    with torch.inference_mode():
        first = model.forward_dense(torch.zeros(1, 3, *SMALL.input_size), (19, 23))
        second = model.forward_dense(torch.zeros(3, 3, *SMALL.input_size), (19, 23))
    assert calls == [SMALL.grid_size, (19, 23)]
    assert len(model._constant_cache) == 2
    assert tuple(model.state_dict()) == keys
    assert tuple(model.named_buffers()) == ()
    assert torch.equal(second[0][0], first[0][0])
    assert torch.equal(second[1][0], first[1][0])
    for (size, device, dtype), constants in model._constant_cache.items():
        assert device == torch.device("cpu") and dtype == torch.float32
        assert torch.equal(constants[0], coarse_identity(size))
        assert all(not value.requires_grad and not value.is_inference() for value in constants)
        if architecture == "surface":
            for axis, degree, basis in ((constants[0][0, :, 0], 4, constants[1]),
                                        (constants[0][:, 0, 1], 3, constants[2])):
                position = (axis + 1) / 2
                expected = torch.stack([
                    comb(degree, i) * position**i * (1 - position)**(degree - i)
                    for i in range(degree + 1)
                ], dim=-1)
                assert torch.equal(basis, expected)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("size", [SMALL.grid_size, (19, 23)])
def test_forward_outputs_never_alias_cached_constants(architecture, size):
    model = make_coordinate_model(architecture, SMALL).eval()
    images = torch.zeros(1, 3, *SMALL.input_size)
    with torch.inference_mode():
        residual, dense = model.forward_dense(images, size)
        expected_residual, expected_dense = residual.clone(), dense.clone()
        residual.fill_(123)
        dense.fill_(-456)
        repeated = model.forward_dense(images, size)
    assert torch.equal(repeated[0], expected_residual)
    assert torch.equal(repeated[1], expected_dense)


@pytest.mark.parametrize("architecture", (*ARCHITECTURES, "oriented_surface"))
def test_decoder_cpu_constants_ignore_non_cpu_default_device_context(architecture):
    model = make_coordinate_model(architecture, SMALL)
    raw = torch.zeros((1, model.parameter_outputs), device="cpu")
    expected = coarse_identity((19, 23))
    with torch.device("meta"):
        constants = model._decoder_constants(raw, (19, 23))
        assert torch.get_default_device().type == "meta"
    assert all(value.device.type == "cpu" for value in constants)
    assert torch.equal(constants[0], expected)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_decoder_constant_lru_is_size_and_byte_bounded(architecture, monkeypatch):
    model = make_coordinate_model(architecture, SMALL)
    raw = torch.zeros(1, model.parameter_outputs)
    sizes = [(11, 9), (13, 9), (15, 9), (17, 9)]
    for size in sizes:
        model._decoder_constants(raw, size)
    model._decoder_constants(raw, sizes[0])  # The first entry is now most recently used.
    model._decoder_constants(raw, (19, 9))
    assert [key[0] for key in model._constant_cache] == [*sizes[2:], sizes[0], (19, 9)]
    assert len(model._constant_cache) == model._constant_cache_limit == 4
    assert model._constant_cache_max_bytes == 8 * 1024 * 1024
    assert model._constant_cache_bytes == sum(
        value.numel() * value.element_size()
        for constants in model._constant_cache.values() for value in constants
    )
    # An oversize request is usable for this call but never retained or allowed
    # to evict useful small constants. Test with a small budget, not large RAM.
    monkeypatch.setattr(model, "_constant_cache_max_bytes", model._constant_cache_bytes)
    keys = tuple(model._constant_cache)
    constants = model._decoder_constants(raw, (37, 31))
    assert constants[0].shape == (37, 31, 2)
    assert tuple(model._constant_cache) == keys
    # An individually fitting entry evicts enough old entries to respect bytes.
    model._decoder_constants(raw, (25, 17))
    assert model._constant_cache_bytes <= model._constant_cache_max_bytes
    assert len(model._constant_cache) < 4


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_decoder_cache_is_cleared_on_dtype_transition_and_state_load_is_compatible(architecture):
    model = make_coordinate_model(architecture, SMALL).eval()
    images = torch.zeros(1, 3, *SMALL.input_size)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.linspace(-0.2, 0.2, model.parameter_outputs))
        expected = model.forward_dense(images, (19, 23))
    checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
    assert model._constant_cache
    model.double()
    assert not model._constant_cache and model._constant_cache_bytes == 0
    with pytest.raises(TypeError, match="model parameters and buffers"):
        model.forward_dense(images, (19, 23))
    model.float()
    fresh = make_coordinate_model(architecture, SMALL).eval()
    fresh.load_state_dict(checkpoint, strict=True)
    with torch.no_grad():
        actual = model.forward_dense(images, (19, 23))
        loaded = fresh.forward_dense(images, (19, 23))
    for left, right, third in zip(expected, actual, loaded, strict=True):
        assert torch.equal(left, right) and torch.equal(left, third)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_validated_dense_path_matches_public_without_coarse_decode_or_host_checks(
    architecture, monkeypatch
):
    model = make_coordinate_model(architecture, SMALL).eval()
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.linspace(-0.2, 0.2, model.parameter_outputs))
    images = torch.full((2, 3, *SMALL.input_size), 0.5)
    with torch.inference_mode():
        expected = model.forward_dense(images, (19, 23))[1]
    calls, decode = [], model._decode

    def counted(raw, size, **kwargs):
        calls.append(size)
        return decode(raw, size, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("validated dense decoding must not synchronize tensor values to host")

    monkeypatch.setattr(model, "_decode", counted)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        with torch.inference_mode():
            actual, finite = model._forward_dense_validated(images, (19, 23))
    assert calls == [(19, 23)]
    assert torch.equal(actual, expected)
    assert finite.shape == () and finite.dtype == torch.bool and finite.device == images.device
    assert finite.item()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_validated_dense_finite_flag_includes_raw_parameters(architecture, invalid):
    model = make_coordinate_model(architecture, SMALL)
    with torch.no_grad():
        model.head[-1].bias[0] = invalid
        dense, finite = model._forward_dense_validated(
            torch.zeros(1, 3, *SMALL.input_size), (19, 23)
        )
    assert not finite.item()
    if abs(invalid) == float("inf"):
        assert torch.isfinite(dense).all()  # Saturation must not conceal the invalid raw value.


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_validated_dense_path_keeps_metadata_and_resource_guards(architecture):
    model = make_coordinate_model(architecture, SMALL)
    images = torch.zeros(1, 3, *SMALL.input_size)
    for bad_images in (images.double(), images[:, :, :23], images.to("meta"), []):
        with pytest.raises((TypeError, ValueError)):
            model._forward_dense_validated(bad_images, (19, 23))
    for size in ((4, 3), (19, 8193), (8192, 8192), [19, 23]):
        with pytest.raises(ValueError):
            model._forward_dense_validated(images, size)
    with pytest.raises(ValueError, match="16000000"):
        model._forward_dense_validated(images.expand(17, -1, -1, -1), (1024, 1024))


@pytest.mark.training
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_inference_warmed_cache_preserves_dense_only_gradients(architecture):
    model = make_coordinate_model(architecture, SMALL)
    uncached = make_coordinate_model(architecture, SMALL)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.linspace(-0.2, 0.2, model.parameter_outputs))
    uncached.load_state_dict(model.state_dict())
    uncached._constant_cache_max_bytes = 0
    images = torch.full((1, 3, *SMALL.input_size), 0.5)
    with torch.inference_mode():
        model.forward_dense(images, (19, 23))
    dense, finite = model._forward_dense_validated(images, (19, 23))
    reference = uncached.forward_dense(images, (19, 23))[1]
    assert finite.item() and torch.equal(dense, reference)
    dense.square().mean().backward()
    reference.square().mean().backward()
    for actual, expected in zip(model.parameters(), uncached.parameters(), strict=True):
        assert actual.grad is not None and expected.grad is not None
        assert torch.isfinite(actual.grad).all()
        assert torch.equal(actual.grad, expected.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_decoder_cache_device_roundtrip_and_validated_cuda_output(architecture):
    model = make_coordinate_model(architecture, SMALL).eval()
    images = torch.zeros(2, 3, *SMALL.input_size)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.linspace(-0.2, 0.2, model.parameter_outputs))
    with torch.inference_mode():
        expected = model.forward_dense(images, (19, 23))[1]
    model.cuda()
    assert not model._constant_cache
    gpu_images = images.cuda()
    with torch.inference_mode():
        public = model.forward_dense(gpu_images, (19, 23))[1]
        actual, finite = model._forward_dense_validated(gpu_images, (19, 23))
    assert finite.device == gpu_images.device and finite.item()
    assert torch.equal(actual, public)
    assert all(key[1] == gpu_images.device for key in model._constant_cache)
    torch.testing.assert_close(actual.cpu(), expected)
    model.cpu()
    assert not model._constant_cache and model._constant_cache_bytes == 0
    with torch.inference_mode():
        assert torch.equal(model.forward_dense(images, (19, 23))[1], expected)


@pytest.mark.training
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_cuda_inference_warmed_decoder_cache_is_safe_for_backward(architecture):
    model = make_coordinate_model(architecture, SMALL).cuda()
    images = torch.full((2, 3, *SMALL.input_size), 0.5, device="cuda")
    with torch.no_grad():
        model.head[-1].bias.copy_(
            torch.linspace(-0.2, 0.2, model.parameter_outputs, device="cuda")
        )
    with torch.inference_mode():
        model.forward_dense(images, (19, 23))
    dense, finite = model._forward_dense_validated(images, (19, 23))
    assert finite.item()
    dense.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head[-1].bias.grad.abs().sum() > 0


def _oriented_reference(raw, points, photo_size, quarter_turn):
    """Independent float64 Bernstein + physical-pixel clockwise homography."""
    raw = np.asarray(raw, dtype=np.float64)
    grid = np.array(points, dtype=np.float64, copy=True)
    coefficients = 0.2 * np.tanh(raw[8:]).reshape(2, 4, 5)
    lx = 2 * np.linalg.norm(np.diff(coefficients, axis=2), axis=0).max()
    ly = 1.5 * np.linalg.norm(np.diff(coefficients, axis=1), axis=0).max()
    bound = np.hypot(lx, ly)
    coefficients *= min(1, 0.7 / bound) if bound else 1
    u, v = (grid[..., 0] + 1) / 2, (grid[..., 1] + 1) / 2
    displacement = np.zeros_like(grid)
    for row in range(4):
        for col in range(5):
            basis = (comb(3, row) * v**row * (1 - v)**(3 - row)
                     * comb(4, col) * u**col * (1 - u)**(4 - col))
            displacement += basis[..., None] * coefficients[:, row, col]
    grid += displacement
    bounded = np.tanh(raw[:8])
    upper = np.array([[np.exp(3 * bounded[3]), 2 * bounded[5]],
                      [0, np.exp(3 * bounded[4])]])
    angle = quarter_turn * np.pi / 2 + np.pi / 4 * bounded[2]
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    physical = np.array([photo_size[1] / 2, photo_size[0] / 2])
    transformed = ((grid @ upper.T) * physical) @ rotation.T / physical
    perspective = 0.6 * bounded[6:8] / (1 + np.abs(bounded[6:8]).sum())
    return (bounded[:2] + transformed / (1 + grid @ perspective)[..., None]).astype(np.float32)


def _oriented_model(raw=None, quarter_turn=0, config=SMALL):
    model = make_coordinate_model("oriented_surface", config).eval()
    with torch.no_grad():
        model.orientation_head.weight.zero_()
        model.orientation_head.bias.zero_()
        if quarter_turn:
            model.orientation_head.bias[quarter_turn] = 3
        if raw is not None:
            model.head[3].bias.copy_(torch.as_tensor(raw, dtype=torch.float32))
    return model


def test_oriented_surface_preserves_legacy_trunk_initialization_and_adds_only_named_heads():
    torch.manual_seed(73)
    original = make_coordinate_model("surface", SMALL)
    torch.manual_seed(73)
    candidate = make_coordinate_model("oriented_surface", SMALL)
    assert isinstance(candidate, spatial.OrientedSurfaceCoordinatePredictor)
    expected_keys = set(original.state_dict()) | {
        "orientation_head.weight", "orientation_head.bias", "page_head.0.weight",
        "page_head.0.bias", "page_head.2.weight", "page_head.2.bias"}
    assert set(candidate.state_dict()) == expected_keys
    for name, tensor in original.state_dict().items():
        assert torch.equal(candidate.state_dict()[name], tensor), name
    assert candidate.orientation_head.in_features == SMALL.hidden
    assert candidate.orientation_head.out_features == 4 and candidate.head[3].out_features == 48
    assert candidate.page_head[0].in_channels == sum(SMALL.channels)
    assert candidate.page_head[0].out_channels == SMALL.channels[0]
    assert not candidate.orientation_head.weight.count_nonzero()
    assert not candidate.orientation_head.bias.count_nonzero()
    assert not candidate.head[3].weight.count_nonzero()
    assert not candidate.head[3].bias.count_nonzero()
    assert parameter_count(candidate) == parameter_count(original) + 68 + 657 == 6171


@pytest.mark.parametrize("photo_size", [(17, 29), (4000, 4000), (1, 8192)])
def test_oriented_identity_and_semantic_boundary_corners_are_exact(photo_size):
    config = replace(SMALL, input_size=(23, 17))
    model = _oriented_model(config=config)
    images = torch.stack((torch.zeros(3, 23, 17), torch.ones(3, 23, 17)))
    before, rng = images.clone(), torch.get_rng_state().clone()
    with torch.inference_mode():
        residual, dense, auxiliary = model.forward_with_geometry(
            images, (19, 23), photo_size=photo_size)
        pair = model.forward_dense(images, (19, 23), photo_size=photo_size)
        coarse = model(images, photo_size=photo_size)
    expected = torch.from_numpy(_centers((19, 23)).astype(np.float32)).expand(2, -1, -1, -1)
    corners = torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    assert torch.equal(dense, expected) and torch.equal(pair[1], dense)
    assert not residual.count_nonzero() and torch.equal(residual, pair[0])
    assert torch.equal(coarse, residual) and residual.shape == (2, 2, *SMALL.grid_size)
    assert set(auxiliary) == {"orientation_logits", "page_logits", "corners"}
    assert auxiliary["orientation_logits"].shape == (2, 4)
    assert not auxiliary["orientation_logits"].count_nonzero()
    assert auxiliary["page_logits"].shape == (2, 1, 6, 5)
    assert torch.equal(auxiliary["corners"], corners.expand(2, -1, -1))
    assert not torch.equal(auxiliary["corners"][:, 0], dense[:, 0, 0])
    assert torch.equal(images, before) and torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("quarter_turn", [0, 1, 2, 3])
@pytest.mark.parametrize("photo_size", [(128, 256), (256, 128)])
@pytest.mark.parametrize("extreme", [False, True])
def test_oriented_dense_coarse_and_boundary_maps_match_independent_physical_formula(
        quarter_turn, photo_size, extreme):
    raw = np.random.default_rng(37).normal(0, 0.3, 48).astype(np.float32)
    if extreme:
        raw = np.where(raw > 0, 12, -12).astype(np.float32)
    model = _oriented_model(raw, quarter_turn)
    images = torch.zeros(1, 3, *SMALL.input_size)
    with torch.inference_mode():
        residual, dense, auxiliary = model.forward_with_geometry(
            images, (31, 27), photo_size=photo_size)
        same_residual, coarse = model.forward_dense(images, SMALL.grid_size, photo_size=photo_size)
    expected = _oriented_reference(raw, _centers((31, 27)), photo_size, quarter_turn)
    expected_coarse = _oriented_reference(raw, _centers(SMALL.grid_size), photo_size, quarter_turn)
    expected_corners = _oriented_reference(
        raw, [[-1, -1], [1, -1], [1, 1], [-1, 1]], photo_size, quarter_turn)
    torch.testing.assert_close(dense[0], torch.from_numpy(expected), rtol=2e-6, atol=3e-5)
    torch.testing.assert_close(coarse[0], torch.from_numpy(expected_coarse),
                               rtol=2e-6, atol=3e-5)
    torch.testing.assert_close(auxiliary["corners"][0], torch.from_numpy(expected_corners),
                               rtol=2e-6, atol=3e-5)
    torch.testing.assert_close(residual[0].permute(1, 2, 0),
                               coarse[0] - coarse_identity(SMALL.grid_size), rtol=0, atol=0)
    assert torch.equal(residual, same_residual)
    assert auxiliary["orientation_logits"].argmax(1).item() == quarter_turn
    assert torch.isfinite(dense).all()
    if not extreme:
        assert _minimum_cell_determinant(dense[0]) > 0


@pytest.mark.parametrize("quarter_turn", [0, 1, 2, 3])
def test_oriented_quarter_turn_is_clockwise_in_physical_photo_not_preview_pixels(quarter_turn):
    model = _oriented_model(quarter_turn=quarter_turn)
    with torch.inference_mode():
        _, dense, auxiliary = model.forward_with_geometry(
            torch.zeros(1, 3, *SMALL.input_size), (17, 11), photo_size=(100, 200))
    grid = coarse_identity((17, 11))
    x, y = grid.unbind(-1)
    expected = ((x, y), (-0.5 * y, 2 * x), (-x, -y), (0.5 * y, -2 * x))[quarter_turn]
    torch.testing.assert_close(dense[0], torch.stack(expected, dim=-1), rtol=0, atol=0)
    if quarter_turn in (1, 3):
        assert dense.abs().max() > 1  # Physical rotation is not fit, clamped, or clipped.
    top_left = ([[-1, -1], [0.5, -2], [1, 1], [-0.5, 2]])[quarter_turn]
    assert torch.equal(auxiliary["corners"][0, 0], torch.tensor(top_left).float())


def test_oriented_direction_selection_is_per_image_with_first_index_ties(monkeypatch):
    model = _oriented_model()
    logits = torch.tensor([[7., 7., 0., 0.], [0., 8., 8., 0.],
                           [0., 0., 9., 9.], [0., 0., 0., 10.]])
    monkeypatch.setattr(model.orientation_head, "forward", lambda hidden: logits.clone())
    with torch.inference_mode():
        _, dense, auxiliary = model.forward_with_geometry(
            torch.zeros(4, 3, *SMALL.input_size), (19, 23), photo_size=(123, 217))
    assert torch.equal(auxiliary["orientation_logits"], logits)
    for index in range(4):
        expected = _oriented_reference(np.zeros(48), _centers((19, 23)), (123, 217), index)
        torch.testing.assert_close(dense[index], torch.from_numpy(expected), rtol=0, atol=2e-6)


def test_oriented_validated_metadata_and_multiscale_budget_reject_before_feature_allocation(
        monkeypatch):
    model = _oriented_model()
    images = torch.zeros(1, 3, *SMALL.input_size)
    for bad_images in (images.double(), images[:, :, :23], images.to("meta"), []):
        with pytest.raises((TypeError, ValueError)):
            model._forward_with_geometry_validated(bad_images, (19, 23), photo_size=(100, 200))
    for output_size in ((4, 3), (19, 8193), (8192, 8192), [19, 23]):
        with pytest.raises(ValueError):
            model._forward_with_geometry_validated(images, output_size, photo_size=(100, 200))
    for need_coarse in (None, 1, "yes"):
        with pytest.raises(TypeError):
            model._forward_with_geometry_validated(
                images, (19, 23), photo_size=(100, 200), need_coarse=need_coarse)
    big = _oriented_model(config=replace(SMALL, input_size=(512, 512)))
    monkeypatch.setattr(big, "_geometry_features",
                        lambda *args: pytest.fail("oversized feature allocation"))
    # Zero-stride metadata-only view: no large image or feature allocation.
    broadcast = torch.zeros(1, 3, 1, 1).expand(218, 3, 512, 512)
    with pytest.raises(ValueError, match="64000000"):
        big._forward_with_geometry_validated(broadcast, (5, 3), photo_size=(100, 200))


@pytest.mark.parametrize("method", ["forward", "forward_dense", "forward_with_geometry",
                                     "_forward_with_geometry_validated"])
def test_oriented_photo_size_is_mandatory(method):
    model = _oriented_model()
    arguments = [torch.zeros(1, 3, *SMALL.input_size)]
    if method != "forward":
        arguments.append((19, 23))
    with pytest.raises(TypeError):
        getattr(model, method)(*arguments)


@pytest.mark.parametrize("photo_size", [None, [100, 200], (100,), (100, 200, 3),
                                        (0, 200), (-1, 200), (True, 200), (100.0, 200),
                                        (8193, 1), (4097, 4096), (np.int64(100), 200)])
def test_oriented_rejects_invalid_photo_size_before_encoder(photo_size):
    model = _oriented_model()
    calls = []
    handle = model.encoder[0].register_forward_hook(lambda *args: calls.append(True))
    try:
        for name in ("forward_with_geometry", "_forward_with_geometry_validated"):
            with pytest.raises((TypeError, ValueError)):
                getattr(model, name)(torch.zeros(1, 3, *SMALL.input_size), (19, 23),
                                     photo_size=photo_size)
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize("component", ["geometry", "orientation", "page"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_oriented_public_rejects_nonfinite_auxiliary_and_geometry_heads(component, nonfinite):
    model = _oriented_model()
    layer = {"geometry": model.head[3], "orientation": model.orientation_head,
             "page": model.page_head[2]}[component]
    with torch.no_grad():
        layer.bias[0] = nonfinite
    images = torch.zeros(1, 3, *SMALL.input_size)
    for name in ("forward_dense", "forward_with_geometry"):
        with pytest.raises(ValueError, match="finite"):
            getattr(model, name)(images, (19, 23), photo_size=(100, 200))
    with torch.inference_mode():
        _, _, _, finite = model._forward_with_geometry_validated(
            images, (19, 23), photo_size=(100, 200))
    assert finite.dtype == torch.bool and finite.shape == () and not finite.item()


@pytest.mark.parametrize("need_coarse", [False, True])
def test_oriented_validated_geometry_matches_public_without_host_scalar_checks(
        need_coarse, monkeypatch):
    model = _oriented_model(np.linspace(-0.2, 0.2, 48), quarter_turn=3)
    images = torch.full((2, 3, *SMALL.input_size), 0.5)
    with torch.inference_mode():
        expected = model.forward_with_geometry(images, (19, 23), photo_size=(123, 217))
    calls = []
    handles = [model.encoder[index].register_forward_hook(
        lambda module, args, output, index=index: calls.append(index)) for index in (0, 2, 4)]

    def forbidden(*args, **kwargs):
        pytest.fail("validated oriented decoder extracted a tensor scalar to the host")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", forbidden)
            patch.setattr(torch.Tensor, "__bool__", forbidden)
            with torch.inference_mode():
                coarse, dense, auxiliary, finite = model._forward_with_geometry_validated(
                    images, (19, 23), photo_size=(123, 217), need_coarse=need_coarse)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [0, 2, 4]
    assert finite.item() and finite.device == images.device and finite.dtype == torch.bool
    assert torch.equal(dense, expected[1])
    assert (coarse is None) == (not need_coarse)
    if need_coarse:
        assert torch.equal(coarse, expected[0])
    assert all(torch.equal(value, expected[2][key]) for key, value in auxiliary.items())


@pytest.mark.parametrize("invalid", ["dtype", "shape", "nonfinite", "range", "sparse"])
def test_oriented_public_keeps_strict_image_admission(invalid):
    images = torch.zeros(1, 3, *SMALL.input_size)
    if invalid == "dtype":
        images = images.double()
    elif invalid == "shape":
        images = images[:, :, :-1]
    elif invalid == "nonfinite":
        images[0, 0, 0, 0] = float("nan")
    elif invalid == "range":
        images[0, 0, 0, 0] = 1.1
    else:
        images = images.to_sparse()
    with pytest.raises((TypeError, ValueError)):
        _oriented_model().forward_with_geometry(images, (19, 23), photo_size=(100, 200))


@pytest.mark.parametrize("quarter_turn", [0, 1, 2, 3])
@pytest.mark.parametrize("photo_size", [(100, 200), (200, 100)])
def test_teacher_geometry_matches_independent_physical_map_and_semantic_corners(
        quarter_turn, photo_size):
    raw = np.linspace(-0.15, 0.15, 48, dtype=np.float32)
    wrong_turn = (quarter_turn + 2) % 4
    model = _oriented_model(raw, quarter_turn=wrong_turn).train()
    images = torch.full((1, 3, *SMALL.input_size), 0.5)
    with torch.inference_mode():
        coarse, dense, auxiliary, finite = model._forward_with_geometry_validated(
            images, (19, 23), photo_size=photo_size, need_coarse=True,
            teacher_direction=torch.tensor([quarter_turn]), teacher_valid=torch.tensor([True]))
    assert finite.item() and auxiliary["orientation_logits"].argmax(1).item() == wrong_turn
    expected = _oriented_reference(raw, _centers((19, 23)), photo_size, quarter_turn)
    expected_coarse = _oriented_reference(raw, _centers(SMALL.grid_size), photo_size, quarter_turn)
    expected_corners = _oriented_reference(
        raw, [[-1, -1], [1, -1], [1, 1], [-1, 1]], photo_size, quarter_turn)
    torch.testing.assert_close(dense[0], torch.from_numpy(expected), rtol=2e-6, atol=3e-6)
    torch.testing.assert_close(coarse[0].permute(1, 2, 0),
                               torch.from_numpy(expected_coarse) - coarse_identity(SMALL.grid_size),
                               rtol=2e-6, atol=3e-6)
    torch.testing.assert_close(auxiliary["corners"][0], torch.from_numpy(expected_corners),
                               rtol=2e-6, atol=3e-6)


def test_teacher_mixed_validity_preserves_argmax_for_invalid_gt_and_all_public_outputs():
    model = _oriented_model(quarter_turn=3).train()
    images = torch.full((4, 3, *SMALL.input_size), 0.5)
    # Invalid GT labels may use arbitrary sentinels; only valid rows select GT.
    target, valid = torch.tensor([0, -1, 2, 999]), torch.tensor([True, False, True, False])
    state = {key: value.clone() for key, value in model.state_dict().items()}
    before_images, before_target, before_valid = images.clone(), target.clone(), valid.clone()
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        before = model.forward_with_geometry(images, (19, 23), photo_size=(100, 200))
        residual, dense, auxiliary, finite = model._forward_with_geometry_validated(
            images, (19, 23), photo_size=(100, 200), need_coarse=True,
            teacher_direction=target, teacher_valid=valid)
        after = model.forward_with_geometry(images, (19, 23), photo_size=(100, 200))
        model.eval()
        evaluation = model.forward_with_geometry(images, (19, 23), photo_size=(100, 200))
    assert finite.item() and residual.shape == (4, 2, *SMALL.grid_size)
    for index, turn in enumerate((0, 3, 2, 3)):
        expected = _oriented_reference(np.zeros(48), _centers((19, 23)), (100, 200), turn)
        torch.testing.assert_close(dense[index], torch.from_numpy(expected), rtol=0, atol=1e-6)
    for actual in (after, evaluation):
        assert torch.equal(actual[0], before[0]) and torch.equal(actual[1], before[1])
        assert all(torch.equal(value, before[2][key]) for key, value in actual[2].items())
    assert torch.equal(auxiliary["orientation_logits"], before[2]["orientation_logits"])
    assert torch.equal(auxiliary["page_logits"], before[2]["page_logits"])
    assert torch.equal(images, before_images) and torch.equal(target, before_target)
    assert torch.equal(valid, before_valid) and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize("target,valid", [
    (None, torch.tensor([True])), (torch.tensor([0]), None),
    ([0], torch.tensor([True])), (torch.tensor([0]), [True]),
    (torch.tensor([0.0]), torch.tensor([True])),
    (torch.tensor([0], dtype=torch.int32), torch.tensor([True])),
    (torch.tensor([False]), torch.tensor([True])),
    (torch.tensor(0), torch.tensor([True])),
    (torch.tensor([[0]]), torch.tensor([True])),
    (torch.tensor([0, 1]), torch.tensor([True, True])),
    (torch.tensor([0]), torch.tensor([1], dtype=torch.uint8)),
    (torch.tensor([0]), torch.tensor([1])),
    (torch.tensor([0]), torch.tensor([1.0])),
    (torch.tensor([0]), torch.tensor(True)),
    (torch.tensor([0]), torch.tensor([[True]])),
    (torch.tensor([0]), torch.tensor([True, False])),
    (torch.tensor([0], device="meta"), torch.tensor([True])),
    (torch.tensor([0]), torch.tensor([True], device="meta")),
    (torch.tensor([0]).to_sparse(), torch.tensor([True])),
    (torch.tensor([0]), torch.tensor([True]).to_sparse()),
])
def test_teacher_rejects_metadata_before_feature_allocation(target, valid, monkeypatch):
    model = _oriented_model().train()
    monkeypatch.setattr(model, "_geometry_features",
                        lambda *args: pytest.fail("invalid teacher metadata reached the encoder"))
    with pytest.raises((TypeError, ValueError)):
        model._forward_with_geometry_validated(
            torch.zeros(1, 3, *SMALL.input_size), (19, 23), photo_size=(100, 200),
            teacher_direction=target, teacher_valid=valid)


@pytest.mark.parametrize("valid", [False, True])
def test_teacher_override_is_forbidden_in_eval_even_when_every_gt_is_invalid(valid, monkeypatch):
    model = _oriented_model().eval()
    monkeypatch.setattr(model, "_geometry_features",
                        lambda *args: pytest.fail("eval teacher override reached the encoder"))
    with pytest.raises((TypeError, ValueError)):
        model._forward_with_geometry_validated(
            torch.zeros(1, 3, *SMALL.input_size), (19, 23), photo_size=(100, 200),
            teacher_direction=torch.tensor([0]), teacher_valid=torch.tensor([valid]))


@pytest.mark.parametrize("method", ["forward", "forward_dense", "forward_with_geometry",
                                     "_forward_dense_validated"])
def test_teacher_inputs_are_not_exposed_through_inference_apis(method):
    model = _oriented_model().train()
    arguments = [torch.zeros(1, 3, *SMALL.input_size)]
    if method != "forward":
        arguments.append((19, 23))
    with pytest.raises(TypeError):
        getattr(model, method)(*arguments, photo_size=(100, 200),
                               teacher_direction=torch.tensor([0]),
                               teacher_valid=torch.tensor([True]))


@pytest.mark.parametrize("target,valid,flag", [
    ([0, 3], [True, True], True),
    ([-1, 4], [False, False], True),
    ([-1, 2], [True, True], False),
    ([1, 4], [True, True], False),
])
def test_teacher_range_uses_safe_fallback_and_device_flag_without_host_scalar_checks(
        target, valid, flag, monkeypatch):
    model = _oriented_model(quarter_turn=2).train()
    images = torch.full((2, 3, *SMALL.input_size), 0.5)
    # Strided noncontiguous N-vectors are accepted without a layout conversion.
    target = torch.tensor([[value, 17] for value in target])[:, 0]
    valid = torch.tensor([[value, False] for value in valid])[:, 0]
    assert not target.is_contiguous() and not valid.is_contiguous()

    def forbidden(*args, **kwargs):
        pytest.fail("teacher hot path extracted a device scalar")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        with torch.inference_mode():
            coarse, dense, auxiliary, finite = model._forward_with_geometry_validated(
                images, (19, 23), photo_size=(100, 200),
                teacher_direction=target, teacher_valid=valid)
    assert coarse is None
    assert finite.dtype == torch.bool and finite.device == images.device and finite.shape == ()
    assert finite.item() is flag and torch.isfinite(dense).all()
    assert torch.equal(auxiliary["orientation_logits"].argmax(1), torch.tensor([2, 2]))
    for index in range(2):
        eligible = valid[index].item() and 0 <= target[index].item() < 4
        chosen = target[index].item() if eligible else 2
        expected = _oriented_reference(np.zeros(48), _centers((19, 23)), (100, 200), chosen)
        torch.testing.assert_close(dense[index], torch.from_numpy(expected), rtol=0, atol=1e-6)


@pytest.mark.parametrize("component", ["geometry", "orientation", "page"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_teacher_does_not_hide_nonfinite_raw_or_auxiliary_predictions(component, nonfinite):
    model = _oriented_model(quarter_turn=2).train()
    layer = {"geometry": model.head[3], "orientation": model.orientation_head,
             "page": model.page_head[2]}[component]
    with torch.no_grad():
        layer.bias[0] = nonfinite
    with torch.inference_mode():
        _, _, _, finite = model._forward_with_geometry_validated(
            torch.zeros(1, 3, *SMALL.input_size), (19, 23), photo_size=(100, 200),
            teacher_direction=torch.tensor([0]), teacher_valid=torch.tensor([True]))
    assert finite.shape == () and finite.dtype == torch.bool and not finite.item()


@pytest.mark.training
def test_teacher_removes_wrong_argmax_coordinate_shrink_gradient_and_keeps_direction_ce():
    """A 180-degree wrong branch shrinks both scales despite positive Jacobians."""
    size, photo_size = (11, 9), (100, 200)
    target = coarse_identity(size)[None]
    physical = torch.tensor([photo_size[1] / 2, photo_size[0] / 2])
    gradients = []
    for teacher in (False, True):
        model = _oriented_model(quarter_turn=2).train()
        kwargs = ({"teacher_direction": torch.tensor([0]), "teacher_valid": torch.tensor([True])}
                  if teacher else {})
        _, dense, auxiliary, finite = model._forward_with_geometry_validated(
            torch.full((1, 3, *SMALL.input_size), 0.5), size, photo_size=photo_size, **kwargs)
        assert finite.item()
        loss = ((dense - target) * physical).square().mean()
        loss.backward(retain_graph=True)
        assert model.orientation_head.bias.grad is None  # Argmax has no coordinate gradient.
        gradient = model.head[3].bias.grad.clone()
        assert torch.isfinite(gradient).all()
        gradients.append(gradient)
        if teacher:
            assert loss.item() == 0 and torch.equal(dense, target)
            assert not gradient.count_nonzero()
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(
            auxiliary["orientation_logits"], torch.tensor([0])).backward()
        assert model.orientation_head.bias.grad[0] < 0 < model.orientation_head.bias.grad[2]
        assert model.head[3].bias.grad is None
    expected = 6 * target.square().mean(dim=(0, 1, 2)) * physical.square()
    torch.testing.assert_close(gradients[0][3:5], expected, rtol=2e-6, atol=1e-3)
    assert (gradients[0][3:5] > 0).all() and not gradients[1][3:5].count_nonzero()


@pytest.mark.training
@pytest.mark.parametrize("zero_head", [True, False])
def test_oriented_mixed_geometry_page_and_direction_loss_has_finite_gradients(zero_head):
    raw = None if zero_head else np.linspace(-0.2, 0.2, 48)
    model = _oriented_model(raw).train()
    images = torch.full((2, 3, *SMALL.input_size), 0.5)
    with torch.inference_mode():
        model.forward_with_geometry(images, (11, 9), photo_size=(100, 200))
    _, dense, auxiliary = model.forward_with_geometry(images, (11, 9), photo_size=(100, 200))
    target = coarse_identity((11, 9))[None] + torch.tensor([0.1, -0.07])
    loss = torch.nn.functional.smooth_l1_loss(dense, target.expand_as(dense))
    loss += torch.nn.functional.cross_entropy(auxiliary["orientation_logits"], torch.tensor([1, 3]))
    loss += torch.nn.functional.binary_cross_entropy_with_logits(
        auxiliary["page_logits"], torch.ones_like(auxiliary["page_logits"]))
    corner_target = torch.tensor([[-0.8, -0.9], [0.9, -0.9], [0.9, 0.8], [-0.8, 0.8]])
    loss += torch.nn.functional.smooth_l1_loss(
        auxiliary["corners"], corner_target.expand_as(auxiliary["corners"]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.orientation_head.bias.grad.abs().sum() > 0
    assert model.page_head[2].bias.grad.abs().sum() > 0
    assert model.head[3].bias.grad[:8].abs().sum() > 0
    assert model.head[3].bias.grad[8:].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_oriented_cuda_forward_matches_cpu_for_physical_aspect_and_semantic_corners():
    cpu = _oriented_model(np.linspace(-0.2, 0.2, 48), quarter_turn=1)
    gpu = _oriented_model().cuda()
    gpu.load_state_dict(cpu.state_dict())
    images = torch.full((2, 3, *SMALL.input_size), 0.5)
    with torch.inference_mode():
        expected = cpu.forward_with_geometry(images, (19, 23), photo_size=(123, 217))
        actual = gpu.forward_with_geometry(images.cuda(), (19, 23), photo_size=(123, 217))
    for a, b in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(a.cpu(), b, rtol=2e-5, atol=2e-5)
    for name, value in actual[2].items():
        assert value.device.type == "cuda"
        torch.testing.assert_close(value.cpu(), expected[2][name], rtol=2e-5, atol=2e-5)
