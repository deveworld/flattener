"""Independent geometry, image ownership, and gradient checks for the first CNN."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from flattener.coordinates import resize_backward_map
from flattener.learning.model import (
    CoordinatePredictor,
    ModelConfig,
    coarse_identity,
    dense_maps,
    preprocess_image,
)

pytestmark = pytest.mark.training


def test_config_is_frozen_and_preserves_explicit_architecture():
    config = ModelConfig()
    assert config.input_size == (96, 72)
    assert config.grid_size == (17, 13)
    assert config.channels == (16, 32, 48)
    assert config.hidden == 128
    with pytest.raises(FrozenInstanceError):
        config.hidden = 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": (0, 72)},
        {"input_size": (96, True)},
        {"input_size": [96, 72]},
        {"input_size": (96.0, 72)},
        {"grid_size": (1, 13)},
        {"grid_size": (17, -1)},
        {"channels": (16, 32)},
        {"channels": (16, 32, 0)},
        {"channels": (16, False, 48)},
        {"channels": [16, 32, 48]},
        {"hidden": False},
        {"hidden": 0},
        {"hidden": 128.0},
    ],
)
def test_invalid_configuration_is_rejected_before_parameter_allocation(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    "helper",
    [CoordinatePredictor, lambda c: preprocess_image(np.zeros((2, 3, 3), np.uint8), c)],
)
def test_configuration_requires_the_public_type(helper):
    with pytest.raises(TypeError, match="ModelConfig"):
        helper({"hidden": 128})


def test_coarse_identity_explicit_non_square_centers_and_singleton_axes():
    expected = torch.tensor(
        [[[-2 / 3, -0.5], [0, -0.5], [2 / 3, -0.5]],
         [[-2 / 3, 0.5], [0, 0.5], [2 / 3, 0.5]]]
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
    expected = np.stack((2 * (xx + 0.5) / 11 - 1 + 0.6,
                         2 * (yy + 0.5) / 7 - 1 - 0.4), axis=-1)
    np.testing.assert_allclose(result[0].numpy(), expected, rtol=0, atol=2e-7)
    assert result[1, ..., 0].max() > 1
    torch.testing.assert_close(residual, original, rtol=0, atol=0)


def test_dense_maps_matches_renderer_expansion_for_varying_residuals():
    residual = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 60 - 0.2
    coarse = coarse_identity((3, 4)) + residual[0].permute(1, 2, 0)
    expected = resize_backward_map(coarse.numpy(), (7, 11))
    np.testing.assert_allclose(dense_maps(residual, (7, 11))[0].numpy(), expected,
                               rtol=0, atol=2e-7)


def test_expansion_preserves_autograd_with_known_constant_gradient():
    residual = torch.zeros(1, 2, 2, 3, requires_grad=True)
    dense_maps(residual, (4, 6)).sum().backward()
    torch.testing.assert_close(residual.grad, torch.full_like(residual, 4.0), rtol=0, atol=1e-6)


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
        (np.zeros((2, 3, 3), np.float32), TypeError),
        (np.zeros((2, 3), np.uint8), ValueError),
        (np.zeros((2, 3, 4), np.uint8), ValueError),
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
        (torch.zeros(1, 3, 96, 72, dtype=torch.float64), TypeError),
        (torch.zeros(3, 96, 72), ValueError),
        (torch.zeros(0, 3, 96, 72), ValueError),
        (torch.zeros(1, 3, 72, 96), ValueError),
        (torch.full((1, 3, 96, 72), float("nan")), ValueError),
        (torch.full((1, 3, 96, 72), -0.01), ValueError),
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
        (torch.zeros(1, 2, 2, 3, dtype=torch.float64), (7, 11), TypeError),
        (torch.zeros(2, 2, 3), (7, 11), ValueError),
        (torch.zeros(0, 2, 2, 3), (7, 11), ValueError),
        (torch.zeros(1, 3, 2, 3), (7, 11), ValueError),
        (torch.zeros(1, 2, 1, 3), (7, 11), ValueError),
        (torch.zeros(1, 2, 2, 3), (1, 11), ValueError),
        (torch.zeros(1, 2, 2, 3), (7, True), ValueError),
        (torch.full((1, 2, 2, 3), float("inf")), (7, 11), ValueError),
    ],
)
def test_dense_maps_rejects_invalid_residuals_or_sizes(residual, size, error):
    with pytest.raises(error):
        dense_maps(residual, size)
