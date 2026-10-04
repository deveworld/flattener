import numpy as np
import pytest

from flattener.scan import sharpen as sh


def step_image(dark=(60, 60, 60), light=(200, 200, 200), size=(40, 40)):
    image = np.empty((*size, 3), dtype=np.uint8)
    image[:, : size[1] // 2] = dark
    image[:, size[1] // 2 :] = light
    return image


def test_amount_zero_and_flat_pages_are_left_as_they_are():
    image = step_image()
    out = sh.sharpen(image, 0)
    assert out is not image and np.array_equal(out, image)
    flat = np.full((30, 30, 3), 180, dtype=np.uint8)
    assert np.array_equal(sh.sharpen(flat, 2.0), flat)


def test_an_edge_gets_more_contrast_and_grey_stays_grey():
    image = step_image()
    out = sh.sharpen(image, 1.0)
    row = out[20, :, 0].astype(int)
    assert row[19] < 60 and row[20] > 200  # dark side darker, light side lighter at the edge
    assert row[0] == 60 and row[-1] == 200  # far from the edge nothing changes
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_colour_channels_move_together_and_noise_below_the_threshold_stays():
    image = step_image(dark=(120, 40, 30), light=(250, 240, 230))
    out = sh.sharpen(image, 1.0).astype(int)
    change = out - image.astype(int)
    unclipped = (out > 0).all(axis=-1) & (out < 255).all(axis=-1)
    assert (change[unclipped][:, 0] == change[unclipped][:, 1]).all()
    assert (change[unclipped][:, 1] == change[unclipped][:, 2]).all()
    rng = np.random.default_rng(0)
    noisy = (180 + rng.integers(-1, 2, size=(30, 30, 1))).repeat(3, axis=-1).astype(np.uint8)
    assert np.array_equal(sh.sharpen(noisy, 1.0), noisy)


def test_pixels_outside_the_valid_mask_keep_their_fill():
    image = step_image()
    valid = np.ones(image.shape[:2], dtype=bool)
    valid[:, 22:] = False
    image[~valid] = (255, 0, 255)
    out = sh.sharpen(image, 2.0, valid)
    assert (out[~valid] == (255, 0, 255)).all()
    assert not np.array_equal(out[valid], image[valid])


@pytest.mark.parametrize("amount", [-0.1, 4.5, True, "1", None, float("nan")])
def test_amounts_outside_zero_to_four_are_refused(amount):
    with pytest.raises(ValueError):
        sh.check_amount(amount)
