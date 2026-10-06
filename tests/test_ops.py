"""NumPy sampling against the PyTorch operations it replaces at inference."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from flattener import ops

RNG = np.random.default_rng(7)
IMAGE = RNG.integers(0, 256, (61, 47, 3), dtype=np.uint8)


def _tensor(image):
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float()
    return tensor / 255 if image.dtype == np.uint8 else tensor


def _array(tensor):
    return tensor[0].permute(1, 2, 0).numpy()


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_grid_sample_matches_border_bilinear_sampling(align_corners, dtype):
    image = IMAGE if dtype == np.uint8 else RNG.normal(size=(23, 19, 2)).astype(np.float32)
    grid = RNG.uniform(-1.2, 1.2, (300, 140, 2)).astype(np.float32)  # some positions outside
    want = F.grid_sample(
        _tensor(image),
        torch.from_numpy(grid)[None],
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )
    got = ops.grid_sample(image, grid, align_corners=align_corners)
    np.testing.assert_allclose(got, _array(want), rtol=0, atol=1e-6)


def test_grid_sample_reads_raw_uint8_values_on_request():
    grid = RNG.uniform(-1, 1, (9, 8, 2)).astype(np.float32)
    raw = ops.grid_sample(IMAGE, grid, scale_uint8=False)
    np.testing.assert_allclose(raw / 255, ops.grid_sample(IMAGE, grid), atol=1e-6)


@pytest.mark.parametrize("size", [(712, 488), (40, 30), (7, 5), (61, 47)])
@pytest.mark.parametrize("antialias", [False, True])
def test_resize_matches_bilinear_interpolation(size, antialias):
    want = F.interpolate(
        _tensor(IMAGE), size=size, mode="bilinear", align_corners=False, antialias=antialias
    )
    got = ops.resize(IMAGE, size, antialias=antialias)
    np.testing.assert_allclose(got, _array(want), rtol=0, atol=1e-6)


def test_avg_pool_drops_partial_blocks():
    image = RNG.uniform(0, 255, (23, 18)).astype(np.float32)
    want = F.avg_pool2d(torch.from_numpy(image)[None, None], 4)[0, 0].numpy()
    np.testing.assert_allclose(ops.avg_pool(image, 4), want, rtol=0, atol=1e-4)


def test_threaded_blocks_give_the_same_samples(monkeypatch):
    grid = RNG.uniform(-1, 1, (700, 30, 2)).astype(np.float32)
    single = ops.grid_sample(IMAGE, grid)
    monkeypatch.setattr(ops, "_threads", 4)
    np.testing.assert_array_equal(ops.grid_sample(IMAGE, grid), single)
    with pytest.raises(ValueError):
        ops.set_threads(0)
