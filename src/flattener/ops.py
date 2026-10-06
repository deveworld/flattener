"""NumPy image sampling for inference, following PyTorch's bilinear operations.

``grid_sample`` is ``torch.nn.functional.grid_sample`` with bilinear sampling and border
padding; ``resize`` is ``interpolate(mode="bilinear", align_corners=False)`` with or
without antialiasing; ``avg_pool`` is ``avg_pool2d`` with a square kernel and equal stride.
Arrays are channels-last (H, W, C) and values are float32, so results agree with PyTorch
to float32 rounding. Large outputs are processed in row blocks on ``threads()`` threads.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray

ROWS_PER_BLOCK = 128
_UINT8_SCALE = np.arange(256, dtype=np.float32) / np.float32(255)
_threads = max(1, min(8, os.cpu_count() or 1))


def threads() -> int:
    """Threads used for large samplings."""
    return _threads


def set_threads(count: int) -> None:
    """Use ``count`` threads for large samplings (at least 1)."""
    global _threads
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("threads must be a positive integer")
    _threads = count


def unit_float(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """uint8 values as float32 in [0, 1] (each value divided by 255, as ``x / 255``)."""
    return _UINT8_SCALE[image]


def _blocks(rows: int, work) -> None:
    starts = range(0, rows, ROWS_PER_BLOCK)
    if _threads == 1 or rows <= ROWS_PER_BLOCK:
        for start in starts:
            work(start, min(start + ROWS_PER_BLOCK, rows))
        return
    with ThreadPoolExecutor(_threads) as pool:
        list(pool.map(lambda start: work(start, min(start + ROWS_PER_BLOCK, rows)), starts))


def _packed(image: NDArray[np.uint8]) -> NDArray[np.uint32]:
    """RGB or RGBA uint8 pixels as one uint32 each (fast single-index gathers)."""
    height, width, channels = image.shape
    if channels == 4:
        return np.ascontiguousarray(image).reshape(-1).view(np.uint32)
    packed = np.zeros((height, width, 4), dtype=np.uint8)
    packed[..., :channels] = image
    return packed.reshape(-1).view(np.uint32)


def grid_sample(
    image: NDArray, grid: NDArray, *, align_corners: bool = False, scale_uint8: bool = True
) -> NDArray[np.float32]:
    """Bilinear samples of ``image`` (H, W, C) at ``grid`` (h, w, 2) normalized (x, y)
    positions with border padding, as float32 (h, w, C).

    A uint8 image reads its values divided by 255 (``scale_uint8``), like a tensor made
    with ``float().div(255)``; positions must be finite."""
    if image.ndim != 3 or grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError("image must be (H, W, C) and grid (h, w, 2)")
    height, width, channels = image.shape
    rows, cols = grid.shape[:2]
    packed = image.dtype == np.uint8 and channels <= 4
    if packed:
        table = _UINT8_SCALE if scale_uint8 else np.arange(256, dtype=np.float32)
        source = _packed(image)
    else:
        source = np.ascontiguousarray(image, dtype=np.float32).reshape(-1, channels)
    output = np.empty((rows, cols, channels), dtype=np.float32)
    one, half = np.float32(1), np.float32(0.5)
    # PyTorch's CPU kernel: (x + 1) * size / 2 - 0.5, or (x + 1) * (size - 1) / 2.
    if align_corners:
        scales = np.float32((width - 1) / 2), np.float32((height - 1) / 2)
        shift = np.float32(0)
    else:
        scales = np.float32(width / 2), np.float32(height / 2)
        shift = half

    def tap(index: NDArray[np.intp]) -> NDArray[np.float32]:
        if packed:
            return table[np.take(source, index).view(np.uint8).reshape(*index.shape, 4)]
        return source[index]

    def work(start: int, stop: int) -> None:
        block = np.asarray(grid[start:stop], dtype=np.float32)
        # One rounding of the product and shift, as the kernel's fused multiply-add.
        x = ((block[..., 0] + one) * np.float64(scales[0]) - shift).astype(np.float32)
        y = ((block[..., 1] + one) * np.float64(scales[1]) - shift).astype(np.float32)
        np.clip(x, np.float32(0), np.float32(width - 1), out=x)
        np.clip(y, np.float32(0), np.float32(height - 1), out=y)
        x0 = np.floor(x)
        y0 = np.floor(y)
        east = x - x0
        south = y - y0
        west = one - east
        north = one - south
        left = x0.astype(np.intp)
        top = y0.astype(np.intp)
        right = np.minimum(left + 1, width - 1)
        bottom = np.minimum(top + 1, height - 1)
        top *= width
        bottom *= width
        result = tap(top + left)
        result *= (north * west)[..., None]
        result += tap(top + right) * (north * east)[..., None]
        result += tap(bottom + left) * (south * west)[..., None]
        result += tap(bottom + right) * (south * east)[..., None]
        output[start:stop] = result[..., :channels]

    _blocks(rows, work)
    return output


def _linear_taps(size: int, count: int) -> tuple[NDArray, NDArray, NDArray[np.float32]]:
    """Source indices and weights of ``interpolate(bilinear, align_corners=False)`` along
    one axis from ``size`` to ``count`` samples."""
    scale = np.float32(size) / np.float32(count)
    centers = np.arange(count, dtype=np.float32) + np.float32(0.5)
    # One rounding of the product and shift, as PyTorch's fused multiply-add.
    source = (np.float64(scale) * centers - 0.5).astype(np.float32)
    source = np.maximum(source, np.float32(0))
    low = source.astype(np.intp)
    high = np.minimum(low + 1, size - 1)
    return low, high, (source - low.astype(np.float32)).astype(np.float32)


def _antialias_weights(size: int, count: int) -> list[tuple[int, NDArray[np.float32]]]:
    """Per output sample, the first source index and normalized triangle-filter weights
    of ``interpolate(bilinear, antialias=True)`` (PIL's resampling), rounded as PyTorch's
    float32 kernel."""
    f32 = np.float32
    scale = f32(f32(size) / f32(count))
    support = f32(scale) if scale >= 1.0 else f32(1.0)
    inverse = f32(f32(1.0) / scale) if scale >= 1.0 else f32(1.0)
    taps = []
    for index in range(count):
        center = f32(float(scale) * (index + 0.5))
        low = max(int(float(center - support) + 0.5), 0)
        high = min(int(float(center + support) + 0.5), size)
        offsets = (np.arange(low, high, dtype=np.float32) - center).astype(np.float64) + 0.5
        weights = np.maximum(f32(0), f32(1) - np.abs((offsets * float(inverse)).astype(f32)))
        total = f32(0)
        for weight in weights:
            total = f32(total + weight)
        if total != 0:
            weights = (weights / total).astype(f32)
        taps.append((low, weights.astype(f32)))
    return taps


def _antialias_axis(values: NDArray[np.float32], count: int, axis: int) -> NDArray[np.float32]:
    shape = list(values.shape)
    shape[axis] = count
    output = np.empty(shape, dtype=np.float32)
    for index, (low, weights) in enumerate(_antialias_weights(values.shape[axis], count)):
        if axis == 0:
            output[index] = np.tensordot(weights, values[low : low + len(weights)], axes=(0, 0))
        else:
            band = values[:, low : low + len(weights)]
            output[:, index] = np.tensordot(band, weights, axes=(1, 0))
    return output


def resize(
    image: NDArray, size: tuple[int, int], *, antialias: bool = False
) -> NDArray[np.float32]:
    """``image`` (H, W, C) resized to ``size`` (height, width) by bilinear interpolation
    with half-pixel centers, as float32 (uint8 values divided by 255).

    ``antialias`` widens the filter by the reduction factor when shrinking."""
    if image.ndim != 3:
        raise ValueError("image must be (H, W, C)")
    height, width = int(size[0]), int(size[1])
    if antialias:
        if image.dtype == np.uint8:
            values = image.astype(np.float32)
            values /= np.float32(255)
        else:
            values = np.asarray(image, dtype=np.float32)
        # Along the rows first, then the columns, as PyTorch.
        if values.shape[1] != width:
            values = _antialias_axis(values, width, 1)
        if values.shape[0] != height:
            values = _antialias_axis(values, height, 0)
        return np.ascontiguousarray(values, dtype=np.float32)
    top, bottom, south = _linear_taps(image.shape[0], height)
    left, right, east = _linear_taps(image.shape[1], width)
    one = np.float32(1)
    output = np.empty((height, width, image.shape[2]), dtype=np.float32)

    def values(rows: NDArray, cols: NDArray) -> NDArray[np.float32]:
        taps = image[rows][:, cols]
        return unit_float(taps) if image.dtype == np.uint8 else taps.astype(np.float32)

    def work(start: int, stop: int) -> None:
        weight = east[None, :, None]
        rows = top[start:stop]
        upper = values(rows, left) * (one - weight) + values(rows, right) * weight
        rows = bottom[start:stop]
        lower = values(rows, left) * (one - weight) + values(rows, right) * weight
        weight = south[start:stop, None, None]
        output[start:stop] = upper * (one - weight) + lower * weight

    _blocks(height, work)
    return output


def avg_pool(image: NDArray[np.float32], factor: int) -> NDArray[np.float32]:
    """Means of ``factor`` x ``factor`` blocks of a 2-D image; trailing rows and columns
    that do not fill a block are dropped."""
    rows, cols = image.shape[0] // factor, image.shape[1] // factor
    blocks = image[: rows * factor, : cols * factor].reshape(rows, factor, cols, factor)
    return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
