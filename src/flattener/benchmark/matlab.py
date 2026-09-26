# Derived from OARDoc (https://github.com/GreatV/oardoc, commit
# 9c45c0e783748554a83abea482924a2443294bef, file oardoc/metrics/uvdoc.py),
# Copyright the OARDoc authors, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).
#
# Modified by the flattener authors: rewritten from PyTorch to NumPy float64,
# grayscale conversion moved before resizing (as in the MATLAB scripts), filters
# made separable and resize taps accumulated in MATLAB order, bKGD-aware PNG alpha
# compositing and explicit dtype validation added. See THIRD_PARTY.md.
"""NumPy emulation of the MATLAB Image Processing Toolbox calls used by DocUNet scripts.

Every function follows MATLAB semantics for ``uint8`` inputs: arithmetic is done in
float64 and results are rounded half away from zero and saturated to ``[0, 255]``
exactly where MATLAB returns ``uint8``. Float64 inputs keep full precision.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

RGB2GRAY_COEFFICIENTS = (0.298936021293776, 0.587043074451121, 0.114020904255103)
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
SSIM_GAUSSIAN_SIGMA = 1.5
SSIM_WINDOW = 11
_BICUBIC_WIDTH = 4.0
_PYRAMID_KERNEL = np.array((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0


def _require_image(image: np.ndarray, *, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(image).__name__}")
    if image.dtype not in (np.uint8, np.float64):
        raise TypeError(f"{name} must be uint8 or float64, got {image.dtype}")
    if image.ndim not in (2, 3) or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"{name} must be HW or HWC with positive size, got {image.shape}")


def matlab_round(values: np.ndarray) -> np.ndarray:
    """Round half away from zero like MATLAB ``round``."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return np.clip(matlab_round(values), 0, 255).astype(np.uint8)
    return values


def imread_rgb(image: Image.Image) -> np.ndarray:
    """Return pixels as one-output MATLAB ``imread`` does for an 8-bit PNG or JPEG.

    Alpha is composited over the PNG ``bKGD`` colour when present and over black
    otherwise; the alpha plane itself is discarded. Grayscale images stay ``HW``.
    """
    mode = image.mode
    if mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    elif mode == "1":
        image = image.convert("L")
    elif mode == "LA":
        image = image.convert("RGBA")
    if image.mode not in ("L", "RGB", "RGBA"):
        raise ValueError(f"unsupported image mode {mode!r}; expected 8-bit L, RGB or RGBA")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, _background_colour(image))
        background.paste(image, (0, 0), image.getchannel("A"))
        pixels = np.asarray(background, dtype=np.uint8)
        if mode == "LA":
            return np.ascontiguousarray(pixels[:, :, 0])
        return pixels.copy()
    return np.asarray(image, dtype=np.uint8).copy()


def _background_colour(image: Image.Image) -> tuple[int, int, int]:
    background = image.info.get("background")
    if isinstance(background, tuple) and len(background) >= 3:
        if all(isinstance(value, int) and 0 <= value <= 255 for value in background[:3]):
            return (background[0], background[1], background[2])
    if isinstance(background, int) and 0 <= background <= 255:
        return (background, background, background)
    return (0, 0, 0)


def rgb2gray(image: np.ndarray) -> np.ndarray:
    """MATLAB ``rgb2gray``; a two-dimensional image is returned unchanged."""
    _require_image(image)
    if image.ndim == 2:
        return image
    if image.shape[2] != 3:
        raise ValueError(f"rgb2gray expects three channels, got {image.shape}")
    work = image.astype(np.float64)
    red, green, blue = RGB2GRAY_COEFFICIENTS
    gray = work[:, :, 0] * red + work[:, :, 1] * green + work[:, :, 2] * blue
    return _cast_like(gray, image.dtype)


def _cubic(distance: np.ndarray) -> np.ndarray:
    absolute = np.abs(distance)
    absolute2 = absolute * absolute
    absolute3 = absolute2 * absolute
    near = (1.5 * absolute3 - 2.5 * absolute2 + 1.0) * (absolute <= 1)
    far = (-0.5 * absolute3 + 2.5 * absolute2 - 4.0 * absolute + 2.0) * (
        (absolute > 1) & (absolute <= 2)
    )
    return near + far


def resize_contributions(
    input_length: int, output_length: int, scale: float, *, antialias: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB ``imresize`` bicubic weights and zero-based source indices."""
    if input_length < 1 or output_length < 1 or not scale > 0:
        raise ValueError("resize lengths and scale must be positive")
    shrinking = scale < 1.0 and antialias
    kernel_width = _BICUBIC_WIDTH / scale if shrinking else _BICUBIC_WIDTH
    output = np.arange(1, output_length + 1, dtype=np.float64)
    source = output / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(source - kernel_width / 2.0)
    taps = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(taps, dtype=np.float64)[None, :]
    distance = source[:, None] - indices
    weights = scale * _cubic(scale * distance) if shrinking else _cubic(distance)
    weights = weights / weights.sum(axis=1, keepdims=True)
    keep = np.any(weights != 0, axis=0)
    weights = weights[:, keep]
    mirror = np.concatenate((np.arange(input_length), np.arange(input_length)[::-1]))
    positions = np.mod(indices[:, keep].astype(np.int64) - 1, 2 * input_length)
    return weights, mirror[positions]


def _resize_axis(work: np.ndarray, axis: int, weights: np.ndarray, indices: np.ndarray):
    shape = [1] * work.ndim
    shape[axis] = weights.shape[0]
    result = None
    for tap in range(weights.shape[1]):
        term = np.take(work, indices[:, tap], axis=axis) * weights[:, tap].reshape(shape)
        result = term if result is None else result + term
    return result


def imresize(
    image: np.ndarray,
    *,
    scale: float | None = None,
    output_shape: tuple[int, int] | None = None,
    antialias: bool = True,
) -> np.ndarray:
    """MATLAB ``imresize(image, scale|[rows cols], 'bicubic')``.

    Dimensions are processed in increasing scale order and a ``uint8`` image is
    rounded after each one-dimensional pass, as MATLAB does.
    """
    _require_image(image)
    if (scale is None) == (output_shape is None):
        raise ValueError("specify exactly one of scale or output_shape")
    height, width = image.shape[:2]
    if scale is not None:
        if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
            raise ValueError("scale must be a positive finite number")
        output_height = int(math.ceil(height * scale))
        output_width = int(math.ceil(width * scale))
        scales = (float(scale), float(scale))
    else:
        output_height, output_width = (int(output_shape[0]), int(output_shape[1]))
        if output_height < 1 or output_width < 1:
            raise ValueError("output_shape must be positive")
        scales = (output_height / height, output_width / width)
    passes = sorted(
        ((scales[0], 0, height, output_height), (scales[1], 1, width, output_width)),
        key=lambda item: item[0],
    )
    work = image.astype(np.float64)
    for dimension_scale, axis, input_length, output_length in passes:
        weights, indices = resize_contributions(
            input_length, output_length, dimension_scale, antialias=antialias
        )
        work = _resize_axis(work, axis, weights, indices)
        if image.dtype == np.uint8:
            work = np.clip(matlab_round(work), 0, 255)
    return work.astype(image.dtype)


def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """Normalised 1-D factor of MATLAB ``fspecial('gaussian', size, sigma)``."""
    if size < 1 or not sigma > 0:
        raise ValueError("Gaussian size and sigma must be positive")
    coordinate = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(coordinate * coordinate) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _filter_axis(work: np.ndarray, kernel: np.ndarray, axis: int, mode: str) -> np.ndarray:
    radius = kernel.size // 2
    padding = [(0, 0)] * work.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(work, padding, mode=mode)
    length = work.shape[axis]
    result = None
    for tap, weight in enumerate(kernel):
        window = [slice(None)] * work.ndim
        window[axis] = slice(tap, tap + length)
        term = padded[tuple(window)] * weight
        result = term if result is None else result + term
    return result


def separable_filter(
    image: np.ndarray,
    kernel_rows: np.ndarray,
    kernel_columns: np.ndarray,
    *,
    padding: str = "replicate",
) -> np.ndarray:
    """MATLAB ``imfilter`` with an outer-product kernel, rounded once for ``uint8``."""
    _require_image(image)
    mode = {"replicate": "edge", "symmetric": "symmetric"}.get(padding)
    if mode is None:
        raise ValueError(f"unsupported padding {padding!r}")
    work = _filter_axis(image.astype(np.float64), kernel_rows, 0, mode)
    work = _filter_axis(work, kernel_columns, 1, mode)
    return _cast_like(work, image.dtype)


def gaussian_filter(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """``imfilter(image, fspecial('gaussian', size, sigma), 'same', 'replicate')``."""
    kernel = gaussian_kernel_1d(size, sigma)
    return separable_filter(image, kernel, kernel, padding="replicate")


def impyramid_reduce(image: np.ndarray) -> np.ndarray:
    """MATLAB ``impyramid(image, 'reduce')``: 5-tap filter per axis, then keep odd samples."""
    _require_image(image)
    work = image
    for axis in (0, 1):
        filtered = _filter_axis(work.astype(np.float64), _PYRAMID_KERNEL, axis, "symmetric")
        work = _cast_like(filtered, image.dtype)
    return np.ascontiguousarray(work[::2, ::2])


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    """MATLAB ``ssim(first, second)`` defaults for two grayscale ``uint8`` images."""
    for name, image in (("first", first), ("second", second)):
        _require_image(image, name=name)
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(f"{name} must be a grayscale uint8 image")
    if first.shape != second.shape:
        raise ValueError(f"SSIM shapes differ: {first.shape} and {second.shape}")
    kernel = gaussian_kernel_1d(SSIM_WINDOW, SSIM_GAUSSIAN_SIGMA)

    def blur(values: np.ndarray) -> np.ndarray:
        return separable_filter(values, kernel, kernel, padding="replicate")

    x = first.astype(np.float64)
    y = second.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = blur(x)
    mu_y = blur(y)
    mu_xy = mu_x * mu_y
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    sigma_x2 = blur(x * x) - mu_x2
    sigma_y2 = blur(y * y) - mu_y2
    sigma_xy = blur(x * y) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerator / denominator))


def ms_ssim(first: np.ndarray, second: np.ndarray) -> float:
    """DocUNet ``evalUnwarp.m`` MS-SSIM: weighted sum of five ``ssim`` levels.

    Identical images score ``sum(MS_SSIM_WEIGHTS) == 1.0001``, as in MATLAB.
    """
    scores = []
    for level in range(len(MS_SSIM_WEIGHTS)):
        scores.append(ssim(first, second))
        if level < len(MS_SSIM_WEIGHTS) - 1:
            first = impyramid_reduce(first)
            second = impyramid_reduce(second)
    return float(sum(weight * score for weight, score in zip(MS_SSIM_WEIGHTS, scores, strict=True)))


def imgradient(image: np.ndarray) -> np.ndarray:
    """MATLAB ``imgradient(image)`` Sobel magnitude as float64, replicate padding."""
    _require_image(image)
    if image.ndim != 2:
        raise ValueError("imgradient expects a grayscale image")
    smooth = np.array((1.0, 2.0, 1.0))
    derivative = np.array((-1.0, 0.0, 1.0))
    work = image.astype(np.float64)
    gradient_x = separable_filter(work, smooth, derivative)
    gradient_y = separable_filter(work, derivative, smooth)
    return np.hypot(gradient_x, gradient_y)


def metric_scale(height: int, width: int, area: float) -> float:
    """Scale factor ``sqrt(area / (height * width))`` used by every DocUNet script."""
    if height < 1 or width < 1:
        raise ValueError("image size must be positive")
    return math.sqrt(area / (height * width))
