"""MATLAB emulation checked against independent scalar oracles on tiny images."""

import math

import numpy as np
import pytest
from PIL import Image

from flattener.benchmark import matlab


def _round_half_away(value: float) -> float:
    return math.copysign(math.floor(abs(value) + 0.5), value)


def _cubic(x: float) -> float:
    x = abs(x)
    if x <= 1:
        return 1.5 * x**3 - 2.5 * x**2 + 1.0
    if x <= 2:
        return -0.5 * x**3 + 2.5 * x**2 - 4.0 * x + 2.0
    return 0.0


def _mirror(index: int, length: int) -> int:
    """MATLAB imresize 'symmetric' boundary for a 1-based index, returned 0-based."""
    position = (index - 1) % (2 * length)
    return position if position < length else 2 * length - 1 - position


def _resize_line(values: list[float], output_length: int, scale: float) -> list[float]:
    shrink = scale < 1.0
    width = 4.0 / scale if shrink else 4.0
    result = []
    for i in range(1, output_length + 1):
        u = i / scale + 0.5 * (1.0 - 1.0 / scale)
        left = math.floor(u - width / 2.0)
        taps = [left + p for p in range(math.ceil(width) + 2)]
        weights = [scale * _cubic(scale * (u - j)) if shrink else _cubic(u - j) for j in taps]
        total = sum(weights)
        result.append(
            sum(w / total * values[_mirror(j, len(values))] for w, j in zip(weights, taps))
        )
    return result


def _oracle_imresize(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Scalar MATLAB imresize: dimension with the smaller scale first, uint8 per pass."""
    work = image.astype(np.float64).tolist()
    scales = [(rows / image.shape[0], 0, rows), (cols / image.shape[1], 1, cols)]
    for scale, axis, length in sorted(scales, key=lambda item: item[0]):
        if axis == 0:
            columns = list(zip(*work))
            columns = [_resize_line(list(column), length, scale) for column in columns]
            work = [list(row) for row in zip(*columns)]
        else:
            work = [_resize_line(row, length, scale) for row in work]
        if image.dtype == np.uint8:
            work = [[min(255.0, max(0.0, _round_half_away(v))) for v in row] for row in work]
    return np.array(work, dtype=image.dtype)


def test_imread_rgb2gray_and_imgradient_follow_matlab():
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[..., :3] = 200
    pixels[0, 0, 3] = 255
    image = Image.fromarray(pixels, "RGBA")
    rgb = matlab.imread_rgb(image)  # alpha over black, or over the bKGD colour
    assert rgb[0, 0].tolist() == [200, 200, 200] and rgb[1, 1].tolist() == [0, 0, 0]
    image.info["background"] = (10, 20, 30)
    assert matlab.imread_rgb(image)[1, 1].tolist() == [10, 20, 30]

    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    red, green, blue = matlab.RGB2GRAY_COEFFICIENTS
    expected = np.array(
        [
            [_round_half_away(red * int(r) + green * int(g) + blue * int(b)) for r, g, b in row]
            for row in image
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(matlab.rgb2gray(image), expected)
    gray = rng.integers(0, 256, size=(3, 4), dtype=np.uint8)
    assert matlab.rgb2gray(gray) is gray

    step = np.zeros((5, 6), dtype=np.uint8)
    step[:, 3:] = 10
    gradient = matlab.imgradient(step)  # Sobel magnitude, replicate padding
    assert gradient[2, 2] == gradient[2, 3] == pytest.approx(40.0) and gradient[2, 0] == 0.0


def test_imresize_matches_scalar_oracle_and_rounds_the_size_up():
    for shape, output in (((9, 7), (4, 3)), ((6, 5), (13, 11))):
        rng = np.random.default_rng(sum(shape) + sum(output))
        image = rng.integers(0, 256, size=shape, dtype=np.uint8)
        got = matlab.imresize(image, output_shape=output)
        assert got.dtype == np.uint8
        assert np.array_equal(got, _oracle_imresize(image, *output))
        double = rng.random(shape)
        assert np.allclose(
            matlab.imresize(double, output_shape=output),
            _oracle_imresize(double, *output),
            rtol=0,
            atol=1e-12,
        )
    image = np.full((7, 9), 77, dtype=np.uint8)
    resized = matlab.imresize(image, scale=0.35)
    assert resized.shape == (math.ceil(7 * 0.35), math.ceil(9 * 0.35))
    assert np.all(resized == 77)
    rgb = np.stack([image, image // 2, image // 3], axis=2)
    assert matlab.imresize(rgb, scale=0.5).shape == (4, 5, 3)


def _oracle_reduce(image: np.ndarray) -> np.ndarray:
    kernel = [0.0625, 0.25, 0.375, 0.25, 0.0625]
    rows, cols = image.shape
    work = image.astype(np.float64)

    def symmetric(index: int, length: int) -> int:
        while index < 0 or index >= length:
            index = -index - 1 if index < 0 else 2 * length - index - 1
        return index

    for axis in (0, 1):
        out = np.zeros_like(work)
        for i in range(rows):
            for j in range(cols):
                total = 0.0
                for offset, weight in zip(range(-2, 3), kernel):
                    if axis == 0:
                        total += weight * work[symmetric(i + offset, rows), j]
                    else:
                        total += weight * work[i, symmetric(j + offset, cols)]
                out[i, j] = total
        if image.dtype == np.uint8:
            out = np.vectorize(lambda v: min(255.0, max(0.0, _round_half_away(v))))(out)
        work = out
    return work[::2, ::2].astype(image.dtype)


def _oracle_ssim(first: np.ndarray, second: np.ndarray) -> float:
    size, sigma = 11, 1.5
    coordinates = np.arange(size) - (size - 1) / 2
    window = np.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * sigma**2))
    window /= window.sum()
    x = first.astype(np.float64)
    y = second.astype(np.float64)

    def blur(values):
        padded = np.pad(values, 5, mode="edge")
        out = np.zeros_like(values)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                out[i, j] = np.sum(padded[i : i + size, j : j + size] * window)
        return out

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    ssim_map = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(ssim_map.mean())


def test_impyramid_ssim_and_ms_ssim_match_direct_oracles():
    rng = np.random.default_rng(3)
    for shape in ((9, 6), (4, 7)):
        image = rng.integers(0, 256, size=shape, dtype=np.uint8)
        reduced = matlab.impyramid_reduce(image)
        assert reduced.shape == (math.ceil(shape[0] / 2), math.ceil(shape[1] / 2))
        assert np.array_equal(reduced, _oracle_reduce(image))

    first = rng.integers(0, 256, size=(13, 15), dtype=np.uint8)
    noise = rng.integers(-40, 41, size=first.shape)
    second = np.clip(first.astype(int) + noise, 0, 255).astype(np.uint8)
    value = matlab.ssim(first, second)
    assert value == pytest.approx(_oracle_ssim(first, second), abs=1e-12)
    assert value == pytest.approx(matlab.ssim(second, first), abs=1e-14)
    assert matlab.ssim(first, first) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        matlab.ssim(first, second[:-1])

    first = rng.integers(0, 256, size=(40, 36), dtype=np.uint8)
    second = matlab.gaussian_filter(first, 5, 1.2)
    assert matlab.ms_ssim(first, first) == pytest.approx(sum(matlab.MS_SSIM_WEIGHTS), abs=1e-12)
    assert sum(matlab.MS_SSIM_WEIGHTS) == pytest.approx(1.0001, abs=1e-15)
    expected = 0.0
    a, b = first, second
    for level, weight in enumerate(matlab.MS_SSIM_WEIGHTS):
        expected += weight * _oracle_ssim(a, b)
        if level < 4:
            a, b = _oracle_reduce(a), _oracle_reduce(b)
    assert matlab.ms_ssim(first, second) == pytest.approx(expected, abs=1e-12)
