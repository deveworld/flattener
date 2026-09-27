"""What a flattened page looks like, measured on a small preview (``SIZE`` long side).

``paper(image)``: the median saturation and the share of dark pixels (luminance below
``DARK``). A page is a light, low-saturation sheet with a small share of ink; a laptop
keyboard is mostly dark, a wooden desk or a stone wall is saturated. The verify stage asks
for this evidence when the gate mask reaches past the grid (``page_region_larger``).

``text(image)``: how many text lines fit across the page and how regular they are, and how
much of each edge has ink in its outer band. Ink is a pixel darker than the local paper
(the local maximum of the luminance over a window of 1/20 of the short side, box
smoothed) by more than ``INK`` of it. The line pitch is the strongest autocorrelation peak
of the ink profile across the lines (rows, or columns for a sideways page). A full page
shows 25 or more lines and blank margins; a close-up whose page runs out of the photo
shows a few large lines cut by two or more edges of the flattened page.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SIZE = 384  # long side of the preview the measures run on
DARK = 0.35  # luminance (0-1) below which a pixel counts as dark
INK = 0.3  # ink: darker than the local paper by this share of it
EDGE_BAND = 0.04  # outer band of each edge, share of the page size
EDGE_SEGMENTS = 20  # an edge is split into this many parts; a part with 2 % ink has ink


def luminance(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def paper(image: NDArray[np.uint8]) -> dict[str, float]:
    """``saturation``: median (max - min) / max over the pixels; ``dark``: share of pixels
    with luminance below ``DARK``."""
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    high, low = rgb.max(axis=-1), rgb.min(axis=-1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)
    return {
        "saturation": float(np.median(saturation)),
        "dark": float((luminance(image) < DARK).mean()),
    }


def _window_max(values: NDArray, size: int) -> NDArray:
    padded = np.pad(values, size // 2, mode="edge")
    rows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=0).max(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(rows, size, axis=1).max(axis=-1)


def _box_mean(values: NDArray, size: int) -> NDArray:
    half = size // 2
    padded = np.pad(values, ((half + 1, half), (half + 1, half)), mode="edge")
    total = padded.cumsum(axis=0).cumsum(axis=1)
    window = total[size:, size:] - total[:-size, size:] - total[size:, :-size]
    return (window + total[:-size, :-size]) / (size * size)


def ink(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """How much darker each pixel is than the local paper, as a share of the paper."""
    lum = luminance(image)
    size = max(5, round(min(lum.shape) / 20)) | 1
    paper_level = _box_mean(_window_max(lum, size), size)
    return np.clip(paper_level - lum, 0.0, 1.0) / np.maximum(paper_level, 0.05)


def _smooth(values: NDArray, size: int) -> NDArray:
    size = max(1, size)
    padded = np.pad(values, size, mode="edge")
    return np.convolve(padded, np.ones(size) / size, mode="same")[size:-size]


def _pitch(profile: NDArray) -> tuple[int | None, float]:
    """(lag, autocorrelation) of the strongest autocorrelation peak of a detrended
    profile, lags 3 to a third of its length."""
    longest = len(profile) // 3
    detrended = profile - _smooth(profile, longest)
    detrended = detrended - detrended.mean()
    if not np.any(detrended):
        return None, 0.0
    ac = np.correlate(detrended, detrended, mode="full")[len(detrended) - 1 :]
    ac = ac / ac[0]
    best, strength = None, 0.0
    for lag in range(3, min(longest, len(ac) - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > strength:
            best, strength = lag, float(ac[lag])
    return best, strength


def text(image: NDArray[np.uint8]) -> dict[str, object]:
    """``lines``: text lines across the page (page length over the line pitch, ``None``
    without a pitch); ``periodicity``: the autocorrelation at that pitch; ``edges``: per
    page edge (top, right, bottom, left) the share of its parts whose outer band holds ink."""
    marks = ink(image)
    height, width = marks.shape
    found = []
    rows = marks[:, int(0.2 * width) : int(0.8 * width)].mean(axis=1)
    columns = marks[int(0.2 * height) : int(0.8 * height), :].mean(axis=0)
    for profile in (rows, columns):
        lag, strength = _pitch(profile)
        found.append((strength, None if lag is None else len(profile) / lag))
    strength, lines = max(found, key=lambda item: item[0])
    inked = marks > INK
    band_h = max(2, round(EDGE_BAND * height))
    band_w = max(2, round(EDGE_BAND * width))
    bands = {
        "top": inked[:band_h, :],
        "right": inked[:, -band_w:].T,
        "bottom": inked[-band_h:, :],
        "left": inked[:, :band_w].T,
    }
    edges = {}
    for side, band in bands.items():
        parts = np.array_split(band, EDGE_SEGMENTS, axis=1)
        edges[side] = float(np.mean([part.mean() >= 0.02 for part in parts]))
    return {"lines": lines, "periodicity": strength, "edges": edges}
