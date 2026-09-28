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

``fold(image)``: the fold of an open-book spread. A spread of two portrait pages is wider
than tall, and down its middle runs a column with no text between two pages of text. Text
shows as horizontal strokes (the luminance change down each column); the fold is the
middle of the stretch with the least of it in the middle band (``FOLD_BAND``), and it counts
when both sides carry at least ``FOLD_TEXT_RATIO`` times as much.

``paper_colour``, ``page_coloured`` and ``crop_edge_share`` work on a small copy of the
photo: the paper colour of a grid is the median colour of the ink-free pixels inside its
outline, and a pixel is page-coloured when every channel lies within ``TOLERANCE`` of it.
The verify stage uses them to tell a first crop that cut the page (blank paper runs along
the crop edge) from a gate mask that took some background.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SIZE = 384  # long side of the preview the measures run on
DARK = 0.35  # luminance (0-1) below which a pixel counts as dark
INK = 0.3  # ink: darker than the local paper by this share of it
EDGE_BAND = 0.04  # outer band of each edge, share of the page size
EDGE_SEGMENTS = 20  # an edge is split into this many parts; a part with 2 % ink has ink
TOLERANCE = 0.08  # page-coloured: every channel (0-1) within this of the paper colour
STRIP = 0.015  # strip inside a crop edge, share of the image's short side
FOLD_BAND = (0.35, 0.65)  # where the fold of a spread may lie, share of the width
FOLD_MIN_ASPECT = 1.0  # width / height of a spread of two portrait pages
FOLD_TEXT_RATIO = 2.5  # text on both sides against the fold column


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


def fold(image: NDArray[np.uint8]) -> dict[str, float] | None:
    """The fold of a flat spread preview: ``u`` (0-1 across the width), the aspect and the
    text ratio; ``None`` when the preview is not a spread."""
    lum = luminance(image)
    height, width = lum.shape
    aspect = width / height
    if aspect < FOLD_MIN_ASPECT or width < 40:
        return None
    strokes = _smooth(np.abs(np.diff(lum, axis=0)).mean(axis=0), max(3, width // 50))
    low, high = (round(share * width) for share in FOLD_BAND)
    band = strokes[low:high]
    lowest = int(np.argmin(band))
    # The middle of the stretch around the minimum that stays near it (a blank gutter).
    near = band <= band[lowest] + 0.25 * (float(np.median(strokes)) - band[lowest])
    start, stop = lowest, lowest
    while start > 0 and near[start - 1]:
        start -= 1
    while stop < len(band) - 1 and near[stop + 1]:
        stop += 1
    column = low + (start + stop) // 2
    pad, outer = round(0.05 * width), round(0.1 * width)
    left = strokes[outer : max(outer + 1, column - pad)].mean()
    right = strokes[column + pad : max(column + pad + 1, width - outer)].mean()
    ratio = float(min(left, right) / max(strokes[column], 1e-6))
    if ratio < FOLD_TEXT_RATIO:
        return None
    return {"u": (column + 0.5) / width, "aspect": aspect, "text_ratio": ratio}


def paper_colour(image: NDArray[np.uint8], inside: NDArray[np.bool_]) -> NDArray | None:
    """Median RGB (0-1) of the ink-free pixels of ``inside``; ``None`` for under 50."""
    blank = inside & (ink(image) < 0.1)
    if blank.sum() < 50:
        return None
    return np.median(np.asarray(image, dtype=np.float32)[blank] / 255.0, axis=0)


def page_coloured(image: NDArray[np.uint8], colour: NDArray) -> NDArray[np.bool_]:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return np.abs(rgb - colour).max(axis=-1) < TOLERANCE


def crop_edge_share(
    coloured: NDArray[np.bool_], box: tuple[float, float, float, float], size: tuple[int, int]
) -> float:
    """The largest share of page-coloured pixels in a strip just inside an edge of ``box``
    (source pixels of a photo of ``size``), over the edges that lie inside the photo; 0
    when every edge is a photo edge."""
    height, width = size
    rows, cols = coloured.shape
    left, top, right, bottom = box
    x0, x1 = round(left * cols / width), round(right * cols / width)
    y0, y1 = round(top * rows / height), round(bottom * rows / height)
    strip = max(2, round(STRIP * min(rows, cols)))
    strips = []
    if left > 1:
        strips.append(coloured[y0:y1, x0 : x0 + strip])
    if right < width - 1:
        strips.append(coloured[y0:y1, x1 - strip : x1])
    if top > 1:
        strips.append(coloured[y0 : y0 + strip, x0:x1])
    if bottom < height - 1:
        strips.append(coloured[y1 - strip : y1, x0:x1])
    return max((float(part.mean()) for part in strips if part.size), default=0.0)
