"""What a flattened page looks like, measured on a small preview (``SIZE`` long side).

``paper(image)``: the median saturation and the share of dark pixels (luminance below
``DARK``). A page is a light, low-saturation sheet with a small share of ink; a laptop
keyboard is mostly dark, a wooden desk or a stone wall is saturated. The verify stage asks
for this evidence when the gate mask reaches past the grid (``page_region_larger``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SIZE = 384  # long side of the preview the measures run on
DARK = 0.35  # luminance (0-1) below which a pixel counts as dark


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
