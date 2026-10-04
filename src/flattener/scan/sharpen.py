"""Output sharpening: a luminance unsharp mask on the rendered page.

The render samples the photo bilinearly at about one source pixel per output pixel, which
softens stroke edges a little more than the photo itself. ``sharpen`` adds ``amount``
times the luminance detail (Y minus its Gaussian blur of ``SIGMA`` pixels) to R, G and B
alike, so a colour keeps its hue and grey stays grey. Detail within ``THRESHOLD`` levels
(paper grain, JPEG noise) is left alone. Output pixels outside ``valid`` keep their fill.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

DEFAULT_AMOUNT = 1.0  # ScanOptions.sharpen
SIGMA = 1.5  # Gaussian standard deviation in output pixels
THRESHOLD = 2  # luminance detail at most this many levels is not added
MAX_AMOUNT = 4.0


def check_amount(amount: Any) -> float:
    """``amount`` as a float in [0, ``MAX_AMOUNT``]; 0 turns sharpening off."""
    if isinstance(amount, bool) or not isinstance(amount, (int, float, np.integer, np.floating)):
        raise ValueError(f"sharpen must be a number in [0, {MAX_AMOUNT:g}]")
    value = float(amount)
    if not 0.0 <= value <= MAX_AMOUNT:
        raise ValueError(f"sharpen must be a number in [0, {MAX_AMOUNT:g}]")
    return value


def sharpen(
    image: NDArray[np.uint8],
    amount: float,
    valid: NDArray[np.bool_] | None = None,
    *,
    sigma: float = SIGMA,
    threshold: int = THRESHOLD,
) -> NDArray[np.uint8]:
    """A sharpened copy of the RGB uint8 ``image`` (the module docstring)."""
    amount = check_amount(amount)
    if amount == 0.0:
        return image.copy()
    luma = Image.fromarray(image).convert("L")
    detail = np.asarray(luma, dtype=np.int16) - np.asarray(
        luma.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.int16
    )
    detail[np.abs(detail) <= threshold] = 0
    step = np.rint(detail * np.float32(amount)).astype(np.int16)
    output = np.empty_like(image)
    for channel in range(3):
        values = image[..., channel].astype(np.int16)
        values += step
        np.clip(values, 0, 255, out=values)
        output[..., channel] = values
    if valid is not None and not valid.all():
        output[~valid] = image[~valid]
    return output


def record(amount: float) -> dict[str, Any]:
    """The settings of one sharpening, for the scan record."""
    return {"amount": amount, "sigma": SIGMA, "threshold": THRESHOLD}
