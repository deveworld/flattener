"""Pixel-center coordinates and residual expansion for backward maps."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from .contracts import FlattenerError, RenderConfig, Status
from .validation import validate_config, validate_size


def _points(points: NDArray[np.float32]) -> NDArray[np.float32]:
    if not isinstance(points, np.ndarray):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_type", "points must be a NumPy array")
    if points.dtype != np.float32:
        raise FlattenerError(Status.INPUT_ERROR, "invalid_dtype", "points must have dtype float32")
    if points.ndim < 1 or points.shape[-1] != 2:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_shape", "points must have shape (..., 2) in (x, y) order"
        )
    if not np.isfinite(points).all():
        raise FlattenerError(
            Status.INPUT_ERROR, "non_finite_coordinates", "points must contain only finite values"
        )
    return points


def _source_size(source_size: tuple[int, int]) -> tuple[int, int]:
    config = RenderConfig()
    return validate_size(
        source_size,
        max_pixels=config.max_input_pixels,
        max_side=config.max_side,
        name="source_size",
    )


def _float32_coordinates(points: NDArray[np.float64]) -> NDArray[np.float32]:
    if np.any(np.abs(points) > np.finfo(np.float32).max):
        raise FlattenerError(
            Status.INPUT_ERROR,
            "coordinate_overflow",
            "converted coordinates cannot be represented as finite float32 values",
        )
    return points.astype(np.float32)


def pixel_to_normalized(
    points: NDArray[np.float32], source_size: tuple[int, int]
) -> NDArray[np.float32]:
    """Convert finite ``(..., 2)`` (x, y) pixel coordinates to normalized values.

    ``source_size`` is (height, width), subject to the default input size limits.
    Pixel (0, 0) denotes the top-left pixel center; the image boundary starts
    at (-0.5, -0.5). Coordinates may lie outside the image. The input is never
    modified, and the returned array has dtype float32.
    """
    points = _points(points)
    height, width = _source_size(source_size)
    scale = np.array([width, height], dtype=np.float64)
    converted = 2.0 * (points.astype(np.float64) + 0.5) / scale - 1.0
    return _float32_coordinates(converted)


def normalized_to_pixel(
    points: NDArray[np.float32], source_size: tuple[int, int]
) -> NDArray[np.float32]:
    """Convert finite ``(..., 2)`` normalized coordinates to (x, y) pixels.

    ``source_size`` is (height, width), subject to the default input size limits.
    The convention is ``align_corners=False``, including singleton dimensions.
    Out-of-image coordinates are accepted if the result is finite float32.
    The input is never modified.
    """
    points = _points(points)
    height, width = _source_size(source_size)
    scale = np.array([width, height], dtype=np.float64)
    converted = (points.astype(np.float64) + 1.0) * scale / 2.0 - 0.5
    return _float32_coordinates(converted)


def identity_map(
    output_size: tuple[int, int], *, config: RenderConfig | None = None
) -> NDArray[np.float32]:
    """Return a fresh ``(height, width, 2)`` float32 pixel-center reference map.

    It is an identity when source and output sizes agree; otherwise it reads
    the full source with half-pixel resizing. Optional ``config`` controls the
    maximum side and output pixel count, checked before allocating the map.
    """
    config = validate_config(config)
    height, width = validate_size(
        output_size,
        max_pixels=config.max_output_pixels,
        max_side=config.max_side,
        name="output_size",
    )
    result = np.empty((height, width, 2), dtype=np.float32)
    result[..., 0] = 2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0
    result[..., 1] = (2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height - 1.0)[:, None]
    return result


def resize_backward_map(
    coarse_map: NDArray[np.float32],
    output_size: tuple[int, int],
    *,
    config: RenderConfig | None = None,
) -> NDArray[np.float32]:
    """Expand a finite coarse backward map by bilinearly resizing its residual.

    Subtract the coarse pixel-center reference, resize using
    ``align_corners=False, antialias=False`` with border extension, then add
    the output reference. This preserves identity and constant displacement;
    an arbitrary affine map is only approximate at the outermost boundary.
    Both coarse axes must be at least two, and neither output axis may shrink.
    A same-size request returns an exact independent copy. Optional ``config``
    applies its output resource limits to both coarse and dense map sizes.
    The helper always raises on invalid input, regardless of ``on_error``.
    """
    config = validate_config(config)
    if not isinstance(coarse_map, np.ndarray):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_type", "coarse_map must be a NumPy array")
    if coarse_map.dtype != np.float32:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_dtype", "coarse_map must have dtype float32"
        )
    if coarse_map.ndim != 3 or coarse_map.shape[-1] != 2:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_shape", "coarse_map must have shape (height, width, 2)"
        )
    coarse_size = validate_size(
        coarse_map.shape[:2],
        max_pixels=config.max_output_pixels,
        max_side=config.max_side,
        name="coarse_map",
    )
    dense_size = validate_size(
        output_size,
        max_pixels=config.max_output_pixels,
        max_side=config.max_side,
        name="output_size",
    )
    if min(coarse_size) < 2:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_shape", "both coarse_map axes must have length at least 2"
        )
    if any(dense < coarse for dense, coarse in zip(dense_size, coarse_size, strict=True)):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_size", "output_size may not shrink either coarse_map axis"
        )
    if not np.isfinite(coarse_map).all():
        raise FlattenerError(
            Status.INVALID_TRANSFORM,
            "non_finite_grid",
            "coarse_map must contain only finite coordinates",
        )
    if dense_size == coarse_size:
        return coarse_map.copy()

    residual = coarse_map - identity_map(coarse_size, config=config)
    tensor = torch.from_numpy(residual).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        expanded = F.interpolate(
            tensor, size=dense_size, mode="bilinear", align_corners=False, antialias=False
        )
    result = identity_map(dense_size, config=config)
    result += expanded.squeeze(0).permute(1, 2, 0).numpy()
    if not np.isfinite(result).all():
        raise FlattenerError(
            Status.INVALID_TRANSFORM,
            "non_finite_grid",
            "residual expansion produced non-finite coordinates",
        )
    return result
