"""Input and local geometry checks for the CPU renderer.

The Jacobian check concerns individual bilinear cells. It cannot establish
global injectivity or the absence of occlusion.
"""

from math import isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contracts import FlattenerError, RenderConfig, Status

_CELL_CHUNK_SIZE = 65_536


def _integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _finite_real(value: Any) -> bool:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _input_error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def validate_config(config: RenderConfig | None) -> RenderConfig:
    """Resolve defaults and reject malformed settings without coercion."""
    if config is None:
        return RenderConfig()
    if not isinstance(config, RenderConfig):
        raise _input_error("invalid_config", "config must be a RenderConfig or None")
    if not isinstance(config.on_error, str) or config.on_error not in (
        "raise",
        "return",
        "original",
    ):
        raise _input_error("invalid_config", "on_error must be 'raise', 'return', or 'original'")
    for name in ("max_input_pixels", "max_output_pixels", "max_side"):
        value = getattr(config, name)
        if not _integer(value) or value <= 0:
            raise _input_error("invalid_config", f"{name} must be a positive integer")
    fraction = config.max_out_of_bounds_fraction
    if not _finite_real(fraction) or not 0 <= fraction <= 1:
        raise _input_error(
            "invalid_config", "max_out_of_bounds_fraction must be finite and in [0, 1]"
        )
    minimum = config.min_jacobian_ratio
    if not _finite_real(minimum) or minimum <= 0:
        raise _input_error("invalid_config", "min_jacobian_ratio must be finite and positive")
    if (
        not isinstance(config.fill_rgb, tuple)
        or len(config.fill_rgb) != 3
        or any(not _integer(value) or not 0 <= value <= 255 for value in config.fill_rgb)
    ):
        raise _input_error(
            "invalid_config", "fill_rgb must be a tuple of three integers in [0, 255]"
        )
    return config


def validate_size(size: Any, *, max_pixels: int, max_side: int, name: str) -> tuple[int, int]:
    """Check a (height, width) pair before allocating image-sized buffers."""
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(not _integer(value) or value <= 0 for value in size)
    ):
        raise _input_error("invalid_size", f"{name} must contain positive height and width")
    height, width = (int(value) for value in size)
    if height > max_side or width > max_side or height * width > max_pixels:
        raise _input_error(
            "size_limit_exceeded",
            f"{name} ({height}, {width}) exceeds the configured size limits",
        )
    return height, width


def validate_image(image: NDArray[np.uint8], config: RenderConfig) -> None:
    """Validate an already decoded RGB image without modifying it."""
    if not isinstance(image, np.ndarray):
        raise _input_error("invalid_type", "image must be a NumPy array")
    if image.dtype != np.uint8:
        raise _input_error("invalid_dtype", "image must have dtype uint8")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise _input_error("invalid_shape", "image must have shape (height, width, 3)")
    validate_size(
        image.shape[:2],
        max_pixels=config.max_input_pixels,
        max_side=config.max_side,
        name="image size",
    )


def validate_map(backward_map: NDArray[np.float32], config: RenderConfig) -> None:
    """Check dense map storage and size; geometry is checked separately."""
    if not isinstance(backward_map, np.ndarray):
        raise _input_error("invalid_type", "backward_map must be a NumPy array")
    if backward_map.dtype != np.float32:
        raise _input_error("invalid_dtype", "backward_map must have dtype float32")
    if backward_map.ndim != 3 or backward_map.shape[-1] != 2:
        raise _input_error("invalid_shape", "backward_map must have shape (height, width, 2)")
    validate_size(
        backward_map.shape[:2],
        max_pixels=config.max_output_pixels,
        max_side=config.max_side,
        name="output size",
    )


def validate_target_mask(
    target_mask: NDArray[np.bool_] | None, output_size: tuple[int, int]
) -> NDArray[np.bool_]:
    """Resolve the target mask to an independently owned boolean array."""
    if target_mask is None:
        return np.ones(output_size, dtype=np.bool_)
    if not isinstance(target_mask, np.ndarray):
        raise _input_error("invalid_type", "target_mask must be a NumPy array or None")
    if target_mask.dtype != np.bool_:
        raise _input_error("invalid_dtype", "target_mask must have dtype bool")
    if target_mask.shape != tuple(output_size):
        raise _input_error("invalid_shape", "target_mask must match the output size")
    return target_mask.copy()


def _check_cells(
    backward_map: NDArray[np.float32],
    target_mask: NDArray[np.bool_],
    minimum: float,
    stats: dict[str, Any],
) -> None:
    """Accumulate four-corner bilinear Jacobians with bounded float64 storage."""
    stats["fold_check"] = "not_applicable"
    height, width = target_mask.shape
    total_cells = (height - 1) * (width - 1)
    if not total_cells:
        return
    rows_per_chunk = max(1, _CELL_CHUNK_SIZE // (width - 1))
    scale = height * width / 4.0
    smallest: float | None = None
    for start in range(0, height - 1, rows_per_chunk):
        stop = min(start + rows_per_chunk, height - 1)
        complete = (
            target_mask[start:stop, :-1]
            & target_mask[start:stop, 1:]
            & target_mask[start + 1 : stop + 1, :-1]
            & target_mask[start + 1 : stop + 1, 1:]
        )
        count = int(np.count_nonzero(complete))
        stats["checked_cells"] += count
        stats["excluded_cells"] += complete.size - count
        if not count:
            continue

        # Select eligible corners before conversion/arithmetic. Coordinates
        # outside the target can be NaN or Inf and must not enter determinants.
        p00 = backward_map[start:stop, :-1][complete].astype(np.float64)
        p10 = backward_map[start:stop, 1:][complete].astype(np.float64)
        p01 = backward_map[start + 1 : stop + 1, :-1][complete].astype(np.float64)
        p11 = backward_map[start + 1 : stop + 1, 1:][complete].astype(np.float64)
        top, bottom = p10 - p00, p11 - p01
        left, right = p01 - p00, p11 - p10

        def determinant(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray:
            return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) * scale

        cell_minimum = determinant(top, left)
        np.minimum(cell_minimum, determinant(top, right), out=cell_minimum)
        np.minimum(cell_minimum, determinant(bottom, left), out=cell_minimum)
        np.minimum(cell_minimum, determinant(bottom, right), out=cell_minimum)
        chunk_minimum = float(np.min(cell_minimum))
        smallest = chunk_minimum if smallest is None else min(smallest, chunk_minimum)
        stats["failed_cells"] += int(np.count_nonzero(cell_minimum <= minimum))

    stats["min_jacobian_ratio"] = smallest
    if stats["checked_cells"]:
        stats["fold_check"] = "failed" if stats["failed_cells"] else "passed"


def validate_geometry(
    backward_map: NDArray[np.float32],
    target_mask: NDArray[np.bool_],
    config: RenderConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]:
    """Check active coordinates, source bounds, local orientation, then coverage.

    Call storage/configuration validation first. Every expected geometry failure
    carries the statistics obtained so far in ``metadata['validation']``.
    """
    finite = np.isfinite(backward_map[..., 0]) & np.isfinite(backward_map[..., 1])
    source_mask = (
        finite
        & (backward_map[..., 0] >= -1)
        & (backward_map[..., 0] <= 1)
        & (backward_map[..., 1] >= -1)
        & (backward_map[..., 1] <= 1)
    )
    valid_mask = source_mask & target_mask
    target_pixels = int(np.count_nonzero(target_mask))
    valid_pixels = int(np.count_nonzero(valid_mask))
    outside = target_pixels - valid_pixels
    stats: dict[str, Any] = {
        "target_pixels": target_pixels,
        "valid_pixels": valid_pixels,
        "out_of_bounds_pixels": outside,
        "out_of_bounds_fraction": outside / target_pixels if target_pixels else None,
        "fold_check": None,
        "checked_cells": 0,
        "excluded_cells": 0,
        "failed_cells": 0,
        "min_jacobian_ratio": None,
    }

    def reject(reason: str, message: str) -> None:
        raise FlattenerError(
            Status.INVALID_TRANSFORM, reason, message, metadata={"validation": stats}
        )

    if not np.all(finite, where=target_mask):
        reject("non_finite_grid", "backward_map contains non-finite target coordinates")
    if not target_pixels:
        reject("empty_target", "target_mask selects no output pixels")
    if stats["out_of_bounds_fraction"] > config.max_out_of_bounds_fraction:
        reject("excessive_out_of_bounds", "too many target coordinates lie outside the source")
    _check_cells(backward_map, target_mask, config.min_jacobian_ratio, stats)
    if stats["failed_cells"]:
        reject("fold_or_degenerate", "a target cell has a reversed or degenerate Jacobian")
    if not valid_pixels:
        reject("empty_valid_region", "no target coordinates lie inside the source")
    return source_mask, valid_mask, stats
