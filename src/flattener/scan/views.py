"""Model inputs of placed views, and one dewarp pass placed in the source.

A pass predicts a grid on the view of a ``Placement`` and places it in the source,
``G = compose(view_values(raw, convention, placement), placement)``. The model input is
made so that the passes of ``dewarp.predict`` come out bit for bit:

* the whole image (``Placement.whole``): ``model_input`` of the image (``predict_grid``);
* a whole-pixel crop box inside the image (``crop_box``) at |angle| <= 45 degrees
  without a quarter turn: ``crop_pass_box`` (the crop resized, or turned on its canvas by
  ``canvas_input``);
* any other view (quarter turns, tight views, boxes past the image edge) samples the
  source once through ``Placement.view_map`` at the model input size (bilinear,
  ``align_corners=False``, border padding). Only the part of the source the view covers
  is converted to a tensor.

``view_source="pyramid"`` samples every view from a box-reduced level of the source
(``scan.io.SourceImage.level``) whose pixel is at most the view's input pixel, so large
photos are antialiased before the model sees them; flattener values are the same on
every level. ``"original"`` (default) keeps the rules above.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from ..contracts import FlattenerError, Status
from ..dewarp.predict import Dewarper, _check_image, crop_pass_box, model_input, predict_input
from .placement import Placement, compose, view_values

VIEW_SOURCES = ("original", "pyramid")
LEVEL_FACTORS = (1, 2, 4, 8)
Levels = Callable[[int], NDArray[np.uint8]]
Box = tuple[int, int, int, int]


def _error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_view", message)


def crop_placement(box: Box, angle_deg: float, source_size: tuple[int, int]) -> Placement:
    """``Placement.from_canvas`` of a whole-pixel crop box (checked)."""
    if not (isinstance(box, (tuple, list)) and len(box) == 4):
        raise _error(f"box must be (left, top, right, bottom), got {box!r}")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in box):
        raise _error(f"box edges must be whole pixels, got {box!r}")
    return Placement.from_canvas(tuple(int(v) for v in box), angle_deg, source_size)


def _crop_pass_box_applies(placement: Placement, box: Box | None) -> bool:
    if box is None or placement.kind != "canvas" or placement.quarter != 0:
        return False
    height, width = placement.source_size
    left, top, right, bottom = box
    if left < 0 or top < 0 or right > width or bottom > height:
        return False
    same = Placement.from_canvas(box, placement.theta_deg, placement.source_size)
    return (
        same.base_matrix.tobytes() == placement.base_matrix.tobytes()
        and same.offset.tobytes() == placement.offset.tobytes()
        and same.view_size == placement.view_size
    )


def source_footprint(placement: Placement, pad: int = 2) -> Box:
    """Whole-pixel box (left, top, right, bottom) of the source that the view reads,
    padded by ``pad`` pixels for the bilinear taps and clipped to the source; at least
    one pixel."""
    height, width = placement.source_size
    corners = compose(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64), placement)
    pixels = (corners + 1.0) / 2.0 * np.array([width, height])
    low = np.floor(pixels.min(axis=0)) - pad
    high = np.ceil(pixels.max(axis=0)) + pad
    left = int(np.clip(low[0], 0, width - 1))
    top = int(np.clip(low[1], 0, height - 1))
    right = int(np.clip(high[0], left + 1, width))
    bottom = int(np.clip(high[1], top + 1, height))
    return left, top, right, bottom


def on_source(placement: Placement, size: tuple[int, int]) -> Placement:
    """The same view on a source of another resolution (flattener values are shared)."""
    if tuple(size) == tuple(placement.source_size):
        return placement
    return Placement(
        placement.kind,
        (int(size[0]), int(size[1])),
        placement.view_size,
        placement.theta_deg,
        placement.base_matrix,
        placement.offset,
        placement.quarter,
        placement.edges,
    )


def level_factor(placement: Placement, input_size: tuple[int, int]) -> int:
    """The largest of ``LEVEL_FACTORS`` not above the view's source pixels per input pixel."""
    view_height, view_width = placement.view_size
    density = math.sqrt(view_height * view_width / (input_size[0] * input_size[1]))
    return max(factor for factor in LEVEL_FACTORS if factor == 1 or factor <= density)


def sample_view(
    image: NDArray[np.uint8],
    placement: Placement,
    size: tuple[int, int],
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """The view of ``placement`` sampled from ``image`` at ``size`` pixels, as a
    (1, 3, height, width) float tensor in [0, 1].

    ``image`` covers the whole source extent at any resolution (the source or a level of
    it). Only the footprint of the view is converted."""
    _check_image(image)
    height, width = image.shape[:2]
    placement = on_source(placement, (height, width))
    view = placement.view_map(size).astype(np.float64)
    left, top, right, bottom = source_footprint(placement)
    # Source values -> values of the footprint crop (same pixel-edge convention).
    pixels = (view + 1.0) / 2.0 * np.array([width, height])
    local = (pixels - np.array([left, top])) / np.array([right - left, bottom - top]) * 2.0 - 1.0
    crop = np.ascontiguousarray(image[top:bottom, left:right])
    tensor = torch.from_numpy(crop).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    grid = torch.from_numpy(local.astype(np.float32))[None].to(device)
    return F.grid_sample(tensor, grid, mode="bilinear", padding_mode="border", align_corners=False)


def view_input(
    dewarper: Dewarper,
    image: NDArray[np.uint8],
    placement: Placement,
    *,
    view_source: str = "original",
    levels: Levels | None = None,
) -> torch.Tensor:
    """The model input of ``placement``'s view of ``image`` (see the module docstring)."""
    if view_source not in VIEW_SOURCES:
        raise _error(f"view_source must be one of {VIEW_SOURCES}, got {view_source!r}")
    size = dewarper.config.input_size
    if view_source == "pyramid":
        factor = level_factor(placement, size)
        level = image if factor == 1 or levels is None else levels(factor)
        return sample_view(level, placement, size, device=dewarper.device)
    if placement.kind == "whole" and placement.quarter == 0:
        return model_input(dewarper, image)
    return sample_view(image, placement, size, device=dewarper.device)


def run_view(
    dewarper: Dewarper,
    image: NDArray[np.uint8],
    placement: Placement,
    *,
    crop_box: Box | None = None,
    view_source: str = "original",
    levels: Levels | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """One pass: the grid of flattener values of ``image`` predicted on ``placement``'s
    view, and its record (``placement``; ``input``: how the model input was made).

    ``crop_box`` names the whole-pixel box of a canvas placement
    (``crop_placement``); a box inside the image at |angle| <= 45 without a quarter turn
    goes through ``crop_pass_box``."""
    _check_image(image)
    if tuple(image.shape[:2]) != tuple(placement.source_size):
        raise _error("placement belongs to a source of another size")
    if view_source not in VIEW_SOURCES:
        raise _error(f"view_source must be one of {VIEW_SOURCES}, got {view_source!r}")
    convention = dewarper.config.grid_convention
    if view_source == "original" and _crop_pass_box_applies(placement, crop_box):
        grid, _ = crop_pass_box(dewarper, image, crop_box, placement.theta_deg)
        source = "crop_pass_box"
    else:
        tensor = view_input(dewarper, image, placement, view_source=view_source, levels=levels)
        raw = predict_input(dewarper, tensor)
        grid = compose(view_values(raw, convention, placement), placement)
        if placement.kind == "whole" and placement.quarter == 0 and view_source == "original":
            source = "model_input"
        else:
            source = f"view_map:{view_source}"
    record = {"placement": placement.record(), "input": source}
    if crop_box is not None:
        record["box"] = [int(v) for v in crop_box]
    return grid, record
