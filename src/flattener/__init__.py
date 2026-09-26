"""CPU document unwarping with explicit coordinate and failure contracts."""

from ._version import __version__
from .contracts import SPEC_VERSION, FlattenerError, RenderConfig, RenderResult, Status
from .coordinates import (
    identity_map,
    normalized_to_pixel,
    pixel_to_normalized,
    resize_backward_map,
)
from .image_io import load_image, render_file, save_result
from .renderer import render

__all__ = [
    "__version__",
    "SPEC_VERSION",
    "FlattenerError",
    "RenderConfig",
    "RenderResult",
    "Status",
    "identity_map",
    "load_image",
    "normalized_to_pixel",
    "pixel_to_normalized",
    "render",
    "render_file",
    "resize_backward_map",
    "save_result",
]
