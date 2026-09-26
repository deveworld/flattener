"""Public data contracts for the CPU renderer (specification 0.1)."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

SPEC_VERSION = "0.1"


class Status(StrEnum):
    SUCCESS = "success"
    INPUT_ERROR = "input_error"
    INVALID_TRANSFORM = "invalid_transform"
    MODEL_ERROR = "model_error"
    RENDER_ERROR = "render_error"
    IO_ERROR = "io_error"


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Resource limits and failure policy; values are validated on entry."""

    max_input_pixels: int = 16_000_000
    max_output_pixels: int = 16_000_000
    max_side: int = 8192
    max_out_of_bounds_fraction: float = 0.25
    min_jacobian_ratio: float = 1e-6
    fill_rgb: tuple[int, int, int] = (255, 255, 255)
    on_error: Literal["raise", "return", "original"] = "raise"


@dataclass(slots=True)
class RenderResult:
    status: Status
    reason_code: str | None = None
    message: str = ""
    image: NDArray[np.uint8] | None = None
    source_mask: NDArray[np.bool_] | None = None
    target_mask: NDArray[np.bool_] | None = None
    valid_mask: NDArray[np.bool_] | None = None
    used_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class FlattenerError(Exception):
    """An expected processing failure with a stable status and reason code."""

    def __init__(
        self,
        status: Status,
        reason_code: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason_code = reason_code
        self.message = message
        self.metadata = {} if metadata is None else metadata
