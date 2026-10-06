"""UVDocnet architecture settings and grid convention, without the network.

Kept free of PyTorch so ONNX Runtime inference can read model configs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ARCHITECTURE = "uvdocnet"
GRID_CONVENTIONS = ("flattener", "uvdoc")
UVDOC_REVISION = "4c9b82b537057aff2526e6dd118a847cdd072e82"


def _pair(value: Any, name: str, low: int, high: int) -> tuple[int, int]:
    if isinstance(value, list):
        value = tuple(value)
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high for v in value)
    ):
        raise ValueError(f"{name} must be a (height, width) pair of integers in [{low}, {high}]")
    return value


def natural_grid_size(input_size: tuple[int, int]) -> tuple[int, int]:
    """Backbone output size: four stride-2 'same' convolutions, i.e. ceil(n / 16) stepwise."""
    height, width = _pair(input_size, "input_size", 16, 8192)
    for _ in range(4):
        height, width = (height + 1) // 2, (width + 1) // 2
    return height, width


@dataclass(frozen=True, slots=True)
class DewarpModelConfig:
    """Architecture and output semantics. Sizes are (height, width).

    ``grid_convention`` names how the 2-D output is interpreted (see
    ``coords``): ``"flattener"`` for models trained here, ``"uvdoc"`` for the
    upstream MIT weights (align_corners=True values, upstream densification).
    """

    input_size: tuple[int, int] = (712, 488)
    grid_size: tuple[int, int] = (45, 31)
    num_filter: int = 32
    kernel_size: int = 5
    with_3d: bool = True
    grid_convention: str = "flattener"
    architecture: str = ARCHITECTURE

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_size", _pair(self.input_size, "input_size", 32, 8192))
        object.__setattr__(self, "grid_size", _pair(self.grid_size, "grid_size", 3, 1024))
        for name, low, high in (("num_filter", 1, 1024), ("kernel_size", 3, 15)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if not isinstance(self.with_3d, bool):
            raise ValueError("with_3d must be a bool")
        if self.grid_convention not in GRID_CONVENTIONS:
            raise ValueError(f"grid_convention must be one of {GRID_CONVENTIONS}")
        if self.architecture != ARCHITECTURE:
            raise ValueError(f"architecture must be {ARCHITECTURE!r}")
        if min(self.grid_size) <= self.kernel_size // 2:
            raise ValueError("grid sides must exceed kernel_size // 2 (reflect padding)")

    @property
    def natural_grid_size(self) -> tuple[int, int]:
        return natural_grid_size(self.input_size)

    def to_dict(self) -> dict[str, Any]:
        return {key: list(v) if isinstance(v, tuple) else v for key, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DewarpModelConfig:
        if not isinstance(data, dict):
            raise ValueError("model config must be a JSON object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown model config fields: {sorted(unknown)}")
        return cls(**data)


UVDOC_CONFIG = DewarpModelConfig(grid_convention="uvdoc")
