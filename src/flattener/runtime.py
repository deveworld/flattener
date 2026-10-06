"""Where the scanner's networks run: ONNX Runtime on the CPU, or PyTorch.

``auto`` uses PyTorch on a CUDA GPU when PyTorch is installed and sees one, and ONNX
Runtime on the CPU otherwise. ONNX models take and return NumPy arrays; their batch size
is 1, so a batch runs one item at a time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import FlattenerError, Status

BACKENDS = ("auto", "onnx", "torch")
_threads: int | None = None


def _usage(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "unsupported_backend", message)


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def cuda_available() -> bool:
    """A CUDA GPU that PyTorch can use (False without PyTorch)."""
    if not torch_installed():
        return False
    import torch

    return bool(torch.cuda.is_available())


def check_device(device: str) -> str:
    """The kind of a device name (``auto``, ``cpu``, ``cuda`` or ``cuda:N``); a CUDA
    device must be available to PyTorch."""
    kind, _, index = device.partition(":") if isinstance(device, str) else ("", "", "")
    if kind not in ("auto", "cpu", "cuda") or (index and (kind != "cuda" or not index.isdigit())):
        raise _usage(f"device must be auto, cpu, cuda or cuda:N, got {device!r}")
    if kind == "cuda" and not cuda_available():
        if not torch_installed():
            raise _usage("a CUDA device needs PyTorch; install torch (uvx --with torch ...)")
        raise _usage("no CUDA GPU is available to PyTorch")
    return kind


def resolve(backend: str = "auto", device: str = "auto") -> tuple[str, str]:
    """The (backend, device) to run on.

    ``auto`` picks PyTorch on CUDA when available, else ONNX Runtime on the CPU. ONNX
    Runtime runs on the CPU only; a CUDA device needs PyTorch."""
    if backend not in BACKENDS:
        raise _usage(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "onnx" and device not in ("auto", "cpu"):
        raise _usage("the onnx backend runs on the cpu; use the torch backend for a GPU")
    kind = check_device(device)
    if backend == "onnx":
        return "onnx", "cpu"
    if backend == "torch" or kind not in ("auto", "cpu"):
        if not torch_installed():
            raise _usage(
                "PyTorch is not installed; install torch to use the torch backend or a GPU"
            )
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        return "torch", device
    if device == "auto" and cuda_available():
        return "torch", "cuda"
    return "onnx", "cpu"


class OnnxModel:
    """An ONNX Runtime CPU session of one exported network."""

    def __init__(self, path: str | Path, *, threads: int | None = None) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        options = ort.SessionOptions()
        threads = _threads if threads is None else threads
        if threads is not None:
            options.intra_op_num_threads = threads
        options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(
                str(self.path), options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:  # onnxruntime raises its own exception types
            raise FlattenerError(
                Status.MODEL_ERROR, "invalid_model", f"{self.path}: {error}"
            ) from error
        self.inputs = [item.name for item in self.session.get_inputs()]
        self.outputs = [item.name for item in self.session.get_outputs()]

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        feeds = {
            name: np.ascontiguousarray(array, dtype=np.float32)
            for name, array in zip(self.inputs, arrays, strict=True)
        }
        return self.session.run(None, feeds)

    def output(self, name: str, *arrays: np.ndarray) -> np.ndarray:
        """The output ``name`` for one call."""
        return self(*arrays)[self.outputs.index(name)]


def device_type(device: Any) -> str:
    """``cpu``, ``cuda``... of a device name or a ``torch.device``."""
    return getattr(device, "type", None) or str(device).split(":", 1)[0]


def set_threads(count: int) -> None:
    """CPU threads for sampling, ONNX sessions made later and PyTorch (if loaded)."""
    global _threads
    from . import ops

    ops.set_threads(count)
    _threads = count
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(count)
