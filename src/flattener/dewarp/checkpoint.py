"""Atomic dewarp checkpoints with sha256 JSON sidecars and a latest pointer.

A checkpoint ``step_XXXXXXXX.pt`` holds tensors and plain Python data only
(it loads with ``weights_only=True``). Its sidecar ``<name>.pt.json`` repeats
every non-state field (track, configs, data spec hash, code hash, lineage,
metrics) plus the file's sha256. ``latest.json`` is replaced only after a
checkpoint and its sidecar are durable, so an interrupted save never breaks
resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from ..contracts import FlattenerError, Status
from .model import sha256_file

CHECKPOINT_SCHEMA = "flattener-dewarp-checkpoint-1"
SIDECAR_SCHEMA = "flattener-dewarp-sidecar-1"
STATE_KEYS = frozenset({"model_state", "optimizer_state", "scaler_state", "rng_state"})


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.IO_ERROR, reason, message)


def code_fingerprint() -> dict[str, Any]:
    """sha256 of ``src/flattener/dewarp/*.py`` (per file and combined)."""
    files = {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": combined, "files": files}


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "cuda": str(torch.version.cuda),
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write via a temporary file in the same directory, fsync, then replace.

    On Windows ``os.replace`` fails while another process (a status monitor)
    has the target open, so the replace is retried for up to about 5 s.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(50):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.1)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_safe(value: Any) -> Any:
    """``value`` with non-finite floats (AMP overflow, diverged metrics) replaced by None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, lambda stream: stream.write(text.encode("utf-8")))


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically save ``payload`` and its sidecar; return the sidecar content."""
    path = Path(path)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise _error("invalid_checkpoint", f"payload schema must be {CHECKPOINT_SCHEMA}")
    atomic_write(path, lambda stream: torch.save(payload, stream))
    sidecar = {key: value for key, value in payload.items() if key not in STATE_KEYS}
    sidecar.update(
        schema=SIDECAR_SCHEMA,
        checkpoint_schema=CHECKPOINT_SCHEMA,
        file=path.name,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
    )
    write_json(sidecar_path(path), sidecar)
    return sidecar


def read_checkpoint(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Load any checkpoint-like file safely; with ``verify``, a present sidecar must match."""
    path = Path(path)
    if not path.is_file():
        raise _error("missing_checkpoint", f"{path} does not exist")
    sidecar = sidecar_path(path)
    if verify and sidecar.is_file():
        expected = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256")
        actual = sha256_file(path)
        if expected != actual:
            raise _error("checksum_mismatch", f"{path}: sha256 {actual} != sidecar {expected}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_checkpoint(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Load a dewarp checkpoint; with ``verify`` the sidecar is required and must match."""
    path = Path(path)
    if verify and not sidecar_path(path).is_file():
        raise _error("missing_sidecar", f"{sidecar_path(path)} is missing")
    payload = read_checkpoint(path, verify=verify)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise _error("invalid_checkpoint", f"{path} is not a {CHECKPOINT_SCHEMA} checkpoint")
    return payload


def latest_checkpoint(directory: str | Path) -> Path | None:
    pointer = Path(directory, "latest.json")
    if not pointer.is_file():
        return None
    return Path(directory, json.loads(pointer.read_text(encoding="utf-8"))["file"])


def commit_checkpoint(
    directory: str | Path, payload: dict[str, Any], *, keep: int, is_best: bool
) -> Path:
    """Save ``step_<step>.pt``, move ``latest.json`` (and ``best.json``), prune old files."""
    directory = Path(directory)
    path = directory / f"step_{int(payload['step']):08d}.pt"
    sidecar = save_checkpoint(path, payload)
    pointer = {"file": path.name, "sha256": sidecar["sha256"], "step": payload["step"]}
    write_json(directory / "latest.json", pointer)
    if is_best:
        write_json(directory / "best.json", {**pointer, "metrics": payload.get("metrics", {})})
    protected = {path.name}
    for name in ("latest.json", "best.json"):
        if (directory / name).is_file():
            protected.add(json.loads((directory / name).read_text(encoding="utf-8"))["file"])
    steps = sorted(directory.glob("step_*.pt"))
    for old in steps[: max(0, len(steps) - max(1, keep))]:
        if old.name not in protected:
            old.unlink(missing_ok=True)
            sidecar_path(old).unlink(missing_ok=True)
    return path
