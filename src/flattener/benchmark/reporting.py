"""Hashing, environment capture and atomic JSON reports shared by the benchmark tools."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._version import __version__
from ..evidence import evidence

PREDICTIONS_FILE = "predictions.json"
PREDICTIONS_SCHEMA = "flattener-dewarp-predictions-2"
LEGACY_PREDICTIONS_SCHEMA = "flattener-dewarp-docunet-1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def environment() -> dict[str, Any]:
    """Versions that can change benchmark numbers."""
    import numpy
    import PIL
    import torch

    return {
        "flattener": __version__,
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def encode_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json_atomic(
    path: str | os.PathLike[str], payload: Any, *, overwrite: bool = False
) -> str:
    """Write ``payload`` via a synced temporary file; return the written SHA-256.

    Without ``overwrite`` the file is created only when absent.
    """
    destination = Path(path)
    encoded = encode_json(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(f"{destination} exists; pass overwrite to replace it")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return sha256_bytes(encoded)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(sum(values) / len(values))


def prediction_record(predictions_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """What ``flattener.dewarp.predict`` recorded about a predictions directory.

    Reads ``predictions.json`` and returns its hash, the model, the force-render flag,
    the counts and ``failures``: output file name -> status, reason and whether a forced
    image was written. Items of the older ``flattener-dewarp-docunet-1`` format that
    were rendered unchecked (``fallback``) count as ``invalid_transform``; that format
    ran no coverage check. Returns ``None`` when the directory has no record, and
    ``failures`` is ``None`` for a record of another format.
    """
    path = Path(predictions_dir) / PREDICTIONS_FILE
    if not path.is_file():
        return None
    data = path.read_bytes()
    payload = json.loads(data)
    payload = payload if isinstance(payload, dict) else {}
    items = [item for item in payload.get("items", []) if isinstance(item, dict)]
    schema = payload.get("schema")
    failures: dict[str, Any] | None = None
    if schema == PREDICTIONS_SCHEMA:
        failures = {
            str(item.get("output")): {
                "status": item.get("status"),
                "reason": item.get("reason"),
                "forced": bool(item.get("forced")),
                "written": item.get("action") in ("written", "kept"),
            }
            for item in items
            if item.get("status") != "success"
        }
    elif schema == LEGACY_PREDICTIONS_SCHEMA:
        failures = {
            str(item.get("output")): {
                "status": "invalid_transform",
                "reason": "recorded_as_fallback",
                "forced": True,
                "written": True,
            }
            for item in items
            if item.get("fallback")
        }
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    return {
        "path": str(path),
        "sha256": sha256_bytes(data),
        "schema": schema,
        "model": {key: model.get(key) for key in ("kind", "path", "sha256", "step", "track")},
        "force_render": payload.get("force_render"),
        "inputs": payload.get("inputs"),
        "items": len(items),
        "failed": None if failures is None else len(failures),
        "failures": failures,
    }


def prediction_status(record: dict[str, Any] | None, name: str) -> str | None:
    """Recorded check status of one output file; ``None`` when nothing was recorded."""
    if record is None or record["failures"] is None:
        return None
    failure = record["failures"].get(name)
    return "success" if failure is None else failure["status"]


def report_evidence(record: dict[str, Any] | None) -> dict[str, Any]:
    """Evidence of a DocUNet report from the predictions record (``prediction_record``).

    Stress-variant inputs make it ``docunet_stress``; anything else is scored as the
    standard benchmark. Without a record the model provenance is ``unknown``.
    """
    inputs = "" if record is None else str(record.get("inputs") or "")
    measurement = "docunet_stress" if inputs.startswith("docunet_stress") else "docunet_standard"
    return evidence(measurement, None if record is None else record["model"])
