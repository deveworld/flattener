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
from ..evidence import evidence, initialization

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


def _no_inset() -> dict[str, Any]:
    """``flattener.dewarp.predict.inset_record(None)``: the outline as predicted."""
    return {"x": 0.0, "y": 0.0, "preset": None}


def _no_rerun() -> dict[str, Any]:
    """``flattener.dewarp.predict.rerun_record()``: the whole-image prediction alone."""
    keys = ("margin", "max_angle_deg", "min_agreement")
    return {"passes": 0, "deskew": False, **dict.fromkeys(keys)}


def _rectify_pipeline() -> dict[str, Any]:
    """``flattener.dewarp.predict.DEFAULT_PIPELINE``: predictions made by ``rectify``."""
    return {"name": "rectify"}


def prediction_record(predictions_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """What ``flattener.dewarp.predict`` recorded about a predictions directory.

    Reads ``predictions.json`` and returns its hash, the model, the force-render flag,
    ``quantize`` (how the images became uint8: ``round`` or ``truncate``; a current-format
    record without the field predates the option and was rounded; ``None`` for other
    formats), ``inset`` (the recorded ``{"x", "y", "preset"}`` inset of the outline; a
    current-format record without the field predates the option and has none; ``None``
    for other formats), ``rerun`` (the recorded crop-and-rerun settings,
    ``flattener.dewarp.predict.rerun_record``; a current-format record without the field
    holds whole-image predictions; ``None`` for other formats), ``pipeline`` (the
    recorded pipeline identity; a current-format record without the field was made by
    ``rectify``; ``None`` for other formats), ``complete`` (whether the
    batch finished; ``None`` for records that predate the flag), ``outputs`` (the recorded
    output file names) and ``failures``: output file name -> status, reason and whether a
    forced image was written. Items of the older ``flattener-dewarp-docunet-1`` format
    that were rendered unchecked (``fallback``) count as ``invalid_transform``; that
    format ran no coverage check. Returns ``None`` when the directory has no record;
    ``failures`` and ``outputs`` are ``None`` for a record of another format.
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
    outputs = None if failures is None else sorted({str(item.get("output")) for item in items})
    return {
        "path": str(path),
        "sha256": sha256_bytes(data),
        "schema": schema,
        "model": {
            **{key: model.get(key) for key in ("kind", "path", "sha256", "step", "track")},
            "initialization": initialization(model),
        },
        "force_render": payload.get("force_render"),
        "quantize": payload.get("quantize", "round" if schema == PREDICTIONS_SCHEMA else None),
        "inset": payload.get("inset", _no_inset() if schema == PREDICTIONS_SCHEMA else None),
        "rerun": payload.get("rerun", _no_rerun() if schema == PREDICTIONS_SCHEMA else None),
        "pipeline": payload.get(
            "pipeline", _rectify_pipeline() if schema == PREDICTIONS_SCHEMA else None
        ),
        "inputs": payload.get("inputs"),
        "complete": payload.get("complete") if schema == PREDICTIONS_SCHEMA else None,
        "items": len(items),
        "outputs": outputs,
        "failed": None if failures is None else len(failures),
        "failures": failures,
    }


def prediction_status(record: dict[str, Any] | None, name: str) -> str | None:
    """Recorded check status of one output file.

    ``"unrecorded"`` when the record does not list the file (an output left from an
    earlier or interrupted run), ``None`` when there is no usable record at all.
    """
    if record is None or record["failures"] is None:
        return None
    failure = record["failures"].get(name)
    if failure is not None:
        return failure["status"]
    outputs = record.get("outputs")
    return "unrecorded" if outputs is not None and name not in outputs else "success"


def failure_counts(statuses: list[str | None]) -> dict[str, int | None]:
    """``prediction_failures``: images not recorded as passing their checks (a failed
    check or no record of the image); ``prediction_unrecorded``: those without a record.
    Both are ``None`` when the predictions carry no record."""
    if any(status is None for status in statuses):
        return {"prediction_failures": None, "prediction_unrecorded": None}
    return {
        "prediction_failures": sum(status != "success" for status in statuses),
        "prediction_unrecorded": sum(status == "unrecorded" for status in statuses),
    }


def report_evidence(
    record: dict[str, Any] | None, measurement: str | None = None
) -> dict[str, Any]:
    """Evidence of a benchmark report from the predictions record (``prediction_record``).

    Stress-variant inputs make it ``docunet_stress``, UVDoc benchmark inputs
    ``uvdoc_benchmark``; anything else is scored as the standard DocUNet benchmark
    (``measurement`` overrides this). Without a record the model provenance is
    ``unknown``. ``quantize`` is the record's output quantization, ``inset`` its
    outline inset and ``rerun`` its crop-and-rerun settings (``None`` when unknown).
    """
    inputs = "" if record is None else str(record.get("inputs") or "")
    if measurement is None:
        measurement = "docunet_standard"
        for prefix in ("docunet_stress", "uvdoc_benchmark"):
            if inputs.startswith(prefix):
                measurement = prefix
    return {
        **evidence(measurement, None if record is None else record["model"]),
        "quantize": None if record is None else record.get("quantize"),
        "inset": None if record is None else record.get("inset"),
        "rerun": None if record is None else record.get("rerun"),
    }
