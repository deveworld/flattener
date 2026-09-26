"""Restricted image input and transactional PNG/JSON output adapters.

Output files are committed atomically one at a time, and a failed commit rolls
back this call's changes. Concurrent readers can see a partly committed group.
Use one writer per output directory. Other files in the directory are left
alone, also with ``overwrite``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from .contracts import FlattenerError, RenderConfig, RenderResult, Status
from .renderer import _handle_failure, _new_metadata, render
from .validation import validate_config, validate_image, validate_size

_ARTIFACT_NAMES = {
    "image": "image.png",
    "source_mask": "source_mask.png",
    "target_mask": "target_mask.png",
    "valid_mask": "valid_mask.png",
    "metadata": "metadata.json",
}


def _path(value: str | os.PathLike[str], *, output: bool = False) -> Path:
    try:
        path = Path(value)
        if "\x00" in str(path):
            raise ValueError("Filesystem paths cannot contain NUL characters.")
        return path
    except (TypeError, ValueError) as exc:
        raise FlattenerError(
            Status.IO_ERROR if output else Status.INPUT_ERROR,
            "invalid_path",
            "Expected a filesystem path.",
        ) from exc


def load_image(
    path: str | os.PathLike[str], *, config: RenderConfig | None = None
) -> NDArray[np.uint8]:
    """Decode one 8-bit RGB/L JPEG or PNG and apply its EXIF orientation once.

    Raises on failure; ``render_file`` applies the configured failure policy and
    returns a ``RenderResult``. Pixel values are used as stored, without ICC conversion.
    """
    cfg = validate_config(config)
    source = _path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with source.open("rb") as stream, Image.open(stream) as decoded:
                if decoded.format not in {"JPEG", "PNG"}:
                    raise FlattenerError(
                        Status.INPUT_ERROR,
                        "unsupported_format",
                        "Only JPEG and PNG image files are supported.",
                    )
                validate_size(
                    (decoded.height, decoded.width),
                    max_pixels=cfg.max_input_pixels,
                    max_side=cfg.max_side,
                    name="input",
                )
                if getattr(decoded, "n_frames", 1) != 1:
                    raise FlattenerError(
                        Status.INPUT_ERROR,
                        "unsupported_frames",
                        "Multi-frame images are not supported.",
                    )
                if decoded.format == "PNG":
                    # PNG IHDR puts bit depth at byte 24. Pillow exposes 16-bit
                    # truecolor PNG as RGB after narrowing, so mode is insufficient.
                    position = stream.tell()
                    stream.seek(24)
                    bit_depth = stream.read(1)
                    stream.seek(position)
                    if bit_depth != b"\x08":
                        raise FlattenerError(
                            Status.INPUT_ERROR,
                            "unsupported_mode",
                            "Only 8-bit PNG samples are supported.",
                        )
                if decoded.mode not in {"RGB", "L"} or "transparency" in decoded.info:
                    raise FlattenerError(
                        Status.INPUT_ERROR,
                        "unsupported_mode",
                        "Only 8-bit RGB or grayscale images without transparency are supported.",
                    )
                oriented = ImageOps.exif_transpose(decoded)
                try:
                    rgb = oriented.convert("RGB")
                    try:
                        image = np.array(rgb, dtype=np.uint8, copy=True)
                    finally:
                        rgb.close()
                finally:
                    oriented.close()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise FlattenerError(
            Status.INPUT_ERROR,
            "size_limit_exceeded",
            "Pillow rejected the image dimensions before decoding.",
        ) from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise FlattenerError(
            Status.INPUT_ERROR,
            "decode_failed",
            f"Could not decode image: {exc}",
        ) from exc
    validate_image(image, cfg)
    return image


def _validate_result(result: RenderResult) -> dict[str, NDArray[Any]]:
    if not isinstance(result, RenderResult):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_result", "Expected a RenderResult.")
    if not isinstance(result.metadata, dict):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_result", "Result metadata must be a dictionary."
        )
    arrays: dict[str, NDArray[Any]] = {}
    for key in ("image", "source_mask", "target_mask", "valid_mask"):
        array = getattr(result, key)
        if array is None:
            continue
        expected_shape = (
            isinstance(array, np.ndarray)
            and (array.ndim == 3 and array.shape[2] == 3 if key == "image" else array.ndim == 2)
            and all(side > 0 for side in array.shape)
        )
        dtype = np.uint8 if key == "image" else np.bool_
        if not expected_shape or array.dtype != dtype:
            raise FlattenerError(
                Status.INPUT_ERROR,
                "invalid_result",
                f"Result {key} has an invalid shape or dtype.",
            )
        arrays[key] = array
    sizes = {array.shape[:2] for array in arrays.values()}
    if len(sizes) > 1:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_result", "Result image and mask sizes must agree."
        )
    return arrays


def save_result(
    result: RenderResult,
    output_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Save present arrays and JSON, refusing existing targets by default.

    Images and masks are PNG. The JSON records the processing status, also for
    failures and fallback output. On success, ``result.metadata`` also gets the
    artifact paths and the save time, which covers image and mask staging.
    """
    arrays = _validate_result(result)
    if not isinstance(overwrite, bool):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_overwrite", "overwrite must be a bool.")
    directory = _path(output_dir, output=True)
    started = time.perf_counter()
    keys = [*arrays, "metadata"]
    paths = {key: str(directory / _ARTIFACT_NAMES[key]) for key in keys}
    staging: Path | None = None
    committed: list[str] = []
    backups: dict[str, Path] = {}
    cleanup = True
    pending_exception: BaseException | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for key in keys:
            target = directory / _ARTIFACT_NAMES[key]
            if target.exists() or target.is_symlink():
                if not overwrite:
                    raise FlattenerError(
                        Status.IO_ERROR,
                        "output_exists",
                        f"Output already exists: {target}",
                    )
                if not target.is_file() or target.is_symlink():
                    raise FlattenerError(
                        Status.IO_ERROR,
                        "invalid_output_target",
                        f"Cannot replace a non-regular output file: {target}",
                    )
        staging = Path(tempfile.mkdtemp(prefix=".flattener-", dir=directory))
        for key, array in arrays.items():
            pixels = array if key == "image" else array.astype(np.uint8) * 255
            with Image.fromarray(pixels) as encoded:
                encoded.save(staging / _ARTIFACT_NAMES[key], format="PNG")

        metadata = dict(result.metadata)
        timings = dict(metadata.get("timings", {}))
        timings["save"] = time.perf_counter() - started
        if metadata.get("timing_scope") == "file" and timings.get("total") is not None:
            timings["total"] += timings["save"]
        metadata["timings"] = timings
        metadata["artifacts"] = paths
        metadata["timing_notes"] = (
            "Save includes image/mask staging; JSON serialization and file commit "
            "are excluded. Measure the call externally for end-to-end wall time."
        )
        record = {
            "status": str(result.status),
            "reason_code": result.reason_code,
            "message": result.message,
            "used_fallback": result.used_fallback,
            "metadata": metadata,
            "artifacts": paths,
        }
        try:
            serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
            (staging / "metadata.json").write_text(serialized + "\n", encoding="utf-8")
        except (TypeError, ValueError) as exc:
            raise FlattenerError(
                Status.IO_ERROR,
                "metadata_serialization_failed",
                f"Result metadata is not finite JSON data: {exc}",
            ) from exc

        for key in keys:
            target = directory / _ARTIFACT_NAMES[key]
            if overwrite:
                if target.exists():
                    backup = staging / f"previous-{_ARTIFACT_NAMES[key]}"
                    os.link(target, backup)
                    backups[key] = backup
                os.replace(staging / _ARTIFACT_NAMES[key], target)
            else:
                # link creates the target atomically and only if it is absent.
                os.link(staging / _ARTIFACT_NAMES[key], target)
            committed.append(key)
    except BaseException as exc:
        # Cleanup must also protect previous files on interruption/allocation
        # failure. Those exceptions are re-raised unchanged after rollback.
        pending_exception = exc
        cleanup = False
        rollback_errors = []
        for key in reversed(committed):
            target = directory / _ARTIFACT_NAMES[key]
            try:
                if key in backups:
                    os.replace(backups[key], target)
                else:
                    target.unlink()
            except (OSError, UnicodeError) as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        cleanup = not rollback_errors
        if not isinstance(exc, (OSError, UnicodeError)):
            if rollback_errors:
                exc.add_note(f"Output rollback incomplete; recover files from {staging}")
            raise
        failure = FlattenerError(
            Status.IO_ERROR,
            "save_failed",
            f"Could not save result: {exc}",
            metadata={
                "save_transaction": {
                    "rolled_back": not rollback_errors,
                    "rollback_errors": rollback_errors,
                    "recovery_directory": str(staging) if rollback_errors else None,
                }
            },
        )
        pending_exception = failure
        raise failure from exc
    finally:
        if staging is not None and cleanup:
            try:
                shutil.rmtree(staging)
            except OSError as exc:
                if pending_exception is not None:
                    pending_exception.add_note(
                        f"Output staging cleanup failed; inspect {staging}: {exc}"
                    )
                    # Preserve the exception already being propagated.
                    raise pending_exception
                raise FlattenerError(
                    Status.IO_ERROR,
                    "save_cleanup_failed",
                    f"Could not remove the output staging directory: {exc}",
                    metadata={"recovery_directory": str(staging)},
                ) from exc
    result.metadata = metadata
    return paths


def render_file(
    input_path: str | os.PathLike[str],
    backward_map: NDArray[np.float32],
    *,
    target_mask: NDArray[np.bool_] | None = None,
    config: RenderConfig | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> RenderResult:
    """Decode, render, optionally save, and apply one consistent failure policy."""
    started = time.perf_counter()
    cfg = config if isinstance(config, RenderConfig) else RenderConfig()
    metadata = _new_metadata(cfg, timing_scope="file")
    if isinstance(backward_map, np.ndarray) and backward_map.ndim >= 2:
        metadata["requested_output_size"] = list(backward_map.shape[:2])
    image = None
    metadata["failure_stage"] = "prepare_input"
    try:
        cfg = validate_config(config)
        image = load_image(input_path, config=cfg)
        metadata["source_size"] = list(image.shape[:2])
        decoded_seconds = time.perf_counter() - started
        metadata["failure_stage"] = "render"
        try:
            result = render(image, backward_map, target_mask=target_mask, config=cfg)
        except FlattenerError as exc:
            metadata.update(exc.metadata)
            metadata["timing_scope"] = "file"
            timings = metadata["timings"]
            timings["prepare_input"] = (timings.get("prepare_input") or 0) + decoded_seconds
            exc.metadata = metadata
            raise
        metadata = result.metadata
        metadata["timing_scope"] = "file"
        timings = metadata["timings"]
        timings["prepare_input"] = (timings.get("prepare_input") or 0) + decoded_seconds
        timings["total"] = time.perf_counter() - started
        metadata["input_path"] = str(_path(input_path))
    except FlattenerError as exc:
        metadata["timing_scope"] = "file"
        timings = metadata["timings"]
        if image is None:
            timings["prepare_input"] = time.perf_counter() - started
        timings["total"] = time.perf_counter() - started
        # A malformed failure policy can never be interpreted as return/original.
        if not isinstance(cfg.on_error, str) or cfg.on_error not in {"raise", "return", "original"}:
            exc.metadata = metadata
            raise
        result = _handle_failure(exc, cfg, image=image, metadata=metadata)
        result.metadata["timings"]["total"] = time.perf_counter() - started

    if output_dir is not None:
        stage_started = time.perf_counter()
        metadata = result.metadata
        metadata["processing_status"] = str(result.status)
        metadata["processing_reason_code"] = result.reason_code
        try:
            save_result(result, output_dir, overwrite=overwrite)
        except FlattenerError as exc:
            metadata["failure_stage"] = "save"
            metadata["timings"]["save"] = time.perf_counter() - stage_started
            metadata["timings"]["total"] = time.perf_counter() - started
            result = _handle_failure(exc, cfg, image=image, metadata=metadata)
            result.metadata["timings"]["total"] = time.perf_counter() - started
    return result
