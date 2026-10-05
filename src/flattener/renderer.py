"""Validated CPU sampling of an oriented RGB source image."""

import math
import platform
from dataclasses import fields
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
import PIL
import torch
import torch.nn.functional as functional

from ._version import __version__
from .contracts import SPEC_VERSION, FlattenerError, RenderConfig, RenderResult, Status
from .validation import (
    validate_config,
    validate_geometry,
    validate_image,
    validate_map,
    validate_target_mask,
)


def _json_value(value: Any) -> Any:
    """Keep even a rejected configuration representable in strict JSON."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        try:
            number = float(value)
        except OverflowError:
            return str(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return f"<{type(value).__name__}>"


def _new_metadata(config: RenderConfig, *, timing_scope: str = "render") -> dict[str, Any]:
    return {
        "spec_version": SPEC_VERSION,
        "config": _json_value(
            {field.name: getattr(config, field.name) for field in fields(config)}
        ),
        "source_size": None,
        "requested_output_size": None,
        "actual_output_size": None,
        "coarse_grid_size": None,
        "model_id": None,
        "model_sha256": None,
        "versions": {
            "flattener": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": str(torch.__version__),
        },
        "cpu_threads": torch.get_num_threads(),
        "validation": {},
        "failure_stage": None,
        "timing_scope": timing_scope,
        "timings": dict.fromkeys(
            (
                "load_model",
                "prepare_input",
                "predict",
                "process_coordinates",
                "render",
                "save",
                "total",
            )
        ),
    }


def _handle_failure(
    error: FlattenerError,
    config: RenderConfig,
    *,
    image: np.ndarray | None,
    metadata: dict[str, Any],
) -> RenderResult:
    """Apply policy to an expected failure; image must have passed validation."""
    metadata.update(error.metadata)
    metadata["actual_output_size"] = None
    if config.on_error == "raise":
        error.metadata = metadata
        raise error
    fallback = image.copy() if config.on_error == "original" and image is not None else None
    if fallback is not None:
        metadata["actual_output_size"] = list(fallback.shape[:2])
    return RenderResult(
        status=error.status,
        reason_code=error.reason_code,
        message=error.message,
        image=fallback,
        used_fallback=fallback is not None,
        metadata=metadata,
    )


def render(
    image: np.ndarray,
    backward_map: np.ndarray,
    *,
    target_mask: np.ndarray | None = None,
    config: RenderConfig | None = None,
) -> RenderResult:
    """Sample ``image`` using a dense, normalized output-to-source map.

    Images must be RGB uint8 HWC; maps must be float32 HWC2 in (x, y)
    order. All processing takes place on CPU. Callers' arrays are never
    modified.
    """
    started = perf_counter()
    candidate = RenderConfig() if config is None else config
    # Without a valid policy it is impossible to select return versus raise.
    if (
        not isinstance(candidate, RenderConfig)
        or not isinstance(candidate.on_error, str)
        or candidate.on_error not in ("raise", "return", "original")
    ):
        metadata = _new_metadata(RenderConfig())
        metadata["requested_config"] = _json_value(config)
        metadata["failure_stage"] = "prepare_input"
        metadata["timings"]["prepare_input"] = perf_counter() - started
        metadata["timings"]["total"] = perf_counter() - started
        raise FlattenerError(
            Status.INPUT_ERROR,
            "invalid_config",
            "config must be RenderConfig with on_error='raise', 'return', or 'original'",
            metadata=metadata,
        )

    metadata = _new_metadata(candidate)
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        metadata["source_size"] = list(image.shape[:2])
    if isinstance(backward_map, np.ndarray) and backward_map.ndim >= 2:
        metadata["requested_output_size"] = list(backward_map.shape[:2])
    trusted_image = None
    stage = "prepare_input"
    stage_started = started
    try:
        candidate = validate_config(candidate)
        validate_image(image, candidate)
        trusted_image = image
        validate_map(backward_map, candidate)
        mask = validate_target_mask(target_mask, backward_map.shape[:2])
        metadata["timings"][stage] = perf_counter() - stage_started

        stage = "process_coordinates"
        stage_started = perf_counter()
        source_mask, valid_mask, stats = validate_geometry(backward_map, mask, candidate)
        metadata["validation"] = stats
        metadata["timings"][stage] = perf_counter() - stage_started

        stage = "render"
        stage_started = perf_counter()
        # Keep owned source storage. A source-borrowing experiment reduced
        # copy time but increased measured whole-pipeline peak RSS; retaining
        # this buffer also normalizes read-only and unusual-stride inputs.
        source = np.array(image, order="C", copy=True)
        all_valid = bool(valid_mask.all())
        # grid_sample reads its grid. A fully valid contiguous writable grid
        # can also be borrowed; sanitizing discarded positions needs a copy.
        grid = (
            backward_map
            if all_valid and backward_map.flags.c_contiguous and backward_map.flags.writeable
            else np.array(backward_map, order="C", copy=True)
        )
        if not all_valid:
            # Never send inactive NaN, Inf, or huge coordinates to the backend.
            grid[~valid_mask] = 0
        with torch.inference_mode():
            tensor = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).float().div_(255)
            try:
                sampled = functional.grid_sample(
                    tensor,
                    torch.from_numpy(grid).unsqueeze(0),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
            except RuntimeError as error:
                # Allocator failures must not be disguised as a normal fallback.
                if "out of memory" in str(error).lower() or "can't allocate memory" in str(error):
                    raise MemoryError(str(error)) from error
                raise FlattenerError(Status.RENDER_ERROR, "sampling_failed", str(error)) from error
            values = sampled.squeeze(0).permute(1, 2, 0).numpy()
            np.clip(values, 0, 1, out=values)
            values *= 255
            values += 0.5
            np.floor(values, out=values)
            output = values.astype(np.uint8)
        if not all_valid:
            output[~valid_mask] = candidate.fill_rgb
        metadata["actual_output_size"] = list(output.shape[:2])
        metadata["timings"][stage] = perf_counter() - stage_started
        metadata["timings"]["total"] = perf_counter() - started
        return RenderResult(
            status=Status.SUCCESS,
            image=output,
            source_mask=source_mask,
            target_mask=mask,
            valid_mask=valid_mask,
            metadata=metadata,
        )
    except FlattenerError as error:
        metadata["failure_stage"] = stage
        metadata["timings"][stage] = perf_counter() - stage_started
        metadata["timings"]["total"] = perf_counter() - started
        result = _handle_failure(error, candidate, image=trusted_image, metadata=metadata)
        result.metadata["timings"]["total"] = perf_counter() - started
        return result
