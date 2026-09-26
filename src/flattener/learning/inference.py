"""CPU inference that feeds predicted coordinate maps to the validated renderer.

``load_predictor`` reads learning-0.1 checkpoints. Load coordinate-study
checkpoints, including the default ``selected.pt``, with
``coordinate_study.load_coordinate_predictor``; ``unwarp`` accepts both.
"""

import hashlib
import io
import os
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray

from ..contracts import FlattenerError, RenderConfig, RenderResult, Status
from ..renderer import _handle_failure, _json_value, _new_metadata, render
from ..validation import validate_config, validate_image, validate_size
from .model import (
    MODEL_VERSION,
    PREPROCESS_VERSION,
    CoordinatePredictor,
    ModelConfig,
    dense_maps,
    preprocess_image,
)
from .spatial import expand_coordinate_residual


def _model_error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.MODEL_ERROR, reason, message)


def _allocation_error(error: RuntimeError) -> None:
    if "out of memory" in str(error).lower() or "can't allocate memory" in str(error):
        raise MemoryError(str(error)) from error


def _output_size(size, config: RenderConfig, model_config: ModelConfig) -> tuple[int, int]:
    size = validate_size(
        size,
        max_pixels=config.max_output_pixels,
        max_side=config.max_side,
        name="output_size",
    )
    if any(a < b for a, b in zip(size, model_config.grid_size, strict=True)):
        raise FlattenerError(
            Status.INPUT_ERROR,
            "output_smaller_than_grid",
            "output_size may not shrink either predicted grid axis",
        )
    return size


@dataclass(frozen=True, slots=True)
class LoadedPredictor:
    """A loaded model reused across images; global torch settings stay untouched.

    ``predict_map`` raises on error; ``unwarp`` applies the renderer failure policy.
    """

    _model: torch.nn.Module
    model_config: ModelConfig
    model_sha256: str
    model_id: str
    checkpoint_path: str
    checksum_verified: bool
    load_seconds: float
    map_mode: str = "legacy"

    def _metadata(self, metadata: dict) -> None:
        metadata.update(
            model_id=self.model_id,
            model_sha256=self.model_sha256,
            coarse_grid_size=list(self.model_config.grid_size),
            model_input_size=list(self.model_config.input_size),
            model_config=_json_value(asdict(self.model_config)),
            model_load_seconds=self.load_seconds,
            model_checksum_verified=self.checksum_verified,
            checkpoint_path=self.checkpoint_path,
            preprocessing_version=PREPROCESS_VERSION,
            input_resize="full extent; independent axes; no crop or padding",
            model_map_mode=self.map_mode,
            quality_status="not_evaluated",
        )
        metadata["timings"]["load_model"] = 0.0

    def _predict(self, image: NDArray[np.uint8], output_size: tuple[int, int], metadata: dict):
        stage = "prepare_input"
        started = perf_counter()
        try:
            inputs = preprocess_image(image, self.model_config).unsqueeze(0)
            metadata["timings"][stage] = (
                (metadata["timings"][stage] or 0) + perf_counter() - started
            )
            stage = "predict"
            started = perf_counter()
            with torch.inference_mode():
                dense = None
                if self.map_mode == "analytic":
                    forward_dense = getattr(self._model, "forward_dense", None)
                    if not callable(forward_dense):
                        raise _model_error("invalid_model_output", "model has no analytic decoder")
                    kwargs = ({"photo_size": tuple(image.shape[:2])}
                              if getattr(self._model, "requires_photo_size", False) else {})
                    outputs = forward_dense(inputs, output_size, **kwargs)
                    if not isinstance(outputs, tuple) or len(outputs) != 2:
                        raise _model_error("invalid_model_output", "invalid analytic output pair")
                    residual, dense = outputs
                elif self.map_mode in ("legacy", "linear"):
                    residual = self._model(inputs)
                else:
                    raise _model_error("invalid_model_output", "unsupported coordinate decoder")
                expected = (1, 2, *self.model_config.grid_size)
                if (
                    not isinstance(residual, torch.Tensor)
                    or tuple(residual.shape) != expected
                    or residual.dtype != torch.float32
                    or residual.device.type != "cpu"
                ):
                    raise _model_error("invalid_model_output", "model returned invalid residuals")
                if not bool(torch.isfinite(residual).all()):
                    raise _model_error(
                        "non_finite_prediction", "model returned non-finite residuals"
                    )
                metadata["timings"][stage] = perf_counter() - started
                stage = "process_coordinates"
                started = perf_counter()
                if self.map_mode == "analytic":
                    if (not isinstance(dense, torch.Tensor)
                            or tuple(dense.shape) != (1, *output_size, 2)
                            or dense.dtype != torch.float32 or dense.device.type != "cpu"
                            or dense.layout != torch.strided):
                        raise _model_error("invalid_model_output", "invalid analytic coordinates")
                elif self.map_mode == "linear":
                    dense = expand_coordinate_residual(residual, output_size, mode="linear")
                else:
                    dense = dense_maps(residual, output_size)
                grid = dense[0].numpy().copy()
                if not np.isfinite(grid).all():
                    raise _model_error("non_finite_prediction", "coordinate expansion overflowed")
            metadata["timings"][stage] = perf_counter() - started
            return grid
        except (FlattenerError, TypeError, ValueError, RuntimeError) as error:
            if isinstance(error, RuntimeError):
                _allocation_error(error)
            metadata["failure_stage"] = stage
            metadata["timings"][stage] = (
                (metadata["timings"][stage] or 0) + perf_counter() - started
            )
            if isinstance(error, FlattenerError):
                raise
            raise _model_error("prediction_failed", str(error)) from error

    def predict_map(self, image: NDArray[np.uint8], output_size: tuple[int, int]) -> NDArray:
        """Return an unclipped float32 (H, W, 2) map for an explicit (height, width)."""
        config = RenderConfig()
        validate_image(image, config)
        output_size = _output_size(output_size, config, self.model_config)
        metadata = _new_metadata(config, timing_scope="predict_map")
        self._metadata(metadata)
        try:
            return self._predict(image, output_size, metadata)
        except FlattenerError as error:
            metadata["processing_status"] = str(error.status)
            error.metadata.update(metadata)
            raise


def load_predictor(checkpoint: str | os.PathLike[str]) -> LoadedPredictor:
    """Load a learning-0.1 checkpoint and check its .sha256 sidecar when present.

    Loads with ``weights_only`` and checks config, state keys and shapes, float32
    dtype and finite values. Global RNG, thread and determinism settings stay as
    they are. A missing sidecar is recorded as ``checksum_verified=False``.
    """
    started = perf_counter()
    try:
        path = Path(checkpoint)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        sidecar = path.with_suffix(path.suffix + ".sha256")
        verified = sidecar.exists()
        if verified:
            expected = sidecar.read_text(encoding="ascii").strip().lower()
            if expected != digest:
                raise _model_error("checkpoint_hash_mismatch", "checkpoint SHA-256 mismatch")
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or saved.get("schema_version") != "learning-0.1":
            raise _model_error("invalid_checkpoint", "unsupported checkpoint schema")
        raw_config = saved["config"]["model_config"]
        if not isinstance(raw_config, dict) or set(raw_config) != {
            f.name for f in fields(ModelConfig)
        }:
            raise _model_error("invalid_checkpoint", "model configuration fields do not match")
        raw_config = dict(raw_config)
        for name in ("input_size", "grid_size", "channels"):
            if isinstance(raw_config[name], list):
                raw_config[name] = tuple(raw_config[name])
        config = ModelConfig(**raw_config)
        # Bound CPU inference resources before building a model from
        # file-controlled dimensions.
        c1, c2, c3 = config.channels
        outputs = 2 * config.grid_size[0] * config.grid_size[1]
        parameters = (
            c1 * (3 * 9 + 1)
            + c2 * (c1 * 9 + 1)
            + c3 * (c2 * 9 + 1)
            + config.hidden * (c3 * 30 + 1)
            + outputs * (config.hidden + 1)
        )
        if parameters > 16_000_000 or max(config.input_size) > 1024:
            raise _model_error("model_resource_limit", "model exceeds CPU inference resource limit")
        state = saved["model_state"]
        if not isinstance(state, dict) or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise _model_error(
                "invalid_checkpoint", "model state must contain finite float32 tensors"
            )
        with torch.random.fork_rng(devices=[]):
            model = CoordinatePredictor(config)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
        return LoadedPredictor(
            _model=model,
            model_config=config,
            model_sha256=digest,
            model_id=f"coordinate-cnn-{MODEL_VERSION}:{digest[:12]}",
            checkpoint_path=str(path.resolve()),
            checksum_verified=verified,
            load_seconds=perf_counter() - started,
        )
    except FlattenerError:
        raise
    except OSError as error:
        raise FlattenerError(Status.IO_ERROR, "checkpoint_read_failed", str(error)) from error
    except (
        TypeError,
        ValueError,
        KeyError,
        EOFError,
        pickle.UnpicklingError,
        RuntimeError,
    ) as error:
        if isinstance(error, RuntimeError):
            _allocation_error(error)
        raise _model_error("invalid_checkpoint", str(error)) from error


def unwarp(
    image: NDArray[np.uint8],
    predictor: LoadedPredictor,
    output_size: tuple[int, int],
    *,
    config: RenderConfig | None = None,
) -> RenderResult:
    """Predict and render one photo under the raise/return/original failure policy.

    Every pixel of the requested ``output_size`` is a target, and the renderer
    applies its geometry checks. Caller arrays, torch settings and weights stay
    unchanged. ``timings.total`` covers this call without model loading.
    """
    started = perf_counter()
    candidate = RenderConfig() if config is None else config
    if (
        not isinstance(candidate, RenderConfig)
        or not isinstance(candidate.on_error, str)
        or candidate.on_error not in ("raise", "return", "original")
    ):
        raise FlattenerError(
            Status.INPUT_ERROR,
            "invalid_config",
            "invalid renderer failure policy",
            metadata={
                "processing_status": str(Status.INPUT_ERROR),
                "quality_status": "not_evaluated",
            },
        )
    metadata = _new_metadata(candidate, timing_scope="unwarp_loaded_model")
    metadata["quality_status"] = "not_evaluated"
    metadata["requested_output_size"] = _json_value(output_size)
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        metadata["source_size"] = list(image.shape[:2])
    trusted_image = None
    try:
        candidate = validate_config(candidate)
        validate_image(image, candidate)
        trusted_image = image
        if not isinstance(predictor, LoadedPredictor):
            raise FlattenerError(Status.INPUT_ERROR, "invalid_predictor", "load a predictor first")
        predictor._metadata(metadata)
        output_size = _output_size(output_size, candidate, predictor.model_config)
        metadata["requested_output_size"] = list(output_size)
        metadata["timings"]["prepare_input"] = perf_counter() - started
        grid = predictor._predict(image, output_size, metadata)
        render_started = perf_counter()
        # Handle all renderer outcomes once here, retaining inference metadata.
        render_config = RenderConfig(**{**asdict(candidate), "on_error": "return"})
        result = render(image, grid, config=render_config)
        render_metadata = result.metadata
        for stage in ("prepare_input", "process_coordinates", "render"):
            metadata["timings"][stage] = (metadata["timings"][stage] or 0) + (
                render_metadata["timings"][stage] or 0
            )
        metadata["render_call_seconds"] = perf_counter() - render_started
        metadata["validation"] = render_metadata["validation"]
        metadata["actual_output_size"] = render_metadata["actual_output_size"]
        metadata["failure_stage"] = render_metadata["failure_stage"]
        metadata["timings"]["total"] = perf_counter() - started
        if result.status != Status.SUCCESS:
            raise FlattenerError(result.status, result.reason_code, result.message)
        metadata["processing_status"] = str(result.status)
        result.metadata = metadata
        return result
    except FlattenerError as error:
        if metadata["failure_stage"] is None:
            metadata["failure_stage"] = "prepare_input"
        if metadata["timings"]["prepare_input"] is None:
            metadata["timings"]["prepare_input"] = perf_counter() - started
        metadata["timings"]["total"] = perf_counter() - started
        error.metadata.update(processing_status=str(error.status), quality_status="not_evaluated")
        return _handle_failure(error, candidate, image=trusted_image, metadata=metadata)
