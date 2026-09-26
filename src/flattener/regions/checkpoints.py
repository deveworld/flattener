"""Integrity-checked, architecture-explicit checkpoints for the region study.

SHA-256 sidecars establish integrity, not authenticity. ``weights_only`` and
resource checks are not a security sandbox; load checkpoints from trusted sources.
The frozen legacy segmentation loader is delegated to without changing its code.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from ..contracts import FlattenerError, RenderConfig, Status
from ..validation import validate_image
from .segmentation import (
    CHECKPOINT_SCHEMA as LEGACY_SCHEMA,
)
from .segmentation import (
    MODEL_VERSION as LOCAL_MODEL_VERSION,
)
from .segmentation import (
    PREPROCESS_VERSION,
    PageSegmenter,
    SegmenterConfig,
    load_segmenter,
    preprocess_image,
)
from .segmentation import (
    read_checkpoint as read_legacy_checkpoint,
)

STUDY_SCHEMA = "region-study-0.2"
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_MODEL_PARAMETERS = 1_000_000
_FIELDS = {
    "schema_version",
    "config",
    "model_spec",
    "model_state",
    "optimizer_state",
    "epoch",
    "history",
    "initial_validation",
    "best_validation",
    "best_epoch",
    "manifest_sha256",
    "dataset_counts",
    "provenance",
    "torch_rng_state",
    "shuffle_rng_state",
}


def _error(message, reason="invalid_checkpoint"):
    return FlattenerError(Status.MODEL_ERROR, reason, message)


def _allocation_error(error):
    if isinstance(error, RuntimeError) and (
        "out of memory" in str(error).lower() or "can't allocate memory" in str(error).lower()
    ):
        raise MemoryError(str(error)) from error


def _architecture(architecture):
    if not isinstance(architecture, str) or architecture not in ("local", "context"):
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_architecture", "architecture must be 'local' or 'context'"
        )
    return architecture


def create_region_model(architecture):
    """Create a fixed-default study model; initialization normally consumes RNG."""
    if _architecture(architecture) == "local":
        return PageSegmenter()
    from .context_segmentation import ContextPageSegmenter

    return ContextPageSegmenter()


def model_spec(architecture):
    """Return the fixed architecture contract without allocating model weights."""
    if _architecture(architecture) == "local":
        return {
            "architecture": "local",
            "model_version": LOCAL_MODEL_VERSION,
            "model_config": asdict(SegmenterConfig()),
        }
    from .context_segmentation import MODEL_VERSION, ContextSegmenterConfig

    return {
        "architecture": "context",
        "model_version": MODEL_VERSION,
        "model_config": asdict(ContextSegmenterConfig()),
    }


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _number(value, low=0, high=math.inf):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and low <= value <= high
    except OverflowError:
        return False


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _metadata(value, depth=0):
    if depth > 24:
        raise _error("metadata nesting exceeds the supported limit")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, dict) and len(value) <= 100_000:
        if any(type(key) is not str for key in value):
            raise _error("metadata dictionaries require string keys")
        for item in value.values():
            _metadata(item, depth + 1)
        return
    if isinstance(value, (tuple, list)) and len(value) <= 100_000:
        for item in value:
            _metadata(item, depth + 1)
        return
    raise _error("metadata must contain finite built-in JSON-compatible values")


def _metrics(value):
    if not isinstance(value, dict) or not _number(value.get("loss")):
        raise _error("validation metrics require a finite nonnegative loss")
    _metadata(value)


def _tensor(value, *, dtype=torch.float32, shape=None):
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and value.dtype == dtype
        and (shape is None or tuple(value.shape) == tuple(shape))
        and bool(torch.isfinite(value).all())
    )


def _rng(value):
    if not _tensor(value, dtype=torch.uint8) or value.ndim != 1 or value.numel() > 16384:
        raise _error("RNG state must be a bounded one-dimensional CPU byte tensor")
    # Validate with an independent generator, never the process-global generator.
    torch.Generator(device="cpu").set_state(value)


def _optimizer(saved, model, config):
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise _error("optimizer state requires state and param_groups")
    state, groups = saved["state"], saved["param_groups"]
    if not isinstance(state, dict) or not isinstance(groups, list) or not groups:
        raise _error("invalid optimizer state dictionaries")
    parameters = list(model.parameters())
    indices = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise _error("optimizer parameter groups require an index list")
        betas = group.get("betas")
        if (
            not _number(group.get("lr"), 0, 1)
            or group["lr"] != config["learning_rate"]
            or not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(not _number(beta, 0, 1) or beta == 1 for beta in betas)
            or not _number(group.get("eps"))
            or group["eps"] == 0
            or not _number(group.get("weight_decay"))
            or type(group.get("amsgrad")) is not bool
        ):
            raise _error("invalid Adam hyperparameters or learning-rate configuration")
        for flag in ("maximize", "capturable", "differentiable", "decoupled_weight_decay"):
            if flag in group and type(group[flag]) is not bool:
                raise _error("Adam flags must be boolean")
        for flag in ("foreach", "fused"):
            if flag in group and group[flag] is not None and type(group[flag]) is not bool:
                raise _error("Adam backend flags must be boolean or None")
        current = group["params"]
        if any(not _integer(item, 0, len(parameters) - 1) for item in current):
            raise _error("optimizer parameter indices are invalid")
        indices.extend(current)
        _metadata({key: value for key, value in group.items() if key != "params"})
    if sorted(indices) != list(range(len(parameters))):
        raise _error("optimizer groups must reference each model parameter exactly once")
    for index, values in state.items():
        if not _integer(index, 0, len(parameters) - 1) or not isinstance(values, dict):
            raise _error("invalid optimizer per-parameter state")
        if set(values) not in (
            {"step", "exp_avg", "exp_avg_sq"},
            {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
        ):
            raise _error("optimizer state must be an Adam moment state")
        if not _tensor(values["step"], shape=()) or float(values["step"]) < 0:
            raise _error("invalid Adam step tensor")
        for name in set(values) - {"step"}:
            if not _tensor(values[name], shape=parameters[index].shape):
                raise _error("optimizer moment shape, dtype, or values are invalid")


def _validate_study(saved):
    if (
        not isinstance(saved, dict)
        or set(saved) != _FIELDS
        or saved["schema_version"] != STUDY_SCHEMA
    ):
        raise _error("unsupported or incomplete region-study checkpoint schema")
    config = saved["config"]
    if not isinstance(config, dict) or set(config) != {
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "threads",
        "architecture",
    }:
        raise _error("study training configuration fields do not match")
    for name, low, high in (
        ("seed", 0, 2**63 - 1),
        ("epochs", 1, 100_000),
        ("batch_size", 1, 1024),
        ("threads", 1, 128),
    ):
        if not _integer(config[name], low, high):
            raise _error(f"invalid study configuration: {name}")
    if not _number(config["learning_rate"], 0, 1) or config["learning_rate"] == 0:
        raise _error("invalid study learning rate")
    if config["architecture"] not in ("local", "context") or not isinstance(
        config["architecture"], str
    ):
        raise _error("invalid checkpoint architecture")
    spec = model_spec(config["architecture"])
    # Check types recursively before equality, avoiding ambiguous tensor truth values.
    _metadata(saved["model_spec"])
    if saved["model_spec"] != spec:
        raise _error("checkpoint model specification differs from fixed study architecture")
    epoch = saved["epoch"]
    if not _integer(epoch, 0, config["epochs"]) or not _integer(saved["best_epoch"], 0, epoch):
        raise _error("invalid checkpoint epoch metadata")
    if not isinstance(saved["history"], list) or len(saved["history"]) != epoch:
        raise _error("history must contain one record per completed epoch")
    _metrics(saved["initial_validation"])
    _metrics(saved["best_validation"])
    metrics = [saved["initial_validation"]]
    for expected, record in enumerate(saved["history"], 1):
        if (
            not isinstance(record, dict)
            or type(record.get("epoch")) is not int
            or record["epoch"] != expected
            or not _number(record.get("train_loss"))
        ):
            raise _error("invalid chronological training history")
        _metadata(record)
        _metrics(record.get("validation"))
        metrics.append(record["validation"])
    best_epoch = min(range(len(metrics)), key=lambda index: metrics[index]["loss"])
    if saved["best_epoch"] != best_epoch or saved["best_validation"] != metrics[best_epoch]:
        raise _error("best checkpoint metadata violates minimum-loss earliest-epoch selection")
    if not _digest(saved["manifest_sha256"]):
        raise _error("invalid dataset manifest SHA-256")
    counts = saved["dataset_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != {"train", "validation", "test"}
        or any(not _integer(value, 1, 1_000_000) for value in counts.values())
    ):
        raise _error("invalid dataset split counts")
    if not isinstance(saved["provenance"], dict):
        raise _error("checkpoint provenance must be a dictionary")
    _metadata(saved["provenance"])
    _rng(saved["torch_rng_state"])
    _rng(saved["shuffle_rng_state"])
    state = saved["model_state"]
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise _error("model state must be a string-keyed dictionary")
    if any(not _tensor(value) for value in state.values()):
        raise _error("model state must contain finite strided CPU float32 tensors")
    if sum(value.numel() for value in state.values()) > MAX_MODEL_PARAMETERS:
        raise _error("model exceeds one million state elements", "model_resource_limit")
    with torch.random.fork_rng(devices=[]):
        model = create_region_model(config["architecture"])
    if sum(value.numel() for value in model.parameters()) > MAX_MODEL_PARAMETERS:
        raise _error("model exceeds one million parameters", "model_resource_limit")
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[key].shape != expected[key].shape for key in state):
        raise _error("model state keys or shapes differ from the declared architecture")
    _optimizer(saved["optimizer_state"], model, config)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def _read(path):
    try:
        path = Path(path)
        with path.open("rb") as stream:
            payload = stream.read(MAX_CHECKPOINT_BYTES + 1)
        if len(payload) > MAX_CHECKPOINT_BYTES:
            raise _error("region checkpoint exceeds 64 MiB", "checkpoint_size_limit")
        with path.with_suffix(path.suffix + ".sha256").open("r", encoding="ascii") as stream:
            expected = stream.read(257).strip().lower()
        digest = hashlib.sha256(payload).hexdigest()
        if not _digest(expected) or expected != digest:
            raise _error("checkpoint differs from SHA-256 sidecar", "checkpoint_hash_mismatch")
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(saved, dict):
            raise _error("checkpoint root must be a dictionary")
        return saved, digest
    except FlattenerError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        _allocation_error(error)
        raise _error(f"Cannot read region checkpoint: {error}") from error


def read_region_checkpoint(path: str | os.PathLike[str]) -> dict:
    """Validate study training/inference state, or delegate the legacy schema."""
    saved, _ = _read(path)
    if saved.get("schema_version") == LEGACY_SCHEMA:
        return read_legacy_checkpoint(path)
    try:
        _validate_study(saved)
        return saved
    except FlattenerError:
        raise
    except (ValueError, TypeError, RuntimeError, KeyError, OverflowError) as error:
        _allocation_error(error)
        raise _error(f"Invalid region checkpoint: {error}") from error


@dataclass(slots=True)
class LoadedRegionModel:
    model: torch.nn.Module
    config: SegmenterConfig
    metadata: dict
    load_seconds: float

    @property
    def model_sha256(self):
        return self.metadata["checkpoint_sha256"]

    @property
    def model_id(self):
        return f"{self.metadata['model_version']}:{self.model_sha256[:12]}"

    def predict_probability(self, image):
        """Full-extent finite float32 probabilities; no cropping or thresholding."""
        validate_image(image, RenderConfig())
        try:
            self.model.eval()
            with torch.inference_mode():
                # Both architectures use the frozen numerical RGB preprocessing;
                # only the spatial input size is relevant to that operation.
                preview_config = SegmenterConfig(input_size=self.config.input_size)
                logits = self.model(preprocess_image(image, preview_config)[None])
                if not _tensor(logits, shape=(1, 1, *self.config.input_size)):
                    raise _error("model returned invalid logits", "invalid_model_output")
                probability = logits.sigmoid()[0, 0].numpy().copy()
            return probability
        except FlattenerError:
            raise
        except (TypeError, ValueError, RuntimeError) as error:
            _allocation_error(error)
            raise _error(f"Region prediction failed: {error}", "prediction_failed") from error


def load_region_model(path: str | os.PathLike[str]):
    """Load once for repeated CPU inference, preserving caller RNG/threads."""
    started = perf_counter()
    saved, digest = _read(path)
    if saved.get("schema_version") == LEGACY_SCHEMA:
        return load_segmenter(path)
    try:
        model = _validate_study(saved)
        return LoadedRegionModel(
            model,
            model.config,
            {
                "checkpoint_sha256": digest,
                "checkpoint_path": str(Path(path).resolve()),
                "manifest_sha256": saved["manifest_sha256"],
                "epoch": saved["epoch"],
                "model_version": saved["model_spec"]["model_version"],
                "model_spec": saved["model_spec"],
                "preprocess_version": PREPROCESS_VERSION,
                "best_validation": saved["best_validation"],
                "probability_grid": "full original image extent; no threshold or crop applied",
            },
            perf_counter() - started,
        )
    except FlattenerError:
        raise
    except (ValueError, TypeError, RuntimeError, KeyError, OverflowError) as error:
        _allocation_error(error)
        raise _error(f"Invalid region checkpoint: {error}") from error
