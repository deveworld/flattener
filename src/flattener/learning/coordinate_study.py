"""Coordinate-model training study, strict checkpoints and its CLI.

The default device is CUDA; choose CPU or XPU explicitly. Training uses only the
requested device. Time and epoch limits stop at the last complete epoch and
leave the saved configuration unchanged. Resume works at epoch boundaries with
the same data, code and runtime; resumed accelerator runs can differ bitwise.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import pickle
import platform
import re
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import PIL
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..contracts import FlattenerError, Status
from .augmentation import (
    AUGMENTATION_PROTOCOL,
    AUGMENTATION_VERSION,
    SEMANTIC_PAGE_PROTOCOL,
    augment_training_batch,
    prepare_semantic_page_labels,
    transform_semantic_page_labels,
)
from .inference import LoadedPredictor
from .model import MODEL_VERSION as GLOBAL_VERSION
from .model import PREPROCESS_VERSION, ModelConfig
from .scene_data import (
    CachedSceneCoordinateDataset,
    SceneCoordinateDataset,
    cache_scene_datasets,
    verify_scene_caches,
)
from .spatial import (
    MODEL_VERSION,
    ORIENTED_SURFACE_MODEL_VERSION,
    PAGE_MODEL_VERSION,
    PROJECTIVE_MODEL_VERSION,
    SURFACE_MODEL_VERSION,
    expand_coordinate_residual,
    make_coordinate_model,
    parameter_count,
)
from .spatial import _config as _bounded_model_config

LEGACY_CHECKPOINT_SCHEMA = "coordinate-study-0.1"
REGULARIZED_CHECKPOINT_SCHEMA = "coordinate-study-0.2"
PAGE_CHECKPOINT_SCHEMA = "coordinate-study-0.3"
PAPER_CHECKPOINT_SCHEMA = "coordinate-study-0.4"
ANALYTIC_CHECKPOINT_SCHEMA = "coordinate-study-0.5"
PAGE_RELATIVE_CHECKPOINT_SCHEMA = "coordinate-study-0.6"
ORIENTATION_CHECKPOINT_SCHEMA = "coordinate-study-0.7"
OPTIMIZED_CHECKPOINT_SCHEMA = "coordinate-study-0.8"
DIVERSITY_CHECKPOINT_SCHEMA = "coordinate-study-0.9"
SEMANTIC_CHECKPOINT_SCHEMA = "coordinate-study-0.10"
XPU_CHECKPOINT_SCHEMA = "coordinate-study-0.11"
CHECKPOINT_SCHEMA = "coordinate-study-0.12"
_SCHEMAS = (LEGACY_CHECKPOINT_SCHEMA, REGULARIZED_CHECKPOINT_SCHEMA,
            PAGE_CHECKPOINT_SCHEMA, PAPER_CHECKPOINT_SCHEMA,
            ANALYTIC_CHECKPOINT_SCHEMA, PAGE_RELATIVE_CHECKPOINT_SCHEMA,
            ORIENTATION_CHECKPOINT_SCHEMA, OPTIMIZED_CHECKPOINT_SCHEMA,
            DIVERSITY_CHECKPOINT_SCHEMA, SEMANTIC_CHECKPOINT_SCHEMA,
            XPU_CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA)
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}")
_CUDA = re.compile(r"cuda(?::(0|[1-9][0-9]{0,3}))?")
_XPU = re.compile(r"xpu(?::(0|[1-9][0-9]{0,3}))?")
_PORTABLE_CHECKPOINTS = os.name == "nt"


def _error(reason, message, status=Status.INPUT_ERROR):
    return FlattenerError(status, reason, message)


@dataclass(frozen=True, slots=True)
class StudyConfig:
    architecture: str = "global"
    device: str = "cuda"
    seed: int = 20260925
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.001
    model_config: ModelConfig = field(default_factory=ModelConfig)
    output_size: tuple[int, int] = (192, 144)
    topology_weight: float = 0.0
    curvature_weight: float = 0.0
    gradient_clip_norm: float = 0.0
    page_weight: float = 0.0
    boundary_weight: float = 0.0
    differential_weight: float = 0.0
    page_relative_weight: float = 0.0
    orientation_weight: float = 0.0
    direction_weight: float = 0.0
    corner_weight: float = 0.0
    direction_training: str = "predicted"
    rotation_probability: float = 0.0
    background_probability: float = 0.0
    rotation_mode: str = "batch"
    include_paper_augmentation: bool = False
    initial_checkpoint_sha256: str | None = None
    map_mode: str = "legacy"
    topology_mode: str = "absolute"
    initialization_mode: str = "full"
    initial_model_config: ModelConfig | None = None


def _config(config):
    if not isinstance(config, StudyConfig):
        raise _error("invalid_config", "config must be StudyConfig")
    if (not isinstance(config.device, str)
            or config.device != "cpu" and not (
                _CUDA.fullmatch(config.device) or _XPU.fullmatch(config.device))):
        reason = ("xpu_required" if isinstance(config.device, str)
                  and config.device.startswith("xpu") else "cuda_required")
        raise _error(reason, "device must be explicit cpu, cuda[:N], or xpu[:N]")
    if (not isinstance(config.architecture, str)
            or config.architecture not in (
                "global", "spatial", "page", "projective", "surface", "oriented_surface")):
        raise _error("invalid_config", "unsupported coordinate architecture")
    for name, choices in (("map_mode", ("legacy", "linear", "analytic")),
                          ("topology_mode", ("absolute", "relative")),
                          ("initialization_mode", ("full", "backbone", "resolution")),
                          ("rotation_mode", ("batch", "per_image_fit")),
                          ("direction_training", ("predicted", "teacher"))):
        if not isinstance(getattr(config, name), str) or getattr(config, name) not in choices:
            raise _error("invalid_config", f"invalid {name}")
    analytic = config.architecture in ("projective", "surface", "oriented_surface")
    if (config.map_mode == "analytic") != analytic:
        raise _error("invalid_config", "analytic maps require projective or surface architecture")
    if config.initialization_mode == "backbone" and not analytic:
        raise _error("invalid_config", "backbone initialization requires projective or surface")
    for name, limit in (("seed", 2**63 - 1), ("epochs", 100_000), ("batch_size", 1024)):
        value = getattr(config, name)
        if type(value) is not int or not (0 if name == "seed" else 1) <= value <= limit:
            raise _error("invalid_config", f"invalid {name}")
    if (type(config.learning_rate) not in (int, float)
            or not math.isfinite(config.learning_rate) or not 0 < config.learning_rate <= 1):
        raise _error("invalid_config", "learning_rate must be finite in (0, 1]")
    for name in ("topology_weight", "curvature_weight", "gradient_clip_norm",
                 "page_weight", "boundary_weight", "differential_weight", "page_relative_weight",
                 "orientation_weight", "direction_weight", "corner_weight"):
        if not _finite_number(getattr(config, name)):
            raise _error("invalid_config", f"{name} must be a finite nonnegative number")
    for name in ("rotation_probability", "background_probability"):
        if not _finite_number(getattr(config, name)) or getattr(config, name) > 1:
            raise _error("invalid_config", f"{name} must be finite in [0, 1]")
    if config.page_weight and config.architecture not in ("page", "oriented_surface"):
        raise _error("invalid_config", "page_weight requires page or oriented_surface")
    if config.architecture == "oriented_surface":
        if not all(getattr(config, key) > 0 for key in (
            "page_weight", "direction_weight", "corner_weight"
        )):
            raise _error("invalid_config", "oriented_surface requires positive page, direction "
                         "and corner supervision weights")
        if config.rotation_probability and config.rotation_mode != "per_image_fit":
            raise _error("invalid_config", "oriented_surface rotation requires per_image_fit")
    elif config.direction_weight or config.corner_weight:
        raise _error("invalid_config", "direction/corner supervision requires oriented_surface")
    if config.direction_training == "teacher" and config.architecture != "oriented_surface":
        raise _error("invalid_config", "teacher direction requires oriented_surface")
    if type(config.include_paper_augmentation) is not bool:
        raise _error("invalid_config", "include_paper_augmentation must be boolean")
    if config.initial_checkpoint_sha256 is not None and (
        not isinstance(config.initial_checkpoint_sha256, str)
        or not _HEX.fullmatch(config.initial_checkpoint_sha256)
    ):
        raise _error("invalid_config", "initial_checkpoint_sha256 must be a lowercase SHA-256")
    if not isinstance(config.model_config, ModelConfig):
        raise _error("invalid_config", "model_config must be ModelConfig")
    if config.initial_model_config is not None:
        if (config.initialization_mode != "resolution"
                or not isinstance(config.initial_model_config, ModelConfig)
                or any(getattr(config.initial_model_config, item.name)
                       != getattr(config.model_config, item.name)
                       for item in fields(ModelConfig) if item.name != "input_size")):
            raise _error("invalid_config", "resolution source may differ only in input_size")
    try:
        _bounded_model_config(config.model_config, config.architecture)
        if config.initial_model_config is not None:
            _bounded_model_config(config.initial_model_config, config.architecture)
        if (not isinstance(config.output_size, tuple) or len(config.output_size) != 2
                or any(type(v) is not int or not 2 <= v <= 512 for v in config.output_size)
                or any(o < g for o, g in zip(config.output_size, config.model_config.grid_size))):
            raise ValueError("output_size must be bounded by 512 and not shrink the model grid")
    except (TypeError, ValueError) as error:
        raise _error("invalid_config", str(error)) from error
    return config


def _require_cuda_device(device="cuda"):
    """Return the CUDA device, raising before any data, output or optimizer work."""
    if not isinstance(device, str) or not _CUDA.fullmatch(device):
        raise _error("cuda_required", "this CUDA request cannot use a CPU fallback")
    if not torch.cuda.is_available():
        raise _error("cuda_unavailable", "CUDA is unavailable; no training was started")
    try:
        selected = torch.device(device)
        index = torch.cuda.current_device() if selected.index is None else selected.index
        if index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device does not exist")
        return torch.device("cuda", index)
    except (ValueError, RuntimeError) as error:
        raise _error("cuda_unavailable", str(error)) from error


def _resolve_device(device="cuda"):
    """Resolve the requested backend; CPU requests skip accelerator APIs."""
    if device == "cpu":
        return torch.device("cpu")
    if isinstance(device, str) and device.startswith("xpu"):
        return _require_xpu_device(device)
    return _require_cuda_device(device)


def _require_xpu_device(device="xpu"):
    """Return the XPU device, raising before any filesystem or training work."""
    if not isinstance(device, str) or not _XPU.fullmatch(device):
        raise _error("xpu_required", "this XPU request cannot use a CPU or CUDA fallback")
    backend = getattr(torch, "xpu", None)
    try:
        if backend is None or not backend.is_available():
            raise ValueError("XPU is unavailable; no training was started")
        selected = torch.device(device)
        index = backend.current_device() if selected.index is None else selected.index
        if index >= backend.device_count():
            raise ValueError("requested XPU device does not exist")
        return torch.device("xpu", index)
    except (ValueError, RuntimeError, AssertionError) as error:
        raise _error("xpu_unavailable", str(error)) from error


@contextmanager
def _training_rng(device, seed):
    """Seed and isolate the CPU and selected accelerator RNGs and current device."""
    backend = getattr(torch, device.type) if device.type in ("cuda", "xpu") else None
    with (torch.device("cpu"), backend.device(device) if backend else nullcontext(),
          torch.random.fork_rng(devices=[device.index] if backend else [],
                                device_type=device.type if backend else "cuda")):
        torch.random.default_generator.manual_seed(seed)
        if backend:
            backend.manual_seed(seed)
        yield


def _run_controls(config, stop_after_epoch, max_seconds):
    if stop_after_epoch is not None and (
        type(stop_after_epoch) is not int or not 0 <= stop_after_epoch <= config.epochs
    ):
        raise _error("invalid_run_control", "stop_after_epoch must be in [0, configured epochs]")
    if max_seconds is not None:
        try:
            valid = (type(max_seconds) in (int, float)
                     and math.isfinite(max_seconds) and max_seconds > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise _error("invalid_run_control", "max_seconds must be finite and positive")


class _DeadlineReached(Exception):
    """Stop at a batch boundary; the unfinished epoch is discarded."""


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise _DeadlineReached


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(schema=CHECKPOINT_SCHEMA, *, include_hashes=True):
    if schema not in _SCHEMAS:
        raise ValueError("unsupported coordinate study provenance schema")
    package = Path(__file__).resolve().parents[1]
    lock = package.parent.parent / "uv.lock"
    result = {
        "source_code_sha256": ({p.relative_to(package).as_posix(): _sha256(p)
                                for p in sorted(package.rglob("*.py"))}
                               if include_hashes else {}),
        "uv_lock_sha256": _sha256(lock) if include_hashes and lock.is_file() else None,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": str(np.__version__), "pillow": str(PIL.__version__),
                     "torch_cuda": torch.version.cuda},
        "torch_settings": {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_version": torch.backends.cudnn.version(),
            "threads": torch.get_num_threads(),
        },
        "coordinate_loss": "valid GT SmoothL1 beta=1 source pixel; mean per coordinate",
        "selection": "earliest minimum validation per-image mean source-pixel endpoint error",
        "resume_scope": "same data/code/runtime at epoch boundaries; CUDA not bit-exact guaranteed",
    }
    if schema != LEGACY_CHECKPOINT_SCHEMA:
        result["loss_protocol"] = {
            "version": "coordinate-objective-0.2",
            "topology": "four-corner normalized det; fixed four-GT-valid cell mask; "
                        "per-image mean plus maximum squared hinge, then eligible-image mean",
            "minimum_normalized_jacobian": 0.001,
            "curvature": "coarse residual source-pixel xy second differences; "
                         "SmoothL1 beta=1 pixel; mean over available axes and coordinates",
            "gradient_clip": "optional global parameter-gradient L2 norm before Adam step; 0 off",
        }
    if schema in _SCHEMAS[2:]:
        result["loss_protocol"].update(
            version="coordinate-objective-0.3",
            page="original source-mask area resize; per-image class-balanced soft-label BCE "
                 "plus soft Dice with numerator/denominator smoothing 1; mean over images",
            boundary="outer 8 output pixels intersect fixed GT valid mask; "
                     "SmoothL1 beta=1 source pixel; mean per eligible coordinate",
            differential="bilinear align_corners=False dense predicted/GT samples at model grid; "
                         "GT invalid samples zero-filled; interpolated validity >= 1-1e-6; "
                         "source-pixel first/second differences on all-valid stencils; "
                         "SmoothL1 beta=1 pixel mean over eligible axes and coordinates",
        )
    if schema in _SCHEMAS[4:]:
        result["loss_protocol"].update(
            version="coordinate-objective-0.5",
            relative_topology="four-corner normalized predicted det / GT det clamped at 1e-4; "
                              "fixed four-GT-valid cells with all four GT det > 0; "
                              "squared hinge at 0.1; per-image mean plus maximum, "
                              "then eligible-image mean",
        )
        result["map_protocol"] = {
            "legacy": "bilinear residual align_corners=False border extension plus dense identity",
            "linear": "separable nearest-two-center linear residual extrapolation plus identity",
            "analytic": "model forward_dense evaluates its parameterized map at dense centers",
        }
        result["initialization_protocol"] = {
            "full": "same architecture and ModelConfig; all model weights; "
                    "fresh optimizer/RNG/history",
            "backbone": "global source to projective/surface with identical ModelConfig; "
                        "encoder.* and head.1.* only; zero head.3; fresh optimizer/RNG/history",
        }
    if schema in _SCHEMAS[5:]:
        result["loss_protocol"].update(
            version="coordinate-objective-0.6",
            page_relative="additional per-image fixed-GT-valid xy SmoothL1 beta=1 source-pixel "
                          "mean times (photo diagonal / max(GT source-mask bbox diagonal, "
                          "0.05 * photo diagonal) - 1), then image mean; original-photo "
                          "physical-pixel bbox widths max-min+1, both >= 2; "
                          "photo diagonal hypot(width,height); empty GT/masks rejected; "
                          "0 weight bypasses the objective and source-mask access",
        )
    if schema in _SCHEMAS[6:]:
        result["loss_protocol"].update(
            version="coordinate-objective-0.7",
            orientation="bilinear align_corners=False maps at model grid; fixed interpolated "
                        "GT validity >= 1-1e-6 at both endpoints; source-pixel first differences; "
                        "GT norm >= 1 pixel; 1 - dot/(max(predicted norm,1 pixel)*GT norm), "
                        "cosine ratio clamped [-1,1] for rounding only; all two-axis eligible "
                        "stencils per-image mean, then eligible-image mean; no eligible => 0; "
                        "exact 180-degree disagreement may be stationary, never a sole objective",
        )
        result["augmentation_protocol"] = {
            "quarter_turn": "training only: one CPU-global torch.rand Bernoulli per batch, "
                            "then CPU-global torch.randint(1,4) if selected; shared k across "
                            "batch; 0 probability bypasses augmentation and RNG; existing "
                            "checkpoint torch_rng_state preserves sampling at epoch boundaries",
            "image": "rot90 then bilinear align_corners=False to original input size; "
                     "validated [0,1] input; only <= 8 float32 eps interpolation range "
                     "overshoot corrected to [0,1]",
            "target": "source normalized xy only: k1=(y,-x), k2=(-x,-y), k3=(-y,x); "
                      "flat-output domain and fixed valid mask unchanged; no coordinate clamp",
            "source_mask": "if present, rot90 then pixel-center nearest-exact resize to "
                           "original photo size; bool",
        }
    if schema in _SCHEMAS[7:]:
        result["loss_protocol"].update(
            version="coordinate-objective-0.8",
            reduction="fixed-shape zero-fill before masked arithmetic; GT-only denominators; "
                      "FP32 loss/norm, FP64 validation sum; mathematically equivalent to "
                      "0.7 but reduction order and final FP32 bits/gradients may differ",
            admission="batch image range and fixed-GT finite checks once; prediction/raw head, "
                      "relative Jacobians and total loss finite before backward; "
                      "all gradients finite before Adam; "
                      "FP32 Adam foreach=False fused=False",
        )
    if schema in _SCHEMAS[8:]:
        result["initialization_protocol"]["resolution"] = (
            "same architecture; ModelConfig differs only in input_size; all model weights; "
            "fresh optimizer/RNG/history; source initial_model_config and checkpoint SHA "
            "bound explicitly, target model_config retained")
        result["augmentation_protocol"].update(
            version="coordinate-augmentation-0.9",
            rotation_mode="batch preserves legacy whole-batch quarter-turn behavior; "
                          "per_image_fit uses independent CPU-global Bernoulli and nonzero "
                          "quarter turns in original-photo physical pixel aspect ratio, "
                          "uniformly fitted without cropping; target xy transformed, "
                          "flat-output valid mask unchanged",
            background="independent per-image CPU-global Bernoulli; procedural texture "
                       "replaces background outside dilated original source-page mask, "
                       "before rotation; no external texture data or held-out sampling; "
                       "original source_mask required whenever probability > 0",
            ordering="per_image_fit delegates background and rotation together; batch mode "
                     "applies background with rotation disabled, then legacy batch rotation; "
                     "zero probabilities bypass the new helper, copying and RNG",
            fit_module_version=AUGMENTATION_VERSION,
            fit_module_protocol=copy.deepcopy(AUGMENTATION_PROTOCOL),
        )
    if schema in _SCHEMAS[9:]:
        result["loss_protocol"].update(
            version="coordinate-objective-0.10",
            direction="four clockwise physical document-x quarter-turn classes; "
                      "cross entropy averaged over fixed GT eligible images; "
                      "ambiguous or degenerate directions excluded by GT only",
            corners="analytic map at semantic flat boundary TL/TR/BR/BL; "
                    "source-pixel error divided by max(GT visible-corner bbox diagonal, "
                    "0.05 * photo diagonal); SmoothL1 beta=0.01; "
                    "fixed GT corner mask, per-image coordinate mean then eligible-image mean",
            semantic_page=copy.deepcopy(SEMANTIC_PAGE_PROTOCOL),
        )
        result["map_protocol"]["oriented_surface"] = (
            "argmax clockwise physical quarter turn (earliest tie), bounded residual angle, "
            "positive projective plus bounded Bernstein residual; source-coordinate map; "
            "photo_size is required at every forward; no canonical crop")
        result["initialization_protocol"]["oriented_backbone"] = (
            "global source with identical ModelConfig; encoder.* and head.1.* only; "
            "fresh orientation/page heads and zero geometry head; fresh optimizer/RNG/history")
    if schema in _SCHEMAS[10:]:
        result["versions"]["torch_xpu"] = getattr(torch.version, "xpu", None)
        result["resume_scope"] = (
            "same data/code/runtime/device at epoch boundaries; "
            "accelerator not bit-exact guaranteed")
        result["execution_protocol"] = {
            "version": "explicit-device-0.11",
            "devices": "explicit cpu, cuda[:N], or xpu[:N]; no automatic backend fallback",
            "precision": "FP32 model, loss and Adam; foreach=False fused=False; "
                         "FP64 validation sum",
            "rng": "CPU plus selected accelerator only; CUDA/XPU states stored separately; "
                   "forked RNG and current device restored on exit",
            "transfer": "CUDA/XPU pinned host batches and nonblocking transfers; CPU unchanged",
        }
    if schema == CHECKPOINT_SCHEMA:
        result["direction_training_protocol"] = {
            "version": "training-direction-0.12",
            "predicted": "unchanged hard argmax geometry in training, validation and inference",
            "teacher": "oriented_surface training only: post-augmentation fixed-GT eligible "
                       "quarter-turn classes override geometry; missing GT falls back to argmax; "
                       "predicted orientation logits retain cross entropy supervision",
            "admission": "same-device int64 classes and bool GT mask; eligible classes in [0,3] "
                         "checked before safe indexing; invalid labels reject before backward; "
                         "no primary GT-mask changes or coordinate clipping",
            "evaluation": "validation/checkpoint selection/public inference always use predicted "
                          "directions without GT overrides; teacher forcing has a train/inference "
                          "gap and does not certify prediction quality",
        }
    return result


def _new_model(config):
    with torch.device("cpu"):
        return make_coordinate_model(config.architecture, config.model_config).float()


def _model_spec(config, model=None):
    if model is None:
        with torch.random.fork_rng(devices=[]):
            model = _new_model(_config(config))
    versions = {"global": GLOBAL_VERSION, "spatial": MODEL_VERSION, "page": PAGE_MODEL_VERSION,
                "projective": PROJECTIVE_MODEL_VERSION, "surface": SURFACE_MODEL_VERSION,
                "oriented_surface": ORIENTED_SURFACE_MODEL_VERSION}
    return {"architecture": config.architecture,
            "version": versions[config.architecture],
            "preprocess_version": PREPROCESS_VERSION, "parameter_count": parameter_count(model)}


def _adam_group(config, parameter_count):
    """Return the canonical Adam parameter group for ``config``."""
    return {"lr": config.learning_rate, "betas": (0.9, 0.999), "eps": 1e-8,
            "weight_decay": 0, "amsgrad": False, "maximize": False, "foreach": False,
            "capturable": False, "differentiable": False, "fused": False,
            "decoupled_weight_decay": False, "params": list(range(parameter_count))}


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _tensor(value, shape, dtype=torch.float32):
    return (isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.layout == torch.strided and value.dtype == dtype
            and tuple(value.shape) == tuple(shape) and bool(torch.isfinite(value).all()))


def _validate_saved(saved):
    """Validate a checkpoint payload on CPU without building a model or optimizer."""
    required = {"schema_version", "checkpoint_kind", "config", "model_spec", "model_state",
                "optimizer_state", "epoch", "history", "best_epoch", "manifest_sha256",
                "initial_validation_epe_pixels", "best_validation_epe_pixels", "provenance",
                "dataset_output_size", "dataset_photo_size", "dataset_counts", "torch_rng_state",
                "shuffle_rng_state", "cuda_rng_state", "device_info", "best_checkpoint_sha256"}
    try:
        if isinstance(saved, dict) and saved.get("schema_version") in _SCHEMAS[10:]:
            required.add("xpu_rng_state")
        if not isinstance(saved, dict) or saved.keys() != required:
            raise ValueError("checkpoint fields are missing or unexpected")
        schema = saved["schema_version"]
        if (schema not in _SCHEMAS
                or saved["checkpoint_kind"] not in ("best", "last")):
            raise ValueError("unsupported checkpoint schema or kind")
        raw = saved["config"]
        config_fields = {f.name for f in fields(StudyConfig)}
        if schema != CHECKPOINT_SCHEMA:
            config_fields -= {"direction_training"}
        if schema not in _SCHEMAS[9:]:
            config_fields -= {"direction_weight", "corner_weight"}
        if schema not in _SCHEMAS[8:]:
            config_fields -= {"background_probability", "rotation_mode", "initial_model_config"}
        if schema not in _SCHEMAS[6:]:
            config_fields -= {"orientation_weight", "rotation_probability"}
        if schema not in _SCHEMAS[5:]:
            config_fields -= {"page_relative_weight"}
        if schema not in _SCHEMAS[4:]:
            config_fields -= {"map_mode", "topology_mode", "initialization_mode"}
        if schema not in _SCHEMAS[3:]:
            config_fields -= {"include_paper_augmentation", "initial_checkpoint_sha256"}
        if schema not in _SCHEMAS[2:]:
            config_fields -= {"page_weight", "boundary_weight", "differential_weight"}
        if schema == LEGACY_CHECKPOINT_SCHEMA:
            config_fields -= {"topology_weight", "curvature_weight", "gradient_clip_norm"}
        if not isinstance(raw, dict) or raw.keys() != config_fields:
            raise ValueError("invalid configuration fields")
        converted = {**raw, "model_config": ModelConfig(**raw["model_config"])}
        if raw.get("initial_model_config") is not None:
            if (not isinstance(raw["initial_model_config"], dict)
                    or raw["initial_model_config"].keys() != {
                        item.name for item in fields(ModelConfig)}):
                raise ValueError("source ModelConfig fields are missing or unexpected")
            converted["initial_model_config"] = ModelConfig(**raw["initial_model_config"])
        config = _config(StudyConfig(**converted))
        if schema not in _SCHEMAS[9:] and config.architecture == "oriented_surface":
            raise ValueError("oriented_surface requires coordinate-study-0.10")
        if schema not in _SCHEMAS[10:] and config.device.startswith("xpu"):
            raise ValueError("XPU execution requires coordinate-study-0.11")
        if schema not in _SCHEMAS[2:] and (
            config.architecture == "page"
        ):
            raise ValueError("page architecture requires coordinate-study-0.3")
        if (schema not in _SCHEMAS[4:]
                and config.architecture in ("projective", "surface")):
            raise ValueError("analytic architecture requires coordinate-study-0.5")
        if config.initialization_mode == "backbone" and config.initial_checkpoint_sha256 is None:
            raise ValueError("backbone initialization requires its source checkpoint SHA")
        if config.initialization_mode == "resolution" and (
            config.initial_checkpoint_sha256 is None or config.initial_model_config is None
        ):
            raise ValueError("resolution initialization requires source config and checkpoint SHA")
        epoch = saved["epoch"]
        if type(epoch) is not int or not 0 <= epoch <= config.epochs:
            raise ValueError("invalid epoch")
        history = saved["history"]
        if not isinstance(history, list) or len(history) != epoch:
            raise ValueError("history length disagrees with epoch")
        initial = saved["initial_validation_epe_pixels"]
        if not _finite_number(initial):
            raise ValueError("invalid initial validation metric")
        metrics = [initial]
        for number, row in enumerate(history, 1):
            if (not isinstance(row, dict) or row.keys() != {
                "epoch", "train_loss", "validation_epe_pixels"
            } or type(row["epoch"]) is not int or row["epoch"] != number
                    or not _finite_number(row["train_loss"])
                    or not _finite_number(row["validation_epe_pixels"])):
                raise ValueError("invalid history row")
            metrics.append(row["validation_epe_pixels"])
        best = min(range(len(metrics)), key=metrics.__getitem__)
        if (type(saved["best_epoch"]) is not int or saved["best_epoch"] != best
                or not _finite_number(saved["best_validation_epe_pixels"])
                or saved["best_validation_epe_pixels"] != metrics[best]
                or saved["checkpoint_kind"] == "best" and epoch != best):
            raise ValueError("best checkpoint selection is inconsistent")
        best_hash = saved["best_checkpoint_sha256"]
        if ((saved["checkpoint_kind"] == "best" and best_hash is not None)
                or (saved["checkpoint_kind"] == "last" and (
                    not isinstance(best_hash, str) or not _HEX.fullmatch(best_hash)))):
            raise ValueError("invalid best checkpoint link")
        if not isinstance(saved["manifest_sha256"], str) or not _HEX.fullmatch(
            saved["manifest_sha256"]
        ):
            raise ValueError("invalid manifest digest")
        if saved["dataset_output_size"] != config.output_size:
            raise ValueError("dataset output size differs from configuration")
        photo = saved["dataset_photo_size"]
        if (not isinstance(photo, tuple) or len(photo) != 2
                or any(type(v) is not int or not 2 <= v <= 512 for v in photo)):
            raise ValueError("invalid source photo size")
        counts = saved["dataset_counts"]
        if (not isinstance(counts, dict) or counts.keys() != {"train", "validation", "test"}
                or any(type(v) is not int or not 1 <= v <= 100_000 for v in counts.values())):
            raise ValueError("invalid positive dataset counts")
        provenance = saved["provenance"]
        expected = _provenance(schema, include_hashes=False)
        if not isinstance(provenance, dict) or provenance.keys() != expected.keys():
            raise ValueError("invalid provenance")
        for key in ("coordinate_loss", "selection", "resume_scope"):
            if provenance[key] != expected[key]:
                raise ValueError("incompatible study contract")
        if schema != LEGACY_CHECKPOINT_SCHEMA and not _same_state(
            provenance["loss_protocol"], expected["loss_protocol"]
        ):
            raise ValueError("incompatible regularization protocol")
        if schema in _SCHEMAS[4:] and any(
            not _same_state(provenance[key], expected[key])
            for key in ("map_protocol", "initialization_protocol")
        ):
            raise ValueError("incompatible map or initialization protocol")
        if schema in _SCHEMAS[6:] and not _same_state(
            provenance["augmentation_protocol"], expected["augmentation_protocol"]
        ):
            raise ValueError("incompatible augmentation protocol")
        if schema in _SCHEMAS[10:] and not _same_state(
            provenance["execution_protocol"], expected["execution_protocol"]
        ):
            raise ValueError("incompatible explicit-device protocol")
        if schema == CHECKPOINT_SCHEMA and not _same_state(
            provenance["direction_training_protocol"], expected["direction_training_protocol"]
        ):
            raise ValueError("incompatible training direction protocol")
        sources = provenance["source_code_sha256"]
        required_sources = {"learning/model.py", "learning/spatial.py", "learning/scene_data.py",
                            "learning/coordinate_study.py"}
        if schema in _SCHEMAS[8:]:
            required_sources.add("learning/augmentation.py")
        if (not isinstance(sources, dict) or not 4 <= len(sources) <= 256
                or not required_sources <= sources.keys()
                or any(not isinstance(k, str) or not re.fullmatch(r"[\w./-]{1,240}", k)
                       or Path(k).is_absolute() or ".." in Path(k).parts
                       or not isinstance(v, str) or not _HEX.fullmatch(v)
                       for k, v in sources.items())):
            raise ValueError("invalid source fingerprints")
        versions = provenance["versions"]
        if (not isinstance(versions, dict) or versions.keys() != expected["versions"].keys()
                or any(not isinstance(v, str) or not 1 <= len(v) <= 256
                       for k, v in versions.items() if k not in ("torch_cuda", "torch_xpu"))
                or any(value is not None and (
                    not isinstance(value, str) or not 1 <= len(value) <= 256)
                    for key, value in versions.items() if key in ("torch_cuda", "torch_xpu"))):
            raise ValueError("invalid runtime versions")
        lock = provenance["uv_lock_sha256"]
        if lock is not None and (not isinstance(lock, str) or not _HEX.fullmatch(lock)):
            raise ValueError("invalid dependency lock digest")
        settings = provenance["torch_settings"]
        expected_settings = expected["torch_settings"].keys()
        if (not isinstance(settings, dict)
                or settings.keys() not in (expected_settings, expected_settings - {"threads"})
                or any(type(settings[k]) is not bool for k in (
                    "deterministic", "warn_only", "cudnn_deterministic", "cudnn_benchmark",
                    "matmul_tf32", "cudnn_tf32"))
                or settings["matmul_precision"] not in ("highest", "high", "medium")
                or (settings["cudnn_version"] is not None and (
                    type(settings["cudnn_version"]) is not int or settings["cudnn_version"] < 1))
                or ("threads" in settings and (
                    type(settings["threads"]) is not int or not 1 <= settings["threads"] <= 4096))):
            raise ValueError("invalid torch runtime settings")
        info = saved["device_info"]
        if (not isinstance(info, dict) or info.keys() != {"device", "name", "capability"}
                or not isinstance(info["name"], str) or not 1 <= len(info["name"]) <= 256):
            raise ValueError("invalid device provenance")
        if config.device == "cpu":
            if info["device"] != "cpu" or info["capability"] is not None:
                raise ValueError("CPU configuration requires CPU provenance")
        elif config.device.startswith("xpu"):
            if (not isinstance(info["device"], str)
                    or not re.fullmatch(r"xpu:(0|[1-9][0-9]{0,3})", info["device"])
                    or info["capability"] is not None
                    or ":" in config.device and config.device != info["device"]):
                raise ValueError("invalid XPU provenance")
        elif (not isinstance(info["device"], str)
                or not re.fullmatch(r"cuda:(0|[1-9][0-9]{0,3})", info["device"])
                or not isinstance(info["capability"], tuple) or len(info["capability"]) != 2
                or any(type(v) is not int or not 0 <= v <= 100 for v in info["capability"])
                or ":" in config.device and config.device != info["device"]):
            raise ValueError("invalid CUDA provenance")
        for name in ("torch_rng_state", "shuffle_rng_state"):
            value = saved[name]
            if not _tensor(value, (torch.get_rng_state().numel(),), torch.uint8):
                raise ValueError("invalid CPU RNG tensor")
            torch.Generator(device="cpu").set_state(value)
        for backend in ("cuda", "xpu"):
            rng = saved.get(f"{backend}_rng_state")
            if not config.device.startswith(backend):
                if rng is not None:
                    raise ValueError(f"configuration must not contain unused {backend} RNG state")
            elif (not isinstance(rng, torch.Tensor) or rng.ndim != 1
                    or not 16 <= rng.numel() <= 65_536
                    or not _tensor(rng, rng.shape, torch.uint8)):
                raise ValueError(f"invalid {backend.upper()} RNG tensor")
        with torch.random.fork_rng(devices=[]):
            model = _new_model(config)
        if not _same_state(saved["model_spec"], _model_spec(config, model)):
            raise ValueError("model specification mismatch")
        state = saved["model_state"]
        if (not isinstance(state, dict) or state.keys() != model.state_dict().keys()
                or any(not _tensor(state[k], v.shape) for k, v in model.state_dict().items())):
            raise ValueError("model weights must be finite dense CPU float32 tensors")
        parameters = list(model.parameters())
        optimizer = saved["optimizer_state"]
        if (not isinstance(optimizer, dict) or optimizer.keys() != {"state", "param_groups"}
                or not _same_state(optimizer["param_groups"],
                                   [_adam_group(config, len(parameters))])
                or not isinstance(optimizer["state"], dict)):
            raise ValueError("invalid Adam parameter group")
        moments = optimizer["state"]
        if moments.keys() != (set(range(len(parameters))) if epoch else set()):
            raise ValueError("Adam state count disagrees with epoch")
        steps = epoch * math.ceil(counts["train"] / config.batch_size)
        for index, entry in moments.items():
            if (type(index) is not int or not isinstance(entry, dict)
                    or entry.keys() != {"step", "exp_avg", "exp_avg_sq"}
                    or not _tensor(entry["step"], ()) or entry["step"].item() != steps
                    or not _tensor(entry["exp_avg"], parameters[index].shape)
                    or not _tensor(entry["exp_avg_sq"], parameters[index].shape)
                    or bool((entry["exp_avg_sq"] < 0).any())):
                raise ValueError("invalid Adam moments")
        return config
    except (ValueError, TypeError, KeyError, RuntimeError, AttributeError, FlattenerError) as error:
        raise _error("invalid_checkpoint", str(error), Status.MODEL_ERROR) from error


def _current_generation(output):
    """Resolve one immutable generation from a bounded portable pointer."""
    pointer = output / "current.json"
    if pointer.is_symlink() or pointer.stat().st_size > 256:
        raise _error("invalid_output", "invalid checkpoint generation pointer")
    record = json.loads(pointer.read_text(encoding="ascii"))
    if (not isinstance(record, dict) or record.keys() != {"generation"}
            or not isinstance(record["generation"], str)
            or not re.fullmatch(r"\.checkpoint-[A-Za-z0-9_-]+", record["generation"])):
        raise _error("invalid_output", "invalid checkpoint generation pointer")
    generation = output / record["generation"]
    if generation.is_symlink() or not generation.is_dir():
        raise _error("invalid_output", "checkpoint generation must be an owned directory")
    return generation.resolve(strict=True)


def _checkpoint_location(path, *, strict=False):
    path = Path(path).absolute()
    pointer = path.parent / "current.json"
    if path.name in {"best.pt", "last.pt"} and (pointer.exists() or pointer.is_symlink()):
        path = _current_generation(path.parent) / path.name
    return path.resolve(strict=strict)


def _read_checkpoint(path):
    """Return the validated payload and SHA of the same bounded byte snapshot."""
    try:
        # Resolve the stable alias once so an atomic current-generation switch
        # cannot pair a previous .pt with the next generation's checksum.
        path = _checkpoint_location(path, strict=True)
        size = path.stat().st_size
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not 0 < size <= MAX_CHECKPOINT_BYTES or sidecar.stat().st_size != 65:
            raise ValueError("checkpoint or checksum exceeds its resource limit")
        with sidecar.open("rb") as stream:
            digest = stream.read(66).decode("ascii")
        if len(digest) != 65 or digest[-1:] != "\n" or not _HEX.fullmatch(digest[:-1]):
            raise ValueError("invalid checksum sidecar")
        with path.open("rb") as stream:
            payload = stream.read(MAX_CHECKPOINT_BYTES + 1)
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest[:-1]:
            raise _error("checkpoint_hash_mismatch", "checkpoint differs from its checksum",
                         Status.MODEL_ERROR)
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        _validate_saved(saved)
        return saved, digest[:-1]
    except OSError as error:
        raise _error("checkpoint_read_failed", str(error), Status.IO_ERROR) from error
    except (ValueError, TypeError, RuntimeError, EOFError,
            UnicodeError, pickle.UnpicklingError) as e:
        raise _error("invalid_checkpoint", str(e), Status.MODEL_ERROR) from e


def coordinate_checkpoint(path):
    """Load a trusted local checkpoint after checking its mandatory .sha256 sidecar."""
    return _read_checkpoint(path)[0]


def _initialization(config, initial_checkpoint, resume):
    """Bind fresh model-only initialization to the validated checkpoint bytes."""
    if resume is not None:
        if initial_checkpoint is not None:
            raise _error("invalid_initialization", "resume and initial_checkpoint are exclusive")
        if config.initialization_mode in ("backbone", "resolution") and (
            config.initial_checkpoint_sha256 is None
            or config.initialization_mode == "resolution" and config.initial_model_config is None
        ):
            raise _error("invalid_initialization",
                         "resume requires its bound initialization source")
        return config, None
    if initial_checkpoint is None:
        if (config.initial_checkpoint_sha256 is not None
                or config.initialization_mode in ("backbone", "resolution")):
            raise _error("invalid_initialization",
                         "fresh initialization requires a checkpoint path")
        return config, None
    saved, digest = _read_checkpoint(initial_checkpoint)
    source_architecture = ("global" if config.initialization_mode == "backbone"
                           else config.architecture)
    source_config = ModelConfig(**saved["config"]["model_config"])
    ignored = {"input_size"} if config.initialization_mode == "resolution" else set()
    if (saved["config"]["architecture"] != source_architecture
            or any(getattr(source_config, item.name) != getattr(config.model_config, item.name)
                   for item in fields(ModelConfig) if item.name not in ignored)):
        raise _error("initial_checkpoint_mismatch", "initial model architecture or config differs")
    if config.initial_model_config is not None and config.initial_model_config != source_config:
        raise _error("initial_checkpoint_mismatch", "initial source ModelConfig differs")
    if (config.initial_checkpoint_sha256 is not None
            and config.initial_checkpoint_sha256 != digest):
        raise _error("initial_checkpoint_mismatch", "initial checkpoint differs from declared SHA")
    state = saved["model_state"]
    if config.initialization_mode == "resolution":
        with torch.random.fork_rng(devices=[]):
            destination = _new_model(config).state_dict()
        if state.keys() != destination.keys() or any(
            not _tensor(state[key], value.shape) for key, value in destination.items()
        ):
            raise _error("initial_checkpoint_mismatch", "resolution model weight shapes differ")
        config = replace(config, initial_model_config=source_config)
    if config.initialization_mode == "backbone":
        with torch.random.fork_rng(devices=[]):
            if config.architecture == "oriented_surface":
                # New page-head weights must depend on the declared run seed,
                # independently of the caller's ambient CPU generator state.
                torch.random.default_generator.manual_seed(config.seed)
            destination = _new_model(config).state_dict()
        transferred = {key for key in state if key.startswith(("encoder.", "head.1."))}
        expected = {key for key in destination if key.startswith(("encoder.", "head.1."))}
        final = {"head.3.weight", "head.3.bias"}
        extra = (set(destination) - expected - final
                 if config.architecture == "oriented_surface" else set())
        if (not transferred or transferred != expected
                or destination.keys() - expected != final | extra
                or state.keys() - transferred != final
                or any(not _tensor(state[key], destination[key].shape) for key in expected)
                or any(bool(destination[key].count_nonzero()) for key in final)):
            raise _error("initial_checkpoint_mismatch",
                         "backbone keys, shapes, or identity head differ")
        destination.update({key: state[key] for key in expected})
        state = destination
    return replace(config, initial_checkpoint_sha256=digest), state


def model_from_checkpoint(saved):
    config = _validate_saved(saved)
    with torch.random.fork_rng(devices=[]):
        model = _new_model(config)
    model.load_state_dict(saved["model_state"], strict=True)
    return model.eval().requires_grad_(False)


def load_coordinate_predictor(checkpoint):
    """Load a coordinate-study checkpoint such as ``selected.pt`` for CPU inference."""
    started = perf_counter()
    path = _checkpoint_location(checkpoint)
    saved, digest = _read_checkpoint(path)
    config = _validate_saved(saved)
    model = model_from_checkpoint(saved)
    return LoadedPredictor(model, model.config, digest,
                           f"coordinate-study-{saved['config']['architecture']}-1.0:{digest[:12]}",
                           str(path), True, perf_counter() - started, map_mode=config.map_mode)


def _atomic_bytes(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_checkpoint(path, payload):
    path = Path(path)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    if len(data) > MAX_CHECKPOINT_BYTES:
        raise _error("checkpoint_size_limit", "checkpoint exceeds 512 MiB")
    digest = hashlib.sha256(data).hexdigest()
    _atomic_bytes(path, data)
    _atomic_bytes(path.with_suffix(path.suffix + ".sha256"), (digest + "\n").encode("ascii"))
    return digest


def _commit_checkpoints(output: Path, payload: dict) -> str:
    """Publish a complete immutable best/last generation with one pointer swap.

    The caller holds .training.lock. Incomplete staging generations stay on disk
    for diagnosis and are unreachable through ``current``; older generations stay.
    """
    output = Path(output).resolve(strict=True)
    # Validate a last-shaped copy first; the incoming model may differ from the
    # selected best, whose digest is set below.
    pending = dict(payload, checkpoint_kind="last", best_checkpoint_sha256="0" * 64)
    _validate_saved(pending)
    portable = _PORTABLE_CHECKPOINTS
    current = output / ("current.json" if portable else "current")
    previous = None
    if current.exists() or current.is_symlink():
        if portable:
            previous = _current_generation(output)
        else:
            target = os.readlink(current) if current.is_symlink() else ""
            if not re.fullmatch(r"\.checkpoint-[A-Za-z0-9_-]+", target):
                raise _error("invalid_output",
                             "current must point to an owned checkpoint generation")
            previous = current.resolve(strict=True)
        if previous.parent != output or not previous.is_dir():
            raise _error("invalid_output", "current points outside the checkpoint directory")
    aliases = ("best.pt", "best.pt.sha256", "last.pt", "last.pt.sha256")
    for name in aliases:
        alias = output / name
        if (alias.exists() or alias.is_symlink()) and (
            portable or not alias.is_symlink() or os.readlink(alias) != f"current/{name}"
        ):
            raise _error("output_exists", f"refusing to replace existing {name}")
    previous_best = None
    previous_digest = None
    if previous is not None:
        previous_best, previous_digest = _read_checkpoint(previous / "best.pt")
        previous_last = coordinate_checkpoint(previous / "last.pt")
        identity = ("config", "model_spec", "provenance", "manifest_sha256", "device_info",
                    "dataset_output_size", "dataset_photo_size", "dataset_counts",
                    "initial_validation_epe_pixels")
        if (previous_best["checkpoint_kind"] != "best"
                or previous_last["checkpoint_kind"] != "last"
                or previous_last["best_checkpoint_sha256"] != previous_digest
                or previous_best["epoch"] != previous_last["best_epoch"]
                or previous_best["best_validation_epe_pixels"]
                != previous_last["best_validation_epe_pixels"]
                or previous_last["epoch"] > pending["epoch"]
                or previous_last["history"] != pending["history"][:previous_last["epoch"]]
                or any(not _same_state(previous_last[k], pending[k])
                       or not _same_state(previous_best[k], pending[k]) for k in identity)):
            raise _error("invalid_checkpoint", "previous generation has a different study identity")
    is_best = pending["best_epoch"] == pending["epoch"]
    if not is_best and (previous_best is None
                        or previous_best["epoch"] != pending["best_epoch"]
                        or previous_best["history"] != pending["history"][:pending["best_epoch"]]
                        or previous_best["best_validation_epe_pixels"]
                        != pending["best_validation_epe_pixels"]):
        raise _error("invalid_checkpoint", "the selected earlier best checkpoint is unavailable")
    generation = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=output))
    for name in (() if portable else aliases):
        alias = output / name
        if not alias.is_symlink():
            alias.symlink_to(f"current/{name}")  # Never replaces existing paths.
    if is_best:
        best_payload = dict(payload, checkpoint_kind="best", best_checkpoint_sha256=None)
        # pending was fully validated above; only these checked kind/link
        # fields differ. Serialized files are fully revalidated below.
        best_digest = _save_checkpoint(generation / "best.pt", best_payload)
    else:
        for name in ("best.pt", "best.pt.sha256"):
            os.link(previous / name, generation / name)
        best_digest = previous_digest
    last_payload = dict(payload, checkpoint_kind="last", best_checkpoint_sha256=best_digest)
    _save_checkpoint(generation / "last.pt", last_payload)
    # Validate the finished files before publication, including linked best.
    stored_best, verified_digest = _read_checkpoint(generation / "best.pt")
    stored_last = coordinate_checkpoint(generation / "last.pt")
    if (verified_digest != best_digest or stored_last["best_checkpoint_sha256"] != best_digest
            or stored_best["epoch"] != stored_last["best_epoch"]):
        raise _error("invalid_checkpoint", "staged generation failed final identity validation")
    if portable:
        # Windows cannot atomically replace a directory symlink, so publish a
        # pointer file. Without directory fsync, durability rests on the filesystem.
        _atomic_bytes(current, (json.dumps({"generation": generation.name}) + "\n").encode("ascii"))
        return best_digest
    pointer = generation / ".publish-current"
    pointer.symlink_to(generation.name)  # Relative to output after the atomic move.
    directory_fd = os.open(generation, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.replace(pointer, current)
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return best_digest


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(v) for v in value)
    return copy.deepcopy(value)


def _same_state(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(map(_same_state, left, right))
    return left == right


def _dense_maps(residual, output_size, mode="legacy"):
    return expand_coordinate_residual(residual, output_size, mode=mode)


def _forward_maps(model, images, config, *, _checks=None, photo_size=None,
                  teacher_direction=None, teacher_valid=None):
    """Return coarse residual, dense map, and optional page logits once per batch."""
    teacher = teacher_direction is not None or teacher_valid is not None
    if teacher and (config.architecture != "oriented_surface"
                    or config.direction_training != "teacher" or not model.training
                    or _checks is None):
        raise _error("invalid_training_batch",
                     "teacher direction requires the checked training path")
    if config.architecture == "oriented_surface":
        if _checks is not None:
            overrides = ({"teacher_direction": teacher_direction, "teacher_valid": teacher_valid}
                         if teacher else {})
            residual, predicted, auxiliary, finite = model._forward_with_geometry_validated(
                images, config.output_size, photo_size=photo_size,
                need_coarse=bool(config.curvature_weight), **overrides)
            _checks.append(finite)
            return residual, predicted, auxiliary
        return model.forward_with_geometry(images, config.output_size, photo_size=photo_size)
    if config.map_mode == "analytic":
        if _checks is not None and not config.curvature_weight:
            predicted, finite = model._forward_dense_validated(images, config.output_size)
            _checks.append(finite)
            return None, predicted, None
        residual, predicted = model.forward_dense(images, config.output_size)
        if _checks is not None:
            _checks.append(torch.isfinite(predicted).all())
        return residual, predicted, None
    if config.architecture == "page":
        residual, logits = model.forward_with_page(images)
    else:
        residual, logits = model(images), None
    predicted = _dense_maps(residual, config.output_size, mode=config.map_mode)
    if _checks is not None:
        _checks.append(torch.isfinite(predicted).all())
    return residual, predicted, logits


def _regularizer_tensor(value, name, *, _validated=False):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.device.type not in ("cpu", "cuda", "xpu") or value.layout != torch.strided):
        raise _error("invalid_training_batch", f"{name} must be a dense CPU/CUDA float32 tensor")
    if not _validated and not bool(torch.isfinite(value).all()):
        raise _error("invalid_training_batch", f"{name} must be finite")


def _topology_loss(predicted, valid, *, _validated=False):
    """Return the mean plus worst corner-Jacobian hinge over fixed GT cells."""
    _regularizer_tensor(predicted, "predicted", _validated=_validated)
    if (predicted.ndim != 4 or predicted.shape[-1] != 2 or predicted.shape[0] < 1
            or min(predicted.shape[1:3]) < 2 or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool or valid.device != predicted.device
            or valid.layout != torch.strided or valid.shape != predicted.shape[:-1]):
        raise _error("invalid_training_batch", "invalid topology map or fixed GT mask")
    height, width = predicted.shape[1:3]
    a, b = predicted[:, :-1, :-1], predicted[:, :-1, 1:]
    c, d = predicted[:, 1:, :-1], predicted[:, 1:, 1:]
    top, bottom = (b - a) * (width / 2), (d - c) * (width / 2)
    left, right = (c - a) * (height / 2), (d - b) * (height / 2)
    determinants = torch.stack([
        dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        for dx, dy in ((top, left), (top, right), (bottom, left), (bottom, right))
    ], dim=-1)
    cells = valid[:, :-1, :-1] & valid[:, :-1, 1:] & valid[:, 1:, :-1] & valid[:, 1:, 1:]
    penalty = torch.where(cells[..., None], F.relu(0.001 - determinants).square(), 0.0)
    count = cells.flatten(1).sum(1) * 4
    mean = penalty.flatten(1).sum(1) / count.clamp_min(1)
    worst = penalty.flatten(1).amax(1)
    return ((mean + worst) * (count > 0)).sum() / (count > 0).sum().clamp_min(1)


def _relative_topology_loss(predicted, target, valid, *, _validated=False, _checks=None):
    """GT-relative four-corner hinge on fixed valid, positive-orientation cells."""
    _map_loss_inputs(predicted, target, valid, (2, 2), _validated=_validated)
    if min(predicted.shape[1:3]) < 2:
        raise _error("invalid_training_batch", "relative topology requires at least a 2x2 map")
    height, width = predicted.shape[1:3]

    def determinants(grid):
        a, b = grid[:, :-1, :-1], grid[:, :-1, 1:]
        c, d = grid[:, 1:, :-1], grid[:, 1:, 1:]
        top, bottom = (b - a) * (width / 2), (d - c) * (width / 2)
        left, right = (c - a) * (height / 2), (d - b) * (height / 2)
        return torch.stack([
            dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
            for dx, dy in ((top, left), (top, right), (bottom, left), (bottom, right))
        ], dim=-1)

    actual = determinants(predicted)
    reference = determinants(torch.where(valid[..., None], target, 0.0))
    cells = valid[:, :-1, :-1] & valid[:, :-1, 1:] & valid[:, 1:, :-1] & valid[:, 1:, 1:]
    finite = torch.isfinite(actual).all() & torch.isfinite(
        torch.where(cells[..., None], reference, 0.0)).all()
    if _checks is not None:
        _checks.append(finite)
    elif not bool(finite):
        raise _error("invalid_training_batch", "relative topology determinant overflow")
    cells = cells & (reference > 0).all(dim=-1)
    # Remove ineligible operands before division, including extreme finite
    # predictions outside GT, so a masked overflow cannot poison gradients.
    ratio = torch.where(cells[..., None], actual, 0.0) / torch.where(
        cells[..., None], reference, 1.0).clamp_min(1e-4)
    penalty = torch.where(cells[..., None], F.relu(0.1 - ratio).square(), 0.0)
    count = cells.flatten(1).sum(1) * 4
    mean = penalty.flatten(1).sum(1) / count.clamp_min(1)
    worst = penalty.flatten(1).amax(1)
    return ((mean + worst) * (count > 0)).sum() / (count > 0).sum().clamp_min(1)


def _curvature_loss(residual, photo_size, *, _validated=False):
    """SmoothL1 on coarse residual second differences in original source pixels."""
    _regularizer_tensor(residual, "residual", _validated=_validated)
    if (residual.ndim != 4 or residual.shape[0] < 1 or residual.shape[1] != 2
            or min(residual.shape[-2:]) < 1 or not isinstance(photo_size, tuple)
            or len(photo_size) != 2 or any(type(v) is not int or v < 1 for v in photo_size)):
        raise _error("invalid_training_batch", "invalid coarse residual or source size")
    scale = residual.new_tensor((photo_size[1] / 2, photo_size[0] / 2))[None, :, None, None]
    scaled = residual * scale
    differences = []
    if residual.shape[-1] >= 3:
        differences.append(scaled[..., 2:] - 2 * scaled[..., 1:-1] + scaled[..., :-2])
    if residual.shape[-2] >= 3:
        differences.append(scaled[:, :, 2:] - 2 * scaled[:, :, 1:-1] + scaled[:, :, :-2])
    if not differences:
        return residual.reshape(-1)[0] * 0
    return sum(F.smooth_l1_loss(d, torch.zeros_like(d), beta=1.0, reduction="sum")
               for d in differences) / sum(d.numel() for d in differences)


def _page_loss(logits, source_mask, *, _validated=False):
    """Per-image balanced soft-label BCE plus Dice on area-resized page truth."""
    _regularizer_tensor(logits, "page logits", _validated=_validated)
    if (logits.ndim != 4 or logits.shape[0] < 1 or logits.shape[1] != 1
            or min(logits.shape[-2:]) < 1 or not isinstance(source_mask, torch.Tensor)
            or source_mask.dtype != torch.bool or source_mask.layout != torch.strided
            or source_mask.device != logits.device or source_mask.ndim != 3
            or source_mask.shape[0] != logits.shape[0] or min(source_mask.shape[-2:]) < 1):
        raise _error("invalid_training_batch", "invalid page logits or original source mask")
    target = F.interpolate(source_mask[:, None].float(), size=logits.shape[-2:], mode="area")
    positive = target.flatten(1).sum(1)
    negative = (1 - target).flatten(1).sum(1)
    pos_loss = (target * F.softplus(-logits)).flatten(1).sum(1)
    neg_loss = ((1 - target) * F.softplus(logits)).flatten(1).sum(1)
    pos_loss = pos_loss / torch.where(positive > 0, positive, 1.0)
    neg_loss = neg_loss / torch.where(negative > 0, negative, 1.0)
    balanced = (pos_loss + neg_loss) / ((positive > 0).float() + (negative > 0).float())
    probabilities = logits.sigmoid()
    dice = 1 - (2 * (probabilities * target).flatten(1).sum(1) + 1) / (
        probabilities.flatten(1).sum(1) + positive + 1
    )
    return (balanced + dice).mean()


def _map_loss_inputs(predicted, target, valid, photo_size, *, _validated=False):
    _regularizer_tensor(predicted, "predicted", _validated=_validated)
    if (predicted.ndim != 4 or predicted.shape[-1] != 2 or predicted.shape[0] < 1
            or min(predicted.shape[1:3]) < 1 or not isinstance(target, torch.Tensor)
            or target.dtype != torch.float32 or target.layout != torch.strided
            or target.device != predicted.device or target.shape != predicted.shape
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.layout != torch.strided or valid.device != predicted.device
            or valid.shape != predicted.shape[:-1] or not isinstance(photo_size, tuple)
            or len(photo_size) != 2
            or any(type(v) is not int or not 1 <= v <= 2**31 - 1 for v in photo_size)):
        raise _error("invalid_training_batch", "invalid differential/boundary supervision")
    if not _validated and not bool(torch.isfinite(
        torch.where(valid[..., None], target, 0.0)).all()
    ):
        raise _error("invalid_training_batch", "GT must be finite at all fixed valid coordinates")


def _masked_coordinate_mean(predicted, target, valid, photo_size):
    selected = valid[..., None]
    delta = (torch.where(selected, predicted, 0.0) - torch.where(selected, target, 0.0))
    delta = delta * predicted.new_tensor((photo_size[1] / 2, photo_size[0] / 2))
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="sum") / (
        2 * valid.sum()).clamp_min(1)


def _coordinate_loss(predicted, target, valid, photo_size, *, _validated=False):
    """Return the SmoothL1 source-pixel coordinate loss on fixed valid GT."""
    _map_loss_inputs(predicted, target, valid, photo_size, _validated=_validated)
    if not _validated and not bool(valid.any()):
        raise _error("invalid_training_batch", "coordinate supervision requires valid GT")
    return _masked_coordinate_mean(predicted, target, valid, photo_size)


def _boundary_loss(predicted, target, valid, photo_size, *, _validated=False):
    """Source-pixel coordinate error on the fixed eight-pixel output boundary."""
    _map_loss_inputs(predicted, target, valid, photo_size, _validated=_validated)
    boundary = torch.zeros_like(valid)
    boundary[:, :8] = boundary[:, -8:] = True
    boundary[:, :, :8] = boundary[:, :, -8:] = True
    boundary &= valid
    return _masked_coordinate_mean(predicted, target, boundary, photo_size)


def _page_relative_loss(predicted, target, valid, source_mask, photo_size, *, _validated=False):
    """Weight each image's coordinate loss up when its GT page is small.

    The weight is photo_diagonal / max(page_diagonal, 0.05 * photo_diagonal) - 1,
    with page bbox sides max - min + 1. Only fixed valid GT enters the mean.
    """
    _map_loss_inputs(predicted, target, valid, photo_size, _validated=_validated)
    if (predicted.shape[0] < 1 or not isinstance(source_mask, torch.Tensor)
            or source_mask.dtype != torch.bool or source_mask.layout != torch.strided
            or source_mask.device != predicted.device
            or tuple(source_mask.shape) != (predicted.shape[0], *photo_size)):
        raise _error("invalid_training_batch", "page-relative loss requires original source_mask")
    height, width = photo_size
    columns, rows = source_mask.any(dim=1), source_mask.any(dim=2)
    x = torch.arange(width, device=predicted.device)
    y = torch.arange(height, device=predicted.device)
    bbox_width = (torch.where(columns, x, -1).amax(1)
                  - torch.where(columns, x, width).amin(1) + 1)
    bbox_height = (torch.where(rows, y, -1).amax(1)
                   - torch.where(rows, y, height).amin(1) + 1)
    counts = valid.flatten(1).sum(1)
    if not bool(((bbox_width >= 2) & (bbox_height >= 2) & (counts > 0)).all()):
        raise _error("invalid_training_batch", "page-relative GT/mask is empty or bbox degenerate")
    bbox = torch.stack((bbox_width, bbox_height), dim=1).float()
    page_diagonal = bbox.square().sum(1).sqrt()
    photo_diagonal = predicted.new_tensor((width, height)).square().sum().sqrt()
    weights = photo_diagonal / page_diagonal.clamp_min(0.05 * photo_diagonal) - 1
    # Select before arithmetic: invalid GT may be NaN, but cannot poison the
    # loss or its gradient. Zero-filled invalid predictions contribute nothing.
    selected = valid[..., None]
    delta = (torch.where(selected, predicted, 0.0) - torch.where(selected, target, 0.0))
    delta = delta * predicted.new_tensor((width / 2, height / 2))
    penalties = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=1.0, reduction="none")
    per_image = penalties.flatten(1).sum(1) / (2 * counts)
    return (weights * per_image).mean()


def _coarse_supervision(predicted, target, valid, grid_size):
    safe_target = torch.where(valid[..., None], target, 0.0)
    actual, truth = [F.interpolate(v.permute(0, 3, 1, 2), size=grid_size,
                                  mode="bilinear", align_corners=False)
                     for v in (predicted, safe_target)]
    mask = F.interpolate(valid[:, None].float(), size=grid_size, mode="bilinear",
                         align_corners=False)[:, 0] >= 1 - 1e-6
    return (torch.where(mask[:, None], actual, 0.0),
            torch.where(mask[:, None], truth, 0.0), mask)


def _differential_loss(predicted, target, valid, photo_size, grid_size, *, _validated=False):
    """Match the GT first and second differences on the coarse grid."""
    _map_loss_inputs(predicted, target, valid, photo_size, _validated=_validated)
    if (not isinstance(grid_size, tuple) or len(grid_size) != 2
            or any(type(v) is not int or v < 1 for v in grid_size)
            or any(g > p for g, p in zip(grid_size, predicted.shape[1:3], strict=True))):
        raise _error("invalid_training_batch", "coarse supervision grid may not exceed dense size")
    actual, truth, mask = _coarse_supervision(predicted, target, valid, grid_size)
    scale = actual.new_tensor((photo_size[1] / 2, photo_size[0] / 2))[None, :, None, None]
    error = ((actual - truth) * scale).permute(0, 2, 3, 1)
    total, count = predicted.reshape(-1)[0] * 0, valid.new_zeros((), dtype=torch.int64)
    for axis in (1, 2):
        if grid_size[axis - 1] >= 2:
            first = torch.diff(error, dim=axis)
            first_valid = mask.narrow(axis, 0, mask.shape[axis] - 1) & mask.narrow(
                axis, 1, mask.shape[axis] - 1
            )
            selected = torch.where(first_valid[..., None], first, 0.0)
            total = total + F.smooth_l1_loss(selected, torch.zeros_like(selected), reduction="sum")
            count = count + first_valid.sum() * 2
            if grid_size[axis - 1] >= 3:
                second = torch.diff(first, dim=axis)
                second_valid = first_valid.narrow(axis, 0, first_valid.shape[axis] - 1) & (
                    first_valid.narrow(axis, 1, first_valid.shape[axis] - 1)
                )
                selected = torch.where(second_valid[..., None], second, 0.0)
                total = total + F.smooth_l1_loss(
                    selected, torch.zeros_like(selected), reduction="sum"
                )
                count = count + second_valid.sum() * 2
    return total / count.clamp_min(1)


def _batch(batch, device):
    if (not isinstance(batch, dict) or any(not isinstance(batch.get(k), torch.Tensor)
                                         for k in ("image", "target", "valid"))):
        raise _error("invalid_training_batch", "batch requires image, target and valid tensors")
    image, target, valid = (batch[k].to(device, non_blocking=device.type in ("cuda", "xpu"))
                            for k in ("image", "target", "valid"))
    if (image.dtype != torch.float32 or image.ndim != 4 or image.shape[1] != 3
            or image.shape[0] < 1 or min(image.shape[-2:]) < 1
            or target.dtype != torch.float32 or valid.dtype != torch.bool
            or target.shape != (*valid.shape, 2) or valid.ndim != 3
            or target.shape[0] != image.shape[0] or min(valid.shape[1:]) < 1
            or any(v.layout != torch.strided for v in (image, target, valid))):
        raise _error("invalid_training_batch", "invalid image or fixed-mask GT tensor metadata")
    admitted = (torch.isfinite(image).all() & (image >= 0).all() & (image <= 1).all()
                & valid.flatten(1).any(1).all()
                & torch.isfinite(torch.where(valid[..., None], target, 0.0)).all())
    if not bool(admitted):
        raise _error("invalid_training_batch", "each image requires finite fixed-mask GT")
    return image, target, valid


def _orientation_loss(predicted, target, valid, photo_size, grid_size, *, _validated=False):
    """Penalize coarse-difference direction error on GT-eligible cells, per image.

    Exactly opposite directions are stationary when the predicted norm is >= 1,
    so use this loss only together with the coordinate loss.
    """
    _map_loss_inputs(predicted, target, valid, photo_size, _validated=_validated)
    if (predicted.shape[0] < 1 or not isinstance(grid_size, tuple) or len(grid_size) != 2
            or any(type(v) is not int or v < 1 for v in grid_size)
            or any(g > p for g, p in zip(grid_size, predicted.shape[1:3], strict=True))):
        raise _error("invalid_training_batch", "invalid orientation supervision grid")
    actual, truth, mask = _coarse_supervision(predicted, target, valid, grid_size)
    scale = actual.new_tensor((photo_size[1] / 2, photo_size[0] / 2))[None, :, None, None]
    totals = predicted[:, 0, 0, 0] * 0
    counts = torch.zeros(predicted.shape[0], dtype=torch.int64, device=predicted.device)
    for axis in (2, 3):
        if grid_size[axis - 2] < 2:
            continue
        a, b = (torch.diff(v, dim=axis) * scale for v in (actual, truth))
        norm_a = torch.linalg.vector_norm(a, dim=1)
        norm_b = torch.linalg.vector_norm(b, dim=1)
        endpoints = mask.narrow(axis - 1, 0, mask.shape[axis - 1] - 1) & mask.narrow(
            axis - 1, 1, mask.shape[axis - 1] - 1)
        eligible = endpoints & (norm_b >= 1.0)
        # The denominator for ineligible zero-length GT must still be finite.
        cosine = (a * b).sum(1) / (norm_a.clamp_min(1.0) * norm_b.clamp_min(1.0))
        penalty = torch.where(eligible, 1 - cosine.clamp(-1.0, 1.0), 0.0)
        totals = totals + penalty.flatten(1).sum(1)
        counts = counts + eligible.flatten(1).sum(1)
    means = totals / counts.clamp_min(1)
    return means.sum() / (counts > 0).sum().clamp_min(1)


def _rotate_training_batch(batch, probability, *, _validated=False, _checks=None):
    """Quarter-turn random items on their device using the CPU RNG; inputs stay intact."""
    if not _finite_number(probability) or probability > 1:
        raise _error("invalid_training_batch", "rotation probability must be in [0,1]")
    if probability == 0:
        return batch
    if not isinstance(batch, dict) or not {"image", "target", "valid"} <= batch.keys():
        raise _error("invalid_training_batch", "rotation requires image, target and valid tensors")
    image, target, valid = (batch[key] for key in ("image", "target", "valid"))
    _regularizer_tensor(image, "rotation image", _validated=_validated)
    if (image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3
            or min(image.shape[-2:]) < 1
            or not _validated and not bool(((image >= 0) & (image <= 1)).all())):
        raise _error("invalid_training_batch", "rotation image must be N3HW in [0,1]")
    if (not isinstance(target, torch.Tensor) or target.dtype != torch.float32
            or target.layout != torch.strided or target.device != image.device
            or target.ndim != 4 or target.shape[-1] != 2
            or target.shape[0] != image.shape[0] or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool or valid.layout != torch.strided
            or valid.device != image.device or valid.shape != target.shape[:-1]):
        raise _error("invalid_training_batch", "invalid rotation target or fixed GT mask")
    if not _validated and not bool(valid.flatten(1).any(1).all() & torch.isfinite(
        torch.where(valid[..., None], target, 0.0)).all()
    ):
        raise _error("invalid_training_batch", "invalid rotation target or fixed GT mask")
    source_mask = batch.get("source_mask")
    if "source_mask" in batch and (
        not isinstance(source_mask, torch.Tensor) or source_mask.dtype != torch.bool
        or source_mask.layout != torch.strided or source_mask.device != image.device
        or source_mask.ndim != 3 or source_mask.shape[0] != image.shape[0]
        or min(source_mask.shape[-2:]) < 1
    ):
        raise _error("invalid_training_batch", "rotation source_mask must be original bool NHW")
    result = dict(batch)
    for key in ("image", "target", "valid", "source_mask"):
        if isinstance(result.get(key), torch.Tensor):
            result[key] = result[key].clone()
    if float(torch.rand((), device="cpu")) >= probability:
        return result
    k = int(torch.randint(1, 4, (), device="cpu"))
    rotated = F.interpolate(torch.rot90(image, k, (-2, -1)), size=image.shape[-2:],
                            mode="bilinear", align_corners=False)
    tolerance = 8 * torch.finfo(torch.float32).eps
    admissible = (torch.isfinite(rotated) & (rotated >= -tolerance)
                  & (rotated <= 1 + tolerance)).all()
    if _checks is not None:
        _checks.append(admissible)
    elif not bool(admissible):
        raise _error("invalid_training_batch", "rotated RGB exceeds interpolation roundoff bounds")
    result["image"] = rotated.clamp(0.0, 1.0)
    x, y = target.unbind(-1)
    result["target"] = torch.stack(((y, -x), (-x, -y), (-y, x))[k - 1], dim=-1)
    if source_mask is not None:
        result["source_mask"] = F.interpolate(
            torch.rot90(source_mask[:, None].float(), k, (-2, -1)),
            size=source_mask.shape[-2:], mode="nearest-exact")[:, 0].bool()
    if "corner_target" in batch:
        matrices = image.new_tensor(([[0, 1], [-1, 0]], [[-1, 0], [0, -1]],
                                     [[0, -1], [1, 0]]))
        # Resizing back to the input shape can change the physical aspect, so
        # recompute direction from the transformed corners.
        photo_size = tuple(source_mask.shape[-2:]) if source_mask is not None else None
        if photo_size is None:
            raise _error("invalid_training_batch", "semantic batch rotation needs source_mask")
        result.update(transform_semantic_page_labels(
            batch, matrices[k - 1].expand(image.shape[0], -1, -1), photo_size,
            _checks=_checks))
    return result


def _semantic_page_losses(auxiliary, batch, photo_size, *, _checks=None):
    """Supervise physical direction and actual map boundary with fixed GT masks."""
    if not isinstance(auxiliary, dict):
        raise _error("invalid_training_batch", "missing semantic model outputs")
    logits, predicted = auxiliary.get("orientation_logits"), auxiliary.get("corners")
    _regularizer_tensor(logits, "orientation logits", _validated=True)
    _regularizer_tensor(predicted, "predicted corners", _validated=True)
    if (logits.ndim != 2 or logits.shape[1] != 4 or logits.shape[0] < 1
            or predicted.shape != (logits.shape[0], 4, 2) or predicted.device != logits.device):
        raise _error("invalid_training_batch", "invalid semantic model output shapes")
    device, count = logits.device, logits.shape[0]
    specs = {"corner_target": (torch.float32, (count, 4, 2)),
             "corner_valid": (torch.bool, (count, 4)),
             "direction_target": (torch.int64, (count,)),
             "direction_valid": (torch.bool, (count,))}
    labels = {}
    for key, (dtype, shape) in specs.items():
        value = batch.get(key)
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.layout != torch.strided or tuple(value.shape) != shape):
            raise _error("invalid_training_batch", f"invalid semantic label {key}")
        labels[key] = value.to(device, non_blocking=device.type in ("cuda", "xpu"))
    target, valid = labels["corner_target"], labels["corner_valid"]
    classes, eligible = labels["direction_target"], labels["direction_valid"]
    safe_target = torch.where(valid[..., None], target, 0.0)
    admitted = (torch.isfinite(logits).all() & torch.isfinite(predicted).all()
                & torch.isfinite(safe_target).all()
                & (safe_target.abs() <= 1).all()
                & torch.where(valid[..., None], True, torch.isnan(target)).all()
                & torch.where(eligible, (classes >= 0) & (classes < 4), True).all())
    # Clamp only labels passed to CE so invalid input cannot trigger a device
    # indexing assertion before the combined admission check rejects the batch.
    safe_classes = torch.where(eligible, classes, 0).clamp(0, 3)
    direction = F.cross_entropy(logits, safe_classes, reduction="none")
    direction = torch.where(eligible, direction, 0.0).sum() / eligible.sum().clamp_min(1)
    scale = predicted.new_tensor((photo_size[1] / 2, photo_size[0] / 2))
    low = torch.where(valid[..., None], safe_target, float("inf")).amin(1)
    high = torch.where(valid[..., None], safe_target, float("-inf")).amax(1)
    extent = torch.where(valid.any(1)[:, None], high - low, 0.0) * scale
    denominator = torch.linalg.vector_norm(extent, dim=-1).clamp_min(
        0.05 * math.hypot(*photo_size))
    error = torch.where(valid[..., None], predicted - safe_target, 0.0) * scale
    penalty = F.smooth_l1_loss(error / denominator[:, None, None],
                               torch.zeros_like(error), beta=0.01, reduction="none")
    counts = valid.sum(1) * 2
    corner = (penalty.sum((1, 2)) / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)
    if _checks is not None:
        _checks.append(admitted)
    elif not bool(admitted):
        raise _error("invalid_training_batch", "invalid semantic labels or predictions")
    return direction, corner


def _training_step(model, optimizer, batch, config, photo_size, device, *, _defer_metrics=False):
    """Run one backward and optimizer step; ``device`` must match ``config.device``."""
    _config(config)
    device = torch.device(device)
    parameters = list(model.parameters())
    backend = config.device.split(":", 1)[0]
    matches = (device.type == backend
               and (":" not in config.device or str(device) == config.device))
    reason = "device_mismatch" if backend == "cpu" else f"{backend}_required"
    if (not matches or not parameters
            or any(p.device != device for p in (*parameters, *model.buffers()))):
        raise _error(reason, "model and execution device must match the explicit configuration")
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if [id(p) for p in parameters] != [id(p) for p in optimized]:
        raise _error(reason, "optimizer must own exactly the configured model parameters")
    image, target, valid = _batch(batch, device)
    checks = []
    semantic = config.architecture == "oriented_surface"
    augmented = None
    if config.rotation_probability or config.background_probability:
        augmented = {"image": image, "target": target, "valid": valid}
        if semantic:
            for key in ("corner_target", "corner_valid", "direction_target", "direction_valid"):
                value = batch.get(key)
                if not isinstance(value, torch.Tensor):
                    raise _error("invalid_training_batch", f"missing semantic label {key}")
                augmented[key] = value.to(device, non_blocking=device.type in ("cuda", "xpu"))
        if config.page_weight or config.page_relative_weight or config.background_probability:
            mask = batch.get("source_mask")
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != (image.shape[0], *photo_size)):
                raise _error("invalid_training_batch", "augmentation requires original source_mask")
            augmented["source_mask"] = mask.to(device, non_blocking=device.type in ("cuda", "xpu"))
        if config.background_probability or config.rotation_mode == "per_image_fit":
            augmented = augment_training_batch(
                augmented, photo_size=photo_size,
                rotation_probability=(config.rotation_probability
                                      if config.rotation_mode == "per_image_fit" else 0.0),
                background_probability=config.background_probability,
                _validated=True, _checks=checks)
        if config.rotation_probability and config.rotation_mode == "batch":
            augmented = _rotate_training_batch(augmented, config.rotation_probability,
                                              _validated=True, _checks=checks)
        image, target, valid = (augmented[key] for key in ("image", "target", "valid"))
    optimizer.zero_grad(set_to_none=True)
    kwargs = {"photo_size": photo_size} if semantic else {}
    if config.direction_training == "teacher":
        labels = augmented if augmented is not None else batch
        for argument, key in (("teacher_direction", "direction_target"),
                              ("teacher_valid", "direction_valid")):
            value = labels.get(key)
            if not isinstance(value, torch.Tensor):
                raise _error("invalid_training_batch", f"missing semantic label {key}")
            kwargs[argument] = value.to(device, non_blocking=device.type in ("cuda", "xpu"))
    residual, predicted, page_logits = _forward_maps(model, image, config, _checks=checks, **kwargs)
    auxiliary = page_logits if semantic else None
    if semantic:
        page_logits = auxiliary["page_logits"]
    loss = _coordinate_loss(predicted, target, valid, photo_size, _validated=True)
    if config.topology_weight:
        topology = (_relative_topology_loss(predicted, target, valid, _validated=True,
                                           _checks=checks)
                    if config.topology_mode == "relative"
                    else _topology_loss(predicted, valid, _validated=True))
        loss = loss + config.topology_weight * topology
    if config.curvature_weight:
        loss = loss + config.curvature_weight * _curvature_loss(
            residual, photo_size, _validated=True)
    if config.page_weight or config.page_relative_weight:
        source_mask = (augmented if augmented is not None else batch).get("source_mask")
        if (not isinstance(source_mask, torch.Tensor) or source_mask.dtype != torch.bool
                or source_mask.shape != (image.shape[0], *photo_size)):
            raise _error("invalid_training_batch", "page supervision requires original source_mask")
        source_mask = source_mask.to(device, non_blocking=device.type in ("cuda", "xpu"))
    if config.page_weight:
        checks.append(torch.isfinite(page_logits).all())
        loss = loss + config.page_weight * _page_loss(page_logits, source_mask, _validated=True)
    if config.boundary_weight:
        loss = loss + config.boundary_weight * _boundary_loss(
            predicted, target, valid, photo_size, _validated=True)
    if config.differential_weight:
        loss = loss + config.differential_weight * _differential_loss(
            predicted, target, valid, photo_size, config.model_config.grid_size, _validated=True
        )
    if config.page_relative_weight:
        loss = loss + config.page_relative_weight * _page_relative_loss(
            predicted, target, valid, source_mask, photo_size, _validated=True)
    if config.orientation_weight:
        loss = loss + config.orientation_weight * _orientation_loss(
            predicted, target, valid, photo_size, config.model_config.grid_size, _validated=True)
    if semantic:
        direction_loss, corner_loss = _semantic_page_losses(
            auxiliary, augmented if augmented is not None else batch, photo_size, _checks=checks)
        loss = loss + config.direction_weight * direction_loss + config.corner_weight * corner_loss
    checks.append(torch.isfinite(loss))
    if not bool(torch.stack(checks).all()):
        raise _error("nonfinite_training", "nonfinite coordinate loss", Status.MODEL_ERROR)
    loss.backward()
    if (any(p.grad is None for p in parameters)
            or not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())):
        raise _error("nonfinite_training", "nonfinite or missing model gradient",
                     Status.MODEL_ERROR)
    if config.gradient_clip_norm:
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm,
                                       error_if_nonfinite=True)
    optimizer.step()
    pixels = int(batch["valid"].sum().item())
    return (loss.detach() if _defer_metrics else float(loss.detach())), pixels


def _validation_epe(model, loader, config, photo_size, device, *, deadline=None, diagnostics=None):
    model.eval()
    values = []
    semantic_rows = []
    with torch.inference_mode():
        for batch in loader:
            _check_deadline(deadline)
            image, target, valid = _batch(batch, device)
            checks = []
            kwargs = {"photo_size": photo_size} if config.architecture == "oriented_surface" else {}
            _, predicted, auxiliary = _forward_maps(model, image, config, _checks=checks, **kwargs)
            scale = predicted.new_tensor((photo_size[1] / 2, photo_size[0] / 2))
            delta = (torch.where(valid[..., None], predicted, 0.0)
                     - torch.where(valid[..., None], target, 0.0)) * scale
            distance = torch.linalg.vector_norm(delta, dim=-1)
            means = distance.double().sum((1, 2)) / valid.sum((1, 2))
            checks.append(torch.isfinite(means).all())
            if diagnostics is not None and config.architecture == "oriented_surface":
                labels = {key: batch[key].to(device, non_blocking=device.type in ("cuda", "xpu"))
                          for key in ("corner_target", "corner_valid",
                                      "direction_target", "direction_valid")}
                _semantic_page_losses(auxiliary, labels, photo_size, _checks=checks)
                correct = auxiliary["orientation_logits"].argmax(1)
                truth = labels["direction_target"].clamp(0, 3)
                eligible = labels["direction_valid"]
                corner_valid = labels["corner_valid"]
                error = torch.where(corner_valid[..., None],
                                    auxiliary["corners"] - labels["corner_target"], 0.0) * scale
                corner_error = torch.linalg.vector_norm(error, dim=-1).double().sum(1)
                corner_error = corner_error / corner_valid.sum(1).clamp_min(1)
                checks.append(torch.isfinite(corner_error).all())
                # One small transfer per batch, independent of output pixels.
                statistics = torch.stack((truth, correct, eligible, corner_error,
                                          corner_valid.any(1)), dim=1)
            if not bool(torch.stack(checks).all()):
                raise _error("nonfinite_training", "nonfinite validation EPE", Status.MODEL_ERROR)
            values.extend(means.cpu().tolist())
            if diagnostics is not None and config.architecture == "oriented_surface":
                semantic_rows.extend(statistics.cpu().tolist())
            _check_deadline(deadline)
    if len(values) != len(loader.dataset) or not values:
        raise _error("invalid_dataset", "validation examples are incomplete")
    if diagnostics is not None:
        data = getattr(loader.dataset, "data", loader.dataset)
        records = data.records
        if len(records) != len(values):
            raise _error("invalid_dataset", "validation diagnostic record order differs")
        domains = {}
        for record, value in zip(records, values, strict=True):
            domain = (record.get("kind", "simulated")
                      if record.get("source") == "paper" else "authored")
            domains.setdefault(domain, {}).setdefault(record["document_group"], []).append(value)
        result = {}
        for domain, groups in sorted(domains.items()):
            frames = [v for group in groups.values() for v in group]
            result[domain] = {"images": len(frames), "documents": len(groups),
                              "frame_mean_epe_pixels": math.fsum(frames) / len(frames),
                              "document_mean_epe_pixels": math.fsum(
                                  math.fsum(group) / len(group) for group in groups.values()
                              ) / len(groups)}
        diagnostics.clear()
        diagnostics.update(images=len(values), by_domain=result, quality_gates_evaluated=False)
        if semantic_rows:
            confusion = [[0] * 4 for _ in range(4)]
            corner_errors = []
            for truth, prediction, eligible, error, corner_eligible in semantic_rows:
                if eligible:
                    confusion[int(truth)][int(prediction)] += 1
                if corner_eligible:
                    corner_errors.append(error)
            count = sum(map(sum, confusion))
            diagnostics["semantic"] = {
                "direction_confusion_gt_rows": confusion,
                "direction_eligible": count,
                "direction_missing": len(values) - count,
                "direction_accuracy": (sum(confusion[i][i] for i in range(4)) / count
                                       if count else None),
                "corner_eligible_images": len(corner_errors),
                "corner_missing_images": len(values) - len(corner_errors),
                "corner_mean_epe_pixels": math.fsum(corner_errors) / len(corner_errors)
                if corner_errors else None,
            }
    return math.fsum(values) / len(values)


def _device_info(device):
    if device.type == "cpu":
        return {"device": "cpu", "name": platform.processor() or platform.machine(),
                "capability": None}
    if device.type == "xpu":
        return {"device": str(device), "name": torch.xpu.get_device_name(device),
                "capability": None}
    return {"device": str(device), "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device)}


def _coordinate_datasets(dataset_dir, config):
    """Admit the dataset once and return separate train and validation views.

    ``SceneCoordinateDataset`` validates the whole manifest and every source for
    any split. The validation view is a deep copy with its own records.
    """
    train = SceneCoordinateDataset(
        dataset_dir, "train", config.model_config, config.output_size,
        include_paper_augmentation=config.include_paper_augmentation)
    validation = copy.deepcopy(train)
    validation.split = "validation"
    native = tuple(copy.deepcopy(record) for record in validation._manifest["samples"]
                   if record["split"] == "validation" and record["has_page"] is True)
    if not native or len(native) != train.source_page_counts["authored"]["validation"]:
        raise _error("invalid_dataset", "admitted validation native record count differs")
    paper = []
    if config.include_paper_augmentation:
        for record in validation._manifest["collection"]["paper_augmentation"]["records"]:
            if record["split"] == "validation":
                record = copy.deepcopy(record)
                # Match scene_data._paper_collection: records without ``kind``
                # are simulated.
                record["source_kind"] = record.get("kind", "simulated")
                paper.append(record)
    if len(paper) != train.source_page_counts["paper"]["validation"]:
        raise _error("invalid_dataset", "admitted validation paper record count differs")
    validation._records = native + tuple(paper)
    if len(validation) != validation.page_counts["validation"]:
        raise _error("invalid_dataset", "admitted validation total record count differs")
    return {"train": train, "validation": validation}


def prepare_coordinate_data(dataset_dir, config, *, cache_max_bytes,
                            persistent_cache=None, rebuild_cache=False):
    """Build one bounded train/validation RAM cache for compatible model variants."""
    config = _config(config)
    if type(cache_max_bytes) is not int or cache_max_bytes < 1:
        raise _error("invalid_run_control", "cache_max_bytes must be a positive integer")
    datasets = _coordinate_datasets(dataset_dir, config)
    return cache_scene_datasets(datasets, max_bytes=cache_max_bytes,
                                persistent_path=persistent_cache, rebuild=rebuild_cache)


class _CachedBatchView:
    """Serve each batch with one gather while DataLoader keeps sampling and RNG."""

    def __init__(self, data, include_source_mask, pin_memory, semantic_labels=None):
        self.data, self.include_source_mask = data, include_source_mask
        self.pin_memory = pin_memory
        self.semantic_labels = semantic_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        batch = self.data[index]
        if self.semantic_labels is not None:
            batch = dict(batch)
            batch.update({key: value[0] for key, value in
                          self.semantic_labels.fetch_batch([index]).items()})
        return batch

    def __getitems__(self, indices):
        if type(self.data) is CachedSceneCoordinateDataset:
            batch = self.data.fetch_batch(indices, include_source_mask=self.include_source_mask,
                                          pin_memory=self.pin_memory)
            if self.semantic_labels is not None:
                batch.update(self.semantic_labels.fetch_batch(indices, pin_memory=self.pin_memory))
            return batch
        return [self[index] for index in indices]


def _already_batched(batch):
    return batch


def _data_loader(data, config, shuffle, cuda, include_source_mask, *, training=None):
    # ``cuda`` requests pinned host memory for CUDA and XPU alike.
    if training is None:
        training = data.split == "train"
    options = {"batch_size": config.batch_size, "shuffle": training,
               "num_workers": 0, "generator": shuffle, "pin_memory": cuda}
    labels = (prepare_semantic_page_labels(data)
              if config.architecture == "oriented_surface" else None)
    if type(data) is CachedSceneCoordinateDataset:
        data = _CachedBatchView(data, include_source_mask, cuda, labels)
        options.update(collate_fn=_already_batched, pin_memory=False)
    elif labels is not None:
        data = _CachedBatchView(data, include_source_mask, cuda, labels)
    return DataLoader(data, **options)


def _prepared_data(datasets, directory, config, budget):
    if (not isinstance(datasets, dict) or datasets.keys() != {"train", "validation"}
            or not budget):
        raise _error("invalid_prepared_data", "prepared data needs train/validation and a budget")
    for split, data in datasets.items():
        if (type(data) is not CachedSceneCoordinateDataset or data.split != split
                or data._directory != directory or data.model_config != config.model_config
                or data.output_size != config.output_size
                or data.include_paper_augmentation != config.include_paper_augmentation):
            raise _error("invalid_prepared_data", "prepared cache identity/preprocessing differs")
    if sum(data.cache_bytes for data in datasets.values()) > budget:
        raise _error("cache_budget_exceeded", "prepared data exceeds the cache budget")
    verify_scene_caches(datasets)
    return dict(datasets)


def train_coordinate_study(
    dataset_dir, output_dir, config=None, *, resume=None, initial_checkpoint=None,
    stop_after_epoch=None, max_seconds=None, cache_max_bytes=0, prepared_data=None,
    persistent_cache=None, rebuild_cache=False,
):
    """Train on the configured device; limits stop at the last complete epoch.

    ``stop_after_epoch`` is an absolute epoch number. ``max_seconds`` is a soft
    wall-clock limit checked at batch boundaries; one batch or checkpoint write
    may overrun it. ``initial_checkpoint`` loads model weights only and binds
    their hash into the config; ``resume`` needs that hash and excludes
    ``initial_checkpoint``. ``prepared_data`` reuses the cache from
    ``prepare_coordinate_data``; cached runs hash all artifacts before and after.
    """
    started = perf_counter()
    config = _config(StudyConfig() if config is None else config)
    _run_controls(config, stop_after_epoch, max_seconds)
    if type(cache_max_bytes) is not int or cache_max_bytes < 0:
        raise _error("invalid_run_control", "cache_max_bytes must be a nonnegative integer")
    if (type(rebuild_cache) is not bool or rebuild_cache and persistent_cache is None
            or persistent_cache is not None and (
                not isinstance(persistent_cache, (str, Path)) or not cache_max_bytes
                or prepared_data is not None)):
        raise _error("invalid_run_control",
                     "persistent cache requires a budget and no prepared data")
    device = _resolve_device(config.device)  # Precedes all filesystem/data/optimizer work.
    deadline = None if max_seconds is None else started + max_seconds
    if torch.get_default_dtype() != torch.float32:
        raise _error("invalid_runtime",
                     "training requires the float32 default; it will not change it")
    config, initial_state = _initialization(config, initial_checkpoint, resume)
    directory, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if directory == output or directory in output.parents or output in directory.parents:
        raise _error("invalid_output", "output must be separate from the dataset")
    provenance, info = _provenance(), _device_info(device)
    preparation_started = perf_counter()
    if prepared_data is not None:
        datasets = _prepared_data(prepared_data, directory, config, cache_max_bytes)
    elif cache_max_bytes:
        datasets = prepare_coordinate_data(directory, config, cache_max_bytes=cache_max_bytes,
                                           persistent_cache=persistent_cache,
                                           rebuild_cache=rebuild_cache)
    else:
        datasets = {split: SceneCoordinateDataset(
                    directory, split, config.model_config, config.output_size,
                    include_paper_augmentation=config.include_paper_augmentation)
                for split in ("train", "validation")}
    preparation_seconds = perf_counter() - preparation_started
    train, validation = datasets["train"], datasets["validation"]
    if (train.manifest_sha256 != validation.manifest_sha256
            or train.photo_size != validation.photo_size
            or train.page_counts != validation.page_counts):
        raise _error("invalid_dataset", "train/validation dataset identity differs")
    if resume is None:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise _error("output_exists", "new training requires a new output directory") from error
    lock = output / ".training.lock"
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write(str(os.getpid()) + "\n")
    except FileExistsError as error:
        raise _error("training_locked", "another training run owns this output") from error
    try:
        resumed = None
        if resume is not None:
            last_path = _checkpoint_location(output / "last.pt", strict=True)
            if _checkpoint_location(resume) != last_path:
                raise _error("invalid_resume", "resume must name this output directory's last.pt")
            resumed = coordinate_checkpoint(last_path)
            best, best_digest = _read_checkpoint(last_path.with_name("best.pt"))
            same = {"config": asdict(config), "provenance": provenance, "device_info": info,
                    "manifest_sha256": train.manifest_sha256,
                    "dataset_photo_size": train.photo_size,
                    "dataset_output_size": config.output_size, "dataset_counts": train.page_counts}
            if (resumed["checkpoint_kind"] != "last" or best["checkpoint_kind"] != "best"
                    or any(resumed[k] != v or best[k] != v for k, v in same.items())
                    or resumed["best_checkpoint_sha256"] != best_digest
                    or best["epoch"] != resumed["best_epoch"]
                    or best["history"] != resumed["history"][:best["epoch"]]
                    or best["initial_validation_epe_pixels"]
                    != resumed["initial_validation_epe_pixels"]
                    or best["best_validation_epe_pixels"] != resumed["best_validation_epe_pixels"]):
                raise _error("invalid_resume", "resume identity or linked best checkpoint differs")
            if best["epoch"] == resumed["epoch"] and any(
                not _same_state(best.get(key), resumed.get(key)) for key in (
                    "model_state", "optimizer_state", "torch_rng_state", "shuffle_rng_state",
                    "cuda_rng_state", "xpu_rng_state")
            ):
                raise _error("invalid_resume", "same-epoch best and last states differ")
        cuda = device.type == "cuda"
        xpu = device.type == "xpu"
        with _training_rng(device, config.seed):
            shuffle = torch.Generator(device="cpu").manual_seed(config.seed)
            model = _new_model(config)
            if initial_state is not None:
                model.load_state_dict(initial_state, strict=True)
            model = model.to(device)
            model_spec = _model_spec(config, model)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                         foreach=False, fused=False)
            loaders = {split: _data_loader(data, config, shuffle, cuda or xpu,
                                          split == "train" and bool(
                                              config.page_weight or config.page_relative_weight
                                              or config.background_probability),
                                          training=split == "train")
                       for split, data in datasets.items()}
            epoch_timings = []
            initial, best_epe, best_epoch, best_hash = None, None, None, None
            history, epoch, checkpoint_available = [], 0, resumed is not None
            diagnostics = {} if config.architecture == "oriented_surface" else None
            diagnostic_epoch = None
            validation_options = {"diagnostics": diagnostics} if diagnostics is not None else {}

            def report_progress(stop_reason=None, discarded_epoch=None):
                final_integrity_seconds = None
                if stop_reason is not None and cache_max_bytes:
                    checked_at = perf_counter()
                    verify_scene_caches(datasets)
                    final_integrity_seconds = perf_counter() - checked_at
                report = {
                    "schema_version": CHECKPOINT_SCHEMA, "epochs_completed": epoch,
                    "configured_epochs": config.epochs, "complete": epoch == config.epochs,
                    "best_epoch": best_epoch, "best_validation_epe_pixels": best_epe,
                    "initial_validation_epe_pixels": initial, "history": history,
                    "best_checkpoint_sha256": best_hash, "device": str(device),
                    "cuda_bit_exact_resume_guaranteed": False, "test_evaluated": False,
                    "xpu_bit_exact_resume_guaranteed": False,
                    "run_elapsed_seconds": perf_counter() - started,
                    "epoch_timings": epoch_timings, "stop_reason": stop_reason,
                    "checkpoint_available": checkpoint_available,
                    "discarded_epoch": discarded_epoch,
                    "include_paper_augmentation": config.include_paper_augmentation,
                    "initial_checkpoint_sha256": config.initial_checkpoint_sha256,
                    "initialization_mode": config.initialization_mode,
                    "initial_model_config": (asdict(config.initial_model_config)
                                             if config.initial_model_config is not None else None),
                    "map_mode": config.map_mode, "topology_mode": config.topology_mode,
                    "stop_after_epoch": stop_after_epoch, "max_seconds": max_seconds,
                    "loss_config": {k: getattr(config, k) for k in (
                        "topology_weight", "curvature_weight", "gradient_clip_norm",
                        "page_weight", "boundary_weight", "differential_weight",
                        "page_relative_weight", "orientation_weight",
                        "direction_weight", "corner_weight")},
                    "loss_protocol": provenance["loss_protocol"],
                    "direction_training": config.direction_training,
                    "direction_training_protocol": provenance["direction_training_protocol"],
                    "rotation_probability": config.rotation_probability,
                    "rotation_mode": config.rotation_mode,
                    "background_probability": config.background_probability,
                    "augmentation_protocol": provenance["augmentation_protocol"],
                    "cache_bytes": sum(getattr(data, "cache_bytes", 0)
                                       for data in datasets.values()),
                    "data_preparation_seconds": preparation_seconds,
                    "prepared_cache_reused": prepared_data is not None,
                    "cache_stats": {split: getattr(data, "cache_stats", None)
                                    for split, data in datasets.items()},
                    "cache_integrity": "full_at_run_boundaries_stat_each_epoch"
                                       if cache_max_bytes else None,
                    "final_integrity_seconds": final_integrity_seconds,
                    "validation_diagnostics": ({"epoch": diagnostic_epoch, **diagnostics}
                                               if diagnostics else None),
                    "quality_selection": {"status": "not_evaluated", "best_eligible": None},
                }
                _atomic_bytes(output / "training.json",
                              (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
                progress = {"event": "progress", "epoch": epoch,
                            "validation_epe_pixels": (history[-1]["validation_epe_pixels"]
                                                      if history else initial),
                            "best_epoch": best_epoch, "best_validation_epe_pixels": best_epe,
                            "run_elapsed_seconds": report["run_elapsed_seconds"],
                            "stop_reason": stop_reason, "complete": report["complete"],
                            "discarded_epoch": discarded_epoch}
                print(json.dumps(progress, allow_nan=False), flush=True)
                return report

            if resumed is None:
                epoch_started = perf_counter()
                try:
                    initial = _validation_epe(
                        model, loaders["validation"], config, train.photo_size, device,
                        deadline=deadline, **validation_options,
                    )
                except _DeadlineReached:
                    return report_progress("max_seconds", discarded_epoch=0)
                best_epoch, best_epe = 0, initial
                diagnostic_epoch = 0
                epoch_timings.append({"epoch": 0, "training_seconds": 0.0,
                                      "validation_seconds": perf_counter() - epoch_started})
            else:
                model.load_state_dict(resumed["model_state"], strict=True)
                optimizer.load_state_dict(resumed["optimizer_state"])
                torch.set_rng_state(resumed["torch_rng_state"])
                if cuda:
                    torch.cuda.set_rng_state(resumed["cuda_rng_state"], device)
                elif xpu:
                    torch.xpu.set_rng_state(resumed["xpu_rng_state"], device)
                shuffle.set_state(resumed["shuffle_rng_state"])
                initial = resumed["initial_validation_epe_pixels"]
                history = list(resumed["history"])
                epoch, best_epoch = resumed["epoch"], resumed["best_epoch"]
                best_epe = resumed["best_validation_epe_pixels"]
                best_hash = resumed["best_checkpoint_sha256"]
            while True:
                integrity_started = perf_counter()
                if cache_max_bytes:
                    for data in datasets.values():
                        data.verify_integrity(full=False)
                if (_provenance() != provenance
                        or _sha256(directory / "manifest.json") != train.manifest_sha256):
                    raise _error("training_inputs_changed",
                                 "source/runtime/data changed during training")
                if epoch_timings:
                    epoch_timings[-1]["integrity_seconds"] = perf_counter() - integrity_started
                if resumed is None or epoch != resumed["epoch"]:
                    commit_started = perf_counter()
                    payload = {"schema_version": CHECKPOINT_SCHEMA, "checkpoint_kind": "best",
                               "config": asdict(config), "model_spec": model_spec,
                               "model_state": _cpu_state(model.state_dict()),
                               "optimizer_state": _cpu_state(optimizer.state_dict()),
                               "epoch": epoch,
                               "history": list(history), "best_epoch": best_epoch,
                               "initial_validation_epe_pixels": initial,
                               "best_validation_epe_pixels": best_epe, "provenance": provenance,
                               "manifest_sha256": train.manifest_sha256,
                               "dataset_output_size": config.output_size,
                               "dataset_photo_size": train.photo_size,
                               "dataset_counts": train.page_counts,
                               "torch_rng_state": torch.get_rng_state(),
                               "shuffle_rng_state": shuffle.get_state(),
                               "cuda_rng_state": torch.cuda.get_rng_state(device) if cuda else None,
                               "xpu_rng_state": torch.xpu.get_rng_state(device) if xpu else None,
                               "device_info": info,
                               "best_checkpoint_sha256": None}
                    best_hash = _commit_checkpoints(output, payload)
                    checkpoint_available = True
                    epoch_timings[-1].update(checkpoint_seconds=perf_counter() - commit_started,
                                             epoch_seconds=perf_counter() - epoch_started)
                stop_reason = None
                if epoch == config.epochs:
                    stop_reason = "completed"
                elif stop_after_epoch is not None and epoch >= stop_after_epoch:
                    stop_reason = "stop_after_epoch"
                elif deadline is not None and perf_counter() >= deadline:
                    stop_reason = "max_seconds"
                report = report_progress(stop_reason)
                if stop_reason is not None:
                    return report
                model.train()
                losses, pixel_counts = [], []
                epoch_started = last_progress = perf_counter()
                print(json.dumps({"event": "epoch_started", "epoch": epoch + 1,
                                  "device": str(device)}), flush=True)
                try:
                    for batch_number, batch in enumerate(loaders["train"], 1):
                        _check_deadline(deadline)
                        loss, pixels = _training_step(
                            model, optimizer, batch, config, train.photo_size, device,
                            _defer_metrics=True,
                        )
                        losses.append(loss)
                        pixel_counts.append(pixels)
                        _check_deadline(deadline)
                        if perf_counter() - last_progress >= 30:
                            print(json.dumps({"event": "training_batch", "epoch": epoch + 1,
                                              "batch": batch_number,
                                              "batches": len(loaders["train"]),
                                              "run_elapsed_seconds": perf_counter() - started}),
                                  flush=True)
                            last_progress = perf_counter()
                    weight = sum(pixel_counts)
                    if not weight:
                        raise _error("invalid_dataset", "empty training split")
                    loss_sum = 0.0
                    for loss, pixels in zip(torch.stack(losses).cpu().tolist(), pixel_counts,
                                            strict=True):
                        loss_sum += loss * pixels
                    training_seconds = perf_counter() - epoch_started
                    validation_started = perf_counter()
                    metric = _validation_epe(
                        model, loaders["validation"], config, train.photo_size, device,
                        deadline=deadline, **validation_options,
                    )
                except _DeadlineReached:
                    return report_progress("max_seconds", discarded_epoch=epoch + 1)
                epoch += 1
                diagnostic_epoch = epoch
                history.append({"epoch": epoch, "train_loss": loss_sum / weight,
                                "validation_epe_pixels": metric})
                epoch_timings.append({"epoch": epoch, "training_seconds": training_seconds,
                                      "validation_seconds": perf_counter() - validation_started})
                if metric < best_epe:
                    best_epoch, best_epe = epoch, metric
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-device", help="check the explicit device without training")
    check.add_argument("--device", default="cuda")
    inspect = commands.add_parser("inspect", help="validate a trusted checkpoint without a GPU")
    inspect.add_argument("checkpoint")
    train = commands.add_parser("train", help="explicit-device training; no automatic fallback")
    train.add_argument("dataset")
    train.add_argument("output")
    train.add_argument("--device", default="cuda")
    train.add_argument("--architecture",
                       choices=("global", "spatial", "page", "projective", "surface",
                                "oriented_surface"),
                       default="global")
    train.add_argument("--map-mode", choices=("legacy", "linear", "analytic"), default="legacy")
    train.add_argument("--topology-mode", choices=("absolute", "relative"), default="absolute")
    train.add_argument("--initialization-mode", choices=("full", "backbone", "resolution"),
                       default="full")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=20260925)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--input-size", type=int, nargs=2, default=(96, 72), metavar=("H", "W"))
    train.add_argument("--grid-size", type=int, nargs=2, default=(17, 13), metavar=("H", "W"))
    train.add_argument("--topology-weight", type=float, default=0.0)
    train.add_argument("--curvature-weight", type=float, default=0.0)
    train.add_argument("--gradient-clip-norm", type=float, default=0.0)
    train.add_argument("--page-weight", type=float, default=0.0)
    train.add_argument("--boundary-weight", type=float, default=0.0)
    train.add_argument("--differential-weight", type=float, default=0.0)
    train.add_argument("--page-relative-weight", type=float, default=0.0)
    train.add_argument("--orientation-weight", type=float, default=0.0)
    train.add_argument("--direction-weight", type=float, default=0.0)
    train.add_argument("--corner-weight", type=float, default=0.0)
    train.add_argument("--direction-training", choices=("predicted", "teacher"),
                       default="predicted")
    train.add_argument("--rotation-probability", type=float, default=0.0)
    train.add_argument("--rotation-mode", choices=("batch", "per_image_fit"), default="batch")
    train.add_argument("--background-probability", type=float, default=0.0)
    train.add_argument("--include-paper-augmentation", action="store_true")
    train.add_argument("--initial-checkpoint")
    train.add_argument("--cache-mib", type=int, default=0)
    train.add_argument("--persistent-cache")
    train.add_argument("--rebuild-cache", action="store_true")
    train.add_argument("--resume")
    train.add_argument("--stop-after-epoch", type=int)
    train.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-device":
            result = _device_info(_resolve_device(args.device))
        elif args.command == "inspect":
            saved = coordinate_checkpoint(args.checkpoint)
            result = {k: saved[k] for k in ("schema_version", "checkpoint_kind", "epoch",
                                           "best_epoch", "model_spec", "device_info", "config")}
        else:
            config = StudyConfig(architecture=args.architecture, device=args.device, seed=args.seed,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 learning_rate=args.learning_rate,
                                 model_config=ModelConfig(input_size=tuple(args.input_size),
                                                          grid_size=tuple(args.grid_size)),
                                 topology_weight=args.topology_weight,
                                 curvature_weight=args.curvature_weight,
                                 gradient_clip_norm=args.gradient_clip_norm,
                                 page_weight=args.page_weight,
                                 boundary_weight=args.boundary_weight,
                                 differential_weight=args.differential_weight,
                                 page_relative_weight=args.page_relative_weight,
                                 orientation_weight=args.orientation_weight,
                                 direction_weight=args.direction_weight,
                                 corner_weight=args.corner_weight,
                                 direction_training=args.direction_training,
                                 rotation_probability=args.rotation_probability,
                                 rotation_mode=args.rotation_mode,
                                 background_probability=args.background_probability,
                                 include_paper_augmentation=args.include_paper_augmentation,
                                 map_mode=args.map_mode, topology_mode=args.topology_mode,
                                 initialization_mode=args.initialization_mode)
            if args.resume is not None and args.initial_checkpoint is None:
                _config(config)
                _resolve_device(config.device)
                saved = coordinate_checkpoint(args.resume)
                config = replace(
                    config,
                    initial_checkpoint_sha256=saved["config"].get("initial_checkpoint_sha256"),
                    initial_model_config=(
                        ModelConfig(**saved["config"]["initial_model_config"])
                        if saved["config"].get("initial_model_config") is not None else None),
                )
            report = train_coordinate_study(
                args.dataset, args.output, config, resume=args.resume,
                initial_checkpoint=args.initial_checkpoint,
                stop_after_epoch=args.stop_after_epoch, max_seconds=args.max_seconds,
                cache_max_bytes=args.cache_mib * 1024**2,
                persistent_cache=args.persistent_cache, rebuild_cache=args.rebuild_cache,
            )
            result = {k: report[k] for k in (
                "epochs_completed", "configured_epochs", "complete", "best_epoch",
                "best_validation_epe_pixels", "run_elapsed_seconds", "stop_reason",
                "checkpoint_available", "discarded_epoch", "device",
                "include_paper_augmentation", "initial_checkpoint_sha256",
                "initialization_mode", "map_mode", "topology_mode",
            )}
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, TypeError) as error:
        print(json.dumps({"status": Status.INPUT_ERROR, "reason_code": "invalid_config",
                          "error": "invalid_config", "cpu_fallback": False,
                          "message": str(error)}), flush=True)
        return 2
    except FlattenerError as error:
        print(json.dumps({"status": error.status, "reason_code": error.reason_code,
                          "error": error.reason_code, "cpu_fallback": False,
                          "message": error.message}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
