"""Step-based dewarp trainer: ``python -m flattener.dewarp.train --config run.json``.

AMP fp16 with GradScaler on CUDA, AdamW, warmup + cosine or UVDoc-style linear decay,
gradient clipping, deterministic data order and augmentation (keys derive from the
step, so resume replays the exact stream) and atomic checkpoints (see ``checkpoint``).
Loader workers return uint8 images; float conversion and colour jitter run per batch
on the training device. The DataLoader setup is Windows-spawn safe: top-level
picklable datasets, no lambdas, ``__main__`` guard.

Validation runs in fp32 over the ``val_views`` (see ``data.ViewConfig`` and
``DEFAULT_VIEWS``: tight and DocUNet-like page crops, the full frame, quarter turns and a
small page on a canvas) and logs a grid endpoint error per view in pixels of
``epe_reference_size``. One view picks the best checkpoint: ``val_select``, or the
first view (``page`` by default); the others are logged only, and the best records
which view chose it. Samples without a valid target node are counted as
``not_evaluable`` and left out; a validation with nothing to evaluate picks no best.
The best metric carries a signature of everything that defines validation; a resume
with a different signature starts a new best. Every new best is saved.

Loss (UVDoc ``train.py`` defaults): 5 * L1(2-D grid) + 5 * L1(3-D grid, only
samples that have one) + optional reconstruction L1 (UVDoc: 1.0 from epoch 10
of 21), all masked by grid-node validity.

``examples/dewarp/train-research-uvdoc-recipe.json`` maps the upstream
schedule to steps: batch 8, two optimizer steps (Doc3D, UVDoc) per mixDataset
batch over the 74,527 Doc3D ``traindoc.txt`` ids = 2 * 9,316 steps per epoch,
21 epochs = 391,272 steps; reconstruction from epoch 10 (step 186,320);
constant lr, then linear decay from about epoch 10.5 (``decay_start`` 0.5);
rotation (p 0.5, 30 deg) only on UVDoc samples. Remaining differences:
batches mix sources (upstream alternates single-source batches), AMP fp16,
fractional crop margins, approximate appearance augmentation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..contracts import FlattenerError, Status
from ..evidence import evidence
from .checkpoint import (
    CHECKPOINT_SCHEMA,
    checkpoint_sha256,
    code_fingerprint,
    commit_checkpoint,
    environment,
    json_safe,
    latest_checkpoint,
    load_checkpoint,
    retire_best,
    utc_now,
    write_json,
)
from .coords import unwarp_torch
from .data import (
    TRACKS,
    AugmentConfig,
    DataSpec,
    DewarpDataset,
    ViewConfig,
    build_dataset,
    canonical_json,
    expand_path,
    require_track,
    target_code_sha256,
    to_float_images,
)
from .model import (
    UVDOC_WEIGHTS_SHA256,
    DewarpModelConfig,
    UVDocNet,
    count_parameters,
    load_uvdoc_weights,
)
from .predict import ieee_float32

SCHEDULES = ("cosine", "linear_decay", "constant")
INITS = ("scratch", "uvdoc", "checkpoint")
# Fields that may change when resuming; everything else must match the checkpoint.
RESUMABLE_CHANGES = frozenset(
    {
        "output_dir",
        "data_spec",
        "init",
        "init_path",
        "max_steps",
        "max_seconds",
        "num_workers",
        "prefetch_factor",
        "pin_memory",
        "device",
        "channels_last",
        "log_every",
        "val_every",
        "val_max_batches",
        "val_batch_size",
        "val_views",
        "val_select",
        "checkpoint_every",
        "keep_checkpoints",
    }
)
EPE_REFERENCE_SIZE = (712, 488)  # UVDoc input size (height, width)
# "page": the page box with a fixed margin (the primary view unless val_select says
# otherwise); "docunet": uneven 2-10% margins like DocUNet crops, the input condition of
# ``predict`` on benchmark crops; "full_frame": the whole image with its background;
# "rot90"/"rot270": the docunet crop turned a quarter counter-clockwise/clockwise;
# "small_page": the docunet crop shrunk to 10-25% of a textured canvas.
DOCUNET_MARGIN = (0.02, 0.1)
DEFAULT_VIEWS = (
    ViewConfig("page", "page", (0.025, 0.025)),
    ViewConfig("docunet", "page", DOCUNET_MARGIN),
    ViewConfig("full_frame", "full_frame"),
    ViewConfig("rot90", "page", DOCUNET_MARGIN, rotation=90),
    ViewConfig("rot270", "page", DOCUNET_MARGIN, rotation=270),
    ViewConfig("small_page", "small_page", DOCUNET_MARGIN),
)
CSV_COLUMNS = (
    "kind",
    "track",
    "step",
    "view",
    "lr",
    "loss",
    "loss_grid2d",
    "loss_grid3d",
    "loss_recon",
    "grad_norm",
    "skipped",
    "overflow_steps",
    "images_per_sec",
    "data_fraction",
    "epe_px",
    "not_evaluable",
    "max_memory_gib",
    "max_reserved_gib",
)


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason, message)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    data_spec: str
    output_dir: str
    track: str
    model: DewarpModelConfig = field(default_factory=DewarpModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    init: str = "scratch"
    init_path: str | None = None
    seed: int = 0
    batch_size: int = 16
    total_steps: int = 100_000
    max_steps: int | None = None
    max_seconds: float | None = None
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    schedule: str = "cosine"
    warmup_steps: int = 500
    decay_start: float = 0.5
    min_lr_ratio: float = 0.0
    grad_clip: float = 1.0
    amp: bool = True
    loss_grid2d: float = 5.0
    loss_grid3d: float = 5.0
    loss_recon: float = 0.0
    recon_start_step: int = 0
    num_workers: int = 6
    prefetch_factor: int = 4
    pin_memory: bool = True
    device: str = "auto"
    # Training memory layout: channels_last suits RTX 50-series GPUs but ran ~4x slower
    # on an RTX 3060 Ti. Checkpoints hold contiguous (NCHW) weights either way.
    channels_last: bool = False
    deterministic: bool = False
    log_every: int = 50
    val_every: int = 2000
    val_max_batches: int | None = None
    val_batch_size: int | None = None  # None: batch_size
    val_views: tuple[ViewConfig, ...] = DEFAULT_VIEWS
    val_select: str | None = None  # view that picks the best checkpoint; None: the first
    checkpoint_every: int = 2000
    keep_checkpoints: int = 2
    # Fixed pixel size for the validation EPE, so runs at any input size compare.
    epe_reference_size: tuple[int, int] = EPE_REFERENCE_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        if isinstance(self.epe_reference_size, list):
            object.__setattr__(self, "epe_reference_size", tuple(self.epe_reference_size))
        try:
            views = tuple(
                v if isinstance(v, ViewConfig) else ViewConfig.from_dict(v) for v in self.val_views
            )
        except (TypeError, ValueError) as error:
            raise _error("invalid_config", f"val_views: {error}") from error
        object.__setattr__(self, "val_views", views)
        names = [view.name for view in views]
        if not names or len(set(names)) != len(names):
            raise _error("invalid_config", "val_views needs views with unique names")
        if self.val_select is not None and self.val_select not in names:
            raise _error("invalid_config", f"val_select must be one of {names}")
        if self.track not in TRACKS or self.schedule not in SCHEDULES or self.init not in INITS:
            raise _error("invalid_config", f"track {TRACKS}, schedule {SCHEDULES}, init {INITS}")
        if self.init != "scratch" and not self.init_path:
            raise _error("invalid_config", f"init {self.init!r} needs init_path")
        if self.model.grid_convention != "flattener":
            raise _error("invalid_config", "training targets use grid_convention 'flattener'")
        integers = {
            "seed": 0,
            "batch_size": 1,
            "total_steps": 1,
            "warmup_steps": 0,
            "recon_start_step": 0,
            "num_workers": 0,
            "prefetch_factor": 1,
            "log_every": 1,
            "val_every": 0,
            "checkpoint_every": 0,
            "keep_checkpoints": 1,
        }
        for name, low in integers.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise _error("invalid_config", f"{name} must be an integer >= {low}")
        for name in ("max_steps", "val_max_batches", "val_batch_size"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise _error("invalid_config", f"{name} must be a positive integer or null")
        numbers = (
            "lr",
            "weight_decay",
            "decay_start",
            "min_lr_ratio",
            "grad_clip",
            "loss_grid2d",
            "loss_grid3d",
            "loss_recon",
        )
        for name in numbers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
                raise _error("invalid_config", f"{name} must be a non-negative number")
        if not 0 < self.lr <= 1 or self.decay_start > 1 or self.min_lr_ratio > 1:
            raise _error("invalid_config", "lr in (0, 1]; decay_start and min_lr_ratio <= 1")
        if self.max_seconds is not None and not self.max_seconds > 0:
            raise _error("invalid_config", "max_seconds must be positive or null")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise _error("invalid_config", "betas must be two numbers in [0, 1)")
        flags = ("amp", "pin_memory", "channels_last", "deterministic")
        if not all(isinstance(getattr(self, k), bool) for k in flags):
            raise _error("invalid_config", f"{', '.join(flags)} must be booleans")
        reference = self.epe_reference_size
        if (
            not isinstance(reference, tuple)
            or len(reference) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in reference)
        ):
            raise _error("invalid_config", "epe_reference_size must be (height, width)")

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            elif item.name == "val_views":
                value = [view.to_dict() for view in value]
            result[item.name] = list(value) if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        if not isinstance(data, dict):
            raise _error("invalid_config", "train config must be a JSON object")
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise _error("invalid_config", f"unknown train config fields: {sorted(unknown)}")
        values = dict(data)
        values["model"] = DewarpModelConfig.from_dict(values.get("model", {}))
        values["augment"] = AugmentConfig(**values.get("augment", {}))
        return cls(**values)

    @property
    def stop_step(self) -> int:
        return min(self.max_steps or self.total_steps, self.total_steps)

    @property
    def selected_view(self) -> str:
        return self.val_select or self.val_views[0].name


def validation_signature(config: TrainConfig, val_sources: list[dict[str, Any]]) -> str:
    """sha256 of everything that defines a validation result: views, selection, sample
    limit, reference size, model sizes and the validation data (ids and contents)."""
    batch = config.val_batch_size or config.batch_size
    body = {
        "views": [view.to_dict() for view in config.val_views],
        "select": config.selected_view,
        "sample_limit": None if config.val_max_batches is None else config.val_max_batches * batch,
        "epe_reference_size": list(config.epe_reference_size),
        "input_size": list(config.model.input_size),
        "grid_size": list(config.model.grid_size),
        "data": val_sources,
    }
    return hashlib.sha256(canonical_json(body)).hexdigest()


def carry_best(
    best: dict[str, Any] | None, signature: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A resumed best and the note of a dropped one.

    The best carries over when it was measured under ``signature``; otherwise the run
    starts a new best and the note records the old step and EPE.
    """
    if best is None or best.get("signature") == signature:
        return best, None
    return None, {"step": best.get("step"), "epe_px": best.get("epe_px")}


def improves(best: dict[str, Any] | None, epe: float | None) -> bool:
    """Whether ``epe`` beats the best; a missing EPE (nothing evaluable) never does."""
    if epe is None or not math.isfinite(epe):
        return False
    return best is None or best.get("epe_px") is None or epe < best["epe_px"]


def save_due(
    step: int, improved: bool, saved_step: int, checkpoint_every: int, *, last: bool = False
) -> bool:
    """Save every new best (so best.json names existing weights), every
    ``checkpoint_every`` steps and the last step once."""
    if improved:
        return True
    if last:
        return saved_step != step
    return bool(checkpoint_every) and step % checkpoint_every == 0


def learning_rate(config: TrainConfig, step: int) -> float:
    """Learning rate for the optimizer step with index ``step`` (0-based)."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    floor = config.min_lr_ratio
    if config.schedule == "cosine":
        progress = min(
            1.0, (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
        )
        factor = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    elif config.schedule == "linear_decay":
        start = max(config.warmup_steps, int(config.decay_start * config.total_steps))
        progress = min(1.0, max(0, step - start) / max(1, config.total_steps - start))
        factor = 1 - (1 - floor) * progress
    else:
        factor = 1.0
    return config.lr * factor


class StepSampler(torch.utils.data.Sampler):
    """Infinite deterministic key stream ``(flat_index, position)`` from ``start``.

    Without weights each epoch is a seeded permutation of the concatenated
    sources; with weights every position draws a source by weight and a
    sample uniformly (with replacement).
    """

    def __init__(self, sizes: list[int], weights: list[float] | None, seed: int, start: int = 0):
        self.sizes, self.weights = list(sizes), weights
        self.seed, self.start = int(seed), int(start)
        self.offsets = np.cumsum([0] + self.sizes)

    def __iter__(self):
        total, position, epoch, order = int(self.offsets[-1]), self.start, -1, None
        while True:
            if self.weights is None:
                if position // total != epoch:
                    epoch = position // total
                    order = np.random.default_rng([self.seed, 1, epoch]).permutation(total)
                yield int(order[position % total]), position
            else:
                rng = np.random.default_rng([self.seed, 2, position])
                source = int(rng.choice(len(self.sizes), p=self.weights))
                yield int(self.offsets[source] + rng.integers(self.sizes[source])), position
            position += 1


def worker_init(worker_id: int) -> None:
    """Avoid CPU oversubscription: one torch thread per loader worker."""
    torch.set_num_threads(1)


def _loader(
    dataset: DewarpDataset,
    config: TrainConfig,
    sampler,
    device: torch.device,
    workers: int,
    *,
    batch_size: int,
    drop_last: bool,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        drop_last=drop_last,
        persistent_workers=workers > 0,
        prefetch_factor=config.prefetch_factor if workers > 0 else None,
        worker_init_fn=worker_init if workers > 0 else None,
    )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(target)
    return ((prediction - target).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def compute_losses(
    config: TrainConfig,
    step: int,
    batch: dict[str, Any],
    grid2d: torch.Tensor,
    grid3d: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    valid = batch["valid"]
    losses = {"loss_grid2d": _masked_l1(grid2d.float(), batch["grid2d"], valid)}
    total = config.loss_grid2d * losses["loss_grid2d"]
    if grid3d is not None and config.loss_grid3d > 0 and bool(batch["has_grid3d"].any()):
        mask = valid * batch["has_grid3d"].to(valid.dtype).view(-1, 1, 1, 1)
        losses["loss_grid3d"] = _masked_l1(grid3d.float(), batch["grid3d"], mask)
        total = total + config.loss_grid3d * losses["loss_grid3d"]
    complete = valid.flatten(1).all(dim=1)
    if config.loss_recon > 0 and step >= config.recon_start_step and bool(complete.any()):
        image, size = batch["image"][complete], config.model.input_size
        predicted = unwarp_torch(image, grid2d.float()[complete], size)
        with torch.no_grad():
            target = unwarp_torch(image, batch["grid2d"][complete], size)
        losses["loss_recon"] = F.l1_loss(predicted, target)
        total = total + config.loss_recon * losses["loss_recon"]
    losses["loss"] = total
    return losses


def _layout(images: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    """``images`` in the training memory layout (``TrainConfig.channels_last``)."""
    if config.channels_last:
        return images.contiguous(memory_format=torch.channels_last)
    return images


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Tensors to ``device``; noise seeds stay on the CPU (read as Python ints)."""
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor) and key != "noise_seed"
        else value
        for key, value in batch.items()
    }


MEMORY_KEYS = ("max_memory_gib", "max_reserved_gib", "device_used_gib")


def memory_stats(device: torch.device) -> dict[str, float]:
    """Peak allocated and reserved CUDA memory since start, and current device use (GiB)."""
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "device_used_gib": (total - free) / 2**30,
    }


@torch.no_grad()
def validate(model: UVDocNet, loader, config: TrainConfig, device: torch.device) -> dict[str, Any]:
    """Mean grid endpoint error per view in pixels of ``epe_reference_size``.

    The EPE pools every valid target node of the view. A sample with no valid node in a
    view cannot be evaluated there: it adds nothing to that view's means and is counted
    in ``not_evaluable``. A view or source without an evaluable sample has ``epe_px``
    ``None``, which ``improves`` never selects. ``loss_grid2d`` is the masked L1 pooled
    the same way.

    The forward runs NCHW in fp32 without TF32, like ``predict``: fp32 channels_last
    convolutions ran ~27x slower on an RTX 5090, so a ``channels_last`` model is
    converted for the validation and back. The CUDA cache is emptied before and after,
    so validation batches reuse the training memory; the memory figures include the
    validation peak.
    """
    model.to(memory_format=torch.contiguous_format)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    height, width = config.epe_reference_size
    scale = torch.tensor([width / 2.0, height / 2.0], device=device).view(1, 2, 1, 1)
    names = [view.name for view in config.val_views]
    # per view and source: error sum, valid nodes, evaluable samples, not evaluable samples
    sums: dict[str, dict[str, list[float]]] = {name: {} for name in names}
    losses = {name: [0.0, 0.0] for name in names}  # masked L1 sum and weight
    batches = samples = 0
    model.eval()
    with ieee_float32():
        for batch in loader:
            batch = _to_device(batch, device)
            for index, name in enumerate(names):
                grid2d, _ = model(batch["image"][:, index].float().div_(255))
                target, valid = batch["grid2d"][:, index], batch["valid"][:, index]
                weight = valid.expand_as(target)
                losses[name][0] += float(((grid2d - target).abs() * weight).sum())
                losses[name][1] += float(weight.sum())
                error = torch.linalg.vector_norm((grid2d - target) * scale, dim=1) * valid[:, 0]
                totals = torch.stack([error.flatten(1).sum(1), valid.flatten(1).sum(1)], 1)
                for source, (error_sum, count) in zip(batch["source"], totals.tolist()):
                    entry = sums[name].setdefault(source, [0.0, 0.0, 0, 0])
                    if count > 0:
                        entry[0] += error_sum
                        entry[1] += count
                        entry[2] += 1
                    else:
                        entry[3] += 1
            batches += 1
            samples += len(batch["source"])
            if config.val_max_batches is not None and batches >= config.val_max_batches:
                break
    model.train()
    if config.channels_last:
        model.to(memory_format=torch.channels_last)
    memory = memory_stats(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    views = {}
    for name in names:
        entries = sums[name]
        total, count = (sum(entry[i] for entry in entries.values()) for i in (0, 1))
        views[name] = {
            "epe_px": total / count if count > 0 else None,
            "loss_grid2d": losses[name][0] / losses[name][1] if losses[name][1] > 0 else None,
            "evaluable": int(sum(entry[2] for entry in entries.values())),
            "not_evaluable": int(sum(entry[3] for entry in entries.values())),
            "per_source_epe_px": {
                source: entry[0] / entry[1] if entry[1] > 0 else None
                for source, entry in sorted(entries.items())
            },
            "per_source_not_evaluable": {
                source: int(entry[3]) for source, entry in sorted(entries.items())
            },
        }
    selected = config.selected_view
    return {
        "epe_px": views[selected]["epe_px"],
        "view": selected,
        "not_evaluable": views[selected]["not_evaluable"],
        "views": views,
        "reference_size": [height, width],
        "batches": batches,
        "samples": samples,
        **memory,
    }


def _amp(config: TrainConfig, device: torch.device) -> bool:
    return config.amp and device.type == "cuda"


def _autocast(config: TrainConfig, device: torch.device):
    if not _amp(config, device):
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.float16)


def restore_scaler(scaler: torch.amp.GradScaler, state: dict[str, Any]) -> bool:
    """Load a saved GradScaler state; False when an enabled scaler starts fresh
    because the checkpoint was saved without AMP (cpu/xpu run resumed on CUDA)."""
    if state:
        scaler.load_state_dict(state)
        return True
    return not scaler.is_enabled()


def _parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay = [p for p in model.parameters() if p.ndim > 1]
    other = [p for p in model.parameters() if p.ndim <= 1]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}]


def _initialize(model: UVDocNet, config: TrainConfig) -> dict[str, Any]:
    """Load the initial weights and return their lineage record.

    The weights do not depend on the input or grid size, so a checkpoint trained at
    another size initializes this model; the record keeps the parent's model config.
    """
    if config.init == "scratch":
        return {"kind": "scratch", "seed": config.seed}
    if config.init == "uvdoc":
        try:
            lineage = load_uvdoc_weights(model, config.init_path)
        except ValueError as error:
            raise _error("invalid_init", str(error)) from error
        if config.track == "release" and not lineage["sha256_matches_upstream"]:
            raise _error(
                "track_violation",
                f"release run needs the published UVDoc weights (sha256 {UVDOC_WEIGHTS_SHA256}); "
                f"{config.init_path} has {lineage['sha256']}",
            )
        return lineage
    parent = load_checkpoint(config.init_path)  # sha256 verified against its sidecar
    if config.track == "release" and parent.get("track") != "release":
        raise _error(
            "track_violation", f"release run cannot start from {parent.get('track')} weights"
        )
    state = parent["model_state"]
    if model.out_point_positions3D is None:
        state = {k: v for k, v in state.items() if not k.startswith("out_point_positions3D.")}
    model.load_state_dict(state)
    return {
        "kind": "checkpoint",
        "path": str(config.init_path),
        "sha256": checkpoint_sha256(config.init_path),
        "step": parent["step"],
        "track": parent["track"],
        "model_config": parent.get("model_config"),
        "data_spec_sha256": parent["data_spec_sha256"],
        "code_sha256": parent["code"]["sha256"],
        "lineage": parent["lineage"],
    }


class _Logger:
    """JSONL (every field, tagged with the license track) plus a fixed-column CSV.

    An existing CSV keeps its own header, so rows stay aligned when the columns change.
    """

    def __init__(self, directory: Path, track: str) -> None:
        self.jsonl, self.csv = directory / "train_log.jsonl", directory / "train_log.csv"
        self.track = track

    def __call__(self, record: dict[str, Any]) -> None:
        record = json_safe({**record, "track": self.track})  # strict JSON: inf/nan -> null
        with self.jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        columns: list[str] | None = None
        if self.csv.is_file():
            with self.csv.open(encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream), None)
        with self.csv.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=columns or CSV_COLUMNS, extrasaction="ignore"
            )
            if not columns:
                writer.writeheader()
            writer.writerow(record)
        print(json.dumps({k: record[k] for k in CSV_COLUMNS if k in record}), flush=True)


def code_record() -> dict[str, Any]:
    """The checkpoint's ``code``: file hashes (``code_fingerprint``) and the target-code
    digest (``target_code_sha256``)."""
    return {**code_fingerprint(), "targets_sha256": target_code_sha256()}


def _check_resume(
    config: TrainConfig,
    payload: dict[str, Any],
    spec: DataSpec,
    sources: dict[str, Any],
    code: dict[str, Any],
) -> None:
    """Refuse a resume whose config, track, data spec, indexed samples, target
    conventions (``DewarpDataset.describe``) or target code (``target_code_sha256``)
    changed. A checkpoint without the target-code digest must have the same
    ``coords.py`` bytes."""
    saved = payload["train_config"]
    current = config.to_dict()
    changed = sorted(
        k for k in current if k not in RESUMABLE_CHANGES and current[k] != saved.get(k)
    )
    if payload["track"] != config.track:
        changed.append("track")
    if payload["data_spec_sha256"] != spec.sha256:
        changed.append("data spec")
    saved_sources = payload.get("data_sources") or {}
    if any("targets_sha256" not in source for split in saved_sources.values() for source in split):
        changed.append("data sources (the checkpoint predates the target-convention signature)")
    elif saved_sources != sources:  # re-converted, re-split, relabelled or reinterpreted data
        changed.append("data sources (samples, license classes or target conventions)")
    saved_code = payload.get("code") or {}
    if "targets_sha256" in saved_code:
        if saved_code["targets_sha256"] != code["targets_sha256"]:
            changed.append("target code (coords.py, grid readers, loaders or crop mapping)")
    elif (saved_code.get("files") or {}).get("coords.py") != code["files"].get("coords.py"):
        changed.append("coords.py (the checkpoint predates the target-code digest)")
    if changed:
        raise _error("resume_mismatch", f"resume differs in {changed}; use a new output_dir")


def _data_sources(train_set: DewarpDataset, val_set: DewarpDataset | None) -> dict[str, Any]:
    """Per-split source manifests; train and val must not share a sample."""
    if val_set is not None:
        shared = train_set.sample_keys() & val_set.sample_keys()
        if shared:
            examples = [f"{root}:{sample}" for root, sample in sorted(shared)[:3]]
            raise _error(
                "invalid_data",
                f"{len(shared)} samples are in both train and val (e.g. {examples}); "
                "give manifest entries an explicit split or use separate roots/ids files",
            )
    return {"train": train_set.describe(), "val": [] if val_set is None else val_set.describe()}


def train(config: TrainConfig, *, resume: str = "auto") -> dict[str, Any]:
    """Train until ``stop_step`` or ``max_seconds``; return a summary.

    ``resume``: ``"auto"`` (latest checkpoint in the output directory, if any),
    ``"none"`` (refuse a directory that has checkpoints) or a checkpoint path.
    Every step consumes exactly ``batch_size`` keys (also a skipped non-finite
    step), so resuming at step s replays the stream from key s * batch_size.
    """
    started = time.perf_counter()
    output = Path(config.output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    step = 0
    lineage: list[dict[str, Any]] | None = None

    def own(measurement: str) -> dict[str, Any]:
        """Evidence of this run's model at the current step."""
        return evidence(measurement, {"kind": "checkpoint", "step": step, "lineage": lineage})

    def status(state: str, **extra: Any) -> None:
        record = {
            "state": state,
            "step": step,
            "track": config.track,
            "pid": os.getpid(),
            "updated_utc": utc_now(),
            "evidence": own("training_loss"),
        }
        try:  # best effort: the run continues while a monitor holds the file
            write_json(output / "status.json", {**record, **extra})
        except OSError as error:
            print(f"warning: status.json not updated: {error}", file=sys.stderr, flush=True)

    status("starting")
    try:
        spec = DataSpec.load(config.data_spec)
        require_track(spec, config.track)
        device = resolve_device(config.device)
        torch.backends.cudnn.benchmark = not config.deterministic
        if config.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        dataset_args = dict(
            track=config.track,
            input_size=config.model.input_size,
            grid_size=config.model.grid_size,
            augment=config.augment,
            seed=config.seed,
        )
        train_set = build_dataset(spec, split="train", train=True, **dataset_args)
        if train_set is None:
            raise _error("missing_data", "data spec has no train sources")
        val_set = build_dataset(
            spec, split="val", train=False, views=config.val_views, **dataset_args
        )
        data_sources = _data_sources(train_set, val_set)
        signature = validation_signature(config, data_sources["val"])

        if resume == "auto":
            resume_path = latest_checkpoint(checkpoints)
        elif resume == "none":
            resume_path = None
            if latest_checkpoint(checkpoints) is not None:
                raise _error(
                    "output_exists", f"{checkpoints} has checkpoints; resume or change dir"
                )
        else:
            resume_path = Path(resume)

        torch.manual_seed(config.seed)
        model = UVDocNet(config.model)
        code = code_record()
        best: dict[str, Any] | None = None
        resumes: list[dict[str, Any]] = []
        counters = {"skipped": 0, "overflow_steps": 0}
        payload = None
        if resume_path is not None:
            payload = load_checkpoint(resume_path)
            _check_resume(config, payload, spec, data_sources, code)
            model.load_state_dict(payload["model_state"])
            lineage, step, best = payload["lineage"], payload["step"], payload.get("best")
            counters.update(payload.get("counters", {}))
            resumes = [
                *payload.get("resumes", []),
                {"step": step, "code_sha256": code["sha256"], "utc": utc_now()},
            ]
            best, dropped = carry_best(best, signature)
            if dropped is not None:
                resumes[-1]["best_reset"] = dropped
                note = {"resumed_step": step, "reason": "validation signature changed"}
                retire_best(checkpoints, step, note)
        else:
            lineage = [
                _initialize(model, config),
                {
                    "kind": "train",
                    "track": config.track,
                    "data_spec_sha256": spec.sha256,
                    "code_sha256": code["sha256"],
                    "started_utc": utc_now(),
                },
            ]
        model.to(device).train()
        if config.channels_last:
            model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(
            _parameter_groups(model, config.weight_decay), lr=config.lr, betas=config.betas
        )
        scaler = torch.amp.GradScaler(device.type, enabled=_amp(config, device))
        if payload is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
            if not restore_scaler(scaler, payload["scaler_state"]):
                resumes[-1]["scaler"] = "fresh: checkpoint was saved without AMP"
            torch.set_rng_state(payload["rng_state"]["torch"])
            if device.type == "cuda" and payload["rng_state"]["cuda"]:
                torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])

        write_json(output / "config.json", config.to_dict())
        first_step = step
        sampler = StepSampler(
            train_set.sizes, train_set.weights, config.seed, step * config.batch_size
        )
        loader = _loader(
            train_set,
            config,
            sampler,
            device,
            config.num_workers,
            batch_size=config.batch_size,
            drop_last=True,
        )
        val_loader = None
        if val_set is not None:
            val_loader = _loader(
                val_set,
                config,
                None,
                device,
                min(config.num_workers, 2),
                batch_size=config.val_batch_size or config.batch_size,
                drop_last=False,
            )
        log = _Logger(output, config.track)
        parameters = count_parameters(model)
        metrics: dict[str, Any] = {} if payload is None else dict(payload.get("metrics", {}))
        saved_step = validated_step = -1

        def save() -> Path:
            nonlocal saved_step
            rng = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            path = commit_checkpoint(
                checkpoints,
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "step": step,
                    "samples_seen": step * config.batch_size,
                    "track": config.track,
                    "model_config": config.model.to_dict(),
                    "train_config": config.to_dict(),
                    "data_spec": spec.to_dict(),
                    "data_spec_sha256": spec.sha256,
                    "data_sources": data_sources,
                    "counters": counters,
                    "code": code,
                    "lineage": lineage,
                    "resumes": resumes,
                    "parameters": parameters,
                    "metrics": metrics,
                    "best": best,
                    "environment": environment(),
                    "created_utc": utc_now(),
                    "model_state": {k: v.contiguous() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "rng_state": rng,
                },
                keep=config.keep_checkpoints,
                is_best=best is not None and best["step"] == step,
            )
            saved_step = step
            return path

        def run_validation() -> bool:
            """Validate once per step; True when this step is the new best."""
            nonlocal best, metrics, validated_step
            if val_loader is None or validated_step == step:
                return False
            metrics = {**validate(model, val_loader, config, device), "signature": signature}
            validated_step = step
            for view in config.val_views:
                crop = "gt_full_frame_validation" if view.crop == "full_frame" else None
                metrics["views"][view.name]["evidence"] = own(crop or "gt_crop_validation")
            metrics["evidence"] = metrics["views"][metrics["view"]]["evidence"]
            memory = {k: v for k, v in metrics.items() if k in MEMORY_KEYS}
            for name, values in metrics["views"].items():
                selected = name == metrics["view"]
                record = {"kind": "val", "step": step, "view": name, "selected": selected}
                log({**record, **values, "samples": metrics["samples"], **memory})
            if improves(best, metrics["epe_px"]):
                best = {"step": step, **metrics}
                return True
            return False

        status("running")
        iterator = iter(loader)
        window = {"images": 0, "data": 0.0, "start": time.perf_counter()}
        sums: dict[str, float] = {}
        grad_norm: float | None = None
        nonfinite_run = 0
        stop_reason = "max_steps"
        while step < config.stop_step:
            if (
                config.max_seconds is not None
                and time.perf_counter() - started >= config.max_seconds
            ):
                stop_reason = "max_seconds"
                break
            waited = time.perf_counter()
            batch = _to_device(next(iterator), device)
            window["data"] += time.perf_counter() - waited
            batch["image"] = _layout(to_float_images(batch), config)  # colour jitter on device
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with _autocast(config, device):
                grid2d, grid3d = model(batch["image"])
            losses = compute_losses(config, step, batch, grid2d, grid3d)
            finite = bool(torch.isfinite(losses["loss"]))
            nonfinite_run = 0 if finite else nonfinite_run + 1
            counters["skipped"] += not finite
            if nonfinite_run > 100:
                raise FlattenerError(
                    Status.MODEL_ERROR,
                    "non_finite_loss",
                    f"100 consecutive non-finite losses at step {step}",
                )
            if finite or scaler.is_enabled():  # the scaler skips non-finite updates itself
                scaler.scale(losses["loss"]).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    grad_norm = float(norm)  # inf/nan on an AMP overflow step
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                counters["overflow_steps"] += scaler.is_enabled() and scaler.get_scale() < scale
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window["images"] += batch["image"].shape[0]
            if finite:
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                sums["count"] = sums.get("count", 0.0) + 1
            if step % config.log_every == 0 or step == config.stop_step:
                elapsed = max(time.perf_counter() - window["start"], 1e-9)
                count = max(sums.pop("count", 0.0), 1.0)
                record = {
                    "kind": "train",
                    "step": step,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "images_per_sec": window["images"] / elapsed,
                    "data_fraction": window["data"] / elapsed,
                    **counters,
                    "scale": scaler.get_scale(),
                    "elapsed_sec": time.perf_counter() - started,
                    **{key: value / count for key, value in sums.items()},
                }
                record.update(memory_stats(device), evidence=own("training_loss"))
                log(record)
                status("running", last=record)
                window = {"images": 0, "data": 0.0, "start": time.perf_counter()}
                sums = {}
            improved = bool(config.val_every) and step % config.val_every == 0 and run_validation()
            if save_due(step, improved, saved_step, config.checkpoint_every):
                save()
        if step > first_step:
            improved = run_validation()
            if save_due(step, improved, saved_step, config.checkpoint_every, last=True):
                save()
        final = latest_checkpoint(checkpoints)
        state = "completed" if step >= config.total_steps else "stopped"
        memory = memory_stats(device)
        label = metrics.get("evidence") or own("training_loss")
        status(state, reason=stop_reason, checkpoint=str(final), memory=memory, evidence=label)
        return {
            "state": state,
            "step": step,
            "checkpoint": str(final),
            "evidence": label,
            "metrics": metrics,
            "best": best,
            "memory": memory,
            "seconds": time.perf_counter() - started,
        }
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="TrainConfig JSON; its relative paths are file-relative"
    )
    parser.add_argument("--resume", default="auto", help="auto | none | checkpoint path")
    for name, kind in (
        ("output-dir", str),
        ("data-spec", str),
        ("track", str),
        ("init", str),
        ("init-path", str),
        ("device", str),
        ("max-steps", int),
        ("max-seconds", float),
        ("num-workers", int),
        ("batch-size", int),
        ("seed", int),
    ):
        parser.add_argument(f"--{name}", type=kind, default=None)
    parser.add_argument("--print-config", action="store_true", help="print resolved config, exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = Path(args.config)
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides = {
        key.replace("-", "_"): value
        for key, value in vars(args).items()
        if value is not None and key not in ("config", "resume", "print_config")
    }
    for key in ("data_spec", "output_dir", "init_path"):  # ${VAR}s expand; relative to the file
        if overrides.get(key):
            overrides[key] = str(Path(expand_path(overrides[key])))
        elif data.get(key):
            value = Path(expand_path(data[key]))
            data[key] = str(value if value.is_absolute() else path.resolve().parent / value)
    config = replace(TrainConfig.from_dict(data), **overrides)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    summary = train(config, resume=args.resume)
    print(json.dumps({k: summary[k] for k in ("state", "step", "checkpoint", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
