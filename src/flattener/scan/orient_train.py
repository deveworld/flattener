"""Train page orientation from upright grayscale pages.

Data comes from tools/data/orientation/build_data.py.
Augmentation adds quarter turns, small rotations, warping and lighting changes.
Validation measures all four orientations and fits the probability temperature.
Checkpoints include weights, optimizer state, training settings and calibration.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageFilter
from torch.nn import functional as F

from ..contracts import FlattenerError, Status
from ..dewarp.checkpoint import environment, json_safe, utc_now, write_json
from ..dewarp.data import expand_path
from ..dewarp.train import StepSampler, memory_stats, resolve_device, worker_init
from ..evidence import evidence
from . import orient
from .orient import (
    CHECKPOINT_SCHEMA,
    OrientConfig,
    OrientNet,
    PageViews,
    SourceSampler,
    ViewGeometry,
    count_parameters,
    load_checkpoint,
    page_views,
    save_checkpoint,
)

DATA_SCHEMA = "flattener-orient-data-1"
DATA_KINDS = (
    "syntheticdoc_flat", "syntheticdoc_crop", "documents", "spread", "text_free", "pages",
)  # fmt: skip
CROP_KINDS = frozenset({"syntheticdoc_crop"})
INITS = ("scratch", "checkpoint")
VAL_SEED = 20260927
RESUMABLE_CHANGES = frozenset(
    {
        "output_dir",
        "data",
        "init",
        "init_path",
        "max_steps",
        "max_seconds",
        "num_workers",
        "prefetch_factor",
        "pin_memory",
        "device",
        "log_every",
        "val_every",
        "val_batch_size",
        "checkpoint_every",
        "keep_checkpoints",
    }
)


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason, message)


# ----------------------------------------------------------------------------- data


class OrientData:
    """A built data directory: its summary (``orient-data.json``) and page records."""

    def __init__(self, root: Path, summary: dict[str, Any], records: list[dict[str, Any]]):
        self.root, self.summary, self.records = root, summary, records

    @classmethod
    def load(cls, root: str | Path) -> OrientData:
        root = Path(root)
        path = root / "orient-data.json"
        if not path.is_file():
            raise _error("missing_data", f"{path} is missing (build not finished?)")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema") != DATA_SCHEMA or summary.get("complete") is not True:
            raise _error("invalid_data", f"{path}: needs schema {DATA_SCHEMA} and complete")
        manifest = root / summary["records_file"]
        data = manifest.read_bytes()
        records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
        for record in records:
            if record.get("kind") not in DATA_KINDS:
                raise _error("invalid_data", f"{record.get('id')}: unknown kind")
        return cls(root, summary, records)

    @property
    def spec(self) -> dict[str, Any]:
        """The dewarp-schema data spec naming every source of the build (``data_spec``)."""
        return self.summary["data_spec"]

    @property
    def data_sources(self) -> dict[str, Any]:
        """The license manifest: per split, sources with track and license classes."""
        return self.summary["data_sources"]

    @property
    def description(self) -> dict[str, Any]:
        """The build summary without creation time and environment (``orient_data``)."""
        return {k: v for k, v in self.summary.items() if k not in ("created_utc", "environment")}

    @property
    def track(self) -> str:
        return self.summary.get("track", "")

    def split(self, name: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record["split"] == name]


def load_page(root: Path, record: dict[str, Any]) -> NDArray[np.uint8]:
    with Image.open(root / record["path"]) as image:
        return np.array(image.convert("L"))


# ----------------------------------------------------------------------------- augmentation


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Geometric and photometric augmentation; lengths are fractions of the page extent,
    ``blur_sigma`` is in pixels of the page at ``page_long_side``."""

    margin: tuple[float, float] = (-0.03, 0.12)
    skew_deg: float = 5.0
    crop_skew_sd_deg: float = 3.0
    crop_skew_max_deg: float = 10.0
    warp_prob: float = 0.7
    warp_max: float = 0.03
    light_prob: float = 0.5
    light_strength: float = 0.3
    shadow_prob: float = 0.3
    shadow_strength: float = 0.5
    contrast: float = 0.3
    brightness: float = 0.12
    gamma: float = 0.3
    blur_prob: float = 0.3
    blur_sigma: float = 1.5
    noise_prob: float = 0.5
    noise_std: float = 6.0
    jpeg_prob: float = 0.5
    jpeg_quality: tuple[int, int] = (35, 95)

    def __post_init__(self) -> None:
        object.__setattr__(self, "margin", tuple(self.margin))
        object.__setattr__(self, "jpeg_quality", tuple(self.jpeg_quality))
        low, high = self.margin
        if not -0.2 <= low <= high <= 0.5:
            raise ValueError("margin must be (low, high) with -0.2 <= low <= high <= 0.5")
        quality = self.jpeg_quality
        if not 1 <= quality[0] <= quality[1] <= 100:
            raise ValueError("jpeg_quality must be (low, high) in [1, 100]")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in ("margin", "jpeg_quality"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{item.name} must be a non-negative number")
            if item.name.endswith("_prob") and value > 1:
                raise ValueError(f"{item.name} must be <= 1")
        if max(self.contrast, self.brightness, self.gamma, self.shadow_strength) >= 1:
            raise ValueError("contrast, brightness, gamma and shadow_strength must be < 1")

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def smooth_warp(rng: np.random.Generator, amplitude: float) -> NDArray[np.float32]:
    """A (2, 4, 4) displacement field whose largest node reaches ``amplitude`` (normalized
    units; 2 is the page extent)."""
    field_ = rng.normal(size=(2, 4, 4))
    return (field_ * amplitude / max(float(np.abs(field_).max()), 1e-9)).astype(np.float32)


def training_geometry(
    record: dict[str, Any],
    shape: tuple[int, int],
    k: int,
    rng: np.random.Generator,
    augment: AugmentConfig,
) -> ViewGeometry:
    """The page box of ``record`` with drawn margins, turned by ``90 k`` plus a skew.

    Warped crops keep their margins within the stored crop and take no extra warp.
    """
    height, width = shape
    box = np.asarray(record["box"], dtype=np.float64)
    x0, x1 = (box[[0, 2]] + 1.0) / 2.0 * width
    y0, y1 = (box[[1, 3]] + 1.0) / 2.0 * height
    page_w, page_h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    margins = rng.uniform(augment.margin[0], augment.margin[1], size=4)  # left, top, right, bottom
    warp = None
    if record["kind"] in CROP_KINDS:
        room = np.array([x0 / page_w, y0 / page_h, (width - x1) / page_w, (height - y1) / page_h])
        margins = np.minimum(margins, room)
        limit = augment.crop_skew_max_deg
        skew = float(np.clip(rng.normal(0.0, augment.crop_skew_sd_deg), -limit, limit))
    else:
        skew = float(rng.uniform(-augment.skew_deg, augment.skew_deg))
        if rng.random() < augment.warp_prob:
            warp = smooth_warp(rng, 2.0 * float(rng.uniform(0.0, augment.warp_max)))
    left, top = x0 - margins[0] * page_w, y0 - margins[1] * page_h
    right, bottom = x1 + margins[2] * page_w, y1 + margins[3] * page_h
    center = ((left + right) / 2.0, (top + bottom) / 2.0)
    return ViewGeometry.turned(center, (bottom - top, right - left), k, skew, warp)


class TrainingRender:
    """Page lookups with a synthetic surface outside flat pages, illumination, shadow,
    contrast and gamma, all as functions of the source point (so the global view and the
    patches agree); ``finish`` adds per-view blur, noise and JPEG."""

    def __init__(
        self,
        sampler: SourceSampler,
        record: dict[str, Any],
        rng: np.random.Generator,
        augment: AugmentConfig,
        config: OrientConfig,
    ) -> None:
        self.sampler, self.rng = sampler, rng
        self.surface = None
        if record["kind"] not in CROP_KINDS:
            self.surface = (rng.uniform(20, 235), rng.uniform(-25, 25, size=2), rng.uniform(1, 8))
        self.light = None
        if rng.random() < augment.light_prob:
            direction = rng.normal(size=2)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            self.light = rng.uniform(0, augment.light_strength) * direction
        self.shadow = None
        if rng.random() < augment.shadow_prob:
            angle = rng.uniform(0, 2 * math.pi)
            normal = np.array([math.cos(angle), math.sin(angle)])
            self.shadow = (
                rng.uniform(0.15, augment.shadow_strength),
                normal,
                rng.uniform(-0.8, 0.8),
                rng.uniform(0.02, 0.3),
            )
        self.contrast = 1.0 + rng.uniform(-augment.contrast, augment.contrast)
        self.brightness = rng.uniform(-augment.brightness, augment.brightness)
        self.gamma = math.exp(rng.uniform(-augment.gamma, augment.gamma))
        self.blur = rng.uniform(0.3, augment.blur_sigma) if rng.random() < augment.blur_prob else 0
        self.noise = rng.uniform(1, augment.noise_std) if rng.random() < augment.noise_prob else 0
        self.jpeg = None
        if rng.random() < augment.jpeg_prob:
            self.jpeg = int(rng.integers(augment.jpeg_quality[0], augment.jpeg_quality[1] + 1))
        self.global_ratio = config.global_size / config.page_long_side

    def __call__(self, points: NDArray[np.float64], density: float) -> NDArray[np.float32]:
        q = self.sampler.normalized(points)
        values = self.sampler.lookup(q, density * self.sampler.geometry.scale)
        if self.surface is not None:
            outside = (np.abs(q[..., 0]) > 1.0) | (np.abs(q[..., 1]) > 1.0)
            if outside.any():
                tone, slope, grain = self.surface
                spots = q[outside]
                values = values.copy()
                noise = self.rng.normal(0, grain, len(spots))
                values[outside] = tone + spots[:, 0] * slope[0] + spots[:, 1] * slope[1] + noise
        qx, qy = q[..., 0].astype(np.float32), q[..., 1].astype(np.float32)
        values = values * np.float32(1 / 255)
        if self.light is not None:
            a, b = (np.float32(v) for v in self.light)
            values *= np.maximum(1.0 + a * qx + b * qy, np.float32(0.2))
        if self.shadow is not None:
            strength, normal, offset, softness = self.shadow
            a, b = (np.float32(v) for v in normal)
            ramp = (a * qx + b * qy - np.float32(offset)) * np.float32(0.5 / softness)
            values *= 1.0 - np.float32(strength / 2) * (1.0 + np.tanh(ramp))
        np.clip(values, 0.0, 1.0, out=values)
        values *= np.float32(self.contrast)
        values += np.float32(0.5 - 0.5 * self.contrast + self.brightness)
        np.clip(values, 0.0, 1.0, out=values)
        if self.gamma != 1.0:
            np.power(values, np.float32(self.gamma), out=values)
        values *= np.float32(255)
        return values

    def finish(self, view: NDArray[np.uint8], role: str) -> NDArray[np.uint8]:
        sigma = self.blur * (self.global_ratio if role == "global" else 1.0)
        image = Image.fromarray(view)
        if sigma >= 0.3:
            image = image.filter(ImageFilter.GaussianBlur(sigma))
        out = np.asarray(image, dtype=np.float32)
        if self.noise:
            out = out + self.rng.normal(0, self.noise, size=out.shape)
        out = np.floor(np.clip(out, 0, 255) + 0.5).astype(np.uint8)
        if self.jpeg is not None:
            buffer = io.BytesIO()
            Image.fromarray(out).save(buffer, "JPEG", quality=self.jpeg)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                out = np.array(decoded.convert("L"))
        return out


def augmented_views(
    record: dict[str, Any],
    image: NDArray[np.uint8],
    k: int,
    rng: np.random.Generator,
    augment: AugmentConfig,
    config: OrientConfig,
) -> PageViews:
    """Model inputs of ``record`` turned by ``k`` quarter turns, with augmentation."""
    geometry = training_geometry(record, image.shape, k, rng, augment)
    render = TrainingRender(SourceSampler(image, geometry), record, rng, augment, config)
    return page_views(render, geometry.size, config, finish=render.finish)


def _stable_seed(text: str) -> int:
    return random.Random(int.from_bytes(text.encode("utf-8"), "little")).getrandbits(64)


class OrientDataset(torch.utils.data.Dataset):
    """Training items (keys ``(flat_index, position)`` from ``StepSampler``) or validation
    items (every labelled page by k = 0..3, text-free pages once)."""

    def __init__(
        self,
        root: Path,
        records: list[dict[str, Any]],
        *,
        model: OrientConfig,
        augment: AugmentConfig,
        seed: int,
        train: bool,
    ) -> None:
        self.root, self.model, self.augment = root, model, augment
        self.seed, self.train = int(seed), train
        # Loader workers get a copy of the records; keep only what an item reads.
        keys = ("id", "path", "kind", "box", "uniform", "source")
        records = [{key: record[key] for key in keys} for record in records]
        if train:  # grouped by source for weighted sampling
            names = list(dict.fromkeys(record["source"] for record in records))
            self.records = [r for name in names for r in records if r["source"] == name]
            self.names = names
            self.sizes = [sum(r["source"] == name for r in records) for name in names]
        else:
            self.records = list(records)
            self.items = [
                (index, k)
                for index, record in enumerate(self.records)
                for k in ((0,) if record["uniform"] else range(4))
            ]

    def __len__(self) -> int:
        return len(self.records) if self.train else len(self.items)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        if self.train:
            index, position = key
            rng = np.random.default_rng([self.seed, 7, int(position)])
            k = int(rng.integers(4))
        else:
            index, k = self.items[key]
            rng = np.random.default_rng([VAL_SEED, _stable_seed(self.records[index]["id"]), k])
        record = self.records[index]
        views = augmented_views(
            record, load_page(self.root, record), k, rng, self.augment, self.model
        )
        return {
            "global_view": torch.from_numpy(views.global_view)[None],
            "patches": torch.from_numpy(views.patches)[:, None],
            "label": torch.tensor(-1 if record["uniform"] else k),
            "k": torch.tensor(k),
            "source": record["source"],
            "id": record["id"],
        }


# ----------------------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class TrainConfig:
    data: str
    output_dir: str
    track: str = "release"
    model: OrientConfig = field(default_factory=OrientConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    init: str = "scratch"
    init_path: str | None = None
    seed: int = 0
    batch_size: int = 128
    total_steps: int = 16_000
    max_steps: int | None = None
    max_seconds: float | None = None
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01
    label_smoothing: float = 0.05
    aux_weight: float = 0.3
    grad_clip: float = 2.0
    amp: bool = True
    num_workers: int = 2  # local default; remote configs set their own
    prefetch_factor: int = 4
    pin_memory: bool = True
    device: str = "auto"
    log_every: int = 50
    val_every: int = 1000
    val_max_pages: int | None = None
    val_batch_size: int | None = None
    checkpoint_every: int = 1000
    keep_checkpoints: int = 2
    source_weights: dict[str, float] | None = None
    max_silent_error: float = 0.005

    def __post_init__(self) -> None:
        if self.track not in ("release", "research") or self.init not in INITS:
            raise _error("invalid_config", f"track release|research, init {INITS}")
        if self.init != "scratch" and not self.init_path:
            raise _error("invalid_config", f"init {self.init!r} needs init_path")
        integers = {
            "seed": 0,
            "batch_size": 1,
            "total_steps": 1,
            "warmup_steps": 0,
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
        for name in ("max_steps", "val_max_pages", "val_batch_size"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise _error("invalid_config", f"{name} must be a positive integer or null")
        for name in ("lr", "weight_decay", "min_lr_ratio", "label_smoothing", "aux_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise _error("invalid_config", f"{name} must be a non-negative number")
        if not 0 < self.lr <= 1 or self.min_lr_ratio > 1 or self.label_smoothing >= 1:
            raise _error("invalid_config", "lr in (0, 1], min_lr_ratio <= 1, smoothing < 1")
        if self.max_seconds is not None and not self.max_seconds > 0:
            raise _error("invalid_config", "max_seconds must be positive or null")
        if not 0 < self.max_silent_error < 1:
            raise _error("invalid_config", "max_silent_error must be in (0, 1)")
        weights = self.source_weights
        if weights is not None and (
            not isinstance(weights, dict)
            or not weights
            or not all(isinstance(v, (int, float)) and v > 0 for v in weights.values())
        ):
            raise _error("invalid_config", "source_weights must map source names to weights > 0")

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            result[item.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        if not isinstance(data, dict):
            raise _error("invalid_config", "train config must be a JSON object")
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise _error("invalid_config", f"unknown train config fields: {sorted(unknown)}")
        values = dict(data)
        try:
            values["model"] = OrientConfig.from_dict(values.get("model", {}))
            values["augment"] = AugmentConfig(**values.get("augment", {}))
        except (TypeError, ValueError) as error:
            raise _error("invalid_config", str(error)) from error
        return cls(**values)

    @property
    def stop_step(self) -> int:
        return min(self.max_steps or self.total_steps, self.total_steps)


def learning_rate(config: TrainConfig, step: int) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    span = max(1, config.total_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / span)
    floor = config.min_lr_ratio
    return config.lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def soft_targets(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Smoothed one-hot targets; rows with label -1 get the uniform distribution."""
    onehot = F.one_hot(labels.clamp_min(0), orient.CLASSES).float()
    smoothed = onehot * (1.0 - smoothing) + smoothing / orient.CLASSES
    uniform = torch.full_like(smoothed, 1.0 / orient.CLASSES)
    return torch.where((labels >= 0)[:, None], smoothed, uniform)


def compute_losses(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    labels: torch.Tensor,
    config: TrainConfig,
) -> dict[str, torch.Tensor]:
    targets = soft_targets(labels, config.label_smoothing)

    def cross_entropy(logits: torch.Tensor) -> torch.Tensor:
        return -(targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()

    both, alone, patches = outputs
    losses = {"loss_main": cross_entropy(both), "loss_global": cross_entropy(alone)}
    aux = losses["loss_global"]
    if patches is not None:
        losses["loss_patches"] = cross_entropy(patches)
        aux = (aux + losses["loss_patches"]) / 2
    losses["loss"] = losses["loss_main"] + config.aux_weight * aux
    return losses


@torch.no_grad()
def validate(model: OrientNet, loader, device: torch.device, config: TrainConfig) -> dict[str, Any]:
    """Metrics of the pooled logits (raw and calibrated) and of the global view alone."""
    was_training = model.training
    model.eval()
    logits, alone, labels, groups = [], [], [], []
    for batch in loader:
        both, only, _ = model.forward_views(
            batch["global_view"].to(device).float(), batch["patches"].to(device).float()
        )
        logits.append(both.float().cpu())
        alone.append(only.float().cpu())
        labels.append(batch["label"])
        groups.extend(batch["source"])
    model.train(was_training)
    logits_ = torch.cat(logits).numpy()
    labels_ = torch.cat(labels).numpy()
    metrics = orient.orientation_metrics(logits_, labels_, groups=groups)
    temperature = orient.fit_temperature(logits_, labels_)
    probabilities = orient.softmax(logits_, temperature)
    point = orient.choose_threshold(probabilities, labels_, config.max_silent_error)
    calibrated = orient.orientation_metrics(
        logits_, labels_, temperature=temperature, threshold=point["threshold"]
    )
    metrics["calibration"] = {
        "source": "synthetic_val",
        "temperature": temperature,
        **point,
        "ece": calibrated.get("ece"),
        "nll": calibrated.get("nll"),
        "text_free_accepted": calibrated.get("text_free_accepted"),
    }
    global_only = orient.orientation_metrics(torch.cat(alone).numpy(), labels_)
    metrics["global_only"] = {k: global_only.get(k) for k in ("accuracy", "macro_accuracy")}
    return metrics


def improves(best: dict[str, Any] | None, metrics: dict[str, Any]) -> bool:
    score = metrics.get("macro_accuracy")
    if score is None or not math.isfinite(score):
        return False
    if best is None or best.get("macro_accuracy") is None:
        return True
    if score != best["macro_accuracy"]:
        return score > best["macro_accuracy"]
    return (metrics.get("nll") or math.inf) < (best.get("nll") or math.inf)


# ----------------------------------------------------------------------------- checkpoints


def commit_checkpoint(
    directory: Path, payload: dict[str, Any], *, keep: int, is_best: bool
) -> Path:
    """Save ``step_<step>.pt``, move ``latest.json`` (and ``best.json``), prune old files."""
    path = directory / f"step_{int(payload['step']):08d}.pt"
    save_checkpoint(path, payload)
    pointer = {"file": path.name, "step": payload["step"]}
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
    return path


def _initialize(model: OrientNet, config: TrainConfig) -> dict[str, Any]:
    if config.init == "scratch":
        return {"kind": "scratch", "seed": config.seed}
    parent = load_checkpoint(config.init_path)
    model.load_state_dict(parent["model_state"])
    return {"kind": "checkpoint", "path": str(config.init_path), "step": parent["step"]}


def _check_resume(config: TrainConfig, payload: dict[str, Any], data: OrientData) -> None:
    saved, current = payload["train_config"], config.to_dict()
    changed = sorted(
        k for k in current if k not in RESUMABLE_CHANGES and current[k] != saved.get(k)
    )
    if payload.get("orient_data") != data.description:
        changed.append("data")
    if changed:
        raise _error("resume_mismatch", f"resume differs in {changed}; use a new output_dir")


def _parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay = [p for p in model.parameters() if p.ndim > 1]
    other = [p for p in model.parameters() if p.ndim <= 1]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}]


def _loader(dataset, sampler, config: TrainConfig, device, workers: int, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        drop_last=sampler is not None,
        persistent_workers=workers > 0,
        prefetch_factor=config.prefetch_factor if workers > 0 else None,
        worker_init_fn=worker_init if workers > 0 else None,
        # Creating a loader iterator must not advance the model's dropout RNG.
        generator=torch.Generator().manual_seed(config.seed),
    )


def _val_pages(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or len(records) <= limit:
        return records
    keep = np.unique(np.linspace(0, len(records) - 1, limit).round().astype(int))
    return [records[i] for i in keep]


# ----------------------------------------------------------------------------- training


def train(config: TrainConfig, *, resume: str = "auto") -> dict[str, Any]:
    """Train until ``stop_step`` or ``max_seconds``; return a summary.

    ``resume``: ``"auto"`` (the latest checkpoint in the output directory, if any),
    ``"none"`` (refuse a directory with checkpoints) or a checkpoint path.
    """
    started = time.perf_counter()
    output = Path(config.output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    step = 0
    lineage: list[dict[str, Any]] | None = None

    def own(measurement: str) -> dict[str, Any]:
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
        try:
            write_json(output / "status.json", {**record, **extra})
        except OSError as error:
            print(f"warning: status.json not updated: {error}", file=sys.stderr, flush=True)

    def log(record: dict[str, Any]) -> None:
        record = json_safe({**record, "track": config.track})
        with (output / "train_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in record.items() if not isinstance(v, dict)}), flush=True)

    status("starting")
    try:
        data = OrientData.load(config.data)
        if config.track == "release":
            problems = orient.data_spec_problems(data.spec)
            problems += orient.data_sources_problems(data.data_sources)
            if problems:
                raise _error("track_violation", f"release training refuses data: {problems}")
        device = resolve_device(config.device)
        torch.backends.cudnn.benchmark = device.type == "cuda"  # fixed input sizes
        train_records = data.split("train")
        if not train_records:
            raise _error("missing_data", "no train pages")
        common = dict(model=config.model, augment=config.augment, seed=config.seed)
        train_set = OrientDataset(data.root, train_records, train=True, **common)
        val_records = _val_pages(data.split("val"), config.val_max_pages)
        val_set = (
            OrientDataset(data.root, val_records, train=False, **common) if val_records else None
        )

        latest = checkpoints / "latest.json"
        if resume == "auto":
            resume_path = (
                checkpoints / json.loads(latest.read_text("utf-8"))["file"]
                if latest.is_file()
                else None
            )
        elif resume == "none":
            if latest.is_file():
                raise _error("output_exists", f"{checkpoints} has checkpoints; resume or change")
            resume_path = None
        else:
            resume_path = Path(resume)

        torch.manual_seed(config.seed)
        model = OrientNet(config.model)
        code = orient.code_record()
        best: dict[str, Any] | None = None
        metrics: dict[str, Any] = {}
        resumes: list[dict[str, Any]] = []
        payload = None
        if resume_path is not None:
            payload = load_checkpoint(resume_path)
            _check_resume(config, payload, data)
            model.load_state_dict(payload["model_state"])
            lineage, step, best = payload["lineage"], payload["step"], payload.get("best")
            metrics = dict(payload.get("metrics") or {})
            resumes = [
                *payload.get("resumes", []),
                {"step": step, "utc": utc_now()},
            ]
        else:
            lineage = [
                _initialize(model, config),
                {
                    "kind": "train",
                    "track": config.track,
                    "started_utc": utc_now(),
                },
            ]
        model.to(device).train()
        optimizer = torch.optim.AdamW(_parameter_groups(model, config.weight_decay), lr=config.lr)
        amp = config.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        if payload is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
            if payload.get("scaler_state"):
                scaler.load_state_dict(payload["scaler_state"])
            torch.set_rng_state(payload["rng_state"]["torch"])
            if device.type == "cuda" and payload["rng_state"].get("cuda") is not None:
                torch.cuda.set_rng_state(payload["rng_state"]["cuda"], device)
        write_json(output / "config.json", config.to_dict())

        weights = None
        if config.source_weights is not None:
            unknown = set(config.source_weights) - set(train_set.names)
            if unknown:
                raise _error("invalid_config", f"source_weights names unknown sources {unknown}")
            raw = [float(config.source_weights.get(name, 0.0)) for name in train_set.names]
            if not sum(raw) > 0:
                raise _error("invalid_config", "source_weights select no train source")
            weights = [w / sum(raw) for w in raw]
        sampler = StepSampler(train_set.sizes, weights, config.seed, step * config.batch_size)
        loader = _loader(train_set, sampler, config, device, config.num_workers, config.batch_size)
        val_loader = None
        if val_set is not None:
            workers = min(config.num_workers, 4)
            batch = config.val_batch_size or config.batch_size
            val_loader = _loader(val_set, None, config, device, workers, batch)
        parameters = count_parameters(model)
        first_step, saved_step, validated_step = step, -1, -1

        def save(is_best: bool) -> Path:
            nonlocal saved_step
            path = commit_checkpoint(
                checkpoints,
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "model_id": orient.MODEL_ID,
                    "step": step,
                    "samples_seen": step * config.batch_size,
                    "track": config.track,
                    "model_config": config.model.to_dict(),
                    "train_config": config.to_dict(),
                    "data_spec": data.spec,
                    "data_sources": data.data_sources,
                    "orient_data": data.description,
                    "code": code,
                    "lineage": lineage,
                    "resumes": resumes,
                    "parameters": parameters,
                    "metrics": metrics,
                    "calibration": metrics.get("calibration"),
                    "best": best,
                    "environment": environment(),
                    "created_utc": utc_now(),
                    "model_state": {k: v.contiguous() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "rng_state": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    },
                },
                keep=config.keep_checkpoints,
                is_best=is_best,
            )
            saved_step = step
            return path

        def run_validation() -> bool:
            nonlocal best, metrics, validated_step
            if val_loader is None or validated_step == step:
                return False
            metrics = validate(model, val_loader, device, config)
            metrics["evidence"] = own("gt_crop_validation")
            validated_step = step
            flat = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
            log({"kind": "val", "step": step, **flat, "calibration": metrics["calibration"]})
            if improves(best, metrics):
                best = {"step": step, **metrics}
                return True
            return False

        status("running")
        iterator = iter(loader)
        window = {"images": 0, "data": 0.0, "start": time.perf_counter()}
        sums: dict[str, float] = {}
        stop_reason = "max_steps"
        while step < config.stop_step:
            if (
                config.max_seconds is not None
                and time.perf_counter() - started >= config.max_seconds
            ):
                stop_reason = "max_seconds"
                break
            waited = time.perf_counter()
            batch = next(iterator)
            window["data"] += time.perf_counter() - waited
            global_view = batch["global_view"].to(device, non_blocking=True).float()
            patches = batch["patches"].to(device, non_blocking=True).float()
            labels = batch["label"].to(device, non_blocking=True)
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            context = torch.autocast(device.type, dtype=torch.float16) if amp else None
            with context or contextlib.nullcontext():
                outputs = model.forward_views(global_view, patches)
            losses = compute_losses(outputs, labels, config)
            if not torch.isfinite(losses["loss"]):
                raise FlattenerError(
                    Status.MODEL_ERROR, "non_finite_loss", f"non-finite loss at step {step}"
                )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window["images"] += global_view.shape[0]
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
            sums["count"] = sums.get("count", 0.0) + 1
            if step % config.log_every == 0 or step == config.stop_step:
                elapsed = max(time.perf_counter() - window["start"], 1e-9)
                count = max(sums.pop("count", 1.0), 1.0)
                record = {
                    "kind": "train",
                    "step": step,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "images_per_sec": window["images"] / elapsed,
                    "data_fraction": window["data"] / elapsed,
                    "elapsed_sec": time.perf_counter() - started,
                    **{key: value / count for key, value in sums.items()},
                    **memory_stats(device),
                }
                log(record)
                status("running", last=record)
                window = {"images": 0, "data": 0.0, "start": time.perf_counter()}
                sums = {}
            improved = bool(config.val_every) and step % config.val_every == 0 and run_validation()
            if improved or (config.checkpoint_every and step % config.checkpoint_every == 0):
                save(improved)
        if step > first_step:
            improved = run_validation()
            if improved or saved_step != step:
                save(improved)
        state = "completed" if step >= config.total_steps else "stopped"
        label = metrics.get("evidence") or own("training_loss")
        best_path = checkpoints / "best.json"
        summary = {
            "state": state,
            "reason": stop_reason,
            "step": step,
            "checkpoint": str(checkpoints / f"step_{step:08d}.pt"),
            "best": json.loads(best_path.read_text("utf-8")) if best_path.is_file() else None,
            "metrics": metrics,
            "evidence": label,
            "seconds": time.perf_counter() - started,
        }
        status(state, reason=stop_reason, checkpoint=summary["checkpoint"], evidence=label)
        return summary
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="TrainConfig JSON; paths are file-relative")
    parser.add_argument("--resume", default="auto", help="auto | none | checkpoint path")
    for name, kind in (
        ("output-dir", str),
        ("data", str),
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
    """Exit codes: 0 finished or stopped by a limit, 1 training error, 2 usage error."""
    args = _parser().parse_args(argv)
    path = Path(args.config)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        overrides = {
            key.replace("-", "_"): value
            for key, value in vars(args).items()
            if value is not None and key not in ("config", "resume", "print_config")
        }
        for key in ("data", "output_dir", "init_path"):
            if overrides.get(key):
                overrides[key] = str(Path(expand_path(overrides[key])))
            elif values.get(key):
                value = Path(expand_path(values[key]))
                values[key] = str(value if value.is_absolute() else path.resolve().parent / value)
        config = replace(TrainConfig.from_dict(values), **overrides)
    except FlattenerError as error:
        print(
            json.dumps(
                {"status": error.status, "reason_code": error.reason_code, "message": error.message}
            )
        )
        return 2
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    try:
        summary = train(config, resume=args.resume)
    except FlattenerError as error:
        print(
            json.dumps(
                {"status": error.status, "reason_code": error.reason_code, "message": error.message}
            )
        )
        return 1
    print(json.dumps({k: summary[k] for k in ("state", "step", "checkpoint", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
