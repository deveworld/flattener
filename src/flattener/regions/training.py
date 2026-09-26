"""Deterministic CPU page-region training using synthetic train/validation only.

Loss and checkpoint selection are fixed before test evaluation. Real benchmark
photos, their scans, and test masks never enter gradients or model selection.
"""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import PIL
import torch
from torch.nn import functional as F

from ..contracts import FlattenerError, Status
from .segmentation import (
    CHECKPOINT_SCHEMA,
    MODEL_VERSION,
    PREPROCESS_VERSION,
    PageSegmenter,
    SegmenterConfig,
    preprocess_image,
    preprocess_mask,
    read_checkpoint,
    sha256,
)

LOSS_PROTOCOL = {
    "bce_weight": 1.0,
    "dice_weight": 1.0,
    "dice_smoothing_pixels": 1.0,
    "negative_sample_weight": 2.0,
    "reduction": "per-image BCE+soft Dice; weighted sum divided by sample weights sum",
    "target_resize": "nearest-exact binary sampling at model input size",
}
SELECTION_PROTOCOL = {
    "threshold": 0.5,
    "criterion": "validation weighted composite BCE+Dice loss, same formula as training",
    "direction": "minimize",
    "ties": "earliest epoch, including initialization",
    "source": "validation only; no test images or real photographs",
}


@dataclass(frozen=True, slots=True)
class RegionTrainConfig:
    seed: int = 20260922
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    threads: int = 1
    model_config: SegmenterConfig = field(default_factory=SegmenterConfig)


def _error(reason, message):
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def _config(config):
    if not isinstance(config, RegionTrainConfig):
        raise _error("invalid_config", "config must be RegionTrainConfig")
    for name, low, high in (
        ("seed", 0, 2**63 - 1),
        ("epochs", 1, 100_000),
        ("batch_size", 1, 1024),
        ("threads", 1, 128),
    ):
        value = getattr(config, name)
        if type(value) is not int or not low <= value <= high:
            raise _error("invalid_config", f"{name} must be an integer in [{low}, {high}]")
    if (
        type(config.learning_rate) not in (int, float)
        or not math.isfinite(config.learning_rate)
        or not 0 < config.learning_rate <= 1
        or not isinstance(config.model_config, SegmenterConfig)
    ):
        raise _error("invalid_config", "invalid learning rate or model config")
    return config


@contextmanager
def _torch_runtime(config):
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        torch.random.default_generator.manual_seed(config.seed)
        yield
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE + per-image soft Dice, empty-mask samples weighted twice.

    Smoothing is one pixel in both numerator and denominator, making empty GT
    defined and penalizing any probability mass on an all-background image.
    Weights are determined exclusively by fixed GT, never by model prediction.
    """
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or logits.ndim != 4
        or logits.shape[0] < 1
        or logits.shape[1] != 1
        or logits.shape != targets.shape
        or min(logits.shape[-2:]) < 1
        or logits.device.type != "cpu"
        or targets.device.type != "cpu"
        or logits.dtype != torch.float32
        or targets.dtype != torch.float32
        or not bool(torch.isfinite(logits).all())
        or not bool(((targets == 0) | (targets == 1)).all())
    ):
        raise _error("invalid_training_batch", "loss requires finite float32 N1HW logits/binary GT")
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(axes)
    target_sum = targets.sum(axes)
    dice = 1 - (2 * (probability * targets).sum(axes) + 1) / (
        probability.sum(axes) + target_sum + 1
    )
    weights = torch.where(target_sum == 0, 2.0, 1.0)
    return ((bce + dice) * weights).sum() / weights.sum()


def validation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Fixed .5 threshold, positive macro IoU and negative pixel FP rate."""
    # Reuse validation without changing gradients or the fixed GT denominator.
    loss = float(segmentation_loss(logits, targets))
    predicted, truth = logits >= 0, targets.bool()
    axes = (1, 2, 3)
    positive = truth.any(axes)
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise _error("invalid_dataset", "validation must contain page and no-page samples")
    intersection = (predicted & truth).sum(axes).double()
    union = (predicted | truth).sum(axes).double()
    foreground_iou = float((intersection[positive] / union[positive]).mean())
    negative_fractions = predicted[negative].double().mean(axes)
    negative_pixel_fpr = float(negative_fractions.mean())
    return {
        "foreground_iou": foreground_iou,
        "negative_pixel_false_positive_rate": negative_pixel_fpr,
        "negative_image_false_positive_rate": float((negative_fractions > 0.01).double().mean()),
        "negative_image_area_threshold": 0.01,
        "loss": loss,
        "positive_images": int(positive.sum()),
        "negative_images": int(negative.sum()),
    }


def _load_splits(directory, manifest, model_config):
    # Local import allows segmentation/checkpoint inference without loading the
    # optional synthetic scene generator and its font/document machinery.
    from ..synthetic.scenes import load_scene

    splits = {}
    pixels = 0
    for split in ("train", "validation"):
        images, masks = [], []
        for record in manifest["samples"]:
            if record["split"] != split:
                continue
            sample = load_scene(directory, record)
            pixels += math.prod(model_config.input_size)
            if pixels > 64_000_000:
                raise _error("training_memory_limit", "train + validation exceed 64 million pixels")
            images.append(preprocess_image(sample.image, model_config))
            masks.append(preprocess_mask(sample.mask, model_config))
        if not images:
            raise _error("invalid_dataset", f"{split} split is empty")
        targets = torch.stack(masks)
        present = targets.flatten(1).any(1)
        if not bool(present.any()) or bool(present.all()):
            raise _error("invalid_dataset", f"{split} needs page and no-page examples")
        splits[split] = (torch.stack(images), targets)
    return splits


def _evaluate(model, split, batch_size):
    model.eval()
    with torch.inference_mode():
        logits = torch.cat(
            [
                model(split[0][start : start + batch_size])
                for start in range(0, len(split[0]), batch_size)
            ]
        )
        return validation_metrics(logits, split[1])


def _provenance():
    package = Path(__file__).resolve().parents[1]
    lock = package.parent.parent / "uv.lock"
    return {
        "source_code_sha256": {
            str(path.relative_to(package)): sha256(path) for path in sorted(package.rglob("*.py"))
        },
        "uv_lock_sha256": sha256(lock) if lock.is_file() else None,
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "pillow": str(PIL.__version__),
        },
        "device": "cpu",
        "model_version": MODEL_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "loss": LOSS_PROTOCOL,
        "selection": SELECTION_PROTOCOL,
        "resume_scope": "epoch boundaries; identical learner source and environment",
    }


def _atomic(path, writer):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_checkpoint(path, state):
    _atomic(path, lambda stream: torch.save(state, stream))
    digest = sha256(path)
    _atomic(
        path.with_suffix(path.suffix + ".sha256"),
        lambda stream: stream.write((digest + "\n").encode()),
    )
    return digest


def _resume_compatible(checkpoint, config, manifest_hash, provenance):
    old, new = dict(checkpoint["config"]), asdict(config)
    old.pop("epochs", None)
    new.pop("epochs")
    if old != new:
        raise _error("resume_config_mismatch", "only the total epoch budget may change")
    if checkpoint["manifest_sha256"] != manifest_hash:
        raise _error("resume_dataset_mismatch", "dataset manifest differs from checkpoint")
    previous = checkpoint["provenance"]
    if not isinstance(previous, dict):
        raise _error("invalid_checkpoint", "missing resume provenance")
    for key in ("versions", "uv_lock_sha256", "loss", "selection"):
        if previous.get(key) != provenance[key]:
            raise _error("resume_environment_mismatch", f"training environment changed: {key}")
    for source in ("regions/segmentation.py", "regions/training.py", "synthetic/scenes.py"):
        if previous.get("source_code_sha256", {}).get(source) != provenance[
            "source_code_sha256"
        ].get(source):
            raise _error("resume_environment_mismatch", f"learner source changed: {source}")
    if not 0 <= checkpoint["epoch"] < config.epochs:
        raise _error("resume_epoch_budget", "epochs must exceed completed checkpoint epoch")


def train_regions(dataset_dir, output_dir, config: RegionTrainConfig | None = None, resume=None):
    """Train, save safe last/best checkpoints, and return a JSON-compatible report.

    New directories must not exist. Explicit resume accepts only that output's
    last.pt, checks hashes and exact environment, and restores epoch-boundary
    optimizer/RNG state. Process-global Torch settings are always restored;
    callers must not concurrently train models in the same Python process.
    """
    from ..synthetic.scenes import validate_scene_dataset

    config = _config(RegionTrainConfig() if config is None else config)
    dataset, directory = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    checked = validate_scene_dataset(dataset)
    manifest_hash, provenance = sha256(dataset / "manifest.json"), _provenance()
    checkpoint = None
    if resume is not None:
        if Path(resume).resolve() != directory / "last.pt":
            raise _error("invalid_resume_path", "resume must name output_dir/last.pt")
        checkpoint = read_checkpoint(resume)
        _resume_compatible(checkpoint, config, manifest_hash, provenance)
        best = read_checkpoint(directory / "best.pt")
        if (
            best["epoch"] != checkpoint["best_epoch"]
            or best["manifest_sha256"] != manifest_hash
            or best["best_validation"] != checkpoint["best_validation"]
        ):
            raise _error("invalid_checkpoint", "best and last checkpoints disagree")
    elif directory.exists():
        raise _error("training_output_exists", "new training output directory must not exist")

    with _torch_runtime(config):
        model = PageSegmenter(config.model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        splits = _load_splits(dataset, checked["manifest"], config.model_config)
        training, validation = splits["train"], splits["validation"]
        shuffle = torch.Generator(device="cpu").manual_seed(config.seed)
        if checkpoint is None:
            epoch, history, best_epoch = 0, [], 0
            initial = _evaluate(model, validation, config.batch_size)
            best_validation = dict(initial)
        else:
            try:
                model.load_state_dict(checkpoint["model_state"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                shuffle.set_state(checkpoint["shuffle_rng_state"])
                torch.set_rng_state(checkpoint["torch_rng_state"])
            except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                raise _error("invalid_checkpoint", f"invalid resume state: {exc}") from exc
            epoch, history = checkpoint["epoch"], list(checkpoint["history"])
            initial = checkpoint["initial_validation"]
            best_validation, best_epoch = checkpoint["best_validation"], checkpoint["best_epoch"]
        if checkpoint is None:
            directory.mkdir(parents=True, exist_ok=False)
        lock = directory / ".training.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise _error("training_in_progress", "output has an active training lock") from exc
        os.close(descriptor)

        def state():
            return {
                "schema_version": CHECKPOINT_SCHEMA,
                "config": asdict(config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "initial_validation": initial,
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "manifest_sha256": manifest_hash,
                "dataset_counts": checked["sample_counts"],
                "provenance": provenance,
                "torch_rng_state": torch.get_rng_state(),
                "shuffle_rng_state": shuffle.get_state(),
            }

        try:
            if checkpoint is None:
                _save_checkpoint(directory / "best.pt", state())
            for epoch in range(epoch + 1, config.epochs + 1):
                model.train()
                order = torch.randperm(len(training[0]), generator=shuffle)
                loss_sum, sample_weights = 0.0, 0
                for start in range(0, len(order), config.batch_size):
                    indices = order[start : start + config.batch_size]
                    targets = training[1][indices]
                    optimizer.zero_grad(set_to_none=True)
                    loss = segmentation_loss(model(training[0][indices]), targets)
                    if not bool(torch.isfinite(loss)):
                        raise _error("nonfinite_training", "nonfinite region training loss")
                    loss.backward()
                    if any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    ):
                        raise _error("nonfinite_training", "nonfinite region training gradient")
                    optimizer.step()
                    weight = len(indices) + int((targets.flatten(1).sum(1) == 0).sum())
                    loss_sum += float(loss.detach()) * weight
                    sample_weights += weight
                current = _evaluate(model, validation, config.batch_size)
                history.append(
                    {"epoch": epoch, "train_loss": loss_sum / sample_weights, "validation": current}
                )
                if current["loss"] < best_validation["loss"]:
                    best_validation, best_epoch = current, epoch
                    _save_checkpoint(directory / "best.pt", state())
                last_hash = _save_checkpoint(directory / "last.pt", state())
                report = {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "config": asdict(config),
                    "epochs_completed": epoch,
                    "history": history,
                    "initial_validation": initial,
                    "last_validation": current,
                    "best_validation": best_validation,
                    "best_epoch": best_epoch,
                    "manifest_sha256": manifest_hash,
                    "dataset_counts": checked["sample_counts"],
                    "provenance": provenance,
                    "checkpoints": {"last.pt": last_hash, "best.pt": sha256(directory / "best.pt")},
                }
                encoded = (
                    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode()
                _atomic(directory / "training.json", lambda stream: stream.write(encoded))
            return report
        finally:
            lock.unlink(missing_ok=True)
