"""Deterministic CPU coordinate training with hash-checked data and safe checkpoints.

Only train and validation samples enter the learner. The test split is checked
for integrity by the dataset validator, but never loaded or used for selection.
Checkpoints contain tensors and ordinary Python data, and load with weights_only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torch.nn.functional as F

from ..contracts import FlattenerError, Status
from ..synthetic.pipeline import load_sample, validate_dataset
from .model import CoordinatePredictor, ModelConfig, dense_maps, preprocess_image

CHECKPOINT_SCHEMA = "learning-0.1"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 20260921
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-3
    threads: int = 1
    model_config: ModelConfig = field(default_factory=ModelConfig)


def _error(reason: str, message: str, status: Status = Status.INPUT_ERROR) -> FlattenerError:
    return FlattenerError(status, reason, message)


def _config(config: TrainConfig) -> TrainConfig:
    if not isinstance(config, TrainConfig):
        raise _error("invalid_config", "config must be TrainConfig")
    for name, low, high in (
        ("seed", 0, 2**63 - 1),
        ("epochs", 1, 100_000),
        ("batch_size", 1, 1024),
        ("threads", 1, 128),
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise _error("invalid_config", f"{name} must be an integer in [{low}, {high}]")
    if (
        isinstance(config.learning_rate, bool)
        or type(config.learning_rate) not in (int, float)
        or not math.isfinite(config.learning_rate)
        or not 0 < config.learning_rate <= 1
        or not isinstance(config.model_config, ModelConfig)
    ):
        raise _error("invalid_config", "learning_rate must be in (0, 1]; model_config invalid")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance() -> dict:
    package = Path(__file__).resolve().parents[1]
    lock = package.parent.parent / "uv.lock"
    return {
        "source_code_sha256": {
            str(path.relative_to(package)): _sha256(path) for path in sorted(package.rglob("*.py"))
        },
        "uv_lock_sha256": _sha256(lock) if lock.is_file() else None,
        "versions": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "pillow": str(PIL.__version__),
            "torch": str(torch.__version__),
        },
        "device": "cpu",
        "coordinate_loss": "valid GT weighted SmoothL1, beta=1 source pixel, mean per coordinate",
        "selection": "minimum validation mean source-pixel endpoint error; test never used",
        "train_mean_residual_method": "valid-weighted train mean, bilinear residual resize",
        "resume_scope": "epoch boundaries; same environment and learner source required",
    }


@contextmanager
def _torch_runtime(config: TrainConfig):
    """The global CPU RNG/settings are restored even when training raises.

    PyTorch's process-global settings mean callers must not train concurrently
    in the same process. CUDA state is never read or changed.
    """
    random_state = torch.get_rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        torch.random.default_generator.manual_seed(config.seed)
        yield
    finally:
        torch.set_rng_state(random_state)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def coordinate_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    photo_size: tuple[int, int],
) -> torch.Tensor:
    """SmoothL1(beta=1 pixel) on a fixed ground-truth mask, averaged over xy.

    Coordinates are normalized align_corners=False backward maps. Invalid GT
    points may be NaN; they are removed before arithmetic. Predicted bad points
    cannot shrink the denominator or select the pixels on which they are scored.
    """
    if (
        predicted.ndim != 4
        or predicted.shape[-1] != 2
        or target.shape != predicted.shape
        or valid.shape != predicted.shape[:-1]
        or valid.dtype != torch.bool
        or not bool(valid.any())
    ):
        raise _error("invalid_training_batch", "incompatible maps or empty GT mask")
    scale = predicted.new_tensor((photo_size[1] / 2, photo_size[0] / 2))
    delta = (predicted[valid] - target[valid]) * scale
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=1.0, reduction="mean")


@dataclass(slots=True)
class _Split:
    images: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor


def _load_splits(directory: Path, manifest: dict, config: ModelConfig):
    splits: dict[str, _Split] = {}
    output_size = photo_size = None
    total_pixels = 0
    for split in ("train", "validation"):
        images, targets, masks = [], [], []
        for record in manifest["samples"]:
            if record["split"] != split:
                continue
            sample = load_sample(directory, record)
            sizes = (tuple(sample.flat.shape[:2]), tuple(sample.photo.shape[:2]))
            if output_size is None:
                output_size, photo_size = sizes
            if sizes != (output_size, photo_size):
                raise _error(
                    "mixed_training_sizes", "this CPU learner requires uniform image sizes"
                )
            total_pixels += math.prod(output_size)
            if total_pixels > 64_000_000:
                raise _error("training_memory_limit", "train + validation exceed 64 million pixels")
            images.append(preprocess_image(sample.photo, config))
            targets.append(torch.from_numpy(sample.backward_map.copy()))
            masks.append(torch.from_numpy(sample.valid_mask.copy()))
        if not images:
            raise _error("invalid_dataset", f"{split} split is empty")
        splits[split] = _Split(torch.stack(images), torch.stack(targets), torch.stack(masks))
    return splits, output_size, photo_size


def _mean_residual(split: _Split, config: ModelConfig, output_size) -> torch.Tensor:
    identity = dense_maps(torch.zeros(1, 2, *config.grid_size), output_size)[0]
    weight = split.valid.sum(0).clamp_min(1)[..., None]
    residuals = torch.where(
        split.valid[..., None], split.targets - identity, torch.zeros_like(split.targets)
    )
    mean = (residuals.sum(0) / weight).permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(mean, size=config.grid_size, mode="bilinear", align_corners=False)


def _validation_epe(model, split, output_size, photo_size, batch_size) -> float:
    model.eval()
    total, count = 0.0, 0
    scale = split.targets.new_tensor((photo_size[1] / 2, photo_size[0] / 2))
    with torch.inference_mode():
        for start in range(0, len(split.images), batch_size):
            stop = start + batch_size
            valid = split.valid[start:stop]
            predicted = dense_maps(model(split.images[start:stop]), output_size)
            distances = torch.linalg.vector_norm(
                (predicted[valid] - split.targets[start:stop][valid]) * scale, dim=-1
            )
            if not bool(torch.isfinite(distances).all()):
                raise _error(
                    "nonfinite_training", "nonfinite validation prediction", Status.MODEL_ERROR
                )
            total += float(distances.double().sum())
            count += distances.numel()
    return total / count


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _save_checkpoint(path: Path, payload: dict) -> str:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = _sha256(path)
    _atomic_bytes(path.with_suffix(path.suffix + ".sha256"), f"{digest}\n".encode())
    return digest


def _read_checkpoint(path: Path) -> dict:
    try:
        digest = path.with_suffix(path.suffix + ".sha256").read_text().strip()
        if len(digest) != 64 or _sha256(path) != digest:
            raise _error("checkpoint_hash_mismatch", "checkpoint hash does not match its sidecar")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        ):
            raise _error("invalid_checkpoint", "unsupported training checkpoint")
        required = {
            "config",
            "model_state",
            "optimizer_state",
            "epoch",
            "history",
            "best_validation_epe_pixels",
            "initial_validation_epe_pixels",
            "best_epoch",
            "manifest_sha256",
            "provenance",
            "torch_rng_state",
            "shuffle_rng_state",
            "train_mean_residual",
            "dataset_output_size",
            "dataset_photo_size",
            "dataset_counts",
        }
        if not required.issubset(checkpoint):
            raise _error("invalid_checkpoint", "training checkpoint fields are missing")
        if (
            not isinstance(checkpoint["config"], dict)
            or not {
                "epochs",
                "seed",
                "batch_size",
                "learning_rate",
                "threads",
                "model_config",
            }.issubset(checkpoint["config"])
            or not isinstance(checkpoint["provenance"], dict)
            or not {"source_code_sha256", "versions", "uv_lock_sha256"}.issubset(
                checkpoint["provenance"]
            )
            or not isinstance(checkpoint["provenance"]["source_code_sha256"], dict)
            or not isinstance(checkpoint["history"], list)
            or isinstance(checkpoint["epoch"], bool)
            or not isinstance(checkpoint["epoch"], int)
            or checkpoint["epoch"] != len(checkpoint["history"])
            or not 0 <= checkpoint["best_epoch"] <= checkpoint["epoch"]
            or not math.isfinite(checkpoint["best_validation_epe_pixels"])
        ):
            raise _error("invalid_checkpoint", "invalid training checkpoint metadata")
        return checkpoint
    except (OSError, ValueError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError) as exc:
        raise _error("invalid_checkpoint", f"Cannot read checkpoint: {exc}") from exc


def _resume_compatible(checkpoint, config, manifest_hash, provenance):
    previous, requested = dict(checkpoint["config"]), asdict(config)
    previous.pop("epochs")
    requested.pop("epochs")
    if previous != requested:
        raise _error("resume_config_mismatch", "only the total epoch budget may change on resume")
    if checkpoint["manifest_sha256"] != manifest_hash:
        raise _error("resume_dataset_mismatch", "dataset manifest differs from checkpoint")
    old_provenance = checkpoint["provenance"]
    # Changes to evaluation or unrelated file adapters cannot affect gradients.
    for source in ("learning/model.py", "learning/training.py", "coordinates.py"):
        if old_provenance["source_code_sha256"].get(source) != provenance["source_code_sha256"].get(
            source
        ):
            raise _error("resume_environment_mismatch", f"learner source changed: {source}")
    for key in ("versions", "uv_lock_sha256"):
        if old_provenance[key] != provenance[key]:
            raise _error("resume_environment_mismatch", f"training environment changed: {key}")
    if not 0 <= checkpoint["epoch"] < config.epochs:
        raise _error("resume_epoch_budget", "epochs must exceed the completed checkpoint epoch")


def train(
    dataset_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: TrainConfig | None = None,
    resume: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Train a CPU model and save last.pt, best.pt and a readable training.json.

    New runs require a nonexistent output directory. Explicit resume accepts
    that directory's hash-checked last.pt, and permits a larger total epoch
    budget. Epoch-boundary resume reproduces uninterrupted weights and history
    in the same recorded environment. Concurrent writers are rejected.
    """
    config = _config(TrainConfig() if config is None else config)
    dataset = Path(dataset_dir).resolve()
    directory = Path(output_dir).resolve()
    checked = validate_dataset(dataset)
    manifest_hash = _sha256(dataset / "manifest.json")
    provenance = _provenance()
    checkpoint = None
    if resume is not None:
        checkpoint_path = Path(resume).resolve()
        if checkpoint_path != directory / "last.pt":
            raise _error("invalid_resume_path", "resume must name output_dir/last.pt")
        checkpoint = _read_checkpoint(checkpoint_path)
        _resume_compatible(checkpoint, config, manifest_hash, provenance)
        best_checkpoint = _read_checkpoint(directory / "best.pt")
        if (
            best_checkpoint["epoch"] != checkpoint["best_epoch"]
            or best_checkpoint["manifest_sha256"] != manifest_hash
            or best_checkpoint["best_validation_epe_pixels"]
            != checkpoint["best_validation_epe_pixels"]
        ):
            raise _error("invalid_checkpoint", "best and last checkpoints disagree")
    elif directory.exists():
        raise _error("training_output_exists", "new training output directory must not exist")

    with _torch_runtime(config):
        # Construct and load data before publishing a run directory, so routine
        # validation failures leave no new outputs behind.
        model = CoordinatePredictor(config.model_config).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        splits, output_size, photo_size = _load_splits(
            dataset, checked["manifest"], config.model_config
        )
        training, validation = splits["train"], splits["validation"]
        shuffle = torch.Generator(device="cpu").manual_seed(config.seed)
        mean_residual = _mean_residual(training, config.model_config, output_size)
        if checkpoint is None:
            epoch, history = 0, []
            initial_epe = _validation_epe(
                model, validation, output_size, photo_size, config.batch_size
            )
            best_epe, best_epoch = initial_epe, 0
        else:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            epoch, history = checkpoint["epoch"], list(checkpoint["history"])
            initial_epe = checkpoint["initial_validation_epe_pixels"]
            best_epe, best_epoch = (
                checkpoint["best_validation_epe_pixels"],
                checkpoint["best_epoch"],
            )
            shuffle.set_state(checkpoint["shuffle_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if not torch.equal(mean_residual, checkpoint["train_mean_residual"]):
                raise _error("invalid_checkpoint", "train baseline differs from dataset")
        if checkpoint is None:
            directory.mkdir(parents=True, exist_ok=False)
        lock = directory / ".training.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise _error(
                "training_in_progress", "output directory has an active training lock"
            ) from exc
        os.close(descriptor)

        def state():
            return {
                "schema_version": CHECKPOINT_SCHEMA,
                "config": asdict(config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "initial_validation_epe_pixels": initial_epe,
                "best_validation_epe_pixels": best_epe,
                "best_epoch": best_epoch,
                "manifest_sha256": manifest_hash,
                "dataset_output_size": output_size,
                "dataset_photo_size": photo_size,
                "dataset_counts": checked["sample_counts"],
                "provenance": provenance,
                "torch_rng_state": torch.get_rng_state(),
                "shuffle_rng_state": shuffle.get_state(),
                "train_mean_residual": mean_residual,
            }

        try:
            if checkpoint is None:
                _save_checkpoint(directory / "best.pt", state())
            for epoch in range(epoch + 1, config.epochs + 1):
                model.train()
                order = torch.randperm(len(training.images), generator=shuffle)
                loss_sum, coordinate_count = 0.0, 0
                for start in range(0, len(order), config.batch_size):
                    indices = order[start : start + config.batch_size]
                    mask = training.valid[indices]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = dense_maps(model(training.images[indices]), output_size)
                    loss = coordinate_loss(prediction, training.targets[indices], mask, photo_size)
                    if not bool(torch.isfinite(loss)):
                        raise _error(
                            "nonfinite_training", "nonfinite training loss", Status.MODEL_ERROR
                        )
                    loss.backward()
                    if any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    ):
                        raise _error("nonfinite_training", "nonfinite gradient", Status.MODEL_ERROR)
                    optimizer.step()
                    weight = int(mask.sum()) * 2
                    loss_sum += float(loss.detach()) * weight
                    coordinate_count += weight
                validation_epe = _validation_epe(
                    model, validation, output_size, photo_size, config.batch_size
                )
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": loss_sum / coordinate_count,
                        "validation_epe_pixels": validation_epe,
                    }
                )
                if validation_epe < best_epe:
                    best_epe, best_epoch = validation_epe, epoch
                    _save_checkpoint(directory / "best.pt", state())
                last_hash = _save_checkpoint(directory / "last.pt", state())
                report = {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "config": asdict(config),
                    "epochs_completed": epoch,
                    "initial_validation_epe_pixels": initial_epe,
                    "best_validation_epe_pixels": best_epe,
                    "best_epoch": best_epoch,
                    "last_validation_epe_pixels": validation_epe,
                    "history": history,
                    "manifest_sha256": manifest_hash,
                    "dataset_counts": checked["sample_counts"],
                    "dataset_output_size": output_size,
                    "dataset_photo_size": photo_size,
                    "provenance": provenance,
                    "checkpoints": {
                        "last.pt": last_hash,
                        "best.pt": _sha256(directory / "best.pt"),
                    },
                }
                _atomic_bytes(
                    directory / "training.json",
                    (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
                )
            return report
        finally:
            lock.unlink(missing_ok=True)
