"""Controlled architecture study with the historical loss and training schedule.

Only train/validation tensors enter this module. Historical trainers and saved
artifacts remain unchanged; new models have an explicit versioned study schema.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from ..contracts import FlattenerError, Status
from ..synthetic.scene_dataset import scene_loader, validate_scene_dataset
from .checkpoints import STUDY_SCHEMA, create_region_model, model_spec, read_region_checkpoint
from .segmentation import SegmenterConfig, preprocess_image, preprocess_mask, sha256
from .training import (
    _atomic,
    _evaluate,
    _provenance,
    _save_checkpoint,
    _torch_runtime,
    segmentation_loss,
)


@dataclass(frozen=True, slots=True)
class StudyTrainConfig:
    seed: int = 20260923
    epochs: int = 80
    batch_size: int = 16
    learning_rate: float = 0.001
    threads: int = 1
    architecture: str = "local"

    def __post_init__(self):
        for name, low, high in (
            ("seed", 0, 2**63 - 1),
            ("epochs", 1, 100_000),
            ("batch_size", 1, 1024),
            ("threads", 1, 128),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise _error("invalid_config", f"{name} must be an integer in [{low}, {high}]")
        try:
            valid_lr = (
                type(self.learning_rate) in (int, float)
                and 0 < self.learning_rate <= 1
                and math.isfinite(self.learning_rate)
            )
        except (OverflowError, TypeError, ValueError):
            valid_lr = False
        if (
            not valid_lr
            or not isinstance(self.architecture, str)
            or self.architecture not in ("local", "context")
        ):
            raise _error("invalid_config", "invalid learning rate or architecture")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))


def _error(reason, message):
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def _splits(directory, manifest, input_size):
    config = SegmenterConfig(input_size=tuple(input_size))
    selected = [row for row in manifest["samples"] if row["split"] in ("train", "validation")]
    if len(selected) * math.prod(config.input_size) > 64_000_000:
        raise _error("training_memory_limit", "train + validation exceed 64 million pixels")
    result = {}
    loader = scene_loader(directory)
    for split in ("train", "validation"):
        images, masks = [], []
        for record in selected:
            if record["split"] != split:
                continue
            sample = loader(directory, record)
            images.append(preprocess_image(sample.image, config))
            masks.append(preprocess_mask(sample.mask, config))
        if not masks:
            raise _error("invalid_dataset", f"{split} is empty")
        truth = torch.stack(masks)
        present = truth.flatten(1).any(1)
        if not bool(present.any()) or bool(present.all()):
            raise _error("invalid_dataset", f"{split} requires positives and negatives")
        result[split] = torch.stack(images), truth
    return result


def _resume(checkpoint, best, config, digest, provenance):
    if checkpoint["schema_version"] != STUDY_SCHEMA or best["schema_version"] != STUDY_SCHEMA:
        raise _error("resume_config_mismatch", "resume requires a study checkpoint")
    before, after, selected_config = (
        dict(checkpoint["config"]),
        asdict(config),
        dict(best["config"]),
    )
    before.pop("epochs", None)
    after.pop("epochs")
    selected_config.pop("epochs", None)
    if (
        before != after
        or selected_config != before
        or checkpoint["model_spec"] != model_spec(config.architecture)
    ):
        raise _error("resume_config_mismatch", "only total epoch budget may change")
    if checkpoint["manifest_sha256"] != digest or best["manifest_sha256"] != digest:
        raise _error("resume_dataset_mismatch", "training manifest changed")
    previous = checkpoint["provenance"]
    for key in ("versions", "uv_lock_sha256", "loss", "selection", "model_version"):
        if previous.get(key) != provenance[key] or best["provenance"].get(key) != provenance[key]:
            raise _error("resume_environment_mismatch", f"changed training provenance: {key}")
    # Compare every computation/data dependency. Evaluation-only additions do
    # not affect optimizer/RNG continuation and are recorded but not forbidden.
    sources = (
        "regions/training.py",
        "regions/experiment_training.py",
        "regions/checkpoints.py",
        "regions/segmentation.py",
        "regions/context_segmentation.py",
        "synthetic/scenes.py",
        "synthetic/scene_v2.py",
        "synthetic/paired_scenes.py",
        "synthetic/scene_dataset.py",
        "synthetic/documents.py",
        "synthetic/geometry.py",
        "synthetic/pipeline.py",
        "renderer.py",
        "coordinates.py",
        "validation.py",
        "image_io.py",
    )
    for name in sources:
        expected_source = provenance["source_code_sha256"].get(name)
        if (
            previous.get("source_code_sha256", {}).get(name) != expected_source
            or best["provenance"].get("source_code_sha256", {}).get(name) != expected_source
        ):
            raise _error("resume_environment_mismatch", f"learner dependency changed: {name}")
    if not 0 <= checkpoint["epoch"] < config.epochs:
        raise _error("resume_epoch_budget", "epochs must exceed completed checkpoint epoch")
    if (
        best["epoch"] != checkpoint["best_epoch"]
        or best["best_validation"] != checkpoint["best_validation"]
        or best["model_spec"] != checkpoint["model_spec"]
    ):
        raise _error("invalid_checkpoint", "best and last checkpoints disagree")


def train_region_study(dataset_dir, output_dir, config=None, resume=None):
    """Fit one fixed architecture; choose validation loss, never test masks."""
    config = StudyTrainConfig() if config is None else config
    if not isinstance(config, StudyTrainConfig):
        raise _error("invalid_config", "config must be StudyTrainConfig")
    directory, dataset = Path(output_dir).resolve(), Path(dataset_dir).resolve()
    checked = validate_scene_dataset(dataset)
    digest, spec = sha256(dataset / "manifest.json"), model_spec(config.architecture)
    provenance = _provenance()
    provenance.update(
        model_version=spec["model_version"],
        study_schema=STUDY_SCHEMA,
        dataset_schema=checked["manifest"]["schema_version"],
        model_spec=spec,
    )
    checkpoint = None
    if resume is not None:
        if Path(resume).resolve() != directory / "last.pt":
            raise _error("invalid_resume_path", "resume must name output_dir/last.pt")
        checkpoint = read_region_checkpoint(resume)
        _resume(
            checkpoint, read_region_checkpoint(directory / "best.pt"), config, digest, provenance
        )
    elif directory.exists():
        raise _error("training_output_exists", "new output directory must not exist")
    started = perf_counter()
    with _torch_runtime(config):
        model = create_region_model(config.architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        splits = _splits(dataset, checked["manifest"], spec["model_config"]["input_size"])
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
            except (TypeError, ValueError, RuntimeError, KeyError) as error:
                raise _error("invalid_checkpoint", str(error)) from error
            epoch, history = checkpoint["epoch"], list(checkpoint["history"])
            initial = checkpoint["initial_validation"]
            best_epoch, best_validation = checkpoint["best_epoch"], checkpoint["best_validation"]
        if checkpoint is None:
            directory.mkdir(parents=True, exist_ok=False)
        lock = directory / ".training.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise _error("training_in_progress", "active training lock exists") from error
        os.close(descriptor)

        def state():
            return {
                "schema_version": STUDY_SCHEMA,
                "config": asdict(config),
                "model_spec": spec,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "initial_validation": initial,
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "manifest_sha256": digest,
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
                loss_sum, weights_sum = 0.0, 0
                for start in range(0, len(order), config.batch_size):
                    indices = order[start : start + config.batch_size]
                    targets = training[1][indices]
                    optimizer.zero_grad(set_to_none=True)
                    loss = segmentation_loss(model(training[0][indices]), targets)
                    if not bool(torch.isfinite(loss)):
                        raise _error("nonfinite_training", "nonfinite study training loss")
                    loss.backward()
                    if any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    ):
                        raise _error("nonfinite_training", "nonfinite study gradient")
                    optimizer.step()
                    weight = len(indices) + int((targets.flatten(1).sum(1) == 0).sum())
                    loss_sum += float(loss.detach()) * weight
                    weights_sum += weight
                current = _evaluate(model, validation, config.batch_size)
                history.append(
                    {"epoch": epoch, "train_loss": loss_sum / weights_sum, "validation": current}
                )
                if current["loss"] < best_validation["loss"]:
                    best_epoch, best_validation = epoch, current
                    _save_checkpoint(directory / "best.pt", state())
                last_hash = _save_checkpoint(directory / "last.pt", state())
                report = {
                    "schema_version": STUDY_SCHEMA,
                    "config": asdict(config),
                    "model_spec": spec,
                    "epochs_completed": epoch,
                    "history": history,
                    "initial_validation": initial,
                    "last_validation": current,
                    "best_validation": best_validation,
                    "best_epoch": best_epoch,
                    "manifest_sha256": digest,
                    "dataset_counts": checked["sample_counts"],
                    "provenance": provenance,
                    "checkpoints": {"last.pt": last_hash, "best.pt": sha256(directory / "best.pt")},
                    "run_elapsed_seconds": perf_counter() - started,
                    "timing_scope": "this invocation only; not a controlled performance comparison",
                }
                encoded = (
                    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode()
                _atomic(directory / "training.json", lambda stream: stream.write(encoded))
                print(
                    f"{config.architecture} epoch {epoch}/{config.epochs}: "
                    f"validation loss={current['loss']:.6f}, best={best_epoch}",
                    flush=True,
                )
            return report
        finally:
            lock.unlink(missing_ok=True)
