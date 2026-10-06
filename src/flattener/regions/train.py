"""Train page detection from scene images and masks.

Data comes from tools/data/pages/build_data.py.
Training uses BCE and Dice losses with AdamW; validation measures page IoU.
The best and final weights are saved as best.pt and last.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ..dewarp.checkpoint import environment, utc_now, write_json
from ..dewarp.data import expand_path
from .context import (
    MODEL_VERSION,
    ContextPageSegmenter,
    ContextSegmenterConfig,
    parameter_count,
    save_context_checkpoint,
)
from .segmentation import PREPROCESS_VERSION, SegmenterConfig, preprocess_image, preprocess_mask

DATA_SCHEMA = "flattener-gate-data-1"


@dataclass
class TrainConfig:
    data: str
    output_dir: str
    name: str = "page-gate"
    track: str = "release"
    seed: int = 0
    input_size: tuple[int, int] = (192, 144)
    channels: tuple[int, int, int, int] = (16, 32, 64, 128)
    source_weights: dict[str, float] = field(default_factory=dict)
    batch_size: int = 64
    total_steps: int = 20000
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.02
    grad_clip: float = 5.0
    amp: bool = True
    jitter: float = 0.25
    num_workers: int = 8
    device: str = "cuda"
    log_every: int = 100
    val_every: int = 2000

    def __post_init__(self) -> None:
        self.input_size = tuple(self.input_size)
        self.channels = tuple(self.channels)
        if self.track not in ("release", "research"):
            raise ValueError("track must be release or research")
        if self.total_steps < 1 or self.batch_size < 1 or self.warmup_steps >= self.total_steps:
            raise ValueError("total_steps, batch_size and warmup_steps are out of range")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        data = dict(data)
        for key in ("data", "output_dir"):
            data[key] = expand_path(data[key])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def load_data(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((root / "gate-data.json").read_text(encoding="utf-8"))
    if summary.get("schema") != DATA_SCHEMA or not summary.get("complete"):
        raise ValueError(f"{root}: not a complete {DATA_SCHEMA} build")
    lines = (root / summary["records_file"]).read_text(encoding="utf-8").splitlines()
    return summary, [json.loads(line) for line in lines if line.strip()]


def augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, jitter: float
) -> tuple[np.ndarray, np.ndarray]:
    turns = int(rng.integers(4))
    image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if jitter > 0:
        values = image.astype(np.float32) / 255.0
        gain = rng.uniform(1 - jitter, 1 + jitter)
        values = (values - 0.5) * rng.uniform(1 - jitter, 1 + jitter) + 0.5 * gain
        values *= rng.uniform(1 - jitter / 3, 1 + jitter / 3, size=3)
        gray = values.mean(axis=-1, keepdims=True)
        values = gray + (values - gray) * rng.uniform(1 - jitter * 2, 1 + jitter)
        values = np.clip(values, 0, 1) ** rng.uniform(0.8, 1.25)
        image = np.clip(values * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class GateDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, rows: list[dict[str, Any]], size, seed: int, jitter: float,
                 train: bool):  # fmt: skip
        self.root, self.rows, self.seed, self.jitter, self.train = root, rows, seed, jitter, train
        self.config = SegmenterConfig(input_size=tuple(size))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        index, draw = key if isinstance(key, tuple) else (key, 0)
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as picture:
            image = np.asarray(picture.convert("RGB"))
        with Image.open(self.root / row["mask"]) as picture:
            mask = np.asarray(picture.convert("L")) > 127
        if self.train:
            rng = np.random.default_rng([self.seed, draw])
            image, mask = augment(image, mask, rng, self.jitter)
        return {
            "image": preprocess_image(image, self.config),
            "mask": preprocess_mask(mask, self.config),
            "source": row["source"],
        }


class DrawSampler(torch.utils.data.Sampler):
    """``steps * batch`` (row, draw) keys; a source's rows share its weight."""

    def __init__(self, rows, weights: dict[str, float], count: int, seed: int, start: int = 0):
        sources = [row["source"] for row in rows]
        totals = {name: sources.count(name) for name in set(sources)}
        share = np.array([weights.get(s, 1.0) / totals[s] for s in sources], np.float64)
        self.p, self.count, self.seed, self.start = share / share.sum(), count, seed, start

    def __len__(self) -> int:
        return self.count - self.start

    def __iter__(self):
        rng = np.random.default_rng([self.seed, 7])
        picks = rng.choice(len(self.p), size=self.count, p=self.p)
        for draw in range(self.start, self.count):
            yield int(picks[draw]), draw


def learning_rate(config: TrainConfig, step: int) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    t = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    low = config.lr * config.min_lr_ratio
    return low + 0.5 * (config.lr - low) * (1 + math.cos(math.pi * min(t, 1.0)))


def gate_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((1, 2, 3))
    probability = torch.sigmoid(logits)
    overlap = (probability * target).sum((1, 2, 3))
    total = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - (2 * overlap + 1) / (total + 1)
    return (bce + dice).mean()


def row_scores(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU at 0.5 per row; one minus the predicted share for a row without page."""
    predicted = logits > 0
    truth = target > 0.5
    inter = (predicted & truth).sum((1, 2, 3)).float()
    union = (predicted | truth).sum((1, 2, 3)).float()
    empty = truth.sum((1, 2, 3)) == 0
    share = predicted.float().mean((1, 2, 3))
    return torch.where(empty, 1 - share, inter / union.clamp(min=1))


@torch.no_grad()
def validate(model, loader, device) -> dict[str, Any]:
    model.eval()
    per_source: dict[str, list[float]] = {}
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        scores = row_scores(logits, batch["mask"].to(device)).cpu().tolist()
        for source, score in zip(batch["source"], scores, strict=True):
            per_source.setdefault(source, []).append(score)
    model.train()
    means = {name: float(np.mean(values)) for name, values in sorted(per_source.items())}
    return {"score": float(np.mean(list(means.values()))), "sources": means,
            "rows": {k: len(v) for k, v in per_source.items()}}  # fmt: skip


def provenance(config: TrainConfig, summary: dict[str, Any], step: int, metrics) -> dict:
    return {
        "name": config.name,
        "model_version": MODEL_VERSION,
        "track": config.track,
        "initialization": "scratch",
        "lineage": [
            {"kind": "scratch"},
            {"kind": "train", "track": config.track, "data": summary["name"], "steps": step},
        ],
        "data_spec": summary["data_spec"],
        "data_sources": summary["data_sources"],
        "training": {
            "config": config.to_dict(),
            "step": step,
            "validation": metrics,
            "loss": "per-image BCE + soft Dice (smooth 1)",
            "preprocess_version": PREPROCESS_VERSION,
            "environment": environment(),
            "saved_utc": utc_now(),
        },
    }


def train(config: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    root, output = Path(config.data), Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, rows = load_data(root)
    if config.track == "release" and summary["track"] != "release":
        raise ValueError("a release run needs a release data build")
    write_json(output / "config.json", config.to_dict())
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    device = torch.device(config.device)
    model_config = ContextSegmenterConfig(input_size=config.input_size, channels=config.channels)
    model = ContextPageSegmenter(model_config).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    other = [p for n, p in model.named_parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": other, "weight_decay": 0.0}], lr=config.lr,
    )  # fmt: skip
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    count = config.total_steps * config.batch_size
    sampler = DrawSampler(train_rows, config.source_weights, count, config.seed)
    loader = torch.utils.data.DataLoader(
        GateDataset(root, train_rows, config.input_size, config.seed, config.jitter, True),
        batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=config.num_workers > 0,
        drop_last=True,
    )  # fmt: skip
    val_loader = torch.utils.data.DataLoader(
        GateDataset(root, val_rows, config.input_size, config.seed, 0.0, False),
        batch_size=256,
        num_workers=min(4, config.num_workers),
    )
    log = (output / "train_log.jsonl").open("a", encoding="utf-8")

    def record(entry: dict[str, Any]) -> None:
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)

    record({"kind": "start", "parameters": parameter_count(model_config),
            "train_rows": len(train_rows), "val_rows": len(val_rows),
            "data": summary["name"], "config": config.to_dict()})  # fmt: skip
    best, started, seen, total = None, time.perf_counter(), 0, 0.0
    model.train()
    step = 0
    for batch in loader:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step)
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model._forward_trusted(images)
        loss = gate_loss(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        seen += images.shape[0]
        total = total + loss.detach()
        if step % config.log_every == 0:
            elapsed = time.perf_counter() - started
            record(
                {
                    "kind": "train",
                    "step": step,
                    "loss": float(total) / config.log_every,
                    "lr": learning_rate(config, step - 1),
                    "images_per_sec": seen / elapsed,
                }
            )
            total = 0.0
        if step % config.val_every == 0 or step == config.total_steps:
            metrics = validate(model, val_loader, device)
            record({"kind": "val", "step": step, **metrics})
            state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
            meta = provenance(config, summary, step, metrics)
            save_context_checkpoint(output / "last.pt", state, model_config, meta)
            if best is None or metrics["score"] > best["score"]:
                best = {"step": step, **metrics}
                save_context_checkpoint(output / "best.pt", state, model_config, meta)
                write_json(output / "best.json", best)
        if step >= config.total_steps:
            break
    record({"kind": "done", "step": step, "best": best,
            "seconds": time.perf_counter() - started})  # fmt: skip
    log.close()
    return best or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="run config JSON")
    args = parser.parse_args(argv)
    config = TrainConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
