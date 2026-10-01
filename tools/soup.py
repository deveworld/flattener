"""Blend two dewarp checkpoints of one lineage: ``python tools/soup.py OUT A B [--weight W]``.

The weights of OUT are ``(1 - W) * A + W * B`` (float tensors, BatchNorm statistics
included; integer counters come from A), the rest of the payload is A's without its
optimizer, scaler and RNG state, and the lineage gains a ``soup`` step naming both
parents as checkpoint entries (sha256, step, track, data spec and its sha256, lineage)
with their weight, which ``flattener.scan.shippable`` checks like a fine-tune's start.
B should be a fine-tune of A (or both fine-tunes of one checkpoint), so the two sit in
one basin and a blend keeps what each learned (weight-space ensembling, as in WiSE-FT).
Both must be release-track checkpoints of the same model configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from flattener.dewarp.checkpoint import (
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    utc_now,
)

DROPPED = ("optimizer_state", "scaler_state", "rng_state")


def blend(a: dict, b: dict, weight: float) -> dict:
    """``a`` and ``b`` model states blended; integer tensors are ``a``'s."""
    if set(a) != set(b):
        raise SystemExit("error: the checkpoints hold different parameters")
    out = {}
    for name, first in a.items():
        second = b[name]
        if first.shape != second.shape:
            raise SystemExit(f"error: {name} has shapes {first.shape} and {second.shape}")
        if first.is_floating_point():
            out[name] = torch.lerp(first.float(), second.float(), weight).to(first.dtype)
        else:
            out[name] = first.clone()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--weight", type=float, default=0.5, help="share of B; default 0.5")
    args = parser.parse_args(argv)
    if not 0.0 <= args.weight <= 1.0:
        raise SystemExit("error: --weight must be in [0, 1]")
    first, second = load_checkpoint(args.a), load_checkpoint(args.b)
    for payload, path in ((first, args.a), (second, args.b)):
        if payload.get("track") != "release":
            raise SystemExit(f"error: {path} is not a release-track checkpoint")
    if first["model_config"] != second["model_config"]:
        raise SystemExit("error: the checkpoints have different model configurations")
    payload = {key: value for key, value in first.items() if key not in DROPPED}
    payload["model_state"] = blend(first["model_state"], second["model_state"], args.weight)
    parents = [  # checkpoint entries as a fine-tune's start records them, plus the share
        {
            "kind": "checkpoint",
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "step": source["step"],
            "track": source["track"],
            "model_config": source["model_config"],
            "data_spec_sha256": source["data_spec_sha256"],
            "data_spec": source.get("data_spec"),
            "lineage": source["lineage"],
            "weight": share,
        }
        for path, source, share in (
            (args.a, first, 1.0 - args.weight),
            (args.b, second, args.weight),
        )
    ]
    payload["lineage"] = [
        *first["lineage"],
        {"kind": "soup", "track": "release", "parents": parents, "created_utc": utc_now()},
    ]
    payload["created_utc"] = utc_now()
    payload["schema"] = CHECKPOINT_SCHEMA
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sidecar = save_checkpoint(args.output, payload)
    print(f"{args.output} sha256={sidecar['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
