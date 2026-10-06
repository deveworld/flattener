"""Blend two compatible dewarp checkpoints.

Floating-point tensors use (1 - weight) * A + weight * B.
Integer counters come from A.
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
    if first["model_config"] != second["model_config"]:
        raise SystemExit("error: the checkpoints have different model configurations")
    payload = {key: value for key, value in first.items() if key not in DROPPED}
    payload["model_state"] = blend(first["model_state"], second["model_state"], args.weight)
    payload["initialization"] = {
        "kind": "blend",
        "sources": [str(Path(args.a).resolve()), str(Path(args.b).resolve())],
        "weight": args.weight,
    }
    payload.pop("lineage", None)
    payload["created_utc"] = utc_now()
    payload["schema"] = CHECKPOINT_SCHEMA
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
