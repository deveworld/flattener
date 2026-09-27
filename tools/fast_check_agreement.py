"""Compare ``flattener.scan.fast_check`` with ``flattener.render`` on many grid maps.

Usage (at low CPU priority, for example ``nice -n 19``, two threads):
  .venv/bin/python tools/fast_check_agreement.py --out REPORT.json
      [--weights R4.pt --docunet-crops DIR --smartdoc DIR [--smartdoc-count N]]
      [--synthetic N] [--seed S]

Each map is a grid (flattener values of a source), an output size, an inset and a border.
Both checks run on the same source; the report lists, per map, both decisions (status and
reason code), the out-of-bounds fractions, the minimum Jacobian ratios, whether the
full-resolution images are byte-identical (maps up to 4 MP that pass both checks) and the
seconds each check took. ``disagreements`` lists every map whose decisions differ.

Families:
* ``r4_docunet``, ``r4_docunet_rerun``: the release model's grid on DocUNet crops (whole
  crop, and one deskewed crop pass), at the crop's size (evaluation data, no tuning).
* ``r4_smartdoc``: its whole-photo grid on SmartDoc captures, at the auto output size.
* ``smooth``: affine, perspective and curl with small noise.
* ``near_fold``: one interior node pushed toward and past its neighbour in steps, so the
  fold decision flips inside the family.
* ``sub_pixel_fold``: a fold confined near one node on a large output.
* ``out_of_bounds``: a grid shifted so the out-of-bounds fraction is near 0.25.
* ``inset_border``: insets and borders on smooth grids.
* ``tiny``: outputs from 2 to 24 pixels a side.
* ``large``: 12 MP outputs, for timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flattener.contracts import RenderConfig, Status  # noqa: E402
from flattener.dewarp.coords import edge_positions  # noqa: E402
from flattener.dewarp.predict import (  # noqa: E402
    DESKEW_MAX_DEG,
    crop_pass,
    load_rgb,
    predict_grid,
)
from flattener.renderer import render  # noqa: E402
from flattener.scan.fast_check import fast_check  # noqa: E402
from flattener.scan.io import SourceImage  # noqa: E402
from flattener.scan.output_policy import plan_output  # noqa: E402
from flattener.scan.placement import dense_map  # noqa: E402
from flattener.scan.records import SourceInfo  # noqa: E402

GRID = (45, 31)
CONFIG = RenderConfig(on_error="return")
IMAGE_COMPARE_PIXELS = 4_000_000


def _nodes(grid_size=GRID) -> np.ndarray:
    gh, gw = grid_size
    return np.stack(np.meshgrid(edge_positions(gw), edge_positions(gh)), -1)


def _smooth(rng: np.random.Generator, grid_size=GRID, noise: float = 0.002) -> np.ndarray:
    """A page seen with some rotation, perspective and curl, inside [-1, 1]."""
    u = _nodes(grid_size)
    angle = math.radians(rng.uniform(-12, 12))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    scale = rng.uniform(0.55, 0.85, 2)
    points = (u * scale) @ rotation.T
    tilt = rng.uniform(-0.25, 0.25, 2)
    points = points / (1.0 + (u @ tilt)[..., None] * 0.3)
    curl = rng.uniform(0, 0.06)
    points[..., 1] += curl * np.cos(u[..., 0] * math.pi / 2)
    points += rng.normal(0, noise, points.shape)
    return points + rng.uniform(-0.05, 0.05, 2)


def synthetic_maps(seed: int, count: int) -> Iterator[dict[str, Any]]:
    """``count`` synthetic maps from the families of the module docstring."""
    rng = np.random.default_rng(seed)
    families = [
        ("smooth", 0.24),
        ("near_fold", 0.22),
        ("sub_pixel_fold", 0.10),
        ("out_of_bounds", 0.16),
        ("inset_border", 0.12),
        ("tiny", 0.12),
        ("large", 0.04),
    ]
    counts = [max(1, round(count * share)) for _, share in families]
    counts[0] += count - sum(counts)
    for (family, _), number in zip(families, counts, strict=True):
        for index in range(number):
            item: dict[str, Any] = {"family": family, "index": index, "inset": 0.0, "border": 0.0}
            size = (int(rng.integers(300, 1400)), int(rng.integers(300, 1400)))
            grid = _smooth(rng)
            if family == "near_fold":
                # Move node (i, j) toward (i, j + 1): the cells between them fold past 100 %.
                i, j = int(rng.integers(2, GRID[0] - 2)), int(rng.integers(2, GRID[1] - 3))
                step = grid[i, j + 1] - grid[i, j]
                fraction = np.linspace(0.8, 1.2, number)[index]
                grid[i, j] = grid[i, j] + step * fraction
                item["fraction"] = float(fraction)
            elif family == "sub_pixel_fold":
                i, j = int(rng.integers(2, GRID[0] - 2)), int(rng.integers(2, GRID[1] - 3))
                step = grid[i, j + 1] - grid[i, j]
                fraction = 1.0 + 10.0 ** rng.uniform(-4, -1)
                grid[i, j] = grid[i, j] + step * fraction
                size = (int(rng.integers(1500, 3000)), int(rng.integers(1100, 2200)))
                item["fraction"] = float(fraction)
            elif family == "out_of_bounds":
                target = rng.uniform(0.2, 0.3)
                grid = _smooth(rng) * 1.05
                grid[..., 0] += _shift_for(grid, size, target)
                item["target_fraction"] = float(target)
            elif family == "inset_border":
                item["inset"] = float(rng.choice([0.0, 0.01, 0.03]))
                item["border"] = float(rng.choice([0.0, 0.02, 0.05]))
            elif family == "tiny":
                size = (int(rng.integers(2, 25)), int(rng.integers(2, 25)))
                grid = _smooth(rng, noise=0.01)
            elif family == "large":
                size = (4000, 3000)
            item.update(grid=grid, size=size)
            yield item


def _shift_for(grid: np.ndarray, size: tuple[int, int], target: float) -> float:
    """The x shift that puts about ``target`` of the output outside the source."""
    low, high = 0.0, 2.0
    for _ in range(30):
        middle = (low + high) / 2
        moved = grid.copy()
        moved[..., 0] += middle
        dense = dense_map(moved, (max(2, size[0] // 8), max(2, size[1] // 8)))
        outside = float((np.abs(dense) > 1).any(axis=-1).mean())
        low, high = (middle, high) if outside < target else (low, middle)
    return (low + high) / 2


def _levels(image: np.ndarray):
    source = SourceImage(image, SourceInfo(size=tuple(image.shape[:2])))
    return source.level


def compare(image: np.ndarray, item: dict[str, Any]) -> dict[str, Any]:
    """Both checks on one map."""
    grid, size, inset, border = item["grid"], tuple(item["size"]), item["inset"], item["border"]
    started = time.perf_counter()
    backward = dense_map(grid, size, inset=inset, border=border)
    full = render(image, backward, config=CONFIG)
    render_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fast = fast_check(
        image, grid, size, inset=inset, border=border, config=CONFIG, levels=_levels(image)
    )
    fast_seconds = time.perf_counter() - started
    same_image = None
    if full.status == Status.SUCCESS and fast.ok and size[0] * size[1] <= IMAGE_COMPARE_PIXELS:
        same_image = fast.sample(image, CONFIG.fill_rgb).tobytes() == full.image.tobytes()
    rendered, checked = full.metadata.get("validation") or {}, fast.validation
    row = {key: value for key, value in item.items() if key not in ("grid",)}
    row.update(
        size=list(size),
        grid_sha256=hashlib.sha256(np.ascontiguousarray(grid).tobytes()).hexdigest()[:16],
        render={
            "status": str(full.status),
            "reason": full.reason_code,
            "out_of_bounds_fraction": rendered.get("out_of_bounds_fraction"),
            "min_jacobian_ratio": rendered.get("min_jacobian_ratio"),
            "failed_cells": rendered.get("failed_cells"),
            "seconds": render_seconds,
        },
        fast={
            "status": str(fast.status),
            "reason": fast.reason_code,
            "out_of_bounds_fraction": checked.get("out_of_bounds_fraction"),
            "min_jacobian_ratio": checked.get("min_jacobian_ratio"),
            "failed_cells": checked.get("failed_cells"),
            "straddling_cells": checked.get("straddling_cells"),
            "continuous_min_jacobian_ratio": checked.get("continuous_min_jacobian_ratio"),
            "proxy_size": fast.metadata["check_resolution"]["proxy"],
            "proxy_status": (fast.metadata.get("proxy") or {}).get("status"),
            "proxy_reason": (fast.metadata.get("proxy") or {}).get("reason_code"),
            "seconds": fast_seconds,
        },
        agree=(full.status, full.reason_code) == (fast.status, fast.reason_code),
        same_image=same_image,
    )
    return row


def r4_maps(args: argparse.Namespace) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """(image, map) pairs from the release model on DocUNet crops and SmartDoc captures."""
    import torch

    from flattener.dewarp.predict import _flattener_values
    from flattener.scan.shippable import load_weights

    torch.set_num_threads(args.threads)
    dewarper = load_weights(args.weights)
    convention = dewarper.config.grid_convention
    if args.docunet_crops is not None:
        paths = sorted(Path(args.docunet_crops).glob("*copy.png"))[: args.docunet_count]
        for path in paths:
            image = load_rgb(path)
            whole = _flattener_values(predict_grid(dewarper, image), convention, image.shape[:2])
            item = {"family": "r4_docunet", "input": path.name, "inset": 0.0, "border": 0.0}
            yield image, {**item, "grid": whole, "size": image.shape[:2]}
            rerun, record = crop_pass(dewarper, image, whole, deskew=True)
            item = {**item, "family": "r4_docunet_rerun", "deskew": record["angle_deg"]}
            yield image, {**item, "grid": rerun, "size": image.shape[:2]}
    if args.smartdoc is not None:
        paths = sorted(Path(args.smartdoc).glob("smartdoc-capture-*.jpeg"))
        step = max(1, len(paths) // max(1, args.smartdoc_count))
        for path in paths[::step][: args.smartdoc_count]:
            image = load_rgb(path)
            whole = _flattener_values(predict_grid(dewarper, image), convention, image.shape[:2])
            try:
                size = plan_output(whole, image.shape[:2]).size
            except Exception:  # a degenerate grid: check it at the photo's size
                size = image.shape[:2]
            item = {"family": "r4_smartdoc", "input": path.name, "inset": 0.0, "border": 0.0}
            yield image, {**item, "grid": whole, "size": tuple(size)}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = families.setdefault(
            row["family"],
            {"maps": 0, "agree": 0, "render_failed": 0, "same_image": 0, "compared_images": 0},
        )
        entry["maps"] += 1
        entry["agree"] += bool(row["agree"])
        entry["render_failed"] += row["render"]["status"] != "success"
        if row["same_image"] is not None:
            entry["compared_images"] += 1
            entry["same_image"] += bool(row["same_image"])
    large = [row for row in rows if row["size"][0] * row["size"][1] >= 6_000_000]

    def median(values):
        return float(np.median(values)) if values else None

    return {
        "maps": len(rows),
        "agree": sum(bool(row["agree"]) for row in rows),
        "disagree": sum(not row["agree"] for row in rows),
        "images_compared": sum(row["same_image"] is not None for row in rows),
        "images_identical": sum(bool(row["same_image"]) for row in rows),
        "sub_pixel_folds": sum(
            row["render"]["status"] == "success"
            and (row["fast"]["continuous_min_jacobian_ratio"] or 1.0) <= CONFIG.min_jacobian_ratio
            for row in rows
        ),
        "families": families,
        "seconds_at_6mp_or_more": {
            "maps": len(large),
            "render_median": median([row["render"]["seconds"] for row in large]),
            "fast_check_median": median([row["fast"]["seconds"] for row in large]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--docunet-crops", type=Path)
    parser.add_argument("--docunet-count", type=int, default=130)
    parser.add_argument("--smartdoc", type=Path)
    parser.add_argument("--smartdoc-count", type=int, default=60)
    parser.add_argument("--synthetic", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    synthetic_source = rng.integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    for item in synthetic_maps(args.seed, args.synthetic):
        rows.append(compare(synthetic_source, item))
    if args.weights is not None:
        for image, item in r4_maps(args):
            rows.append(compare(image, item))
    report = {
        "schema": "flattener-fast-check-agreement-1",
        "created_seconds": time.perf_counter() - started,
        "settings": {
            "seed": args.seed,
            "synthetic": args.synthetic,
            "weights": None if args.weights is None else str(args.weights),
            "docunet_crops": None if args.docunet_crops is None else str(args.docunet_crops),
            "smartdoc": None if args.smartdoc is None else str(args.smartdoc),
            "deskew_max_deg": DESKEW_MAX_DEG,
            "render_config": {
                "max_out_of_bounds_fraction": CONFIG.max_out_of_bounds_fraction,
                "min_jacobian_ratio": CONFIG.min_jacobian_ratio,
            },
        },
        "summary": summary(rows),
        "disagreements": [row for row in rows if not row["agree"]],
        "different_images": [row for row in rows if row["same_image"] is False],
        "maps": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    return 0 if not report["disagreements"] else 1


if __name__ == "__main__":
    sys.exit(main())
