"""Tilt probe: how well a dewarp checkpoint follows camera perspective, without benchmarks.

The probe predicts grids for the same validation samples in two views: the page crop
(``ViewConfig`` crop ``page``) and that crop seen by a tilted camera (crop ``tilt``,
frame corners moved by up to ``--tilt`` of the side). Both views are seeded by the view
name and the sample, so every run and checkpoint sees the same inputs. Per view it
reports:

- ``grid_error_pct``: mean distance between predicted and target nodes, in percent of
  the page diagonal (target corner to corner) in view pixels.
- ``undercorrection``: alpha of the fit e = -alpha f over all nodes of all samples,
  where e is the prediction error and f the foreshortening of the target, i.e. the
  homography patch through the four target corners minus their bilinear patch. 0 means
  the model follows the foreshortening; 1 means it spaces the page content like a
  bilinear patch through the corners. ``undercorrection_median`` is the median of the
  per-sample fits.
- ``epe_px``: mean node error in pixels of the model input.

A model that handles perspective keeps the tilted numbers close to the page-view
numbers. Grids must use the flattener convention (checkpoints of this package).

    python -m flattener.dewarp.probe --checkpoint runs/P1 \\
        --data-spec data-release.json --samples 400 --output probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ..contracts import FlattenerError, Status
from .composite import apply_homography, corner_homography
from .data import DataSpec, ViewConfig, build_dataset, expand_path
from .predict import Dewarper, ieee_float32, load_dewarper

PAGE_MARGIN = (0.02, 0.1)  # the docunet validation view of training
DEFAULT_SAMPLES = 400


def probe_views(tilt: float, margin: tuple[float, float] = PAGE_MARGIN) -> tuple[ViewConfig, ...]:
    """The page view and the tilted view the probe compares."""
    return (
        ViewConfig("page", "page", margin),
        ViewConfig(f"tilt{round(100 * tilt)}", "tilt", margin, tilt=tilt),
    )


def corner_patches(grid: NDArray) -> tuple[NDArray, NDArray]:
    """Bilinear and homography patches through the four corners of a (Gh, Gw, 2) grid,
    on the same node lattice."""
    rows, cols = grid.shape[:2]
    u, v = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    corners = np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]], dtype=np.float64)
    top_left, top_right, bottom_right, bottom_left = corners
    u3, v3 = u[..., None], v[..., None]
    bilinear = (1 - v3) * ((1 - u3) * top_left + u3 * top_right) + v3 * (
        (1 - u3) * bottom_left + u3 * bottom_right
    )
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    homography = apply_homography(corner_homography(unit, corners), np.stack([u, v], axis=-1))
    return bilinear, homography


def grid_metrics(prediction: NDArray, target: NDArray) -> dict[str, float]:
    """Metrics of one sample from (Gh, Gw, 2) grids in view pixels: error in % of the
    page diagonal, the under-correction fit terms (``ef``, ``ff``) and the mean error."""
    prediction, target = prediction.astype(np.float64), target.astype(np.float64)
    diagonal = float(np.hypot(*(target[-1, -1] - target[0, 0])))
    bilinear, homography = corner_patches(target)
    foreshortening = homography - bilinear
    error = prediction - target
    distance = np.hypot(error[..., 0], error[..., 1])
    ef = float(-(error * foreshortening).sum())
    ff = float((foreshortening * foreshortening).sum())
    return {
        "grid_error_pct": float(distance.mean() / diagonal * 100.0),
        "foreshortening_pct": float(
            np.hypot(foreshortening[..., 0], foreshortening[..., 1]).mean() / diagonal * 100.0
        ),
        "undercorrection": ef / ff if ff > 0 else float("nan"),
        "epe_px": float(distance.mean()),
        "ef": ef,
        "ff": ff,
    }


def _to_pixels(values: NDArray, size: tuple[int, int]) -> NDArray:
    height, width = size
    return (values.astype(np.float64) + 1.0) / 2.0 * np.array([width, height])


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Means over samples; the pooled under-correction; the median per-sample fit."""
    keys = ("grid_error_pct", "foreshortening_pct", "epe_px")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["grid_error_pct_median"] = float(np.median([row["grid_error_pct"] for row in rows]))
    ff = sum(row["ff"] for row in rows)
    summary["undercorrection"] = sum(row["ef"] for row in rows) / ff if ff > 0 else float("nan")
    fits = [row["undercorrection"] for row in rows if np.isfinite(row["undercorrection"])]
    summary["undercorrection_median"] = float(np.median(fits)) if fits else float("nan")
    summary["samples"] = len(rows)
    return summary


def tilt_probe(
    dewarper: Dewarper,
    spec: DataSpec,
    *,
    split: str = "val",
    samples: int = DEFAULT_SAMPLES,
    tilt: float = 0.15,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run the probe on the first ``samples`` items of ``split`` (see module docs)."""
    config = dewarper.config
    if config.grid_convention != "flattener":
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_checkpoint", "the probe needs flattener-convention grids"
        )
    views = probe_views(tilt)
    dataset = build_dataset(
        spec,
        track=spec.track,
        split=split,
        input_size=config.input_size,
        grid_size=config.grid_size,
        train=False,
        views=views,
    )
    if dataset is None:
        raise FlattenerError(Status.INPUT_ERROR, "missing_data", f"the spec has no {split} split")
    count = min(samples, len(dataset))
    rows: dict[str, list[dict[str, float]]] = {view.name: [] for view in views}
    for start in range(0, count, batch_size):
        items = [dataset[index] for index in range(start, min(start + batch_size, count))]
        images = torch.cat([item["image"] for item in items]).float().div_(255)
        with torch.inference_mode(), ieee_float32():
            grids, _ = dewarper.model(images.to(dewarper.device))
        grids = grids.float().cpu().permute(0, 2, 3, 1).numpy()
        for number, item in enumerate(items):
            for slot, view in enumerate(views):
                valid = item["valid"][slot, 0].numpy() > 0
                if not valid.all():
                    continue
                target = item["grid2d"][slot].permute(1, 2, 0).numpy()
                prediction = grids[number * len(views) + slot]
                rows[view.name].append(
                    grid_metrics(
                        _to_pixels(prediction, config.input_size),
                        _to_pixels(target, config.input_size),
                    )
                )
    return {
        "measurement": "tilt_probe",
        "checkpoint": dewarper.info.get("path"),
        "checkpoint_sha256": dewarper.info.get("sha256"),
        "data_spec": spec.name,
        "data_spec_sha256": spec.sha256,
        "split": split,
        "samples_requested": samples,
        "tilt": tilt,
        "views": [view.to_dict() for view in views],
        "results": {name: summarize(view_rows) for name, view_rows in rows.items() if view_rows},
    }


def resolve_checkpoint(path: str | Path) -> Path:
    """A checkpoint file; a ``best.json`` pointer or a run (or checkpoints) folder
    resolves to the best checkpoint it names."""
    path = Path(path)
    if path.is_dir():
        pointer = path / "best.json"
        path = pointer if pointer.is_file() else path / "checkpoints" / "best.json"
    if path.suffix == ".json":
        return path.parent / json.loads(path.read_text(encoding="utf-8"))["file"]
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True, help="checkpoint file, best.json or run folder"
    )
    parser.add_argument("--data-spec", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--tilt", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="write the JSON result here as well")
    args = parser.parse_args(argv)
    spec = DataSpec.load(expand_path(args.data_spec))
    dewarper = load_dewarper(resolve_checkpoint(args.checkpoint), device=args.device)
    result = tilt_probe(
        dewarper,
        spec,
        split=args.split,
        samples=args.samples,
        tilt=args.tilt,
        batch_size=args.batch_size,
    )
    text = json.dumps(result, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
