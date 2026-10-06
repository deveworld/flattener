"""Regenerate browser parity fixtures from the existing Python implementation.

uv run python tools/web/generate_fixtures.py
These fixtures exercise spatial preprocessing, view placement, text patch selection,
quarter turns, output sizing, and sampling, independently of the TypeScript port.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from flattener.dewarp.predict import estimate_page_angle, model_input, outline_points
from flattener.regions.context import ContextSegmenterConfig, preprocess_image
from flattener.scan.orient import OrientConfig, prepare_views
from flattener.scan.output_policy import plan_output
from flattener.scan.placement import Placement, compose, dense_map, quarter_turn_grid


def main():
    height, width = 281, 397
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        [(x * 17 + y * 3) % 256, (x * 5 + y * 13) % 256, ((x // 7 + y // 5) % 2) * 230], -1
    ).astype(np.uint8)
    dewarper = SimpleNamespace(device="cpu", config=SimpleNamespace(input_size=(13, 11)))
    yy, xx = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 9), indexing="ij")
    grid = np.stack(
        [xx * 0.8 + 0.03 * np.sin(yy * np.pi), yy * 0.84 + 0.045 * np.sin(xx * np.pi)], -1
    )
    angle, _ = estimate_page_angle(grid, (height, width))
    placement = Placement.from_outline(
        outline_points(grid, (height, width)), angle, (height, width)
    )
    config = OrientConfig(global_size=32, patch_size=16, page_long_side=128)
    views = prepare_views(image, config)
    size = (17, 13)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float()
    sampled = F.grid_sample(
        tensor,
        torch.from_numpy(dense_map(grid, size))[None],
        padding_mode="border",
        align_corners=False,
    )
    rgb = np.floor(sampled[0].permute(1, 2, 0).numpy().clip(0, 255) + 0.5).astype(np.uint8)

    def flat(value):
        return np.asarray(value).reshape(-1).tolist()

    result = {
        "source": {
            "width": width,
            "height": height,
            "formula": "see tools/web/generate_fixtures.py",
        },
        "model_input": flat(model_input(dewarper, image)),
        "gate_input": flat(preprocess_image(image, ContextSegmenterConfig(input_size=(19, 17)))),
        "grid": {"rows": 7, "cols": 9, "data": flat(grid)},
        "angle": angle,
        "placement": {
            "matrix": flat(placement.matrix),
            "offset": flat(placement.offset),
            "height": placement.view_size[0],
            "width": placement.view_size[1],
        },
        "composed": flat(compose(grid, placement)),
        "turned": flat(quarter_turn_grid(grid, 1)),
        "dense": flat(dense_map(grid, size)),
        "render": flat(rgb),
        "output_size": list(plan_output(grid, (height, width), max_side=2560).size),
        "orientation": {
            "config": config.to_dict(),
            "global": flat(views.global_view),
            "patches": flat(views.patches),
            "cells": flat(views.cells),
            "energy": flat(views.energy),
        },
    }
    path = Path(__file__).resolve().parents[2] / "web/tests/fixtures/python.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
