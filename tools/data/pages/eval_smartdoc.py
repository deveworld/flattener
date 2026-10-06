"""Page gate IoU on SmartDoc 2015 challenge 1 captures.

``python tools/data/pages/eval_smartdoc.py GATE.pt [GATE2.pt ...] [--families 5] [--root DIR]``

Each capture's page probability grid (the gate's own preprocessing over the whole frame)
is upsampled bilinearly to the frame at a long side of 384, thresholded at 0.5 and
compared with the page quad. The report gives the mean IoU over the chosen captures.
Document family 5, held out from training, is the default evaluation split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from flattener.regions import context

ROOT = Path(__file__).resolve().parents[3]


def iou(gate, path: Path, corners, long_side: int = 384) -> float:
    with Image.open(path) as picture:
        image = np.asarray(picture.convert("RGB"))
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    size = (round(height * scale), round(width * scale))
    probability = torch.from_numpy(gate.predict_probability(image))[None, None]
    predicted = F.interpolate(probability, size=size, mode="bilinear", align_corners=False)
    predicted = predicted[0, 0].numpy() >= 0.5
    mask = Image.new("L", (size[1], size[0]), 0)
    ImageDraw.Draw(mask).polygon([(x * scale, y * scale) for x, y in corners], fill=255)
    truth = np.asarray(mask) > 127
    union = (predicted | truth).sum()
    return float((predicted & truth).sum() / union) if union else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", nargs="+")
    parser.add_argument("--root", default=str(ROOT / "outputs/benchmarks/smartdoc"))
    parser.add_argument("--families", type=int, nargs="+", default=[5])
    args = parser.parse_args(argv)
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    captures = [
        c for c in manifest["captures"]
        if int(re.search(r"(\d+)$", c["template"]).group(1)) in args.families
    ]  # fmt: skip
    torch.set_num_threads(2)
    report = {}
    for weights in args.weights:
        gate = context.load_context_segmenter(weights)
        scores = [(c["template"], iou(gate, root / c["path"], c["corners_xy"])) for c in captures]
        report[weights] = {
            "captures": len(scores),
            "mean_iou": round(float(np.mean([s for _, s in scores])), 4),
            "below_0.9": sum(s < 0.9 for _, s in scores),
        }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
