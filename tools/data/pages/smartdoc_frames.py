"""A capture manifest over the full SmartDoc 2015 challenge 1 frames, for the gate builder.

``python tools/data/pages/smartdoc_frames.py ROOT [--every 6]``

ROOT holds the extracted ``frames.tar.gz`` of the v2.0.0 release (CC BY 4.0):
``metadata.csv.gz`` and ``background0N/<document>/frame_NNNN.jpeg``. Every ``--every``-th
frame of each video (from the first) goes to ``ROOT/manifest.json`` as a capture with its
page corners (TL, TR, BR, BL, frame pixels), in the layout ``build_data.py`` reads for
``smartdoc_capture`` sources. Frames whose corners are not finite or whose quad is not
convex and clockwise are left out; a page partly outside the frame stays in.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = "flattener-smartdoc-frames-1"


def convex_clockwise(quad: np.ndarray) -> bool:
    edges = np.roll(quad, -1, axis=0) - quad
    turns = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    return bool((turns > 0).all())  # y points down: clockwise on screen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root")
    parser.add_argument("--every", type=int, default=6)
    args = parser.parse_args(argv)
    root = Path(args.root)
    metadata = root / "metadata.csv.gz"
    with gzip.open(metadata, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    captures, skipped = [], 0
    for row in rows:
        if (int(row["frame_index"]) - 1) % args.every:
            continue
        quad = np.array(
            [[float(row[f"{c}_x"]), float(row[f"{c}_y"])] for c in ("tl", "tr", "br", "bl")]
        )
        if not np.isfinite(quad).all() or not convex_clockwise(quad):
            skipped += 1
            continue
        captures.append(
            {
                "path": row["image_path"],
                "template": row["model_name"],
                "background": row["bg_name"],
                "frame": int(row["frame_index"]),
                "corners_xy": quad.round(3).tolist(),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "source": "SmartDoc 2015 Challenge 1 frames v2.0.0",
        "license_spdx": "CC-BY-4.0",
        "every": args.every,
        "skipped": skipped,
        "captures": captures,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"captures": len(captures), "skipped": skipped}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
