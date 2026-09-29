"""A development benchmark for dewarp settings: SmartDoc captures against their templates.

``python tools/smartdoc_dev.py WEIGHTS [--rerun N --rect|--perspective] [--inset P]
[--snap-border] [--every K] [--siftflow]``

Captures of SmartDoc 2015 challenge 1 documents 1-4 (``outputs/benchmarks/smartdoc``;
document 5 is the gate's held-out test) are cropped to their page quad's box plus 15%
per side, like a phone photo that frames the page with some table around it, rectified
by ``flattener.dewarp.predict.rectify`` with the given settings at half the template size,
and scored against the template as the benchmarks score (``benchmark.metrics``: MS-SSIM
at 598,400 pixels; LD and AD with ``--siftflow``). Framing choices (insets, border
snapping, crop passes) are compared here instead of on the test benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights")
    parser.add_argument("--rerun", type=int, default=0)
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--perspective", action="store_true")
    parser.add_argument("--snap-border", action="store_true")
    parser.add_argument("--inset", default=None)
    parser.add_argument("--every", type=int, default=3, help="every K-th capture")
    parser.add_argument("--siftflow", action="store_true", help="also LD and AD")
    parser.add_argument("--output", help="per-capture rows as JSON")
    args = parser.parse_args(argv)

    import torch
    from PIL import Image

    from flattener.benchmark import metrics
    from flattener.contracts import RenderConfig
    from flattener.dewarp.predict import load_dewarper, parse_inset, rectify

    torch.set_num_threads(2)
    root = ROOT / "outputs/benchmarks/smartdoc"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    templates = {t["name"]: root / t["path"] for t in manifest["templates"]}
    captures = [
        c for c in manifest["captures"]
        if int(re.search(r"(\d+)$", c["template"]).group(1)) <= 4
    ][:: args.every]  # fmt: skip
    siftflow = None
    if args.siftflow:
        from flattener.benchmark.siftflow import load_siftflow

        siftflow = load_siftflow(os.environ.get("FLATTENER_SIFTFLOW_SOURCE"))
    dewarper = load_dewarper(args.weights)
    inset = parse_inset(args.inset) if args.inset else None
    prepared: dict[str, metrics.PreparedReference] = {}
    rows = []
    for capture in captures:
        with Image.open(root / capture["path"]) as picture:
            frame = np.asarray(picture.convert("RGB"))
        quad = np.array(capture["corners_xy"])
        low, high = quad.min(axis=0), quad.max(axis=0)
        pad = 0.15 * (high - low)
        x0, y0 = np.maximum(0, np.floor(low - pad)).astype(int)
        x1 = int(min(frame.shape[1], np.ceil(high[0] + pad[0])))
        y1 = int(min(frame.shape[0], np.ceil(high[1] + pad[1])))
        crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
        name = capture["template"]
        if name not in prepared:
            with Image.open(templates[name]) as picture:
                prepared[name] = metrics.prepare_reference(np.asarray(picture.convert("RGB")))
        height, width = prepared[name].source_shape
        result = rectify(
            dewarper, crop, (height // 2, width // 2),
            render_config=RenderConfig(on_error="return"), allow_unchecked=True,
            quantize="truncate", inset=inset, rerun=args.rerun, rect=args.rect,
            perspective=args.perspective, snap=args.snap_border,
        )  # fmt: skip
        if result.image is None:
            rows.append({"capture": capture["path"], "failed": str(result.reason_code)})
            continue
        score = metrics.score_prepared(result.image, prepared[name], siftflow=siftflow)
        rows.append({"capture": capture["path"], "ms_ssim": score.ms_ssim, "ld": score.ld,
                     "ad": score.ad})  # fmt: skip
    scored = [row for row in rows if "ms_ssim" in row]
    summary = {"captures": len(rows), "failed": len(rows) - len(scored)}
    for key in ("ms_ssim", "ld", "ad"):
        values = [row[key] for row in scored if row.get(key) is not None]
        if values:
            summary[key] = round(float(np.mean(values)), 4)
    print(json.dumps(summary))
    if args.output:
        Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
