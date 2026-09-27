"""``python -m flattener.scan IMG... --out DIR``: scan photos of single pages.

Each photo gives ``DIR/<stem>.jpg`` (JPEG quality 95, 4:4:4, the source's ICC profile; or
``--format png``) and ``DIR/<stem>.scan.json``, the scan record with its plan, reasons,
guidance, timings and model records. A photo that falls back keeps its failure status in
the record; with the default ``--on-error original`` the saved image is the original.
One JSON line per photo goes to stdout.

Weights: ``--bundle B`` (a ``flattener-scan-bundle-1`` JSON file) or ``--weights W`` with
optional ``--gate G`` and ``--orient-weights O``. Weights that fail the ship gate load only
with ``--research``, and every record then says ``research: true``.

Exit status: 0 when every photo was scanned (rectified, with or without warnings; a
``--no-dewarp`` perspective result counts as scanned), 1 when any photo fell back or
failed, 2 on a usage error (bad options, unreadable weights).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..contracts import FlattenerError, Status
from ..dewarp.predict import parse_inset
from .io import FORMATS, image_format, read_source, save_scan
from .locate import outline_polygon
from .output_policy import PAPER_NAMES
from .pipeline import (
    PASS_CHOICES,
    Scanner,
    ScanOptions,
    ScanOverrides,
)
from .records import EXIT_FALLBACK, EXIT_SCANNED, EXIT_USAGE, SCANNED_OUTCOMES, Outcome
from .timing import append_jsonl, megapixel_bucket

DEBUG_LONG_SIDE = 1024


def _numbers(text: str, count: int, name: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} takes {count} numbers, got {text!r}") from None
    if len(values) != count or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(f"{name} takes {count} finite numbers, got {text!r}")
    return values


def _region(text: str) -> tuple[float, ...]:
    return tuple(_numbers(text, 4, "--region"))


def _quad(text: str) -> list[list[float]]:
    values = _numbers(text, 8, "--quad")
    return [values[i : i + 2] for i in range(0, 8, 2)]


def _passes(text: str) -> str | int:
    if text == "auto":
        return text
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value not in PASS_CHOICES:
        raise argparse.ArgumentTypeError(f"--passes takes auto, 0, 1 or 2, got {text!r}")
    return value


def _size(text: str) -> tuple[int, int]:
    values = _numbers(text, 2, "--output-size")
    if any(value != int(value) or value < 2 for value in values):
        raise argparse.ArgumentTypeError(f"--output-size takes H,W >= 2, got {text!r}")
    return int(values[0]), int(values[1])


def _aspect(text: str) -> str | float:
    if text.lower() in ("auto", "estimate", "snap") or text.lower() in PAPER_NAMES:
        return text.lower()
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--aspect takes auto, estimate, snap, a paper name or width/height, got {text!r}"
        ) from None
    if not (np.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"--aspect must be positive, got {text!r}")
    return value


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flattener.scan",
        description="Scan photos of single pages into flat page images.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="JPEG or PNG photos")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    weights = parser.add_argument_group("weights")
    weights.add_argument("--bundle", type=Path, help="flattener-scan-bundle-1 JSON file")
    weights.add_argument("--weights", type=Path, help="dewarp checkpoint (without --bundle)")
    weights.add_argument("--gate", type=Path, help="page gate checkpoint (with --weights)")
    weights.add_argument(
        "--orient-weights", type=Path, help="OrientNet checkpoint (with --weights)"
    )
    weights.add_argument("--device", default="cpu")
    weights.add_argument("--threads", type=int, help="torch CPU threads")
    weights.add_argument(
        "--research",
        action="store_true",
        help="load weights that fail the ship gate; records say research: true",
    )
    manual = parser.add_argument_group("manual corrections")
    area = manual.add_mutually_exclusive_group()
    area.add_argument("--region", type=_region, metavar="L,T,R,B", help="page box in pixels")
    area.add_argument(
        "--quad",
        type=_quad,
        metavar="x1,y1,...,x4,y4",
        help="page corners in reading order: top-left, top-right, bottom-right, bottom-left",
    )
    manual.add_argument(
        "--quarter",
        type=int,
        choices=range(4),
        help="clockwise quarter turns of the page's text-up direction in the photo",
    )
    processing = parser.add_argument_group("processing")
    processing.add_argument("--passes", type=_passes, default="auto", help="auto, 0, 1 or 2")
    processing.add_argument("--no-deskew", action="store_true")
    processing.add_argument(
        "--no-dewarp", action="store_true", help="flatten the page quad by a homography only"
    )
    processing.add_argument(
        "--no-orient", action="store_true", help="keep the photo's orientation (EXIF applied)"
    )
    processing.add_argument("--no-chain", action="store_true", help="try the first candidate only")
    output = parser.add_argument_group("output")
    size = output.add_mutually_exclusive_group()
    size.add_argument("--long-side", type=int, help="long side of the output in pixels")
    size.add_argument("--output-size", type=_size, metavar="H,W")
    output.add_argument("--aspect", type=_aspect, default="auto")
    output.add_argument("--inset", type=parse_inset, default=None)
    output.add_argument("--border", type=float, default=0.0)
    output.add_argument("--format", default="jpg", choices=sorted({*FORMATS, "jpg"}))
    output.add_argument(
        "--fallback",
        choices=("original", "perspective"),
        default="original",
        help="when every candidate fails: the original, or the page quad flattened when sure",
    )
    output.add_argument(
        "--on-error",
        choices=("original", "return"),
        default="original",
        help="original: save the photo itself on failure; return: save no image",
    )
    output.add_argument("--overwrite", action="store_true")
    records = parser.add_argument_group("records")
    records.add_argument("--timing", type=Path, help="append one JSON line of timings per photo")
    records.add_argument(
        "--debug", action="store_true", help="also write <stem>.debug.jpg with the page outline"
    )
    return parser


def _options(args: argparse.Namespace) -> ScanOptions:
    if args.output_size is not None:
        output: Any = args.output_size
    elif args.long_side is not None:
        output = args.long_side
    else:
        output = "auto"
    return ScanOptions(
        passes=args.passes,
        deskew=not args.no_deskew,
        orient="none" if args.no_orient else "model",
        output=output,
        aspect=args.aspect,
        inset=args.inset,
        border=args.border,
        fallback=args.fallback,
        on_error=args.on_error,
        chain=not args.no_chain,
    )


def _overrides(args: argparse.Namespace) -> ScanOverrides:
    return ScanOverrides(
        region=args.region,
        quad=args.quad,
        quarter=args.quarter,
        skip_dewarp=args.no_dewarp,
    )


def _load(args: argparse.Namespace) -> Scanner:
    if (args.bundle is None) == (args.weights is None):
        raise FlattenerError(
            Status.INPUT_ERROR, "usage", "give exactly one of --bundle or --weights"
        )
    if args.bundle is not None:
        if args.gate is not None or args.orient_weights is not None:
            raise FlattenerError(
                Status.INPUT_ERROR, "usage", "--gate and --orient-weights go with --weights"
            )
        return Scanner.load(args.bundle, device=args.device, research=args.research)
    return Scanner.from_files(
        args.weights,
        gate=args.gate,
        orient=args.orient_weights,
        device=args.device,
        research=args.research,
    )


def _stems(images: Sequence[Path]) -> list[str]:
    seen: dict[str, int] = {}
    stems = []
    for path in images:
        stem = path.stem
        count = seen.get(stem, 0)
        seen[stem] = count + 1
        stems.append(stem if count == 0 else f"{stem}-{count + 1}")
    return stems


def _debug_image(source: np.ndarray, result) -> np.ndarray:
    """The photo (long side ``DEBUG_LONG_SIDE``) with the plan's page outline in green and
    the first candidate's box in blue."""
    height, width = source.shape[:2]
    scale = min(1.0, DEBUG_LONG_SIDE / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    with Image.fromarray(source) as full:
        canvas = full.resize(size, Image.Resampling.BOX)
    draw = ImageDraw.Draw(canvas)
    box = (result.metadata.get("localization") or {}).get("box")
    if box:
        draw.rectangle([value * scale for value in box], outline=(40, 90, 255), width=3)
    plan = result.plan
    if plan is not None and plan.grid is not None:
        values = outline_polygon(plan.grid)
        points = (values + 1.0) / 2.0 * np.array([width, height]) * scale
        loop = [tuple(point) for point in points.tolist()]
        draw.line([*loop, loop[0]], fill=(30, 200, 60), width=3)
    return np.asarray(canvas)


def _scanned(result, skip_dewarp: bool) -> bool:
    if result.outcome in SCANNED_OUTCOMES:
        return True
    return skip_dewarp and result.outcome == Outcome.PERSPECTIVE and not result.used_fallback


def main(argv: Sequence[str] | None = None) -> int:
    parse = parser()
    try:
        args = parse.parse_args(argv)
    except SystemExit as exit_:
        return EXIT_USAGE if exit_.code not in (0, None) else EXIT_SCANNED
    try:
        options = _options(args)
        overrides = _overrides(args)
        fmt = image_format(args.format)
        if args.threads is not None:
            if args.threads < 1:
                raise FlattenerError(Status.INPUT_ERROR, "usage", "--threads must be at least 1")
            torch.set_num_threads(args.threads)
        scanner = _load(args)
    except FlattenerError as error:
        failure = {"status": str(error.status), "reason_code": error.reason_code}
        print(json.dumps({**failure, "message": error.message}), file=sys.stderr)
        return EXIT_USAGE
    scanned = True
    for path, stem in zip(args.images, _stems(args.images), strict=True):
        line: dict[str, Any] = {"input": str(path)}
        try:
            source = read_source(path)
            result = scanner.scan(source, options, overrides)
            result.input = str(path)
            written = save_scan(
                result,
                args.out,
                stem,
                fmt=fmt,
                icc_profile=source.icc_profile,
                overwrite=args.overwrite,
            )
            if args.debug:
                debug = Path(args.out) / f"{stem}.debug.jpg"
                if debug.exists() and not args.overwrite:
                    raise FlattenerError(Status.IO_ERROR, "output_exists", f"{debug} exists")
                Image.fromarray(_debug_image(source.image, result)).save(debug, quality=90)
                written["debug"] = str(debug)
        except FlattenerError as error:
            scanned = False
            line.update(status=str(error.status), reason_code=error.reason_code)
            line.update(outcome=Outcome.FAILED.value, message=error.message)
            print(json.dumps(line, ensure_ascii=False))
            continue
        ok = _scanned(result, args.no_dewarp)
        scanned = scanned and ok
        line.update(
            outcome=result.outcome.value,
            status=result.status.value,
            reason_code=result.reason_code,
            reasons=[item.code for item in result.reasons],
            research=result.research,
            files=written,
            seconds=result.timings.get("total"),
        )
        print(json.dumps(line, ensure_ascii=False))
        if args.timing is not None:
            full = source.info.full_size
            append_jsonl(
                args.timing,
                {
                    "input": str(path),
                    "bucket": megapixel_bucket(*full),
                    "full_size": list(full),
                    "size": list(source.info.size),
                    "timings": result.timings,
                    "outcome": result.outcome.value,
                    "device": args.device,
                    "threads": torch.get_num_threads(),
                },
            )
    return EXIT_SCANNED if scanned else EXIT_FALLBACK


if __name__ == "__main__":
    sys.exit(main())
