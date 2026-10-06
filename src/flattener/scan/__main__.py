"""``flattener-scan -i PHOTO [OUTPUT]``: scan photos of pages into flat page images.

Like ffmpeg, inputs follow ``-i`` and the output comes last. ``OUTPUT`` is

* an image file (``.jpg`` or ``.png``) for one input;
* a directory (an existing one, or a path ending in ``/``), which gets ``<stem>.jpg``
  per input; with several inputs the output must be a directory;
* left out: ``<stem>-scan.jpg`` per input in the current directory.

``--format`` sets the format of directory and default outputs (JPEG quality 95, 4:4:4,
the source's ICC profile; or PNG). An open-book spread is split at its fold into
``<name>-p1`` and ``<name>-p2`` in reading order (``--pages auto``, the default;
``--pages 1`` keeps one image, ``--page-order rtl`` numbers the right page first). A
photo that falls back saves the original with the default ``--on-error original``.
Existing files are kept unless ``-y`` is given, and an output that is one of the input
photos is a usage error, so a scan never replaces a photo it reads.

One line per saved page goes to stderr (``-q`` turns it off). ``--json`` prints one JSON
line per page to stdout instead, and ``--record`` writes the scan record (plan, reasons,
guidance, timings, models) next to each image as ``<name>.json``.

The models run on ONNX Runtime on the CPU. With PyTorch installed (for example
``uvx --with torch flattener-scan ...``) and a CUDA GPU present, they run on the GPU
(``--backend``, ``--device``). The default models are downloaded from Hugging Face on
first use and cached. Use ``--bundle B`` for a local bundle (ONNX or PyTorch), or
``--weights W`` with optional ``--gate G`` and ``--orient-weights O`` for PyTorch
checkpoints.

``--long-side``, ``--output-size`` and ``--aspect`` are manual corrections
(``ScanOverrides``), recorded as ``manual`` provenance.

``--timing F.jsonl`` appends one row per photo: the stage seconds of the scan (decoding
included), ``end_to_end`` (the scan plus saving), the stage cache hits and misses
(``cached`` is true when a model output came from the cache; ``--no-cache`` turns the
cache off for repeated timing runs) and, on the first row, the model load. The file is
opened before the first photo; a row that cannot be written is reported on stderr.

``--debug`` writes ``<name>.debug.jpg`` with the page outline after the scan; if that
fails, the scan keeps its outcome.

Exit status: 0 when every photo was scanned (rectified, with or without warnings; a
``--no-dewarp`` perspective result counts as scanned), 1 when any photo fell back or
failed, 2 on a usage error (bad options, an unavailable device, unreadable weights, an
output that would replace an input). A photo that raises an unexpected error is reported
and the batch goes on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .. import ops, runtime
from ..contracts import FlattenerError, Status
from ..dewarp.predict import parse_inset
from .io import FORMATS, image_format, save_image, save_scan_files
from .locate import outline_polygon
from .output_policy import PAPER_NAMES
from .pipeline import (
    CACHE_SIZE,
    PASS_CHOICES,
    Scanner,
    ScanOptions,
    ScanOverrides,
)
from .records import EXIT_FALLBACK, EXIT_SCANNED, EXIT_USAGE, SCANNED_OUTCOMES, Outcome
from .sharpen import DEFAULT_AMOUNT
from .timing import append_jsonl, megapixel_bucket

PROG = "flattener-scan"
DEFAULT_SUFFIX = "-scan"
DEBUG_LONG_SIDE = 1024
DEBUG_QUALITY = 90


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


def _pages(text: str) -> str | int:
    if text == "auto":
        return text
    if text in ("1", "2"):
        return int(text)
    raise argparse.ArgumentTypeError(f"--pages takes auto, 1 or 2, got {text!r}")


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
        prog=PROG,
        usage=f"{PROG} -i PHOTO [-i PHOTO ...] [OUTPUT] [options]",
        description="Scan photos of pages into flat page images.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=Path,
        metavar="PHOTO",
        help="JPEG or PNG photo; repeat for several",
    )
    parser.add_argument(
        "output",
        nargs="?",
        metavar="OUTPUT",
        help="image file (.jpg, .png) for one photo, or a directory; "
        f"default: <stem>{DEFAULT_SUFFIX}.jpg in the current directory",
    )
    parser.add_argument("-y", "--overwrite", action="store_true", help="replace existing files")
    parser.add_argument("-q", "--quiet", action="store_true", help="print nothing on stderr")
    parser.add_argument("--json", action="store_true", help="print one JSON line per page")
    parser.add_argument(
        "--record", action="store_true", help="also write the scan record as <name>.json"
    )
    run = parser.add_argument_group("models")
    run.add_argument(
        "--backend",
        choices=runtime.BACKENDS,
        default="auto",
        help="auto: PyTorch on a CUDA GPU when installed, else ONNX Runtime on the CPU",
    )
    run.add_argument("--device", default="auto", help="auto, cpu or cuda[:N]")
    run.add_argument("--threads", type=int, help="CPU threads")
    run.add_argument(
        "--bundle", type=Path, help="local bundle JSON; default: download models from Hugging Face"
    )
    run.add_argument("--weights", type=Path, help="dewarp checkpoint (without --bundle)")
    run.add_argument("--gate", type=Path, help="page gate checkpoint (with --weights)")
    run.add_argument("--orient-weights", type=Path, help="OrientNet checkpoint (with --weights)")
    run.add_argument(
        "--orient-threshold",
        type=float,
        help="OrientNet threshold measured on real photos; replaces the bundle's and 0.9",
    )
    manual = parser.add_argument_group(
        "manual corrections",
        "Coordinates use EXIF-oriented decoded image pixels. For a JPEG decoded at reduced "
        "resolution, divide full-resolution coordinates by source.scale in its scan record.",
    )
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
    processing.add_argument(
        "--pages",
        type=_pages,
        default="auto",
        help="auto: split an open-book spread at its fold; 1: one image; 2: always split",
    )
    processing.add_argument(
        "--page-order",
        choices=("ltr", "rtl"),
        default="ltr",
        help="number the left page (ltr) or the right page (rtl) first",
    )
    processing.add_argument(
        "--full-check",
        action="store_true",
        help="run the renderer's checks at full size (default: from the grid and a small proxy)",
    )
    output = parser.add_argument_group("output")
    size = output.add_mutually_exclusive_group()
    size.add_argument("--long-side", type=int, help="long side of the output in pixels")
    size.add_argument("--output-size", type=_size, metavar="H,W")
    output.add_argument("--aspect", type=_aspect, default="auto")
    output.add_argument("--inset", type=parse_inset, default=None)
    output.add_argument("--border", type=float, default=0.0)
    output.add_argument(
        "--sharpen",
        type=float,
        default=DEFAULT_AMOUNT,
        help=f"sharpening amount on the flattened page, 0 for none (default {DEFAULT_AMOUNT:g})",
    )
    output.add_argument(
        "--format",
        choices=sorted({*FORMATS, "jpg"}),
        help="format of directory and default outputs (default jpg)",
    )
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
    records = parser.add_argument_group("records")
    records.add_argument("--timing", type=Path, help="append one JSON line of timings per photo")
    records.add_argument(
        "--no-cache",
        action="store_true",
        help="keep no model outputs between photos (repeated timing runs of one photo)",
    )
    records.add_argument(
        "--debug", action="store_true", help="also write <name>.debug.jpg with the page outline"
    )
    return parser


def _options(args: argparse.Namespace) -> ScanOptions:
    return ScanOptions(
        passes=args.passes,
        deskew=not args.no_deskew,
        orient="none" if args.no_orient else "model",
        inset=args.inset,
        border=args.border,
        sharpen=args.sharpen,
        fallback=args.fallback,
        on_error=args.on_error,
        fast_check=not args.full_check,
        chain=not args.no_chain,
        pages=args.pages,
        page_order=args.page_order,
    )


def _overrides(args: argparse.Namespace) -> ScanOverrides:
    """The manual corrections, output size and aspect included."""
    if args.output_size is not None:
        output: Any = args.output_size
    elif args.long_side is not None:
        output = args.long_side
    else:
        output = None
    return ScanOverrides(
        region=args.region,
        quad=args.quad,
        quarter=args.quarter,
        skip_dewarp=args.no_dewarp,
        output_size=output,
        aspect=args.aspect,
    )


def _usage(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "usage", message)


def _targets(args: argparse.Namespace) -> list[Path]:
    """The image path of each input (before page numbers)."""
    stems = _stems(args.inputs)
    suffix = FORMATS[image_format(args.format or "jpg")]
    if args.output is None:
        return [Path(f"{stem}{DEFAULT_SUFFIX}{suffix}") for stem in stems]
    output = Path(args.output)
    directory = output.is_dir() or args.output.endswith(("/", os.sep))
    if len(args.inputs) > 1 or directory:
        if output.exists() and not output.is_dir():
            raise _usage(f"{output} is a file; several photos need an output directory")
        return [output / f"{stem}{suffix}" for stem in stems]
    try:
        fmt = image_format(output.suffix)
    except FlattenerError:
        raise _usage(f"{output}: the output needs a .jpg, .jpeg or .png suffix") from None
    if args.format is not None and image_format(args.format) != fmt:
        raise _usage(f"--format {args.format} disagrees with {output}")
    return [output]


def _page_path(target: Path, number: int | None) -> Path:
    if number is None:
        return target
    return target.with_name(f"{target.stem}-p{number}{target.suffix}")


def _record_path(target: Path) -> Path:
    return target.with_suffix(".json")


def _debug_path(target: Path) -> Path:
    return target.with_name(f"{target.stem}.debug.jpg")


def _names(args: argparse.Namespace, target: Path) -> list[Path]:
    """Every image path a photo may use: its own and, unless ``--pages 1``, its pages'."""
    numbers = [None] if args.pages == 1 else [None, 1, 2]
    return [_page_path(target, number) for number in numbers]


def _check_inputs_kept(args: argparse.Namespace, targets: list[Path]) -> None:
    """Refuse an output name that is an input photo: the scan would replace the photo,
    or a later photo would be read after an earlier scan replaced it."""
    inputs = []
    for path in args.inputs:
        try:
            info = path.stat()
        except OSError:
            continue  # a photo that cannot be read fails on its own
        key = (info.st_dev, info.st_ino) if info.st_ino else None
        inputs.append((os.path.normcase(os.path.realpath(path)), key))
    for name in (name for target in targets for name in _names(args, target)):
        paths = [name]
        if args.record:
            paths.append(_record_path(name))
        if args.debug:
            paths.append(_debug_path(name))
        for path in paths:
            if _same_file(path, inputs):
                raise _usage(f"{path} is an input photo; choose another output")


def _load(args: argparse.Namespace) -> Scanner:
    if args.bundle is not None and args.weights is not None:
        raise _usage("use either --bundle or --weights")
    if args.weights is None and (args.gate is not None or args.orient_weights is not None):
        raise _usage("--gate and --orient-weights go with --weights")
    cache_size = 0 if args.no_cache else CACHE_SIZE
    if args.weights is None:
        return Scanner.load(
            args.bundle,
            device=args.device,
            backend=args.backend,
            cache_size=cache_size,
            orient_threshold=args.orient_threshold,
        )
    if args.backend == "onnx":
        raise _usage("--weights are PyTorch checkpoints; use --bundle for ONNX models")
    return Scanner.from_files(
        args.weights,
        gate=args.gate,
        orient=args.orient_weights,
        orient_threshold=args.orient_threshold,
        device=args.device,
        cache_size=cache_size,
    )


def _say(args: argparse.Namespace, text: str) -> None:
    if not args.quiet:
        print(text, file=sys.stderr, flush=True)


def _summary(result) -> str:
    outcome = result.outcome.value.replace("_", " ")
    codes = [item.code for item in result.reasons if item.code != "source_downscaled"]
    if result.outcome != Outcome.RECTIFIED and codes:
        outcome += ": " + ", ".join(codes)
    seconds = result.timings.get("total")
    return outcome if seconds is None else f"{outcome} ({seconds:.1f} s)"


def _same_file(path: Path, others: list[tuple[str, tuple[int, int] | None]]) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    real = os.path.normcase(os.path.realpath(path))
    key = (info.st_dev, info.st_ino) if info.st_ino else None
    return any(real == other or (key is not None and key == ino) for other, ino in others)


def _stems(images: Sequence[Path]) -> list[str]:
    seen: dict[str, int] = {}
    reserved = {path.stem for path in images}
    stems = []
    for path in images:
        stem = path.stem
        count = seen.get(stem, 0)
        name = stem if count == 0 else f"{stem}-{count + 1}"
        while count and name in reserved:
            count += 1
            name = f"{stem}-{count + 1}"
        seen[stem] = count + 1
        reserved.add(name)
        stems.append(name)
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
        targets = _targets(args)
        runtime.check_device(args.device)
        if args.threads is not None:
            if args.threads < 1:
                raise _usage("--threads must be at least 1")
            runtime.set_threads(args.threads)
        if args.timing is not None:
            try:
                args.timing.open("a", encoding="utf-8").close()
            except OSError as error:
                raise _usage(f"--timing: {error}") from None
        _check_inputs_kept(args, targets)
        try:
            scanner = _load(args)
        except FlattenerError:
            raise
        except Exception as error:  # a corrupt or incompatible checkpoint
            raise FlattenerError(
                Status.MODEL_ERROR,
                "load_failed",
                f"could not load the models: {type(error).__name__}: {error}",
            ) from error
        if args.threads is not None:
            runtime.set_threads(args.threads)  # PyTorch is loaded now
    except FlattenerError as error:
        print(f"{PROG}: error: {error.message}", file=sys.stderr)
        return EXIT_USAGE
    scanned = True
    load_written = False
    for path, target in zip(args.inputs, targets, strict=True):
        saved: list[tuple[Path, Any, dict[str, str]]] = []
        try:
            if args.debug and not args.overwrite:
                for name in _names(args, target):
                    debug = _debug_path(name)
                    if debug.exists() or debug.is_symlink():
                        raise FlattenerError(Status.IO_ERROR, "output_exists", f"{debug} exists")
            planned = scanner.plan(path, options, overrides)  # decoding is the read stage
            source = planned.source
            pages = scanner.split(planned)
            for number, page in enumerate(pages, start=1):
                name = _page_path(target, None if len(pages) == 1 else number)
                result = scanner.render(source, page)
                result.input = str(path)
                written = save_scan_files(
                    result,
                    name,
                    _record_path(name) if args.record else None,
                    icc_profile=source.icc_profile,
                    overwrite=args.overwrite,
                )
                saved.append((name, result, written))
        except Exception as error:  # one photo's failure never stops the batch
            scanned = False
            if isinstance(error, FlattenerError):
                status, code, message = error.status, error.reason_code, error.message
            else:
                io = isinstance(error, OSError)
                status = Status.IO_ERROR if io else Status.MODEL_ERROR
                code = "unexpected_io_error" if io else "unexpected_error"
                message = f"{type(error).__name__}: {error}"
            hint = " (use -y to replace it)" if code == "output_exists" else ""
            _say(args, f"{path}: failed: {message}{hint}")
            if args.json:
                line = {"input": str(path), "status": str(status), "reason_code": code}
                line.update(outcome=Outcome.FAILED.value, message=message)
                if saved:
                    line["files"] = [written for _, _, written in saved]
                print(json.dumps(line, ensure_ascii=False), flush=True)
            continue
        for number, (name, result, written) in enumerate(saved, start=1):
            line: dict[str, Any] = {"input": str(path)}
            if len(saved) > 1:
                line["page"] = number
            if args.debug:
                try:
                    picture = _debug_image(source.image, result)
                    written["debug"] = save_image(
                        _debug_path(name),
                        picture,
                        quality=DEBUG_QUALITY,
                        overwrite=args.overwrite,
                    )
                except Exception as error:  # the scan itself is saved
                    line["debug_error"] = f"{type(error).__name__}: {error}"
            scanned = scanned and _scanned(result, args.no_dewarp)
            shown = written.get("image", "no image")
            _say(args, f"{path} -> {shown}  {_summary(result)}")
            if args.json:
                line.update(
                    outcome=result.outcome.value,
                    status=result.status.value,
                    reason_code=result.reason_code,
                    reasons=[item.code for item in result.reasons],
                    files=written,
                    seconds=result.timings.get("total"),
                )
                print(json.dumps(line, ensure_ascii=False), flush=True)
        if args.timing is not None:
            result = saved[0][1]
            full = source.info.full_size
            cache = result.metadata.get("cache") or {"hits": {}, "misses": {}}
            timings = result.timings
            save = sum(page.timings.get("save", 0.0) for _, page, _ in saved)
            row = {
                "input": str(path),
                "bucket": megapixel_bucket(*full),
                "full_size": list(full),
                "size": list(source.info.size),
                "pages": len(saved),
                "timings": timings,
                "end_to_end": timings.get("total", 0.0) + save,
                "cache": cache,
                "cached": any(cache["hits"].values()),
                "outcome": result.outcome.value,
                "backend": scanner.backend,
                "device": str(scanner.dewarper.device),
                "threads": ops.threads(),
                "fast_check": options.fast_check,
            }
            if not load_written:
                row["model_load"] = scanner.info.get("load")
                load_written = True
            try:
                append_jsonl(args.timing, row)
            except OSError as error:
                _say(args, f"{path}: could not write the timing row: {error}")
    return EXIT_SCANNED if scanned else EXIT_FALLBACK


if __name__ == "__main__":
    sys.exit(main())
