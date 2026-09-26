#!/usr/bin/env python3
"""Compact SyntheticDoc training copy: resized JPEG renders, fp16 maps, sha256 manifest.

    python compact.py --source <data>/release/syntheticdoc --output <cache>/syntheticdoc-release

Reads the download layout ``samples/<shard>/<sid>.render.jpg``, ``.bm16.npy`` and ``.json``.
The downloader writes the JSON last, so a sample counts as complete when all three files
exist and the JSON's recorded JPEG size matches the file. Writes under ``--output``::

    compact.json                     settings, fixed by the first run (scale, size, quality)
    samples/<shard>/<sid>.jpg        render resized by ``scale`` (bicubic, antialiased)
    samples/<shard>/<sid>.bm16.npy   backward map, copied byte for byte (fp16 177x121, raw)
    samples/<shard>/<sid>.json       sample record: source identity, license, attribution
    manifest.jsonl                   flattener-dewarp-sample-1 records with sha256 and sizes
    compact-manifest.json            counts, license summary, manifest and tool sha256

``--scale auto`` (first run only) picks the scale from the page extents of up to
``--stats-samples`` maps: the ``--percentile`` of the per-sample scale at which a
page-tight crop still covers ``--input-size``, rounded up to 1/16. The maps keep the raw
SyntheticDoc convention with the original render size, which stays valid for the resized
image because normalized coordinates span the whole image.

Runs are incremental: a sample is converted again when its source files changed (size or
modification time), its outputs are missing or the settings differ; ``--verify`` also
re-hashes kept outputs. ``manifest.jsonl`` is rebuilt from the sample records at the end
of every run and lists the shards of that run (``--shards`` selects a subset for a partial
copy; a run without it lists everything). Standard library, NumPy and Pillow; up to 8
worker processes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import io
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

COMPACT_SCHEMA = "flattener-syntheticdoc-compact-1"
MANIFEST_SCHEMA = "flattener-syntheticdoc-compact-manifest-1"
SAMPLE_SCHEMA = "flattener-dewarp-sample-1"
TOOL_VERSION = "syntheticdoc-compact-1.0"
SUFFIXES = {".render.jpg": "image", ".bm16.npy": "backward_map", ".json": "record"}
MAX_WORKERS = 8
SCALE_STEP = 16  # auto scale is a multiple of 1/16


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*parts: Any) -> None:
    print(utcnow(), *parts, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Temporary file, then ``os.replace`` (retried: Windows readers block it briefly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    temporary.write_bytes(data)
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 39:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.25)


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    write_atomic(path, text.encode("utf-8"))


def scan_shard(directory: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """``{stem: {role: (bytes, mtime_ns)}}`` from one directory listing."""
    found: dict[str, dict[str, tuple[int, int]]] = {}
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        return found
    for entry in entries:
        for suffix, role in SUFFIXES.items():
            if entry.name.endswith(suffix) and entry.is_file():
                info = entry.stat()
                stem = entry.name[: -len(suffix)]
                found.setdefault(stem, {})[role] = (info.st_size, info.st_mtime_ns)
    return found


def complete_stems(files: dict[str, dict[str, tuple[int, int]]]) -> list[str]:
    return sorted(stem for stem, roles in files.items() if len(roles) == len(SUFFIXES))


def source_shards(source: Path) -> list[str]:
    root = source / "samples"
    return sorted(entry.name for entry in os.scandir(root) if entry.is_dir())


# --------------------------------------------------------------------------- scale


def page_scale(backward_map: np.ndarray, render_size: tuple[int, int], input_size) -> float:
    """Scale at which the page's bounding box just covers ``input_size`` (both axes)."""
    values = backward_map.astype(np.float64)
    finite = np.isfinite(values).all(axis=-1)
    x, y = values[..., 0][finite], values[..., 1][finite]
    height, width = render_size
    extent_x = (x.max() - x.min()) / 2 * width
    extent_y = (y.max() - y.min()) / 2 * height
    return max(input_size[0] / max(extent_y, 1.0), input_size[1] / max(extent_x, 1.0))


def choose_scale(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Scale statistics over evenly spaced complete samples (deterministic)."""
    paths = []
    for shard in source_shards(source):
        directory = source / "samples" / shard
        paths += [directory / stem for stem in complete_stems(scan_shard(directory))]
    if not paths:
        raise SystemExit(f"no complete samples under {source / 'samples'}")
    picks = [paths[i] for i in np.linspace(0, len(paths) - 1, args.stats_samples).astype(int)]
    scales, render_sizes = [], Counter()
    for stem in dict.fromkeys(picks):
        record = json.loads(stem.with_name(stem.name + ".json").read_text(encoding="utf-8"))
        render_size = tuple(record["original_render_size"])
        render_sizes[render_size] += 1
        values = np.load(stem.with_name(stem.name + ".bm16.npy"), allow_pickle=False)
        scales.append(page_scale(values, render_size, args.input_size))
    chosen = float(np.percentile(scales, args.percentile))
    scale = min(1.0, max(0.5, math.ceil(chosen * SCALE_STEP) / SCALE_STEP))
    quantiles = (5, 25, 50, 75, 90, 95)
    return {
        "scale": scale,
        "statistics": {
            "samples": len(scales),
            "input_size": list(args.input_size),
            "percentile": args.percentile,
            "page_tight_scale_percentiles": {
                str(q): round(float(np.percentile(scales, q)), 4) for q in quantiles
            },
            "render_sizes": {f"{h}x{w}": n for (h, w), n in sorted(render_sizes.items())},
        },
    }


def load_settings(output: Path, source: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Settings of the copy: read from ``compact.json`` or fixed now by the first run."""
    path = output / "compact.json"
    if path.is_file():
        settings = json.loads(path.read_text(encoding="utf-8"))
        if settings.get("schema") != COMPACT_SCHEMA:
            raise SystemExit(f"{path}: schema must be {COMPACT_SCHEMA}")
        wanted = {"track": args.track, "jpeg_quality": args.quality}
        if args.scale != "auto":
            wanted["scale"] = float(args.scale)
        conflicts = {k: (settings.get(k), v) for k, v in wanted.items() if settings.get(k) != v}
        if conflicts:
            raise SystemExit(f"{path} fixes {conflicts} (stored, requested); use a new --output")
        return settings
    if args.scale == "auto":
        chosen = choose_scale(source, args)
    else:
        chosen = {"scale": float(args.scale), "statistics": None}
    if not 0 < chosen["scale"] <= 1:
        raise SystemExit("--scale must be in (0, 1]")
    settings = {
        "schema": COMPACT_SCHEMA,
        "track": args.track,
        "scale": chosen["scale"],
        "scale_statistics": chosen["statistics"],
        "jpeg_quality": args.quality,
        "chroma_subsampling": "4:2:0",
        "resample": "bicubic (antialiased)",
        "backward_map": "copied unchanged (raw SyntheticDoc values, original_render_size kept)",
        "created_at": utcnow(),
        "tool": TOOL_VERSION,
    }
    write_json(path, settings)
    return settings


# --------------------------------------------------------------------------- conversion


def _identity(files: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    return {role: list(value) for role, value in sorted(files.items())}


def _conversion(settings: dict[str, Any]) -> dict[str, Any]:
    return {k: settings[k] for k in ("scale", "jpeg_quality", "chroma_subsampling", "resample")}


def _entry(record: dict[str, Any], shard: str) -> dict[str, Any]:
    """Manifest line: the sample record with paths relative to the copy's root."""
    folder = f"samples/{shard}"
    keys = (
        "schema",
        "sample_id",
        "split",
        "backward_map_convention",
        "original_render_size",
        "image_size",
        "track",
        "license_class",
        "image_sha256",
        "image_bytes",
        "backward_map_sha256",
        "backward_map_bytes",
    )
    entry = {key: record[key] for key in keys}
    entry.update(
        image=f"{folder}/{record['image']}",
        backward_map=f"{folder}/{record['backward_map']}",
        meta=f"{folder}/{record['stem']}.json",
        shard=shard,
        official_split=record["official"].get("split"),
        doc_group=record["groups"].get("doc_group"),
        mesh_group=record["groups"].get("mesh_group"),
        document_license=record["license"].get("document"),
        background_license=record["license"].get("background"),
    )
    return entry


def _keep(
    record_path: Path, files: dict[str, tuple[int, int]], settings: dict[str, Any], verify: bool
) -> dict[str, Any] | None:
    """The existing record when its source and outputs are unchanged, else None."""
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("source", {}).get("files") != _identity(files):
        return None
    if record.get("conversion") != _conversion(settings):
        return None
    folder = record_path.parent
    for key in ("image", "backward_map"):
        path = folder / record[key]
        try:
            if path.stat().st_size != record[f"{key}_bytes"]:
                return None
            if verify and sha256_bytes(path.read_bytes()) != record[f"{key}_sha256"]:
                return None
        except OSError:
            return None
    return record


def convert_sample(
    source_dir: Path,
    output_dir: Path,
    stem: str,
    files: dict[str, tuple[int, int]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    original = json.loads((source_dir / f"{stem}.json").read_text(encoding="utf-8"))
    if original.get("schema") != SAMPLE_SCHEMA:
        raise ValueError(f"{stem}: schema {original.get('schema')!r}")
    if original.get("track") != settings["track"]:
        raise ValueError(f"{stem}: track {original.get('track')!r}")
    if settings["track"] == "release" and original.get("license_class") != "clean":
        raise ValueError(f"{stem}: license class {original.get('license_class')!r}")
    jpeg = (source_dir / f"{stem}.render.jpg").read_bytes()
    recorded = original.get("conversion", {}).get("image", {}).get("bytes")
    if recorded is not None and recorded != len(jpeg):
        raise ValueError(f"{stem}: JPEG has {len(jpeg)} bytes, the record says {recorded}")
    backward_map = (source_dir / f"{stem}.bm16.npy").read_bytes()
    values = np.load(io.BytesIO(backward_map), allow_pickle=False)
    if values.ndim != 3 or values.shape[2] != 2 or values.dtype.kind != "f":
        raise ValueError(f"{stem}: backward map {values.shape} {values.dtype}")
    with Image.open(io.BytesIO(jpeg)) as image:
        image = image.convert("RGB")
        width, height = image.size
        size = (max(1, round(width * settings["scale"])), max(1, round(height * settings["scale"])))
        if size != image.size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=settings["jpeg_quality"], subsampling=2)
    compact = buffer.getvalue()
    record = {
        "schema": SAMPLE_SCHEMA,
        "sample_id": original["sample_id"],
        "stem": stem,
        "split": original["split"],
        "image": f"{stem}.jpg",
        "backward_map": f"{stem}.bm16.npy",
        "backward_map_convention": original["backward_map_convention"],
        "original_render_size": original["original_render_size"],
        "image_size": [size[1], size[0]],
        "track": original["track"],
        "license_class": original["license_class"],
        "image_sha256": sha256_bytes(compact),
        "image_bytes": len(compact),
        "backward_map_sha256": sha256_bytes(backward_map),
        "backward_map_bytes": len(backward_map),
        "dataset": original.get("dataset"),
        "official": original.get("official", {}),
        "groups": original.get("groups", {}),
        "license": original.get("license", {}),
        "attribution": original.get("attribution"),
        "syntheticdoc_metadata": original.get("syntheticdoc_metadata"),
        "source": {
            "files": _identity(files),
            "image_sha256": sha256_bytes(jpeg),
            "image_size": [height, width],
            "backward_map_sha256": sha256_bytes(backward_map),
        },
        "conversion": _conversion(settings),
        "tool": TOOL_VERSION,
        "converted_at": utcnow(),
    }
    write_atomic(output_dir / record["image"], compact)
    write_atomic(output_dir / record["backward_map"], backward_map)
    write_json(output_dir / f"{stem}.json", record)  # last: marks the sample complete
    return record


def process_shard(
    source: str, output: str, shard: str, settings: dict[str, Any], verify: bool
) -> dict[str, Any]:
    """Convert or keep every complete sample of one shard; return its manifest entries."""
    source_dir, output_dir = Path(source, "samples", shard), Path(output, "samples", shard)
    files = scan_shard(source_dir)
    stems = complete_stems(files)
    result = {
        "shard": shard,
        "entries": [],
        "converted": 0,
        "kept": 0,
        "incomplete": len(files) - len(stems),
        "errors": [],
    }
    for stem in stems:
        record = _keep(output_dir / f"{stem}.json", files[stem], settings, verify)
        try:
            if record is None:
                record = convert_sample(source_dir, output_dir, stem, files[stem], settings)
                result["converted"] += 1
            else:
                result["kept"] += 1
            result["entries"].append(_entry(record, shard))
        except Exception as error:  # noqa: BLE001 - reported per sample; the run exits 1
            result["errors"].append(f"{shard}/{stem}: {type(error).__name__}: {error}")
    return result


# --------------------------------------------------------------------------- manifests


def write_manifests(
    output: Path, source: Path, settings: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda result: result["shard"])
    entries = [entry for result in ordered for entry in result["entries"]]
    lines = [json.dumps(e, separators=(",", ":"), ensure_ascii=False) + "\n" for e in entries]
    data = "".join(lines).encode("utf-8")
    write_atomic(output / "manifest.jsonl", data)
    errors = [error for result in results for error in result["errors"]]
    counts = {
        "samples": len(entries),
        "converted_this_run": sum(r["converted"] for r in results),
        "kept": sum(r["kept"] for r in results),
        "incomplete_skipped": sum(r["incomplete"] for r in results),
        "errors": len(errors),
        "by_split": dict(sorted(Counter(e["split"] for e in entries).items())),
    }
    download = source / "dataset-manifest.json"
    download_complete = None
    if download.is_file():
        download_complete = json.loads(download.read_text(encoding="utf-8")).get("complete")
    summary = {
        "schema": MANIFEST_SCHEMA,
        "track": settings["track"],
        "generated_at": utcnow(),
        "tool": {
            "version": TOOL_VERSION,
            "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "settings": settings,
        "source": str(source),
        "counts": counts,
        "source_download_complete": download_complete,
        "complete": download_complete is True and not errors and not counts["incomplete_skipped"],
        "license_classes": dict(Counter(e["license_class"] for e in entries)),
        "document_licenses": dict(Counter(e["document_license"] for e in entries)),
        "background_licenses": dict(Counter(e["background_license"] for e in entries)),
        "bytes": {
            "image": sum(e["image_bytes"] for e in entries),
            "backward_map": sum(e["backward_map_bytes"] for e in entries),
        },
        "manifest_sha256": sha256_bytes(data),
        "errors": errors[:100],
    }
    write_json(output / "compact-manifest.json", summary)
    return summary


def run(args: argparse.Namespace) -> int:
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if not (source / "samples").is_dir():
        raise SystemExit(f"{source / 'samples'} is missing")
    output.mkdir(parents=True, exist_ok=True)
    settings = load_settings(output, source, args)
    log(f"scale {settings['scale']}, JPEG q{settings['jpeg_quality']}, track {settings['track']}")
    if args.stats_only:
        print(json.dumps(settings, indent=2))
        return 0
    shards = [
        shard
        for shard in source_shards(source)
        if not args.shards or any(fnmatch.fnmatchcase(shard, p) for p in args.shards)
    ]
    started, results = time.monotonic(), []

    def collect(result: dict[str, Any]) -> None:
        results.append(result)
        if len(results) % 50 == 0 or len(results) == len(shards):
            done = sum(r["converted"] for r in results)
            rate = done / max(time.monotonic() - started, 1e-9)
            log(f"{len(results)}/{len(shards)} shards, {done} converted ({rate:.1f}/s)")

    task = (str(source), str(output))
    if args.workers == 1:
        for shard in shards:
            collect(process_shard(*task, shard, settings, args.verify))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [
                pool.submit(process_shard, *task, shard, settings, args.verify) for shard in shards
            ]
            for future in as_completed(futures):
                collect(future.result())
    summary = write_manifests(output, source, settings, results)
    log("counts", json.dumps(summary["counts"]))
    for error in summary["errors"][:20]:
        log("error", error)
    return 1 if summary["counts"]["errors"] else 0


def _workers(text: str) -> int:
    value = int(text)
    if not 1 <= value <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"--workers must be in 1..{MAX_WORKERS}")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--source", required=True, help="download root holding samples/")
    result.add_argument("--output", required=True, help="training copy root")
    result.add_argument("--track", default="release", choices=("release", "research"))
    result.add_argument("--scale", default="auto", help="'auto' or a factor in (0, 1]")
    result.add_argument("--input-size", type=int, nargs=2, default=(712, 488), metavar=("H", "W"))
    result.add_argument("--percentile", type=float, default=75.0)
    result.add_argument("--stats-samples", type=int, default=512)
    result.add_argument("--quality", type=int, default=92)
    result.add_argument("--workers", type=_workers, default=MAX_WORKERS)
    result.add_argument("--verify", action="store_true", help="re-hash kept outputs")
    result.add_argument("--shards", nargs="+", help="shard name patterns for a partial copy")
    result.add_argument("--stats-only", action="store_true", help="fix settings and exit")
    return result


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
