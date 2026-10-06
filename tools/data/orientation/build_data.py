"""Build upright grayscale pages for orientation training.

Sources may contain rendered documents, generated pages or text-free scenes.
The output contains page images, a sample manifest and the build settings.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import random
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pages  # noqa: E402

from flattener import FlattenerError, __version__  # noqa: E402
from flattener.dewarp.checkpoint import environment, utc_now, write_json  # noqa: E402
from flattener.dewarp.coords import map_to_grid, syntheticdoc_to_flattener  # noqa: E402
from flattener.dewarp.data import (  # noqa: E402
    SOURCE_TYPES,
    SPEC_SCHEMA,
    DataSpec,
    expand_path,
    release_license_ok,
    require_track,
)
from flattener.scan.orient_train import DATA_KINDS, DATA_SCHEMA  # noqa: E402
from flattener.synthetic import documents  # noqa: E402

BUILD_SCHEMA = "flattener-orient-build-1"
SYNTHETICDOC_KINDS = ("syntheticdoc_flat", "syntheticdoc_crop")
DEWARP_KINDS = ("converted", "syntheticdoc_compact")
OWN_LICENSE = "Apache-2.0 (text, layout, code); OFL-1.1 (Noto Sans KR glyphs)"
SOURCE_KEYS = {
    "name",
    "kind",
    "track",
    "license",
    "count",
    "dewarp_spec",
    "part",
    "grid_size",
    "stored_margin",
    "root",
    "document_prob",
    "form_prob",
    "sparse_prob",
    "print_prob",
}


def _error(message: str) -> SystemExit:
    return SystemExit(f"error: {message}")


def load_build_spec(path: str | Path) -> dict[str, Any]:
    """The build spec with paths resolved; ``spec["build_spec"]`` is what identifies it."""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != BUILD_SCHEMA:
        raise _error(f"{path}: schema must be {BUILD_SCHEMA}")
    track = spec.get("track")
    if track not in ("release", "research") or not spec.get("name"):
        raise _error("spec needs a name and track release|research")
    names = set()
    for source in spec.get("sources", []):
        unknown = set(source) - SOURCE_KEYS
        if unknown or source.get("kind") not in DATA_KINDS or source.get("name") in names:
            raise _error(f"bad source {source.get('name')!r}: keys {sorted(unknown)} or kind/name")
        names.add(source["name"])
        if source.get("track") not in ("release", "research") or not source.get("license"):
            raise _error(f"source {source['name']!r} needs a track and a license")
        if track == "release" and source["track"] != "release":
            raise _error(f"release build lists {source['track']} source {source['name']!r}")
        if source["kind"] in SYNTHETICDOC_KINDS and not source.get("dewarp_spec"):
            raise _error(f"source {source['name']!r} needs dewarp_spec")
        if source["kind"] == "pages" and not source.get("root"):
            raise _error(f"source {source['name']!r} needs root")
    if not names:
        raise _error("spec lists no sources")
    spec.setdefault("long_side", 1024)
    spec.setdefault("jpeg_quality", 90)
    spec.setdefault("chunk", 500)
    spec["base_dir"] = str(path.resolve().parent)
    return spec


def _seed(*parts: Any) -> int:
    text = "\0".join(str(part) for part in parts)
    return random.Random(int.from_bytes(text.encode("utf-8"), "little")).getrandbits(64)


def _evenly(items: list, count: int | None) -> list:
    if count is None or len(items) <= count:
        return items
    keep = np.unique(np.linspace(0, len(items) - 1, count).round().astype(int))
    return [items[i] for i in keep]


@lru_cache(maxsize=4)
def index_dewarp_spec(path: str, grid_size: tuple[int, int]):
    """The dewarp spec, its sample records per split and its source descriptions.

    The dewarp source index checks every sample's track and license class for release
    sources. Flat pages and crops usually share one spec, so it is indexed once.
    """
    dewarp = DataSpec.load(path)
    by_split: dict[str, list] = {}
    described = []
    for dewarp_source in dewarp.sources:
        if dewarp_source.kind not in DEWARP_KINDS:
            raise _error(f"{dewarp_source.name}: kind {dewarp_source.kind} is unsupported")
        loaded = SOURCE_TYPES[dewarp_source.kind](
            dewarp_source, dewarp.root(dewarp_source), grid_size
        )
        described.append(loaded.describe())
        for record in loaded.records:
            by_split.setdefault(dewarp_source.split, []).append(
                {
                    "sample_id": record["sample_id"],
                    "image": record["image"],
                    "backward_map": record["backward_map"],
                    "render_size": record["render_size"],
                    "track": record["track"],
                    "license_class": record["license_class"] or dewarp_source.license,
                    "grid_size": list(grid_size),
                }
            )
    return dewarp, by_split, described


def plan_source(spec: dict[str, Any], source: dict[str, Any], limit: int | None):
    """Per split, the items to build, plus the source's summary fields."""
    counts = source.get("count") or {}
    info: dict[str, Any] = {
        key: source[key] for key in ("name", "kind", "track", "license") if key in source
    }
    plan: dict[str, list] = {}

    def cap(split: str) -> int | None:
        values = [v for v in (counts.get(split), limit) if v is not None]
        return min(values) if values else None

    if source["kind"] in SYNTHETICDOC_KINDS:
        path = Path(spec["base_dir"], expand_path(source["dewarp_spec"])).resolve()
        grid_size = tuple(source.get("grid_size", (89, 61)))
        dewarp, by_split, described = index_dewarp_spec(str(path), grid_size)
        if source["track"] == "release":
            try:
                require_track(dewarp, "release")
            except FlattenerError as error:
                raise _error(error.message) from error
        part, parts = source.get("part", (0, 1))
        for split, records in by_split.items():
            chosen = [r for index, r in enumerate(records) if index % parts == part]
            plan[split] = _evenly(chosen, cap(split))
        info.update(dewarp_spec=dewarp.to_dict(), dewarp_sources=described)
    elif source["kind"] == "pages":
        root = Path(spec["base_dir"], expand_path(source["root"]))
        rows = [
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            if source["track"] == "release" and (
                row.get("track") != "release"
                or not release_license_ok(str(row.get("license_class", "")))
            ):
                raise _error(f"release source {source['name']!r} lists {row.get('path')!r}")
            plan.setdefault(row["split"], []).append({**row, "path": str(root / row["path"])})
        for split in plan:
            plan[split] = _evenly(plan[split], cap(split))
    else:
        for split, count in counts.items():
            plan[split] = list(range(cap(split) or count))
    return plan, info


def make_page(kind: str, item: Any, rng: np.random.Generator, settings: dict[str, Any]):
    """(gray page, box, row fields) for one item; raises ``pages.Skip``."""
    long_side = settings["long_side"]
    if kind in SYNTHETICDOC_KINDS:
        with Image.open(item["image"]) as image:
            rgb = np.array(image.convert("RGB"))
        values = np.load(item["backward_map"], allow_pickle=False)
        if item["render_size"] is not None:
            values = syntheticdoc_to_flattener(values, item["render_size"])
        grid = map_to_grid(values, tuple(item["grid_size"]))
        fields = {
            "document_group": item["sample_id"],
            "license_class": item["license_class"],
            "track": item["track"],
            "origin": {"sample_id": item["sample_id"]},
        }
        if kind == "syntheticdoc_flat":
            gray, box = pages.flatten_syntheticdoc(rgb, grid, long_side)
        else:
            margin = settings.get("stored_margin", 0.15)
            gray, box, angle = pages.crop_syntheticdoc(rgb, grid, long_side, margin)
            fields["origin"]["up_angle_deg"] = round(angle, 3)
        return gray, box, fields
    if kind == "pages":
        with Image.open(item["path"]) as image:
            upright = ImageOps.exif_transpose(image)
            gray = pages.resized_page(np.array(upright.convert("L")), long_side)
        fields = {
            "document_group": item.get("document_group", item["path"]),
            "license_class": item["license_class"],
            "track": item["track"],
            "uniform": bool(item.get("uniform", False)),
            "origin": {"path": Path(item["path"]).name},
        }
        return gray, list(pages.FULL_BOX), fields
    own = {"license_class": "own", "track": settings["track"]}
    if kind == "documents":
        page_id = f"{settings['source']}-{settings['split']}-{item}"
        gray, box, layout = pages.document_page(
            page_id,
            rng,
            long_side,
            settings.get("document_prob", 0.35),
            settings.get("form_prob", 0.0),
            settings.get("sparse_prob", 0.0),
            settings.get("print_prob", 0.0),
        )
        return gray, box, {**own, "document_group": page_id, "origin": {"layout": layout}}
    if kind == "spread":
        gray, box, layout = pages.spread_page(rng, long_side)
        group = f"{settings['source']}-{settings['split']}-{item}"
        return gray, box, {**own, "document_group": group, "origin": {"layout": layout}}
    gray, box, layout = pages.text_free_page(rng, long_side)
    group = f"{settings['source']}-{settings['split']}-{item}"
    return (
        gray,
        box,
        {**own, "document_group": group, "uniform": True, "origin": {"layout": layout}},
    )


def run_chunk(task: dict[str, Any]) -> dict[str, Any]:
    """Build one chunk; returns its rows and skip counts (the caller writes the part)."""
    import torch

    torch.set_num_threads(1)
    settings, output = task["settings"], Path(task["output"])
    rows, skipped = [], Counter()
    folder = output / "files" / settings["source"] / settings["split"]
    folder.mkdir(parents=True, exist_ok=True)
    for index, item in zip(task["indices"], task["items"], strict=True):
        rng = np.random.default_rng(
            _seed(settings["spec_name"], settings["source"], settings["split"], index)
        )
        try:
            gray, box, fields = make_page(settings["kind"], item, rng, settings)
        except pages.Skip as skip:
            skipped[skip.reason] += 1
            continue
        identifier = f"{settings['source']}-{settings['split']}-{index:07d}"
        buffer = io.BytesIO()
        Image.fromarray(gray).save(buffer, "JPEG", quality=settings["jpeg_quality"])
        data = buffer.getvalue()
        relative = f"files/{settings['source']}/{settings['split']}/{identifier}.jpg"
        (output / relative).write_bytes(data)
        rows.append(
            {
                "id": identifier,
                "split": settings["split"],
                "source": settings["source"],
                "kind": settings["kind"],
                "path": relative,
                "bytes": len(data),
                "size": list(gray.shape),
                "box": [round(float(v), 6) for v in box],
                "uniform": bool(fields.pop("uniform", False)),
                **fields,
            }
        )
    return {"key": task["key"], "rows": rows, "skipped": dict(skipped)}


def provenance_spec(name: str, track: str, infos: list[dict[str, Any]]) -> dict[str, Any]:
    """A dewarp-schema data spec (``flattener-dewarp-data-1``) naming every source that fed
    the build, so checkpoints include the data specification for release validation.

    SyntheticDoc sources list the dewarp spec's own sources; own and listed pages appear
    as ``converted`` sources rooted at their page folder, one per split.
    """
    sources = []
    for info in infos:
        if "dewarp_spec" in info:
            for source in info["dewarp_spec"]["sources"]:
                sources.append({**source, "name": f"{info['name']}/{source['name']}"})
            continue
        for split in info.get("samples", {}):
            sources.append(
                {
                    "name": f"{info['name']}/{split}",
                    "kind": "converted",
                    "root": f"files/{info['name']}",
                    "track": info["track"],
                    "split": split,
                    "license": info["license"],
                }
            )
    spec = {"schema": SPEC_SCHEMA, "name": name, "track": track, "sources": sources}
    return DataSpec.from_dict(spec).to_dict()


def license_manifest(infos: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, list]:
    """Per split, each source's track, license and sample counts per license class."""
    manifest: dict[str, list] = {}
    for info in infos:
        for split in info.get("samples", {}):
            mine = [r for r in rows if r["source"] == info["name"] and r["split"] == split]
            classes = Counter(row["license_class"] for row in mine)
            manifest.setdefault(split, []).append(
                {
                    "name": info["name"],
                    "kind": info["kind"],
                    "track": info["track"],
                    "license": info["license"],
                    "samples": len(mine),
                    "license_classes": dict(sorted(classes.items())),
                }
            )
    return dict(sorted(manifest.items()))


def build(
    spec_path: str | Path, output: str | Path, *, workers: int = 2, limit: int | None = None
) -> dict[str, Any]:
    spec = load_build_spec(spec_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    build_spec = {k: v for k, v in spec.items() if k != "base_dir"}
    build_spec["limit"] = limit
    marker = output / "build-spec.json"
    previous = json.loads(marker.read_text("utf-8")) if marker.is_file() else {}
    if previous and previous.get("spec") != build_spec:
        raise _error(f"{output} holds a build of another spec; use a new output directory")
    (output / "parts").mkdir(exist_ok=True)

    tasks, infos = [], []
    for source in spec["sources"]:
        plan, info = plan_source(spec, source, limit)
        infos.append(info)
        for split in sorted(plan):
            items = plan[split]
            settings = {
                "spec_name": spec["name"],
                "source": source["name"],
                "kind": source["kind"],
                "split": split,
                "track": source["track"],
                "long_side": spec["long_side"],
                "jpeg_quality": spec["jpeg_quality"],
                "stored_margin": source.get("stored_margin", 0.15),
                "document_prob": source.get("document_prob", 0.35),
                "form_prob": source.get("form_prob", 0.0),
                "sparse_prob": source.get("sparse_prob", 0.0),
                "print_prob": source.get("print_prob", 0.0),
            }
            for start in range(0, len(items), spec["chunk"]):
                tasks.append(
                    {
                        "key": f"{source['name']}/{split}-{start // spec['chunk']:05d}",
                        "settings": settings,
                        "output": str(output),
                        "indices": list(range(start, min(start + spec["chunk"], len(items)))),
                        "items": items[start : start + spec["chunk"]],
                    }
                )

    # The finished parts hold items planned from the data as it was; a source that has
    # changed since (a longer download, an edited manifest) would mix two plans.
    write_json(marker, {"spec": build_spec})

    def part_path(key: str) -> Path:
        return output / "parts" / f"{key}.jsonl"

    pending = tasks
    started, done = time.perf_counter(), 0
    sizes = {task["key"]: len(task["items"]) for task in pending}
    total_items = sum(sizes.values())

    def finish(result: dict[str, Any]) -> None:
        nonlocal done
        path = part_path(result["key"])
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"skipped": result["skipped"]})] + [
            json.dumps(r) for r in result["rows"]
        ]
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)
        done += sizes[result["key"]]
        elapsed = time.perf_counter() - started
        rate = done / max(elapsed, 1e-9)
        print(
            json.dumps(
                {
                    "part": result["key"],
                    "done": done,
                    "of": total_items,
                    "items_per_sec": rate,
                    "eta_sec": (total_items - done) / max(rate, 1e-9),
                }
            ),
            flush=True,
        )

    if workers <= 1:
        for task in pending:
            finish(run_chunk(task))
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap_unordered(run_chunk, pending):
                finish(result)

    rows, skipped = [], {}
    for task in tasks:
        lines = part_path(task["key"]).read_text(encoding="utf-8").splitlines()
        counts = json.loads(lines[0])["skipped"]
        split_counts = skipped.setdefault(task["settings"]["source"], {}).setdefault(
            task["settings"]["split"], Counter()
        )
        split_counts.update(counts)
        rows.extend(json.loads(line) for line in lines[1:] if line)
    manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    (output / "manifest.jsonl").write_bytes(manifest)
    for info in infos:
        mine = [row for row in rows if row["source"] == info["name"]]
        info["samples"] = dict(sorted(Counter(row["split"] for row in mine).items()))
        info["license_classes"] = dict(
            sorted(Counter(row["license_class"] for row in mine).items())
        )
        info["uniform"] = sum(row["uniform"] for row in mine)
        info["skipped"] = {k: dict(v) for k, v in skipped.get(info["name"], {}).items() if v}
    data_spec = provenance_spec(spec["name"], spec["track"], infos)
    if spec["track"] == "release":
        try:
            require_track(DataSpec.from_dict(data_spec), "release")
        except FlattenerError as error:
            raise _error(error.message) from error
    summary = {
        "schema": DATA_SCHEMA,
        "name": spec["name"],
        "track": spec["track"],
        "complete": True,
        "build_spec": build_spec,
        "sources": infos,
        "data_spec": data_spec,
        "data_sources": license_manifest(infos, rows),
        "records_file": "manifest.jsonl",
        "records": len(rows),
        "counts": dict(sorted(Counter(f"{r['source']}/{r['split']}" for r in rows).items())),
        "generator": {
            "flattener": __version__,
            "documents_version": documents.DOCUMENT_VERSION,
            "long_side": spec["long_side"],
            "jpeg_quality": spec["jpeg_quality"],
        },
        "environment": environment(),
        "created_utc": utc_now(),
    }
    write_json(output / "orient-data.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="build spec JSON (flattener-orient-build-1)")
    parser.add_argument("--output", required=True, help="output directory (${VAR}s expand)")
    parser.add_argument("--workers", type=int, default=2, help="build processes (default 2)")
    parser.add_argument("--limit", type=int, default=None, help="cap every split (smoke builds)")
    args = parser.parse_args(argv)
    summary = build(
        args.spec, Path(expand_path(args.output)), workers=args.workers, limit=args.limit
    )
    print(json.dumps({"records": summary["records"], "counts": summary["counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
