"""Build page gate training data: ``python tools/gate/build_data.py --spec S --output D``.

The build spec (``flattener-gate-build-1``, see ``examples/gate``) names a scene config
(``scenes.SceneConfig`` fields), a ``background_dir`` of CC0 textures and sources:

* ``syntheticdoc_scene``: renders of a dewarp data spec (``dewarp_spec``, relative to the
  build spec; sources ``converted`` or ``syntheticdoc_compact``) placed in scenes, with
  another render of the same split as the facing page. A release build requires
  ``require_track(spec, "release")``.
* ``documents_scene``: own generated pages (``tools/orient/pages.py``) placed in scenes.
* ``negative_scene``: scenes without a page (empty mask).
* ``smartdoc_capture``: SmartDoc 2015 challenge 1 captures under ``root`` (the benchmark
  folder's ``manifest.json``, or one written by ``smartdoc_frames.py``), their page quads
  as masks. ``families`` maps each split to the document numbers it takes
  (``datasheet004`` is number 4); other captures are left out.
* ``labeled_photos``: own photos under ``root`` with the page polygons of
  ``real_labels.py manifest`` (EXIF-applied pixels; none for a photo without page), in
  the splits written there.

Output: ``files/<source>/<split>/<id>.jpg`` (scene) and ``<id>.png`` (mask, 255 = page),
``parts/`` with one JSONL per finished chunk (an interrupted build resumes from them when
the spec and the planned items are unchanged), then ``manifest.jsonl`` and
``gate-data.json`` (schema ``flattener-gate-data-1``, read by ``flattener.regions.train``)
with a dewarp-schema ``data_spec`` and a per-split license manifest ``data_sources``, which
checkpoints copy for the ship gate. Each item is seeded by (spec name, source, split,
index), so a rebuild gives the same scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import multiprocessing
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _orient_builder():
    """``tools/orient/build_data.py`` (its planning and license helpers)."""
    path = HERE.parent / "orient" / "build_data.py"
    spec = importlib.util.spec_from_file_location("orient_build_data", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


orient = _orient_builder()
_evenly, _seed = orient._evenly, orient._seed
index_dewarp_spec, license_manifest = orient.index_dewarp_spec, orient.license_manifest
provenance_spec = orient.provenance_spec
pages = orient.pages

import scenes  # noqa: E402

from flattener import FlattenerError, __version__  # noqa: E402
from flattener.dewarp.checkpoint import environment, utc_now, write_json  # noqa: E402
from flattener.dewarp.composite import BackgroundBank  # noqa: E402
from flattener.dewarp.coords import map_to_grid, syntheticdoc_to_flattener  # noqa: E402
from flattener.dewarp.data import (  # noqa: E402
    DataSpec,
    canonical_json,
    expand_path,
    release_license_ok,
    require_track,
)
from flattener.dewarp.model import sha256_file  # noqa: E402

BUILD_SCHEMA = "flattener-gate-build-1"
DATA_SCHEMA = "flattener-gate-data-1"
KINDS = (
    "syntheticdoc_scene", "documents_scene", "negative_scene", "smartdoc_capture",
    "labeled_photos",
)  # fmt: skip
SOURCE_KEYS = {
    "name", "kind", "track", "license", "count", "dewarp_spec", "part", "root", "families",
    "document_prob", "form_prob", "sparse_prob",
}  # fmt: skip
OUTLINE_GRID = (33, 23)
PAPER = ((238, 250), (232, 246), (220, 238))


def _error(message: str) -> SystemExit:
    return SystemExit(f"error: {message}")


def load_build_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != BUILD_SCHEMA:
        raise _error(f"{path}: schema must be {BUILD_SCHEMA}")
    if spec.get("track") not in ("release", "research") or not spec.get("name"):
        raise _error("spec needs a name and track release|research")
    names = set()
    for source in spec.get("sources", []):
        unknown = set(source) - SOURCE_KEYS
        if unknown or source.get("kind") not in KINDS or source.get("name") in names:
            raise _error(f"bad source {source.get('name')!r}: keys {sorted(unknown)} or kind/name")
        names.add(source["name"])
        if source.get("track") not in ("release", "research") or not source.get("license"):
            raise _error(f"source {source['name']!r} needs a track and a license")
        if spec["track"] == "release" and source["track"] != "release":
            raise _error(f"release build lists {source['track']} source {source['name']!r}")
        if spec["track"] == "release" and not release_license_ok(source["license"]):
            raise _error(f"source {source['name']!r}: license {source['license']!r} is not release")
    if not names:
        raise _error("spec lists no sources")
    scenes.SceneConfig.from_dict(spec.get("scene"))  # fails early on unknown fields
    spec.setdefault("jpeg_quality", 92)
    spec.setdefault("chunk", 250)
    spec["base_dir"] = str(path.resolve().parent)
    return spec


def _family(template: str) -> int:
    match = re.search(r"(\d+)$", template)
    return int(match.group(1)) if match else -1


def plan_source(spec: dict[str, Any], source: dict[str, Any], limit: int | None):
    """Per split, the items to build, plus the source's summary fields."""
    counts = source.get("count") or {}
    info: dict[str, Any] = {k: source[k] for k in ("name", "kind", "track", "license")}
    plan: dict[str, list] = {}

    def cap(split: str) -> int | None:
        values = [v for v in (counts.get(split), limit) if v is not None]
        return min(values) if values else None

    kind = source["kind"]
    if kind == "syntheticdoc_scene":
        path = Path(spec["base_dir"], expand_path(source["dewarp_spec"])).resolve()
        dewarp, by_split, described = index_dewarp_spec(str(path), OUTLINE_GRID)
        if source["track"] == "release":
            try:
                require_track(dewarp, "release")
            except FlattenerError as error:
                raise _error(error.message) from error
        part, parts = source.get("part", (0, 1))
        for split, records in by_split.items():
            if split not in counts:
                continue
            chosen = _evenly([r for i, r in enumerate(records) if i % parts == part], cap(split))
            rng = np.random.default_rng(_seed(spec["name"], source["name"], split, "neighbors"))
            others = rng.integers(len(records), size=len(chosen))
            plan[split] = [
                {"page": record, "neighbor": records[int(other)]}
                for record, other in zip(chosen, others, strict=True)
            ]
            if scenes.SceneConfig.from_dict(spec.get("scene")).under_prob > 0:
                rng = np.random.default_rng(_seed(spec["name"], source["name"], split, "under"))
                for item, other in zip(plan[split], rng.integers(len(records), size=len(chosen))):
                    item["under"] = records[int(other)]
        info.update(
            dewarp_spec=dewarp.to_dict(), dewarp_spec_sha256=dewarp.sha256, dewarp_sources=described
        )
    elif kind == "smartdoc_capture":
        root = Path(spec["base_dir"], expand_path(source["root"])).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for split, families in (source.get("families") or {}).items():
            rows = [
                {"path": str(root / c["path"]), "corners": c["corners_xy"],
                 "template": c["template"], "background": c["background"]}
                for c in manifest["captures"] if _family(c["template"]) in families
            ]  # fmt: skip
            plan[split] = _evenly(rows, cap(split))
        info["root_manifest_sha256"] = sha256_file(root / "manifest.json")
    elif kind == "labeled_photos":
        root = Path(spec["base_dir"], expand_path(source["root"])).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for row in manifest["photos"]:
            plan.setdefault(row["split"], []).append({**row, "path": str(root / row["path"])})
        for split in plan:
            plan[split] = _evenly(plan[split], cap(split))
        info["root_manifest_sha256"] = sha256_file(root / "manifest.json")
    else:
        for split, count in counts.items():
            plan[split] = list(range(cap(split) or count))
    return plan, info


@lru_cache(maxsize=2)
def _bank(directory: str | None) -> BackgroundBank | None:
    if not directory:
        return None
    return BackgroundBank(directory, license_ok=release_license_ok)


def _render(record: dict[str, Any]) -> tuple[Image.Image, np.ndarray]:
    with Image.open(record["image"]) as image:
        rgb = image.convert("RGB")
    values = np.load(record["backward_map"], allow_pickle=False)
    if record["render_size"] is not None:
        values = syntheticdoc_to_flattener(values, record["render_size"])
    return rgb, map_to_grid(values, OUTLINE_GRID)


def _own_page(page_id: str, rng: np.random.Generator, settings: dict[str, Any]):
    gray, _, layout = pages.document_page(
        page_id, rng, 1024, settings["document_prob"], settings["form_prob"],
        settings["sparse_prob"],
    )  # fmt: skip
    paper = np.array([rng.uniform(*low_high) for low_high in PAPER], np.float32) / 255.0
    rgb = (gray.astype(np.float32)[..., None] * paper).astype(np.uint8)
    return (Image.fromarray(rgb), scenes.flat_grid()), layout


def make_scene(kind: str, item: Any, rng: np.random.Generator, settings: dict[str, Any]):
    """(scene RGB, mask, row fields) for one item."""
    config = scenes.SceneConfig.from_dict(settings["scene"])
    if kind == "smartdoc_capture":
        with Image.open(item["path"]) as image:
            rgb = image.convert("RGB")
        scale = config.long_side / max(rgb.size)
        size = (round(rgb.size[0] * scale), round(rgb.size[1] * scale))
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        mask = Image.new("L", size, 0)
        corners = [(x * scale, y * scale) for x, y in item["corners"]]
        ImageDraw.Draw(mask).polygon(corners, fill=255)
        fields = {
            "license_class": "CC-BY-4.0",
            "track": "release",
            "origin": {k: item[k] for k in ("template", "background")}
            | {"file": Path(item["path"]).name},
        }
        return np.asarray(rgb), np.asarray(mask) > 127, fields
    if kind == "labeled_photos":
        with Image.open(item["path"]) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        scale = config.long_side / max(rgb.size)
        size = (round(rgb.size[0] * scale), round(rgb.size[1] * scale))
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        mask = Image.new("L", size, 0)
        for polygon in item["polygons"]:
            ImageDraw.Draw(mask).polygon([(x * scale, y * scale) for x, y in polygon], fill=255)
        fields = {
            "license_class": "own",
            "track": settings["track"],
            "origin": {"id": item["id"], "group": item["group"]},
        }
        return np.asarray(rgb), np.asarray(mask) > 127, fields
    bank = _bank(settings["background_dir"])
    if kind == "syntheticdoc_scene":
        page, neighbor = _render(item["page"]), _render(item["neighbor"])
        under = [_render(item["under"])] if "under" in item else None
        image, mask, info = scenes.scene(rng, config, page, bank, neighbor, under)
        used = [item[key] for key in ("page", "neighbor", "under") if key in item]
        fields = {
            "license_class": item["page"]["license_class"],
            "track": item["page"]["track"],
            "origin": {
                name: item[key]["sample_id"]
                for key, name in (("page", "sample_id"), ("neighbor", "neighbor_id"),
                                  ("under", "under_id"))
                if key in item
            },
        }  # fmt: skip
        if any(record["license_class"] != item["page"]["license_class"] for record in used):
            fields["license_class"] = "clean"
        return image, mask, fields | {"scene": info}
    own = {"license_class": "own", "track": settings["track"]}
    if kind == "documents_scene":
        page_id = f"{settings['source']}-{settings['split']}-{item}"
        page, layout = _own_page(page_id, rng, settings)
        neighbor, _ = _own_page(page_id + "-facing", rng, settings)
        under = None
        if config.under_prob > 0:
            under = [_own_page(page_id + "-under", rng, settings)[0]]
        image, mask, info = scenes.scene(rng, config, page, bank, neighbor, under)
        return image, mask, own | {"origin": {"layout": layout}, "scene": info}
    image, mask, info = scenes.scene(rng, config, None, bank)
    return image, mask, own | {"scene": info}


def run_chunk(task: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    settings, output = task["settings"], Path(task["output"])
    folder = output / "files" / settings["source"] / settings["split"]
    folder.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], Counter()
    for index, item in zip(task["indices"], task["items"], strict=True):
        rng = np.random.default_rng(
            _seed(settings["spec_name"], settings["source"], settings["split"], index)
        )
        try:
            image, mask, fields = make_scene(settings["kind"], item, rng, settings)
        except (OSError, ValueError, pages.Skip) as error:
            skipped[type(error).__name__] += 1
            continue
        identifier = f"{settings['source']}-{settings['split']}-{index:07d}"
        relative = f"files/{settings['source']}/{settings['split']}/{identifier}"
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, "JPEG", quality=settings["jpeg_quality"])
        data = buffer.getvalue()
        (output / f"{relative}.jpg").write_bytes(data)
        Image.fromarray((mask * 255).astype(np.uint8)).save(output / f"{relative}.png")
        rows.append(
            {
                "id": identifier,
                "split": settings["split"],
                "source": settings["source"],
                "kind": settings["kind"],
                "path": f"{relative}.jpg",
                "mask": f"{relative}.png",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": list(image.shape[:2]),
                "area_fraction": round(float(mask.mean()), 4),
                **fields,
            }
        )
    return {"key": task["key"], "rows": rows, "skipped": dict(skipped)}


def code_record() -> dict[str, str]:
    return {
        "tools/gate/build_data.py": sha256_file(HERE / "build_data.py"),
        "tools/gate/scenes.py": sha256_file(HERE / "scenes.py"),
        "tools/orient/pages.py": sha256_file(HERE.parent / "orient" / "pages.py"),
        "flattener/dewarp/composite.py": sha256_file(Path(scenes.cp.__file__)),
    }


def build(
    spec_path: str | Path, output: str | Path, *, workers: int = 2, limit: int | None = None
) -> dict[str, Any]:
    spec = load_build_spec(spec_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashed = {k: v for k, v in spec.items() if k != "base_dir"}
    hashed["limit"] = limit
    spec_sha = hashlib.sha256(canonical_json(hashed)).hexdigest()
    marker = output / "build-spec.json"
    previous = json.loads(marker.read_text("utf-8")) if marker.is_file() else {}
    if previous and previous.get("sha256") != spec_sha:
        raise _error(f"{output} holds a build of another spec; use a new output directory")
    (output / "parts").mkdir(exist_ok=True)
    background_dir = spec.get("background_dir")
    if background_dir:
        background_dir = str(Path(spec["base_dir"], expand_path(background_dir)))
        bank = _bank(background_dir)
        background = bank.describe()
    else:
        background = None

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
                "jpeg_quality": spec["jpeg_quality"],
                "scene": spec.get("scene") or {},
                "background_dir": background_dir,
                "document_prob": source.get("document_prob", 0.35),
                "form_prob": source.get("form_prob", 0.0),
                "sparse_prob": source.get("sparse_prob", 0.0),
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
    planned = [[task["key"], task["items"]] for task in tasks]
    plan_sha = hashlib.sha256(
        json.dumps(planned, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if previous.get("plan_sha256") not in (None, plan_sha):
        raise _error(f"{output} was built from other input data; use a new output directory")
    write_json(marker, {"sha256": spec_sha, "plan_sha256": plan_sha, "spec": hashed})

    def part_path(key: str) -> Path:
        return output / "parts" / f"{key}.jsonl"

    pending = [task for task in tasks if not part_path(task["key"]).is_file()]
    started, done = time.perf_counter(), 0
    sizes = {task["key"]: len(task["items"]) for task in pending}
    total = sum(sizes.values())

    def finish(result: dict[str, Any]) -> None:
        nonlocal done
        path = part_path(result["key"])
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"skipped": result["skipped"]})]
        lines += [json.dumps(row) for row in result["rows"]]
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)
        done += sizes[result["key"]]
        rate = done / max(time.perf_counter() - started, 1e-9)
        eta = round((total - done) / max(rate, 1e-9))
        progress = {"part": result["key"], "done": done, "of": total, "eta_sec": eta}
        print(json.dumps(progress | {"items_per_sec": round(rate, 1)}), flush=True)

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
        counts = skipped.setdefault(task["settings"]["source"], {}).setdefault(
            task["settings"]["split"], Counter()
        )
        counts.update(json.loads(lines[0])["skipped"])
        rows.extend(json.loads(line) for line in lines[1:] if line)
    manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    (output / "manifest.jsonl").write_bytes(manifest)
    for info in infos:
        mine = [row for row in rows if row["source"] == info["name"]]
        info["samples"] = dict(sorted(Counter(row["split"] for row in mine).items()))
        info["license_classes"] = dict(sorted(Counter(r["license_class"] for r in mine).items()))
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
        "build_spec": hashed,
        "build_spec_sha256": spec_sha,
        "sources": infos,
        "backgrounds": background,
        "data_spec": data_spec,
        "data_sources": license_manifest(infos, rows),
        "records_file": "manifest.jsonl",
        "records": len(rows),
        "records_sha256": hashlib.sha256(manifest).hexdigest(),
        "counts": dict(sorted(Counter(f"{r['source']}/{r['split']}" for r in rows).items())),
        "generator": {"flattener": __version__, "code_sha256": code_record()},
        "environment": environment(),
        "created_utc": utc_now(),
    }
    write_json(output / "gate-data.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="build spec JSON (flattener-gate-build-1)")
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
