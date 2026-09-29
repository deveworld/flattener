"""Build crumpled-page dewarp data: ``python tools/crumple/build.py --spec S --output D``.

The build spec (``flattener-crumple-build-1``, see ``examples/dewarp``) names a dewarp data
spec (``dewarp_spec``; ``converted`` or ``syntheticdoc_compact`` release sources), the
counts per split, the photo size, a background folder and optional ``modes`` weights of
``render.render`` modes (default all ``crumple``). Each item flattens one
SyntheticDoc render through its ground truth into a page texture (the aspect from the
grid's arc lengths), crumples and photographs it (``render.render``) on a CC0 or
procedural background, and writes ``files/<split>/<id>.jpg`` and ``<id>.bm.npy`` (the
exact forward map, float16, flattener values) plus a ``manifest.jsonl`` row in the
converted layout (``flattener.dewarp.data``), license class and track from the render.
Chunks are written as parts and a rerun resumes from them; each item is seeded by
(spec name, split, index).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import multiprocessing
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render  # noqa: E402

from flattener.dewarp.checkpoint import write_json  # noqa: E402
from flattener.dewarp.composite import BackgroundBank, procedural_background  # noqa: E402
from flattener.dewarp.coords import map_to_grid, syntheticdoc_to_flattener  # noqa: E402
from flattener.dewarp.data import canonical_json, expand_path, release_license_ok  # noqa: E402

BUILD_SCHEMA = "flattener-crumple-build-1"


def _orient_builder():
    path = HERE.parent / "orient" / "build_data.py"
    spec = importlib.util.spec_from_file_location("orient_build_data", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(*parts: Any) -> int:
    text = "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


@lru_cache(maxsize=1)
def _bank(directory: str | None) -> BackgroundBank | None:
    if not directory:
        return None
    return BackgroundBank(directory, license_ok=release_license_ok)


def flat_texture(record: dict[str, Any], long_side: int = 1024, device: str = "cpu") -> np.ndarray:
    """The render flattened through its ground truth, sized by the grid's arc lengths."""
    import torch
    import torch.nn.functional as F

    with Image.open(record["image"]) as image:
        rgb = np.array(image.convert("RGB"))
    values = np.load(record["backward_map"], allow_pickle=False).astype(np.float32)
    if record["render_size"] is not None:
        values = syntheticdoc_to_flattener(values, record["render_size"])
    grid = map_to_grid(values, (45, 31)).astype(np.float64)
    height, width = rgb.shape[:2]
    px = (grid + 1) / 2 * np.array([width, height])
    across = np.linalg.norm(np.diff(px, axis=1), axis=-1).sum(axis=1).mean()
    down = np.linalg.norm(np.diff(px, axis=0), axis=-1).sum(axis=0).mean()
    aspect = down / max(across, 1e-6)
    size = (
        (long_side, round(long_side / aspect))
        if aspect >= 1
        else (round(long_side * aspect), long_side)
    )
    dense = torch.from_numpy(np.asarray(values, np.float32)).to(device).permute(2, 0, 1)[None]
    dense = F.interpolate(dense, size=size, mode="bilinear", align_corners=False)
    source = torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float()
    flat = F.grid_sample(source, dense.permute(0, 2, 3, 1), align_corners=False, mode="bilinear")
    return flat[0].permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy()


def run_chunk(task: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    settings, output = task["settings"], Path(task["output"])
    folder = output / "files" / settings["split"]
    folder.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    out_hw = tuple(settings["photo_hw"])
    for index, record in zip(task["indices"], task["items"], strict=True):
        rng = np.random.default_rng(_seed(settings["name"], settings["split"], index))
        try:
            texture = flat_texture(record, device=settings["device"])
            bank = _bank(settings["background_dir"])
            if bank is not None and len(bank) and rng.random() < 0.75:
                background = bank.draw(rng, out_hw)
            else:
                background = procedural_background(rng, out_hw)
            modes = settings.get("modes") or {"crumple": 1.0}
            names = sorted(modes)
            weights = np.array([modes[name] for name in names], dtype=np.float64)
            mode = names[int(rng.choice(len(names), p=weights / weights.sum()))]
            photo, forward = render.render(
                texture, rng, out_hw, np.clip(background, 0, 255).astype(np.uint8), mode,
                settings["device"],
            )  # fmt: skip
        except (OSError, ValueError, RuntimeError):
            skipped += 1
            continue
        if not np.isfinite(forward).all() or np.abs(forward).max() > 1.0:
            skipped += 1
            continue
        ident = f"crumple-{settings['split']}-{index:07d}"
        buffer = io.BytesIO()
        Image.fromarray(photo).save(buffer, "JPEG", quality=92)
        image_bytes = buffer.getvalue()
        (folder / f"{ident}.jpg").write_bytes(image_bytes)
        map_buffer = io.BytesIO()
        np.save(map_buffer, forward.astype(np.float16))
        map_bytes = map_buffer.getvalue()
        (folder / f"{ident}.bm.npy").write_bytes(map_bytes)
        rows.append(
            {
                "schema": "flattener-dewarp-sample-1",
                "sample_id": ident,
                "split": settings["split"],
                "image": f"files/{settings['split']}/{ident}.jpg",
                "backward_map": f"files/{settings['split']}/{ident}.bm.npy",
                "backward_map_convention": "flattener",
                "image_size": list(out_hw),
                "track": record["track"],
                "license_class": record["license_class"],
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_bytes": len(image_bytes),
                "backward_map_sha256": hashlib.sha256(map_bytes).hexdigest(),
                "backward_map_bytes": len(map_bytes),
                "texture_sample_id": record["sample_id"],
                "mode": mode,
            }
        )
    return {"key": task["key"], "rows": rows, "skipped": skipped}


def build(
    spec_path: Path, output: Path, *, workers: int, limit: int | None, device: str = "cpu"
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != BUILD_SCHEMA:
        raise SystemExit(f"error: {spec_path}: schema must be {BUILD_SCHEMA}")
    base = spec_path.resolve().parent
    orient = _orient_builder()
    dewarp_spec = Path(base, expand_path(spec["dewarp_spec"])).resolve()
    dewarp, by_split, described = orient.index_dewarp_spec(str(dewarp_spec), (45, 31))
    if spec["track"] == "release":
        from flattener.dewarp.data import require_track

        require_track(dewarp, "release")
    background_dir = spec.get("background_dir")
    if background_dir:
        background_dir = str(Path(base, expand_path(background_dir)))
    output.mkdir(parents=True, exist_ok=True)
    (output / "parts").mkdir(exist_ok=True)
    hashed = {**spec, "limit": limit}
    marker = output / "build-spec.json"
    spec_sha = hashlib.sha256(canonical_json(hashed)).hexdigest()
    if marker.is_file() and json.loads(marker.read_text())["sha256"] != spec_sha:
        raise SystemExit(f"error: {output} holds another build")
    write_json(marker, {"sha256": spec_sha, "spec": hashed})
    tasks = []
    for split, count in spec["count"].items():
        count = min(count, limit) if limit else count
        records = by_split[split]
        rng = np.random.default_rng(_seed(spec["name"], split, "pick"))
        picks = rng.choice(len(records), size=count, replace=count > len(records))
        items = [records[int(i)] for i in picks]
        settings = {"name": spec["name"], "split": split, "photo_hw": spec["photo_hw"],
                    "background_dir": background_dir, "modes": spec.get("modes"),
                    "device": device}  # fmt: skip
        for start in range(0, count, spec.get("chunk", 200)):
            stop = min(count, start + spec.get("chunk", 200))
            tasks.append({"key": f"{split}-{start:07d}", "settings": settings,
                          "output": str(output), "indices": list(range(start, stop)),
                          "items": items[start:stop]})  # fmt: skip
    pending = [t for t in tasks if not (output / "parts" / f"{t['key']}.jsonl").is_file()]
    total, done, started = sum(len(t["items"]) for t in pending), 0, time.perf_counter()

    def finish(result):
        nonlocal done
        path = output / "parts" / f"{result['key']}.jsonl"
        lines = [json.dumps({"skipped": result["skipped"]})]
        lines += [json.dumps(row) for row in result["rows"]]
        path.with_suffix(".tmp").write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.with_suffix(".tmp").replace(path)
        done += len(result["rows"]) + result["skipped"]
        rate = done / max(time.perf_counter() - started, 1e-9)
        print(json.dumps({"part": result["key"], "done": done, "of": total,
                          "items_per_sec": round(rate, 1)}), flush=True)  # fmt: skip

    if workers <= 1:
        for task in pending:
            finish(run_chunk(task))
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap_unordered(run_chunk, pending):
                finish(result)
    rows, skipped = [], 0
    for task in tasks:
        lines = (output / "parts" / f"{task['key']}.jsonl").read_text().splitlines()
        skipped += json.loads(lines[0])["skipped"]
        rows.extend(json.loads(line) for line in lines[1:] if line)
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {"schema": BUILD_SCHEMA, "name": spec["name"], "records": len(rows),
               "skipped": skipped, "texture_sources": described,
               "dewarp_spec_sha256": dewarp.sha256}  # fmt: skip
    write_json(output / "crumple-data.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="torch device for the rendering")
    args = parser.parse_args(argv)
    summary = build(args.spec, Path(expand_path(args.output)), workers=args.workers,
                    limit=args.limit, device=args.device)  # fmt: skip
    print(json.dumps({"records": summary["records"], "skipped": summary["skipped"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
