"""Re-texture UVDoc's raw captures with license-clean pages: ``python tools/uvdoc/retexture.py
--spec S --output D``.

UVDoc's raw set (github.com/tanguymagne/UVDoc-Dataset, MIT, (c) 2023 Tanguy Magne) holds the
authors' photos of deformed blank paper: ``samples/rgb/<id>.png``, a per-pixel texture
coordinate map ``samples/uvmap/<id>.mat`` (``uv``, 0-1, NaN off the page), a page mask
``samples/seg/<id>.mat`` and the page grid ``samples/grid2d/<id>.mat`` (89x61 pixel
coordinates). Its own ``create_final.py`` lays a document on each photo through the UV
map, multiplies it into the photographed paper (shading strength 0.75) and swaps the
background; the released composites use DTD backgrounds (research only) and partly
DeepFloyd-generated pages (non-commercial), so they stay off the release track. This tool
does the same with our pages: clean SyntheticDoc renders flattened through their ground
truth (``tools/crumple/build.py``'s ``flat_texture``) or own generated pages, on CC0 or
procedural backgrounds, and it never reads the raw ``textures/`` or ``backgrounds/``.

The build spec (``flattener-uvdoc-retexture-1``, see ``examples/dewarp``) names the raw
root, the benchmark whose captures stay out (``exclude_benchmark``: a folder with
``metadata_geom/*.json``; their ``sample_id`` captures are skipped), the dewarp data
spec the pages come from, the photo size, counts per split (captures are split by a hash
of their id, ``val_share`` to val), ``own_texture_prob``, ``background_dir``,
``capture_background_prob`` (keep the photo's own surroundings), ``shade`` (strength
range) and ``margin`` (crop margin around the page, share of its size per side). Each
item flips the capture as UVDoc does (texture coordinates follow, so text never
mirrors), crops a window around the page, resizes it to the photo size and stores the
exact page grid as a center-aligned flattener map in the converted layout, with
``manifest.jsonl`` rows like ``tools/crumple/build.py``. Captures are processed one at a
time (all items of a capture share its files); parts resume.
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
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flattener.dewarp.checkpoint import write_json  # noqa: E402
from flattener.dewarp.composite import procedural_background  # noqa: E402
from flattener.dewarp.coords import grid_to_map, pixels_to_flattener  # noqa: E402
from flattener.dewarp.data import canonical_json, expand_path  # noqa: E402

BUILD_SCHEMA = "flattener-uvdoc-retexture-1"
MAP_HW = (177, 121)
SOURCE = "UVDoc raw captures, MIT, (c) 2023 Tanguy Magne (github.com/tanguymagne/UVDoc-Dataset)"


def _crumple_build():
    path = HERE.parent / "crumple" / "build.py"
    spec = importlib.util.spec_from_file_location("crumple_build", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(*parts: Any) -> int:
    text = "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def read_matrix(path: Path, key: str) -> np.ndarray:
    """A MATLAB v7.3 (HDF5) or older .mat array in numpy axis order."""
    try:
        import h5py

        with h5py.File(path, "r") as stream:
            return np.array(stream[key][:]).T
    except OSError:
        from scipy.io import loadmat

        return np.asarray(loadmat(path)[key])


def load_capture(raw: Path, capture: str) -> dict[str, np.ndarray]:
    """One raw capture: the photo (H, W, 3) uint8, ``uv`` (H, W, 2) with NaN off the page,
    the page mask (H, W) bool and the page grid (89, 61, 2) in pixel coordinates (rows
    down the page)."""
    samples = raw / "samples"
    with Image.open(samples / "rgb" / f"{capture}.png") as image:
        photo = np.asarray(image.convert("RGB"))
    uv = read_matrix(samples / "uvmap" / f"{capture}.mat", "uv").astype(np.float32)
    seg = read_matrix(samples / "seg" / f"{capture}.mat", "seg") > 0.5
    grid = read_matrix(samples / "grid2d" / f"{capture}.mat", "grid2d").astype(np.float64)
    height, width = photo.shape[:2]
    if uv.shape[:2] != (height, width) and uv.shape[:2] == (width, height):
        uv = uv.transpose(1, 0, 2)
    if seg.shape != (height, width) and seg.shape == (width, height):
        seg = seg.T
    if grid.shape[0] < grid.shape[1]:  # rows run down the page (89 of them)
        grid = grid.transpose(1, 0, 2)
    return {"photo": photo, "uv": uv, "seg": seg, "grid": grid}


def flip(capture: dict[str, np.ndarray], horizontal: bool, vertical: bool) -> dict:
    """The capture mirrored as UVDoc's augment_with_flip does: the photo, mask and texture
    coordinates flip together and the grid's node order follows, so the page reads the
    same way round. Pixel centers sit at integers here (x -> W - 1 - x)."""
    photo, uv, seg, grid = capture["photo"], capture["uv"], capture["seg"], capture["grid"]
    height, width = photo.shape[:2]
    uv, grid = uv.copy(), grid.copy()
    if horizontal:
        photo, seg, uv = photo[:, ::-1], seg[:, ::-1], uv[:, ::-1]
        uv[..., 0] = 1 - uv[..., 0]
        grid[..., 0] = width - 1 - grid[..., 0]
        grid = grid[:, ::-1]
    if vertical:
        photo, seg, uv = photo[::-1], seg[::-1], uv[::-1]
        uv[..., 1] = 1 - uv[..., 1]
        grid[..., 1] = height - 1 - grid[..., 1]
        grid = grid[::-1]
    return {"photo": np.ascontiguousarray(photo), "uv": np.ascontiguousarray(uv),
            "seg": np.ascontiguousarray(seg), "grid": np.ascontiguousarray(grid)}  # fmt: skip


def compose(
    capture: dict[str, np.ndarray],
    texture: np.ndarray,
    background: np.ndarray | None,
    shade: float,
) -> np.ndarray:
    """The page laid on the photographed paper through the UV map, ``shade`` of the
    paper's light multiplied in (create_final.py uses 0.75), on ``background`` (None keeps
    the photo's own surroundings)."""
    import torch
    import torch.nn.functional as F

    photo = capture["photo"].astype(np.float32) / 255
    uv = capture["uv"]
    inside = np.isfinite(uv).all(axis=-1) & capture["seg"]
    coords = np.where(inside[..., None], uv * 2 - 1, -2.0).astype(np.float32)
    source = torch.from_numpy(np.ascontiguousarray(texture)).permute(2, 0, 1)[None].float() / 255
    warped = F.grid_sample(source, torch.from_numpy(coords)[None], align_corners=False)
    warped = _blur(warped, 0.5)[0].permute(1, 2, 0).numpy()
    page = photo * (shade * warped + (1 - shade))
    # A soft edge one pixel past the page, so the paper's border blends into the background.
    mask = torch.from_numpy(inside.astype(np.float32))[None, None]
    grown = F.max_pool2d(mask, 3, stride=1, padding=1)
    edge = np.maximum(_blur(grown, 0.75)[0, 0].numpy(), inside)[..., None]
    surround = photo if background is None else background.astype(np.float32) / 255
    out = page * inside[..., None] + surround * (1 - inside[..., None])
    out = out * edge + surround * (1 - edge) if background is not None else out
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def _blur(image, sigma: float):
    """Gaussian blur of an (N, C, H, W) tensor, per channel, edges replicated."""
    import torch
    import torch.nn.functional as F

    radius = max(1, int(np.ceil(3 * sigma)))
    taps = torch.exp(
        -(torch.arange(-radius, radius + 1, dtype=torch.float32) ** 2) / (2 * sigma**2)
    )
    taps = taps / taps.sum()
    channels = image.shape[1]
    out = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    out = F.conv2d(out, taps.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    return F.conv2d(out, taps.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


def crop_window(
    grid: np.ndarray, size: tuple[int, int], out_hw: tuple[int, int], margin: np.ndarray
) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) around the page with ``margin`` (left, top, right, bottom, share of
    the page's extent) widened to the photo's aspect where the image allows, clipped."""
    height, width = size
    low, high = grid.reshape(-1, 2).min(0), grid.reshape(-1, 2).max(0)
    extent = high - low
    x0, y0 = low - margin[:2] * extent
    x1, y1 = high + margin[2:] * extent
    aspect = out_hw[0] / out_hw[1]  # height / width
    if (y1 - y0) / (x1 - x0) < aspect:  # too wide: grow the height
        grow = (x1 - x0) * aspect - (y1 - y0)
        y0, y1 = y0 - grow / 2, y1 + grow / 2
    else:
        grow = (y1 - y0) / aspect - (x1 - x0)
        x0, x1 = x0 - grow / 2, x1 + grow / 2
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    return x0, y0, x1, y1


def render_item(
    capture: dict[str, np.ndarray],
    texture: np.ndarray,
    background: np.ndarray | None,
    rng: np.random.Generator,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """(photo, center-aligned flattener map MAP_HW) of one item."""
    out_hw = tuple(settings["photo_hw"])
    flipped = flip(capture, bool(rng.random() < 0.5), bool(rng.random() < 0.5))
    height, width = flipped["photo"].shape[:2]
    shade = float(rng.uniform(*settings.get("shade", (0.65, 0.85))))
    composed = compose(flipped, texture, background, shade)
    margin = rng.uniform(*settings.get("margin", (0.02, 0.15)), size=4)
    x0, y0, x1, y1 = crop_window(flipped["grid"], (height, width), out_hw, margin)
    crop = Image.fromarray(composed[y0:y1, x0:x1]).resize(
        (out_hw[1], out_hw[0]), Image.Resampling.BILINEAR
    )
    scale = np.array([out_hw[1] / (x1 - x0), out_hw[0] / (y1 - y0)])
    pixels = (flipped["grid"] - [x0, y0] + 0.5) * scale - 0.5  # pixel centers at integers
    values = pixels_to_flattener(pixels, out_hw)
    forward = grid_to_map(values, MAP_HW)
    return np.asarray(crop), forward


def run_capture(task: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    settings, output = task["settings"], Path(task["output"])
    split, capture_id = settings["split"], task["capture"]
    folder = output / "files" / split
    folder.mkdir(parents=True, exist_ok=True)
    build = _crumple_build()
    capture = load_capture(Path(settings["raw_root"]), capture_id)
    out_hw = tuple(settings["photo_hw"])
    rows, skipped = [], 0
    for index, record in zip(task["indices"], task["items"], strict=True):
        rng = np.random.default_rng(_seed(settings["name"], split, index))
        try:
            own = rng.random() < settings.get("own_texture_prob", 0.0)
            if own:
                texture = build.own_texture(rng, f"{settings['name']}-{split}-{index}")
            else:
                texture = build.flat_texture(record)
            if rng.random() < settings.get("capture_background_prob", 0.0):
                background = None
            else:
                height, width = capture["photo"].shape[:2]
                bank = build._bank(settings["background_dir"])
                if bank is not None and len(bank) and rng.random() < 0.75:
                    background = bank.draw(rng, (height, width))
                else:
                    background = procedural_background(rng, (height, width))
                background = np.clip(background, 0, 255).astype(np.uint8)
            photo, forward = render_item(capture, texture, background, rng, settings)
        except (OSError, ValueError, RuntimeError):
            skipped += 1
            continue
        if not np.isfinite(forward).all() or np.abs(forward).max() > 1.0:
            skipped += 1
            continue
        ident = f"uvdoc-{split}-{index:07d}"
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
                "split": split,
                "image": f"files/{split}/{ident}.jpg",
                "backward_map": f"files/{split}/{ident}.bm.npy",
                "backward_map_convention": "flattener",
                "image_size": list(out_hw),
                "track": record["track"],
                "license_class": "own" if own else record["license_class"],
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_bytes": len(image_bytes),
                "backward_map_sha256": hashlib.sha256(map_bytes).hexdigest(),
                "backward_map_bytes": len(map_bytes),
                "texture_sample_id": None if own else record["sample_id"],
                "capture": capture_id,
                "geometry_source": SOURCE,
            }
        )
    return {"key": task["key"], "rows": rows, "skipped": skipped}


def captures(raw: Path, exclude: Path | None) -> list[str]:
    """Raw capture ids, less the benchmark's."""
    ids = sorted(p.stem for p in (raw / "samples" / "rgb").glob("*.png"))
    if exclude is not None:
        held = {
            json.loads(p.read_text(encoding="utf-8")).get("sample_id")
            for p in (exclude / "metadata_geom").glob("*.json")
        }
        ids = [capture for capture in ids if capture not in held]
    return ids


def build(spec_path: Path, output: Path, *, workers: int, limit: int | None) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != BUILD_SCHEMA:
        raise SystemExit(f"error: {spec_path}: schema must be {BUILD_SCHEMA}")
    base = spec_path.resolve().parent
    raw = Path(expand_path(spec["raw_root"]))
    exclude = (
        Path(expand_path(spec["exclude_benchmark"])) if spec.get("exclude_benchmark") else None
    )
    ids = captures(raw, exclude)
    if not ids:
        raise SystemExit(f"error: no captures under {raw / 'samples' / 'rgb'}")
    share = spec.get("val_share", 0.05)
    by_split = {
        "val": [c for c in ids if _seed(spec["name"], "split", c) % 1000 < share * 1000],
    }
    by_split["train"] = [c for c in ids if c not in set(by_split["val"])]
    build_module = _crumple_build()
    orient = build_module._orient_builder()
    dewarp_spec = Path(base, expand_path(spec["dewarp_spec"])).resolve()
    _, pages, described = orient.index_dewarp_spec(str(dewarp_spec), (45, 31))
    background_dir = expand_path(spec["background_dir"]) if spec.get("background_dir") else None
    output.mkdir(parents=True, exist_ok=True)
    parts = output / "parts"
    parts.mkdir(exist_ok=True)
    tasks = []
    for split, count in spec["count"].items():
        count = min(count, limit) if limit else count
        geometries = by_split[split]
        records = pages[split]
        rng = np.random.default_rng(_seed(spec["name"], split, "pick"))
        picks = rng.choice(len(records), size=count, replace=count > len(records))
        order = rng.permutation(len(geometries))
        assigned: dict[str, list[int]] = {}
        for index in range(count):  # every capture gets items in turn
            assigned.setdefault(geometries[order[index % len(geometries)]], []).append(index)
        settings = {key: spec.get(key) for key in ("name", "photo_hw", "own_texture_prob",
                                                   "capture_background_prob", "shade",
                                                   "margin")}  # fmt: skip
        settings.update(split=split, raw_root=str(raw), background_dir=background_dir)
        for capture, indices in assigned.items():
            tasks.append({"key": f"{split}-{capture}", "settings": settings,
                          "output": str(output), "capture": capture, "indices": indices,
                          "items": [records[int(picks[i])] for i in indices]})  # fmt: skip
    pending = [t for t in tasks if not (parts / f"{t['key']}.jsonl").is_file()]
    started, done = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) if workers > 1 else _Inline() as pool:
        for result in pool.imap_unordered(run_capture, pending):
            lines = "".join(json.dumps(row) + "\n" for row in result["rows"])
            (parts / f"{result['key']}.jsonl").write_text(lines, encoding="utf-8")
            done += 1
            if done % 20 == 0 or done == len(pending):
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(json.dumps({"captures": done, "of": len(pending), "per_sec": round(rate, 2)}))
    rows = []
    for task in tasks:
        rows.extend(
            json.loads(line)
            for line in (parts / f"{task['key']}.jsonl").read_text(encoding="utf-8").splitlines()
        )
    rows.sort(key=lambda row: (row["split"], row["sample_id"]))
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = {
        "schema": BUILD_SCHEMA,
        "spec": spec,
        "spec_sha256": hashlib.sha256(canonical_json(spec)).hexdigest(),
        "captures": {split: len(by_split[split]) for split in by_split},
        "excluded_benchmark": str(exclude) if exclude else None,
        "geometry_source": SOURCE,
        "pages": described,
        "records": len(rows),
    }
    write_json(output / "build.json", record)
    return {"records": len(rows), "captures": record["captures"]}


class _Inline:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, function, items):
        return map(function, items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="items per split, for a trial")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.spec, args.output, workers=args.workers, limit=args.limit)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
