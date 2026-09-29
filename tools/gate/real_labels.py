"""Page masks for real training photos (KR-Train): scanner proposals, review, manifest.

``python tools/gate/real_labels.py propose ROOT [--bundle B]``
    Scans every photo under ``ROOT/incoming/`` (one page, no split: an open-book spread
    stays one region) and writes ``ROOT/proposals/<id>.json`` with the grid outline as a
    polygon in EXIF-applied photo pixels, or none when the scanner returned the original.
    ``<id>`` is ``<folder>-<file stem>``.
``python tools/gate/real_labels.py sheets ROOT``
    Review sheets in ``ROOT/review/``: each photo with its polygon (yellow) and a grid of
    tenths of the width and height, numbered, for reading corner positions.
``python tools/gate/real_labels.py manifest ROOT``
    ``ROOT/manifest.json`` for the ``labeled_photos`` builder source. A file
    ``ROOT/fixes/<id>.json`` overrides the proposal: ``{"quads": [[[x, y] x4], ...]}`` or
    ``{"fractions": [[[fx, fy] x4], ...]}`` (tenths grid, 0-1), ``{"none": true}`` for a
    photo without page, ``{"skip": true}`` to leave the photo out. A photo with neither a
    proposal polygon nor a fix is left out and listed. The split is by the photo's sha256
    (``val`` for about 15 %).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
SCHEMA = "flattener-gate-photos-1"


def photos(root: Path) -> list[tuple[str, str, Path]]:
    """(id, folder, path) of every photo under ``root/incoming``."""
    found = []
    for path in sorted((root / "incoming").rglob("*")):
        if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES:
            folder = path.parent.relative_to(root / "incoming").as_posix().replace("/", "_")
            folder = folder if folder != "." else "root"
            found.append((f"{folder}-{path.stem}", folder, path))
    return found


def upright(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def propose(root: Path, bundle: str) -> None:
    from flattener.scan import locate
    from flattener.scan.pipeline import Scanner, ScanOptions

    scanner = Scanner.load(bundle)
    out = root / "proposals"
    out.mkdir(exist_ok=True)
    for ident, _, path in photos(root):
        target = out / f"{ident}.json"
        if target.is_file():
            continue
        result = scanner.scan(str(path), ScanOptions(pages=1))
        record = {"photo": path.relative_to(root).as_posix(), "outcome": str(result.outcome)}
        grid = None if result.plan is None else result.plan.grid
        if grid is not None:
            height, width = result.plan.source.size
            polygon = (locate.outline_polygon(grid) + 1.0) / 2.0 * np.array([width, height])
            record.update(size=[height, width], polygons=[polygon.round(1).tolist()])
        else:
            record.update(polygons=[], reason=result.reason_code)
        target.write_text(json.dumps(record) + "\n", encoding="utf-8")
        print(ident, record["outcome"], len(record["polygons"]), flush=True)


def _polygons(root: Path, ident: str, size: tuple[int, int]) -> list | None:
    """The polygons (pixels) of a photo from its fix or proposal; None to leave it out."""
    height, width = size
    fix_path = root / "fixes" / f"{ident}.json"
    if fix_path.is_file():
        fix = json.loads(fix_path.read_text(encoding="utf-8"))
        if fix.get("skip"):
            return None
        if fix.get("none"):
            return []
        if "fractions" in fix:
            return [[[fx * width, fy * height] for fx, fy in quad] for quad in fix["fractions"]]
        return fix["quads"]
    proposal_path = root / "proposals" / f"{ident}.json"
    if not proposal_path.is_file():
        return None
    polygons = json.loads(proposal_path.read_text(encoding="utf-8"))["polygons"]
    return polygons or None


def sheets(root: Path, per_sheet: int = 12) -> None:
    out = root / "review"
    out.mkdir(exist_ok=True)
    tiles = []
    for ident, _, path in photos(root):
        image = upright(path)
        width, height = image.size
        polygons = _polygons(root, ident, (height, width)) or []
        image.thumbnail((520, 520))
        scale = image.size[0] / width
        draw = ImageDraw.Draw(image)
        for k in range(1, 10):
            x, y = k * image.size[0] / 10, k * image.size[1] / 10
            draw.line((x, 0, x, image.size[1]), fill=(0, 200, 255), width=1)
            draw.line((0, y, image.size[0], y), fill=(0, 200, 255), width=1)
            draw.text((x + 2, 2), str(k), fill=(0, 200, 255))
            draw.text((2, y + 2), str(k), fill=(0, 200, 255))
        for polygon in polygons:
            points = [(x * scale, y * scale) for x, y in polygon]
            draw.line(points + points[:1], fill=(255, 230, 0), width=3)
        label = Image.new("RGB", (image.size[0], 18), "black")
        ImageDraw.Draw(label).text((4, 3), ident, fill="white")
        tile = Image.new("RGB", (image.size[0], image.size[1] + 18), "black")
        tile.paste(label, (0, 0))
        tile.paste(image, (0, 18))
        tiles.append(tile)
    for start in range(0, len(tiles), per_sheet):
        group = tiles[start : start + per_sheet]
        sheet = Image.new("RGB", (4 * 530, 3 * 550), "gray")
        for index, tile in enumerate(group):
            sheet.paste(tile, ((index % 4) * 530, (index // 4) * 550))
        sheet.save(out / f"sheet_{start // per_sheet:02d}.jpg", quality=85)


def manifest(root: Path, val_share: float = 0.15) -> None:
    rows, missing = [], []
    for ident, folder, path in photos(root):
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
        polygons = _polygons(root, ident, (height, width))
        if polygons is None:
            missing.append(ident)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        split = "val" if int(digest[:8], 16) / 0xFFFFFFFF < val_share else "train"
        rows.append(
            {
                "id": ident,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size": [height, width],
                "polygons": polygons,
                "split": split,
                "group": folder,
            }
        )
    payload = {"schema": SCHEMA, "license": "own", "photos": rows, "left_out": missing}
    (root / "manifest.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(json.dumps({"photos": len(rows), "left_out": missing}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("propose", "sheets", "manifest"))
    parser.add_argument("root")
    parser.add_argument(
        "--bundle", default=str(Path.home() / ".cache/flattener/scan-weights/bundle-v2.json")
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "propose":
        propose(root, args.bundle)
    elif args.command == "sheets":
        sheets(root)
    else:
        manifest(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
