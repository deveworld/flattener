r"""Download CC0 surface textures from Poly Haven for composite training backgrounds.

Writes ``<output>/<asset>.jpg`` (the diffuse map at ``--resolution``, resized to at most
``--max-side`` pixels, JPEG q90) and ``<output>/manifest.json`` in the
``flattener-backgrounds-1`` schema read by ``flattener.dewarp.composite.BackgroundBank``.
The manifest records each asset’s source URL, authors and license.

Poly Haven publishes every asset under CC0 (https://polyhaven.com/license). The API
terms (https://github.com/Poly-Haven/Public-API/blob/master/ToS.md) allow any use,
commercial included, and ask for a user agent that names the tool, which this script
sends. Downloads run one at a time with a short pause.

Asset selection is deterministic: the texture list sorted by asset id, optionally
limited to ``--categories``, then ``--count`` assets evenly spaced over it. Files that
already exist are kept, so an interrupted run resumes.

    python tools/data/backgrounds/fetch_polyhaven.py \
        --output "$FLATTENER_CACHE/backgrounds/polyhaven-768"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

API = "https://api.polyhaven.com"
USER_AGENT = "flattener-background-fetch/1"
SCHEMA = "flattener-backgrounds-1"
LICENSE = "CC0-1.0"
LICENSE_URL = "https://polyhaven.com/license"


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _get_json(url: str) -> Any:
    return json.loads(_get(url))


def select_assets(assets: dict[str, Any], count: int, categories: list[str] | None) -> list[str]:
    """``count`` asset ids evenly spaced over the sorted list (filtered by category)."""
    names = sorted(
        name
        for name, info in assets.items()
        if not categories or set(categories) & set(info.get("categories", []))
    )
    if count >= len(names):
        return names
    return [names[round(i * (len(names) - 1) / max(count - 1, 1))] for i in range(count)]


def _store(data: bytes, path: Path, max_side: int) -> None:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        scale = max_side / max(image.size)
        if scale < 1:
            size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(size, Image.Resampling.LANCZOS)
        partial = path.with_suffix(".part")
        image.save(partial, format="JPEG", quality=90)
    partial.replace(path)


def fetch(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assets = _get_json(f"{API}/assets?t=textures")
    chosen = select_assets(assets, args.count, args.categories)
    entries = []
    for number, name in enumerate(chosen, 1):
        path = output / f"{name}.jpg"
        info = assets[name]
        files = _get_json(f"{API}/files/{name}")
        diffuse = (files.get("Diffuse") or {}).get(args.resolution, {}).get("jpg")
        if not diffuse:
            print(f"[{number}/{len(chosen)}] {name}: no {args.resolution} diffuse jpg, skipped")
            continue
        if not path.exists():
            data = _get(diffuse["url"])
            _store(data, path, args.max_side)
            time.sleep(args.delay)
        entries.append(
            {
                "file": path.name,
                "asset": name,
                "name": info.get("name", name),
                "page": f"https://polyhaven.com/a/{name}",
                "url": diffuse["url"],
                "license": LICENSE,
                "authors": sorted(info.get("authors", {})),
                "categories": info.get("categories", []),
            }
        )
        print(f"[{number}/{len(chosen)}] {name}")
    manifest = {
        "schema": SCHEMA,
        "source": "Poly Haven (https://polyhaven.com) through api.polyhaven.com",
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "tools/data/backgrounds/fetch_polyhaven.py",
        "settings": {
            "count": args.count,
            "categories": args.categories,
            "resolution": args.resolution,
            "max_side": args.max_side,
        },
        "files": entries,
    }
    partial = output / "manifest.json.part"
    partial.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    partial.replace(output / "manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="folder for the textures and manifest")
    parser.add_argument("--count", type=int, default=300, help="number of assets (default 300)")
    parser.add_argument(
        "--categories", nargs="*", default=None, help="keep assets in any of these categories"
    )
    parser.add_argument("--resolution", default="1k", help="API resolution key (default 1k)")
    parser.add_argument("--max-side", type=int, default=768, help="longest stored side (768)")
    parser.add_argument("--delay", type=float, default=0.2, help="pause between downloads (s)")
    args = parser.parse_args(argv)
    manifest = fetch(args)
    print(json.dumps({"files": len(manifest["files"]), "output": args.output}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
