"""Decode photos to raw RGBA for the web scanner's Node trace harness (web/scripts/trace.ts).

Applies EXIF orientation and converts embedded color profiles to sRGB, as the browser
does when it decodes a photo, and writes ``<stem>.rgba`` with a ``photos.json`` index.

    uv run python tools/web/decode_photos.py OUT_DIR PHOTO [PHOTO ...]
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import ExifTags, Image, ImageCms, ImageOps

MAX_PIXELS = 20_000_000  # the web scanner's input limit


def decode(path: Path) -> tuple[Image.Image, dict]:
    image = Image.open(path)
    exif = image.getexif()
    tags = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
    details = exif.get_ifd(0x8769)
    info = {
        "orientation": tags.get("Orientation"),
        "model": tags.get("Model"),
        "focal_35mm": details.get(0xA405),
    }
    image = ImageOps.exif_transpose(image)
    profile = image.info.get("icc_profile")
    if profile:
        source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        image = ImageCms.profileToProfile(
            image, source, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
    return image.convert("RGBA"), info


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip())
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    photos = []
    for name in argv[2:]:
        path = Path(name)
        image, info = decode(path)
        if image.width * image.height > MAX_PIXELS:
            print(f"{path}: above {MAX_PIXELS} pixels, skipped", file=sys.stderr)
            continue
        (out / f"{path.stem}.rgba").write_bytes(image.tobytes())
        photos.append(
            {
                "stem": path.stem,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": image.width,
                "height": image.height,
                **info,
            }
        )
    (out / "photos.json").write_text(json.dumps(photos, indent=1))
    print(f"{len(photos)} photos in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
