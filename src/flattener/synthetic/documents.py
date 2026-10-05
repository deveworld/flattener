"""Locally authored pages with a bundled, pinned font (Noto Sans KR, used for its Latin glyphs).

Coordinates and ``output_size`` follow the renderer's (height, width) order.
The same document ID, seed, size, generator version, and locked Pillow/FreeType
environment produce the same pixels. No system fonts, network calls, copied
documents, external photographs, or global random state are used.
"""

import hashlib
import io
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, features
from PIL import __version__ as pillow_version

DOCUMENT_VERSION = "2.0"  # 2.0: English text only
FONT_SHA256 = "194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252"
FONT_COMMIT = "b38c5c93af322c45f633e17ac440ec1e6c94d489"
FONT_SOURCE = (
    f"https://raw.githubusercontent.com/google/fonts/{FONT_COMMIT}/ofl/notosanskr/"
    "NotoSansKR%5Bwght%5D.ttf"
)

# Original sentences written for this generator, licensed with the code.
# The list is deliberately finite and auditable; these are not book excerpts.
ENGLISH_SENTENCES = (
    "A quiet morning leaves room for a new idea.",
    "We record each change before the next trial.",
    "The paper holds a small map of our observations.",
    "Three careful measurements form a useful pattern.",
    "Draw a clear line and compare the spaces around it.",
    "A simple question can lead to another experiment.",
    "Light moves slowly across the garden table.",
    "Each page brings words and shapes together.",
    "Small notes from the morning gather the thoughts of the day.",
    "Today's observations were written on the paper on the desk.",
    "The shadow of the tree moves slowly with the wind.",
    "We draw lines of different shapes side by side.",
    "A new question becomes the starting point of the next trial.",
    "Read the numbers in the table and compare the changes calmly.",
    "Record the light gathered in the small garden in order.",
    "One sheet holds both writing and figures.",
)
TITLES = (
    "Field Notes",
    "A Small Trial",
    "Today's Page",
    "A Map of Ideas",
    "Garden Records",
    "New Questions",
)


@dataclass(slots=True)
class Document:
    """An authored flat page and JSON-safe provenance for its exact pixels."""

    image: NDArray[np.uint8]
    text: list[str]
    metadata: dict[str, Any]


@lru_cache(maxsize=1)
def _font_bytes() -> bytes:
    asset = files("flattener").joinpath("assets", "fonts", "NotoSansKR[wght].ttf")
    data = asset.read_bytes()
    if hashlib.sha256(data).hexdigest() != FONT_SHA256:
        raise RuntimeError("Bundled Noto Sans KR font failed its SHA-256 integrity check")
    return data


@lru_cache(maxsize=32)
def _font(size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    # BASIC avoids environment-dependent optional shaping engines. All text is
    # simple Latin, so complex shaping is not required.
    font = ImageFont.truetype(
        io.BytesIO(_font_bytes()), size=size, layout_engine=ImageFont.Layout.BASIC
    )
    font.set_variation_by_axes([weight])
    return font


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    """Break text at spaces, and a word too long for a line at characters, without
    dropping text."""
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        candidate = f"{line} {word}" if line else word
        if line and font.getlength(candidate) > width:
            lines.append(line)
            line = ""
        candidate = f"{line} {word}" if line else word
        if font.getlength(candidate) <= width:
            line = candidate
        else:
            # A long word or identifier can exceed one line.
            for char in word:
                if line and font.getlength(line + char) > width:
                    lines.append(line)
                    line = ""
                line += char
    if line:
        lines.append(line)
    return lines


def generate_document(
    document_id: str, seed: int, output_size: tuple[int, int] = (256, 192)
) -> Document:
    """Generate a flat RGB page with English text, a table, and shapes.

    ``output_size`` is (height, width), each dimension in [128, 2048]. The ID is
    a document-group identifier, not a path. It contributes to a SHA-256-derived
    local generator state, so distinct IDs remain independent at a common seed.
    Generated prose, numbers, and vectors use Apache-2.0; the bundled font has
    its separate OFL-1.1 license.
    """
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    if (
        not isinstance(output_size, tuple)
        or len(output_size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in output_size)
        or any(v < 128 or v > 2048 for v in output_size)
    ):
        raise ValueError("output_size must be a (height, width) tuple with sides in [128, 2048]")
    height, width = output_size
    identity = document_id.encode("utf-8") + b"\0" + seed.to_bytes(8, "little")
    rng_seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
    rng = np.random.default_rng(rng_seed)
    supersample = 3
    h, w = height * supersample, width * supersample
    image = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    unit = min(w, h)
    margin = round(unit * float(rng.uniform(0.07, 0.095)))
    available = w - 2 * margin
    ink = tuple(int(c) for c in rng.integers(15, 50, size=3))
    accent = ((34, 68, 99), (58, 78, 60), (98, 52, 44))[int(rng.integers(3))]
    body_size = max(12, round(unit * 0.038))
    body = _font(body_size)
    small = _font(max(10, round(unit * 0.03)))
    title_font = _font(round(unit * 0.067), 650)
    text: list[str] = []
    text_boxes: list[dict[str, Any]] = []

    def write(x: float, y: float, value: str, font: ImageFont.FreeTypeFont, color=ink):
        draw.text((round(x), round(y)), value, font=font, fill=color, anchor="lt")
        bounds = draw.textbbox((round(x), round(y)), value, font=font, anchor="lt")
        text.append(value)
        text_boxes.append(
            {"text": value, "bounds_xyxy": [round(v / supersample, 4) for v in bounds]}
        )

    title = TITLES[int(rng.integers(len(TITLES)))]
    serial = int(rng.integers(100, 1000))
    write(margin, h * 0.065, title, title_font, accent)
    write(margin, h * 0.15, f"FIELD NOTES / {serial}", small)
    rule_y = round(h * 0.205)
    draw.line((margin, rule_y, w - margin, rule_y), fill=accent, width=max(2, round(unit / 220)))

    # Two sentences, or three on a tall page, drawn without repeats.
    # Each paragraph fits in its own measured region; text is never cropped.
    count = 3 if height / width > 1.15 else 2
    picks = rng.choice(len(ENGLISH_SENTENCES), size=count, replace=False)
    paragraphs = [ENGLISH_SENTENCES[int(index)] for index in picks]
    paragraph_y = h * 0.24
    paragraph_end = h * 0.545
    paragraph_font = body
    while True:
        wrapped = [_wrap(p, paragraph_font, available) for p in paragraphs]
        step = paragraph_font.size * 1.38
        needed = sum(len(lines) * step for lines in wrapped) + step * 0.45 * (len(wrapped) - 1)
        if needed <= paragraph_end - paragraph_y:
            break
        paragraph_font = _font(paragraph_font.size - 1)
        if paragraph_font.size < 6:
            raise ValueError("output_size has an aspect ratio too extreme for document layout")
    for lines in wrapped:
        for line in lines:
            write(margin, paragraph_y, line, paragraph_font)
            paragraph_y += step
        paragraph_y += step * 0.45

    # Table rows and measurements are newly generated numeric content.
    x0, x1 = margin, w - margin
    table_y = round(h * float(rng.uniform(0.57, 0.595)))
    row_h = round(h * 0.059)
    columns = [x0, round(x0 + available * 0.44), round(x0 + available * 0.72), x1]
    draw.rectangle((x0, table_y, x1, table_y + row_h), fill=(229, 234, 237))
    rows = [
        ("Item", "Count", "Value"),
        ("Sample A", str(int(rng.integers(2, 35))), f"{rng.uniform(1, 9):.2f}"),
        ("Sample B", str(int(rng.integers(2, 35))), f"{rng.uniform(1, 9):.2f}"),
    ]
    table_font = small
    while max(table_font.getlength(cell) for row in rows for cell in row) > available * 0.4:
        table_font = _font(table_font.size - 1)
    for row_index, row in enumerate(rows):
        y = table_y + row_index * row_h
        for index, cell in enumerate(row):
            write(columns[index] + unit * 0.015, y + row_h * 0.23, cell, table_font)
    for index in range(4):
        y = table_y + row_h * index
        draw.line((x0, y, x1, y), fill=accent, width=max(1, round(unit / 380)))
    for x in columns:
        draw.line(
            (x, table_y, x, table_y + 3 * row_h), fill=accent, width=max(1, round(unit / 380))
        )

    # Independent vector marks expose bending, blur, and interpolation errors.
    shapes_y = round(h * 0.82)
    r = round(min(unit * 0.026, h * 0.024))
    centers = [round(margin + available * v) for v in (0.09, 0.32, 0.56)]
    line_width = max(2, round(unit / 210))
    draw.ellipse(
        (centers[0] - r, shapes_y - r, centers[0] + r, shapes_y + r),
        outline=accent,
        width=line_width,
    )
    draw.rectangle(
        (centers[1] - r, shapes_y - r, centers[1] + r, shapes_y + r),
        outline=accent,
        width=line_width,
    )
    draw.polygon(
        [
            (centers[2], shapes_y - r),
            (centers[2] + r, shapes_y + r),
            (centers[2] - r, shapes_y + r),
        ],
        fill=accent,
    )
    write(w - margin - small.getlength("01 / 01"), h * 0.90, "01 / 01", small)
    result = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS)).copy()
    sha256 = hashlib.sha256(result.tobytes(order="C")).hexdigest()
    metadata = {
        "document_id": document_id,
        "document_group": document_id,
        "generator": "flattener.synthetic.documents",
        "generator_version": DOCUMENT_VERSION,
        "seed": seed,
        "derived_rng_seed": rng_seed,
        "output_size": [height, width],
        "original_aspect_ratio": width / height,
        "source_sha256": sha256,
        "source_hash_encoding": "RGB uint8 C-order raw pixels; size recorded separately",
        "settings": {
            "supersample": supersample,
            "resampling": "lanczos",
            "layout": "notes",
        },
        "text": {
            "source": "Original text authored in flattener.synthetic.documents",
            "license": "Apache-2.0",
            "normalization": "NFC; literal spaces preserved per drawn line",
            "lines": text.copy(),
            "boxes": text_boxes,
        },
        "font": {
            "family": "Noto Sans KR",
            "version": "2.004",
            "sha256": FONT_SHA256,
            "source": FONT_SOURCE,
            "commit": FONT_COMMIT,
            "license": "OFL-1.1",
            "license_file": "flattener/assets/fonts/OFL.txt",
            "modified": False,
            "weights": [400, 650],
        },
        "background": {"source": "programmatically generated solid white", "license": "Apache-2.0"},
        "shapes": {
            "source": "programmatically generated lines and geometric shapes",
            "license": "Apache-2.0",
        },
        "external_photographs": False,
        "environment": {"pillow": pillow_version, "freetype": features.version_module("freetype2")},
    }
    return Document(image=result, text=text, metadata=metadata)
