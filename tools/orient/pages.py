"""Upright grayscale pages for OrientNet-1 data (used by ``build_data.py``).

Every function returns an upright page as (H, W) uint8 and the page box inside it,
``[x0, y0, x1, y1]`` in normalized image coordinates (flattener convention). Page
generators draw from the ``numpy`` generator they are given and nothing else, so a seed
fixes the pixels for a fixed Pillow/FreeType build.

* ``flatten_syntheticdoc``: a render sampled through its ground-truth grid with
  ``flattener.render``; the output aspect comes from the grid's arc lengths.
* ``crop_syntheticdoc``: the warped page cut from the render with a margin and turned so
  its up vector (mean of the grid's column vectors, bottom row to top row) points up.
* ``document_page``: an own page, either ``flattener.synthetic.documents`` (imported,
  unchanged) or a composed page of procedural Hangul/Latin text, formulas, tables and
  figures in the bundled Noto Sans KR; with ``form_prob`` and ``sparse_prob`` also a
  ruled form (``form_page``) or a nearly empty page (``sparse_page``).
* ``text_free_page``: blank, figure-only or ruled paper.
"""

from __future__ import annotations

import hashlib
import io
import math
from functools import lru_cache
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from flattener.contracts import RenderConfig, Status
from flattener.dewarp.coords import grid_to_map
from flattener.renderer import render
from flattener.scan.orient import SourceSampler, ViewGeometry, clockwise, quantize, to_gray
from flattener.synthetic import documents

FULL_BOX = [-1.0, -1.0, 1.0, 1.0]
# Width / height of common portrait papers (A/B series, 신국판, 4x6판, Letter) and two others.
ASPECTS = (0.707, 0.707, 0.676, 0.681, 0.773, 0.65, 0.75)
LATIN = "etaoinshrdlcumwfgypbvkjxqz"
# English letter frequencies (percent) in the order of LATIN.
_FREQUENCIES = (
    "12.7 9.1 8.2 7.5 7.0 6.7 6.3 6.1 6.0 4.3 4.0 2.8 2.8 2.4 "
    "2.4 2.2 2.0 2.0 1.9 1.5 1.0 0.8 0.2 0.2 0.1 0.1 "
)
LATIN_WEIGHTS = np.array([float(v) for v in _FREQUENCIES.split()])
LATIN_WEIGHTS = LATIN_WEIGHTS / LATIN_WEIGHTS.sum()


class Skip(Exception):
    """A source sample that gives no usable page; the reason is counted in the summary."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ----------------------------------------------------------------------------- SyntheticDoc


def _edge_points(grid: NDArray, size: tuple[int, int]) -> NDArray[np.float64]:
    height, width = size
    return (np.asarray(grid, dtype=np.float64) + 1.0) / 2.0 * np.array([width, height])


def flatten_syntheticdoc(
    rgb: NDArray[np.uint8], grid: NDArray, long_side: int = 1024, min_long_side: int = 768
) -> tuple[NDArray[np.uint8], list[float]]:
    """The flat page of a render through its (Gh, Gw, 2) grid. Its long side is the
    longest grid row or column in render pixels, clipped to [``min_long_side``,
    ``long_side``]; its aspect is the ratio of the mean row and column lengths."""
    if not np.isfinite(grid).all():
        raise Skip("invalid_grid")
    points = _edge_points(grid, rgb.shape[:2])
    rows = np.linalg.norm(np.diff(points, axis=1), axis=-1).sum(axis=1)
    cols = np.linalg.norm(np.diff(points, axis=0), axis=-1).sum(axis=0)
    native = float(max(rows.max(), cols.max()))
    if native < 300:
        raise Skip("small_page")
    aspect = float(rows.mean() / max(cols.mean(), 1e-9))  # width / height
    long = int(round(min(max(native, min_long_side), long_side)))
    size = (
        (max(2, round(long / aspect)), long)
        if aspect >= 1
        else (long, max(2, round(long * aspect)))
    )
    result = render(rgb, grid_to_map(grid, size), config=RenderConfig(on_error="return"))
    if result.status != Status.SUCCESS:
        raise Skip(str(result.reason_code))
    return to_gray(result.image), list(FULL_BOX)


def up_angle(grid: NDArray, size: tuple[int, int]) -> float:
    """Clockwise angle (degrees) of the page's up vector from screen up."""
    points = _edge_points(grid, size)
    up = points[0] - points[-1]  # per column: bottom node -> top node
    mean = up.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not norm > 1.0:
        raise Skip("degenerate_page")
    if (up @ (mean / norm)).min() <= 0:
        raise Skip("folded_page")
    return math.degrees(math.atan2(mean[0], -mean[1]))


def crop_syntheticdoc(
    rgb: NDArray[np.uint8], grid: NDArray, long_side: int = 1024, margin: float = 0.15
) -> tuple[NDArray[np.uint8], list[float], float]:
    """The warped page, turned upright, with ``margin`` of its extent on every side.

    The page is stored at its native density up to a page long side of ``long_side``.
    Returns the crop, the page box in it and the up angle that was undone.
    """
    if not np.isfinite(grid).all():
        raise Skip("invalid_grid")
    angle = up_angle(grid, rgb.shape[:2])
    upright = clockwise(-angle)  # source offset -> upright offset
    points = _edge_points(grid, rgb.shape[:2]).reshape(-1, 2)
    center = points.mean(axis=0)
    offsets = (points - center) @ upright.T
    low, high = offsets.min(axis=0), offsets.max(axis=0)
    extent = high - low
    if extent.max() < 300:
        raise Skip("small_page")
    scale = min(1.0, long_side / float(extent.max()))  # stored pixels per source pixel
    crop_low = low - margin * extent
    width, height = np.ceil((high - low + 2 * margin * extent) * scale).astype(int)
    back = upright.T / scale  # stored pixel -> source offset
    geometry = ViewGeometry(
        (float(height), float(width)), np.column_stack([back, center + upright.T @ crop_low])
    )
    sampler = SourceSampler(to_gray(rgb), geometry)
    xs, ys = np.arange(width) + 0.5, np.arange(height) + 0.5
    values = sampler(np.stack(np.meshgrid(xs, ys), -1), 1.0)
    covered = np.array([width, height]) / scale
    box = np.concatenate([(low - crop_low) / covered, (high - crop_low) / covered]) * 2 - 1
    return quantize(values), [float(v) for v in box], angle


# ----------------------------------------------------------------------------- own pages


@lru_cache(maxsize=1)
def _font_bytes() -> bytes:
    data = files("flattener").joinpath("assets", "fonts", "NotoSansKR[wght].ttf").read_bytes()
    if hashlib.sha256(data).hexdigest() != documents.FONT_SHA256:
        raise RuntimeError("bundled Noto Sans KR failed its sha256 check")
    return data


@lru_cache(maxsize=96)
def font(size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    face = ImageFont.truetype(
        io.BytesIO(_font_bytes()), size=size, layout_engine=ImageFont.Layout.BASIC
    )
    face.set_variation_by_axes([weight])
    return face


def hangul_word(rng: np.random.Generator) -> str:
    """1-4 random precomposed syllables (initial, vowel, optional final)."""
    syllables = []
    for _ in range(int(rng.integers(1, 5))):
        final = 0 if rng.random() < 0.55 else int(rng.integers(1, 28))
        syllables.append(
            chr(0xAC00 + (int(rng.integers(19)) * 21 + int(rng.integers(21))) * 28 + final)
        )
    return "".join(syllables)


def latin_word(rng: np.random.Generator) -> str:
    length = int(min(12, 1 + rng.geometric(0.28)))
    word = "".join(rng.choice(list(LATIN), size=length, p=LATIN_WEIGHTS))
    return word.capitalize() if rng.random() < 0.15 else word


def number_word(rng: np.random.Generator) -> str:
    value = rng.uniform(0, 1000)
    formats = (f"{int(value)}", f"{value:.2f}", f"({int(value) % 20 + 1})", f"{int(value) % 100}%")
    return formats[int(rng.integers(len(formats)))]


def words(rng: np.random.Generator, language: str):
    while True:
        roll = rng.random()
        if roll < 0.06:
            word = number_word(rng)
        elif language == "ko" or (language == "mixed" and roll < 0.6):
            word = hangul_word(rng)
        else:
            word = latin_word(rng)
        if rng.random() < 0.09:
            word += "." if rng.random() < 0.6 else ","
        yield word


def formula(rng: np.random.Generator) -> str:
    a, b, c = (round(float(v), 1) for v in rng.uniform(0.5, 9.5, size=3))
    names = "xyzabnt"
    x, y = names[int(rng.integers(len(names)))], names[int(rng.integers(len(names)))]
    choices = (
        f"{y} = {a}{x} + {b}",
        f"({x} + {a}) / {b} = {c}",
        f"f({x}) = {a}{x}^2 - {b}{x} + {c}",
        f"{x}_1 + {x}_2 + ... + {x}_n = {c}",
        f"sqrt({a}^2 + {b}^2) = {round(math.hypot(a, b), 2)}",
    )
    return choices[int(rng.integers(len(choices)))]


def _fit_line(face, source, width: float) -> str:
    line = ""
    for word in source:
        candidate = f"{line} {word}" if line else word
        if line and face.getlength(candidate) > width:
            return line
        line = candidate
    return line


def composed_page(rng: np.random.Generator, size: tuple[int, int]) -> NDArray[np.uint8]:
    """Header, title, one or two columns of paragraphs, formulas, tables, figures and a
    page number; body text 1.1-2.2% of the page height."""
    height, width = size
    paper = int(rng.integers(226, 256))
    image = Image.new("L", (width, height), paper)
    draw = ImageDraw.Draw(image)
    ink = int(rng.integers(0, 70))
    left = round(width * rng.uniform(0.06, 0.12))
    top = round(height * rng.uniform(0.05, 0.09))
    bottom = height - round(height * rng.uniform(0.06, 0.1))
    body_size = max(9, round(height * rng.uniform(0.011, 0.022)))
    weight = 400 if rng.random() < 0.7 else 500
    body = font(body_size, weight)
    small = font(max(8, round(body_size * 0.8)))
    leading = body_size * rng.uniform(1.3, 1.75)
    language = rng.choice(["ko", "en", "mixed"], p=[0.45, 0.35, 0.2])
    y = float(top)
    if rng.random() < 0.7:  # running head and rule
        head = _fit_line(small, words(rng, language), (width - 2 * left) * rng.uniform(0.2, 0.6))
        x = left if rng.random() < 0.5 else width - left - small.getlength(head)
        draw.text((x, y), head, font=small, fill=ink)
        y += small.size * 1.6
        if rng.random() < 0.5:
            draw.line((left, y, width - left, y), fill=ink, width=max(1, body_size // 12))
            y += small.size * 0.8
    if rng.random() < 0.4:  # title
        title_font = font(round(body_size * rng.uniform(1.4, 2.2)), 650)
        title = _fit_line(title_font, words(rng, language), (width - 2 * left) * 0.8)
        draw.text((left, y), title, font=title_font, fill=ink)
        y += title_font.size * 1.8
    columns = 2 if width > 640 and rng.random() < 0.35 else 1
    gutter = round(width * 0.04)
    column_width = (width - 2 * left - (columns - 1) * gutter) / columns
    body_top = y
    for column in range(columns):
        x0, y = left + column * (column_width + gutter), body_top
        while y + leading < bottom:
            roll = rng.random()
            if roll < 0.72:  # paragraph
                source = words(rng, language)
                indent = body_size * 1.5 if rng.random() < 0.5 else 0.0
                for line_index in range(int(rng.integers(2, 9))):
                    if y + leading >= bottom:
                        break
                    shift = indent if line_index == 0 else 0.0
                    draw.text(
                        (x0 + shift, y),
                        _fit_line(body, source, column_width - shift),
                        font=body,
                        fill=ink,
                    )
                    y += leading
                y += leading * rng.uniform(0.2, 0.8)
            elif roll < 0.82:  # formula line
                text = formula(rng)
                x = x0 + max(0.0, (column_width - body.getlength(text)) / 2)
                draw.text((x, y), text, font=body, fill=ink)
                y += leading * 1.6
            else:  # table or figure block with a caption
                block_h = int(min(bottom - y - leading * 2, column_width * rng.uniform(0.3, 0.7)))
                if block_h < 3 * leading:
                    y += leading
                    continue
                _block(
                    rng,
                    image,
                    draw,
                    (int(x0), int(y), int(x0 + column_width), int(y + block_h)),
                    ink,
                    body_size,
                )
                y += block_h + leading * 0.4
                caption = _fit_line(small, words(rng, language), column_width * 0.9)
                draw.text((x0, y), caption, font=small, fill=ink)
                y += leading * 1.4
    if rng.random() < 0.85:  # page number
        number = str(int(rng.integers(1, 400)))
        spots = (
            width / 2 - small.getlength(number) / 2,
            left,
            width - left - small.getlength(number),
        )
        draw.text(
            (spots[int(rng.integers(3))], bottom + small.size * 0.6), number, font=small, fill=ink
        )
    return np.array(image)


def _paper(rng: np.random.Generator, size: tuple[int, int]):
    height, width = size
    image = Image.new("L", (width, height), int(rng.integers(230, 256)))
    return image, ImageDraw.Draw(image), int(rng.integers(0, 60))


def _foot(rng, draw, size: tuple[int, int], face, ink: int, left: int, bottom: int) -> None:
    """A page count in the middle of the foot and, often, an office name at the right."""
    height, width = size
    count = f"{int(rng.integers(1, 20))} / {int(rng.integers(20, 40))}"
    draw.text(((width - face.getlength(count)) / 2, bottom - face.size), count, font=face, fill=ink)
    if rng.random() < 0.8:
        name = _fit_line(face, words(rng, "ko"), width * 0.25)
        draw.text(
            (width - left - face.getlength(name), bottom - face.size), name, font=face, fill=ink
        )


def form_page(rng: np.random.Generator, size: tuple[int, int]) -> NDArray[np.uint8]:
    """A ruled form such as a school or office record: a date and a title at the top, a
    table down the page with short labels in a narrow left column and dense Hangul text in
    the wide cells, and a page count and an office name at the foot."""
    height, width = size
    image, draw, ink = _paper(rng, size)
    left = round(width * rng.uniform(0.05, 0.09))
    right = width - left
    top = round(height * rng.uniform(0.04, 0.07))
    bottom = height - round(height * rng.uniform(0.05, 0.08))
    body = font(max(8, round(height * rng.uniform(0.009, 0.014))))
    small = font(max(8, round(body.size * 0.9)))
    rule = max(1, body.size // 10)
    year, month, day = (
        int(rng.integers(1990, 2031)),
        int(rng.integers(1, 13)),
        int(rng.integers(1, 29)),
    )
    date = f"{year}.{month:02d}.{day:02d}"
    draw.text((right - small.getlength(date), top), date, font=small, fill=ink)
    if rng.random() < 0.6:
        draw.text(
            (left, top), _fit_line(small, words(rng, "ko"), width * 0.4), font=small, fill=ink
        )
    y = top + small.size * 2.2
    title_font = font(round(body.size * rng.uniform(1.3, 1.8)), 600)
    title = _fit_line(title_font, words(rng, "ko"), width * 0.45)
    draw.text(((width - title_font.getlength(title)) / 2, y), title, font=title_font, fill=ink)
    y += title_font.size * 2.0
    table_bottom = bottom - small.size * 3
    label_width = round((right - left) * rng.uniform(0.1, 0.2))
    draw.rectangle((left, y, right, table_bottom), outline=ink, width=rule)
    draw.line((left + label_width, y, left + label_width, table_bottom), fill=ink, width=rule)
    leading = body.size * rng.uniform(1.35, 1.6)
    row_top = y
    while row_top < table_bottom - 2 * leading:
        row_bottom = min(table_bottom, row_top + int(rng.integers(2, 14)) * leading + leading)
        label = _fit_line(body, words(rng, "ko"), label_width - 6)
        draw.text((left + 4, row_top + leading * 0.4), label, font=body, fill=ink)
        line_y = row_top + leading * 0.4
        while line_y + leading < row_bottom:
            text = _fit_line(body, words(rng, "ko"), right - left - label_width - 10)
            draw.text((left + label_width + 5, line_y), text, font=body, fill=ink)
            line_y += leading
        if row_bottom < table_bottom:
            draw.line((left, row_bottom, right, row_bottom), fill=ink, width=rule)
        row_top = row_bottom
    _foot(rng, draw, size, small, ink, left, bottom)
    return np.array(image)


def sparse_page(rng: np.random.Generator, size: tuple[int, int]) -> NDArray[np.uint8]:
    """A nearly empty page: a running head or a date, one short spaced-out phrase (an
    end-of-text note or a section title) and a foot; the rest is blank paper."""
    height, width = size
    image, draw, ink = _paper(rng, size)
    left = round(width * rng.uniform(0.05, 0.1))
    top = round(height * rng.uniform(0.04, 0.08))
    bottom = height - round(height * rng.uniform(0.05, 0.09))
    small = font(max(8, round(height * rng.uniform(0.009, 0.014))))
    if rng.random() < 0.8:
        head = _fit_line(small, words(rng, "ko"), width * rng.uniform(0.15, 0.4))
        x = left if rng.random() < 0.5 else width - left - small.getlength(head)
        draw.text((x, top), head, font=small, fill=ink)
        if rng.random() < 0.3:
            y = top + small.size * 1.6
            draw.line((left, y, width - left, y), fill=ink, width=max(1, small.size // 10))
    big = font(round(height * rng.uniform(0.018, 0.04)), 500)
    phrase = " ".join(
        hangul_word(rng)[: int(rng.integers(2, 5))] for _ in range(int(rng.integers(1, 3)))
    )
    if rng.random() < 0.6:
        phrase = " ".join(phrase.replace(" ", ""))  # spaced syllables, as in a closing note
    y = height * rng.uniform(0.2, 0.7)
    draw.text(((width - big.getlength(phrase)) / 2, y), phrase, font=big, fill=ink)
    if rng.random() < 0.85:
        _foot(rng, draw, size, small, ink, left, bottom)
    return np.array(image)


def _block(
    rng, image: Image.Image, draw: ImageDraw.ImageDraw, box, ink: int, text_size: int
) -> None:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if rng.random() < 0.5 and height >= 128 and width >= 128:
        # A flattener.synthetic.documents page (title, prose, table, shapes) as a figure.
        seed = int(rng.integers(0, 2**63))
        doc_size = (int(min(max(height, 128), 2048)), int(min(max(width, 128), 2048)))
        try:
            tile = documents.generate_document(f"orient-tile-{seed}", seed, doc_size).image
        except ValueError:
            tile = None
        if tile is not None:
            image.paste(Image.fromarray(tile).convert("L").resize((width, height)), (x0, y0))
            return
    # A ruled table of numbers.
    rows, cols = int(rng.integers(3, 8)), int(rng.integers(2, 6))
    face = font(max(8, min(int(height / rows * 0.45), round(text_size * 1.1))))
    line = max(1, height // 150)
    for r in range(rows + 1):
        y = y0 + round(r * height / rows)
        draw.line((x0, y, x1, y), fill=ink, width=line)
    for c in range(cols + 1):
        x = x0 + round(c * width / cols)
        draw.line((x, y0, x, y1), fill=ink, width=line)
    for r in range(rows):
        for c in range(cols):
            text = number_word(rng) if r or rng.random() < 0.5 else latin_word(rng)
            draw.text(
                (x0 + c * width / cols + 3, y0 + r * height / rows + 2), text, font=face, fill=ink
            )


def page_size(
    rng: np.random.Generator, long_side: int, landscape_prob: float = 0.12
) -> tuple[int, int]:
    """(height, width): a paper aspect, long side in [768, long_side]."""
    aspect = float(rng.choice(ASPECTS)) * rng.uniform(0.97, 1.03)
    long = int(rng.integers(min(768, long_side), long_side + 1))
    short = max(128, round(long * aspect))
    return (short, long) if rng.random() < landscape_prob else (long, short)


def document_page(
    page_id: str,
    rng: np.random.Generator,
    long_side: int = 1024,
    document_prob: float = 0.35,
    form_prob: float = 0.0,
    sparse_prob: float = 0.0,
) -> tuple[NDArray[np.uint8], list[float], str]:
    """An own text page; returns the page, its box and its layout name. Without
    ``form_prob`` and ``sparse_prob`` the draws are those of earlier builds."""
    size = page_size(rng, long_side)
    if form_prob or sparse_prob:
        roll = rng.random()
        if roll < form_prob:
            return form_page(rng, size), list(FULL_BOX), "form"
        if roll < form_prob + sparse_prob:
            return sparse_page(rng, size), list(FULL_BOX), "sparse"
    if rng.random() < document_prob:
        seed = int(rng.integers(0, 2**63))
        try:
            image = documents.generate_document(page_id, seed, size).image
            return to_gray(image), list(FULL_BOX), "documents"
        except ValueError:
            pass
    return composed_page(rng, size), list(FULL_BOX), "composed"


def text_free_page(
    rng: np.random.Generator, long_side: int = 1024
) -> tuple[NDArray[np.uint8], list[float], str]:
    """Blank, figure-only or ruled paper (trained toward a uniform target)."""
    height, width = page_size(rng, long_side)
    base = rng.uniform(170, 252)
    shading = rng.uniform(-12, 12, size=(3, 3))
    shading = np.asarray(
        Image.fromarray(shading.astype(np.float32)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    )
    page = base + shading + rng.normal(0, rng.uniform(0.5, 3), size=(height, width))
    image = Image.fromarray(np.clip(page, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    layout = str(rng.choice(["blank", "figure", "ruled"], p=[0.35, 0.45, 0.2]))
    if layout == "figure":
        for _ in range(int(rng.integers(1, 5))):
            w, h = int(width * rng.uniform(0.15, 0.8)), int(height * rng.uniform(0.1, 0.5))
            x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
            kind = rng.random()
            if kind < 0.4:  # smooth photo-like texture
                cells = rng.uniform(0, 255, size=(int(rng.integers(3, 9)), int(rng.integers(3, 9))))
                patch = Image.fromarray(cells.astype(np.float32)).resize(
                    (w, h), Image.Resampling.BICUBIC
                )
                image.paste(
                    Image.fromarray(np.clip(np.asarray(patch), 0, 255).astype(np.uint8)), (x, y)
                )
            elif kind < 0.7:
                draw.rectangle((x, y, x + w, y + h), fill=int(rng.integers(30, 230)))
            else:
                draw.ellipse(
                    (x, y, x + w, y + h),
                    outline=int(rng.integers(0, 120)),
                    width=int(rng.integers(1, 6)),
                    fill=int(rng.integers(60, 250)),
                )
    elif layout == "ruled":
        tone = int(rng.integers(90, 200))
        step = int(rng.integers(18, 60))
        if rng.random() < 0.5:
            for y in range(step, height, step):
                draw.line((0, y, width, y), fill=tone, width=1)
        if rng.random() < 0.5:
            for x in range(step, width, step):
                draw.line((x, 0, x, height), fill=tone, width=1)
        inset = int(rng.integers(10, 60))
        draw.rectangle((inset, inset, width - inset, height - inset), outline=tone, width=2)
    return np.array(image), list(FULL_BOX), layout


def resized_page(gray: NDArray[np.uint8], long_side: int) -> NDArray[np.uint8]:
    """``gray`` scaled down (Lanczos) so its long side is at most ``long_side``."""
    height, width = gray.shape
    scale = long_side / max(height, width)
    if scale >= 1:
        return gray
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.array(Image.fromarray(gray).resize(size, Image.Resampling.LANCZOS))
