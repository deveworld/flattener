"""Whole-photo training scenes for the page gate, and the mask of the paper it should find.

``scene`` places a page (a render with its grid, or a flat page) in a photo-sized frame
on a CC0 or procedural background, through ``dewarp.composite``'s placement: tilt,
quarter turns, perspective and area. A close-up then zooms in about a point of the page,
so that the page runs out of the frame on some or all sides. With ``neighbor`` the facing
page of an open book is drawn along the page's left or right edge; it counts as page,
since the scanner flattens a spread whole and splits it afterwards.

Desk objects are drawn into the background and are never page: laptops and keyboards,
screens (dark or light), book covers, mugs and cables. Thumbs over the page edge are cut
out of the mask. Page, objects and background share one illumination field, cast shadow
and white balance; the finished scene may be blurred, noisy and JPEG-compressed.
Without a page the scene is a negative (empty mask).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

from flattener.dewarp import composite as cp

ASPECTS = {"3:4": (3, 4), "4:3": (4, 3), "9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1)}
OBJECTS = ("laptop", "keyboard", "screen", "cover", "mug", "cable")


@dataclass(frozen=True)
class SceneConfig:
    """Scene draws. ``aspects`` and ``objects`` are weights by name; ``object_count``
    weights 0, 1, 2, ... objects. ``composite`` overrides ``CompositeConfig`` fields."""

    long_side: int = 384
    aspects: dict[str, float] = field(
        default_factory=lambda: {"3:4": 0.45, "4:3": 0.3, "9:16": 0.08, "16:9": 0.07, "1:1": 0.1}
    )
    closeup_prob: float = 0.3
    closeup_zoom: tuple[float, float] = (1.2, 3.5)
    neighbor_prob: float = 0.35
    object_count: tuple[float, ...] = (0.35, 0.3, 0.2, 0.15)
    objects: dict[str, float] = field(
        default_factory=lambda: {
            "laptop": 0.3, "keyboard": 0.1, "screen": 0.2, "cover": 0.2, "mug": 0.1, "cable": 0.1
        }
    )  # fmt: skip
    thumb_prob: float = 0.35
    book_prob: float = 0.2
    book_prob_spread: float = 0.7
    blur_prob: float = 0.3
    noise: float = 4.0
    jpeg_quality: tuple[int, int] = (55, 95)
    composite: dict[str, Any] = field(
        default_factory=lambda: {
            "prob": 1.0,
            "area": (0.05, 0.9),
            "overhang": 0.08,
            "perspective": 0.2,
            "angle_std": 10.0,
            "wide_angle_prob": 0.3,
            "wide_angle": 90.0,
            "turns": (1.0, 1.0, 1.0, 1.0),
            "texture_prob": 0.7,
            "panel_prob": 0.5,
            "illumination": 0.4,
            "neighbor_width": (0.4, 1.05),
            "fold_shadow": (0.05, 0.5),
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SceneConfig:
        data = dict(data or {})
        for name in ("closeup_zoom", "object_count", "jpeg_quality"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


def _choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    values = np.asarray([weights[name] for name in names], dtype=np.float64)
    return names[int(rng.choice(len(names), p=values / values.sum()))]


def frame_size(rng: np.random.Generator, config: SceneConfig) -> tuple[int, int]:
    """(height, width) of a frame of a drawn aspect with the configured long side."""
    wide, high = ASPECTS[_choice(rng, config.aspects)]
    scale = config.long_side / max(wide, high)
    return round(high * scale), round(wide * scale)


# --------------------------------------------------------------------------- objects


def _noise(rng: np.random.Generator, size: tuple[int, int], base, amount: float) -> Image.Image:
    width, height = size
    values = np.asarray(base, np.float32) + rng.normal(0, amount, (height, width, 1))
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8), "RGB")


def _gray(rng: np.random.Generator, low: float, high: float, tint: float = 6.0):
    level = rng.uniform(low, high)
    return tuple(int(np.clip(level + rng.uniform(-tint, tint), 0, 255)) for _ in range(3))


def _keys(rng, draw: ImageDraw.ImageDraw, box, rows: int, cols: int, key, glyph) -> None:
    left, top, right, bottom = box
    pitch_x, pitch_y = (right - left) / cols, (bottom - top) / rows
    gap = rng.uniform(0.1, 0.25)
    for row in range(rows):
        count = cols if row else cols + 1
        pitch = (right - left) / count
        for col in range(count):
            span = 1
            if row == rows - 1 and col == count // 2 - 2:
                span = 5  # space bar
            elif row == rows - 1 and count // 2 - 2 < col < count // 2 + 3:
                continue
            x0 = left + col * pitch + gap * pitch / 2
            y0 = top + row * pitch_y + gap * pitch_y / 2
            x1 = left + (col + span) * pitch - gap * pitch / 2
            y1 = top + (row + 1) * pitch_y - gap * pitch_y / 2
            if row == 0:
                y1 = y0 + 0.6 * (y1 - y0)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=0.15 * pitch_x, fill=key)
            if rng.random() < 0.8 and span == 1:
                cx, cy = x0 + 0.3 * (x1 - x0), y0 + 0.35 * (y1 - y0)
                size = 0.18 * (x1 - x0)
                if rng.random() < 0.5:
                    draw.rectangle((cx, cy, cx + size, cy + 0.15 * size + 1), fill=glyph)
                else:
                    draw.ellipse((cx, cy, cx + size, cy + size), outline=glyph, width=2)


def laptop(rng: np.random.Generator, with_keys_only: bool = False) -> Image.Image:
    """A laptop deck (keyboard and touchpad) or, with ``with_keys_only``, a keyboard."""
    width = 1000
    height = int(width * (rng.uniform(0.28, 0.38) if with_keys_only else rng.uniform(0.6, 0.7)))
    silver = rng.random() < 0.55
    body = _gray(rng, 175, 225) if silver else _gray(rng, 25, 110)
    image = _noise(rng, (width, height), body, 3.0).convert("RGBA")
    draw = ImageDraw.Draw(image)
    dark_keys = rng.random() < 0.7
    key = _gray(rng, 15, 50, 3) if dark_keys else _gray(rng, 190, 235, 3)
    glyph = _gray(rng, 190, 240, 2) if dark_keys else _gray(rng, 20, 70, 2)
    margin = 0.05 * width
    keys_bottom = height - margin if with_keys_only else 0.55 * height
    _keys(rng, draw, (margin, margin, width - margin, keys_bottom), 6, int(rng.integers(13, 16)),
          key, glyph)  # fmt: skip
    if not with_keys_only:
        pad = _gray(rng, *(sorted((body[0] - 25, body[0] + 12))))
        pad_w = rng.uniform(0.3, 0.45) * width
        draw.rounded_rectangle(
            ((width - pad_w) / 2, 0.6 * height, (width + pad_w) / 2, 0.93 * height),
            radius=12, fill=pad,
        )  # fmt: skip
        if rng.random() < 0.6:  # stickers
            for _ in range(int(rng.integers(1, 3))):
                x = rng.uniform(0.62, 0.9) * width
                y = rng.uniform(0.62, 0.85) * height
                colour = tuple(int(v) for v in rng.integers(0, 255, 3))
                draw.rectangle((x, y, x + 40, y + 45), fill=colour)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=30, fill=255)
    image.putalpha(mask)
    return image


def screen(rng: np.random.Generator) -> Image.Image:
    """A display in its bezel, dark or light, full of text lines and interface bars."""
    width = 1000
    height = int(width * rng.uniform(0.55, 0.68))
    bezel = int(rng.uniform(0.015, 0.05) * width)
    image = Image.new("RGBA", (width, height), _gray(rng, 5, 35, 2) + (255,))
    dark = rng.random() < 0.55
    back = _gray(rng, 18, 50, 8) if dark else _gray(rng, 228, 252, 4)
    inner = (bezel, bezel, width - bezel, height - bezel)
    draw = ImageDraw.Draw(image)
    draw.rectangle(inner, fill=back)
    ink = _gray(rng, 150, 230, 30) if dark else _gray(rng, 20, 90, 20)
    left = inner[0] + 20
    if rng.random() < 0.6:  # a side bar or header of another shade
        shade = _gray(rng, 30, 70, 10) if dark else _gray(rng, 200, 235, 10)
        side = int(rng.uniform(0.12, 0.3) * width)
        draw.rectangle((inner[0], inner[1], inner[0] + side, inner[3]), fill=shade)
        left = inner[0] + side + 20
    y = inner[1] + 20
    size = rng.uniform(8, 16)
    while y < inner[3] - 20:
        x = left
        while x < inner[2] - 40:
            word = rng.uniform(2, 9) * size
            colour = ink
            if rng.random() < 0.05:
                colour = tuple(int(v) for v in rng.integers(40, 255, 3))
            draw.rectangle((x, y, min(x + word, inner[2] - 20), y + 0.7 * size), fill=colour)
            x += word + size
            if rng.random() < 0.08:
                break
        y += size * rng.uniform(1.6, 2.6)
    return image


def cover(rng: np.random.Generator) -> Image.Image:
    """A closed book or folder: a coloured cover with a title band."""
    width = 700
    height = int(width * rng.uniform(1.2, 1.5))
    colour = tuple(int(v) for v in rng.integers(0, 256, 3))
    image = _noise(rng, (width, height), colour, 4.0).convert("RGBA")
    draw = ImageDraw.Draw(image)
    band = rng.uniform(0.1, 0.6) * height
    text = (255, 255, 255) if sum(colour) < 380 else (15, 15, 15)
    for row in range(int(rng.integers(1, 4))):
        y = band + row * 70
        draw.rectangle((70, y, 70 + rng.uniform(0.3, 0.8) * (width - 140), y + 45), fill=text)
    draw.rectangle((0, 0, 30, height), fill=tuple(max(0, v - 40) for v in colour))
    return image


def mug(rng: np.random.Generator) -> Image.Image:
    width = 500
    image = Image.new("RGBA", (width, width), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    colour = tuple(int(v) for v in rng.integers(0, 256, 3)) + (255,)
    draw.ellipse((40, 40, width - 110, width - 40), fill=colour)
    inside = tuple(max(0, v - 70) for v in colour[:3]) + (255,)
    draw.ellipse((90, 90, width - 160, width - 90), fill=inside)
    draw.ellipse((width - 170, 190, width - 10, 310), outline=colour, width=30)
    return image


def cable(rng: np.random.Generator) -> Image.Image:
    width = 1000
    image = Image.new("RGBA", (width, width // 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    colour = _gray(rng, 10, 60) if rng.random() < 0.6 else _gray(rng, 220, 250)
    phase, wave = rng.uniform(0, 6), rng.uniform(0.02, 0.1)
    xs = np.linspace(0, width, 60)
    ys = width / 8 + wave * width * np.sin(xs / width * rng.uniform(2, 8) + phase)
    draw.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=colour + (255,), width=18)
    return image


def _draw_object(rng: np.random.Generator, kind: str) -> tuple[Image.Image, float, float]:
    """The object's RGBA picture and its long-side range as a fraction of the frame."""
    if kind == "laptop":
        return laptop(rng), 0.5, 1.4
    if kind == "keyboard":
        return laptop(rng, with_keys_only=True), 0.4, 1.1
    if kind == "screen":
        return screen(rng), 0.4, 1.2
    if kind == "cover":
        return cover(rng), 0.25, 0.7
    if kind == "mug":
        return mug(rng), 0.12, 0.3
    return cable(rng), 0.4, 1.2


def place_picture(
    surface: NDArray[np.float32],
    picture: Image.Image,
    rng: np.random.Generator,
    size_range: tuple[float, float],
    perspective: float = 0.12,
) -> NDArray[np.float32]:
    """Draw an RGBA ``picture`` onto ``surface`` at a random scale, angle and position
    through a perspective transform; return its alpha over the frame."""
    height, width = surface.shape[:2]
    pw, ph = picture.size
    long = rng.uniform(*size_range) * max(height, width)
    scale = long / max(pw, ph)
    center = rng.uniform([-0.1 * width, -0.1 * height], [1.1 * width, 1.1 * height])
    angle = rng.uniform(-math.pi, math.pi) if rng.random() < 0.3 else rng.normal(0, 0.35)
    source = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float64)
    placed = cp.apply_homography(
        cp.similarity(scale, angle, np.array([pw / 2, ph / 2]), center), source
    )
    placed += rng.uniform(-perspective, perspective, (4, 2)) * long
    if cp.polygon_area(placed) * cp.polygon_area(source) <= 0:
        return np.zeros((height, width), np.float32)
    transform = cp.corner_homography(source, placed)
    inverse = np.linalg.inv(transform)
    inverse /= inverse[2, 2]
    warped = picture.transform(
        (width, height), Image.Transform.PERSPECTIVE, tuple(inverse.ravel()[:8]),
        Image.Resampling.BILINEAR,
    )  # fmt: skip
    rgba = np.asarray(warped, dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    surface += alpha * (rgba[..., :3] - surface)
    return alpha[..., 0]


# --------------------------------------------------------------------------- thumbs


def _ellipse(center, axis, along: float, across: float, count: int = 40) -> list[tuple]:
    """Points of an ellipse with semi-axes ``along`` (direction ``axis``) and ``across``."""
    normal = np.array([-axis[1], axis[0]])
    t = np.linspace(0, 2 * math.pi, count, endpoint=False)[:, None]
    points = center + np.cos(t) * along * axis + np.sin(t) * across * normal
    return [tuple(p) for p in points.tolist()]


def thumb(
    surface: NDArray[np.float32],
    polygon: NDArray,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """A thumb over the page edge (``polygon``, output px) and the hand behind it,
    outside the page; returns the thumb's pixels (they leave the mask)."""
    height, width = surface.shape[:2]
    index = int(rng.integers(len(polygon)))
    point, after = polygon[index], polygon[(index + 1) % len(polygon)]
    along = after - point
    along = along / max(float(np.hypot(*along)), 1e-6)
    inward = np.array([-along[1], along[0]])
    if np.dot(polygon.mean(axis=0) - point, inward) < 0:
        inward = -inward
    tilt = rng.normal(0, 0.35)
    axis = inward * math.cos(tilt) + along * math.sin(tilt)
    length = rng.uniform(0.1, 0.25) * float(min(np.ptp(polygon, axis=0)))
    skin = np.array([rng.uniform(195, 240), rng.uniform(150, 195), rng.uniform(120, 170)])
    skin *= rng.uniform(0.6, 1.05)
    layers = []
    palm = Image.new("L", (width, height), 0)
    ImageDraw.Draw(palm).polygon(
        _ellipse(point - axis * 1.1 * length, axis, 1.3 * length, 1.1 * length), fill=255
    )
    layers.append((palm, 0.92))
    finger = Image.new("L", (width, height), 0)
    ImageDraw.Draw(finger).polygon(
        _ellipse(point + axis * rng.uniform(0.0, 0.35) * length, axis, 0.65 * length,
                 rng.uniform(0.25, 0.33) * length), fill=255,
    )  # fmt: skip
    layers.append((finger, 1.0))
    for layer, gain in layers:
        weight = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
        edge = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.15 * length + 1)), np.float32)
        shade = 0.7 + 0.3 * (edge / 255.0)
        colour = (skin * gain)[None, None, :] * shade[..., None]
        surface += weight[..., None] * (colour.astype(np.float32) - surface)
    return np.asarray(finger) > 127


def book_block(
    surface: NDArray[np.float32],
    polygons: list[NDArray],
    rng: np.random.Generator,
) -> None:
    """The page stack under the page(s): their hull swept by a thickness vector in a
    paper-edge colour, sometimes inside a hard cover. The pages drawn afterwards cover
    its top; only the edge band stays visible, and it is never page."""
    height, width = surface.shape[:2]
    points = np.concatenate(polygons)
    hull = _hull(points)
    size = float(np.sqrt(abs(cp.polygon_area(hull))))
    angle = rng.uniform(0, 2 * math.pi)
    sweep = size * rng.uniform(0.015, 0.06) * np.array([math.cos(angle), math.sin(angle)])
    layers = []
    if rng.random() < 0.45:  # hard cover a little larger and further down
        centre = hull.mean(axis=0)
        grown = centre + (hull - centre) * rng.uniform(1.01, 1.04)
        colour = rng.integers(0, 256, 3) if rng.random() < 0.5 else rng.integers(10, 70, 3)
        layers.append((grown, sweep * 1.4, np.asarray(colour, np.float32)))
    layers.append((hull, sweep, np.full(3, rng.uniform(170, 235), np.float32)))
    for polygon, shift, colour in layers:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for t in np.linspace(0, 1, 8):
            draw.polygon([tuple(p) for p in (polygon + t * shift).tolist()], fill=255)
        weight = np.asarray(mask, np.float32)[..., None] / 255.0
        grain = rng.normal(0, 6, (height, width, 1)).astype(np.float32)
        lines = np.sin(np.arange(height)[:, None, None] * rng.uniform(0.8, 2.5)) * 6
        paint = colour * rng.uniform(0.8, 1.0) + grain + lines.astype(np.float32)
        surface += weight * (paint - surface)


def _hull(points: NDArray) -> NDArray:
    """Convex hull (monotone chain), clockwise on screen like ``cp.outline``."""
    pts = sorted(map(tuple, np.asarray(points, float).tolist()))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


# --------------------------------------------------------------------------- scene


def grid_nodes(grid: NDArray, size: tuple[int, int]) -> NDArray[np.float64]:
    """Grid values (flattener, w.r.t. an image of ``size`` = (height, width)) in px."""
    height, width = size
    return (np.asarray(grid, np.float64) + 1.0) / 2.0 * np.array([width, height])


def flat_grid(rows: int = 9, cols: int = 7) -> NDArray[np.float32]:
    """The grid of a flat page filling its image."""
    ys, xs = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij")
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def _polygon_mask(polygon: NDArray, out_hw: tuple[int, int]) -> NDArray[np.bool_]:
    height, width = out_hw
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon.tolist()], fill=255)
    return np.asarray(mask) > 127


def scene(
    rng: np.random.Generator,
    config: SceneConfig,
    page: tuple[Image.Image, NDArray] | None,
    bank: cp.BackgroundBank | None = None,
    neighbor: tuple[Image.Image, NDArray] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_], dict[str, Any]]:
    """One scene: (H, W, 3) uint8 image, (H, W) page mask and what was drawn.

    ``page`` and ``neighbor`` are (image, grid) with the grid in flattener values of the
    image; ``page`` None gives a negative. ``neighbor`` is drawn only with
    ``config.neighbor_prob``."""
    settings = cp.CompositeConfig.from_value(config.composite)
    out_hw = frame_size(rng, config)
    height, width = out_hw
    surface = cp._background(settings, bank, rng, out_hw)
    info: dict[str, Any] = {"frame": [height, width], "objects": []}
    count = int(rng.choice(len(config.object_count), p=np.divide(
        config.object_count, sum(config.object_count))))  # fmt: skip
    for _ in range(count):
        kind = _choice(rng, config.objects)
        picture, low, high = _draw_object(rng, kind)
        place_picture(surface, picture, rng, (low, high))
        info["objects"].append(kind)
    mask = np.zeros(out_hw, dtype=bool)
    if page is None:
        info["page"] = False
        return _finish(surface, mask, rng, settings, config, out_hw, info)
    image, grid = page
    nodes_px = grid_nodes(grid, (image.size[1], image.size[0]))
    outline = cp.outline(nodes_px)
    transform, turn = cp.plan_placement(nodes_px, out_hw, settings, rng)
    info.update(page=True, turn=turn, closeup=False)
    if rng.random() < config.closeup_prob:
        zoom = math.exp(rng.uniform(*np.log(config.closeup_zoom)))
        rows, cols = nodes_px.shape[:2]
        node = nodes_px[
            int(rng.integers(rows // 10, rows - rows // 10)),
            int(rng.integers(cols // 10, cols - cols // 10)),
        ]
        anchor = cp.apply_homography(transform, node[None])[0]
        target = np.array([width, height]) * rng.uniform(0.3, 0.7, 2)
        transform = cp.similarity(zoom, 0.0, anchor, target) @ transform
        info.update(closeup=True, zoom=round(zoom, 3))
    placed = cp.apply_homography(transform, outline)
    corners = cp.apply_homography(transform, cp._corners(nodes_px))
    fold = None
    info.update(neighbor=False, book=False)
    if neighbor is not None and rng.random() < config.neighbor_prob:
        other, other_grid = neighbor
        neighbor_seed = int(rng.integers(2**63))
        drawn = cp._draw_neighbor(
            surface, (other, other_grid, np.ones(other_grid.shape[:2], bool)),
            corners, out_hw, settings, np.random.default_rng(neighbor_seed),
        )  # fmt: skip
        if drawn is not None:
            fold, facing = drawn
            mask |= _polygon_mask(facing, out_hw)
            info["neighbor"] = True
    if rng.random() < (config.book_prob_spread if info["neighbor"] else config.book_prob):
        book_block(surface, [placed] + ([facing] if info["neighbor"] else []), rng)
        if info["neighbor"]:  # the facing page stays on top of the stack
            cp._draw_neighbor(
                surface, (other, other_grid, np.ones(other_grid.shape[:2], bool)),
                corners, out_hw, settings, np.random.default_rng(neighbor_seed),
            )  # fmt: skip
        info["book"] = True
    box = cp.polygon_box(placed, out_hw)
    alpha = np.zeros(out_hw, dtype=np.float32)
    if box is not None:
        left, top, right, bottom = box
        step = math.sqrt(abs(cp.polygon_area(outline)) / max(abs(cp.polygon_area(placed)), 1e-9))
        pixels = cp.warp_homography(image, transform, out_hw, step=step, box=box)
        alpha[top:bottom, left:right] = cp.page_alpha(
            placed, box, rng.uniform(0.0, settings.edge_trim)
        )
    if rng.random() < settings.drop_shadow_prob:
        shift = rng.normal(0.0, 0.012, size=2) * [width, height]
        moved = np.roll(alpha, (round(shift[1]), round(shift[0])), axis=(0, 1))
        shade = cp._shadow(Image.fromarray(np.rint(moved * 255.0).astype(np.uint8)),
                           rng.uniform(3.0, 18.0))  # fmt: skip
        surface *= (1.0 - rng.uniform(0.1, 0.45) * shade)[..., None]
    if box is not None:
        gain = np.float32(rng.uniform(*settings.page_gain))
        region = surface[top:bottom, left:right]
        weight = alpha[top:bottom, left:right, None]
        region += weight * (pixels.astype(np.float32) * gain - region)
    mask |= _polygon_mask(placed, out_hw)
    if fold is not None:
        page_width = float(np.hypot(*(corners[1] - corners[0])))
        strength = rng.uniform(*settings.fold_shadow)
        surface *= cp._fold_shade(fold, page_width, out_hw, strength, rng)[..., None]
    info["thumbs"] = 0
    if rng.random() < config.thumb_prob:
        for _ in range(int(rng.integers(1, 3))):
            mask &= ~thumb(surface, placed, rng)
            info["thumbs"] += 1
    return _finish(surface, mask, rng, settings, config, out_hw, info)


def _finish(surface, mask, rng, settings, config, out_hw, info):
    height, width = out_hw
    light = rng.uniform(1.0 - settings.illumination, 1.0 + 0.3 * settings.illumination, (3, 3))
    light_field = cp.smooth_field(light, out_hw)
    if rng.random() < settings.cast_shadow_prob:
        shadow = Image.new("L", (width, height), 0)
        corners = int(rng.integers(3, 6))
        ImageDraw.Draw(shadow).polygon(cp._random_polygon(rng, out_hw, corners, 0.2), fill=255)
        light_field *= 1.0 - rng.uniform(0.15, 0.45) * cp._shadow(shadow, rng.uniform(2.0, 25.0))
    balance = rng.uniform(1.0 - settings.white_balance, 1.0 + settings.white_balance, size=3)
    surface *= light_field[..., None] * balance.astype(np.float32)
    if config.noise > 0:
        surface += rng.normal(0, rng.uniform(0, config.noise), surface.shape).astype(np.float32)
    result = np.clip(np.rint(surface), 0, 255).astype(np.uint8)
    picture = Image.fromarray(result)
    if rng.random() < config.blur_prob:
        picture = picture.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=int(rng.integers(*config.jpeg_quality)))
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        result = np.asarray(decoded.convert("RGB"))
    info["area_fraction"] = round(float(mask.mean()), 4)
    return result, mask, info
