"""Training composites: a page placed on a new background by one homography.

``compose`` maps the page of a training sample into the model input with a homography T
from source edge pixels to output edge pixels. T combines an in-plane tilt, optional
quarter turns, a camera-like perspective (the placed page's corners move by up to
``CompositeConfig.perspective`` of its size), a page area drawn from ``area`` and a
position in the frame. The image and the target grid go through the same T, so the
targets stay exact up to bilinear sampling. Inside its outline the page keeps its own
pixels behind a hard, anti-aliased edge; outside, the background is a CC0 texture from a
``BackgroundBank`` or a procedural surface. Page and background share one illumination
field, cast shadow and white balance.

With ``neighbor_prob`` a page of another sample is drawn next to the placed page's left or
right edge, sharing that edge like the facing page of an open book, with a fold shadow
across both. The target stays the placed page, so the model learns to stop at the fold
where the paper goes on.

``tilt_transform`` builds the ``tilt`` evaluation view of ``data.evaluation_view``: the
page crop seen by a tilted camera, the frame corners moved by up to a fixed fraction.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

BACKGROUND_SCHEMA = "flattener-backgrounds-1"
# Tested placements per sample before the perspective is dropped.
PLACEMENT_ATTEMPTS = 8
# Least homogeneous denominator over the page (T) and over the frame (T^-1). Keeps the
# horizon line far from both, so no part of the page is stretched towards infinity.
MIN_DENOMINATOR = 0.2
# A source step (source pixels per output pixel) above this is box-reduced first.
REDUCE_ABOVE = 4.0 / 3.0
ALPHA_SUPERSAMPLING = 4
BILINEAR = Image.Resampling.BILINEAR


@dataclass(frozen=True, slots=True)
class CompositeConfig:
    """How often and how a training sample becomes a composite (``prob`` 0: never).

    - ``area``: page area as a fraction of the frame, drawn log-uniformly. A page that
      does not fit at that size is shrunk until it does.
    - ``overhang``: how far the page's bounding box may leave the frame on a side, as a
      fraction of the box (0.02 cuts at most 2% of the page off).
    - ``perspective``: largest shift of each placed corner, as a fraction of the placed
      page size; the strength of a sample is drawn uniformly up to it.
    - ``angle_std``, ``wide_angle_prob``, ``wide_angle``: in-plane tilt in degrees,
      normal(0, angle_std), or uniform within ±wide_angle with wide_angle_prob.
    - ``turns``: weights of counter-clockwise quarter turns by 0, 90, 180 and 270
      degrees. The target stays the upright page and the item's ``turn`` holds 0-3.
    - ``background_dir``: folder with ``manifest.json`` and texture files
      (``tools/backgrounds/fetch_polyhaven.py``). Without it every background is
      procedural. ``${VAR}`` expands when the dataset opens it.
    - ``texture_prob``: chance of a texture over a procedural surface.
    - ``panel_prob``: chance of a second flat surface in the background (a table edge,
      a folder or a box side).
    - ``drop_shadow_prob``, ``cast_shadow_prob``, ``illumination``, ``page_gain``,
      ``white_balance``: lighting shared by page and background.
    - ``jpeg_prob``, ``jpeg_quality``: JPEG round trip of the finished composite.
    - ``edge_trim``: largest inward shift of the page edge in output pixels, drawn per
      sample, so the render's original background does not show along the outline.
    - ``neighbor_prob``: chance that a composite shows a facing page (another sample's
      page, passed to ``compose`` as ``neighbor``) along the page's left or right edge.
    - ``neighbor_width``: how much of the facing page shows, as a fraction of the page
      width, drawn uniformly.
    - ``neighbor_fold``: largest change of the facing page's far edge, as a fraction of
      its length (scale) and of the page height (shift), for a page lifted at the fold.
    - ``fold_shadow``: strength range of the shadow along the fold (0 to 1).
    """

    prob: float = 0.0
    area: tuple[float, float] = (0.05, 1.0)
    overhang: float = 0.02
    perspective: float = 0.12
    angle_std: float = 8.0
    wide_angle_prob: float = 0.2
    wide_angle: float = 45.0
    turns: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    background_dir: str | None = None
    texture_prob: float = 0.75
    panel_prob: float = 0.35
    drop_shadow_prob: float = 0.6
    cast_shadow_prob: float = 0.25
    illumination: float = 0.35
    page_gain: tuple[float, float] = (0.8, 1.08)
    white_balance: float = 0.08
    jpeg_prob: float = 0.0
    jpeg_quality: tuple[int, int] = (45, 95)
    edge_trim: float = 1.0
    neighbor_prob: float = 0.0
    neighbor_width: tuple[float, float] = (0.3, 1.0)
    neighbor_fold: float = 0.15
    fold_shadow: tuple[float, float] = (0.05, 0.45)

    def __post_init__(self) -> None:
        pairs = ("area", "page_gain", "jpeg_quality", "neighbor_width", "fold_shadow")
        for name in (*pairs, "turns"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "background_dir":
                if value is not None and (not isinstance(value, str) or not value):
                    raise ValueError("background_dir must be a non-empty string or null")
            elif item.name in (*pairs, "turns"):
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
                    raise ValueError(f"{item.name} must hold numbers")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{item.name} must be a non-negative number")
            if (item.name == "prob" or item.name.endswith("_prob")) and value > 1:
                raise ValueError(f"{item.name} must be <= 1")
        low, high = self.area
        if len(self.area) != 2 or not 0 < low <= high <= 1:
            raise ValueError("area must be (low, high) with 0 < low <= high <= 1")
        if len(self.turns) != 4 or min(self.turns) < 0 or sum(self.turns) <= 0:
            raise ValueError("turns must be four non-negative weights with a positive sum")
        gain, quality = self.page_gain, self.jpeg_quality
        if len(gain) != 2 or not 0 < gain[0] <= gain[1] <= 2:
            raise ValueError("page_gain must be (low, high) with 0 < low <= high <= 2")
        if len(quality) != 2 or not 1 <= quality[0] <= quality[1] <= 100:
            raise ValueError("jpeg_quality must be (low, high) within 1-100")
        if self.overhang >= 0.25 or self.perspective > 0.3 or self.illumination >= 1:
            raise ValueError("overhang < 0.25, perspective <= 0.3 and illumination < 1")
        if self.wide_angle > 180 or self.white_balance >= 0.5:
            raise ValueError("wide_angle <= 180 and white_balance < 0.5")
        near, far = self.neighbor_width
        if len(self.neighbor_width) != 2 or not 0 < near <= far <= 1.5:
            raise ValueError("neighbor_width must be (low, high) with 0 < low <= high <= 1.5")
        low, high = self.fold_shadow
        if len(self.fold_shadow) != 2 or not 0 <= low <= high < 1 or self.neighbor_fold > 0.5:
            raise ValueError("fold_shadow must be (low, high) within [0, 1); neighbor_fold <= 0.5")

    @property
    def enabled(self) -> bool:
        return self.prob > 0

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = list(value) if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_value(cls, value: CompositeConfig | dict[str, Any] | None) -> CompositeConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("composite must be an object")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown composite fields: {sorted(unknown)}")
        return cls(**value)


# --------------------------------------------------------------------------- geometry


def apply_homography(transform: NDArray, points: NDArray) -> NDArray[np.float64]:
    """``transform`` (3x3) applied to (..., 2) points."""
    points = np.asarray(points, dtype=np.float64)
    mapped = points @ transform[:, :2].T + transform[:, 2]
    return mapped[..., :2] / mapped[..., 2:3]


def corner_homography(source: NDArray, target: NDArray) -> NDArray[np.float64]:
    """The homography that maps four ``source`` points onto four ``target`` points."""
    rows, values = [], []
    for (x, y), (u, v) in zip(np.asarray(source, float), np.asarray(target, float), strict=True):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        values += [u, v]
    solution = np.linalg.solve(np.array(rows), np.array(values))
    return np.append(solution, 1.0).reshape(3, 3)


def similarity(scale: float, angle: float, origin: NDArray, destination: NDArray) -> NDArray:
    """Scale by ``scale`` and turn by ``angle`` (radians) about ``origin``, then move
    ``origin`` onto ``destination``."""
    cos, sin = math.cos(angle), math.sin(angle)
    linear = scale * np.array([[cos, -sin], [sin, cos]])
    transform = np.eye(3)
    transform[:2, :2] = linear
    transform[:2, 2] = np.asarray(destination, float) - linear @ np.asarray(origin, float)
    return transform


def outline(nodes: NDArray) -> NDArray:
    """The border nodes of a (Gh, Gw, 2) grid as a closed polygon, clockwise on screen:
    top row, right column, bottom row, left column."""
    return np.concatenate([nodes[0, :-1], nodes[:-1, -1], nodes[-1, :0:-1], nodes[:0:-1, 0]])


def polygon_area(polygon: NDArray) -> float:
    """Signed area (shoelace)."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _normalized(transform: NDArray) -> NDArray:
    return transform / transform[2, 2]


def _frame(out_hw: tuple[int, int]) -> NDArray:
    height, width = out_hw
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def placement_ok(transform: NDArray, page: NDArray, out_hw: tuple[int, int]) -> bool:
    """Whether ``transform`` keeps the horizon away from the page and the frame and
    keeps the page's orientation (no mirror)."""
    transform = _normalized(transform)
    inverse = _normalized(np.linalg.inv(transform))
    page_denominator = page @ transform[2, :2] + transform[2, 2]
    frame_denominator = _frame(out_hw) @ inverse[2, :2] + inverse[2, 2]
    placed = apply_homography(transform, page)
    return bool(
        (page_denominator > MIN_DENOMINATOR).all()
        and (frame_denominator > MIN_DENOMINATOR).all()
        and polygon_area(placed) * polygon_area(page) > 0
    )


def _fit_in_frame(
    transform: NDArray,
    page: NDArray,
    out_hw: tuple[int, int],
    overhang: float,
    rng: np.random.Generator,
) -> NDArray:
    """Shrink the placed page about its centre until its box fits the frame (with the
    allowed overhang), then move it to a random position where it does."""
    height, width = out_hw
    frame = np.array([width, height], dtype=np.float64)
    placed = apply_homography(transform, page)
    span = np.maximum(placed.max(axis=0) - placed.min(axis=0), 1e-6)
    shrink = float(min(1.0, *(frame / (span * (1.0 - 2.0 * overhang)))))
    if shrink < 1.0:
        center = placed.mean(axis=0)
        transform = similarity(shrink, 0.0, center, center) @ transform
        placed = apply_homography(transform, page)
        span = placed.max(axis=0) - placed.min(axis=0)
    low_shift = -overhang * span - placed.min(axis=0)
    high_shift = frame + overhang * span - placed.max(axis=0)
    shift = rng.uniform(np.minimum(low_shift, high_shift), np.maximum(low_shift, high_shift))
    move = np.eye(3)
    move[:2, 2] = shift
    return _normalized(move @ transform)


def plan_placement(
    nodes_px: NDArray, out_hw: tuple[int, int], config: CompositeConfig, rng: np.random.Generator
) -> tuple[NDArray[np.float64], int]:
    """A placement homography (source edge px to output edge px) and its quarter turn.

    The page turns about its centre (in-plane tilt plus ``turn`` quarter turns counter-
    clockwise on screen), its four placed corners move by up to the drawn perspective
    strength, it scales to the drawn area and moves to a random position. A placement
    that brings the horizon near the page or the frame retries with half the
    perspective; after ``PLACEMENT_ATTEMPTS`` the page is placed flat.
    """
    height, width = out_hw
    weights = np.asarray(config.turns, dtype=np.float64)
    turn = int(rng.choice(4, p=weights / weights.sum()))
    if rng.random() < config.wide_angle_prob:
        tilt = rng.uniform(-config.wide_angle, config.wide_angle)
    else:
        tilt = rng.normal(0.0, config.angle_std)
    low, high = config.area
    area = math.exp(rng.uniform(math.log(low), math.log(high)))
    strength = rng.uniform(0.0, config.perspective)
    page = outline(nodes_px)
    corners = nodes_px[[0, 0, -1, -1], [0, -1, -1, 0]]
    # y points down, so a negative angle turns counter-clockwise on screen
    angle = math.radians(tilt) - turn * math.pi / 2.0
    upright = similarity(1.0, angle, page.mean(axis=0), (width / 2.0, height / 2.0))
    for attempt in range(PLACEMENT_ATTEMPTS + 1):
        if attempt == PLACEMENT_ATTEMPTS:
            strength = 0.0
        placed = apply_homography(upright, corners)
        size = placed.max(axis=0) - placed.min(axis=0)
        jitter = rng.uniform(-1.0, 1.0, size=(4, 2)) * strength * size
        transform = corner_homography(placed, placed + jitter) @ upright
        polygon = apply_homography(transform, page)
        scale = math.sqrt(area * height * width / max(abs(polygon_area(polygon)), 1e-9))
        center = polygon.mean(axis=0)
        transform = similarity(scale, 0.0, center, center) @ transform
        transform = _fit_in_frame(transform, page, out_hw, config.overhang, rng)
        if placement_ok(transform, page, out_hw):
            return transform, turn
        strength *= 0.5
    return transform, turn


def tilt_transform(
    base: NDArray,
    nodes_px: NDArray,
    out_hw: tuple[int, int],
    strength: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """``base`` (source to output) followed by a camera tilt: the four frame corners move
    by up to ``strength`` of the frame side. If the page leaves the frame, the view
    shrinks about the frame centre until every page node is inside."""
    height, width = out_hw
    frame = _frame(out_hw)
    moved = frame + rng.uniform(-strength, strength, size=(4, 2)) * [width, height]
    transform = corner_homography(frame, moved) @ base
    placed = apply_homography(transform, nodes_px.reshape(-1, 2))
    center = np.array([width / 2.0, height / 2.0])
    reach = np.abs(placed - center).max(axis=0)
    shrink = float(min(1.0, *(center / np.maximum(reach, 1e-9))))
    if shrink < 1.0:
        transform = similarity(shrink * (1.0 - 1e-6), 0.0, center, center) @ transform
    return _normalized(transform)


# --------------------------------------------------------------------------- sampling


def _reduce_source(image: Image.Image, step: float) -> tuple[Image.Image, int]:
    """``image`` box-reduced by a whole factor near ``step`` (source px per output px)
    and that factor. An unloaded JPEG decodes at a reduced DCT scale first."""
    factor = max(1, round(step)) if step > REDUCE_ABOVE else 1
    if factor == 1:
        return image, 1
    width = image.size[0]
    done = 1
    if image.format == "JPEG":  # a loaded image ignores draft()
        result = image.draft("RGB", (math.ceil(image.size[0] / factor), 1))
        if result is not None and image.size[0] < width:
            done = round(width / image.size[0])
    rest = max(1, round(factor / done))
    if image.mode not in ("RGB", "F"):
        image = image.convert("RGB")
    if rest > 1:
        image = image.reduce(rest)
    return image, done * rest


Box = tuple[int, int, int, int]


def warp_homography(
    image: Image.Image,
    transform: NDArray,
    out_hw: tuple[int, int],
    *,
    step: float = 1.0,
    edge: bool = False,
    box: Box | None = None,
) -> NDArray[np.uint8]:
    """Sample ``image`` through ``transform`` (source edge px to output edge px) into an
    (H, W, 3) uint8 array, bilinear; with ``box`` (left, top, right, bottom in output
    pixels) only that part of the output. A float image (mode F) stays float, (H, W);
    other modes become RGB.

    PIL maps each output pixel centre (i + 0.5) through the inverse and samples the
    source at that edge coordinate. ``step`` (source px per output px) above
    ``REDUCE_ABOVE`` box-reduces the source by a whole factor first, which keeps edge
    coordinates exact (reduced pixel k covers [k f, (k + 1) f)). Outside the source the
    result is black, or with ``edge`` the border pixels repeat.
    """
    out_h, out_w = out_hw
    left, top, right, bottom = (0, 0, out_w, out_h) if box is None else box
    source, factor = _reduce_source(image, step)
    if source.mode not in ("RGB", "F"):
        source = source.convert("RGB")
    move = np.eye(3)
    move[:2, 2] = (left, top)
    inverse = np.diag([1.0 / factor, 1.0 / factor, 1.0]) @ np.linalg.inv(transform) @ move
    inverse = _normalized(inverse)
    size = (right - left, bottom - top)
    if edge:
        width, height = source.size
        reach = apply_homography(inverse, _frame((size[1], size[0])))
        low = np.floor(reach.min(axis=0)) - 1
        high = np.ceil(reach.max(axis=0)) + 1
        if (low < 0).any() or high[0] > width or high[1] > height:
            low = np.maximum(low, -4.0 * np.array([width, height]))
            high = np.minimum(high, 5.0 * np.array([width, height]))
            pad_low = np.maximum(-low, 0).astype(int)
            pad_high = np.maximum(high - [width, height], 0).astype(int)
            pixels = np.asarray(source)
            widths = [(pad_low[1], pad_high[1]), (pad_low[0], pad_high[0])]
            pixels = np.pad(pixels, widths + [(0, 0)] * (pixels.ndim - 2), mode="edge")
            source = Image.fromarray(pixels)
            shift = np.eye(3)
            shift[:2, 2] = pad_low
            inverse = _normalized(shift @ inverse)
    data = tuple(inverse.ravel()[:8])
    output = source.transform(size, Image.Transform.PERSPECTIVE, data, Image.Resampling.BILINEAR)
    return np.asarray(output)


def polygon_box(polygon: NDArray, out_hw: tuple[int, int]) -> Box | None:
    """Whole-pixel box around ``polygon`` inside the frame, or None when it misses."""
    height, width = out_hw
    low = np.clip(np.floor(polygon.min(axis=0)), 0, [width, height]).astype(int)
    high = np.clip(np.ceil(polygon.max(axis=0)), 0, [width, height]).astype(int)
    if (high <= low).any():
        return None
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def page_alpha(polygon: NDArray, box: Box, trim: float = 0.0) -> NDArray[np.float32]:
    """Coverage of ``polygon`` (output edge px) per pixel of ``box``, 4x supersampled,
    float32 in [0, 1]. ``trim`` (0-1 px) lowers partial coverage, moving the edge
    inward."""
    left, top, right, bottom = box
    scale = ALPHA_SUPERSAMPLING
    mask = Image.new("L", ((right - left) * scale, (bottom - top) * scale), 0)
    local = (polygon - [left, top]) * scale
    ImageDraw.Draw(mask).polygon([tuple(p) for p in local], fill=255)
    alpha = np.asarray(mask.reduce(scale), dtype=np.float32) / 255.0
    if trim > 0:
        cut = 0.5 * min(trim, 1.0)
        alpha = np.clip((alpha - cut) / (1.0 - cut), 0.0, 1.0)
    return alpha


# --------------------------------------------------------------------------- backgrounds


class BackgroundBank:
    """Texture files listed in ``<directory>/manifest.json`` (``BACKGROUND_SCHEMA``).

    Each entry names a ``file`` and its ``license``. ``license_ok`` (the release-track
    check of ``data``) must accept every entry, or the bank refuses the folder. Decoded,
    tiled textures are kept per process in a small LRU cache that pickling drops.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        license_ok: Callable[[str], bool] | None = None,
        cache_size: int = 24,
    ) -> None:
        self.directory = Path(directory)
        raw = (self.directory / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema") != BACKGROUND_SCHEMA:
            raise ValueError(f"{self.directory}: manifest schema must be {BACKGROUND_SCHEMA}")
        entries = manifest.get("files") or []
        if not entries:
            raise ValueError(f"{self.directory}: the manifest lists no files")
        refused = [e["file"] for e in entries if license_ok and not license_ok(e["license"])]
        if refused:
            raise PermissionError(f"{self.directory}: licenses not allowed for {refused[:3]}")
        self.files = [str(entry["file"]) for entry in entries]
        self.licenses = Counter(str(entry["license"]) for entry in entries)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[int, int], Image.Image] = OrderedDict()
        self._sides: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getstate__(self) -> dict[str, Any]:
        return {**vars(self), "_cache": OrderedDict(), "_sides": {}}

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)

    def describe(self) -> dict[str, Any]:
        return {
            "files": len(self.files),
            "licenses": dict(sorted(self.licenses.items())),
            "manifest_sha256": self.manifest_sha256,
        }

    def tiled(self, index: int, count: int = 2) -> Image.Image:
        """Texture ``index`` as an RGB image tiled ``count`` x ``count`` (the textures
        repeat seamlessly)."""
        key = (index, count)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with Image.open(self.directory / self.files[index]) as image:
            pixels = np.asarray(image.convert("RGB"))
        tiled = Image.fromarray(np.tile(pixels, (count, count, 1)))
        self._cache[key] = tiled
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tiled

    def draw(self, rng: np.random.Generator, out_hw: tuple[int, int]) -> NDArray[np.float32]:
        """A random view of a random texture: turned, scaled (0.9-2.2 output px per
        texture px; the fine end is dropped when a 2x2 tile would not cover the frame at
        every offset) and shifted, with a brightness and tint."""
        height, width = out_hw
        index = int(rng.integers(len(self.files)))
        if index not in self._sides:
            with Image.open(self.directory / self.files[index]) as image:
                self._sides[index] = float(min(image.size))  # reads the header only
        side = self._sides[index]  # one tile
        reach = 0.5 * math.hypot(width, height)  # output centre to corner
        coarse, fine = 1.0 / 2.2, 1.0 / 0.9  # texture px per output px
        count = max(2, math.ceil(1.0 + 2.0 * reach * coarse / side))
        fine = min(fine, (count - 1) * side / (2.0 * reach))
        texels = rng.uniform(min(coarse, fine), fine)
        tiled = self.tiled(index, count)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        center = rng.uniform(reach * texels, count * side - reach * texels, size=2)
        cos, sin = math.cos(angle) * texels, math.sin(angle) * texels
        # PIL affine: output (x, y) -> texture (a x + b y + c, d x + e y + f)
        offset = center - np.array([[cos, -sin], [sin, cos]]) @ [width / 2.0, height / 2.0]
        data = (cos, -sin, offset[0], sin, cos, offset[1])
        view = tiled.transform((width, height), Image.Transform.AFFINE, data, BILINEAR)
        tone = rng.uniform(0.55, 1.15) * rng.uniform(0.9, 1.1, size=3)
        return np.asarray(view, dtype=np.float32) * tone.astype(np.float32)


def _quarter(out_hw: tuple[int, int]) -> tuple[int, int]:
    return max(2, out_hw[0] // 4), max(2, out_hw[1] // 4)


def smooth_field(values: NDArray, out_hw: tuple[int, int]) -> NDArray[np.float32]:
    """A coarse (h, w) grid of values as a smooth (H, W) float32 field: bicubic to a
    quarter of ``out_hw``, then bilinear to full size."""
    coarse = torch.from_numpy(values.astype(np.float32))[None, None]
    quarter = F.interpolate(coarse, size=_quarter(out_hw), mode="bicubic", align_corners=False)
    image = Image.fromarray(quarter[0, 0].numpy(), mode="F")
    return np.array(image.resize((out_hw[1], out_hw[0]), BILINEAR), dtype=np.float32)


def procedural_background(rng: np.random.Generator, out_hw: tuple[int, int]) -> NDArray:
    """A surface without assets: a base colour with smooth, medium and fine variation,
    built at a quarter of the size and enlarged bilinearly."""
    quarter = _quarter(out_hw)
    values = torch.from_numpy(rng.random(3 + 12 + 64 + 1024).astype(np.float32))

    def up(coarse: torch.Tensor) -> torch.Tensor:
        return F.interpolate(coarse - 0.5, size=quarter, mode="bilinear", align_corners=False)

    surface = (
        0.10
        + 0.78 * values[:3].view(1, 3, 1, 1)
        + 0.24 * up(values[3:15].view(1, 3, 2, 2))
        + 0.16 * up(values[15:79].view(1, 1, 8, 8))
        + 0.06 * up(values[79:].view(1, 1, 32, 32))
    ).clamp(0, 1)
    small = (surface[0] * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    image = Image.fromarray(small.numpy()).resize((out_hw[1], out_hw[0]), BILINEAR)
    return np.array(image, dtype=np.float32)


def _shadow(mask: Image.Image, sigma: float) -> NDArray[np.float32]:
    """``mask`` (L, full size) blurred by ``sigma`` pixels, computed at half size."""
    small = mask.reduce(2).filter(ImageFilter.GaussianBlur(sigma / 2.0))
    return np.asarray(small.resize(mask.size, BILINEAR), dtype=np.float32) / 255.0


def _random_polygon(rng: np.random.Generator, out_hw: tuple[int, int], count: int, spill: float):
    height, width = out_hw
    low = [-spill * width, -spill * height]
    high = [(1 + spill) * width, (1 + spill) * height]
    return [tuple(p) for p in rng.uniform(low, high, size=(count, 2))]


def _background(
    config: CompositeConfig,
    bank: BackgroundBank | None,
    rng: np.random.Generator,
    out_hw: tuple[int, int],
) -> NDArray[np.float32]:
    height, width = out_hw
    if bank is not None and len(bank) and rng.random() < config.texture_prob:
        surface = bank.draw(rng, out_hw)
    else:
        surface = procedural_background(rng, out_hw)
    if rng.random() < config.panel_prob:  # a second surface: table edge, folder, box
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(_random_polygon(rng, out_hw, 4, 0.3), fill=255)
        weight = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.0)), np.float32)[..., None]
        weight /= 255.0
        other = rng.uniform(20, 235, size=3).astype(np.float32)
        grain = rng.standard_normal((height, width, 1), dtype=np.float32) * np.float32(3.0)
        surface += weight * (other + grain - surface)
    return surface


# --------------------------------------------------------------------------- composite


def _corners(nodes_px: NDArray) -> NDArray[np.float64]:
    """Top-left, top-right, bottom-right and bottom-left corner nodes."""
    return np.array([nodes_px[0, 0], nodes_px[0, -1], nodes_px[-1, -1], nodes_px[-1, 0]])


def _draw_neighbor(
    surface: NDArray[np.float32],
    neighbor: tuple[Image.Image | NDArray[np.uint8], NDArray, NDArray],
    placed: NDArray,
    out_hw: tuple[int, int],
    config: CompositeConfig,
    rng: np.random.Generator,
) -> NDArray | None:
    """Draw the facing page along the left or right edge of the placed page (corners
    ``placed``, output px) onto ``surface``; return the fold as two output points."""
    image, grid, valid = neighbor
    if not valid.all():
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    width, height = image.size
    other = _corners((grid.astype(np.float64) + 1.0) / 2.0 * np.array([width, height]))
    tl, tr, br, bl = placed
    share = rng.uniform(*config.neighbor_width)
    right = rng.random() < 0.5
    if right:  # facing page on the right: its left part shows
        fold = np.array([tr, br])
        far = fold + share * np.array([tr - tl, br - bl])
        source = [other[0], other[0] + share * (other[1] - other[0]),
                  other[3] + share * (other[2] - other[3]), other[3]]  # fmt: skip
    else:  # facing page on the left: its right part shows
        fold = np.array([tl, bl])
        far = fold + share * np.array([tl - tr, bl - br])
        source = [other[1] - share * (other[1] - other[0]), other[1],
                  other[2], other[2] - share * (other[2] - other[3])]  # fmt: skip
    scale, shift = rng.uniform(-config.neighbor_fold, config.neighbor_fold, size=2)
    middle = far.mean(axis=0)
    far = middle + (far - middle) * (1.0 + scale) + shift * (bl - tl)
    target = np.array([tr, far[0], far[1], br] if right else [far[0], tl, bl, far[1]])
    if abs(polygon_area(target)) < 1.0:
        return None
    box = polygon_box(target, out_hw)
    if box is None:
        return fold
    transform = corner_homography(np.array(source), target)
    step = math.sqrt(abs(polygon_area(np.array(source))) / abs(polygon_area(target)))
    pixels = warp_homography(image, transform, out_hw, step=step, box=box)
    x0, y0, x1, y1 = box
    alpha = page_alpha(target, box, rng.uniform(0.0, config.edge_trim))[..., None]
    region = surface[y0:y1, x0:x1]
    gain = np.float32(rng.uniform(*config.page_gain))
    region += alpha * (pixels.astype(np.float32) * gain - region)
    return fold


def _fold_shade(
    fold: NDArray, page_width: float, out_hw: tuple[int, int], strength: float, rng
) -> NDArray[np.float32]:
    """A shadow along the fold line: 1 far from it, down to 1 - strength on it."""
    height, width = out_hw
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) + np.float32(0.5)
    a, b = fold.astype(np.float32)
    direction = (b - a) / max(float(np.hypot(*(b - a))), 1e-6)
    distance = np.abs((xx - a[0]) * direction[1] - (yy - a[1]) * direction[0])
    sigma = np.float32(max(rng.uniform(0.01, 0.05) * page_width, 1.0))
    return (1.0 - np.float32(strength) * np.exp(-distance / sigma)).astype(np.float32)


def compose(
    image: Image.Image | NDArray[np.uint8],
    nodes: NDArray,
    valid: NDArray,
    out_hw: tuple[int, int],
    config: CompositeConfig,
    rng: np.random.Generator,
    bank: BackgroundBank | None = None,
    neighbor: tuple[Image.Image | NDArray[np.uint8], NDArray, NDArray] | None = None,
) -> tuple[torch.Tensor, NDArray[np.float32], int] | None:
    """The composite of one sample: (3, H, W) uint8 image, target grid in output
    flattener values, quarter turn index. ``nodes`` are the sample's edge-aligned grid
    values with respect to ``image``. ``neighbor`` (image, grid values, valid) is another
    sample whose page is drawn as the facing page.

    Returns None when the sample cannot be placed exactly: an invalid node, or a page
    outline that leaves the source image (its pixels would be missing).
    """
    if not valid.all():
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    width, height = image.size
    nodes_px = (nodes.astype(np.float64) + 1.0) / 2.0 * np.array([width, height])
    page = outline(nodes_px)
    if (page < -0.5).any() or (page[:, 0] > width + 0.5).any() or (page[:, 1] > height + 0.5).any():
        return None
    out_h, out_w = out_hw
    transform, turn = plan_placement(nodes_px, out_hw, config, rng)
    placed = apply_homography(transform, page)
    box = polygon_box(placed, out_hw)
    surface = _background(config, bank, rng, out_hw)
    fold = None
    if neighbor is not None:
        corners = apply_homography(transform, _corners(nodes_px))
        fold = _draw_neighbor(surface, neighbor, corners, out_hw, config, rng)
    alpha = np.zeros(out_hw, dtype=np.float32)
    if box is not None:
        left, top, right, bottom = box
        step = math.sqrt(abs(polygon_area(page)) / max(abs(polygon_area(placed)), 1e-9))
        pixels = warp_homography(image, transform, out_hw, step=step, box=box)
        alpha[top:bottom, left:right] = page_alpha(placed, box, rng.uniform(0.0, config.edge_trim))
    if rng.random() < config.drop_shadow_prob:  # the page's shadow on the background
        shift = rng.normal(0.0, 0.012, size=2) * [out_w, out_h]
        moved = np.roll(alpha, (round(shift[1]), round(shift[0])), axis=(0, 1))
        mask = Image.fromarray(np.rint(moved * 255.0).astype(np.uint8))
        shade = _shadow(mask, rng.uniform(3.0, 18.0))
        surface *= (1.0 - rng.uniform(0.1, 0.45) * shade)[..., None]
    gain = rng.uniform(*config.page_gain)
    result = surface
    if box is not None:
        region = result[top:bottom, left:right]
        weight = alpha[top:bottom, left:right, None]
        region += weight * (pixels.astype(np.float32) * np.float32(gain) - region)
    if fold is not None:
        page_width = float(np.hypot(*(corners[1] - corners[0])))
        strength = rng.uniform(*config.fold_shadow)
        result *= _fold_shade(fold, page_width, out_hw, strength, rng)[..., None]
    light = rng.uniform(1.0 - config.illumination, 1.0 + 0.3 * config.illumination, (3, 3))
    field = smooth_field(light, out_hw)
    if rng.random() < config.cast_shadow_prob:  # an object between the light and the scene
        mask = Image.new("L", (out_w, out_h), 0)
        corners = int(rng.integers(3, 6))
        ImageDraw.Draw(mask).polygon(_random_polygon(rng, out_hw, corners, 0.2), fill=255)
        field *= 1.0 - rng.uniform(0.15, 0.45) * _shadow(mask, rng.uniform(2.0, 25.0))
    balance = rng.uniform(1.0 - config.white_balance, 1.0 + config.white_balance, size=3)
    result *= field[..., None] * balance.astype(np.float32)
    np.rint(result, out=result)
    result = np.clip(result, 0, 255, out=result).astype(np.uint8)
    if rng.random() < config.jpeg_prob:
        buffer = io.BytesIO()
        quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
        Image.fromarray(result).save(buffer, format="JPEG", quality=quality)
        with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
            result = np.asarray(decoded.convert("RGB"))
    target = 2.0 * apply_homography(transform, nodes_px) / np.array([out_w, out_h]) - 1.0
    tensor = torch.from_numpy(np.ascontiguousarray(result)).permute(2, 0, 1).contiguous()
    return tensor, target.astype(np.float32), turn
