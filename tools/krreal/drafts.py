"""Draft labels and transcript templates, with hooks for prefilling them.

A label provider is ``provider(photo_path, info) -> dict``. ``info`` is
``kitlib.photo_info`` plus ``name`` (a ``kitlib.PhotoName``). The returned dict may hold

* ``pages``: a list of pages, each with ``quad`` (TL, TR, BR, BL in the page's reading
  orientation) and optionally ``curved_edges``, ``text_polygons`` and
  ``reference_lines`` as in ``label.schema.json``;
* ``flags``, ``cut_off_sides`` and ``notes``.

Points are pixels of the EXIF-applied photo with (0, 0) at the centre of the top-left
pixel. Curve end points are replaced by the quad corners. A proposal that fails the
label checks is dropped and the draft keeps only the photo data.

The built-in label provider ``none`` prefills no geometry. The planned scanner provider
runs ``flattener.scan`` on the photo, converts the final grid to a quad and curved edges
with ``pages_from_grid`` and adds text polygons and baselines from OCR word boxes; it is
passed as ``module:function`` or ``path/to/file.py:function`` once it exists.

A text provider is ``provider(photo_path, info) -> list[str]`` (lines in reading order)
and fills the body of a transcript template. Built in: ``none`` and ``tesseract``
(Tesseract with ``kor+eng`` on the EXIF-applied photo).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import string
import sys
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from kitlib import PhotoName, now
from labelcheck import EDGE_CORNERS, check_label, errors, page_quarter, page_rotation_deg
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "transcript-template.txt"
Provider = Callable[[Path, dict[str, Any]], Any]


def load_hook(spec: str) -> Provider:
    """``package.module:function`` or ``path/to/file.py:function``."""
    target, _, attribute = spec.rpartition(":")
    if not target or not attribute:
        raise ValueError(f"hook {spec!r}: expected module:function or file.py:function")
    if target.endswith(".py"):
        path = Path(target).expanduser().resolve()
        name = "krreal_hook_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"hook {spec!r}: cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    function = getattr(module, attribute, None)
    if not callable(function):
        raise ValueError(f"hook {spec!r}: {attribute} is not a function")
    return function


# Label drafts ------------------------------------------------------------------------


def no_geometry(photo: Path, info: dict[str, Any]) -> dict[str, Any]:
    return {}


LABEL_PROVIDERS: dict[str, Provider] = {"none": no_geometry}


def label_provider(spec: str) -> Provider:
    return LABEL_PROVIDERS[spec] if spec in LABEL_PROVIDERS else load_hook(spec)


def _points(points: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points]


def _page(page: dict[str, Any], order: int) -> dict[str, Any]:
    quad = _points(page["quad"])
    curves = {}
    for side, curve in (page.get("curved_edges") or {}).items():
        start, end = EDGE_CORNERS[side]
        curves[side] = [quad[start], *_points(curve)[1:-1], quad[end]]
    angle = page_rotation_deg(quad)
    return {
        "reading_order": order,
        "quad": quad,
        "page_rotation_deg": angle,
        "page_quarter": page_quarter(angle),
        "curved_edges": curves,
        "text_polygons": [
            {"kind": polygon.get("kind", "text"), "points": _points(polygon["points"])}
            for polygon in page.get("text_polygons") or []
        ],
        "reference_lines": [
            {"kind": line.get("kind", "baseline"), "points": _points(line["points"])}
            for line in page.get("reference_lines") or []
        ],
    }


def skeleton_label(info: dict[str, Any], name: PhotoName) -> dict[str, Any]:
    """A draft label with the photo data and no geometry."""
    return {
        "schema": "krreal-label-1",
        "photo": {key: info[key] for key in ("file", "sha256", "bytes", "raw_size", "size")}
        | {"exif_orientation": info["exif_orientation"]},
        "capture": {
            "phone_id": name.phone_id,
            "stratum": name.stratum,
            "source_id": name.source_id,
            "seq": name.seq,
            "device": dict(info["device"]),
            "exif": dict(info["exif"]),
        },
        "coordinates": {
            "frame": "exif_applied",
            "origin": "pixel_center",
            "x": "right",
            "y": "down",
            "units": "px",
        },
        "page_count": 0,
        "pages": [],
        "flags": [],
        "cut_off_sides": [],
        "notes": "",
        "review": {
            "status": "draft",
            "labeler": "",
            "labeled_at": None,
            "checker": "",
            "checked_at": None,
            "seconds_spent": 0,
            "tool": "krreal.py draft",
        },
        "draft": None,
    }


def build_draft(
    info: dict[str, Any],
    name: PhotoName,
    *,
    provider_name: str,
    proposal: dict[str, Any] | None,
    kit: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """The draft label for one photo and the reasons a proposal was dropped."""
    label = skeleton_label(info, name)
    fields: list[str] = []
    proposal = proposal or {}
    try:
        pages = [_page(page, order) for order, page in enumerate(proposal.get("pages", []), 1)]
    except (KeyError, TypeError, ValueError) as error:
        pages, proposal = [], {}
        dropped = [f"{name.file}: proposal dropped ({error!r})"]
    else:
        dropped = []
    if pages:
        label["pages"], label["page_count"] = pages, len(pages)
        fields.append("quad")
        for key in ("curved_edges", "text_polygons", "reference_lines"):
            if any(page[key] for page in pages):
                fields.append(key)
    for key in ("flags", "cut_off_sides", "notes"):
        if proposal.get(key):
            label[key] = proposal[key]
            fields.append(key)
    label["draft"] = {"provider": provider_name, "created_at": now(), "fields": fields}
    problems = errors(check_label(label, kit=kit))
    if problems:
        dropped.append(
            f"{name.file}: proposal dropped: "
            + "; ".join(f"{p.where} {p.message}" for p in problems)
        )
        label = skeleton_label(info, name)
        label["draft"] = {"provider": provider_name, "created_at": now(), "fields": []}
    return label, dropped


def _resample(line: np.ndarray, samples: int) -> np.ndarray:
    steps = np.hypot(*np.diff(line, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    targets = np.linspace(0.0, along[-1], samples)
    points = np.stack([np.interp(targets, along, line[:, k]) for k in (0, 1)], axis=1)
    points[0], points[-1] = line[0], line[-1]
    return points


def pages_from_grid(
    grid: Any,
    *,
    image_size: tuple[int, int] | None = None,
    samples: int = 9,
    straight_px: float = 1.5,
) -> dict[str, Any]:
    """Quad and curved edges of one page from an edge-aligned grid ``(Gh, Gw, 2)``.

    Row 0 is the page top and column 0 the page left, so the corner nodes are TL, TR,
    BR, BL. Values are EXIF-applied pixels, or flattener values in [-1, 1] when
    ``image_size`` = (width, height) is given. Each border is resampled by arc length to
    ``samples`` points (at least 7); a border whose points stay within ``straight_px``
    of its chord is left out as straight.
    """
    if samples < 7:
        raise ValueError("curved edges need at least 7 points")
    nodes = np.asarray(grid, dtype=np.float64)
    if nodes.ndim != 3 or nodes.shape[2] != 2 or min(nodes.shape[:2]) < 2:
        raise ValueError(f"expected a (Gh, Gw, 2) grid, got {nodes.shape}")
    if image_size is not None:
        width, height = image_size
        nodes = np.stack(
            [(nodes[..., 0] + 1) * width / 2 - 0.5, (nodes[..., 1] + 1) * height / 2 - 0.5],
            axis=-1,
        )
    if not np.isfinite(nodes).all():
        raise ValueError("the grid has non-finite values")
    quad = [nodes[0, 0], nodes[0, -1], nodes[-1, -1], nodes[-1, 0]]
    borders = {"top": nodes[0], "right": nodes[:, -1], "bottom": nodes[-1], "left": nodes[:, 0]}
    curves = {}
    for side, border in borders.items():
        points = _resample(border, samples)
        chord = points[-1] - points[0]
        length = float(np.hypot(*chord))
        relative = points - points[0]
        offsets = np.abs(chord[0] * relative[:, 1] - chord[1] * relative[:, 0])
        offsets /= max(length, 1e-9)
        if float(offsets.max()) > straight_px:
            curves[side] = points.tolist()
    return {"quad": [point.tolist() for point in quad], "curved_edges": curves}


# Transcripts -------------------------------------------------------------------------


def no_text(photo: Path, info: dict[str, Any]) -> list[str]:
    return []


def tesseract_lines(photo: Path, info: dict[str, Any]) -> list[str]:
    """Tesseract ``kor+eng`` on the EXIF-applied photo (needs the ``kor`` language data)."""
    source = HERE.parents[1] / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from flattener.benchmark.ocr import Tesseract

    with Image.open(photo) as image:
        upright = ImageOps.exif_transpose(image).convert("RGB")
    return Tesseract().image_to_string(upright, "-l kor+eng").splitlines()


TEXT_PROVIDERS: dict[str, Provider] = {"none": no_text, "tesseract": tesseract_lines}


def text_provider(spec: str) -> Provider:
    return TEXT_PROVIDERS[spec] if spec in TEXT_PROVIDERS else load_hook(spec)


def clean_lines(lines: Sequence[str] | str) -> list[str]:
    """NFC, no trailing spaces or form feeds, at most one blank line in a row."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    cleaned: list[str] = []
    for line in lines:
        line = unicodedata.normalize("NFC", line.replace("\f", "")).rstrip()
        if line or (cleaned and cleaned[-1]):
            cleaned.append(line)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def render_transcript(
    *, photo: str, sha256: str, page: int, draft: str, lines: Sequence[str]
) -> str:
    template = string.Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    head = template.substitute(photo=photo, sha256=sha256, page=page, draft=draft)
    body = "\n".join(clean_lines(lines))
    return head + (body + "\n" if body else "")


def pick_for_transcripts(names: Sequence[PhotoName], kit: dict[str, Any]) -> list[PhotoName]:
    """The ``transcribe`` quota of each stratum, in a fixed order within the stratum."""
    chosen: list[PhotoName] = []
    for stratum in kit["strata"]:
        pool = sorted(
            (name for name in names if name.stratum == stratum["code"]),
            key=lambda name: hashlib.sha256(f"{kit['split_salt']}:{name.file}".encode()).digest(),
        )
        chosen += pool[: stratum["transcribe"]]
    return chosen
