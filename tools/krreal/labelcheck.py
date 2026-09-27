"""Checks for KR-Real label files and transcripts.

``validate_schema`` runs the JSON Schema subset that ``label.schema.json`` uses, so the
schema file stays the single description of the label structure. ``check_label`` adds
the geometry and review rules the schema cannot state. ``parse_transcript`` and
``check_transcript`` read the plain-text transcript files.

Every check returns a list of ``Issue``; an ``error`` blocks the photo from counting as
labelled or transcribed, a ``warning`` is only shown.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "label.schema.json"
KIT_PATH = HERE / "kit.json"

LABEL_SCHEMA = "krreal-label-1"
TRANSCRIPT_MAGIC = "# krreal-transcript-1"
TRANSCRIPT_STATUSES = ("draft", "transcribed", "checked")
REVIEW_STATUSES = ("draft", "labeled", "checked")
EDGE_CORNERS = {"top": (0, 1), "right": (1, 2), "bottom": (3, 2), "left": (0, 3)}
CORNER_TOLERANCE_PX = 0.5
OUTSIDE_MARGIN = 0.5
MIN_EDGE_PX = 4.0
REFERENCE_LINES = (3, 5)
_ANNOTATIONS = {"$schema", "$id", "$comment", "title", "description", "$defs", "default"}


@dataclass(frozen=True)
class Issue:
    level: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


def errors(issues: Sequence[Issue]) -> list[Issue]:
    return [issue for issue in issues if issue.level == "error"]


def load_json(path: Path) -> Any:
    """Strict JSON: NaN and infinities are refused."""

    def refuse(name: str) -> None:
        raise ValueError(f"{path}: {name} is not valid JSON")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=refuse)


def load_kit() -> dict[str, Any]:
    return load_json(KIT_PATH)


def load_schema() -> dict[str, Any]:
    return load_json(SCHEMA_PATH)


# JSON Schema subset ------------------------------------------------------------------


class SchemaError(ValueError):
    """The schema uses a keyword this validator does not implement."""


def _is_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise SchemaError(f"unknown type {name!r}")


def _resolve(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise SchemaError(f"only local references are supported: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate_schema(instance: Any, schema: dict[str, Any] | None = None) -> list[str]:
    """Errors of ``instance`` against the schema subset, as ``"<pointer>: <message>"``."""
    root = load_schema() if schema is None else schema
    found: list[str] = []
    _validate(instance, root, root, "", found)
    return found


def _validate(value: Any, schema: Any, root: dict, pointer: str, found: list[str]) -> None:
    where = pointer or "/"
    if schema is True:
        return
    if schema is False:
        found.append(f"{where}: no value is allowed here")
        return
    unknown = (
        set(schema)
        - _ANNOTATIONS
        - {
            "$ref",
            "type",
            "enum",
            "const",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "prefixItems",
            "minItems",
            "maxItems",
            "uniqueItems",
            "minimum",
            "maximum",
            "pattern",
            "minLength",
            "maxLength",
            "anyOf",
        }
    )
    if unknown:
        raise SchemaError(f"unsupported keywords at {where}: {sorted(unknown)}")
    if "$ref" in schema:
        _validate(value, _resolve(root, schema["$ref"]), root, pointer, found)
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, name) for name in names):
            found.append(f"{where}: expected {' or '.join(names)}")
            return
    if "const" in schema and value != schema["const"]:
        found.append(f"{where}: must be {json.dumps(schema['const'], ensure_ascii=False)}")
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if "anyOf" in schema:
        if not any(_passes(value, option, root, pointer) for option in schema["anyOf"]):
            found.append(f"{where}: matches none of the allowed forms")
    if _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            found.append(f"{where}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            found.append(f"{where}: above {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            found.append(f"{where}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            found.append(f"{where}: longer than {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            found.append(f"{where}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, list):
        _validate_array(value, schema, root, pointer, found)
    if isinstance(value, dict):
        _validate_object(value, schema, root, pointer, found)


def _passes(value: Any, schema: Any, root: dict, pointer: str) -> bool:
    trial: list[str] = []
    _validate(value, schema, root, pointer, trial)
    return not trial


def _validate_array(value: list, schema: dict, root: dict, pointer: str, found: list) -> None:
    where = pointer or "/"
    if "minItems" in schema and len(value) < schema["minItems"]:
        found.append(f"{where}: needs at least {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        found.append(f"{where}: allows at most {schema['maxItems']} items")
    if schema.get("uniqueItems"):
        seen = [json.dumps(item, sort_keys=True) for item in value]
        if len(set(seen)) != len(seen):
            found.append(f"{where}: items repeat")
    prefix = schema.get("prefixItems", [])
    for index, item in enumerate(value):
        item_pointer = f"{pointer}/{index}"
        if index < len(prefix):
            _validate(item, prefix[index], root, item_pointer, found)
        elif "items" in schema:
            _validate(item, schema["items"], root, item_pointer, found)


def _validate_object(value: dict, schema: dict, root: dict, pointer: str, found: list) -> None:
    where = pointer or "/"
    for key in schema.get("required", []):
        if key not in value:
            found.append(f"{where}: missing {key!r}")
    properties = schema.get("properties", {})
    for key, item in value.items():
        if key in properties:
            _validate(item, properties[key], root, f"{pointer}/{key}", found)
        elif schema.get("additionalProperties") is False:
            found.append(f"{where}: unknown field {key!r}")


# Geometry ----------------------------------------------------------------------------


def signed_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace area in image axes (y down): positive when the points go clockwise on
    screen, as TL, TR, BR, BL do."""
    total = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def page_rotation_deg(quad: Sequence[Sequence[float]]) -> float:
    """Clockwise angle of the page's left-to-right direction, in (-180, 180].

    The direction is the sum of the top (TL to TR) and bottom (BL to BR) edge vectors.
    """
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = quad
    dx = (trx - tlx) + (brx - blx)
    dy = (try_ - tly) + (bry - bly)
    angle = math.degrees(math.atan2(dy, dx))
    angle = 180.0 if angle <= -180.0 else angle
    return round(angle, 2)


def page_quarter(angle_deg: float) -> int:
    """Nearest clockwise quarter turn: floor(angle / 90 + 0.5) mod 4."""
    return int(math.floor(angle_deg / 90.0 + 0.5)) % 4


def _cross(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def quad_problems(quad: Sequence[Sequence[float]]) -> list[str]:
    """Why a quad is unusable: too short an edge, mirrored order or not convex."""
    problems = []
    edges = [_distance(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    if min(edges) < MIN_EDGE_PX:
        problems.append(f"an edge is shorter than {MIN_EDGE_PX:g} px")
    area = signed_area(quad)
    if area <= 0:
        problems.append("corners go counter-clockwise on screen; expected TL, TR, BR, BL")
    turns = [_cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) for i in range(4)]
    if area > 0 and not all(turn > 0 for turn in turns):
        problems.append("the quad is not convex")
    return problems


def curve_problems(
    curve: Sequence[Sequence[float]], start: Sequence[float], end: Sequence[float]
) -> list[str]:
    """A curved edge must start and end on its corners and run along the chord."""
    problems = []
    if _distance(curve[0], start) > CORNER_TOLERANCE_PX:
        problems.append("the first point is not the start corner")
    if _distance(curve[-1], end) > CORNER_TOLERANCE_PX:
        problems.append("the last point is not the end corner")
    chord = (end[0] - start[0], end[1] - start[1])
    length = math.hypot(*chord)
    if length < MIN_EDGE_PX:
        return problems + ["the corners are too close"]
    steps = [((p[0] - start[0]) * chord[0] + (p[1] - start[1]) * chord[1]) / length for p in curve]
    if any(b <= a for a, b in zip(steps, steps[1:], strict=False)):
        problems.append("points do not advance from the start corner to the end corner")
    offsets = [abs(_cross(start, end, p)) / length for p in curve]
    if max(offsets) > 0.5 * length:
        problems.append("a point lies further from the chord than half its length")
    return problems


# Labels ------------------------------------------------------------------------------


def check_label(
    label: Any,
    *,
    kit: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
    file_name: str | None = None,
) -> list[Issue]:
    """Schema errors plus the geometry and review rules of one label.

    A ``draft`` label only needs to pass the schema and the geometry of what it has;
    ``labeled`` and ``checked`` labels also need every required part.
    """
    kit = load_kit() if kit is None else kit
    issues = [Issue("error", "schema", message) for message in validate_schema(label, schema)]
    if issues:
        return issues
    photo, capture, review = label["photo"], label["capture"], label["review"]
    status = review["status"]
    complete = status in ("labeled", "checked")
    if file_name is not None and photo["file"] != file_name:
        issues.append(Issue("error", "photo.file", f"label is for {photo['file']}"))
    issues += _check_name(photo["file"], capture, kit)
    width, height = photo["size"]
    raw_w, raw_h = photo["raw_size"]
    swapped = photo["exif_orientation"] >= 5
    if (width, height) != ((raw_h, raw_w) if swapped else (raw_w, raw_h)):
        issues.append(Issue("error", "photo.size", "size does not follow raw_size and EXIF"))
    flags = set(label["flags"])
    pages = label["pages"]
    if label["page_count"] != len(pages):
        issues.append(Issue("error", "page_count", f"{len(pages)} pages are labelled"))
    if complete:
        if ("no_page" in flags) != (label["page_count"] == 0):
            issues.append(Issue("error", "flags", "no_page goes with page_count 0, only then"))
        if ("two_page" in flags) != (label["page_count"] == 2):
            issues.append(Issue("error", "flags", "two_page goes with page_count 2, only then"))
        if ("cut_off" in flags) != bool(label["cut_off_sides"]):
            issues.append(Issue("error", "cut_off_sides", "cut_off needs its sides, only then"))
        if capture["stratum"] == "spread" and label["page_count"] != 2:
            issues.append(Issue("warning", "page_count", "a spread photo usually has 2 pages"))
    orders = sorted(page["reading_order"] for page in pages)
    if orders != list(range(1, len(pages) + 1)):
        issues.append(Issue("error", "pages", "reading_order must be 1..n once each"))
    for index, page in enumerate(pages):
        issues += _check_page(page, f"pages[{index}]", (width, height), capture, complete)
    issues += _check_review(review)
    return issues


def _check_name(file_name: str, capture: dict[str, Any], kit: dict[str, Any]) -> list[Issue]:
    match = re.fullmatch(kit["name_pattern"], file_name)
    if match is None:
        return [Issue("error", "photo.file", f"name does not follow {kit['name_example']}")]
    phone, stratum, source, seq = match.groups()
    expected = {"phone_id": phone, "stratum": stratum, "source_id": source, "seq": int(seq)}
    wrong = [key for key, value in expected.items() if capture[key] != value]
    if wrong:
        return [Issue("error", "capture", f"{', '.join(wrong)} differ from the file name")]
    return []


def _check_page(
    page: dict[str, Any],
    where: str,
    size: tuple[int, int],
    capture: dict[str, Any],
    complete: bool,
) -> list[Issue]:
    issues = []
    width, height = size
    quad = page["quad"]
    points = [*quad]
    for curve in page["curved_edges"].values():
        points += curve
    for polygon in page["text_polygons"]:
        points += polygon["points"]
    for line in page["reference_lines"]:
        points += line["points"]
    low_x, high_x = -OUTSIDE_MARGIN * width, (1 + OUTSIDE_MARGIN) * width
    low_y, high_y = -OUTSIDE_MARGIN * height, (1 + OUTSIDE_MARGIN) * height
    if any(not (low_x <= x <= high_x and low_y <= y <= high_y) for x, y in points):
        issues.append(Issue("error", where, "a point lies far outside the image"))
    for problem in quad_problems(quad):
        issues.append(Issue("error", f"{where}.quad", problem))
    angle = page_rotation_deg(quad)
    if abs(((page["page_rotation_deg"] - angle) + 180.0) % 360.0 - 180.0) > 0.05:
        issues.append(Issue("error", f"{where}.page_rotation_deg", f"quad gives {angle}"))
    if page["page_quarter"] != page_quarter(angle):
        issues.append(Issue("error", f"{where}.page_quarter", f"quad gives {page_quarter(angle)}"))
    for side, curve in page["curved_edges"].items():
        start, end = (quad[i] for i in EDGE_CORNERS[side])
        for problem in curve_problems(curve, start, end):
            issues.append(Issue("error", f"{where}.curved_edges.{side}", problem))
    for index, polygon in enumerate(page["text_polygons"]):
        if abs(signed_area(polygon["points"])) < 1.0:
            issues.append(Issue("error", f"{where}.text_polygons[{index}]", "has no area"))
    for index, line in enumerate(page["reference_lines"]):
        pts = line["points"]
        length = sum(_distance(a, b) for a, b in zip(pts, pts[1:], strict=False))
        if length < 2 * MIN_EDGE_PX:
            issues.append(Issue("error", f"{where}.reference_lines[{index}]", "is too short"))
    if complete and capture["stratum"] != "spread":
        if not page["text_polygons"]:
            issues.append(Issue("error", f"{where}.text_polygons", "needs a text polygon"))
        low, high = REFERENCE_LINES
        count = len(page["reference_lines"])
        if not low <= count <= high:
            issues.append(Issue("error", f"{where}.reference_lines", f"needs {low}-{high} lines"))
    return issues


def _check_review(review: dict[str, Any]) -> list[Issue]:
    issues = []
    status = review["status"]
    if status in ("labeled", "checked"):
        if not review["labeler"].strip():
            issues.append(Issue("error", "review.labeler", "needs the labeller's name"))
        if not review["labeled_at"]:
            issues.append(Issue("error", "review.labeled_at", "needs the labelling time"))
    if status == "checked":
        checker = review["checker"].strip()
        if not checker or not review["checked_at"]:
            issues.append(Issue("error", "review.checker", "needs the checker and time"))
        elif checker == review["labeler"].strip():
            issues.append(Issue("error", "review.checker", "the checker must be another person"))
    return issues


# Transcripts -------------------------------------------------------------------------

TRANSCRIPT_FIELDS = ("photo", "sha256", "page", "draft", "status", "transcribed_by", "checked_by")


@dataclass
class Transcript:
    header: dict[str, str]
    lines: list[str]


def parse_transcript(text: str) -> Transcript:
    """Header lines ``# key: value`` up to a line ``---``; every later line is text."""
    rows = text.replace("\r\n", "\n").split("\n")
    if not rows or rows[0].strip() != TRANSCRIPT_MAGIC:
        raise ValueError(f"the first line must be {TRANSCRIPT_MAGIC!r}")
    header: dict[str, str] = {}
    for number, row in enumerate(rows[1:], start=2):
        if row.strip() == "---":
            body = rows[number:]
            while body and not body[-1].strip():
                body.pop()
            return Transcript(header, [line.rstrip() for line in body])
        match = re.fullmatch(r"#\s*([a-z0-9_]+)\s*:(.*)", row)
        if match and match.group(1) in TRANSCRIPT_FIELDS:
            header[match.group(1)] = match.group(2).strip()
        elif row.strip() and not row.startswith("#"):
            raise ValueError(f"line {number}: header lines start with '#'")
    raise ValueError("the '---' line before the text is missing")


def check_transcript(
    transcript: Transcript,
    *,
    photo_file: str | None = None,
    photo_sha256: str | None = None,
    markers: Sequence[str] = (),
) -> list[Issue]:
    issues = []
    header = transcript.header
    missing = [field for field in TRANSCRIPT_FIELDS if field not in header]
    if missing:
        issues.append(Issue("error", "header", f"missing {', '.join(missing)}"))
        return issues
    if photo_file is not None and header["photo"] != photo_file:
        issues.append(Issue("error", "photo", f"the header names {header['photo']}"))
    if photo_sha256 is not None and header["sha256"] != photo_sha256:
        issues.append(Issue("error", "sha256", "does not match the photo"))
    status = header["status"]
    if status not in TRANSCRIPT_STATUSES:
        issues.append(Issue("error", "status", f"use one of {', '.join(TRANSCRIPT_STATUSES)}"))
        return issues
    for number, line in enumerate(transcript.lines, start=1):
        if unicodedata.normalize("NFC", line) != line:
            issues.append(Issue("error", f"text line {number}", "is not NFC"))
        for marker in re.findall(r"\[\[[^\]]*\]\]", line):
            if marker not in markers:
                issues.append(Issue("error", f"text line {number}", f"unknown marker {marker}"))
    if status in ("transcribed", "checked"):
        if not any(line.strip() for line in transcript.lines):
            issues.append(Issue("error", "text", "is empty"))
        if not header["transcribed_by"]:
            issues.append(Issue("error", "transcribed_by", "needs the transcriber's name"))
    if status == "checked":
        checker = header["checked_by"]
        if not checker:
            issues.append(Issue("error", "checked_by", "needs the checker's name"))
        elif checker == header["transcribed_by"]:
            issues.append(Issue("error", "checked_by", "the checker must be another person"))
    return issues
