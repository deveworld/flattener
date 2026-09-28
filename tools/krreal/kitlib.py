"""Photo files of the KR-Real kit: names, EXIF, SHA-256, shot list, ingest and manifest.

The photo set lives under a git-ignored data root (default ``outputs/krreal/v1``)::

    incoming/p1/, incoming/p2/   photos as copied from each phone
    shotlist.csv                 the shot plan, with the phone's file name per shot
    photos/                      kit copies under the naming rule (bytes unchanged)
    drafts/<stem>.json           prefilled labels (krreal.py draft)
    labels/<stem>.json           labels saved by labeler.html
    transcripts/<stem>.txt       transcripts (krreal.py transcripts)
    sources.csv                  optional: source_id,kind,public,note
    manifest.json                krreal.py manifest

Photo names are ``kr1-<phone>-<stratum>-<source>-<shot>.jpg``, for example
``kr1-p1-curved-b03-007.jpg``: phone ``p1``-``p9``, a stratum code from ``kit.json``,
the book or document id (``b`` book, ``d`` single document, ``x00`` no document) and
the shot number of that phone (001-999).
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labelcheck import (
    Issue,
    check_label,
    check_transcript,
    errors,
    load_json,
    load_kit,
    load_schema,
    parse_transcript,
)
from PIL import Image, UnidentifiedImageError

MANIFEST_SCHEMA = "krreal-manifest-1"
SHOTLIST_FIELDS = ("phone", "shot", "stratum", "variant", "description", "source", "file")
_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825
_IFD0 = {"make": 0x010F, "model": 0x0110, "software": 0x0131}
_EXIF_TAGS = {
    "datetime_original": 0x9003,
    "exposure_s": 0x829A,
    "f_number": 0x829D,
    "iso": 0x8827,
    "focal_length_mm": 0x920A,
    "focal_length_35mm": 0xA405,
    "lens_model": 0xA434,
}


def now() -> str:
    return dt.datetime.now(dt.UTC).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Names -------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhotoName:
    file: str
    phone_id: str
    stratum: str
    source_id: str
    seq: int

    @property
    def stem(self) -> str:
        return self.file[: -len(".jpg")]


def photo_name(phone: str, stratum: str, source: str, shot: int) -> str:
    return f"kr1-{phone}-{stratum}-{source}-{shot:03d}.jpg"


def parse_name(file_name: str, kit: dict[str, Any] | None = None) -> PhotoName:
    """Split a kit photo name; ``ValueError`` names the broken part."""
    kit = load_kit() if kit is None else kit
    match = re.fullmatch(kit["name_pattern"], file_name)
    if match is None:
        raise ValueError(f"{file_name}: expected a name like {kit['name_example']}")
    phone, stratum, source, seq = match.groups()
    codes = [stratum_["code"] for stratum_ in kit["strata"]]
    if stratum not in codes:
        raise ValueError(f"{file_name}: stratum {stratum!r} is not one of {', '.join(codes)}")
    if int(seq) < 1:
        raise ValueError(f"{file_name}: shot numbers start at 001")
    if source.startswith("x") and source != "x00":
        raise ValueError(f"{file_name}: a photo without a document uses source x00")
    return PhotoName(file_name, phone, stratum, source, int(seq))


def strata_by_code(kit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {stratum["code"]: stratum for stratum in kit["strata"]}


def planned_counts(kit: dict[str, Any]) -> dict[str, int]:
    """Pilot photos per stratum: the shots of every phone in the shot list."""
    return {
        stratum["code"]: sum(len(shots) for shots in stratum["shots"].values())
        for stratum in kit["strata"]
    }


# Photo files -------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = str(value).strip("\x00 ").strip()
    return text[:200] or None


def _number(value: Any) -> float | None:
    if isinstance(value, tuple) and value:
        value = value[0]
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def photo_info(path: Path) -> dict[str, Any]:
    """File identity, pixel sizes and EXIF of one photo. GPS values are never read out;
    ``gps_present`` only says whether the file carries them."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            image_format = image.format
            raw_size = [int(image.size[0]), int(image.size[1])]
            frames = int(getattr(image, "n_frames", 1))
            exif = image.getexif()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"{path.name}: cannot read the image ({error})") from error
    orientation = exif.get(0x0112, 1)
    orientation = orientation if orientation in range(1, 9) else 1
    size = raw_size[::-1] if orientation >= 5 else raw_size
    sub = exif.get_ifd(_EXIF_IFD)
    info_exif: dict[str, Any] = {}
    for key, tag in _EXIF_TAGS.items():
        value = sub.get(tag)
        info_exif[key] = (
            _text(value)
            if key in ("datetime_original", "lens_model")
            else (_number(value) if value is not None else None)
        )
    info_exif["software"] = _text(exif.get(_IFD0["software"]))
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "format": image_format,
        "frames": frames,
        "raw_size": raw_size,
        "size": size,
        "exif_orientation": int(orientation),
        "megapixels": round(raw_size[0] * raw_size[1] / 1e6, 2),
        "device": {
            "make": _text(exif.get(_IFD0["make"])),
            "model": _text(exif.get(_IFD0["model"])),
        },
        "exif": info_exif,
        "exif_present": len(exif) > 0,
        "gps_present": len(exif.get_ifd(_GPS_IFD)) > 0,
    }


def photo_warnings(info: dict[str, Any]) -> list[str]:
    name = info["file"]
    found = []
    if info["format"] != "JPEG":
        found.append(f"{name}: format {info['format']} ({info['frames']} frames); expected JPEG")
    if not info["exif_present"]:
        found.append(f"{name}: no EXIF; the file may have been re-encoded on the way")
    elif info["device"]["model"] is None:
        found.append(f"{name}: EXIF has no camera model")
    if info["gps_present"]:
        found.append(f"{name}: carries GPS data; turn off location tags on the phone")
    return found


# Git safety --------------------------------------------------------------------------


def ensure_ignored(root: Path) -> None:
    """Refuse a data root inside the git work tree unless git ignores it."""
    root = Path(root).resolve()
    probe = root
    while not probe.exists():
        probe = probe.parent
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=probe if probe.is_dir() else probe.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    if top.returncode != 0:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", str(root / "probe.jpg")],
        cwd=top.stdout.strip(),
        capture_output=True,
        check=False,
    )
    if ignored.returncode == 1:
        raise SystemExit(
            f"{root} is inside the git work tree and not ignored; keep photos under outputs/"
        )


# Shot list and ingest ----------------------------------------------------------------


def shotlist_rows(kit: dict[str, Any]) -> list[dict[str, str]]:
    """The pilot shot plan: one row per photo, numbered per phone in stratum order."""
    rows = []
    for phone in kit["phones"]:
        shot = 0
        for stratum in kit["strata"]:
            for variant in stratum["shots"].get(phone, []):
                shot += 1
                rows.append(
                    {
                        "phone": phone,
                        "shot": str(shot),
                        "stratum": stratum["code"],
                        "variant": variant,
                        "description": stratum["variants"][variant],
                        "source": "x00" if variant == "none" else "",
                        "file": "",
                    }
                )
    return rows


def write_shotlist(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SHOTLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_shotlist(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(SHOTLIST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        return [{key: (row.get(key) or "").strip() for key in SHOTLIST_FIELDS} for row in reader]


def ingest(
    root: Path, rows: list[dict[str, str]], *, kit: dict[str, Any], dry_run: bool = False
) -> tuple[list[str], list[str]]:
    """Copy each shot-list row with a ``file`` into ``photos/`` under its kit name.

    Bytes are copied unchanged and compared by SHA-256. An existing kit photo is kept
    when it has the same bytes and refused when it differs. Returns (done, problems).
    """
    photos = root / "photos"
    done: list[str] = []
    problems: list[str] = []
    targets: dict[str, str] = {}
    for number, row in enumerate(rows, start=2):
        if not row["file"]:
            continue
        where = f"shotlist line {number}"
        if not re.fullmatch(r"[bdx][0-9]{2}", row["source"]):
            problems.append(f"{where}: source must look like b03, d01 or x00")
            continue
        if not row["shot"].isdigit():
            problems.append(f"{where}: shot must be a number")
            continue
        name = photo_name(row["phone"], row["stratum"], row["source"], int(row["shot"]))
        try:
            parse_name(name, kit)
        except ValueError as error:
            problems.append(f"{where}: {error}")
            continue
        source = Path(row["file"]).expanduser()
        if not source.is_absolute():
            source = root / "incoming" / row["phone"] / row["file"]
        suffix = source.suffix.lower()
        if suffix in (".heic", ".heif"):
            problems.append(f"{where}: {source.name} is HEIC; set the camera to JPEG and reshoot")
            continue
        if suffix not in (".jpg", ".jpeg"):
            problems.append(f"{where}: {source.name} is not a JPEG file")
            continue
        if not source.is_file():
            problems.append(f"{where}: {source} not found")
            continue
        if name in targets:
            problems.append(f"{where}: {name} is already taken by {targets[name]}")
            continue
        targets[name] = source.name
        digest = sha256_file(source)
        target = photos / name
        if target.exists():
            if sha256_file(target) != digest:
                problems.append(f"{where}: {name} exists with other bytes; not replaced")
            continue
        if dry_run:
            done.append(f"{source} -> {target}")
            continue
        photos.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        shutil.copyfile(source, partial)
        if sha256_file(partial) != digest:
            partial.unlink()
            problems.append(f"{where}: copy of {source.name} did not match; try again")
            continue
        partial.replace(target)
        done.append(f"{source} -> {target}")
    return done, problems


# Splits ------------------------------------------------------------------------------


def assign_splits(
    counts: dict[str, int], previous: dict[str, str], *, dev_fraction: float, salt: str
) -> dict[str, str]:
    """dev/test per source id (book-level split).

    Sources already in ``previous`` keep their split. New sources go in a fixed order
    (SHA-256 of salt and id) to dev while that brings the dev share closer to
    ``dev_fraction`` of the photos, and to test otherwise.
    """
    splits = {s: previous[s] for s in counts if previous.get(s) in ("dev", "test")}
    target = dev_fraction * sum(counts.values())
    dev = sum(counts[s] for s, split in splits.items() if split == "dev")
    order = sorted(
        (s for s in counts if s not in splits),
        key=lambda s: hashlib.sha256(f"{salt}:{s}".encode()).hexdigest(),
    )
    for source in order:
        if dev + counts[source] / 2 <= target:
            splits[source] = "dev"
            dev += counts[source]
        else:
            splits[source] = "test"
    return splits


# Manifest ----------------------------------------------------------------------------


def _read_sources(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return {
            (row.get("source_id") or "").strip(): {k: (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(stream)
        }


def _label_entry(path: Path, info: dict, kit: dict, schema: dict) -> tuple[dict, list[Issue]]:
    try:
        label = load_json(path)
    except (OSError, ValueError) as error:
        return {"file": path.name, "status": "invalid"}, [Issue("error", path.name, str(error))]
    issues = check_label(label, kit=kit, schema=schema, file_name=info["file"])
    if not errors(issues) and label["photo"]["sha256"] != info["sha256"]:
        issues.append(Issue("error", "photo.sha256", "label was made for other bytes"))
    status = "invalid" if errors(issues) else label["review"]["status"]
    entry = {"file": path.name, "sha256": sha256_file(path), "status": status}
    return entry, issues


def _transcript_entry(path: Path, info: dict, kit: dict) -> tuple[dict, list[Issue]]:
    try:
        transcript = parse_transcript(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return {"file": path.name, "status": "invalid"}, [Issue("error", path.name, str(error))]
    issues = check_transcript(
        transcript,
        photo_file=info["file"],
        photo_sha256=info["sha256"],
        markers=kit["transcript_markers"],
    )
    status = "invalid" if errors(issues) else transcript.header["status"]
    lines = sum(1 for line in transcript.lines if line.strip())
    entry = {"file": path.name, "sha256": sha256_file(path), "status": status, "lines": lines}
    return entry, issues


def build_manifest(root: Path, *, kit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Scan ``root`` and return the manifest (photos, labels, transcripts, summary)."""
    kit = load_kit() if kit is None else kit
    schema = load_schema()
    root = Path(root)
    previous_path = root / "manifest.json"
    previous: dict[str, str] = {}
    if previous_path.is_file():
        try:
            previous = load_json(previous_path).get("splits", {})
        except (OSError, ValueError):
            previous = {}
    problems: list[str] = []
    warnings: list[str] = []
    variants: dict[tuple[str, int], str] = {}
    if (root / "shotlist.csv").is_file():
        try:
            rows = read_shotlist(root / "shotlist.csv")
        except (OSError, ValueError) as error:
            warnings.append(f"shotlist.csv not used: {error}")
            rows = []
        for row in rows:
            if row["shot"].isdigit():
                variants[(row["phone"], int(row["shot"]))] = row["variant"]
    sources = _read_sources(root / "sources.csv")
    photos: list[dict[str, Any]] = []
    seen_sha: dict[str, str] = {}
    seen_shot: dict[tuple[str, int], str] = {}
    photo_dir = root / "photos"
    files = sorted(photo_dir.iterdir()) if photo_dir.is_dir() else []
    for path in files:
        if path.name.endswith(".part") or path.is_dir():
            continue
        try:
            name = parse_name(path.name, kit)
        except ValueError as error:
            problems.append(str(error))
            continue
        try:
            info = photo_info(path)
        except ValueError as error:
            problems.append(str(error))
            continue
        warnings += photo_warnings(info)
        if info["sha256"] in seen_sha:
            problems.append(f"{path.name}: same bytes as {seen_sha[info['sha256']]}")
        seen_sha[info["sha256"]] = path.name
        shot = (name.phone_id, name.seq)
        if shot in seen_shot:
            problems.append(f"{path.name}: shot {name.seq} of {name.phone_id} repeats")
        seen_shot[shot] = path.name
        public = sources.get(name.source_id, {}).get("public", "").lower() in ("yes", "y", "1")
        public = public and not info["gps_present"]  # a photo with GPS data stays private
        entry = {
            "file": path.name,
            "sha256": info["sha256"],
            "bytes": info["bytes"],
            "phone_id": name.phone_id,
            "stratum": name.stratum,
            "variant": variants.get(shot),
            "source_id": name.source_id,
            "seq": name.seq,
            "split": None,
            "visibility": "public_candidate" if public else "private",
            "raw_size": info["raw_size"],
            "size": info["size"],
            "exif_orientation": info["exif_orientation"],
            "megapixels": info["megapixels"],
            "device": info["device"],
            "datetime_original": info["exif"]["datetime_original"],
            "label": None,
            "transcript": None,
        }
        label_path = root / "labels" / f"{name.stem}.json"
        if label_path.is_file():
            entry["label"], issues = _label_entry(label_path, info, kit, schema)
            problems += [
                f"{label_path.name}: {issue.where}: {issue.message}" for issue in errors(issues)
            ]
        transcript_path = root / "transcripts" / f"{name.stem}.txt"
        if transcript_path.is_file():
            entry["transcript"], issues = _transcript_entry(transcript_path, info, kit)
            problems += [
                f"{transcript_path.name}: {issue.where}: {issue.message}"
                for issue in errors(issues)
            ]
        photos.append(entry)
    counts = Counter(photo["source_id"] for photo in photos)
    splits = assign_splits(
        dict(counts), previous, dev_fraction=kit["dev_fraction"], salt=kit["split_salt"]
    )
    for photo in photos:
        photo["split"] = splits[photo["source_id"]]
    total = len(photos)
    for source, count in sorted(counts.items()):
        if total >= 20 and count > 0.15 * total:
            warnings.append(f"source {source} has {count} of {total} photos (over 15%)")
    return {
        "schema": MANIFEST_SCHEMA,
        "set": kit["set"],
        "use": "evaluation_only",
        "visibility": "private",
        "created_at": now(),
        "split_rule": {
            "unit": "source_id",
            "dev_fraction": kit["dev_fraction"],
            "salt": kit["split_salt"],
            "dev_use": "threshold tuning",
            "test_use": "reported once per milestone",
        },
        "splits": dict(sorted(splits.items())),
        "photos": photos,
        "summary": summarize(photos, kit),
        "problems": problems,
        "warnings": warnings,
    }


def summarize(photos: list[dict[str, Any]], kit: dict[str, Any]) -> dict[str, Any]:
    planned = planned_counts(kit)
    by_stratum = Counter(photo["stratum"] for photo in photos)
    labels = Counter((photo["label"] or {}).get("status", "missing") for photo in photos)
    transcripts = Counter(
        (photo["transcript"] or {}).get("status") for photo in photos if photo["transcript"]
    )
    models: dict[str, set[str]] = {}
    for photo in photos:
        device = photo["device"]
        model = " ".join(part for part in (device["make"], device["model"]) if part) or "unknown"
        models.setdefault(photo["phone_id"], set()).add(model)
    known_models = {m for found in models.values() for m in found if m != "unknown"}
    missing = []
    short = {
        code: planned[code] - by_stratum.get(code, 0)
        for code in planned
        if by_stratum.get(code, 0) < planned[code]
    }
    if short:
        missing.append(
            "photos short: " + ", ".join(f"{code} {count}" for code, count in short.items())
        )
    if len(models) < 2 or len(known_models) < 2:
        missing.append("photos from 2 phones with EXIF camera models")
    done = labels.get("labeled", 0) + labels.get("checked", 0)
    if done < len(photos) or not photos:
        missing.append(f"labels: {done} of {len(photos)} labeled")
    checked = transcripts.get("checked", 0)
    if checked < kit["transcripts_checked_target"]:
        missing.append(f"transcripts: {checked} of {kit['transcripts_checked_target']} checked")
    return {
        "photos": len(photos),
        "planned": sum(planned.values()),
        "by_stratum": {
            code: {"have": by_stratum.get(code, 0), "planned": planned[code]} for code in planned
        },
        "by_phone": dict(sorted(Counter(photo["phone_id"] for photo in photos).items())),
        "devices": {phone: sorted(found) for phone, found in sorted(models.items())},
        "by_split": dict(sorted(Counter(photo["split"] for photo in photos).items())),
        "labels": dict(sorted(labels.items())),
        "transcripts": dict(sorted(transcripts.items())),
        "m1_complete": not missing,
        "m1_missing": missing,
    }


# Labeler ----------------------------------------------------------------------------

LABELER_PATH = Path(__file__).resolve().parent / "labeler.html"
_KIT_BLOCK = re.compile(r'(<script type="application/json" id="kit">\n)(.*?)(\n</script>)', re.S)


def embedded_kit(html: str) -> dict[str, Any]:
    """The kit JSON that labeler.html carries."""
    match = _KIT_BLOCK.search(html)
    if match is None:
        raise ValueError("labeler.html has no kit block")
    return json.loads(match.group(2))


def embed_kit(html: str, kit_text: str) -> str:
    """labeler.html with its kit block replaced by ``kit_text`` (the kit.json file)."""
    if _KIT_BLOCK.search(html) is None:
        raise ValueError("labeler.html has no kit block")
    return _KIT_BLOCK.sub(lambda m: m.group(1) + kit_text.strip() + m.group(3), html, count=1)


def write_json(path: Path, value: Any) -> None:
    """Write through a temporary file so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)
