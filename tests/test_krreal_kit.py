"""tools/krreal: the KR-Real v1 pilot kit (plan, schema, checks, drafts, manifest, labeler)."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

KIT_DIR = Path(__file__).resolve().parents[1] / "tools" / "krreal"
sys.path.insert(0, str(KIT_DIR))

import drafts  # noqa: E402
import kitlib  # noqa: E402
import krreal  # noqa: E402
import labelcheck  # noqa: E402

KIT = labelcheck.load_kit()
SCHEMA = labelcheck.load_schema()
HTML = (KIT_DIR / "labeler.html").read_text(encoding="utf-8")


def _jpeg(path: Path, *, size=(40, 30), orientation=1, make="samsung", model="SM-S911N", gps=False):
    exif = Image.Exif()
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0112] = orientation
    exif.get_ifd(0x8769)[0x9003] = "2026:09:27 10:00:00"
    if gps:
        exif.get_ifd(0x8825)[1] = "N"
    path.parent.mkdir(parents=True, exist_ok=True)
    color = tuple(int(b) for b in hashlib.sha256(path.name.encode()).digest()[:3])
    Image.new("RGB", size, color).save(path, exif=exif, quality=90)
    return path


def _label(tmp_path: Path, name="kr1-p1-curved-b03-007.jpg") -> dict:
    """A complete, valid label for a 400x300 photo."""
    info = kitlib.photo_info(_jpeg(tmp_path / "photos" / name, size=(400, 300)))
    label = drafts.skeleton_label(info, kitlib.parse_name(name))
    quad = [[50.0, 40.0], [350.0, 50.0], [340.0, 260.0], [60.0, 250.0]]
    top = [[50.0 + 50 * k, 40.0 + 10 * k / 6 - 8 * math.sin(math.pi * k / 6)] for k in range(7)]
    top[-1] = quad[1]
    page = drafts._page(
        {
            "quad": quad,
            "curved_edges": {"top": top},
            "text_polygons": [
                {"kind": "text", "points": [[80, 70], [320, 75], [310, 230], [85, 225]]}
            ],
            "reference_lines": [
                {"kind": "baseline", "points": [[90, 100 + 40 * k], [300, 102 + 40 * k]]}
                for k in range(3)
            ],
        },
        1,
    )
    label["pages"], label["page_count"] = [page], 1
    label["review"].update(status="labeled", labeler="A", labeled_at="2026-09-27T10:00:00+09:00")
    return label


def _errors(label) -> list[str]:
    return [issue.message for issue in labelcheck.errors(labelcheck.check_label(label, kit=KIT))]


# Plan and vocabulary -----------------------------------------------------------------


def test_pilot_plan_has_60_shots_on_two_phones_in_every_stratum():
    counts = kitlib.planned_counts(KIT)
    assert sum(counts.values()) == 60
    assert all(count >= 4 for count in counts.values())
    for stratum in KIT["strata"]:
        assert counts[stratum["code"]] * 5 == stratum["target_v1"]
        for shots in stratum["shots"].values():
            assert set(shots) <= set(stratum["variants"])
    rows = kitlib.shotlist_rows(KIT)
    assert [sum(row["phone"] == phone for row in rows) for phone in KIT["phones"]] == [30, 30]
    assert sum(stratum["target_v1"] for stratum in KIT["strata"]) == 300
    assert sum(stratum["transcribe"] for stratum in KIT["strata"]) >= 20


def test_schema_enums_follow_kit_json():
    defs = SCHEMA["$defs"]
    assert defs["capture"]["properties"]["stratum"]["enum"] == [s["code"] for s in KIT["strata"]]
    assert SCHEMA["properties"]["flags"]["items"]["enum"] == list(KIT["flags"])
    assert SCHEMA["properties"]["cut_off_sides"]["items"]["enum"] == KIT["sides"]
    assert defs["text_polygon"]["properties"]["kind"]["enum"] == list(KIT["text_kinds"])
    assert defs["reference_line"]["properties"]["kind"]["enum"] == list(KIT["reference_kinds"])


def test_labeler_is_one_offline_file_carrying_kit_json():
    assert kitlib.embedded_kit(HTML) == KIT
    assert re.search(r"<(script|link|img|iframe)[^>]+(src|href)=\"(https?:)?//", HTML) is None
    assert "@import" not in HTML and "fetch(" not in HTML
    assert re.findall(r"https?://", HTML) == []
    stale = kitlib.embed_kit(HTML, '{"set": "old"}')
    assert kitlib.embedded_kit(stale) == {"set": "old"}
    assert kitlib.embed_kit(stale, (KIT_DIR / "kit.json").read_text(encoding="utf-8")) == HTML


# Label checks ------------------------------------------------------------------------


def test_a_complete_label_passes_and_each_broken_rule_is_reported(tmp_path):
    label = _label(tmp_path)
    assert labelcheck.check_label(label, kit=KIT) == []

    broken = copy.deepcopy(label)
    page = broken["pages"][0]
    page["quad"] = [page["quad"][i] for i in (0, 3, 2, 1)]
    assert any("counter-clockwise" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["pages"][0]["quad"][1] = [200.0, 200.0]
    assert any(
        "not convex" in m or "first point" in m or "last point" in m for m in _errors(broken)
    )

    broken = copy.deepcopy(label)
    broken["pages"][0]["curved_edges"]["top"][3], broken["pages"][0]["curved_edges"]["top"][4] = (
        broken["pages"][0]["curved_edges"]["top"][4],
        broken["pages"][0]["curved_edges"]["top"][3],
    )
    assert any("do not advance" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["pages"][0]["curved_edges"]["top"][0] = [52.0, 40.0]
    assert any("start corner" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["pages"][0]["reference_lines"].pop()
    assert any("3-5 lines" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["pages"][0]["text_polygons"] = []
    assert any("text polygon" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["pages"][0]["page_quarter"] = 1
    assert any("quad gives 0" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["flags"] = ["two_page"]
    assert any("two_page" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["cut_off_sides"] = ["left"]
    assert any("cut_off" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["capture"]["seq"] = 8
    assert any("differ from the file name" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["photo"]["size"] = [300, 400]
    assert any("raw_size" in m for m in _errors(broken))

    broken = copy.deepcopy(label)
    broken["review"].update(status="checked", checker="A", checked_at="2026-09-27T11:00:00+09:00")
    assert any("another person" in m for m in _errors(broken))
    broken["review"]["checker"] = "B"
    assert _errors(broken) == []


def test_a_draft_needs_only_its_structure_and_the_geometry_it_has(tmp_path):
    label = _label(tmp_path)
    draft = copy.deepcopy(label)
    draft["review"].update(status="draft", labeler="", labeled_at=None)
    draft["pages"][0]["reference_lines"] = []
    draft["pages"][0]["text_polygons"] = []
    assert _errors(draft) == []
    draft["pages"], draft["page_count"] = [], 0
    assert _errors(draft) == []
    draft["page_count"] = 1
    assert _errors(draft) == ["0 pages are labelled"]


def test_page_angle_and_quarter_follow_the_reading_order_corners():
    square = [[0, 0], [100, 0], [100, 100], [0, 100]]
    for turn in range(4):
        quad = square[turn:] + square[:turn]
        angle = labelcheck.page_rotation_deg(quad)
        assert labelcheck.page_quarter(angle) == turn
        assert labelcheck.quad_problems(quad) == []
    assert labelcheck.page_rotation_deg([[0, 0], [100, 30], [90, 130], [-10, 100]]) == 16.7
    assert labelcheck.page_quarter(-45.01) == 3 and labelcheck.page_quarter(45.0) == 1


def test_the_schema_subset_validator_is_strict():
    assert labelcheck.validate_schema({"schema": "krreal-label-1"}, SCHEMA)
    with pytest.raises(labelcheck.SchemaError):
        labelcheck.validate_schema(1, {"type": "integer", "multipleOf": 2})
    assert labelcheck.validate_schema(True, {"type": "integer"}) == ["/: expected integer"]
    assert labelcheck.validate_schema([1, 2, 3], {"prefixItems": [{}, {}], "items": False})


def test_nan_is_not_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        labelcheck.load_json(path)


# Transcripts -------------------------------------------------------------------------


def test_transcript_template_round_trip_and_review_rules():
    text = drafts.render_transcript(
        photo="kr1-p1-curved-b03-007.jpg",
        sha256="a" * 64,
        page=1,
        draft="none",
        lines=["첫 줄  ", "", "", "둘째 줄\f"],
    )
    parsed = labelcheck.parse_transcript(text)
    assert parsed.header["photo"] == "kr1-p1-curved-b03-007.jpg"
    assert parsed.header["status"] == "draft"
    assert parsed.lines == ["첫 줄", "", "둘째 줄"]
    check = lambda t: labelcheck.check_transcript(  # noqa: E731
        t, photo_sha256="a" * 64, markers=KIT["transcript_markers"]
    )
    assert check(parsed) == []
    parsed.header.update(status="checked", transcribed_by="A", checked_by="A")
    assert [i.message for i in check(parsed)] == ["the checker must be another person"]
    parsed.header["checked_by"] = "B"
    parsed.lines.append("수식 [[수식]] 과 [[모름]]")
    parsed.lines.append("한")  # 한 in NFD
    messages = [i.message for i in check(parsed)]
    assert messages == ["unknown marker [[모름]]", "is not NFC"]
    with pytest.raises(ValueError):
        labelcheck.parse_transcript(text.replace("\n---\n", "\n"))


# Drafts and hooks --------------------------------------------------------------------


def test_pages_from_grid_gives_corners_and_only_curved_edges():
    gh, gw = 5, 7
    u = np.linspace(0, 1, gw)
    v = np.linspace(0, 1, gh)
    x = 100 + 400 * u[None, :] + 0 * v[:, None]
    y = 50 + 600 * v[:, None] + 30 * np.sin(np.pi * u[None, :])
    grid = np.stack([x, y], axis=-1)
    page = drafts.pages_from_grid(grid)
    assert np.allclose(page["quad"], [[100, 50], [500, 50], [500, 650], [100, 650]])
    assert sorted(page["curved_edges"]) == ["bottom", "top"]
    assert len(page["curved_edges"]["top"]) == 9
    assert page["curved_edges"]["top"][0] == page["quad"][0]
    size = (800, 700)
    normalized = np.stack(
        [2 * (grid[..., 0] + 0.5) / size[0] - 1, 2 * (grid[..., 1] + 0.5) / size[1] - 1], axis=-1
    )
    again = drafts.pages_from_grid(normalized, image_size=size)
    assert np.allclose(again["quad"], page["quad"])
    with pytest.raises(ValueError):
        drafts.pages_from_grid(grid, samples=5)


def test_a_hook_provider_prefills_a_draft_and_a_bad_proposal_is_dropped(tmp_path):
    photo = _jpeg(tmp_path / "photos" / "kr1-p2-table-d01-003.jpg", size=(400, 300))
    hook = tmp_path / "hook.py"
    hook.write_text(
        "def propose(photo, info):\n"
        "    w, h = info['size']\n"
        "    quad = [[10, 10], [w - 10, 12], [w - 12, h - 10], [12, h - 12]]\n"
        "    return {'pages': [{'quad': quad}],\n"
        "            'flags': ['glare']}\n"
        "def bad(photo, info):\n"
        "    return {'pages': [{'quad': [[10, 10], [10, 200], [300, 200], [300, 10]]}]}\n",
        encoding="utf-8",
    )
    info = kitlib.photo_info(photo)
    name = kitlib.parse_name(photo.name)
    provider = drafts.label_provider(f"{hook}:propose")
    label, dropped = drafts.build_draft(
        info, name, provider_name="hook", proposal=provider(photo, info), kit=KIT
    )
    assert dropped == []
    assert label["draft"]["fields"] == ["quad", "flags"]
    assert label["pages"][0]["page_quarter"] == 0 and label["flags"] == ["glare"]
    assert _errors(label) == []
    bad = drafts.label_provider(f"{hook}:bad")(photo, info)
    label, dropped = drafts.build_draft(info, name, provider_name="bad", proposal=bad, kit=KIT)
    assert label["pages"] == [] and "counter-clockwise" in dropped[0]
    assert drafts.label_provider("none") is drafts.no_geometry
    with pytest.raises(ValueError):
        drafts.load_hook("no_colon")


# Photos, ingest, splits and manifest -------------------------------------------------


def test_names_and_photo_info(tmp_path):
    name = kitlib.parse_name("kr1-p2-rotate-b11-030.jpg")
    assert (name.phone_id, name.stratum, name.source_id, name.seq) == ("p2", "rotate", "b11", 30)
    for bad in (
        "kr1-p1-bent-b01-001.jpg",
        "kr1-p1-hard-x03-001.jpg",
        "IMG_0001.jpg",
        "kr1-p1-flat-b01-000.jpg",
        "kr1-p1-flat-b01-001.JPG",
    ):
        with pytest.raises(ValueError):
            kitlib.parse_name(bad)
    photo = _jpeg(tmp_path / "a.jpg", size=(40, 30), orientation=6, gps=True)
    info = kitlib.photo_info(photo)
    assert info["raw_size"] == [40, 30] and info["size"] == [30, 40]
    assert info["exif_orientation"] == 6 and info["gps_present"]
    assert info["device"] == {"make": "samsung", "model": "SM-S911N"}
    assert info["sha256"] == hashlib.sha256(photo.read_bytes()).hexdigest()
    assert any("GPS" in w for w in kitlib.photo_warnings(info))


def test_ingest_copies_bytes_under_kit_names_and_never_overwrites(tmp_path):
    root = tmp_path / "v1"
    source = _jpeg(root / "incoming" / "p1" / "IMG_0001.JPG")
    (root / "incoming" / "p1" / "IMG_0002.HEIC").write_bytes(b"heic")
    rows = kitlib.shotlist_rows(KIT)
    rows[0].update(file="IMG_0001.JPG", source="b01")
    rows[1].update(file="IMG_0002.HEIC", source="b01")
    rows[2].update(file="IMG_0003.jpg", source="b1")
    done, problems = kitlib.ingest(root, rows, kit=KIT, dry_run=True)
    assert len(done) == 1 and not (root / "photos").exists()
    done, problems = kitlib.ingest(root, rows, kit=KIT)
    target = root / "photos" / "kr1-p1-curved-b01-001.jpg"
    assert target.read_bytes() == source.read_bytes()
    assert any("HEIC" in p for p in problems) and any("b03, d01 or x00" in p for p in problems)
    source.write_bytes(source.read_bytes() + b"x")
    _, problems = kitlib.ingest(root, rows[:1], kit=KIT)
    assert problems == [
        "shotlist line 2: kr1-p1-curved-b01-001.jpg exists with other bytes; not replaced"
    ]


def test_splits_are_by_source_and_stay_put():
    counts = {f"b{k:02d}": 3 + k % 4 for k in range(1, 21)}
    splits = kitlib.assign_splits(counts, {}, dev_fraction=0.4, salt="krreal-v1")
    dev = sum(counts[s] for s, split in splits.items() if split == "dev")
    assert abs(dev / sum(counts.values()) - 0.4) < 0.05
    assert kitlib.assign_splits(counts, {}, dev_fraction=0.4, salt="krreal-v1") == splits
    grown = counts | {"b99": 5}
    again = kitlib.assign_splits(grown, splits, dev_fraction=0.4, salt="krreal-v1")
    assert {s: again[s] for s in splits} == splits


def test_manifest_counts_labels_transcripts_and_m1_state(tmp_path):
    root = tmp_path / "v1"
    label = _label(root)
    (root / "labels").mkdir()
    kitlib.write_json(root / "labels" / "kr1-p1-curved-b03-007.json", label)
    _jpeg(root / "photos" / "kr1-p2-flat-d01-001.jpg", make="Apple", model="iPhone 13")
    other = root / "photos" / "kr1-p2-flat-d01-002.jpg"
    shutil.copy(root / "photos" / "kr1-p2-flat-d01-001.jpg", other)
    (root / "photos" / "notes.txt").write_text("x", encoding="utf-8")
    sha = kitlib.sha256_file(root / "photos" / "kr1-p2-flat-d01-001.jpg")
    transcript = drafts.render_transcript(
        photo="kr1-p2-flat-d01-001.jpg", sha256=sha, page=1, draft="none", lines=["글"]
    )
    (root / "transcripts").mkdir()
    (root / "transcripts" / "kr1-p2-flat-d01-001.txt").write_text(transcript, encoding="utf-8")
    manifest = kitlib.build_manifest(root)
    files = [photo["file"] for photo in manifest["photos"]]
    assert files == sorted(files) and len(files) == 3
    by_file = {photo["file"]: photo for photo in manifest["photos"]}
    assert by_file["kr1-p1-curved-b03-007.jpg"]["label"]["status"] == "labeled"
    assert by_file["kr1-p2-flat-d01-001.jpg"]["transcript"]["status"] == "draft"
    assert (
        by_file["kr1-p2-flat-d01-001.jpg"]["split"] == by_file["kr1-p2-flat-d01-002.jpg"]["split"]
    )
    assert any("same bytes" in p for p in manifest["problems"])
    assert any("notes.txt" in p for p in manifest["problems"])
    summary = manifest["summary"]
    assert summary["devices"] == {"p1": ["samsung SM-S911N"], "p2": ["Apple iPhone 13"]}
    assert summary["labels"] == {"labeled": 1, "missing": 2}
    assert not summary["m1_complete"]
    assert summary["m1_missing"][0].startswith("photos short: curved 11")
    assert all(photo["visibility"] == "private" for photo in manifest["photos"])
    moved = copy.deepcopy(label)
    moved["photo"]["sha256"] = "0" * 64
    kitlib.write_json(root / "labels" / "kr1-p1-curved-b03-007.json", moved)
    again = kitlib.build_manifest(root)
    assert any("other bytes" in p for p in again["problems"])


def test_cli_writes_the_shot_list_and_refuses_tracked_roots(tmp_path, capsys):
    assert krreal.main(["shotlist", "--root", str(tmp_path)]) == 0
    with open(tmp_path / "shotlist.csv", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60 and rows[0]["variant"] == "thick_left"
    assert krreal.main(["shotlist", "--root", str(tmp_path)]) == 1
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    kitlib.ensure_ignored(krreal.DEFAULT_ROOT)
    with pytest.raises(SystemExit, match="not ignored"):
        kitlib.ensure_ignored(KIT_DIR / "photos-here")


def test_cli_end_to_end_on_a_small_root(tmp_path, capsys):
    root = tmp_path / "v1"
    _jpeg(root / "photos" / "kr1-p1-curved-b01-001.jpg", orientation=6)
    _jpeg(root / "photos" / "kr1-p1-small-b02-007.jpg")
    assert krreal.main(["draft", "--root", str(root)]) == 0
    draft = labelcheck.load_json(root / "drafts" / "kr1-p1-curved-b01-001.json")
    assert draft["photo"]["size"] == [30, 40] and draft["review"]["status"] == "draft"
    assert krreal.main(["transcripts", "--root", str(root)]) == 0
    assert sorted(p.name for p in (root / "transcripts").iterdir()) == [
        "kr1-p1-curved-b01-001.txt",
        "kr1-p1-small-b02-007.txt",
    ]
    assert krreal.main(["check", "--root", str(root), "--drafts"]) == 0
    assert krreal.main(["manifest", "--root", str(root), "--strict"]) == 0
    manifest = labelcheck.load_json(root / "manifest.json")
    assert manifest["summary"]["labels"] == {"missing": 2}
    (root / "labels").mkdir()
    shutil.copy(root / "drafts" / "kr1-p1-curved-b01-001.json", root / "labels")
    bad = labelcheck.load_json(root / "labels" / "kr1-p1-curved-b01-001.json")
    bad["review"]["status"] = "labeled"
    kitlib.write_json(root / "labels" / "kr1-p1-curved-b01-001.json", bad)
    assert krreal.main(["check", "--root", str(root)]) == 1
    assert "labeller's name" in capsys.readouterr().err


# The labeler's JavaScript ------------------------------------------------------------

NODE = shutil.which("node")


def _run_core(tmp_path: Path, program: str, data) -> dict:
    core = re.search(r'<script id="core">(.*?)</script>', HTML, re.S).group(1)
    script = tmp_path / "core_test.js"
    script.write_text(
        core
        + "\nconst input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        + program
        + "\nprocess.stdout.write(JSON.stringify(out));\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [NODE, str(script)], input=json.dumps(data), capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_labeler_sha256_and_jpeg_reader_match_python(tmp_path):
    rng = random.Random(3)
    blobs = [bytes(rng.randrange(256) for _ in range(n)) for n in (0, 3, 55, 56, 63, 64, 65, 1000)]
    photo = _jpeg(tmp_path / "p.jpg", size=(40, 30), orientation=6, make="Apple", model="iPhone 13")
    out = _run_core(
        tmp_path,
        "const bytes = (h) => Uint8Array.from(Buffer.from(h, 'hex'));\n"
        "const out = { sha: input.blobs.map((b) => sha256Fallback(bytes(b))),"
        " jpeg: parseJpegInfo(bytes(input.jpeg).buffer) };",
        {"blobs": [b.hex() for b in blobs], "jpeg": photo.read_bytes().hex()},
    )
    assert out["sha"] == [hashlib.sha256(b).hexdigest() for b in blobs]
    assert out["jpeg"] == {
        "orientation": 6,
        "rawSize": [40, 30],
        "make": "Apple",
        "model": "iPhone 13",
        "software": None,
        "datetime": "2026:09:27 10:00:00",
    }


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_labeler_geometry_and_checks_agree_with_python(tmp_path):
    rng = random.Random(7)
    quads = []
    for _ in range(40):
        angle = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(angle), math.sin(angle)
        corners = [(-100, -140), (100, -140), (100, 140), (-100, 140)]
        quads.append(
            [[round(500 + c * x - s * y, 2), round(500 + s * x + c * y, 2)] for x, y in corners]
        )
    label = _label(tmp_path)
    cases = {"valid": label}
    mirrored = copy.deepcopy(label)
    mirrored["pages"][0]["quad"] = [mirrored["pages"][0]["quad"][i] for i in (1, 0, 3, 2)]
    mirrored["pages"][0]["curved_edges"] = {}
    cases["mirrored"] = mirrored
    few = copy.deepcopy(label)
    few["pages"][0]["reference_lines"] = few["pages"][0]["reference_lines"][:2]
    cases["two_lines"] = few
    short = copy.deepcopy(label)
    short["pages"][0]["curved_edges"]["top"] = short["pages"][0]["curved_edges"]["top"][::2]
    cases["short_curve"] = short
    same = copy.deepcopy(label)
    same["review"].update(status="checked", checker="A", checked_at="x")
    cases["same_checker"] = same
    name = copy.deepcopy(label)
    name["capture"]["source_id"] = "b04"
    cases["capture"] = name
    flags = copy.deepcopy(label)
    flags["flags"] = ["no_page"]
    cases["flags"] = flags
    out = _run_core(
        tmp_path,
        "const out = { angles: input.quads.map((q) => [rotationDeg(q), quarterOf(rotationDeg(q)),"
        " quadProblems(q).length]), checks: {} };\n"
        "for (const [k, v] of Object.entries(input.cases))\n"
        "  out.checks[k] = validateLabel(v, input.kit).errors.length;\n"
        "const raw = JSON.parse(JSON.stringify(input.cases.valid));\n"
        "raw.pages[0].page_rotation_deg = 99; raw.flags = ['no_page', 'glare'];\n"
        "out.normalized = normalizeLabel(raw, input.kit);",
        {"quads": quads, "cases": cases, "kit": KIT},
    )
    for quad, (angle, quarter, problems) in zip(quads, out["angles"], strict=True):
        assert abs(angle - labelcheck.page_rotation_deg(quad)) < 0.011
        assert quarter == labelcheck.page_quarter(labelcheck.page_rotation_deg(quad))
        assert problems == len(labelcheck.quad_problems(quad)) == 0
    for key, count in out["checks"].items():
        assert (count == 0) == (_errors(cases[key]) == []), key
    assert out["checks"]["valid"] == 0 and out["checks"]["mirrored"] > 0
    assert out["normalized"]["flags"] == ["glare"]
    assert _errors(out["normalized"]) == []


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_labeler_scripts_parse(tmp_path):
    for name in ("core", "ui"):
        source = re.search(rf'<script id="{name}">(.*?)</script>', HTML, re.S).group(1)
        path = tmp_path / f"{name}.js"
        path.write_text(source, encoding="utf-8")
        subprocess.run([NODE, "--check", str(path)], check=True, capture_output=True)
