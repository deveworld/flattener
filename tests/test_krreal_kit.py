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


def test_page_angle_and_quarter_follow_the_reading_order_corners():
    square = [[0, 0], [100, 0], [100, 100], [0, 100]]
    for turn in range(4):
        quad = square[turn:] + square[:turn]
        angle = labelcheck.page_rotation_deg(quad)
        assert labelcheck.page_quarter(angle) == turn
        assert labelcheck.quad_problems(quad) == []
    assert labelcheck.page_rotation_deg([[0, 0], [100, 30], [90, 130], [-10, 100]]) == 16.7
    assert labelcheck.page_quarter(-45.01) == 3 and labelcheck.page_quarter(45.0) == 1


def test_scan_provider_turns_a_saved_grid_into_a_draft_quad(tmp_path):
    import scan_provider

    # 40x30 on file, EXIF 6: the labeled photo is 30 wide and 40 tall.
    photo = _jpeg(tmp_path / "photos" / "kr1-p1-curved-b03-007.jpg", orientation=6)
    info = kitlib.photo_info(photo) | {"name": kitlib.parse_name(photo.name)}
    grid = [[[x, y] for x in (-0.5, 0.0, 0.5)] for y in (-0.5, 0.0, 0.5)]
    record = {
        "outcome": "rectified",
        "source": {"file_sha256": info["sha256"]},
        "plan": {"kind": "dewarp", "grid": grid},
    }
    saved = tmp_path / "scan-quarter" / f"{photo.stem}.scan.json"
    kitlib.write_json(saved, record)
    proposal = scan_provider.propose(photo, info)
    assert proposal["pages"][0]["quad"] == [[7.0, 9.5], [22.0, 9.5], [22.0, 29.5], [7.0, 29.5]]
    label, dropped = drafts.build_draft(
        info, info["name"], provider_name="scan", proposal=proposal, kit=KIT
    )
    assert dropped == [] and label["draft"]["fields"] == ["quad"]
    kitlib.write_json(saved, record | {"outcome": "original_returned"})
    assert scan_provider.propose(photo, info) == {}


# Splits and the shot list ------------------------------------------------------------


def test_splits_are_by_source_and_stay_put():
    counts = {f"b{k:02d}": 3 + k % 4 for k in range(1, 21)}
    splits = kitlib.assign_splits(counts, {}, dev_fraction=0.4, salt="krreal-v1")
    dev = sum(counts[s] for s, split in splits.items() if split == "dev")
    assert abs(dev / sum(counts.values()) - 0.4) < 0.05
    assert kitlib.assign_splits(counts, {}, dev_fraction=0.4, salt="krreal-v1") == splits
    grown = counts | {"b99": 5}
    again = kitlib.assign_splits(grown, splits, dev_fraction=0.4, salt="krreal-v1")
    assert {s: again[s] for s in splits} == splits


def test_a_photo_with_gps_data_is_never_a_public_candidate(tmp_path):
    _jpeg(tmp_path / "photos" / "kr1-p1-flat-d01-001.jpg")
    _jpeg(tmp_path / "photos" / "kr1-p1-flat-d01-002.jpg", gps=True)
    (tmp_path / "sources.csv").write_text("source_id,kind,public,note\nd01,document,yes,\n")
    manifest = kitlib.build_manifest(tmp_path)
    visibility = {photo["file"]: photo["visibility"] for photo in manifest["photos"]}
    assert visibility == {
        "kr1-p1-flat-d01-001.jpg": "public_candidate",
        "kr1-p1-flat-d01-002.jpg": "private",
    }
    assert [w for w in manifest["warnings"] if "GPS" in w] == [
        "kr1-p1-flat-d01-002.jpg: carries GPS data; turn off location tags on the phone"
    ]


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


def test_the_manifest_goes_only_to_a_folder_git_ignores(tmp_path, capsys):
    _jpeg(tmp_path / "photos" / "kr1-p1-flat-d01-001.jpg")
    out = tmp_path / "elsewhere" / "manifest.json"
    assert krreal.main(["manifest", "--root", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["photos"][0]["device"]["model"]
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    tracked = KIT_DIR / "manifest-here" / "manifest.json"
    with pytest.raises(SystemExit, match="not ignored"):
        krreal.main(["manifest", "--root", str(tmp_path), "--out", str(tracked)])
    assert not tracked.parent.exists()


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
