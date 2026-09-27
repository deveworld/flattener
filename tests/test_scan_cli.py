"""``python -m flattener.scan``: files, stdout lines, timings and exit codes."""

import json

import numpy as np
import pytest
from PIL import Image
from scan_fakes import FakeGate, Page, oracle_dewarper, ramp_scene, twist

from flattener.scan import __main__ as cli
from flattener.scan import locate
from flattener.scan.io import read_scan_record
from flattener.scan.pipeline import Scanner
from flattener.scan.timing import read_jsonl

SIZE = (240, 320)
PAGE = Page((170, 125), (150, 190), 7.0)


@pytest.fixture(autouse=True)
def small_pages(monkeypatch):
    monkeypatch.setitem(locate.THRESHOLDS, "min_page_long_side_px", 50)


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.png"
    Image.fromarray(ramp_scene(SIZE, PAGE)).save(path)
    return path


def _install(monkeypatch, *, always=None):
    loaded = []

    def load(*args, **kwargs):
        dewarper = oracle_dewarper(PAGE, SIZE)
        dewarper.model.always = always
        loaded.append((args, kwargs))
        return Scanner(dewarper, gate=FakeGate([PAGE], SIZE))

    monkeypatch.setattr(cli.Scanner, "load", load)
    monkeypatch.setattr(cli.Scanner, "from_files", load)
    return loaded


def _lines(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_a_scanned_photo_writes_a_jpeg_and_its_record(tmp_path, photo, capsys, monkeypatch):
    loaded = _install(monkeypatch)
    out = tmp_path / "out"
    timing = tmp_path / "timing.jsonl"
    code = cli.main(
        [str(photo), "--out", str(out), "--weights", "w.pt", "--timing", str(timing), "--debug"]
    )
    assert code == 0 and loaded[0][1]["research"] is False
    [line] = _lines(capsys)
    assert line["outcome"] == "rectified_with_warnings" and line["status"] == "success"
    assert set(line["files"]) == {"image", "record", "debug"}
    with Image.open(out / "photo.jpg") as written:
        assert written.format == "JPEG" and written.size == (150, 190)
    record = read_scan_record(out / "photo.scan.json")
    assert record.input == str(photo) and record.plan.output.size == (190, 150)
    assert "save" in record.timings
    [row] = read_jsonl(timing)
    assert row["bucket"] == "<=2" and row["full_size"] == list(SIZE)
    assert {"detect", "infer", "render", "total"} <= set(row["timings"])
    assert (out / "photo.debug.jpg").is_file()
    # A second run needs --overwrite.
    assert cli.main([str(photo), "--out", str(out), "--weights", "w.pt"]) == 1
    assert _lines(capsys)[0]["reason_code"] == "output_exists"
    args = [str(photo), "--out", str(out), "--weights", "w.pt", "--overwrite", "--format", "png"]
    assert cli.main(args) == 0
    assert (out / "photo.png").is_file()


def test_a_fallback_exits_1_and_keeps_the_failure(tmp_path, photo, capsys, monkeypatch):
    _install(monkeypatch, always=twist)
    out = tmp_path / "out"
    assert cli.main([str(photo), "--out", str(out), "--bundle", "b.json"]) == 1
    [line] = _lines(capsys)
    assert line["outcome"] == "original_returned" and line["status"] == "invalid_transform"
    assert line["reason_code"] == "fold_or_degenerate"
    with Image.open(out / "photo.jpg") as written:
        assert written.size == (SIZE[1], SIZE[0])  # the original comes back
    record = json.loads((out / "photo.scan.json").read_text())
    assert record["used_fallback"] and record["guidance"]
    out2 = tmp_path / "out2"
    assert cli.main([str(photo), "--out", str(out2), "--bundle", "b", "--on-error", "return"]) == 1
    assert not (out2 / "photo.jpg").exists() and (out2 / "photo.scan.json").is_file()


def test_manual_corrections_and_a_requested_perspective(tmp_path, photo, capsys, monkeypatch):
    _install(monkeypatch, always=twist)
    quad = ",".join(f"{value:.3f}" for value in PAGE.corners().ravel())
    out = tmp_path / "out"
    args = [str(photo), "--out", str(out), "--weights", "w", "--quad", quad, "--no-dewarp"]
    assert cli.main(args) == 0  # asked for: counts as scanned
    [line] = _lines(capsys)
    assert line["outcome"] == "perspective" and line["status"] == "success"
    record = json.loads((out / "photo.scan.json").read_text())
    assert record["plan"]["kind"] == "perspective"
    assert record["plan"]["provenance"]["dewarp"] == "manual"
    _install(monkeypatch)
    args = [
        str(photo),
        "--out",
        str(tmp_path / "region"),
        "--weights",
        "w",
        "--region",
        "80,20,262,232",
        "--quarter",
        "0",
        "--long-side",
        "300",
        "--aspect",
        "a4",
        "--passes",
        "0",
        "--no-deskew",
    ]
    assert cli.main(args) == 0
    record = json.loads((tmp_path / "region" / "photo.scan.json").read_text())
    assert max(record["plan"]["output"]["height"], record["plan"]["output"]["width"]) == 300
    assert record["plan"]["provenance"]["region"] == "manual"
    assert record["plan"]["provenance"]["orientation"] == "manual"
    assert [view["pass"] for view in record["plan"]["views"]] == ["A"]


@pytest.mark.parametrize(
    "extra",
    [
        [],  # no weights
        ["--weights", "w", "--bundle", "b"],
        ["--weights", "w", "--region", "1,2,3"],
        ["--weights", "w", "--quad", "1,2,3,4"],
        ["--weights", "w", "--passes", "5"],
        ["--weights", "w", "--quarter", "4"],
        ["--weights", "w", "--aspect", "square"],
        ["--weights", "w", "--region", "0,0,1,1", "--quad", "0,0,1,0,1,1,0,1"],
        ["--bundle", "b", "--gate", "g"],
        ["--weights", "w", "--threads", "0"],
        ["--weights", "w", "--border", "1.5"],
    ],
)
def test_usage_errors_exit_2(tmp_path, photo, monkeypatch, extra, capsys):
    _install(monkeypatch)
    assert cli.main([str(photo), "--out", str(tmp_path / "out"), *extra]) == 2
    assert not (tmp_path / "out").exists()


def test_same_stems_get_numbered_and_unreadable_photos_fail(tmp_path, photo, capsys, monkeypatch):
    _install(monkeypatch)
    other = tmp_path / "again"
    other.mkdir()
    Image.fromarray(ramp_scene(SIZE, PAGE)).save(other / "photo.png")
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    out = tmp_path / "out"
    args = [str(photo), str(other / "photo.png"), str(broken), "--out", str(out), "--weights", "w"]
    assert cli.main(args) == 1
    lines = _lines(capsys)
    assert [line["outcome"] for line in lines] == [
        "rectified_with_warnings",
        "rectified_with_warnings",
        "failed",
    ]
    assert lines[2]["status"] == "input_error"
    assert (out / "photo.jpg").is_file() and (out / "photo-2.jpg").is_file()
    assert np.asarray(Image.open(out / "photo-2.jpg")).shape == (190, 150, 3)
