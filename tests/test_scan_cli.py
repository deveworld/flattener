"""``python -m flattener.scan``: files, stdout lines, timings and exit codes."""

import json

import pytest
from PIL import Image
from scan_fakes import FakeGate, Page, oracle_dewarper, ramp_scene, twist

from flattener.scan import __main__ as cli
from flattener.scan import locate
from flattener.scan.io import read_scan_record
from flattener.scan.pipeline import Scanner
from flattener.scan.timing import read_jsonl, summarize_by_bucket

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
        return Scanner(
            dewarper,
            gate=FakeGate([PAGE], SIZE),
            cache_size=kwargs.get("cache_size", 4),
            info={"load": {"seconds": 0.25, "device": "cpu"}},
        )

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
    assert record["plan"]["provenance"]["output_size"] == "manual"
    assert record["plan"]["provenance"]["aspect"] == "manual"
    assert [view["pass"] for view in record["plan"]["views"]] == ["A"]
    auto = [str(photo), "--out", str(tmp_path / "auto"), "--weights", "w"]
    assert cli.main(auto) == 0
    record = json.loads((tmp_path / "auto" / "photo.scan.json").read_text())
    assert record["plan"]["provenance"]["output_size"] == "auto"
    assert record["plan"]["provenance"]["aspect"] == "auto"


def test_usage_errors_exit_2(tmp_path, photo, monkeypatch, capsys):
    _install(monkeypatch)
    extra = ["--weights", "w", "--output-size", "9000,9000"]  # above the renderer limits
    assert cli.main([str(photo), "--out", str(tmp_path / "out"), *extra]) == 2
    assert not (tmp_path / "out").exists()


def test_timing_rows_cover_decoding_and_flag_cached_repeats(tmp_path, photo, capsys, monkeypatch):
    loaded = _install(monkeypatch)
    timing = tmp_path / "timing.jsonl"
    photos = [str(photo)] * 3
    args = [*photos, "--out", str(tmp_path / "out"), "--weights", "w", "--timing", str(timing)]
    assert cli.main(args) == 0
    rows = read_jsonl(timing)
    assert rows[0]["model_load"] == {"seconds": 0.25, "device": "cpu"}
    assert all("model_load" not in row for row in rows[1:])
    assert [row["cached"] for row in rows] == [False, True, True]
    assert rows[1]["cache"]["hits"]["infer"] == 2
    assert rows[0]["timings"]["read"] > 0 and "infer_pass_A" in rows[0]["timings"]
    assert rows[0]["end_to_end"] >= rows[0]["timings"]["total"] + rows[0]["timings"]["save"] - 1e-9
    assert summarize_by_bucket(rows)["<=2"]["total"]["count"] == 1  # the fresh row only
    fresh = tmp_path / "fresh.jsonl"
    args = [*photos, "--out", str(tmp_path / "fresh"), "--weights", "w", "--timing", str(fresh)]
    assert cli.main([*args, "--no-cache"]) == 0
    assert loaded[-1][1]["cache_size"] == 0
    assert [row["cached"] for row in read_jsonl(fresh)] == [False, False, False]


def test_an_unexpected_error_fails_one_photo_and_the_batch_goes_on(
    tmp_path, photo, capsys, monkeypatch
):
    _install(monkeypatch)
    calls = []
    original = Scanner.plan

    def plan(self, source, *args, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            raise RuntimeError("the model broke")
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(Scanner, "plan", plan)
    args = [str(photo), str(photo), "--out", str(tmp_path / "out"), "--weights", "w"]
    assert cli.main([*args, "--overwrite"]) == 1
    first, second = _lines(capsys)
    assert first["outcome"] == "failed" and first["status"] == "model_error"
    assert first["reason_code"] == "unexpected_error" and "RuntimeError" in first["message"]
    assert second["outcome"] == "rectified_with_warnings"
