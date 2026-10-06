"""``flattener-scan -i PHOTO [OUTPUT]``: files, messages, timings and exit codes."""

import json

import numpy as np
import pytest
from PIL import Image
from scan_fakes import FakeGate, Page, oracle_dewarper, ramp_scene, twist

from flattener import runtime
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


def test_default_output_is_written_next_to_the_working_directory(
    tmp_path, photo, capsys, monkeypatch
):
    loaded = _install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["-i", str(photo)]) == 0
    assert loaded[0][0] == (None,)
    assert loaded[0][1]["backend"] == "auto" and loaded[0][1]["device"] == "auto"
    message = capsys.readouterr()
    assert message.out == ""  # files only, no JSON
    assert "photo-scan.jpg" in message.err and "rectified" in message.err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["photo-scan.jpg", "photo.png"]
    # An existing output is kept unless -y is given.
    assert cli.main(["-i", str(photo)]) == 1
    assert "-y" in capsys.readouterr().err
    assert cli.main(["-i", str(photo), "-y", "-q"]) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "options",
    [
        ["--bundle", "b.json", "--weights", "w.pt"],
        ["--gate", "g.pt"],
        ["--orient-weights", "o.pt"],
        ["--weights", "w.pt", "--backend", "onnx"],
    ],
)
def test_invalid_weight_options_fail_before_loading(photo, tmp_path, capsys, monkeypatch, options):
    loaded = _install(monkeypatch)
    assert cli.main(["-i", str(photo), str(tmp_path / "out.jpg"), *options]) == 2
    assert loaded == []
    assert capsys.readouterr().err.startswith("flattener-scan: error:")


def test_a_scanned_photo_writes_the_named_image(tmp_path, photo, capsys, monkeypatch):
    loaded = _install(monkeypatch)
    out = tmp_path / "scans" / "page.jpg"
    timing = tmp_path / "timing.jsonl"
    args = ["-i", str(photo), str(out), "--weights", "w.pt", "--orient-weights", "o.pt"]
    code = cli.main([*args, "--orient-threshold", "0.95", "--timing", str(timing), "--debug"])
    assert code == 0
    assert loaded[0][1]["orient_threshold"] == 0.95
    with Image.open(out) as written:
        assert written.format == "JPEG" and written.size == (150, 190)
    assert sorted(path.name for path in out.parent.iterdir()) == ["page.debug.jpg", "page.jpg"]
    [row] = read_jsonl(timing)
    assert row["bucket"] == "<=2" and row["full_size"] == list(SIZE)
    assert row["backend"] == "torch"
    assert {"detect", "infer", "render", "total"} <= set(row["timings"])
    # The format follows the suffix; --record and --json add the record and a line.
    png = tmp_path / "page.png"
    assert cli.main(["-i", str(photo), str(png), "--weights", "w", "--record", "--json"]) == 0
    capsys.readouterr()
    with Image.open(png) as written:
        assert written.format == "PNG"
    record = read_scan_record(tmp_path / "page.json")
    assert record.input == str(photo) and record.plan.output.size == (190, 150)
    assert "save" in record.timings
    # A leftover debug image fails the photo before its scan is saved.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "page.debug.jpg").write_bytes(b"left over")
    args = ["-i", str(photo), str(fresh / "page.jpg"), "--weights", "w", "--debug", "--json"]
    assert cli.main(args) == 1
    (line,) = _lines(capsys)
    assert (line["outcome"], line["reason_code"]) == ("failed", "output_exists")
    assert sorted(path.name for path in fresh.iterdir()) == ["page.debug.jpg"]

    # A debug image that cannot be written leaves the scan and its outcome.
    def no_debug(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "save_image", no_debug)
    assert cli.main([*args, "-y"]) == 0
    (line,) = _lines(capsys)
    assert line["outcome"] == "rectified_with_warnings" and "disk full" in line["debug_error"]
    assert "debug" not in line["files"] and (fresh / "page.jpg").is_file()


@pytest.mark.parametrize("output", ["page", "page.tif"])
def test_a_file_output_needs_an_image_suffix(tmp_path, photo, capsys, monkeypatch, output):
    _install(monkeypatch)
    assert cli.main(["-i", str(photo), str(tmp_path / output), "--weights", "w"]) == 2
    assert ".jpg" in capsys.readouterr().err
    assert cli.main(["-i", str(photo), str(tmp_path / "a.png"), "--format", "jpg"]) == 2


def test_a_fallback_exits_1_and_keeps_the_failure(tmp_path, photo, capsys, monkeypatch):
    loaded = _install(monkeypatch, always=twist)
    out = tmp_path / "out.jpg"
    args = ["-i", str(photo), str(out), "--bundle", "b.json", "--orient-threshold", "0.9"]
    assert cli.main([*args, "--json", "--record"]) == 1
    assert loaded[0][1]["orient_threshold"] == 0.9
    [line] = _lines(capsys)
    assert line["outcome"] == "original_returned" and line["status"] == "invalid_transform"
    assert line["reason_code"] == "fold_or_degenerate"
    with Image.open(out) as written:
        assert written.size == (SIZE[1], SIZE[0])  # the original comes back
    record = json.loads((tmp_path / "out.json").read_text())
    assert record["used_fallback"] and record["guidance"]
    out2 = tmp_path / "out2.jpg"
    args = ["-i", str(photo), str(out2), "--bundle", "b", "--on-error", "return"]
    assert cli.main(args) == 1
    assert "no image  failed: " in capsys.readouterr().err
    assert not out2.exists()


def test_manual_corrections_and_a_requested_perspective(tmp_path, photo, capsys, monkeypatch):
    _install(monkeypatch, always=twist)
    quad = ",".join(f"{value:.3f}" for value in PAGE.corners().ravel())
    out = tmp_path / "quad.jpg"
    args = ["-i", str(photo), str(out), "--weights", "w", "--quad", quad, "--no-dewarp"]
    assert cli.main([*args, "--record", "--json"]) == 0  # asked for: counts as scanned
    [line] = _lines(capsys)
    assert line["outcome"] == "perspective" and line["status"] == "success"
    record = json.loads((tmp_path / "quad.json").read_text())
    assert record["plan"]["kind"] == "perspective"
    assert record["plan"]["provenance"]["dewarp"] == "manual"
    _install(monkeypatch)
    corrections = "--region 80,20,262,232 --quarter 0 --long-side 300 --aspect a4 --passes 0"
    args = ["-i", str(photo), str(tmp_path / "region.jpg"), "--weights", "w", "--no-deskew"]
    assert cli.main([*args, "--record", *corrections.split()]) == 0
    record = json.loads((tmp_path / "region.json").read_text())
    assert max(record["plan"]["output"]["height"], record["plan"]["output"]["width"]) == 300
    assert record["plan"]["provenance"]["region"] == "manual"
    assert record["plan"]["provenance"]["orientation"] == "manual"
    assert record["plan"]["provenance"]["output_size"] == "manual"
    assert record["plan"]["provenance"]["aspect"] == "manual"
    assert [view["pass"] for view in record["plan"]["views"]] == ["initial"]
    auto = ["-i", str(photo), str(tmp_path / "auto.jpg"), "--weights", "w", "--record"]
    assert cli.main(auto) == 0
    record = json.loads((tmp_path / "auto.json").read_text())
    assert record["plan"]["provenance"]["output_size"] == "auto"
    assert record["plan"]["provenance"]["aspect"] == "auto"


def test_several_photos_need_a_directory_and_failing_photos_leave_the_batch(
    tmp_path, photo, capsys, monkeypatch
):
    _install(monkeypatch)
    original = Scanner.plan

    def plan(self, source, *args, **kwargs):
        if str(source).endswith("boom.png"):
            raise RuntimeError("the model broke")
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(Scanner, "plan", plan)
    other = tmp_path / "again"
    other.mkdir()
    Image.fromarray(ramp_scene(SIZE, PAGE)).save(other / "photo.png")
    numbered = tmp_path / "photo-2.png"
    numbered.write_bytes(photo.read_bytes())
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    boom = tmp_path / "boom.png"
    boom.write_bytes(photo.read_bytes())
    inputs = [photo, other / "photo.png", boom, numbered, broken]
    flags = [part for path in inputs for part in ("-i", str(path))]
    (tmp_path / "taken.jpg").write_bytes(b"a file")
    assert cli.main([*flags, str(tmp_path / "taken.jpg"), "--weights", "w"]) == 2
    assert "directory" in capsys.readouterr().err
    out = tmp_path / "out"
    assert cli.main([*flags, f"{out}/", "--weights", "w", "--json", "--record"]) == 1
    lines = _lines(capsys)
    assert [line["outcome"] for line in lines] == [
        "rectified_with_warnings",
        "rectified_with_warnings",
        "failed",
        "rectified_with_warnings",
        "failed",
    ]
    assert lines[2]["status"] == "model_error" and lines[2]["reason_code"] == "unexpected_error"
    assert "RuntimeError" in lines[2]["message"] and lines[-1]["status"] == "input_error"
    assert (out / "photo.jpg").is_file() and (out / "photo-2.jpg").is_file()
    assert np.asarray(Image.open(out / "photo-2.jpg")).shape == (190, 150, 3)
    assert len(list(out.glob("*.jpg"))) == 3
    records = [read_scan_record(lines[i]["files"]["record"]).input for i in (0, 1, 3)]
    assert records == [str(inputs[i]) for i in (0, 1, 3)]


def test_timing_rows_cover_decoding_and_flag_cached_repeats(tmp_path, photo, capsys, monkeypatch):
    loaded = _install(monkeypatch)
    timing = tmp_path / "timing.jsonl"
    flags = ["-i", str(photo)] * 3
    args = [*flags, str(tmp_path / "out"), "--weights", "w", "--timing", str(timing)]
    assert cli.main(args) == 0
    rows = read_jsonl(timing)
    assert rows[0]["model_load"] == {"seconds": 0.25, "device": "cpu"}
    assert all("model_load" not in row for row in rows[1:])
    assert [row["cached"] for row in rows] == [False, True, True]
    assert rows[1]["cache"]["hits"]["infer"] == 2
    assert rows[0]["timings"]["read"] > 0 and "infer_pass_initial" in rows[0]["timings"]
    assert rows[0]["end_to_end"] >= rows[0]["timings"]["total"] + rows[0]["timings"]["save"] - 1e-9
    assert summarize_by_bucket(rows)["<=2"]["total"]["count"] == 1  # the fresh row only
    fresh = tmp_path / "fresh.jsonl"
    args = [*flags, str(tmp_path / "fresh"), "--weights", "w", "--timing", str(fresh)]
    assert cli.main([*args, "--no-cache"]) == 0
    assert loaded[-1][1]["cache_size"] == 0
    assert [row["cached"] for row in read_jsonl(fresh)] == [False, False, False]


def test_setup_errors_exit_2_before_anything_is_written(tmp_path, photo, capsys, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("invalid load key")

    monkeypatch.setattr(cli.Scanner, "from_files", broken)
    args = ["-i", str(photo), str(tmp_path / "out.jpg"), "--weights", "w"]
    assert cli.main(args) == 2
    assert "invalid load key" in capsys.readouterr().err
    _install(monkeypatch)
    assert cli.main([*args, "--output-size", "9000,9000"]) == 2  # above the renderer limits
    assert cli.main([*args, "--timing", str(tmp_path / "missing" / "t.jsonl")]) == 2
    capsys.readouterr()
    assert cli.main([*args, "--device", "toaster"]) == 2
    assert "device must be" in capsys.readouterr().err
    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    assert cli.main([*args, "--device", "cuda"]) == 2
    assert "CUDA" in capsys.readouterr().err
    # An output name that is an input photo would replace the photo with its scan.
    before = photo.read_bytes()
    assert cli.main(["-i", str(photo), str(photo), "--weights", "w", "-y"]) == 2
    here = ["-i", str(photo), str(tmp_path), "--weights", "w", "--format", "png", "-y"]
    assert cli.main(here) == 2
    assert photo.read_bytes() == before
    assert "is an input photo" in capsys.readouterr().err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["photo.png"]


def test_a_split_spread_saves_its_pages_as_p1_and_p2(tmp_path, photo, capsys, monkeypatch):
    _install(monkeypatch)
    out = tmp_path / "out" / "book.jpg"
    args = ["-i", str(photo), str(out), "--weights", "w", "--pages", "2", "--json", "--record"]
    assert cli.main(args) == 0
    lines = _lines(capsys)
    assert [line["page"] for line in lines] == [1, 2]
    names = sorted(path.name for path in out.parent.iterdir())
    assert names == ["book-p1.jpg", "book-p1.json", "book-p2.jpg", "book-p2.json"]
    assert read_scan_record(out.parent / "book-p2.json").metadata["page"]["page"] == 2
