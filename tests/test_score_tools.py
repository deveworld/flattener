"""tools/score_docunet_local.py and tools/score_dir300_local.py: when a saved report is
reused, and when a run stops before scoring."""

import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


docunet_tool = _load("score_docunet_local")
dir300_tool = _load("score_dir300_local")


def _docunet_report(images, ref_resize, record):
    return {
        "config": {"ref_resize": ref_resize, "conventions": list(docunet_tool.CONVENTIONS)},
        "comparable": True,
        "siftflow": {"source_sha256": "s"},
        "inputs": {"predictions": {"1_1": docunet_tool.sha256_file(images / "1_1.png")}},
        "predictions_record": record,
    }


def _one_image(tmp_path):
    images = tmp_path / "pred"
    images.mkdir()
    (images / "1_1.png").write_bytes(b"image")
    return images


# DocUNet


def test_a_report_matches_only_its_own_settings(tmp_path):
    tool = docunet_tool
    images = _one_image(tmp_path)
    report = _docunet_report(images, "uvdoc_view2_double", {"quantize": "truncate"})
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert tool.report_matches(path, images, "s", "uvdoc_view2_double", "truncate")
    assert not tool.report_matches(path, images, "s", "once", "truncate")
    assert not tool.report_matches(path, images, "s", "uvdoc_view2_double", "round")
    del report["predictions_record"]["quantize"]  # scored before the field: rounded images
    path.write_text(json.dumps(report))
    assert tool.report_matches(path, images, "s", "uvdoc_view2_double", "round")
    assert not tool.report_matches(path, images, "s", "uvdoc_view2_double", "truncate")


def test_rerun_settings_get_their_own_files_and_reports(tmp_path, monkeypatch):
    from flattener.dewarp.predict import rerun_record

    tool = docunet_tool
    args = tool.build_parser().parse_args(["model.pt", "demo", "--rerun", "1", "--deskew"])
    assert (args.rerun, args.deskew) == (1, True)
    assert tool.build_parser().parse_args(["model.pt", "demo"]).rerun == 0
    assert tool.prediction_dir(tmp_path, "demo", "round", None, 0) == tmp_path / "pred" / "demo"
    assert tool.prediction_dir(tmp_path, "demo", "round", None, 1) == (
        tmp_path / "pred-rerun1" / "demo"
    )
    assert tool.prediction_dir(tmp_path, "demo", "truncate", 0.02, 2, True) == (
        tmp_path / "pred-truncate-inset-0.02-rerun2-deskew" / "demo"
    )
    assert tool.report_name("once", "round", None, 1, True) == "docunet-once-rerun1-deskew.json"
    assert tool.record_name("truncate", None, 1) == "predictions-truncate-rerun1.json"
    assert tool.ocr_report_name("round", None, 3) == "ocr-doctr-rerun3.json"
    assert tool.rerun_argument() == [] and tool.rerun_argument(0) == []
    assert tool.rerun_argument(2, True) == ["--rerun", "2", "--deskew"]
    images = _one_image(tmp_path)
    record = {"quantize": "round", "rerun": rerun_record(1, True)}
    report = _docunet_report(images, "once", record)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    settings = (images, "s", "once", "round", None)
    assert tool.report_matches(path, *settings, 1, True)
    assert not tool.report_matches(path, *settings)
    assert not tool.report_matches(path, *settings, 1)
    assert not tool.report_matches(path, *settings, 2, True)
    del record["rerun"]  # scored before the field: whole-image predictions
    path.write_text(json.dumps(report))
    assert tool.report_matches(path, *settings) and not tool.report_matches(path, *settings, 1)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    monkeypatch.setenv("FLATTENER_SIFTFLOW_SOURCE", str(tmp_path))
    with pytest.raises(SystemExit, match="--rerun"):
        tool.main([str(weights), "demo", "--rerun", "1", "--stress"])
    with pytest.raises(SystemExit, match="--deskew"):
        tool.main([str(weights), "demo", "--deskew"])


def test_ocr_reports_have_their_own_names_and_match_engine_and_images(tmp_path):
    tool = docunet_tool
    assert tool.ocr_report_name("round") == "ocr-doctr.json"
    assert tool.ocr_report_name("truncate") == "ocr-doctr-truncate.json"
    args = tool.build_parser().parse_args(["model.pt", "demo", "--ocr", "--tessdata-dir", "td"])
    assert args.ocr and args.tessdata_dir == Path("td")
    images = _one_image(tmp_path)
    engine = {
        "executable_sha256": "e",
        "traineddata": {"eng.traineddata": {"sha256": "t"}},
    }
    report = {
        "config": {"protocols": ["doctr"]},
        "inputs": {"predictions": {"1_1": tool.sha256_file(images / "1_1.png")}},
        "predictions_record": {"quantize": "truncate"},
        "tesseract": engine,
    }
    path = tmp_path / "ocr.json"
    path.write_text(json.dumps(report))
    assert tool.ocr_report_matches(path, images, engine, "truncate")
    assert not tool.ocr_report_matches(path, images, engine, "round")
    assert not tool.ocr_report_matches(
        path, images, {**engine, "executable_sha256": "x"}, "truncate"
    )
    assert not tool.ocr_report_matches(path, images, {**engine, "traineddata": {}}, "truncate")
    (images / "1_1.png").write_bytes(b"other")
    assert not tool.ocr_report_matches(path, images, engine, "truncate")
    report["inputs"]["predictions"] = {}
    path.write_text(json.dumps(report))
    assert not tool.ocr_report_matches(path, images, engine, "truncate")


# DIR300


def test_a_metric_report_matches_only_its_own_images_and_settings(tmp_path):
    tool = dir300_tool
    images = tmp_path / "pred"
    images.mkdir()
    for k in tool.dir300.DOCUMENTS:
        (images / f"{k}.png").write_bytes(str(k).encode())
    hashes = {str(k): tool.local.sha256_file(images / f"{k}.png") for k in tool.dir300.DOCUMENTS}
    report = {
        "benchmark": "dir300",
        "comparable": True,
        "config": {"documents": "all"},
        "siftflow": {"source_sha256": "s"},
        "inputs": {"predictions": hashes},
        "predictions_record": {"quantize": "truncate"},
    }
    path = tmp_path / "dir300.json"
    path.write_text(json.dumps(report))
    assert tool.report_matches(path, images, "s", "truncate")
    assert not tool.report_matches(path, images, "s", "round")
    assert not tool.report_matches(path, images, "other", "truncate")
    (images / "7.png").write_bytes(b"changed")
    assert not tool.report_matches(path, images, "s", "truncate")
    report["inputs"]["predictions"]["7"] = tool.local.sha256_file(images / "7.png")
    report["config"]["documents"] = [7]
    path.write_text(json.dumps(report))
    assert not tool.report_matches(path, images, "s", "truncate")


def test_photos_left_unrectified_stop_the_run_before_scoring(tmp_path, monkeypatch):
    tool = dir300_tool
    args = tool.build_parser().parse_args([str(tmp_path / "model.pt"), "demo"])
    images = tool.prediction_dir(tmp_path, "demo", "round")
    items = [{"output": f"{k}.png", "action": "written"} for k in tool.dir300.DOCUMENTS]
    items[200] = {"output": "201.png", "action": "not_written", "reason": "size_limit_exceeded"}

    def fake_run(command, env, ok=(0,)):
        images.mkdir(parents=True, exist_ok=True)
        record = {"complete": True, "quantize": "round", "counts": {"items": 300}, "items": items}
        (images / "predictions.json").write_text(json.dumps(record))
        return 1

    monkeypatch.setattr(tool.local, "run", fake_run)
    with pytest.raises(SystemExit, match="1 photos were not rectified .size_limit_exceeded"):
        tool.predict(args, tmp_path, {})
    items[200] = {"output": "201.png", "action": "written"}
    record, _ = tool.predict(args, tmp_path, {})
    assert record["counts"]["items"] == 300
