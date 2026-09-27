"""tools/score_dir300_local.py: file names per quantization and when a report is kept."""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "score_dir300_local.py"
_spec = importlib.util.spec_from_file_location("score_dir300_local", TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_each_quantization_has_its_own_files(tmp_path):
    assert tool.build_parser().parse_args(["model.pt", "r1"]).quantize == "round"
    assert tool.report_name("round") == "dir300.json"
    assert tool.report_name("truncate") == "dir300-truncate.json"
    assert tool.ocr_report_name("truncate") == "dir300-ocr-truncate.json"
    assert tool.record_name("round") == "dir300-predictions.json"
    assert tool.prediction_dir(tmp_path, "r1", "round") == tmp_path / "dir300" / "pred" / "r1"
    assert tool.prediction_dir(tmp_path, "r1", "truncate") == (
        tmp_path / "dir300" / "pred-truncate" / "r1"
    )
    names = {tool.report_name(q) for q in tool.local.QUANTIZE_MODES}
    names |= {tool.ocr_report_name(q) for q in tool.local.QUANTIZE_MODES}
    names |= {tool.record_name(q) for q in tool.local.QUANTIZE_MODES}
    assert len(names) == 6


def _images(tmp_path, documents):
    images = tmp_path / "pred"
    images.mkdir()
    for k in documents:
        (images / f"{k}.png").write_bytes(str(k).encode())
    return images, {str(k): tool.local.sha256_file(images / f"{k}.png") for k in documents}


def test_a_metric_report_matches_only_its_own_images_and_settings(tmp_path):
    images, hashes = _images(tmp_path, tool.dir300.DOCUMENTS)
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


def test_an_ocr_report_matches_only_its_engine(tmp_path):
    images, hashes = _images(tmp_path, tool.dir300.DOCUMENTS)
    engine = {"executable_sha256": "e", "traineddata": {"eng.traineddata": {"sha256": "t"}}}
    report = {
        "benchmark": "dir300-ocr",
        "config": {"documents": "all", "references_verified": True},
        "tesseract": engine,
        "inputs": {"predictions": {str(k): hashes[str(k)] for k in tool.dir300.OCR_DOCUMENTS}},
        "predictions_record": {"quantize": "round"},
    }
    path = tmp_path / "dir300-ocr.json"
    path.write_text(json.dumps(report))
    assert tool.ocr_report_matches(path, images, engine, "round")
    assert not tool.ocr_report_matches(path, images, {**engine, "executable_sha256": "x"}, "round")
    assert not tool.ocr_report_matches(path, images, engine, "truncate")


def test_a_missing_checkpoint_stops_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setenv("FLATTENER_SIFTFLOW_SOURCE", str(tmp_path))
    with pytest.raises(SystemExit, match="no checkpoint"):
        tool.main([str(tmp_path / "missing.pt"), "r1"])
    with pytest.raises(SystemExit, match="RUN may use"):
        tool.main([str(tmp_path / "missing.pt"), "../r1"])


def test_photos_left_unrectified_stop_the_run_before_scoring(tmp_path, monkeypatch):
    args = tool.build_parser().parse_args([str(tmp_path / "model.pt"), "r1"])
    images = tool.prediction_dir(tmp_path, "r1", "round")
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
