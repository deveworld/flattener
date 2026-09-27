"""tools/score_docunet_local.py: every quantization and ref-resize gets its own files."""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "score_docunet_local.py"
_spec = importlib.util.spec_from_file_location("score_docunet_local", TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_defaults_keep_the_existing_names(tmp_path):
    args = tool.build_parser().parse_args(["model.pt", "r1"])
    assert (args.quantize, args.ref_resize) == ("round", "once")
    assert tool.report_name(args.ref_resize, args.quantize) == "docunet-once.json"
    assert tool.record_name(args.quantize) == "predictions.json"
    assert tool.prediction_dir(tmp_path, "r1", args.quantize) == tmp_path / "pred" / "r1"


def test_every_setting_has_its_own_files(tmp_path):
    settings = [(q, r) for q in tool.QUANTIZE_MODES for r in tool.REF_RESIZE_MODES]
    assert len({tool.report_name(r, q) for q, r in settings}) == len(settings) == 4
    assert tool.report_name("uvdoc_view2_double", "truncate") == (
        "docunet-uvdoc_view2_double-truncate.json"
    )
    assert tool.record_name("truncate") == "predictions-truncate.json"
    assert tool.prediction_dir(tmp_path, "r1", "truncate") == tmp_path / "pred-truncate" / "r1"


def test_a_report_matches_only_its_own_settings(tmp_path):
    images = tmp_path / "pred"
    images.mkdir()
    (images / "1_1.png").write_bytes(b"image")
    report = {
        "config": {"ref_resize": "uvdoc_view2_double", "conventions": list(tool.CONVENTIONS)},
        "comparable": True,
        "siftflow": {"source_sha256": "s"},
        "inputs": {"predictions": {"1_1": tool.sha256_file(images / "1_1.png")}},
        "predictions_record": {"quantize": "truncate"},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert tool.report_matches(path, images, "s", "uvdoc_view2_double", "truncate")
    assert not tool.report_matches(path, images, "s", "once", "truncate")
    assert not tool.report_matches(path, images, "s", "uvdoc_view2_double", "round")
    del report["predictions_record"]["quantize"]  # scored before the field: rounded images
    path.write_text(json.dumps(report))
    assert tool.report_matches(path, images, "s", "uvdoc_view2_double", "round")
    assert not tool.report_matches(path, images, "s", "uvdoc_view2_double", "truncate")


def test_stress_runs_only_with_the_default_protocol(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    monkeypatch.setenv("FLATTENER_SIFTFLOW_SOURCE", str(tmp_path))
    with pytest.raises(SystemExit, match="--stress"):
        tool.main([str(weights), "r1", "--quantize", "truncate", "--stress"])
    with pytest.raises(SystemExit, match="--stress"):
        tool.main([str(weights), "r1", "--ref-resize", "uvdoc_view2_double", "--stress"])


def test_ocr_reports_have_their_own_names_and_match_engine_and_images(tmp_path):
    assert tool.ocr_report_name("round") == "ocr-doctr.json"
    assert tool.ocr_report_name("truncate") == "ocr-doctr-truncate.json"
    args = tool.build_parser().parse_args(["model.pt", "r1", "--ocr", "--tessdata-dir", "td"])
    assert args.ocr and args.tessdata_dir == Path("td")
    images = tmp_path / "pred"
    images.mkdir()
    (images / "1_1.png").write_bytes(b"image")
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


def test_ocr_text_marks_reports_that_are_not_comparable():
    setting = {"ed": 375.6, "cer": 0.1541}
    report = {
        "summary": {"doctr": {"setting1_doctr_60": setting, "setting2_dewarpnet_50": setting}},
        "comparable": True,
        "comparability": {"tesseract_build": "build-a"},
    }
    assert tool.ocr_text(report) == (
        "ocr S1 ed=375.60 cer=0.1541 S2 ed=375.60 cer=0.1541 (build-a)"
    )
    report["comparable"] = False
    assert tool.ocr_text(report).endswith("(not comparable)")
