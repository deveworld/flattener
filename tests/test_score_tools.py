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


def test_metric_reports_match_only_their_own_images_and_settings(tmp_path, monkeypatch):
    from flattener.dewarp.predict import rerun_record

    tool = docunet_tool
    images = _one_image(tmp_path)
    record = {"quantize": "truncate", "rerun": rerun_record(1, True)}
    report = _docunet_report(images, "uvdoc_view2_double", record)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    own = ("uvdoc_view2_double", "truncate", None, 1, True)
    assert tool.report_matches(path, images, "s", *own)
    assert not tool.report_matches(path, images, "other", *own)
    assert not tool.report_matches(path, images, "s", "once", *own[1:])
    assert not tool.report_matches(path, images, "s", "uvdoc_view2_double", "round", None, 1, True)
    assert not tool.report_matches(path, images, "s", *own[:3])
    assert not tool.report_matches(path, images, "s", *own[:3], 2, True)
    del record["quantize"], record["rerun"]  # scored before the fields: rounded, one pass
    path.write_text(json.dumps(report))
    assert tool.report_matches(path, images, "s", "uvdoc_view2_double", "round")
    assert not tool.report_matches(path, images, "s", "uvdoc_view2_double", "truncate")
    assert not tool.report_matches(path, images, "s", "uvdoc_view2_double", "round", None, 1)
    assert tool.prediction_dir(tmp_path, "demo", "round") == tmp_path / "pred" / "demo"
    assert tool.prediction_dir(tmp_path, "demo", "truncate", 0.02, 2, True) == (
        tmp_path / "pred-truncate-inset-0.02-rerun2-deskew" / "demo"
    )
    assert tool.report_name("once", "round", None, 1, True) == "docunet-once-rerun1-deskew.json"
    assert tool.record_name("truncate", None, 1) == "predictions-truncate-rerun1.json"
    assert tool.rerun_argument() == []
    assert tool.rerun_argument(2, True) == ["--rerun", "2", "--deskew"]
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    monkeypatch.setenv("FLATTENER_SIFTFLOW_SOURCE", str(tmp_path))
    with pytest.raises(SystemExit, match="--rerun"):
        tool.main([str(weights), "demo", "--rerun", "1", "--stress"])
    with pytest.raises(SystemExit, match="--deskew"):
        tool.main([str(weights), "demo", "--deskew"])

    tool = dir300_tool
    images = tmp_path / "dir300"
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


def test_ocr_reports_have_their_own_names_and_match_engine_and_images(tmp_path):
    tool = docunet_tool
    assert tool.ocr_report_name("round") == "ocr-doctr.json"
    assert tool.ocr_report_name("truncate", None, 3) == "ocr-doctr-truncate-rerun3.json"
    args = tool.build_parser().parse_args(["model.pt", "demo", "--ocr", "--tessdata-dir", "td"])
    assert args.ocr and args.tessdata_dir == Path("td")
    images = _one_image(tmp_path)
    engine = {"executable_sha256": "e", "traineddata": {"eng.traineddata": {"sha256": "t"}}}
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
    other = {**engine, "executable_sha256": "x"}
    assert not tool.ocr_report_matches(path, images, other, "truncate")
    assert not tool.ocr_report_matches(path, images, {**engine, "traineddata": {}}, "truncate")
    (images / "1_1.png").write_bytes(b"other")
    assert not tool.ocr_report_matches(path, images, engine, "truncate")
    report["inputs"]["predictions"] = {}
    path.write_text(json.dumps(report))
    assert not tool.ocr_report_matches(path, images, engine, "truncate")


def _record(weights, items, count):
    return {
        "model": {"sha256": weights},
        "complete": True,
        "quantize": "round",
        "counts": {"items": count},
        "items": items,
    }


def test_predict_refuses_a_record_of_other_weights_or_photos_left_unrectified(
    tmp_path, monkeypatch
):
    tool = docunet_tool
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"new weights")
    args = tool.build_parser().parse_args([str(weights), "demo", "--stress", "rot90"])
    images = tool.prediction_dir(tmp_path, "demo", "round")
    stress_dir = tmp_path / "stress" / "demo"
    for folder in (images, stress_dir, tmp_path / "scores" / "stress"):
        folder.mkdir(parents=True)
    record = _record("old", [], 130)
    summary = {"model": {"sha256": "old"}, "config": {"variants": ["rot90"]}, "variants": {}}
    (images / "predictions.json").write_text(json.dumps(record))
    (stress_dir / "stress.json").write_text(json.dumps(summary))
    monkeypatch.setattr(tool, "run", lambda command, env, ok=(0,): 1)  # a crash
    with pytest.raises(SystemExit, match="predictions.json holds predictions of other weights"):
        tool.predict(args, tmp_path, {})
    with pytest.raises(SystemExit, match="stress.json holds predictions of other weights"):
        tool.stress(args, tmp_path, {}, tmp_path / "scores")
    record["model"]["sha256"] = summary["model"]["sha256"] = tool.sha256_file(weights)
    (images / "predictions.json").write_text(json.dumps(record))
    (stress_dir / "stress.json").write_text(json.dumps(summary))
    assert tool.predict(args, tmp_path, {})[0] == record
    assert tool.stress(args, tmp_path, {}, tmp_path / "scores") == summary
    summary["variants"] = {"rot90": {"status": "scored"}}
    (stress_dir / "stress.json").write_text(json.dumps(summary))
    (stress_dir / "rot90").mkdir()
    (stress_dir / "rot90" / "predictions.json").write_text(json.dumps({"model": {"sha256": "o"}}))
    with pytest.raises(SystemExit, match="rot90/predictions.json holds predictions of other"):
        tool.stress(args, tmp_path, {}, tmp_path / "scores")
    args.stress = ["rot90", "small_page"]
    with pytest.raises(SystemExit, match="stress.json holds other variants"):
        tool.stress(args, tmp_path, {}, tmp_path / "scores")

    tool = dir300_tool
    args = tool.build_parser().parse_args([str(weights), "demo"])
    images = tool.prediction_dir(tmp_path, "demo", "round")
    items = [{"output": f"{k}.png", "action": "written"} for k in tool.dir300.DOCUMENTS]
    items[200] = {"output": "201.png", "action": "not_written", "reason": "size_limit_exceeded"}
    written = {"sha256": "old"}

    def fake_run(command, env, ok=(0,)):  # a crash that leaves a record
        images.mkdir(parents=True, exist_ok=True)
        record = _record(written["sha256"], items, 300)
        (images / "predictions.json").write_text(json.dumps(record))
        return 1

    monkeypatch.setattr(tool.local, "run", fake_run)
    with pytest.raises(SystemExit, match="predictions.json holds predictions of other weights"):
        tool.predict(args, tmp_path, {})
    written["sha256"] = tool.local.sha256_file(weights)
    with pytest.raises(SystemExit, match="1 photos were not rectified .size_limit_exceeded"):
        tool.predict(args, tmp_path, {})
    items[200] = {"output": "201.png", "action": "written"}
    record, _ = tool.predict(args, tmp_path, {})
    assert record["counts"]["items"] == 300


def test_summary_lines_show_an_empty_reference_cer_as_na():
    views = {
        "setting1_doctr_60": {"ed": 3.0, "cer": None},
        "setting2_dewarpnet_50": {"ed": 2.0, "cer": 0.25},
    }
    docunet = docunet_tool.ocr_text({"summary": {"doctr": views}, "comparable": False})
    assert docunet == "ocr S1 ed=3.00 cer=n/a S2 ed=2.00 cer=0.2500 (not comparable)"
    dir300 = dir300_tool.ocr_text({"summary": {"ed": 1.5, "cer": None}, "comparable": False})
    assert dir300 == "ocr 90 ed=1.50 cer=n/a (not comparable)"
