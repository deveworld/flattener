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
