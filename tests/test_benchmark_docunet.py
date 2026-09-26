"""DocUNet conventions, report schema and the command line, without SIFT flow."""

import json

import numpy as np
import pytest
from PIL import Image

from flattener.benchmark import __main__ as cli
from flattener.benchmark import docunet, metrics
from flattener.benchmark.reporting import write_json_atomic


def test_conventions_select_the_published_image_sets():
    raw = docunet.convention_views("130_raw")
    rotated = docunet.convention_views("130_rot64")
    excluded = docunet.convention_views("128_excl64")
    assert len(raw) == len(rotated) == 130 and len(excluded) == 128
    assert not any(flag for _, _, flag in raw)
    assert [(k, m) for k, m, flag in rotated if flag] == [(64, 1), (64, 2)]
    assert all(k != 64 for k, _, _ in excluded)
    assert {(k, m) for k, m, _ in raw} == {(k, m) for k in range(1, 66) for m in (1, 2)}
    union = docunet.required_views(["130_rot64", "128_excl64"])
    assert len(union) == 130
    assert len(docunet.required_views(["130_raw", "130_rot64"])) == 132
    with pytest.raises(ValueError):
        docunet.convention_views("129")
    with pytest.raises(ValueError):
        docunet.validate_conventions([])


def test_reference_resize_modes():
    assert [docunet.reference_resizes("once", m) for m in (1, 2)] == [1, 1]
    assert [docunet.reference_resizes("uvdoc_view2_double", m) for m in (1, 2)] == [1, 2]
    with pytest.raises(ValueError):
        docunet.reference_resizes("twice", 1)


def test_format_name_requires_exact_fields_and_relative_paths():
    assert docunet.format_name("{k}_{m} copy.png", fields=("k", "m"), k=3, m=2) == "3_2 copy.png"
    for pattern in (
        "{k}.png",
        "{k}_{m:02d}.png",
        "../{k}_{m}.png",
        "/abs/{k}_{m}.png",
        "C:/x/{k}_{m}.png",
        "\\\\server\\{k}_{m}.png",
        "a\\..\\..\\{k}_{m}.png",
    ):
        with pytest.raises(ValueError):
            docunet.format_name(pattern, fields=("k", "m"), k=1, m=1)


def _row(k, m, rotated, value):
    return {
        "k": k, "m": m, "rotated_180": rotated, "ms_ssim": value, "ld": 10 * value,
        "ad": value / 2, "ad_exact_lstsq": value / 2, "ad_lsqr_4iter": value / 3,
    }  # fmt: skip


def test_summarize_averages_each_convention_over_its_own_rows():
    rows = [_row(k, m, False, float(k)) for k in range(1, 66) for m in (1, 2)]
    rows += [_row(64, m, True, 100.0) for m in (1, 2)]
    summary = docunet.summarize(rows, ["130_raw", "130_rot64", "128_excl64"])
    raw_mean = np.mean([k for k in range(1, 66) for _ in (1, 2)])
    assert summary["130_raw"]["ms_ssim"] == pytest.approx(raw_mean)
    rotated = [k for k in range(1, 66) if k != 64 for _ in (1, 2)] + [100.0, 100.0]
    assert summary["130_rot64"]["ms_ssim"] == pytest.approx(np.mean(rotated))
    excluded = [k for k in range(1, 66) if k != 64 for _ in (1, 2)]
    assert summary["128_excl64"]["ld"] == pytest.approx(10 * np.mean(excluded))
    assert summary["128_excl64"]["n"] == 128
    assert summary["128_excl64"]["views"]["2"]["n"] == 64
    assert summary["128_excl64"]["ad_lsqr_4iter"] == pytest.approx(np.mean(excluded) / 3)


@pytest.fixture
def tiny_docunet(tmp_path, monkeypatch):
    """65 tiny scans and 130 predictions; the metric area is shrunk to keep it fast."""
    monkeypatch.setattr(metrics, "METRIC_AREA", 300)
    rng = np.random.default_rng(7)
    scans = tmp_path / "scans"
    predictions = tmp_path / "pred"
    scans.mkdir()
    predictions.mkdir()
    for k in range(1, 66):
        scan = rng.integers(0, 256, size=(24, 18, 4), dtype=np.uint8)
        Image.fromarray(scan, "RGBA").save(scans / f"docunet-reference-{k}.png")
        for m in (1, 2):
            prediction = rng.integers(0, 256, size=(20, 16, 3), dtype=np.uint8)
            Image.fromarray(prediction).save(predictions / f"{k}_{m}.png")
    return scans, predictions


def test_evaluate_docunet_report_schema_without_siftflow(tiny_docunet, tmp_path):
    scans, predictions = tiny_docunet
    report = docunet.evaluate_docunet(
        predictions,
        pattern="{k}_{m}.png",
        conventions=["130_rot64", "128_excl64"],
        build=None,
        references_dir=scans,
        verify_references=False,
        workers=1,
    )
    assert report["schema"] == docunet.REPORT_SCHEMA
    assert report["comparable"] is False and report["siftflow"] is None
    assert report["config"]["ad_solver"] == "lsqr_4iter"
    assert report["config"]["conventions"] == ["130_rot64", "128_excl64"]
    assert len(report["rows"]) == 130
    assert [row["id"] for row in report["rows"] if row["rotated_180"]] == ["64_1", "64_2"]
    assert set(report["inputs"]["references"]) == {str(k) for k in range(1, 66)}
    assert report["summary"]["130_rot64"]["n"] == 130
    assert report["summary"]["128_excl64"]["n"] == 128
    assert report["summary"]["128_excl64"]["ld"] is None
    assert {"environment", "protocol", "timing", "created_utc"} <= set(report)
    row = report["rows"][0]
    assert {
        "ms_ssim",
        "ld",
        "ad",
        "prediction_sha256",
        "reference_sha256",
        "metric_shape_hw",
    } <= set(row)
    digest = write_json_atomic(tmp_path / "report.json", report)
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        write_json_atomic(tmp_path / "report.json", report)
    assert json.loads((tmp_path / "report.json").read_text())["schema"] == docunet.REPORT_SCHEMA


def test_evaluate_docunet_checks_inputs(tiny_docunet):
    scans, predictions = tiny_docunet
    (predictions / "5_2.png").unlink()
    with pytest.raises(FileNotFoundError, match="predictions missing"):
        docunet.evaluate_docunet(
            predictions, pattern="{k}_{m}.png", conventions=["130_raw"], build=None,
            references_dir=scans, verify_references=False, workers=1,
        )  # fmt: skip
    with pytest.raises(ValueError, match="published DocUNet scans"):
        docunet.evaluate_docunet(
            predictions, pattern="{k}_{m}.png", conventions=["128_excl64"], build=None,
            references_dir=scans, workers=1,
        )  # fmt: skip
    with pytest.raises(ValueError):
        docunet.evaluate_docunet(
            predictions, pattern="{k}_{m}.png", conventions=["128_excl64"], build=None,
            references_dir=scans, verify_references=False, workers=0,
        )  # fmt: skip


def test_worker_initializer_limits_torch_to_one_thread():
    import torch

    before = torch.get_num_threads()
    try:
        docunet._initialize_worker(None)
        assert torch.get_num_threads() == 1
        assert docunet._WORKER_SIFTFLOW is None
    finally:
        torch.set_num_threads(before)


def test_cli_defaults_to_two_workers_and_documents_every_subcommand(capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["docunet", "pred", "--output", "r.json"])
    assert args.workers == 2 == docunet.DEFAULT_WORKERS
    assert args.ref_resize == "once" and args.ad_solver == "lsqr_4iter"
    args = parser.parse_args(["ocr", "pred", "--output", "r.json"])
    assert args.workers == 2 and args.tesseract == "tesseract"
    args = parser.parse_args(["calibrate", "--reports-dir", "reports"])
    assert args.workers == 2
    for command in ("docunet", "ocr", "calibrate", "build-siftflow"):
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--help"])
        assert "--workers" in capsys.readouterr().out or command == "build-siftflow"
    with pytest.raises(SystemExit):
        parser.parse_args(["docunet", "pred", "--output", "r.json", "--workers", "0"])


def test_cli_docunet_ms_ssim_only_writes_report(tiny_docunet, tmp_path, capsys):
    scans, predictions = tiny_docunet
    output = tmp_path / "out.json"
    code = cli.main(
        [
            "docunet", str(predictions), "--output", str(output), "--ms-ssim-only",
            "--references-dir", str(scans), "--no-verify-references", "--workers", "1",
            "--convention", "128_excl64", "--quiet",
        ]
    )  # fmt: skip
    assert code == 0
    report = json.loads(output.read_text())
    assert report["config"]["workers"] == 1 and report["summary"]["128_excl64"]["n"] == 128
    assert "128_excl64" in json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit, match="exists"):
        cli.main(["docunet", str(predictions), "--output", str(output), "--ms-ssim-only"])


def _asymmetric_document(tmp_path, scan_shape=(25, 19)):
    """A scan and a prediction whose 180-degree rotation changes the MS-SSIM score."""
    rng = np.random.default_rng(12)
    height, width = scan_shape
    ramp = np.linspace(0, 255, height * width).reshape(height, width)
    scan = np.clip(ramp + rng.normal(scale=20, size=ramp.shape), 0, 255).astype(np.uint8)
    prediction = np.clip(ramp[:22, :17] + rng.normal(scale=20, size=(22, 17)), 0, 255)
    prediction = prediction.astype(np.uint8)
    Image.fromarray(scan).save(tmp_path / "scan.png")
    Image.fromarray(prediction).save(tmp_path / "pred.png")
    return scan, prediction, str(tmp_path / "scan.png"), str(tmp_path / "pred.png")


def test_rot64_scores_the_prediction_rotated_by_180_degrees(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRIC_AREA", 300)
    scan, prediction, scan_path, prediction_path = _asymmetric_document(tmp_path)
    selected = [view for view in docunet.required_views(["130_raw", "130_rot64"]) if view[0] == 64]
    assert selected == [(64, 1, False), (64, 1, True), (64, 2, False), (64, 2, True)]
    views = tuple(docunet.ViewTask(m, rotated, prediction_path) for _, m, rotated in selected)
    task = docunet.DocumentTask(64, scan_path, views, "once", metrics.DEFAULT_AD_SOLVER)
    raw, rotated, *_ = docunet.score_document(task, None)
    assert (raw["rotated_180"], rotated["rotated_180"]) == (False, True)
    assert raw["prediction_sha256"] == rotated["prediction_sha256"]
    assert raw["ms_ssim"] == metrics.score_pair(prediction, scan).ms_ssim
    expected = metrics.score_pair(np.ascontiguousarray(np.rot90(prediction, 2)), scan)
    assert rotated["ms_ssim"] == expected.ms_ssim
    assert abs(rotated["ms_ssim"] - raw["ms_ssim"]) > 1e-3


def test_ref_resize_modes_prepare_the_scan_once_or_per_view(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRIC_AREA", 300)
    scan, prediction, scan_path, prediction_path = _asymmetric_document(tmp_path)
    calls = []
    original = metrics.prepare_reference

    def counting(reference, *, resizes=1):
        calls.append(resizes)
        return original(reference, resizes=resizes)

    monkeypatch.setattr(metrics, "prepare_reference", counting)
    views = (
        docunet.ViewTask(1, False, prediction_path),
        docunet.ViewTask(2, False, prediction_path),
    )
    once = docunet.score_document(
        docunet.DocumentTask(3, scan_path, views, "once", "lsqr_4iter"), None
    )
    assert calls == [1]
    assert [row["reference_resizes"] for row in once] == [1, 1]
    assert once[0]["ms_ssim"] == once[1]["ms_ssim"]
    calls.clear()
    task = docunet.DocumentTask(3, scan_path, views, "uvdoc_view2_double", "lsqr_4iter")
    uvdoc = docunet.score_document(task, None)
    assert calls == [1, 2]
    assert [row["reference_resizes"] for row in uvdoc] == [1, 2]
    assert uvdoc[0]["ms_ssim"] == once[0]["ms_ssim"]
    assert uvdoc[1]["ms_ssim"] != uvdoc[0]["ms_ssim"]
    twice = original(scan, resizes=2)
    assert uvdoc[1]["ms_ssim"] == metrics.score_prepared(prediction, twice).ms_ssim


def test_cli_docunet_reports_every_convention_by_default(monkeypatch, tmp_path):
    seen = {}

    def fake_evaluate(predictions, **kwargs):
        seen.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(cli.docunet, "evaluate_docunet", fake_evaluate)
    monkeypatch.setattr(cli, "_write_report", lambda *args: None)
    output = str(tmp_path / "r.json")
    assert cli.main(["docunet", "pred", "--output", output, "--ms-ssim-only"]) == 0
    assert tuple(seen["conventions"]) == docunet.CONVENTIONS == docunet.DEFAULT_CONVENTIONS
    assert set(docunet.PROTOCOL["conventions"]) == set(docunet.CONVENTIONS)


def test_cli_calibrate_builds_siftflow_only_for_docunet_runs(monkeypatch, tmp_path):
    builds = []
    monkeypatch.setattr(cli.siftflow, "build_siftflow", lambda *a, **k: builds.append(a) or "b")
    monkeypatch.setattr(cli.calibration, "run_calibration", lambda *a, **k: {})
    monkeypatch.setattr(
        cli.calibration, "summarize_calibration", lambda *a: {"gates": [], "all_pass": True}
    )
    common = ["calibrate", "--reports-dir", str(tmp_path), "--paperedge-dir", str(tmp_path)]
    assert cli.main([*common, "--run", "paperedge-ocr", "--rerun", "--no-ocr"]) == 0
    assert builds == []
    assert cli.main([*common, "--run", "paperedge-once", "--rerun", "--no-ocr"]) == 0
    assert len(builds) == 1
    write_json_atomic(tmp_path / "paperedge-once.json", {})
    assert cli.main([*common, "--run", "paperedge-once", "--no-ocr"]) == 0
    assert len(builds) == 1
