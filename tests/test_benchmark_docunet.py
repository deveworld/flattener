"""DocUNet conventions, report schema and the command line, without SIFT flow."""

import json

import numpy as np
import pytest
from PIL import Image

from flattener.benchmark import __main__ as cli
from flattener.benchmark import docunet, metrics
from flattener.benchmark.reporting import prediction_record, prediction_status, write_json_atomic


def _row(k, m, rotated, value):
    return {
        "k": k, "m": m, "rotated_180": rotated, "ms_ssim": value, "ld": 10 * value,
        "ad": value / 2, "ad_exact_lstsq": value / 2, "ad_lsqr_4iter": value / 3,
    }  # fmt: skip


def test_conventions_select_the_published_image_sets_and_average_their_own_rows():
    raw = docunet.convention_views("130_raw")
    rotated = docunet.convention_views("130_rot64")
    excluded = docunet.convention_views("128_excl64")
    assert len(raw) == len(rotated) == 130 and len(excluded) == 128
    assert not any(flag for _, _, flag in raw)
    assert [(k, m) for k, m, flag in rotated if flag] == [(64, 1), (64, 2)]
    assert all(k != 64 for k, _, _ in excluded)
    assert {(k, m) for k, m, _ in raw} == {(k, m) for k in range(1, 66) for m in (1, 2)}
    assert len(docunet.required_views(["130_rot64", "128_excl64"])) == 130
    assert len(docunet.required_views(["130_raw", "130_rot64"])) == 132
    with pytest.raises(ValueError):
        docunet.convention_views("129")
    with pytest.raises(ValueError):
        docunet.validate_conventions([])
    rows = [_row(k, m, False, float(k)) for k in range(1, 66) for m in (1, 2)]
    rows += [_row(64, m, True, 100.0) for m in (1, 2)]
    summary = docunet.summarize(rows, ["130_raw", "130_rot64", "128_excl64"])
    assert summary["130_raw"]["ms_ssim"] == pytest.approx(33.0)
    others = [k for k in range(1, 66) if k != 64 for _ in (1, 2)]
    assert summary["130_rot64"]["ms_ssim"] == pytest.approx(np.mean(others + [100.0, 100.0]))
    assert summary["128_excl64"]["ld"] == pytest.approx(10 * np.mean(others))
    assert summary["128_excl64"]["n"] == 128 and summary["128_excl64"]["views"]["2"]["n"] == 64
    assert summary["128_excl64"]["ad_lsqr_4iter"] == pytest.approx(np.mean(others) / 3)


def test_cli_writes_the_report_without_siftflow(tiny_docunet, tmp_path, capsys):
    scans, predictions = tiny_docunet
    output = tmp_path / "out.json"
    code = cli.main(
        [
            "docunet", str(predictions), "--output", str(output), "--ms-ssim-only",
            "--references-dir", str(scans), "--no-verify-references", "--workers", "1",
            "--convention", "130_rot64", "--convention", "128_excl64", "--quiet",
        ]
    )  # fmt: skip
    assert code == 0
    report = json.loads(output.read_text())
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
    assert {"prediction_sha256", "reference_sha256", "metric_shape_hw"} <= set(report["rows"][0])
    assert "128_excl64" in json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit, match="exists"):
        cli.main(["docunet", str(predictions), "--output", str(output), "--ms-ssim-only"])
    with pytest.raises(FileExistsError):
        write_json_atomic(output, report)


def _record(predictions, failed):
    items = [
        {"output": f"{k}_{m}.png", "action": "written", "status": "success"}
        for k in range(1, 66)
        for m in (1, 2)
    ]
    for item in items:
        if item["output"] in failed:
            item.update(failed[item["output"]], reason="fold_or_degenerate")
    record = {"schema": "flattener-dewarp-predictions-2", "model": {"sha256": "a"}, "items": items}
    (predictions / "predictions.json").write_text(json.dumps(record))


def test_reports_count_recorded_failures_and_unrecorded_outputs(tiny_docunet):
    scans, predictions = tiny_docunet
    arguments = dict(
        pattern="{k}_{m}.png", build=None, references_dir=scans, verify_references=False, workers=1
    )
    report = docunet.evaluate_docunet(predictions, conventions=["128_excl64"], **arguments)
    assert report["predictions_record"] is None
    assert report["summary"]["128_excl64"]["prediction_failures"] is None
    forced = {"status": "invalid_transform", "forced": True}
    (predictions / "5_2.png").unlink()
    _record(predictions, {"3_1.png": forced, "5_2.png": {**forced, "action": "not_written"}})
    with pytest.raises(FileNotFoundError, match="records 1 of them as failed checks"):
        docunet.evaluate_docunet(predictions, conventions=["128_excl64"], **arguments)
    Image.fromarray(np.zeros((20, 16, 3), np.uint8)).save(predictions / "5_2.png")
    _record(predictions, {"3_1.png": forced, "5_2.png": forced, "64_1.png": forced})
    report = docunet.evaluate_docunet(
        predictions, conventions=["130_raw", "128_excl64"], **arguments
    )
    assert report["predictions_record"]["failed"] == 3
    assert report["summary"]["130_raw"]["prediction_failures"] == 3
    assert report["summary"]["128_excl64"]["prediction_failures"] == 2
    assert report["summary"]["128_excl64"]["views"]["2"]["prediction_failures"] == 1
    statuses = {row["id"]: row["prediction_status"] for row in report["rows"]}
    assert statuses["3_1"] == "invalid_transform" and statuses["3_2"] == "success"
    assert report["summary"]["130_raw"]["prediction_unrecorded"] == 0
    # An interrupted run whose record does not list 3_2.png, left from an earlier run.
    _record(predictions, {})
    record = json.loads((predictions / "predictions.json").read_text())
    record["items"] = [item for item in record["items"] if item["output"] != "3_2.png"]
    record["complete"] = False
    (predictions / "predictions.json").write_text(json.dumps(record))
    assert prediction_status(prediction_record(predictions), "3_2.png") == "unrecorded"
    report = docunet.evaluate_docunet(predictions, conventions=["128_excl64"], **arguments)
    means = report["summary"]["128_excl64"]
    assert means["prediction_failures"] == 1 and means["prediction_unrecorded"] == 1
    assert means["views"]["1"]["prediction_failures"] == 0


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


def test_score_document_rotates_for_rot64_and_prepares_the_scan_once_or_per_view(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(metrics, "METRIC_AREA", 300)
    scan, prediction, scan_path, prediction_path = _asymmetric_document(tmp_path)
    selected = [view for view in docunet.required_views(["130_raw", "130_rot64"]) if view[0] == 64]
    assert selected == [(64, 1, False), (64, 1, True), (64, 2, False), (64, 2, True)]
    views = tuple(docunet.ViewTask(m, rotated, prediction_path) for _, m, rotated in selected)
    task = docunet.DocumentTask(64, scan_path, views, "once", metrics.DEFAULT_AD_SOLVER)
    raw, rotated, *_ = docunet.score_document(task, None)
    assert (raw["rotated_180"], rotated["rotated_180"]) == (False, True)
    assert raw["ms_ssim"] == metrics.score_pair(prediction, scan).ms_ssim
    expected = metrics.score_pair(np.ascontiguousarray(np.rot90(prediction, 2)), scan)
    assert rotated["ms_ssim"] == expected.ms_ssim
    assert abs(rotated["ms_ssim"] - raw["ms_ssim"]) > 1e-3

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
    assert calls == [1] and [row["reference_resizes"] for row in once] == [1, 1]
    assert once[0]["ms_ssim"] == once[1]["ms_ssim"]
    calls.clear()
    task = docunet.DocumentTask(3, scan_path, views, "uvdoc_view2_double", "lsqr_4iter")
    uvdoc = docunet.score_document(task, None)
    assert calls == [1, 2] and [row["reference_resizes"] for row in uvdoc] == [1, 2]
    assert uvdoc[0]["ms_ssim"] == once[0]["ms_ssim"]
    twice = original(scan, resizes=2)
    assert uvdoc[1]["ms_ssim"] == metrics.score_prepared(prediction, twice).ms_ssim
    assert uvdoc[1]["ms_ssim"] != uvdoc[0]["ms_ssim"]
