"""The four evaluator calibration checks (named in calibration.json) on synthetic reports,
with no images and no SIFT flow."""

import json

import numpy as np
import pytest

from tools.benchmark import calibration, docunet, ocr
from tools.benchmark.reporting import write_json_atomic


def _docunet_report(name, values, *, shift=None):
    spec = calibration.RUNS[name]
    shift = shift or {}
    rows = []
    for (k, m), value in values.items():
        rows.append(
            {
                "id": f"{k}_{m}", "k": k, "m": m, "rotated_180": False,
                "ms_ssim": value["ms_ssim"] + shift.get("ms_ssim", 0.0),
                "ld": value["ld"] + shift.get("ld", 0.0),
                "ad": value["ad"], "ad_exact_lstsq": value["ad"] + shift.get("exact", 0.0),
                "ad_lsqr_4iter": value["ad"] + shift.get("lsqr", 0.0),
            }
        )  # fmt: skip
    return {
        "comparable": True,
        "config": {"conventions": list(spec["conventions"]), "ref_resize": spec["ref_resize"]},
        "rows": rows,
        "schema": docunet.REPORT_SCHEMA,
        "summary": docunet.summarize(rows, spec["conventions"]),
    }


def _values(ms, ld, ad, keys):
    return {key: {"ms_ssim": ms, "ld": ld, "ad": ad} for key in keys}


ALL = [(k, m) for k in range(1, 66) for m in (1, 2)]
EXCL = [(k, m) for k, m in ALL if k != 64]


def test_paperedge_compares_means_with_both_solvers():
    values = _values(0.47, 8.5, 0.39, ALL)
    adres = {"rows": {f"{k}_{m}": v for (k, m), v in values.items()}, "file_mean_row": {}}
    ok = calibration.gate_paperedge_matlab(_docunet_report("paperedge-once", values), adres)
    assert ok["status"] == "pass" and len(ok["checks"]) == 4
    assert {c.get("ad_solver") for c in ok["checks"] if c["metric"] == "ad"} == {
        "lsqr_4iter", "exact_lstsq"
    }  # fmt: skip
    near = calibration.gate_paperedge_matlab(
        _docunet_report("paperedge-once", values, shift={"ld": 0.004, "exact": 0.005}), adres
    )
    assert near["pass"]  # exact_lstsq is informational
    assert [c["pass"] for c in near["checks"]] == [True, True, True, False]
    far = calibration.gate_paperedge_matlab(
        _docunet_report("paperedge-once", values, shift={"ld": 0.006}), adres
    )
    assert far["status"] == "fail"
    assert calibration.gate_paperedge_matlab(None, adres)["status"] == "not_run"
    wrong = _docunet_report("paperedge-once", values)
    wrong["config"]["ref_resize"] = "uvdoc_view2_double"
    with pytest.raises(ValueError):
        calibration.gate_paperedge_matlab(wrong, adres)


def test_published_tables_use_rounding_and_absolute_tolerances(tmp_path):
    good = _values(0.4722, 7.9815, 0.3668, EXCL)
    write_json_atomic(tmp_path / "paperedge-uvdoc.json", _docunet_report("paperedge-uvdoc", good))
    bad = _values(0.4726, 7.98, 0.367, EXCL)
    write_json_atomic(tmp_path / "uvdoc-uvdoc.json", _docunet_report("uvdoc-uvdoc", bad))
    gates = {gate["gate"]: gate for gate in calibration.evaluate_gates(tmp_path)}
    assert (
        gates["paperedge_matlab"]["status"] == "not_run"
        and gates["ocr_table"]["status"] == "not_run"
    )
    assert gates["uvdoc_table_paperedge"]["status"] == "pass"
    assert gates["uvdoc_table_paperedge"]["checks"][0]["tolerance"] == pytest.approx(0.0005)
    assert gates["uvdoc_table_paperedge"]["checks"][1]["tolerance"] == pytest.approx(0.005)
    assert gates["uvdoc_table_uvdoc"]["status"] == "fail"  # 0.4726 vs 0.544 +- 0.002
    report = _docunet_report("paperedge-uvdoc", _values(0.4724, 7.98, 0.3664, EXCL))
    rounding = calibration._docunet_gate(
        "uvdoc_table_paperedge", "x", report,
        "paperedge-uvdoc", calibration.UVDOC_TABLE_PAPEREDGE, "rounding",
    )  # fmt: skip
    assert [c["pass"] for c in rounding["checks"]] == [True, True, False, False]


def _ocr_report(scale, version="tesseract v5.0.1.20220118"):
    return {
        "config": {
            "protocols": ["doctr"],
            "tesseract_config": {"doctr": ""},
        },
        "schema": ocr.REPORT_SCHEMA,
        "summary": {
            "doctr": {
                setting: {metric: target * scale for metric, target in targets.items()}
                for setting, targets in calibration.OCR_TABLE_TARGET.items()
            }
        },
        "tesseract": {"version_first_line": version},
    }


def test_ocr_checks_tolerances_engine_version_and_protocol():
    assert calibration.gate_ocr_table(_ocr_report(1.019))["status"] == "pass"
    assert calibration.gate_ocr_table(_ocr_report(0.979))["status"] == "fail"
    gate = calibration.gate_ocr_table(_ocr_report(1.0))
    assert gate["tesseract_version"] == "tesseract v5.0.1.20220118"
    assert gate["checks"][-1]["metric"] == "tesseract_version" and gate["checks"][-1]["pass"]
    assert "'tesseract v5.0.1.20220118' must be 5.0.1 -> ok" in calibration.format_gates([gate])
    other = calibration.gate_ocr_table(_ocr_report(1.0, version="tesseract 5.5.3"))
    assert other["status"] == "fail"
    assert [check["pass"] for check in other["checks"]] == [True, True, True, True, False]
    assert calibration.gate_ocr_table(_ocr_report(1.0, version=None))["status"] == "fail"
    for key, value in (
        ("protocols", ["doctr", "uvdoc"]),
        ("tesseract_config", {"doctr": "--psm 6"}),
    ):
        report = _ocr_report(1.0)
        report["config"][key] = value
        with pytest.raises(ValueError, match=key):
            calibration.gate_ocr_table(report)


def test_calibrate_reads_adres_reuses_reports_and_writes_relative_paths(tmp_path):
    values = {(k, m): {"ms_ssim": 0.4 + k / 1000, "ld": 8 + m / 10, "ad": 0.39} for k, m in ALL}
    lines = [f"{k} {m} {v['ad']} {v['ms_ssim']} {v['ld']}" for (k, m), v in values.items()]
    means = [np.mean([v[key] for v in values.values()]) for key in ("ad", "ms_ssim", "ld")]
    lines.append(f"33 1.5 {means[0]} {means[1]} {means[2]}")
    (tmp_path / calibration.ADRES_NAME).write_text("\n".join(lines) + "\n")
    (tmp_path / "bad.txt").write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError):
        calibration.parse_adres(tmp_path / "bad.txt")
    reports = tmp_path / "reports"
    reports.mkdir()
    write_json_atomic(reports / "paperedge-once.json", _docunet_report("paperedge-once", values))
    status = calibration.run_calibration(reports, paperedge_dir=tmp_path, log=lambda _: None)
    assert status["paperedge-once"] == "reused"
    assert status["uvdoc-uvdoc"].startswith("skipped")
    adres = json.loads((reports / "paperedge-adres.json").read_text())
    assert len(adres["rows"]) == 130 and adres["rows"]["7_2"]["ms_ssim"] == pytest.approx(0.407)
    assert adres["file_mean_row"]["ld"] == pytest.approx(8.15)
    assert adres["evidence"] == calibration.CALIBRATION_EVIDENCE
    summary = calibration.summarize_calibration(reports, tmp_path / "calibration.json")
    written = json.loads((tmp_path / "calibration.json").read_text())
    assert written["schema"] == calibration.SCHEMA
    entry = written["reports"]["paperedge-once"]  # a report without its own label
    assert entry["path"] == "reports/paperedge-once.json" and entry["evidence_in_file"] is False
    assert entry["evidence"] == written["evidence"] == calibration.CALIBRATION_EVIDENCE
    assert [gate["status"] for gate in summary["gates"]] == ["pass", *["not_run"] * 3]
    assert summary["all_pass"] is False
    assert "paperedge_matlab: PASS" in calibration.format_gates(summary["gates"])
