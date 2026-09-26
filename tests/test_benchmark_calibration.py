"""Calibration gates evaluated on synthetic reports (no images, no SIFT flow)."""

import json

import numpy as np
import pytest

from flattener.benchmark import calibration, docunet, ocr
from flattener.benchmark.reporting import write_json_atomic


def _adres(path, rows):
    lines = [f"{k} {m} {v['ad']} {v['ms_ssim']} {v['ld']}" for (k, m), v in rows.items()]
    means = [np.mean([v[key] for v in rows.values()]) for key in ("ad", "ms_ssim", "ld")]
    lines.append(f"33 1.5 {means[0]} {means[1]} {means[2]}")
    path.write_text("\n".join(lines) + "\n")


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


def test_parse_adres_reads_rows_and_mean_row(tmp_path):
    rng = np.random.default_rng(0)
    rows = {
        key: {"ad": rng.random(), "ms_ssim": rng.random(), "ld": 10 * rng.random()} for key in ALL
    }
    _adres(tmp_path / "adres.txt", rows)
    parsed = calibration.parse_adres(tmp_path / "adres.txt")
    assert len(parsed["rows"]) == 130
    assert parsed["rows"]["7_2"]["ld"] == pytest.approx(rows[(7, 2)]["ld"])
    assert parsed["file_mean_row"]["ms_ssim"] == pytest.approx(
        np.mean([v["ms_ssim"] for v in rows.values()])
    )
    (tmp_path / "bad.txt").write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError):
        calibration.parse_adres(tmp_path / "bad.txt")


def test_g1_compares_means_with_both_solvers(tmp_path):
    values = _values(0.47, 8.5, 0.39, ALL)
    adres = {"rows": {f"{k}_{m}": v for (k, m), v in values.items()}, "file_mean_row": {}}
    ok = calibration.gate_g1(_docunet_report("paperedge-once", values), adres)
    assert ok["status"] == "pass" and len(ok["checks"]) == 4
    assert {c.get("ad_solver") for c in ok["checks"] if c["metric"] == "ad"} == {
        "lsqr_4iter", "exact_lstsq"
    }  # fmt: skip
    near = calibration.gate_g1(
        _docunet_report("paperedge-once", values, shift={"ld": 0.004, "exact": 0.005}), adres
    )
    assert near["pass"]  # exact_lstsq is informational
    assert [c["pass"] for c in near["checks"]] == [True, True, True, False]
    far = calibration.gate_g1(_docunet_report("paperedge-once", values, shift={"ld": 0.006}), adres)
    assert far["status"] == "fail"
    assert calibration.gate_g1(None, adres)["status"] == "not_run"
    wrong = _docunet_report("paperedge-once", values)
    wrong["config"]["ref_resize"] = "uvdoc_view2_double"
    with pytest.raises(ValueError):
        calibration.gate_g1(wrong, adres)


def test_g2_uses_rounding_and_g3_uses_absolute_tolerances(tmp_path):
    good = _values(0.4722, 7.9815, 0.3668, EXCL)
    write_json_atomic(tmp_path / "paperedge-uvdoc.json", _docunet_report("paperedge-uvdoc", good))
    bad = _values(0.4726, 7.98, 0.367, EXCL)
    write_json_atomic(tmp_path / "uvdoc-uvdoc.json", _docunet_report("uvdoc-uvdoc", bad))
    gates = {gate["gate"]: gate for gate in calibration.evaluate_gates(tmp_path)}
    assert gates["G1"]["status"] == "not_run" and gates["G4"]["status"] == "not_run"
    assert gates["G2"]["status"] == "pass"
    assert gates["G2"]["checks"][0]["tolerance"] == pytest.approx(0.0005)
    assert gates["G2"]["checks"][1]["tolerance"] == pytest.approx(0.005)
    assert gates["G3"]["status"] == "fail"  # 0.4726 vs 0.544 +- 0.002
    rounding = calibration._docunet_gate(
        "G2", "x", _docunet_report("paperedge-uvdoc", _values(0.4724, 7.98, 0.3664, EXCL)),
        "paperedge-uvdoc", calibration.G2_TARGET, "rounding",
    )  # fmt: skip
    assert [c["pass"] for c in rounding["checks"]] == [True, True, False, False]


def _ocr_report(scale, version="tesseract v5.0.1.20220118"):
    return {
        "config": {
            "protocols": ["doctr"],
            "references_verified": True,
            "tesseract_config": {"doctr": ""},
        },
        "schema": ocr.REPORT_SCHEMA,
        "summary": {
            "doctr": {
                setting: {metric: target * scale for metric, target in targets.items()}
                for setting, targets in calibration.G4_TARGET.items()
            }
        },
        "tesseract": {"version_first_line": version},
    }


def test_g4_relative_tolerance_on_doctr_settings():
    assert calibration.gate_g4(_ocr_report(1.019))["status"] == "pass"
    assert calibration.gate_g4(_ocr_report(0.979))["status"] == "fail"
    gate = calibration.gate_g4(_ocr_report(1.0))
    assert gate["tesseract_version"] == "tesseract v5.0.1.20220118"
    assert gate["checks"][-1]["metric"] == "tesseract_version" and gate["checks"][-1]["pass"]
    assert "'tesseract v5.0.1.20220118' must be 5.0.1 -> ok" in calibration.format_gates([gate])


def test_g4_requires_tesseract_501_and_the_plain_doctr_configuration():
    other = calibration.gate_g4(_ocr_report(1.0, version="tesseract 5.5.3"))
    assert other["status"] == "fail"
    assert [check["pass"] for check in other["checks"]] == [True, True, True, True, False]
    assert "-> FAIL" in calibration.format_gates([other])
    assert calibration.gate_g4(_ocr_report(1.0, version=None))["status"] == "fail"
    for key, value in (
        ("protocols", ["doctr", "uvdoc"]),
        ("tesseract_config", {"doctr": "--psm 6"}),
        ("references_verified", False),
    ):
        report = _ocr_report(1.0)
        report["config"][key] = value
        with pytest.raises(ValueError, match=key):
            calibration.gate_g4(report)


def test_run_and_summarize_reuse_reports_and_write_relative_paths(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    values = _values(0.47, 8.5, 0.39, ALL)
    write_json_atomic(reports / "paperedge-once.json", _docunet_report("paperedge-once", values))
    adres = {
        "rows": {f"{k}_{m}": v for (k, m), v in values.items()},
        "file_mean_row": {"ad": 0.39, "ld": 8.5, "ms_ssim": 0.47},
        "schema": calibration.ADRES_SCHEMA,
    }
    write_json_atomic(reports / "paperedge-adres.json", adres)
    status = calibration.run_calibration(reports, log=lambda message: None)
    assert status["paperedge-once"] == "reused"
    assert status["uvdoc-uvdoc"].startswith("skipped")
    summary = calibration.summarize_calibration(reports, tmp_path / "calibration.json")
    written = json.loads((tmp_path / "calibration.json").read_text())
    assert written["schema"] == calibration.SCHEMA
    assert written["reports"]["paperedge-once"]["path"] == "reports/paperedge-once.json"
    assert [gate["status"] for gate in summary["gates"]] == [
        "pass",
        "not_run",
        "not_run",
        "not_run",
    ]
    assert summary["all_pass"] is False
    assert "G1: PASS" in calibration.format_gates(summary["gates"])
