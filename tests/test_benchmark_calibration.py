"""Calibration gates evaluated on synthetic reports (no images, no SIFT flow)."""

import pytest

from flattener.benchmark import calibration, docunet, ocr
from flattener.benchmark.reporting import write_json_atomic


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
    assert gate["tesseract_build"] is None and gate["tesseract_leptonica_version"] is None
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
