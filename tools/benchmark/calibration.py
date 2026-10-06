"""Check evaluator scores against published reference results.

The checks compare image metrics with the PaperEdge and UVDoc reference outputs,
and OCR results with the DocScanner table.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from flattener.evidence import evidence

from . import docunet, ocr
from .reporting import environment, mean, utc_now, write_json_atomic
from .siftflow import SiftFlowBuild

SCHEMA = "flattener-benchmark-calibration-1.0"
ADRES_SCHEMA = "flattener-benchmark-adres-1.0"
PREDICTION_PATTERN = "{k}_{m}.png"
ADRES_NAME = "adres.txt"
GATE_AD_SOLVER = "lsqr_4iter"

RUNS: dict[str, dict[str, Any]] = {
    "paperedge-once": {
        "kind": "docunet",
        "source": "paperedge",
        "conventions": ("130_raw", "128_excl64"),
        "ref_resize": "once",
    },
    "paperedge-uvdoc": {
        "kind": "docunet",
        "source": "paperedge",
        "conventions": ("128_excl64",),
        "ref_resize": "uvdoc_view2_double",
    },
    "uvdoc-uvdoc": {
        "kind": "docunet",
        "source": "uvdoc",
        "conventions": ("128_excl64",),
        "ref_resize": "uvdoc_view2_double",
    },
    "paperedge-ocr": {"kind": "ocr", "source": "paperedge", "protocols": ("doctr",)},
}

PAPEREDGE_MATLAB_TOLERANCE = {"ms_ssim": 1e-5, "ld": 0.005, "ad": 0.001}
UVDOC_TABLE_PAPEREDGE = {"ms_ssim": (0.472, 3), "ld": (7.98, 2), "ad": (0.367, 3)}
UVDOC_TABLE_UVDOC = {"ms_ssim": (0.544, 0.002), "ld": (6.83, 0.03), "ad": (0.315, 0.003)}
OCR_TABLE_TARGET = {
    "setting1_doctr_60": {"ed": 375.60, "cer": 0.1541},
    "setting2_dewarpnet_50": {"ed": 777.76, "cer": 0.2014},
}
OCR_RELATIVE_TOLERANCE = 0.02
# Reference outputs for evaluator calibration.
ARCHIVES = {
    "paperedge": {
        "name": "docunet_benchmark_paperedge.zip (res_test/, 130 PNG + adres.txt)",
        "url": "https://drive.google.com/file/d/1QM3Y5Ty96ydVCQPNqR0_bnMG9oqIQkGm/view",
    },
    "uvdoc": {
        "name": "DocUnet_results.zip (DocUnet_results/, 128 PNG)",
        "url": "https://igl.ethz.ch/projects/uvdoc/DocUnet_results.zip",
    },
}
SOURCES = {
    "paperedge_matlab": "PaperEdge res_test/adres.txt (MATLAB per-image AD, MS-SSIM, LD)",
    "uvdoc_table_paperedge": "UVDoc paper Table 2, PaperEdge row (https://arxiv.org/abs/2302.02887)",
    "uvdoc_table_uvdoc": "UVDoc paper Table 2, UVDoc row; https://igl.ethz.ch/projects/uvdoc/DocUnet_results.zip",
    "ocr_table": "DocScanner paper OCR table, PaperEdge row (https://arxiv.org/abs/2110.14968)",
}


def parse_adres(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse PaperEdge ``adres.txt``: rows ``k m AD MS-SSIM LD`` and a final mean row."""
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] != 5 or values.shape[0] < 2:
        raise ValueError(f"{path} is not a PaperEdge adres.txt table")
    rows = {}
    for k, m, ad, ms_ssim, ld in values[:-1]:
        if k != round(k) or m not in (1.0, 2.0):
            raise ValueError(f"unexpected image id {k} {m} in {path}")
        rows[f"{int(k)}_{int(m)}"] = {"ad": float(ad), "ld": float(ld), "ms_ssim": float(ms_ssim)}
    _, _, ad, ms_ssim, ld = values[-1]
    return {
        "file_mean_row": {"ad": float(ad), "ld": float(ld), "ms_ssim": float(ms_ssim)},
        "rows": rows,
        "schema": ADRES_SCHEMA,
        "source": str(Path(path).resolve()),
    }


def _check(
    metric: str,
    measured: float | None,
    target: float,
    *,
    tolerance: float,
    kind: str,
    gating: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    if measured is None or not math.isfinite(measured):
        passed = False
        difference = None
    else:
        difference = measured - target
        if kind == "relative":
            passed = abs(difference) <= tolerance * abs(target) + 1e-12
        else:
            passed = abs(difference) <= tolerance + 1e-12
    return {
        "difference": difference,
        "gating": gating,
        "measured": measured,
        "metric": metric,
        "pass": passed,
        "target": target,
        "tolerance": tolerance,
        "tolerance_kind": kind,
        **extra,
    }


def _gate(name: str, description: str, checks: list[dict[str, Any]], **extra: Any) -> dict:
    gating = [check for check in checks if check["gating"]]
    return {
        "checks": checks,
        "description": description,
        "gate": name,
        "pass": bool(gating) and all(check["pass"] for check in gating),
        "source": SOURCES[name],
        "status": "pass" if gating and all(check["pass"] for check in gating) else "fail",
        **extra,
    }


def _not_run(name: str, description: str, reason: str) -> dict[str, Any]:
    return {
        "checks": [],
        "description": description,
        "gate": name,
        "pass": False,
        "reason": reason,
        "source": SOURCES[name],
        "status": "not_run",
    }


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _require_run(report: dict[str, Any], name: str) -> None:
    spec = RUNS[name]
    if spec["kind"] == "docunet":
        config = report.get("config", {})
        if report.get("schema") != docunet.REPORT_SCHEMA:
            raise ValueError(f"{name}: not a DocUNet report")
        if tuple(config.get("conventions", ())) != spec["conventions"]:
            raise ValueError(
                f"{name}: conventions {config.get('conventions')} != {spec['conventions']}"
            )
        if config.get("ref_resize") != spec["ref_resize"]:
            raise ValueError(
                f"{name}: ref_resize {config.get('ref_resize')} != {spec['ref_resize']}"
            )
        if not report.get("comparable"):
            raise ValueError(f"{name}: report needs SIFT flow metrics")
    else:
        config = report.get("config", {})
        if report.get("schema") != ocr.REPORT_SCHEMA:
            raise ValueError(f"{name}: not an OCR report")
        if tuple(config.get("protocols", ())) != spec["protocols"]:
            raise ValueError(f"{name}: protocols {config.get('protocols')} != {spec['protocols']}")
        expected = {protocol: ocr.PROTOCOL_CONFIG[protocol] for protocol in spec["protocols"]}
        if config.get("tesseract_config") != expected:
            raise ValueError(
                f"{name}: tesseract_config {config.get('tesseract_config')} != {expected}"
            )


def gate_paperedge_matlab(
    once: dict[str, Any] | None, adres: dict[str, Any] | None
) -> dict[str, Any]:
    description = "PaperEdge outputs, 130_raw, mean difference to MATLAB adres.txt"
    if once is None or adres is None:
        return _not_run(
            "paperedge_matlab", description, "paperedge-once.json or paperedge-adres.json missing"
        )
    _require_run(once, "paperedge-once")
    rows = {row["id"]: row for row in once["rows"] if not row["rotated_180"]}
    ids = [f"{k}_{m}" for k, m, _ in docunet.convention_views("130_raw")]
    if set(adres["rows"]) != set(ids):
        raise ValueError("adres.txt does not list the 130 DocUNet images")
    matlab = {
        metric: mean([adres["rows"][item][metric] for item in ids])
        for metric in ("ms_ssim", "ld", "ad")
    }
    measured = once["summary"]["130_raw"]
    per_image = {}
    for name, key, matlab_key in (
        ("ms_ssim", "ms_ssim", "ms_ssim"),
        ("ld", "ld", "ld"),
        ("ad_exact_lstsq", "ad_exact_lstsq", "ad"),
        ("ad_lsqr_4iter", "ad_lsqr_4iter", "ad"),
    ):
        differences = [rows[item][key] - adres["rows"][item][matlab_key] for item in ids]
        worst = int(np.argmax(np.abs(differences)))
        per_image[name] = {
            "max_abs_difference": float(abs(differences[worst])),
            "max_at": ids[worst],
            "mean_abs_difference": float(np.mean(np.abs(differences))),
        }
    checks = [
        _check(
            "ms_ssim",
            measured["ms_ssim"],
            matlab["ms_ssim"],
            tolerance=PAPEREDGE_MATLAB_TOLERANCE["ms_ssim"],
            kind="absolute",
        ),
        _check(
            "ld",
            measured["ld"],
            matlab["ld"],
            tolerance=PAPEREDGE_MATLAB_TOLERANCE["ld"],
            kind="absolute",
        ),
        _check(
            "ad",
            measured["ad_lsqr_4iter"],
            matlab["ad"],
            tolerance=PAPEREDGE_MATLAB_TOLERANCE["ad"],
            kind="absolute",
            ad_solver="lsqr_4iter",
        ),
        _check(
            "ad",
            measured["ad_exact_lstsq"],
            matlab["ad"],
            tolerance=PAPEREDGE_MATLAB_TOLERANCE["ad"],
            kind="absolute",
            ad_solver="exact_lstsq",
            gating=False,
        ),
    ]
    return _gate(
        "paperedge_matlab",
        description,
        checks,
        convention="130_raw",
        ref_resize="once",
        matlab_file_mean_row=adres["file_mean_row"],
        per_image=per_image,
    )


def _docunet_gate(
    name: str,
    description: str,
    report: dict[str, Any] | None,
    run: str,
    targets: dict[str, tuple[float, float]],
    kind: str,
) -> dict[str, Any]:
    if report is None:
        return _not_run(name, description, f"{run}.json missing")
    _require_run(report, run)
    measured = report["summary"]["128_excl64"]
    checks = []
    for metric in ("ms_ssim", "ld"):
        target, tolerance = targets[metric]
        if kind == "rounding":
            tolerance = 0.5 * 10.0**-tolerance
        checks.append(_check(metric, measured[metric], target, tolerance=tolerance, kind=kind))
    target, tolerance = targets["ad"]
    if kind == "rounding":
        tolerance = 0.5 * 10.0**-tolerance
    for solver in docunet.metrics.AD_SOLVERS:
        checks.append(
            _check(
                "ad",
                measured[f"ad_{solver}"],
                target,
                tolerance=tolerance,
                kind=kind,
                ad_solver=solver,
                gating=solver == GATE_AD_SOLVER,
            )
        )
    return _gate(
        name,
        description,
        checks,
        convention="128_excl64",
        ref_resize="uvdoc_view2_double",
        n=measured["n"],
    )


def gate_ocr_table(report: dict[str, Any] | None) -> dict[str, Any]:
    description = "DocTr OCR protocol on PaperEdge outputs vs DocScanner table (Tesseract 5.0.1)"
    if report is None:
        return _not_run("ocr_table", description, "paperedge-ocr.json missing")
    _require_run(report, "paperedge-ocr")
    checks = []
    for setting, targets in OCR_TABLE_TARGET.items():
        measured = report["summary"]["doctr"][setting]
        for metric, target in targets.items():
            checks.append(
                _check(
                    metric,
                    measured[metric],
                    target,
                    tolerance=OCR_RELATIVE_TOLERANCE,
                    kind="relative",
                    setting=setting,
                )
            )
    tesseract = report.get("tesseract", {})
    version = tesseract.get("version_first_line")
    checks.append(
        {
            "difference": None,
            "gating": True,
            "measured": version,
            "metric": "tesseract_version",
            "pass": ocr.is_calibrated_tesseract(version),
            "target": ocr.CALIBRATED_TESSERACT_VERSION,
            "tolerance": None,
            "tolerance_kind": "requirement",
        }
    )
    return _gate(
        "ocr_table",
        description,
        checks,
        tesseract_version=tesseract.get("version_first_line"),
        tesseract_leptonica_version=ocr.leptonica_version(tesseract.get("version")),
        tesseract_build=ocr.calibrated_build(tesseract),
        traineddata=tesseract.get("traineddata"),
    )


def evaluate_gates(reports_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = Path(reports_dir)
    once = _load(root / "paperedge-once.json")
    adres = _load(root / "paperedge-adres.json")
    return [
        gate_paperedge_matlab(once, adres),
        _docunet_gate(
            "uvdoc_table_paperedge",
            "PaperEdge outputs, 128_excl64 + uvdoc_view2_double vs UVDoc Table 2",
            _load(root / "paperedge-uvdoc.json"),
            "paperedge-uvdoc",
            UVDOC_TABLE_PAPEREDGE,
            "rounding",
        ),
        _docunet_gate(
            "uvdoc_table_uvdoc",
            "UVDoc DocUnet_results, 128_excl64 + uvdoc_view2_double vs UVDoc Table 2",
            _load(root / "uvdoc-uvdoc.json"),
            "uvdoc-uvdoc",
            UVDOC_TABLE_UVDOC,
            "absolute",
        ),
        gate_ocr_table(_load(root / "paperedge-ocr.json")),
    ]


# Published PaperEdge and UVDoc outputs scored to check the evaluator.
CALIBRATION_EVIDENCE = evidence("evaluator_calibration", {"provenance": "external_pretrained"})


def run_calibration(
    reports_dir: str | os.PathLike[str],
    *,
    paperedge_dir: str | os.PathLike[str] | None = None,
    uvdoc_dir: str | os.PathLike[str] | None = None,
    build: SiftFlowBuild | None = None,
    engine: ocr.OcrEngine | None = None,
    references_dir: str | os.PathLike[str] = docunet.DEFAULT_REFERENCES_DIR,
    reference_pattern: str = docunet.DEFAULT_REFERENCE_PATTERN,
    workers: int = docunet.DEFAULT_WORKERS,
    runs: tuple[str, ...] = tuple(RUNS),
    rerun: bool = False,
    log: Callable[[str], None] = print,
    docunet_progress: Callable[[dict[str, Any], int, int], None] | None = None,
    ocr_progress: Callable[..., None] | None = None,
) -> dict[str, str]:
    """Write the calibration reports that are missing (or all with ``rerun``).

    Returns ``{run: status}``. A run without its inputs is skipped.
    """
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    sources = {"paperedge": paperedge_dir, "uvdoc": uvdoc_dir}
    status: dict[str, str] = {}
    if paperedge_dir is not None:
        adres_path = root / "paperedge-adres.json"
        if rerun or not adres_path.is_file():
            adres = parse_adres(Path(paperedge_dir) / ADRES_NAME)
            adres["evidence"] = CALIBRATION_EVIDENCE
            write_json_atomic(adres_path, adres, overwrite=True)
    for name in runs:
        spec = RUNS[name]
        path = root / f"{name}.json"
        if path.is_file() and not rerun:
            status[name] = "reused"
            continue
        source = sources[spec["source"]]
        if source is None:
            status[name] = f"skipped (no {spec['source']} directory)"
            continue
        log(f"calibration: running {name}")
        if spec["kind"] == "docunet":
            if build is None:
                status[name] = "skipped (no SIFT-flow build)"
                continue
            report = docunet.evaluate_docunet(
                source,
                pattern=PREDICTION_PATTERN,
                conventions=spec["conventions"],
                build=build,
                ref_resize=spec["ref_resize"],
                references_dir=references_dir,
                reference_pattern=reference_pattern,
                workers=workers,
                progress=docunet_progress,
            )
        else:
            if engine is None:
                status[name] = "skipped (no Tesseract)"
                continue
            report = ocr.evaluate_ocr(
                source,
                pattern=PREDICTION_PATTERN,
                protocols=spec["protocols"],
                engine=engine,
                references_dir=references_dir,
                reference_pattern=reference_pattern,
                workers=workers,
                progress=ocr_progress,
            )
        report["evidence"] = CALIBRATION_EVIDENCE
        write_json_atomic(path, report, overwrite=True)
        status[name] = "written"
    return status


def summarize_calibration(
    reports_dir: str | os.PathLike[str], summary_path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Evaluate every gate from the reports and write the summary JSON."""
    root = Path(reports_dir).resolve()
    destination = Path(summary_path).resolve()
    gates = evaluate_gates(root)

    def listed(path: Path) -> dict[str, Any]:
        report = _load(path)
        own = report.get("evidence") if isinstance(report, dict) else None
        return {
            "path": os.path.relpath(path, destination.parent).replace(os.sep, "/"),
            "evidence": own or CALIBRATION_EVIDENCE,
            "evidence_in_file": own is not None,
        }

    reports = {}
    for name in (*RUNS, "paperedge-adres"):
        path = root / f"{name}.json"
        if path.is_file():
            reports[name] = listed(path)
    other_reports = {}
    for path in sorted(root.glob("*.json")):
        if path.stem not in reports and path.resolve() != destination:
            other_reports[path.stem] = listed(path)
    siftflow = None
    for name in ("paperedge-once", "paperedge-uvdoc", "uvdoc-uvdoc"):
        report = _load(root / f"{name}.json")
        if report is not None and report.get("siftflow"):
            siftflow = report["siftflow"]
            evaluation_environment = report.get("environment")
            break
    else:
        evaluation_environment = None
    summary = {
        "all_pass": all(gate["pass"] for gate in gates),
        "archives": ARCHIVES,
        "created_utc": utc_now(),
        "evaluation_environment": evaluation_environment,
        "evidence": CALIBRATION_EVIDENCE,
        "gates": gates,
        "other_reports": other_reports,
        "reports": reports,
        "schema": SCHEMA,
        "siftflow": siftflow,
        "summary_environment": environment(),
    }
    write_json_atomic(destination, summary, overwrite=True)
    return summary


def format_gates(gates: list[dict[str, Any]]) -> str:
    lines = []
    for gate in gates:
        lines.append(f"{gate['gate']}: {gate['status'].upper()} - {gate['description']}")
        for check in gate["checks"]:
            label = check["metric"]
            for key in ("ad_solver", "setting"):
                if key in check:
                    label += f"[{check[key]}]"
            flag = "ok" if check["pass"] else "FAIL"
            if not check["gating"]:
                flag += " (info)"
            if check["tolerance_kind"] == "requirement":
                lines.append(
                    f"  {label}: {check['measured']!r} must be {check['target']} -> {flag}"
                )
                continue
            measured = "n/a" if check["measured"] is None else f"{check['measured']:.6g}"
            lines.append(
                f"  {label}: measured {measured} target {check['target']:.6g} "
                f"tol {check['tolerance']:.3g} {check['tolerance_kind']} -> {flag}"
            )
        if gate["status"] == "not_run":
            lines.append(f"  not run: {gate['reason']}")
    return "\n".join(lines)
