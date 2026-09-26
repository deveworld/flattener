"""Command line: ``python -m flattener.benchmark {docunet,ocr,calibrate,build-siftflow}``.

``docunet``, ``ocr`` and ``calibrate`` take ``--workers`` (default 2). Scoring LD/AD
needs the SIFT-flow C++ sources, which are not distributed: pass
``--siftflow-source DIR`` or set ``FLATTENER_SIFTFLOW_SOURCE``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import calibration, docunet, metrics, ocr, siftflow
from .reporting import write_json_atomic

_EPILOG = """\
examples:
  python -m flattener.benchmark docunet OUT/ --pattern "{k}_{m}.png" \\
      --siftflow-source ~/eval/SIFTflow --output reports/docunet.json
  python -m flattener.benchmark ocr OUT/ --pattern "{k}_{m}.png" \\
      --tesseract /usr/bin/tesseract --output reports/ocr.json
  python -m flattener.benchmark calibrate --reports-dir reports/calibration \\
      --paperedge-dir res_test --uvdoc-dir DocUnet_results --tesseract tesseract
"""


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _add_common(parser: argparse.ArgumentParser, *, predictions: bool = True) -> None:
    if predictions:
        parser.add_argument("predictions", type=Path, help="directory of rectified outputs")
        parser.add_argument(
            "--pattern",
            default="{k}_{m}.png",
            help="prediction file name with {k} (document 1-65) and {m} (view 1-2); "
            "default %(default)s",
        )
        parser.add_argument("--output", type=Path, required=True, help="report JSON path")
        parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument(
        "--references-dir",
        type=Path,
        default=docunet.DEFAULT_REFERENCES_DIR,
        help="directory of the 65 DocUNet scans; default %(default)s",
    )
    parser.add_argument(
        "--reference-pattern",
        default=docunet.DEFAULT_REFERENCE_PATTERN,
        help="scan file name with {k}; default %(default)s",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=docunet.DEFAULT_WORKERS,
        help="parallel worker processes/threads; default %(default)s (1 torch thread each)",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-image progress on stderr")


def _add_siftflow(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--siftflow-source",
        type=Path,
        default=None,
        help=f"SIFT-flow source folder with mexDenseSIFT/ and mexDiscreteFlow/ "
        f"(DocUNet eval.zip or UVDoc eval_code); default ${siftflow.SIFTFLOW_SOURCE_ENV}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"build cache; default ${siftflow.CACHE_DIR_ENV} or ~/.cache/flattener",
    )


def _add_tesseract(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tesseract", default="tesseract", help="tesseract executable; default %(default)s"
    )
    parser.add_argument("--temp-dir", default=None, help="directory for Tesseract temp files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flattener.benchmark",
        description="Standard DocUNet evaluation calibrated against the MATLAB/OCR protocols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser(
        "docunet",
        help="MS-SSIM / LD / AD with the MATLAB protocol",
        description="Score DocUNet predictions (MS-SSIM, LD, AD). Rotations, exclusions "
        "and the scan resize mode are recorded in the report.",
    )
    _add_common(score)
    _add_siftflow(score)
    score.add_argument(
        "--convention",
        action="append",
        choices=docunet.CONVENTIONS,
        help="aggregation convention (repeatable); default all three. 130_raw is what "
        "PaperEdge and the released DocTr-family scripts compute, 128_excl64 is UVDoc's; "
        "130_rot64 (document 64 rotated 180 degrees) is not validated by any gate",
    )
    score.add_argument(
        "--ref-resize",
        choices=docunet.REF_RESIZE_MODES,
        default="once",
        help="scan resize: once (DocUNet/DocTr/PaperEdge) or uvdoc_view2_double "
        "(UVDoc evalScript.m resizes the scan again for view 2); default %(default)s",
    )
    score.add_argument(
        "--ad-solver",
        choices=metrics.AD_SOLVERS,
        default=metrics.DEFAULT_AD_SOLVER,
        help="AD alignment solver reported as 'ad' (both are always stored); "
        "default %(default)s (MATLAB lsqr defaults; matches MATLAB means, per-image "
        "AD may differ by ~0.02)",
    )
    score.add_argument(
        "--ms-ssim-only",
        action="store_true",
        help="skip SIFT flow (no LD/AD); the report is marked not comparable",
    )
    score.add_argument(
        "--no-verify-references",
        action="store_true",
        help="accept scans whose bytes differ from the published DocUNet scans",
    )

    reading = commands.add_parser(
        "ocr",
        help="OCR edit distance / CER with Tesseract",
        description="OCR DocUNet predictions with the DocTr protocol (primary: Setting 1 "
        "60 images, Setting 2 50 images) and/or the UVDoc protocol (secondary).",
    )
    _add_common(reading)
    _add_tesseract(reading)
    reading.add_argument(
        "--protocol",
        action="append",
        choices=ocr.PROTOCOLS,
        help="OCR protocol (repeatable); default doctr",
    )
    reading.add_argument(
        "--crop-dir",
        type=Path,
        default=ocr.DEFAULT_CROP_DIR,
        help="DocUNet crop photos (UVDoc protocol resizes predictions to them); "
        "default %(default)s",
    )
    reading.add_argument(
        "--crop-pattern", default=ocr.DEFAULT_CROP_PATTERN, help="default %(default)s"
    )
    reading.add_argument(
        "--no-verify-references",
        action="store_true",
        help="accept scans whose bytes differ from the published DocUNet scans",
    )

    calibrate = commands.add_parser(
        "calibrate",
        help="run gates G1-G4 against published MATLAB/OCR numbers",
        description="Write the calibration reports that are missing in --reports-dir "
        "(PaperEdge outputs with adres.txt, UVDoc DocUnet_results, OCR) and a gate "
        "summary. Exit status 0 only if every gate passes.",
    )
    _add_common(calibrate, predictions=False)
    _add_siftflow(calibrate)
    _add_tesseract(calibrate)
    calibrate.add_argument("--reports-dir", type=Path, required=True)
    calibrate.add_argument(
        "--summary", type=Path, default=None, help="default REPORTS_DIR/calibration.json"
    )
    calibrate.add_argument(
        "--paperedge-dir",
        type=Path,
        default=None,
        help="PaperEdge res_test/ ({k}_{m}.png and adres.txt)",
    )
    calibrate.add_argument(
        "--uvdoc-dir", type=Path, default=None, help="UVDoc DocUnet_results/ ({k}_{m}.png)"
    )
    calibrate.add_argument(
        "--run",
        action="append",
        choices=tuple(calibration.RUNS),
        help="only these runs (repeatable); default all",
    )
    calibrate.add_argument("--no-ocr", action="store_true", help="skip the OCR run (G4)")
    calibrate.add_argument(
        "--rerun", action="store_true", help="recompute reports that already exist"
    )

    build = commands.add_parser(
        "build-siftflow",
        help="compile the SIFT-flow extension and print its build record",
        description="Compile (or reuse) the SIFT-flow torch extension from a user-supplied "
        "source tree. Needs a C++ compiler (MSVC via vcvars64.bat on Windows) and ninja.",
    )
    _add_siftflow(build)
    build.add_argument("--verbose", action="store_true")
    return parser


def _write_report(path: Path, report: dict, overwrite: bool) -> None:
    digest = write_json_atomic(path, report, overwrite=overwrite)
    print(f"report: {path} sha256={digest}", file=sys.stderr)


def _check_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"error: {path} exists; pass --overwrite to replace it")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-siftflow":
        build = siftflow.build_siftflow(
            args.siftflow_source, cache_dir=args.cache_dir, verbose=args.verbose
        )
        print(json.dumps(build.metadata(), indent=2, sort_keys=True))
        return 0
    if args.command == "docunet":
        _check_output(args.output, args.overwrite)
        build = None
        if not args.ms_ssim_only:
            build = siftflow.build_siftflow(args.siftflow_source, cache_dir=args.cache_dir)
        report = docunet.evaluate_docunet(
            args.predictions,
            pattern=args.pattern,
            conventions=args.convention or docunet.DEFAULT_CONVENTIONS,
            build=build,
            ref_resize=args.ref_resize,
            ad_solver=args.ad_solver,
            references_dir=args.references_dir,
            reference_pattern=args.reference_pattern,
            verify_references=not args.no_verify_references,
            workers=args.workers,
            progress=None if args.quiet else docunet.print_progress,
        )
        _write_report(args.output, report, args.overwrite)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "ocr":
        _check_output(args.output, args.overwrite)
        report = ocr.evaluate_ocr(
            args.predictions,
            pattern=args.pattern,
            protocols=args.protocol or ("doctr",),
            engine=ocr.Tesseract(args.tesseract, temp_dir=args.temp_dir),
            references_dir=args.references_dir,
            reference_pattern=args.reference_pattern,
            verify_references=not args.no_verify_references,
            crop_dir=args.crop_dir,
            crop_pattern=args.crop_pattern,
            workers=args.workers,
            progress=None if args.quiet else ocr.print_progress,
        )
        _write_report(args.output, report, args.overwrite)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    # calibrate
    runs = tuple(args.run or calibration.RUNS)
    needs_flow = any(
        calibration.RUNS[name]["kind"] == "docunet"
        and (args.rerun or not (args.reports_dir / f"{name}.json").is_file())
        for name in runs
    )
    has_inputs = args.paperedge_dir is not None or args.uvdoc_dir is not None
    build = None
    if has_inputs and needs_flow:
        build = siftflow.build_siftflow(args.siftflow_source, cache_dir=args.cache_dir)
    engine = None if args.no_ocr else ocr.Tesseract(args.tesseract, temp_dir=args.temp_dir)
    status = calibration.run_calibration(
        args.reports_dir,
        paperedge_dir=args.paperedge_dir,
        uvdoc_dir=args.uvdoc_dir,
        build=build,
        engine=engine,
        references_dir=args.references_dir,
        reference_pattern=args.reference_pattern,
        workers=args.workers,
        runs=runs,
        rerun=args.rerun,
        log=lambda message: print(message, file=sys.stderr, flush=True),
        docunet_progress=None if args.quiet else docunet.print_progress,
        ocr_progress=None if args.quiet else ocr.print_progress,
    )
    for name, state in status.items():
        print(f"{name}: {state}", file=sys.stderr)
    summary_path = args.summary or args.reports_dir / "calibration.json"
    summary = calibration.summarize_calibration(args.reports_dir, summary_path)
    print(calibration.format_gates(summary["gates"]))
    print(f"summary: {summary_path}", file=sys.stderr)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
