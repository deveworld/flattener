"""Command line: ``python -m flattener.benchmark
{docunet,ocr,calibrate,docunet-stress,uvdoc-prepare,uvdoc-predict,uvdoc,build-siftflow}``.

``docunet``, ``ocr``, ``calibrate``, ``docunet-stress`` and ``uvdoc`` take ``--workers``
(default 2).
Scoring LD/AD needs your copy of the SIFT-flow C++ sources: pass ``--siftflow-source
DIR`` or set ``FLATTENER_SIFTFLOW_SOURCE``. ``dir300-predict``, ``dir300`` and
``dir300-ocr`` run the DIR300 benchmark.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import calibration, dir300, docunet, metrics, ocr, siftflow, stress, uvdoc
from .reporting import write_json_atomic

_EPILOG = """\
examples:
  python -m flattener.benchmark docunet OUT/ --pattern "{k}_{m}.png" \\
      --siftflow-source ~/eval/SIFTflow --output reports/docunet.json
  python -m flattener.benchmark ocr OUT/ --pattern "{k}_{m}.png" \\
      --tesseract /usr/bin/tesseract --output reports/ocr.json
  python -m flattener.benchmark calibrate --reports-dir reports/calibration \\
      --paperedge-dir res_test --uvdoc-dir DocUnet_results --tesseract tesseract
  python -m flattener.benchmark docunet-stress --weights best_model.pkl \\
      --output-dir stress/uvdoc --force-render --device cuda
  python -m flattener.benchmark uvdoc-prepare UVDoc_benchmark.zip
  python -m flattener.benchmark uvdoc-predict --weights best_model.pkl --output-dir OUT
  python -m flattener.benchmark uvdoc OUT --output reports/uvdoc.json
  python -m flattener.benchmark dir300-predict --weights best_model.pkl \\
      --output-dir dir300/uvdoc --force-render --quantize truncate
  python -m flattener.benchmark dir300 dir300/uvdoc --output reports/dir300.json
  python -m flattener.benchmark dir300-ocr dir300/uvdoc --output reports/dir300-ocr.json
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
    parser.add_argument(
        "--tessdata-dir",
        default=None,
        help="tessdata folder, passed as TESSDATA_PREFIX (conda-forge builds need it); "
        "default: the inherited TESSDATA_PREFIX or the build's own path",
    )
    parser.add_argument(
        "--launcher",
        default=None,
        help='command prefix for the executable, for example "wine" to run tesseract.exe; '
        "the report still hashes the executable itself",
    )


def _tesseract(args: argparse.Namespace) -> ocr.Tesseract:
    return ocr.Tesseract(
        args.tesseract,
        temp_dir=args.temp_dir,
        tessdata_dir=args.tessdata_dir,
        launcher=tuple(shlex.split(args.launcher or "")),
    )


def _add_uvdoc_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=uvdoc.DEFAULT_ROOT,
        help="extracted UVDoc benchmark; default %(default)s",
    )


def _add_dir300_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("predictions", type=Path, help="directory of rectified outputs")
    parser.add_argument(
        "--pattern",
        default=dir300.DEFAULT_PATTERN,
        help="prediction file name with {k} (document 1-300); DocGeoNet's outputs are "
        "{k}_rec.png; default %(default)s",
    )
    parser.add_argument("--output", type=Path, required=True, help="report JSON path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument(
        "--references-dir",
        type=Path,
        default=dir300.DEFAULT_REFERENCES_DIR,
        help="the 300 DIR300 references gt/{k}.png; default %(default)s",
    )
    parser.add_argument(
        "--no-verify-references",
        action="store_true",
        help="accept references whose bytes differ from the official DIR300 files",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=docunet.DEFAULT_WORKERS,
        help="parallel worker processes/threads; default %(default)s (1 torch thread each)",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-image progress on stderr")


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

    rectify300 = commands.add_parser(
        "dir300-predict",
        help="rectify the 300 DIR300 photos with a dewarp model",
        description="Rectify dist/{k}.png into OUT/{k}.png at each photo's own size and "
        "record every check in OUT/predictions.json (see flattener.dewarp.predict). Exit "
        "status 1 when an image failed its checks.",
    )
    rectify300.add_argument("--weights", required=True, help="dewarp checkpoint or best_model.pkl")
    rectify300.add_argument("--output-dir", type=Path, required=True)
    rectify300.add_argument(
        "--inputs-dir",
        type=Path,
        default=dir300.DEFAULT_INPUTS_DIR,
        help="the 300 DIR300 photos dist/{k}.png; default %(default)s",
    )
    rectify300.add_argument("--device", default="cpu")
    rectify300.add_argument(
        "--threads", type=_positive_int, default=2, help="torch threads; default %(default)s"
    )
    rectify300.add_argument(
        "--force-render",
        action="store_true",
        help="write images whose map fails the geometry checks (the failures are still "
        "recorded); published methods are scored that way",
    )
    rectify300.add_argument(
        "--quantize",
        choices=("round", "truncate"),  # flattener.dewarp.predict.QUANTIZE_MODES
        default="round",
        help="uint8 conversion: round (flattener.render) or truncate (floor(x*255), as "
        "published outputs were saved); default %(default)s",
    )
    rectify300.add_argument("--overwrite", action="store_true", help="predict every image again")
    rectify300.add_argument(
        "--no-verify-inputs",
        action="store_true",
        help="accept photos whose bytes differ from the official DIR300 files",
    )

    score300 = commands.add_parser(
        "dir300",
        help="MS-SSIM / LD / AD on DIR300",
        description="Score PREDICTIONS/{k}.png against gt/{k}.png (k = 1-300) as DocGeoNet "
        "ssim_ld_eval_DIR300.m does, with AD from PaperEdge evalAlignedUnwarp.m; plain "
        "means over the 300 pairs.",
    )
    _add_dir300_common(score300)
    _add_siftflow(score300)
    score300.add_argument(
        "--ad-solver",
        choices=metrics.AD_SOLVERS,
        default=metrics.DEFAULT_AD_SOLVER,
        help="AD alignment solver reported as 'ad' (both are always stored); default %(default)s",
    )
    score300.add_argument(
        "--ms-ssim-only",
        action="store_true",
        help="skip SIFT flow (no LD/AD); the report is marked not comparable",
    )

    reading300 = commands.add_parser(
        "dir300-ocr",
        help="OCR edit distance / CER on the DIR300 OCR subset",
        description="OCR the 90 DIR300 images of DocGeoNet's ocr_img_DIR300.txt with the "
        "protocol of its OCR_eval_DIR300.py (Tesseract defaults, raw strings, plain means).",
    )
    _add_dir300_common(reading300)
    _add_tesseract(reading300)

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

    strain = commands.add_parser(
        "docunet-stress",
        help="predict and score DocUNet stress variants with a dewarp model",
        description="Build the stress variants (whole photos, crops turned by a quarter, "
        "small pages on a canvas), rectify them with a flattener.dewarp checkpoint or the "
        "UVDoc weights, score them against the upright scans and write OUT/stress.json. "
        "Exit status 1 when a prediction failed its checks or a variant could not be scored.",
    )
    _add_common(strain, predictions=False)
    _add_siftflow(strain)
    strain.add_argument("--weights", required=True, help="dewarp checkpoint or best_model.pkl")
    strain.add_argument("--output-dir", type=Path, required=True)
    strain.add_argument(
        "--variant",
        action="append",
        choices=tuple(stress.VARIANTS),
        help="variant (repeatable); default all: " + ", ".join(stress.VARIANTS),
    )
    strain.add_argument("--device", default="cpu")
    strain.add_argument(
        "--force-render",
        action="store_true",
        help="write and score images whose map fails the geometry checks (the failures "
        "are still counted)",
    )
    strain.add_argument("--overwrite", action="store_true", help="predict every image again")
    strain.add_argument("--crop-dir", type=Path, default=stress.DEFAULT_CROP_DIR)
    strain.add_argument(
        "--photo-dir",
        type=Path,
        default=stress.DEFAULT_PHOTO_DIR,
        help="directory of docunet-photo-{k}_{m}.jpg; default %(default)s",
    )
    strain.add_argument(
        "--convention", action="append", choices=docunet.CONVENTIONS, help="default all three"
    )
    strain.add_argument("--ms-ssim-only", action="store_true", help="skip SIFT flow (no LD/AD)")
    strain.add_argument("--seed", type=int, default=0, help="small-page canvas seed")
    strain.add_argument(
        "--no-verify-references",
        action="store_true",
        help="accept scans whose bytes differ from the published DocUNet scans",
    )

    prepare = commands.add_parser(
        "uvdoc-prepare",
        help="extract UVDoc_benchmark.zip and write its manifest",
        description="Check the archive hash, extract it into --root and write "
        "ROOT/manifest.json (source URL, archive and file hashes). Download: " + uvdoc.ARCHIVE_URL,
    )
    prepare.add_argument("archive", type=Path, help="UVDoc_benchmark.zip")
    _add_uvdoc_root(prepare)
    prepare.add_argument("--url", default=uvdoc.ARCHIVE_URL, help="recorded source URL")
    prepare.add_argument("--retrieved", default=None, help="download date for the manifest")
    prepare.add_argument(
        "--no-verify", action="store_true", help="accept other archive and image bytes"
    )

    rectify = commands.add_parser(
        "uvdoc-predict",
        help="rectify the 50 UVDoc benchmark samples with a dewarp model",
        description="Predict a grid from img/{id}.png and apply it to warped_textures/"
        "{id}.png (uwp_texture/) and to the photo (uwp_img/), truncated to uint8 as "
        "uvdocBenchmark_pred.py does. Every image is written; render checks go to "
        "OUT/predictions.json.",
    )
    rectify.add_argument("--weights", required=True, help="dewarp checkpoint or best_model.pkl")
    rectify.add_argument("--output-dir", type=Path, required=True)
    _add_uvdoc_root(rectify)
    rectify.add_argument("--device", default="cpu")
    rectify.add_argument(
        "--threads", type=_positive_int, default=2, help="torch threads; default %(default)s"
    )
    rectify.add_argument("--overwrite", action="store_true", help="predict every image again")
    rectify.add_argument("--no-verify-inputs", action="store_true", help="accept other image bytes")
    rectify.add_argument("--quiet", action="store_true", help="no per-image progress")

    bench = commands.add_parser(
        "uvdoc",
        help="MS-SSIM / AD / LD on the UVDoc benchmark",
        description="Score PREDICTIONS/{pred-type}/{id}.png (00000-00049) against "
        "texture_sample/{id}.png as evalScriptUVDoc.m does; means over all 50 samples.",
    )
    bench.add_argument("predictions", type=Path, help="folder with uwp_texture/ (and uwp_img/)")
    bench.add_argument("--output", type=Path, required=True, help="report JSON path")
    bench.add_argument("--overwrite", action="store_true", help="replace an existing report")
    bench.add_argument(
        "--pred-type",
        choices=uvdoc.PRED_TYPES,
        default=uvdoc.DEFAULT_PRED_TYPE,
        help="unwarped textures (the published protocol) or unwarped photos; default %(default)s",
    )
    _add_uvdoc_root(bench)
    _add_siftflow(bench)
    bench.add_argument("--ad-solver", choices=metrics.AD_SOLVERS, default=metrics.DEFAULT_AD_SOLVER)
    bench.add_argument(
        "--workers",
        type=_positive_int,
        default=docunet.DEFAULT_WORKERS,
        help="parallel worker processes; default %(default)s (1 torch thread each)",
    )
    bench.add_argument("--ms-ssim-only", action="store_true", help="skip SIFT flow (no LD/AD)")
    bench.add_argument(
        "--no-verify-references",
        action="store_true",
        help="accept textures whose bytes differ from the published benchmark",
    )
    bench.add_argument("--quiet", action="store_true", help="no per-image progress on stderr")

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
    if args.command == "uvdoc-prepare":
        manifest = uvdoc.prepare_benchmark(
            args.archive,
            args.root,
            url=args.url,
            verify=not args.no_verify,
            retrieved=args.retrieved,
        )
        print(json.dumps(manifest["sets"], indent=2, sort_keys=True))
        return 0
    if args.command == "uvdoc-predict":
        import torch

        from ..dewarp.predict import load_dewarper

        torch.set_num_threads(args.threads)
        record = uvdoc.predict_uvdoc(
            load_dewarper(args.weights, device=args.device),
            args.output_dir,
            root=args.root,
            verify_inputs=not args.no_verify_inputs,
            overwrite=args.overwrite,
            log=None if args.quiet else lambda text: print(text, file=sys.stderr, flush=True),
        )
        print(json.dumps(record["counts"]))
        return 0
    if args.command == "uvdoc":
        _check_output(args.output, args.overwrite)
        build = None
        if not args.ms_ssim_only:
            build = siftflow.build_siftflow(args.siftflow_source, cache_dir=args.cache_dir)
        report = uvdoc.evaluate_uvdoc(
            args.predictions,
            build=build,
            pred_type=args.pred_type,
            root=args.root,
            ad_solver=args.ad_solver,
            verify_references=not args.no_verify_references,
            workers=args.workers,
            progress=None if args.quiet else uvdoc.print_progress,
        )
        _write_report(args.output, report, args.overwrite)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(uvdoc.format_comparison(report["summary"]))
        return 0
    if args.command == "docunet-stress":
        build = None
        if not args.ms_ssim_only:
            build = siftflow.build_siftflow(args.siftflow_source, cache_dir=args.cache_dir)
        summary = stress.run_stress(
            args.weights,
            args.output_dir,
            variants=args.variant or tuple(stress.VARIANTS),
            device=args.device,
            force_render=args.force_render,
            overwrite=args.overwrite,
            crop_dir=args.crop_dir,
            photo_dir=args.photo_dir,
            conventions=args.convention or docunet.DEFAULT_CONVENTIONS,
            build=build,
            references_dir=args.references_dir,
            reference_pattern=args.reference_pattern,
            verify_references=not args.no_verify_references,
            workers=args.workers,
            seed=args.seed,
            progress=None if args.quiet else docunet.print_progress,
            log=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(stress.format_table(summary))
        entries = summary["variants"].values()
        failed = any(e["status"] != "scored" or e["predictions"]["failed"] for e in entries)
        return 1 if failed else 0
    if args.command == "ocr":
        _check_output(args.output, args.overwrite)
        report = ocr.evaluate_ocr(
            args.predictions,
            pattern=args.pattern,
            protocols=args.protocol or ("doctr",),
            engine=_tesseract(args),
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
    if args.command == "dir300-predict":
        import torch

        from ..dewarp.predict import load_dewarper

        torch.set_num_threads(args.threads)
        record = dir300.predict_dir300(
            load_dewarper(args.weights, device=args.device),
            args.output_dir,
            inputs_dir=args.inputs_dir,
            verify_inputs=not args.no_verify_inputs,
            overwrite=args.overwrite,
            force_render=args.force_render,
            quantize=args.quantize,
        )
        print(json.dumps(record["counts"]))
        return 0 if record["counts"]["failed"] == 0 else 1
    if args.command == "dir300":
        _check_output(args.output, args.overwrite)
        build = None
        if not args.ms_ssim_only:
            build = siftflow.build_siftflow(args.siftflow_source, cache_dir=args.cache_dir)
        report = dir300.evaluate_dir300(
            args.predictions,
            build=build,
            pattern=args.pattern,
            ad_solver=args.ad_solver,
            references_dir=args.references_dir,
            verify_references=not args.no_verify_references,
            workers=args.workers,
            progress=None if args.quiet else dir300.print_progress,
        )
        _write_report(args.output, report, args.overwrite)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(dir300.format_comparison(report["summary"]))
        return 0
    if args.command == "dir300-ocr":
        _check_output(args.output, args.overwrite)
        report = dir300.evaluate_dir300_ocr(
            args.predictions,
            engine=_tesseract(args),
            pattern=args.pattern,
            references_dir=args.references_dir,
            verify_references=not args.no_verify_references,
            workers=args.workers,
            progress=None if args.quiet else ocr.print_progress,
        )
        _write_report(args.output, report, args.overwrite)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(dir300.format_comparison(report["summary"]))
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
    engine = None if args.no_ocr else _tesseract(args)
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
