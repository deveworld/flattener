"""Predict and score one dewarp checkpoint on DIR300 on this machine, at low CPU priority.

Usage (from any directory):
  .venv/bin/python tools/score_dir300_local.py CHECKPOINT RUN [--quantize truncate]
  .venv/bin/python tools/score_dir300_local.py CHECKPOINT RUN --ocr \
      --tesseract ~/tess501-win/tesseract.exe --tesseract-launcher wine

Steps:
  1. ``flattener.benchmark dir300-predict`` rectifies the 300 photos of
     outputs/benchmarks/dir300/dist300 at each photo's own size with ``--force-render``
     and ``--quantize Q`` into CACHE/dir300/pred/RUN (Q=round, the default) or
     CACHE/dir300/pred-Q/RUN. Images that predictions.json lists for the same weights,
     quantization and inputs are kept.
  2. ``flattener.benchmark dir300`` scores them (MS-SSIM, LD, AD, plain means over the
     300 pairs) into outputs/dewarp-runs/RUN/dir300.json, or dir300-Q.json for
     Q=truncate. An existing report is kept when its prediction hashes, quantization and
     SIFT-flow sources match; a report of other images or settings stops the script, and
     --overwrite replaces it.
  3. With ``--ocr``: ``flattener.benchmark dir300-ocr`` (DocGeoNet protocol, 90 images)
     into dir300-ocr.json or dir300-ocr-Q.json, kept on the same terms when the
     Tesseract executable and eng.traineddata also match. Its numbers are comparable
     only when the report says so (see flattener.benchmark.ocr).

outputs/dewarp-runs/RUN/ also gets dir300-predictions.json or dir300-predictions-Q.json
(the prediction record, no images). Images (about 9 MB each) stay under CACHE:
$FLATTENER_CACHE_DIR, default ~/.cache/flattener.

UVDoc ``demo.py`` and DocGeoNet ``inference.py`` save their outputs with truncation
(``--quantize truncate``). The photos include background, so a model that rectifies
the whole photo in one pass scores lower than methods that locate or crop the page
first.

CPU limits as in tools/score_docunet_local.py: nice 19, 2 prediction threads, at most
2 scoring workers with one thread each, temporary files under CACHE. Wrap the call in
``flock ~/.cache/flattener/score.lock`` when other heavy jobs share the machine.

LD and AD need the SIFT-flow sources: --siftflow-source DIR or $FLATTENER_SIFTFLOW_SOURCE.
The script ends with one summary line on stdout. Exit status 1 when a step failed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_docunet_local as local  # noqa: E402

from flattener.benchmark import dir300, ocr, siftflow  # noqa: E402

INPUTS_DIR = local.REPO / dir300.DEFAULT_INPUTS_DIR
ITEMS = len(dir300.DOCUMENTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="example: .venv/bin/python tools/score_dir300_local.py "
        "runs/r3/checkpoints/step_00046880.pt R3 --quantize truncate",
    )
    parser.add_argument("checkpoint", type=Path, help="dewarp checkpoint or UVDoc best_model.pkl")
    parser.add_argument("run", help="run name: reports go to outputs/dewarp-runs/RUN/")
    parser.add_argument("--device", default="cpu", help="prediction device; default %(default)s")
    parser.add_argument(
        "--threads",
        type=local._bounded(local.MAX_THREADS),
        default=local.MAX_THREADS,
        help="torch threads for prediction; default %(default)s",
    )
    parser.add_argument(
        "--workers",
        type=local._bounded(local.MAX_WORKERS),
        default=local.MAX_WORKERS,
        help="scoring worker processes; default %(default)s",
    )
    parser.add_argument(
        "--quantize",
        choices=local.QUANTIZE_MODES,
        default=local.DEFAULT_QUANTIZE,
        help="uint8 conversion of the rectified images: round (flattener.render) or "
        "truncate (floor(x*255), as published outputs were saved); default %(default)s",
    )
    parser.add_argument(
        "--ocr", action="store_true", help="also OCR the 90-image subset (dir300-ocr*.json)"
    )
    parser.add_argument(
        "--tesseract",
        default=os.environ.get("FLATTENER_TESSERACT", "tesseract"),
        help="tesseract executable for --ocr; default $FLATTENER_TESSERACT or tesseract",
    )
    parser.add_argument(
        "--tessdata-dir",
        type=Path,
        default=None,
        help="tessdata folder for --ocr (TESSDATA_PREFIX); default: inherited",
    )
    parser.add_argument(
        "--tesseract-launcher",
        default=os.environ.get("FLATTENER_TESSERACT_LAUNCHER"),
        help='command prefix for --tesseract, for example "wine"; '
        "default $FLATTENER_TESSERACT_LAUNCHER",
    )
    parser.add_argument(
        "--siftflow-source",
        type=Path,
        default=None,
        help="SIFT-flow folder with mexDenseSIFT/ and mexDiscreteFlow/; "
        "default $FLATTENER_SIFTFLOW_SOURCE",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="images, builds and temp files; default $FLATTENER_CACHE_DIR or ~/.cache/flattener",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="predict and score again even when outputs for this run exist",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-image progress")
    return parser


def suffix(quantize: str) -> str:
    return "" if quantize == local.DEFAULT_QUANTIZE else f"-{quantize}"


def prediction_dir(cache: Path, run_name: str, quantize: str) -> Path:
    """CACHE/dir300/pred/RUN for round, CACHE/dir300/pred-Q/RUN otherwise."""
    return cache / "dir300" / f"pred{suffix(quantize)}" / run_name


def report_name(quantize: str) -> str:
    return f"dir300{suffix(quantize)}.json"


def ocr_report_name(quantize: str) -> str:
    return f"dir300-ocr{suffix(quantize)}.json"


def record_name(quantize: str) -> str:
    return f"dir300-predictions{suffix(quantize)}.json"


def _scored_current_images(
    report: dict, pred_dir: Path, quantize: str, documents: tuple[int, ...]
) -> bool:
    """Whether a report read all ``documents`` and exactly the images now in
    ``pred_dir``, quantized by ``quantize``."""
    if report.get("config", {}).get("documents") != "all":
        return False
    if (report.get("predictions_record") or {}).get("quantize") != quantize:
        return False
    scored = report.get("inputs", {}).get("predictions", {})
    if sorted(scored) != sorted(str(k) for k in documents):
        return False
    return all(
        (pred_dir / f"{key}.png").is_file() and local.sha256_file(pred_dir / f"{key}.png") == digest
        for key, digest in scored.items()
    )


def report_matches(report_path: Path, pred_dir: Path, source_sha256: str, quantize: str) -> bool:
    """Whether the metric report scored exactly the current images with these SIFT-flow
    sources and verified references."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("benchmark") != "dir300" or not report.get("comparable"):
        return False
    if (report.get("siftflow") or {}).get("source_sha256") != source_sha256:
        return False
    return _scored_current_images(report, pred_dir, quantize, dir300.DOCUMENTS)


def ocr_report_matches(report_path: Path, pred_dir: Path, engine: dict, quantize: str) -> bool:
    """Whether the OCR report read exactly the current images with the Tesseract
    executable and eng.traineddata that ``engine`` (``Tesseract.metadata()``) describes."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("benchmark") != "dir300-ocr":
        return False
    if not (report.get("config") or {}).get("references_verified"):
        return False
    used = report.get("tesseract") or {}

    def eng(metadata: dict) -> str | None:
        return ((metadata.get("traineddata") or {}).get("eng.traineddata") or {}).get("sha256")

    if used.get("executable_sha256") != engine.get("executable_sha256"):
        return False
    if eng(used) is None or eng(used) != eng(engine):
        return False
    return _scored_current_images(report, pred_dir, quantize, dir300.OCR_DOCUMENTS)


def predict(args, cache: Path, env: dict[str, str]) -> tuple[dict, float]:
    images = prediction_dir(cache, args.run, args.quantize)
    command = [
        sys.executable, "-m", "flattener.benchmark", "dir300-predict",
        "--weights", str(args.checkpoint), "--device", args.device,
        "--threads", str(args.threads), "--inputs-dir", str(INPUTS_DIR),
        "--output-dir", str(images), "--force-render", "--quantize", args.quantize,
    ]  # fmt: skip
    if args.overwrite:
        command.append("--overwrite")
    started = time.perf_counter()
    local.run(command, env, ok=(0, 1))  # 1: some map failed its checks and was force-rendered
    wall = time.perf_counter() - started
    record = json.loads((images / "predictions.json").read_text(encoding="utf-8"))
    if not record.get("complete") or record["counts"]["items"] != ITEMS:
        raise SystemExit(f"error: {images}/predictions.json is incomplete")
    if record.get("quantize") != args.quantize:
        raise SystemExit(f"error: {images}/predictions.json holds other quantization")
    missing = [item for item in record["items"] if item["action"] == "not_written"]
    if missing:
        reasons = sorted({str(item.get("reason")) for item in missing})
        raise SystemExit(
            f"error: {len(missing)} photos were not rectified ({', '.join(reasons)}), "
            f"first: {missing[0]['output']}; see {images}/predictions.json"
        )
    return record, wall


def score(args, cache: Path, env: dict[str, str], out_dir: Path) -> tuple[dict, str]:
    images = prediction_dir(cache, args.run, args.quantize)
    report_path = out_dir / report_name(args.quantize)
    if report_path.is_file() and not args.overwrite:
        source = siftflow.resolve_source_dir(env["FLATTENER_SIFTFLOW_SOURCE"])
        digest = siftflow.source_tree_sha256(source)
        if report_matches(report_path, images, digest, args.quantize):
            return json.loads(report_path.read_text(encoding="utf-8")), "kept"
        raise SystemExit(f"error: {report_path} scores other images or settings; pass --overwrite")
    command = [
        sys.executable, "-m", "flattener.benchmark", "dir300", str(images),
        "--workers", str(args.workers), "--output", str(report_path), "--overwrite",
    ]  # fmt: skip
    if args.quiet:
        command.append("--quiet")
    local.run(command, {**env, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return json.loads(report_path.read_text(encoding="utf-8")), "scored"


def run_ocr(args, cache: Path, env: dict[str, str], out_dir: Path) -> tuple[dict, str]:
    images = prediction_dir(cache, args.run, args.quantize)
    report_path = out_dir / ocr_report_name(args.quantize)
    tessdata = None if args.tessdata_dir is None else str(args.tessdata_dir)
    if report_path.is_file() and not args.overwrite:
        launcher = tuple(shlex.split(args.tesseract_launcher or ""))
        engine = ocr.Tesseract(args.tesseract, tessdata_dir=tessdata, launcher=launcher).metadata()
        if ocr_report_matches(report_path, images, engine, args.quantize):
            return json.loads(report_path.read_text(encoding="utf-8")), "kept"
        raise SystemExit(f"error: {report_path} read other images or Tesseract; pass --overwrite")
    command = [
        sys.executable, "-m", "flattener.benchmark", "dir300-ocr", str(images),
        "--tesseract", args.tesseract, "--temp-dir", str(cache / "tmp"),
        "--workers", str(args.workers), "--output", str(report_path), "--overwrite",
    ]  # fmt: skip
    if tessdata is not None:
        command += ["--tessdata-dir", tessdata]
    if args.tesseract_launcher:
        command += ["--launcher", args.tesseract_launcher]
    if args.quiet:
        command.append("--quiet")
    local.run(command, {**env, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return json.loads(report_path.read_text(encoding="utf-8")), "scored"


def ocr_text(report: dict) -> str:
    summary = report["summary"]
    text = f"ocr 90 ed={summary['ed']:.2f} cer={summary['cer']:.4f}"
    build = (report.get("comparability") or {}).get("tesseract_build")
    return text + (f" ({build})" if report.get("comparable") else " (not comparable)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run):
        raise SystemExit("error: RUN may use letters, digits, '.', '_' and '-'")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"error: no checkpoint at {args.checkpoint}")
    if args.siftflow_source is None and not os.environ.get("FLATTENER_SIFTFLOW_SOURCE"):
        raise SystemExit("error: pass --siftflow-source DIR or set FLATTENER_SIFTFLOW_SOURCE")
    local.lower_priority()
    cache_value = args.cache_dir or os.environ.get("FLATTENER_CACHE_DIR") or "~/.cache/flattener"
    cache = Path(cache_value).expanduser().resolve()
    try:
        env = local.child_env(cache, args.threads, args.siftflow_source)
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}") from None
    out_dir = local.RUNS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    record, predict_wall = predict(args, cache, env)
    written = [item for item in record["items"] if item["action"] == "written"]
    per_image = sum(item["seconds"] for item in written) / len(written) if written else None
    report, score_state = score(args, cache, env, out_dir)
    shutil.copyfile(
        prediction_dir(cache, args.run, args.quantize) / "predictions.json",
        out_dir / record_name(args.quantize),
    )

    counts = record["counts"]
    parts = [
        f"{args.run} step={record['model'].get('step')} quantize={args.quantize}",
        f"dir300 {local.metric_text(report['summary'])}",
        f"predict {args.device} "
        + (f"{per_image:.2f} s/img" if per_image is not None else "all kept")
        + f" ({counts['written']} written, {counts['kept']} kept, {counts['failed']} failed)",
        f"score {score_state}: {report['timing']['wall_seconds']:.0f} s wall "
        f"with {report['config']['workers']} workers",
    ]
    if args.ocr:
        ocr_report, ocr_state = run_ocr(args, cache, env, out_dir)
        parts.append(
            f"{ocr_text(ocr_report)} {ocr_state}: {ocr_report['timing']['wall_seconds']:.0f} s wall"
        )
    print(f"total predict wall {predict_wall:.0f} s", file=sys.stderr)
    print(" | ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
