"""Predict and score one dewarp checkpoint on DocUNet on this machine, at low CPU priority.

Usage (from any directory; put --stress after CHECKPOINT and RUN):
  .venv/bin/python tools/score_docunet_local.py CHECKPOINT RUN [--stress [VARIANT ...]]
  .venv/bin/python tools/score_docunet_local.py CHECKPOINT RUN \
      --quantize truncate --ref-resize uvdoc_view2_double    # the published protocol
  .venv/bin/python tools/score_docunet_local.py CHECKPOINT RUN \
      --quantize truncate --ref-resize uvdoc_view2_double --inset syntheticdoc_half_cell
  .venv/bin/python tools/score_docunet_local.py CHECKPOINT RUN \
      --quantize truncate --ref-resize uvdoc_view2_double --rerun 1 --deskew
  .venv/bin/python tools/score_docunet_local.py CHECKPOINT RUN --ocr \
      --tesseract TREE/tesseract.exe --tesseract-launcher wine   # UB-Mannheim 5.0.1 via Wine

Steps:
  1. ``flattener.dewarp.predict docunet`` rectifies the 130 crops with ``--force-render``
     (published methods are scored that way), ``--quantize Q``, ``--inset I`` and
     ``--rerun R [--deskew]`` into CACHE/pred/RUN (Q=round, no inset and no rerun, the
     defaults), CACHE/pred-Q/RUN, or CACHE/pred[-Q]-inset-L/RUN with an inset (L: the
     preset name or the values, see ``flattener.dewarp.predict.inset_label``), with
     ``-rerunR`` and then ``-deskew`` appended for crop passes. Images that
     predictions.json lists for the same weights, quantization, inset, rerun settings
     and inputs are kept.
  2. ``flattener.benchmark docunet`` scores them with all three conventions and
     ``--ref-resize R`` into outputs/dewarp-runs/RUN/docunet-R.json, or
     docunet-R-truncate.json for Q=truncate (default: docunet-once.json), with
     ``-inset-L``, ``-rerunR`` and ``-deskew`` before .json as set. An existing report
     is kept when its prediction hashes, ref-resize, quantization, inset and rerun
     settings match; a report of other images or settings stops the script, and
     --overwrite replaces it.
  3. With ``--stress`` (default --quantize, --ref-resize, no inset and no rerun only):
     ``flattener.benchmark docunet-stress`` predicts and scores the
     stress variants (default: full_frame, rot90, rot270, small_page) into
     CACHE/stress/RUN and copies stress.json and each variant's docunet.json and
     predictions.json to outputs/dewarp-runs/RUN/stress/.
  4. With ``--ocr``: ``flattener.benchmark ocr`` (DocTr protocol, Setting 1 and 2) into
     outputs/dewarp-runs/RUN/ocr-doctr.json, or ocr-doctr-Q.json for Q=truncate
     (``-inset-L``, ``-rerunR`` and ``-deskew`` before .json as set). An existing
     report is kept when its prediction hashes, Tesseract executable and eng.traineddata
     match. Its numbers are comparable only when the report says so (a Tesseract build
     that passed gate G4, see flattener.benchmark.ocr).

outputs/dewarp-runs/RUN/ also gets predictions.json, or predictions-truncate.json for
Q=truncate (the prediction record, no images; ``-inset-L``, ``-rerunR`` and ``-deskew``
before .json as set). Each quantization, ref-resize, inset and rerun setting has its own
file names, so one setting never overwrites another's images or reports.
Images stay under CACHE: $FLATTENER_CACHE_DIR, default ~/.cache/flattener.

Published DocUNet numbers (UVDoc, SyntheticDoc) come from outputs saved with truncation
and scored with the UVDoc script (``--quantize truncate --ref-resize uvdoc_view2_double``);
truncation instead of rounding alone moves LD by about 0.3.

CPU limits: the script and its children run at nice 19. Prediction uses 2 threads
(OMP_NUM_THREADS, MKL_NUM_THREADS). Scoring uses at most 2 worker processes with one
thread each. Temporary files and extension builds go under CACHE.

LD and AD need the SIFT-flow sources: --siftflow-source DIR or $FLATTENER_SIFTFLOW_SOURCE.
The script ends with one summary line on stdout. Exit status 1 when a step failed or a
stress variant was not scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flattener.benchmark import ocr, siftflow
from flattener.dewarp.predict import (
    INSET_PRESETS,
    inset_label,
    parse_inset,
    parse_rerun,
    recorded_inset,
    recorded_rerun,
    rerun_record,
    resolve_inset,
)

REPO = Path(__file__).resolve().parents[1]
CROP_DIR = REPO / "outputs/benchmarks/docunet/crop"
RUNS_DIR = REPO / "outputs/dewarp-runs"
CONVENTIONS = ("128_excl64", "130_raw", "130_rot64")
STRESS_VARIANTS = ("crop", "full_frame", "rot90", "rot270", "small_page")
QUANTIZE_MODES = ("round", "truncate")  # flattener.dewarp.predict --quantize
REF_RESIZE_MODES = ("once", "uvdoc_view2_double")  # flattener.benchmark docunet --ref-resize
DEFAULT_QUANTIZE, DEFAULT_REF_RESIZE = "round", "once"
DEFAULT_STRESS = ("full_frame", "rot90", "rot270", "small_page")
MAX_WORKERS = 2
MAX_THREADS = 2


def _bounded(limit: int):
    def parse(text: str) -> int:
        value = int(text)
        if not 1 <= value <= limit:
            raise argparse.ArgumentTypeError(f"must be between 1 and {limit}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="example: .venv/bin/python tools/score_docunet_local.py "
        "runs/r2/checkpoints/step_00003600.pt r2 --stress rot90 small_page",
    )
    parser.add_argument("checkpoint", type=Path, help="dewarp checkpoint or UVDoc best_model.pkl")
    parser.add_argument("run", help="run name: reports go to outputs/dewarp-runs/RUN/")
    parser.add_argument("--device", default="cpu", help="prediction device; default %(default)s")
    parser.add_argument(
        "--threads",
        type=_bounded(MAX_THREADS),
        default=MAX_THREADS,
        help="torch threads for prediction; default %(default)s",
    )
    parser.add_argument(
        "--workers",
        type=_bounded(MAX_WORKERS),
        default=MAX_WORKERS,
        help="scoring worker processes; default %(default)s",
    )
    parser.add_argument(
        "--quantize",
        choices=QUANTIZE_MODES,
        default=DEFAULT_QUANTIZE,
        help="uint8 conversion of the rectified images: round (flattener.render) or "
        "truncate (floor(x*255), as published outputs were saved); default %(default)s",
    )
    parser.add_argument(
        "--ref-resize",
        choices=REF_RESIZE_MODES,
        default=DEFAULT_REF_RESIZE,
        help="scan resize of the scorer: once (DocUNet/DocTr/PaperEdge) or "
        "uvdoc_view2_double (UVDoc script); default %(default)s",
    )
    parser.add_argument(
        "--inset",
        type=parse_inset,
        default=None,
        help="move the page outline inward before rendering (flattener.dewarp.predict "
        "--inset): " + ", ".join(INSET_PRESETS) + ", E or EX,EY; default 0",
    )
    parser.add_argument(
        "--rerun",
        type=parse_rerun,
        default=0,
        help="crop passes after the whole-image prediction (flattener.dewarp.predict "
        "--rerun); default 0",
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="with --rerun: turn each crop upright (flattener.dewarp.predict --deskew)",
    )
    parser.add_argument(
        "--stress",
        nargs="*",
        choices=STRESS_VARIANTS,
        default=None,
        metavar="VARIANT",
        help="also run stress variants; without names: " + " ".join(DEFAULT_STRESS),
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
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="also OCR the predictions with the DocTr protocol (ocr-doctr*.json)",
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
    parser.add_argument("--quiet", action="store_true", help="no per-image progress")
    return parser


def lower_priority() -> None:
    current = os.nice(0)
    if current < 19:
        os.nice(19 - current)


def child_env(cache: Path, threads: int, siftflow_source: Path | None) -> dict[str, str]:
    temp = cache / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        MAX_JOBS="2",
        TMPDIR=str(temp),
        FLATTENER_CACHE_DIR=str(cache),
    )
    source = siftflow_source if siftflow_source is not None else env["FLATTENER_SIFTFLOW_SOURCE"]
    env["FLATTENER_SIFTFLOW_SOURCE"] = str(siftflow.resolve_source_dir(source))
    return env


def run(command: list[str], env: dict[str, str], ok: tuple[int, ...] = (0,)) -> int:
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    code = subprocess.run(command, cwd=REPO, env=env, check=False).returncode
    if code not in ok:
        raise SystemExit(f"error: command exited with status {code}")
    return code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _suffix(quantize: str, inset=None, rerun: int = 0, deskew: bool = False) -> str:
    """-Q unless Q is round, then -inset-L for an inset (``inset_label``), -rerunR for
    crop passes and -deskew with deskew."""
    rerun_record(rerun, deskew)  # refuses deskew without passes
    suffix = "" if quantize == DEFAULT_QUANTIZE else f"-{quantize}"
    label = inset_label(inset)
    suffix += f"-inset-{label}" if label else ""
    suffix += f"-rerun{rerun}" if rerun else ""
    return suffix + ("-deskew" if deskew else "")


def prediction_dir(
    cache: Path, run_name: str, quantize: str, inset=None, rerun: int = 0, deskew: bool = False
) -> Path:
    """Prediction images of RUN: CACHE/pred/RUN for round without inset or rerun,
    otherwise CACHE/pred-Q/RUN, CACHE/pred-inset-L/RUN, CACHE/pred-Q-inset-L/RUN, and so on
    with -rerunR[-deskew] appended."""
    return cache / f"pred{_suffix(quantize, inset, rerun, deskew)}" / run_name


def report_name(
    ref_resize: str, quantize: str, inset=None, rerun: int = 0, deskew: bool = False
) -> str:
    """docunet-R.json, with -Q before .json unless Q is round, -inset-L for an inset and
    -rerunR[-deskew] for crop passes (default docunet-once.json)."""
    return f"docunet-{ref_resize}{_suffix(quantize, inset, rerun, deskew)}.json"


def record_name(quantize: str, inset=None, rerun: int = 0, deskew: bool = False) -> str:
    """predictions.json for round without inset or rerun,
    predictions-Q[-inset-L][-rerunR[-deskew]].json otherwise."""
    return f"predictions{_suffix(quantize, inset, rerun, deskew)}.json"


def ocr_report_name(quantize: str, inset=None, rerun: int = 0, deskew: bool = False) -> str:
    """ocr-doctr.json for round without inset or rerun,
    ocr-doctr-Q[-inset-L][-rerunR[-deskew]].json otherwise."""
    return f"ocr-doctr{_suffix(quantize, inset, rerun, deskew)}.json"


def record_inset(record: dict | None) -> tuple[float, float]:
    """(ex, ey) of a predictions record or a report's ``predictions_record``; one
    without the field predates the option and has no inset."""
    return recorded_inset((record or {}).get("inset"))


def record_rerun(record: dict | None) -> dict:
    """Rerun settings of a predictions record or a report's ``predictions_record``; one
    without the field predates the option and holds whole-image predictions."""
    return recorded_rerun((record or {}).get("rerun"))


def ocr_report_matches(
    report_path: Path,
    pred_dir: Path,
    engine: dict,
    quantize: str = DEFAULT_QUANTIZE,
    inset=None,
    rerun: int = 0,
    deskew: bool = False,
) -> bool:
    """Whether the OCR report read exactly the current images (DocTr protocol, images
    quantized by ``quantize`` with ``inset``, ``rerun`` and ``deskew``) with the Tesseract
    executable and eng.traineddata that ``engine`` (``Tesseract.metadata()``) describes."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("config", {}).get("protocols") != ["doctr"]:
        return False
    if (report.get("predictions_record") or {}).get("quantize", DEFAULT_QUANTIZE) != quantize:
        return False
    if record_inset(report.get("predictions_record")) != resolve_inset(inset):
        return False
    if record_rerun(report.get("predictions_record")) != rerun_record(rerun, deskew):
        return False
    used = report.get("tesseract") or {}

    def eng(metadata: dict) -> str | None:
        return ((metadata.get("traineddata") or {}).get("eng.traineddata") or {}).get("sha256")

    if used.get("executable_sha256") != engine.get("executable_sha256"):
        return False
    if eng(used) is None or eng(used) != eng(engine):
        return False
    scored = report.get("inputs", {}).get("predictions", {})
    if not scored:
        return False
    return all(
        (pred_dir / f"{key}.png").is_file() and sha256_file(pred_dir / f"{key}.png") == digest
        for key, digest in scored.items()
    )


def report_matches(
    report_path: Path,
    pred_dir: Path,
    source_sha256: str,
    ref_resize: str = DEFAULT_REF_RESIZE,
    quantize: str = DEFAULT_QUANTIZE,
    inset=None,
    rerun: int = 0,
    deskew: bool = False,
) -> bool:
    """Whether the report scored exactly the current images with these settings (all
    conventions, ``ref_resize``, images quantized by ``quantize`` with ``inset``,
    ``rerun`` and ``deskew``) and the same SIFT-flow sources."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report.get("config", {})
    if config.get("ref_resize") != ref_resize:
        return False
    if set(config.get("conventions", ())) != set(CONVENTIONS):
        return False
    # Reports made before predictions recorded the quantization scored rounded images.
    if (report.get("predictions_record") or {}).get("quantize", DEFAULT_QUANTIZE) != quantize:
        return False
    if record_inset(report.get("predictions_record")) != resolve_inset(inset):
        return False
    if record_rerun(report.get("predictions_record")) != rerun_record(rerun, deskew):
        return False
    if not report.get("comparable"):
        return False
    if (report.get("siftflow") or {}).get("source_sha256") != source_sha256:
        return False
    scored = report.get("inputs", {}).get("predictions", {})
    current = sorted(path.stem for path in pred_dir.glob("*_*.png"))
    if sorted(scored) != current:
        return False
    return all(sha256_file(pred_dir / f"{key}.png") == scored[key] for key in current)


def inset_argument(inset) -> list[str]:
    """``--inset`` for the child commands; none without an inset."""
    if resolve_inset(inset) == resolve_inset(None):
        return []
    if isinstance(inset, str):
        return ["--inset", inset]
    ex, ey = resolve_inset(inset)
    return ["--inset", f"{ex!r},{ey!r}"]


def rerun_argument(rerun: int = 0, deskew: bool = False) -> list[str]:
    """``--rerun R [--deskew]`` for the child commands; none without passes."""
    rerun_record(rerun, deskew)
    return (["--rerun", str(rerun)] if rerun else []) + (["--deskew"] if deskew else [])


def _settings(args) -> tuple:
    """(quantize, inset, rerun, deskew) of the command line, as the name helpers take them."""
    return args.quantize, args.inset, args.rerun, args.deskew


def predict(args, cache: Path, env: dict[str, str]) -> tuple[dict, float]:
    images = prediction_dir(cache, args.run, *_settings(args))
    command = [
        sys.executable, "-m", "flattener.dewarp.predict",
        "--weights", str(args.checkpoint), "--device", args.device,
        "docunet", "--input-dir", str(CROP_DIR), "--output-dir", str(images),
        "--force-render", "--quantize", args.quantize, *inset_argument(args.inset),
        *rerun_argument(args.rerun, args.deskew),
    ]  # fmt: skip
    if args.overwrite:
        command.append("--overwrite")
    started = time.perf_counter()
    run(command, env, ok=(0, 1))  # 1: some map failed its checks and was force-rendered
    wall = time.perf_counter() - started
    record = json.loads((images / "predictions.json").read_text(encoding="utf-8"))
    if not record.get("complete") or record["counts"]["items"] != 130:
        raise SystemExit(f"error: {images}/predictions.json is incomplete")
    if record.get("quantize", DEFAULT_QUANTIZE) != args.quantize:
        raise SystemExit(f"error: {images}/predictions.json holds other quantization")
    if record_inset(record) != resolve_inset(args.inset):
        raise SystemExit(f"error: {images}/predictions.json holds another inset")
    if record_rerun(record) != rerun_record(args.rerun, args.deskew):
        raise SystemExit(f"error: {images}/predictions.json holds other rerun settings")
    return record, wall


def score(args, cache: Path, env: dict[str, str], out_dir: Path) -> tuple[dict, str]:
    images = prediction_dir(cache, args.run, *_settings(args))
    report_path = out_dir / report_name(args.ref_resize, *_settings(args))
    if report_path.is_file() and not args.overwrite:
        source = siftflow.resolve_source_dir(env["FLATTENER_SIFTFLOW_SOURCE"])
        digest = siftflow.source_tree_sha256(source)
        if report_matches(report_path, images, digest, args.ref_resize, *_settings(args)):
            return json.loads(report_path.read_text(encoding="utf-8")), "kept"
        raise SystemExit(f"error: {report_path} scores other images or settings; pass --overwrite")
    command = [
        sys.executable, "-m", "flattener.benchmark", "docunet", str(images),
        "--ref-resize", args.ref_resize, "--workers", str(args.workers),
        "--output", str(report_path), "--overwrite",
    ]  # fmt: skip
    if args.quiet:
        command.append("--quiet")
    run(command, {**env, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return json.loads(report_path.read_text(encoding="utf-8")), "scored"


def stress(args, cache: Path, env: dict[str, str], out_dir: Path) -> dict:
    variants = args.stress or list(DEFAULT_STRESS)
    stress_dir = cache / "stress" / args.run
    command = [
        sys.executable, "-m", "flattener.benchmark", "docunet-stress",
        "--weights", str(args.checkpoint), "--output-dir", str(stress_dir),
        "--device", args.device, "--workers", str(args.workers), "--force-render",
    ]  # fmt: skip
    for variant in variants:
        command += ["--variant", variant]
    if args.overwrite:
        command.append("--overwrite")
    if args.quiet:
        command.append("--quiet")
    run(command, env, ok=(0, 1))  # 1: a check failed (still scored) or a variant was not
    summary = json.loads((stress_dir / "stress.json").read_text(encoding="utf-8"))
    target = out_dir / "stress"
    for variant in summary["variants"]:
        for name in ("docunet.json", "predictions.json"):
            source = stress_dir / variant / name
            if source.is_file():
                (target / variant).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target / variant / name)
    shutil.copyfile(stress_dir / "stress.json", target / "stress.json")
    return summary


def run_ocr(args, cache: Path, env: dict[str, str], out_dir: Path) -> tuple[dict, str]:
    images = prediction_dir(cache, args.run, *_settings(args))
    report_path = out_dir / ocr_report_name(*_settings(args))
    tessdata = None if args.tessdata_dir is None else str(args.tessdata_dir)
    launcher = args.tesseract_launcher or ""
    if report_path.is_file() and not args.overwrite:
        engine = ocr.Tesseract(
            args.tesseract, tessdata_dir=tessdata, launcher=tuple(shlex.split(launcher))
        ).metadata()
        if ocr_report_matches(report_path, images, engine, *_settings(args)):
            return json.loads(report_path.read_text(encoding="utf-8")), "kept"
        raise SystemExit(f"error: {report_path} read other images or Tesseract; pass --overwrite")
    command = [
        sys.executable, "-m", "flattener.benchmark", "ocr", str(images),
        "--tesseract", args.tesseract, "--temp-dir", str(cache / "tmp"),
        "--workers", str(args.workers), "--output", str(report_path), "--overwrite",
    ]  # fmt: skip
    if tessdata is not None:
        command += ["--tessdata-dir", tessdata]
    if launcher:
        command += ["--launcher", launcher]
    if args.quiet:
        command.append("--quiet")
    run(command, {**env, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return json.loads(report_path.read_text(encoding="utf-8")), "scored"


def ocr_text(report: dict) -> str:
    doctr = report["summary"]["doctr"]
    s1, s2 = doctr["setting1_doctr_60"], doctr["setting2_dewarpnet_50"]
    text = f"ocr S1 ed={s1['ed']:.2f} cer={s1['cer']:.4f} S2 ed={s2['ed']:.2f} cer={s2['cer']:.4f}"
    build = (report.get("comparability") or {}).get("tesseract_build")
    return text + (f" ({build})" if report.get("comparable") else " (not comparable)")


def metric_text(summary: dict) -> str:
    return f"ms_ssim={summary['ms_ssim']:.4f} ld={summary['ld']:.3f} ad={summary['ad']:.4f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run):
        raise SystemExit("error: RUN may use letters, digits, '.', '_' and '-'")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"error: no checkpoint at {args.checkpoint}")
    if args.siftflow_source is None and not os.environ.get("FLATTENER_SIFTFLOW_SOURCE"):
        raise SystemExit("error: pass --siftflow-source DIR or set FLATTENER_SIFTFLOW_SOURCE")
    default_protocol = (args.quantize, args.ref_resize) == (DEFAULT_QUANTIZE, DEFAULT_REF_RESIZE)
    label = inset_label(args.inset)
    if args.deskew and not args.rerun:
        raise SystemExit("error: --deskew needs --rerun 1 or more")
    if args.stress is not None and (not default_protocol or label or args.rerun):
        raise SystemExit(
            "error: --stress runs with the default --quantize and --ref-resize and no "
            "--inset or --rerun only"
        )
    lower_priority()
    cache_value = args.cache_dir or os.environ.get("FLATTENER_CACHE_DIR") or "~/.cache/flattener"
    cache = Path(cache_value).expanduser().resolve()
    try:
        env = child_env(cache, args.threads, args.siftflow_source)
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}") from None
    out_dir = RUNS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    record, predict_wall = predict(args, cache, env)
    written = [item for item in record["items"] if item["action"] == "written"]
    per_image = sum(item["seconds"] for item in written) / len(written) if written else None
    report, score_state = score(args, cache, env, out_dir)
    shutil.copyfile(
        prediction_dir(cache, args.run, *_settings(args)) / "predictions.json",
        out_dir / record_name(*_settings(args)),
    )

    counts = record["counts"]
    settings = "" if default_protocol else f" quantize={args.quantize} ref_resize={args.ref_resize}"
    settings += f" inset={label}" if label else ""
    settings += f" rerun={args.rerun}" if args.rerun else ""
    settings += " deskew" if args.deskew else ""
    parts = [
        f"{args.run} step={record['model'].get('step')}{settings}",
        f"128_excl64 {metric_text(report['summary']['128_excl64'])}",
        f"130_raw {metric_text(report['summary']['130_raw'])}",
        f"predict {args.device} "
        + (f"{per_image:.2f} s/img" if per_image is not None else "all kept")
        + f" ({counts['written']} written, {counts['kept']} kept, {counts['failed']} failed)",
        f"score {score_state}: {report['timing']['wall_seconds']:.0f} s wall "
        f"with {report['config']['workers']} workers",
    ]
    status = 0
    if args.stress is not None:
        summary = stress(args, cache, env, out_dir)
        for name, entry in summary["variants"].items():
            if entry["status"] != "scored":
                parts.append(f"stress {name} {entry['status']}")
                status = 1
                continue
            parts.append(f"stress {name} 128_excl64 {metric_text(entry['summary']['128_excl64'])}")
    if args.ocr:
        ocr_report, ocr_state = run_ocr(args, cache, env, out_dir)
        parts.append(
            f"{ocr_text(ocr_report)} {ocr_state}: {ocr_report['timing']['wall_seconds']:.0f} s wall"
        )
    print(f"total predict wall {predict_wall:.0f} s", file=sys.stderr)
    print(" | ".join(parts), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
