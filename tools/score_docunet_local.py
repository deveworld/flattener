"""Predict and score one dewarp checkpoint on DocUNet on this machine, at low CPU priority.

Usage (from any directory; put --stress after CHECKPOINT and RUN):
  .venv/bin/python tools/score_docunet_local.py CHECKPOINT RUN [--stress [VARIANT ...]]

Steps:
  1. ``flattener.dewarp.predict docunet`` rectifies the 130 crops with ``--force-render``
     (published methods are scored that way) into CACHE/pred/RUN. Images that
     predictions.json lists for the same weights and inputs are kept.
  2. ``flattener.benchmark docunet`` scores them with all three conventions and
     ``--ref-resize once`` into outputs/dewarp-runs/RUN/docunet-once.json. An existing
     report is kept when its prediction hashes match the current images; a report of
     other images stops the script, and --overwrite replaces it.
  3. With ``--stress``: ``flattener.benchmark docunet-stress`` predicts and scores the
     stress variants (default: full_frame, rot90, rot270, small_page) into
     CACHE/stress/RUN and copies stress.json and each variant's docunet.json and
     predictions.json to outputs/dewarp-runs/RUN/stress/.

outputs/dewarp-runs/RUN/ also gets predictions.json (the prediction record, no images).
Images stay under CACHE: $FLATTENER_CACHE_DIR, default ~/.cache/flattener.

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
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flattener.benchmark import siftflow

REPO = Path(__file__).resolve().parents[1]
CROP_DIR = REPO / "outputs/benchmarks/docunet/crop"
RUNS_DIR = REPO / "outputs/dewarp-runs"
CONVENTIONS = ("128_excl64", "130_raw", "130_rot64")
STRESS_VARIANTS = ("crop", "full_frame", "rot90", "rot270", "small_page")
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


def report_matches(report_path: Path, pred_dir: Path, source_sha256: str) -> bool:
    """Whether the report scored exactly the current images with the standard settings
    and the same SIFT-flow sources."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report.get("config", {})
    if config.get("ref_resize") != "once" or set(config.get("conventions", ())) != set(CONVENTIONS):
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


def predict(args, cache: Path, env: dict[str, str]) -> tuple[dict, float]:
    pred_dir = cache / "pred" / args.run
    command = [
        sys.executable, "-m", "flattener.dewarp.predict",
        "--weights", str(args.checkpoint), "--device", args.device,
        "docunet", "--input-dir", str(CROP_DIR), "--output-dir", str(pred_dir),
        "--force-render",
    ]  # fmt: skip
    if args.overwrite:
        command.append("--overwrite")
    started = time.perf_counter()
    run(command, env, ok=(0, 1))  # 1: some map failed its checks and was force-rendered
    wall = time.perf_counter() - started
    record = json.loads((pred_dir / "predictions.json").read_text(encoding="utf-8"))
    if not record.get("complete") or record["counts"]["items"] != 130:
        raise SystemExit(f"error: {pred_dir}/predictions.json is incomplete")
    return record, wall


def score(args, cache: Path, env: dict[str, str], out_dir: Path) -> tuple[dict, str]:
    pred_dir = cache / "pred" / args.run
    report_path = out_dir / "docunet-once.json"
    if report_path.is_file() and not args.overwrite:
        source = siftflow.resolve_source_dir(env["FLATTENER_SIFTFLOW_SOURCE"])
        if report_matches(report_path, pred_dir, siftflow.source_tree_sha256(source)):
            return json.loads(report_path.read_text(encoding="utf-8")), "kept"
        raise SystemExit(f"error: {report_path} scores other images or settings; pass --overwrite")
    command = [
        sys.executable, "-m", "flattener.benchmark", "docunet", str(pred_dir),
        "--ref-resize", "once", "--workers", str(args.workers),
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
    shutil.copyfile(cache / "pred" / args.run / "predictions.json", out_dir / "predictions.json")

    counts = record["counts"]
    parts = [
        f"{args.run} step={record['model'].get('step')}",
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
    print(f"total predict wall {predict_wall:.0f} s", file=sys.stderr)
    print(" | ".join(parts), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
