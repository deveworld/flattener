"""Benchmark one frozen checkpoint/photo on Linux; write one exclusive JSON report.

Example (sizes are explicit, no aspect-ratio inference):
  python benchmarks/benchmark_pipeline.py --checkpoint path/to/legacy-checkpoint.pt \
    --input page.jpg --output-height 96 --output-width 144 \
    --threads 1 --warmup 1 --repeats 5 --output-json outputs/pipeline-before.json

Add --save-samples to include PNG/mask/JSON saving in each end-to-end measurement.
Samples go in a newly created temporary directory; its path is recorded and the
directory is retained for inspection. Input images/checkpoints are never copied
into the report or a package. Warmup results and failures remain explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import PIL
import torch

import flattener
from flattener import FlattenerError, RenderConfig, RenderResult, Status, load_image, render
from flattener.image_io import save_result
from flattener.learning.inference import LoadedPredictor, load_predictor
from flattener.learning.model import PREPROCESS_VERSION
from flattener.renderer import _new_metadata
from flattener.validation import validate_size


def read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def current_rss_bytes() -> int | None:
    for line in (read_text("/proc/self/status") or "").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def array_record(array: np.ndarray | None, *, mask: bool = False) -> dict | None:
    if array is None:
        return None
    # Hash a contiguous view without allocating another image-sized bytes object.
    value = np.ascontiguousarray(array)
    record = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": value.nbytes,
        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
    }
    if mask:
        record["true_pixels"] = int(np.count_nonzero(value))
        record["true_fraction"] = record["true_pixels"] / value.size
    return record


def environment() -> dict:
    cpu = "unknown"
    for line in (read_text("/proc/cpuinfo") or "").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware"}:
            cpu = value.strip()
            break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu,
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu0_scaling_governor": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu0_energy_performance_preference": read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
        ),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "versions": {
            "python": platform.python_version(),
            "flattener": flattener.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": str(torch.__version__),
        },
    }


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "count": len(values),
        "median_seconds": statistics.median(values),
        "population_variance_seconds2": statistics.pvariance(values),
        "p95_seconds": float(np.percentile(values, 95, method="linear")),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def timing_summaries(runs: list[dict]) -> dict:
    names = sorted({name for run in runs for name in run["seconds"]})
    return {
        name: summarize([run["seconds"][name] for run in runs if run["seconds"][name] is not None])
        for name in names
    }


def run_once(
    input_path: Path,
    predictor: LoadedPredictor,
    output_size: tuple[int, int],
    config: RenderConfig,
    run_id: str,
    sample_root: Path | None,
) -> dict:
    seconds = dict.fromkeys(
        (
            "decode",
            "preprocess",
            "predict",
            "coordinate_expansion",
            "geometry_validation",
            "coordinates",
            "renderer_prepare_input",
            "render",
            "render_call_wall",
            "save_internal",
            "save_wall",
            "before_save_wall",
            "end_to_end",
        )
    )
    seconds["model_load"] = 0.0  # The same immutable loaded predictor serves every run.
    image = grid = None
    metadata = _new_metadata(config, timing_scope="pipeline_benchmark")
    predictor._metadata(metadata)
    metadata["requested_output_size"] = list(output_size)
    started = perf_counter()
    stage_started = started
    stage = "decode"
    try:
        image = load_image(input_path, config=config)
        seconds["decode"] = perf_counter() - stage_started
        metadata["source_size"] = list(image.shape[:2])
        stage = "inference"
        # This is the existing validated inference path, not a benchmark model.
        # Its private stage metadata avoids doing a second inference to hash maps.
        grid = predictor._predict(image, output_size, metadata)
        seconds["preprocess"] = metadata["timings"]["prepare_input"]
        seconds["predict"] = metadata["timings"]["predict"]
        seconds["coordinate_expansion"] = metadata["timings"]["process_coordinates"]
        stage = "render"
        stage_started = perf_counter()
        result = render(image, grid, config=config)
        seconds["render_call_wall"] = perf_counter() - stage_started
        render_metadata = result.metadata
        seconds["renderer_prepare_input"] = render_metadata["timings"]["prepare_input"]
        seconds["geometry_validation"] = render_metadata["timings"]["process_coordinates"]
        seconds["render"] = render_metadata["timings"]["render"]
        seconds["coordinates"] = (seconds["coordinate_expansion"] or 0) + (
            seconds["geometry_validation"] or 0
        )
        metadata["validation"] = render_metadata["validation"]
        metadata["actual_output_size"] = render_metadata["actual_output_size"]
        metadata["failure_stage"] = render_metadata["failure_stage"]
        metadata["timings"]["prepare_input"] = (
            seconds["decode"]
            + (seconds["preprocess"] or 0)
            + (seconds["renderer_prepare_input"] or 0)
        )
        metadata["timings"]["process_coordinates"] = seconds["coordinates"]
        metadata["timings"]["render"] = seconds["render"]
        result.metadata = metadata
    except FlattenerError as error:
        if stage == "decode":
            seconds["decode"] = perf_counter() - stage_started
        elif stage == "inference":
            seconds["preprocess"] = metadata["timings"]["prepare_input"]
            seconds["predict"] = metadata["timings"]["predict"]
            seconds["coordinate_expansion"] = metadata["timings"]["process_coordinates"]
        metadata.update(error.metadata)
        metadata["failure_stage"] = metadata.get("failure_stage") or stage
        metadata["timings"]["prepare_input"] = (seconds["decode"] or 0) + (
            seconds["preprocess"] or 0
        )
        result = RenderResult(
            status=error.status,
            reason_code=error.reason_code,
            message=error.message,
            metadata=metadata,
        )

    seconds["before_save_wall"] = perf_counter() - started
    result.metadata["timing_scope"] = "file"
    result.metadata["timings"]["total"] = seconds["before_save_wall"]
    result.metadata["benchmark_run_id"] = run_id
    processing_status = str(result.status)
    processing_reason = result.reason_code
    artifacts = None
    save_failure = None
    if sample_root is not None:
        save_started = perf_counter()
        try:
            artifacts = save_result(result, sample_root / run_id)
            seconds["save_internal"] = result.metadata["timings"]["save"]
        except FlattenerError as error:
            save_failure = {
                "status": str(error.status),
                "reason_code": error.reason_code,
                "message": error.message,
                "metadata": error.metadata,
            }
        seconds["save_wall"] = perf_counter() - save_started
    seconds["end_to_end"] = perf_counter() - started
    # Hashing, mask counts, and report assembly are outside all per-image timers.
    return {
        "run_id": run_id,
        "status": save_failure["status"] if save_failure else str(result.status),
        "reason_code": save_failure["reason_code"] if save_failure else result.reason_code,
        "message": save_failure["message"] if save_failure else result.message,
        "processing_status": processing_status,
        "processing_reason_code": processing_reason,
        "used_fallback": result.used_fallback,
        "failure_stage": "save" if save_failure else result.metadata["failure_stage"],
        "seconds": seconds,
        "input_image": array_record(image),
        "backward_map": array_record(grid),
        "output_image": array_record(result.image),
        "source_mask": array_record(result.source_mask, mask=True),
        "target_mask": array_record(result.target_mask, mask=True),
        "valid_mask": array_record(result.valid_mask, mask=True),
        "validation": result.metadata["validation"],
        "artifacts": artifacts,
        "save_failure": save_failure,
    }


def consistency(runs: list[dict]) -> dict:
    fields = (
        "input_image",
        "backward_map",
        "output_image",
        "source_mask",
        "target_mask",
        "valid_mask",
    )
    unique = {
        field: sorted({run[field]["sha256"] for run in runs if run[field] is not None})
        for field in fields
    }
    return {
        "unique_hashes": unique,
        "available_arrays_repeat_exactly": all(len(hashes) <= 1 for hashes in unique.values()),
        "scope": (
            "Exact raw-array SHA-256 across warmup/measured runs, outside timed regions. "
            "This checks repeatability, not geometric or real-photo quality. Missing arrays "
            "are reported per run; their absence is never evidence of correctness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-height", required=True, type=int)
    parser.add_argument("--output-width", required=True, type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux is required for RSS units, process snapshots, and CPU metadata")
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("threads/repeats must be positive; warmup must be nonnegative")
    if args.output_json.exists() or args.output_json.is_symlink():
        parser.error("output JSON already exists; choose a new path")
    if not args.input.is_file() or not args.checkpoint.is_file():
        parser.error("--input and --checkpoint must be existing files")
    config = RenderConfig(on_error="return")
    try:
        output_size = validate_size(
            (args.output_height, args.output_width),
            max_pixels=config.max_output_pixels,
            max_side=config.max_side,
            name="benchmark output size",
        )
    except FlattenerError as error:
        parser.error(error.message)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rss_imports = current_rss_bytes()
    report = {
        "schema_version": "pipeline-benchmark-1.0",
        "benchmark": "cpu_loaded_checkpoint_pipeline",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "environment": environment(),
        "config": asdict(config),
        "input": {"path": str(args.input.resolve()), "file_sha256": file_hash(args.input)},
        "checkpoint": {"path": str(args.checkpoint.resolve())},
        "output_size": list(output_size),
        "warmup_count": args.warmup,
        "repeat_count": args.repeats,
        "save_samples": args.save_samples,
        "sample_directory": None,
        "timing_scope": (
            "One checkpoint load is timed separately. Every hot run decodes the same input, "
            "uses LoadedPredictor._predict for preprocessing/prediction/map expansion, then "
            "calls the public renderer. These are the production inference operations with "
            "benchmark orchestration, not a timing of public unwarp(). Optional save_wall "
            "includes PNG/mask/JSON creation and commit. End-to-end runs from decode through "
            "the save call, excluding model load, array hashes, report serialization, and "
            "sample-root creation. Repeated file reads can benefit from the OS page cache."
            " Saved files use buffered writes; this does not measure fsync durability."
        ),
        "stage_definitions": {
            "decode": "File I/O, decoding, EXIF orientation, RGB conversion, and input validation.",
            "preprocess": "Model input resize/tensor preparation only.",
            "predict": "Model forward pass plus output shape/dtype/finite checks.",
            "coordinate_expansion": "Dense expansion, NumPy copy, and finite validation.",
            "coordinates": (
                "coordinate_expansion + renderer geometry_validation (overlapping aggregate)."
            ),
            "render": "Renderer sampling/quantization/fill stage; input checks are separate.",
            "render_call_wall": "Entire render() call; overlaps its reported internal stages.",
            "save_internal": (
                "save_result staging timer, excluding JSON and commit; use save_wall for I/O."
            ),
            "model_load": "Zero per hot run; the actual one-time load is under checkpoint.",
        },
        "summary_scope": (
            "Measured runs only; warmups listed separately. No failed run is discarded."
        ),
        "p95_method": "NumPy percentile with linear interpolation; small samples are descriptive.",
    }
    package = Path(flattener.__file__).parent
    report["implementation_sha256"] = {
        name: file_hash(package / name)
        for name in (
            "renderer.py",
            "validation.py",
            "coordinates.py",
            "image_io.py",
            "learning/inference.py",
            "learning/model.py",
        )
    }
    load_started = perf_counter()
    try:
        predictor = load_predictor(args.checkpoint)
    except FlattenerError as error:
        report["load_failure"] = {
            "status": str(error.status),
            "reason_code": error.reason_code,
            "message": error.message,
        }
        report["checkpoint"]["load_wall_seconds"] = perf_counter() - load_started
        warmup_runs = runs = []
        rss_model = rss_warmup = current_rss_bytes()
    else:
        report["checkpoint"].update(
            load_wall_seconds=perf_counter() - load_started,
            load_internal_seconds=predictor.load_seconds,
            model_id=predictor.model_id,
            sha256=predictor.model_sha256,
            checksum_verified=predictor.checksum_verified,
            preprocessing_version=PREPROCESS_VERSION,
            model_config=asdict(predictor.model_config),
            parameter_tensor_bytes=sum(
                value.numel() * value.element_size() for value in predictor._model.parameters()
            ),
        )
        if any(a < b for a, b in zip(output_size, predictor.model_config.grid_size, strict=True)):
            parser.error("output dimensions may not be smaller than the checkpoint grid")
        rss_model = current_rss_bytes()
        sample_root = (
            Path(tempfile.mkdtemp(prefix="flattener-pipeline-")) if args.save_samples else None
        )
        report["sample_directory"] = str(sample_root) if sample_root is not None else None
        warmup_runs = [
            run_once(args.input, predictor, output_size, config, f"warmup-{i:03d}", sample_root)
            for i in range(args.warmup)
        ]
        rss_warmup = current_rss_bytes()
        runs = [
            run_once(args.input, predictor, output_size, config, f"measured-{i:03d}", sample_root)
            for i in range(args.repeats)
        ]
    rss_runs = current_rss_bytes()
    process_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    all_runs = warmup_runs + runs
    report["input"]["file_sha256_after_runs"] = file_hash(args.input)
    report["input"]["unchanged"] = (
        report["input"]["file_sha256"] == report["input"]["file_sha256_after_runs"]
    )
    report["checkpoint"]["file_sha256_after_runs"] = file_hash(args.checkpoint)
    report["checkpoint"]["unchanged"] = (
        report["checkpoint"].get("sha256") == report["checkpoint"]["file_sha256_after_runs"]
    )
    report["warmup_runs"] = warmup_runs
    report["measured_runs"] = runs
    report["status_counts"] = dict(Counter(run["status"] for run in runs))
    successful = [run for run in runs if run["status"] == Status.SUCCESS]
    report["measured_success_count"] = len(successful)
    report["measured_failure_count"] = len(runs) - len(successful)
    report["timing_summary_all_outcomes"] = timing_summaries(runs)
    report["timing_summary_success_only"] = timing_summaries(successful)
    report["consistency"] = consistency(all_runs)
    report["memory"] = {
        "current_rss_after_imports_bytes": rss_imports,
        "current_rss_after_model_load_bytes": rss_model,
        "current_rss_after_warmup_bytes": rss_warmup,
        "current_rss_after_measured_runs_bytes": rss_runs,
        "process_peak_rss_bytes": process_peak,
        "rss_source": "/proc/self/status VmRSS (KiB converted to bytes)",
        "peak_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss on Linux (KiB to bytes)",
        "scope": (
            "Whole process through measured runs: imports, model loading, decoding, inference, "
            "rendering, optional saving, warmup, and untimed hashing/checks. Arrays are released "
            "before phase RSS snapshots; loaded model remains. Includes allocator retention. "
            "This is not a renderer-only or model-only allocation peak. No short-peak sampling."
        ),
    }
    passed = (
        len(runs) == args.repeats
        and all(run["status"] == Status.SUCCESS for run in all_runs)
        and report["consistency"]["available_arrays_repeat_exactly"]
        and report["input"]["unchanged"]
        and report["checkpoint"]["unchanged"]
    )
    report["all_runs_successful_and_repeatable"] = passed
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is authoritative even if another process races the preflight.
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
