"""Reproducible Linux CPU identity-render smoke baseline; writes JSON to stdout."""

import argparse
import json
import os
import platform
import resource
import statistics
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import PIL
import torch

import flattener
from flattener import RenderConfig, Status, identity_map, render
from flattener.validation import validate_size


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def current_rss_bytes():
    status = read_text("/proc/self/status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def cpu_model():
    for line in (read_text("/proc/cpuinfo") or "").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in ("model name", "Hardware"):
            return value.strip()
    return "unknown"


def summarize(values):
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


def run_once(image, grid, config):
    started = perf_counter()
    result = render(image, grid, config=config)
    wall_seconds = perf_counter() - started
    # Correctness checks are deliberately outside the render-call timing.
    exact = bool(np.array_equal(result.image, image))
    return {
        "status": str(result.status),
        "identity_exact": exact,
        "wall_render_call_seconds": wall_seconds,
        "stage_seconds": result.metadata["timings"],
        "validation": result.metadata["validation"],
    }


def main():
    rss_after_imports = current_rss_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this benchmark's ru_maxrss units and RSS snapshots require Linux")
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("repeats/threads must be positive and warmup must be nonnegative")
    config = RenderConfig()
    size = validate_size(
        (args.height, args.width),
        max_pixels=min(config.max_input_pixels, config.max_output_pixels),
        max_side=config.max_side,
        name="benchmark size",
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    fixture_started = perf_counter()
    # Independent integer channel patterns; only 1D coordinate temporaries.
    y = np.arange(args.height, dtype=np.uint32)[:, None]
    x = np.arange(args.width, dtype=np.uint32)[None, :]
    image = np.empty((*size, 3), dtype=np.uint8)
    image[..., 0] = (3 * x + 5 * y) % 256
    image[..., 1] = (11 * x + 7 * y + 31) % 256
    image[..., 2] = (x + 13 * y + 97) % 256
    grid = identity_map(size, config=config)
    del x, y
    fixture_seconds = perf_counter() - fixture_started
    rss_after_fixture = current_rss_bytes()

    warmup_runs = [run_once(image, grid, config) for _ in range(args.warmup)]
    rss_after_warmup = current_rss_bytes()
    runs = [run_once(image, grid, config) for _ in range(args.repeats)]
    rss_after_runs = current_rss_bytes()
    # Linux reports a lifetime process high-water mark in KiB. This also
    # includes imports, fixture construction, warmup, and correctness checks.
    process_peak_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    stage_summary = {
        stage: summarize([run["stage_seconds"][stage] for run in runs])
        if all(run["stage_seconds"][stage] is not None for run in runs)
        else None
        for stage in runs[0]["stage_seconds"]
    }
    all_exact = all(
        run["status"] == Status.SUCCESS and run["identity_exact"] for run in warmup_runs + runs
    )
    report = {
        "benchmark": "cpu_identity_renderer",
        "purpose": "Smoke baseline; not optimization evidence or model quality evaluation.",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu0_scaling_governor": read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            )
            or "unknown",
            "cpu0_energy_performance_preference": read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
            )
            or "unknown",
            "versions": {
                "python": platform.python_version(),
                "flattener": flattener.__version__,
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "pillow": PIL.__version__,
            },
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "input_size": list(size),
        "output_size": list(size),
        "config": asdict(config),
        "model_id": None,
        "model_memory_bytes": None,
        "fixture": {
            "description": "Analytic RGB uint8 patterns and float32 normalized identity map.",
            "creation_seconds": fixture_seconds,
            "image_bytes": image.nbytes,
            "map_bytes": grid.nbytes,
            "timing_scope": "Fixture construction only; excluded from render timing summaries.",
        },
        "timing_scope": (
            "Wall times cover each render() call only. Renderer stages use internal monotonic "
            "timings. Fixture creation, identity checks, and JSON serialization are excluded. "
            "Warmup runs are listed separately and excluded from all timing summaries."
        ),
        "p95_method": "NumPy percentile with linear interpolation; small samples are descriptive.",
        "warmup_runs": warmup_runs,
        "measured_runs": runs,
        "timing_summary": {
            "wall_render_call": summarize([run["wall_render_call_seconds"] for run in runs]),
            "stages": stage_summary,
        },
        "all_identity_checks_exact": all_exact,
        "memory": {
            "current_rss_after_imports_bytes": rss_after_imports,
            "current_rss_after_fixture_bytes": rss_after_fixture,
            "current_rss_after_warmup_bytes": rss_after_warmup,
            "current_rss_after_measured_runs_bytes": rss_after_runs,
            "process_peak_rss_bytes": process_peak_bytes,
            "rss_source": "/proc/self/status VmRSS (KiB converted to bytes)",
            "peak_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss on Linux (KiB to bytes)",
            "scope": (
                "Whole process through the end of measured runs, including imports, fixtures, "
                "warmup, output buffers, and identity checks. RSS snapshots occur after each "
                "phase with run results released. This is not a renderer-only allocation peak. "
                "Allocator-retained memory can remain in current RSS. The two kernel sources "
                "have different accounting granularity. No short-peak sampling."
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
