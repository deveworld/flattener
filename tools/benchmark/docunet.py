"""DocUNet evaluation with MS-SSIM, local distortion and aligned distortion.

Reports contain per-image scores, model settings and aggregate results.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import string
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from . import metrics
from .reporting import (
    environment,
    failure_counts,
    mean,
    prediction_record,
    prediction_status,
    report_evidence,
    utc_now,
)
from .siftflow import SiftFlow, SiftFlowBuild

REPORT_SCHEMA = "flattener-benchmark-docunet-1.0"
DOCUMENTS = tuple(range(1, 66))
VIEWS = (1, 2)
ROTATED_DOCUMENT = 64
CONVENTIONS = ("130_raw", "130_rot64", "128_excl64")
# Report every convention unless told otherwise: published tables mix all three.
DEFAULT_CONVENTIONS = CONVENTIONS
CONVENTION_DESCRIPTIONS = {
    "130_raw": "all 130 photos as published (64_1/64_2 upside down); what PaperEdge "
    "adres.txt and the released DocTr/DocScanner/DocGeoNet MATLAB scripts compute "
    "(their rot90 block for document 64 is commented out); validated by the PaperEdge "
    "MATLAB check",
    "130_rot64": "64_1/64_2 rotated 180 degrees before scoring (those scripts with the "
    "rot90 block enabled); not covered by the calibration checks",
    "128_excl64": "64_1/64_2 excluded (UVDoc: DocUnet_results has 128 images and "
    "evalScript.m skips missing files); validated by the UVDoc table checks",
}
REF_RESIZE_MODES = ("once", "uvdoc_view2_double")
DEFAULT_WORKERS = 2
DEFAULT_REFERENCE_PATTERN = "docunet-reference-{k}.png"
DEFAULT_REFERENCES_DIR = Path("outputs/benchmarks/docunet/reference")

PROTOCOL = {
    "alpha_composite_for_metrics": "black (MATLAB one-output imread; PNG bKGD if present)",
    "flow_direction": "scan -> prediction (siftFlow(scan, prediction))",
    "gray_before_resize": True,
    "metric_area": metrics.METRIC_AREA,
    "ms_ssim": "weighted sum of 5 MATLAB ssim levels with impyramid reduce",
    "ms_ssim_weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
    "resize": "MATLAB imresize bicubic (Keys a=-0.5), antialias when shrinking, ceil size",
    "siftflow": {
        "alpha": 510.0,
        "cell_size": 3,
        "d": 10200.0,
        "gamma": 1.275,
        "grid_spacing": 1,
        "nIterations": 30,
        "nTopIterations": 60,
        "nlevels": 4,
        "prefilter": "7x7 sigma=1 gaussian (replicate), then 0.5x bicubic",
        "topwsize": 10,
        "wsize": 2,
    },
    "ad": "PaperEdge evalAlignedUnwarp.m: Sobel mask > 0.5, per-axis scale+shift removed",
    "ad_solvers": {
        "exact_lstsq": "exact least squares (numpy lstsq)",
        "lsqr_4iter": "LSQR with MATLAB lsqr(A,B) defaults; its means match MATLAB in the "
        "DocUNet calibration checks, per-image AD can differ by up to ~0.016",
    },
    "conventions": CONVENTION_DESCRIPTIONS,
}


def convention_views(convention: str) -> tuple[tuple[int, int, bool], ...]:
    """Return ``(document, view, rotated_180)`` triples scored under a convention."""
    if convention == "130_raw":
        return tuple((k, m, False) for k in DOCUMENTS for m in VIEWS)
    if convention == "130_rot64":
        return tuple((k, m, k == ROTATED_DOCUMENT) for k in DOCUMENTS for m in VIEWS)
    if convention == "128_excl64":
        return tuple((k, m, False) for k in DOCUMENTS if k != ROTATED_DOCUMENT for m in VIEWS)
    raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")


def validate_conventions(conventions: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(conventions))
    if not selected:
        raise ValueError("at least one convention is required")
    for convention in selected:
        convention_views(convention)
    return selected


def required_views(conventions: Iterable[str]) -> tuple[tuple[int, int, bool], ...]:
    views = {view for convention in conventions for view in convention_views(convention)}
    return tuple(sorted(views))


def reference_resizes(ref_resize: str, view: int) -> int:
    """How many times the scan is resized before scoring ``view``."""
    if ref_resize == "once":
        return 1
    if ref_resize == "uvdoc_view2_double":
        # UVDoc evalScript.m resizes rimg inside the view loop, so view 2 sees it twice.
        return view
    raise ValueError(f"ref_resize must be one of {REF_RESIZE_MODES}, got {ref_resize!r}")


def format_name(pattern: str, *, fields: Sequence[str], **values: int) -> str:
    """Fill ``{k}``/``{m}`` in a relative file pattern after strict validation."""
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("file pattern must be a non-empty string")
    used = set()
    for _, field, spec, conversion in string.Formatter().parse(pattern):
        if field is None:
            continue
        if field not in fields or spec or conversion:
            raise ValueError(f"pattern {pattern!r} may only use plain {sorted(fields)} fields")
        used.add(field)
    if used != set(fields):
        raise ValueError(f"pattern {pattern!r} must contain {sorted(fields)}")
    name = pattern.format(**values)
    # Check both path flavours so a rooted ("/x", "\\x") or drive path cannot escape
    # the directory on either platform.
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.anchor or ".." in windows.parts or not posix.parts:
        raise ValueError(f"pattern {pattern!r} must produce a relative path inside the directory")
    return name


@dataclass(frozen=True)
class ViewTask:
    view: int
    rotated: bool
    prediction_path: str


@dataclass(frozen=True)
class DocumentTask:
    document: int
    reference_path: str
    views: tuple[ViewTask, ...]
    ref_resize: str
    ad_solver: str


def score_document(task: DocumentTask, siftflow: SiftFlow | None) -> list[dict[str, Any]]:
    """Score all requested views of one document; the scan is decoded once."""
    reference_bytes = Path(task.reference_path).read_bytes()
    reference = metrics.read_image(reference_bytes)
    prepared: dict[int, metrics.PreparedReference] = {}
    rows = []
    for view in task.views:
        started = time.perf_counter()
        resizes = reference_resizes(task.ref_resize, view.view)
        if resizes not in prepared:
            prepared[resizes] = metrics.prepare_reference(reference, resizes=resizes)
        prediction_bytes = Path(view.prediction_path).read_bytes()
        prediction = metrics.read_image(prediction_bytes)
        if view.rotated:
            prediction = np.ascontiguousarray(np.rot90(prediction, 2))
        score = metrics.score_prepared(
            prediction, prepared[resizes], siftflow=siftflow, ad_solver=task.ad_solver
        )
        row = score.to_dict()
        row.update(
            {
                "id": f"{task.document}_{view.view}",
                "k": task.document,
                "m": view.view,
                "rotated_180": view.rotated,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        rows.append(row)
    return rows


ScoreTask = Callable[[Any, SiftFlow | None], list[dict[str, Any]]]
_WORKER_SIFTFLOW: SiftFlow | None = None


def _initialize_worker(build: SiftFlowBuild | None) -> None:
    global _WORKER_SIFTFLOW
    import torch

    torch.set_num_threads(1)
    _WORKER_SIFTFLOW = None if build is None else SiftFlow(build)


def _score_in_worker(score: ScoreTask, task: Any) -> list[dict[str, Any]]:
    return score(task, _WORKER_SIFTFLOW)


@contextlib.contextmanager
def _single_thread_environment():
    """Children inherit one-thread BLAS/OpenMP settings; the parent is restored."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_tasks(
    tasks: Sequence[Any],
    build: SiftFlowBuild | None,
    *,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
    score: ScoreTask = score_document,
    total: int | None = None,
) -> list[dict[str, Any]]:
    """Score tasks in-process (``workers == 1``) or in a spawned process pool.

    ``score`` turns one task into rows; it must be a module-level function so the pool
    can pickle it. ``total`` is the expected row count for ``progress`` (default: the
    views of the ``DocumentTask`` list).
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if total is None:
        total = sum(len(task.views) for task in tasks)
    rows: list[dict[str, Any]] = []

    def collect(new_rows: list[dict[str, Any]]) -> None:
        for row in new_rows:
            rows.append(row)
            if progress is not None:
                progress(row, len(rows), total)

    if workers == 1:
        siftflow = None if build is None else SiftFlow(build)
        for task in tasks:
            collect(score(task, siftflow))
        return rows
    context = multiprocessing.get_context("spawn")
    with _single_thread_environment():
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)) or 1,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(build,),
        ) as executor:
            futures = [executor.submit(_score_in_worker, score, task) for task in tasks]
            try:
                for future in as_completed(futures):
                    collect(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    return rows


def _row_key(row: dict[str, Any]) -> tuple[int, int, bool]:
    return (row["k"], row["m"], row["rotated_180"])


def summarize(rows: Sequence[dict[str, Any]], conventions: Sequence[str]) -> dict[str, Any]:
    """Means per convention (and per view) over the rows each convention selects."""
    by_key = {_row_key(row): row for row in rows}
    summary = {}
    for convention in conventions:
        selected = [by_key[view] for view in convention_views(convention)]
        summary[convention] = _means(selected)
        summary[convention]["views"] = {
            str(view): _means([row for row in selected if row["m"] == view]) for view in VIEWS
        }
    return summary


def _means(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Means, plus ``prediction_failures`` (scored images not recorded as passing their
    checks: force-rendered failures and images ``predictions.json`` does not list) and
    ``prediction_unrecorded`` (the latter); both ``None`` when the predictions carry no
    record."""
    result: dict[str, Any] = {"n": len(rows)}
    for name in ("ms_ssim", "ld", "ad", "ad_exact_lstsq", "ad_lsqr_4iter"):
        values = [row[name] for row in rows]
        result[name] = None if any(value is None for value in values) else mean(values)
    result.update(failure_counts([row.get("prediction_status") for row in rows]))
    return result


def reference_paths(
    references_dir: Path,
    reference_pattern: str,
    documents: Iterable[int],
) -> dict[int, str]:
    """Locate the reference scans required by this evaluation."""
    paths = {}
    for document in sorted(set(documents)):
        path = references_dir / format_name(reference_pattern, fields=("k",), k=document)
        if not path.is_file():
            raise FileNotFoundError(f"missing DocUNet scan {path}")
        paths[document] = str(path)
    return paths


def evaluate_docunet(
    predictions_dir: str | os.PathLike[str],
    *,
    pattern: str,
    conventions: Sequence[str],
    build: SiftFlowBuild | None,
    ref_resize: str = "once",
    ad_solver: str = metrics.DEFAULT_AD_SOLVER,
    references_dir: str | os.PathLike[str] = DEFAULT_REFERENCES_DIR,
    reference_pattern: str = DEFAULT_REFERENCE_PATTERN,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Score a directory of DocUNet predictions and return the report dictionary.

    ``build`` comes from :func:`tools.benchmark.siftflow.build_siftflow`; ``None``
    computes MS-SSIM only, and the report has ``comparable: false``.
    """
    conventions = validate_conventions(conventions)
    reference_resizes(ref_resize, 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if ad_solver not in metrics.AD_SOLVERS:
        raise ValueError(f"ad_solver must be one of {metrics.AD_SOLVERS}, got {ad_solver!r}")
    predictions_root = Path(predictions_dir).expanduser().resolve()
    references_root = Path(references_dir).expanduser().resolve()
    if not predictions_root.is_dir():
        raise NotADirectoryError(f"predictions directory not found: {predictions_root}")
    views = required_views(conventions)
    documents = sorted({k for k, _, _ in views})
    reference_paths(references_root, reference_pattern, documents)
    record = prediction_record(predictions_root)
    missing, names = [], {}
    grouped: dict[int, list[ViewTask]] = {}
    for k, m, rotated in views:
        names[f"{k}_{m}"] = format_name(pattern, fields=("k", "m"), k=k, m=m)
        path = predictions_root / names[f"{k}_{m}"]
        if not path.is_file():
            missing.append(names[f"{k}_{m}"])
        grouped.setdefault(k, []).append(ViewTask(m, rotated, str(path)))
    if missing:
        recorded = [prediction_status(record, name) for name in missing]
        failed = [status for status in recorded if status not in (None, "success", "unrecorded")]
        note = (
            f"; predictions.json records {len(failed)} of them as failed checks "
            "(predict with --force-render to score the forced images)"
            if failed
            else ""
        )
        raise FileNotFoundError(
            f"{len(missing)} predictions missing in {predictions_root}, first: {missing[:5]}{note}"
        )
    tasks = [
        DocumentTask(
            document=k,
            reference_path=str(
                references_root / format_name(reference_pattern, fields=("k",), k=k)
            ),
            views=tuple(sorted(grouped[k], key=lambda item: (item.view, item.rotated))),
            ref_resize=ref_resize,
            ad_solver=ad_solver,
        )
        for k in documents
    ]
    started = time.perf_counter()
    rows = run_tasks(tasks, build, workers=workers, progress=progress)
    wall_seconds = time.perf_counter() - started
    rows.sort(key=_row_key)
    for row in rows:
        row["prediction_status"] = prediction_status(record, names[row["id"]])
    return {
        "benchmark": "docunet",
        "comparable": build is not None,
        "config": {
            "ad_solver": ad_solver,
            "conventions": list(conventions),
            "prediction_pattern": pattern,
            "ref_resize": ref_resize,
            "reference_pattern": reference_pattern,
            "workers": workers,
        },
        "created_utc": utc_now(),
        "environment": environment(),
        "evidence": report_evidence(record),
        "inputs": {
            "predictions_dir": str(predictions_root),
            "references_dir": str(references_root),
        },
        "predictions_record": record,
        "protocol": PROTOCOL,
        "rows": rows,
        "schema": REPORT_SCHEMA,
        "siftflow": None if build is None else build.metadata(),
        "summary": summarize(rows, conventions),
        "timing": {
            "image_seconds_total": round(sum(row["seconds"] for row in rows), 3),
            "wall_seconds": round(wall_seconds, 3),
        },
    }


def print_progress(row: dict[str, Any], done: int, total: int) -> None:
    rotated = " rot180" if row["rotated_180"] else ""
    parts = [f"ms_ssim={row['ms_ssim']:.5f}"]
    if row["ld"] is not None:
        parts += [f"ld={row['ld']:.3f}", f"ad={row['ad']:.4f}"]
    print(
        f"[{done}/{total}] {row['id']}{rotated} {' '.join(parts)} ({row['seconds']:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
