"""Coordinate metrics against fixed ground truth, and audited scene evaluation.

Metrics use raw predictions in source pixels, before renderer validation, and
never clip coordinates. The target set is always the full GT mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import PIL
import torch

from ..contracts import FlattenerError, RenderConfig
from ..renderer import render
from ..validation import validate_geometry


def _metric_inputs(prediction, truth, gt_mask, source_size):
    for name, array in (("prediction", prediction), ("truth", truth)):
        if (
            not isinstance(array, np.ndarray)
            or array.ndim != 3
            or array.shape[-1] != 2
            or min(array.shape[:2]) < 1
            or not np.issubdtype(array.dtype, np.floating)
        ):
            raise ValueError(f"{name} must be a floating (height, width, 2) array")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have equal shapes")
    if (
        not isinstance(gt_mask, np.ndarray)
        or gt_mask.dtype != np.bool_
        or gt_mask.shape != truth.shape[:2]
    ):
        raise ValueError("gt_mask must be a boolean array matching the output shape")
    if (
        not isinstance(source_size, (tuple, list))
        or len(source_size) != 2
        or any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in source_size)
    ):
        raise ValueError("source_size must be positive integer (height, width)")
    if not gt_mask.any():
        raise ValueError("gt_mask selects no target pixels")
    if not np.isfinite(truth[gt_mask]).all():
        raise ValueError("ground truth has non-finite selected coordinates")
    if not np.isfinite(prediction[gt_mask]).all():
        raise ValueError("prediction has non-finite selected coordinates")


def endpoint_error(
    prediction: np.ndarray,
    truth: np.ndarray,
    gt_mask: np.ndarray,
    source_size: tuple[int, int],
) -> float:
    """Mean Euclidean source-pixel error on the fixed GT-visible target set.

    Inputs are normalized backward maps in (x, y) order with
    ``align_corners=False``. Out-of-source predictions remain in the error;
    non-finite selected predictions invalidate the metric. Values outside the
    ground-truth mask are ignored, including NaN/Inf values.
    """
    _metric_inputs(prediction, truth, gt_mask, source_size)
    difference = prediction[gt_mask].astype(np.float64) - truth[gt_mask].astype(np.float64)
    difference *= np.asarray([source_size[1], source_size[0]], dtype=np.float64) / 2
    result = float(np.hypot(difference[:, 0], difference[:, 1]).mean())
    if not np.isfinite(result):
        raise ValueError("endpoint error overflowed to a non-finite value")
    return result


def boundary_mask(gt_mask: np.ndarray, width: int = 8) -> np.ndarray:
    """Select the outer ``width`` output pixels intersected with fixed GT."""
    if not isinstance(gt_mask, np.ndarray) or gt_mask.dtype != bool or gt_mask.ndim != 2:
        raise ValueError("gt_mask must be a two-dimensional boolean array")
    if isinstance(width, bool) or not isinstance(width, Integral) or width < 1:
        raise ValueError("boundary width must be a positive integer")
    result = np.zeros_like(gt_mask)
    result[:width] = True
    result[-width:] = True
    result[:, :width] = True
    result[:, -width:] = True
    return result & gt_mask


def coordinate_metrics(prediction, truth, gt_mask, source_size) -> dict[str, Any]:
    """Measure raw error and geometry independently of rendering acceptance."""
    epe = endpoint_error(prediction, truth, gt_mask, source_size)
    boundary = boundary_mask(gt_mask)
    # Allow any out-of-bounds fraction so the fold check also runs on maps that
    # lie mostly outside the source.
    with np.errstate(over="ignore"):
        coordinates = prediction.astype(np.float32)
    try:
        _, _, geometry = validate_geometry(
            coordinates, gt_mask, RenderConfig(max_out_of_bounds_fraction=1)
        )
        geometry_reason = None
    except FlattenerError as exc:
        geometry = exc.metadata.get("validation", {})
        geometry_reason = exc.reason_code
    return {
        "epe_pixels": epe,
        "boundary8_epe_pixels": (
            endpoint_error(prediction, truth, boundary, source_size) if boundary.any() else None
        ),
        "gt_pixels": int(gt_mask.sum()),
        "boundary8_gt_pixels": int(boundary.sum()),
        "non_finite_target_pixels": 0,
        "geometry": geometry,
        "geometry_reason": geometry_reason,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


SCENE_EVALUATION_SCHEMA = "scene-coordinate-evaluation-1.1"
_EXPECTED_EVALUATION_ERRORS = (FlattenerError, OSError, ValueError)
_ANNOTATION_MODELS = {
    "authored": "known analytic synthetic correspondence",
    "simulated": "known cylindrical/homography synthetic correspondence",
    "capture": "manual-corner planar approximation; not dense measured GT",
}


def _scene_metric_definitions():
    return {
        "epe": "per-image mean source-pixel Euclidean xy error on fixed GT visible targets; "
        "no clipping or predicted masking; finite render failures remain included",
        "summary": "mean/median/p95 across per-image metrics, not pooled output pixels",
        "source_kind": "authored: native scenes; simulated: paper augmentation; "
        "capture: actual camera frames; keep capture and simulated metrics separate",
        "boundary8": "outermost 8 requested output pixels intersect fixed GT visibility",
        "page_relative_epe": "source-pixel EPE / GT source-page physical pixel-edge bbox "
        "diagonal hypot(x_max-x_min+1,y_max-y_min+1); reject width/height<2",
        "page_missed": "precision<0.5 OR coverage<0.5; incomplete metrics reported separately",
        "sampling_precision": "all requested output centers denominator; fraction in a GT "
        "source-page pixel; nonfinite and outside [-1,1) are misses",
        "coverage": "GT source-page pixel centers inside the half-pixel-linearly-extrapolated "
        "simple output perimeter / all GT source-page pixels; unavailable for "
        "nonfinite, folded, or nonsimple maps; extent diagnostic, not text fidelity",
        "iou": "same source-center footprint intersect GT page / footprint union GT page",
        "ground_truth_page_metrics": "identical diagnostics on analytic GT; incomplete GT "
        "maps have missing coverage, never assumed perfect",
        "renderer": "actual render, full requested target, no GT target_mask, default limits",
    }


def _extended_centers(coordinates):
    """Extrapolate center samples by half a pixel to the requested output edges."""
    padded = np.pad(coordinates.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    padded[0, 1:-1] = 1.5 * padded[1, 1:-1] - 0.5 * padded[2, 1:-1]
    padded[-1, 1:-1] = 1.5 * padded[-2, 1:-1] - 0.5 * padded[-3, 1:-1]
    padded[:, 0] = 1.5 * padded[:, 1] - 0.5 * padded[:, 2]
    padded[:, -1] = 1.5 * padded[:, -2] - 0.5 * padded[:, -3]
    return padded


def _simple_boundary(points):
    """Reject crossing or touching non-neighbor boundary segments, including overlaps."""
    ends = np.roll(points, -1, axis=0)
    delta = ends - points
    tolerance = 1e-12
    if np.any(np.linalg.norm(delta, axis=1) <= tolerance):
        return False
    for i in range(len(points) - 2):
        indexes = np.arange(i + 2, len(points) - (i == 0))
        a, b = points[i], ends[i]
        c, d = points[indexes], ends[indexes]
        overlap = np.all(
            np.maximum(np.minimum(a, b), np.minimum(c, d))
            <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + tolerance,
            axis=1,
        )
        if not overlap.any():
            continue
        c, d = c[overlap], d[overlap]

        def cross(first, second):
            return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

        ab_c, ab_d = cross(b - a, c - a), cross(b - a, d - a)
        cd_a, cd_b = cross(d - c, a - c), cross(d - c, b - c)
        if np.any((ab_c * ab_d <= tolerance) & (cd_a * cd_b <= tolerance)):
            return False
    return True


def _polygon_mask(points, size):
    """Rasterize the even-odd interior at source pixel centers with half-open edges."""
    height, width = size
    points = (points + 1) * np.asarray([width, height], dtype=np.float64) / 2
    ends = np.roll(points, -1, axis=0)
    result = np.zeros(size, dtype=bool)
    for row in range(height):
        y = row + 0.5
        crossing = (points[:, 1] <= y) != (ends[:, 1] <= y)
        start, stop = points[crossing], ends[crossing]
        x = np.sort(
            start[:, 0]
            + (y - start[:, 1]) * (stop[:, 0] - start[:, 0]) / (stop[:, 1] - start[:, 1])
        )
        if len(x) % 2:
            raise ValueError("boundary rasterization has an odd intersection count")
        # Clip intervals, never vertices: vertex clipping would change the polygon.
        bounds = np.ceil(np.clip(x - 0.5, 0, width)).astype(np.int64)
        for left, right in bounds.reshape(-1, 2):
            result[row, left:right] = True
    return result


def page_sampling_metrics(prediction: np.ndarray, source_mask: np.ndarray) -> dict[str, Any]:
    """Measure how the output samples the GT source page.

    Precision is the fraction of all output centers that land on a GT page pixel;
    nonfinite and outside centers count as misses. Coverage and IoU rasterize the
    output perimeter, extrapolated by half a pixel, at source centers. Coverage
    is reported only for finite, unfolded maps with a simple boundary.
    """
    if (
        not isinstance(prediction, np.ndarray)
        or prediction.ndim != 3
        or prediction.shape[-1] != 2
        or min(prediction.shape[:2]) < 2
        or not np.issubdtype(prediction.dtype, np.floating)
    ):
        raise ValueError("prediction must be a floating (height>=2, width>=2, 2) array")
    if (
        not isinstance(source_mask, np.ndarray)
        or source_mask.ndim != 2
        or source_mask.dtype != np.bool_
        or not source_mask.any()
    ):
        raise ValueError("source_mask must be a nonempty boolean GT page mask")
    finite = np.isfinite(prediction).all(axis=-1)
    inside = finite & (prediction >= -1).all(axis=-1) & (prediction < 1).all(axis=-1)
    height, width = source_mask.shape
    xy = np.floor((prediction[inside].astype(np.float64) + 1) * [width / 2, height / 2])
    xy = xy.astype(np.int64)
    # Float64 u < 1 can still round to exactly W; bound the pixel index only
    # after the inside test.
    xy = np.minimum(xy, [width - 1, height - 1])
    hits = int(source_mask[xy[:, 1], xy[:, 0]].sum())
    result = {
        "page_sampling_precision": hits / finite.size,
        "page_sampling_hits": hits,
        "page_sampling_denominator": finite.size,
        "non_finite_sampling_count": int((~finite).sum()),
        "out_of_source_sampling_count": int((finite & ~inside).sum()),
        "source_page_pixels": int(source_mask.sum()),
        "source_page_coverage": None,
        "source_page_iou": None,
        "coverage_status": "not_evaluated",
        "coverage_reason": None,
    }
    if not finite.all():
        result["coverage_reason"] = "non_finite_map"
        return result
    extended = _extended_centers(prediction)
    # Include extrapolated strips as well as original cells in the fold check.
    with np.errstate(over="ignore"):
        extended_float = extended.astype(np.float32)
    try:
        validate_geometry(
            extended_float,
            np.ones(extended.shape[:2], dtype=bool),
            RenderConfig(max_out_of_bounds_fraction=1),
        )
    except FlattenerError as error:
        # All centers can be outside while the polygon crosses or misses the source.
        if error.reason_code != "empty_valid_region":
            result["coverage_reason"] = error.reason_code
            return result
    boundary = np.concatenate(
        (extended[0], extended[1:, -1], extended[-1, -2::-1], extended[-2:0:-1, 0])
    )
    if not np.isfinite(boundary).all() or not _simple_boundary(boundary):
        result["coverage_reason"] = "non_simple_extrapolated_boundary"
        return result
    footprint = _polygon_mask(boundary, source_mask.shape)
    intersection = int((footprint & source_mask).sum())
    result.update(
        {
            "source_page_coverage": intersection / int(source_mask.sum()),
            "source_page_iou": intersection / int((footprint | source_mask).sum()),
            "footprint_pixels": int(footprint.sum()),
            "coverage_status": "evaluated",
        }
    )
    return result


def page_relative_endpoint_error(prediction, truth, gt_mask, source_mask):
    """Return EPE divided by the GT page bounding-box diagonal in source pixels.

    The box spans pixel edges: width = x_max - x_min + 1, height = y_max - y_min + 1.
    Masks narrower than two pixels on either axis raise ValueError.
    """
    if (
        not isinstance(source_mask, np.ndarray)
        or source_mask.dtype != np.bool_
        or source_mask.ndim != 2
        or not source_mask.any()
    ):
        raise ValueError("source_mask must be a nonempty boolean GT page mask")
    y, x = np.nonzero(source_mask)
    width, height = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
    if min(width, height) < 2:
        raise ValueError("GT page bounding box is degenerate")
    diagonal = float(np.hypot(width, height))
    return {
        "page_relative_epe": endpoint_error(prediction, truth, gt_mask, source_mask.shape)
        / diagonal,
        "source_page_bbox_diagonal_pixels": diagonal,
    }


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _scene_provenance():
    package = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(package)): _sha256(path) for path in sorted(package.rglob("*.py"))
    }
    lock = package.parent.parent / "uv.lock"
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "pillow": str(PIL.__version__),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "uv_lock_sha256": _sha256(lock) if lock.is_file() else None,
    }
    return {
        "source_code_sha256": sources,
        "source_sha256": _json_digest(sources),
        "runtime": runtime,
        "runtime_sha256": _json_digest(runtime),
    }


def _atomic_scene_report(path, report):
    payload = json.dumps(report, indent=2, allow_nan=False).encode() + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evaluation-", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _scene_summary(rows, records):
    values = [row["epe_pixels"] for row in rows if row.get("epe_pixels") is not None]
    result = {
        "expected_count": len(records),
        "processed_count": len(rows),
        "failed_count": sum(row["status"] != "completed" for row in rows),
        "unprocessed_count": len(records) - len(rows),
        "epe_missing_count": len(records) - len(values),
        "non_finite_failure_count": sum(row.get("non_finite_prediction", False) for row in rows),
        "renderer_success_count": sum(row.get("renderer_status") == "success" for row in rows),
        "renderer_failure_count": sum(
            row.get("renderer_status") not in (None, "success") for row in rows
        ),
        "renderer_missing_count": len(records)
        - sum(row.get("renderer_status") is not None for row in rows),
        "epe_pixels": _stats(values),
    }
    for metric in (
        "boundary8_epe_pixels",
        "page_relative_epe",
        "page_sampling_precision",
        "source_page_coverage",
        "source_page_iou",
    ):
        observed = [row[metric] for row in rows if row.get(metric) is not None]
        result[metric] = {**_stats(observed), "missing_count": len(records) - len(observed)}
    result["ground_truth_page_metrics"] = {}
    for metric in ("page_sampling_precision", "source_page_coverage", "source_page_iou"):
        observed = [
            row["ground_truth_page_metrics"][metric]
            for row in rows
            if row.get("ground_truth_page_metrics", {}).get(metric) is not None
        ]
        result["ground_truth_page_metrics"][metric] = {
            **_stats(observed),
            "missing_count": len(records) - len(observed),
        }
    result["failure_reasons"] = dict(
        Counter(error["reason"] for row in rows for error in row.get("errors", []))
    )
    page_rows = [
        row
        for row in rows
        if row.get("page_sampling_precision") is not None
        and row.get("source_page_coverage") is not None
    ]
    result["page_missed_count"] = sum(
        row["page_sampling_precision"] < 0.5 or row["source_page_coverage"] < 0.5
        for row in page_rows
    )
    result["page_missed_missing_count"] = len(records) - len(page_rows)
    result["page_metrics_missing_count"] = len(records) - len(page_rows)
    return result


def _record_source_kind(record):
    source = record.get("source", "authored")
    kind = record.get("source_kind", "simulated" if source == "paper" else "authored")
    allowed = ("capture", "simulated") if source == "paper" else ("authored",)
    if kind not in allowed:
        raise ValueError("record source_kind is incompatible with its source")
    return kind


def _scene_grouped_summary(rows, records):
    """Summarize rows overall, by size bin and by family for reports and selection."""
    summary = _scene_summary(rows, records)
    for field in ("size_bin", "family"):
        summary[f"by_{field}"] = {
            group: _scene_summary(
                [row for row in rows if row[field] == group],
                [record for record in records if record[field] == group],
            )
            for group in sorted({record[field] for record in records})
        }
    summary["by_source"] = {
        source: _scene_summary(
            [row for row in rows if row["source"] == source],
            [record for record in records if record.get("source", "authored") == source],
        )
        for source in sorted({record.get("source", "authored") for record in records})
    }
    for source, source_summary in summary["by_source"].items():
        selected = [row for row in rows if row["source"] == source]
        expected = [record for record in records if record.get("source", "authored") == source]
        source_summary["by_source_kind"] = {
            kind: _scene_summary(
                [row for row in selected if row["source_kind"] == kind],
                [record for record in expected if _record_source_kind(record) == kind],
            )
            for kind in sorted({_record_source_kind(record) for record in expected})
        }
        for kind, kind_summary in source_summary["by_source_kind"].items():
            kind_summary["annotation_model"] = _ANNOTATION_MODELS[kind]
        for field in ("size_bin", "family"):
            source_summary[f"by_{field}"] = {
                group: _scene_summary(
                    [row for row in selected if row[field] == group],
                    [record for record in expected if record[field] == group],
                )
                for group in sorted({record[field] for record in expected})
            }
    return summary


def _scene_case(dataset, index, predictor, render_config):
    record = dataset.records[index]
    row = {key: record[key] for key in ("id", "document_group", "size_bin", "family")}
    row["source"] = record.get("source", "authored")
    row["source_kind"] = _record_source_kind(record)
    row["annotation_model"] = _ANNOTATION_MODELS[row["source_kind"]]
    row.update(
        status="failed",
        epe_pixels=None,
        page_relative_epe=None,
        boundary8_epe_pixels=None,
        page_sampling_precision=None,
        source_page_coverage=None,
        source_page_iou=None,
        renderer_status=None,
        renderer_reason=None,
        errors=[],
        quality_status="not_evaluated",
    )

    def failure(stage, error):
        reason = error.reason_code if isinstance(error, FlattenerError) else type(error).__name__
        row["errors"].append({"stage": stage, "reason": reason, "message": str(error)})
        if "non_finite" in reason or "non-finite" in str(error):
            row["non_finite_prediction"] = True

    started = perf_counter()
    try:
        sample = dataset[index]
        if sample.get("source", row["source"]) != row["source"]:
            raise ValueError("sample source differs from its frozen record")
        if sample.get("source_kind", row["source_kind"]) != row["source_kind"]:
            raise ValueError("sample source_kind differs from its frozen record")
        prediction = predictor.predict_map(sample["original_image"], dataset.output_size)
    except _EXPECTED_EVALUATION_ERRORS as error:
        failure("load_or_prediction", error)
        row["seconds"] = perf_counter() - started
        return row
    truth, valid = sample["target"].numpy(), sample["valid"].numpy()
    try:
        row["epe_pixels"] = endpoint_error(prediction, truth, valid, dataset.photo_size)
        boundary = boundary_mask(valid)
        row["gt_pixels"] = int(valid.sum())
        row["boundary8_gt_pixels"] = int(boundary.sum())
        if boundary.any():
            row["boundary8_epe_pixels"] = endpoint_error(
                prediction, truth, boundary, dataset.photo_size
            )
        row.update(
            page_relative_endpoint_error(prediction, truth, valid, sample["source_mask"].numpy())
        )
    except _EXPECTED_EVALUATION_ERRORS as error:
        failure("coordinate_metrics", error)
    try:
        row.update(page_sampling_metrics(prediction, sample["source_mask"].numpy()))
        row["ground_truth_page_metrics"] = page_sampling_metrics(
            truth, sample["source_mask"].numpy()
        )
    except _EXPECTED_EVALUATION_ERRORS as error:
        failure("page_metrics", error)
    try:
        # Render the full target without the GT mask, as the public path does.
        rendered = render(sample["original_image"], prediction, config=render_config)
        row["renderer_status"] = str(rendered.status)
        row["renderer_reason"] = rendered.reason_code
        row["renderer_validation"] = rendered.metadata.get("validation", {})
        if rendered.status != "success":
            row["errors"].append(
                {"stage": "render", "reason": rendered.reason_code, "message": rendered.message}
            )
    except _EXPECTED_EVALUATION_ERRORS as error:
        failure("render", error)
    row["status"] = "completed" if not row["errors"] else "failed"
    row["seconds"] = perf_counter() - started
    return row


def evaluate_scene_checkpoint(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "validation",
    output_dir: str | Path,
    output_size: tuple[int, int] = (192, 144),
    include_paper_augmentation: bool = False,
    resume: bool = False,
    save_every: int = 16,
) -> dict[str, Any]:
    """Evaluate one coordinate-study checkpoint on one positive scene split.

    ``output_dir`` must be new unless ``resume`` continues the same protocol from
    an intact report. The checkpoint may come from another training manifest;
    the report records it. Paper augmentation is opt-in and reported per source.
    """
    from .coordinate_study import (
        _checkpoint_location,
        coordinate_checkpoint,
        load_coordinate_predictor,
    )
    from .scene_data import SceneCoordinateDataset

    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test; evaluation never trains")
    if type(include_paper_augmentation) is not bool:
        raise ValueError("include_paper_augmentation must be boolean")
    if type(resume) is not bool or type(save_every) is not int or not 1 <= save_every <= 1024:
        raise ValueError("resume must be boolean and save_every an integer in [1, 1024]")
    root, checkpoint, output = (
        Path(value).resolve() for value in (dataset_dir, checkpoint_path, output_dir)
    )
    checkpoint = _checkpoint_location(checkpoint, strict=True)
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("evaluation output must be separate from the dataset")
    if output.is_relative_to(checkpoint) or checkpoint.is_relative_to(output):
        raise ValueError("evaluation output must not contain the checkpoint")
    if not resume and output.exists():
        raise FileExistsError("fresh output directory required; use --resume for an identical run")
    if resume and (not output.is_dir() or not (output / "report.json").is_file()):
        raise ValueError("resume requires an existing evaluation report")
    provenance = _scene_provenance()
    checkpoint_digest = _sha256(checkpoint)
    saved = coordinate_checkpoint(checkpoint)
    predictor = load_coordinate_predictor(checkpoint)
    if not checkpoint_digest == _sha256(checkpoint) == predictor.model_sha256:
        raise ValueError("checkpoint changed while loading")
    dataset = SceneCoordinateDataset(
        root,
        split,
        predictor.model_config,
        output_size,
        include_paper_augmentation=include_paper_augmentation,
    )
    records = dataset.records
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("evaluation requires nonempty, uniquely identified cases")
    if any(record.get("source", "authored") not in ("authored", "paper") for record in records):
        raise ValueError("evaluation records require a recognized source")
    source_kind_counts = dict(Counter(_record_source_kind(record) for record in records))
    if not include_paper_augmentation and any(
        record.get("source") == "paper" for record in records
    ):
        raise ValueError("paper records require explicit include_paper_augmentation=True")
    render_config = RenderConfig(on_error="return")
    protocol = {
        "checkpoint_sha256": predictor.model_sha256,
        "checkpoint_schema": saved["schema_version"],
        "model_id": predictor.model_id,
        "checkpoint_checksum_verified": predictor.checksum_verified,
        "checkpoint_sidecar_sha256": _sha256(checkpoint.with_suffix(checkpoint.suffix + ".sha256")),
        "model_config": asdict(predictor.model_config),
        "training_manifest_sha256": saved.get("manifest_sha256"),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "training_manifest_matches_evaluation": saved.get("manifest_sha256")
        == dataset.manifest_sha256,
        "dataset_records_sha256": _json_digest(records),
        "split": split,
        "include_paper_augmentation": include_paper_augmentation,
        "source_counts": dict(Counter(record.get("source", "authored") for record in records)),
        "source_kind_counts": source_kind_counts,
        "annotation_models": {
            kind: _ANNOTATION_MODELS[kind] for kind in sorted(source_kind_counts)
        },
        "output_size": list(output_size),
        "source_size": list(dataset.photo_size),
        "render_config": asdict(render_config),
        "definitions": _scene_metric_definitions(),
        **provenance,
    }
    # Normalize tuples to their serialized form before strict resume comparisons.
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    digest = _json_digest(protocol)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / ".evaluation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    report_path = output / "report.json"
    report = None
    try:
        if resume:
            if report_path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("evaluation report exceeds 64 MiB")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("resume requires an evaluation report object")
            rows = report.get("samples")
            if (
                report.get("schema_version") != SCENE_EVALUATION_SCHEMA
                or report.get("quality_status") != "not_evaluated"
                or report.get("quality_claim_allowed") is not False
                or report.get("protocol_sha256") != digest
                or report.get("protocol") != protocol
                or not isinstance(rows, list)
                or len(rows) > len(records)
                or any(not isinstance(row, dict) for row in rows)
                or [row.get("id") for row in rows] != [r["id"] for r in records[: len(rows)]]
                or report.get("samples_sha256") != _json_digest(rows)
            ):
                raise ValueError(
                    "resume report/protocol/sample prefix differs from frozen evaluation"
                )
        else:
            report = {
                "schema_version": SCENE_EVALUATION_SCHEMA,
                "protocol": protocol,
                "protocol_sha256": digest,
                "dataset_path": str(root),
                "checkpoint_path": str(checkpoint),
                "started_at": datetime.now(UTC).isoformat(),
                "quality_status": "not_evaluated",
                "quality_claim_allowed": False,
                "evaluation_complete": False,
                "integrity_verified": False,
                "samples": [],
            }

        def save(state):
            report["state"] = state
            rows = report["samples"]
            report["samples_sha256"] = _json_digest(rows)
            report["summary"] = _scene_grouped_summary(rows, records)
            _atomic_scene_report(report_path, report)

        report["evaluation_complete"] = report["integrity_verified"] = False
        report["dataset_path"], report["checkpoint_path"] = str(root), str(checkpoint)
        report.pop("finished_at", None)
        save("running")
        try:
            for index in range(len(report["samples"]), len(records)):
                report["samples"].append(_scene_case(dataset, index, predictor, render_config))
                if len(report["samples"]) % save_every == 0:
                    save("running")
            # Re-run metadata/artifact hashing (no extra pixel decode) at completion.
            checked = SceneCoordinateDataset(
                root,
                split,
                predictor.model_config,
                output_size,
                include_paper_augmentation=include_paper_augmentation,
            )
            if (
                checked.manifest_sha256 != dataset.manifest_sha256
                or checked.records != records
                or _scene_provenance() != provenance
                or _sha256(checkpoint) != predictor.model_sha256
            ):
                raise ValueError(
                    "dataset, checkpoint, source, or runtime changed during evaluation"
                )
            # Also revalidate the mandatory checkpoint sidecar and checkpoint schema.
            coordinate_checkpoint(checkpoint)
            report["evaluation_complete"] = report["integrity_verified"] = True
            report["finished_at"] = datetime.now(UTC).isoformat()
            save("complete")
        except BaseException:
            # Preserve complete rows only. KeyboardInterrupt/SystemExit still propagate.
            save("interrupted")
            raise
        return report
    finally:
        lock.unlink(missing_ok=True)


SCENE_SELECTION_POLICY_ID = "scene-promotion-1.0"


def scene_selection_policy() -> dict[str, Any]:
    """Return a fresh copy of the fixed promotion policy."""
    return {
        "id": SCENE_SELECTION_POLICY_ID,
        "authored_epe_max_ratio": 1.10,
        "authored_small_epe_max_ratio": 1.10,
        "capture_epe_max_ratio": 0.90,
        "simulated_epe_max_ratio": 0.90,
        "renderer_failure_plus_missing": "nonincrease in every domain",
        "severe_page_plus_missing": "nonincrease in every domain",
        "page_score_max_drop": 0.005,
        "page_score_denominator": "all domain cases; missing precision/coverage score zero",
        "finite_complete": "every EPE/relative/boundary value and renderer status present; "
        "no nonfinite prediction; nullable page coverage is accounted for, not discarded",
        "ranking": ["mean capture/simulated EPE ratio", "maximum EPE ratio", "authored EPE",
                    "checkpoint SHA256", "candidate ID"],
        "zero_baseline_ratio": "0/0 is 0; positive/0 cannot pass the improvement gate",
        "split": "validation",
    }


@dataclass(frozen=True)
class SceneSelectionCandidate:
    """Checkpoint and report evidence for one selection candidate.

    Paths may name any copy with the declared SHA-256; aliases are resolved once.
    Selection hashes bytes and sidecars and never loads a model.
    """

    id: str
    checkpoint_path: str | Path
    checkpoint_sha256: str
    report_path: str | Path
    report_sha256: str


def _selection_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _selection_json(path, limit=64 * 1024 * 1024):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("selection evidence is missing or exceeds its byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("selection evidence exceeds its byte limit")
    value = json.loads(raw, object_pairs_hook=unique)
    _json_digest(value)  # Reject NaN/Infinity, including unused nested fields.
    return value, hashlib.sha256(raw).hexdigest()


_SELECTION_CASE_FIELDS = (
    "id", "document_group", "size_bin", "family", "source", "source_kind", "annotation_model",
)
_SELECTION_COMMON_FIELDS = (
    "dataset_manifest_sha256", "dataset_records_sha256", "split", "output_size", "source_size",
    "definitions", "source_sha256", "runtime_sha256", "render_config",
    "include_paper_augmentation", "source_counts", "source_kind_counts", "annotation_models",
)
_SELECTION_METRICS = (
    "epe_pixels", "page_relative_epe", "boundary8_epe_pixels", "page_sampling_precision",
    "source_page_coverage", "source_page_iou",
)


def _selection_evidence(candidate):
    from .coordinate_study import MAX_CHECKPOINT_BYTES, _checkpoint_location

    if (type(candidate) is not SceneSelectionCandidate
            or not isinstance(candidate.id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate.id) is None
            or not _selection_hash(candidate.checkpoint_sha256)
            or not _selection_hash(candidate.report_sha256)):
        raise ValueError("selection requires a named, SHA256-bound candidate")
    try:
        checkpoint = _checkpoint_location(Path(candidate.checkpoint_path).resolve(), strict=True)
        path = Path(candidate.report_path).resolve()
        sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
        if (not 0 < checkpoint.stat().st_size <= MAX_CHECKPOINT_BYTES
                or _sha256(checkpoint) != candidate.checkpoint_sha256
                or not 0 < sidecar.stat().st_size <= 256
                or sidecar.read_text(encoding="ascii").strip() != candidate.checkpoint_sha256):
            raise ValueError("candidate checkpoint or sidecar SHA differs")
        report, digest = _selection_json(path)
        if digest != candidate.report_sha256:
            raise ValueError("candidate report SHA differs")
        protocol, rows = report["protocol"], report["samples"]
        if (report["schema_version"] != SCENE_EVALUATION_SCHEMA
                or report["state"] != "complete"
                or report["evaluation_complete"] is not True
                or report["integrity_verified"] is not True
                or report["quality_status"] != "not_evaluated"
                or report["quality_claim_allowed"] is not False
                or protocol["split"] != "validation"
                or protocol["include_paper_augmentation"] is not True
                or protocol["checkpoint_checksum_verified"] is not True
                or protocol["checkpoint_sha256"] != candidate.checkpoint_sha256
                or protocol["checkpoint_sidecar_sha256"] != _sha256(sidecar)
                or report["protocol_sha256"] != _json_digest(protocol)
                or report["samples_sha256"] != _json_digest(rows)):
            raise ValueError("selection requires intact, complete validation-only evidence")
        if any(key not in protocol for key in _SELECTION_COMMON_FIELDS):
            raise ValueError("selection evidence omits a common evaluation contract")
        if protocol["definitions"] != _scene_metric_definitions():
            raise ValueError("selection evidence uses noncanonical metric definitions")
        for key in ("dataset_manifest_sha256", "dataset_records_sha256", "source_sha256",
                    "runtime_sha256"):
            if not _selection_hash(protocol[key]):
                raise ValueError("invalid evaluation identity hash")
        sources = protocol["source_code_sha256"]
        if (not isinstance(sources, dict) or not sources
                or any(not isinstance(key, str) or not _selection_hash(value)
                       for key, value in sources.items())
                or _json_digest(sources) != protocol["source_sha256"]
                or not _selection_runtime_valid(protocol["runtime"])
                or _json_digest(protocol["runtime"]) != protocol["runtime_sha256"]
                or _json_digest(protocol["render_config"])
                != _json_digest(asdict(RenderConfig(on_error="return")))):
            raise ValueError("evaluation source/runtime/renderer contract differs")
        for key in ("output_size", "source_size"):
            size = protocol[key]
            if (not isinstance(size, list) or len(size) != 2
                    or any(type(side) is not int or not 2 <= side <= 512 for side in size)):
                raise ValueError("invalid physical dimensions in evaluation contract")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100_000:
            raise ValueError("selection requires bounded nonempty cases")
        for row in rows:
            if (any(not isinstance(row[key], str) or not row[key] for key in _SELECTION_CASE_FIELDS)
                    or row["source"] not in ("authored", "paper")
                    or row["size_bin"] not in ("small", "medium", "large")
                    or row["status"] not in ("completed", "failed")
                    or row["quality_status"] != "not_evaluated"
                    or row["source_kind"] != _record_source_kind(row)
                    or row["annotation_model"] != _ANNOTATION_MODELS[row["source_kind"]]
                    or type(row.get("non_finite_prediction", False)) is not bool
                    or row["renderer_status"] not in (
                        None, "success", "input_error", "invalid_transform", "model_error",
                        "render_error", "io_error")
                    or not isinstance(row["errors"], list)
                    or any(not isinstance(error, dict) or not isinstance(error.get("reason"), str)
                           for error in row["errors"])
                    or (row["status"] == "completed") != (not row["errors"])):
                raise ValueError("invalid case identity/status in selection evidence")
            for metric in _SELECTION_METRICS:
                value = row[metric]
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or metric in _SELECTION_METRICS[3:] and value > 1
                ):
                    raise ValueError("invalid numeric metric in selection evidence")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate selection cases")
        for field, key in (("source", "source_counts"), ("source_kind", "source_kind_counts")):
            counts = protocol[key]
            if (not isinstance(counts, dict)
                    or any(type(count) is not int or count < 1 for count in counts.values())
                    or counts != dict(Counter(row[field] for row in rows))):
                raise ValueError("selection case denominators differ from the frozen protocol")
        if (set(protocol["source_kind_counts"]) != {"authored", "capture", "simulated"}
                or protocol["annotation_models"] != _ANNOTATION_MODELS
                or not any(row["source_kind"] == "authored" and row["size_bin"] == "small"
                           for row in rows)):
            raise ValueError("policy requires all three domains and authored-small cases")
        summary = _scene_grouped_summary(rows, rows)
        if _json_digest(summary) != _json_digest(report["summary"]):
            raise ValueError("selection report summary differs from its case rows")
        reference = {
            "id": candidate.id, "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": candidate.checkpoint_sha256, "report_path": str(path),
            "report_sha256": candidate.report_sha256,
            "checkpoint_sidecar_sha256": protocol["checkpoint_sidecar_sha256"],
        }
        return reference, report
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError(f"malformed selection evidence for {candidate.id}") from error


def _selection_runtime_valid(runtime):
    if not isinstance(runtime, dict):
        return False
    text = ("python", "platform", "machine", "torch", "numpy", "pillow")
    flags = ("deterministic", "deterministic_warn_only", "mkldnn_enabled")
    return (
        all(isinstance(runtime.get(key), str) and runtime[key] for key in text)
        and runtime.get("device") == "cpu"
        and all(type(runtime.get(key)) is bool for key in flags)
        and all(type(runtime.get(key)) is int and 1 <= runtime[key] <= 4096
                for key in ("torch_threads", "torch_interop_threads"))
        and runtime.get("matmul_precision") in ("highest", "high", "medium")
        and "uv_lock_sha256" in runtime
        and (runtime["uv_lock_sha256"] is None or _selection_hash(runtime["uv_lock_sha256"]))
    )


def _selection_scores(rows):
    summary = _scene_summary(rows, rows)
    summary["fixed_denominator_page_scores"] = {
        metric: math.fsum(row[metric] if row[metric] is not None else 0.0 for row in rows)
        / len(rows)
        for metric in ("source_page_coverage", "page_sampling_precision")
    }
    return summary


def _selection_diagnostics(report):
    rows = report["samples"]
    domains, documents = {}, {}
    for kind in ("authored", "capture", "simulated"):
        selected = [row for row in rows if row["source_kind"] == kind]
        domains[kind] = _selection_scores(selected)
        groups = {
            group: _selection_scores([row for row in selected if row["document_group"] == group])
            for group in sorted({row["document_group"] for row in selected})}
        complete = [value["epe_pixels"]["mean"] for value in groups.values()
                    if value["epe_missing_count"] == 0]
        documents[kind] = {
            "document_count": len(groups), "incomplete_document_count": len(groups) - len(complete),
            "epe_pixels": (math.fsum(complete) / len(groups)
                           if len(complete) == len(groups) else None),
            "fixed_denominator_page_scores": {
                metric: math.fsum(value["fixed_denominator_page_scores"][metric]
                                  for value in groups.values()) / len(groups)
                for metric in ("source_page_coverage", "page_sampling_precision")},
            "by_document": groups,
        }
    finite_complete = all(
        value["epe_missing_count"] == value["renderer_missing_count"]
        == value["non_finite_failure_count"] == 0
        and all(value[metric]["count"] == value["expected_count"]
                for metric in ("epe_pixels", "page_relative_epe", "boundary8_epe_pixels"))
        for value in domains.values()
    )
    return {"finite_complete": finite_complete, "domains": domains,
            "authored_small_epe": report["summary"]["by_source"]["authored"]
            ["by_size_bin"]["small"]["epe_pixels"]["mean"], "document_macro": documents}


def _selection_gates(new, old):
    if not new["finite_complete"]:
        return {"eligible": False}
    left, right = new["domains"], old["domains"]
    gates = {
        "eligible": True,
        "authored_epe_within_10_percent": left["authored"]["epe_pixels"]["mean"]
        <= 1.1 * right["authored"]["epe_pixels"]["mean"],
        "authored_small_epe_within_10_percent": new["authored_small_epe"]
        <= 1.1 * old["authored_small_epe"],
    }
    for kind in ("capture", "simulated"):
        gates[kind + "_epe_improved_at_least_10_percent"] = (
            left[kind]["epe_pixels"]["mean"] <= .9 * right[kind]["epe_pixels"]["mean"])
    for kind, value in left.items():
        previous = right[kind]
        gates[kind + "_renderer_failures_not_increased"] = (
            value["renderer_failure_count"] + value["renderer_missing_count"]
            <= previous["renderer_failure_count"] + previous["renderer_missing_count"])
        gates[kind + "_severe_page_errors_not_increased"] = (
            value["page_missed_count"] + value["page_metrics_missing_count"]
            <= previous["page_missed_count"] + previous["page_metrics_missing_count"])
        for metric in ("source_page_coverage", "page_sampling_precision"):
            gates[kind + "_" + metric + "_within_half_percentage_point"] = (
                value["fixed_denominator_page_scores"][metric]
                >= previous["fixed_denominator_page_scores"][metric] - .005)
    return gates


def select_scene_checkpoints(
    baseline: SceneSelectionCandidate,
    candidates: list[SceneSelectionCandidate] | tuple[SceneSelectionCandidate, ...],
    *,
    policy_id: str = SCENE_SELECTION_POLICY_ID,
    policy_sha256: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pick the best eligible candidate from frozen validation reports.

    Reads only report and checkpoint bytes. Corrupt or incompatible evidence
    raises; quality failures stay in the candidate table. With no passing
    candidate, ``best_eligible`` is None. Provenance is compared across reports.
    ``output_path`` must be a new path; it is written atomically.
    """
    policy = scene_selection_policy()
    policy_digest = _json_digest(policy)
    if policy_id != policy["id"] or policy_sha256 not in (None, policy_digest):
        raise ValueError("unknown or changed scene selection policy")
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 1000:
        raise ValueError("selection requires 1..1000 explicitly frozen candidates")
    references = [baseline, *candidates]
    if (any(type(value) is not SceneSelectionCandidate for value in references)
            or any(not isinstance(value.id, str) for value in references)
            or len({value.id for value in references}) != len(references)):
        raise ValueError("selection candidate and baseline IDs must be distinct")
    baseline_ref, baseline_report = _selection_evidence(baseline)
    admitted = [baseline_ref]
    old = _selection_diagnostics(baseline_report)
    if not old["finite_complete"]:
        raise ValueError("baseline requires finite complete coordinate/renderer evidence")
    common = {key: baseline_report["protocol"][key] for key in _SELECTION_COMMON_FIELDS}
    identities = [tuple(row[key] for key in _SELECTION_CASE_FIELDS)
                  for row in baseline_report["samples"]]
    entries = []
    for candidate in candidates:
        reference, report = _selection_evidence(candidate)
        admitted.append(reference)
        if (any(report["protocol"][key] != value for key, value in common.items())
                or [tuple(row[key] for key in _SELECTION_CASE_FIELDS)
                    for row in report["samples"]] != identities):
            raise ValueError("candidate differs from frozen baseline evaluation protocol/cases")
        diagnostics = _selection_diagnostics(report)
        gates = _selection_gates(diagnostics, old)
        eligible = all(gates.values())
        rank = None
        if eligible:
            ratios = [diagnostics["domains"][kind]["epe_pixels"]["mean"]
                      / old["domains"][kind]["epe_pixels"]["mean"]
                      if old["domains"][kind]["epe_pixels"]["mean"] else 0.0
                      for kind in ("capture", "simulated")]
            rank = [math.fsum(ratios) / 2, max(ratios),
                    diagnostics["domains"]["authored"]["epe_pixels"]["mean"],
                    reference["checkpoint_sha256"], reference["id"]]
        entries.append({"candidate": reference, "eligible": eligible, "gates": gates,
                        "rank": rank, "diagnostics": diagnostics})
    passing = [entry for entry in entries if entry["eligible"]]
    winner = min(passing, key=lambda entry: entry["rank"]) if passing else None
    result = {
        "schema_version": "scene-checkpoint-selection-1.0", "complete": True,
        "policy": policy, "policy_sha256": policy_digest,
        "baseline": {"candidate": baseline_ref, "diagnostics": old},
        "candidates": sorted(entries, key=lambda entry: entry["candidate"]["id"]),
        "best_eligible": winner["candidate"] if winner else None,
        "quality_status": "not_evaluated", "quality_claim_allowed": False,
        "test_evaluated": False, "checkpoint_modified": False,
        "common_evaluation_protocol_sha256": _json_digest(common),
        "selector_source_sha256": _sha256(Path(__file__)),
        "limitations": ["Selection reuses development validation; it is not an independent test.",
                        "Document macros weight document groups equally and do not alter gates.",
                        "Evidence hashes checked; original dataset artifacts not revalidated.",
                        "No checkpoint is installed; best.pt and selected.pt remain unchanged."],
    }
    # Recheck the resolved immutable evidence, independent of later alias switches.
    for reference in admitted:
        checkpoint = Path(reference["checkpoint_path"])
        if (_sha256(checkpoint) != reference["checkpoint_sha256"]
                or _sha256(Path(reference["report_path"])) != reference["report_sha256"]
                or _sha256(checkpoint.with_suffix(checkpoint.suffix + ".sha256"))
                != reference["checkpoint_sidecar_sha256"]):
            raise ValueError("selection evidence changed during verification")
    if output_path is not None:
        output = Path(output_path).absolute()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".selection-",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(result, indent=2, allow_nan=False).encode() + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, output)  # Atomic and exclusive; never clobber a prior selection.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result


def _selection_cli(argv):
    parser = argparse.ArgumentParser(description="Select from frozen formal validation reports")
    parser.add_argument("--spec", type=Path, required=True,
                        help="scene-selection-spec-1.0 JSON; paths relative to this file")
    parser.add_argument("--output", type=Path, required=True, help="new selection JSON file")
    args = parser.parse_args(argv)
    try:
        spec, _ = _selection_json(args.spec, limit=1024 * 1024)
        if (spec.keys() != {
            "schema_version", "policy_id", "policy_sha256", "baseline", "candidates"}
                or spec["schema_version"] != "scene-selection-spec-1.0"
                or not _selection_hash(spec["policy_sha256"])):
            raise ValueError("invalid frozen selection specification")

        def candidate(value):
            value = dict(value)
            for key in ("checkpoint_path", "report_path"):
                value[key] = args.spec.resolve().parent / value[key]
            return SceneSelectionCandidate(**value)

        result = select_scene_checkpoints(
            candidate(spec["baseline"]), [candidate(value) for value in spec["candidates"]],
            policy_id=spec["policy_id"], policy_sha256=spec["policy_sha256"],
            output_path=args.output,
        )
    except (*_EXPECTED_EVALUATION_ERRORS, TypeError, KeyError, AttributeError) as error:
        print(json.dumps({"error": str(error), "quality_status": "not_evaluated"}))
        return 2
    print(json.dumps({"selection": str(args.output.resolve()),
                      "best_eligible": result["best_eligible"],
                      "quality_status": "not_evaluated"}))
    return 0 if result["best_eligible"] is not None else 1


def main(argv: list[str] | None = None) -> int:
    """Evaluate a fixed checkpoint on a synthetic scene split."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["select"]:
        return _selection_cli(arguments[1:])
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh report directory")
    parser.add_argument("--output-size", type=int, nargs=2, metavar=("H", "W"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include-paper-augmentation",
        action="store_true",
        help="explicitly include paper augmentation and report per-source metrics",
    )
    args = parser.parse_args(arguments)
    try:
        report = evaluate_scene_checkpoint(
            args.dataset,
            args.checkpoint,
            split=args.split,
            output_dir=args.output,
            output_size=tuple(args.output_size),
            include_paper_augmentation=args.include_paper_augmentation,
            resume=args.resume,
        )
    except _EXPECTED_EVALUATION_ERRORS as error:
        print(json.dumps({"error": str(error), "quality_status": "not_evaluated"}))
        return 2
    print(
        json.dumps(
            {
                "report": str(args.output.resolve() / "report.json"),
                "evaluation_complete": report["evaluation_complete"],
                "quality_status": report["quality_status"],
                "summary": report["summary"],
            }
        )
    )
    return 0 if report["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
