"""Independent source-grid page-region metrics with complete denominators.

Ground truth never changes in response to a predicted mask. Bounding boxes use
source-pixel edges, unlike the pixel-center coordinates in backward maps.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import FlattenerError, Status

FOREGROUND_IOU_GATE = 0.80
NEGATIVE_FALSE_POSITIVE_GATE = 0.10
METHODS = ("learned", "all_foreground", "center_box")


def _mask(mask: np.ndarray, name: str) -> None:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.ndim != 2
        or min(mask.shape) < 1
    ):
        raise ValueError(f"{name} must be a nonempty two-dimensional boolean array")


def _size(size: tuple[int, int], name: str) -> None:
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in size)
    ):
        raise ValueError(f"{name} must be positive integer (height, width)")


def _box(box: Sequence[float], source_size: tuple[int, int]) -> np.ndarray:
    if not isinstance(box, (tuple, list, np.ndarray)):
        raise ValueError("crop_box must contain four source-pixel edges")
    try:
        result = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("crop_box must contain four finite source-pixel edges") from exc
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("crop_box must contain four finite source-pixel edges")
    left, top, right, bottom = result
    height, width = source_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("crop_box must have positive area and remain inside the source image")
    return result


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the minimal source-edge box of a boolean mask, or None if empty."""
    _mask(mask, "mask")
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1


def rescale_box(
    box: Sequence[float], source_size: tuple[int, int], destination_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Scale edge coordinates; the pixel-center half-pixel offset does not apply."""
    _size(source_size, "source_size")
    _size(destination_size, "destination_size")
    edges = _box(box, source_size)
    scale = np.asarray([destination_size[1] / source_size[1], destination_size[0] / source_size[0]])
    return tuple(float(v) for v in edges * np.tile(scale, 2))


def crop_metrics(truth: np.ndarray, crop_box: Sequence[float] | None) -> dict[str, Any]:
    """Measure source-pixel area retained and excess crop relative to fixed GT.

    Fractional crop edges retain the corresponding fractional area of each GT
    pixel. Negative images have no GT-box/retention ratio: those values are None.
    """
    _mask(truth, "truth")
    ground_box = mask_box(truth)
    predicted_box = None if crop_box is None else _box(crop_box, truth.shape)
    crop_area = (
        0.0 if predicted_box is None else float(np.prod(predicted_box[2:] - predicted_box[:2]))
    )
    result = {
        "crop_box": None if predicted_box is None else predicted_box.tolist(),
        "gt_box": None if ground_box is None else list(ground_box),
        "crop_area_pixels": crop_area,
        "crop_image_fraction": crop_area / truth.size,
        "gt_box_area_pixels": None,
        "gt_box_contained": None,
        "gt_box_area_retained_fraction": None,
        "gt_foreground_area_retained_fraction": None,
        "excess_crop_area_over_gt_box": None,
        "crop_area_over_gt_box": None,
    }
    if ground_box is None:
        return result
    gt = np.asarray(ground_box, dtype=np.float64)
    gt_area = float(np.prod(gt[2:] - gt[:2]))
    intersection = retained = 0.0
    contained = False
    if predicted_box is not None:
        intersection = float(
            np.prod(
                np.maximum(
                    0.0,
                    np.minimum(gt[2:], predicted_box[2:]) - np.maximum(gt[:2], predicted_box[:2]),
                )
            )
        )
        contained = bool(
            np.all(predicted_box[:2] <= gt[:2]) and np.all(predicted_box[2:] >= gt[2:])
        )
        rows, columns = np.nonzero(truth)
        retained = float(
            (
                np.maximum(
                    0.0,
                    np.minimum(columns + 1, predicted_box[2])
                    - np.maximum(columns, predicted_box[0]),
                )
                * np.maximum(
                    0.0, np.minimum(rows + 1, predicted_box[3]) - np.maximum(rows, predicted_box[1])
                )
            ).sum()
        )
    result.update(
        gt_box_area_pixels=gt_area,
        gt_box_contained=contained,
        gt_box_area_retained_fraction=intersection / gt_area,
        gt_foreground_area_retained_fraction=retained / int(truth.sum()),
        excess_crop_area_over_gt_box=(crop_area - intersection) / gt_area,
        crop_area_over_gt_box=crop_area / gt_area,
    )
    return result


def mask_metrics(
    prediction: np.ndarray, truth: np.ndarray, *, crop_box: Sequence[float] | None = None
) -> dict[str, Any]:
    """Pixel confusion/overlap and tight predicted-box crop diagnostics.

    Empty/empty IoU and precision are one by convention; foreground-only IoU
    aggregation explicitly excludes negative images. Positive GT with an empty
    prediction has precision zero, recall zero, and IoU zero. If crop_box is None,
    the tight predicted-mask box is measured (also None for an empty prediction).
    """
    _mask(truth, "truth")
    _mask(prediction, "prediction")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have equal source-grid shapes")
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = truth.size - tp - fp - fn
    gt_count, predicted_count = tp + fn, tp + fp
    union = tp + fp + fn
    return {
        "source_size": list(truth.shape),
        "source_pixels": int(truth.size),
        "gt_pixels": gt_count,
        "predicted_pixels": predicted_count,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "has_page": gt_count > 0,
        "detected_page": predicted_count > 0,
        "iou": tp / union if union else 1.0,
        "precision": tp / predicted_count if predicted_count else float(gt_count == 0),
        "recall": tp / gt_count if gt_count else 1.0,
        "negative_pixel_false_positive_rate": fp / truth.size if gt_count == 0 else None,
        **crop_metrics(truth, mask_box(prediction) if crop_box is None else crop_box),
    }


def page_size_category(truth: np.ndarray) -> str:
    """Fixed GT-area categories independent of prediction or model selection."""
    _mask(truth, "truth")
    fraction = float(truth.mean())
    return (
        "negative"
        if fraction == 0
        else "small"
        if fraction < 0.25
        else "medium"
        if fraction < 0.5
        else "large"
    )


@dataclass(frozen=True)
class RegionCase:
    sample_id: str
    truth: np.ndarray
    background_type: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaskPrediction:
    mask: np.ndarray
    crop_box: Sequence[float] | None = None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["has_page"]]
    negatives = [row for row in rows if not row["has_page"]]
    completed = [row for row in rows if row["status"] == "completed"]
    completed_positive = [row for row in positives if row["status"] == "completed"]
    iou = _stats([row.get("iou", 0.0) for row in positives])
    false_positive_count = sum(
        row["status"] != "completed" or row["detected_page"] for row in negatives
    )
    false_positive_rate = false_positive_count / len(negatives) if negatives else None
    return {
        "expected_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(rows) - len(completed),
        "foreground_count": len(positives),
        "negative_count": len(negatives),
        "foreground_iou_all_cases": iou,
        "foreground_iou_completed_only": _stats([row["iou"] for row in completed_positive]),
        "foreground_precision_completed_only": _stats(
            [row["precision"] for row in completed_positive]
        ),
        "foreground_recall_completed_only": _stats([row["recall"] for row in completed_positive]),
        "negative_false_positive_count_conservative": false_positive_count,
        "negative_false_positive_rate_conservative": false_positive_rate,
        "negative_detection_count_completed_only": sum(
            row.get("detected_page", False) for row in negatives if row["status"] == "completed"
        ),
        "gt_box_containment_count": sum(row["gt_box_contained"] for row in completed_positive),
        "gt_box_containment_rate_all_foreground": (
            sum(row["gt_box_contained"] for row in completed_positive) / len(positives)
            if positives
            else None
        ),
        "gt_foreground_area_retained_fraction_completed_only": _stats(
            [row["gt_foreground_area_retained_fraction"] for row in completed_positive]
        ),
        "excess_crop_area_over_gt_box_completed_only": _stats(
            [row["excess_crop_area_over_gt_box"] for row in completed_positive]
        ),
        "gate_passed": bool(
            rows
            and len(completed) == len(rows)
            and positives
            and negatives
            and iou["mean"] >= FOREGROUND_IOU_GATE
            and false_positive_rate <= NEGATIVE_FALSE_POSITIVE_GATE
        ),
    }


def evaluate_masks(
    cases: Sequence[RegionCase],
    predictions: Mapping[str, np.ndarray | MaskPrediction | None],
    *,
    method: str = "learned",
) -> dict[str, Any]:
    """Evaluate every expected case; malformed/missing predictions are failures.

    Malformed GT is a dataset error and raises. Extra prediction IDs raise rather
    than silently evaluating a mismatched collection. Expected failures cannot
    disappear from foreground or negative gate denominators.
    """
    if not isinstance(cases, Sequence) or not cases:
        raise ValueError("cases must be a nonempty sequence")
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must map sample IDs to masks")
    identifiers = set()
    for case in cases:
        if (
            not isinstance(case, RegionCase)
            or not isinstance(case.sample_id, str)
            or not case.sample_id
        ):
            raise ValueError("each case requires a nonempty sample ID")
        if case.sample_id in identifiers:
            raise ValueError("duplicate sample ID")
        identifiers.add(case.sample_id)
        _mask(case.truth, "truth")
        if not isinstance(case.background_type, str) or not case.background_type:
            raise ValueError("background_type must be a nonempty string")
    if set(predictions) - identifiers:
        raise ValueError("predictions contain IDs outside the expected cases")
    rows = []
    for case in cases:
        row = {
            "id": case.sample_id,
            "background_type": case.background_type,
            "page_size_category": page_size_category(case.truth),
            "has_page": bool(case.truth.any()),
            "gt_pixels": int(case.truth.sum()),
            "source_pixels": int(case.truth.size),
            "metadata": case.metadata,
        }
        prediction = predictions.get(case.sample_id)
        if prediction is None:
            row.update(status="failed", reason="missing_prediction")
        else:
            try:
                result = (
                    mask_metrics(prediction.mask, case.truth, crop_box=prediction.crop_box)
                    if isinstance(prediction, MaskPrediction)
                    else mask_metrics(prediction, case.truth)
                )
            except (TypeError, ValueError) as exc:
                row.update(status="failed", reason="invalid_prediction", message=str(exc))
            else:
                row.update(status="completed", **result)
        rows.append(row)
    return {
        "schema_version": "region-evaluation-0.1",
        "method": method,
        "evaluation_grid": "original_source_pixels",
        "bbox_convention": "source_pixel_edges_xyxy_exclusive_right_bottom",
        "empty_empty_iou": 1.0,
        "thresholds": {
            "foreground_iou_min": FOREGROUND_IOU_GATE,
            "negative_image_false_positive_rate_max": NEGATIVE_FALSE_POSITIVE_GATE,
        },
        "summary": _aggregate(rows),
        "by_background": {
            name: _aggregate([row for row in rows if row["background_type"] == name])
            for name in sorted({row["background_type"] for row in rows})
        },
        "by_page_size": {
            name: _aggregate([row for row in rows if row["page_size_category"] == name])
            for name in sorted({row["page_size_category"] for row in rows})
        },
        "samples": rows,
    }


def evaluate_baselines(cases: Sequence[RegionCase]) -> dict[str, dict[str, Any]]:
    """Fixed all-foreground and middle-half-width/height masks, including negatives."""
    foreground, centers = {}, {}
    for case in cases:
        _mask(case.truth, "truth")
        height, width = case.truth.shape
        foreground[case.sample_id] = np.ones_like(case.truth)
        # Select centers in the fixed continuous central half of each axis.
        ys, xs = np.indices(case.truth.shape)
        centers[case.sample_id] = (
            (xs + 0.5 >= width * 0.25)
            & (xs + 0.5 < width * 0.75)
            & (ys + 0.5 >= height * 0.25)
            & (ys + 0.5 < height * 0.75)
        )
    return {
        "all_foreground": evaluate_masks(cases, foreground, method="all_foreground"),
        "center_box": evaluate_masks(cases, centers, method="center_box"),
    }


def mask_to_source(mask: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    """Lift full-extent grid cells to source-pixel centers independently of Torch.

    The source pixel center ``x + .5`` maps to cell
    ``floor((x + .5) * grid_width / source_width)``. This is nearest-exact
    half-pixel sampling, not align-corners interpolation or top-left sampling.
    """
    _mask(mask, "mask")
    _size(source_size, "source_size")
    rows = np.floor((np.arange(source_size[0]) + 0.5) * mask.shape[0] / source_size[0]).astype(int)
    columns = np.floor((np.arange(source_size[1]) + 0.5) * mask.shape[1] / source_size[1]).astype(
        int
    )
    return mask[np.ix_(rows, columns)].copy()


def evaluate_probabilities(
    cases: Sequence[RegionCase],
    probabilities: Mapping[str, np.ndarray | None],
    *,
    config=None,
) -> dict[str, Any]:
    """Compare raw threshold and fixed component-filtered source-grid predictions.

    An intentional detector rejection predicts an empty mask and remains a
    completed case: positives score zero and negatives predict no detection.
    Invalid input or missing inference remains a failed case in both summaries.
    """
    from .geometry import RegionConfig, detect_region

    config = RegionConfig() if config is None else config
    if not isinstance(config, RegionConfig):
        raise ValueError("config must be a RegionConfig")
    # Validate the expected case collection before invoking any predictions.
    evaluate_masks(cases, {})
    if not isinstance(probabilities, Mapping):
        raise ValueError("probabilities must map sample IDs to arrays")
    if set(probabilities) - {case.sample_id for case in cases}:
        raise ValueError("probabilities contain IDs outside the expected cases")
    raw, postprocessed, diagnostics = {}, {}, []
    rejection_reasons = {
        "page_not_found",
        "page_area_rejected",
        "ambiguous_page_region",
        "uncertain_page_region",
    }
    for case in cases:
        probability = probabilities.get(case.sample_id)
        diagnostic = {"id": case.sample_id}
        if probability is None:
            diagnostic.update(status="failed", reason="missing_prediction")
        elif (
            not isinstance(probability, np.ndarray)
            or probability.ndim != 2
            or probability.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not 0 < probability.size <= 512 * 512
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or np.any(probability > 1)
        ):
            diagnostic.update(status="failed", reason="invalid_probability")
        else:
            raw[case.sample_id] = mask_to_source(probability >= config.threshold, case.truth.shape)
            try:
                estimate = detect_region(probability, case.truth.shape, config=config)
            except FlattenerError as exc:
                if exc.status != Status.MODEL_ERROR or exc.reason_code not in rejection_reasons:
                    diagnostic.update(status="failed", reason=exc.reason_code, message=str(exc))
                else:
                    postprocessed[case.sample_id] = np.zeros_like(case.truth)
                    diagnostic.update(
                        status="rejected", reason=exc.reason_code, metadata=exc.metadata
                    )
            else:
                region = estimate.region
                postprocessed[case.sample_id] = MaskPrediction(
                    mask_to_source(estimate.mask, case.truth.shape),
                    (region.left, region.top, region.right, region.bottom),
                )
                diagnostic.update(status="accepted", metadata=estimate.metadata)
        diagnostics.append(diagnostic)
    reports = evaluate_baselines(cases)
    reports["learned_raw_threshold"] = evaluate_masks(cases, raw, method="learned_raw_threshold")
    reports["learned_postprocessed"] = evaluate_masks(
        cases, postprocessed, method="learned_postprocessed"
    )
    return {
        "schema_version": "region-probability-evaluation-0.1",
        "postprocessor": asdict(config),
        "score_is_calibrated_confidence": False,
        "gate_method": "learned_postprocessed",
        "gate_passed": reports["learned_postprocessed"]["summary"]["gate_passed"],
        "region_accepted_count": sum(row["status"] == "accepted" for row in diagnostics),
        "region_rejected_count": sum(row["status"] == "rejected" for row in diagnostics),
        "region_failure_count": sum(row["status"] == "failed" for row in diagnostics),
        "methods": reports,
        "region_decisions": diagnostics,
    }


def evaluate_region_checkpoint(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "validation",
    config=None,
    output_dir: str | Path | None = None,
    threads: int = 1,
) -> dict[str, Any]:
    """Evaluate one selected split; no other image or label is decoded.

    Dataset validation checks provenance and file hashes for all splits without
    decoding images or masks. A matching training-manifest hash is required.
    The report is exclusive-create: previous experiment evidence is not replaced.
    """
    import torch

    from ..synthetic.scenes import load_scene, validate_scene_dataset
    from .segmentation import load_segmenter

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError("threads must be a positive integer")
    directory = Path(dataset_dir).resolve()
    validated = validate_scene_dataset(directory)
    manifest_sha = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    loaded = load_segmenter(checkpoint_path)
    if loaded.metadata["manifest_sha256"] != manifest_sha:
        raise ValueError("checkpoint training manifest does not match the evaluation dataset")
    records = [record for record in validated["manifest"]["samples"] if record["split"] == split]
    if not records:
        raise ValueError("the selected evaluation split is empty")
    cases, probabilities, inference_errors = [], {}, []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        for record in records:
            sample = load_scene(directory, record)
            cases.append(
                RegionCase(
                    record["id"],
                    sample.mask,
                    record["background_type"],
                    {
                        "document_group": record["document_group"],
                        "capture_session": record["capture_session"],
                    },
                )
            )
            try:
                probabilities[record["id"]] = loaded.predict_probability(sample.image)
            except (FlattenerError, TypeError, ValueError) as exc:
                inference_errors.append({"id": record["id"], "message": str(exc)})
                probabilities[record["id"]] = None
        report = evaluate_probabilities(cases, probabilities, config=config)
    finally:
        torch.set_num_threads(previous_threads)
    report.update(
        split=split,
        checkpoint_sha256=loaded.metadata["checkpoint_sha256"],
        manifest_sha256=manifest_sha,
        checkpoint_epoch=loaded.metadata["epoch"],
        expected_count=len(records),
        cpu_threads=int(threads),
        inference_errors=inference_errors,
        test_used_for_model_selection=False,
        ground_truth_grid="original source pixels, never resized for evaluation",
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / f"region-evaluation-{split}.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return report
