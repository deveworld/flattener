"""Write-once validation choice for the preregistered region architecture study.

Only explicitly supplied validation/training reports, checkpoints and protocol
bytes are read. No dataset loader, test report discovery or inference is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..contracts import RenderConfig
from ..learning.inference import load_predictor
from .checkpoints import STUDY_SCHEMA, model_spec, read_region_checkpoint
from .geometry import RegionConfig
from .segmentation import CHECKPOINT_SCHEMA as LEGACY_SCHEMA
from .segmentation import SegmenterConfig, sha256
from .training import LOSS_PROTOCOL, SELECTION_PROTOCOL

LEGACY_CHECKPOINT_SHA256 = "fcab597654271c897a5adc3162ae77d7d81bdc9a5ab4a96f4026a7954bbddba6"
STUDY_MANIFEST_SHA256 = "06d3c61c360a4db2818516296e20153cc9ec98ee218704a7e1f133cf3e8d8f19"
COORDINATE_CHECKPOINT_SHA256 = "dc8914d65b913bee7611870a1ab031414609e6bc217344f756336397f97e1a15"
LEGACY_SELECTED_EPOCH = 77
EXPECTED_COUNTS = {"train": 576, "validation": 144, "test": 144}
GATE_THRESHOLDS = {"foreground_iou_min": 0.8, "negative_image_false_positive_rate_max": 0.1}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_file(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        payload = stream.read(64 * 1024 * 1024 + 1)
    _require(len(payload) <= 64 * 1024 * 1024, f"JSON report exceeds 64 MiB: {path}")

    def nonfinite(value):
        raise ValueError(f"non-finite JSON value: {value}")

    value = json.loads(payload.decode("utf-8"), parse_constant=nonfinite)
    _require(isinstance(value, dict), "report must be a JSON object")
    return value, path, hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value, low=0, high=math.inf):
    try:
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    except OverflowError:
        return False


def _gate(report, *, expected_dataset_schema="scene-0.2"):
    """Recompute the fixed gate from complete validation rows, including failures."""
    _require(
        report.get("schema_version") == "region-study-evaluation-0.2",
        "unsupported evaluation schema",
    )
    _require(report.get("split") == "validation", "selection requires validation reports only")
    _require(
        report.get("dataset_schema") == expected_dataset_schema,
        f"study requires the {expected_dataset_schema} dataset",
    )
    _require(
        report.get("test_used_for_model_selection") is False, "test-based selection is forbidden"
    )
    _require(
        report.get("postprocessor") == asdict(RegionConfig()),
        "postprocessor differs from fixed protocol",
    )
    _require(
        report.get("gate_method") == "learned_postprocessed",
        "gate must use postprocessed predictions",
    )
    _require(
        type(report.get("expected_count")) is int and report["expected_count"] == 144,
        "validation must declare all 144 cases",
    )
    method = report["methods"]["learned_postprocessed"]
    _require(
        method.get("evaluation_grid") == "original_source_pixels",
        "gate requires the original source-pixel grid",
    )
    _require(method.get("thresholds") == GATE_THRESHOLDS, "acceptance thresholds changed")
    rows = method["samples"]
    _require(
        isinstance(rows, list) and len(rows) == 144, "validation rows must include all 144 cases"
    )
    labels, ious, negative_fp = {}, [], 0
    completed = positives = negatives = 0
    for row in rows:
        identifier = row.get("id")
        _require(
            isinstance(identifier, str) and identifier and identifier not in labels,
            "duplicate or invalid validation sample ID",
        )
        _require(type(row.get("has_page")) is bool, "validation labels must be boolean")
        _require(row.get("status") in ("completed", "failed"), "invalid evaluation case status")
        labels[identifier] = row["has_page"]
        valid = row["status"] == "completed"
        if valid:
            _require(_number(row.get("iou"), 0, 1), "completed IoU must be finite and in [0,1]")
            _require(type(row.get("detected_page")) is bool, "completed detection must be boolean")
            completed += 1
        if row["has_page"]:
            positives += 1
            ious.append(float(row["iou"]) if valid else 0.0)
        else:
            negatives += 1
            negative_fp += int(not valid or row["detected_page"])
    _require(
        (positives, negatives) == (96, 48), "validation requires 96 positive and 48 negative cases"
    )
    mean_iou, fp_rate = float(np.mean(ious)), negative_fp / negatives
    summary = method["summary"]
    counts = {
        "expected_count": 144,
        "completed_count": completed,
        "failed_count": 144 - completed,
        "foreground_count": positives,
        "negative_count": negatives,
        "negative_false_positive_count_conservative": negative_fp,
    }
    for key, expected in counts.items():
        _require(
            type(summary.get(key)) is int and summary[key] == expected,
            f"summary count differs from rows: {key}",
        )
    reported_iou = summary["foreground_iou_all_cases"]
    _require(
        reported_iou.get("count") == 96 and _number(reported_iou.get("mean"), 0, 1),
        "invalid all-positive IoU summary",
    )
    _require(
        math.isclose(reported_iou["mean"], mean_iou, rel_tol=0, abs_tol=1e-12),
        "IoU summary disagrees with fixed-denominator rows",
    )
    _require(
        _number(summary.get("negative_false_positive_rate_conservative"), 0, 1),
        "invalid conservative negative FP rate",
    )
    _require(
        summary["negative_false_positive_rate_conservative"] == fp_rate,
        "negative FP summary disagrees with rows",
    )
    passed = completed == 144 and mean_iou >= 0.8 and fp_rate <= 0.1
    _require(
        summary.get("gate_passed") is passed and report.get("gate_passed") is passed,
        "reported gate disagrees with recomputed fixed gate",
    )
    _require(
        report.get("region_failure_count") == 144 - completed,
        "detector/evaluator failure counts disagree",
    )
    decisions = report["region_decisions"]
    _require(
        isinstance(decisions, list) and len(decisions) == 144,
        "all detector decisions must be recorded",
    )
    decision_map = {row["id"]: row["status"] for row in decisions}
    _require(
        set(decision_map) == set(labels) and len(decision_map) == 144,
        "detector decision sample IDs differ",
    )
    for status, key in (
        ("accepted", "region_accepted_count"),
        ("rejected", "region_rejected_count"),
        ("failed", "region_failure_count"),
    ):
        _require(
            report.get(key) == sum(value == status for value in decision_map.values()),
            f"decision count mismatch: {key}",
        )
    for row in rows:
        expected = (
            "failed"
            if row["status"] == "failed"
            else "accepted"
            if row["detected_page"]
            else "rejected"
        )
        _require(
            decision_map[row["id"]] == expected,
            "per-case detection and evaluation statuses disagree",
        )
    return {
        **counts,
        "foreground_iou_all_cases": mean_iou,
        "negative_false_positive_rate_conservative": fp_rate,
        "gate_passed": passed,
    }, labels


def _validate_training(
    training, checkpoint, digest, architecture, manifest, *, expected_dataset_schema="scene-0.2"
):
    expected_config = {
        "seed": 20260923,
        "epochs": 80,
        "batch_size": 16,
        "learning_rate": 0.001,
        "threads": 1,
        "architecture": architecture,
    }
    _require(training.get("schema_version") == STUDY_SCHEMA, "invalid study training report schema")
    _require(
        type(training.get("epochs_completed")) is int and training["epochs_completed"] == 80,
        "both new runs must complete exactly 80 epochs",
    )
    _require(
        training.get("config") == expected_config and checkpoint["config"] == expected_config,
        "training settings differ from preregistered budget or architecture",
    )
    _require(
        _canonical(training.get("model_spec"))
        == _canonical(model_spec(architecture))
        == _canonical(checkpoint["model_spec"]),
        "architecture specification mismatch",
    )
    _require(
        training.get("manifest_sha256") == checkpoint["manifest_sha256"] == manifest,
        "new checkpoint/training/evaluation manifests differ",
    )
    _require(
        training.get("dataset_counts") == EXPECTED_COUNTS == checkpoint["dataset_counts"],
        "training dataset counts differ from the fixed study",
    )
    _require(
        training.get("checkpoints", {}).get("best.pt") == digest,
        "checkpoint is not training report's selected best",
    )
    _require(
        training.get("best_epoch") == checkpoint["epoch"] == checkpoint["best_epoch"],
        "selected checkpoint epoch differs from training report",
    )
    _require(
        training.get("best_validation") == checkpoint["best_validation"],
        "selected validation loss/metrics disagree",
    )
    _require(
        training.get("initial_validation") == checkpoint["initial_validation"],
        "initial validation records disagree",
    )
    history = training.get("history")
    _require(
        isinstance(history, list) and len(history) == 80, "80 completed epochs must be recorded"
    )
    initial_loss = training["initial_validation"].get("loss")
    _require(_number(initial_loss), "invalid initial validation loss")
    losses = [initial_loss]
    for epoch, row in enumerate(history, 1):
        _require(
            type(row.get("epoch")) is int and row["epoch"] == epoch,
            "training history is not chronological",
        )
        value = row.get("validation", {}).get("loss")
        _require(_number(value), "training history requires finite validation losses")
        losses.append(value)
    chosen_epoch = min(range(81), key=losses.__getitem__)
    _require(
        chosen_epoch == training["best_epoch"],
        "checkpoint choice is not earliest minimum validation weighted loss",
    )
    selected_metrics = (
        training["initial_validation"]
        if chosen_epoch == 0
        else history[chosen_epoch - 1]["validation"]
    )
    _require(
        selected_metrics == training["best_validation"],
        "best validation metrics do not match selected history row",
    )
    _require(
        history[: checkpoint["epoch"]] == checkpoint["history"],
        "checkpoint history differs from completed run",
    )
    for provenance in (training["provenance"], checkpoint["provenance"]):
        _require(
            provenance.get("loss") == LOSS_PROTOCOL
            and provenance.get("selection") == SELECTION_PROTOCOL,
            "loss or checkpoint-selection protocol changed",
        )
        _require(
            provenance.get("dataset_schema") == expected_dataset_schema,
            f"new arms must be trained on {expected_dataset_schema}",
        )


def _arm(name, supplied):
    _require(isinstance(supplied, Mapping), f"arm {name} must provide file paths")
    required = {"checkpoint", "validation_report"} | ({"training_report"} if name != "A" else set())
    _require(set(supplied) == required, f"arm {name} requires exactly {sorted(required)}")
    evaluation, evaluation_path, evaluation_hash = _json_file(supplied["validation_report"])
    gate, labels = _gate(evaluation)
    path = Path(supplied["checkpoint"]).resolve()
    digest = sha256(path)
    checkpoint = read_region_checkpoint(path)
    _require(sha256(path) == digest, "checkpoint changed while being validated")
    _require(
        evaluation.get("checkpoint_sha256") == digest,
        f"arm {name} evaluation checkpoint SHA-256 differs from file",
    )
    _require(
        evaluation.get("checkpoint_epoch") == checkpoint["epoch"],
        "evaluation checkpoint epoch differs",
    )
    manifest, training_manifest = evaluation.get("manifest_sha256"), checkpoint["manifest_sha256"]
    _require(_digest(manifest) and _digest(training_manifest), "invalid manifest SHA-256")
    _require(
        evaluation.get("training_manifest_sha256") == training_manifest,
        "declared training manifest differs from checkpoint",
    )
    _require(
        evaluation.get("explicit_transfer_evaluation") is (manifest != training_manifest),
        "cross-dataset transfer must be declared explicitly",
    )
    loss = checkpoint["best_validation"].get("loss")
    _require(_number(loss), "selected checkpoint validation loss must be finite")
    training_path = training_hash = None
    if name == "A":
        _require(
            digest == LEGACY_CHECKPOINT_SHA256 and checkpoint["epoch"] == LEGACY_SELECTED_EPOCH,
            "arm A differs from the frozen legacy checkpoint",
        )
        _require(checkpoint["schema_version"] == LEGACY_SCHEMA, "arm A must use the legacy schema")
        _require(
            _canonical(checkpoint["config"]["model_config"])
            == _canonical(asdict(SegmenterConfig())),
            "arm A architecture differs from frozen established model",
        )
        _require(
            manifest != training_manifest,
            "arm A must explicitly transfer from its original training manifest",
        )
        spec = model_spec("local")
    else:
        _require(checkpoint["schema_version"] == STUDY_SCHEMA, "new arms require study checkpoints")
        training, training_path, training_hash = _json_file(supplied["training_report"])
        architecture = "local" if name == "B" else "context"
        _validate_training(training, checkpoint, digest, architecture, manifest)
        spec = checkpoint["model_spec"]
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_epoch": checkpoint["epoch"],
        "model_spec": spec,
        "validation_report_path": str(evaluation_path),
        "validation_report_sha256": evaluation_hash,
        "training_report_path": str(training_path) if training_path else None,
        "training_report_sha256": training_hash,
        "evaluation_manifest_sha256": manifest,
        "training_manifest_sha256": training_manifest,
        "explicit_transfer_evaluation": manifest != training_manifest,
        "selection_validation_weighted_loss": loss,
        "loss_manifest_sha256": training_manifest,
        "loss_scope": "original training-validation distribution"
        if name == "A"
        else "new shared validation distribution",
        "eligible_for_selection": name in ("B", "C"),
        "gate_passed": gate["gate_passed"],
        "gate_metrics": gate,
        "metrics": {key: value["summary"] for key, value in evaluation["methods"].items()},
        "rationale": "Frozen comparison baseline; excluded from candidate choice"
        if name == "A"
        else "Eligible new candidate; fixed validation gate applies",
    }, labels


def freeze_region_study(
    arms,
    output_path,
    *,
    coordinate_checkpoint,
    protocol_path,
) -> dict:
    """Validate A/B/C evidence, then exclusively create a validation-only freeze.

    ``arms`` maps A/B/C to checkpoint and validation_report paths; B/C also
    require training_report paths. The protocol file is hashed verbatim and may
    be the immutable pre-training JSON. It is never inferred from test results.
    """
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"selection already exists: {destination}")
    _require(
        isinstance(arms, Mapping) and set(arms) == {"A", "B", "C"},
        "exactly arms A, B and C are required",
    )
    records, identities = {}, None
    for name in ("A", "B", "C"):
        records[name], current_ids = _arm(name, arms[name])
        if identities is None:
            identities = current_ids
        _require(
            current_ids == identities,
            "arms did not evaluate the same validation sample identities/labels",
        )
    manifests = {value["evaluation_manifest_sha256"] for value in records.values()}
    _require(len(manifests) == 1, "all arms must use the same new evaluation manifest")
    _require(
        manifests == {STUDY_MANIFEST_SHA256}, "evaluation manifest differs from pretraining freeze"
    )
    candidates = [name for name in ("B", "C") if records[name]["gate_passed"]]
    selected = (
        min(candidates, key=lambda name: records[name]["selection_validation_weighted_loss"])
        if candidates
        else None
    )
    rationale = (
        "Neither new arm passes the fixed validation gate; promote neither"
        if selected is None
        else (
            f"Select {selected} among gate-passing new arms by minimum validation weighted loss; "
            "exact ties retain B"
        )
    )
    coordinate = load_predictor(coordinate_checkpoint)
    _require(
        coordinate.model_sha256 == COORDINATE_CHECKPOINT_SHA256,
        "coordinate checkpoint differs from the fixed legacy coordinate model",
    )
    protocol = Path(protocol_path).resolve()
    record = _canonical(
        {
            "schema_version": "region-study-selection-0.2",
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "record_policy": "exclusive creation; this tool never replaces a freeze record",
            "protocol_path": str(protocol),
            "protocol_sha256": sha256(protocol),
            "evaluation_manifest_sha256": next(iter(manifests)),
            "selection_split": "validation",
            "selected_arm": selected,
            "rationale": rationale,
            "arms": records,
            "gate_thresholds": GATE_THRESHOLDS,
            "gate_requires_no_failed_cases": True,
            "postprocessor": asdict(RegionConfig()),
            "component_connectivity": 8,
            "boundary_expansion_bisection_steps": 8,
            "renderer_config": asdict(RenderConfig()),
            "pipeline_output_size_hw": [192, 144],
            "coordinate_checkpoint_path": coordinate.checkpoint_path,
            "coordinate_checkpoint_sha256": coordinate.model_sha256,
            "completion_evidence": (
                "80-epoch training.json history and selected best checkpoint; "
                "last checkpoint is not loaded"
            ),
            "test_used_for_selection": False,
            "test_evaluation_performed": False,
            "test_reselection_allowed": False,
            "real_photo_evaluation_performed": False,
            "quality_claim_allowed": False,
            "scope": (
                "Printed-page detector study; blank documents unsupported; "
                "a gate pass is not a real-photo quality guarantee"
            ),
        }
    )
    encoded = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    for arm in records.values():
        for prefix in ("checkpoint", "validation_report", "training_report"):
            path, digest = arm.get(prefix + "_path"), arm.get(prefix + "_sha256")
            if path is not None:
                _require(
                    sha256(Path(path)) == digest, f"{prefix} changed before freeze publication"
                )
    _require(
        sha256(Path(coordinate.checkpoint_path)) == coordinate.model_sha256,
        "coordinate checkpoint changed before freeze publication",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".region-freeze-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)  # Atomic create-if-absent; never replace another record.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record
