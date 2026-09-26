"""Secondary paired-scene diagnostics, preserving the existing region gate.

Pair metadata is used only for reporting, never as a model input. A successful
pair does not establish a causal mechanism for learning from matched examples.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from ..synthetic.scene_dataset import validate_scene_dataset
from .segmentation import sha256
from .study_evaluation import evaluate_region_study
from .training import _provenance

_OUTCOMES = (
    "printed_accepted_blank_rejected",
    "both_accepted",
    "both_rejected",
    "wrong_way",
    "error",
    "missing",
)


def _evaluation_provenance():
    provenance = _provenance()
    return {
        key: provenance[key]
        for key in ("source_code_sha256", "versions", "device", "preprocess_version")
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifier(value):
    return isinstance(value, str) and bool(value)


def _index(rows, description):
    _require(isinstance(rows, list), f"{description} must be a list")
    indexed = {}
    for row in rows:
        _require(isinstance(row, dict), f"invalid {description} row")
        key = row.get("id")
        _require(_identifier(key) and key not in indexed, f"duplicate or invalid {description} ID")
        indexed[key] = row
    return indexed


def _selected_pairs(manifest, split):
    _require(split in ("validation", "test"), "split must be validation or test")
    _require(
        isinstance(manifest, dict) and manifest.get("schema_version") == "scene-0.3",
        "paired evaluation requires scene-0.3",
    )
    all_records, all_pairs = manifest.get("samples"), manifest.get("pairs")
    _require(isinstance(all_records, list) and isinstance(all_pairs, list), "invalid pair manifest")
    _require(
        all(isinstance(row, dict) for row in all_records + all_pairs),
        "manifest samples and pairs require object rows",
    )
    records = _index([row for row in all_records if row.get("split") == split], "selected sample")
    pairs = _index([row for row in all_pairs if row.get("split") == split], "selected pair")
    _require(records and pairs, "selected split must contain samples and declared pairs")
    members = set()
    for pair_id, pair in pairs.items():
        printed_id, blank_id = pair.get("printed_id"), pair.get("blank_id")
        _require(
            _identifier(printed_id) and _identifier(blank_id) and printed_id != blank_id,
            "pair requires distinct printed and blank IDs",
        )
        _require(
            printed_id in records and blank_id in records,
            "pair partner is missing or outside the selected split",
        )
        _require(
            printed_id not in members and blank_id not in members,
            "sample belongs to more than one pair",
        )
        group = pair.get("document_group")
        _require(_identifier(group), "pair requires a document group")
        for identifier, partner, role, has_page in (
            (printed_id, blank_id, "printed", True),
            (blank_id, printed_id, "blank", False),
        ):
            row = records[identifier]
            _require(
                row.get("pair_id") == pair_id
                and row.get("pair_role") == role
                and row.get("partner_id") == partner,
                "pair member metadata or reciprocal partner differs from declaration",
            )
            _require(row.get("has_page") is has_page, "pair role and page label disagree")
            _require(row.get("document_group") == group, "pair document groups disagree")
            _require(
                row.get("negative_type") == (None if has_page else "paper_panel"),
                "pair role and negative type disagree",
            )
            members.add(identifier)
    for identifier, row in records.items():
        _require(type(row.get("has_page")) is bool, "selected sample page labels must be boolean")
        if identifier not in members:
            _require(
                all(row.get(key) is None for key in ("pair_id", "pair_role", "partner_id")),
                "undeclared pair metadata on selected sample",
            )
            _require(
                row.get("negative_type") != "paper_panel",
                "selected blank panel is missing its declared pair",
            )
    return records, pairs


def _finite_iou(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        return False


def paired_region_diagnostics(manifest, report, *, split="validation") -> dict:
    """Diagnose every declared pair using fixed source-grid postprocessed results.

    Malformed pair declarations or contradictory results raise. Missing or failed
    results are explicitly counted, never interpreted as successful blank rejection.
    No primary metric, acceptance threshold, or checkpoint choice is changed.
    """
    records, pairs = _selected_pairs(manifest, split)
    _require(
        report.get("schema_version") == "region-study-evaluation-0.2"
        and report.get("dataset_schema") == "scene-0.3"
        and report.get("split") == split,
        "report schema or selected split differs from paired dataset",
    )
    _require(
        type(report.get("expected_count")) is int and report["expected_count"] == len(records),
        "report must declare the complete selected split",
    )
    method = report.get("methods", {}).get("learned_postprocessed")
    if method is not None:
        _require(
            method.get("evaluation_grid") == "original_source_pixels",
            "paired IoU requires the original source-pixel grid",
        )
    rows = _index([] if method is None else method.get("samples", []), "evaluation sample")
    decisions = _index(report.get("region_decisions", []), "region decision")
    _require(
        set(rows) <= set(records) and set(decisions) <= set(records),
        "results include IDs outside the selected split",
    )
    statuses, missing, errors = {}, [], []
    for identifier, record in records.items():
        row, decision = rows.get(identifier), decisions.get(identifier)
        if row is not None:
            _require(
                row.get("has_page") is record["has_page"], "evaluation page label differs from GT"
            )
            _require(row.get("status") in ("completed", "failed"), "invalid evaluation status")
            if row["status"] == "completed":
                _require(_finite_iou(row.get("iou")), "completed IoU must be finite and in [0,1]")
                _require(type(row.get("detected_page")) is bool, "invalid completed detection")
                if not record["has_page"]:
                    _require(
                        row["iou"] == float(not row["detected_page"]),
                        "negative IoU violates the fixed empty-mask convention",
                    )
        if decision is not None:
            _require(
                decision.get("status") in ("accepted", "rejected", "failed"),
                "invalid region decision status",
            )
        if row is None or decision is None:
            statuses[identifier] = "missing"
            missing.append(identifier)
        elif row["status"] == "failed" or decision["status"] == "failed":
            statuses[identifier] = "failed"
            errors.append(identifier)
        else:
            accepted = decision["status"] == "accepted"
            _require(row["detected_page"] is accepted, "decision and detected-page result disagree")
            if not accepted and record["has_page"]:
                _require(row["iou"] == 0, "rejected positive must retain IoU zero")
            statuses[identifier] = decision["status"]
    counts = dict.fromkeys(_OUTCOMES, 0)
    samples, secondary_count = [], 0
    for pair_id, pair in sorted(pairs.items()):
        printed_id, blank_id = pair["printed_id"], pair["blank_id"]
        printed, blank = statuses[printed_id], statuses[blank_id]
        printed_row = rows.get(printed_id)
        iou = (
            float(printed_row["iou"])
            if printed_row is not None and printed_row["status"] == "completed"
            else None
        )
        if "missing" in (printed, blank):
            outcome = "missing"
        elif "failed" in (printed, blank):
            outcome = "error"
        else:
            outcome = {
                ("accepted", "rejected"): "printed_accepted_blank_rejected",
                ("accepted", "accepted"): "both_accepted",
                ("rejected", "rejected"): "both_rejected",
                ("rejected", "accepted"): "wrong_way",
            }[(printed, blank)]
        success = outcome == "printed_accepted_blank_rejected" and iou >= 0.8
        counts[outcome] += 1
        secondary_count += int(success)
        samples.append(
            {
                "pair_id": pair_id,
                "document_group": pair["document_group"],
                "printed_id": printed_id,
                "blank_id": blank_id,
                "printed_size_bin": records[printed_id].get("size_bin"),
                "printed_status": printed,
                "blank_status": blank,
                "printed_iou": iou,
                "outcome": outcome,
                "printed_iou_ge_0_8_and_blank_rejected": success,
            }
        )
    expected = len(pairs)
    failed = counts["error"] + counts["missing"]
    return {
        "schema_version": "paired-region-diagnostics-0.1",
        "split": split,
        "method": "learned_postprocessed",
        "evaluation_grid": "original_source_pixels",
        "expected_pair_count": expected,
        "completed_pair_count": expected - failed,
        "failed_pair_count": failed,
        **{key + "_count": value for key, value in counts.items()},
        "printed_accepted_blank_rejected_rate": counts["printed_accepted_blank_rejected"]
        / expected,
        "printed_iou_ge_0_8_and_blank_rejected_count": secondary_count,
        "printed_iou_ge_0_8_and_blank_rejected_rate": secondary_count / expected,
        "selected_split_expected_count": len(records),
        "unpaired_sample_count": len(records) - 2 * expected,
        "missing_sample_ids": missing,
        "error_sample_ids": errors,
        "complete_selected_split": not missing and not errors,
        "additional_gate": False,
        "used_for_model_selection": False,
        "preprocessing_qa": manifest.get("preprocessing_qa", {}).get(split),
        "scope": (
            "Secondary diagnostics only; rates retain all declared pairs in the denominator. "
            "Pair metadata is not a model input; these observations do not establish a pairing "
            "mechanism or change the primary acceptance gate."
        ),
        "samples": samples,
    }


def evaluate_paired_regions(
    dataset_dir,
    checkpoint_path,
    *,
    split="validation",
    output_dir=None,
    expected_training_manifest=None,
    threads=1,
) -> dict:
    """Evaluate scene-0.3 with unchanged primary metrics and write-once output."""
    _require(split in ("validation", "test"), "split must be validation or test")
    _require(type(threads) is int and 1 <= threads <= 128, "threads must be an integer in [1,128]")
    destination = None if output_dir is None else Path(output_dir) / f"region-study-{split}.json"
    if destination is not None and destination.exists():
        raise FileExistsError(f"report already exists: {destination}")
    directory = Path(dataset_dir).resolve()
    manifest_path = directory / "manifest.json"
    digest = sha256(manifest_path)
    checked = validate_scene_dataset(directory)
    _require(sha256(manifest_path) == digest, "manifest changed while being validated")
    _selected_pairs(checked["manifest"], split)
    provenance = _evaluation_provenance()
    report = evaluate_region_study(
        directory,
        checkpoint_path,
        split=split,
        output_dir=None,
        expected_training_manifest=expected_training_manifest,
        threads=threads,
    )
    _require(
        report.get("manifest_sha256") == digest and sha256(manifest_path) == digest,
        "evaluation and pair metadata manifest identities differ",
    )
    report["paired_diagnostics"] = paired_region_diagnostics(
        checked["manifest"], report, split=split
    )
    _require(
        _evaluation_provenance() == provenance,
        "evaluation source or runtime changed during evaluation",
    )
    report["evaluation_provenance"] = provenance
    if destination is not None:
        encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".paired-evaluation-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return report
