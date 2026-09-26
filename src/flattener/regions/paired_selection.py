"""Write-once pretraining lock and validation-only choice for the paired study.

The lock hashes saved dataset artifacts and reads metadata, but never decodes
pixels. Selection reads only locked manifest bytes and explicitly supplied
validation/training reports and checkpoints; it never evaluates a model or
discovers test reports. SHA sidecars detect accidental changes, not hostile
replacement of both a file and its sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..contracts import RenderConfig
from ..learning.inference import load_predictor
from ..synthetic.paired_scenes import PairedSceneConfig, validate_paired_scene_dataset
from .checkpoints import STUDY_SCHEMA, model_spec, read_region_checkpoint
from .geometry import RegionConfig
from .paired_evaluation import paired_region_diagnostics
from .segmentation import sha256
from .study_selection import (
    COORDINATE_CHECKPOINT_SHA256,
    EXPECTED_COUNTS,
    GATE_THRESHOLDS,
    _canonical,
    _digest,
    _gate,
    _json_file,
    _number,
    _require,
    _validate_training,
)
from .training import LOSS_PROTOCOL, SELECTION_PROTOCOL, _provenance

BASELINE_CHECKPOINT_SHA256 = "2b25676a2968d126fec6a5b34595d8a4b317e3722e4f9ea87d4820723af2a116"
BASELINE_MANIFEST_SHA256 = "06d3c61c360a4db2818516296e20153cc9ec98ee218704a7e1f133cf3e8d8f19"
BASELINE_EPOCH = 73
LOCK_SCHEMA = "paired-region-preregistration-0.3"
SELECTION_SCHEMA = "paired-region-selection-0.3"
DATASET_SCHEMA = "scene-0.3"
TRAINING_CONFIG = {
    "seed": 20260923,
    "epochs": 80,
    "batch_size": 16,
    "learning_rate": 0.001,
    "threads": 1,
    "architecture": "context",
}
_SPLITS = ("train", "validation", "test")
_PAIR_FIELDS = {"capture_session", "pair_id", "pair_role", "partner_id"}
_PAIR_METADATA = _PAIR_FIELDS | {"pair_context", "pair_context_sha256"}
_ENVIRONMENT_KEYS = (
    "source_code_sha256",
    "uv_lock_sha256",
    "versions",
    "device",
    "preprocess_version",
)
_GENERATOR_SOURCES = (
    "synthetic/paired_scenes.py",
    "synthetic/scene_v2.py",
    "synthetic/scenes.py",
    "synthetic/documents.py",
    "synthetic/geometry.py",
    "synthetic/pipeline.py",
    "renderer.py",
    "coordinates.py",
    "validation.py",
    "contracts.py",
    "regions/segmentation.py",
)


def _encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _publish(path, payload):
    """Atomically create one immutable file without replacing concurrent writers."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".paired-freeze-", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _environment():
    provenance = _provenance()
    root = Path(__file__).resolve().parents[3]
    _require(
        (root / "pyproject.toml").is_file() and (root / "examples").is_dir(),
        "paired locking/freezing requires the source checkout, not an isolated installed wheel",
    )
    return {
        **{key: provenance[key] for key in _ENVIRONMENT_KEYS},
        "example_code_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted((root / "examples").rglob("*.py"))
        },
        "pyproject_sha256": sha256(root / "pyproject.toml"),
    }


def _fixed_protocol():
    return _canonical(
        {
            "dataset_schema": DATASET_SCHEMA,
            "dataset_seed": 20260924,
            "training_config": TRAINING_CONFIG,
            "model_spec": model_spec("context"),
            "parameter_count": 140545,
            "loss": LOSS_PROTOCOL,
            "checkpoint_selection": SELECTION_PROTOCOL,
            "gate_thresholds": GATE_THRESHOLDS,
            "postprocessor": asdict(RegionConfig()),
            "renderer_config": asdict(RenderConfig()),
            "pipeline_output_size_hw": [192, 144],
            "candidate_rule": (
                "gate-passing B/C; minimum best validation weighted loss; exact ties B"
            ),
            "baseline_eligible": False,
            "test_reselection_allowed": False,
        }
    )


def _expected_counts():
    return {
        split: {
            "page": documents * 4,
            "negative": documents * 2,
            "total": documents * 6,
            "size_bins": {name: documents * 4 // 3 for name in ("small", "medium", "large")},
            "blank_reference_size_bins": {
                name: documents // 3 for name in ("small", "medium", "large")
            },
            "negative_types": {"background": documents, "paper_panel": documents},
        }
        for split, documents in zip(_SPLITS, (96, 24, 24), strict=True)
    }


def _manifest(path, mode, *, validate):
    root = Path(path).resolve()
    manifest, filename, digest = _json_file(root / "manifest.json")
    if validate:
        checked = validate_paired_scene_dataset(root)
        _require(checked["manifest"] == manifest, "manifest changed during dataset validation")
    _require(manifest.get("schema_version") == DATASET_SCHEMA, "paired study requires scene-0.3")
    _require(
        manifest.get("config") == _canonical(asdict(PairedSceneConfig(training_panels=mode))),
        f"{mode} dataset configuration differs from fixed protocol",
    )
    _require(manifest.get("complete") is True, "dataset is not complete")
    _require(manifest.get("counts") == _expected_counts(), "dataset counts/strata differ")
    records = manifest.get("samples")
    _require(isinstance(records, list) and len(records) == 864, "all 864 dataset records required")
    ids = [row["id"] for row in records]
    _require(len(set(ids)) == len(ids), "dataset sample IDs must be unique")
    for split, expected in EXPECTED_COUNTS.items():
        rows = [row for row in records if row["split"] == split]
        labels = [row.get("has_page") for row in rows]
        _require(all(type(label) is bool for label in labels), "dataset labels must be boolean")
        count = _expected_counts()[split]
        _require(
            len(rows) == expected
            and sum(labels) == count["page"]
            and sum(row["negative_type"] == "paper_panel" for row in rows)
            == count["negative_types"]["paper_panel"]
            and sum(row["negative_type"] == "background" for row in rows)
            == count["negative_types"]["background"],
            "dataset rows differ from fixed split/label counts",
        )
    _require(sha256(filename) == digest, "manifest changed while being checked")
    return manifest, {
        "directory": str(root),
        "manifest_path": str(filename),
        "manifest_sha256": digest,
        "config": manifest["config"],
        "counts": manifest["counts"],
        "preprocessing_qa": manifest["preprocessing_qa"],
        "generator_code_sha256": manifest["generator_code_sha256"],
    }


def _without(value, keys):
    return {key: item for key, item in value.items() if key not in keys}


def _contrast(matched, independent, matched_root, independent_root):
    """Verify the treatment without decoding any stored image or mask."""
    varying = {"config", "samples", "pairs", "preprocessing_qa"}
    _require(
        _without(matched, varying) == _without(independent, varying),
        "dataset metadata outside treatment changed",
    )
    left, right = matched["samples"], independent["samples"]
    _require(
        [row["id"] for row in left] == [row["id"] for row in right],
        "sample ordering/identities differ",
    )
    changed_images, changed_metadata = [], []
    for a, b in zip(left, right, strict=True):
        identifier = a["id"]
        _require(a["paths"] == b["paths"], "artifact relative paths differ")
        if a["split"] != "train":
            _require(a == b, "validation/test records and artifact hashes must be exactly equal")
            continue
        blank = not a["has_page"] and a["negative_type"] == "paper_panel"
        _require(
            a["sha256"]["mask.png"] == b["sha256"]["mask.png"],
            "segmentation target changed across modes",
        )
        image_changed = a["sha256"]["image.png"] != b["sha256"]["image.png"]
        _require(image_changed is blank, "only every training blank RGB image may change")
        if image_changed:
            changed_images.append(identifier)
        if a["sha256"]["metadata.json"] != b["sha256"]["metadata.json"]:
            changed_metadata.append(identifier)
        ma, _, ma_hash = _json_file(matched_root / a["paths"]["metadata.json"])
        mb, _, mb_hash = _json_file(independent_root / b["paths"]["metadata.json"])
        _require(
            ma_hash == a["sha256"]["metadata.json"] and mb_hash == b["sha256"]["metadata.json"],
            "training metadata hash changed during contrast validation",
        )
        if blank:
            stable = (
                "id",
                "document_group",
                "split",
                "has_page",
                "negative_type",
                "context_id",
                "background_type",
            )
            _require(
                all(a[key] == b[key] for key in stable), "training blank identity/label changed"
            )
            stable_metadata = (
                "sample_id",
                "document_group",
                "split",
                "context_id",
                "image_size",
                "source_document_sha256",
                "blank_reference_size_bin",
            )
            _require(
                all(ma[key] == mb[key] for key in stable_metadata),
                "training blank source identity/seed changed",
            )
            _require(
                all(
                    ma["seeds"][key] == mb["seeds"][key]
                    for key in ("document", "scene", "background", "photometric")
                ),
                "training blank shared context seeds changed",
            )
        else:
            _require(
                _without(a, _PAIR_FIELDS | {"sha256"}) == _without(b, _PAIR_FIELDS | {"sha256"})
                and _without(ma, _PAIR_METADATA) == _without(mb, _PAIR_METADATA),
                "nonblank training metadata changed beyond pair linkage",
            )
    _require(
        len(changed_images) == 96, "contrast must contain exactly 96 training blank RGB changes"
    )
    for split in ("validation", "test"):
        _require(
            [row for row in matched["pairs"] if row["split"] == split]
            == [row for row in independent["pairs"] if row["split"] == split]
            and matched["preprocessing_qa"][split] == independent["preprocessing_qa"][split],
            "evaluation pair graph or generation QA differs",
        )
    _require(
        sum(row["split"] == "train" for row in matched["pairs"]) == 96
        and not any(row["split"] == "train" for row in independent["pairs"]),
        "training pairing must be 96 matched versus zero independent pairs",
    )
    return {
        "changed_training_rgb_count": 96,
        "changed_training_rgb_ids": changed_images,
        "changed_training_metadata_count": len(changed_metadata),
        "changed_training_metadata_ids": changed_metadata,
        "all_target_hashes_equal": True,
        "nonblank_rgb_hashes_equal": True,
        "sample_order_equal": True,
        "evaluation_records_and_artifact_hashes_equal": True,
        "saved_test_pixels_decoded": False,
    }


def _baseline(path):
    path = Path(path).resolve()
    digest = sha256(path)
    _require(digest == BASELINE_CHECKPOINT_SHA256, "A differs from frozen v2 context checkpoint")
    checkpoint = read_region_checkpoint(path)
    _require(
        checkpoint["schema_version"] == STUDY_SCHEMA
        and checkpoint["epoch"] == checkpoint["best_epoch"] == BASELINE_EPOCH
        and checkpoint["manifest_sha256"] == BASELINE_MANIFEST_SHA256
        and _canonical(checkpoint["model_spec"]) == _canonical(model_spec("context")),
        "A checkpoint architecture, epoch or original manifest differs",
    )
    _require(sha256(path) == digest, "baseline changed during validation")
    return {
        "path": str(path),
        "sha256": digest,
        "epoch": BASELINE_EPOCH,
        "training_manifest_sha256": BASELINE_MANIFEST_SHA256,
    }


def _coordinate(path):
    model = load_predictor(path)
    _require(
        model.model_sha256 == COORDINATE_CHECKPOINT_SHA256,
        "coordinate checkpoint differs from frozen model",
    )
    _require(
        sha256(Path(model.checkpoint_path)) == model.model_sha256,
        "coordinate checkpoint changed during validation",
    )
    return {"path": str(Path(model.checkpoint_path).resolve()), "sha256": model.model_sha256}


def _check_files(files):
    for path, digest in files:
        _require(
            _digest(digest) and sha256(Path(path)) == digest, f"locked evidence changed: {path}"
        )


def lock_paired_study(
    matched_dataset,
    independent_dataset,
    output_dir,
    *,
    protocol_path,
    baseline_checkpoint,
    coordinate_checkpoint,
) -> dict:
    """Validate the fixed data-only treatment and exclusively create a new lock directory."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"pretraining lock directory already exists: {destination}")
    protocol = Path(protocol_path).resolve()
    protocol_bytes = protocol.read_bytes()
    _require(0 < len(protocol_bytes) <= 1024 * 1024, "protocol must contain at most 1 MiB")
    protocol_text = protocol_bytes.decode("utf-8")
    _require(
        not any(
            marker in protocol_text
            for marker in ("Pending:", "pending generation lock", "identity lock pending")
        ),
        "protocol contains unresolved pretraining identity placeholders",
    )
    matched, matched_record = _manifest(matched_dataset, "matched", validate=True)
    independent, independent_record = _manifest(independent_dataset, "independent", validate=True)
    contrast = _contrast(
        matched,
        independent,
        Path(matched_record["directory"]),
        Path(independent_record["directory"]),
    )
    baseline, coordinate = _baseline(baseline_checkpoint), _coordinate(coordinate_checkpoint)
    environment = _environment()
    _require(_digest(environment["uv_lock_sha256"]), "dependency uv.lock SHA-256 required")
    generator = {key: environment["source_code_sha256"].get(key) for key in _GENERATOR_SOURCES}
    _require(all(_digest(value) for value in generator.values()), "generator source hashes missing")
    for manifest in (matched, independent):
        _require(
            manifest["generator_code_sha256"] == generator
            and manifest.get("versions") == environment["versions"],
            "dataset generation provenance differs from locked sources/environment",
        )
    record = _canonical(
        {
            "schema_version": LOCK_SCHEMA,
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "protocol_path": str(protocol),
            "protocol_snapshot_path": str(destination / "protocol.md"),
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "fixed_protocol": _fixed_protocol(),
            "environment": environment,
            "datasets": {"matched": matched_record, "independent": independent_record},
            "contrast": contrast,
            "baseline": baseline,
            "coordinate": coordinate,
            "validation_identities": {
                row["id"]: row["has_page"]
                for row in matched["samples"]
                if row["split"] == "validation"
            },
            "test_pixels_decoded": False,
            "test_evaluation_performed": False,
            "record_policy": (
                "exclusive creation; SHA-256 integrity is not an authenticity boundary"
            ),
        }
    )
    _check_files(
        [
            (protocol, record["protocol_sha256"]),
            (baseline["path"], baseline["sha256"]),
            (coordinate["path"], coordinate["sha256"]),
        ]
        + [
            (value["manifest_path"], value["manifest_sha256"])
            for value in record["datasets"].values()
        ]
    )
    _require(_environment() == environment, "sources/environment changed during pretraining lock")
    # Rehash every stored artifact immediately before publishing. This reads
    # encoded bytes only, including test artifacts; no pixels are decoded.
    _check_files(
        [
            (Path(root) / row["paths"][name], digest)
            for root, manifest in (
                (matched_record["directory"], matched),
                (independent_record["directory"], independent),
            )
            for row in manifest["samples"]
            for name, digest in row["sha256"].items()
        ]
    )
    destination.mkdir(parents=True, exist_ok=False)
    payload = _encoded(record)
    _publish(destination / "protocol.md", protocol_bytes)
    _publish(
        destination / "preregistration.json.sha256",
        (hashlib.sha256(payload).hexdigest() + "\n").encode(),
    )
    _publish(destination / "preregistration.json", payload)
    return record


def _read_lock(path):
    record, filename, digest = _json_file(path)
    sidecar = filename.with_name(filename.name + ".sha256")
    _require(sidecar.read_text().strip() == digest, "pretraining lock SHA-256 sidecar mismatch")
    _require(record.get("schema_version") == LOCK_SCHEMA, "invalid paired pretraining lock schema")
    _require(record.get("fixed_protocol") == _fixed_protocol(), "locked study protocol differs")
    _require(
        record.get("environment") == _environment(),
        "sources/environment changed since pretraining lock",
    )
    _require(
        record.get("test_pixels_decoded") is False
        and record.get("test_evaluation_performed") is False,
        "lock must precede saved test decoding/evaluation",
    )
    files = [(filename, digest), (sidecar, sha256(sidecar))]
    files += [
        (record[key], record["protocol_sha256"])
        for key in ("protocol_path", "protocol_snapshot_path")
    ]
    matched = None
    for mode in ("matched", "independent"):
        stored = record["datasets"][mode]
        manifest, current = _manifest(stored["directory"], mode, validate=False)
        if mode == "matched":
            matched = manifest
        _require(stored == current, "dataset manifest differs from pretraining lock")
        files.append((stored["manifest_path"], stored["manifest_sha256"]))
        labels = {
            row["id"]: row["has_page"]
            for row in manifest["samples"]
            if row["split"] == "validation"
        }
        _require(
            labels == record["validation_identities"],
            "locked validation identities differ from manifests",
        )
    _require(
        record["baseline"]["sha256"] == BASELINE_CHECKPOINT_SHA256
        and record["coordinate"]["sha256"] == COORDINATE_CHECKPOINT_SHA256,
        "fixed checkpoint identities differ from lock",
    )
    files += [(record[key]["path"], record[key]["sha256"]) for key in ("baseline", "coordinate")]
    _check_files(files)
    return record, filename, digest, files, matched


def _arm(name, supplied, lock, manifest):
    required = {"checkpoint", "validation_report"} | ({"training_report"} if name != "A" else set())
    _require(
        isinstance(supplied, Mapping) and set(supplied) == required,
        f"arm {name} requires exactly {sorted(required)}",
    )
    evaluation, evaluation_path, evaluation_hash = _json_file(supplied["validation_report"])
    gate, labels = _gate(evaluation, expected_dataset_schema=DATASET_SCHEMA)
    _require(
        labels == lock["validation_identities"],
        "evaluation differs from locked validation identities/labels",
    )
    _require(
        type(evaluation.get("cpu_threads")) is int and evaluation["cpu_threads"] == 1,
        "paired validation must use one CPU thread",
    )
    _require(
        evaluation.get("evaluation_provenance")
        == {key: lock["environment"][key] for key in _ENVIRONMENT_KEYS if key != "uv_lock_sha256"},
        "evaluation provenance differs from pretraining source/environment lock",
    )
    diagnostics = paired_region_diagnostics(manifest, evaluation, split="validation")
    _require(diagnostics["expected_pair_count"] == 24, "validation requires exactly 24 pairs")
    _require(
        _encoded(evaluation.get("paired_diagnostics")) == _encoded(diagnostics),
        "paired diagnostics differ from recomputed locked-manifest validation evidence",
    )
    path = Path(supplied["checkpoint"]).resolve()
    digest = sha256(path)
    checkpoint = read_region_checkpoint(path)
    _require(
        checkpoint["schema_version"] == STUDY_SCHEMA
        and _canonical(checkpoint["model_spec"]) == _canonical(model_spec("context")),
        "all paired arms require the unchanged context architecture",
    )
    expected_training = (
        BASELINE_MANIFEST_SHA256
        if name == "A"
        else lock["datasets"]["independent" if name == "B" else "matched"]["manifest_sha256"]
    )
    common_manifest = lock["datasets"]["matched"]["manifest_sha256"]
    _require(
        evaluation.get("manifest_sha256") == common_manifest,
        "all evaluations require the locked common matched manifest",
    )
    _require(
        evaluation.get("checkpoint_sha256") == digest
        and evaluation.get("checkpoint_epoch") == checkpoint["epoch"],
        "evaluation checkpoint hash/epoch differs",
    )
    _require(
        evaluation.get("training_manifest_sha256")
        == checkpoint["manifest_sha256"]
        == expected_training,
        "arm training manifest differs from fixed treatment",
    )
    _require(
        evaluation.get("explicit_transfer_evaluation") is (expected_training != common_manifest),
        "cross-dataset transfer must be declared explicitly",
    )
    loss = checkpoint["best_validation"].get("loss")
    _require(_number(loss), "checkpoint validation weighted loss must be finite")
    training_path = training_hash = None
    if name == "A":
        _require(
            digest == BASELINE_CHECKPOINT_SHA256 and checkpoint["epoch"] == BASELINE_EPOCH,
            "A differs from frozen v2 context checkpoint",
        )
    else:
        training, training_path, training_hash = _json_file(supplied["training_report"])
        _validate_training(
            training,
            checkpoint,
            digest,
            "context",
            expected_training,
            expected_dataset_schema=DATASET_SCHEMA,
        )
        for evidence in (training, checkpoint):
            _require(
                all(
                    evidence["provenance"].get(key) == lock["environment"][key]
                    for key in _ENVIRONMENT_KEYS
                ),
                "training provenance differs from pretraining source/environment lock",
            )
    files = [(path, digest), (evaluation_path, evaluation_hash)]
    if training_path is not None:
        files.append((training_path, training_hash))
    _check_files(files)
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_epoch": checkpoint["epoch"],
        "model_spec": checkpoint["model_spec"],
        "validation_report_path": str(evaluation_path),
        "validation_report_sha256": evaluation_hash,
        "training_report_path": str(training_path) if training_path else None,
        "training_report_sha256": training_hash,
        "evaluation_manifest_sha256": common_manifest,
        "training_manifest_sha256": expected_training,
        "explicit_transfer_evaluation": expected_training != common_manifest,
        "initial_validation": checkpoint["initial_validation"],
        "selection_validation_weighted_loss": loss,
        "loss_manifest_sha256": expected_training,
        "loss_scope": "original v2 validation distribution; not comparable to B/C"
        if name == "A"
        else "shared v3 matched validation distribution",
        "eligible_for_selection": name != "A",
        "gate_passed": gate["gate_passed"],
        "gate_metrics": gate,
        "paired_diagnostics": diagnostics,
        "metrics": {key: value["summary"] for key, value in evaluation["methods"].items()},
    }, files


def freeze_paired_study(arms, output_path, *, lock_path, coordinate_checkpoint) -> dict:
    """Exclusively freeze a validation-only B/C choice after both fixed 80-epoch runs."""
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"selection already exists: {destination}")
    _require(
        isinstance(arms, Mapping) and set(arms) == {"A", "B", "C"},
        "exactly arms A, B and C required",
    )
    lock, lock_file, lock_hash, files, manifest = _read_lock(lock_path)
    records = {}
    for name in ("A", "B", "C"):
        records[name], evidence = _arm(name, arms[name], lock, manifest)
        files.extend(evidence)
    _require(
        records["B"]["initial_validation"] == records["C"]["initial_validation"],
        "B/C initialized validation metrics differ; expected identical initialization/evaluation",
    )
    candidates = [name for name in ("B", "C") if records[name]["gate_passed"]]
    selected = (
        min(candidates, key=lambda name: records[name]["selection_validation_weighted_loss"])
        if candidates
        else None
    )
    coordinate = _coordinate(coordinate_checkpoint)
    record = _canonical(
        {
            "schema_version": SELECTION_SCHEMA,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "pretraining_lock_path": str(lock_file),
            "pretraining_lock_sha256": lock_hash,
            "protocol_snapshot_path": lock["protocol_snapshot_path"],
            "protocol_sha256": lock["protocol_sha256"],
            "fixed_protocol": lock["fixed_protocol"],
            "environment": lock["environment"],
            "evaluation_manifest_sha256": lock["datasets"]["matched"]["manifest_sha256"],
            "selection_split": "validation",
            "selected_arm": selected,
            "arms": records,
            "rationale": "Neither B nor C passes the fixed validation gate; promote neither"
            if selected is None
            else (
                f"Select {selected} among gate-passing B/C by minimum validation weighted loss; "
                "exact ties retain B"
            ),
            "initial_validation_metrics_equal": True,
            "initial_validation_equality_scope": (
                "diagnostic consistency check, not proof of equal initialized tensor hashes"
            ),
            "coordinate_checkpoint_path": coordinate["path"],
            "coordinate_checkpoint_sha256": coordinate["sha256"],
            "test_used_for_selection": False,
            "test_evaluation_performed": False,
            "test_reselection_allowed": False,
            "real_photo_evaluation_performed": False,
            "quality_claim_allowed": False,
            "record_policy": (
                "exclusive creation; evidence/source hashes rechecked before publication"
            ),
        }
    )
    files.append((coordinate["path"], coordinate["sha256"]))
    _check_files(files)
    _require(
        _environment() == lock["environment"],
        "sources/environment changed before selection publication",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish(destination, _encoded(record))
    return record
