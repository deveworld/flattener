"""Paired evaluation invariants using generated metadata and in-memory images only."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flattener.regions import paired_evaluation as paired
from flattener.regions import study_evaluation as study


def _manifest(count=2):
    manifest = {"schema_version": "scene-0.3", "samples": [], "pairs": []}
    for split in ("train", "validation", "test"):
        for number in range(count):
            pair_id = f"{split}-pair-{number}"
            group = f"{split}-document-{number}"
            printed, blank = f"{pair_id}-printed", f"{pair_id}-blank"
            manifest["pairs"].append(
                {
                    "id": pair_id,
                    "split": split,
                    "printed_id": printed,
                    "blank_id": blank,
                    "document_group": group,
                    "context_sha256": "c" * 64,
                }
            )
            for identifier, partner, role, has_page in (
                (printed, blank, "printed", True),
                (blank, printed, "blank", False),
            ):
                manifest["samples"].append(
                    {
                        "id": identifier,
                        "split": split,
                        "has_page": has_page,
                        "document_group": group,
                        "capture_session": pair_id,
                        "negative_type": None if has_page else "paper_panel",
                        "size_bin": "medium" if has_page else "negative",
                        "background_type": "gradient",
                        "pair_id": pair_id,
                        "pair_role": role,
                        "partner_id": partner,
                    }
                )
        for has_page in (True, False):
            identifier = f"{split}-unpaired-{has_page}"
            manifest["samples"].append(
                {
                    "id": identifier,
                    "split": split,
                    "has_page": has_page,
                    "document_group": identifier,
                    "capture_session": identifier,
                    "negative_type": None if has_page else "background",
                    "size_bin": "medium" if has_page else "negative",
                    "background_type": "gradient",
                    "pair_id": None,
                    "pair_role": None,
                    "partner_id": None,
                }
            )
    return manifest


def _report(manifest, split="validation"):
    records = [row for row in manifest["samples"] if row["split"] == split]
    return {
        "schema_version": "region-study-evaluation-0.2",
        "dataset_schema": "scene-0.3",
        "split": split,
        "expected_count": len(records),
        "gate_passed": True,
        "methods": {
            "learned_postprocessed": {
                "evaluation_grid": "original_source_pixels",
                "samples": [
                    {
                        "id": row["id"],
                        "has_page": row["has_page"],
                        "status": "completed",
                        "detected_page": row["has_page"],
                        "iou": 0.9 if row["has_page"] else 1.0,
                    }
                    for row in records
                ],
            }
        },
        "region_decisions": [
            {"id": row["id"], "status": "accepted" if row["has_page"] else "rejected"}
            for row in records
        ],
    }


def _result(report, identifier, status, *, iou=None):
    row = next(
        row
        for row in report["methods"]["learned_postprocessed"]["samples"]
        if row["id"] == identifier
    )
    decision = next(row for row in report["region_decisions"] if row["id"] == identifier)
    decision["status"] = status
    row["status"] = "failed" if status == "failed" else "completed"
    if status == "failed":
        row.pop("iou", None)
        row.pop("detected_page", None)
    else:
        row["detected_page"] = status == "accepted"
        row["iou"] = (
            iou
            if iou is not None
            else 0.9
            if row["has_page"] and status == "accepted"
            else float(not row["has_page"] and status == "rejected")
        )


def _diagnostics(manifest, report):
    return paired.paired_region_diagnostics(manifest, report)


def test_all_declared_pairs_are_accounted_and_secondary_threshold_is_inclusive():
    manifest = _manifest(6)
    report = _report(manifest)
    _result(report, "validation-pair-0-printed", "accepted", iou=0.8)
    _result(report, "validation-pair-1-blank", "accepted")
    _result(report, "validation-pair-2-printed", "rejected")
    _result(report, "validation-pair-3-printed", "rejected")
    _result(report, "validation-pair-3-blank", "accepted")
    _result(report, "validation-pair-4-blank", "failed")
    report["region_decisions"] = [
        row for row in report["region_decisions"] if row["id"] != "validation-pair-5-blank"
    ]
    original = copy.deepcopy(report)
    result = _diagnostics(manifest, report)
    assert result["expected_pair_count"] == 6
    assert result["completed_pair_count"] == 4
    assert result["failed_pair_count"] == 2
    for outcome in paired._OUTCOMES:
        assert result[outcome + "_count"] == 1
    assert result["printed_accepted_blank_rejected_rate"] == 1 / 6
    assert result["printed_iou_ge_0_8_and_blank_rejected_count"] == 1
    assert result["printed_iou_ge_0_8_and_blank_rejected_rate"] == 1 / 6
    assert result["missing_sample_ids"] == ["validation-pair-5-blank"]
    assert result["error_sample_ids"] == ["validation-pair-4-blank"]
    assert result["additional_gate"] is False
    assert result["used_for_model_selection"] is False
    assert not result["complete_selected_split"]
    assert result["unpaired_sample_count"] == 2
    assert report == original


def test_secondary_uses_post_iou_not_raw_and_not_just_acceptance():
    manifest, report = _manifest(1), _report(_manifest(1))
    _result(report, "validation-pair-0-printed", "accepted", iou=0.799999)
    report["methods"]["learned_raw_threshold"] = {
        "samples": [{"id": "validation-pair-0-printed", "iou": 1.0}]
    }
    result = _diagnostics(manifest, report)
    assert result["printed_accepted_blank_rejected_count"] == 1
    assert result["printed_iou_ge_0_8_and_blank_rejected_count"] == 0


def test_default_twenty_four_pairs_and_global_preprocessing_qa_are_reported():
    manifest = _manifest(24)
    manifest["preprocessing_qa"] = {
        "validation": {
            "comparisons_checked": 24,
            "matched_pairs_checked": 24,
            "byte_identical_after_128x96": 2,
            "excluded_or_regenerated": 0,
        }
    }
    result = _diagnostics(manifest, _report(manifest))
    assert result["expected_pair_count"] == 24
    assert result["completed_pair_count"] == 24
    assert result["printed_accepted_blank_rejected_count"] == 24
    assert result["preprocessing_qa"] == manifest["preprocessing_qa"]["validation"]
    assert all(row["printed_size_bin"] == "medium" for row in result["samples"])
    assert all("identical_input" not in key for row in result["samples"] for key in row)


@pytest.mark.parametrize("key", ["samples", "pairs"])
def test_nonobject_manifest_rows_raise_deliberate_value_error(key):
    manifest = _manifest(1)
    report = _report(manifest)
    manifest[key].append(None)
    with pytest.raises(ValueError, match="object rows"):
        _diagnostics(manifest, report)


def test_blank_panel_cannot_silently_become_an_unpaired_negative():
    manifest = _manifest(2)
    report = _report(manifest)
    manifest["pairs"] = [row for row in manifest["pairs"] if row["id"] != "validation-pair-0"]
    for row in manifest["samples"]:
        if row["pair_id"] == "validation-pair-0":
            row.update(pair_id=None, pair_role=None, partner_id=None)
    with pytest.raises(ValueError, match="missing its declared pair"):
        _diagnostics(manifest, report)


@pytest.mark.parametrize(
    "missing",
    [
        "printed_row",
        "blank_row",
        "printed_decision",
        "blank_decision",
        "whole_method",
        "whole_decisions",
    ],
)
def test_missing_results_stay_in_fixed_pair_denominator(missing):
    manifest, report = _manifest(2), _report(_manifest(2))
    expected = 1
    if missing == "whole_method":
        report["methods"].pop("learned_postprocessed")
        expected = 2
    elif missing == "whole_decisions":
        report.pop("region_decisions")
        expected = 2
    else:
        role, kind = missing.split("_")
        key = "region_decisions" if kind == "decision" else "samples"
        owner = report if kind == "decision" else report["methods"]["learned_postprocessed"]
        owner[key] = [row for row in owner[key] if row["id"] != f"validation-pair-0-{role}"]
    result = _diagnostics(manifest, report)
    assert result["expected_pair_count"] == 2
    assert result["missing_count"] == expected
    assert result["printed_iou_ge_0_8_and_blank_rejected_rate"] == (2 - expected) / 2


@pytest.mark.parametrize("role", ["printed", "blank"])
def test_failed_inference_is_never_a_successful_rejection(role):
    manifest, report = _manifest(1), _report(_manifest(1))
    _result(report, f"validation-pair-0-{role}", "failed")
    result = _diagnostics(manifest, report)
    assert result["error_count"] == 1
    assert result["printed_accepted_blank_rejected_count"] == 0
    assert result["printed_iou_ge_0_8_and_blank_rejected_count"] == 0


def test_missing_takes_precedence_when_other_pair_member_failed():
    manifest, report = _manifest(1), _report(_manifest(1))
    _result(report, "validation-pair-0-printed", "failed")
    report["region_decisions"] = [
        row for row in report["region_decisions"] if row["id"] != "validation-pair-0-blank"
    ]
    result = _diagnostics(manifest, report)
    assert result["missing_count"] == 1
    assert result["error_count"] == 0
    assert result["error_sample_ids"] == ["validation-pair-0-printed"]


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "duplicate_sample",
        "duplicate_pair",
        "missing_partner",
        "self_partner",
        "reciprocal",
        "role",
        "label",
        "group",
        "negative_type",
        "undeclared",
        "missing_pair",
        "cross_split",
        "empty",
    ],
)
def test_malformed_pair_metadata_is_rejected(change):
    manifest, report = _manifest(2), _report(_manifest(2))
    record = next(row for row in manifest["samples"] if row["id"] == "validation-pair-0-printed")
    declaration = next(row for row in manifest["pairs"] if row["id"] == "validation-pair-0")
    if change == "schema":
        manifest["schema_version"] = "scene-0.2"
    elif change == "duplicate_sample":
        manifest["samples"].append(copy.deepcopy(record))
    elif change == "duplicate_pair":
        manifest["pairs"].append(copy.deepcopy(declaration))
    elif change == "missing_partner":
        manifest["samples"] = [
            row for row in manifest["samples"] if row["id"] != declaration["blank_id"]
        ]
    elif change == "self_partner":
        declaration["blank_id"] = declaration["printed_id"]
    elif change == "reciprocal":
        record["partner_id"] = "validation-pair-1-blank"
    elif change == "role":
        record["pair_role"] = "blank"
    elif change == "label":
        record["has_page"] = False
    elif change == "group":
        record["document_group"] = "other"
    elif change == "negative_type":
        record["negative_type"] = "paper_panel"
    elif change == "undeclared":
        next(row for row in manifest["samples"] if row["id"] == "validation-unpaired-True")[
            "pair_id"
        ] = "undeclared"
    elif change == "missing_pair":
        manifest["pairs"].remove(declaration)
    elif change == "cross_split":
        declaration["blank_id"] = "test-pair-0-blank"
    else:
        manifest["pairs"] = []
    with pytest.raises(ValueError):
        _diagnostics(manifest, report)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.01, True, 10**1000])
def test_nonfinite_or_invalid_iou_cannot_be_a_success(value):
    manifest, report = _manifest(1), _report(_manifest(1))
    report["methods"]["learned_postprocessed"]["samples"][0]["iou"] = value
    with pytest.raises(ValueError, match="finite"):
        _diagnostics(manifest, report)


@pytest.mark.parametrize("status,iou", [("accepted", 0.9), ("rejected", 0.4)])
def test_negative_iou_must_match_fixed_empty_mask_convention(status, iou):
    manifest, report = _manifest(1), _report(_manifest(1))
    _result(report, "validation-pair-0-blank", status, iou=iou)
    with pytest.raises(ValueError, match="empty-mask convention"):
        _diagnostics(manifest, report)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_row",
        "duplicate_decision",
        "extra_id",
        "wrong_label",
        "wrong_detected",
        "wrong_grid",
        "wrong_split",
        "wrong_count",
    ],
)
def test_contradictory_result_metadata_is_rejected(change):
    manifest, report = _manifest(1), _report(_manifest(1))
    method = report["methods"]["learned_postprocessed"]
    if change == "duplicate_row":
        method["samples"].append(copy.deepcopy(method["samples"][0]))
    elif change == "duplicate_decision":
        report["region_decisions"].append(copy.deepcopy(report["region_decisions"][0]))
    elif change == "extra_id":
        report["region_decisions"].append({"id": "test-pair-0-printed", "status": "accepted"})
    elif change == "wrong_label":
        method["samples"][0]["has_page"] = False
    elif change == "wrong_detected":
        method["samples"][0]["detected_page"] = False
    elif change == "wrong_grid":
        method["evaluation_grid"] = "model_input"
    elif change == "wrong_split":
        report["split"] = "test"
    else:
        report["expected_count"] -= 1
    with pytest.raises(ValueError):
        _diagnostics(manifest, report)


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    """Exercise the real generic evaluator, but never load any image from disk."""
    manifest = _manifest(1)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    positive = np.zeros((10, 10), dtype=bool)
    positive[2:8, 2:8] = True
    probability = positive.astype(np.float32) * 0.9
    decoded, observed_inputs = [], []
    samples = {}
    for number, record in enumerate(manifest["samples"]):
        samples[record["id"]] = SimpleNamespace(
            image=np.full((10, 10, 3), number, dtype=np.uint8),
            mask=positive.copy() if record["has_page"] else np.zeros_like(positive),
        )

    def load(directory, record):
        assert record["split"] == "validation", "unselected test or train image was decoded"
        decoded.append(record["id"])
        return samples[record["id"]]

    def predict(image):
        observed_inputs.append(image)
        record = manifest["samples"][int(image[0, 0, 0])]
        if record["id"] == "validation-unpaired-True":
            raise ValueError("deliberate non-pair inference failure")
        return probability.copy() if record["has_page"] else np.zeros_like(probability)

    loaded = SimpleNamespace(
        metadata={"manifest_sha256": digest, "epoch": 2},
        model_id="fixture-context",
        model_sha256="a" * 64,
        predict_probability=predict,
    )

    def validate(root):
        return {"manifest": manifest}

    monkeypatch.setattr(paired, "validate_scene_dataset", validate)
    monkeypatch.setattr(study, "validate_scene_dataset", validate)
    monkeypatch.setattr(study, "scene_loader", lambda root: load)
    monkeypatch.setattr(study, "load_region_model", lambda path: loaded)
    provenance = {
        "source_code_sha256": {"fixture.py": "f" * 64},
        "versions": {"python": "fixture", "torch": "fixture"},
        "device": "cpu",
        "preprocess_version": "fixture",
    }
    monkeypatch.setattr(paired, "_evaluation_provenance", lambda: copy.deepcopy(provenance))
    return SimpleNamespace(
        root=tmp_path,
        manifest=manifest,
        digest=digest,
        decoded=decoded,
        observed_inputs=observed_inputs,
    )


def _evaluate(data, **kwargs):
    return paired.evaluate_paired_regions(data.root, data.root / "fixture.pt", **kwargs)


def test_wrapper_preserves_whole_split_primary_report_and_does_not_decode_test(fixture_data):
    data = fixture_data
    original = study.evaluate_region_study(data.root, data.root / "fixture.pt")
    data.decoded.clear()
    before_rng, before_threads = torch.get_rng_state().clone(), torch.get_num_threads()
    result = _evaluate(data)
    diagnostics = result.pop("paired_diagnostics")
    provenance = result.pop("evaluation_provenance")
    assert set(provenance) == {"source_code_sha256", "versions", "device", "preprocess_version"}
    assert result == original
    assert result["gate_passed"] is False  # Unpaired positive failure remains in the primary gate.
    assert diagnostics["printed_iou_ge_0_8_and_blank_rejected_count"] == 1
    assert diagnostics["error_sample_ids"] == ["validation-unpaired-True"]
    assert not diagnostics["complete_selected_split"]
    assert len(data.decoded) == 4
    assert all(identifier.startswith("validation-") for identifier in data.decoded)
    assert all(isinstance(value, np.ndarray) for value in data.observed_inputs)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_threads == torch.get_num_threads()


def test_output_is_standard_named_json_and_cannot_be_overwritten(fixture_data, monkeypatch):
    data = fixture_data
    output = data.root / "evaluation"
    report = _evaluate(data, output_dir=output)
    destination = output / "region-study-validation.json"
    assert json.loads(destination.read_text()) == report
    original = destination.read_bytes()
    monkeypatch.setattr(
        paired,
        "validate_scene_dataset",
        lambda *args: pytest.fail("existing report not rejected first"),
    )
    with pytest.raises(FileExistsError):
        _evaluate(data, output_dir=output)
    assert destination.read_bytes() == original


def test_bad_scene_schema_is_rejected_before_inference(fixture_data):
    data = fixture_data
    data.manifest["schema_version"] = "scene-0.2"
    with pytest.raises(ValueError, match="scene-0.3"):
        _evaluate(data)
    assert data.decoded == []


def test_evaluation_provenance_factory_includes_only_evaluator_source_and_runtime(monkeypatch):
    values = {
        "source_code_sha256": {"module.py": "d" * 64},
        "versions": {"python": "3.fixture", "torch": "fixture"},
        "device": "cpu",
        "preprocess_version": "fixture",
        "uv_lock_sha256": "e" * 64,
        "model_version": "not-an-evaluator-property",
        "loss": {"not": "an evaluator property"},
        "selection": {"not": "an evaluator property"},
    }
    monkeypatch.setattr(paired, "_provenance", lambda: values)
    assert paired._evaluation_provenance() == {
        key: values[key]
        for key in ("source_code_sha256", "versions", "device", "preprocess_version")
    }


@pytest.mark.parametrize("key", ["source_code_sha256", "versions", "device", "preprocess_version"])
def test_changed_evaluation_provenance_cannot_publish(fixture_data, monkeypatch, key):
    before = paired._evaluation_provenance()
    after = copy.deepcopy(before)
    after[key] = "changed during evaluation"
    snapshots = iter([before, after])
    monkeypatch.setattr(paired, "_evaluation_provenance", lambda: next(snapshots))
    output = fixture_data.root / "evaluation"
    with pytest.raises(ValueError, match="source or runtime changed"):
        _evaluate(fixture_data, output_dir=output)
    assert not (output / "region-study-validation.json").exists()


def test_generic_evaluator_gets_no_output_path_and_all_options(fixture_data, monkeypatch):
    data = fixture_data
    calls = []

    def evaluate(dataset, checkpoint, **kwargs):
        calls.append((dataset, checkpoint, kwargs))
        report = _report(data.manifest, split="test")
        report["manifest_sha256"] = data.digest
        return report

    monkeypatch.setattr(paired, "evaluate_region_study", evaluate)
    result = _evaluate(data, split="test", expected_training_manifest="b" * 64, threads=2)
    assert result["split"] == "test"
    assert calls[0][2] == {
        "split": "test",
        "output_dir": None,
        "expected_training_manifest": "b" * 64,
        "threads": 2,
    }


def test_manifest_change_during_evaluation_cannot_publish(fixture_data, monkeypatch):
    data = fixture_data
    original = paired.evaluate_region_study

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        (data.root / "manifest.json").write_text("changed after evaluation")
        return report

    monkeypatch.setattr(paired, "evaluate_region_study", changed)
    with pytest.raises(ValueError, match="manifest identities"):
        _evaluate(data, output_dir=data.root / "evaluation")
    assert not (data.root / "evaluation" / "region-study-validation.json").exists()


def test_competing_publication_keeps_winner_and_cleans_temporary(fixture_data, monkeypatch):
    original_link = paired.os.link
    temporary_paths = []
    winner = b'{"other":"writer"}\n'

    def race(source, destination):
        temporary_paths.append(Path(source))
        Path(destination).write_bytes(winner)
        original_link(source, destination)

    monkeypatch.setattr(paired.os, "link", race)
    output = fixture_data.root / "evaluation"
    with pytest.raises(FileExistsError):
        _evaluate(fixture_data, output_dir=output)
    assert (output / "region-study-validation.json").read_bytes() == winner
    assert temporary_paths and all(not path.exists() for path in temporary_paths)


def test_serialization_failure_leaves_no_partial_report(fixture_data, monkeypatch):
    original = paired.evaluate_region_study

    def nonfinite(*args, **kwargs):
        report = original(*args, **kwargs)
        report["invalid_extra"] = float("nan")
        return report

    monkeypatch.setattr(paired, "evaluate_region_study", nonfinite)
    output = fixture_data.root / "evaluation"
    with pytest.raises(ValueError):
        _evaluate(fixture_data, output_dir=output)
    assert not (output / "region-study-validation.json").exists()


def test_fsync_failure_leaves_no_partial_report_or_temporary(fixture_data, monkeypatch):
    def failed_sync(descriptor):
        raise OSError("fixture disk sync failure")

    monkeypatch.setattr(paired.os, "fsync", failed_sync)
    output = fixture_data.root / "evaluation"
    with pytest.raises(OSError, match="disk sync"):
        _evaluate(fixture_data, output_dir=output)
    assert not (output / "region-study-validation.json").exists()
    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    "options", [{"split": "train"}, {"threads": True}, {"threads": 0}, {"threads": 129}]
)
def test_invalid_configuration_is_rejected_before_image_load(fixture_data, options):
    with pytest.raises(ValueError):
        _evaluate(fixture_data, **options)
    assert fixture_data.decoded == []
