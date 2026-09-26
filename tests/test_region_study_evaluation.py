"""Independent split/provenance/denominator tests using in-memory scene fixtures.

No production scene image, mask, or held-out test label is opened by this module.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flattener.contracts import FlattenerError, Status
from flattener.regions import study_evaluation as study


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"fixture":"in-memory study evaluation only"}', encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    positive = np.zeros((10, 10), dtype=bool)
    positive[2:8, 2:8] = True
    negative = np.zeros_like(positive)
    good = np.zeros((5, 5), dtype=np.float32)
    good[1:4, 1:4] = 0.9
    partial = np.zeros_like(good)
    partial[1:4, 1:3] = 0.9
    predictions = {
        1: good,
        2: partial,
        3: ValueError("fixture positive inference failure"),
        4: good,
        5: np.zeros_like(good),
        6: RuntimeError("fixture negative inference failure"),
    }
    records, samples = [], {}
    for split in ("train", "validation", "test"):
        for code in range(1, 7):
            identifier = f"{split}-{code}"
            has_page = code <= 3
            records.append(
                {
                    "id": identifier,
                    "split": split,
                    "has_page": has_page,
                    "background_type": "printed-background" if has_page else "clutter",
                    "document_group": f"document-{identifier}",
                    "capture_session": f"capture-{identifier}",
                    "negative_type": None
                    if has_page
                    else "texture"
                    if code == 5
                    else "blank-panel",
                    "size_bin": "medium" if has_page else None,
                }
            )
            samples[identifier] = SimpleNamespace(
                image=np.full((10, 10, 3), code, dtype=np.uint8),
                mask=(positive if has_page else negative).copy(),
            )
    decoded, backend_requests, validation_requests, model_requests, model_threads = (
        [],
        [],
        [],
        [],
        [],
    )

    def load(root, record):
        decoded.append(record["id"])
        value = samples[record["id"]]
        if isinstance(value, BaseException):
            raise value
        return value

    def loader(root):
        backend_requests.append(root)
        return load

    def predict(image):
        model_threads.append(torch.get_num_threads())
        response = predictions[int(image[0, 0, 0])]
        if isinstance(response, BaseException):
            raise response
        return response.copy() if isinstance(response, np.ndarray) else response

    loaded = SimpleNamespace(
        metadata={"manifest_sha256": digest, "epoch": 17},
        model_sha256="a" * 64,
        model_id="fixture-region-study",
        predict_probability=predict,
    )
    manifest = {
        "schema_version": "scene-0.2",
        "task_scope": "printed-page fixture",
        "samples": records,
    }

    def validate(root):
        validation_requests.append(root)
        return {"manifest": manifest}

    def model_loader(path):
        model_requests.append(path)
        return loaded

    monkeypatch.setattr(study, "validate_scene_dataset", validate)
    monkeypatch.setattr(study, "scene_loader", loader)
    monkeypatch.setattr(study, "load_region_model", model_loader)
    return SimpleNamespace(
        root=tmp_path,
        digest=digest,
        loaded=loaded,
        manifest=manifest,
        records=records,
        predictions=predictions,
        samples=samples,
        decoded=decoded,
        backend_requests=backend_requests,
        validation_requests=validation_requests,
        model_requests=model_requests,
        model_threads=model_threads,
    )


def run(data, **options):
    return study.evaluate_region_study(data.root, data.root / "fixture-checkpoint.pt", **options)


def test_same_training_manifest_is_allowed_by_default_and_report_identifies_it(fixture_data):
    data = fixture_data
    report = run(data)
    assert report["manifest_sha256"] == report["training_manifest_sha256"] == data.digest
    assert report["explicit_transfer_evaluation"] is False
    assert report["expected_count"] == 6
    assert report["checkpoint_epoch"] == 17
    assert report["checkpoint_sha256"] == "a" * 64
    assert report["model_id"] == "fixture-region-study"
    assert report["dataset_schema"] == "scene-0.2"
    assert report["dataset_task_scope"] == "printed-page fixture"
    assert report["test_used_for_model_selection"] is False


def test_manifest_mismatch_is_rejected_by_default_before_any_image_decode(fixture_data):
    data = fixture_data
    data.loaded.metadata["manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="declared training dataset"):
        run(data)
    assert data.decoded == []
    assert data.backend_requests == []


def test_explicit_transfer_requires_the_exact_old_training_digest(fixture_data):
    data = fixture_data
    data.loaded.metadata["manifest_sha256"] = "b" * 64
    before = data.loaded.metadata.copy()
    with pytest.raises(ValueError, match="declared training dataset"):
        run(data, expected_training_manifest="c" * 64)
    assert data.decoded == []
    report = run(data, expected_training_manifest="b" * 64)
    assert report["manifest_sha256"] == data.digest
    assert report["training_manifest_sha256"] == "b" * 64
    assert report["explicit_transfer_evaluation"] is True
    assert data.loaded.metadata == before


def test_explicit_same_manifest_declaration_is_not_mislabeled_as_transfer(fixture_data):
    report = run(fixture_data, expected_training_manifest=fixture_data.digest)
    assert report["explicit_transfer_evaluation"] is False


@pytest.mark.parametrize("split", ["validation", "test"])
def test_only_selected_split_is_decoded_and_backend_is_resolved_once(fixture_data, split):
    data = fixture_data
    report = run(data, split=split)
    assert data.decoded == [f"{split}-{code}" for code in range(1, 7)]
    assert data.backend_requests == [data.root.resolve()]
    assert report["split"] == split
    assert {row["id"] for row in report["methods"]["learned_postprocessed"]["samples"]} == set(
        data.decoded
    )


def test_hand_computed_iou_and_negative_rates_keep_inference_failures_in_denominators(fixture_data):
    report = run(fixture_data)
    assert report["region_accepted_count"] == 3
    assert report["region_rejected_count"] == 1
    assert report["region_failure_count"] == 2
    assert len(report["inference_errors"]) == 2
    assert report["gate_passed"] is False
    for name in ("learned_raw_threshold", "learned_postprocessed"):
        summary = report["methods"][name]["summary"]
        assert summary["expected_count"] == 6
        assert summary["completed_count"] == 4
        assert summary["failed_count"] == 2
        assert summary["foreground_count"] == 3
        assert summary["negative_count"] == 3
        # Independent masks: 36/36, 24/36, then failed-positive zero => 5/9.
        assert summary["foreground_iou_all_cases"]["mean"] == pytest.approx(5 / 9)
        assert summary["foreground_iou_completed_only"]["mean"] == pytest.approx(5 / 6)
        # One accepted negative, one correct rejection, one failed inference.
        assert summary["negative_false_positive_count_conservative"] == 2
        assert summary["negative_false_positive_rate_conservative"] == pytest.approx(2 / 3)
        assert summary["negative_detection_count_completed_only"] == 1
    rows = {row["id"]: row for row in report["methods"]["learned_postprocessed"]["samples"]}
    partial = rows["validation-2"]
    assert partial["true_positive_pixels"] == 24
    assert partial["false_negative_pixels"] == 12
    assert partial["false_positive_pixels"] == 0
    assert partial["precision"] == 1
    assert partial["recall"] == pytest.approx(2 / 3)
    assert rows["validation-4"]["false_positive_pixels"] == 36
    assert rows["validation-4"]["negative_pixel_false_positive_rate"] == 0.36
    assert rows["validation-5"]["status"] == "completed"
    assert rows["validation-5"]["detected_page"] is False


def test_negative_type_breakdown_excludes_positives_and_retains_every_negative(fixture_data):
    report = run(fixture_data)
    for name, method in report["methods"].items():
        groups = method["by_negative_type"]
        assert set(groups) == {"blank-panel", "texture"}
        assert sum(group["expected_count"] for group in groups.values()) == 3
        assert all(group["foreground_count"] == 0 for group in groups.values())
        assert groups["blank-panel"]["negative_count"] == 2
        assert groups["texture"]["negative_count"] == 1
        if name.startswith("learned"):
            assert groups["blank-panel"]["failed_count"] == 1
            assert groups["blank-panel"]["negative_false_positive_rate_conservative"] == 1
            assert groups["texture"]["negative_false_positive_rate_conservative"] == 0
        else:
            assert groups["blank-panel"]["failed_count"] == 0
            assert groups["texture"]["negative_false_positive_rate_conservative"] == 1


def test_legacy_records_without_negative_type_are_still_fully_accounted(fixture_data):
    data = fixture_data
    data.manifest["schema_version"] = "scene-0.1"
    data.manifest.pop("task_scope")
    for record in data.records:
        record.pop("negative_type")
        record.pop("size_bin")
    report = run(data, output_dir=data.root / "reports")
    assert report["dataset_schema"] == "scene-0.1"
    assert report["dataset_task_scope"] is None
    groups = report["methods"]["learned_postprocessed"]["by_negative_type"]
    assert len(groups) == 1
    assert next(iter(groups.values()))["negative_count"] == 3
    saved = json.loads((data.root / "reports/region-study-validation.json").read_text())
    assert saved["expected_count"] == 6


@pytest.mark.parametrize(
    "failure",
    [
        FlattenerError(Status.MODEL_ERROR, "fixture", "fixture model failure"),
        TypeError("fixture wrong type"),
        ValueError("fixture invalid value"),
        RuntimeError("fixture execution failure"),
    ],
)
def test_each_expected_inference_error_is_recorded_without_losing_cases(fixture_data, failure):
    data = fixture_data

    def failed(image):
        raise failure

    data.loaded.predict_probability = failed
    report = run(data)
    assert report["expected_count"] == report["region_failure_count"] == 6
    assert len(report["inference_errors"]) == 6
    summary = report["methods"]["learned_postprocessed"]["summary"]
    assert summary["foreground_iou_all_cases"]["mean"] == 0
    assert summary["negative_false_positive_rate_conservative"] == 1
    assert summary["failed_count"] == 6


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        np.full((5, 5), np.nan, np.float32),
        np.full((5, 5), 1.2, np.float32),
        np.zeros((5, 5), np.uint8),
    ],
)
def test_malformed_probability_outputs_fail_without_shrinking_denominators(fixture_data, invalid):
    data = fixture_data
    for code in range(1, 7):
        data.predictions[code] = invalid
    report = run(data)
    assert report["region_failure_count"] == 6
    assert report["gate_passed"] is False
    assert report["methods"]["learned_postprocessed"]["summary"]["negative_count"] == 3


def test_thread_count_is_used_during_inference_and_restored_after_normal_run(fixture_data):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        report = run(fixture_data, threads=1)
        assert fixture_data.model_threads == [1] * 6
        assert report["cpu_threads"] == 1
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "failure", [MemoryError("fixture memory failure"), OSError("fixture unreadable GT")]
)
def test_unavailable_ground_truth_aborts_and_restores_threads(fixture_data, failure):
    data = fixture_data
    data.samples["validation-1"] = failure
    previous = torch.get_num_threads()
    with pytest.raises(type(failure)):
        run(data, threads=1, output_dir=data.root / "reports")
    assert torch.get_num_threads() == previous
    assert not (data.root / "reports/region-study-validation.json").exists()


@pytest.mark.parametrize(
    "message", ["CUDA out of memory", "DefaultCPUAllocator: can't allocate memory"]
)
def test_allocation_errors_abort_instead_of_becoming_model_quality_failures(fixture_data, message):
    data = fixture_data
    data.predictions[1] = RuntimeError(message)
    previous = torch.get_num_threads()
    with pytest.raises(MemoryError):
        run(data, threads=1, output_dir=data.root / "reports")
    assert torch.get_num_threads() == previous
    assert not (data.root / "reports/region-study-validation.json").exists()


def test_existing_report_is_never_overwritten_and_preflight_avoids_loading_models(fixture_data):
    data = fixture_data
    report = run(data, output_dir=data.root / "reports")
    path = data.root / "reports/region-study-validation.json"
    before = path.read_bytes()
    assert json.loads(before)["expected_count"] == report["expected_count"]
    calls = (len(data.decoded), len(data.model_requests), len(data.validation_requests))
    with pytest.raises(FileExistsError):
        run(data, output_dir=data.root / "reports")
    assert (len(data.decoded), len(data.model_requests), len(data.validation_requests)) == calls
    assert path.read_bytes() == before


def test_empty_selected_split_is_rejected_before_loader_resolution(fixture_data):
    data = fixture_data
    data.records[:] = [record for record in data.records if record["split"] != "validation"]
    with pytest.raises(ValueError, match="empty"):
        run(data)
    assert data.decoded == []
    assert data.backend_requests == []


@pytest.mark.parametrize(
    "options",
    [
        {"split": "train"},
        {"split": "holdout"},
        {"threads": True},
        {"threads": 0},
        {"threads": 129},
        {"threads": 1.0},
    ],
)
def test_invalid_options_fail_before_dataset_or_model_access(fixture_data, options):
    data = fixture_data
    with pytest.raises(ValueError):
        run(data, **options)
    assert data.validation_requests == []
    assert data.model_requests == []


@pytest.mark.parametrize("digest", ["", "0" * 63, "0" * 65, "g" * 64, "A" * 64, " " * 64, 123])
def test_transfer_declaration_must_be_a_valid_sha256_not_only_matching_metadata(
    fixture_data, digest
):
    data = fixture_data
    data.loaded.metadata["manifest_sha256"] = digest
    with pytest.raises(ValueError, match="manifest|digest|SHA|training"):
        run(data, expected_training_manifest=digest)
    assert data.decoded == []
