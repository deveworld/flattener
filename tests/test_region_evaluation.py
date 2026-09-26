"""Independent pixel/edge answers and failure-inclusive region evaluation."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from flattener.regions.evaluation import (
    MaskPrediction,
    RegionCase,
    crop_metrics,
    evaluate_baselines,
    evaluate_masks,
    evaluate_probabilities,
    evaluate_region_checkpoint,
    mask_box,
    mask_metrics,
    mask_to_source,
    page_size_category,
    rescale_box,
)


def test_confusion_counts_have_independent_hand_computed_answer():
    truth = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)
    prediction = np.array([[1, 0, 1], [0, 0, 1]], dtype=bool)
    result = mask_metrics(prediction, truth)
    assert result["true_positive_pixels"] == 2
    assert result["false_positive_pixels"] == 1
    assert result["false_negative_pixels"] == 1
    assert result["true_negative_pixels"] == 2
    assert result["iou"] == 0.5
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)


def test_empty_prediction_on_positive_cannot_inflate_precision_or_recall():
    truth = np.ones((3, 5), dtype=bool)
    result = mask_metrics(np.zeros_like(truth), truth)
    for key in ("iou", "precision", "recall", "gt_foreground_area_retained_fraction"):
        assert result[key] == 0
    assert result["gt_box_contained"] is False
    assert result["crop_box"] is None


def test_empty_empty_is_explicit_not_foreground_quality():
    empty = np.zeros((3, 5), dtype=bool)
    result = mask_metrics(empty, empty)
    assert result["iou"] == result["precision"] == result["recall"] == 1
    assert result["negative_pixel_false_positive_rate"] == 0
    assert result["gt_box_contained"] is None
    report = evaluate_masks([RegionCase("negative", empty)], {"negative": empty})
    assert report["summary"]["foreground_iou_all_cases"]["mean"] is None
    assert report["summary"]["gate_passed"] is False


def test_one_false_positive_pixel_is_negative_image_detection():
    truth = np.zeros((100, 100), dtype=bool)
    prediction = truth.copy()
    prediction[50, 50] = True
    report = evaluate_masks([RegionCase("negative", truth)], {"negative": prediction})
    assert report["samples"][0]["negative_pixel_false_positive_rate"] == 0.0001
    assert report["summary"]["negative_false_positive_rate_conservative"] == 1


def test_mask_box_uses_pixel_edges_and_single_pixels_have_unit_area():
    mask = np.zeros((5, 9), dtype=bool)
    mask[3, 7] = True
    assert mask_box(mask) == (7, 3, 8, 4)
    assert crop_metrics(mask, (7, 3, 8, 4))["gt_box_area_pixels"] == 1
    assert mask_box(np.zeros_like(mask)) is None


def test_bbox_resizing_scales_edges_without_half_pixel_translation():
    assert rescale_box((0, 0, 10, 20), (20, 10), (150, 300)) == (0, 0, 300, 150)
    assert rescale_box((2, 4, 6, 8), (20, 10), (100, 200)) == (40, 20, 120, 40)
    # First source pixel spans [0, 1], not [-.5, .5] in edge coordinates.
    assert rescale_box((0, 0, 1, 1), (20, 10), (40, 50)) == (0, 0, 5, 2)


def test_fractional_crop_tracks_fractional_foreground_pixel_area():
    truth = np.zeros((4, 6), dtype=bool)
    truth[1:3, 2:4] = True
    result = crop_metrics(truth, (2.5, 1.0, 4.0, 3.0))
    assert result["gt_box_area_pixels"] == 4
    assert result["crop_area_pixels"] == 3
    assert result["gt_box_area_retained_fraction"] == 0.75
    assert result["gt_foreground_area_retained_fraction"] == 0.75
    assert result["gt_box_contained"] is False
    assert result["excess_crop_area_over_gt_box"] == 0


def test_irregular_mask_retention_differs_from_its_box_area():
    truth = np.eye(4, dtype=bool)
    result = crop_metrics(truth, (0, 0, 2, 2))
    assert result["gt_box_area_retained_fraction"] == 0.25
    assert result["gt_foreground_area_retained_fraction"] == 0.5


def test_oversized_crop_preserves_bbox_but_pays_excess_area():
    truth = np.zeros((10, 10), dtype=bool)
    truth[3:7, 3:7] = True
    result = crop_metrics(truth, (0, 0, 10, 10))
    assert result["gt_box_contained"] is True
    assert result["gt_box_area_retained_fraction"] == 1
    assert result["crop_area_over_gt_box"] == 6.25
    assert result["excess_crop_area_over_gt_box"] == 5.25


def test_custom_crop_is_measured_separately_from_mask_quality():
    truth = np.ones((5, 5), dtype=bool)
    result = mask_metrics(truth, truth, crop_box=(1, 1, 4, 4))
    assert result["iou"] == 1
    assert result["gt_box_contained"] is False
    assert result["gt_foreground_area_retained_fraction"] == 9 / 25


@pytest.mark.parametrize("bad", [None, [], [[True]], np.ones((3, 3)), np.ones((0, 3), dtype=bool)])
def test_masks_must_be_nonempty_boolean_arrays(bad):
    with pytest.raises(ValueError, match="boolean array"):
        mask_metrics(bad, np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="boolean array"):
        mask_metrics(np.ones((3, 3), dtype=bool), bad)


def test_shape_mismatch_is_not_resized_silently():
    with pytest.raises(ValueError, match="source-grid"):
        mask_metrics(np.ones((2, 3), dtype=bool), np.ones((3, 2), dtype=bool))


@pytest.mark.parametrize(
    "bad",
    [
        (-1, 0, 3, 3),
        (0, 0, 4, 3),
        (0, 0, 0, 2),
        (0, 3, 2, 2),
        (0, 0, np.nan, 3),
        (0, 0, np.inf, 3),
        (0, 0, 3),
        "box",
    ],
)
def test_crop_outside_image_and_nonfinite_boxes_are_rejected(bad):
    with pytest.raises(ValueError, match="crop_box"):
        crop_metrics(np.ones((3, 3), dtype=bool), bad)


@pytest.mark.parametrize("bad", [(0, 3), (True, 3), (3.0, 3), (2,), "2x3"])
def test_rescale_requires_integer_sizes(bad):
    with pytest.raises(ValueError, match="source_size"):
        rescale_box((0, 0, 1, 1), bad, (10, 20))
    with pytest.raises(ValueError, match="destination_size"):
        rescale_box((0, 0, 1, 1), (10, 20), bad)


@pytest.mark.parametrize(
    "count,expected",
    [(0, "negative"), (24, "small"), (25, "medium"), (49, "medium"), (50, "large"), (100, "large")],
)
def test_page_size_bins_are_fixed_ground_truth_area(count, expected):
    truth = np.zeros((10, 10), dtype=bool)
    truth.flat[:count] = True
    assert page_size_category(truth) == expected


def test_missing_predictions_never_disappear_from_gate_denominators():
    positive = np.ones((10, 10), dtype=bool)
    negative = ~positive
    cases = [
        RegionCase("p1", positive),
        RegionCase("p2", positive),
        RegionCase("n1", negative),
        RegionCase("n2", negative),
    ]
    report = evaluate_masks(cases, {"p1": positive, "n1": negative})
    summary = report["summary"]
    assert summary["expected_count"] == 4
    assert summary["completed_count"] == 2
    assert summary["failed_count"] == 2
    assert summary["foreground_iou_all_cases"]["mean"] == 0.5
    assert summary["foreground_iou_completed_only"]["mean"] == 1
    assert summary["negative_false_positive_rate_conservative"] == 0.5
    assert summary["gt_box_containment_rate_all_foreground"] == 0.5
    assert summary["gate_passed"] is False
    assert len(report["samples"]) == 4


def test_invalid_masks_and_crops_are_recorded_as_failures_not_excluded():
    positive = np.ones((10, 10), dtype=bool)
    cases = [RegionCase("bad-mask", positive), RegionCase("bad-box", positive)]
    report = evaluate_masks(
        cases,
        {"bad-mask": positive.astype(float), "bad-box": MaskPrediction(positive, (0, 0, 11, 10))},
    )
    assert report["summary"]["failed_count"] == 2
    assert report["summary"]["foreground_iou_all_cases"]["mean"] == 0
    assert {row["reason"] for row in report["samples"]} == {"invalid_prediction"}


def test_perfect_positive_and_negative_pass_fixed_gate():
    positive = np.ones((10, 10), dtype=bool)
    negative = ~positive
    report = evaluate_masks(
        [RegionCase("p", positive, "desk"), RegionCase("n", negative, "desk")],
        {"p": positive, "n": negative},
    )
    assert report["summary"]["gate_passed"] is True
    assert report["by_background"]["desk"]["gate_passed"] is True
    assert report["by_page_size"]["large"]["gate_passed"] is False


def test_threshold_boundary_is_inclusive_and_negative_metric_is_image_level():
    positive = np.ones((10, 10), dtype=bool)
    prediction = positive.copy()
    prediction[:2] = False
    negative = ~positive
    cases = [RegionCase("p", positive)] + [RegionCase(f"n{i}", negative) for i in range(10)]
    predictions = {case.sample_id: negative.copy() for case in cases}
    predictions["p"] = prediction
    predictions["n0"][0, 0] = True
    summary = evaluate_masks(cases, predictions)["summary"]
    assert summary["foreground_iou_all_cases"]["mean"] == 0.8
    assert summary["negative_false_positive_rate_conservative"] == 0.1
    assert summary["gate_passed"] is True
    predictions["n1"][0, 0] = True
    assert evaluate_masks(cases, predictions)["summary"]["gate_passed"] is False


def test_baselines_apply_same_masks_on_negative_images():
    positive = np.zeros((8, 12), dtype=bool)
    positive[2:6, 3:9] = True
    negative = np.zeros_like(positive)
    reports = evaluate_baselines([RegionCase("p", positive), RegionCase("n", negative)])
    assert reports["center_box"]["summary"]["foreground_iou_all_cases"]["mean"] == 1
    assert reports["all_foreground"]["summary"]["foreground_iou_all_cases"]["mean"] == 0.25
    for report in reports.values():
        assert report["summary"]["negative_false_positive_rate_conservative"] == 1
        assert report["summary"]["gate_passed"] is False


def test_case_errors_raise_instead_of_fabricating_dataset_denominator():
    mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="nonempty sequence"):
        evaluate_masks([], {})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_masks([RegionCase("p", mask), RegionCase("p", mask)], {})
    with pytest.raises(ValueError, match="outside"):
        evaluate_masks([RegionCase("p", mask)], {"extra": mask})
    with pytest.raises(ValueError, match="sample ID"):
        evaluate_masks([RegionCase("", mask)], {})
    with pytest.raises(ValueError, match="background_type"):
        evaluate_masks([RegionCase("p", mask, "")], {})
    with pytest.raises(ValueError, match="truth"):
        evaluate_masks([RegionCase("p", mask.astype(float))], {})


def test_mask_lifting_matches_independent_half_pixel_center_answer():
    # 5 destination centers map to grid indices 0, 0, 1, 2, 2, not 0, 0, 1, 1, 2.
    small = np.array([[False, True, False]], dtype=bool)
    expected = np.array([[False, False, True, False, False]] * 2)
    np.testing.assert_array_equal(mask_to_source(small, (2, 5)), expected)
    assert not np.shares_memory(mask_to_source(small, small.shape), small)


def test_raw_and_postprocessed_reports_do_not_confuse_rejection_with_missing_inference():
    positive = np.zeros((10, 10), dtype=bool)
    positive[2:8, 2:8] = True
    negative = np.zeros_like(positive)
    good = np.zeros((5, 5), dtype=np.float32)
    good[1:4, 1:4] = 0.9
    low_score = good * (2 / 3)
    report = evaluate_probabilities(
        [
            RegionCase("good", positive),
            RegionCase("weak", positive),
            RegionCase("empty", negative),
            RegionCase("missing", positive),
        ],
        {"good": good, "weak": low_score, "empty": np.ones_like(good)},
    )
    assert report["region_accepted_count"] == 1
    assert report["region_rejected_count"] == 2
    assert report["region_failure_count"] == 1
    raw = report["methods"]["learned_raw_threshold"]["summary"]
    post = report["methods"]["learned_postprocessed"]["summary"]
    assert raw["foreground_iou_all_cases"]["mean"] == pytest.approx(2 / 3)
    assert post["foreground_iou_all_cases"]["mean"] == pytest.approx(1 / 3)
    assert raw["negative_false_positive_rate_conservative"] == 1
    assert post["negative_false_positive_rate_conservative"] == 0
    assert post["failed_count"] == 1
    reasons = {row["id"]: row.get("reason") for row in report["region_decisions"]}
    assert reasons["weak"] == "uncertain_page_region"
    assert reasons["empty"] == "page_area_rejected"
    assert reasons["missing"] == "missing_prediction"


@pytest.mark.parametrize(
    "bad",
    [
        np.full((5, 5), np.nan, dtype=np.float32),
        np.full((5, 5), 1.1, dtype=np.float32),
        np.zeros((0, 5), dtype=np.float32),
        np.zeros((5, 5), dtype=bool),
    ],
)
def test_invalid_probability_is_a_failed_positive_and_failed_negative(bad):
    positive = np.ones((10, 10), dtype=bool)
    cases = [RegionCase("p", positive), RegionCase("n", ~positive)]
    report = evaluate_probabilities(cases, {"p": bad, "n": bad})
    assert report["region_failure_count"] == 2
    for name in ("learned_raw_threshold", "learned_postprocessed"):
        summary = report["methods"][name]["summary"]
        assert summary["foreground_iou_all_cases"]["mean"] == 0
        assert summary["negative_false_positive_rate_conservative"] == 1
        assert summary["failed_count"] == 2


def test_probabilities_require_matching_case_ids_and_valid_config():
    cases = [RegionCase("p", np.ones((10, 10), dtype=bool))]
    with pytest.raises(ValueError, match="outside"):
        evaluate_probabilities(cases, {"extra": None})
    with pytest.raises(ValueError, match="map sample"):
        evaluate_probabilities(cases, [])
    with pytest.raises(ValueError, match="RegionConfig"):
        evaluate_probabilities(cases, {}, config={})


@pytest.fixture
def mocked_checkpoint_data(tmp_path, monkeypatch):
    from flattener.regions import segmentation
    from flattener.synthetic import scenes

    (tmp_path / "manifest.json").write_text('{"test": "provenance only"}')
    records = [
        {
            "id": split,
            "split": split,
            "document_group": f"doc-{split}",
            "capture_session": f"capture-{split}",
            "background_type": "desk",
        }
        for split in ("train", "validation", "test")
    ]
    loaded_splits = []
    truth = np.zeros((10, 10), dtype=bool)
    truth[2:8, 2:8] = True
    probability = np.zeros((5, 5), dtype=np.float32)
    probability[1:4, 1:4] = 0.9

    def load_scene(_root, record):
        loaded_splits.append(record["split"])
        return SimpleNamespace(mask=truth, image=np.zeros((10, 10, 3), dtype=np.uint8))

    loaded = SimpleNamespace(
        metadata={
            "manifest_sha256": hashlib.sha256(
                (tmp_path / "manifest.json").read_bytes()
            ).hexdigest(),
            "checkpoint_sha256": "0" * 64,
            "epoch": 7,
        },
        predict_probability=lambda image: probability,
    )
    monkeypatch.setattr(
        scenes, "validate_scene_dataset", lambda root: {"manifest": {"samples": records}}
    )
    monkeypatch.setattr(scenes, "load_scene", load_scene)
    monkeypatch.setattr(segmentation, "load_segmenter", lambda path: loaded)
    return tmp_path, loaded_splits, loaded


def test_checkpoint_evaluation_decodes_only_selected_split_and_preserves_threads(
    mocked_checkpoint_data,
):
    import torch

    directory, loaded_splits, _ = mocked_checkpoint_data
    previous = torch.get_num_threads()
    report = evaluate_region_checkpoint(
        directory, directory / "fake.pt", split="validation", threads=1
    )
    assert loaded_splits == ["validation"]
    assert torch.get_num_threads() == previous
    assert report["expected_count"] == 1
    assert report["checkpoint_epoch"] == 7
    assert (
        report["methods"]["learned_postprocessed"]["summary"]["foreground_iou_all_cases"]["mean"]
        == 1
    )


def test_checkpoint_dataset_mismatch_fails_before_opening_any_labels(mocked_checkpoint_data):
    directory, loaded_splits, loaded = mocked_checkpoint_data
    loaded.metadata["manifest_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="manifest"):
        evaluate_region_checkpoint(directory, directory / "fake.pt")
    assert loaded_splits == []


def test_inference_failures_keep_denominators_and_evidence_is_not_overwritten(
    mocked_checkpoint_data,
):
    directory, loaded_splits, loaded = mocked_checkpoint_data

    def failed(image):
        raise ValueError("bad inference")

    loaded.predict_probability = failed
    report = evaluate_region_checkpoint(
        directory, directory / "fake.pt", output_dir=directory / "reports"
    )
    assert loaded_splits == ["validation"]
    assert report["region_failure_count"] == 1
    assert report["methods"]["learned_postprocessed"]["summary"]["failed_count"] == 1
    assert report["inference_errors"][0]["message"] == "bad inference"
    original = (directory / "reports/region-evaluation-validation.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_region_checkpoint(
            directory, directory / "fake.pt", output_dir=directory / "reports"
        )
    assert (directory / "reports/region-evaluation-validation.json").read_bytes() == original


@pytest.mark.parametrize(
    "kwargs", [{"split": "holdout"}, {"threads": 0}, {"threads": True}, {"threads": 1.2}]
)
def test_checkpoint_evaluation_validates_options_before_loading(mocked_checkpoint_data, kwargs):
    directory, loaded_splits, _ = mocked_checkpoint_data
    with pytest.raises(ValueError):
        evaluate_region_checkpoint(directory, directory / "fake.pt", **kwargs)
    assert loaded_splits == []
