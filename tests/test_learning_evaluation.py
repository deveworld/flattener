"""Independent coordinate answers and complete evaluation denominators."""

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flattener.contracts import FlattenerError
from flattener.coordinates import identity_map
from flattener.learning import evaluation
from flattener.learning.evaluation import (
    METHODS,
    boundary_mask,
    coarse_representation_survey,
    coordinate_metrics,
    endpoint_error,
    evaluate_checkpoint,
)
from flattener.learning.model import CoordinatePredictor, ModelConfig
from flattener.synthetic.pipeline import DatasetConfig, generate_dataset, load_sample


def test_three_four_source_pixel_displacement_is_five_at_unequal_resolution():
    truth = identity_map((11, 7))
    prediction = truth.astype(np.float64) + [2 * 3 / 200, 2 * 4 / 300]
    mask = np.ones(truth.shape[:2], dtype=bool)
    assert endpoint_error(prediction, truth, mask, (300, 200)) == pytest.approx(5, abs=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_selected_predictions_invalidate_instead_of_shrinking_mask(value):
    truth = identity_map((4, 5))
    prediction = truth.copy()
    prediction[1, 2, 0] = value
    with pytest.raises(ValueError, match="prediction.*non-finite"):
        endpoint_error(prediction, truth, np.ones((4, 5), dtype=bool), (4, 5))


def test_nonfinite_coordinates_outside_gt_mask_are_ignored():
    truth = identity_map((3, 5))
    prediction = truth.copy()
    truth[0, 0] = np.nan
    prediction[0, 0] = np.inf
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    assert endpoint_error(prediction, truth, mask, (90, 150)) == 0
    assert coordinate_metrics(prediction, truth, mask, (90, 150))["gt_pixels"] == 14


def test_nonfinite_ground_truth_cannot_be_silently_masked():
    truth = identity_map((4, 5))
    prediction = truth.copy()
    truth[0, 0] = np.nan
    with pytest.raises(ValueError, match="ground truth.*non-finite"):
        endpoint_error(prediction, truth, np.ones((4, 5), dtype=bool), (4, 5))


def test_out_of_bounds_error_and_counts_are_unclipped():
    truth = identity_map((20, 10))
    prediction = truth + np.float32(4)
    result = coordinate_metrics(prediction, truth, np.ones((20, 10), dtype=bool), (20, 10))
    assert result["epe_pixels"] == pytest.approx(np.hypot(20, 40), abs=1e-5)
    assert result["geometry"]["out_of_bounds_pixels"] == 200
    assert result["geometry"]["out_of_bounds_fraction"] == 1
    assert result["geometry"]["valid_pixels"] == 0
    assert result["geometry"]["checked_cells"] == 19 * 9
    assert result["geometry"]["failed_cells"] == 0


def test_reversed_map_counts_all_cells_even_when_outside_source():
    truth = identity_map((20, 10))
    prediction = truth.copy()
    prediction[..., 0] *= -1
    prediction += 5
    result = coordinate_metrics(prediction, truth, np.ones((20, 10), dtype=bool), (20, 10))
    assert result["geometry"]["failed_cells"] == 19 * 9
    assert result["geometry"]["out_of_bounds_pixels"] == 200
    assert result["geometry_reason"] == "fold_or_degenerate"


def test_outer_eight_pixels_boundary_intersects_gt_and_is_measured_separately():
    mask = np.ones((30, 40), dtype=bool)
    mask[0] = False
    boundary = boundary_mask(mask)
    assert boundary.sum() == 30 * 40 - 14 * 24 - 40
    truth = identity_map(mask.shape)
    prediction = truth.copy()
    prediction[~boundary & mask, 0] += np.float32(0.5)
    result = coordinate_metrics(prediction, truth, mask, (30, 40))
    assert result["epe_pixels"] > 0
    assert result["boundary8_epe_pixels"] == 0


@pytest.mark.parametrize("source_size", [(0, 4), (4, 0), (True, 4), (4.0, 5), "45"])
def test_invalid_source_sizes_are_rejected(source_size):
    truth = identity_map((4, 5))
    with pytest.raises(ValueError, match="source_size"):
        endpoint_error(truth, truth, np.ones((4, 5), dtype=bool), source_size)


def test_empty_and_nonboolean_masks_are_rejected():
    truth = identity_map((4, 5))
    for mask in (np.zeros((4, 5), dtype=bool), np.ones((4, 5), dtype=np.float32)):
        with pytest.raises(ValueError, match="gt_mask"):
            endpoint_error(truth, truth, mask, (4, 5))


@pytest.fixture(scope="module")
def evaluation_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("learning-evaluation")
    root = directory / "data"
    manifest = generate_dataset(
        root,
        config=DatasetConfig(
            seed=741,
            train_documents=1,
            validation_documents=2,
            test_documents=1,
            variants_per_document=2,
            output_size=(128, 128),
            photo_size=(128, 128),
        ),
    )
    config = ModelConfig(input_size=(32, 24), grid_size=(5, 3), channels=(2, 3, 4), hidden=8)
    with torch.random.fork_rng():
        torch.manual_seed(12)
        model = CoordinatePredictor(config)
    checkpoint = directory / "identity.pt"
    torch.save(
        {
            "schema_version": "learning-0.1",
            "model_state": model.state_dict(),
            "config": {"model_config": asdict(config)},
            "epoch": 0,
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        },
        checkpoint,
    )
    return root, checkpoint, manifest


@pytest.fixture(autouse=True)
def efficient_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_all_methods_count_every_sample_and_identity_predictor_agrees(evaluation_data, tmp_path):
    root, checkpoint, _ = evaluation_data
    report = evaluate_checkpoint(root, checkpoint, output_dir=tmp_path)
    assert report["expected_count"] == report["completed_count"] == 4
    assert report["failed_count"] == 0
    assert report["evaluation_complete"] is True
    assert report["quality_claim_allowed"] is False
    for method in METHODS:
        assert report["methods"][method]["completed_count"] == 4
        assert report["methods"][method]["failed_count"] == 0
        assert len(report["samples"][method]) == 4
        assert len(report["methods"][method]["by_document_group"]) == 2
        assert (
            sum(row["expected_count"] for row in report["methods"][method]["by_family"].values())
            == 4
        )
    identity = report["methods"]["identity"]["epe_pixels"]
    for method in ("learned", "constant_gray", "permuted_image"):
        assert report["methods"][method]["epe_pixels"] == identity
    assert all(row["input_sample_id"] != row["id"] for row in report["samples"]["permuted_image"])
    assert json.loads((tmp_path / "evaluation-validation.json").read_text()) == report
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(root, checkpoint, output_dir=tmp_path)


def test_dense_mean_is_direct_average_of_training_truth_only(evaluation_data):
    root, _, manifest = evaluation_data
    maps = [
        load_sample(root, record).backward_map
        for record in manifest["samples"]
        if record["split"] == "train"
    ]
    expected = np.mean(np.stack(maps).astype(np.float64), axis=0).astype(np.float32)
    actual = evaluation._training_mean(root, manifest["samples"])
    np.testing.assert_array_equal(actual, expected)


def test_missing_file_stays_in_expected_count_and_failed_rows(evaluation_data, tmp_path):
    original, checkpoint, manifest = evaluation_data
    root = tmp_path / "missing"
    shutil.copytree(original, root)
    record = next(row for row in manifest["samples"] if row["split"] == "validation")
    (root / record["paths"]["photo.png"]).unlink()
    report = evaluate_checkpoint(root, checkpoint)
    assert report["expected_count"] == 4
    assert report["failed_count"] >= 1
    assert report["evaluation_complete"] is False
    assert report["quality_claim_allowed"] is False
    for method in METHODS:
        rows = report["samples"][method]
        missing = next(row for row in rows if row["id"] == record["id"])
        assert missing["status"] == "failed"
        assert missing["reason"] == "sample_load_failed"
        assert report["methods"][method]["expected_count"] == 4
        assert report["methods"][method]["failed_count"] >= 1


def test_nonfinite_model_output_is_counted_separately(evaluation_data, monkeypatch):
    root, checkpoint, _ = evaluation_data

    def nonfinite(self, inputs):
        return torch.full((inputs.shape[0], 2, *self.config.grid_size), float("nan"))

    monkeypatch.setattr(CoordinatePredictor, "forward", nonfinite)
    report = evaluate_checkpoint(root, checkpoint)
    for method in ("learned", "constant_gray", "permuted_image"):
        assert report["methods"][method]["non_finite_failure_count"] == 4
        assert report["methods"][method]["failed_count"] == 4
        assert report["methods"][method]["epe_pixels"]["mean"] is None
    assert report["methods"]["identity"]["completed_count"] == 4
    assert report["evaluation_complete"] is False


def test_survey_and_validation_evaluation_never_read_test_artifacts(evaluation_data, monkeypatch):
    root, checkpoint, _ = evaluation_data
    checked = evaluation._checked_sample
    visited = []

    def recording(directory, record):
        assert record["split"] != "test"
        visited.append(record["split"])
        return checked(directory, record)

    monkeypatch.setattr(evaluation, "_checked_sample", recording)
    report = coarse_representation_survey(root)
    assert report["test_samples_loaded"] == 0
    assert report["evaluation_complete"] is True
    assert set(report["grids"]) == {"9x7", "17x13", "33x25"}
    for summary in report["grids"].values():
        assert summary["expected_count"] == summary["completed_count"] == 6
        assert summary["by_split"]["train"]["expected_count"] == 2
        assert summary["by_split"]["validation"]["expected_count"] == 4
    evaluate_checkpoint(root, checkpoint)
    assert set(visited) == {"train", "validation"}


def test_survey_masked_constant_residual_remains_constant(evaluation_data, monkeypatch):
    root, _, _ = evaluation_data
    checked = evaluation._checked_sample

    def shifted(directory, record):
        sample = checked(directory, record)
        sample.backward_map = identity_map(sample.valid_mask.shape)
        sample.backward_map += np.asarray([0.04, -0.03], dtype=np.float32)
        sample.valid_mask[::3, ::3] = False
        sample.backward_map[~sample.valid_mask] = np.nan
        return sample

    monkeypatch.setattr(evaluation, "_checked_sample", shifted)
    report = coarse_representation_survey(root)
    assert report["evaluation_complete"] is True
    for summary in report["grids"].values():
        assert summary["epe_pixels"]["mean"] < 0.00002


def test_checkpoint_for_other_manifest_is_rejected(evaluation_data, tmp_path):
    root, checkpoint, _ = evaluation_data
    state = torch.load(checkpoint, weights_only=True)
    state["manifest_sha256"] = "0" * 64
    modified = tmp_path / "mismatched.pt"
    torch.save(state, modified)
    with pytest.raises(ValueError, match="manifest does not match"):
        evaluate_checkpoint(root, modified)


def test_evaluation_preserves_torch_rng(evaluation_data):
    root, checkpoint, _ = evaluation_data
    before = torch.get_rng_state().clone()
    evaluate_checkpoint(root, checkpoint)
    assert torch.equal(torch.get_rng_state(), before)


def test_evaluation_rejects_bad_checkpoint_sidecar(evaluation_data, tmp_path):
    root, checkpoint, _ = evaluation_data
    copied = tmp_path / "modified.pt"
    shutil.copyfile(checkpoint, copied)
    copied.with_suffix(".pt.sha256").write_text("0" * 64)
    with pytest.raises(FlattenerError) as error:
        evaluate_checkpoint(root, copied)
    assert error.value.reason_code == "checkpoint_hash_mismatch"


def test_evaluation_rejects_wrong_checkpoint_weight_dtype(evaluation_data, tmp_path):
    root, checkpoint, _ = evaluation_data
    saved = torch.load(checkpoint, weights_only=True)
    key = next(iter(saved["model_state"]))
    saved["model_state"][key] = saved["model_state"][key].double()
    modified = tmp_path / "wrong-dtype.pt"
    torch.save(saved, modified)
    with pytest.raises(FlattenerError) as error:
        evaluate_checkpoint(root, modified)
    assert error.value.reason_code == "invalid_checkpoint"


@pytest.mark.parametrize("output_size", [(4, 6), (12, 16), (24, 32)])
def test_page_footprint_is_not_point_occupancy_and_detects_cropping(output_size):
    source_mask = np.ones((48, 64), dtype=bool)
    grid = identity_map(output_size)
    full = evaluation.page_sampling_metrics(grid, source_mask)
    assert full["page_sampling_precision"] == full["source_page_coverage"] == 1
    assert full["source_page_iou"] == 1
    crop = evaluation.page_sampling_metrics(grid * np.float32(0.5), source_mask)
    assert crop["page_sampling_precision"] == 1
    assert crop["source_page_coverage"] == crop["source_page_iou"] == 0.25


def test_source_page_sampling_separates_background_from_page_extent():
    source_mask = np.zeros((48, 64), dtype=bool)
    source_mask[12:36, 16:48] = True
    grid = identity_map((12, 16))
    background = evaluation.page_sampling_metrics(grid, source_mask)
    page = evaluation.page_sampling_metrics(grid * np.float32(0.5), source_mask)
    assert background["page_sampling_precision"] == 0.25
    assert background["source_page_coverage"] == 1
    assert background["source_page_iou"] == 0.25
    assert page["page_sampling_precision"] == page["source_page_coverage"] == 1
    assert page["source_page_iou"] == 1


@pytest.mark.parametrize("failure", ["nan", "constant", "reflection"])
def test_invalid_page_extent_is_missing_not_zero_or_perfect(failure):
    grid = identity_map((12, 16))
    if failure == "nan":
        grid[2, 3] = np.nan
    elif failure == "constant":
        grid[:] = 0
    else:
        grid[..., 0] *= -1
    result = evaluation.page_sampling_metrics(grid, np.ones((48, 64), dtype=bool))
    assert result["source_page_coverage"] is None
    assert result["source_page_iou"] is None
    assert result["coverage_status"] == "not_evaluated"
    assert result["coverage_reason"]
    assert result["page_sampling_precision"] == (191 / 192 if failure == "nan" else 1)


def test_wholly_outside_positive_map_has_zero_page_coverage():
    result = evaluation.page_sampling_metrics(
        identity_map((12, 16)) + np.float32(4), np.ones((48, 64), dtype=bool)
    )
    assert result["page_sampling_precision"] == result["source_page_coverage"] == 0
    assert result["source_page_iou"] == 0
    assert result["coverage_status"] == "evaluated"


def test_sampling_mask_index_handles_float64_rounding_at_positive_edge():
    grid = identity_map((12, 16)).astype(np.float64)
    grid[:, -1, 0] = np.nextafter(1.0, 0.0)
    result = evaluation.page_sampling_metrics(grid, np.ones((48, 64), dtype=bool))
    assert result["page_sampling_precision"] == 1


def test_self_intersecting_and_touching_boundaries_are_rejected():
    simple = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
    assert evaluation._simple_boundary(simple)
    assert not evaluation._simple_boundary(simple[[0, 2, 1, 3]])
    touching = np.array([[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]], dtype=np.float64)
    assert not evaluation._simple_boundary(touching)


def test_page_relative_epe_uses_original_mask_physical_bbox_diagonal():
    truth = identity_map((12, 16)).astype(np.float64)
    predicted = truth + [2 * 3 / 64, 2 * 4 / 48]
    page = np.zeros((48, 64), dtype=bool)
    page[10:40, 20:60] = True
    result = evaluation.page_relative_endpoint_error(
        predicted, truth, np.ones((12, 16), dtype=bool), page
    )
    assert result["source_page_bbox_diagonal_pixels"] == 50
    assert result["page_relative_epe"] == pytest.approx(0.1)
    page[:] = False
    page[0, :3] = True
    with pytest.raises(ValueError, match="degenerate"):
        evaluation.page_relative_endpoint_error(predicted, truth, np.ones((12, 16), bool), page)


@pytest.fixture
def scene_evaluation_fixture(tmp_path, monkeypatch):
    """No real dataset, training, optimizer, or heldout files are accessed."""
    from flattener.learning import coordinate_study, scene_data

    root, checkpoint = tmp_path / "tiny-scenes", tmp_path / "tiny.pt"
    root.mkdir()
    (root / "manifest.json").write_text('{"fixture":true}')
    checkpoint.write_bytes(b"frozen fixture weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".pt.sha256").write_text(digest + "\n")
    config = ModelConfig(input_size=(12, 10), grid_size=(3, 3), channels=(2, 2, 2), hidden=4)
    state = SimpleNamespace(
        visited=[],
        loaded=0,
        fail_index=None,
        nonfinite=False,
        prediction=None,
        render_calls=[],
        schema="coordinate-study-0.3",
        constructor_options=[],
        sample_source_override=None,
        sample_kind_override=None,
        paper_source_kinds=None,
    )

    class TinyDataset:
        def __init__(
            self, directory, split, model_config, output_size, *, include_paper_augmentation=False
        ):
            assert directory == root
            assert split in ("validation", "test")
            assert model_config == config
            self.split, self.model_config, self.output_size = split, model_config, output_size
            state.constructor_options.append(include_paper_augmentation)
            self.photo_size = (48, 64)
            self.manifest_sha256 = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            self.records = tuple(
                {
                    "id": f"{split}-{index}",
                    "document_group": f"doc-{index}",
                    "family": "perspective",
                    "size_bin": "small",
                }
                for index in range(3)
            )
            if include_paper_augmentation:
                self.records += tuple(
                    {
                        "id": f"{split}-paper-{index}",
                        "document_group": f"paper-doc-{index}",
                        "family": "flat",
                        "size_bin": "small",
                        "source": "paper",
                    }
                    for index in range(2)
                )
                if state.paper_source_kinds is not None:
                    for record, kind in zip(
                        self.records[3:], state.paper_source_kinds, strict=True
                    ):
                        record["source_kind"] = kind
                        if kind == "capture":
                            record["family"] = "capture_perspective"

        def __getitem__(self, index):
            state.visited.append((self.split, index))
            if index == state.fail_index:
                raise OSError("fixture image unavailable")
            return {
                "original_image": np.zeros((*self.photo_size, 3), dtype=np.uint8),
                "target": torch.from_numpy(identity_map(self.output_size)),
                "valid": torch.ones(self.output_size, dtype=torch.bool),
                "source_mask": torch.ones(self.photo_size, dtype=torch.bool),
                "source": state.sample_source_override
                or self.records[index].get("source", "authored"),
                "source_kind": state.sample_kind_override
                or self.records[index].get(
                    "source_kind", "simulated" if index >= 3 else "authored"
                ),
            }

    def predict(image, size):
        assert image.shape == (48, 64, 3)
        if state.nonfinite:
            return np.full((*size, 2), np.nan, dtype=np.float32)
        return identity_map(size) if state.prediction is None else state.prediction(size)

    def load(path):
        assert path == checkpoint
        state.loaded += 1
        return SimpleNamespace(
            model_config=config,
            model_sha256=digest,
            model_id="fixture",
            checksum_verified=True,
            predict_map=predict,
        )

    monkeypatch.setattr(scene_data, "SceneCoordinateDataset", TinyDataset)
    monkeypatch.setattr(coordinate_study, "load_coordinate_predictor", load)
    monkeypatch.setattr(
        coordinate_study,
        "coordinate_checkpoint",
        lambda path: {"schema_version": state.schema, "manifest_sha256": "0" * 64},
    )
    actual_render = evaluation.render

    def recording_render(image, grid, **kwargs):
        assert "target_mask" not in kwargs
        state.render_calls.append(grid.shape)
        return actual_render(image, grid, **kwargs)

    monkeypatch.setattr(evaluation, "render", recording_render)
    return root, checkpoint, state


@pytest.mark.parametrize(
    "schema", ["coordinate-study-0.1", "coordinate-study-0.2", "coordinate-study-0.3"]
)
def test_scene_frozen_protocol_full_render_and_cross_manifest(
    scene_evaluation_fixture, tmp_path, schema
):
    root, checkpoint, state = scene_evaluation_fixture
    state.schema = schema
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    output = tmp_path / "evaluation"
    report = evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=output, output_size=(12, 16)
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert report["evaluation_complete"] is report["integrity_verified"] is True
    assert report["quality_status"] == "not_evaluated"
    assert report["quality_claim_allowed"] is False
    assert report["protocol"]["training_manifest_matches_evaluation"] is False
    assert report["protocol"]["checkpoint_schema"] == schema
    assert report["protocol"]["output_size"] == [12, 16]
    assert report["protocol"]["source_size"] == [48, 64]
    assert report["summary"]["epe_pixels"] == {"count": 3, "mean": 0, "median": 0, "p95": 0}
    assert report["summary"]["source_page_coverage"]["mean"] == 1
    assert report["summary"]["page_relative_epe"]["mean"] == 0
    assert report["summary"]["renderer_success_count"] == 3
    assert report["summary"]["page_missed_count"] == 0
    assert report["summary"]["by_size_bin"]["small"]["expected_count"] == 3
    assert report["summary"]["by_family"]["perspective"]["processed_count"] == 3
    assert all(
        row["ground_truth_page_metrics"]["source_page_coverage"] == 1 for row in report["samples"]
    )
    assert state.loaded == 1 and len(state.render_calls) == 3
    assert state.constructor_options == [False, False]
    assert report["protocol"]["include_paper_augmentation"] is False
    assert report["protocol"]["source_counts"] == {"authored": 3}
    assert report["protocol"]["source_kind_counts"] == {"authored": 3}
    assert report["summary"]["by_source"]["authored"]["expected_count"] == 3
    assert state.visited == [("validation", index) for index in range(3)]
    assert json.loads((output / "report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        evaluation.evaluate_scene_checkpoint(
            root, checkpoint, output_dir=output, output_size=(12, 16)
        )


def test_scene_renderer_failures_still_have_fixed_gt_epe(scene_evaluation_fixture, tmp_path):
    root, checkpoint, state = scene_evaluation_fixture
    state.prediction = lambda size: identity_map(size) + np.float32(4)
    report = evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=tmp_path / "oob", output_size=(12, 16)
    )
    summary = report["summary"]
    assert summary["expected_count"] == summary["processed_count"] == 3
    assert summary["renderer_failure_count"] == summary["failed_count"] == 3
    assert summary["epe_pixels"]["count"] == 3
    assert summary["epe_pixels"]["mean"] == pytest.approx(160)
    assert summary["source_page_coverage"]["mean"] == 0
    assert summary["page_missed_count"] == 3
    assert report["evaluation_complete"] is True  # accounted-for failures, not quality success


@pytest.mark.parametrize("mode", ["missing", "nonfinite"])
def test_scene_missing_and_nonfinite_remain_in_denominator(
    scene_evaluation_fixture, tmp_path, mode
):
    root, checkpoint, state = scene_evaluation_fixture
    if mode == "missing":
        state.fail_index = 1
    else:
        state.nonfinite = True
    report = evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=tmp_path / mode, output_size=(12, 16)
    )
    summary = report["summary"]
    assert summary["expected_count"] == summary["processed_count"] == 3
    assert summary["epe_missing_count"] == (1 if mode == "missing" else 3)
    assert summary["non_finite_failure_count"] == (0 if mode == "missing" else 3)
    assert summary["page_missed_missing_count"] == (1 if mode == "missing" else 3)
    assert summary["renderer_missing_count"] == (1 if mode == "missing" else 0)


def test_scene_interrupt_resume_has_exact_prefix_no_duplicate_cases(
    scene_evaluation_fixture, tmp_path, monkeypatch
):
    root, checkpoint, state = scene_evaluation_fixture
    output = tmp_path / "resumable"
    actual = evaluation._scene_case

    def interrupt(dataset, index, predictor, render_config):
        if index == 1:
            raise KeyboardInterrupt
        return actual(dataset, index, predictor, render_config)

    monkeypatch.setattr(evaluation, "_scene_case", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate_scene_checkpoint(
            root, checkpoint, output_dir=output, output_size=(12, 16), save_every=2
        )
    partial = json.loads((output / "report.json").read_text())
    assert partial["state"] == "interrupted"
    assert partial["summary"]["processed_count"] == 1
    assert partial["summary"]["expected_count"] == 3
    assert partial["evaluation_complete"] is False
    assert not (output / ".evaluation.lock").exists()
    monkeypatch.setattr(evaluation, "_scene_case", actual)
    report = evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=output, output_size=(12, 16), resume=True
    )
    assert report["evaluation_complete"] is True
    assert state.visited == [("validation", index) for index in range(3)]
    # Completed runs revalidate hashes but never re-decode their cases.
    evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=output, output_size=(12, 16), resume=True
    )
    assert len(state.visited) == 3


@pytest.mark.parametrize(
    "change", ["rows", "split", "output_size", "source", "manifest", "quality"]
)
def test_scene_resume_rejects_changed_protocol_or_report(
    scene_evaluation_fixture, tmp_path, monkeypatch, change
):
    root, checkpoint, _ = scene_evaluation_fixture
    output = tmp_path / "frozen"
    evaluation.evaluate_scene_checkpoint(root, checkpoint, output_dir=output, output_size=(12, 16))
    arguments = {"output_dir": output, "output_size": (12, 16), "resume": True}
    if change in ("rows", "quality"):
        report = json.loads((output / "report.json").read_text())
        if change == "rows":
            report["samples"].append(report["samples"][0])
        else:
            report["quality_claim_allowed"] = True
        (output / "report.json").write_text(json.dumps(report))
    elif change == "split":
        arguments["split"] = "test"
    elif change == "output_size":
        arguments["output_size"] = (14, 18)
    elif change == "source":
        original = evaluation._scene_provenance()
        original["source_sha256"] = "0" * 64
        monkeypatch.setattr(evaluation, "_scene_provenance", lambda: original)
    else:
        (root / "manifest.json").write_text('{"fixture":"changed"}')
    with pytest.raises(ValueError, match="resume"):
        evaluation.evaluate_scene_checkpoint(root, checkpoint, **arguments)


def test_scene_final_integrity_failure_cannot_be_complete(
    scene_evaluation_fixture, tmp_path, monkeypatch
):
    root, checkpoint, _ = scene_evaluation_fixture
    actual = evaluation._scene_case

    def changed(dataset, index, predictor, render_config):
        row = actual(dataset, index, predictor, render_config)
        if index == 2:
            (root / "manifest.json").write_text('{"fixture":"changed-mid-run"}')
        return row

    monkeypatch.setattr(evaluation, "_scene_case", changed)
    output = tmp_path / "integrity"
    with pytest.raises(ValueError, match="changed during"):
        evaluation.evaluate_scene_checkpoint(
            root, checkpoint, output_dir=output, output_size=(12, 16)
        )
    report = json.loads((output / "report.json").read_text())
    assert report["evaluation_complete"] is report["integrity_verified"] is False
    assert report["state"] == "interrupted"
    assert report["summary"]["processed_count"] == 3


def test_scene_cli_is_explicit_and_reports_operational_failure(
    scene_evaluation_fixture, tmp_path, capsys
):
    root, checkpoint, state = scene_evaluation_fixture
    state.fail_index = 0
    assert (
        evaluation.main(
            [
                "--dataset",
                str(root),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "validation",
                "--output",
                str(tmp_path / "cli"),
                "--output-size",
                "12",
                "16",
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["evaluation_complete"] is True
    assert result["quality_status"] == "not_evaluated"
    assert result["summary"]["epe_missing_count"] == 1


def test_scene_paper_opt_in_has_separate_failure_inclusive_source_denominators(
    scene_evaluation_fixture, tmp_path
):
    root, checkpoint, state = scene_evaluation_fixture
    state.fail_index = 3
    report = evaluation.evaluate_scene_checkpoint(
        root,
        checkpoint,
        output_dir=tmp_path / "mixed",
        output_size=(12, 16),
        include_paper_augmentation=True,
    )
    assert report["schema_version"] == "scene-coordinate-evaluation-1.1"
    assert report["protocol"]["include_paper_augmentation"] is True
    assert report["protocol"]["source_counts"] == {"authored": 3, "paper": 2}
    assert state.constructor_options == [True, True]
    summary = report["summary"]
    assert summary["expected_count"] == summary["processed_count"] == 5
    authored, paper = summary["by_source"]["authored"], summary["by_source"]["paper"]
    assert authored["expected_count"] == authored["epe_pixels"]["count"] == 3
    assert authored["failed_count"] == authored["epe_missing_count"] == 0
    assert paper["expected_count"] == paper["processed_count"] == 2
    assert paper["epe_pixels"]["count"] == paper["epe_missing_count"] == 1
    assert paper["renderer_missing_count"] == paper["page_metrics_missing_count"] == 1
    assert paper["by_size_bin"]["small"]["expected_count"] == 2
    assert paper["by_family"]["flat"]["epe_missing_count"] == 1
    assert paper["by_source_kind"]["simulated"]["expected_count"] == 2
    assert paper["by_source_kind"]["simulated"]["epe_missing_count"] == 1
    assert report["protocol"]["source_kind_counts"] == {"authored": 3, "simulated": 2}
    assert [row["source"] for row in report["samples"]] == ["authored"] * 3 + ["paper"] * 2


@pytest.mark.parametrize("invalid", [1, 0, None, "true", []])
def test_scene_paper_opt_in_requires_strict_boolean_before_dataset_access(
    scene_evaluation_fixture, tmp_path, invalid
):
    root, checkpoint, state = scene_evaluation_fixture
    with pytest.raises(ValueError, match="include_paper_augmentation must be boolean"):
        evaluation.evaluate_scene_checkpoint(
            root,
            checkpoint,
            output_dir=tmp_path / "invalid",
            output_size=(12, 16),
            include_paper_augmentation=invalid,
        )
    assert state.constructor_options == [] and state.visited == []


def test_scene_resume_cannot_change_paper_opt_in(scene_evaluation_fixture, tmp_path):
    root, checkpoint, _ = scene_evaluation_fixture
    output = tmp_path / "source-frozen"
    evaluation.evaluate_scene_checkpoint(root, checkpoint, output_dir=output, output_size=(12, 16))
    with pytest.raises(ValueError, match="resume"):
        evaluation.evaluate_scene_checkpoint(
            root,
            checkpoint,
            output_dir=output,
            output_size=(12, 16),
            resume=True,
            include_paper_augmentation=True,
        )


def test_scene_source_mismatch_is_not_silently_relabelled(scene_evaluation_fixture, tmp_path):
    root, checkpoint, state = scene_evaluation_fixture
    state.sample_source_override = "paper"
    report = evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=tmp_path / "source-mismatch", output_size=(12, 16)
    )
    assert report["summary"]["by_source"]["authored"]["epe_missing_count"] == 3
    assert "paper" not in report["summary"]["by_source"]
    assert all("source differs" in row["errors"][0]["message"] for row in report["samples"])


def test_scene_cli_paper_opt_in_reaches_constructor_and_final_integrity(
    scene_evaluation_fixture, tmp_path, capsys
):
    root, checkpoint, state = scene_evaluation_fixture
    assert (
        evaluation.main(
            [
                "--dataset",
                str(root),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "validation",
                "--output",
                str(tmp_path / "paper-cli"),
                "--output-size",
                "12",
                "16",
                "--include-paper-augmentation",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert state.constructor_options == [True, True]
    assert summary["by_source"]["paper"]["expected_count"] == 2


def test_scene_actual_capture_and_simulated_paper_have_distinct_complete_denominators(
    scene_evaluation_fixture, tmp_path
):
    root, checkpoint, state = scene_evaluation_fixture
    state.paper_source_kinds = ("capture", "simulated")
    state.fail_index = 3
    report = evaluation.evaluate_scene_checkpoint(
        root,
        checkpoint,
        output_dir=tmp_path / "kinds",
        output_size=(12, 16),
        include_paper_augmentation=True,
    )
    assert report["protocol"]["source_kind_counts"] == {"authored": 3, "capture": 1, "simulated": 1}
    assert [row["source_kind"] for row in report["samples"]] == ["authored"] * 3 + [
        "capture",
        "simulated",
    ]
    kinds = report["summary"]["by_source"]["paper"]["by_source_kind"]
    capture, simulated = kinds["capture"], kinds["simulated"]
    assert capture["annotation_model"] == report["protocol"]["annotation_models"]["capture"]
    assert capture["annotation_model"] == report["samples"][3]["annotation_model"]
    assert "not dense measured GT" in capture["annotation_model"]
    assert capture["expected_count"] == capture["processed_count"] == 1
    assert capture["epe_missing_count"] == capture["renderer_missing_count"] == 1
    assert capture["source_page_coverage"]["mean"] is None
    assert capture["source_page_coverage"]["missing_count"] == 1
    assert simulated["expected_count"] == simulated["renderer_success_count"] == 1
    assert simulated["epe_pixels"]["mean"] == 0
    assert simulated["source_page_coverage"]["mean"] == 1


@pytest.mark.parametrize("kind", ["authored", "unknown", None, 1])
def test_scene_invalid_paper_kind_is_rejected_before_case_access(
    scene_evaluation_fixture, tmp_path, kind
):
    root, checkpoint, state = scene_evaluation_fixture
    state.paper_source_kinds = (kind, "simulated")
    with pytest.raises(ValueError, match="source_kind"):
        evaluation.evaluate_scene_checkpoint(
            root,
            checkpoint,
            output_dir=tmp_path / "invalid-kind",
            output_size=(12, 16),
            include_paper_augmentation=True,
        )
    assert state.visited == []


def test_scene_sample_kind_mismatch_stays_in_original_kind_denominator(
    scene_evaluation_fixture, tmp_path
):
    root, checkpoint, state = scene_evaluation_fixture
    state.sample_kind_override = "capture"
    report = evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=tmp_path / "mismatch-kind", output_size=(12, 16)
    )
    assert (
        report["summary"]["by_source"]["authored"]["by_source_kind"]["authored"][
            "epe_missing_count"
        ]
        == 3
    )
    assert all(row["source_kind"] == "authored" for row in report["samples"])
    assert all("source_kind differs" in row["errors"][0]["message"] for row in report["samples"])


def test_scene_resume_cannot_reclassify_capture_as_simulated(scene_evaluation_fixture, tmp_path):
    root, checkpoint, state = scene_evaluation_fixture
    state.paper_source_kinds = ("capture", "simulated")
    output = tmp_path / "kind-frozen"
    evaluation.evaluate_scene_checkpoint(
        root, checkpoint, output_dir=output, output_size=(12, 16), include_paper_augmentation=True
    )
    state.paper_source_kinds = ("simulated", "simulated")
    with pytest.raises(ValueError, match="resume"):
        evaluation.evaluate_scene_checkpoint(
            root,
            checkpoint,
            output_dir=output,
            output_size=(12, 16),
            include_paper_augmentation=True,
            resume=True,
        )


@pytest.fixture
def selection_evidence(scene_evaluation_fixture, tmp_path, monkeypatch):
    """Formal five-case schema with inert weights; selection never executes a model."""
    root, _, state = scene_evaluation_fixture
    state.paper_source_kinds = ("capture", "simulated")
    template = evaluation.evaluate_scene_checkpoint(
        root, tmp_path / "tiny.pt", output_dir=tmp_path / "template",
        output_size=(12, 16), include_paper_augmentation=True,
    )

    def make(name, *, epe=10., change=None, finalize=None):
        checkpoint, path = tmp_path / f"{name}.pt", tmp_path / f"{name}.json"
        checkpoint.write_bytes(("inert frozen weights " + name).encode())
        digest = evaluation._sha256(checkpoint)
        sidecar = checkpoint.with_suffix(".pt.sha256")
        sidecar.write_text(digest + "\n")
        report = deepcopy(template)
        protocol = report["protocol"]
        protocol["checkpoint_sha256"] = digest
        protocol["checkpoint_sidecar_sha256"] = evaluation._sha256(sidecar)
        for row in report["samples"]:
            value = epe[row["source_kind"]] if isinstance(epe, dict) else epe
            row.update(epe_pixels=value, boundary8_epe_pixels=value, page_relative_epe=value / 80,
                       source_page_coverage=.8, page_sampling_precision=.9, source_page_iou=.7)
        if change:
            change(report)
        report["protocol_sha256"] = evaluation._json_digest(protocol)
        report["samples_sha256"] = evaluation._json_digest(report["samples"])
        report["summary"] = evaluation._scene_grouped_summary(report["samples"], report["samples"])
        if finalize:
            finalize(report)
        path.write_text(json.dumps(report, allow_nan=False))
        return evaluation.SceneSelectionCandidate(name, checkpoint, digest, path,
                                                  evaluation._sha256(path))

    def forbidden(*args, **kwargs):
        pytest.fail("report-only selection must not evaluate, load a model, or open a dataset")

    from flattener.learning import coordinate_study, scene_data

    monkeypatch.setattr(evaluation, "evaluate_scene_checkpoint", forbidden)
    monkeypatch.setattr(scene_data, "SceneCoordinateDataset", forbidden)
    monkeypatch.setattr(coordinate_study, "load_coordinate_predictor", forbidden)
    monkeypatch.setattr(coordinate_study, "coordinate_checkpoint", forbidden)
    return make


def _selection_failure(report, kind="capture", *, coverage=None):
    row = next(row for row in report["samples"] if row["source_kind"] == kind)
    row.update(status="failed", renderer_status="invalid_transform",
               source_page_coverage=coverage, source_page_iou=None,
               errors=[{"stage": "render", "reason": "fold_or_degenerate", "message": "fixture"}])


def test_selection_best_eligible_is_not_mixed_epe_best_and_preserves_weights(
    selection_evidence, tmp_path
):
    baseline = selection_evidence("baseline")
    unsafe = selection_evidence("lowest-epe", epe=1, change=_selection_failure)
    safe = selection_evidence("safe", epe=8)
    before = {item.checkpoint_path: Path(item.checkpoint_path).read_bytes()
              for item in (baseline, unsafe, safe)}
    output = tmp_path / "selection.json"
    result = evaluation.select_scene_checkpoints(baseline, [unsafe, safe], output_path=output)
    assert result["best_eligible"]["id"] == "safe"
    assert result["quality_status"] == "not_evaluated"
    assert result["checkpoint_modified"] is result["test_evaluated"] is False
    assert next(e for e in result["candidates"] if e["candidate"]["id"] == "lowest-epe")[
        "gates"]["capture_renderer_failures_not_increased"] is False
    assert json.loads(output.read_text()) == result
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(FileExistsError):
        evaluation.select_scene_checkpoints(baseline, [safe], output_path=output)
    assert json.loads(output.read_text()) == result
    assert not list(tmp_path.glob(".selection-*"))


def test_selection_no_passing_candidate_and_boundary_thresholds(selection_evidence):
    def perfect_page(report):
        for row in report["samples"]:
            row["source_page_coverage"] = row["page_sampling_precision"] = 1.

    baseline = selection_evidence("baseline", change=perfect_page)

    def at_boundary(report):
        for row in report["samples"]:
            row["source_page_coverage"] = 1. - .005
            row["page_sampling_precision"] = 1. - .005

    edge = selection_evidence("edge", epe={"authored": 11., "capture": 9., "simulated": 9.},
                              change=at_boundary)
    assert evaluation.select_scene_checkpoints(baseline, [edge])["best_eligible"]["id"] == "edge"
    failed = selection_evidence("failed", epe=9.01)
    result = evaluation.select_scene_checkpoints(baseline, [failed])
    assert result["best_eligible"] is None and result["complete"] is True
    assert result["candidates"][0]["eligible"] is False


@pytest.mark.parametrize("fault", ["epe", "boundary", "relative", "renderer", "nonfinite"])
def test_selection_incomplete_candidate_is_accounted_for_not_selected(selection_evidence, fault):
    def missing(report):
        row = report["samples"][3]
        field = {"epe": "epe_pixels", "boundary": "boundary8_epe_pixels",
                 "relative": "page_relative_epe", "renderer": "renderer_status"}
        if fault == "nonfinite":
            row["non_finite_prediction"] = True
        else:
            row[field[fault]] = None
        row.update(status="failed", errors=[{"reason": "fixture_missing"}])

    baseline = selection_evidence("baseline")
    candidate = selection_evidence("missing", epe=8, change=missing)
    result = evaluation.select_scene_checkpoints(baseline, [candidate])
    assert result["best_eligible"] is None
    assert result["candidates"][0]["gates"] == {"eligible": False}
    assert result["candidates"][0]["diagnostics"]["domains"]["capture"]["expected_count"] == 1
    with pytest.raises(ValueError, match="baseline requires finite"):
        evaluation.select_scene_checkpoints(candidate, [baseline])


def test_selection_folded_baseline_and_nullable_page_metrics_keep_fixed_denominator(
    selection_evidence,
):
    baseline = selection_evidence("baseline", change=_selection_failure)
    candidate = selection_evidence("candidate", epe=8, change=_selection_failure)
    result = evaluation.select_scene_checkpoints(baseline, [candidate])
    assert result["best_eligible"]["id"] == "candidate"  # Existing policy, not stricter gating.
    capture = result["candidates"][0]["diagnostics"]["domains"]["capture"]
    assert capture["source_page_coverage"]["mean"] is None
    assert capture["fixed_denominator_page_scores"]["source_page_coverage"] == 0
    assert capture["page_metrics_missing_count"] == 1
    assert capture["page_missed_count"] == 0
    assert capture["renderer_failure_count"] == 1


def test_selection_ranking_is_order_independent_and_zero_baseline_safe(selection_evidence):
    baseline = selection_evidence("baseline")
    first, second = selection_evidence("a", epe=8), selection_evidence("b", epe=8)
    a = evaluation.select_scene_checkpoints(baseline, [first, second])
    b = evaluation.select_scene_checkpoints(baseline, [second, first])
    assert a == b
    assert a["best_eligible"]["checkpoint_sha256"] == min(first.checkpoint_sha256,
                                                         second.checkpoint_sha256)
    zero = selection_evidence("zero", epe=0)
    another = selection_evidence("another", epe=0)
    result = evaluation.select_scene_checkpoints(zero, [another, first])
    assert result["best_eligible"]["id"] == "another"
    assert next(e for e in result["candidates"] if e["candidate"]["id"] == "another")[
        "rank"][:2] == [0, 0]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["summary", "samples_sha", "protocol_sha", "schema", "complete",
                                  "integrity", "state", "quality", "test", "checkpoint_binding",
                                  "source_digest", "runtime_digest", "source_kind_count", "bool",
                                  "negative", "over_one", "duplicate", "bad_source"])
def test_selection_rejects_malformed_even_if_outer_report_sha_is_updated(selection_evidence, fault):
    def corrupt(report):
        row, protocol = report["samples"][0], report["protocol"]
        if fault == "summary":
            report["summary"]["by_source"]["authored"]["by_size_bin"]["small"][
                "epe_pixels"]["mean"] += 1
        elif fault in ("samples_sha", "protocol_sha"):
            report[fault + "256"] = "0" * 64
        elif fault == "schema":
            report["schema_version"] = "unknown"
        elif fault in ("complete", "integrity"):
            report["evaluation_complete" if fault == "complete" else "integrity_verified"] = 1
        elif fault == "state":
            report["state"] = "interrupted"
        elif fault == "quality":
            report["quality_claim_allowed"] = True
        elif fault == "test":
            protocol["split"] = "test"
        elif fault == "checkpoint_binding":
            protocol["checkpoint_sha256"] = "0" * 64
        elif fault in ("source_digest", "runtime_digest"):
            protocol[fault.replace("digest", "sha256")] = "0" * 64
        elif fault == "source_kind_count":
            protocol["source_kind_counts"]["capture"] = True
        elif fault == "bool":
            row["epe_pixels"] = True
        elif fault == "negative":
            row["page_relative_epe"] = -1
        elif fault == "over_one":
            row["source_page_coverage"] = 1.001
        elif fault == "duplicate":
            report["samples"][1]["id"] = row["id"]
        elif fault == "bad_source":
            row["source"] = "unknown"
        if fault != "samples_sha":
            report["samples_sha256"] = evaluation._json_digest(report["samples"])
        if fault != "protocol_sha":
            report["protocol_sha256"] = evaluation._json_digest(protocol)

    baseline = selection_evidence("baseline")
    broken = selection_evidence("broken", finalize=corrupt)
    with pytest.raises(ValueError):
        evaluation.select_scene_checkpoints(baseline, [broken])


@pytest.mark.parametrize("fault", ["manifest", "record_hash", "runtime", "definitions", "order",
                                  "document", "source_size"])
def test_selection_rejects_different_protocol_or_ordered_cases(selection_evidence, fault):
    def different(report):
        protocol = report["protocol"]
        if fault in ("manifest", "record_hash"):
            protocol["dataset_manifest_sha256" if fault == "manifest" else
                     "dataset_records_sha256"] = "a" * 64
        elif fault == "runtime":
            protocol["runtime"]["torch_threads"] += 1
            protocol["runtime_sha256"] = evaluation._json_digest(protocol["runtime"])
        elif fault == "definitions":
            protocol["definitions"]["epe"] = "different units"
        elif fault == "order":
            report["samples"].reverse()
        elif fault == "document":
            report["samples"][0]["document_group"] = "different"
        else:
            protocol["source_size"] = [64, 48]

    baseline = selection_evidence("baseline")
    other = selection_evidence("other", change=different)
    message = "definitions" if fault == "definitions" else "protocol/cases"
    with pytest.raises(ValueError, match=message):
        evaluation.select_scene_checkpoints(baseline, [other])


def test_selection_document_macro_not_frame_macro_and_model_configs_may_differ(selection_evidence):
    def documents(report):
        report["samples"][1]["document_group"] = report["samples"][0]["document_group"]

    baseline = selection_evidence("baseline", change=documents)

    def different_model(report):
        documents(report)
        report["protocol"]["model_config"]["input_size"] = [24, 20]
        report["protocol"]["checkpoint_schema"] = "coordinate-study-0.8"
        report["protocol"]["training_manifest_sha256"] = "b" * 64
        for row, value in zip(report["samples"][:3], (2., 2., 8.), strict=True):
            row["epe_pixels"] = value

    candidate = selection_evidence("candidate", epe=8, change=different_model)
    result = evaluation.select_scene_checkpoints(baseline, [candidate])
    diagnostics = result["candidates"][0]["diagnostics"]
    assert diagnostics["domains"]["authored"]["epe_pixels"]["mean"] == 4
    assert diagnostics["document_macro"]["authored"]["epe_pixels"] == 5
    assert diagnostics["document_macro"]["authored"]["document_count"] == 2
    assert result["best_eligible"]["id"] == "candidate"


@pytest.mark.parametrize("target", ["checkpoint", "sidecar", "report"])
def test_selection_actual_bytes_are_verified(selection_evidence, target):
    baseline, candidate = selection_evidence("baseline"), selection_evidence("candidate")
    path = Path(candidate.checkpoint_path if target != "report" else candidate.report_path)
    if target == "sidecar":
        path = path.with_suffix(".pt.sha256")
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        evaluation.select_scene_checkpoints(baseline, [candidate])


def test_selection_policy_ids_and_final_evidence_guard(selection_evidence, monkeypatch):
    baseline, candidate = selection_evidence("baseline"), selection_evidence("candidate", epe=8)
    with pytest.raises(ValueError, match="policy"):
        evaluation.select_scene_checkpoints(baseline, [candidate], policy_id="looser")
    with pytest.raises(ValueError, match="policy"):
        evaluation.select_scene_checkpoints(baseline, [candidate], policy_sha256="0" * 64)
    with pytest.raises(ValueError, match="distinct"):
        evaluation.select_scene_checkpoints(baseline, [replace(candidate, id=baseline.id)])
    original = evaluation._selection_gates

    def changed(new, old):
        Path(candidate.checkpoint_path).write_bytes(b"changed during selection")
        return original(new, old)

    monkeypatch.setattr(evaluation, "_selection_gates", changed)
    with pytest.raises(ValueError, match="changed during"):
        evaluation.select_scene_checkpoints(baseline, [candidate])


def test_selection_cli_frozen_relative_spec_and_exit_codes(selection_evidence, tmp_path, capsys):
    baseline, candidate = selection_evidence("baseline"), selection_evidence("candidate", epe=8)

    def reference(item):
        value = asdict(item)
        for key in ("checkpoint_path", "report_path"):
            value[key] = Path(value[key]).name
        return value

    spec = {"schema_version": "scene-selection-spec-1.0",
            "policy_id": evaluation.SCENE_SELECTION_POLICY_ID,
            "policy_sha256": evaluation._json_digest(evaluation.scene_selection_policy()),
            "baseline": reference(baseline), "candidates": [reference(candidate)]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    output = tmp_path / "selection.json"
    arguments = ["select", "--spec", str(path), "--output", str(output)]
    assert evaluation.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["best_eligible"]["id"] == "candidate"
    assert evaluation.main(arguments) == 2  # Fresh output only.
    capsys.readouterr()
    spec["candidates"] = [reference(selection_evidence("unqualified", epe=10))]
    path.write_text(json.dumps(spec))
    assert evaluation.main(arguments[:-1] + [str(tmp_path / "no-eligible.json")]) == 1
    capsys.readouterr()
    spec["policy_sha256"] = None
    path.write_text(json.dumps(spec))
    assert evaluation.main(arguments[:-1] + [str(tmp_path / "invalid.json")]) == 2
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("fault", ["definitions", "runtime", "cuda", "runtime_boolean"])
def test_selection_rejects_shared_noncanonical_contract(selection_evidence, fault):
    def corrupt(report):
        protocol = report["protocol"]
        if fault == "definitions":
            protocol["definitions"] = {}
        elif fault == "runtime":
            protocol["runtime"] = {}
        elif fault == "cuda":
            protocol["runtime"]["device"] = "cuda:0"
        else:
            protocol["runtime"]["torch_threads"] = True
        protocol["runtime_sha256"] = evaluation._json_digest(protocol["runtime"])

    baseline = selection_evidence("baseline", change=corrupt)
    candidate = selection_evidence("candidate", epe=8, change=corrupt)
    with pytest.raises(ValueError):
        evaluation.select_scene_checkpoints(baseline, [candidate])


def test_selection_authored_small_gate_cannot_be_hidden_by_other_sizes(selection_evidence):
    def sizes(report):
        report["samples"][2]["size_bin"] = "medium"

    baseline = selection_evidence("baseline", change=sizes)

    def regression(report):
        sizes(report)
        for row, error in zip(report["samples"][:3], (12., 12., 0.), strict=True):
            row["epe_pixels"] = error

    candidate = selection_evidence("candidate", epe=8, change=regression)
    result = evaluation.select_scene_checkpoints(baseline, [candidate])
    gates = result["candidates"][0]["gates"]
    assert gates["authored_epe_within_10_percent"] is True
    assert gates["authored_small_epe_within_10_percent"] is False
    assert result["best_eligible"] is None


@pytest.mark.parametrize("fault", ["nan", "duplicate_key"])
def test_selection_json_rejects_nonstandard_or_ambiguous_values(selection_evidence, fault):
    baseline, candidate = selection_evidence("baseline"), selection_evidence("candidate")
    path = Path(candidate.report_path)
    text = path.read_text()
    if fault == "nan":
        text = text.replace('"epe_pixels": 10.0', '"epe_pixels": NaN', 1)
    else:
        text = '{"state":"complete",' + text[1:]
    path.write_text(text)
    candidate = replace(candidate, report_sha256=evaluation._sha256(path))
    with pytest.raises(ValueError):
        evaluation.select_scene_checkpoints(baseline, [candidate])
