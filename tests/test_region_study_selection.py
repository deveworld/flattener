"""Validation-only candidate choice, frozen identities and immutable publication."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from flattener.learning.model import CoordinatePredictor, ModelConfig
from flattener.regions import study_selection as selection
from flattener.regions.checkpoints import create_region_model, model_spec
from flattener.regions.geometry import RegionConfig
from flattener.regions.segmentation import SegmenterConfig
from flattener.regions.study_selection import freeze_region_study
from flattener.regions.training import LOSS_PROTOCOL, SELECTION_PROTOCOL

NEW_MANIFEST = "a" * 64
OLD_MANIFEST = "b" * 64


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _checkpoint(path, value):
    torch.save(value, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".pt.sha256").write_text(digest + "\n")
    return digest


def _evaluation(checkpoint_hash, epoch=80, *, old=False, iou=0.875, fp=0, failed=None):
    rows, decisions = [], []
    for index in range(144):
        has_page = index < 96
        valid = index != failed
        detected = has_page or index - 96 < fp
        identifier = f"validation-{index:03d}"
        row = {"id": identifier, "has_page": has_page, "status": "completed" if valid else "failed"}
        if valid:
            row.update(iou=iou if has_page else float(not detected), detected_page=detected)
        rows.append(row)
        decisions.append(
            {
                "id": identifier,
                "status": "failed" if not valid else "accepted" if detected else "rejected",
            }
        )
    completed = sum(row["status"] == "completed" for row in rows)
    mean = float(np.mean([row.get("iou", 0) for row in rows[:96]]))
    negative_fp = sum(row["status"] == "failed" or row["detected_page"] for row in rows[96:])
    passed = completed == 144 and mean >= 0.8 and negative_fp / 48 <= 0.1
    summary = {
        "expected_count": 144,
        "completed_count": completed,
        "failed_count": 144 - completed,
        "foreground_count": 96,
        "negative_count": 48,
        "foreground_iou_all_cases": {"count": 96, "mean": mean},
        "negative_false_positive_count_conservative": negative_fp,
        "negative_false_positive_rate_conservative": negative_fp / 48,
        "gate_passed": passed,
    }
    return {
        "schema_version": "region-study-evaluation-0.2",
        "split": "validation",
        "dataset_schema": "scene-0.2",
        "test_used_for_model_selection": False,
        "postprocessor": asdict(RegionConfig()),
        "gate_method": "learned_postprocessed",
        "gate_passed": passed,
        "expected_count": 144,
        "methods": {
            "learned_postprocessed": {
                "evaluation_grid": "original_source_pixels",
                "thresholds": dict(selection.GATE_THRESHOLDS),
                "samples": rows,
                "summary": summary,
            }
        },
        "region_decisions": decisions,
        "region_accepted_count": sum(row["status"] == "accepted" for row in decisions),
        "region_rejected_count": sum(row["status"] == "rejected" for row in decisions),
        "region_failure_count": 144 - completed,
        "manifest_sha256": NEW_MANIFEST,
        "training_manifest_sha256": OLD_MANIFEST if old else NEW_MANIFEST,
        "explicit_transfer_evaluation": old,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": epoch,
    }


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """Schema-valid generated checkpoint fixtures; no training or image evaluation."""
    arms = {}
    monkeypatch.setattr(selection, "STUDY_MANIFEST_SHA256", NEW_MANIFEST)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        for name, architecture in (("A", "local"), ("B", "local"), ("C", "context")):
            folder = tmp_path / name
            folder.mkdir()
            model = create_region_model(architecture)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            epoch = 77 if name == "A" else 80
            loss = {"A": 0.1, "B": 0.5, "C": 0.4}[name]
            history = [
                {"epoch": i, "train_loss": 1.0, "validation": {"loss": loss + (epoch - i) / 100}}
                for i in range(1, epoch + 1)
            ]
            config = {
                "seed": 20260923,
                "epochs": 80,
                "batch_size": 16,
                "learning_rate": 0.001,
                "threads": 1,
                "architecture": architecture,
            }
            if name == "A":
                config.pop("architecture")
                config.update(seed=20260922, model_config=asdict(SegmenterConfig()))
            provenance = {
                "loss": copy.deepcopy(LOSS_PROTOCOL),
                "selection": copy.deepcopy(SELECTION_PROTOCOL),
                "dataset_schema": "scene-0.1" if name == "A" else "scene-0.2",
            }
            state = {
                "schema_version": "region-segmentation-0.1" if name == "A" else "region-study-0.2",
                "config": config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "initial_validation": {"loss": loss + 1},
                "best_validation": {"loss": loss},
                "best_epoch": epoch,
                "manifest_sha256": OLD_MANIFEST if name == "A" else NEW_MANIFEST,
                "dataset_counts": dict(selection.EXPECTED_COUNTS),
                "provenance": provenance,
                "torch_rng_state": torch.get_rng_state(),
                "shuffle_rng_state": torch.Generator().manual_seed(5).get_state(),
            }
            if name != "A":
                state["model_spec"] = model_spec(architecture)
            path = folder / "best.pt"
            digest = _checkpoint(path, state)
            evaluation = folder / "validation.json"
            _json(evaluation, _evaluation(digest, epoch, old=name == "A"))
            arms[name] = {"checkpoint": path, "validation_report": evaluation}
            if name == "A":
                monkeypatch.setattr(selection, "LEGACY_CHECKPOINT_SHA256", digest)
            else:
                training = folder / "training.json"
                _json(
                    training,
                    {
                        "schema_version": "region-study-0.2",
                        "epochs_completed": 80,
                        "config": config,
                        "model_spec": state["model_spec"],
                        "manifest_sha256": NEW_MANIFEST,
                        "dataset_counts": dict(selection.EXPECTED_COUNTS),
                        "checkpoints": {"best.pt": digest},
                        "best_epoch": epoch,
                        "best_validation": state["best_validation"],
                        "initial_validation": state["initial_validation"],
                        "history": history,
                        "provenance": provenance,
                    },
                )
                arms[name]["training_report"] = training
        coordinate = tmp_path / "coordinate.pt"
        config = ModelConfig(input_size=(16, 16), grid_size=(3, 3), channels=(2, 3, 4), hidden=5)
        coordinate_digest = _checkpoint(
            coordinate,
            {
                "schema_version": "learning-0.1",
                "config": {"model_config": asdict(config)},
                "model_state": CoordinatePredictor(config).state_dict(),
            },
        )
        monkeypatch.setattr(selection, "COORDINATE_CHECKPOINT_SHA256", coordinate_digest)
    protocol = tmp_path / "pretraining-freeze.json"
    _json(protocol, {"scope": "unit-test protocol bytes; not a production experiment"})
    return arms, coordinate, protocol, tmp_path / "selection.json"


def _freeze(evidence):
    arms, coordinate, protocol, destination = evidence
    return freeze_region_study(
        arms, destination, coordinate_checkpoint=coordinate, protocol_path=protocol
    )


def _change_report(evidence, arm, callback):
    path = evidence[0][arm]["validation_report"]
    value = json.loads(path.read_text())
    callback(value)
    _json(path, value)


def test_lower_loss_passing_candidate_selected_and_all_arms_frozen(evidence):
    before_rng = torch.get_rng_state().clone()
    result = _freeze(evidence)
    assert result["selected_arm"] == "C"
    assert set(result["arms"]) == {"A", "B", "C"}
    assert (
        result["arms"]["A"]["selection_validation_weighted_loss"]
        < result["arms"]["C"]["selection_validation_weighted_loss"]
    )
    assert result["arms"]["A"]["eligible_for_selection"] is False
    assert result["arms"]["A"]["loss_manifest_sha256"] == OLD_MANIFEST
    assert result["arms"]["A"]["explicit_transfer_evaluation"] is True
    assert result["arms"]["B"]["loss_manifest_sha256"] == NEW_MANIFEST
    assert result["postprocessor"] == asdict(RegionConfig())
    assert result["pipeline_output_size_hw"] == [192, 144]
    assert (
        result["coordinate_checkpoint_sha256"]
        == hashlib.sha256(evidence[1].read_bytes()).hexdigest()
    )
    assert result["protocol_sha256"] == hashlib.sha256(evidence[2].read_bytes()).hexdigest()
    assert result["test_used_for_selection"] is False
    assert result["test_evaluation_performed"] is False
    assert result["quality_claim_allowed"] is False
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert json.loads(evidence[3].read_text()) == result


def test_exact_loss_tie_retains_b(evidence):
    arms = evidence[0]
    path = arms["C"]["checkpoint"]
    state = torch.load(path, weights_only=True)
    state["best_validation"]["loss"] = 0.5
    state["initial_validation"]["loss"] = 1.5
    for row in state["history"]:
        row["validation"]["loss"] = 0.5 + (80 - row["epoch"]) / 100
    digest = _checkpoint(path, state)
    training_path = arms["C"]["training_report"]
    training = json.loads(training_path.read_text())
    training.update(
        best_validation=state["best_validation"],
        initial_validation=state["initial_validation"],
        history=state["history"],
    )
    training["checkpoints"]["best.pt"] = digest
    _json(training_path, training)
    _change_report(evidence, "C", lambda value: value.update(checkpoint_sha256=digest))
    assert _freeze(evidence)["selected_arm"] == "B"


@pytest.mark.parametrize(
    "b_iou,c_iou,selected", [(0.875, 0.75, "B"), (0.75, 0.875, "C"), (0.75, 0.75, None)]
)
def test_fixed_gate_precedes_loss_and_no_candidate_is_promoted_on_two_misses(
    evidence, b_iou, c_iou, selected
):
    for arm, iou in (("B", b_iou), ("C", c_iou)):
        path = evidence[0][arm]["validation_report"]
        previous = json.loads(path.read_text())
        _json(path, _evaluation(previous["checkpoint_sha256"], iou=iou))
    result = _freeze(evidence)
    assert result["selected_arm"] == selected
    if selected is None:
        assert "promote neither" in result["rationale"]


@pytest.mark.parametrize("false_positives,selected", [(4, "C"), (5, "B")])
def test_48_negative_gate_accepts_four_but_not_five(evidence, false_positives, selected):
    path = evidence[0]["C"]["validation_report"]
    previous = json.loads(path.read_text())
    _json(path, _evaluation(previous["checkpoint_sha256"], fp=false_positives))
    assert _freeze(evidence)["selected_arm"] == selected


def test_one_evaluator_failure_disqualifies_candidate_even_with_good_remaining_iou(evidence):
    path = evidence[0]["C"]["validation_report"]
    previous = json.loads(path.read_text())
    _json(path, _evaluation(previous["checkpoint_sha256"], failed=0))
    result = _freeze(evidence)
    assert result["selected_arm"] == "B"
    assert result["arms"]["C"]["gate_metrics"]["failed_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "test_split",
        "checkpoint_hash",
        "checkpoint_epoch",
        "manifest",
        "transfer",
        "postprocessor",
        "gate_lie",
        "summary_lie",
        "missing_case",
        "wrong_label",
    ],
)
def test_invalid_evaluation_evidence_is_rejected_before_publication(evidence, change):
    def mutate(value):
        if change == "test_split":
            value["split"] = "test"
        elif change == "checkpoint_hash":
            value["checkpoint_sha256"] = "0" * 64
        elif change == "checkpoint_epoch":
            value["checkpoint_epoch"] = 1
        elif change == "manifest":
            value["manifest_sha256"] = "c" * 64
        elif change == "transfer":
            value["training_manifest_sha256"] = OLD_MANIFEST
        elif change == "postprocessor":
            value["postprocessor"]["threshold"] = 0.4
        elif change == "gate_lie":
            value["gate_passed"] = False
        elif change == "summary_lie":
            value["methods"]["learned_postprocessed"]["summary"]["foreground_iou_all_cases"][
                "mean"
            ] = 0.99
        elif change == "missing_case":
            value["methods"]["learned_postprocessed"]["samples"].pop()
        else:
            value["methods"]["learned_postprocessed"]["samples"][0]["has_page"] = False

    _change_report(evidence, "C", mutate)
    with pytest.raises(ValueError):
        _freeze(evidence)
    assert not evidence[3].exists()


@pytest.mark.parametrize(
    "change",
    ["budget", "seed", "architecture", "best_hash", "best_epoch", "best_loss", "history_minimum"],
)
def test_incomplete_or_inconsistent_new_training_is_rejected(evidence, change):
    path = evidence[0]["B"]["training_report"]
    value = json.loads(path.read_text())
    if change == "budget":
        value["epochs_completed"] = 79
    elif change == "seed":
        value["config"]["seed"] += 1
    elif change == "architecture":
        value["model_spec"]["architecture"] = "context"
    elif change == "best_hash":
        value["checkpoints"]["best.pt"] = "0" * 64
    elif change == "best_epoch":
        value["best_epoch"] = 79
    elif change == "best_loss":
        value["best_validation"]["loss"] = 0.01
    else:
        value["history"][0]["validation"]["loss"] = 0.001
    _json(path, value)
    with pytest.raises(ValueError):
        _freeze(evidence)
    assert not evidence[3].exists()


def test_frozen_legacy_identity_is_not_optional(evidence, monkeypatch):
    monkeypatch.setattr(selection, "LEGACY_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen legacy"):
        _freeze(evidence)


@pytest.mark.parametrize(
    "constant,message",
    [
        ("STUDY_MANIFEST_SHA256", "pretraining freeze"),
        ("COORDINATE_CHECKPOINT_SHA256", "fixed legacy coordinate"),
    ],
)
def test_preregistered_manifest_and_coordinate_identities_are_required(
    evidence, monkeypatch, constant, message
):
    monkeypatch.setattr(selection, constant, "0" * 64)
    with pytest.raises(ValueError, match=message):
        _freeze(evidence)
    assert not evidence[3].exists()


def test_extreme_integer_metric_has_a_validation_error_not_numeric_overflow(evidence):
    def mutate(value):
        value["methods"]["learned_postprocessed"]["samples"][0]["iou"] = 10**1000

    _change_report(evidence, "C", mutate)
    with pytest.raises(ValueError, match="IoU must be finite"):
        _freeze(evidence)
    assert not evidence[3].exists()


@pytest.mark.parametrize("best_epoch", [0, 50])
def test_completed_eighty_epochs_may_select_an_earlier_best_checkpoint(evidence, best_epoch):
    arm = evidence[0]["C"]
    state = torch.load(arm["checkpoint"], weights_only=True)
    history = [
        {
            "epoch": epoch,
            "train_loss": 1.0,
            "validation": {"loss": 0.4 + abs(epoch - best_epoch) / 100},
        }
        for epoch in range(1, 81)
    ]
    initial = {"loss": 0.4 + best_epoch / 100}
    state.update(
        epoch=best_epoch,
        best_epoch=best_epoch,
        initial_validation=initial,
        history=history[:best_epoch],
    )
    digest = _checkpoint(arm["checkpoint"], state)
    training = json.loads(arm["training_report"].read_text())
    training.update(best_epoch=best_epoch, initial_validation=initial, history=history)
    training["checkpoints"]["best.pt"] = digest
    _json(arm["training_report"], training)
    _change_report(
        evidence,
        "C",
        lambda value: value.update(checkpoint_sha256=digest, checkpoint_epoch=best_epoch),
    )
    result = _freeze(evidence)
    assert result["selected_arm"] == "C"
    assert result["arms"]["C"]["checkpoint_epoch"] == best_epoch
    assert "last checkpoint is not loaded" in result["completion_evidence"]


def test_report_mutation_after_read_is_detected_before_publication(evidence, monkeypatch):
    original_gate = selection._gate
    calls = 0

    def changed_after_read(report):
        nonlocal calls
        if calls == 0:
            _change_report(evidence, "A", lambda value: value.update(extra="changed after read"))
        calls += 1
        return original_gate(report)

    monkeypatch.setattr(selection, "_gate", changed_after_read)
    with pytest.raises(ValueError, match="validation_report changed before freeze"):
        _freeze(evidence)
    assert not evidence[3].exists()


def test_checkpoint_mutation_during_load_is_detected(evidence, monkeypatch):
    original_read = selection.read_region_checkpoint

    def changed_after_load(path):
        checkpoint = original_read(path)
        path.write_bytes(path.read_bytes() + b"\n")
        return checkpoint

    monkeypatch.setattr(selection, "read_region_checkpoint", changed_after_load)
    with pytest.raises(ValueError, match="checkpoint changed while being validated"):
        _freeze(evidence)
    assert not evidence[3].exists()


def test_competing_freeze_publication_is_preserved_and_temporary_is_removed(evidence, monkeypatch):
    original_link = selection.os.link
    competing_bytes = b'{"created_by":"another publisher"}\n'
    temporary_paths = []

    def competing_publisher(source, destination):
        temporary_paths.append(Path(source))
        Path(destination).write_bytes(competing_bytes)
        return original_link(source, destination)

    monkeypatch.setattr(selection.os, "link", competing_publisher)
    with pytest.raises(FileExistsError):
        _freeze(evidence)
    assert evidence[3].read_bytes() == competing_bytes
    assert temporary_paths and all(not path.exists() for path in temporary_paths)


def test_existing_freeze_is_preserved_without_reading_more_inputs(evidence, monkeypatch):
    _freeze(evidence)
    original = evidence[3].read_bytes()
    monkeypatch.setattr(
        selection, "_arm", lambda *args: pytest.fail("existing freeze was not rejected first")
    )
    with pytest.raises(FileExistsError):
        _freeze(evidence)
    assert evidence[3].read_bytes() == original


def test_selection_reads_only_explicit_evidence_without_test_discovery(evidence, monkeypatch):
    protected = evidence[3].parent / "region-study-test.json"
    protected.write_text("do not read held-out test metrics")
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.resolve() != protected.resolve(), "held-out test report was opened"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(
        Path, "glob", lambda *args, **kwargs: pytest.fail("unexpected report discovery")
    )
    monkeypatch.setattr(
        Path, "rglob", lambda *args, **kwargs: pytest.fail("unexpected dataset discovery")
    )
    _freeze(evidence)


def test_arm_identity_must_cover_exactly_a_b_c(evidence):
    evidence[0].pop("A")
    with pytest.raises(ValueError, match="exactly arms"):
        _freeze(evidence)
