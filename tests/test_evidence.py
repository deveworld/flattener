"""Measurement and model metadata in evaluation reports."""

import json

import pytest

from flattener.evidence import evidence, initialization
from tools.benchmark import calibration, docunet

OWN = {"kind": "checkpoint", "step": 40000}
UVDOC = {"kind": "uvdoc", "path": "uvdoc.pkl"}


def test_reports_preserve_measurements_and_checkpoint_metadata():
    measurements = (
        "docunet_standard",
        "docunet_stress",
        "gt_crop_validation",
        "gt_full_frame_validation",
        "training_loss",
        "predictions",
    )
    for measurement in measurements:
        for step in (None, 3, 9999, 10000, 40000):
            record = evidence(measurement, {**OWN, "step": step})
            assert record == {
                "measurement": measurement,
                "model_provenance": "own_checkpoint",
                "checkpoint_step": step,
                "initialization": "unknown",
            }
    for model, provenance in (
        (None, "unknown"),
        (UVDOC, "external_pretrained"),
        ({"provenance": "own_previous_model"}, "own_previous_model"),
    ):
        assert evidence("docunet_standard", model) == {
            "measurement": "docunet_standard",
            "model_provenance": provenance,
            "checkpoint_step": None,
            "initialization": None,
        }
    assert calibration.CALIBRATION_EVIDENCE == evidence("evaluator_calibration", UVDOC)
    with pytest.raises(ValueError):
        evidence("docunet")


def test_initialization_follows_the_lineage_to_its_start():
    uvdoc_start = [{"kind": "uvdoc", "path": "uvdoc.pkl"}, {"kind": "train"}]
    parent = [{"kind": "checkpoint", "lineage": uvdoc_start}, {"kind": "train"}]
    cases = {
        "scratch": [{"kind": "scratch", "seed": 0}, {"kind": "train"}],
        "external_pretrained": parent,
        "unknown": [{"kind": "checkpoint"}],
    }
    for expected, lineage in cases.items():
        record = evidence("docunet_standard", {**OWN, "lineage": lineage})
        assert record["initialization"] == expected
    assert initialization({**OWN, "initialization": "scratch"}) == "scratch"
    assert initialization(OWN) == "unknown"
    assert initialization(UVDOC) is None and initialization(None) is None


def _record(predictions, model, inputs):
    items = [
        {"output": f"{k}_{m}.png", "action": "written", "status": "success"}
        for k in range(1, 66)
        for m in (1, 2)
    ]
    record = {"schema": "flattener-dewarp-predictions-2", "model": model, "inputs": inputs}
    (predictions / "predictions.json").write_text(json.dumps({**record, "items": items}))


def test_docunet_reports_carry_the_evidence_of_their_predictions(tiny_docunet):
    scans, predictions = tiny_docunet
    for model in (OWN, {**OWN, "step": 50}):
        _record(predictions, model, "docunet_crop")
        report = docunet.evaluate_docunet(
            predictions,
            pattern="{k}_{m}.png",
            conventions=["128_excl64"],
            build=None,
            references_dir=scans,
            workers=1,
        )
        assert report["evidence"]["checkpoint_step"] == model["step"]
        assert report["evidence"]["model_provenance"] == "own_checkpoint"
        assert report["evidence"]["measurement"] == "docunet_standard"
