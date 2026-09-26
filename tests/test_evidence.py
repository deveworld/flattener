"""Evidence labels: what a report measures and whether it is a new-model real-photo result."""

import json

import pytest

from flattener.benchmark import calibration, docunet
from flattener.evidence import SMOKE_STEPS, evidence

OWN = {"kind": "checkpoint", "step": SMOKE_STEPS}
UVDOC = {"kind": "uvdoc", "sha256": "7e90"}


def test_only_trained_own_checkpoints_on_docunet_count_as_new_model_real_photo():
    cases = {
        ("docunet_standard", "own"): "new_model_real_photo",
        ("docunet_stress", "own"): "new_model_real_photo",
        ("gt_crop_validation", "own"): "gt_crop_validation",
        ("gt_full_frame_validation", "own"): "gt_full_frame_validation",
        ("docunet_standard", "smoke"): "smoke_training",
        ("gt_crop_validation", "smoke"): "smoke_training",
        ("docunet_stress", "uvdoc"): "external_pretrained",
        ("docunet_standard", None): "docunet_standard",
    }
    models = {"own": OWN, "smoke": {**OWN, "step": SMOKE_STEPS - 1}, "uvdoc": UVDOC, None: None}
    for (measurement, model), kind in cases.items():
        record = evidence(measurement, models[model])
        assert record["evidence_kind"] == kind, (measurement, model)
        assert record["measurement"] == measurement
        assert record["new_model_real_photo"] == (kind == "new_model_real_photo")
    assert evidence("docunet_stress", UVDOC)["model_provenance"] == "external_pretrained"
    assert evidence("predictions", {"kind": "checkpoint"})["training"] == "smoke_training"
    external = evidence("evaluator_calibration", {"provenance": "external_pretrained"})
    assert external["evidence_kind"] == "external_pretrained"
    assert calibration.CALIBRATION_EVIDENCE == external
    with pytest.raises(ValueError):
        evidence("docunet")


def _record(predictions, model, inputs):
    items = [
        {"output": f"{k}_{m}.png", "action": "written", "status": "success"}
        for k in range(1, 66)
        for m in (1, 2)
    ]
    record = {"schema": "flattener-dewarp-predictions-2", "model": model, "inputs": inputs}
    (predictions / "predictions.json").write_text(json.dumps({**record, "items": items}))


@pytest.mark.parametrize(
    ("model", "inputs", "kind"),
    [
        (None, None, "docunet_standard"),
        (OWN, "docunet_crop", "new_model_real_photo"),
        ({**OWN, "step": 50}, "docunet_crop", "smoke_training"),
        (UVDOC, "docunet_stress:rot90", "external_pretrained"),
    ],
)
def test_docunet_reports_carry_the_evidence_of_their_predictions(tiny_docunet, model, inputs, kind):
    scans, predictions = tiny_docunet
    if model is not None:
        _record(predictions, model, inputs)
    report = docunet.evaluate_docunet(
        predictions,
        pattern="{k}_{m}.png",
        conventions=["128_excl64"],
        build=None,
        references_dir=scans,
        verify_references=False,
        workers=1,
    )
    assert report["evidence"]["evidence_kind"] == kind
    stress = inputs is not None and inputs.startswith("docunet_stress")
    assert report["evidence"]["measurement"] == ("docunet_stress" if stress else "docunet_standard")
