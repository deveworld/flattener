"""Evidence labels for the reports of ``flattener.dewarp`` and ``flattener.benchmark``.

Every report carries ``evidence``: what its numbers measure, where the model came from
and whether they count as a real-photo result of a new model.

``measurement``:

* ``gt_crop_validation``: training validation on held-out samples with ground truth,
  in views cut around the true page box (page, docunet, quarter turns, small page).
* ``gt_full_frame_validation``: the same samples, whole frame.
* ``docunet_standard``: DocUNet crops scored with ``flattener.benchmark``.
* ``docunet_stress``: DocUNet stress variants (whole photos, quarter turns, small pages).
* ``predictions``: rectified images without scores (``predictions.json``).
* ``evaluator_calibration``: the evaluator checked against published outputs.

``model_provenance`` is ``own_checkpoint`` (a ``flattener.dewarp`` checkpoint),
``external_pretrained`` (for example the UVDoc MIT weights or published outputs) or
``unknown``. ``training`` is ``smoke_training`` for an own checkpoint trained fewer than
``SMOKE_STEPS`` optimizer steps and ``training`` above that.

``new_model_real_photo`` is true only for DocUNet results (standard or stress) of an
own checkpoint that is not a smoke run. ``evidence_kind`` is one label for the whole
record: ``external_pretrained``, then ``smoke_training``, then ``new_model_real_photo``,
and otherwise the measurement.
"""

from __future__ import annotations

from typing import Any

MEASUREMENTS = (
    "gt_crop_validation",
    "gt_full_frame_validation",
    "docunet_standard",
    "docunet_stress",
    "predictions",
    "evaluator_calibration",
)
REAL_PHOTO_MEASUREMENTS = frozenset({"docunet_standard", "docunet_stress"})
PROVENANCES = ("own_checkpoint", "external_pretrained", "unknown")
SMOKE_STEPS = 10_000


def model_provenance(model: dict[str, Any] | None) -> tuple[str, int | None]:
    """Provenance and training step of a model record (``Dewarper.info`` and the like).

    ``kind: "checkpoint"`` is an own checkpoint, ``kind: "uvdoc"`` the external UVDoc
    weights; an explicit ``provenance`` field wins.
    """
    if not isinstance(model, dict):
        return "unknown", None
    provenance = model.get("provenance")
    if provenance not in PROVENANCES:
        provenance = {"checkpoint": "own_checkpoint", "uvdoc": "external_pretrained"}.get(
            model.get("kind"), "unknown"
        )
    step = model.get("step") if provenance == "own_checkpoint" else None
    return provenance, step if isinstance(step, int) and not isinstance(step, bool) else None


def evidence(measurement: str, model: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``evidence`` record of a report (see the module docstring)."""
    if measurement not in MEASUREMENTS:
        raise ValueError(f"measurement must be one of {MEASUREMENTS}")
    provenance, step = model_provenance(model)
    training = None
    if provenance == "own_checkpoint":
        training = "training" if step is not None and step >= SMOKE_STEPS else "smoke_training"
    real_photo = measurement in REAL_PHOTO_MEASUREMENTS and training == "training"
    if provenance == "external_pretrained":
        kind = "external_pretrained"
    elif training == "smoke_training":
        kind = "smoke_training"
    elif real_photo:
        kind = "new_model_real_photo"
    else:
        kind = measurement
    return {
        "evidence_kind": kind,
        "measurement": measurement,
        "model_provenance": provenance,
        "training": training,
        "checkpoint_step": step,
        "new_model_real_photo": real_photo,
    }
