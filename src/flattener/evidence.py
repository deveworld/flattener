"""Evidence labels for the reports of ``flattener.dewarp`` and ``flattener.benchmark``.

Every report carries ``evidence``: what its numbers measure, where the model came from
and whether they count as a real-photo result of a new model.

``measurement``:

* ``gt_crop_validation``: training validation on held-out samples with ground truth,
  in views cut around the true page box (page, docunet, quarter turns, small page).
* ``gt_full_frame_validation``: the same samples, whole frame.
* ``training_loss``: losses on augmented training samples (training log, run status).
* ``docunet_standard``: DocUNet crops scored with ``flattener.benchmark``.
* ``docunet_stress``: DocUNet stress variants (whole photos, quarter turns, small pages).
* ``uvdoc_benchmark``: the 50 UVDoc benchmark photos scored with ``flattener.benchmark``.
* ``dir300_standard``: the 300 DIR300 photos scored with ``flattener.benchmark``.
* ``predictions``: rectified images without scores (``predictions.json``).
* ``evaluator_calibration``: the evaluator checked against published outputs.

``model_provenance`` is ``own_checkpoint`` (a ``flattener.dewarp`` checkpoint),
``own_previous_model`` (a model from the first study, such as page gate v0),
``external_pretrained`` (for example the UVDoc MIT weights or published outputs) or
``unknown``. ``training`` is ``smoke_training`` for an own checkpoint trained fewer than
``SMOKE_STEPS`` optimizer steps and ``training`` above that. ``initialization`` says
where an own checkpoint's training started: ``scratch``, ``external_pretrained`` (the
UVDoc weights, also through a parent checkpoint) or ``unknown``.

``new_model_real_photo`` is true only for real-photo benchmark results (DocUNet standard
or stress, UVDoc benchmark, DIR300) of an own checkpoint that is not a smoke run, whatever its
initialization. ``evidence_kind`` is one label for the whole record:
``external_pretrained``, then ``smoke_training``, then ``new_model_real_photo``, and
otherwise the measurement.
"""

from __future__ import annotations

from typing import Any

MEASUREMENTS = (
    "gt_crop_validation",
    "gt_full_frame_validation",
    "training_loss",
    "docunet_standard",
    "docunet_stress",
    "uvdoc_benchmark",
    "dir300_standard",
    "predictions",
    "evaluator_calibration",
)
REAL_PHOTO_MEASUREMENTS = frozenset(
    {"docunet_standard", "docunet_stress", "uvdoc_benchmark", "dir300_standard"}
)
PROVENANCES = ("own_checkpoint", "own_previous_model", "external_pretrained", "unknown")
INITIALIZATIONS = ("scratch", "external_pretrained", "unknown")
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


def initialization(model: dict[str, Any] | None) -> str | None:
    """Where an own checkpoint's training started (``INITIALIZATIONS``); ``None`` for
    other models.

    Reads an explicit ``initialization`` field, or else follows the first entry of
    ``lineage`` (through parent checkpoints) to a ``scratch`` or ``uvdoc`` start.
    """
    if not isinstance(model, dict) or model_provenance(model)[0] != "own_checkpoint":
        return None
    if model.get("initialization") in INITIALIZATIONS:
        return model["initialization"]
    lineage = model.get("lineage")
    while isinstance(lineage, list) and lineage and isinstance(lineage[0], dict):
        kind = lineage[0].get("kind")
        if kind == "scratch":
            return "scratch"
        if kind == "uvdoc":
            return "external_pretrained"
        if kind != "checkpoint":
            break
        lineage = lineage[0].get("lineage")
    return "unknown"


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
        "initialization": initialization(model),
        "new_model_real_photo": real_photo,
    }
