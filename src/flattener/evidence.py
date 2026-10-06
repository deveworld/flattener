"""Measurement and model metadata for training and evaluation reports.

Records contain the measurement type, model origin, training step and initialization.
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
PROVENANCES = ("own_checkpoint", "own_previous_model", "external_pretrained", "unknown")
INITIALIZATIONS = ("scratch", "external_pretrained", "unknown")


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
    """Record the measurement and available model metadata."""
    if measurement not in MEASUREMENTS:
        raise ValueError(f"measurement must be one of {MEASUREMENTS}")
    provenance, step = model_provenance(model)
    return {
        "measurement": measurement,
        "model_provenance": provenance,
        "checkpoint_step": step,
        "initialization": initialization(model),
    }
