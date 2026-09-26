"""New and frozen legacy models on the same explicitly identified distribution."""

import hashlib
import json
import re
from pathlib import Path

import torch

from ..contracts import FlattenerError
from ..synthetic.scene_dataset import scene_loader, validate_scene_dataset
from .checkpoints import load_region_model
from .evaluation import RegionCase, _aggregate, evaluate_probabilities


def evaluate_region_study(
    dataset_dir,
    checkpoint_path,
    *,
    split="validation",
    output_dir=None,
    expected_training_manifest=None,
    threads=1,
):
    """Evaluate one frozen split; transfer requires an explicit training digest.

    An explicit digest does not make different distributions comparable to
    historical scores. It records which old model is intentionally transferred
    to this new dataset, rather than silently disabling the provenance guard.
    """
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if type(threads) is not int or not 1 <= threads <= 128:
        raise ValueError("threads must be an integer in [1,128]")
    destination = None if output_dir is None else Path(output_dir) / f"region-study-{split}.json"
    if destination is not None and destination.exists():
        raise FileExistsError(f"report already exists: {destination}")
    directory = Path(dataset_dir).resolve()
    checked = validate_scene_dataset(directory)
    digest = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    loaded = load_region_model(checkpoint_path)
    training_digest = loaded.metadata["manifest_sha256"]
    expected = digest if expected_training_manifest is None else expected_training_manifest
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or training_digest != expected
    ):
        raise ValueError(
            "checkpoint training manifest does not match the declared training dataset"
        )
    records = [row for row in checked["manifest"]["samples"] if row["split"] == split]
    if not records:
        raise ValueError("selected split is empty")
    cases, probabilities, errors = [], {}, []
    load = scene_loader(directory)
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        for record in records:
            sample = load(directory, record)
            cases.append(
                RegionCase(
                    record["id"],
                    sample.mask,
                    record["background_type"],
                    {
                        "document_group": record["document_group"],
                        "capture_session": record["capture_session"],
                        "negative_type": record.get("negative_type"),
                        "size_bin": record.get("size_bin"),
                    },
                )
            )
            try:
                probabilities[record["id"]] = loaded.predict_probability(sample.image)
            except (FlattenerError, TypeError, ValueError, RuntimeError) as error:
                if isinstance(error, RuntimeError) and any(
                    message in str(error).lower()
                    for message in ("out of memory", "can't allocate memory")
                ):
                    raise MemoryError(str(error)) from error
                probabilities[record["id"]] = None
                errors.append({"id": record["id"], "message": str(error)})
        report = evaluate_probabilities(cases, probabilities)
    finally:
        torch.set_num_threads(before)
    for method in report["methods"].values():
        negative = [row for row in method["samples"] if not row["has_page"]]
        method["by_negative_type"] = {
            kind: _aggregate([row for row in negative if row["metadata"]["negative_type"] == kind])
            for kind in sorted({row["metadata"]["negative_type"] for row in negative}, key=str)
        }
    report.update(
        schema_version="region-study-evaluation-0.2",
        split=split,
        dataset_schema=checked["manifest"]["schema_version"],
        dataset_task_scope=checked["manifest"].get("task_scope"),
        manifest_sha256=digest,
        training_manifest_sha256=training_digest,
        explicit_transfer_evaluation=(training_digest != digest),
        checkpoint_sha256=loaded.model_sha256,
        model_id=loaded.model_id,
        checkpoint_epoch=loaded.metadata["epoch"],
        expected_count=len(records),
        inference_errors=errors,
        cpu_threads=threads,
        test_used_for_model_selection=False,
        comparison_scope=(
            "compare models on this same split; not historical different-distribution scores"
        ),
        ground_truth_grid="original source pixels, never resized for evaluation",
    )
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return report
