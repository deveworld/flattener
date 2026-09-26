"""Metadata-only fixtures exercise the paired lock without training or test decoding."""

import copy
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flattener.regions import paired_selection as selection
from flattener.regions.checkpoints import model_spec
from flattener.regions.geometry import RegionConfig
from flattener.regions.training import LOSS_PROTOCOL, SELECTION_PROTOCOL
from flattener.synthetic.paired_scenes import PairedSceneConfig


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _dataset(root, mode):
    """Structured unit fixture; production dataset validator is separately tested."""
    samples, pairs, qa = [], [], {}
    for split, documents in zip(("train", "validation", "test"), (96, 24, 24), strict=True):
        matched = split != "train" or mode == "matched"
        qa[split] = {
            "comparisons_checked": documents,
            "matched_pairs_checked": documents if matched else 0,
            "byte_identical_after_128x96": 0,
            "excluded_or_regenerated": 0,
        }
        for document in range(documents):
            group = f"{split}-{document:03d}"
            first, blank, pair = group + "-view-000", group + "-blank-000", group + "-pair-000"
            if matched:
                pairs.append(
                    {
                        "id": pair,
                        "split": split,
                        "document_group": group,
                        "printed_id": first,
                        "blank_id": blank,
                    }
                )
            for variant in range(6):
                has_page = variant < 4
                is_blank = variant == 4
                identifier = (
                    group + f"-view-{variant:03d}"
                    if has_page
                    else blank
                    if is_blank
                    else group + "-background"
                )
                member = matched and variant in (0, 4)
                context_id = first if is_blank else identifier
                fields = {
                    "document_group": group,
                    "split": split,
                    "has_page": has_page,
                    "negative_type": None
                    if has_page
                    else "paper_panel"
                    if is_blank
                    else "background",
                    "context_id": context_id,
                    "background_type": "wood",
                    "size_bin": "small" if has_page else None,
                    "family": "perspective",
                    "panel_color_mode": mode if is_blank and split == "train" else None,
                    "pair_id": pair if member else None,
                    "pair_role": ("printed" if has_page else "blank") if member else None,
                    "partner_id": (blank if has_page else first) if member else None,
                    "capture_session": pair if member else identifier,
                }
                metadata = {
                    **fields,
                    "sample_id": identifier,
                    "image_size": [192, 144],
                    "seeds": {
                        "document": document,
                        "scene": context_id,
                        "background": document,
                        "photometric": document,
                    },
                    "source_document_sha256": _hash(group),
                    "blank_reference_size_bin": "small",
                    "pair_context": {"shared": group} if member else None,
                    "pair_context_sha256": _hash(group) if member else None,
                }
                paths = {
                    name: f"samples/{identifier}/{name}"
                    for name in ("image.png", "mask.png", "metadata.json")
                }
                _json(root / paths["metadata.json"], metadata)
                (root / paths["image.png"]).write_bytes(
                    (identifier + (mode if is_blank and split == "train" else "")).encode()
                )
                (root / paths["mask.png"]).write_bytes((identifier + "mask").encode())
                samples.append(
                    {
                        "id": identifier,
                        **fields,
                        "paths": paths,
                        "sha256": {
                            "image.png": _hash(
                                identifier + (mode if is_blank and split == "train" else "")
                            ),
                            "mask.png": _hash(identifier + "mask"),
                            "metadata.json": selection.sha256(root / paths["metadata.json"]),
                        },
                    }
                )
    manifest = {
        "schema_version": "scene-0.3",
        "complete": True,
        "config": selection._canonical(asdict(PairedSceneConfig(training_panels=mode))),
        "samples": samples,
        "pairs": pairs,
        "groups": [],
        "counts": selection._expected_counts(),
        "preprocessing_qa": qa,
        "generator_code_sha256": {key: _hash(key) for key in selection._GENERATOR_SOURCES},
        "versions": {"python": "fixture"},
    }
    _json(root / "manifest.json", manifest)
    return manifest


def _evaluation(state, digest, labels, common, manifest, *, iou=0.875, fp=0):
    rows, decisions = [], []
    negative = 0
    for identifier, has_page in labels.items():
        detected = has_page or negative < fp
        if not has_page:
            negative += 1
        rows.append(
            {
                "id": identifier,
                "has_page": has_page,
                "status": "completed",
                "iou": iou if has_page else float(not detected),
                "detected_page": detected,
            }
        )
        decisions.append({"id": identifier, "status": "accepted" if detected else "rejected"})
    mean = float(np.mean([row["iou"] for row in rows if row["has_page"]]))
    passed = mean >= 0.8 and fp / 48 <= 0.1
    summary = {
        "expected_count": 144,
        "completed_count": 144,
        "failed_count": 0,
        "foreground_count": 96,
        "negative_count": 48,
        "foreground_iou_all_cases": {"count": 96, "mean": mean},
        "negative_false_positive_count_conservative": fp,
        "negative_false_positive_rate_conservative": fp / 48,
        "gate_passed": passed,
    }
    report = {
        "schema_version": "region-study-evaluation-0.2",
        "dataset_schema": "scene-0.3",
        "split": "validation",
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
        "region_accepted_count": 96 + fp,
        "region_rejected_count": 48 - fp,
        "region_failure_count": 0,
        "manifest_sha256": common,
        "training_manifest_sha256": state["manifest_sha256"],
        "explicit_transfer_evaluation": state["manifest_sha256"] != common,
        "checkpoint_sha256": digest,
        "checkpoint_epoch": state["epoch"],
        "cpu_threads": 1,
        "evaluation_provenance": {
            key: state["provenance"][key]
            for key in selection._ENVIRONMENT_KEYS
            if key != "uv_lock_sha256"
        },
    }
    report["paired_diagnostics"] = selection.paired_region_diagnostics(manifest, report)
    return report


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    matched, independent = tmp_path / "matched", tmp_path / "independent"
    matched_manifest = _dataset(matched, "matched")
    _dataset(independent, "independent")
    # Tests exercise the selection layer with structured metadata, never invoke
    # the generator or train a model. Production always uses the real validator.
    monkeypatch.setattr(
        selection,
        "validate_paired_scene_dataset",
        lambda root: {"manifest": json.loads((root / "manifest.json").read_text())},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no image decode or inference permitted")

    monkeypatch.setattr(Image, "open", forbidden)
    environment = {
        "source_code_sha256": {key: _hash(key) for key in selection._GENERATOR_SOURCES},
        "uv_lock_sha256": _hash("dependencies"),
        "versions": {"python": "fixture"},
        "device": "cpu",
        "preprocess_version": "fixture",
        "example_code_sha256": {"examples/evaluate_region_study.py": _hash("evaluation")},
        "pyproject_sha256": _hash("project"),
    }
    monkeypatch.setattr(selection, "_environment", lambda: copy.deepcopy(environment))
    states, arms = {}, {}
    common = selection.sha256(matched / "manifest.json")
    labels = {
        row["id"]: row["has_page"]
        for row in json.loads((matched / "manifest.json").read_text())["samples"]
        if row["split"] == "validation"
    }
    for name, loss in (("A", 0.1), ("B", 0.5), ("C", 0.4)):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "best.pt"
        path.write_bytes(name.encode())
        digest = selection.sha256(path)
        manifest = (
            selection.BASELINE_MANIFEST_SHA256
            if name == "A"
            else selection.sha256((independent if name == "B" else matched) / "manifest.json")
        )
        epoch = 73 if name == "A" else 80
        provenance = {
            **{key: copy.deepcopy(environment[key]) for key in selection._ENVIRONMENT_KEYS},
            "loss": copy.deepcopy(LOSS_PROTOCOL),
            "selection": copy.deepcopy(SELECTION_PROTOCOL),
            "dataset_schema": "scene-0.2" if name == "A" else "scene-0.3",
        }
        history = [
            {"epoch": i, "train_loss": 1.0, "validation": {"loss": loss + (epoch - i) / 100}}
            for i in range(1, epoch + 1)
        ]
        state = {
            "schema_version": "region-study-0.2",
            "epoch": epoch,
            "best_epoch": epoch,
            "model_spec": selection._canonical(model_spec("context")),
            "config": dict(selection.TRAINING_CONFIG),
            "manifest_sha256": manifest,
            "dataset_counts": dict(selection.EXPECTED_COUNTS),
            "initial_validation": {"loss": 2.0},
            "best_validation": {"loss": loss},
            "history": history,
            "provenance": provenance,
        }
        states[path.resolve()] = state
        evaluation = folder / "validation.json"
        _json(evaluation, _evaluation(state, digest, labels, common, matched_manifest))
        arms[name] = {"checkpoint": path, "validation_report": evaluation}
        if name == "A":
            monkeypatch.setattr(selection, "BASELINE_CHECKPOINT_SHA256", digest)
        else:
            training_path = folder / "training.json"
            _json(
                training_path, {**state, "epochs_completed": 80, "checkpoints": {"best.pt": digest}}
            )
            arms[name]["training_report"] = training_path
    monkeypatch.setattr(
        selection,
        "read_region_checkpoint",
        lambda path: copy.deepcopy(states[Path(path).resolve()]),
    )
    coordinate = tmp_path / "coordinate.pt"
    coordinate.write_bytes(b"coordinate-fixture")
    coordinate_digest = selection.sha256(coordinate)
    monkeypatch.setattr(selection, "COORDINATE_CHECKPOINT_SHA256", coordinate_digest)
    monkeypatch.setattr(
        selection,
        "load_predictor",
        lambda path: SimpleNamespace(
            checkpoint_path=str(Path(path).resolve()),
            model_sha256=selection.sha256(path),
            predict=forbidden,
        ),
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Fixed unit-test design; no production evidence.\n")
    yield SimpleNamespace(
        matched=matched,
        independent=independent,
        arms=arms,
        states=states,
        coordinate=coordinate,
        protocol=protocol,
        directory=tmp_path / "lock",
        output=tmp_path / "selection.json",
        environment=environment,
        labels=labels,
    )
    # These two dense, generated fixture trees are disposable. Do not retain
    # 60 x 1,728 tiny dataset artifacts in pytest's three historical runs.
    shutil.rmtree(matched)
    shutil.rmtree(independent)


def _lock(e):
    return selection.lock_paired_study(
        e.matched,
        e.independent,
        e.directory,
        protocol_path=e.protocol,
        baseline_checkpoint=e.arms["A"]["checkpoint"],
        coordinate_checkpoint=e.coordinate,
    )


def _freeze(e):
    return selection.freeze_paired_study(
        e.arms,
        e.output,
        lock_path=e.directory / "preregistration.json",
        coordinate_checkpoint=e.coordinate,
    )


def _edit(path, callback):
    value = json.loads(path.read_text())
    callback(value)
    _json(path, value)


def _metrics(e, name, *, iou=0.875, fp=0):
    state = e.states[e.arms[name]["checkpoint"].resolve()]
    _json(
        e.arms[name]["validation_report"],
        _evaluation(
            state,
            selection.sha256(e.arms[name]["checkpoint"]),
            e.labels,
            selection.sha256(e.matched / "manifest.json"),
            json.loads((e.matched / "manifest.json").read_text()),
            iou=iou,
            fp=fp,
        ),
    )


def test_lock_records_exact_protocol_contrast_and_no_decoding(evidence):
    result = _lock(evidence)
    assert result["contrast"]["changed_training_rgb_count"] == 96
    assert result["contrast"]["changed_training_metadata_count"] == 192
    assert result["contrast"]["evaluation_records_and_artifact_hashes_equal"] is True
    assert result["test_pixels_decoded"] is False
    assert (evidence.directory / "protocol.md").read_bytes() == evidence.protocol.read_bytes()
    lockfile = evidence.directory / "preregistration.json"
    assert lockfile.with_name(lockfile.name + ".sha256").read_text().strip() == selection.sha256(
        lockfile
    )
    assert result["fixed_protocol"]["training_config"]["seed"] == 20260923
    assert result["fixed_protocol"]["dataset_seed"] == 20260924


def test_gate_selects_lower_loss_and_baseline_never_competes(evidence):
    _lock(evidence)
    result = _freeze(evidence)
    assert result["selected_arm"] == "C"
    assert result["arms"]["A"]["eligible_for_selection"] is False
    assert result["arms"]["A"]["selection_validation_weighted_loss"] == 0.1
    assert result["arms"]["B"]["explicit_transfer_evaluation"] is True
    assert result["arms"]["C"]["explicit_transfer_evaluation"] is False
    assert result["initial_validation_metrics_equal"] is True
    assert result["test_evaluation_performed"] is False
    assert json.loads(evidence.output.read_text()) == result


@pytest.mark.parametrize(
    "b_iou,c_iou,b_fp,c_fp,expected",
    [
        (0.875, 0.7, 0, 0, "B"),
        (0.7, 0.875, 0, 0, "C"),
        (0.7, 0.7, 0, 0, None),
        (0.875, 0.875, 4, 5, "B"),
        (0.875, 0.875, 5, 5, None),
    ],
)
def test_fixed_gate_before_loss(evidence, b_iou, c_iou, b_fp, c_fp, expected):
    _lock(evidence)
    _metrics(evidence, "B", iou=b_iou, fp=b_fp)
    _metrics(evidence, "C", iou=c_iou, fp=c_fp)
    assert _freeze(evidence)["selected_arm"] == expected


def test_exact_loss_tie_retains_b(evidence):
    state = evidence.states[evidence.arms["C"]["checkpoint"].resolve()]
    state["best_validation"]["loss"] = 0.5
    state["history"][-1]["validation"]["loss"] = 0.5
    # Make all earlier losses strictly larger than the tied selected minimum.
    for row in state["history"][:-1]:
        row["validation"]["loss"] += 0.1
    _edit(
        evidence.arms["C"]["training_report"],
        lambda value: value.update(
            best_validation=state["best_validation"], history=state["history"]
        ),
    )
    _lock(evidence)
    assert _freeze(evidence)["selected_arm"] == "B"


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("independent", "seed", 5),
        ("matched", "image_size", [128, 96]),
        ("matched", "training_panels", "independent"),
    ],
)
def test_nonprotocol_data_config_rejected_before_output(evidence, target, field, value):
    _edit(getattr(evidence, target) / "manifest.json", lambda m: m["config"].update({field: value}))
    with pytest.raises(ValueError, match="configuration"):
        _lock(evidence)
    assert not evidence.directory.exists()


@pytest.mark.parametrize(
    "kind",
    ["positive_rgb", "blank_rgb_unchanged", "mask", "eval_record", "ordering", "outside_metadata"],
)
def test_controlled_contrast_rejects_unplanned_differences(evidence, kind):
    def alter(m):
        if kind == "positive_rgb":
            m["samples"][0]["sha256"]["image.png"] = "f" * 64
        elif kind == "blank_rgb_unchanged":
            original = json.loads((evidence.matched / "manifest.json").read_text())
            m["samples"][4]["sha256"]["image.png"] = original["samples"][4]["sha256"]["image.png"]
        elif kind == "mask":
            m["samples"][4]["sha256"]["mask.png"] = "f" * 64
        elif kind == "eval_record":
            m["samples"][576]["capture_session"] = "changed"
        elif kind == "ordering":
            m["samples"][0], m["samples"][1] = m["samples"][1], m["samples"][0]
        else:
            m["unexpected"] = True

    _edit(evidence.independent / "manifest.json", alter)
    with pytest.raises(ValueError):
        _lock(evidence)
    assert not evidence.directory.exists()


def test_nonblank_metadata_only_allows_pair_linkage(evidence):
    manifest = json.loads((evidence.independent / "manifest.json").read_text())
    _edit(
        evidence.independent / manifest["samples"][0]["paths"]["metadata.json"],
        lambda value: value.update(unplanned="change"),
    )
    _edit(
        evidence.independent / "manifest.json",
        lambda value: value["samples"][0]["sha256"].update(
            {
                "metadata.json": selection.sha256(
                    evidence.independent / manifest["samples"][0]["paths"]["metadata.json"]
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="nonblank training metadata"):
        _lock(evidence)


@pytest.mark.parametrize(
    "target",
    [
        "original_protocol",
        "snapshot_protocol",
        "manifest",
        "source",
        "example",
        "versions",
        "uv_lock",
        "lock_bytes",
    ],
)
def test_freeze_rejects_mutation_after_lock(evidence, target):
    _lock(evidence)
    if target == "original_protocol":
        evidence.protocol.write_text("changed")
    elif target == "snapshot_protocol":
        (evidence.directory / "protocol.md").write_text("changed")
    elif target == "manifest":
        path = evidence.matched / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif target == "source":
        evidence.environment["source_code_sha256"]["new.py"] = "f" * 64
    elif target == "example":
        evidence.environment["example_code_sha256"]["examples/new.py"] = "f" * 64
    elif target == "versions":
        evidence.environment["versions"]["python"] = "changed"
    elif target == "uv_lock":
        evidence.environment["uv_lock_sha256"] = "f" * 64
    else:
        path = evidence.directory / "preregistration.json"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        _freeze(evidence)
    assert not evidence.output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("dataset_schema", "scene-0.2"),
        ("manifest_sha256", "f" * 64),
        ("training_manifest_sha256", "f" * 64),
        ("checkpoint_sha256", "f" * 64),
        ("checkpoint_epoch", 79),
        ("explicit_transfer_evaluation", False),
        ("test_used_for_model_selection", True),
    ],
)
def test_invalid_evaluation_evidence_cannot_freeze(evidence, field, value):
    _lock(evidence)
    _edit(evidence.arms["B"]["validation_report"], lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        _freeze(evidence)
    assert not evidence.output.exists()


def test_reports_must_match_actual_locked_validation_ids(evidence):
    _lock(evidence)
    for arm in evidence.arms.values():

        def replace(report):
            report["methods"]["learned_postprocessed"]["samples"][0]["id"] = "invented-id"
            report["region_decisions"][0]["id"] = "invented-id"

        _edit(arm["validation_report"], replace)
    with pytest.raises(ValueError, match="locked validation identities"):
        _freeze(evidence)


@pytest.mark.parametrize(
    "change",
    [
        "short_budget",
        "different_seed",
        "wrong_manifest",
        "wrong_loss",
        "wrong_source",
        "initial_metrics",
        "wrong_architecture",
        "late_tie",
    ],
)
def test_incompatible_training_evidence_rejected(evidence, change):
    _lock(evidence)
    path = evidence.arms["B"]["training_report"]
    state = evidence.states[evidence.arms["B"]["checkpoint"].resolve()]

    def alter(training):
        if change == "short_budget":
            training["epochs_completed"] = 79
        elif change == "different_seed":
            training["config"]["seed"] = 20260924
        elif change == "wrong_manifest":
            training["manifest_sha256"] = selection.sha256(evidence.matched / "manifest.json")
        elif change == "wrong_loss":
            training["best_validation"]["loss"] = 0.001
        elif change == "wrong_source":
            training["provenance"]["source_code_sha256"] = {"wrong": "f" * 64}
        elif change == "initial_metrics":
            training["initial_validation"] = state["initial_validation"] = {"loss": 2.1}
        elif change == "wrong_architecture":
            training["model_spec"] = selection._canonical(model_spec("local"))
        else:
            training["history"][-2]["validation"] = copy.deepcopy(training["best_validation"])
            state["history"] = copy.deepcopy(training["history"])

    _edit(path, alter)
    with pytest.raises(ValueError):
        _freeze(evidence)
    assert not evidence.output.exists()


def test_no_dataset_validation_artifact_reads_during_freeze(evidence, monkeypatch):
    _lock(evidence)

    def forbidden(*args, **kwargs):
        raise AssertionError("freeze must not validate/decode saved dataset artifacts")

    monkeypatch.setattr(selection, "validate_paired_scene_dataset", forbidden)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "samples" not in path.parts, "saved dataset artifact read during freeze"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert _freeze(evidence)["selected_arm"] == "C"


def test_write_once_lock_and_selection_preserve_existing_bytes(evidence):
    _lock(evidence)
    lockbytes = (evidence.directory / "preregistration.json").read_bytes()
    with pytest.raises(FileExistsError):
        _lock(evidence)
    assert (evidence.directory / "preregistration.json").read_bytes() == lockbytes
    _freeze(evidence)
    selected = evidence.output.read_bytes()
    with pytest.raises(FileExistsError):
        _freeze(evidence)
    assert evidence.output.read_bytes() == selected


def test_publication_race_preserves_concurrent_selection(evidence, monkeypatch):
    _lock(evidence)
    original = selection.os.link

    def competing(source, target):
        if Path(target) == evidence.output:
            evidence.output.write_text("concurrent record")
        return original(source, target)

    monkeypatch.setattr(selection.os, "link", competing)
    with pytest.raises(FileExistsError):
        _freeze(evidence)
    assert evidence.output.read_text() == "concurrent record"
    assert not list(evidence.output.parent.glob(".paired-freeze-*"))


def test_evidence_change_during_freeze_not_published(evidence, monkeypatch):
    _lock(evidence)
    original = selection._coordinate

    def altered(path):
        result = original(path)
        file = evidence.arms["B"]["validation_report"]
        file.write_bytes(file.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(selection, "_coordinate", altered)
    with pytest.raises(ValueError, match="evidence changed"):
        _freeze(evidence)
    assert not evidence.output.exists()


@pytest.mark.parametrize(
    "placeholder", ["Pending: sources", "manifest pending generation lock", "identity lock pending"]
)
def test_incomplete_protocol_is_not_lockable(evidence, placeholder):
    evidence.protocol.write_text(placeholder)
    with pytest.raises(ValueError, match="unresolved"):
        _lock(evidence)


def test_nonfinite_and_oversized_numeric_reports_fail(evidence):
    _lock(evidence)
    report = evidence.arms["B"]["validation_report"]
    _edit(
        report,
        lambda value: value["methods"]["learned_postprocessed"]["samples"][0].update(iou=10**1000),
    )
    with pytest.raises(ValueError, match="finite"):
        _freeze(evidence)
    assert not evidence.output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_threads", 2),
        ("cpu_threads", True),
        ("evaluation_provenance", {}),
        ("paired_diagnostics", {}),
    ],
)
def test_fixed_evaluation_runtime_and_recomputed_diagnostics_required(evidence, field, value):
    _lock(evidence)
    _edit(evidence.arms["B"]["validation_report"], lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        _freeze(evidence)
    assert not evidence.output.exists()


def test_pair_secondary_failures_are_not_an_additional_selection_gate(evidence):
    _lock(evidence)
    manifest = json.loads((evidence.matched / "manifest.json").read_text())
    paired = {pair["printed_id"] for pair in manifest["pairs"] if pair["split"] == "validation"}

    def secondary_fails(report):
        rows = report["methods"]["learned_postprocessed"]["samples"]
        for row in rows:
            if row["has_page"]:
                row["iou"] = 0.79 if row["id"] in paired else 1.0
        report["methods"]["learned_postprocessed"]["summary"]["foreground_iou_all_cases"][
            "mean"
        ] = float(np.mean([row["iou"] for row in rows if row["has_page"]]))
        report["paired_diagnostics"] = selection.paired_region_diagnostics(manifest, report)

    _edit(evidence.arms["B"]["validation_report"], secondary_fails)
    _metrics(evidence, "C", iou=0.7)
    result = _freeze(evidence)
    assert result["selected_arm"] == "B"
    assert result["arms"]["B"]["gate_passed"] is True
    diagnostic = result["arms"]["B"]["paired_diagnostics"]
    assert diagnostic["expected_pair_count"] == 24
    assert diagnostic["printed_iou_ge_0_8_and_blank_rejected_count"] == 0
    assert diagnostic["additional_gate"] is False


def test_generation_provenance_must_match_current_sources(evidence):
    for root in (evidence.matched, evidence.independent):
        _edit(
            root / "manifest.json",
            lambda m: m["generator_code_sha256"].update({"synthetic/paired_scenes.py": "f" * 64}),
        )
    with pytest.raises(ValueError, match="generation provenance"):
        _lock(evidence)
    assert not evidence.directory.exists()


def test_artifact_mutation_after_initial_validation_cannot_be_locked(evidence, monkeypatch):
    original = selection._environment
    calls = 0

    def mutate_after_validation():
        nonlocal calls
        calls += 1
        if calls == 2:
            manifest = json.loads((evidence.matched / "manifest.json").read_text())
            path = evidence.matched / manifest["samples"][0]["paths"]["image.png"]
            path.write_bytes(b"changed after metadata validation")
        return original()

    monkeypatch.setattr(selection, "_environment", mutate_after_validation)
    with pytest.raises(ValueError, match="locked evidence changed"):
        _lock(evidence)
    assert not evidence.directory.exists()


def test_metadata_mutation_must_match_recorded_hash(evidence):
    manifest = json.loads((evidence.independent / "manifest.json").read_text())
    path = evidence.independent / manifest["samples"][0]["paths"]["metadata.json"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="metadata hash changed"):
        _lock(evidence)


def test_blank_paper_warp_seed_extras_are_treatment_specific(evidence):
    # The matched blank inherits paper/warp seeds; independent v2 panels do not.
    manifest = json.loads((evidence.matched / "manifest.json").read_text())
    row = manifest["samples"][4]
    path = evidence.matched / row["paths"]["metadata.json"]
    _edit(path, lambda m: m["seeds"].update(paper=123, warp=456))
    row["sha256"]["metadata.json"] = selection.sha256(path)
    _json(evidence.matched / "manifest.json", manifest)
    assert _lock(evidence)["contrast"]["changed_training_rgb_count"] == 96
