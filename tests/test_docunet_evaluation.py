"""Real-photo evaluation accounting, reference handling, and selection evidence."""

import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flattener import identity_map, render
from flattener.learning import inference


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_docunet.py"
    spec = importlib.util.spec_from_file_location("evaluate_docunet_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def forbid_real_ocr_processes(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A fixture must mock subprocess.run; real OCR is forbidden")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)


def test_discovery_preserves_all_130_cases_and_catches_duplicate_case_insensitive_names(
    runner, tmp_path
):
    photos = tmp_path / "extracted" / "original" / "original"
    scans = tmp_path / "extracted" / "scan" / "scan"
    photos.mkdir(parents=True)
    scans.mkdir(parents=True)
    for name in ("1_1.jpg", "1_2.JPG", "66_1.jpg"):
        (photos / name).touch()
    (scans / "1.png").touch()
    cases, unexpected = runner.discover_cases(tmp_path)
    assert len(cases) == 130
    assert cases[0]["id"] == "1_1"
    assert cases[1]["id"] == "1_2"
    assert cases[-1]["id"] == "65_2"
    assert not cases[0]["discovery_errors"]
    assert not cases[1]["discovery_errors"]
    assert len(cases[2]["discovery_errors"]) == 2
    assert unexpected == [str(photos / "66_1.jpg")]
    (photos / "1_1.JPG").touch()
    cases, _ = runner.discover_cases(tmp_path)
    assert cases[0]["photo"] is None
    assert "found 2" in cases[0]["discovery_errors"][0]


@pytest.mark.parametrize(
    "source,expected",
    [
        ((1000, 500), (512, 256)),
        ((500, 1000), (256, 512)),
        ((200, 150), (200, 150)),
        ((1200, 800), (512, 341)),
    ],
)
def test_reference_aspect_ratio_is_explicit_and_no_upscaling(runner, source, expected):
    assert runner.reference_output_size(source, 512) == expected


def test_reference_alpha_is_composited_on_white_without_changing_source(runner, tmp_path):
    path = tmp_path / "alpha.png"
    pixels = np.array([[[255, 0, 0, 0], [0, 0, 0, 255], [0, 0, 0, 128]]], dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    before = path.read_bytes()
    reference, info = runner.load_reference(path)
    np.testing.assert_array_equal(reference[0], [[255, 255, 255], [0, 0, 0], [127, 127, 127]])
    assert info["alpha_composited_on_white"] is True
    assert path.read_bytes() == before


def test_reference_exif_orientation_controls_output_aspect(runner, tmp_path):
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = 6
    with Image.new("RGB", (80, 40), (10, 20, 30)) as image:
        image.save(path, exif=exif)
    reference, info = runner.load_reference(path)
    assert reference.shape == (80, 40, 3)
    assert info["exif_orientation"] == 6
    assert runner.reference_output_size(reference.shape[:2], 64) == (64, 32)


def test_mismatched_checkpoint_validation_rejected_before_output_creation(runner, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"a checkpoint used only for hashing")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {"split": "validation", "evaluation_complete": True, "checkpoint_sha256": "0" * 64}
        )
    )
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner.evaluate_docunet(tmp_path, checkpoint, validation, output)
    assert not output.exists()


def test_missing_real_cases_are_reported_and_failures_get_preview_entries(
    runner, monkeypatch, tmp_path
):
    root = tmp_path / "data"
    photos = root / "extracted" / "original" / "original"
    scans = root / "extracted" / "scan" / "scan"
    photos.mkdir(parents=True)
    scans.mkdir(parents=True)
    with Image.new("RGB", (48, 64), (220, 210, 200)) as image:
        image.save(photos / "1_1.JPG")
        image.save(scans / "1.png")
    (root / "provenance.json").write_text(
        json.dumps({"license_spdx": None, "source_page": "local-test"})
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture checkpoint; predictor is mocked")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "split": "validation",
                "evaluation_complete": True,
                "checkpoint_sha256": runner.sha256(checkpoint),
            }
        )
    )
    sheets = []
    monkeypatch.setattr(
        inference, "load_predictor", lambda path: SimpleNamespace(model_sha256=runner.sha256(path))
    )
    monkeypatch.setattr(
        inference,
        "unwarp",
        lambda photo, predictor, output_size, config: render(
            photo, identity_map(output_size), config=config
        ),
    )
    monkeypatch.setattr(
        runner,
        "save_contact_sheet",
        lambda path, case_id, status, panels: sheets.append((case_id, status)),
    )
    report = runner.evaluate_docunet(root, checkpoint, validation, tmp_path / "output", max_side=64)
    assert report["expected_count"] == 130
    assert report["completed_count"] == 1
    assert report["failed_count"] == 129
    assert len(report["samples"]) == 130
    assert report["all_cases_accounted_for"] is True
    assert report["evaluation_complete"] is False
    assert report["quality_claim_allowed"] is False
    assert len(sheets) == 130
    first = report["samples"][0]
    assert first["status"] == "success"
    assert first["input_size_hw"] == first["output_size_hw"] == [64, 48]
    assert first["photo_sha256"] == runner.sha256(photos / "1_1.JPG")
    assert first["artifact_sha256"]["image"] == runner.sha256(Path(first["artifacts"]["image"]))
    assert report["samples"][-1]["id"] == "65_2"
    assert all(row["status"] == "input_error" for row in report["samples"][1:])
    assert "EPE" not in report.get("metrics", {})
    assert json.loads((tmp_path / "output" / "report.json").read_text()) == report


def test_contact_sheet_keeps_failed_panels_visible(runner, tmp_path):
    path = tmp_path / "preview.png"
    pixels = np.full((80, 40, 3), 100, dtype=np.uint8)
    runner.save_contact_sheet(
        path,
        "1_1",
        "invalid_transform",
        [
            ("Original", pixels),
            ("Reference", pixels),
            ("Identity", pixels),
            ("Model failed", None),
            ("Mask unavailable", None),
        ],
    )
    with Image.open(path) as sheet:
        assert sheet.width == 1200
        assert sheet.height > 300
        assert sheet.mode == "RGB"


def make_collection(root, runner, documents=range(1, 2)):
    files = root / "files"
    files.mkdir(parents=True)
    source = {
        "provenance": {"source_page": "fixture-source", "license_spdx": None},
        "samples": [],
        "files": [],
    }
    for document in documents:
        reference = files / f"docunet-reference-{document}.png"
        with Image.new("RGB", (16, 24), "white") as image:
            image.save(reference)
        for view in (1, 2):
            case_id = f"{document}_{view}"
            photo = files / f"docunet-photo-{case_id}.jpg"
            with Image.new("RGB", (16, 24), (180, 190, 200)) as image:
                image.save(photo)
            source["samples"].append(
                {
                    "id": case_id,
                    "document_group": f"docunet:{document}",
                    "split": "evaluation",
                    "training_eligible": False,
                    "paths": {
                        "photo": str(photo.relative_to(root)),
                        "reference": str(reference.relative_to(root)),
                    },
                    "sha256": {
                        "photo": runner.sha256(photo),
                        "reference": runner.sha256(reference),
                    },
                }
            )
            source["files"].append(
                {
                    "path": str(photo.relative_to(root)),
                    "original_path": f"extracted/original/{case_id}.jpg",
                    "sha256": runner.sha256(photo),
                    "bytes": photo.stat().st_size,
                }
            )
        source["files"].append(
            {
                "path": str(reference.relative_to(root)),
                "original_path": f"extracted/scan/{document}.png",
                "sha256": runner.sha256(reference),
                "bytes": reference.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "scene-0.3",
        "complete": True,
        "collection": {"schema_version": "dataset-collection-1.0", "sources": {"docunet": source}},
    }
    write_collection(root, manifest)
    return manifest, source


def write_collection(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def evaluate_fixture(root, runner, monkeypatch, tmp_path, *, ocr=False):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"never loaded; fixture inference only")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "split": "validation",
                "evaluation_complete": True,
                "checkpoint_sha256": runner.sha256(checkpoint),
            }
        )
    )
    calls = []
    monkeypatch.setattr(
        inference, "load_predictor", lambda path: SimpleNamespace(model_sha256=runner.sha256(path))
    )

    def fixture_unwarp(photo, predictor, output_size, config):
        calls.append(photo.shape)
        return render(photo, identity_map(output_size), config=config)

    monkeypatch.setattr(inference, "unwarp", fixture_unwarp)
    monkeypatch.setattr(runner, "save_contact_sheet", lambda *args: None)
    report = runner.evaluate_docunet(
        root, checkpoint, validation, tmp_path / "report", max_side=32, ocr=ocr
    )
    return report, calls


def test_collection_all_ids_and_shared_references_discovered(runner, tmp_path):
    make_collection(tmp_path, runner, range(1, 66))
    cases, unexpected = runner.discover_cases(tmp_path)
    assert [c["id"] for c in cases] == [f"{d}_{v}" for d in range(1, 66) for v in (1, 2)]
    assert not unexpected
    assert all(not c["discovery_errors"] for c in cases)
    assert cases[0]["reference"] == cases[1]["reference"]
    assert all(len(c["expected_sha256"]) == 2 for c in cases)


def test_collection_missing_duplicate_and_unknown_ids_are_not_hidden(runner, tmp_path):
    manifest, source = make_collection(tmp_path, runner)
    source["samples"].append(copy.deepcopy(source["samples"][0]))
    source["samples"].append({"id": "66_1"})
    write_collection(tmp_path, manifest)
    cases, unexpected = runner.discover_cases(tmp_path)
    assert len(cases) == 130
    assert "found 2" in cases[0]["discovery_errors"][0]
    assert cases[0]["photo"] is None
    assert not cases[1]["discovery_errors"]
    assert "found 0" in cases[-1]["discovery_errors"][0]
    assert any("66_1" in message for message in unexpected)


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.jpg",
        "/tmp/outside.jpg",
        "files/../outside.jpg",
        "files/./photo.jpg",
        "files//photo.jpg",
        "files\\photo.jpg",
        "C:/outside.jpg",
    ],
)
def test_collection_unsafe_sample_paths_fail_discovery(runner, tmp_path, relative):
    manifest, source = make_collection(tmp_path, runner)
    source["samples"][0]["paths"]["photo"] = relative
    write_collection(tmp_path, manifest)
    cases, _ = runner.discover_cases(tmp_path)
    assert cases[0]["photo"] is None
    assert any("path" in e for e in cases[0]["discovery_errors"])


@pytest.mark.parametrize("relative", ["../outside.jpg", "/tmp/outside.jpg", "files/../outside.jpg"])
def test_collection_unsafe_index_paths_are_rejected(runner, tmp_path, relative):
    manifest, source = make_collection(tmp_path, runner)
    source["files"][0]["path"] = relative
    write_collection(tmp_path, manifest)
    with pytest.raises(ValueError, match="path"):
        runner.discover_cases(tmp_path)


@pytest.mark.parametrize("external", [False, True])
def test_collection_rejects_symlink_even_if_target_stays_in_dataset(runner, tmp_path, external):
    root = tmp_path / "data"
    manifest, source = make_collection(root, runner)
    photo = root / source["samples"][0]["paths"]["photo"]
    target = tmp_path / "outside.jpg" if external else root / "files" / "other.jpg"
    target.write_bytes(photo.read_bytes())
    photo.unlink()
    photo.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runner.discover_cases(root)


def test_collection_rejects_symlinked_files_directory(runner, tmp_path):
    root = tmp_path / "data"
    make_collection(root, runner)
    (root / "files").rename(tmp_path / "outside")
    (root / "files").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner.discover_cases(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "train"),
        ("training_eligible", True),
        ("training_eligible", 0),
        ("document_group", "docunet:2"),
    ],
)
def test_collection_requires_correct_evaluation_only_identity(runner, tmp_path, field, value):
    manifest, source = make_collection(tmp_path, runner)
    source["samples"][0][field] = value
    write_collection(tmp_path, manifest)
    cases, _ = runner.discover_cases(tmp_path)
    assert "evaluation-only" in cases[0]["discovery_errors"][0]


def test_collection_conflicting_or_missing_sha_is_a_case_failure(runner, tmp_path):
    manifest, source = make_collection(tmp_path, runner)
    source["samples"][0]["sha256"]["photo"] = "0" * 64
    del source["samples"][1]["sha256"]["reference"]
    write_collection(tmp_path, manifest)
    cases, _ = runner.discover_cases(tmp_path)
    assert any("conflicts" in e for e in cases[0]["discovery_errors"])
    assert any("invalid reference SHA256" in e for e in cases[1]["discovery_errors"])


def test_collection_duplicate_file_index_is_rejected(runner, tmp_path):
    manifest, source = make_collection(tmp_path, runner)
    source["files"].append(copy.deepcopy(source["files"][0]))
    write_collection(tmp_path, manifest)
    with pytest.raises(ValueError, match="duplicate"):
        runner.discover_cases(tmp_path)


def test_collection_photo_path_cannot_cover_multiple_view_ids(runner, tmp_path):
    manifest, source = make_collection(tmp_path, runner)
    source["samples"][1]["paths"]["photo"] = source["samples"][0]["paths"]["photo"]
    write_collection(tmp_path, manifest)
    cases, unexpected = runner.discover_cases(tmp_path)
    assert all(any("reused" in e for e in c["discovery_errors"]) for c in cases[:2])
    assert any("unreferenced" in e for e in unexpected)


def test_collection_missing_payload_is_a_case_failure(runner, tmp_path):
    manifest, source = make_collection(tmp_path, runner)
    (tmp_path / source["samples"][0]["paths"]["photo"]).unlink()
    cases, _ = runner.discover_cases(tmp_path)
    assert any("missing photo" in e for e in cases[0]["discovery_errors"])


@pytest.mark.parametrize("role", ["photo", "reference"])
def test_collection_runtime_sha_mismatch_prevents_inference_and_counts_failure(
    runner, tmp_path, monkeypatch, role
):
    root = tmp_path / "data"
    manifest, source = make_collection(root, runner)
    target = root / source["samples"][0]["paths"][role]
    target.write_bytes(target.read_bytes() + b"corruption")
    report, calls = evaluate_fixture(root, runner, monkeypatch, tmp_path)
    failures = 129 if role == "photo" else 130
    assert report["failed_count"] == failures
    assert len(calls) == 130 - failures
    assert report["all_cases_accounted_for"] is True
    assert report["evaluation_complete"] is False
    assert "SHA256 mismatch" in report["samples"][0]["message"]
    assert report["samples"][0]["source_hashes_verified"] is False
    assert report["samples"][0][f"{role}_sha256"] != report["samples"][0]["expected_sha256"][role]


def test_collection_runtime_byte_mismatch_is_not_hidden(runner, tmp_path, monkeypatch):
    root = tmp_path / "data"
    manifest, source = make_collection(root, runner)
    source["files"][0]["bytes"] += 1
    write_collection(root, manifest)
    report, calls = evaluate_fixture(root, runner, monkeypatch, tmp_path)
    assert len(calls) == 1
    assert "byte length mismatch" in report["samples"][0]["message"]


def test_collection_report_embeds_provenance_and_manifest_identity(runner, tmp_path, monkeypatch):
    root = tmp_path / "data"
    make_collection(root, runner)
    report, calls = evaluate_fixture(root, runner, monkeypatch, tmp_path)
    assert len(calls) == report["completed_count"] == 2
    assert report["failed_count"] == 128
    assert report["dataset_layout"] == "collection"
    assert report["dataset_manifest_sha256"] == runner.sha256(root / "manifest.json")
    assert report["dataset_provenance_hash_method"] == "canonical-json-sorted-compact"
    assert report["source_page"] == "fixture-source"
    assert all(row["source_hashes_verified"] for row in report["samples"][:2])
    assert json.loads((tmp_path / "report" / "report.json").read_text()) == report


def test_present_invalid_manifest_never_falls_back_to_legacy(runner, tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": "scene-0.3", "complete": True})
    )
    with pytest.raises(ValueError, match="collection"):
        runner.discover_cases(tmp_path)


@pytest.mark.parametrize("complete", [None, False, 1, "true"])
def test_collection_requires_explicit_complete_true(runner, tmp_path, complete):
    manifest, _ = make_collection(tmp_path, runner)
    manifest["complete"] = complete
    write_collection(tmp_path, manifest)
    with pytest.raises(ValueError, match="complete: true"):
        runner.discover_cases(tmp_path)


def test_collection_manifest_size_is_bounded_before_json_decoding(runner, tmp_path):
    assert runner.MAX_MANIFEST_BYTES == 64 * 1024 * 1024
    with (tmp_path / "manifest.json").open("wb") as stream:
        stream.truncate(runner.MAX_MANIFEST_BYTES + 1)
    with pytest.raises(ValueError, match="64 MiB"):
        runner.discover_cases(tmp_path)


@pytest.mark.parametrize("unexpected_id", [False, True])
def test_complete_collection_report_requires_all_cases_and_no_extra_ids(
    runner, tmp_path, monkeypatch, unexpected_id
):
    root = tmp_path / "data"
    manifest, source = make_collection(root, runner, range(1, 66))
    if unexpected_id:
        source["samples"].append({"id": "66_1"})
        write_collection(root, manifest)
    report, calls = evaluate_fixture(root, runner, monkeypatch, tmp_path)
    assert len(calls) == report["completed_count"] == 130
    assert report["failed_count"] == 0
    assert report["all_cases_accounted_for"] is True
    assert report["evaluation_complete"] is (not unexpected_id)
    assert bool(report["unexpected_input_files"]) is unexpected_id
    assert all(row["source_hashes_verified"] is True for row in report["samples"])


@pytest.mark.parametrize("ocr", [False, True])
def test_cli_defaults_to_integrated_dataset(runner, monkeypatch, ocr):
    observed = {}

    def fake_evaluate(dataset_dir, *args, **kwargs):
        observed["dataset_dir"] = dataset_dir
        observed["ocr"] = kwargs["ocr"]
        return {"expected_count": 130, "completed_count": 0, "failed_count": 130}

    monkeypatch.setattr(runner, "evaluate_docunet", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_docunet",
            "--checkpoint",
            "fixture",
            "--validation-report",
            "fixture",
            "--output-dir",
            "fixture",
        ]
        + (["--ocr"] if ocr else []),
    )
    runner.main()
    assert observed["dataset_dir"] == Path("outputs/dataset")
    assert observed["ocr"] is ocr


def scene_validation(runner, model_hash):
    source = {"src/flattener/learning/evaluation.py": "1" * 64}
    runtime = {"torch": "fixture", "threads": 1}
    protocol = {
        "split": "validation",
        "checkpoint_sha256": model_hash,
        "training_manifest_sha256": "2" * 64,
        "dataset_manifest_sha256": "2" * 64,
        "source_code_sha256": source,
        "source_sha256": runner._canonical_sha256(source),
        "runtime": runtime,
        "runtime_sha256": runner._canonical_sha256(runtime),
    }
    samples = [
        {"id": "a", "epe_pixels": 2.0, "renderer_status": "success"},
        {"id": "b", "epe_pixels": 4.0, "renderer_status": "invalid_transform"},
    ]
    return {
        "schema_version": runner.SCENE_EVALUATION_SCHEMA,
        "evaluation_complete": True,
        "integrity_verified": True,
        "protocol": protocol,
        "protocol_sha256": runner._canonical_sha256(protocol),
        "summary": {
            "expected_count": 2,
            "processed_count": 2,
            "epe_missing_count": 0,
            "non_finite_failure_count": 0,
            "renderer_success_count": 1,
            "renderer_failure_count": 1,
            "renderer_missing_count": 0,
            "epe_pixels": {"count": 2, "mean": 3.0, "median": 3.0, "p95": 3.9},
        },
        "samples": samples,
        "samples_sha256": runner._canonical_sha256(samples),
    }


def test_scene_validation_allows_accounted_geometry_failures_without_quality_pass(runner):
    validation = scene_validation(runner, "3" * 64)
    assert runner._validation_checkpoint_format(validation, "3" * 64) == "coordinate-study"


@pytest.mark.parametrize("include_paper", [False, True])
def test_paper_validation_retains_explicit_dataset_scope(runner, include_paper):
    validation = scene_validation(runner, "3" * 64)
    validation["schema_version"] = runner.PAPER_EVALUATION_SCHEMA
    validation["protocol"]["include_paper_augmentation"] = include_paper
    for index, sample in enumerate(validation["samples"]):
        sample["source"] = "paper" if include_paper and index else "authored"
    validation["protocol_sha256"] = runner._canonical_sha256(validation["protocol"])
    validation["samples_sha256"] = runner._canonical_sha256(validation["samples"])
    assert runner._validation_checkpoint_format(validation, "3" * 64) == "coordinate-study"
    validation["protocol"].pop("include_paper_augmentation")
    validation["protocol_sha256"] = runner._canonical_sha256(validation["protocol"])
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner._validation_checkpoint_format(validation, "3" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_complete", False),
        ("evaluation_complete", 1),
        ("integrity_verified", False),
        ("integrity_verified", 1),
        ("schema_version", "unknown-evaluation"),
        ("protocol_sha256", "0" * 64),
        ("samples_sha256", "0" * 64),
        ("samples", []),
    ],
)
def test_scene_validation_requires_complete_integrity_bound_report(runner, field, value):
    validation = scene_validation(runner, "3" * 64)
    validation[field] = value
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner._validation_checkpoint_format(validation, "3" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("checkpoint_sha256", "4" * 64),
        ("dataset_manifest_sha256", "invalid"),
        ("training_manifest_sha256", None),
        ("source_sha256", "0" * 64),
        ("runtime_sha256", "0" * 64),
        ("source_code_sha256", {}),
        ("runtime", {}),
    ],
)
def test_scene_validation_rejects_invalid_protocol_even_with_refreshed_outer_hash(
    runner, field, value
):
    validation = scene_validation(runner, "3" * 64)
    validation["protocol"][field] = value
    validation["protocol_sha256"] = runner._canonical_sha256(validation["protocol"])
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner._validation_checkpoint_format(validation, "3" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_count", 0),
        ("processed_count", 1),
        ("processed_count", True),
        ("epe_missing_count", 1),
        ("non_finite_failure_count", 1),
        ("renderer_missing_count", 1),
        ("renderer_failure_count", 0),
        ("renderer_success_count", True),
        ("epe_pixels", {"count": 1}),
    ],
)
def test_scene_validation_rejects_missing_or_inconsistent_accounting(runner, field, value):
    validation = scene_validation(runner, "3" * 64)
    validation["summary"][field] = value
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner._validation_checkpoint_format(validation, "3" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "a"),
        ("epe_pixels", None),
        ("epe_pixels", True),
        ("epe_pixels", float("nan")),
        ("epe_pixels", -1.0),
        ("renderer_status", None),
        ("renderer_status", "success"),
    ],
)
def test_scene_validation_checks_rows_instead_of_trusting_summary(runner, field, value):
    validation = scene_validation(runner, "3" * 64)
    validation["samples"][1][field] = value
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner._validation_checkpoint_format(validation, "3" * 64)


@pytest.mark.parametrize(
    "field,value",
    [("split", "test"), ("evaluation_complete", 1), ("checkpoint_sha256", "4" * 64)],
)
def test_legacy_validation_gate_remains_strict(runner, field, value):
    validation = {
        "split": "validation",
        "evaluation_complete": True,
        "checkpoint_sha256": "3" * 64,
    }
    assert runner._validation_checkpoint_format(validation, "3" * 64) == "learning-0.1"
    validation[field] = value
    with pytest.raises(ValueError, match="frozen checkpoint"):
        runner._validation_checkpoint_format(validation, "3" * 64)


def test_scene_checkpoint_loader_and_manual_rubric_on_fixture_images(runner, tmp_path, monkeypatch):
    from flattener.learning import coordinate_study

    root = tmp_path / "data"
    make_collection(root, runner)
    checkpoint = tmp_path / "coordinate.pt"
    checkpoint.write_bytes(b"fixture weights; loading delegated to a checked mock")
    digest = runner.sha256(checkpoint)
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps(scene_validation(runner, digest)))
    loaded = []

    def load_coordinate(path):
        loaded.append(path)
        return SimpleNamespace(model_sha256=digest)

    monkeypatch.setattr(coordinate_study, "load_coordinate_predictor", load_coordinate)
    monkeypatch.setattr(inference, "load_predictor", lambda path: pytest.fail("legacy loader used"))
    monkeypatch.setattr(
        inference,
        "unwarp",
        lambda photo, predictor, output_size, config: render(
            photo, identity_map(output_size), config=config
        ),
    )
    monkeypatch.setattr(runner, "save_contact_sheet", lambda *args: None)
    report = runner.evaluate_docunet(root, checkpoint, validation, tmp_path / "report", max_side=32)
    assert loaded == [checkpoint]
    assert report["checkpoint_format"] == "coordinate-study"
    assert report["synthetic_validation_report_schema"] == runner.SCENE_EVALUATION_SCHEMA
    assert report["synthetic_validation_report_sha256"] == runner.sha256(validation)
    assert report["processed_count"] == 130
    assert report["quality_status"] == "not_evaluated"
    assert report["quality_claim_allowed"] is False
    assert report["manual_quality_rubric"]["criteria"] == runner.MANUAL_QUALITY_RUBRIC
    assert len(report["preselected_preview_ids"]) == 12
    assert "Oracle output aspect ratio" in report["reference_usage"]
    assert "OCR accuracy" in report["metrics_not_available"]
    assert "residual text-line curvature" in report["metrics_not_available"]
    for row in report["samples"]:
        assert row["quality_status"] == "not_evaluated"
        assert row["manual_quality_review"] == dict.fromkeys(
            runner.MANUAL_QUALITY_RUBRIC, "not_evaluated"
        )


def test_loaded_model_must_match_validation_snapshot(runner, tmp_path, monkeypatch):
    root = tmp_path / "data"
    make_collection(root, runner)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture model bytes")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "split": "validation",
                "evaluation_complete": True,
                "checkpoint_sha256": runner.sha256(checkpoint),
            }
        )
    )
    monkeypatch.setattr(
        inference, "load_predictor", lambda path: SimpleNamespace(model_sha256="0" * 64)
    )
    monkeypatch.setattr(inference, "unwarp", lambda *args, **kwargs: pytest.fail("inference ran"))
    with pytest.raises(ValueError, match="loaded checkpoint differs"):
        runner.evaluate_docunet(root, checkpoint, validation, tmp_path / "report", max_side=32)


def test_validation_report_is_bounded_before_decode(runner, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    validation = tmp_path / "validation.json"
    with validation.open("wb") as stream:
        stream.truncate(runner.MAX_MANIFEST_BYTES + 1)
    with pytest.raises(ValueError, match="validation report exceeds"):
        runner.evaluate_docunet(tmp_path, checkpoint, validation, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def text_info(text):
    return {
        "status": "success",
        "character_count": len(text),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def test_ocr_normalization_is_explicit_and_preserves_case_punctuation(runner):
    assert runner._normalize_ocr_text("  ﬁＡ\tCafe\u0301\n X!  ") == "fiA Café X!"
    assert runner._normalize_ocr_text("Aa,  aA.") == "Aa, aA."
    assert runner._normalize_ocr_text("\t\n \u00a0") == ""


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("", "", 0),
        ("a", "", 1),
        ("kitten", "sitting", 3),
        ("문서", "문자", 1),
        ("abc", "abc", 0),
        ("abc", "abXc", 1),
        ("a", "ABC", 3),
    ],
)
def test_ocr_edit_distance_is_exact_unicode_codepoint_distance(runner, left, right, expected):
    assert runner._edit_distance(left, right) == expected
    assert runner._edit_distance(right, left) == expected


def test_ocr_bit_vector_distance_matches_small_dynamic_programming_oracle(runner):
    rng = random.Random(1024)
    for _ in range(200):
        left = "".join(rng.choices("abé문 ", k=rng.randrange(17)))
        right = "".join(rng.choices("abé문 ", k=rng.randrange(17)))
        row = list(range(len(right) + 1))
        for i, first in enumerate(left, 1):
            current = [i]
            for j, second in enumerate(right, 1):
                current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (first != second)))
            row = current
        assert runner._edit_distance(left, right) == row[-1]


def test_ocr_request_fails_before_dataset_access_if_tesseract_missing(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner.shutil, "which", lambda executable: None)
    with pytest.raises(ValueError, match="--ocr requires"):
        runner.evaluate_docunet(
            "absent-data", "absent-model", "absent-report", tmp_path / "out", ocr=True
        )
    assert not (tmp_path / "out").exists()


def test_ocr_preflight_records_engine_and_language_hashes_without_images(
    runner, tmp_path, monkeypatch
):
    executable = tmp_path / "tesseract"
    executable.write_bytes(b"fixture executable")
    language = tmp_path / "eng.traineddata"
    language.write_bytes(b"fixture language")
    monkeypatch.setattr(runner.shutil, "which", lambda name: str(executable))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["shell"] is False and kwargs["timeout"] == 5
        assert kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
        output = (
            b"tesseract 5.5.3\n"
            if command[1] == "--version"
            else f'List of available languages in "{tmp_path}" (1):\neng\n'.encode()
        )
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(runner.subprocess, "run", run)
    engine = runner._prepare_ocr()
    assert calls == [[str(executable), "--version"], [str(executable), "--list-langs"]]
    assert engine["executable_sha256"] == runner.sha256(executable)
    assert engine["language_data_sha256"] == runner.sha256(language)
    assert engine["version"] == "tesseract 5.5.3"
    assert engine["language"] == "eng" and engine["psm"] == 6 and engine["threads"] == 1


def test_ocr_uses_temporary_equal_size_grayscale_and_never_saves_text(runner, monkeypatch):
    paths = []
    secret = "PRIVATE content α"

    def run(command, **kwargs):
        image = Path(command[1])
        paths.append(image)
        assert command[0] == "fixture-tesseract"
        assert command[2:] == ["stdout", "-l", "eng", "--psm", "6", "--dpi", "300"]
        assert kwargs["shell"] is False and kwargs["timeout"] == runner.OCR_TIMEOUT_SECONDS
        assert kwargs["cwd"] == str(image.parent)
        assert kwargs["env"]["OMP_THREAD_LIMIT"] == kwargs["env"]["OMP_NUM_THREADS"] == "1"
        with Image.open(image) as encoded:
            assert encoded.mode == "L" and encoded.size == (12, 16)
        return SimpleNamespace(returncode=0, stdout=f" {secret}\n".encode(), stderr=b"PRIVATE log")

    monkeypatch.setattr(runner.subprocess, "run", run)
    text, info = runner._recognize_ocr(
        np.full((24, 18, 3), 180, dtype=np.uint8), (16, 12), {"executable": "fixture-tesseract"}
    )
    assert text == secret
    assert info["character_count"] == len(secret)
    assert info["text_sha256"] == hashlib.sha256(secret.encode()).hexdigest()
    assert secret not in json.dumps(info)
    assert "PRIVATE log" not in json.dumps(info)
    assert all(not path.exists() and not path.parent.exists() for path in paths)


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("timeout", "timeout"),
        ("exit", "engine_failed"),
        ("invalid_utf8", "recognition_failed"),
        ("io", "recognition_failed"),
    ],
)
def test_ocr_failures_clean_temporary_files_and_hide_engine_output(
    runner, monkeypatch, mode, reason
):
    paths = []

    def run(command, **kwargs):
        paths.append(Path(command[1]))
        if mode == "timeout":
            raise runner.subprocess.TimeoutExpired(command, 30, output=b"PRIVATE text")
        if mode == "io":
            raise OSError("PRIVATE engine detail")
        return SimpleNamespace(
            returncode=7 if mode == "exit" else 0, stdout=b"\xff", stderr=b"PRIVATE"
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    text, info = runner._recognize_ocr(
        np.zeros((16, 12, 3), dtype=np.uint8), (16, 12), {"executable": "fixture"}
    )
    assert text is None and info["status"] == "failed" and info["reason"] == reason
    assert "PRIVATE" not in json.dumps(info)
    assert all(not path.exists() and not path.parent.exists() for path in paths)


@pytest.mark.parametrize(
    "limit,reason",
    [("OCR_MAX_TEXT_BYTES", "text_byte_limit"), ("OCR_MAX_TEXT_CHARS", "text_character_limit")],
)
def test_ocr_text_resource_limits_are_explicit_missing_results(runner, monkeypatch, limit, reason):
    monkeypatch.setattr(runner, limit, 2)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"abcd", stderr=b""),
    )
    text, info = runner._recognize_ocr(
        np.zeros((16, 12, 3), dtype=np.uint8), (16, 12), {"executable": "fixture"}
    )
    assert text is None and info["reason"] == reason


def test_ocr_missing_empty_failed_and_not_run_cases_keep_denominator_130(runner, monkeypatch):
    answers = iter(
        [
            (None, {"status": "failed", "reason": "timeout"}),
            ("", text_info("")),
            ("abc", text_info("abc")),
            (None, {"status": "failed", "reason": "timeout"}),
            ("", text_info("")),
        ]
    )
    calls = []

    def recognize(*args):
        calls.append(1)
        return next(answers)

    monkeypatch.setattr(runner, "_recognize_ocr", recognize)
    image = np.zeros((16, 12, 3), dtype=np.uint8)
    cache = {}
    rows = []
    for document in range(3):
        for _ in range(2):
            rows.append({"ocr": runner._ocr_case(image, image, (16, 12), {}, cache, document)})
    rows.extend({"ocr": {"status": "not_run", "scan_ocr_disagreement": None}} for _ in range(124))
    summary = runner._ocr_summary(rows, {}, cache)
    assert len(calls) == 5 and summary["reference_cache_entries"] == 3
    assert summary["reference_cache_hits"] == 3
    assert summary["expected_count"] == 130
    assert summary["measured_count"] == 1 and summary["missing_count"] == 129
    assert summary["failed_count"] == 3 and summary["reference_empty_count"] == 2
    assert summary["not_run_count"] == 124
    assert summary["scan_ocr_disagreement"]["mean"] == 1.0
    assert summary["unique_reference_status_counts"] == {"failed": 1, "empty": 1, "recognized": 1}
    assert summary["quality_status"] == "not_evaluated"


def test_ocr_is_opt_in_with_no_default_engine_calls_or_report_fields(runner, tmp_path, monkeypatch):
    root = tmp_path / "data"
    make_collection(root, runner)
    monkeypatch.setattr(runner, "_prepare_ocr", lambda: pytest.fail("default OCR preflight"))
    monkeypatch.setattr(runner, "_recognize_ocr", lambda *args: pytest.fail("default OCR run"))
    report, _ = evaluate_fixture(root, runner, monkeypatch, tmp_path)
    assert "ocr_diagnostic" not in report
    assert all("ocr" not in row for row in report["samples"])


def test_opt_in_ocr_shares_65_references_and_keeps_text_out_of_report(
    runner, tmp_path, monkeypatch
):
    root = tmp_path / "data"
    make_collection(root, runner, range(1, 66))
    monkeypatch.setattr(runner, "_prepare_ocr", lambda: {"version": "fixture", "language": "eng"})
    calls = []
    reference_text, prediction_text = "PRIVATE SCAN text", "PRIVATE SCAN texX"

    def recognize(pixels, size, engine):
        text = reference_text if len(calls) % 3 == 0 else prediction_text
        calls.append(size)
        return text, text_info(text)

    monkeypatch.setattr(runner, "_recognize_ocr", recognize)
    report, _ = evaluate_fixture(root, runner, monkeypatch, tmp_path, ocr=True)
    diagnostic = report["ocr_diagnostic"]
    assert len(calls) == 65 + 130
    assert diagnostic["reference_cache_entries"] == diagnostic["reference_cache_hits"] == 65
    assert diagnostic["measured_count"] == diagnostic["expected_count"] == 130
    assert diagnostic["missing_count"] == diagnostic["failed_count"] == 0
    assert diagnostic["scan_ocr_disagreement"]["mean"] == pytest.approx(1 / len(reference_text))
    assert report["quality_status"] == "not_evaluated" and report["quality_claim_allowed"] is False
    assert all(row["status"] == "success" for row in report["samples"])
    saved = (tmp_path / "report" / "report.json").read_text()
    assert reference_text not in saved and prediction_text not in saved
    assert "OCR accuracy" in report["metrics_not_available"]


def test_ocr_failure_does_not_change_successful_processing_status(runner, tmp_path, monkeypatch):
    root = tmp_path / "data"
    make_collection(root, runner)
    monkeypatch.setattr(runner, "_prepare_ocr", lambda: {"version": "fixture", "language": "eng"})
    calls = []

    def recognize(*args):
        calls.append(1)
        return None, {"status": "failed", "reason": "timeout"}

    monkeypatch.setattr(runner, "_recognize_ocr", recognize)
    report, _ = evaluate_fixture(root, runner, monkeypatch, tmp_path, ocr=True)
    assert len(calls) == 1
    assert report["completed_count"] == 2 and report["failed_count"] == 128
    assert all(row["status"] == "success" for row in report["samples"][:2])
    assert report["ocr_diagnostic"]["failed_count"] == 2
    assert report["ocr_diagnostic"]["not_run_count"] == 128
    assert report["ocr_diagnostic"]["missing_count"] == 130
    assert report["ocr_diagnostic"]["scan_ocr_disagreement"]["mean"] is None
