"""Reproducible paired data and split integrity at the public pipeline boundary."""

import copy
import hashlib
import json
import random
import shutil
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from flattener import FlattenerError, Status, render
from flattener.synthetic.pipeline import (
    FAMILIES,
    DatasetConfig,
    PhotometricConfig,
    apply_photometry,
    derive_seed,
    generate_dataset,
    generate_sample,
    validate_dataset,
)


@pytest.fixture(scope="module")
def tiny_config():
    return DatasetConfig(
        seed=42,
        train_documents=1,
        validation_documents=1,
        test_documents=1,
        variants_per_document=2,
        output_size=(128, 128),
        photo_size=(128, 128),
    )


@pytest.fixture(scope="module")
def dataset_pair(tmp_path_factory, tiny_config):
    root = tmp_path_factory.mktemp("synthetic-pipeline")
    first, second = root / "first", root / "second"
    generate_dataset(first, config=tiny_config)
    generate_dataset(second, config=tiny_config)
    return first, second


@pytest.fixture
def editable_dataset(tmp_path, dataset_pair):
    root = tmp_path / "dataset"
    shutil.copytree(dataset_pair[0], root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return root, manifest


def _write_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _sample(**overrides):
    kwargs = {
        "document_id": "original-document",
        "document_seed": 71,
        "warp_seed": 82,
        "photometric_seed": 93,
        "output_size": (128, 128),
        "photo_size": (144, 160),
    }
    kwargs.update(overrides)
    return generate_sample(**kwargs)


def test_same_config_produces_identical_manifest_assets_and_content_hashes(dataset_pair):
    first, second = dataset_pair
    first_paths = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_paths = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_paths == second_paths
    assert len(first_paths) == 6 * 4 + 1
    for path in first_paths:
        assert (first / path).read_bytes() == (second / path).read_bytes(), str(path)
    result = validate_dataset(first)
    assert result["sample_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert result["documents"] == 3
    for sample in result["manifest"]["samples"]:
        for name, relative in sample["paths"].items():
            assert hashlib.sha256((first / relative).read_bytes()).hexdigest() == sample[
                "sha256"
            ][name]


def test_split_is_by_document_and_all_views_reuse_exact_flat_ground_truth(dataset_pair):
    root = dataset_pair[0]
    manifest = validate_dataset(root)["manifest"]
    document_splits = {document["id"]: document["split"] for document in manifest["documents"]}
    assert len(document_splits) == 3
    assert len({document["seed"] for document in manifest["documents"]}) == 3
    groups = {}
    for sample in manifest["samples"]:
        group = sample["document_group"]
        assert sample["split"] == document_splits[group]
        groups.setdefault(group, []).append(sample)
        metadata = json.loads((root / sample["paths"]["metadata.json"]).read_text())
        assert metadata["document_id"] == group
        assert metadata["document_group"] == group
        assert metadata["sample_id"] == sample["id"]
        assert metadata["split"] == sample["split"]
        assert metadata["capture_session"] == sample["capture_session"]
        assert metadata["is_real_photo"] is False
    assert set(sample["family"] for sample in manifest["samples"]) == set(FAMILIES)
    assert len({sample["capture_session"] for sample in manifest["samples"]}) == 6
    for samples in groups.values():
        assert len(samples) == 2
        assert samples[0]["sha256"]["flat.png"] == samples[1]["sha256"]["flat.png"]
        with Image.open(root / samples[0]["paths"]["flat.png"]) as first:
            with Image.open(root / samples[1]["paths"]["flat.png"]) as second:
                np.testing.assert_array_equal(np.array(first), np.array(second))
        assert samples[0]["sha256"]["photo.png"] != samples[1]["sha256"]["photo.png"]


@pytest.mark.parametrize("family", FAMILIES)
def test_all_families_create_renderer_compatible_geometry_and_masks(family):
    sample = _sample(family=family, photometric=PhotometricConfig(enabled=False))
    assert sample.flat.shape == (128, 128, 3)
    assert sample.photo.shape == (144, 160, 3)
    assert sample.flat.dtype == sample.photo.dtype == np.uint8
    assert sample.backward_map.shape == (128, 128, 2)
    assert sample.backward_map.dtype == np.float32
    assert sample.valid_mask.dtype == sample.photo_page_mask.dtype == np.bool_
    assert sample.valid_mask.shape == (128, 128)
    assert sample.photo_page_mask.shape == (144, 160)
    assert sample.valid_mask.all()
    assert sample.photo_page_mask.any() and not sample.photo_page_mask.all()
    assert np.isfinite(sample.backward_map[sample.valid_mask]).all()
    assert (np.abs(sample.backward_map[sample.valid_mask]) <= 1).all()

    corrected = render(sample.photo, sample.backward_map, target_mask=sample.valid_mask)

    assert corrected.status == Status.SUCCESS
    assert corrected.image.shape == sample.flat.shape
    np.testing.assert_array_equal(corrected.valid_mask, sample.valid_mask)
    assert corrected.metadata["validation"]["fold_check"] == "passed"
    # Two bilinear samples blur the text. Recovering the page should still be
    # much closer to its original than a spatially scrambled corrected page.
    flat = sample.flat.astype(np.float32)
    restored = corrected.image.astype(np.float32)
    reconstruction_error = np.mean(np.abs(restored - flat))
    scrambled_error = np.mean(np.abs(np.roll(restored, 32, axis=0) - flat))
    assert reconstruction_error < scrambled_error * 0.65


def test_photometric_settings_and_seeds_change_pixels_without_changing_ground_truth():
    plain = _sample(photometric=PhotometricConfig(enabled=False))
    augmented = _sample()
    different_seed = _sample(photometric_seed=94)

    for changed in (augmented, different_seed):
        assert not np.array_equal(changed.photo, plain.photo)
        for name in ("flat", "backward_map", "valid_mask", "photo_page_mask"):
            np.testing.assert_array_equal(getattr(changed, name), getattr(plain, name))
        assert changed.metadata["geometry"] == plain.metadata["geometry"]
        assert changed.metadata["document"] == plain.metadata["document"]
    assert not np.array_equal(augmented.photo, different_seed.photo)


def test_generating_sample_preserves_global_numpy_and_python_random_state():
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    _sample()

    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_photometry_is_repeatable_and_does_not_modify_or_alias_input():
    image = np.full((16, 19, 3), 180, dtype=np.uint8)
    original = image.copy()
    first, first_metadata = apply_photometry(image, 123)
    second, second_metadata = apply_photometry(image, 123)
    disabled, disabled_metadata = apply_photometry(image, 123, PhotometricConfig(enabled=False))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(disabled, original)
    assert first_metadata == second_metadata
    assert first_metadata["applied"] is True
    assert disabled_metadata["applied"] is False
    assert not np.shares_memory(image, first)
    assert not np.shares_memory(image, disabled)
    assert not np.array_equal(first, image)


def test_stored_geometry_can_be_loaded_without_pickle_and_consumed(dataset_pair):
    root = dataset_pair[0]
    manifest = validate_dataset(root)["manifest"]
    sample = manifest["samples"][0]
    with np.load(root / sample["paths"]["geometry.npz"], allow_pickle=False) as archive:
        assert set(archive.files) == {"backward_map", "valid_mask", "photo_page_mask"}
        with Image.open(root / sample["paths"]["photo.png"]) as photo:
            result = render(
                np.array(photo), archive["backward_map"], target_mask=archive["valid_mask"]
            )
        assert result.status == Status.SUCCESS
        np.testing.assert_array_equal(result.valid_mask, archive["valid_mask"])


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_generation_refuses_existing_output_and_preserves_its_contents(
    tmp_path, tiny_config, existing_kind
):
    destination = tmp_path / "existing"
    if existing_kind == "directory":
        destination.mkdir()
        sentinel = destination / "sentinel.txt"
    else:
        sentinel = destination
    sentinel.write_bytes(b"user-owned data")

    with pytest.raises(FlattenerError) as caught:
        generate_dataset(destination, config=tiny_config)

    assert caught.value.status == Status.IO_ERROR
    assert caught.value.reason_code == "dataset_create_failed"
    assert sentinel.read_bytes() == b"user-owned data"
    if destination.is_dir():
        assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("artifact", ["flat.png", "photo.png", "geometry.npz", "metadata.json"])
def test_changed_artifact_fails_hash_validation(editable_dataset, artifact):
    root, manifest = editable_dataset
    path = root / manifest["samples"][0]["paths"][artifact]
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(FlattenerError) as caught:
        validate_dataset(root)

    assert caught.value.reason_code == "artifact_hash_mismatch"


@pytest.mark.parametrize("collision", ["document", "sample", "session", "flat"])
def test_validator_rejects_document_sample_session_and_flat_leakage(editable_dataset, collision):
    root, manifest = editable_dataset
    first = manifest["samples"][0]
    other_split = manifest["samples"][2]
    if collision == "document":
        other_split["document_group"] = first["document_group"]
    elif collision == "sample":
        other_split["id"] = first["id"]
    elif collision == "session":
        other_split["capture_session"] = first["capture_session"]
    else:
        other_split["paths"]["flat.png"] = first["paths"]["flat.png"]
        other_split["sha256"]["flat.png"] = first["sha256"]["flat.png"]
    _write_manifest(root, manifest)

    with pytest.raises(FlattenerError) as caught:
        validate_dataset(root)

    assert caught.value.reason_code == "split_leakage"


@pytest.mark.parametrize("use_symlink", [False, True])
def test_validator_rejects_artifacts_outside_dataset_even_with_correct_hash(
    editable_dataset, use_symlink
):
    root, manifest = editable_dataset
    sample = manifest["samples"][0]
    external = root.parent / "outside.png"
    shutil.copyfile(root / sample["paths"]["flat.png"], external)
    if use_symlink:
        link = root / "escape.png"
        link.symlink_to(external)
        sample["paths"]["flat.png"] = "escape.png"
    else:
        sample["paths"]["flat.png"] = "../outside.png"
    _write_manifest(root, manifest)

    with pytest.raises(FlattenerError) as caught:
        validate_dataset(root)

    assert caught.value.reason_code == "invalid_dataset"


@pytest.mark.parametrize(
    "change",
    [
        "root-list",
        "missing-documents",
        "missing-samples",
        "documents-null",
        "samples-null",
        "document-null",
        "sample-null",
        "document-missing-id",
        "document-id-list",
        "sample-missing-group",
        "sample-missing-session",
        "sample-session-list",
        "sample-missing-split",
        "sample-paths-null",
        "sample-missing-path",
        "sample-path-null",
        "sample-hashes-null",
        "sample-missing-hash",
    ],
)
def test_malformed_manifest_returns_structured_validation_error(editable_dataset, change):
    root, manifest = editable_dataset
    if change == "root-list":
        manifest = []
    elif change.startswith("missing-"):
        del manifest[change.removeprefix("missing-")]
    elif change in ("documents-null", "samples-null"):
        manifest[change.removesuffix("-null")] = None
    elif change == "document-null":
        manifest["documents"][0] = None
    elif change == "sample-null":
        manifest["samples"][0] = None
    elif change == "document-missing-id":
        del manifest["documents"][0]["id"]
    elif change == "document-id-list":
        manifest["documents"][0]["id"] = []
    elif change == "sample-missing-group":
        del manifest["samples"][0]["document_group"]
    elif change == "sample-missing-session":
        del manifest["samples"][0]["capture_session"]
    elif change == "sample-session-list":
        manifest["samples"][0]["capture_session"] = []
    elif change == "sample-missing-split":
        del manifest["samples"][0]["split"]
    elif change == "sample-paths-null":
        manifest["samples"][0]["paths"] = None
    elif change == "sample-missing-path":
        del manifest["samples"][0]["paths"]["flat.png"]
    elif change == "sample-path-null":
        manifest["samples"][0]["paths"]["flat.png"] = None
    elif change == "sample-hashes-null":
        manifest["samples"][0]["sha256"] = None
    elif change == "sample-missing-hash":
        del manifest["samples"][0]["sha256"]["flat.png"]
    _write_manifest(root, manifest)

    with pytest.raises(FlattenerError) as caught:
        validate_dataset(root)

    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.reason_code in {"invalid_dataset", "split_leakage"}


@pytest.mark.parametrize("change", ["duplicate-document", "invalid-split", "empty-split"])
def test_invalid_split_membership_is_rejected(editable_dataset, change):
    root, manifest = editable_dataset
    if change == "duplicate-document":
        manifest["documents"].append(copy.deepcopy(manifest["documents"][0]))
    elif change == "invalid-split":
        manifest["documents"][0]["split"] = "evaluation"
    else:
        manifest["samples"] = [
            sample for sample in manifest["samples"] if sample["split"] != "test"
        ]
    _write_manifest(root, manifest)

    with pytest.raises(FlattenerError) as caught:
        validate_dataset(root)

    assert caught.value.reason_code in {"invalid_dataset", "split_leakage"}


@pytest.mark.parametrize("manifest_contents", [None, "{", '{"complete": false}', "{}"])
def test_missing_incomplete_or_unreadable_manifest_is_rejected(tmp_path, manifest_contents):
    if manifest_contents is not None:
        (tmp_path / "manifest.json").write_text(manifest_contents, encoding="utf-8")

    with pytest.raises(FlattenerError) as caught:
        validate_dataset(tmp_path)

    assert caught.value.reason_code == "invalid_dataset"


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5, "1", None])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(FlattenerError) as caught:
        derive_seed(seed, "document")
    assert caught.value.reason_code == "invalid_seed"


def test_derived_seeds_are_stable_label_delimited_and_domain_separated():
    assert derive_seed(123, "document", "한글") == derive_seed(123, "document", "한글")
    distinct = {
        derive_seed(123, "ab", "c"),
        derive_seed(123, "a", "bc"),
        derive_seed(123, "c", "ab"),
        derive_seed(124, "ab", "c"),
        derive_seed(123, "warp", "ab", "c"),
    }
    assert len(distinct) == 5
    assert all(0 <= value < 2**63 for value in distinct)


@pytest.mark.parametrize(
    "updates",
    [
        {"train_documents": 0},
        {"validation_documents": True},
        {"test_documents": 1.5},
        {"variants_per_document": 0},
        {"variants_per_document": 10_001},
        {"train_documents": 10_000, "variants_per_document": 11},
        {"families": ()},
        {"families": ("unknown",)},
        {"families": ["gentle"]},
        {"output_size": (127, 128)},
        {"output_size": (2049, 128)},
        {"photo_size": (0, 128)},
        {"photometric": None},
        {"photometric": PhotometricConfig(brightness_min=1.2, brightness_max=1.1)},
        {"photometric": PhotometricConfig(noise_std=float("nan"))},
    ],
)
def test_invalid_dataset_config_fails_before_creating_directory(tmp_path, tiny_config, updates):
    destination = tmp_path / "invalid-config"
    with pytest.raises(FlattenerError) as caught:
        generate_dataset(destination, config=replace(tiny_config, **updates))
    assert caught.value.status == Status.INPUT_ERROR
    assert not destination.exists()


@pytest.mark.parametrize(
    "config",
    [
        PhotometricConfig(enabled=1),
        PhotometricConfig(brightness_min=0),
        PhotometricConfig(brightness_max=float("inf")),
        PhotometricConfig(max_shadow=1),
        PhotometricConfig(max_blur_radius=-1),
        PhotometricConfig(noise_std=True),
    ],
)
def test_invalid_photometry_is_rejected(config):
    with pytest.raises(FlattenerError) as caught:
        apply_photometry(np.zeros((2, 2, 3), dtype=np.uint8), 1, config)
    assert caught.value.reason_code == "invalid_config"
