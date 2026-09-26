"""Artifact loading validates numeric ground truth and sample identity."""

import hashlib
import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from flattener import FlattenerError, Status
from flattener.synthetic.pipeline import (
    DatasetConfig,
    generate_dataset,
    load_sample,
    validate_dataset,
)


@pytest.fixture(scope="module")
def loading_dataset(tmp_path_factory):
    destination = tmp_path_factory.mktemp("loading") / "dataset"
    generate_dataset(
        destination,
        config=DatasetConfig(
            seed=301,
            train_documents=1,
            validation_documents=1,
            test_documents=1,
            variants_per_document=1,
            output_size=(128, 144),
            photo_size=(144, 160),
        ),
    )
    return destination


@pytest.fixture
def editable_sample(tmp_path, loading_dataset):
    root = tmp_path / "dataset"
    shutil.copytree(loading_dataset, root)
    manifest = validate_dataset(root)["manifest"]
    record = manifest["samples"][0]
    with np.load(root / record["paths"]["geometry.npz"], allow_pickle=False) as archive:
        geometry = {name: archive[name].copy() for name in archive.files}
    return root, manifest, record, geometry


def _publish_changed_artifact(root, manifest, record, name, contents):
    path = root / record["paths"][name]
    path.write_bytes(contents)
    record["sha256"][name] = hashlib.sha256(contents).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # The content and declared checksum agree; structural validation must still
    # prevent invalid arrays or identity metadata reaching a dataset consumer.
    validate_dataset(root)


def _publish_geometry(root, manifest, record, geometry):
    contents = io.BytesIO()
    np.savez_compressed(contents, **geometry)
    _publish_changed_artifact(root, manifest, record, "geometry.npz", contents.getvalue())


def test_loading_roundtrip_preserves_rgb_arrays_geometry_and_metadata(loading_dataset):
    root = loading_dataset
    record = validate_dataset(root)["manifest"]["samples"][0]

    sample = load_sample(root, record)

    for field, filename in (("flat", "flat.png"), ("photo", "photo.png")):
        with Image.open(root / record["paths"][filename]) as encoded:
            np.testing.assert_array_equal(getattr(sample, field), np.array(encoded))
        assert getattr(sample, field).dtype == np.uint8
    with np.load(root / record["paths"]["geometry.npz"], allow_pickle=False) as archive:
        for name in ("backward_map", "valid_mask", "photo_page_mask"):
            np.testing.assert_array_equal(getattr(sample, name), archive[name])
    assert sample.metadata == json.loads((root / record["paths"]["metadata.json"]).read_text())
    assert sample.flat.shape == (128, 144, 3)
    assert sample.photo.shape == (144, 160, 3)


@pytest.mark.parametrize(
    "mutation",
    [
        "map-float64",
        "map-integer",
        "map-wrong-channels",
        "map-wrong-resolution",
        "valid-integer",
        "valid-wrong-resolution",
        "page-integer",
        "page-wrong-resolution",
        "empty-valid",
        "active-nan",
        "active-infinity",
        "active-out-of-bounds",
        "inactive-finite",
        "inactive-one-nan",
        "inactive-infinity",
        "missing-array",
        "object-array",
    ],
)
def test_corrupt_geometry_is_rejected_even_with_updated_artifact_hash(editable_sample, mutation):
    root, manifest, record, geometry = editable_sample
    if mutation == "map-float64":
        geometry["backward_map"] = geometry["backward_map"].astype(np.float64)
    elif mutation == "map-integer":
        geometry["backward_map"] = geometry["backward_map"].astype(np.int32)
    elif mutation == "map-wrong-channels":
        geometry["backward_map"] = geometry["backward_map"][..., :1]
    elif mutation == "map-wrong-resolution":
        geometry["backward_map"] = geometry["backward_map"][:-1]
    elif mutation == "valid-integer":
        geometry["valid_mask"] = geometry["valid_mask"].astype(np.uint8)
    elif mutation == "valid-wrong-resolution":
        geometry["valid_mask"] = geometry["valid_mask"][:-1]
    elif mutation == "page-integer":
        geometry["photo_page_mask"] = geometry["photo_page_mask"].astype(np.uint8)
    elif mutation == "page-wrong-resolution":
        geometry["photo_page_mask"] = geometry["photo_page_mask"][:-1]
    elif mutation == "empty-valid":
        geometry["valid_mask"][:] = False
        geometry["backward_map"][:] = np.nan
    elif mutation == "active-nan":
        geometry["backward_map"][0, 0] = np.nan
    elif mutation == "active-infinity":
        geometry["backward_map"][0, 0] = np.inf
    elif mutation == "active-out-of-bounds":
        geometry["backward_map"][0, 0] = (1.01, 0)
    elif mutation.startswith("inactive-"):
        geometry["valid_mask"][0, 0] = False
        geometry["backward_map"][0, 0] = {
            "inactive-finite": (0, 0),
            "inactive-one-nan": (np.nan, 0),
            "inactive-infinity": (np.inf, -np.inf),
        }[mutation]
    elif mutation == "missing-array":
        del geometry["backward_map"]
    elif mutation == "object-array":
        geometry["backward_map"] = geometry["backward_map"].astype(object)
    _publish_geometry(root, manifest, record, geometry)

    with pytest.raises(FlattenerError) as caught:
        load_sample(root, record)

    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.reason_code == "invalid_dataset"


def test_inactive_coordinates_accept_only_paired_nan_and_are_preserved(editable_sample):
    root, manifest, record, geometry = editable_sample
    geometry["valid_mask"][3:5, 7:9] = False
    geometry["backward_map"][3:5, 7:9] = np.nan
    _publish_geometry(root, manifest, record, geometry)

    sample = load_sample(root, record)

    np.testing.assert_array_equal(sample.valid_mask, geometry["valid_mask"])
    assert np.isnan(sample.backward_map[~sample.valid_mask]).all()
    assert np.isfinite(sample.backward_map[sample.valid_mask]).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_group", "other-document"),
        ("capture_session", "other-session"),
        ("split", "test"),
        ("family", "unknown"),
        ("sample_id", "other-sample"),
        ("document_id", "other-document"),
        ("output_size", [10, 10]),
        ("photo_size", [10, 10]),
    ],
)
def test_metadata_mismatches_are_rejected(editable_sample, field, value):
    root, manifest, record, _ = editable_sample
    metadata = json.loads((root / record["paths"]["metadata.json"]).read_text())
    metadata[field] = value
    _publish_changed_artifact(
        root, manifest, record, "metadata.json", json.dumps(metadata).encode("utf-8")
    )

    with pytest.raises(FlattenerError) as caught:
        load_sample(root, record)

    assert caught.value.reason_code == "invalid_dataset"


@pytest.mark.parametrize("contents", [b"not-json", b"[]", b"null", b"{}"])
def test_missing_or_malformed_metadata_is_rejected(editable_sample, contents):
    root, manifest, record, _ = editable_sample
    _publish_changed_artifact(root, manifest, record, "metadata.json", contents)

    with pytest.raises(FlattenerError) as caught:
        load_sample(root, record)

    assert caught.value.reason_code == "invalid_dataset"


@pytest.mark.parametrize("contents", [b"not a numpy archive", b"PK\x03\x04truncated"])
def test_corrupt_numpy_archive_returns_structured_error(editable_sample, contents):
    root, manifest, record, _ = editable_sample
    _publish_changed_artifact(root, manifest, record, "geometry.npz", contents)

    with pytest.raises(FlattenerError) as caught:
        load_sample(root, record)

    assert caught.value.reason_code == "invalid_dataset"


def test_loading_rejects_path_escape_without_following_it(editable_sample):
    root, _, record, _ = editable_sample
    record["paths"]["flat.png"] = "../outside.png"

    with pytest.raises(FlattenerError) as caught:
        load_sample(root, record)

    assert caught.value.reason_code == "invalid_dataset"


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_loading_requires_rgb_images(editable_sample, mode):
    root, manifest, record, _ = editable_sample
    encoded = io.BytesIO()
    with Image.new(mode, (144, 128)) as image:
        image.save(encoded, format="PNG")
    _publish_changed_artifact(root, manifest, record, "flat.png", encoded.getvalue())

    with pytest.raises(FlattenerError) as caught:
        load_sample(root, record)

    assert caught.value.status == Status.INPUT_ERROR
