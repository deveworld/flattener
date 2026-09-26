"""Coordinate data contracts; no optimizer, backward pass, or model fitting."""

import hashlib
import json
import os
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError
from flattener.learning import scene_data
from flattener.learning.model import ModelConfig, preprocess_image

PHOTO_SIZE = (128, 160)
CONFIG = ModelConfig(input_size=(24, 20), grid_size=(7, 5))


@pytest.fixture
def mutable(scene_source, tmp_path):
    directory = tmp_path / "own-coordinate-data-copy"
    shutil.copytree(scene_source, directory)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _manifest(directory):
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def _rewrite_manifest(directory, value):
    (directory / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


def _mutate_unpaired_metadata(directory, change):
    manifest = _manifest(directory)
    record = next(
        item
        for item in manifest["samples"]
        if item["split"] == "train" and item["has_page"] and item["pair_id"] is None
    )
    path = directory / record["paths"]["metadata.json"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    change(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    record["sha256"]["metadata.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _rewrite_manifest(directory, manifest)


def _independent_forward(size, parameters):
    """Explicit float64 half-center/shear/projective formula, not make_geometry."""
    height, width = size
    x, y = np.meshgrid(
        2 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1,
        2 * (np.arange(height, dtype=np.float64) + 0.5) / height - 1,
    )
    sheared_x = x + parameters["shear_x"] * np.sin(np.pi * (y + 1) / 2)
    if parameters["family"] == "local":
        profile = np.exp(
            -0.5 * ((sheared_x - parameters["local_center"]) / parameters["local_width"]) ** 2
        )
    elif parameters["family"] == "perspective":
        profile = np.zeros_like(sheared_x)
    else:
        profile = np.sin(np.pi * (sheared_x + 1) / 2)
    sheared_y = y + parameters["bend_y"] * profile
    c, s = np.cos(parameters["rotation"]), np.sin(parameters["rotation"])
    divisor = parameters["perspective_x"] * sheared_x + parameters["perspective_y"] * sheared_y + 1
    result = np.stack(
        (
            (
                parameters["scale_x"] * c * sheared_x
                - parameters["scale_y"] * s * sheared_y
                + parameters["translate_x"]
            )
            / divisor,
            (
                parameters["scale_x"] * s * sheared_x
                + parameters["scale_y"] * c * sheared_y
                + parameters["translate_y"]
            )
            / divisor,
        ),
        axis=-1,
    )
    visible = np.isfinite(result).all(axis=-1) & (np.abs(result) <= 1).all(axis=-1)
    result = result.astype(np.float32)
    result[~visible] = np.nan
    return result, visible


def test_constructor_checks_whole_manifest_without_pixel_or_map_decode(scene_source, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructor may not decode pixels or create coordinate maps")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(scene_data, "make_geometry", forbidden)
    for split in ("train", "validation"):
        dataset = scene_data.SceneCoordinateDataset(scene_source, split, CONFIG)
        assert len(dataset) == 4
        assert dataset.sample_counts == dict.fromkeys(("train", "validation", "test"), 6)
        assert dataset.page_counts == dict.fromkeys(("train", "validation", "test"), 4)
        assert dataset.photo_size == PHOTO_SIZE
        assert dataset.output_size == (192, 144)
        assert all(record["split"] == split and record["has_page"] for record in dataset.records)


def test_flat_collection_preserves_scene_targets_and_excludes_external_sources(mutable):
    before = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)[0]
    manifest = _manifest(mutable)
    (mutable / "files").mkdir()
    for record in manifest["samples"]:
        for name, relative in record["paths"].items():
            flat = f"files/scene-{record['id']}-{name}"
            (mutable / relative).rename(mutable / flat)
            record["paths"][name] = flat
    manifest["collection"] = {
        "schema_version": "dataset-collection-1.0",
        "sources": {
            "syntheticdoc": {"training_ready": False, "samples": [{"id": "external"}]},
            "docunet": {"training_ready": False, "samples": [{"id": "1_1"}]},
        },
    }
    _rewrite_manifest(mutable, manifest)
    dataset = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)
    assert len(dataset) == 4
    after = dataset[0]
    assert after["id"] == before["id"]
    np.testing.assert_array_equal(after["original_image"], before["original_image"])
    torch.testing.assert_close(after["target"], before["target"], equal_nan=True)
    torch.testing.assert_close(after["valid"], before["valid"])


def test_getitem_opens_only_selected_positive_once_per_artifact(scene_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "validation", CONFIG)
    record = dataset.records[1]
    expected = [scene_source / record["paths"][name] for name in ("image.png", "mask.png")]
    original, opened = Image.open, []

    def selected_only(path, *args, **kwargs):
        assert path in expected
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", selected_only)
    value = dataset[1]
    assert opened == expected
    assert value["id"] == record["id"]
    assert value["document_group"] == record["document_group"]
    assert value["size_bin"] == record["size_bin"]
    assert value["geometry_family"] == value["family"] == record["family"]


def test_original_pixels_and_preprocessing_are_independent_of_output_size(scene_source):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG, (137, 149))
    value = dataset[0]
    record = dataset.records[0]
    with Image.open(scene_source / record["paths"]["image.png"]) as image:
        original = np.array(image)
    np.testing.assert_array_equal(value["original_image"], original)
    assert value["original_image"].dtype == np.uint8
    assert value["original_image"].shape == (*PHOTO_SIZE, 3)
    assert value["original_image"].flags.c_contiguous
    assert torch.equal(value["image"], preprocess_image(original, CONFIG))
    assert value["image"].shape == (3, *CONFIG.input_size)
    assert value["image"].dtype == torch.float32
    assert value["target"].shape == (137, 149, 2)
    assert value["source_mask"].shape == PHOTO_SIZE
    assert value["valid"].dtype == value["source_mask"].dtype == torch.bool
    assert not value["image"].requires_grad


def test_varying_output_sizes_use_analytic_original_frame_half_centers(scene_source):
    for size in ((128, 160), (137, 149), (257, 131)):
        dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG, size)
        for index, record in enumerate(dataset.records):
            metadata = json.loads(
                (scene_source / record["paths"]["metadata.json"]).read_text(encoding="utf-8"))
            expected, valid = _independent_forward(size, metadata["geometry"]["parameters"])
            value = dataset[index]
            assert value["target"].dtype == torch.float32
            np.testing.assert_array_equal(value["valid"].numpy(), valid)
            np.testing.assert_allclose(value["target"].numpy(), expected, rtol=0, atol=6e-8)
            # Errors are scaled to the original source axes, not the preview/output axes.
            error = (value["target"].numpy()[valid] - expected[valid]) * np.array(
                [PHOTO_SIZE[1] / 2, PHOTO_SIZE[0] / 2]
            )
            assert np.linalg.norm(error, axis=-1).max() < 1e-5


def test_samples_and_public_snapshots_do_not_share_trusted_state(scene_source):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    expected = dataset[0]
    changed = dataset[0]
    changed["image"].zero_()
    changed["target"].zero_()
    changed["original_image"][:] = 0
    changed["source_mask"].zero_()
    records = dataset.records
    records[0]["sha256"]["image.png"] = "0" * 64
    manifest = dataset.manifest
    manifest["samples"].clear()
    sources = dataset.source_code_sha256
    sources.clear()
    actual = dataset[0]
    for key in ("image", "target", "valid", "source_mask"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(actual["original_image"], expected["original_image"])
    assert len(dataset.records) == 4 and len(dataset.manifest["samples"]) == 18
    assert len(dataset.source_code_sha256) == 13


def test_invalid_configuration_is_rejected_before_io():
    cases = [
        {"split": "all"},
        {"split": True},
        {"model_config": {}},
        {"model_config": ModelConfig(input_size=(513, 8))},
        {"model_config": ModelConfig(grid_size=(513, 8))},
        {"output_size": [128, 128]},
        {"output_size": (True, 128)},
        {"output_size": (1, 128)},
        {"output_size": (513, 128)},
        {"output_size": (6, 128)},
    ]
    for change in cases:
        arguments = {"split": "train", "model_config": CONFIG, **change}
        with pytest.raises(FlattenerError) as error:
            scene_data.SceneCoordinateDataset("/not-a-dataset", **arguments)
        assert error.value.reason_code == "invalid_coordinate_scene_dataset"


def test_index_semantics_are_integer_only_and_preserve_index_error(scene_source):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    for index in (True, np.bool_(False), 0.0, "0", slice(None)):
        with pytest.raises(FlattenerError, match="index must be an integer"):
            dataset[index]
    for index in (4, -5):
        with pytest.raises(IndexError):
            dataset[index]
    assert dataset[-1]["id"] == dataset.records[-1]["id"]
    assert dataset[np.int64(0)]["id"] == dataset.records[0]["id"]


def test_encoded_manifest_changes_after_initialization_are_rejected(mutable):
    dataset = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)
    path = mutable / "manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(FlattenerError, match="manifest changed"):
        dataset[0]


def test_selected_artifact_changes_after_initialization_are_rejected(mutable):
    dataset = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)
    path = mutable / dataset.records[0]["paths"]["image.png"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(FlattenerError, match="artifact changed"):
        dataset[0]


def test_constructor_hashes_even_held_out_artifacts_without_decoding(mutable, monkeypatch):
    record = next(item for item in _manifest(mutable)["samples"] if item["split"] == "test")
    path = mutable / record["paths"]["image.png"]
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("held-out decode"))
    with pytest.raises(FlattenerError, match="artifact changed"):
        scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)


def test_wrong_coordinate_convention_is_rejected_even_with_updated_hashes(mutable):
    def alter(metadata):
        metadata["geometry"]["align_corners"] = True
        metadata["physical_page"]["geometry"]["align_corners"] = True

    _mutate_unpaired_metadata(mutable, alter)
    with pytest.raises(FlattenerError, match="original full-scene coordinate frame"):
        scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)


def test_arbitrary_geometry_cannot_be_substituted_for_seeded_truth(mutable):
    def alter(metadata):
        for geometry in (metadata["geometry"], metadata["physical_page"]["geometry"]):
            geometry["parameters"]["rotation"] += 0.01

    _mutate_unpaired_metadata(mutable, alter)
    with pytest.raises(FlattenerError, match="recorded generator seed"):
        scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)


def test_seeded_parameter_roundoff_is_limited_to_derived_placement():
    from dataclasses import replace

    from flattener.synthetic.geometry import WarpParameters

    expected = WarpParameters()
    assert scene_data._same_seeded_parameters(expected, expected)
    for name in ("scale_x", "scale_y", "translate_x", "translate_y"):
        reference = getattr(expected, name)
        nearby = replace(expected, **{name: reference + 2 * np.finfo(np.float64).eps})
        assert scene_data._same_seeded_parameters(nearby, expected)
        changed = replace(expected, **{name: reference + 1e-12})
        assert not scene_data._same_seeded_parameters(changed, expected)
    rotated = replace(expected, rotation=np.finfo(np.float64).eps)
    assert not scene_data._same_seeded_parameters(rotated, expected)


def test_generator_source_claims_must_match_the_current_implementation(mutable):
    manifest = _manifest(mutable)
    manifest["generator_code_sha256"]["synthetic/geometry.py"] = "0" * 64
    _rewrite_manifest(mutable, manifest)
    with pytest.raises(FlattenerError, match="generator sources differ"):
        scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)


def test_source_changes_after_construction_are_rejected(scene_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    changed = dataset.source_code_sha256
    changed["learning/model.py"] = "0" * 64
    monkeypatch.setattr(scene_data, "_source_hashes", lambda: changed)
    with pytest.raises(FlattenerError, match="sources changed after initialization"):
        dataset[0]


def test_resource_limits_reject_before_image_decode(scene_source, monkeypatch):
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("oversized decode"))
    monkeypatch.setattr(scene_data, "_ARTIFACT_LIMIT", 1)
    with pytest.raises(FlattenerError, match="encoded resource limit"):
        scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)


def test_missing_or_malformed_manifest_has_a_structured_error(tmp_path):
    with pytest.raises(FlattenerError, match="Cannot initialize coordinate scenes"):
        scene_data.SceneCoordinateDataset(tmp_path, "train", CONFIG)
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(FlattenerError, match="Cannot initialize coordinate scenes"):
        scene_data.SceneCoordinateDataset(tmp_path, "train", CONFIG)


def test_cache_is_exact_tensor_only_snapshot_and_loads_each_example_once(scene_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    expected = [dataset[index] for index in range(len(dataset))]
    original = scene_data.SceneCoordinateDataset.__getitem__
    accessed = []

    def counted(self, index):
        accessed.append(index)
        return original(self, index)

    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", counted)
    required = scene_data.scene_cache_bytes(dataset)
    cache = scene_data.cache_scene_dataset(dataset, max_bytes=required)
    assert accessed == list(range(len(dataset)))
    assert cache.cache_bytes == required == 4 * (3 * 24 * 20 * 4 + 192 * 144 * 9 + 128 * 160)
    assert cache.records == dataset.records
    assert cache.manifest_sha256 == dataset.manifest_sha256
    assert cache.source_code_sha256 == dataset.source_code_sha256
    assert cache.page_counts == dataset.page_counts
    assert cache.photo_size == dataset.photo_size
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("cached decode"))
    monkeypatch.setattr(scene_data, "_manifest_digest", lambda *args: pytest.fail("per-item hash"))
    for index, sample in enumerate(expected):
        actual = cache[index]
        assert actual.keys() == {"image", "target", "valid", "source_mask"}
        assert actual["source_mask"].dtype == torch.bool
        assert actual["source_mask"].shape == PHOTO_SIZE
        for key in actual:
            torch.testing.assert_close(actual[key], sample[key], rtol=0, atol=0, equal_nan=True)
            assert not actual[key].requires_grad
    assert accessed == list(range(len(dataset)))


def test_cache_returns_copies_and_preserves_index_semantics(scene_source):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "validation", CONFIG)
    cache = scene_data.cache_scene_dataset(dataset)
    expected = cache[0]
    changed = cache[0]
    for value in changed.values():
        value.zero_()
    for key, value in expected.items():
        torch.testing.assert_close(cache[0][key], value, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(cache[-1]["image"], cache[len(cache) - 1]["image"])
    for index in (True, np.bool_(False), 0.0, "0", slice(None)):
        with pytest.raises(FlattenerError):
            cache[index]
    with pytest.raises(IndexError):
        cache[len(cache)]


def test_cache_budget_rejects_before_loading_any_example(scene_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    required = scene_data.scene_cache_bytes(dataset)
    monkeypatch.setattr(
        scene_data.SceneCoordinateDataset,
        "__getitem__",
        lambda *args: pytest.fail("budget must be checked before loading"),
    )
    old_three_tensor_budget = required - len(dataset) * PHOTO_SIZE[0] * PHOTO_SIZE[1]
    for budget in (0, -1, True, 1.5, "1000", required - 1, old_three_tensor_budget):
        with pytest.raises(FlattenerError):
            scene_data.cache_scene_dataset(dataset, max_bytes=budget)


@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_cache_rejects_non_original_boolean_source_mask(scene_source, monkeypatch, fault):
    dataset = scene_data.SceneCoordinateDataset(
        scene_source, "train", CONFIG, output_size=(137, 149)
    )
    original = scene_data.SceneCoordinateDataset.__getitem__

    def invalid_mask(self, index):
        sample = original(self, index)
        if fault == "dtype":
            sample["source_mask"] = sample["source_mask"].float()
        else:
            sample["source_mask"] = torch.zeros(self.output_size, dtype=torch.bool)
        return sample

    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", invalid_mask)
    with pytest.raises(FlattenerError, match="invalid source_mask tensor"):
        scene_data.cache_scene_dataset(dataset)


def test_cache_rejects_test_split_and_non_dataset_without_decode(scene_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "test", CONFIG)
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("test cache decode"))
    for value in (dataset, None, object()):
        with pytest.raises(FlattenerError, match="uncached train or validation"):
            scene_data.cache_scene_dataset(value)


def test_cached_integrity_rejects_later_artifact_changes_without_decode(mutable, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)
    cache = scene_data.cache_scene_dataset(dataset)
    path = mutable / dataset.records[0]["paths"]["image.png"]
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("integrity decode"))
    with pytest.raises(FlattenerError, match="artifact changed"):
        cache.verify_integrity()


def test_paper_opt_in_is_strict_and_default_remains_native(paper_source):
    native = scene_data.SceneCoordinateDataset(paper_source, "train", CONFIG)
    mixed = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, include_paper_augmentation=True)
    assert len(native) == 4 and len(mixed) == 6
    assert mixed.records[:4] == native.records
    assert len(native.source_code_sha256) == 13 and len(mixed.source_code_sha256) == 14
    assert mixed.page_counts == dict.fromkeys(("train", "validation", "test"), 6)
    assert mixed.sample_counts == dict.fromkeys(("train", "validation", "test"), 8)
    assert mixed.source_page_counts["paper"] == dict.fromkeys(("train", "validation", "test"), 2)
    assert native.paper_augmentation_sha256 is None
    assert len(mixed.paper_augmentation_sha256) == 64
    for flag in (0, 1, None, "true"):
        with pytest.raises(FlattenerError, match="must be boolean"):
            scene_data.SceneCoordinateDataset("/missing", "train", include_paper_augmentation=flag)


def test_paper_constructor_reads_no_pixels_or_geometry_maps(paper_source, monkeypatch):
    from flattener.synthetic import paper

    def forbidden(*args, **kwargs):
        pytest.fail("constructor may not decode any split or allocate geometry maps")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(paper, "make_geometry", forbidden)
    monkeypatch.setattr(paper, "make_capture_geometry", forbidden)
    for split in ("train", "validation", "test"):
        dataset = scene_data.SceneCoordinateDataset(
            paper_source, split, CONFIG, include_paper_augmentation=True)
        assert {r["source_kind"] for r in dataset.records[4:]} == {"capture", "simulated"}


@pytest.mark.parametrize("output_size", [(137, 149), (257, 131)])
def test_paper_real_helpers_and_original_frame_maps(paper_source, output_size):
    from flattener.synthetic import paper

    dataset = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, output_size, include_paper_augmentation=True)
    for index in (4, 5):
        record, actual = dataset.records[index], dataset[index]
        metadata = json.loads(
            (paper_source / record["paths"]["metadata.json"]).read_text(encoding="utf-8"))
        if record["kind"] == "capture":
            expected = paper.make_capture_geometry(metadata["corners_xy"], PHOTO_SIZE,
                                                   output_size, PHOTO_SIZE)
        else:
            expected = paper.make_geometry(output_size, PHOTO_SIZE,
                                          paper.PaperParameters(**metadata["geometry"]["parameters"]))
        np.testing.assert_array_equal(actual["target"].numpy(), expected.backward_map)
        np.testing.assert_array_equal(actual["valid"].numpy(), expected.valid_mask)
        np.testing.assert_array_equal(actual["source_mask"].numpy(), expected.photo_page_mask)
        assert actual["source"] == "paper" and actual["source_kind"] == record["kind"]
        assert actual["original_image"].shape == (*PHOTO_SIZE, 3)


@pytest.mark.parametrize("field", ["document_group", "content_sha256"])
def test_paper_source_split_leakage_is_rejected(mutable_paper, field):
    manifest = _manifest(mutable_paper)
    templates = manifest["collection"]["paper_augmentation"]["templates"]
    templates[1][field] = templates[0][field]
    _rewrite_manifest(mutable_paper, manifest)
    with pytest.raises(FlattenerError):
        scene_data.SceneCoordinateDataset(
            mutable_paper, "train", CONFIG, include_paper_augmentation=True)


@pytest.mark.parametrize("change", ["source", "generator"])
def test_paper_provenance_rejections(mutable_paper, change):
    manifest = _manifest(mutable_paper)
    block = manifest["collection"]["paper_augmentation"]
    if change == "source":
        block["templates"][0]["source_dataset"] = "docunet"
    else:
        block["generator_code_sha256"]["synthetic/paper.py"] = "0" * 64
    _rewrite_manifest(mutable_paper, manifest)
    with pytest.raises(FlattenerError):
        scene_data.SceneCoordinateDataset(
            mutable_paper, "train", CONFIG, include_paper_augmentation=True)


def test_paper_cached_integrity_checks_templates_without_decoding(mutable_paper, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    cache = scene_data.cache_scene_dataset(dataset, max_bytes=scene_data.scene_cache_bytes(dataset))
    for index in (4, 5):
        for key, tensor in cache[index].items():
            torch.testing.assert_close(tensor, dataset[index][key], equal_nan=True)
    template = _manifest(mutable_paper)["collection"]["paper_augmentation"]["templates"][-1]
    path = mutable_paper / template["path"]
    path.write_bytes(path.read_bytes() + b"mutation")
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("integrity must not decode"))
    with pytest.raises(FlattenerError, match="template exceeds|template artifact changed"):
        cache.verify_integrity()


def test_paper_requested_split_only_and_test_cache_rejected(paper_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(
        paper_source, "validation", CONFIG, include_paper_augmentation=True)
    original, opened = Image.open, []

    def checked(path, *args, **kwargs):
        assert "validation" in str(path)
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", checked)
    dataset[4]
    dataset[5]
    assert opened
    test_data = scene_data.SceneCoordinateDataset(
        paper_source, "test", CONFIG, include_paper_augmentation=True)
    with pytest.raises(FlattenerError, match="uncached train or validation"):
        scene_data.cache_scene_dataset(test_data)


def test_paper_multiple_templates_in_one_original_document_group_are_allowed(mutable_paper):
    manifest = _manifest(mutable_paper)
    block = manifest["collection"]["paper_augmentation"]
    template = deepcopy(block["templates"][0])
    template.update(id="second-view-of-train-document", path="files/second-flat-view.png")
    shutil.copyfile(mutable_paper / block["templates"][0]["path"], mutable_paper / template["path"])
    block["templates"].append(template)
    record = deepcopy(block["records"][0])
    record.update(id="second-template-record", template_id=template["id"])
    metadata = json.loads(
        (mutable_paper / record["paths"]["metadata.json"]).read_text(encoding="utf-8"))
    metadata.update(sample_id=record["id"], template_id=template["id"])
    for name, source in list(record["paths"].items()):
        target = f"files/second-template-{name}"
        shutil.copyfile(mutable_paper / source, mutable_paper / target)
        record["paths"][name] = target
    (mutable_paper / record["paths"]["metadata.json"]).write_text(
        json.dumps(metadata), encoding="utf-8")
    record["sha256"]["metadata.json"] = scene_data._sha256(
        mutable_paper / record["paths"]["metadata.json"])
    block["records"].append(record)
    _rewrite_manifest(mutable_paper, manifest)
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    assert len(dataset) == 7
    assert dataset[-1]["document_group"] == dataset[4]["document_group"]


@pytest.mark.parametrize("kind", ["simulated", "capture"])
def test_paper_metadata_gt_substitution_rejected_without_maps(mutable_paper, monkeypatch, kind):
    from flattener.synthetic import paper

    manifest = _manifest(mutable_paper)
    block = manifest["collection"]["paper_augmentation"]
    if kind == "capture":
        block["captures"][-1]["corners_xy"] = [[0, 0], [150, 115], [150, 0], [0, 115]]
    else:
        record = next(r for r in block["records"] if r["kind"] == kind)
        path = mutable_paper / record["paths"]["metadata.json"]
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["geometry"]["parameters"]["rotation"] += .001
        path.write_text(json.dumps(metadata), encoding="utf-8")
        record["sha256"]["metadata.json"] = scene_data._sha256(path)
    _rewrite_manifest(mutable_paper, manifest)
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("constructor decode"))
    monkeypatch.setattr(paper, "make_geometry", lambda *args: pytest.fail("constructor GT maps"))
    monkeypatch.setattr(paper, "make_capture_geometry", lambda *args: pytest.fail("capture maps"))
    with pytest.raises(FlattenerError, match="generator seed|clockwise convex"):
        scene_data.SceneCoordinateDataset(
            mutable_paper, "train", CONFIG, include_paper_augmentation=True)


def test_paper_held_out_artifact_change_detected_without_decode(mutable_paper, monkeypatch):
    manifest = _manifest(mutable_paper)
    record = manifest["collection"]["paper_augmentation"]["records"][-1]
    path = mutable_paper / record["paths"]["image.png"]
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("held-out decode"))
    # The opt-out path remains independent of unused paper assets.
    scene_data.SceneCoordinateDataset(mutable_paper, "train", CONFIG)
    with pytest.raises(FlattenerError, match="artifact changed"):
        scene_data.SceneCoordinateDataset(
            mutable_paper, "train", CONFIG, include_paper_augmentation=True)


def test_paper_decoded_template_content_is_checked_on_access(mutable_paper):
    manifest = _manifest(mutable_paper)
    block = manifest["collection"]["paper_augmentation"]
    template = block["templates"][0]
    template["content_sha256"] = "0" * 64
    for record in block["records"]:
        if record["template_id"] == template["id"]:
            path = mutable_paper / record["paths"]["metadata.json"]
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["template_content_sha256"] = template["content_sha256"]
            path.write_text(json.dumps(metadata), encoding="utf-8")
            record["sha256"]["metadata.json"] = scene_data._sha256(path)
    _rewrite_manifest(mutable_paper, manifest)
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    with pytest.raises(FlattenerError, match="content hash differs"):
        dataset[4]


def test_paper_cache_checks_capture_file_and_enabled_code(mutable_paper, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    cached = scene_data.cache_scene_dataset(dataset)
    expected = dataset.source_code_sha256
    changed = {**expected, "synthetic/paper.py": "0" * 64}
    original = scene_data._data_hashes
    monkeypatch.setattr(scene_data, "_data_hashes", lambda *args: changed)
    with pytest.raises(FlattenerError, match="implementation changed"):
        cached.verify_integrity()
    monkeypatch.setattr(scene_data, "_data_hashes", original)
    capture = _manifest(mutable_paper)["collection"]["paper_augmentation"]["captures"][-1]
    path = mutable_paper / capture["path"]
    path.write_bytes(path.read_bytes() + b"change")
    with pytest.raises(FlattenerError, match="resource limit|artifact changed"):
        cached.verify_integrity()


def test_cache_postflight_catches_earlier_sample_changed_during_load(mutable, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)
    original = scene_data.SceneCoordinateDataset.__getitem__

    def changed_earlier_sample(self, index):
        result = original(self, index)
        if index == len(self) - 1:
            path = mutable / self.records[0]["paths"]["image.png"]
            path.write_bytes(path.read_bytes() + b"changed after its load")
        return result

    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", changed_earlier_sample)
    with pytest.raises(FlattenerError, match="artifact changed"):
        scene_data.cache_scene_dataset(dataset)


def test_cached_integrity_rejects_implementation_change(scene_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "validation", CONFIG)
    cache = scene_data.cache_scene_dataset(dataset)
    changed = cache.source_code_sha256
    changed["learning/model.py"] = "0" * 64
    monkeypatch.setattr(scene_data, "_source_hashes", lambda: changed)
    with pytest.raises(FlattenerError, match="cached coordinate implementation changed"):
        cache.verify_integrity()


@pytest.mark.parametrize("include_paper", [False, True])
def test_cache_build_hashes_manifest_and_code_only_at_boundaries(paper_source, monkeypatch,
                                                               include_paper):
    dataset = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, include_paper_augmentation=include_paper)
    counts = Counter()
    for name in ("_manifest_digest", "_data_hashes"):
        original = getattr(scene_data, name)

        def counted(*args, _name=name, _original=original):
            counts[_name] += 1
            return _original(*args)

        monkeypatch.setattr(scene_data, name, counted)
    original_getitem = scene_data.SceneCoordinateDataset.__getitem__
    loaded = []

    def checked_loader(self, index):
        before = counts.copy()
        result = original_getitem(self, index)
        # Per-record artifact checks remain; whole-manifest/source scans do not.
        assert counts == before
        loaded.append(index)
        return result

    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", checked_loader)
    scene_data.cache_scene_dataset(dataset)
    assert loaded == list(range(len(dataset)))
    assert counts == {"_manifest_digest": 4, "_data_hashes": 4}
    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", original_getitem)
    counts.clear()
    dataset[0]
    assert counts == {"_manifest_digest": 2, "_data_hashes": 2}


@pytest.mark.parametrize("fail_after_paper_load", [False, True])
def test_cache_private_loader_preserves_original_state_even_on_failure(
        paper_source, monkeypatch, fail_after_paper_load):
    dataset = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, include_paper_augmentation=True)
    before = deepcopy(dataset.__dict__)
    original = scene_data.SceneCoordinateDataset.__getitem__
    loaders = []

    def checked_loader(self, index):
        assert self is not dataset
        assert self._verified_template_contents is not dataset._verified_template_contents
        assert hasattr(self, "_building_source_paths")
        loaders.append(self)
        result = original(self, index)
        if fail_after_paper_load and self.records[index].get("source") == "paper":
            assert self._verified_template_contents
            raise RuntimeError("injected cache decode interruption")
        return result

    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", checked_loader)
    if fail_after_paper_load:
        with pytest.raises(RuntimeError, match="injected cache decode interruption"):
            scene_data.cache_scene_dataset(dataset)
    else:
        cache = scene_data.cache_scene_dataset(dataset)
        assert not hasattr(cache, "_building_source_paths")
    assert loaders and all(loader is loaders[0] for loader in loaders)
    assert loaders[0]._verified_template_contents
    assert dataset.__dict__ == before
    assert not hasattr(dataset, "_building_source_paths")


def test_paper_cache_full_verification_hashes_each_shared_source_once(mutable_paper, monkeypatch):
    # Add another real record using the same capture/template, not an aliased path.
    manifest = _manifest(mutable_paper)
    block = manifest["collection"]["paper_augmentation"]
    record = deepcopy(next(r for r in block["records"]
                           if r["split"] == "train" and r["kind"] == "capture"))
    record["id"] = "repeated-capture-view"
    metadata = json.loads(
        (mutable_paper / record["paths"]["metadata.json"]).read_text(encoding="utf-8"))
    metadata["sample_id"] = record["id"]
    for name, relative in list(record["paths"].items()):
        target = f"files/repeated-capture-{name}"
        shutil.copyfile(mutable_paper / relative, mutable_paper / target)
        record["paths"][name] = target
    (mutable_paper / record["paths"]["metadata.json"]).write_text(
        json.dumps(metadata), encoding="utf-8")
    record["sha256"]["metadata.json"] = scene_data._sha256(
        mutable_paper / record["paths"]["metadata.json"])
    block["records"].append(record)
    _rewrite_manifest(mutable_paper, manifest)
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    cache = scene_data.cache_scene_dataset(dataset)
    source_paths = {mutable_paper / entry["path"]
                    for entry in (*block["templates"], *block["captures"])}
    original, calls = scene_data._sha256, Counter()

    def counted(path):
        if path in source_paths:
            calls[path] += 1
        return original(path)

    monkeypatch.setattr(scene_data, "_sha256", counted)
    monkeypatch.setattr(
        Image, "open", lambda *args: pytest.fail("full verification decoded pixels"))
    cache.verify_integrity(full=True)
    assert calls == dict.fromkeys(source_paths, 1)
    assert any("test" in path.name for path in calls)


def test_paper_cache_tensors_are_bitwise_equal_to_uncached_samples(paper_source):
    dataset = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, (137, 149), include_paper_augmentation=True)
    cache = scene_data.cache_scene_dataset(dataset)
    for index in range(len(dataset)):
        expected, actual = dataset[index], cache[index]
        assert actual.keys() == {"image", "target", "valid", "source_mask"}
        for key, tensor in actual.items():
            assert tensor.dtype == expected[key].dtype
            assert tensor.shape == expected[key].shape
            assert tensor.numpy().tobytes() == expected[key].numpy().tobytes()


@pytest.mark.parametrize("change", [None, "size", "missing"])
def test_cache_light_verification_does_not_read_payloads_and_detects_file_changes(
        mutable_paper, monkeypatch, change):
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    cache = scene_data.cache_scene_dataset(dataset)
    # Held-out source files must remain in the integrity inventory as well.
    template = _manifest(mutable_paper)["collection"]["paper_augmentation"]["templates"][-1]
    path = mutable_paper / template["path"]
    if change == "size":
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "missing":
        path.unlink()
    payloads = set(cache._artifact_files)
    original_open, reads = Path.open, []

    def no_payload_reads(self, *args, **kwargs):
        assert self not in payloads, f"light verification read payload {self}"
        reads.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_payload_reads)
    if change is None:
        cache.verify_integrity(full=False)
        assert reads.count(mutable_paper / "manifest.json") == 2
    else:
        with pytest.raises(FlattenerError, match="artifact metadata changed|Cannot verify"):
            cache.verify_integrity(full=False)


def test_cache_full_verification_hashes_same_size_restored_mtime_tampering(
        mutable_paper, monkeypatch):
    from flattener.synthetic import scenes

    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    cache = scene_data.cache_scene_dataset(dataset)
    trusted_state = cache._artifact_state
    path = mutable_paper / dataset.records[0]["paths"]["image.png"]
    before = path.stat()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    original, hashed = scene_data._sha256, []

    def counted(target):
        hashed.append(target)
        return original(target)

    monkeypatch.setattr(scene_data, "_sha256", counted)
    monkeypatch.setattr(scenes, "_sha256", counted)
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("integrity decoded tampered file"))
    with pytest.raises(FlattenerError, match="artifact changed"):
        cache.verify_integrity(full=True)
    assert path in hashed
    assert cache._artifact_state == trusted_state


@pytest.mark.parametrize("changed_identity", ["manifest", "sources"])
def test_cache_postflight_retains_global_identity_checks_after_private_loading(
        mutable, monkeypatch, changed_identity):
    dataset = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)
    original = scene_data.SceneCoordinateDataset.__getitem__

    def changed_after_last_load(self, index):
        result = original(self, index)
        if index == len(self) - 1:
            if changed_identity == "manifest":
                path = mutable / "manifest.json"
                path.write_bytes(path.read_bytes() + b"\n")
            else:
                changed = dataset.source_code_sha256
                changed["learning/model.py"] = "0" * 64
                monkeypatch.setattr(scene_data, "_data_hashes", lambda *args: changed)
        return result

    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", changed_after_last_load)
    with pytest.raises(FlattenerError, match="cached coordinate (manifest|implementation) changed"):
        scene_data.cache_scene_dataset(dataset)
    assert not hasattr(dataset, "_building_source_paths")


def test_cache_postflight_detects_mutation_of_private_loader_reused_source(
        mutable_paper, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    before = deepcopy(dataset.__dict__)
    block = _manifest(mutable_paper)["collection"]["paper_augmentation"]
    source = next(item for item in block["templates"] if item["split"] == "train")
    path = mutable_paper / source["path"]
    original = scene_data.SceneCoordinateDataset.__getitem__

    def changed_after_reuse(self, index):
        result = original(self, index)
        if index == len(self) - 1:
            assert path in self._building_source_paths.values()
            content = bytearray(path.read_bytes())
            content[-1] ^= 1
            path.write_bytes(content)
        return result

    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__", changed_after_reuse)
    with pytest.raises(FlattenerError, match="template artifact changed"):
        scene_data.cache_scene_dataset(dataset)
    assert dataset.__dict__ == before


def test_cache_full_verification_detects_change_after_an_earlier_artifact_hash(
        mutable_paper, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    cache = scene_data.cache_scene_dataset(dataset)
    trusted_state = cache._artifact_state
    earlier = mutable_paper / dataset.records[0]["paths"]["image.png"]
    captures = _manifest(mutable_paper)["collection"]["paper_augmentation"]["captures"]
    last_source = mutable_paper / captures[-1]["path"]
    original = scene_data._sha256

    def mutate_previously_hashed_artifact(path):
        digest = original(path)
        if path == last_source:
            earlier.write_bytes(earlier.read_bytes() + b"changed after full-pass hash")
        return digest

    monkeypatch.setattr(scene_data, "_sha256", mutate_previously_hashed_artifact)
    with pytest.raises(FlattenerError, match="artifacts changed during verification"):
        cache.verify_integrity(full=True)
    assert cache._artifact_state == trusted_state


def _cache_pair(directory):
    return {split: scene_data.SceneCoordinateDataset(directory, split, CONFIG)
            for split in ("train", "validation")}


def test_contiguous_cache_batch_matches_ordered_clones_without_aliases(scene_source, monkeypatch):
    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    cache = scene_data.cache_scene_dataset(dataset)
    original = {key: value.clone() for key, value in cache._tensors.items()}
    assert all(tensor.is_contiguous() for tensor in cache._tensors.values())
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU batch initialized CUDA"))
    indices = [2, 0, 2, -1]
    batch = cache.fetch_batch(indices)
    for key, tensor in batch.items():
        expected = torch.stack([cache[index][key] for index in indices])
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0, equal_nan=True)
        assert tensor.is_contiguous() and tensor.data_ptr() != cache._tensors[key].data_ptr()
        tensor.zero_()
        torch.testing.assert_close(cache._tensors[key], original[key], equal_nan=True)
    assert set(cache.fetch_batch([0, 1], include_source_mask=False)) == {"image", "target", "valid"}


@pytest.mark.parametrize("indices", [[], [True]])
def test_cache_batch_rejects_invalid_indices(scene_source, indices):
    cache = scene_data.cache_scene_dataset(_cache_pair(scene_source)["train"])
    with pytest.raises(FlattenerError):
        cache.fetch_batch(indices)
    with pytest.raises(IndexError):
        cache.fetch_batch([len(cache)])
    with pytest.raises(FlattenerError):
        cache.fetch_batch([0], pin_memory=1)


def test_batch_pin_request_allocates_final_destination_once(scene_source, monkeypatch):
    cache = scene_data.cache_scene_dataset(_cache_pair(scene_source)["train"])
    original, allocations = torch.empty, []

    def spy(shape, **kwargs):
        allocations.append((shape, kwargs["pin_memory"]))
        # Exercise destination routing without requiring an accelerator driver.
        return original(shape, **{**kwargs, "pin_memory": False})

    monkeypatch.setattr(torch, "empty", spy)
    batch = cache.fetch_batch([1, 0], include_source_mask=False, pin_memory=True)
    assert len(allocations) == 3 and all(pinned for _, pinned in allocations)
    assert all(value.shape[0] == 2 for value in batch.values())


def test_persistent_pair_round_trip_weights_only_mmap_and_no_warm_decode(mutable, monkeypatch):
    datasets = _cache_pair(mutable)
    budget = sum(scene_data.scene_cache_bytes(value) for value in datasets.values())
    path = mutable / "preprocessed.pt"
    cold = scene_data.cache_scene_datasets(datasets, max_bytes=budget, persistent_path=path)
    assert path.is_file() and path.with_suffix(".pt.json").is_file()
    assert not path.with_suffix(".pt.lock").exists()
    assert cold["train"].cache_stats["status"] == "built"
    original, loads = torch.load, []

    def safe_load(*args, **kwargs):
        loads.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("warm image decode"))
    monkeypatch.setattr(scene_data.SceneCoordinateDataset, "__getitem__",
                        lambda *args: pytest.fail("warm dataset decode/geometry"))
    warm = scene_data.cache_scene_datasets(datasets, max_bytes=budget, persistent_path=path)
    assert loads == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    assert warm["train"].cache_stats["status"] == "loaded"
    assert warm["train"].cache_stats["seconds"] > 0
    print(json.dumps({"fixture_cache_seconds": {
        "cold": cold["train"].cache_stats["seconds"],
        "warm": warm["train"].cache_stats["seconds"]}, "samples": 8}))
    for split in warm:
        assert warm[split].cache_bytes == cold[split].cache_bytes
        for key in warm[split]._tensors:
            assert (warm[split]._tensors[key].numpy().tobytes()
                    == cold[split]._tensors[key].numpy().tobytes())
        warm[split].verify_integrity(full=False)
    warm["train"].fetch_batch([1, 0])["image"].zero_()
    torch.testing.assert_close(warm["train"][0]["image"], cold["train"][0]["image"])


def test_persistent_aggregate_budget_and_test_split_reject_before_decode(scene_source, monkeypatch):
    datasets = _cache_pair(scene_source)
    required = sum(scene_data.scene_cache_bytes(value) for value in datasets.values())
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("budget check decoded"))
    with pytest.raises(FlattenerError, match="requires.*budget"):
        scene_data.cache_scene_datasets(datasets, max_bytes=required - 1)
    with pytest.raises(FlattenerError, match="train and validation"):
        scene_data.cache_scene_datasets(
            {"train": datasets["train"], "test": datasets["validation"]})
    with pytest.raises(FlattenerError, match="persistent_path"):
        scene_data.cache_scene_datasets(datasets, rebuild=True)


def test_persistent_stale_identity_requires_explicit_rebuild(mutable, monkeypatch):
    path = mutable / "preprocessed.pt"
    old = scene_data.cache_scene_datasets(_cache_pair(mutable), persistent_path=path)
    payload = path.read_bytes()
    manifest = mutable / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    fresh = _cache_pair(mutable)
    with pytest.raises(FlattenerError, match="stale"):
        scene_data.cache_scene_datasets(fresh, persistent_path=path)
    assert path.read_bytes() == payload
    rebuilt = scene_data.cache_scene_datasets(fresh, persistent_path=path, rebuild=True)
    assert rebuilt["train"].cache_stats["status"] == "rebuilt"
    assert rebuilt["train"].manifest_sha256 != old["train"].manifest_sha256


@pytest.mark.parametrize("fault", ["payload", "missing_sidecar"])
def test_persistent_corruption_is_fail_closed(mutable, fault):
    datasets = _cache_pair(mutable)
    path = mutable / "preprocessed.pt"
    scene_data.cache_scene_datasets(datasets, persistent_path=path)
    metadata = path.with_suffix(".pt.json")
    if fault == "missing_sidecar":
        metadata.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"damaged")
    before = {p: p.read_bytes() for p in (path, metadata) if p.exists()}
    with pytest.raises(FlattenerError):
        scene_data.cache_scene_datasets(datasets, persistent_path=path)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not path.with_suffix(".pt.lock").exists()


def _rewrite_persistent_payload(path, change):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    change(saved)
    torch.save(saved, path)
    sidecar = path.with_suffix(".pt.json")
    metadata = json.loads(sidecar.read_text())
    metadata.update(payload_sha256=scene_data._sha256(path), payload_bytes=path.stat().st_size)
    sidecar.write_text(json.dumps(metadata))


@pytest.mark.parametrize("fault", ["image_nan", "coordinate_ulp", "storage"])
def test_persistent_tensor_admission_rejects_tampering_even_with_new_hashes(mutable, fault):
    datasets = _cache_pair(mutable)
    path = mutable / "preprocessed.pt"
    scene_data.cache_scene_datasets(datasets, persistent_path=path)

    def alter(saved):
        fields = saved["splits"]["train"]
        if fault == "image_nan":
            fields["image"][0, 0, 0, 0] = torch.nan
        elif fault == "coordinate_ulp":
            above_one = torch.nextafter(torch.tensor(1.), torch.tensor(torch.inf))
            fields["target"][0, 0, 0, 0] = above_one
        else:
            shape = fields["image"].shape
            fields["image"] = torch.zeros((shape[0] + 1, *shape[1:]))[:shape[0]]

    _rewrite_persistent_payload(path, alter)
    with pytest.raises(FlattenerError, match="semantics|budget|invalid.*tensor|storage"):
        scene_data.cache_scene_datasets(datasets, persistent_path=path)


def test_persistent_active_lock_is_not_removed_or_ignored_by_rebuild(mutable):
    path = mutable / "preprocessed.pt"
    lock = path.with_suffix(".pt.lock")
    lock.write_text("another-owner\n")
    with pytest.raises(FlattenerError, match="locked"):
        scene_data.cache_scene_datasets(_cache_pair(mutable), persistent_path=path, rebuild=True)
    assert lock.read_text() == "another-owner\n"
    assert not path.exists()


def test_persistent_source_mutation_during_publication_fails_final_guard(mutable, monkeypatch):
    datasets = _cache_pair(mutable)
    path = mutable / "preprocessed.pt"
    original = scene_data._publish_persistent

    def changed(*args):
        original(*args)
        artifact = mutable / datasets["train"].records[0]["paths"]["image.png"]
        artifact.write_bytes(artifact.read_bytes() + b"source changed during save")

    monkeypatch.setattr(scene_data, "_publish_persistent", changed)
    with pytest.raises(FlattenerError, match="artifact changed"):
        scene_data.cache_scene_datasets(datasets, persistent_path=path)
    assert not path.with_suffix(".pt.lock").exists()


def test_persistent_mmap_backing_change_is_detected_before_fetch(mutable):
    datasets = _cache_pair(mutable)
    path = mutable / "preprocessed.pt"
    scene_data.cache_scene_datasets(datasets, persistent_path=path)
    cache = scene_data.cache_scene_datasets(datasets, persistent_path=path)["train"]
    sidecar = path.with_suffix(".pt.json")
    sidecar.write_bytes(sidecar.read_bytes() + b"\n")
    for read in (lambda: cache[0], lambda: cache.fetch_batch([0]),
                 lambda: cache.verify_integrity(full=False)):
        with pytest.raises(FlattenerError, match="backing file changed"):
            read()


def test_paper_cache_rejects_same_bytes_inside_root_symlink(mutable_paper):
    dataset = scene_data.SceneCoordinateDataset(
        mutable_paper, "train", CONFIG, include_paper_augmentation=True)
    cache = scene_data.cache_scene_dataset(dataset)
    block = _manifest(mutable_paper)["collection"]["paper_augmentation"]
    path = mutable_paper / block["records"][0]["paths"]["image.png"]
    same = path.with_name(path.name + ".same")
    path.rename(same)
    try:
        path.symlink_to(same.name)
    except OSError:
        pytest.skip("host does not grant symlink creation")
    for full in (False, True):
        with pytest.raises(FlattenerError, match="symlink"):
            cache.verify_integrity(full=full)


def test_persistent_atomic_replace_failures_are_closed_and_never_unlink_old_payload(
        mutable, monkeypatch):
    datasets = _cache_pair(mutable)
    path = mutable / "preprocessed.pt"
    scene_data.cache_scene_datasets(datasets, persistent_path=path)
    metadata_path = path.with_suffix(".pt.json")
    before = {p: p.read_bytes() for p in (path, metadata_path)}
    # Alter only identity by an innocuous manifest-byte change so publication
    # two has a detectably different payload even when the underlying GT is equal.
    manifest = mutable / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    datasets = _cache_pair(mutable)
    original, calls = os.replace, []

    def interrupted(source, destination):
        calls.append(destination)
        if len(calls) == 1:
            raise PermissionError("simulated Windows sharing/publication interruption")
        return original(source, destination)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(FlattenerError, match="sharing/publication"):
        scene_data.cache_scene_datasets(datasets, persistent_path=path, rebuild=True)
    assert path.is_file() and metadata_path.is_file()
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not path.with_suffix(".pt.lock").exists()
    assert not list(mutable.glob(".preprocessed-*"))
    with pytest.raises(FlattenerError):
        scene_data.cache_scene_datasets(datasets, persistent_path=path)


def test_persistent_compressed_archive_rejected_before_torch_load(mutable, monkeypatch):
    import zipfile

    datasets = _cache_pair(mutable)
    path = mutable / "preprocessed.pt"
    scene_data.cache_scene_datasets(datasets, persistent_path=path)
    with zipfile.ZipFile(path) as archive:
        entries = [(info.filename, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    metadata_path = path.with_suffix(".pt.json")
    metadata = json.loads(metadata_path.read_text())
    metadata.update(payload_sha256=scene_data._sha256(path), payload_bytes=path.stat().st_size)
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unsafe archive loaded"))
    with pytest.raises(FlattenerError, match="compressed"):
        scene_data.cache_scene_datasets(datasets, persistent_path=path)


def test_shared_pair_hashes_common_sources_and_persistent_payload_once(mutable_paper, monkeypatch):
    datasets = {split: scene_data.SceneCoordinateDataset(
        mutable_paper, split, CONFIG, include_paper_augmentation=True)
        for split in ("train", "validation")}
    path = mutable_paper / "preprocessed.pt"
    scene_data.cache_scene_datasets(datasets, persistent_path=path)
    caches = scene_data.cache_scene_datasets(datasets, persistent_path=path)
    original, calls = scene_data._sha256, Counter()

    def counted(target):
        calls[target] += 1
        return original(target)

    monkeypatch.setattr(scene_data, "_sha256", counted)
    scene_data.verify_scene_caches(caches)
    assert calls[path] == 1 and calls[path.with_suffix(".pt.json")] == 1
    for template in caches["train"]._templates.values():
        assert calls[mutable_paper / template["path"]] == 1
    with pytest.raises(FlattenerError, match="cached train/validation"):
        scene_data.verify_scene_caches(datasets)


def test_exact_semantic_labels_are_metadata_only_not_dense_corners(paper_source, monkeypatch):
    from flattener.learning.augmentation import prepare_semantic_page_labels
    from flattener.synthetic import geometry, paper

    dataset = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, include_paper_augmentation=True)
    original_hash = scene_data._sha256(Path(scene_data.__file__))
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("semantic labels decoded pixels"))
    monkeypatch.setattr(scene_data, "make_geometry", lambda *a: pytest.fail("dense GT allocated"))
    monkeypatch.setattr(paper, "make_geometry", lambda *args: pytest.fail("paper GT allocated"))
    monkeypatch.setattr(paper, "make_capture_geometry", lambda *a: pytest.fail("capture map"))
    with torch.device("meta"):
        labels = prepare_semantic_page_labels(dataset)
        assert torch.get_default_device().type == "meta"
    assert labels.cache_bytes == 45 * len(dataset)
    assert labels.record_ids == tuple(record["id"] for record in dataset.records)
    assert labels.manifest_sha256 == dataset.manifest_sha256
    assert scene_data._sha256(Path(scene_data.__file__)) == original_hash
    actual = labels.fetch_batch(list(range(len(dataset))))
    assert all(value.device.type == "cpu" for value in actual.values())
    flat = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    for index, record in enumerate(dataset.records):
        metadata = json.loads((paper_source / record["paths"]["metadata.json"]).read_text())
        if record.get("kind") == "capture":
            expected = (2 * (np.asarray(metadata["corners_xy"]) + 0.5)
                        / np.array(metadata["original_size"][::-1]) - 1).astype(np.float32)
        else:
            backend = paper if record.get("source") == "paper" else geometry
            parameter_class = paper.PaperParameters if backend is paper else geometry.WarpParameters
            params = parameter_class(**metadata["geometry"]["parameters"])
            expected = backend.flat_to_photo(flat, params)
            h, w = dataset.output_size
            centers = flat * np.array((1 - 1 / w, 1 - 1 / h))
            wrong_pixel_corners = backend.flat_to_photo(centers, params)
            assert not np.allclose(expected, wrong_pixel_corners, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(actual["corner_target"][index].numpy(), expected)
        assert actual["corner_valid"][index].all()
    labels.verify_integrity()


def test_semantic_table_preserves_persistent_cache_bytes_and_warm_no_decode(mutable, monkeypatch):
    from flattener.learning.augmentation import prepare_semantic_page_labels

    pair = _cache_pair(mutable)
    path = mutable / "preprocessed.pt"
    scene_data.cache_scene_datasets(pair, persistent_path=path)
    digests = {item: scene_data._sha256(item) for item in (path, path.with_suffix(".pt.json"))}
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("warm cache decoded pixels"))
    warm = scene_data.cache_scene_datasets(pair, persistent_path=path)
    assert warm["train"].cache_stats["status"] == "loaded"
    labels = prepare_semantic_page_labels(warm["train"])
    batch = warm["train"].fetch_batch([2, 0, 2])
    batch.update(labels.fetch_batch([2, 0, 2]))
    assert batch["corner_target"].shape == (3, 4, 2)
    assert digests == {item: scene_data._sha256(item) for item in digests}
    labels.fetch_batch([0])["corner_target"].zero_()
    assert not torch.equal(labels.fetch_batch([0])["corner_target"], torch.zeros((1, 4, 2)))
    scene_data.verify_scene_caches(warm)


def test_semantic_label_split_order_and_output_resolution_invariance(paper_source):
    from flattener.learning.augmentation import prepare_semantic_page_labels

    first = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, (101, 83), include_paper_augmentation=True)
    second = scene_data.SceneCoordinateDataset(
        paper_source, "train", CONFIG, (173, 131), include_paper_augmentation=True)
    a, b = (prepare_semantic_page_labels(value) for value in (first, second))
    assert a.record_ids == b.record_ids and a.split == b.split == "train"
    for key in a.fetch_batch([0, -1]):
        torch.testing.assert_close(a.fetch_batch([0, -1])[key], b.fetch_batch([0, -1])[key],
                                   rtol=0, atol=0, equal_nan=True)


def test_semantic_corner_out_of_frame_or_nonfinite_is_nan_not_clipped(scene_source, monkeypatch):
    from flattener.learning.augmentation import prepare_semantic_page_labels
    from flattener.synthetic import geometry

    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    raw = np.array([[-1.2, -.5], [.5, -.5], [.5, .5], [-1.2, .5]], dtype=np.float32)
    monkeypatch.setattr(geometry, "flat_to_photo", lambda points, params: raw.copy())
    result = prepare_semantic_page_labels(dataset).fetch_batch([0])
    assert result["corner_valid"].tolist() == [[False, True, True, False]]
    assert torch.isnan(result["corner_target"][0, [0, 3]]).all()
    assert result["direction_valid"].tolist() == [True]
    assert result["direction_target"].tolist() == [0]
    raw[0] = np.nan
    result = prepare_semantic_page_labels(dataset).fetch_batch([0])
    assert not result["direction_valid"][0]


def test_semantic_metadata_mutation_detected_before_and_after_prepare(mutable, monkeypatch):
    from flattener.learning.augmentation import prepare_semantic_page_labels
    from flattener.synthetic import geometry

    dataset = scene_data.SceneCoordinateDataset(mutable, "train", CONFIG)
    path = mutable / dataset.records[0]["paths"]["metadata.json"]
    labels = prepare_semantic_page_labels(dataset)
    original = geometry.flat_to_photo

    def mutate_after_read(points, params):
        path.write_bytes(path.read_bytes() + b" ")
        return original(points, params)

    monkeypatch.setattr(geometry, "flat_to_photo", mutate_after_read)
    with pytest.raises(FlattenerError, match="changed"):
        prepare_semantic_page_labels(dataset)
    with pytest.raises(FlattenerError, match="changed"):
        labels.verify_integrity()


def test_semantic_table_budget_and_batch_index_contract(scene_source, monkeypatch):
    from flattener.learning.augmentation import prepare_semantic_page_labels

    dataset = scene_data.SceneCoordinateDataset(scene_source, "train", CONFIG)
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("labels decoded pixels"))
    with pytest.raises(FlattenerError, match="budget"):
        prepare_semantic_page_labels(dataset, max_bytes=len(dataset) * 45 - 1)
    labels = prepare_semantic_page_labels(dataset, max_bytes=len(dataset) * 45)
    for indices in ([True], [0.5], [], [[0]], torch.tensor([0.0]), np.array([True])):
        with pytest.raises(FlattenerError):
            labels.fetch_batch(indices)
    with pytest.raises(IndexError):
        labels.fetch_batch([len(dataset)])
