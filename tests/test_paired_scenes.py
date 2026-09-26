"""Controlled paired-data contrast, without opening production held-out pixels."""

import hashlib
import json
import random
import shutil
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from flattener import FlattenerError
from flattener.synthetic.paired_scenes import (
    PairedSceneConfig,
    generate_paired_scene_dataset,
    load_paired_scene,
    validate_paired_scene_dataset,
)
from flattener.synthetic.scene_v2 import SIZE_BINS, generate_scene_v2
from flattener.synthetic.scenes import generate_scene

SMALL = PairedSceneConfig(
    train_documents=3, validation_documents=3, test_documents=3, image_size=(128, 128)
)


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-scene-source")
    result = {}
    for condition in ("matched", "independent"):
        path = root / condition
        generate_paired_scene_dataset(path, replace(SMALL, training_panels=condition))
        result[condition] = path
    return result


@pytest.fixture
def dataset(datasets, tmp_path):
    path = tmp_path / "dataset"
    shutil.copytree(datasets["matched"], path)
    return path


def manifest_at(path):
    return json.loads((path / "manifest.json").read_text())


def write_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest))


def change_metadata(path, record, change):
    target = path / record["paths"]["metadata.json"]
    value = json.loads(target.read_text())
    change(value)
    target.write_text(json.dumps(value))
    record["sha256"]["metadata.json"] = hashlib.sha256(target.read_bytes()).hexdigest()


def test_default_counts_and_schema_contract():
    from flattener.synthetic.paired_scenes import SCENE_SCHEMA_VERSION, _counts

    config = PairedSceneConfig()
    assert config.seed == 20260924 and config.training_panels == "matched"
    assert config.image_size == (192, 144) and config.variants_per_document == 4
    assert SCENE_SCHEMA_VERSION == "scene-0.3"
    train, evaluation = _counts(96), _counts(24)
    assert (train["page"], train["negative"], train["total"]) == (384, 192, 576)
    assert train["negative_types"] == {"background": 96, "paper_panel": 96}
    assert train["blank_reference_size_bins"] == dict.fromkeys(SIZE_BINS, 32)
    assert (evaluation["page"], evaluation["negative"], evaluation["total"]) == (96, 48, 144)
    assert evaluation["blank_reference_size_bins"] == dict.fromkeys(SIZE_BINS, 8)


def test_controlled_contrast_changes_only_train_blank_images_and_pair_metadata(datasets):
    matched, independent = (manifest_at(datasets[mode]) for mode in ("matched", "independent"))
    assert matched["groups"] == independent["groups"]
    assert matched["counts"] == independent["counts"]
    assert [r["id"] for r in matched["samples"]] == [r["id"] for r in independent["samples"]]
    changed_rgb = []
    for a, b in zip(matched["samples"], independent["samples"], strict=True):
        if a["split"] != "train":
            assert a == b
            for artifact in a["paths"]:
                assert (datasets["matched"] / a["paths"][artifact]).read_bytes() == (
                    datasets["independent"] / b["paths"][artifact]
                ).read_bytes()
        assert a["sha256"]["mask.png"] == b["sha256"]["mask.png"]
        if a["sha256"]["image.png"] != b["sha256"]["image.png"]:
            changed_rgb.append(a)
            assert a["split"] == "train" and a["negative_type"] == "paper_panel"
        if a["split"] == "train":
            assert b["pair_id"] is b["pair_role"] is b["partner_id"] is None
    assert len(changed_rgb) == SMALL.train_documents
    assert len(matched["pairs"]) == 9 and len(independent["pairs"]) == 6


def test_pair_has_identical_material_context_and_different_printed_gt(datasets):
    path = datasets["matched"]
    checked = validate_paired_scene_dataset(path)
    assert checked["sample_counts"] == dict.fromkeys(("train", "validation", "test"), 18)
    manifest = checked["manifest"]
    records = {r["id"]: r for r in manifest["samples"]}
    for pair in manifest["pairs"]:
        if pair["split"] == "test":
            continue
        printed = load_paired_scene(path, records[pair["printed_id"]])
        blank = load_paired_scene(path, records[pair["blank_id"]])
        a, b = printed.metadata, blank.metadata
        assert printed.mask.any() and not blank.mask.any()
        assert not np.array_equal(printed.image, blank.image)
        assert a["physical_page"] == b["physical_page"]
        assert a["pair_context"] == b["pair_context"]
        assert a["seeds"] == b["seeds"]
        assert a["background"] == b["background"] and a["photometric"] == b["photometric"]
        assert a["source_document_sha256"] == b["source_document_sha256"]
        assert a["capture_session"] == b["capture_session"]
        assert b["geometry"] is None and b["document"] is None and b["foreground_fraction"] == 0
        assert b["physical_page"]["foreground_fraction"] > 0
        expected_color = np.floor(
            np.clip(np.array(a["paper"]["tint_rgb_multiplier"]) * 255, 0, 255) + 0.5
        ).astype(np.uint8)
        assert b["panel"]["rgb"] == expected_color.tolist()


def test_printed_pixels_and_metadata_inputs_reuse_exact_v2_generator(datasets):
    directory = datasets["matched"]
    record = next(
        r for r in manifest_at(directory)["samples"] if r["split"] == "train" and r["has_page"]
    )
    paired = load_paired_scene(directory, record)
    metadata = paired.metadata
    baseline = generate_scene_v2(
        record["document_group"],
        metadata["seeds"]["document"],
        metadata["seeds"]["scene"],
        image_size=SMALL.image_size,
        size_bin=record["size_bin"],
        family=record["family"],
        background_type=record["background_type"],
    )
    np.testing.assert_array_equal(paired.image, baseline.image)
    np.testing.assert_array_equal(paired.mask, baseline.mask)
    for key in ("document", "geometry", "paper", "background", "photometric", "seeds"):
        assert metadata[key] == baseline.metadata[key]


def test_generator_does_not_decode_any_generated_image_or_filter_pairs(tmp_path, monkeypatch):
    from flattener.synthetic import paired_scenes

    monkeypatch.setattr(
        Image, "open", lambda *args: pytest.fail("no generated file may be decoded")
    )
    monkeypatch.setattr(paired_scenes.torch, "equal", lambda *args: True)
    path = tmp_path / "all-identical-qa"
    manifest = generate_paired_scene_dataset(
        path, replace(SMALL, train_documents=1, validation_documents=1, test_documents=1)
    )
    assert len(manifest["samples"]) == 18
    for report in manifest["preprocessing_qa"].values():
        assert report["comparisons_checked"] == report["byte_identical_after_128x96"] == 1
        assert report["excluded_or_regenerated"] == 0


def test_validator_never_decodes_images_or_gt(datasets, monkeypatch):
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("no held-out decode"))
    for path in datasets.values():
        validate_paired_scene_dataset(path)


def test_loader_decodes_only_requested_artifacts(datasets, monkeypatch):
    directory = datasets["matched"]
    record = next(r for r in manifest_at(directory)["samples"] if r["pair_role"] == "blank")
    expected = {(directory / record["paths"][name]).resolve() for name in ("image.png", "mask.png")}
    original = Image.open
    seen = []

    def open_only_selected(path, *args, **kwargs):
        assert path.resolve() in expected
        seen.append(path.resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", open_only_selected)
    assert not load_paired_scene(directory, record).mask.any()
    assert set(seen) == expected


def test_full_byte_reproducibility(datasets, tmp_path):
    destination = tmp_path / "repeat"
    generate_paired_scene_dataset(destination, SMALL)
    first = datasets["matched"]
    paths = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert paths == sorted(
        p.relative_to(destination) for p in destination.rglob("*") if p.is_file()
    )
    for relative in paths:
        assert (first / relative).read_bytes() == (destination / relative).read_bytes()


def test_v1_v2_outputs_and_global_rng_unchanged(tmp_path):
    random.seed(100)
    np.random.seed(100)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    old1 = generate_scene("old", 1, 2, image_size=(128, 128))
    old2 = generate_scene_v2("old", 1, 2, image_size=(128, 128))
    generate_paired_scene_dataset(
        tmp_path / "new",
        replace(SMALL, train_documents=1, validation_documents=1, test_documents=1),
    )
    new1 = generate_scene("old", 1, 2, image_size=(128, 128))
    new2 = generate_scene_v2("old", 1, 2, image_size=(128, 128))
    for before, after in ((old1, new1), (old2, new2)):
        np.testing.assert_array_equal(before.image, after.image)
        np.testing.assert_array_equal(before.mask, after.mask)
        assert before.metadata == after.metadata
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[0] == numpy_state[0] and actual_numpy[2:] == numpy_state[2:]


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 10**400},
        {"seed": True},
        {"train_documents": 0},
        {"train_documents": 10**400},
        {"test_documents": float("inf")},
        {"validation_documents": float("nan")},
        {"variants_per_document": 3},
        {"variants_per_document": True},
        {"image_size": (128, 10**400)},
        {"image_size": (127, 128)},
        {"training_panels": "automatic"},
        {"training_panels": True},
        {"train_documents": 10000, "validation_documents": 10000},
    ],
)
def test_bad_config_never_creates_output_or_leaks_raw_exceptions(tmp_path, change):
    destination = tmp_path / "bad"
    with pytest.raises(FlattenerError):
        generate_paired_scene_dataset(destination, replace(SMALL, **change))
    assert not destination.exists()


def test_existing_output_never_replaced(dataset):
    original = (dataset / "manifest.json").read_bytes()
    with pytest.raises(FlattenerError) as raised:
        generate_paired_scene_dataset(dataset, SMALL)
    assert raised.value.reason_code == "dataset_create_failed"
    assert (dataset / "manifest.json").read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(complete=False),
        lambda m: m.update(schema_version="scene-0.2"),
        lambda m: m["samples"].pop(),
        lambda m: m["pairs"].pop(),
        lambda m: m["pairs"][0].update(blank_id=m["pairs"][1]["blank_id"]),
        lambda m: m["groups"][1].update(seed=m["groups"][0]["seed"]),
        lambda m: m["samples"][0].update(split="validation"),
        lambda m: m["counts"]["train"]["size_bins"].update(small=999),
        lambda m: m["config"].update(training_panels="independent"),
        lambda m: m["preprocessing_qa"]["train"].update(excluded_or_regenerated=1),
        lambda m: m.pop("preprocessing_qa_protocol"),
        lambda m: m["preprocessing_qa_protocol"].update(input_size=[1, 1]),
        lambda m: m["preprocessing_qa_protocol"].update(selection="filter indistinguishable pairs"),
    ],
)
def test_manifest_corruption_rejected(dataset, mutation):
    manifest = manifest_at(dataset)
    mutation(manifest)
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError):
        validate_paired_scene_dataset(dataset)


@pytest.mark.parametrize(
    "field,value",
    [
        ("foreground_fraction", 10**400),
        ("foreground_fraction", float("inf")),
        ("foreground_fraction", float("nan")),
        ("bbox", [0, 0, 10**400, 10]),
        ("bbox", [10, 10, 2, 2]),
        ("bbox", [0, 0, 4.5, 10]),
    ],
)
def test_extreme_numeric_metadata_is_structured_failure(dataset, field, value):
    manifest = manifest_at(dataset)
    record = manifest["samples"][0]
    change_metadata(dataset, record, lambda metadata: metadata.update({field: value}))
    write_manifest(dataset, manifest)
    for action in (
        lambda: validate_paired_scene_dataset(dataset),
        lambda: load_paired_scene(dataset, record),
    ):
        with pytest.raises(FlattenerError):
            action()


def test_modified_counterfactual_tint_is_rejected(dataset):
    manifest = manifest_at(dataset)
    record = next(r for r in manifest["samples"] if r["pair_role"] == "blank")
    change_metadata(dataset, record, lambda metadata: metadata["panel"].update(rgb=[0, 0, 0]))
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="paper color"):
        validate_paired_scene_dataset(dataset)


def test_panel_geometry_cannot_disagree_with_physical_surface(dataset):
    manifest = manifest_at(dataset)
    record = next(r for r in manifest["samples"] if r["pair_role"] == "blank")
    change_metadata(
        dataset,
        record,
        lambda metadata: metadata["panel"]["geometry"].update(
            photo_page_fraction=0.01,
        ),
    )
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="physical geometry"):
        validate_paired_scene_dataset(dataset)


def test_modified_context_cannot_keep_its_old_hash(dataset):
    manifest = manifest_at(dataset)
    record = manifest["samples"][0]
    change_metadata(
        dataset, record, lambda metadata: metadata["pair_context"]["paper"].update(seed=33)
    )
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="context hash"):
        validate_paired_scene_dataset(dataset)


@pytest.mark.parametrize("artifact", ("image.png", "mask.png", "metadata.json"))
def test_hash_changes_rejected(dataset, artifact):
    record = manifest_at(dataset)["samples"][0]
    with (dataset / record["paths"][artifact]).open("ab") as stream:
        stream.write(b"tampered")
    for action in (
        lambda: validate_paired_scene_dataset(dataset),
        lambda: load_paired_scene(dataset, record),
    ):
        with pytest.raises(FlattenerError) as raised:
            action()
        assert raised.value.reason_code == "artifact_hash_mismatch"


@pytest.mark.parametrize("symlink", (False, True))
def test_artifact_escape_rejected(dataset, tmp_path, symlink):
    manifest = manifest_at(dataset)
    record = manifest["samples"][0]
    outside = tmp_path / "outside.png"
    shutil.copyfile(dataset / record["paths"]["image.png"], outside)
    if symlink:
        (dataset / "link.png").symlink_to(outside)
        record["paths"]["image.png"] = "link.png"
    else:
        record["paths"]["image.png"] = "../outside.png"
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="escapes"):
        validate_paired_scene_dataset(dataset)


def test_json_size_limits_are_enforced_before_parse(dataset, monkeypatch):
    from flattener.synthetic import paired_scenes

    record = manifest_at(dataset)["samples"][0]
    monkeypatch.setattr(paired_scenes, "_MANIFEST_LIMIT", 16)
    with pytest.raises(FlattenerError, match="size limit"):
        validate_paired_scene_dataset(dataset)
    monkeypatch.setattr(paired_scenes, "_METADATA_LIMIT", 16)
    with pytest.raises(FlattenerError, match="size limit"):
        load_paired_scene(dataset, record)


def test_blank_mask_cannot_become_physical_surface_gt(dataset):
    manifest = manifest_at(dataset)
    record = next(r for r in manifest["samples"] if r["pair_role"] == "blank")
    target = dataset / record["paths"]["mask.png"]
    values = np.full((128, 128), 255, dtype=np.uint8)
    with Image.fromarray(values) as encoded:
        encoded.save(target)
    record["sha256"]["mask.png"] = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FlattenerError, match="presence/area/bbox"):
        load_paired_scene(dataset, record)
