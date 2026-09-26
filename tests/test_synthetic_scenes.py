"""Contract tests for the independent page-region scene distribution."""

import copy
import hashlib
import json
import random
import shutil
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from flattener import FlattenerError
from flattener.synthetic.geometry import WarpParameters, flat_to_photo, make_geometry
from flattener.synthetic.pipeline import PhotometricConfig
from flattener.synthetic.scenes import (
    BACKGROUND_TYPES,
    SCENE_FAMILIES,
    SceneConfig,
    generate_background,
    generate_scene,
    generate_scene_dataset,
    load_scene,
    sample_scene_warp,
    validate_scene_dataset,
)

SMALL = SceneConfig(
    seed=19,
    train_documents=2,
    validation_documents=1,
    test_documents=1,
    variants_per_document=2,
    image_size=(128, 128),
    negative_fraction=1 / 3,
)
OFF = PhotometricConfig(enabled=False)


@pytest.fixture(scope="module")
def source_dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("scene-source") / "dataset"
    generate_scene_dataset(directory, SMALL)
    return directory


@pytest.fixture
def dataset(source_dataset, tmp_path):
    directory = tmp_path / "data"
    shutil.copytree(source_dataset, directory)
    return directory


def read_manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def write_manifest(directory, manifest):
    (directory / "manifest.json").write_text(json.dumps(manifest))


def update_metadata(directory, record, change):
    path = directory / record["paths"]["metadata.json"]
    metadata = json.loads(path.read_text())
    change(metadata)
    path.write_text(json.dumps(metadata))
    record["sha256"]["metadata.json"] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("family", SCENE_FAMILIES)
def test_scene_is_reproducible_and_analytic(family):
    first = generate_scene("authored-1", 11, 23, image_size=(128, 160), family=family)
    second = generate_scene("authored-1", 11, 23, image_size=(128, 160), family=family)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.metadata == second.metadata
    assert first.image.shape == (128, 160, 3)
    assert first.image.dtype == np.uint8
    assert first.mask.shape == (128, 160) and first.mask.dtype == bool
    params = WarpParameters(**first.metadata["geometry"]["parameters"])
    expected = make_geometry((128, 160), (128, 160), params).photo_page_mask
    np.testing.assert_array_equal(first.mask, expected)
    assert 0.015 < first.mask.mean() < 0.85
    assert not first.mask[[0, -1]].any() and not first.mask[:, [0, -1]].any()
    assert first.metadata["compositing"]["page_render_calls"] == 1
    assert first.metadata["document"]["font"]["license"] == "OFL-1.1"


@pytest.mark.parametrize("family", SCENE_FAMILIES)
def test_continuous_page_is_inside_frame(family):
    coordinates = np.linspace(-1, 1, 61)
    x, y = np.meshgrid(coordinates, coordinates)
    points = np.stack((x, y), axis=-1)
    extents, translations, angles = [], [], []
    for seed in range(40):
        params = sample_scene_warp(family, seed)
        scene = flat_to_photo(points, params)
        assert np.isfinite(scene).all()
        assert np.max(np.abs(scene)) <= 0.970001
        extents.append(np.ptp(scene, axis=(0, 1)).max())
        translations.append(max(abs(params.translate_x), abs(params.translate_y)))
        angles.append(params.rotation)
    assert max(extents) > min(extents) * 1.8
    assert max(translations) > 0.2
    assert min(angles) < -0.4 and max(angles) > 0.4


@pytest.mark.parametrize("kind", BACKGROUND_TYPES)
def test_backgrounds_are_deterministic_and_nonconstant(kind):
    first, first_meta = generate_background((128, 144), 47, kind)
    second, second_meta = generate_background((128, 144), 47, kind)
    np.testing.assert_array_equal(first, second)
    assert first_meta == second_meta
    assert first.shape == (128, 144, 3) and first.dtype == np.uint8
    assert first.std() > 5
    assert first_meta["type"] == kind
    assert "not a photograph" in first_meta["source"]
    changed, _ = generate_background((128, 144), 48, kind)
    assert not np.array_equal(first, changed)


def test_background_and_photometry_do_not_change_geometry():
    first = generate_scene("a", 5, 29, background_seed=100, photometric_seed=200)
    background = generate_scene("a", 5, 29, background_seed=101, photometric_seed=200)
    appearance = generate_scene("a", 5, 29, background_seed=100, photometric_seed=201)
    for changed in (background, appearance):
        np.testing.assert_array_equal(first.mask, changed.mask)
        assert first.metadata["geometry"] == changed.metadata["geometry"]
        assert first.metadata["document"] == changed.metadata["document"]
        assert not np.array_equal(first.image, changed.image)


def test_negative_is_exact_procedural_background_without_document(monkeypatch):
    from flattener.synthetic import scenes

    def forbidden(*args, **kwargs):
        raise AssertionError("negative scene must not generate a page")

    monkeypatch.setattr(scenes, "generate_document", forbidden)
    sample = generate_scene(
        "negative",
        2,
        31,
        has_page=False,
        background_seed=101,
        background_type="shapes",
        photometric=OFF,
    )
    expected, _ = generate_background((192, 144), 101, "shapes")
    np.testing.assert_array_equal(sample.image, expected)
    assert not sample.mask.any()
    assert sample.metadata["document"] is None
    assert sample.metadata["geometry"] is None
    assert sample.metadata["foreground_fraction"] == 0
    assert sample.metadata["bbox"] is None
    assert sample.metadata["family"] == "none"


def test_local_rng_does_not_touch_global_state():
    random.seed(20)
    np.random.seed(20)
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    generate_scene("rng", 5, 16, image_size=(128, 128))
    assert random.getstate() == expected_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]


def test_boundary_composition_has_no_black_or_white_fill_halo(monkeypatch):
    from flattener.synthetic import scenes

    # Set a constant paper and constant background. Every subpixel-composited
    # channel must lie between those colors, never leak renderer fill black.
    authored = scenes.generate_document("constant", 1, (128, 128))
    authored.image[:] = 180
    monkeypatch.setattr(scenes, "generate_document", lambda *args: authored)
    monkeypatch.setattr(
        scenes,
        "generate_background",
        lambda *args: (
            np.full((128, 128, 3), 70, dtype=np.uint8),
            {"type": "gradient"},
        ),
    )
    sample = generate_scene("constant", 1, 9, image_size=(128, 128), photometric=OFF)
    tint = np.array(sample.metadata["paper"]["tint_rgb_multiplier"])
    paper_rgb = np.floor(np.clip(180 * tint, 0, 255) + 0.5).astype(np.uint8)
    assert (sample.image >= np.minimum(paper_rgb, 70)).all()
    assert (sample.image <= np.maximum(paper_rgb, 70)).all()
    assert np.unique(sample.image.reshape(-1, 3), axis=0).shape[0] > 2


def test_full_dataset_reproducible_bytes(source_dataset, tmp_path):
    second = tmp_path / "second"
    generate_scene_dataset(second, SMALL)
    first_paths = sorted(
        p.relative_to(source_dataset) for p in source_dataset.rglob("*") if p.is_file()
    )
    second_paths = sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    assert first_paths == second_paths
    for relative in first_paths:
        assert (source_dataset / relative).read_bytes() == (second / relative).read_bytes()


def test_counts_groups_and_loaded_masks(source_dataset):
    verified = validate_scene_dataset(source_dataset)
    assert verified["sample_counts"] == {"train": 6, "validation": 3, "test": 3}
    assert verified["page_counts"] == {"train": 4, "validation": 2, "test": 2}
    assert verified["negative_counts"] == {"train": 2, "validation": 1, "test": 1}
    manifest = verified["manifest"]
    group_splits = {group["id"]: group["split"] for group in manifest["groups"]}
    for record in manifest["samples"]:
        sample = load_scene(source_dataset, record)
        assert group_splits[record["document_group"]] == record["split"]
        assert bool(sample.mask.any()) == record["has_page"]
        assert sample.metadata["is_real_photo"] is False
        assert sample.image.shape == (128, 128, 3)
        if sample.mask.any():
            ys, xs = np.nonzero(sample.mask)
            assert sample.metadata["bbox"] == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    assert manifest["external_images"] is False
    assert set(manifest["generator_code_sha256"]) == {
        "scenes.py",
        "documents.py",
        "geometry.py",
        "pipeline.py",
    }


def test_default_exact_counts_without_generation():
    from flattener.synthetic.scenes import _negative_count

    assert _negative_count(384, 0.25) == 128
    assert _negative_count(64, 0.25) == 21


def test_validation_does_not_decode_held_out_images(source_dataset, monkeypatch):
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("must not decode masks"))
    validate_scene_dataset(source_dataset)


def test_existing_directory_never_overwritten(dataset):
    before = (dataset / "manifest.json").read_bytes()
    with pytest.raises(FlattenerError) as raised:
        generate_scene_dataset(dataset, SMALL)
    assert raised.value.reason_code == "dataset_create_failed"
    assert (dataset / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**63},
        {"train_documents": 0},
        {"validation_documents": False},
        {"test_documents": 1.5},
        {"variants_per_document": 0},
        {"image_size": (127, 128)},
        {"image_size": (128, 513)},
        {"image_size": (True, 128)},
        {"image_size": [128, 128]},
        {"negative_fraction": -0.1},
        {"negative_fraction": 0.76},
        {"negative_fraction": float("nan")},
        {"negative_fraction": True},
        {"train_documents": 10_000, "variants_per_document": 10_000},
    ],
)
def test_config_errors_happen_before_directory_creation(tmp_path, change):
    directory = tmp_path / "invalid"
    with pytest.raises(FlattenerError):
        generate_scene_dataset(directory, replace(SMALL, **change))
    assert not directory.exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(complete=False),
        lambda m: m.update(schema_version="unknown"),
        lambda m: m.update(groups=None),
        lambda m: m.update(samples={}),
        lambda m: m["samples"].pop(),
        lambda m: m["samples"].append(copy.deepcopy(m["samples"][0])),
        lambda m: m["groups"].append(copy.deepcopy(m["groups"][0])),
        lambda m: m["groups"][1].update(seed=m["groups"][0]["seed"]),
        lambda m: m["samples"][0].update(split="test"),
        lambda m: m["samples"][0].update(has_page=False),
        lambda m: m["samples"][0].update(background_type="photo"),
        lambda m: m["counts"]["train"].update(total=100),
    ],
)
def test_manifest_corruption_rejected(dataset, change):
    manifest = read_manifest(dataset)
    change(manifest)
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError):
        validate_scene_dataset(dataset)


def test_metadata_mismatch_rejected_even_with_updated_hash(dataset):
    manifest = read_manifest(dataset)
    record = manifest["samples"][0]
    update_metadata(dataset, record, lambda metadata: metadata.update(split="test"))
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="metadata disagrees"):
        validate_scene_dataset(dataset)
    with pytest.raises(FlattenerError, match="metadata disagrees"):
        load_scene(dataset, record)


def test_document_variant_hash_mismatch_rejected(dataset):
    manifest = read_manifest(dataset)
    record = manifest["samples"][1]
    update_metadata(
        dataset, record, lambda metadata: metadata["document"].update(source_sha256="0" * 64)
    )
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="share one authored document"):
        validate_scene_dataset(dataset)


def test_document_hash_leakage_rejected(dataset):
    manifest = read_manifest(dataset)
    source = manifest["samples"][0]
    target = next(record for record in manifest["samples"] if record["split"] == "validation")
    source_meta = json.loads((dataset / source["paths"]["metadata.json"]).read_text())
    update_metadata(
        dataset,
        target,
        lambda metadata: metadata["document"].update(
            source_sha256=source_meta["document"]["source_sha256"],
        ),
    )
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="identical authored document crosses splits"):
        validate_scene_dataset(dataset)


@pytest.mark.parametrize("artifact", ("image.png", "mask.png", "metadata.json"))
def test_hash_mismatch_rejected_on_validate_and_load(dataset, artifact):
    record = read_manifest(dataset)["samples"][0]
    with (dataset / record["paths"][artifact]).open("ab") as stream:
        stream.write(b"corrupt")
    for action in (lambda: validate_scene_dataset(dataset), lambda: load_scene(dataset, record)):
        with pytest.raises(FlattenerError) as raised:
            action()
        assert raised.value.reason_code == "artifact_hash_mismatch"


@pytest.mark.parametrize("use_symlink", (False, True))
def test_escape_path_rejected(dataset, tmp_path, use_symlink):
    manifest = read_manifest(dataset)
    record = manifest["samples"][0]
    external = tmp_path / "outside.png"
    shutil.copyfile(dataset / record["paths"]["image.png"], external)
    if use_symlink:
        (dataset / "link.png").symlink_to(external)
        record["paths"]["image.png"] = "link.png"
    else:
        record["paths"]["image.png"] = "../outside.png"
    write_manifest(dataset, manifest)
    for action in (lambda: validate_scene_dataset(dataset), lambda: load_scene(dataset, record)):
        with pytest.raises(FlattenerError, match="escapes"):
            action()


@pytest.mark.parametrize("corruption", ("mode", "value", "empty", "geometry"))
def test_invalid_mask_with_updated_hash_rejected(dataset, corruption):
    manifest = read_manifest(dataset)
    record = manifest["samples"][0]
    path = dataset / record["paths"]["mask.png"]
    with Image.open(path) as encoded:
        values = np.array(encoded)
    if corruption == "mode":
        values = np.repeat(values[..., None], 3, axis=-1)
    elif corruption == "value":
        values[1, 1] = 127
    elif corruption == "empty":
        values[:] = 0
    else:
        values[values == 255] = 0
        values[50:60, 50:60] = 255
        update_metadata(
            dataset,
            record,
            lambda metadata: metadata.update(
                bbox=[50, 50, 60, 60],
                foreground_fraction=100 / (128 * 128),
            ),
        )
    with Image.fromarray(values) as encoded:
        encoded.save(path)
    record["sha256"]["mask.png"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FlattenerError):
        load_scene(dataset, record)


def test_negative_fraction_zero_supported(tmp_path):
    directory = tmp_path / "positive-only"
    generate_scene_dataset(
        directory, replace(SMALL, train_documents=1, variants_per_document=1, negative_fraction=0)
    )
    result = validate_scene_dataset(directory)
    assert result["negative_counts"] == {"train": 0, "validation": 0, "test": 0}
    assert result["sample_counts"] == {"train": 1, "validation": 1, "test": 1}
