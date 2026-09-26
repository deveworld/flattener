"""Independent contracts for actual-area strata and printed-page negatives."""

import copy
import hashlib
import json
import random
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flattener import FlattenerError
from flattener.synthetic.geometry import flat_to_photo
from flattener.synthetic.pipeline import PhotometricConfig, derive_seed
from flattener.synthetic.scene_v2 import (
    SIZE_BINS,
    TASK_SCOPE,
    SceneV2Config,
    generate_scene_v2,
    generate_scene_v2_dataset,
    load_scene_v2,
    sample_stratified_geometry,
    size_bin_for_fraction,
    validate_scene_v2_dataset,
)
from flattener.synthetic.scenes import (
    BACKGROUND_TYPES,
    SCENE_FAMILIES,
    generate_background,
    generate_scene,
)

SMALL = SceneV2Config(
    train_documents=3,
    validation_documents=3,
    test_documents=3,
    variants_per_document=2,
    image_size=(128, 128),
)
OFF = PhotometricConfig(enabled=False)


@pytest.fixture(scope="module")
def source_dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("scene-v2-source") / "dataset"
    generate_scene_v2_dataset(path, SMALL)
    return path


@pytest.fixture
def dataset(source_dataset, tmp_path):
    path = tmp_path / "data"
    shutil.copytree(source_dataset, path)
    return path


def manifest_at(path):
    return json.loads((path / "manifest.json").read_text())


def write_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest))


def mutate_metadata(path, record, change):
    target = path / record["paths"]["metadata.json"]
    value = json.loads(target.read_text())
    change(value)
    target.write_text(json.dumps(value))
    record["sha256"]["metadata.json"] = hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "fraction,expected",
    [
        (0.01, "small"),
        (0.249999, "small"),
        (0.25, "medium"),
        (0.499999, "medium"),
        (0.5, "large"),
        (1, "large"),
    ],
)
def test_exact_size_bin_boundaries(fraction, expected):
    assert size_bin_for_fraction(fraction) == expected


@pytest.mark.parametrize(
    "fraction", (10**400, -(10**400), float("inf"), -float("inf"), float("nan"))
)
def test_extreme_integer_fraction_is_structured_input_error(fraction, tmp_path):
    with pytest.raises(FlattenerError):
        size_bin_for_fraction(fraction)
    output = tmp_path / "extreme-input"
    with pytest.raises(FlattenerError):
        generate_scene_v2_dataset(output, replace(SMALL, negative_fraction=fraction))
    assert not output.exists()


@pytest.mark.parametrize("value", (10**400, -(10**400), float("inf"), float("nan")))
def test_invalid_photometry_never_leaks_numeric_conversion_error(value, monkeypatch):
    from flattener.synthetic import scene_v2

    monkeypatch.setattr(
        scene_v2, "generate_background", lambda *args: pytest.fail("validate first")
    )
    with pytest.raises(FlattenerError):
        generate_scene_v2("numeric", 1, 2, photometric=PhotometricConfig(brightness_min=value))


@pytest.mark.parametrize("family", SCENE_FAMILIES)
@pytest.mark.parametrize("size_bin", SIZE_BINS)
def test_stratified_geometry_uses_real_source_grid_area_and_is_visible(family, size_bin):
    u, v = np.meshgrid(np.linspace(-1, 1, 45), np.linspace(-1, 1, 45))
    page_points = np.stack((u, v), axis=-1)
    for seed in (0, 1, 3, 47, 831):
        params, geometry, sampling = sample_stratified_geometry(family, size_bin, seed)
        fraction = float(geometry.photo_page_mask.mean())
        assert size_bin_for_fraction(fraction) == size_bin
        assert sampling["foreground_fraction"] == fraction
        assert sampling["actual_size_bin"] == sampling["requested_size_bin"] == size_bin
        assert 1 <= sampling["attempts"] <= 256
        assert geometry.valid_mask.all()
        transformed = flat_to_photo(page_points, params)
        assert np.isfinite(transformed).all()
        assert np.abs(transformed).max() <= 0.985001
        assert geometry.metadata["folds"] is False
        assert geometry.metadata["synthesis_map_direction"] == "photo_output_to_flat_source"


def test_rejection_cap_is_exact_and_failure_explicit(monkeypatch):
    from flattener.synthetic import scene_v2

    attempts = []
    candidate = scene_v2._candidate_warp("perspective", "large", 1)

    def always_wrong(family, size_bin, seed):
        attempts.append(seed)
        return candidate

    monkeypatch.setattr(scene_v2, "_candidate_warp", always_wrong)
    monkeypatch.setattr(
        scene_v2,
        "make_geometry",
        lambda *args: SimpleNamespace(
            photo_page_mask=np.ones((128, 128), dtype=bool),
        ),
    )
    with pytest.raises(FlattenerError) as raised:
        sample_stratified_geometry("perspective", "small", 3, (128, 128))
    assert raised.value.reason_code == "scene_geometry_sampling_failed"
    assert len(attempts) == len(set(attempts)) == 256
    assert "256 attempts" in raised.value.message


@pytest.mark.parametrize("size_bin", SIZE_BINS)
def test_sample_pixels_masks_and_metadata_reproduce(size_bin):
    first = generate_scene_v2("printed-1", 23, 41, size_bin=size_bin)
    second = generate_scene_v2("printed-1", 23, 41, size_bin=size_bin)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.metadata == second.metadata
    assert first.image.shape == (192, 144, 3) and first.image.dtype == np.uint8
    assert first.mask.shape == (192, 144) and first.mask.dtype == bool
    assert size_bin_for_fraction(float(first.mask.mean())) == size_bin
    assert first.metadata["task_scope"] == TASK_SCOPE
    assert first.metadata["document"]["font"]["license"] == "OFL-1.1"
    assert first.metadata["compositing"]["page_render_calls"] == 1


def test_appearance_and_background_do_not_change_size_or_geometry():
    first = generate_scene_v2("a", 1, 5, size_bin="large", background_seed=10, photometric_seed=30)
    second = generate_scene_v2("a", 1, 5, size_bin="large", background_seed=11, photometric_seed=31)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.metadata["geometry"] == second.metadata["geometry"]
    assert first.metadata["sampling"] == second.metadata["sampling"]
    assert first.metadata["document"] == second.metadata["document"]
    assert not np.array_equal(first.image, second.image)


@pytest.mark.parametrize("color_mode", ("white", "colored"))
def test_unprinted_panels_are_explicit_zero_target_negatives(color_mode, monkeypatch):
    from flattener.synthetic import scene_v2

    monkeypatch.setattr(
        scene_v2, "generate_document", lambda *args: pytest.fail("must not print text")
    )
    sample = generate_scene_v2(
        "panel",
        1,
        11,
        has_page=False,
        negative_type="paper_panel",
        panel_color_mode=color_mode,
        background_seed=19,
        photometric=OFF,
    )
    background, _ = generate_background((192, 144), 19)
    assert not sample.mask.any()
    assert not np.array_equal(sample.image, background)
    assert sample.metadata["size_bin"] == "negative"
    assert sample.metadata["negative_type"] == "paper_panel"
    assert sample.metadata["document"] is None and sample.metadata["geometry"] is None
    assert sample.metadata["bbox"] is None and sample.metadata["foreground_fraction"] == 0
    panel = sample.metadata["panel"]
    assert panel["color_mode"] == color_mode
    assert panel["printed_content"] is False and panel["is_target"] is False
    assert "blank-paper" in sample.metadata["task_scope"]
    if color_mode == "white":
        assert len(set(panel["rgb"])) == 1 and min(panel["rgb"]) >= 220


@pytest.mark.parametrize("kind", BACKGROUND_TYPES)
def test_background_negatives_reuse_exact_v1_procedural_pixels(kind):
    sample = generate_scene_v2(
        "negative", 1, 5, has_page=False, background_type=kind, background_seed=9, photometric=OFF
    )
    expected, _ = generate_background((192, 144), 9, kind)
    np.testing.assert_array_equal(sample.image, expected)
    assert not sample.mask.any()
    assert sample.metadata["panel"] is None


def test_global_random_state_and_v1_output_are_unchanged():
    random.seed(63)
    np.random.seed(63)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    before = generate_scene("old-v1", 2, 3, image_size=(128, 128))
    generate_scene_v2("new-v2", 4, 5, image_size=(128, 128), size_bin="large")
    after = generate_scene("old-v1", 2, 3, image_size=(128, 128))
    np.testing.assert_array_equal(before.image, after.image)
    np.testing.assert_array_equal(before.mask, after.mask)
    assert before.metadata == after.metadata
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    np.testing.assert_array_equal(numpy_state[1], actual_numpy[1])
    assert numpy_state[0] == actual_numpy[0] and numpy_state[2:] == actual_numpy[2:]


def test_default_counts_are_exact_without_generating_default_data():
    from flattener.synthetic.scene_v2 import _counts, _negative_count

    config = SceneV2Config()
    assert config.seed == 20260923 and config.image_size == (192, 144)
    for documents, positive, negative, bin_count in ((96, 384, 192, 128), (24, 96, 48, 32)):
        assert documents * config.variants_per_document == positive
        assert _negative_count(positive, config.negative_fraction) == negative
        count = _counts(positive, negative)
        assert count["size_bins"] == dict.fromkeys(SIZE_BINS, bin_count)
        assert count["negative_types"] == {
            "background": negative // 2,
            "paper_panel": negative // 2,
        }


def test_dataset_group_and_stratum_counts_and_train_validation_loads(source_dataset):
    checked = validate_scene_v2_dataset(source_dataset)
    assert checked["sample_counts"] == dict.fromkeys(("train", "validation", "test"), 9)
    assert checked["page_counts"] == dict.fromkeys(("train", "validation", "test"), 6)
    assert checked["negative_counts"] == dict.fromkeys(("train", "validation", "test"), 3)
    for split in ("train", "validation", "test"):
        assert checked["size_bin_counts"][split] == dict.fromkeys(SIZE_BINS, 2)
        assert checked["negative_type_counts"][split] == {"background": 2, "paper_panel": 1}
    manifest = checked["manifest"]
    assert manifest["schema_version"] == "scene-0.2"
    assert manifest["task_scope"] == TASK_SCOPE
    assert manifest["external_images"] is False
    old_seeds = {
        derive_seed(20260922, "scene-document", f"scene-document-{index:06d}")
        for index in range(128)
    }
    assert not old_seeds.intersection(group["seed"] for group in manifest["groups"])
    for record in manifest["samples"]:
        if record["split"] == "test":
            continue  # Hold-out image/mask decoding is not needed for integrity checks.
        sample = load_scene_v2(source_dataset, record)
        assert bool(sample.mask.any()) == record["has_page"]
        if record["has_page"]:
            assert size_bin_for_fraction(float(sample.mask.mean())) == record["size_bin"]


def test_validator_never_decodes_images_or_gt(source_dataset, monkeypatch):
    monkeypatch.setattr(Image, "open", lambda *args: pytest.fail("must not decode held-out labels"))
    validate_scene_v2_dataset(source_dataset)


def test_dataset_is_byte_reproducible(source_dataset, tmp_path):
    other = tmp_path / "repeat"
    generate_scene_v2_dataset(other, SMALL)
    files = sorted(
        path.relative_to(source_dataset) for path in source_dataset.rglob("*") if path.is_file()
    )
    assert files == sorted(path.relative_to(other) for path in other.rglob("*") if path.is_file())
    for relative in files:
        assert (source_dataset / relative).read_bytes() == (other / relative).read_bytes()


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"seed": True},
        {"train_documents": 0},
        {"test_documents": 1.5},
        {"variants_per_document": 0},
        {"image_size": (127, 128)},
        {"image_size": (128, 513)},
        {"image_size": [128, 128]},
        {"negative_fraction": -0.1},
        {"negative_fraction": 0.76},
        {"negative_fraction": float("nan")},
        {"negative_fraction": True},
        {"train_documents": 10000, "variants_per_document": 10000},
    ],
)
def test_invalid_config_rejected_before_output_creation(tmp_path, change):
    output = tmp_path / "bad"
    with pytest.raises(FlattenerError):
        generate_scene_v2_dataset(output, replace(SMALL, **change))
    assert not output.exists()


def test_no_overwrite(dataset):
    original = (dataset / "manifest.json").read_bytes()
    with pytest.raises(FlattenerError) as raised:
        generate_scene_v2_dataset(dataset, SMALL)
    assert raised.value.reason_code == "dataset_create_failed"
    assert (dataset / "manifest.json").read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(schema_version="scene-0.1"),
        lambda m: m.update(complete=False),
        lambda m: m.update(task_scope="arbitrary paper"),
        lambda m: m["samples"].pop(),
        lambda m: m["samples"].append(copy.deepcopy(m["samples"][0])),
        lambda m: m["groups"][1].update(seed=m["groups"][0]["seed"]),
        lambda m: m["samples"][0].update(split="validation"),
        lambda m: m["counts"]["train"]["size_bins"].update(small=999),
    ],
)
def test_manifest_corruption_is_rejected(dataset, change):
    manifest = manifest_at(dataset)
    change(manifest)
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError):
        validate_scene_v2_dataset(dataset)


def test_wrong_size_bin_with_updated_hash_is_rejected(dataset):
    manifest = manifest_at(dataset)
    record = manifest["samples"][0]
    record["size_bin"] = "large"
    mutate_metadata(dataset, record, lambda metadata: metadata.update(size_bin="large"))
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="size bin"):
        validate_scene_v2_dataset(dataset)
    with pytest.raises(FlattenerError, match="size bin"):
        load_scene_v2(dataset, record)


@pytest.mark.parametrize("fraction", (10**400, -(10**400), float("inf"), float("nan")))
def test_untrusted_numeric_metadata_never_leaks_overflow(dataset, fraction):
    manifest = manifest_at(dataset)
    record = manifest["samples"][0]
    mutate_metadata(dataset, record, lambda metadata: metadata.update(foreground_fraction=fraction))
    write_manifest(dataset, manifest)
    for action in (
        lambda: validate_scene_v2_dataset(dataset),
        lambda: load_scene_v2(dataset, record),
    ):
        with pytest.raises(FlattenerError):
            action()


def test_blank_panel_cannot_claim_printed_target(dataset):
    manifest = manifest_at(dataset)
    record = next(
        record for record in manifest["samples"] if record["negative_type"] == "paper_panel"
    )
    mutate_metadata(dataset, record, lambda metadata: metadata["panel"].update(is_target=True))
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="blank panels"):
        validate_scene_v2_dataset(dataset)


def test_same_document_variants_must_share_original_hash(dataset):
    manifest = manifest_at(dataset)
    record = manifest["samples"][1]
    mutate_metadata(
        dataset, record, lambda metadata: metadata["document"].update(source_sha256="0" * 64)
    )
    write_manifest(dataset, manifest)
    with pytest.raises(FlattenerError, match="same authored document"):
        validate_scene_v2_dataset(dataset)


@pytest.mark.parametrize("name", ("image.png", "mask.png", "metadata.json"))
def test_tampered_artifact_rejected(dataset, name):
    record = manifest_at(dataset)["samples"][0]
    with (dataset / record["paths"][name]).open("ab") as stream:
        stream.write(b"changed")
    for action in (
        lambda: validate_scene_v2_dataset(dataset),
        lambda: load_scene_v2(dataset, record),
    ):
        with pytest.raises(FlattenerError) as raised:
            action()
        assert raised.value.reason_code == "artifact_hash_mismatch"


@pytest.mark.parametrize("symlink", (False, True))
def test_artifact_path_escape_rejected(dataset, tmp_path, symlink):
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
    for action in (
        lambda: validate_scene_v2_dataset(dataset),
        lambda: load_scene_v2(dataset, record),
    ):
        with pytest.raises(FlattenerError, match="escapes"):
            action()


def test_mask_not_reconstructed_from_false_metadata(dataset):
    manifest = manifest_at(dataset)
    record = manifest["samples"][0]
    path = dataset / record["paths"]["mask.png"]
    wrong = np.zeros((128, 128), dtype=np.uint8)
    wrong[20:40, 20:40] = 255
    with Image.fromarray(wrong) as encoded:
        encoded.save(path)
    record["sha256"]["mask.png"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def mutate(metadata):
        metadata.update(bbox=[20, 20, 40, 40], foreground_fraction=400 / (128 * 128))
        metadata["sampling"]["foreground_fraction"] = metadata["foreground_fraction"]
        metadata["geometry"]["photo_page_fraction"] = metadata["foreground_fraction"]

    mutate_metadata(dataset, record, mutate)
    with pytest.raises(FlattenerError, match="analytic geometry"):
        load_scene_v2(dataset, record)
