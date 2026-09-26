"""Seeds, photometry and reproducible paired samples from the synthetic pipeline."""

import hashlib

import numpy as np
import pytest

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


@pytest.mark.parametrize("seed", [-1, True, "1"])
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
    "config",
    [
        PhotometricConfig(enabled=1),
        PhotometricConfig(brightness_min=0),
        PhotometricConfig(brightness_max=float("inf")),
        PhotometricConfig(max_shadow=1),
    ],
)
def test_invalid_photometry_is_rejected(config):
    with pytest.raises(FlattenerError) as caught:
        apply_photometry(np.zeros((2, 2, 3), dtype=np.uint8), 1, config)
    assert caught.value.reason_code == "invalid_config"
