"""Independent contracts for actual-area strata and printed-page negatives."""

from types import SimpleNamespace

import numpy as np
import pytest

from flattener import FlattenerError
from flattener.synthetic.geometry import flat_to_photo
from flattener.synthetic.pipeline import PhotometricConfig
from flattener.synthetic.scene_v2 import (
    SIZE_BINS,
    TASK_SCOPE,
    generate_scene_v2,
    sample_stratified_geometry,
    size_bin_for_fraction,
)
from flattener.synthetic.scenes import generate_background

OFF = PhotometricConfig(enabled=False)


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


@pytest.mark.parametrize("fraction", (10**400, float("nan")))
def test_extreme_integer_fraction_is_structured_input_error(fraction):
    with pytest.raises(FlattenerError):
        size_bin_for_fraction(fraction)


@pytest.mark.parametrize("value", (10**400, float("nan")))
def test_invalid_photometry_never_leaks_numeric_conversion_error(value, monkeypatch):
    from flattener.synthetic import scene_v2

    monkeypatch.setattr(
        scene_v2, "generate_background", lambda *args: pytest.fail("validate first")
    )
    with pytest.raises(FlattenerError):
        generate_scene_v2("numeric", 1, 2, photometric=PhotometricConfig(brightness_min=value))


@pytest.mark.parametrize(
    "family,size_bin",
    [("perspective", "small"), ("gentle", "medium"), ("strong", "large"), ("local", "small")],
)
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


@pytest.mark.parametrize("kind", ("gradient", "shapes"))
def test_background_negatives_reuse_exact_v1_procedural_pixels(kind):
    sample = generate_scene_v2(
        "negative", 1, 5, has_page=False, background_type=kind, background_seed=9, photometric=OFF
    )
    expected, _ = generate_background((192, 144), 9, kind)
    np.testing.assert_array_equal(sample.image, expected)
    assert not sample.mask.any()
    assert sample.metadata["panel"] is None
