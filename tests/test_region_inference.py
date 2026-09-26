"""Independent original-pixel integration checks for opt-in page-region inference."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from flattener.contracts import FlattenerError, RenderConfig, Status
from flattener.learning.inference import load_predictor, unwarp
from flattener.learning.model import CoordinatePredictor, ModelConfig, coarse_identity
from flattener.regions import (
    PageRegion,
    RegionConfig,
    crop_page,
    estimate_page_region,
    prepare_page_view,
    unwarp_page,
)


@pytest.fixture
def image():
    yy, xx = np.mgrid[:48, :72]
    # High-frequency channels make sampling from a tiny resized preview visible.
    return np.stack((xx * 3, yy * 5, ((xx + yy) % 2) * 255), axis=-1).astype(np.uint8)


@pytest.fixture
def predictor(tmp_path):
    config = ModelConfig(input_size=(16, 12), grid_size=(5, 3), channels=(4, 6, 8), hidden=16)
    with torch.random.fork_rng(devices=[]):
        model = CoordinatePredictor(config)
    checkpoint = tmp_path / "zero.pt"
    torch.save(
        {
            "schema_version": "learning-0.1",
            "config": {"model_config": asdict(config)},
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    return load_predictor(checkpoint)


class _Segmenter:
    model_id = "test-segmenter"
    model_sha256 = hashlib.sha256(b"test-segmenter").hexdigest()
    load_seconds = 0.125

    def __init__(self, probability=None):
        self.probability = np.zeros((12, 18), np.float32) if probability is None else probability
        if probability is None:
            self.probability[4:8, 6:12] = 0.9
        self.seen_shapes = []

    def predict_probability(self, image):
        self.seen_shapes.append(image.shape)
        return self.probability.copy()


def test_manual_region_crop_reads_exact_original_rgb_pixels_at_non_square_source(image):
    original = image.copy()
    result = crop_page(image, (24, 40), region=PageRegion(16, 12, 56, 36))
    assert result.status == Status.SUCCESS
    assert not result.used_fallback
    np.testing.assert_array_equal(result.image, image[12:36, 16:56])
    assert result.image.shape == (24, 40, 3)
    assert result.valid_mask.all()
    assert result.source_mask.all()
    assert result.target_mask.all()
    assert not np.shares_memory(result.image, image)
    np.testing.assert_array_equal(image, original)
    metadata = result.metadata
    assert metadata["processing_status"] == "success"
    assert metadata["quality_status"] == "not_evaluated"
    assert metadata["page_method"] == "region_only"
    assert metadata["page_region"]["method"] == "manual_region"
    assert metadata["source_size"] == [48, 72]
    assert metadata["requested_output_size"] == [24, 40]
    assert metadata["actual_output_size"] == [24, 40]
    assert metadata["model_id"] is None
    assert metadata["page_region"]["boundary_expansion"] == 1
    assert metadata["page_region"]["final_estimated_page_coverage"] == 1
    assert metadata["page_region"]["coverage_is_ground_truth"] is False
    assert metadata["failure_stage"] is None
    assert metadata["timing_scope"] == "page_region_pipeline"
    for name in ("prepare_input", "estimate_region", "prepare_region", "page_coverage", "render"):
        assert metadata["region_timings"][name] >= 0
    assert metadata["timings"]["total"] >= 0
    json.dumps(metadata, allow_nan=False)


def test_prepare_view_exact_crop_without_unnecessary_resampling(image):
    original = image.copy()
    preview, virtual, metadata = prepare_page_view(
        image,
        PageRegion(16, 12, 56, 36),
        (24, 40),
        config=RegionConfig(canonical_occupancy=1),
    )
    np.testing.assert_array_equal(preview, image[12:36, 16:56])
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(preview, image)
    assert virtual == PageRegion(16, 12, 56, 36)
    assert metadata["preview_size"] == [24, 40]
    assert metadata["working_size"] == [48, 72]
    assert metadata["preview_source_support_fraction"] == 1


def test_prepare_view_outside_source_is_gray_without_clipping_virtual_geometry():
    source = np.full((8, 12, 3), 255, np.uint8)
    preview, virtual, metadata = prepare_page_view(
        source,
        PageRegion(0, 0, 12, 8),
        (8, 12),
        config=RegionConfig(canonical_occupancy=0.5),
    )
    assert preview.dtype == np.uint8
    assert preview.shape == (8, 12, 3)
    assert virtual == PageRegion(-6, -4, 18, 12)
    expected = np.full((8, 12, 3), 64, np.uint8)
    expected[2:6, 3:9] = 255
    np.testing.assert_array_equal(preview, expected)
    assert metadata["outside_source_fill_rgb"] == [64, 64, 64]
    assert metadata["preview_source_support_fraction"] == 0.25
    assert metadata["virtual_source_box"] == {"left": -6, "top": -4, "right": 18, "bottom": 12}


def test_zero_predictor_composes_virtual_box_and_renders_original_not_tiny_preview(
    image, predictor, monkeypatch
):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint reloaded"))
    result = unwarp_page(
        image,
        predictor,
        (32, 48),
        region=PageRegion(24, 16, 48, 32),
        region_config=RegionConfig(canonical_occupancy=0.5),
    )
    assert result.status == Status.SUCCESS
    np.testing.assert_array_equal(result.image, image[8:40, 12:60])
    metadata = result.metadata
    assert metadata["processing_status"] == "success"
    assert metadata["quality_status"] == "not_evaluated"
    assert metadata["page_method"] == "region_conditioned_model"
    assert metadata["source_size"] == [48, 72]
    assert metadata["actual_output_size"] == metadata["requested_output_size"] == [32, 48]
    assert metadata["model_id"] == predictor.model_id
    assert metadata["model_sha256"] == predictor.model_sha256
    assert metadata["model_input_size"] == [16, 12]
    assert metadata["coarse_grid_size"] == [5, 3]
    assert metadata["canonical_view"]["virtual_source_box"] == {
        "left": 12,
        "top": 8,
        "right": 60,
        "bottom": 40,
    }
    assert metadata["canonical_view"]["preview_size"] == [16, 12]
    assert "virtual source box" in metadata["input_resize"]
    assert "RGB(64,64,64)" in metadata["input_resize"]
    assert metadata["model_preview_resize"] == "full extent; independent axes; no crop or padding"
    assert (
        metadata["region_preprocessing_version"]
        == metadata["canonical_view"]["preprocessing_version"]
    )
    assert metadata["region_timings"]["prepare_input"] >= 0
    assert metadata["page_region"]["boundary_expansion"] == 1
    assert metadata["model_load_seconds"] >= 0
    assert metadata["timings"]["load_model"] == 0
    for name in ("prepare_input", "predict", "process_coordinates", "render", "total"):
        assert metadata["timings"][name] >= 0
    json.dumps(metadata, allow_nan=False)


def test_learned_region_crop_and_estimate_report_segmenter_provenance(image):
    segmenter = _Segmenter()
    config = RegionConfig(padding_fraction=0)
    estimate = estimate_page_region(image, segmenter, config=config)
    assert estimate.region == PageRegion(24, 16, 48, 32)
    assert estimate.mask.sum() == 24
    assert estimate.metadata["source_size"] == [48, 72]
    assert estimate.metadata["segmenter_id"] == segmenter.model_id
    assert estimate.metadata["segmenter_sha256"] == segmenter.model_sha256
    assert estimate.metadata["segmenter_load_seconds"] == 0.125
    assert estimate.metadata["score_is_calibrated_confidence"] is False
    result = crop_page(image, (16, 24), segmenter=segmenter, region_config=config)
    assert result.status == Status.SUCCESS
    np.testing.assert_array_equal(result.image, image[16:32, 24:48])
    assert result.metadata["page_region"]["method"] == "learned_page_region"
    assert result.metadata["page_region"]["segmenter_id"] == segmenter.model_id
    assert segmenter.seen_shapes == [(48, 72, 3), (48, 72, 3)]


def test_learned_and_manual_equal_regions_have_identical_zero_model_output(image, predictor):
    config = RegionConfig(padding_fraction=0, canonical_occupancy=0.5)
    automatic = unwarp_page(
        image, predictor, (32, 48), segmenter=_Segmenter(), region_config=config
    )
    manual = unwarp_page(
        image, predictor, (32, 48), region=PageRegion(24, 16, 48, 32), region_config=config
    )
    assert automatic.status == manual.status == Status.SUCCESS
    np.testing.assert_array_equal(automatic.image, manual.image)
    np.testing.assert_array_equal(automatic.valid_mask, manual.valid_mask)


def test_coverage_describes_selected_component_not_all_foreground_or_rendering_validity(image):
    probability = np.zeros((20, 20), np.float32)
    probability[5:15, 5:15] = 0.9
    probability[18, 1:18] = 0.9
    result = crop_page(
        image,
        (24, 36),
        segmenter=_Segmenter(probability),
        region_config=RegionConfig(padding_fraction=0),
    )
    assert result.status == Status.SUCCESS
    region = result.metadata["page_region"]
    assert region["selected_pixels"] == 100
    assert region["foreground_pixels"] == 117
    assert region["component_sizes"] == [100, 17]
    assert region["dominance"] == pytest.approx(100 / 117)
    assert region["final_estimated_page_coverage"] == 1
    assert region["coverage_is_ground_truth"] is False
    # All three masks mark valid sampling positions in the output grid.
    assert (
        result.valid_mask.shape == result.source_mask.shape == result.target_mask.shape == (24, 36)
    )
    assert result.valid_mask.all() and result.source_mask.all() and result.target_mask.all()
    np.testing.assert_array_equal(result.image, image[12:36, 18:54])


def test_rejected_learned_region_retains_both_model_identities_and_original_source_size(
    image, predictor
):
    segmenter = _Segmenter(np.zeros((12, 18), np.float32))
    result = unwarp_page(
        image,
        predictor,
        (32, 48),
        segmenter=segmenter,
        config=RenderConfig(on_error="return"),
    )
    assert result.status == Status.MODEL_ERROR
    assert result.metadata["model_id"] == predictor.model_id
    assert result.metadata["model_sha256"] == predictor.model_sha256
    region = result.metadata["page_region"]
    assert region["source_size"] == [48, 72]
    assert region["segmenter_id"] == segmenter.model_id
    assert region["segmenter_sha256"] == segmenter.model_sha256
    assert region["segmenter_load_seconds"] == segmenter.load_seconds
    json.dumps(result.metadata, allow_nan=False)


def test_estimate_rejection_itself_retains_segmenter_provenance(image):
    segmenter = _Segmenter(np.zeros((12, 18), np.float32))
    with pytest.raises(FlattenerError) as caught:
        estimate_page_region(image, segmenter)
    assert caught.value.metadata["source_size"] == [48, 72]
    assert caught.value.metadata["segmenter_id"] == segmenter.model_id
    assert caught.value.metadata["segmenter_sha256"] == segmenter.model_sha256


@pytest.mark.parametrize("method,supplied", [("crop", "neither"), ("unwarp", "both")])
def test_exactly_one_region_source_is_required(image, predictor, method, supplied):
    kwargs = (
        {"region": PageRegion(24, 16, 48, 32), "segmenter": _Segmenter()}
        if supplied == "both"
        else {}
    )
    if method == "crop":
        result = crop_page(image, (32, 48), config=RenderConfig(on_error="return"), **kwargs)
    else:
        result = unwarp_page(
            image, predictor, (32, 48), config=RenderConfig(on_error="return"), **kwargs
        )
    assert result.status == Status.INPUT_ERROR
    assert result.reason_code == "region_source_required"
    assert result.image is None
    assert result.metadata["failure_stage"] == "prepare_input"


@pytest.mark.parametrize("bad_predictor,policy", [(None, "raise"), (object(), "original")])
def test_invalid_predictor_follows_every_failure_policy_including_none(
    image, policy, bad_predictor
):
    def call():
        return unwarp_page(
            image,
            bad_predictor,
            (32, 48),
            region=PageRegion(24, 16, 48, 32),
            config=RenderConfig(on_error=policy),
        )

    if policy == "raise":
        with pytest.raises(FlattenerError) as caught:
            call()
        assert caught.value.status == Status.INPUT_ERROR
        assert caught.value.reason_code == "invalid_predictor"
        assert caught.value.metadata["source_size"] == [48, 72]
        assert caught.value.metadata["requested_output_size"] == [32, 48]
        return
    result = call()
    assert result.status == Status.INPUT_ERROR
    assert result.reason_code == "invalid_predictor"
    assert result.used_fallback == (policy == "original")
    if policy == "original":
        np.testing.assert_array_equal(result.image, image)
        assert not np.shares_memory(result.image, image)
    else:
        assert result.image is None


def test_detection_rejection_is_never_reported_as_success_or_a_hidden_crop(image):
    segmenter = _Segmenter(np.zeros((12, 18), np.float32))
    result = crop_page(
        image, (16, 24), segmenter=segmenter, config=RenderConfig(on_error="original")
    )
    assert result.status == Status.MODEL_ERROR
    assert result.reason_code == "page_not_found"
    assert result.metadata["processing_status"] == "model_error"
    assert result.metadata["quality_status"] == "not_evaluated"
    assert result.used_fallback
    assert result.metadata["requested_output_size"] == [16, 24]
    assert result.metadata["failure_stage"] == "estimate_region"
    assert result.metadata["page_region"]["foreground_pixels"] == 0
    assert result.metadata["actual_output_size"] == [48, 72]
    np.testing.assert_array_equal(result.image, image)
    assert not np.shares_memory(result.image, image)


def test_original_fallback_preserves_preoriented_readonly_negative_stride_rgb(image):
    oriented = np.rot90(image)
    oriented.setflags(write=False)
    original = oriented.copy()
    result = crop_page(
        oriented,
        (16, 24),
        segmenter=_Segmenter(np.zeros((12, 18), np.float32)),
        config=RenderConfig(on_error="original"),
    )
    assert result.status == Status.MODEL_ERROR
    assert result.used_fallback
    np.testing.assert_array_equal(result.image, original)
    np.testing.assert_array_equal(oriented, original)
    assert result.metadata["source_size"] == result.metadata["actual_output_size"] == [72, 48]
    assert not np.shares_memory(result.image, oriented)


def test_invalid_image_is_not_returned_as_a_trusted_original_fallback(image):
    result = crop_page(
        image.astype(np.float32),
        (24, 40),
        region=PageRegion(16, 12, 56, 36),
        config=RenderConfig(on_error="original"),
    )
    assert result.status == Status.INPUT_ERROR
    assert result.image is None
    assert not result.used_fallback
    assert result.metadata["processing_status"] == "input_error"
    assert result.metadata["quality_status"] == "not_evaluated"


@pytest.mark.parametrize(
    "method,config", [("crop", {}), ("unwarp", RenderConfig(on_error="silent"))]
)
def test_invalid_failure_policy_always_raises_a_structured_input_error(
    image, predictor, method, config
):
    kwargs = {"region": PageRegion(24, 16, 48, 32), "config": config}
    with pytest.raises(FlattenerError) as caught:
        if method == "crop":
            crop_page(image, (32, 48), **kwargs)
        else:
            unwarp_page(image, predictor, (32, 48), **kwargs)
    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.reason_code == "invalid_config"
    assert caught.value.metadata["processing_status"] == "input_error"
    assert caught.value.metadata["quality_status"] == "not_evaluated"


def test_out_of_bounds_virtual_prediction_is_rejected_by_existing_renderer_gate(image, predictor):
    result = unwarp_page(
        image,
        predictor,
        (32, 48),
        region=PageRegion(0, 0, 72, 48),
        region_config=RegionConfig(canonical_occupancy=0.5),
        config=RenderConfig(on_error="original"),
    )
    assert result.status == Status.INVALID_TRANSFORM
    assert result.metadata["failure_stage"] == "process_coordinates"
    assert result.metadata["model_id"] == predictor.model_id
    assert result.used_fallback
    np.testing.assert_array_equal(result.image, image)


def _set_shrinking_prediction(predictor, monkeypatch, scale):
    residual = (coarse_identity(predictor.model_config.grid_size) * (scale - 1)).permute(2, 0, 1)
    monkeypatch.setattr(predictor._model, "forward", lambda _: residual[None].clone())


def test_roi_guard_widens_an_undersized_footprint_only_within_configured_budget(
    image, predictor, monkeypatch
):
    _set_shrinking_prediction(predictor, monkeypatch, 0.9)
    result = unwarp_page(
        image,
        predictor,
        (24, 40),
        region=PageRegion(16, 12, 56, 36),
        region_config=RegionConfig(canonical_occupancy=1),
    )
    assert result.status == Status.SUCCESS
    coverage = result.metadata["page_region"]
    assert coverage["initial_estimated_page_coverage"] < 0.98
    assert coverage["final_estimated_page_coverage"] >= 0.98
    assert 1 < coverage["boundary_expansion"] <= 1.2
    assert coverage["coverage_is_ground_truth"] is False
    assert result.metadata["validation"]["fold_check"] == "passed"


def test_roi_guard_rejects_uncorrectable_crop_with_structured_diagnostics(
    image, predictor, monkeypatch
):
    _set_shrinking_prediction(predictor, monkeypatch, 0.2)
    result = unwarp_page(
        image,
        predictor,
        (24, 40),
        region=PageRegion(16, 12, 56, 36),
        region_config=RegionConfig(canonical_occupancy=1),
        config=RenderConfig(on_error="return"),
    )
    assert result.status == Status.MODEL_ERROR
    assert result.reason_code == "page_coverage_insufficient"
    assert result.metadata["failure_stage"] == "page_coverage"
    assert result.metadata["page_region"]["final_estimated_page_coverage"] < 0.98
    assert not result.used_fallback
    assert result.image is None


@pytest.mark.parametrize("bad_size", [(1, 24), (32, True)])
def test_output_size_rejected_before_segmenter_or_model_runs(image, predictor, bad_size):
    class UncalledSegmenter:
        def predict_probability(self, image):
            pytest.fail("segmenter should not run for invalid output dimensions")

    result = unwarp_page(
        image,
        predictor,
        bad_size,
        segmenter=UncalledSegmenter(),
        config=RenderConfig(on_error="return"),
    )
    assert result.status == Status.INPUT_ERROR
    assert result.metadata["failure_stage"] == "prepare_input"


def test_region_path_preserves_original_full_frame_unwarp_and_global_torch_state(image, predictor):
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    image.setflags(write=False)
    page = unwarp_page(
        image,
        predictor,
        (32, 48),
        region=PageRegion(24, 16, 48, 32),
        region_config=RegionConfig(canonical_occupancy=0.5),
    )
    full = unwarp(image, predictor, image.shape[:2])
    assert page.status == full.status == Status.SUCCESS
    np.testing.assert_array_equal(full.image, image)
    assert full.metadata["timing_scope"] == "unwarp_loaded_model"
    assert "page_region" not in full.metadata
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


@pytest.mark.parametrize("error", [MemoryError("allocation failed"), KeyboardInterrupt()])
def test_resource_and_interrupt_failures_are_not_hidden_as_original_fallback(image, error):
    class FailingSegmenter:
        def predict_probability(self, image):
            raise error

    with pytest.raises(type(error)):
        crop_page(
            image, (16, 24), segmenter=FailingSegmenter(), config=RenderConfig(on_error="original")
        )
