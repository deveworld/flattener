"""Opt-in page localization, region-only cropping, and guarded rectification."""

from dataclasses import asdict
from math import ceil
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..contracts import FlattenerError, RenderConfig, Status
from ..coordinates import identity_map
from ..learning.inference import LoadedPredictor
from ..renderer import _handle_failure, _json_value, _new_metadata, render
from ..validation import validate_config, validate_image, validate_size
from .geometry import (
    RegionEstimate,
    _config,
    detect_region,
    ensure_page_coverage,
    region_map_to_source,
    validate_region,
    virtual_page_box,
)

REGION_PREPROCESS_VERSION = "virtual-box-antialias-original-map-1.0"


def prepare_page_view(image, region, input_size, *, config=None):
    """Prepare a canonical RGB preview and its exact virtual source-edge box.

    First antialias the whole original to a working resolution sufficient for
    the virtual view; then sample that view. Its normalized coordinates always
    refer to the original physical extent, independently of working resolution.
    Original sampling for the final output happens only once, after composition.
    """
    config = _config(config)
    validate_image(image, RenderConfig())
    region = validate_region(region, image.shape[:2])
    input_size = validate_size(input_size, max_pixels=1024 * 1024, max_side=1024, name="input_size")
    if min(input_size) < 2:
        raise FlattenerError(
            Status.INPUT_ERROR, "invalid_region_input", "preview sides must be >=2"
        )
    virtual = virtual_page_box(region, config.canonical_occupancy)
    height, width = image.shape[:2]
    work_size = (
        min(height, max(1, ceil(height * input_size[0] / virtual.height))),
        min(width, max(1, ceil(width * input_size[1] / virtual.width))),
    )
    grid = region_map_to_source(identity_map(input_size), virtual, (height, width))
    source_support = np.all((grid >= -1) & (grid <= 1), axis=-1)
    with torch.inference_mode():
        source = torch.from_numpy(image.copy(order="C")).permute(2, 0, 1)[None].float().div_(255)
        if work_size != (height, width):
            source = F.interpolate(
                source, size=work_size, mode="bilinear", align_corners=False, antialias=True
            )
        sampled = F.grid_sample(
            source,
            torch.from_numpy(grid)[None],
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled.clamp_(0, 1)
        values = sampled[0].permute(1, 2, 0).numpy()
        preview = np.floor(values * 255 + 0.5).astype(np.uint8)
    preview[~source_support] = (64, 64, 64)
    metadata = {
        "preprocessing_version": REGION_PREPROCESS_VERSION,
        "virtual_source_box": asdict(virtual),
        "working_size": list(work_size),
        "preview_size": list(input_size),
        "outside_source_fill_rgb": [64, 64, 64],
        "preview_source_support_fraction": float(source_support.mean()),
        "canonical_occupancy": config.canonical_occupancy,
        "final_sampling_source": "original oriented RGB, never the resized preview",
    }
    return preview, virtual, metadata


def estimate_page_region(image, segmenter, *, config=None):
    """Run a loaded segmenter; return a component estimate or an explicit error."""
    config = _config(config)
    validate_image(image, RenderConfig())
    if not callable(getattr(segmenter, "predict_probability", None)):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_segmenter", "load a page segmenter first")
    context = dict(
        method="learned_page_region",
        source_size=list(image.shape[:2]),
        segmenter_id=getattr(segmenter, "model_id", None),
        segmenter_sha256=getattr(segmenter, "model_sha256", None),
        segmenter_load_seconds=getattr(segmenter, "load_seconds", None),
    )
    try:
        probability = segmenter.predict_probability(image)
        estimate = detect_region(probability, image.shape[:2], config=config)
    except FlattenerError as error:
        error.metadata.update(context)
        raise
    estimate.metadata.update(context)
    return estimate


def _manual_estimate(region, source_size):
    region = validate_region(region, source_size)
    # Manual rectangles conservatively cover all intersecting low-resolution
    # cells. Very small boxes can be rejected by the subsequent coverage guard.
    height, width = source_size
    mask_h, mask_w = min(height, 128), min(width, 96)
    x0 = np.arange(mask_w) * width / mask_w
    x1 = (np.arange(mask_w) + 1) * width / mask_w
    y0 = np.arange(mask_h) * height / mask_h
    y1 = (np.arange(mask_h) + 1) * height / mask_h
    mask = ((y0 < region.bottom) & (y1 > region.top))[:, None] & (
        (x0 < region.right) & (x1 > region.left)
    )[None, :]
    return RegionEstimate(
        region,
        mask,
        {
            "method": "manual_region",
            "source_size": list(source_size),
            "region": asdict(region),
            "mask_semantics": "cells intersecting caller rectangle, not a page segmentation",
            "score_is_calibrated_confidence": False,
        },
    )


def _pipeline(
    image,
    output_size,
    predictor,
    region,
    segmenter,
    region_config,
    config,
    *,
    require_predictor=False,
):
    started = perf_counter()
    candidate = RenderConfig() if config is None else config
    if (
        not isinstance(candidate, RenderConfig)
        or not isinstance(candidate.on_error, str)
        or candidate.on_error
        not in (
            "raise",
            "return",
            "original",
        )
    ):
        raise FlattenerError(
            Status.INPUT_ERROR,
            "invalid_config",
            "invalid renderer failure policy",
            metadata={
                "processing_status": str(Status.INPUT_ERROR),
                "quality_status": "not_evaluated",
            },
        )
    metadata = _new_metadata(candidate, timing_scope="page_region_pipeline")
    metadata["quality_status"] = "not_evaluated"
    metadata["requested_output_size"] = _json_value(output_size)
    metadata["page_region"] = {}
    metadata["region_timings"] = {}
    metadata["page_method"] = "region_conditioned_model" if require_predictor else "region_only"
    trusted = None
    stage = "prepare_input"
    stage_started = started
    try:
        candidate = validate_config(candidate)
        validate_image(image, candidate)
        trusted = image
        metadata["source_size"] = list(image.shape[:2])
        region_config = _config(region_config)
        output_size = validate_size(
            output_size,
            max_pixels=candidate.max_output_pixels,
            max_side=candidate.max_side,
            name="output_size",
        )
        if min(output_size) < 2:
            raise FlattenerError(
                Status.INPUT_ERROR, "invalid_region_input", "output sides must be >=2"
            )
        if (region is None) == (segmenter is None):
            raise FlattenerError(
                Status.INPUT_ERROR,
                "region_source_required",
                "provide exactly one manual region or loaded segmenter",
            )
        if (require_predictor or predictor is not None) and not isinstance(
            predictor, LoadedPredictor
        ):
            raise FlattenerError(
                Status.INPUT_ERROR, "invalid_predictor", "load a coordinate predictor first"
            )
        if predictor is not None:
            predictor._metadata(metadata)
            metadata["model_preview_resize"] = metadata["input_resize"]
            metadata["input_resize"] = (
                "region-conditioned virtual source box; antialiased original; "
                "outside-source preview padding RGB(64,64,64)"
            )
            metadata["region_preprocessing_version"] = REGION_PREPROCESS_VERSION
        metadata["timings"][stage] = perf_counter() - stage_started
        metadata["region_timings"][stage] = metadata["timings"][stage]

        stage, stage_started = "estimate_region", perf_counter()
        estimate = (
            _manual_estimate(region, image.shape[:2])
            if region is not None
            else estimate_page_region(image, segmenter, config=region_config)
        )
        metadata["page_region"] = estimate.metadata
        metadata["region_timings"][stage] = perf_counter() - stage_started
        stage, stage_started = "prepare_region", perf_counter()
        if predictor is None:
            grid = region_map_to_source(
                identity_map(output_size, config=candidate), estimate.region, image.shape[:2]
            )
        else:
            if any(
                a < b for a, b in zip(output_size, predictor.model_config.grid_size, strict=True)
            ):
                raise FlattenerError(
                    Status.INPUT_ERROR,
                    "output_smaller_than_grid",
                    "output must not shrink the predicted grid",
                )
            preview, virtual, view_metadata = prepare_page_view(
                image, estimate.region, predictor.model_config.input_size, config=region_config
            )
            metadata["canonical_view"] = view_metadata
            virtual_grid = predictor._predict(preview, output_size, metadata)
            grid = region_map_to_source(virtual_grid, virtual, image.shape[:2])
        metadata["region_timings"][stage] = perf_counter() - stage_started

        stage, stage_started = "page_coverage", perf_counter()
        grid, coverage = ensure_page_coverage(grid, estimate.mask, config=region_config)
        metadata["page_region"].update(coverage)
        metadata["region_timings"][stage] = perf_counter() - stage_started
        stage, stage_started = "render", perf_counter()
        result = render(
            image, grid, config=RenderConfig(**{**asdict(candidate), "on_error": "return"})
        )
        for name in ("prepare_input", "process_coordinates", "render"):
            metadata["timings"][name] = (metadata["timings"][name] or 0) + (
                result.metadata["timings"][name] or 0
            )
        metadata["region_timings"][stage] = perf_counter() - stage_started
        metadata["validation"] = result.metadata["validation"]
        metadata["actual_output_size"] = result.metadata["actual_output_size"]
        metadata["failure_stage"] = result.metadata["failure_stage"]
        if result.status != Status.SUCCESS:
            raise FlattenerError(result.status, result.reason_code, result.message)
        metadata["timings"]["total"] = perf_counter() - started
        metadata["processing_status"] = str(result.status)
        result.metadata = metadata
        return result
    except FlattenerError as error:
        metadata["region_timings"][stage] = perf_counter() - stage_started
        if stage in ("estimate_region", "page_coverage"):
            metadata["page_region"].update(error.metadata)
            error.metadata = {}
        metadata["failure_stage"] = metadata["failure_stage"] or stage
        metadata["timings"]["total"] = perf_counter() - started
        error.metadata.update(processing_status=str(error.status), quality_status="not_evaluated")
        return _handle_failure(error, candidate, image=trusted, metadata=metadata)


def crop_page(image, output_size, *, region=None, segmenter=None, region_config=None, config=None):
    """Localize/crop one page without learned curvature correction.

    This is an explicit baseline, not a dewarping claim. Original-file decoding
    and model loading are outside the per-image timing scope.
    """
    return _pipeline(image, output_size, None, region, segmenter, region_config, config)


def unwarp_page(
    image, predictor, output_size, *, region=None, segmenter=None, region_config=None, config=None
):
    """Opt-in region-conditioned correction with an estimated-coverage guard.

    Existing full-frame ``learning.unwarp`` remains unchanged. Choose exactly
    one manual rectangle or loaded page segmenter. Output size is still explicit;
    neither page mask confidence nor true content preservation is guaranteed.
    Metadata separates processing status from unevaluated rectification quality.
    """
    return _pipeline(
        image,
        output_size,
        predictor,
        region,
        segmenter,
        region_config,
        config,
        require_predictor=True,
    )
