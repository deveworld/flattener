"""Page detection and dewarping of a detected region.

Model predictions map back to the original image before rendering.
"""

from dataclasses import asdict
from time import perf_counter

import numpy as np

from ..contracts import FlattenerError, RenderConfig, Status
from ..coordinates import identity_map
from ..renderer import _handle_failure, _json_value, _new_metadata, render
from ..validation import validate_config, validate_image, validate_size
from .geometry import (
    RegionEstimate,
    _config,
    detect_region,
    ensure_page_coverage,
    page_coverage,
    region_map_to_source,
    validate_region,
)


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


def _is_dewarper(predictor):
    from ..dewarp.predict import Dewarper  # the dewarp stack loads only when it is used

    return isinstance(predictor, Dewarper)


def _dewarper_identity(dewarper):
    """Describe the model used for this correction."""
    return {"model": dewarper.info.get("path"), "model_id": dewarper.info.get("model_id")}


def _dewarp_region_grid(image, dewarper, region, metadata):
    """Grid (source flattener values) of one ``Dewarper`` pass on the region's box."""
    from ..dewarp.predict import RERUN_MARGIN
    from ..scan.locate import expand_box
    from ..scan.views import crop_placement, run_view

    source_size = image.shape[:2]
    bounds = (region.left, region.top, region.right, region.bottom)
    box = expand_box(bounds, source_size, RERUN_MARGIN)
    placement = crop_placement(box, 0.0, source_size)
    grid, record = run_view(dewarper, image, placement, crop_box=box)
    metadata["dewarp"] = {
        "model": dewarper.info.get("path"),
        "grid_convention": dewarper.config.grid_convention,
        "grid_size": list(grid.shape[:2]),
        "box": list(box),
        "margin": RERUN_MARGIN,
        "pass": record,
    }
    if not np.isfinite(grid).all():
        raise FlattenerError(
            Status.INVALID_TRANSFORM, "non_finite_grid", "the dewarp pass gave non-finite values"
        )
    return grid


def _check_dewarp_coverage(grid, mask, region_config, *, enforce):
    """The estimated page coverage of a placed grid, as a check only (no widening). A
    manual rectangle is not a page mask (a skewed page leaves its corners uncovered), so
    its coverage is recorded and ``enforce`` is false."""
    coverage = page_coverage(grid.astype(np.float32), mask)
    check = {
        "initial_estimated_page_coverage": coverage,
        "required_estimated_page_coverage": region_config.required_coverage,
        "final_estimated_page_coverage": coverage,
        "boundary_expansion": 1.0,
        "coverage_grid_size": list(mask.shape),
        "coverage_is_ground_truth": False,
        "coverage_enforced": enforce,
    }
    if enforce and coverage < region_config.required_coverage:
        raise FlattenerError(
            Status.MODEL_ERROR,
            "page_coverage_insufficient",
            "estimated page extends beyond predicted footprint",
            metadata=check,
        )
    return check


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
        dewarper = predictor is not None and _is_dewarper(predictor)
        if (require_predictor or predictor is not None) and not dewarper:
            raise FlattenerError(Status.INPUT_ERROR, "invalid_predictor", "load a Dewarper first")
        if dewarper:
            metadata["page_method"] = "region_dewarp"
            metadata.update(_dewarper_identity(predictor))
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
            grid = _dewarp_region_grid(image, predictor, estimate.region, metadata)
        metadata["region_timings"][stage] = perf_counter() - stage_started

        stage, stage_started = "page_coverage", perf_counter()
        if dewarper:
            coverage = _check_dewarp_coverage(
                grid, estimate.mask, region_config, enforce=region is None
            )
            from ..scan.placement import dense_map

            grid = dense_map(grid, output_size, nodes=predictor.config.grid_convention)
        else:
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
    """Localize and crop one page with no learned correction.

    Per-image timings exclude file decoding and model loading.
    """
    return _pipeline(image, output_size, None, region, segmenter, region_config, config)


def unwarp_page(
    image, predictor, output_size, *, region=None, segmenter=None, region_config=None, config=None
):
    """Correct one page inside a region, guarded by estimated page coverage.

    Pass exactly one of ``region`` (a manual rectangle) or ``segmenter``, and an
    explicit ``output_size``. ``predictor`` is a ``flattener.dewarp`` ``Dewarper`` (see the
    module docstring). Metadata reports ``quality_status="not_evaluated"``.
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
