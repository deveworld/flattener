"""``regions.unwarp_page`` with a ``flattener.dewarp`` ``Dewarper``."""

import numpy as np
from scan_fakes import FakeGate, Page, oracle_dewarper, ramp_scene, shrink

from flattener.contracts import RenderConfig, Status
from flattener.regions import PageRegion, unwarp_page
from flattener.renderer import render
from flattener.scan.placement import dense_map
from flattener.scan.views import crop_placement, run_view

SIZE = (240, 320)
PAGE = Page((170, 125), (150, 190), 5.0)


def _region():
    corners = PAGE.corners()
    low, high = corners.min(axis=0), corners.max(axis=0)
    return PageRegion(float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def test_a_dewarper_predicts_once_on_the_region_box_and_renders_the_original():
    dewarper = oracle_dewarper(PAGE, SIZE)
    image = ramp_scene(SIZE, PAGE)
    config = RenderConfig(on_error="return")
    result = unwarp_page(image, dewarper, (190, 150), region=_region(), config=config)
    assert result.status == Status.SUCCESS and result.image.shape == (190, 150, 3)
    assert dewarper.model.calls == 1
    metadata = result.metadata
    assert metadata["page_method"] == "region_dewarp"
    assert metadata["dewarp"]["box"] == [80, 15, 260, 235]  # the region + 4 % per side
    assert metadata["page_region"]["boundary_expansion"] == 1.0
    assert metadata["page_region"]["coverage_enforced"] is False  # the caller's rectangle
    assert 0.8 < metadata["page_region"]["final_estimated_page_coverage"] < 0.95
    assert metadata["quality_status"] == "not_evaluated"
    # The same pass placed by hand and rendered once gives the same image.
    box = tuple(metadata["dewarp"]["box"])
    again = oracle_dewarper(PAGE, SIZE)
    grid, _ = run_view(again, image, crop_placement(box, 0.0, SIZE), crop_box=box)
    expected = render(image, dense_map(grid, (190, 150)), config=config)
    assert result.image.tobytes() == expected.image.tobytes()


def test_a_segmenter_region_and_a_coverage_check_that_never_widens():
    dewarper = oracle_dewarper(PAGE, SIZE)
    image = ramp_scene(SIZE, PAGE)
    gate = FakeGate([PAGE], SIZE)
    result = unwarp_page(image, dewarper, (190, 150), segmenter=gate)
    assert result.status == Status.SUCCESS
    assert result.metadata["page_region"]["coverage_enforced"] is True
    assert result.metadata["page_region"]["segmenter_id"] == "fake-gate"
    dewarper.model.always = shrink(0.8)
    config = RenderConfig(on_error="original")
    result = unwarp_page(image, dewarper, (190, 150), segmenter=gate, config=config)
    assert result.status == Status.MODEL_ERROR
    assert result.reason_code == "page_coverage_insufficient"
    assert result.used_fallback and result.image.tobytes() == image.tobytes()
    assert result.metadata["page_region"]["boundary_expansion"] == 1.0
    assert result.metadata["failure_stage"] == "page_coverage"


def test_a_failing_dewarper_map_keeps_the_renderer_status():
    dewarper = oracle_dewarper(PAGE, SIZE)
    dewarper.model.always = lambda grid: np.full_like(grid, np.nan)
    image = ramp_scene(SIZE, PAGE)
    result = unwarp_page(
        image, dewarper, (190, 150), region=_region(), config=RenderConfig(on_error="return")
    )
    assert result.status == Status.INVALID_TRANSFORM and result.reason_code == "non_finite_grid"
    assert result.image is None
