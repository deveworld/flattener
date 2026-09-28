"""``regions.unwarp_page`` with a ``flattener.dewarp`` ``Dewarper``."""

from dataclasses import replace

import torch
from scan_fakes import (
    CONFIG,
    RELEASE_INFO,
    Oracle,
    Page,
    oracle_dewarper,
    ramp_scene,
)

from flattener.contracts import RenderConfig, Status
from flattener.dewarp.predict import Dewarper
from flattener.regions import PageRegion, unwarp_page
from flattener.renderer import render
from flattener.scan.placement import dense_map
from flattener.scan.views import crop_placement, run_view

SIZE = (240, 320)
PAGE = Page((170, 125), (150, 190), 5.0)
RETURN = RenderConfig(on_error="return")


def _region():
    corners = PAGE.corners()
    low, high = corners.min(axis=0), corners.max(axis=0)
    return PageRegion(float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def test_a_dewarper_predicts_once_on_the_region_box_and_stamps_its_weights():
    dewarper = oracle_dewarper(PAGE, SIZE)
    image = ramp_scene(SIZE, PAGE)
    result = unwarp_page(image, dewarper, (190, 150), region=_region(), config=RETURN)
    assert result.status == Status.SUCCESS and result.image.shape == (190, 150, 3)
    assert dewarper.model.calls == 1
    metadata = result.metadata
    assert metadata["page_method"] == "region_dewarp"
    assert metadata["dewarp"]["box"] == [80, 15, 260, 235]  # the region + 4 % per side
    assert metadata["page_region"]["boundary_expansion"] == 1.0
    assert metadata["page_region"]["coverage_enforced"] is False  # the caller's rectangle
    assert 0.8 < metadata["page_region"]["final_estimated_page_coverage"] < 0.95
    assert metadata["quality_status"] == "not_evaluated"
    # Weights the ship gate never checked count as research.
    assert metadata["model_sha256"] == "fake-dewarp"
    assert metadata["model_id"] == "dewarp-checkpoint-fake-dewarp"
    assert metadata["research"] is True and metadata["ship_gate"] is None
    # The same pass placed by hand and rendered once gives the same image.
    box = tuple(metadata["dewarp"]["box"])
    again = oracle_dewarper(PAGE, SIZE)
    grid, _ = run_view(again, image, crop_placement(box, 0.0, SIZE), crop_box=box)
    expected = render(image, dense_map(grid, (190, 150)), config=RETURN)
    assert result.image.tobytes() == expected.image.tobytes()

    # The result carries the ship gate verdict of the weights.
    for verdict in (
        {"shippable": False, "research": True, "reasons": ["external_init"]},
        {"shippable": True, "research": False, "reasons": []},
    ):
        stamped = oracle_dewarper(PAGE, SIZE, {**RELEASE_INFO, "ship_gate": verdict})
        metadata = unwarp_page(image, stamped, (190, 150), region=_region(), config=RETURN).metadata
        assert metadata["ship_gate"] == verdict
        assert metadata["research"] is verdict["research"]
        assert metadata["shippable"] is verdict["shippable"]


def test_uvdoc_weights_are_densified_with_their_own_nodes():
    model_config = replace(CONFIG, grid_convention="uvdoc")
    dewarper = Dewarper(Oracle(PAGE, SIZE), model_config, torch.device("cpu"), dict(RELEASE_INFO))
    image = ramp_scene(SIZE, PAGE)
    result = unwarp_page(image, dewarper, (190, 150), region=_region(), config=RETURN)
    box = tuple(result.metadata["dewarp"]["box"])
    grid, _ = run_view(dewarper, image, crop_placement(box, 0.0, SIZE), crop_box=box)
    expected = render(image, dense_map(grid, (190, 150), nodes="uvdoc"), config=RETURN)
    edge_aligned = render(image, dense_map(grid, (190, 150)), config=RETURN)
    assert result.image.tobytes() == expected.image.tobytes()
    assert result.image.tobytes() != edge_aligned.image.tobytes()
