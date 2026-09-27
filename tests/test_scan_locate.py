"""``scan.locate``: the page gate's decision, the consensus rule and page conditions."""

import numpy as np
import pytest
from scan_fakes import FakeGate, Page

from flattener.contracts import FlattenerError, Status
from flattener.regions.geometry import PageRegion
from flattener.scan import locate
from flattener.scan.locate import (
    alternative_box,
    expand_box,
    locate_page,
    manual_quad,
    manual_region,
)

SIZE = (600, 800)  # (height, width)
SHAPE = (96, 72)  # gate grid


@pytest.fixture(autouse=True)
def small_pages(monkeypatch):
    monkeypatch.setitem(locate.THRESHOLDS, "min_page_long_side_px", 100)


def _gate(pages, probability=0.97, low=0.02):
    gate = FakeGate(pages, SIZE, probability=probability, low=low)
    gate.config.input_size = SHAPE
    return gate.predict_probability(None)


def test_a_confident_gate_gives_the_mask_box_and_angle():
    page = Page((420, 290), (260, 340), 9.0)
    calls = []

    def pass0():
        calls.append(1)
        raise AssertionError("pass 0 is not needed")

    found = locate_page(SIZE, probability=_gate([page]), pass0=pass0)
    assert found.failure is None and found.method == "gate" and calls == []
    assert found.angle_deg == pytest.approx(9.0, abs=2.0)
    left, top, right, bottom = found.box
    corners = page.corners()
    assert left <= corners[:, 0].min() and right >= corners[:, 0].max()
    assert top <= corners[:, 1].min() and bottom >= corners[:, 1].max()
    assert found.reasons == [] and found.metrics["confident"]


def test_a_page_filling_the_frame_takes_the_whole_image():
    page = Page((400, 300), (820, 640))
    found = locate_page(SIZE, probability=_gate([page]))
    assert found.method == "fills_frame" and found.box == (0, 0, 800, 600)
    assert [item.code for item in found.reasons] == ["page_fills_frame"]
    assert found.angle_deg == 0.0


def _outline(page, grid=(9, 7)):
    return page.grid(grid, SIZE)


def test_weak_scores_go_through_the_consensus_rule():
    page = Page((420, 290), (260, 340), 9.0)
    weak = _gate([page], probability=0.8)
    # The outline agrees: its box and angle.
    found = locate_page(SIZE, probability=weak, pass0=lambda: _outline(page))
    assert found.method == "outline" and found.metrics["consensus_coverage"] >= 0.95
    assert found.angle_deg == pytest.approx(9.0, abs=0.01)
    assert found.pass0 is not None
    # The outline swallows background: the mask box and the mask's axis.
    wide = Page((400, 300), (700, 520), 0.0)
    found = locate_page(SIZE, probability=weak, pass0=lambda: _outline(wide))
    assert found.method == "mask" and found.metrics["consensus_precision"] < 0.85
    assert found.angle_deg == pytest.approx(9.0, abs=3.0)
    # The outline cuts the page: the union of both boxes.
    small = Page((420, 290), (180, 240), 9.0)
    found = locate_page(SIZE, probability=weak, pass0=lambda: _outline(small))
    assert found.method == "union" and found.metrics["consensus_coverage"] < 0.95
    mask_box = expand_box(locate.mask_bounds(found.mask, SIZE), SIZE)
    assert found.box[0] <= mask_box[0] and found.box[2] >= mask_box[2]
    # The other localizer for candidate C3.
    other = alternative_box(found, SIZE, lambda: _outline(small))
    assert other is not None and other[2] == "gate"


def test_gate_rejections_end_the_scan_with_candidates():
    nothing = np.full(SHAPE, 0.01, np.float32)
    found = locate_page(SIZE, probability=nothing)
    assert found.box is None and found.failure.code == "page_not_found"
    assert found.failure.status == Status.MODEL_ERROR
    left = Page((200, 300), (300, 420))
    right = Page((600, 300), (300, 420))
    found = locate_page(SIZE, probability=_gate([left, right]))
    assert found.failure.code == "two_page_candidate"
    assert len(found.failure.detail["boxes"]) == 2 and len(found.failure.detail["gutter"]) == 2
    stacked = [Page((400, 150), (500, 200)), Page((250, 450), (200, 150))]
    found = locate_page(SIZE, probability=_gate(stacked))
    assert found.failure.code == "ambiguous_page_region"
    tiny = Page((400, 300), (60, 40))
    found = locate_page(SIZE, probability=_gate([tiny]))
    assert found.failure.code == "page_too_small"


def test_small_pages_by_long_side_and_neighbors(monkeypatch):
    monkeypatch.setitem(locate.THRESHOLDS, "min_page_long_side_px", 600)
    page = Page((400, 300), (300, 400))
    found = locate_page(SIZE, probability=_gate([page]))
    assert found.failure.code == "page_too_small"
    # The same page in a draft-decoded photo of twice the size is large enough.
    found = locate_page(SIZE, probability=_gate([page]), full_size=(1200, 1600))
    assert found.failure is None
    monkeypatch.setitem(locate.THRESHOLDS, "min_page_long_side_px", 100)
    neighbor = Page((770, 300), (60, 200))
    found = locate_page(SIZE, probability=_gate([page, neighbor]))
    assert found.failure is None
    assert [item.code for item in found.reasons] == ["neighbor_page_partial"]


def test_manual_region_and_quad():
    found = manual_region(PageRegion(100, 50, 500, 450), SIZE)
    assert found.box == (84, 34, 516, 466) and found.provenance == "manual"
    assert manual_region((0, 0, 800, 600), SIZE).box == (0, 0, 800, 600)
    page = Page((400, 300), (240, 320), 90.0 + 6.0)  # text up points right
    found = manual_quad(page.corners(), SIZE, SHAPE)
    assert found.quarter == 1 and found.angle_deg == pytest.approx(6.0)
    assert found.mask is not None and found.mask.shape == SHAPE
    assert found.quad.shape == (4, 2)
    for bad in (
        (1, 2, 3),
        [[0, 0], [10, 0], [0, 10], [10, 10]],  # crossing
        [[900, 900], [950, 900], [950, 950], [900, 950]],  # outside
    ):
        with pytest.raises(FlattenerError):
            manual_quad(bad, SIZE)
    with pytest.raises(FlattenerError):
        manual_region((10, 10, 900, 20), SIZE)
