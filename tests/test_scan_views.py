"""``scan.views``: model inputs of placed views and one pass placed in the source."""

import numpy as np
import torch
from scan_fakes import Page, oracle_dewarper, ramp_scene
from torch.nn import functional as F

from flattener.scan.io import SourceImage
from flattener.scan.placement import Placement, compose
from flattener.scan.records import SourceInfo
from flattener.scan.views import (
    crop_placement,
    on_source,
    run_view,
    sample_view,
    source_footprint,
)


def test_a_quarter_turned_view_shows_a_turned_page_upright():
    size = (240, 320)
    page = Page((160, 120), (120, 170), 90.0 + 4.0)
    image = ramp_scene(size, page)
    dewarper = oracle_dewarper(page, size)
    box = (60, 40, 262, 205)
    placement = crop_placement(box, 90.0 + 4.0, size)
    grid, record = run_view(dewarper, image, placement, crop_box=box)
    assert record["input"] == "view_map:original"
    assert dewarper.model.seen == [0]  # the model saw the page upright
    np.testing.assert_allclose(grid, page.grid(source_size=size), atol=2e-3)
    # The view is sampled once from its footprint, as from the whole image.
    source = torch.tensor(image).permute(2, 0, 1)[None].float() / 255
    whole = F.grid_sample(
        source,
        torch.from_numpy(placement.view_map((64, 48)))[None],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sample_view(image, placement, (64, 48))
    np.testing.assert_allclose(sampled, whole.numpy(), rtol=0, atol=5e-5)  # float32 coordinates
    left, top, right, bottom = source_footprint(placement)
    assert 0 <= left < right <= size[1] and 0 <= top < bottom <= size[0]
    # Without the quarter turn the page lies turned in its view and the nodes follow the view.
    upright = crop_placement(box, 4.0, size)
    other, _ = run_view(dewarper, image, upright, crop_box=box)
    assert dewarper.model.seen[-1] == 1
    assert np.abs(other - page.grid(source_size=size)).max() > 0.1


def test_a_box_view_keeps_its_place_on_a_reduced_level():
    box = Placement.from_box((10, 10, 110, 110), (200, 200))
    corners = np.array([[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(compose(corners, box), [[-0.9, -0.9], [0.1, 0.1]])
    level = on_source(box, (100, 100))
    assert level.edges == (5.0, 5.0, 55.0, 55.0)
    np.testing.assert_allclose(compose(corners, level), compose(corners, box), atol=1e-12)
    image = ramp_scene((200, 200))  # red and green encode the source position
    source = SourceImage(image, SourceInfo(size=(200, 200)))
    full = sample_view(image, box, (20, 20))
    reduced = sample_view(source.level(2), box, (20, 20))
    assert float(np.abs(full - reduced)[0, :2].max()) < 2 / 255  # the same place
