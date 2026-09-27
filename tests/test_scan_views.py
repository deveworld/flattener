"""``scan.views``: model inputs of placed views and one pass placed in the source."""

import numpy as np
import torch
from scan_fakes import Page, oracle_dewarper, ramp_scene
from torch.nn import functional as F

from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import Dewarper, crop_pass_box, rerun_grid
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

SIZE = (150, 200)


def _seeded(seed=0):
    torch.manual_seed(seed)
    config = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
    return Dewarper(UVDocNet(config).eval(), config, torch.device("cpu"), {"sha256": "s"})


def _image(seed=2, size=SIZE):
    return np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)


def test_the_whole_image_pass_is_rectifys_first_prediction():
    dewarper, image = _seeded(), _image()
    grid, record = run_view(dewarper, image, Placement.whole(SIZE))
    expected, _ = rerun_grid(dewarper, image, 0)
    assert grid.tobytes() == expected.tobytes()
    assert record["input"] == "model_input"


def test_a_crop_box_at_up_to_45_degrees_is_crop_pass_box():
    angle = 12.5
    dewarper, image = _seeded(1), _image(3)
    box = (13, 22, 170, 131)
    placement = crop_placement(box, angle, SIZE)
    grid, record = run_view(dewarper, image, placement, crop_box=box)
    expected, _ = crop_pass_box(dewarper, image, box, angle)
    assert grid.tobytes() == expected.tobytes()
    assert record["input"] == "crop_pass_box" and record["box"] == list(box)


def test_other_views_sample_the_source_once():
    dewarper, image = _seeded(2), _image(4)
    box = (13, 22, 170, 131)
    turned = crop_placement(box, 90.0 + 10.0, SIZE)
    grid, record = run_view(dewarper, image, turned, crop_box=box)
    assert record["input"] == "view_map:original" and grid.shape == (5, 4, 2)
    # The footprint crop gives the same input as sampling the whole image.
    source = torch.tensor(image).permute(2, 0, 1)[None].float() / 255
    whole = F.grid_sample(
        source,
        torch.from_numpy(turned.view_map((64, 48)))[None],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sample_view(image, turned, (64, 48))
    torch.testing.assert_close(sampled, whole, rtol=0, atol=5e-5)  # float32 coordinates
    left, top, right, bottom = source_footprint(turned)
    assert 0 <= left < right <= SIZE[1] and 0 <= top < bottom <= SIZE[0]


def test_a_quarter_turned_view_shows_a_turned_page_upright():
    size = (240, 320)
    page = Page((160, 120), (120, 170), 90.0 + 4.0)
    image = ramp_scene(size, page)
    dewarper = oracle_dewarper(page, size)
    box = (60, 40, 262, 205)
    placement = crop_placement(box, 90.0 + 4.0, size)
    grid, _ = run_view(dewarper, image, placement, crop_box=box)
    assert dewarper.model.seen == [0]  # the model saw the page upright
    np.testing.assert_allclose(grid, page.grid(source_size=size), atol=2e-3)
    upright = crop_placement(box, 4.0, size)
    other, _ = run_view(dewarper, image, upright, crop_box=box)
    assert dewarper.model.seen[-1] == 1  # turned in its view: nodes follow the view
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
    assert float((full - reduced)[0, :2].abs().max()) < 2 / 255  # the same place
