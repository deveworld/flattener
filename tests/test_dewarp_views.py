"""Evaluation views: quarter turns and the small page keep exact targets."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from flattener.dewarp.coords import edge_positions
from flattener.dewarp.data import RawSample, ViewConfig, evaluation_view

INPUT = (48, 40)
MARGIN = (0.05, 0.05)


def _sample():
    """Colour ramps along x (red) and y (green) and a slanted page inside the image.

    Bilinear resampling reproduces a linear ramp, so a view sampled at its target
    positions gives back the colours of the page nodes in any correct view.
    """
    height, width = 90, 70
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.stack([xx * 255 / (width - 1), yy * 255 / (height - 1), 0 * xx + 128], -1)
    u, v = np.meshgrid(edge_positions(4), edge_positions(5))
    grid = np.stack([0.55 * u + 0.1 * v - 0.05, 0.6 * v + 0.05 * u + 0.1], -1)
    return RawSample(
        image.round().astype(np.uint8), grid.astype(np.float32), np.ones((5, 4), bool), None
    )


def _colours(image, target):
    grid = torch.from_numpy(target)[None].double()
    sampled = F.grid_sample(image[None].double(), grid, mode="bilinear", align_corners=False)
    return sampled[0].permute(1, 2, 0).numpy()


def test_quarter_turns_turn_the_targets_with_the_image():
    sample = _sample()
    upright_image, upright = evaluation_view(sample, INPUT, ViewConfig("a", margin=MARGIN), "k")
    expected = _colours(upright_image, upright)
    for rotation in (90, 180, 270):
        view = ViewConfig("a", margin=MARGIN, rotation=rotation)
        image, target = evaluation_view(sample, INPUT, view, "k")
        assert image.shape == (3, *INPUT)
        np.testing.assert_allclose(_colours(image, target), expected, atol=3)
    image, turned = evaluation_view(sample, INPUT, ViewConfig("a", margin=MARGIN, rotation=90), "k")
    x, y = upright[..., 0], upright[..., 1]
    np.testing.assert_allclose(turned, np.stack([y, -x], -1), atol=1e-6)  # counter-clockwise
    assert image[0, 0].float().mean() > image[0, -1].float().mean()  # red now grows upward
    frame, _ = evaluation_view(sample, INPUT, ViewConfig("f", "full_frame", rotation=270), "k")
    assert frame.shape == (3, *INPUT)


def test_the_small_page_covers_its_fraction_of_a_textured_canvas():
    sample = _sample()
    page_image, page = evaluation_view(sample, INPUT, ViewConfig("s", margin=MARGIN), "k")
    view = ViewConfig("s", "small_page", MARGIN)
    image, target = evaluation_view(sample, INPUT, view, "k")
    assert image.shape == (3, *INPUT)
    np.testing.assert_allclose(_colours(image, target), _colours(page_image, page), atol=4)
    extent = (target[..., 0].max() - target[..., 0].min()) * (
        target[..., 1].max() - target[..., 1].min()
    )
    assert 0.05 < extent / 4 <= 0.25  # the page nodes lie inside 10-25% of the canvas
    again, _ = evaluation_view(sample, INPUT, view, "k")
    assert torch.equal(again, image)  # seeded by the view name and the sample key
    other, _ = evaluation_view(sample, INPUT, view, "other")
    assert not torch.equal(other, image)
    outside = np.ones(INPUT, bool)
    pixels = ((target + 1) / 2 * [INPUT[1], INPUT[0]]).astype(int)
    outside[pixels[..., 1].min() : pixels[..., 1].max(), pixels[..., 0].min() :] = False
    assert image[:, torch.from_numpy(outside)].float().std() > 1  # canvas grain and shading


def test_view_config_checks_and_keeps_old_signatures():
    assert ViewConfig("x").to_dict() == {"name": "x", "crop": "page", "margin": [0.025, 0.025]}
    for view in (
        ViewConfig("r", "page", (0.02, 0.1), rotation=270),
        ViewConfig("s", "small_page", (0.02, 0.1), page_fraction=(0.2, 0.3)),
    ):
        assert ViewConfig.from_dict(view.to_dict()) == view
    for bad in (
        {"rotation": 45},
        {"rotation": True},
        {"page_fraction": (0.2, 0.3)},  # page crop
        {"crop": "small_page", "page_fraction": (0.3, 0.2)},
        {"crop": "small_page", "page_fraction": (0.0, 0.2)},
    ):
        with pytest.raises(ValueError):
            ViewConfig("v", **bad)
