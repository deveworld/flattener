"""View placement and quarter turns (``flattener.scan.placement``)."""

import json

import numpy as np
import pytest

from flattener import RenderConfig
from flattener.dewarp.coords import center_positions, edge_positions
from flattener.dewarp.predict import (
    _flattener_values,
    crop_box,
    dense_backward_map,
    estimate_page_angle,
    outline_points,
)
from flattener.regions.geometry import PageRegion, region_map_to_source, virtual_page_box
from flattener.renderer import render
from flattener.scan.placement import (
    Placement,
    compose,
    dense_map,
    quarter_turn_grid,
    split_angle,
    turned_size,
    view_values,
)

SOURCE = (160, 210)  # (height, width)
PAGE = (48, 64)
GRID = (9, 7)


def _identity(height, width):
    """Edge-aligned node positions (x, y) of a (height, width) grid."""
    return np.stack(np.meshgrid(edge_positions(width), edge_positions(height)), -1)


def _rotation(angle_deg):
    theta = np.radians(angle_deg)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _bent_view_grid(shape=GRID, seed=3):
    """A curved page in view values, with a little noise."""
    x, y = np.moveaxis(_identity(*shape), -1, 0)
    bent = np.stack([0.8 * x + 0.05 * np.sin(np.pi * y), 0.85 * y + 0.04 * x**2 - 0.02], -1)
    return bent + np.random.default_rng(seed).normal(0, 0.004, bent.shape)


# ------------------------------------------------------------- 1. box = region path


def test_a_box_placement_is_region_map_to_source_bit_for_bit():
    rng = np.random.default_rng(0)
    cases = [
        ((1000, 700), virtual_page_box(PageRegion(33.3, 71.9, 612.7, 905.2), 0.68), (45, 31)),
        ((4000, 3000), PageRegion(123.456, 789.012, 2987.6, 3999.9), (89, 61)),
    ]
    for size, box, shape in cases:
        grid = rng.uniform(-1.2, 1.2, (*shape, 2)).astype(np.float32)
        expected = region_map_to_source(grid, box, size)
        for given in (box, (box.left, box.top, box.right, box.bottom)):
            placement = Placement.from_box(given, size)
            assert compose(grid, placement).astype(np.float32).tobytes() == expected.tobytes()
            # The flattener form q = A q' + c holds too, to rounding.
            linear = grid.astype(np.float64) @ placement.matrix.T + placement.offset
            np.testing.assert_allclose(compose(grid, placement), linear, rtol=0, atol=1e-12)
        assert placement.view_size == (box.height, box.width)


# ------------------------------------------------------- 2. k * 90 + theta upright


def _page_image():
    """Red rises left to right, green top to bottom; the top-left quarter is blue."""
    rows, cols = np.mgrid[0 : PAGE[0], 0 : PAGE[1]]
    blue = np.where((rows < PAGE[0] // 2) & (cols < PAGE[1] // 2), 220, 40)
    return np.stack([30 + 3 * cols, 30 + 4 * rows, blue], -1).astype(np.uint8)


def _page_half():
    return np.array([PAGE[1], PAGE[0]], dtype=np.float64) / 2


def _photo(page, angle_deg, center):
    """``page`` rendered at scale 1 into a black photo, turned clockwise by ``angle_deg``
    about ``center`` (source pixel edges)."""
    rows, cols = np.mgrid[0 : SOURCE[0], 0 : SOURCE[1]] + 0.5
    points = np.stack([cols, rows], -1) - center
    values = points @ _rotation(angle_deg) / _page_half()  # R^T (p - center) / half
    inside = (np.abs(values) <= 1).all(axis=-1)
    config = RenderConfig(fill_rgb=(0, 0, 0))
    return render(page, values.astype(np.float32), target_mask=inside, config=config).image


def _page_grid(angle_deg, center, shape=GRID):
    """The true grid of the page in the photo, in source values."""
    pixels = center + (_identity(*shape) * _page_half()) @ _rotation(angle_deg).T
    return pixels / np.array([SOURCE[1], SOURCE[0]]) * 2 - 1


def _mismatch(image, page, border=3):
    """Largest red/green difference away from the page border."""
    inner = (slice(border, -border), slice(border, -border), slice(0, 2))
    return int(np.abs(image[inner].astype(int) - page[inner]).max())


def _assert_upright(image, page):
    assert _mismatch(image, page) <= 2
    blue = image[..., 2].astype(float)
    top, left = PAGE[0] // 2, PAGE[1] // 2
    assert blue[:top, :left].mean() > 180
    assert blue[top:].mean() < 90 and blue[:, left:].mean() < 90


@pytest.mark.parametrize("theta, quarter", [(13.0, 1), (-28.0, 3)])
def test_a_page_turned_by_quarters_and_an_angle_comes_back_upright(quarter, theta):
    angle = theta + 90 * quarter  # clockwise on screen
    center = np.array([101.3, 77.6])
    page = _page_image()
    photo = _photo(page, angle, center)
    truth = _page_grid(angle, center)
    # The outline angle carries the quarter turns; split_angle takes them apart.
    estimate, reason = estimate_page_angle(truth, SOURCE)
    assert reason is None
    split_theta, split_quarter = split_angle(estimate)
    assert split_quarter == quarter and split_theta == pytest.approx(theta, abs=1e-9)
    assert split_angle(135.0) == (-45.0, 2) and split_angle(-170.0) == (10.0, 2)
    outline = outline_points(truth, SOURCE)
    placement = Placement.from_outline(outline, estimate, SOURCE, margin=0.0)
    assert placement.quarter == quarter
    np.testing.assert_allclose(placement.view_size, PAGE, rtol=0, atol=1e-9)
    # The model input view shows the page upright; UVDoc values use its turned size.
    _assert_upright(render(photo, placement.view_map(PAGE)).image, page)
    raw = np.random.default_rng(quarter).uniform(-1, 1, (*GRID, 2))
    expected = _flattener_values(raw, "uvdoc", placement.view_size)
    np.testing.assert_array_equal(view_values(raw, "uvdoc", placement), expected)
    # An identity prediction on that view places the true page; the scan is upright.
    nodes = _identity(*GRID)
    grid = compose(nodes, placement)
    np.testing.assert_allclose(grid, truth, rtol=0, atol=1e-9)
    _assert_upright(render(photo, dense_map(grid, PAGE)).image, page)
    # A result from the view without the quarter turns is set upright by re-indexing its
    # nodes k quarter turns (the alternative to predicting again).
    unturned = placement.turned(-quarter)
    assert unturned.quarter == 0 and unturned.view_size == turned_size(placement.view_size, quarter)
    sideways = compose(_identity(*turned_size(GRID, quarter)), unturned)
    np.testing.assert_allclose(quarter_turn_grid(sideways, quarter), truth, rtol=0, atol=1e-9)
    # The crop canvas view also shows the page upright: level top and bottom, plumb sides.
    canvas = Placement.from_canvas(crop_box(outline, SOURCE, margin=0.0), estimate, SOURCE)
    seen = canvas.to_view(truth)
    for edge in (seen[0, :, 1], seen[-1, :, 1], seen[:, 0, 0], seen[:, -1, 0]):
        assert np.ptp(edge) < 1e-9
    assert (seen[0, 0] < 0).all() and (seen[-1, -1] > 0).all()
    np.testing.assert_allclose(compose(seen, canvas), truth, rtol=0, atol=1e-12)
    # The other sign, or a half turn more, does not come back upright.
    flipped = compose(nodes, placement.turned(2))
    assert _mismatch(render(photo, dense_map(flipped, PAGE)).image, page) > 50
    mirror = compose(nodes, Placement.from_outline(outline, -estimate, SOURCE, margin=0.0))
    assert _mismatch(render(photo, dense_map(mirror, PAGE)).image, page) > 50


# ------------------------------------------ 3. nodes vs dense map, JSON records


def _placements():
    outline = outline_points(_page_grid(200.0, np.array([120.0, 90.0])), SOURCE)
    return [
        Placement.whole(SOURCE),
        Placement.from_box(PageRegion(12.5, 20.25, 190.0, 150.5), SOURCE),
        Placement.from_canvas((40, 30, 180, 140), 17.0, SOURCE),
        Placement.from_canvas((40, 30, 180, 140), 17.0 + 270, SOURCE),
        Placement.from_outline(outline, 200.0, SOURCE),
    ]


def test_placed_nodes_densify_like_the_dewarp_path_and_round_trip():
    # Placing the nodes and then densifying equals densifying and then placing.
    inset, border = 0.1, (0.03, 0.08)
    view_grid = _bent_view_grid()
    size = (53, 38)
    for base in _placements():
        for placement in (base, base.turned(1)):
            nodes = dense_map(compose(view_grid, placement), size, inset=inset, border=border)
            dense = compose(dense_map(view_grid, size, inset=inset, border=border), placement)
            assert np.abs(nodes - dense).max() <= 1e-6
    # The dense map is the dewarp densification.
    grid = compose(_bent_view_grid(), _placements()[2])
    for inset in (None, 0.1, (0.02, 0.05), "syntheticdoc_half_cell"):
        expected = dense_backward_map(
            grid, "flattener", (53, 38), SOURCE, inset, flattener_values=True
        )
        assert dense_map(grid, (53, 38), inset=inset).tobytes() == expected.tobytes()
    # A border shows the page's affine surroundings: each side moves out by b / 2.
    flat = dense_map(_identity(*GRID), (20, 30), border=(0.1, 0.2))
    np.testing.assert_allclose(flat[0, :, 0], 1.1 * center_positions(30), rtol=0, atol=1e-6)
    np.testing.assert_allclose(flat[:, 0, 1], 1.2 * center_positions(20), rtol=0, atol=1e-6)
    # A quarter turn is exact: the view's top-right corner is the box's bottom-right.
    quarter = Placement.from_canvas((13, 22, 170, 131), 90.0, (150, 200))
    corner = compose(np.array([1.0, -1.0]), quarter)
    np.testing.assert_allclose(corner, [2 * 170 / 200 - 1, 2 * 131 / 150 - 1], rtol=0, atol=1e-15)
    # Placements round-trip through JSON.
    points = np.random.default_rng(7).uniform(-1.3, 1.3, (5, 6, 2))
    for base in _placements():
        placement = base.turned(3)
        record = json.loads(json.dumps(placement.record()))
        again = Placement.from_record(record)
        assert again.record() == placement.record()
        assert compose(points, again).tobytes() == compose(points, placement).tobytes()
        assert record["angle_deg"] == placement.angle_deg
