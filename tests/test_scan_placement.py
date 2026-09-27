"""View placement and quarter turns (``flattener.scan.placement``)."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from flattener import RenderConfig
from flattener.contracts import Status
from flattener.dewarp.coords import center_positions, edge_positions
from flattener.dewarp.model import DewarpModelConfig
from flattener.dewarp.predict import (
    DESKEW_MAX_DEG,
    Dewarper,
    _flattener_values,
    canvas_input,
    canvas_size,
    canvas_to_source,
    crop_box,
    dense_backward_map,
    estimate_page_angle,
    outline_points,
    predict_grid,
    predict_input,
    rectify,
    rerun_grid,
)
from flattener.regions.geometry import PageRegion, region_map_to_source, virtual_page_box
from flattener.renderer import render
from flattener.scan.placement import (
    CONVENTIONS,
    Placement,
    compose,
    dense_map,
    flip_grid,
    quarter_turn_grid,
    split_angle,
    turned_size,
    view_canvas_size,
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
        ((120, 90), PageRegion(10.25, 7.5, 80.125, 101.0), (7, 9)),
        ((1000, 700), virtual_page_box(PageRegion(33.3, 71.9, 612.7, 905.2), 0.68), (45, 31)),
        ((37, 29), PageRegion(0, 0, 29, 37), (37, 29)),
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


# ------------------------------------------------------ 2. node turns = image turns


def test_turning_the_grid_nodes_turns_the_rendered_image():
    source = np.random.default_rng(1).integers(0, 256, (90, 120, 3), dtype=np.uint8)
    # A curved page reaching past the right edge of the source, so part of it is filled.
    placement = Placement.from_canvas((40, 10, 134, 80), 8.0, (90, 120))
    grid = compose(_bent_view_grid(), placement)
    assert grid[..., 0].max() > 1.0
    size = (41, 57)
    config = RenderConfig(on_error="raise")
    for inset in (None, 0.1):
        base = render(source, dense_map(grid, size, inset=inset), config=config)
        assert not base.valid_mask.all()
        for k in (1, 2, 3):
            turned = quarter_turn_grid(grid, k)
            assert turned.shape == (*turned_size(GRID, k), 2)
            result = render(source, dense_map(turned, turned_size(size, k), inset=inset))
            np.testing.assert_array_equal(result.image, np.rot90(base.image, k))
            np.testing.assert_array_equal(result.valid_mask, np.rot90(base.valid_mask, k))
        flipped = render(source, dense_map(flip_grid(grid), size, inset=inset))
        np.testing.assert_array_equal(flipped.image, base.image[::-1, ::-1])
    np.testing.assert_array_equal(flip_grid(grid), quarter_turn_grid(grid, 2))
    np.testing.assert_array_equal(quarter_turn_grid(grid, -1), quarter_turn_grid(grid, 3))
    np.testing.assert_array_equal(quarter_turn_grid(quarter_turn_grid(grid, 1), 3), grid)


# ------------------------------------------------------- 3. k * 90 + theta upright


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


@pytest.mark.parametrize("theta, quarter", [(13.0, 0), (13.0, 1), (-28.0, 2), (-28.0, 3)])
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
    outline = outline_points(truth, SOURCE)
    placement = Placement.from_outline(outline, estimate, SOURCE, margin=0.0)
    assert placement.quarter == quarter
    np.testing.assert_allclose(placement.view_size, PAGE, rtol=0, atol=1e-9)
    # The model input view shows the page upright.
    _assert_upright(render(photo, placement.view_map(PAGE)).image, page)
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
    if angle % 180:
        mirror = compose(nodes, Placement.from_outline(outline, -estimate, SOURCE, margin=0.0))
        assert _mismatch(render(photo, dense_map(mirror, PAGE)).image, page) > 50


# ------------------------------------------------------ 4. nodes vs dense map


def _placements():
    outline = outline_points(_page_grid(200.0, np.array([120.0, 90.0])), SOURCE)
    return [
        Placement.whole(SOURCE),
        Placement.from_box(PageRegion(12.5, 20.25, 190.0, 150.5), SOURCE),
        Placement.from_canvas((40, 30, 180, 140), 17.0, SOURCE),
        Placement.from_canvas((40, 30, 180, 140), 17.0 + 270, SOURCE),
        Placement.from_outline(outline, 200.0, SOURCE),
    ]


def test_composing_the_nodes_equals_composing_the_dense_map():
    inset, border = 0.1, (0.03, 0.08)
    view_grid = _bent_view_grid()
    size = (53, 38)
    for base in _placements():
        for placement in (base, base.turned(1)):
            nodes = dense_map(compose(view_grid, placement), size, inset=inset, border=border)
            dense = compose(dense_map(view_grid, size, inset=inset, border=border), placement)
            assert np.abs(nodes - dense).max() <= 1e-6


def test_the_dense_map_is_the_dewarp_densification_and_extends_the_border():
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


# ------------------------------------------- crop_pass placement, rerun and rectify


class _FixedGrid(torch.nn.Module):
    """The same grid, in its own input frame, for every input."""

    def __init__(self, grid):
        super().__init__()
        self.grid = torch.from_numpy(np.moveaxis(grid, -1, 0).astype(np.float32))[None]

    def forward(self, x):
        return self.grid.expand(x.shape[0], -1, -1, -1), None


def _tilted_grid():
    x, y = np.moveaxis(_identity(*GRID), -1, 0)
    t = np.radians(9.0)
    gx = 0.7 * x * np.cos(t) - 0.75 * y * np.sin(t) + 0.03 * np.sin(np.pi * y) + 0.02
    gy = 0.7 * x * np.sin(t) + 0.75 * y * np.cos(t) - 0.01
    return np.stack([gx, gy], -1)


def _dewarper(convention):
    config = DewarpModelConfig(
        input_size=(96, 80), grid_size=GRID, num_filter=4, grid_convention=convention
    )
    return Dewarper(_FixedGrid(_tilted_grid()), config, torch.device("cpu"))


def _placed_rerun(dewarper, image, rerun, deskew):
    """``rerun_grid`` written with placements: each pass is placed in the source."""
    size = image.shape[:2]
    convention = dewarper.config.grid_convention
    whole = Placement.whole(size)
    grid = compose(view_values(predict_grid(dewarper, image), convention, whole), whole)
    for _ in range(rerun):
        box = crop_box(outline_points(grid, size), size)
        angle = 0.0
        if deskew:
            estimate, reason = estimate_page_angle(grid, size)
            if reason is None and abs(estimate) <= DESKEW_MAX_DEG:
                angle = estimate
        placement = Placement.from_canvas(box, angle, size)
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        if angle == 0.0:
            raw = predict_grid(dewarper, crop)
        else:
            raw = predict_input(dewarper, canvas_input(dewarper, crop, angle, placement.view_size))
        grid = compose(view_values(raw, convention, placement), placement)
    return grid


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_placed_passes_are_rerun_grid_and_rectify_byte_for_byte(convention):
    rerun, deskew = 1, True
    image = np.random.default_rng(4).integers(0, 256, (150, 200, 3), dtype=np.uint8)
    dewarper = _dewarper(convention)
    expected, records = rerun_grid(dewarper, image, rerun, deskew)
    assert all(record["rotated"] == deskew for record in records)
    placed = _placed_rerun(dewarper, image, rerun, deskew)
    np.testing.assert_array_equal(placed, expected)
    if convention == "flattener":
        size = (48, 64)
        config = RenderConfig(on_error="return")
        result = rectify(dewarper, image, size, rerun=rerun, deskew=deskew, render_config=config)
        assert result.status == Status.SUCCESS
        mine = render(image, dense_map(placed, size), config=config)
        assert mine.image.tobytes() == result.image.tobytes()


def test_crop_placements_are_canvas_to_source_and_quarter_turns_are_exact():
    box, size = (13, 22, 170, 131), (150, 200)
    for theta in (0.0, 12.5, -40.0, 45.0, -45.0):
        base = Placement.from_canvas(box, theta, size)
        matrix, offset = canvas_to_source(box, theta, canvas_size(box, theta), size)
        assert base.quarter == 0 and base.view_size == canvas_size(box, theta)
        assert base.matrix.tobytes() == matrix.tobytes()
        assert base.offset.tobytes() == offset.tobytes()
        if abs(theta) == 45.0:  # theta + 90 k splits at the other end of [-45, 45]
            continue
        for k in (1, 2, 3):
            angle = theta + 90 * k
            turned = Placement.from_canvas(box, angle, size)
            assert (turned.theta_deg, turned.quarter) == (theta, k)
            assert (
                turned.view_size == turned_size(base.view_size, k) == view_canvas_size(box, angle)
            )
            assert turned.matrix.tobytes() == base.turned(k).matrix.tobytes()
            np.testing.assert_allclose(turned.view_size, canvas_size(box, angle), rtol=1e-12)
            matrix, offset = canvas_to_source(box, angle, canvas_size(box, angle), size)
            np.testing.assert_allclose(turned.matrix, matrix, rtol=0, atol=1e-15)
            np.testing.assert_array_equal(turned.offset, offset)
            # UVDoc values convert with the size of the turned view.
            raw = np.random.default_rng(k).uniform(-1, 1, (*GRID, 2))
            np.testing.assert_array_equal(
                view_values(raw, "uvdoc", turned), _flattener_values(raw, "uvdoc", turned.view_size)
            )
    # A quarter turn is exact: the view's top-right corner is the box's bottom-right.
    quarter = Placement.from_canvas(box, 90.0, size)
    corner = compose(np.array([1.0, -1.0]), quarter)
    np.testing.assert_allclose(corner, [2 * 170 / 200 - 1, 2 * 131 / 150 - 1], rtol=0, atol=1e-15)
    points = np.random.default_rng(5).uniform(-1.5, 1.5, (20, 2))
    for placement in _placements():
        for k in range(4):
            turned = placement.turned(k)
            np.testing.assert_allclose(
                turned.to_view(compose(points, turned)), points, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                compose(points, turned), points @ turned.matrix.T + turned.offset, atol=1e-12
            )
    assert [split_angle(a) for a in (0.0, 90.0, -90.0, 180.0, -180.0, 135.0, 100.0, -170.0)] == [
        (0.0, 0),
        (0.0, 1),
        (0.0, 3),
        (0.0, 2),
        (0.0, 2),
        (-45.0, 2),
        (10.0, 1),
        (10.0, 2),
    ]


def test_the_view_map_samples_the_model_input_of_a_turned_crop():
    crop = np.random.default_rng(6).integers(0, 256, (41, 57, 3), dtype=np.uint8)
    dewarper = _dewarper("flattener")
    source = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    for angle in (0.0, 17.0, -30.0):
        canvas = canvas_size((0, 0, 57, 41), angle)
        placement = Placement.from_canvas((0, 0, 57, 41), angle, (41, 57), canvas=canvas)
        view = torch.from_numpy(placement.view_map((96, 80)))[None]
        sampled = F.grid_sample(
            source, view, mode="bilinear", padding_mode="border", align_corners=False
        )
        assert torch.equal(sampled, canvas_input(dewarper, crop, angle, canvas))


def test_placements_round_trip_through_json():
    points = np.random.default_rng(7).uniform(-1.3, 1.3, (5, 6, 2))
    for base in _placements():
        placement = base.turned(3)
        record = json.loads(json.dumps(placement.record()))
        again = Placement.from_record(record)
        assert again.record() == placement.record()
        assert compose(points, again).tobytes() == compose(points, placement).tobytes()
        assert record["angle_deg"] == placement.angle_deg
