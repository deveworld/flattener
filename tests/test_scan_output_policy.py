"""Output size and aspect policy on synthetic page grids with known page shape."""

import math

import numpy as np
import pytest

from flattener import FlattenerError, Status
from flattener.scan import output_policy as op

SOURCE = (3000, 4000)  # (height, width) of the photo
GRID = (31, 45)


def to_values(points, source=SOURCE):
    """Source pixel-edge positions -> flattener values."""
    height, width = source
    return points / np.array([width, height]) * 2.0 - 1.0


def page_nodes(width, height, grid=GRID):
    """Page positions (x, y) of the grid nodes of a width x height page, origin at the center."""
    x, y = np.meshgrid(
        np.linspace(-width / 2, width / 2, grid[1]), np.linspace(-height / 2, height / 2, grid[0])
    )
    return np.stack((x, y), axis=-1)


def similarity_grid(width, height, *, angle=0.0, scale=1.0, center=(2000.0, 1500.0), grid=GRID):
    nodes = page_nodes(width, height, grid)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    turned = nodes @ np.array([[c, s], [-s, c]])  # x toward y: clockwise on screen
    return to_values(turned * scale + np.array(center))


def camera_grid(width, height, *, tilt_x=0.0, tilt_y=0.0, roll=0.0, distance=2.0, focal=3000.0):
    """A flat page photographed by a pinhole camera at the source center."""
    nodes = page_nodes(width, height)
    points = np.concatenate((nodes, np.zeros(nodes.shape[:2] + (1,))), axis=-1)
    ax, ay, az = (math.radians(v) for v in (tilt_x, tilt_y, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    camera = points @ (rz @ ry @ rx).T + np.array([0.0, 0.0, distance])
    pixels = focal * camera[..., :2] / camera[..., 2:] + np.array([SOURCE[1] / 2, SOURCE[0] / 2])
    return to_values(pixels)


def relative_error(value, truth):
    return abs(value / truth - 1.0)


@pytest.mark.parametrize("aspect", [1 / math.sqrt(2), 0.773, 1.0, 1.414, 0.52])
@pytest.mark.parametrize("angle", [0.0, 17.0, 90.0, -135.0])
def test_arc_aspect_is_exact_on_similarity_grids(aspect, angle):
    grid = similarity_grid(1200 * aspect, 1200, angle=angle, scale=1.3)
    assert relative_error(op.arc_aspect(grid, SOURCE), aspect) < 1e-9
    assert relative_error(op.arc_aspect(grid.astype(np.float32), SOURCE), aspect) < 1e-5


@pytest.mark.parametrize(
    ("tilt_x", "tilt_y", "roll"), [(6, 0, 0), (0, 6, 0), (-5, 4, 25), (4, -4, -60), (3, 3, 180)]
)
def test_arc_aspect_is_within_one_percent_under_mild_perspective(tilt_x, tilt_y, roll):
    aspect = 1 / math.sqrt(2)
    grid = camera_grid(aspect, 1.0, tilt_x=tilt_x, tilt_y=tilt_y, roll=roll)
    assert relative_error(op.arc_aspect(grid, SOURCE), aspect) < 0.01


@pytest.mark.parametrize(
    ("tilt_x", "tilt_y", "roll"),
    [(30, 0, 0), (0, -35, 0), (25, 20, 15), (-20, 30, -100), (6, 0, 0), (0, 0, 0)],
)
@pytest.mark.parametrize("aspect", [1 / math.sqrt(2), 8.5 / 11, 1.5])
def test_focal_length_removes_perspective(tilt_x, tilt_y, roll, aspect):
    grid = camera_grid(aspect, 1.0, tilt_x=tilt_x, tilt_y=tilt_y, roll=roll, focal=3000.0)
    estimate, method = op.estimate_aspect(grid, SOURCE, focal_px=3000.0)
    assert method == "perspective"
    assert relative_error(estimate, aspect) < 1e-6
    # A 10 % focal length error still keeps a 25 degree tilt within 1 %.
    if max(abs(tilt_x), abs(tilt_y)) <= 25:
        near = op.perspective_aspect(grid, SOURCE, 3300.0)
        assert relative_error(near, aspect) < 0.01


def test_strong_tilt_needs_the_focal_length():
    grid = camera_grid(1 / math.sqrt(2), 1.0, tilt_x=30)
    assert relative_error(op.arc_aspect(grid, SOURCE), 1 / math.sqrt(2)) > 0.05
    assert op.estimate_aspect(grid, SOURCE) == (op.arc_aspect(grid, SOURCE), "arc_length")


def test_focal_length_from_35mm_equivalent():
    assert op.focal_px_from_35mm(op.DIAGONAL_35MM, (3000, 4000)) == pytest.approx(5000.0)
    assert op.focal_px_from_35mm(26, (3000, 4000)) == pytest.approx(26 * 5000 / 43.2666, 1e-4)


def test_auto_width_samples_the_longest_row_at_source_density():
    grid = similarity_grid(1000, 1500, angle=30, scale=2.0)
    plan = op.plan_output(grid, SOURCE, aspect="estimate")
    assert (plan.height, plan.width) == (3000, 2000)
    assert plan.density_width == pytest.approx(2000.0)
    assert plan.aspect_source == "estimated" and plan.size_source == "auto"
    assert plan.density_min == pytest.approx(1.0) and plan.density_median == pytest.approx(1.0)
    keystone = camera_grid(0.7, 1.0, tilt_x=5, focal=3000.0)
    rows, _ = op.arc_lengths(keystone, SOURCE)
    plan = op.plan_output(keystone, SOURCE, aspect="estimate")
    assert plan.width == op.round_half_up(rows.max())
    assert plan.height == op.round_half_up(plan.width / plan.aspect)
    assert plan.density_min < 1.0


@pytest.mark.parametrize(
    ("aspect", "expected", "paper"),
    [
        (1 / math.sqrt(2) * 1.012, 1 / math.sqrt(2), "iso"),
        (1 / math.sqrt(2) * 0.99, 1 / math.sqrt(2), "iso"),
        (0.6795, 128 / 188, "4x6"),
        (0.6760, 152 / 225, "singukpan"),
        (0.773 * 1.01, 8.5 / 11, "letter"),
        (math.sqrt(2) * 1.01, math.sqrt(2), "iso-landscape"),
    ],
)
def test_auto_snaps_within_tolerance(aspect, expected, paper):
    grid = similarity_grid(1500 * aspect, 1500)
    plan = op.plan_output(grid, SOURCE)
    assert plan.aspect_source == "snapped" and plan.paper == paper
    assert plan.aspect == pytest.approx(expected, abs=1e-12)
    assert plan.aspect_estimate == pytest.approx(aspect, 1e-9)
    assert plan.height == op.round_half_up(plan.width / expected)


@pytest.mark.parametrize("aspect", [1 / math.sqrt(2) * 1.02, 0.60, 0.74, 1.0])
def test_auto_keeps_the_estimate_outside_tolerance(aspect):
    plan = op.plan_output(similarity_grid(1500 * aspect, 1500), SOURCE)
    assert plan.aspect_source == "estimated" and plan.paper is None
    assert plan.aspect == pytest.approx(aspect, 1e-9)


def test_aspect_options():
    grid = similarity_grid(1500 * 0.74, 1500)
    assert op.resolve_aspect("snap", 0.74)[1:] == ("snapped", "letter")
    assert op.resolve_aspect("estimate", 0.708) == (0.708, "estimated", None)
    assert op.resolve_aspect("A4", 0.74) == (pytest.approx(1 / math.sqrt(2)), "paper", "iso")
    assert op.resolve_aspect("a4", 1.3) == (pytest.approx(math.sqrt(2)), "paper", "iso-landscape")
    assert op.resolve_aspect(0.5, 0.74) == (0.5, "user", None)
    plan = op.plan_output(grid, SOURCE, aspect="letter")
    assert (plan.aspect_source, plan.paper) == ("paper", "letter")
    for bad in ("square", -1.0, math.nan, True, None):
        with pytest.raises(FlattenerError) as caught:
            op.plan_output(grid, SOURCE, aspect=bad)
        assert caught.value.reason_code == "invalid_output_option"


def test_explicit_size_and_long_side():
    grid = similarity_grid(700, 1000, scale=2.0)
    fixed = op.plan_output(grid, SOURCE, output=(488, 712))
    assert (fixed.height, fixed.width, fixed.aspect_source) == (488, 712, "output_size")
    assert fixed.size_source == "output_size" and fixed.scale == 1.0
    portrait = op.plan_output(grid, SOURCE, output=3000, aspect="estimate")
    assert (portrait.height, portrait.width) == (3000, 2100)
    landscape = op.plan_output(similarity_grid(1000, 700), SOURCE, output=1000, aspect=1.5)
    assert (landscape.height, landscape.width, landscape.size_source) == (667, 1000, "long_side")
    for bad in (0, "big", (0, 10), 2.5):
        with pytest.raises(FlattenerError):
            op.plan_output(grid, SOURCE, output=bad)


@pytest.mark.parametrize(
    ("width", "height", "limited_by"),
    [(10000, 14142, "max_pixels"), (12000, 1000, "max_side"), (9000, 2250, "max_pixels")],
)
def test_limits_shrink_uniformly(width, height, limited_by):
    big = (20000, 20000)
    nodes = page_nodes(width, height) + np.array([10000.0, 10000.0])
    plan = op.plan_output(to_values(nodes, big), big, aspect="estimate")
    assert max(plan.size) <= op.MAX_SIDE and plan.height * plan.width <= op.MAX_PIXELS
    assert plan.limited_by == limited_by and plan.scale < 1
    rounding = 0.5 / min(plan.size) + 0.5 / max(plan.size)
    assert relative_error(plan.width / plan.height, width / height) <= rounding
    loose = op.plan_output(to_values(nodes, big), big, aspect="estimate", max_pixels=10**9)
    assert max(loose.size) <= op.MAX_SIDE


def test_inset_follows_the_visible_page():
    grid = similarity_grid(1000, 1500, scale=2.0)
    plan = op.plan_output(grid, SOURCE, aspect="estimate", inset=(0.1, 0.2))
    assert plan.width == 1800
    assert plan.height == op.round_half_up(1800 / (1000 / 1500 * 0.9 / 0.8))
    assert plan.density_min == pytest.approx(1.0, rel=2e-3)


def test_degenerate_and_malformed_grids():
    flat = np.zeros((5, 5, 2))
    with pytest.raises(FlattenerError) as caught:
        op.plan_output(flat, SOURCE)
    assert (caught.value.status, caught.value.reason_code) == (
        Status.INVALID_TRANSFORM,
        "degenerate_grid",
    )
    nan = similarity_grid(700, 1000)
    nan[3, 4, 0] = np.nan
    with pytest.raises(FlattenerError) as caught:
        op.plan_output(nan, SOURCE)
    assert caught.value.reason_code == "degenerate_grid"
    with pytest.raises(FlattenerError) as caught:
        op.plan_output(np.zeros((1, 5, 2)), SOURCE)
    assert (caught.value.status, caught.value.reason_code) == (Status.INPUT_ERROR, "invalid_grid")
    line = similarity_grid(700, 1000)
    line[..., 1] = 0.0  # every node on one horizontal line
    assert op.perspective_aspect(line, SOURCE, 3000.0) is None


def test_output_plan_round_trip():
    plan = op.plan_output(camera_grid(0.7, 1.0, tilt_x=20), SOURCE, focal_px=3000.0)
    assert plan.aspect_method == "perspective"
    assert op.OutputPlan.from_dict(plan.to_dict()) == plan
    assert plan.to_dict()["aspect_source"] in op.ASPECT_SOURCES
