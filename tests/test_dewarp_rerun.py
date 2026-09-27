"""Crop-and-rerun inference (``flattener.dewarp.predict --rerun`` and ``--deskew``)."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError, RenderConfig
from flattener.benchmark import reporting
from flattener.benchmark.reporting import prediction_record, report_evidence
from flattener.contracts import Status
from flattener.dewarp import predict
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import center_positions, edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import (
    DESKEW_MAX_DEG,
    EXIT_FAILED,
    RERUN_MARGIN,
    Dewarper,
    canvas_input,
    canvas_size,
    canvas_to_source,
    crop_box,
    crop_pass,
    estimate_page_angle,
    main,
    model_input,
    predict_grid,
    rectify,
    rerun_grid,
    rerun_record,
)

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
GRID = (9, 7)
SOURCE = (150, 200)  # (height, width): small enough for the ramp image


def _config(convention="flattener"):
    return DewarpModelConfig(
        input_size=(96, 80), grid_size=GRID, num_filter=4, grid_convention=convention
    )


def ramp_image(height, width):
    """Red holds each pixel's column and green its row, so a bilinear sample of the image
    tells which source point it shows (exact while both stay below 256)."""
    assert max(height, width) <= 256
    rows, cols = np.mgrid[0:height, 0:width]
    return np.stack([cols, rows, (rows + cols) % 7 * 30], -1).astype(np.uint8)


class Page:
    """A page that is an affine image of the output square: output position u in
    [-1, 1]^2 lies at ``center + R(angle) (u * size / 2)`` source pixels (edges at 0, W)."""

    def __init__(self, center, size, angle_deg=0.0):
        theta = np.radians(angle_deg)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.matrix = rotation @ np.diag(np.asarray(size, float) / 2)
        self.center = np.asarray(center, float)

    def pixels(self, grid_size):
        u = np.stack(np.meshgrid(edge_positions(grid_size[1]), edge_positions(grid_size[0])), -1)
        return u @ self.matrix.T + self.center

    def grid(self, grid_size=GRID, source_size=SOURCE):
        """The true grid: flattener values of the source at the edge-aligned nodes."""
        height, width = source_size
        return self.pixels(grid_size) / np.array([width, height]) * 2 - 1


class Oracle(torch.nn.Module):
    """A perfect predictor of ``page`` on a ramp image, plus a fixed ``bias`` in its own
    input frame. It fits which source point each input pixel shows (an affine map) on the
    middle of its input, which lies inside the crop at any turn, and returns the page's
    grid in its input frame, as align_corners=True values of the image it saw for
    ``uvdoc``."""

    def __init__(self, page, convention="flattener", bias=(0.0, 0.0)):
        super().__init__()
        self.page, self.convention, self.bias = page, convention, np.asarray(bias)
        self.seen = []

    def forward(self, x):
        _, _, height, width = x.shape
        cols = x[0, 0].double().cpu().numpy() * 255
        rows = x[0, 1].double().cpu().numpy() * 255
        qx, qy = np.meshgrid(center_positions(width), center_positions(height))
        middle = (np.abs(qx) < 0.2) & (np.abs(qy) < 0.2)
        design = np.stack([qx[middle], qy[middle], np.ones(middle.sum())], 1)
        fit_x = np.linalg.lstsq(design, cols[middle] + 0.5, rcond=None)[0]
        fit_y = np.linalg.lstsq(design, rows[middle] + 0.5, rcond=None)[0]
        matrix = np.array([fit_x[:2], fit_y[:2]])  # input values -> source pixels
        offset = np.array([fit_x[2], fit_y[2]])
        local = (self.page.pixels(GRID) - offset) @ np.linalg.inv(matrix).T + self.bias
        self.seen.append(local)
        if self.convention == "uvdoc":
            n = 2 * np.hypot(matrix[0], matrix[1])  # size of the image seen, in pixels
            local = local * n / (n - 1)
        grid = torch.from_numpy(local.transpose(2, 0, 1).astype(np.float32))[None]
        return grid, None


def _dewarper(page, convention="flattener", bias=(0.0, 0.0)):
    return Dewarper(Oracle(page, convention, bias), _config(convention), torch.device("cpu"))


def _identity(height, width):
    return np.stack(np.meshgrid(edge_positions(width), edge_positions(height)), -1).astype(
        np.float32
    )


class _IdentityGrid(torch.nn.Module):
    def forward(self, x):
        grid = _identity(5, 4).transpose(2, 0, 1)
        return torch.from_numpy(grid)[None].expand(x.shape[0], -1, -1, -1), None


class _FoldGrid(torch.nn.Module):
    """Identity grid with the two middle columns swapped: a reversed Jacobian."""

    def forward(self, x):
        grid = _identity(5, 4)[:, [0, 2, 1, 3]].transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(grid))[None].expand(
            x.shape[0], -1, -1, -1
        ), None


class _ConstantGrid(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 2, 5, 4), self.value), None


# ------------------------------------------------------------------------ R = 0


@pytest.mark.parametrize("convention", ["flattener", "uvdoc"])
def test_no_passes_is_the_whole_image_prediction(convention):
    torch.manual_seed(0)
    config = DewarpModelConfig(
        input_size=(64, 48), grid_size=(5, 4), num_filter=4, grid_convention=convention
    )
    model = UVDocNet(config).eval()
    dewarper = Dewarper(model, config, torch.device("cpu"))
    image = np.random.default_rng(1).integers(0, 256, (37, 29, 3), dtype=np.uint8)
    config = RenderConfig(on_error="return")
    plain = rectify(dewarper, image, render_config=config, allow_unchecked=True)
    zero = rectify(dewarper, image, render_config=config, allow_unchecked=True, rerun=0)
    np.testing.assert_array_equal(zero.image, plain.image)
    assert zero.status == plain.status and zero.reason_code == plain.reason_code
    assert zero.metadata["validation"] == plain.metadata["validation"]
    assert zero.metadata["dewarp"]["rerun"] == rerun_record() == reporting._no_rerun()
    assert zero.metadata["dewarp"]["crops"] == []
    assert rerun_record() == {
        "passes": 0,
        "deskew": False,
        "margin": None,
        "max_angle_deg": None,
        "min_agreement": None,
    }
    # The whole-image grid of rerun_grid is that prediction in flattener values.
    grid, passes = rerun_grid(dewarper, image, 0)
    raw = predict_grid(dewarper, image).astype(np.float64)
    if convention == "uvdoc":
        raw *= (np.array([29, 37]) - 1) / np.array([29, 37])
    np.testing.assert_allclose(grid, raw, rtol=0, atol=1e-12)
    assert passes == []


def test_a_pass_on_an_identity_grid_crops_the_whole_image():
    image = np.random.default_rng(2).integers(0, 256, (37, 29, 3), dtype=np.uint8)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    plain = rectify(dewarper, image)
    again = rectify(dewarper, image, rerun=2)
    np.testing.assert_array_equal(again.image, plain.image)
    crops = again.metadata["dewarp"]["crops"]
    assert [crop["pass"] for crop in crops] == [1, 2]
    assert all(crop["box"] == [0, 0, 29, 37] and crop["skipped"] is None for crop in crops)
    assert crops[0]["outline_box"] == [0.0, 0.0, 29.0, 37.0]
    assert again.metadata["dewarp"]["rerun"] == {
        "passes": 2,
        "deskew": False,
        "margin": RERUN_MARGIN,
        "max_angle_deg": None,
        "min_agreement": None,
    }


# ------------------------------------------------------------------- composition


@pytest.mark.parametrize("convention", ["flattener", "uvdoc"])
def test_a_perfect_predictor_keeps_the_true_map_through_a_crop(convention):
    image = ramp_image(*SOURCE)
    page = Page(center=(120.3, 60.7), size=(50, 36))  # x 95.3..145.3, y 42.7..78.7
    truth = page.grid()
    dewarper = _dewarper(page, convention)
    whole, _ = rerun_grid(dewarper, image, 0)
    np.testing.assert_allclose(whole, truth, rtol=0, atol=1e-5)
    grid, [record] = rerun_grid(dewarper, image, 1)
    np.testing.assert_allclose(grid, truth, rtol=0, atol=1e-5)
    # Outline bounds plus 4% of the page extent per side, in whole pixels.
    np.testing.assert_allclose(record["outline_box"], [95.3, 42.7, 145.3, 78.7], atol=1e-4)
    assert record["box"] == [93, 41, 148, 81]  # 93.3, 41.26, 147.3, 80.14 widened
    assert record["canvas_size"] == [40.0, 55.0] and record["pass"] == 1
    assert record["angle_deg"] is None and not record["rotated"] and record["skipped"] is None


@pytest.mark.parametrize("convention", ["flattener", "uvdoc"])
def test_a_rerun_equals_the_prediction_on_its_crop_placed_in_the_source(convention):
    image = ramp_image(*SOURCE)
    page = Page(center=(70.4, 90.2), size=(60, 44), angle_deg=4.0)
    dewarper = _dewarper(page, convention, bias=(0.03, -0.02))  # an imperfect predictor
    grid, [record] = rerun_grid(dewarper, image, 1)
    left, top, right, bottom = record["box"]
    crop = image[top:bottom, left:right]
    local = predict_grid(dewarper, crop).astype(np.float64)
    size = np.array([right - left, bottom - top], float)
    if convention == "uvdoc":
        local *= (size - 1) / size
    # Crop value q_c is source pixel left + (q_c + 1) / 2 * width, i.e. value 2 x / W - 1.
    placed = ((local + 1) / 2 * size + [left, top]) / [SOURCE[1], SOURCE[0]] * 2 - 1
    np.testing.assert_allclose(grid, placed, rtol=0, atol=1e-12)
    # The bias lives in the crop frame, so the rerun differs from the whole-image map.
    whole, _ = rerun_grid(dewarper, image, 0)
    assert np.abs(whole - grid).max() > 1e-3


@pytest.mark.parametrize("convention", ["flattener", "uvdoc"])
@pytest.mark.parametrize("angle", [20.0, -33.0])
def test_deskew_turns_the_page_upright_and_composes_back(convention, angle):
    image = ramp_image(*SOURCE)
    page = Page(center=(98.6, 77.1), size=(70, 50), angle_deg=angle)
    truth = page.grid()
    dewarper = _dewarper(page, convention)
    plain, [straight] = rerun_grid(dewarper, image, 1)
    grid, [record] = rerun_grid(dewarper, image, 1, deskew=True)
    np.testing.assert_allclose(grid, truth, rtol=0, atol=1e-5)
    np.testing.assert_allclose(plain, truth, rtol=0, atol=1e-5)
    assert record["rotated"] and record["deskew_skipped"] is None
    assert record["angle_deg"] == pytest.approx(angle, abs=1e-5)
    assert record["box"] == straight["box"]  # the same axis-aligned crop, then turned
    box = tuple(record["box"])
    assert record["canvas_size"] == pytest.approx(list(canvas_size(box, record["angle_deg"])))
    # The model saw the page upright: its top and bottom rows are level, its sides plumb.
    seen = dewarper.model.seen[-1]
    for edge in (seen[0, :, 1], seen[-1, :, 1], seen[:, 0, 0], seen[:, -1, 0]):
        assert np.ptp(edge) < 1e-5
    # The canvas keeps the whole crop: its corners lie inside the canvas.
    matrix, offset = canvas_to_source(box, record["angle_deg"], record["canvas_size"], SOURCE)
    left, top, right, bottom = box
    corners = np.array([[left, top], [right, top], [left, bottom], [right, bottom]], float)
    values = corners / [SOURCE[1], SOURCE[0]] * 2 - 1
    inside = (values - offset) @ np.linalg.inv(matrix).T
    assert np.abs(inside).max() <= 1 + 1e-9
    assert np.abs(inside).max() == pytest.approx(1.0)


def test_the_canvas_input_without_a_turn_is_the_resized_crop():
    crop = np.random.default_rng(3).integers(0, 256, (41, 57, 3), dtype=np.uint8)
    dewarper = Dewarper(_IdentityGrid(), _config(), torch.device("cpu"))
    box = (10, 20, 67, 61)
    straight = canvas_input(dewarper, crop, 0.0, canvas_size(box, 0.0))
    np.testing.assert_allclose(straight, model_input(dewarper, crop), rtol=0, atol=1e-5)
    # Round trip: canvas values -> source values -> canvas values, for a turn.
    matrix, offset = canvas_to_source(box, 17.0, canvas_size(box, 17.0), (90, 120))
    points = np.random.default_rng(4).uniform(-1, 1, (50, 2))
    source = points @ matrix.T + offset
    np.testing.assert_allclose((source - offset) @ np.linalg.inv(matrix).T, points, atol=1e-12)
    # The canvas center is the box center; its x axis follows the page direction.
    center = np.array([(10 + 67) / 120 - 1, (20 + 61) / 90 - 1])
    np.testing.assert_allclose(offset, center, atol=1e-15)
    pixels = matrix[:, 0] * [60, 45]  # the canvas x axis in source pixels
    assert np.degrees(np.arctan2(pixels[1], pixels[0])) == pytest.approx(17.0)


# ---------------------------------------------------------------------- clipping


def test_the_crop_box_is_clipped_to_the_image():
    outline = np.array([[-12.5, 30.2], [80.0, 30.2], [80.0, 160.7], [-12.5, 160.7]])
    assert crop_box(outline, (150, 100)) == (0, 24, 84, 150)  # 24.98 -> 24, 83.7 -> 84
    assert crop_box(outline, (150, 100), margin=0.0) == (0, 30, 80, 150)
    far = np.array([[250.0, 10.0], [300.0, 40.0]])
    assert crop_box(far, (150, 200)) == (200, 8, 200, 42)


@pytest.mark.parametrize("deskew", [False, True])
def test_a_page_across_the_image_border_is_cropped_to_the_border(deskew):
    image = ramp_image(*SOURCE)
    page = Page(center=(185.5, 132.5), size=(40, 36), angle_deg=6.0)  # past right and bottom
    truth = page.grid()
    assert truth.max() > 1.02
    dewarper = _dewarper(page)
    grid, [record] = rerun_grid(dewarper, image, 1, deskew=deskew)
    assert record["box"][2:] == [200, 150] and record["skipped"] is None
    assert record["rotated"] == deskew
    np.testing.assert_allclose(grid, truth, rtol=0, atol=1e-5)


def test_passes_that_cannot_crop_keep_the_grid():
    image = np.zeros((30, 40, 3), np.uint8)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    grid = _identity(5, 4).astype(np.float64)
    outside = grid + [3.0, 0.0]  # the whole page right of the image
    kept, record = crop_pass(dewarper, image, outside)
    assert record["skipped"] == "empty_crop" and record["box"] == [40, 0, 40, 30]
    np.testing.assert_array_equal(kept, outside)
    broken = grid.copy()
    broken[0, 2] = np.nan
    kept, record = crop_pass(dewarper, image, broken, deskew=True)
    assert record["skipped"] == "non_finite_outline" and record["box"] is None
    np.testing.assert_array_equal(kept, broken)


# ------------------------------------------------------------------ deskew rules


def test_the_angle_estimate_is_robust_and_refuses_bad_outlines():
    page = Page(center=(100, 75), size=(80, 60), angle_deg=10.0)
    grid = page.grid()
    assert estimate_page_angle(grid, SOURCE) == (pytest.approx(10.0), None)
    spiked = grid.copy()
    spiked[0, 3, 1] -= 0.2  # one top node pulled far up: two outlying segments
    angle, reason = estimate_page_angle(spiked, SOURCE)
    assert reason is None and angle == pytest.approx(10.0, abs=1e-9)
    collapsed = grid.copy()
    collapsed[0] = collapsed[0, 0]  # the top edge is one point
    assert estimate_page_angle(collapsed, SOURCE) == (None, "degenerate_outline")
    missing = grid.copy()
    missing[-1, 4] = np.nan
    assert estimate_page_angle(missing, SOURCE) == (None, "degenerate_outline")
    crossed = grid.copy()
    crossed[-1] = crossed[-1, ::-1]  # the bottom edge runs the other way
    assert estimate_page_angle(crossed, SOURCE) == (None, "inconsistent_edges")
    upside_down = Page(center=(100, 75), size=(80, 60), angle_deg=180.0).grid()
    assert abs(estimate_page_angle(upside_down, SOURCE)[0]) == pytest.approx(180.0)


def test_deskew_skips_steep_or_unusable_angles():
    image = ramp_image(*SOURCE)
    steep = Page(center=(100.2, 74.9), size=(60, 40), angle_deg=60.0)
    dewarper = _dewarper(steep)
    plain, [straight] = rerun_grid(dewarper, image, 1)
    grid, [record] = rerun_grid(dewarper, image, 1, deskew=True)
    assert record["angle_deg"] == pytest.approx(60.0) and 60 > DESKEW_MAX_DEG
    assert record["deskew_skipped"] == "angle_above_limit" and not record["rotated"]
    assert record["box"] == straight["box"] and record["canvas_size"] == straight["canvas_size"]
    np.testing.assert_array_equal(grid, plain)
    identity = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    crossed = _identity(5, 4).astype(np.float64)
    crossed[-1] = crossed[-1, ::-1]
    _, record = crop_pass(identity, image, crossed, deskew=True)
    assert record["angle_deg"] is None and record["deskew_skipped"] == "inconsistent_edges"
    assert not record["rotated"] and record["skipped"] is None


def test_invalid_rerun_settings_are_refused():
    for rerun, deskew in ((-1, False), (True, False), (1.0, False), (0, True), (1, 1)):
        with pytest.raises(FlattenerError, match="rerun|deskew"):
            rerun_record(rerun, deskew)
    image = np.zeros((20, 20, 3), np.uint8)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    with pytest.raises(FlattenerError, match="deskew"):
        rectify(dewarper, image, deskew=True)
    assert rerun_record(2, True) == {
        "passes": 2,
        "deskew": True,
        "margin": RERUN_MARGIN,
        "max_angle_deg": DESKEW_MAX_DEG,
        "min_agreement": predict.DESKEW_MIN_AGREEMENT,
    }
    assert predict.parse_rerun("3") == 3
    for bad in ("-1", "x", "1.5"):
        with pytest.raises(Exception, match="integer"):
            predict.parse_rerun(bad)


# ----------------------------------------------------------------- failure path


def test_rerun_keeps_every_failure_rule():
    image = np.random.default_rng(5).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    dewarper = Dewarper(_FoldGrid(), TINY, torch.device("cpu"))
    for rerun, deskew in ((1, False), (2, True)):
        settings = {"rerun": rerun, "deskew": deskew}
        with pytest.raises(FlattenerError) as error:  # the checked path rejects the fold
            rectify(dewarper, image, **settings)
        assert error.value.status == Status.INVALID_TRANSFORM
        returned = rectify(
            dewarper, image, render_config=RenderConfig(on_error="return"), **settings
        )
        assert returned.status == Status.INVALID_TRANSFORM and returned.image is None
        assert returned.reason_code == "fold_or_degenerate"
        assert len(returned.metadata["dewarp"]["crops"]) == rerun
        original = rectify(
            dewarper, image, render_config=RenderConfig(on_error="original"), **settings
        )
        assert original.used_fallback and np.array_equal(original.image, image)
        forced = rectify(dewarper, image, allow_unchecked=True, **settings)
        assert forced.status == Status.INVALID_TRANSFORM and forced.used_fallback
        assert forced.metadata["forced_render"]
        assert forced.metadata["dewarp"]["rerun"] == rerun_record(rerun, deskew)
    # A map that leaves the source keeps failing after a skipped pass.
    outside = Dewarper(_ConstantGrid(5.0), TINY, torch.device("cpu"))
    result = rectify(outside, image, render_config=RenderConfig(on_error="return"), rerun=1)
    assert result.status != Status.SUCCESS and result.image is None
    assert result.metadata["dewarp"]["crops"][0]["skipped"] == "empty_crop"


def _checkpoint(path):
    torch.manual_seed(0)
    save_checkpoint(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": 3,
            "track": "release",
            "model_config": TINY.to_dict(),
            "lineage": [{"kind": "scratch"}],
            "data_spec_sha256": "0" * 64,
            "code": {"sha256": "1" * 64},
            "model_state": UVDocNet(TINY).eval().state_dict(),
        },
    )


def _docunet(tmp_path):
    crops, output = tmp_path / "crop", tmp_path / "out"
    crops.mkdir()
    pixels = np.random.default_rng(6).integers(0, 256, (45, 33, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(crops / "1_1 copy.png")
    weights = tmp_path / "model.pt"
    _checkpoint(weights)
    arguments = ["--weights", str(weights), "docunet", "--input-dir", str(crops)]
    return [*arguments, "--output-dir", str(output)], output


def _identity_grid(dewarper, image):
    return _identity(*dewarper.config.grid_size)


def _folding_grid(dewarper, image):
    return _identity_grid(dewarper, image)[:, [0, 2, 1, 3]]


def test_docunet_batch_never_keeps_outputs_of_other_rerun_settings(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
    arguments, output = _docunet(tmp_path)
    record_path = output / "predictions.json"

    def run(*flags):
        assert main([*arguments, *flags]) == 0
        counts = json.loads(capsys.readouterr().out)
        record = json.loads(record_path.read_text())
        assert record["rerun"] == record["evidence"]["rerun"]
        return counts["written"], counts["kept"], record["rerun"]["passes"]

    assert run() == (1, 0, 0)
    assert run("--rerun", "0") == (0, 1, 0)
    assert run("--rerun", "1") == (1, 0, 1)  # the same pixels, never kept across settings
    assert run("--rerun", "1") == (0, 1, 1)
    [item] = json.loads(record_path.read_text())["items"]
    assert item["crops"][0]["box"] == [0, 0, 33, 45]
    assert run("--rerun", "1", "--deskew") == (1, 0, 1)
    assert run("--rerun", "2", "--deskew") == (1, 0, 2)
    found = prediction_record(output)
    assert found["rerun"] == report_evidence(found)["rerun"] == rerun_record(2, True)
    assert run() == (1, 0, 0)
    # A record without the field predates the option; its images are whole-image ones.
    record = json.loads(record_path.read_text())
    del record["rerun"]
    record_path.write_text(json.dumps(record))
    assert prediction_record(output)["rerun"] == rerun_record()
    assert run()[:2] == (0, 1)
    record = json.loads(record_path.read_text())
    del record["rerun"]
    record_path.write_text(json.dumps(record))
    assert run("--rerun", "1")[:2] == (1, 0)
    with pytest.raises(SystemExit):
        main([*arguments, "--deskew"])


def test_cli_failures_still_fail_with_a_rerun(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _folding_grid)
    arguments, output = _docunet(tmp_path)
    assert main([*arguments, "--rerun", "1"]) == EXIT_FAILED
    assert not (output / "1_1.png").exists()
    assert json.loads(capsys.readouterr().out)["failed"] == 1
    assert main([*arguments, "--rerun", "1", "--force-render"]) == EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["forced"] == 1
    [item] = json.loads((output / "predictions.json").read_text())["items"]
    assert item["status"] == Status.INVALID_TRANSFORM.value and item["forced"]
    assert len(item["crops"]) == 1
    source = tmp_path / "crop" / "1_1 copy.png"
    target = tmp_path / "single.png"
    single = ["--weights", arguments[1], "image", "--input", str(source), "--output", str(target)]
    assert main([*single, "--rerun", "2", "--deskew"]) == EXIT_FAILED and not target.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == Status.INVALID_TRANSFORM.value
    assert printed["rerun"] == printed["evidence"]["rerun"] == rerun_record(2, True)
    assert [crop["pass"] for crop in printed["crops"]] == [1, 2]
    assert main([*single, "--rerun", "1", "--force-render"]) == EXIT_FAILED and target.exists()
    with pytest.raises(SystemExit):
        main([*single, "--rerun", "-1"])
    with pytest.raises(SystemExit):
        main([*single, "--deskew"])


# ------------------------------------------------------------------- dense maps


def test_flattener_values_only_place_the_nodes():
    rng = np.random.default_rng(7)
    values = rng.uniform(-1.1, 1.1, (6, 5, 2))
    size, out = (31, 23), (40, 27)
    n = np.array([size[1], size[0]], float)
    as_uvdoc = (values * n / (n - 1)).astype(np.float32)  # the same points, align_corners
    np.testing.assert_allclose(
        predict.dense_backward_map(values, "uvdoc", out, size, flattener_values=True),
        predict.dense_backward_map(as_uvdoc, "uvdoc", out, size),
        rtol=0,
        atol=1e-6,
    )
    for inset in (None, 0.1):
        np.testing.assert_array_equal(
            predict.dense_backward_map(
                values, "flattener", out, size, inset, flattener_values=True
            ),
            predict.dense_backward_map(values, "flattener", out, size, inset),
        )


@pytest.mark.parametrize("convention", ["flattener", "uvdoc"])
def test_rectify_renders_the_composed_map_from_the_original_image(convention):
    image = ramp_image(*SOURCE)
    page = Page(center=(101.3, 70.6), size=(64, 48), angle_deg=12.0)
    dewarper = _dewarper(page, convention)
    result = rectify(dewarper, image, (48, 64), rerun=1, deskew=True)
    assert result.status == Status.SUCCESS and result.metadata["dewarp"]["crops"][0]["rotated"]
    truth = predict.dense_backward_map(
        page.grid(), convention, (48, 64), SOURCE, flattener_values=True
    )
    expected = predict.render_unchecked(image, truth, (255, 255, 255))
    assert np.abs(result.image.astype(int) - expected).max() <= 1
