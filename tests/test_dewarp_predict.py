"""Dewarp prediction: grid conventions, checkpoint loading, output quantization, the
grid-level inset, crop-and-rerun with deskew, the DocUNet batch CLI and batches run by
another pipeline."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from flattener import FlattenerError, RenderConfig
from flattener.benchmark.reporting import prediction_record, report_evidence
from flattener.contracts import Status
from flattener.dewarp import predict
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import (
    center_positions,
    edge_positions,
    grid_to_map,
    uvdoc_dense_map,
)
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import (
    DEFAULT_PIPELINE,
    BatchInput,
    Dewarper,
    canvas_size,
    canvas_to_source,
    crop_box,
    dense_backward_map,
    estimate_page_angle,
    ieee_float32,
    load_dewarper,
    main,
    predict_grid,
    prediction_evidence,
    quantize_rgb,
    rectify,
    rectify_batch,
    rerun_box,
    rerun_grid,
    resolve_inset,
    sample_rgb,
    with_tta,
)
from flattener.renderer import render

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)


class _IdentityGrid(torch.nn.Module):
    def forward(self, x):
        grid = np.stack(np.meshgrid(edge_positions(4), edge_positions(5)), 0)
        return torch.from_numpy(grid).float()[None].expand(x.shape[0], -1, -1, -1), None


class _RecordFlags(_IdentityGrid):
    def forward(self, x):
        self.seen = _flags()
        return super().forward(x)


def _flags():
    backends = torch.backends
    return (
        backends.cudnn.conv.fp32_precision,
        backends.cuda.matmul.fp32_precision,
        backends.mkldnn.matmul.fp32_precision,
        torch.get_float32_matmul_precision(),
        backends.cudnn.benchmark,
    )


def _set_flags(conv, cuda_matmul, mkldnn_matmul, legacy, benchmark):
    torch.set_float32_matmul_precision(legacy)
    backends = torch.backends
    backends.cudnn.conv.fp32_precision = conv
    backends.cuda.matmul.fp32_precision = cuda_matmul
    backends.mkldnn.matmul.fp32_precision = mkldnn_matmul
    backends.cudnn.benchmark = benchmark


def _checkpoint(path, seed=0):
    torch.manual_seed(seed)
    model = UVDocNet(TINY).eval()
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
            "model_state": model.state_dict(),
        },
    )
    return model


def test_identity_grid_reproduces_the_input_without_tf32():
    image = np.random.default_rng(0).integers(0, 256, (37, 29, 3), dtype=np.uint8)
    original = _flags()
    model = _RecordFlags()
    dewarper = Dewarper(model, TINY, torch.device("cpu"))
    try:
        _set_flags("tf32", "tf32", "tf32", "high", True)
        result = rectify(dewarper, image)
        assert model.seen == ("ieee", "ieee", "ieee", "highest", False)
        assert _flags() == ("tf32", "tf32", "tf32", "high", True)
        with pytest.raises(ValueError), ieee_float32():
            raise ValueError
        assert _flags() == ("tf32", "tf32", "tf32", "high", True)
    finally:
        _set_flags(*original)
    assert _flags() == original
    np.testing.assert_array_equal(result.image, image)
    assert not result.used_fallback and result.metadata["dewarp"]["grid_size"] == [5, 4]
    assert rectify(dewarper, image, (50, 20)).image.shape == (50, 20, 3)
    with pytest.raises(FlattenerError):
        rectify(dewarper, image.astype(np.float32))


def _upstream_weights():
    if not os.environ.get("FLATTENER_UVDOC_REPO"):
        pytest.skip("set FLATTENER_UVDOC_REPO to an upstream UVDoc checkout")
    repo = Path(os.environ["FLATTENER_UVDOC_REPO"])
    weights = repo / "model/best_model.pkl"
    if not weights.is_file():
        pytest.skip(f"UVDoc weights not found at {weights}")
    return weights


def _checkerboard_page():
    yy, xx = np.mgrid[0:120, 0:90]
    image = np.full((120, 90, 3), 40, np.uint8)
    image[10:110, 12:80] = 235
    image[(yy // 8 + xx // 8) % 2 == 0] //= 2
    return image


def test_a_checkpoint_predicts_what_its_model_gives_on_the_resized_image(tmp_path):
    path = tmp_path / "model.pt"
    model = _checkpoint(path)
    dewarper = load_dewarper(path)
    assert dewarper.info["track"] == "release" and dewarper.info["kind"] == "checkpoint"
    image = np.random.default_rng(1).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    tensor = F.interpolate(
        torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255,
        size=(64, 48),
        mode="bilinear",
        align_corners=False,
    )
    with torch.no_grad():
        expected = model(tensor)[0][0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(predict_grid(dewarper, image), expected, atol=1e-6)


def _identity(height, width):
    return np.stack(np.meshgrid(edge_positions(width), edge_positions(height)), -1).astype(
        np.float32
    )


def _identity_grid(dewarper, image):
    return _identity(*dewarper.config.grid_size)


def test_docunet_batch_writes_crop_sized_outputs_and_keeps_only_matching_ones(
    tmp_path, capsys, monkeypatch
):
    # A random tiny model folds its map; an identity grid passes the render checks.
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
    crops, output = tmp_path / "crop", tmp_path / "out"
    crops.mkdir()
    rng = np.random.default_rng(2)
    sizes = {"1_1": (45, 33), "10_2": (30, 52)}
    for name, size in sizes.items():
        Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(
            crops / f"{name} copy.png"
        )
    (crops / "notes.txt").write_text("ignored")
    weights = tmp_path / "model.pt"
    _checkpoint(weights)
    arguments = [
        "--weights",
        str(weights),
        "docunet",
        "--input-dir",
        str(crops),
        "--output-dir",
        str(output),
    ]
    assert main(arguments) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts == {"items": 2, "written": 2, "kept": 0, "failed": 0, "forced": 0}
    for name, size in sizes.items():
        with Image.open(output / f"{name}.png") as written:
            assert written.size == size[::-1]
    report = json.loads((output / "predictions.json").read_text())
    assert [item["output"] for item in report["items"]] == ["1_1.png", "10_2.png"]
    assert report["model"]["track"] == "release" and report["inputs"] == "docunet_crop"
    assert report["evidence"]["evidence_kind"] == "smoke_training"  # a 3-step checkpoint
    assert report["evidence"]["measurement"] == "predictions"
    assert main(arguments) == 0 and json.loads(capsys.readouterr().out)["written"] == 0
    kept = json.loads((output / "predictions.json").read_text())["items"]
    assert [(item["action"], item["status"], item["size"]) for item in kept] == [
        ("kept", "success", [45, 33]),
        ("kept", "success", [30, 52]),
    ]
    other = tmp_path / "other.pt"
    _checkpoint(other, seed=1)
    with pytest.raises(FlattenerError, match="other weights"):
        main(["--weights", str(other), *arguments[2:]])
    # A file the same-weights report does not list is re-rendered, not kept.
    report = json.loads((output / "predictions.json").read_text())
    report["items"] = report["items"][:1]
    (output / "predictions.json").write_text(json.dumps(report))
    assert main(arguments) == 0 and json.loads(capsys.readouterr().out)["written"] == 1
    # Outputs of unknown provenance (no report) are refused unless --overwrite.
    (output / "predictions.json").unlink()
    with pytest.raises(FlattenerError, match="without predictions.json"):
        main(arguments)
    assert main([*arguments, "--overwrite"]) == 0
    assert json.loads(capsys.readouterr().out)["written"] == 2
    # Outputs of another quantization are never kept.
    record_path = output / "predictions.json"

    def run(quantize):
        assert main([*arguments, "--quantize", quantize]) == 0
        counts = json.loads(capsys.readouterr().out)
        record = json.loads(record_path.read_text())
        assert record["quantize"] == record["evidence"]["quantize"] == quantize
        return counts["written"], counts["kept"]

    assert run("truncate") == (2, 0)  # the rounded images are not kept for truncation
    assert run("truncate") == (0, 2)
    found = prediction_record(output)
    assert found["quantize"] == report_evidence(found)["quantize"] == "truncate"
    assert run("round") == (2, 0)
    # A record without the field predates the option; its images were rounded.
    for quantize, expected in (("round", (0, 2)), ("truncate", (2, 0))):
        record = json.loads(record_path.read_text())
        del record["quantize"]
        record_path.write_text(json.dumps(record))
        assert prediction_record(output)["quantize"] == "round"
        assert run(quantize) == expected


def test_upstream_weights_match_the_upstream_unwarping():
    dewarper = load_dewarper(_upstream_weights())
    assert dewarper.info["kind"] == "uvdoc" and dewarper.config.grid_convention == "uvdoc"
    image = _checkerboard_page()
    result = rectify(dewarper, image, allow_unchecked=True)
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
    small = F.interpolate(tensor, size=(712, 488), mode="bilinear", align_corners=False)
    with torch.no_grad():
        grid = dewarper.model(small)[0]
    dense = F.interpolate(grid, size=(120, 90), mode="bilinear", align_corners=True)
    upstream = F.grid_sample(tensor, dense.permute(0, 2, 3, 1), align_corners=True)
    expected = np.floor(upstream[0].permute(1, 2, 0).numpy() * 255 + 0.5)
    inside = (dense[0].abs() <= 0.999).all(0).numpy()
    assert inside.mean() > 0.5
    assert np.abs(result.image[inside].astype(int) - expected[inside]).max() <= 1


def test_truncate_floors_the_samples_that_round_rounds():
    # Source columns a and b; a 4-wide output samples 1/4 and 3/4 of the way between them.
    image = np.zeros((4, 2, 3), np.uint8)
    image[:, 0], image[:, 1] = (0, 10, 200), (101, 31, 253)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    exact = np.array([[0, 10, 200], [25.25, 15.25, 213.25], [75.75, 25.75, 239.75], image[0, 1]])
    truncated = rectify(dewarper, image, (4, 4), quantize="truncate")
    rounded = rectify(dewarper, image, (4, 4))
    assert truncated.status == rounded.status == Status.SUCCESS
    np.testing.assert_array_equal(truncated.image, np.broadcast_to(np.floor(exact), (4, 4, 3)))
    np.testing.assert_array_equal(rounded.image, np.broadcast_to(np.floor(exact + 0.5), (4, 4, 3)))
    assert truncated.metadata["dewarp"]["quantize"] == "truncate"
    assert rounded.metadata["dewarp"]["quantize"] == "round"
    values = np.array([0, 0.1, 0.5, 0.7, 0.999, 1, 1.25, -0.5], np.float32)
    assert quantize_rgb(values, "truncate").tolist() == [0, 25, 127, 178, 254, 255, 255, 0]
    assert quantize_rgb(values, "round").tolist() == [0, 26, 128, 179, 255, 255, 255, 0]
    inside = values[(values >= 0) & (values <= 1)]  # the paper's (x * 255).astype(uint8)
    np.testing.assert_array_equal(quantize_rgb(inside, "truncate"), (inside * 255).astype(np.uint8))
    # On a map partly outside the source, round is what flattener.render gives.
    image = np.random.default_rng(4).integers(0, 256, (30, 20, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(np.linspace(-1.1, 0.9, 25), np.linspace(-0.8, 1.05, 18), indexing="ij")
    backward_map = np.stack([xx, yy], -1).astype(np.float32)  # some pixels outside the source
    fill = (1, 2, 3)
    checked = render(image, backward_map, config=RenderConfig(fill_rgb=fill))
    assert checked.status == Status.SUCCESS and not checked.valid_mask.all()
    rounded = sample_rgb(image, backward_map, checked.valid_mask, fill, "round")
    np.testing.assert_array_equal(rounded, checked.image)
    truncated = sample_rgb(image, backward_map, checked.valid_mask, fill, "truncate")
    gap = checked.image.astype(int) - truncated
    assert gap.min() == 0 and gap.max() == 1  # truncation is at most one level lower
    assert (truncated[~checked.valid_mask] == fill).all()
    with pytest.raises(FlattenerError, match="quantize"):
        quantize_rgb(np.zeros(3, np.float32), "nearest")


# Grid-level inset of the predicted outline (--inset)

PRESET = "syntheticdoc_half_cell"


def test_the_inset_moves_the_page_edges_to_one_minus_e():
    grid = np.random.default_rng(0).uniform(-1.1, 1.1, (7, 5, 2)).astype(np.float32)
    for inset in (None, 0, (0, 0)):
        assert resolve_inset(inset) == (0.0, 0.0)
        np.testing.assert_array_equal(
            dense_backward_map(grid, "flattener", (31, 23), (40, 30), inset),
            grid_to_map(grid, (31, 23)),
        )
        np.testing.assert_array_equal(
            dense_backward_map(grid, "uvdoc", (31, 23), (40, 30), inset),
            uvdoc_dense_map(grid, (31, 23), (40, 30)),
        )
    ex, ey = resolve_inset(PRESET)
    height, width = 41, 27
    for convention in ("flattener", "uvdoc"):
        # The inset path at e -> 0 reads the grid where the plain map does.
        plain = dense_backward_map(grid, convention, (31, 23), (40, 30))
        tiny = dense_backward_map(grid, convention, (31, 23), (40, 30), (1e-12, 0.0))
        np.testing.assert_allclose(tiny, plain, rtol=0, atol=1e-6)
        identity = _identity(6, 5)  # align_corners values for "uvdoc", flattener values otherwise
        dense = dense_backward_map(identity, convention, (height, width), (height, width), PRESET)
        # Output pixel center u shows the page position (1 - e) u: the grid on [-(1-e), 1-e].
        np.testing.assert_allclose(dense[5, :, 0], (1 - ex) * center_positions(width), atol=1e-6)
        np.testing.assert_allclose(dense[:, 3, 1], (1 - ey) * center_positions(height), atol=1e-6)
        # The outer pixel edges (u = -1, +1) lie at -(1 - e) and 1 - e.
        left = 1.5 * dense[5, 0, 0] - 0.5 * dense[5, 1, 0]
        right = 1.5 * dense[5, -1, 0] - 0.5 * dense[5, -2, 0]
        top = 1.5 * dense[0, 3, 1] - 0.5 * dense[1, 3, 1]
        bottom = 1.5 * dense[-1, 3, 1] - 0.5 * dense[-2, 3, 1]
        np.testing.assert_allclose([left, right], [-(1 - ex), 1 - ex], atol=1e-6)
        np.testing.assert_allclose([top, bottom], [-(1 - ey), 1 - ey], atol=1e-6)


# Crop-and-rerun (--rerun and --deskew)

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
    """A perfect predictor of ``page`` on a ramp image. It fits which source point each
    input pixel shows (an affine map) on the middle of its input, which lies inside the
    crop at any turn, and returns the page's grid in its input frame, as
    align_corners=True values of the image it saw for ``uvdoc``."""

    def __init__(self, page, convention="flattener"):
        super().__init__()
        self.page, self.convention = page, convention
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
        local = (self.page.pixels(GRID) - offset) @ np.linalg.inv(matrix).T
        self.seen.append(local)
        if self.convention == "uvdoc":
            n = 2 * np.hypot(matrix[0], matrix[1])  # size of the image seen, in pixels
            local = local * n / (n - 1)
        grid = torch.from_numpy(local.transpose(2, 0, 1).astype(np.float32))[None]
        return grid, None


def _dewarper(page, convention="flattener"):
    return Dewarper(Oracle(page, convention), _config(convention), torch.device("cpu"))


@pytest.mark.parametrize("convention", ["flattener", "uvdoc"])
def test_a_perfect_predictor_keeps_the_true_map_through_crop_and_deskew(convention):
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

    angle = -33.0
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
    # rectify renders the composed map from the original image.
    result = rectify(dewarper, image, (50, 70), rerun=1, deskew=True)
    assert result.status == Status.SUCCESS and result.metadata["dewarp"]["crops"][0]["rotated"]
    truth_map = predict.dense_backward_map(
        truth, convention, (50, 70), SOURCE, flattener_values=True
    )
    expected = predict.render_unchecked(image, truth_map, (255, 255, 255))
    assert np.abs(result.image.astype(int) - expected).max() <= 1


# Rerun boxes and rectify_batch with another pipeline


class Tilted(torch.nn.Module):
    """A seeded UVDocNet whose grid is added to a page turned by ``angle`` at 70% of the
    view, so crop passes and deskew have something to do (a bare random network predicts
    a collapsed grid)."""

    def __init__(self, config, seed=0, angle=8.0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = UVDocNet(config).eval()
        gh, gw = config.grid_size
        nodes = np.stack(np.meshgrid(edge_positions(gw), edge_positions(gh)), -1) * 0.7
        theta = np.radians(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        base = (nodes @ rotation.T).transpose(2, 0, 1).astype(np.float32)
        self.register_buffer("base", torch.from_numpy(base)[None])

    def forward(self, x):
        grid, grid3d = self.net(x)
        return self.base + 0.5 * grid, grid3d


def _tilted_dewarper():
    config = DewarpModelConfig(input_size=(96, 72), grid_size=(9, 7), num_filter=4)
    info = {"sha256": "seeded-uvdocnet", "kind": "checkpoint", "step": 1, "track": "release"}
    return Dewarper(Tilted(config).eval(), config, torch.device("cpu"), info)


def test_crop_boxes_page_angles_and_rerun_skips():
    outline = np.array([[-12.5, 30.2], [80.0, 30.2], [80.0, 160.7], [-12.5, 160.7]])
    assert crop_box(outline, (150, 100)) == (0, 24, 84, 150)  # 24.98 -> 24, 83.7 -> 84
    assert crop_box(outline, (150, 100), margin=0.0) == (0, 30, 80, 150)
    far = np.array([[250.0, 10.0], [300.0, 40.0]])
    assert crop_box(far, (150, 200)) == (200, 8, 200, 42)
    # The page angle ignores outlying edge segments and refuses broken outlines.
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
    # Rerun boxes: deskew stays within the angle limit; empty or broken outlines skip.
    size = (200, 200)  # square: turning normalized values turns source pixels alike
    theta = np.radians(60.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    nodes = np.stack(np.meshgrid(edge_positions(7), edge_positions(9)), -1) * 0.4 @ rotation.T
    box, angle, record = rerun_box(nodes, size, deskew=True)
    assert angle == 0.0 and record["deskew_skipped"] == "angle_above_limit"
    box, angle, record = rerun_box(nodes, size, deskew=True, max_angle_deg=None)
    assert record["deskew_skipped"] is None and angle == pytest.approx(60.0)
    box, angle, record = rerun_box(nodes, size)
    assert angle == 0.0 and record["angle_deg"] is None and box is not None
    point = np.zeros((9, 7, 2))
    assert (
        rerun_box(point, size)[0] is None and rerun_box(point, size)[2]["skipped"] == "empty_crop"
    )
    broken = nodes.copy()
    broken[0, 0] = np.nan
    assert rerun_box(broken, size)[2]["skipped"] == "non_finite_outline"


def _inputs(images):
    return [
        BatchInput(f"{index}.png", f"{index} copy.png", lambda image=image: image)
        for index, image in enumerate(images)
    ]


def _pipeline_rectifier(dewarper, calls):
    def rectify_one(image, output_size):
        calls.append(output_size)
        result = rectify(
            dewarper, image, output_size, render_config=RenderConfig(on_error="return")
        )
        result.metadata["batch_item"] = {"outcome": "rectified", "status": "ignored"}
        return result

    return rectify_one


def test_the_batch_writes_what_rectify_renders_and_refuses_another_pipeline(tmp_path):
    dewarper = _tilted_dewarper()
    images = [
        np.random.default_rng(seed).integers(0, 256, (150, 200, 3), np.uint8) for seed in (5, 6)
    ]
    inputs = _inputs(images)
    report = rectify_batch(dewarper, inputs, tmp_path, force_render=True, rerun=1, deskew=True)
    assert report["pipeline"] == DEFAULT_PIPELINE == report["evidence"]["pipeline"]
    for index, image in enumerate(images):
        result = rectify(
            dewarper,
            image,
            render_config=RenderConfig(on_error="return"),
            allow_unchecked=True,
            rerun=1,
            deskew=True,
        )
        with Image.open(tmp_path / f"{index}.png") as written:
            assert np.asarray(written).tobytes() == result.image.tobytes()
        item = report["items"][index]
        assert item["status"] == str(result.status)
        assert item["crops"] == json.loads(json.dumps(result.metadata["dewarp"]["crops"]))
    calls = []
    other = {"name": "flattener.scan", "passes": 1}
    with pytest.raises(FlattenerError, match="another pipeline") as error:
        rectify_batch(
            dewarper,
            inputs,
            tmp_path,
            force_render=True,
            rectifier=_pipeline_rectifier(dewarper, calls),
            pipeline=other,
        )
    assert error.value.status == Status.IO_ERROR and error.value.reason_code == "output_exists"
    assert calls == []
    report = rectify_batch(
        dewarper,
        inputs,
        tmp_path,
        overwrite=True,
        rectifier=_pipeline_rectifier(dewarper, calls),
        pipeline=other,
    )
    assert len(calls) == 2 and report["pipeline"] == other
    assert [item["outcome"] for item in report["items"]] == ["rectified", "rectified"]
    assert all(item["status"] != "ignored" for item in report["items"])
    # The same pipeline keeps its outputs; the default pipeline is now the other one.
    again = rectify_batch(
        dewarper, inputs, tmp_path, rectifier=_pipeline_rectifier(dewarper, calls), pipeline=other
    )
    assert len(calls) == 2 and again["counts"]["kept"] == 2
    with pytest.raises(FlattenerError, match="another pipeline"):
        rectify_batch(dewarper, inputs, tmp_path)
    assert prediction_record(tmp_path)["pipeline"] == other


def test_batch_workers_write_the_images_of_one_worker_in_input_order(tmp_path):
    dewarper = _tilted_dewarper()
    images = [
        np.random.default_rng(seed).integers(0, 256, (150, 200, 3), np.uint8) for seed in range(6)
    ]
    one = rectify_batch(dewarper, _inputs(images), tmp_path / "one", force_render=True, rerun=1)
    many = rectify_batch(
        dewarper, _inputs(images), tmp_path / "many", force_render=True, rerun=1, workers=3
    )
    strip = ("seconds", "created_utc")
    assert [item["output"] for item in many["items"]] == [item["output"] for item in one["items"]]
    for a, b in zip(one["items"], many["items"], strict=True):
        assert {k: v for k, v in a.items() if k not in strip} == {
            k: v for k, v in b.items() if k not in strip
        }
    for index in range(len(images)):
        with Image.open(tmp_path / "one" / f"{index}.png") as a:
            with Image.open(tmp_path / "many" / f"{index}.png") as b:
                assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    with pytest.raises(FlattenerError, match="workers"):
        rectify_batch(
            dewarper,
            _inputs(images),
            tmp_path / "other",
            rectifier=_pipeline_rectifier(dewarper, []),
            pipeline={"name": "flattener.scan", "passes": 1},
            workers=2,
        )


def test_flip_tta_averages_in_the_mirror_and_is_mirror_symmetric():
    torch.manual_seed(0)
    dewarper = Dewarper(UVDocNet(TINY).eval(), TINY, torch.device("cpu"))
    flipped = with_tta(dewarper, "flip")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(80, 60, 3), dtype=np.uint8)
    mirror = np.ascontiguousarray(image[:, ::-1])
    plain, plain_mirror = predict_grid(dewarper, image), predict_grid(dewarper, mirror)
    turned_back = plain_mirror[:, ::-1] * np.array([-1.0, 1.0], np.float32)
    np.testing.assert_allclose(predict_grid(flipped, image), (plain + turned_back) / 2, atol=1e-6)
    # The averaged grid of the mirror image is the mirrored averaged grid.
    both = predict_grid(flipped, mirror)[:, ::-1] * np.array([-1.0, 1.0], np.float32)
    np.testing.assert_allclose(both, predict_grid(flipped, image), atol=1e-6)
    assert "tta" not in prediction_evidence(dewarper, "round")
    assert prediction_evidence(flipped, "round")["tta"] == "flip"
    with pytest.raises(FlattenerError):
        with_tta(dewarper, "rotate")
