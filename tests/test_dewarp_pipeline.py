"""``crop_pass_box``, ``rerun_box`` and ``rectify_batch`` with another pipeline."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError, RenderConfig
from flattener.benchmark.reporting import prediction_record
from flattener.contracts import RenderResult, Status
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import (
    DEFAULT_PIPELINE,
    DESKEW_MAX_DEG,
    BatchInput,
    Dewarper,
    _flattener_values,
    canvas_input,
    canvas_size,
    canvas_to_source,
    crop_box,
    crop_pass,
    crop_pass_box,
    docunet_inputs,
    estimate_page_angle,
    outline_points,
    predict_grid,
    predict_input,
    rectify,
    rectify_batch,
    rerun_box,
    rerun_grid,
)

DOCUNET_CROPS = Path(
    os.environ.get("FLATTENER_DOCUNET_CROP", "~/.cache/flattener-sota-20260926/crop/crop")
).expanduser()


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


def _dewarper(config=None, seed=0, sha="seeded-uvdocnet"):
    config = config or DewarpModelConfig(input_size=(96, 72), grid_size=(9, 7), num_filter=4)
    info = {"sha256": sha, "kind": "checkpoint", "step": 1, "track": "release"}
    return Dewarper(Tilted(config, seed).eval(), config, torch.device("cpu"), info)


def _image(seed=3, size=(150, 200)):
    return np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)


def _old_crop_pass(dewarper, image, grid, deskew):
    """``crop_pass`` as it was written before ``crop_pass_box`` was factored out."""
    source_size = image.shape[:2]
    box = crop_box(outline_points(grid, source_size), source_size)
    left, top, right, bottom = box
    angle = 0.0
    if deskew:
        estimate, reason = estimate_page_angle(grid, source_size)
        if reason is None and abs(estimate) <= DESKEW_MAX_DEG:
            angle = estimate
    crop = image[top:bottom, left:right]
    canvas = canvas_size(box, angle)
    if angle == 0.0:
        raw = predict_grid(dewarper, crop)
    else:
        raw = predict_input(dewarper, canvas_input(dewarper, crop, angle, canvas))
    local = _flattener_values(raw, dewarper.config.grid_convention, canvas)
    matrix, offset = canvas_to_source(box, angle, canvas, source_size)
    return local @ matrix.T + offset


def _old_rerun_grid(dewarper, image, rerun, deskew):
    grid = _flattener_values(
        predict_grid(dewarper, image), dewarper.config.grid_convention, image.shape[:2]
    )
    for _ in range(rerun):
        grid = _old_crop_pass(dewarper, image, grid, deskew)
    return grid


@pytest.mark.parametrize("deskew", [False, True])
def test_crop_pass_is_rerun_box_then_crop_pass_box(deskew):
    dewarper, image = _dewarper(), _image()
    whole, _ = rerun_grid(dewarper, image, 0)
    box, angle, record = rerun_box(whole, image.shape[:2], deskew=deskew)
    assert record["skipped"] is None and list(box) == record["box"]
    assert (angle != 0.0) == deskew
    grid, placed = crop_pass_box(dewarper, image, box, angle)
    assert placed == {"rotated": deskew, "canvas_size": list(canvas_size(box, angle))}
    again, full = crop_pass(dewarper, image, whole, deskew)
    assert again.tobytes() == grid.tobytes()
    assert full == {**record, **placed}
    assert grid.tobytes() == _old_crop_pass(dewarper, image, whole, deskew).tobytes()


def test_rerun_box_limits_and_skips():
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


@pytest.mark.parametrize("rerun, deskew", [(0, False), (1, False), (1, True), (2, True)])
def test_rerun_grid_is_unchanged_by_the_refactor(rerun, deskew):
    dewarper, image = _dewarper(seed=1), _image(seed=4)
    grid, _ = rerun_grid(dewarper, image, rerun, deskew)
    assert grid.tobytes() == _old_rerun_grid(dewarper, image, rerun, deskew).tobytes()


def _inputs(images):
    return [
        BatchInput(f"{index}.png", f"{index} copy.png", lambda image=image: image)
        for index, image in enumerate(images)
    ]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("rerun, deskew", [(0, False), (1, False), (1, True)])
def test_the_batch_writes_what_rectify_renders(tmp_path, rerun, deskew):
    dewarper = _dewarper()
    images = [_image(seed) for seed in (5, 6, 7)]
    report = rectify_batch(
        dewarper, _inputs(images), tmp_path, force_render=True, rerun=rerun, deskew=deskew
    )
    assert report["pipeline"] == DEFAULT_PIPELINE
    assert report["evidence"]["pipeline"] == DEFAULT_PIPELINE
    for index, image in enumerate(images):
        result = rectify(
            dewarper,
            image,
            render_config=RenderConfig(on_error="return"),
            allow_unchecked=True,
            rerun=rerun,
            deskew=deskew,
        )
        with Image.open(tmp_path / f"{index}.png") as written:
            assert np.asarray(written).tobytes() == result.image.tobytes()
        item = report["items"][index]
        assert item["status"] == str(result.status)
        assert item["crops"] == json.loads(json.dumps(result.metadata["dewarp"]["crops"]))


def _pipeline_rectifier(dewarper, calls):
    def rectify_one(image, output_size):
        calls.append(output_size)
        result = rectify(
            dewarper, image, output_size, render_config=RenderConfig(on_error="return")
        )
        result.metadata["batch_item"] = {"outcome": "rectified", "status": "ignored"}
        return result

    return rectify_one


def test_predictions_of_another_pipeline_are_refused(tmp_path):
    dewarper = _dewarper()
    inputs = _inputs([_image(8), _image(9)])
    rectify_batch(dewarper, inputs, tmp_path, force_render=True)
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


def test_a_record_without_a_pipeline_was_made_by_rectify(tmp_path):
    dewarper = _dewarper()
    inputs = _inputs([_image(10)])
    rectify_batch(dewarper, inputs, tmp_path, force_render=True)
    path = tmp_path / "predictions.json"
    record = json.loads(path.read_text())
    del record["pipeline"]
    path.write_text(json.dumps(record))
    assert prediction_record(tmp_path)["pipeline"] == DEFAULT_PIPELINE
    report = rectify_batch(dewarper, inputs, tmp_path, force_render=True)
    assert report["counts"]["kept"] == 1
    with pytest.raises(FlattenerError, match="another pipeline"):
        rectify_batch(
            dewarper,
            inputs,
            tmp_path,
            rectifier=lambda image, size: RenderResult(Status.SUCCESS, image=image),
            pipeline={"name": "other"},
        )


def test_invalid_pipeline_settings_are_refused(tmp_path):
    dewarper = _dewarper()
    inputs = _inputs([_image(11)])

    def one(image, size):
        return RenderResult(Status.SUCCESS, image=image)

    for kwargs in (
        {"rectifier": one},
        {"pipeline": {"name": "other"}},
        {"rectifier": one, "pipeline": {"name": ""}},
        {"rectifier": one, "pipeline": {"label": "x"}},
        {"rectifier": one, "pipeline": {"name": "x", "value": object()}},
    ):
        with pytest.raises(FlattenerError) as error:
            rectify_batch(dewarper, inputs, tmp_path / "out", **kwargs)
        assert error.value.status == Status.INPUT_ERROR
    report = rectify_batch(dewarper, inputs, tmp_path / "ok", pipeline={"name": "rectify"})
    assert report["counts"]["items"] == 1


@pytest.mark.skipif(not DOCUNET_CROPS.is_dir(), reason="DocUNet crops are not available")
@pytest.mark.parametrize("rerun, deskew", [(0, False), (1, False), (1, True)])
def test_docunet_crops_through_the_batch_are_byte_identical(tmp_path, rerun, deskew):
    config = DewarpModelConfig(input_size=(712, 488), grid_size=(45, 31), num_filter=8)
    dewarper = _dewarper(config)
    names = ("1_1.png", "10_2.png", "25_1.png")
    inputs = [task for task in docunet_inputs(DOCUNET_CROPS) if task.output in names]
    report = rectify_batch(
        dewarper, inputs, tmp_path, force_render=True, rerun=rerun, deskew=deskew
    )
    assert [item["action"] for item in report["items"]] == ["written"] * 3
    if rerun:
        assert all(item["crops"][0]["rotated"] == deskew for item in report["items"])
    from flattener.dewarp.predict import dense_backward_map, render_unchecked
    from flattener.renderer import render

    for task in inputs:
        image = task.load()
        grid = _old_rerun_grid(dewarper, image, rerun, deskew)
        if rerun == 0:
            dense = dense_backward_map(
                predict_grid(dewarper, image), "flattener", image.shape[:2], image.shape[:2]
            )
        else:
            dense = dense_backward_map(
                grid, "flattener", image.shape[:2], image.shape[:2], flattener_values=True
            )
        checked = render(image, dense, config=RenderConfig(on_error="return"))
        expected = checked.image
        if expected is None:
            expected = render_unchecked(image, dense, (255, 255, 255))
        with Image.open(tmp_path / task.output) as written:
            assert np.asarray(written).tobytes() == expected.tobytes()
