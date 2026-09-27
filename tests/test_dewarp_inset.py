"""Grid-level inset of the predicted outline (``flattener.dewarp.predict --inset``)."""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from flattener import FlattenerError, RenderConfig
from flattener.benchmark import uvdoc
from flattener.benchmark.reporting import prediction_record, report_evidence
from flattener.contracts import Status
from flattener.dewarp import predict
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import center_positions, edge_positions, grid_to_map, uvdoc_dense_map
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import (
    EXIT_FAILED,
    INSET_PRESETS,
    Dewarper,
    dense_backward_map,
    inset_label,
    inset_record,
    main,
    parse_inset,
    rectify,
    resolve_inset,
)

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
PRESET = "syntheticdoc_half_cell"


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


def test_zero_inset_is_the_plain_map():
    grid = np.random.default_rng(0).uniform(-1.1, 1.1, (7, 5, 2)).astype(np.float32)
    for inset in (None, 0, 0.0, (0, 0), [0.0, 0.0]):
        assert resolve_inset(inset) == (0.0, 0.0)
        np.testing.assert_array_equal(
            dense_backward_map(grid, "flattener", (31, 23), (40, 30), inset),
            grid_to_map(grid, (31, 23)),
        )
        np.testing.assert_array_equal(
            dense_backward_map(grid, "uvdoc", (31, 23), (40, 30), inset),
            uvdoc_dense_map(grid, (31, 23), (40, 30)),
        )
    # The inset path at e -> 0 reads the grid where the plain map does, in both conventions.
    for convention in ("flattener", "uvdoc"):
        plain = dense_backward_map(grid, convention, (31, 23), (40, 30))
        tiny = dense_backward_map(grid, convention, (31, 23), (40, 30), (1e-12, 0.0))
        np.testing.assert_allclose(tiny, plain, rtol=0, atol=1e-6)


def test_zero_inset_rectify_is_unchanged():
    image = np.random.default_rng(1).integers(0, 256, (37, 29, 3), dtype=np.uint8)
    dewarper = Dewarper(_IdentityGrid(), TINY, torch.device("cpu"))
    plain = rectify(dewarper, image)
    np.testing.assert_array_equal(plain.image, image)
    for inset in (0, (0.0, 0.0)):
        result = rectify(dewarper, image, inset=inset)
        np.testing.assert_array_equal(result.image, plain.image)
        assert result.metadata["validation"] == plain.metadata["validation"]
        assert result.metadata["dewarp"]["inset"] == {"x": 0.0, "y": 0.0, "preset": None}


@pytest.mark.parametrize("convention", ["flattener", "uvdoc"])
@pytest.mark.parametrize("inset", [(0.1, 0.1), (0.02, 0.3), PRESET])
def test_identity_grid_edges_move_to_one_minus_e(convention, inset):
    ex, ey = resolve_inset(inset)
    height, width = 41, 27
    grid = _identity(6, 5)  # align_corners values for "uvdoc", flattener values otherwise
    dense = dense_backward_map(grid, convention, (height, width), (height, width), inset)
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


def test_syntheticdoc_half_cell_is_the_align_corners_resize_of_its_121x177_map():
    # SyntheticDoc predicts a center-aligned map of bm_out_size (121, 177): 121 rows (y),
    # 177 columns (x). Value (i, j) belongs to the page position
    # ((2j + 1) / 177 - 1, (2i + 1) / 121 - 1). Inference resizes the map to the image
    # with align_corners=True.
    rows, cols = 121, 177
    xs, ys = np.meshgrid(center_positions(cols), center_positions(rows))
    page = torch.from_numpy(np.stack([xs, ys], 0)).float()[None]
    for height, width in ((1200, 900), (4000, 3000)):
        shown = F.interpolate(page, size=(height, width), mode="bilinear", align_corners=True)
        shown = shown[0].permute(1, 2, 0).double().numpy()
        # The flat page (an identity grid) with the preset inset.
        ours = dense_backward_map(_identity(2, 2), "flattener", (height, width), (1, 1), PRESET)
        # They differ only by the factor Wo / (Wo - 1) of align_corners=True, which
        # vanishes for large outputs: at most 1 / (Wo - 1) at the outer pixels.
        np.testing.assert_allclose(shown[..., 0], ours[..., 0] * width / (width - 1), atol=1e-6)
        np.testing.assert_allclose(shown[..., 1], ours[..., 1] * height / (height - 1), atol=1e-6)
        gap_x = np.abs(shown[..., 0] - ours[..., 0]).max()
        gap_y = np.abs(shown[..., 1] - ours[..., 1]).max()
        assert gap_x <= 1 / (width - 1) and gap_y <= 1 / (height - 1)
    ex, ey = INSET_PRESETS[PRESET]
    assert predict.SYNTHETICDOC_MAP_SIZE == (rows, cols)
    assert (ex, ey) == (1 / cols, 1 / rows)  # half a map cell per side, in [-1, 1] units
    # Per side, in % of the page extent: x 0.282 (width), y 0.413 (height).
    assert round(100 * ex / 2, 3) == 0.282 and round(100 * ey / 2, 3) == 0.413


def test_inset_parsing_and_labels():
    assert parse_inset(PRESET) == PRESET
    assert parse_inset("0.02") == 0.02
    assert parse_inset("0.01,0.03") == (0.01, 0.03)
    for bad in ("-0.1", "1", "abc", "0.1,0.2,0.3", "nan"):
        with pytest.raises(Exception, match="inset|expected"):
            parse_inset(bad)
    for bad in ("other", -0.01, 1.0, (0.1,), (0.1, True), float("nan")):
        with pytest.raises(FlattenerError, match="inset"):
            resolve_inset(bad)
    assert inset_label(None) == inset_label(0) == ""
    assert inset_label(PRESET) == PRESET
    assert inset_label(0.02) == "0.02" and inset_label((0.01, 0.03)) == "0.01x0.03"
    assert inset_record(PRESET) == {"x": 1 / 177, "y": 1 / 121, "preset": PRESET}
    assert inset_record((0.01, 0.03)) == {"x": 0.01, "y": 0.03, "preset": None}


def test_inset_keeps_every_failure_rule():
    image = np.random.default_rng(2).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    dewarper = Dewarper(_FoldGrid(), TINY, torch.device("cpu"))
    for inset in (None, PRESET, 0.2):
        with pytest.raises(FlattenerError) as error:  # the checked path rejects the fold
            rectify(dewarper, image, inset=inset)
        assert error.value.status == Status.INVALID_TRANSFORM
        returned = rectify(
            dewarper, image, render_config=RenderConfig(on_error="return"), inset=inset
        )
        assert returned.status == Status.INVALID_TRANSFORM and returned.image is None
        assert returned.reason_code == "fold_or_degenerate"
        original = rectify(
            dewarper, image, render_config=RenderConfig(on_error="original"), inset=inset
        )
        assert original.used_fallback and np.array_equal(original.image, image)
        forced = rectify(dewarper, image, allow_unchecked=True, inset=inset)
        assert forced.status == Status.INVALID_TRANSFORM and forced.used_fallback
        assert forced.reason_code == "fold_or_degenerate" and forced.metadata["forced_render"]
        assert forced.metadata["dewarp"]["inset"] == inset_record(inset)
    with pytest.raises(FlattenerError, match="inset"):
        rectify(dewarper, image, inset=1.5)


def _identity_grid(dewarper, image):
    height, width = dewarper.config.grid_size
    return _identity(height, width)


def _folding_grid(dewarper, image):
    return _identity_grid(dewarper, image)[:, [0, 2, 1, 3]]


def _checkpoint(path):
    torch.manual_seed(0)
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


def _docunet(tmp_path):
    crops, output = tmp_path / "crop", tmp_path / "out"
    crops.mkdir()
    pixels = np.random.default_rng(3).integers(0, 256, (45, 33, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(crops / "1_1 copy.png")
    weights = tmp_path / "model.pt"
    _checkpoint(weights)
    arguments = ["--weights", str(weights), "docunet", "--input-dir", str(crops)]
    return [*arguments, "--output-dir", str(output)], output


def test_docunet_batch_never_keeps_outputs_of_another_inset(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _identity_grid)
    arguments, output = _docunet(tmp_path)
    record_path = output / "predictions.json"

    def run(*inset):
        assert main([*arguments, *inset]) == 0
        counts = json.loads(capsys.readouterr().out)
        record = json.loads(record_path.read_text())
        assert record["inset"] == record["evidence"]["inset"]
        return counts["written"], counts["kept"], record["inset"]

    none = {"x": 0.0, "y": 0.0, "preset": None}
    assert run() == (1, 0, none)
    assert run("--inset", "0") == (0, 1, none)
    assert run("--inset", PRESET) == (1, 0, inset_record(PRESET))  # never kept across insets
    image = np.asarray(Image.open(output / "1_1.png"))
    assert run("--inset", PRESET) == (0, 1, inset_record(PRESET))
    found = prediction_record(output)
    assert found["inset"] == report_evidence(found)["inset"] == inset_record(PRESET)
    assert run("--inset", "0.01,0.02")[:2] == (1, 0)
    assert not np.array_equal(np.asarray(Image.open(output / "1_1.png")), image)
    assert run()[:2] == (1, 0)
    # A record without the field predates the option; its images have no inset.
    record = json.loads(record_path.read_text())
    del record["inset"]
    record_path.write_text(json.dumps(record))
    assert prediction_record(output)["inset"] == none
    assert run()[:2] == (0, 1)
    record = json.loads(record_path.read_text())
    del record["inset"]
    record_path.write_text(json.dumps(record))
    assert run("--inset", PRESET)[:2] == (1, 0)


def test_cli_failures_still_fail_with_an_inset(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _folding_grid)
    arguments, output = _docunet(tmp_path)
    assert main([*arguments, "--inset", PRESET]) == EXIT_FAILED
    assert not (output / "1_1.png").exists()
    assert json.loads(capsys.readouterr().out)["failed"] == 1
    assert main([*arguments, "--inset", PRESET, "--force-render"]) == EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["forced"] == 1
    [item] = json.loads((output / "predictions.json").read_text())["items"]
    assert item["status"] == Status.INVALID_TRANSFORM.value and item["forced"]
    source = tmp_path / "crop" / "1_1 copy.png"
    target = tmp_path / "single.png"
    single = ["--weights", arguments[1], "image", "--input", str(source), "--output", str(target)]
    assert main([*single, "--inset", PRESET]) == EXIT_FAILED and not target.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == Status.INVALID_TRANSFORM.value
    assert printed["inset"] == printed["evidence"]["inset"] == inset_record(PRESET)
    with pytest.raises(SystemExit):
        main([*single, "--inset", "-0.5"])


def test_uvdoc_unwarp_applies_the_inset_in_both_conventions():
    image = np.random.default_rng(4).integers(0, 256, (23, 17, 3), np.uint8)
    identity = _identity(5, 4)
    for convention in ("uvdoc", "flattener"):
        plain = uvdoc.unwarp(image, identity, convention)
        assert np.array_equal(uvdoc.unwarp(image, identity, convention, 0), plain)
    size = image.shape[:2]
    dense = dense_backward_map(identity, "flattener", size, size, PRESET)
    tensor = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float()[None]
    sampled = F.grid_sample(
        tensor, torch.from_numpy(dense)[None], padding_mode="zeros", align_corners=False
    )
    expected = (sampled[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    flattener = uvdoc.unwarp(image, identity, "flattener", PRESET)
    assert np.array_equal(flattener, expected)
    # An identity grid in UVDoc's align_corners values shows the same page region.
    difference = uvdoc.unwarp(image, identity, "uvdoc", PRESET).astype(int) - flattener
    assert np.abs(difference).max() <= 1
