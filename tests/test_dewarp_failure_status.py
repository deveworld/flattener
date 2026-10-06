"""A failed geometry check keeps its status through the dewarp CLI.

The checkpoint below predicts one constant grid, so every dense map it gives is
degenerate and ``flattener.render`` rejects it with ``invalid_transform``
(``fold_or_degenerate``). The CLI renders through the checked path by default,
exits non-zero when an image fails, and writes an image despite a failed check only
with ``--force-render``, which still records the failure.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError, RenderConfig
from flattener.contracts import Status
from flattener.dewarp import predict
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import EXIT_FAILED, Dewarper, main, rectify

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)


class _FoldGrid(torch.nn.Module):
    """Identity grid with the two middle columns swapped: a reversed Jacobian."""

    def forward(self, x):
        columns = edge_positions(4)[[0, 2, 1, 3]]
        grid = np.stack(np.meshgrid(columns, edge_positions(5)), 0)
        return torch.from_numpy(grid).float()[None].expand(x.shape[0], -1, -1, -1), None


def _degenerate_checkpoint(path):
    """A real checkpoint whose 2-D head outputs one constant node position."""
    torch.manual_seed(0)
    model = UVDocNet(TINY).eval()
    with torch.no_grad():
        last = model.out_point_positions2D[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor([0.1, -0.2]))
    save_checkpoint(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": 1,
            "track": "release",
            "model_config": TINY.to_dict(),
            "lineage": [{"kind": "scratch"}],
            "model_state": model.state_dict(),
        },
    )
    return path


def test_rectify_follows_the_failure_policy_and_checks_the_output_size_first(monkeypatch):
    image = np.random.default_rng(0).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    dewarper = Dewarper(_FoldGrid(), TINY, torch.device("cpu"))
    with pytest.raises(FlattenerError) as error:  # the checked path rejects the fold
        rectify(dewarper, image)
    assert error.value.status == Status.INVALID_TRANSFORM
    returned = rectify(dewarper, image, render_config=RenderConfig(on_error="return"))
    assert returned.status == Status.INVALID_TRANSFORM and returned.image is None
    assert returned.metadata["validation"]["failed_cells"] > 0
    original = rectify(dewarper, image, render_config=RenderConfig(on_error="original"))
    assert original.status == Status.INVALID_TRANSFORM and original.used_fallback
    np.testing.assert_array_equal(original.image, image)
    assert "forced_render" not in original.metadata
    forced = rectify(dewarper, image, allow_unchecked=True)  # an image, still marked failed
    assert forced.image is not None and forced.used_fallback
    assert forced.status == Status.INVALID_TRANSFORM
    assert forced.reason_code == "fold_or_degenerate"
    # An oversized output is refused before the model runs.

    def unexpected(*args, **kwargs):
        pytest.fail("oversized output reached prediction")

    monkeypatch.setattr(predict, "predict_grid", unexpected)
    with pytest.raises(FlattenerError) as error:
        rectify(dewarper, image, (20, 20), render_config=RenderConfig(max_output_pixels=100))
    assert error.value.reason_code == "size_limit_exceeded"


def _photo(path, size=(40, 30)):
    Image.fromarray(np.random.default_rng(3).integers(0, 256, (*size, 3), np.uint8)).save(path)
    return path


def _exit_code(argv):
    """Exit status of the CLI: main's return value, a SystemExit code, or 1 for an error."""
    try:
        return main(argv), None
    except SystemExit as exit:
        return exit.code if isinstance(exit.code, int) else 1, None
    except FlattenerError as error:
        return 1, error


def test_image_cli_fails_on_a_degenerate_map_by_default(tmp_path, capsys):
    weights = _degenerate_checkpoint(tmp_path / "model.pt")
    source, output = _photo(tmp_path / "in.png"), tmp_path / "out.png"
    _photo(output)  # an image of an earlier run
    argv = ["--weights", str(weights), "image", "--input", str(source), "--output", str(output)]
    code, error = _exit_code(argv)
    captured = capsys.readouterr()
    assert code != 0
    assert not output.exists()  # no image for a failed check, and the earlier one is gone
    shown = captured.out + captured.err + ("" if error is None else str(error.status))
    assert Status.INVALID_TRANSFORM.value in shown
    assert json.loads(captured.out)["removed_earlier_output"] is True


def _docunet_args(tmp_path):
    crops = tmp_path / "crop"
    crops.mkdir()
    _photo(crops / "1_1 copy.png")
    weights = _degenerate_checkpoint(tmp_path / "model.pt")
    output = tmp_path / "out"
    argv = ["--weights", str(weights), "docunet", "--input-dir", str(crops)]
    return [*argv, "--output-dir", str(output)], output


def _folds_dark_inputs(dewarper, image):
    """Identity grid for a bright input, the middle columns swapped for a dark one."""
    columns = edge_positions(4)
    if image.mean() < 128:
        columns = columns[[0, 2, 1, 3]]
    return np.stack(np.meshgrid(columns, edge_positions(5)), -1).astype(np.float32)


def test_repeated_batches_render_the_current_input(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(predict, "predict_grid", _folds_dark_inputs)
    argv, output = _docunet_args(tmp_path)
    crop, image = tmp_path / "crop" / "1_1 copy.png", output / "1_1.png"

    def run(*extra):
        code = main([*argv, *extra])
        return code, json.loads(capsys.readouterr().out)

    Image.new("RGB", (30, 40), (200, 200, 200)).save(crop)
    assert run() == (0, {"items": 1, "written": 1, "kept": 0, "failed": 0, "forced": 0})
    assert run()[1]["written"] == 1
    Image.new("RGB", (30, 40), (20, 20, 20)).save(crop)  # same size and name, now failing
    code, counts = run()
    assert code == EXIT_FAILED and counts["kept"] == 0 and counts["failed"] == 1
    assert not image.exists()
    record = json.loads((output / "predictions.json").read_text())
    assert record["complete"] is True
    [item] = record["items"]
    assert item["status"] == Status.INVALID_TRANSFORM.value and item["action"] == "not_written"
    # A run without force removes the previous forced output.
    assert run("--force-render")[1]["forced"] == 1 and image.is_file()
    assert run("--force-render")[1]["written"] == 1
    code, counts = run()
    assert code == EXIT_FAILED and counts["kept"] == 0 and not image.exists()
