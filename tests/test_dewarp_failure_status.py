"""Codex review F1: a failed geometry check keeps its status through the dewarp CLI.

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
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.model import DewarpModelConfig, UVDocNet, sha256_file
from flattener.dewarp.predict import Dewarper, main, rectify

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
            "data_spec_sha256": "0" * 64,
            "code": {"sha256": "1" * 64},
            "model_state": model.state_dict(),
        },
    )
    return path


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


def _records_invalid_transform(item):
    return Status.INVALID_TRANSFORM.value in {str(value) for value in item.values()}


def test_forced_rectify_keeps_the_invalid_transform_status():
    image = np.random.default_rng(0).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    dewarper = Dewarper(_FoldGrid(), TINY, torch.device("cpu"))
    with pytest.raises(FlattenerError) as error:  # the checked path rejects the fold
        rectify(dewarper, image)
    assert error.value.status == Status.INVALID_TRANSFORM
    forced = rectify(dewarper, image, allow_unchecked=True)
    assert forced.image is not None and forced.used_fallback
    assert forced.status == Status.INVALID_TRANSFORM
    assert forced.reason_code == "fold_or_degenerate"


def test_image_cli_fails_on_a_degenerate_map_by_default(tmp_path, capsys):
    weights = _degenerate_checkpoint(tmp_path / "model.pt")
    source, output = _photo(tmp_path / "in.png"), tmp_path / "out.png"
    argv = ["--weights", str(weights), "image", "--input", str(source), "--output", str(output)]
    code, error = _exit_code(argv)
    captured = capsys.readouterr()
    assert code != 0
    assert not output.exists()  # no image is written for a failed check by default
    shown = captured.out + captured.err + ("" if error is None else str(error.status))
    assert Status.INVALID_TRANSFORM.value in shown


def test_image_cli_force_render_writes_the_image_and_still_fails(tmp_path, capsys):
    weights = _degenerate_checkpoint(tmp_path / "model.pt")
    source, output = _photo(tmp_path / "in.png"), tmp_path / "out.png"
    argv = ["--weights", str(weights), "image", "--input", str(source), "--output", str(output)]
    code = main([*argv, "--force-render"])
    assert code != 0 and output.is_file()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == Status.INVALID_TRANSFORM.value


def _docunet_args(tmp_path):
    crops = tmp_path / "crop"
    crops.mkdir()
    _photo(crops / "1_1 copy.png")
    weights = _degenerate_checkpoint(tmp_path / "model.pt")
    output = tmp_path / "out"
    argv = ["--weights", str(weights), "docunet", "--input-dir", str(crops)]
    return [*argv, "--output-dir", str(output)], output


def test_docunet_cli_records_failures_and_exits_non_zero(tmp_path, capsys):
    argv, output = _docunet_args(tmp_path)
    code, _ = _exit_code(argv)
    assert code != 0
    assert not (output / "1_1.png").exists()
    [item] = json.loads((output / "predictions.json").read_text())["items"]
    assert _records_invalid_transform(item)
    assert json.loads(capsys.readouterr().out)["failed"] == 1


def test_docunet_cli_force_render_keeps_the_failure_on_every_run(tmp_path, capsys):
    argv, output = _docunet_args(tmp_path)
    assert main([*argv, "--force-render"]) != 0
    assert (output / "1_1.png").is_file()
    [item] = json.loads((output / "predictions.json").read_text())["items"]
    assert _records_invalid_transform(item)
    assert json.loads(capsys.readouterr().out)["failed"] == 1
    # A later run keeps the forced image; its recorded failure stays and counts again.
    code, _ = _exit_code([*argv, "--force-render"])
    assert code != 0
    [kept] = json.loads((output / "predictions.json").read_text())["items"]
    assert _records_invalid_transform(kept)
    assert json.loads(capsys.readouterr().out)["failed"] == 1


def test_checked_rectify_follows_the_failure_policy():
    image = np.random.default_rng(0).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    dewarper = Dewarper(_FoldGrid(), TINY, torch.device("cpu"))
    returned = rectify(dewarper, image, render_config=RenderConfig(on_error="return"))
    assert returned.status == Status.INVALID_TRANSFORM and returned.image is None
    assert returned.metadata["validation"]["failed_cells"] > 0
    original = rectify(dewarper, image, render_config=RenderConfig(on_error="original"))
    assert original.status == Status.INVALID_TRANSFORM and original.used_fallback
    np.testing.assert_array_equal(original.image, image)
    assert "forced_render" not in original.metadata


def test_docunet_cli_refuses_predictions_of_the_older_format(tmp_path):
    argv, output = _docunet_args(tmp_path)
    output.mkdir()
    old = {"schema": "flattener-dewarp-docunet-1", "items": []}
    old["model"] = {"sha256": sha256_file(tmp_path / "model.pt")}
    (output / "predictions.json").write_text(json.dumps(old))
    with pytest.raises(FlattenerError, match="older format"):
        main(argv)
    assert main([*argv, "--overwrite", "--force-render"]) != 0
