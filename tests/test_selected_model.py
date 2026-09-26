"""The shipped selected.pt loads and unwarps with the current code."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from flattener import Status
from flattener.learning.coordinate_study import coordinate_checkpoint, load_coordinate_predictor
from flattener.learning.inference import unwarp

CHECKPOINT = Path(__file__).resolve().parents[1] / "outputs" / "coordinate-fold" / "selected.pt"
SHA256 = "ee7e2cf7a82bc238d1054cb2de44d15dcd007ea215e8c44efa86a4f9da877ea8"

pytestmark = pytest.mark.skipif(not CHECKPOINT.is_file(), reason="selected.pt is not present")


def test_selected_model_loads_and_unwarps():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA256
    saved = coordinate_checkpoint(CHECKPOINT)
    assert saved["schema_version"] == "coordinate-study-0.3"
    assert saved["config"]["architecture"] == "global"

    predictor = load_coordinate_predictor(CHECKPOINT)
    assert predictor.model_sha256 == SHA256
    assert predictor.map_mode == "legacy"

    rows, cols = np.mgrid[0:288, 0:384]
    image = np.stack(
        [(rows * 255) // 287, (cols * 255) // 383, ((rows + cols) * 7) % 256], axis=-1
    ).astype(np.uint8)
    grid = predictor.predict_map(image, (192, 144))
    assert grid.shape == (192, 144, 2)
    assert np.isfinite(grid).all()

    result = unwarp(image, predictor, (192, 144))
    assert result.status == Status.SUCCESS
    assert result.image.shape == (192, 144, 3)
