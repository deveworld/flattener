"""Shared test options and fixtures.

Tests marked ``training`` run only with ``--run-training``. ``tiny_docunet`` is a small
DocUNet-shaped set of scans and predictions for the benchmark tests.
"""

import numpy as np
import pytest
from PIL import Image


def pytest_addoption(parser):
    parser.addoption(
        "--run-training",
        action="store_true",
        default=False,
        help="Explicitly allow tests marked training (may perform CPU training).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-training"):
        return
    skip_training = pytest.mark.skip(reason="training requires explicit --run-training")
    for item in items:
        if item.get_closest_marker("training") is not None:
            item.add_marker(skip_training)


@pytest.fixture
def tiny_docunet(tmp_path, monkeypatch):
    """65 tiny scans and 130 predictions; the metric area is shrunk to keep it fast."""
    from tools.benchmark import metrics

    monkeypatch.setattr(metrics, "METRIC_AREA", 300)
    rng = np.random.default_rng(7)
    scans = tmp_path / "scans"
    predictions = tmp_path / "pred"
    scans.mkdir()
    predictions.mkdir()
    for k in range(1, 66):
        scan = rng.integers(0, 256, size=(24, 18, 4), dtype=np.uint8)
        Image.fromarray(scan, "RGBA").save(scans / f"docunet-reference-{k}.png")
        for m in (1, 2):
            prediction = rng.integers(0, 256, size=(20, 16, 3), dtype=np.uint8)
            Image.fromarray(prediction).save(predictions / f"{k}_{m}.png")
    return scans, predictions
