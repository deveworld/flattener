"""Regressions for accepted numeric settings and rejected configuration logging."""

import json
import threading
from fractions import Fraction

import numpy as np
import pytest

from flattener import RenderConfig, Status, render


@pytest.mark.parametrize("fraction", [np.longdouble("0.25"), Fraction(1, 4), np.float32(0.25)])
def test_accepted_real_settings_produce_strict_json(fraction):
    result = render(
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((1, 1, 2), dtype=np.float32),
        config=RenderConfig(max_out_of_bounds_fraction=fraction),
    )
    saved = json.loads(json.dumps(result.metadata, allow_nan=False))
    assert result.status == Status.SUCCESS
    assert saved["config"]["max_out_of_bounds_fraction"] == 0.25


def test_invalid_non_copyable_config_value_uses_failure_policy():
    result = render(
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((1, 1, 2), dtype=np.float32),
        config=RenderConfig(fill_rgb=threading.Lock(), on_error="return"),
    )
    assert result.status == Status.INPUT_ERROR
    assert result.reason_code == "invalid_config"
    assert result.image is None
    json.dumps(result.metadata, allow_nan=False)


def test_early_rejection_does_not_claim_fold_check_was_inapplicable():
    grid = np.array([[[2.0, 0.0], [3.0, 0.0]], [[2.0, 0.5], [3.0, 0.5]]], dtype=np.float32)
    result = render(
        np.zeros((2, 2, 3), dtype=np.uint8), grid, config=RenderConfig(on_error="return")
    )
    assert result.reason_code == "excessive_out_of_bounds"
    assert result.metadata["validation"]["fold_check"] is None
    assert result.metadata["validation"]["checked_cells"] == 0
