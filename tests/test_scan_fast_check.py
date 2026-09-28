"""``scan.fast_check``: the renderer's decisions from the grid, a proxy render, full sampling."""

import numpy as np
import pytest
from scan_fakes import twist

from flattener.contracts import RenderConfig, Status
from flattener.dewarp.coords import edge_positions
from flattener.renderer import render
from flattener.scan.fast_check import fast_check, grid_jacobian
from flattener.scan.placement import dense_map

CONFIG = RenderConfig(on_error="return")


def _grid(rng, rows=9, cols=7, noise=0.02, scale=0.8):
    nodes = np.stack(np.meshgrid(edge_positions(cols), edge_positions(rows)), -1) * scale
    return nodes + rng.normal(0, noise, nodes.shape)


def _image(seed=0, size=(240, 320)):
    return np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)


def test_the_grid_gives_the_renderers_fold_decision_and_minimum():
    rng = np.random.default_rng(11)
    image = _image()
    loose = RenderConfig(on_error="return", max_out_of_bounds_fraction=1.0)
    folds = 0
    for nodes in ["flattener", "uvdoc"] * 40:
        rows, cols = int(rng.integers(3, 12)), int(rng.integers(3, 10))
        grid = _grid(rng, rows, cols, rng.choice([0.01, 0.06]))
        size = (int(rng.choice([2, 3, 6, 17, 90, 240])), int(rng.choice([2, 5, 13, 64, 200])))
        inset, border = float(rng.choice([0.0, 0.02])), float(rng.choice([0.0, 0.05, 0.2]))
        dense = dense_map(grid, size, inset=inset, border=border, nodes=nodes)
        expected = render(image, dense, config=loose).metadata["validation"]
        found = grid_jacobian(grid, size, inset=inset, border=border, nodes=nodes)
        assert found["fold_check"] == expected["fold_check"]
        folds += found["fold_check"] == "failed"
        if expected["min_jacobian_ratio"] is None:
            assert found["min_jacobian_ratio"] is None
        else:
            assert found["min_jacobian_ratio"] == pytest.approx(
                expected["min_jacobian_ratio"], rel=1e-3, abs=1e-9
            )
    assert 0 < folds < 80  # both decisions occur


def test_decisions_and_reasons_match_render_in_order():
    rng = np.random.default_rng(5)
    image = _image(2)
    smooth = _grid(rng, 45, 31, noise=0.0005)
    shifted = smooth.copy()
    shifted[..., 0] += 0.6
    folded = twist(smooth)
    broken = smooth.copy()
    broken[3, 3] = np.nan
    outside = smooth + 5.0
    cases = [
        (smooth, CONFIG, Status.SUCCESS, None),
        (shifted, CONFIG, Status.INVALID_TRANSFORM, "excessive_out_of_bounds"),
        (folded, CONFIG, Status.INVALID_TRANSFORM, "fold_or_degenerate"),
        (broken, CONFIG, Status.INVALID_TRANSFORM, "non_finite_grid"),
        (outside, CONFIG, Status.INVALID_TRANSFORM, "excessive_out_of_bounds"),
        (
            outside,
            RenderConfig(on_error="return", max_out_of_bounds_fraction=1.0),
            Status.INVALID_TRANSFORM,
            "empty_valid_region",
        ),
    ]
    for grid, config, status, reason in cases:
        size = (230, 170)
        found = fast_check(image, grid, size, config=config)
        expected = render(image, dense_map(grid, size), config=config)
        assert (found.status, found.reason_code) == (status, reason)
        assert (expected.status, expected.reason_code) == (status, reason)
        result = found.result()
        assert result.image is None and (result.valid_mask is None) == (status != Status.SUCCESS)
