"""``scan.fast_check``: the renderer's decisions from the grid, a proxy render, full sampling."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scan_fakes import Page, oracle_dewarper, ramp_scene, twist

from flattener.contracts import FlattenerError, RenderConfig, Status
from flattener.dewarp.coords import edge_positions
from flattener.dewarp.predict import sample_rgb
from flattener.renderer import render
from flattener.scan import locate
from flattener.scan.fast_check import PROXY_PIXELS, fast_check, grid_jacobian, proxy_size
from flattener.scan.pipeline import Scanner, ScanOptions
from flattener.scan.placement import dense_map

TOOL = Path(__file__).resolve().parents[1] / "tools" / "fast_check_agreement.py"
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
    for _ in range(80):
        rows, cols = int(rng.integers(3, 12)), int(rng.integers(3, 10))
        grid = _grid(rng, rows, cols, rng.choice([0.01, 0.06]))
        size = (int(rng.choice([2, 3, 6, 17, 90, 240])), int(rng.choice([2, 5, 13, 64, 200])))
        inset, border = float(rng.choice([0.0, 0.02])), float(rng.choice([0.0, 0.05, 0.2]))
        dense = dense_map(grid, size, inset=inset, border=border)
        expected = render(image, dense, config=loose).metadata["validation"]
        found = grid_jacobian(grid, size, inset=inset, border=border)
        assert found["fold_check"] == expected["fold_check"]
        folds += found["fold_check"] == "failed"
        if expected["min_jacobian_ratio"] is None:
            assert found["min_jacobian_ratio"] is None
        else:
            assert found["min_jacobian_ratio"] == pytest.approx(
                expected["min_jacobian_ratio"], rel=1e-3, abs=1e-9
            )
    assert 0 < folds < 80  # both decisions occur


def test_a_fold_narrower_than_a_pixel_passes_both_checks_and_is_flagged():
    grid = _grid(np.random.default_rng(3), 45, 31, noise=0.0)
    step = grid[20, 16] - grid[20, 15]
    grid[20, 15] += step * 1.0005  # past its neighbour by 0.05 % of a cell
    size = (900, 700)
    image = _image(1)
    found = fast_check(image, grid, size, config=CONFIG)
    expected = render(image, dense_map(grid, size), config=CONFIG)
    assert found.status == expected.status == Status.SUCCESS
    assert found.validation["continuous_min_jacobian_ratio"] <= CONFIG.min_jacobian_ratio
    assert found.validation["min_jacobian_ratio"] > CONFIG.min_jacobian_ratio


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


def test_the_full_sample_is_the_render_and_the_proxy_is_recorded():
    rng = np.random.default_rng(7)
    grid = _grid(rng, 45, 31, noise=0.001)
    image = _image(3, (600, 800))
    size = (1800, 1400)  # 2.5 MP: the proxy has 1/16 of its pixels
    found = fast_check(image, grid, size, inset=0.01, border=0.02, config=CONFIG)
    expected = render(image, dense_map(grid, size, inset=0.01, border=0.02), config=CONFIG)
    assert found.ok and expected.status == Status.SUCCESS
    assert found.sample(image, CONFIG.fill_rgb).tobytes() == expected.image.tobytes()
    assert np.array_equal(found.valid_mask, expected.valid_mask)
    truncated = found.sample(image, CONFIG.fill_rgb, "truncate")
    dense = dense_map(grid, size, inset=0.01, border=0.02)
    reference = sample_rgb(image, dense, expected.valid_mask, (255,) * 3, "truncate")
    assert truncated.tobytes() == reference.tobytes()
    assert (
        found.validation["out_of_bounds_fraction"]
        == expected.metadata["validation"]["out_of_bounds_fraction"]
    )
    small = proxy_size(size)
    assert small[0] * small[1] <= PROXY_PIXELS and small == (450, 350)
    assert proxy_size((4000, 3000)) == (1000, 750) and proxy_size((8000, 6000)) == (1632, 1224)
    assert found.metadata["check_resolution"]["proxy"] == list(small)
    assert found.metadata["proxy"]["status"] == "success"
    assert found.proxy.image.shape[:2] == small
    rendered = found.result(image)
    assert rendered.image.tobytes() == expected.image.tobytes()
    skipped = fast_check(image, grid, size, config=CONFIG, proxy_pixels=None)
    assert skipped.ok and skipped.proxy is None and "proxy" not in skipped.metadata["timings"]
    with pytest.raises(FlattenerError):
        fast_check(image, grid + 5.0, size, config=CONFIG).sample(image, (255, 255, 255))


def test_the_scanner_with_fast_check_saves_the_same_image(monkeypatch):
    monkeypatch.setitem(locate.THRESHOLDS, "min_page_long_side_px", 50)
    size = (240, 320)
    page = Page((170, 125), (150, 190), 7.0)
    image = ramp_scene(size, page)
    scanner = Scanner(oracle_dewarper(page, size))
    default = scanner.scan(image, ScanOptions(orient="none"))
    planned = scanner.plan(image, ScanOptions(orient="none", fast_check=True))
    assert planned.checked.fast is not None and planned.checked.result.image is None
    fast = scanner.render(image, planned)
    assert fast.image.tobytes() == default.image.tobytes()
    assert fast.plan.plan_sha256 == default.plan.plan_sha256
    assert fast.metadata["renderer"]["check"] == "fast_check"
    assert fast.metadata["renderer"]["validation"]["fold_check"] == "passed"
    proxy = fast.metadata["renderer"]["check_resolution"]["proxy"]
    assert proxy[0] * proxy[1] <= fast.plan.output.height * fast.plan.output.width / 16
    assert scanner.pipeline_record(ScanOptions(fast_check=True)) != scanner.pipeline_record()


def _tool():
    spec = importlib.util.spec_from_file_location("fast_check_agreement", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_agreement_tool_finds_no_disagreement_on_small_synthetic_maps():
    tool = _tool()
    image = _image(4, (300, 400))
    rows = []
    for item in tool.synthetic_maps(9, 24):
        if item["size"][0] * item["size"][1] > 1_500_000:
            item["size"] = (item["size"][0] // 4, item["size"][1] // 4)
        rows.append(tool.compare(image, item))
    summary = tool.summary(rows)
    assert summary["maps"] == 24 and summary["disagree"] == 0
    assert summary["images_identical"] == summary["images_compared"] > 0
    assert {"smooth", "near_fold", "tiny", "out_of_bounds"} <= set(summary["families"])
