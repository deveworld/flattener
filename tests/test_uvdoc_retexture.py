"""tools/uvdoc/retexture.py on a synthetic capture: a flat page under a known affine map."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "uvdoc_retexture", ROOT / "tools/uvdoc/retexture.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture(height=300, width=400):
    """A page whose texture point (u, v) sits at pixel origin + A @ (u, v), pixel centers
    at integers, as the raw captures store it."""
    origin = np.array([80.0, 40.0])
    axes = np.array([[220.0, 30.0], [-20.0, 210.0]])  # columns: u and v directions
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    uv = np.stack([x - origin[0], y - origin[1]], -1) @ np.linalg.inv(axes).T
    outside = (uv < 0).any(-1) | (uv > 1).any(-1)
    uv[outside] = np.nan
    v, u = np.meshgrid(np.linspace(0, 1, 89), np.linspace(0, 1, 61), indexing="ij")
    grid = origin + np.stack([u, v], -1) @ axes.T  # (89, 61, 2)
    photo = np.full((height, width, 3), 200, np.uint8)
    return {"photo": photo, "uv": uv.astype(np.float32), "seg": ~outside, "grid": grid}


def _texture():
    y, x = np.mgrid[0:160, 0:120]
    pattern = ((x // 12 + y // 9) % 2) * 160 + x + y // 2  # asymmetric, so mirroring shows
    return np.stack([pattern, 255 - pattern // 2, (x * 2) % 255], -1).clip(0, 255).astype(np.uint8)


def _unwarp(photo, forward, size):
    source = torch.from_numpy(photo.copy()).permute(2, 0, 1)[None].float()
    grid = torch.from_numpy(forward.astype(np.float32)).permute(2, 0, 1)[None]
    dense = F.interpolate(grid, size=size, mode="bilinear", align_corners=False)
    flat = F.grid_sample(source, dense.permute(0, 2, 3, 1), align_corners=False)
    return flat[0].permute(1, 2, 0).numpy()


def test_the_stored_map_flattens_the_composed_photo_back_to_the_page():
    tool = _tool()
    capture, texture = _capture(), _texture()
    settings = {"photo_hw": [240, 180], "shade": [1.0, 1.0], "margin": [0.05, 0.1]}
    seen = set()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        horizontal, vertical = rng.random() < 0.5, rng.random() < 0.5
        seen.add((horizontal, vertical))
        photo, forward = tool.render_item(
            capture, texture, None, np.random.default_rng(seed), settings
        )
        assert photo.shape == (240, 180, 3) and forward.shape == (177, 121, 2)
        flat = _unwarp(photo, forward, (160, 120))
        expected = texture.astype(np.float32) * 200 / 255  # the page times the paper's light
        inner = (slice(8, -8), slice(8, -8))
        error = np.abs(flat[inner] - expected[inner]).mean()
        assert error < 12, (seed, error)  # a mirrored page would differ by far more
    assert len(seen) == 4  # every flip combination came up


def test_benchmark_captures_stay_out(tmp_path):
    tool = _tool()
    raw = tmp_path / "raw/samples/rgb"
    raw.mkdir(parents=True)
    for capture in ("09_00000_1", "09_00001_1", "03_00007_1"):
        (raw / f"{capture}.png").write_bytes(b"")
    bench = tmp_path / "bench/metadata_geom"
    bench.mkdir(parents=True)
    (bench / "09_00000_1_0_0.json").write_text('{"sample_id": "09_00000_1"}')
    assert tool.captures(tmp_path / "raw", tmp_path / "bench") == ["03_00007_1", "09_00001_1"]


def test_builds_naming_one_split_hold_out_the_same_captures():
    tool = _tool()
    ids = [f"{session:02d}_{n:05d}_1" for session in range(4) for n in range(100)]
    first = tool.split_captures(ids, {"name": "a"})
    assert 0 < len(first["val"]) < 60 and len(first["val"]) + len(first["train"]) == len(ids)
    assert tool.split_captures(ids, {"name": "b", "split_name": "a"}) == first
    assert tool.split_captures(ids, {"name": "b"})["val"] != first["val"]
