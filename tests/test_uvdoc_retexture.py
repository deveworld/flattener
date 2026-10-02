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


def _doc3d_geometry():
    """A flat page in a Doc3D-sized render: nodes at origin + A @ (u, v), 45x31 of them."""
    origin = np.array([110.0, 40.0])
    axes = np.array([[230.0, 40.0], [-30.0, 330.0]])
    v, u = np.meshgrid(np.linspace(0, 1, 45), np.linspace(0, 1, 31), indexing="ij")
    grid = origin + np.stack([u, v], -1) @ axes.T
    grid3d = np.stack([np.zeros_like(u), u * 1.6 - 0.8, v * 2.2 - 1.1], -1)
    return {"grid": grid, "grid3d": grid3d}


def test_a_doc3d_page_flattens_back_through_its_map():
    tool = _tool()
    texture = _texture()
    settings = {"photo_hw": [240, 180], "shade": [1.0, 1.0], "margin": [0.05, 0.1]}

    def background(size):
        return np.full((*size, 3), 90, np.uint8)

    for seed in range(6):
        photo, forward = tool.render_doc3d_item(
            _doc3d_geometry(), texture, background, np.random.default_rng(seed), settings
        )
        assert photo.shape == (240, 180, 3) and forward.shape == (177, 121, 2)
        flat = _unwarp(photo, forward, (160, 120)).mean(-1)[8:-8, 8:-8]
        page = texture.astype(np.float32).mean(-1)[8:-8, 8:-8]
        same = np.corrcoef(flat.ravel(), page.ravel())[0, 1]
        mirrored = np.corrcoef(flat.ravel(), page[:, ::-1].ravel())[0, 1]
        assert same > 0.9 and same > mirrored + 0.3, (seed, same, mirrored)


def test_a_page_turned_over_in_view_is_found():
    tool = _tool()
    flat = torch.from_numpy(_doc3d_geometry()["grid"]).float()
    assert not tool.folded(flat)
    over = flat.clone()
    over[:, 20:, 0] = 2 * over[:, 19:20, 0] - over[:, 20:, 0]  # the right part folds back
    assert tool.folded(over)


def test_renders_of_one_doc3d_mesh_share_a_split():
    tool = _tool()
    assert tool.doc3d_group("1/1000_4-pp_Page_847-Pwi0001") == "1000"
    assert tool.doc3d_group("7/1_106_1-pp_Page_361-mTF0001") == "1_106"
    ids = [f"{f}/{m}_{k}-tc_Page_{k:03d}-x0001" for f in (1, 2) for m in range(60) for k in (1, 2)]
    split = tool.split_captures(ids, {"name": "d", "val_share": 0.2}, tool.doc3d_group)
    meshes = {side: {tool.doc3d_group(i) for i in split[side]} for side in ("train", "val")}
    assert meshes["val"] and not meshes["train"] & meshes["val"]


def test_cleaning_divides_a_render_s_light_out_of_the_page():
    tool = _tool()
    page = np.full((256, 180, 3), 235.0, np.float32)  # white paper with lines of text
    rng = np.random.default_rng(0)
    for row in range(12, 244, 10):
        for start in range(10, 160, 30):
            page[row : row + 4, start : start + int(rng.integers(10, 26))] = 30
    light = 0.5 + 0.45 * np.linspace(0, 1, 256)[:, None, None]  # dark at the top
    lit = (page * light).clip(0, 255).astype(np.uint8)
    clean = tool.clean_texture(lit).astype(np.float32)
    paper = page[..., 0] > 100
    before = lit[..., 0][:40][paper[:40]].mean() / lit[..., 0][-40:][paper[-40:]].mean()
    after = clean[..., 0][:40][paper[:40]].mean() / clean[..., 0][-40:][paper[-40:]].mean()
    assert before < 0.6 and after > 0.9, (before, after)
    ink = clean[..., 0][~paper].mean()
    assert ink < 0.4 * clean[..., 0][paper].mean()  # the text stays dark


def test_pattern_backgrounds_are_busy_and_vary():
    tool = _tool()
    drawn = [tool.pattern_background(np.random.default_rng(seed), (90, 60)) for seed in range(8)]
    assert all(d.shape == (90, 60, 3) and d.dtype == np.uint8 for d in drawn)
    assert all(d.astype(np.float32).std() > 5 for d in drawn)
    assert len({d.tobytes() for d in drawn}) == 8
