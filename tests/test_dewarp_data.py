"""Dewarp data adapters, license tracks and seeded augmentation on tiny fixtures."""

import json
import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from flattener import FlattenerError
from flattener.dewarp.coords import flattener_to_syntheticdoc, pixels_to_flattener
from flattener.dewarp.data import (
    SAMPLE_SCHEMA,
    SPEC_SCHEMA,
    AugmentConfig,
    DataSpec,
    RawSample,
    SourceSpec,
    apply_geometry,
    build_dataset,
    license_key,
    require_track,
)

INPUT, GRID = (48, 36), (5, 4)
EVAL = AugmentConfig(margin=(0.0, 0.0))


def _page(size, corner=(0.15, 0.1), extent=(0.7, 0.8)):
    """Affine page: (u, v) in [0, 1]^2 -> edge-pixel coordinates inside an image."""
    height, width = size
    origin = np.array([corner[0] * width, corner[1] * height])
    axes = np.array([[extent[0] * width, 0.05 * width], [-0.04 * height, extent[1] * height]])
    return lambda uv: origin + uv @ axes


def _center_map(page, map_size, image_size):
    height, width = map_size
    u, v = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
    edges = page(np.stack([u, v], -1))
    return pixels_to_flattener(edges - 0.5, image_size)


def _image(size, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)


def _write_converted(root, count=3, *, track="release", license_class="clean", manifest=True):
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(count):
        size = (60, 44)
        Image.fromarray(_image(size, index)).save(root / f"s{index}.jpg", quality=95)
        values = _center_map(_page(size), (18, 12), size).astype(np.float16)
        np.save(root / f"s{index}.npy", values)
        entry = {
            "schema": SAMPLE_SCHEMA,
            "sample_id": f"s{index}",
            "split": "train",
            "image": f"s{index}.jpg",
            "backward_map": f"s{index}.npy",
            "track": track,
            "license_class": license_class,
        }
        entries.append(entry)
        if not manifest:
            (root / f"s{index}.json").write_text(json.dumps(entry))
    if manifest:
        (root / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    return entries


def _write_raw(root, split="train", count=2):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        size = (72, 51)
        prefix = root / f"syntheticdoc-{split}-{index:07d}"
        Image.fromarray(_image(size, 10 + index)).save(f"{prefix}-render.png")
        flattener_values = _center_map(_page(size), (18, 12), size)
        np.save(f"{prefix}-backward_map.npy", flattener_to_syntheticdoc(flattener_values, size))
        (root / f"{prefix.name}-metadata.json").write_text(json.dumps({"sample_id": index}))


def _grid_pixels(page, grid_size):
    rows, cols = grid_size
    u, v = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    return (page(np.stack([u, v], -1)) - 0.5).astype(np.float32)


def _write_doc3d(root):
    (root / "img/1").mkdir(parents=True)
    (root / "grid2D/1").mkdir(parents=True)
    (root / "grid3D/1").mkdir(parents=True)
    size = (64, 64)
    Image.fromarray(_image(size, 3)).save(root / "img/1/a.png")
    np.save(root / "grid2D/1/a.npy", _grid_pixels(_page(size), GRID))
    grid3d = np.zeros((*GRID, 3), np.float32)
    grid3d[..., 0] = 0.6436657  # z max -> channel 0 == 1
    ymin, ymax = -1.2289206, 1.2396319  # channel 1 is flipped left-right, then inverted
    grid3d[..., 1] = ymin + np.array([0.0, 0.1, 0.5, 1.0]) * (ymax - ymin)
    np.save(root / "grid3D/1/a.npy", grid3d)
    (root / "traindoc.txt").write_text("1/a\n")


def _write_uvdoc(root):
    for name in ("img", "metadata_sample", "grid2d", "grid3d"):
        (root / name).mkdir(parents=True)
    size = (71, 49)
    Image.fromarray(_image(size, 4)).save(root / "img/0001.png")
    (root / "metadata_sample/0001.json").write_text(json.dumps({"geom_name": "geom"}))
    np.save(root / "grid2d/geom.npy", _grid_pixels(_page(size), (9, 7)))
    np.save(root / "grid3d/geom.npy", np.full((9, 7, 3), -0.12551452, np.float32))


def _spec(tmp_path, sources, track="research", name="spec"):
    data = {"schema": SPEC_SCHEMA, "name": name, "track": track, "sources": sources}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return DataSpec.load(path)


def _build(spec, track="research", split="train", train=False, augment=EVAL, **kw):
    return build_dataset(
        spec,
        track=track,
        split=split,
        input_size=INPUT,
        grid_size=GRID,
        augment=augment,
        seed=7,
        train=train,
        **kw,
    )


def test_converted_source_items_and_relative_spec_roots(tmp_path):
    _write_converted(tmp_path / "data/conv", manifest=False)
    source = {
        "name": "clean",
        "kind": "converted",
        "root": "data/conv",
        "track": "release",
        "license": "CC BY 4.0 subset",
    }
    spec = _spec(tmp_path, [source], track="release")
    dataset = _build(spec, track="release")
    described = dataset.describe()[0]
    assert len(dataset) == 3 and described["samples"] == 3 and described["track"] == "release"
    assert described["license_classes"] == {"clean": 3} and len(described["records_sha256"]) == 64
    item = dataset[1]
    assert item["image"].shape == (3, *INPUT) and item["image"].dtype == torch.float32
    assert item["grid2d"].shape == (2, *GRID) and item["valid"].shape == (1, *GRID)
    assert item["valid"].all() and not item["has_grid3d"]
    assert (item["track"], item["source"], item["sample_id"]) == ("release", "clean", "s1")
    # With zero margin the page fills the crop: edge nodes reach +-1.
    grid = item["grid2d"].numpy()
    assert abs(grid[0].min() + 1) < 0.02 and abs(grid[1].max() - 1) < 0.02
    moved = DataSpec.from_dict(spec.to_dict(), base_dir="/elsewhere")
    assert moved.sha256 == spec.sha256


def test_raw_syntheticdoc_and_converted_conventions_agree(tmp_path):
    _write_raw(tmp_path / "raw")
    raw_spec = _spec(
        tmp_path,
        [
            {
                "name": "raw",
                "kind": "syntheticdoc_raw",
                "root": str(tmp_path / "raw"),
                "track": "research",
            }
        ],
    )
    raw = _build(raw_spec)[0]
    assert raw["sample_id"] == "train-0000000" and raw["track"] == "research"
    conv = tmp_path / "conv"
    conv.mkdir()
    render = tmp_path / "raw/syntheticdoc-train-0000000-render.png"
    stored = np.load(tmp_path / "raw/syntheticdoc-train-0000000-backward_map.npy")
    Image.open(render).save(conv / "r.png")
    np.save(conv / "b.npy", stored)
    entry = {
        "schema": SAMPLE_SCHEMA,
        "sample_id": "x",
        "image": "r.png",
        "backward_map": "b.npy",
        "backward_map_convention": "syntheticdoc",
        "original_render_size": [72, 51],
        "track": "research",
        "license_class": "nc",
    }
    (conv / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
    conv_spec = _spec(
        tmp_path,
        [{"name": "c", "kind": "converted", "root": str(conv), "track": "research"}],
        name="conv",
    )
    converted = _build(conv_spec)[0]
    torch.testing.assert_close(converted["grid2d"], raw["grid2d"])
    torch.testing.assert_close(converted["image"], raw["image"])


def test_release_track_refuses_research_sources(tmp_path):
    _write_converted(tmp_path / "clean")
    _write_doc3d(tmp_path / "doc3d")
    clean = {
        "name": "clean",
        "kind": "converted",
        "root": str(tmp_path / "clean"),
        "track": "release",
        "license": "CC BY 4.0",
    }
    doc3d = {"name": "doc3d", "kind": "doc3d", "root": str(tmp_path / "doc3d"), "track": "research"}
    mixed = _spec(tmp_path, [clean, doc3d], name="mixed")
    assert _build(mixed).sizes == [3, 1]  # research track accepts both
    for call in (lambda: require_track(mixed, "release"), lambda: _build(mixed, track="release")):
        with pytest.raises(FlattenerError) as error:
            call()
        assert error.value.reason_code == "track_violation"
    with pytest.raises(FlattenerError, match="research-only"):
        SourceSpec(name="d", kind="doc3d", root=".", track="release", license="MIT")
    with pytest.raises(FlattenerError) as error:
        _spec(tmp_path, [clean, doc3d], track="release", name="bad")
    assert error.value.reason_code == "track_violation"
    _write_converted(tmp_path / "leaky", track="research")
    leaky = {**clean, "name": "leaky", "root": str(tmp_path / "leaky")}
    with pytest.raises(FlattenerError, match="release source"):
        _build(_spec(tmp_path, [leaky], track="release", name="leaky"), track="release")
    # Only allowlisted license classes pass, however NC/ND/SA are spelled.
    refused = ("NC", "by-nc-sa", "cc-by-nc-nd-2.0", "CC BY-NC 4.0", "cc-by-sa-4.0", "unclear", "")
    for index, license_class in enumerate(refused):
        root = tmp_path / f"refused{index}"
        _write_converted(root, license_class=license_class)
        source = {**clean, "name": f"r{index}", "root": str(root)}
        with pytest.raises(FlattenerError, match="release source") as error:
            _build(_spec(tmp_path, [source], track="release", name=f"r{index}"), track="release")
        assert error.value.reason_code == "track_violation", license_class
    for index, license_class in enumerate(("CC BY 4.0", "cc0-1.0", "Apache-2.0")):
        root = tmp_path / f"accepted{index}"
        _write_converted(root, count=1, license_class=license_class)
        source = {**clean, "name": f"a{index}", "root": str(root)}
        spec = _spec(tmp_path, [source], track="release", name=f"a{index}")
        assert len(_build(spec, track="release")) == 1
    assert [license_key(c) for c in ("CC BY-NC-SA 4.0", "cc0-1.0")] == ["by-nc-sa", "cc0"]


def test_doc3d_and_uvdoc_adapters_follow_upstream_layouts(tmp_path):
    _write_doc3d(tmp_path / "doc3d")
    _write_uvdoc(tmp_path / "uvdoc")
    spec = _spec(
        tmp_path,
        [
            {
                "name": "doc3d",
                "kind": "doc3d",
                "root": str(tmp_path / "doc3d"),
                "track": "research",
            },
            {
                "name": "uvdoc",
                "kind": "uvdoc",
                "root": str(tmp_path / "uvdoc"),
                "track": "research",
            },
        ],
    )
    dataset = _build(spec)
    doc3d, uvdoc = dataset[0], dataset[1]
    for item in (doc3d, uvdoc):
        assert item["has_grid3d"] and item["valid"].all() and item["grid3d"].shape == (3, *GRID)
        grid = item["grid2d"].numpy()  # page fills the zero-margin crop
        assert abs(grid[0].min() + 1) < 1e-4 and abs(grid[1].max() - 1) < 1e-4
    torch.testing.assert_close(doc3d["grid3d"][0], torch.ones(GRID))
    expected = torch.tensor([0.0, 0.5, 0.9, 1.0]).expand(GRID)
    torch.testing.assert_close(doc3d["grid3d"][1], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(uvdoc["grid3d"][0], torch.zeros(GRID), atol=1e-6, rtol=0)
    assert uvdoc["sample_id"] == "0001" and uvdoc["kind"] == "uvdoc"


def test_matlab_v73_grids_are_read_with_h5py(tmp_path):
    h5py = pytest.importorskip("h5py")
    _write_doc3d(tmp_path / "doc3d")
    for name, key in (("grid2D", "grid2D"), ("grid3D", "grid3D")):
        array = np.load(tmp_path / f"doc3d/{name}/1/a.npy")
        (tmp_path / f"doc3d/{name}/1/a.npy").unlink()
        with h5py.File(tmp_path / f"doc3d/{name}/1/a.mat", "w") as stream:
            stream.create_dataset(key, data=array.T)
    spec = _spec(
        tmp_path,
        [{"name": "d", "kind": "doc3d", "root": str(tmp_path / "doc3d"), "track": "research"}],
    )
    assert _build(spec)[0]["valid"].all()


def test_geometry_keeps_image_and_grid_consistent_under_rotation():
    size = 256
    ramp = np.arange(size, dtype=np.uint8)
    image = np.stack(
        [
            np.tile(ramp, (size, 1)),
            np.tile(ramp[:, None], (1, size)),
            np.zeros((size, size), np.uint8),
        ],
        axis=-1,
    )
    page = _page((size, size), corner=(0.25, 0.2), extent=(0.45, 0.5))
    grid = pixels_to_flattener(_grid_pixels(page, (9, 7)), (size, size))
    sample = RawSample(image, grid, np.ones((9, 7), bool), None)
    augment = AugmentConfig(margin=(0.05, 0.1), rotation_deg=25.0, rotation_prob=1.0)
    for seed in range(4):
        rng = np.random.default_rng(seed)
        output, target = apply_geometry(sample, (120, 90), augment, rng)
        points = torch.from_numpy(target).unsqueeze(0)
        sampled = F.grid_sample(output[None], points, align_corners=False)[0, :2].numpy()
        expected = ((grid + 1) / 2 * size - 0.5).transpose(2, 0, 1) / 255
        assert np.abs(sampled - expected).max() < 2e-3
    # rotation_kinds: upstream rotates UVDoc samples only.
    only_uvdoc = AugmentConfig(
        margin=(0.05, 0.1), rotation_deg=25.0, rotation_prob=1.0, rotation_kinds=["uvdoc"]
    )
    unrotated = AugmentConfig(margin=(0.05, 0.1), rotation_deg=0.0)
    for kind, rotated in (("converted", False), ("uvdoc", True)):
        tagged = RawSample(image, grid, np.ones((9, 7), bool), None, {"kind": kind})
        a = apply_geometry(tagged, (120, 90), only_uvdoc, np.random.default_rng(3))[1]
        b = apply_geometry(tagged, (120, 90), unrotated, np.random.default_rng(3))[1]
        assert np.array_equal(a, b) != rotated


def test_augmentation_is_seeded_by_position_and_dataset_pickles(tmp_path):
    _write_converted(tmp_path / "conv")
    spec = _spec(
        tmp_path,
        [
            {
                "name": "c",
                "kind": "converted",
                "root": str(tmp_path / "conv"),
                "track": "release",
                "license": "CC0",
            }
        ],
        track="release",
    )
    augment = AugmentConfig(rotation_deg=10.0, full_frame_prob=0.3, blur_prob=0.5)
    dataset = _build(spec, track="release", train=True, augment=augment)
    first, again, other = dataset[(1, 42)], dataset[(1, 42)], dataset[(1, 43)]
    assert torch.equal(first["image"], again["image"])
    assert torch.equal(first["grid2d"], again["grid2d"])
    assert not torch.equal(first["image"], other["image"])
    clone = pickle.loads(pickle.dumps(dataset))
    assert torch.equal(clone[(1, 42)]["image"], first["image"])
    evaluation = _build(spec, track="release", train=False, augment=augment)
    assert torch.equal(evaluation[(1, 1)]["image"], evaluation[(1, 2)]["image"])


def test_spec_and_augment_validation(tmp_path):
    with pytest.raises(FlattenerError):
        DataSpec.from_dict({"schema": "other", "name": "x", "track": "research", "sources": []})
    with pytest.raises(FlattenerError, match="bad source"):
        DataSpec.from_dict(
            {
                "schema": SPEC_SCHEMA,
                "name": "x",
                "track": "research",
                "sources": [
                    {"name": "a", "kind": "converted", "root": ".", "track": "research", "shard": 1}
                ],
            }
        )
    with pytest.raises(FlattenerError, match="unique"):
        source = {"name": "a", "kind": "converted", "root": ".", "track": "research"}
        _spec(tmp_path, [source, source])
    with pytest.raises(FlattenerError, match="license"):
        SourceSpec(name="a", kind="converted", root=".", track="release")
    assert AugmentConfig(rotation_kinds=["uvdoc"]).to_dict()["rotation_kinds"] == ["uvdoc"]
    for bad in (
        {"margin": (0.2, 0.1)},
        {"blur_prob": 1.5},
        {"rotation_deg": 90.0},
        {"rotation_kinds": ["doc4d"]},
    ):
        with pytest.raises(ValueError):
            AugmentConfig(**bad)
    _write_converted(tmp_path / "conv", count=5)
    limited = _spec(
        tmp_path,
        [
            {
                "name": "c",
                "kind": "converted",
                "root": str(tmp_path / "conv"),
                "track": "research",
                "max_samples": 2,
            }
        ],
        name="limited",
    )
    assert [item["sample_id"] for item in _build(limited)] == ["s0", "s4"]
    assert _build(limited, split="val") is None
