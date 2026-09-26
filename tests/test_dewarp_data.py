"""Dewarp data adapters, license tracks, views and seeded augmentation on tiny fixtures."""

import hashlib
import json
import os
import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from flattener import FlattenerError
from flattener.dewarp.coords import flattener_to_syntheticdoc, pixels_to_flattener
from flattener.dewarp.data import (
    COLOR_PARAMS,
    COMPACT_SCHEMA,
    IDENTITY_COLOR,
    SAMPLE_SCHEMA,
    SPEC_SCHEMA,
    AugmentConfig,
    DataSpec,
    RawSample,
    SourceSpec,
    ViewConfig,
    apply_color_batch,
    apply_geometry,
    build_dataset,
    draw_color_params,
    license_key,
    require_track,
    to_float_images,
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


def _write_converted(
    root, count=3, *, track="release", license_class="clean", manifest=True, size=(60, 44)
):
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(count):
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
    Image.fromarray(_image(size, 3)).convert("RGBA").save(root / "img/1/a.png")  # as upstream
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


def _fills_crop(grid, tolerance):
    """Page nodes span the crop up to ``tolerance`` (whole-pixel boxes snap outward)."""
    return (
        -1 - 1e-4 <= grid[0].min() < -1 + tolerance
        and 1 - tolerance < grid[0].max() <= 1 + 1e-4
        and -1 - 1e-4 <= grid[1].min() < -1 + tolerance
        and 1 - tolerance < grid[1].max() <= 1 + 1e-4
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
    item = dataset[1]  # evaluation items stack one entry per view
    assert item["image"].shape == (1, 3, *INPUT) and item["image"].dtype == torch.uint8
    assert item["grid2d"].shape == (1, 2, *GRID) and item["valid"].shape == (1, 1, *GRID)
    assert item["valid"].all() and not item["has_grid3d"]
    assert (item["track"], item["source"], item["sample_id"]) == ("release", "clean", "s1")
    assert _fills_crop(item["grid2d"][0].numpy(), 0.08)  # zero margin: the page fills the crop
    moved = DataSpec.from_dict(spec.to_dict(), base_dir="/elsewhere")
    assert moved.sha256 == spec.sha256


def test_training_items_carry_uint8_images_and_color_parameters(tmp_path):
    _write_converted(tmp_path / "conv")
    source = {"name": "c", "kind": "converted", "root": str(tmp_path / "conv"), "track": "research"}
    dataset = _build(_spec(tmp_path, [source]), train=True, augment=AugmentConfig())
    item = dataset[(0, 5)]
    assert item["image"].shape == (3, *INPUT) and item["image"].dtype == torch.uint8
    assert item["color"].shape == (COLOR_PARAMS,) and item["noise_seed"].dtype == torch.int64
    batch = torch.utils.data.default_collate([dataset[(i, i)] for i in range(3)])
    images = to_float_images(batch)
    assert images.dtype == torch.float32 and images.shape == (3, 3, *INPUT)
    assert 0 <= float(images.min()) and float(images.max()) <= 1
    plain = build_dataset(
        _spec(tmp_path, [source], name="plain"),
        track="research",
        split="train",
        input_size=INPUT,
        grid_size=GRID,
        augment=AugmentConfig(enabled=False),
        train=True,
    )[(0, 5)]
    torch.testing.assert_close(plain["color"], torch.from_numpy(IDENTITY_COLOR))


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


def _write_compact(root, *, track="release", count=2):
    """A tiny copy in the layout written by tools/syntheticdoc/compact.py."""
    root.mkdir(parents=True)
    (root / "compact.json").write_text(json.dumps({"schema": COMPACT_SCHEMA, "track": track}))
    lines = []
    size = (60, 44)
    for index in range(count):
        Image.fromarray(_image(size, index)).save(root / f"c{index}.jpg", quality=92)
        stored = flattener_to_syntheticdoc(_center_map(_page(size), (18, 12), size), (120, 88))
        np.save(root / f"c{index}.npy", stored.astype(np.float16))
        entry = {
            "schema": SAMPLE_SCHEMA,
            "sample_id": f"c{index}",
            "split": "train",
            "image": f"c{index}.jpg",
            "backward_map": f"c{index}.npy",
            "backward_map_convention": "syntheticdoc",
            "original_render_size": [120, 88],  # the copy is a half-size render
            "track": track,
            "license_class": "clean",
        }
        for key, name in (("image", f"c{index}.jpg"), ("backward_map", f"c{index}.npy")):
            data = (root / name).read_bytes()
            entry[f"{key}_sha256"] = hashlib.sha256(data).hexdigest()
            entry[f"{key}_bytes"] = len(data)
        lines.append(json.dumps(entry) + "\n")
    (root / "manifest.jsonl").write_text("".join(lines))


def test_compact_source_checks_hashes_sizes_and_track(tmp_path):
    _write_compact(tmp_path / "compact")
    source = {
        "name": "sd",
        "kind": "syntheticdoc_compact",
        "root": str(tmp_path / "compact"),
        "track": "release",
        "license": "SyntheticDoc CC BY 4.0, clean",
    }
    spec = _spec(tmp_path, [source], track="release")
    dataset = _build(spec, track="release")
    assert len(dataset) == 2 and _fills_crop(dataset[0]["grid2d"][0].numpy(), 0.08)
    digest = dataset.describe()[0]["content_sha256"]
    # A re-encoded file under the same id changes the content digest (resume refuses).
    manifest = tmp_path / "compact/manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    entries[0]["image_sha256"] = "0" * 64
    manifest.write_text("".join(json.dumps(e) + "\n" for e in entries))
    assert _build(spec, track="release").describe()[0]["content_sha256"] != digest
    # A file that no longer matches the manifest size is refused before training.
    Image.fromarray(_image((60, 44), 9)).save(tmp_path / "compact/c1.jpg", quality=50)
    with pytest.raises(FlattenerError) as error:
        _build(spec, track="release")
    assert error.value.reason_code == "stale_manifest"
    del entries[1]["image_sha256"]
    manifest.write_text("".join(json.dumps(e) + "\n" for e in entries))
    with pytest.raises(FlattenerError, match="sha256"):
        _build(spec, track="release")
    entries[1]["image_sha256"] = "0" * 64
    del entries[1]["backward_map_bytes"]
    manifest.write_text("".join(json.dumps(e) + "\n" for e in entries))
    with pytest.raises(FlattenerError, match="bytes missing"):
        _build(spec, track="release")
    _write_compact(tmp_path / "research", track="research")
    leaky = {**source, "name": "r", "root": str(tmp_path / "research")}
    with pytest.raises(FlattenerError) as error:
        _build(_spec(tmp_path, [leaky], track="release", name="r"), track="release")
    assert error.value.reason_code == "track_violation"


def test_content_digest_follows_file_sizes_without_manifest_hashes(tmp_path):
    _write_converted(tmp_path / "conv", count=2)
    source = {"name": "c", "kind": "converted", "root": str(tmp_path / "conv"), "track": "research"}
    spec = _spec(tmp_path, [source])
    before = _build(spec).describe()[0]
    Image.fromarray(_image((60, 44), 0)).save(tmp_path / "conv/s0.jpg", quality=40)
    after = _build(spec).describe()[0]
    assert after["records_sha256"] == before["records_sha256"]
    assert after["content_sha256"] != before["content_sha256"]
    (tmp_path / "conv/s1.npy").unlink()
    with pytest.raises(FlattenerError) as error:
        _build(spec)
    assert error.value.reason_code == "missing_data"


def test_spec_roots_expand_environment_variables(tmp_path, monkeypatch):
    _write_converted(tmp_path / "data/conv", count=1)
    source = {
        "name": "c",
        "kind": "converted",
        "root": "${FLATTENER_TEST_DATA}/conv",
        "track": "research",
    }
    spec = _spec(tmp_path, [source])
    monkeypatch.delenv("FLATTENER_TEST_DATA", raising=False)
    with pytest.raises(FlattenerError) as error:
        _build(spec)
    assert error.value.reason_code == "unset_variable"
    monkeypatch.setenv("FLATTENER_TEST_DATA", str(tmp_path / "data"))
    assert len(_build(spec)) == 1
    assert spec.to_dict()["sources"][0]["root"] == "${FLATTENER_TEST_DATA}/conv"


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
        assert _fills_crop(item["grid2d"][0].numpy(), 0.06)
        assert item["image"].shape == (1, 3, *INPUT)  # the RGBA Doc3D image as RGB
    trained = _build(spec, train=True, augment=AugmentConfig())
    assert trained[(0, 3)]["image"].shape == (3, *INPUT)
    torch.testing.assert_close(doc3d["grid3d"][0], torch.ones(GRID))
    expected = torch.tensor([0.0, 0.5, 0.9, 1.0]).expand(GRID)
    torch.testing.assert_close(doc3d["grid3d"][1], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(uvdoc["grid3d"][0], torch.zeros(GRID), atol=1e-6, rtol=0)
    assert uvdoc["sample_id"] == "0001" and uvdoc["kind"] == "uvdoc"


def _digest(spec):
    return _build(spec).describe()[0]["content_sha256"]


def _touch(path, seconds):
    os.utime(path, ns=(seconds * 10**9, seconds * 10**9))


def test_npy_grids_are_checked_against_their_mat_files(tmp_path):
    _write_doc3d(tmp_path / "doc3d")
    grid2d = tmp_path / "doc3d/grid2D/1"
    source = {"name": "d", "kind": "doc3d", "root": str(tmp_path / "doc3d"), "track": "research"}
    spec = _spec(tmp_path, [source])
    npy_only = _digest(spec)
    _touch(grid2d / "a.npy", 2_000_000_000)
    (grid2d / "a.mat").write_bytes(b"x" * 100)  # the .mat the .npy was converted from
    _touch(grid2d / "a.mat", 1_900_000_000)
    with_mat = _digest(spec)
    assert with_mat != npy_only
    (grid2d / "a.mat").write_bytes(b"x" * 120)  # re-fetched with another size
    _touch(grid2d / "a.mat", 1_900_000_000)
    assert _digest(spec) != with_mat
    _touch(grid2d / "a.mat", 2_100_000_000)  # re-fetched after the conversion
    with pytest.raises(FlattenerError) as error:
        _build(spec)
    assert error.value.reason_code == "stale_cache"


def test_uvdoc_digest_covers_the_shared_geometry_files(tmp_path):
    _write_uvdoc(tmp_path / "uvdoc")
    source = {"name": "u", "kind": "uvdoc", "root": str(tmp_path / "uvdoc"), "track": "research"}
    spec = _spec(tmp_path, [source])
    before = _digest(spec)
    grid3d = tmp_path / "uvdoc/grid3d/geom.npy"
    np.save(grid3d, np.full((9, 7, 3), -0.12551452, np.float64))  # replaced geometry
    assert _digest(spec) != before
    _touch(grid3d, 2_000_000_000)
    (grid3d.with_suffix(".mat")).write_bytes(b"x")
    _touch(grid3d.with_suffix(".mat"), 2_100_000_000)
    with pytest.raises(FlattenerError) as error:
        _build(spec)
    assert error.value.reason_code == "stale_cache"


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


def _ramp(size, slope):
    ramp = (np.arange(size) * slope).astype(np.uint8)
    return np.stack(
        [np.tile(ramp, (size, 1)), np.tile(ramp[:, None], (1, size)), np.zeros((size,) * 2)],
        axis=-1,
    ).astype(np.uint8)


def test_geometry_keeps_image_and_grid_consistent_under_rotation():
    size, slope = 64, 3.0  # ramp values = slope * pixel index
    page = _page((size, size), corner=(0.25, 0.2), extent=(0.45, 0.5))
    grid = pixels_to_flattener(_grid_pixels(page, (9, 7)), (size, size))
    sample = RawSample(_ramp(size, slope), grid, np.ones((9, 7), bool), None)
    augment = AugmentConfig(margin=(0.05, 0.1), rotation_deg=25.0, rotation_prob=1.0)
    expected = ((grid + 1) / 2 * size - 0.5).transpose(2, 0, 1) * slope
    for seed in range(4):
        rng = np.random.default_rng(seed)
        output, target = apply_geometry(sample, (120, 90), augment, rng)
        assert output.dtype == torch.uint8 and output.shape == (3, 120, 90)
        points = torch.from_numpy(target).unsqueeze(0)
        sampled = F.grid_sample(output[None].double(), points.double(), align_corners=False)
        # PIL truncation costs at most one level; a 0.1 px shift would add 0.3.
        assert np.abs(sampled[0, :2].numpy() - expected + 0.5).max() < 0.8
    # Axis-aligned crops snap to whole pixels and resize with rounding.
    straight = AugmentConfig(margin=(0.05, 0.1), rotation_deg=0.0)
    for seed in range(4):
        output, target = apply_geometry(sample, (120, 90), straight, np.random.default_rng(seed))
        points = torch.from_numpy(target).unsqueeze(0)
        sampled = F.grid_sample(output[None].double(), points.double(), align_corners=False)
        assert np.abs(sampled[0, :2].numpy() - expected).max() < 0.6
    # rotation_kinds: upstream rotates UVDoc samples only.
    only_uvdoc = AugmentConfig(
        margin=(0.05, 0.1), rotation_deg=25.0, rotation_prob=1.0, rotation_kinds=["uvdoc"]
    )
    unrotated = AugmentConfig(margin=(0.05, 0.1), rotation_deg=0.0)
    for kind, rotated in (("converted", False), ("uvdoc", True)):
        tagged = RawSample(sample.image, grid, np.ones((9, 7), bool), None, {"kind": kind})
        a = apply_geometry(tagged, (120, 90), only_uvdoc, np.random.default_rng(3))[1]
        b = apply_geometry(tagged, (120, 90), unrotated, np.random.default_rng(3))[1]
        assert np.array_equal(a, b) != rotated


def test_jpeg_draft_decodes_less_and_keeps_the_geometry(tmp_path):
    size = 256
    Image.fromarray(_ramp(size, 1.0)).save(tmp_path / "ramp.jpg", quality=98)
    page = _page((size, size), corner=(0.1, 0.1), extent=(0.8, 0.8))
    grid = pixels_to_flattener(_grid_pixels(page, (5, 4)), (size, size))
    outputs = {}
    for draft in (False, True):
        sample = RawSample(Image.open(tmp_path / "ramp.jpg"), grid, np.ones((5, 4), bool), None)
        augment = AugmentConfig(full_frame_prob=1.0, jpeg_draft=draft)
        outputs[draft] = apply_geometry(sample, (48, 36), augment, np.random.default_rng(0))
        decoded = sample.image.size
        sample.image.close()
    assert decoded == (64, 64)  # 256 px -> 48 px output: 1/4 DCT scale
    np.testing.assert_array_equal(outputs[True][1], outputs[False][1])
    difference = outputs[True][0].double() - outputs[False][0].double()
    assert difference.abs().mean() < 2.0  # same ramp values, no half-pixel shift


def test_evaluation_views_resize_whole_pixel_crops_like_predict(tmp_path):
    _write_converted(tmp_path / "conv", count=2, size=(120, 88))
    source = {"name": "c", "kind": "converted", "root": str(tmp_path / "conv"), "track": "research"}
    spec = _spec(tmp_path, [source])
    views = (
        ViewConfig("page", "page", (0.0, 0.0)),
        ViewConfig("docunet", "page", (0.02, 0.1)),
        ViewConfig("frame", "full_frame"),
    )
    item = _build(spec, views=views)[1]
    assert item["image"].shape == (3, 3, *INPUT) and item["grid2d"].shape == (3, 2, *GRID)
    docunet = item["grid2d"][1].numpy()  # every side keeps a margin
    assert -1 + 0.03 < docunet.min() and docunet.max() < 1 - 0.03
    again = build_dataset(
        spec,
        track="research",
        split="train",
        input_size=INPUT,
        grid_size=GRID,
        augment=AugmentConfig(margin=(0.2, 0.3)),
        seed=99,
        train=False,
        views=views,
    )[1]
    assert torch.equal(again["image"], item["image"])  # views ignore the run seed and augment
    assert not torch.equal(item["grid2d"][0], item["grid2d"][1])
    # The full-frame view is exactly predict's input: the whole image resized.
    with Image.open(tmp_path / "conv/s1.jpg") as opened:
        source_image = torch.from_numpy(np.array(opened))
    reference = F.interpolate(
        source_image.permute(2, 0, 1)[None].double(),
        size=INPUT,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )[0]
    difference = item["image"][2].double() - reference
    assert difference.abs().max() < 1 + 1e-6 and abs(float(difference.mean())) < 0.25
    grid = item["grid2d"][2].numpy()  # page nodes relative to the whole image
    assert grid[0].min() > -1 and grid[0].max() < 1


def test_images_of_other_modes_load_as_rgb_in_every_view(tmp_path):
    root = tmp_path / "modes"
    _write_converted(root, count=3, size=(60, 44))
    entries = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    for entry, mode in zip(entries, ("RGBA", "L", "P"), strict=True):
        with Image.open(root / entry["image"]) as opened:
            opened.convert(mode).save(root / f"{entry['sample_id']}.png")
        entry["image"] = f"{entry['sample_id']}.png"
    (root / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    source = {"name": "m", "kind": "converted", "root": str(root), "track": "research"}
    spec = _spec(tmp_path, [source])
    views = (ViewConfig("page"), ViewConfig("frame", "full_frame"))
    dataset = _build(spec, views=views)
    trained = _build(spec, train=True, augment=AugmentConfig(rotation_deg=20, rotation_prob=1))
    for index, entry in enumerate(entries):
        item = dataset[index]
        assert item["image"].shape == (2, 3, *INPUT)
        with Image.open(root / entry["image"]) as opened:
            rgb = torch.from_numpy(np.array(opened.convert("RGB"))).permute(2, 0, 1)
        reference = F.interpolate(rgb[None].double(), size=INPUT, mode="bilinear")[0]
        assert (item["image"][1].double() - reference).abs().max() <= 1
        assert trained[(index, index)]["image"].shape == (3, *INPUT)


def test_color_jitter_runs_per_batch_and_matches_per_sample(tmp_path):
    augment = AugmentConfig(color_prob=1.0, grayscale_prob=0.3, blur_prob=0.5, noise_prob=0.7)
    draws = [draw_color_params(augment, np.random.default_rng([3, i])) for i in range(6)]
    params = torch.from_numpy(np.stack([p for p, _ in draws]))
    seeds = torch.tensor([s for _, s in draws])
    assert params[:, 9].any() and params[:, 11].any()  # blur and noise both occur
    images = torch.rand(6, 3, 20, 16)
    batch = apply_color_batch(images.clone(), params, seeds)
    for index in range(6):
        single = apply_color_batch(
            images[index : index + 1].clone(), params[index : index + 1], seeds[index : index + 1]
        )
        torch.testing.assert_close(batch[index : index + 1], single, atol=1e-6, rtol=0)
    torch.testing.assert_close(apply_color_batch(images.clone(), params, seeds), batch)
    identity = torch.from_numpy(np.tile(IDENTITY_COLOR, (6, 1)))
    torch.testing.assert_close(apply_color_batch(images.clone(), identity, [0] * 6), images)
    assert 0 <= float(batch.min()) and float(batch.max()) <= 1


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
    for key in ("image", "grid2d", "color", "noise_seed"):
        assert torch.equal(first[key], again[key]), key
    assert not torch.equal(first["image"], other["image"])
    assert not torch.equal(first["color"], other["color"])
    clone = pickle.loads(pickle.dumps(dataset))
    assert torch.equal(clone[(1, 42)]["image"], first["image"])
    evaluation = _build(spec, track="release", train=False, augment=augment)
    assert torch.equal(evaluation[(1, 1)]["image"], evaluation[(1, 2)]["image"])


def test_spec_view_and_augment_validation(tmp_path):
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
        {"jpeg_draft": 1},
    ):
        with pytest.raises(ValueError):
            AugmentConfig(**bad)
    view = ViewConfig("docunet", "page", [0.02, 0.1])
    assert ViewConfig.from_dict(view.to_dict()) == view
    for bad in ({"name": ""}, {"name": "a", "crop": "tight"}, {"name": "a", "margin": (0.3, 0.1)}):
        with pytest.raises(ValueError):
            ViewConfig(**bad)
    with pytest.raises(ValueError):
        ViewConfig.from_dict({"name": "a", "seed": 1})
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
    with pytest.raises(FlattenerError, match="unique"):
        _build(limited, views=(ViewConfig("a"), ViewConfig("a")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_color_jitter_on_cuda_matches_the_cpu():
    augment = AugmentConfig(color_prob=1.0, grayscale_prob=0.3, blur_prob=0.5, noise_prob=0.0)
    draws = [draw_color_params(augment, np.random.default_rng([4, i])) for i in range(8)]
    params = torch.from_numpy(np.stack([p for p, _ in draws]))
    images = torch.rand(8, 3, 40, 32)
    cpu = apply_color_batch(images.clone(), params, [0] * 8)
    cuda = apply_color_batch(images.cuda(), params.cuda(), [0] * 8).cpu()
    torch.testing.assert_close(cuda, cpu, atol=1e-5, rtol=0)
