"""Dewarp data adapters, license tracks, views, seeded augmentation and the data
signature, on tiny fixtures."""

import hashlib
import inspect
import json
import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import flattener.dewarp.data as dewarp_data
from flattener import FlattenerError
from flattener.dewarp import coords
from flattener.dewarp.coords import edge_positions, flattener_to_syntheticdoc, pixels_to_flattener
from flattener.dewarp.data import (
    COMPACT_SCHEMA,
    SAMPLE_SCHEMA,
    SPEC_SCHEMA,
    AugmentConfig,
    DataSpec,
    RawSample,
    SourceSpec,
    ViewConfig,
    apply_geometry,
    build_dataset,
    evaluation_view,
    license_key,
    require_track,
)
from flattener.dewarp.model import DewarpModelConfig
from flattener.dewarp.train import (
    TrainConfig,
    _check_resume,
    _data_sources,
    code_record,
    validation_signature,
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
    refused = ("by-nc-sa", "cc-by-nc-nd-2.0", "cc-by-sa-4.0", "unclear")
    for index, license_class in enumerate(refused):
        root = tmp_path / f"refused{index}"
        _write_converted(root, license_class=license_class)
        source = {**clean, "name": f"r{index}", "root": str(root)}
        with pytest.raises(FlattenerError, match="release source") as error:
            _build(_spec(tmp_path, [source], track="release", name=f"r{index}"), track="release")
        assert error.value.reason_code == "track_violation", license_class
    for index, license_class in enumerate(("CC BY 4.0", "Apache-2.0")):
        root = tmp_path / f"accepted{index}"
        _write_converted(root, count=1, license_class=license_class)
        source = {**clean, "name": f"a{index}", "root": str(root)}
        spec = _spec(tmp_path, [source], track="release", name=f"a{index}")
        assert len(_build(spec, track="release")) == 1
    assert [license_key(c) for c in ("CC BY-NC-SA 4.0", "cc0-1.0")] == ["by-nc-sa", "cc0"]


def test_sources_refuse_benchmark_folders(tmp_path):
    from flattener.benchmark.uvdoc import MANIFEST_SCHEMA

    bench = tmp_path / "uvdoc-bench"  # as `python -m flattener.benchmark uvdoc-prepare` leaves it
    for folder in ("img", "grid2d", "grid3d", "metadata_sample"):
        (bench / folder).mkdir(parents=True)
    Image.fromarray(_image((64, 48))).save(bench / "img/00000.png")
    manifest = {"schema": MANIFEST_SCHEMA, "benchmark": "uvdoc"}
    (bench / "manifest.json").write_text(json.dumps(manifest))
    uvdoc = {"name": "bench", "kind": "uvdoc", "root": str(bench), "track": "research"}
    with pytest.raises(FlattenerError, match="manifest.json") as error:
        _build(_spec(tmp_path, [uvdoc]))
    assert error.value.reason_code == "benchmark_data"
    # A copy inside a benchmark folder or under outputs/benchmarks, and folders with the
    # eval-only record of fetch_research.py or DIR300's provenance.json.
    marked = {
        bench / "converted": None,
        tmp_path / "outputs" / "benchmarks" / "docunet": None,
        tmp_path / "eval" / "uvdoc_benchmark": ("TRACK.json", {"track": "eval-only"}),
        tmp_path / "dir300": ("provenance.json", {"training_use": False}),
    }
    for index, (root, marker) in enumerate(marked.items()):
        _write_converted(root, track="research")
        if marker is not None:
            (root / marker[0]).write_text(json.dumps(marker[1]))
        source = {"name": f"m{index}", "kind": "converted", "root": str(root), "track": "research"}
        with pytest.raises(FlattenerError) as error:
            _build(_spec(tmp_path, [source], name=f"m{index}"))
        assert error.value.reason_code == "benchmark_data", root
    plain = tmp_path / "research" / "uvdoc_final"
    _write_converted(plain, track="research")
    (plain / "TRACK.json").write_text(json.dumps({"track": "research"}))
    source = {"name": "plain", "kind": "converted", "root": str(plain), "track": "research"}
    assert len(_build(_spec(tmp_path, [source], name="plain"))) == 3


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


def _ramp(size, slope):
    ramp = (np.arange(size) * slope).astype(np.uint8)
    return np.stack(
        [np.tile(ramp, (size, 1)), np.tile(ramp[:, None], (1, size)), np.zeros((size,) * 2)],
        axis=-1,
    ).astype(np.uint8)


def test_augmentation_keeps_image_and_grid_consistent_and_is_seeded(tmp_path):
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
    # Train items are seeded by (index, position); the dataset pickles for workers.
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


# Evaluation views keep exact targets.

VIEW_INPUT = (48, 40)
MARGIN = (0.05, 0.05)


def _sample():
    """Colour ramps along x (red) and y (green) and a slanted page inside the image.

    Bilinear resampling reproduces a linear ramp, so a view sampled at its target
    positions gives back the colours of the page nodes in any correct view.
    """
    height, width = 90, 70
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.stack([xx * 255 / (width - 1), yy * 255 / (height - 1), 0 * xx + 128], -1)
    u, v = np.meshgrid(edge_positions(4), edge_positions(5))
    grid = np.stack([0.55 * u + 0.1 * v - 0.05, 0.6 * v + 0.05 * u + 0.1], -1)
    return RawSample(
        image.round().astype(np.uint8), grid.astype(np.float32), np.ones((5, 4), bool), None
    )


def _colours(image, target):
    grid = torch.from_numpy(target)[None].double()
    sampled = F.grid_sample(image[None].double(), grid, mode="bilinear", align_corners=False)
    return sampled[0].permute(1, 2, 0).numpy()


def test_evaluation_views_keep_exact_targets(tmp_path):
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
    # Quarter turns turn the targets with the image.
    sample = _sample()
    upright_view = ViewConfig("a", margin=MARGIN)
    upright_image, upright = evaluation_view(sample, VIEW_INPUT, upright_view, "k")
    expected = _colours(upright_image, upright)
    for rotation in (90, 180, 270):
        view = ViewConfig("a", margin=MARGIN, rotation=rotation)
        image, target = evaluation_view(sample, VIEW_INPUT, view, "k")
        assert image.shape == (3, *VIEW_INPUT)
        np.testing.assert_allclose(_colours(image, target), expected, atol=3)
    turned_view = ViewConfig("a", margin=MARGIN, rotation=90)
    image, turned = evaluation_view(sample, VIEW_INPUT, turned_view, "k")
    x, y = upright[..., 0], upright[..., 1]
    np.testing.assert_allclose(turned, np.stack([y, -x], -1), atol=1e-6)  # counter-clockwise
    assert image[0, 0].float().mean() > image[0, -1].float().mean()  # red now grows upward
    # The small page covers its fraction of a textured canvas.
    page_image, page = evaluation_view(sample, VIEW_INPUT, ViewConfig("s", margin=MARGIN), "k")
    view = ViewConfig("s", "small_page", MARGIN)
    image, target = evaluation_view(sample, VIEW_INPUT, view, "k")
    assert image.shape == (3, *VIEW_INPUT)
    np.testing.assert_allclose(_colours(image, target), _colours(page_image, page), atol=4)
    extent = (target[..., 0].max() - target[..., 0].min()) * (
        target[..., 1].max() - target[..., 1].min()
    )
    assert 0.05 < extent / 4 <= 0.25  # the page nodes lie inside 10-25% of the canvas
    again, _ = evaluation_view(sample, VIEW_INPUT, view, "k")
    assert torch.equal(again, image)  # seeded by the view name and the sample key
    other, _ = evaluation_view(sample, VIEW_INPUT, view, "other")
    assert not torch.equal(other, image)
    outside = np.ones(VIEW_INPUT, bool)
    pixels = ((target + 1) / 2 * [VIEW_INPUT[1], VIEW_INPUT[0]]).astype(int)
    outside[pixels[..., 1].min() : pixels[..., 1].max(), pixels[..., 0].min() :] = False
    assert image[:, torch.from_numpy(outside)].float().std() > 1  # canvas grain and shading


# The data signature covers how targets are read from the files.
#
# A resume compares ``data_sources`` (``DewarpDataset.describe``) with the checkpoint.
# Records that keep their files and ids but change how the stored map is interpreted
# (``backward_map_convention``) or a compact copy whose settings changed (``compact.json``
# ``scale``) give other targets, so the resume is refused and the validation signature
# changes.

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
SIZE = (60, 44)
CODE = code_record()


def _write_samples(root, *, compact=False):
    """Two train and one val sample in the SyntheticDoc convention of a 2x render."""
    root.mkdir()
    base = np.stack(np.meshgrid(np.linspace(-0.7, 0.7, 12), np.linspace(-0.8, 0.8, 18)), -1)
    rng = np.random.default_rng(0)
    lines = []
    for index, split in enumerate(("train", "train", "val")):
        Image.fromarray(rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)).save(
            root / f"{index}.png"
        )
        np.save(root / f"{index}.npy", (base + rng.normal(0, 0.01, base.shape)).astype(np.float16))
        entry = {
            "schema": SAMPLE_SCHEMA,
            "sample_id": str(index),
            "split": split,
            "image": f"{index}.png",
            "backward_map": f"{index}.npy",
            "backward_map_convention": "syntheticdoc",
            "original_render_size": [2 * SIZE[0], 2 * SIZE[1]],
            "track": "release",
            "license_class": "clean",
        }
        if compact:
            for key in ("image", "backward_map"):
                data = (root / entry[key]).read_bytes()
                entry[f"{key}_sha256"] = hashlib.sha256(data).hexdigest()
                entry[f"{key}_bytes"] = len(data)
        lines.append(json.dumps(entry) + "\n")
    (root / "manifest.jsonl").write_text("".join(lines))
    if compact:
        settings = {"schema": COMPACT_SCHEMA, "track": "release", "scale": 0.5}
        (root / "compact.json").write_text(json.dumps(settings))


def _signature_spec(tmp_path, kind):
    sources = [
        {
            "name": f"sd-{split}",
            "kind": kind,
            "root": "data",
            "track": "release",
            "license": "SyntheticDoc CC BY 4.0, clean",
            "split": split,
        }
        for split in ("train", "val")
    ]
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps({"schema": SPEC_SCHEMA, "name": "sd", "track": "release", "sources": sources})
    )
    return DataSpec.load(path)


def _sources(spec, config):
    """``data_sources`` as ``train`` computes them."""
    arguments = dict(
        track=config.track,
        input_size=config.model.input_size,
        grid_size=config.model.grid_size,
        augment=config.augment,
        seed=config.seed,
    )
    train_set = build_dataset(spec, split="train", train=True, **arguments)
    val_set = build_dataset(spec, split="val", train=False, views=config.val_views, **arguments)
    return _data_sources(train_set, val_set)


def _edit_records(root, change):
    manifest = root / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest.write_text("".join(json.dumps(change(entry)) + "\n" for entry in entries))


def _convention(entry):
    return {**entry, "backward_map_convention": "flattener"}


def _compact_scale(root):
    settings = json.loads((root / "compact.json").read_text())
    (root / "compact.json").write_text(json.dumps({**settings, "scale": 0.25}))


CASES = {
    "backward_map_convention": ("converted", lambda root: _edit_records(root, _convention)),
    "compact_scale": ("syntheticdoc_compact", _compact_scale),
}


def _config(tmp_path, model=TINY):
    return TrainConfig(
        data_spec=str(tmp_path / "spec.json"),
        output_dir=str(tmp_path / "run"),
        track="release",
        model=model,
    )


def _resume_payload(config, spec, sources):
    return {
        "train_config": config.to_dict(),
        "track": "release",
        "data_spec_sha256": spec.sha256,
        "data_sources": sources,
        "code": CODE,
    }


def test_resume_refuses_records_read_in_another_coordinate_convention(tmp_path):
    for case, (kind, change) in CASES.items():
        root = tmp_path / case
        root.mkdir()
        _write_samples(root / "data", compact=kind == "syntheticdoc_compact")
        spec = _signature_spec(root, kind)
        config = _config(root)
        before = _sources(spec, config)
        payload = _resume_payload(config, spec, before)
        _check_resume(config, payload, spec, _sources(spec, config), CODE)  # unchanged data
        change(root / "data")
        after = _sources(spec, config)
        with pytest.raises(FlattenerError, match="resume") as error:
            _check_resume(config, payload, spec, after, CODE)
        assert error.value.reason_code == "resume_mismatch", case
        before_signature = validation_signature(config, before["val"])
        assert validation_signature(config, after["val"]) != before_signature, case


def test_resume_refuses_other_target_code_and_conventions(tmp_path, monkeypatch):
    _write_samples(tmp_path / "data")
    spec = _signature_spec(tmp_path, "converted")
    config = _config(tmp_path)
    sources = _sources(spec, config)
    monkeypatch.setitem(dewarp_data.TARGET_CONVENTIONS, "version", 2)
    assert _sources(spec, config) != sources
    monkeypatch.undo()
    finer = DewarpModelConfig(input_size=(64, 48), grid_size=(9, 7), num_filter=4)
    assert _sources(spec, _config(tmp_path, finer))["val"][0]["grid_size"] == [9, 7]
    payload = _resume_payload(config, spec, sources)
    _check_resume(config, payload, spec, sources, code_record())
    source = inspect.getsource

    def edited(part):  # coords.py with one more statement
        return source(part) + "\nEXTRA = 1\n" if part is coords else source(part)

    monkeypatch.setattr(inspect, "getsource", edited)
    changed = code_record()
    assert changed["targets_sha256"] != CODE["targets_sha256"]
    with pytest.raises(FlattenerError, match="target code"):
        _check_resume(config, payload, spec, sources, changed)
    # A checkpoint written before the digest resumes only with the same coords.py bytes.
    older = {key: value for key, value in CODE.items() if key != "targets_sha256"}
    _check_resume(config, {**payload, "code": older}, spec, sources, changed)
    files = {**older["files"], "coords.py": "0" * 64}
    with pytest.raises(FlattenerError, match="predates the target-code digest"):
        _check_resume(config, {**payload, "code": {**older, "files": files}}, spec, sources, CODE)
