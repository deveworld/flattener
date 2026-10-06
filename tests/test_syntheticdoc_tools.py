"""SyntheticDoc data tooling (tools/data/syntheticdoc): licenses, split, sample conversion and
the compact training copy."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flattener.dewarp.coords import flattener_to_syntheticdoc, pixels_to_flattener
from flattener.dewarp.data import SPEC_SCHEMA, DataSpec, build_dataset

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "data" / "syntheticdoc"
sys.path.insert(0, str(TOOLS))

import compact  # noqa: E402

# Compact training copy (tools/data/syntheticdoc/compact.py) on a tiny download

RENDER = (144, 102)  # (height, width) of the fake upstream renders


def _map(seed):
    """A 18x12 raw SyntheticDoc map of an affine page covering about 70% of the render."""
    height, width = RENDER
    rng = np.random.default_rng(seed)
    corner = np.array([0.12 * width, 0.1 * height]) + rng.uniform(0, 4, 2)
    axes = np.array([[0.7 * width, 0.03 * width], [-0.02 * height, 0.75 * height]])
    u, v = np.meshgrid((np.arange(12) + 0.5) / 12, (np.arange(18) + 0.5) / 18)
    edges = corner + np.stack([u, v], -1) @ axes
    values = pixels_to_flattener(edges - 0.5, RENDER)
    return flattener_to_syntheticdoc(values, RENDER).astype(np.float16)


def _write_sample(root, shard, stem, *, seed=0, track="release", record=True):
    folder = root / "samples" / shard
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (*RENDER, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(folder / f"{stem}.render.jpg", quality=95)
    np.save(folder / f"{stem}.bm16.npy", _map(seed))
    if not record:
        return
    entry = {
        "schema": "flattener-dewarp-sample-1",
        "sample_id": f"syntheticdoc-train-{stem}",
        "split": "val" if stem.endswith("9") else "train",
        "image": f"{stem}.render.jpg",
        "backward_map": f"{stem}.bm16.npy",
        "backward_map_convention": "syntheticdoc",
        "original_render_size": list(RENDER),
        "track": track,
        "license_class": "clean" if track == "release" else "nc",
        "official": {"split": "train", "shard": shard, "sample_id": stem},
        "groups": {"doc_group": f"arxiv:{stem}", "mesh_group": "curve/00001"},
        "license": {"class": "clean", "document": "CC-BY-4.0", "background": "CC0-1.0"},
        "attribution": {"dataset": {"name": "SyntheticDoc"}},
        "conversion": {"image": {"bytes": (folder / f"{stem}.render.jpg").stat().st_size}},
        "syntheticdoc_metadata": {"camera": {"distance": 0.6}},
    }
    (folder / f"{stem}.json").write_text(json.dumps(entry))


def _download(tmp_path):
    root = tmp_path / "download"
    for index, (shard, stem) in enumerate(
        [("train_000", "0000001"), ("train_000", "0000002"), ("train_001", "0000009")]
    ):
        _write_sample(root, shard, stem, seed=index)
    _write_sample(root, "train_001", "0000004", seed=7, record=False)  # still downloading
    (root / "samples/train_001/0000005.json.tmp123-4").write_text("{")
    (root / "dataset-manifest.json").write_text(json.dumps({"complete": False}))
    return root


def _run(source, output, *extra):
    return compact.main(
        [
            "--source", str(source), "--output", str(output), "--workers", "1",
            "--input-size", "48", "36", "--stats-samples", "8", *extra,
        ]
    )  # fmt: skip


def test_compact_copy_is_readable_by_the_loader_and_keeps_its_settings_and_track(tmp_path):
    source, output = _download(tmp_path), tmp_path / "cache"
    assert _run(source, output) == 0
    settings = json.loads((output / "compact.json").read_text())
    statistics = settings["scale_statistics"]
    # A page-tight crop covers 36 px of width at about 36 / (0.7 * 102) ~ 0.51; rounded up
    # to 1/16 that is 0.5625.
    needed = statistics["page_tight_scale_percentiles"]["75"]
    assert 0.5 < needed < settings["scale"] == 0.5625 and statistics["samples"] == 3
    entries = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert [e["sample_id"][-7:] for e in entries] == ["0000001", "0000002", "0000009"]
    for entry in entries:
        image = output / entry["image"]
        with Image.open(image) as opened:
            assert opened.size == (57, 81) and entry["image_size"] == [81, 57]
        assert image.stat().st_size == entry["image_bytes"]
        original = source / "samples" / entry["shard"] / f"{entry['sample_id'][-7:]}.bm16.npy"
        assert (output / entry["backward_map"]).read_bytes() == original.read_bytes()
        assert entry["original_render_size"] == list(RENDER) and entry["track"] == "release"
    summary = json.loads((output / "compact-manifest.json").read_text())
    assert summary["counts"]["samples"] == 3 and summary["counts"]["incomplete_skipped"] == 1
    assert summary["complete"] is False and summary["license_classes"] == {"clean": 3}
    record = json.loads((output / entries[0]["meta"]).read_text())
    assert record["attribution"]["dataset"]["name"] == "SyntheticDoc"
    assert record["source"]["image_size"] == list(RENDER)
    # The dewarp loader reads the copy; its targets equal those of the full-size download.
    source_spec = {"track": "release", "split": "train", "license": "CC BY 4.0"}
    sources = [
        {"name": "copy", "kind": "syntheticdoc_compact", "root": str(output), **source_spec},
        {"name": "full", "kind": "converted", "root": str(source), **source_spec},
    ]
    spec = {"schema": SPEC_SCHEMA, "name": "c", "track": "release", "sources": sources}
    spec = DataSpec.from_dict(spec)
    dataset = build_dataset(
        spec, track="release", split="train", input_size=(48, 36), grid_size=(5, 4), train=False
    )
    copy, full = dataset.sources
    for index in range(2):
        a, b = copy.load(index), full.load(index)
        np.testing.assert_array_equal(a.grid, b.grid)
        assert a.size == (81, 57) and b.size == RENDER
        a.image.close(), b.image.close()
    # The first run fixes the settings; a research sample is reported and left out.
    assert _run(source, tmp_path / "fixed", "--scale", "0.75", "--stats-only") == 0
    assert json.loads((tmp_path / "fixed" / "compact.json").read_text())["scale"] == 0.75
    for extra in (("--quality", "80"), ("--scale", "0.5"), ("--track", "research")):
        with pytest.raises(SystemExit, match="compact.json fixes"):
            _run(source, output, *extra)
    _write_sample(source, "train_002", "0000003", seed=3, track="research")
    assert _run(source, output) == 1
    summary = json.loads((output / "compact-manifest.json").read_text())
    assert summary["counts"]["samples"] == 3 and "track 'research'" in summary["errors"][0]
