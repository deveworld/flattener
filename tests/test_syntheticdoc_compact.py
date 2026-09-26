"""Compact SyntheticDoc training copy (tools/syntheticdoc/compact.py) on a tiny download."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flattener.dewarp.coords import flattener_to_syntheticdoc, pixels_to_flattener
from flattener.dewarp.data import SPEC_SCHEMA, DataSpec, build_dataset

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "syntheticdoc"
sys.path.insert(0, str(TOOLS))

import compact  # noqa: E402

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
            "--source",
            str(source),
            "--output",
            str(output),
            "--workers",
            "1",
            "--input-size",
            "48",
            "36",
            "--stats-samples",
            "8",
            *extra,
        ]
    )


def _manifest(output):
    lines = (output / "manifest.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_compact_copy_is_resized_hashed_and_readable_by_the_loader(tmp_path):
    source, output = _download(tmp_path), tmp_path / "cache"
    assert _run(source, output) == 0
    settings = json.loads((output / "compact.json").read_text())
    statistics = settings["scale_statistics"]
    # A page-tight crop covers 36 px of width at about 36 / (0.7 * 102) ~ 0.51; rounded up
    # to 1/16 that is 0.5625.
    needed = statistics["page_tight_scale_percentiles"]["75"]
    assert 0.5 < needed < settings["scale"] == 0.5625 and statistics["samples"] == 3
    entries = _manifest(output)
    assert [e["sample_id"][-7:] for e in entries] == ["0000001", "0000002", "0000009"]
    for entry in entries:
        image = output / entry["image"]
        assert Image.open(image).size == (57, 81) and entry["image_size"] == [81, 57]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == entry["image_sha256"]
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
    spec = DataSpec.from_dict(
        {
            "schema": SPEC_SCHEMA,
            "name": "c",
            "track": "release",
            "sources": [
                {
                    "name": name,
                    "kind": kind,
                    "root": str(root),
                    "track": "release",
                    "split": "train",
                    "license": "CC BY 4.0",
                }
                for name, kind, root in (
                    ("copy", "syntheticdoc_compact", output),
                    ("full", "converted", source),
                )
            ],
        }
    )
    dataset = build_dataset(
        spec, track="release", split="train", input_size=(48, 36), grid_size=(5, 4), train=False
    )
    copy, full = dataset.sources
    for index in range(2):
        a, b = copy.load(index), full.load(index)
        np.testing.assert_array_equal(a.grid, b.grid)
        assert a.size == (81, 57) and b.size == RENDER
        a.image.close(), b.image.close()


def test_rerun_keeps_unchanged_samples_and_rechecks_sources_and_outputs(tmp_path):
    source, output = _download(tmp_path), tmp_path / "cache"
    assert _run(source, output) == 0
    first = _manifest(output)
    assert _run(source, output) == 0
    counts = json.loads((output / "compact-manifest.json").read_text())["counts"]
    assert counts["converted_this_run"] == 0 and counts["kept"] == 3 and _manifest(output) == first
    # The download finishes a sample and re-writes another: both are converted.
    _write_sample(source, "train_001", "0000004", seed=7)
    _write_sample(source, "train_000", "0000002", seed=11)
    path = source / "samples/train_000/0000002.render.jpg"
    stamp = path.stat().st_mtime_ns + 10**9
    os.utime(path, ns=(stamp, stamp))
    assert _run(source, output) == 0
    counts = json.loads((output / "compact-manifest.json").read_text())["counts"]
    assert counts["converted_this_run"] == 2 and counts["samples"] == 4
    assert counts["incomplete_skipped"] == 0
    # --verify re-hashes kept outputs and replaces a corrupted one.
    image = output / _manifest(output)[0]["image"]
    data = bytearray(image.read_bytes())
    data[-10] ^= 0xFF
    image.write_bytes(bytes(data))
    assert _run(source, output, "--shards", "train_001") == 0
    assert [e["shard"] for e in _manifest(output)] == ["train_001", "train_001"]
    assert _run(source, output) == 0  # same size: kept without --verify
    assert json.loads((output / "compact-manifest.json").read_text())["counts"]["kept"] == 4
    assert _run(source, output, "--verify") == 0
    assert json.loads((output / "compact-manifest.json").read_text())["counts"]["kept"] == 3
    entry = _manifest(output)[0]
    assert hashlib.sha256(image.read_bytes()).hexdigest() == entry["image_sha256"]


def test_settings_are_fixed_by_the_first_run_and_tracks_are_enforced(tmp_path):
    source, output = _download(tmp_path), tmp_path / "cache"
    assert _run(source, output, "--scale", "0.75", "--stats-only") == 0
    assert json.loads((output / "compact.json").read_text())["scale"] == 0.75
    for extra in (("--quality", "80"), ("--scale", "0.5"), ("--track", "research")):
        with pytest.raises(SystemExit, match="compact.json fixes"):
            _run(source, output, *extra)
    _write_sample(source, "train_002", "0000003", seed=3, track="research")
    assert _run(source, output) == 1  # the research sample is reported and left out
    summary = json.loads((output / "compact-manifest.json").read_text())
    assert summary["counts"]["samples"] == 3 and "track 'research'" in summary["errors"][0]
    with pytest.raises(SystemExit):
        compact.parser().parse_args(["--source", "s", "--output", "o", "--workers", "9"])
