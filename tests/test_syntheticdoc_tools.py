"""SyntheticDoc data tooling (tools/syntheticdoc): licenses, split, sample conversion and
the compact training copy."""

import hashlib
import io
import json
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
import fetch  # noqa: E402
import sdcommon  # noqa: E402
import sdlicense  # noqa: E402
import sdsplit  # noqa: E402


def test_license_normalization():
    cc = sdlicense.cc_license
    assert cc("https://creativecommons.org/licenses/by/4.0/legalcode", "CC BY") == (
        "CC-BY-4.0",
        "clean",
    )
    assert cc("http://creativecommons.org/licenses/by-4.0/", "CC BY 4.0") == ("CC-BY-4.0", "clean")
    assert cc("https://creativecommons.org/licenses/by-sa/4.0/deed.de", "CC BY SA")[1] == "sa"
    assert cc("https://creativecommons.org/licenses/by-nc/3.0/", "CC BY NC") == (
        "CC-BY-NC-3.0",
        "nc",
    )
    assert cc("https://creativecommons.org/licenses/by-nc-sa/4.0/", "")[1] == "nc"
    assert cc("", "CC BY NC SA")[1] == "nc"
    assert cc("", "CC BY-ND")[1] == "unknown"
    assert cc("", "all rights reserved")[1] == "unknown"
    assert sdlicense.background_class("CC0") == ("CC0-1.0", "clean")
    assert sdlicense.background_class("CC-BY") == ("CC-BY", "clean")
    assert sdlicense.background_class("CC BY-NC-ND 2.0")[1] == "nc"
    assert sdlicense.background_class(None)[1] == "unknown"
    assert sdlicense.combine("clean", "sa") == "sa"
    assert sdlicense.combine("sa", "unknown") == "unknown"
    assert sdlicense.combine("unknown", "nc", "clean") == "nc"


def test_split_is_group_disjoint_and_deterministic():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(6000):
        split = "test" if k % 50 == 0 else ("val" if k % 10 == 0 else "train")
        source = ["arxiv", "doab", "res_txt2img"][k % 3]
        doc = f"{source}:{int(rng.integers(0, 400 if source != 'res_txt2img' else 5000))}"
        mesh = f"fam{k % 6}/{int(rng.integers(0, 1500))}"
        status = "success" if k % 97 else "incomplete"
        rows.append((f"{split}/{k:07d}", split, status, source, "clean", doc, mesh))
    out, info = sdsplit.assign(rows, val_fraction=0.02)
    assert out == sdsplit.assign(rows, val_fraction=0.02)[0]
    by = {r[0]: r for r in rows}
    train = [by[u] for u, (s, _) in out.items() if s == "train"]
    val = [by[u] for u, (s, _) in out.items() if s == "val"]
    assert {r[5] for r in train}.isdisjoint({r[5] for r in val})
    assert {r[6] for r in train}.isdisjoint({r[6] for r in val})
    assert all(out[r[0]][0] == "official_test" for r in rows if r[1] == "test")
    assert all(
        out[r[0]][0] == "excluded_invalid" for r in rows if r[2] != "success" and r[1] != "test"
    )
    pool = sum(1 for r in rows if r[1] != "test" and r[2] == "success")
    assert 0.01 < len(val) / pool < 0.035
    assert info["leak_check"] == {"doc_groups_train_and_val": 0, "mesh_groups_train_and_val": 0}


def test_convert_sample_full_resolution_and_fp16():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (1440, 1024, 3), dtype=np.uint8)
    png = io.BytesIO()
    Image.fromarray(image).save(png, "PNG", compress_level=1)
    bm = rng.uniform(-1, 1, (177, 121, 2)).astype(np.float32)
    npy = io.BytesIO()
    np.save(npy, bm)
    jpeg, bm16_bytes, info = fetch.convert_sample(png.getvalue(), npy.getvalue())
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.format == "JPEG" and im.size == (1024, 1440) and im.mode == "RGB"
    bm16 = np.load(io.BytesIO(bm16_bytes))
    assert bm16.dtype == np.float16 and bm16.shape == (177, 121, 2)
    assert np.abs(bm16.astype(np.float32) - bm).max() <= 2.0**-11 + 1e-9
    assert info["backward_map"]["fp16_max_error_render_px_xy"][0] <= 0.25 + 1e-6
    wrong = io.BytesIO()
    np.save(wrong, bm[:100])
    with pytest.raises(sdcommon.ConsistencyError):
        fetch.convert_sample(png.getvalue(), wrong.getvalue())


# Compact training copy (tools/syntheticdoc/compact.py) on a tiny download

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
        with Image.open(image) as opened:
            assert opened.size == (57, 81) and entry["image_size"] == [81, 57]
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
