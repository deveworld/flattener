"""Page gate data (``tools/data/pages``) and trainer (``regions.train``): scenes, a build and a
short run whose checkpoint passes the release validation."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener.regions import train
from flattener.regions.context import load_context_segmenter

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "data" / "pages"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate_build = _module("gate_build_data", TOOLS / "build_data.py")
scenes = gate_build.scenes


def test_scenes_mark_the_paper_and_cut_out_thumbs():
    page = (Image.new("RGB", (300, 400), (250, 250, 245)), scenes.flat_grid())
    plain = scenes.SceneConfig(object_count=(1.0,), thumb_prob=0.0, closeup_prob=0.0)
    image, mask, info = scenes.scene(np.random.default_rng(1), plain, page)
    assert image.shape[:2] == mask.shape and 0.02 < mask.mean() < 0.95
    assert not info["closeup"] and info["thumbs"] == 0
    close = scenes.SceneConfig(object_count=(1.0,), closeup_prob=1.0, closeup_zoom=(3.4, 3.5))
    _, mask, info = scenes.scene(np.random.default_rng(2), close, page)
    edges = [mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()]
    assert info["closeup"] and sum(edges) >= 2 and mask.mean() > 0.3
    # The same draws up to the thumbs: a thumb only removes page from the mask.
    thumbs = scenes.SceneConfig(object_count=(1.0,), closeup_prob=0.0, thumb_prob=1.0)
    for seed in range(3):
        _, full, _ = scenes.scene(np.random.default_rng(seed), plain, page)
        _, cut, info = scenes.scene(np.random.default_rng(seed), thumbs, page)
        assert info["thumbs"] >= 1 and cut.sum() < full.sum() and not (cut & ~full).any()
    _, empty, info = scenes.scene(np.random.default_rng(3), plain, None)
    assert not empty.any() and info["page"] is False


@pytest.mark.training
def test_a_short_run_on_a_small_build_is_release(tmp_path):
    spec = {
        "schema": gate_build.BUILD_SCHEMA,
        "name": "tiny",
        "track": "release",
        "chunk": 4,
        "scene": {"long_side": 96},
        "sources": [
            {"name": "own", "kind": "documents_scene", "track": "release", "license": "own",
             "count": {"train": 6, "val": 2}, "document_prob": 0.0},
            {"name": "none", "kind": "negative_scene", "track": "release", "license": "own",
             "count": {"train": 2, "val": 1}},
        ],
    }  # fmt: skip
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    summary = gate_build.build(tmp_path / "spec.json", tmp_path / "data", workers=1)
    assert summary["counts"] == {"none/train": 2, "none/val": 1, "own/train": 6, "own/val": 2}
    row = json.loads((tmp_path / "data" / "manifest.jsonl").read_text().splitlines()[0])
    assert (tmp_path / "data" / row["mask"]).is_file()
    config = train.TrainConfig(
        data=str(tmp_path / "data"), output_dir=str(tmp_path / "run"), input_size=(32, 24),
        channels=(4, 8, 8, 8), batch_size=4, total_steps=4, warmup_steps=1, num_workers=0,
        device="cpu", amp=False, log_every=2, val_every=2,
    )  # fmt: skip
    best = train.train(config)
    assert set(best["sources"]) == {"own", "none"}
    gate = load_context_segmenter(tmp_path / "run" / "best.pt")
    assert gate.metadata["provenance"]["training"]["step"] in (2, 4)
    # Scores: IoU for a page, the unclaimed share for a scene without one.
    logits = torch.full((2, 1, 4, 4), -5.0)
    logits[0, 0, :2] = 5.0
    target = torch.zeros(2, 1, 4, 4)
    target[0, 0, :1] = 1.0
    assert train.row_scores(logits, target).tolist() == [0.5, 1.0]
