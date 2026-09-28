"""OrientNet-1 data builder (tools/orient) and trainer (flattener.scan.orient_train)."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError
from flattener.dewarp.data import DataSpec
from flattener.scan import orient
from flattener.scan.orient import SourceSampler, ViewGeometry, clockwise, quantize
from flattener.scan.orient_train import (
    AugmentConfig,
    OrientData,
    OrientDataset,
    TrainConfig,
    _initialize,
    augmented_views,
    compute_losses,
    soft_targets,
    train,
)
from flattener.scan.shippable import gate_file

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "orient"
sys.path.insert(0, str(TOOLS))

import build_data  # noqa: E402
import pages  # noqa: E402

PAGE = (400, 283)
RENDER = (560, 560)
QUIET = AugmentConfig(
    margin=(0.0, 0.0),
    skew_deg=0.0,
    crop_skew_sd_deg=0.0,
    warp_prob=0.0,
    light_prob=0.0,
    shadow_prob=0.0,
    contrast=0.0,
    brightness=0.0,
    gamma=0.0,
    blur_prob=0.0,
    noise_prob=0.0,
    jpeg_prob=0.0,
)


def _asymmetric_page(size=PAGE):
    """Text-like lines in the top half and a mark at the bottom right."""
    height, width = size
    page = np.full(size, 236, np.uint8)
    rng = np.random.default_rng(5)
    for y in range(20, height // 2, 12):
        page[y : y + 4, 20 : int(rng.integers(width // 2, width - 20))] = 25
    page[height - 60 : height - 25, width - 70 : width - 25] = 10
    return page


def _render(page, angle, map_size=(40, 28)):
    """A photo of ``page`` turned clockwise by ``angle`` and its backward map."""
    hp, wp = page.shape
    height, width = RENDER
    center = np.array([width / 2, height / 2])
    rotation = clockwise(angle)
    inverse = rotation.T
    offset = np.array([wp / 2, hp / 2]) - inverse @ center
    geometry = ViewGeometry((float(height), float(width)), np.column_stack([inverse, offset]))
    sampler = SourceSampler(page, geometry)
    points = np.stack(np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5), -1)
    q = sampler.normalized(points)
    values = np.where((np.abs(q) > 1).any(-1), 90.0, sampler.lookup(q, 1.0))
    rgb = np.repeat(quantize(values)[..., None], 3, axis=2)
    hm, wm = map_size
    flat = np.stack(
        np.meshgrid((np.arange(wm) + 0.5) / wm * wp, (np.arange(hm) + 0.5) / hm * hp), -1
    )
    photo = center + (flat - [wp / 2, hp / 2]) @ rotation.T
    return rgb, (2 * photo / [width, height] - 1).astype(np.float32)


def _write_syntheticdoc(root, track="release"):
    root.mkdir(parents=True)
    lines = []
    for index, angle in enumerate((0.0, 90.0, 180.0, -20.0, 0.0, 30.0)):
        page = pages.composed_page(np.random.default_rng(index), PAGE)
        rgb, values = _render(page, angle)
        if index == 3:
            values[:] = np.nan  # no usable page: counted as skipped
        Image.fromarray(rgb).save(root / f"{index}.png")
        np.save(root / f"{index}.npy", values.astype(np.float16))
        entry = {
            "schema": "flattener-dewarp-sample-1",
            "sample_id": f"sd{index}",
            "split": "val" if index >= 4 else "train",
            "image": f"{index}.png",
            "backward_map": f"{index}.npy",
            "track": track,
            "license_class": "clean",
        }
        lines.append(json.dumps(entry) + "\n")
    (root / "manifest.jsonl").write_text("".join(lines))
    spec = {
        "schema": "flattener-dewarp-data-1",
        "name": "tiny-syntheticdoc",
        "track": "release",
        "sources": [
            {
                "name": f"sd-{split}",
                "kind": "converted",
                "root": root.name,
                "track": "release",
                "split": split,
                "license": "test pages",
            }
            for split in ("train", "val")
        ],
    }
    (root.parent / "dewarp.json").write_text(json.dumps(spec))


def _build_spec(directory, name="build.json", **changes):
    syntheticdoc = {"track": "release", "license": "test pages", "dewarp_spec": "dewarp.json"}
    spec = {
        "schema": "flattener-orient-build-1",
        "name": "orient-tiny",
        "track": "release",
        "long_side": 384,
        "chunk": 2,
        "sources": [
            {
                "name": "sd-flat",
                "kind": "syntheticdoc_flat",
                "part": [0, 1],
                "count": {"train": 4, "val": 2},
                **syntheticdoc,
            },
            {"name": "sd-crop", "kind": "syntheticdoc_crop", **syntheticdoc},
            {
                "name": "own",
                "kind": "documents",
                "track": "release",
                "license": build_data.OWN_LICENSE,
                "count": {"train": 3, "val": 1},
            },
            {
                "name": "blank",
                "kind": "text_free",
                "track": "release",
                "license": "Apache-2.0",
                "count": {"train": 2, "val": 1},
            },
        ],
        **changes,
    }
    path = directory / name
    path.write_text(json.dumps(spec))
    return path


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    directory = tmp_path_factory.mktemp("orient-build")
    _write_syntheticdoc(directory / "sd")
    spec = _build_spec(directory)
    summary = build_data.build(spec, directory / "out", workers=1)
    return directory, summary


def test_crops_and_training_views_follow_the_label_convention():
    from flattener.dewarp.coords import map_to_grid

    page = _asymmetric_page()
    # A SyntheticDoc photo turned 90 degrees clockwise is cut out upright.
    rgb, values = _render(page, 90.0)
    crop, box, found = pages.crop_syntheticdoc(rgb, map_to_grid(values, (40, 28)), 1024, 0.1)
    assert abs((found - 90.0 + 180) % 360 - 180) < 0.5
    height, width = crop.shape
    x0, y0, x1, y1 = ((np.array(box) + 1) / 2 * [width, height, width, height]).round().astype(int)
    region = np.asarray(Image.fromarray(crop[y0:y1, x0:x1]).resize(PAGE[::-1]), dtype=float)
    upright = np.corrcoef(region.ravel(), page.astype(float).ravel())[0, 1]
    flipped = np.corrcoef(region.ravel(), np.rot90(page, 2).astype(float).ravel())[0, 1]
    assert upright > 0.8 and upright > flipped + 0.3
    # Label k shows the page turned clockwise by k quarter turns.
    record = {"kind": "documents", "box": [-1.0, -1.0, 1.0, 1.0], "uniform": False}
    config = orient.OrientConfig()
    upright = augmented_views(record, page, 0, np.random.default_rng(0), QUIET, config)
    for k in (1, 2, 3):
        turned = augmented_views(record, page, k, np.random.default_rng(0), QUIET, config)
        difference = np.rot90(upright.global_view, -k).astype(int) - turned.global_view
        assert np.abs(difference).max() <= 1
        # np.rot90(view, k) restores the upright page.
        assert np.abs(np.rot90(turned.global_view, k).astype(int) - upright.global_view).max() <= 1
    noisy = augmented_views(record, page, 1, np.random.default_rng(3), AugmentConfig(), config)
    assert noisy.global_view.shape == (256, 256) and noisy.patches.shape == (4, 128, 128)


def test_dataset_items_and_targets(built):
    directory, _ = built
    data = OrientData.load(directory / "out")
    config = orient.OrientConfig()
    common = {"model": config, "augment": AugmentConfig(), "seed": 3}
    train_set = OrientDataset(data.root, data.split("train"), train=True, **common)
    assert train_set.names == ["sd-flat", "sd-crop", "own", "blank"]
    assert train_set.sizes == [3, 3, 3, 2]
    item = train_set[(len(train_set) - 1, 17)]
    assert item["label"] == -1 and item["global_view"].shape == (1, 256, 256)
    assert item["patches"].shape == (4, 1, 128, 128)
    again = train_set[(len(train_set) - 1, 17)]
    assert torch.equal(item["global_view"], again["global_view"])
    val_set = OrientDataset(data.root, data.split("val"), train=False, **common)
    assert len(val_set) == 5 * 4 + 1
    labels = soft_targets(torch.tensor([2, -1]), 0.1)
    torch.testing.assert_close(labels, torch.tensor([[0.025, 0.025, 0.925, 0.025], [0.25] * 4]))
    logits = torch.zeros(2, 4)
    config_ = TrainConfig(data=".", output_dir=".", aux_weight=0.5, label_smoothing=0.0)
    losses = compute_losses((logits, logits, logits), torch.tensor([0, -1]), config_)
    assert losses["loss"].item() == pytest.approx(1.5 * np.log(4))


@pytest.mark.training
def test_training_smoke_records_a_release_lineage(built, tmp_path, monkeypatch):
    directory, _ = built
    torch.set_num_threads(2)
    config = TrainConfig(
        data=str(directory / "out"),
        output_dir=str(tmp_path / "run"),
        batch_size=4,
        total_steps=3,
        warmup_steps=1,
        num_workers=0,
        log_every=1,
        val_every=3,
        val_max_pages=3,
        checkpoint_every=0,
        amp=False,
        device="cpu",
    )
    summary = train(config, resume="none")
    assert summary["state"] == "completed" and summary["step"] == 3
    assert summary["evidence"]["training"] == "smoke_training"
    checkpoints = tmp_path / "run" / "checkpoints"
    best = json.loads((checkpoints / "best.json").read_text())
    sidecar = json.loads((checkpoints / f"{best['file']}.json").read_text())
    assert sidecar["track"] == "release" and sidecar["lineage"][0] == {"kind": "scratch", "seed": 0}
    assert sidecar["lineage"][1]["kind"] == "train" and sidecar["lineage"][1]["track"] == "release"
    assert orient.release_problems(sidecar) == []
    assert sidecar["data_spec_sha256"] == sidecar["lineage"][1]["data_spec_sha256"]
    assert gate_file(checkpoints / best["file"]).shippable
    assert sidecar["calibration"]["source"] == "synthetic_val"
    assert sidecar["metrics"]["samples"] == 8 and sidecar["metrics"]["text_free_samples"] == 1
    orienter = orient.Orienter.load(checkpoints / best["file"])
    assert (
        orienter.info["shippable"] and orienter.temperature == sidecar["calibration"]["temperature"]
    )
    log = [
        json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in log] == ["train", "train", "train", "val"]
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert (
        status["state"] == "completed" and status["evidence"]["measurement"] == "gt_crop_validation"
    )

    resumed = train(config)  # the finished run resumes at its last step
    assert resumed["step"] == 3
    with pytest.raises(FlattenerError) as caught:
        train(replace(config, lr=1e-3))
    assert caught.value.reason_code == "resume_mismatch"

    # Final validation without improvement must still save; dropout must resume exactly.
    monkeypatch.setattr("flattener.scan.orient_train.improves", lambda *args: False)
    split = replace(config, output_dir=str(tmp_path / "split"), max_steps=2)
    train(split, resume="none")
    resumed = train(replace(split, max_steps=3))
    expected = orient.load_checkpoint(summary["checkpoint"])
    actual = orient.load_checkpoint(resumed["checkpoint"])
    assert actual["step"] == expected["step"] == 3
    for key, value in expected["model_state"].items():
        assert torch.equal(value, actual["model_state"][key]), key


def test_a_fine_tune_start_carries_the_parent_data_spec(tmp_path):
    def spec(name):
        source = {"name": name, "kind": "converted", "root": "files", "track": "release"}
        source.update(split="train", license="Apache-2.0")
        return {
            "schema": "flattener-dewarp-data-1",
            "name": name,
            "track": "release",
            "sources": [source],
        }

    def record(data_spec, lineage):
        digest = DataSpec.from_dict(data_spec).sha256
        sources = {"train": [{"name": "own", "track": "release", "license_classes": {"own": 1}}]}
        steps = [{"kind": "train", "track": "release", "data_spec_sha256": digest}]
        return {
            "track": "release",
            "lineage": [*lineage, *steps],
            "data_spec": data_spec,
            "data_spec_sha256": digest,
            "data_sources": sources,
        }

    model = orient.OrientNet()
    parent = record(spec("parent"), [{"kind": "scratch", "seed": 0}])
    path = tmp_path / "parent.pt"
    orient.save_checkpoint(
        path,
        {
            "schema": orient.CHECKPOINT_SCHEMA,
            "step": 5,
            "model_config": model.config.to_dict(),
            "code": orient.code_record(),
            "model_state": model.state_dict(),
            **parent,
        },
    )
    config = TrainConfig(data=".", output_dir=".", init="checkpoint", init_path=str(path))
    start = _initialize(orient.OrientNet(), config)
    assert start["data_spec"] == parent["data_spec"]
    path.unlink()
    (tmp_path / "parent.pt.json").unlink()  # only the lineage entry is left
    assert orient.release_problems(record(spec("child"), [start])) == []
    bare = {key: value for key, value in start.items() if key != "data_spec"}
    assert orient.release_problems(record(spec("child"), [bare])) == [
        "ship gate: unverified_data_spec"
    ]


def _pages_source(tmp_path, rows, license_class="own"):
    root = tmp_path / "templates"
    root.mkdir(exist_ok=True)
    lines = [
        {"path": name, "split": "train", "track": "release", "license_class": license_class}
        for name in rows
    ]
    (root / "manifest.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    spec = json.loads(_build_spec(tmp_path).read_text())
    source = {"name": "templates", "kind": "pages", "root": "templates", "track": "release"}
    spec["sources"] = [{**source, "license": "own"}]
    (tmp_path / "pages.json").write_text(json.dumps(spec))
    return root, tmp_path / "pages.json"


def test_the_data_build_reads_pages_upright_and_refuses_research_and_changed_sources(tmp_path):
    _write_syntheticdoc(tmp_path / "sd", track="research")
    with pytest.raises(FlattenerError) as caught:
        build_data.build(_build_spec(tmp_path), tmp_path / "out", workers=1)
    assert caught.value.reason_code == "track_violation"
    root, spec = _pages_source(tmp_path, ["a.jpg"], license_class="CC BY-NC 4.0")
    with pytest.raises(SystemExit, match="release source"):
        build_data.build(spec, tmp_path / "out", workers=1)

    upright = np.full((80, 60), 230, np.uint8)
    upright[:20] = 20  # a dark band at the top of the page
    root, spec = _pages_source(tmp_path, ["a.jpg"])
    exif = Image.Exif()
    exif[0x0112] = 6  # stored turned; shown upright after a clockwise quarter turn
    Image.fromarray(np.rot90(upright, 1)).save(root / "a.jpg", exif=exif.tobytes())
    settings = {"long_side": 80}
    gray, _, _ = build_data.make_page(
        "pages",
        {"path": str(root / "a.jpg"), "license_class": "own", "track": "release"},
        np.random.default_rng(0),
        settings,
    )
    assert gray[: gray.shape[0] // 4].mean() < gray[-gray.shape[0] // 4 :].mean()

    Image.fromarray(upright).save(root / "b.png")
    build_data.build(spec, tmp_path / "pages", workers=1)
    build_data.build(spec, tmp_path / "pages", workers=1)  # an unchanged source resumes
    _pages_source(tmp_path, ["a.jpg", "b.png"])
    with pytest.raises(SystemExit, match="other input data"):
        build_data.build(spec, tmp_path / "pages", workers=1)
