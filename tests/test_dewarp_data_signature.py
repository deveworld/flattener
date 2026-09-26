"""Codex review F3: the data signature covers how targets are read from the files.

A resume compares ``data_sources`` (``DewarpDataset.describe``) with the checkpoint.
Records that keep their files and ids but change how the stored map is interpreted
(``backward_map_convention``, ``original_render_size``) or a compact copy whose
settings changed (``compact.json`` ``scale``) give other targets, so the resume is
refused and the validation signature changes.
"""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from flattener import FlattenerError
from flattener.dewarp.data import (
    COMPACT_SCHEMA,
    SAMPLE_SCHEMA,
    SPEC_SCHEMA,
    DataSpec,
    build_dataset,
)
from flattener.dewarp.model import DewarpModelConfig
from flattener.dewarp.train import TrainConfig, _check_resume, _data_sources, validation_signature

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
F3 = (
    "F3 (Codex review): _Source.describe digests sample ids, tracks, license classes, "
    "hashes and sizes; backward_map_convention, original_render_size and compact.json "
    "settings stay out of data_sources, so _check_resume accepts the changed data"
)
SIZE = (60, 44)


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


def _spec(tmp_path, kind):
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


def _render_size(entry):
    return {**entry, "original_render_size": list(SIZE)}


def _compact_scale(root):
    settings = json.loads((root / "compact.json").read_text())
    (root / "compact.json").write_text(json.dumps({**settings, "scale": 0.25}))


CASES = {
    "backward_map_convention": ("converted", lambda root: _edit_records(root, _convention)),
    "original_render_size": ("converted", lambda root: _edit_records(root, _render_size)),
    "compact_backward_map_convention": (
        "syntheticdoc_compact",
        lambda root: _edit_records(root, _convention),
    ),
    "compact_original_render_size": (
        "syntheticdoc_compact",
        lambda root: _edit_records(root, _render_size),
    ),
    "compact_scale": ("syntheticdoc_compact", _compact_scale),
}


@pytest.mark.xfail(strict=True, reason=F3)
@pytest.mark.parametrize("case", sorted(CASES))
def test_resume_refuses_records_read_in_another_coordinate_convention(tmp_path, case):
    kind, change = CASES[case]
    _write_samples(tmp_path / "data", compact=kind == "syntheticdoc_compact")
    spec = _spec(tmp_path, kind)
    config = TrainConfig(
        data_spec=str(tmp_path / "spec.json"),
        output_dir=str(tmp_path / "run"),
        track="release",
        model=TINY,
    )
    before = _sources(spec, config)
    payload = {
        "train_config": config.to_dict(),
        "track": "release",
        "data_spec_sha256": spec.sha256,
        "data_sources": before,
    }
    _check_resume(config, payload, spec, _sources(spec, config))  # unchanged data resumes
    change(tmp_path / "data")
    after = _sources(spec, config)
    with pytest.raises(FlattenerError, match="resume") as error:
        _check_resume(config, payload, spec, after)
    assert error.value.reason_code == "resume_mismatch"
    assert validation_signature(config, after["val"]) != validation_signature(config, before["val"])


@pytest.mark.parametrize("change", [_convention, _render_size], ids=["convention", "render_size"])
def test_the_convention_fields_change_the_targets(tmp_path, change):
    _write_samples(tmp_path / "data")
    spec = _spec(tmp_path, "converted")

    def target():
        dataset = build_dataset(
            spec, track="release", split="val", train=False, input_size=(64, 48), grid_size=(5, 4)
        )
        return dataset[0]["grid2d"]

    before = target()
    _edit_records(tmp_path / "data", change)
    assert (target() - before).abs().max() > 1e-3
