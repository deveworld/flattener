"""Dewarp trainer: sampler, checkpoints, resume, validation, losses and a CPU smoke run."""

import contextlib
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import FlattenerError
from flattener.dewarp.checkpoint import (
    CHECKPOINT_SCHEMA,
    commit_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from flattener.dewarp.data import SAMPLE_SCHEMA, SPEC_SCHEMA, DataSpec, ViewConfig, require_track
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.train import (
    DEFAULT_VIEWS,
    StepSampler,
    TrainConfig,
    _check_resume,
    _initialize,
    axis_alignment_loss,
    code_record,
    compute_losses,
    improves,
    page_space_loss,
    train,
    validate,
    validation_signature,
)

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)


def _spec(tmp_path, track="release", split_field=True):
    root = tmp_path / "data"
    root.mkdir()
    lines = []
    rng = np.random.default_rng(0)
    base = np.stack(np.meshgrid(np.linspace(-0.7, 0.7, 12), np.linspace(-0.8, 0.8, 18)), -1)
    for index in range(6):
        Image.fromarray(rng.integers(0, 256, (60, 44, 3), dtype=np.uint8)).save(
            root / f"{index}.png"
        )
        np.save(root / f"{index}.npy", (base + rng.normal(0, 0.01, base.shape)).astype(np.float16))
        split = "val" if index >= 4 else "train"
        entry = {
            "schema": SAMPLE_SCHEMA,
            "sample_id": str(index),
            "split": split,
            "image": f"{index}.png",
            "backward_map": f"{index}.npy",
            "track": "release",
            "license_class": "clean",
        }
        if not split_field:
            del entry["split"]
        lines.append(json.dumps(entry) + "\n")
    (root / "manifest.jsonl").write_text("".join(lines))
    sources = [
        {
            "name": f"own-{split}",
            "kind": "converted",
            "root": "data",
            "track": "release",
            "license": "Apache-2.0",
            "split": split,
        }
        for split in ("train", "val")
    ]
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps({"schema": SPEC_SCHEMA, "name": "tiny", "track": track, "sources": sources})
    )
    return path


def _config(tmp_path, name="run", **changes):
    values = dict(
        data_spec=str(tmp_path / "spec.json"),
        output_dir=str(tmp_path / name),
        track="release",
        model=TINY,
        batch_size=2,
        total_steps=4,
        warmup_steps=1,
        num_workers=0,
        device="cpu",
        log_every=1,
        val_every=2,
        checkpoint_every=2,
        loss_recon=1.0,
        recon_start_step=1,
    )
    values.update(changes)
    return TrainConfig(**values)


def _payload(step, **extra):
    model = UVDocNet(TINY)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "step": step,
        "track": "release",
        "model_config": TINY.to_dict(),
        "lineage": [{"kind": "scratch"}],
        "data_spec_sha256": "0" * 64,
        "code": {"sha256": "1" * 64},
        "metrics": {},
        "model_state": model.state_dict(),
        **extra,
    }


def test_step_sampler_is_deterministic_resumable_and_weighted():
    def take(sampler, count):
        iterator = iter(sampler)
        return [next(iterator) for _ in range(count)]

    stream = take(StepSampler([3, 4], None, seed=5), 21)
    assert [position for _, position in stream] == list(range(21))
    assert sorted(index for index, _ in stream[:7]) == list(range(7))
    assert take(StepSampler([3, 4], None, seed=5, start=9), 12) == stream[9:]
    weighted = take(StepSampler([10, 1000], [0.5, 0.5], seed=1), 2000)
    assert 800 < sum(index < 10 for index, _ in weighted) < 1200
    assert take(StepSampler([10, 1000], [0.5, 0.5], seed=1, start=100), 5) == weighted[100:105]


def test_checkpoint_commit_verify_prune_and_tamper_detection(tmp_path):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3, 4):
        commit_checkpoint(
            directory, _payload(step, metrics={"epe_px": step}), keep=2, is_best=step == 1
        )
    assert sorted(p.name for p in directory.glob("*.pt")) == [
        "step_00000001.pt",
        "step_00000003.pt",
        "step_00000004.pt",
    ]
    latest = latest_checkpoint(directory)
    sidecar = json.loads((directory / "step_00000004.pt.json").read_text())
    assert latest.name == "step_00000004.pt" and sidecar["track"] == "release"
    assert "model_state" not in sidecar and len(sidecar["sha256"]) == 64
    payload = load_checkpoint(latest)
    reference = _payload(4)["model_state"]
    assert payload["model_state"].keys() == reference.keys()
    with latest.open("r+b") as stream:
        stream.seek(-8, 2)
        stream.write(b"tampered")
    with pytest.raises(FlattenerError) as error:
        load_checkpoint(latest)
    assert error.value.reason_code == "checksum_mismatch"
    with pytest.raises(FlattenerError, match="schema"):
        save_checkpoint(tmp_path / "x.pt", {"step": 1})


def test_resume_compatibility_and_initialization_lineage(tmp_path):
    spec_path = _spec(tmp_path)
    spec = DataSpec.load(spec_path)
    config = _config(tmp_path)
    source = {"name": "own-train", "samples": 4, "records_sha256": "a", "targets_sha256": "t"}
    sources = {"train": [source], "val": []}
    code = code_record()
    payload = {
        "train_config": config.to_dict(),
        "track": "release",
        "data_spec_sha256": spec.sha256,
        "data_sources": sources,
        "code": code,
    }
    _check_resume(
        replace(config, max_steps=2, num_workers=3, device="cuda"), payload, spec, sources, code
    )
    _check_resume(  # validation settings may change; they start a new best instead
        replace(config, val_max_batches=1, val_views=[ViewConfig("x")]),
        payload,
        spec,
        sources,
        code,
    )
    # The validation signature (which marks a new best) covers what defines a result.
    val = [{"name": "own-val", "records_sha256": "a", "content_sha256": "b"}]
    reference = validation_signature(config, val)
    for same in (
        replace(config, val_batch_size=7),  # no sample limit: batch size does not matter
        replace(config, num_workers=3, val_every=5, lr=1e-3),
    ):
        assert validation_signature(same, val) == reference
    for changed in (
        replace(config, val_max_batches=1),
        replace(config, val_views=[ViewConfig("page", "page", (0.1, 0.1))]),
        replace(config, val_select="docunet"),
        replace(config, epe_reference_size=(1440, 1024)),
    ):
        assert validation_signature(changed, val) != reference
    assert validation_signature(config, [{**val[0], "content_sha256": "c"}]) != reference
    limited = replace(config, val_max_batches=1)
    assert validation_signature(replace(limited, val_batch_size=1), val) != validation_signature(
        limited, val
    )
    for changed in (replace(config, lr=1e-3), replace(config, seed=1)):
        with pytest.raises(FlattenerError, match="resume"):
            _check_resume(changed, payload, spec, sources, code)
    with pytest.raises(FlattenerError, match="resume"):
        _check_resume(config, {**payload, "data_spec_sha256": "x"}, spec, sources, code)
    reconverted = {"train": [{**sources["train"][0], "records_sha256": "b"}], "val": []}
    with pytest.raises(FlattenerError, match="data sources"):
        _check_resume(config, payload, spec, reconverted, code)
    legacy = {"train": [{k: v for k, v in source.items() if k != "targets_sha256"}], "val": []}
    with pytest.raises(FlattenerError, match="predates the target-convention signature"):
        _check_resume(config, {**payload, "data_sources": legacy}, spec, sources, code)
    research = tmp_path / "research.pt"
    save_checkpoint(research, _payload(7, track="research"))
    with pytest.raises(FlattenerError) as error:
        _initialize(UVDocNet(TINY), replace(config, init="checkpoint", init_path=str(research)))
    assert error.value.reason_code == "track_violation"
    lineage = _initialize(
        UVDocNet(TINY),
        replace(config, track="research", init="checkpoint", init_path=str(research)),
    )
    assert lineage["kind"] == "checkpoint" and lineage["track"] == "research"
    assert lineage["sha256"] == json.loads((tmp_path / "research.pt.json").read_text())["sha256"]
    assert "data_spec" not in lineage  # the parent recorded none
    # A release fine-tune carries its parent's data spec, so the ship gate can check the
    # parent's training step without the parent's sidecar.
    release = tmp_path / "release.pt"
    save_checkpoint(release, _payload(9, data_spec=spec.to_dict(), data_spec_sha256=spec.sha256))
    lineage = _initialize(
        UVDocNet(TINY), replace(config, init="checkpoint", init_path=str(release))
    )
    assert lineage["data_spec"] == spec.to_dict()
    assert DataSpec.from_dict(lineage["data_spec"]).sha256 == lineage["data_spec_sha256"]
    # The weights do not depend on the sizes: a checkpoint trained at another input and
    # grid size initializes a larger model, and the lineage keeps the parent's sizes.
    larger = UVDocNet(DewarpModelConfig(input_size=(96, 64), grid_size=(6, 4), num_filter=4))
    lineage = _initialize(
        larger, replace(config, track="research", init="checkpoint", init_path=str(research))
    )
    assert lineage["model_config"] == TINY.to_dict()
    parent = load_checkpoint(research)["model_state"]
    assert all(torch.equal(v, parent[k]) for k, v in larger.state_dict().items())
    with torch.no_grad():
        grid2d, _ = larger.eval()(torch.rand(1, 3, 96, 64))
    assert grid2d.shape == (1, 2, 6, 4)
    # init="uvdoc": a release run needs the published upstream file (sha256), and a
    # dewarp checkpoint (research weights) is never accepted as upstream weights.
    fake_uvdoc = tmp_path / "best_model.pkl"
    torch.save({"model_state": UVDocNet(TINY).state_dict()}, fake_uvdoc)
    uvdoc = replace(config, init="uvdoc", init_path=str(fake_uvdoc))
    with pytest.raises(FlattenerError) as error:
        _initialize(UVDocNet(TINY), uvdoc)
    assert error.value.reason_code == "track_violation"
    lineage = _initialize(UVDocNet(TINY), replace(uvdoc, track="research"))
    assert lineage["kind"] == "uvdoc" and lineage["license"] == "unverified"
    for track in ("release", "research"):
        disguised = replace(uvdoc, track=track, init_path=str(research))
        with pytest.raises(FlattenerError) as error:
            _initialize(UVDocNet(TINY), disguised)
        assert error.value.reason_code == "invalid_init"


def test_train_and_val_must_not_share_samples(tmp_path):
    _spec(tmp_path, split_field=False)  # both sources read every manifest entry
    with pytest.raises(FlattenerError, match="both train and val") as error:
        train(_config(tmp_path))
    assert error.value.reason_code == "invalid_data"
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["state"] == "failed"
    assert status["evidence"]["measurement"] == "training_loss"  # labelled before any step


@pytest.mark.training
def test_two_step_cpu_training_smoke_and_exact_resume(tmp_path):
    _spec(tmp_path)
    first = train(_config(tmp_path, max_steps=2))
    assert first["state"] == "stopped" and first["step"] == 2
    run = tmp_path / "run"
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "stopped" and status["evidence"] == first["evidence"]
    log = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
    trained = [r["evidence"] for r in log if r["kind"] == "train"]
    assert {(e["measurement"], e["evidence_kind"]) for e in trained} == {
        ("training_loss", "smoke_training")
    }
    names = [view.name for view in DEFAULT_VIEWS]
    assert [r["kind"] for r in log] == ["train", "train"] + ["val"] * len(names)
    assert [(r["view"], r["selected"]) for r in log[2:]] == [(n, n == "page") for n in names]
    assert {r["track"] for r in log} == {"release"}
    with (run / "train_log.csv").open(newline="") as stream:
        assert {row["track"] for row in csv.DictReader(stream)} == {"release"}
    assert all(np.isfinite(r["loss"]) for r in log if r["kind"] == "train")
    checkpoint = load_checkpoint(first["checkpoint"])
    assert checkpoint["track"] == "release" and checkpoint["lineage"][0]["kind"] == "scratch"
    assert list(checkpoint["metrics"]["views"]) == names
    assert all(checkpoint["metrics"]["views"][name]["epe_px"] > 0 for name in names)
    views = checkpoint["metrics"]["views"]
    assert views["full_frame"]["evidence"]["measurement"] == "gt_full_frame_validation"
    assert views["rot90"]["evidence"]["measurement"] == "gt_crop_validation"
    assert checkpoint["metrics"]["evidence"] == views["page"]["evidence"]
    assert first["evidence"]["evidence_kind"] == "smoke_training"
    assert first["evidence"]["initialization"] == "scratch"
    assert checkpoint["best"]["signature"] == checkpoint["metrics"]["signature"]
    manifest = checkpoint["data_sources"]
    assert [s["license_classes"] for s in manifest["train"]] == [{"clean": 4}]
    assert [s["samples"] for s in manifest["val"]] == [2]
    assert checkpoint["code"]["files"].keys() >= {"model.py", "train.py", "data.py"}
    resumed = train(_config(tmp_path))
    assert resumed["state"] == "completed" and resumed["step"] == 4
    straight = train(_config(tmp_path, name="straight"))
    a = load_checkpoint(resumed["checkpoint"])
    b = load_checkpoint(straight["checkpoint"])
    assert a["resumes"][0]["step"] == 2 and not b["resumes"]
    for key, value in a["model_state"].items():
        assert torch.equal(value, b["model_state"][key]), key
    with pytest.raises(FlattenerError, match="checkpoints"):
        train(_config(tmp_path), resume="none")


def test_device_tallies_train_the_same_weights_as_host_checks(tmp_path, monkeypatch):
    """With a loss scaler the loop keeps its per-step checks on the device; a power-of-two
    loss scale leaves the gradients, so the weights and logged losses match a plain run."""
    _spec(tmp_path)
    plain = train(_config(tmp_path, max_steps=3, name="plain"))
    import flattener.dewarp.train as module

    monkeypatch.setattr(module, "_amp", lambda config, device: True)
    monkeypatch.setattr(module, "_autocast", lambda config, device: contextlib.nullcontext())
    scaled = train(_config(tmp_path, max_steps=3, name="scaled"))
    a, b = load_checkpoint(plain["checkpoint"]), load_checkpoint(scaled["checkpoint"])
    for key, value in a["model_state"].items():
        torch.testing.assert_close(b["model_state"][key], value, rtol=0, atol=0, msg=key)
    assert b["counters"] == a["counters"] == {"skipped": 0, "overflow_steps": 0}

    def losses(name):
        rows = (tmp_path / name / "train_log.jsonl").read_text().splitlines()
        return [json.loads(r)["loss"] for r in rows if json.loads(r)["kind"] == "train"]

    assert losses("scaled") == pytest.approx(losses("plain"), rel=1e-6)


def test_example_configs_parse_and_respect_tracks():
    examples = Path(__file__).resolve().parents[1] / "examples/dewarp"
    for path in sorted(examples.glob("train-*.json")):
        config = TrainConfig.from_dict(json.loads(path.read_text()))
        spec = DataSpec.load(examples / config.data_spec)
        require_track(spec, config.track)
        assert spec.track == config.track


# Targets without valid nodes.
#
# Losses skip invalid nodes and samples without a 3-D target. A validation sample whose
# target grid has no valid node has no endpoint error: validation leaves it out of every
# mean, counts it per view as ``not_evaluable``, and a view where no sample is evaluable
# has no EPE, so it never becomes the best checkpoint.


def _validation_config(**changes):
    values = dict(data_spec="s", output_dir="o", track="research", model=TINY, batch_size=2)
    values.update(changes)
    return TrainConfig(**values)


def _batch(valid_samples, sources):
    """One collated evaluation batch; ``valid_samples[i]`` says whether sample i has GT."""
    views = len(DEFAULT_VIEWS)
    valid = torch.zeros(len(sources), views, 1, *TINY.grid_size)
    for index, has_target in enumerate(valid_samples):
        valid[index] = float(has_target)
    image = torch.stack(  # sample i looks the same in every batch
        [
            torch.randint(0, 256, (views, 3, *TINY.input_size), generator=_seeded(index))
            for index in range(len(sources))
        ]
    ).to(torch.uint8)
    return {
        "image": image,
        "grid2d": torch.zeros(len(sources), views, 2, *TINY.grid_size),
        "valid": valid,
        "source": list(sources),
    }


def _seeded(seed):
    return torch.Generator().manual_seed(seed)


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _model():
    torch.manual_seed(0)
    return UVDocNet(TINY)


def test_invalid_targets_are_left_out_of_losses_and_validation():
    config = TrainConfig(
        data_spec="s", output_dir="o", track="research", model=TINY, loss_recon=1.0
    )
    target = torch.zeros(2, 2, 5, 4)
    valid = torch.ones(2, 1, 5, 4)
    valid[1, :, 0, 0] = 0
    prediction = torch.zeros(2, 2, 5, 4)
    prediction[1, :, 0, 0] = 100.0  # ignored: invalid node
    batch = {
        "grid2d": target,
        "valid": valid,
        "grid3d": torch.ones(2, 3, 5, 4),
        "has_grid3d": torch.tensor([True, False]),
        "image": torch.rand(2, 3, 64, 48),
    }
    grid3d = torch.ones(2, 3, 5, 4)
    grid3d[1] = 50.0  # ignored: sample without a 3-D target
    losses = compute_losses(config, 0, batch, prediction, grid3d)
    assert losses["loss_grid2d"] == 0 and losses["loss_grid3d"] == 0
    assert "loss_recon" in losses and torch.isfinite(losses["loss"])
    config = _validation_config()
    model = _model()
    device = torch.device("cpu")
    metrics = validate(model, [_batch([False, False], ["a", "a"])], config, device)
    assert _missing(metrics["epe_px"])
    for name, view in metrics["views"].items():
        assert _missing(view["epe_px"]), name
        assert view["not_evaluable"] == 2, name
        assert _missing(view["per_source_epe_px"]["a"]), name
    assert not improves(None, metrics["epe_px"])
    evaluable = validate(model, [_batch([True], ["a"])], config, device)
    mixed = validate(model, [_batch([True, False], ["a", "b"])], config, device)
    for name, view in mixed["views"].items():
        assert view["epe_px"] == pytest.approx(evaluable["views"][name]["epe_px"]), name
        assert view["not_evaluable"] == 1, name
        assert view["per_source_epe_px"]["a"] == pytest.approx(view["epe_px"]), name
        assert _missing(view["per_source_epe_px"]["b"]), name  # source b has no valid GT


def test_the_axis_loss_charges_bent_rows_and_columns_but_not_moved_ones():
    rows, cols = 9, 7
    y, x = torch.meshgrid(
        torch.linspace(-0.8, 0.8, rows), torch.linspace(-0.6, 0.6, cols), indexing="ij"
    )
    angle = torch.tensor(0.3)  # a turned page: the loss works in page coordinates
    target = torch.stack([x * angle.cos() - y * angle.sin(), x * angle.sin() + y * angle.cos()])[
        None
    ]
    valid = torch.ones(1, 1, rows, cols)
    assert axis_alignment_loss(target, target, valid) == pytest.approx(0.0, abs=1e-10)
    moved = target + torch.tensor([0.05, -0.03]).view(1, 2, 1, 1)  # every node alike
    assert axis_alignment_loss(moved, target, valid) == pytest.approx(0.0, abs=1e-9)
    bent = target.clone()
    normal = torch.stack([-angle.sin(), angle.cos()])  # the page's down direction
    bent[0, :, 4, 3] += 0.04 * normal  # the middle of one row pushed down the page
    loss = axis_alignment_loss(bent.requires_grad_(), target, valid)
    assert loss > 1e-5
    loss.backward()
    assert torch.isfinite(bent.grad).all()
    small = target * 0.3  # the same bend on a page a third the size costs the same
    small_bent = small.clone()
    small_bent[0, :, 4, 3] += 0.04 * normal
    assert axis_alignment_loss(small_bent, small, valid) == pytest.approx(loss.item(), rel=1e-3)
    valid[0, 0, 4, 3] = 0  # an invalid node does not count
    assert axis_alignment_loss(bent.detach(), target, valid) == pytest.approx(0.0, abs=1e-9)
    config = TrainConfig(data_spec="s", output_dir="o", track="research", model=TINY, loss_axis=1.0)
    batch = {
        "grid2d": target,
        "valid": torch.ones(1, 1, rows, cols),
        "has_grid3d": torch.tensor([False]),
    }
    losses = compute_losses(config, 0, batch, bent.detach(), None)
    assert losses["loss_axis"] > 0 and losses["loss"] > losses["loss_grid2d"] * 5.0


def test_the_page_loss_weighs_errors_where_the_page_is_foreshortened():
    rows, cols = 9, 7
    v = torch.linspace(-1, 1, rows)
    far = 0.5 * (v + 1) ** 2 / 2 - 0.5  # rows bunch up at the top (v = -1), as far away
    y, x = torch.meshgrid(far * 1.6, torch.linspace(-0.6, 0.6, cols), indexing="ij")
    target = torch.stack([x, y])[None]
    valid = torch.ones(1, 1, rows, cols)
    assert page_space_loss(target, target, valid) == pytest.approx(0.0, abs=1e-10)
    even = torch.stack(
        torch.meshgrid(
            torch.linspace(-0.8, 0.8, rows), torch.linspace(-0.6, 0.6, cols), indexing="ij"
        )[::-1]
    )[None]
    moved = even + torch.tensor([0.02, -0.01]).view(1, 2, 1, 1)
    # An evenly spaced page scores as the image-space L1 over the nodes it weighs.
    assert page_space_loss(moved, even, valid) == pytest.approx(0.03, rel=1e-4)
    near, bunched = target.clone(), target.clone()
    near[0, 1, 6, 3] += 0.01  # the same image-space error, near and far
    bunched[0, 1, 2, 3] += 0.01
    assert page_space_loss(bunched, target, valid) > 1.5 * page_space_loss(near, target, valid)
    loss = page_space_loss(bunched.requires_grad_(), target, valid)
    loss.backward()
    assert torch.isfinite(bunched.grad).all()
    config = TrainConfig(data_spec="s", output_dir="o", track="research", model=TINY, loss_page=2.0)
    batch = {"grid2d": target, "valid": valid, "has_grid3d": torch.tensor([False])}
    losses = compute_losses(config, 0, batch, near, None)
    assert losses["loss"] == pytest.approx(5.0 * losses["loss_grid2d"] + 2.0 * losses["loss_page"])
