"""Dewarp trainer: configuration, schedule, sampler, checkpoints and a CPU smoke run."""

import csv
import json
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
    write_json,
)
from flattener.dewarp.data import SAMPLE_SCHEMA, SPEC_SCHEMA, DataSpec, ViewConfig, require_track
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.train import (
    DEFAULT_VIEWS,
    StepSampler,
    TrainConfig,
    _check_resume,
    _initialize,
    carry_best,
    code_record,
    compute_losses,
    improves,
    learning_rate,
    main,
    memory_stats,
    restore_scaler,
    save_due,
    train,
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


def test_train_config_round_trip_and_validation(tmp_path):
    config = _config(
        tmp_path,
        epe_reference_size=(1440, 1024),
        betas=[0.9, 0.99],
        val_views=[{"name": "tight", "margin": [0.0, 0.0]}, ViewConfig("frame", "full_frame")],
        val_select="frame",
        val_batch_size=4,
    )
    data = json.loads(json.dumps(config.to_dict()))
    assert TrainConfig.from_dict(data) == config
    assert config.val_views[0] == ViewConfig("tight", "page", (0.0, 0.0))
    assert config.selected_view == "frame" and _config(tmp_path).selected_view == "page"
    assert _config(tmp_path).val_views == DEFAULT_VIEWS
    assert _config(tmp_path).epe_reference_size == (712, 488)  # fixed across input sizes
    for bad in (
        {"track": "public"},
        {"schedule": "step"},
        {"init": "uvdoc"},
        {"batch_size": 0},
        {"model": DewarpModelConfig(grid_convention="uvdoc")},
        {"lr": 2.0},
        {"max_steps": 0},
        {"epe_reference_size": None},
        {"val_batch_size": 0},
        {"val_views": []},
        {"val_views": [ViewConfig("a"), ViewConfig("a")]},
        {"val_views": [{"name": "a", "crop": "tight"}]},
        {"val_select": "missing"},
    ):
        with pytest.raises(FlattenerError):
            replace(config, **bad)
    with pytest.raises(FlattenerError, match="unknown"):
        TrainConfig.from_dict({**data, "channels_last": True})


def test_learning_rate_schedules():
    config = TrainConfig(
        data_spec="s",
        output_dir="o",
        track="research",
        total_steps=100,
        warmup_steps=10,
        lr=1e-3,
        min_lr_ratio=0.1,
    )
    assert learning_rate(config, 0) == pytest.approx(1e-4)
    assert learning_rate(config, 9) == pytest.approx(1e-3)
    assert learning_rate(config, 100) == pytest.approx(1e-4)
    linear = replace(config, schedule="linear_decay", decay_start=0.5, min_lr_ratio=0.0)
    assert learning_rate(linear, 50) == pytest.approx(1e-3)
    assert learning_rate(linear, 75) == pytest.approx(5e-4)
    assert learning_rate(replace(config, schedule="constant"), 99) == pytest.approx(1e-3)


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


def test_validation_signature_covers_what_defines_a_validation_result(tmp_path):
    config = _config(tmp_path)
    val = [{"name": "own-val", "records_sha256": "a", "content_sha256": "b"}]
    reference = validation_signature(config, val)
    same = (
        replace(config, val_batch_size=7),  # no sample limit: batch size does not matter
        replace(config, num_workers=3, val_every=5, lr=1e-3),
    )
    for changed in same:
        assert validation_signature(changed, val) == reference
    different = (
        replace(config, val_max_batches=1),
        replace(config, val_views=[ViewConfig("page", "page", (0.1, 0.1))]),
        replace(config, val_select="docunet"),
        replace(config, epe_reference_size=(1440, 1024)),
    )
    for changed in different:
        assert validation_signature(changed, val) != reference
    assert validation_signature(config, [{**val[0], "content_sha256": "c"}]) != reference
    limited = replace(config, val_max_batches=1)
    assert validation_signature(replace(limited, val_batch_size=1), val) != validation_signature(
        limited, val
    )
    assert memory_stats(torch.device("cpu")) == {}


def test_default_views_cover_the_validation_modes_and_page_stays_primary(tmp_path):
    names = ["page", "docunet", "full_frame", "rot90", "rot270", "small_page"]
    assert [view.name for view in DEFAULT_VIEWS] == names
    crops = [view for view in DEFAULT_VIEWS if view.crop != "full_frame"]
    assert all(view.margin[0] >= 0.02 for view in crops)  # every side keeps a margin
    assert [view.rotation for view in DEFAULT_VIEWS] == [0, 0, 0, 90, 270, 0]
    assert _config(tmp_path).selected_view == "page"


def test_best_carries_over_only_under_the_same_validation_signature(tmp_path):
    config = _config(tmp_path)
    val = [{"name": "own-val", "records_sha256": "a", "content_sha256": "b"}]
    signature = validation_signature(config, val)
    best = {"step": 2, "epe_px": 3.5, "signature": signature}
    assert carry_best(best, signature) == (best, None)
    assert carry_best(None, signature) == (None, None)
    fewer = validation_signature(replace(config, val_max_batches=1), val)
    assert carry_best(best, fewer) == (None, {"step": 2, "epe_px": 3.5})
    assert improves(None, 4.0) and improves(best, 3.0)
    assert not improves(best, 3.5) and not improves(None, float("nan"))


def test_every_new_best_and_the_last_step_are_saved():
    # A new best at the closing validation of an already saved step is saved again.
    assert save_due(4, True, 4, 4, last=True)
    assert not save_due(4, False, 4, 4, last=True)
    assert save_due(3, False, 2, 2, last=True)
    assert save_due(2, False, 0, 2) and not save_due(3, False, 2, 2)
    assert save_due(3, True, 2, 0) and not save_due(4, False, 2, 0)


def test_scaler_restore_and_strict_json_for_non_finite_values(tmp_path):
    enabled = torch.amp.GradScaler("cpu", enabled=True)
    assert restore_scaler(enabled, {}) is False  # cpu/xpu checkpoint resumed with AMP
    assert restore_scaler(torch.amp.GradScaler("cpu", enabled=False), {}) is True
    assert restore_scaler(enabled, torch.amp.GradScaler("cpu").state_dict()) is True
    path = tmp_path / "status.json"
    write_json(path, {"grad_norm": float("inf"), "metrics": [float("nan"), 1.5]})
    assert json.loads(path.read_text()) == {"grad_norm": None, "metrics": [None, 1.5]}


def test_train_and_val_must_not_share_samples(tmp_path):
    _spec(tmp_path, split_field=False)  # both sources read every manifest entry
    with pytest.raises(FlattenerError, match="both train and val") as error:
        train(_config(tmp_path))
    assert error.value.reason_code == "invalid_data"
    assert json.loads((tmp_path / "run/status.json").read_text())["state"] == "failed"


def test_losses_are_masked_by_node_validity_and_3d_availability():
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


def test_cli_resolves_file_relative_paths_and_environment_variables(tmp_path, capsys, monkeypatch):
    _spec(tmp_path)
    config = _config(tmp_path).to_dict()
    config.update(data_spec="spec.json", output_dir="${FLATTENER_TEST_RUNS}/a")
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("FLATTENER_TEST_RUNS", str(tmp_path / "runs"))
    assert main(["--config", str(path), "--print-config", "--max-steps", "3"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["data_spec"] == str(tmp_path / "spec.json")
    assert printed["output_dir"] == str(tmp_path / "runs" / "a") and printed["max_steps"] == 3
    monkeypatch.delenv("FLATTENER_TEST_RUNS")
    with pytest.raises(FlattenerError, match="FLATTENER_TEST_RUNS"):
        main(["--config", str(path), "--print-config"])


@pytest.mark.training
def test_two_step_cpu_training_smoke_and_exact_resume(tmp_path):
    _spec(tmp_path)
    first = train(_config(tmp_path, max_steps=2))
    assert first["state"] == "stopped" and first["step"] == 2
    run = tmp_path / "run"
    assert json.loads((run / "status.json").read_text())["state"] == "stopped"
    log = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
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


@pytest.mark.training
def test_overflow_grad_norm_does_not_stop_training_and_best_is_saved(tmp_path, monkeypatch):
    _spec(tmp_path)
    clip = torch.nn.utils.clip_grad_norm_

    def overflow(parameters, max_norm, *args, **kwargs):  # what an AMP overflow step returns
        clip(parameters, max_norm, *args, **kwargs)
        return torch.tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", overflow)
    result = train(_config(tmp_path, max_steps=3, val_every=1, checkpoint_every=0))
    assert result["state"] == "stopped" and result["step"] == 3
    run = tmp_path / "run"
    log = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
    assert all(r["grad_norm"] is None for r in log if r["kind"] == "train")
    best = json.loads((run / "checkpoints/best.json").read_text())
    assert best["step"] == result["best"]["step"]  # every new best is saved
    assert (run / "checkpoints" / best["file"]).is_file()
    assert load_checkpoint(run / "checkpoints" / best["file"])["step"] == best["step"]


def test_example_configs_parse_and_respect_tracks():
    examples = Path(__file__).resolve().parents[1] / "examples/dewarp"
    for path in sorted(examples.glob("train-*.json")):
        config = TrainConfig.from_dict(json.loads(path.read_text()))
        spec = DataSpec.load(examples / config.data_spec)
        require_track(spec, config.track)
        assert spec.track == config.track


@pytest.mark.training
def test_best_found_at_the_final_validation_is_saved(tmp_path):
    _spec(tmp_path)
    # The last step is saved by checkpoint_every before any validation; the closing
    # validation then finds the first best, which must be saved and pointed to.
    result = train(_config(tmp_path, val_every=0, checkpoint_every=4))
    assert result["state"] == "completed" and result["best"]["step"] == 4
    directory = tmp_path / "run/checkpoints"
    best = json.loads((directory / "best.json").read_text())
    assert best["file"] == "step_00000004.pt" and best["metrics"]["view"] == "page"
    assert load_checkpoint(directory / best["file"])["best"]["step"] == 4


@pytest.mark.training
def test_resume_with_other_validation_settings_starts_a_new_best(tmp_path):
    _spec(tmp_path)
    first = train(_config(tmp_path, max_steps=2))
    old = load_checkpoint(first["checkpoint"])["best"]
    assert old["step"] == 2
    same = train(_config(tmp_path, max_steps=3, val_every=1))
    assert "best_reset" not in load_checkpoint(same["checkpoint"])["resumes"][-1]
    resumed = train(_config(tmp_path, val_max_batches=1))
    checkpoint = load_checkpoint(resumed["checkpoint"])
    assert checkpoint["resumes"][-1]["best_reset"]["step"] in (2, 3)
    assert checkpoint["best"]["step"] == 4 and checkpoint["best"]["samples"] == 2
    assert checkpoint["best"]["signature"] != old["signature"]
