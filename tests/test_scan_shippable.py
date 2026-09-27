"""Ship gate: lineage walk, categories, the research stamp and the CLI.

The fixtures copy fields of real checkpoint sidecars and run-info files: the scratch
run on 409k pages, the UVDoc-start runs on 199k pages (normal and long), the
89x61-grid trial and the UVDoc-recipe reproduction. Run names inside the copied
records are the real folder names; outputs/models/scores.json gives their plain names.
"""

import copy
import hashlib
import json

import pytest
import torch

from flattener.contracts import FlattenerError, Status
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, SIDECAR_SCHEMA, save_checkpoint
from flattener.dewarp.data import DataSpec
from flattener.dewarp.model import UVDOC_WEIGHTS_SHA256, DewarpModelConfig, UVDocNet
from flattener.evidence import initialization
from flattener.scan.shippable import (
    CATEGORIES,
    EXTERNAL_WEIGHTS_SHA256,
    REASONS,
    gate_file,
    gate_verdict,
    load_weights,
    main,
    shippable,
)

SYNTHETICDOC_LICENSE = (
    "SyntheticDoc CC BY 4.0; documents CC BY, CC0 or ODC-BY, backgrounds CC0 or CC BY, meshes CC BY"
)
DOC3D_LICENSE = "Doc3D: MIT code; YES! Magazine textures CC BY-NC-ND 3.0; Laval HDR non-commercial"
UVDOC_DATA_LICENSE = (
    "UVDoc dataset: MIT repository; DeepFloyd IF textures (non-commercial), DTD backgrounds"
)


def _source(name, root, split="train", max_samples=None, **extra):
    source = {
        "ids_file": None,
        "kind": "syntheticdoc_compact",
        "license": SYNTHETICDOC_LICENSE,
        "max_samples": max_samples,
        "name": name,
        "root": root,
        "split": split,
        "track": "release",
        "weight": None,
    }
    return {**source, **extra}


def _spec(name, sources, track="release"):
    return {"name": name, "schema": "flattener-dewarp-data-1", "sources": sources, "track": track}


BASE = "${FLATTENER_CACHE}/syntheticdoc-release"
SPEC_199K = _spec(
    "release-clean-v2",
    [
        _source("syntheticdoc-clean-train", BASE),
        _source("syntheticdoc-clean-val", BASE, "val", 2000),
    ],
)
SPEC_409K = _spec(
    "release-clean-v3-extra",
    [
        _source("syntheticdoc-clean-train", BASE),
        _source("syntheticdoc-clean-train-extra", "${FLATTENER_CACHE}/syntheticdoc-release-extra"),
        _source("syntheticdoc-clean-val", BASE, "val", 2000),
    ],
)
RESEARCH = "${FLATTENER_DATA}/research"
SPEC_RESEARCH = _spec(
    "research-doc3d-uvdoc",
    [
        _source(
            "doc3d-train",
            f"{RESEARCH}/doc3d",
            kind="doc3d",
            license=DOC3D_LICENSE,
            track="research",
            weight=0.5,
        ),
        _source(
            "uvdoc-train",
            f"{RESEARCH}/uvdoc_final",
            kind="uvdoc",
            license=UVDOC_DATA_LICENSE,
            track="research",
            weight=0.5,
        ),
        _source(
            "doc3d-val",
            f"{RESEARCH}/doc3d",
            "val",
            2000,
            kind="doc3d",
            license=DOC3D_LICENSE,
            track="research",
        ),
    ],
    track="research",
)
SPEC_199K_SHA = "ba3418c5fdb553990c1a73e19b177d02f9df8d132062d9ee00de5454e84c3947"
SPEC_409K_SHA = "567ff319e4e8b5f9c6d9160f9222b31188b4736394b15179d267de9ab34b5bb9"
SPEC_RESEARCH_SHA = "857fb9c4787ac17d8deaebb35bbbaf79d5a20a30a9acf6846f8ab0e0f81c772f"
UVDOC_199K_SHA = "30e8eabfa98ba259078501850acea35d00cee68e5e339b57135743baa521c033"
UVDOC_199K_LONG_SHA = "47fabf17ddb59366bb26ce507ae87cbdd8e539be129559503971154582c10cd2"
SCRATCH_409K_SHA = "417e2ce9eadaf85bcc7b3b54f2e937c53513c9bb91f335db264e802ad321ac4d"
GRID89_SHA = "d31b814ebe5197b2b34cd46d54189e03bac60dae345dc28b6ba50afbf93cdbfb"
RECIPE_REPRO_SHA = "c01b437fcbe178287b16ffa20874281832e94bb6cfd778ade6edd29b1b8cf529"
CODE_UVDOC_199K = "0e192f24eae16c059a7ae5de30f68b15ea571bd3abcc5a90f3ed43d582967821"
CODE_UVDOC_199K_LONG = "9d3b5c2f41bbc1cb0c755fb1f807d883d17a9b7b1d0ab8c518b63fc4d6ad13ab"
CODE_SCRATCH_409K = "d6024bc229ebba5f7afafd668e7d943652dd343e9bc6b2f463d5b04a18c48546"
MODEL_CONFIG = {
    "architecture": "uvdocnet",
    "grid_convention": "flattener",
    "grid_size": [45, 31],
    "input_size": [712, 488],
    "kernel_size": 5,
    "num_filter": 32,
    "with_3d": True,
}
UVDOC_START = {
    "expected_sha256": UVDOC_WEIGHTS_SHA256,
    "kind": "uvdoc",
    "license": "MIT",
    "note": "weights MIT; upstream training data (Doc3D, UVDoc) has non-commercial parts",
    "path": "/workspace/uvdoc/model/best_model.pkl",
    "sha256": UVDOC_WEIGHTS_SHA256,
    "sha256_matches_upstream": True,
    "source": "https://github.com/tanguymagne/UVDoc@4c9b82b537057aff2526e6dd118a847cdd072e82",
}
SCRATCH = {"kind": "scratch", "seed": 0}


def _train(spec_sha256, code=CODE_UVDOC_199K, track="release"):
    return {
        "code_sha256": code,
        "data_spec_sha256": spec_sha256,
        "kind": "train",
        "started_utc": "2026-09-27T04:50:39+00:00",
        "track": track,
    }


def _manifest(spec):
    """``data_sources`` as ``DewarpDataset.describe`` writes it (the fields the gate reads)."""
    sources = {"train": [], "val": []}
    for source in spec["sources"]:
        license_class = "clean" if source["track"] == "release" else source["license"]
        entry = {
            "name": source["name"],
            "kind": source["kind"],
            "track": source["track"],
            "samples": 2000,
            "license_classes": {license_class: 2000},
        }
        sources[source["split"]].append(entry)
    return sources


def _sidecar(sha256, spec, spec_sha256, lineage, *, step, track="release"):
    return {
        "schema": SIDECAR_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "file": f"step_{step:08d}.pt",
        "sha256": sha256,
        "step": step,
        "track": track,
        "model_config": MODEL_CONFIG,
        "lineage": lineage,
        "data_spec": spec,
        "data_spec_sha256": spec_sha256,
        "data_sources": _manifest(spec),
    }


def _parent(sidecar, path):
    """A ``kind: "checkpoint"`` start as ``train._initialize`` records it."""
    return {
        "kind": "checkpoint",
        "path": path,
        "sha256": sidecar["sha256"],
        "step": sidecar["step"],
        "track": sidecar["track"],
        "model_config": MODEL_CONFIG,
        "data_spec_sha256": sidecar["data_spec_sha256"],
        "code_sha256": sidecar["lineage"][-1]["code_sha256"],
        "lineage": sidecar["lineage"],
    }


UVDOC_199K = _sidecar(
    UVDOC_199K_SHA, SPEC_199K, SPEC_199K_SHA, [UVDOC_START, _train(SPEC_199K_SHA)], step=46880
)
SCRATCH_409K = _sidecar(
    SCRATCH_409K_SHA,
    SPEC_409K,
    SPEC_409K_SHA,
    [SCRATCH, _train(SPEC_409K_SHA, CODE_SCRATCH_409K)],
    step=140625,
)
GRID89 = _sidecar(
    GRID89_SHA,
    SPEC_199K,
    SPEC_199K_SHA,
    [
        _parent(UVDOC_199K, "/workspace/init/R3/step_00046880.pt"),
        _train(SPEC_199K_SHA, CODE_UVDOC_199K_LONG),
    ],
    step=23440,
)
RECIPE_REPRO = _sidecar(
    RECIPE_REPRO_SHA,
    SPEC_RESEARCH,
    SPEC_RESEARCH_SHA,
    [SCRATCH, _train(SPEC_RESEARCH_SHA, track="research")],
    step=391272,
    track="research",
)


def _config(name, init, spec_path, init_path=None):
    return {
        "init": init,
        "init_path": init_path,
        "track": "release",
        "data_spec": spec_path,
        "output_dir": f"/workspace/runs/{name}",
        "seed": 0,
        "model": MODEL_CONFIG,
    }


def _run_info(name, init, spec, spec_sha256, best_sha256, step, *, lineage=None):
    return {
        "schema": "flattener-release-run-info-1",
        "name": name,
        "track": "release",
        "init": init,
        "config": _config(name, init["kind"], "/workspace/specs/data-release.json"),
        "data_spec_content": spec,
        "data_spec_sha256": spec_sha256,
        "lineage": lineage,
        "result": {
            "best_checkpoint": {"file": f"step_{step:08d}.pt", "sha256": best_sha256, "step": step},
            "state": "completed",
        },
    }


UVDOC_INIT = {
    "kind": "uvdoc",
    "license": "UVDoc MIT",
    "path": "/workspace/uvdoc/model/best_model.pkl",
    "sha256_expected": UVDOC_WEIGHTS_SHA256,
}
UVDOC_INIT_LONG = {**UVDOC_INIT, "same_as": "R3", "sha256": UVDOC_WEIGHTS_SHA256}
UVDOC_NOTE = {"kind": "uvdoc-init", "chain": "R3long <- UVDoc MIT weights"}
SCRATCH_409K_INFO = _run_info(
    "R4",
    {"kind": "scratch"},
    SPEC_409K,
    SPEC_409K_SHA,
    SCRATCH_409K_SHA,
    140625,
    lineage={"kind": "clean", "note": "scratch init; release data spec only"},
)
UVDOC_199K_LONG_INFO = _run_info(
    "R3long",
    {**UVDOC_INIT_LONG, "sha256_matches": True},
    SPEC_199K,
    SPEC_199K_SHA,
    UVDOC_199K_LONG_SHA,
    93760,
    lineage=UVDOC_NOTE,
)


def test_fixture_specs_hash_to_the_recorded_sha256():
    assert DataSpec.from_dict(SPEC_199K).sha256 == SPEC_199K_SHA
    assert DataSpec.from_dict(SPEC_409K).sha256 == SPEC_409K_SHA
    assert DataSpec.from_dict(SPEC_RESEARCH).sha256 == SPEC_RESEARCH_SHA
    assert set(REASONS) >= {"external_init", "external_weights", "unverified_data_spec"}


@pytest.mark.parametrize(
    "record",
    [SCRATCH_409K, SCRATCH_409K_INFO],
    ids=["scratch-409k-sidecar", "scratch-409k-run-info"],
)
def test_scratch_release_runs_are_shippable(record):
    verdict = shippable(record)
    assert verdict.shippable and not verdict.research
    assert verdict.category == "release" and verdict.reasons == ()
    assert verdict.track == "release" and verdict.initialization == "scratch"
    assert verdict.sha256 == record.get("sha256", SCRATCH_409K_SHA)
    assert [node["depth"] for node in verdict.chain] == [0, 1]
    assert verdict.chain[0]["training"][0]["data"] == "release"
    assert verdict.chain[-1] == {"depth": 1, "kind": "scratch"}
    payload = verdict.to_dict()
    assert payload["schema"] == "flattener-ship-gate-1"
    assert payload["shippable"] is True and payload["research"] is False
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "record", [UVDOC_199K, UVDOC_199K_LONG_INFO], ids=["uvdoc-199k", "uvdoc-199k-long-run-info"]
)
def test_uvdoc_initialized_release_runs_are_research(record):
    verdict = shippable(record)
    assert not verdict.shippable and verdict.research
    assert verdict.category == "research"
    assert verdict.reasons == ("external_init",)  # track and data are release
    assert verdict.track == "release" and verdict.initialization == "external_pretrained"
    assert verdict.chain[-1]["kind"] == "uvdoc" and verdict.chain[-1]["category"] == "external"


def test_parent_checkpoints_are_walked_to_the_start():
    verdict = shippable(GRID89)
    assert verdict.reasons == ("external_init",)
    assert [(node["depth"], node["kind"]) for node in verdict.chain] == [
        (0, "checkpoint"),
        (1, "checkpoint"),
        (2, "uvdoc"),
    ]
    assert verdict.chain[1]["sha256"] == UVDOC_199K_SHA and verdict.chain[1]["track"] == "release"
    assert verdict.initialization == initialization({"kind": "checkpoint", **GRID89})


def test_release_finetune_of_a_scratch_run_needs_the_parent_data_spec():
    child = _sidecar(
        "c" * 64,
        SPEC_199K,
        SPEC_199K_SHA,
        [_parent(SCRATCH_409K, "/workspace/init/scratch-409k.pt"), _train(SPEC_199K_SHA)],
        step=1000,
    )
    verdict = shippable(child)
    assert verdict.reasons == ("unverified_data_spec",)
    assert verdict.chain[1]["training"][0] == {
        "kind": "train",
        "track": "release",
        "data_spec_sha256": SPEC_409K_SHA,
        "data": "unverified",
    }
    for parent_spec in (SCRATCH_409K, SCRATCH_409K_INFO, SPEC_409K, DataSpec.from_dict(SPEC_409K)):
        verdict = shippable(child, specs=[{"unrelated": 1}, SPEC_RESEARCH, parent_spec])
        assert verdict.shippable, parent_spec
        assert verdict.initialization == "scratch"
        assert [node["kind"] for node in verdict.chain] == ["checkpoint", "checkpoint", "scratch"]


def test_a_parent_start_may_carry_its_own_data_spec():
    carried = {**_parent(SCRATCH_409K, "/workspace/init/scratch-409k.pt"), "data_spec": SPEC_409K}
    child = _sidecar(
        "c" * 64, SPEC_199K, SPEC_199K_SHA, [carried, _train(SPEC_199K_SHA)], step=1000
    )
    assert shippable(child).shippable
    wrong = {**carried, "data_spec": SPEC_199K}  # hashes to another step's spec
    child = _sidecar("c" * 64, SPEC_199K, SPEC_199K_SHA, [wrong, _train(SPEC_199K_SHA)], step=1000)
    assert shippable(child).reasons == ("unverified_data_spec",)


GATE_DATA = [
    {"kind": "authored", "license": "Apache-2.0 AND OFL-1.1"},
    {"kind": "simulated", "license": "CC-BY-4.0"},
    {"kind": "capture", "license": "CC-BY-4.0"},
]
GATE_V0 = {"track": "release", "lineage": [{"kind": "scratch"}], "data": GATE_DATA}


@pytest.mark.parametrize(
    "edit, reasons",
    [
        ({}, ()),
        ({"track": "research"}, ("not_release_track",)),
        ({"data": None}, ("unverified_data_spec",)),
        (
            {"data": [*GATE_DATA, {"kind": "doc3d", "license": "CC BY-NC-ND 3.0"}]},
            ("data_license_not_release",),
        ),
        ({"data": [{**GATE_DATA[0], "track": "research"}]}, ("data_license_not_release",)),
        ({"data_spec": SPEC_RESEARCH}, ("research_data",)),
        ({"data_spec": SPEC_409K, "data": None}, ()),
        (
            {"lineage": [{"kind": "scratch"}, {"kind": "train", "track": "research"}]},
            ("not_release_track",),
        ),
        ({"lineage": [UVDOC_START, {"kind": "train", "track": "release"}]}, ("external_init",)),
        ({"lineage": None}, ("unknown_init",)),
        ({"lineage": [{"kind": "scratch"}, {"kind": "distill"}]}, ("unknown_lineage_entry",)),
    ],
)
def test_page_gates_go_through_the_same_rules(edit, reasons):
    record = {key: value for key, value in {**GATE_V0, **edit}.items() if value is not None}
    verdict = gate_verdict(record, sha256="g" * 64)
    assert verdict.reasons == reasons and verdict.shippable == (not reasons)
    if not reasons:
        assert verdict.category == "release" and verdict.initialization == "scratch"


def test_a_fine_tuned_page_gate_needs_its_parents_data():
    parent = {"kind": "checkpoint", "track": "release", "sha256": "p" * 64, **GATE_V0}
    child = {
        "track": "release",
        "lineage": [parent, {"kind": "train", "track": "release"}],
        "data": GATE_DATA,
    }
    verdict = gate_verdict(child)
    assert verdict.shippable and verdict.initialization == "scratch"
    assert [node["kind"] for node in verdict.chain] == ["page_gate", "checkpoint", "scratch"]
    bare = {key: value for key, value in parent.items() if key != "data"}
    child["lineage"] = [bare, {"kind": "train", "track": "release"}]
    assert gate_verdict(child).reasons == ("unverified_data_spec",)
    child["lineage"] = [{**parent, "track": "research"}, {"kind": "train", "track": "release"}]
    assert gate_verdict(child).reasons == ("not_release_track",)
    assert gate_verdict(None).category == "external"


def test_research_parents_fail_a_release_child():
    child = _sidecar(
        "d" * 64,
        SPEC_199K,
        SPEC_199K_SHA,
        [_parent(RECIPE_REPRO, "/recipe-repro.pt"), _train(SPEC_199K_SHA)],
        step=10,
    )
    verdict = shippable(child, specs=[RECIPE_REPRO])
    assert verdict.reasons == ("not_release_track", "research_data")
    assert verdict.category == "research" and verdict.initialization == "scratch"


def test_research_track_and_data_fail():
    verdict = shippable(RECIPE_REPRO)
    assert verdict.category == "research" and verdict.track == "research"
    assert set(verdict.reasons) == {
        "not_release_track",
        "research_data",
        "data_license_not_release",
    }


@pytest.mark.parametrize(
    ("record", "kind"),
    [
        (
            {
                "kind": "uvdoc",
                "track": "external",
                "license": "MIT",
                "sha256": UVDOC_WEIGHTS_SHA256,
            },
            "uvdoc",
        ),
        (
            {"sha256": "557af82579157c000f3cbd27c79d02b2a7cfea177849dbb04b9499ff88c81b1f"},
            "syntheticdoc",
        ),
        ({"provenance": "external_pretrained"}, "external"),
    ],
)
def test_outside_weights_are_external(record, kind):
    verdict = shippable(record)
    assert not verdict.shippable and verdict.research
    assert verdict.category == "external" and verdict.reasons == ("external_weights",)
    assert verdict.chain[0]["kind"] == kind and verdict.initialization is None
    assert set(EXTERNAL_WEIGHTS_SHA256.values()) == {"uvdoc", "syntheticdoc"}


def test_records_without_lineage():
    for record in ({"sha256": "f" * 64}, {"kind": ["uvdoc"], "provenance": ["own_checkpoint"]}):
        verdict = shippable(record)
        assert verdict.category == "external" and verdict.reasons == ("no_lineage_record",)
    for record in (
        {"kind": "checkpoint", "sha256": "f" * 64},
        {"provenance": "own_previous_model"},
    ):
        verdict = shippable(record)
        assert verdict.category == "research" and verdict.reasons == ("no_lineage_record",)
    assert set(CATEGORIES) == {"release", "research", "external"}
    with pytest.raises(FlattenerError) as caught:
        shippable([SCRATCH_409K])
    assert caught.value.status is Status.INPUT_ERROR


def test_dewarper_info_needs_the_spec_through_specs():
    info = {
        "kind": "checkpoint",
        "step": SCRATCH_409K["step"],
        "track": "release",
        "lineage": SCRATCH_409K["lineage"],
        "data_spec_sha256": SPEC_409K_SHA,
        "code_sha256": CODE_SCRATCH_409K,
        "sha256": SCRATCH_409K_SHA,
    }
    verdict = shippable(info)
    assert verdict.reasons == ("unverified_data_spec",)
    assert verdict.initialization == initialization(info) == "scratch"
    assert shippable(info, specs=[SPEC_409K]).shippable
    uvdoc_info = {**info, "lineage": UVDOC_199K["lineage"], "data_spec_sha256": SPEC_199K_SHA}
    verdict = shippable(uvdoc_info, specs=[SPEC_199K])
    assert verdict.reasons == ("external_init",)
    assert verdict.initialization == initialization(uvdoc_info) == "external_pretrained"


def _edited(record, edit):
    record = copy.deepcopy(record)
    edit(record)
    return record


def _research_source(record):
    record["data_spec"]["sources"].append(
        _source("uvdoc-train", "/data/uvdoc", kind="uvdoc", track="research")
    )


def _nc_license(record):
    record["data_sources"]["train"][0]["license_classes"] = {"CC BY-NC 4.0": 10}


@pytest.mark.parametrize(
    ("edit", "reasons"),
    [
        (
            lambda r: r["data_spec"].update(name="renamed"),
            {"data_spec_mismatch", "unverified_data_spec"},
        ),
        (_research_source, {"research_data", "unverified_data_spec"}),
        (lambda r: r["lineage"][1].update(track="research"), {"not_release_track"}),
        (_nc_license, {"data_license_not_release"}),
        (
            lambda r: r["lineage"][1].update(data_spec_sha256=SPEC_199K_SHA),
            {"lineage_mismatch", "unverified_data_spec"},
        ),
        (
            lambda r: r["lineage"][1].update(kind="distill"),
            {"unknown_lineage_entry", "lineage_mismatch"},
        ),
        (lambda r: r["lineage"].clear(), {"unknown_init"}),
    ],
)
def test_edited_records_fail_closed(edit, reasons):
    verdict = shippable(_edited(SCRATCH_409K, edit))
    assert not verdict.shippable and verdict.category == "research"
    assert set(verdict.reasons) == reasons
    assert set(verdict.reasons) <= set(REASONS)


def _weights(path, content, sidecar=None):
    path.write_bytes(content)
    if sidecar is not None:
        digest = hashlib.sha256(content).hexdigest()
        record = {**sidecar, "sha256": digest, "file": path.name}
        path.with_name(path.name + ".json").write_text(json.dumps(record), encoding="utf-8")
    return path


def test_gate_file_reads_the_sidecar(tmp_path):
    release = _weights(tmp_path / "scratch-409k.pt", b"release weights", SCRATCH_409K)
    verdict = gate_file(release)
    assert verdict.shippable and verdict.sha256 == hashlib.sha256(b"release weights").hexdigest()
    uvdoc_init = _weights(tmp_path / "uvdoc-199k.pt", b"uvdoc-init weights", UVDOC_199K)
    assert gate_file(uvdoc_init).reasons == ("external_init",)
    bare = _weights(tmp_path / "best_model.pkl", b"outside weights")
    assert gate_file(bare).category == "external"
    assert gate_file(bare).reasons == ("no_lineage_record",)
    release.write_bytes(b"other bytes")
    verdict = gate_file(release)
    assert verdict.reasons == ("sidecar_mismatch",) and verdict.category == "research"
    with pytest.raises(FlattenerError) as caught:
        gate_file(tmp_path / "missing.pt")
    assert caught.value.status is Status.IO_ERROR


def test_gate_file_finds_parent_sidecars_at_their_recorded_path(tmp_path):
    parent = _weights(tmp_path / "parent.pt", b"parent weights", SCRATCH_409K)
    parent_record = json.loads((tmp_path / "parent.pt.json").read_text(encoding="utf-8"))
    lineage = [_parent(parent_record, str(parent)), _train(SPEC_199K_SHA)]
    child = _weights(
        tmp_path / "child.pt",
        b"fine-tuned",
        _sidecar("", SPEC_199K, SPEC_199K_SHA, lineage, step=9),
    )
    assert gate_file(child).shippable
    (tmp_path / "parent.pt.json").unlink()
    assert gate_file(child).reasons == ("unverified_data_spec",)
    assert gate_file(child, specs=[SPEC_409K]).shippable


TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)


def _checkpoint(path, sidecar):
    torch.manual_seed(0)
    fields = ("track", "lineage", "data_spec", "data_spec_sha256", "data_sources")
    save_checkpoint(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": 3,
            "model_config": TINY.to_dict(),
            "code": {"sha256": "1" * 64},
            "model_state": UVDocNet(TINY).state_dict(),
            **{key: sidecar[key] for key in fields},
        },
    )
    return path


def test_load_weights_refuses_research_weights_without_the_flag(tmp_path):
    release = _checkpoint(tmp_path / "release.pt", SCRATCH_409K)
    dewarper = load_weights(release)
    gate = dewarper.info["ship_gate"]
    assert gate["shippable"] is True and gate["research"] is False
    assert gate["sha256"] == dewarper.info["sha256"] and dewarper.config == TINY

    uvdoc_init = _checkpoint(tmp_path / "uvdoc-init.pt", UVDOC_199K)
    with pytest.raises(FlattenerError) as caught:
        load_weights(uvdoc_init)
    error = caught.value
    assert error.status is Status.MODEL_ERROR and error.reason_code == "not_shippable"
    assert error.metadata["ship_gate"]["reasons"] == ["external_init"]
    dewarper = load_weights(uvdoc_init, research=True)
    gate = dewarper.info["ship_gate"]
    assert gate["research"] is True and gate["category"] == "research"
    assert gate["initialization"] == "external_pretrained"


def test_cli_prints_verdicts_and_exit_status(tmp_path, capsys):
    paths = {}
    for name, record in (
        ("release", SCRATCH_409K),
        ("uvdoc-init", UVDOC_199K),
        ("release-info", SCRATCH_409K_INFO),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(record), encoding="utf-8")
    assert main([str(paths["release"]), str(paths["release-info"])]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["shippable"] for line in lines] == [True, True]
    assert main([str(paths["release"]), str(paths["uvdoc-init"])]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["category"] for line in lines] == ["release", "research"]
    assert lines[1]["path"] == str(paths["uvdoc-init"])

    weights = _weights(
        tmp_path / "ft.pt",
        b"ft",
        _sidecar(
            "",
            SPEC_199K,
            SPEC_199K_SHA,
            [_parent(SCRATCH_409K, "/nowhere/parent.pt"), _train(SPEC_199K_SHA)],
            step=5,
        ),
    )
    assert main([str(weights)]) == 1
    assert main([str(weights), "--spec", str(paths["release"])]) == 0
    capsys.readouterr()
    assert main([str(tmp_path / "missing.json")]) == 2
    assert json.loads(capsys.readouterr().err)["reason_code"] == "unreadable_record"
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    assert main([str(tmp_path / "list.json")]) == 2
