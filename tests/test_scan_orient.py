"""OrientNet-1 runtime: network, views, decision, calibration, ship gate and export."""

import json

import numpy as np
import pytest
import torch

from flattener import FlattenerError, Status
from flattener.dewarp.data import DataSpec
from flattener.evidence import evidence, initialization
from flattener.scan import orient
from flattener.scan.orient import (
    CHECKPOINT_SCHEMA,
    OrientConfig,
    Orienter,
    OrientNet,
    SourceSampler,
    ViewGeometry,
    choose_threshold,
    decide,
    prepare_views,
    quantize,
    release_problems,
    save_checkpoint,
)


def _textured_page(height=300, width=210, seed=0):
    """Dark 'text lines' in the upper part, a blank lower part and a corner mark."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width), 235, np.uint8)
    for y in range(20, height // 2, 9):
        length = int(rng.integers(width // 2, width - 30))
        page[y : y + 3, 15 : 15 + length] = rng.integers(0, 60, size=(3, length))
    page[height - 40 : height - 20, width - 40 : width - 20] = 20
    return page


def _data_spec(track="release", kind="converted"):
    """A dewarp-schema spec like the one ``tools/orient/build_data.py`` records."""
    source = {
        "name": "own/train",
        "kind": kind,
        "root": "files/own",
        "track": track,
        "split": "train",
        "license": "Apache-2.0",
    }
    return {
        "schema": "flattener-dewarp-data-1",
        "name": "orient",
        "track": track,
        "sources": [source],
    }


SPEC_SHA256 = DataSpec.from_dict(_data_spec()).sha256


def _sources(classes=None, track="release"):
    source = {"name": "own", "kind": "documents", "track": track, "license": "Apache-2.0"}
    return {"train": [{**source, "samples": 3, "license_classes": classes or {"own": 3}}]}


def _lineage(start=None, train_track="release", digest=SPEC_SHA256):
    start = start or {"kind": "scratch", "seed": 0}
    return [start, {"kind": "train", "track": train_track, "data_spec_sha256": digest}]


def _record(**changes):
    record = {
        "track": "release",
        "lineage": _lineage(),
        "data_spec": _data_spec(),
        "data_spec_sha256": SPEC_SHA256,
        "data_sources": _sources(),
    }
    return {**record, **changes}


def _payload(model, *, calibration=None, **changes):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "step": 12,
        "model_config": model.config.to_dict(),
        "code": orient.code_record(),
        "calibration": calibration,
        "model_state": model.state_dict(),
        **_record(**changes),
    }


def test_forward_pools_global_and_patch_views():
    torch.manual_seed(0)
    model = OrientNet().eval()
    global_view, patches = orient.example_inputs(model.config, 2)
    global_view = global_view + torch.rand_like(global_view) * 50
    with torch.no_grad():
        both, alone, patch_only = model.forward_views(global_view, patches)
        assert both.shape == alone.shape == patch_only.shape == (2, 4)
        assert torch.equal(model(global_view), alone)
        assert torch.allclose(model(global_view, patches), both)


@pytest.mark.parametrize("quarter", [1, 3])
def test_turned_geometry_is_a_clockwise_quarter_turn(quarter):
    image = np.random.default_rng(quarter).integers(0, 256, size=(7, 5), dtype=np.uint8)
    geometry = ViewGeometry.turned((2.5, 3.5), (7, 5), quarter)
    height, width = (int(v) for v in geometry.size)
    points = np.stack(np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5), -1)
    view = quantize(SourceSampler(image, geometry)(points, 1.0))
    # Content turned clockwise by k quarter turns; rot90(view, k) restores it.
    np.testing.assert_array_equal(view, np.rot90(image, -quarter))
    np.testing.assert_array_equal(np.rot90(view, quarter), image)


def test_patch_selection_turns_with_the_page():
    config = OrientConfig()
    page = _textured_page()
    upright = prepare_views(page, config)
    turned = prepare_views(np.ascontiguousarray(np.rot90(page, -1)), config)
    grid = config.energy_grid
    moved = {(int(c), grid - 1 - int(r)) for r, c in upright.cells}
    assert moved == {(int(r), int(c)) for r, c in turned.cells}
    difference = np.abs(np.rot90(upright.global_view, -1).astype(int) - turned.global_view)
    assert difference.max() <= 1
    assert upright.patches.shape == (config.patches, config.patch_size, config.patch_size)
    # The selected cells hold the text lines, never the outer ring.
    assert all(1 <= r <= grid - 2 and 1 <= c <= grid - 2 for r, c in upright.cells)
    assert all(r < grid // 2 + 1 for r, _ in upright.cells)


def test_view_geometry_in_a_photo_matches_the_cut_crop():
    photo = np.random.default_rng(3).integers(0, 256, size=(400, 300), dtype=np.uint8)
    photo[100:340, 60:230] = _textured_page(240, 170)
    geometry = ViewGeometry.turned((145.0, 220.0), (240, 170))
    sampled = prepare_views(photo, geometry=geometry)
    cut = prepare_views(photo[100:340, 60:230])
    # Away from the crop edge (where the photo continues and the cut crop repeats its
    # border) the two agree up to float rounding.
    x0, y0, x1, y1 = (round(v) for v in cut.content)
    inner = (slice(y0 + 2, y1 - 2), slice(x0 + 2, x1 - 2))
    difference = sampled.global_view[inner].astype(int) - cut.global_view[inner]
    assert np.abs(difference).max() <= 1
    np.testing.assert_array_equal(sampled.cells, cut.cells)
    assert np.abs(sampled.patches.astype(int) - cut.patches).max() <= 1


def test_decide_accepts_only_above_the_threshold():
    sure = decide(np.array([0.0, 6.0, 0.0, 0.0]), 1.0, 0.9)
    assert sure.k == 1 and sure.accepted and sure.reason_code is None
    assert sum(sure.probabilities) == pytest.approx(1.0)
    unsure = decide(np.array([1.0, 1.2, 0.0, 0.0]), 1.0, 0.9)
    assert not unsure.accepted and unsure.reason_code == orient.UNCERTAIN
    assert unsure.margin == pytest.approx(unsure.probabilities[1] - unsure.probabilities[0])
    # A higher temperature flattens the same logits below the threshold.
    assert not decide(np.array([0.0, 6.0, 0.0, 0.0]), 4.0, 0.9).accepted


def test_choose_threshold_keeps_silent_errors_within_the_bound():
    rng = np.random.default_rng(1)
    count = 2000
    labels = rng.integers(0, 4, size=count)
    confidence = rng.uniform(0.3, 1.0, size=count)
    wrong = rng.random(count) < (1.0 - confidence) * 0.2
    predicted = np.where(wrong, (labels + 1) % 4, labels)
    probabilities = np.full((count, 4), 0.0)
    probabilities[np.arange(count), predicted] = confidence
    probabilities[np.arange(count), (predicted + 2) % 4] = 1.0 - confidence
    point = choose_threshold(probabilities, labels, 0.005)
    accepted = confidence >= point["threshold"]
    assert point["feasible"] and (accepted & wrong).sum() / count <= 0.005
    assert point["silent_error"] == pytest.approx((accepted & wrong).sum() / count)
    lower = confidence[confidence < point["threshold"]]
    if len(lower) and point["threshold"] > 0.5:  # one step lower breaks the bound
        widened = confidence >= lower.max()
        assert (widened & wrong).sum() / count > 0.005
    none = choose_threshold(np.eye(4)[[1, 2]], np.array([0, 0]))
    assert not none["feasible"] and none["threshold"] == 1.0


def test_release_problems_follow_the_whole_lineage():
    assert release_problems(_record()) == []
    assert release_problems(_record(track="research"))
    uvdoc = {"kind": "uvdoc", "sha256_matches_upstream": True}
    assert "uvdoc" in " ".join(release_problems(_record(lineage=_lineage(uvdoc))))
    assert release_problems(_record(lineage=_lineage(train_track="research")))
    parent_ok = {"kind": "checkpoint", "track": "release", "lineage": _lineage()}
    assert release_problems(_record(lineage=_lineage(parent_ok))) == []
    parent_uvdoc = {"kind": "checkpoint", "track": "release", "lineage": _lineage(uvdoc)}
    assert release_problems(_record(lineage=_lineage(parent_uvdoc)))
    parent_research = {**parent_ok, "track": "research"}
    assert release_problems(_record(lineage=_lineage(parent_research)))
    assert release_problems(_record(lineage=None))
    research = _data_spec(track="research", kind="uvdoc")
    digest = DataSpec.from_dict(research).sha256
    assert release_problems(
        _record(data_spec=research, data_spec_sha256=digest, lineage=_lineage(digest=digest))
    )
    assert release_problems(_record(data_spec={"schema": "flattener-orient-data-1"}))
    assert release_problems(_record(data_spec_sha256="0" * 64))
    assert release_problems(_record(lineage=_lineage(digest="0" * 64)))
    assert release_problems(_record(data_sources=_sources({"CC BY-NC 4.0": 1})))
    assert release_problems(_record(data_sources=_sources(track="research")))
    assert release_problems(_record(data_sources=None))


def test_a_fine_tuned_model_needs_every_parent_data_spec(tmp_path):
    parent_spec = {**_data_spec(), "name": "orient-parent"}
    digest = DataSpec.from_dict(parent_spec).sha256
    parent = {"kind": "checkpoint", "track": "release", "lineage": _lineage(digest=digest)}
    child = _record(lineage=_lineage(parent))
    problems = release_problems(child)
    assert problems == ["ship gate: unverified_data_spec"]
    shippable = pytest.importorskip("flattener.scan.shippable")
    assert shippable.shippable(child).reasons == ("unverified_data_spec",)  # same verdict
    carried = _record(lineage=_lineage({**parent, "data_spec": parent_spec}))
    assert release_problems(carried) == [] and shippable.shippable(carried).shippable
    assert release_problems(child, specs=[parent_spec]) == []
    sidecar = tmp_path / "parent.pt.json"  # the parent's sidecar at its recorded path
    sidecar.write_text(json.dumps({"data_spec": parent_spec}), encoding="utf-8")
    found = _record(lineage=_lineage({**parent, "path": str(tmp_path / "parent.pt")}))
    assert release_problems(found) == []


def test_checkpoint_loads_only_inside_the_release_lineage(tmp_path):
    torch.manual_seed(0)
    model = OrientNet()
    calibration = {"temperature": 1.7, "threshold": 0.8, "source": "kr_real_dev"}
    path = tmp_path / "release.pt"
    sidecar = save_checkpoint(path, _payload(model, calibration=calibration))
    synthetic = tmp_path / "synthetic.pt"
    save_checkpoint(
        synthetic, _payload(model, calibration={**calibration, "source": "synthetic_val"})
    )
    assert Orienter.load(synthetic).threshold == orient.DEFAULT_THRESHOLD
    given = Orienter.load(synthetic, threshold=0.0)  # a threshold measured on real photos
    assert given.threshold == 0.0 and given.info["threshold_given"]
    assert sidecar["schema"] == orient.SIDECAR_SCHEMA and "model_state" not in sidecar
    assert json.loads((tmp_path / "release.pt.json").read_text())["track"] == "release"
    orienter = Orienter.load(path)
    assert orienter.info["shippable"] and not orienter.info["research"]
    assert (orienter.temperature, orienter.threshold) == (1.7, 0.8)
    assert initialization(orienter.info) == "scratch"
    assert evidence("gt_crop_validation", orienter.info)["model_provenance"] == "own_checkpoint"
    result = orienter.predict(_textured_page())
    assert 0 <= result.k < 4 and result.temperature == 1.7

    research = tmp_path / "research.pt"
    save_checkpoint(research, _payload(model, track="research"))
    with pytest.raises(FlattenerError) as caught:
        Orienter.load(research)
    assert caught.value.status == Status.MODEL_ERROR
    assert caught.value.reason_code == "not_shippable"
    loaded = Orienter.load(research, research=True)
    assert loaded.info["research"] and not loaded.info["shippable"]

    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(FlattenerError) as caught:
        Orienter.load(path)
    assert caught.value.reason_code == "checksum_mismatch"


def test_cli_predict_and_cost(tmp_path, capsys):
    from PIL import Image

    torch.manual_seed(0)
    save_checkpoint(tmp_path / "release.pt", _payload(OrientNet()))
    save_checkpoint(tmp_path / "research.pt", _payload(OrientNet(), track="research"))
    Image.fromarray(_textured_page()).convert("RGB").save(tmp_path / "page.png")
    missing = str(tmp_path / "missing.png")
    code = orient.main(
        [
            "predict",
            str(tmp_path / "page.png"),
            missing,
            "--checkpoint",
            str(tmp_path / "release.pt"),
        ]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 1 and lines[0]["status"] == "success" and lines[1]["status"] == "input_error"
    code = orient.main(["predict", missing, "--checkpoint", str(tmp_path / "research.pt")])
    assert code == 2 and json.loads(capsys.readouterr().out)["reason_code"] == "not_shippable"
    assert orient.main(["cost"]) == 0
    assert json.loads(capsys.readouterr().out)["parameters"] > 0
