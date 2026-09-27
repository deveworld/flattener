"""OrientNet-1 runtime: network, views, decision, calibration, ship gate and export."""

import hashlib
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
    fit_temperature,
    orientation_metrics,
    prepare_views,
    quantize,
    release_problems,
    save_checkpoint,
)

LICENSE = "SyntheticDoc CC BY 4.0"


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


def test_model_size_and_cost_match_the_design_budget():
    cost = orient.model_cost()
    assert 400_000 <= cost["parameters"] <= 600_000
    assert 0.25e9 <= cost["flops"] <= 0.5e9
    assert cost["inputs"] == {"global_view": [1, 1, 256, 256], "patches": [1, 4, 1, 128, 128]}


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


@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
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


def test_letterbox_pads_outside_the_page():
    config = OrientConfig()
    views = prepare_views(np.zeros((100, 400), np.uint8), config)
    x0, y0, x1, y1 = views.content
    assert (x0, x1) == (0.0, 256.0) and y0 == pytest.approx(96.0) and y1 == pytest.approx(160.0)
    assert (views.global_view[:90] == config.pad_value).all()
    assert (views.global_view[100:150] == 0).all()


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


def test_rgb_input_uses_luma():
    rgb = np.zeros((10, 10, 3), np.uint8)
    rgb[..., 1] = 200
    assert (orient.to_gray(rgb) == 117).all()  # 200 * 587 / 1000
    with pytest.raises(FlattenerError) as caught:
        orient.to_gray(np.zeros((4, 4, 2), np.uint8))
    assert caught.value.status == Status.INPUT_ERROR


def test_decide_accepts_only_above_the_threshold():
    sure = decide(np.array([0.0, 6.0, 0.0, 0.0]), 1.0, 0.9)
    assert sure.k == 1 and sure.accepted and sure.reason_code is None
    assert sum(sure.probabilities) == pytest.approx(1.0)
    unsure = decide(np.array([1.0, 1.2, 0.0, 0.0]), 1.0, 0.9)
    assert not unsure.accepted and unsure.reason_code == orient.UNCERTAIN
    assert unsure.margin == pytest.approx(unsure.probabilities[1] - unsure.probabilities[0])
    # A higher temperature flattens the same logits below the threshold.
    assert not decide(np.array([0.0, 6.0, 0.0, 0.0]), 4.0, 0.9).accepted


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 2.0, size=(4000, 4))
    probabilities = orient.softmax(base)
    labels = np.array([rng.choice(4, p=p) for p in probabilities])
    assert fit_temperature(base * 3.0, labels) == pytest.approx(3.0, rel=0.1)
    labels_with_text_free = np.concatenate([labels, [-1, -1]])
    logits = np.concatenate([base, np.zeros((2, 4))])
    assert fit_temperature(logits, labels_with_text_free) == pytest.approx(1.0, rel=0.1)


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


def test_orientation_metrics_split_quarter_and_half_errors():
    labels = np.array([0, 1, 2, 3, 0, -1])
    predicted = np.array([0, 2, 0, 3, 3, 1])
    logits = np.eye(4)[predicted] * 5.0
    metrics = orientation_metrics(logits, labels, groups=list("aabbbc"))
    assert metrics["samples"] == 5 and metrics["text_free_samples"] == 1
    assert metrics["accuracy"] == pytest.approx(2 / 5)
    assert metrics["quarter_errors"] == 2 and metrics["half_errors"] == 1
    assert metrics["confusion"][1][2] == 1 and metrics["confusion"][2][0] == 1
    assert metrics["groups"]["a"] == {"samples": 2, "accuracy": 0.5}
    assert "text_free_confidence" in metrics


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


def test_records_pass_the_scanner_ship_gate():
    shippable = pytest.importorskip("flattener.scan.shippable")
    assert shippable.shippable({**_record(), "sha256": "a" * 64}).shippable
    verdict = shippable.shippable({**_record(track="research"), "sha256": "a" * 64})
    assert not verdict.shippable and verdict.category == "research"


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


def test_export_module_traces_and_matches_eager(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "model.pt"
    save_checkpoint(path, _payload(OrientNet(), calibration={"temperature": 2.0}))
    orienter = Orienter.load(path)
    module = orient.exportable(orienter)
    inputs = orient.example_inputs(orienter.config, 2)
    inputs = (inputs[0] + torch.rand_like(inputs[0]) * 80, inputs[1] - torch.rand_like(inputs[1]))
    batch = torch.export.Dim("batch", min=1, max=1024)
    program = torch.export.export(
        module, inputs, dynamic_shapes={"global_view": {0: batch}, "patches": {0: batch}}
    )
    with torch.no_grad():
        logits, probabilities = module(*inputs)
        got_logits, got_probabilities = program.module()(*inputs)
    torch.testing.assert_close(got_logits, logits)
    torch.testing.assert_close(probabilities, torch.softmax(logits / 2.0, dim=1))
    torch.testing.assert_close(got_probabilities, probabilities)


def test_onnx_export_writes_model_and_record(tmp_path):
    pytest.importorskip("onnxscript")
    torch.manual_seed(0)
    path = tmp_path / "model.pt"
    save_checkpoint(path, _payload(OrientNet()))
    record = orient.export_onnx(path, tmp_path / "orientnet.onnx")
    assert (tmp_path / "orientnet.onnx").is_file() and record["shippable"]
    saved = json.loads((tmp_path / "orientnet.onnx.json").read_text())
    assert saved["schema"] == orient.EXPORT_SCHEMA and saved["lineage"][0]["kind"] == "scratch"


def test_export_record_and_ship_gate_with_a_stub_exporter(tmp_path, monkeypatch, capsys):
    calls = []

    class Program:
        def save(self, path, external_data=False):
            path.write_bytes(b"onnx-bytes")

    def fake_export(module, args, **options):
        calls.append(options)
        return Program()

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    torch.manual_seed(0)
    calibration = {"temperature": 1.5, "threshold": 0.7, "source": "synthetic_val"}
    save_checkpoint(tmp_path / "release.pt", _payload(OrientNet(), calibration=calibration))
    record = orient.export_onnx(tmp_path / "release.pt", tmp_path / "out" / "o.onnx")
    assert calls[0]["dynamo"] and calls[0]["input_names"] == ["global_view", "patches"]
    assert set(calls[0]["dynamic_shapes"]) == {"global_view", "patches"}
    saved = json.loads((tmp_path / "out" / "o.onnx.json").read_text())
    assert saved == json.loads(json.dumps(record))
    assert record["sha256"] == hashlib.sha256(b"onnx-bytes").hexdigest()
    assert (record["temperature"], record["threshold"]) == (1.5, orient.DEFAULT_THRESHOLD)
    assert record["shippable"] and not record["research"] and record["parity"] is None
    assert release_problems(record) == []
    try:
        from flattener.scan import shippable
    except ImportError:
        pass
    else:  # the export's JSON record gates the .onnx file like a checkpoint sidecar
        assert shippable.gate_file(tmp_path / "out" / "o.onnx").shippable

    save_checkpoint(tmp_path / "research.pt", _payload(OrientNet(), track="research"))
    with pytest.raises(FlattenerError) as caught:
        orient.export_onnx(tmp_path / "research.pt", tmp_path / "r.onnx")
    assert caught.value.reason_code == "not_shippable"
    record = orient.export_onnx(tmp_path / "research.pt", tmp_path / "r.onnx", research=True)
    assert record["research"] and not record["shippable"]

    def missing(*args, **options):
        raise ModuleNotFoundError("No module named 'onnxscript'")

    monkeypatch.setattr(torch.onnx, "export", missing)
    code = orient.main(
        ["export", "--checkpoint", str(tmp_path / "release.pt"), "--output", str(tmp_path / "m")]
    )
    assert code == 2 and json.loads(capsys.readouterr().out)["reason_code"] == "missing_dependency"


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
