"""regions.context: the ported context U-Net, its checkpoints and the page gate decision."""

import hashlib
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from flattener import FlattenerError, Status
from flattener.regions import context
from flattener.regions.context import (
    CHECKPOINT_SCHEMA,
    MODEL_VERSION,
    PILOT_SCHEMA,
    ContextPageSegmenter,
    ContextSegmenterConfig,
    GateConfig,
    gate_page,
    gate_page_region,
    load_context_segmenter,
    parameter_count,
    preprocess_image,
    preprocess_mask,
    save_context_checkpoint,
    smartdoc_exposure,
    touched_borders,
)
from flattener.regions.geometry import PageRegion
from flattener.regions.segmentation import SegmenterConfig
from flattener.regions.segmentation import preprocess_image as pinned_preprocess_image
from flattener.regions.segmentation import preprocess_mask as pinned_preprocess_mask

SMALL = ContextSegmenterConfig(input_size=(17, 13), channels=(2, 3, 4, 5))


def _model(config=SMALL, seed=5):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ContextPageSegmenter(config).eval()


def _pilot(path, model, config=SMALL, **changes):
    payload = {
        "schema": PILOT_SCHEMA,
        "model_config": asdict(config),
        "epoch": 3,
        "best_epoch": 2,
        "best_score": 0.5,
        "model_state": model.state_dict(),
        "protocol": {"seed": 1, "epochs": 3, "device": {"device": "xpu:0"}},
    }
    payload.update(changes)
    torch.save(payload, path)
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )
    return path


def test_v0_architecture_has_the_recorded_parameter_count():
    config = ContextSegmenterConfig(input_size=(192, 144))
    model = ContextPageSegmenter(config)
    assert parameter_count(config) == sum(p.numel() for p in model.parameters()) == 140_545
    assert asdict(config)["model_version"] == MODEL_VERSION == "region-context-unet-2.0"
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())


@pytest.mark.parametrize(
    "changes",
    [
        {"input_size": (7, 96)},
        {"input_size": [128, 96]},
        {"channels": (8, 16, 32)},
        {"channels": (1, 2, 3, True)},
        {"channels": (128, 128, 128, 128)},
        {"model_version": "region-unet-1.0"},
    ],
)
def test_config_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        ContextSegmenterConfig(**changes)


def test_preprocessing_is_the_pinned_segmentation_preprocessing():
    image = np.random.default_rng(3).integers(0, 256, (60, 80, 3), dtype=np.uint8)[::-1]
    mask = image[:, :, 0] > 128
    before = image.copy()
    pinned = SegmenterConfig(input_size=SMALL.input_size)
    assert torch.equal(preprocess_image(image, SMALL), pinned_preprocess_image(image, pinned))
    assert torch.equal(preprocess_mask(mask, SMALL), pinned_preprocess_mask(mask, pinned))
    np.testing.assert_array_equal(image, before)


def test_forward_follows_the_archived_operation_order_and_keeps_rng():
    model = _model()
    image = torch.rand(2, 3, *SMALL.input_size)
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        first = model.encoder1(image)
        second = model.encoder2(F.avg_pool2d(first, 2))
        third = model.encoder3(F.avg_pool2d(second, 2))
        bottom = model.bottom(F.avg_pool2d(third, 2))
        up = F.interpolate(bottom, size=third.shape[-2:], mode="bilinear", align_corners=False)
        scene = model.scene_pool(bottom).expand(-1, -1, *third.shape[-2:])
        up = model.decoder3(torch.cat((up, third, scene), dim=1))
        up = F.interpolate(up, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up = model.decoder2(torch.cat((up, second), dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        expected = model.head(model.decoder1(torch.cat((up, first), dim=1)))
        assert torch.equal(model(image), expected)
        assert torch.equal(model._forward_trusted(image), expected)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.zeros(1, 3, 17, 13, dtype=torch.float64),
        torch.zeros(1, 3, 16, 13),
        torch.full((1, 3, 17, 13), float("nan")),
        torch.full((1, 3, 17, 13), 1.01),
    ],
)
def test_forward_rejects_invalid_input(invalid):
    with pytest.raises((TypeError, ValueError)):
        _model()(invalid)


def test_region_context_checkpoint_round_trip(tmp_path):
    model = _model()
    provenance = {"name": "test", "initialization": "scratch", "lineage": [{"kind": "scratch"}]}
    path = tmp_path / "gate.pt"
    digest = save_context_checkpoint(path, model.state_dict(), SMALL, provenance)
    assert (tmp_path / "gate.pt.sha256").read_text() == digest + "\n"
    rng = torch.get_rng_state().clone()
    loaded = load_context_segmenter(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded.config == SMALL
    assert loaded.model_sha256 == digest
    assert loaded.model_id == f"{MODEL_VERSION}:{digest[:12]}"
    assert loaded.metadata["checkpoint_schema"] == CHECKPOINT_SCHEMA
    assert loaded.metadata["provenance"] == provenance
    image = np.random.default_rng(1).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    probability = loaded.predict_probability(image)
    assert probability.shape == SMALL.input_size and probability.dtype == np.float32
    with torch.inference_mode():
        expected = model(preprocess_image(image, SMALL)[None]).sigmoid()[0, 0].numpy()
    np.testing.assert_array_equal(probability, expected)
    logits = torch.from_numpy(loaded.predict_logits(image))
    np.testing.assert_array_equal(logits.sigmoid().numpy(), probability)
    assert np.all((probability >= 0) & (probability <= 1))


def test_pilot_checkpoint_loads_with_an_unknown_initialization(tmp_path):
    loaded = load_context_segmenter(_pilot(tmp_path / "best.pt", _model()))
    assert loaded.metadata["checkpoint_schema"] == PILOT_SCHEMA
    assert (loaded.metadata["epoch"], loaded.metadata["best_epoch"]) == (3, 2)
    assert loaded.metadata["provenance"]["initialization"] == "unknown"
    assert loaded.metadata["provenance"]["device"] == {"device": "xpu:0"}


def _reason(call):
    with pytest.raises(FlattenerError) as caught:
        call()
    assert caught.value.status == Status.MODEL_ERROR
    return caught.value.reason_code


def test_checkpoint_needs_a_matching_sidecar(tmp_path):
    path = _pilot(tmp_path / "best.pt", _model())
    sidecar = tmp_path / "best.pt.sha256"
    sidecar.write_text("0" * 64 + "\n")
    assert _reason(lambda: load_context_segmenter(path)) == "checkpoint_hash_mismatch"
    sidecar.unlink()
    assert _reason(lambda: load_context_segmenter(path)) == "missing_sidecar"
    assert _reason(lambda: load_context_segmenter(tmp_path / "none.pt")) == "missing_checkpoint"


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "region-segmentation-0.1"},
        {"model_config": {"input_size": [17, 13], "channels": [2, 3, 4]}},
        {"best_epoch": 4},
        {"model_state": {}},
    ],
)
def test_invalid_checkpoints_are_rejected(tmp_path, changes):
    path = _pilot(tmp_path / "best.pt", _model(), **changes)
    assert _reason(lambda: load_context_segmenter(path)) == "invalid_checkpoint"


def test_mismatched_or_nonfinite_weights_are_rejected(tmp_path):
    other = _model(ContextSegmenterConfig(input_size=(17, 13), channels=(2, 3, 4, 6)))
    path = _pilot(tmp_path / "other.pt", other)
    assert _reason(lambda: load_context_segmenter(path)) == "invalid_checkpoint"
    state = _model().state_dict()
    state["head.bias"] = torch.tensor([float("nan")])
    path = _pilot(tmp_path / "nan.pt", _model(), model_state=state)
    assert _reason(lambda: load_context_segmenter(path)) == "invalid_checkpoint"


def test_save_rejects_bad_provenance_and_state(tmp_path):
    with pytest.raises(ValueError):
        save_context_checkpoint(tmp_path / "a.pt", _model().state_dict(), SMALL, {"x": object()})
    with pytest.raises(RuntimeError):
        save_context_checkpoint(tmp_path / "b.pt", {}, SMALL, {})
    assert not list(tmp_path.iterdir())


def test_load_rejects_an_unknown_device(tmp_path):
    path = _pilot(tmp_path / "best.pt", _model())
    with pytest.raises(FlattenerError) as caught:
        load_context_segmenter(path, device="meta")
    assert caught.value.status == Status.INPUT_ERROR


def _probability(box, shape=(96, 72), value=0.98):
    probability = np.full(shape, 0.02, np.float32)
    top, left, bottom, right = box
    probability[top:bottom, left:right] = value
    return probability


def test_centered_page_keeps_the_detected_box():
    gate = gate_page_region(_probability((24, 18, 72, 54)), (960, 720))
    assert not gate.fills_frame and gate.reasons == () and gate.touched_borders == ()
    assert gate.region == gate.estimate.region
    assert gate.region.left < 180 and gate.region.right > 540
    assert gate.confident and gate.area_fraction == pytest.approx(0.25)
    assert gate.metadata["reasons"] == [] and gate.metadata["fills_frame"] is False


@pytest.mark.parametrize(
    "box, borders",
    [
        ((0, 0, 96, 72), ("top", "right", "bottom", "left")),  # the default 0.95 cap rejects it
        ((2, 2, 94, 70), ()),  # 90.5% of the frame, no border
        ((0, 20, 96, 72), ("top", "right", "bottom")),  # 72% touching three borders
    ],
)
def test_page_fills_frame_is_a_success_with_the_whole_image(box, borders):
    gate = gate_page_region(_probability(box), (960, 720))
    assert gate.fills_frame and gate.reasons == ("page_fills_frame",)
    assert gate.touched_borders == borders
    assert gate.region == PageRegion(0, 0, 720, 960)
    assert gate.metadata["region"] == asdict(PageRegion(0, 0, 720, 960))


def test_two_borders_do_not_fill_the_frame():
    gate = gate_page_region(_probability((0, 0, 48, 40)), (960, 720))
    assert not gate.fills_frame and gate.touched_borders == ("top", "left")
    assert gate.region == gate.estimate.region


def test_weak_scores_pass_detection_without_confidence():
    gate = gate_page_region(_probability((24, 18, 72, 54), value=0.8), (960, 720))
    assert not gate.confident


def test_rejections_keep_their_status_and_reason():
    empty = np.zeros((96, 72), np.float32)
    assert _reason(lambda: gate_page_region(empty, (960, 720))) == "page_not_found"
    two = _probability((10, 4, 50, 30))
    two[10:50, 40:66] = 0.98
    assert _reason(lambda: gate_page_region(two, (960, 720))) == "ambiguous_page_region"


def test_gate_page_names_the_segmenter():
    class Stub:
        model_id, model_sha256, load_seconds = "stub:1", "a" * 64, 0.0

        def predict_probability(self, image):
            return _probability((24, 18, 72, 54))

    image = np.zeros((96, 72, 3), np.uint8)
    gate = gate_page(image, Stub())
    assert gate.metadata["detection"]["segmenter_id"] == "stub:1"
    assert gate.region == gate.estimate.region


@pytest.mark.parametrize(
    "changes",
    [
        {"fills_frame_area": 1.5},
        {"fills_frame_area": True},
        {"fills_frame_borders": 0},
        {"fills_frame_borders": 2.0},
        {"confident_dominance": float("nan")},
        {"region": "default"},
    ],
)
def test_gate_config_rejects_invalid_values(changes):
    with pytest.raises((TypeError, ValueError)):
        GateConfig(**changes)


def test_touched_borders_checks_its_input():
    with pytest.raises(FlattenerError):
        touched_borders(np.zeros((4, 4), np.float32))
    mask = np.zeros((4, 5), bool)
    mask[1:, 4] = True
    assert touched_borders(mask) == ("right", "bottom")


def test_smartdoc_exposure_lists_every_template_once():
    record = smartdoc_exposure()
    groups = ("exclude_from_evaluation", "selection_exposed", "unexposed")
    names = [name for group in groups for name in record[group]]
    assert len(names) == len(set(names)) == len(record["templates"]) == 30
    assert len(record["exclude_from_evaluation"]) == 22
    assert record["selection_exposed"] == ["datasheet004", "letter004", "paper004", "patent004"]
    assert record["unexposed"] == ["datasheet005", "letter005", "paper005", "patent005"]
    assert all(
        f"{kind}00{i}" in record["exclude_from_evaluation"]
        for kind in ("magazine", "tax")
        for i in range(1, 6)
    )
    level = {"train": groups[0], "selection": groups[1], "none": groups[2]}
    for template in record["templates"]:
        assert template["model_name"] in record[level[template["exposure"]]]
        assert bool(template["frames_used"]) == (template["exposure"] != "none")
    assert record["model"]["checkpoint_sha256"].startswith("adca808d")
    record["unexposed"].clear()
    assert smartdoc_exposure()["unexposed"]


def test_module_imports_only_preprocessing_from_the_pinned_segmentation_module():
    source = context.__file__
    with open(source, encoding="utf-8") as stream:
        lines = [line for line in stream if ".segmentation import" in line]
    imported = {
        name.strip(" ()\n").split(" as ")[0]
        for line in lines
        for name in line.split("import", 1)[1].split(",")
    }
    assert imported == {
        "PREPROCESS_VERSION",
        "SegmenterConfig",
        "preprocess_image",
        "preprocess_mask",
    }
