"""Page detector architecture, checkpoints, preprocessing and region decisions."""

from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from flattener import FlattenerError, Status
from flattener.regions.context import (
    CHECKPOINT_SCHEMA,
    MODEL_VERSION,
    ContextPageSegmenter,
    ContextSegmenterConfig,
    gate_page_region,
    load_context_segmenter,
    parameter_count,
    preprocess_image,
    preprocess_mask,
    save_context_checkpoint,
)
from flattener.regions.geometry import PageRegion
from flattener.regions.segmentation import SegmenterConfig
from flattener.regions.segmentation import preprocess_image as pinned_preprocess_image
from flattener.regions.segmentation import preprocess_mask as pinned_preprocess_mask

SMALL = ContextSegmenterConfig(input_size=(17, 13), channels=(2, 3, 4, 5))


def _model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        return ContextPageSegmenter(SMALL).eval()


def _reason(call):
    with pytest.raises(FlattenerError) as caught:
        call()
    assert caught.value.status == Status.MODEL_ERROR
    return caught.value.reason_code


def _probability(box, shape=(96, 72), value=0.98):
    probability = np.full(shape, 0.02, np.float32)
    top, left, bottom, right = box
    probability[top:bottom, left:right] = value
    return probability


def test_default_architecture_and_operation_order():
    config = ContextSegmenterConfig()
    assert config.input_size == (192, 144) and config.channels == (16, 32, 64, 128)
    model = ContextPageSegmenter(config)
    assert parameter_count(config) == sum(p.numel() for p in model.parameters()) == 561_025
    assert asdict(config)["model_version"] == MODEL_VERSION == "region-context-unet-2.0"
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())

    # The forward pass leaves the RNG alone.
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


def test_preprocessing_is_the_pinned_segmentation_preprocessing():
    image = np.random.default_rng(3).integers(0, 256, (60, 80, 3), dtype=np.uint8)[::-1]
    mask = image[:, :, 0] > 128
    before = image.copy()
    pinned = SegmenterConfig(input_size=SMALL.input_size)
    assert torch.equal(preprocess_image(image, SMALL), pinned_preprocess_image(image, pinned))
    assert torch.equal(preprocess_mask(mask, SMALL), pinned_preprocess_mask(mask, pinned))
    np.testing.assert_array_equal(image, before)

    # The whole image is resized in RGB order, without a crop, into a new tensor.
    halves = np.zeros((40, 20, 3), np.uint8)
    halves[:20, :, 0] = 255
    halves[20:, :, 2] = 255
    original = halves.copy()
    output = preprocess_image(halves[::-1], SMALL)
    assert output.shape == (3, *SMALL.input_size)
    assert output.dtype == torch.float32 and output.device.type == "cpu"
    assert output[2, 0].mean() > 0.99 and output[0, -1].mean() > 0.99
    output.fill_(0.7)
    np.testing.assert_array_equal(halves, original)


def test_region_context_checkpoint_round_trip(tmp_path):
    model = _model()
    provenance = {"name": "test", "initialization": "scratch", "lineage": [{"kind": "scratch"}]}
    path = tmp_path / "gate.pt"
    save_context_checkpoint(path, model.state_dict(), SMALL, provenance)
    rng = torch.get_rng_state().clone()
    loaded = load_context_segmenter(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded.config == SMALL
    assert loaded.model_id == str(path.resolve())
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


def test_checkpoint_requires_an_existing_file(tmp_path):
    assert _reason(lambda: load_context_segmenter(tmp_path / "none.pt")) == "missing_checkpoint"


@pytest.mark.parametrize(
    "field, value",
    [("schema", "unsupported-schema"), ("preprocess_version", "unknown"), ("provenance", None)],
)
def test_checkpoint_requires_current_schema_and_metadata(tmp_path, field, value):
    path = tmp_path / "gate.pt"
    save_context_checkpoint(path, _model().state_dict(), SMALL, {"initialization": "scratch"})
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    assert _reason(lambda: load_context_segmenter(path)) == "invalid_checkpoint"


def test_centered_page_keeps_the_detected_box():
    gate = gate_page_region(_probability((24, 18, 72, 54)), (960, 720))
    assert not gate.fills_frame and gate.reasons == () and gate.touched_borders == ()
    assert gate.region == gate.estimate.region
    assert gate.region.left < 180 and gate.region.right > 540
    assert gate.confident and gate.area_fraction == pytest.approx(0.25)
    assert gate.metadata["reasons"] == [] and gate.metadata["fills_frame"] is False
    # Weaker scores still pass detection, without confidence.
    assert not gate_page_region(_probability((24, 18, 72, 54), value=0.8), (960, 720)).confident


def test_page_fills_frame_is_a_success_with_the_whole_image():
    for box, borders in (
        ((0, 0, 96, 72), ("top", "right", "bottom", "left")),  # past the default 0.95 area cap
        ((2, 2, 94, 70), ()),  # 90.5% of the frame, no border
        ((0, 20, 96, 72), ("top", "right", "bottom")),  # 72% touching three borders
    ):
        gate = gate_page_region(_probability(box), (960, 720))
        assert gate.fills_frame and gate.reasons == ("page_fills_frame",), box
        assert gate.touched_borders == borders
        assert gate.region == PageRegion(0, 0, 720, 960)
        assert gate.metadata["region"] == asdict(PageRegion(0, 0, 720, 960))


def test_rejections_keep_their_status_and_reason():
    empty = np.zeros((96, 72), np.float32)
    assert _reason(lambda: gate_page_region(empty, (960, 720))) == "page_not_found"
    speck = _probability((40, 30, 41, 31))
    assert _reason(lambda: gate_page_region(speck, (960, 720))) == "page_area_rejected"
    two = _probability((10, 4, 50, 30))
    two[10:50, 40:66] = 0.98
    assert _reason(lambda: gate_page_region(two, (960, 720))) == "ambiguous_page_region"
    unsure = _probability((24, 18, 72, 54), value=0.6)
    assert _reason(lambda: gate_page_region(unsure, (960, 720))) == "uncertain_page_region"
