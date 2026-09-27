"""DIR300 benchmark: pairing, reference preparation, reports, OCR subset and CLI."""

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from flattener.benchmark import __main__ as cli
from flattener.benchmark import dir300, matlab, metrics
from flattener.benchmark.reporting import PREDICTIONS_SCHEMA
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import load_dewarper

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)


def _page(seed, shape=(40, 30)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*shape, 3), np.uint8)


def _folders(tmp_path, documents=(1, 2, 248)):
    """References (248 stored as 248.PNG like the official download) and predictions
    identical to them."""
    references, predictions = tmp_path / "gt", tmp_path / "pred"
    references.mkdir()
    predictions.mkdir()
    for k in documents:
        page = _page(k)
        name = f"{k}.PNG" if k == 248 else f"{k}.png"
        Image.fromarray(page).save(references / name)
        Image.fromarray(page).save(predictions / f"{k}.png")
    return references, predictions


def test_reference_scale_divides_as_docgeonet():
    # 598400/H/W and 598400/(H*W) differ in the last bit for this size.
    page = _page(0, (23, 27))
    scale = math.sqrt(598400 / 23 / 27)
    assert scale != matlab.metric_scale(23, 27, 598400)
    prepared = dir300.prepare_reference(page)
    assert prepared.image.shape == (
        math.ceil(page.shape[0] * scale),
        math.ceil(page.shape[1] * scale),
    )
    assert prepared.resizes == 1 and prepared.source_shape == page.shape[:2]
    shared = metrics.prepare_reference(page)
    assert np.array_equal(prepared.image, shared.image)
    assert np.allclose(prepared.gradient, shared.gradient)


def test_evaluate_scores_each_pair_once(tmp_path):
    references, predictions = _folders(tmp_path)
    report = dir300.evaluate_dir300(
        predictions,
        build=None,
        references_dir=references,
        verify_references=False,
        documents=(248, 1, 2),
        workers=1,
    )
    assert report["benchmark"] == "dir300" and report["comparable"] is False
    assert [row["id"] for row in report["rows"]] == ["1", "2", "248"]
    assert report["rows"][2]["reference_file"] == "248.PNG"
    assert report["config"]["documents"] == [1, 2, 248]
    assert report["inputs"]["references_set_sha256"] is None
    summary = report["summary"]
    # The prediction is resized to the reference's metric size, not by the same scale.
    assert summary["n"] == 3 and summary["ms_ssim"] == pytest.approx(1.0, abs=0.01)
    assert summary["ld"] is None and summary["prediction_failures"] is None
    assert report["evidence"]["measurement"] == "dir300_standard"
    assert report["evidence"]["model_provenance"] == "unknown"
    assert "(uvdoc 0.622, diff" in dir300.format_comparison(summary)
    json.dumps(report, allow_nan=False)


def test_evaluate_reads_the_prediction_record_and_refuses_gaps(tmp_path):
    references, predictions = _folders(tmp_path, (1, 2))
    record = {
        "schema": PREDICTIONS_SCHEMA,
        "model": {"kind": "checkpoint", "step": 46880, "lineage": [{"kind": "scratch"}]},
        "inputs": dir300.PREDICTION_INPUTS,
        "quantize": "truncate",
        "complete": True,
        "items": [
            {"output": "1.png", "status": "success", "action": "written"},
            {"output": "2.png", "status": "invalid_transform", "action": "written", "forced": True},
        ],
    }
    (predictions / "predictions.json").write_text(json.dumps(record))
    report = dir300.evaluate_dir300(
        predictions,
        build=None,
        references_dir=references,
        verify_references=False,
        documents=(1, 2),
        workers=1,
    )
    assert report["summary"]["prediction_failures"] == 1
    assert report["evidence"]["new_model_real_photo"] is True
    assert report["evidence"]["quantize"] == "truncate"
    (predictions / "2.png").unlink()
    with pytest.raises(FileNotFoundError, match="2.png"):
        dir300.evaluate_dir300(
            predictions, build=None, references_dir=references, verify_references=False,
            documents=(1, 2), workers=1,
        )  # fmt: skip
    with pytest.raises(ValueError, match="not DIR300 documents"):
        dir300.evaluate_dir300(
            predictions, build=None, references_dir=references, verify_references=False,
            documents=(0, 1), workers=1,
        )  # fmt: skip


def test_ocr_scoring_divides_by_the_reference_length():
    summary, rows = dir300.score_ocr({5: "abcd", 7: ""}, {5: "abXd", 7: "zz"}, (5, 7))
    assert [row["ed"] for row in rows] == [1, 2]
    assert rows[0]["cer"] == 0.25 and math.isnan(rows[1]["cer"])
    assert summary == {"cer": 0.25, "ed": 1.5, "n": 2, "n_nan_cer": 1}


class _FakeEngine:
    def __init__(self):
        self.seen = []

    def image_to_string(self, image, config=""):
        self.seen.append((image.size, config))
        return f"size {image.size[0]}x{image.size[1]}\n"

    def metadata(self):
        return {"version_first_line": "fake 1.0"}


def test_ocr_reads_the_subset_at_stored_size_with_default_config(tmp_path):
    references, predictions = _folders(tmp_path, (5, 7))
    Image.fromarray(_page(99, (80, 60))).save(predictions / "7.png")
    engine = _FakeEngine()
    report = dir300.evaluate_dir300_ocr(
        predictions,
        engine=engine,
        references_dir=references,
        verify_references=False,
        documents=(7, 5),
        workers=1,
    )
    assert report["benchmark"] == "dir300-ocr" and report["comparable"] is False
    assert {config for _, config in engine.seen} == {""}
    assert (60, 80) in {size for size, _ in engine.seen}  # no resize before OCR
    rows = {row["id"]: row for row in report["rows"]}
    assert rows["5"]["ed"] == 0 and rows["7"]["ed"] > 0
    assert report["texts"]["reference"]["5"] == report["texts"]["prediction"]["5"]
    assert report["summary"]["n"] == 2
    assert report["evidence"]["measurement"] == "dir300_standard"
    with pytest.raises(ValueError, match="not DIR300 documents"):
        dir300.evaluate_dir300_ocr(
            predictions, engine=engine, references_dir=references, verify_references=False,
            documents=(1,), workers=1,
        )  # fmt: skip


@pytest.fixture
def tiny_weights(tmp_path):
    torch.manual_seed(0)
    weights = tmp_path / "model.pt"
    save_checkpoint(
        weights,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": 3,
            "track": "release",
            "model_config": TINY.to_dict(),
            "lineage": [{"kind": "scratch"}],
            "data_spec_sha256": "0" * 64,
            "code": {"sha256": "1" * 64},
            "model_state": UVDocNet(TINY).state_dict(),
        },
    )
    return weights


def test_predict_writes_each_photo_at_its_own_size(tmp_path, tiny_weights):
    photos = tmp_path / "dist"
    photos.mkdir()
    Image.fromarray(_page(1, (36, 27))).save(photos / "1.png")
    Image.fromarray(_page(2, (30, 40))).save(photos / "2.PNG")
    output = tmp_path / "out"
    record = dir300.predict_dir300(
        load_dewarper(tiny_weights),
        output,
        inputs_dir=photos,
        documents=(1, 2),
        verify_inputs=False,
        force_render=True,
        quantize="truncate",
    )
    assert record["inputs"] == dir300.PREDICTION_INPUTS and record["quantize"] == "truncate"
    assert record["inputs_verified"] is False and set(record["input_files_sha256"]) == {"1", "2"}
    assert [item["output"] for item in record["items"]] == ["1.png", "2.png"]
    assert [item["input"] for item in record["items"]] == ["1.png", "2.PNG"]
    assert record["render_config"]["max_input_pixels"] == dir300.MAX_PHOTO_PIXELS > 3472 * 4624
    with Image.open(output / "2.png") as image:
        assert image.size == (40, 30)
    with pytest.raises(ValueError, match="differs from the official DIR300 files"):
        for k in range(3, 301):
            Image.fromarray(_page(k, (4, 3))).save(photos / f"{k}.png")
        dir300.predict_dir300(load_dewarper(tiny_weights), output, inputs_dir=photos)


def test_predict_passes_rerun_settings_through(tmp_path, tiny_weights):
    from flattener.dewarp.predict import rerun_record

    photos = tmp_path / "dist"
    photos.mkdir()
    Image.fromarray(_page(1, (36, 27))).save(photos / "1.png")
    record = dir300.predict_dir300(
        load_dewarper(tiny_weights),
        tmp_path / "out",
        inputs_dir=photos,
        documents=(1,),
        verify_inputs=False,
        force_render=True,
        rerun=1,
        deskew=True,
    )
    assert record["rerun"] == record["evidence"]["rerun"] == rerun_record(1, True)
    [item] = record["items"]
    assert [crop["pass"] for crop in item["crops"]] == [1]
    parse = cli.build_parser().parse_args
    base = ["dir300-predict", "--weights", "w.pt", "--output-dir", "out"]
    assert (parse(base).rerun, parse(base).deskew) == (0, False)
    args = parse([*base, "--rerun", "2", "--deskew"])
    assert (args.rerun, args.deskew) == (2, True)
    uvdoc_args = parse(["uvdoc-predict", "--weights", "w.pt", "--output-dir", "o", "--rerun", "1"])
    assert (uvdoc_args.rerun, uvdoc_args.deskew) == (1, False)
    for argv in ([*base, "--deskew"], [*base, "--rerun", "-1"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
