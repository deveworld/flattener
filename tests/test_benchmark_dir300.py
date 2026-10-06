"""DIR300 benchmark: pairing, reference preparation, reports, OCR subset and CLI."""

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import load_dewarper
from tools.benchmark import __main__ as cli
from tools.benchmark import dir300, matlab, metrics
from tools.benchmark.reporting import PREDICTIONS_SCHEMA

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


def test_evaluate_pairs_by_id_counts_recorded_failures_and_refuses_gaps(tmp_path):
    # References scale by 598400/H/W, which differs from 598400/(H*W) in the last bit here.
    page = _page(0, (23, 27))
    scale = math.sqrt(598400 / 23 / 27)
    assert scale != matlab.metric_scale(23, 27, 598400)
    prepared = dir300.prepare_reference(page)
    assert prepared.image.shape == (math.ceil(23 * scale), math.ceil(27 * scale))
    assert np.array_equal(prepared.image, metrics.prepare_reference(page).image)

    references, predictions = _folders(tmp_path)
    arguments = dict(build=None, references_dir=references, workers=1)
    report = dir300.evaluate_dir300(predictions, documents=(248, 1, 2), **arguments)
    assert report["benchmark"] == "dir300" and report["comparable"] is False
    assert [row["id"] for row in report["rows"]] == ["1", "2", "248"]
    assert report["rows"][2]["reference_file"] == "248.PNG"
    summary = report["summary"]
    # The prediction is resized to the reference's metric size, not by the same scale.
    assert summary["n"] == 3 and summary["ms_ssim"] == pytest.approx(1.0, abs=0.01)
    assert summary["ld"] is None and summary["prediction_failures"] is None
    assert report["evidence"]["measurement"] == "dir300_standard"
    assert report["evidence"]["model_provenance"] == "unknown"
    assert "(uvdoc 0.622, diff" in dir300.format_comparison(summary)
    json.dumps(report, allow_nan=False)

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
    report = dir300.evaluate_dir300(predictions, documents=(1, 2), **arguments)
    assert report["summary"]["prediction_failures"] == 1
    assert report["evidence"]["model_provenance"] == "own_checkpoint"
    assert report["evidence"]["checkpoint_step"] == 46880
    assert report["evidence"]["initialization"] == "scratch"
    assert report["evidence"]["quantize"] == "truncate"
    (predictions / "2.png").unlink()
    with pytest.raises(FileNotFoundError, match="2.png"):
        dir300.evaluate_dir300(predictions, documents=(1, 2), **arguments)
    with pytest.raises(ValueError, match="not DIR300 documents"):
        dir300.evaluate_dir300(predictions, documents=(0, 1), **arguments)


class _FakeEngine:
    def __init__(self):
        self.seen = []

    def image_to_string(self, image, config=""):
        self.seen.append((image.size, config))
        return f"size {image.size[0]}x{image.size[1]}\n"

    def metadata(self):
        return {"version_first_line": "fake 1.0"}


def test_ocr_reads_the_subset_at_stored_size_and_divides_by_the_reference_length(tmp_path):
    references, predictions = _folders(tmp_path, (5, 7))
    Image.fromarray(_page(99, (80, 60))).save(predictions / "7.png")
    engine = _FakeEngine()
    report = dir300.evaluate_dir300_ocr(
        predictions,
        engine=engine,
        references_dir=references,
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
            predictions, engine=engine, references_dir=references,
            documents=(1,), workers=1,
        )  # fmt: skip
    # CER divides by the reference length; an empty reference has no CER.
    summary, rows = dir300.score_ocr({5: "abcd", 7: ""}, {5: "abXd", 7: "zz"}, (5, 7))
    assert [row["ed"] for row in rows] == [1, 2] and rows[1]["cer"] is None
    assert summary == {"cer": 0.25, "ed": 1.5, "n": 2, "n_nan_cer": 1}
    json.dumps({"summary": summary, "rows": rows}, allow_nan=False)


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
            "model_state": UVDocNet(TINY).state_dict(),
        },
    )
    return weights


def test_predict_writes_each_photo_at_its_own_size_and_records_rerun_settings(
    tmp_path, tiny_weights
):
    from flattener.dewarp.predict import rerun_record

    photos = tmp_path / "dist"
    photos.mkdir()
    Image.fromarray(_page(1, (36, 27))).save(photos / "1.png")
    Image.fromarray(_page(2, (30, 40))).save(photos / "2.PNG")
    dewarper = load_dewarper(tiny_weights)
    arguments = dict(inputs_dir=photos, force_render=True)
    output = tmp_path / "out"
    record = dir300.predict_dir300(
        dewarper, output, documents=(1, 2), quantize="truncate", **arguments
    )
    assert record["inputs"] == dir300.PREDICTION_INPUTS and record["quantize"] == "truncate"
    assert record["input_dir"] == str(photos.resolve())
    assert [item["input"] for item in record["items"]] == ["1.png", "2.PNG"]
    assert record["render_config"]["max_input_pixels"] == dir300.MAX_PHOTO_PIXELS > 3472 * 4624
    with Image.open(output / "2.png") as image:
        assert image.size == (40, 30)
    rerun = dir300.predict_dir300(
        dewarper, tmp_path / "rerun", documents=(1,), rerun=1, deskew=True, **arguments
    )
    assert rerun["rerun"] == rerun["evidence"]["rerun"] == rerun_record(1, True)
    assert [crop["pass"] for crop in rerun["items"][0]["crops"]] == [1]
    base = ["dir300-predict", "--weights", "w.pt", "--output-dir", "out"]
    args = cli.build_parser().parse_args([*base, "--rerun", "2", "--deskew"])
    assert (args.rerun, args.deskew) == (2, True)
    for argv in ([*base, "--deskew"], [*base, "--rerun", "-1"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    with pytest.raises(FileNotFoundError, match="missing DIR300 image"):
        dir300.predict_dir300(dewarper, output, inputs_dir=photos)
