"""OCR protocols on fixed strings and a fake engine; Tesseract itself is never run."""

import json
import random
import subprocess

import numpy as np
import pytest
from PIL import Image

from tools.benchmark import ocr


def _dp_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def test_levenshtein_fixed_strings_and_random_dp_oracle():
    assert ocr.levenshtein("kitten", "sitting") == 3
    assert ocr.levenshtein("", "abc") == ocr.levenshtein("abc", "") == 3
    assert ocr.levenshtein("", "") == 0
    assert ocr.levenshtein("flaw", "lawn") == 2
    assert ocr.levenshtein("Straße\n\x0c", "Strasse\n") == 3
    rng = random.Random(11)
    for _ in range(200):
        a = "".join(rng.choice("ab c\ne") for _ in range(rng.randint(0, 70)))
        b = "".join(rng.choice("ab c\nf") for _ in range(rng.randint(0, 70)))
        assert ocr.levenshtein(a, b) == _dp_distance(a, b) == ocr.levenshtein(b, a)
    long_a = "".join(rng.choice("abcdefgh \n") for _ in range(300))
    long_b = long_a[5:150] + "xyz" + long_a[150:]
    assert ocr.levenshtein(long_a, long_b) == _dp_distance(long_a, long_b)


def test_doctr_averages_views_then_settings_and_uvdoc_strips_only_for_cer():
    references = {k: "x" * 10 for k in ocr.DOCTR_SETTING_1}
    predictions = {}
    for k in ocr.DOCTR_SETTING_1:
        predictions[(k, 1)] = "x" * 10  # ED 0
        predictions[(k, 2)] = "x" * 8  # ED 2
    summary, rows = ocr.score_doctr(references, predictions, ocr.DOCTR_SETTING_1)
    assert summary["n"] == 60 and len(rows) == 60
    assert summary["views"]["1"]["ed"] == 0.0 and summary["views"]["2"]["ed"] == 2.0
    assert summary["ed"] == pytest.approx(1.0) and summary["cer"] == pytest.approx(0.1)
    assert len(ocr.DEWARPNET_SETTING_2) == 25 and len(ocr.DOCTR_SETTING_1) == 30
    references = {k: "  abcd\n\x0c" for k in ocr.DEWARPNET_SETTING_2}
    predictions = {(k, m): "abcd" for k in ocr.DEWARPNET_SETTING_2 for m in (1, 2)}
    summary, rows = ocr.score_uvdoc(references, predictions, ocr.DEWARPNET_SETTING_2)
    assert summary["n"] == 50 and summary["cer"] == 0.0
    assert summary["ed"] == pytest.approx(4.0)


def test_empty_reference_cer_serializes_as_null():
    for score, second, expected_cer in (
        (ocr.score_doctr, "page", None),
        (ocr.score_uvdoc, "page", 0.0),
        (ocr.score_uvdoc, "", None),
    ):
        references = {1: "", 9: second}
        predictions = {(k, m): "page" for k in references for m in (1, 2)}
        summary, rows = score(references, predictions, tuple(references))
        assert rows[0]["cer"] is None and summary["cer"] == expected_cer
        json.dumps({"summary": summary, "rows": rows}, allow_nan=False)


def test_tesseract_is_called_like_pytesseract(monkeypatch, tmp_path):
    pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    pixels[0, 0] = (10, 20, 30, 255)
    prepared, extension = ocr.prepare_like_pytesseract(Image.fromarray(pixels, "RGBA"))
    assert extension == "PNG" and prepared.format == "PNG" and prepared.mode == "RGB"
    assert prepared.getpixel((0, 0)) == (10, 20, 30)
    assert prepared.getpixel((2, 1)) == (255, 255, 255)  # alpha over white
    jpeg = Image.new("RGB", (2, 2))
    jpeg.format = "JPEG"
    assert ocr.prepare_like_pytesseract(jpeg)[1] == "JPEG"
    jpeg.format = "ICO"
    with pytest.raises(TypeError):
        ocr.prepare_like_pytesseract(jpeg)
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        with open(arguments[2] + ".txt", "wb") as stream:
            stream.write("Ünïcode text\n\x0c".encode())
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)
    engine = ocr.Tesseract("/opt/tess/tesseract", temp_dir=str(tmp_path))
    image = Image.new("RGBA", (4, 3), (0, 0, 0, 0))
    assert engine.image_to_string(image) == "Ünïcode text\n\x0c"
    assert engine.image_to_string(image, "--oem 1") == "Ünïcode text\n\x0c"
    (first, kwargs), (second, _) = calls
    assert first[0] == "/opt/tess/tesseract" and first[1] == first[2] + ".PNG"
    assert first[3:] == ["txt"] and second[3:] == ["--oem", "1", "txt"]
    assert kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
    assert list(tmp_path.iterdir()) == []


def _build_metadata(name, builds=ocr.CALIBRATED_TESSERACT_BUILDS):
    build = builds[name]
    return {
        "version": f"{build['version_first_line']}\n leptonica-{build['leptonica_version']}\n",
        "version_first_line": build["version_first_line"],
    }


def test_report_is_comparable_only_with_a_calibrated_build(tmp_path, monkeypatch):
    scans, predictions = tmp_path / "scans", tmp_path / "pred"
    scans.mkdir()
    predictions.mkdir()
    for k in ocr.protocol_documents("doctr"):
        Image.new("L", (6, 8), 100).save(scans / f"docunet-reference-{k}.png")
        for m in (1, 2):
            Image.new("L", (5, 7), 100).save(predictions / f"{k}_{m}.png")
    monkeypatch.setattr(ocr, "reference_paths", lambda *args, **kwargs: {})
    name = sorted(ocr.CALIBRATED_TESSERACT_BUILDS)[0]

    class Calibrated(_FakeEngine):
        def metadata(self):
            return _build_metadata(name)

    report = ocr.evaluate_ocr(
        predictions, pattern="{k}_{m}.png", engine=Calibrated(), references_dir=scans, workers=1
    )
    assert report["comparable"] is True
    assert report["comparability"]["tesseract_build"] == name
    same_version = {**_build_metadata(name), "version": "tesseract 5.0.1\n leptonica-0.1\n"}
    Calibrated.metadata = lambda self: same_version
    other = ocr.evaluate_ocr(
        predictions, pattern="{k}_{m}.png", engine=Calibrated(), references_dir=scans, workers=1
    )
    assert other["comparable"] is False
    assert other["comparability"]["tesseract_build"] is None
    assert other["comparability"]["tesseract_version_matches"] is True


class _FakeEngine:
    """Returns the mean gray level as text so every image maps to a known string."""

    def __init__(self):
        self.seen = []

    def image_to_string(self, image, config=""):
        self.seen.append((image.size, config))
        level = int(np.asarray(image.convert("L")).mean())
        return f"page {level}\n\x0c"

    def metadata(self):
        return {"version_first_line": "fake 1.0"}


def test_evaluate_ocr_report_with_fake_engine(tmp_path):
    scans, predictions, crops = (tmp_path / name for name in ("scans", "pred", "crop"))
    for folder in (scans, predictions, crops):
        folder.mkdir()
    documents = sorted(set(ocr.DOCTR_SETTING_1) | set(ocr.DEWARPNET_SETTING_2))
    for k in documents:
        Image.new("L", (6, 8), 100).save(scans / f"docunet-reference-{k}.png")
        for m in (1, 2):
            Image.new("RGB", (5, 7), (100, 100, 100) if m == 1 else (0, 0, 0)).save(
                predictions / f"{k}_{m}.png"
            )
            Image.new("RGB", (9, 4)).save(crops / f"{k}_{m} copy.png")
    engine = _FakeEngine()
    report = ocr.evaluate_ocr(
        predictions,
        pattern="{k}_{m}.png",
        protocols=["doctr", "uvdoc"],
        engine=engine,
        references_dir=scans,
        crop_dir=crops,
        workers=2,
    )
    assert report["schema"] == ocr.REPORT_SCHEMA
    doctr = report["summary"]["doctr"]
    assert doctr["setting1_doctr_60"]["n"] == 60 and doctr["setting2_dewarpnet_50"]["n"] == 50
    reference = "page 100\n\x0c"
    distance = ocr.levenshtein(reference, "page 0\n\x0c")
    assert doctr["setting1_doctr_60"]["views"]["1"]["ed"] == 0.0
    assert doctr["setting1_doctr_60"]["ed"] == pytest.approx(distance / 2)
    assert doctr["setting1_doctr_60"]["cer"] == pytest.approx(distance / len(reference) / 2)
    uvdoc = report["summary"]["uvdoc"]["uvdoc_50"]
    assert uvdoc["n"] == 50
    assert ((9, 4), "--oem 1") in engine.seen and ((5, 7), "") in engine.seen
    assert ((5, 7), "--oem 1") not in engine.seen
    assert report["texts"]["doctr"]["reference"]["1"] == reference
    assert report["comparable"] is False
    assert report["config"]["tesseract_config"] == {"doctr": "", "uvdoc": "--oem 1"}
    assert report["inputs"]["crops_dir"] == str(crops.resolve())
    assert doctr["setting1_doctr_60"]["prediction_failures"] is None  # no predictions.json
    items = [
        {"output": f"{k}_{m}.png", "action": "written", "status": "success"}
        for k in documents
        for m in (1, 2)
        if (k, m) != (2, 1)  # 2_1.png is left unrecorded
    ]
    items[0].update(status="invalid_transform", forced=True)  # 1_1.png
    record = {"schema": "flattener-dewarp-predictions-2", "model": {}, "items": items}
    (predictions / "predictions.json").write_text(json.dumps(record))
    counted = ocr.evaluate_ocr(
        predictions, pattern="{k}_{m}.png", engine=engine, references_dir=scans,
         workers=1,
    )["summary"]["doctr"]  # fmt: skip
    assert counted["setting1_doctr_60"]["prediction_failures"] == 2  # documents 1 and 2
    assert counted["setting1_doctr_60"]["prediction_unrecorded"] == 1
    assert counted["setting2_dewarpnet_50"]["prediction_failures"] == 1  # document 1 only
    (predictions / "predictions.json").unlink()
    (crops / "1_1 copy.png").unlink()
    with pytest.raises(FileNotFoundError):
        ocr.evaluate_ocr(
            predictions, pattern="{k}_{m}.png", protocols=["uvdoc"], engine=engine,
            references_dir=scans,  crop_dir=crops,
        )  # fmt: skip
    with pytest.raises(ValueError):
        ocr.evaluate_ocr(predictions, pattern="{k}_{m}.png", protocols=["tesseract"])
