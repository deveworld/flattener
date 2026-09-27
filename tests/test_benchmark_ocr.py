"""OCR protocols on fixed strings and a fake engine; Tesseract itself is never run."""

import json
import random
import subprocess

import numpy as np
import pytest
from PIL import Image

from flattener.benchmark import ocr
from flattener.benchmark.reporting import sha256_file, write_json_atomic


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


def test_character_error_rate_divides_by_reference_length():
    assert ocr.character_error_rate("abcd", "abxd") == pytest.approx(0.25)
    assert ocr.character_error_rate("ab", "abcdef") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        ocr.character_error_rate("", "a")


def test_doctr_score_averages_views_then_settings():
    references = {k: "x" * 10 for k in ocr.DOCTR_SETTING_1}
    predictions = {}
    for k in ocr.DOCTR_SETTING_1:
        predictions[(k, 1)] = "x" * 10  # ED 0
        predictions[(k, 2)] = "x" * 8  # ED 2
    summary, rows = ocr.score_doctr(references, predictions, ocr.DOCTR_SETTING_1)
    assert summary["n"] == 60 and len(rows) == 60
    assert summary["views"]["1"]["ed"] == 0.0 and summary["views"]["2"]["ed"] == 2.0
    assert summary["ed"] == pytest.approx(1.0)
    assert summary["cer"] == pytest.approx(0.1)
    assert len(ocr.DEWARPNET_SETTING_2) == 25 and len(ocr.DOCTR_SETTING_1) == 30


def test_uvdoc_score_strips_for_cer_but_not_for_ed():
    references = {k: "  abcd\n\x0c" for k in ocr.DEWARPNET_SETTING_2}
    predictions = {(k, m): "abcd" for k in ocr.DEWARPNET_SETTING_2 for m in (1, 2)}
    summary, rows = ocr.score_uvdoc(references, predictions, ocr.DEWARPNET_SETTING_2)
    assert summary["n"] == 50
    assert summary["cer"] == 0.0
    assert summary["ed"] == pytest.approx(4.0)


def test_prepare_like_pytesseract_composites_alpha_on_white_and_keeps_format():
    pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    pixels[0, 0] = (10, 20, 30, 255)
    image = Image.fromarray(pixels, "RGBA")
    prepared, extension = ocr.prepare_like_pytesseract(image)
    assert extension == "PNG" and prepared.format == "PNG" and prepared.mode == "RGB"
    assert prepared.getpixel((0, 0)) == (10, 20, 30)
    assert prepared.getpixel((2, 1)) == (255, 255, 255)
    jpeg = Image.new("RGB", (2, 2))
    jpeg.format = "JPEG"
    assert ocr.prepare_like_pytesseract(jpeg)[1] == "JPEG"
    odd = Image.new("RGB", (2, 2))
    odd.format = "ICO"
    with pytest.raises(TypeError):
        ocr.prepare_like_pytesseract(odd)


def test_tesseract_command_line_mimics_pytesseract(monkeypatch, tmp_path):
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        with Image.open(arguments[1]) as saved:
            calls[-1] += (saved.mode, saved.getpixel((0, 0)))
        with open(arguments[2] + ".txt", "wb") as stream:
            stream.write("Ünïcode text\n\x0c".encode())
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)
    engine = ocr.Tesseract("/opt/tess/tesseract", temp_dir=str(tmp_path))
    image = Image.new("RGBA", (4, 3), (0, 0, 0, 0))
    assert engine.image_to_string(image) == "Ünïcode text\n\x0c"
    assert engine.image_to_string(image, "--oem 1") == "Ünïcode text\n\x0c"
    (first, kwargs, mode, pixel), (second, *_) = calls
    assert first[0] == "/opt/tess/tesseract" and first[1].endswith(".PNG")
    assert first[3:] == ["txt"] and second[3:] == ["--oem", "1", "txt"]
    assert first[1] == first[2] + ".PNG"
    assert mode == "RGB" and pixel == (255, 255, 255)
    assert kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
    assert list(tmp_path.iterdir()) == []


def test_tesseract_failure_raises(monkeypatch):
    def failing(arguments, **kwargs):
        return subprocess.CompletedProcess(arguments, 1, b"", b"boom")

    monkeypatch.setattr(ocr.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="boom"):
        ocr.Tesseract().image_to_string(Image.new("L", (2, 2)))


def test_tesseract_metadata_records_version_and_hashes(monkeypatch, tmp_path):
    executable = tmp_path / "tesseract"
    executable.write_bytes(b"binary")
    executable.chmod(0o755)
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"model")

    def fake_run(arguments, **kwargs):
        if arguments[1] == "--version":
            return subprocess.CompletedProcess(arguments, 0, b"tesseract v5.0.1\r\n lept\r\n", b"")
        listing = f'List of available languages in "{tessdata}/" (1):\neng\n'.encode()
        return subprocess.CompletedProcess(arguments, 0, listing, b"")

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    record = ocr.Tesseract(str(executable)).metadata()
    assert record["version_first_line"] == "tesseract v5.0.1"
    assert record["executable"] == str(executable.resolve())
    assert len(record["executable_sha256"]) == 64
    assert record["traineddata"]["eng.traineddata"]["bytes"] == 5
    assert record["leptonica_version"] is None and record["libraries"] == {}
    assert record["env"] == {"OMP_THREAD_LIMIT": "1"}


def test_tessdata_dir_sets_tessdata_prefix_and_libraries_are_hashed(monkeypatch, tmp_path):
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    executable = prefix / "bin" / "tesseract"
    executable.write_bytes(b"binary")
    (prefix / "lib" / "libtesseract.so.5.0.0").write_bytes(b"tess")
    (prefix / "lib" / "libtesseract.so.5").symlink_to("libtesseract.so.5.0.0")
    (prefix / "lib" / "liblept.so.5.0.4").write_bytes(b"lept")
    (prefix / "lib" / "liblept.a").write_bytes(b"static")
    (prefix / "lib" / "libz.so.1").write_bytes(b"zlib")
    tessdata = prefix / "share" / "tessdata"
    tessdata.mkdir(parents=True)
    (tessdata / "eng.traineddata").write_bytes(b"model")
    envs = []

    def fake_run(arguments, **kwargs):
        envs.append(kwargs["env"])
        if arguments[1] == "--version":
            banner = b"tesseract 5.0.1\n leptonica-1.82.0\n  libgif 5.2.1\n"
            return subprocess.CompletedProcess(arguments, 0, banner, b"")
        if arguments[1] == "--list-langs":
            where = kwargs["env"]["TESSDATA_PREFIX"]
            listing = f'List of available languages in "{where}/" (1):\neng\n'.encode()
            return subprocess.CompletedProcess(arguments, 0, listing, b"")
        with open(arguments[2] + ".txt", "wb") as stream:
            stream.write(b"text")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)
    engine = ocr.Tesseract(str(executable), temp_dir=str(tmp_path), tessdata_dir=str(tessdata))
    record = engine.metadata()
    assert engine.image_to_string(Image.new("L", (2, 2))) == "text"
    assert all(env["TESSDATA_PREFIX"] == str(tessdata) for env in envs)
    assert all(env["OMP_THREAD_LIMIT"] == "1" for env in envs)
    assert record["env"] == {"OMP_THREAD_LIMIT": "1", "TESSDATA_PREFIX": str(tessdata)}
    assert record["tessdata_dir"] == str(tessdata)
    assert record["leptonica_version"] == "1.82.0"
    assert sorted(record["libraries"]) == ["lib/liblept.so.5.0.4", "lib/libtesseract.so.5.0.0"]
    monkeypatch.setenv("TESSDATA_PREFIX", str(tessdata))
    inherited = ocr.Tesseract(str(executable)).metadata()
    assert inherited["env"]["TESSDATA_PREFIX"] == str(tessdata)


def test_launcher_prefixes_every_call_and_wine_tessdata_paths_map_to_unix(monkeypatch, tmp_path):
    tree = tmp_path / "tree"
    (tree / "tessdata").mkdir(parents=True)
    executable = tree / "tesseract.exe"
    executable.write_bytes(b"exe")
    (tree / "libtesseract-5.dll").write_bytes(b"dll")
    (tree / "tessdata" / "eng.traineddata").write_bytes(b"model")
    wine_path = "Z:" + str(tree).replace("/", "\\") + "/tessdata/"
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        if arguments[2] == "--version":
            return subprocess.CompletedProcess(arguments, 0, b"tesseract v5.0.1.20220118\r\n", b"")
        if arguments[2] == "--list-langs":
            listing = f'List of available languages in "{wine_path}" (1):\r\neng\r\n'
            return subprocess.CompletedProcess(arguments, 0, listing.encode(), b"")
        with open(arguments[3] + ".txt", "wb") as stream:
            stream.write(b"text")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)
    engine = ocr.Tesseract(str(executable), temp_dir=str(tmp_path), launcher=("wine",))
    record = engine.metadata()
    assert engine.image_to_string(Image.new("L", (2, 2))) == "text"
    assert all(call[:2] == ["wine", str(executable)] for call in calls)
    assert record["launcher"] == ["wine"]
    assert record["executable"] == str(executable.resolve())
    assert record["tessdata_dir"] == str(tree / "tessdata")
    assert record["traineddata"]["eng.traineddata"]["bytes"] == 5
    assert record["libraries"] == {
        "tree/libtesseract-5.dll": sha256_file(tree / "libtesseract-5.dll")
    }
    missing = 'List of available languages in "Z:\\no\\such\\dir/" (0):'
    assert ocr._tessdata_dir(missing) is None


def _build_metadata(name, builds=ocr.CALIBRATED_TESSERACT_BUILDS):
    build = builds[name]
    return {
        "executable_sha256": build["executable_sha256"],
        "traineddata": {"eng.traineddata": {"sha256": build["eng_traineddata_sha256"]}},
        "version": f"{build['version_first_line']}\n leptonica-{build['leptonica_version']}\n",
        "version_first_line": build["version_first_line"],
    }


def test_calibrated_build_needs_version_leptonica_executable_and_traineddata():
    assert ocr.leptonica_version("tesseract 5.0.1\n leptonica-1.82.0\n") == "1.82.0"
    assert ocr.leptonica_version("tesseract 5.0.1") is None
    for name in ocr.CALIBRATED_TESSERACT_BUILDS:
        metadata = _build_metadata(name)
        assert ocr.calibrated_build(metadata) == name
        assert ocr.is_calibrated_tesseract(metadata["version_first_line"])
        for key, value in (
            ("executable_sha256", "0" * 64),
            ("version", metadata["version"].replace("leptonica-1.", "leptonica-9.")),
            ("version_first_line", "tesseract 5.5.3"),
            ("traineddata", {"eng.traineddata": {"sha256": "1" * 64}}),
            ("traineddata", {}),
        ):
            assert ocr.calibrated_build({**metadata, key: value}) is None, (name, key)
    assert ocr.calibrated_build(None) is None and ocr.calibrated_build({}) is None
    failed = ocr.G4_FAILED_TESSERACT_BUILDS
    assert not set(failed) & set(ocr.CALIBRATED_TESSERACT_BUILDS)
    for name in failed:
        metadata = _build_metadata(name, failed)
        assert ocr.calibrated_build(metadata) is None
        assert ocr.calibrated_build(metadata, failed) == name


def test_report_is_comparable_only_with_a_calibrated_build(tmp_path, monkeypatch):
    scans, predictions = tmp_path / "scans", tmp_path / "pred"
    scans.mkdir()
    predictions.mkdir()
    for k in ocr.protocol_documents("doctr"):
        Image.new("L", (6, 8), 100).save(scans / f"docunet-reference-{k}.png")
        for m in (1, 2):
            Image.new("L", (5, 7), 100).save(predictions / f"{k}_{m}.png")
    monkeypatch.setattr(ocr, "hash_references", lambda *args, **kwargs: {})
    name = sorted(ocr.CALIBRATED_TESSERACT_BUILDS)[0]

    class Calibrated(_FakeEngine):
        def metadata(self):
            return _build_metadata(name)

    report = ocr.evaluate_ocr(
        predictions, pattern="{k}_{m}.png", engine=Calibrated(), references_dir=scans, workers=1
    )
    assert report["comparable"] is True
    assert report["comparability"]["tesseract_build"] == name
    same_version = {**_build_metadata(name), "executable_sha256": "0" * 64}
    Calibrated.metadata = lambda self: same_version
    other = ocr.evaluate_ocr(
        predictions, pattern="{k}_{m}.png", engine=Calibrated(), references_dir=scans, workers=1
    )
    assert other["comparable"] is False
    assert other["comparability"]["tesseract_build"] is None
    assert other["comparability"]["tesseract_build_failed_g4"] is None
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
        verify_references=False,
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
    assert len(report["texts"]["doctr"]["prediction"]) == 2 * len(documents)
    assert report["tesseract"] == {"version_first_line": "fake 1.0"}
    assert report["comparable"] is False
    assert report["comparability"]["tesseract_version_matches"] is False
    assert report["comparability"]["tesseract_build"] is None
    assert report["config"]["tesseract_config"] == {"doctr": "", "uvdoc": "--oem 1"}
    assert len(report["inputs"]["crops"]) == 50
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
        verify_references=False, workers=1,
    )["summary"]["doctr"]  # fmt: skip
    assert counted["setting1_doctr_60"]["prediction_failures"] == 2  # documents 1 and 2
    assert counted["setting1_doctr_60"]["prediction_unrecorded"] == 1
    assert counted["setting2_dewarpnet_50"]["prediction_failures"] == 1  # document 1 only
    (predictions / "predictions.json").unlink()
    write_json_atomic(tmp_path / "ocr.json", report)
    assert json.loads((tmp_path / "ocr.json").read_text())["summary"] == report["summary"]
    (crops / "1_1 copy.png").unlink()
    with pytest.raises(FileNotFoundError):
        ocr.evaluate_ocr(
            predictions, pattern="{k}_{m}.png", protocols=["uvdoc"], engine=engine,
            references_dir=scans, verify_references=False, crop_dir=crops,
        )  # fmt: skip
    with pytest.raises(ValueError):
        ocr.evaluate_ocr(predictions, pattern="{k}_{m}.png", protocols=["tesseract"])


def test_only_tesseract_501_is_the_calibrated_engine():
    for banner in ("tesseract v5.0.1.20220118", "tesseract 5.0.1"):
        assert ocr.is_calibrated_tesseract(banner), banner
    for banner in ("tesseract 5.5.3", "tesseract v5.0.10", "tesseract 15.0.1", "fake 1.0", None):
        assert not ocr.is_calibrated_tesseract(banner), banner
