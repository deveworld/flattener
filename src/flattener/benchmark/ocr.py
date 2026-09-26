"""DocUNet OCR metrics (edit distance and CER) with the Tesseract command line.

Two published protocols are reimplemented from their descriptions (no DocTr or UVDoc
code is copied; only the OCR document lists are cited):

``doctr`` (primary; DocTr ``OCR_eval.py``, used by DocScanner/DocGeoNet tables)
    ``pytesseract.image_to_string`` defaults (no ``-l``/``--psm``/``--oem`` flags),
    raw strings, ``ED = Levenshtein(ref, pred)``, ``CER = ED / len(ref)``. Setting 1
    uses the 30 DocTr documents, Setting 2 the 25 DewarpNet documents, both views.
    Each view is averaged over the documents, then the two view means are averaged.
``uvdoc`` (secondary; UVDoc ``ocr_eval.py``)
    ``--oem 1``; the prediction is resized to its ``crop`` photo size with Pillow's
    default ``Image.resize``; ``CER`` is computed on stripped strings (jiwer),
    ``ED`` on raw strings; plain means over the 50 Setting 2 images.

Images are passed to Tesseract exactly as pytesseract 0.3.8 does: alpha is
composited over **white**, the image is re-saved in its own format (PNG by default)
to a temporary file and ``tesseract IN OUTBASE [config] txt`` is run; the UTF-8 text
file is returned verbatim. Metric scoring (``docunet.py``) composites over black.

OCR numbers depend on the Tesseract build: gate G4 reproduces the DocScanner table
only with Tesseract 5.0.1 (UB-Mannheim 5.0.1.20220118), while 5.5.3 gives 3-5 % lower
edit distances. Reports are marked ``comparable`` only for that engine version and
the published DocUNet scans.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .docunet import (
    DEFAULT_REFERENCE_PATTERN,
    DEFAULT_REFERENCES_DIR,
    VIEWS,
    format_name,
    hash_references,
)
from .reporting import (
    environment,
    mean,
    prediction_record,
    report_evidence,
    sha256_file,
    utc_now,
)

REPORT_SCHEMA = "flattener-benchmark-ocr-1.0"
DEFAULT_WORKERS = 2
PROTOCOLS = ("doctr", "uvdoc")
DOCTR_SETTING_1 = (
    1, 2, 3, 4, 5, 6, 7, 9, 10, 21, 22, 23, 24, 27, 30,
    31, 32, 36, 38, 40, 41, 44, 45, 46, 47, 48, 50, 51, 52, 53,
)  # fmt: skip
DEWARPNET_SETTING_2 = (
    1, 9, 10, 12, 19, 20, 21, 22, 23, 24, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40,
    44, 45, 46, 47, 49,
)  # fmt: skip
SETTINGS = {
    "setting1_doctr_60": DOCTR_SETTING_1,
    "setting2_dewarpnet_50": DEWARPNET_SETTING_2,
}
PROTOCOL_CONFIG = {"doctr": "", "uvdoc": "--oem 1"}
# The engine gate G4 is calibrated with; other versions give different OCR numbers.
CALIBRATED_TESSERACT_VERSION = "5.0.1"
_CALIBRATED_VERSION_PATTERN = re.compile(r"(?:^|[^\d.])v?5\.0\.1(?:$|[^\d])")
DEFAULT_CROP_DIR = Path("outputs/benchmarks/docunet/crop")
DEFAULT_CROP_PATTERN = "{k}_{m} copy.png"
# Image formats pytesseract 0.3.8 accepts (SUPPORTED_FORMATS).
_SUPPORTED_FORMATS = {"JPEG", "PNG", "PBM", "PGM", "PPM", "TIFF", "BMP", "GIF", "WEBP"}

PROTOCOL_DESCRIPTION = {
    "doctr": {
        "cer": "ED / len(reference OCR string)",
        "ed": "Levenshtein distance between raw OCR strings",
        "aggregation": "mean over documents per view, then mean of the two views",
        "tesseract_args": "tesseract IN OUT txt (pytesseract.image_to_string defaults)",
        "settings": {name: list(documents) for name, documents in SETTINGS.items()},
        "source": "https://github.com/fh2019ustc/DocTr/blob/main/OCR_eval.py",
    },
    "uvdoc": {
        "cer": "Levenshtein(ref.strip(), pred.strip()) / len(ref.strip()) (jiwer cer)",
        "ed": "Levenshtein distance between raw OCR strings",
        "aggregation": "nanmean over the 50 images",
        "prediction_resize": "PIL Image.resize(crop photo size) with the default filter",
        "tesseract_args": "tesseract IN OUT --oem 1 txt",
        "settings": {"uvdoc_50": list(DEWARPNET_SETTING_2)},
        "source": "https://github.com/tanguymagne/UVDoc/blob/main/eval/ocr_eval/ocr_eval.py",
    },
    "alpha_composite": "white (pytesseract prepare); metric scoring uses black",
}


def levenshtein(first: str, second: str) -> int:
    """Unit-cost edit distance (Myers/Hyyrö bit-parallel algorithm on Python ints)."""
    if not isinstance(first, str) or not isinstance(second, str):
        raise TypeError("levenshtein expects two strings")
    if len(first) < len(second):
        first, second = second, first
    length = len(second)
    if length == 0:
        return len(first)
    peq: dict[str, int] = {}
    for index, character in enumerate(second):
        peq[character] = peq.get(character, 0) | (1 << index)
    mask = (1 << length) - 1
    last = 1 << (length - 1)
    positive = mask
    negative = 0
    score = length
    for character in first:
        equal = peq.get(character, 0)
        vertical = equal | negative
        horizontal = ((((equal & positive) + positive) & mask) ^ positive) | equal
        plus = negative | (~(horizontal | positive) & mask)
        minus = positive & horizontal
        if plus & last:
            score += 1
        elif minus & last:
            score -= 1
        plus = ((plus << 1) | 1) & mask
        minus = (minus << 1) & mask
        positive = minus | (~(vertical | plus) & mask)
        negative = plus & vertical
    return score


def character_error_rate(reference: str, prediction: str) -> float:
    """DocTr CER: raw edit distance divided by the reference OCR length."""
    if not reference:
        raise ValueError("CER is undefined for an empty reference OCR string")
    return levenshtein(reference, prediction) / len(reference)


# ``prepare_like_pytesseract`` follows ``prepare()`` of pytesseract 0.3.8
# (https://github.com/madmaze/pytesseract, Copyright the pytesseract authors,
# Apache License 2.0) statement for statement, so OCR inputs match the published
# protocols byte for byte.
def prepare_like_pytesseract(image: Image.Image) -> tuple[Image.Image, str]:
    """pytesseract 0.3.8 ``prepare``: keep the format, composite alpha over white."""
    if not isinstance(image, Image.Image):
        raise TypeError("expected a PIL image")
    extension = image.format or "PNG"
    if extension not in _SUPPORTED_FORMATS:
        raise TypeError(f"unsupported image format for Tesseract: {extension}")
    if "A" in image.getbands():
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, (0, 0), image.getchannel("A"))
        image = background
    image.format = extension
    return image, extension


class OcrEngine(Protocol):
    def image_to_string(self, image: Image.Image, config: str = "") -> str: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Tesseract:
    """The ``tesseract`` executable, called the way ``pytesseract.image_to_string`` does."""

    executable: str = "tesseract"
    temp_dir: str | None = None

    def _command(self) -> str:
        return self.executable

    def image_to_string(self, image: Image.Image, config: str = "") -> str:
        prepared, extension = prepare_like_pytesseract(image)
        with tempfile.TemporaryDirectory(prefix="tess_", dir=self.temp_dir) as directory:
            base = os.path.join(directory, "tess")
            input_path = f"{base}.{extension}"
            prepared.save(input_path, format=prepared.format)
            arguments = [self._command(), input_path, base, *shlex.split(config), "txt"]
            completed = subprocess.run(
                arguments,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                check=False,
            )
            if completed.returncode != 0:
                message = completed.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(f"tesseract failed ({completed.returncode}): {message}")
            with open(f"{base}.txt", "rb") as stream:
                return stream.read().decode("utf-8")

    def metadata(self) -> dict[str, Any]:
        """Version banner plus SHA-256 of the executable and the language data."""
        version = subprocess.run(
            [self._command(), "--version"],
            capture_output=True,
            check=True,
            stdin=subprocess.DEVNULL,
        )
        banner = (version.stdout or version.stderr).decode("utf-8", "replace").replace("\r\n", "\n")
        languages = subprocess.run(
            [self._command(), "--list-langs"],
            capture_output=True,
            check=True,
            stdin=subprocess.DEVNULL,
        )
        listing = (languages.stdout + languages.stderr).decode("utf-8", "replace")
        resolved = _resolve_executable(self._command())
        tessdata = _tessdata_dir(listing)
        traineddata = {}
        if tessdata is not None:
            for name in ("eng.traineddata", "osd.traineddata"):
                path = tessdata / name
                if path.is_file():
                    traineddata[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        return {
            "executable": None if resolved is None else str(resolved),
            "executable_sha256": None if resolved is None else sha256_file(resolved),
            "tessdata_dir": None if tessdata is None else str(tessdata),
            "traineddata": traineddata,
            "version": banner.strip(),
            "version_first_line": banner.strip().splitlines()[0] if banner.strip() else None,
            "env": {"OMP_THREAD_LIMIT": "1"},
        }


def is_calibrated_tesseract(version_first_line: str | None) -> bool:
    """Whether a ``tesseract --version`` banner is the version gate G4 calibrates."""
    if not isinstance(version_first_line, str):
        return False
    return _CALIBRATED_VERSION_PATTERN.search(version_first_line.strip()) is not None


def _resolve_executable(command: str) -> Path | None:
    import shutil

    if Path(command).is_file():
        return Path(command).resolve()
    found = shutil.which(command)
    return None if found is None else Path(found).resolve()


def _tessdata_dir(listing: str) -> Path | None:
    match = re.search(r'List of available languages in "(.+?)"', listing)
    if match is None:
        return None
    path = Path(match.group(1).rstrip("/\\"))
    return path if path.is_dir() else None


@dataclass(frozen=True)
class OcrJob:
    protocol: str
    kind: str  # "reference" or "prediction"
    document: int
    view: int  # 0 for references
    path: str
    resize_to: tuple[int, int] | None = None  # (width, height), UVDoc protocol only

    @property
    def key(self) -> str:
        return f"{self.document}" if self.kind == "reference" else f"{self.document}_{self.view}"


def run_job(job: OcrJob, engine: OcrEngine) -> str:
    with Image.open(job.path) as image:
        image.load()
        if job.resize_to is not None:
            # Pillow's default filter, as ``our_img.resize(input_image_shape)`` in UVDoc.
            image = image.resize(job.resize_to)
        return engine.image_to_string(image, PROTOCOL_CONFIG[job.protocol])


def run_jobs(
    jobs: Sequence[OcrJob],
    engine: OcrEngine,
    *,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[OcrJob, int, int, float], None] | None = None,
) -> dict[tuple[str, str, str], str]:
    """Run OCR jobs on ``workers`` threads (each Tesseract process is single-threaded)."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    results: dict[tuple[str, str, str], str] = {}

    def task(job: OcrJob) -> tuple[OcrJob, str, float]:
        started = time.perf_counter()
        return job, run_job(job, engine), time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for done, (job, text, seconds) in enumerate(executor.map(task, jobs), start=1):
            results[(job.protocol, job.kind, job.key)] = text
            if progress is not None:
                progress(job, done, len(jobs), seconds)
    return results


def score_doctr(
    references: dict[int, str], predictions: dict[tuple[int, int], str], documents: Sequence[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """DocTr aggregation: per-view means over ``documents``, then the mean of both views."""
    rows = []
    per_view: dict[int, dict[str, list[float]]] = {view: {"ed": [], "cer": []} for view in VIEWS}
    for document in documents:
        reference = references[document]
        for view in VIEWS:
            prediction = predictions[(document, view)]
            distance = levenshtein(reference, prediction)
            cer = distance / len(reference) if reference else float("nan")
            per_view[view]["ed"].append(float(distance))
            per_view[view]["cer"].append(cer)
            rows.append(
                {
                    "cer": cer,
                    "ed": distance,
                    "id": f"{document}_{view}",
                    "k": document,
                    "m": view,
                    "prediction_chars": len(prediction),
                    "reference_chars": len(reference),
                }
            )
    views = {
        str(view): {
            "cer": mean(values["cer"]),
            "ed": mean(values["ed"]),
            "n": len(values["ed"]),
        }
        for view, values in per_view.items()
    }
    summary = {
        "cer": mean([views[str(view)]["cer"] for view in VIEWS]),
        "ed": mean([views[str(view)]["ed"] for view in VIEWS]),
        "n": sum(item["n"] for item in views.values()),
        "views": views,
    }
    return summary, rows


def score_uvdoc(
    references: dict[int, str], predictions: dict[tuple[int, int], str], documents: Sequence[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """UVDoc aggregation: stripped-string CER, raw ED, plain (nan)means over all images."""
    rows = []
    for document in documents:
        reference = references[document]
        for view in VIEWS:
            prediction = predictions[(document, view)]
            stripped = reference.strip()
            cer = (
                levenshtein(stripped, prediction.strip()) / len(stripped)
                if stripped
                else float("nan")
            )
            rows.append(
                {
                    "cer": cer,
                    "ed": levenshtein(prediction, reference),
                    "id": f"{document}_{view}",
                    "k": document,
                    "m": view,
                    "prediction_chars": len(prediction),
                    "reference_chars": len(reference),
                }
            )
    cers = [row["cer"] for row in rows if row["cer"] == row["cer"]]
    summary = {
        "cer": mean(cers),
        "ed": mean([float(row["ed"]) for row in rows]),
        "n": len(rows),
        "n_nan_cer": len(rows) - len(cers),
    }
    return summary, rows


def validate_protocols(protocols: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(protocols))
    if not selected:
        raise ValueError("at least one OCR protocol is required")
    for protocol in selected:
        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    return selected


def protocol_documents(protocol: str) -> tuple[int, ...]:
    if protocol == "doctr":
        return tuple(sorted(set(DOCTR_SETTING_1) | set(DEWARPNET_SETTING_2)))
    if protocol == "uvdoc":
        return DEWARPNET_SETTING_2
    raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")


def evaluate_ocr(
    predictions_dir: str | os.PathLike[str],
    *,
    pattern: str,
    protocols: Sequence[str] = ("doctr",),
    engine: OcrEngine | None = None,
    references_dir: str | os.PathLike[str] = DEFAULT_REFERENCES_DIR,
    reference_pattern: str = DEFAULT_REFERENCE_PATTERN,
    verify_references: bool = True,
    crop_dir: str | os.PathLike[str] = DEFAULT_CROP_DIR,
    crop_pattern: str = DEFAULT_CROP_PATTERN,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[OcrJob, int, int, float], None] | None = None,
) -> dict[str, Any]:
    """OCR a directory of DocUNet predictions and return the report dictionary."""
    protocols = validate_protocols(protocols)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    engine = Tesseract() if engine is None else engine
    predictions_root = Path(predictions_dir).expanduser().resolve()
    references_root = Path(references_dir).expanduser().resolve()
    crop_root = Path(crop_dir).expanduser().resolve()
    if not predictions_root.is_dir():
        raise NotADirectoryError(f"predictions directory not found: {predictions_root}")
    documents = sorted({k for protocol in protocols for k in protocol_documents(protocol)})
    reference_hashes = hash_references(
        references_root, reference_pattern, documents, verify=verify_references
    )
    jobs: list[OcrJob] = []
    prediction_hashes: dict[str, str] = {}
    crop_hashes: dict[str, str] = {}
    missing = []
    for protocol in protocols:
        for k in protocol_documents(protocol):
            reference = references_root / format_name(reference_pattern, fields=("k",), k=k)
            jobs.append(OcrJob(protocol, "reference", k, 0, str(reference)))
            for m in VIEWS:
                path = predictions_root / format_name(pattern, fields=("k", "m"), k=k, m=m)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                prediction_hashes.setdefault(f"{k}_{m}", sha256_file(path))
                resize_to = None
                if protocol == "uvdoc":
                    crop = crop_root / format_name(crop_pattern, fields=("k", "m"), k=k, m=m)
                    if not crop.is_file():
                        missing.append(str(crop))
                        continue
                    crop_hashes.setdefault(f"{k}_{m}", sha256_file(crop))
                    with Image.open(crop) as photo:
                        resize_to = photo.size
                jobs.append(OcrJob(protocol, "prediction", k, m, str(path), resize_to))
    if missing:
        raise FileNotFoundError(f"{len(missing)} OCR inputs missing, first: {missing[:5]}")
    engine_metadata = engine.metadata()
    started = time.perf_counter()
    texts = run_jobs(jobs, engine, workers=workers, progress=progress)
    wall_seconds = time.perf_counter() - started
    summary: dict[str, Any] = {}
    rows: dict[str, Any] = {}
    stored_texts: dict[str, Any] = {}
    for protocol in protocols:
        references = {
            int(key): text
            for (name, kind, key), text in texts.items()
            if name == protocol and kind == "reference"
        }
        predictions = {
            tuple(int(part) for part in key.split("_")): text
            for (name, kind, key), text in texts.items()
            if name == protocol and kind == "prediction"
        }
        stored_texts[protocol] = {
            "prediction": {f"{k}_{m}": predictions[(k, m)] for k, m in sorted(predictions)},
            "reference": {str(k): references[k] for k in sorted(references)},
        }
        if protocol == "doctr":
            summary["doctr"] = {}
            rows["doctr"] = {}
            for setting, setting_documents in SETTINGS.items():
                summary["doctr"][setting], rows["doctr"][setting] = score_doctr(
                    references, predictions, setting_documents
                )
        else:
            uvdoc_summary, uvdoc_rows = score_uvdoc(references, predictions, DEWARPNET_SETTING_2)
            summary["uvdoc"] = {"uvdoc_50": uvdoc_summary}
            rows["uvdoc"] = {"uvdoc_50": uvdoc_rows}
    calibrated_engine = is_calibrated_tesseract(engine_metadata.get("version_first_line"))
    record = prediction_record(predictions_root)
    return {
        "benchmark": "docunet-ocr",
        "comparable": calibrated_engine and verify_references,
        "comparability": {
            "calibrated_tesseract_version": CALIBRATED_TESSERACT_VERSION,
            "references_verified": verify_references,
            "tesseract_version_matches": calibrated_engine,
        },
        "config": {
            "crop_pattern": crop_pattern if "uvdoc" in protocols else None,
            "prediction_pattern": pattern,
            "protocols": list(protocols),
            "reference_pattern": reference_pattern,
            "references_verified": verify_references,
            "tesseract_config": {protocol: PROTOCOL_CONFIG[protocol] for protocol in protocols},
            "workers": workers,
        },
        "created_utc": utc_now(),
        "environment": environment(),
        "evidence": report_evidence(record),
        "inputs": {
            "crops": crop_hashes,
            "crops_dir": str(crop_root) if "uvdoc" in protocols else None,
            "predictions": dict(sorted(prediction_hashes.items())),
            "predictions_dir": str(predictions_root),
            "references": {str(k): digest for k, digest in reference_hashes.items()},
            "references_dir": str(references_root),
        },
        "predictions_record": record,
        "protocol": {name: PROTOCOL_DESCRIPTION[name] for name in protocols}
        | {"alpha_composite": PROTOCOL_DESCRIPTION["alpha_composite"]},
        "rows": rows,
        "schema": REPORT_SCHEMA,
        "summary": summary,
        "tesseract": engine_metadata,
        "texts": stored_texts,
        "timing": {"wall_seconds": round(wall_seconds, 3)},
    }


def print_progress(job: OcrJob, done: int, total: int, seconds: float) -> None:
    print(
        f"[{done}/{total}] {job.protocol} {job.kind} {job.key} ({seconds:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
