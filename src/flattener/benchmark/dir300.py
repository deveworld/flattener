"""DIR300 benchmark (300 photos with flat references) with the calibrated MATLAB and OCR
protocols.

DIR300 (DocGeoNet, https://github.com/fh2019ustc/DocGeoNet) pairs each distorted photo
``dist/{k}.png`` with a flat reference ``gt/{k}.png``, k = 1 to 300. The photos show the
page on its background, so a model that rectifies the whole frame in one pass scores
well below methods that first locate or crop the page and run again; published DIR300
tables mix both kinds.

Prediction (:func:`predict_dir300`): every photo is rectified at its own size with
``flattener.dewarp.predict.rectify_batch`` into ``{k}.png``; ``predictions.json``
records the checks, the quantization and the photo hashes.

Metrics (:func:`evaluate_dir300`) follow DocGeoNet ``ssim_ld_eval_DIR300.m``: the
rectified image (there ``{k}_rec.png``) and the reference ``{k}.png`` are read,
converted with ``rgb2gray``, the reference is resized by ``b = sqrt(598400/H/W)`` (the
DocUNet metric area) and the rectified image to the reference size (MATLAB bicubic
``imresize``). MS-SSIM and LD are DocUNet's ``evalUnwarp.m``; AD is PaperEdge's
``evalAlignedUnwarp.m`` on the same SIFT flow. Every pair counts once in plain means
over the 300 pairs; nothing is rotated or skipped. The MATLAB emulation is the one
used for DocUNet, and the DocUNet calibration checks confirm it.

OCR (:func:`evaluate_dir300_ocr`) follows DocGeoNet ``OCR_eval_DIR300.py`` on the 90
images of ``ocr_img_DIR300.txt``: ``pytesseract.image_to_string`` defaults on the
reference and on the rectified image as stored, ED = Levenshtein distance of the raw
strings, CER = ED / len(reference string), plain means over the 90 images.

No DocGeoNet code is copied (its license is non-commercial); the file pairing, the OCR
list and the constants are cited as facts. The official download stores reference 248
as ``248.PNG``, so references are found case-insensitively.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import matlab, metrics, ocr
from .docunet import DEFAULT_WORKERS, format_name, run_tasks
from .docunet import PROTOCOL as DOCUNET_PROTOCOL
from .reporting import (
    environment,
    failure_counts,
    mean,
    prediction_record,
    prediction_status,
    report_evidence,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from .siftflow import SiftFlow, SiftFlowBuild

REPORT_SCHEMA = "flattener-benchmark-dir300-1.0"
OCR_REPORT_SCHEMA = "flattener-benchmark-dir300-ocr-1.0"
MEASUREMENT = "dir300_standard"
PREDICTION_INPUTS = "dir300_dist"
DOCUMENTS = tuple(range(1, 301))
# ocr_img_DIR300.txt of DocGeoNet (also the list in its OCR_eval_DIR300.py).
OCR_DOCUMENTS = (
    5, 7, 8, 10, 12, 27, 28, 29, 31, 36, 53, 55, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 85, 94, 96,
    103, 107, 108, 111, 115, 126, 128, 129, 130, 133, 135, 139, 140, 148, 149, 151, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 173, 174, 177,
    201, 202, 203, 205, 217, 218, 222, 223, 225, 227, 228, 237, 238, 239, 264, 265, 266,
    271, 273, 277, 278, 285, 286, 288, 291, 294, 295, 296, 298, 300,
)  # fmt: skip
DEFAULT_ROOT = Path("outputs/benchmarks/dir300")
DEFAULT_REFERENCES_DIR = DEFAULT_ROOT / "gt300"
DEFAULT_INPUTS_DIR = DEFAULT_ROOT / "dist300"
DEFAULT_PATTERN = "{k}.png"
# set_sha256 of the official gt/ and dist/ files (DocGeoNet Google Drive folders).
REFERENCE_SET_SHA256 = "0543020ec51fcbcd891dc2e4efb992e41d329ad853ee57fcd234e8962976f578"
INPUT_SET_SHA256 = "94fe31b1eef46243850b9e5cd55e3919af165cfc7e5784cfb67183a62d92d26f"
# The largest photos are 3472x4624 (16,054,528 pixels), above RenderConfig's default
# 16,000,000-pixel limits.
MAX_PHOTO_PIXELS = 20_000_000

PUBLISHED = {
    "uvdoc": {"ms_ssim": 0.622, "ld": 7.73, "ad": 0.218, "ed": 605.24, "cer": 0.260},
    "uvdoc_docres": {"ms_ssim": 0.676, "ld": 4.83, "ad": 0.165},
    "docgeonet": {"ms_ssim": 0.6380, "ld": 6.40, "ed": 664.96, "cer": 0.2189},
}
PUBLISHED_SOURCES = {
    "uvdoc": "TADoc, https://arxiv.org/abs/2508.06988 (UVDoc without preprocessing)",
    "uvdoc_docres": "Q-DRFormer, https://doi.org/10.1145/3845993 (UVDoc after DocRes "
    "page localization)",
    "docgeonet": "https://github.com/fh2019ustc/DocGeoNet README (no AD)",
}

PROTOCOL = {
    "source": "https://github.com/fh2019ustc/DocGeoNet/blob/main/ssim_ld_eval_DIR300.m",
    "pairs": "rectified {k} with gt/{k}.png, k = 1..300 (dist/{k}.png is the input)",
    "reference_scale": "sqrt(598400 / H / W) (divisions in this order)",
    "aggregation": "plain mean over the 300 pairs",
    # evalUnwarp.m (MS-SSIM, LD) and evalAlignedUnwarp.m (AD) as in the DocUNet report.
    **{
        key: DOCUNET_PROTOCOL[key]
        for key in (
            "ad",
            "ad_solvers",
            "alpha_composite_for_metrics",
            "flow_direction",
            "gray_before_resize",
            "metric_area",
            "ms_ssim",
            "ms_ssim_weights",
            "resize",
            "siftflow",
        )
    },
}
OCR_PROTOCOL = {
    "source": "https://github.com/fh2019ustc/DocGeoNet/blob/main/OCR_eval_DIR300.py",
    "documents": list(OCR_DOCUMENTS),
    "tesseract_args": "tesseract IN OUT txt (pytesseract.image_to_string defaults)",
    "prediction_resize": None,
    "ed": "Levenshtein distance between raw OCR strings",
    "cer": "ED / len(reference OCR string)",
    "aggregation": "plain mean over the 90 images",
    "alpha_composite": "white (pytesseract prepare)",
}


def set_sha256(hashes: dict[int, str]) -> str:
    """One digest of a DIR300 folder: sha256 of ``"{k} {sha256}\\n"`` lines in order."""
    return sha256_bytes("".join(f"{k} {hashes[k]}\n" for k in sorted(hashes)).encode())


def image_path(directory: str | os.PathLike[str], document: int) -> Path:
    """``{document}.png`` in ``directory``, matched case-insensitively (``248.PNG``)."""
    directory = Path(directory)
    name = f"{document}.png"
    if (directory / name).is_file():
        return directory / name
    if directory.is_dir():
        found = [
            path for path in directory.iterdir() if path.name.lower() == name and path.is_file()
        ]
        if len(found) == 1:
            return found[0]
    raise FileNotFoundError(f"missing DIR300 image {directory / name}")


def validate_documents(documents: Sequence[int], allowed: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(sorted(set(documents)))
    if not selected:
        raise ValueError("at least one document is required")
    unknown = [k for k in selected if k not in allowed]
    if unknown:
        raise ValueError(f"not DIR300 documents of this protocol: {unknown[:5]}")
    return selected


def hash_folder(
    directory: str | os.PathLike[str],
    documents: Sequence[int],
    expected_set_sha256: str,
    *,
    verify: bool = True,
) -> dict[int, str]:
    """SHA-256 of ``{k}.png`` for ``documents``; with ``verify`` all 300 files are hashed
    and must match ``expected_set_sha256`` (the official download)."""
    hashed = DOCUMENTS if verify else tuple(documents)
    hashes = {k: sha256_file(image_path(directory, k)) for k in hashed}
    if verify and set_sha256(hashes) != expected_set_sha256:
        raise ValueError(
            f"{directory} differs from the official DIR300 files "
            f"(set sha256 {set_sha256(hashes)}); pass verify=False to accept it"
        )
    return {k: hashes[k] for k in documents}


# --------------------------------------------------------------------------- predict


def predict_dir300(
    dewarper: Any,
    output_dir: str | os.PathLike[str],
    *,
    inputs_dir: str | os.PathLike[str] = DEFAULT_INPUTS_DIR,
    documents: Sequence[int] = DOCUMENTS,
    verify_inputs: bool = True,
    overwrite: bool = False,
    force_render: bool = False,
    quantize: str = "round",
    rerun: int = 0,
    deskew: bool = False,
    perspective: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Rectify ``dist/{k}.png`` into ``output_dir/{k}.png`` at the photo's own size.

    ``dewarper`` comes from ``flattener.dewarp.predict.load_dewarper``. What is kept,
    refused and recorded is ``rectify_batch``'s, with the default ``RenderConfig``
    except for input and output limits of ``MAX_PHOTO_PIXELS``. The record also names
    the input folder, the photo hashes and whether they matched the official photos.
    ``rerun`` and ``deskew`` add crop passes, ``perspective`` makes them perspective passes
    (``flattener.dewarp.predict.rerun_grid``). ``workers`` photos are rectified at once
    (``rectify_batch``), with the same images as one at a time.
    """
    from ..contracts import RenderConfig
    from ..dewarp.predict import BatchInput, load_rgb, rectify_batch

    documents = validate_documents(documents, DOCUMENTS)
    inputs_root = Path(inputs_dir).expanduser().resolve()
    hashes = hash_folder(inputs_root, documents, INPUT_SET_SHA256, verify=verify_inputs)
    paths = {k: image_path(inputs_root, k) for k in documents}
    inputs = [
        BatchInput(f"{k}.png", paths[k].name, lambda path=paths[k]: load_rgb(path))
        for k in documents
    ]
    return rectify_batch(
        dewarper,
        inputs,
        output_dir,
        overwrite=overwrite,
        force_render=force_render,
        render_config=RenderConfig(
            max_input_pixels=MAX_PHOTO_PIXELS, max_output_pixels=MAX_PHOTO_PIXELS
        ),
        fields={
            "input_dir": str(inputs_root),
            "inputs": PREDICTION_INPUTS,
            "input_files_sha256": {str(k): hashes[k] for k in documents},
            "inputs_verified": verify_inputs,
        },
        quantize=quantize,
        rerun=rerun,
        deskew=deskew,
        perspective=perspective,
        snap=snap,
        rect=rect,
        refine=refine,
        workers=workers,
    )


# ----------------------------------------------------------------------------- score


def prepare_reference(reference: np.ndarray) -> metrics.PreparedReference:
    """``rgb2gray`` then ``imresize(ref, sqrt(598400/H/W))`` as ``ssim_ld_eval_DIR300.m``.

    :func:`flattener.benchmark.metrics.prepare_reference` computes the scale as
    ``sqrt(598400/(H*W))``; the order of the divisions can change its last bit (40 of
    the 300 references), never the resized size.
    """
    gray = matlab.rgb2gray(reference)
    scale = math.sqrt(metrics.METRIC_AREA / gray.shape[0] / gray.shape[1])
    gray = matlab.imresize(gray, scale=scale)
    gradient = matlab.imgradient(gray)
    peak = float(gradient.max())
    return metrics.PreparedReference(
        image=gray,
        gradient=gradient / peak if peak > 0 else None,
        source_shape=(int(reference.shape[0]), int(reference.shape[1])),
        resizes=1,
    )


@dataclass(frozen=True)
class ImageTask:
    document: int
    reference_path: str
    prediction_path: str
    ad_solver: str


def score_image(task: ImageTask, siftflow: SiftFlow | None) -> list[dict[str, Any]]:
    """Score one rectified image against its reference (one pair of the MATLAB loop)."""
    started = time.perf_counter()
    reference_bytes = Path(task.reference_path).read_bytes()
    prediction_bytes = Path(task.prediction_path).read_bytes()
    score = metrics.score_prepared(
        metrics.read_image(prediction_bytes),
        prepare_reference(metrics.read_image(reference_bytes)),
        siftflow=siftflow,
        ad_solver=task.ad_solver,
    )
    row = score.to_dict()
    row.update(
        id=str(task.document),
        k=task.document,
        prediction_sha256=sha256_bytes(prediction_bytes),
        reference_file=Path(task.reference_path).name,
        reference_sha256=sha256_bytes(reference_bytes),
        seconds=round(time.perf_counter() - started, 3),
    )
    return [row]


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Plain means over the rows and the prediction failure counts."""
    result: dict[str, Any] = {"n": len(rows)}
    for name in ("ms_ssim", "ld", "ad", "ad_exact_lstsq", "ad_lsqr_4iter"):
        values = [row[name] for row in rows]
        result[name] = None if any(value is None for value in values) else mean(values)
    result.update(failure_counts([row.get("prediction_status") for row in rows]))
    return result


def _check_workers(workers: int) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")


def _predictions(
    predictions_dir: str | os.PathLike[str], pattern: str, documents: Sequence[int]
) -> tuple[Path, dict[int, str]]:
    root = Path(predictions_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"predictions directory not found: {root}")
    names = {k: format_name(pattern, fields=("k",), k=k) for k in documents}
    missing = [name for name in names.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} predictions missing in {root}, first: {missing[:5]}"
        )
    return root, names


def evaluate_dir300(
    predictions_dir: str | os.PathLike[str],
    *,
    build: SiftFlowBuild | None,
    pattern: str = DEFAULT_PATTERN,
    ad_solver: str = metrics.DEFAULT_AD_SOLVER,
    references_dir: str | os.PathLike[str] = DEFAULT_REFERENCES_DIR,
    verify_references: bool = True,
    documents: Sequence[int] = DOCUMENTS,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Score ``predictions_dir/{pattern}`` against the references; return the report.

    ``build`` ``None`` computes MS-SSIM only. The report is ``comparable`` only with
    SIFT flow, verified references and all 300 documents.
    """
    if ad_solver not in metrics.AD_SOLVERS:
        raise ValueError(f"ad_solver must be one of {metrics.AD_SOLVERS}, got {ad_solver!r}")
    _check_workers(workers)
    documents = validate_documents(documents, DOCUMENTS)
    references_root = Path(references_dir).expanduser().resolve()
    references = hash_folder(
        references_root, documents, REFERENCE_SET_SHA256, verify=verify_references
    )
    predictions_root, names = _predictions(predictions_dir, pattern, documents)
    record = prediction_record(predictions_root)
    tasks = [
        ImageTask(
            document=k,
            reference_path=str(image_path(references_root, k)),
            prediction_path=str(predictions_root / names[k]),
            ad_solver=ad_solver,
        )
        for k in documents
    ]
    started = time.perf_counter()
    rows = run_tasks(
        tasks, build, workers=workers, progress=progress, score=score_image, total=len(tasks)
    )
    wall_seconds = time.perf_counter() - started
    rows.sort(key=lambda row: row["k"])
    for row in rows:
        if row["reference_sha256"] != references[row["k"]]:
            raise RuntimeError(f"reference {row['k']} changed during evaluation")
        row["prediction_status"] = prediction_status(record, names[row["k"]])
    complete = documents == DOCUMENTS
    return {
        "benchmark": "dir300",
        "comparable": build is not None and verify_references and complete,
        "config": {
            "ad_solver": ad_solver,
            "documents": "all" if complete else list(documents),
            "prediction_pattern": pattern,
            "references_verified": verify_references,
            "workers": workers,
        },
        "created_utc": utc_now(),
        "environment": environment(),
        "evidence": report_evidence(record, MEASUREMENT),
        "inputs": {
            "predictions": {row["id"]: row["prediction_sha256"] for row in rows},
            "predictions_dir": str(predictions_root),
            "references": {str(k): digest for k, digest in references.items()},
            "references_dir": str(references_root),
            "references_set_sha256": set_sha256(references) if complete else None,
        },
        "predictions_record": record,
        "protocol": PROTOCOL,
        "published": {"values": PUBLISHED, "sources": PUBLISHED_SOURCES},
        "rows": rows,
        "schema": REPORT_SCHEMA,
        "siftflow": None if build is None else build.metadata(),
        "summary": summarize(rows),
        "timing": {
            "image_seconds_total": round(sum(row["seconds"] for row in rows), 3),
            "wall_seconds": round(wall_seconds, 3),
        },
    }


# ------------------------------------------------------------------------------- OCR


def score_ocr(
    references: dict[int, str], predictions: dict[int, str], documents: Sequence[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``OCR_eval_DIR300.py`` scoring: raw-string ED, ED / len(reference), plain means."""
    rows = []
    for k in documents:
        reference, prediction = references[k], predictions[k]
        distance = ocr.levenshtein(reference, prediction)
        rows.append(
            {
                "cer": distance / len(reference) if reference else None,
                "ed": distance,
                "id": str(k),
                "k": k,
                "prediction_chars": len(prediction),
                "reference_chars": len(reference),
            }
        )
    cers = [row["cer"] for row in rows if row["cer"] is not None]
    summary = {
        "cer": mean(cers) if cers else None,
        "ed": mean([float(row["ed"]) for row in rows]),
        "n": len(rows),
        "n_nan_cer": len(rows) - len(cers),
    }
    return summary, rows


def evaluate_dir300_ocr(
    predictions_dir: str | os.PathLike[str],
    *,
    engine: ocr.OcrEngine | None = None,
    pattern: str = DEFAULT_PATTERN,
    references_dir: str | os.PathLike[str] = DEFAULT_REFERENCES_DIR,
    verify_references: bool = True,
    documents: Sequence[int] = OCR_DOCUMENTS,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[ocr.OcrJob, int, int, float], None] | None = None,
) -> dict[str, Any]:
    """OCR the DIR300 OCR subset of ``predictions_dir`` and return the report.

    The report is ``comparable`` only with a Tesseract build that passed the OCR score
    check (``ocr.calibrated_build``), verified references and all 90 documents.
    """
    _check_workers(workers)
    documents = validate_documents(documents, OCR_DOCUMENTS)
    engine = ocr.Tesseract() if engine is None else engine
    references_root = Path(references_dir).expanduser().resolve()
    references = hash_folder(
        references_root, documents, REFERENCE_SET_SHA256, verify=verify_references
    )
    predictions_root, names = _predictions(predictions_dir, pattern, documents)
    jobs = []
    for k in documents:
        jobs.append(ocr.OcrJob("doctr", "reference", k, 0, str(image_path(references_root, k))))
        jobs.append(ocr.OcrJob("doctr", "prediction", k, 0, str(predictions_root / names[k])))
    prediction_hashes = {str(k): sha256_file(predictions_root / names[k]) for k in documents}
    engine_metadata = engine.metadata()
    started = time.perf_counter()
    texts = run_jobs_by_document(jobs, engine, workers=workers, progress=progress)
    wall_seconds = time.perf_counter() - started
    record = prediction_record(predictions_root)
    summary, rows = score_ocr(texts["reference"], texts["prediction"], documents)
    for row in rows:
        row["prediction_status"] = prediction_status(record, names[row["k"]])
    summary.update(failure_counts([row["prediction_status"] for row in rows]))
    build = ocr.calibrated_build(engine_metadata)
    complete = documents == OCR_DOCUMENTS
    return {
        "benchmark": "dir300-ocr",
        "comparable": build is not None and verify_references and complete,
        "comparability": {
            "calibrated_builds": sorted(ocr.CALIBRATED_TESSERACT_BUILDS),
            "references_verified": verify_references,
            "tesseract_build": build,
            "all_documents": complete,
        },
        "config": {
            "documents": "all" if complete else list(documents),
            "prediction_pattern": pattern,
            "references_verified": verify_references,
            "tesseract_config": ocr.PROTOCOL_CONFIG["doctr"],
            "workers": workers,
        },
        "created_utc": utc_now(),
        "environment": environment(),
        "evidence": report_evidence(record, MEASUREMENT),
        "inputs": {
            "predictions": prediction_hashes,
            "predictions_dir": str(predictions_root),
            "references": {str(k): digest for k, digest in references.items()},
            "references_dir": str(references_root),
        },
        "predictions_record": record,
        "protocol": OCR_PROTOCOL,
        "published": {"values": PUBLISHED, "sources": PUBLISHED_SOURCES},
        "rows": rows,
        "schema": OCR_REPORT_SCHEMA,
        "summary": summary,
        "tesseract": engine_metadata,
        "texts": {
            kind: {str(k): texts[kind][k] for k in documents}
            for kind in ("prediction", "reference")
        },
        "timing": {"wall_seconds": round(wall_seconds, 3)},
    }


def run_jobs_by_document(
    jobs: Sequence[ocr.OcrJob],
    engine: ocr.OcrEngine,
    *,
    workers: int,
    progress: Callable[[ocr.OcrJob, int, int, float], None] | None = None,
) -> dict[str, dict[int, str]]:
    """``ocr.run_jobs`` texts as ``{"reference": {k: text}, "prediction": {k: text}}``."""
    texts = ocr.run_jobs(jobs, engine, workers=workers, progress=progress)
    result: dict[str, dict[int, str]] = {"reference": {}, "prediction": {}}
    for job in jobs:
        result[job.kind][job.document] = texts[(job.protocol, job.kind, job.key)]
    return result


# ---------------------------------------------------------------------------- output


def print_progress(row: dict[str, Any], done: int, total: int) -> None:
    parts = [f"ms_ssim={row['ms_ssim']:.5f}"]
    if row["ld"] is not None:
        parts += [f"ld={row['ld']:.3f}", f"ad={row['ad']:.4f}"]
    print(
        f"[{done}/{total}] {row['id']} {' '.join(parts)} ({row['seconds']:.1f}s)",
        file=sys.stderr,
        flush=True,
    )


def format_comparison(summary: dict[str, Any], method: str = "uvdoc") -> str:
    """The summary's metrics next to a published row of ``PUBLISHED`` and the difference."""
    published = PUBLISHED[method]
    parts = []
    for name in ("ms_ssim", "ld", "ad", "ed", "cer"):
        value = summary.get(name)
        if value is None:
            continue
        text = f"{name}={value:.4f}"
        if name in published:
            text += f" ({method} {published[name]:g}, diff {value - published[name]:+.4f})"
        parts.append(text)
    return " ".join(parts)
