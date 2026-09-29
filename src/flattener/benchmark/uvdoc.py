"""UVDoc benchmark (50 real photos) with the protocol of the UVDoc release.

Ported from UVDoc (https://github.com/tanguymagne/UVDoc at commit 4c9b82b, MIT,
Copyright (c) 2023 Tanguy MAGNE): ``uvdocBenchmark_pred.py`` (prediction),
``uvdocBenchmark_eval.py`` and ``eval/eval_code/evalScriptUVDoc.m`` (scoring).

Data: ``UVDoc_benchmark.zip`` from ``ARCHIVE_URL``, extracted by
:func:`prepare_benchmark` into ROOT (default ``outputs/benchmarks/uvdoc-bench``) with a
``manifest.json`` of the archive and file hashes. Samples are ``00000`` to ``00049``.

Prediction (:func:`predict_uvdoc`, as ``uvdocBenchmark_pred.py``):

* The model sees the photo ``img/{id}.png``, resized to its input size.
* The predicted grid is densified at the size of ``warped_textures/{id}.png`` (the
  flat texture warped like the photo, without lighting or background) and applied to
  it with bilinear sampling and zero padding. The result is stored as
  ``floor(x * 255)`` in ``uwp_texture/{id}.png``.
* ``uwp_img/{id}.png`` applies the same grid to the photo at the photo's size.
* UVDoc weights (``grid_convention="uvdoc"``) run upstream ``bilinear_unwarping``
  (``align_corners=True``) unchanged. flattener checkpoints use their own grid
  convention with the same zero padding and truncation.
* An inset (``flattener.dewarp.predict.resolve_inset``, default none) moves the
  outline inward on the dense map before sampling, as ``flattener.dewarp.predict``
  does; it is recorded in ``predictions.json`` and its evidence block.
* Crop passes (``rerun``, optionally ``deskew``; default none) refine the grid on crops
  of the photo as ``flattener.dewarp.predict.rerun_grid`` does. The composed grid holds
  flattener values of the photo, which address the texture's pixels by the same
  normalized positions; both outputs then sample with align_corners=False, also for
  UVDoc weights. The settings and each image's passes are recorded.

Scoring (:func:`evaluate_uvdoc`, as ``evalScriptUVDoc.m`` with the default
``--pred-type uwp_texture``): each prediction against the flat texture
``texture_sample/{id}.png`` with the calibrated DocUNet MATLAB emulation
(:func:`flattener.benchmark.metrics.score_pair`): gray, texture resized once to
598,400 pixels, prediction resized to it, MS-SSIM, SIFT flow, AD (PaperEdge) and LD.
The means are over all 50 samples, as the README erratum values are (the paper's
first table used half of the benchmark).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import metrics
from .docunet import DEFAULT_WORKERS, run_tasks
from .reporting import (
    PREDICTIONS_SCHEMA,
    environment,
    failure_counts,
    mean,
    prediction_record,
    prediction_status,
    report_evidence,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .siftflow import SiftFlow, SiftFlowBuild

REPORT_SCHEMA = "flattener-benchmark-uvdoc-1.0"
MANIFEST_SCHEMA = "flattener-benchmark-uvdoc-manifest-1.0"
DEFAULT_ROOT = Path("outputs/benchmarks/uvdoc-bench")
SAMPLES = tuple(f"{index:05d}" for index in range(50))
PRED_TYPES = ("uwp_texture", "uwp_img")
DEFAULT_PRED_TYPE = "uwp_texture"
UPSTREAM = "https://github.com/tanguymagne/UVDoc/tree/4c9b82b537057aff2526e6dd118a847cdd072e82"

ARCHIVE_URL = "https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip"
ARCHIVE_SHA256 = "8bf91a3e0d9bfa3655a8bcb38b74c1a8c02a9f1ba5ce34cfc51e5ae0660067ac"
# set_sha256 of the three folders the protocol reads, from ARCHIVE_SHA256.
SET_SHA256 = {
    "img": "8c48ce4c5917c3572dff1a9461b344363d877656a3981ff9c4f6c556c3398342",
    "warped_textures": "3acdf094cb2f872c831ed0b1bf9c15ec4830ce807e747bb47b2afbc550a0a495",
    "texture_sample": "940c6a8e3e31d33987e69750a56ab79fcb51dc9ab46352b6b0f88dcef42d3905",
}
LICENSE = {
    "declared": "MIT (github.com/tanguymagne/UVDoc); the archive has no license file",
    "rights_holder": "Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung (ETH Zurich IGL)",
    "note": "evaluation only; kept local, never used for training",
}

# Published means over all 50 samples. UVDoc and the related works: README erratum
# (MS-SSIM, AD); LD for UVDoc and AADD: AADD Table S1.
PUBLISHED = {
    "uvdoc": {"ms_ssim": 0.785, "ad": 0.119, "ld": 6.810},
    "aadd": {"ms_ssim": 0.800, "ad": 0.098, "ld": 5.834},
    "docgeonet": {"ms_ssim": 0.706, "ad": 0.168},
    "doctr": {"ms_ssim": 0.697, "ad": 0.160},
    "rdgr": {"ms_ssim": 0.610, "ad": 0.280},
    "dewarpnet": {"ms_ssim": 0.589, "ad": 0.193},
    "ddcp": {"ms_ssim": 0.585, "ad": 0.290},
}
PUBLISHED_SOURCES = {
    "readme_erratum": "https://github.com/tanguymagne/UVDoc#exclamation-erratum",
    "aadd_table_s1": "https://arxiv.org/html/2507.15000",
}

PROTOCOL = {
    "upstream": UPSTREAM,
    "samples": f"{SAMPLES[0]}-{SAMPLES[-1]} ({len(SAMPLES)}), mean over all",
    "model_input": "img/{id}.png resized to the model input (bilinear, no antialias)",
    "uwp_texture": "grid applied to warped_textures/{id}.png at its size (the default)",
    "uwp_img": "grid applied to img/{id}.png at its size",
    "sampling": "bilinear, zero padding; UVDoc grids: interpolate + grid_sample with "
    "align_corners=True; flattener grids: dense map + grid_sample align_corners=False",
    "quantize": "floor(x * 255) ((x * 255).astype(uint8))",
    "reference": "texture_sample/{id}.png",
    "metrics": "DocUNet MATLAB protocol (evalScriptUVDoc.m): rgb2gray, reference "
    "resized once to 598400 px (bicubic), prediction resized to it, MS-SSIM, "
    "siftFlow(reference, prediction), AD (evalAlignedUnwarp.m) and LD",
}


def sample_name(sample: str) -> str:
    return f"{sample}.png"


def set_sha256(hashes: dict[str, str]) -> str:
    """One digest of a folder's 50 sample files: sha256 of ``"{id}.png {sha256}\\n"``
    lines in sample order."""
    lines = "".join(f"{sample_name(s)} {hashes[sample_name(s)]}\n" for s in SAMPLES)
    return sha256_bytes(lines.encode())


def hash_folder(root: Path, folder: str, *, verify: bool = True) -> dict[str, str]:
    """SHA-256 of the 50 sample files of ``root/folder``; with ``verify`` the folder
    must match the published archive (``SET_SHA256``)."""
    hashes = {}
    missing = []
    for sample in SAMPLES:
        path = root / folder / sample_name(sample)
        if not path.is_file():
            missing.append(str(path))
            continue
        hashes[sample_name(sample)] = sha256_file(path)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} UVDoc benchmark files missing, first: {missing[:3]}"
        )
    if verify and set_sha256(hashes) != SET_SHA256[folder]:
        raise ValueError(
            f"{root / folder} differs from the published UVDoc benchmark "
            f"(set sha256 {set_sha256(hashes)}); pass verify=False to accept it"
        )
    return hashes


def _member_path(name: str, prefix: str) -> PurePosixPath | None:
    """Relative path of an archive member below ``prefix``; None for folders."""
    if name.endswith("/"):
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe archive member {name!r}")
    parts = path.parts[1:] if prefix and path.parts[0] == prefix else path.parts
    if not parts:
        raise ValueError(f"unsafe archive member {name!r}")
    return PurePosixPath(*parts)


def prepare_benchmark(
    archive: str | os.PathLike[str],
    root: str | os.PathLike[str] = DEFAULT_ROOT,
    *,
    url: str = ARCHIVE_URL,
    verify: bool = True,
    retrieved: str | None = None,
) -> dict[str, Any]:
    """Extract ``UVDoc_benchmark.zip`` into ``root`` and write ``root/manifest.json``.

    With ``verify`` the archive must have ``ARCHIVE_SHA256`` and the extracted
    ``img``, ``warped_textures`` and ``texture_sample`` folders ``SET_SHA256``. The
    top-level ``UVDoc_benchmark/`` folder of the archive is dropped. Every member is
    CRC-checked while it is read and written through a temporary file.
    """
    archive, root = Path(archive), Path(root)
    archive_sha256 = sha256_file(archive)
    if verify and archive_sha256 != ARCHIVE_SHA256:
        raise ValueError(f"{archive} has sha256 {archive_sha256}, expected {ARCHIVE_SHA256}")
    files = []
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        tops = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        prefix = next(iter(tops)) if len(tops) == 1 and all("/" in n for n in names) else ""
        for info in bundle.infolist():
            relative = _member_path(info.filename, prefix)
            if relative is None:
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            handle, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
            try:
                with os.fdopen(handle, "wb") as output, bundle.open(info) as source:
                    while block := source.read(1 << 20):
                        digest.update(block)
                        output.write(block)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            files.append(
                {"path": relative.as_posix(), "bytes": info.file_size, "sha256": digest.hexdigest()}
            )
    files.sort(key=lambda item: item["path"])
    sets = {}
    for folder in SET_SHA256:
        hashes = hash_folder(root, folder, verify=verify)
        sets[folder] = {"count": len(hashes), "set_sha256": set_sha256(hashes)}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "benchmark": "uvdoc",
        "created_utc": utc_now(),
        "source": {
            "url": url,
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "sha256_expected": ARCHIVE_SHA256,
            "retrieved": retrieved,
            "transport": "HTTPS from the publisher (the hash verifies the retrieved bytes)",
        },
        "license": LICENSE,
        "citation": "Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung. UVDoc: Neural "
        "Grid-based Document Unwarping. SIGGRAPH Asia 2023.",
        "samples": list(SAMPLES),
        "sets": sets,
        "files": files,
    }
    write_json_atomic(root / "manifest.json", manifest, overwrite=True)
    return manifest


# --------------------------------------------------------------------------- predict


def unwarp(
    image: NDArray[np.uint8],
    grid: NDArray[np.floating],
    convention: str,
    inset: Any = None,
    *,
    flattener_values: bool = False,
) -> NDArray[np.uint8]:
    """Apply a model grid (Gh, Gw, 2) to ``image`` at its own size, as upstream
    ``bilinear_unwarping`` followed by ``(x * 255).astype(uint8)``.

    ``convention`` ``"uvdoc"``: the upstream code path (align_corners=True
    interpolation and sampling). ``"flattener"``: the dense flattener map sampled
    with align_corners=False. Both pad with zeros outside the source. With an
    ``inset`` or ``flattener_values`` (a rerun's composed grid of flattener values,
    ``flattener.dewarp.predict.rerun_grid``) both sample
    ``flattener.dewarp.predict.dense_backward_map`` of the grid with align_corners=False.
    """
    import torch
    from torch.nn import functional as F

    from ..dewarp.predict import NO_INSET, dense_backward_map, resolve_inset

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB uint8 HxWx3")
    height, width = image.shape[:2]
    # Upstream divides the uint8 array by 255.0 in float64, then casts to float32.
    source = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float().unsqueeze(0)
    if convention not in ("uvdoc", "flattener"):
        raise ValueError(f"unknown grid convention {convention!r}")
    upstream = resolve_inset(inset) == NO_INSET and not flattener_values
    with torch.inference_mode():
        if convention == "uvdoc" and upstream:
            points = torch.from_numpy(np.ascontiguousarray(grid.transpose(2, 0, 1)))
            points = points.float().unsqueeze(0)
            dense = F.interpolate(
                points, size=(height, width), mode="bilinear", align_corners=True
            ).permute(0, 2, 3, 1)
            sampled = F.grid_sample(source, dense, align_corners=True)
        else:
            size = (height, width)
            dense = dense_backward_map(
                grid, convention, size, size, inset, flattener_values=flattener_values
            )
            sampled = F.grid_sample(
                source,
                torch.from_numpy(dense).unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
    return (sampled[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _write_png(path: Path, pixels: NDArray[np.uint8]) -> str:
    from PIL import Image

    from ..dewarp.checkpoint import atomic_write

    atomic_write(path, lambda stream: Image.fromarray(pixels).save(stream, "PNG"))
    return sha256_file(path)


def _check_status(
    texture: NDArray[np.uint8],
    grid: NDArray,
    convention: str,
    inset: Any = None,
    *,
    flattener_values: bool = False,
) -> dict[str, Any]:
    """``flattener.render``'s checks of the dense map on the texture (recorded only)."""
    from ..contracts import RenderConfig
    from ..dewarp.predict import dense_backward_map
    from ..renderer import render

    size = texture.shape[:2]
    result = render(
        texture,
        dense_backward_map(grid, convention, size, size, inset, flattener_values=flattener_values),
        config=RenderConfig(on_error="return"),
    )
    stats = result.metadata.get("validation")
    keys = ("out_of_bounds_fraction", "failed_cells", "checked_cells", "min_jacobian_ratio")
    return {
        "status": str(result.status),
        "reason": result.reason_code,
        "validation": {key: stats.get(key) for key in keys} if isinstance(stats, dict) else None,
    }


def predict_uvdoc(
    dewarper: Any,
    output_dir: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] = DEFAULT_ROOT,
    verify_inputs: bool = True,
    overwrite: bool = False,
    log: Callable[[str], None] | None = None,
    inset: Any = None,
    rerun: int = 0,
    deskew: bool = False,
    perspective: bool = False,
    snap: bool = False,
    rect: bool = False,
    refine: bool = False,
) -> dict[str, Any]:
    """Rectify the 50 samples into ``output_dir/{uwp_texture,uwp_img}/{id}.png``.

    ``dewarper`` comes from ``flattener.dewarp.predict.load_dewarper``. Every image is
    written (published methods are scored that way); ``flattener.render``'s checks of
    the map are recorded per image in ``output_dir/predictions.json`` (one item per
    output file, ``inputs: "uvdoc_benchmark"``), and a failed check sets ``forced``.
    Without ``overwrite`` an existing ``predictions.json`` stops the run. ``inset``
    (``flattener.dewarp.predict.resolve_inset``) moves the outline inward on the dense
    map of both outputs and of the checks; ``predictions.json`` records it. ``rerun``
    crop passes (with ``deskew``; perspective passes with ``perspective``) refine the grid
    on the photo (``flattener.dewarp.predict.rerun_grid``) before both outputs and the checks;
    ``predictions.json`` records the settings and each item its passes (``crops``).
    """
    from ..dewarp.predict import (
        inset_record,
        load_rgb,
        predict_grid,
        prediction_evidence,
        rerun_grid,
        rerun_record,
    )

    root, output = Path(root).expanduser().resolve(), Path(output_dir)
    record_path = output / "predictions.json"
    if record_path.exists() and not overwrite:
        raise FileExistsError(f"{record_path} exists; pass overwrite to predict again")
    inset_info = inset_record(inset)  # refuses an invalid inset before any work
    rerun_info = rerun_record(
        rerun, deskew, perspective, snap, rect, refine
    )  # and invalid settings
    photos = hash_folder(root, "img", verify=verify_inputs)
    textures = hash_folder(root, "warped_textures", verify=verify_inputs)
    convention = dewarper.config.grid_convention
    record: dict[str, Any] = {
        "schema": PREDICTIONS_SCHEMA,
        "created_utc": utc_now(),
        "model": dewarper.info,
        "evidence": prediction_evidence(
            dewarper,
            "truncate",
            inset,
            rerun,
            deskew,
            perspective=perspective,
            snap=snap,
            rect=rect,
        ),
        "inputs": "uvdoc_benchmark",
        "root": str(root),
        "inputs_verified": verify_inputs,
        "input_sets": {"img": set_sha256(photos), "warped_textures": set_sha256(textures)},
        "force_render": True,
        "quantize": "truncate",
        "inset": inset_info,
        "rerun": rerun_info,
        "protocol": {key: PROTOCOL[key] for key in ("model_input", *PRED_TYPES, "sampling")},
        "complete": False,
        "counts": {},
        "items": [],
    }
    write_json_atomic(record_path, record, overwrite=True)
    for index, sample in enumerate(SAMPLES, 1):
        started = time.perf_counter()
        name = sample_name(sample)
        photo = load_rgb(root / "img" / name)
        texture = load_rgb(root / "warped_textures" / name)
        if rerun or snap or refine:
            grid, crops = rerun_grid(
                dewarper, photo, rerun, deskew, perspective, snap, rect, refine
            )
        else:
            grid, crops = predict_grid(dewarper, photo), []
        values = {"flattener_values": rerun > 0 or snap or refine}
        check = _check_status(texture, grid, convention, inset, **values)
        sources = {"uwp_texture": ("warped_textures", texture), "uwp_img": ("img", photo)}
        items = []
        for pred_type, (folder, source) in sources.items():
            target = output / pred_type / name
            digest = _write_png(target, unwarp(source, grid, convention, inset, **values))
            items.append(
                {
                    "input": f"img/{name}",
                    "input_sha256": photos[name],
                    "source": f"{folder}/{name}",
                    "source_sha256": (textures if folder == "warped_textures" else photos)[name],
                    "output": f"{pred_type}/{name}",
                    "output_sha256": digest,
                    "action": "written",
                    **check,
                    "forced": check["status"] != "success",
                    "size": list(photo.shape[:2]),
                    "output_size": list(source.shape[:2]),
                    "crops": crops,
                }
            )
        seconds = time.perf_counter() - started
        for item in items:
            item["seconds"] = seconds
        record["items"] += items
        record["counts"] = _counts(record["items"])
        write_json_atomic(record_path, record, overwrite=True)
        if log is not None:
            log(f"[{index}/{len(SAMPLES)}] {sample} {check['status']} ({seconds:.1f}s)")
    record["complete"] = True
    write_json_atomic(record_path, record, overwrite=True)
    return record


def _counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "items": len(items),
        "written": sum(item["action"] == "written" for item in items),
        "kept": 0,
        "failed": sum(item["status"] != "success" for item in items),
        "forced": sum(bool(item["forced"]) for item in items),
    }


# ----------------------------------------------------------------------------- score


@dataclass(frozen=True)
class SampleTask:
    sample: str
    reference_path: str
    prediction_path: str
    ad_solver: str


def score_sample(task: SampleTask, siftflow: SiftFlow | None) -> list[dict[str, Any]]:
    """Score one prediction against its flat texture (``evalScriptUVDoc.m`` body)."""
    started = time.perf_counter()
    reference_bytes = Path(task.reference_path).read_bytes()
    prediction_bytes = Path(task.prediction_path).read_bytes()
    score = metrics.score_pair(
        metrics.read_image(prediction_bytes),
        metrics.read_image(reference_bytes),
        siftflow=siftflow,
        ad_solver=task.ad_solver,
    )
    row = score.to_dict()
    row.update(
        id=task.sample,
        prediction_sha256=sha256_bytes(prediction_bytes),
        reference_sha256=sha256_bytes(reference_bytes),
        seconds=round(time.perf_counter() - started, 3),
    )
    return [row]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over the rows (all 50 samples) and the prediction failure counts."""
    result: dict[str, Any] = {"n": len(rows)}
    for name in ("ms_ssim", "ld", "ad", "ad_exact_lstsq", "ad_lsqr_4iter"):
        values = [row[name] for row in rows]
        result[name] = None if any(value is None for value in values) else mean(values)
    result.update(failure_counts([row.get("prediction_status") for row in rows]))
    return result


def evaluate_uvdoc(
    predictions_dir: str | os.PathLike[str],
    *,
    build: SiftFlowBuild | None,
    pred_type: str = DEFAULT_PRED_TYPE,
    root: str | os.PathLike[str] = DEFAULT_ROOT,
    ad_solver: str = metrics.DEFAULT_AD_SOLVER,
    verify_references: bool = True,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Score ``predictions_dir/{pred_type}/{id}.png`` against ``texture_sample``.

    ``predictions_dir`` is the folder :func:`predict_uvdoc` writes (or UVDoc's
    ``output_uvdoc`` / ``UVDocBenchmark_results``). Its ``predictions.json``, when
    present, supplies the model evidence and per-image check statuses. ``build``
    ``None`` computes MS-SSIM only and the report has ``comparable: false``.
    """
    if pred_type not in PRED_TYPES:
        raise ValueError(f"pred_type must be one of {PRED_TYPES}, got {pred_type!r}")
    if ad_solver not in metrics.AD_SOLVERS:
        raise ValueError(f"ad_solver must be one of {metrics.AD_SOLVERS}, got {ad_solver!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    predictions_root = Path(predictions_dir).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    if not predictions_root.is_dir():
        raise NotADirectoryError(f"predictions directory not found: {predictions_root}")
    references = hash_folder(root, "texture_sample", verify=verify_references)
    missing = [
        f"{pred_type}/{sample_name(s)}"
        for s in SAMPLES
        if not (predictions_root / pred_type / sample_name(s)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} predictions missing in {predictions_root}, first: {missing[:5]}"
        )
    record = prediction_record(predictions_root)
    tasks = [
        SampleTask(
            sample=sample,
            reference_path=str(root / "texture_sample" / sample_name(sample)),
            prediction_path=str(predictions_root / pred_type / sample_name(sample)),
            ad_solver=ad_solver,
        )
        for sample in SAMPLES
    ]
    started = time.perf_counter()
    rows = run_tasks(
        tasks, build, workers=workers, progress=progress, score=score_sample, total=len(tasks)
    )
    wall_seconds = time.perf_counter() - started
    rows.sort(key=lambda row: row["id"])
    for row in rows:
        if row["reference_sha256"] != references[sample_name(row["id"])]:
            raise RuntimeError(f"texture {row['id']} changed during evaluation")
        row["prediction_status"] = prediction_status(
            record, f"{pred_type}/{sample_name(row['id'])}"
        )
    return {
        "benchmark": "uvdoc",
        "comparable": build is not None and verify_references,
        "config": {
            "ad_solver": ad_solver,
            "pred_type": pred_type,
            "references_verified": verify_references,
            "workers": workers,
        },
        "created_utc": utc_now(),
        "environment": environment(),
        "evidence": report_evidence(record, "uvdoc_benchmark"),
        "inputs": {
            "predictions": {row["id"]: row["prediction_sha256"] for row in rows},
            "predictions_dir": str(predictions_root),
            "references_set_sha256": set_sha256(references),
            "root": str(root),
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
    """``ms_ssim``/``ad``/``ld`` of a summary next to a published row and the difference."""
    published = PUBLISHED[method]
    parts = []
    for name in ("ms_ssim", "ad", "ld"):
        value = summary.get(name)
        if value is None:
            continue
        text = f"{name}={value:.4f}"
        if name in published:
            text += f" ({method} {published[name]:.3f}, diff {value - published[name]:+.4f})"
        parts.append(text)
    return " ".join(parts)
