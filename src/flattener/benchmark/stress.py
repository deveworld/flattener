"""DocUNet stress variants: whole photos, quarter turns and small pages.

``python -m flattener.benchmark docunet-stress --weights W --output-dir OUT`` builds the
inputs of each variant from the DocUNet crops and original photos, rectifies them with
a ``flattener.dewarp`` model (an own checkpoint or the UVDoc weights), scores the outputs
against the upright scans with the calibrated evaluator and writes ``OUT/stress.json``.
Every output is rendered at the size of its upright crop.

Variants:

* ``crop``: the standard crop, as a reference scored in the same run.
* ``full_frame``: the whole original photo (``docunet-photo-{k}_{m}.jpg``) with its
  background.
* ``rot90`` and ``rot270``: the crop turned a quarter counter-clockwise and clockwise.
  The scans stay upright, so a model that returns a turned page scores low.
* ``small_page``: the crop at its own resolution on a larger plain canvas with shading
  and grain, covering a seeded 10-25% of the canvas area. A canvas above
  ``CANVAS_MAX_PIXELS`` is scaled down together with the crop. The crop keeps its own
  background, so the crop rectangle stays visible around the page.

Each variant lives in ``OUT/<variant>/``: ``predictions.json`` and ``{k}_{m}.png`` from
``flattener.dewarp.predict.rectify_batch`` and ``docunet.json`` from
``evaluate_docunet``. A rerun keeps predictions of the same weights and reuses a report
whose prediction hashes still match.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..evidence import evidence
from . import docunet
from .reporting import prediction_record, sha256_file, utc_now, write_json_atomic
from .siftflow import SiftFlowBuild

SCHEMA = "flattener-benchmark-docunet-stress-1"
VARIANTS = {
    "crop": "standard DocUNet crop (reference)",
    "full_frame": "whole original photo with background",
    "rot90": "crop turned 90 degrees counter-clockwise",
    "rot270": "crop turned 90 degrees clockwise",
    "small_page": "crop covering 10-25% of a plain textured canvas",
}
DEFAULT_CROP_DIR = Path("outputs/benchmarks/docunet/crop")
DEFAULT_PHOTO_DIR = Path("outputs/dataset/files")
CROP_PATTERN = "{k}_{m} copy.png"
PHOTO_PATTERN = "docunet-photo-{k}_{m}.jpg"
PAGE_FRACTION = (0.1, 0.25)
CANVAS_MAX_PIXELS = 12_000_000
_TURNS = {"rot90": Image.Transpose.ROTATE_90, "rot270": Image.Transpose.ROTATE_270}


def _rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def canvas(size: tuple[int, int], rng: np.random.Generator) -> NDArray[np.uint8]:
    """A plain surface of ``size`` (height, width): a grey tone with a slight tint, smooth
    shading shared by the channels and fine grain."""
    height, width = size
    tone = rng.uniform(70, 190) + rng.uniform(-12, 12, size=3)
    coarse = Image.fromarray(rng.uniform(-25, 25, size=(3, 4)).astype(np.float32))
    shading = np.asarray(coarse.resize((width, height), Image.BICUBIC))[..., None]
    values = shading + tone.astype(np.float32)
    values = values + 4 * rng.standard_normal(size=(height, width, 3), dtype=np.float32)
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def small_page(crop: Image.Image, rng: np.random.Generator) -> NDArray[np.uint8]:
    """``crop`` pasted on a canvas it covers ``PAGE_FRACTION`` of, at a seeded position."""
    fraction = rng.uniform(*PAGE_FRACTION)
    width, height = crop.size
    scale = min(1.0, math.sqrt(CANVAS_MAX_PIXELS * fraction / (width * height)))
    if scale < 1.0:
        crop = crop.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
        width, height = crop.size
    canvas_w, canvas_h = round(width / math.sqrt(fraction)), round(height / math.sqrt(fraction))
    result = canvas((canvas_h, canvas_w), rng)
    top = int(rng.integers(0, canvas_h - height + 1))
    left = int(rng.integers(0, canvas_w - width + 1))
    result[top : top + height, left : left + width] = np.asarray(crop)
    return result


def variant_image(
    variant: str, k: int, m: int, crop_dir: Path, photo_dir: Path, seed: int = 0
) -> NDArray[np.uint8]:
    """Input image of one DocUNet photo in ``variant`` (RGB uint8)."""
    if variant == "full_frame":
        return np.asarray(_rgb(photo_dir / PHOTO_PATTERN.format(k=k, m=m)))
    crop = _rgb(crop_dir / CROP_PATTERN.format(k=k, m=m))
    if variant in _TURNS:
        return np.asarray(crop.transpose(_TURNS[variant]))
    if variant == "small_page":
        rng = np.random.default_rng([seed, zlib.crc32(variant.encode()), k, m])
        return small_page(crop, rng)
    if variant == "crop":
        return np.asarray(crop)
    raise ValueError(f"variant must be one of {tuple(VARIANTS)}")


def variant_inputs(variant: str, crop_dir: Path, photo_dir: Path, seed: int = 0) -> list[Any]:
    """Batch inputs (``flattener.dewarp.predict.BatchInput``) of the 130 photos."""
    from ..dewarp.predict import BatchInput

    inputs = []
    for k in docunet.DOCUMENTS:
        for m in docunet.VIEWS:
            with Image.open(crop_dir / CROP_PATTERN.format(k=k, m=m)) as crop:
                size = (crop.height, crop.width)

            def load(k: int = k, m: int = m) -> NDArray[np.uint8]:
                return variant_image(variant, k, m, crop_dir, photo_dir, seed)

            source = f"{variant}:{k}_{m}"
            inputs.append(BatchInput(f"{k}_{m}.png", source, load, output_size=size))
    return inputs


def _reusable(path: Path, predictions_dir: Path) -> dict[str, Any] | None:
    """The report at ``path`` when its prediction hashes and recorded failures still match."""
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    hashes = report.get("inputs", {}).get("predictions", {})
    for name, digest in hashes.items():
        file = predictions_dir / f"{name}.png"
        if not file.is_file() or sha256_file(file) != digest:
            return None
    record, current = report.get("predictions_record") or {}, prediction_record(predictions_dir)
    if not hashes or current is None or record.get("failures") != current["failures"]:
        return None
    return report


def run_stress(
    weights: str | Path,
    output_dir: str | Path,
    *,
    variants: Sequence[str] = tuple(VARIANTS),
    device: str = "cpu",
    force_render: bool = False,
    overwrite: bool = False,
    crop_dir: str | Path = DEFAULT_CROP_DIR,
    photo_dir: str | Path = DEFAULT_PHOTO_DIR,
    conventions: Sequence[str] = docunet.DEFAULT_CONVENTIONS,
    build: SiftFlowBuild | None = None,
    references_dir: str | Path = docunet.DEFAULT_REFERENCES_DIR,
    reference_pattern: str = docunet.DEFAULT_REFERENCE_PATTERN,
    verify_references: bool = True,
    workers: int = docunet.DEFAULT_WORKERS,
    seed: int = 0,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Predict and score every variant; write and return ``stress.json``.

    A variant with a missing prediction (a failed check without ``force_render``) is
    reported as ``not_scored`` with the error.
    """
    from ..dewarp.predict import load_dewarper, rectify_batch

    unknown = [name for name in variants if name not in VARIANTS]
    if unknown or not variants:
        raise ValueError(f"variants must be among {tuple(VARIANTS)}, got {list(variants)}")
    conventions = docunet.validate_conventions(conventions)
    output, crops, photos = Path(output_dir), Path(crop_dir), Path(photo_dir)
    dewarper = load_dewarper(weights, device=device)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "evidence": evidence("docunet_stress", dewarper.info),
        "model": dewarper.info,
        "config": {
            "variants": list(variants),
            "conventions": list(conventions),
            "ref_resize": "once",
            "force_render": force_render,
            "device": device,
            "seed": seed,
            "page_fraction": list(PAGE_FRACTION),
            "canvas_max_pixels": CANVAS_MAX_PIXELS,
            "crop_dir": str(crops.resolve()),
            "photo_dir": str(photos.resolve()),
            "output_size": "the upright crop's size",
            "siftflow": build is not None,
        },
        "variants": {},
    }
    for variant in variants:
        directory = output / variant
        log(f"stress: predicting {variant}")
        predictions = rectify_batch(
            dewarper,
            variant_inputs(variant, crops, photos, seed),
            directory,
            overwrite=overwrite,
            force_render=force_render,
            fields={"inputs": f"docunet_stress:{variant}", "variant": variant},
        )
        entry: dict[str, Any] = {
            "description": VARIANTS[variant],
            "predictions": predictions["counts"],
            "failed_ids": [
                item["output"][: -len(".png")]
                for item in predictions["items"]
                if item["status"] != "success"
            ],
        }
        report_path = directory / "docunet.json"
        report = _reusable(report_path, directory)
        if report is None:
            log(f"stress: scoring {variant}")
            try:
                report = docunet.evaluate_docunet(
                    directory,
                    pattern="{k}_{m}.png",
                    conventions=conventions,
                    build=build,
                    references_dir=references_dir,
                    reference_pattern=reference_pattern,
                    verify_references=verify_references,
                    workers=workers,
                    progress=progress,
                )
            except FileNotFoundError as error:
                entry.update(status="not_scored", error=str(error))
                summary["variants"][variant] = entry
                continue
            write_json_atomic(report_path, report, overwrite=True)
        entry.update(
            status="scored",
            comparable=report["comparable"],
            report=f"{variant}/docunet.json",
            report_sha256=sha256_file(report_path),
            evidence=report["evidence"],
            summary=report["summary"],
        )
        summary["variants"][variant] = entry
    write_json_atomic(output / "stress.json", summary, overwrite=True)
    return summary


def format_table(summary: dict[str, Any], convention: str | None = None) -> str:
    """One line per variant: MS-SSIM, LD, AD and failed predictions."""
    lines = [f"{'variant':<11} {'ms_ssim':>8} {'ld':>8} {'ad':>7} {'failed':>7}"]
    for name, entry in summary["variants"].items():
        failed = entry["predictions"]["failed"]
        if entry["status"] != "scored":
            lines.append(f"{name:<11} {'not scored':>25} {failed:>7}")
            continue
        chosen = convention or next(iter(entry["summary"]))
        row = entry["summary"][chosen]
        values = [row["ms_ssim"], row["ld"], row["ad"]]
        text = [f"{v:.4f}" if isinstance(v, float) else "-" for v in values]
        lines.append(f"{name:<11} {text[0]:>8} {text[1]:>8} {text[2]:>7} {failed:>7}")
    return "\n".join(lines)
