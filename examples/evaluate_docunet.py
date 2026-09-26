"""Evaluate 130 DocUNet photos using a frozen checkpoint and scan aspect ratios.

Save execution results and preview images for local review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw, ImageOps

from flattener import FlattenerError, RenderConfig, Status, identity_map, load_image, render
from flattener.image_io import save_result

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
SCENE_EVALUATION_SCHEMA = "scene-coordinate-evaluation-1.0"
PAPER_EVALUATION_SCHEMA = "scene-coordinate-evaluation-1.1"
MANUAL_QUALITY_RUBRIC = {
    "page_completeness": "The entire intended page and its content are included.",
    "background_residue": "Describe any background remaining outside the page.",
    "text_clipping": "Describe cropped characters or missing content at output boundaries.",
    "residual_text_line_curvature": "Describe remaining curvature of visible text baselines.",
}
OCR_TIMEOUT_SECONDS = 30
OCR_MAX_TEXT_BYTES = 1_048_576
OCR_MAX_TEXT_CHARS = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_source(root: Path) -> tuple[Path, dict] | None:
    """A present collection manifest is authoritative; never silently fall back."""
    path = root / "manifest.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        raise ValueError("dataset manifest must not be a symlink")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("dataset manifest exceeds 64 MiB limit")
    with path.open("rb") as stream:
        payload = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("dataset manifest exceeds 64 MiB limit")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "scene-0.3":
        raise ValueError("DocUNet collection requires scene-0.3 manifest")
    if manifest.get("complete") is not True:
        raise ValueError("dataset manifest must declare complete: true")
    collection = manifest.get("collection")
    if (
        not isinstance(collection, dict)
        or collection.get("schema_version") != "dataset-collection-1.0"
        or not isinstance(collection.get("sources"), dict)
    ):
        raise ValueError("invalid dataset-collection-1.0 collection")
    source = collection["sources"].get("docunet")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("provenance"), dict)
        or not isinstance(source.get("samples"), list)
        or not isinstance(source.get("files"), list)
    ):
        raise ValueError("DocUNet source requires provenance, samples, and files")
    return path, source


def _collection_file(root: Path, relative: str) -> Path:
    """Resolve a collection payload without accepting aliases or symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("payload path must be a relative files/ path")
    parts = relative.split("/")
    if parts[0] != "files" or len(parts) < 2 or any(p in {"", ".", ".."} for p in parts):
        raise ValueError("payload path must stay within files/ without traversal")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("payload path must not contain a symlink")
    if not path.resolve().is_relative_to(root):
        raise ValueError("payload path escapes the dataset")
    return path


def _valid_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validation_checkpoint_format(validation: dict, model_hash: str) -> str:
    """Require complete, hash-bound validation evidence before real-photo inference."""
    message = "complete synthetic validation report for this frozen checkpoint is required"
    if not isinstance(validation, dict) or validation.get("evaluation_complete") is not True:
        raise ValueError(message)
    schema = validation.get("schema_version")
    if schema in (None, "learning-evaluation-0.1"):
        if (
            validation.get("split") != "validation"
            or validation.get("checkpoint_sha256") != model_hash
        ):
            raise ValueError(message)
        return "learning-0.1"
    if schema not in (SCENE_EVALUATION_SCHEMA, PAPER_EVALUATION_SCHEMA):
        raise ValueError(message)
    protocol, summary, samples = (validation.get(key) for key in ("protocol", "summary", "samples"))
    if (
        not isinstance(protocol, dict)
        or not isinstance(summary, dict)
        or not isinstance(samples, list)
        or validation.get("integrity_verified") is not True
        or protocol.get("split") != "validation"
        or protocol.get("checkpoint_sha256") != model_hash
    ):
        raise ValueError(message)
    if schema == PAPER_EVALUATION_SCHEMA and (
        type(protocol.get("include_paper_augmentation")) is not bool
        or any(
            not isinstance(sample, dict) or sample.get("source") not in ("authored", "paper")
            for sample in samples
        )
    ):
        raise ValueError(message)
    for field in ("dataset_manifest_sha256", "training_manifest_sha256"):
        if not _valid_sha256(protocol.get(field)):
            raise ValueError(message)
    source_hashes, runtime = protocol.get("source_code_sha256"), protocol.get("runtime")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or not all(
            isinstance(name, str) and _valid_sha256(value) for name, value in source_hashes.items()
        )
        or not isinstance(runtime, dict)
        or not runtime
        or protocol.get("source_sha256") != _canonical_sha256(source_hashes)
        or protocol.get("runtime_sha256") != _canonical_sha256(runtime)
        or validation.get("protocol_sha256") != _canonical_sha256(protocol)
    ):
        raise ValueError(message)
    count_fields = (
        "expected_count",
        "processed_count",
        "epe_missing_count",
        "non_finite_failure_count",
        "renderer_success_count",
        "renderer_failure_count",
        "renderer_missing_count",
    )
    if any(type(summary.get(field)) is not int or summary[field] < 0 for field in count_fields):
        raise ValueError(message)
    expected = summary["expected_count"]
    epe = summary.get("epe_pixels")
    if (
        expected <= 0
        or summary["processed_count"] != expected
        or len(samples) != expected
        or any(
            summary[field]
            for field in ("epe_missing_count", "non_finite_failure_count", "renderer_missing_count")
        )
        or summary["renderer_success_count"] + summary["renderer_failure_count"] != expected
        or not isinstance(epe, dict)
        or type(epe.get("count")) is not int
        or epe["count"] != expected
    ):
        raise ValueError(message)
    sample_ids = set()
    renderer_successes = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(message)
        identity, error = sample.get("id"), sample.get("epe_pixels")
        if (
            not isinstance(identity, str)
            or not identity
            or identity in sample_ids
            or isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not np.isfinite(error)
            or error < 0
            or sample.get("renderer_status") not in {str(status) for status in Status}
        ):
            raise ValueError(message)
        sample_ids.add(identity)
        renderer_successes += sample["renderer_status"] == "success"
    if renderer_successes != summary["renderer_success_count"] or validation.get(
        "samples_sha256"
    ) != _canonical_sha256(samples):
        raise ValueError(message)
    return "coordinate-study"


def _load_evaluation_predictor(checkpoint: Path, checkpoint_format: str):
    if checkpoint_format == "coordinate-study":
        from flattener.learning.coordinate_study import load_coordinate_predictor

        return load_coordinate_predictor(checkpoint)
    from flattener.learning.inference import load_predictor

    return load_predictor(checkpoint)


def _ocr_environment() -> dict:
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    return environment


def _prepare_ocr() -> dict:
    """Read engine identity before output creation; never recognize an image here."""
    executable = shutil.which("tesseract")
    if executable is None:
        raise ValueError("--ocr requires a local tesseract executable and eng language data")
    executable = str(Path(executable).resolve())
    try:
        options = {
            "capture_output": True,
            "timeout": 5,
            "check": True,
            "shell": False,
            "env": _ocr_environment(),
        }
        version = subprocess.run([executable, "--version"], **options).stdout.decode("utf-8")
        languages = subprocess.run([executable, "--list-langs"], **options).stdout.decode("utf-8")
        if "eng" not in languages.splitlines():
            raise ValueError("--ocr requires the installed eng language data")
        header = re.search(r'List of available languages in "([^"]+)"', languages)
        if header is None:
            raise ValueError("--ocr cannot identify the installed eng language data")
        language_file = (Path(header[1]) / "eng.traineddata").resolve(strict=True)
        return {
            "executable": executable,
            "executable_sha256": sha256(Path(executable)),
            "version": version.splitlines()[0],
            "language": "eng",
            "language_data_sha256": sha256(language_file),
            "psm": 6,
            "threads": 1,
            "dpi": 300,
            "timeout_seconds": OCR_TIMEOUT_SECONDS,
        }
    except (OSError, subprocess.SubprocessError, UnicodeError, IndexError) as error:
        raise ValueError("--ocr could not verify the local tesseract/eng installation") from error


def _normalize_ocr_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _edit_distance(left: str, right: str) -> int:
    """Exact Unicode-codepoint Levenshtein distance using bounded bit-vector state."""
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)
    masks = {}
    for index, character in enumerate(left):
        masks[character] = masks.get(character, 0) | (1 << index)
    width = len(left)
    mask, highest = (1 << width) - 1, 1 << (width - 1)
    positive, negative, distance = mask, 0, width
    for character in right:
        equal = masks.get(character, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & highest:
            distance += 1
        elif negative_horizontal & highest:
            distance -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & mask
        negative_horizontal = (negative_horizontal << 1) & mask
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = positive_horizontal & vertical
    return distance


def _recognize_ocr(pixels: np.ndarray, size: tuple[int, int], engine: dict) -> tuple:
    """Keep OCR text only in memory; temporary grayscale PNGs are always removed."""
    try:
        with tempfile.TemporaryDirectory(prefix="flattener-docunet-ocr-") as directory:
            image_path = Path(directory) / "input.png"
            with Image.fromarray(pixels) as image, image.convert("L") as gray:
                with gray.resize((size[1], size[0]), Image.Resampling.LANCZOS) as resized:
                    resized.save(image_path, format="PNG", dpi=(300, 300))
            image_hash = sha256(image_path)
            try:
                completed = subprocess.run(
                    [
                        engine["executable"],
                        str(image_path),
                        "stdout",
                        "-l",
                        "eng",
                        "--psm",
                        "6",
                        "--dpi",
                        "300",
                    ],
                    capture_output=True,
                    timeout=OCR_TIMEOUT_SECONDS,
                    check=False,
                    shell=False,
                    cwd=directory,
                    env=_ocr_environment(),
                )
            except subprocess.TimeoutExpired:
                return None, {"status": "failed", "reason": "timeout"}
            if completed.returncode != 0:
                return None, {
                    "status": "failed",
                    "reason": "engine_failed",
                    "exit_code": completed.returncode,
                }
            if len(completed.stdout) > OCR_MAX_TEXT_BYTES:
                return None, {"status": "failed", "reason": "text_byte_limit"}
            normalized = _normalize_ocr_text(completed.stdout.decode("utf-8"))
            if len(normalized) > OCR_MAX_TEXT_CHARS:
                return None, {"status": "failed", "reason": "text_character_limit"}
            return normalized, {
                "status": "success",
                "character_count": len(normalized),
                "text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                "input_png_sha256": image_hash,
            }
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, {"status": "failed", "reason": "recognition_failed"}


def _ocr_case(reference, corrected, size, engine, reference_cache, cache_key) -> dict:
    cached = cache_key in reference_cache
    if not cached:
        reference_cache[cache_key] = _recognize_ocr(reference, size, engine)
    reference_text, reference_info = reference_cache[cache_key]
    result = {
        "status": "reference_failed",
        "scan_ocr_disagreement": None,
        "reference_cache_hit": cached,
        "reference": dict(reference_info),
        "prediction": {"status": "not_run", "reason": "reference_failed"},
    }
    if reference_text is None:
        return result
    if not reference_text:
        result.update(
            status="reference_empty", prediction={"status": "not_run", "reason": "reference_empty"}
        )
        return result
    prediction_text, prediction_info = _recognize_ocr(corrected, size, engine)
    result["prediction"] = prediction_info
    if prediction_text is None:
        result["status"] = "prediction_failed"
        return result
    distance = _edit_distance(reference_text, prediction_text)
    denominator = max(len(reference_text), len(prediction_text))
    result.update(
        status="measured",
        edit_distance=distance,
        normalizer_characters=denominator,
        scan_ocr_disagreement=distance / denominator,
    )
    return result


def _ocr_summary(rows, engine, reference_cache) -> dict:
    values = [
        row["ocr"]["scan_ocr_disagreement"] for row in rows if row["ocr"]["status"] == "measured"
    ]
    status_counts = Counter(row["ocr"]["status"] for row in rows)
    return {
        "status": "diagnostic_only",
        "quality_status": "not_evaluated",
        "engine": engine,
        "expected_count": 130,
        "measured_count": len(values),
        "missing_count": 130 - len(values),
        "status_counts": dict(status_counts),
        "failed_count": status_counts["reference_failed"] + status_counts["prediction_failed"],
        "reference_empty_count": status_counts["reference_empty"],
        "not_run_count": status_counts["not_run"],
        "reference_cache_entries": len(reference_cache),
        "reference_cache_hits": sum(row["ocr"].get("reference_cache_hit", False) for row in rows),
        "unique_reference_status_counts": dict(
            Counter(
                "failed" if text is None else "empty" if not text else "recognized"
                for text, _ in reference_cache.values()
            )
        ),
        "scan_ocr_disagreement": {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        },
        "preprocessing": {
            "mode": "grayscale",
            "size": "same per-case output_size_hw for scan and corrected image",
            "resize": "Pillow LANCZOS",
            "dpi": 300,
        },
        "text_normalization": "NFKC; collapse/strip whitespace; preserve case and punctuation",
        "metric_definition": (
            "Unicode-codepoint Levenshtein distance divided by max(reference_characters, "
            "prediction_characters); empty reference is missing; empty prediction has value 1."
        ),
        "text_storage": "In-memory text; saved records contain counts, hashes and distances only.",
        "limits": {"max_text_bytes": OCR_MAX_TEXT_BYTES, "max_text_characters": OCR_MAX_TEXT_CHARS},
        "limitations": [
            "Unverified scan OCR can contain recognition errors that affect agreement.",
            "Fixed eng/psm6 can miss complex layouts and non-English content.",
            "Means describe measured cases only; missing cases remain in the denominator of 130.",
            "No quality acceptance threshold; excluded from model selection.",
        ],
    }


def _discover_collection_cases(root: Path, source: dict) -> tuple[list[dict], list[str]]:
    index, samples, unexpected = {}, {}, []
    for entry in source["files"]:
        if not isinstance(entry, dict):
            raise ValueError("DocUNet file index entries must be objects")
        path = _collection_file(root, entry.get("path"))
        size = entry.get("bytes")
        if (
            not _valid_sha256(entry.get("sha256"))
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError("DocUNet file index requires SHA256 and byte length")
        if path in index:
            raise ValueError("duplicate DocUNet file index path")
        index[path] = entry
    for entry in source["samples"]:
        case_id = entry.get("id") if isinstance(entry, dict) else None
        matched = (
            re.fullmatch(r"([1-9][0-9]*)_([12])", case_id or "")
            if isinstance(case_id, str)
            else None
        )
        if matched is None or not 1 <= int(matched[1]) <= 65:
            unexpected.append(f"manifest sample has invalid DocUNet id: {case_id!r}")
            continue
        samples.setdefault(case_id, []).append(entry)
    cases, used = [], set()
    for document in range(1, 66):
        for view in (1, 2):
            case_id = f"{document}_{view}"
            entries = samples.get(case_id, [])
            case = {
                "id": case_id,
                "document_id": document,
                "view": view,
                "photo": None,
                "reference": None,
                "expected_sha256": {},
                "expected_bytes": {},
                "discovery_errors": [],
            }
            errors = case["discovery_errors"]
            if len(entries) != 1:
                errors.append(f"expected one manifest sample, found {len(entries)}")
            else:
                entry = entries[0]
                if (
                    entry.get("document_group") != f"docunet:{document}"
                    or entry.get("split") != "evaluation"
                    or entry.get("training_eligible") is not False
                ):
                    errors.append("DocUNet sample must be evaluation-only with its document group")
                paths, hashes = entry.get("paths"), entry.get("sha256")
                for role in ("photo", "reference"):
                    try:
                        if not isinstance(paths, dict) or not isinstance(hashes, dict):
                            raise ValueError("sample requires paths and SHA256 objects")
                        path = _collection_file(root, paths.get(role))
                        used.add(path)
                        if not path.is_file():
                            raise ValueError(f"missing {role} file")
                        expected = hashes.get(role)
                        if not _valid_sha256(expected):
                            raise ValueError(f"missing or invalid {role} SHA256")
                        file_entry = index.get(path)
                        if file_entry is None or file_entry["sha256"].lower() != expected.lower():
                            raise ValueError(
                                f"{role} SHA256 conflicts with or is absent from file index"
                            )
                        case[role] = path
                        case["expected_sha256"][role] = expected.lower()
                        case["expected_bytes"][role] = file_entry["bytes"]
                    except (OSError, ValueError, TypeError, RuntimeError) as exc:
                        errors.append(f"{role}: {exc}")
            cases.append(case)
    # Shared references are expected for the two views of one document only.
    for role in ("photo", "reference"):
        owners = {}
        for case in cases:
            if case[role] is not None:
                owners.setdefault(case[role], []).append(case)
        for owners_for_path in owners.values():
            identities = {c["id"] if role == "photo" else c["document_id"] for c in owners_for_path}
            if len(identities) > 1:
                for case in owners_for_path:
                    case["discovery_errors"].append(
                        f"{role} file reused by different DocUNet identities"
                    )
    for first, second in zip(cases[::2], cases[1::2]):
        if first["reference"] and second["reference"] and first["reference"] != second["reference"]:
            for case in (first, second):
                case["discovery_errors"].append("document views have different reference files")
    unexpected.extend(
        f"unreferenced DocUNet file index path: {p.relative_to(root)}"
        for p in sorted(index.keys() - used)
    )
    return cases, unexpected


def discover_cases(dataset_dir: Path) -> tuple[list[dict], list[str]]:
    """Enumerate 65 documents x 2 views from collection or legacy extracted data."""
    root = Path(dataset_dir).resolve()
    collection = _collection_source(root)
    if collection is not None:
        return _discover_collection_cases(root, collection[1])
    return _discover_legacy_cases(root)


def _discover_legacy_cases(root: Path) -> tuple[list[dict], list[str]]:
    photos, scans, unexpected = {}, {}, []
    for kind, mapping, pattern, suffixes in (
        ("original", photos, r"([1-9][0-9]*)_([12])", {".jpg", ".jpeg"}),
        ("scan", scans, r"([1-9][0-9]*)", {".png"}),
    ):
        for path in sorted((root / "extracted" / kind).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if not path.resolve().is_relative_to(root):
                unexpected.append(str(path))
                continue
            matched = re.fullmatch(pattern, path.stem)
            if not matched or not 1 <= int(matched[1]) <= 65:
                unexpected.append(str(path))
                continue
            key = tuple(int(value) for value in matched.groups())
            mapping.setdefault(key, []).append(path)
    cases = []
    for document in range(1, 66):
        for view in (1, 2):
            errors = []
            photo = photos.get((document, view), [])
            reference = scans.get((document,), [])
            if len(photo) != 1:
                errors.append(f"expected one photo, found {len(photo)}")
            if len(reference) != 1:
                errors.append(f"expected one reference scan, found {len(reference)}")
            cases.append(
                {
                    "id": f"{document}_{view}",
                    "document_id": document,
                    "view": view,
                    "photo": photo[0] if len(photo) == 1 else None,
                    "reference": reference[0] if len(reference) == 1 else None,
                    "discovery_errors": errors,
                }
            )
    return cases, unexpected


def reference_output_size(reference_size: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Fit the oriented reference's aspect ratio within max_side, without upscaling."""
    if (
        len(reference_size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in reference_size)
        or isinstance(max_side, bool)
        or not isinstance(max_side, int)
        or not 32 <= max_side <= 2048
    ):
        raise ValueError(
            "reference size must be positive; max_side must be an integer in [32,2048]"
        )
    scale = min(1.0, max_side / max(reference_size))
    return tuple(max(1, round(value * scale)) for value in reference_size)


def load_reference(path: Path) -> tuple[np.ndarray, dict]:
    """Orient scans; composite reference-only alpha onto white explicitly."""
    with Image.open(path) as image:
        if image.width * image.height > 16_000_000 or max(image.size) > 8192:
            raise ValueError("reference image exceeds resource limits")
        if image.mode not in {"RGB", "RGBA", "L"}:
            raise ValueError(f"unsupported reference mode {image.mode}")
        info = {
            "file_mode": image.mode,
            "file_size_hw": [image.height, image.width],
            "exif_orientation": image.getexif().get(274, 1),
            "alpha_composited_on_white": image.mode == "RGBA",
        }
        with ImageOps.exif_transpose(image) as oriented:
            if oriented.mode == "RGBA":
                with Image.new("RGBA", oriented.size, (255, 255, 255, 255)) as background:
                    with Image.alpha_composite(background, oriented) as composed:
                        with composed.convert("RGB") as rgb:
                            result = np.array(rgb, dtype=np.uint8, copy=True)
            else:
                with oriented.convert("RGB") as rgb:
                    result = np.array(rgb, dtype=np.uint8, copy=True)
    info["oriented_size_hw"] = list(result.shape[:2])
    return result, info


def save_contact_sheet(
    path: Path, case_id: str, status: str, panels: list[tuple[str, np.ndarray | None]]
):
    """Make labeled, letterboxed previews without hiding failed model outputs."""
    cell_width, cell_height, header = 240, 320, 48
    with Image.new(
        "RGB", (cell_width * len(panels), cell_height + header), (236, 238, 242)
    ) as sheet:
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 8), f"DocUNet {case_id} | {status} | local research preview", fill=(20, 28, 38)
        )
        draw.text(
            (12, 25),
            "Reference sets output aspect ratio. Valid mask is predicted source coverage.",
            fill=(50, 60, 72),
        )
        for index, (label, pixels) in enumerate(panels):
            left = index * cell_width
            draw.text((left + 10, header + 8), label, fill=(20, 28, 38))
            if pixels is None:
                draw.text((left + 10, header + 50), "Unavailable / failed", fill=(155, 30, 30))
                continue
            with Image.fromarray(pixels) as original:
                with ImageOps.contain(
                    original.convert("RGB"), (cell_width - 20, cell_height - 42)
                ) as preview:
                    sheet.paste(preview, (left + (cell_width - preview.width) // 2, header + 32))
        sheet.save(path, format="PNG")


def evaluate_docunet(
    dataset_dir,
    checkpoint_path,
    validation_report,
    output_dir,
    *,
    max_side=512,
    threads=1,
    all_previews=False,
    ocr=False,
) -> dict:
    """Evaluate all declared cases; write new local artifacts and a complete report."""
    from flattener.learning.inference import unwarp

    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 128:
        raise ValueError("threads must be an integer in [1,128]")
    if not isinstance(all_previews, bool):
        raise ValueError("all_previews must be a boolean")
    if not isinstance(ocr, bool):
        raise ValueError("ocr must be a boolean")
    reference_output_size((512, 384), max_side)
    ocr_engine = _prepare_ocr() if ocr else None
    reference_ocr_cache = {}
    root = Path(dataset_dir).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    validation_path = Path(validation_report).resolve()
    if validation_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("validation report exceeds 64 MiB limit")
    with validation_path.open("rb") as stream:
        validation_payload = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(validation_payload) > MAX_MANIFEST_BYTES:
        raise ValueError("validation report exceeds 64 MiB limit")
    validation = json.loads(validation_payload)
    validation_hash = hashlib.sha256(validation_payload).hexdigest()
    model_hash = sha256(checkpoint)
    checkpoint_format = _validation_checkpoint_format(validation, model_hash)
    collection = _collection_source(root)
    if collection is None:
        provenance_path = root / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance_hash = sha256(provenance_path)
        manifest_hash = None
        cases, unexpected = _discover_legacy_cases(root)
    else:
        provenance_path, source = collection
        provenance = source["provenance"]
        manifest_hash = sha256(provenance_path)
        provenance_hash = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest()
        cases, unexpected = _discover_collection_cases(root, source)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    (output / "contact-sheets").mkdir()
    # Fixed, uniform index selection precedes inference. All operational
    # failures additionally get contact sheets, without subjective ranking.
    subset = {cases[int(index)]["id"] for index in np.linspace(0, len(cases) - 1, 12).round()}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    started = perf_counter()
    rows = []
    try:
        load_started = perf_counter()
        predictor = _load_evaluation_predictor(checkpoint, checkpoint_format)
        if predictor.model_sha256 != model_hash:
            raise ValueError("loaded checkpoint differs from frozen validation evidence")
        model_load_seconds = perf_counter() - load_started
        for case in cases:
            begin = perf_counter()
            row = {
                "id": case["id"],
                "document_id": case["document_id"],
                "view": case["view"],
                "photo_path": str(case["photo"]) if case["photo"] else None,
                "reference_path": str(case["reference"]) if case["reference"] else None,
                "model_sha256": model_hash,
                "status": "input_error",
                "quality_status": "not_evaluated",
                "manual_quality_review": {
                    criterion: "not_evaluated" for criterion in MANUAL_QUALITY_RUBRIC
                },
                "reason": None,
                "input_size_hw": None,
                "output_size_hw": None,
                "photo_sha256": None,
                "reference_sha256": None,
                "expected_sha256": case.get("expected_sha256", {}),
                "source_hashes_verified": False if collection else None,
                "timings": {},
            }
            if ocr:
                row["ocr"] = {
                    "status": "not_run",
                    "reason": "processing_not_successful",
                    "scan_ocr_disagreement": None,
                }
            photo = reference = identity = corrected = mask = None
            try:
                if case["discovery_errors"]:
                    raise ValueError("; ".join(case["discovery_errors"]))
                for role in case.get("expected_sha256", {}):
                    _collection_file(root, str(case[role].relative_to(root)))
                row["photo_sha256"] = sha256(case["photo"])
                row["reference_sha256"] = sha256(case["reference"])
                for role, expected in case.get("expected_sha256", {}).items():
                    if row[f"{role}_sha256"] != expected:
                        raise ValueError(f"{role} SHA256 mismatch")
                    if case[role].stat().st_size != case["expected_bytes"][role]:
                        raise ValueError(f"{role} byte length mismatch")
                if collection:
                    row["source_hashes_verified"] = True
                decode_started = perf_counter()
                with Image.open(case["photo"]) as encoded_photo:
                    row["photo_decode"] = {
                        "file_size_hw": [encoded_photo.height, encoded_photo.width],
                        "file_mode": encoded_photo.mode,
                        "exif_orientation": encoded_photo.getexif().get(274, 1),
                    }
                photo = load_image(case["photo"])
                reference, reference_info = load_reference(case["reference"])
                row["reference_decode"] = reference_info
                row["input_size_hw"] = list(photo.shape[:2])
                row["oriented_photo_pixels_sha256"] = hashlib.sha256(photo.tobytes()).hexdigest()
                row["timings"]["decode_seconds"] = perf_counter() - decode_started
                size = reference_output_size(reference.shape[:2], max_side)
                row["output_size_hw"] = list(size)
                inference_started = perf_counter()
                result = unwarp(
                    photo, predictor, output_size=size, config=RenderConfig(on_error="return")
                )
                row["timings"]["predict_and_render_seconds"] = perf_counter() - inference_started
                row["status"] = str(result.status)
                row["reason"] = result.reason_code
                row["message"] = result.message
                row["used_fallback"] = result.used_fallback
                row["validation"] = result.metadata.get("validation", {})
                row["model_stage_timings"] = result.metadata.get("timings", {})
                save_started = perf_counter()
                row["artifacts"] = save_result(result, output / "cases" / case["id"])
                row["artifact_sha256"] = {
                    key: sha256(Path(path)) for key, path in row["artifacts"].items()
                }
                row["timings"]["save_seconds"] = perf_counter() - save_started
                corrected = result.image
                if ocr and result.status == Status.SUCCESS and corrected is not None:
                    ocr_started = perf_counter()
                    cache_key = (case["document_id"], row["reference_sha256"], tuple(size))
                    row["ocr"] = _ocr_case(
                        reference, corrected, size, ocr_engine, reference_ocr_cache, cache_key
                    )
                    row["timings"]["ocr_diagnostic_seconds"] = perf_counter() - ocr_started
                if result.valid_mask is not None:
                    mask = result.valid_mask.astype(np.uint8) * 255
                if all_previews or case["id"] in subset or result.status != Status.SUCCESS:
                    identity = render(photo, identity_map(size)).image
            except (FlattenerError, OSError, ValueError, TypeError, RuntimeError) as exc:
                row["status"] = (
                    str(exc.status) if isinstance(exc, FlattenerError) else "input_error"
                )
                row["reason"] = (
                    exc.reason_code if isinstance(exc, FlattenerError) else "case_failed"
                )
                row["message"] = str(exc)
            if all_previews or case["id"] in subset or row["status"] != "success":
                sheet = output / "contact-sheets" / f"{case['id']}.png"
                try:
                    save_contact_sheet(
                        sheet,
                        case["id"],
                        row["status"],
                        [
                            ("Original (oriented)", photo),
                            ("Reference scan", reference),
                            ("Identity resized", identity),
                            ("Model corrected", corrected),
                            ("Predicted valid mask", mask),
                        ],
                    )
                    row["contact_sheet"] = str(sheet)
                except (OSError, ValueError) as exc:
                    row["contact_sheet_error"] = str(exc)
            row["timings"]["case_total_seconds"] = perf_counter() - begin
            rows.append(row)
            print(f"{case['id']}: {row['status']} ({len(rows)}/{len(cases)})", flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    successes = sum(row["status"] == "success" for row in rows)
    inference_times = [
        row["timings"]["predict_and_render_seconds"]
        for row in rows
        if "predict_and_render_seconds" in row["timings"]
    ]
    report = {
        "schema_version": "docunet-local-evaluation-0.1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "DocUNet benchmark",
        "evaluator_source_sha256": sha256(Path(__file__)),
        "dataset_dir": str(root),
        "dataset_layout": "collection" if collection else "legacy-extracted",
        "dataset_manifest_sha256": manifest_hash,
        "dataset_provenance_sha256": provenance_hash,
        "dataset_provenance_hash_method": "canonical-json-sorted-compact"
        if collection
        else "file-bytes",
        "source_page": provenance.get("source_page"),
        "license_spdx": provenance.get("license_spdx"),
        "usage": (
            "Local research evaluation only; external inputs and derived previews "
            "are not redistributed."
        ),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": model_hash,
        "checkpoint_format": checkpoint_format,
        "synthetic_validation_report": str(validation_path),
        "synthetic_validation_report_sha256": validation_hash,
        "synthetic_validation_report_schema": validation.get("schema_version"),
        "checkpoint_selection": (
            "Frozen before real-photo evaluation; no DocUNet training or model selection"
        ),
        "expected_count": 130,
        "processed_count": len(rows),
        "completed_count": successes,
        "failed_count": 130 - successes,
        "all_cases_accounted_for": len(rows) == 130,
        "evaluation_complete": successes == 130 and not unexpected,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "document_count": 65,
        "unexpected_input_files": unexpected,
        "quality_claim_allowed": False,
        "quality_status": "not_evaluated",
        "manual_quality_rubric": {
            "version": "1.0",
            "status": "not_evaluated",
            "criteria": dict(MANUAL_QUALITY_RUBRIC),
            "instructions": (
                "Review original, corrected, and reference panels for the fixed 12 cases; "
                "record observations for each criterion. No overall quality threshold is set."
            ),
        },
        "reference_usage": (
            "Oracle output aspect ratio from EXIF-oriented scan, max-side fit without upscaling"
        ),
        "max_output_side": max_side,
        "metrics_not_available": [
            "dense correspondence EPE",
            "official MS-SSIM",
            "official LD",
            "OCR accuracy",
            "ground-truth page completeness and text clipping",
            "residual text-line curvature",
        ],
        "limitations": [
            "Renderer success establishes execution and geometry checks, not document quality.",
            "Scans are not registered to predictions; no pixel similarity proxy is reported.",
            "Output aspect ratio uses reference information unavailable during normal deployment.",
            "Synthetic-only training does not establish generalization to real photographs.",
            "Validity masks describe predicted source bounds, not ground-truth correctness.",
        ],
        "contact_sheet_selection": (
            "All 130 cases; 12 uniform preselected cases also recorded separately"
            if all_previews
            else "12 uniform sample indices chosen before inference, plus every failure"
        ),
        "preselected_preview_ids": sorted(subset),
        "selected_preview_ids": sorted(case["id"] for case in cases)
        if all_previews
        else sorted(subset),
        "model_load_seconds": model_load_seconds,
        "total_seconds": perf_counter() - started,
        "predict_and_render_seconds": {
            "count": len(inference_times),
            "median": float(np.median(inference_times)) if inference_times else None,
            "p95": float(np.percentile(inference_times, 95)) if inference_times else None,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": str(torch.__version__),
        },
        "cpu_threads": threads,
        "samples": rows,
    }
    if ocr:
        report["ocr_diagnostic"] = _ocr_summary(rows, ocr_engine, reference_ocr_cache)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--all-previews", action="store_true")
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Optional local Tesseract scan/output text-agreement diagnostic; no quality pass",
    )
    args = parser.parse_args()
    report = evaluate_docunet(
        args.dataset_dir,
        args.checkpoint,
        args.validation_report,
        args.output_dir,
        max_side=args.max_side,
        threads=args.threads,
        all_previews=args.all_previews,
        ocr=args.ocr,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("expected_count", "completed_count", "failed_count")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
