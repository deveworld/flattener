"""Reproducible synthetic samples, group splits, and hash-checked datasets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import PIL
from PIL import Image, ImageFilter

from ..contracts import FlattenerError, RenderConfig, Status
from ..renderer import render
from ..validation import validate_image, validate_size
from .documents import generate_document
from .geometry import make_geometry, sample_warp

SCHEMA_VERSION = "synthetic-0.1"
FAMILIES = ("perspective", "gentle", "strong", "local")


@dataclass(frozen=True, slots=True)
class PhotometricConfig:
    enabled: bool = True
    brightness_min: float = 0.85
    brightness_max: float = 1.1
    max_shadow: float = 0.2
    max_blur_radius: float = 0.5
    noise_std: float = 0.008


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    seed: int = 20260921
    train_documents: int = 32
    validation_documents: int = 8
    test_documents: int = 8
    variants_per_document: int = 4
    output_size: tuple[int, int] = (256, 192)
    photo_size: tuple[int, int] = (256, 192)
    families: tuple[str, ...] = FAMILIES
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)


@dataclass(slots=True)
class SyntheticSample:
    flat: np.ndarray
    photo: np.ndarray
    backward_map: np.ndarray
    valid_mask: np.ndarray
    photo_page_mask: np.ndarray
    metadata: dict[str, Any]


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise _error("invalid_seed", "seed must be a nonnegative integer below 2**63")
    if not 0 <= int(value) < 2**63:
        raise _error("invalid_seed", "seed must be a nonnegative integer below 2**63")
    return int(value)


def derive_seed(seed: int, *labels: str) -> int:
    """Domain-separated stable seeds, independent of Python's randomized hash."""
    encoded = json.dumps([_seed(seed), *labels], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], "big") % 2**63


def _photometric_config(config: PhotometricConfig) -> PhotometricConfig:
    if not isinstance(config, PhotometricConfig) or not isinstance(config.enabled, bool):
        raise _error("invalid_config", "photometric must be a PhotometricConfig")
    limits = {
        "brightness_min": (0.1, 2.0),
        "brightness_max": (0.1, 2.0),
        "max_shadow": (0.0, 0.8),
        "max_blur_radius": (0.0, 4.0),
        "noise_std": (0.0, 0.2),
    }
    for name, (low, high) in limits.items():
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not low <= value <= high
        ):
            raise _error("invalid_config", f"{name} must be finite and in [{low}, {high}]")
    if config.brightness_min > config.brightness_max:
        raise _error("invalid_config", "brightness_min must not exceed brightness_max")
    return config


def apply_photometry(
    image: np.ndarray, seed: int, config: PhotometricConfig | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Change appearance only; RNG state and geometric maps are never modified."""
    validate_image(image, RenderConfig())
    config = _photometric_config(PhotometricConfig() if config is None else config)
    rng = np.random.default_rng(_seed(seed))
    metadata: dict[str, Any] = {"seed": int(seed), "config": asdict(config)}
    if not config.enabled:
        metadata["applied"] = False
        return image.copy(), metadata
    brightness = float(rng.uniform(config.brightness_min, config.brightness_max))
    shadow = float(rng.uniform(0, config.max_shadow))
    direction = float(rng.uniform(-np.pi, np.pi))
    blur = float(rng.uniform(0, config.max_blur_radius))
    height, width = image.shape[:2]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    gradient = np.clip((x * np.cos(direction) + y * np.sin(direction) + 1) / 2, 0, 1)
    values = image.astype(np.float32) * brightness * (1 - shadow * gradient[..., None])
    values += rng.normal(0, config.noise_std * 255, image.shape).astype(np.float32)
    output = np.floor(np.clip(values, 0, 255) + 0.5).astype(np.uint8)
    if blur:
        with Image.fromarray(output) as pil_image:
            with pil_image.filter(ImageFilter.GaussianBlur(blur)) as blurred:
                output = np.array(blurred, dtype=np.uint8, copy=True)
    metadata.update(
        applied=True,
        brightness=brightness,
        shadow_strength=shadow,
        shadow_direction_radians=direction,
        blur_radius_pixels=blur,
        noise_std_255=config.noise_std * 255,
    )
    return output, metadata


def generate_sample(
    document_id: str,
    document_seed: int,
    warp_seed: int,
    photometric_seed: int,
    *,
    family: str = "gentle",
    output_size: tuple[int, int] = (256, 192),
    photo_size: tuple[int, int] | None = None,
    photometric: PhotometricConfig | None = None,
) -> SyntheticSample:
    """Synthesize a photo with its analytic flat-output-to-photo-source truth."""
    _seed(document_seed)
    _seed(warp_seed)
    _seed(photometric_seed)
    appearance_config = _photometric_config(
        PhotometricConfig() if photometric is None else photometric
    )
    document = generate_document(document_id, document_seed, output_size)
    params = sample_warp(family, warp_seed)
    geometry = make_geometry(output_size, output_size if photo_size is None else photo_size, params)
    # The fill is background, not an invented ground-truth correspondence.
    background_rng = np.random.default_rng(derive_seed(photometric_seed, "background"))
    background = tuple(int(v) for v in background_rng.integers(35, 100, size=3))
    photographed = render(
        document.image,
        geometry.synthesis_map,
        target_mask=geometry.photo_page_mask,
        config=RenderConfig(fill_rgb=background, max_out_of_bounds_fraction=1),
    )
    photo, appearance = apply_photometry(photographed.image, photometric_seed, appearance_config)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "seeds": {
            "document": int(document_seed),
            "warp": int(warp_seed),
            "photometric": int(photometric_seed),
        },
        "family": family,
        "output_size": list(output_size),
        "photo_size": list(photo.shape[:2]),
        "original_aspect_ratio_width_over_height": document.image.shape[1]
        / document.image.shape[0],
        "document": document.metadata,
        "text": document.text,
        "geometry": geometry.metadata,
        "photometric": appearance,
        "background_rgb": list(background),
        "source": "Project-authored procedural document and analytic invertible 2D deformation",
        "terms": "Project-authored text/code Apache-2.0; font terms recorded separately",
        "is_real_photo": False,
    }
    return SyntheticSample(
        flat=document.image,
        photo=photo,
        backward_map=geometry.backward_map,
        valid_mask=geometry.valid_mask,
        photo_page_mask=geometry.photo_page_mask,
        metadata=metadata,
    )


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _write_png(path: Path, image: np.ndarray) -> None:
    with Image.fromarray(image) as encoded:
        encoded.save(path, format="PNG")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_config(config: DatasetConfig) -> DatasetConfig:
    if not isinstance(config, DatasetConfig):
        raise _error("invalid_config", "config must be DatasetConfig")
    _seed(config.seed)
    for name in (
        "train_documents",
        "validation_documents",
        "test_documents",
        "variants_per_document",
    ):
        count = getattr(config, name)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10_000:
            raise _error("invalid_config", f"{name} must be an integer in [1, 10000]")
    total_documents = config.train_documents + config.validation_documents + config.test_documents
    if total_documents * config.variants_per_document > 100_000:
        raise _error("invalid_config", "dataset exceeds the 100000-sample generation limit")
    for name in ("output_size", "photo_size"):
        size = validate_size(getattr(config, name), max_pixels=2048**2, max_side=2048, name=name)
        if min(size) < 128:
            raise _error("invalid_config", f"{name} dimensions must be at least 128")
    if (
        not isinstance(config.families, tuple)
        or not config.families
        or any(family not in FAMILIES for family in config.families)
    ):
        raise _error("invalid_config", "families must be a nonempty tuple of supported families")
    _photometric_config(config.photometric)
    return config


def generate_dataset(
    output_dir: str | os.PathLike[str], *, config: DatasetConfig | None = None
) -> dict:
    """Create a new dataset; manifest publication marks successful completion.

    The output directory must not exist. An interrupted generation leaves its
    partial files for inspection, without a completed manifest; no user files
    are deleted or overwritten. Do not write to this directory concurrently.
    """
    config = _dataset_config(DatasetConfig() if config is None else config)
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise FlattenerError(Status.IO_ERROR, "dataset_create_failed", str(exc)) from exc
    documents = []
    samples = []
    index = 0
    # Split first, before producing any variant. Document IDs and seeds are
    # disjoint, and all captures/augmentations of a document stay in that split.
    for split, count in (
        ("train", config.train_documents),
        ("validation", config.validation_documents),
        ("test", config.test_documents),
    ):
        for _ in range(count):
            document_id = f"document-{index:06d}"
            document_seed = derive_seed(config.seed, "document", document_id)
            documents.append({"id": document_id, "seed": document_seed, "split": split})
            for variant in range(config.variants_per_document):
                sample_id = f"{document_id}-view-{variant:03d}"
                family = config.families[(index + variant) % len(config.families)]
                sample = generate_sample(
                    document_id,
                    document_seed,
                    derive_seed(config.seed, "warp", sample_id),
                    derive_seed(config.seed, "appearance", sample_id),
                    family=family,
                    output_size=config.output_size,
                    photo_size=config.photo_size,
                    photometric=config.photometric,
                )
                sample_dir = directory / "samples" / sample_id
                sample_dir.mkdir(parents=True)
                _write_png(sample_dir / "flat.png", sample.flat)
                _write_png(sample_dir / "photo.png", sample.photo)
                archive = io.BytesIO()
                np.savez_compressed(
                    archive,
                    backward_map=sample.backward_map,
                    valid_mask=sample.valid_mask,
                    photo_page_mask=sample.photo_page_mask,
                )
                (sample_dir / "geometry.npz").write_bytes(archive.getvalue())
                sample.metadata.update(
                    sample_id=sample_id,
                    document_group=document_id,
                    capture_session=f"synthetic:{sample_id}",
                    split=split,
                )
                (sample_dir / "metadata.json").write_bytes(_canonical_json(sample.metadata))
                paths = {
                    name: f"samples/{sample_id}/{name}"
                    for name in ("flat.png", "photo.png", "geometry.npz", "metadata.json")
                }
                samples.append(
                    {
                        "id": sample_id,
                        "document_group": document_id,
                        "capture_session": f"synthetic:{sample_id}",
                        "split": split,
                        "family": family,
                        "paths": paths,
                        "sha256": {
                            name: _sha256(directory / relative) for name, relative in paths.items()
                        },
                    }
                )
            index += 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "config": asdict(config),
        "documents": documents,
        "samples": samples,
        "split_unit": "document_group; every synthetic capture stays with its source document",
        "is_real_photo": False,
        "generator_code_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("pipeline.py", "documents.py", "geometry.py")
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
    }
    (directory / "manifest.json").write_bytes(_canonical_json(manifest))
    validate_dataset(directory)
    return manifest


def _validate_dataset(directory: str | os.PathLike[str]) -> dict:
    root = Path(directory).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _error("invalid_dataset", f"Cannot read completed dataset manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise _error("invalid_dataset", "manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("complete") is not True:
        raise _error("invalid_dataset", "unsupported or incomplete synthetic dataset")
    document_splits = {}
    document_seeds = set()
    if not isinstance(manifest.get("documents"), list) or not isinstance(
        manifest.get("samples"), list
    ):
        raise _error("invalid_dataset", "documents and samples must be lists")
    for document in manifest["documents"]:
        if not isinstance(document, dict) or not isinstance(document.get("id"), str):
            raise _error("invalid_dataset", "documents must have string IDs")
        if document["id"] in document_splits or document["split"] not in {
            "train",
            "validation",
            "test",
        }:
            raise _error("split_leakage", "duplicate document or invalid split")
        document_splits[document["id"]] = document["split"]
        if document["seed"] in document_seeds:
            raise _error("split_leakage", "document seeds must be unique")
        document_seeds.add(_seed(document["seed"]))
    seen_samples, sessions, flat_splits = set(), {}, {}
    document_hashes = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            raise _error("invalid_dataset", "samples must be objects")
        identifier, split = sample["id"], sample["split"]
        if identifier in seen_samples or document_splits.get(sample["document_group"]) != split:
            raise _error("split_leakage", "sample split disagrees with its document")
        seen_samples.add(identifier)
        session = sample["capture_session"]
        if session in sessions and sessions[session] != split:
            raise _error("split_leakage", "capture session occurs in multiple splits")
        sessions[session] = split
        for name in ("flat.png", "photo.png", "geometry.npz", "metadata.json"):
            path = (root / sample["paths"][name]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise _error("invalid_dataset", "artifact path is missing or escapes dataset root")
            if _sha256(path) != sample["sha256"][name]:
                raise _error("artifact_hash_mismatch", f"artifact changed: {identifier}/{name}")
        flat_hash = sample["sha256"]["flat.png"]
        if flat_hash in flat_splits and flat_splits[flat_hash] != split:
            raise _error("split_leakage", "identical flat document occurs in multiple splits")
        flat_splits[flat_hash] = split
        group = sample["document_group"]
        if group in document_hashes and document_hashes[group] != flat_hash:
            raise _error("invalid_dataset", "variants of a document must use the same flat image")
        document_hashes[group] = flat_hash
        counts[split] += 1
    if not all(counts.values()):
        raise _error("invalid_dataset", "all three splits must contain samples")
    return {"sample_counts": counts, "documents": len(document_splits), "manifest": manifest}


def validate_dataset(directory: str | os.PathLike[str]) -> dict:
    """Check completion, artifact hashes, and split leakage before consumption."""
    try:
        return _validate_dataset(directory)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise _error("invalid_dataset", f"Malformed dataset manifest or artifact: {exc}") from exc


def load_sample(directory: str | os.PathLike[str], record: dict) -> SyntheticSample:
    """Load a sample after validate_dataset(); never deserialize pickled arrays."""
    root = Path(directory).resolve()

    def path(name):
        resolved = (root / record["paths"][name]).resolve()
        if not resolved.is_relative_to(root):
            raise _error("invalid_dataset", "artifact escapes dataset root")
        return resolved

    try:
        with Image.open(path("flat.png")) as encoded:
            flat = np.array(encoded, copy=True)
        with Image.open(path("photo.png")) as encoded:
            photo = np.array(encoded, copy=True)
        with np.load(path("geometry.npz"), allow_pickle=False) as archive:
            backward_map = archive["backward_map"]
            valid_mask = archive["valid_mask"]
            page_mask = archive["photo_page_mask"]
        metadata = json.loads(path("metadata.json").read_text(encoding="utf-8"))
        validate_image(flat, RenderConfig())
        validate_image(photo, RenderConfig())
        if (
            backward_map.dtype != np.float32
            or backward_map.shape != (*flat.shape[:2], 2)
            or valid_mask.dtype != bool
            or valid_mask.shape != flat.shape[:2]
            or page_mask.dtype != bool
            or page_mask.shape != photo.shape[:2]
            or not np.any(valid_mask)
            or not np.all(np.isfinite(backward_map[valid_mask]))
            or np.any(np.abs(backward_map[valid_mask]) > 1)
            or not np.all(np.isnan(backward_map[~valid_mask]))
        ):
            raise _error(
                "invalid_dataset", "invalid geometric shapes, types, or missing-value mask"
            )
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != record[key]
            for key in ("document_group", "capture_session", "split", "family")
        ):
            raise _error("invalid_dataset", "sample metadata disagrees with manifest")
        if (
            metadata.get("sample_id") != record["id"]
            or metadata.get("document_id") != record["document_group"]
            or metadata.get("output_size") != list(flat.shape[:2])
            or metadata.get("photo_size") != list(photo.shape[:2])
        ):
            raise _error("invalid_dataset", "sample identity or dimensions disagree with artifacts")
        return SyntheticSample(flat, photo, backward_map, valid_mask, page_mask, metadata)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        raise _error("invalid_dataset", f"Cannot load sample: {exc}") from exc
