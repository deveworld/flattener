"""Self-authored, split-safe page-region scenes and procedural negatives.

This is a deliberately separate distribution from synthetic-0.1. Its masks
label geometric page membership at scene pixel centers, not paper whiteness,
text, shadows, or alpha coverage. Backgrounds and negative scenes are authored
procedures, never real photographs. They cannot establish real-world accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image, ImageDraw

from ..contracts import FlattenerError, RenderConfig, Status
from ..renderer import render
from .documents import generate_document
from .geometry import WarpParameters, make_geometry, sample_warp
from .pipeline import PhotometricConfig, apply_photometry, derive_seed

SCENE_SCHEMA_VERSION = "scene-0.1"
BACKGROUND_TYPES = ("gradient", "stripes", "wood", "tile", "shapes")
SCENE_FAMILIES = ("perspective", "gentle", "strong", "local")
_SPLITS = ("train", "validation", "test")
_ARTIFACTS = ("image.png", "mask.png", "metadata.json")
_APPEARANCE = PhotometricConfig(
    brightness_min=0.65,
    brightness_max=1.15,
    max_shadow=0.35,
    max_blur_radius=0.65,
    noise_std=0.012,
)


@dataclass(frozen=True, slots=True)
class SceneConfig:
    seed: int = 20260922
    train_documents: int = 96
    validation_documents: int = 16
    test_documents: int = 16
    variants_per_document: int = 4
    image_size: tuple[int, int] = (192, 144)
    negative_fraction: float = 0.25


@dataclass(slots=True)
class SceneSample:
    image: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any]


def _invalid(message: str, reason: str = "invalid_scene_dataset") -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise _invalid("seed must be a nonboolean integer in [0, 2**63)")
    return value


def _size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or not 128 <= v <= 512 for v in value)
    ):
        raise _invalid("image_size must be a (height, width) tuple with sides in [128, 512]")
    return value


def _negative_count(positive_count: int, fraction: float) -> int:
    # Closest representable fraction without changing the positive document count.
    return math.floor(positive_count * fraction / (1 - fraction) + 0.5)


def _config(config: SceneConfig) -> SceneConfig:
    if not isinstance(config, SceneConfig):
        raise _invalid("config must be SceneConfig")
    _seed(config.seed)
    _size(config.image_size)
    for name in (
        "train_documents",
        "validation_documents",
        "test_documents",
        "variants_per_document",
    ):
        count = getattr(config, name)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10_000:
            raise _invalid(f"{name} must be an integer in [1, 10000]")
    fraction = config.negative_fraction
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 0.75
    ):
        raise _invalid("negative_fraction must be finite and in [0, 0.75]")
    total = sum(
        count * config.variants_per_document
        + _negative_count(count * config.variants_per_document, fraction)
        for count in (config.train_documents, config.validation_documents, config.test_documents)
    )
    if total > 100_000:
        raise _invalid("dataset exceeds the 100000-scene generation limit")
    return config


def generate_background(
    image_size: tuple[int, int],
    seed: int,
    background_type: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Procedural RGB texture; all randomness belongs to this local seed."""
    height, width = _size(image_size)
    rng = np.random.default_rng(_seed(seed))
    if background_type is None:
        background_type = BACKGROUND_TYPES[int(rng.integers(len(BACKGROUND_TYPES)))]
    if background_type not in BACKGROUND_TYPES:
        raise _invalid(f"background_type must be one of {BACKGROUND_TYPES}")
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x, y = x / width, y / height
    base = rng.uniform(20, 245, size=3)
    # Light backgrounds can be brighter than colored/dim paper. No white-mask rule.
    if rng.random() < 0.3:
        base[:] = rng.uniform(195, 248, size=3)
    color_delta = rng.uniform(-60, 60, size=3)
    angle = rng.uniform(-np.pi, np.pi)
    projected = x * np.cos(angle) + y * np.sin(angle)
    gradient = projected - projected.mean()
    values = base[None, None, :] + gradient[..., None] * color_delta
    frequency = float(rng.uniform(4, 22))
    phase = float(rng.uniform(-np.pi, np.pi))
    if background_type == "stripes":
        pattern = np.sin(projected * frequency * np.pi + phase)
        values += pattern[..., None] * rng.uniform(10, 65, size=3)
    elif background_type == "wood":
        bent_x = x + 0.055 * np.sin(y * 7 + phase) + 0.012 * np.sin(y * 29)
        pattern = np.sin(bent_x * frequency * np.pi + phase)
        grain = np.sin(bent_x * frequency * 6 * np.pi + np.sin(y * 14))
        values += (pattern * 15 + grain * 5)[..., None] * rng.uniform(0.4, 1.5, size=3)
    elif background_type == "tile":
        cols, rows = int(rng.integers(3, 10)), int(rng.integers(3, 10))
        tx, ty = x * cols + rng.random(), y * rows + rng.random()
        pattern = (np.floor(tx) + np.floor(ty)) % 2
        grout = (tx % 1 < 0.035) | (ty % 1 < 0.035)
        values += pattern[..., None] * rng.uniform(-35, 35, size=3)
        values[grout] *= float(rng.uniform(0.5, 0.85))
    noise = rng.normal(0, float(rng.uniform(0.5, 5)), values.shape)
    result = np.floor(np.clip(values + noise, 0, 255) + 0.5).astype(np.uint8)
    if background_type == "shapes":
        with Image.fromarray(result) as image:
            draw = ImageDraw.Draw(image)
            for index in range(int(rng.integers(5, 13))):
                x0, y0 = rng.uniform(-0.15, 0.9, size=2)
                dx, dy = rng.uniform(0.08, 0.5, size=2)
                bounds = (
                    round(x0 * width),
                    round(y0 * height),
                    round((x0 + dx) * width),
                    round((y0 + dy) * height),
                )
                color = tuple(int(v) for v in rng.integers(15, 251, size=3))
                if index % 2:
                    draw.ellipse(bounds, fill=color)
                else:
                    draw.rectangle(bounds, fill=color)
            result = np.array(image, copy=True)
    return result, {
        "type": background_type,
        "seed": seed,
        "source": "Project-authored procedural gradient/pattern/shapes; not a photograph",
        "license": "Apache-2.0",
        "base_rgb": base.tolist(),
        "frequency": frequency,
        "phase": phase,
        "angle_radians": float(angle),
    }


def sample_scene_warp(family: str, seed: int) -> WarpParameters:
    """Broader placement than v1, with a conservative continuous-page margin.

    Bounding the two analytic shears before projection bounds the whole page,
    including between corners. The homography denominator has a positive lower
    bound. Scale and translation are chosen against that bound, not a sampled
    contour; the full continuous page stays within [-0.97, 0.97] on both axes.
    """
    rng = np.random.default_rng(_seed(seed))
    base = sample_warp(family, derive_seed(seed, "base-warp"))
    rotation = float(rng.uniform(-0.5, 0.5))
    sx, sy = rng.uniform(0.65, 1.0, size=2)
    px, py = rng.uniform(-0.075, 0.075, size=2)
    u_bound, v_bound = 1 + abs(base.shear_x), 1 + abs(base.bend_y)
    denominator = 1 - abs(px) * u_bound - abs(py) * v_bound
    c, s = abs(np.cos(rotation)), abs(np.sin(rotation))
    extents = np.array((sx * c * u_bound + sy * s * v_bound, sx * s * u_bound + sy * c * v_bound))
    target_extent = float(rng.uniform(0.42, 0.91))
    multiplier = target_extent * denominator / float(extents.max())
    sx, sy = float(sx * multiplier), float(sy * multiplier)
    bounds = extents * multiplier / denominator
    translation_limits = (0.97 - bounds) * denominator
    tx, ty = rng.uniform(-translation_limits, translation_limits)
    return replace(
        base,
        scale_x=sx,
        scale_y=sy,
        rotation=rotation,
        translate_x=float(tx),
        translate_y=float(ty),
        perspective_x=float(px),
        perspective_y=float(py),
    )


def _bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def generate_scene(
    document_id: str,
    document_seed: int,
    scene_seed: int,
    *,
    image_size: tuple[int, int] = (192, 144),
    has_page: bool = True,
    family: str = "gentle",
    background_type: str | None = None,
    background_seed: int | None = None,
    photometric_seed: int | None = None,
    photometric: PhotometricConfig | None = None,
) -> SceneSample:
    """Render a page once at 2x size, then composite premultiplied coverage.

    ``mask`` is analytic pixel-center membership. A 2x2 subpixel mean supplies
    boundary alpha independently; it is not mislabeled as the binary target.
    Photometry changes only appearance. Negative scenes skip document creation.
    """
    image_size = _size(image_size)
    _seed(document_seed)
    _seed(scene_seed)
    if not isinstance(document_id, str) or not document_id.strip():
        raise _invalid("document_id must be a nonempty string")
    if not isinstance(has_page, bool):
        raise _invalid("has_page must be boolean")
    if family not in SCENE_FAMILIES:
        raise _invalid(f"family must be one of {SCENE_FAMILIES}")
    background_seed = (
        derive_seed(scene_seed, "background") if background_seed is None else _seed(background_seed)
    )
    photometric_seed = (
        derive_seed(scene_seed, "photometric")
        if photometric_seed is None
        else _seed(photometric_seed)
    )
    background, background_meta = generate_background(image_size, background_seed, background_type)
    height, width = image_size
    mask = np.zeros(image_size, dtype=bool)
    metadata = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "document_id": document_id,
        "has_page": has_page,
        "family": family if has_page else "none",
        "background_type": background_meta["type"],
        "background": background_meta,
        "image_size": list(image_size),
        "is_real_photo": False,
        "seeds": {
            "document": document_seed,
            "scene": scene_seed,
            "background": background_seed,
            "photometric": photometric_seed,
        },
        "source": "Project-authored documents and procedural backgrounds only",
        "terms": "Authored text/images/code Apache-2.0; bundled font OFL-1.1",
        "negative_scope": "Procedural no-page scene; not a real-photo negative benchmark",
        "mask_definition": "Analytic page membership at scene pixel centers",
        "bbox_definition": "[left, top, right-exclusive, bottom-exclusive] of all true mask pixels",
        "geometry": None,
        "document": None,
        "paper": None,
    }
    if has_page:
        document = generate_document(document_id, document_seed, image_size)
        paper_seed = derive_seed(scene_seed, "paper")
        rng = np.random.default_rng(paper_seed)
        paper_tint = rng.uniform(0.55, 1.02, size=3)
        # Most pages remain near neutral, with a meaningful colored-paper tail.
        if rng.random() < 0.7:
            paper_tint = rng.uniform(0.65, 1.02) * rng.uniform(0.94, 1.03, size=3)
        paper = np.floor(np.clip(document.image * paper_tint, 0, 255) + 0.5).astype(np.uint8)
        warp_seed = derive_seed(scene_seed, "warp")
        params = sample_scene_warp(family, warp_seed)
        geometry = make_geometry(image_size, image_size, params)
        mask = geometry.photo_page_mask
        high_geometry = make_geometry(image_size, (height * 2, width * 2), params)
        result = render(
            paper,
            high_geometry.synthesis_map,
            target_mask=high_geometry.photo_page_mask,
            config=RenderConfig(fill_rgb=(0, 0, 0), max_out_of_bounds_fraction=1),
        )
        # Zero fill makes the sampled page premultiplied at subpixel resolution.
        page = result.image.astype(np.float32).reshape(height, 2, width, 2, 3).mean(axis=(1, 3))
        alpha = high_geometry.photo_page_mask.reshape(height, 2, width, 2).mean(axis=(1, 3))
        scene = page + background.astype(np.float32) * (1 - alpha[..., None])
        scene = np.floor(np.clip(scene, 0, 255) + 0.5).astype(np.uint8)
        metadata.update(
            document=document.metadata,
            geometry=geometry.metadata,
            paper={"seed": paper_seed, "tint_rgb_multiplier": paper_tint.tolist()},
            compositing={
                "supersampling": 2,
                "alpha": "Mean analytic subpixel occupancy",
                "page_render_calls": 1,
                "premultiplied_black_fill": True,
            },
        )
        metadata["seeds"]["warp"] = warp_seed
        metadata["seeds"]["paper"] = paper_seed
    else:
        scene = background
        metadata["compositing"] = {"supersampling": 0, "page_render_calls": 0}
    image, appearance = apply_photometry(
        scene,
        photometric_seed,
        _APPEARANCE if photometric is None else photometric,
    )
    metadata.update(
        photometric=appearance, foreground_fraction=float(mask.mean()), bbox=_bbox(mask)
    )
    return SceneSample(image, mask, metadata)


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_scene_dataset(
    output_dir: str | os.PathLike[str],
    config: SceneConfig | None = None,
) -> dict:
    """Create a new directory; publish the manifest only after full generation.

    Document counts are positive source groups. For each split, negative count
    is floor(positive_count * f / (1-f) + 0.5). The exact counts and realized
    fraction are recorded. Negative scenes each have their own source group.
    An interruption leaves inspectable partial files; nothing is overwritten.
    """
    config = _config(SceneConfig() if config is None else config)
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise FlattenerError(Status.IO_ERROR, "dataset_create_failed", str(exc)) from exc
    groups, samples, counts = [], [], {}
    document_index = 0
    for split, count in zip(
        _SPLITS,
        (config.train_documents, config.validation_documents, config.test_documents),
        strict=True,
    ):
        positives = count * config.variants_per_document
        negatives = _negative_count(positives, config.negative_fraction)
        counts[split] = {
            "page": positives,
            "negative": negatives,
            "total": positives + negatives,
            "realized_negative_fraction": negatives / (positives + negatives),
        }
        group_specs = []
        for _ in range(count):
            group_specs.append(
                (f"scene-document-{document_index:06d}", True, config.variants_per_document)
            )
            document_index += 1
        group_specs.extend(
            (f"scene-negative-{split}-{index:06d}", False, 1) for index in range(negatives)
        )
        for group_id, has_page, variants in group_specs:
            document_seed = derive_seed(config.seed, "scene-document", group_id)
            groups.append(
                {
                    "id": group_id,
                    "seed": document_seed,
                    "split": split,
                    "has_page": has_page,
                    "variants": variants,
                }
            )
            for variant in range(variants):
                identifier = f"{group_id}-view-{variant:03d}"
                family = SCENE_FAMILIES[
                    derive_seed(config.seed, "family", identifier) % len(SCENE_FAMILIES)
                ]
                background_type = BACKGROUND_TYPES[
                    derive_seed(config.seed, "background-type", identifier) % len(BACKGROUND_TYPES)
                ]
                sample = generate_scene(
                    group_id,
                    document_seed,
                    derive_seed(config.seed, "scene", identifier),
                    image_size=config.image_size,
                    has_page=has_page,
                    family=family,
                    background_type=background_type,
                )
                session = f"procedural:{identifier}"
                sample.metadata.update(
                    sample_id=identifier,
                    document_group=group_id,
                    capture_session=session,
                    split=split,
                )
                sample_dir = directory / "samples" / identifier
                sample_dir.mkdir(parents=True)
                with Image.fromarray(sample.image) as image:
                    image.save(sample_dir / "image.png")
                with Image.fromarray(sample.mask.astype(np.uint8) * 255) as mask:
                    mask.save(sample_dir / "mask.png")
                (sample_dir / "metadata.json").write_bytes(_json(sample.metadata))
                paths = {name: f"samples/{identifier}/{name}" for name in _ARTIFACTS}
                samples.append(
                    {
                        "id": identifier,
                        "document_group": group_id,
                        "capture_session": session,
                        "split": split,
                        "has_page": has_page,
                        "family": sample.metadata["family"],
                        "background_type": background_type,
                        "paths": paths,
                        "sha256": {name: _sha256(directory / path) for name, path in paths.items()},
                    }
                )
    manifest = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "complete": True,
        "config": asdict(config),
        "groups": groups,
        "samples": samples,
        "counts": counts,
        "split_unit": (
            "Positive document groups and negative procedural source groups; "
            "captures never cross splits"
        ),
        "negative_count_rule": (
            "floor(positive_samples * negative_fraction / (1-negative_fraction) + 0.5)"
        ),
        "is_real_photo": False,
        "external_images": False,
        "source": "Project-authored bilingual pages, analytic geometry, procedural backgrounds",
        "terms": "Authored text/images/code Apache-2.0; bundled font OFL-1.1",
        "generator_code_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("scenes.py", "documents.py", "geometry.py", "pipeline.py")
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
    }
    (directory / "manifest.json").write_bytes(_json(manifest))
    validate_scene_dataset(directory)
    return manifest


def _artifact_paths(root: Path, record: dict, *, check_hashes: bool = True) -> dict[str, Path]:
    paths = {}
    if not isinstance(record, dict):
        raise _invalid("sample record must be an object")
    if not isinstance(record.get("paths"), dict) or not isinstance(record.get("sha256"), dict):
        raise _invalid("sample paths and hashes must be objects")
    for name in _ARTIFACTS:
        relative, expected_hash = record["paths"][name], record["sha256"][name]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise _invalid("artifact paths must be relative strings")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise _invalid("artifact is missing or escapes the dataset root")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise _invalid("artifact SHA-256 must be a hexadecimal string")
        if check_hashes and _sha256(path) != expected_hash:
            raise _invalid(f"artifact changed: {record.get('id')}/{name}", "artifact_hash_mismatch")
        paths[name] = path
    return paths


def _metadata(record: dict, paths: dict[str, Path]) -> dict:
    metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCENE_SCHEMA_VERSION:
        raise _invalid("metadata must be a supported scene object")
    for field in (
        "document_group",
        "capture_session",
        "split",
        "has_page",
        "family",
        "background_type",
    ):
        if metadata.get(field) != record[field]:
            raise _invalid(f"metadata disagrees with record: {field}")
    if (
        metadata.get("sample_id") != record["id"]
        or metadata.get("document_id") != record["document_group"]
    ):
        raise _invalid("metadata sample or document identity disagrees with record")
    if metadata.get("is_real_photo") is not False:
        raise _invalid("scene metadata must identify synthetic provenance")
    return metadata


def _validate_scene_dataset(directory: str | os.PathLike[str]) -> dict:
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCENE_SCHEMA_VERSION
        or manifest.get("complete") is not True
    ):
        raise _invalid("unsupported or incomplete scene dataset")
    config_fields = dict(manifest["config"])
    config_fields["image_size"] = tuple(config_fields["image_size"])
    config = _config(SceneConfig(**config_fields))
    if not isinstance(manifest.get("groups"), list) or not isinstance(
        manifest.get("samples"), list
    ):
        raise _invalid("groups and samples must be lists")
    groups, seeds = {}, set()
    for group in manifest["groups"]:
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("id"), str)
            or not group["id"]
            or group["id"] in groups
            or group.get("split") not in _SPLITS
            or not isinstance(group.get("has_page"), bool)
        ):
            raise _invalid("invalid or duplicate source group", "split_leakage")
        seed = _seed(group["seed"])
        if seed in seeds:
            raise _invalid("source seeds must be unique", "split_leakage")
        if group["variants"] != (config.variants_per_document if group["has_page"] else 1):
            raise _invalid("source-group variants disagree with config")
        groups[group["id"]], seeds = group, seeds | {seed}
    seen, sessions, paths_seen, image_splits, document_splits = set(), {}, set(), {}, {}
    document_hashes = {}
    group_counts = dict.fromkeys(groups, 0)
    page_counts, negative_counts = dict.fromkeys(_SPLITS, 0), dict.fromkeys(_SPLITS, 0)
    for record in manifest["samples"]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
        ):
            raise _invalid("sample must have a nonempty string ID")
        identifier, group, split = (
            record["id"],
            groups.get(record["document_group"]),
            record["split"],
        )
        if identifier in seen or group is None or group["split"] != split:
            raise _invalid("sample split disagrees with source group", "split_leakage")
        seen.add(identifier)
        if not isinstance(record.get("has_page"), bool) or record["has_page"] != group["has_page"]:
            raise _invalid("sample page-presence flag disagrees with source group")
        if record["family"] not in (SCENE_FAMILIES if record["has_page"] else ("none",)):
            raise _invalid("invalid scene family")
        if record["background_type"] not in BACKGROUND_TYPES:
            raise _invalid("invalid background type")
        session = record["capture_session"]
        if not isinstance(session, str) or not session:
            raise _invalid("capture session must be a nonempty string")
        if session in sessions and sessions[session] != split:
            raise _invalid("capture session crosses splits", "split_leakage")
        sessions[session] = split
        paths = _artifact_paths(root, record)
        for path in paths.values():
            if path in paths_seen:
                raise _invalid("artifact path shared by multiple samples")
            paths_seen.add(path)
        metadata = _metadata(record, paths)
        if metadata.get("image_size") != list(config.image_size):
            raise _invalid("scene image size disagrees with config")
        if (
            not isinstance(metadata.get("seeds"), dict)
            or metadata["seeds"].get("document") != group["seed"]
        ):
            raise _invalid("scene document seed disagrees with source group")
        image_hash = record["sha256"]["image.png"]
        if image_hash in image_splits and image_splits[image_hash] != split:
            raise _invalid("identical scene image crosses splits", "split_leakage")
        image_splits[image_hash] = split
        if record["has_page"]:
            document = metadata["document"]
            original_hash = document["source_sha256"]
            if original_hash in document_splits and document_splits[original_hash] != split:
                raise _invalid("identical authored document crosses splits", "split_leakage")
            if (
                record["document_group"] in document_hashes
                and document_hashes[record["document_group"]] != original_hash
            ):
                raise _invalid("variants of a source group must share one authored document")
            document_hashes[record["document_group"]] = original_hash
            document_splits[original_hash] = split
            page_counts[split] += 1
        else:
            negative_counts[split] += 1
        group_counts[record["document_group"]] += 1
    if any(group_counts[key] != group["variants"] for key, group in groups.items()):
        raise _invalid("source group has missing or extra variants")
    for split, document_count in zip(
        _SPLITS,
        (config.train_documents, config.validation_documents, config.test_documents),
        strict=True,
    ):
        expected_positive = document_count * config.variants_per_document
        expected_negative = _negative_count(expected_positive, config.negative_fraction)
        expected = {
            "page": expected_positive,
            "negative": expected_negative,
            "total": expected_positive + expected_negative,
            "realized_negative_fraction": expected_negative
            / (expected_positive + expected_negative),
        }
        if page_counts[split] != expected_positive or negative_counts[split] != expected_negative:
            raise _invalid("sample counts disagree with configured source groups")
        if manifest["counts"][split] != expected:
            raise _invalid("manifest summary counts disagree with samples")
    return {
        "manifest": manifest,
        "sample_counts": {split: page_counts[split] + negative_counts[split] for split in _SPLITS},
        "page_counts": page_counts,
        "negative_counts": negative_counts,
    }


def validate_scene_dataset(directory: str | os.PathLike[str]) -> dict:
    """Check every hash and group, without decoding any split's image/GT mask.

    Consumers select train/validation/test records before ``load_scene``. Reading
    manifest/provenance and hashing files is not selection on held-out targets.
    """
    try:
        return _validate_scene_dataset(directory)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _invalid(f"Malformed scene dataset: {exc}") from exc


def load_scene(directory: str | os.PathLike[str], record: dict) -> SceneSample:
    """Load one selected record, independently verifying paths, hashes and GT."""
    try:
        root = Path(directory).resolve()
        paths = _artifact_paths(root, record)
        metadata = _metadata(record, paths)
        image_size = _size(tuple(metadata["image_size"]))
        with Image.open(paths["image.png"]) as encoded:
            if encoded.mode != "RGB" or encoded.size != (image_size[1], image_size[0]):
                raise _invalid("scene image must be RGB with the recorded dimensions")
            image = np.array(encoded, dtype=np.uint8, copy=True)
        with Image.open(paths["mask.png"]) as encoded:
            if encoded.mode != "L" or encoded.size != (image_size[1], image_size[0]):
                raise _invalid("mask must be grayscale with the recorded dimensions")
            values = np.array(encoded, copy=True)
        if not np.isin(values, (0, 255)).all():
            raise _invalid("mask pixels must be exactly 0 or 255")
        mask = values == 255
        if bool(mask.any()) != record["has_page"]:
            raise _invalid("mask page presence disagrees with the manifest")
        if metadata.get("bbox") != _bbox(mask) or metadata.get("foreground_fraction") != float(
            mask.mean()
        ):
            raise _invalid("mask statistics disagree with metadata")
        if record["has_page"]:
            params = WarpParameters(**metadata["geometry"]["parameters"])
            expected = make_geometry(image_size, image_size, params).photo_page_mask
            if not np.array_equal(mask, expected):
                raise _invalid("stored mask disagrees with its analytic geometry")
        elif metadata.get("geometry") is not None or metadata.get("document") is not None:
            raise _invalid("negative scene must not claim page geometry or document provenance")
        return SceneSample(image, mask, metadata)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _invalid(f"Cannot load scene: {exc}") from exc
