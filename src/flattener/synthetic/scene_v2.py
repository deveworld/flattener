"""Area-stratified printed-document scenes, separate from scene-0.1.

The target is a page containing printed content. Unprinted white/colored panels
are deliberately negatives; this is not a detector for arbitrary blank paper.
All backgrounds/panels/text are project-authored procedures, not photographs.
"""

from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from ..contracts import FlattenerError, RenderConfig, Status
from ..renderer import render
from .documents import generate_document
from .geometry import WarpParameters, make_geometry, sample_warp
from .pipeline import PhotometricConfig, _photometric_config, apply_photometry, derive_seed
from .scenes import (
    _APPEARANCE,
    BACKGROUND_TYPES,
    SCENE_FAMILIES,
    SceneSample,
    _artifact_paths,
    _bbox,
    _json,
    _seed,
    _sha256,
    _size,
    generate_background,
)

SCENE_V2_SCHEMA_VERSION = "scene-0.2"
SIZE_BINS = ("small", "medium", "large")
NEGATIVE_TYPES = ("background", "paper_panel")
TASK_SCOPE = (
    "Printed-document page-region detection: a page with authored printed text is positive; "
    "unprinted white/colored panels are negative. Not arbitrary blank-paper detection."
)
_SPLITS = ("train", "validation", "test")
_ARTIFACTS = ("image.png", "mask.png", "metadata.json")
_MARGIN_BOUND = 0.985
_MAX_ATTEMPTS = 256


@dataclass(frozen=True, slots=True)
class SceneV2Config:
    seed: int = 20260923
    train_documents: int = 96
    validation_documents: int = 24
    test_documents: int = 24
    variants_per_document: int = 4
    image_size: tuple[int, int] = (192, 144)
    negative_fraction: float = 1 / 3


def _invalid(message, reason="invalid_scene_v2_dataset"):
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def _negative_count(positives, fraction):
    return math.floor(positives * fraction / (1 - fraction) + 0.5)


def _config(config):
    if not isinstance(config, SceneV2Config):
        raise _invalid("config must be SceneV2Config")
    _seed(config.seed)
    _size(config.image_size)
    for name in (
        "train_documents",
        "validation_documents",
        "test_documents",
        "variants_per_document",
    ):
        value = getattr(config, name)
        if type(value) is not int or not 1 <= value <= 10_000:
            raise _invalid(f"{name} must be an integer in [1, 10000]")
    fraction = config.negative_fraction
    if (
        type(fraction) not in (int, float)
        or not 0 <= fraction <= 0.75
        or not math.isfinite(fraction)
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


def size_bin_for_fraction(fraction: float) -> str:
    """Classify positive GT area; boundaries are [0,.25), [.25,.5), [.5,1]."""
    if type(fraction) not in (int, float) or not 0 < fraction <= 1 or not math.isfinite(fraction):
        raise _invalid("foreground fraction must be a finite positive fraction <=1")
    return "small" if fraction < 0.25 else "medium" if fraction < 0.5 else "large"


def _candidate_warp(family: str, size_bin: str, seed: int) -> WarpParameters:
    rng = np.random.default_rng(seed)
    base = sample_warp(family, derive_seed(seed, "v2-base-warp"))
    ranges = {
        "small": (0.35, 0.64, 0.4, 0.06),
        "medium": (0.72, 0.9, 0.12, 0.035),
        "large": (0.94, 0.978, 0.02, 0.008),
    }
    low, high, rotation_limit, perspective_limit = ranges[size_bin]
    rotation = float(rng.uniform(-rotation_limit, rotation_limit))
    px, py = rng.uniform(-perspective_limit, perspective_limit, size=2)
    ub, vb = 1 + abs(base.shear_x), 1 + abs(base.bend_y)
    # Compensate for the known continuous shear bounds before fitting a box.
    # This permits genuinely large strong/local pages without clipping them.
    sx, sy = rng.uniform(0.92, 1.0, size=2) / np.array((ub, vb))
    denominator = 1 - abs(px) * ub - abs(py) * vb
    c, s = abs(np.cos(rotation)), abs(np.sin(rotation))
    extents = np.array((sx * c * ub + sy * s * vb, sx * s * ub + sy * c * vb))
    multiplier = float(rng.uniform(low, high)) * denominator / float(extents.max())
    sx, sy = float(sx * multiplier), float(sy * multiplier)
    limits = (_MARGIN_BOUND - extents * multiplier / denominator) * denominator
    tx, ty = rng.uniform(-limits, limits)
    return replace(
        base,
        scale_x=sx,
        scale_y=sy,
        rotation=rotation,
        perspective_x=float(px),
        perspective_y=float(py),
        translate_x=float(tx),
        translate_y=float(ty),
    )


def sample_stratified_geometry(
    family: str,
    size_bin: str,
    seed: int,
    image_size: tuple[int, int] = (192, 144),
):
    """Return parameters, two-direction geometry and bounded rejection metadata.

    Assignment is based on the actual final source-grid GT mask area, never a
    proxy scale/bounding box. Every attempt has a domain-separated seed. A
    256-attempt failure raises explicitly instead of relabeling or looping.
    """
    _seed(seed)
    image_size = _size(image_size)
    if family not in SCENE_FAMILIES or size_bin not in SIZE_BINS:
        raise _invalid("family and size_bin must be supported values")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        attempt_seed = derive_seed(seed, "scene-v2-geometry-attempt", str(attempt))
        params = _candidate_warp(family, size_bin, attempt_seed)
        geometry = make_geometry(image_size, image_size, params)
        fraction = float(geometry.photo_page_mask.mean())
        if fraction > 0 and size_bin_for_fraction(fraction) == size_bin:
            return (
                params,
                geometry,
                {
                    "requested_size_bin": size_bin,
                    "actual_size_bin": size_bin,
                    "foreground_fraction": fraction,
                    "attempts": attempt,
                    "maximum_attempts": _MAX_ATTEMPTS,
                    "accepted_attempt_seed": attempt_seed,
                    "continuous_page_absolute_bound": _MARGIN_BOUND,
                    "criterion": "Actual analytic source-grid mask.mean(), not bounding-box area",
                },
            )
    raise _invalid(
        f"Could not generate {family}/{size_bin} geometry in {_MAX_ATTEMPTS} attempts",
        "scene_geometry_sampling_failed",
    )


def _composite_page(paper, background, params):
    height, width = background.shape[:2]
    high = make_geometry(paper.shape[:2], (height * 2, width * 2), params)
    sampled = render(
        paper,
        high.synthesis_map,
        target_mask=high.photo_page_mask,
        config=RenderConfig(fill_rgb=(0, 0, 0), max_out_of_bounds_fraction=1),
    )
    premultiplied = (
        sampled.image.astype(np.float32).reshape(height, 2, width, 2, 3).mean(axis=(1, 3))
    )
    alpha = high.photo_page_mask.reshape(height, 2, width, 2).mean(axis=(1, 3))
    result = premultiplied + background.astype(np.float32) * (1 - alpha[..., None])
    return np.floor(np.clip(result, 0, 255) + 0.5).astype(np.uint8)


def _blank_panel(background, scene_seed, color_mode):
    """An unprinted panel: paper-like appearance but no printed-page target."""
    rng = np.random.default_rng(derive_seed(scene_seed, "v2-panel-color"))
    color = (
        np.repeat(rng.integers(220, 256), 3)
        if color_mode == "white"
        else rng.integers(110, 241, size=3)
    )
    panel = np.empty_like(background)
    panel[:] = color
    panel_seed = derive_seed(scene_seed, "v2-blank-panel-geometry")
    size_bin = SIZE_BINS[panel_seed % len(SIZE_BINS)]
    params, geometry, sampling = sample_stratified_geometry(
        "perspective",
        size_bin,
        panel_seed,
        background.shape[:2],
    )
    image = _composite_page(panel, background, params)
    return image, {
        "kind": "unprinted paper-like panel",
        "printed_content": False,
        "color_mode": color_mode,
        "rgb": color.tolist(),
        "geometry": geometry.metadata,
        "sampling": sampling,
        "is_target": False,
        "source": "Project-authored solid-color geometric panel; no external image",
        "license": "Apache-2.0",
        "mask_semantics": "Panel is excluded from printed-page GT",
    }


def generate_scene_v2(
    document_id: str,
    document_seed: int,
    scene_seed: int,
    *,
    image_size: tuple[int, int] = (192, 144),
    size_bin: str = "medium",
    family: str = "gentle",
    has_page: bool = True,
    negative_type: str = "background",
    panel_color_mode: str = "white",
    background_type: str | None = None,
    background_seed: int | None = None,
    photometric_seed: int | None = None,
    photometric: PhotometricConfig | None = None,
) -> SceneSample:
    """One deterministic image and printed-page binary GT with complete provenance."""
    _seed(document_seed)
    _seed(scene_seed)
    image_size = _size(image_size)
    if not isinstance(document_id, str) or not document_id.strip():
        raise _invalid("document_id must be a nonempty string")
    if type(has_page) is not bool or size_bin not in SIZE_BINS or family not in SCENE_FAMILIES:
        raise _invalid("invalid has_page, size_bin or family")
    if negative_type not in NEGATIVE_TYPES or panel_color_mode not in ("white", "colored"):
        raise _invalid("invalid negative type or panel color mode")
    appearance = _APPEARANCE if photometric is None else photometric
    try:
        _photometric_config(appearance)
    except (OverflowError, TypeError, ValueError) as error:
        # The shared v1 validator can overflow on arbitrary-size Python ints.
        # Keep v1 unchanged and normalize only invalid v2 configuration input.
        raise _invalid(f"invalid photometric configuration: {error}") from error
    background_seed = (
        derive_seed(scene_seed, "v2-background")
        if background_seed is None
        else _seed(background_seed)
    )
    photometric_seed = (
        derive_seed(scene_seed, "v2-photometry")
        if photometric_seed is None
        else _seed(photometric_seed)
    )
    background, background_metadata = generate_background(
        image_size, background_seed, background_type
    )
    mask = np.zeros(image_size, dtype=bool)
    metadata = {
        "schema_version": SCENE_V2_SCHEMA_VERSION,
        "task_scope": TASK_SCOPE,
        "document_id": document_id,
        "has_page": has_page,
        "family": family if has_page else "none",
        "size_bin": size_bin if has_page else "negative",
        "negative_type": None if has_page else negative_type,
        "panel_color_mode": panel_color_mode
        if not has_page and negative_type == "paper_panel"
        else None,
        "background_type": background_metadata["type"],
        "background": background_metadata,
        "image_size": list(image_size),
        "is_real_photo": False,
        "seeds": {
            "document": document_seed,
            "scene": scene_seed,
            "background": background_seed,
            "photometric": photometric_seed,
        },
        "source": "Project-authored printed documents, procedural backgrounds, unprinted panels",
        "terms": "Authored text/images/code Apache-2.0; unmodified bundled font OFL-1.1",
        "mask_definition": (
            "Analytic pixel-center membership of printed document only; blank panels excluded"
        ),
        "bbox_definition": "[left, top, right-exclusive, bottom-exclusive] of all true GT pixels",
        "geometry": None,
        "sampling": None,
        "document": None,
        "paper": None,
        "panel": None,
    }
    if has_page:
        document = generate_document(document_id, document_seed, image_size)
        paper_seed = derive_seed(scene_seed, "v2-paper")
        rng = np.random.default_rng(paper_seed)
        tint = rng.uniform(0.55, 1.02, size=3)
        if rng.random() < 0.7:
            tint = rng.uniform(0.65, 1.02) * rng.uniform(0.94, 1.03, size=3)
        paper = np.floor(np.clip(document.image * tint, 0, 255) + 0.5).astype(np.uint8)
        warp_seed = derive_seed(scene_seed, "v2-warp")
        params, geometry, sampling = sample_stratified_geometry(
            family, size_bin, warp_seed, image_size
        )
        scene = _composite_page(paper, background, params)
        mask = geometry.photo_page_mask
        metadata.update(
            document=document.metadata,
            geometry=geometry.metadata,
            sampling=sampling,
            paper={"seed": paper_seed, "tint_rgb_multiplier": tint.tolist()},
        )
        metadata["seeds"].update(paper=paper_seed, warp=warp_seed)
    elif negative_type == "paper_panel":
        scene, metadata["panel"] = _blank_panel(background, scene_seed, panel_color_mode)
    else:
        scene = background
    image, appearance_metadata = apply_photometry(scene, photometric_seed, appearance)
    metadata.update(
        photometric=appearance_metadata,
        bbox=_bbox(mask),
        foreground_fraction=float(mask.mean()),
        compositing={
            "supersampling": 2 if has_page or negative_type == "paper_panel" else 0,
            "page_render_calls": 1 if has_page else 0,
            "blank_panel_render_calls": 1 if not has_page and negative_type == "paper_panel" else 0,
            "alpha": "Mean analytic subpixel occupancy, premultiplied black fill",
        },
    )
    return SceneSample(image, mask, metadata)


def _counts(positive_count, negative_count):
    return {
        "page": positive_count,
        "negative": negative_count,
        "total": positive_count + negative_count,
        "realized_negative_fraction": negative_count / (positive_count + negative_count),
        "size_bins": {
            size_bin: (positive_count + 2 - index) // 3 for index, size_bin in enumerate(SIZE_BINS)
        },
        "negative_types": {
            "background": (negative_count + 1) // 2,
            "paper_panel": negative_count // 2,
        },
    }


def generate_scene_v2_dataset(
    output_dir: str | os.PathLike[str], config: SceneV2Config | None = None
) -> dict:
    """Generate new split-first groups; test artifacts are written, never selected.

    Each split cycles positive size bins small/medium/large, giving exactly equal
    counts at default sizes. Negative backgrounds/panels alternate. Counts that
    are not divisible are balanced with at most one extra in the earlier bins.
    """
    config = _config(SceneV2Config() if config is None else config)
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise FlattenerError(Status.IO_ERROR, "dataset_create_failed", str(error)) from error
    groups, samples, counts = [], [], {}
    group_index = 0
    for split, document_count in zip(
        _SPLITS,
        (config.train_documents, config.validation_documents, config.test_documents),
        strict=True,
    ):
        positive_count = document_count * config.variants_per_document
        negative_count = _negative_count(positive_count, config.negative_fraction)
        counts[split] = _counts(positive_count, negative_count)
        specs = []
        for local_index in range(document_count):
            group_id = f"scene-v2-document-{group_index:06d}"
            group_index += 1
            specs.append((group_id, True, config.variants_per_document, local_index))
        specs.extend(
            (f"scene-v2-negative-{split}-{index:06d}", False, 1, index)
            for index in range(negative_count)
        )
        for group_id, has_page, variants, index in specs:
            document_seed = derive_seed(config.seed, "scene-v2-document", group_id)
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
                size_bin = SIZE_BINS[(index * variants + variant) % len(SIZE_BINS)]
                family = SCENE_FAMILIES[
                    derive_seed(config.seed, "v2-family", identifier) % len(SCENE_FAMILIES)
                ]
                background_type = BACKGROUND_TYPES[
                    derive_seed(config.seed, "v2-background-type", identifier)
                    % len(BACKGROUND_TYPES)
                ]
                negative_type = NEGATIVE_TYPES[index % 2]
                panel_color_mode = ("white", "colored")[(index // 2) % 2]
                scene_seed = derive_seed(config.seed, "scene-v2-sample", identifier)
                sample = generate_scene_v2(
                    group_id,
                    document_seed,
                    scene_seed,
                    image_size=config.image_size,
                    size_bin=size_bin,
                    family=family,
                    has_page=has_page,
                    negative_type=negative_type,
                    panel_color_mode=panel_color_mode,
                    background_type=background_type,
                )
                session = f"procedural-v2:{identifier}"
                sample.metadata.update(
                    sample_id=identifier,
                    document_group=group_id,
                    capture_session=session,
                    split=split,
                )
                sample_dir = directory / "samples" / identifier
                sample_dir.mkdir(parents=True)
                with Image.fromarray(sample.image) as encoded:
                    encoded.save(sample_dir / "image.png")
                with Image.fromarray(sample.mask.astype(np.uint8) * 255) as encoded:
                    encoded.save(sample_dir / "mask.png")
                (sample_dir / "metadata.json").write_bytes(_json(sample.metadata))
                paths = {name: f"samples/{identifier}/{name}" for name in _ARTIFACTS}
                record = {
                    key: sample.metadata[key]
                    for key in (
                        "document_group",
                        "capture_session",
                        "split",
                        "has_page",
                        "family",
                        "background_type",
                        "size_bin",
                        "negative_type",
                        "panel_color_mode",
                    )
                }
                record.update(
                    id=identifier,
                    paths=paths,
                    sha256={name: _sha256(directory / path) for name, path in paths.items()},
                )
                samples.append(record)
    manifest = {
        "schema_version": SCENE_V2_SCHEMA_VERSION,
        "complete": True,
        "config": asdict(config),
        "task_scope": TASK_SCOPE,
        "groups": groups,
        "samples": samples,
        "counts": counts,
        "split_unit": (
            "Positive source document / negative procedural group; all captures stay in one split"
        ),
        "size_bin_definition": {
            "small": "0 < actual GT fraction < .25",
            "medium": ".25 <= actual GT fraction < .5",
            "large": ".5 <= actual GT fraction <= 1",
        },
        "size_assignment": (
            "Cycle small/medium/large independently within each split, before generating pixels"
        ),
        "negative_assignment": (
            "Alternate background/paper_panel; paper panels alternate white/colored"
        ),
        "negative_count_rule": "floor(positive_count * fraction / (1-fraction) + 0.5)",
        "geometry_attempt_limit": _MAX_ATTEMPTS,
        "is_real_photo": False,
        "external_images": False,
        "source": (
            "Project-authored bilingual content, existing unmodified Noto font, analytic warps, "
            "procedural backgrounds/panels"
        ),
        "terms": "Authored text/images/code Apache-2.0; bundled font OFL-1.1",
        "generator_code_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                "scene_v2.py",
                "scenes.py",
                "documents.py",
                "geometry.py",
                "pipeline.py",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
    }
    (directory / "manifest.json").write_bytes(_json(manifest))
    validate_scene_v2_dataset(directory)
    return manifest


def _metadata(record, paths):
    metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCENE_V2_SCHEMA_VERSION:
        raise _invalid("metadata must be a scene-0.2 object")
    if metadata.get("task_scope") != TASK_SCOPE or metadata.get("is_real_photo") is not False:
        raise _invalid("printed-document task scope or synthetic provenance is missing")
    for field in (
        "document_group",
        "capture_session",
        "split",
        "has_page",
        "family",
        "background_type",
        "size_bin",
        "negative_type",
        "panel_color_mode",
    ):
        if metadata.get(field) != record[field]:
            raise _invalid(f"metadata disagrees with record: {field}")
    if (
        metadata.get("sample_id") != record["id"]
        or metadata.get("document_id") != record["document_group"]
    ):
        raise _invalid("metadata identity disagrees with record")
    if (
        type(record.get("has_page")) is not bool
        or record.get("background_type") not in BACKGROUND_TYPES
    ):
        raise _invalid("invalid page flag or background type")
    fraction = metadata["foreground_fraction"]
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1 or not math.isfinite(fraction):
        raise _invalid("recorded GT fraction must be finite and in [0, 1]")
    if record["has_page"]:
        if record["family"] not in SCENE_FAMILIES or record["size_bin"] != size_bin_for_fraction(
            fraction
        ):
            raise _invalid("positive family or size bin disagrees with recorded GT fraction")
        if (
            record["negative_type"] is not None
            or record["panel_color_mode"] is not None
            or metadata["panel"] is not None
        ):
            raise _invalid("positive sample must not claim negative content")
        geometry = metadata["geometry"]
        if (
            geometry["family"] != record["family"]
            or geometry["parameters"]["family"] != record["family"]
            or geometry["photo_page_fraction"] != fraction
        ):
            raise _invalid("geometry family or mask area disagrees with sample")
        sampling = metadata["sampling"]
        if (
            sampling["actual_size_bin"] != record["size_bin"]
            or sampling["requested_size_bin"] != record["size_bin"]
            or sampling["foreground_fraction"] != fraction
            or type(sampling["attempts"]) is not int
            or not 1 <= sampling["attempts"] <= _MAX_ATTEMPTS
            or sampling["maximum_attempts"] != _MAX_ATTEMPTS
        ):
            raise _invalid("invalid bounded geometry sampling record")
    else:
        if (
            record["size_bin"] != "negative"
            or record["family"] != "none"
            or record["negative_type"] not in NEGATIVE_TYPES
        ):
            raise _invalid("invalid negative type or size bin")
        if (
            fraction != 0
            or metadata["bbox"] is not None
            or metadata["document"] is not None
            or metadata["geometry"] is not None
            or metadata["sampling"] is not None
            or metadata["paper"] is not None
        ):
            raise _invalid("negative scene cannot have printed-document GT")
        if record["negative_type"] == "paper_panel":
            panel = metadata["panel"]
            if (
                record["panel_color_mode"] not in ("white", "colored")
                or not isinstance(panel, dict)
                or panel.get("printed_content") is not False
                or panel.get("is_target") is not False
                or panel.get("color_mode") != record["panel_color_mode"]
            ):
                raise _invalid(
                    "blank panels must explicitly exclude printed-page content and targets"
                )
        elif metadata["panel"] is not None or record["panel_color_mode"] is not None:
            raise _invalid("background-only negative must not claim a panel")
    return metadata


def _validate(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCENE_V2_SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("task_scope") != TASK_SCOPE
    ):
        raise _invalid("unsupported, incomplete or unscoped scene-0.2 dataset")
    fields = dict(manifest["config"])
    fields["image_size"] = tuple(fields["image_size"])
    config = _config(SceneV2Config(**fields))
    if not isinstance(manifest.get("groups"), list) or not isinstance(
        manifest.get("samples"), list
    ):
        raise _invalid("groups/samples must be lists")
    groups, group_counts, seeds = {}, {}, set()
    for group in manifest["groups"]:
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("id"), str)
            or not group["id"]
            or group["id"] in groups
            or group.get("split") not in _SPLITS
            or type(group.get("has_page")) is not bool
        ):
            raise _invalid("invalid or duplicate source group", "split_leakage")
        seed = _seed(group["seed"])
        if seed in seeds or seed != derive_seed(config.seed, "scene-v2-document", group["id"]):
            raise _invalid(
                "source seeds must be unique, v2-domain-separated and reproducible", "split_leakage"
            )
        if group["variants"] != (config.variants_per_document if group["has_page"] else 1):
            raise _invalid("source-group variant count disagrees with config")
        groups[group["id"]], group_counts[group["id"]] = group, 0
        seeds.add(seed)
    seen, sessions, artifacts, image_splits, source_splits, source_hashes = (
        set(),
        {},
        set(),
        {},
        {},
        {},
    )
    positive_counts, negative_counts = dict.fromkeys(_SPLITS, 0), dict.fromkeys(_SPLITS, 0)
    bins = {split: dict.fromkeys(SIZE_BINS, 0) for split in _SPLITS}
    negatives = {split: dict.fromkeys(NEGATIVE_TYPES, 0) for split in _SPLITS}
    for record in manifest["samples"]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
        ):
            raise _invalid("sample must have a nonempty string ID")
        group, split, identifier = (
            groups.get(record["document_group"]),
            record["split"],
            record["id"],
        )
        if (
            identifier in seen
            or group is None
            or group["split"] != split
            or group["has_page"] != record["has_page"]
        ):
            raise _invalid("sample disagrees with source group or duplicates ID", "split_leakage")
        seen.add(identifier)
        session = record["capture_session"]
        if (
            not isinstance(session, str)
            or not session
            or session in sessions
            and sessions[session] != split
        ):
            raise _invalid("invalid capture session or session crossing splits", "split_leakage")
        sessions[session] = split
        paths = _artifact_paths(root, record)
        for path in paths.values():
            if path in artifacts:
                raise _invalid("artifact path shared by multiple samples")
            artifacts.add(path)
        metadata = _metadata(record, paths)
        if (
            metadata["image_size"] != list(config.image_size)
            or metadata["seeds"]["document"] != group["seed"]
        ):
            raise _invalid("metadata dimensions or source seed disagree")
        if metadata["seeds"]["scene"] != derive_seed(config.seed, "scene-v2-sample", identifier):
            raise _invalid("sample seed is not reproducible from v2 domain")
        image_hash = record["sha256"]["image.png"]
        if image_hash in image_splits and image_splits[image_hash] != split:
            raise _invalid("identical scene crosses splits", "split_leakage")
        image_splits[image_hash] = split
        if record["has_page"]:
            document = metadata["document"]
            source_hash = document["source_sha256"]
            if (
                document["document_id"] != group["id"]
                or document["seed"] != group["seed"]
                or source_hash in source_splits
                and source_splits[source_hash] != split
            ):
                raise _invalid(
                    "source document identity/seed mismatch or cross-split duplicate",
                    "split_leakage",
                )
            if group["id"] in source_hashes and source_hashes[group["id"]] != source_hash:
                raise _invalid("all variants must share the same authored document")
            source_hashes[group["id"]], source_splits[source_hash] = source_hash, split
            positive_counts[split] += 1
            bins[split][record["size_bin"]] += 1
        else:
            negative_counts[split] += 1
            negatives[split][record["negative_type"]] += 1
        group_counts[group["id"]] += 1
    if any(group_counts[key] != group["variants"] for key, group in groups.items()):
        raise _invalid("missing or extra source-group variants")
    for split, document_count in zip(
        _SPLITS,
        (config.train_documents, config.validation_documents, config.test_documents),
        strict=True,
    ):
        positive = document_count * config.variants_per_document
        negative = _negative_count(positive, config.negative_fraction)
        expected = _counts(positive, negative)
        if (
            positive_counts[split] != positive
            or negative_counts[split] != negative
            or bins[split] != expected["size_bins"]
            or negatives[split] != expected["negative_types"]
            or manifest["counts"][split] != expected
        ):
            raise _invalid("sample/bin/type counts disagree with deterministic allocation")
    return {
        "manifest": manifest,
        "sample_counts": {
            split: positive_counts[split] + negative_counts[split] for split in _SPLITS
        },
        "page_counts": positive_counts,
        "negative_counts": negative_counts,
        "size_bin_counts": bins,
        "negative_type_counts": negatives,
    }


def validate_scene_v2_dataset(directory: str | os.PathLike[str]) -> dict:
    """Check hashes, exact strata, groups and provenance without decoding any GT."""
    try:
        return _validate(directory)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise _invalid(f"Malformed scene-v2 dataset: {error}") from error


def load_scene_v2(directory: str | os.PathLike[str], record: dict) -> SceneSample:
    """Load one selected record; independently check hashes, modes and analytic GT."""
    try:
        paths = _artifact_paths(Path(directory).resolve(), record)
        metadata = _metadata(record, paths)
        image_size = _size(tuple(metadata["image_size"]))
        with Image.open(paths["image.png"]) as encoded:
            if encoded.mode != "RGB" or encoded.size != (image_size[1], image_size[0]):
                raise _invalid("image must be RGB with recorded dimensions")
            image = np.array(encoded, copy=True)
        with Image.open(paths["mask.png"]) as encoded:
            if encoded.mode != "L" or encoded.size != (image_size[1], image_size[0]):
                raise _invalid("mask must be grayscale with recorded dimensions")
            values = np.array(encoded, copy=True)
        if values.dtype != np.uint8 or not np.isin(values, (0, 255)).all():
            raise _invalid("mask must contain uint8 0/255 values only")
        mask = values == 255
        if (
            bool(mask.any()) != record["has_page"]
            or _bbox(mask) != metadata["bbox"]
            or float(mask.mean()) != metadata["foreground_fraction"]
        ):
            raise _invalid("mask differs from recorded presence, bbox or area")
        if record["has_page"]:
            if size_bin_for_fraction(float(mask.mean())) != record["size_bin"]:
                raise _invalid("actual source-grid GT area is outside requested size bin")
            params = WarpParameters(**metadata["geometry"]["parameters"])
            expected = make_geometry(image_size, image_size, params).photo_page_mask
            if not np.array_equal(mask, expected):
                raise _invalid("stored GT mask disagrees with analytic geometry")
        return SceneSample(image, mask, metadata)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise _invalid(f"Cannot load scene-v2 sample: {error}") from error
