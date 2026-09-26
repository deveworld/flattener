"""Controlled printed/blank counterfactuals without changing the v1/v2 generators.

Each source document has four printed views and one blank variant-0 control.
The default control removes all authored printed raster marks while preserving
paper tint, analytic geometry, background and global photometry. Independent
training controls change only those blank training images using the v2 panel
procedure; validation and test remain exactly the same matched counterfactuals.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image

from ..contracts import FlattenerError, Status
from ..regions.segmentation import PREPROCESS_VERSION, SegmenterConfig, preprocess_image
from .geometry import WarpParameters, make_geometry
from .pipeline import PhotometricConfig, apply_photometry, derive_seed
from .scene_v2 import (
    SIZE_BINS,
    TASK_SCOPE,
    _composite_page,
    generate_scene_v2,
    size_bin_for_fraction,
)
from .scenes import (
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

PAIRED_SCHEMA_VERSION = "scene-0.3"
SCENE_SCHEMA_VERSION = PAIRED_SCHEMA_VERSION
_MANIFEST_LIMIT = 64 * 1024 * 1024
_METADATA_LIMIT = 1024 * 1024
_SPLITS = ("train", "validation", "test")
_ARTIFACTS = ("image.png", "mask.png", "metadata.json")
_RECORD_FIELDS = (
    "document_group",
    "capture_session",
    "split",
    "has_page",
    "family",
    "background_type",
    "size_bin",
    "negative_type",
    "panel_color_mode",
    "pair_id",
    "pair_role",
    "partner_id",
    "context_id",
)
_QA_CONFIG = SegmenterConfig(input_size=(128, 96))


@dataclass(frozen=True, slots=True)
class PairedSceneConfig:
    seed: int = 20260924
    train_documents: int = 96
    validation_documents: int = 24
    test_documents: int = 24
    variants_per_document: int = 4
    image_size: tuple[int, int] = (192, 144)
    training_panels: str = "matched"


def _invalid(message, reason="invalid_paired_scene_dataset"):
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def _config(config):
    if not isinstance(config, PairedSceneConfig):
        raise _invalid("config must be PairedSceneConfig")
    _seed(config.seed)
    _size(config.image_size)
    for name in ("train_documents", "validation_documents", "test_documents"):
        value = getattr(config, name)
        if type(value) is not int or not 1 <= value <= 10_000:
            raise _invalid(f"{name} must be an integer in [1, 10000]")
    if type(config.variants_per_document) is not int or config.variants_per_document != 4:
        raise _invalid("this controlled protocol requires exactly four printed views per document")
    if type(config.training_panels) is not str or config.training_panels not in (
        "matched",
        "independent",
    ):
        raise _invalid("training_panels must be matched or independent")
    if (config.train_documents + config.validation_documents + config.test_documents) * 6 > 100_000:
        raise _invalid("dataset exceeds the 100000-scene generation limit")
    return config


def _hash_json(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _qa_protocol():
    return {
        "version": PREPROCESS_VERSION,
        "input_size": [128, 96],
        "equality": "Exact CPU float32 tensor equality",
        "selection": "Count only; never filter or regenerate",
        "source": "In-memory generation arrays; no held-out files decoded",
    }


def _read_json(path, limit):
    def integer(value):
        if len(value.lstrip("-")) > 128:
            raise _invalid("JSON integer exceeds the bounded metadata precision limit")
        return int(value)

    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise _invalid("JSON metadata cannot contain nonfinite numbers")
        return result

    def constant(value):
        raise _invalid(f"JSON metadata cannot contain {value}")

    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise _invalid(f"JSON artifact exceeds its {limit}-byte size limit")
    return json.loads(payload, parse_int=integer, parse_float=floating, parse_constant=constant)


def _valid_hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _validate_bbox(value, size):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(edge) is not int for edge in value)
        or not 0 <= value[0] < value[2] <= size[1]
        or not 0 <= value[1] < value[3] <= size[0]
    ):
        raise _invalid(
            "bbox must be bounded integer [left, top, right-exclusive, bottom-exclusive]"
        )


def _physical_page(metadata):
    return {
        "geometry": copy.deepcopy(metadata["geometry"]),
        "bbox": copy.deepcopy(metadata["bbox"]),
        "foreground_fraction": metadata["foreground_fraction"],
        "size_bin": metadata["size_bin"],
        "definition": "Physical paper surface, independent of whether authored print is present",
    }


def _context(printed):
    metadata = printed.metadata
    return {
        "source_document_sha256": metadata["document"]["source_sha256"],
        "image_size": metadata["image_size"],
        "seeds": metadata["seeds"],
        "geometry": metadata["geometry"],
        "paper": metadata["paper"],
        "background": metadata["background"],
        "photometric": metadata["photometric"],
    }


def _matched_blank(printed):
    """Replace the authored white-page raster with white, retaining every nuisance."""
    metadata = copy.deepcopy(printed.metadata)
    size = tuple(metadata["image_size"])
    tint = np.asarray(metadata["paper"]["tint_rgb_multiplier"])
    paper_rgb = np.floor(np.clip(255 * tint, 0, 255) + 0.5).astype(np.uint8)
    blank_paper = np.empty((*size, 3), dtype=np.uint8)
    blank_paper[:] = paper_rgb
    background, background_metadata = generate_background(
        size,
        metadata["seeds"]["background"],
        metadata["background_type"],
    )
    if background_metadata != metadata["background"]:
        raise _invalid("counterfactual background reconstruction changed")
    params = WarpParameters(**metadata["geometry"]["parameters"])
    blank_scene = _composite_page(blank_paper, background, params)
    image, appearance = apply_photometry(
        blank_scene,
        metadata["seeds"]["photometric"],
        PhotometricConfig(**metadata["photometric"]["config"]),
    )
    if appearance != metadata["photometric"]:
        raise _invalid("counterfactual photometry reconstruction changed")
    metadata.update(
        has_page=False,
        family="none",
        size_bin="negative",
        negative_type="paper_panel",
        panel_color_mode="matched_paper_tint",
        geometry=None,
        sampling=None,
        document=None,
        paper=None,
        bbox=None,
        foreground_fraction=0.0,
        panel={
            "kind": "unprinted counterfactual paper",
            "printed_content": False,
            "is_target": False,
            "color_mode": "matched_paper_tint",
            "rgb": paper_rgb.tolist(),
            "geometry": copy.deepcopy(printed.metadata["geometry"]),
        },
        removed_content_definition=(
            "All authored non-white raster marks: text, table fill/rules and vector marks; not OCR"
        ),
    )
    metadata["compositing"]["page_render_calls"] = 0
    metadata["compositing"]["blank_panel_render_calls"] = 1
    return SceneSample(image, np.zeros(size, dtype=bool), metadata)


def _independent_physical(blank):
    geometry = copy.deepcopy(blank.metadata["panel"]["geometry"])
    params = WarpParameters(**geometry["parameters"])
    size = tuple(blank.metadata["image_size"])
    mask = make_geometry(size, size, params).photo_page_mask
    return {
        "geometry": geometry,
        "bbox": _bbox(mask),
        "foreground_fraction": float(mask.mean()),
        "size_bin": size_bin_for_fraction(float(mask.mean())),
        "definition": "Physical paper surface, independent of whether authored print is present",
    }


def _annotate(
    sample,
    *,
    identifier,
    group,
    split,
    context_id,
    source_hash=None,
    physical_page=None,
    pair_id=None,
    pair_role=None,
    partner_id=None,
    pair_context=None,
):
    metadata = sample.metadata
    metadata.update(
        schema_version=PAIRED_SCHEMA_VERSION,
        task_scope=TASK_SCOPE,
        sample_id=identifier,
        document_group=group,
        split=split,
        context_id=context_id,
        capture_session=f"paired-v3:{pair_id or identifier}",
        pair_id=pair_id,
        pair_role=pair_role,
        partner_id=partner_id,
        pair_context=copy.deepcopy(pair_context),
        pair_context_sha256=_hash_json(pair_context) if pair_context is not None else None,
        source_document_sha256=source_hash,
        physical_page=copy.deepcopy(physical_page),
        schema_notes=(
            "Printed-page GT excludes blank paper. Physical paper geometry is recorded separately. "
            "All controls share the source document split."
        ),
    )
    return sample


def _counts(documents):
    positives = documents * 4
    return {
        "page": positives,
        "negative": documents * 2,
        "total": documents * 6,
        "size_bins": {name: (positives + 2 - index) // 3 for index, name in enumerate(SIZE_BINS)},
        "blank_reference_size_bins": {
            name: (documents + 2 - index) // 3 for index, name in enumerate(SIZE_BINS)
        },
        "negative_types": {"background": documents, "paper_panel": documents},
    }


def _save_sample(directory, sample):
    identifier = sample.metadata["sample_id"]
    destination = directory / "samples" / identifier
    destination.mkdir(parents=True, exist_ok=False)
    with Image.fromarray(sample.image) as encoded:
        encoded.save(destination / "image.png")
    with Image.fromarray(sample.mask.astype(np.uint8) * 255) as encoded:
        encoded.save(destination / "mask.png")
    (destination / "metadata.json").write_bytes(_json(sample.metadata))
    paths = {name: f"samples/{identifier}/{name}" for name in _ARTIFACTS}
    return {
        "id": identifier,
        **{name: sample.metadata[name] for name in _RECORD_FIELDS},
        "paths": paths,
        "sha256": {name: _sha256(directory / path) for name, path in paths.items()},
    }


def generate_paired_scene_dataset(output_dir: str | os.PathLike[str], config=None):
    """Create a new dataset exclusively; no pair is filtered using preprocessing QA."""
    config = _config(PairedSceneConfig() if config is None else config)
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise FlattenerError(Status.IO_ERROR, "dataset_create_failed", str(error)) from error
    groups, samples, pairs, counts, qa = [], [], [], {}, {}
    document_index = 0
    for split, documents in zip(
        _SPLITS,
        (config.train_documents, config.validation_documents, config.test_documents),
        strict=True,
    ):
        counts[split] = _counts(documents)
        matched = split != "train" or config.training_panels == "matched"
        qa[split] = {
            "comparisons_checked": 0,
            "matched_pairs_checked": 0,
            "byte_identical_after_128x96": 0,
            "excluded_or_regenerated": 0,
        }
        for local_index in range(documents):
            group = f"paired-v3-document-{document_index:06d}"
            document_index += 1
            document_seed = derive_seed(config.seed, "paired-v3-document", group)
            printed = []
            for variant in range(4):
                identifier = f"{group}-view-{variant:03d}"
                scene_seed = derive_seed(config.seed, "paired-v3-scene", identifier)
                family = SCENE_FAMILIES[
                    derive_seed(config.seed, "paired-v3-family", identifier) % 4
                ]
                background = BACKGROUND_TYPES[
                    derive_seed(config.seed, "paired-v3-background-type", identifier)
                    % len(BACKGROUND_TYPES)
                ]
                sample = generate_scene_v2(
                    group,
                    document_seed,
                    scene_seed,
                    image_size=config.image_size,
                    size_bin=SIZE_BINS[(local_index * 4 + variant) % 3],
                    family=family,
                    background_type=background,
                )
                printed.append((identifier, sample))
            first_id, first = printed[0]
            blank_id, pair_id = f"{group}-blank-000", f"{group}-pair-000"
            shared_context = _context(first)
            source_hash = first.metadata["document"]["source_sha256"]
            if matched:
                blank = _matched_blank(first)
                blank_physical = _physical_page(first.metadata)
            else:
                blank = generate_scene_v2(
                    group,
                    document_seed,
                    first.metadata["seeds"]["scene"],
                    image_size=config.image_size,
                    has_page=False,
                    negative_type="paper_panel",
                    panel_color_mode=("white", "colored")[local_index % 2],
                    background_type=first.metadata["background_type"],
                )
                blank_physical = _independent_physical(blank)
            # Only already-generated arrays are examined, including test. No
            # file is decoded, no model is loaded, and no pair is filtered.
            same = torch.equal(
                preprocess_image(first.image, _QA_CONFIG), preprocess_image(blank.image, _QA_CONFIG)
            )
            qa[split]["comparisons_checked"] += 1
            qa[split]["matched_pairs_checked"] += int(matched)
            qa[split]["byte_identical_after_128x96"] += int(same)
            groups.append(
                {
                    "id": group,
                    "kind": "document",
                    "seed": document_seed,
                    "split": split,
                    "positive_views": 4,
                    "blank_controls": 1,
                    "samples": 5,
                    "source_document_sha256": source_hash,
                }
            )
            for variant, (identifier, sample) in enumerate(printed):
                member = matched and variant == 0
                _annotate(
                    sample,
                    identifier=identifier,
                    group=group,
                    split=split,
                    context_id=identifier,
                    source_hash=source_hash,
                    physical_page=_physical_page(sample.metadata),
                    pair_id=pair_id if member else None,
                    pair_role="printed" if member else None,
                    partner_id=blank_id if member else None,
                    pair_context=shared_context if member else None,
                )
                samples.append(_save_sample(directory, sample))
            _annotate(
                blank,
                identifier=blank_id,
                group=group,
                split=split,
                context_id=first_id,
                source_hash=source_hash,
                physical_page=blank_physical,
                pair_id=pair_id if matched else None,
                pair_role="blank" if matched else None,
                partner_id=first_id if matched else None,
                pair_context=shared_context if matched else None,
            )
            blank.metadata["blank_reference_size_bin"] = first.metadata["size_bin"]
            blank.metadata["panel_condition"] = "matched" if matched else "independent"
            samples.append(_save_sample(directory, blank))
            if matched:
                pairs.append(
                    {
                        "id": pair_id,
                        "document_group": group,
                        "split": split,
                        "printed_id": first_id,
                        "blank_id": blank_id,
                        "context_sha256": _hash_json(shared_context),
                    }
                )
            background_group = f"paired-v3-background-{split}-{local_index:06d}"
            background_id = f"{background_group}-view-000"
            background_seed = derive_seed(config.seed, "paired-v3-document", background_group)
            background = generate_scene_v2(
                background_group,
                background_seed,
                derive_seed(config.seed, "paired-v3-scene", background_id),
                image_size=config.image_size,
                has_page=False,
                negative_type="background",
                background_type=BACKGROUND_TYPES[
                    derive_seed(config.seed, "paired-v3-background-type", background_id)
                    % len(BACKGROUND_TYPES)
                ],
            )
            _annotate(
                background,
                identifier=background_id,
                group=background_group,
                split=split,
                context_id=background_id,
            )
            groups.append(
                {
                    "id": background_group,
                    "kind": "background",
                    "seed": background_seed,
                    "split": split,
                    "positive_views": 0,
                    "blank_controls": 0,
                    "samples": 1,
                    "source_document_sha256": None,
                }
            )
            samples.append(_save_sample(directory, background))
    package = Path(__file__).resolve().parent.parent
    source_names = (
        "synthetic/paired_scenes.py",
        "synthetic/scene_v2.py",
        "synthetic/scenes.py",
        "synthetic/documents.py",
        "synthetic/geometry.py",
        "synthetic/pipeline.py",
        "renderer.py",
        "coordinates.py",
        "validation.py",
        "contracts.py",
        "regions/segmentation.py",
    )
    manifest = {
        "schema_version": PAIRED_SCHEMA_VERSION,
        "complete": True,
        "config": asdict(config),
        "task_scope": TASK_SCOPE,
        "groups": groups,
        "samples": samples,
        "pairs": pairs,
        "counts": counts,
        "preprocessing_qa": qa,
        "preprocessing_qa_protocol": _qa_protocol(),
        "pair_selection": (
            "Variant 0 only, decided before generation; all views/control share document split"
        ),
        "size_assignment": (
            "Printed views cycle small/medium/large within each split; "
            "variant0 blank references retained"
        ),
        "is_real_photo": False,
        "external_images": False,
        "source": (
            "Project-authored text, unchanged bundled font, analytic warps "
            "and procedural backgrounds"
        ),
        "terms": "Authored text/images/code Apache-2.0; font OFL-1.1",
        "generator_code_sha256": {name: _sha256(package / name) for name in source_names},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": str(torch.__version__),
        },
    }
    (directory / "manifest.json").write_bytes(_json(manifest))
    validate_paired_scene_dataset(directory)
    return manifest


def _metadata(record, paths):
    metadata = _read_json(paths["metadata.json"], _METADATA_LIMIT)
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != PAIRED_SCHEMA_VERSION
        or metadata.get("task_scope") != TASK_SCOPE
        or metadata.get("is_real_photo") is not False
    ):
        raise _invalid("unsupported/unscoped paired scene metadata")
    if (
        metadata.get("sample_id") != record["id"]
        or metadata.get("document_id") != record["document_group"]
    ):
        raise _invalid("sample identity disagrees with record")
    for key in _RECORD_FIELDS:
        if metadata.get(key) != record[key]:
            raise _invalid(f"metadata disagrees with record: {key}")
    size = _size(tuple(metadata["image_size"]))
    fraction = metadata.get("foreground_fraction")
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1 or not math.isfinite(fraction):
        raise _invalid("foreground_fraction must be finite and in [0, 1]")
    if type(record["has_page"]) is not bool or record["background_type"] not in BACKGROUND_TYPES:
        raise _invalid("invalid has_page or background type")
    if record["has_page"]:
        _validate_bbox(metadata["bbox"], size)
        if (
            record["family"] not in SCENE_FAMILIES
            or record["size_bin"] != size_bin_for_fraction(fraction)
            or record["negative_type"] is not None
            or record["panel_color_mode"] is not None
        ):
            raise _invalid("invalid printed-page labels")
        geometry = metadata["geometry"]
        if (
            geometry["family"] != record["family"]
            or geometry["parameters"]["family"] != record["family"]
            or geometry["photo_page_fraction"] != fraction
            or metadata["document"] is None
        ):
            raise _invalid("printed geometry disagrees with labels")
    elif (
        fraction != 0
        or metadata["bbox"] is not None
        or metadata["geometry"] is not None
        or metadata["document"] is not None
        or record["family"] != "none"
        or record["size_bin"] != "negative"
        or record["negative_type"] not in ("background", "paper_panel")
    ):
        raise _invalid("negative cannot claim printed-page GT")
    physical = metadata.get("physical_page")
    if record["has_page"] or record["negative_type"] == "paper_panel":
        if not isinstance(physical, dict):
            raise _invalid("paper surface must retain independent physical geometry")
        physical_fraction = physical["foreground_fraction"]
        if physical["size_bin"] != size_bin_for_fraction(physical_fraction):
            raise _invalid("invalid physical page area")
        _validate_bbox(physical["bbox"], size)
        geometry = physical["geometry"]
        params = WarpParameters(**geometry["parameters"])
        if (
            geometry["flat_size"] != list(size)
            or geometry["photo_size"] != list(size)
            or geometry["family"] != params.family
            or geometry["photo_page_fraction"] != physical_fraction
            or geometry["flat_visible_fraction"] != 1
            or geometry["folds"] is not False
        ):
            raise _invalid("physical geometry metadata differs from the source-page contract")
        if record["has_page"] and physical != _physical_page(metadata):
            raise _invalid("printed/physical page geometry disagree")
        if not record["has_page"]:
            panel = metadata["panel"]
            if (
                not isinstance(panel, dict)
                or panel.get("printed_content") is not False
                or panel.get("is_target") is not False
                or panel.get("geometry") != physical["geometry"]
            ):
                raise _invalid("blank panel must exclude print and preserve its physical geometry")
    elif physical is not None or metadata["source_document_sha256"] is not None:
        raise _invalid("background-only negative cannot claim a physical paper source")
    if record["pair_id"] is None:
        if (
            any(record[name] is not None for name in ("pair_role", "partner_id"))
            or metadata["pair_context"] is not None
            or metadata["pair_context_sha256"] is not None
        ):
            raise _invalid("unpaired records must have null pair fields")
    else:
        context = metadata["pair_context"]
        if (
            not isinstance(record["pair_id"], str)
            or not record["pair_id"]
            or record["pair_role"] not in ("printed", "blank")
            or not isinstance(record["partner_id"], str)
            or record["has_page"] != (record["pair_role"] == "printed")
            or not isinstance(context, dict)
            or metadata["pair_context_sha256"] != _hash_json(context)
        ):
            raise _invalid("invalid pair identity, role or context hash")
        if (
            context["source_document_sha256"] != metadata["source_document_sha256"]
            or context["image_size"] != metadata["image_size"]
            or context["seeds"] != metadata["seeds"]
            or context["background"] != metadata["background"]
            or context["photometric"] != metadata["photometric"]
            or context["geometry"] != physical["geometry"]
        ):
            raise _invalid("pair material/nuisance context disagrees with sample")
        if record["pair_role"] == "printed" and context["paper"] != metadata["paper"]:
            raise _invalid("paired printed paper tint disagrees")
        if record["pair_role"] == "blank":
            tint = context["paper"]["tint_rgb_multiplier"]
            if (
                not isinstance(tint, list)
                or len(tint) != 3
                or any(
                    type(v) not in (int, float) or not 0 <= v <= 2 or not math.isfinite(v)
                    for v in tint
                )
            ):
                raise _invalid("paired paper tint must contain three bounded finite values")
            rgb = np.floor(np.clip(np.asarray(tint) * 255, 0, 255) + 0.5).astype(np.uint8).tolist()
            if (
                metadata["panel"]["rgb"] != rgb
                or metadata["panel_color_mode"] != "matched_paper_tint"
            ):
                raise _invalid("matched blank does not preserve printed paper color")
    return metadata


def _validate(directory):
    root = Path(directory).resolve()
    manifest = _read_json(root / "manifest.json", _MANIFEST_LIMIT)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != PAIRED_SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("task_scope") != TASK_SCOPE
    ):
        raise _invalid("unsupported/incomplete paired-scene manifest")
    raw = dict(manifest["config"])
    raw["image_size"] = tuple(raw["image_size"])
    config = _config(PairedSceneConfig(**raw))
    if manifest.get("preprocessing_qa_protocol") != _qa_protocol():
        raise _invalid("preprocessing QA must use the fixed 128x96 exact/count-only protocol")
    if any(not isinstance(manifest.get(key), list) for key in ("groups", "samples", "pairs")):
        raise _invalid("groups, samples and pairs must be lists")
    groups, seeds, group_members = {}, set(), {}
    for group in manifest["groups"]:
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("id"), str)
            or not group["id"]
            or group["id"] in groups
            or group.get("kind") not in ("document", "background")
            or group.get("split") not in _SPLITS
        ):
            raise _invalid("invalid/duplicate source group", "split_leakage")
        seed = _seed(group["seed"])
        if seed in seeds or seed != derive_seed(config.seed, "paired-v3-document", group["id"]):
            raise _invalid("invalid/nonunique source-group seed", "split_leakage")
        seeds.add(seed)
        expected = (4, 1, 5) if group["kind"] == "document" else (0, 0, 1)
        if (group["positive_views"], group["blank_controls"], group["samples"]) != expected:
            raise _invalid("source-group cardinality differs from protocol")
        if (group["kind"] == "document" and not _valid_hash(group["source_document_sha256"])) or (
            group["kind"] == "background" and group["source_document_sha256"] is not None
        ):
            raise _invalid("source group requires an exact document hash or explicit absence")
        groups[group["id"]], group_members[group["id"]] = group, []
    records, metadata_by_id, used_paths, sessions, image_splits, source_splits = (
        {},
        {},
        set(),
        {},
        {},
        {},
    )
    for record in manifest["samples"]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
        ):
            raise _invalid("sample requires a nonempty string ID")
        identifier, split, group = (
            record["id"],
            record["split"],
            groups.get(record["document_group"]),
        )
        if identifier in records or group is None or group["split"] != split:
            raise _invalid("duplicate sample or cross-split source group", "split_leakage")
        session = record["capture_session"]
        if (
            not isinstance(session, str)
            or not session
            or session in sessions
            and sessions[session] != split
        ):
            raise _invalid("capture session crosses splits", "split_leakage")
        sessions[session] = split
        paths = _artifact_paths(root, record)
        for path in paths.values():
            if path in used_paths:
                raise _invalid("artifact path reused by multiple records")
            used_paths.add(path)
        metadata = _metadata(record, paths)
        if (
            metadata["image_size"] != list(config.image_size)
            or metadata["seeds"]["document"] != group["seed"]
            or metadata["seeds"]["scene"]
            != derive_seed(config.seed, "paired-v3-scene", record["context_id"])
        ):
            raise _invalid("sample size or seed provenance disagrees")
        image_hash = record["sha256"]["image.png"]
        if image_hash in image_splits and image_splits[image_hash] != split:
            raise _invalid("identical RGB scene crosses splits", "split_leakage")
        image_splits[image_hash] = split
        source_hash = metadata["source_document_sha256"]
        if source_hash != group["source_document_sha256"]:
            raise _invalid("source-group document hash disagrees")
        if record["has_page"]:
            doc = metadata["document"]
            if (
                doc["document_id"] != group["id"]
                or doc["seed"] != group["seed"]
                or doc["source_sha256"] != source_hash
            ):
                raise _invalid("printed document provenance disagrees")
        if source_hash is not None:
            if source_hash in source_splits and source_splits[source_hash] != split:
                raise _invalid("authored document crosses splits", "split_leakage")
            source_splits[source_hash] = split
        records[identifier], metadata_by_id[identifier] = record, metadata
        group_members[group["id"]].append(record)
    expected_pairs = {}
    for group_id, group in groups.items():
        members = group_members[group_id]
        if len(members) != group["samples"]:
            raise _invalid("source-group members missing or duplicated")
        if group["kind"] == "background":
            if (
                members[0]["has_page"]
                or members[0]["negative_type"] != "background"
                or members[0]["pair_id"] is not None
            ):
                raise _invalid("invalid background group membership")
            continue
        by_id = {record["id"]: record for record in members}
        positive_ids = [f"{group_id}-view-{variant:03d}" for variant in range(4)]
        blank_id = f"{group_id}-blank-000"
        if set(by_id) != {*positive_ids, blank_id} or any(
            not by_id[key]["has_page"] for key in positive_ids
        ):
            raise _invalid(
                "document groups require exactly four printed views and one blank control"
            )
        blank = by_id[blank_id]
        if (
            blank["has_page"]
            or blank["negative_type"] != "paper_panel"
            or blank["context_id"] != positive_ids[0]
        ):
            raise _invalid("blank control must reference predetermined printed variant0")
        matched = group["split"] != "train" or config.training_panels == "matched"
        if metadata_by_id[blank_id]["panel_condition"] != ("matched" if matched else "independent"):
            raise _invalid("blank control condition differs from config")
        if (
            metadata_by_id[blank_id]["blank_reference_size_bin"]
            != by_id[positive_ids[0]]["size_bin"]
        ):
            raise _invalid("blank reference size bin differs from printed partner")
        for key in positive_ids[1:]:
            if by_id[key]["pair_id"] is not None or by_id[key]["context_id"] != key:
                raise _invalid("only variant0 may participate in a counterfactual pair")
        first = by_id[positive_ids[0]]
        if matched:
            pair_id = f"{group_id}-pair-000"
            if (
                first["pair_id"] != pair_id
                or blank["pair_id"] != pair_id
                or first["pair_role"] != "printed"
                or blank["pair_role"] != "blank"
                or first["partner_id"] != blank_id
                or blank["partner_id"] != first["id"]
                or first["capture_session"] != blank["capture_session"]
            ):
                raise _invalid("counterfactual partner links are not reciprocal")
            a, b = metadata_by_id[first["id"]], metadata_by_id[blank_id]
            if a["pair_context"] != b["pair_context"] or a["physical_page"] != b["physical_page"]:
                raise _invalid("matched pair does not share exact physical/material context")
            expected_pairs[pair_id] = {
                "id": pair_id,
                "document_group": group_id,
                "split": group["split"],
                "printed_id": first["id"],
                "blank_id": blank_id,
                "context_sha256": a["pair_context_sha256"],
            }
        elif first["pair_id"] is not None or blank["pair_id"] is not None:
            raise _invalid("independent training controls must not claim counterfactual pairs")
    if (
        len(manifest["pairs"]) != len(expected_pairs)
        or {pair["id"]: pair for pair in manifest["pairs"]} != expected_pairs
    ):
        raise _invalid("manifest pair graph differs from sample links")
    page_counts, negative_counts, size_counts = {}, {}, {}
    for split, documents in zip(
        _SPLITS,
        (config.train_documents, config.validation_documents, config.test_documents),
        strict=True,
    ):
        selected = [record for record in records.values() if record["split"] == split]
        page_counts[split] = sum(record["has_page"] for record in selected)
        negative_counts[split] = len(selected) - page_counts[split]
        size_counts[split] = {
            name: sum(record["has_page"] and record["size_bin"] == name for record in selected)
            for name in SIZE_BINS
        }
        blank_bins = {
            name: sum(
                record["negative_type"] == "paper_panel"
                and metadata_by_id[record["id"]]["blank_reference_size_bin"] == name
                for record in selected
            )
            for name in SIZE_BINS
        }
        negative_types = {
            name: sum(record["negative_type"] == name for record in selected)
            for name in ("background", "paper_panel")
        }
        actual = {
            "page": page_counts[split],
            "negative": negative_counts[split],
            "total": len(selected),
            "size_bins": size_counts[split],
            "blank_reference_size_bins": blank_bins,
            "negative_types": negative_types,
        }
        if actual != _counts(documents) or manifest["counts"][split] != actual:
            raise _invalid("split/strata/control counts differ from predetermined protocol")
        qa = manifest["preprocessing_qa"][split]
        expected_matched = (
            documents if split != "train" or config.training_panels == "matched" else 0
        )
        if (
            qa["comparisons_checked"] != documents
            or qa["matched_pairs_checked"] != expected_matched
            or type(qa["byte_identical_after_128x96"]) is not int
            or not 0 <= qa["byte_identical_after_128x96"] <= documents
            or qa["excluded_or_regenerated"] != 0
        ):
            raise _invalid("invalid generation-only preprocessing QA counts")
    return {
        "manifest": manifest,
        "sample_counts": {split: page_counts[split] + negative_counts[split] for split in _SPLITS},
        "page_counts": page_counts,
        "negative_counts": negative_counts,
        "size_bin_counts": size_counts,
    }


def validate_paired_scene_dataset(directory):
    """Validate every hash/group/pair/stratum without decoding any image or GT."""
    try:
        return _validate(directory)
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as error:
        raise _invalid(f"Malformed paired-scene dataset: {error}") from error


def load_paired_scene(directory, record):
    """Decode only this selected sample; physical blank geometry is not printed GT."""
    try:
        paths = _artifact_paths(Path(directory).resolve(), record)
        metadata = _metadata(record, paths)
        size = _size(tuple(metadata["image_size"]))
        with Image.open(paths["image.png"]) as encoded:
            if encoded.mode != "RGB" or encoded.size != (size[1], size[0]):
                raise _invalid("image must be RGB with recorded dimensions")
            image = np.array(encoded, copy=True)
        with Image.open(paths["mask.png"]) as encoded:
            if encoded.mode != "L" or encoded.size != (size[1], size[0]):
                raise _invalid("mask must be grayscale with recorded dimensions")
            values = np.array(encoded, copy=True)
        if values.dtype != np.uint8 or not np.isin(values, (0, 255)).all():
            raise _invalid("mask values must be exactly uint8 0/255")
        mask = values == 255
        if (
            bool(mask.any()) != record["has_page"]
            or float(mask.mean()) != metadata["foreground_fraction"]
            or _bbox(mask) != metadata["bbox"]
        ):
            raise _invalid("stored mask differs from recorded presence/area/bbox")
        physical = metadata["physical_page"]
        if physical is not None:
            params = WarpParameters(**physical["geometry"]["parameters"])
            expected = make_geometry(size, size, params).photo_page_mask
            if (
                float(expected.mean()) != physical["foreground_fraction"]
                or _bbox(expected) != physical["bbox"]
            ):
                raise _invalid("physical paper surface metadata differs from analytic geometry")
            if record["has_page"] and not np.array_equal(mask, expected):
                raise _invalid("printed GT mask differs from analytic physical surface")
        return SceneSample(image, mask, metadata)
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as error:
        raise _invalid(f"Cannot load paired scene: {error}") from error
