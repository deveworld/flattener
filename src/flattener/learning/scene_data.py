"""Lazy full-scene coordinate supervision from immutable paired scenes.

Only printed positives are examples. A blank control's physical paper geometry
is never a coordinate target. Initialization validates every artifact hash and
metadata relationship, without decoding pixels from any split. Indexing then
decodes only the requested positive and derives its map analytically.

Targets map requested flat-output pixel centers into the *original full scene*
in normalized physical-image coordinates, xy order, align_corners=False. There
is no region crop, ROI normalization, coordinate clipping, or resized-map GT.
"""

from __future__ import annotations

import copy
import hashlib
import json
import operator
import os
import pickle
import platform
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import __version__ as pillow_version
from torch.utils.data import Dataset

from ..contracts import FlattenerError, Status
from ..synthetic.geometry import WarpParameters, make_geometry
from ..synthetic.paired_scenes import (
    _metadata,
    _read_json,
    load_paired_scene,
    validate_paired_scene_dataset,
)
from ..synthetic.pipeline import derive_seed
from ..synthetic.scene_v2 import _candidate_warp
from ..synthetic.scenes import SceneSample, _artifact_paths, _sha256
from .model import ModelConfig, preprocess_image

SCENE_COORDINATE_DATA_VERSION = "full-scene-analytic-coordinates-1.0"
_SPLITS = ("train", "validation", "test")
_MAX_SIDE = 512
_MANIFEST_LIMIT = 64 * 1024 * 1024
_ARTIFACT_LIMIT = 4 * 1024 * 1024
_GENERATOR_SOURCES = (
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
_DATA_SOURCES = (*_GENERATOR_SOURCES, "learning/scene_data.py", "learning/model.py")
_PAPER_VERSION = "paper-augmentation-1.0"
_PAPER_SOURCE = "synthetic/paper.py"
_TEMPLATE_LIMIT = 64 * 1024 * 1024
_TEMPLATE_PIXELS = 32_000_000
_CACHE_VERSION = "coordinate-preprocessed-cache-1.0"
_CACHE_METADATA_LIMIT = 1024 * 1024
_CACHE_ARCHIVE_OVERHEAD = 8 * 1024 * 1024
_PAPER_FIELDS = (
    "source", "document_group", "split", "capture_session", "has_page", "template_id",
    "template_sha256", "family", "size_bin",
)
_GEOMETRY_CONVENTIONS = {
    "model": "analytic_triangular_shears_then_homography_v1",
    "coordinate_order": "xy",
    "coordinate_units": "normalized_physical_image_bounds_minus1_plus1",
    "backward_map_direction": "flat_output_to_photo_source",
    "synthesis_map_direction": "photo_output_to_flat_source",
    "missing_coordinates": "NaN_pairs_with_false_mask",
    "visibility": "continuous_point_correspondence_not_bilinear_footprint",
    "occlusion_model": "none_bijective_2d_deformation",
}
_DATA_ERRORS = (
    OSError,
    UnicodeError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    OverflowError,
    RecursionError,
    Image.DecompressionBombError,
    Image.DecompressionBombWarning,
)


def _invalid(message):
    return FlattenerError(Status.INPUT_ERROR, "invalid_coordinate_scene_dataset", message)


def _same_seeded_parameters(actual, expected):
    """Compare seeded draws exactly, allowing roundoff in derived placement only.

    Linux/Windows trig implementations can change derived scale/translation
    by a few float64 ulps. Recorded parameters remain the coordinate truth;
    hashes, seeds, random draws and all other fields retain exact checks.
    """
    left, right = asdict(actual), asdict(expected)
    if type(actual) is not type(expected) or left.keys() != right.keys():
        return False
    derived = {"scale_x", "scale_y", "translate_x", "translate_y"}
    for name, value in left.items():
        reference = right[name]
        if value == reference:
            continue
        if (name not in derived or type(value) is not float or type(reference) is not float
                or not np.isfinite(value) or not np.isfinite(reference)
                or abs(value - reference) > 8 * np.finfo(np.float64).eps * max(1, abs(reference))):
            return False
    return True


def _size(value, name, minimum=1):
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(side) is not int or not minimum <= side <= _MAX_SIDE for side in value)
    ):
        raise _invalid(f"{name} must be a (height, width) integer tuple in [{minimum}, 512]")
    return value


def _manifest_digest(directory):
    path = directory / "manifest.json"
    if path.stat().st_size > _MANIFEST_LIMIT:
        raise _invalid("manifest exceeds 64 MiB")
    return _sha256(path)


def _source_hashes():
    package = Path(__file__).resolve().parent.parent
    return {name: _sha256(package / name) for name in _DATA_SOURCES}


def _data_hashes(include_paper):
    result = _source_hashes()
    if include_paper:
        result[_PAPER_SOURCE] = _sha256(Path(__file__).resolve().parent.parent / _PAPER_SOURCE)
    return result


def _paper_backend():
    from ..synthetic import paper

    return paper


def _hash_string(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 256 and value == value.strip()


def _paper_path(directory, relative):
    # New artifacts use the existing flat payload directory, without aliases.
    if (not isinstance(relative, str) or len(Path(relative).parts) != 2
            or Path(relative).parts[0] != "files" or ".." in Path(relative).parts):
        raise _invalid("paper artifacts must be direct files/<name> payloads")
    path = directory / relative
    if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(directory)):
        raise _invalid("paper artifact is missing, aliased, or outside the dataset")
    return path


def _template_artifact(directory, template):
    path = _paper_path(directory, template["path"])
    if (type(template["bytes"]) is not int or not 0 < template["bytes"] <= _TEMPLATE_LIMIT
            or path.stat().st_size != template["bytes"]):
        raise _invalid("template exceeds its resource limit or recorded byte size")
    if _sha256(path) != template["sha256"]:
        raise _invalid("paper template artifact changed")
    return path


def _paper_artifacts(directory, record):
    if set(record["paths"]) != {"image.png", "mask.png", "metadata.json"}:
        raise _invalid("paper record requires exactly image, mask and metadata artifacts")
    if set(record["sha256"]) != set(record["paths"]):
        raise _invalid("paper artifact checksum inventory differs")
    for name, relative in record["paths"].items():
        path = _paper_path(directory, relative)
        limit = 1024 * 1024 if name == "metadata.json" else _ARTIFACT_LIMIT
        if not 0 < path.stat().st_size <= limit or not _hash_string(record["sha256"][name]):
            raise _invalid("paper artifact exceeds its resource limit or has an invalid hash")
    return _artifact_paths(directory, record)


def _paper_metadata(record, paths, template, photo_size, capture=None):
    metadata = _read_json(paths["metadata.json"], 1024 * 1024)
    kind = record.get("kind", "simulated")
    if (metadata.get("schema_version") != _PAPER_VERSION
            or metadata.get("sample_id") != record["id"]
            or any(metadata.get(key) != record[key] for key in _PAPER_FIELDS)
            or metadata.get("image_size") != list(photo_size)
            or metadata.get("is_real_photo") is not (kind == "capture")
            or metadata.get("source_template_is_real_capture") is not True
            or metadata.get("template_content_sha256") != template["content_sha256"]):
        raise _invalid("paper metadata identity or source-template provenance disagrees")
    geometry = metadata["geometry"]
    model = ("manual_quad_planar_capture_v1" if kind == "capture"
             else "extruded_cylinder_perspective_v1")
    if (geometry.get("model") != model
            or geometry.get("photo_size") != list(photo_size)
            or geometry.get("align_corners") is not False
            or geometry.get("coordinate_order") != "xy"
            or geometry.get("coordinate_units") !=
            "normalized_physical_image_bounds_minus1_plus1"
            or geometry.get("backward_map_direction") != "flat_output_to_photo_source"):
        raise _invalid("paper geometry must retain the original full-scene coordinate frame")
    if kind == "capture":
        if (capture is None or metadata.get("capture_id") != capture["id"]
                or metadata.get("capture_sha256") != capture["sha256"]
                or metadata.get("original_size") != capture["original_size"]
                or metadata.get("corners_xy") != capture["corners_xy"]
                or geometry.get("original_size") != capture["original_size"]
                or geometry.get("corners_xy") != capture["corners_xy"]
                or type(metadata.get("photometric_augmented")) is not bool):
            raise _invalid("paper capture metadata differs from its frozen source annotation")
        return metadata, None
    sampling = metadata["sampling"]
    attempt, seed = sampling["attempts"], metadata["geometry_seed"]
    fraction = sampling["foreground_fraction"]
    if (type(attempt) is not int or not 1 <= attempt <= 256
            or type(seed) is not int or not 0 <= seed < 2**63
            or sampling.get("maximum_attempts") != 256
            or sampling.get("geometry_seed") != seed
            or sampling.get("requested_size_bin") != record["size_bin"]
            or sampling.get("actual_size_bin") != record["size_bin"]
            or type(fraction) not in (int, float) or not 0 < fraction <= 1):
        raise _invalid("paper sampling provenance disagrees")
    paper = _paper_backend()
    params = paper.PaperParameters(**geometry["parameters"])
    expected = paper.paper_parameters_for_attempt(
        record["family"], seed, size_bin=record["size_bin"], attempt=attempt)
    if not _same_seeded_parameters(params, expected) or geometry["parameters"] != asdict(params):
        raise _invalid("paper geometry differs from its recorded generator seed")
    return metadata, params


def _paper_collection(directory, manifest, photo_size, source_hashes):
    """Validate all source identities and hashes, never decode even held-out pixels."""
    block = manifest.get("collection", {}).get("paper_augmentation")
    if (not isinstance(block, dict) or block.get("schema_version") != _PAPER_VERSION
            or block.get("complete") is not True or block.get("photo_size") != list(photo_size)
            or block.get("generator_code_sha256") != {_PAPER_SOURCE: source_hashes[_PAPER_SOURCE]}):
        raise _invalid("paper augmentation is missing, incomplete, or has changed generator code")
    templates, records, captures = block["templates"], block["records"], block.get("captures", [])
    if (not isinstance(templates, list) or not 1 <= len(templates) <= 10_000
            or not isinstance(captures, list) or len(captures) > 100_000
            or not isinstance(records, list) or not records
            or len(records) + len(manifest["samples"]) > 100_000):
        raise _invalid("paper augmentation exceeds bounded template/sample counts")
    docunet = manifest.get("collection", {}).get("sources", {}).get("docunet", {})
    forbidden = {item["sha256"] for item in docunet.get("files", [])
                 if isinstance(item.get("sha256"), str)}
    native_groups = {record["document_group"] for record in manifest["samples"]}
    ids = {record["id"] for record in manifest["samples"]}
    paths = {path for record in manifest["samples"] for path in record["paths"].values()}
    identities, by_id = {}, {}

    def identity(kind, value, group, split):
        old = identities.setdefault((kind, value), (group, split))
        if old != (group, split):
            raise _invalid("paper source/document/capture identity crosses groups or splits")

    for template in templates:
        group, split = template["document_group"], template["split"]
        if (any(not _identity(template[key]) for key in ("id", "document_group", "capture_id"))
                or template["id"] in by_id or group in native_groups or split not in _SPLITS
                or template.get("source_dataset") != "smartdoc2015-ch1"
                or template.get("license_spdx") != "CC-BY-4.0"
                or any(not _hash_string(template[key]) for key in ("sha256", "content_sha256"))
                or template["sha256"] in forbidden or template["content_sha256"] in forbidden
                or "docunet" in template["path"].lower()):
            raise _invalid("paper template must be an eligible, disjoint SmartDoc source")
        _template_artifact(directory, template)
        if template["path"] in paths:
            raise _invalid("paper template path is reused")
        paths.add(template["path"])
        for kind, value in (("document", group), ("capture", template["capture_id"]),
                            ("raw", template["sha256"]), ("content", template["content_sha256"])):
            identity(kind, value, group, split)
        by_id[template["id"]] = copy.deepcopy(template)
    by_capture = {}
    for capture in captures:
        group, split = capture["document_group"], capture["split"]
        size, corners = capture["original_size"], np.asarray(capture["corners_xy"])
        if (not _identity(capture["id"]) or capture["id"] in by_capture
                or not _identity(group) or group in native_groups or split not in _SPLITS
                or capture.get("source_dataset") != "smartdoc2015-ch1"
                or capture.get("license_spdx") != "CC-BY-4.0"
                or not _hash_string(capture["sha256"]) or capture["sha256"] in forbidden
                or "docunet" in capture["path"].lower()
                or not isinstance(size, list) or len(size) != 2
                or any(type(side) is not int or not 2 <= side <= 16_384 for side in size)
                or size[0] * size[1] > _TEMPLATE_PIXELS
                or corners.shape != (4, 2) or corners.dtype.kind not in "fiu"
                or not np.isfinite(corners).all()):
            raise _invalid("paper capture must be a bounded, eligible annotated SmartDoc frame")
        _template_artifact(directory, capture)
        normalized = 2 * (corners.astype(np.float64) + 0.5) / np.array(size[::-1]) - 1
        edges = np.roll(normalized, -1, axis=0) - normalized
        cross = (edges[:, 0] * np.roll(edges[:, 1], -1)
                 - edges[:, 1] * np.roll(edges[:, 0], -1))
        if (np.abs(normalized) > 1).any() or (cross <= 1e-6).any():
            raise _invalid("paper capture corners must form a bounded clockwise convex quad")
        if capture["path"] in paths:
            raise _invalid("paper capture path is reused")
        paths.add(capture["path"])
        identity("document", group, group, split)
        identity("capture", capture["id"], group, split)
        identity("raw", capture["sha256"], group, split)
        by_capture[capture["id"]] = copy.deepcopy(capture)
    counts = dict.fromkeys(_SPLITS, 0)
    used_templates, used_captures = set(), set()
    for record in records:
        template = by_id.get(record["template_id"])
        kind = record.get("kind", "simulated")
        capture = by_capture.get(record.get("capture_id"))
        if (not _identity(record["id"]) or record["id"] in ids or template is None
                or record.get("source") != "paper" or record.get("has_page") is not True
                or kind not in ("simulated", "capture")
                or record.get("source_kind", kind) != kind
                or record["family"] not in (("capture_perspective",) if kind == "capture"
                                           else ("flat", "gentle", "strong"))
                or record["size_bin"] not in ("small", "medium", "large")
                or not _identity(record["capture_session"])
                or record["document_group"] != template["document_group"]
                or record["split"] != template["split"]
                or record["template_sha256"] != template["sha256"]):
            raise _invalid("paper record identity, group, split or template disagrees")
        if kind == "capture":
            if (capture is None or capture["document_group"] != template["document_group"]
                    or capture["split"] != template["split"]):
                raise _invalid("paper capture and template must share the source document split")
            used_captures.add(capture["id"])
        elif record.get("capture_id") is not None:
            raise _invalid("simulated paper cannot claim an actual source capture")
        identity("capture", record["capture_session"], record["document_group"], record["split"])
        ids.add(record["id"])
        selected_paths = _paper_artifacts(directory, record)
        _paper_metadata(record, selected_paths, template, photo_size, capture)
        for path in record["paths"].values():
            if path in paths:
                raise _invalid("paper artifact path is reused")
            paths.add(path)
        used_templates.add(record["template_id"])
        counts[record["split"]] += 1
    if set(by_id) != used_templates or set(by_capture) != used_captures or not all(counts.values()):
        raise _invalid("paper augmentation requires used templates and positives in every split")
    normalized = copy.deepcopy(records)
    for record in normalized:
        record["source_kind"] = record.get("kind", "simulated")
    return by_id, by_capture, normalized, counts


def _bounded_artifacts(directory, manifest):
    # Reject pathological encoded files before the whole-dataset hash pass.
    # This is metadata/stat inspection only: even held-out PNGs stay undecoded.
    records = manifest["samples"]
    if not isinstance(records, list) or not 1 <= len(records) <= 100_000:
        raise _invalid("manifest requires at most 100000 samples")
    for record in records:
        for name, path in _artifact_paths(directory, record, check_hashes=False).items():
            limit = 1024 * 1024 if name == "metadata.json" else _ARTIFACT_LIMIT
            if not 0 < path.stat().st_size <= limit:
                raise _invalid(f"{name} is empty or exceeds its encoded resource limit")


def _geometry_contract(metadata, photo_size):
    geometry = metadata["geometry"]
    if (
        metadata["has_page"] is not True
        or metadata["negative_type"] is not None
        or metadata["document"] is None
        or metadata["image_size"] != list(photo_size)
        or metadata["document"]["output_size"] != list(photo_size)
        or geometry["flat_size"] != list(photo_size)
        or geometry["photo_size"] != list(photo_size)
        or geometry.get("align_corners") is not False
        or geometry.get("folds") is not False
        or any(geometry.get(key) != value for key, value in _GEOMETRY_CONVENTIONS.items())
    ):
        raise _invalid("positive geometry must describe the original full-scene coordinate frame")
    parameters = WarpParameters(**geometry["parameters"])
    if parameters.family != metadata["family"]:
        raise _invalid("coordinate geometry family disagrees with sample")
    # Do not accept arbitrary plausible geometry merely because a JSON file
    # claims to be synthetic. Reproduce its recorded, domain-separated draw
    # without rendering a document or allocating any held-out coordinate map.
    sampling = metadata["sampling"]
    attempt = sampling["attempts"]
    warp_seed = derive_seed(metadata["seeds"]["scene"], "v2-warp")
    if (
        type(attempt) is not int
        or not 1 <= attempt <= 256
        or sampling["maximum_attempts"] != 256
        or metadata["seeds"]["warp"] != warp_seed
        or sampling["accepted_attempt_seed"]
        != derive_seed(warp_seed, "scene-v2-geometry-attempt", str(attempt))
        or sampling["requested_size_bin"] != metadata["size_bin"]
        or sampling["actual_size_bin"] != metadata["size_bin"]
        or sampling["foreground_fraction"] != metadata["foreground_fraction"]
        or not _same_seeded_parameters(parameters, _candidate_warp(
            metadata["family"], metadata["size_bin"], sampling["accepted_attempt_seed"]
        ))
    ):
        raise _invalid("coordinate parameters differ from their recorded generator seed")
    return parameters


class SceneCoordinateDataset(Dataset):
    """Hash-checked, positive-only scene-0.3 coordinate supervision.

    ``split`` is explicit. A trainer must construct train/validation only; test
    construction is reserved for a locked evaluation. All sizes are (H, W).
    ``photo_size`` is the original scene size, *not* the model preview size, so
    normalized xy errors become source-pixel errors by multiplying by (W/2,H/2).
    Requested output sizes may differ from the authored flat raster: geometry
    is evaluated at their normalized pixel centers, not rerendered document text.
    Tensors are staged on CPU; this adapter does not train a model or choose its
    execution device. ``original_image`` retains the decoded RGB uint8 pixels
    for evaluation without a second decode; it is not an additional model input.
    """

    def __init__(
        self,
        directory,
        split,
        model_config: ModelConfig | None = None,
        output_size: tuple[int, int] = (192, 144),
        *,
        include_paper_augmentation: bool = False,
    ):
        if not isinstance(split, str) or split not in _SPLITS:
            raise _invalid("split must be train, validation, or test")
        if type(include_paper_augmentation) is not bool:
            raise _invalid("include_paper_augmentation must be boolean")
        config = ModelConfig() if model_config is None else model_config
        if not isinstance(config, ModelConfig):
            raise _invalid("model_config must be ModelConfig or None")
        _size(config.input_size, "model input_size")
        _size(config.grid_size, "model grid_size", minimum=2)
        output_size = _size(output_size, "output_size", minimum=2)
        if any(out < grid for out, grid in zip(output_size, config.grid_size, strict=True)):
            raise _invalid("output_size may not shrink either model grid axis")
        try:
            directory = Path(directory).resolve()
            digest = _manifest_digest(directory)
            raw_manifest = _read_json(directory / "manifest.json", _MANIFEST_LIMIT)
            source_hashes = _data_hashes(include_paper_augmentation)
            if raw_manifest.get("generator_code_sha256") != {
                name: source_hashes[name] for name in _GENERATOR_SOURCES
            }:
                raise _invalid("generator sources differ from the recorded scene implementation")
            _bounded_artifacts(directory, raw_manifest)
            checked = validate_paired_scene_dataset(directory)
            manifest = checked["manifest"]
            photo_size = _size(tuple(manifest["config"]["image_size"]), "photo_size", minimum=128)
            # Read conventions from every positive's metadata, including held
            # out records. This is not image/mask decoding or GT-map allocation.
            for record in manifest["samples"]:
                if record.get("source", "authored") != "authored":
                    raise _invalid("native scene records cannot claim an external source")
                if record["has_page"] is True:
                    paths = _artifact_paths(directory, record)
                    _geometry_contract(_metadata(record, paths), photo_size)
            records = tuple(
                copy.deepcopy(record)
                for record in manifest["samples"]
                if record["split"] == split and record["has_page"] is True
            )
            if not records or len(records) != checked["page_counts"][split]:
                raise _invalid("selected split has missing or inconsistent printed positives")
            templates, captures, paper_records, paper_counts = {}, {}, [], dict.fromkeys(_SPLITS, 0)
            if include_paper_augmentation:
                templates, captures, paper_records, paper_counts = _paper_collection(
                    directory, manifest, photo_size, source_hashes)
                records += tuple(record for record in paper_records if record["split"] == split)
            if _manifest_digest(directory) != digest:
                raise _invalid("manifest changed during validation")
            if _data_hashes(include_paper_augmentation) != source_hashes:
                raise _invalid("coordinate data sources changed during validation")
        except _DATA_ERRORS as error:
            raise _invalid(f"Cannot initialize coordinate scenes: {error}") from error
        self._directory = directory
        self._manifest = copy.deepcopy(manifest)
        self._records = records
        self._templates, self._captures = templates, captures
        self._verified_template_contents = set()
        self.include_paper_augmentation = include_paper_augmentation
        self.split = split
        self.model_config = copy.deepcopy(config)
        self.output_size = output_size
        self.photo_size = photo_size
        self.manifest_sha256 = digest
        self._source_code_sha256 = source_hashes
        self.sample_counts = copy.deepcopy(checked["sample_counts"])
        self.page_counts = copy.deepcopy(checked["page_counts"])
        self.source_page_counts = {"authored": dict(self.page_counts), "paper": paper_counts}
        for name in _SPLITS:
            self.sample_counts[name] += paper_counts[name]
            self.page_counts[name] += paper_counts[name]
        self.paper_augmentation_sha256 = (
            hashlib.sha256(json.dumps(
                manifest["collection"]["paper_augmentation"], sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if include_paper_augmentation else None)

    def _verify_record_artifacts(self, record):
        if record.get("source", "authored") != "paper":
            _bounded_artifacts(self._directory, {"samples": [record]})
            return _artifact_paths(self._directory, record)
        paths = _paper_artifacts(self._directory, record)
        self._source_artifact(self._templates[record["template_id"]])
        if record.get("kind", "simulated") == "capture":
            self._source_artifact(self._captures[record["capture_id"]])
        return paths

    def _source_artifact(self, source):
        # Cache construction has full artifact checks before and after decoding.
        # Its private loader can reuse shared source checks within that interval.
        verified = getattr(self, "_building_source_paths", None)
        if verified is None:
            return _template_artifact(self._directory, source)
        key = (source["path"], source["sha256"], source["bytes"])
        if key not in verified:
            verified[key] = _template_artifact(self._directory, source)
        return verified[key]

    def _load_paper(self, record):
        paths = self._verify_record_artifacts(record)
        template = self._templates[record["template_id"]]
        capture = self._captures.get(record.get("capture_id"))
        metadata, params = _paper_metadata(record, paths, template, self.photo_size, capture)
        # Validate canonical template content only on access, never held-out initialization.
        if template["id"] not in self._verified_template_contents:
            with Image.open(self._source_artifact(template)) as encoded:
                if (encoded.format not in ("PNG", "JPEG")
                        or encoded.width * encoded.height > _TEMPLATE_PIXELS):
                    raise _invalid("paper template pixel budget or encoding differs")
                rgb = np.array(encoded.convert("RGB"), dtype=np.uint8, copy=True)
            if hashlib.sha256(rgb.tobytes(order="C")).hexdigest() != template["content_sha256"]:
                raise _invalid("paper template content hash differs")
            self._verified_template_contents.add(template["id"])
        if capture is not None:
            with Image.open(self._source_artifact(capture)) as encoded:
                if (encoded.format not in ("PNG", "JPEG")
                        or list(reversed(encoded.size)) != capture["original_size"]):
                    raise _invalid("paper capture dimensions differ from source annotation")
        with Image.open(paths["image.png"]) as encoded:
            if encoded.mode != "RGB" or encoded.size != self.photo_size[::-1]:
                raise _invalid("paper image must have the original RGB photo dimensions")
            image = np.array(encoded, copy=True)
        with Image.open(paths["mask.png"]) as encoded:
            if encoded.mode != "L" or encoded.size != self.photo_size[::-1]:
                raise _invalid("paper mask must have the original grayscale photo dimensions")
            mask = np.array(encoded, copy=True)
        if mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
            raise _invalid("paper mask must contain only uint8 0/255")
        paper = _paper_backend()
        geometry = (paper.make_capture_geometry(
                        capture["corners_xy"], tuple(capture["original_size"]),
                        self.output_size, self.photo_size)
                    if capture is not None else
                    paper.make_geometry(self.output_size, self.photo_size, params))
        fraction = float(geometry.photo_page_mask.mean())
        size_bin = "small" if fraction < 0.25 else "medium" if fraction < 0.5 else "large"
        if (fraction <= 0 or size_bin != record["size_bin"]
                or capture is None and metadata["sampling"]["foreground_fraction"] != fraction):
            raise _invalid("paper source mask area differs from its recorded size bin")
        return SceneSample(image, mask == 255, metadata), geometry

    @property
    def records(self):
        """An independent tuple snapshot; callers cannot replace trusted hashes."""
        return copy.deepcopy(self._records)

    @property
    def manifest(self):
        """Metadata-only independent snapshot of the fully validated manifest."""
        return copy.deepcopy(self._manifest)

    @property
    def source_code_sha256(self):
        """Independent snapshot of the generator and target/preprocessing code."""
        return dict(self._source_code_sha256)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        if isinstance(index, (bool, np.bool_)):
            raise _invalid("dataset index must be an integer, not boolean")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise _invalid("dataset index must be an integer") from error
        record = self._records[index]  # Preserve normal Dataset IndexError semantics.
        try:
            building = hasattr(self, "_building_source_paths")
            if not building and _manifest_digest(self._directory) != self.manifest_sha256:
                raise _invalid("manifest changed after dataset initialization")
            if (not building
                    and _data_hashes(self.include_paper_augmentation) != self._source_code_sha256):
                raise _invalid("coordinate data sources changed after initialization")
            if record.get("source", "authored") == "paper":
                sample, geometry = self._load_paper(record)
            else:
                _bounded_artifacts(self._directory, {"samples": [record]})
                sample = load_paired_scene(self._directory, record)
                params = _geometry_contract(sample.metadata, self.photo_size)
                geometry = make_geometry(self.output_size, self.photo_size, params)
            target, valid = geometry.backward_map, geometry.valid_mask
            if (
                sample.image.shape != (*self.photo_size, 3)
                or not np.array_equal(geometry.photo_page_mask, sample.mask)
                or target.dtype != np.float32
                or valid.dtype != np.bool_
                or target.shape != (*self.output_size, 2)
                or valid.shape != self.output_size
                or not valid.any()
                or not np.isfinite(target[valid]).all()
                or not (np.abs(target[valid]) <= 1).all()
                or not np.isnan(target[~valid]).all()
            ):
                raise _invalid("analytic coordinate truth differs from the source scene contract")
            result = {
                "image": preprocess_image(sample.image, self.model_config),
                "original_image": sample.image.copy(order="C"),
                "target": torch.from_numpy(target.copy(order="C")),
                "valid": torch.from_numpy(valid.copy(order="C")),
                "source_mask": torch.from_numpy(sample.mask.copy(order="C")),
                "id": record["id"],
                "document_group": record["document_group"],
                "size_bin": record["size_bin"],
                "family": record["family"],
                "geometry_family": record["family"],
                "source": record.get("source", "authored"),
                "source_kind": record.get("kind", "simulated")
                if record.get("source") == "paper" else "authored",
            }
            self._verify_record_artifacts(record)
            if not building and _manifest_digest(self._directory) != self.manifest_sha256:
                raise _invalid("manifest changed while loading a scene")
            if (not building
                    and _data_hashes(self.include_paper_augmentation) != self._source_code_sha256):
                raise _invalid("coordinate data sources changed while loading a scene")
            return result
        except _DATA_ERRORS as error:
            raise _invalid(f"Cannot load coordinate scene {record['id']}: {error}") from error


def scene_cache_bytes(dataset: SceneCoordinateDataset) -> int:
    """Exact retained tensor payload, excluding bounded metadata/Python overhead."""
    if type(dataset) is not SceneCoordinateDataset or dataset.split not in ("train", "validation"):
        raise _invalid("coordinate cache requires an uncached train or validation dataset")
    input_h, input_w = _size(dataset.model_config.input_size, "model input_size")
    output_h, output_w = _size(dataset.output_size, "output_size", minimum=2)
    photo_h, photo_w = _size(dataset.photo_size, "photo_size", minimum=128)
    return len(dataset) * (
        3 * input_h * input_w * 4 + output_h * output_w * (2 * 4 + 1) + photo_h * photo_w
    )


def _cache_tensor_spec(dataset):
    return {
        "image": (torch.float32, (3, *dataset.model_config.input_size)),
        "target": (torch.float32, (*dataset.output_size, 2)),
        "valid": (torch.bool, dataset.output_size),
        "source_mask": (torch.bool, dataset.photo_size),
    }


def _file_state(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _checked_hash(path, expected, limit, session=None, label="artifact"):
    before = _file_state(path)
    if not path.is_file() or not 0 < before[3] <= limit:
        raise _invalid(f"cached artifact exceeds its resource limit: {path}")
    cached = None if session is None else session.get(path)
    digest = cached[1] if cached is not None and cached[0] == before else _sha256(path)
    if digest != expected:
        raise _invalid(f"cached {label} changed: {path}")
    if before != _file_state(path):
        raise _invalid("cached coordinate artifacts changed during verification")
    if session is not None:
        session[path] = (before, digest)


class CachedSceneCoordinateDataset(SceneCoordinateDataset):
    """Frozen train/validation image, coordinates, visibility and source-page mask.

    Construction brackets checked decoding with full artifact verification.
    ``max_bytes`` bounds retained tensor payload before allocation; source JSON,
    Python objects and one transient decoded sample are additional bounded
    overhead. Items are cloned to prevent training code mutating this snapshot.
    No file is re-read per item. Full integrity checks hash all artifacts;
    lightweight checks compare file identities and metadata between full checks.
    ``source_mask`` stays boolean at original ``photo_size``; it is neither a
    resized preview mask nor the flat-output coordinate-visibility mask.
    """

    def __init__(self, dataset: SceneCoordinateDataset, *, max_bytes: int = 3 * 1024**3):
        required = scene_cache_bytes(dataset)
        if type(max_bytes) is not int or max_bytes < 1:
            raise _invalid("cache max_bytes must be a positive nonboolean integer")
        if required > max_bytes:
            raise _invalid(f"coordinate cache requires {required} bytes; budget is {max_bytes}")
        self._initialize(dataset)
        self.verify_integrity()
        loader = copy.deepcopy(dataset)
        loader._building_source_paths = {}
        # Allocate each field exactly once. Do not hold thousands of separately
        # cloned items and then allocate another complete stack of their bytes.
        self._tensors = {
            key: torch.empty((len(dataset), *shape), dtype=dtype, device="cpu")
            for key, (dtype, shape) in _cache_tensor_spec(dataset).items()
        }
        retained = 0
        for index in range(len(dataset)):
            sample = loader[index]
            for key, (dtype, shape) in _cache_tensor_spec(dataset).items():
                value = sample[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.dtype != dtype
                    or value.layout != torch.strided
                    or tuple(value.shape) != shape
                    or value.requires_grad
                ):
                    raise _invalid(f"coordinate cache received invalid {key} tensor")
                self._tensors[key][index].copy_(value.detach())
                retained += value.numel() * value.element_size()
            del sample
        if retained != required:
            raise _invalid("coordinate cache allocation differs from its preflight size")
        self.verify_integrity()
        self.cache_stats = {"status": "built", "path": None, "bytes": required}

    def _initialize(self, dataset):
        # Preserve the source/data identity, not decoded originals or caller aliases.
        self.__dict__.update(copy.deepcopy(dataset.__dict__))
        self.cache_bytes = scene_cache_bytes(dataset)
        self._expected_artifacts = {}
        self._paper_artifact_files = set()
        for record in self._records:
            for name, relative in record["paths"].items():
                limit = 1024 * 1024 if name == "metadata.json" else _ARTIFACT_LIMIT
                self._expected_artifacts[self._directory / relative] = (
                    record["sha256"][name], limit)
                if record.get("source") == "paper":
                    self._paper_artifact_files.add(self._directory / relative)
        for source in (*self._templates.values(), *self._captures.values()):
            self._expected_artifacts[self._directory / source["path"]] = (
                source["sha256"], _TEMPLATE_LIMIT)
            self._paper_artifact_files.add(self._directory / source["path"])
        self._artifact_files = tuple(self._expected_artifacts)

    def _artifact_snapshot(self):
        for path in self._paper_artifact_files:
            if path.is_symlink() or path.parent.is_symlink():
                raise _invalid("cached paper artifact may not become a symlink")
        return tuple(_file_state(path) for path in self._artifact_files)

    def _verify_persistent(self, *, full=False, session=None):
        try:
            for path, (digest, limit, expected_state) in getattr(
                    self, "_persistent_files", {}).items():
                if path.is_symlink() or _file_state(path) != expected_state:
                    raise _invalid("persistent cache backing file changed; reload explicitly")
                if full:
                    _checked_hash(path, digest, limit, session)
        except _DATA_ERRORS as error:
            raise _invalid(f"Cannot verify persistent cache backing file: {error}") from error

    def verify_integrity(self, *, full=True, _hash_session=None):
        """Hash sources and artifacts, or check metadata against the last full pass.

        ``full=False`` is a change detector for a frozen RAM cache. Call the full
        check before reuse and at run completion; metadata alone cannot prove
        file contents unchanged. Test data is never cached or decoded here.
        """
        if type(full) is not bool:
            raise _invalid("integrity full must be boolean")
        try:
            self._verify_persistent(full=full, session=_hash_session)
            before = self._artifact_snapshot()
            for before_or_after in range(2):
                if _manifest_digest(self._directory) != self.manifest_sha256:
                    raise _invalid("cached coordinate manifest changed")
                if _data_hashes(self.include_paper_augmentation) != self._source_code_sha256:
                    raise _invalid("cached coordinate implementation changed")
                if before_or_after == 0 and full:
                    for path, (digest, limit) in self._expected_artifacts.items():
                        if not path.resolve().is_relative_to(self._directory):
                            raise _invalid("cached coordinate artifact escaped the dataset")
                        label = "template artifact" if limit == _TEMPLATE_LIMIT else "artifact"
                        _checked_hash(path, digest, limit, _hash_session, label)
            if full:
                if before != self._artifact_snapshot():
                    raise _invalid("cached coordinate artifacts changed during verification")
                self._artifact_state = before
            elif before != self._artifact_state:
                raise _invalid("cached coordinate artifact metadata changed")
            self._verify_persistent()
        except _DATA_ERRORS as error:
            raise _invalid(f"Cannot verify cached coordinate scenes: {error}") from error

    def __getitem__(self, index):
        if isinstance(index, (bool, np.bool_)):
            raise _invalid("dataset index must be an integer, not boolean")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise _invalid("dataset index must be an integer") from error
        self._verify_persistent()
        result = {key: value[index].clone() for key, value in self._tensors.items()}
        self._verify_persistent()
        return result

    def fetch_batch(self, indices, *, include_source_mask=True, pin_memory=False):
        """One independent contiguous copy per field; no shared writable cache views.

        A pinned request allocates its final destination directly, avoiding a
        pageable clone/stack followed by a second pinning copy. No device/RNG
        state is touched by the default CPU path. Caller owns batch ordering.
        """
        if type(include_source_mask) is not bool or type(pin_memory) is not bool:
            raise _invalid("batch include_source_mask and pin_memory must be boolean")
        if isinstance(indices, torch.Tensor):
            if (indices.device.type != "cpu" or indices.ndim != 1
                    or indices.dtype not in (torch.int32, torch.int64)):
                raise _invalid("batch indices must be a one-dimensional CPU integer sequence")
        elif not isinstance(indices, (list, tuple, np.ndarray)):
            raise _invalid("batch indices must be a bounded integer sequence")
        if (isinstance(indices, np.ndarray)
                and (indices.ndim != 1 or indices.dtype.kind not in "iu")):
            raise _invalid("batch indices must be a one-dimensional integer sequence")
        if not 1 <= len(indices) <= len(self):
            raise _invalid("batch index count must lie within the dataset size")
        selected = []
        for value in indices:
            if isinstance(value, (bool, np.bool_)):
                raise _invalid("batch indices must not be boolean")
            try:
                index = operator.index(value)
            except TypeError as error:
                raise _invalid("batch indices must be integer") from error
            if not -len(self) <= index < len(self):
                raise IndexError("batch index out of range")
            selected.append(index % len(self))
        index_tensor = torch.tensor(selected, dtype=torch.int64, device="cpu")
        self._verify_persistent()
        result = {}
        for key, value in self._tensors.items():
            if key == "source_mask" and not include_source_mask:
                continue
            destination = torch.empty((len(selected), *value.shape[1:]), dtype=value.dtype,
                                      device="cpu", pin_memory=pin_memory)
            torch.index_select(value, 0, index_tensor, out=destination)
            result[key] = destination
        self._verify_persistent()
        return result


def cache_scene_dataset(
    dataset: SceneCoordinateDataset, *, max_bytes: int = 3 * 1024**3
) -> CachedSceneCoordinateDataset:
    """Opt in to a bounded immutable tensor cache; never accepts the test split."""
    return CachedSceneCoordinateDataset(dataset, max_bytes=max_bytes)


def _cache_identity(datasets):
    value = {
        "schema_version": _CACHE_VERSION,
        "runtime": {"python": platform.python_version(), "platform": platform.system(),
                    "numpy": np.__version__, "pillow": pillow_version,
                    "torch": str(torch.__version__)},
        "splits": {},
    }
    for split, dataset in datasets.items():
        records = json.dumps(dataset.records, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        value["splits"][split] = {
            "manifest_sha256": dataset.manifest_sha256,
            "source_code_sha256": dataset.source_code_sha256,
            "records_sha256": hashlib.sha256(records).hexdigest(), "count": len(dataset),
            "model_config": asdict(dataset.model_config), "output_size": dataset.output_size,
            "photo_size": dataset.photo_size,
            "include_paper_augmentation": dataset.include_paper_augmentation,
            "paper_augmentation_sha256": dataset.paper_augmentation_sha256,
            "tensor_payload_bytes": scene_cache_bytes(dataset),
            "tensors": {key: {"dtype": str(dtype), "shape": [len(dataset), *shape]}
                        for key, (dtype, shape) in _cache_tensor_spec(dataset).items()},
        }
    # A single primitive representation is used by torch.save and JSON sidecars.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _verify_cache_pair(caches):
    """Reuse full hashes only inside this bounded before/after-stat session."""
    states = {split: cache._artifact_snapshot() for split, cache in caches.items()}
    session = {}
    for cache in caches.values():
        cache.verify_integrity(_hash_session=session)
    for split, cache in caches.items():
        if cache._artifact_snapshot() != states[split]:
            raise _invalid("cached coordinate artifacts changed during shared verification")
        cache._verify_persistent()


def verify_scene_caches(datasets):
    """Fully verify a prepared train/validation pair with shared one-session hashes."""
    if (not isinstance(datasets, dict) or datasets.keys() != {"train", "validation"}
            or any(type(cache) is not CachedSceneCoordinateDataset or cache.split != split
                   for split, cache in datasets.items())):
        raise _invalid("integrity verification requires a cached train/validation pair")
    _verify_cache_pair(datasets)


def _validate_cached_tensors(caches, splits):
    """Admit exact bounded, nonaliased storages; scan values in small chunks."""
    if not isinstance(splits, dict) or splits.keys() != caches.keys():
        raise _invalid("persistent cache split inventory differs")
    storages = set()
    for split, cache in caches.items():
        fields = splits[split]
        spec = _cache_tensor_spec(cache)
        if not isinstance(fields, dict) or fields.keys() != spec.keys():
            raise _invalid("persistent cache tensor fields differ")
        retained = 0
        for key, (dtype, shape) in spec.items():
            tensor = fields[key]
            if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
                    or tensor.dtype != dtype or tensor.layout != torch.strided
                    or tuple(tensor.shape) != (len(cache), *shape)
                    or not tensor.is_contiguous() or tensor.requires_grad
                    or tensor.storage_offset() != 0):
                raise _invalid(f"persistent cache has invalid {split}/{key} tensor")
            payload = tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if storage.nbytes() != payload or identity in storages:
                raise _invalid("persistent cache contains oversized or shared tensor storage")
            storages.add(identity)
            retained += payload
        if retained != cache.cache_bytes:
            raise _invalid("persistent cache tensor bytes differ from the memory budget")
        for start in range(0, len(cache), 16):
            image = fields["image"][start:start + 16]
            target = fields["target"][start:start + 16]
            valid = fields["valid"][start:start + 16]
            mask = fields["source_mask"][start:start + 16]
            if (not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any()
                    or not valid.flatten(1).any(1).all() or not mask.flatten(1).any(1).all()
                    or not torch.isfinite(target[valid]).all()
                    or (target[valid].abs() > 1).any()
                    or not torch.isnan(target[~valid]).all()):
                raise _invalid("persistent cache violates image/coordinate/validity semantics")


def _cache_archive_preflight(path, required):
    """Bound serialized storage and pickle metadata before weights_only admission."""
    if not 0 < path.stat().st_size <= required + _CACHE_ARCHIVE_OVERHEAD:
        raise _invalid("persistent cache file exceeds its serialized resource budget")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if (not 1 <= len(members) <= 64 or len(set(names)) != len(names)
                or any(item.compress_type != zipfile.ZIP_STORED for item in members)):
            raise _invalid("persistent cache ZIP has duplicate, compressed, or excessive members")
        prefixes = {name.split("/", 1)[0] for name in names}
        if len(prefixes) != 1:
            raise _invalid("persistent cache ZIP has inconsistent member roots")
        storage_count = storage_bytes = metadata_bytes = 0
        metadata_names = {"data.pkl", "byteorder", "version", ".data/serialization_id",
                          ".format_version", ".storage_alignment"}
        for info in members:
            parts = Path(info.filename).parts
            if (Path(info.filename).is_absolute() or ".." in parts
                    or "\\" in info.filename or len(parts) < 2):
                raise _invalid("persistent cache ZIP has unsafe member paths")
            name = info.filename.split("/", 1)[1]
            if name.startswith("data/") and name[5:].isdigit():
                storage_count += 1
                storage_bytes += info.file_size
            elif name in metadata_names:
                metadata_bytes += info.file_size
            else:
                raise _invalid("persistent cache ZIP has an unexpected member")
        if (storage_count != 8 or storage_bytes != required
                or metadata_bytes > _CACHE_ARCHIVE_OVERHEAD
                or not any(name.endswith("/data.pkl") for name in names)):
            raise _invalid("persistent cache ZIP storage/metadata budget differs")


def _persistent_paths(dataset, persistent_path):
    path = Path(persistent_path).absolute()
    if (path.suffix != ".pt" or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != dataset._directory):
        raise _invalid("persistent cache must be a direct dataset-root .pt file, not a symlink")
    path = path.parent.resolve() / path.name
    metadata = path.with_suffix(path.suffix + ".json")
    lock = path.with_suffix(path.suffix + ".lock")
    if metadata.is_symlink() or lock.is_symlink():
        raise _invalid("persistent cache sidecar/lock may not be symlinks")
    protected = {dataset._directory / "manifest.json", *dataset._artifact_files}
    if path in protected or metadata in protected or lock in protected:
        raise _invalid("persistent cache may not replace a source artifact")
    return path, metadata, lock


def _load_persistent(path, metadata_path, identity, caches, required):
    metadata = _read_json(metadata_path, _CACHE_METADATA_LIMIT)
    if (not isinstance(metadata, dict) or set(metadata) != {
            "schema_version", "identity", "payload_sha256", "payload_bytes"}
            or metadata.get("schema_version") != _CACHE_VERSION
            or metadata.get("identity") != identity
            or not _hash_string(metadata.get("payload_sha256"))
            or type(metadata.get("payload_bytes")) is not int
            or metadata["payload_bytes"] != path.stat().st_size):
        raise _invalid("persistent cache is stale or has invalid metadata; use explicit rebuild")
    before = (_file_state(path), _file_state(metadata_path))
    metadata_digest = _sha256(metadata_path)
    if _sha256(path) != metadata["payload_sha256"]:
        raise _invalid("persistent cache payload checksum mismatch; use explicit rebuild")
    _cache_archive_preflight(path, required)
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(saved, dict) or set(saved) != {"schema_version", "identity", "splits"}
            or saved["schema_version"] != _CACHE_VERSION or saved["identity"] != identity):
        raise _invalid("persistent cache payload identity differs")
    _validate_cached_tensors(caches, saved["splits"])
    if (before != (_file_state(path), _file_state(metadata_path))
            or _sha256(path) != metadata["payload_sha256"]
            or _sha256(metadata_path) != metadata_digest):
        raise _invalid("persistent cache changed during admission")
    for split, cache in caches.items():
        cache._tensors = saved["splits"][split]
        cache._persistent_files = {
            path: (metadata["payload_sha256"], required + _CACHE_ARCHIVE_OVERHEAD, before[0]),
            metadata_path: (metadata_digest, _CACHE_METADATA_LIMIT, before[1]),
        }


def _publish_persistent(path, metadata_path, identity, caches):
    """Each replace is atomic; an interrupted two-file publication fails closed."""
    temporary = []
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".preprocessed-",
                                         suffix=".pt", delete=False) as stream:
            payload_tmp = Path(stream.name)
            temporary.append(payload_tmp)
            payload = {"schema_version": _CACHE_VERSION, "identity": identity,
                       "splits": {split: cache._tensors for split, cache in caches.items()}}
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {"schema_version": _CACHE_VERSION, "identity": identity,
                    "payload_sha256": _sha256(payload_tmp),
                    "payload_bytes": payload_tmp.stat().st_size}
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".preprocessed-",
                                         suffix=".json", encoding="utf-8", delete=False) as stream:
            metadata_tmp = Path(stream.name)
            temporary.append(metadata_tmp)
            json.dump(metadata, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Never unlink an old destination to work around Windows mmap sharing.
        # If replace fails, close previous cache users before an explicit retry.
        os.replace(payload_tmp, path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        for target in temporary:
            target.unlink(missing_ok=True)


def cache_scene_datasets(datasets, *, max_bytes=3 * 1024**3, persistent_path=None, rebuild=False):
    """Cache train/validation together, optionally reusing one audited disk artifact.

    Sources are fully verified on cold/warm entry and completion; warm reuse
    performs no pixel decoding or geometry construction. A stale/corrupt pair
    fails closed unless ``rebuild=True`` explicitly requests replacement. Test
    caching is forbidden. Existing single-dataset RAM APIs remain unchanged.
    The aggregate budget covers retained tensor payload, with bounded JSON,
    one decoded sample, and 16-sample validation chunks as documented overhead.
    """
    started = time.perf_counter()
    if (not isinstance(datasets, dict) or datasets.keys() != {"train", "validation"}
            or any(type(data) is not SceneCoordinateDataset or data.split != split
                   for split, data in datasets.items())):
        raise _invalid("persistent cache requires uncached train and validation datasets only")
    if type(max_bytes) is not int or max_bytes < 1 or type(rebuild) is not bool:
        raise _invalid("cache max_bytes must be positive and rebuild must be boolean")
    if rebuild and persistent_path is None:
        raise _invalid("explicit cache rebuild requires a persistent_path")
    datasets = {split: datasets[split] for split in ("train", "validation")}
    first = datasets["train"]
    for data in datasets.values():
        for name in ("_directory", "manifest_sha256", "_source_code_sha256", "model_config",
                     "output_size", "photo_size", "include_paper_augmentation"):
            if getattr(data, name) != getattr(first, name):
                raise _invalid("train/validation preprocessing identity differs")
    required = sum(scene_cache_bytes(data) for data in datasets.values())
    if required > max_bytes:
        raise _invalid(f"coordinate cache requires {required} bytes; budget is {max_bytes}")
    caches = {}
    for split, data in datasets.items():
        cache = CachedSceneCoordinateDataset.__new__(CachedSceneCoordinateDataset)
        cache._initialize(data)
        caches[split] = cache
    identity = _cache_identity(datasets)
    lock = None
    owns_lock = False
    lock_identity = None
    try:
        if persistent_path is not None:
            path, metadata_path, lock = _persistent_paths(caches["train"], persistent_path)
            with lock.open("x", encoding="ascii") as stream:
                owns_lock = True
                info = os.fstat(stream.fileno())
                lock_identity = (info.st_dev, info.st_ino)
                stream.write(str(os.getpid()) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            present = path.exists() or metadata_path.exists()
            if present and not rebuild:
                if not path.is_file() or not metadata_path.is_file():
                    raise _invalid("incomplete persistent cache pair; use explicit rebuild")
                _verify_cache_pair(caches)
                _load_persistent(path, metadata_path, identity, caches, required)
                _verify_cache_pair(caches)
                status = "loaded"
            else:
                present = False
        else:
            present = False
        if not present:
            remaining = max_bytes
            for split, data in datasets.items():
                caches[split] = cache_scene_dataset(data, max_bytes=remaining)
                remaining -= caches[split].cache_bytes
            _validate_cached_tensors(caches, {k: v._tensors for k, v in caches.items()})
            _verify_cache_pair(caches)
            if persistent_path is not None:
                _publish_persistent(path, metadata_path, identity, caches)
                # Verify the exact persisted bytes/metadata while retaining the
                # already checked RAM storage; do not allocate a second cache.
                for cache in caches.values():
                    cache._verify_persistent()
                saved_meta = _read_json(metadata_path, _CACHE_METADATA_LIMIT)
                _cache_archive_preflight(path, required)
                if (saved_meta["identity"] != identity
                        or saved_meta["payload_sha256"] != _sha256(path)):
                    raise _invalid("persistent cache publication verification failed")
                # Publication can take appreciable time for multi-GiB caches.
                # Source changes during serialization/fsync must fail this run.
                _verify_cache_pair(caches)
            status = "rebuilt" if rebuild else "built"
        seconds = time.perf_counter() - started
        for cache in caches.values():
            cache.cache_stats = {"status": status, "path": str(path) if persistent_path else None,
                                 "seconds": seconds, "bytes": required}
        return caches
    except FileExistsError as error:
        raise _invalid("persistent cache is locked; inspect the owner before retrying") from error
    except (*_DATA_ERRORS, RuntimeError, EOFError, pickle.UnpicklingError,
            zipfile.BadZipFile) as error:
        raise _invalid(f"Cannot reuse/publish persistent cache: {error}") from error
    finally:
        if owns_lock:
            if lock.exists() and _file_state(lock)[:2] == lock_identity:
                lock.unlink()
