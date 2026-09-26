"""Training data for dewarp models: licensed sources, adapters and augmentation.

A data spec (JSON, schema ``flattener-dewarp-data-1``) lists sources and the license
*track* of each: ``release`` (license-clean; weights may ship) or ``research`` (weights
stay private). Doc3D, the UVDoc dataset and the raw SyntheticDoc layout are
research-only kinds. A release-track dataset accepts release sources only, and a
converted release source accepts only samples labelled release with an allowlisted
license class. Source roots may name environment variables (``${FLATTENER_DATA}/...``);
the spec hash uses the unexpanded text.

Items hold the model input as uint8 ``image`` (3, H, W), the target 2-D grid in the
flattener convention (edge-aligned nodes, align_corners=False values relative to the
returned image; see ``coords``), a node validity mask, an optional UVDoc-normalized
3-D grid and string metadata including the track. Training items also carry
``color`` and ``noise_seed``: per-sample photometric parameters that
``apply_color_batch`` applies to a float batch on the training device. Evaluation
items stack one entry per ``ViewConfig`` along dim 0 of ``image``, ``grid2d`` and
``valid``.

Converted layout (``kind: "converted"``, written by the data pipeline)::

    <root>/manifest.jsonl          one JSON object per sample (preferred), or
    <root>/**/<sample>.json        one object per file (scanned when no manifest)

    {"schema": "flattener-dewarp-sample-1", "sample_id": "...", "split": "train",
     "image": "relative/path.jpg", "backward_map": "relative/path.npy",
     "backward_map_convention": "flattener" | "syntheticdoc",
     "original_render_size": [1440, 1024], "track": "release",
     "license_class": "clean", "image_sha256": "...", "backward_map_sha256": "...", ...}

Paths are relative to the JSON document that lists them. ``backward_map`` is a
center-aligned (H, W, 2) float16/float32 map holding flattener values, or raw
SyntheticDoc values when the convention is ``"syntheticdoc"`` (then
``original_render_size`` (H, W) of the upstream render is required; the image may be
a resized copy). Entries without ``split`` belong to the source's split. The optional
sha256 and ``*_bytes`` fields feed the content digest and the size check.

``kind: "syntheticdoc_compact"`` reads the training copy written by
``tools/syntheticdoc/compact.py``: the converted layout plus ``compact.json`` and
required per-file sha256 and sizes.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.nn import functional as F

from ..contracts import FlattenerError, Status
from .coords import (
    map_to_grid,
    pixels_to_flattener,
    resize_grid,
    syntheticdoc_to_flattener,
)

SPEC_SCHEMA = "flattener-dewarp-data-1"
SAMPLE_SCHEMA = "flattener-dewarp-sample-1"
COMPACT_SCHEMA = "flattener-syntheticdoc-compact-1"
TRACKS = ("release", "research")
KINDS = ("converted", "syntheticdoc_compact", "syntheticdoc_raw", "doc3d", "uvdoc")
RESEARCH_ONLY_KINDS = frozenset({"syntheticdoc_raw", "doc3d", "uvdoc"})
# Release-track samples need a license class whose ``license_key`` is on this list.
RELEASE_LICENSE_KEYS = frozenset(
    {
        "clean",  # the data pipeline's class for permissive sub-sources
        "own",
        "cc0",
        "by",  # CC BY, any version
        "pd",
        "pdm",
        "public-domain",
        "mit",
        "apache",
        "bsd",
        "bsd-clause",
        "odc-by",
        "cdla-permissive",
    }
)
# Upstream UVDoc 3-D normalization constants (xmax, xmin, ymax, ymin, zmax, zmin).
DOC3D_GRID3D_RANGE = (1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497)
UVDOC_GRID3D_RANGE = (0.11433014, -0.12551452, 0.12401487, -0.12401487, 0.1952378, -0.1952378)
VIEW_CROPS = ("page", "full_frame")
# ``color`` item layout: on, brightness, contrast, saturation, gain r/g/b, gamma,
# grayscale, blur on, blur sigma, noise on, noise std.
COLOR_PARAMS = 13
IDENTITY_COLOR = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.float32)
_UNSET_VARIABLE = re.compile(r"\$\{?\w+\}?|%\w+%" if os.name == "nt" else r"\$\{?\w+\}?")


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def expand_path(text: str) -> str:
    """``text`` with ``${VAR}``/``$VAR`` (and ``%VAR%`` on Windows) expanded; unset names raise."""
    expanded = os.path.expandvars(text)
    unset = _UNSET_VARIABLE.findall(expanded)
    if unset:
        raise _error("unset_variable", f"{text!r}: set environment variable(s) {unset}")
    return expanded


def license_key(license_class: str) -> str:
    """Lowercase alphanumeric tokens without version numbers or the words cc/license.

    "CC BY-NC-SA 4.0" -> "by-nc-sa".
    """
    tokens = re.findall(r"[a-z0-9]+", license_class.lower())
    return "-".join(t for t in tokens if not t.isdigit() and t not in {"cc", "license", "licence"})


def release_license_ok(license_class: str) -> bool:
    return license_key(license_class) in RELEASE_LICENSE_KEYS


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    kind: str
    root: str
    track: str
    split: str = "train"
    license: str = ""
    weight: float | None = None
    ids_file: str | None = None
    max_samples: int | None = None

    def __post_init__(self) -> None:
        for name in ("name", "root", "split"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise _error("invalid_spec", f"source {name} must be a non-empty string")
        if self.kind not in KINDS:
            raise _error("invalid_spec", f"source {self.name!r}: kind must be one of {KINDS}")
        if self.track not in TRACKS:
            raise _error("invalid_spec", f"source {self.name!r}: track must be one of {TRACKS}")
        if self.kind in RESEARCH_ONLY_KINDS and self.track != "research":
            raise _error(
                "track_violation", f"source {self.name!r}: kind {self.kind!r} is research-only"
            )
        if not isinstance(self.license, str) or (self.track == "release" and not self.license):
            raise _error("invalid_spec", f"source {self.name!r}: release sources need a license")
        weight = self.weight
        if weight is not None and (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise _error("invalid_spec", f"source {self.name!r}: weight must be positive")
        if self.ids_file is not None and not isinstance(self.ids_file, str):
            raise _error("invalid_spec", f"source {self.name!r}: ids_file must be a string")
        count = self.max_samples
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 1
        ):
            raise _error("invalid_spec", f"source {self.name!r}: max_samples must be >= 1")


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Sources by track. ``base_dir`` resolves relative roots and stays out of the hash."""

    name: str
    track: str
    sources: tuple[SourceSpec, ...]
    base_dir: str = "."

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _error("invalid_spec", "spec name must be a non-empty string")
        if self.track not in TRACKS:
            raise _error("invalid_spec", f"spec track must be one of {TRACKS}")
        names = [source.name for source in self.sources]
        if not names or len(set(names)) != len(names):
            raise _error("invalid_spec", "spec needs sources with unique names")
        if self.track == "release":
            research = [s.name for s in self.sources if s.track != "release"]
            if research:
                raise _error("track_violation", f"release spec lists research sources {research}")

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str | Path = ".") -> DataSpec:
        if not isinstance(data, dict) or data.get("schema") != SPEC_SCHEMA:
            raise _error(
                "invalid_spec", f"data spec must be a JSON object with schema {SPEC_SCHEMA}"
            )
        unknown = set(data) - {"schema", "name", "track", "sources"}
        allowed = {f.name for f in fields(SourceSpec)}
        sources = data.get("sources")
        if unknown or not isinstance(sources, list):
            raise _error("invalid_spec", f"bad spec keys {sorted(unknown)} or missing sources list")
        for source in sources:
            if not isinstance(source, dict) or set(source) - allowed:
                raise _error("invalid_spec", f"bad source entry {source!r}")
        return cls(
            name=data.get("name", ""),
            track=data.get("track", ""),
            sources=tuple(SourceSpec(**source) for source in sources),
            base_dir=str(base_dir),
        )

    @classmethod
    def load(cls, path: str | Path) -> DataSpec:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path.resolve().parent)

    def to_dict(self) -> dict[str, Any]:
        sources = [asdict(source) for source in self.sources]
        return {"schema": SPEC_SCHEMA, "name": self.name, "track": self.track, "sources": sources}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict())).hexdigest()

    def root(self, source: SourceSpec) -> Path:
        return Path(self.base_dir, expand_path(source.root))


def require_track(spec: DataSpec, track: str) -> None:
    """Accept a release-track build only from a release spec with release sources."""
    if track not in TRACKS:
        raise _error("invalid_track", f"track must be one of {TRACKS}")
    if track == "release":
        bad = [
            s.name for s in spec.sources if s.track != "release" or s.kind in RESEARCH_ONLY_KINDS
        ]
        if spec.track != "release" or bad:
            raise _error(
                "track_violation",
                f"release track refuses spec {spec.name!r} (track {spec.track}); "
                f"research-only sources: {bad}",
            )


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Seeded crop and appearance augmentation (UVDoc-style), applied in training only.

    ``margin`` is the per-side crop margin around the page (grid extent) as a fraction
    of the page extent. Axis-aligned crops are clipped to the image (``clip_to_image``,
    as upstream); rotated crops may leave it and replicate the border (upstream
    ``BORDER_REPLICATE``). ``full_frame_prob`` keeps the whole image.
    ``rotation_kinds`` limits rotation to those source kinds (empty: all; upstream UVDoc
    rotates its own dataset only). ``jpeg_draft`` lets JPEG decoding skip resolution
    the crop discards (DCT scaling by 1/2, 1/4 or 1/8).
    """

    enabled: bool = True
    margin: tuple[float, float] = (0.0, 0.05)
    clip_to_image: bool = True
    full_frame_prob: float = 0.0
    rotation_deg: float = 0.0
    rotation_prob: float = 0.5
    rotation_kinds: tuple[str, ...] = ()
    color_prob: float = 0.95
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.3
    channel_gain: float = 0.05
    gamma: float = 0.2
    grayscale_prob: float = 0.05
    blur_prob: float = 0.1
    noise_prob: float = 0.5
    noise_std: float = 0.02
    jpeg_draft: bool = True

    def __post_init__(self) -> None:
        margin = tuple(self.margin)
        object.__setattr__(self, "margin", margin)
        if len(margin) != 2 or not 0 <= margin[0] <= margin[1] <= 1:
            raise ValueError("margin must be (low, high) with 0 <= low <= high <= 1")
        kinds = self.rotation_kinds
        kinds = tuple(kinds) if isinstance(kinds, (list, tuple)) else (kinds,)
        object.__setattr__(self, "rotation_kinds", kinds)
        if not set(kinds) <= set(KINDS):
            raise ValueError(f"rotation_kinds must be a list of {KINDS}")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in ("enabled", "clip_to_image", "jpeg_draft"):
                if not isinstance(value, bool):
                    raise ValueError(f"{item.name} must be a bool")
            elif item.name not in ("margin", "rotation_kinds") and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0
            ):
                raise ValueError(f"{item.name} must be a non-negative number")
            elif item.name.endswith("_prob") and value > 1:
                raise ValueError(f"{item.name} must be <= 1")
        if self.rotation_deg > 45 or max(self.brightness, self.contrast, self.channel_gain) >= 1:
            raise ValueError("rotation_deg must be <= 45; brightness/contrast/gain < 1")

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class ViewConfig:
    """An evaluation input: a fixed crop, resized to the model input like ``predict``.

    ``page`` crops the page's bounding box (from the target grid) plus per-side margins,
    as fractions of the page extent, drawn from ``margin`` by a generator keyed by the
    view name and the sample (fixed when low == high). The box snaps outward to whole
    pixels and is clipped to the image, like a saved crop file. ``full_frame`` uses the
    whole image. The crop is resampled bilinearly without antialiasing (values within
    one level of ``predict``'s float resize).
    """

    name: str
    crop: str = "page"
    margin: tuple[float, float] = (0.025, 0.025)

    def __post_init__(self) -> None:
        margin = tuple(self.margin)
        object.__setattr__(self, "margin", margin)
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError("view name must be a non-empty [A-Za-z0-9_.-] string")
        if self.crop not in VIEW_CROPS:
            raise ValueError(f"view crop must be one of {VIEW_CROPS}")
        if (
            len(margin) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in margin)
            or not 0 <= margin[0] <= margin[1] <= 1
        ):
            raise ValueError("view margin must be (low, high) with 0 <= low <= high <= 1")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "crop": self.crop, "margin": list(self.margin)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViewConfig:
        if not isinstance(data, dict) or set(data) - {"name", "crop", "margin"}:
            raise ValueError(f"bad view {data!r}")
        return cls(**data)


@dataclass(slots=True)
class RawSample:
    image: Image.Image | NDArray[np.uint8]  # opened lazily by the sources
    grid: NDArray[np.float32]  # (Gh, Gw, 2) edge-aligned, flattener values w.r.t. image
    valid: NDArray[np.bool_]
    grid3d: NDArray[np.float32] | None
    meta: dict[str, str] = field(default_factory=dict)
    size: tuple[int, int] = field(init=False)  # (height, width) as stored, before any draft

    def __post_init__(self) -> None:
        if isinstance(self.image, Image.Image):
            self.size = (self.image.size[1], self.image.size[0])
        else:
            self.size = (self.image.shape[0], self.image.shape[1])


def _read_matrix(path_without_suffix: Path, key: str) -> NDArray:
    """Upstream ``np.array(h5py.File(mat)[key][:].T)``, or the same array saved as .npy."""
    npy = path_without_suffix.with_name(path_without_suffix.name + ".npy")
    if npy.is_file():
        return np.load(npy, allow_pickle=False)
    mat = path_without_suffix.with_name(path_without_suffix.name + ".mat")
    try:
        import h5py
    except ImportError as error:
        raise _error(
            "missing_dependency", f"{mat} needs h5py (MATLAB v7.3) or a pre-converted {npy.name}"
        ) from error
    with h5py.File(mat, "r") as stream:
        return np.array(stream[key][:]).T


def _normalize_3d(grid3d: NDArray, ranges: tuple[float, ...], order: str) -> NDArray[np.float32]:
    xmx, xmn, ymx, ymn, zmx, zmn = ranges
    bounds = {"x": (xmn, xmx), "y": (ymn, ymx), "z": (zmn, zmx)}
    result = np.empty(grid3d.shape, dtype=np.float64)
    for channel, axis in enumerate(order):
        low, high = bounds[axis]
        result[..., channel] = (grid3d[..., channel] - low) / (high - low)
    return result.astype(np.float32)


class _FileSizes:
    """File sizes from one ``os.scandir`` per directory (cheap on NTFS and NVMe)."""

    def __init__(self) -> None:
        self.listings: dict[str, dict[str, int]] = {}

    def __call__(self, path: str | Path) -> int | None:
        directory, name = os.path.split(os.fspath(path))
        listing = self.listings.get(directory)
        if listing is None:
            listing = {}
            try:
                with os.scandir(directory or ".") as entries:
                    for entry in entries:
                        if entry.is_file():
                            listing[entry.name] = entry.stat().st_size
            except FileNotFoundError:
                pass
            self.listings[directory] = listing
        return listing.get(name)


class _Source:
    """One spec source: indexes records on construction; ``load`` opens one sample.

    Construction also digests the sample contents (manifest sha256 where given, file
    sizes otherwise) so a resume can refuse re-encoded data that keeps its ids.
    """

    def __init__(self, spec: SourceSpec, root: Path, grid_size: tuple[int, int]) -> None:
        if not root.is_dir():
            raise _error("missing_data", f"source {spec.name!r}: root {root} is not a directory")
        self.spec, self.root, self.grid_size = spec, root, grid_size
        self.hashes: dict[str, tuple[str, ...]] = {}
        records = self._index()
        if spec.ids_file is not None:
            wanted = set(Path(root, spec.ids_file).read_text(encoding="utf-8").split())
            records = [r for r in records if r["sample_id"] in wanted]
        if spec.max_samples is not None and len(records) > spec.max_samples:
            keep = np.unique(np.linspace(0, len(records) - 1, spec.max_samples).round().astype(int))
            records = [records[i] for i in keep]
        if not records:
            raise _error("missing_data", f"source {spec.name!r}: no samples for split {spec.split}")
        self.records = records
        self.content_sha256 = self._content_digest()
        self.hashes = {}  # pickled into loader workers; the digest is enough

    def __len__(self) -> int:
        return len(self.records)

    def _index(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def load(self, index: int) -> RawSample:
        raise NotImplementedError

    def content_files(self, record: dict[str, Any], sizes: _FileSizes) -> list[str]:
        """Files whose sizes identify the sample (with the manifest hashes, if any)."""
        raise NotImplementedError

    def _content_digest(self) -> str:
        sizes = _FileSizes()
        digest = hashlib.sha256()
        for record in self.records:
            files = self.content_files(record, sizes)
            found = [sizes(path) for path in files]
            missing = [path for path, size in zip(files, found, strict=True) if size is None]
            if missing:
                raise _error("missing_data", f"source {self.spec.name!r}: missing {missing[:3]}")
            expected = record.get("bytes")
            if expected is not None and tuple(found) != tuple(expected):
                raise _error(
                    "stale_manifest",
                    f"source {self.spec.name!r}: {record['sample_id']} files have sizes {found}, "
                    f"the manifest records {list(expected)}",
                )
            hashes = self.hashes.get(record["sample_id"], ())
            line = "\t".join([record["sample_id"], *hashes, *map(str, found)])
            digest.update(line.encode("utf-8") + b"\n")
        return digest.hexdigest()

    def _meta(self, record: dict[str, Any], track: str, license_text: str) -> dict[str, str]:
        return {
            "source": self.spec.name,
            "kind": self.spec.kind,
            "sample_id": str(record["sample_id"]),
            "track": track,
            "license": license_text,
        }

    def _grid(self, values: NDArray) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        grid = values if values.shape[:2] == self.grid_size else resize_grid(values, self.grid_size)
        return grid.astype(np.float32), np.isfinite(grid).all(axis=-1)

    def describe(self) -> dict[str, Any]:
        """Name, track, license manifest (sample counts per license class), a digest of the
        sample list (ids, tracks, license classes; paths excluded) and the content digest."""
        fallback = self.spec.license or self.spec.kind
        classes = Counter(r.get("license_class") or fallback for r in self.records)
        listing = "\n".join(
            f"{r['sample_id']}\t{r.get('track', '')}\t{r.get('license_class', '')}"
            for r in self.records
        )
        return {
            "name": self.spec.name,
            "kind": self.spec.kind,
            "track": self.spec.track,
            "samples": len(self.records),
            "license_classes": dict(sorted(classes.items())),
            "records_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
            "content_sha256": self.content_sha256,
        }


class ConvertedSource(_Source):
    require_manifest = False
    require_hashes = False

    def _documents(self) -> list[tuple[Any, Path]]:
        manifest = self.root / "manifest.jsonl"
        if manifest.is_file():
            lines = manifest.read_text(encoding="utf-8").splitlines()
            return [(json.loads(line), self.root) for line in lines if line.strip()]
        if self.require_manifest:
            raise _error("missing_data", f"source {self.spec.name!r}: {manifest} is missing")
        documents = []
        for path in sorted(self.root.rglob("*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(entry, dict) and entry.get("schema") == SAMPLE_SCHEMA:
                documents.append((entry, path.parent))
        return documents

    def _index(self) -> list[dict[str, Any]]:
        records = []
        required = {"schema", "sample_id", "image", "backward_map", "track"}
        for entry, base in self._documents():
            if not isinstance(entry, dict) or not required <= set(entry):
                raise _error("invalid_data", f"source {self.spec.name!r}: bad entry {entry!r}")
            if entry["schema"] != SAMPLE_SCHEMA:
                raise _error(
                    "invalid_data", f"{entry['sample_id']}: schema must be {SAMPLE_SCHEMA}"
                )
            if entry.get("split", self.spec.split) != self.spec.split:
                continue
            track, license_class = entry["track"], str(entry.get("license_class", ""))
            if track not in TRACKS:
                raise _error("invalid_data", f"{entry['sample_id']}: track must be in {TRACKS}")
            if self.spec.track == "release" and (
                track != "release" or not release_license_ok(license_class)
            ):
                raise _error(
                    "track_violation",
                    f"release source {self.spec.name!r} lists {entry['sample_id']} "
                    f"(track {track}, license class {license_class!r}); release samples need "
                    f"track 'release' and a license class in {sorted(RELEASE_LICENSE_KEYS)}",
                )
            convention = entry.get("backward_map_convention", "flattener")
            render_size = entry.get("original_render_size")
            if convention not in ("flattener", "syntheticdoc") or (
                convention == "syntheticdoc"
                and (not isinstance(render_size, list) or len(render_size) != 2)
            ):
                raise _error("invalid_data", f"{entry['sample_id']}: bad map convention")
            sample_id = str(entry["sample_id"])
            hashes = tuple(entry.get(k) for k in ("image_sha256", "backward_map_sha256"))
            if all(isinstance(h, str) and h for h in hashes):
                self.hashes[sample_id] = hashes
            elif self.require_hashes:
                raise _error("invalid_data", f"{sample_id}: image/backward_map sha256 missing")
            sizes = tuple(entry.get(k) for k in ("image_bytes", "backward_map_bytes"))
            records.append(  # compact: records are pickled into every loader worker
                {
                    "sample_id": sample_id,
                    "image": str(Path(base, entry["image"])),
                    "backward_map": str(Path(base, entry["backward_map"])),
                    "track": track,
                    "license_class": license_class,
                    "render_size": tuple(render_size) if convention == "syntheticdoc" else None,
                    "bytes": sizes if all(isinstance(s, int) for s in sizes) else None,
                }
            )
        return records

    def content_files(self, record: dict[str, Any], sizes: _FileSizes) -> list[str]:
        return [record["image"], record["backward_map"]]

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        image = Image.open(record["image"])
        values = np.load(record["backward_map"], allow_pickle=False)
        if record["render_size"] is not None:
            values = syntheticdoc_to_flattener(values, record["render_size"])
        grid = map_to_grid(values, self.grid_size)
        meta = self._meta(record, record["track"], record["license_class"] or self.spec.license)
        return RawSample(image, grid, np.isfinite(grid).all(axis=-1), None, meta)


class SyntheticDocCompactSource(ConvertedSource):
    """Training copy from ``tools/syntheticdoc/compact.py`` (``compact.json`` + manifest)."""

    require_manifest = True
    require_hashes = True

    def _index(self) -> list[dict[str, Any]]:
        path = self.root / "compact.json"
        if not path.is_file():
            raise _error("missing_data", f"source {self.spec.name!r}: {path} is missing")
        settings = json.loads(path.read_text(encoding="utf-8"))
        if settings.get("schema") != COMPACT_SCHEMA:
            raise _error("invalid_data", f"{path}: schema must be {COMPACT_SCHEMA}")
        if self.spec.track == "release" and settings.get("track") != "release":
            raise _error("track_violation", f"{path}: copy of track {settings.get('track')!r}")
        return super()._index()


class SyntheticDocRawSource(_Source):
    """Upstream samples as stored locally: flat ``syntheticdoc-{split}-{id}-render.png``,
    ``-backward_map.npy`` (177x121 raw values), ``-metadata.json``; or per-sample
    directories holding ``render.png`` and ``backward_map.npy``."""

    LICENSE = "SyntheticDoc CC BY 4.0 distribution; sub-source licenses unclassified"

    def _index(self) -> list[dict[str, Any]]:
        prefix = f"syntheticdoc-{self.spec.split}-"
        records = [
            {
                "sample_id": path.name[len("syntheticdoc-") : -len("-backward_map.npy")],
                "image": str(path.with_name(path.name.replace("-backward_map.npy", "-render.png"))),
                "backward_map": str(path),
            }
            for path in sorted(self.root.glob(f"{prefix}*-backward_map.npy"))
        ]
        if not records:
            records = [
                {
                    "sample_id": str(path.parent.relative_to(self.root).as_posix()),
                    "image": str(path.with_name("render.png")),
                    "backward_map": str(path),
                }
                for path in sorted(self.root.rglob("backward_map.npy"))
            ]
        return records

    def content_files(self, record: dict[str, Any], sizes: _FileSizes) -> list[str]:
        return [record["image"], record["backward_map"]]

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        image = Image.open(record["image"])
        raw = np.load(record["backward_map"], allow_pickle=False)
        size = (image.size[1], image.size[0])
        grid = map_to_grid(syntheticdoc_to_flattener(raw, size), self.grid_size)
        meta = self._meta(record, "research", self.LICENSE)
        return RawSample(image, grid, np.isfinite(grid).all(axis=-1), None, meta)


def _matrix_file(path_without_suffix: Path, sizes: _FileSizes) -> str:
    npy = path_without_suffix.with_name(path_without_suffix.name + ".npy")
    return str(npy if sizes(npy) is not None else npy.with_suffix(".mat"))


class Doc3DSource(_Source):
    """Upstream ``data_doc3D.py`` layout: ``img/<id>.png``, ``grid2D/<id>.mat`` (pixel
    coordinates of the 448x448 image), ``grid3D/<id>.mat``; ids from ``traindoc.txt``
    (train) or ``valdoc3D.txt`` (val) unless ``ids_file`` is given."""

    SPLIT_FILES = {"train": "traindoc.txt", "val": "valdoc3D.txt"}

    def _index(self) -> list[dict[str, Any]]:
        name = self.spec.ids_file or self.SPLIT_FILES.get(self.spec.split)
        if name is None:
            raise _error("invalid_spec", f"doc3d split {self.spec.split!r} needs ids_file")
        ids = Path(self.root, name).read_text(encoding="utf-8").split()
        return [{"sample_id": sample_id} for sample_id in ids]

    def content_files(self, record: dict[str, Any], sizes: _FileSizes) -> list[str]:
        sample_id = record["sample_id"]
        return [
            str(self.root / "img" / f"{sample_id}.png"),
            _matrix_file(self.root / "grid2D" / sample_id, sizes),
            _matrix_file(self.root / "grid3D" / sample_id, sizes),
        ]

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        sample_id = record["sample_id"]
        image = Image.open(self.root / "img" / f"{sample_id}.png")
        pixels = _read_matrix(self.root / "grid2D" / sample_id, "grid2D")
        grid, valid = self._grid(pixels_to_flattener(pixels, (image.size[1], image.size[0])))
        grid3d = _normalize_3d(
            _read_matrix(self.root / "grid3D" / sample_id, "grid3D"), DOC3D_GRID3D_RANGE, "zyx"
        )
        grid3d[..., 1] = 1.0 - grid3d[:, ::-1, 1]  # upstream flips and inverts channel 1
        meta = self._meta(record, "research", self.spec.license or "Doc3D (non-commercial parts)")
        return RawSample(image, grid, valid, self._grid(grid3d)[0], meta)


class UVDocSource(_Source):
    """Upstream ``data_UVDoc.py`` layout: ``img/<id>.png``, ``metadata_sample/<id>.json``
    (``geom_name``), ``grid2d/<geom>.mat`` (89x61 pixel coordinates), ``grid3d/<geom>.mat``."""

    def _index(self) -> list[dict[str, Any]]:
        return [{"sample_id": p.stem} for p in sorted((self.root / "img").glob("*.png"))]

    def content_files(self, record: dict[str, Any], sizes: _FileSizes) -> list[str]:
        sample_id = record["sample_id"]
        return [
            str(self.root / "img" / f"{sample_id}.png"),
            str(self.root / "metadata_sample" / f"{sample_id}.json"),
        ]

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        sample_id = record["sample_id"]
        meta_path = self.root / "metadata_sample" / f"{sample_id}.json"
        geometry = json.loads(meta_path.read_text(encoding="utf-8"))["geom_name"]
        image = Image.open(self.root / "img" / f"{sample_id}.png")
        pixels = _read_matrix(self.root / "grid2d" / geometry, "grid2d")
        grid, valid = self._grid(pixels_to_flattener(pixels, (image.size[1], image.size[0])))
        grid3d = _normalize_3d(
            _read_matrix(self.root / "grid3d" / geometry, "grid3d"), UVDOC_GRID3D_RANGE, "xyz"
        )
        meta = self._meta(record, "research", self.spec.license or "UVDoc dataset (NC textures)")
        return RawSample(image, grid, valid, self._grid(grid3d)[0], meta)


SOURCE_TYPES: dict[str, type[_Source]] = {
    "converted": ConvertedSource,
    "syntheticdoc_compact": SyntheticDocCompactSource,
    "syntheticdoc_raw": SyntheticDocRawSource,
    "doc3d": Doc3DSource,
    "uvdoc": UVDocSource,
}


def _rotation(theta: float) -> NDArray[np.float64]:
    cos, sin = math.cos(theta), math.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])


@dataclass(frozen=True, slots=True)
class Crop:
    """Output normalized o in [-1, 1]^2 maps to source edge-pixel ``center + R diag(half) o``."""

    theta: float
    center: NDArray[np.float64]  # (x, y)
    half: NDArray[np.float64]  # (x, y), in the rotated crop frame

    @property
    def linear(self) -> NDArray[np.float64]:
        return _rotation(self.theta) * self.half

    def target(self, points: NDArray[np.float64]) -> NDArray[np.float32]:
        """Edge-pixel source points -> normalized output coordinates (the target grid)."""
        return (((points - self.center) @ _rotation(self.theta)) / self.half).astype(np.float32)


def _edge_points(sample: RawSample) -> NDArray[np.float64]:
    height, width = sample.size
    return (sample.grid.astype(np.float64) + 1.0) / 2.0 * np.array([width, height])


def plan_crop(sample: RawSample, augment: AugmentConfig, rng: np.random.Generator | None) -> Crop:
    """Training crop around the page extent: seeded margins, rotation and full frame.

    Without ``rng`` the margins are the midpoint of ``augment.margin``.
    """
    height, width = sample.size
    size = np.array([width, height], dtype=np.float64)
    center = size / 2.0
    theta, full_frame = 0.0, False
    if rng is not None:
        full_frame = bool(rng.random() < augment.full_frame_prob)
        rotates = not augment.rotation_kinds or sample.meta.get("kind") in augment.rotation_kinds
        if augment.rotation_deg > 0 and rotates and rng.random() < augment.rotation_prob:
            theta = math.radians(rng.uniform(-augment.rotation_deg, augment.rotation_deg))
    if full_frame or not sample.valid.any():
        return Crop(0.0, center, size / 2.0)
    rotation = _rotation(theta)
    rotated = (_edge_points(sample)[sample.valid] - center) @ rotation  # page in the crop frame
    low, high = rotated.min(axis=0), rotated.max(axis=0)
    extent = np.maximum(high - low, 1.0)
    if rng is None:
        margins = np.full(4, sum(augment.margin) / 2.0)
    else:
        margins = rng.uniform(augment.margin[0], augment.margin[1], size=4)
    low, high = low - margins[:2] * extent, high + margins[2:] * extent
    if theta == 0.0 and augment.clip_to_image:
        low, high = np.maximum(low, -center), np.minimum(high, center)
    half = np.maximum((high - low) / 2.0, 0.5)
    return Crop(theta, center + rotation @ ((low + high) / 2.0), half)


def view_box(sample: RawSample, view: ViewConfig, rng: np.random.Generator) -> tuple[int, ...]:
    """Whole-pixel crop box ``(left, top, right, bottom)`` of an evaluation view."""
    height, width = sample.size
    if view.crop == "full_frame" or not sample.valid.any():
        return 0, 0, width, height
    points = _edge_points(sample)[sample.valid]
    low, high = points.min(axis=0), points.max(axis=0)
    extent = np.maximum(high - low, 1.0)
    margins = rng.uniform(view.margin[0], view.margin[1], size=4)
    low = np.floor(low - margins[:2] * extent).clip(0, [width - 1, height - 1]).astype(int)
    high = np.ceil(high + margins[2:] * extent).clip(low + 1, [width, height]).astype(int)
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def warp_image(
    image: Image.Image, crop: Crop, input_size: tuple[int, int], *, draft: bool = False
) -> NDArray[np.uint8]:
    """Sample ``crop`` of ``image`` into an (H, W, 3) uint8 array.

    Bilinear without antialiasing (like ``predict``); PIL truncates, so values sit up
    to one level below a float resize. Outside the image the border is replicated. With
    ``draft``, an unloaded JPEG decodes at the smallest DCT scale that still has one
    source pixel per output pixel.
    """
    out_h, out_w = input_size
    width, height = image.size
    center, linear = crop.center, crop.linear
    if draft and image.format == "JPEG":  # a loaded image ignores draft()
        step = min(2 * crop.half[0] / out_w, 2 * crop.half[1] / out_h)  # source px per output px
        if step >= 2:
            result = image.draft("RGB", (math.ceil(width / step), math.ceil(height / step)))
            if result is not None:
                factor = result[1][2] / width  # 1 / DCT scale
                center, linear = center * factor, linear * factor
                width, height = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")
    # PIL maps output edge coordinates (i + 0.5) to source edge coordinates.
    scale = linear * np.array([2.0 / out_w, 2.0 / out_h])
    offset = center - linear.sum(axis=1)
    centers = np.array([(x, y) for x in (0.5, out_w - 0.5) for y in (0.5, out_h - 0.5)])
    reach = centers @ scale.T + offset
    if (reach < 0).any() or (reach >= [width, height]).any():  # PIL fills outside [0, size)
        low, high = np.floor(reach.min(axis=0)) - 1, np.ceil(reach.max(axis=0)) + 1
        inside_low = np.clip(low, 0, [width - 1, height - 1]).astype(int)
        inside_high = np.clip(high, inside_low + 1, [width, height]).astype(int)
        region = np.asarray(image.crop((*inside_low, *inside_high)))
        pad = (
            (int(inside_low[1] - low[1]), int(high[1] - inside_high[1])),
            (int(inside_low[0] - low[0]), int(high[0] - inside_high[0])),
            (0, 0),
        )
        image = Image.fromarray(np.pad(region, pad, mode="edge"))
        offset = offset - low
    data = (scale[0, 0], scale[0, 1], offset[0], scale[1, 0], scale[1, 1], offset[1])
    bilinear = Image.Resampling.BILINEAR
    output = image.transform((out_w, out_h), Image.Transform.AFFINE, data, bilinear)
    return np.array(output)  # writable, for torch.from_numpy


def _as_image(image: Image.Image | NDArray[np.uint8]) -> Image.Image:
    return image if isinstance(image, Image.Image) else Image.fromarray(image)


def _chw(pixels: NDArray[np.uint8]) -> torch.Tensor:
    return torch.from_numpy(pixels).permute(2, 0, 1)


def apply_geometry(
    sample: RawSample,
    input_size: tuple[int, int],
    augment: AugmentConfig,
    rng: np.random.Generator | None,
) -> tuple[torch.Tensor, NDArray[np.float32]]:
    """Crop (optionally rotated) around the page extent and resample to ``input_size``.

    Returns the (3, H, W) uint8 image and the grid in its flattener coordinates. Image
    and grid share one affine map, so the target stays exact up to bilinear sampling.
    """
    crop = plan_crop(sample, augment, rng)
    draft = rng is not None and augment.jpeg_draft
    pixels = warp_image(_as_image(sample.image), crop, input_size, draft=draft)
    return _chw(pixels), crop.target(_edge_points(sample))


def evaluation_view(
    sample: RawSample, input_size: tuple[int, int], view: ViewConfig, key: str
) -> tuple[torch.Tensor, NDArray[np.float32]]:
    """``view`` of ``sample`` as a (3, H, W) uint8 image and its target grid.

    ``key`` (source and sample id) seeds the view's margins, so every run and seed
    evaluates the same crops.
    """
    rng = np.random.default_rng([zlib.crc32(view.name.encode()), zlib.crc32(key.encode())])
    left, top, right, bottom = view_box(sample, view, rng)
    region = _as_image(sample.image).crop((left, top, right, bottom))
    half = np.array([right - left, bottom - top], dtype=np.float64) / 2.0
    pixels = warp_image(region, Crop(0.0, half, half), input_size)
    box = Crop(0.0, np.array([left, top], dtype=np.float64) + half, half)
    return _chw(pixels), box.target(_edge_points(sample))


def draw_color_params(
    augment: AugmentConfig, rng: np.random.Generator
) -> tuple[NDArray[np.float32], int]:
    """Photometric parameters (``COLOR_PARAMS`` layout) and a noise seed for one sample."""
    params, seed = IDENTITY_COLOR.copy(), 0
    if rng.random() < augment.color_prob:
        params[0] = 1
        params[1] = rng.uniform(1 - augment.brightness, 1 + augment.brightness)
        params[2] = rng.uniform(1 - augment.contrast, 1 + augment.contrast)
        params[3] = rng.uniform(1 - augment.saturation, 1 + augment.saturation)
        params[4:7] = rng.uniform(1 - augment.channel_gain, 1 + augment.channel_gain, size=3)
        params[7] = math.exp(rng.uniform(-augment.gamma, augment.gamma))
    if rng.random() < augment.grayscale_prob:
        params[8] = 1
    if rng.random() < augment.blur_prob:
        params[9], params[10] = 1, rng.uniform(0.3, 1.2)
    if rng.random() < augment.noise_prob and augment.noise_std > 0:
        seed = int(rng.integers(2**62))
        params[11], params[12] = 1, rng.uniform(0, augment.noise_std)
    return params, seed


def _grayscale(images: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=images.dtype, device=images.device)
    return (images * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def _blur(images: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-sample 5x5 Gaussian with reflect padding."""
    count, channels, height, width = images.shape
    offsets = torch.arange(-2, 3, dtype=images.dtype, device=images.device) ** 2
    taps = torch.exp(-offsets.view(1, 5) / (2 * sigma.view(-1, 1) ** 2))
    taps = taps / taps.sum(dim=1, keepdim=True)
    kernel = (taps[:, :, None] * taps[:, None, :]).repeat_interleave(channels, dim=0)
    padded = F.pad(images.reshape(1, count * channels, height, width), (2, 2, 2, 2), "reflect")
    blurred = F.conv2d(padded, kernel.unsqueeze(1), groups=count * channels)
    return blurred.view(count, channels, height, width)


def apply_color_batch(
    images: torch.Tensor, params: torch.Tensor, noise_seeds: torch.Tensor | list[int]
) -> torch.Tensor:
    """Photometric jitter of float (B, 3, H, W) images in [0, 1] on their device.

    ``params`` (B, ``COLOR_PARAMS``) and ``noise_seeds`` (B,) come from
    ``draw_color_params``; the noise of each sample draws from its own seeded generator.
    """
    p = params.to(device=images.device, dtype=images.dtype)

    def per_sample(column: int, on: torch.Tensor) -> torch.Tensor:
        return torch.where(on, p[:, column], torch.ones_like(p[:, column])).view(-1, 1, 1, 1)

    on = p[:, 0] > 0
    x = images * per_sample(1, on)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * per_sample(2, on) + mean
    gray = _grayscale(x)
    x = (x - gray) * per_sample(3, on) + gray
    gains = torch.where(on.view(-1, 1), p[:, 4:7], torch.ones_like(p[:, 4:7]))
    x = (x * gains.view(-1, 3, 1, 1)).clamp(0, 1).pow(per_sample(7, on))
    x = torch.where((p[:, 8] > 0).view(-1, 1, 1, 1), _grayscale(x).expand_as(x), x)
    blur = torch.nonzero(p[:, 9] > 0).flatten()
    if len(blur):
        x[blur] = _blur(x[blur], p[blur, 10])
    seeds = noise_seeds.tolist() if isinstance(noise_seeds, torch.Tensor) else list(noise_seeds)
    for index in torch.nonzero(p[:, 11] > 0).flatten().tolist():
        generator = torch.Generator(device=x.device).manual_seed(int(seeds[index]))
        noise = torch.randn(x.shape[1:], generator=generator, device=x.device, dtype=x.dtype)
        x[index] += noise * p[index, 12]
    return x.clamp_(0, 1)


def to_float_images(batch: dict[str, Any]) -> torch.Tensor:
    """Float [0, 1] images of a collated training batch, jittered when it carries ``color``."""
    images = batch["image"].float().div_(255)
    if "color" in batch:
        images = apply_color_batch(images, batch["color"], batch["noise_seed"])
    return images


class DewarpDataset(torch.utils.data.Dataset):
    """Concatenation of sources. Keys are ints or ``(flat_index, position)``;
    augmentation draws from ``default_rng([seed, position])``, so every worker
    assignment gives the same items. Top-level and picklable (Windows spawn)."""

    def __init__(
        self,
        sources: list[_Source],
        *,
        input_size: tuple[int, int],
        grid_size: tuple[int, int],
        augment: AugmentConfig,
        seed: int,
        train: bool,
        track: str,
        views: tuple[ViewConfig, ...] = (),
    ) -> None:
        self.sources = sources
        self.input_size, self.grid_size = tuple(input_size), tuple(grid_size)
        self.augment, self.seed, self.train, self.track = augment, int(seed), train, track
        if not views:
            middle = sum(augment.margin) / 2.0
            views = (ViewConfig("page", "page", (middle, middle)),)
        self.views = tuple(views)
        if len({view.name for view in self.views}) != len(self.views):
            raise _error("invalid_config", "view names must be unique")
        self.offsets = np.cumsum([0] + [len(source) for source in sources]).tolist()

    def __len__(self) -> int:
        return self.offsets[-1]

    @property
    def sizes(self) -> list[int]:
        return [len(source) for source in self.sources]

    @property
    def weights(self) -> list[float] | None:
        weights = [source.spec.weight for source in self.sources]
        if all(weight is None for weight in weights):
            return None
        if any(weight is None for weight in weights):
            raise _error("invalid_spec", "set weight on all sources of a split or on none")
        return [float(weight) / sum(weights) for weight in weights]

    def describe(self) -> list[dict[str, Any]]:
        return [source.describe() for source in self.sources]

    def sample_keys(self) -> set[tuple[str, str]]:
        """``(resolved root, sample_id)`` of every sample (overlap checks)."""
        keys: set[tuple[str, str]] = set()
        for source in self.sources:
            root = str(source.root.resolve())
            keys.update((root, record["sample_id"]) for record in source.records)
        return keys

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        flat, position = (key, key) if isinstance(key, (int, np.integer)) else key
        if not 0 <= flat < len(self):
            raise IndexError(flat)
        which = bisect.bisect_right(self.offsets, flat) - 1
        sample = self.sources[which].load(flat - self.offsets[which])
        try:
            if self.track == "release" and sample.meta["track"] != "release":
                raise _error("track_violation", f"release dataset loaded {sample.meta}")
            item = self._train_item(sample, position) if self.train else self._eval_item(sample)
        finally:
            if isinstance(sample.image, Image.Image):
                sample.image.close()
        grid3d = sample.grid3d
        has_3d = grid3d is not None and bool(np.isfinite(grid3d).all())
        if not has_3d:
            grid3d = np.zeros((*self.grid_size, 3), dtype=np.float32)
        item["grid3d"] = torch.from_numpy(np.ascontiguousarray(grid3d.transpose(2, 0, 1)))
        item["has_grid3d"] = torch.tensor(has_3d)
        return {**item, **sample.meta}

    def _target(self, sample: RawSample, grid: NDArray) -> tuple[torch.Tensor, torch.Tensor]:
        valid = sample.valid & np.isfinite(grid).all(axis=-1)
        grid = np.where(valid[..., None], grid, 0.0).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(grid.transpose(2, 0, 1))),
            torch.from_numpy(valid[None].astype(np.float32)),
        )

    def _train_item(self, sample: RawSample, position: int) -> dict[str, Any]:
        rng = None
        if self.augment.enabled:
            rng = np.random.default_rng([self.seed, int(position)])
        image, grid = apply_geometry(sample, self.input_size, self.augment, rng)
        color, noise_seed = IDENTITY_COLOR, 0
        if rng is not None:
            color, noise_seed = draw_color_params(self.augment, rng)
        grid2d, valid = self._target(sample, grid)
        return {
            "image": image,
            "grid2d": grid2d,
            "valid": valid,
            "color": torch.from_numpy(color.copy()),
            "noise_seed": torch.tensor(noise_seed, dtype=torch.int64),
        }

    def _eval_item(self, sample: RawSample) -> dict[str, Any]:
        key = f"{sample.meta.get('source', '')}/{sample.meta.get('sample_id', '')}"
        images, grids, valids = [], [], []
        for view in self.views:
            image, grid = evaluation_view(sample, self.input_size, view, key)
            grid2d, valid = self._target(sample, grid)
            images.append(image)
            grids.append(grid2d)
            valids.append(valid)
        return {
            "image": torch.stack(images),
            "grid2d": torch.stack(grids),
            "valid": torch.stack(valids),
        }


def build_dataset(
    spec: DataSpec,
    *,
    track: str,
    split: str,
    input_size: tuple[int, int],
    grid_size: tuple[int, int],
    augment: AugmentConfig | None = None,
    seed: int = 0,
    train: bool = True,
    views: tuple[ViewConfig, ...] = (),
) -> DewarpDataset | None:
    """Dataset over the spec's sources for ``split``; ``None`` if it has none.

    Raises ``track_violation`` before touching data when a release-track build
    would include a research-only source. ``views`` define evaluation items
    (``train=False``); by default one page view at the ``augment.margin`` midpoint.
    """
    require_track(spec, track)
    grid_size = tuple(grid_size)
    sources = [
        SOURCE_TYPES[source.kind](source, spec.root(source), grid_size)
        for source in spec.sources
        if source.split == split
    ]
    if not sources:
        return None
    return DewarpDataset(
        sources,
        input_size=input_size,
        grid_size=grid_size,
        augment=AugmentConfig() if augment is None else augment,
        seed=seed,
        train=train,
        track=track,
        views=tuple(views),
    )
