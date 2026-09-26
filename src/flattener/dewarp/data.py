"""Training data for dewarp models: licensed sources, adapters and augmentation.

A data spec (JSON, schema ``flattener-dewarp-data-1``) lists sources and the
license *track* of each: ``release`` (license-clean; weights may ship) or
``research`` (weights never released). Doc3D, the UVDoc dataset and the
unclassified raw SyntheticDoc layout are research-only kinds; a release-track
dataset refuses them, and refuses converted samples not labelled release.

Every sample is returned at the model's input size with its target 2-D grid
in the flattener convention (edge-aligned nodes, align_corners=False values
relative to the returned image; see ``coords``), a node validity mask, an
optional UVDoc-normalized 3-D grid, and string metadata including the track.

Converted layout (``kind: "converted"``, produced by the data pipeline)::

    <root>/manifest.jsonl          one JSON object per sample (preferred), or
    <root>/**/<sample>.json        one object per file (scanned when no manifest)

    {"schema": "flattener-dewarp-sample-1", "sample_id": "...", "split": "train",
     "image": "relative/path.jpg", "backward_map": "relative/path.npy",
     "backward_map_convention": "flattener" | "syntheticdoc",
     "original_render_size": [1440, 1024], "track": "release",
     "license_class": "clean", ...}

Paths are relative to the JSON document that lists them. ``backward_map`` is
a center-aligned (H, W, 2) float16/float32 map (SyntheticDoc: 177x121) holding
flattener values, or raw SyntheticDoc values when the convention is
``"syntheticdoc"`` (then ``original_render_size`` (H, W) is required).
Entries without ``split`` belong to the source's split; the trainer refuses
train and val sets that share a sample (same root and ``sample_id``).
Release-track sources accept only samples with ``track: "release"`` and a
``license_class`` whose ``license_key`` is on the ``RELEASE_LICENSE_KEYS``
allowlist ("clean", "CC BY 4.0", "cc0-1.0", ...); any NC/ND/SA variant,
"unclear" or an unknown class is refused.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
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
TRACKS = ("release", "research")
KINDS = ("converted", "syntheticdoc_raw", "doc3d", "uvdoc")
RESEARCH_ONLY_KINDS = frozenset({"syntheticdoc_raw", "doc3d", "uvdoc"})
# Release-track samples need a license class whose normalized key (see
# ``license_key``) is on this allowlist; everything else (any NC/ND/SA
# variant, unclear, unknown, research-only, ...) is refused.
RELEASE_LICENSE_KEYS = frozenset(
    {
        "clean",  # the data pipeline's classification of permissive sub-sources
        "own",
        "cc0",
        "by",  # CC BY any version: "CC BY 4.0", "cc-by-4.0", "BY"
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


def _error(reason: str, message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason, message)


def license_key(license_class: str) -> str:
    """Normalized license class: lowercase alphanumeric tokens without version
    numbers or the ``cc``/``license`` words ("CC BY-NC-SA 4.0" -> "by-nc-sa")."""
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
    """Sources by track. ``base_dir`` resolves relative roots and is not hashed."""

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
        return Path(self.base_dir, source.root)


def require_track(spec: DataSpec, track: str) -> None:
    """Refuse a release-track build from anything but a release-only spec."""
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
    """Seeded appearance and crop augmentation (UVDoc-style), applied in training only.

    ``margin`` is the per-side crop margin around the page (grid extent) as a
    fraction of the page extent; evaluation uses the range midpoint. Axis-aligned
    crops are clipped to the image (``clip_to_image``, as upstream); rotated
    crops may leave it and replicate the border (upstream ``BORDER_REPLICATE``).
    ``full_frame_prob`` keeps the whole image (uncropped-photo inputs).
    ``rotation_kinds`` limits rotation to samples of those source kinds (empty:
    all; upstream UVDoc rotates only its own dataset, not Doc3D).
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
            if item.name in ("enabled", "clip_to_image"):
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


@dataclass(slots=True)
class RawSample:
    image: NDArray[np.uint8]
    grid: NDArray[np.float32]  # (Gh, Gw, 2) edge-aligned, flattener values w.r.t. image
    valid: NDArray[np.bool_]
    grid3d: NDArray[np.float32] | None
    meta: dict[str, str] = field(default_factory=dict)


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))  # writable: torch.from_numpy needs it


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


class _Source:
    """One spec source: indexes records on construction; ``load`` reads one sample."""

    def __init__(self, spec: SourceSpec, root: Path, grid_size: tuple[int, int]) -> None:
        if not root.is_dir():
            raise _error("missing_data", f"source {spec.name!r}: root {root} is not a directory")
        self.spec, self.root, self.grid_size = spec, root, grid_size
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

    def __len__(self) -> int:
        return len(self.records)

    def _index(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def load(self, index: int) -> RawSample:
        raise NotImplementedError

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
        """Name, track, license manifest (sample counts per license class) and a
        digest of the sample list (ids, tracks, license classes; not paths)."""
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
        }


class ConvertedSource(_Source):
    def _index(self) -> list[dict[str, Any]]:
        manifest = self.root / "manifest.jsonl"
        if manifest.is_file():
            lines = manifest.read_text(encoding="utf-8").splitlines()
            documents = [(json.loads(line), self.root) for line in lines if line.strip()]
        else:
            documents = []
            for path in sorted(self.root.rglob("*.json")):
                entry = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(entry, dict) and entry.get("schema") == SAMPLE_SCHEMA:
                    documents.append((entry, path.parent))
        records = []
        required = {"schema", "sample_id", "image", "backward_map", "track"}
        for entry, base in documents:
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
            records.append(  # compact: records are pickled into every loader worker
                {
                    "sample_id": str(entry["sample_id"]),
                    "image": str(Path(base, entry["image"])),
                    "backward_map": str(Path(base, entry["backward_map"])),
                    "track": track,
                    "license_class": license_class,
                    "render_size": tuple(render_size) if convention == "syntheticdoc" else None,
                }
            )
        return records

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        image = _read_rgb(Path(record["image"]))
        values = np.load(record["backward_map"], allow_pickle=False).astype(np.float32)
        if record["render_size"] is not None:
            values = syntheticdoc_to_flattener(values, record["render_size"])
        grid = map_to_grid(values, self.grid_size)
        meta = self._meta(record, record["track"], record["license_class"] or self.spec.license)
        return RawSample(image, grid, np.isfinite(grid).all(axis=-1), None, meta)


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

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        image = _read_rgb(Path(record["image"]))
        raw = np.load(record["backward_map"], allow_pickle=False)
        grid = map_to_grid(syntheticdoc_to_flattener(raw, image.shape[:2]), self.grid_size)
        meta = self._meta(record, "research", self.LICENSE)
        return RawSample(image, grid, np.isfinite(grid).all(axis=-1), None, meta)


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

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        sample_id = record["sample_id"]
        image = _read_rgb(self.root / "img" / f"{sample_id}.png")
        pixels = _read_matrix(self.root / "grid2D" / sample_id, "grid2D")
        grid, valid = self._grid(pixels_to_flattener(pixels, image.shape[:2]))
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

    def load(self, index: int) -> RawSample:
        record = self.records[index]
        sample_id = record["sample_id"]
        meta_path = self.root / "metadata_sample" / f"{sample_id}.json"
        geometry = json.loads(meta_path.read_text(encoding="utf-8"))["geom_name"]
        image = _read_rgb(self.root / "img" / f"{sample_id}.png")
        pixels = _read_matrix(self.root / "grid2d" / geometry, "grid2d")
        grid, valid = self._grid(pixels_to_flattener(pixels, image.shape[:2]))
        grid3d = _normalize_3d(
            _read_matrix(self.root / "grid3d" / geometry, "grid3d"), UVDOC_GRID3D_RANGE, "xyz"
        )
        meta = self._meta(record, "research", self.spec.license or "UVDoc dataset (NC textures)")
        return RawSample(image, grid, valid, self._grid(grid3d)[0], meta)


SOURCE_TYPES: dict[str, type[_Source]] = {
    "converted": ConvertedSource,
    "syntheticdoc_raw": SyntheticDocRawSource,
    "doc3d": Doc3DSource,
    "uvdoc": UVDocSource,
}


def _rotation(theta: float) -> NDArray[np.float64]:
    cos, sin = math.cos(theta), math.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])


def apply_geometry(
    sample: RawSample,
    input_size: tuple[int, int],
    augment: AugmentConfig,
    rng: np.random.Generator | None,
) -> tuple[torch.Tensor, NDArray[np.float32]]:
    """Crop (optionally rotated) around the page extent and resample to ``input_size``.

    Returns the (3, H, W) float image and the grid re-expressed in its
    flattener coordinates. Image and grid share one affine map (output
    normalized o -> source edge pixel ``center + R (half * o)``), so the target
    stays exact up to bilinear image sampling; outside the image the border
    is replicated.
    """
    height, width = sample.image.shape[:2]
    size = np.array([width, height], dtype=np.float64)
    points = (sample.grid.astype(np.float64) + 1.0) / 2.0 * size  # edge-pixel coordinates
    center = size / 2.0
    theta, full_frame = 0.0, False
    if rng is not None:
        full_frame = bool(rng.random() < augment.full_frame_prob)
        rotates = not augment.rotation_kinds or sample.meta.get("kind") in augment.rotation_kinds
        if augment.rotation_deg > 0 and rotates and rng.random() < augment.rotation_prob:
            theta = math.radians(rng.uniform(-augment.rotation_deg, augment.rotation_deg))
    if full_frame or not sample.valid.any():
        theta, box_center, half = 0.0, center, size / 2.0
    else:
        rotation = _rotation(theta)
        rotated = (points[sample.valid] - center) @ rotation  # page in the crop frame
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
        box_center = center + rotation @ ((low + high) / 2.0)
    linear = _rotation(theta) * half  # R @ diag(half)
    # Decode only the source region the output corners can reach (the map is affine).
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]) @ linear.T + box_center
    low = np.clip(np.floor(corners.min(axis=0) - 0.5), 0, size - 1).astype(int)
    high = np.maximum(np.clip(np.ceil(corners.max(axis=0) + 0.5), 0, size - 1).astype(int), low)
    crop = np.ascontiguousarray(sample.image[low[1] : high[1] + 1, low[0] : high[0] + 1])
    tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    extent = (high + 1 - low).astype(np.float64)
    affine = np.concatenate(
        [2.0 * linear / extent[:, None], (2.0 * (box_center - low) / extent - 1.0)[:, None]], 1
    )
    out_h, out_w = input_size
    sampling = F.affine_grid(
        torch.from_numpy(affine).float().unsqueeze(0), [1, 3, out_h, out_w], align_corners=False
    )
    image = F.grid_sample(
        tensor, sampling, mode="bilinear", padding_mode="border", align_corners=False
    )[0]
    grid = ((points - box_center) @ _rotation(theta)) / half
    return image, grid.astype(np.float32)


def _grayscale(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype).view(3, 1, 1)
    return (image * weights).sum(dim=0, keepdim=True)


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """5x5 Gaussian, reflect padding; channels as batch (grouped conv is slow on CPU)."""
    taps = torch.exp(-(torch.arange(-2, 3, dtype=image.dtype) ** 2) / (2 * sigma * sigma))
    kernel = torch.outer(taps, taps).div_(taps.sum() ** 2).view(1, 1, 5, 5)
    padded = F.pad(image.unsqueeze(1), (2, 2, 2, 2), mode="reflect")
    return F.conv2d(padded, kernel)[:, 0]


def apply_color(
    image: torch.Tensor, augment: AugmentConfig, rng: np.random.Generator
) -> torch.Tensor:
    """Seeded photometric jitter on a (3, H, W) float image in [0, 1] (modified in place)."""
    if rng.random() < augment.color_prob:
        image.mul_(rng.uniform(1 - augment.brightness, 1 + augment.brightness))
        mean = image.mean()
        image.sub_(mean).mul_(rng.uniform(1 - augment.contrast, 1 + augment.contrast)).add_(mean)
        gray = _grayscale(image)
        image.sub_(gray).mul_(rng.uniform(1 - augment.saturation, 1 + augment.saturation))
        image.add_(gray)
        gains = rng.uniform(1 - augment.channel_gain, 1 + augment.channel_gain, size=3)
        image.mul_(torch.tensor(gains, dtype=image.dtype).view(3, 1, 1)).clamp_(0, 1)
        image.pow_(math.exp(rng.uniform(-augment.gamma, augment.gamma)))
    if rng.random() < augment.grayscale_prob:
        image.copy_(_grayscale(image).expand(3, -1, -1))
    if rng.random() < augment.blur_prob:
        image = _blur(image, rng.uniform(0.3, 1.2))
    if rng.random() < augment.noise_prob and augment.noise_std > 0:
        generator = torch.Generator().manual_seed(int(rng.integers(2**62)))
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        image.add_(noise, alpha=rng.uniform(0, augment.noise_std))
    return image.clamp_(0, 1)


class DewarpDataset(torch.utils.data.Dataset):
    """Concatenation of sources. Keys are ints or ``(flat_index, position)``;
    augmentation draws from ``default_rng([seed, position])`` so results do not
    depend on worker assignment. Top-level and picklable (Windows spawn)."""

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
    ) -> None:
        self.sources = sources
        self.input_size, self.grid_size = tuple(input_size), tuple(grid_size)
        self.augment, self.seed, self.train, self.track = augment, int(seed), train, track
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
        if self.track == "release" and sample.meta["track"] != "release":
            raise _error("track_violation", f"release dataset loaded {sample.meta}")
        augment = self.augment.enabled and self.train
        rng = np.random.default_rng([self.seed, int(position)]) if augment else None
        image, grid = apply_geometry(sample, self.input_size, self.augment, rng)
        if rng is not None:
            image = apply_color(image, self.augment, rng)
        valid = sample.valid & np.isfinite(grid).all(axis=-1)
        grid = np.where(valid[..., None], grid, 0.0).astype(np.float32)
        grid3d = sample.grid3d
        has_3d = grid3d is not None and bool(np.isfinite(grid3d).all())
        if not has_3d:
            grid3d = np.zeros((*self.grid_size, 3), dtype=np.float32)
        return {
            "image": image,
            "grid2d": torch.from_numpy(np.ascontiguousarray(grid.transpose(2, 0, 1))),
            "grid3d": torch.from_numpy(np.ascontiguousarray(grid3d.transpose(2, 0, 1))),
            "has_grid3d": torch.tensor(has_3d),
            "valid": torch.from_numpy(valid[None].astype(np.float32)),
            **sample.meta,
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
) -> DewarpDataset | None:
    """Dataset over the spec's sources for ``split``; ``None`` if it has none.

    Raises ``track_violation`` before touching data when a release-track build
    would include a research-only source.
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
    )
