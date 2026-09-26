"""Training-only batch augmentation on the batch's device; cached data stays intact.

Images are float32 (N, 3, H, W) in [0, 1]. Coordinates are xy values normalized
to the original photo's physical bounds. The flat document domain and its valid
mask keep their shape. Random draws use the CPU global torch RNG the trainer saves.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..contracts import FlattenerError, Status

AUGMENTATION_VERSION = "protected-background-quarter-turn-fit-1.0"
AUGMENTATION_PROTOCOL = {
    "version": AUGMENTATION_VERSION,
    "scope": "training only; no cache, validation, test, or source artifact mutation",
    "rng": "CPU global torch RNG; per-image gates and procedural fields; saved torch_rng_state",
    "background": "asset-free multiscale color/noise/stripe fields outside dilated page coverage",
    "foreground": "3 photo-pixel dilation, preview max-pooling, 2 preview-pixel dilation",
    "rotation": "per-image probability of uniform nonzero CCW quarter turn; else exact upright",
    "fit": "uniform physical-photo pixel scale min(H/W,W/H) for odd turns; centered same canvas",
    "sampling": "align_corners=False; antialias prefilter then inverse-affine bilinear RGB",
    "padding": "procedural background with bilinear coverage compositing, never black fringe",
    "source_mask": "same physical affine at original photo resolution, nearest label sampling",
    "coordinates": "forward affine applied to xy values only; no clipping; flat domain unchanged",
    "valid": "original mask and denominator unchanged; invalid coordinates remain NaN pairs",
    "disabled": "both probabilities zero returns identical batch and consumes no RNG",
}

# Kept apart from AUGMENTATION_PROTOCOL, which DIVERSITY_CHECKPOINT_SCHEMA embeds.
SEMANTIC_PAGE_PROTOCOL = {
    "version": "exact-semantic-page-labels-1.0",
    "corners": "TL,TR,BR,BL semantic document order; never re-sort by screen position",
    "flat_boundary": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    "coordinate_units": "original-photo normalized physical xy bounds; align_corners=False",
    "truth": "metadata analytic flat_to_photo at exact boundaries; not dense-grid pixel centers",
    "capture_truth": "manual planar quad pixel centers: 2*(xy+0.5)/(original W,H)-1",
    "corner_visibility": "finite and inside [-1,1]; unavailable corners are NaN pairs; no clipping",
    "direction_axis": "physical-pixel mean semantic horizontal edge ((TR-TL)+(BR-BL))/2",
    "direction_classes": "nearest CLOCKWISE quarter: 0 right/upright, 1 down, 2 left, 3 up",
    "direction_eligibility": "finite raw corners, axis >1e-6 pixels, top-two score gap >1e-6 norm",
    "crop_direction": "raw metadata axis may remain known when individual corners leave the frame",
    "rotation": "rederive from transformed visible corners; conformal CCW k crop fallback q-k mod4",
    "legacy_rotation": "anisotropic normalized rotation rederives; partial-corner direction masked",
    "cache": "metadata-only table in record order; eight persistent tensor storages unchanged",
}
_SEMANTIC_FIELDS = ("corner_target", "corner_valid", "direction_target", "direction_valid")


def _direction_labels(corners, photo_size):
    """Return the nearest clockwise quarter of the semantic x-axis in photo pixels."""
    finite = torch.isfinite(corners).all(dim=(1, 2))
    safe = torch.where(torch.isfinite(corners), corners, 0.0)
    axis = ((safe[:, 1] - safe[:, 0]) + (safe[:, 2] - safe[:, 3])) * 0.5
    scale = corners.new_tensor((photo_size[1] / 2, photo_size[0] / 2))
    axis = axis * scale
    dx, dy = axis.unbind(-1)
    scores = torch.stack((dx, dy, -dx, -dy), dim=1)
    first_two = scores.topk(2, dim=1).values
    norm = torch.linalg.vector_norm(axis, dim=1)
    valid = finite & (norm > 1e-6) & ((first_two[:, 0] - first_two[:, 1]) > 1e-6 * norm)
    direction = scores.argmax(dim=1)
    return torch.where(valid, direction, 0), valid


def _semantic_contract(batch, count, device, checks):
    present = set(_SEMANTIC_FIELDS) & batch.keys()
    if not present:
        return False
    if present != set(_SEMANTIC_FIELDS):
        raise _error("semantic supervision requires all four corner/direction fields")
    spec = {"corner_target": (torch.float32, (count, 4, 2)),
            "corner_valid": (torch.bool, (count, 4)),
            "direction_target": (torch.int64, (count,)),
            "direction_valid": (torch.bool, (count,))}
    for key, (dtype, shape) in spec.items():
        value = batch[key]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.layout != torch.strided or value.device != device
                or tuple(value.shape) != shape):
            raise _error(f"invalid semantic supervision tensor {key}")
    target, valid = batch["corner_target"], batch["corner_valid"]
    safe = torch.where(valid[..., None], target, 0.0)
    admissible = (torch.isfinite(safe).all() & (safe.abs() <= 1).all()
                  & torch.where(valid[..., None], True, torch.isnan(target)).all()
                  & (batch["direction_target"] >= 0).all()
                  & (batch["direction_target"] < 4).all())
    _check(admissible, checks, "semantic corner/direction values violate their contract")
    return True


def transform_semantic_page_labels(batch, forward, photo_size, *, turns=None, _checks=None):
    """Return the transformed label fields for the caller to merge into its batch.

    ``forward`` is (N, 2, 2) in normalized original-photo coordinates. Pass
    ``turns`` (CPU int64, N) only for the aspect-preserving fit transform; it
    gives exact q-k labels for cropped corners. Without it, partially observed
    directions are masked.
    """
    if not isinstance(batch, dict):
        raise _error("semantic transform requires a batch dictionary")
    if not set(_SEMANTIC_FIELDS) & batch.keys():
        return {}
    corners = batch.get("corner_target")
    if not isinstance(corners, torch.Tensor) or corners.ndim != 3 or len(corners) < 1:
        raise _error("semantic corners must have shape N4x2")
    count, device = len(corners), corners.device
    _semantic_contract(batch, count, device, _checks)
    if (not isinstance(photo_size, tuple) or len(photo_size) != 2
            or any(type(side) is not int or not 1 <= side <= 16_384 for side in photo_size)
            or not isinstance(forward, torch.Tensor) or forward.dtype != torch.float32
            or forward.layout != torch.strided
            or forward.device != device or tuple(forward.shape) != (count, 2, 2)):
        raise _error("semantic transform requires bounded photo_size and float32 N2x2 matrix")
    _check(torch.isfinite(forward).all(), _checks, "semantic transform matrix must be finite")
    transformed = torch.einsum("nij,nkj->nki", forward, corners)
    valid = batch["corner_valid"]
    direction, direction_valid = _direction_labels(transformed, photo_size)
    complete = valid.all(dim=1)
    direction_valid = direction_valid & complete & batch["direction_valid"]
    if turns is not None:
        if (not isinstance(turns, torch.Tensor) or turns.dtype != torch.int64
                or turns.device.type != "cpu" or tuple(turns.shape) != (count,)
                or not bool(((turns >= 0) & (turns < 4)).all())):
            raise _error("semantic fit turns must be a CPU int64 N tensor in [0,3]")
        expected = _fit_matrices(turns, photo_size, device)[0]
        _check(torch.isfinite(forward).all() & (forward == expected).all(), _checks,
               "crop direction fallback requires the exact physical fit transform")
        adjusted = (batch["direction_target"] - turns.to(device, non_blocking=True)) % 4
        direction = torch.where(complete, direction, adjusted)
        direction_valid = torch.where(complete, direction_valid, batch["direction_valid"])
    direction = torch.where(direction_valid, direction, 0)
    safe = torch.where(valid[..., None], transformed, 0.0)
    _check(torch.isfinite(safe).all() & (safe.abs() <= 1).all(), _checks,
           "semantic transform moved an available corner outside physical image bounds")
    return {"corner_target": transformed, "corner_valid": valid,
            "direction_target": direction, "direction_valid": direction_valid}


class SemanticPageLabels:
    """CPU table of semantic page labels built from metadata only."""

    def __init__(self, dataset, tensors, metadata_paths, identity):
        self._dataset = dataset
        self._tensors = tensors
        self._metadata_paths = tuple(metadata_paths)
        self._identity = copy.deepcopy(identity)
        self.cache_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        self.split, self.photo_size = dataset.split, dataset.photo_size
        self.manifest_sha256 = dataset.manifest_sha256
        self.record_ids = tuple(record["id"] for record in dataset.records)

    @property
    def provenance(self):
        return copy.deepcopy(self._identity)

    def __len__(self):
        return len(self.record_ids)

    def verify_integrity(self):
        """Check manifest, source and metadata hashes; cache owners check pixel files."""
        from . import scene_data

        dataset = self._dataset
        if (scene_data._manifest_digest(dataset._directory) != self.manifest_sha256
                or scene_data._data_hashes(dataset.include_paper_augmentation)
                != self._identity["data_source_code_sha256"]
                or scene_data._sha256(Path(__file__)) != self._identity["label_source_sha256"]):
            raise _error("semantic label manifest or source code changed")
        for path, digest, is_paper in self._metadata_paths:
            if is_paper and (path.is_symlink() or path.parent.is_symlink()):
                raise _error("semantic paper metadata cannot be a symlink")
            if not path.resolve().is_relative_to(dataset._directory):
                raise _error("semantic metadata escaped the dataset")
            scene_data._checked_hash(path, digest, 1024 * 1024)

    def fetch_batch(self, indices, *, pin_memory=False):
        """Return fresh contiguous label tensors in the given sample order."""
        if type(pin_memory) is not bool:
            raise _error("semantic pin_memory must be boolean")
        if isinstance(indices, torch.Tensor):
            if (indices.device.type != "cpu" or indices.ndim != 1
                    or indices.dtype not in (torch.int32, torch.int64)):
                raise _error("semantic indices must be one-dimensional CPU integers")
        elif not isinstance(indices, (list, tuple, np.ndarray)):
            raise _error("semantic indices must be a bounded integer sequence")
        if (isinstance(indices, np.ndarray)
                and (indices.ndim != 1 or indices.dtype.kind not in "iu")):
            raise _error("semantic indices must be one-dimensional integers")
        if not 1 <= len(indices) <= len(self):
            raise _error("semantic index count must lie within the dataset size")
        selected = []
        for value in indices:
            if isinstance(value, (bool, np.bool_)):
                raise _error("semantic indices must not be boolean")
            try:
                index = operator.index(value)
            except TypeError as error:
                raise _error("semantic indices must be integers") from error
            if not -len(self) <= index < len(self):
                raise IndexError("semantic index out of range")
            selected.append(index % len(self))
        index = torch.tensor(selected, dtype=torch.int64, device="cpu")
        result = {}
        for name, value in self._tensors.items():
            output = torch.empty((len(selected), *value.shape[1:]), dtype=value.dtype,
                                  device="cpu", pin_memory=pin_memory)
            torch.index_select(value, 0, index, out=output)
            result[name] = output
        return result


def prepare_semantic_page_labels(dataset, *, max_bytes=16 * 1024**2):
    """Build corner and direction labels for the dataset's records from metadata.

    The table sits beside the cache and leaves its tensors and identity intact.
    Image pixels are never read.
    """
    from ..synthetic.geometry import flat_to_photo
    from . import scene_data

    if (type(dataset) not in (scene_data.SceneCoordinateDataset,
                             scene_data.CachedSceneCoordinateDataset)
            or type(max_bytes) is not int or max_bytes < 1):
        raise _error("semantic labels require a scene dataset and positive byte budget")
    if len(dataset) * 45 > max_bytes:
        raise _error("semantic label table exceeds its retained memory budget")
    identity = {"protocol": copy.deepcopy(SEMANTIC_PAGE_PROTOCOL),
                "manifest_sha256": dataset.manifest_sha256, "split": dataset.split,
                "data_source_code_sha256": dataset.source_code_sha256,
                "label_source_sha256": scene_data._sha256(Path(__file__))}
    records = dataset.records
    identity["records_sha256"] = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    paths = []
    for record in records:
        path = dataset._directory / record["paths"]["metadata.json"]
        paths.append((path, record["sha256"]["metadata.json"], record.get("source") == "paper"))
    tensors = {
        "corner_target": torch.empty((len(dataset), 4, 2), dtype=torch.float32, device="cpu"),
        "corner_valid": torch.empty((len(dataset), 4), dtype=torch.bool, device="cpu"),
        "direction_target": torch.empty(len(dataset), dtype=torch.int64, device="cpu"),
        "direction_valid": torch.empty(len(dataset), dtype=torch.bool, device="cpu"),
    }
    labels = SemanticPageLabels(dataset, tensors, paths, identity)
    labels.verify_integrity()
    flat = np.asarray(SEMANTIC_PAGE_PROTOCOL["flat_boundary"], dtype=np.float64)
    for index, (record, (path, _, _)) in enumerate(zip(records, paths, strict=True)):
        if record.get("source") == "paper":
            template = dataset._templates[record["template_id"]]
            capture = dataset._captures.get(record.get("capture_id"))
            _, params = scene_data._paper_metadata(
                record, {"metadata.json": path}, template, dataset.photo_size, capture)
            if capture is None:
                raw = scene_data._paper_backend().flat_to_photo(flat, params)
            else:
                raw = (2 * (np.asarray(capture["corners_xy"], dtype=np.float64) + 0.5)
                       / np.asarray(capture["original_size"][::-1]) - 1).astype(np.float32)
        else:
            metadata = scene_data._metadata(record, {"metadata.json": path})
            raw = flat_to_photo(flat, scene_data._geometry_contract(metadata, dataset.photo_size))
        raw = torch.from_numpy(np.asarray(raw, dtype=np.float32).copy())
        visible = torch.isfinite(raw).all(dim=1) & (raw.abs() <= 1).all(dim=1)
        tensors["corner_target"][index] = torch.where(visible[:, None], raw, torch.nan)
        tensors["corner_valid"][index] = visible
        direction, eligible = _direction_labels(raw[None].double(), dataset.photo_size)
        tensors["direction_target"][index] = direction[0]
        tensors["direction_valid"][index] = eligible[0]
    labels._identity["labels_sha256"] = hashlib.sha256(b"".join(
        tensors[name].numpy().tobytes() for name in _SEMANTIC_FIELDS)).hexdigest()
    labels.verify_integrity()
    return labels


def _error(message):
    return FlattenerError(Status.INPUT_ERROR, "invalid_training_augmentation", message)


def _probability(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise _error(f"{name} must be finite in [0,1]")


def _check(flag, checks, message):
    if checks is not None:
        checks.append(flag)
    elif not bool(flag):
        raise _error(message)


def _admit(batch, photo_size, required_mask, validated, checks):
    if (not isinstance(photo_size, tuple) or len(photo_size) != 2
            or any(type(side) is not int or side < 1 for side in photo_size)):
        raise _error("photo_size must be the original positive (height, width) tuple")
    if type(validated) is not bool or checks is not None and not isinstance(checks, list):
        raise _error("validated/checks controls have invalid types")
    if not isinstance(batch, dict) or not {"image", "target", "valid"} <= batch.keys():
        raise _error("batch requires image, target and valid")
    image, target, valid = (batch[key] for key in ("image", "target", "valid"))
    if (not isinstance(image, torch.Tensor) or image.dtype != torch.float32
            or image.layout != torch.strided or image.ndim != 4
            or image.shape[0] < 1 or image.shape[1] != 3 or min(image.shape[-2:]) < 1
            or image.device.type not in ("cpu", "cuda", "xpu")):
        raise _error("image must be a CPU/CUDA/XPU float32 N3HW tensor")
    if (not isinstance(target, torch.Tensor) or target.dtype != torch.float32
            or target.layout != torch.strided or target.device != image.device
            or target.ndim != 4 or target.shape[0] != image.shape[0]
            or target.shape[-1] != 2 or min(target.shape[1:3]) < 1
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.layout != torch.strided or valid.device != image.device
            or valid.shape != target.shape[:-1]):
        raise _error("target/valid must be matching float32 NHW2 and bool NHW tensors")
    mask = batch.get("source_mask")
    if ((required_mask or "source_mask" in batch) and (
            not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.layout != torch.strided or mask.device != image.device
            or mask.shape != (image.shape[0], *photo_size))):
        raise _error("background/source mask requires original-photo bool NHW on the image device")
    if not validated:
        admissible = (torch.isfinite(image).all() & (image >= 0).all() & (image <= 1).all()
                      & valid.flatten(1).any(1).all()
                      & torch.isfinite(torch.where(valid[..., None], target, 0.0)).all()
                      & (torch.where(valid[..., None], target, 0.0).abs() <= 1).all()
                      & torch.where(valid[..., None], True, torch.isnan(target)).all())
        if mask is not None:
            admissible = admissible & mask.flatten(1).any(1).all()
        _check(admissible, checks, "image/coordinate/page-mask values violate training contract")
    elif required_mask:
        _check(mask.flatten(1).any(1).all(), checks, "background source masks must be nonempty")
    _semantic_contract(batch, len(image), image.device, checks)


def _random(shape, device):
    # Draw on CPU to keep the checkpointed RNG stream and avoid device scalar reads.
    return torch.rand(shape, dtype=torch.float32, device="cpu").to(device, non_blocking=True)


def _texture(image):
    count, _, height, width = image.shape
    device = image.device
    # Move all random fields in one small transfer; the hot path reads nothing back.
    fields = _random((count, 3 + 12 + 64 + 1024 + 4), device)
    color = 0.10 + 0.78 * fields[:, :3, None, None]
    smooth = F.interpolate(fields[:, 3:15].reshape(count, 3, 2, 2) - 0.5,
                           size=(height, width), mode="bilinear", align_corners=False)
    medium = F.interpolate(fields[:, 15:79].reshape(count, 1, 8, 8) - 0.5,
                           size=(height, width), mode="bilinear", align_corners=False)
    grain = F.interpolate(fields[:, 79:1103].reshape(count, 1, 32, 32) - 0.5,
                          size=(height, width), mode="bilinear", align_corners=False)
    parameters = fields[:, 1103:, None, None]
    y = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height
    x = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    angle = parameters[:, 0:1] * (2 * math.pi)
    position = x[None, None, None, :] * angle.cos() + y[None, None, :, None] * angle.sin()
    stripe = ((2 * math.pi) * (position * (2 + 22 * parameters[:, 1:2])
                              + parameters[:, 2:3])).sin()
    return (color + 0.24 * smooth + 0.16 * medium + 0.06 * grain
            + (0.01 + 0.05 * parameters[:, 3:4]) * stripe).clamp_(0.0, 1.0)


def _protected_preview(source_mask, image_size):
    mask = F.max_pool2d(source_mask[:, None].float(), 7, stride=1, padding=3)
    # Max-pooling keeps any covered source pixel; the extra preview dilation
    # covers the antialiased resize footprint plus a boundary margin.
    mask = F.adaptive_max_pool2d(mask, image_size)
    return F.max_pool2d(mask, 5, stride=1, padding=2).bool()


def _sample_turns(count, probability):
    selected = torch.rand((count,), device="cpu") < probability
    choices = torch.randint(1, 4, (count,), device="cpu")
    return torch.where(selected, choices, 0)


def _fit_matrices(turns, photo_size, device):
    height, width = photo_size
    scale = min(height / width, width / height)
    b, c = scale * height / width, scale * width / height
    forward = torch.tensor([
        [[1, 0], [0, 1]], [[0, b], [-c, 0]],
        [[-1, 0], [0, -1]], [[0, -b], [c, 0]],
    ], dtype=torch.float32, device="cpu")[turns].to(device, non_blocking=True)
    inverse = torch.tensor([
        [[1, 0, 0], [0, 1, 0]], [[0, -1 / c, 0], [1 / b, 0, 0]],
        [[-1, 0, 0], [0, -1, 0]], [[0, 1 / c, 0], [-1 / b, 0, 0]],
    ], dtype=torch.float32, device="cpu")[turns].to(device, non_blocking=True)
    return forward, inverse, b, c


def _rotate_fit(batch, turns, photo_size, fill, checks):
    image, target = batch["image"], batch["target"]
    count, _, height, width = image.shape
    forward, inverse, b, c = _fit_matrices(turns, photo_size, image.device)
    # An odd physical rotation can shrink different preview axes when the
    # preview aspect differs from the photo. Prefilter each source axis by its
    # physical pixel scale; resizing never changes its normalized extent.
    filtered_size = (max(1, math.floor(height * min(1.0, b * width / height))),
                     max(1, math.floor(width * min(1.0, c * height / width))))
    filtered = F.interpolate(image, size=filtered_size, mode="bilinear",
                             align_corners=False, antialias=True)
    grid = F.affine_grid(inverse, image.shape, align_corners=False)
    rotated = F.grid_sample(filtered, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)
    coverage = F.grid_sample(torch.ones_like(filtered[:, :1]), grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
    rotated = rotated + fill * (1 - coverage)
    device_turns = turns.to(image.device, non_blocking=True)
    upright = device_turns[:, None, None, None] == 0
    half = device_turns[:, None, None, None] == 2
    result = dict(batch)
    result["image"] = torch.where(upright, image, torch.where(
        half, image.flip((-2, -1)), rotated.clamp(0.0, 1.0)))
    # The flat output lattice keeps the upright reading order; only the
    # source-photo xy values rotate.
    transformed = torch.einsum("nij,nhwj->nhwi", forward, target)
    result["target"] = torch.where(upright, target, transformed)
    result.update(transform_semantic_page_labels(
        batch, forward, photo_size, turns=turns, _checks=checks))
    if "source_mask" in batch:
        mask = batch["source_mask"]
        mask_grid = F.affine_grid(inverse, (count, 1, *photo_size), align_corners=False)
        rotated_mask = F.grid_sample(mask[:, None].float(), mask_grid, mode="nearest",
                                     padding_mode="zeros", align_corners=False)[:, 0].bool()
        result["source_mask"] = torch.where(upright[:, 0], mask, torch.where(
            half[:, 0], mask.flip((-2, -1)), rotated_mask))
    tolerance = 8 * torch.finfo(torch.float32).eps
    finite_target = torch.where(batch["valid"][..., None], result["target"], 0.0)
    admissible = (torch.isfinite(finite_target).all()
                  & (finite_target.abs() <= 1 + tolerance).all()
                  & torch.isfinite(result["image"]).all()
                  & (rotated >= -tolerance).all() & (rotated <= 1 + tolerance).all())
    _check(admissible, checks, "fit rotation violates the source coordinate/RGB bounds")
    return result


@torch.no_grad()
def augment_training_batch(batch, *, photo_size, rotation_probability=0.0,
                           background_probability=0.0, _validated=False, _checks=None):
    """Return an augmented copy of a training batch; call it only during training.

    Background replacement keeps the protected page pixels. Quarter turns fit the
    whole photo into the same canvas at one scale and keep the valid mask. With
    both probabilities zero, the input object is returned and no RNG is consumed.
    ``_validated`` with ``_checks`` leaves data checks as device tensors for the trainer.
    """
    _probability(rotation_probability, "rotation_probability")
    _probability(background_probability, "background_probability")
    if rotation_probability == 0 and background_probability == 0:
        return batch
    _admit(batch, photo_size, background_probability > 0, _validated, _checks)
    image = batch["image"]
    turns = (_sample_turns(len(image), rotation_probability)
             if rotation_probability else None)
    selected = ((torch.rand((len(image),), device="cpu") < background_probability).to(
                    image.device, non_blocking=True)
                if background_probability else None)
    texture = _texture(image)
    result = dict(batch)
    if background_probability:
        protected = _protected_preview(batch["source_mask"], image.shape[-2:])
        replace = selected[:, None, None, None] & ~protected
        result["image"] = torch.where(replace, texture, image)
    if rotation_probability:
        result = _rotate_fit(result, turns, photo_size, texture, _checks)
    return result
