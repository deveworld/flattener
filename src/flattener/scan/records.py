"""Scan records: reason codes with guidance, outcomes, ``ScanPlan`` and ``ScanResult``.

``contracts.Status`` is fixed, so scan conditions are ``reason_code`` strings on the
existing statuses. ``REASONS`` registers the scanner's reason codes, each with the
status it carries, its severity, how the scanner handles it and, where the user can
act, guidance in Korean and English:

* ``info``: recorded; the scan goes on unchanged.
* ``warning``: the scan goes on and the result is ``rectified_with_warnings``.
* ``failure``: the scan falls back to a perspective result or returns the original.

A code outside the registry keeps the status it came with; with the
``invalid_transform`` status (the renderer's geometry checks) it takes the guidance of
``invalid_transform``.

``Outcome`` names the image a result holds: ``rectified``, ``rectified_with_warnings``,
``perspective`` (page outline flattened, curl kept), ``original_returned``, or
``failed`` when there is no image (the input could not be read). The CLI exit statuses
are ``EXIT_SCANNED``, ``EXIT_FALLBACK`` and ``EXIT_USAGE`` (see ``flattener.scan``'s
``__main__``).

``ScanPlan`` holds what a render needs: the source record, the geometry (a grid of
flattener values in source coordinates, or a homography), the output size and the
settings. ``plan_sha256`` is the sha256 of its canonical JSON (sorted keys, no spaces,
floats as Python ``repr``), so a preview and a saved image with the same plan_sha256
share their geometry. ``ScanResult`` is the record of one scan. Its JSON holds everything
except the pixel arrays, which ``scan.io`` writes as image files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import FlattenerError, Status
from .output_policy import OutputPlan, focal_px_from_35mm

PLAN_SCHEMA = "flattener-scan-plan-1"
RECORD_SCHEMA = "flattener-scan-record-1"
PLAN_KINDS = ("dewarp", "perspective")
PLAN_KEYS = frozenset(
    {
        "schema",
        "kind",
        "source",
        "output",
        "grid",
        "homography",
        "region",
        "angle_deg",
        "quarter",
        "orientation_probs",
        "views",
        "inset",
        "border",
        "quantize",
        "nodes",
        "fill_rgb",
        "models",
        "thresholds",
        "provenance",
        "plan_sha256",
    }
)
RENDER_KINDS = ("dewarp", "perspective", "original")
QUANTIZE_MODES = ("round", "truncate")  # as flattener.dewarp.predict.QUANTIZE_MODES
NODE_CONVENTIONS = ("flattener", "uvdoc")  # as flattener.scan.placement.CONVENTIONS
PROVENANCE_VALUES = ("auto", "manual")
EXIT_SCANNED = 0
EXIT_FALLBACK = 1  # some page fell back (perspective or original) or failed
EXIT_USAGE = 2


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    FAILURE = "failure"


class Action(StrEnum):
    """What the user can do about a reason."""

    RESHOOT = "reshoot"
    SET_REGION = "set_region"
    CHOOSE_REGION = "choose_region"
    SET_ORIENTATION = "set_orientation"


class Outcome(StrEnum):
    RECTIFIED = "rectified"
    RECTIFIED_WITH_WARNINGS = "rectified_with_warnings"
    PERSPECTIVE = "perspective"
    ORIGINAL_RETURNED = "original_returned"
    FAILED = "failed"


SCANNED_OUTCOMES = frozenset({Outcome.RECTIFIED, Outcome.RECTIFIED_WITH_WARNINGS})


@dataclass(frozen=True, slots=True)
class ReasonSpec:
    code: str
    status: Status
    severity: Severity
    handling: str
    actions: tuple[Action, ...] = ()
    message_ko: str | None = None
    message_en: str | None = None


_RESHOOT_OR_REGION = (Action.RESHOOT, Action.SET_REGION)
_GEOMETRY_KO = "다시 찍거나 영역을 지정해 주세요"
_GEOMETRY_EN = "Take the photo again or set the page region"
_CHOOSE_KO = "페이지를 골라 주세요"
_CHOOSE_EN = "Choose the page"
_CUT_OFF_KO = "{sides}이 잘렸어요. 조금 물러나 찍어 주세요"
_CUT_OFF_EN = "The {sides} cut off. Step back a little and take the photo again"
_SIDES_KO = {"top": "위쪽", "bottom": "아래쪽", "left": "왼쪽", "right": "오른쪽"}

REASONS: dict[str, ReasonSpec] = {
    spec.code: spec
    for spec in (
        ReasonSpec(
            "page_not_found",
            Status.MODEL_ERROR,
            Severity.FAILURE,
            "Return the original; the user sets the page region. Also a grid the gate mask "
            "reaches past that does not look like paper (detail: check 'paper').",
            _RESHOOT_OR_REGION,
            "페이지 전체가 화면에 들어오게 찍어 주세요",
            "Take the photo with the whole page in the frame",
        ),
        ReasonSpec(
            "page_too_small",
            Status.MODEL_ERROR,
            Severity.FAILURE,
            "Return the original (page below 1 % of the photo or long side below 600 px).",
            (Action.RESHOOT,),
            "더 가까이에서 찍어 주세요",
            "Move closer and take the photo again",
        ),
        ReasonSpec(
            "page_fills_frame",
            Status.SUCCESS,
            Severity.INFO,
            "Go on with the whole photo as the page box.",
        ),
        ReasonSpec(
            "ambiguous_page_region",
            Status.MODEL_ERROR,
            Severity.FAILURE,
            "The gate split the page into pieces (dominance below 0.85): go on with the "
            "pass-0 outline and warn (success, warning) when it holds a piece; otherwise "
            "return the original with the candidate page boxes.",
            (Action.CHOOSE_REGION,),
            _CHOOSE_KO,
            _CHOOSE_EN,
        ),
        ReasonSpec(
            "two_page_candidate",
            Status.MODEL_ERROR,
            Severity.FAILURE,
            "As ambiguous_page_region, for two large components side by side.",
            (Action.CHOOSE_REGION,),
            _CHOOSE_KO,
            _CHOOSE_EN,
        ),
        ReasonSpec(
            "page_region_larger",
            Status.SUCCESS,
            Severity.WARNING,
            "The gate mask reaches past the flattened page (a neighbouring page, the book "
            "edge or paper under the page); the dewarp result is kept. Detail: coverage and "
            "the mask past each page side.",
            (Action.SET_REGION,),
            "옆 쪽이나 밑에 깔린 종이가 같이 보여요. 원하는 쪽이 펴졌는지 확인해 주세요",
            "Something next to the page shows too. Check that the right page was flattened",
        ),
        ReasonSpec(
            "neighbor_page_partial",
            Status.SUCCESS,
            Severity.INFO,
            "Leave out the small components that touch the border and go on.",
        ),
        ReasonSpec(
            "page_cut_off",
            Status.SUCCESS,
            Severity.WARNING,
            "Warn and go on when the gate mask runs along image sides (detail 'sides'). A "
            "flattened page that shows only a zoomed piece of text cut by its edges (check "
            "'fragment') fails with model_error: the chain stops and the original comes back "
            "with the reshoot guidance.",
            (Action.RESHOOT,),
            _CUT_OFF_KO,
            _CUT_OFF_EN,
        ),
        ReasonSpec(
            "possible_occlusion",
            Status.SUCCESS,
            Severity.WARNING,
            "Warn and go on (page solidity below 0.90).",
            (Action.RESHOOT,),
            "손가락이나 물건이 페이지를 가려요",
            "A finger or an object covers the page",
        ),
        ReasonSpec(
            "orientation_uncertain",
            Status.SUCCESS,
            Severity.WARNING,
            "Go on with the geometric prior (EXIF, portrait page) and offer quick rotation.",
            (Action.SET_ORIENTATION,),
            "방향을 확인해 주세요",
            "Check the page orientation",
        ),
        ReasonSpec(
            "unstable_geometry",
            Status.INVALID_TRANSFORM,
            Severity.FAILURE,
            "Try the candidate chain, then the perspective fallback or the original.",
            _RESHOOT_OR_REGION,
            _GEOMETRY_KO,
            _GEOMETRY_EN,
        ),
        ReasonSpec(
            "invalid_transform",
            Status.INVALID_TRANSFORM,
            Severity.FAILURE,
            "Try the candidate chain, then the perspective fallback or the original.",
            _RESHOOT_OR_REGION,
            _GEOMETRY_KO,
            _GEOMETRY_EN,
        ),
        ReasonSpec(
            "candidate_fallback",
            Status.SUCCESS,
            Severity.WARNING,
            "The first candidate failed and a later one passed; detail names the candidate "
            "and the first failure.",
            (Action.SET_REGION,),
            "결과를 확인하고, 필요하면 영역을 지정해 주세요",
            "Check the result and set the page region if needed",
        ),
        ReasonSpec(
            "rerun_discarded",
            Status.SUCCESS,
            Severity.INFO,
            "The crop passes failed their checks; the first pass of the candidate is used.",
        ),
        ReasonSpec(
            "curvature_not_corrected",
            Status.INVALID_TRANSFORM,
            Severity.WARNING,
            "Perspective fallback: the page outline is flattened and its curl stays.",
        ),
        ReasonSpec(
            "unverified_region",
            Status.SUCCESS,
            Severity.WARNING,
            "Use the predicted outline without a page gate check.",
        ),
        ReasonSpec(
            "source_downscaled",
            Status.SUCCESS,
            Severity.WARNING,
            "The JPEG was decoded at a reduced scale to fit the input pixel limit.",
        ),
        ReasonSpec(
            "low_resolution",
            Status.SUCCESS,
            Severity.WARNING,
            "Part of the page has fewer source pixels than output pixels.",
        ),
    )
}


def reason_spec(code: str, status: Status | str | None = None) -> ReasonSpec | None:
    """The registry entry of ``code``; for an unregistered code with the
    ``invalid_transform`` status, the entry of ``invalid_transform``."""
    if code in REASONS:
        return REASONS[code]
    if status is not None and Status(status) == Status.INVALID_TRANSFORM:
        return REASONS["invalid_transform"]
    return None


def json_plain(value: Any) -> Any:
    """``value`` as plain JSON data: NumPy scalars and arrays, tuples, enums and paths
    become JSON types, and non-finite floats become ``None``."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_plain(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"{type(value).__name__} is not JSON data")


def canonical_json(value: Any) -> bytes:
    """Sorted keys, no spaces, UTF-8: the bytes behind ``plan_sha256``."""
    text = json.dumps(
        json_plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def _plan_error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_plan", message)


def _record_error(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_scan_record", message)


@dataclass(frozen=True, slots=True)
class Reason:
    """One recorded condition; ``detail`` holds its parameters (e.g. ``sides``)."""

    code: str
    severity: Severity
    status: Status
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("reason code must be a non-empty string")
        object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "status", Status(self.status))
        object.__setattr__(self, "detail", json_plain(dict(self.detail)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "status": self.status.value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reason:
        return cls(data["code"], data["severity"], data["status"], data.get("detail") or {})


def reason(
    code: str,
    *,
    status: Status | str | None = None,
    severity: Severity | str | None = None,
    **detail: Any,
) -> Reason:
    """A ``Reason`` with the registry's status and severity for ``code``.

    An unregistered code needs ``status``; its severity defaults to ``failure`` for a
    status other than ``success`` and to ``warning`` otherwise.
    """
    spec = REASONS.get(code)
    if spec is not None:
        status = spec.status if status is None else status
        severity = spec.severity if severity is None else severity
    elif status is None:
        raise ValueError(f"unregistered reason code {code!r} needs a status")
    status = Status(status)
    if severity is None:
        severity = Severity.WARNING if status == Status.SUCCESS else Severity.FAILURE
    return Reason(code, Severity(severity), status, detail)


@dataclass(frozen=True, slots=True)
class Guidance:
    """A message the user can act on, in Korean and English."""

    code: str
    actions: tuple[Action, ...]
    message_ko: str
    message_en: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "actions": [action.value for action in self.actions],
            "message_ko": self.message_ko,
            "message_en": self.message_en,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Guidance:
        return cls(
            data["code"],
            tuple(Action(action) for action in data["actions"]),
            data["message_ko"],
            data["message_en"],
        )


def _sides(detail: dict[str, Any]) -> list[str]:
    sides = detail.get("sides", detail.get("side"))
    if isinstance(sides, str):
        sides = [sides]
    if not isinstance(sides, list) or not sides or any(side not in _SIDES_KO for side in sides):
        return []
    return [side for side in _SIDES_KO if side in sides]


def _cut_off_messages(detail: dict[str, Any]) -> tuple[str, str]:
    sides = _sides(detail)
    if not sides:
        return (
            "페이지 끝이 사진 밖으로 잘렸어요. 조금 떨어져서 페이지 전체가 보이게 다시 찍어 주세요",
            "The page runs out of the photo. Step back and take it again with the whole page "
            "in view",
        )
    names_ko = [_SIDES_KO[side] for side in sides]
    ko = f"{names_ko[0]}과 {names_ko[1]}" if len(names_ko) == 2 else ", ".join(names_ko)
    if len(sides) == 1:
        en = f"{sides[0]} edge is"
    else:
        en = ", ".join(sides[:-1]) + f" and {sides[-1]} edges are"
    return _CUT_OFF_KO.format(sides=ko), _CUT_OFF_EN.format(sides=en)


def guidance_for(item: Reason) -> Guidance | None:
    """The guidance of one reason, or ``None`` when the user has nothing to do."""
    spec = reason_spec(item.code, item.status)
    if spec is None or spec.message_ko is None or spec.message_en is None:
        return None
    ko, en = spec.message_ko, spec.message_en
    if spec.code == "page_cut_off":
        ko, en = _cut_off_messages(item.detail)
    return Guidance(item.code, spec.actions, ko, en)


def collect_guidance(reasons: Iterable[Reason]) -> list[Guidance]:
    """Guidance of every reason in order, each message once."""
    seen: set[str] = set()
    result = []
    for item in reasons:
        entry = guidance_for(item)
        if entry is not None and entry.message_ko not in seen:
            seen.add(entry.message_ko)
            result.append(entry)
    return result


def decide_outcome(kind: str | None, reasons: Iterable[Reason] = ()) -> Outcome:
    """Outcome of a result whose image came from ``kind`` (``RENDER_KINDS``; ``None``: no
    image). A dewarped image with any warning or failure reason has warnings."""
    if kind is None:
        return Outcome.FAILED
    if kind not in RENDER_KINDS:
        raise ValueError(f"kind must be one of {RENDER_KINDS} or None")
    if kind == "perspective":
        return Outcome.PERSPECTIVE
    if kind == "original":
        return Outcome.ORIGINAL_RETURNED
    if any(item.severity != Severity.INFO for item in reasons):
        return Outcome.RECTIFIED_WITH_WARNINGS
    return Outcome.RECTIFIED


def _pair(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer)) or item < 1
            for item in value
        )
    ):
        raise _plan_error(f"{name} must be a (height, width) pair of positive integers")
    return int(value[0]), int(value[1])


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """The decoded source a plan refers to, without its pixels.

    ``size`` is the (height, width) of the EXIF-oriented image the geometry uses,
    ``full_size`` that of the file at full resolution. ``scale`` is the draft decode
    reduction: a full-resolution pixel edge is ``scale`` times a decoded one.
    """

    size: tuple[int, int]
    full_size: tuple[int, int] | None = None
    scale: int = 1
    format: str | None = None
    file_sha256: str | None = None
    pixels_sha256: str | None = None
    exif_orientation: int | None = None
    focal_length_35mm: float | None = None
    icc_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", _pair(self.size, "size"))
        full = self.size if self.full_size is None else _pair(self.full_size, "full_size")
        object.__setattr__(self, "full_size", full)
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, np.integer)):
            raise _plan_error("scale must be a positive integer")
        if self.scale < 1:
            raise _plan_error("scale must be a positive integer")
        object.__setattr__(self, "scale", int(self.scale))
        if self.focal_length_35mm is not None:
            focal = float(self.focal_length_35mm)
            if not (math.isfinite(focal) and focal > 0):
                raise _plan_error("focal_length_35mm must be positive")
            object.__setattr__(self, "focal_length_35mm", focal)
        if self.exif_orientation is not None:
            object.__setattr__(self, "exif_orientation", int(self.exif_orientation))

    @property
    def downscaled(self) -> bool:
        return self.scale > 1

    def focal_px(self) -> float | None:
        """Focal length in pixels of ``size`` from the EXIF 35 mm equivalent."""
        if self.focal_length_35mm is None:
            return None
        return focal_px_from_35mm(self.focal_length_35mm, self.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": list(self.size),
            "full_size": list(self.full_size),
            "scale": self.scale,
            "format": self.format,
            "file_sha256": self.file_sha256,
            "pixels_sha256": self.pixels_sha256,
            "exif_orientation": self.exif_orientation,
            "focal_length_35mm": self.focal_length_35mm,
            "icc_sha256": self.icc_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceInfo:
        return cls(**data)


def _finite_array(value: Any, shape: tuple[int, ...] | None, name: str) -> NDArray[np.float64]:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise _plan_error(f"{name} must be numeric") from exc
    if shape is not None and array.shape != shape:
        raise _plan_error(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise _plan_error(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise _plan_error(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise _plan_error(f"{name} must be finite")
    return value


@dataclass(frozen=True, eq=False)
class ScanPlan:
    """Everything the render of one scan needs (see the module docstring).

    ``grid`` (``kind="dewarp"``): ``(Gh, Gw, 2)`` flattener values in source coordinates,
    ``flattener.dewarp`` grid convention, upright page order. ``homography``
    (``kind="perspective"``): 3x3 matrix taking output positions ``(u, v, 1)``, u and v in
    [-1, 1] in the flattener output convention, to homogeneous source flattener values.
    ``angle_deg`` is the page angle (clockwise positive, as
    ``dewarp.predict.estimate_page_angle``), ``quarter`` the number k of clockwise
    quarter turns of the page's text-up direction, ``views`` one record per model pass,
    ``provenance`` ``auto`` or ``manual`` per stage. ``nodes`` says where the output
    places the grid nodes (``scan.placement.dense_map``): ``flattener`` (edge-aligned) or
    ``uvdoc`` (align_corners=True, the upstream UVDoc weights). ``fill_rgb`` fills output
    pixels outside the source.
    """

    source: SourceInfo
    output: OutputPlan
    kind: str = "dewarp"
    grid: NDArray[np.float64] | None = None
    homography: NDArray[np.float64] | None = None
    region: dict[str, Any] | None = None
    angle_deg: float = 0.0
    quarter: int = 0
    orientation_probs: tuple[float, ...] | None = None
    views: tuple[dict[str, Any], ...] = ()
    inset: tuple[float, float] = (0.0, 0.0)
    border: float = 0.0
    quantize: str = "round"
    nodes: str = "flattener"
    fill_rgb: tuple[int, int, int] = (255, 255, 255)
    models: tuple[dict[str, Any], ...] = ()
    thresholds: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        def put(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        if not isinstance(self.source, SourceInfo) or not isinstance(self.output, OutputPlan):
            raise _plan_error("source must be a SourceInfo and output an OutputPlan")
        if self.kind not in PLAN_KINDS:
            raise _plan_error(f"kind must be one of {PLAN_KINDS}")
        if self.kind == "dewarp":
            if self.grid is None or self.homography is not None:
                raise _plan_error("a dewarp plan holds a grid and no homography")
            grid = _finite_array(self.grid, None, "grid")
            if grid.ndim != 3 or grid.shape[-1] != 2 or min(grid.shape[:2]) < 2:
                raise _plan_error("grid must have shape (height >= 2, width >= 2, 2)")
            put("grid", grid)
        else:
            if self.homography is None or self.grid is not None:
                raise _plan_error("a perspective plan holds a homography and no grid")
            put("homography", _finite_array(self.homography, (3, 3), "homography"))
        put("angle_deg", _finite(self.angle_deg, "angle_deg"))
        if (
            isinstance(self.quarter, bool)
            or not isinstance(self.quarter, (int, np.integer))
            or not 0 <= self.quarter <= 3
        ):
            raise _plan_error("quarter must be an integer in 0..3")
        put("quarter", int(self.quarter))
        if self.orientation_probs is not None:
            probs = tuple(_finite(p, "orientation_probs") for p in self.orientation_probs)
            if len(probs) != 4:
                raise _plan_error("orientation_probs must hold 4 values")
            put("orientation_probs", probs)
        inset = tuple(_finite(value, "inset") for value in self.inset)
        if len(inset) != 2 or not all(0.0 <= value < 1.0 for value in inset):
            raise _plan_error("inset must be two values in [0, 1)")
        put("inset", inset)
        border = _finite(self.border, "border")
        if border < 0:
            raise _plan_error("border must be at least 0")
        put("border", border)
        if self.quantize not in QUANTIZE_MODES:
            raise _plan_error(f"quantize must be one of {QUANTIZE_MODES}")
        if self.nodes not in NODE_CONVENTIONS:
            raise _plan_error(f"nodes must be one of {NODE_CONVENTIONS}")
        fill = tuple(self.fill_rgb) if isinstance(self.fill_rgb, (tuple, list)) else ()
        if len(fill) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 255
            for v in fill
        ):
            raise _plan_error("fill_rgb must be three integers in 0..255")
        put("fill_rgb", tuple(int(v) for v in fill))
        try:
            put("region", None if self.region is None else json_plain(dict(self.region)))
            put("views", tuple(json_plain(dict(view)) for view in self.views))
            put("models", tuple(json_plain(dict(model)) for model in self.models))
            put("thresholds", json_plain(dict(self.thresholds)))
            put("provenance", json_plain(dict(self.provenance)))
        except (TypeError, ValueError) as exc:
            raise _plan_error(f"plan records must be JSON data: {exc}") from exc
        if any(value not in PROVENANCE_VALUES for value in self.provenance.values()):
            raise _plan_error(f"provenance values must be one of {PROVENANCE_VALUES}")

    def body(self) -> dict[str, Any]:
        """The plan as JSON data without ``plan_sha256``."""
        return {
            "schema": PLAN_SCHEMA,
            "kind": self.kind,
            "source": self.source.to_dict(),
            "output": self.output.to_dict(),
            "grid": None if self.grid is None else self.grid.tolist(),
            "homography": None if self.homography is None else self.homography.tolist(),
            "region": self.region,
            "angle_deg": self.angle_deg,
            "quarter": self.quarter,
            "orientation_probs": (
                None if self.orientation_probs is None else list(self.orientation_probs)
            ),
            "views": list(self.views),
            "inset": list(self.inset),
            "border": self.border,
            "quantize": self.quantize,
            "nodes": self.nodes,
            "fill_rgb": list(self.fill_rgb),
            "models": list(self.models),
            "thresholds": self.thresholds,
            "provenance": self.provenance,
        }

    @cached_property
    def plan_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.body())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "plan_sha256": self.plan_sha256}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, verify: bool = True) -> ScanPlan:
        """Rebuild a plan; with ``verify`` a stored ``plan_sha256`` must match."""
        if not isinstance(data, dict) or data.get("schema") != PLAN_SCHEMA:
            raise _plan_error(f"plan schema must be {PLAN_SCHEMA}")
        unknown = set(data) - PLAN_KEYS
        if unknown:
            raise _plan_error(f"unknown plan fields: {sorted(unknown)}")
        try:
            probs = data.get("orientation_probs")
            plan = cls(
                source=SourceInfo.from_dict(data["source"]),
                output=OutputPlan.from_dict(data["output"]),
                kind=data["kind"],
                grid=data.get("grid"),
                homography=data.get("homography"),
                region=data.get("region"),
                angle_deg=data.get("angle_deg", 0.0),
                quarter=data.get("quarter", 0),
                orientation_probs=None if probs is None else tuple(probs),
                views=tuple(data.get("views", ())),
                inset=tuple(data.get("inset", (0.0, 0.0))),
                border=data.get("border", 0.0),
                quantize=data.get("quantize", "round"),
                nodes=data.get("nodes", "flattener"),
                fill_rgb=tuple(data.get("fill_rgb", (255, 255, 255))),
                models=tuple(data.get("models", ())),
                thresholds=data.get("thresholds", {}),
                provenance=data.get("provenance", {}),
            )
        except FlattenerError as exc:
            if exc.reason_code == "invalid_plan":
                raise
            raise _plan_error(exc.message) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise _plan_error(f"malformed plan: {exc}") from exc
        stored = data.get("plan_sha256")
        if verify and stored is not None and stored != plan.plan_sha256:
            raise FlattenerError(
                Status.INPUT_ERROR,
                "plan_sha256_mismatch",
                f"plan_sha256 {stored} does not match the plan ({plan.plan_sha256})",
            )
        return plan

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScanPlan):
            return NotImplemented
        return self.plan_sha256 == other.plan_sha256

    def __hash__(self) -> int:
        return hash(self.plan_sha256)


@dataclass(eq=False)
class ScanResult:
    """The record of one scan (see the module docstring).

    ``status`` and ``reason_code`` follow ``RenderResult``: a fallback keeps the failure
    status and sets ``used_fallback``. ``candidates`` holds alternatives the user can
    pick (page boxes, the top orientations), ``qa`` the checks (coverage, drift,
    orientation margin, density), ``timings`` seconds per stage and ``metadata`` the
    renderer metadata.
    """

    status: Status
    outcome: Outcome
    reason_code: str | None = None
    message: str = ""
    used_fallback: bool = False
    reasons: list[Reason] = field(default_factory=list)
    guidance: list[Guidance] = field(default_factory=list)
    candidates: dict[str, Any] = field(default_factory=dict)
    image: NDArray[np.uint8] | None = None
    preview: NDArray[np.uint8] | None = None
    valid_mask: NDArray[np.bool_] | None = None
    plan: ScanPlan | None = None
    source: SourceInfo | None = None
    qa: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    models: list[dict[str, Any]] = field(default_factory=list)
    shippable: bool | None = None
    research: bool = False
    evidence: dict[str, Any] | None = None
    quality_status: str = "not_evaluated"
    input: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = Status(self.status)
        self.outcome = Outcome(self.outcome)

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON data (arrays left out)."""
        return {
            "schema": RECORD_SCHEMA,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "message": self.message,
            "outcome": self.outcome.value,
            "used_fallback": bool(self.used_fallback),
            "reasons": [item.to_dict() for item in self.reasons],
            "guidance": [item.to_dict() for item in self.guidance],
            "candidates": json_plain(self.candidates),
            "qa": json_plain(self.qa),
            "timings": json_plain(self.timings),
            "input": self.input,
            "source": None if self.source is None else self.source.to_dict(),
            "plan_sha256": None if self.plan is None else self.plan.plan_sha256,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "models": json_plain(self.models),
            "shippable": self.shippable,
            "research": bool(self.research),
            "evidence": json_plain(self.evidence),
            "quality_status": self.quality_status,
            "artifacts": json_plain(self.artifacts),
            "metadata": json_plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScanResult:
        """Rebuild a record from ``to_dict`` output (no arrays); the plan is verified."""
        if not isinstance(data, dict) or data.get("schema") != RECORD_SCHEMA:
            raise _record_error(f"record schema must be {RECORD_SCHEMA}")
        plan = None if data.get("plan") is None else ScanPlan.from_dict(data["plan"])
        if plan is not None and data.get("plan_sha256") not in (None, plan.plan_sha256):
            raise FlattenerError(
                Status.INPUT_ERROR,
                "plan_sha256_mismatch",
                "the record's plan_sha256 does not match its plan",
            )
        try:
            return cls(
                status=data["status"],
                outcome=data["outcome"],
                reason_code=data.get("reason_code"),
                message=data.get("message", ""),
                used_fallback=bool(data.get("used_fallback", False)),
                reasons=[Reason.from_dict(item) for item in data.get("reasons", [])],
                guidance=[Guidance.from_dict(item) for item in data.get("guidance", [])],
                candidates=data.get("candidates") or {},
                plan=plan,
                source=None if data.get("source") is None else SourceInfo.from_dict(data["source"]),
                qa=data.get("qa") or {},
                timings=data.get("timings") or {},
                metadata=data.get("metadata") or {},
                models=data.get("models") or [],
                shippable=data.get("shippable"),
                research=bool(data.get("research", False)),
                evidence=data.get("evidence"),
                quality_status=data.get("quality_status", "not_evaluated"),
                input=data.get("input"),
                artifacts=data.get("artifacts") or {},
            )
        except FlattenerError as exc:
            raise _record_error(exc.message) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise _record_error(f"malformed scan record: {exc}") from exc
