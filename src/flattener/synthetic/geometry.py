"""Independent analytic geometry for reproducible synthetic document pairs.

The flat page is the physical square ``[-1, 1] ** 2``. Its coordinates first
undergo two triangular shears, then an orientation-preserving homography::

    x1 = u + shear_x * sin(pi * (v + 1) / 2)
    y1 = v + bend_y * profile(x1)
    [x, y, 1] ~ H @ [x1, y1, 1]

The sine profile is used by ``gentle`` and ``strong``; ``local`` uses a
Gaussian. ``perspective`` has both shear amplitudes zero. Inversion applies
``H ** -1``, recovers ``v`` from ``y1``, then ``u`` from ``x1``. No renderer,
optimization, resampled field, or model output supplies coordinate truth.

This is a two-dimensional, bijective image deformation model, not a physical
three-dimensional sheet simulation. It models neither self-occlusion nor
folds. Cropping can remove visible correspondences and is recorded by masks.
"""

from dataclasses import asdict, dataclass
from math import cos, isfinite, sin
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..contracts import FlattenerError, RenderConfig, Status
from ..validation import validate_size

FAMILIES = ("perspective", "gentle", "strong", "local")
_CHUNK_POINTS = 65_536
_POLE_TOLERANCE = 1e-12


def _invalid(message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, "invalid_synthetic_geometry", message)


@dataclass(frozen=True, slots=True)
class WarpParameters:
    """An invertible page warp, in normalized physical-image coordinates.

    Rotation is in radians. Positive scales, a positive homography
    determinant, and a conservative positive-denominator bound over the
    entire page are required. Translation may intentionally crop the page.
    The conservative bound can reject some otherwise valid homographies.
    """

    family: str = "perspective"
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    perspective_x: float = 0.0
    perspective_y: float = 0.0
    shear_x: float = 0.0
    bend_y: float = 0.0
    local_center: float = 0.0
    local_width: float = 0.35

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or self.family not in FAMILIES:
            raise _invalid(f"family must be one of {FAMILIES}")
        for name in self.__dataclass_fields__:
            if name == "family":
                continue
            value = getattr(self, name)
            try:
                valid = (
                    isinstance(value, Real)
                    and not isinstance(value, (bool, np.bool_))
                    and isfinite(value)
                )
            except (OverflowError, TypeError, ValueError):
                valid = False
            if not valid:
                raise _invalid(f"{name} must be a finite real number")
            # Canonical Python floats keep parameter metadata JSON-serializable.
            object.__setattr__(self, name, float(value))
        if min(self.scale_x, self.scale_y, self.local_width) <= 0:
            raise _invalid("scale_x, scale_y, and local_width must be positive")
        if self.family == "perspective" and (self.shear_x != 0 or self.bend_y != 0):
            raise _invalid("perspective geometry requires zero shear_x and bend_y")
        minimum = _denominator_lower_bound(self)
        if not isfinite(minimum) or minimum <= _POLE_TOLERANCE:
            raise _invalid("homography denominator must remain positive over the entire page")
        with np.errstate(over="ignore", invalid="ignore"):
            determinant = float(np.linalg.det(_homography(self)))
        if not isfinite(determinant) or determinant <= _POLE_TOLERANCE:
            raise _invalid("homography must be nonsingular and orientation-preserving")


@dataclass(frozen=True, slots=True)
class GeometryMaps:
    """Both sampling directions and explicit masks for available ground truth.

    ``backward_map`` is indexed by flat output pixels and reads the photo.
    ``synthesis_map`` is indexed by photo pixels and reads the flat document.
    Arrays are float32 in (x, y) order, with align_corners=False coordinates.
    Missing correspondence pairs are both NaN, with the matching mask false.
    """

    backward_map: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    synthesis_map: NDArray[np.float32]
    photo_page_mask: NDArray[np.bool_]
    metadata: dict[str, Any]


def _homography(params: WarpParameters) -> NDArray[np.float64]:
    c, s = cos(params.rotation), sin(params.rotation)
    return np.array(
        [
            [params.scale_x * c, -params.scale_y * s, params.translate_x],
            [params.scale_x * s, params.scale_y * c, params.translate_y],
            [params.perspective_x, params.perspective_y, 1.0],
        ],
        dtype=np.float64,
    )


def _denominator_lower_bound(params: WarpParameters) -> float:
    return (
        1.0
        - abs(params.perspective_x) * (1.0 + abs(params.shear_x))
        - abs(params.perspective_y) * (1.0 + abs(params.bend_y))
    )


def sample_warp(family: str, seed: int) -> WarpParameters:
    """Sample an independently designed parameter distribution using PCG64.

    All ranges below were chosen for this project, not copied from another
    implementation. Scales are sampled independently on the two axes.
    ``rotation``, translations, perspective, and unmarked shear amplitudes
    use symmetric uniform ranges. Nonzero bends have a random sign.

    =========== =========== ======== =========== =========== ============= =============
    family      scales      rotation translation perspective shear_x       abs(bend_y)
    =========== =========== ======== =========== =========== ============= =============
    perspective [0.65,0.80] 0.09     0.035       0.045       0             0
    gentle      [0.60,0.72] 0.07     0.025       0.035       0.07          [0.08,0.15]
    strong      [0.50,0.60] 0.06     0.020       0.025       abs[.08,.16]  [0.24,0.38]
    local       [0.57,0.67] 0.06     0.020       0.030       0.06          [0.17,0.28]
    =========== =========== ======== =========== =========== ============= =============

    Local Gaussian center is in [-0.35,0.35] and width in [0.16,0.30]. These
    bounds keep the entire continuous page inside the photo, including curve
    extrema between corners. Manually supplied parameters may crop a page.
    Seeds are nonboolean integers in [0, 2**63).
    """
    if not isinstance(family, str) or family not in FAMILIES:
        raise _invalid(f"family must be one of {FAMILIES}")
    if (
        not isinstance(seed, Integral)
        or isinstance(seed, (bool, np.bool_))
        or not 0 <= seed < 2**63
    ):
        raise _invalid("seed must be a nonboolean integer in [0, 2**63)")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    ranges = {
        "perspective": (0.65, 0.80, 0.09, 0.035, 0.045, 0.0, 0.0, 0.0),
        "gentle": (0.60, 0.72, 0.07, 0.025, 0.035, 0.07, 0.08, 0.15),
        "strong": (0.50, 0.60, 0.06, 0.020, 0.025, 0.16, 0.24, 0.38),
        "local": (0.57, 0.67, 0.06, 0.020, 0.030, 0.06, 0.17, 0.28),
    }
    lo, hi, rotation, translation, perspective, shear, bend_lo, bend_hi = ranges[family]

    def signed(low: float, high: float) -> float:
        return float(rng.uniform(low, high) * rng.choice([-1.0, 1.0]))

    return WarpParameters(
        family=family,
        scale_x=float(rng.uniform(lo, hi)),
        scale_y=float(rng.uniform(lo, hi)),
        rotation=float(rng.uniform(-rotation, rotation)),
        translate_x=float(rng.uniform(-translation, translation)),
        translate_y=float(rng.uniform(-translation, translation)),
        perspective_x=float(rng.uniform(-perspective, perspective)),
        perspective_y=float(rng.uniform(-perspective, perspective)),
        shear_x=signed(0.08, shear) if family == "strong" else float(rng.uniform(-shear, shear)),
        bend_y=signed(bend_lo, bend_hi) if family != "perspective" else 0.0,
        local_center=float(rng.uniform(-0.35, 0.35)) if family == "local" else 0.0,
        local_width=float(rng.uniform(0.16, 0.30)) if family == "local" else 0.35,
    )


def _points(points: NDArray) -> NDArray[np.float64]:
    if not isinstance(points, np.ndarray):
        raise _invalid("points must be a NumPy array")
    if points.ndim < 1 or points.shape[-1] != 2:
        raise _invalid("points must have shape (..., 2) in (x, y) order")
    if points.dtype.kind not in "fiu":
        raise _invalid("points must have a real numeric dtype")
    return points.astype(np.float64)


def _parameters(params: WarpParameters) -> WarpParameters:
    if not isinstance(params, WarpParameters):
        raise _invalid("params must be WarpParameters")
    return params


def _profile(x: NDArray[np.float64], params: WarpParameters) -> NDArray[np.float64]:
    if params.family == "local":
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return np.exp(-0.5 * ((x - params.local_center) / params.local_width) ** 2)
    if params.family == "perspective":
        return np.zeros_like(x)
    return np.sin(np.pi * (x + 1.0) / 2.0)


def _project(points: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        numerator = points @ matrix[:2, :2].T + matrix[:2, 2]
        denominator = points @ matrix[2, :2] + matrix[2, 2]
        result = numerator / denominator[..., None]
    valid = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(result).all(axis=-1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > _POLE_TOLERANCE)
    )
    return np.where(valid[..., None], result, np.nan)


def _forward(points: NDArray[np.float64], params: WarpParameters) -> NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        x = points[..., 0] + params.shear_x * np.sin(np.pi * (points[..., 1] + 1.0) / 2.0)
        y = points[..., 1] + params.bend_y * _profile(x, params)
    return _project(np.stack((x, y), axis=-1), _homography(params))


def _inverse(points: NDArray[np.float64], params: WarpParameters) -> NDArray[np.float64]:
    sheared = _project(points, np.linalg.inv(_homography(params)))
    with np.errstate(over="ignore", invalid="ignore"):
        v = sheared[..., 1] - params.bend_y * _profile(sheared[..., 0], params)
        u = sheared[..., 0] - params.shear_x * np.sin(np.pi * (v + 1.0) / 2.0)
    result = np.stack((u, v), axis=-1)
    return np.where(np.isfinite(result).all(axis=-1)[..., None], result, np.nan)


def _float32(points: NDArray[np.float64]) -> NDArray[np.float32]:
    valid = np.isfinite(points).all(axis=-1) & (np.abs(points) <= np.finfo(np.float32).max).all(
        axis=-1
    )
    return np.where(valid[..., None], points, np.nan).astype(np.float32)


def flat_to_photo(points: NDArray, params: WarpParameters) -> NDArray[np.float32]:
    """Map normalized flat points to photo points without clipping.

    Nonfinite input pairs, projective poles, and float32 overflow yield NaN
    pairs. Finite coordinates outside either image remain outside the image;
    use ``make_geometry`` when visibility masks and missing labels are needed.
    """
    return _float32(_forward(_points(points), _parameters(params)))


def photo_to_flat(points: NDArray, params: WarpParameters) -> NDArray[np.float32]:
    """Apply the analytic inverse; poles yield NaN, never invented labels."""
    return _float32(_inverse(_points(points), _parameters(params)))


def _pixel_centers(height: int, width: int, start: int, stop: int) -> NDArray[np.float64]:
    result = np.empty((stop - start, width, 2), dtype=np.float64)
    result[..., 0] = 2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0
    result[..., 1] = (2.0 * (np.arange(start, stop, dtype=np.float64) + 0.5) / height - 1.0)[
        :, None
    ]
    return result


def _make_map(
    output_size: tuple[int, int], params: WarpParameters, *, forward: bool
) -> tuple[NDArray[np.float32], NDArray[np.bool_], float | None]:
    height, width = output_size
    coordinate_map = np.empty((height, width, 2), dtype=np.float32)
    mask = np.empty((height, width), dtype=np.bool_)
    maximum_error: float | None = None
    transform, inverse = (_forward, _inverse) if forward else (_inverse, _forward)
    rows = max(1, _CHUNK_POINTS // width)
    for start in range(0, height, rows):
        stop = min(start + rows, height)
        centers = _pixel_centers(height, width, start, stop)
        mapped = transform(centers, params)
        # Test physical visibility before float32 rounding. Never clip a
        # coordinate into range and then count it as a visible correspondence.
        visible = np.isfinite(mapped).all(axis=-1) & (np.abs(mapped) <= 1.0).all(axis=-1)
        stored = _float32(mapped)
        stored[~visible] = np.nan
        coordinate_map[start:stop] = stored
        mask[start:stop] = visible
        if visible.any():
            # Include float32 label quantization in the round-trip diagnostic.
            restored = inverse(stored[visible].astype(np.float64), params)
            pixel_error = (restored - centers[visible]) * np.array([width / 2, height / 2])
            chunk_error = float(np.linalg.norm(pixel_error, axis=-1).max())
            maximum_error = (
                chunk_error if maximum_error is None else max(maximum_error, chunk_error)
            )
    return coordinate_map, mask, maximum_error


def make_geometry(
    flat_size: tuple[int, int], photo_size: tuple[int, int], params: WarpParameters
) -> GeometryMaps:
    """Construct independent analytic maps for the requested (height, width).

    Default renderer resource limits apply before allocation. Masks indicate
    point correspondence in the continuous images; they do not claim full
    bilinear footprint coverage or account for later appearance corruption.
    The two inversion diagnostics are in each output image's pixel units.
    """
    params = _parameters(params)
    limits = RenderConfig()
    flat_size = validate_size(
        flat_size,
        max_pixels=limits.max_output_pixels,
        max_side=limits.max_side,
        name="flat_size",
    )
    photo_size = validate_size(
        photo_size,
        max_pixels=limits.max_input_pixels,
        max_side=limits.max_side,
        name="photo_size",
    )
    backward, valid, flat_error = _make_map(flat_size, params, forward=True)
    synthesis, page, photo_error = _make_map(photo_size, params, forward=False)
    errors = [value for value in (flat_error, photo_error) if value is not None]
    metadata: dict[str, Any] = {
        "model": "analytic_triangular_shears_then_homography_v1",
        "family": params.family,
        "parameters": asdict(params),
        "flat_size": list(flat_size),
        "photo_size": list(photo_size),
        "coordinate_order": "xy",
        "coordinate_units": "normalized_physical_image_bounds_minus1_plus1",
        "align_corners": False,
        "backward_map_direction": "flat_output_to_photo_source",
        "synthesis_map_direction": "photo_output_to_flat_source",
        "missing_coordinates": "NaN_pairs_with_false_mask",
        "visibility": "continuous_point_correspondence_not_bilinear_footprint",
        "occlusion_model": "none_bijective_2d_deformation",
        "folds": False,
        "homography_determinant": float(np.linalg.det(_homography(params))),
        "homography_denominator_lower_bound": _denominator_lower_bound(params),
        "flat_visible_fraction": float(valid.mean()),
        "photo_page_fraction": float(page.mean()),
        "inverse_error_flat_pixels_max": flat_error,
        "inverse_error_photo_pixels_max": photo_error,
        "inverse_error_pixels_max": max(errors) if errors else None,
        "inverse_error_includes_float32_quantization": True,
    }
    return GeometryMaps(backward, valid, synthesis, page, metadata)
