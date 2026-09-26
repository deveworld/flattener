"""Known cylindrical 3-D augmentation of caller-supplied flat RGB documents.

The printed texture is mapped onto an extruded cylinder, viewed by a pinhole
camera, then placed by a pole-free, orientation-preserving homography. The
closed forward map supplies labels; a monotone 65,537-point lookup table
supplies the inverse used to photograph the texture. Appearance never moves
coordinates. This is controlled augmentation, not ground truth for natural
curled photographs or a claim about a supplied template's licensing.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from PIL import Image

from ..contracts import FlattenerError, RenderConfig, Status
from ..renderer import render
from .geometry import GeometryMaps
from .pipeline import PhotometricConfig, apply_photometry, derive_seed
from .scenes import generate_background

FAMILIES = ("flat", "gentle", "strong")
SIZE_BINS = ("small", "medium", "large")
MODEL_VERSION = "extruded_cylinder_perspective_v1"
_LUT_SIZE = 65_537
_PAGE_MARGIN = 0.98


def _invalid(message):
    return FlattenerError(Status.INPUT_ERROR, "invalid_paper_geometry", message)


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise _invalid("seed must be an integer in [0, 2**63)")
    return seed


def _size(size, low, high, name):
    if (not isinstance(size, tuple) or len(size) != 2
            or any(type(v) is not int or not low <= v <= high for v in size)):
        raise _invalid(f"{name} must have integer (height, width) sides in [{low}, {high}]")
    return size


@dataclass(frozen=True, slots=True)
class PaperParameters:
    """Cylinder curvature in radians per normalized unit; camera focal = distance.

    The flat physical domain is [-1,1]^2. Anisotropic placement scales may
    change its apparent aspect ratio. Signed curvature bends toward/away from
    the camera; the parameter bounds exclude self-occlusion and foldovers.
    """

    axis: str = "x"
    curvature: float = 0.0
    camera_distance: float = 3.0
    scale_x: float = 0.75
    scale_y: float = 0.75
    rotation: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    perspective_x: float = 0.0
    perspective_y: float = 0.0

    def __post_init__(self):
        if not isinstance(self.axis, str) or self.axis not in ("x", "y"):
            raise _invalid("axis must be x or y")
        limits = {
            "curvature": (-1.2, 1.2), "camera_distance": (2.5, 8.0),
            "scale_x": (0.05, 2.0), "scale_y": (0.05, 2.0),
            "rotation": (-math.pi, math.pi), "translate_x": (-2.0, 2.0),
            "translate_y": (-2.0, 2.0), "perspective_x": (-0.2, 0.2),
            "perspective_y": (-0.2, 0.2),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if (type(value) not in (int, float) or not low <= value <= high
                    or not math.isfinite(value)):
                raise _invalid(f"{name} must be finite in [{low}, {high}]")
            object.__setattr__(self, name, float(value))
        if _denominator_bound(self) <= 0.05 or np.linalg.det(_homography(self)) <= 1e-6:
            raise _invalid("placement must be nonsingular, orientation preserving and pole free")


@dataclass(slots=True)
class PaperSample:
    photo: np.ndarray
    geometry: GeometryMaps
    metadata: dict

    @property
    def image(self):
        return self.photo

    @property
    def mask(self):
        return self.geometry.photo_page_mask


def _parameters(params):
    if not isinstance(params, PaperParameters):
        raise _invalid("params must be PaperParameters")
    return params


def _cylinder(coordinate, params):
    # sinc and sin² avoid cancellation and remain exact in the k=0 limit.
    k = params.curvature
    x = coordinate * np.sinc(k * coordinate / np.pi)
    z = 0.5 * k * coordinate**2 * np.sinc(k * coordinate / (2 * np.pi))**2
    return x, z


def _projected_bounds(params):
    x, z = _cylinder(np.float64(1), params)
    bent = params.camera_distance * x / (params.camera_distance - z)
    # For negative curvature the transverse maximum is at the edge midpoint,
    # not at a corner. This rectangle bounds the complete continuous page.
    straight = max(1.0, params.camera_distance / (params.camera_distance - z))
    return np.array((bent, straight) if params.axis == "x" else (straight, bent))


def _homography(params):
    c, s = math.cos(params.rotation), math.sin(params.rotation)
    return np.array([[params.scale_x * c, -params.scale_y * s, params.translate_x],
                     [params.scale_x * s, params.scale_y * c, params.translate_y],
                     [params.perspective_x, params.perspective_y, 1.0]], dtype=np.float64)


def _denominator_bound(params):
    return 1 - np.dot(np.abs((params.perspective_x, params.perspective_y)),
                      _projected_bounds(params))


def _project(points, matrix):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        denominator = points @ matrix[2, :2] + matrix[2, 2]
        result = (points @ matrix[:2, :2].T + matrix[:2, 2]) / denominator[..., None]
    valid = np.isfinite(result).all(-1) & np.isfinite(points).all(-1)
    valid &= np.abs(denominator) > 1e-12
    return np.where(valid[..., None], result, np.nan)


def _points(points):
    if (not isinstance(points, np.ndarray) or points.ndim < 1 or points.shape[-1] != 2
            or points.dtype.kind not in "fiu" or points.size > 4_194_304):
        raise _invalid("points must be a bounded real array with shape (...,2)")
    return points.astype(np.float64)


def _forward(points, params):
    axis = 0 if params.axis == "x" else 1
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        curved, depth = _cylinder(points[..., axis], params)
        world = points.copy()
        world[..., axis] = curved
        projected = world * (
            params.camera_distance / (params.camera_distance - depth)
        )[..., None]
    return _project(projected, _homography(params))


def _inverse(points, params):
    projected = _project(points, np.linalg.inv(_homography(params)))
    axis = 0 if params.axis == "x" else 1
    # A narrow continuation avoids losing float32-quantized boundary points.
    # Outside the actual page this returns |u|>1, NOT a clamped edge label.
    # da/du = d*[d*cos(ku)+(1-cos(ku))/k]/(d-z)^2 > 0 on the
    # allowed page neighborhood. At negative k the numerator is still >.3.
    coordinate = np.linspace(-1.01, 1.01, _LUT_SIZE)
    curved, depth = _cylinder(coordinate, params)
    lookup = params.camera_distance * curved / (params.camera_distance - depth)
    bent = np.interp(projected[..., axis], lookup, coordinate, left=np.nan, right=np.nan)
    _, depth = _cylinder(bent, params)
    flat = projected * ((params.camera_distance - depth) / params.camera_distance)[..., None]
    flat[..., axis] = bent
    return np.where(np.isfinite(flat).all(-1)[..., None], flat, np.nan)


def flat_to_photo(points, params):
    """Closed 3-D forward projection in xy normalized physical-image coordinates."""
    points, params = _points(points), _parameters(params)
    if np.any(np.abs(points[np.isfinite(points)]) > 1.01):
        raise _invalid("flat points must lie within the supported page neighborhood [-1.01,1.01]")
    return _forward(points, params).astype(np.float32)


def photo_to_flat(points, params):
    """Invert the monotone cylinder; no endpoint clamping outside LUT support."""
    return _inverse(_points(points), _parameters(params)).astype(np.float32)


def _centers(size):
    h, w = size
    x, y = np.meshgrid(2 * (np.arange(w) + 0.5) / w - 1,
                       2 * (np.arange(h) + 0.5) / h - 1)
    return np.stack((x, y), -1)


def _map(size, transform, inverse):
    centers = _centers(size)
    mapped = transform(centers)
    valid = np.isfinite(mapped).all(-1) & (np.abs(mapped) <= 1).all(-1)
    stored = np.where(valid[..., None], mapped, np.nan).astype(np.float32)
    error = None
    if valid.any():
        restored = inverse(stored[valid].astype(np.float64))
        delta = (restored - centers[valid]) * np.array((size[1] / 2, size[0] / 2))
        error = float(np.linalg.norm(delta, axis=-1).max())
        if not math.isfinite(error) or error > 0.01:
            raise _invalid("quantized coordinate inversion exceeds 0.01 pixel tolerance")
    return stored, valid, error


def _maps(output_size, photo_size, forward, inverse, metadata):
    backward, valid, flat_error = _map(output_size, forward, inverse)
    synthesis, page, photo_error = _map(photo_size, inverse, forward)
    errors = [v for v in (flat_error, photo_error) if v is not None]
    metadata.update(
        flat_size=list(output_size), photo_size=list(photo_size), coordinate_order="xy",
        coordinate_units="normalized_physical_image_bounds_minus1_plus1", align_corners=False,
        backward_map_direction="flat_output_to_photo_source",
        synthesis_map_direction="photo_output_to_flat_source",
        missing_coordinates="NaN_pairs_with_false_mask",
        visibility="continuous_point_correspondence_not_bilinear_footprint",
        flat_visible_fraction=float(valid.mean()), photo_page_fraction=float(page.mean()),
        inverse_error_flat_pixels_max=flat_error, inverse_error_photo_pixels_max=photo_error,
        inverse_error_pixels_max=max(errors) if errors else None,
        inverse_error_includes_float32_quantization=True,
    )
    return GeometryMaps(backward, valid, synthesis, page, metadata)


def make_geometry(output_size, photo_size, params):
    """Recompute known maps; photo sides 128..512 and flat-output sides 2..512."""
    _size(output_size, 2, 512, "output_size")
    _size(photo_size, 128, 512, "photo_size")
    params = _parameters(params)
    return _maps(output_size, photo_size, lambda p: _forward(p, params),
                 lambda p: _inverse(p, params), {
                     "model": MODEL_VERSION, "parameters": asdict(params),
                     "occlusion_model": "none_monotone_extruded_cylinder", "folds": False,
                     "camera_focal_length": params.camera_distance,
                     "inverse_lut_points": _LUT_SIZE,
                     "homography_denominator_lower_bound": float(_denominator_bound(params)),
                     "homography_determinant": float(np.linalg.det(_homography(params))),
                 })


def _sampling_args(family, seed, size_bin):
    _seed(seed)
    if (not isinstance(family, str) or family not in FAMILIES
            or not isinstance(size_bin, str) or size_bin not in SIZE_BINS):
        raise _invalid("unsupported paper family or size bin")


def paper_parameters_for_attempt(family, seed, *, size_bin="medium", attempt=1):
    """Reproduce a candidate without constructing pixel maps or reading images."""
    _sampling_args(family, seed, size_bin)
    if type(attempt) is not int or not 1 <= attempt <= 256:
        raise _invalid("attempt must be an integer in [1,256]")
    rng = np.random.default_rng(derive_seed(seed, "paper-geometry-attempt", str(attempt)))
    low, high = {"flat": (0, 0), "gentle": (0.2, 0.6), "strong": (0.7, 1.2)}[family]
    params = PaperParameters(axis=str(rng.choice(("x", "y"))),
                             curvature=float(rng.uniform(low, high) * rng.choice((-1, 1))),
                             camera_distance=float(rng.uniform(2.5, 4.5)))
    bounds = _projected_bounds(params)
    rotation_limit = {"small": np.pi / 12, "medium": 0.13, "large": 0.05}[size_bin]
    rotation = float(rng.uniform(-rotation_limit, rotation_limit))
    perspective = rng.uniform(-0.015, 0.015, 2)
    denominator = 1 - np.dot(np.abs(perspective), bounds)
    scales = rng.uniform(0.94, 1.0, 2) / bounds
    c, s = abs(math.cos(rotation)), abs(math.sin(rotation))
    extents = np.array((c * scales[0] * bounds[0] + s * scales[1] * bounds[1],
                        s * scales[0] * bounds[0] + c * scales[1] * bounds[1]))
    extent_range = {"small": (0.35, 0.48), "medium": (0.64, 0.77),
                    "large": (0.90, 0.965)}[size_bin]
    multiplier = rng.uniform(*extent_range) * denominator / extents.max()
    limits = _PAGE_MARGIN * denominator - extents * multiplier
    translation = rng.uniform(-limits, limits)
    return replace(params, scale_x=float(scales[0] * multiplier),
                   scale_y=float(scales[1] * multiplier), rotation=rotation,
                   translate_x=float(translation[0]), translate_y=float(translation[1]),
                   perspective_x=float(perspective[0]), perspective_y=float(perspective[1]))


def sample_paper_geometry(family, seed, *, size_bin="medium", photo_size=(384, 288),
                          output_size=(192, 144)):
    """Choose in <=256 geometry-only attempts by actual source-grid page area."""
    _sampling_args(family, seed, size_bin)
    _size(output_size, 2, 512, "output_size")
    _size(photo_size, 128, 512, "photo_size")
    for attempt in range(1, 257):
        params = paper_parameters_for_attempt(family, seed, size_bin=size_bin, attempt=attempt)
        geometry = make_geometry(output_size, photo_size, params)
        fraction = float(geometry.photo_page_mask.mean())
        actual = "small" if fraction < 0.25 else "medium" if fraction < 0.5 else "large"
        if fraction > 0 and actual == size_bin:
            return params, geometry, {
                "requested_size_bin": size_bin, "actual_size_bin": actual,
                "foreground_fraction": fraction, "geometry_seed": seed, "attempts": attempt,
                "maximum_attempts": 256, "continuous_page_absolute_bound": _PAGE_MARGIN,
                "criterion": "actual analytic source-grid mask.mean(); geometry only",
            }
    raise _invalid("could not sample requested paper size bin in 256 attempts")


def sample_paper_parameters(family, seed, *, size_bin="medium", photo_size=(384, 288)):
    return sample_paper_geometry(family, seed, size_bin=size_bin, photo_size=photo_size)[0]


def generate_paper_sample(flat_rgb, seed, params, *, photo_size=(384, 288),
                           output_size=(192, 144), background=None,
                           photometric: PhotometricConfig | None = None,
                           white_balance_strength=0.08, jpeg_quality=90):
    """Composite a supplied flat template, then apply geometry-invariant appearance.

    Inputs must already be RGB and oriented; no EXIF or license is inferred.
    Set photometric=PhotometricConfig(enabled=False), white_balance_strength=0,
    jpeg_quality=None to disable appearance corruption. The seed here controls
    appearance/background, NOT parameter sampling; record that sampling seed
    separately when constructing a dataset manifest.
    """
    _seed(seed)
    if (not isinstance(flat_rgb, np.ndarray) or flat_rgb.dtype != np.uint8
            or flat_rgb.ndim != 3 or flat_rgb.shape[-1] != 3
            or any(not 2 <= v <= 4096 for v in flat_rgb.shape[:2])):
        raise _invalid("flat_rgb must be RGB uint8 with sides in [2,4096]")
    if (type(white_balance_strength) not in (int, float)
            or not 0 <= white_balance_strength <= 0.3 or not math.isfinite(white_balance_strength)):
        raise _invalid("white_balance_strength must be finite in [0,0.3]")
    if jpeg_quality is not None and (
            type(jpeg_quality) is not int or not 30 <= jpeg_quality <= 100):
        raise _invalid("jpeg_quality must be None or an integer in [30,100]")
    geometry = make_geometry(output_size, photo_size, params)
    background_seed = derive_seed(seed, "paper-background")
    if background is None:
        background, background_meta = generate_background(photo_size, background_seed)
    else:
        if (not isinstance(background, np.ndarray) or background.dtype != np.uint8
                or background.shape != (*photo_size, 3)):
            raise _invalid("background must be RGB uint8 at photo_size")
        background_meta = {"source": "caller supplied", "sha256": hashlib.sha256(
            background.tobytes(order="C")).hexdigest()}
    h, w = photo_size
    # Supersampling changes only raster coverage, never the pixel-center masks.
    high_map, high_mask, _ = _map((h * 2, w * 2), lambda p: _inverse(p, params),
                                lambda p: _forward(p, params))
    rendered = render(flat_rgb, high_map, target_mask=high_mask, config=RenderConfig(
        fill_rgb=(0, 0, 0), max_out_of_bounds_fraction=1, max_input_pixels=4096**2,
    ))
    page = rendered.image.astype(np.float32).reshape(h, 2, w, 2, 3).mean((1, 3))
    alpha = high_mask.reshape(h, 2, w, 2).mean((1, 3))
    scene = np.floor(np.clip(page + background * (1 - alpha[..., None]), 0, 255) + 0.5)
    appearance_seed = derive_seed(seed, "paper-photometry")
    rng = np.random.default_rng(derive_seed(seed, "paper-white-balance"))
    white_balance = rng.uniform(1 - white_balance_strength, 1 + white_balance_strength, 3)
    scene = np.floor(np.clip(scene * white_balance, 0, 255) + 0.5).astype(np.uint8)
    photo, appearance = apply_photometry(scene, appearance_seed, photometric)
    if jpeg_quality is not None:
        with io.BytesIO() as stream:
            with Image.fromarray(photo) as encoded:
                encoded.save(stream, format="JPEG", quality=jpeg_quality, subsampling=0)
            stream.seek(0)
            with Image.open(stream) as decoded:
                photo = np.array(decoded, dtype=np.uint8, copy=True)
    metadata = {
        "schema_version": "paper-0.1", "geometry": geometry.metadata,
        "seeds": {"sample": seed, "background": background_seed, "photometric": appearance_seed},
        "background": background_meta, "photometric": appearance,
        "white_balance_rgb": white_balance.tolist(), "jpeg_quality": jpeg_quality,
        "template_size": list(flat_rgb.shape[:2]),
        "template_sha256": hashlib.sha256(flat_rgb.tobytes(order="C")).hexdigest(),
        "template_aspect_ratio_width_over_height": flat_rgb.shape[1] / flat_rgb.shape[0],
        "is_real_photo": False,
        "source": "caller-supplied flat RGB texture; controlled 3-D augmentation",
        "template_terms": "caller must verify and record source permission separately",
        "mask_definition": "analytic page membership at photo pixel centers",
        "compositing": {"supersampling": 2, "alpha": "mean analytic subpixel occupancy"},
    }
    return PaperSample(photo, geometry, metadata)


def make_capture_geometry(corners_xy, original_size, output_size, photo_size):
    """Planar approximation from manual original-photo TL,TR,BR,BL pixel centers.

    Normalized source coordinates survive full-image resize. This is not
    dense measured truth for curvature, occlusion, or inaccurate annotations.
    """
    _size(original_size, 2, 16_384, "original_size")
    _size(output_size, 2, 512, "output_size")
    _size(photo_size, 128, 512, "photo_size")
    try:
        raw = np.asarray(corners_xy)
    except (TypeError, ValueError) as exc:
        raise _invalid("corners_xy must be a rectangular numeric array") from exc
    if (not isinstance(corners_xy, (list, tuple, np.ndarray)) or raw.shape != (4, 2)):
        raise _invalid("corners_xy must contain TL,TR,BR,BL pixel-center pairs")
    if raw.dtype.kind not in "fiu":
        raise _invalid("corners must be real numeric coordinates")
    corners = raw.astype(np.float64)
    if not np.isfinite(corners).all():
        raise _invalid("corners must be finite")
    q = 2 * (corners + 0.5) / np.array(original_size[::-1]) - 1
    if (np.abs(q) > 1).any():
        raise _invalid("manual corners must lie within the original physical image")
    edges = np.roll(q, -1, axis=0) - q
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if (cross <= 1e-6).any():
        raise _invalid("corners must form a nondegenerate convex clockwise image quad")
    flat = np.array(((-1, -1), (1, -1), (1, 1), (-1, 1)), dtype=np.float64)
    rows, values = [], []
    for (u, v), (x, y) in zip(flat, q, strict=True):
        rows.extend(((u, v, 1, 0, 0, 0, -x * u, -x * v),
                     (0, 0, 0, u, v, 1, -y * u, -y * v)))
        values.extend((x, y))
    try:
        matrix = np.append(np.linalg.solve(rows, values), 1).reshape(3, 3)
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise _invalid("manual corner homography is singular") from exc
    if 1 - np.abs(matrix[2, :2]).sum() <= 1e-6 or np.linalg.det(matrix) <= 1e-9:
        raise _invalid("manual corner homography must preserve orientation without poles")
    return _maps(output_size, photo_size, lambda p: _project(p, matrix),
                 lambda p: _project(p, inverse), {
                     "model": "manual_quad_planar_capture_v1", "corners_xy": corners.tolist(),
                     "original_size": list(original_size),
                     "annotation_model": (
                         "manual-corner planar approximation; not dense measured GT"
                     ),
                     "occlusion_model": "unobserved; quad does not establish visibility",
                     "folds": False,
                 })
