"""Page-region decisions and exact pixel-edge coordinate composition.

Probabilities are uncalibrated model scores. Coverage is measured against the
estimated page mask.
"""

from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np

from ..contracts import FlattenerError, Status


def _input(message):
    return FlattenerError(Status.INPUT_ERROR, "invalid_region_input", message)


def _rejected(reason, message, metadata=None):
    return FlattenerError(Status.MODEL_ERROR, reason, message, metadata=metadata)


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _source_size(size):
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in size)
    ):
        raise _input("source_size must be positive (height,width) within renderer defaults")
    # Convert before multiplication: narrow NumPy integer dtypes can overflow
    # and otherwise incorrectly admit an image above the pixel resource limit.
    size = tuple(int(v) for v in size)
    if min(size) < 1 or max(size) > 8192 or size[0] * size[1] > 16_000_000:
        raise _input("source_size must be positive (height,width) within renderer defaults")
    return size


@dataclass(frozen=True, slots=True)
class PageRegion:
    """Pixel-edge rectangle: the complete W×H image is (0,0,W,H).

    A virtual rectangle may extend outside the source. Callers providing a
    detected/manual source region must additionally validate source bounds.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        if not all(_finite(v) for v in (self.left, self.top, self.right, self.bottom)):
            raise _input("page-region edges must be finite real values")
        for name in ("left", "top", "right", "bottom"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not all(isfinite(v) and v > 0 for v in (self.width, self.height)):
            raise _input("page region must have finite positive width and height")

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class RegionConfig:
    threshold: float = 0.5
    min_area_fraction: float = 0.01
    max_area_fraction: float = 0.95
    min_mean_probability: float = 0.70
    min_dominance: float = 0.85
    padding_fraction: float = 0.04
    canonical_occupancy: float = 0.68
    required_coverage: float = 0.98
    max_boundary_expansion: float = 1.20

    def __post_init__(self):
        limits = {
            "threshold": (0.01, 0.99),
            "min_area_fraction": (0.0, 1.0),
            "max_area_fraction": (0.0, 1.0),
            "min_mean_probability": (0.0, 1.0),
            "min_dominance": (0.5, 1.0),
            "padding_fraction": (0.0, 0.25),
            "canonical_occupancy": (0.1, 1.0),
            "required_coverage": (0.5, 1.0),
            "max_boundary_expansion": (1.0, 1.5),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if not _finite(value) or not low <= value <= high:
                raise _input(f"{name} must be finite and in [{low},{high}]")
            object.__setattr__(self, name, float(value))
        if not 0 < self.min_area_fraction < self.max_area_fraction <= 1:
            raise _input("region area limits must satisfy 0 < minimum < maximum <= 1")


def _config(config):
    if config is None:
        return RegionConfig()
    if not isinstance(config, RegionConfig):
        raise _input("config must be a RegionConfig")
    return config


@dataclass(slots=True)
class RegionEstimate:
    region: PageRegion
    mask: np.ndarray
    metadata: dict


def _largest_component(binary):
    """Return the largest 8-connected component; ties keep the first in row-major order."""
    height, width = binary.shape
    visited = np.zeros_like(binary)
    largest = []
    sizes = []
    for y, x in zip(*np.nonzero(binary), strict=True):
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = deque([(int(y), int(x))])
        component = []
        while queue:
            yy, xx = queue.popleft()
            component.append((yy, xx))
            for ny in range(max(0, yy - 1), min(height, yy + 2)):
                for nx in range(max(0, xx - 1), min(width, xx + 2)):
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        sizes.append(len(component))
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(binary)
    if largest:
        points = np.asarray(largest)
        result[points[:, 0], points[:, 1]] = True
    return result, sorted(sizes, reverse=True)


def detect_region(probability, source_size, *, config=None):
    """Select one supported component, retaining explicit rejection diagnostics."""
    config = _config(config)
    height, width = _source_size(source_size)
    if (
        not isinstance(probability, np.ndarray)
        or probability.ndim != 2
        or probability.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or probability.size == 0
        or probability.size > 512 * 512
        or not np.isfinite(probability).all()
        or np.any(probability < 0)
        or np.any(probability > 1)
    ):
        raise _input("probability must be finite float32/64 H×W in [0,1], up to 262144 pixels")
    binary = probability >= config.threshold
    mask, sizes = _largest_component(binary)
    count = int(mask.sum())
    total = int(binary.sum())
    area = count / mask.size
    dominance = count / total if total else 0.0
    mean = float(probability[mask].mean()) if count else 0.0
    metadata = {
        "region_config": asdict(config),
        "probability_grid_size": list(mask.shape),
        "component_sizes": sizes,
        "selected_pixels": count,
        "foreground_pixels": total,
        "area_fraction": area,
        "dominance": dominance,
        "mean_probability": mean,
        "score_is_calibrated_confidence": False,
    }
    if not count:
        raise _rejected("page_not_found", "no page-region support above threshold", metadata)
    if not config.min_area_fraction <= area <= config.max_area_fraction:
        raise _rejected("page_area_rejected", "page-region area outside accepted range", metadata)
    if dominance < config.min_dominance:
        raise _rejected("ambiguous_page_region", "multiple substantial page components", metadata)
    if mean < config.min_mean_probability:
        raise _rejected("uncertain_page_region", "page-region scores too weak", metadata)
    yy, xx = np.nonzero(mask)
    left, right = (
        float(xx.min()) / mask.shape[1] * width,
        (float(xx.max()) + 1) / mask.shape[1] * width,
    )
    top, bottom = (
        float(yy.min()) / mask.shape[0] * height,
        (float(yy.max()) + 1) / mask.shape[0] * height,
    )
    px, py = (right - left) * config.padding_fraction, (bottom - top) * config.padding_fraction
    region = PageRegion(
        max(0, left - px), max(0, top - py), min(width, right + px), min(height, bottom + py)
    )
    metadata.update(region=asdict(region), unpadded_region=[left, top, right, bottom])
    return RegionEstimate(region, mask, metadata)


def validate_region(region, source_size):
    height, width = _source_size(source_size)
    if not isinstance(region, PageRegion):
        raise _input("region must be a PageRegion")
    if region.left < 0 or region.top < 0 or region.right > width or region.bottom > height:
        raise _input("source page region must lie within the image's pixel edges")
    return region


def virtual_page_box(region, occupancy):
    if not isinstance(region, PageRegion) or not _finite(occupancy) or not 0.1 <= occupancy <= 1:
        raise _input("region and occupancy in [0.1,1] are required")
    cx, cy = (region.left + region.right) / 2, (region.top + region.bottom) / 2
    half_w, half_h = region.width / (2 * occupancy), region.height / (2 * occupancy)
    return PageRegion(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _map(grid):
    if (
        not isinstance(grid, np.ndarray)
        or grid.dtype != np.float32
        or grid.ndim != 3
        or grid.shape[-1] != 2
        or min(grid.shape[:2]) < 2
        or grid.shape[0] * grid.shape[1] > 16_000_000
        or not np.isfinite(grid).all()
    ):
        raise _input("map must be finite float32 H×W×2 with sides >=2 within 16MP")
    return grid


def region_map_to_source(grid, virtual_box, source_size):
    """Compose virtual-preview coordinates into original normalized coordinates."""
    _map(grid)
    height, width = _source_size(source_size)
    if not isinstance(virtual_box, PageRegion):
        raise _input("virtual_box must be a PageRegion")
    # Float64 arithmetic avoids an extra large-source pixel of cancellation.
    result = grid.astype(np.float64)
    result[..., 0] = (
        2 * (virtual_box.left + (result[..., 0] + 1) * virtual_box.width / 2) / width - 1
    )
    result[..., 1] = (
        2 * (virtual_box.top + (result[..., 1] + 1) * virtual_box.height / 2) / height - 1
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise _input("coordinate composition overflowed")
    return result


def page_coverage(grid, mask):
    """Return the fraction of page-mask centers inside the mapped output boundary.

    Scanlines run through pixel centers of the probability grid. Fold checks
    belong to the renderer.
    """
    _map(grid)
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != bool
        or mask.ndim != 2
        or mask.size > 512 * 512
        or not mask.any()
    ):
        raise _input("page mask must be nonempty bool H×W, up to 262144 pixels")
    boundary = np.concatenate((grid[0], grid[1:, -1], grid[-1, -2::-1], grid[-2:0:-1, 0]))
    polygon = boundary.astype(np.float64)
    following = np.roll(polygon, -1, axis=0)
    x0, y0 = polygon.T
    x1, y1 = following.T
    # Float32 boundary centers can be a few ULP off their exact values; this
    # tolerance stays below 0.00013 grid pixels at the maximum 512 width.
    epsilon = 4 * np.finfo(np.float32).eps
    x_centers = 2 * (np.arange(mask.shape[1], dtype=np.float64) + 0.5) / mask.shape[1] - 1
    horizontal = np.abs(y1 - y0) <= epsilon
    covered_count = 0
    for row in np.flatnonzero(mask.any(axis=1)):
        y = 2 * (row + 0.5) / mask.shape[0] - 1
        crossing = (y0 > y) != (y1 > y)
        # Crossings exclude horizontal edges, so there is no zero divisor.
        intersections = x0[crossing] + (y - y0[crossing]) * (x1[crossing] - x0[crossing]) / (
            y1[crossing] - y0[crossing]
        )
        intersections.sort()
        index = np.searchsorted(intersections, x_centers, side="right")
        inside = (index % 2) == 1
        if intersections.size:
            left = intersections[np.clip(index - 1, 0, len(intersections) - 1)]
            right = intersections[np.clip(index, 0, len(intersections) - 1)]
            inside |= (np.abs(x_centers - left) <= epsilon) | (np.abs(x_centers - right) <= epsilon)
        # Include boundary centers on horizontal edges and exact extrema.
        on_row = horizontal & (np.abs(y - y0) <= epsilon)
        for edge in np.flatnonzero(on_row):
            inside |= (x_centers >= min(x0[edge], x1[edge]) - epsilon) & (
                x_centers <= max(x0[edge], x1[edge]) + epsilon
            )
        # A non-horizontal local maximum has no half-open scanline crossing.
        # Its vertex itself is nevertheless part of the closed boundary.
        for vertex in np.flatnonzero(np.abs(y - y0) <= epsilon):
            inside |= np.abs(x_centers - x0[vertex]) <= epsilon
        covered_count += int(np.count_nonzero(inside & mask[row]))
    return float(covered_count / np.count_nonzero(mask))


def ensure_page_coverage(grid, mask, *, config=None):
    """Conservatively expand an estimated footprint, or explicitly reject it."""
    config = _config(config)
    _map(grid)
    initial = page_coverage(grid, mask)
    metadata = {
        "initial_estimated_page_coverage": initial,
        "required_estimated_page_coverage": config.required_coverage,
        "max_boundary_expansion": config.max_boundary_expansion,
        "coverage_grid_size": list(mask.shape),
        "coverage_is_ground_truth": False,
    }
    if initial >= config.required_coverage:
        metadata.update(final_estimated_page_coverage=initial, boundary_expansion=1.0)
        return grid.copy(), metadata
    center = grid.astype(np.float64).mean(axis=(0, 1))

    def expanded(factor):
        return (center + (grid.astype(np.float64) - center) * factor).astype(np.float32)

    candidate = expanded(config.max_boundary_expansion)
    final = page_coverage(candidate, mask)
    if final < config.required_coverage:
        metadata.update(
            final_estimated_page_coverage=final, boundary_expansion=config.max_boundary_expansion
        )
        raise _rejected(
            "page_coverage_insufficient",
            "estimated page extends beyond predicted footprint",
            metadata,
        )
    low, high = 1.0, config.max_boundary_expansion
    # Bisect for an accepted scale and check every probe, since coverage of a
    # non-convex footprint can be non-monotonic in scale.
    for _ in range(8):
        mid = (low + high) / 2
        probe = expanded(mid)
        coverage = page_coverage(probe, mask)
        if coverage >= config.required_coverage:
            high, candidate, final = mid, probe, coverage
        else:
            low = mid
    metadata.update(final_estimated_page_coverage=final, boundary_expansion=high)
    return candidate, metadata
