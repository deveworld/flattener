"""Score one rectified image against its flatbed scan with the DocUNet MATLAB protocol.

The functions here are the reusable core of the benchmark: they take in-memory
``uint8`` images, so training-time validation can call :func:`score_pair` directly.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from . import matlab

METRIC_AREA = 598_400
AD_SOLVERS = ("exact_lstsq", "lsqr_4iter")
# LSQR with the defaults of MATLAB ``lsqr(A, B)`` as called by PaperEdge/UVDoc
# ``alignLD.m``: its benchmark means are closest to MATLAB (the DocUNet calibration
# checks); per-image values carry solver noise (see ``lsqr_solve``).
DEFAULT_AD_SOLVER = "lsqr_4iter"
LSQR_STOP_REASONS = ("zero_rhs", "zero_normal", "relres", "normal_equations", "maxit")
AD_GRADIENT_THRESHOLD = 0.5
LSQR_TOLERANCE = 1e-6
_MAX_IMAGE_PIXELS = 200_000_000


class FlowBackend(Protocol):
    def flow(self, reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(vx, vy)`` from the reference (scan) to the prediction."""
        ...


def read_image(source: str | os.PathLike[str] | bytes) -> np.ndarray:
    """Decode an image as MATLAB's one-output ``imread`` (alpha over black or bKGD)."""
    data = source if isinstance(source, bytes) else None
    if data is None:
        with open(source, "rb") as stream:
            data = stream.read()
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > _MAX_IMAGE_PIXELS:
            raise ValueError(f"image is too large: {image.width}x{image.height}")
        image.load()
        return matlab.imread_rgb(image)


@dataclass(frozen=True)
class PreparedReference:
    """A scan converted to gray and resized to the DocUNet metric area."""

    image: np.ndarray
    gradient: np.ndarray | None
    source_shape: tuple[int, int]
    resizes: int


def _require_uint8_image(image: np.ndarray, name: str) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8, got {image.dtype}")
    if image.ndim == 3 and image.shape[2] == 3:
        return
    if image.ndim != 2:
        raise ValueError(f"{name} must be HW gray or HWx3 RGB, got {image.shape}")


def prepare_reference(reference: np.ndarray, *, resizes: int = 1) -> PreparedReference:
    """``rgb2gray`` then ``imresize(scan, sqrt(598400/(H*W)))``, repeated ``resizes`` times.

    ``resizes=2`` reproduces UVDoc ``evalScript.m``, which resizes the scan again
    before scoring the second view of a document.
    """
    _require_uint8_image(reference, "reference")
    if isinstance(resizes, bool) or not isinstance(resizes, int) or resizes < 1:
        raise ValueError("resizes must be a positive integer")
    gray = matlab.rgb2gray(reference)
    for _ in range(resizes):
        gray = matlab.imresize(gray, scale=matlab.metric_scale(*gray.shape, METRIC_AREA))
    gradient = matlab.imgradient(gray)
    peak = float(gradient.max())
    return PreparedReference(
        image=gray,
        gradient=gradient / peak if peak > 0 else None,
        source_shape=(int(reference.shape[0]), int(reference.shape[1])),
        resizes=resizes,
    )


def resize_prediction(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``imresize(rgb2gray(prediction), [rows cols])``."""
    _require_uint8_image(prediction, "prediction")
    return matlab.imresize(matlab.rgb2gray(prediction), output_shape=shape)


def local_distortion(vx: np.ndarray, vy: np.ndarray) -> float:
    """DocUNet LD: ``mean(sqrt(vx.^2 + vy.^2))`` over the SIFT-flow grid."""
    return float(np.mean(np.sqrt(vx * vx + vy * vy)))


@dataclass(frozen=True)
class LsqrResult:
    """LSQR solution, completed iterations and why it stopped (see LSQR_STOP_REASONS)."""

    solution: np.ndarray
    iterations: int
    stop: str


def lsqr_solve(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = LSQR_TOLERANCE,
    max_iterations=None,
) -> LsqrResult:
    """Paige-Saunders LSQR with MATLAB ``lsqr`` defaults (x0 = 0, maxit = min(n, 20)).

    Stops when ``relres <= tol`` or ``norm(A'r) / (normA * norm(r)) <= tol`` with the
    Frobenius-norm estimate, as MATLAB documents. PaperEdge ``alignLD.m`` calls
    ``lsqr(A, B)`` without options, which for four unknowns means at most four
    iterations.

    Implemented from MATLAB's documentation. On PaperEdge's outputs the benchmark
    means match MATLAB within the DocUNet calibration checks; single images differ by
    up to ~0.016 AD (45_1) even where the SIFT flow matches MATLAB, and ``exact_lstsq``
    is closer on most of them. There 129 of 130 systems stop at ``maxit`` after four
    iterations, so the result is a partially converged iterate: treat per-image AD as
    carrying solver noise.
    """
    rows, columns = matrix.shape
    if target.shape != (rows,):
        raise ValueError("target must be a vector matching the matrix rows")
    iterations = min(columns, 20) if max_iterations is None else int(max_iterations)
    solution = np.zeros(columns)
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        return LsqrResult(solution, 0, "zero_rhs")
    u = target / target_norm
    v = matrix.T @ u
    alpha = float(np.linalg.norm(v))
    if alpha == 0.0:
        return LsqrResult(solution, 0, "zero_normal")
    v = v / alpha
    phi_bar = target_norm
    rho_bar = alpha
    w = v.copy()
    norm_a2 = alpha * alpha
    completed = 0
    stop = "maxit"
    for completed in range(1, iterations + 1):
        u = matrix @ v - alpha * u
        beta = float(np.linalg.norm(u))
        if beta > 0.0:
            u = u / beta
        norm_a2 += beta * beta
        v = matrix.T @ u - beta * v
        alpha = float(np.linalg.norm(v))
        if alpha > 0.0:
            v = v / alpha
        norm_a2 += alpha * alpha
        rho = math.hypot(rho_bar, beta)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar
        solution = solution + (phi / rho) * w
        w = v - (theta / rho) * w
        residual = abs(phi_bar)
        if residual / target_norm <= tolerance or residual == 0.0:
            stop = "relres"
            break
        if alpha * abs(s * phi) / (math.sqrt(norm_a2) * residual) <= tolerance:
            stop = "normal_equations"
            break
    return LsqrResult(solution, completed, stop)


def lsqr_matlab(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = LSQR_TOLERANCE,
    max_iterations=None,
) -> tuple[np.ndarray, int]:
    """``(solution, iterations)`` from :func:`lsqr_solve`."""
    result = lsqr_solve(matrix, target, tolerance=tolerance, max_iterations=max_iterations)
    return result.solution, result.iterations


def _alignment_system(vx: np.ndarray, vy: np.ndarray, mask: np.ndarray):
    """Build ``alignLD.m``'s interleaved system in MATLAB column-major order."""
    height, width = vx.shape
    yy, xx = np.meshgrid(
        np.arange(1, height + 1, dtype=np.float64),
        np.arange(1, width + 1, dtype=np.float64),
        indexing="ij",
    )
    selected = mask.T
    source_x = xx.T[selected]
    source_y = yy.T[selected]
    moved_x = (xx + vx).T[selected]
    moved_y = (yy + vy).T[selected]
    count = source_x.size
    matrix = np.zeros((2 * count, 4))
    matrix[0::2, 0] = moved_x
    matrix[1::2, 1] = moved_y
    matrix[0::2, 2] = 1.0
    matrix[1::2, 3] = 1.0
    target = np.empty(2 * count)
    target[0::2] = source_x
    target[1::2] = source_y
    return matrix, target, xx, yy


def aligned_distortion_details(
    vx: np.ndarray, vy: np.ndarray, gradient: np.ndarray | None
) -> tuple[dict[str, float], LsqrResult]:
    """:func:`aligned_distortions` plus the LSQR run behind ``lsqr_4iter``."""
    if gradient is None:
        raise ValueError("AD is undefined for a scan without intensity gradient")
    weight = matlab.imresize(gradient, output_shape=vx.shape)
    mask = weight > AD_GRADIENT_THRESHOLD
    if int(mask.sum()) < 2:
        raise ValueError("AD needs at least two high-gradient scan pixels")
    matrix, target, xx, yy = _alignment_system(vx, vy, mask)
    lsqr = lsqr_solve(matrix, target)
    solutions = {
        "exact_lstsq": np.linalg.lstsq(matrix, target, rcond=None)[0],
        "lsqr_4iter": lsqr.solution,
    }
    results = {}
    for name, solution in solutions.items():
        residual_x = solution[0] * (xx + vx) + solution[2] - xx
        residual_y = solution[1] * (yy + vy) + solution[3] - yy
        weighted_x = weight * residual_x
        weighted_y = weight * residual_y
        results[name] = float(np.mean(np.sqrt(weighted_x * weighted_x + weighted_y * weighted_y)))
    return results, lsqr


def aligned_distortions(
    vx: np.ndarray, vy: np.ndarray, gradient: np.ndarray | None
) -> dict[str, float]:
    """PaperEdge AD (``evalAlignedUnwarp.m``) for every solver in :data:`AD_SOLVERS`.

    ``gradient`` is the scan's normalised Sobel magnitude at the metric size. It is
    resized to the flow grid, pixels above 0.5 fix a per-axis scale and shift, and
    the gradient-weighted residual magnitude (unsquared) is averaged.
    """
    return aligned_distortion_details(vx, vy, gradient)[0]


@dataclass(frozen=True)
class PairScore:
    """Per-image DocUNet metrics. ``ld``/``ad`` are ``None`` without a flow backend."""

    ms_ssim: float
    ld: float | None
    ad: float | None
    ad_exact_lstsq: float | None
    ad_lsqr_4iter: float | None
    ad_solver: str
    prediction_shape: tuple[int, int]
    reference_shape: tuple[int, int]
    metric_shape: tuple[int, int]
    flow_shape: tuple[int, int] | None
    reference_resizes: int
    ad_lsqr_iterations: int | None = None
    ad_lsqr_stop: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ad": self.ad,
            "ad_exact_lstsq": self.ad_exact_lstsq,
            "ad_lsqr_4iter": self.ad_lsqr_4iter,
            "ad_lsqr_iterations": self.ad_lsqr_iterations,
            "ad_lsqr_stop": self.ad_lsqr_stop,
            "ad_solver": self.ad_solver,
            "flow_shape_hw": None if self.flow_shape is None else list(self.flow_shape),
            "ld": self.ld,
            "metric_shape_hw": list(self.metric_shape),
            "ms_ssim": self.ms_ssim,
            "prediction_shape_hw": list(self.prediction_shape),
            "reference_resizes": self.reference_resizes,
            "reference_shape_hw": list(self.reference_shape),
        }


def _check_solver(ad_solver: str) -> None:
    if ad_solver not in AD_SOLVERS:
        raise ValueError(f"ad_solver must be one of {AD_SOLVERS}, got {ad_solver!r}")


def score_prepared(
    prediction: np.ndarray,
    reference: PreparedReference,
    *,
    siftflow: FlowBackend | None = None,
    ad_solver: str = DEFAULT_AD_SOLVER,
) -> PairScore:
    """Score a prediction against a :class:`PreparedReference`."""
    _check_solver(ad_solver)
    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be a PreparedReference")
    resized = resize_prediction(prediction, reference.image.shape)
    ms_ssim = matlab.ms_ssim(resized, reference.image)
    ld = ad = None
    ad_values: dict[str, float] = {}
    lsqr = None
    flow_shape = None
    if siftflow is not None:
        vx, vy = siftflow.flow(reference.image, resized)
        flow_shape = (int(vx.shape[0]), int(vx.shape[1]))
        ld = local_distortion(vx, vy)
        ad_values, lsqr = aligned_distortion_details(vx, vy, reference.gradient)
        ad = ad_values[ad_solver]
    return PairScore(
        ms_ssim=ms_ssim,
        ld=ld,
        ad=ad,
        ad_exact_lstsq=ad_values.get("exact_lstsq"),
        ad_lsqr_4iter=ad_values.get("lsqr_4iter"),
        ad_solver=ad_solver,
        prediction_shape=(int(prediction.shape[0]), int(prediction.shape[1])),
        reference_shape=reference.source_shape,
        metric_shape=(int(reference.image.shape[0]), int(reference.image.shape[1])),
        flow_shape=flow_shape,
        reference_resizes=reference.resizes,
        ad_lsqr_iterations=None if lsqr is None else lsqr.iterations,
        ad_lsqr_stop=None if lsqr is None else lsqr.stop,
    )


def score_pair(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    siftflow: FlowBackend | None = None,
    ad_solver: str = DEFAULT_AD_SOLVER,
    reference_resizes: int = 1,
) -> PairScore:
    """Score one rectified image against its scan (both ``uint8`` RGB or gray arrays).

    Decode files with :func:`read_image` so RGBA scans are composited over black as
    MATLAB does. Without ``siftflow`` only MS-SSIM is computed.
    """
    _check_solver(ad_solver)
    prepared = prepare_reference(reference, resizes=reference_resizes)
    return score_prepared(prediction, prepared, siftflow=siftflow, ad_solver=ad_solver)
