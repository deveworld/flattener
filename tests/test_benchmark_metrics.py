"""DocUNet metric core: LSQR emulation, AD alignment and pair scoring on tiny inputs."""

import os
from pathlib import Path

import numpy as np
import pytest

from flattener.benchmark import matlab, metrics


def _krylov_solution(matrix: np.ndarray, target: np.ndarray, iterations: int) -> np.ndarray:
    """Independent LSQR oracle: least squares over the Krylov space K_k(A'A, A'b)."""
    basis = [matrix.T @ target]
    for _ in range(iterations - 1):
        basis.append(matrix.T @ (matrix @ basis[-1]))
    space = np.stack(basis, axis=1)
    space, _ = np.linalg.qr(space)
    coefficients = np.linalg.lstsq(matrix @ space, target, rcond=None)[0]
    return space @ coefficients


def test_lsqr_iterates_match_the_krylov_least_squares_oracle():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 4)) * np.array([1.0, 3.0, 0.5, 2.0])
    target = rng.normal(size=40)
    for iterations in (1, 2, 3):
        solution, used = metrics.lsqr_matlab(matrix, target, max_iterations=iterations)
        assert used == iterations
        assert np.allclose(solution, _krylov_solution(matrix, target, iterations), atol=1e-9)
    first, _ = metrics.lsqr_matlab(matrix, target, max_iterations=1)
    direction = matrix.T @ target
    step = direction @ direction / np.sum((matrix @ direction) ** 2)
    assert np.allclose(first, step * direction, atol=1e-12)
    default, used = metrics.lsqr_matlab(matrix, target)
    assert used <= 4
    assert np.allclose(default, np.linalg.lstsq(matrix, target, rcond=None)[0], atol=1e-8)


def _affine_flow(shape, scale_x, scale_y, shift_x, shift_y):
    height, width = shape
    yy, xx = np.meshgrid(np.arange(1.0, height + 1), np.arange(1.0, width + 1), indexing="ij")
    # a * (x + vx) + c = x  ->  vx = (x - c) / a - x
    vx = (xx - shift_x) / scale_x - xx
    vy = (yy - shift_y) / scale_y - yy
    return vx, vy


def test_ad_is_zero_for_a_pure_per_axis_scale_and_shift_flow():
    shape = (12, 10)
    vx, vy = _affine_flow(shape, 1.05, 0.97, 1.5, -2.0)
    gradient = np.random.default_rng(1).random(shape)
    values = metrics.aligned_distortions(vx, vy, gradient)
    assert set(values) == set(metrics.AD_SOLVERS)
    assert values["exact_lstsq"] == pytest.approx(0.0, abs=1e-10)
    assert values["lsqr_4iter"] == pytest.approx(0.0, abs=1e-5)


def test_ad_matches_an_independent_weighted_residual_oracle():
    shape = (9, 11)
    rng = np.random.default_rng(2)
    vx, vy = _affine_flow(shape, 1.1, 0.9, 0.5, 1.0)
    vx = vx + rng.normal(scale=0.7, size=shape)
    vy = vy + rng.normal(scale=0.7, size=shape)
    gradient = rng.random(shape)
    weight = matlab.imresize(gradient, output_shape=shape)
    mask = weight > 0.5
    height, width = shape
    yy, xx = np.meshgrid(np.arange(1.0, height + 1), np.arange(1.0, width + 1), indexing="ij")
    # Axes decouple: x' = a*(x+vx)+c and y' = b*(y+vy)+d are independent problems.
    ax = np.stack([(xx + vx)[mask], np.ones(mask.sum())], axis=1)
    ay = np.stack([(yy + vy)[mask], np.ones(mask.sum())], axis=1)
    a, c = np.linalg.lstsq(ax, xx[mask], rcond=None)[0]
    b, d = np.linalg.lstsq(ay, yy[mask], rcond=None)[0]
    residual = np.hypot(weight * (a * (xx + vx) + c - xx), weight * (b * (yy + vy) + d - yy))
    values = metrics.aligned_distortions(vx, vy, gradient)
    assert values["exact_lstsq"] == pytest.approx(residual.mean(), abs=1e-10)
    assert values["lsqr_4iter"] == pytest.approx(residual.mean(), abs=1e-3)
    with pytest.raises(ValueError):
        metrics.aligned_distortions(vx, vy, None)
    with pytest.raises(ValueError):
        metrics.aligned_distortions(vx, vy, np.zeros(shape))


def test_local_distortion_is_mean_flow_magnitude():
    vx = np.array([[3.0, 0.0], [0.0, 1.0]])
    vy = np.array([[4.0, 0.0], [2.0, 0.0]])
    assert metrics.local_distortion(vx, vy) == pytest.approx((5.0 + 0.0 + 2.0 + 1.0) / 4)


def test_prepare_reference_converts_to_gray_before_resizing_and_can_resize_twice(monkeypatch):
    monkeypatch.setattr(metrics, "METRIC_AREA", 120)
    rng = np.random.default_rng(3)
    scan = rng.integers(0, 256, size=(20, 14, 3), dtype=np.uint8)
    prepared = metrics.prepare_reference(scan)
    gray = matlab.rgb2gray(scan)
    scale = matlab.metric_scale(20, 14, 120)
    assert np.array_equal(prepared.image, matlab.imresize(gray, scale=scale))
    assert prepared.source_shape == (20, 14)
    assert prepared.gradient.max() == pytest.approx(1.0)
    twice = metrics.prepare_reference(scan, resizes=2)
    again = matlab.imresize(prepared.image, scale=matlab.metric_scale(*prepared.image.shape, 120))
    assert np.array_equal(twice.image, again)
    with pytest.raises(ValueError):
        metrics.prepare_reference(scan, resizes=0)


class _ConstantFlow:
    def __init__(self, dx: float, dy: float) -> None:
        self.dx, self.dy = dx, dy
        self.calls = []

    def flow(self, reference, prediction):
        self.calls.append((reference.shape, prediction.shape))
        half = ((reference.shape[0] + 1) // 2, (reference.shape[1] + 1) // 2)
        return np.full(half, self.dx), np.full(half, self.dy)


def test_score_pair_resizes_prediction_to_scan_metric_size_and_uses_flow_backend(monkeypatch):
    monkeypatch.setattr(metrics, "METRIC_AREA", 600)
    rng = np.random.default_rng(4)
    scan = rng.integers(0, 256, size=(40, 30), dtype=np.uint8)
    prediction = rng.integers(0, 256, size=(33, 21, 3), dtype=np.uint8)
    backend = _ConstantFlow(3.0, 4.0)
    score = metrics.score_pair(prediction, scan, siftflow=backend)
    assert score.metric_shape == backend.calls[0][0] == backend.calls[0][1]
    assert score.ld == pytest.approx(5.0)
    assert score.ad == score.ad_lsqr_4iter
    assert score.ad_solver == metrics.DEFAULT_AD_SOLVER == "lsqr_4iter"
    # A constant flow is a pure shift: AD removes it.
    assert score.ad_exact_lstsq == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < score.ms_ssim < 1.0
    plain = metrics.score_pair(prediction, scan)
    assert plain.ld is None and plain.ad is None and plain.ms_ssim == score.ms_ssim
    record = score.to_dict()
    assert record["metric_shape_hw"] == list(score.metric_shape)
    with pytest.raises(ValueError):
        metrics.score_pair(prediction, scan, ad_solver="svd")


def test_lsqr_solve_reports_iterations_and_stop_reason():
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(30, 4))
    target = rng.normal(size=30)
    capped = metrics.lsqr_solve(matrix, target, max_iterations=1)
    assert (capped.iterations, capped.stop) == (1, "maxit")
    consistent = metrics.lsqr_solve(matrix, matrix @ np.array([1.0, -2.0, 0.5, 3.0]))
    assert consistent.stop == "relres" and consistent.iterations <= 4
    default = metrics.lsqr_solve(matrix, target)
    assert default.stop in ("normal_equations", "maxit") and default.iterations <= 4
    assert metrics.lsqr_solve(matrix, np.zeros(30)).stop == "zero_rhs"
    assert metrics.lsqr_solve(np.zeros((30, 4)), target).stop == "zero_normal"
    solution, used = metrics.lsqr_matlab(matrix, target)
    assert np.array_equal(solution, default.solution) and used == default.iterations
    assert {capped.stop, consistent.stop, default.stop} <= set(metrics.LSQR_STOP_REASONS)


def test_score_rows_record_the_lsqr_run(monkeypatch):
    monkeypatch.setattr(metrics, "METRIC_AREA", 600)
    rng = np.random.default_rng(6)
    scan = rng.integers(0, 256, size=(40, 30), dtype=np.uint8)
    prediction = rng.integers(0, 256, size=(33, 21), dtype=np.uint8)
    record = metrics.score_pair(prediction, scan, siftflow=_ConstantFlow(1.0, -2.0)).to_dict()
    assert isinstance(record["ad_lsqr_iterations"], int) and 1 <= record["ad_lsqr_iterations"] <= 4
    assert record["ad_lsqr_stop"] in metrics.LSQR_STOP_REASONS
    plain = metrics.score_pair(prediction, scan).to_dict()
    assert plain["ad_lsqr_iterations"] is None and plain["ad_lsqr_stop"] is None


PAPEREDGE_ENV = "FLATTENER_BENCH_PAPEREDGE_DIR"
REFERENCES_ENV = "FLATTENER_BENCH_REFERENCES_DIR"


@pytest.mark.skipif(
    not os.environ.get(PAPEREDGE_ENV),
    reason=f"set {PAPEREDGE_ENV} to PaperEdge res_test/ (with adres.txt) to compare with MATLAB",
)
def test_ms_ssim_matches_matlab_adres_on_two_paperedge_images():
    """MS-SSIM only (no SIFT flow): two small scans against PaperEdge's MATLAB values."""
    from flattener.benchmark import calibration, docunet

    outputs = Path(os.environ[PAPEREDGE_ENV])
    references = Path(os.environ.get(REFERENCES_ENV, docunet.DEFAULT_REFERENCES_DIR))
    adres = calibration.parse_adres(outputs / calibration.ADRES_NAME)["rows"]
    for k, m in ((1, 1), (5, 1)):
        scan = references / docunet.DEFAULT_REFERENCE_PATTERN.format(k=k)
        if not scan.is_file():
            pytest.skip(f"DocUNet scan {scan} not found; set {REFERENCES_ENV}")
        score = metrics.score_pair(
            metrics.read_image(outputs / f"{k}_{m}.png"), metrics.read_image(scan)
        )
        assert score.ms_ssim == pytest.approx(adres[f"{k}_{m}"]["ms_ssim"], abs=2e-5)
