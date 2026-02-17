from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import scripts.analyze_vp_signals as analyze_vp_signals
from src.vacuum_pressure.spectrum import IndependentCellSpectrum, ProjectionModelConfig


def _kernel(projection_model: ProjectionModelConfig) -> IndependentCellSpectrum:
    return IndependentCellSpectrum(
        n_cells=1,
        windows=[2, 4],
        rollup_weights=[1.0, 1.0],
        derivative_weights=[0.55, 0.30, 0.15],
        tanh_scale=3.0,
        neutral_threshold=0.15,
        zscore_window_bins=8,
        zscore_min_periods=2,
        projection_horizons_ms=[100],
        default_dt_s=0.1,
        projection_model=projection_model,
    )


def test_projection_quadratic_path_matches_formula() -> None:
    kernel = _kernel(
        ProjectionModelConfig(use_cubic=False, cubic_scale=1.0 / 6.0, damping_lambda=0.0)
    )
    score = np.array([0.2], dtype=np.float64)
    score_d1 = np.array([0.5], dtype=np.float64)
    score_d2 = np.array([-0.1], dtype=np.float64)
    score_d3 = np.array([0.3], dtype=np.float64)
    horizon_s = 2.5

    projected = kernel._project_score_horizon(
        score=score,
        score_d1=score_d1,
        score_d2=score_d2,
        score_d3=score_d3,
        horizon_s=horizon_s,
    )
    expected = np.clip(score + score_d1 * horizon_s + 0.5 * score_d2 * horizon_s**2, -1.0, 1.0)
    np.testing.assert_allclose(projected, expected)


def test_projection_cubic_damped_path_matches_formula() -> None:
    cubic_scale = 1.0 / 6.0
    damping_lambda = 0.001
    kernel = _kernel(
        ProjectionModelConfig(
            use_cubic=True,
            cubic_scale=cubic_scale,
            damping_lambda=damping_lambda,
        )
    )
    score = np.array([0.2], dtype=np.float64)
    score_d1 = np.array([0.5], dtype=np.float64)
    score_d2 = np.array([-0.1], dtype=np.float64)
    score_d3 = np.array([0.3], dtype=np.float64)
    horizon_s = 2.5

    projected = kernel._project_score_horizon(
        score=score,
        score_d1=score_d1,
        score_d2=score_d2,
        score_d3=score_d3,
        horizon_s=horizon_s,
    )

    raw = (
        score
        + score_d1 * horizon_s
        + 0.5 * score_d2 * horizon_s**2
        + cubic_scale * score_d3 * horizon_s**3
    )
    expected = np.clip(raw * np.exp(-damping_lambda * horizon_s), -1.0, 1.0)
    np.testing.assert_allclose(projected, expected)


def test_rolling_slope_recovers_linear_trend() -> None:
    x = np.arange(50, dtype=np.float64)
    y = 2.0 * x + 1.0
    slope = analyze_vp_signals._rolling_slope(y, window=10)
    assert np.allclose(slope[9:], 2.0, atol=1e-12)


def test_projection_regime_shift_metrics_emit_finite_values() -> None:
    n = 120
    ts_ns = np.arange(n, dtype=np.int64) * 100_000_000
    base = np.linspace(-0.4, 0.7, n, dtype=np.float64)
    wave = 0.05 * np.sin(np.linspace(0.0, 8.0 * np.pi, n, dtype=np.float64))
    projected = np.stack([base + wave, base, base - wave], axis=1)

    metrics = analyze_vp_signals._projection_regime_shift_metrics(
        ts_ns=ts_ns,
        projected_score=projected,
        slope_window_bins=20,
        shift_z_threshold=1.5,
    )

    assert metrics["n_valid_slope_bins"] > 0.0
    assert np.isfinite(metrics["mean_abs_d1"])
    assert np.isfinite(metrics["mean_abs_d2"])
    assert np.isfinite(metrics["mean_abs_d3"])
    assert np.isfinite(metrics["regime_shift_rate"])
