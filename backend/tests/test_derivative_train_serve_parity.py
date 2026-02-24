from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.signals.statistical.derivative import DerivativeSignal
from src.models.vacuum_pressure.runtime_model import DerivativeRuntime, DerivativeRuntimeParams


def test_derivative_batch_runtime_score_parity() -> None:
    rng = np.random.default_rng(seed=20260220)

    n_bins = 160
    n_ticks = 101
    k_values = np.arange(-50, 51, dtype=np.int32)
    state5 = rng.integers(-2, 3, size=(n_bins, n_ticks), dtype=np.int8)

    params: dict[str, float | int] = {
        "center_exclusion_radius": 1,
        "spatial_decay_power": 0.35,
        "zscore_window_bins": 32,
        "zscore_min_periods": 8,
        "tanh_scale": 2.75,
        "d1_weight": 1.0,
        "d2_weight": 0.5,
        "d3_weight": 0.25,
        "bull_pressure_weight": 1.2,
        "bull_vacuum_weight": 0.8,
        "bear_pressure_weight": 1.1,
        "bear_vacuum_weight": 0.9,
        "mixed_weight": 0.2,
    }
    cell_width_ms = 100

    batch_signal = DerivativeSignal(cell_width_ms=cell_width_ms, **params)
    dataset = {
        "n_bins": n_bins,
        "k_values": k_values,
        "state5_code": state5.astype(np.float64),
        "microstate_id": np.full((n_bins, n_ticks), 4.0, dtype=np.float64),
    }
    batch_scores = batch_signal.compute(dataset).signal

    runtime = DerivativeRuntime(
        k_values=k_values,
        cell_width_ms=cell_width_ms,
        params=DerivativeRuntimeParams(**params),
    )
    runtime_scores = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        runtime_scores[i] = runtime.update(state5[i]).score

    np.testing.assert_allclose(runtime_scores, batch_scores, atol=1e-12, rtol=0.0)
