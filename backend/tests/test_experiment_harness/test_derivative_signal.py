from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.signals.statistical.derivative import (  # noqa: E402
    DerivativeSignal,
)


def _dataset_from_state_code(state_code: int, n_bins: int = 64) -> dict[str, np.ndarray | int]:
    state_grid = np.full((n_bins, 101), float(state_code), dtype=np.float64)
    micro_grid = np.full((n_bins, 101), 4.0, dtype=np.float64)
    return {
        "n_bins": n_bins,
        "k_values": np.arange(-50, 51, dtype=np.int32),
        "state5_code": state_grid,
        "microstate_id": micro_grid,
    }


def test_derivative_rejects_invalid_grid_shape() -> None:
    signal = DerivativeSignal(zscore_window_bins=8, zscore_min_periods=2)
    dataset = _dataset_from_state_code(0, n_bins=32)
    dataset["state5_code"] = np.zeros((32, 99), dtype=np.float64)
    with pytest.raises(ValueError, match="must have 101 columns"):
        signal.compute(dataset)


def test_derivative_zero_state_is_stable() -> None:
    signal = DerivativeSignal(
        zscore_window_bins=8,
        zscore_min_periods=2,
        d1_weight=1.0,
        d2_weight=0.0,
        d3_weight=0.0,
    )
    result = signal.compute(_dataset_from_state_code(0))
    assert result.signal.shape == (64,)
    assert np.allclose(result.signal, 0.0)
    assert "state5_distribution" in result.metadata
    assert "micro9_distribution" in result.metadata
    assert "state5_transition_matrix" in result.metadata


def test_derivative_detects_regime_shift_direction() -> None:
    n_bins = 64
    state = np.zeros((n_bins, 101), dtype=np.float64)
    # First half mixed, second half bullish (above spot vacuum, below spot pressure).
    state[:32, :] = 0.0
    state[32:, :50] = 1.0
    state[32:, 51:] = 2.0
    state[32:, 50] = 0.0

    dataset = {
        "n_bins": n_bins,
        "k_values": np.arange(-50, 51, dtype=np.int32),
        "state5_code": state,
        "microstate_id": np.full((n_bins, 101), 8.0, dtype=np.float64),
    }
    signal = DerivativeSignal(
        zscore_window_bins=12,
        zscore_min_periods=4,
        d1_weight=1.0,
        d2_weight=0.0,
        d3_weight=0.0,
    )
    result = signal.compute(dataset)

    assert np.nanmax(result.signal) > 0.0
    assert np.isfinite(result.signal).all()

