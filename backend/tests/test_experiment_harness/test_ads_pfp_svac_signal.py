from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.signals.statistical.ads_pfp_svac import (  # noqa: E402
    ADSPFPSVacSignal,
)


def _empty_dataset(n_bins: int = 64) -> dict[str, np.ndarray | int]:
    grid = np.zeros((n_bins, 101), dtype=np.float64)
    return {
        "n_bins": n_bins,
        "v_add": grid.copy(),
        "v_fill": grid.copy(),
        "v_pull": grid.copy(),
        "vacuum_variant": grid.copy(),
    }


def test_ads_pfp_svac_zero_input_stays_zero_and_uses_default_thresholds() -> None:
    signal = ADSPFPSVacSignal()
    result = signal.compute(_empty_dataset())
    assert result.signal.shape == (64,)
    assert np.allclose(result.signal, 0.0)
    assert result.metadata["adaptive_thresholds"] == signal.default_thresholds()


def test_ads_pfp_svac_rejects_mismatched_ads_weight_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ADSPFPSVacSignal(
            ads_slope_windows_ms=[1000, 2500, 5000],
            ads_blend_weights=[0.5, 0.5],
        )


def test_ads_pfp_svac_rejects_invalid_grid_shape() -> None:
    signal = ADSPFPSVacSignal()
    dataset = _empty_dataset()
    dataset["v_add"] = np.zeros((32, 99), dtype=np.float64)
    with pytest.raises(ValueError, match="must have 101 columns"):
        signal.compute(dataset)

