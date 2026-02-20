from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.runtime_model import (  # noqa: E402
    DerivativeRuntime,
    DerivativeRuntimeParams,
)


def test_runtime_zero_mixed_is_stable() -> None:
    runtime = DerivativeRuntime(
        k_values=np.arange(-2, 3, dtype=np.int32),
        cell_width_ms=100,
        params=DerivativeRuntimeParams(
            zscore_window_bins=8,
            zscore_min_periods=2,
        ),
    )
    out = runtime.update(np.zeros(5, dtype=np.int8))
    assert out.name == "derivative"
    assert out.score == pytest.approx(0.0)
    assert out.base == pytest.approx(0.0)
    assert out.ready is False
    assert out.sample_count == 1


def test_runtime_detects_bullish_regime_shift() -> None:
    runtime = DerivativeRuntime(
        k_values=np.arange(-3, 4, dtype=np.int32),
        cell_width_ms=100,
        params=DerivativeRuntimeParams(
            zscore_window_bins=6,
            zscore_min_periods=2,
            d1_weight=1.0,
            d2_weight=0.0,
            d3_weight=0.0,
        ),
    )

    # 3 bins mixed (0), then shift to bullish structure:
    # below spot pressure (1), above spot vacuum (2), center mixed (0).
    mixed = np.zeros(7, dtype=np.int8)
    bullish = np.array([1, 1, 1, 0, 2, 2, 2], dtype=np.int8)

    runtime.update(mixed)
    runtime.update(mixed)
    runtime.update(mixed)
    out = runtime.update(bullish)

    assert out.ready is True
    assert out.score > 0.0
    assert out.bull_intensity > out.bear_intensity


def test_runtime_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="zscore_min_periods cannot exceed"):
        DerivativeRuntime(
            k_values=np.arange(-2, 3, dtype=np.int32),
            cell_width_ms=100,
            params=DerivativeRuntimeParams(
                zscore_window_bins=10,
                zscore_min_periods=12,
            ),
        )

