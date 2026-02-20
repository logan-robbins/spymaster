from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.serving_config import (
    ProjectionConfig,
    ScoringConfig,
    ServingSpec,
    SignalConfig,
)


def test_serving_spec_runtime_fields_maps_derivative_and_projection() -> None:
    spec = ServingSpec(
        name="derivative_baseline",
        pipeline="mnq_60m_baseline",
        scoring=ScoringConfig(
            zscore_window_bins=180,
            zscore_min_periods=60,
            derivative_weights=[0.6, 0.3, 0.1],
            tanh_scale=2.5,
            threshold_neutral=0.2,
        ),
        signal=SignalConfig(
            name="derivative",
            params={
                "zscore_window_bins": 240,
                "zscore_min_periods": 80,
                "d1_weight": 0.9,
                "d2_weight": 0.1,
                "d3_weight": 0.0,
            },
            weights={
                "bull_pressure": 1.1,
                "bull_vacuum": 1.2,
                "bear_pressure": 0.9,
                "bear_vacuum": 1.0,
                "mixed": 0.2,
            },
        ),
        projection=ProjectionConfig(
            horizons_ms=[100, 300, 500],
            use_cubic=True,
            cubic_scale=1.0 / 6.0,
            damping_lambda=0.1,
        ),
    )

    runtime_fields = spec.to_runtime_fields(cell_width_ms=100)

    assert runtime_fields["flow_zscore_window_bins"] == 180
    assert runtime_fields["flow_zscore_min_periods"] == 60
    assert runtime_fields["flow_derivative_weights"] == [0.6, 0.3, 0.1]
    assert runtime_fields["flow_tanh_scale"] == 2.5
    assert runtime_fields["flow_neutral_threshold"] == 0.2
    assert runtime_fields["state_model_zscore_window_bins"] == 240
    assert runtime_fields["state_model_zscore_min_periods"] == 80
    assert runtime_fields["state_model_d1_weight"] == 0.9
    assert runtime_fields["state_model_d2_weight"] == 0.1
    assert runtime_fields["state_model_d3_weight"] == 0.0
    assert runtime_fields["state_model_bull_pressure_weight"] == 1.1
    assert runtime_fields["state_model_bull_vacuum_weight"] == 1.2
    assert runtime_fields["state_model_bear_pressure_weight"] == 0.9
    assert runtime_fields["state_model_bear_vacuum_weight"] == 1.0
    assert runtime_fields["state_model_mixed_weight"] == 0.2
    assert runtime_fields["projection_horizons_bins"] == [1, 3, 5]


def test_serving_spec_runtime_fields_rejects_unknown_derivative_param() -> None:
    spec = ServingSpec(
        name="bad_derivative",
        pipeline="mnq_60m_baseline",
        signal=SignalConfig(
            name="derivative",
            params={"unknown_param": 123},
        ),
    )
    with pytest.raises(ValueError, match="Unknown derivative signal params"):
        spec.to_runtime_fields(cell_width_ms=100)


def test_serving_spec_runtime_fields_rejects_non_multiple_projection_horizon() -> None:
    spec = ServingSpec(
        name="bad_projection",
        pipeline="mnq_60m_baseline",
        projection=ProjectionConfig(horizons_ms=[250]),
    )
    with pytest.raises(ValueError, match="exact multiples of cell_width_ms"):
        spec.to_runtime_fields(cell_width_ms=100)
