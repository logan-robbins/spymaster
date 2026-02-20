from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import (
    VPRuntimeConfig,
    build_config_with_overrides,
    parse_projection_horizons_bins_override,
)


def _base_config() -> VPRuntimeConfig:
    return VPRuntimeConfig(
        product_type="future_mbo",
        symbol="TESTH6",
        symbol_root="TEST",
        price_scale=1e-9,
        tick_size=0.25,
        bucket_size_dollars=0.25,
        rel_tick_size=0.25,
        grid_radius_ticks=50,
        cell_width_ms=100,
        n_absolute_ticks=8192,
        spectrum_windows=(5, 10, 20, 40),
        spectrum_rollup_weights=(1.0, 1.0, 1.0, 1.0),
        spectrum_derivative_weights=(0.55, 0.30, 0.15),
        spectrum_tanh_scale=3.0,
        spectrum_threshold_neutral=0.15,
        zscore_window_bins=300,
        zscore_min_periods=75,
        projection_horizons_bins=(1, 2, 3, 4),
        projection_horizons_ms=(100, 200, 300, 400),
        contract_multiplier=2.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="base",
        tau_velocity=2.0,
        tau_acceleration=5.0,
        tau_jerk=10.0,
        tau_rest_decay=30.0,
        c1_v_add=1.0,
        c2_v_rest_pos=0.5,
        c3_a_add=0.3,
        c4_v_pull=1.0,
        c5_v_fill=1.5,
        c6_v_rest_neg=0.5,
        c7_a_pull=0.3,
    )


def test_build_config_with_overrides_recomputes_derived_fields_and_version() -> None:
    base = _base_config()

    updated = build_config_with_overrides(
        base,
        {
            "cell_width_ms": 200,
            "projection_horizons_bins": [1, 3],
            "tau_velocity": 1.5,
            "c1_v_add": 1.2,
        },
    )

    assert updated.cell_width_ms == 200
    assert updated.projection_horizons_bins == (1, 3)
    assert updated.projection_horizons_ms == (200, 600)
    assert updated.tau_velocity == pytest.approx(1.5)
    assert updated.c1_v_add == pytest.approx(1.2)

    assert updated.config_version != base.config_version

    # Ensure base config remains immutable.
    assert base.cell_width_ms == 100
    assert base.projection_horizons_ms == (100, 200, 300, 400)


def test_build_config_with_overrides_rejects_unknown_key() -> None:
    base = _base_config()
    with pytest.raises(ValueError, match="Unknown runtime override keys"):
        build_config_with_overrides(base, {"not_a_real_key": 123})


def test_build_config_with_overrides_validates_tau_positive() -> None:
    base = _base_config()
    with pytest.raises(ValueError, match="tau_velocity"):
        build_config_with_overrides(base, {"tau_velocity": 0.0})


def test_parse_projection_horizons_bins_override_csv() -> None:
    assert parse_projection_horizons_bins_override("1,2,4") == (1, 2, 4)


def test_parse_projection_horizons_bins_override_empty_is_none() -> None:
    assert parse_projection_horizons_bins_override("") is None
    assert parse_projection_horizons_bins_override(None) is None


def test_parse_projection_horizons_bins_override_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="projection_horizons_bins"):
        parse_projection_horizons_bins_override("1,0,2")


def test_build_config_with_overrides_accepts_perm_runtime_params() -> None:
    base = _base_config()
    updated = build_config_with_overrides(
        base,
        {
            "perm_runtime_enabled": False,
            "perm_zscore_window_bins": 120,
            "perm_zscore_min_periods": 30,
            "perm_d1_weight": 0.7,
            "perm_d2_weight": 0.2,
            "perm_d3_weight": 0.1,
        },
    )
    assert updated.perm_runtime_enabled is False
    assert updated.perm_zscore_window_bins == 120
    assert updated.perm_zscore_min_periods == 30
    assert updated.perm_d1_weight == pytest.approx(0.7)
    assert updated.perm_d2_weight == pytest.approx(0.2)
    assert updated.perm_d3_weight == pytest.approx(0.1)


def test_build_config_with_overrides_rejects_invalid_perm_weights() -> None:
    base = _base_config()
    with pytest.raises(ValueError, match="At least one of perm_d1_weight"):
        build_config_with_overrides(
            base,
            {
                "perm_d1_weight": 0.0,
                "perm_d2_weight": 0.0,
                "perm_d3_weight": 0.0,
            },
        )
