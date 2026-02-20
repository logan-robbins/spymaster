from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import scripts.build_gold_dataset_campaign as campaign
from src.vacuum_pressure.config import VPRuntimeConfig


def _base_campaign() -> campaign.CampaignConfig:
    return campaign.CampaignConfig(
        name="mnq_grid_force",
        base_capture=campaign.BaseCapture(
            product_type="future_mbo",
            symbol="MNQH6",
            dt="2026-02-06",
            capture_start_et="09:25:00",
            capture_end_et="10:25:00",
        ),
        publish=campaign.PublishConfig(agents=("eda", "projection")),
        sweep_axes={
            "cell_width_ms": [100, 200],
            "spectrum_tanh_scale": [3.0, 4.0],
        },
        bundles=(
            {"c1_v_add": 1.0, "tau_velocity": 2.0},
            {"c1_v_add": 1.2, "tau_velocity": 1.6},
            {"c1_v_add": 0.8, "tau_velocity": 2.4},
        ),
    )


def _base_runtime_config() -> VPRuntimeConfig:
    return VPRuntimeConfig(
        product_type="future_mbo",
        symbol="MNQH6",
        symbol_root="MNQ",
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
    )


def test_expand_variants_cartesian_with_bundles() -> None:
    variants = campaign._expand_variants(_base_campaign())
    assert len(variants) == 12


def test_variant_hash_is_deterministic_for_equivalent_mapping() -> None:
    cfg = _base_campaign()
    v1 = {"cell_width_ms": 100, "spectrum_tanh_scale": 3.0, "c1_v_add": 1.0}
    v2 = {"c1_v_add": 1.0, "spectrum_tanh_scale": 3.0, "cell_width_ms": 100}

    h1 = campaign._variant_hash(campaign=cfg, variant=v1)
    h2 = campaign._variant_hash(campaign=cfg, variant=v2)
    assert h1 == h2


def test_validate_variant_keys_rejects_unknown() -> None:
    runtime_allowed = campaign._runtime_override_keys_from_config(_base_runtime_config())
    with pytest.raises(ValueError, match="Unknown sweep/bundle keys"):
        campaign._validate_variant_keys(
            [{"cell_width_ms": 100, "not_supported": 1}],
            runtime_allowed,
        )


def test_validate_variant_keys_rejects_projection_horizon_override() -> None:
    runtime_allowed = campaign._runtime_override_keys_from_config(_base_runtime_config())
    with pytest.raises(ValueError, match="Unknown sweep/bundle keys"):
        campaign._validate_variant_keys(
            [{"projection_horizons_bins": [1, 2, 3]}],
            runtime_allowed,
        )
