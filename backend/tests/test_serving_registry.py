from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import VPRuntimeConfig
from src.vacuum_pressure.serving_config import (
    PublishedServingSource,
    PublishedServingSpec,
)
from src.vacuum_pressure.serving_registry import ServingRegistry


def _runtime_snapshot() -> dict[str, object]:
    cfg = VPRuntimeConfig(
        product_type="future_mbo",
        symbol="MNQH6",
        symbol_root="MNQ",
        price_scale=1e-9,
        tick_size=0.25,
        bucket_size_dollars=0.25,
        rel_tick_size=1.0,
        grid_radius_ticks=50,
        cell_width_ms=100,
        n_absolute_ticks=8192,
        flow_windows=(5, 10, 20, 40),
        flow_rollup_weights=(1.0, 1.0, 1.0, 1.0),
        flow_derivative_weights=(0.55, 0.30, 0.15),
        flow_tanh_scale=3.0,
        flow_neutral_threshold=0.15,
        flow_zscore_window_bins=300,
        flow_zscore_min_periods=75,
        projection_horizons_bins=(1, 2, 4),
        projection_horizons_ms=(100, 200, 400),
        contract_multiplier=2.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="snapshot",
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
        state_model_center_exclusion_radius=0,
        state_model_spatial_decay_power=0.0,
        state_model_zscore_window_bins=240,
        state_model_zscore_min_periods=60,
        state_model_tanh_scale=3.0,
        state_model_d1_weight=1.0,
        state_model_d2_weight=0.0,
        state_model_d3_weight=0.0,
        state_model_bull_pressure_weight=1.0,
        state_model_bull_vacuum_weight=1.0,
        state_model_bear_pressure_weight=1.0,
        state_model_bear_vacuum_weight=1.0,
        state_model_mixed_weight=0.0,
    )
    snapshot = cfg.to_dict()
    snapshot["stream_dt"] = "2026-02-06"
    snapshot["stream_start_time"] = "09:25"
    return snapshot


def _published_spec(
    *,
    serving_id: str,
    run_id: str,
    experiment_name: str,
    config_hash: str,
) -> PublishedServingSpec:
    return PublishedServingSpec(
        serving_id=serving_id,
        description="published for parity",
        runtime_snapshot=_runtime_snapshot(),
        source=PublishedServingSource(
            run_id=run_id,
            experiment_name=experiment_name,
            config_hash=config_hash,
            promoted_at_utc="2026-02-20T00:00:00+00:00",
            serving_spec_name="serving_test",
            signal_name="derivative",
        ),
    )


def test_promote_and_resolve_alias_and_id(tmp_path: Path) -> None:
    registry = ServingRegistry(tmp_path)
    serving_id = registry.build_serving_id(
        experiment_name="sweep_derivative_rr20",
        run_id="runabc1234567890",
        config_hash="cfg11223344556677",
    )
    spec = _published_spec(
        serving_id=serving_id,
        run_id="runabc1234567890",
        experiment_name="sweep_derivative_rr20",
        config_hash="cfg11223344556677",
    )
    result = registry.promote(alias="vp_main", spec=spec, actor="test")

    assert result.alias == "vp_main"
    assert result.serving_id == serving_id
    assert result.spec_path.exists()
    assert result.reused_existing is False

    resolved_alias = registry.resolve("vp_main")
    assert resolved_alias.serving_id == serving_id
    assert resolved_alias.alias == "vp_main"
    assert resolved_alias.spec.runtime_snapshot["symbol"] == "MNQH6"

    resolved_id = registry.resolve(serving_id)
    assert resolved_id.serving_id == serving_id
    assert resolved_id.spec.source.run_id == "runabc1234567890"

    preferred = registry.preferred_alias_for_run("runabc1234567890")
    assert preferred == ("vp_main", serving_id)


def test_promote_dedups_same_run_and_config_hash(tmp_path: Path) -> None:
    registry = ServingRegistry(tmp_path)

    id_a = registry.build_serving_id(
        experiment_name="sweep_derivative_rr20",
        run_id="run0000000000001",
        config_hash="cfg000000000001",
    )
    spec_a = _published_spec(
        serving_id=id_a,
        run_id="run0000000000001",
        experiment_name="sweep_derivative_rr20",
        config_hash="cfg000000000001",
    )
    first = registry.promote(alias="vp_primary", spec=spec_a, actor="test")

    id_b = registry.build_serving_id(
        experiment_name="sweep_derivative_rr20",
        run_id="run0000000000001",
        config_hash="cfg000000000001",
    )
    spec_b = _published_spec(
        serving_id=id_b,
        run_id="run0000000000001",
        experiment_name="sweep_derivative_rr20",
        config_hash="cfg000000000001",
    )
    second = registry.promote(alias="vp_shadow", spec=spec_b, actor="test")

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert second.serving_id == first.serving_id

    resolved_shadow = registry.resolve("vp_shadow")
    assert resolved_shadow.serving_id == first.serving_id


def test_build_serving_id_is_deterministic(tmp_path: Path) -> None:
    registry = ServingRegistry(tmp_path)
    serving_id_a = registry.build_serving_id(
        experiment_name="sweep_derivative_rr20",
        run_id="run0000000000001",
        config_hash="cfg000000000001",
    )
    serving_id_b = registry.build_serving_id(
        experiment_name="sweep_derivative_rr20",
        run_id="run0000000000001",
        config_hash="cfg000000000001",
    )
    assert serving_id_a == serving_id_b
    assert serving_id_a.startswith("srv_sweep_derivative_rr20_run00000_cfg00000")
