from __future__ import annotations

import numpy as np
import pandas as pd

from src.vacuum_pressure.evaluation import (
    ThresholdGate,
    ensure_event_columns,
    evaluate_threshold_gate,
    evaluate_fire_events,
    prepare_signal_frame,
    sweep_fire_operating_grid,
    sweep_threshold_grid,
)


def _base_frame() -> pd.DataFrame:
    ts = np.arange(12, dtype=np.int64) * 1_000_000_000
    mid = np.array(
        [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0],
        dtype=np.float64,
    )
    net_lift = np.array(
        [1.0, -1.0, 1.0, -1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        dtype=np.float64,
    )
    regime = np.array(
        ["LIFT", "DRAG", "LIFT", "DRAG", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL"],
        dtype=object,
    )
    return pd.DataFrame(
        {
            "window_end_ts_ns": ts,
            "mid_price": mid,
            "net_lift": net_lift,
            "cross_confidence": 0.9,
            "d1_15s": 0.2,
            "regime": regime,
            "book_valid": True,
        }
    )


def test_evaluate_threshold_gate_core_metrics() -> None:
    frame = prepare_signal_frame(_base_frame())
    gate = ThresholdGate(
        min_abs_net_lift=0.5,
        min_cross_confidence=0.5,
        min_abs_d1_15s=0.1,
        require_regime_alignment=True,
    )
    result = evaluate_threshold_gate(
        frame=frame,
        gate=gate,
        horizons_s=[2, 5, 10],
        tick_size=1.0,
        min_move_ticks=1.0,
    )

    h2 = result["horizons"]["2s"]
    assert h2["alerts"] == 4
    assert h2["hits"] == 2
    assert h2["false_alerts"] == 1
    assert h2["stale_alerts"] == 1
    assert h2["hit_rate"] == 0.5
    assert h2["mean_lead_time_s"] == 1.0
    assert h2["false_alert_density_per_minute"] == 6.0

    regimes = h2["regime_stratified"]
    assert regimes["LIFT"]["alerts"] == 2
    assert regimes["LIFT"]["hits"] == 1
    assert regimes["LIFT"]["false_alerts"] == 1
    assert regimes["DRAG"]["alerts"] == 2
    assert regimes["DRAG"]["hits"] == 1
    assert regimes["DRAG"]["false_alerts"] == 0


def test_regime_alignment_gate_reduces_alerts() -> None:
    df = _base_frame()
    df.loc[0, "regime"] = "DRAG"  # mismatched with positive net_lift
    frame = prepare_signal_frame(df)

    loose = evaluate_threshold_gate(
        frame=frame,
        gate=ThresholdGate(0.5, 0.5, 0.1, False),
        horizons_s=[2],
        tick_size=1.0,
        min_move_ticks=1.0,
    )
    aligned = evaluate_threshold_gate(
        frame=frame,
        gate=ThresholdGate(0.5, 0.5, 0.1, True),
        horizons_s=[2],
        tick_size=1.0,
        min_move_ticks=1.0,
    )

    assert loose["horizons"]["2s"]["alerts"] == 4
    assert aligned["horizons"]["2s"]["alerts"] == 3


def test_sweep_threshold_grid_prefers_higher_quality_threshold() -> None:
    ts = np.arange(12, dtype=np.int64) * 1_000_000_000
    mid = np.array(
        [100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0],
        dtype=np.float64,
    )
    net_lift = np.array(
        [0.2, -0.2, 1.2, 1.2, 1.2, -1.2, -1.2, 0.2, -0.2, 0.2, 0.0, 0.0],
        dtype=np.float64,
    )
    frame = prepare_signal_frame(
        pd.DataFrame(
            {
                "window_end_ts_ns": ts,
                "mid_price": mid,
                "net_lift": net_lift,
                "cross_confidence": 0.9,
                "d1_15s": 0.2,
                "regime": "LIFT",
                "book_valid": True,
            }
        )
    )

    grid = sweep_threshold_grid(
        frame=frame,
        horizons_s=[2, 5, 10],
        tick_size=1.0,
        min_move_ticks=1.0,
        net_lift_thresholds=[0.1, 0.8],
        confidence_thresholds=[0.0],
        d1_15s_thresholds=[0.0],
        require_regime_alignment_values=[False],
        primary_horizon_s=2,
        min_alerts=2,
        target_alert_rate=0.01,
        top_k=2,
    )

    rec = grid["recommended"]
    assert rec is not None
    assert rec["thresholds"]["min_abs_net_lift"] == 0.8


def _fire_eval_frame() -> pd.DataFrame:
    ts = np.arange(10, dtype=np.int64) * 1_000_000_000
    mid = np.array(
        [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0, 100.0, 101.0],
        dtype=np.float64,
    )
    event_state = np.array(["WATCH"] * 10, dtype=object)
    event_direction = np.array(["NONE"] * 10, dtype=object)
    regime = np.array(["NEUTRAL"] * 10, dtype=object)

    fire_rows = {
        0: ("UP", "LIFT"),
        2: ("DOWN", "DRAG"),
        4: ("UP", "LIFT"),
        7: ("DOWN", "DRAG"),
        8: ("UP", "LIFT"),
    }
    for idx, (direction, reg) in fire_rows.items():
        event_state[idx] = "FIRE"
        event_direction[idx] = direction
        regime[idx] = reg

    return prepare_signal_frame(
        pd.DataFrame(
            {
                "window_end_ts_ns": ts,
                "mid_price": mid,
                "net_lift": 0.0,
                "cross_confidence": 0.0,
                "d1_15s": 0.0,
                "regime": regime,
                "book_valid": True,
                "event_state": event_state,
                "event_direction": event_direction,
            }
        ),
        extra_columns=("event_state", "event_direction"),
    )


def test_evaluate_fire_events_first_touch_metrics() -> None:
    frame = _fire_eval_frame()
    result = evaluate_fire_events(
        frame=frame,
        horizon_s=2,
        tick_size=1.0,
        target_ticks=2.0,
    )

    overall = result["overall"]
    assert overall["total_fire_events"] == 5
    assert overall["evaluable_fire_events"] == 4
    assert overall["unevaluable_fire_events"] == 1
    assert overall["hit_events"] == 2
    assert overall["false_fire_events"] == 2
    assert overall["unresolved_events"] == 0
    assert overall["hit_rate"] == 0.5
    assert overall["false_fire_rate"] == 0.5
    assert overall["unresolved_rate"] == 0.0
    assert overall["mean_time_to_hit_s"] == 2.0
    assert overall["median_time_to_hit_s"] == 2.0

    by_direction = result["by_event_direction"]
    assert by_direction["UP"]["total_fire_events"] == 3
    assert by_direction["UP"]["hit_events"] == 1
    assert by_direction["UP"]["false_fire_events"] == 1
    assert by_direction["UP"]["unresolved_events"] == 0
    assert by_direction["UP"]["unevaluable_fire_events"] == 1
    assert by_direction["DOWN"]["total_fire_events"] == 2
    assert by_direction["DOWN"]["hit_events"] == 1
    assert by_direction["DOWN"]["false_fire_events"] == 1
    assert by_direction["DOWN"]["unresolved_events"] == 0

    by_regime = result["by_regime"]
    assert by_regime["LIFT"]["total_fire_events"] == 3
    assert by_regime["DRAG"]["total_fire_events"] == 2


def test_sweep_fire_operating_grid_prefers_less_timeout_horizon() -> None:
    frame = _fire_eval_frame()
    sweep = sweep_fire_operating_grid(
        frame=frame,
        horizons_s=[1, 2],
        target_ticks_values=[2.0],
        tick_size=1.0,
        min_evaluable_fires=1,
        top_k=2,
    )
    rec = sweep["recommended"]
    assert rec is not None
    assert rec["horizon_s"] == 2
    assert rec["target_ticks"] == 2.0


def test_ensure_event_columns_reconstructs_replay_event_labels() -> None:
    ts = np.arange(6, dtype=np.int64) * 1_000_000_000
    df = pd.DataFrame(
        {
            "window_end_ts_ns": ts,
            "mid_price": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
            "composite": [0.0, 0.9, 1.2, 1.1, 0.2, 0.1],
            "d1_smooth": [0.0, 0.08, 0.10, 0.09, 0.01, 0.00],
            "d2_smooth": [0.0, 0.00, 0.01, 0.01, -0.05, -0.05],
            "confidence": [0.0, 0.5, 0.6, 0.6, 0.0, 0.0],
            "wtd_deriv_conf": [0.0, 0.5, 0.6, 0.6, 0.0, 0.0],
            "wtd_projection": [0.0, 0.3, 0.4, 0.4, 0.0, 0.0],
        }
    )
    out = ensure_event_columns(df)
    assert "event_state" in out.columns
    assert "event_direction" in out.columns
    assert "regime" in out.columns
    assert ((out["event_state"] == "ARMED") | (out["event_state"] == "FIRE")).any()
