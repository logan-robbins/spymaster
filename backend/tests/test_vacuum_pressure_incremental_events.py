"""Deterministic event-state tests for incremental vacuum-pressure signals."""
from __future__ import annotations

import pytest
import pandas as pd

from src.vacuum_pressure.incremental import (
    EVENT_STATE_ARMED,
    EVENT_STATE_COOLDOWN,
    EVENT_STATE_FIRE,
    EVENT_STATE_WATCH,
    DirectionalEventStateMachine,
    EventStateConfig,
    IncrementalSignalEngine,
)


def _tick(
    machine: DirectionalEventStateMachine,
    *,
    net_lift: float,
    d1_15s: float,
    d2_15s: float,
    cross_conf: float,
    proj_coh: float,
    proj_dir: int,
) -> dict:
    return machine.update(
        net_lift=net_lift,
        d1_15s=d1_15s,
        d2_15s=d2_15s,
        cross_confidence=cross_conf,
        projection_coherence=proj_coh,
        projection_direction=proj_dir,
    )


def test_event_machine_full_transition_with_refractory_cooldown() -> None:
    """WATCH -> ARMED -> FIRE -> COOLDOWN -> WATCH progression is deterministic."""
    cfg = EventStateConfig(
        arm_net_lift_threshold=0.60,
        disarm_net_lift_threshold=0.30,
        fire_net_lift_threshold=0.90,
        min_abs_d1_arm=0.02,
        min_abs_d1_fire=0.05,
        max_countertrend_d2=0.02,
        min_cross_conf_arm=0.20,
        min_cross_conf_fire=0.35,
        min_projection_coh_arm=0.20,
        min_projection_coh_fire=0.30,
        arm_persistence_windows=2,
        fire_persistence_windows=2,
        fire_min_hold_windows=1,
        cooldown_windows=3,
    )
    machine = DirectionalEventStateMachine(cfg)

    arm_step = dict(
        net_lift=0.70, d1_15s=0.04, d2_15s=0.00, cross_conf=0.30, proj_coh=0.30, proj_dir=1
    )
    fire_step = dict(
        net_lift=1.15, d1_15s=0.09, d2_15s=0.01, cross_conf=0.55, proj_coh=0.50, proj_dir=1
    )
    weak_step = dict(
        net_lift=0.10, d1_15s=0.00, d2_15s=-0.03, cross_conf=0.00, proj_coh=0.00, proj_dir=0
    )

    s1 = _tick(machine, **arm_step)
    s2 = _tick(machine, **arm_step)
    s3 = _tick(machine, **fire_step)
    s4 = _tick(machine, **fire_step)
    s5 = _tick(machine, **weak_step)
    s6 = _tick(machine, **fire_step)
    s7 = _tick(machine, **fire_step)
    s8 = _tick(machine, **weak_step)
    s9 = _tick(machine, **arm_step)
    s10 = _tick(machine, **arm_step)

    assert s1["event_state"] == EVENT_STATE_WATCH
    assert s2["event_state"] == EVENT_STATE_ARMED
    assert s3["event_state"] == EVENT_STATE_ARMED
    assert s4["event_state"] == EVENT_STATE_FIRE
    assert s4["event_direction"] == "UP"

    # Hold fails -> refractory starts.
    assert s5["event_state"] == EVENT_STATE_COOLDOWN
    # During cooldown, even strong inputs cannot re-fire.
    assert s6["event_state"] == EVENT_STATE_COOLDOWN
    assert s7["event_state"] == EVENT_STATE_COOLDOWN
    # Cooldown expires back to WATCH.
    assert s8["event_state"] == EVENT_STATE_WATCH

    # Arming requires persistence again after cooldown.
    assert s9["event_state"] == EVENT_STATE_WATCH
    assert s10["event_state"] == EVENT_STATE_ARMED


def test_event_machine_anti_flicker_alternating_directions() -> None:
    """Alternating directional evidence does not arm/fire due persistence gates."""
    machine = DirectionalEventStateMachine()

    states: list[str] = []
    for i in range(12):
        sign = 1 if i % 2 == 0 else -1
        out = _tick(
            machine,
            net_lift=0.90 * sign,
            d1_15s=0.07 * sign,
            d2_15s=0.00,
            cross_conf=0.50,
            proj_coh=0.45,
            proj_dir=sign,
        )
        states.append(out["event_state"])
        assert out["event_direction"] == "NONE"

    assert set(states) == {EVENT_STATE_WATCH}


def test_event_machine_default_downward_direction_is_not_reversed() -> None:
    """Default machine preserves bearish sign as DOWN through arm/fire."""
    machine = DirectionalEventStateMachine()

    bearish_step = dict(
        net_lift=-1.05,
        d1_15s=-0.10,
        d2_15s=-0.01,
        cross_conf=0.65,
        proj_coh=0.55,
        proj_dir=-1,
    )

    s1 = _tick(machine, **bearish_step)
    s2 = _tick(machine, **bearish_step)
    s3 = _tick(machine, **bearish_step)
    s4 = _tick(machine, **bearish_step)

    assert s1["event_state"] == EVENT_STATE_WATCH
    assert s1["event_direction"] == "NONE"
    assert s2["event_state"] == EVENT_STATE_ARMED
    assert s2["event_direction"] == "DOWN"
    assert s3["event_state"] == EVENT_STATE_ARMED
    assert s3["event_direction"] == "DOWN"
    assert s4["event_state"] == EVENT_STATE_FIRE
    assert s4["event_direction"] == "DOWN"


def test_event_machine_blocks_countertrend_acceleration() -> None:
    """Large opposing d2_15s blocks arming even with strong net_lift and d1."""
    machine = DirectionalEventStateMachine(
        EventStateConfig(
            arm_net_lift_threshold=0.60,
            disarm_net_lift_threshold=0.30,
            fire_net_lift_threshold=0.90,
            min_abs_d1_arm=0.02,
            min_abs_d1_fire=0.05,
            max_countertrend_d2=0.01,
            min_cross_conf_arm=0.20,
            min_cross_conf_fire=0.35,
            min_projection_coh_arm=0.20,
            min_projection_coh_fire=0.30,
            arm_persistence_windows=2,
            fire_persistence_windows=2,
            fire_min_hold_windows=1,
            cooldown_windows=3,
        )
    )

    for _ in range(5):
        out = _tick(
            machine,
            net_lift=1.20,
            d1_15s=0.10,
            d2_15s=-0.25,
            cross_conf=0.60,
            proj_coh=0.60,
            proj_dir=1,
        )
        assert out["event_state"] == EVENT_STATE_WATCH


def test_event_state_config_fails_fast_on_invalid_hysteresis() -> None:
    """Invalid threshold ordering fails fast."""
    bad_cfg = EventStateConfig(
        arm_net_lift_threshold=0.50,
        disarm_net_lift_threshold=0.55,
    )
    with pytest.raises(ValueError, match="disarm_net_lift_threshold"):
        bad_cfg.validate()


def test_incremental_engine_emits_explicit_event_fields() -> None:
    """process_window emits event state/direction/strength/confidence fields."""
    engine = IncrementalSignalEngine(bucket_size_dollars=0.50)
    snap = {
        "window_end_ts_ns": 1_000_000_000,
        "mid_price": 100.0,
        "best_bid_price_int": 9990,
        "best_ask_price_int": 10010,
        "book_valid": True,
    }
    flow = pd.DataFrame([
        {
            "window_end_ts_ns": 1_000_000_000,
            "rel_ticks": -1,
            "side": "B",
            "add_qty": 12.0,
            "pull_qty": 2.0,
            "fill_qty": 4.0,
            "depth_qty_end": 30.0,
            "depth_qty_rest": 14.0,
            "pull_qty_rest": 1.0,
            "spot_ref_price_int": 10000,
            "window_valid": True,
        },
        {
            "window_end_ts_ns": 1_000_000_000,
            "rel_ticks": 1,
            "side": "A",
            "add_qty": 6.0,
            "pull_qty": 10.0,
            "fill_qty": 2.0,
            "depth_qty_end": 28.0,
            "depth_qty_rest": 12.0,
            "pull_qty_rest": 1.0,
            "spot_ref_price_int": 10000,
            "window_valid": True,
        },
    ])

    out = engine.process_window(snap, flow)
    for key in (
        "projection_coherence",
        "event_state",
        "event_direction",
        "event_strength",
        "event_confidence",
    ):
        assert key in out
