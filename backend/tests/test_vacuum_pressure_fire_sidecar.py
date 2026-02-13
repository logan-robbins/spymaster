from __future__ import annotations

import pytest

from src.vacuum_pressure.fire_sidecar import FireOutcomeTracker


def test_fire_transition_only_triggers_once_per_fire_state_run() -> None:
    tracker = FireOutcomeTracker(tick_size=0.25, tick_target=8.0, max_horizon_s=5.0)
    t0 = 1_000_000_000

    fire_1, outcomes_1 = tracker.update(
        window_end_ts_ns=t0,
        mid_price=100.0,
        event_state="FIRE",
        event_direction="UP",
    )
    assert fire_1 is not None
    assert outcomes_1 == []
    assert tracker.metrics()["fires"] == 1

    fire_2, outcomes_2 = tracker.update(
        window_end_ts_ns=t0 + 1_000_000_000,
        mid_price=100.1,
        event_state="FIRE",
        event_direction="UP",
    )
    assert fire_2 is None
    assert outcomes_2 == []
    assert tracker.metrics()["fires"] == 1

    # Exit FIRE then re-enter FIRE should create a new entry.
    tracker.update(
        window_end_ts_ns=t0 + 2_000_000_000,
        mid_price=100.1,
        event_state="WATCH",
        event_direction="NONE",
    )
    fire_3, _ = tracker.update(
        window_end_ts_ns=t0 + 3_000_000_000,
        mid_price=100.2,
        event_state="FIRE",
        event_direction="UP",
    )
    assert fire_3 is not None
    assert tracker.metrics()["fires"] == 2


def test_hit_resolution_within_horizon() -> None:
    tracker = FireOutcomeTracker(tick_size=0.25, tick_target=8.0, max_horizon_s=10.0)
    t0 = 5_000_000_000

    fire, _ = tracker.update(
        window_end_ts_ns=t0,
        mid_price=100.0,
        event_state="FIRE",
        event_direction="UP",
    )
    assert fire is not None
    assert fire.target_price == 102.0

    # 8 ticks up reached in 3s.
    _, outcomes = tracker.update(
        window_end_ts_ns=t0 + 3_000_000_000,
        mid_price=102.0,
        event_state="WATCH",
        event_direction="NONE",
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "hit"
    assert outcomes[0].time_to_outcome_s == 3.0

    metrics = tracker.metrics()
    assert metrics["resolved"] == 1
    assert metrics["hits"] == 1
    assert metrics["misses"] == 0
    assert metrics["hit_rate"] == 1.0
    assert metrics["avg_time_to_hit"] == 3.0


def test_miss_resolution_after_horizon() -> None:
    tracker = FireOutcomeTracker(tick_size=0.5, tick_target=8.0, max_horizon_s=2.0)
    t0 = 10_000_000_000

    fire, _ = tracker.update(
        window_end_ts_ns=t0,
        mid_price=100.0,
        event_state="FIRE",
        event_direction="DOWN",
    )
    assert fire is not None
    assert fire.target_price == 96.0

    # Horizon expires without reaching target.
    _, outcomes = tracker.update(
        window_end_ts_ns=t0 + 2_000_000_000,
        mid_price=99.5,
        event_state="WATCH",
        event_direction="NONE",
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "miss"

    metrics = tracker.metrics()
    assert metrics["resolved"] == 1
    assert metrics["hits"] == 0
    assert metrics["misses"] == 1
    assert metrics["hit_rate"] == 0.0
    assert metrics["avg_time_to_hit"] is None


def test_invalid_fire_direction_fails_fast() -> None:
    tracker = FireOutcomeTracker(tick_size=0.25, tick_target=8.0, max_horizon_s=5.0)
    with pytest.raises(ValueError, match="Unsupported event_direction"):
        tracker.update(
            window_end_ts_ns=1_000_000_000,
            mid_price=100.0,
            event_state="FIRE",
            event_direction="NONE",
        )
