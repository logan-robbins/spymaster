"""Tests for incremental vacuum-pressure feasibility gate fields."""
from __future__ import annotations

import pandas as pd
import pytest

from src.vacuum_pressure.incremental import IncrementalSignalEngine
from src.vacuum_pressure.server import SIGNALS_SCHEMA


def _snap_dict(window_end_ts_ns: int = 1_700_000_000_000_000_000) -> dict:
    """Build a minimal valid snap dict for incremental processing."""
    return {
        "window_end_ts_ns": window_end_ts_ns,
        "mid_price": 100.0,
        "best_bid_price_int": 99_500_000_000,
        "best_ask_price_int": 100_500_000_000,
        "book_valid": True,
    }


def _flow_df(direction: str) -> pd.DataFrame:
    """Build a one-window flow dataframe with directional asymmetry."""
    if direction == "up":
        rows = [
            # Ask above: vacuum above (pull >> add), low resistance above.
            {"rel_ticks": 1, "side": "A", "add_qty": 1.0, "pull_qty": 25.0, "fill_qty": 0.0, "depth_qty_end": 8.0, "depth_qty_rest": 1.0, "pull_qty_rest": 10.0},
            {"rel_ticks": 2, "side": "A", "add_qty": 1.0, "pull_qty": 12.0, "fill_qty": 0.0, "depth_qty_end": 6.0, "depth_qty_rest": 1.0, "pull_qty_rest": 6.0},
            # Bid below: pressure from below (add+fill high), support below.
            {"rel_ticks": -1, "side": "B", "add_qty": 30.0, "pull_qty": 1.0, "fill_qty": 14.0, "depth_qty_end": 70.0, "depth_qty_rest": 30.0, "pull_qty_rest": 1.0},
            {"rel_ticks": -2, "side": "B", "add_qty": 18.0, "pull_qty": 1.0, "fill_qty": 8.0, "depth_qty_end": 60.0, "depth_qty_rest": 25.0, "pull_qty_rest": 1.0},
        ]
    elif direction == "down":
        rows = [
            # Ask above: pressure from above (add+fill high), wall above.
            {"rel_ticks": 1, "side": "A", "add_qty": 30.0, "pull_qty": 1.0, "fill_qty": 14.0, "depth_qty_end": 70.0, "depth_qty_rest": 30.0, "pull_qty_rest": 1.0},
            {"rel_ticks": 2, "side": "A", "add_qty": 18.0, "pull_qty": 1.0, "fill_qty": 8.0, "depth_qty_end": 60.0, "depth_qty_rest": 25.0, "pull_qty_rest": 1.0},
            # Bid below: vacuum below (pull >> add), low resistance below.
            {"rel_ticks": -1, "side": "B", "add_qty": 1.0, "pull_qty": 25.0, "fill_qty": 0.0, "depth_qty_end": 8.0, "depth_qty_rest": 1.0, "pull_qty_rest": 10.0},
            {"rel_ticks": -2, "side": "B", "add_qty": 1.0, "pull_qty": 12.0, "fill_qty": 0.0, "depth_qty_end": 6.0, "depth_qty_rest": 1.0, "pull_qty_rest": 6.0},
        ]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    window_end_ts_ns = 1_700_000_000_000_000_000
    spot_ref_price_int = 100_000_000_000
    for row in rows:
        row["window_end_ts_ns"] = window_end_ts_ns
        row["spot_ref_price_int"] = spot_ref_price_int
        row["window_valid"] = True
    return pd.DataFrame(rows)


def test_feasibility_gate_direction_and_range() -> None:
    """Feasibility fields are bounded and follow directional asymmetry."""
    up_engine = IncrementalSignalEngine(bucket_size_dollars=0.50)
    up_signals = up_engine.process_window(_snap_dict(), _flow_df("up"))

    for field in ("feasibility_up", "feasibility_down"):
        assert 0.0 <= up_signals[field] <= 1.0
    assert -1.0 <= up_signals["directional_bias"] <= 1.0
    assert up_signals["feasibility_up"] > up_signals["feasibility_down"]
    assert up_signals["directional_bias"] > 0.0

    down_engine = IncrementalSignalEngine(bucket_size_dollars=0.50)
    down_signals = down_engine.process_window(_snap_dict(), _flow_df("down"))

    for field in ("feasibility_up", "feasibility_down"):
        assert 0.0 <= down_signals[field] <= 1.0
    assert -1.0 <= down_signals["directional_bias"] <= 1.0
    assert down_signals["feasibility_down"] > down_signals["feasibility_up"]
    assert down_signals["directional_bias"] < 0.0

    # Backward-compatible fields remain present.
    for legacy_field in ("net_lift", "pressure_above", "pressure_below", "alert_flags"):
        assert legacy_field in up_signals
        assert legacy_field in down_signals


def test_incremental_fail_fast_on_missing_required_column() -> None:
    """Missing required flow columns raise immediately."""
    engine = IncrementalSignalEngine(bucket_size_dollars=0.50)
    flow = _flow_df("up").drop(columns=["depth_qty_rest"])

    with pytest.raises(KeyError, match="missing required columns"):
        engine.process_window(_snap_dict(), flow)


def test_incremental_fail_fast_on_negative_quantities() -> None:
    """Negative quantity inputs are treated as invalid and fail fast."""
    engine = IncrementalSignalEngine(bucket_size_dollars=0.50)
    flow = _flow_df("up").copy()
    flow.loc[flow.index[0], "depth_qty_rest"] = -1.0

    with pytest.raises(ValueError, match="Negative quantity"):
        engine.process_window(_snap_dict(), flow)


def test_signals_schema_contains_feasibility_fields() -> None:
    """Wire schema includes newly added feasibility gate fields."""
    names = {field.name for field in SIGNALS_SCHEMA}
    assert "feasibility_up" in names
    assert "feasibility_down" in names
    assert "directional_bias" in names
