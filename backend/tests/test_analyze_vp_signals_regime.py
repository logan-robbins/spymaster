from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.analyze_vp_signals import (
    SPECTRUM_PRESSURE,
    SPECTRUM_TRANSITION,
    SPECTRUM_VACUUM,
    capture_grids,
    compute_directional_spectrum,
    detect_direction_switch_events,
    evaluate_hourly_stability,
    evaluate_trade_targets,
    summarize_hourly_trade_metrics,
    summarize_trade_metrics,
)
from src.vacuum_pressure.config import resolve_config


RAW_MNQ_DB_PATH = (
    BACKEND_ROOT
    / "lake"
    / "raw"
    / "source=databento"
    / "product_type=future_mbo"
    / "symbol=MNQ"
    / "table=market_by_order_dbn"
    / "glbx-mdp3-20260206.mbo.dbn"
)
GOLDEN_PATH = Path(__file__).resolve().with_name(
    "golden_mnq_20260206_0900_1200.json"
)


def _run_real_replay() -> Dict[str, Any]:
    products_yaml_path = BACKEND_ROOT / "src" / "data_eng" / "config" / "products.yaml"
    config = resolve_config("future_mbo", "MNQH6", products_yaml_path)
    ts_ns, mid_price, bucket_data = capture_grids(
        lake_root=BACKEND_ROOT / "lake",
        config=config,
        dt="2026-02-06",
        start_time="09:00",
        throttle_ms=25.0,
        end_time_et="12:00",
    )

    eval_start_ns = int(
        pd.Timestamp("2026-02-06 09:00:00", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )
    eval_end_ns = int(
        pd.Timestamp("2026-02-06 12:00:00", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )
    eval_mask = (ts_ns >= eval_start_ns) & (ts_ns < eval_end_ns)

    directional_df = compute_directional_spectrum(
        ts_ns=ts_ns,
        bucket_data=bucket_data,
        directional_bands=[4, 8, 16],
        micro_windows=[25, 50, 100, 200],
        normalization_window=300,
        normalization_min_periods=75,
        spectrum_threshold=0.15,
        directional_edge_threshold=0.20,
    )
    events = detect_direction_switch_events(
        eval_mask=eval_mask,
        direction_state=directional_df["direction_state"].values,
        cooldown_snapshots=8,
    )
    outcomes = evaluate_trade_targets(
        ts_ns=ts_ns,
        mid_price=mid_price,
        events=events,
        tick_size=float(config.tick_size),
        tp_ticks=8,
        sl_ticks=4,
        max_hold_snapshots=1200,
    )

    eval_duration_hours = max(1e-9, (eval_end_ns - eval_start_ns) / 3.6e12)
    trade_metrics = summarize_trade_metrics(outcomes, eval_duration_hours)
    hourly_metrics = summarize_hourly_trade_metrics(
        ts_ns=ts_ns,
        outcomes=outcomes,
        dt="2026-02-06",
        eval_start="09:00",
        eval_end="12:00",
    )
    stability = evaluate_hourly_stability(
        hourly_metrics=hourly_metrics,
        max_drift=0.35,
        min_signals_per_hour=5,
    )

    return {
        "ts_ns": ts_ns,
        "eval_mask": eval_mask,
        "directional_df": directional_df,
        "outcomes": outcomes,
        "trade_metrics": trade_metrics,
        "hourly_metrics": hourly_metrics,
        "stability": stability,
    }


@pytest.fixture(scope="session")
def real_replay_result() -> Dict[str, Any]:
    if os.getenv("VP_ENABLE_REAL_REPLAY_TESTS", "0") != "1":
        pytest.skip("Set VP_ENABLE_REAL_REPLAY_TESTS=1 to run real MNQ replay tests.")
    if not RAW_MNQ_DB_PATH.exists():
        pytest.skip(f"Missing real MNQ DBN file: {RAW_MNQ_DB_PATH}")
    return _run_real_replay()


def test_real_replay_path_is_raw_databento_not_synthetic() -> None:
    raw_path = str(RAW_MNQ_DB_PATH)
    assert "lake/raw/source=databento" in raw_path
    assert "synthetic" not in raw_path.lower()


def test_real_replay_spectrum_and_trade_invariants(real_replay_result: Dict[str, Any]) -> None:
    directional_df = real_replay_result["directional_df"]
    eval_mask = real_replay_result["eval_mask"]
    outcomes = real_replay_result["outcomes"]
    trade_metrics = real_replay_result["trade_metrics"]

    score_cols = [c for c in directional_df.columns if c.endswith("_score")]
    for col in score_cols:
        vals = directional_df.loc[eval_mask, col].values
        assert np.isfinite(vals).all(), f"Non-finite values in {col}"

    for state_col in [c for c in directional_df.columns if c.endswith("_state")]:
        allowed = {SPECTRUM_PRESSURE, SPECTRUM_TRANSITION, SPECTRUM_VACUUM}
        if state_col in ("direction_state", "posture_state"):
            continue
        got = set(directional_df.loc[eval_mask, state_col].astype(str).unique().tolist())
        assert got.issubset(allowed), f"Unexpected states in {state_col}: {got}"

    n_events = int(trade_metrics["n_events"])
    tp = int((outcomes["outcome"] == "tp_before_sl").sum())
    sl = int((outcomes["outcome"] == "sl_before_tp").sum())
    timeout = int((outcomes["outcome"] == "timeout").sum())
    assert tp + sl + timeout == n_events
    assert n_events >= 0


def test_real_replay_matches_golden_metrics(real_replay_result: Dict[str, Any]) -> None:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"Golden metrics file missing: {GOLDEN_PATH}")

    golden = json.loads(GOLDEN_PATH.read_text())

    trade_metrics = real_replay_result["trade_metrics"]
    hourly_metrics: List[Dict[str, Any]] = real_replay_result["hourly_metrics"]
    stability = real_replay_result["stability"]

    assert int(trade_metrics["n_events"]) == int(golden["trade_metrics"]["n_events"])
    assert float(trade_metrics["events_per_hour"]) == pytest.approx(
        float(golden["trade_metrics"]["events_per_hour"]), rel=1e-9, abs=1e-9
    )
    assert float(trade_metrics["tp_before_sl_rate"]) == pytest.approx(
        float(golden["trade_metrics"]["tp_before_sl_rate"]), rel=1e-9, abs=1e-9
    )
    assert float(trade_metrics["sl_before_tp_rate"]) == pytest.approx(
        float(golden["trade_metrics"]["sl_before_tp_rate"]), rel=1e-9, abs=1e-9
    )
    assert float(trade_metrics["timeout_rate"]) == pytest.approx(
        float(golden["trade_metrics"]["timeout_rate"]), rel=1e-9, abs=1e-9
    )

    assert len(hourly_metrics) == len(golden["hourly_metrics"])
    for got, exp in zip(hourly_metrics, golden["hourly_metrics"]):
        assert str(got["window"]) == str(exp["window"])
        assert int(got["n_events"]) == int(exp["n_events"])
        assert float(got["events_per_hour"]) == pytest.approx(
            float(exp["events_per_hour"]), rel=1e-9, abs=1e-9
        )
        assert float(got["tp_before_sl_rate"]) == pytest.approx(
            float(exp["tp_before_sl_rate"]), rel=1e-9, abs=1e-9
        )

    assert bool(stability["passed"]) == bool(golden["stability"]["passed"])
    assert float(stability["tp_before_sl_rate_drift"]) == pytest.approx(
        float(golden["stability"]["tp_before_sl_rate_drift"]), rel=1e-9, abs=1e-9
    )
    assert float(stability["events_per_hour_drift"]) == pytest.approx(
        float(golden["stability"]["events_per_hour_drift"]), rel=1e-9, abs=1e-9
    )
