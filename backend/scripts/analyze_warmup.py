"""Empirical analysis of warmup window for equity MBO silver pipeline.

Compares book engine output for QQQ 2026-02-06 under three warmup
scenarios (full / zero / 1-minute) and measures convergence.

Usage:
    cd backend
    uv run python scripts/analyze_warmup.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_eng.stages.silver.equity_mbo.book_engine import (
    DEPTH_FLOW_COLUMNS,
    SNAP_COLUMNS,
    EquityBookEngine,
    compute_equity_surfaces_1s_from_batches,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "QQQ"
SESSION_DATE = "2026-02-06"

BRONZE_PATH = (
    Path(__file__).resolve().parent.parent
    / "lake"
    / "bronze"
    / "source=databento"
    / "product_type=equity_mbo"
    / f"symbol={SYMBOL}"
    / "table=mbo"
    / f"dt={SESSION_DATE}"
    / "part-00000.parquet"
)

MBO_COLUMNS = [
    "ts_event",
    "action",
    "side",
    "price",
    "size",
    "order_id",
    "sequence",
    "flags",
]

# Timestamps in nanoseconds UTC
OPEN_0930_NS = int(
    pd.Timestamp(f"{SESSION_DATE} 09:30:00", tz="US/Eastern")
    .tz_convert("UTC")
    .value
)
CLOSE_0940_NS = int(
    pd.Timestamp(f"{SESSION_DATE} 09:40:00", tz="US/Eastern")
    .tz_convert("UTC")
    .value
)
PREMARKET_0600_NS = int(
    pd.Timestamp(f"{SESSION_DATE} 06:00:00", tz="US/Eastern")
    .tz_convert("UTC")
    .value
)
ONE_MIN_BEFORE_NS = int(
    pd.Timestamp(f"{SESSION_DATE} 09:29:00", tz="US/Eastern")
    .tz_convert("UTC")
    .value
)
FIVE_MIN_BEFORE_NS = int(
    pd.Timestamp(f"{SESSION_DATE} 09:25:00", tz="US/Eastern")
    .tz_convert("UTC")
    .value
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts_label(ns: int) -> str:
    """Convert nanosecond UTC timestamp to ET string."""
    return str(pd.Timestamp(ns, unit="ns", tz="UTC").tz_convert("US/Eastern"))


def run_engine(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run book engine on a pre-filtered DataFrame, return (snap, depth_flow)."""
    t0 = time.perf_counter()
    snap, flow, _ = compute_equity_surfaces_1s_from_batches(
        [df], compute_depth_flow=True
    )
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] engine ran in {elapsed:.2f}s  |  input rows: {len(df):,}  |  snap windows: {len(snap):,}  |  flow rows: {len(flow):,}")
    return snap, flow


def filter_output_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows in the [09:30, 09:40) output window."""
    return df[
        (df["window_start_ts_ns"] >= OPEN_0930_NS)
        & (df["window_end_ts_ns"] <= CLOSE_0940_NS)
    ].copy()


def compare_depth_flow(
    baseline: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Compare two depth_and_flow_1s DataFrames row-by-row.

    Returns a summary DataFrame showing per-second divergence stats.
    """
    # Align on (window_start_ts_ns, price_int, side)
    key = ["window_start_ts_ns", "price_int", "side"]
    value_cols = [
        "depth_qty_start",
        "depth_qty_end",
        "add_qty",
        "pull_qty",
        "depth_qty_rest",
        "pull_qty_rest",
        "fill_qty",
    ]

    merged = baseline.merge(
        test,
        on=key,
        how="outer",
        suffixes=("_base", "_test"),
        indicator=True,
    )

    # Per-window-second summary
    results: List[Dict] = []
    for ws_ns, grp in merged.groupby("window_start_ts_ns"):
        n_total = len(grp)
        n_base_only = (grp["_merge"] == "left_only").sum()
        n_test_only = (grp["_merge"] == "right_only").sum()
        n_both = (grp["_merge"] == "both").sum()

        # For rows present in both, compute metric diffs
        both = grp[grp["_merge"] == "both"]
        max_diffs: Dict[str, float] = {}
        mean_diffs: Dict[str, float] = {}
        for col in value_cols:
            base_vals = both[f"{col}_base"].fillna(0).to_numpy(dtype=np.float64)
            test_vals = both[f"{col}_test"].fillna(0).to_numpy(dtype=np.float64)
            abs_diff = np.abs(base_vals - test_vals)
            max_diffs[col] = float(abs_diff.max()) if len(abs_diff) > 0 else 0.0
            mean_diffs[col] = float(abs_diff.mean()) if len(abs_diff) > 0 else 0.0

        sec_offset = (int(ws_ns) - OPEN_0930_NS) // 1_000_000_000
        results.append(
            {
                "second_offset": sec_offset,
                "window_ts": _ts_label(int(ws_ns)),
                "rows_base_only": int(n_base_only),
                "rows_test_only": int(n_test_only),
                "rows_matched": int(n_both),
                "rows_total": int(n_total),
                **{f"max_diff_{c}": max_diffs.get(c, 0.0) for c in value_cols},
                **{f"mean_diff_{c}": mean_diffs.get(c, 0.0) for c in value_cols},
            }
        )

    summary = pd.DataFrame(results)
    return summary


def compare_snapshots(
    baseline: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Compare two book_snapshot_1s DataFrames row-by-row."""
    key = ["window_start_ts_ns"]
    snap_value_cols = [
        "best_bid_price_int",
        "best_bid_qty",
        "best_ask_price_int",
        "best_ask_qty",
        "mid_price",
        "last_trade_price_int",
    ]

    merged = baseline.merge(
        test,
        on=key,
        how="outer",
        suffixes=("_base", "_test"),
        indicator=True,
    )

    results: List[Dict] = []
    for _, row in merged.iterrows():
        ws_ns = int(row["window_start_ts_ns"])
        sec_offset = (ws_ns - OPEN_0930_NS) // 1_000_000_000
        diffs: Dict[str, float] = {}
        if row["_merge"] == "both":
            for col in snap_value_cols:
                base_v = float(row.get(f"{col}_base", 0) or 0)
                test_v = float(row.get(f"{col}_test", 0) or 0)
                diffs[col] = abs(base_v - test_v)
        else:
            for col in snap_value_cols:
                diffs[col] = float("nan")

        results.append(
            {
                "second_offset": sec_offset,
                "window_ts": _ts_label(ws_ns),
                "merge_status": row["_merge"],
                **{f"abs_diff_{c}": diffs[c] for c in snap_value_cols},
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("WARMUP ANALYSIS: QQQ equity MBO silver pipeline")
    print(f"Session: {SESSION_DATE}")
    print(f"Output window: 09:30:00 - 09:40:00 ET")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load bronze data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading bronze data from {BRONZE_PATH.name} ...")
    if not BRONZE_PATH.exists():
        print(f"  ERROR: file not found: {BRONZE_PATH}")
        sys.exit(1)

    df_full = pd.read_parquet(BRONZE_PATH, columns=MBO_COLUMNS)
    print(f"  Total bronze rows: {len(df_full):,}")
    print(f"  ts_event range: {_ts_label(int(df_full['ts_event'].min()))} → {_ts_label(int(df_full['ts_event'].max()))}")

    # ------------------------------------------------------------------
    # 2. Pre-market event counts
    # ------------------------------------------------------------------
    print("\n[2] Event distribution:")
    premarket = df_full[df_full["ts_event"] < OPEN_0930_NS]
    rth = df_full[df_full["ts_event"] >= OPEN_0930_NS]
    print(f"  Pre-market (before 09:30): {len(premarket):,} events")
    print(f"  RTH (09:30+):              {len(rth):,} events")

    # Break down pre-market by hour
    for h in range(6, 10):
        h_start = int(
            pd.Timestamp(f"{SESSION_DATE} {h:02d}:00:00", tz="US/Eastern")
            .tz_convert("UTC")
            .value
        )
        h_end = int(
            pd.Timestamp(f"{SESSION_DATE} {h + 1:02d}:00:00", tz="US/Eastern")
            .tz_convert("UTC")
            .value
        )
        if h == 9:
            h_end = OPEN_0930_NS  # only up to 09:30
        count = len(df_full[(df_full["ts_event"] >= h_start) & (df_full["ts_event"] < h_end)])
        end_label = f"{h + 1:02d}:00" if h < 9 else "09:30"
        print(f"    {h:02d}:00-{end_label} ET: {count:>10,} events")

    # Action breakdown in pre-market
    print("\n  Pre-market action breakdown:")
    for action, cnt in premarket["action"].value_counts().items():
        print(f"    {action}: {cnt:,}")

    # ------------------------------------------------------------------
    # 3. Simulate opening auction — count orders alive at 09:30
    # ------------------------------------------------------------------
    print("\n[3] Simulating pre-market book to count resting orders at 09:30 ...")
    alive_orders: Dict[int, Tuple[str, int, int]] = {}  # order_id -> (side, price, qty)
    for row in premarket.itertuples(index=False):
        action = str(row.action)
        oid = int(row.order_id)
        if action == "A":
            alive_orders[oid] = (str(row.side), int(row.price), int(row.size))
        elif action == "C" or action == "R":
            alive_orders.pop(oid, None)
        elif action == "F":
            if oid in alive_orders:
                s, p, q = alive_orders[oid]
                remaining = q - int(row.size)
                if remaining > 0:
                    alive_orders[oid] = (s, p, remaining)
                else:
                    alive_orders.pop(oid, None)
        elif action == "M":
            if oid in alive_orders:
                s, _, _ = alive_orders[oid]
                alive_orders[oid] = (s, int(row.price), int(row.size))

    n_bids = sum(1 for s, _, _ in alive_orders.values() if s == "B")
    n_asks = sum(1 for s, _, _ in alive_orders.values() if s == "A")
    bid_qty = sum(q for s, _, q in alive_orders.values() if s == "B")
    ask_qty = sum(q for s, _, q in alive_orders.values() if s == "A")
    print(f"  Resting orders at 09:30 ET: {len(alive_orders):,} total")
    print(f"    Bids: {n_bids:,} orders, {bid_qty:,} shares")
    print(f"    Asks: {n_asks:,} orders, {ask_qty:,} shares")

    # How many of these survive the first second of RTH?
    first_sec = rth[rth["ts_event"] < OPEN_0930_NS + 1_000_000_000]
    cancelled_in_first_sec = set()
    filled_in_first_sec = set()
    for row in first_sec.itertuples(index=False):
        action = str(row.action)
        oid = int(row.order_id)
        if action == "C":
            cancelled_in_first_sec.add(oid)
        elif action == "F":
            filled_in_first_sec.add(oid)
    premarket_oids = set(alive_orders.keys())
    cancelled_premarket = premarket_oids & cancelled_in_first_sec
    filled_premarket = premarket_oids & filled_in_first_sec
    print(f"\n  Of those, in the first 1s of RTH (09:30:00-09:30:01):")
    print(f"    Cancelled: {len(cancelled_premarket):,}")
    print(f"    Filled:    {len(filled_premarket):,}")
    print(f"    Surviving: {len(premarket_oids) - len(cancelled_premarket) - len(filled_premarket):,}")

    # ------------------------------------------------------------------
    # 4. Run book engine under three scenarios
    # ------------------------------------------------------------------
    print("\n[4] Running book engine under three warmup scenarios ...")

    # Scenario A: Full warmup (06:00 ET start = all bronze data through 09:40)
    df_a = df_full[df_full["ts_event"] < CLOSE_0940_NS].copy()
    print(f"\n  --- Scenario A: FULL warmup (from 06:00 ET) ---")
    snap_a, flow_a = run_engine(df_a, "FULL")

    # Scenario B: Zero warmup (from 09:30 ET)
    df_b = df_full[
        (df_full["ts_event"] >= OPEN_0930_NS) & (df_full["ts_event"] < CLOSE_0940_NS)
    ].copy()
    print(f"\n  --- Scenario B: ZERO warmup (from 09:30 ET) ---")
    snap_b, flow_b = run_engine(df_b, "ZERO")

    # Scenario C: 1-minute warmup (from 09:29 ET)
    df_c = df_full[
        (df_full["ts_event"] >= ONE_MIN_BEFORE_NS) & (df_full["ts_event"] < CLOSE_0940_NS)
    ].copy()
    print(f"\n  --- Scenario C: 1-MINUTE warmup (from 09:29 ET) ---")
    snap_c, flow_c = run_engine(df_c, "1MIN")

    # Scenario D: 5-minute warmup (from 09:25 ET)
    df_d = df_full[
        (df_full["ts_event"] >= FIVE_MIN_BEFORE_NS) & (df_full["ts_event"] < CLOSE_0940_NS)
    ].copy()
    print(f"\n  --- Scenario D: 5-MINUTE warmup (from 09:25 ET) ---")
    snap_d, flow_d = run_engine(df_d, "5MIN")

    # ------------------------------------------------------------------
    # 5. Filter all outputs to the 09:30 - 09:40 window
    # ------------------------------------------------------------------
    print("\n[5] Filtering outputs to 09:30-09:40 ET window ...")
    snap_a_w = filter_output_window(snap_a)
    snap_b_w = filter_output_window(snap_b)
    snap_c_w = filter_output_window(snap_c)
    snap_d_w = filter_output_window(snap_d)
    flow_a_w = filter_output_window(flow_a)
    flow_b_w = filter_output_window(flow_b)
    flow_c_w = filter_output_window(flow_c)
    flow_d_w = filter_output_window(flow_d)

    print(f"  FULL:  {len(snap_a_w)} snap windows, {len(flow_a_w)} flow rows")
    print(f"  ZERO:  {len(snap_b_w)} snap windows, {len(flow_b_w)} flow rows")
    print(f"  1MIN:  {len(snap_c_w)} snap windows, {len(flow_c_w)} flow rows")
    print(f"  5MIN:  {len(snap_d_w)} snap windows, {len(flow_d_w)} flow rows")

    # ------------------------------------------------------------------
    # 6. Compare snapshots
    # ------------------------------------------------------------------
    print("\n[6] Snapshot comparison (FULL vs ZERO) — first 10 seconds:")
    snap_cmp_zero = compare_snapshots(snap_a_w, snap_b_w, "FULL vs ZERO")
    diff_cols = [c for c in snap_cmp_zero.columns if c.startswith("abs_diff_")]
    early = snap_cmp_zero[snap_cmp_zero["second_offset"] < 10]
    print(early[["second_offset", "window_ts", "merge_status"] + diff_cols].to_string(index=False))

    # Find convergence second for snapshots
    snap_cmp_zero["any_diff"] = snap_cmp_zero[diff_cols].apply(
        lambda r: any(v > 0 for v in r if not np.isnan(v)), axis=1
    )
    divergent_secs = snap_cmp_zero[snap_cmp_zero["any_diff"]]["second_offset"]
    if len(divergent_secs) == 0:
        print("\n  Snapshots IDENTICAL across all 600 seconds.")
    else:
        print(f"\n  Snapshots diverge for seconds: {sorted(divergent_secs.tolist())}")
        print(f"  Last divergent second: {divergent_secs.max()}")

    # ------------------------------------------------------------------
    # 7. Compare depth_flow: FULL vs ZERO
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[7] Depth & Flow comparison: FULL vs ZERO warmup")
    print("=" * 80)
    cmp_zero = compare_depth_flow(flow_a_w, flow_b_w, "FULL vs ZERO")
    _print_convergence(cmp_zero, "FULL vs ZERO")

    # ------------------------------------------------------------------
    # 8. Compare depth_flow: FULL vs 1-MINUTE
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[8] Depth & Flow comparison: FULL vs 1-MINUTE warmup")
    print("=" * 80)
    cmp_1min = compare_depth_flow(flow_a_w, flow_c_w, "FULL vs 1MIN")
    _print_convergence(cmp_1min, "FULL vs 1MIN")

    # ------------------------------------------------------------------
    # 9. Compare depth_flow: FULL vs 5-MINUTE
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[9] Depth & Flow comparison: FULL vs 5-MINUTE warmup")
    print("=" * 80)
    cmp_5min = compare_depth_flow(flow_a_w, flow_d_w, "FULL vs 5MIN")
    _print_convergence(cmp_5min, "FULL vs 5MIN")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[10] CONVERGENCE SUMMARY")
    print("=" * 80)
    for label, cmp_df in [
        ("ZERO warmup", cmp_zero),
        ("1-MINUTE warmup", cmp_1min),
        ("5-MINUTE warmup", cmp_5min),
    ]:
        _print_convergence_summary(cmp_df, label)

    # ------------------------------------------------------------------
    # 11. Relative magnitude analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[11] RELATIVE MAGNITUDE: how large are the depth errors?")
    print("=" * 80)

    # For FULL warmup, compute total depth per second
    total_depth_per_sec = (
        flow_a_w.groupby("window_start_ts_ns")["depth_qty_end"]
        .sum()
        .reset_index()
    )
    total_depth_per_sec["second_offset"] = (
        (total_depth_per_sec["window_start_ts_ns"] - OPEN_0930_NS) // 1_000_000_000
    )

    avg_total_depth = total_depth_per_sec["depth_qty_end"].mean()
    median_total_depth = total_depth_per_sec["depth_qty_end"].median()
    print(f"  FULL warmup total depth_qty_end per window:")
    print(f"    Mean:   {avg_total_depth:,.0f} shares")
    print(f"    Median: {median_total_depth:,.0f} shares")

    # Sum of max_diff across all price levels per second (sum, not max)
    for label, cmp_df in [
        ("ZERO warmup", cmp_zero),
        ("1-MINUTE warmup", cmp_1min),
        ("5-MINUTE warmup", cmp_5min),
    ]:
        # max_diff_depth_qty_end is the max across matched rows for that second
        # We already have this per-second; use it as a rough bound
        worst_end = cmp_df["max_diff_depth_qty_end"].max()
        mean_end = cmp_df["max_diff_depth_qty_end"].mean()
        pct_worst = 100.0 * worst_end / avg_total_depth if avg_total_depth > 0 else 0
        pct_mean = 100.0 * mean_end / avg_total_depth if avg_total_depth > 0 else 0
        print(f"\n  {label}:")
        print(f"    Max per-bucket depth error:  {worst_end:,.0f} shares ({pct_worst:.2f}% of avg total depth)")
        print(f"    Mean per-bucket depth error: {mean_end:,.0f} shares ({pct_mean:.2f}% of avg total depth)")

    # ------------------------------------------------------------------
    # 12. How long do pre-market orders actually survive into RTH?
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[12] PRE-MARKET ORDER SURVIVAL CURVE")
    print("=" * 80)

    # Track when pre-market orders die during RTH
    premarket_oids = set(alive_orders.keys())
    remaining = dict(alive_orders)  # copy
    checkpoints_sec = [0, 1, 2, 5, 10, 30, 60, 120, 300, 600]

    rth_window = df_full[
        (df_full["ts_event"] >= OPEN_0930_NS) & (df_full["ts_event"] < CLOSE_0940_NS)
    ]

    survival: Dict[int, int] = {0: len(remaining)}
    check_idx = 1
    for row in rth_window.itertuples(index=False):
        ts = int(row.ts_event)
        sec_from_open = (ts - OPEN_0930_NS) / 1_000_000_000

        while check_idx < len(checkpoints_sec) and sec_from_open >= checkpoints_sec[check_idx]:
            survival[checkpoints_sec[check_idx]] = len(remaining)
            check_idx += 1

        oid = int(row.order_id)
        if oid not in remaining:
            continue
        action = str(row.action)
        if action == "C":
            remaining.pop(oid, None)
        elif action == "F":
            s, p, q = remaining[oid]
            rem_qty = q - int(row.size)
            if rem_qty > 0:
                remaining[oid] = (s, p, rem_qty)
            else:
                remaining.pop(oid, None)
        elif action == "M":
            s, _, _ = remaining[oid]
            remaining[oid] = (s, int(row.price), int(row.size))

    # Fill any remaining checkpoints
    while check_idx < len(checkpoints_sec):
        survival[checkpoints_sec[check_idx]] = len(remaining)
        check_idx += 1

    print(f"  Pre-market orders alive at each checkpoint:")
    for sec in checkpoints_sec:
        cnt = survival.get(sec, 0)
        pct = 100.0 * cnt / len(alive_orders) if len(alive_orders) > 0 else 0
        print(f"    t+{sec:>3d}s: {cnt:>5,} orders ({pct:>5.1f}%)")

    total_qty_remaining = sum(q for _, _, q in remaining.values())
    print(f"\n  After 600s (end of window): {len(remaining):,} orders, {total_qty_remaining:,} shares still resting")
    print(f"  These orders NEVER converge — they are a permanent bias without warmup.")

    print("\nDone.")


def _print_convergence(cmp_df: pd.DataFrame, label: str) -> None:
    """Print per-second convergence detail for the first 15 seconds."""
    max_diff_cols = [c for c in cmp_df.columns if c.startswith("max_diff_")]
    display_cols = ["second_offset", "rows_base_only", "rows_test_only", "rows_matched"] + max_diff_cols

    print(f"\n  First 15 seconds detail ({label}):")
    early = cmp_df[cmp_df["second_offset"] < 15]
    if early.empty:
        print("  (no data)")
        return

    # Format for readability
    for col in max_diff_cols:
        early[col] = early[col].apply(lambda v: f"{v:.0f}" if v > 0 else "0")
    print(early[display_cols].to_string(index=False))


def _print_convergence_summary(cmp_df: pd.DataFrame, label: str) -> None:
    """Print when each metric converges (max_diff == 0)."""
    max_diff_cols = [c for c in cmp_df.columns if c.startswith("max_diff_")]

    print(f"\n  {label}:")
    has_row_diff = (cmp_df["rows_base_only"] > 0) | (cmp_df["rows_test_only"] > 0)
    if has_row_diff.any():
        last_row_diff = cmp_df[has_row_diff]["second_offset"].max()
        print(f"    Row count diverges until second: {last_row_diff}")
    else:
        print(f"    Row counts: IDENTICAL")

    for col in max_diff_cols:
        metric = col.replace("max_diff_", "")
        nonzero = cmp_df[cmp_df[col] > 0]
        if nonzero.empty:
            print(f"    {metric}: IDENTICAL (converged at t=0)")
        else:
            last_sec = nonzero["second_offset"].max()
            worst = nonzero[col].max()
            print(f"    {metric}: diverges until second {last_sec}, worst={worst:.1f}")


if __name__ == "__main__":
    main()
