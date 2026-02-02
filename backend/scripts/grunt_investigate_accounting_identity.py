#!/usr/bin/env python3
"""
Deep investigation of accounting identity violations in future_option_mbo depth_and_flow_1s.

Last Grunted: 02/01/2026 11:15:00 PM EST
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
TICK_INT = 250_000_000
STRIKE_STEP_INT = 5_000_000_000

DT = "2026-01-06"
FLOW_PATH = Path("lake/silver/product_type=future_option_mbo/symbol=ESH6/table=depth_and_flow_1s/dt=2026-01-06/part-00000.parquet")


def main() -> None:
    print("=" * 80)
    print("  ACCOUNTING IDENTITY INVESTIGATION")
    print("=" * 80)
    
    df = pd.read_parquet(FLOW_PATH)
    print(f"\nTotal rows: {len(df):,}")
    
    # Calculate expected end
    df["calc_end"] = df["depth_qty_start"] + df["add_qty"] - df["pull_qty"] - df["fill_qty"]
    df["identity_diff"] = (df["calc_end"] - df["depth_qty_end"]).abs()
    df["identity_ok"] = df["identity_diff"] < 0.001
    
    print(f"Identity violations: {(~df['identity_ok']).sum():,}")
    
    # Analyze violations
    violations = df[~df["identity_ok"]].copy()
    
    print("\n--- Violation Characteristics ---")
    print(f"Mean diff: {violations['identity_diff'].mean():.2f}")
    print(f"Median diff: {violations['identity_diff'].median():.2f}")
    print(f"Max diff: {violations['identity_diff'].max():.2f}")
    
    # Direction of violations
    violations["diff_signed"] = violations["calc_end"] - violations["depth_qty_end"]
    over_estimate = (violations["diff_signed"] > 0).sum()
    under_estimate = (violations["diff_signed"] < 0).sum()
    print(f"\nOver-estimates (calc > actual): {over_estimate:,}")
    print(f"Under-estimates (calc < actual): {under_estimate:,}")
    
    # By right
    print("\n--- Violations by Right ---")
    viol_by_right = violations.groupby("right").size()
    print(viol_by_right.to_string())
    
    # By side
    print("\n--- Violations by Side ---")
    viol_by_side = violations.groupby("side").size()
    print(viol_by_side.to_string())
    
    # Check if violations correlate with activity levels
    print("\n--- Violations by Activity Level ---")
    violations["total_flow"] = violations["add_qty"] + violations["pull_qty"] + violations["fill_qty"]
    df["total_flow"] = df["add_qty"] + df["pull_qty"] + df["fill_qty"]
    
    # Correlation
    print(f"Mean total_flow (violations): {violations['total_flow'].mean():.2f}")
    print(f"Mean total_flow (all rows): {df['total_flow'].mean():.2f}")
    
    # Do violations occur more with active flow?
    active = df[df["total_flow"] > 0]
    active_violations = active[~active["identity_ok"]]
    print(f"\nActive rows: {len(active):,}")
    print(f"Active violations: {len(active_violations):,} ({len(active_violations)/len(active)*100:.1f}%)")
    
    # Zero flow rows
    zero_flow = df[df["total_flow"] == 0]
    zero_violations = zero_flow[~zero_flow["identity_ok"]]
    print(f"\nZero-flow rows: {len(zero_flow):,}")
    print(f"Zero-flow violations: {len(zero_violations):,} ({len(zero_violations)/len(zero_flow)*100:.2f}%)")
    
    # Check depth_qty_start consistency
    print("\n--- Depth Start Analysis ---")
    # When there's no activity, depth_qty_end should equal depth_qty_start
    no_activity = df[(df["add_qty"] == 0) & (df["pull_qty"] == 0) & (df["fill_qty"] == 0)]
    start_end_diff = (no_activity["depth_qty_start"] - no_activity["depth_qty_end"]).abs()
    no_activity_drift = (start_end_diff > 0.001).sum()
    print(f"Rows with no activity: {len(no_activity):,}")
    print(f"Rows where start != end (no activity): {no_activity_drift:,}")
    
    # Sample of no-activity rows where start != end
    if no_activity_drift > 0:
        drift_rows = no_activity[start_end_diff > 0.001].head(10)
        print("\nSample no-activity drift:")
        print(drift_rows[["window_end_ts_ns", "strike_price_int", "right", "side", 
                         "depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty"]].to_string())
    
    # Cross-window depth consistency
    print("\n--- Cross-Window Depth Consistency ---")
    # Sort by window_end_ts_ns, strike_price_int, right, side
    df_sorted = df.sort_values(["strike_price_int", "right", "side", "window_end_ts_ns"]).copy()
    
    # For each (strike, right, side), depth_qty_end of window N should equal depth_qty_start of window N+1
    # (if windows are consecutive)
    
    # Group and check
    sample_group = df_sorted[(df_sorted["strike_price_int"] == df_sorted["strike_price_int"].iloc[0]) & 
                             (df_sorted["right"] == "C") & 
                             (df_sorted["side"] == "A")].head(20)
    
    print("\nSample sequential windows (C, A, first strike):")
    print(sample_group[["window_end_ts_ns", "depth_qty_start", "depth_qty_end", 
                        "add_qty", "pull_qty", "fill_qty"]].to_string())
    
    # Check consecutive window transitions
    df_sorted["prev_end"] = df_sorted.groupby(["strike_price_int", "right", "side"])["depth_qty_end"].shift(1)
    df_sorted["prev_window"] = df_sorted.groupby(["strike_price_int", "right", "side"])["window_end_ts_ns"].shift(1)
    df_sorted["is_consecutive"] = (df_sorted["window_end_ts_ns"] - df_sorted["prev_window"]) == WINDOW_NS
    
    consecutive = df_sorted[df_sorted["is_consecutive"] == True].copy()
    consecutive["start_vs_prev_end"] = (consecutive["depth_qty_start"] - consecutive["prev_end"]).abs()
    
    mismatches = consecutive[consecutive["start_vs_prev_end"] > 0.001]
    print(f"\nConsecutive window pairs: {len(consecutive):,}")
    print(f"Mismatches (start != prev_end): {len(mismatches):,} ({len(mismatches)/len(consecutive)*100:.2f}%)")
    
    if len(mismatches) > 0:
        print("\nSample mismatches (prev_end → current_start):")
        sample = mismatches.head(5)
        print(sample[["window_end_ts_ns", "strike_price_int", "right", "side", 
                     "prev_end", "depth_qty_start", "depth_qty_end"]].to_string())
    
    # depth_qty_rest > depth_qty_end investigation
    print("\n" + "=" * 80)
    print("  DEPTH_QTY_REST > DEPTH_QTY_END INVESTIGATION")
    print("=" * 80)
    
    rest_violations = df[df["depth_qty_rest"] > df["depth_qty_end"]].copy()
    print(f"\nViolations: {len(rest_violations):,}")
    
    rest_violations["rest_excess"] = rest_violations["depth_qty_rest"] - rest_violations["depth_qty_end"]
    print(f"Mean excess: {rest_violations['rest_excess'].mean():.2f}")
    print(f"Max excess: {rest_violations['rest_excess'].max():.2f}")
    
    print("\nSample violations:")
    sample = rest_violations.head(10)
    print(sample[["window_end_ts_ns", "strike_price_int", "right", "side",
                  "depth_qty_start", "depth_qty_end", "depth_qty_rest", "rest_excess"]].to_string())
    
    # Do rest violations correlate with identity violations?
    both_violations = df[(~df["identity_ok"]) & (df["depth_qty_rest"] > df["depth_qty_end"])]
    print(f"\nRows with BOTH violations: {len(both_violations):,}")
    
    print("\n" + "=" * 80)
    print("  ROOT CAUSE HYPOTHESIS")
    print("=" * 80)
    
    print("""
The accounting identity violation likely stems from the aggregation process:

1. The engine tracks depth at the (instrument_id, side, price_int) level
2. The silver layer aggregates to (strike_price_int, right, side) level
3. Multiple option instruments can have the same strike_price_int

The aggregation in compute_book_states_1s.py sums across instruments:
- depth_qty_end = sum(depth_total)
- depth_qty_start = sum(depth_start) 

But if depth_start is captured at the BEGINNING of the window for ALL instruments,
and then some instruments were created/destroyed during the window, the accounting
identity won't hold at the aggregate level.

Recommendation: This is an inherent limitation of strike-level aggregation for options.
The accounting identity holds per-(instrument, price) but not per-strike.

For institution-grade quality, the documentation should clarify this limitation,
or the silver layer should output per-instrument data.
""")


if __name__ == "__main__":
    main()
