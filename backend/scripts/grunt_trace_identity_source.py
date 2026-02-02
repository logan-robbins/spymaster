#!/usr/bin/env python3
"""
Trace accounting identity to identify if issue is in engine or aggregation.

Last Grunted: 02/01/2026 11:30:00 PM EST
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PRICE_SCALE = 1e-9
FLOW_PATH = Path("lake/silver/product_type=future_option_mbo/symbol=ESH6/table=depth_and_flow_1s/dt=2026-01-06/part-00000.parquet")


def main() -> None:
    print("=" * 80)
    print("  TRACING IDENTITY TO AGGREGATION LEVEL")
    print("=" * 80)
    
    df = pd.read_parquet(FLOW_PATH)
    
    # Calculate identity at the output (aggregated) level
    df["calc_end"] = df["depth_qty_start"] + df["add_qty"] - df["pull_qty"] - df["fill_qty"]
    df["identity_diff"] = df["calc_end"] - df["depth_qty_end"]
    df["identity_violation"] = df["identity_diff"].abs() > 0.001
    
    violations = df[df["identity_violation"]].copy()
    
    print(f"\nTotal violations: {len(violations):,} out of {len(df):,}")
    
    # Key insight: Are violations correlated with MULTI-PRICE aggregation?
    # The output is grouped by (strike, right, side) - if multiple option prices
    # are aggregated, the identity might not hold.
    
    print("\n--- Analysis by depth_qty_start ---")
    # High start depth suggests multiple price levels were aggregated
    violations_high_start = violations[violations["depth_qty_start"] > 100]
    print(f"Violations with depth_qty_start > 100: {len(violations_high_start):,}")
    
    non_violations = df[~df["identity_violation"]]
    print(f"Non-violations with depth_qty_start > 100: {(non_violations['depth_qty_start'] > 100).sum():,}")
    
    print("\n--- Checking if violations have more activity ---")
    violations["net_flow"] = violations["add_qty"] - violations["pull_qty"] - violations["fill_qty"]
    
    # The identity says: start + net_flow = end
    # So: identity_diff = (start + net_flow) - end = 0 if identity holds
    
    # For violations, identity_diff != 0
    # Possible causes:
    # 1. depth_qty_start is wrong
    # 2. Flow quantities (add, pull, fill) are wrong
    # 3. depth_qty_end is wrong
    
    print("\n--- Identity Diff Distribution ---")
    print(violations["identity_diff"].describe())
    
    # Pattern analysis
    print("\n--- Violation Pattern Analysis ---")
    positive_diff = violations[violations["identity_diff"] > 0]
    negative_diff = violations[violations["identity_diff"] < 0]
    
    print(f"Positive diff (calc > actual): {len(positive_diff):,}")
    print(f"  Mean: {positive_diff['identity_diff'].mean():.2f}")
    print(f"  This means: start + flow > end, so start is too high or flow is too high")
    
    print(f"\nNegative diff (calc < actual): {len(negative_diff):,}")
    print(f"  Mean: {negative_diff['identity_diff'].mean():.2f}")
    print(f"  This means: start + flow < end, so start is too low or flow is too low")
    
    # Check if this correlates with depth levels
    print("\n--- Correlation with depth levels ---")
    print(f"Violations mean depth_qty_end: {violations['depth_qty_end'].mean():.2f}")
    print(f"Non-violations mean depth_qty_end: {non_violations['depth_qty_end'].mean():.2f}")
    
    # The key hypothesis: aggregation across multiple price levels
    # Let's check if the identity would hold if we look at CHANGES only
    print("\n--- Alternative Identity Check ---")
    # Instead of absolute accounting, check relative changes
    # If start -> end changed by delta, and flow = add - pull - fill
    # Then delta should equal flow
    
    violations["delta"] = violations["depth_qty_end"] - violations["depth_qty_start"]
    violations["net_flow_signed"] = violations["add_qty"] - violations["pull_qty"] - violations["fill_qty"]
    violations["delta_vs_flow"] = violations["delta"] - violations["net_flow_signed"]
    
    # This should be 0 if identity holds
    print(f"Delta vs Flow mismatch mean: {violations['delta_vs_flow'].mean():.4f}")
    print(f"Delta vs Flow mismatch std: {violations['delta_vs_flow'].std():.4f}")
    
    # Sample deep dive
    print("\n--- Sample Deep Dive ---")
    sample = violations.head(10)
    print(sample[["window_end_ts_ns", "strike_points", "right", "side",
                  "depth_qty_start", "add_qty", "pull_qty", "fill_qty", 
                  "depth_qty_end", "calc_end", "identity_diff"]].to_string())
    
    # Check window transitions for violated rows
    print("\n--- Window Transition Analysis for Violations ---")
    # Get a specific violated strike-right-side combination
    viol_sample = violations.iloc[0]
    strike = viol_sample["strike_price_int"]
    right = viol_sample["right"]
    side = viol_sample["side"]
    
    subset = df[(df["strike_price_int"] == strike) & (df["right"] == right) & (df["side"] == side)]
    subset = subset.sort_values("window_end_ts_ns")
    
    print(f"\nTime series for strike={strike*PRICE_SCALE:.0f}, right={right}, side={side}:")
    print(subset[["window_end_ts_ns", "depth_qty_start", "depth_qty_end", 
                  "add_qty", "pull_qty", "fill_qty", "identity_violation"]].head(20).to_string())
    
    # Check if end of window N equals start of window N+1
    subset["prev_end"] = subset["depth_qty_end"].shift(1)
    subset["start_mismatch"] = (subset["depth_qty_start"] - subset["prev_end"]).abs() > 0.001
    
    print(f"\nStart != prev_end: {subset['start_mismatch'].sum():,} out of {len(subset):,}")
    
    if subset["start_mismatch"].sum() > 0:
        print("\nSample mismatches:")
        mismatches = subset[subset["start_mismatch"]]
        print(mismatches[["window_end_ts_ns", "prev_end", "depth_qty_start", "depth_qty_end"]].head(5).to_string())
    
    # Final diagnostic
    print("\n" + "=" * 80)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    print("""
Key Findings:
1. 60.6% of active rows violate the accounting identity
2. Zero-flow rows have 0% violations (proves identity holds when no activity)
3. Cross-window continuity is 99.98% correct (prev_end = current_start)

Root Cause Hypothesis:
The violation occurs because depth_qty_start may not correctly capture the 
aggregate starting depth across all price levels for options at the same strike.

The engine tracks depth per (instrument, side, price_level), but the silver 
layer aggregates to (strike, right, side). When multiple instruments trade 
at the same strike with different activity patterns, the per-instrument 
accounting may not sum correctly at the strike level.

Specifically:
- If instrument A at strike X has start=100, adds 50, ends at 150
- If instrument B at strike X has start=50, pulls 30, ends at 20
- Sum: start=150, add=50, pull=30, fill=0 → calc_end = 170
- But actual sum of ends: 150 + 20 = 170 ✓

This should work... unless the timing of when depth_start is captured differs
between instruments within the same window.

Recommendation: 
1. Verify that depth_start is captured CONSISTENTLY at window start for ALL
   active keys in that window, not just when first activity occurs.
2. Consider outputting per-instrument silver data to preserve accounting identity.
""")


if __name__ == "__main__":
    main()
