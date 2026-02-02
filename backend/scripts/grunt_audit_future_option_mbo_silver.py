#!/usr/bin/env python3
"""
Institution-Grade Silver Layer Audit for FUTURE_OPTION_MBO
Performs comprehensive semantic and statistical validation.

Last Grunted: 02/01/2026 11:00:00 PM EST
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Constants from source code
PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))  # 250_000_000
STRIKE_STEP_POINTS = 5.0
STRIKE_STEP_INT = int(round(STRIKE_STEP_POINTS / PRICE_SCALE))  # 5_000_000_000
MAX_STRIKE_OFFSETS = 10  # +/- $50 around spot
RIGHTS = ("C", "P")
SIDES = ("A", "B")

# Data paths
DT = "2026-01-06"
SNAP_PATH = Path("lake/silver/product_type=future_option_mbo/symbol=ESH6/table=book_snapshot_1s/dt=2026-01-06/part-00000.parquet")
FLOW_PATH = Path("lake/silver/product_type=future_option_mbo/symbol=ESH6/table=depth_and_flow_1s/dt=2026-01-06/part-00000.parquet")


def section(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---\n")


def main() -> None:
    print("=" * 80)
    print("  FUTURE_OPTION_MBO SILVER LAYER AUDIT")
    print(f"  Date: {DT}")
    print("=" * 80)
    
    # =========================================================================
    # SECTION 1: Load and Inspect Data
    # =========================================================================
    section("1. DATA LOADING AND SCHEMA INSPECTION")
    
    if not SNAP_PATH.exists():
        print(f"ERROR: book_snapshot_1s not found at {SNAP_PATH}")
        sys.exit(1)
    if not FLOW_PATH.exists():
        print(f"ERROR: depth_and_flow_1s not found at {FLOW_PATH}")
        sys.exit(1)
    
    df_snap = pd.read_parquet(SNAP_PATH)
    df_flow = pd.read_parquet(FLOW_PATH)
    
    print(f"book_snapshot_1s: {len(df_snap):,} rows, {df_snap.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"depth_and_flow_1s: {len(df_flow):,} rows, {df_flow.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    subsection("book_snapshot_1s Schema")
    print(df_snap.dtypes.to_string())
    
    subsection("depth_and_flow_1s Schema")
    print(df_flow.dtypes.to_string())
    
    subsection("book_snapshot_1s Sample (first 3 rows)")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_snap.head(3).to_string())
    
    subsection("depth_and_flow_1s Sample (first 3 rows)")
    print(df_flow.head(3).to_string())
    
    # =========================================================================
    # SECTION 2: Semantic Analysis
    # =========================================================================
    section("2. SEMANTIC ANALYSIS - BOOK_SNAPSHOT_1S")
    
    issues_snap = []
    issues_flow = []
    
    # window_start_ts_ns, window_end_ts_ns
    subsection("Time Boundaries")
    snap_start_min = df_snap["window_start_ts_ns"].min()
    snap_start_max = df_snap["window_start_ts_ns"].max()
    snap_end_min = df_snap["window_end_ts_ns"].min()
    snap_end_max = df_snap["window_end_ts_ns"].max()
    
    print(f"window_start_ts_ns: min={snap_start_min:,}, max={snap_start_max:,}")
    print(f"window_end_ts_ns:   min={snap_end_min:,}, max={snap_end_max:,}")
    
    # Verify window duration = 1s
    window_durations = df_snap["window_end_ts_ns"] - df_snap["window_start_ts_ns"]
    all_1s = (window_durations == WINDOW_NS).all()
    print(f"All windows = 1 second: {all_1s}")
    if not all_1s:
        issues_snap.append("Some windows != 1 second duration")
    
    # Convert to human time
    first_ts_ns = snap_start_min
    last_ts_ns = snap_end_max
    first_ts = pd.to_datetime(first_ts_ns, unit='ns', utc=True)
    last_ts = pd.to_datetime(last_ts_ns, unit='ns', utc=True)
    print(f"Time range: {first_ts} to {last_ts}")
    
    n_unique_windows = df_snap["window_end_ts_ns"].nunique()
    expected_windows_1hr = 3600
    print(f"Unique windows (1-second): {n_unique_windows}")
    print(f"Expected for 1 hour: {expected_windows_1hr}")
    
    # instrument_id
    subsection("Instrument ID")
    n_instruments = df_snap["instrument_id"].nunique()
    print(f"Unique option instruments: {n_instruments:,}")
    
    # right (C/P)
    subsection("Right (Call/Put)")
    right_counts = df_snap["right"].value_counts()
    print(right_counts.to_string())
    invalid_rights = df_snap[~df_snap["right"].isin(RIGHTS)]
    if len(invalid_rights) > 0:
        issues_snap.append(f"Invalid rights found: {invalid_rights['right'].unique()}")
        print(f"ERROR: Invalid rights found: {invalid_rights['right'].unique()}")
    else:
        print("All rights are valid (C or P)")
    
    # strike_price_int
    subsection("Strike Price")
    strike_min = df_snap["strike_price_int"].min()
    strike_max = df_snap["strike_price_int"].max()
    strike_min_pts = strike_min * PRICE_SCALE
    strike_max_pts = strike_max * PRICE_SCALE
    print(f"strike_price_int: min={strike_min:,} (${strike_min_pts:.2f}), max={strike_max:,} (${strike_max_pts:.2f})")
    
    # Check $5 grid alignment
    strikes = df_snap["strike_price_int"].unique()
    misaligned = [s for s in strikes if s % STRIKE_STEP_INT != 0]
    if misaligned:
        issues_snap.append(f"Strikes not aligned to $5 grid: {len(misaligned)} found")
        print(f"WARNING: {len(misaligned)} strikes not aligned to $5 grid")
    else:
        print("All strikes aligned to $5 grid")
    
    # bid/ask/mid
    subsection("BBO Prices")
    print(f"bid_price_int: min={df_snap['bid_price_int'].min():,}, max={df_snap['bid_price_int'].max():,}")
    print(f"ask_price_int: min={df_snap['ask_price_int'].min():,}, max={df_snap['ask_price_int'].max():,}")
    print(f"mid_price: min={df_snap['mid_price'].min():.6f}, max={df_snap['mid_price'].max():.6f}")
    print(f"mid_price_int: min={df_snap['mid_price_int'].min():,.0f}, max={df_snap['mid_price_int'].max():,.0f}")
    
    # Verify mid_price_int = (bid + ask) / 2
    expected_mid = (df_snap["bid_price_int"] + df_snap["ask_price_int"]) / 2
    mid_diff = (df_snap["mid_price_int"] - expected_mid).abs()
    mid_tolerance = 1  # Allow for rounding
    mid_valid = (mid_diff <= mid_tolerance).all()
    print(f"mid_price_int == (bid + ask) / 2: {mid_valid}")
    if not mid_valid:
        n_invalid = (mid_diff > mid_tolerance).sum()
        issues_snap.append(f"mid_price_int formula violation: {n_invalid:,} rows")
    
    # Verify mid_price = mid_price_int * PRICE_SCALE
    expected_mid_price = df_snap["mid_price_int"] * PRICE_SCALE
    mid_price_diff = (df_snap["mid_price"] - expected_mid_price).abs()
    mid_price_valid = (mid_price_diff < 1e-12).all()
    print(f"mid_price == mid_price_int * 1e-9: {mid_price_valid}")
    if not mid_price_valid:
        n_invalid = (mid_price_diff >= 1e-12).sum()
        issues_snap.append(f"mid_price scaling violation: {n_invalid:,} rows")
    
    # Spread positive when book_valid
    valid_books = df_snap[df_snap["book_valid"]]
    spread = valid_books["ask_price_int"] - valid_books["bid_price_int"]
    positive_spread = (spread > 0).all()
    print(f"Spread positive when book_valid: {positive_spread}")
    if not positive_spread:
        n_invalid = (spread <= 0).sum()
        issues_snap.append(f"Non-positive spread when book_valid: {n_invalid:,} rows")
    
    # spot_ref_price_int
    subsection("Spot Reference Price")
    spot_min = df_snap["spot_ref_price_int"].min()
    spot_max = df_snap["spot_ref_price_int"].max()
    spot_min_pts = spot_min * PRICE_SCALE
    spot_max_pts = spot_max * PRICE_SCALE
    print(f"spot_ref_price_int: min={spot_min:,} (${spot_min_pts:.2f}), max={spot_max:,} (${spot_max_pts:.2f})")
    
    # book_valid
    subsection("Book Validity")
    book_valid_counts = df_snap["book_valid"].value_counts()
    print(book_valid_counts.to_string())
    valid_pct = df_snap["book_valid"].mean() * 100
    print(f"book_valid rate: {valid_pct:.2f}%")
    
    # =========================================================================
    section("3. SEMANTIC ANALYSIS - DEPTH_AND_FLOW_1S")
    # =========================================================================
    
    # Time boundaries
    subsection("Time Boundaries")
    flow_start_min = df_flow["window_start_ts_ns"].min()
    flow_start_max = df_flow["window_start_ts_ns"].max()
    flow_end_min = df_flow["window_end_ts_ns"].min()
    flow_end_max = df_flow["window_end_ts_ns"].max()
    
    print(f"window_start_ts_ns: min={flow_start_min:,}, max={flow_start_max:,}")
    print(f"window_end_ts_ns:   min={flow_end_min:,}, max={flow_end_max:,}")
    
    # Window duration
    flow_durations = df_flow["window_end_ts_ns"] - df_flow["window_start_ts_ns"]
    all_1s_flow = (flow_durations == WINDOW_NS).all()
    print(f"All windows = 1 second: {all_1s_flow}")
    if not all_1s_flow:
        issues_flow.append("Some windows != 1 second duration")
    
    n_unique_windows_flow = df_flow["window_end_ts_ns"].nunique()
    print(f"Unique windows: {n_unique_windows_flow}")
    
    # strike_price_int, strike_points
    subsection("Strike Grid")
    flow_strikes = df_flow["strike_price_int"].unique()
    n_strikes = len(flow_strikes)
    print(f"Unique strikes: {n_strikes}")
    
    # Check $5 grid alignment
    misaligned_flow = [s for s in flow_strikes if s % STRIKE_STEP_INT != 0]
    if misaligned_flow:
        issues_flow.append(f"Strikes not aligned to $5 grid: {len(misaligned_flow)} found")
        print(f"WARNING: {len(misaligned_flow)} strikes not aligned to $5 grid")
    else:
        print("All strikes aligned to $5 grid")
    
    # Verify strike_points = strike_price_int * PRICE_SCALE
    expected_strike_pts = df_flow["strike_price_int"].astype(float) * PRICE_SCALE
    strike_pts_diff = (df_flow["strike_points"] - expected_strike_pts).abs()
    strike_pts_valid = (strike_pts_diff < 1e-12).all()
    print(f"strike_points == strike_price_int * 1e-9: {strike_pts_valid}")
    if not strike_pts_valid:
        n_invalid = (strike_pts_diff >= 1e-12).sum()
        issues_flow.append(f"strike_points scaling violation: {n_invalid:,} rows")
    
    # right (C/P)
    subsection("Right (Call/Put)")
    flow_right_counts = df_flow["right"].value_counts()
    print(flow_right_counts.to_string())
    invalid_rights_flow = df_flow[~df_flow["right"].isin(RIGHTS)]
    if len(invalid_rights_flow) > 0:
        issues_flow.append(f"Invalid rights: {invalid_rights_flow['right'].unique()}")
    
    # side (A/B)
    subsection("Side (Ask/Bid)")
    side_counts = df_flow["side"].value_counts()
    print(side_counts.to_string())
    invalid_sides = df_flow[~df_flow["side"].isin(SIDES)]
    if len(invalid_sides) > 0:
        issues_flow.append(f"Invalid sides: {invalid_sides['side'].unique()}")
    
    # rel_ticks
    subsection("Relative Ticks")
    rel_ticks_min = df_flow["rel_ticks"].min()
    rel_ticks_max = df_flow["rel_ticks"].max()
    print(f"rel_ticks: min={rel_ticks_min}, max={rel_ticks_max}")
    
    # Verify rel_ticks = (strike_price_int - spot_ref_price_int) / TICK_INT
    expected_rel_ticks = (df_flow["strike_price_int"] - df_flow["spot_ref_price_int"]) // TICK_INT
    rel_ticks_diff = (df_flow["rel_ticks"] - expected_rel_ticks).abs()
    rel_ticks_valid = (rel_ticks_diff == 0).all()
    print(f"rel_ticks formula correct: {rel_ticks_valid}")
    if not rel_ticks_valid:
        n_invalid = (rel_ticks_diff != 0).sum()
        issues_flow.append(f"rel_ticks formula violation: {n_invalid:,} rows")
    
    # Check rel_ticks corresponds to offset / 0.25
    # rel_ticks should be integer when (strike - spot) % TICK_INT == 0
    tick_remainder = (df_flow["strike_price_int"] - df_flow["spot_ref_price_int"]) % TICK_INT
    tick_aligned = (tick_remainder == 0).all()
    print(f"All strike-spot differences aligned to $0.25 ticks: {tick_aligned}")
    if not tick_aligned:
        n_misaligned = (tick_remainder != 0).sum()
        issues_flow.append(f"Strike-spot not tick-aligned: {n_misaligned:,} rows")
    
    # spot_ref_price_int
    subsection("Spot Reference Price")
    flow_spot_min = df_flow["spot_ref_price_int"].min()
    flow_spot_max = df_flow["spot_ref_price_int"].max()
    print(f"spot_ref_price_int: min={flow_spot_min:,} (${flow_spot_min * PRICE_SCALE:.2f}), max={flow_spot_max:,} (${flow_spot_max * PRICE_SCALE:.2f})")
    
    # =========================================================================
    section("4. STATISTICAL ANALYSIS - ACCOUNTING IDENTITY")
    # =========================================================================
    
    subsection("Quantity Fields Summary")
    qty_cols = ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "pull_qty_rest", "fill_qty", "depth_qty_rest"]
    for col in qty_cols:
        print(f"{col}: min={df_flow[col].min():.0f}, max={df_flow[col].max():.0f}, mean={df_flow[col].mean():.2f}")
    
    subsection("Non-Negative Quantities Check")
    for col in qty_cols:
        has_negative = (df_flow[col] < 0).any()
        n_negative = (df_flow[col] < 0).sum()
        if has_negative:
            print(f"{col}: FAIL - {n_negative:,} negative values")
            issues_flow.append(f"{col} has {n_negative:,} negative values")
        else:
            print(f"{col}: PASS - all values >= 0")
    
    subsection("Accounting Identity Check")
    # depth_qty_start + add_qty - pull_qty - fill_qty = depth_qty_end
    calculated_end = df_flow["depth_qty_start"] + df_flow["add_qty"] - df_flow["pull_qty"] - df_flow["fill_qty"]
    identity_diff = (calculated_end - df_flow["depth_qty_end"]).abs()
    
    tolerance = 1e-6
    identity_valid = (identity_diff <= tolerance).all()
    
    if identity_valid:
        print("ACCOUNTING IDENTITY: PASS")
        print("  depth_qty_start + add_qty - pull_qty - fill_qty = depth_qty_end")
    else:
        n_violations = (identity_diff > tolerance).sum()
        max_diff = identity_diff.max()
        print(f"ACCOUNTING IDENTITY: FAIL")
        print(f"  Violations: {n_violations:,} rows")
        print(f"  Max difference: {max_diff:.6f}")
        issues_flow.append(f"Accounting identity violated in {n_violations:,} rows")
        
        # Show sample violations
        violation_mask = identity_diff > tolerance
        violations = df_flow[violation_mask].head(5)
        print("\nSample violations:")
        print(violations[["depth_qty_start", "add_qty", "pull_qty", "fill_qty", "depth_qty_end"]].to_string())
        print(f"Calculated end: {calculated_end[violation_mask].head(5).values}")
    
    subsection("depth_qty_rest <= depth_qty_end Check")
    rest_valid = (df_flow["depth_qty_rest"] <= df_flow["depth_qty_end"] + tolerance).all()
    if rest_valid:
        print("PASS: depth_qty_rest <= depth_qty_end")
    else:
        n_violations = (df_flow["depth_qty_rest"] > df_flow["depth_qty_end"] + tolerance).sum()
        print(f"FAIL: depth_qty_rest > depth_qty_end in {n_violations:,} rows")
        issues_flow.append(f"depth_qty_rest > depth_qty_end in {n_violations:,} rows")
    
    # =========================================================================
    section("5. STRIKE GRID ANALYSIS")
    # =========================================================================
    
    subsection("Strike Distribution by Right")
    strike_by_right = df_flow.groupby("right")["strike_price_int"].agg(["nunique", "min", "max"])
    strike_by_right["min_pts"] = strike_by_right["min"] * PRICE_SCALE
    strike_by_right["max_pts"] = strike_by_right["max"] * PRICE_SCALE
    print(strike_by_right.to_string())
    
    subsection("Call vs Put Row Counts")
    call_rows = len(df_flow[df_flow["right"] == "C"])
    put_rows = len(df_flow[df_flow["right"] == "P"])
    print(f"Call rows: {call_rows:,}")
    print(f"Put rows: {put_rows:,}")
    print(f"Ratio (Call/Put): {call_rows / put_rows:.2f}" if put_rows > 0 else "No put rows")
    
    subsection("Strike Offsets from ATM")
    # Group by window and count unique strikes
    strikes_per_window = df_flow.groupby("window_end_ts_ns")["strike_price_int"].nunique()
    expected_strikes = 2 * MAX_STRIKE_OFFSETS + 1  # 21 strikes total
    print(f"Expected strikes per window: {expected_strikes} (21)")
    print(f"Actual strikes per window: min={strikes_per_window.min()}, max={strikes_per_window.max()}, mean={strikes_per_window.mean():.1f}")
    
    subsection("Rows per Strike-Right-Side-Window")
    # Should be 1 row per unique (window, strike, right, side) combination
    duplicates = df_flow.groupby(["window_end_ts_ns", "strike_price_int", "right", "side"]).size()
    if (duplicates > 1).any():
        n_dups = (duplicates > 1).sum()
        print(f"WARNING: {n_dups} duplicate (window, strike, right, side) combinations")
        issues_flow.append(f"Duplicate rows: {n_dups}")
    else:
        print("PASS: No duplicate (window, strike, right, side) combinations")
    
    # =========================================================================
    section("6. DATA QUALITY REPORT")
    # =========================================================================
    
    subsection("book_snapshot_1s Quality")
    print(f"Total rows: {len(df_snap):,}")
    print(f"book_valid=True: {df_snap['book_valid'].sum():,} ({df_snap['book_valid'].mean()*100:.1f}%)")
    print(f"book_valid=False: {(~df_snap['book_valid']).sum():,} ({(~df_snap['book_valid']).mean()*100:.1f}%)")
    print(f"Unique instruments: {df_snap['instrument_id'].nunique():,}")
    print(f"Unique windows: {df_snap['window_end_ts_ns'].nunique():,}")
    
    subsection("depth_and_flow_1s Quality")
    print(f"Total rows: {len(df_flow):,}")
    print(f"window_valid=True: {df_flow['window_valid'].sum():,} ({df_flow['window_valid'].mean()*100:.1f}%)")
    print(f"window_valid=False: {(~df_flow['window_valid']).sum():,} ({(~df_flow['window_valid']).mean()*100:.1f}%)")
    print(f"Unique strikes: {df_flow['strike_price_int'].nunique()}")
    print(f"Unique windows: {df_flow['window_end_ts_ns'].nunique()}")
    
    subsection("Activity Analysis")
    # Windows with actual trading activity
    active_windows = df_flow[(df_flow["add_qty"] > 0) | (df_flow["pull_qty"] > 0) | (df_flow["fill_qty"] > 0)]
    print(f"Rows with activity (add/pull/fill > 0): {len(active_windows):,} ({len(active_windows)/len(df_flow)*100:.1f}%)")
    
    fills_only = df_flow[df_flow["fill_qty"] > 0]
    print(f"Rows with fills: {len(fills_only):,}")
    print(f"Total fill_qty: {df_flow['fill_qty'].sum():,.0f}")
    
    # =========================================================================
    section("7. ISSUES SUMMARY")
    # =========================================================================
    
    print("book_snapshot_1s Issues:")
    if issues_snap:
        for i, issue in enumerate(issues_snap, 1):
            print(f"  {i}. {issue}")
    else:
        print("  None")
    
    print("\ndepth_and_flow_1s Issues:")
    if issues_flow:
        for i, issue in enumerate(issues_flow, 1):
            print(f"  {i}. {issue}")
    else:
        print("  None")
    
    # =========================================================================
    section("8. FINAL GRADE")
    # =========================================================================
    
    total_issues = len(issues_snap) + len(issues_flow)
    
    if total_issues == 0:
        grade = "A"
        assessment = "Excellent - All checks passed"
    elif total_issues <= 2:
        grade = "B"
        assessment = "Good - Minor issues detected"
    elif total_issues <= 5:
        grade = "C"
        assessment = "Acceptable - Moderate issues require attention"
    elif total_issues <= 10:
        grade = "D"
        assessment = "Poor - Significant issues need resolution"
    else:
        grade = "F"
        assessment = "Failing - Critical issues in data quality"
    
    print(f"GRADE: {grade}")
    print(f"Assessment: {assessment}")
    print(f"Total Issues: {total_issues}")
    
    print("\n" + "=" * 80)
    print("  AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
