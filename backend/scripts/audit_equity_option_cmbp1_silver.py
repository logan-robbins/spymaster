#!/usr/bin/env python3
"""
Institution-Grade Silver Layer Audit for EQUITY_OPTION_CMBP_1
Performs semantic and statistical validation of book_snapshot_1s and depth_and_flow_1s
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LAKE_ROOT = Path(__file__).parent.parent / "lake"
SILVER_PATH = LAKE_ROOT / "silver" / "product_type=equity_option_cmbp_1" / "symbol=QQQ"
BOOK_SNAP_PATH = SILVER_PATH / "table=book_snapshot_1s" / "dt=2026-01-08" / "part-00000.parquet"
DEPTH_FLOW_PATH = SILVER_PATH / "table=depth_and_flow_1s" / "dt=2026-01-08" / "part-00000.parquet"

PRICE_SCALE = 1e-9


def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load both parquet files."""
    df_snap = None
    df_flow = None
    
    if BOOK_SNAP_PATH.exists():
        df_snap = pd.read_parquet(BOOK_SNAP_PATH)
        logger.info(f"Loaded book_snapshot_1s: {len(df_snap):,} rows")
    else:
        logger.error(f"Missing: {BOOK_SNAP_PATH}")
        
    if DEPTH_FLOW_PATH.exists():
        df_flow = pd.read_parquet(DEPTH_FLOW_PATH)
        logger.info(f"Loaded depth_and_flow_1s: {len(df_flow):,} rows")
    else:
        logger.error(f"Missing: {DEPTH_FLOW_PATH}")
        
    return df_snap, df_flow


def analyze_schema(df: pd.DataFrame, name: str) -> dict:
    """Analyze schema and basic statistics."""
    print(f"\n{'='*80}")
    print(f"SCHEMA ANALYSIS: {name}")
    print(f"{'='*80}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn Details:")
    print("-" * 80)
    
    schema_info = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        unique_count = df[col].nunique()
        
        schema_info[col] = {
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
        }
        
        print(f"  {col:30} | {dtype:15} | nulls: {null_count:8,} ({null_pct:5.2f}%) | unique: {unique_count:10,}")
    
    return schema_info


def analyze_book_snapshot(df: pd.DataFrame) -> dict:
    """Semantic analysis of book_snapshot_1s."""
    print(f"\n{'='*80}")
    print("SEMANTIC ANALYSIS: book_snapshot_1s")
    print(f"{'='*80}")
    
    issues = []
    stats = {}
    
    # 1. Time boundaries
    print("\n1. TIME BOUNDARIES")
    print("-" * 40)
    if "window_start_ts_ns" in df.columns and "window_end_ts_ns" in df.columns:
        start_min = pd.to_datetime(df["window_start_ts_ns"].min(), unit="ns", utc=True)
        start_max = pd.to_datetime(df["window_start_ts_ns"].max(), unit="ns", utc=True)
        end_min = pd.to_datetime(df["window_end_ts_ns"].min(), unit="ns", utc=True)
        end_max = pd.to_datetime(df["window_end_ts_ns"].max(), unit="ns", utc=True)
        
        print(f"  window_start_ts_ns: {start_min} to {start_max}")
        print(f"  window_end_ts_ns:   {end_min} to {end_max}")
        
        # Check window duration
        window_durations = df["window_end_ts_ns"] - df["window_start_ts_ns"]
        unique_durations = window_durations.unique()
        print(f"  Window durations (ns): {sorted(unique_durations)}")
        if len(unique_durations) == 1 and unique_durations[0] == 1_000_000_000:
            print("  ✓ All windows are exactly 1 second")
        else:
            issues.append("Window durations not all 1 second")
            print("  ✗ Window duration issue detected")
    
    # 2. Instrument IDs
    print("\n2. INSTRUMENT_ID")
    print("-" * 40)
    if "instrument_id" in df.columns:
        n_instruments = df["instrument_id"].nunique()
        print(f"  Unique instruments: {n_instruments:,}")
        stats["n_instruments"] = n_instruments
    
    # 3. Right (C/P)
    print("\n3. RIGHT (C/P)")
    print("-" * 40)
    if "right" in df.columns:
        right_counts = df["right"].value_counts()
        print(f"  Distribution:")
        for r, c in right_counts.items():
            print(f"    {r}: {c:,} ({c/len(df)*100:.2f}%)")
        
        invalid_rights = df[~df["right"].isin(["C", "P"])]
        if len(invalid_rights) > 0:
            issues.append(f"Invalid right values: {invalid_rights['right'].unique().tolist()}")
            print(f"  ✗ Invalid rights found: {invalid_rights['right'].unique().tolist()}")
        else:
            print("  ✓ All rights are C or P")
    
    # 4. Strike price
    print("\n4. STRIKE_PRICE_INT")
    print("-" * 40)
    if "strike_price_int" in df.columns:
        strike_min = df["strike_price_int"].min() * PRICE_SCALE
        strike_max = df["strike_price_int"].max() * PRICE_SCALE
        strike_unique = df["strike_price_int"].nunique()
        print(f"  Range: ${strike_min:.2f} to ${strike_max:.2f}")
        print(f"  Unique strikes: {strike_unique}")
        
        # Check $1 grid alignment
        strikes_dollars = df["strike_price_int"].unique() * PRICE_SCALE
        non_integer = [s for s in strikes_dollars if not np.isclose(s % 1.0, 0.0) and not np.isclose(s % 1.0, 1.0)]
        if len(non_integer) > 0:
            issues.append(f"Strikes not on $1 grid: {non_integer[:5]}...")
            print(f"  ✗ Non-integer strikes found: {non_integer[:5]}")
        else:
            print("  ✓ All strikes on integer dollar grid")
    
    # 5. BBO prices
    print("\n5. BID/ASK PRICES")
    print("-" * 40)
    if "bid_price_int" in df.columns and "ask_price_int" in df.columns:
        bid_min = df["bid_price_int"].min() * PRICE_SCALE
        bid_max = df["bid_price_int"].max() * PRICE_SCALE
        ask_min = df["ask_price_int"].min() * PRICE_SCALE
        ask_max = df["ask_price_int"].max() * PRICE_SCALE
        
        print(f"  bid_price_int: ${bid_min:.4f} to ${bid_max:.4f}")
        print(f"  ask_price_int: ${ask_min:.4f} to ${ask_max:.4f}")
        
        # Check spread
        spread = df["ask_price_int"] - df["bid_price_int"]
        negative_spreads = (spread < 0).sum()
        zero_spreads = (spread == 0).sum()
        
        print(f"  Negative spreads: {negative_spreads:,}")
        print(f"  Zero spreads: {zero_spreads:,}")
        
        if negative_spreads > 0:
            issues.append(f"Negative spreads: {negative_spreads}")
            print("  ✗ Crossed markets detected")
        else:
            print("  ✓ No crossed markets")
    
    # 6. Mid price validation
    print("\n6. MID_PRICE VALIDATION")
    print("-" * 40)
    if all(c in df.columns for c in ["mid_price", "mid_price_int", "bid_price_int", "ask_price_int"]):
        expected_mid_int = (df["bid_price_int"] + df["ask_price_int"]) / 2
        mid_int_match = np.allclose(df["mid_price_int"], expected_mid_int, rtol=1e-9)
        
        expected_mid = df["mid_price_int"] * PRICE_SCALE
        mid_match = np.allclose(df["mid_price"], expected_mid, rtol=1e-9)
        
        print(f"  mid_price_int == (bid + ask) / 2: {mid_int_match}")
        print(f"  mid_price == mid_price_int * PRICE_SCALE: {mid_match}")
        
        if not mid_int_match:
            diff = (df["mid_price_int"] - expected_mid_int).abs()
            issues.append(f"mid_price_int mismatch, max diff: {diff.max()}")
            print(f"  ✗ mid_price_int max diff: {diff.max()}")
        if not mid_match:
            diff = (df["mid_price"] - expected_mid).abs()
            issues.append(f"mid_price mismatch, max diff: {diff.max()}")
            print(f"  ✗ mid_price max diff: {diff.max()}")
    
    # 7. Spot reference
    print("\n7. SPOT_REF_PRICE_INT")
    print("-" * 40)
    if "spot_ref_price_int" in df.columns:
        spot_min = df["spot_ref_price_int"].min() * PRICE_SCALE
        spot_max = df["spot_ref_price_int"].max() * PRICE_SCALE
        spot_unique = df["spot_ref_price_int"].nunique()
        
        print(f"  Range: ${spot_min:.2f} to ${spot_max:.2f}")
        print(f"  Unique values: {spot_unique}")
        
        zero_spots = (df["spot_ref_price_int"] == 0).sum()
        if zero_spots > 0:
            issues.append(f"Zero spot reference: {zero_spots}")
            print(f"  ✗ Zero spot refs: {zero_spots}")
        else:
            print("  ✓ All spot references non-zero")
    
    # 8. book_valid
    print("\n8. BOOK_VALID")
    print("-" * 40)
    if "book_valid" in df.columns:
        valid_counts = df["book_valid"].value_counts()
        for v, c in valid_counts.items():
            print(f"  {v}: {c:,} ({c/len(df)*100:.2f}%)")
        stats["book_valid_pct"] = (df["book_valid"].sum() / len(df)) * 100 if len(df) > 0 else 0
    
    return {"issues": issues, "stats": stats}


def analyze_depth_and_flow(df: pd.DataFrame) -> dict:
    """Semantic analysis of depth_and_flow_1s."""
    print(f"\n{'='*80}")
    print("SEMANTIC ANALYSIS: depth_and_flow_1s")
    print(f"{'='*80}")
    
    issues = []
    stats = {}
    
    # 1. Time boundaries
    print("\n1. TIME BOUNDARIES")
    print("-" * 40)
    if "window_start_ts_ns" in df.columns and "window_end_ts_ns" in df.columns:
        start_min = pd.to_datetime(df["window_start_ts_ns"].min(), unit="ns", utc=True)
        start_max = pd.to_datetime(df["window_start_ts_ns"].max(), unit="ns", utc=True)
        end_min = pd.to_datetime(df["window_end_ts_ns"].min(), unit="ns", utc=True)
        end_max = pd.to_datetime(df["window_end_ts_ns"].max(), unit="ns", utc=True)
        
        print(f"  window_start_ts_ns: {start_min} to {start_max}")
        print(f"  window_end_ts_ns:   {end_min} to {end_max}")
    
    # 2. Strike price / strike_points
    print("\n2. STRIKE_PRICE_INT / STRIKE_POINTS")
    print("-" * 40)
    if "strike_price_int" in df.columns:
        strike_min = df["strike_price_int"].min() * PRICE_SCALE
        strike_max = df["strike_price_int"].max() * PRICE_SCALE
        print(f"  strike_price_int range: ${strike_min:.2f} to ${strike_max:.2f}")
    
    if "strike_points" in df.columns:
        sp_min = df["strike_points"].min()
        sp_max = df["strike_points"].max()
        print(f"  strike_points range: ${sp_min:.2f} to ${sp_max:.2f}")
    
    # 3. Right (C/P)
    print("\n3. RIGHT (C/P)")
    print("-" * 40)
    if "right" in df.columns:
        right_counts = df["right"].value_counts()
        for r, c in right_counts.items():
            print(f"  {r}: {c:,} ({c/len(df)*100:.2f}%)")
        
        invalid_rights = df[~df["right"].isin(["C", "P"])]
        if len(invalid_rights) > 0:
            issues.append(f"Invalid right values: {invalid_rights['right'].unique().tolist()}")
        else:
            print("  ✓ All rights are C or P")
    
    # 4. Side (A/B)
    print("\n4. SIDE (A/B)")
    print("-" * 40)
    if "side" in df.columns:
        side_counts = df["side"].value_counts()
        for s, c in side_counts.items():
            print(f"  {s}: {c:,} ({c/len(df)*100:.2f}%)")
        
        invalid_sides = df[~df["side"].isin(["A", "B"])]
        if len(invalid_sides) > 0:
            issues.append(f"Invalid side values: {invalid_sides['side'].unique().tolist()}")
        else:
            print("  ✓ All sides are A or B")
    
    # 5. Spot reference
    print("\n5. SPOT_REF_PRICE_INT")
    print("-" * 40)
    if "spot_ref_price_int" in df.columns:
        spot_min = df["spot_ref_price_int"].min() * PRICE_SCALE
        spot_max = df["spot_ref_price_int"].max() * PRICE_SCALE
        print(f"  Range: ${spot_min:.2f} to ${spot_max:.2f}")
    
    # 6. REL_TICKS (critical for CMBP-1)
    print("\n6. REL_TICKS (Strike offset)")
    print("-" * 40)
    if "rel_ticks" in df.columns:
        rel_min = df["rel_ticks"].min()
        rel_max = df["rel_ticks"].max()
        print(f"  Range: {rel_min} to {rel_max}")
        
        # Check even alignment (must be multiples of 2 for $1 grid)
        odd_ticks = (df["rel_ticks"] % 2 != 0).sum()
        if odd_ticks > 0:
            issues.append(f"Odd rel_ticks (not on $1 grid): {odd_ticks}")
            print(f"  ✗ Odd rel_ticks: {odd_ticks}")
        else:
            print("  ✓ All rel_ticks are even (correct $1 grid alignment)")
        
        # Check expected range (-50 to +50 for +/- $25)
        out_of_range = ((df["rel_ticks"] < -50) | (df["rel_ticks"] > 50)).sum()
        if out_of_range > 0:
            issues.append(f"rel_ticks out of [-50, 50] range: {out_of_range}")
            print(f"  ✗ Out of range: {out_of_range}")
        else:
            print("  ✓ All rel_ticks within [-50, 50] range")
    
    # 7. Depth quantities
    print("\n7. DEPTH QUANTITIES")
    print("-" * 40)
    for col in ["depth_qty_start", "depth_qty_end"]:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            col_sum = df[col].sum()
            negative_count = (df[col] < 0).sum()
            
            print(f"  {col}: min={col_min:.2f}, max={col_max:.2f}, sum={col_sum:.0f}")
            if negative_count > 0:
                issues.append(f"Negative {col}: {negative_count}")
                print(f"    ✗ Negative values: {negative_count}")
            else:
                print(f"    ✓ All values >= 0")
    
    # 8. Flow quantities (add_qty, pull_qty)
    print("\n8. FLOW QUANTITIES")
    print("-" * 40)
    for col in ["add_qty", "pull_qty"]:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            col_sum = df[col].sum()
            negative_count = (df[col] < 0).sum()
            
            print(f"  {col}: min={col_min:.2f}, max={col_max:.2f}, sum={col_sum:.0f}")
            if negative_count > 0:
                issues.append(f"Negative {col}: {negative_count}")
                print(f"    ✗ Negative values: {negative_count}")
            else:
                print(f"    ✓ All values >= 0")
    
    # 9. CMBP-1 specific: fill_qty and pull_qty_rest should be 0
    print("\n9. CMBP-1 LIMITATIONS (fill_qty, pull_qty_rest should be 0)")
    print("-" * 40)
    for col in ["fill_qty", "pull_qty_rest"]:
        if col in df.columns:
            non_zero = (df[col] != 0).sum()
            col_sum = df[col].sum()
            
            print(f"  {col}: non-zero count={non_zero}, sum={col_sum}")
            if non_zero > 0:
                issues.append(f"Non-zero {col} (should be 0 for CMBP-1): {non_zero}")
                print(f"    ✗ Non-zero values found (CMBP-1 limitation violated)")
            else:
                print(f"    ✓ All values are 0 (correct for CMBP-1)")
    
    # 10. Accounting identity (modified for CMBP-1: no fill_qty)
    print("\n10. ACCOUNTING IDENTITY")
    print("-" * 40)
    print("  Identity: depth_qty_start + add_qty - pull_qty = depth_qty_end")
    
    if all(c in df.columns for c in ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty"]):
        expected_end = df["depth_qty_start"] + df["add_qty"] - df["pull_qty"]
        diff = (df["depth_qty_end"] - expected_end).abs()
        
        violations = (diff > 1e-6).sum()
        max_diff = diff.max()
        
        print(f"  Violations: {violations:,}")
        print(f"  Max absolute diff: {max_diff:.6f}")
        
        if violations > 0:
            issues.append(f"Accounting identity violations: {violations}")
            print("  ✗ Accounting identity violated")
            
            # Show sample violations
            violation_idx = diff[diff > 1e-6].head(5).index
            print("  Sample violations:")
            for idx in violation_idx:
                row = df.loc[idx]
                print(f"    start={row['depth_qty_start']:.2f}, add={row['add_qty']:.2f}, "
                      f"pull={row['pull_qty']:.2f}, end={row['depth_qty_end']:.2f}, "
                      f"expected={expected_end.loc[idx]:.2f}")
        else:
            print("  ✓ Accounting identity holds for all rows")
    
    # 11. window_valid
    print("\n11. WINDOW_VALID")
    print("-" * 40)
    if "window_valid" in df.columns:
        valid_counts = df["window_valid"].value_counts()
        for v, c in valid_counts.items():
            print(f"  {v}: {c:,} ({c/len(df)*100:.2f}%)")
        stats["window_valid_pct"] = (df["window_valid"].sum() / len(df)) * 100 if len(df) > 0 else 0
    
    return {"issues": issues, "stats": stats}


def statistical_summary(df_snap: pd.DataFrame, df_flow: pd.DataFrame) -> dict:
    """Generate statistical summary."""
    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*80}")
    
    summary = {}
    
    # Cross-table consistency
    print("\n1. CROSS-TABLE CONSISTENCY")
    print("-" * 40)
    
    if df_snap is not None and df_flow is not None:
        snap_windows = set(df_snap["window_end_ts_ns"].unique())
        flow_windows = set(df_flow["window_end_ts_ns"].unique())
        
        common_windows = snap_windows & flow_windows
        snap_only = snap_windows - flow_windows
        flow_only = flow_windows - snap_windows
        
        print(f"  Common windows: {len(common_windows):,}")
        print(f"  Snap-only windows: {len(snap_only):,}")
        print(f"  Flow-only windows: {len(flow_only):,}")
        
        if len(snap_only) > 0 or len(flow_only) > 0:
            print("  ⚠ Window mismatch between tables")
    
    # Distribution analysis
    print("\n2. DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    if df_snap is not None and "right" in df_snap.columns:
        print("\n  book_snapshot_1s by right:")
        for r in ["C", "P"]:
            subset = df_snap[df_snap["right"] == r]
            print(f"    {r}: {len(subset):,} rows")
    
    if df_flow is not None:
        print("\n  depth_and_flow_1s by right x side:")
        for r in ["C", "P"]:
            for s in ["A", "B"]:
                subset = df_flow[(df_flow["right"] == r) & (df_flow["side"] == s)]
                print(f"    {r}/{s}: {len(subset):,} rows")
    
    # Activity metrics
    print("\n3. ACTIVITY METRICS")
    print("-" * 40)
    
    if df_flow is not None:
        total_add = df_flow["add_qty"].sum()
        total_pull = df_flow["pull_qty"].sum()
        net_flow = total_add - total_pull
        
        print(f"  Total add_qty: {total_add:,.0f}")
        print(f"  Total pull_qty: {total_pull:,.0f}")
        print(f"  Net flow: {net_flow:,.0f}")
        
        summary["total_add"] = total_add
        summary["total_pull"] = total_pull
        
        # By right
        print("\n  By right:")
        for r in ["C", "P"]:
            subset = df_flow[df_flow["right"] == r]
            print(f"    {r}: add={subset['add_qty'].sum():,.0f}, pull={subset['pull_qty'].sum():,.0f}")
    
    return summary


def generate_report(snap_analysis: dict, flow_analysis: dict) -> str:
    """Generate final grade and report."""
    print(f"\n{'='*80}")
    print("FINAL QUALITY REPORT")
    print(f"{'='*80}")
    
    all_issues = []
    if snap_analysis:
        all_issues.extend([f"[book_snapshot] {i}" for i in snap_analysis.get("issues", [])])
    if flow_analysis:
        all_issues.extend([f"[depth_and_flow] {i}" for i in flow_analysis.get("issues", [])])
    
    # Grade calculation
    critical_issues = [i for i in all_issues if any(k in i.lower() for k in ["negative", "violation", "crossed", "mismatch"])]
    warning_issues = [i for i in all_issues if i not in critical_issues]
    
    if len(critical_issues) == 0 and len(warning_issues) == 0:
        grade = "A"
        status = "EXCELLENT - Production Ready"
    elif len(critical_issues) == 0 and len(warning_issues) <= 2:
        grade = "B"
        status = "GOOD - Minor issues, production acceptable"
    elif len(critical_issues) <= 2:
        grade = "C"
        status = "ACCEPTABLE - Some issues need attention"
    elif len(critical_issues) <= 5:
        grade = "D"
        status = "POOR - Significant issues"
    else:
        grade = "F"
        status = "FAIL - Critical issues, not production ready"
    
    print(f"\n  GRADE: {grade}")
    print(f"  STATUS: {status}")
    
    if all_issues:
        print(f"\n  ISSUES ({len(all_issues)} total):")
        print(f"  Critical: {len(critical_issues)}")
        print(f"  Warnings: {len(warning_issues)}")
        
        if critical_issues:
            print("\n  Critical Issues:")
            for i, issue in enumerate(critical_issues, 1):
                print(f"    {i}. {issue}")
        
        if warning_issues:
            print("\n  Warnings:")
            for i, issue in enumerate(warning_issues, 1):
                print(f"    {i}. {issue}")
    else:
        print("\n  ✓ No issues detected")
    
    return grade


def main():
    """Main audit entry point."""
    print("=" * 80)
    print("INSTITUTION-GRADE SILVER LAYER AUDIT: EQUITY_OPTION_CMBP_1")
    print("=" * 80)
    print(f"Date: 2026-01-08")
    print(f"Symbol: QQQ")
    print(f"Product Type: equity_option_cmbp_1")
    
    # Load data
    df_snap, df_flow = load_data()
    
    if df_snap is None and df_flow is None:
        print("\n✗ FATAL: No data files found")
        return
    
    # Schema analysis
    snap_schema = None
    flow_schema = None
    
    if df_snap is not None:
        snap_schema = analyze_schema(df_snap, "book_snapshot_1s")
    
    if df_flow is not None:
        flow_schema = analyze_schema(df_flow, "depth_and_flow_1s")
    
    # Semantic analysis
    snap_analysis = None
    flow_analysis = None
    
    if df_snap is not None:
        snap_analysis = analyze_book_snapshot(df_snap)
    
    if df_flow is not None:
        flow_analysis = analyze_depth_and_flow(df_flow)
    
    # Statistical summary
    if df_snap is not None and df_flow is not None:
        statistical_summary(df_snap, df_flow)
    
    # Final report
    grade = generate_report(snap_analysis, flow_analysis)
    
    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
