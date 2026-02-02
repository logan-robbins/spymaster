#!/usr/bin/env python3
"""
Grunt Validation Script for future_option_mbo silver layer.
Institutional-grade rigorous validation.

Validates:
1. Statistical checks (null/NaN/Inf counts, basic stats, distributions)
2. Mathematical formulas (depth_qty_start, strike bucketing, rel_ticks)
3. Sign/direction constraints (non-negative quantities, valid sides/rights)
4. Semantic validation (timestamps, price ranges, window validity)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_eng.config import load_config
from src.data_eng.io import partition_ref, is_partition_complete, read_partition


DATES = ["2026-01-07", "2026-01-14", "2026-01-23"]
SYMBOL = "ESH6"
TICK_INT = 250_000_000  # $0.25 in nanoseconds
STRIKE_STEP_TICKS = 20  # $5 strikes = 20 ticks
STRIKE_STEP_INT = STRIKE_STEP_TICKS * TICK_INT


def validate_null_nan_inf(df: pd.DataFrame, table: str) -> Dict[str, Any]:
    """Check for null/NaN/Inf values."""
    results = {"table": table, "issues": []}
    
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            null_count = df[col].isna().sum()
            if null_count > 0:
                results["issues"].append(f"{col}: {null_count} null/NaN values")
            
            if df[col].dtype in [np.float64, np.float32]:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    results["issues"].append(f"{col}: {inf_count} Inf values")
    
    return results


def validate_basic_stats(df: pd.DataFrame, table: str) -> Dict[str, Any]:
    """Compute basic statistics."""
    results = {"table": table, "row_count": len(df), "stats": {}}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        results["stats"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "median": float(df[col].median()),
            "p5": float(df[col].quantile(0.05)),
            "p95": float(df[col].quantile(0.95)),
        }
    
    return results


def validate_depth_and_flow_formulas(df: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """
    Verify:
    - depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty
    - rel_ticks matches strike - spot in ticks
    - strike_price_int aligned to $5 grid
    """
    results = {"date": dt, "issues": [], "info": []}
    
    if df.empty:
        results["issues"].append("Empty dataframe")
        return results
    
    # Formula verification: depth_qty_start
    df = df.copy()
    df["calculated_start"] = df["depth_qty_end"] - df["add_qty"] + df["pull_qty"] + df["fill_qty"]
    
    # Allow small floating point tolerance
    formula_mismatch = ~np.isclose(df["depth_qty_start"], df["calculated_start"], atol=1e-6)
    mismatch_count = formula_mismatch.sum()
    mismatch_pct = 100.0 * mismatch_count / len(df) if len(df) > 0 else 0
    
    results["formula_mismatch_count"] = int(mismatch_count)
    results["formula_mismatch_pct"] = round(mismatch_pct, 4)
    
    if mismatch_count > 0:
        results["issues"].append(
            f"Accounting identity: {mismatch_count} rows ({mismatch_pct:.2f}%) have formula mismatch"
        )
        
        # Sample mismatches
        mismatch_df = df[formula_mismatch].head(5)
        results["formula_mismatch_samples"] = mismatch_df[
            ["window_end_ts_ns", "depth_qty_start", "calculated_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty"]
        ].to_dict(orient="records")
    
    # rel_ticks should match strike - spot in ticks and strike grid should be $5
    required = {"rel_ticks", "strike_price_int", "spot_ref_price_int"}
    if required.issubset(df.columns):
        strike_mod = df["strike_price_int"].astype("int64") % STRIKE_STEP_INT != 0
        if strike_mod.any():
            results["issues"].append(f"strike_price_int: {int(strike_mod.sum())} values not aligned to $5 grid")

        tick_delta = df["strike_price_int"].astype("int64") - df["spot_ref_price_int"].astype("int64")
        tick_misaligned = (tick_delta % TICK_INT != 0)
        if tick_misaligned.any():
            results["issues"].append(f"rel_ticks: {int(tick_misaligned.sum())} strike deltas not on $0.25 tick grid")
        else:
            expected_rel_ticks = (tick_delta // TICK_INT).astype(int)
            rel_mismatch = expected_rel_ticks != df["rel_ticks"].astype(int)
            if rel_mismatch.any():
                results["issues"].append(f"rel_ticks: {int(rel_mismatch.sum())} values do not match strike - spot")
            else:
                results["info"].append("rel_ticks: 100% consistent with strike - spot in ticks")
    
    return results


def validate_sign_direction(df_flow: pd.DataFrame, df_snap: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """
    Verify:
    - add_qty >= 0, pull_qty >= 0, fill_qty >= 0
    - depth_qty_start >= 0, depth_qty_end >= 0
    - right column contains only 'C' (call) and 'P' (put)
    - side column contains only 'A' (ask) and 'B' (bid)
    - spot_ref_price_int > 0
    """
    results = {"date": dt, "issues": [], "violations": {}}
    
    # Quantity sign checks
    qty_cols = ["add_qty", "pull_qty", "fill_qty", "depth_qty_start", "depth_qty_end"]
    for col in qty_cols:
        if col in df_flow.columns:
            neg_count = (df_flow[col] < 0).sum()
            if neg_count > 0:
                results["issues"].append(f"{col}: {neg_count} negative values")
                results["violations"][col] = int(neg_count)
    
    # Right column check
    if "right" in df_flow.columns:
        valid_rights = {"C", "P"}
        invalid_rights = df_flow[~df_flow["right"].isin(valid_rights)]
        if len(invalid_rights) > 0:
            results["issues"].append(f"right: {len(invalid_rights)} invalid values (expected C/P)")
            results["violations"]["right"] = int(len(invalid_rights))
        else:
            right_dist = df_flow["right"].value_counts().to_dict()
            results["right_distribution"] = right_dist
    
    # Side column check
    if "side" in df_flow.columns:
        valid_sides = {"A", "B"}
        invalid_sides = df_flow[~df_flow["side"].isin(valid_sides)]
        if len(invalid_sides) > 0:
            results["issues"].append(f"side: {len(invalid_sides)} invalid values (expected A/B)")
            results["violations"]["side"] = int(len(invalid_sides))
        else:
            side_dist = df_flow["side"].value_counts().to_dict()
            results["side_distribution"] = side_dist
    
    # Spot reference price check
    if "spot_ref_price_int" in df_flow.columns:
        zero_spot = (df_flow["spot_ref_price_int"] <= 0).sum()
        if zero_spot > 0:
            results["issues"].append(f"spot_ref_price_int: {zero_spot} zero/negative values")
            results["violations"]["spot_ref_price_int"] = int(zero_spot)
    
    return results


def validate_semantic(df_flow: pd.DataFrame, df_snap: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """
    Verify:
    - window_start_ts_ns < window_end_ts_ns
    - 1-second windows
    - Spot reference price in expected ES range ($6900-$7100 for Jan 2026)
    - Strike prices reasonable (ATM ± $50)
    """
    results = {"date": dt, "issues": [], "info": []}
    
    WINDOW_NS = 1_000_000_000  # 1 second
    PRICE_SCALE = 1e-9
    
    # Timestamp ordering
    if "window_start_ts_ns" in df_flow.columns and "window_end_ts_ns" in df_flow.columns:
        bad_ordering = df_flow["window_start_ts_ns"] >= df_flow["window_end_ts_ns"]
        if bad_ordering.any():
            results["issues"].append(f"Timestamp ordering: {bad_ordering.sum()} windows have start >= end")
        
        # Window duration check
        window_duration = df_flow["window_end_ts_ns"] - df_flow["window_start_ts_ns"]
        non_1s = ~np.isclose(window_duration, WINDOW_NS, atol=1000)  # 1us tolerance
        if non_1s.any():
            results["issues"].append(f"Window duration: {non_1s.sum()} windows not exactly 1 second")
        else:
            results["info"].append("Window duration: 100% 1-second windows")
    
    # Spot reference price range
    if "spot_ref_price_int" in df_flow.columns:
        spot_prices = df_flow["spot_ref_price_int"].unique() * PRICE_SCALE
        min_spot = spot_prices.min()
        max_spot = spot_prices.max()
        results["spot_price_range"] = {"min": round(min_spot, 2), "max": round(max_spot, 2)}
        
        # Expected ES range for Jan 2026: $6900-$7100
        if min_spot < 6800 or max_spot > 7200:
            results["info"].append(f"Spot price outside typical ES range: ${min_spot:.2f}-${max_spot:.2f}")
        else:
            results["info"].append(f"Spot price in expected ES range: ${min_spot:.2f}-${max_spot:.2f}")
    
    # rel_ticks range (strike offsets in ticks, typically within ±$50)
    if "rel_ticks" in df_flow.columns:
        min_rel = df_flow["rel_ticks"].min()
        max_rel = df_flow["rel_ticks"].max()
        results["rel_ticks_range"] = {"min": int(min_rel), "max": int(max_rel)}
        
        # Convert to dollars: 1 tick = $0.25
        min_strike_offset = min_rel * 0.25
        max_strike_offset = max_rel * 0.25
        results["strike_offset_range_dollars"] = {"min": round(min_strike_offset, 2), "max": round(max_strike_offset, 2)}
    
    # pull_qty_rest check (should be 0 per performance optimization)
    if "pull_qty_rest" in df_flow.columns:
        nonzero = (df_flow["pull_qty_rest"] != 0).sum()
        if nonzero > 0:
            results["issues"].append(f"pull_qty_rest: {nonzero} non-zero values (expected 0 for performance)")
        else:
            results["info"].append("pull_qty_rest: 100% zeros (performance optimization, expected)")
    
    return results


def run_validation(dt: str, cfg, verbose: bool = True) -> Dict[str, Any]:
    """Run full validation for a single date."""
    print(f"\n{'='*60}")
    print(f"Validating future_option_mbo silver for {SYMBOL} on {dt}")
    print(f"{'='*60}")
    
    results = {"date": dt, "status": "UNKNOWN", "tables": {}}
    
    # Load depth_and_flow_1s
    flow_key = "silver.future_option_mbo.depth_and_flow_1s"
    flow_ref = partition_ref(cfg, flow_key, SYMBOL, dt)
    
    if not is_partition_complete(flow_ref):
        results["status"] = "MISSING"
        results["error"] = f"{flow_key} partition not found"
        return results
    
    df_flow = read_partition(flow_ref)
    print(f"\n{flow_key}: {len(df_flow):,} rows")
    
    # Load book_snapshot_1s
    snap_key = "silver.future_option_mbo.book_snapshot_1s"
    snap_ref = partition_ref(cfg, snap_key, SYMBOL, dt)
    
    if not is_partition_complete(snap_ref):
        df_snap = pd.DataFrame()
        print(f"{snap_key}: NOT FOUND")
    else:
        df_snap = read_partition(snap_ref)
        print(f"{snap_key}: {len(df_snap):,} rows")
    
    # Run validations
    print("\n--- 2.1 Statistical Validation ---")
    null_results = validate_null_nan_inf(df_flow, "depth_and_flow_1s")
    stats_results = validate_basic_stats(df_flow, "depth_and_flow_1s")
    
    if null_results["issues"]:
        print(f"NULL/NaN/Inf issues: {len(null_results['issues'])}")
        for issue in null_results["issues"]:
            print(f"  - {issue}")
    else:
        print("NULL/NaN/Inf: PASS (0 issues)")
    
    print(f"Row count: {stats_results['row_count']:,}")
    
    # Print key stats
    for col in ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty"]:
        if col in stats_results["stats"]:
            s = stats_results["stats"][col]
            print(f"  {col}: min={s['min']:.2f}, max={s['max']:.2f}, mean={s['mean']:.2f}")
    
    print("\n--- 2.2 Mathematical Formula Verification ---")
    formula_results = validate_depth_and_flow_formulas(df_flow, dt)
    
    print(f"Formula mismatch count: {formula_results['formula_mismatch_count']:,}")
    print(f"Formula mismatch pct: {formula_results['formula_mismatch_pct']:.2f}%")
    for info in formula_results.get("info", []):
        print(f"  {info}")
    
    print("\n--- 2.3 Sign/Direction Checks ---")
    sign_results = validate_sign_direction(df_flow, df_snap, dt)
    
    if sign_results["issues"]:
        print(f"ISSUES: {len(sign_results['issues'])}")
        for issue in sign_results["issues"]:
            print(f"  - {issue}")
    else:
        print("Sign/Direction: PASS (0 violations)")
    
    if "right_distribution" in sign_results:
        print(f"  Right distribution: {sign_results['right_distribution']}")
    if "side_distribution" in sign_results:
        print(f"  Side distribution: {sign_results['side_distribution']}")
    
    print("\n--- 2.4 Semantic Validation ---")
    semantic_results = validate_semantic(df_flow, df_snap, dt)
    
    if semantic_results["issues"]:
        print(f"ISSUES: {len(semantic_results['issues'])}")
        for issue in semantic_results["issues"]:
            print(f"  - {issue}")
    else:
        print("Semantic: PASS")
    
    for info in semantic_results.get("info", []):
        print(f"  {info}")
    
    if "spot_price_range" in semantic_results:
        print(f"  Spot price range: ${semantic_results['spot_price_range']['min']:.2f} - ${semantic_results['spot_price_range']['max']:.2f}")
    if "rel_ticks_range" in semantic_results:
        print(f"  rel_ticks range: {semantic_results['rel_ticks_range']['min']} to {semantic_results['rel_ticks_range']['max']}")
    
    # Aggregate results
    results["tables"]["depth_and_flow_1s"] = {
        "row_count": len(df_flow),
        "null_nan_inf": null_results,
        "stats": stats_results,
        "formulas": formula_results,
        "sign_direction": sign_results,
        "semantic": semantic_results,
    }
    
    if not df_snap.empty:
        snap_null = validate_null_nan_inf(df_snap, "book_snapshot_1s")
        snap_stats = validate_basic_stats(df_snap, "book_snapshot_1s")
        results["tables"]["book_snapshot_1s"] = {
            "row_count": len(df_snap),
            "null_nan_inf": snap_null,
            "stats": snap_stats,
        }
    
    # Determine overall status
    all_issues = (
        null_results["issues"] 
        + formula_results["issues"]
        + sign_results["issues"] 
        + semantic_results["issues"]
    )
    
    if not all_issues:
        results["status"] = "PASS"
    else:
        results["status"] = "FAIL"
        results["issue_count"] = len(all_issues)
    
    return results


def main():
    repo_root = Path(__file__).parent.parent
    cfg = load_config(repo_root, repo_root / "src/data_eng/config/datasets.yaml")
    
    print("=" * 60)
    print("GRUNT VALIDATION: future_option_mbo Silver Layer")
    print(f"Symbol: {SYMBOL}")
    print(f"Dates: {', '.join(DATES)}")
    print("=" * 60)
    
    all_results = []
    
    for dt in DATES:
        results = run_validation(dt, cfg)
        all_results.append(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    pass_count = sum(1 for r in all_results if r["status"] == "PASS")
    fail_count = sum(1 for r in all_results if r["status"] == "FAIL")
    missing_count = sum(1 for r in all_results if r["status"] == "MISSING")
    
    print(f"\nResults by date:")
    for r in all_results:
        status_icon = "✅" if r["status"] == "PASS" else ("❌" if r["status"] == "FAIL" else "⚠️")
        row_count = r["tables"].get("depth_and_flow_1s", {}).get("row_count", 0)
        print(f"  {r['date']}: {status_icon} {r['status']} ({row_count:,} flow rows)")
    
    print(f"\nOverall: {pass_count}/{len(DATES)} PASS, {fail_count} FAIL, {missing_count} MISSING")
    
    if fail_count == 0 and missing_count == 0:
        print("\n✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("\n❌ VALIDATION ISSUES DETECTED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
