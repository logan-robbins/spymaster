#!/usr/bin/env python3
"""
Validation script for future_option_mbo silver layer.
Validates: formula identity, warm-up handling, data quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_eng.config import load_config
from src.data_eng.io import partition_ref, is_partition_complete, read_partition


def validate_depth_and_flow(df: pd.DataFrame, table_name: str = "depth_and_flow_1s") -> dict:
    """Validate depth_and_flow_1s data for formula violations."""
    results = {
        "table": table_name,
        "row_count": len(df),
        "issues": [],
        "info": [],
    }
    
    if df.empty:
        results["issues"].append("Empty dataframe")
        return results
    
    # Check required columns
    required_cols = [
        "depth_qty_start", "depth_qty_end", "add_qty", 
        "pull_qty", "fill_qty", "spot_ref_price_int",
        "window_end_ts_ns"
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        results["issues"].append(f"Missing columns: {missing}")
        return results
    
    # Check stored depth_qty_start for negative values (the key constraint)
    negative_stored_mask = df["depth_qty_start"] < 0
    negative_stored_count = negative_stored_mask.sum()
    negative_stored_pct = 100.0 * negative_stored_count / len(df) if len(df) > 0 else 0
    
    results["negative_depth_qty_start_count"] = int(negative_stored_count)
    results["negative_depth_qty_start_pct"] = round(negative_stored_pct, 4)
    
    if negative_stored_count > 0:
        results["issues"].append(
            f"Constraint violation: {negative_stored_count} rows ({negative_stored_pct:.2f}%) have negative depth_qty_start"
        )
        sample = df[negative_stored_mask].head(5)[
            ["window_end_ts_ns", "depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty"]
        ].to_dict(orient="records")
        results["negative_depth_start_samples"] = sample
    
    # Verify accounting identity: start + add - pull - fill = end
    # This shows us the quality of the tracking
    df_check = df.copy()
    df_check["calculated_end"] = (
        df_check["depth_qty_start"] 
        + df_check["add_qty"] 
        - df_check["pull_qty"] 
        - df_check["fill_qty"]
    )
    
    identity_mismatch = ~np.isclose(df_check["depth_qty_end"], df_check["calculated_end"], atol=1e-6)
    identity_mismatch_count = identity_mismatch.sum()
    identity_mismatch_pct = 100.0 * identity_mismatch_count / len(df_check) if len(df_check) > 0 else 0
    
    results["identity_mismatch_count"] = int(identity_mismatch_count)
    results["identity_mismatch_pct"] = round(identity_mismatch_pct, 4)
    
    if identity_mismatch_count > 0:
        results["info"].append(
            f"Accounting identity: {identity_mismatch_count} rows ({identity_mismatch_pct:.2f}%) have start+add-pull-fill != end"
        )
    
    # Null/NaN checks
    for col in required_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            results["issues"].append(f"Column {col} has {null_count} nulls")
    
    # Negative quantity checks (quantities should be >= 0)
    qty_cols = ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty", "depth_qty_rest"]
    for col in qty_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                results["issues"].append(f"Column {col} has {neg_count} negative values")
    
    # Warm-up check: first window ref_price
    first_ts = df["window_end_ts_ns"].min()
    first_window = df[df["window_end_ts_ns"] == first_ts]
    if len(first_window) > 0:
        first_ref_price = first_window["spot_ref_price_int"].iloc[0]
        ref_price_float = first_ref_price * 1e-9
        results["first_window_ref_price"] = ref_price_float
        
        # ES futures should be around 5500-7000 range in 2026
        if ref_price_float < 4000:
            results["issues"].append(f"Warm-up artifact: first window ref_price={ref_price_float:.2f} is suspiciously low")
    
    # Statistics
    results["depth_qty_end_stats"] = {
        "min": float(df["depth_qty_end"].min()),
        "max": float(df["depth_qty_end"].max()),
        "mean": float(df["depth_qty_end"].mean()),
    }
    results["add_qty_stats"] = {
        "min": float(df["add_qty"].min()),
        "max": float(df["add_qty"].max()),
        "mean": float(df["add_qty"].mean()),
    }
    
    return results


def validate_book_snapshot(df: pd.DataFrame) -> dict:
    """Validate book_snapshot_1s data."""
    results = {
        "table": "book_snapshot_1s",
        "row_count": len(df),
        "issues": [],
    }
    
    if df.empty:
        results["issues"].append("Empty dataframe")
        return results
    
    # Check required columns
    required_cols = ["bid_price_int", "ask_price_int", "mid_price_int", "spot_ref_price_int"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        results["issues"].append(f"Missing columns: {missing}")
        return results
    
    # Crossed book check (ask should be > bid)
    if "bid_price_int" in df.columns and "ask_price_int" in df.columns:
        crossed = df[df["ask_price_int"] <= df["bid_price_int"]]
        if len(crossed) > 0:
            results["issues"].append(f"Crossed books: {len(crossed)} rows have ask <= bid")
    
    # Warm-up check
    if "spot_ref_price_int" in df.columns:
        first_ref = df.iloc[0]["spot_ref_price_int"] * 1e-9 if len(df) > 0 else 0
        results["first_window_ref_price"] = first_ref
        if first_ref < 4000:
            results["issues"].append(f"Warm-up artifact: first row ref_price={first_ref:.2f} is suspiciously low")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate future_option_mbo silver layer")
    parser.add_argument("--symbol", default="ESH6", help="Symbol to validate")
    parser.add_argument("--dt", required=True, help="Date to validate (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    cfg = load_config(repo_root, repo_root / "src/data_eng/config/datasets.yaml")
    
    print(f"Validating future_option_mbo silver layer for {args.symbol} on {args.dt}")
    print("=" * 60)
    
    all_results = []
    
    # Validate depth_and_flow_1s
    flow_key = "silver.future_option_mbo.depth_and_flow_1s"
    flow_ref = partition_ref(cfg, flow_key, args.symbol, args.dt)
    
    if is_partition_complete(flow_ref):
        print(f"\nValidating {flow_key}...")
        df_flow = read_partition(flow_ref)
        flow_results = validate_depth_and_flow(df_flow)
        all_results.append(flow_results)
        
        print(f"  Rows: {flow_results['row_count']}")
        print(f"  Negative depth_qty_start count: {flow_results.get('negative_depth_qty_start_count', 'N/A')}")
        print(f"  Negative depth_qty_start pct: {flow_results.get('negative_depth_qty_start_pct', 'N/A')}%")
        print(f"  Identity mismatch count: {flow_results.get('identity_mismatch_count', 'N/A')}")
        print(f"  Identity mismatch pct: {flow_results.get('identity_mismatch_pct', 'N/A')}%")
        print(f"  First window ref_price: {flow_results.get('first_window_ref_price', 'N/A')}")
        
        if flow_results["issues"]:
            print(f"  ISSUES:")
            for issue in flow_results["issues"]:
                print(f"    - {issue}")
        else:
            print(f"  STATUS: PASS")
        
        if flow_results.get("info"):
            print(f"  INFO:")
            for info in flow_results["info"]:
                print(f"    - {info}")
            
        if args.verbose and "negative_depth_start_samples" in flow_results:
            print(f"  Sample violations:")
            for sample in flow_results["negative_depth_start_samples"]:
                print(f"    {sample}")
    else:
        print(f"\n{flow_key}: NOT FOUND (partition incomplete)")
    
    # Validate book_snapshot_1s
    snap_key = "silver.future_option_mbo.book_snapshot_1s"
    snap_ref = partition_ref(cfg, snap_key, args.symbol, args.dt)
    
    if is_partition_complete(snap_ref):
        print(f"\nValidating {snap_key}...")
        df_snap = read_partition(snap_ref)
        snap_results = validate_book_snapshot(df_snap)
        all_results.append(snap_results)
        
        print(f"  Rows: {snap_results['row_count']}")
        print(f"  First window ref_price: {snap_results.get('first_window_ref_price', 'N/A')}")
        
        if snap_results["issues"]:
            print(f"  ISSUES:")
            for issue in snap_results["issues"]:
                print(f"    - {issue}")
        else:
            print(f"  STATUS: PASS")
    else:
        print(f"\n{snap_key}: NOT FOUND (partition incomplete)")
    
    # Summary
    print("\n" + "=" * 60)
    total_issues = sum(len(r.get("issues", [])) for r in all_results)
    if total_issues == 0:
        print("OVERALL STATUS: PASS")
    else:
        print(f"OVERALL STATUS: FAIL ({total_issues} issues)")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
