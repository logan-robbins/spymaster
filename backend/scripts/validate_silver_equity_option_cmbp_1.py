#!/usr/bin/env python3
"""
Rigorous validation of equity_option_cmbp_1 silver layer.

Validates:
1. Statistical: null/NaN/Inf counts, basic stats, row distribution
2. Mathematical: depth_qty_start formula, rel_ticks alignment
3. Sign/direction: quantity constraints, right/side values
4. Semantic: timestamps, spot reference, window validity
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Lake paths
LAKE_ROOT = Path(__file__).parent.parent / "lake"
SILVER_SNAP_KEY = "silver.equity_option_cmbp_1.book_snapshot_1s"
SILVER_FLOW_KEY = "silver.equity_option_cmbp_1.depth_and_flow_1s"

PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000

# Expected QQQ spot range
SPOT_MIN = 600.0
SPOT_MAX = 700.0


def partition_path(dataset_key: str, symbol: str, dt: str) -> Path:
    """Build partition path from dataset key."""
    parts = dataset_key.split(".")
    layer = parts[0]
    product_type = parts[1]
    table = parts[2]
    
    # Silver and Gold don't have source=databento in path
    if layer in ("silver", "gold"):
        return (
            LAKE_ROOT
            / layer
            / f"product_type={product_type}"
            / f"symbol={symbol}"
            / f"table={table}"
            / f"dt={dt}"
        )
    else:
        return (
            LAKE_ROOT
            / layer
            / f"source=databento"
            / f"product_type={product_type}"
            / f"symbol={symbol}"
            / f"table={table}"
            / f"dt={dt}"
        )


def read_partition(dataset_key: str, symbol: str, dt: str) -> pd.DataFrame:
    """Read a partition and return DataFrame."""
    path = partition_path(dataset_key, symbol, dt)
    if not path.exists():
        raise FileNotFoundError(f"Partition not found: {path}")
    
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in: {path}")
    
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def validate_nulls_nans_infs(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    """Check for null, NaN, and Inf values."""
    results = {"table": table_name, "issues": [], "column_counts": {}}
    
    for col in df.columns:
        null_count = df[col].isna().sum()
        
        if df[col].dtype in [np.float64, np.float32]:
            inf_count = np.isinf(df[col]).sum()
        else:
            inf_count = 0
        
        if null_count > 0 or inf_count > 0:
            results["issues"].append({
                "column": col,
                "null_count": int(null_count),
                "inf_count": int(inf_count),
            })
        
        results["column_counts"][col] = {
            "null": int(null_count),
            "inf": int(inf_count),
        }
    
    return results


def compute_basic_stats(df: pd.DataFrame, table_name: str, numeric_cols: List[str]) -> Dict[str, Any]:
    """Compute basic statistics for numeric columns."""
    stats = {"table": table_name, "columns": {}}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        stats["columns"][col] = {
            "count": int(len(series)),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "p1": float(np.percentile(series, 1)),
            "p5": float(np.percentile(series, 5)),
            "p25": float(np.percentile(series, 25)),
            "p75": float(np.percentile(series, 75)),
            "p95": float(np.percentile(series, 95)),
            "p99": float(np.percentile(series, 99)),
        }
    
    return stats


def validate_snap_distribution(df_snap: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate book_snapshot_1s distribution."""
    results = {"dt": dt, "row_count": len(df_snap)}
    
    if len(df_snap) == 0:
        results["error"] = "Empty DataFrame"
        return results
    
    # Right distribution
    if "right" in df_snap.columns:
        right_dist = df_snap["right"].value_counts().to_dict()
        results["right_distribution"] = {str(k): int(v) for k, v in right_dist.items()}
    
    # Unique instruments
    if "instrument_id" in df_snap.columns:
        results["unique_instruments"] = int(df_snap["instrument_id"].nunique())
    
    # Unique windows
    if "window_end_ts_ns" in df_snap.columns:
        results["unique_windows"] = int(df_snap["window_end_ts_ns"].nunique())
    
    return results


def validate_flow_distribution(df_flow: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate depth_and_flow_1s distribution."""
    results = {"dt": dt, "row_count": len(df_flow)}
    
    if len(df_flow) == 0:
        results["error"] = "Empty DataFrame"
        return results
    
    # Right distribution
    if "right" in df_flow.columns:
        right_dist = df_flow["right"].value_counts().to_dict()
        results["right_distribution"] = {str(k): int(v) for k, v in right_dist.items()}
    
    # Side distribution
    if "side" in df_flow.columns:
        side_dist = df_flow["side"].value_counts().to_dict()
        results["side_distribution"] = {str(k): int(v) for k, v in side_dist.items()}
    
    # rel_ticks range
    if "rel_ticks" in df_flow.columns:
        results["rel_ticks_min"] = int(df_flow["rel_ticks"].min())
        results["rel_ticks_max"] = int(df_flow["rel_ticks"].max())
        results["rel_ticks_unique"] = int(df_flow["rel_ticks"].nunique())
    
    # Unique windows
    if "window_end_ts_ns" in df_flow.columns:
        results["unique_windows"] = int(df_flow["window_end_ts_ns"].nunique())
    
    return results


def validate_depth_qty_start_formula(df_flow: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate: depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty."""
    results = {"dt": dt, "formula": "depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty"}
    
    required = ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty"]
    missing = [c for c in required if c not in df_flow.columns]
    if missing:
        results["error"] = f"Missing columns: {missing}"
        return results
    
    computed = (
        df_flow["depth_qty_end"]
        - df_flow["add_qty"]
        + df_flow["pull_qty"]
        + df_flow["fill_qty"]
    )
    
    # Actual might be clamped to 0
    computed_clamped = computed.clip(lower=0.0)
    
    diff = np.abs(df_flow["depth_qty_start"] - computed_clamped)
    
    results["total_rows"] = len(df_flow)
    results["max_abs_error"] = float(diff.max())
    results["mean_abs_error"] = float(diff.mean())
    results["violations_count"] = int((diff > 1e-6).sum())
    results["formula_valid"] = results["violations_count"] == 0
    
    if results["violations_count"] > 0:
        violation_rows = df_flow[diff > 1e-6].head(5)
        results["sample_violations"] = violation_rows[required].to_dict(orient="records")
    
    return results


def validate_rel_ticks_alignment(df_flow: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate rel_ticks are aligned to $1 grid (even values)."""
    results = {"dt": dt, "expectation": "rel_ticks must be even (multiples of 2)"}
    
    if "rel_ticks" not in df_flow.columns:
        results["error"] = "Missing rel_ticks column"
        return results
    
    odd_ticks = df_flow["rel_ticks"] % 2 != 0
    odd_count = odd_ticks.sum()
    
    results["total_rows"] = len(df_flow)
    results["odd_rel_ticks_count"] = int(odd_count)
    results["aligned"] = odd_count == 0
    
    if odd_count > 0:
        results["sample_odd_ticks"] = df_flow.loc[odd_ticks, "rel_ticks"].head(10).tolist()
    
    return results


def validate_sign_direction(df_flow: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate sign/direction constraints."""
    results = {"dt": dt, "checks": {}}
    
    # add_qty >= 0
    if "add_qty" in df_flow.columns:
        neg_add = (df_flow["add_qty"] < 0).sum()
        results["checks"]["add_qty_negative"] = int(neg_add)
    
    # pull_qty >= 0
    if "pull_qty" in df_flow.columns:
        neg_pull = (df_flow["pull_qty"] < 0).sum()
        results["checks"]["pull_qty_negative"] = int(neg_pull)
    
    # fill_qty >= 0 (and should be 0 for CMBP-1)
    if "fill_qty" in df_flow.columns:
        neg_fill = (df_flow["fill_qty"] < 0).sum()
        nonzero_fill = (df_flow["fill_qty"] != 0).sum()
        results["checks"]["fill_qty_negative"] = int(neg_fill)
        results["checks"]["fill_qty_nonzero"] = int(nonzero_fill)
    
    # pull_qty_rest >= 0 (and should be 0 for CMBP-1)
    if "pull_qty_rest" in df_flow.columns:
        neg_pull_rest = (df_flow["pull_qty_rest"] < 0).sum()
        nonzero_pull_rest = (df_flow["pull_qty_rest"] != 0).sum()
        results["checks"]["pull_qty_rest_negative"] = int(neg_pull_rest)
        results["checks"]["pull_qty_rest_nonzero"] = int(nonzero_pull_rest)
    
    # depth_qty_start >= 0
    if "depth_qty_start" in df_flow.columns:
        neg_start = (df_flow["depth_qty_start"] < 0).sum()
        results["checks"]["depth_qty_start_negative"] = int(neg_start)
    
    # depth_qty_end >= 0
    if "depth_qty_end" in df_flow.columns:
        neg_end = (df_flow["depth_qty_end"] < 0).sum()
        results["checks"]["depth_qty_end_negative"] = int(neg_end)
    
    all_pass = all(v == 0 for k, v in results["checks"].items() if "negative" in k)
    results["all_constraints_valid"] = all_pass
    
    return results


def validate_right_side_values(df_snap: pd.DataFrame, df_flow: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate right and side columns contain only expected values."""
    results = {"dt": dt}
    
    # Snap right values
    if "right" in df_snap.columns:
        snap_rights = set(df_snap["right"].unique())
        results["snap_right_values"] = list(snap_rights)
        results["snap_right_valid"] = snap_rights <= {"C", "P"}
    
    # Flow right values
    if "right" in df_flow.columns:
        flow_rights = set(df_flow["right"].unique())
        results["flow_right_values"] = list(flow_rights)
        results["flow_right_valid"] = flow_rights <= {"C", "P"}
    
    # Flow side values
    if "side" in df_flow.columns:
        flow_sides = set(df_flow["side"].unique())
        results["flow_side_values"] = list(flow_sides)
        results["flow_side_valid"] = flow_sides <= {"A", "B"}
    
    return results


def validate_semantic(df_snap: pd.DataFrame, df_flow: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate semantic constraints."""
    results = {"dt": dt}
    
    # Timestamp ordering in flow
    if "window_start_ts_ns" in df_flow.columns and "window_end_ts_ns" in df_flow.columns:
        invalid_ts = (df_flow["window_start_ts_ns"] >= df_flow["window_end_ts_ns"]).sum()
        results["flow_invalid_timestamp_order"] = int(invalid_ts)
    
    # 1-second windows
    if "window_start_ts_ns" in df_flow.columns and "window_end_ts_ns" in df_flow.columns:
        window_dur = df_flow["window_end_ts_ns"] - df_flow["window_start_ts_ns"]
        non_1s_windows = (window_dur != WINDOW_NS).sum()
        results["flow_non_1s_windows"] = int(non_1s_windows)
    
    # Spot reference in QQQ range
    if "spot_ref_price_int" in df_flow.columns:
        spot_prices = df_flow["spot_ref_price_int"].unique() * PRICE_SCALE
        spot_min = float(spot_prices.min())
        spot_max = float(spot_prices.max())
        results["spot_min"] = spot_min
        results["spot_max"] = spot_max
        results["spot_in_range"] = (spot_min >= SPOT_MIN) and (spot_max <= SPOT_MAX)
    
    # book_valid
    if "book_valid" in df_snap.columns:
        invalid_books = (~df_snap["book_valid"]).sum()
        results["snap_invalid_book_count"] = int(invalid_books)
    
    # window_valid
    if "window_valid" in df_flow.columns:
        invalid_windows = (~df_flow["window_valid"]).sum()
        results["flow_invalid_window_count"] = int(invalid_windows)
    
    # spot_ref_price_int > 0
    if "spot_ref_price_int" in df_flow.columns:
        zero_spot = (df_flow["spot_ref_price_int"] <= 0).sum()
        results["flow_zero_spot_ref"] = int(zero_spot)
    
    return results


def validate_crossed_books(df_snap: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Check for crossed books (bid >= ask)."""
    results = {"dt": dt}
    
    if "bid_price_int" in df_snap.columns and "ask_price_int" in df_snap.columns:
        # Only check where both bid and ask are positive
        valid_mask = (df_snap["bid_price_int"] > 0) & (df_snap["ask_price_int"] > 0)
        valid_df = df_snap[valid_mask]
        
        crossed = (valid_df["bid_price_int"] >= valid_df["ask_price_int"]).sum()
        results["crossed_books"] = int(crossed)
        results["total_valid_books"] = len(valid_df)
        results["no_crossed_books"] = crossed == 0
    
    return results


def run_validation(dates: List[str], symbol: str = "QQQ", verbose: bool = False) -> Dict[str, Any]:
    """Run full validation for given dates."""
    all_results = {
        "symbol": symbol,
        "dates": dates,
        "pipeline_status": {},
        "statistical": {},
        "formula_verification": {},
        "sign_direction": {},
        "semantic": {},
        "summary": {},
    }
    
    snap_numeric_cols = [
        "bid_price_int", "ask_price_int", "mid_price", "mid_price_int", 
        "strike_price_int", "spot_ref_price_int"
    ]
    flow_numeric_cols = [
        "spot_ref_price_int", "rel_ticks", "depth_qty_start", "depth_qty_end",
        "add_qty", "pull_qty", "pull_qty_rest", "fill_qty", "strike_points"
    ]
    
    for dt in dates:
        print(f"\n{'='*60}")
        print(f"Validating {dt}")
        print(f"{'='*60}")
        
        try:
            df_snap = read_partition(SILVER_SNAP_KEY, symbol, dt)
            df_flow = read_partition(SILVER_FLOW_KEY, symbol, dt)
            
            all_results["pipeline_status"][dt] = {
                "status": "SUCCESS",
                "snap_rows": len(df_snap),
                "flow_rows": len(df_flow),
            }
            
            print(f"  book_snapshot_1s rows: {len(df_snap)}")
            print(f"  depth_and_flow_1s rows: {len(df_flow)}")
            
        except FileNotFoundError as e:
            all_results["pipeline_status"][dt] = {
                "status": "FAIL",
                "error": str(e),
            }
            print(f"  ERROR: {e}")
            continue
        
        # Statistical validation
        print("\n  [Statistical Validation]")
        
        snap_nulls = validate_nulls_nans_infs(df_snap, "book_snapshot_1s")
        flow_nulls = validate_nulls_nans_infs(df_flow, "depth_and_flow_1s")
        
        snap_null_issues = len(snap_nulls["issues"])
        flow_null_issues = len(flow_nulls["issues"])
        print(f"    Null/NaN/Inf issues - snap: {snap_null_issues}, flow: {flow_null_issues}")
        
        snap_stats = compute_basic_stats(df_snap, "book_snapshot_1s", snap_numeric_cols)
        flow_stats = compute_basic_stats(df_flow, "depth_and_flow_1s", flow_numeric_cols)
        
        snap_dist = validate_snap_distribution(df_snap, dt)
        flow_dist = validate_flow_distribution(df_flow, dt)
        
        print(f"    Snap unique instruments: {snap_dist.get('unique_instruments', 'N/A')}")
        print(f"    Snap unique windows: {snap_dist.get('unique_windows', 'N/A')}")
        print(f"    Flow unique windows: {flow_dist.get('unique_windows', 'N/A')}")
        print(f"    Flow rel_ticks range: [{flow_dist.get('rel_ticks_min', 'N/A')}, {flow_dist.get('rel_ticks_max', 'N/A')}]")
        
        all_results["statistical"][dt] = {
            "snap_nulls": snap_nulls,
            "flow_nulls": flow_nulls,
            "snap_stats": snap_stats,
            "flow_stats": flow_stats,
            "snap_distribution": snap_dist,
            "flow_distribution": flow_dist,
        }
        
        # Formula verification
        print("\n  [Formula Verification]")
        
        depth_formula = validate_depth_qty_start_formula(df_flow, dt)
        rel_ticks_alignment = validate_rel_ticks_alignment(df_flow, dt)
        
        print(f"    depth_qty_start formula: {'PASS' if depth_formula['formula_valid'] else 'FAIL'}")
        print(f"      max_abs_error: {depth_formula['max_abs_error']:.6f}")
        print(f"      violations: {depth_formula['violations_count']}")
        
        print(f"    rel_ticks alignment ($1 grid): {'PASS' if rel_ticks_alignment['aligned'] else 'FAIL'}")
        print(f"      odd_rel_ticks: {rel_ticks_alignment['odd_rel_ticks_count']}")
        
        all_results["formula_verification"][dt] = {
            "depth_qty_start": depth_formula,
            "rel_ticks_alignment": rel_ticks_alignment,
        }
        
        # Sign/direction checks
        print("\n  [Sign/Direction Checks]")
        
        sign_checks = validate_sign_direction(df_flow, dt)
        right_side = validate_right_side_values(df_snap, df_flow, dt)
        crossed = validate_crossed_books(df_snap, dt)
        
        print(f"    Quantity constraints: {'PASS' if sign_checks['all_constraints_valid'] else 'FAIL'}")
        for check, count in sign_checks["checks"].items():
            if count > 0:
                print(f"      {check}: {count}")
        
        print(f"    Right values valid: {right_side.get('flow_right_valid', 'N/A')}")
        print(f"    Side values valid: {right_side.get('flow_side_valid', 'N/A')}")
        print(f"    Crossed books: {crossed.get('crossed_books', 'N/A')}")
        
        all_results["sign_direction"][dt] = {
            "quantity_checks": sign_checks,
            "right_side_values": right_side,
            "crossed_books": crossed,
        }
        
        # Semantic validation
        print("\n  [Semantic Validation]")
        
        semantic = validate_semantic(df_snap, df_flow, dt)
        
        print(f"    Spot range: ${semantic.get('spot_min', 0):.2f} - ${semantic.get('spot_max', 0):.2f}")
        print(f"    Spot in expected QQQ range: {semantic.get('spot_in_range', 'N/A')}")
        print(f"    Invalid timestamp order: {semantic.get('flow_invalid_timestamp_order', 'N/A')}")
        print(f"    Non-1s windows: {semantic.get('flow_non_1s_windows', 'N/A')}")
        print(f"    Invalid book_valid: {semantic.get('snap_invalid_book_count', 'N/A')}")
        print(f"    Invalid window_valid: {semantic.get('flow_invalid_window_count', 'N/A')}")
        
        all_results["semantic"][dt] = semantic
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_pass = True
    for dt in dates:
        if all_results["pipeline_status"].get(dt, {}).get("status") != "SUCCESS":
            all_pass = False
            print(f"  {dt}: PIPELINE FAILED")
            continue
        
        formula_pass = (
            all_results["formula_verification"].get(dt, {}).get("depth_qty_start", {}).get("formula_valid", False)
            and all_results["formula_verification"].get(dt, {}).get("rel_ticks_alignment", {}).get("aligned", False)
        )
        
        sign_pass = all_results["sign_direction"].get(dt, {}).get("quantity_checks", {}).get("all_constraints_valid", False)
        crossed_pass = all_results["sign_direction"].get(dt, {}).get("crossed_books", {}).get("no_crossed_books", False)
        semantic_pass = all_results["semantic"].get(dt, {}).get("spot_in_range", False)
        
        dt_pass = formula_pass and sign_pass and crossed_pass and semantic_pass
        all_pass = all_pass and dt_pass
        
        status = "PASS" if dt_pass else "FAIL"
        print(f"  {dt}: {status}")
        print(f"    Formula: {'PASS' if formula_pass else 'FAIL'}")
        print(f"    Sign/Direction: {'PASS' if sign_pass else 'FAIL'}")
        print(f"    No Crossed Books: {'PASS' if crossed_pass else 'FAIL'}")
        print(f"    Semantic: {'PASS' if semantic_pass else 'FAIL'}")
    
    all_results["summary"]["all_pass"] = all_pass
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Validate equity_option_cmbp_1 silver layer")
    parser.add_argument("--dates", nargs="+", default=["2026-01-09", "2026-01-16", "2026-01-27"])
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    results = run_validation(args.dates, args.symbol, args.verbose)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    
    return 0 if results["summary"]["all_pass"] else 1


if __name__ == "__main__":
    exit(main())
