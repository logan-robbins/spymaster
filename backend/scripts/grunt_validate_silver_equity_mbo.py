"""
Grunt validation script for equity_mbo silver layer.
Performs institutional-grade validation of book_snapshot_1s and depth_and_flow_1s tables.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

LAKE_ROOT = Path(__file__).parent.parent / "lake"
PRICE_SCALE = 1e-9
BUCKET_SIZE = 0.50
BUCKET_INT = int(round(BUCKET_SIZE / PRICE_SCALE))  # 500_000_000
WINDOW_NS = 1_000_000_000  # 1 second

# Validation dates
DATES = ["2026-01-08", "2026-01-16", "2026-01-27"]

# Expected QQQ price range for January 2026 (from README)
QQQ_PRICE_MIN = 600.0
QQQ_PRICE_MAX = 650.0

# Expected spread range for equities (tighter than futures)
SPREAD_MIN = 0.01
SPREAD_MAX = 0.10


def load_silver_data(table: str, dt: str) -> pd.DataFrame:
    """Load silver equity_mbo table for a given date."""
    path = LAKE_ROOT / f"silver/product_type=equity_mbo/symbol=QQQ/table={table}/dt={dt}"
    files = list(path.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def load_bronze_data(dt: str) -> pd.DataFrame:
    """Load bronze equity_mbo data for a given date."""
    path = LAKE_ROOT / f"bronze/source=databento/product_type=equity_mbo/symbol=QQQ/table=mbo/dt={dt}"
    files = list(path.glob("part-*.parquet"))
    if not files:
        return pd.DataFrame()
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    return dataset.to_table(columns=["action", "size"]).to_pandas()


def validate_nulls_inf(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    """Check for null, NaN, and Inf values."""
    results = {"table": table_name}
    
    null_counts = df.isnull().sum()
    results["null_total"] = int(null_counts.sum())
    results["null_by_column"] = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    
    # Check numeric columns for Inf
    inf_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = int(np.isinf(df[col].values).sum())
        if inf_count > 0:
            inf_counts[col] = inf_count
    results["inf_by_column"] = inf_counts
    results["inf_total"] = sum(inf_counts.values())
    
    results["pass"] = results["null_total"] == 0 and results["inf_total"] == 0
    return results


def compute_basic_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute basic statistics for specified columns."""
    stats = {}
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        stats[col] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(vals.median()),
            "p25": float(vals.quantile(0.25)),
            "p75": float(vals.quantile(0.75)),
            "p95": float(vals.quantile(0.95)),
            "p99": float(vals.quantile(0.99)),
        }
    return stats


def validate_book_snapshot_formulas(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate mathematical formulas in book_snapshot_1s."""
    results = {}
    
    valid_mask = (df["best_bid_price_int"] > 0) & (df["best_ask_price_int"] > 0)
    df_valid = df.loc[valid_mask].copy()
    
    if len(df_valid) == 0:
        return {"error": "No valid BBO rows", "pass": False}
    
    # 1. mid_price = (best_bid_price_int + best_ask_price_int) * 0.5 * 1e-9
    expected_mid = (df_valid["best_bid_price_int"] + df_valid["best_ask_price_int"]) * 0.5 * PRICE_SCALE
    mid_errors = (df_valid["mid_price"] - expected_mid).abs()
    results["mid_price_max_error"] = float(mid_errors.max())
    results["mid_price_formula_valid"] = mid_errors.max() < 1e-9
    
    # 2. mid_price_int = round((best_bid_price_int + best_ask_price_int) * 0.5)
    expected_mid_int = np.round((df_valid["best_bid_price_int"] + df_valid["best_ask_price_int"]) * 0.5).astype("int64")
    mid_int_errors = (df_valid["mid_price_int"] - expected_mid_int).abs()
    results["mid_price_int_max_error"] = int(mid_int_errors.max())
    results["mid_price_int_formula_valid"] = mid_int_errors.max() <= 1
    
    results["pass"] = results["mid_price_formula_valid"] and results["mid_price_int_formula_valid"]
    return results


def validate_depth_and_flow_formulas(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate mathematical formulas in depth_and_flow_1s."""
    results = {}
    
    # 1. depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty
    expected_start = df["depth_qty_end"] - df["add_qty"] + df["pull_qty"] + df["fill_qty"]
    # Account for clamping at 0
    expected_start_clamped = expected_start.clip(lower=0)
    start_errors = (df["depth_qty_start"] - expected_start).abs()
    start_errors_clamped = (df["depth_qty_start"] - expected_start_clamped).abs()
    
    results["depth_qty_start_max_error_raw"] = float(start_errors.max())
    results["depth_qty_start_max_error_clamped"] = float(start_errors_clamped.max())
    results["depth_qty_start_formula_valid"] = start_errors_clamped.max() < 0.01
    
    # 2. rel_ticks = (price_int - spot_ref_price_int) / BUCKET_INT
    expected_rel_ticks = ((df["price_int"] - df["spot_ref_price_int"]) / BUCKET_INT).astype("int32")
    rel_ticks_match = df["rel_ticks"] == expected_rel_ticks
    results["rel_ticks_match_count"] = int(rel_ticks_match.sum())
    results["rel_ticks_total_count"] = len(df)
    results["rel_ticks_match_pct"] = float(rel_ticks_match.sum() / len(df)) * 100 if len(df) > 0 else 0.0
    results["rel_ticks_formula_valid"] = results["rel_ticks_match_pct"] >= 99.9
    
    results["pass"] = results["depth_qty_start_formula_valid"] and results["rel_ticks_formula_valid"]
    return results


def validate_sign_direction_constraints(df_snap: pd.DataFrame, df_flow: pd.DataFrame) -> Dict[str, Any]:
    """Validate sign/direction constraints."""
    results = {}
    violations = []
    
    # Book snapshot constraints (when book_valid=True)
    valid_snap = df_snap.loc[df_snap["book_valid"] == True].copy()
    
    # best_bid_price_int > 0
    bid_positive = valid_snap["best_bid_price_int"] > 0
    bid_zero = (~bid_positive).sum()
    if bid_zero > 0:
        violations.append(f"best_bid_price_int <= 0: {bid_zero} rows")
    results["bid_price_positive_violation_count"] = int(bid_zero)
    
    # best_ask_price_int > 0
    ask_positive = valid_snap["best_ask_price_int"] > 0
    ask_zero = (~ask_positive).sum()
    if ask_zero > 0:
        violations.append(f"best_ask_price_int <= 0: {ask_zero} rows")
    results["ask_price_positive_violation_count"] = int(ask_zero)
    
    # best_bid_price_int < best_ask_price_int (no crossed books)
    valid_bbo = valid_snap.loc[(valid_snap["best_bid_price_int"] > 0) & (valid_snap["best_ask_price_int"] > 0)]
    crossed = valid_bbo["best_bid_price_int"] >= valid_bbo["best_ask_price_int"]
    if crossed.sum() > 0:
        violations.append(f"crossed books (bid >= ask): {crossed.sum()} rows")
        # Sample some crossed books for debugging
        crossed_samples = valid_bbo.loc[crossed].head(3)[
            ["window_start_ts_ns", "best_bid_price_int", "best_ask_price_int"]
        ].to_dict("records")
        results["crossed_book_examples"] = crossed_samples
    results["crossed_book_count"] = int(crossed.sum())
    
    # Depth and flow constraints
    qty_cols = ["add_qty", "pull_qty", "fill_qty", "depth_qty_start", "depth_qty_end"]
    negative_counts = {}
    for col in qty_cols:
        if col in df_flow.columns:
            neg_count = int((df_flow[col] < 0).sum())
            if neg_count > 0:
                violations.append(f"{col} < 0: {neg_count} rows")
            negative_counts[col] = neg_count
    results["negative_quantity_counts"] = negative_counts
    
    # Side column contains only 'B' and 'A'
    side_values = set(df_flow["side"].unique())
    expected_sides = {"A", "B"}
    invalid_sides = side_values - expected_sides
    if invalid_sides:
        violations.append(f"invalid side values: {invalid_sides}")
    results["side_values"] = list(side_values)
    results["side_valid"] = side_values == expected_sides or side_values.issubset(expected_sides)
    
    results["violations"] = violations
    results["pass"] = len(violations) == 0
    return results


def validate_semantic_constraints(df_snap: pd.DataFrame, df_flow: pd.DataFrame, dt: str) -> Dict[str, Any]:
    """Validate semantic constraints (timestamps, price ranges, etc.)."""
    results = {"dt": dt}
    violations = []
    
    # 1. window_start_ts_ns < window_end_ts_ns
    window_order = df_snap["window_start_ts_ns"] < df_snap["window_end_ts_ns"]
    if not window_order.all():
        violations.append(f"window_start >= window_end: {(~window_order).sum()} rows")
    results["window_order_valid"] = window_order.all()
    
    # 2. 1-second windows
    window_durations = df_snap["window_end_ts_ns"] - df_snap["window_start_ts_ns"]
    not_1s = window_durations != WINDOW_NS
    if not_1s.any():
        violations.append(f"window duration != 1s: {not_1s.sum()} rows")
        results["window_duration_examples"] = window_durations[not_1s].head(3).tolist()
    results["window_1s_valid"] = not not_1s.any()
    
    # 3. QQQ price in expected range ($600-$650 in Jan 2026)
    valid_bbo = df_snap.loc[(df_snap["best_bid_price_int"] > 0) & (df_snap["best_ask_price_int"] > 0)]
    mid_prices = valid_bbo["mid_price"]
    price_min = float(mid_prices.min()) if len(mid_prices) > 0 else 0.0
    price_max = float(mid_prices.max()) if len(mid_prices) > 0 else 0.0
    results["price_range"] = {"min": price_min, "max": price_max}
    
    price_in_range = (mid_prices >= QQQ_PRICE_MIN) & (mid_prices <= QQQ_PRICE_MAX)
    if not price_in_range.all():
        out_of_range = (~price_in_range).sum()
        violations.append(f"price outside ${QQQ_PRICE_MIN}-${QQQ_PRICE_MAX}: {out_of_range} rows")
    results["price_in_expected_range"] = price_in_range.all() if len(price_in_range) > 0 else True
    
    # 4. Spread typically $0.01-$0.05 (check spread is reasonable)
    spreads_int = valid_bbo["best_ask_price_int"] - valid_bbo["best_bid_price_int"]
    spreads = spreads_int * PRICE_SCALE
    results["spread_stats"] = {
        "min": float(spreads.min()) if len(spreads) > 0 else 0.0,
        "max": float(spreads.max()) if len(spreads) > 0 else 0.0,
        "mean": float(spreads.mean()) if len(spreads) > 0 else 0.0,
        "median": float(spreads.median()) if len(spreads) > 0 else 0.0,
    }
    
    spread_ok = (spreads >= SPREAD_MIN) & (spreads <= SPREAD_MAX)
    spread_outliers = (~spread_ok).sum()
    if spread_outliers > 0:
        # Allow some outliers (5% tolerance)
        outlier_pct = spread_outliers / len(spreads) * 100
        if outlier_pct > 5:
            violations.append(f"spread outliers > 5%: {outlier_pct:.1f}%")
    results["spread_outlier_pct"] = float(spread_outliers / len(spreads) * 100) if len(spreads) > 0 else 0.0
    
    # 5. rel_ticks range: [-100, 100] ($50 grid from spot)
    rel_ticks_min = int(df_flow["rel_ticks"].min())
    rel_ticks_max = int(df_flow["rel_ticks"].max())
    results["rel_ticks_range"] = {"min": rel_ticks_min, "max": rel_ticks_max}
    
    if rel_ticks_min < -100 or rel_ticks_max > 100:
        violations.append(f"rel_ticks outside [-100, 100]: [{rel_ticks_min}, {rel_ticks_max}]")
    results["rel_ticks_in_range"] = (-100 <= rel_ticks_min) and (rel_ticks_max <= 100)
    
    # 6. Depth/flow flow quantities are SUMMED within price buckets (verify > 0)
    total_add = df_flow["add_qty"].sum()
    total_fill = df_flow["fill_qty"].sum()
    total_pull = df_flow["pull_qty"].sum()
    results["total_volumes"] = {
        "add_qty": float(total_add),
        "fill_qty": float(total_fill),
        "pull_qty": float(total_pull),
    }
    
    results["violations"] = violations
    results["pass"] = len(violations) == 0
    return results


def validate_bronze_silver_reconciliation(dt: str, df_flow: pd.DataFrame) -> Dict[str, Any]:
    """Cross-validate bronze inputs against silver outputs."""
    bronze_df = load_bronze_data(dt)
    if bronze_df.empty:
        return {"dt": dt, "error": "Bronze data not found", "pass": True}
    
    results = {"dt": dt}
    
    bronze_adds = bronze_df.loc[bronze_df["action"] == "A", "size"].sum()
    silver_adds = df_flow["add_qty"].sum()
    results["bronze_add_volume"] = int(bronze_adds)
    results["silver_add_volume"] = float(silver_adds)
    results["add_volume_match_pct"] = float(silver_adds / bronze_adds) * 100 if bronze_adds > 0 else 0.0
    
    bronze_fills = bronze_df.loc[bronze_df["action"] == "F", "size"].sum()
    silver_fills = df_flow["fill_qty"].sum()
    results["bronze_fill_volume"] = int(bronze_fills)
    results["silver_fill_volume"] = float(silver_fills)
    results["fill_volume_match_pct"] = float(silver_fills / bronze_fills) * 100 if bronze_fills > 0 else 0.0
    
    # Match percentage should be in reasonable range (silver processes subset of bronze window)
    # Silver processes 10-min dev window vs full-day bronze, so expect ~10-15%
    results["pass"] = True  # This is informational, not a pass/fail check
    return results


def run_full_validation(dates: List[str] = DATES) -> Dict[str, Any]:
    """Run full validation across all dates."""
    all_results = {
        "dates": dates,
        "by_date": {},
        "summary": {
            "total_issues": 0,
            "null_inf_issues": 0,
            "formula_issues": 0,
            "constraint_issues": 0,
            "semantic_issues": 0,
        },
    }
    
    for dt in dates:
        print(f"\n{'='*60}")
        print(f"VALIDATING DATE: {dt}")
        print(f"{'='*60}")
        
        date_results = {"dt": dt}
        
        try:
            df_snap = load_silver_data("book_snapshot_1s", dt)
            df_flow = load_silver_data("depth_and_flow_1s", dt)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            date_results["error"] = str(e)
            all_results["by_date"][dt] = date_results
            continue
        
        date_results["row_counts"] = {
            "book_snapshot_1s": len(df_snap),
            "depth_and_flow_1s": len(df_flow),
        }
        print(f"\nRow counts: book_snapshot_1s={len(df_snap):,}, depth_and_flow_1s={len(df_flow):,}")
        
        # 1. Null/NaN/Inf validation
        print("\n1. NULL/NaN/Inf VALIDATION")
        snap_nulls = validate_nulls_inf(df_snap, "book_snapshot_1s")
        flow_nulls = validate_nulls_inf(df_flow, "depth_and_flow_1s")
        date_results["nulls_inf"] = {"book_snapshot_1s": snap_nulls, "depth_and_flow_1s": flow_nulls}
        
        null_pass = snap_nulls["pass"] and flow_nulls["pass"]
        print(f"  book_snapshot_1s: nulls={snap_nulls['null_total']}, inf={snap_nulls['inf_total']} {'PASS' if snap_nulls['pass'] else 'FAIL'}")
        print(f"  depth_and_flow_1s: nulls={flow_nulls['null_total']}, inf={flow_nulls['inf_total']} {'PASS' if flow_nulls['pass'] else 'FAIL'}")
        
        if not null_pass:
            all_results["summary"]["null_inf_issues"] += 1
        
        # 2. Basic statistics
        print("\n2. BASIC STATISTICS")
        snap_stats = compute_basic_stats(df_snap, ["best_bid_price_int", "best_ask_price_int", "mid_price"])
        flow_stats = compute_basic_stats(df_flow, ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty"])
        date_results["statistics"] = {"book_snapshot_1s": snap_stats, "depth_and_flow_1s": flow_stats}
        
        if "mid_price" in snap_stats:
            print(f"  mid_price: min=${snap_stats['mid_price']['min']:.2f}, max=${snap_stats['mid_price']['max']:.2f}, mean=${snap_stats['mid_price']['mean']:.2f}")
        
        # Side distribution
        side_dist = df_flow["side"].value_counts().to_dict()
        date_results["side_distribution"] = side_dist
        total_rows = len(df_flow)
        for side, count in side_dist.items():
            print(f"  side={side}: {count:,} rows ({count/total_rows*100:.1f}%)")
        
        # 3. Mathematical formula verification
        print("\n3. MATHEMATICAL FORMULA VERIFICATION")
        snap_formulas = validate_book_snapshot_formulas(df_snap)
        flow_formulas = validate_depth_and_flow_formulas(df_flow)
        date_results["formulas"] = {"book_snapshot_1s": snap_formulas, "depth_and_flow_1s": flow_formulas}
        
        print(f"  mid_price: max_error={snap_formulas.get('mid_price_max_error', 'N/A'):.2e} {'PASS' if snap_formulas.get('mid_price_formula_valid', False) else 'FAIL'}")
        print(f"  mid_price_int: max_error={snap_formulas.get('mid_price_int_max_error', 'N/A')} {'PASS' if snap_formulas.get('mid_price_int_formula_valid', False) else 'FAIL'}")
        print(f"  depth_qty_start: max_error={flow_formulas.get('depth_qty_start_max_error_clamped', 'N/A'):.2e} {'PASS' if flow_formulas.get('depth_qty_start_formula_valid', False) else 'FAIL'}")
        print(f"  rel_ticks: match={flow_formulas.get('rel_ticks_match_pct', 0):.1f}% {'PASS' if flow_formulas.get('rel_ticks_formula_valid', False) else 'FAIL'}")
        
        formula_pass = snap_formulas.get("pass", False) and flow_formulas.get("pass", False)
        if not formula_pass:
            all_results["summary"]["formula_issues"] += 1
        
        # 4. Sign/direction constraints
        print("\n4. SIGN/DIRECTION CONSTRAINTS")
        constraints = validate_sign_direction_constraints(df_snap, df_flow)
        date_results["constraints"] = constraints
        
        print(f"  bid_price > 0 violations: {constraints.get('bid_price_positive_violation_count', 0)}")
        print(f"  ask_price > 0 violations: {constraints.get('ask_price_positive_violation_count', 0)}")
        print(f"  crossed book violations: {constraints.get('crossed_book_count', 0)}")
        print(f"  side values: {constraints.get('side_values', [])} {'PASS' if constraints.get('side_valid', False) else 'FAIL'}")
        
        neg_counts = constraints.get("negative_quantity_counts", {})
        for col, cnt in neg_counts.items():
            if cnt > 0:
                print(f"  {col} < 0: {cnt} violations")
        
        print(f"  Overall: {'PASS' if constraints.get('pass', False) else 'FAIL'}")
        if not constraints.get("pass", False):
            all_results["summary"]["constraint_issues"] += 1
        
        # 5. Semantic validation
        print("\n5. SEMANTIC VALIDATION")
        semantics = validate_semantic_constraints(df_snap, df_flow, dt)
        date_results["semantics"] = semantics
        
        print(f"  window_order_valid: {'PASS' if semantics.get('window_order_valid', False) else 'FAIL'}")
        print(f"  window_1s_valid: {'PASS' if semantics.get('window_1s_valid', False) else 'FAIL'}")
        print(f"  price_range: ${semantics['price_range']['min']:.2f} - ${semantics['price_range']['max']:.2f}")
        print(f"  price_in_expected_range: {'PASS' if semantics.get('price_in_expected_range', False) else 'FAIL'}")
        print(f"  spread: min=${semantics['spread_stats']['min']:.4f}, max=${semantics['spread_stats']['max']:.4f}, mean=${semantics['spread_stats']['mean']:.4f}")
        print(f"  rel_ticks_range: {semantics['rel_ticks_range']}")
        
        if semantics.get("violations"):
            print(f"  Violations: {semantics['violations']}")
        print(f"  Overall: {'PASS' if semantics.get('pass', False) else 'FAIL'}")
        if not semantics.get("pass", False):
            all_results["summary"]["semantic_issues"] += 1
        
        # 6. Bronze-Silver reconciliation
        print("\n6. BRONZE-SILVER RECONCILIATION")
        recon = validate_bronze_silver_reconciliation(dt, df_flow)
        date_results["reconciliation"] = recon
        
        if "error" not in recon:
            print(f"  add_volume: bronze={recon['bronze_add_volume']:,}, silver={recon['silver_add_volume']:,.0f} ({recon['add_volume_match_pct']:.1f}%)")
            print(f"  fill_volume: bronze={recon['bronze_fill_volume']:,}, silver={recon['silver_fill_volume']:,.0f} ({recon['fill_volume_match_pct']:.1f}%)")
        else:
            print(f"  {recon['error']}")
        
        # Date summary
        date_pass = (
            null_pass
            and formula_pass
            and constraints.get("pass", False)
            and semantics.get("pass", False)
        )
        date_results["overall_pass"] = date_pass
        
        all_results["by_date"][dt] = date_results
    
    # Overall summary
    all_results["summary"]["total_issues"] = (
        all_results["summary"]["null_inf_issues"]
        + all_results["summary"]["formula_issues"]
        + all_results["summary"]["constraint_issues"]
        + all_results["summary"]["semantic_issues"]
    )
    
    all_pass = all(d.get("overall_pass", False) for d in all_results["by_date"].values() if "error" not in d)
    all_results["overall_pass"] = all_pass
    
    return all_results


def main():
    print("=" * 80)
    print("GRUNT VALIDATION: EQUITY MBO SILVER LAYER")
    print("=" * 80)
    
    results = run_full_validation()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for dt, date_results in results["by_date"].items():
        status = "PASS" if date_results.get("overall_pass", False) else "FAIL"
        if "error" in date_results:
            status = "ERROR"
        print(f"  {dt}: {status}")
    
    print(f"\nIssue counts:")
    print(f"  null_inf_issues: {results['summary']['null_inf_issues']}")
    print(f"  formula_issues: {results['summary']['formula_issues']}")
    print(f"  constraint_issues: {results['summary']['constraint_issues']}")
    print(f"  semantic_issues: {results['summary']['semantic_issues']}")
    print(f"  total_issues: {results['summary']['total_issues']}")
    
    print("\n" + "=" * 80)
    if results["overall_pass"]:
        print("VALIDATION PASSED: All formulas and constraints verified")
    else:
        print("VALIDATION FAILED: Issues found (see above)")
    print("=" * 80)
    
    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
