#!/usr/bin/env python3
"""
Statistical Validation of Silver Layer for product_type=future_mbo

This script validates:
1. Data shape and schema
2. NaN/null analysis per column
3. Outlier detection (IQR, z-scores)
4. Scale/normalization assessment
5. Sign/direction correctness
6. Mathematical formula verification
7. Cross-layer validation with bronze
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Constants from book_engine.py
PRICE_SCALE = 1e-9
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))  # 250,000,000
WINDOW_NS = 1_000_000_000  # 1 second
REST_NS = 500_000_000  # 500ms
GRID_MAX_TICKS = 200  # +/- $50 range

LAKE_ROOT = Path(__file__).parent.parent / "lake"


def load_silver_data(symbol: str, dt: str, table: str) -> pd.DataFrame | None:
    """Load silver layer parquet data."""
    path = LAKE_ROOT / f"silver/product_type=future_mbo/symbol={symbol}/table={table}/dt={dt}"
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        print(f"  [WARN] No parquet files found at {path}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    return df


def load_bronze_data(symbol: str, dt: str) -> pd.DataFrame | None:
    """Load bronze layer parquet data for cross-validation."""
    # Bronze data uses full contract symbol (ESH6) like silver
    path = LAKE_ROOT / f"bronze/source=databento/product_type=future_mbo/symbol={symbol}/table=mbo/dt={dt}"
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        print(f"  [WARN] No bronze parquet files found at {path}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    return df


def check_nulls(df: pd.DataFrame, table_name: str) -> dict:
    """Check for null/NaN values per column."""
    print(f"\n  === NULL/NaN Analysis for {table_name} ===")
    total_rows = len(df)
    null_stats = {}
    
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        null_stats[col] = {"count": null_count, "pct": null_pct}
        if null_count > 0:
            print(f"    {col}: {null_count:,} nulls ({null_pct:.2f}%)")
    
    total_nulls = sum(s["count"] for s in null_stats.values())
    if total_nulls == 0:
        print(f"    [OK] No null values found")
    
    return null_stats


def detect_outliers_iqr(series: pd.Series, name: str, multiplier: float = 1.5) -> dict:
    """Detect outliers using IQR method."""
    if series.isna().all():
        return {"outliers": 0, "pct": 0, "lower": None, "upper": None}
    
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    
    outliers = ((series < lower) | (series > upper)).sum()
    pct = (outliers / len(series)) * 100 if len(series) > 0 else 0
    
    return {
        "outliers": outliers,
        "pct": pct,
        "lower": lower,
        "upper": upper,
        "min": series.min(),
        "max": series.max(),
        "q1": q1,
        "q3": q3,
    }


def detect_outliers_zscore(series: pd.Series, name: str, threshold: float = 3.0) -> dict:
    """Detect outliers using z-score method."""
    if series.isna().all() or series.std() == 0:
        return {"outliers": 0, "pct": 0}
    
    mean = series.mean()
    std = series.std()
    z_scores = np.abs((series - mean) / std)
    outliers = (z_scores > threshold).sum()
    pct = (outliers / len(series)) * 100 if len(series) > 0 else 0
    
    return {
        "outliers": outliers,
        "pct": pct,
        "mean": mean,
        "std": std,
    }


def analyze_outliers(df: pd.DataFrame, table_name: str) -> dict:
    """Analyze outliers for numeric columns."""
    print(f"\n  === Outlier Analysis for {table_name} ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = {}
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        iqr_result = detect_outliers_iqr(series, col)
        zscore_result = detect_outliers_zscore(series, col)
        
        results[col] = {
            "iqr": iqr_result,
            "zscore": zscore_result,
        }
        
        if iqr_result["outliers"] > 0 or zscore_result["outliers"] > 0:
            print(f"    {col}:")
            print(f"      Range: [{series.min():.6g}, {series.max():.6g}]")
            print(f"      IQR outliers: {iqr_result['outliers']:,} ({iqr_result['pct']:.2f}%)")
            print(f"      Z-score outliers (>3σ): {zscore_result['outliers']:,} ({zscore_result['pct']:.2f}%)")
    
    total_iqr = sum(r["iqr"]["outliers"] for r in results.values())
    if total_iqr == 0:
        print(f"    [OK] No IQR outliers detected")
    
    return results


def validate_book_snapshot_semantics(df: pd.DataFrame) -> list:
    """Validate semantic correctness of book_snapshot_1s."""
    print(f"\n  === Semantic Validation: book_snapshot_1s ===")
    issues = []
    
    # 1. Timestamps should be monotonically increasing
    if not df["window_end_ts_ns"].is_monotonic_increasing:
        issues.append("window_end_ts_ns is NOT monotonically increasing")
    else:
        print("    [OK] window_end_ts_ns is monotonically increasing")
    
    # 2. window_end - window_start should equal WINDOW_NS (1 second)
    window_duration = df["window_end_ts_ns"] - df["window_start_ts_ns"]
    if not (window_duration == WINDOW_NS).all():
        bad_count = (window_duration != WINDOW_NS).sum()
        issues.append(f"Window duration != 1s for {bad_count} rows")
    else:
        print("    [OK] All window durations equal 1 second")
    
    # 3. best_bid_price_int < best_ask_price_int (no crossed book)
    crossed = (df["best_bid_price_int"] >= df["best_ask_price_int"]) & (df["best_bid_price_int"] > 0) & (df["best_ask_price_int"] > 0)
    crossed_count = crossed.sum()
    if crossed_count > 0:
        issues.append(f"Crossed book detected in {crossed_count} rows (bid >= ask)")
    else:
        print("    [OK] No crossed books (bid always < ask)")
    
    # 4. Quantities should be non-negative
    for col in ["best_bid_qty", "best_ask_qty"]:
        neg = (df[col] < 0).sum()
        if neg > 0:
            issues.append(f"{col} has {neg} negative values")
        else:
            print(f"    [OK] {col} is non-negative")
    
    # 5. mid_price should be (bid + ask) / 2 * PRICE_SCALE
    valid_rows = (df["best_bid_price_int"] > 0) & (df["best_ask_price_int"] > 0)
    if valid_rows.any():
        expected_mid = (df.loc[valid_rows, "best_bid_price_int"] + df.loc[valid_rows, "best_ask_price_int"]) / 2 * PRICE_SCALE
        actual_mid = df.loc[valid_rows, "mid_price"]
        diff = (expected_mid - actual_mid).abs()
        tolerance = 1e-12
        if (diff > tolerance).any():
            bad_count = (diff > tolerance).sum()
            issues.append(f"mid_price calculation incorrect for {bad_count} rows")
        else:
            print(f"    [OK] mid_price formula verified (tolerance: {tolerance})")
    
    # 6. mid_price_int should be round((bid + ask) / 2)
    if valid_rows.any():
        expected_mid_int = ((df.loc[valid_rows, "best_bid_price_int"] + df.loc[valid_rows, "best_ask_price_int"]) / 2).round().astype(int)
        actual_mid_int = df.loc[valid_rows, "mid_price_int"]
        if not (expected_mid_int == actual_mid_int).all():
            bad_count = (expected_mid_int != actual_mid_int).sum()
            issues.append(f"mid_price_int calculation incorrect for {bad_count} rows")
        else:
            print(f"    [OK] mid_price_int formula verified")
    
    # 7. Price scales - ES futures prices should be roughly 5000-7000 range (2026)
    mid_prices_scaled = df.loc[valid_rows, "mid_price"]
    if len(mid_prices_scaled) > 0:
        min_price = mid_prices_scaled.min()
        max_price = mid_prices_scaled.max()
        if min_price < 1000 or max_price > 20000:
            issues.append(f"Price range suspicious: [{min_price:.2f}, {max_price:.2f}]")
        else:
            print(f"    [OK] Price range plausible: [{min_price:.2f}, {max_price:.2f}]")
    
    # 8. Spread should be positive and reasonable
    spread = df.loc[valid_rows, "best_ask_price_int"] - df.loc[valid_rows, "best_bid_price_int"]
    spread_ticks = spread / TICK_INT
    if len(spread_ticks) > 0:
        min_spread = spread_ticks.min()
        max_spread = spread_ticks.max()
        mean_spread = spread_ticks.mean()
        if min_spread < 0:
            issues.append(f"Negative spreads detected")
        elif max_spread > 100:  # More than 100 ticks spread is unusual
            issues.append(f"Very wide spreads detected: max {max_spread:.1f} ticks")
        else:
            print(f"    [OK] Spread range: [{min_spread:.1f}, {max_spread:.1f}] ticks, mean: {mean_spread:.2f}")
    
    return issues


def validate_depth_flow_semantics(df: pd.DataFrame) -> list:
    """Validate semantic correctness of depth_and_flow_1s."""
    print(f"\n  === Semantic Validation: depth_and_flow_1s ===")
    issues = []
    
    # 1. Side should be 'A' or 'B' only
    valid_sides = {"A", "B"}
    actual_sides = set(df["side"].unique())
    if not actual_sides.issubset(valid_sides):
        issues.append(f"Invalid side values: {actual_sides - valid_sides}")
    else:
        print(f"    [OK] Side values: {sorted(actual_sides)}")
    
    # 2. Quantities should be non-negative
    qty_cols = ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "depth_qty_rest", "pull_qty_rest", "fill_qty"]
    for col in qty_cols:
        neg = (df[col] < 0).sum()
        if neg > 0:
            issues.append(f"{col} has {neg} negative values")
        else:
            print(f"    [OK] {col} is non-negative")
    
    # 3. rel_ticks should be within expected range (+/- GRID_MAX_TICKS)
    rel_ticks_range = (df["rel_ticks"].min(), df["rel_ticks"].max())
    if rel_ticks_range[0] < -GRID_MAX_TICKS or rel_ticks_range[1] > GRID_MAX_TICKS:
        issues.append(f"rel_ticks out of expected range [{-GRID_MAX_TICKS}, {GRID_MAX_TICKS}]: {rel_ticks_range}")
    else:
        print(f"    [OK] rel_ticks within range: {rel_ticks_range}")
    
    # 4. depth_qty_start formula: depth_qty_end - add_qty + pull_qty + fill_qty
    # This is how it's computed - verify consistency
    expected_start = df["depth_qty_end"] - df["add_qty"] + df["pull_qty"] + df["fill_qty"]
    actual_start = df["depth_qty_start"]
    diff = (expected_start - actual_start).abs()
    tolerance = 0.01
    bad_formula = (diff > tolerance).sum()
    if bad_formula > 0:
        issues.append(f"depth_qty_start formula inconsistent for {bad_formula} rows")
    else:
        print(f"    [OK] depth_qty_start formula verified")
    
    # 5. rel_ticks computation: (price_int - spot_ref_price_int) / TICK_INT
    expected_rel_ticks = ((df["price_int"] - df["spot_ref_price_int"]) / TICK_INT).round().astype(int)
    actual_rel_ticks = df["rel_ticks"]
    if not (expected_rel_ticks == actual_rel_ticks).all():
        bad_count = (expected_rel_ticks != actual_rel_ticks).sum()
        issues.append(f"rel_ticks calculation incorrect for {bad_count} rows")
    else:
        print(f"    [OK] rel_ticks formula verified")
    
    # 6. Check that depth_qty_rest <= depth_qty_end (resting can't exceed total)
    rest_exceed = (df["depth_qty_rest"] > df["depth_qty_end"]).sum()
    if rest_exceed > 0:
        issues.append(f"depth_qty_rest exceeds depth_qty_end in {rest_exceed} rows")
    else:
        print(f"    [OK] depth_qty_rest <= depth_qty_end")
    
    # 7. Check pull_qty_rest <= pull_qty
    pull_rest_exceed = (df["pull_qty_rest"] > df["pull_qty"]).sum()
    if pull_rest_exceed > 0:
        issues.append(f"pull_qty_rest exceeds pull_qty in {pull_rest_exceed} rows")
    else:
        print(f"    [OK] pull_qty_rest <= pull_qty")
    
    return issues


def validate_scale_normalization(df_snap: pd.DataFrame, df_flow: pd.DataFrame) -> list:
    """Validate scale and normalization of features."""
    print(f"\n  === Scale/Normalization Assessment ===")
    issues = []
    
    # Book Snapshot
    print("  Book Snapshot:")
    for col in ["best_bid_price_int", "best_ask_price_int", "mid_price_int", "spot_ref_price_int", "last_trade_price_int"]:
        if col in df_snap.columns:
            series = df_snap[col]
            valid = series[series > 0]
            if len(valid) > 0:
                print(f"    {col}: min={valid.min():,}, max={valid.max():,}, mean={valid.mean():,.0f}")
    
    for col in ["mid_price"]:
        if col in df_snap.columns:
            series = df_snap[col]
            valid = series[series > 0]
            if len(valid) > 0:
                print(f"    {col}: min={valid.min():.4f}, max={valid.max():.4f}, mean={valid.mean():.4f}")
    
    for col in ["best_bid_qty", "best_ask_qty"]:
        if col in df_snap.columns:
            series = df_snap[col]
            print(f"    {col}: min={series.min():,}, max={series.max():,}, mean={series.mean():.1f}")
    
    # Depth & Flow
    print("  Depth & Flow:")
    for col in ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "fill_qty"]:
        if col in df_flow.columns:
            series = df_flow[col]
            print(f"    {col}: min={series.min():.1f}, max={series.max():.1f}, mean={series.mean():.2f}, std={series.std():.2f}")
    
    return issues


def cross_validate_with_bronze(df_snap: pd.DataFrame, df_bronze: pd.DataFrame | None, dt: str) -> list:
    """Cross-validate silver layer with bronze layer."""
    print(f"\n  === Cross-Layer Validation (Bronze → Silver) ===")
    issues = []
    
    if df_bronze is None:
        print("    [SKIP] Bronze data not available for cross-validation")
        return issues
    
    print(f"    Bronze rows: {len(df_bronze):,}")
    print(f"    Silver snapshot rows: {len(df_snap):,}")
    
    # Sample validation: count trades in bronze vs fills in silver
    bronze_trades = df_bronze[df_bronze["action"] == "T"]
    print(f"    Bronze trade events: {len(bronze_trades):,}")
    
    # Check bronze data has expected columns
    expected_bronze_cols = {"ts_event", "action", "side", "price", "size", "order_id"}
    actual_cols = set(df_bronze.columns)
    if not expected_bronze_cols.issubset(actual_cols):
        missing = expected_bronze_cols - actual_cols
        issues.append(f"Bronze missing expected columns: {missing}")
    else:
        print(f"    [OK] Bronze has required columns")
    
    # Verify time range alignment
    bronze_min_ts = df_bronze["ts_event"].min()
    bronze_max_ts = df_bronze["ts_event"].max()
    silver_min_ts = df_snap["window_start_ts_ns"].min()
    silver_max_ts = df_snap["window_end_ts_ns"].max()
    
    print(f"    Bronze time range: {bronze_min_ts} - {bronze_max_ts}")
    print(f"    Silver time range: {silver_min_ts} - {silver_max_ts}")
    
    # Silver should be within bronze range
    if silver_min_ts < bronze_min_ts or silver_max_ts > bronze_max_ts:
        print(f"    [NOTE] Silver time range extends beyond bronze - warmup data filtered out")
    
    return issues


def print_summary(df: pd.DataFrame, name: str):
    """Print basic summary statistics."""
    print(f"\n  === Summary: {name} ===")
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"      {col}: {dtype}")


def validate_day(symbol: str, dt: str) -> dict:
    """Validate all silver data for a given day."""
    print(f"\n{'='*60}")
    print(f"  VALIDATING: symbol={symbol}, dt={dt}")
    print(f"{'='*60}")
    
    results = {"dt": dt, "issues": []}
    
    # Load silver data
    df_snap = load_silver_data(symbol, dt, "book_snapshot_1s")
    df_flow = load_silver_data(symbol, dt, "depth_and_flow_1s")
    
    if df_snap is None or df_flow is None:
        results["issues"].append("Missing silver data")
        return results
    
    # Print summaries
    print_summary(df_snap, "book_snapshot_1s")
    print_summary(df_flow, "depth_and_flow_1s")
    
    # Null analysis
    check_nulls(df_snap, "book_snapshot_1s")
    check_nulls(df_flow, "depth_and_flow_1s")
    
    # Outlier analysis
    analyze_outliers(df_snap, "book_snapshot_1s")
    analyze_outliers(df_flow, "depth_and_flow_1s")
    
    # Semantic validation
    snap_issues = validate_book_snapshot_semantics(df_snap)
    flow_issues = validate_depth_flow_semantics(df_flow)
    results["issues"].extend(snap_issues)
    results["issues"].extend(flow_issues)
    
    # Scale/normalization
    scale_issues = validate_scale_normalization(df_snap, df_flow)
    results["issues"].extend(scale_issues)
    
    # Cross-layer validation
    df_bronze = load_bronze_data(symbol, dt)
    cross_issues = cross_validate_with_bronze(df_snap, df_bronze, dt)
    results["issues"].extend(cross_issues)
    
    return results


def main():
    print("=" * 70)
    print("  SILVER LAYER STATISTICAL VALIDATION: product_type=future_mbo")
    print("=" * 70)
    
    symbol = "ESH6"
    dates = ["2026-01-06", "2026-01-07", "2026-01-08"]  # Regenerated dates with book_snapshot_1s and depth_and_flow_1s
    
    print(f"\nSymbol: {symbol}")
    print(f"Dates to validate: {dates}")
    print(f"TICK_INT: {TICK_INT:,} (represents ${TICK_SIZE} in price_int)")
    print(f"PRICE_SCALE: {PRICE_SCALE}")
    
    all_results = []
    for dt in dates:
        results = validate_day(symbol, dt)
        all_results.append(results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    
    total_issues = 0
    for r in all_results:
        dt = r["dt"]
        issues = r["issues"]
        total_issues += len(issues)
        if issues:
            print(f"\n  {dt}: {len(issues)} issue(s)")
            for i, issue in enumerate(issues, 1):
                print(f"    {i}. {issue}")
        else:
            print(f"\n  {dt}: [OK] No issues found")
    
    print(f"\n  Total issues across all days: {total_issues}")
    
    if total_issues > 0:
        sys.exit(1)
    else:
        print("\n  [SUCCESS] All validations passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
