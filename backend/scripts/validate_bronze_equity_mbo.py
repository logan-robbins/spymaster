#!/usr/bin/env python3
"""
Bronze equity_mbo validation script.
Validates schema, semantics, and statistics for QQQ MBO data.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

BRONZE_ROOT = Path(__file__).parent.parent / "lake" / "bronze" / "source=databento" / "product_type=equity_mbo"
EXPECTED_SYMBOL = "QQQ"

# Expected schema fields from avsc contract
EXPECTED_FIELDS = {
    "ts_recv": "int64",
    "size": "int64",
    "ts_event": "int64",
    "channel_id": "int64",
    "rtype": "int64",
    "order_id": "int64",
    "publisher_id": "int64",
    "flags": "int64",
    "instrument_id": "int64",
    "ts_in_delta": "int64",
    "action": "object",  # string
    "sequence": "int64",
    "side": "object",  # string
    "symbol": "object",  # string
    "price": "int64",
}

# Expected MBO action codes (Databento)
VALID_ACTIONS = {"A", "C", "M", "R", "T", "F", "B", "S", "E"}  # Add, Cancel, Modify, Clear, Trade, Fill, etc.
VALID_SIDES = {"A", "B", "N"}  # Ask, Bid, None

# QQQ price range from README: ~$620-$630 range
# Price is in fixed-point with 1e-9 scale for equities
PRICE_SCALE = 1e-9
EXPECTED_MIN_PRICE = 600.0
EXPECTED_MAX_PRICE = 650.0


def validate_date(dt: str, verbose: bool = False) -> dict:
    """Validate bronze equity_mbo data for a single date."""
    results = {"dt": dt, "status": "PASS", "errors": [], "warnings": [], "stats": {}}
    
    partition_path = BRONZE_ROOT / f"symbol={EXPECTED_SYMBOL}" / "table=mbo" / f"dt={dt}"
    
    if not partition_path.exists():
        results["status"] = "FAIL"
        results["errors"].append(f"Partition not found: {partition_path}")
        return results
    
    # Find parquet files
    parquet_files = list(partition_path.glob("part-*.parquet"))
    if not parquet_files:
        parquet_files = list(partition_path.glob("*.parquet"))
    
    if not parquet_files:
        results["status"] = "FAIL"
        results["errors"].append(f"No parquet files in {partition_path}")
        return results
    
    # Read data (handle potential schema differences between files)
    dfs = []
    for f in parquet_files:
        # Use ParquetFile for single file reads to avoid schema merge issues
        pf = pq.ParquetFile(str(f))
        df_part = pf.read().to_pandas()
        # Normalize string columns to avoid dictionary encoding mismatches
        for col in ["symbol", "action", "side"]:
            if col in df_part.columns:
                df_part[col] = df_part[col].astype(str)
        dfs.append(df_part)
    df = pd.concat(dfs, ignore_index=True)
    results["stats"]["row_count"] = len(df)
    
    if verbose:
        print(f"\n=== Validating {dt} ===")
        print(f"Row count: {len(df):,}")
    
    # 1. SCHEMA VALIDATION
    schema_errors = []
    for field, expected_dtype in EXPECTED_FIELDS.items():
        if field not in df.columns:
            schema_errors.append(f"Missing column: {field}")
        else:
            actual_dtype = str(df[field].dtype)
            if expected_dtype == "object":
                if actual_dtype not in ["object", "string"]:
                    schema_errors.append(f"{field}: expected string-like, got {actual_dtype}")
            elif expected_dtype == "int64":
                if actual_dtype not in ["int64", "Int64"]:
                    schema_errors.append(f"{field}: expected int64, got {actual_dtype}")
    
    extra_cols = set(df.columns) - set(EXPECTED_FIELDS.keys())
    if extra_cols:
        results["warnings"].append(f"Extra columns (ok): {extra_cols}")
    
    if schema_errors:
        results["status"] = "FAIL"
        results["errors"].extend(schema_errors)
        return results
    
    if verbose:
        print(f"Schema: PASS ({len(EXPECTED_FIELDS)} fields)")
    
    # 2. SEMANTIC VALIDATION
    
    # 2a. Symbol must be QQQ throughout
    unique_symbols = df["symbol"].unique().tolist()
    if unique_symbols != [EXPECTED_SYMBOL]:
        results["warnings"].append(f"Unexpected symbols: {unique_symbols}")
    results["stats"]["unique_symbols"] = unique_symbols
    
    # 2b. Timestamp ordering (events should be chronological)
    ts_sorted = df["ts_event"].is_monotonic_increasing
    if not ts_sorted:
        # Check if sorted by (ts_event, sequence)
        df_check = df.sort_values(["ts_event", "sequence"])
        if (df_check["ts_event"].values != df["ts_event"].values).any():
            results["warnings"].append("Data not strictly sorted by ts_event (may have same-ts events)")
    results["stats"]["ts_monotonic"] = ts_sorted
    
    if verbose:
        print(f"Timestamp ordering: {'strict monotonic' if ts_sorted else 'non-strict (same-ts events)'}")
    
    # 2c. Valid action codes
    invalid_actions = set(df["action"].unique()) - VALID_ACTIONS
    if invalid_actions:
        results["warnings"].append(f"Unknown action codes: {invalid_actions}")
    results["stats"]["action_distribution"] = df["action"].value_counts().to_dict()
    
    if verbose:
        print(f"Actions: {df['action'].value_counts().to_dict()}")
    
    # 2d. Valid side codes
    invalid_sides = set(df["side"].unique()) - VALID_SIDES
    if invalid_sides:
        results["warnings"].append(f"Unknown side codes: {invalid_sides}")
    results["stats"]["side_distribution"] = df["side"].value_counts().to_dict()
    
    if verbose:
        print(f"Sides: {df['side'].value_counts().to_dict()}")
    
    # 3. STATISTICAL VALIDATION
    
    # 3a. Null/NaN counts
    null_counts = df.isnull().sum()
    critical_nulls = {col: int(null_counts[col]) for col in ["ts_event", "price", "size", "action", "side", "order_id"] if null_counts[col] > 0}
    if critical_nulls:
        results["errors"].append(f"Critical nulls: {critical_nulls}")
        results["status"] = "FAIL"
    results["stats"]["null_counts"] = {k: int(v) for k, v in null_counts.items() if v > 0}
    
    if verbose:
        print(f"Null counts: {results['stats']['null_counts'] or 'None'}")
    
    # 3b. Price range sanity
    # Filter for add/modify actions which should have valid prices
    df_priced = df[df["action"].isin({"A", "M"})].copy()
    if len(df_priced) > 0:
        prices_dollars = df_priced["price"] * PRICE_SCALE
        min_price = prices_dollars.min()
        max_price = prices_dollars.max()
        mean_price = prices_dollars.mean()
        
        results["stats"]["price_min_dollars"] = float(min_price)
        results["stats"]["price_max_dollars"] = float(max_price)
        results["stats"]["price_mean_dollars"] = float(mean_price)
        
        if min_price < EXPECTED_MIN_PRICE or max_price > EXPECTED_MAX_PRICE:
            results["warnings"].append(f"Price outside expected range: ${min_price:.2f} - ${max_price:.2f}")
        
        if verbose:
            print(f"Price range: ${min_price:.2f} - ${max_price:.2f} (mean: ${mean_price:.2f})")
    
    # 3c. Size distribution
    sizes = df_priced["size"] if len(df_priced) > 0 else df["size"]
    size_stats = {
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
        "median": float(sizes.median()),
        "p99": float(sizes.quantile(0.99)),
    }
    results["stats"]["size_stats"] = size_stats
    
    # Check for extreme outliers (> 100x median)
    outlier_threshold = max(1000, size_stats["median"] * 100)
    extreme_sizes = (sizes > outlier_threshold).sum()
    if extreme_sizes > 0:
        results["warnings"].append(f"Extreme size outliers (>{outlier_threshold:,.0f}): {extreme_sizes:,} rows")
    
    if verbose:
        print(f"Size stats: min={size_stats['min']}, max={size_stats['max']:,}, mean={size_stats['mean']:.1f}, median={size_stats['median']:.1f}, p99={size_stats['p99']:.1f}")
    
    # 3d. Negative values check
    negative_sizes = (df["size"] < 0).sum()
    negative_prices = (df_priced["price"] < 0).sum() if len(df_priced) > 0 else 0
    if negative_sizes > 0:
        results["errors"].append(f"Negative sizes: {negative_sizes}")
        results["status"] = "FAIL"
    if negative_prices > 0:
        results["errors"].append(f"Negative prices: {negative_prices}")
        results["status"] = "FAIL"
    
    # 3e. Order ID distribution (should have many unique IDs)
    unique_order_ids = df["order_id"].nunique()
    results["stats"]["unique_order_ids"] = unique_order_ids
    
    if verbose:
        print(f"Unique order IDs: {unique_order_ids:,}")
    
    # 3f. Timestamp range
    ts_min = pd.Timestamp(df["ts_event"].min(), unit="ns")
    ts_max = pd.Timestamp(df["ts_event"].max(), unit="ns")
    results["stats"]["ts_range"] = f"{ts_min} - {ts_max}"
    
    if verbose:
        print(f"Timestamp range: {ts_min} - {ts_max}")
    
    if verbose:
        print(f"\nResult: {results['status']}")
        if results["errors"]:
            print(f"Errors: {results['errors']}")
        if results["warnings"]:
            print(f"Warnings: {results['warnings']}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", help="Dates to validate (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    dates = args.dates or ["2026-01-13", "2026-01-22", "2026-01-28"]
    
    print(f"Validating {len(dates)} bronze equity_mbo dates...")
    all_results = []
    
    for dt in dates:
        result = validate_date(dt, verbose=args.verbose)
        all_results.append(result)
        
        status_emoji = "✅" if result["status"] == "PASS" else "❌"
        print(f"{status_emoji} {dt}: {result['status']} - {result['stats'].get('row_count', 0):,} rows")
    
    # Cross-date consistency check
    print("\n=== Cross-Date Consistency ===")
    row_counts = [r["stats"].get("row_count", 0) for r in all_results]
    print(f"Row counts: {row_counts}")
    print(f"Row count range: {min(row_counts):,} - {max(row_counts):,}")
    
    price_means = [r["stats"].get("price_mean_dollars", 0) for r in all_results if r["stats"].get("price_mean_dollars")]
    if price_means:
        print(f"Mean prices: ${min(price_means):.2f} - ${max(price_means):.2f}")
    
    # Summary
    passes = sum(1 for r in all_results if r["status"] == "PASS")
    fails = len(all_results) - passes
    print(f"\n=== Summary ===")
    print(f"PASS: {passes}/{len(all_results)}")
    if fails > 0:
        print(f"FAIL: {fails}/{len(all_results)}")
        for r in all_results:
            if r["status"] == "FAIL":
                print(f"  {r['dt']}: {r['errors']}")
    
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
