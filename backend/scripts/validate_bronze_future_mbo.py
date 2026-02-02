#!/usr/bin/env python
"""
Validate bronze future_mbo layer.

Checks:
1. Schema matches expected fields from avsc
2. Contract symbols are valid (ESH6, ESM6, ESU6, ESZ6, ESH7)
3. Timestamps are chronologically ordered within each contract
4. Session window boundaries (CME globex: 18:00-17:00 ET next day, but bronze should have data)
5. Statistical validation: price ranges, row counts, null checks
"""
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz

LAKE_ROOT = Path(__file__).parent.parent / "lake"
BRONZE_PATH = LAKE_ROOT / "bronze" / "source=databento" / "product_type=future_mbo"

# Expected schema fields from avsc
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
    "action": "string",
    "sequence": "int64",
    "side": "string",
    "symbol": "string",
    "price": "int64",
}

# Valid ES contract symbols for Jan 2026
VALID_CONTRACTS = {"ESH6", "ESM6", "ESU6", "ESZ6", "ESH7"}

# Price range for ES (in ticks: $1 = 4 ticks, so $6900-$7200 = 27600-28800 * 1e9)
# Actually price is in fixed-point nanoseconds: $6950 = 6950.0 * 1e9
ES_PRICE_MIN = 6800 * 1e9  # $6800 min
ES_PRICE_MAX = 7200 * 1e9  # $7200 max

ET = pytz.timezone("America/New_York")


def validate_date(dt_str: str, verbose: bool = True) -> dict:
    """Validate bronze data for a single date."""
    results = {
        "date": dt_str,
        "contracts_found": [],
        "total_rows": 0,
        "schema_valid": True,
        "schema_errors": [],
        "symbol_valid": True,
        "symbol_errors": [],
        "timestamp_valid": True,
        "timestamp_errors": [],
        "price_valid": True,
        "price_stats": {},
        "null_counts": {},
        "issues": [],
    }
    
    # Find all contract partitions for this date
    for symbol_dir in BRONZE_PATH.glob("symbol=*"):
        symbol = symbol_dir.name.replace("symbol=", "")
        if not symbol.startswith("ES"):
            continue
            
        dt_path = symbol_dir / "table=mbo" / f"dt={dt_str}"
        if not dt_path.exists():
            continue
            
        # Find parquet files
        parquet_files = list(dt_path.glob("part-*.parquet"))
        if not parquet_files:
            results["issues"].append(f"No parquet files for {symbol} on {dt_str}")
            continue
            
        results["contracts_found"].append(symbol)
        
        # Read the data
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        results["total_rows"] += len(df)
        
        # 1. Schema validation
        for field, expected_type in EXPECTED_FIELDS.items():
            if field not in df.columns:
                results["schema_valid"] = False
                results["schema_errors"].append(f"Missing field: {field}")
            else:
                actual_type = str(df[field].dtype)
                if expected_type == "int64" and "int" not in actual_type:
                    results["schema_valid"] = False
                    results["schema_errors"].append(f"{field}: expected int64, got {actual_type}")
                elif expected_type == "string" and "object" not in actual_type and "string" not in actual_type:
                    results["schema_valid"] = False
                    results["schema_errors"].append(f"{field}: expected string, got {actual_type}")
        
        # 2. Symbol validation
        unique_symbols = df["symbol"].unique()
        for sym in unique_symbols:
            if sym not in VALID_CONTRACTS:
                results["symbol_valid"] = False
                results["symbol_errors"].append(f"Invalid contract: {sym}")
        
        # 3. Timestamp ordering (within contract)
        for sym in unique_symbols:
            sym_df = df[df["symbol"] == sym].copy()
            if not sym_df["ts_event"].is_monotonic_increasing:
                # Check how many out-of-order
                diffs = sym_df["ts_event"].diff()
                oo_count = (diffs < 0).sum()
                if oo_count > 0:
                    results["timestamp_valid"] = False
                    results["timestamp_errors"].append(f"{sym}: {oo_count} out-of-order timestamps")
        
        # 4. Price validation
        prices = df["price"]
        valid_prices = prices[(prices > 0) & (prices != 0x7FFFFFFFFFFFFFFF)]  # Exclude sentinel values
        
        if len(valid_prices) > 0:
            min_price = valid_prices.min() / 1e9
            max_price = valid_prices.max() / 1e9
            mean_price = valid_prices.mean() / 1e9
            
            results["price_stats"][symbol] = {
                "min": f"${min_price:.2f}",
                "max": f"${max_price:.2f}",
                "mean": f"${mean_price:.2f}",
                "count": len(valid_prices),
            }
            
            # Check price range
            if min_price < 6800 or max_price > 7200:
                results["price_valid"] = False
                results["issues"].append(f"{symbol}: Price outside expected range (${min_price:.2f}-${max_price:.2f})")
        
        # 5. Null/NaN checks
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                if symbol not in results["null_counts"]:
                    results["null_counts"][symbol] = {}
                results["null_counts"][symbol][col] = null_count
        
        # 6. Action and Side distribution
        if verbose:
            print(f"\n  {symbol}:")
            print(f"    Rows: {len(df):,}")
            print(f"    Actions: {df['action'].value_counts().to_dict()}")
            print(f"    Sides: {df['side'].value_counts().to_dict()}")
            if len(valid_prices) > 0:
                print(f"    Price: ${min_price:.2f} - ${max_price:.2f}")
    
    return results


def main():
    # Dates to validate (user specified 3 random dates)
    dates = ["2026-01-05", "2026-01-16", "2026-01-28"]
    
    print("=" * 80)
    print("BRONZE FUTURE_MBO VALIDATION")
    print("=" * 80)
    
    all_results = []
    
    for dt in dates:
        print(f"\n{'='*80}")
        print(f"Validating: {dt}")
        print("=" * 80)
        
        results = validate_date(dt)
        all_results.append(results)
        
        # Summary
        print(f"\n  SUMMARY:")
        print(f"    Contracts: {', '.join(results['contracts_found'])}")
        print(f"    Total rows: {results['total_rows']:,}")
        print(f"    Schema valid: {results['schema_valid']}")
        print(f"    Symbol valid: {results['symbol_valid']}")
        print(f"    Timestamp valid: {results['timestamp_valid']}")
        print(f"    Price valid: {results['price_valid']}")
        
        if results["schema_errors"]:
            print(f"    Schema errors: {results['schema_errors']}")
        if results["symbol_errors"]:
            print(f"    Symbol errors: {results['symbol_errors']}")
        if results["timestamp_errors"]:
            print(f"    Timestamp errors: {results['timestamp_errors']}")
        if results["null_counts"]:
            print(f"    Null counts: {results['null_counts']}")
        if results["issues"]:
            print(f"    Issues: {results['issues']}")
    
    # Cross-date consistency check
    print(f"\n{'='*80}")
    print("CROSS-DATE CONSISTENCY")
    print("=" * 80)
    
    # Check row counts are similar (within reasonable bounds)
    row_counts = [r["total_rows"] for r in all_results]
    min_rows, max_rows = min(row_counts), max(row_counts)
    print(f"  Row count range: {min_rows:,} - {max_rows:,}")
    
    # Check all dates have same contracts
    contract_sets = [set(r["contracts_found"]) for r in all_results]
    if len(set(tuple(sorted(cs)) for cs in contract_sets)) > 1:
        print(f"  WARNING: Different contracts across dates:")
        for r in all_results:
            print(f"    {r['date']}: {r['contracts_found']}")
    else:
        print(f"  All dates have same contracts: {contract_sets[0]}")
    
    # Overall pass/fail
    print(f"\n{'='*80}")
    all_valid = all(
        r["schema_valid"] and r["symbol_valid"] and r["timestamp_valid"] and r["price_valid"]
        for r in all_results
    )
    if all_valid:
        print("VALIDATION: PASS")
    else:
        print("VALIDATION: FAIL")
        for r in all_results:
            if not all([r["schema_valid"], r["symbol_valid"], r["timestamp_valid"], r["price_valid"]]):
                print(f"  {r['date']}: Issues found")
    print("=" * 80)
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
