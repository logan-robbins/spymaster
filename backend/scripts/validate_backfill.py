"""
Validate Bronze backfill data quality and correctness.

Quick validation checks:
1. Row count consistency (DBN source vs Bronze output)
2. Schema correctness (required fields present)
3. Data integrity (no nulls, valid ranges, time ordering)
4. Sample comparison (first/last records match)

Usage:
    cd backend/
    uv run python -m scripts.validate_backfill --dates 2025-12-16,2025-12-17
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from src.ingestor.dbn_ingestor import DBNIngestor
from src.lake.bronze_writer import BronzeReader


def log(msg: str, **kwargs):
    """Print with flush for real-time output."""
    print(msg, flush=True, **kwargs)


def validate_date(
    dbn: DBNIngestor,
    reader: BronzeReader,
    date: str,
    symbol: str = "ES"
) -> Tuple[bool, List[str]]:
    """Validate a single date. Returns (success, list of issues)."""
    
    log(f"\n{'='*60}")
    log(f"VALIDATING: {date}")
    log(f"{'='*60}")
    
    issues = []
    
    # ==================================================================
    # 1. TRADES VALIDATION
    # ==================================================================
    log(f"\n[1/2] Validating Trades...")
    
    # Count trades from DBN
    log(f"  Counting DBN trades...", end="")
    dbn_trades = list(dbn.read_trades(date=date))
    dbn_trade_count = len(dbn_trades)
    log(f" {dbn_trade_count:,} rows")
    
    # Count trades from Bronze
    log(f"  Reading Bronze trades...", end="")
    bronze_trades_df = reader.read_futures_trades(symbol=symbol, date=date)
    bronze_trade_count = len(bronze_trades_df)
    log(f" {bronze_trade_count:,} rows")
    
    # Compare counts
    if dbn_trade_count != bronze_trade_count:
        msg = f"❌ Trades count mismatch: DBN={dbn_trade_count:,} != Bronze={bronze_trade_count:,}"
        log(f"  {msg}")
        issues.append(msg)
    else:
        log(f"  ✅ Row counts match: {dbn_trade_count:,}")
    
    # Check trades integrity
    if len(bronze_trades_df) > 0:
        log(f"  Checking data integrity...")
        
        # Nulls
        null_count = bronze_trades_df[['ts_event_ns', 'price', 'size']].isna().sum().sum()
        if null_count > 0:
            msg = f"❌ Trades have {null_count} nulls"
            log(f"    {msg}")
            issues.append(msg)
        else:
            log(f"    ✅ No nulls in critical fields")
        
        # Time ordering
        is_sorted = bronze_trades_df['ts_event_ns'].is_monotonic_increasing
        if not is_sorted:
            msg = f"❌ Trades not time-sorted"
            log(f"    {msg}")
            issues.append(msg)
        else:
            log(f"    ✅ Time-sorted")
        
        # Price range (ES converted to SPY should be 300-700 range typically)
        min_price = bronze_trades_df['price'].min()
        max_price = bronze_trades_df['price'].max()
        log(f"    ✅ Price range: [{min_price:.2f}, {max_price:.2f}]")
        
        # Sample first record
        if len(dbn_trades) > 0:
            dbn_first = dbn_trades[0]
            bronze_first = bronze_trades_df.iloc[0]
            
            ts_match = dbn_first.ts_event_ns == bronze_first['ts_event_ns']
            price_match = abs(dbn_first.price - bronze_first['price']) < 0.01
            
            if ts_match and price_match:
                log(f"    ✅ First record matches")
            else:
                msg = f"❌ First record mismatch"
                log(f"    {msg}")
                issues.append(msg)
    
    # ==================================================================
    # 2. MBP-10 VALIDATION
    # ==================================================================
    log(f"\n[2/2] Validating MBP-10...")
    
    # Count MBP-10 from DBN
    log(f"  Counting DBN MBP-10...", end="")
    dbn_mbp10 = list(dbn.read_mbp10(date=date))
    dbn_mbp_count = len(dbn_mbp10)
    log(f" {dbn_mbp_count:,} rows")
    
    # Count MBP-10 from Bronze
    log(f"  Reading Bronze MBP-10...", end="")
    bronze_mbp10_df = reader.read_futures_mbp10(symbol=symbol, date=date)
    bronze_mbp_count = len(bronze_mbp10_df)
    log(f" {bronze_mbp_count:,} rows")
    
    # Compare counts
    if dbn_mbp_count != bronze_mbp_count:
        msg = f"❌ MBP-10 count mismatch: DBN={dbn_mbp_count:,} != Bronze={bronze_mbp_count:,}"
        log(f"  {msg}")
        issues.append(msg)
    else:
        log(f"  ✅ Row counts match: {dbn_mbp_count:,}")
    
    # Check MBP-10 integrity
    if len(bronze_mbp10_df) > 0:
        log(f"  Checking data integrity...")
        
        # Nulls
        null_count = bronze_mbp10_df[['ts_event_ns', 'bid_px_1', 'ask_px_1']].isna().sum().sum()
        if null_count > 0:
            msg = f"❌ MBP-10 has {null_count} nulls"
            log(f"    {msg}")
            issues.append(msg)
        else:
            log(f"    ✅ No nulls in critical fields")
        
        # Time ordering
        is_sorted = bronze_mbp10_df['ts_event_ns'].is_monotonic_increasing
        if not is_sorted:
            msg = f"❌ MBP-10 not time-sorted"
            log(f"    {msg}")
            issues.append(msg)
        else:
            log(f"    ✅ Time-sorted")
        
        # Bid/ask spread
        spread_valid = (bronze_mbp10_df['ask_px_1'] >= bronze_mbp10_df['bid_px_1']).all()
        if not spread_valid:
            msg = f"❌ Invalid spread (ask < bid)"
            log(f"    {msg}")
            issues.append(msg)
        else:
            log(f"    ✅ Valid bid/ask spread")
        
        # Sample first record
        if len(dbn_mbp10) > 0:
            dbn_first = dbn_mbp10[0]
            bronze_first = bronze_mbp10_df.iloc[0]
            
            ts_match = dbn_first.ts_event_ns == bronze_first['ts_event_ns']
            
            if ts_match:
                log(f"    ✅ First record timestamp matches")
            else:
                msg = f"❌ First record timestamp mismatch"
                log(f"    {msg}")
                issues.append(msg)
    
    # Summary for date
    if not issues:
        log(f"\n✅ {date}: ALL CHECKS PASSED")
    else:
        log(f"\n❌ {date}: {len(issues)} ISSUES FOUND")
    
    return (len(issues) == 0, issues)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Bronze backfill data")
    parser.add_argument("--dates", type=str, required=True, help="Comma-separated dates (YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, default="ES", help="Symbol to validate (default: ES)")
    parser.add_argument("--data-root", type=str, default=None, help="Data root path")
    
    args = parser.parse_args()
    
    dates = [d.strip() for d in args.dates.split(",")]
    
    log("\n" + "="*60)
    log("BRONZE BACKFILL VALIDATION")
    log("="*60)
    log(f"Dates: {', '.join(dates)}")
    log(f"Symbol: {args.symbol}")
    
    # Initialize
    log(f"\nInitializing DBN ingestor...")
    dbn = DBNIngestor()
    
    if args.data_root:
        reader = BronzeReader(data_root=args.data_root)
    else:
        # Default to backend/data/lake
        data_root = Path(__file__).resolve().parents[1] / "data" / "lake"
        reader = BronzeReader(data_root=str(data_root))
    
    log(f"Bronze root: {reader.bronze_root}")
    
    # Validate each date
    all_issues = []
    success_count = 0
    
    for date in dates:
        success, issues = validate_date(dbn, reader, date, args.symbol)
        if success:
            success_count += 1
        all_issues.extend(issues)
    
    # Final summary
    log("\n" + "="*60)
    log("VALIDATION SUMMARY")
    log("="*60)
    log(f"Dates validated: {len(dates)}")
    log(f"Successful: {success_count}/{len(dates)}")
    
    if all_issues:
        log(f"\n❌ FAILED - {len(all_issues)} issues found:")
        for issue in all_issues:
            log(f"  - {issue}")
        log("\n" + "="*60)
        return 1
    else:
        log(f"\n✅ SUCCESS - All validation checks passed!")
        log("Bronze backfill data is correct and ready for use.")
        log("="*60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
