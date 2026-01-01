"""
Backfill Bronze futures trades + MBP-10 from DBN files.

Usage:
    cd backend/
    uv run python -m scripts.backfill_bronze_futures --date 2025-12-16
    uv run python -m scripts.backfill_bronze_futures --dates 2025-12-16,2025-12-17
    uv run python -m scripts.backfill_bronze_futures --all
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

from src.ingestion.databento.dbn_reader import DBNReader
from src.io.bronze import BronzeWriter, dataclass_to_dict, _flatten_mbp10_event


def _resolve_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _parse_dates(args: argparse.Namespace, available_dates: List[str]) -> List[str]:
    if args.all:
        return available_dates
    if args.date:
        return [args.date]
    if args.dates:
        return [d.strip() for d in args.dates.split(",") if d.strip()]
    raise ValueError("Provide --date, --dates, or --all")


def _bronze_date_path(data_root: Path, schema_path: str, symbol: str, date: str) -> Path:
    root = data_root / "bronze"
    return root / schema_path / f"symbol={symbol}" / f"date={date}"


def _check_existing(data_root: Path, schema_path: str, symbol: str, date: str) -> bool:
    date_path = _bronze_date_path(data_root, schema_path, symbol, date)
    if not date_path.exists():
        return False
    return any(date_path.rglob("*.parquet"))


def _iter_trades(dbn: DBNReader, date: str, start_ns: Optional[int], end_ns: Optional[int]):
    yield from dbn.read_trades(date=date, start_ns=start_ns, end_ns=end_ns)


def _iter_mbp10(dbn: DBNReader, date: str, start_ns: Optional[int], end_ns: Optional[int]):
    yield from dbn.read_mbp10(date=date, start_ns=start_ns, end_ns=end_ns)


def _flush_batches(
    writer: BronzeWriter,
    schema_name: str,
    records: List[dict]
) -> None:
    if records:
        writer._write_parquet(schema_name, records)
        records.clear()


def backfill_trades(
    dbn: DBNReader,
    writer: BronzeWriter,
    date: str,
    symbol_partition: str,
    batch_size: int,
    start_ns: Optional[int],
    end_ns: Optional[int],
    verbose: bool,
    log_every: int
) -> int:
    count = 0
    start_time = time.time()
    batch: List[dict] = []
    for trade in _iter_trades(dbn, date, start_ns, end_ns):
        record = dataclass_to_dict(trade)
        record["_partition_key"] = symbol_partition
        batch.append(record)
        count += 1
        if verbose and log_every > 0 and count % log_every == 0:
            elapsed = time.time() - start_time
            print(f"  Trades progress: {count:,} rows in {elapsed:.1f}s")
        if len(batch) >= batch_size:
            _flush_batches(writer, "futures.trades", batch)
    _flush_batches(writer, "futures.trades", batch)
    return count


def backfill_mbp10(
    dbn: DBNReader,
    writer: BronzeWriter,
    date: str,
    symbol_partition: str,
    batch_size: int,
    start_ns: Optional[int],
    end_ns: Optional[int],
    verbose: bool,
    log_every: int
) -> int:
    count = 0
    start_time = time.time()
    batch: List[dict] = []
    for mbp10 in _iter_mbp10(dbn, date, start_ns, end_ns):
        record = _flatten_mbp10_event(mbp10)
        record["_partition_key"] = symbol_partition
        batch.append(record)
        count += 1
        if verbose and log_every > 0 and count % log_every == 0:
            elapsed = time.time() - start_time
            print(f"  MBP-10 progress: {count:,} rows in {elapsed:.1f}s")
        if len(batch) >= batch_size:
            _flush_batches(writer, "futures.mbp10", batch)
    _flush_batches(writer, "futures.mbp10", batch)
    return count


def process_single_date(args_tuple):
    """
    Process a single date (for parallel execution).
    
    Args:
        args_tuple: Tuple of (date, data_root, symbol, batch_size, mbp10_batch_size, 
                             skip_trades, skip_mbp10, force, verbose, start_ns, end_ns, log_every)
    
    Returns:
        Dict with status
    """
    (date, data_root, symbol, batch_size, mbp10_batch_size, 
     skip_trades, skip_mbp10, force, verbose, start_ns, end_ns, log_every) = args_tuple
    
    try:
        dbn = DBNReader()
        
        # Check if data exists
        if not force:
            if not skip_trades and _check_existing(data_root, "futures/trades", symbol, date):
                return {'date': date, 'status': 'skipped', 'reason': 'exists'}
            if not skip_mbp10 and _check_existing(data_root, "futures/mbp10", symbol, date):
                return {'date': date, 'status': 'skipped', 'reason': 'exists'}
        
        start_time = time.time()
        writer = BronzeWriter(data_root=str(data_root), use_s3=False)
        
        trades_count = 0
        mbp10_count = 0
        
        if not skip_trades:
            trades_count = backfill_trades(
                dbn=dbn,
                writer=writer,
                date=date,
                symbol_partition=symbol,
                batch_size=batch_size,
                start_ns=start_ns,
                end_ns=end_ns,
                verbose=False,  # Disable verbose in parallel to avoid output collision
                log_every=0
            )
        
        if not skip_mbp10:
            mbp10_count = backfill_mbp10(
                dbn=dbn,
                writer=writer,
                date=date,
                symbol_partition=symbol,
                batch_size=mbp10_batch_size,
                start_ns=start_ns,
                end_ns=end_ns,
                verbose=False,
                log_every=0
            )
        
        elapsed = time.time() - start_time
        return {
            'date': date,
            'status': 'success',
            'trades': trades_count,
            'mbp10': mbp10_count,
            'elapsed': elapsed
        }
    
    except Exception as e:
        return {
            'date': date,
            'status': 'error',
            'error': str(e)
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill Bronze futures trades + MBP-10 from DBN files"
    )
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--dates", type=str, help="Comma-separated dates")
    parser.add_argument("--all", action="store_true", help="Backfill all DBN dates")
    parser.add_argument("--symbol", type=str, default="ES", help="Partition symbol (default: ES)")
    parser.add_argument("--skip-trades", action="store_true", help="Skip futures trades backfill")
    parser.add_argument("--skip-mbp10", action="store_true", help="Skip MBP-10 backfill")
    parser.add_argument("--batch-size", type=int, default=7_500_000, help="Batch size for Parquet writes (trades, default: 7.5M)")
    parser.add_argument("--mbp10-batch-size", type=int, default=None, help="Batch size for MBP-10 (default: 2x trades batch)")
    parser.add_argument("--verbose", action="store_true", help="Log progress during backfill")
    parser.add_argument("--log-every", type=int, default=1000000, help="Log progress every N rows")
    parser.add_argument("--start-ns", type=int, default=None, help="Start timestamp (ns)")
    parser.add_argument("--end-ns", type=int, default=None, help="End timestamp (ns)")
    parser.add_argument("--force", action="store_true", help="Allow writing even if Bronze already exists")
    parser.add_argument("--clean", action="store_true", help="Delete existing Bronze data before backfill")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be backfilled")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")

    args = parser.parse_args()

    if args.skip_trades and args.skip_mbp10:
        raise ValueError("Cannot skip both trades and mbp10")

    dbn = DBNReader()
    data_root = _resolve_data_root()
    trades_dates = set(dbn.get_available_dates("trades"))
    mbp10_dates = set(dbn.get_available_dates("MBP-10"))

    available = sorted(trades_dates | mbp10_dates)
    dates = _parse_dates(args, available)

    # Validate dates
    for date in dates:
        if not args.skip_trades and date not in trades_dates:
            raise ValueError(f"Trades DBN missing for {date}")
        if not args.skip_mbp10 and date not in mbp10_dates:
            raise ValueError(f"MBP-10 DBN missing for {date}")
    
    # Clean existing data if requested
    if args.clean:
        for date in dates:
            if not args.skip_trades:
                trades_path = _bronze_date_path(data_root, "futures/trades", args.symbol, date)
                if trades_path.exists():
                    print(f"  Cleaning existing trades: {trades_path}")
                    shutil.rmtree(trades_path)
            if not args.skip_mbp10:
                mbp10_path = _bronze_date_path(data_root, "futures/mbp10", args.symbol, date)
                if mbp10_path.exists():
                    print(f"  Cleaning existing MBP-10: {mbp10_path}")
                    shutil.rmtree(mbp10_path)
    
    if args.dry_run:
        print(f"Would backfill {len(dates)} dates")
        return 0
    
    print(f"\nBackfilling {len(dates)} dates with {args.workers} workers")
    print(f"  Symbol: {args.symbol}")
    print(f"  Trades batch size: {args.batch_size:,}")
    mbp10_batch = args.mbp10_batch_size or (args.batch_size * 2)
    print(f"  MBP-10 batch size: {mbp10_batch:,}")
    print()
    
    # Prepare arguments for parallel processing
    job_args = [
        (date, data_root, args.symbol, args.batch_size, mbp10_batch,
         args.skip_trades, args.skip_mbp10, args.force, args.verbose,
         args.start_ns, args.end_ns, args.log_every)
        for date in dates
    ]
    
    # Process dates in parallel
    start_time = time.time()
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_date = {executor.submit(process_single_date, args): args[0] for args in job_args}
        
        for future in as_completed(future_to_date):
            result = future.result()
            
            if result['status'] == 'success':
                success_count += 1
                trades_str = f"{result['trades']:,} trades" if result['trades'] > 0 else ""
                mbp10_str = f"{result['mbp10']:,} mbp10" if result['mbp10'] > 0 else ""
                parts = [p for p in [trades_str, mbp10_str] if p]
                print(f"✅ {result['date']}: {', '.join(parts)} ({result['elapsed']:.1f}s)")
            elif result['status'] == 'skipped':
                skipped_count += 1
                print(f"⏭️  {result['date']}: skipped ({result['reason']})")
            else:
                error_count += 1
                print(f"❌ {result['date']}: {result['error']}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: {elapsed/60:.1f} minutes")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
