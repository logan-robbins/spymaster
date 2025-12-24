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
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

from src.ingestor.dbn_ingestor import DBNIngestor
from src.lake.bronze_writer import BronzeWriter, dataclass_to_dict, _flatten_mbp10_event


def _resolve_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "lake"


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


def _iter_trades(dbn: DBNIngestor, date: str, start_ns: Optional[int], end_ns: Optional[int]):
    yield from dbn.read_trades(date=date, start_ns=start_ns, end_ns=end_ns)


def _iter_mbp10(dbn: DBNIngestor, date: str, start_ns: Optional[int], end_ns: Optional[int]):
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
    dbn: DBNIngestor,
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
    dbn: DBNIngestor,
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
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for Parquet writes")
    parser.add_argument("--verbose", action="store_true", help="Log progress during backfill")
    parser.add_argument("--log-every", type=int, default=250000, help="Log progress every N rows")
    parser.add_argument("--start-ns", type=int, default=None, help="Start timestamp (ns)")
    parser.add_argument("--end-ns", type=int, default=None, help="End timestamp (ns)")
    parser.add_argument("--force", action="store_true", help="Allow writing even if Bronze already exists")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be backfilled")

    args = parser.parse_args()

    if args.skip_trades and args.skip_mbp10:
        raise ValueError("Cannot skip both trades and mbp10")

    dbn = DBNIngestor()
    data_root = _resolve_data_root()
    trades_dates = set(dbn.get_available_dates("trades"))
    mbp10_dates = set(dbn.get_available_dates("MBP-10"))

    available = sorted(trades_dates | mbp10_dates)
    dates = _parse_dates(args, available)

    for date in dates:
        if not args.skip_trades and date not in trades_dates:
            raise ValueError(f"Trades DBN missing for {date}")
        if not args.skip_mbp10 and date not in mbp10_dates:
            raise ValueError(f"MBP-10 DBN missing for {date}")

        if not args.force:
            if not args.skip_trades and _check_existing(data_root, "futures/trades", args.symbol, date):
                raise ValueError(f"Bronze futures trades already exist for {date} (use --force to override)")
            if not args.skip_mbp10 and _check_existing(data_root, "futures/mbp10", args.symbol, date):
                raise ValueError(f"Bronze futures MBP-10 already exist for {date} (use --force to override)")

        print(f"\nBackfill {date} (symbol={args.symbol})")
        if args.dry_run:
            print("  dry-run: skipping writes")
            continue

        writer = BronzeWriter(data_root=str(data_root), use_s3=False)

        if not args.skip_trades:
            trades_count = backfill_trades(
                dbn=dbn,
                writer=writer,
                date=date,
                symbol_partition=args.symbol,
                batch_size=args.batch_size,
                start_ns=args.start_ns,
                end_ns=args.end_ns,
                verbose=args.verbose,
                log_every=args.log_every
            )
            if trades_count == 0:
                raise ValueError(f"No trades found for {date}")
            print(f"  Trades: {trades_count:,}")

        if not args.skip_mbp10:
            mbp10_count = backfill_mbp10(
                dbn=dbn,
                writer=writer,
                date=date,
                symbol_partition=args.symbol,
                batch_size=max(10000, args.batch_size // 5),
                start_ns=args.start_ns,
                end_ns=args.end_ns,
                verbose=args.verbose,
                log_every=args.log_every
            )
            if mbp10_count == 0:
                raise ValueError(f"No MBP-10 found for {date}")
            print(f"  MBP-10: {mbp10_count:,}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
