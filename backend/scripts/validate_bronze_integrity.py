"""
Validate Bronze data integrity per VALIDATE.md Step 1.

Checks:
1. File integrity (expected partitions exist)
2. Row count consistency (reasonable ranges)
3. Duplicate detection (sample hour=15 trades file)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


EXPECTED_DATES = [
    "2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05",
    "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12",
    "2025-12-14", "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19",
]


def _require_files(path: Path, label: str) -> None:
    if not path.exists():
        raise AssertionError(f"Missing {label}: {path}")
    if not any(path.rglob("*.parquet")):
        raise AssertionError(f"No parquet files for {label}: {path}")


def _count_rows(glob_pattern: str) -> int:
    query = f"SELECT COUNT(*) FROM read_parquet('{glob_pattern}', hive_partitioning=true)"
    return duckdb.execute(query).fetchone()[0]


def check_file_integrity(bronze_root: Path, dates: list[str]) -> None:
    for date in dates:
        trades_path = bronze_root / f"futures/trades/symbol=ES/date={date}"
        _require_files(trades_path, f"ES trades {date}")

        mbp10_path = bronze_root / f"futures/mbp10/symbol=ES/date={date}"
        _require_files(mbp10_path, f"ES MBP-10 {date}")

        if date != "2025-12-14":
            opts_path = bronze_root / f"options/trades/underlying=SPY/date={date}"
            _require_files(opts_path, f"SPY options {date}")


def validate_row_counts(bronze_root: Path, date: str) -> None:
    trades_glob = str(bronze_root / f"futures/trades/symbol=ES/date={date}/**/*.parquet")
    mbp10_glob = str(bronze_root / f"futures/mbp10/symbol=ES/date={date}/**/*.parquet")
    options_glob = str(bronze_root / f"options/trades/underlying=SPY/date={date}/*.parquet")

    trade_count = _count_rows(trades_glob)
    mbp10_count = _count_rows(mbp10_glob)
    opts_count = _count_rows(options_glob) if date != "2025-12-14" else 0

    print(f"{date}:")
    print(f"  Trades: {trade_count:,}")
    print(f"  MBP-10: {mbp10_count:,}")
    print(f"  Options: {opts_count:,}")

    if date == "2025-12-14":
        assert trade_count < 20000, "12-14 Saturday should have minimal trades"
    else:
        assert 100000 < trade_count < 1000000, f"Unusual trade count: {trade_count:,}"
        assert 2000000 < mbp10_count < 30000000, f"Unusual MBP-10 count: {mbp10_count:,}"


def check_duplicates(bronze_root: Path, date: str) -> None:
    trades_dir = bronze_root / f"futures/trades/symbol=ES/date={date}/hour=15"
    if not trades_dir.exists():
        raise AssertionError(f"Missing hour=15 trades directory: {trades_dir}")

    trades_files = list(trades_dir.glob("*.parquet"))
    if not trades_files:
        raise AssertionError(f"No parquet files in: {trades_dir}")

    df = pd.read_parquet(trades_files[0])
    if df.empty:
        raise AssertionError(f"Empty parquet file: {trades_files[0]}")

    exact_dups = df.duplicated(keep=False).sum()
    trade_dups = df.duplicated(subset=["ts_event_ns", "price", "size"], keep=False).sum()

    print(f"{date} duplicates (sample {trades_files[0].name}):")
    print(f"  Exact duplicates: {exact_dups:,} ({exact_dups/len(df)*100:.2f}%)")
    print(f"  Trade duplicates: {trade_dups:,} ({trade_dups/len(df)*100:.2f}%)")

    assert exact_dups / len(df) < 0.01, "Too many duplicates! Old data not cleaned?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Bronze integrity (Step 1)")
    parser.add_argument(
        "--dates",
        type=str,
        default=",".join(EXPECTED_DATES),
        help="Comma-separated dates (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bronze-root",
        type=str,
        default=None,
        help="Override Bronze root (defaults to backend/data/bronze)"
    )
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    bronze_root = Path(args.bronze_root) if args.bronze_root else (
        Path(__file__).resolve().parents[1] / "data" / "bronze"
    )

    print(f"Bronze root: {bronze_root}")
    check_file_integrity(bronze_root, dates)

    for date in dates:
        validate_row_counts(bronze_root, date)
        if date != "2025-12-14":
            check_duplicates(bronze_root, date)

    print("\nBronze integrity validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
