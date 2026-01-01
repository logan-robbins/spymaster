"""
Filter Bronze futures data to the dominant ES contract per date (Step 2).

This overwrites existing Bronze Parquet files in-place.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


EXPECTED_DATES = [
    "2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05",
    "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12",
    "2025-12-14", "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19",
]


def _main_symbol_for_date(bronze_root: Path, date: str) -> tuple[str, float]:
    base_dir = bronze_root / f"futures/trades/symbol=ES/date={date}"
    if not base_dir.exists():
        raise AssertionError(f"No trades directory found for {date}: {base_dir}")
    if not any(base_dir.rglob("*.parquet")):
        raise AssertionError(f"No trades parquet files found for {date}")

    pattern = base_dir / "**/*.parquet"
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    is_weekend = date_obj.weekday() >= 5

    if is_weekend:
        time_filter = ""
    else:
        start_ns = int(datetime(date_obj.year, date_obj.month, date_obj.day, 14, 30, tzinfo=timezone.utc).timestamp() * 1e9)
        end_ns = int(datetime(date_obj.year, date_obj.month, date_obj.day, 21, 0, tzinfo=timezone.utc).timestamp() * 1e9)
        time_filter = f"WHERE ts_event_ns >= {start_ns} AND ts_event_ns <= {end_ns}"

    query = f"""
        SELECT symbol, COUNT(*) AS cnt
        FROM read_parquet('{pattern}', hive_partitioning=false)
        {time_filter}
        GROUP BY symbol
        ORDER BY cnt DESC
    """
    df = duckdb.execute(query).fetchdf()
    if df.empty:
        raise AssertionError(f"No symbol data found for {date}")

    main_symbol = df.iloc[0]["symbol"]
    volume_pct = df.iloc[0]["cnt"] / df["cnt"].sum()
    print(f"{date}: Main contract = {main_symbol} ({volume_pct*100:.1f}% volume)")
    if volume_pct < 0.90:
        print(f"  WARNING: Main contract share below 90% (likely roll day).")
    assert volume_pct > 0.50, f"Main contract only {volume_pct*100:.1f}% - data issue?"
    return main_symbol, volume_pct


def _filter_files(
    bronze_root: Path,
    schema_path: str,
    date: str,
    main_symbol: str
) -> None:
    base_dir = bronze_root / f"{schema_path}/symbol=ES/date={date}"
    if not base_dir.exists():
        raise AssertionError(f"Missing Bronze path: {base_dir}")

    files = list(base_dir.rglob("*.parquet"))
    if not files:
        raise AssertionError(f"No parquet files under: {base_dir}")

    for parquet_file in files:
        df = pd.read_parquet(parquet_file)
        if "symbol" not in df.columns:
            raise AssertionError(f"Missing symbol column in {parquet_file}")

        df_clean = df[df["symbol"] == main_symbol].copy()
        removed = len(df) - len(df_clean)
        pct = (removed / len(df) * 100.0) if len(df) > 0 else 0.0
        print(f"  {parquet_file.name}: removed {removed} non-main rows ({pct:.1f}%)")

        if df_clean.empty:
            parquet_file.unlink()
            continue
        if removed == 0:
            continue

        tmp_path = parquet_file.with_suffix(".tmp.parquet")
        df_clean.to_parquet(tmp_path, index=False, compression="zstd", compression_level=3)
        tmp_path.replace(parquet_file)


def _validate_price_range(bronze_root: Path, date: str) -> None:
    trades_dir = bronze_root / f"futures/trades/symbol=ES/date={date}"
    sample_files = list(trades_dir.rglob("*.parquet"))
    if not sample_files:
        raise AssertionError(f"No trades files for price range check: {trades_dir}")

    df = pd.read_parquet(sample_files[0])
    if df.empty:
        raise AssertionError(f"Empty trades file for price range check: {sample_files[0]}")

    price_min = df["price"].min()
    price_max = df["price"].max()

    print(f"{date} price range: ${price_min:.2f} - ${price_max:.2f}")

    assert 6700 < price_min < 7000, f"Min price {price_min:.2f} out of expected range"
    assert 6750 < price_max < 7100, f"Max price {price_max:.2f} out of expected range"

    spy_min = price_min / 10
    spy_max = price_max / 10
    print(f"  SPY equivalent: ${spy_min:.2f} - ${spy_max:.2f}")
    assert 670 < spy_min < 700, "SPY equivalent out of expected range"
    assert 675 < spy_max < 710, "SPY equivalent out of expected range"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter Bronze futures data to main contract (in-place)"
    )
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

    for date in dates:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        if date_obj.weekday() >= 5:
            print(f"{date}: weekend date - skipping contract filter.")
            continue

        main_symbol, _ = _main_symbol_for_date(bronze_root, date)

        print(f"Filtering trades for {date}...")
        _filter_files(bronze_root, "futures/trades", date, main_symbol)

        print(f"Filtering MBP-10 for {date}...")
        _filter_files(bronze_root, "futures/mbp10", date, main_symbol)

        _validate_price_range(bronze_root, date)

    print("\nBronze contract filtering complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
