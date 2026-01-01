"""
Remove exact duplicate rows from Bronze futures trades files in-place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_DATES = [
    "2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05",
    "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12",
    "2025-12-14", "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19",
]


def _dedup_file(parquet_file: Path) -> tuple[int, int]:
    df = pd.read_parquet(parquet_file)
    before = len(df)
    if before == 0:
        return 0, 0

    df = df.drop_duplicates(keep="first")
    after = len(df)

    if after == before:
        return before, after

    tmp_path = parquet_file.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False, compression="zstd", compression_level=3)
    tmp_path.replace(parquet_file)
    return before, after


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate Bronze futures trades files")
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
    total_removed = 0

    for date in dates:
        trades_dir = bronze_root / f"futures/trades/symbol=ES/date={date}"
        files = sorted(trades_dir.rglob("*.parquet"))
        if not files:
            raise AssertionError(f"No trades files for {date}")

        for parquet_file in files:
            before, after = _dedup_file(parquet_file)
            removed = before - after
            if removed > 0:
                total_removed += removed
                print(f"{parquet_file.name}: removed {removed} duplicates")

    print(f"\nTotal duplicates removed: {total_removed:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
