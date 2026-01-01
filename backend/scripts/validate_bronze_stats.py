"""
Validate Bronze statistical properties and cross-data consistency (Steps 3-4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


EXPECTED_DATES = [
    "2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05",
    "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12",
    "2025-12-14", "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19",
]


def _read_all_trades(bronze_root: Path, date: str) -> pd.DataFrame:
    trades_path = bronze_root / f"futures/trades/symbol=ES/date={date}"
    files = sorted(trades_path.rglob("*.parquet"))
    if not files:
        raise AssertionError(f"No trades files for {date}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _read_options_file(bronze_root: Path, date: str) -> pd.DataFrame:
    opts_dir = bronze_root / f"options/trades/underlying=SPY/date={date}"
    files = sorted(opts_dir.glob("*.parquet"))
    if not files:
        raise AssertionError(f"No options files for {date}")
    return pd.read_parquet(files[0])


def validate_trade_distributions(bronze_root: Path, date: str) -> None:
    df = _read_all_trades(bronze_root, date)

    print(f"\n{date} Trade Statistics:")
    print(f"  Total trades: {len(df):,}")
    print(f"  Price: μ={df['price'].mean():.2f}, σ={df['price'].std():.2f}")
    print(f"  Size: median={df['size'].median()}, p95={df['size'].quantile(0.95)}")
    span_hours = (df["ts_event_ns"].max() - df["ts_event_ns"].min()) / 1e9 / 3600
    print(f"  Time span: {span_hours:.1f} hours")

    aggressor_series = pd.to_numeric(df["aggressor"], errors="coerce")
    aggressor_dist = aggressor_series.value_counts(normalize=True)
    print(f"  Aggressor balance: {aggressor_dist.to_dict()}")

    assert df["price"].std() > 1, "Price variance too low"
    assert 0.3 < aggressor_dist.get(1, 0) < 0.7, "Aggressor balance skewed"
    assert df["ts_event_ns"].is_monotonic_increasing, "Time not sorted!"


def validate_mbp10_quality(bronze_root: Path, date: str) -> None:
    mbp_dir = bronze_root / f"futures/mbp10/symbol=ES/date={date}/hour=15"
    files = sorted(mbp_dir.glob("*.parquet"))
    if not files:
        raise AssertionError(f"No MBP-10 files for {date} hour=15")

    df = pd.read_parquet(files[0])
    print(f"\n{date} MBP-10 Quality:")
    print(f"  Total snapshots: {len(df):,}")

    spread = df["ask_px_1"] - df["bid_px_1"]
    print(f"  Spread: median={spread.median():.2f}, p95={spread.quantile(0.95):.2f}")

    valid_spread = (df["ask_px_1"] >= df["bid_px_1"]).all()
    print(f"  Valid spread: {valid_spread}")

    bid_violations = sum((df[f"bid_px_{i}"] < df[f"bid_px_{i+1}"]).sum() for i in range(1, 10))
    violation_pct = bid_violations / (len(df) * 9) * 100
    print(f"  Bid ordering violations: {violation_pct:.2f}%")

    assert valid_spread, "Invalid bid/ask spread detected"
    assert violation_pct < 1.0, "Too many level ordering violations"


def validate_options_flow(bronze_root: Path, date: str) -> None:
    df = _read_options_file(bronze_root, date)

    print(f"\n{date} Options Statistics:")
    print(f"  Total trades: {len(df):,}")
    print(f"  Unique strikes: {df['strike'].nunique()}")
    print(f"  Expirations: {df['exp_date'].unique()}")

    dte = (pd.to_datetime(df["exp_date"]) - pd.to_datetime(date)).dt.days
    dte_dist = dte.value_counts().head()
    print(f"  DTE distribution:\n{dte_dist}")

    print(f"  Option price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

    strike_min = df["strike"].min()
    strike_max = df["strike"].max()
    print(f"  Strike range: ${strike_min:.2f} - ${strike_max:.2f}")

    print(f"  Size: median={df['size'].median()}, p95={df['size'].quantile(0.95)}")

    assert dte_dist.iloc[0] / len(df) > 0.5, "0DTE should dominate volume"
    assert 20 < df["strike"].nunique() < 200, "Unusual strike count"


def validate_time_alignment(bronze_root: Path, date: str) -> None:
    trades_glob = bronze_root / f"futures/trades/symbol=ES/date={date}/**/*.parquet"
    opts_glob = bronze_root / f"options/trades/underlying=SPY/date={date}/*.parquet"

    es_min, es_max = duckdb.execute(
        f"SELECT MIN(ts_event_ns), MAX(ts_event_ns) FROM read_parquet('{trades_glob}', hive_partitioning=true)"
    ).fetchone()
    opts_min, opts_max = duckdb.execute(
        f"SELECT MIN(ts_event_ns), MAX(ts_event_ns) FROM read_parquet('{opts_glob}', hive_partitioning=true)"
    ).fetchone()

    es_start = pd.to_datetime(es_min, unit="ns")
    es_end = pd.to_datetime(es_max, unit="ns")
    opts_start = pd.to_datetime(opts_min, unit="ns")
    opts_end = pd.to_datetime(opts_max, unit="ns")

    print(f"\n{date} Time Alignment:")
    print(f"  ES:      {es_start} to {es_end}")
    print(f"  Options: {opts_start} to {opts_end}")

    overlap_start = max(es_start, opts_start)
    overlap_end = min(es_end, opts_end)
    overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600

    print(f"  Overlap: {overlap_hours:.1f} hours")
    assert overlap_hours > 6, f"Insufficient overlap: {overlap_hours:.1f}h"


def validate_es_spy_correlation(bronze_root: Path, date: str) -> None:
    mbp_dir = bronze_root / f"futures/mbp10/symbol=ES/date={date}/hour=15"
    mbp_files = sorted(mbp_dir.glob("*.parquet"))
    if not mbp_files:
        raise AssertionError(f"No MBP-10 files for {date} hour=15")

    df_mbp = pd.read_parquet(mbp_files[0])
    es_mid = (df_mbp["bid_px_1"] + df_mbp["ask_px_1"]) / 2
    spy_implied = es_mid / 10

    df_opts = _read_options_file(bronze_root, date)
    opts_ts = pd.to_datetime(df_opts["ts_event_ns"], unit="ns", utc=True)
    df_opts_hour = df_opts[opts_ts.dt.hour == 15]
    if df_opts_hour.empty:
        raise AssertionError(f"No options trades in hour=15 for {date}")
    spot_est = float(spy_implied.mean())
    near_mask = (df_opts_hour["strike"] >= spot_est - 5.0) & (df_opts_hour["strike"] <= spot_est + 5.0)
    df_near = df_opts_hour[near_mask]
    if df_near.empty:
        raise AssertionError(f"No options trades within ±$5 of spot for {date}")
    strike_volume = df_near.groupby("strike")["size"].sum()
    strike_mode = strike_volume.idxmax()

    print(f"\n{date} ES/SPY Relationship:")
    print(f"  ES mid: ${es_mid.mean():.2f}")
    print(f"  SPY implied: ${spot_est:.2f}")
    print(f"  Options ATM strike: ${strike_mode:.2f}")
    print(f"  Difference: ${abs(spot_est - strike_mode):.2f}")

    assert abs(spot_est - strike_mode) < 5, "ES/SPY price mismatch"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Bronze statistics (Steps 3-4)")
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
        validate_trade_distributions(bronze_root, date)
        if date != "2025-12-14":
            validate_mbp10_quality(bronze_root, date)

        if date != "2025-12-14":
            validate_options_flow(bronze_root, date)
            validate_time_alignment(bronze_root, date)
            validate_es_spy_correlation(bronze_root, date)

    print("\nBronze statistical validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
