from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from ..utils import expand_date_range

REQUIRED_COLUMNS = [
    "session_date",
    "dominant_symbol",
    "dominance_ratio",
    "total_trade_count",
    "trade_count_dominant",
    "run_id",
    "run_pos",
    "run_len",
    "median_trade_count_run",
    "trade_count_ratio",
    "include_flag",
    "selected_symbol",
]


@dataclass
class SessionSelection:
    session_date: str
    dominant_symbol: str
    dominance_ratio: float
    total_trade_count: int
    trade_count_dominant: int
    run_id: int
    run_pos: int
    run_len: int
    median_trade_count_run: float
    trade_count_ratio: float
    include_flag: int
    selected_symbol: str


def _rth_window_ns(session_date: str) -> tuple[int, int]:
    tz = ZoneInfo("America/New_York")
    date_obj = datetime.strptime(session_date, "%Y-%m-%d")
    start_local = datetime(date_obj.year, date_obj.month, date_obj.day, 9, 30, tzinfo=tz)
    end_local = start_local + timedelta(hours=3)
    start_ns = int(start_local.astimezone(timezone.utc).timestamp() * 1e9)
    end_ns = int(end_local.astimezone(timezone.utc).timestamp() * 1e9)
    return start_ns, end_ns


def _load_trade_counts(
    conn: duckdb.DuckDBPyConnection,
    bronze_root: Path,
    session_date: str,
) -> pd.DataFrame:
    pattern = (
        bronze_root
        / "source=databento"
        / "product_type=future_mbo"
        / "symbol=*"
        / "table=mbo"
        / f"dt={session_date}"
        / "*.parquet"
    )
    files = glob.glob(pattern.as_posix())
    if not files:
        return pd.DataFrame(columns=["symbol", "trade_count"])

    start_ns, end_ns = _rth_window_ns(session_date)
    query = f"""
        SELECT
            symbol,
            COUNT(*) AS trade_count
        FROM read_parquet('{pattern.as_posix()}', hive_partitioning=true)
        WHERE action = 'T'
          AND ts_event >= {start_ns}
          AND ts_event < {end_ns}
        GROUP BY symbol
        ORDER BY trade_count DESC
    """
    return conn.execute(query).fetchdf()


def build_selection(
    dates: List[str],
    bronze_root: Path,
    dominance_threshold: float = 0.80,
    trim_count: int = 2,
    liquidity_ratio: float = 0.50,
) -> pd.DataFrame:
    conn = duckdb.connect(":memory:")
    entries: List[Dict[str, object]] = []

    for session_date in dates:
        df_counts = _load_trade_counts(conn, bronze_root, session_date)
        if df_counts.empty:
            entries.append(
                {
                    "session_date": session_date,
                    "dominant_symbol": "",
                    "dominance_ratio": 0.0,
                    "total_trade_count": 0,
                    "trade_count_dominant": 0,
                    "include_by_dominance": 0,
                    "has_data": 0,
                }
            )
            continue

        total_trade_count = int(df_counts["trade_count"].sum())
        dominant_symbol = str(df_counts.iloc[0]["symbol"])
        trade_count_dominant = int(df_counts.iloc[0]["trade_count"])
        dominance_ratio = trade_count_dominant / max(total_trade_count, 1)
        include_by_dominance = 1 if dominance_ratio >= dominance_threshold else 0
        entries.append(
            {
                "session_date": session_date,
                "dominant_symbol": dominant_symbol,
                "dominance_ratio": float(dominance_ratio),
                "total_trade_count": total_trade_count,
                "trade_count_dominant": trade_count_dominant,
                "include_by_dominance": include_by_dominance,
                "has_data": 1,
            }
        )

    run_id = -1
    current_symbol = ""
    run_indices: List[int] = []

    def finalize_run(indices: List[int]) -> None:
        nonlocal run_id
        if not indices:
            return
        run_id += 1
        run_len = len(indices)
        counts = [entries[i]["trade_count_dominant"] for i in indices]
        median_count = float(np.median(counts)) if counts else 0.0
        for pos, idx in enumerate(indices):
            entries[idx]["run_id"] = run_id
            entries[idx]["run_pos"] = pos
            entries[idx]["run_len"] = run_len
            entries[idx]["median_trade_count_run"] = median_count
            ratio = entries[idx]["trade_count_dominant"] / max(median_count, 1.0)
            entries[idx]["trade_count_ratio"] = float(ratio)

    for idx, entry in enumerate(entries):
        if entry["has_data"] == 0 or entry["include_by_dominance"] == 0:
            finalize_run(run_indices)
            run_indices = []
            current_symbol = ""
            continue
        dominant_symbol = str(entry["dominant_symbol"])
        if current_symbol == "":
            current_symbol = dominant_symbol
            run_indices = [idx]
            continue
        if dominant_symbol != current_symbol:
            finalize_run(run_indices)
            current_symbol = dominant_symbol
            run_indices = [idx]
            continue
        run_indices.append(idx)

    finalize_run(run_indices)

    rows: List[SessionSelection] = []
    for entry in entries:
        run_len = int(entry.get("run_len", 0))
        run_pos = int(entry.get("run_pos", -1))
        include_by_dominance = int(entry.get("include_by_dominance", 0))
        include_by_run = 0
        if run_len > trim_count * 2:
            include_by_run = 1 if trim_count <= run_pos <= (run_len - trim_count - 1) else 0
        median_count = float(entry.get("median_trade_count_run", 0.0))
        trade_count_ratio = float(entry.get("trade_count_ratio", 0.0))
        include_by_liquidity = 1 if trade_count_ratio >= liquidity_ratio else 0
        include_flag = 1 if include_by_dominance and include_by_run and include_by_liquidity else 0
        dominant_symbol = str(entry.get("dominant_symbol", ""))
        selected_symbol = dominant_symbol if include_flag == 1 else ""
        rows.append(
            SessionSelection(
                session_date=str(entry["session_date"]),
                dominant_symbol=dominant_symbol,
                dominance_ratio=float(entry.get("dominance_ratio", 0.0)),
                total_trade_count=int(entry.get("total_trade_count", 0)),
                trade_count_dominant=int(entry.get("trade_count_dominant", 0)),
                run_id=int(entry.get("run_id", -1)),
                run_pos=run_pos,
                run_len=run_len,
                median_trade_count_run=median_count,
                trade_count_ratio=trade_count_ratio,
                include_flag=include_flag,
                selected_symbol=selected_symbol,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df.loc[:, REQUIRED_COLUMNS]


def load_selection(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Selection map not found: {path}")
    df = pd.read_parquet(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Selection map missing columns: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MBO contract-day selection map.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--dates", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    dates = expand_date_range(dates=args.dates)
    if not dates:
        raise ValueError("No dates provided")

    bronze_root = args.repo_root / "lake" / "bronze"
    df = build_selection(dates=dates, bronze_root=bronze_root)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    main()
