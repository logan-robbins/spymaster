from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd

from ..utils import expand_date_range

REQUIRED_COLUMNS = [
    "session_date",
    "selected_symbol",
    "selected_exp_date",
    "prev_trading_date",
    "prev_volume_selected",
    "prev_volume_rank2",
    "roll_flag",
]

SYMBOL_RE = re.compile(r"^ES[HMUZ]\d{1,2}$")
MONTH_MAP = {
    "H": 3,
    "M": 6,
    "U": 9,
    "Z": 12,
}


@dataclass
class SelectionRow:
    session_date: str
    selected_symbol: str
    selected_exp_date: str
    prev_trading_date: str
    prev_volume_selected: int
    prev_volume_rank2: int
    roll_flag: int


def _discover_symbols(bronze_root: Path) -> List[str]:
    base = bronze_root / "source=databento" / "product_type=future_mbo"
    if not base.exists():
        raise FileNotFoundError(f"Missing bronze root: {base}")
    symbols: List[str] = []
    for path in base.glob("symbol=*"):
        symbol = path.name.split("symbol=")[-1]
        if "-" in symbol:
            continue
        if SYMBOL_RE.match(symbol):
            symbols.append(symbol)
    return sorted(set(symbols))


def _parse_contract_symbol(symbol: str) -> Tuple[int, int]:
    match = SYMBOL_RE.match(symbol)
    if not match:
        raise ValueError(f"Invalid contract symbol: {symbol}")
    month_code = symbol[2]
    year_part = symbol[3:]
    month = MONTH_MAP[month_code]
    if len(year_part) == 1:
        year = 2020 + int(year_part)
    else:
        year = 2000 + int(year_part)
    return year, month


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    offset = (4 - d.weekday() + 7) % 7
    first_friday = d.toordinal() + offset
    third_friday = date.fromordinal(first_friday + 14)
    return third_friday


def _build_expirations(symbols: List[str]) -> Dict[str, date]:
    expirations: Dict[str, date] = {}
    for symbol in symbols:
        year, month = _parse_contract_symbol(symbol)
        expirations[symbol] = _third_friday(year, month)
    return expirations


def _premarket_window_ns(session_date: str) -> tuple[int, int]:
    start_local = pd.Timestamp(f"{session_date} 05:00:00", tz="America/New_York")
    end_local = pd.Timestamp(f"{session_date} 08:30:00", tz="America/New_York")
    return int(start_local.tz_convert("UTC").value), int(end_local.tz_convert("UTC").value)


def _premarket_trade_dates(
    conn: duckdb.DuckDBPyConnection,
    bronze_root: Path,
    symbols: List[str],
    dates: List[str],
) -> List[str]:
    conn.register("eligible_symbols", pd.DataFrame({"symbol": symbols}))
    valid: List[str] = []
    for session_date in dates:
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
            continue
        start_ns, end_ns = _premarket_window_ns(session_date)
        query = f"""
            SELECT COUNT(*) AS trade_count
            FROM read_parquet('{pattern.as_posix()}', hive_partitioning=true)
            WHERE action = 'T'
              AND ts_event >= {start_ns}
              AND ts_event < {end_ns}
              AND symbol IN (SELECT symbol FROM eligible_symbols)
        """
        count = int(conn.execute(query).fetchone()[0])
        if count > 0:
            valid.append(session_date)
    return valid


def _load_daily_volumes(
    conn: duckdb.DuckDBPyConnection,
    bronze_root: Path,
    symbols: List[str],
) -> pd.DataFrame:
    pattern = (
        bronze_root
        / "source=databento"
        / "product_type=future_mbo"
        / "symbol=*"
        / "table=mbo"
        / "dt=*"
        / "*.parquet"
    )
    files = glob.glob(pattern.as_posix())
    if not files:
        return pd.DataFrame(columns=["session_date", "symbol", "daily_volume"])

    conn.register("eligible_symbols", pd.DataFrame({"symbol": symbols}))
    query = f"""
        SELECT
            dt AS session_date,
            symbol,
            SUM(size) AS daily_volume
        FROM read_parquet('{pattern.as_posix()}', hive_partitioning=true)
        WHERE action = 'T'
          AND symbol IN (SELECT symbol FROM eligible_symbols)
        GROUP BY dt, symbol
    """
    return conn.execute(query).fetchdf()


def _build_trading_dates(volumes: pd.DataFrame) -> List[str]:
    totals = volumes.groupby("session_date", as_index=False)["daily_volume"].sum()
    trading = totals.loc[totals["daily_volume"] > 0, "session_date"].astype(str).tolist()
    return sorted(trading)


def _select_symbol_for_date(
    session_date: str,
    prev_date: str,
    expirations: Dict[str, date],
    vol_map: Dict[Tuple[str, str], int],
) -> Tuple[str, int, int]:
    session_dt = date.fromisoformat(session_date)
    candidates = [s for s, exp_date in expirations.items() if session_dt < exp_date]
    if not candidates:
        raise ValueError(f"No eligible contracts for session_date {session_date}")

    volumes = [int(vol_map.get((prev_date, symbol), 0)) for symbol in candidates]
    max_volume = max(volumes)
    top_symbols = [s for s, v in zip(candidates, volumes) if v == max_volume]
    if len(top_symbols) > 1:
        min_exp = min(expirations[s] for s in top_symbols)
        top_symbols = [s for s in top_symbols if expirations[s] == min_exp]
    selected_symbol = sorted(top_symbols)[0]

    sorted_vols = sorted(volumes, reverse=True)
    rank2 = int(sorted_vols[1]) if len(sorted_vols) > 1 else 0
    prev_selected = int(vol_map.get((prev_date, selected_symbol), 0))
    return selected_symbol, prev_selected, rank2


def build_selection(
    dates: List[str],
    bronze_root: Path,
) -> pd.DataFrame:
    if not dates:
        raise ValueError("No dates provided")

    symbols = _discover_symbols(bronze_root)
    if not symbols:
        raise ValueError("No eligible ES symbols found in bronze data")

    expirations = _build_expirations(symbols)
    conn = duckdb.connect(":memory:")
    volumes = _load_daily_volumes(conn, bronze_root, symbols)
    if volumes.empty:
        raise ValueError("No trade volume found in bronze data")

    volumes["session_date"] = volumes["session_date"].astype(str)
    volumes["symbol"] = volumes["symbol"].astype(str)
    volumes["daily_volume"] = volumes["daily_volume"].astype("int64")

    trading_dates = _build_trading_dates(volumes)
    if not trading_dates:
        raise ValueError("No trading dates found in bronze data")

    premarket_dates = set(_premarket_trade_dates(conn, bronze_root, symbols, trading_dates))
    trading_dates = [d for d in trading_dates if d in premarket_dates]
    if not trading_dates:
        raise ValueError("No trading dates with premarket trades")

    start_date = min(dates)
    end_date = max(dates)
    selection_dates = [d for d in trading_dates if start_date <= d <= end_date]
    if not selection_dates:
        raise ValueError("No trading dates within requested range")

    date_index = {d: idx for idx, d in enumerate(trading_dates)}
    start_idx = date_index[selection_dates[0]]
    end_idx = date_index[selection_dates[-1]]

    vol_map = {
        (str(row.session_date), str(row.symbol)): int(row.daily_volume)
        for row in volumes.itertuples(index=False)
    }

    selection_start_idx = start_idx - 1 if start_idx > 0 else start_idx
    selected_by_date: Dict[str, str] = {}
    prev_vol_by_date: Dict[str, int] = {}
    rank2_by_date: Dict[str, int] = {}

    for idx in range(selection_start_idx, end_idx + 1):
        session_date = trading_dates[idx]
        prev_date = trading_dates[idx - 1] if idx > 0 else ""
        selected_symbol, prev_volume_selected, prev_volume_rank2 = _select_symbol_for_date(
            session_date,
            prev_date,
            expirations,
            vol_map,
        )
        selected_by_date[session_date] = selected_symbol
        prev_vol_by_date[session_date] = prev_volume_selected
        rank2_by_date[session_date] = prev_volume_rank2

    rows: List[SelectionRow] = []
    for session_date in selection_dates:
        idx = date_index[session_date]
        prev_date = trading_dates[idx - 1] if idx > 0 else ""
        selected_symbol = selected_by_date[session_date]
        prev_selected_symbol = selected_by_date.get(prev_date, "")
        roll_flag = 1 if prev_selected_symbol and selected_symbol != prev_selected_symbol else 0
        rows.append(
            SelectionRow(
                session_date=session_date,
                selected_symbol=selected_symbol,
                selected_exp_date=expirations[selected_symbol].isoformat(),
                prev_trading_date=prev_date,
                prev_volume_selected=prev_vol_by_date[session_date],
                prev_volume_rank2=rank2_by_date[session_date],
                roll_flag=roll_flag,
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
