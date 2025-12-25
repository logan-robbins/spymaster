from __future__ import annotations

from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.config import CONFIG


def _session_bounds(date_str: str, tz: ZoneInfo) -> Tuple[int, int]:
    try:
        date = pd.Timestamp(date_str)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid date value: {date_str}") from exc

    if date.tzinfo is None:
        date = date.tz_localize(tz)
    else:
        date = date.tz_convert(tz)

    start = date + pd.Timedelta(hours=9, minutes=30)
    end = date + pd.Timedelta(hours=16)
    return int(start.tz_convert("UTC").value), int(end.tz_convert("UTC").value)


def filter_rth_signals(
    df: pd.DataFrame,
    date_col: str = "date",
    ts_col: str = "ts_ns",
) -> pd.DataFrame:
    """
    Filter signals to the regular trading session (09:30-16:00 ET) with full forward window.
    """
    if df.empty:
        return df

    missing_cols = [col for col in (date_col, ts_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Signals dataset missing columns for RTH filter: {missing_cols}")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("Signals dataset has missing or invalid dates; cannot apply RTH filter.")

    ts_series = pd.to_numeric(df[ts_col], errors="coerce")
    if ts_series.isna().any():
        raise ValueError("Signals dataset has non-numeric timestamps; cannot apply RTH filter.")

    date_str = dates.dt.strftime("%Y-%m-%d")
    unique_dates = sorted(set(date_str))
    tz = ZoneInfo("America/New_York")
    bounds: Dict[str, Tuple[int, int]] = {d: _session_bounds(d, tz) for d in unique_dates}

    session_start_ns = date_str.map(lambda d: bounds[d][0]).astype("int64")
    session_end_ns = date_str.map(lambda d: bounds[d][1]).astype("int64")

    ts_ns = ts_series.astype("int64")
    max_confirm = max(CONFIG.CONFIRMATION_WINDOWS_MULTI or [CONFIG.CONFIRMATION_WINDOW_SECONDS])
    max_window_ns = int((max_confirm + CONFIG.LOOKFORWARD_MINUTES * 60) * 1e9)
    latest_end_ns = ts_ns + max_window_ns

    mask = (
        (ts_ns >= session_start_ns)
        & (ts_ns <= session_end_ns)
        & (latest_end_ns <= session_end_ns)
    )
    return df.loc[mask].copy()
