"""
Session time utilities for ET-canonical timing.

All "since open" features must be relative
to 09:30 ET, NOT the first bar in a UTC-partitioned file.
"""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

from src.common.config import CONFIG


def get_session_start_ns(date: str) -> int:
    """
    Get RTH session start timestamp (09:30 ET) in nanoseconds UTC.
    
    Args:
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Timestamp in nanoseconds UTC
    """
    session_start = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(
        hours=CONFIG.RTH_START_HOUR,
        minutes=CONFIG.RTH_START_MINUTE
    )
    return session_start.tz_convert("UTC").value


def get_session_end_ns(date: str) -> int:
    """
    Get RTH session end timestamp (13:30 ET for v1) in nanoseconds UTC.
    
    Args:
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Timestamp in nanoseconds UTC
    """
    session_end = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(
        hours=CONFIG.RTH_END_HOUR,
        minutes=CONFIG.RTH_END_MINUTE
    )
    return session_end.tz_convert("UTC").value


def get_premarket_start_ns(date: str) -> int:
    """
    Get premarket start timestamp (04:00 ET) in nanoseconds UTC.
    
    Args:
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Timestamp in nanoseconds UTC
    """
    pm_start = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(
        hours=CONFIG.PREMARKET_START_HOUR,
        minutes=CONFIG.PREMARKET_START_MINUTE
    )
    return pm_start.tz_convert("UTC").value


def compute_minutes_since_open(ts_ns: np.ndarray, date: str) -> np.ndarray:
    """
    Compute minutes since RTH open (09:30 ET) for timestamps.
    
    This is the correct implementation for v1.
    Previous implementation incorrectly used "first bar in file".
    
    Args:
        ts_ns: Array of timestamps in nanoseconds UTC
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Array of minutes since open (float, can be negative for premarket)
    """
    session_start_ns = get_session_start_ns(date)
    minutes_since_open = (ts_ns - session_start_ns) / 1e9 / 60.0
    return minutes_since_open


def compute_bars_since_open(
    ts_ns: np.ndarray,
    date: str,
    bar_duration_minutes: int = 1
) -> np.ndarray:
    """
    Compute bars since RTH open (09:30 ET) for timestamps.
    
    Args:
        ts_ns: Array of timestamps in nanoseconds UTC
        date: Date string (YYYY-MM-DD)
        bar_duration_minutes: Bar size in minutes (default 1)
    
    Returns:
        Array of bars since open (int, 0 at/before open)
    """
    minutes = compute_minutes_since_open(ts_ns, date)
    bars = np.maximum(0, (minutes / bar_duration_minutes).astype(np.int32))
    session_minutes = ((CONFIG.RTH_END_HOUR * 60 + CONFIG.RTH_END_MINUTE)
                       - (CONFIG.RTH_START_HOUR * 60 + CONFIG.RTH_START_MINUTE))
    max_bars = max(0, int(session_minutes / max(1, bar_duration_minutes)))
    bars = np.minimum(bars, max_bars)
    return bars


def compute_session_phase(ts_ns: np.ndarray, date: str) -> np.ndarray:
    """
    Compute session phase bucket for timestamps.
    
    Session phases:
    - 0: premarket (< 09:30)
    - 1: first 15 minutes (09:30-09:45)
    - 2: 15-30 minutes (09:45-10:00)
    - 3: 30-60 minutes (10:00-10:30)
    - 4: 60-120 minutes (10:30-11:30)
    - 5: 120-240 minutes (11:30-13:30)
    - 6: after 13:30
    
    Args:
        ts_ns: Array of timestamps in nanoseconds UTC
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Array of phase buckets (int)
    """
    minutes = compute_minutes_since_open(ts_ns, date)
    
    phases = np.zeros(len(minutes), dtype=np.int8)
    phases[minutes < 0] = 0  # premarket
    phases[(minutes >= 0) & (minutes < 15)] = 1  # first 15 min
    phases[(minutes >= 15) & (minutes < 30)] = 2  # 15-30 min
    phases[(minutes >= 30) & (minutes < 60)] = 3  # 30-60 min
    phases[(minutes >= 60) & (minutes < 120)] = 4  # 60-120 min
    phases[(minutes >= 120) & (minutes < 240)] = 5  # 120-240 min
    phases[minutes >= 240] = 6  # after 4 hours
    
    return phases


def is_first_15_minutes(ts_ns: np.ndarray, date: str) -> np.ndarray:
    """
    Check if timestamps are in first 15 minutes of RTH.
    
    Args:
        ts_ns: Array of timestamps in nanoseconds UTC
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Boolean array
    """
    minutes = compute_minutes_since_open(ts_ns, date)
    return (minutes >= 0) & (minutes < 15)


def filter_rth_only(df: pd.DataFrame, date: str, ts_col: str = 'ts_ns') -> pd.DataFrame:
    """
    Filter DataFrame to RTH timestamps only (09:30-13:30 ET for v1).
    
    Args:
        df: DataFrame with timestamp column
        date: Date string (YYYY-MM-DD)
        ts_col: Name of timestamp column (default 'ts_ns')
    
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    session_start_ns = get_session_start_ns(date)
    session_end_ns = get_session_end_ns(date)
    
    mask = (df[ts_col] >= session_start_ns) & (df[ts_col] <= session_end_ns)
    return df[mask].copy()


def filter_premarket_only(df: pd.DataFrame, date: str, ts_col: str = 'ts_ns') -> pd.DataFrame:
    """
    Filter DataFrame to premarket timestamps only (04:00-09:30 ET).
    
    Args:
        df: DataFrame with timestamp column
        date: Date string (YYYY-MM-DD)
        ts_col: Name of timestamp column (default 'ts_ns')
    
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    pm_start_ns = get_premarket_start_ns(date)
    session_start_ns = get_session_start_ns(date)
    
    mask = (df[ts_col] >= pm_start_ns) & (df[ts_col] < session_start_ns)
    return df[mask].copy()
