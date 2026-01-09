from __future__ import annotations

import pandas as pd

from ...base import Stage, StageIO


class SilverAddSessionLevels(Stage):
    """Silver stage: add Pre-market and Opening Range high/low levels.

    - Input:  silver.future.market_by_price_10
    - Output: silver.future.market_by_price_10_session_levels

    Adds 4 columns (scalar values broadcast to all rows):
    - pm_high: Pre-market high (05:00-09:30 EST)
    - pm_low: Pre-market low (05:00-09:30 EST)
    - or_high: Opening Range high (09:30-09:45 EST)
    - or_low: Opening Range low (09:30-09:45 EST)
    """

    def __init__(self) -> None:
        super().__init__(
            name="silver_add_session_levels",
            io=StageIO(
                inputs=["silver.future.market_by_price_10"],
                output="silver.future.market_by_price_10_session_levels",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        ts_ny = pd.to_datetime(df["ts_event_est"])
        time_only = ts_ny.dt.time

        pm_start = pd.Timestamp("05:00:00").time()
        pm_end = pd.Timestamp("09:30:00").time()
        or_start = pd.Timestamp("09:30:00").time()
        or_end = pd.Timestamp("09:45:00").time()

        pm_mask = (time_only >= pm_start) & (time_only < pm_end)
        or_mask = (time_only >= or_start) & (time_only < or_end)

        mid_px = (df["bid_px_00"] + df["ask_px_00"]) / 2

        pm_high = mid_px.loc[pm_mask].max() if pm_mask.any() else float("nan")
        pm_low = mid_px.loc[pm_mask].min() if pm_mask.any() else float("nan")
        or_high = mid_px.loc[or_mask].max() if or_mask.any() else float("nan")
        or_low = mid_px.loc[or_mask].min() if or_mask.any() else float("nan")

        df = df.copy()
        df["pm_high"] = pm_high
        df["pm_low"] = pm_low
        df["or_high"] = or_high
        df["or_low"] = or_low

        return df

