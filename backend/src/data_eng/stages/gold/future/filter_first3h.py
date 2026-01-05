from __future__ import annotations

import pandas as pd

from ...base import Stage, StageIO


class GoldFilterFirst3Hours(Stage):
    """Gold stage: keep only the first 3 hours of RTH (09:30–12:30 NY).

    - Input:  silver.future.market_by_price_10_with_levels
    - Output: gold.future.market_by_price_10_first3h

    Filtering is based on `ts_event_est` (America/New_York).
    """

    def __init__(self) -> None:
        super().__init__(
            name="gold_filter_first3hours",
            io=StageIO(
                inputs=["silver.future.market_by_price_10_with_levels"],
                output="gold.future.market_by_price_10_first3h",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        ts = pd.to_datetime(df["ts_event_est"], utc=False)
        
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")
        
        partition_date = pd.Timestamp(dt, tz="America/New_York")
        start = partition_date + pd.Timedelta(hours=9, minutes=30)
        end = start + pd.Timedelta(hours=3)
        
        mask = (ts >= start) & (ts < end)
        return df.loc[mask].copy()
