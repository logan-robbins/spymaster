from __future__ import annotations

import pandas as pd

from ...base import Stage, StageIO


class SilverFilterFirst4Hours(Stage):
    """Silver stage: keep 1 hour premarket + first 3 hours RTH (08:30–12:30 NY).

    - Input:  silver.future.market_by_price_10_session_levels (with PM_HIGH/PM_LOW computed)
    - Output: silver.future.market_by_price_10_first4h

    Filtering is based on `ts_event_est` (America/New_York).
    Window: 08:30-12:30 NY (1 hour premarket context + 3 hours RTH).
    """

    def __init__(self) -> None:
        super().__init__(
            name="silver_filter_first4hours",
            io=StageIO(
                inputs=["silver.future.market_by_price_10_session_levels"],
                output="silver.future.market_by_price_10_first4h",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        ts = pd.to_datetime(df["ts_event_est"], utc=False)

        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")

        partition_date = pd.Timestamp(dt, tz="America/New_York")
        start = partition_date + pd.Timedelta(hours=8, minutes=30)
        end = partition_date + pd.Timedelta(hours=12, minutes=30)

        mask = (ts >= start) & (ts < end)
        return df.loc[mask].copy()
