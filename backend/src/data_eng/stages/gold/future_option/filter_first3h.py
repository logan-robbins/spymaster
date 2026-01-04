from __future__ import annotations

import pandas as pd

from ...base import Stage, StageIO


class GoldFilterFirst3Hours(Stage):
    """Gold stage: keep only the first 3 hours of RTH (09:30–12:30 NY).

    - Input:  silver.future_option.{trades,nbbo,statistics}_clean
    - Output: gold.future_option.{trades,nbbo,statistics}_first3h

    Filtering is based on `ts_event_est` (America/New_York).
    """

    def __init__(self) -> None:
        super().__init__(
            name="gold_filter_first3hours_trades",
            io=StageIO(
                inputs=["silver.future_option.trades_clean"],
                output="gold.future_option.trades_first3h",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        ts = pd.to_datetime(df["ts_event_est"], utc=False)
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")

        date = ts.dt.normalize().iloc[0]
        start = date + pd.Timedelta(hours=9, minutes=30)
        end = start + pd.Timedelta(hours=3)

        mask = (ts >= start) & (ts < end)
        return df.loc[mask].copy()

