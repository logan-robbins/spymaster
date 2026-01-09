from __future__ import annotations

import pandas as pd

from ...base import Stage, StageIO


class SilverConvertUtcToEst(Stage):
    """Silver stage: add `ts_event_est` derived from `ts_event`.

    - Input:  bronze.futures.market_by_price_10
    - Output: silver.futures.market_by_price_10

    `ts_event` is nanoseconds since epoch (UTC). We convert it to an ISO-8601
    string in America/New_York, including the timezone offset.
    """

    def __init__(self) -> None:
        super().__init__(
            name="silver_convert_utc_to_est",
            io=StageIO(
                        inputs=["bronze.future.market_by_price_10"],
                output="silver.future.market_by_price_10",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        ts = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
        ts_ny = ts.dt.tz_convert("America/New_York")

        # ISO-8601 with timezone offset -05:00 / -04:00
        est_str = ts_ny.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        # Convert -0500 -> -05:00 (insert colon for ISO compliance)
        est_str = est_str.str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)

        df = df.copy()
        df["ts_event_est"] = est_str
        return df
