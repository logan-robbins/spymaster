from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)


class SilverComputeStatisticsClean(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_statistics_clean",
            io=StageIO(
                inputs=["bronze.future_option.statistics"],
                output="silver.future_option.statistics_clean",
            ),
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("ts_event_ns").copy()
        df_clean = df.groupby("option_symbol", as_index=False).last()

        ts_event_est = (
            pd.to_datetime(df_clean["ts_event_ns"], utc=True)
            .dt.tz_convert("America/New_York")
            .astype(str)
        )

        return pd.DataFrame(
            {
                "ts_event_ns": df_clean["ts_event_ns"].astype("int64"),
                "ts_event_est": ts_event_est,
                "ts_recv_ns": df_clean["ts_recv_ns"].astype("int64"),
                "source": df_clean["source"].astype(str),
                "underlying": df_clean["underlying"].astype(str),
                "option_symbol": df_clean["option_symbol"].astype(str),
                "exp_date": df_clean["exp_date"].astype(object),
                "strike": df_clean["strike"].astype(float),
                "right": df_clean["right"].astype(str),
                "open_interest": df_clean["open_interest"].astype(float),
            }
        )
