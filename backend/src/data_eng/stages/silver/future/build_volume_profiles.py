from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_partition,
    write_partition,
)

LOOKBACK_DAYS = 7
MIN_DAYS = 3
N_BUCKETS = 48
RTH_START_HOUR = 9
RTH_START_MINUTE = 30
BUCKET_MINUTES = 5
EPSILON = 1e-9

FLOW_BANDS = ["p0_1", "p1_2", "p2_3", "p3_5", "p5_10"]


def get_bucket_id(ts_ns: int) -> int:
    dt = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")
    minutes_since_open = (dt.hour - RTH_START_HOUR) * 60 + dt.minute - RTH_START_MINUTE
    bucket_id = minutes_since_open // BUCKET_MINUTES
    return max(0, min(N_BUCKETS - 1, bucket_id))


def bucket_start_time(bucket_id: int) -> str:
    total_minutes = RTH_START_HOUR * 60 + RTH_START_MINUTE + bucket_id * BUCKET_MINUTES
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}:00"


def get_prior_trading_days(for_date: str, max_lookback: int = 14) -> List[str]:
    target = datetime.strptime(for_date, "%Y-%m-%d").date()
    dates = []
    check_date = target - timedelta(days=1)
    attempts = 0
    while len(dates) < LOOKBACK_DAYS and attempts < max_lookback:
        if check_date.weekday() < 5:
            dates.append(check_date.strftime("%Y-%m-%d"))
        check_date -= timedelta(days=1)
        attempts += 1
    return dates


class SilverBuildVolumeProfiles(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_build_volume_profiles",
            io=StageIO(
                inputs=["silver.future.market_by_price_10_bar5s"],
                output="silver.future.volume_profiles",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        lookback_dates = get_prior_trading_days(dt)
        available_dates = []
        for lookback_dt in lookback_dates:
            in_ref = partition_ref(cfg, self.io.inputs[0], symbol, lookback_dt)
            if is_partition_complete(in_ref):
                available_dates.append(lookback_dt)

        if len(available_dates) < MIN_DAYS:
            df_out = self._build_fallback_profile(len(available_dates))
        else:
            dfs = []
            for lookback_dt in available_dates:
                in_ref = partition_ref(cfg, self.io.inputs[0], symbol, lookback_dt)
                df = read_partition(in_ref)
                df["_source_date"] = lookback_dt
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)
            df_out = self._build_profile(df_all, len(available_dates))

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=out_contract_path,
            inputs=[],
            stage=self.name,
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        raise NotImplementedError("Use run() directly")

    def _build_profile(self, df: pd.DataFrame, n_days: int) -> pd.DataFrame:
        df = df.copy()
        df["bucket_id"] = df["bar_ts"].apply(get_bucket_id)
        df["_date"] = df["_source_date"]

        for side in ["bid", "ask"]:
            for flow_type in ["add", "rem", "net"]:
                band_cols = [
                    f"bar5s_flow_{flow_type}_vol_{side}_{band}_sum"
                    for band in FLOW_BANDS
                    if f"bar5s_flow_{flow_type}_vol_{side}_{band}_sum" in df.columns
                ]
                if band_cols:
                    df[f"_flow_{flow_type}_{side}_total"] = df[band_cols].sum(axis=1)
                else:
                    df[f"_flow_{flow_type}_{side}_total"] = 0.0

        agg_cols = {
            "bar5s_trade_vol_sum": "sum",
            "bar5s_trade_cnt_sum": "sum",
            "bar5s_trade_aggbuy_vol_sum": "sum",
            "bar5s_trade_aggsell_vol_sum": "sum",
            "bar5s_trade_signed_vol_sum": "sum",
            "bar5s_meta_msg_cnt_sum": "sum",
            "_flow_add_bid_total": "sum",
            "_flow_add_ask_total": "sum",
            "_flow_rem_bid_total": "sum",
            "_flow_rem_ask_total": "sum",
            "_flow_net_bid_total": "sum",
            "_flow_net_ask_total": "sum",
        }
        daily_buckets = df.groupby(["_date", "bucket_id"]).agg(agg_cols).reset_index()

        profile_rows = []
        for bucket_id in range(N_BUCKETS):
            bucket_data = daily_buckets[daily_buckets["bucket_id"] == bucket_id]

            row = {
                "bucket_id": bucket_id,
                "bucket_start_time": bucket_start_time(bucket_id),
                "trade_vol_mean": bucket_data["bar5s_trade_vol_sum"].mean() if len(bucket_data) > 0 else 0.0,
                "trade_vol_std": bucket_data["bar5s_trade_vol_sum"].std() if len(bucket_data) > 1 else 0.0,
                "trade_cnt_mean": bucket_data["bar5s_trade_cnt_sum"].mean() if len(bucket_data) > 0 else 0.0,
                "trade_cnt_std": bucket_data["bar5s_trade_cnt_sum"].std() if len(bucket_data) > 1 else 0.0,
                "trade_aggbuy_vol_mean": bucket_data["bar5s_trade_aggbuy_vol_sum"].mean() if len(bucket_data) > 0 else 0.0,
                "trade_aggbuy_vol_std": bucket_data["bar5s_trade_aggbuy_vol_sum"].std() if len(bucket_data) > 1 else 0.0,
                "trade_aggsell_vol_mean": bucket_data["bar5s_trade_aggsell_vol_sum"].mean() if len(bucket_data) > 0 else 0.0,
                "trade_aggsell_vol_std": bucket_data["bar5s_trade_aggsell_vol_sum"].std() if len(bucket_data) > 1 else 0.0,
                "trade_signed_vol_mean": bucket_data["bar5s_trade_signed_vol_sum"].mean() if len(bucket_data) > 0 else 0.0,
                "trade_signed_vol_std": bucket_data["bar5s_trade_signed_vol_sum"].std() if len(bucket_data) > 1 else 0.0,
                "flow_add_vol_bid_mean": bucket_data["_flow_add_bid_total"].mean() if len(bucket_data) > 0 else 0.0,
                "flow_add_vol_bid_std": bucket_data["_flow_add_bid_total"].std() if len(bucket_data) > 1 else 0.0,
                "flow_add_vol_ask_mean": bucket_data["_flow_add_ask_total"].mean() if len(bucket_data) > 0 else 0.0,
                "flow_add_vol_ask_std": bucket_data["_flow_add_ask_total"].std() if len(bucket_data) > 1 else 0.0,
                "flow_rem_vol_bid_mean": bucket_data["_flow_rem_bid_total"].mean() if len(bucket_data) > 0 else 0.0,
                "flow_rem_vol_bid_std": bucket_data["_flow_rem_bid_total"].std() if len(bucket_data) > 1 else 0.0,
                "flow_rem_vol_ask_mean": bucket_data["_flow_rem_ask_total"].mean() if len(bucket_data) > 0 else 0.0,
                "flow_rem_vol_ask_std": bucket_data["_flow_rem_ask_total"].std() if len(bucket_data) > 1 else 0.0,
                "flow_net_vol_bid_mean": bucket_data["_flow_net_bid_total"].mean() if len(bucket_data) > 0 else 0.0,
                "flow_net_vol_bid_std": bucket_data["_flow_net_bid_total"].std() if len(bucket_data) > 1 else 0.0,
                "flow_net_vol_ask_mean": bucket_data["_flow_net_ask_total"].mean() if len(bucket_data) > 0 else 0.0,
                "flow_net_vol_ask_std": bucket_data["_flow_net_ask_total"].std() if len(bucket_data) > 1 else 0.0,
                "msg_cnt_mean": bucket_data["bar5s_meta_msg_cnt_sum"].mean() if len(bucket_data) > 0 else 0.0,
                "msg_cnt_std": bucket_data["bar5s_meta_msg_cnt_sum"].std() if len(bucket_data) > 1 else 0.0,
                "days_in_lookback": n_days,
            }
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    row[k] = 0.0
            profile_rows.append(row)

        return pd.DataFrame(profile_rows)

    def _build_fallback_profile(self, n_days: int) -> pd.DataFrame:
        rows = []
        for bucket_id in range(N_BUCKETS):
            row = {
                "bucket_id": bucket_id,
                "bucket_start_time": bucket_start_time(bucket_id),
                "trade_vol_mean": 0.0,
                "trade_vol_std": 0.0,
                "trade_cnt_mean": 0.0,
                "trade_cnt_std": 0.0,
                "trade_aggbuy_vol_mean": 0.0,
                "trade_aggbuy_vol_std": 0.0,
                "trade_aggsell_vol_mean": 0.0,
                "trade_aggsell_vol_std": 0.0,
                "trade_signed_vol_mean": 0.0,
                "trade_signed_vol_std": 0.0,
                "flow_add_vol_bid_mean": 0.0,
                "flow_add_vol_bid_std": 0.0,
                "flow_add_vol_ask_mean": 0.0,
                "flow_add_vol_ask_std": 0.0,
                "flow_rem_vol_bid_mean": 0.0,
                "flow_rem_vol_bid_std": 0.0,
                "flow_rem_vol_ask_mean": 0.0,
                "flow_rem_vol_ask_std": 0.0,
                "flow_net_vol_bid_mean": 0.0,
                "flow_net_vol_bid_std": 0.0,
                "flow_net_vol_ask_mean": 0.0,
                "flow_net_vol_ask_std": 0.0,
                "msg_cnt_mean": 0.0,
                "msg_cnt_std": 0.0,
                "days_in_lookback": n_days,
            }
            rows.append(row)
        return pd.DataFrame(rows)
