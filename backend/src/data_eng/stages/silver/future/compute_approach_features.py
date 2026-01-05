from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

EPSILON = 1e-9
POINT = 0.25
BARS_PER_BUCKET = 60
RTH_START_HOUR = 9
RTH_START_MINUTE = 30
BUCKET_MINUTES = 5
N_BUCKETS = 48

LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]

DERIV_WINDOWS = [3, 12, 36, 72]

DERIV_BASE_FEATURES = {
    "dist": "bar5s_approach_dist_to_level_pts_eob",
    "obi0": "bar5s_state_obi0_eob",
    "obi10": "bar5s_state_obi10_eob",
    "cdi01": "bar5s_state_cdi_p0_1_eob",
    "cdi12": "bar5s_state_cdi_p1_2_eob",
    "dbid10": "bar5s_depth_bid10_qty_eob",
    "dask10": "bar5s_depth_ask10_qty_eob",
    "dbelow01": "bar5s_depth_below_p0_1_qty_eob",
    "dabove01": "bar5s_depth_above_p0_1_qty_eob",
    "wbidz": "bar5s_wall_bid_maxz_eob",
    "waskz": "bar5s_wall_ask_maxz_eob",
}

FLOW_BANDS = ["p0_1", "p1_2", "p2_3"]


def get_bucket_id(ts_ns: int) -> int:
    dt = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")
    minutes_since_open = (dt.hour - RTH_START_HOUR) * 60 + dt.minute - RTH_START_MINUTE
    bucket_id = minutes_since_open // BUCKET_MINUTES
    return max(0, min(N_BUCKETS - 1, bucket_id))


class SilverComputeApproachFeatures(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_approach_features",
            io=StageIO(
                inputs=[
                    f"silver.future.market_by_price_10_{lt.lower()}_episodes"
                    for lt in LEVEL_TYPES
                ] + ["silver.future.volume_profiles"],
                output="silver.future.market_by_price_10_pm_high_approach",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        output_keys = [
            f"silver.future.market_by_price_10_{lt.lower()}_approach"
            for lt in LEVEL_TYPES
        ]

        all_complete = all(
            is_partition_complete(partition_ref(cfg, k, symbol, dt))
            for k in output_keys
        )
        if all_complete:
            return

        profile_ref = partition_ref(cfg, "silver.future.volume_profiles", symbol, dt)
        df_profile: Optional[pd.DataFrame] = None
        if is_partition_complete(profile_ref):
            df_profile = read_partition(profile_ref)

        for level_type in LEVEL_TYPES:
            input_key = f"silver.future.market_by_price_10_{level_type.lower()}_episodes"
            output_key = f"silver.future.market_by_price_10_{level_type.lower()}_approach"

            out_ref = partition_ref(cfg, output_key, symbol, dt)
            if is_partition_complete(out_ref):
                continue

            in_ref = partition_ref(cfg, input_key, symbol, dt)
            if not is_partition_complete(in_ref):
                df_out = pd.DataFrame()
            else:
                in_contract_path = repo_root / cfg.dataset(input_key).contract
                in_contract = load_avro_contract(in_contract_path)
                df_in = read_partition(in_ref)

                if len(df_in) == 0:
                    df_out = pd.DataFrame()
                else:
                    df_in = enforce_contract(df_in, in_contract)
                    df_out = self._compute_approach_features(df_in, level_type, df_profile)

            out_contract_path = repo_root / cfg.dataset(output_key).contract
            out_contract = load_avro_contract(out_contract_path)

            if len(df_out) > 0:
                df_out = enforce_contract(df_out, out_contract)

            lineage = []
            if is_partition_complete(in_ref):
                lineage.append({
                    "dataset": in_ref.dataset_key,
                    "dt": dt,
                    "manifest_sha256": read_manifest_hash(in_ref),
                })

            write_partition(
                cfg=cfg,
                dataset_key=output_key,
                symbol=symbol,
                dt=dt,
                df=df_out,
                contract_path=out_contract_path,
                inputs=lineage,
                stage=self.name,
            )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        raise NotImplementedError("Use run() directly")

    def _compute_approach_features(
        self, df: pd.DataFrame, level_type: str, df_profile: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df = df.copy()

        df = self._compute_position_features(df, level_type)
        df = self._compute_cumulative_features(df)
        df = self._compute_derivative_features(df)
        df = self._compute_level_relative_book_features(df)
        df = self._compute_setup_signature_features(df)
        df = self._compute_relative_volume_features(df, df_profile)

        return df

    def _compute_position_features(
        self, df: pd.DataFrame, level_type: str
    ) -> pd.DataFrame:
        level_price = df["level_price"].values
        microprice = df["bar5s_microprice_eob"].values

        df["bar5s_approach_dist_to_level_pts_eob"] = (microprice - level_price) / POINT

        mid_bid = df["bar5s_depth_bid10_qty_eob"].values
        mid_ask = df["bar5s_depth_ask10_qty_eob"].values
        bid_px_00 = microprice - df["bar5s_state_spread_pts_eob"].values * POINT / 2
        ask_px_00 = microprice + df["bar5s_state_spread_pts_eob"].values * POINT / 2
        bid_sz_00 = df["bar5s_shape_bid_sz_l00_eob"].values
        ask_sz_00 = df["bar5s_shape_ask_sz_l00_eob"].values

        micro_twa = np.where(
            (bid_sz_00 + ask_sz_00) > EPSILON,
            (ask_px_00 * bid_sz_00 + bid_px_00 * ask_sz_00) / (bid_sz_00 + ask_sz_00 + EPSILON),
            microprice,
        )
        df["bar5s_approach_dist_to_level_pts_twa"] = (micro_twa - level_price) / POINT

        df["bar5s_approach_abs_dist_to_level_pts_eob"] = np.abs(
            df["bar5s_approach_dist_to_level_pts_eob"]
        )

        df["bar5s_approach_side_of_level_eob"] = np.where(
            microprice > level_price, 1, -1
        ).astype(np.int8)

        df["bar5s_approach_is_pm_high"] = 1 if level_type == "PM_HIGH" else 0
        df["bar5s_approach_is_pm_low"] = 1 if level_type == "PM_LOW" else 0
        df["bar5s_approach_is_or_high"] = 1 if level_type == "OR_HIGH" else 0
        df["bar5s_approach_is_or_low"] = 1 if level_type == "OR_LOW" else 0

        df["bar5s_approach_level_polarity"] = (
            1 if level_type in ["PM_HIGH", "OR_HIGH"] else -1
        )

        df["bar5s_approach_alignment_eob"] = (
            -1 * df["bar5s_approach_side_of_level_eob"] * df["bar5s_approach_level_polarity"]
        )

        return df

    def _compute_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("touch_id", sort=False)

        cumsum_mappings = {
            "bar5s_cumul_trade_vol": "bar5s_trade_vol_sum",
            "bar5s_cumul_signed_trade_vol": "bar5s_trade_signed_vol_sum",
            "bar5s_cumul_aggbuy_vol": "bar5s_trade_aggbuy_vol_sum",
            "bar5s_cumul_aggsell_vol": "bar5s_trade_aggsell_vol_sum",
            "bar5s_cumul_msg_cnt": "bar5s_meta_msg_cnt_sum",
            "bar5s_cumul_trade_cnt": "bar5s_trade_cnt_sum",
            "bar5s_cumul_add_cnt": "bar5s_meta_add_cnt_sum",
            "bar5s_cumul_cancel_cnt": "bar5s_meta_cancel_cnt_sum",
        }

        for out_col, in_col in cumsum_mappings.items():
            if in_col in df.columns:
                df[out_col] = g[in_col].cumsum()

        bars_elapsed = g.cumcount() + 1
        df["bar5s_cumul_signed_trade_vol_rate"] = df["bar5s_cumul_signed_trade_vol"] / bars_elapsed

        flow_bid_total = np.zeros(len(df), dtype=np.float64)
        flow_ask_total = np.zeros(len(df), dtype=np.float64)

        for band in FLOW_BANDS:
            bid_col = f"bar5s_flow_net_vol_bid_{band}_sum"
            ask_col = f"bar5s_flow_net_vol_ask_{band}_sum"

            if bid_col in df.columns:
                df[f"bar5s_cumul_flow_net_bid_{band}"] = g[bid_col].cumsum()
                flow_bid_total += df[f"bar5s_cumul_flow_net_bid_{band}"].values

            if ask_col in df.columns:
                df[f"bar5s_cumul_flow_net_ask_{band}"] = g[ask_col].cumsum()
                flow_ask_total += df[f"bar5s_cumul_flow_net_ask_{band}"].values

        df["bar5s_cumul_flow_net_bid"] = flow_bid_total
        df["bar5s_cumul_flow_net_ask"] = flow_ask_total
        df["bar5s_cumul_flow_imbal"] = flow_bid_total - flow_ask_total
        df["bar5s_cumul_flow_imbal_rate"] = df["bar5s_cumul_flow_imbal"] / bars_elapsed

        return df

    def _compute_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("touch_id", sort=False)

        for short_name, full_col in DERIV_BASE_FEATURES.items():
            if full_col not in df.columns:
                continue

            for window in DERIV_WINDOWS:
                d1_col = f"bar5s_deriv_{short_name}_d1_w{window}"
                d2_col = f"bar5s_deriv_{short_name}_d2_w{window}"

                df[d1_col] = g[full_col].transform(lambda x: (x - x.shift(window)) / window)
                df[d2_col] = g[d1_col].transform(lambda x: (x - x.shift(window)) / window)

        return df

    def _compute_level_relative_book_features(self, df: pd.DataFrame) -> pd.DataFrame:
        level_price = df["level_price"].values
        microprice = df["bar5s_microprice_eob"].values
        spread = df["bar5s_state_spread_pts_eob"].values * POINT

        bid_px_00 = microprice - spread / 2
        ask_px_00 = microprice + spread / 2

        df["bar5s_lvl_depth_above_qty_eob"] = np.where(
            ask_px_00 > level_price,
            df["bar5s_depth_ask10_qty_eob"].values,
            0,
        )
        df["bar5s_lvl_depth_below_qty_eob"] = np.where(
            bid_px_00 < level_price,
            df["bar5s_depth_bid10_qty_eob"].values,
            0,
        )

        dist_from_level = np.abs(microprice - level_price)
        df["bar5s_lvl_depth_at_qty_eob"] = np.where(
            dist_from_level <= 0.5 * POINT,
            df["bar5s_shape_bid_sz_l00_eob"].values + df["bar5s_shape_ask_sz_l00_eob"].values,
            0,
        )

        above = df["bar5s_lvl_depth_above_qty_eob"].values
        below = df["bar5s_lvl_depth_below_qty_eob"].values
        df["bar5s_lvl_depth_imbal_eob"] = (below - above) / (below + above + EPSILON)

        for band in ["p0_1", "p1_2", "p2_3"]:
            below_col = f"bar5s_depth_below_{band}_qty_eob"
            above_col = f"bar5s_depth_above_{band}_qty_eob"

            if below_col in df.columns and above_col in df.columns:
                df[f"bar5s_lvl_depth_above_{band}_qty_eob"] = df[above_col].values
                df[f"bar5s_lvl_depth_below_{band}_qty_eob"] = df[below_col].values

                b = df[below_col].values
                a = df[above_col].values
                df[f"bar5s_lvl_cdi_{band}_eob"] = (b - a) / (b + a + EPSILON)

        side_of_level = df["bar5s_approach_side_of_level_eob"].values

        flow_bid_sum = np.zeros(len(df))
        flow_ask_sum = np.zeros(len(df))

        for band in FLOW_BANDS:
            bid_col = f"bar5s_flow_net_vol_bid_{band}_sum"
            ask_col = f"bar5s_flow_net_vol_ask_{band}_sum"
            if bid_col in df.columns:
                flow_bid_sum += df[bid_col].values
            if ask_col in df.columns:
                flow_ask_sum += df[ask_col].values

        toward_flow = np.where(side_of_level < 0, flow_ask_sum, flow_bid_sum)
        away_flow = np.where(side_of_level < 0, flow_bid_sum, flow_ask_sum)

        df["bar5s_lvl_flow_toward_net_sum"] = toward_flow
        df["bar5s_lvl_flow_away_net_sum"] = away_flow
        df["bar5s_lvl_flow_toward_away_imbal_sum"] = toward_flow - away_flow

        return df

    def _compute_setup_signature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("touch_id", sort=False)
        dist_col = "bar5s_approach_dist_to_level_pts_eob"

        df["_abs_dist"] = np.abs(df[dist_col])

        df["bar5s_setup_start_dist_pts"] = g[dist_col].transform("first")
        df["bar5s_setup_min_dist_pts"] = g["_abs_dist"].transform("min")
        df["bar5s_setup_max_dist_pts"] = g["_abs_dist"].transform("max")
        df["bar5s_setup_dist_range_pts"] = df["bar5s_setup_max_dist_pts"] - df["bar5s_setup_min_dist_pts"]

        df["_delta_dist"] = g["_abs_dist"].diff().fillna(0.0)
        df["bar5s_setup_approach_bars"] = g["_delta_dist"].transform(lambda x: (x < 0).sum())
        df["bar5s_setup_retreat_bars"] = g["_delta_dist"].transform(lambda x: (x > 0).sum())
        n_bars = g["_delta_dist"].transform("count")
        df["bar5s_setup_approach_ratio"] = df["bar5s_setup_approach_bars"] / (n_bars + EPSILON)

        d1_col = "bar5s_deriv_dist_d1_w3"
        if d1_col in df.columns:
            df["_bar_idx"] = g.cumcount()
            df["_n_bars"] = n_bars
            df["_third"] = np.maximum(1, df["_n_bars"] // 3)

            def _velocity_stats(grp: pd.DataFrame) -> pd.Series:
                d1 = grp[d1_col].values
                n = len(d1)
                third = max(1, n // 3)
                early = np.nanmean(d1[:third]) if third > 0 else 0.0
                mid = np.nanmean(d1[third:2*third]) if 2*third > third else 0.0
                late = np.nanmean(d1[2*third:]) if n > 2*third else 0.0
                return pd.Series({"early": early, "mid": mid, "late": late, "trend": late - early}, index=["early", "mid", "late", "trend"])

            vel_stats = g.apply(_velocity_stats, include_groups=False).reset_index()
            vel_stats.columns = ["touch_id", "early", "mid", "late", "trend"]
            df = df.merge(vel_stats.rename(columns={
                "early": "bar5s_setup_early_velocity",
                "mid": "bar5s_setup_mid_velocity",
                "late": "bar5s_setup_late_velocity",
                "trend": "bar5s_setup_velocity_trend",
            }), on="touch_id", how="left")
        else:
            df["bar5s_setup_early_velocity"] = 0.0
            df["bar5s_setup_mid_velocity"] = 0.0
            df["bar5s_setup_late_velocity"] = 0.0
            df["bar5s_setup_velocity_trend"] = 0.0

        g = df.groupby("touch_id", sort=False)

        for metric, col in [("obi0", "bar5s_state_obi0_eob"), ("obi10", "bar5s_state_obi10_eob")]:
            if col in df.columns:
                df[f"bar5s_setup_{metric}_start"] = g[col].transform("first")
                df[f"bar5s_setup_{metric}_end"] = g[col].transform("last")
                df[f"bar5s_setup_{metric}_delta"] = df[f"bar5s_setup_{metric}_end"] - df[f"bar5s_setup_{metric}_start"]
                df[f"bar5s_setup_{metric}_min"] = g[col].transform("min")
                df[f"bar5s_setup_{metric}_max"] = g[col].transform("max")

        df["bar5s_setup_total_trade_vol"] = g["bar5s_trade_vol_sum"].transform("sum")
        df["bar5s_setup_total_signed_vol"] = g["bar5s_trade_signed_vol_sum"].transform("sum")
        df["bar5s_setup_trade_imbal_pct"] = df["bar5s_setup_total_signed_vol"] / (df["bar5s_setup_total_trade_vol"] + EPSILON)

        df["bar5s_setup_flow_imbal_total"] = g["bar5s_cumul_flow_imbal"].transform("last")

        df["bar5s_setup_bid_wall_max_z"] = g["bar5s_wall_bid_maxz_eob"].transform("max")
        df["bar5s_setup_ask_wall_max_z"] = g["bar5s_wall_ask_maxz_eob"].transform("max")

        df["_bid_wall_strong"] = (df["bar5s_wall_bid_maxz_eob"] > 2.0).astype(np.int32)
        df["_ask_wall_strong"] = (df["bar5s_wall_ask_maxz_eob"] > 2.0).astype(np.int32)

        g = df.groupby("touch_id", sort=False)
        df["bar5s_setup_bid_wall_bars"] = g["_bid_wall_strong"].transform("sum")
        df["bar5s_setup_ask_wall_bars"] = g["_ask_wall_strong"].transform("sum")
        df["bar5s_setup_wall_imbal"] = df["bar5s_setup_ask_wall_bars"] - df["bar5s_setup_bid_wall_bars"]

        temp_cols = [c for c in df.columns if c.startswith("_")]
        df.drop(columns=temp_cols, inplace=True)

        return df

    def _compute_relative_volume_features(
        self, df: pd.DataFrame, df_profile: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        n = len(df)
        rvol_cols = [
            "rvol_trade_vol_ratio", "rvol_trade_vol_zscore",
            "rvol_trade_cnt_ratio", "rvol_trade_cnt_zscore",
            "rvol_trade_aggbuy_ratio", "rvol_trade_aggsell_ratio",
            "rvol_trade_aggbuy_zscore", "rvol_trade_aggsell_zscore",
            "rvol_flow_add_bid_ratio", "rvol_flow_add_ask_ratio",
            "rvol_flow_add_bid_zscore", "rvol_flow_add_ask_zscore",
            "rvol_flow_net_bid_ratio", "rvol_flow_net_ask_ratio",
            "rvol_flow_net_bid_zscore", "rvol_flow_net_ask_zscore",
            "rvol_flow_add_total_ratio", "rvol_flow_add_total_zscore",
            "rvol_cumul_trade_vol_dev", "rvol_cumul_trade_vol_dev_pct",
            "rvol_cumul_flow_imbal_dev", "rvol_cumul_msg_dev",
            "rvol_bid_ask_add_asymmetry", "rvol_bid_ask_rem_asymmetry",
            "rvol_bid_ask_net_asymmetry", "rvol_aggbuy_aggsell_asymmetry",
            "rvol_lookback_trade_vol_mean_ratio", "rvol_lookback_trade_vol_max_ratio",
            "rvol_lookback_trade_vol_trend", "rvol_lookback_elevated_bars",
            "rvol_lookback_depressed_bars", "rvol_lookback_asymmetry_mean",
            "rvol_recent_vs_lookback_vol_ratio", "rvol_recent_vs_lookback_asymmetry",
        ]

        if df_profile is None or len(df_profile) == 0 or n == 0:
            for col in rvol_cols:
                df[col] = np.nan
            return df

        df["_bucket_id"] = df["bar_ts"].apply(get_bucket_id)

        profile_dict = {}
        for _, row in df_profile.iterrows():
            bid = int(row["bucket_id"])
            profile_dict[bid] = row.to_dict()

        def safe_get(bucket_id: int, key: str, default: float = 0.0) -> float:
            if bucket_id in profile_dict:
                val = profile_dict[bucket_id].get(key, default)
                return val if not np.isnan(val) else default
            return default

        bucket_ids = df["_bucket_id"].values
        trade_vol = df["bar5s_trade_vol_sum"].values
        trade_cnt = df["bar5s_trade_cnt_sum"].values
        trade_aggbuy = df["bar5s_trade_aggbuy_vol_sum"].values
        trade_aggsell = df["bar5s_trade_aggsell_vol_sum"].values
        msg_cnt = df["bar5s_meta_msg_cnt_sum"].values

        flow_add_bid = np.zeros(n)
        flow_add_ask = np.zeros(n)
        flow_net_bid = np.zeros(n)
        flow_net_ask = np.zeros(n)
        flow_rem_bid = np.zeros(n)
        flow_rem_ask = np.zeros(n)

        for band in FLOW_BANDS:
            bid_add_col = f"bar5s_flow_add_vol_bid_{band}_sum"
            ask_add_col = f"bar5s_flow_add_vol_ask_{band}_sum"
            bid_net_col = f"bar5s_flow_net_vol_bid_{band}_sum"
            ask_net_col = f"bar5s_flow_net_vol_ask_{band}_sum"
            bid_rem_col = f"bar5s_flow_rem_vol_bid_{band}_sum"
            ask_rem_col = f"bar5s_flow_rem_vol_ask_{band}_sum"

            if bid_add_col in df.columns:
                flow_add_bid += df[bid_add_col].values
            if ask_add_col in df.columns:
                flow_add_ask += df[ask_add_col].values
            if bid_net_col in df.columns:
                flow_net_bid += df[bid_net_col].values
            if ask_net_col in df.columns:
                flow_net_ask += df[ask_net_col].values
            if bid_rem_col in df.columns:
                flow_rem_bid += df[bid_rem_col].values
            if ask_rem_col in df.columns:
                flow_rem_ask += df[ask_rem_col].values

        rvol_trade_vol_ratio = np.zeros(n)
        rvol_trade_vol_zscore = np.zeros(n)
        rvol_trade_cnt_ratio = np.zeros(n)
        rvol_trade_cnt_zscore = np.zeros(n)
        rvol_trade_aggbuy_ratio = np.zeros(n)
        rvol_trade_aggsell_ratio = np.zeros(n)
        rvol_trade_aggbuy_zscore = np.zeros(n)
        rvol_trade_aggsell_zscore = np.zeros(n)
        rvol_flow_add_bid_ratio = np.zeros(n)
        rvol_flow_add_ask_ratio = np.zeros(n)
        rvol_flow_add_bid_zscore = np.zeros(n)
        rvol_flow_add_ask_zscore = np.zeros(n)
        rvol_flow_net_bid_ratio = np.zeros(n)
        rvol_flow_net_ask_ratio = np.zeros(n)
        rvol_flow_net_bid_zscore = np.zeros(n)
        rvol_flow_net_ask_zscore = np.zeros(n)
        rvol_flow_add_total_ratio = np.zeros(n)
        rvol_flow_add_total_zscore = np.zeros(n)
        rvol_flow_rem_bid_zscore = np.zeros(n)
        rvol_flow_rem_ask_zscore = np.zeros(n)
        expected_trade_vol = np.zeros(n)
        expected_msg_cnt = np.zeros(n)
        expected_flow_imbal = np.zeros(n)

        for i in range(n):
            bid = int(bucket_ids[i])

            tv_mean = safe_get(bid, "trade_vol_mean") / BARS_PER_BUCKET
            tv_std = safe_get(bid, "trade_vol_std") / np.sqrt(BARS_PER_BUCKET)
            tc_mean = safe_get(bid, "trade_cnt_mean") / BARS_PER_BUCKET
            tc_std = safe_get(bid, "trade_cnt_std") / np.sqrt(BARS_PER_BUCKET)
            ab_mean = safe_get(bid, "trade_aggbuy_vol_mean") / BARS_PER_BUCKET
            ab_std = safe_get(bid, "trade_aggbuy_vol_std") / np.sqrt(BARS_PER_BUCKET)
            as_mean = safe_get(bid, "trade_aggsell_vol_mean") / BARS_PER_BUCKET
            as_std = safe_get(bid, "trade_aggsell_vol_std") / np.sqrt(BARS_PER_BUCKET)
            fab_mean = safe_get(bid, "flow_add_vol_bid_mean") / BARS_PER_BUCKET
            fab_std = safe_get(bid, "flow_add_vol_bid_std") / np.sqrt(BARS_PER_BUCKET)
            faa_mean = safe_get(bid, "flow_add_vol_ask_mean") / BARS_PER_BUCKET
            faa_std = safe_get(bid, "flow_add_vol_ask_std") / np.sqrt(BARS_PER_BUCKET)
            fnb_mean = safe_get(bid, "flow_net_vol_bid_mean") / BARS_PER_BUCKET
            fnb_std = safe_get(bid, "flow_net_vol_bid_std") / np.sqrt(BARS_PER_BUCKET)
            fna_mean = safe_get(bid, "flow_net_vol_ask_mean") / BARS_PER_BUCKET
            fna_std = safe_get(bid, "flow_net_vol_ask_std") / np.sqrt(BARS_PER_BUCKET)
            frb_mean = safe_get(bid, "flow_rem_vol_bid_mean") / BARS_PER_BUCKET
            frb_std = safe_get(bid, "flow_rem_vol_bid_std") / np.sqrt(BARS_PER_BUCKET)
            fra_mean = safe_get(bid, "flow_rem_vol_ask_mean") / BARS_PER_BUCKET
            fra_std = safe_get(bid, "flow_rem_vol_ask_std") / np.sqrt(BARS_PER_BUCKET)
            msg_mean = safe_get(bid, "msg_cnt_mean") / BARS_PER_BUCKET
            msg_std = safe_get(bid, "msg_cnt_std") / np.sqrt(BARS_PER_BUCKET)

            rvol_trade_vol_ratio[i] = trade_vol[i] / (tv_mean + EPSILON)
            rvol_trade_vol_zscore[i] = (trade_vol[i] - tv_mean) / (tv_std + EPSILON)
            rvol_trade_cnt_ratio[i] = trade_cnt[i] / (tc_mean + EPSILON)
            rvol_trade_cnt_zscore[i] = (trade_cnt[i] - tc_mean) / (tc_std + EPSILON)
            rvol_trade_aggbuy_ratio[i] = trade_aggbuy[i] / (ab_mean + EPSILON)
            rvol_trade_aggsell_ratio[i] = trade_aggsell[i] / (as_mean + EPSILON)
            rvol_trade_aggbuy_zscore[i] = (trade_aggbuy[i] - ab_mean) / (ab_std + EPSILON)
            rvol_trade_aggsell_zscore[i] = (trade_aggsell[i] - as_mean) / (as_std + EPSILON)
            rvol_flow_add_bid_ratio[i] = flow_add_bid[i] / (fab_mean + EPSILON)
            rvol_flow_add_ask_ratio[i] = flow_add_ask[i] / (faa_mean + EPSILON)
            rvol_flow_add_bid_zscore[i] = (flow_add_bid[i] - fab_mean) / (fab_std + EPSILON)
            rvol_flow_add_ask_zscore[i] = (flow_add_ask[i] - faa_mean) / (faa_std + EPSILON)
            rvol_flow_net_bid_ratio[i] = flow_net_bid[i] / (fnb_mean + EPSILON) if fnb_mean != 0 else 1.0
            rvol_flow_net_ask_ratio[i] = flow_net_ask[i] / (fna_mean + EPSILON) if fna_mean != 0 else 1.0
            rvol_flow_net_bid_zscore[i] = (flow_net_bid[i] - fnb_mean) / (fnb_std + EPSILON)
            rvol_flow_net_ask_zscore[i] = (flow_net_ask[i] - fna_mean) / (fna_std + EPSILON)
            rvol_flow_add_total_ratio[i] = (flow_add_bid[i] + flow_add_ask[i]) / (fab_mean + faa_mean + EPSILON)
            rvol_flow_add_total_zscore[i] = ((flow_add_bid[i] + flow_add_ask[i]) - (fab_mean + faa_mean)) / (np.sqrt(fab_std**2 + faa_std**2) + EPSILON)
            rvol_flow_rem_bid_zscore[i] = (flow_rem_bid[i] - frb_mean) / (frb_std + EPSILON)
            rvol_flow_rem_ask_zscore[i] = (flow_rem_ask[i] - fra_mean) / (fra_std + EPSILON)

            expected_trade_vol[i] = tv_mean
            expected_msg_cnt[i] = msg_mean
            expected_flow_imbal[i] = fnb_mean - fna_mean

        df["rvol_trade_vol_ratio"] = rvol_trade_vol_ratio
        df["rvol_trade_vol_zscore"] = rvol_trade_vol_zscore
        df["rvol_trade_cnt_ratio"] = rvol_trade_cnt_ratio
        df["rvol_trade_cnt_zscore"] = rvol_trade_cnt_zscore
        df["rvol_trade_aggbuy_ratio"] = rvol_trade_aggbuy_ratio
        df["rvol_trade_aggsell_ratio"] = rvol_trade_aggsell_ratio
        df["rvol_trade_aggbuy_zscore"] = rvol_trade_aggbuy_zscore
        df["rvol_trade_aggsell_zscore"] = rvol_trade_aggsell_zscore
        df["rvol_flow_add_bid_ratio"] = rvol_flow_add_bid_ratio
        df["rvol_flow_add_ask_ratio"] = rvol_flow_add_ask_ratio
        df["rvol_flow_add_bid_zscore"] = rvol_flow_add_bid_zscore
        df["rvol_flow_add_ask_zscore"] = rvol_flow_add_ask_zscore
        df["rvol_flow_net_bid_ratio"] = rvol_flow_net_bid_ratio
        df["rvol_flow_net_ask_ratio"] = rvol_flow_net_ask_ratio
        df["rvol_flow_net_bid_zscore"] = rvol_flow_net_bid_zscore
        df["rvol_flow_net_ask_zscore"] = rvol_flow_net_ask_zscore
        df["rvol_flow_add_total_ratio"] = rvol_flow_add_total_ratio
        df["rvol_flow_add_total_zscore"] = rvol_flow_add_total_zscore

        df["rvol_bid_ask_add_asymmetry"] = rvol_flow_add_bid_zscore - rvol_flow_add_ask_zscore
        df["rvol_bid_ask_rem_asymmetry"] = rvol_flow_rem_bid_zscore - rvol_flow_rem_ask_zscore
        df["rvol_bid_ask_net_asymmetry"] = rvol_flow_net_bid_zscore - rvol_flow_net_ask_zscore
        df["rvol_aggbuy_aggsell_asymmetry"] = rvol_trade_aggbuy_zscore - rvol_trade_aggsell_zscore

        g = df.groupby("touch_id", sort=False)
        df["_expected_trade_vol_cumul"] = g.apply(
            lambda x: pd.Series(expected_trade_vol[x.index]).cumsum().values,
            include_groups=False
        ).explode().astype(float).values
        df["_actual_trade_vol_cumul"] = g["bar5s_trade_vol_sum"].cumsum()
        df["rvol_cumul_trade_vol_dev"] = df["_actual_trade_vol_cumul"] - df["_expected_trade_vol_cumul"]
        df["rvol_cumul_trade_vol_dev_pct"] = df["rvol_cumul_trade_vol_dev"] / (df["_expected_trade_vol_cumul"] + EPSILON)

        df["_expected_msg_cnt_cumul"] = g.apply(
            lambda x: pd.Series(expected_msg_cnt[x.index]).cumsum().values,
            include_groups=False
        ).explode().astype(float).values
        df["_actual_msg_cnt_cumul"] = g["bar5s_meta_msg_cnt_sum"].cumsum()
        df["rvol_cumul_msg_dev"] = df["_actual_msg_cnt_cumul"] - df["_expected_msg_cnt_cumul"]

        actual_flow_imbal = flow_net_bid - flow_net_ask
        df["_actual_flow_imbal"] = actual_flow_imbal
        df["_expected_flow_imbal"] = expected_flow_imbal
        df["_expected_flow_imbal_cumul"] = g.apply(
            lambda x: pd.Series(expected_flow_imbal[x.index]).cumsum().values,
            include_groups=False
        ).explode().astype(float).values
        df["_actual_flow_imbal_cumul"] = g["_actual_flow_imbal"].cumsum()
        df["rvol_cumul_flow_imbal_dev"] = df["_actual_flow_imbal_cumul"] - df["_expected_flow_imbal_cumul"]

        df["rvol_lookback_trade_vol_mean_ratio"] = g["rvol_trade_vol_ratio"].transform("mean")
        df["rvol_lookback_trade_vol_max_ratio"] = g["rvol_trade_vol_ratio"].transform("max")

        def compute_trend(grp: pd.DataFrame) -> pd.Series:
            vals = grp["rvol_trade_vol_ratio"].values
            n_vals = len(vals)
            if n_vals < 2:
                return pd.Series([0.0] * n_vals, index=grp.index)
            split = max(1, n_vals // 3)
            early_mean = np.nanmean(vals[:split])
            late_mean = np.nanmean(vals[-split:])
            trend = late_mean - early_mean
            return pd.Series([trend] * n_vals, index=grp.index)

        df["rvol_lookback_trade_vol_trend"] = g.apply(compute_trend, include_groups=False).values

        df["_elevated"] = (df["rvol_trade_vol_ratio"] > 1.5).astype(float)
        df["_depressed"] = (df["rvol_trade_vol_ratio"] < 0.5).astype(float)
        df["rvol_lookback_elevated_bars"] = g["_elevated"].transform("sum")
        df["rvol_lookback_depressed_bars"] = g["_depressed"].transform("sum")

        df["rvol_lookback_asymmetry_mean"] = g["rvol_bid_ask_net_asymmetry"].transform("mean")

        def compute_recent_vs_lookback(grp: pd.DataFrame, col: str) -> pd.Series:
            vals = grp[col].values
            n_vals = len(vals)
            result = np.zeros(n_vals)
            for i in range(n_vals):
                lookback_mean = np.nanmean(vals[:i+1]) if i > 0 else vals[0]
                recent_start = max(0, i - 11)
                recent_mean = np.nanmean(vals[recent_start:i+1])
                result[i] = recent_mean / (lookback_mean + EPSILON) if lookback_mean != 0 else 1.0
            return pd.Series(result, index=grp.index)

        df["rvol_recent_vs_lookback_vol_ratio"] = g.apply(
            lambda x: compute_recent_vs_lookback(x, "rvol_trade_vol_ratio"),
            include_groups=False
        ).values
        df["rvol_recent_vs_lookback_asymmetry"] = g.apply(
            lambda x: compute_recent_vs_lookback(x, "rvol_bid_ask_net_asymmetry"),
            include_groups=False
        ).values

        temp_cols = [c for c in df.columns if c.startswith("_")]
        df.drop(columns=temp_cols, inplace=True)

        return df
