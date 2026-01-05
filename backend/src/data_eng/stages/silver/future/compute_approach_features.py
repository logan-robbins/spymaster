from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

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


class SilverComputeApproachFeatures(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_approach_features",
            io=StageIO(
                inputs=[
                    f"silver.future.market_by_price_10_{lt.lower()}_episodes"
                    for lt in LEVEL_TYPES
                ],
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
                    df_out = self._compute_approach_features(df_in, level_type)

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
        self, df: pd.DataFrame, level_type: str
    ) -> pd.DataFrame:
        df = df.copy()

        df = self._compute_position_features(df, level_type)
        df = self._compute_cumulative_features(df)
        df = self._compute_derivative_features(df)
        df = self._compute_level_relative_book_features(df)
        df = self._compute_setup_signature_features(df)

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
