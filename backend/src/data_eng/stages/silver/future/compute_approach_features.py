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
POINT = 1.0
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


def _vectorized_bucket_id(ts_ns: np.ndarray) -> np.ndarray:
    ts_series = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("America/New_York")
    minutes_since_open = (ts_series.hour - RTH_START_HOUR) * 60 + ts_series.minute - RTH_START_MINUTE
    bucket_ids = minutes_since_open // BUCKET_MINUTES
    return np.clip(bucket_ids.values, 0, N_BUCKETS - 1).astype(np.int32)


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

        new_cols: Dict[str, np.ndarray] = {}
        self._compute_position_features(df, level_type, new_cols)
        df["bar5s_approach_dist_to_level_pts_eob"] = new_cols["bar5s_approach_dist_to_level_pts_eob"]
        df["bar5s_approach_side_of_level_eob"] = new_cols["bar5s_approach_side_of_level_eob"]

        self._compute_spatial_lookahead_features(df, new_cols)
        self._compute_level_centric_bands(df, new_cols)
        self._compute_cumulative_features(df, new_cols)
        self._compute_derivative_features(df, new_cols)
        self._compute_level_relative_book_features(df, new_cols)
        self._compute_setup_signature_features(df, new_cols)
        self._compute_relative_volume_features(df, df_profile, new_cols)

        cols_to_add = {k: v for k, v in new_cols.items() if k not in df.columns}
        new_df = pd.DataFrame(cols_to_add, index=df.index)
        return pd.concat([df, new_df], axis=1)

    def _compute_position_features(
        self, df: pd.DataFrame, level_type: str, out: Dict[str, np.ndarray]
    ) -> None:
        n = len(df)
        level_price = df["level_price"].values
        microprice = df["bar5s_microprice_eob"].values

        out["bar5s_approach_dist_to_level_pts_eob"] = (microprice - level_price) / POINT

        bid_px_00 = microprice - df["bar5s_state_spread_pts_eob"].values * POINT / 2
        ask_px_00 = microprice + df["bar5s_state_spread_pts_eob"].values * POINT / 2
        bid_sz_00 = df["bar5s_shape_bid_sz_l00_eob"].values
        ask_sz_00 = df["bar5s_shape_ask_sz_l00_eob"].values

        denom = bid_sz_00 + ask_sz_00 + EPSILON
        micro_twa = np.where(
            (bid_sz_00 + ask_sz_00) > EPSILON,
            (ask_px_00 * bid_sz_00 + bid_px_00 * ask_sz_00) / denom,
            microprice,
        )
        out["bar5s_approach_dist_to_level_pts_twa"] = (micro_twa - level_price) / POINT
        out["bar5s_approach_abs_dist_to_level_pts_eob"] = np.abs(out["bar5s_approach_dist_to_level_pts_eob"])
        out["bar5s_approach_side_of_level_eob"] = np.where(microprice > level_price, 1, -1).astype(np.int8)

        out["bar5s_approach_is_pm_high"] = np.full(n, 1 if level_type == "PM_HIGH" else 0, dtype=np.int8)
        out["bar5s_approach_is_pm_low"] = np.full(n, 1 if level_type == "PM_LOW" else 0, dtype=np.int8)
        out["bar5s_approach_is_or_high"] = np.full(n, 1 if level_type == "OR_HIGH" else 0, dtype=np.int8)
        out["bar5s_approach_is_or_low"] = np.full(n, 1 if level_type == "OR_LOW" else 0, dtype=np.int8)

        level_polarity = 1 if level_type in ["PM_HIGH", "OR_HIGH"] else -1
        out["bar5s_approach_level_polarity"] = np.full(n, level_polarity, dtype=np.int8)
        out["bar5s_approach_alignment_eob"] = -1 * out["bar5s_approach_side_of_level_eob"] * level_polarity

    def _compute_spatial_lookahead_features(self, df: pd.DataFrame, out: Dict[str, np.ndarray]) -> None:
        n = len(df)
        level_price = df["level_price"].values
        
        bid_px_cols = [f"bar5s_shape_bid_px_l{i:02d}_eob" for i in range(10)]
        ask_px_cols = [f"bar5s_shape_ask_px_l{i:02d}_eob" for i in range(10)]
        bid_sz_cols = [f"bar5s_shape_bid_sz_l{i:02d}_eob" for i in range(10)]
        ask_sz_cols = [f"bar5s_shape_ask_sz_l{i:02d}_eob" for i in range(10)]
        
        bid_px = df[bid_px_cols].values
        ask_px = df[ask_px_cols].values
        bid_sz = df[bid_sz_cols].values
        ask_sz = df[ask_sz_cols].values
        
        bid_dist = np.abs(bid_px - level_price[:, np.newaxis])
        ask_dist = np.abs(ask_px - level_price[:, np.newaxis])
        
        bid_matches = bid_dist < EPSILON
        ask_matches = ask_dist < EPSILON
        
        bid_size_at_level = np.where(bid_matches.any(axis=1), (bid_sz * bid_matches).sum(axis=1), 0.0)
        ask_size_at_level = np.where(ask_matches.any(axis=1), (ask_sz * ask_matches).sum(axis=1), 0.0)
        
        level_is_visible = ((bid_matches.any(axis=1)) | (ask_matches.any(axis=1))).astype(np.int8)
        
        level_book_index_bid = np.where(
            bid_matches.any(axis=1),
            np.argmax(bid_matches, axis=1),
            -1
        ).astype(np.float64)
        
        level_book_index_ask = np.where(
            ask_matches.any(axis=1),
            np.argmax(ask_matches, axis=1),
            -1
        ).astype(np.float64)
        
        out["bar5s_lvl_bid_size_at_level_eob"] = bid_size_at_level
        out["bar5s_lvl_ask_size_at_level_eob"] = ask_size_at_level
        out["bar5s_lvl_total_size_at_level_eob"] = bid_size_at_level + ask_size_at_level
        out["bar5s_lvl_level_is_visible"] = level_is_visible
        out["bar5s_lvl_level_book_index_bid"] = level_book_index_bid
        out["bar5s_lvl_level_book_index_ask"] = level_book_index_ask
        
        imbal_denom = bid_size_at_level + ask_size_at_level + EPSILON
        out["bar5s_lvl_size_at_level_imbal_eob"] = (bid_size_at_level - ask_size_at_level) / imbal_denom
        
        bid_wall_z = df["bar5s_wall_bid_maxz_eob"].values
        ask_wall_z = df["bar5s_wall_ask_maxz_eob"].values
        bid_wall_idx = df["bar5s_wall_bid_maxz_levelidx_eob"].values
        ask_wall_idx = df["bar5s_wall_ask_maxz_levelidx_eob"].values
        
        bid_wall_at_level = (bid_wall_z > 2.0) & (np.abs(bid_wall_idx - level_book_index_bid) < 0.5)
        ask_wall_at_level = (ask_wall_z > 2.0) & (np.abs(ask_wall_idx - level_book_index_ask) < 0.5)
        wall_at_level = (bid_wall_at_level | ask_wall_at_level).astype(np.int8)
        
        out["bar5s_lvl_wall_at_level"] = wall_at_level

    def _compute_level_centric_bands(self, df: pd.DataFrame, out: Dict[str, np.ndarray]) -> None:
        n = len(df)
        level_price = df["level_price"].values
        
        bid_px_cols = [f"bar5s_shape_bid_px_l{i:02d}_eob" for i in range(10)]
        ask_px_cols = [f"bar5s_shape_ask_px_l{i:02d}_eob" for i in range(10)]
        bid_sz_cols = [f"bar5s_shape_bid_sz_l{i:02d}_eob" for i in range(10)]
        ask_sz_cols = [f"bar5s_shape_ask_sz_l{i:02d}_eob" for i in range(10)]
        
        bid_px = df[bid_px_cols].values
        ask_px = df[ask_px_cols].values
        bid_sz = df[bid_sz_cols].values
        ask_sz = df[ask_sz_cols].values
        
        bid_dist = np.abs(bid_px - level_price[:, np.newaxis]) / POINT
        ask_dist = np.abs(ask_px - level_price[:, np.newaxis]) / POINT
        
        bid_valid = bid_px > EPSILON
        ask_valid = ask_px > EPSILON
        
        bid_mask_0to1 = (bid_dist <= 1.0) & bid_valid
        bid_mask_1to2 = (bid_dist > 1.0) & (bid_dist <= 2.0) & bid_valid
        bid_mask_beyond2 = (bid_dist > 2.0) & bid_valid
        
        ask_mask_0to1 = (ask_dist <= 1.0) & ask_valid
        ask_mask_1to2 = (ask_dist > 1.0) & (ask_dist <= 2.0) & ask_valid
        ask_mask_beyond2 = (ask_dist > 2.0) & ask_valid
        
        band_0to1 = (bid_sz * bid_mask_0to1).sum(axis=1) + (ask_sz * ask_mask_0to1).sum(axis=1)
        band_1to2 = (bid_sz * bid_mask_1to2).sum(axis=1) + (ask_sz * ask_mask_1to2).sum(axis=1)
        band_beyond2 = (bid_sz * bid_mask_beyond2).sum(axis=1) + (ask_sz * ask_mask_beyond2).sum(axis=1)
        
        out["bar5s_lvl_depth_band_0to1_qty_eob"] = band_0to1
        out["bar5s_lvl_depth_band_1to2_qty_eob"] = band_1to2
        out["bar5s_lvl_depth_band_beyond2_qty_eob"] = band_beyond2
        
        total_banded = band_0to1 + band_1to2 + band_beyond2 + EPSILON
        out["bar5s_lvl_depth_band_0to1_frac_eob"] = band_0to1 / total_banded
        out["bar5s_lvl_depth_band_1to2_frac_eob"] = band_1to2 / total_banded
        out["bar5s_lvl_depth_band_beyond2_frac_eob"] = band_beyond2 / total_banded

    def _compute_cumulative_features(self, df: pd.DataFrame, out: Dict[str, np.ndarray]) -> None:
        touch_id = df["touch_id"].values
        unique_ids, inverse = np.unique(touch_id, return_inverse=True)
        n_groups = len(unique_ids)
        n = len(df)

        group_starts = np.zeros(n_groups + 1, dtype=np.int64)
        for i in range(n):
            group_starts[inverse[i] + 1] += 1
        group_starts = np.cumsum(group_starts)

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

        g = df.groupby("touch_id", sort=False)
        for out_col, in_col in cumsum_mappings.items():
            if in_col in df.columns:
                out[out_col] = g[in_col].cumsum().values

        bars_elapsed = g.cumcount().values + 1
        out["bar5s_cumul_signed_trade_vol_rate"] = out.get("bar5s_cumul_signed_trade_vol", np.zeros(n)) / bars_elapsed

        flow_bid_total = np.zeros(n, dtype=np.float64)
        flow_ask_total = np.zeros(n, dtype=np.float64)

        for band in FLOW_BANDS:
            bid_col = f"bar5s_flow_net_vol_bid_{band}_sum"
            ask_col = f"bar5s_flow_net_vol_ask_{band}_sum"

            if bid_col in df.columns:
                cumul = g[bid_col].cumsum().values
                out[f"bar5s_cumul_flow_net_bid_{band}"] = cumul
                flow_bid_total += cumul

            if ask_col in df.columns:
                cumul = g[ask_col].cumsum().values
                out[f"bar5s_cumul_flow_net_ask_{band}"] = cumul
                flow_ask_total += cumul

        out["bar5s_cumul_flow_net_bid"] = flow_bid_total
        out["bar5s_cumul_flow_net_ask"] = flow_ask_total
        out["bar5s_cumul_flow_imbal"] = flow_bid_total - flow_ask_total
        out["bar5s_cumul_flow_imbal_rate"] = out["bar5s_cumul_flow_imbal"] / bars_elapsed

    def _compute_derivative_features(self, df: pd.DataFrame, out: Dict[str, np.ndarray]) -> None:
        g = df.groupby("touch_id", sort=False)

        for short_name, full_col in DERIV_BASE_FEATURES.items():
            if full_col not in df.columns:
                continue

            base_vals = df[full_col].values if full_col in df.columns else out.get(full_col)
            if base_vals is None:
                continue

            for window in DERIV_WINDOWS:
                d1_col = f"bar5s_deriv_{short_name}_d1_w{window}"
                d2_col = f"bar5s_deriv_{short_name}_d2_w{window}"

                d1_vals = g[full_col].transform(lambda x: (x - x.shift(window)) / window).values
                out[d1_col] = d1_vals

                df_temp = df[["touch_id"]].copy()
                df_temp["_d1"] = d1_vals
                g_temp = df_temp.groupby("touch_id", sort=False)
                out[d2_col] = g_temp["_d1"].transform(lambda x: (x - x.shift(window)) / window).values
        
        if "bar5s_lvl_total_size_at_level_eob" in out:
            df_size = df[["touch_id"]].copy()
            df_size["_size"] = out["bar5s_lvl_total_size_at_level_eob"]
            g_size = df_size.groupby("touch_id", sort=False)
            
            out["bar5s_lvl_size_at_level_d1_w3"] = g_size["_size"].transform(lambda x: (x - x.shift(3)) / 3).values
            out["bar5s_lvl_size_at_level_d1_w12"] = g_size["_size"].transform(lambda x: (x - x.shift(12)) / 12).values
            
            df_size["_d1"] = out["bar5s_lvl_size_at_level_d1_w12"]
            g_size2 = df_size.groupby("touch_id", sort=False)
            out["bar5s_lvl_size_at_level_d2_w12"] = g_size2["_d1"].transform(lambda x: (x - x.shift(12)) / 12).values

    def _compute_level_relative_book_features(self, df: pd.DataFrame, out: Dict[str, np.ndarray]) -> None:
        n = len(df)
        level_price = df["level_price"].values
        microprice = df["bar5s_microprice_eob"].values
        spread = df["bar5s_state_spread_pts_eob"].values * POINT

        bid_px_00 = microprice - spread / 2
        ask_px_00 = microprice + spread / 2

        out["bar5s_lvl_depth_above_qty_eob"] = np.where(
            ask_px_00 > level_price,
            df["bar5s_depth_ask10_qty_eob"].values,
            0,
        )
        out["bar5s_lvl_depth_below_qty_eob"] = np.where(
            bid_px_00 < level_price,
            df["bar5s_depth_bid10_qty_eob"].values,
            0,
        )

        dist_from_level = np.abs(microprice - level_price)
        out["bar5s_lvl_depth_at_qty_eob"] = np.where(
            dist_from_level <= 0.5 * POINT,
            df["bar5s_shape_bid_sz_l00_eob"].values + df["bar5s_shape_ask_sz_l00_eob"].values,
            0,
        )

        above = out["bar5s_lvl_depth_above_qty_eob"]
        below = out["bar5s_lvl_depth_below_qty_eob"]
        out["bar5s_lvl_depth_imbal_eob"] = (below - above) / (below + above + EPSILON)

        for band in ["p0_1", "p1_2", "p2_3"]:
            below_col = f"bar5s_depth_below_{band}_qty_eob"
            above_col = f"bar5s_depth_above_{band}_qty_eob"

            if below_col in df.columns and above_col in df.columns:
                b = df[below_col].values
                a = df[above_col].values
                out[f"bar5s_lvl_depth_above_{band}_qty_eob"] = a
                out[f"bar5s_lvl_depth_below_{band}_qty_eob"] = b
                out[f"bar5s_lvl_cdi_{band}_eob"] = (b - a) / (b + a + EPSILON)

        side_of_level = out["bar5s_approach_side_of_level_eob"]

        flow_bid_sum = np.zeros(n, dtype=np.float64)
        flow_ask_sum = np.zeros(n, dtype=np.float64)

        for band in FLOW_BANDS:
            bid_col = f"bar5s_flow_net_vol_bid_{band}_sum"
            ask_col = f"bar5s_flow_net_vol_ask_{band}_sum"
            if bid_col in df.columns:
                flow_bid_sum += df[bid_col].values
            if ask_col in df.columns:
                flow_ask_sum += df[ask_col].values

        toward_flow = np.where(side_of_level < 0, flow_ask_sum, flow_bid_sum)
        away_flow = np.where(side_of_level < 0, flow_bid_sum, flow_ask_sum)

        out["bar5s_lvl_flow_toward_net_sum"] = toward_flow
        out["bar5s_lvl_flow_away_net_sum"] = away_flow
        out["bar5s_lvl_flow_toward_away_imbal_sum"] = toward_flow - away_flow

    def _compute_setup_signature_features(self, df: pd.DataFrame, out: Dict[str, np.ndarray]) -> None:
        n = len(df)
        g = df.groupby("touch_id", sort=False)
        dist_col = "bar5s_approach_dist_to_level_pts_eob"

        dist_vals = out.get(dist_col, df[dist_col].values if dist_col in df.columns else np.zeros(n))
        abs_dist = np.abs(dist_vals)

        out["bar5s_setup_start_dist_pts"] = g[dist_col].transform("first").values

        df_temp = df[["touch_id"]].copy()
        df_temp["_abs_dist"] = abs_dist
        g_temp = df_temp.groupby("touch_id", sort=False)

        out["bar5s_setup_min_dist_pts"] = g_temp["_abs_dist"].transform("min").values
        out["bar5s_setup_max_dist_pts"] = g_temp["_abs_dist"].transform("max").values
        out["bar5s_setup_dist_range_pts"] = out["bar5s_setup_max_dist_pts"] - out["bar5s_setup_min_dist_pts"]

        delta_dist = g_temp["_abs_dist"].diff().fillna(0.0).values
        df_temp["_delta_dist"] = delta_dist
        g_temp = df_temp.groupby("touch_id", sort=False)

        out["bar5s_setup_approach_bars"] = g_temp["_delta_dist"].transform(lambda x: (x < 0).sum()).values.astype(np.float64)
        out["bar5s_setup_retreat_bars"] = g_temp["_delta_dist"].transform(lambda x: (x > 0).sum()).values.astype(np.float64)
        n_bars = g_temp["_delta_dist"].transform("count").values.astype(np.float64)
        out["bar5s_setup_approach_ratio"] = out["bar5s_setup_approach_bars"] / (n_bars + EPSILON)

        d1_col = "bar5s_deriv_dist_d1_w3"
        if d1_col in out:
            d1_vals = out[d1_col]
            df_vel = df[["touch_id"]].copy()
            df_vel["_d1"] = d1_vals

            vel_stats = df_vel.groupby("touch_id", sort=False).apply(
                self._compute_velocity_stats, include_groups=False
            )

            if isinstance(vel_stats, pd.DataFrame):
                vel_stats = vel_stats.reset_index()
            else:
                vel_stats = vel_stats.reset_index(name="stats")
                vel_stats = pd.concat([vel_stats["touch_id"], vel_stats["stats"].apply(pd.Series)], axis=1)

            df_vel = df_vel.merge(vel_stats, on="touch_id", how="left")
            out["bar5s_setup_early_velocity"] = df_vel["early"].values.astype(np.float64)
            out["bar5s_setup_mid_velocity"] = df_vel["mid"].values.astype(np.float64)
            out["bar5s_setup_late_velocity"] = df_vel["late"].values.astype(np.float64)
            out["bar5s_setup_velocity_trend"] = df_vel["trend"].values.astype(np.float64)
        else:
            out["bar5s_setup_early_velocity"] = np.zeros(n, dtype=np.float64)
            out["bar5s_setup_mid_velocity"] = np.zeros(n, dtype=np.float64)
            out["bar5s_setup_late_velocity"] = np.zeros(n, dtype=np.float64)
            out["bar5s_setup_velocity_trend"] = np.zeros(n, dtype=np.float64)

        for metric, col in [("obi0", "bar5s_state_obi0_eob"), ("obi10", "bar5s_state_obi10_eob")]:
            if col in df.columns:
                out[f"bar5s_setup_{metric}_start"] = g[col].transform("first").values
                out[f"bar5s_setup_{metric}_end"] = g[col].transform("last").values
                out[f"bar5s_setup_{metric}_delta"] = out[f"bar5s_setup_{metric}_end"] - out[f"bar5s_setup_{metric}_start"]
                out[f"bar5s_setup_{metric}_min"] = g[col].transform("min").values
                out[f"bar5s_setup_{metric}_max"] = g[col].transform("max").values

        out["bar5s_setup_total_trade_vol"] = g["bar5s_trade_vol_sum"].transform("sum").values
        out["bar5s_setup_total_signed_vol"] = g["bar5s_trade_signed_vol_sum"].transform("sum").values
        out["bar5s_setup_trade_imbal_pct"] = out["bar5s_setup_total_signed_vol"] / (out["bar5s_setup_total_trade_vol"] + EPSILON)

        cumul_imbal = out.get("bar5s_cumul_flow_imbal", np.zeros(n))
        df_fi = df[["touch_id"]].copy()
        df_fi["_imbal"] = cumul_imbal
        out["bar5s_setup_flow_imbal_total"] = df_fi.groupby("touch_id", sort=False)["_imbal"].transform("last").values

        out["bar5s_setup_bid_wall_max_z"] = g["bar5s_wall_bid_maxz_eob"].transform("max").values
        out["bar5s_setup_ask_wall_max_z"] = g["bar5s_wall_ask_maxz_eob"].transform("max").values

        bid_wall_strong = (df["bar5s_wall_bid_maxz_eob"].values > 2.0).astype(np.int32)
        ask_wall_strong = (df["bar5s_wall_ask_maxz_eob"].values > 2.0).astype(np.int32)

        df_wall = df[["touch_id"]].copy()
        df_wall["_bid"] = bid_wall_strong
        df_wall["_ask"] = ask_wall_strong
        g_wall = df_wall.groupby("touch_id", sort=False)
        out["bar5s_setup_bid_wall_bars"] = g_wall["_bid"].transform("sum").values.astype(np.float64)
        out["bar5s_setup_ask_wall_bars"] = g_wall["_ask"].transform("sum").values.astype(np.float64)
        out["bar5s_setup_wall_imbal"] = out["bar5s_setup_ask_wall_bars"] - out["bar5s_setup_bid_wall_bars"]
        
        if "bar5s_lvl_total_size_at_level_eob" in out:
            df_lvl = df[["touch_id"]].copy()
            df_lvl["_size"] = out["bar5s_lvl_total_size_at_level_eob"]
            g_lvl = df_lvl.groupby("touch_id", sort=False)
            
            out["bar5s_setup_size_at_level_start"] = g_lvl["_size"].transform("first").values
            out["bar5s_setup_size_at_level_end"] = g_lvl["_size"].transform("last").values
            out["bar5s_setup_size_at_level_delta"] = out["bar5s_setup_size_at_level_end"] - out["bar5s_setup_size_at_level_start"]
            out["bar5s_setup_size_at_level_max"] = g_lvl["_size"].transform("max").values
            
            recent_12 = df_lvl.groupby("touch_id", sort=False).apply(
                lambda x: x["_size"].iloc[-12:].sum() if len(x) >= 12 else x["_size"].sum(),
                include_groups=False
            )
            df_lvl = df_lvl.merge(recent_12.reset_index(name="_recent12"), on="touch_id", how="left")
            out["bar5s_setup_size_at_level_recent12_sum"] = df_lvl["_recent12"].values.astype(np.float64)
            
            early_late_ratio = df_lvl.groupby("touch_id", sort=False).apply(
                self._compute_early_late_ratio, include_groups=False
            )
            if isinstance(early_late_ratio, pd.DataFrame):
                early_late_ratio = early_late_ratio.reset_index()
            else:
                early_late_ratio = early_late_ratio.reset_index(name="ratio")
            df_lvl = df_lvl.merge(early_late_ratio[["touch_id", "ratio"]], on="touch_id", how="left")
            out["bar5s_setup_size_at_level_early_late_ratio"] = df_lvl["ratio"].values.astype(np.float64)
        
        toward_col = "bar5s_lvl_flow_toward_net_sum"
        if toward_col in out:
            df_toward = df[["touch_id"]].copy()
            df_toward[toward_col] = out[toward_col]
            g_toward = df_toward.groupby("touch_id", sort=False)
            
            recent_12_toward = g_toward.apply(
                lambda x: x[toward_col].iloc[-12:].sum() if len(x) >= 12 else x[toward_col].sum(),
                include_groups=False
            )
            df_toward = df_toward.merge(recent_12_toward.reset_index(name="_recent12"), on="touch_id", how="left")
            out["bar5s_setup_flow_toward_recent12_sum"] = df_toward["_recent12"].values.astype(np.float64)
            
            early_late_flow = g_toward.apply(self._compute_early_late_ratio_from_col, toward_col, include_groups=False)
            if isinstance(early_late_flow, pd.DataFrame):
                early_late_flow = early_late_flow.reset_index()
            else:
                early_late_flow = early_late_flow.reset_index(name="ratio")
            df_toward = df_toward.merge(early_late_flow[["touch_id", "ratio"]], on="touch_id", how="left")
            out["bar5s_setup_flow_toward_early_late_ratio"] = df_toward["ratio"].values.astype(np.float64)
        else:
            out["bar5s_setup_flow_toward_recent12_sum"] = np.zeros(n, dtype=np.float64)
            out["bar5s_setup_flow_toward_early_late_ratio"] = np.zeros(n, dtype=np.float64)

    def _compute_early_late_ratio(self, grp: pd.DataFrame) -> pd.Series:
        vals = grp["_size"].values
        n_vals = len(vals)
        if n_vals < 3:
            return pd.Series({"ratio": 0.0})
        third = max(1, n_vals // 3)
        early_sum = np.sum(vals[:third])
        late_sum = np.sum(vals[-third:])
        ratio = late_sum / (early_sum + EPSILON) if early_sum > 0 else 0.0
        return pd.Series({"ratio": ratio})
    
    def _compute_early_late_ratio_from_col(self, grp: pd.DataFrame, col: str) -> pd.Series:
        vals = grp[col].values
        n_vals = len(vals)
        if n_vals < 3:
            return pd.Series({"ratio": 0.0})
        third = max(1, n_vals // 3)
        early_sum = np.sum(vals[:third])
        late_sum = np.sum(vals[-third:])
        ratio = late_sum / (early_sum + EPSILON) if early_sum > 0 else 0.0
        return pd.Series({"ratio": ratio})

    def _compute_velocity_stats(self, grp: pd.DataFrame) -> pd.Series:
        d1 = grp["_d1"].values
        n_vals = len(d1)
        third = max(1, n_vals // 3)
        early = np.nanmean(d1[:third]) if third > 0 else 0.0
        mid = np.nanmean(d1[third:2*third]) if 2*third > third else 0.0
        late = np.nanmean(d1[2*third:]) if n_vals > 2*third else 0.0
        return pd.Series({"early": early, "mid": mid, "late": late, "trend": late - early})

    def _compute_relative_volume_features(
        self, df: pd.DataFrame, df_profile: Optional[pd.DataFrame], out: Dict[str, np.ndarray]
    ) -> None:
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
                out[col] = np.full(n, np.nan, dtype=np.float64)
            return

        bucket_ids = _vectorized_bucket_id(df["bar_ts"].values)

        profile_lookup = {}
        for _, row in df_profile.iterrows():
            bid = int(row["bucket_id"])
            profile_lookup[bid] = {
                "trade_vol_mean": row.get("trade_vol_mean", 0.0),
                "trade_vol_std": row.get("trade_vol_std", 0.0),
                "trade_cnt_mean": row.get("trade_cnt_mean", 0.0),
                "trade_cnt_std": row.get("trade_cnt_std", 0.0),
                "trade_aggbuy_vol_mean": row.get("trade_aggbuy_vol_mean", 0.0),
                "trade_aggbuy_vol_std": row.get("trade_aggbuy_vol_std", 0.0),
                "trade_aggsell_vol_mean": row.get("trade_aggsell_vol_mean", 0.0),
                "trade_aggsell_vol_std": row.get("trade_aggsell_vol_std", 0.0),
                "flow_add_vol_bid_mean": row.get("flow_add_vol_bid_mean", 0.0),
                "flow_add_vol_bid_std": row.get("flow_add_vol_bid_std", 0.0),
                "flow_add_vol_ask_mean": row.get("flow_add_vol_ask_mean", 0.0),
                "flow_add_vol_ask_std": row.get("flow_add_vol_ask_std", 0.0),
                "flow_net_vol_bid_mean": row.get("flow_net_vol_bid_mean", 0.0),
                "flow_net_vol_bid_std": row.get("flow_net_vol_bid_std", 0.0),
                "flow_net_vol_ask_mean": row.get("flow_net_vol_ask_mean", 0.0),
                "flow_net_vol_ask_std": row.get("flow_net_vol_ask_std", 0.0),
                "flow_rem_vol_bid_mean": row.get("flow_rem_vol_bid_mean", 0.0),
                "flow_rem_vol_bid_std": row.get("flow_rem_vol_bid_std", 0.0),
                "flow_rem_vol_ask_mean": row.get("flow_rem_vol_ask_mean", 0.0),
                "flow_rem_vol_ask_std": row.get("flow_rem_vol_ask_std", 0.0),
                "msg_cnt_mean": row.get("msg_cnt_mean", 0.0),
                "msg_cnt_std": row.get("msg_cnt_std", 0.0),
            }

        trade_vol = df["bar5s_trade_vol_sum"].values
        trade_cnt = df["bar5s_trade_cnt_sum"].values
        trade_aggbuy = df["bar5s_trade_aggbuy_vol_sum"].values
        trade_aggsell = df["bar5s_trade_aggsell_vol_sum"].values
        msg_cnt = df["bar5s_meta_msg_cnt_sum"].values

        flow_add_bid = np.zeros(n, dtype=np.float64)
        flow_add_ask = np.zeros(n, dtype=np.float64)
        flow_net_bid = np.zeros(n, dtype=np.float64)
        flow_net_ask = np.zeros(n, dtype=np.float64)
        flow_rem_bid = np.zeros(n, dtype=np.float64)
        flow_rem_ask = np.zeros(n, dtype=np.float64)

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

        tv_mean = np.zeros(n, dtype=np.float64)
        tv_std = np.zeros(n, dtype=np.float64)
        tc_mean = np.zeros(n, dtype=np.float64)
        tc_std = np.zeros(n, dtype=np.float64)
        ab_mean = np.zeros(n, dtype=np.float64)
        ab_std = np.zeros(n, dtype=np.float64)
        as_mean = np.zeros(n, dtype=np.float64)
        as_std = np.zeros(n, dtype=np.float64)
        fab_mean = np.zeros(n, dtype=np.float64)
        fab_std = np.zeros(n, dtype=np.float64)
        faa_mean = np.zeros(n, dtype=np.float64)
        faa_std = np.zeros(n, dtype=np.float64)
        fnb_mean = np.zeros(n, dtype=np.float64)
        fnb_std = np.zeros(n, dtype=np.float64)
        fna_mean = np.zeros(n, dtype=np.float64)
        fna_std = np.zeros(n, dtype=np.float64)
        frb_mean = np.zeros(n, dtype=np.float64)
        frb_std = np.zeros(n, dtype=np.float64)
        fra_mean = np.zeros(n, dtype=np.float64)
        fra_std = np.zeros(n, dtype=np.float64)
        msg_mean = np.zeros(n, dtype=np.float64)

        sqrt_bars = np.sqrt(BARS_PER_BUCKET)
        for i in range(n):
            bid = int(bucket_ids[i])
            p = profile_lookup.get(bid, {})
            tv_mean[i] = p.get("trade_vol_mean", 0.0) / BARS_PER_BUCKET
            tv_std[i] = p.get("trade_vol_std", 0.0) / sqrt_bars
            tc_mean[i] = p.get("trade_cnt_mean", 0.0) / BARS_PER_BUCKET
            tc_std[i] = p.get("trade_cnt_std", 0.0) / sqrt_bars
            ab_mean[i] = p.get("trade_aggbuy_vol_mean", 0.0) / BARS_PER_BUCKET
            ab_std[i] = p.get("trade_aggbuy_vol_std", 0.0) / sqrt_bars
            as_mean[i] = p.get("trade_aggsell_vol_mean", 0.0) / BARS_PER_BUCKET
            as_std[i] = p.get("trade_aggsell_vol_std", 0.0) / sqrt_bars
            fab_mean[i] = p.get("flow_add_vol_bid_mean", 0.0) / BARS_PER_BUCKET
            fab_std[i] = p.get("flow_add_vol_bid_std", 0.0) / sqrt_bars
            faa_mean[i] = p.get("flow_add_vol_ask_mean", 0.0) / BARS_PER_BUCKET
            faa_std[i] = p.get("flow_add_vol_ask_std", 0.0) / sqrt_bars
            fnb_mean[i] = p.get("flow_net_vol_bid_mean", 0.0) / BARS_PER_BUCKET
            fnb_std[i] = p.get("flow_net_vol_bid_std", 0.0) / sqrt_bars
            fna_mean[i] = p.get("flow_net_vol_ask_mean", 0.0) / BARS_PER_BUCKET
            fna_std[i] = p.get("flow_net_vol_ask_std", 0.0) / sqrt_bars
            frb_mean[i] = p.get("flow_rem_vol_bid_mean", 0.0) / BARS_PER_BUCKET
            frb_std[i] = p.get("flow_rem_vol_bid_std", 0.0) / sqrt_bars
            fra_mean[i] = p.get("flow_rem_vol_ask_mean", 0.0) / BARS_PER_BUCKET
            fra_std[i] = p.get("flow_rem_vol_ask_std", 0.0) / sqrt_bars
            msg_mean[i] = p.get("msg_cnt_mean", 0.0) / BARS_PER_BUCKET

        out["rvol_trade_vol_ratio"] = trade_vol / (tv_mean + EPSILON)
        out["rvol_trade_vol_zscore"] = (trade_vol - tv_mean) / (tv_std + EPSILON)
        out["rvol_trade_cnt_ratio"] = trade_cnt / (tc_mean + EPSILON)
        out["rvol_trade_cnt_zscore"] = (trade_cnt - tc_mean) / (tc_std + EPSILON)
        out["rvol_trade_aggbuy_ratio"] = trade_aggbuy / (ab_mean + EPSILON)
        out["rvol_trade_aggsell_ratio"] = trade_aggsell / (as_mean + EPSILON)
        out["rvol_trade_aggbuy_zscore"] = (trade_aggbuy - ab_mean) / (ab_std + EPSILON)
        out["rvol_trade_aggsell_zscore"] = (trade_aggsell - as_mean) / (as_std + EPSILON)
        out["rvol_flow_add_bid_ratio"] = flow_add_bid / (fab_mean + EPSILON)
        out["rvol_flow_add_ask_ratio"] = flow_add_ask / (faa_mean + EPSILON)
        out["rvol_flow_add_bid_zscore"] = (flow_add_bid - fab_mean) / (fab_std + EPSILON)
        out["rvol_flow_add_ask_zscore"] = (flow_add_ask - faa_mean) / (faa_std + EPSILON)
        out["rvol_flow_net_bid_ratio"] = np.where(fnb_mean != 0, flow_net_bid / (fnb_mean + EPSILON), 1.0)
        out["rvol_flow_net_ask_ratio"] = np.where(fna_mean != 0, flow_net_ask / (fna_mean + EPSILON), 1.0)
        out["rvol_flow_net_bid_zscore"] = (flow_net_bid - fnb_mean) / (fnb_std + EPSILON)
        out["rvol_flow_net_ask_zscore"] = (flow_net_ask - fna_mean) / (fna_std + EPSILON)
        out["rvol_flow_add_total_ratio"] = (flow_add_bid + flow_add_ask) / (fab_mean + faa_mean + EPSILON)
        out["rvol_flow_add_total_zscore"] = ((flow_add_bid + flow_add_ask) - (fab_mean + faa_mean)) / (np.sqrt(fab_std**2 + faa_std**2) + EPSILON)

        rvol_flow_rem_bid_zscore = (flow_rem_bid - frb_mean) / (frb_std + EPSILON)
        rvol_flow_rem_ask_zscore = (flow_rem_ask - fra_mean) / (fra_std + EPSILON)

        out["rvol_bid_ask_add_asymmetry"] = out["rvol_flow_add_bid_zscore"] - out["rvol_flow_add_ask_zscore"]
        out["rvol_bid_ask_rem_asymmetry"] = rvol_flow_rem_bid_zscore - rvol_flow_rem_ask_zscore
        out["rvol_bid_ask_net_asymmetry"] = out["rvol_flow_net_bid_zscore"] - out["rvol_flow_net_ask_zscore"]
        out["rvol_aggbuy_aggsell_asymmetry"] = out["rvol_trade_aggbuy_zscore"] - out["rvol_trade_aggsell_zscore"]

        g = df.groupby("touch_id", sort=False)

        df_cumul = df[["touch_id"]].copy()
        df_cumul["_exp_tv"] = tv_mean
        df_cumul["_act_tv"] = trade_vol
        df_cumul["_exp_msg"] = msg_mean
        df_cumul["_act_msg"] = msg_cnt
        df_cumul["_exp_fi"] = fnb_mean - fna_mean
        df_cumul["_act_fi"] = flow_net_bid - flow_net_ask

        g_cumul = df_cumul.groupby("touch_id", sort=False)

        exp_tv_cumul = g_cumul["_exp_tv"].cumsum().values
        act_tv_cumul = g_cumul["_act_tv"].cumsum().values
        out["rvol_cumul_trade_vol_dev"] = act_tv_cumul - exp_tv_cumul
        out["rvol_cumul_trade_vol_dev_pct"] = out["rvol_cumul_trade_vol_dev"] / (exp_tv_cumul + EPSILON)

        exp_msg_cumul = g_cumul["_exp_msg"].cumsum().values
        act_msg_cumul = g_cumul["_act_msg"].cumsum().values
        out["rvol_cumul_msg_dev"] = act_msg_cumul - exp_msg_cumul

        exp_fi_cumul = g_cumul["_exp_fi"].cumsum().values
        act_fi_cumul = g_cumul["_act_fi"].cumsum().values
        out["rvol_cumul_flow_imbal_dev"] = act_fi_cumul - exp_fi_cumul

        df_rvol = df[["touch_id"]].copy()
        df_rvol["_vol_ratio"] = out["rvol_trade_vol_ratio"]
        df_rvol["_net_asymmetry"] = out["rvol_bid_ask_net_asymmetry"]
        g_rvol = df_rvol.groupby("touch_id", sort=False)

        out["rvol_lookback_trade_vol_mean_ratio"] = g_rvol["_vol_ratio"].transform("mean").values
        out["rvol_lookback_trade_vol_max_ratio"] = g_rvol["_vol_ratio"].transform("max").values

        trend_stats = g_rvol.apply(self._compute_trend_stats, include_groups=False)
        if isinstance(trend_stats, pd.DataFrame):
            trend_stats = trend_stats.reset_index()
        else:
            trend_stats = trend_stats.reset_index(name="trend")
        df_rvol = df_rvol.merge(trend_stats[["touch_id", "trend"]], on="touch_id", how="left")
        out["rvol_lookback_trade_vol_trend"] = df_rvol["trend"].values.astype(np.float64)

        elevated = (out["rvol_trade_vol_ratio"] > 1.5).astype(np.float64)
        depressed = (out["rvol_trade_vol_ratio"] < 0.5).astype(np.float64)
        df_elev = df[["touch_id"]].copy()
        df_elev["_elev"] = elevated
        df_elev["_depr"] = depressed
        g_elev = df_elev.groupby("touch_id", sort=False)
        out["rvol_lookback_elevated_bars"] = g_elev["_elev"].transform("sum").values
        out["rvol_lookback_depressed_bars"] = g_elev["_depr"].transform("sum").values

        out["rvol_lookback_asymmetry_mean"] = g_rvol["_net_asymmetry"].transform("mean").values

        recent_vol = self._compute_recent_vs_lookback(df, out["rvol_trade_vol_ratio"])
        recent_asym = self._compute_recent_vs_lookback(df, out["rvol_bid_ask_net_asymmetry"])
        out["rvol_recent_vs_lookback_vol_ratio"] = recent_vol
        out["rvol_recent_vs_lookback_asymmetry"] = recent_asym

    def _compute_trend_stats(self, grp: pd.DataFrame) -> pd.Series:
        vals = grp["_vol_ratio"].values
        n_vals = len(vals)
        if n_vals < 2:
            return pd.Series({"trend": 0.0})
        split = max(1, n_vals // 3)
        early_mean = np.nanmean(vals[:split])
        late_mean = np.nanmean(vals[-split:])
        return pd.Series({"trend": late_mean - early_mean})

    def _compute_recent_vs_lookback(self, df: pd.DataFrame, vals: np.ndarray) -> np.ndarray:
        n = len(df)
        result = np.zeros(n, dtype=np.float64)

        touch_id = df["touch_id"].values
        unique_ids, inverse = np.unique(touch_id, return_inverse=True)

        for grp_idx in range(len(unique_ids)):
            mask = inverse == grp_idx
            grp_vals = vals[mask]
            grp_n = len(grp_vals)
            grp_result = np.zeros(grp_n, dtype=np.float64)

            for i in range(grp_n):
                lookback_mean = np.nanmean(grp_vals[:i+1]) if i > 0 else grp_vals[0]
                recent_start = max(0, i - 11)
                recent_mean = np.nanmean(grp_vals[recent_start:i+1])
                grp_result[i] = recent_mean / (lookback_mean + EPSILON) if lookback_mean != 0 else 1.0

            result[mask] = grp_result

        return result
