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
LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]
TARGET_DIM = 256
LOOKBACK_BARS = 180
RECENT_BARS = 12

SNAPSHOT_FEATURES = [
    "bar5s_approach_dist_to_level_pts_eob",
    "bar5s_approach_side_of_level_eob",
    "bar5s_approach_alignment_eob",
    "bar5s_approach_level_polarity",
    "is_standard_approach",
    "bar5s_state_obi0_eob",
    "bar5s_state_obi10_eob",
    "bar5s_state_spread_pts_eob",
    "bar5s_state_cdi_p0_1_eob",
    "bar5s_state_cdi_p1_2_eob",
    "bar5s_state_cdi_p2_3_eob",
    "bar5s_lvl_depth_imbal_eob",
    "bar5s_lvl_cdi_p0_1_eob",
    "bar5s_lvl_cdi_p1_2_eob",
    "bar5s_depth_bid10_qty_eob",
    "bar5s_depth_ask10_qty_eob",
    "bar5s_lvl_depth_above_qty_eob",
    "bar5s_lvl_depth_below_qty_eob",
    "bar5s_wall_bid_maxz_eob",
    "bar5s_wall_ask_maxz_eob",
    "bar5s_wall_bid_maxz_levelidx_eob",
    "bar5s_wall_ask_maxz_levelidx_eob",
    "bar5s_wall_bid_nearest_strong_dist_pts_eob",
    "bar5s_wall_ask_nearest_strong_dist_pts_eob",
    "bar5s_wall_bid_nearest_strong_levelidx_eob",
    "bar5s_wall_ask_nearest_strong_levelidx_eob",
    "bar5s_cumul_signed_trade_vol",
    "bar5s_cumul_flow_imbal",
    "bar5s_cumul_flow_net_bid",
    "bar5s_cumul_flow_net_ask",
    "bar5s_lvl_flow_toward_net_sum",
    "bar5s_lvl_flow_away_net_sum",
    "bar5s_lvl_flow_toward_away_imbal_sum",
    "bar5s_trade_signed_vol_sum",
    "bar5s_trade_aggbuy_vol_sum",
    "bar5s_trade_aggsell_vol_sum",
]

DERIVATIVE_FEATURES = [
    "bar5s_deriv_dist_d1_w3",
    "bar5s_deriv_dist_d1_w12",
    "bar5s_deriv_dist_d1_w36",
    "bar5s_deriv_dist_d1_w72",
    "bar5s_deriv_dist_d2_w3",
    "bar5s_deriv_dist_d2_w12",
    "bar5s_deriv_dist_d2_w36",
    "bar5s_deriv_dist_d2_w72",
    "bar5s_deriv_obi0_d1_w12",
    "bar5s_deriv_obi0_d1_w36",
    "bar5s_deriv_obi10_d1_w12",
    "bar5s_deriv_obi10_d1_w36",
    "bar5s_deriv_cdi01_d1_w12",
    "bar5s_deriv_cdi01_d1_w36",
    "bar5s_deriv_cdi12_d1_w12",
    "bar5s_deriv_cdi12_d1_w36",
    "bar5s_deriv_obi0_d2_w12",
    "bar5s_deriv_obi0_d2_w36",
    "bar5s_deriv_obi10_d2_w12",
    "bar5s_deriv_obi10_d2_w36",
    "bar5s_deriv_cdi01_d2_w12",
    "bar5s_deriv_cdi01_d2_w36",
    "bar5s_deriv_cdi12_d2_w12",
    "bar5s_deriv_cdi12_d2_w36",
    "bar5s_deriv_dbid10_d1_w12",
    "bar5s_deriv_dbid10_d1_w36",
    "bar5s_deriv_dask10_d1_w12",
    "bar5s_deriv_dask10_d1_w36",
    "bar5s_deriv_dbelow01_d1_w12",
    "bar5s_deriv_dbelow01_d1_w36",
    "bar5s_deriv_dabove01_d1_w12",
    "bar5s_deriv_dabove01_d1_w36",
    "bar5s_deriv_wbidz_d1_w12",
    "bar5s_deriv_wbidz_d1_w36",
    "bar5s_deriv_waskz_d1_w12",
    "bar5s_deriv_waskz_d1_w36",
    "bar5s_deriv_wbidz_d2_w12",
    "bar5s_deriv_wbidz_d2_w36",
    "bar5s_deriv_waskz_d2_w12",
    "bar5s_deriv_waskz_d2_w36",
]

PROFILE_FEATURES = [
    "bar5s_setup_start_dist_pts",
    "bar5s_setup_min_dist_pts",
    "bar5s_setup_max_dist_pts",
    "bar5s_setup_dist_range_pts",
    "bar5s_setup_approach_bars",
    "bar5s_setup_retreat_bars",
    "bar5s_setup_approach_ratio",
    "bar5s_setup_early_velocity",
    "bar5s_setup_mid_velocity",
    "bar5s_setup_late_velocity",
    "bar5s_setup_velocity_trend",
    "bar5s_setup_obi0_start",
    "bar5s_setup_obi0_end",
    "bar5s_setup_obi0_delta",
    "bar5s_setup_obi0_min",
    "bar5s_setup_obi0_max",
    "bar5s_setup_obi10_start",
    "bar5s_setup_obi10_end",
    "bar5s_setup_obi10_delta",
    "bar5s_setup_obi10_min",
    "bar5s_setup_obi10_max",
    "bar5s_setup_total_trade_vol",
    "bar5s_setup_total_signed_vol",
    "bar5s_setup_trade_imbal_pct",
    "bar5s_setup_flow_imbal_total",
    "bar5s_setup_bid_wall_max_z",
    "bar5s_setup_ask_wall_max_z",
    "bar5s_setup_bid_wall_bars",
    "bar5s_setup_ask_wall_bars",
    "bar5s_setup_wall_imbal",
]

VOLUME_FEATURES = [
    "bar5s_cumul_signed_trade_vol",
    "bar5s_cumul_trade_vol",
    "bar5s_setup_total_trade_vol",
    "bar5s_trade_vol_sum",
    "bar5s_depth_bid10_qty_eob",
    "bar5s_depth_ask10_qty_eob",
    "bar5s_cumul_aggbuy_vol",
    "bar5s_cumul_aggsell_vol",
    "bar5s_cumul_flow_net_bid",
    "bar5s_cumul_flow_net_ask",
]


def compute_recent_momentum(lookback_bars: pd.DataFrame) -> np.ndarray:
    if len(lookback_bars) < RECENT_BARS:
        return np.zeros(12)

    recent = lookback_bars.tail(RECENT_BARS)
    first_row = lookback_bars.iloc[-(RECENT_BARS + 1)] if len(lookback_bars) > RECENT_BARS else lookback_bars.iloc[0]
    last_row = recent.iloc[-1]

    dist_col = "bar5s_approach_dist_to_level_pts_eob"
    dist_delta = last_row.get(dist_col, 0.0) - first_row.get(dist_col, 0.0)

    obi0_delta = last_row.get("bar5s_state_obi0_eob", 0.0) - first_row.get("bar5s_state_obi0_eob", 0.0)
    obi10_delta = last_row.get("bar5s_state_obi10_eob", 0.0) - first_row.get("bar5s_state_obi10_eob", 0.0)
    cdi01_delta = last_row.get("bar5s_state_cdi_p0_1_eob", 0.0) - first_row.get("bar5s_state_cdi_p0_1_eob", 0.0)

    trade_vol = recent["bar5s_trade_vol_sum"].sum() if "bar5s_trade_vol_sum" in recent.columns else 0.0
    signed_vol = recent["bar5s_trade_signed_vol_sum"].sum() if "bar5s_trade_signed_vol_sum" in recent.columns else 0.0
    flow_toward = recent["bar5s_lvl_flow_toward_net_sum"].sum() if "bar5s_lvl_flow_toward_net_sum" in recent.columns else 0.0
    flow_away = recent["bar5s_lvl_flow_away_net_sum"].sum() if "bar5s_lvl_flow_away_net_sum" in recent.columns else 0.0
    aggbuy_vol = recent["bar5s_trade_aggbuy_vol_sum"].sum() if "bar5s_trade_aggbuy_vol_sum" in recent.columns else 0.0
    aggsell_vol = recent["bar5s_trade_aggsell_vol_sum"].sum() if "bar5s_trade_aggsell_vol_sum" in recent.columns else 0.0

    bid_depth_delta = last_row.get("bar5s_depth_bid10_qty_eob", 0.0) - first_row.get("bar5s_depth_bid10_qty_eob", 0.0)
    ask_depth_delta = last_row.get("bar5s_depth_ask10_qty_eob", 0.0) - first_row.get("bar5s_depth_ask10_qty_eob", 0.0)

    return np.array([
        dist_delta, obi0_delta, obi10_delta, cdi01_delta,
        trade_vol, signed_vol, flow_toward, flow_away,
        aggbuy_vol, aggsell_vol, bid_depth_delta, ask_depth_delta
    ], dtype=np.float64)


def compute_additional_profile_features(lookback_bars: pd.DataFrame) -> np.ndarray:
    if len(lookback_bars) == 0:
        return np.zeros(30)

    features = []

    vel_col = "bar5s_deriv_dist_d1_w3"
    if vel_col in lookback_bars.columns:
        vel_vals = lookback_bars[vel_col].dropna().values
        features.append(np.std(vel_vals) if len(vel_vals) > 0 else 0.0)
    else:
        features.append(0.0)

    obi0_col = "bar5s_state_obi0_eob"
    if obi0_col in lookback_bars.columns:
        features.append(lookback_bars[obi0_col].mean())
        features.append(lookback_bars[obi0_col].std())
    else:
        features.extend([0.0, 0.0])

    obi10_col = "bar5s_state_obi10_eob"
    if obi10_col in lookback_bars.columns:
        features.append(lookback_bars[obi10_col].mean())
        features.append(lookback_bars[obi10_col].std())
    else:
        features.extend([0.0, 0.0])

    cdi01_col = "bar5s_state_cdi_p0_1_eob"
    if cdi01_col in lookback_bars.columns:
        features.append(lookback_bars[cdi01_col].mean())
        features.append(lookback_bars[cdi01_col].std())
    else:
        features.extend([0.0, 0.0])

    lvl_imbal_col = "bar5s_lvl_depth_imbal_eob"
    if lvl_imbal_col in lookback_bars.columns:
        features.append(lookback_bars[lvl_imbal_col].mean())
        features.append(lookback_bars[lvl_imbal_col].std())
        features.append(lookback_bars[lvl_imbal_col].iloc[-1] - lookback_bars[lvl_imbal_col].iloc[0])
    else:
        features.extend([0.0, 0.0, 0.0])

    spread_col = "bar5s_state_spread_pts_eob"
    if spread_col in lookback_bars.columns:
        features.append(lookback_bars[spread_col].mean())
    else:
        features.append(0.0)

    toward_col = "bar5s_lvl_flow_toward_net_sum"
    away_col = "bar5s_lvl_flow_away_net_sum"
    if toward_col in lookback_bars.columns and away_col in lookback_bars.columns:
        toward_total = lookback_bars[toward_col].sum()
        away_total = lookback_bars[away_col].sum()
        features.append(toward_total)
        features.append(away_total)
        features.append(toward_total / (away_total + EPSILON) if away_total != 0 else 0.0)
    else:
        features.extend([0.0, 0.0, 0.0])

    n_bars = len(lookback_bars)
    third = max(1, n_bars // 3)
    trade_vol_col = "bar5s_trade_vol_sum"
    if trade_vol_col in lookback_bars.columns:
        early = lookback_bars[trade_vol_col].iloc[:third].sum()
        mid = lookback_bars[trade_vol_col].iloc[third:2*third].sum()
        late = lookback_bars[trade_vol_col].iloc[2*third:].sum()
        features.extend([early, mid, late, late - early])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    signed_vol_col = "bar5s_trade_signed_vol_sum"
    if signed_vol_col in lookback_bars.columns:
        early_s = lookback_bars[signed_vol_col].iloc[:third].sum()
        mid_s = lookback_bars[signed_vol_col].iloc[third:2*third].sum()
        late_s = lookback_bars[signed_vol_col].iloc[2*third:].sum()
        features.extend([early_s, mid_s, late_s])
    else:
        features.extend([0.0, 0.0, 0.0])

    bid_wall_col = "bar5s_wall_bid_maxz_eob"
    ask_wall_col = "bar5s_wall_ask_maxz_eob"
    if bid_wall_col in lookback_bars.columns and ask_wall_col in lookback_bars.columns:
        features.append(lookback_bars[bid_wall_col].mean())
        features.append(lookback_bars[ask_wall_col].mean())

        bid_dist_col = "bar5s_wall_bid_nearest_strong_dist_pts_eob"
        ask_dist_col = "bar5s_wall_ask_nearest_strong_dist_pts_eob"
        if bid_dist_col in lookback_bars.columns and ask_dist_col in lookback_bars.columns:
            features.append(lookback_bars[bid_dist_col].min())
            features.append(lookback_bars[ask_dist_col].min())
        else:
            features.extend([10.0, 10.0])

        features.append(1 if (lookback_bars[bid_wall_col].iloc[-3:] > 2.0).any() and (lookback_bars[bid_wall_col].iloc[:3] <= 2.0).all() else 0)
        features.append(1 if (lookback_bars[ask_wall_col].iloc[-3:] > 2.0).any() and (lookback_bars[ask_wall_col].iloc[:3] <= 2.0).all() else 0)
        features.append(1 if (lookback_bars[bid_wall_col].iloc[:3] > 2.0).any() and (lookback_bars[bid_wall_col].iloc[-3:] <= 2.0).all() else 0)
    else:
        features.extend([0.0, 0.0, 10.0, 10.0, 0, 0, 0])

    return np.array(features, dtype=np.float64)


def extract_setup_vector(trigger_bar: pd.Series, lookback_bars: pd.DataFrame) -> np.ndarray:
    components = []

    snapshot_vals = []
    for f in SNAPSHOT_FEATURES:
        val = trigger_bar.get(f, 0.0)
        if pd.isna(val):
            val = 0.0
        snapshot_vals.append(float(val))
    components.append(np.array(snapshot_vals, dtype=np.float64))

    deriv_vals = []
    for f in DERIVATIVE_FEATURES:
        val = trigger_bar.get(f, 0.0)
        if pd.isna(val):
            val = 0.0
        deriv_vals.append(float(val))
    components.append(np.array(deriv_vals, dtype=np.float64))

    profile_vals = []
    for f in PROFILE_FEATURES:
        val = trigger_bar.get(f, 0.0)
        if pd.isna(val):
            val = 0.0
        profile_vals.append(float(val))
    components.append(np.array(profile_vals, dtype=np.float64))

    additional_profile = compute_additional_profile_features(lookback_bars)
    components.append(additional_profile)

    recent_momentum = compute_recent_momentum(lookback_bars)
    components.append(recent_momentum)

    raw_vector = np.concatenate(components)

    if len(raw_vector) < TARGET_DIM:
        padded = np.zeros(TARGET_DIM, dtype=np.float64)
        padded[:len(raw_vector)] = raw_vector
        return padded
    return raw_vector[:TARGET_DIM]


def log_transform_volumes(vector: np.ndarray, volume_indices: List[int]) -> np.ndarray:
    vec = vector.copy()
    for idx in volume_indices:
        if idx < len(vec):
            vec[idx] = np.sign(vec[idx]) * np.log1p(np.abs(vec[idx]))
    return vec


class GoldExtractSetupVectors(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_extract_setup_vectors",
            io=StageIO(
                inputs=[
                    f"silver.future.market_by_price_10_{lt.lower()}_approach"
                    for lt in LEVEL_TYPES
                ],
                output="gold.future.setup_vectors",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        output_key = self.io.output
        out_ref = partition_ref(cfg, output_key, symbol, dt)
        if is_partition_complete(out_ref):
            return

        all_vectors = []
        all_metadata = []

        for level_type in LEVEL_TYPES:
            input_key = f"silver.future.market_by_price_10_{level_type.lower()}_approach"
            in_ref = partition_ref(cfg, input_key, symbol, dt)

            if not is_partition_complete(in_ref):
                continue

            in_contract_path = repo_root / cfg.dataset(input_key).contract
            in_contract = load_avro_contract(in_contract_path)
            df_in = read_partition(in_ref)

            if len(df_in) == 0:
                continue

            df_in = enforce_contract(df_in, in_contract)

            vectors, metadata = self._extract_vectors_from_level(df_in, level_type, dt, symbol)
            all_vectors.extend(vectors)
            all_metadata.extend(metadata)

        if not all_vectors:
            df_out = pd.DataFrame(columns=[
                "vector_id", "episode_id", "dt", "symbol", "level_type",
                "level_price", "trigger_bar_ts", "approach_direction",
                "outcome", "outcome_score", "velocity_at_trigger",
                "obi0_at_trigger", "wall_imbal_at_trigger"
            ] + [f"v_{i}" for i in range(TARGET_DIM)])
        else:
            vectors_array = np.vstack(all_vectors)

            df_meta = pd.DataFrame(all_metadata)
            df_meta["vector_id"] = range(len(df_meta))
            df_meta["dt"] = dt

            vector_df = pd.DataFrame(
                vectors_array,
                columns=[f"v_{i}" for i in range(TARGET_DIM)]
            )

            df_out = pd.concat([df_meta, vector_df], axis=1)

        out_contract_path = repo_root / cfg.dataset(output_key).contract
        out_contract = load_avro_contract(out_contract_path)

        if len(df_out) > 0:
            df_out = enforce_contract(df_out, out_contract)

        lineage = []
        for level_type in LEVEL_TYPES:
            input_key = f"silver.future.market_by_price_10_{level_type.lower()}_approach"
            in_ref = partition_ref(cfg, input_key, symbol, dt)
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

    def _extract_vectors_from_level(
        self,
        df: pd.DataFrame,
        level_type: str,
        dt: str,
        symbol: str,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        vectors = []
        metadata = []

        df = df.sort_values("bar_ts").reset_index(drop=True)

        for episode_id in df["episode_id"].unique():
            episode_df = df[df["episode_id"] == episode_id].sort_values("bar_ts")

            trigger_mask = episode_df["is_trigger_bar"] == True
            if not trigger_mask.any():
                continue

            trigger_idx = trigger_mask.idxmax()
            trigger_bar = episode_df.loc[trigger_idx]

            pre_trigger = episode_df[episode_df["is_pre_trigger"] == True]

            vector = extract_setup_vector(trigger_bar, pre_trigger)
            vectors.append(vector)

            metadata.append({
                "episode_id": episode_id,
                "symbol": symbol,
                "level_type": level_type,
                "level_price": float(trigger_bar.get("level_price", 0.0)),
                "trigger_bar_ts": int(trigger_bar.get("trigger_bar_ts", 0)),
                "approach_direction": int(trigger_bar.get("approach_direction", 0)),
                "outcome": str(trigger_bar.get("outcome", "UNKNOWN")),
                "outcome_score": float(trigger_bar.get("outcome_score", 0.0)),
                "velocity_at_trigger": float(trigger_bar.get("bar5s_deriv_dist_d1_w12", 0.0) or 0.0),
                "obi0_at_trigger": float(trigger_bar.get("bar5s_state_obi0_eob", 0.0) or 0.0),
                "wall_imbal_at_trigger": float(trigger_bar.get("bar5s_setup_wall_imbal", 0.0) or 0.0),
            })

        return vectors, metadata
