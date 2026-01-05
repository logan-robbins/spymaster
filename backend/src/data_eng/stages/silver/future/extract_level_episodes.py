from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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

BAR_DURATION_NS = 5_000_000_000
LOOKBACK_BARS = 180
MIN_FORWARD_BARS = 96
EXTENSION_BARS = 96
COOLDOWN_BARS = 24
TRIGGER_BAND_PTS = 1.0
OUTCOME_W1_END = 48
OUTCOME_W2_START = 49
OUTCOME_W2_END = 72
EPSILON = 1e-9

LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]

BAND_COLS_TO_DROP = [
    "bar5s_state_cdi_p3_5_twa",
    "bar5s_state_cdi_p3_5_eob",
    "bar5s_state_cdi_p5_10_twa",
    "bar5s_state_cdi_p5_10_eob",
    "bar5s_depth_below_p3_5_qty_twa",
    "bar5s_depth_below_p3_5_qty_eob",
    "bar5s_depth_below_p5_10_qty_twa",
    "bar5s_depth_below_p5_10_qty_eob",
    "bar5s_depth_above_p3_5_qty_twa",
    "bar5s_depth_above_p3_5_qty_eob",
    "bar5s_depth_above_p5_10_qty_twa",
    "bar5s_depth_above_p5_10_qty_eob",
    "bar5s_depth_below_p3_5_frac_twa",
    "bar5s_depth_below_p3_5_frac_eob",
    "bar5s_depth_below_p5_10_frac_twa",
    "bar5s_depth_below_p5_10_frac_eob",
    "bar5s_depth_above_p3_5_frac_twa",
    "bar5s_depth_above_p3_5_frac_eob",
    "bar5s_depth_above_p5_10_frac_twa",
    "bar5s_depth_above_p5_10_frac_eob",
    "bar5s_flow_add_vol_bid_p3_5_sum",
    "bar5s_flow_add_vol_bid_p5_10_sum",
    "bar5s_flow_add_vol_ask_p3_5_sum",
    "bar5s_flow_add_vol_ask_p5_10_sum",
    "bar5s_flow_rem_vol_bid_p3_5_sum",
    "bar5s_flow_rem_vol_bid_p5_10_sum",
    "bar5s_flow_rem_vol_ask_p3_5_sum",
    "bar5s_flow_rem_vol_ask_p5_10_sum",
    "bar5s_flow_net_vol_bid_p3_5_sum",
    "bar5s_flow_net_vol_bid_p5_10_sum",
    "bar5s_flow_net_vol_ask_p3_5_sum",
    "bar5s_flow_net_vol_ask_p5_10_sum",
    "bar5s_flow_cnt_add_bid_p3_5_sum",
    "bar5s_flow_cnt_add_bid_p5_10_sum",
    "bar5s_flow_cnt_add_ask_p3_5_sum",
    "bar5s_flow_cnt_add_ask_p5_10_sum",
    "bar5s_flow_cnt_cancel_bid_p3_5_sum",
    "bar5s_flow_cnt_cancel_bid_p5_10_sum",
    "bar5s_flow_cnt_cancel_ask_p3_5_sum",
    "bar5s_flow_cnt_cancel_ask_p5_10_sum",
    "bar5s_flow_cnt_modify_bid_p3_5_sum",
    "bar5s_flow_cnt_modify_bid_p5_10_sum",
    "bar5s_flow_cnt_modify_ask_p3_5_sum",
    "bar5s_flow_cnt_modify_ask_p5_10_sum",
    "bar5s_flow_net_volnorm_bid_p3_5_sum",
    "bar5s_flow_net_volnorm_bid_p5_10_sum",
    "bar5s_flow_net_volnorm_ask_p3_5_sum",
    "bar5s_flow_net_volnorm_ask_p5_10_sum",
]


class SilverExtractLevelEpisodes(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_extract_level_episodes",
            io=StageIO(
                inputs=["silver.future.market_by_price_10_bar5s"],
                output="silver.future.market_by_price_10_pm_high_episodes",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        output_keys = [
            f"silver.future.market_by_price_10_{lt.lower()}_episodes"
            for lt in LEVEL_TYPES
        ]

        all_complete = all(
            is_partition_complete(partition_ref(cfg, k, symbol, dt))
            for k in output_keys
        )
        if all_complete:
            return

        in_ref = partition_ref(cfg, self.io.inputs[0], symbol, dt)
        if not is_partition_complete(in_ref):
            raise FileNotFoundError(
                f"Input not ready: {in_ref.dataset_key} dt={dt}"
            )

        in_contract_path = repo_root / cfg.dataset(in_ref.dataset_key).contract
        in_contract = load_avro_contract(in_contract_path)
        df_in = read_partition(in_ref)
        df_in = enforce_contract(df_in, in_contract)

        if len(df_in) == 0:
            return

        levels = self._compute_levels(df_in, dt)

        for level_type in LEVEL_TYPES:
            output_key = f"silver.future.market_by_price_10_{level_type.lower()}_episodes"
            out_ref = partition_ref(cfg, output_key, symbol, dt)
            if is_partition_complete(out_ref):
                continue

            level_price = levels.get(level_type)
            if level_price is None or np.isnan(level_price):
                df_episodes = pd.DataFrame()
            else:
                df_episodes = self._extract_episodes_for_level(
                    df_in, dt, symbol, level_type, level_price
                )

            out_contract_path = repo_root / cfg.dataset(output_key).contract
            out_contract = load_avro_contract(out_contract_path)

            if len(df_episodes) > 0:
                cols_to_drop = [c for c in BAND_COLS_TO_DROP if c in df_episodes.columns]
                if cols_to_drop:
                    df_episodes = df_episodes.drop(columns=cols_to_drop)
                df_episodes = enforce_contract(df_episodes, out_contract)

            lineage = [
                {
                    "dataset": in_ref.dataset_key,
                    "dt": dt,
                    "manifest_sha256": read_manifest_hash(in_ref),
                }
            ]

            write_partition(
                cfg=cfg,
                dataset_key=output_key,
                symbol=symbol,
                dt=dt,
                df=df_episodes,
                contract_path=out_contract_path,
                inputs=lineage,
                stage=self.name,
            )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        raise NotImplementedError("Use run() directly")

    def _compute_levels(self, df: pd.DataFrame, dt: str) -> Dict[str, float]:
        bar_ts = df["bar_ts"].values
        microprice = df["bar5s_microprice_eob"].values

        date_ns = pd.Timestamp(dt, tz="America/New_York").value
        market_open_ns = date_ns + int(9.5 * 3600 * 1e9)
        pm_start_ns = date_ns + int(5 * 3600 * 1e9)
        or_end_ns = market_open_ns + int(30 * 60 * 1e9)

        pm_mask = (bar_ts >= pm_start_ns) & (bar_ts < market_open_ns)
        or_mask = (bar_ts >= market_open_ns) & (bar_ts < or_end_ns)

        pm_high = float(microprice[pm_mask].max()) if pm_mask.any() else np.nan
        pm_low = float(microprice[pm_mask].min()) if pm_mask.any() else np.nan
        or_high = float(microprice[or_mask].max()) if or_mask.any() else np.nan
        or_low = float(microprice[or_mask].min()) if or_mask.any() else np.nan

        return {
            "PM_HIGH": pm_high,
            "PM_LOW": pm_low,
            "OR_HIGH": or_high,
            "OR_LOW": or_low,
        }

    def _extract_episodes_for_level(
        self,
        df: pd.DataFrame,
        dt: str,
        symbol: str,
        level_type: str,
        level_price: float,
    ) -> pd.DataFrame:
        df = df.sort_values("bar_ts").reset_index(drop=True)
        n = len(df)
        bar_ts = df["bar_ts"].values
        microprice = df["bar5s_microprice_eob"].values
        trade_cnt = df["bar5s_trade_cnt_sum"].values

        date_ns = pd.Timestamp(dt, tz="America/New_York").value
        market_open_ns = date_ns + int(9.5 * 3600 * 1e9)
        session_end_ns = market_open_ns + int(4 * 3600 * 1e9)
        or_end_ns = market_open_ns + int(30 * 60 * 1e9)

        if level_type in ["OR_HIGH", "OR_LOW"]:
            valid_start_ns = or_end_ns
        else:
            valid_start_ns = market_open_ns

        valid_start_idx = np.searchsorted(bar_ts, valid_start_ns)
        session_end_idx = np.searchsorted(bar_ts, session_end_ns)

        level_polarity = 1 if level_type in ["PM_HIGH", "OR_HIGH"] else -1

        episodes = []
        last_trigger_idx = -COOLDOWN_BARS

        for t in range(valid_start_idx, min(session_end_idx, n)):
            if t < last_trigger_idx + COOLDOWN_BARS:
                continue

            dist = abs(microprice[t] - level_price)
            if dist > TRIGGER_BAND_PTS:
                continue

            if trade_cnt[t] == 0:
                if t > 0:
                    prev_dist = abs(microprice[t - 1] - level_price)
                    if prev_dist <= TRIGGER_BAND_PTS:
                        continue

            approach_direction = self._compute_approach_direction(
                microprice, t, level_price
            )
            is_standard_approach = (approach_direction == -level_polarity)

            lookback_start = max(0, t - LOOKBACK_BARS)
            forward_end = self._compute_forward_end(
                microprice, t, level_price, min(session_end_idx, n)
            )

            outcome, outcome_score = self._compute_outcome(
                df, t, level_price, approach_direction
            )

            is_truncated_lookback = (t - lookback_start) < LOOKBACK_BARS
            is_truncated_forward = forward_end >= min(session_end_idx, n) - 1
            is_extended_forward = forward_end > t + MIN_FORWARD_BARS
            extension_count = max(0, (forward_end - t - MIN_FORWARD_BARS) // EXTENSION_BARS)

            episode_df = df.iloc[lookback_start : forward_end + 1].copy()
            trigger_offset = t - lookback_start

            episode_id = f"{dt}_{symbol}_{level_type}_{int(bar_ts[t])}"

            episode_df["episode_id"] = episode_id
            episode_df["touch_id"] = episode_id
            episode_df["level_type"] = level_type
            episode_df["level_price"] = level_price
            episode_df["trigger_bar_ts"] = int(bar_ts[t])
            episode_df["bar_index_in_episode"] = range(len(episode_df))
            episode_df["bar_index_in_touch"] = episode_df["bar_index_in_episode"]
            episode_df["bars_to_trigger"] = (
                episode_df["bar_index_in_episode"] - trigger_offset
            )
            episode_df["is_pre_trigger"] = episode_df["bars_to_trigger"] < 0
            episode_df["is_pre_touch"] = episode_df["is_pre_trigger"]
            episode_df["is_trigger_bar"] = episode_df["bars_to_trigger"] == 0
            episode_df["is_post_trigger"] = episode_df["bars_to_trigger"] > 0
            episode_df["is_post_touch"] = episode_df["is_post_trigger"]
            episode_df["approach_direction"] = approach_direction
            episode_df["is_standard_approach"] = is_standard_approach
            episode_df["dist_to_level_pts"] = (
                episode_df["bar5s_microprice_eob"] - level_price
            )
            episode_df["signed_dist_pts"] = (
                episode_df["dist_to_level_pts"] * approach_direction
            )
            episode_df["outcome"] = outcome
            episode_df["outcome_score"] = outcome_score
            episode_df["is_truncated_lookback"] = is_truncated_lookback
            episode_df["is_truncated_forward"] = is_truncated_forward
            episode_df["is_extended_forward"] = is_extended_forward
            episode_df["extension_count"] = extension_count

            episodes.append(episode_df)
            last_trigger_idx = t

        if not episodes:
            return pd.DataFrame()

        return pd.concat(episodes, ignore_index=True)

    def _compute_approach_direction(
        self, microprice: np.ndarray, trigger_idx: int, level_price: float
    ) -> int:
        lookback_bars = 12
        start = max(0, trigger_idx - lookback_bars)
        pre_microprice = microprice[start:trigger_idx].mean()
        if pre_microprice < level_price:
            return 1
        return -1

    def _compute_forward_end(
        self,
        microprice: np.ndarray,
        trigger_idx: int,
        level_price: float,
        max_idx: int,
    ) -> int:
        forward_end = min(trigger_idx + MIN_FORWARD_BARS, max_idx - 1)
        exited = False
        exit_duration = 0

        for t in range(trigger_idx + 1, min(max_idx, trigger_idx + 500)):
            dist = abs(microprice[t] - level_price)
            if dist > TRIGGER_BAND_PTS:
                exit_duration += 1
                if exit_duration >= 6:
                    exited = True
            else:
                if exited:
                    forward_end = max(forward_end, min(t + EXTENSION_BARS, max_idx - 1))
                    exited = False
                exit_duration = 0

            if t > forward_end and exited:
                break

        return forward_end

    def _compute_outcome(
        self,
        df: pd.DataFrame,
        trigger_idx: int,
        level_price: float,
        approach_direction: int,
    ) -> tuple:
        n = len(df)
        microprice = df["bar5s_microprice_eob"].values
        trade_vol = df["bar5s_trade_vol_sum"].values

        def vwap_signed_dist(start_offset: int, end_offset: int) -> float:
            start = trigger_idx + start_offset
            end = min(trigger_idx + end_offset, n - 1)
            if start >= n or end < start:
                return 0.0

            window_microprice = microprice[start : end + 1]
            window_vol = trade_vol[start : end + 1]
            dist = (window_microprice - level_price) * approach_direction

            if window_vol.sum() > 0:
                return float((dist * window_vol).sum() / window_vol.sum())
            return float(dist.mean())

        w1_dist = vwap_signed_dist(1, OUTCOME_W1_END)
        w2_dist = vwap_signed_dist(OUTCOME_W2_START, OUTCOME_W2_END)

        if w1_dist > 1.0:
            outcome = "STRONG_BREAK" if w2_dist > 2.0 else "WEAK_BREAK"
        elif w1_dist < -1.0:
            outcome = "STRONG_BOUNCE" if w2_dist < -2.0 else "WEAK_BOUNCE"
        else:
            outcome = "CHOP"

        outcome_score = (w1_dist + w2_dist) / 2.0

        return outcome, outcome_score
