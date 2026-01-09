from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

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

TICK_SIZE = 0.25
ZONE_WIDTH_TICKS = 4
MOVE_THRESHOLD_TICKS = 8
OUTCOME_HORIZON_CANDLES = 4
PRE_WINDOW_CANDLES = 5
COOLDOWN_CANDLES = 2

PRESSURE_BURST_THRESHOLD = 0.5
MAX_WALL_DIST_TICKS = 10.0
GAP_SPREAD_SCALE = 4.0

RTH_START_HOUR = 9
RTH_START_MINUTE = 30
RTH_HOURS = 3

REQUIRED_COLS = [
    "bar_ts",
    "bar2m_ts",
    "symbol",
    "episode_id",
    "touch_id",
    "level_type",
    "level_price",
    "approach_direction",
    "is_standard_approach",
    "bar5s_trade_cnt_sum",
    "bar5s_trade_last_px",
    "bar5s_trade_last_ts",
    "bar5s_microprice_eob",
    "bar5s_state_obi0_eob",
    "bar5s_state_obi10_eob",
    "bar5s_state_cdi_p0_1_eob",
    "bar5s_lvl_flow_toward_net_sum",
    "bar5s_lvl_flow_away_net_sum",
    "bar5s_lvldepth_below_total_qty_eob",
    "bar5s_lvldepth_above_total_qty_eob",
    "bar5s_trade_signed_vol_sum",
    "bar5s_trade_vol_sum",
    "bar5s_wall_bid_maxz_eob",
    "bar5s_wall_ask_maxz_eob",
    "bar5s_wall_bid_nearest_strong_dist_pts_eob",
    "bar5s_wall_ask_nearest_strong_dist_pts_eob",
    "bar5s_ladder_ask_gap_max_pts_eob",
    "bar5s_ladder_bid_gap_max_pts_eob",
    "bar5s_state_spread_pts_eob",
    "rvol_trade_cnt_zscore",
]


def _linearize_bounded(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return np.arctanh(clipped) / 2.0


def _signal_ops(values: np.ndarray, threshold: float, prefix: str) -> Dict[str, float]:
    vals = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(vals)
    if n == 0:
        return {f"{prefix}_{name}": 0.0 for name in [
            "start", "end", "min", "max", "mean", "std", "slope", "energy",
            "sign_flip_cnt", "burst_frac", "mean_early", "mean_mid", "mean_late",
            "energy_early", "energy_late", "late_minus_early", "late_over_early",
        ]}

    start = float(vals[0])
    end = float(vals[-1])
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    slope = float((end - start) / max(n, 1))
    energy_raw = float(np.sum(np.abs(np.diff(vals)))) if n > 1 else 0.0

    signs = np.sign(vals)
    sign_flip_cnt_raw = float(np.sum((signs[1:] * signs[:-1]) < 0)) if n > 1 else 0.0
    burst_frac = float(np.mean(vals > threshold)) if n > 0 else 0.0

    third = max(1, n // 3)
    early = vals[:third]
    mid = vals[third:2 * third] if n > 2 * third else vals[third:]
    late = vals[2 * third:] if n > 2 * third else vals[third:]

    mean_early = float(np.mean(early)) if len(early) > 0 else 0.0
    mean_mid = float(np.mean(mid)) if len(mid) > 0 else 0.0
    mean_late = float(np.mean(late)) if len(late) > 0 else 0.0

    energy_early_raw = float(np.sum(np.abs(np.diff(early)))) if len(early) > 1 else 0.0
    energy_late_raw = float(np.sum(np.abs(np.diff(late)))) if len(late) > 1 else 0.0

    energy = float(np.log1p(energy_raw))
    sign_flip_cnt = float(np.log1p(sign_flip_cnt_raw))
    energy_early = float(np.log1p(energy_early_raw))
    energy_late = float(np.log1p(energy_late_raw))

    late_minus_early = mean_late - mean_early
    late_over_early = mean_late / (mean_early + EPSILON)

    return {
        f"{prefix}_start": start,
        f"{prefix}_end": end,
        f"{prefix}_min": vmin,
        f"{prefix}_max": vmax,
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_slope": slope,
        f"{prefix}_energy": energy,
        f"{prefix}_sign_flip_cnt": sign_flip_cnt,
        f"{prefix}_burst_frac": burst_frac,
        f"{prefix}_mean_early": mean_early,
        f"{prefix}_mean_mid": mean_mid,
        f"{prefix}_mean_late": mean_late,
        f"{prefix}_energy_early": energy_early,
        f"{prefix}_energy_late": energy_late,
        f"{prefix}_late_minus_early": late_minus_early,
        f"{prefix}_late_over_early": late_over_early,
    }


def _pressure_components(df: pd.DataFrame, approach_direction: int) -> Dict[str, np.ndarray]:
    obi0 = df["bar5s_state_obi0_eob"].values * approach_direction
    obi10 = df["bar5s_state_obi10_eob"].values * approach_direction
    cdi = df["bar5s_state_cdi_p0_1_eob"].values * approach_direction

    obi0_lin = np.clip(_linearize_bounded(obi0), -1.0, 1.0)
    obi10_lin = np.clip(_linearize_bounded(obi10), -1.0, 1.0)
    cdi_lin = np.clip(_linearize_bounded(cdi), -1.0, 1.0)

    imbal_lin = (obi0_lin + obi10_lin + cdi_lin) / 3.0

    toward = df["bar5s_lvl_flow_toward_net_sum"].values
    away = df["bar5s_lvl_flow_away_net_sum"].values
    depth_total = (
        df["bar5s_lvldepth_below_total_qty_eob"].values
        + df["bar5s_lvldepth_above_total_qty_eob"].values
    )
    flow_norm = (toward - away) / (depth_total + EPSILON)
    flow_norm = np.clip(flow_norm, -1.0, 1.0)

    trade_signed = df["bar5s_trade_signed_vol_sum"].values
    trade_total = df["bar5s_trade_vol_sum"].values
    trade_imbal = (trade_signed / (trade_total + EPSILON)) * approach_direction
    trade_imbal = np.clip(trade_imbal, -1.0, 1.0)

    wall_bid = df["bar5s_wall_bid_maxz_eob"].values
    wall_ask = df["bar5s_wall_ask_maxz_eob"].values
    if approach_direction == 1:
        wall_same = wall_bid
        wall_opp = wall_ask
    else:
        wall_same = wall_ask
        wall_opp = wall_bid
    wall_support = (wall_same - wall_opp) / 4.0
    wall_support = np.clip(wall_support, -1.0, 1.0)

    wall_bid_dist = df["bar5s_wall_bid_nearest_strong_dist_pts_eob"].values
    wall_ask_dist = df["bar5s_wall_ask_nearest_strong_dist_pts_eob"].values
    if approach_direction == 1:
        wall_opp_dist = wall_ask_dist
    else:
        wall_opp_dist = wall_bid_dist
    wall_opp_dist_ticks = np.where(
        np.isnan(wall_opp_dist),
        MAX_WALL_DIST_TICKS,
        wall_opp_dist / TICK_SIZE,
    )
    wall_dist_support = np.clip(wall_opp_dist_ticks / MAX_WALL_DIST_TICKS, 0.0, 1.0)
    wall_dist_support = (2.0 * wall_dist_support) - 1.0

    gap_max = np.maximum(
        df["bar5s_ladder_ask_gap_max_pts_eob"].values,
        df["bar5s_ladder_bid_gap_max_pts_eob"].values,
    )
    spread = df["bar5s_state_spread_pts_eob"].values
    gap_ticks = gap_max / TICK_SIZE
    spread_ticks = spread / TICK_SIZE
    gap_spread_raw = np.log1p(gap_ticks) + np.log1p(spread_ticks)
    gap_spread = np.clip(gap_spread_raw / GAP_SPREAD_SCALE, 0.0, 1.0)
    gap_spread = (2.0 * gap_spread) - 1.0

    trade_activity = np.tanh(df["rvol_trade_cnt_zscore"].values / 3.0)

    obi0_lin = np.nan_to_num(obi0_lin, nan=0.0, posinf=0.0, neginf=0.0)
    obi10_lin = np.nan_to_num(obi10_lin, nan=0.0, posinf=0.0, neginf=0.0)
    cdi_lin = np.nan_to_num(cdi_lin, nan=0.0, posinf=0.0, neginf=0.0)
    imbal_lin = np.nan_to_num(imbal_lin, nan=0.0, posinf=0.0, neginf=0.0)
    flow_norm = np.nan_to_num(flow_norm, nan=0.0, posinf=0.0, neginf=0.0)
    trade_imbal = np.nan_to_num(trade_imbal, nan=0.0, posinf=0.0, neginf=0.0)
    wall_support = np.nan_to_num(wall_support, nan=0.0, posinf=0.0, neginf=0.0)
    wall_dist_support = np.nan_to_num(wall_dist_support, nan=0.0, posinf=0.0, neginf=0.0)
    gap_spread = np.nan_to_num(gap_spread, nan=0.0, posinf=0.0, neginf=0.0)
    trade_activity = np.nan_to_num(trade_activity, nan=0.0, posinf=0.0, neginf=0.0)

    pressure = np.mean(
        np.stack([
            imbal_lin,
            flow_norm,
            trade_imbal,
            wall_support,
            wall_dist_support,
            gap_spread,
            trade_activity,
        ], axis=0),
        axis=0,
    )

    return {
        "pressure": pressure,
        "obi0_lin": obi0_lin,
        "obi10_lin": obi10_lin,
        "cdi_lin": cdi_lin,
        "flow_norm": flow_norm,
        "trade_imbal": trade_imbal,
        "wall_support": wall_support,
        "wall_dist_support": wall_dist_support,
        "gap_spread": gap_spread,
        "trade_activity": trade_activity,
    }


def _rth_bounds_ns(dt: str) -> Tuple[int, int]:
    tz = ZoneInfo("America/New_York")
    date_obj = pd.Timestamp(dt, tz=tz)
    start = date_obj.replace(hour=RTH_START_HOUR, minute=RTH_START_MINUTE, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=RTH_HOURS)
    start_ns = int(start.tz_convert("UTC").value)
    end_ns = int(end.tz_convert("UTC").value)
    return start_ns, end_ns


def _is_rth_bar(bar_ts: int, dt: str) -> bool:
    start_ns, end_ns = _rth_bounds_ns(dt)
    return start_ns <= bar_ts < end_ns


def _compute_outcome(
    df_touch: pd.DataFrame,
    trigger_candle_ts: int,
    horizon_end_ts: int,
    level_price: float,
    approach_direction: int,
) -> Tuple[str, float]:
    mask = (df_touch["bar2m_ts"].values > trigger_candle_ts) & \
           (df_touch["bar2m_ts"].values <= horizon_end_ts)
    df_future = df_touch.loc[mask]
    if len(df_future) == 0:
        return "CHOP", 0.0

    trade_cnt = df_future["bar5s_trade_cnt_sum"].values
    trade_last_px = df_future["bar5s_trade_last_px"].values
    microprice = df_future["bar5s_microprice_eob"].values
    trade_close_px = np.where(trade_cnt > 0, trade_last_px, microprice)
    u = approach_direction * ((trade_close_px - level_price) / TICK_SIZE)

    u_max = float(np.max(u)) if len(u) > 0 else 0.0
    u_min = float(np.min(u)) if len(u) > 0 else 0.0

    break_idx = next((i for i, val in enumerate(u) if val >= MOVE_THRESHOLD_TICKS), None)
    reject_idx = next((i for i, val in enumerate(u) if val <= -MOVE_THRESHOLD_TICKS), None)

    if break_idx is not None and reject_idx is not None:
        outcome = "BREAK" if break_idx < reject_idx else "REJECT"
    elif break_idx is not None:
        outcome = "BREAK"
    elif reject_idx is not None:
        outcome = "REJECT"
    else:
        outcome = "CHOP"

    if outcome == "BREAK":
        score = u_max
    elif outcome == "REJECT":
        score = u_min
    else:
        score = 0.0

    return outcome, score


class SilverComputeApproach2m(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_approach2m",
            io=StageIO(
                inputs=[
                    f"silver.future.market_by_price_10_{lt.lower()}_approach"
                    for lt in LEVEL_TYPES
                ],
                output="silver.future.market_by_price_10_pm_high_approach2m",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        output_keys = [
            f"silver.future.market_by_price_10_{lt.lower()}_approach2m"
            for lt in LEVEL_TYPES
        ]

        all_complete = all(
            is_partition_complete(partition_ref(cfg, k, symbol, dt))
            for k in output_keys
        )
        if all_complete:
            return

        for level_type in LEVEL_TYPES:
            input_key = f"silver.future.market_by_price_10_{level_type.lower()}_approach"
            output_key = f"silver.future.market_by_price_10_{level_type.lower()}_approach2m"

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
                    df_out = self._build_approach2m(df_in, level_type, dt)

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

    def _build_approach2m(self, df: pd.DataFrame, level_type: str, dt: str) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for approach2m: {missing}")

        df = df.sort_values(["touch_id", "bar_ts"]).reset_index(drop=True)
        frames: List[pd.DataFrame] = []

        for touch_id, df_touch in df.groupby("touch_id", sort=False):
            df_touch = df_touch.sort_values("bar_ts").reset_index(drop=True)
            approach_direction = int(df_touch["approach_direction"].iloc[0])
            level_price = float(df_touch["level_price"].iloc[0])
            symbol = str(df_touch["symbol"].iloc[0])
            episode_id = str(df_touch["episode_id"].iloc[0])
            is_standard = bool(df_touch["is_standard_approach"].iloc[0])

            candles = self._build_candles(
                df_touch, level_price, approach_direction, level_type, symbol, episode_id, touch_id, is_standard
            )
            if not candles:
                continue

            trigger_idx = self._find_trigger(candles, dt)
            if trigger_idx is None:
                continue

            pre_start = max(0, trigger_idx - (PRE_WINDOW_CANDLES - 1))
            horizon_end = min(trigger_idx + OUTCOME_HORIZON_CANDLES, len(candles) - 1)
            trigger_candle_ts = candles[trigger_idx]["bar_ts"]

            outcome, outcome_score = _compute_outcome(
                df_touch,
                trigger_candle_ts,
                candles[horizon_end]["bar_ts"],
                level_price,
                approach_direction,
            )

            truncated = pre_start != trigger_idx - (PRE_WINDOW_CANDLES - 1)

            window_rows = []
            for idx in range(pre_start, horizon_end + 1):
                row = dict(candles[idx])
                row["trigger_candle_ts"] = trigger_candle_ts
                row["bars_to_trigger"] = idx - trigger_idx
                row["is_pre_trigger"] = row["bars_to_trigger"] < 0
                row["is_trigger_candle"] = row["bars_to_trigger"] == 0
                row["is_post_trigger"] = row["bars_to_trigger"] > 0
                row["outcome"] = outcome
                row["outcome_score"] = outcome_score
                row["is_premarket_context_truncated"] = truncated
                window_rows.append(row)

            frames.append(pd.DataFrame(window_rows))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _build_candles(
        self,
        df_touch: pd.DataFrame,
        level_price: float,
        approach_direction: int,
        level_type: str,
        symbol: str,
        episode_id: str,
        touch_id: str,
        is_standard: bool,
    ) -> List[Dict[str, float]]:
        candles: List[Dict[str, float]] = []
        grouped = df_touch.groupby("bar2m_ts", sort=True)

        for idx, (bar2m_ts, df_c) in enumerate(grouped):
            trade_cnt = df_c["bar5s_trade_cnt_sum"].values
            trade_last_px = df_c["bar5s_trade_last_px"].values
            trade_last_ts = df_c["bar5s_trade_last_ts"].values
            microprice = df_c["bar5s_microprice_eob"].values

            trade_close_px = np.where(trade_cnt > 0, trade_last_px, microprice)
            u = approach_direction * ((trade_close_px - level_price) / TICK_SIZE)

            zone_width_pts = ZONE_WIDTH_TICKS * TICK_SIZE
            in_zone = ((trade_cnt > 0) & (np.abs(trade_last_px - level_price) <= zone_width_pts)).astype(np.float64)

            touched_in_zone = bool(in_zone.sum() > 0)
            first_touch_offset = int(np.argmax(in_zone)) if touched_in_zone else -1

            has_trade = trade_last_ts >= 0
            if np.any(has_trade):
                last_idx = int(np.argmax(trade_last_ts))
                close_px = float(trade_last_px[last_idx])
                close_in_zone = abs(close_px - level_price) <= zone_width_pts
                close_side = int(np.sign((close_px - level_price) * approach_direction))
            else:
                close_in_zone = False
                close_side = 0

            third = max(1, len(u) // 3)
            late_slice = u[2 * third:] if len(u) > 2 * third else u[third:]

            time_in_zone_frac = float(np.mean(in_zone)) if len(in_zone) > 0 else 0.0
            time_far_side_frac = float(np.mean(u > 0)) if len(u) > 0 else 0.0
            late_time_far_side_frac = float(np.mean(late_slice > 0)) if len(late_slice) > 0 else 0.0

            comps = _pressure_components(df_c, approach_direction)

            row: Dict[str, float] = {
                "bar_ts": int(bar2m_ts),
                "symbol": symbol,
                "episode_id": episode_id,
                "touch_id": touch_id,
                "level_type": level_type,
                "level_price": level_price,
                "bar_index_in_episode": idx,
                "bar_index_in_touch": idx,
                "approach_direction": approach_direction,
                "is_standard_approach": is_standard,
                "bar2m_touched_in_zone": touched_in_zone,
                "bar2m_close_in_zone": close_in_zone,
                "bar2m_first_touch_offset": first_touch_offset,
                "bar2m_time_in_zone_frac": time_in_zone_frac,
                "bar2m_time_far_side_frac": time_far_side_frac,
                "bar2m_late_time_far_side_frac": late_time_far_side_frac,
                "bar2m_close_side": close_side,
            }

            row = {**row, **_signal_ops(u, 0.0, "bar2m_sig_u")}
            row = {**row, **_signal_ops(comps["pressure"], PRESSURE_BURST_THRESHOLD, "bar2m_sig_pressure")}

            row["bar2m_comp_obi0_lin_mean"] = float(np.mean(comps["obi0_lin"]))
            row["bar2m_comp_obi10_lin_mean"] = float(np.mean(comps["obi10_lin"]))
            row["bar2m_comp_cdi_lin_mean"] = float(np.mean(comps["cdi_lin"]))
            row["bar2m_comp_flow_norm_mean"] = float(np.mean(comps["flow_norm"]))
            row["bar2m_comp_trade_imbal_mean"] = float(np.mean(comps["trade_imbal"]))
            row["bar2m_comp_wall_support_mean"] = float(np.mean(comps["wall_support"]))
            row["bar2m_comp_wall_dist_support_mean"] = float(np.mean(comps["wall_dist_support"]))
            row["bar2m_comp_gap_spread_mean"] = float(np.mean(comps["gap_spread"]))
            row["bar2m_comp_trade_activity_mean"] = float(np.mean(comps["trade_activity"]))

            candles.append(row)

        return candles

    def _find_trigger(self, candles: List[Dict[str, float]], dt: str) -> Optional[int]:
        last_trigger_idx = -COOLDOWN_CANDLES
        for idx, row in enumerate(candles):
            if idx < last_trigger_idx + COOLDOWN_CANDLES:
                continue
            if not _is_rth_bar(int(row["bar_ts"]), dt):
                continue
            if row["bar2m_touched_in_zone"] and row["bar2m_close_in_zone"]:
                last_trigger_idx = idx
                return idx
        return None
