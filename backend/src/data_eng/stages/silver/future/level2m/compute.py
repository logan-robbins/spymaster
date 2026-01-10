"""Compliant level-approach setup extraction.

Implements the exact semantics from COMPLIANCE_GPT.md:
- Trigger: first candle that crosses level (prev fully on approach side)
- Confirmation: trigger + CONFIRM_BARS
- Inference: after confirmation close
- Lookback: LOOKBACK_BARS_INFER ending at confirmation (inclusive)
- Look-forward: LOOKFWD_BARS_LABEL starting after confirmation
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .setup_config import SetupConfig, DEFAULT_CONFIG

EPSILON = 1e-9
TICK_SIZE = 0.25

BAR2M_DURATION_NS = 120_000_000_000

RTH_START_HOUR = 9
RTH_START_MINUTE = 30
RTH_HOURS = 3

PRESSURE_BURST_THRESHOLD = 0.5
MAX_WALL_DIST_TICKS = 10.0
GAP_SPREAD_SCALE = 4.0


def _linearize_bounded(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return np.arctanh(clipped) / 2.0


def _signal_ops(values: np.ndarray, threshold: float, prefix: str) -> Dict[str, float]:
    vals = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(vals)

    if n == 0:
        return {
            f"{prefix}_start": 0.0, f"{prefix}_end": 0.0,
            f"{prefix}_min": 0.0, f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
            f"{prefix}_slope": 0.0, f"{prefix}_energy": 0.0,
            f"{prefix}_sign_flip_cnt": 0.0, f"{prefix}_burst_frac": 0.0,
            f"{prefix}_mean_early": 0.0, f"{prefix}_mean_mid": 0.0,
            f"{prefix}_mean_late": 0.0, f"{prefix}_energy_early": 0.0,
            f"{prefix}_energy_late": 0.0, f"{prefix}_late_minus_early": 0.0,
            f"{prefix}_late_over_early": 0.0,
        }

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
    late_over_early = mean_late / (abs(mean_early) + EPSILON)

    return {
        f"{prefix}_start": start, f"{prefix}_end": end,
        f"{prefix}_min": vmin, f"{prefix}_max": vmax,
        f"{prefix}_mean": mean, f"{prefix}_std": std,
        f"{prefix}_slope": slope, f"{prefix}_energy": energy,
        f"{prefix}_sign_flip_cnt": sign_flip_cnt, f"{prefix}_burst_frac": burst_frac,
        f"{prefix}_mean_early": mean_early, f"{prefix}_mean_mid": mean_mid,
        f"{prefix}_mean_late": mean_late, f"{prefix}_energy_early": energy_early,
        f"{prefix}_energy_late": energy_late, f"{prefix}_late_minus_early": late_minus_early,
        f"{prefix}_late_over_early": late_over_early,
    }


def _compute_pressure_components(df: pd.DataFrame, approach_direction: int) -> Dict[str, np.ndarray]:
    obi0 = df["bar5s_state_obi0_eob"].values * approach_direction
    obi10 = df["bar5s_state_obi10_eob"].values * approach_direction
    cdi = df["bar5s_state_cdi_p0_1_eob"].values * approach_direction

    obi0_lin = np.clip(_linearize_bounded(obi0), -1.0, 1.0)
    obi10_lin = np.clip(_linearize_bounded(obi10), -1.0, 1.0)
    cdi_lin = np.clip(_linearize_bounded(cdi), -1.0, 1.0)

    flow_bid = df["bar5s_flow_net_volnorm_bid_p0_1_sum"].values
    flow_ask = df["bar5s_flow_net_volnorm_ask_p0_1_sum"].values
    flow_norm = np.clip((flow_bid - flow_ask) * approach_direction, -1.0, 1.0)

    trade_signed = df["bar5s_trade_signed_vol_sum"].values
    trade_total = df["bar5s_trade_vol_sum"].values
    trade_imbal = np.clip((trade_signed / (trade_total + EPSILON)) * approach_direction, -1.0, 1.0)

    wall_bid = df["bar5s_wall_bid_maxz_eob"].values
    wall_ask = df["bar5s_wall_ask_maxz_eob"].values
    if approach_direction == 1:
        wall_same = wall_bid
        wall_opp = wall_ask
    else:
        wall_same = wall_ask
        wall_opp = wall_bid
    wall_support = np.clip((wall_same - wall_opp) / 4.0, -1.0, 1.0)

    wall_bid_dist = df["bar5s_wall_bid_nearest_strong_dist_pts_eob"].values
    wall_ask_dist = df["bar5s_wall_ask_nearest_strong_dist_pts_eob"].values
    if approach_direction == 1:
        wall_opp_dist = wall_ask_dist
    else:
        wall_opp_dist = wall_bid_dist
    wall_opp_dist_ticks = np.where(
        np.isnan(wall_opp_dist), MAX_WALL_DIST_TICKS, wall_opp_dist / TICK_SIZE
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

    trade_cnt = df["bar5s_trade_cnt_sum"].values
    trade_activity = np.tanh(np.log1p(trade_cnt) / 3.0)

    for arr in [obi0_lin, obi10_lin, cdi_lin, flow_norm, trade_imbal,
                wall_support, wall_dist_support, gap_spread, trade_activity]:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    pressure = np.mean(
        np.stack([
            (obi0_lin + obi10_lin + cdi_lin) / 3.0,
            flow_norm, trade_imbal, wall_support,
            wall_dist_support, gap_spread, trade_activity,
        ], axis=0),
        axis=0,
    )

    return {
        "pressure": pressure,
        "obi0_lin": obi0_lin, "obi10_lin": obi10_lin, "cdi_lin": cdi_lin,
        "flow_norm": flow_norm, "trade_imbal": trade_imbal,
        "wall_support": wall_support, "wall_dist_support": wall_dist_support,
        "gap_spread": gap_spread, "trade_activity": trade_activity,
    }


def _rth_bounds_ns(dt: str) -> Tuple[int, int]:
    tz = ZoneInfo("America/New_York")
    date_obj = pd.Timestamp(dt, tz=tz)
    start = date_obj.replace(hour=RTH_START_HOUR, minute=RTH_START_MINUTE, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=RTH_HOURS)
    start_ns = int(start.tz_convert("UTC").value)
    end_ns = int(end.tz_convert("UTC").value)
    return start_ns, end_ns


def _compute_bar2m_ts(bar_ts: np.ndarray) -> np.ndarray:
    return (bar_ts // BAR2M_DURATION_NS) * BAR2M_DURATION_NS


def _get_candle_ohlc(df_candle: pd.DataFrame, level_price: float) -> Dict[str, float]:
    """Extract OHLC from 5s micro bars within a 2m candle."""
    trade_cnt = df_candle["bar5s_trade_cnt_sum"].values
    trade_last_px = df_candle["bar5s_trade_last_px"].values
    trade_last_ts = df_candle["bar5s_trade_last_ts"].values
    microprice = df_candle["bar5s_microprice_eob"].values

    trade_mask = trade_cnt > 0
    if not np.any(trade_mask):
        mp = microprice[0] if len(microprice) > 0 else level_price
        return {"open": mp, "high": mp, "low": mp, "close": mp}

    traded_prices = trade_last_px[trade_mask]
    traded_ts = trade_last_ts[trade_mask]

    first_idx = np.argmin(traded_ts)
    last_idx = np.argmax(traded_ts)

    return {
        "open": float(traded_prices[first_idx]),
        "high": float(np.max(traded_prices)),
        "low": float(np.min(traded_prices)),
        "close": float(traded_prices[last_idx]),
    }


def _detect_triggers(
    df: pd.DataFrame,
    candle_ts_list: List[int],
    level_price: float,
    approach_direction: int,
    cfg: SetupConfig,
    rth_start_ns: int,
    rth_end_ns: int,
) -> List[Tuple[int, int, int]]:
    """Detect trigger candles per COMPLIANCE spec.

    Trigger: candle t where:
      - Previous candle fully on approach side (high[t-1] < L for from_below)
      - Current candle crosses level (high[t] >= L for from_below)
      - At least approach_side_min_bars bars were on approach side before this

    Returns: List of (trigger_idx, trigger_ts, confirm_ts)
    """
    triggers = []
    last_trigger_idx = -cfg.min_bars_between_touches - 1
    reset_active = True
    bars_on_approach_side = 0

    for idx, bar2m_ts in enumerate(candle_ts_list):
        if bar2m_ts < rth_start_ns or bar2m_ts >= rth_end_ns:
            continue

        df_candle = df[df["bar2m_ts"] == bar2m_ts]
        if len(df_candle) == 0:
            continue

        ohlc = _get_candle_ohlc(df_candle, level_price)

        if approach_direction == 1:
            fully_on_approach = ohlc["high"] < level_price - cfg.cross_epsilon_pts
            crossed = ohlc["high"] >= level_price + cfg.cross_epsilon_pts
            reset_dist = level_price - ohlc["close"]
        else:
            fully_on_approach = ohlc["low"] > level_price + cfg.cross_epsilon_pts
            crossed = ohlc["low"] <= level_price - cfg.cross_epsilon_pts
            reset_dist = ohlc["close"] - level_price

        bars_before_this_candle = bars_on_approach_side

        if reset_active:
            if fully_on_approach:
                bars_on_approach_side += 1
            else:
                if reset_dist >= cfg.reset_distance_pts:
                    bars_on_approach_side = 1
                else:
                    bars_on_approach_side = 0
        else:
            if reset_dist >= cfg.reset_distance_pts:
                reset_active = True
                if fully_on_approach:
                    bars_on_approach_side = 1
                else:
                    bars_on_approach_side = 0
            else:
                bars_on_approach_side = 0

        if idx <= last_trigger_idx + cfg.min_bars_between_touches:
            continue

        if len(triggers) >= cfg.max_touches_per_level_per_session:
            break

        if idx == 0:
            continue

        prev_ts = candle_ts_list[idx - 1]
        df_prev = df[df["bar2m_ts"] == prev_ts]
        if len(df_prev) == 0:
            continue

        prev_ohlc = _get_candle_ohlc(df_prev, level_price)

        if approach_direction == 1:
            prev_fully_approach = prev_ohlc["high"] < level_price - cfg.cross_epsilon_pts
        else:
            prev_fully_approach = prev_ohlc["low"] > level_price + cfg.cross_epsilon_pts

        if not prev_fully_approach:
            continue

        if bars_before_this_candle < cfg.approach_side_min_bars:
            continue

        if crossed:
            confirm_idx = min(idx + cfg.confirm_bars, len(candle_ts_list) - 1)
            confirm_ts = candle_ts_list[confirm_idx]
            triggers.append((idx, int(bar2m_ts), int(confirm_ts)))
            last_trigger_idx = idx
            reset_active = False
            bars_on_approach_side = 0

    return triggers


def _compute_outcome_and_flags(
    df: pd.DataFrame,
    candle_ts_list: List[int],
    confirm_candle_idx: int,
    level_price: float,
    approach_direction: int,
    trigger_ohlc: Dict[str, float],
    confirm_ohlc: Dict[str, float],
    cfg: SetupConfig,
) -> Dict:
    """Compute outcome label, score, and all flags per COMPLIANCE spec."""
    if confirm_candle_idx >= len(candle_ts_list) - 1:
        return {
            "outcome": "CHOP",
            "outcome_score": 0.0,
            "max_signed_dist": 0.0,
            "min_signed_dist": 0.0,
            "chop_flag": True,
            "failed_break_flag": True,
            "both_sides_hit_flag": False,
            "first_hit_side": "none",
            "first_hit_offset_bars": -1,
            "mae_pts": 0.0,
            "mfe_pts": 0.0,
        }

    lookfwd_start_idx = confirm_candle_idx + 1
    lookfwd_end_idx = min(confirm_candle_idx + cfg.lookfwd_bars_label, len(candle_ts_list) - 1)

    lookfwd_candles = candle_ts_list[lookfwd_start_idx:lookfwd_end_idx + 1]
    df_horizon = df[df["bar2m_ts"].isin(lookfwd_candles)]

    if len(df_horizon) == 0:
        return {
            "outcome": "CHOP",
            "outcome_score": 0.0,
            "max_signed_dist": 0.0,
            "min_signed_dist": 0.0,
            "chop_flag": True,
            "failed_break_flag": True,
            "both_sides_hit_flag": False,
            "first_hit_side": "none",
            "first_hit_offset_bars": -1,
            "mae_pts": 0.0,
            "mfe_pts": 0.0,
        }

    signed_dists = []
    high_dists = []
    low_dists = []

    for candle_ts in lookfwd_candles:
        df_c = df_horizon[df_horizon["bar2m_ts"] == candle_ts]
        if len(df_c) == 0:
            continue
        ohlc = _get_candle_ohlc(df_c, level_price)

        close_dist = (ohlc["close"] - level_price) * approach_direction
        high_dist = (ohlc["high"] - level_price) * approach_direction
        low_dist = (ohlc["low"] - level_price) * approach_direction

        signed_dists.append(close_dist)
        high_dists.append(high_dist)
        low_dists.append(low_dist)

    if len(signed_dists) == 0:
        return {
            "outcome": "CHOP",
            "outcome_score": 0.0,
            "max_signed_dist": 0.0,
            "min_signed_dist": 0.0,
            "chop_flag": True,
            "failed_break_flag": True,
            "both_sides_hit_flag": False,
            "first_hit_side": "none",
            "first_hit_offset_bars": -1,
            "mae_pts": 0.0,
            "mfe_pts": 0.0,
        }

    if cfg.outcome_price_basis == "close":
        upside_arr = np.array(signed_dists)
        downside_arr = np.array(signed_dists)
    else:
        upside_arr = np.array(high_dists)
        downside_arr = np.array(low_dists)

    max_signed_dist = float(np.max(upside_arr))
    min_signed_dist = float(np.min(downside_arr))

    break_idx = next((i for i, v in enumerate(upside_arr) if v >= cfg.break_threshold_pts), None)
    reject_idx = next((i for i, v in enumerate(downside_arr) if v <= -cfg.reject_threshold_pts), None)

    if break_idx is not None and reject_idx is not None:
        if break_idx < reject_idx:
            outcome = "BREAK"
            first_hit_side = "break"
            first_hit_offset_bars = break_idx + 1
        elif reject_idx < break_idx:
            outcome = "REJECT"
            first_hit_side = "reject"
            first_hit_offset_bars = reject_idx + 1
        else:
            outcome = "CHOP"
            first_hit_side = "tie"
            first_hit_offset_bars = break_idx + 1
        both_sides_hit_flag = True
    elif break_idx is not None:
        outcome = "BREAK"
        both_sides_hit_flag = False
        first_hit_side = "break"
        first_hit_offset_bars = break_idx + 1
    elif reject_idx is not None:
        outcome = "REJECT"
        both_sides_hit_flag = False
        first_hit_side = "reject"
        first_hit_offset_bars = reject_idx + 1
    else:
        outcome = "CHOP"
        both_sides_hit_flag = False
        first_hit_side = "none"
        first_hit_offset_bars = -1

    if outcome == "BREAK":
        outcome_score = max_signed_dist
    elif outcome == "REJECT":
        outcome_score = min_signed_dist
    else:
        outcome_score = 0.0

    if approach_direction == 1:
        chop_retrace = confirm_ohlc["low"] <= trigger_ohlc["open"] - cfg.chop_retrace_to_trigger_open_pts
    else:
        chop_retrace = confirm_ohlc["high"] >= trigger_ohlc["open"] + cfg.chop_retrace_to_trigger_open_pts

    chop_flag = chop_retrace

    if chop_retrace and len(lookfwd_candles) > 1:
        second_candle_ts = lookfwd_candles[0]
        df_second = df_horizon[df_horizon["bar2m_ts"] == second_candle_ts]
        if len(df_second) > 0:
            second_ohlc = _get_candle_ohlc(df_second, level_price)
            if approach_direction == 1:
                override = second_ohlc["close"] >= confirm_ohlc["close"] + cfg.chop_override_next_close_delta_pts
            else:
                override = second_ohlc["close"] <= confirm_ohlc["close"] - cfg.chop_override_next_close_delta_pts
            if override:
                chop_flag = False

    if approach_direction == 1:
        failed_break_flag = confirm_ohlc["close"] <= level_price - cfg.failed_break_confirm_close_below_level_pts
    else:
        failed_break_flag = confirm_ohlc["close"] >= level_price + cfg.failed_break_confirm_close_below_level_pts

    entry_ref = level_price
    mae_pts = float(np.min(low_dists)) if len(low_dists) > 0 else 0.0
    mfe_pts = float(np.max(high_dists)) if len(high_dists) > 0 else 0.0

    return {
        "outcome": outcome,
        "outcome_score": outcome_score,
        "max_signed_dist": max_signed_dist,
        "min_signed_dist": min_signed_dist,
        "chop_flag": chop_flag,
        "failed_break_flag": failed_break_flag,
        "both_sides_hit_flag": both_sides_hit_flag,
        "first_hit_side": first_hit_side,
        "first_hit_offset_bars": first_hit_offset_bars,
        "mae_pts": mae_pts,
        "mfe_pts": mfe_pts,
    }


def _build_candle_row(
    df_candle: pd.DataFrame,
    level_price: float,
    approach_direction: int,
    bar2m_ts: int,
    candle_id: int,
    metadata: Dict,
) -> Dict:
    """Build a single candle row with all signature features."""
    trade_cnt = df_candle["bar5s_trade_cnt_sum"].values
    trade_last_px = df_candle["bar5s_trade_last_px"].values
    microprice = df_candle["bar5s_microprice_eob"].values

    trade_close_px = np.where(trade_cnt > 0, trade_last_px, microprice)
    u = approach_direction * ((trade_close_px - level_price) / TICK_SIZE)

    touch_zone_pts = 4 * TICK_SIZE
    in_zone = ((trade_cnt > 0) & (np.abs(trade_last_px - level_price) <= touch_zone_pts)).astype(np.float64)
    touched_in_zone = bool(in_zone.sum() > 0)
    first_touch_offset = int(np.argmax(in_zone)) if touched_in_zone else -1

    ohlc = _get_candle_ohlc(df_candle, level_price)
    close_in_zone = abs(ohlc["close"] - level_price) <= touch_zone_pts
    close_side = int(np.sign((ohlc["close"] - level_price) * approach_direction))

    n = len(u)
    third = max(1, n // 3)
    late_slice = u[2 * third:] if n > 2 * third else u[third:]

    time_in_zone_frac = float(np.mean(in_zone)) if len(in_zone) > 0 else 0.0
    time_far_side_frac = float(np.mean(u > 0)) if n > 0 else 0.0
    late_time_far_side_frac = float(np.mean(late_slice > 0)) if len(late_slice) > 0 else 0.0

    comps = _compute_pressure_components(df_candle, approach_direction)

    row = {
        "bar_ts": int(bar2m_ts),
        "symbol": metadata["symbol"],
        "episode_id": metadata["episode_id"],
        "touch_id": metadata["episode_id"],
        "level_type": metadata["level_type"],
        "level_price": level_price,
        "candle_id": candle_id,
        "trigger_candle_id": metadata["trigger_candle_id"],
        "confirm_candle_id": metadata["confirm_candle_id"],
        "infer_candle_id": metadata["infer_candle_id"],
        "trigger_candle_ts": metadata["trigger_candle_ts"],
        "confirm_candle_ts": metadata["confirm_candle_ts"],
        "infer_ts": metadata["infer_ts"],
        "lookback_start_id": metadata["lookback_start_id"],
        "lookback_end_id": metadata["lookback_end_id"],
        "lookfwd_start_id": metadata["lookfwd_start_id"],
        "lookfwd_end_id": metadata["lookfwd_end_id"],
        "bar_index_in_episode": metadata["bar_index_offset"] + (candle_id - metadata["lookback_start_id"]),
        "bar_index_in_touch": metadata["bar_index_offset"] + (candle_id - metadata["lookback_start_id"]),
        "bars_to_trigger": candle_id - metadata["trigger_candle_id"],
        "bars_to_confirm": candle_id - metadata["confirm_candle_id"],
        "is_pre_trigger": candle_id < metadata["trigger_candle_id"],
        "is_trigger_candle": candle_id == metadata["trigger_candle_id"],
        "is_confirm_candle": candle_id == metadata["confirm_candle_id"],
        "is_post_confirm": candle_id > metadata["confirm_candle_id"],
        "is_in_lookback": metadata["lookback_start_id"] <= candle_id <= metadata["lookback_end_id"],
        "is_in_lookfwd": metadata["lookfwd_start_id"] <= candle_id <= metadata["lookfwd_end_id"],
        "approach_direction": approach_direction,
        "is_standard_approach": metadata["is_standard_approach"],
        "outcome": metadata["outcome"],
        "outcome_score": metadata["outcome_score"],
        "max_signed_dist": metadata["max_signed_dist"],
        "min_signed_dist": metadata["min_signed_dist"],
        "chop_flag": metadata["chop_flag"],
        "failed_break_flag": metadata["failed_break_flag"],
        "both_sides_hit_flag": metadata["both_sides_hit_flag"],
        "first_hit_side": metadata["first_hit_side"],
        "first_hit_offset_bars": metadata["first_hit_offset_bars"],
        "mae_pts": metadata["mae_pts"],
        "mfe_pts": metadata["mfe_pts"],
        "bar2m_open": ohlc["open"],
        "bar2m_high": ohlc["high"],
        "bar2m_low": ohlc["low"],
        "bar2m_close": ohlc["close"],
        "bar2m_touched_in_zone": touched_in_zone,
        "bar2m_close_in_zone": close_in_zone,
        "bar2m_first_touch_offset": first_touch_offset,
        "bar2m_time_in_zone_frac": time_in_zone_frac,
        "bar2m_time_far_side_frac": time_far_side_frac,
        "bar2m_late_time_far_side_frac": late_time_far_side_frac,
        "bar2m_close_side": close_side,
    }

    row.update(_signal_ops(u, 0.0, "bar2m_sig_u"))
    row.update(_signal_ops(comps["pressure"], PRESSURE_BURST_THRESHOLD, "bar2m_sig_pressure"))

    row["bar2m_comp_obi0_lin_mean"] = float(np.mean(comps["obi0_lin"]))
    row["bar2m_comp_obi10_lin_mean"] = float(np.mean(comps["obi10_lin"]))
    row["bar2m_comp_cdi_lin_mean"] = float(np.mean(comps["cdi_lin"]))
    row["bar2m_comp_flow_norm_mean"] = float(np.mean(comps["flow_norm"]))
    row["bar2m_comp_trade_imbal_mean"] = float(np.mean(comps["trade_imbal"]))
    row["bar2m_comp_wall_support_mean"] = float(np.mean(comps["wall_support"]))
    row["bar2m_comp_wall_dist_support_mean"] = float(np.mean(comps["wall_dist_support"]))
    row["bar2m_comp_gap_spread_mean"] = float(np.mean(comps["gap_spread"]))
    row["bar2m_comp_trade_activity_mean"] = float(np.mean(comps["trade_activity"]))

    cfg_dict = metadata["cfg"].to_dict()
    for k, v in cfg_dict.items():
        row[f"cfg_{k}"] = v

    return row


def compute_level_approach2m(
    df_bar5s: pd.DataFrame,
    level_price: float,
    level_type: str,
    dt: str,
    symbol: str,
    cfg: SetupConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute 2-minute candle level approach dataset per COMPLIANCE spec."""
    if len(df_bar5s) == 0 or np.isnan(level_price):
        return pd.DataFrame()

    df = df_bar5s.sort_values("bar_ts").reset_index(drop=True)
    df["bar2m_ts"] = _compute_bar2m_ts(df["bar_ts"].values)

    rth_start_ns, rth_end_ns = _rth_bounds_ns(dt)

    microprice = df["bar5s_microprice_eob"].values
    first_price = microprice[0] if len(microprice) > 0 else level_price
    approach_direction = 1 if first_price < level_price else -1

    is_standard = (
        (level_type.endswith("high") and approach_direction == 1) or
        (level_type.endswith("low") and approach_direction == -1)
    )

    candle_ts_list = sorted(df["bar2m_ts"].unique())

    triggers = _detect_triggers(
        df, candle_ts_list, level_price, approach_direction, cfg, rth_start_ns, rth_end_ns
    )

    if not triggers:
        return pd.DataFrame()

    all_rows: List[Dict] = []

    for trigger_idx, trigger_ts, confirm_ts in triggers:
        confirm_idx = candle_ts_list.index(confirm_ts) if confirm_ts in candle_ts_list else trigger_idx + cfg.confirm_bars
        infer_idx = confirm_idx
        infer_ts = confirm_ts

        lookback_end_id = confirm_idx
        lookback_start_id = max(0, confirm_idx - cfg.lookback_bars_infer + 1)

        lookfwd_start_id = confirm_idx + 1
        lookfwd_end_id = min(confirm_idx + cfg.lookfwd_bars_label, len(candle_ts_list) - 1)

        df_trigger = df[df["bar2m_ts"] == trigger_ts]
        trigger_ohlc = _get_candle_ohlc(df_trigger, level_price) if len(df_trigger) > 0 else {"open": level_price, "high": level_price, "low": level_price, "close": level_price}

        df_confirm = df[df["bar2m_ts"] == confirm_ts]
        confirm_ohlc = _get_candle_ohlc(df_confirm, level_price) if len(df_confirm) > 0 else {"open": level_price, "high": level_price, "low": level_price, "close": level_price}

        outcome_data = _compute_outcome_and_flags(
            df, candle_ts_list, confirm_idx, level_price, approach_direction, trigger_ohlc, confirm_ohlc, cfg
        )

        episode_id = f"{dt}_{symbol}_{level_type}_{trigger_ts}"

        metadata = {
            "symbol": symbol,
            "episode_id": episode_id,
            "level_type": level_type,
            "is_standard_approach": is_standard,
            "trigger_candle_id": trigger_idx,
            "confirm_candle_id": confirm_idx,
            "infer_candle_id": infer_idx,
            "trigger_candle_ts": trigger_ts,
            "confirm_candle_ts": confirm_ts,
            "infer_ts": infer_ts,
            "lookback_start_id": lookback_start_id,
            "lookback_end_id": lookback_end_id,
            "lookfwd_start_id": lookfwd_start_id,
            "lookfwd_end_id": lookfwd_end_id,
            "bar_index_offset": 0,
            "cfg": cfg,
            **outcome_data,
        }

        for candle_idx in range(lookback_start_id, min(lookfwd_end_id + 1, len(candle_ts_list))):
            bar2m_ts = candle_ts_list[candle_idx]
            df_candle = df[df["bar2m_ts"] == bar2m_ts]
            if len(df_candle) == 0:
                continue

            row = _build_candle_row(
                df_candle, level_price, approach_direction, bar2m_ts, candle_idx, metadata
            )
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)
