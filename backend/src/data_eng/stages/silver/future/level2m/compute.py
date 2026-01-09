"""2-minute candle level approach computation.

Takes bar5s data and level price, produces approach2m rows matching the contract.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

EPSILON = 1e-9
TICK_SIZE = 0.25

TOUCH_ZONE_TICKS = 4
CLOSE_ZONE_TICKS = 4
TOUCH_ZONE_PTS = TOUCH_ZONE_TICKS * TICK_SIZE
CLOSE_ZONE_PTS = CLOSE_ZONE_TICKS * TICK_SIZE

MOVE_THRESHOLD_TICKS = 8
OUTCOME_HORIZON_CANDLES = 6
PRE_WINDOW_CANDLES = 5
COOLDOWN_CANDLES = 3

BAR2M_DURATION_NS = 120_000_000_000

RTH_START_HOUR = 9
RTH_START_MINUTE = 30
RTH_HOURS = 3

PRESSURE_BURST_THRESHOLD = 0.5
MAX_WALL_DIST_TICKS = 10.0
GAP_SPREAD_SCALE = 4.0


def _linearize_bounded(x: np.ndarray) -> np.ndarray:
    """Linearize bounded [-1, 1] signals using arctanh."""
    clipped = np.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return np.arctanh(clipped) / 2.0


def _signal_ops(values: np.ndarray, threshold: float, prefix: str) -> Dict[str, float]:
    """Compute standard operators on a micro signal array."""
    vals = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(vals)

    if n == 0:
        return {
            f"{prefix}_start": 0.0,
            f"{prefix}_end": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_slope": 0.0,
            f"{prefix}_energy": 0.0,
            f"{prefix}_sign_flip_cnt": 0.0,
            f"{prefix}_burst_frac": 0.0,
            f"{prefix}_mean_early": 0.0,
            f"{prefix}_mean_mid": 0.0,
            f"{prefix}_mean_late": 0.0,
            f"{prefix}_energy_early": 0.0,
            f"{prefix}_energy_late": 0.0,
            f"{prefix}_late_minus_early": 0.0,
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


def _compute_pressure_components(df: pd.DataFrame, approach_direction: int) -> Dict[str, np.ndarray]:
    """Compute pressure components from bar5s micro data."""
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

    trade_cnt = df["bar5s_trade_cnt_sum"].values
    trade_activity = np.tanh(np.log1p(trade_cnt) / 3.0)

    for arr in [obi0_lin, obi10_lin, cdi_lin, flow_norm, trade_imbal,
                wall_support, wall_dist_support, gap_spread, trade_activity]:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    pressure = np.mean(
        np.stack([
            (obi0_lin + obi10_lin + cdi_lin) / 3.0,
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
    """Get RTH start/end timestamps in nanoseconds."""
    tz = ZoneInfo("America/New_York")
    date_obj = pd.Timestamp(dt, tz=tz)
    start = date_obj.replace(hour=RTH_START_HOUR, minute=RTH_START_MINUTE, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=RTH_HOURS)
    start_ns = int(start.tz_convert("UTC").value)
    end_ns = int(end.tz_convert("UTC").value)
    return start_ns, end_ns


def _compute_bar2m_ts(bar_ts: np.ndarray) -> np.ndarray:
    """Compute 2-minute candle timestamps aligned to NY trading day."""
    return (bar_ts // BAR2M_DURATION_NS) * BAR2M_DURATION_NS


def _compute_approach_direction(df: pd.DataFrame, level_price: float) -> int:
    """Compute approach direction: +1 from below, -1 from above."""
    microprice = df["bar5s_microprice_eob"].values
    first_price = microprice[0] if len(microprice) > 0 else level_price
    return 1 if first_price < level_price else -1


def _compute_outcome(
    df: pd.DataFrame,
    trigger_candle_ts: int,
    level_price: float,
    approach_direction: int,
) -> Tuple[str, float]:
    """Compute outcome label and score."""
    df_post = df[df["bar2m_ts"] > trigger_candle_ts].copy()
    candle_ts_unique = df_post["bar2m_ts"].unique()
    horizon_candles = sorted(candle_ts_unique)[:OUTCOME_HORIZON_CANDLES]

    if len(horizon_candles) == 0:
        return "CHOP", 0.0

    df_horizon = df_post[df_post["bar2m_ts"].isin(horizon_candles)]
    if len(df_horizon) == 0:
        return "CHOP", 0.0

    trade_cnt = df_horizon["bar5s_trade_cnt_sum"].values
    trade_last_px = df_horizon["bar5s_trade_last_px"].values
    microprice = df_horizon["bar5s_microprice_eob"].values
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


def _build_candle_row(
    df_candle: pd.DataFrame,
    level_price: float,
    approach_direction: int,
    bar2m_ts: int,
    bar_index: int,
    metadata: Dict,
) -> Dict:
    """Build a single candle row with all signature features."""
    trade_cnt = df_candle["bar5s_trade_cnt_sum"].values
    trade_last_px = df_candle["bar5s_trade_last_px"].values
    trade_last_ts = df_candle["bar5s_trade_last_ts"].values
    microprice = df_candle["bar5s_microprice_eob"].values

    trade_close_px = np.where(trade_cnt > 0, trade_last_px, microprice)
    u = approach_direction * ((trade_close_px - level_price) / TICK_SIZE)

    in_zone = ((trade_cnt > 0) & (np.abs(trade_last_px - level_price) <= TOUCH_ZONE_PTS)).astype(np.float64)
    touched_in_zone = bool(in_zone.sum() > 0)
    first_touch_offset = int(np.argmax(in_zone)) if touched_in_zone else -1

    has_trade = trade_last_ts >= 0
    if np.any(has_trade):
        last_idx = int(np.argmax(trade_last_ts))
        close_px = float(trade_last_px[last_idx])
        close_in_zone = abs(close_px - level_price) <= CLOSE_ZONE_PTS
        close_side = int(np.sign((close_px - level_price) * approach_direction))
    else:
        close_in_zone = False
        close_side = 0

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
        "bar_index_in_episode": bar_index,
        "bar_index_in_touch": bar_index,
        "approach_direction": approach_direction,
        "is_standard_approach": metadata["is_standard_approach"],
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

    return row


def compute_level_approach2m(
    df_bar5s: pd.DataFrame,
    level_price: float,
    level_type: str,
    dt: str,
    symbol: str,
) -> pd.DataFrame:
    """Compute 2-minute candle level approach dataset.

    Args:
        df_bar5s: DataFrame of 5s bars
        level_price: The level price
        level_type: pm_high, pm_low, or_high, or_low
        dt: Date string
        symbol: Symbol

    Returns:
        DataFrame matching market_by_price_10_level_approach2m contract
    """
    if len(df_bar5s) == 0 or np.isnan(level_price):
        return pd.DataFrame()

    df = df_bar5s.sort_values("bar_ts").reset_index(drop=True)
    df["bar2m_ts"] = _compute_bar2m_ts(df["bar_ts"].values)

    rth_start_ns, rth_end_ns = _rth_bounds_ns(dt)
    approach_direction = _compute_approach_direction(df, level_price)
    is_standard = (
        (level_type.endswith("high") and approach_direction == 1) or
        (level_type.endswith("low") and approach_direction == -1)
    )

    candle_ts_list = sorted(df["bar2m_ts"].unique())
    candle_triggers: List[Tuple[int, int]] = []
    last_trigger_idx = -COOLDOWN_CANDLES - 1

    for idx, bar2m_ts in enumerate(candle_ts_list):
        if idx <= last_trigger_idx + COOLDOWN_CANDLES:
            continue
        if bar2m_ts < rth_start_ns or bar2m_ts >= rth_end_ns:
            continue

        df_candle = df[df["bar2m_ts"] == bar2m_ts]
        trade_cnt = df_candle["bar5s_trade_cnt_sum"].values
        trade_last_px = df_candle["bar5s_trade_last_px"].values
        trade_last_ts = df_candle["bar5s_trade_last_ts"].values

        in_zone = ((trade_cnt > 0) & (np.abs(trade_last_px - level_price) <= TOUCH_ZONE_PTS))
        touched_in_zone = bool(in_zone.sum() > 0)

        has_trade = trade_last_ts >= 0
        if np.any(has_trade):
            last_idx = int(np.argmax(trade_last_ts))
            close_px = float(trade_last_px[last_idx])
            close_in_zone = abs(close_px - level_price) <= CLOSE_ZONE_PTS
        else:
            close_in_zone = False

        if touched_in_zone and close_in_zone:
            candle_triggers.append((idx, int(bar2m_ts)))
            last_trigger_idx = idx

    if not candle_triggers:
        return pd.DataFrame()

    all_rows: List[Dict] = []

    for trigger_idx, trigger_candle_ts in candle_triggers:
        episode_id = f"{dt}_{symbol}_{level_type}_{trigger_candle_ts}"

        outcome, outcome_score = _compute_outcome(
            df, trigger_candle_ts, level_price, approach_direction
        )

        pre_start_idx = max(0, trigger_idx - PRE_WINDOW_CANDLES + 1)
        post_end_idx = min(trigger_idx + OUTCOME_HORIZON_CANDLES + 1, len(candle_ts_list))
        is_truncated = (trigger_idx - pre_start_idx) < (PRE_WINDOW_CANDLES - 1)

        metadata = {
            "symbol": symbol,
            "episode_id": episode_id,
            "level_type": level_type,
            "is_standard_approach": is_standard,
        }

        for window_idx in range(pre_start_idx, post_end_idx):
            bar2m_ts = candle_ts_list[window_idx]
            bars_to_trigger = window_idx - trigger_idx

            df_candle = df[df["bar2m_ts"] == bar2m_ts]
            if len(df_candle) == 0:
                continue

            row = _build_candle_row(
                df_candle, level_price, approach_direction,
                bar2m_ts, window_idx, metadata
            )

            row["trigger_candle_ts"] = trigger_candle_ts
            row["bars_to_trigger"] = bars_to_trigger
            row["is_pre_trigger"] = bars_to_trigger < 0
            row["is_trigger_candle"] = bars_to_trigger == 0
            row["is_post_trigger"] = bars_to_trigger > 0
            row["outcome"] = outcome
            row["outcome_score"] = outcome_score
            row["is_premarket_context_truncated"] = is_truncated

            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)
