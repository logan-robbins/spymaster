"""
Compute confluence features stage (v2.0+).

DISABLED FOR V1:
- Confluence features are over-engineered for v1
- Composite pressure metrics don't define similarity space
- Keep raw physics features only

This stage is stubbed to maintain pipeline compatibility but returns
minimal/zero-valued features.
"""
from typing import Any, Dict, List, Optional, Tuple
from datetime import time as dt_time
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.duckdb_reader import DuckDBReader
from src.common.config import CONFIG


def compute_confluence_features_dynamic(
    signals_df: pd.DataFrame,
    dynamic_levels: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    DISABLED FOR V1: Return minimal stub features.
    
    Drop confluence features for v1.
    This stub maintains pipeline compatibility.
    """
    if signals_df.empty:
        return signals_df

    # DISABLED FOR V1 - return zeros/nulls
    # Original complex confluence computation removed per spec

    dynamic_arrays = {
        name: series.to_numpy(dtype=np.float64)
        for name, series in dynamic_levels.items()
        if name in key_weights
    }

    if not dynamic_arrays:
        result = signals_df.copy()
        result['confluence_count'] = 0
        result['confluence_weighted_score'] = 0.0
        result['confluence_min_distance'] = np.nan
        result['confluence_pressure'] = 0.0
        return result

    band = CONFIG.CONFLUENCE_BAND
    eps = 1e-6

    n = len(signals_df)
    confluence_count = np.zeros(n, dtype=np.int32)
    confluence_weighted = np.zeros(n, dtype=np.float64)
    confluence_min_dist = np.full(n, np.nan, dtype=np.float64)
    confluence_pressure = np.zeros(n, dtype=np.float64)

    bar_idx = signals_df['bar_idx'].values.astype(np.int64)
    level_prices = signals_df['level_price'].values.astype(np.float64)
    level_kinds = signals_df['level_kind_name'].values.astype(object)

    total_weight = sum(key_weights.values())

    for i in range(n):
        idx = bar_idx[i]
        level_price = level_prices[i]
        level_kind = level_kinds[i]

        distances = []
        weights = []
        for name, arr in dynamic_arrays.items():
            if idx < 0 or idx >= len(arr):
                continue
            val = arr[idx]
            if not np.isfinite(val):
                continue
            if name == level_kind and abs(val - level_price) < eps:
                continue
            dist = abs(val - level_price)
            distances.append(dist)
            weights.append(key_weights[name])

        if not distances:
            continue

        distances = np.array(distances, dtype=np.float64)
        weights_arr = np.array(weights, dtype=np.float64)
        confluence_min_dist[i] = float(np.min(distances))
        within = distances <= band
        if not np.any(within):
            continue

        distance_decay = np.clip(1.0 - (distances[within] / band), 0.0, 1.0)
        confluence_count[i] = int(np.sum(within))
        confluence_weighted[i] = float(np.sum(weights_arr[within] * distance_decay))
        if total_weight > 0:
            confluence_pressure[i] = confluence_weighted[i] / total_weight

    result = signals_df.copy()
    result['confluence_count'] = confluence_count
    result['confluence_weighted_score'] = confluence_weighted
    result['confluence_min_distance'] = confluence_min_dist
    result['confluence_pressure'] = confluence_pressure
    return result


def compute_confluence_alignment(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute confluence alignment based on SMA position/slope and approach direction."""
    if signals_df.empty:
        return signals_df

    level_prices = signals_df['level_price'].values.astype(np.float64)
    directions = signals_df['direction'].values.astype(object)
    sma_200 = signals_df.get('sma_200')
    sma_400 = signals_df.get('sma_400')
    sma_200_slope = signals_df.get('sma_200_slope')
    sma_400_slope = signals_df.get('sma_400_slope')

    if sma_200 is None or sma_400 is None or sma_200_slope is None or sma_400_slope is None:
        result = signals_df.copy()
        result['confluence_alignment'] = np.zeros(len(signals_df), dtype=np.int8)
        return result

    sma_200_vals = sma_200.values.astype(np.float64)
    sma_400_vals = sma_400.values.astype(np.float64)
    sma_200_slope_vals = sma_200_slope.values.astype(np.float64)
    sma_400_slope_vals = sma_400_slope.values.astype(np.float64)

    alignment = np.zeros(len(signals_df), dtype=np.int8)
    for i in range(len(signals_df)):
        if not (np.isfinite(sma_200_vals[i]) and np.isfinite(sma_400_vals[i])):
            continue
        if not (np.isfinite(sma_200_slope_vals[i]) and np.isfinite(sma_400_slope_vals[i])):
            continue

        below_both = level_prices[i] < sma_200_vals[i] and level_prices[i] < sma_400_vals[i]
        above_both = level_prices[i] > sma_200_vals[i] and level_prices[i] > sma_400_vals[i]
        slopes_negative = sma_200_slope_vals[i] < 0 and sma_400_slope_vals[i] < 0
        slopes_positive = sma_200_slope_vals[i] > 0 and sma_400_slope_vals[i] > 0
        spread_slope = sma_200_slope_vals[i] - sma_400_slope_vals[i]

        if directions[i] == 'UP':
            if below_both and slopes_negative and spread_slope > 0:
                alignment[i] = 1
            elif above_both and slopes_positive and spread_slope < 0:
                alignment[i] = -1
        else:
            if above_both and slopes_positive and spread_slope > 0:
                alignment[i] = 1
            elif below_both and slopes_negative and spread_slope < 0:
                alignment[i] = -1

    result = signals_df.copy()
    result['confluence_alignment'] = alignment
    return result


def compute_dealer_velocity_features(
    signals_df: pd.DataFrame,
    option_trades_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute dealer gamma flow velocity and acceleration features from option trades."""
    if signals_df.empty:
        return signals_df

    n = len(signals_df)
    gamma_flow_velocity = np.zeros(n, dtype=np.float64)
    gamma_flow_impulse = np.zeros(n, dtype=np.float64)
    gamma_flow_accel_1m = np.zeros(n, dtype=np.float64)
    gamma_flow_accel_3m = np.zeros(n, dtype=np.float64)
    dealer_pressure = np.zeros(n, dtype=np.float64)
    dealer_pressure_accel = np.zeros(n, dtype=np.float64)

    if option_trades_df is None or option_trades_df.empty:
        result = signals_df.copy()
        result['gamma_flow_velocity'] = gamma_flow_velocity
        result['gamma_flow_impulse'] = gamma_flow_impulse
        result['gamma_flow_accel_1m'] = gamma_flow_accel_1m
        result['gamma_flow_accel_3m'] = gamma_flow_accel_3m
        result['dealer_pressure'] = dealer_pressure
        result['dealer_pressure_accel'] = dealer_pressure_accel
        return result

    required_cols = ['ts_event_ns', 'strike', 'size', 'gamma', 'aggressor']
    if not set(required_cols).issubset(option_trades_df.columns):
        result = signals_df.copy()
        result['gamma_flow_velocity'] = gamma_flow_velocity
        result['gamma_flow_impulse'] = gamma_flow_impulse
        result['gamma_flow_accel_1m'] = gamma_flow_accel_1m
        result['gamma_flow_accel_3m'] = gamma_flow_accel_3m
        result['dealer_pressure'] = dealer_pressure
        result['dealer_pressure_accel'] = dealer_pressure_accel
        return result

    opt_df = option_trades_df[required_cols].copy()
    opt_df['aggressor'] = pd.to_numeric(opt_df['aggressor'], errors='coerce').fillna(0).astype(np.int8)
    opt_df['size'] = pd.to_numeric(opt_df['size'], errors='coerce').fillna(0).astype(np.float64)
    opt_df['gamma'] = pd.to_numeric(opt_df['gamma'], errors='coerce').fillna(0).astype(np.float64)
    opt_df['strike'] = pd.to_numeric(opt_df['strike'], errors='coerce').fillna(0).astype(np.float64)
    opt_df['ts_event_ns'] = pd.to_numeric(opt_df['ts_event_ns'], errors='coerce').fillna(0).astype(np.int64)

    # Dealer gamma flow: dealer takes opposite of customer aggressor
    dealer_flow = -opt_df['aggressor'].values * opt_df['size'].values * opt_df['gamma'].values * 100.0
    opt_ts = opt_df['ts_event_ns'].values
    opt_strike = opt_df['strike'].values

    sort_idx = np.argsort(opt_ts)
    opt_ts = opt_ts[sort_idx]
    opt_strike = opt_strike[sort_idx]
    dealer_flow = dealer_flow[sort_idx]

    window_minutes = max(1, CONFIG.DEALER_FLOW_WINDOW_MINUTES)
    baseline_minutes = max(window_minutes + 1, CONFIG.DEALER_FLOW_BASELINE_MINUTES)
    window_ns = int(window_minutes * 60 * 1e9)
    baseline_ns = int(baseline_minutes * 60 * 1e9)
    accel_short_minutes = max(1, CONFIG.DEALER_FLOW_ACCEL_SHORT_MINUTES)
    accel_long_minutes = max(accel_short_minutes, CONFIG.DEALER_FLOW_ACCEL_LONG_MINUTES)
    accel_short_ns = int(accel_short_minutes * 60 * 1e9)
    accel_long_ns = int(accel_long_minutes * 60 * 1e9)
    strike_range = CONFIG.DEALER_FLOW_STRIKE_RANGE

    signal_ts = signals_df['ts_ns'].values.astype(np.int64)
    level_prices = signals_df['level_price'].values.astype(np.float64)

    def flow_in_window(end_ts: int, window: int, level_price: float) -> float:
        start = np.searchsorted(opt_ts, end_ts - window, side='left')
        end = np.searchsorted(opt_ts, end_ts, side='right')
        if end <= start:
            return 0.0
        strikes = opt_strike[start:end]
        mask = np.abs(strikes - level_price) <= strike_range
        if not np.any(mask):
            return 0.0
        return float(np.sum(dealer_flow[start:end][mask]))

    for i in range(n):
        ts = signal_ts[i]
        level = level_prices[i]

        short_flow = flow_in_window(ts, window_ns, level)
        short_vel = short_flow / window_minutes
        gamma_flow_velocity[i] = short_vel

        base_flow = flow_in_window(ts, baseline_ns, level)
        baseline_rate = base_flow / baseline_minutes if baseline_minutes > 0 else 0.0

        if abs(baseline_rate) > 1e-6:
            gamma_flow_impulse[i] = (short_vel - baseline_rate) / abs(baseline_rate)
        else:
            gamma_flow_impulse[i] = 0.0

        # Acceleration (1m and 3m windows)
        short_accel_flow = flow_in_window(ts, accel_short_ns, level)
        short_accel_prev = flow_in_window(ts - accel_short_ns, accel_short_ns, level)
        short_accel_vel = short_accel_flow / accel_short_minutes
        short_accel_prev_vel = short_accel_prev / accel_short_minutes
        gamma_flow_accel_1m[i] = short_accel_vel - short_accel_prev_vel

        long_accel_flow = flow_in_window(ts, accel_long_ns, level)
        long_accel_prev = flow_in_window(ts - accel_long_ns, accel_long_ns, level)
        long_accel_vel = long_accel_flow / accel_long_minutes
        long_accel_prev_vel = long_accel_prev / accel_long_minutes
        gamma_flow_accel_3m[i] = long_accel_vel - long_accel_prev_vel

        dealer_pressure[i] = np.tanh(-short_vel / CONFIG.GAMMA_FLOW_NORM)
        dealer_pressure_accel[i] = np.tanh(-gamma_flow_accel_1m[i] / CONFIG.GAMMA_FLOW_ACCEL_NORM)

    result = signals_df.copy()
    result['gamma_flow_velocity'] = gamma_flow_velocity
    result['gamma_flow_impulse'] = gamma_flow_impulse
    result['gamma_flow_accel_1m'] = gamma_flow_accel_1m
    result['gamma_flow_accel_3m'] = gamma_flow_accel_3m
    result['dealer_pressure'] = dealer_pressure
    result['dealer_pressure_accel'] = dealer_pressure_accel

    return result


def compute_pressure_indicators(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute continuous pressure indicators for break/bounce strength."""
    if signals_df.empty:
        return signals_df

    direction_sign = np.where(signals_df['direction'].values == 'UP', 1.0, -1.0)
    barrier_delta = np.nan_to_num(signals_df['barrier_delta_liq'].values.astype(np.float64), nan=0.0)
    wall_ratio = np.nan_to_num(signals_df['wall_ratio'].values.astype(np.float64), nan=0.0)
    tape_imbalance = np.nan_to_num(signals_df['tape_imbalance'].values.astype(np.float64), nan=0.0)
    tape_velocity = np.nan_to_num(signals_df['tape_velocity'].values.astype(np.float64), nan=0.0)
    gamma_exposure = np.nan_to_num(signals_df['gamma_exposure'].values.astype(np.float64), nan=0.0)

    delta_norm = np.tanh(-barrier_delta / CONFIG.BARRIER_DELTA_LIQ_NORM)
    wall_effect = np.clip(1.0 - (wall_ratio / CONFIG.WALL_RATIO_NORM), -1.0, 1.0)
    liquidity_pressure = np.clip((delta_norm + wall_effect) / 2.0, -1.0, 1.0)

    velocity_norm = np.clip(tape_velocity / CONFIG.TAPE_VELOCITY_NORM, -1.0, 1.0)
    tape_pressure = np.clip(direction_sign * (tape_imbalance + velocity_norm) / 2.0, -1.0, 1.0)

    gamma_pressure = np.tanh(-gamma_exposure / CONFIG.GAMMA_EXPOSURE_NORM)
    if 'gamma_flow_accel_3m' in signals_df.columns:
        gamma_pressure_accel = np.tanh(
            -signals_df['gamma_flow_accel_3m'].values.astype(np.float64) / CONFIG.GAMMA_FLOW_ACCEL_NORM
        )
    else:
        gamma_pressure_accel = np.zeros(len(signals_df), dtype=np.float64)

    mean_rev_200 = signals_df.get('mean_reversion_pressure_200')
    mean_rev_400 = signals_df.get('mean_reversion_pressure_400')
    if mean_rev_200 is None and mean_rev_400 is None:
        mean_rev_combined = np.zeros(len(signals_df), dtype=np.float64)
    else:
        mr_200 = mean_rev_200.values.astype(np.float64) if mean_rev_200 is not None else np.full(len(signals_df), np.nan)
        mr_400 = mean_rev_400.values.astype(np.float64) if mean_rev_400 is not None else np.full(len(signals_df), np.nan)
        both = np.isfinite(mr_200) & np.isfinite(mr_400)
        only_200 = np.isfinite(mr_200) & ~np.isfinite(mr_400)
        only_400 = np.isfinite(mr_400) & ~np.isfinite(mr_200)
        mean_rev_combined = np.zeros(len(signals_df), dtype=np.float64)
        mean_rev_combined[both] = (mr_200[both] + mr_400[both]) / 2.0
        mean_rev_combined[only_200] = mr_200[only_200]
        mean_rev_combined[only_400] = mr_400[only_400]

    reversion_pressure = np.tanh(-direction_sign * mean_rev_combined)

    if 'confluence_pressure' in signals_df.columns:
        confluence_pressure = np.nan_to_num(
            signals_df['confluence_pressure'].values.astype(np.float64),
            nan=0.0
        )
    else:
        confluence_pressure = np.zeros(len(signals_df), dtype=np.float64)

    net_break_pressure = np.clip(
        (liquidity_pressure + tape_pressure + gamma_pressure + reversion_pressure + confluence_pressure) / 5.0,
        -1.0,
        1.0
    )

    result = signals_df.copy()
    result['liquidity_pressure'] = liquidity_pressure
    result['tape_pressure'] = tape_pressure
    result['gamma_pressure'] = gamma_pressure
    result['gamma_pressure_accel'] = gamma_pressure_accel
    result['reversion_pressure'] = reversion_pressure
    result['net_break_pressure'] = net_break_pressure

    return result


def compute_confluence_level_features(
    signals_df: pd.DataFrame,
    dynamic_levels_df: pd.DataFrame,
    hourly_cumvol: Dict[str, Dict[int, float]],
    date: str
) -> pd.DataFrame:
    """Compute hierarchical confluence level (1-10) based on 5 dimensions."""
    if signals_df.empty:
        result = signals_df.copy()
        result['confluence_level'] = np.int8(0)
        result['rel_vol_ratio'] = np.nan
        result['gex_alignment'] = np.int8(0)
        result['breakout_state'] = np.int8(0)
        return result

    n = len(signals_df)
    result = signals_df.copy()

    # Extract required columns
    spot_vals = result['anchor_spot'].values.astype(np.float64) if 'anchor_spot' in result.columns else result['level_price'].values.astype(np.float64)
    ts_ns = result['ts_ns'].values.astype(np.int64)

    # Get time in ET
    ts_dt = pd.to_datetime(ts_ns, unit='ns', utc=True)
    time_et = ts_dt.tz_convert('America/New_York')
    hour_et = time_et.hour.values

    # Initialize output arrays
    confluence_level = np.zeros(n, dtype=np.int8)
    rel_vol_ratio = np.full(n, np.nan, dtype=np.float64)
    gex_alignment = np.zeros(n, dtype=np.int8)
    breakout_state = np.zeros(n, dtype=np.int8)

    # Compute relative volume ratio
    prior_dates = [d for d in hourly_cumvol.keys() if d < date]
    for i in range(n):
        hour = hour_et[i]
        if hour < 9 or hour > 15:
            continue

        if date not in hourly_cumvol or hour not in hourly_cumvol[date]:
            continue
        cumvol_now = hourly_cumvol[date][hour]

        prior_cumvols = [
            hourly_cumvol[d][hour]
            for d in prior_dates
            if d in hourly_cumvol and hour in hourly_cumvol[d]
        ]
        if prior_cumvols:
            avg_cumvol = np.mean(prior_cumvols)
            if avg_cumvol > 0:
                rel_vol_ratio[i] = cumvol_now / avg_cumvol

    result['confluence_level'] = confluence_level
    result['rel_vol_ratio'] = rel_vol_ratio
    result['gex_alignment'] = gex_alignment
    result['breakout_state'] = breakout_state

    return result


class ComputeConfluenceStage(BaseStage):
    """
    DISABLED FOR V1: Confluence features stubbed out.
    
    Confluence computation (disabled for v1):
    - Drop confluence features (over-engineered)
    - Drop dealer velocity features
    - Drop pressure indicators
    - Drop attempt clustering
    
    This stage maintained for pipeline compatibility but returns pass-through.
    """

    @property
    def name(self) -> str:
        return "compute_confluence"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        
        # DISABLED FOR V1: All confluence/pressure features removed
        # Return pass-through with stub columns for compatibility
        
        # Minimal stub columns (zeros) to prevent downstream breakage
        signals_df['confluence_level'] = 0
        signals_df['breakout_state'] = 0
        signals_df['gex_alignment'] = 0
        
        return {'signals_df': signals_df}

    def _build_hourly_cumvol_table(
        self, ctx: StageContext, ohlcv_df: pd.DataFrame
    ) -> Dict[str, Dict[int, float]]:
        """Build hourly cumulative volume table for relative volume."""
        hourly_cumvol: Dict[str, Dict[int, float]] = {}

        def _compute_hourly_cumvol(ohlcv: pd.DataFrame, date_str: str) -> Dict[int, float]:
            if ohlcv.empty or 'timestamp' not in ohlcv.columns:
                return {}

            df = ohlcv.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York')
            df['hour_et'] = df['time_et'].dt.hour

            # Filter to RTH (9:30-16:00)
            rth_mask = (
                (df['time_et'].dt.time >= dt_time(9, 30)) &
                (df['time_et'].dt.time < dt_time(16, 0))
            )
            df = df[rth_mask]

            if df.empty:
                return {}

            # Compute cumulative volume up to end of each hour
            hourly = {}
            for hour in [9, 10, 11, 12, 13, 14, 15]:
                if hour == 9:
                    hour_end = dt_time(9, 59, 59)
                else:
                    hour_end = dt_time(hour, 59, 59)

                mask = df['time_et'].dt.time <= hour_end
                if mask.any():
                    hourly[hour] = df.loc[mask, 'volume'].sum()

            return hourly

        # Get prior dates for lookback
        reader = ctx.data.get('_reader')
        if reader is None:
            reader = DuckDBReader()

        warmup_days = max(0, CONFIG.VOLUME_LOOKBACK_DAYS)
        prior_dates = reader.get_warmup_dates(ctx.date, warmup_days) if warmup_days > 0 else []

        from src.pipeline.stages.build_ohlcv import build_ohlcv
        from src.pipeline.stages.load_bronze import futures_trades_from_df

        for prior_date in prior_dates:
            trades_df = reader.read_futures_trades(symbol='ES', date=prior_date)
            trades = futures_trades_from_df(trades_df)
            if not trades:
                continue
            ohlcv = build_ohlcv(trades, convert_to_spy=True, freq='1min')
            if not ohlcv.empty:
                hourly_cumvol[prior_date] = _compute_hourly_cumvol(ohlcv, prior_date)

        # Add current date
        if not ohlcv_df.empty:
            hourly_cumvol[ctx.date] = _compute_hourly_cumvol(ohlcv_df, ctx.date)

        return hourly_cumvol
