"""Compute approach context features stage (v2.0+)."""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_approach_context(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    lookback_minutes: int = None,
    date_override: str = None
) -> pd.DataFrame:
    """
    Compute backward-looking approach context features.

    Captures how price approached the level - critical for ML to understand
    the setup leading into the touch.

    Features computed:
        - approach_velocity: Price change per minute over lookback window ($/min)
        - approach_bars: Number of consecutive bars moving toward level
        - approach_distance: Total price distance traveled toward level
        - prior_touches: Count of previous touches at this level today
        - bars_since_open: Session timing context

    Args:
        signals_df: DataFrame with signals
        ohlcv_df: OHLCV DataFrame
        lookback_minutes: Backward window (defaults to CONFIG.LOOKBACK_MINUTES)

    Returns:
        DataFrame with approach context features added
    """
    if lookback_minutes is None:
        lookback_minutes = CONFIG.LOOKBACK_MINUTES

    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    required_cols = {'ts_ns', 'level_price', 'direction'}
    missing = required_cols - set(signals_df.columns)
    if missing:
        raise ValueError(f"signals_df missing required columns for approach context: {sorted(missing)}")

    # Prepare OHLCV data for fast lookup
    ohlcv = ohlcv_df.copy()
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        ohlcv = ohlcv.reset_index()
        if 'timestamp' not in ohlcv.columns:
            ohlcv = ohlcv.rename(columns={'index': 'timestamp'})

    if 'timestamp' not in ohlcv.columns:
        raise ValueError("ohlcv_df must have DatetimeIndex or 'timestamp' column")

    ohlcv_sorted = ohlcv.sort_values('timestamp')
    ohlcv_ts = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    ohlcv_open = ohlcv_sorted['open'].values.astype(np.float64)

    # Lookback in nanoseconds
    lookback_ns = int(lookback_minutes * 60 * 1e9)

    n = len(signals_df)
    approach_velocity = np.zeros(n, dtype=np.float64)
    approach_bars = np.zeros(n, dtype=np.int32)
    approach_distance = np.zeros(n, dtype=np.float64)
    bars_since_open = np.zeros(n, dtype=np.int32)
    minutes_since_open = np.zeros(n, dtype=np.float32)
    

    signal_ts = signals_df['ts_ns'].values.astype(np.int64)
    level_prices = signals_df['level_price'].values.astype(np.float64)
    directions = signals_df['direction'].values
    entry_prices = signals_df['entry_price'].values.astype(np.float64) if 'entry_price' in signals_df.columns else None

    # Import session timing utilities
    from src.common.utils.session_time import (
        compute_minutes_since_open,
        compute_bars_since_open,
        get_session_start_ns
    )
    
    # Get date (prioritize override > column > infer)
    if date_override:
        date_str = date_override
    elif 'date' in signals_df.columns:
        # Get from first row if it's a column
        date_str = str(signals_df['date'].iloc[0])
    else:
        # Infer from first timestamp
        first_ts_dt = pd.Timestamp(signal_ts[0], unit='ns', tz='UTC')
        date_str = first_ts_dt.tz_convert('America/New_York').strftime('%Y-%m-%d')
    
    # Compute correct session timing (relative to 09:30 ET, NOT first bar)
    minutes_since_open = compute_minutes_since_open(signal_ts, date_str)
    bars_since_open = compute_bars_since_open(signal_ts, date_str, bar_duration_minutes=1)
    
    # Compute approach dynamics for each signal
    for i in range(n):
        ts = signal_ts[i]
        start_ts = ts - lookback_ns
        level = level_prices[i]
        direction = directions[i]

        # Find bars in lookback window using binary search
        start_idx = np.searchsorted(ohlcv_ts, start_ts, side='right')
        end_idx = np.searchsorted(ohlcv_ts, ts, side='right')

        if start_idx >= end_idx or end_idx > len(ohlcv_ts):
            continue

        # Get historical prices
        hist_close = ohlcv_close[start_idx:end_idx]
        hist_open = ohlcv_open[start_idx:end_idx]

        if len(hist_close) < 2:
            continue

        # Approach velocity: price change per minute
        price_change = hist_close[-1] - hist_close[0]
        time_minutes = (ohlcv_ts[end_idx - 1] - ohlcv_ts[start_idx]) / (60.0 * 1e9)
        if time_minutes <= 0:
            time_minutes = max(len(hist_close) - 1, 1)

        if direction == 'UP':
            # Approaching resistance from below - positive velocity = moving up toward level
            approach_velocity[i] = price_change / time_minutes
        else:
            # Approaching support from above - positive velocity = moving down toward level
            approach_velocity[i] = -price_change / time_minutes

        # Approach bars: consecutive bars moving toward level
        consecutive = 0
        for j in range(len(hist_close) - 1, 0, -1):
            bar_move = hist_close[j] - hist_close[j-1]
            if direction == 'UP':
                # For UP, we want bars that closed higher (moving toward resistance)
                if bar_move > 0:
                    consecutive += 1
                else:
                    break
            else:
                # For DOWN, we want bars that closed lower (moving toward support)
                if bar_move < 0:
                    consecutive += 1
                else:
                    break
        approach_bars[i] = consecutive

        # Approach distance: total price traveled toward level
        approach_distance[i] = abs(hist_close[-1] - hist_close[0])

    result = signals_df.copy()
    result['approach_velocity'] = approach_velocity
    result['approach_bars'] = approach_bars
    result['approach_distance'] = approach_distance
    result['minutes_since_open'] = minutes_since_open
    result['bars_since_open'] = bars_since_open
    
    # or_active: Open Range active flag (1 if >= 15 min since open)
    result['or_active'] = (minutes_since_open >= 15).astype(np.int32)

    return result


def compute_attempt_features(
    signals_df: pd.DataFrame,
    time_window_minutes: Optional[int] = None,
    price_band: Optional[float] = None
) -> pd.DataFrame:
    """Compute touch clustering, attempt index, and deterioration trends."""
    if signals_df.empty:
        return signals_df

    if time_window_minutes is None:
        time_window_minutes = CONFIG.TOUCH_CLUSTER_TIME_MINUTES
    if price_band is None:
        price_band = CONFIG.TOUCH_CLUSTER_PRICE_BAND

    df = signals_df.copy()
    df['attempt_index'] = 0
    df['attempt_cluster_id'] = 0
    df['barrier_replenishment_trend'] = 0.0
    df['barrier_delta_liq_trend'] = 0.0
    df['tape_velocity_trend'] = 0.0
    df['tape_imbalance_trend'] = 0.0

    sort_cols = ['date', 'level_kind_name', 'direction', 'ts_ns']
    df_sorted = df.sort_values(sort_cols)

    time_window_ns = int(time_window_minutes * 60 * 1e9)

    def _safe_array(series: Optional[pd.Series], size: int) -> np.ndarray:
        if series is None:
            return np.zeros(size, dtype=np.float64)
        values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    grouped = df_sorted.groupby(['date', 'level_kind_name', 'direction'], sort=False)
    for _, group in grouped:
        idxs = group.index.to_numpy()
        size = len(group)
        if size == 0:
            continue

        ts_values = group['ts_ns'].to_numpy(dtype=np.int64)
        level_prices = group['level_price'].to_numpy(dtype=np.float64)
        repl_ratio = _safe_array(group['barrier_replenishment_ratio'] if 'barrier_replenishment_ratio' in group.columns else None, size)
        delta_liq = _safe_array(group['barrier_delta_liq'] if 'barrier_delta_liq' in group.columns else None, size)
        tape_velocity = _safe_array(group['tape_velocity'] if 'tape_velocity' in group.columns else None, size)
        tape_imbalance = _safe_array(group['tape_imbalance'] if 'tape_imbalance' in group.columns else None, size)

        cluster_ids = np.zeros(size, dtype=np.int32)
        attempt_indices = np.zeros(size, dtype=np.int32)
        repl_trend = np.zeros(size, dtype=np.float64)
        delta_liq_trend = np.zeros(size, dtype=np.float64)
        tape_velocity_trend = np.zeros(size, dtype=np.float64)
        tape_imbalance_trend = np.zeros(size, dtype=np.float64)

        cluster_id = 0
        attempt_index = 0
        last_ts: Optional[int] = None
        last_price: Optional[float] = None
        first_repl = 0.0
        first_delta = 0.0
        first_tape_velocity = 0.0
        first_tape_imbalance = 0.0

        for i in range(size):
            ts = ts_values[i]
            level_price = level_prices[i]
            new_cluster = False
            if last_ts is None:
                new_cluster = True
            else:
                if ts - last_ts > time_window_ns:
                    new_cluster = True
                if abs(level_price - last_price) > price_band:
                    new_cluster = True

            if new_cluster:
                cluster_id += 1
                attempt_index = 1
                first_repl = repl_ratio[i]
                first_delta = delta_liq[i]
                first_tape_velocity = tape_velocity[i]
                first_tape_imbalance = tape_imbalance[i]
            else:
                attempt_index += 1

            cluster_ids[i] = cluster_id
            attempt_indices[i] = attempt_index
            repl_trend[i] = repl_ratio[i] - first_repl
            delta_liq_trend[i] = delta_liq[i] - first_delta
            tape_velocity_trend[i] = tape_velocity[i] - first_tape_velocity
            tape_imbalance_trend[i] = tape_imbalance[i] - first_tape_imbalance

            last_ts = ts
            last_price = level_price

        df.loc[idxs, 'attempt_cluster_id'] = cluster_ids
        df.loc[idxs, 'attempt_index'] = attempt_indices
        df.loc[idxs, 'barrier_replenishment_trend'] = repl_trend
        df.loc[idxs, 'barrier_delta_liq_trend'] = delta_liq_trend
        df.loc[idxs, 'tape_velocity_trend'] = tape_velocity_trend
        df.loc[idxs, 'tape_imbalance_trend'] = tape_imbalance_trend

    return df


def add_sparse_feature_transforms(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator and signed-log transforms for sparse features."""
    if signals_df.empty:
        return signals_df

    result = signals_df.copy()
    sparse_cols = ['wall_ratio', 'barrier_delta_liq']
    for col in sparse_cols:
        if col not in result.columns:
            continue
        values = result[col].astype(np.float64).to_numpy()
        result[f"{col}_nonzero"] = (values != 0).astype(np.int8)
        result[f"{col}_log"] = np.sign(values) * np.log1p(np.abs(values))

    return result


def add_normalized_features(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR and spot-normalized distance features."""
    if signals_df.empty:
        return signals_df

    if 'atr' not in signals_df.columns:
        raise ValueError("ATR values missing; compute ATR before normalization.")

    result = signals_df.copy()
    
    # Use 'entry_price' as spot if 'spot' not present
    if 'spot' in result.columns:
        spot = result['spot'].astype(np.float64).to_numpy()
    elif 'entry_price' in result.columns:
        spot = result['entry_price'].astype(np.float64).to_numpy()
        result['spot'] = spot  # Add for consistency
    else:
        raise ValueError("Neither 'spot' nor 'entry_price' column found")
    
    atr = result['atr'].astype(np.float64).to_numpy()
    eps = 1e-6

    result['distance_signed'] = spot - result['level_price'].astype(np.float64)

    distance_cols = [
        'distance',
        'distance_signed',
        'dist_to_pm_high',
        'dist_to_pm_low',
        'dist_to_or_high',
        'dist_to_or_low',
        'dist_to_sma_90',
        'dist_to_ema_20',
        'confluence_min_distance',
        'approach_distance'
    ]
    for col in distance_cols:
        if col not in result.columns:
            continue
        values = result[col].astype(np.float64).to_numpy()
        result[f"{col}_atr"] = values / (atr + eps)
        result[f"{col}_pct"] = values / (spot + eps)

    result['level_price_pct'] = (result['level_price'].astype(np.float64) - spot) / (spot + eps)

    return result


class ComputeApproachFeaturesStage(BaseStage):
    """Compute approach context and normalized features.

    This stage is used in v2.0+ pipelines to add:
    - Approach context (velocity, bars, distance)
    - Sparse feature transforms
    - Normalized features
    - Attempt clustering and deterioration trends

    Outputs:
        signals_df: Updated with approach features
    """

    @property
    def name(self) -> str:
        return "compute_approach_features"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min', 'atr']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']
        atr_series = ctx.data['atr']

        # Attach ATR to signals if not already present
        if 'atr' not in signals_df.columns:
            if 'ts_ns' not in signals_df.columns:
                raise ValueError("signals_df missing ts_ns; cannot align ATR.")
            if ohlcv_df.empty:
                raise ValueError("ohlcv_1min is empty; cannot align ATR.")
            if 'ts_ns' not in ohlcv_df.columns:
                raise ValueError("ohlcv_1min missing ts_ns; cannot align ATR.")
            if atr_series is None or len(atr_series) == 0:
                raise ValueError("ATR series missing; compute ATR before approach features.")

            # Match signals to OHLCV bars by timestamp to get ATR
            ohlcv_df = ohlcv_df.copy()
            ohlcv_df['atr_val'] = atr_series
            signals_with_atr = pd.merge_asof(
                signals_df.sort_values('ts_ns'),
                ohlcv_df[['ts_ns', 'atr_val']].sort_values('ts_ns').rename(columns={'atr_val': 'atr'}),
                on='ts_ns',
                direction='backward'
            )
            signals_df = signals_with_atr.sort_index()

        # Compute approach context (pass date from context)
        signals_df = compute_approach_context(signals_df, ohlcv_df, lookback_minutes=None)

        # Sparse feature transforms + normalization
        signals_df = add_sparse_feature_transforms(signals_df)
        signals_df = add_normalized_features(signals_df)

        # Attempt clustering + deterioration trends
        signals_df = compute_attempt_features(signals_df)

        return {'signals_df': signals_df}
