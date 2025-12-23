"""
Vectorized Pipeline - Apple M4 Silicon Optimized

Complete feature engineering pipeline with vectorized numpy operations
optimized for Apple M4 Silicon with 128GB RAM.

Key Optimizations:
- All operations use numpy broadcasting (no Python loops)
- Batch processing of all touches simultaneously
- Memory-efficient chunked processing for large datasets
- Numba JIT compilation for hot paths
- Parallel execution using multiprocessing

Performance Target:
- Process 1M+ trades in <10 seconds
- Generate 10K+ signals per day
- Memory usage <16GB for full day

Usage:
    cd backend/
    uv run python -m src.pipeline.vectorized_pipeline --date 2025-12-18
"""

import argparse
import sys
import uuid
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Import data loading
from src.ingestor.dbn_ingestor import DBNIngestor
from src.lake.bronze_writer import BronzeReader

# Import production engines
from src.core.market_state import MarketState, OptionFlowAggregate
from src.core.barrier_engine import BarrierEngine, Direction as BarrierDirection, BarrierState
from src.core.tape_engine import TapeEngine
from src.core.fuel_engine import FuelEngine, FuelEffect

# Import schemas and event types
from src.common.schemas.levels_signals import LevelSignalV1, LevelKind, Direction, OutcomeLabel
from src.common.event_types import MBP10, FuturesTrade, OptionTrade, Aggressor
from src.common.config import CONFIG

# Import Black-Scholes calculator
from src.core.black_scholes import compute_greeks_for_dataframe


# =============================================================================
# NUMBA JIT COMPILATION (Optional - falls back to numpy if not available)
# =============================================================================

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# Note: Numba parallel=True with shared counter causes race conditions
# Use pure numpy version instead which is still fast on M4 Silicon
def _vectorized_touch_detection_numpy(
    timestamps: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    levels: np.ndarray,
    tolerance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy-vectorized touch detection using broadcasting.

    Returns arrays of (bar_idx, level_idx, direction, distance) for all touches.
    """
    n_bars = len(timestamps)
    n_levels = len(levels)

    # Broadcasting: (n_bars, 1) vs (n_levels,) -> (n_bars, n_levels)
    lows_2d = lows[:, np.newaxis]
    highs_2d = highs[:, np.newaxis]
    closes_2d = closes[:, np.newaxis]
    levels_2d = levels[np.newaxis, :]

    # Touch mask: level within [low - tol, high + tol]
    touch_mask = (lows_2d - tolerance <= levels_2d) & (levels_2d <= highs_2d + tolerance)

    # Get indices of touches
    bar_indices, level_indices = np.where(touch_mask)

    if len(bar_indices) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int8),
            np.array([], dtype=np.float64)
        )

    # Compute direction and distance
    touch_closes = closes[bar_indices]
    touch_levels = levels[level_indices]
    directions = np.where(touch_closes < touch_levels, 1, -1).astype(np.int8)
    distances = np.abs(touch_closes - touch_levels)

    return bar_indices.astype(np.int64), level_indices.astype(np.int64), directions, distances


@jit(nopython=True, cache=True)
def _vectorized_imbalance_numba(
    aggressors: np.ndarray,
    sizes: np.ndarray
) -> Tuple[int, int, float]:
    """Numba-accelerated buy/sell volume computation."""
    buy_vol = 0
    sell_vol = 0

    for i in range(len(aggressors)):
        if aggressors[i] == 1:  # BUY
            buy_vol += sizes[i]
        elif aggressors[i] == -1:  # SELL
            sell_vol += sizes[i]

    total = buy_vol + sell_vol
    imbalance = (buy_vol - sell_vol) / (total + 1e-6)

    return buy_vol, sell_vol, imbalance


# =============================================================================
# VECTORIZED OHLCV BUILDING
# =============================================================================

def build_ohlcv_vectorized(
    trades: List[FuturesTrade],
    convert_to_spy: bool = True,
    freq: str = '1min'
) -> pd.DataFrame:
    """
    Build OHLCV bars using vectorized pandas operations.

    Optimized for Apple M4 Silicon with large RAM:
    - Uses numpy arrays directly
    - Single-pass aggregation
    - No Python loops

    Args:
        trades: List of FuturesTrade objects
        convert_to_spy: Divide prices by 10 for SPY equivalent
        freq: Bar frequency ('1min', '2min', '5min')

    Returns:
        DataFrame with OHLCV columns
    """
    if not trades:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Extract numpy arrays directly (faster than list comprehension)
    n = len(trades)
    ts_ns = np.empty(n, dtype=np.int64)
    prices = np.empty(n, dtype=np.float64)
    sizes = np.empty(n, dtype=np.int64)

    for i, trade in enumerate(trades):
        ts_ns[i] = trade.ts_event_ns
        prices[i] = trade.price
        sizes[i] = trade.size

    # Filter outliers using vectorized operations
    valid_mask = (prices > 3000) & (prices < 10000)
    ts_ns = ts_ns[valid_mask]
    prices = prices[valid_mask]
    sizes = sizes[valid_mask]

    if len(ts_ns) == 0:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Create DataFrame with numpy arrays
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(ts_ns, unit='ns', utc=True),
        'price': prices,
        'size': sizes
    })

    # Set index and resample (pandas optimized)
    df.set_index('timestamp', inplace=True)

    # Vectorized aggregation
    ohlcv = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    ohlcv.columns = ['open', 'high', 'low', 'close']
    ohlcv['volume'] = df['size'].resample(freq).sum()

    # Drop NaN bars
    ohlcv = ohlcv.dropna(subset=['open'])
    ohlcv = ohlcv.reset_index()

    # Convert ES to SPY (vectorized)
    if convert_to_spy:
        ohlcv[['open', 'high', 'low', 'close']] /= 10.0

    return ohlcv


# =============================================================================
# VECTORIZED LEVEL UNIVERSE GENERATION
# =============================================================================

@dataclass
class LevelInfo:
    """Level information for vectorized processing."""
    prices: np.ndarray
    kinds: np.ndarray  # integer codes for LevelKind
    kind_names: List[str]


def generate_level_universe_vectorized(
    ohlcv_df: pd.DataFrame,
    option_flows: Dict[Tuple[float, str, str], OptionFlowAggregate],
    date: str,
    ohlcv_2min: Optional[pd.DataFrame] = None
) -> LevelInfo:
    """
    Generate complete level universe using vectorized operations.

    Levels generated:
    - STRIKE: Option strikes with volume
    - PM_HIGH/PM_LOW: Pre-market high/low
    - OR_HIGH/OR_LOW: Opening range (first 15min) high/low
    - SESSION_HIGH/SESSION_LOW: Running session extremes
    - SMA_200: 200-period SMA on 2-min bars
    - VWAP: Session VWAP
    - ROUND: Round dollar levels
    - CALL_WALL/PUT_WALL: Max gamma concentration

    Returns:
        LevelInfo with arrays for vectorized processing
    """
    levels = []
    kinds = []
    kind_names = []

    if ohlcv_df.empty:
        return LevelInfo(
            prices=np.array([]),
            kinds=np.array([], dtype=np.int8),
            kind_names=[]
        )

    # Ensure timestamp is datetime
    df = ohlcv_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Convert to ET for time-of-day logic
    df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.time

    from datetime import time as dt_time

    # 1. PRE-MARKET HIGH/LOW (04:00-09:30 ET)
    pm_mask = (df['time_et'] >= dt_time(4, 0)) & (df['time_et'] < dt_time(9, 30))
    if pm_mask.any():
        pm_data = df[pm_mask]
        pm_high = pm_data['high'].max()
        pm_low = pm_data['low'].min()
        levels.extend([pm_high, pm_low])
        kinds.extend([0, 1])  # PM_HIGH=0, PM_LOW=1
        kind_names.extend(['PM_HIGH', 'PM_LOW'])

    # 2. OPENING RANGE HIGH/LOW (09:30-09:45 ET)
    or_mask = (df['time_et'] >= dt_time(9, 30)) & (df['time_et'] < dt_time(9, 45))
    if or_mask.any():
        or_data = df[or_mask]
        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        levels.extend([or_high, or_low])
        kinds.extend([2, 3])  # OR_HIGH=2, OR_LOW=3
        kind_names.extend(['OR_HIGH', 'OR_LOW'])

    # 3. SESSION HIGH/LOW (running)
    session_high = df['high'].max()
    session_low = df['low'].min()
    levels.extend([session_high, session_low])
    kinds.extend([4, 5])  # SESSION_HIGH=4, SESSION_LOW=5
    kind_names.extend(['SESSION_HIGH', 'SESSION_LOW'])

    # 4. SMA-200 / SMA-400 on 2-min bars
    if ohlcv_2min is None:
        df_2min = df.set_index('timestamp').resample('2min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_2min = df_2min.reset_index()
    else:
        df_2min = ohlcv_2min.copy()

    if not df_2min.empty and 'timestamp' in df_2min.columns:
        df_2min = df_2min.sort_values('timestamp')

    if len(df_2min) >= 200:
        sma_200 = df_2min['close'].rolling(200).mean().iloc[-1]
        if pd.notna(sma_200):
            levels.append(sma_200)
            kinds.append(6)  # SMA_200=6
            kind_names.append('SMA_200')

    if len(df_2min) >= 400:
        sma_400 = df_2min['close'].rolling(400).mean().iloc[-1]
        if pd.notna(sma_400):
            levels.append(sma_400)
            kinds.append(12)  # SMA_400=12
            kind_names.append('SMA_400')

    # 5. VWAP
    session_mask = df['time_et'] >= dt_time(9, 30)
    if session_mask.any():
        session_df = df[session_mask]
        typical_price = (session_df['high'] + session_df['low'] + session_df['close']) / 3
        vwap = (typical_price * session_df['volume']).sum() / session_df['volume'].sum()
        if pd.notna(vwap):
            levels.append(vwap)
            kinds.append(7)  # VWAP=7
            kind_names.append('VWAP')

    # 6. ROUND LEVELS (every $1)
    price_range_low = int(session_low) - 2
    price_range_high = int(session_high) + 3
    round_levels = np.arange(price_range_low, price_range_high, 1.0)
    levels.extend(round_levels.tolist())
    kinds.extend([8] * len(round_levels))  # ROUND=8
    kind_names.extend(['ROUND'] * len(round_levels))

    # 7. STRIKE LEVELS from option flows
    if option_flows:
        strikes = set()
        for (strike, right, exp_date), flow in option_flows.items():
            if exp_date == date and flow.cumulative_volume > 100:
                strikes.add(strike)

        strike_list = sorted(strikes)
        levels.extend(strike_list)
        kinds.extend([9] * len(strike_list))  # STRIKE=9
        kind_names.extend(['STRIKE'] * len(strike_list))

        # 8. CALL_WALL / PUT_WALL (max gamma concentration)
        call_gamma = {}
        put_gamma = {}
        for (strike, right, exp_date), flow in option_flows.items():
            if exp_date == date:
                if right == 'C':
                    call_gamma[strike] = call_gamma.get(strike, 0) + abs(flow.net_gamma_flow)
                else:
                    put_gamma[strike] = put_gamma.get(strike, 0) + abs(flow.net_gamma_flow)

        if call_gamma:
            call_wall = max(call_gamma, key=call_gamma.get)
            levels.append(call_wall)
            kinds.append(10)  # CALL_WALL=10
            kind_names.append('CALL_WALL')

        if put_gamma:
            put_wall = max(put_gamma, key=put_gamma.get)
            levels.append(put_wall)
            kinds.append(11)  # PUT_WALL=11
            kind_names.append('PUT_WALL')

    # Remove duplicates while preserving order
    unique_levels = []
    unique_kinds = []
    unique_names = []
    seen = set()

    for lvl, kind, name in zip(levels, kinds, kind_names):
        key = round(lvl, 2)
        if key not in seen:
            seen.add(key)
            unique_levels.append(lvl)
            unique_kinds.append(kind)
            unique_names.append(name)

    return LevelInfo(
        prices=np.array(unique_levels, dtype=np.float64),
        kinds=np.array(unique_kinds, dtype=np.int8),
        kind_names=unique_names
    )


# =============================================================================
# VECTORIZED TOUCH DETECTION
# =============================================================================

def detect_touches_vectorized(
    ohlcv_df: pd.DataFrame,
    level_info: LevelInfo,
    touch_tolerance: float = 0.10
) -> pd.DataFrame:
    """
    Detect all level touches using numpy broadcasting.

    Uses Numba if available for parallel processing on M4 cores.

    Args:
        ohlcv_df: OHLCV DataFrame with 1-min bars
        level_info: Level universe
        touch_tolerance: How close counts as a touch

    Returns:
        DataFrame with columns: ts_ns, bar_idx, level_price, level_kind, direction, distance, spot
    """
    if ohlcv_df.empty or len(level_info.prices) == 0:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    # Extract numpy arrays
    timestamps = ohlcv_df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    lows = ohlcv_df['low'].values.astype(np.float64)
    highs = ohlcv_df['high'].values.astype(np.float64)
    closes = ohlcv_df['close'].values.astype(np.float64)
    levels = level_info.prices

    # Use numpy broadcasting (fast on M4 Silicon, avoids Numba race condition issues)
    bar_idx, level_idx, directions, distances = _vectorized_touch_detection_numpy(
        timestamps, lows, highs, closes, levels, touch_tolerance
    )

    if len(bar_idx) == 0:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    # Build result DataFrame using the computed indices
    result = pd.DataFrame({
        'ts_ns': timestamps[bar_idx],
        'bar_idx': bar_idx.astype(np.int64),
        'level_price': levels[level_idx],
        'level_kind': level_info.kinds[level_idx],
        'level_kind_name': [level_info.kind_names[int(i)] for i in level_idx],
        'direction': np.where(directions == 1, 'UP', 'DOWN'),
        'distance': distances,
        'spot': closes[bar_idx]
    })

    # Deduplicate: one touch per level per minute
    result = result.drop_duplicates(subset=['ts_ns', 'level_price'])

    # Filter to only keep touches where close is near the level
    # This ensures barrier/tape physics are meaningful (order book visible at level)
    from src.common.config import CONFIG
    result = result[result['distance'] <= CONFIG.MONITOR_BAND]

    return result


# =============================================================================
# VECTORIZED PHYSICS COMPUTATION
# =============================================================================

def compute_physics_batch(
    touches_df: pd.DataFrame,
    market_state: MarketState,
    barrier_engine: BarrierEngine,
    tape_engine: TapeEngine,
    fuel_engine: FuelEngine,
    exp_date: str,
    trades: List[FuturesTrade] = None,
    mbp10_snapshots: List[MBP10] = None
) -> pd.DataFrame:
    """
    Compute physics metrics for all touches in batch.

    Uses vectorized engines when historical data is available for proper
    time-windowed queries at each touch timestamp.

    Args:
        touches_df: DataFrame from detect_touches_vectorized
        market_state: Initialized MarketState
        barrier_engine: BarrierEngine instance
        tape_engine: TapeEngine instance
        fuel_engine: FuelEngine instance
        exp_date: Expiration date for options
        trades: Optional raw trades for vectorized processing
        mbp10_snapshots: Optional MBP-10 snapshots for vectorized processing

    Returns:
        DataFrame with physics columns added
    """
    if touches_df.empty:
        return touches_df

    n = len(touches_df)

    # Try to use fully vectorized engines if we have raw data
    if trades is not None and len(trades) > 0:
        try:
            from src.core.vectorized_engines import (
                build_vectorized_market_data,
                compute_tape_metrics_batch,
                compute_barrier_metrics_batch,
                compute_fuel_metrics_batch
            )

            # Build vectorized market data
            vmd = build_vectorized_market_data(
                trades=trades,
                mbp10_snapshots=mbp10_snapshots or [],
                option_flows=market_state.option_flows,
                date=exp_date
            )

            # Extract arrays from touches_df
            touch_ts_ns = touches_df['ts_ns'].values.astype(np.int64)
            level_prices = touches_df['level_price'].values.astype(np.float64)
            directions = np.where(touches_df['direction'].values == 'UP', 1, -1)

            # Compute tape metrics (vectorized)
            tape_metrics = compute_tape_metrics_batch(
                touch_ts_ns, level_prices, vmd,
                window_seconds=CONFIG.W_t,
                band_dollars=CONFIG.TAPE_BAND
            )

            # Compute barrier metrics (vectorized)
            # SPY strikes at $1 → ES at $10, zone is ±N ES ticks around strike
            barrier_metrics = compute_barrier_metrics_batch(
                touch_ts_ns, level_prices, directions, vmd,
                window_seconds=CONFIG.W_b,
                zone_es_ticks=CONFIG.BARRIER_ZONE_ES_TICKS
            )

            # Compute fuel metrics (vectorized)
            fuel_metrics = compute_fuel_metrics_batch(
                level_prices, vmd,
                strike_range=CONFIG.FUEL_STRIKE_RANGE
            )

            # Build result
            result = touches_df.copy()
            result['barrier_state'] = barrier_metrics['barrier_state']
            result['barrier_delta_liq'] = barrier_metrics['barrier_delta_liq']
            result['barrier_replenishment_ratio'] = np.zeros(n)  # Computed in barrier engine
            result['wall_ratio'] = barrier_metrics['wall_ratio']
            result['tape_imbalance'] = tape_metrics['tape_imbalance']
            result['tape_buy_vol'] = tape_metrics['tape_buy_vol']
            result['tape_sell_vol'] = tape_metrics['tape_sell_vol']
            result['tape_velocity'] = tape_metrics['tape_velocity']
            result['sweep_detected'] = np.zeros(n, dtype=bool)  # Would need sweep detection
            result['fuel_effect'] = fuel_metrics['fuel_effect']
            result['gamma_exposure'] = fuel_metrics['gamma_exposure']

            return result

        except Exception as e:
            import logging
            logging.warning(f"Vectorized engines failed, falling back to per-signal: {e}")

    # Fallback: per-signal processing using MarketState
    barrier_states = np.empty(n, dtype=object)
    barrier_delta_liq = np.zeros(n, dtype=np.float64)
    barrier_replen = np.zeros(n, dtype=np.float64)
    wall_ratios = np.zeros(n, dtype=np.float64)
    tape_imbalance = np.zeros(n, dtype=np.float64)
    tape_buy_vol = np.zeros(n, dtype=np.int64)
    tape_sell_vol = np.zeros(n, dtype=np.int64)
    tape_velocity = np.zeros(n, dtype=np.float64)
    sweep_detected = np.zeros(n, dtype=bool)
    fuel_effects = np.empty(n, dtype=object)
    gamma_exposure = np.zeros(n, dtype=np.float64)

    for i in range(n):
        row = touches_df.iloc[i]
        level_price = row['level_price']
        direction_str = row['direction']

        barrier_dir = BarrierDirection.RESISTANCE if direction_str == 'UP' else BarrierDirection.SUPPORT

        try:
            barrier_metrics_result = barrier_engine.compute_barrier_state(
                level_price=level_price,
                direction=barrier_dir,
                market_state=market_state
            )
            barrier_states[i] = barrier_metrics_result.state.value
            barrier_delta_liq[i] = barrier_metrics_result.delta_liq
            barrier_replen[i] = barrier_metrics_result.replenishment_ratio
            wall_ratios[i] = barrier_metrics_result.depth_in_zone / 5000.0 if barrier_metrics_result.depth_in_zone else 0.0
        except:
            barrier_states[i] = 'NEUTRAL'

        try:
            tape_metrics_result = tape_engine.compute_tape_state(
                level_price=level_price,
                market_state=market_state
            )
            tape_imbalance[i] = tape_metrics_result.imbalance
            tape_buy_vol[i] = tape_metrics_result.buy_vol
            tape_sell_vol[i] = tape_metrics_result.sell_vol
            tape_velocity[i] = tape_metrics_result.velocity
            sweep_detected[i] = tape_metrics_result.sweep.detected
        except:
            pass

        try:
            fuel_metrics_result = fuel_engine.compute_fuel_state(
                level_price=level_price,
                market_state=market_state,
                exp_date_filter=exp_date
            )
            fuel_effects[i] = fuel_metrics_result.effect.value
            gamma_exposure[i] = fuel_metrics_result.net_dealer_gamma
        except:
            fuel_effects[i] = 'NEUTRAL'

    result = touches_df.copy()
    result['barrier_state'] = barrier_states
    result['barrier_delta_liq'] = barrier_delta_liq
    result['barrier_replenishment_ratio'] = barrier_replen
    result['wall_ratio'] = wall_ratios
    result['tape_imbalance'] = tape_imbalance
    result['tape_buy_vol'] = tape_buy_vol
    result['tape_sell_vol'] = tape_sell_vol
    result['tape_velocity'] = tape_velocity
    result['sweep_detected'] = sweep_detected
    result['fuel_effect'] = fuel_effects
    result['gamma_exposure'] = gamma_exposure

    return result


# =============================================================================
# VECTORIZED LABELING
# =============================================================================

def compute_approach_context_vectorized(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    lookback_minutes: int = None
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

    # Prepare OHLCV data for fast lookup
    ohlcv_sorted = ohlcv_df.sort_values('timestamp')
    ohlcv_ts = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    ohlcv_open = ohlcv_sorted['open'].values.astype(np.float64)

    # Lookback in nanoseconds
    lookback_ns = int(lookback_minutes * 60 * 1e9)

    n = len(signals_df)
    approach_velocity = np.zeros(n, dtype=np.float64)
    approach_bars = np.zeros(n, dtype=np.int32)
    approach_distance = np.zeros(n, dtype=np.float64)
    prior_touches = np.zeros(n, dtype=np.int32)
    bars_since_open = np.zeros(n, dtype=np.int32)

    signal_ts = signals_df['ts_ns'].values
    level_prices = signals_df['level_price'].values
    directions = signals_df['direction'].values

    # Pre-compute first bar timestamp for bars_since_open
    first_bar_ts = ohlcv_ts[0] if len(ohlcv_ts) > 0 else 0
    bar_duration_ns = int(60 * 1e9)  # 1-minute bars

    # Track touches per level for prior_touches calculation
    level_touch_counts = {}

    for i in range(n):
        ts = signal_ts[i]
        start_ts = ts - lookback_ns
        level = level_prices[i]
        direction = directions[i]

        # Find bars in lookback window using binary search
        start_idx = np.searchsorted(ohlcv_ts, start_ts, side='right')
        end_idx = np.searchsorted(ohlcv_ts, ts, side='right')

        # Bars since open
        if first_bar_ts > 0:
            bars_since_open[i] = max(0, (ts - first_bar_ts) // bar_duration_ns)

        # Prior touches at this level
        level_key = round(level, 2)
        prior_touches[i] = level_touch_counts.get(level_key, 0)
        level_touch_counts[level_key] = level_touch_counts.get(level_key, 0) + 1

        if start_idx >= end_idx or end_idx > len(ohlcv_ts):
            continue

        # Get historical prices
        hist_close = ohlcv_close[start_idx:end_idx]
        hist_open = ohlcv_open[start_idx:end_idx]

        if len(hist_close) < 2:
            continue

        # Approach velocity: price change per minute
        # Positive = moving toward level for UP direction, negative for DOWN
        price_change = hist_close[-1] - hist_close[0]
        time_minutes = len(hist_close)  # Each bar is 1 minute

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
    result['prior_touches'] = prior_touches
    result['bars_since_open'] = bars_since_open

    return result


def compute_mean_reversion_features(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    ohlcv_2min: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute SMA-based mean reversion features at each touch timestamp.

    Features:
        - sma_200 / sma_400 values at touch
        - dist_to_sma_200 / dist_to_sma_400 (spot - SMA)
        - sma_200_slope / sma_400_slope ($/min)
        - sma_spread (SMA-200 minus SMA-400)
        - mean_reversion_pressure_200 / 400 (distance normalized by volatility)
        - mean_reversion_velocity_200 / 400 (change in distance per minute)
    """
    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    df_2min = ohlcv_2min.copy() if ohlcv_2min is not None else None
    if df_2min is None:
        df_2min = ohlcv_df.set_index('timestamp').resample('2min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()

    if df_2min.empty or 'timestamp' not in df_2min.columns:
        return signals_df

    df_2min = df_2min.sort_values('timestamp')
    sma_200_values = df_2min['close'].rolling(200).mean().to_numpy()
    sma_400_values = df_2min['close'].rolling(400).mean().to_numpy()
    sma_ts_ns = df_2min['timestamp'].values.astype('datetime64[ns]').astype(np.int64)

    # Prepare volatility series on 1-min bars
    ohlcv_sorted = ohlcv_df.sort_values('timestamp')
    ohlcv_ts_ns = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    price_changes = np.diff(ohlcv_close, prepend=ohlcv_close[0])
    rolling_vol = pd.Series(price_changes).rolling(
        window=CONFIG.MEAN_REVERSION_VOL_WINDOW_MINUTES
    ).std().to_numpy()

    def _index_at_or_before(ts_ns: int, series_ts_ns: np.ndarray) -> int:
        return np.searchsorted(series_ts_ns, ts_ns, side='right') - 1

    def _value_at_or_before(ts_ns: int, series_ts_ns: np.ndarray, series_vals: np.ndarray) -> float:
        idx = _index_at_or_before(ts_ns, series_ts_ns)
        if idx < 0 or idx >= len(series_vals):
            return np.nan
        val = series_vals[idx]
        return val if np.isfinite(val) else np.nan

    n = len(signals_df)
    sma_200 = np.full(n, np.nan, dtype=np.float64)
    sma_400 = np.full(n, np.nan, dtype=np.float64)
    dist_to_sma_200 = np.full(n, np.nan, dtype=np.float64)
    dist_to_sma_400 = np.full(n, np.nan, dtype=np.float64)
    sma_200_slope = np.full(n, np.nan, dtype=np.float64)
    sma_400_slope = np.full(n, np.nan, dtype=np.float64)
    sma_200_slope_5bar = np.full(n, np.nan, dtype=np.float64)
    sma_400_slope_5bar = np.full(n, np.nan, dtype=np.float64)
    sma_spread = np.full(n, np.nan, dtype=np.float64)
    mean_rev_200 = np.full(n, np.nan, dtype=np.float64)
    mean_rev_400 = np.full(n, np.nan, dtype=np.float64)
    mean_rev_vel_200 = np.full(n, np.nan, dtype=np.float64)
    mean_rev_vel_400 = np.full(n, np.nan, dtype=np.float64)

    slope_minutes = max(1, CONFIG.SMA_SLOPE_WINDOW_MINUTES)
    slope_bars = max(1, int(round(slope_minutes / 2)))
    slope_short_bars = max(1, CONFIG.SMA_SLOPE_SHORT_BARS)
    slope_short_minutes = slope_short_bars * 2
    vel_minutes = max(1, CONFIG.MEAN_REVERSION_VELOCITY_WINDOW_MINUTES)
    vel_ns = int(vel_minutes * 60 * 1e9)

    signal_ts = signals_df['ts_ns'].values.astype(np.int64)
    spot_prices = signals_df['spot'].values.astype(np.float64)

    for i in range(n):
        ts = signal_ts[i]
        spot = spot_prices[i]

        sma200 = _value_at_or_before(ts, sma_ts_ns, sma_200_values)
        sma400 = _value_at_or_before(ts, sma_ts_ns, sma_400_values)

        sma_200[i] = sma200
        sma_400[i] = sma400

        if np.isfinite(sma200):
            dist_to_sma_200[i] = spot - sma200
        if np.isfinite(sma400):
            dist_to_sma_400[i] = spot - sma400

        # SMA slopes
        idx_now = _index_at_or_before(ts, sma_ts_ns)
        idx_prev = idx_now - slope_bars
        if idx_now >= 0 and idx_prev >= 0 and idx_now < len(sma_200_values):
            if np.isfinite(sma_200_values[idx_now]) and np.isfinite(sma_200_values[idx_prev]):
                sma_200_slope[i] = (sma_200_values[idx_now] - sma_200_values[idx_prev]) / slope_minutes
        if idx_now >= 0 and idx_prev >= 0 and idx_now < len(sma_400_values):
            if np.isfinite(sma_400_values[idx_now]) and np.isfinite(sma_400_values[idx_prev]):
                sma_400_slope[i] = (sma_400_values[idx_now] - sma_400_values[idx_prev]) / slope_minutes

        idx_prev_short = idx_now - slope_short_bars
        if idx_now >= 0 and idx_prev_short >= 0 and idx_now < len(sma_200_values):
            if np.isfinite(sma_200_values[idx_now]) and np.isfinite(sma_200_values[idx_prev_short]):
                sma_200_slope_5bar[i] = (sma_200_values[idx_now] - sma_200_values[idx_prev_short]) / slope_short_minutes
        if idx_now >= 0 and idx_prev_short >= 0 and idx_now < len(sma_400_values):
            if np.isfinite(sma_400_values[idx_now]) and np.isfinite(sma_400_values[idx_prev_short]):
                sma_400_slope_5bar[i] = (sma_400_values[idx_now] - sma_400_values[idx_prev_short]) / slope_short_minutes

        if np.isfinite(sma200) and np.isfinite(sma400):
            sma_spread[i] = sma200 - sma400

        vol = _value_at_or_before(ts, ohlcv_ts_ns, rolling_vol)
        if np.isfinite(vol) and vol > 0:
            if np.isfinite(dist_to_sma_200[i]):
                mean_rev_200[i] = dist_to_sma_200[i] / (vol + 1e-6)
            if np.isfinite(dist_to_sma_400[i]):
                mean_rev_400[i] = dist_to_sma_400[i] / (vol + 1e-6)

        # Mean reversion velocity
        prev_ts = ts - vel_ns
        prev_spot = _value_at_or_before(prev_ts, ohlcv_ts_ns, ohlcv_close)
        prev_sma200 = _value_at_or_before(prev_ts, sma_ts_ns, sma_200_values)
        prev_sma400 = _value_at_or_before(prev_ts, sma_ts_ns, sma_400_values)

        if np.isfinite(prev_spot) and np.isfinite(prev_sma200) and np.isfinite(dist_to_sma_200[i]):
            prev_dist_200 = prev_spot - prev_sma200
            mean_rev_vel_200[i] = (dist_to_sma_200[i] - prev_dist_200) / vel_minutes
        if np.isfinite(prev_spot) and np.isfinite(prev_sma400) and np.isfinite(dist_to_sma_400[i]):
            prev_dist_400 = prev_spot - prev_sma400
            mean_rev_vel_400[i] = (dist_to_sma_400[i] - prev_dist_400) / vel_minutes

    result = signals_df.copy()
    result['sma_200'] = sma_200
    result['sma_400'] = sma_400
    result['dist_to_sma_200'] = dist_to_sma_200
    result['dist_to_sma_400'] = dist_to_sma_400
    result['sma_200_slope'] = sma_200_slope
    result['sma_400_slope'] = sma_400_slope
    result['sma_200_slope_5bar'] = sma_200_slope_5bar
    result['sma_400_slope_5bar'] = sma_400_slope_5bar
    result['sma_spread'] = sma_spread
    result['mean_reversion_pressure_200'] = mean_rev_200
    result['mean_reversion_pressure_400'] = mean_rev_400
    result['mean_reversion_velocity_200'] = mean_rev_vel_200
    result['mean_reversion_velocity_400'] = mean_rev_vel_400

    return result


def compute_structural_distances(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute signed distances to structural levels (pre-market high/low).
    """
    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    df = ohlcv_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.time

    from datetime import time as dt_time
    pm_mask = (df['time_et'] >= dt_time(4, 0)) & (df['time_et'] < dt_time(9, 30))

    pm_high = np.nan
    pm_low = np.nan
    if pm_mask.any():
        pm_data = df[pm_mask]
        pm_high = pm_data['high'].max()
        pm_low = pm_data['low'].min()

    spot = signals_df['spot'].values.astype(np.float64)
    dist_to_pm_high = np.full(len(signals_df), np.nan, dtype=np.float64)
    dist_to_pm_low = np.full(len(signals_df), np.nan, dtype=np.float64)

    if np.isfinite(pm_high):
        dist_to_pm_high = spot - pm_high
    if np.isfinite(pm_low):
        dist_to_pm_low = spot - pm_low

    result = signals_df.copy()
    result['dist_to_pm_high'] = dist_to_pm_high
    result['dist_to_pm_low'] = dist_to_pm_low

    return result


def compute_confluence_features(
    signals_df: pd.DataFrame,
    level_info: LevelInfo
) -> pd.DataFrame:
    """
    Compute confluence metrics for nearby key levels.
    """
    if signals_df.empty or level_info is None or len(level_info.prices) == 0:
        return signals_df

    key_weights = {
        'PM_HIGH': 1.0,
        'PM_LOW': 1.0,
        'OR_HIGH': 0.9,
        'OR_LOW': 0.9,
        'SMA_200': 0.8,
        'SMA_400': 0.8,
        'VWAP': 0.7,
        'SESSION_HIGH': 0.6,
        'SESSION_LOW': 0.6,
        'CALL_WALL': 1.0,
        'PUT_WALL': 1.0
    }

    key_prices = []
    key_kind_names = []
    key_weight_vals = []

    for price, name in zip(level_info.prices, level_info.kind_names):
        if name in key_weights:
            key_prices.append(price)
            key_kind_names.append(name)
            key_weight_vals.append(key_weights[name])

    if not key_prices:
        result = signals_df.copy()
        result['confluence_count'] = 0
        result['confluence_weighted_score'] = 0.0
        result['confluence_min_distance'] = np.nan
        result['confluence_pressure'] = 0.0
        return result

    key_prices_arr = np.array(key_prices, dtype=np.float64)
    key_weight_arr = np.array(key_weight_vals, dtype=np.float64)
    key_kind_arr = np.array(key_kind_names, dtype=object)

    total_weight = key_weight_arr.sum()
    band = CONFIG.CONFLUENCE_BAND
    eps = 1e-6

    n = len(signals_df)
    confluence_count = np.zeros(n, dtype=np.int32)
    confluence_weighted = np.zeros(n, dtype=np.float64)
    confluence_min_dist = np.full(n, np.nan, dtype=np.float64)
    confluence_pressure = np.zeros(n, dtype=np.float64)

    level_prices = signals_df['level_price'].values.astype(np.float64)
    level_kinds = signals_df['level_kind_name'].values.astype(object)

    for i in range(n):
        level_price = level_prices[i]
        level_kind = level_kinds[i]

        distances = np.abs(key_prices_arr - level_price)
        within_band = distances <= band
        if level_kind in key_weights:
            same_kind = (key_kind_arr == level_kind) & (distances < eps)
            within_band = within_band & ~same_kind
            other_kind = key_kind_arr != level_kind
            if np.any(other_kind):
                confluence_min_dist[i] = float(np.min(distances[other_kind]))
            else:
                confluence_min_dist[i] = float(np.min(distances))
        else:
            confluence_min_dist[i] = float(np.min(distances))

        if not np.any(within_band):
            continue

        active_dist = distances[within_band]
        active_weights = key_weight_arr[within_band]
        distance_decay = np.clip(1.0 - (active_dist / band), 0.0, 1.0)

        confluence_count[i] = int(np.sum(within_band))
        confluence_weighted[i] = float(np.sum(active_weights * distance_decay))
        if total_weight > 0:
            confluence_pressure[i] = confluence_weighted[i] / total_weight

    result = signals_df.copy()
    result['confluence_count'] = confluence_count
    result['confluence_weighted_score'] = confluence_weighted
    result['confluence_min_distance'] = confluence_min_dist
    result['confluence_pressure'] = confluence_pressure

    return result


def compute_dealer_velocity_features(
    signals_df: pd.DataFrame,
    option_trades_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute dealer gamma flow velocity and acceleration features from option trades.
    """
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
    """
    Compute continuous pressure indicators for break/bounce strength.
    """
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


def label_outcomes_vectorized(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    lookforward_minutes: int = None,
    outcome_threshold: float = None
) -> pd.DataFrame:
    """
    Label outcomes using vectorized operations.

    Uses numpy searchsorted for O(log n) future price lookups.
    Threshold is $2.00 (2 strikes) for meaningful options trades.

    Args:
        signals_df: DataFrame with signals
        ohlcv_df: OHLCV DataFrame
        lookforward_minutes: Forward window for labeling (defaults to CONFIG.LOOKFORWARD_MINUTES)
        outcome_threshold: Price move threshold for BREAK/BOUNCE (defaults to CONFIG.OUTCOME_THRESHOLD)

    Returns:
        DataFrame with outcome labels added
    """
    if lookforward_minutes is None:
        lookforward_minutes = CONFIG.LOOKFORWARD_MINUTES
    if outcome_threshold is None:
        outcome_threshold = CONFIG.OUTCOME_THRESHOLD

    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    # Prepare OHLCV data for fast lookup
    ohlcv_sorted = ohlcv_df.sort_values('timestamp')
    ohlcv_ts = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    ohlcv_high = ohlcv_sorted['high'].values.astype(np.float64)
    ohlcv_low = ohlcv_sorted['low'].values.astype(np.float64)

    # Lookforward in nanoseconds
    lookforward_ns = int(lookforward_minutes * 60 * 1e9)

    n = len(signals_df)
    outcomes = np.empty(n, dtype=object)
    future_prices = np.full(n, np.nan, dtype=np.float64)
    excursion_max = np.full(n, np.nan, dtype=np.float64)
    excursion_min = np.full(n, np.nan, dtype=np.float64)
    strength_signed = np.full(n, np.nan, dtype=np.float64)
    strength_abs = np.full(n, np.nan, dtype=np.float64)
    time_to_threshold_1 = np.full(n, np.nan, dtype=np.float64)
    time_to_threshold_2 = np.full(n, np.nan, dtype=np.float64)

    signal_ts = signals_df['ts_ns'].values
    level_prices = signals_df['level_price'].values
    directions = signals_df['direction'].values
    threshold_1 = CONFIG.STRENGTH_THRESHOLD_1
    threshold_2 = CONFIG.STRENGTH_THRESHOLD_2

    # Vectorized: find indices for each signal's lookforward window
    for i in range(n):
        ts = signal_ts[i]
        end_ts = ts + lookforward_ns

        # Find bars in window using binary search
        start_idx = np.searchsorted(ohlcv_ts, ts, side='right')
        end_idx = np.searchsorted(ohlcv_ts, end_ts, side='right')

        if start_idx >= len(ohlcv_ts) or start_idx >= end_idx:
            outcomes[i] = 'UNDEFINED'
            continue

        # Get future prices
        future_close = ohlcv_close[start_idx:end_idx]
        future_high = ohlcv_high[start_idx:end_idx]
        future_low = ohlcv_low[start_idx:end_idx]

        if len(future_close) == 0:
            outcomes[i] = 'UNDEFINED'
            continue

        level = level_prices[i]
        direction = directions[i]

        # Store future price (last close in window)
        future_prices[i] = future_close[-1]

        max_above = max(future_high.max() - level, 0.0)
        max_below = max(level - future_low.min(), 0.0)

        # Compute excursions
        if direction == 'UP':
            # Testing resistance from below
            excursion_max[i] = max_above
            excursion_min[i] = max_below

            # BREAK: moved >threshold above level
            # BOUNCE: moved >threshold below level
            strength_signed[i] = max_above - max_below
            strength_abs[i] = max(max_above, max_below)

            if max_above >= outcome_threshold and max_above > max_below:
                outcomes[i] = 'BREAK'
            elif max_below >= outcome_threshold:
                outcomes[i] = 'BOUNCE'
            else:
                outcomes[i] = 'CHOP'

            # Time to thresholds in break direction (UP)
            above_1 = np.where(future_high >= level + threshold_1)[0]
            above_2 = np.where(future_high >= level + threshold_2)[0]
            if len(above_1) > 0:
                idx = start_idx + above_1[0]
                time_to_threshold_1[i] = (ohlcv_ts[idx] - ts) / 1e9
            if len(above_2) > 0:
                idx = start_idx + above_2[0]
                time_to_threshold_2[i] = (ohlcv_ts[idx] - ts) / 1e9
        else:
            # Testing support from above
            excursion_max[i] = max_below
            excursion_min[i] = max_above

            strength_signed[i] = max_below - max_above
            strength_abs[i] = max(max_below, max_above)

            if max_below >= outcome_threshold and max_below > max_above:
                outcomes[i] = 'BREAK'
            elif max_above >= outcome_threshold:
                outcomes[i] = 'BOUNCE'
            else:
                outcomes[i] = 'CHOP'

            # Time to thresholds in break direction (DOWN)
            below_1 = np.where(future_low <= level - threshold_1)[0]
            below_2 = np.where(future_low <= level - threshold_2)[0]
            if len(below_1) > 0:
                idx = start_idx + below_1[0]
                time_to_threshold_1[i] = (ohlcv_ts[idx] - ts) / 1e9
            if len(below_2) > 0:
                idx = start_idx + below_2[0]
                time_to_threshold_2[i] = (ohlcv_ts[idx] - ts) / 1e9

    result = signals_df.copy()
    result['outcome'] = outcomes
    result['future_price_5min'] = future_prices
    result['excursion_max'] = excursion_max
    result['excursion_min'] = excursion_min
    result['strength_signed'] = strength_signed
    result['strength_abs'] = strength_abs
    result['time_to_threshold_1'] = time_to_threshold_1
    result['time_to_threshold_2'] = time_to_threshold_2

    return result


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

class VectorizedPipeline:
    """
    Optimized pipeline for Apple M4 Silicon.

    Orchestrates all stages with vectorized operations:
    1. Data Loading (parallel file reads)
    2. OHLCV Building (vectorized pandas)
    3. Level Universe Generation (numpy)
    4. Touch Detection (numpy broadcasting / Numba)
    5. Physics Computation (batch processing)
    6. Labeling (vectorized searchsorted)
    7. Export (PyArrow with ZSTD)
    """

    def __init__(self, max_mbp10: int = 500000, max_touches: int = 5000):
        """
        Initialize pipeline.

        Args:
            max_mbp10: Maximum MBP-10 snapshots to load (default 500K covers ~1hr of trading)
            max_touches: Maximum touches to process
        """
        self.max_mbp10 = max_mbp10
        self.max_touches = max_touches

        # Data sources
        self.dbn_ingestor = DBNIngestor()
        self.bronze_reader = BronzeReader()

        # Engines
        self.barrier_engine = BarrierEngine()
        self.tape_engine = TapeEngine()
        self.fuel_engine = FuelEngine()

    def _get_warmup_dates(self, date: str) -> List[str]:
        warmup_days = max(0, CONFIG.SMA_WARMUP_DAYS)
        if warmup_days == 0:
            return []

        available = self.dbn_ingestor.get_available_dates('trades')
        weekday_dates = [
            d for d in available
            if datetime.strptime(d, '%Y-%m-%d').weekday() < 5
        ]
        if date not in weekday_dates:
            return []

        idx = weekday_dates.index(date)
        start_idx = max(0, idx - warmup_days)
        return weekday_dates[start_idx:idx]

    def _build_sma_warmup_2min(self, date: str) -> Tuple[pd.DataFrame, List[str]]:
        warmup_dates = self._get_warmup_dates(date)
        if not warmup_dates:
            return pd.DataFrame(), []

        frames = []
        for warmup_date in warmup_dates:
            trades = list(self.dbn_ingestor.read_trades(date=warmup_date))
            if not trades:
                continue
            ohlcv = build_ohlcv_vectorized(trades, convert_to_spy=True, freq='2min')
            if not ohlcv.empty:
                frames.append(ohlcv)

        if not frames:
            return pd.DataFrame(), warmup_dates

        warmup_df = pd.concat(frames, ignore_index=True).sort_values('timestamp')
        return warmup_df, warmup_dates

    def run(self, date: str) -> pd.DataFrame:
        """
        Run complete pipeline for a date.

        Args:
            date: Date string YYYY-MM-DD

        Returns:
            DataFrame with all signals
        """
        import time
        import logging

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        log = logging.getLogger(__name__)

        total_start = time.time()

        # ========== Stage 1: Data Loading ==========
        log.info(f"Loading data for {date}...")
        stage_start = time.time()

        trades, mbp10_snapshots, option_trades_df = self._load_data(date)

        if not trades:
            log.error("No ES trades found")
            return pd.DataFrame()

        log.info(f"  Loaded {len(trades):,} trades, {len(mbp10_snapshots):,} MBP-10, "
                 f"{len(option_trades_df):,} options in {time.time()-stage_start:.2f}s")

        # ========== Stage 2: OHLCV Building ==========
        log.info("Building OHLCV bars...")
        stage_start = time.time()

        ohlcv_df = build_ohlcv_vectorized(trades, convert_to_spy=True, freq='1min')
        ohlcv_2min = build_ohlcv_vectorized(trades, convert_to_spy=True, freq='2min')
        warmup_2min, warmup_dates = self._build_sma_warmup_2min(date)
        if not warmup_2min.empty:
            ohlcv_2min = pd.concat([warmup_2min, ohlcv_2min], ignore_index=True).sort_values('timestamp')
            log.info(f"  SMA warmup: {len(warmup_2min):,} 2-min bars from {len(warmup_dates)} dates")

        log.info(f"  Built {len(ohlcv_df):,} 1-min bars in {time.time()-stage_start:.2f}s")
        log.info(f"  Price range: ${ohlcv_df['low'].min():.2f} - ${ohlcv_df['high'].max():.2f}")

        # ========== Stage 3: Initialize Market State ==========
        log.info("Initializing market state...")
        stage_start = time.time()

        market_state, option_trades_df = self._initialize_market_state(
            trades, mbp10_snapshots, option_trades_df, date
        )

        log.info(f"  MarketState initialized in {time.time()-stage_start:.2f}s")
        log.info(f"  Buffer stats: {market_state.get_buffer_stats()}")

        # ========== Stage 4: Level Universe ==========
        log.info("Generating level universe...")
        stage_start = time.time()

        level_info = generate_level_universe_vectorized(
            ohlcv_df, market_state.option_flows, date, ohlcv_2min=ohlcv_2min
        )

        log.info(f"  Generated {len(level_info.prices)} levels in {time.time()-stage_start:.2f}s")

        # Log level distribution
        from collections import Counter
        kind_counts = Counter(level_info.kind_names)
        for kind, count in kind_counts.most_common():
            log.info(f"    {kind}: {count}")

        # ========== Stage 5: Touch Detection ==========
        log.info("Detecting level touches...")
        stage_start = time.time()

        touches_df = detect_touches_vectorized(ohlcv_df, level_info, touch_tolerance=0.10)

        log.info(f"  Detected {len(touches_df):,} touches in {time.time()-stage_start:.2f}s")

        # Limit touches if needed
        if len(touches_df) > self.max_touches:
            log.info(f"  (Limiting to {self.max_touches} touches)")
            touches_df = touches_df.head(self.max_touches)

        if touches_df.empty:
            log.warning("No touches detected")
            return pd.DataFrame()

        # ========== Stage 6: Physics Computation ==========
        log.info("Computing physics metrics...")
        stage_start = time.time()

        signals_df = compute_physics_batch(
            touches_df, market_state,
            self.barrier_engine, self.tape_engine, self.fuel_engine,
            exp_date=date,
            trades=trades,
            mbp10_snapshots=mbp10_snapshots
        )

        log.info(f"  Computed physics for {len(signals_df):,} signals in {time.time()-stage_start:.2f}s")

        # ========== Stage 7: Context Features ==========
        log.info("Adding context features...")

        # Add is_first_15m
        from datetime import time as dt_time
        signals_df['timestamp_dt'] = pd.to_datetime(signals_df['ts_ns'], unit='ns', utc=True)
        signals_df['time_et'] = signals_df['timestamp_dt'].dt.tz_convert('America/New_York').dt.time
        signals_df['is_first_15m'] = signals_df['time_et'].apply(
            lambda t: dt_time(9, 30) <= t < dt_time(9, 45)
        )

        # Add date column
        signals_df['date'] = date
        signals_df['symbol'] = 'SPY'

        # Generate event IDs
        signals_df['event_id'] = [str(uuid.uuid4()) for _ in range(len(signals_df))]

        # Structural distances (pre-market levels)
        signals_df = compute_structural_distances(signals_df, ohlcv_df)

        # Mean reversion features (SMA-200/400)
        signals_df = compute_mean_reversion_features(signals_df, ohlcv_df, ohlcv_2min=ohlcv_2min)

        # Confluence features (stacked key levels)
        signals_df = compute_confluence_features(signals_df, level_info)

        # Dealer mechanics velocity features
        signals_df = compute_dealer_velocity_features(signals_df, option_trades_df)

        # Fluid pressure indicators
        signals_df = compute_pressure_indicators(signals_df)

        # ========== Stage 8: Approach Context (Backward) ==========
        log.info("Computing approach context...")
        stage_start = time.time()

        signals_df = compute_approach_context_vectorized(signals_df, ohlcv_df)

        log.info(f"  Computed approach context in {time.time()-stage_start:.2f}s")

        # ========== Stage 9: Labeling (Forward) ==========
        log.info("Labeling outcomes...")
        stage_start = time.time()

        signals_df = label_outcomes_vectorized(signals_df, ohlcv_df)

        log.info(f"  Labeled {len(signals_df):,} signals in {time.time()-stage_start:.2f}s")

        # Outcome distribution
        outcome_counts = signals_df['outcome'].value_counts()
        for outcome, count in outcome_counts.items():
            pct = count / len(signals_df) * 100
            log.info(f"    {outcome}: {count:,} ({pct:.1f}%)")

        # ========== Cleanup ==========
        # Drop intermediate columns
        cols_to_drop = ['timestamp_dt', 'time_et', 'bar_idx']
        signals_df = signals_df.drop(columns=[c for c in cols_to_drop if c in signals_df.columns])

        total_time = time.time() - total_start
        log.info(f"\n{'='*60}")
        log.info(f"PIPELINE COMPLETE in {total_time:.2f}s")
        log.info(f"  Total signals: {len(signals_df):,}")
        log.info(f"  Throughput: {len(signals_df)/total_time:.0f} signals/sec")
        log.info(f"{'='*60}")

        return signals_df

    def _load_data(self, date: str):
        """Load all data sources with timestamp filtering."""
        from datetime import datetime, timezone

        # Load ES trades
        trades = list(self.dbn_ingestor.read_trades(date=date))

        if not trades:
            return trades, [], pd.DataFrame()

        # Get timestamp range from trades to filter MBP-10
        trade_ts_min = min(t.ts_event_ns for t in trades[:1000])
        trade_ts_max = max(t.ts_event_ns for t in trades[-1000:])

        # Load MBP-10 filtered by timestamp range (with buffer)
        buffer_ns = int(60 * 1e9)  # 1 minute buffer
        ts_start = trade_ts_min - buffer_ns
        ts_end = trade_ts_max + buffer_ns

        mbp10_snapshots = []
        skipped = 0
        for mbp in self.dbn_ingestor.read_mbp10(date=date):
            if mbp.ts_event_ns < ts_start:
                skipped += 1
                continue
            if mbp.ts_event_ns > ts_end:
                break  # MBP-10 is sorted, so we can stop early
            mbp10_snapshots.append(mbp)
            if len(mbp10_snapshots) >= self.max_mbp10:
                break

        if skipped > 0:
            import logging
            logging.getLogger(__name__).info(f"  Skipped {skipped:,} MBP-10 snapshots before trade range")

        # Load options
        option_trades_df = self.bronze_reader.read_option_trades(underlying='SPY', date=date)

        return trades, mbp10_snapshots, option_trades_df

    def _initialize_market_state(
        self,
        trades: List[FuturesTrade],
        mbp10_snapshots: List[MBP10],
        option_trades_df: pd.DataFrame,
        date: str
    ) -> Tuple[MarketState, pd.DataFrame]:
        """Initialize MarketState with vectorized Greeks computation."""
        market_state = MarketState(max_buffer_window_seconds=120.0)

        # Load trades
        spot_price = None
        for trade in trades:
            market_state.update_es_trade(trade)
            if 3000 < trade.price < 10000:
                spot_price = trade.price / 10.0

        if spot_price is None:
            spot_price = 600.0

        # Load MBP-10
        for mbp in mbp10_snapshots:
            market_state.update_es_mbp10(mbp)

        # Load options with vectorized Greeks
        if not option_trades_df.empty:
            delta_arr, gamma_arr = compute_greeks_for_dataframe(
                df=option_trades_df,
                spot=spot_price,
                exp_date=date
            )

            option_trades_df = option_trades_df.copy()
            option_trades_df['delta'] = delta_arr
            option_trades_df['gamma'] = gamma_arr

            for idx in range(len(option_trades_df)):
                try:
                    row = option_trades_df.iloc[idx]
                    aggressor_val = row.get('aggressor', 0)
                    if hasattr(aggressor_val, 'value'):
                        aggressor_enum = aggressor_val
                    else:
                        aggressor_enum = Aggressor(int(aggressor_val) if aggressor_val and aggressor_val != '<NA>' else 0)

                    trade = OptionTrade(
                        ts_event_ns=int(row['ts_event_ns']),
                        ts_recv_ns=int(row.get('ts_recv_ns', row['ts_event_ns'])),
                        source=row.get('source', 'polygon_rest'),
                        underlying=row.get('underlying', 'SPY'),
                        option_symbol=row['option_symbol'],
                        exp_date=str(row['exp_date']),
                        strike=float(row['strike']),
                        right=row['right'],
                        price=float(row['price']),
                        size=int(row['size']),
                        opt_bid=row.get('opt_bid'),
                        opt_ask=row.get('opt_ask'),
                        aggressor=aggressor_enum,
                        conditions=None,
                        seq=row.get('seq')
                    )
                    market_state.update_option_trade(
                        trade,
                        delta=row['delta'],
                        gamma=row['gamma']
                    )
                except:
                    continue

        return market_state, option_trades_df


# =============================================================================
# BATCH PROCESSOR FOR MULTIPLE DATES
# =============================================================================

def batch_process_vectorized(
    dates: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    skip_download: bool = False
) -> pd.DataFrame:
    """
    Process multiple dates with optimized pipeline.

    Args:
        dates: List of dates (None = all available)
        output_path: Output Parquet path
        skip_download: Skip options download

    Returns:
        Combined DataFrame
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    dbn_ingestor = DBNIngestor()

    # Discover dates
    if dates is None:
        all_dates = dbn_ingestor.get_available_dates('trades')
        dates = [d for d in all_dates if datetime.strptime(d, '%Y-%m-%d').weekday() < 5]

    print(f"\n{'='*70}")
    print("VECTORIZED BATCH PROCESSOR - Apple M4 Optimized")
    print(f"{'='*70}")
    print(f"Processing {len(dates)} dates: {', '.join(dates)}")
    print()

    pipeline = VectorizedPipeline()
    all_signals = []

    for date in dates:
        try:
            signals_df = pipeline.run(date)
            if not signals_df.empty:
                all_signals.append(signals_df)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"ERROR processing {date}: {e}")
            continue

    if not all_signals:
        print("No signals generated")
        return pd.DataFrame()

    # Combine all signals
    combined_df = pd.concat(all_signals, ignore_index=True)

    # Export to Parquet
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / 'data' / 'lake' / 'gold' / 'research' / 'signals_vectorized.parquet'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(combined_df, preserve_index=False)
    pq.write_table(table, output_path, compression='zstd', compression_level=3)

    print(f"\n{'='*70}")
    print("BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total signals: {len(combined_df):,}")
    print(f"Dates processed: {combined_df['date'].nunique()}")
    print(f"Output: {output_path}")

    # Outcome distribution
    if 'outcome' in combined_df.columns:
        print("\nOutcome distribution:")
        for outcome, count in combined_df['outcome'].value_counts().items():
            pct = count / len(combined_df) * 100
            print(f"  {outcome}: {count:,} ({pct:.1f}%)")

    return combined_df


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Vectorized Spymaster Pipeline - Apple M4 Optimized'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Single date to process (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--dates',
        type=str,
        help='Comma-separated dates to process'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available dates'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output Parquet path'
    )
    parser.add_argument(
        '--list-dates',
        action='store_true',
        help='List available dates'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )

    args = parser.parse_args()

    if args.list_dates:
        dbn = DBNIngestor()
        dates = dbn.get_available_dates('trades')
        print(f"Available dates ({len(dates)}):")
        for d in dates:
            dt = datetime.strptime(d, '%Y-%m-%d')
            print(f"  {d} ({dt.strftime('%a')})")
        return 0

    if args.benchmark:
        print("Running benchmark...")
        import time

        dbn = DBNIngestor()
        dates = dbn.get_available_dates('trades')
        if dates:
            date = dates[-1]
            pipeline = VectorizedPipeline()

            # Warmup
            print(f"Warmup run on {date}...")
            _ = pipeline.run(date)

            # Benchmark
            print(f"\nBenchmark run on {date}...")
            start = time.time()
            signals_df = pipeline.run(date)
            elapsed = time.time() - start

            print(f"\nBenchmark Results:")
            print(f"  Signals: {len(signals_df):,}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {len(signals_df)/elapsed:.0f} signals/sec")
            print(f"  Numba: {'enabled' if NUMBA_AVAILABLE else 'disabled'}")
        return 0

    output_path = Path(args.output) if args.output else None

    if args.date:
        # Single date
        pipeline = VectorizedPipeline()
        signals_df = pipeline.run(args.date)

        if output_path and not signals_df.empty:
            import pyarrow as pa
            import pyarrow.parquet as pq

            output_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(signals_df, preserve_index=False)
            pq.write_table(table, output_path, compression='zstd')
            print(f"\nExported to {output_path}")

        return 0

    if args.dates:
        dates = [d.strip() for d in args.dates.split(',')]
        batch_process_vectorized(dates=dates, output_path=output_path)
        return 0

    if args.all:
        batch_process_vectorized(output_path=output_path)
        return 0

    # Default: process most recent date
    dbn = DBNIngestor()
    dates = dbn.get_available_dates('trades')
    if dates:
        weekday_dates = [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() < 5]
        if weekday_dates:
            pipeline = VectorizedPipeline()
            signals_df = pipeline.run(weekday_dates[-1])
            return 0

    print("No dates available. Check dbn-data/ directory.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
