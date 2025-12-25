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
from src.lake.bronze_writer import BronzeReader

# Import production engines
from src.core.market_state import MarketState, OptionFlowAggregate
from src.core.barrier_engine import BarrierEngine, Direction as BarrierDirection, BarrierState
from src.core.tape_engine import TapeEngine
from src.core.fuel_engine import FuelEngine, FuelEffect

# Import schemas and event types
from src.common.schemas.levels_signals import LevelSignalV1, LevelKind, Direction, OutcomeLabel
from src.common.event_types import (
    MBP10,
    FuturesTrade,
    OptionTrade,
    Aggressor,
    BidAskLevel,
    EventSource,
)
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


def _futures_trades_from_df(trades_df: pd.DataFrame) -> List[FuturesTrade]:
    """Convert Bronze futures trades DataFrame to FuturesTrade objects."""
    if trades_df.empty:
        return []

    df = trades_df
    if not df["ts_event_ns"].is_monotonic_increasing:
        df = df.sort_values("ts_event_ns")

    ts_event = df["ts_event_ns"].to_numpy()
    ts_recv = df["ts_recv_ns"].to_numpy() if "ts_recv_ns" in df.columns else ts_event
    prices = df["price"].to_numpy()
    sizes = df["size"].to_numpy()
    symbols = df["symbol"].to_numpy() if "symbol" in df.columns else np.array(["ES"] * len(df))

    if "aggressor" in df.columns:
        aggressors = pd.to_numeric(df["aggressor"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        aggressors = np.zeros(len(df), dtype=int)

    exchange_vals = (
        pd.to_numeric(df["exchange"], errors="coerce").to_numpy()
        if "exchange" in df.columns
        else None
    )
    seq_vals = df["seq"].to_numpy() if "seq" in df.columns else None

    agg_map = {1: Aggressor.BUY, -1: Aggressor.SELL, 0: Aggressor.MID}

    trades: List[FuturesTrade] = []
    for i in range(len(df)):
        exchange = None
        if exchange_vals is not None:
            val = exchange_vals[i]
            exchange = None if pd.isna(val) else int(val)

        seq = None
        if seq_vals is not None:
            val = seq_vals[i]
            seq = None if pd.isna(val) else int(val)

        trades.append(FuturesTrade(
            ts_event_ns=int(ts_event[i]),
            ts_recv_ns=int(ts_recv[i]),
            source=EventSource.DIRECT_FEED,
            symbol=str(symbols[i]),
            price=float(prices[i]),
            size=int(sizes[i]),
            aggressor=agg_map.get(int(aggressors[i]), Aggressor.MID),
            exchange=exchange,
            conditions=None,
            seq=seq
        ))

    return trades


def _mbp10_from_df(mbp_df: pd.DataFrame) -> List[MBP10]:
    """Convert Bronze MBP-10 DataFrame to MBP10 objects."""
    if mbp_df.empty:
        return []

    df = mbp_df
    if not df["ts_event_ns"].is_monotonic_increasing:
        df = df.sort_values("ts_event_ns")

    mbp_list: List[MBP10] = []
    for row in df.itertuples(index=False):
        levels = [
            BidAskLevel(
                bid_px=getattr(row, f"bid_px_{i}"),
                bid_sz=getattr(row, f"bid_sz_{i}"),
                ask_px=getattr(row, f"ask_px_{i}"),
                ask_sz=getattr(row, f"ask_sz_{i}")
            )
            for i in range(1, 11)
        ]
        symbol = getattr(row, "symbol", "ES")
        is_snapshot = bool(getattr(row, "is_snapshot", False))
        seq = getattr(row, "seq", None)
        seq_val = None if pd.isna(seq) else int(seq)

        mbp_list.append(MBP10(
            ts_event_ns=int(row.ts_event_ns),
            ts_recv_ns=int(getattr(row, "ts_recv_ns", row.ts_event_ns)),
            source=EventSource.DIRECT_FEED,
            symbol=str(symbol),
            levels=levels,
            is_snapshot=is_snapshot,
            seq=seq_val
        ))

    return mbp_list


def compute_atr_vectorized(
    ohlcv_df: pd.DataFrame,
    window_minutes: Optional[int] = None
) -> pd.Series:
    """
    Compute ATR on 1-minute bars for normalization.
    """
    if window_minutes is None:
        window_minutes = CONFIG.ATR_WINDOW_MINUTES
    if ohlcv_df.empty:
        return pd.Series(dtype=np.float64)

    df = ohlcv_df.sort_values('timestamp').copy()
    high = df['high'].astype(np.float64).to_numpy()
    low = df['low'].astype(np.float64).to_numpy()
    close = df['close'].astype(np.float64).to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(window=window_minutes, min_periods=1).mean().to_numpy()

    return pd.Series(atr, index=df.index)


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

    Levels generated (structural only for SPY):
    - PM_HIGH/PM_LOW: Pre-market high/low (04:00-09:30 ET)
    - OR_HIGH/OR_LOW: Opening range (first 15min) high/low
    - SESSION_HIGH/SESSION_LOW: Running session extremes
    - SMA_200/SMA_400: Moving averages on 2-min bars
    - VWAP: Session VWAP
    - CALL_WALL/PUT_WALL: Max gamma concentration

    NOTE: ROUND (8) and STRIKE (9) levels removed for SPY due to
    duplicative $1 strike spacing. Re-enable for other instruments.

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

    # NOTE: ROUND (8) and STRIKE (9) levels removed for SPY - duplicative with $1 strike spacing.
    # These level kinds can be re-enabled for other instruments via config flag.

    # 6. CALL_WALL / PUT_WALL (max gamma concentration)
    if option_flows:
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


def compute_wall_series(
    option_trades_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    date: str
) -> tuple[pd.Series, pd.Series]:
    """
    Compute rolling call/put wall strikes using option flow data.
    """
    if option_trades_df is None or option_trades_df.empty:
        empty = pd.Series(np.nan, index=ohlcv_df.index)
        return empty, empty

    opt_df = option_trades_df.copy()
    required_cols = ['ts_event_ns', 'strike', 'size', 'gamma', 'aggressor', 'right']
    if not set(required_cols).issubset(opt_df.columns):
        empty = pd.Series(np.nan, index=ohlcv_df.index)
        return empty, empty

    opt_df = opt_df[required_cols + (['exp_date'] if 'exp_date' in opt_df.columns else [])].copy()
    if 'exp_date' in opt_df.columns:
        opt_df = opt_df[opt_df['exp_date'].astype(str) == date]
    if opt_df.empty:
        empty = pd.Series(np.nan, index=ohlcv_df.index)
        return empty, empty

    opt_df['ts_event_ns'] = pd.to_numeric(opt_df['ts_event_ns'], errors='coerce').fillna(0).astype(np.int64)
    opt_df['strike'] = pd.to_numeric(opt_df['strike'], errors='coerce').fillna(0).astype(np.float64)
    opt_df['size'] = pd.to_numeric(opt_df['size'], errors='coerce').fillna(0).astype(np.float64)
    opt_df['gamma'] = pd.to_numeric(opt_df['gamma'], errors='coerce').fillna(0).astype(np.float64)
    opt_df['aggressor'] = pd.to_numeric(opt_df['aggressor'], errors='coerce').fillna(0).astype(np.int8)
    opt_df['right'] = opt_df['right'].astype(str)

    opt_df['minute'] = pd.to_datetime(opt_df['ts_event_ns'], unit='ns', utc=True).dt.floor('1min')
    opt_df['dealer_flow'] = -opt_df['aggressor'] * opt_df['size'] * opt_df['gamma'] * 100.0

    window_minutes = max(1, int(round(CONFIG.W_wall / 60.0)))

    def _wall_series_for_right(right: str) -> pd.Series:
        subset = opt_df[opt_df['right'] == right]
        if subset.empty:
            return pd.Series(np.nan, index=ohlcv_df.index)

        grouped = subset.groupby(['minute', 'strike'])['dealer_flow'].sum().reset_index()
        pivot = grouped.pivot_table(
            index='minute',
            columns='strike',
            values='dealer_flow',
            aggfunc='sum',
            fill_value=0.0
        ).sort_index()

        rolling = pivot.rolling(window=window_minutes, min_periods=1).sum()
        total_abs = rolling.abs().sum(axis=1)
        wall_strikes = rolling.idxmin(axis=1)
        wall_strikes[total_abs == 0.0] = np.nan

        ohlcv_minutes = ohlcv_df['timestamp'].dt.floor('1min')
        aligned = pd.merge_asof(
            pd.DataFrame({'minute': ohlcv_minutes}),
            wall_strikes.rename('wall_strike').reset_index(),
            on='minute',
            direction='backward'
        )['wall_strike']
        return aligned

    call_wall = _wall_series_for_right('C')
    put_wall = _wall_series_for_right('P')
    return call_wall, put_wall


def compute_dynamic_level_series(
    ohlcv_df: pd.DataFrame,
    ohlcv_2min: pd.DataFrame,
    option_trades_df: pd.DataFrame,
    date: str
) -> Dict[str, pd.Series]:
    """
    Build per-bar dynamic level series (causal) for structural levels.
    """
    df = ohlcv_df.copy()
    df = df.sort_values('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.time

    from datetime import time as dt_time
    premarket_mask = (df['time_et'] >= dt_time(4, 0)) & (df['time_et'] < dt_time(9, 30))
    session_mask = df['time_et'] >= dt_time(9, 30)
    or_mask = (df['time_et'] >= dt_time(9, 30)) & (df['time_et'] < dt_time(9, 45))

    pm_high_series = pd.Series(np.nan, index=df.index)
    pm_low_series = pd.Series(np.nan, index=df.index)
    if premarket_mask.any():
        pm_high = df.loc[premarket_mask, 'high'].to_numpy(dtype=np.float64)
        pm_low = df.loc[premarket_mask, 'low'].to_numpy(dtype=np.float64)
        pm_high_series.loc[premarket_mask] = pd.Series(pm_high).cummax().values
        pm_low_series.loc[premarket_mask] = pd.Series(pm_low).cummin().values
        final_pm_high = float(np.nanmax(pm_high))
        final_pm_low = float(np.nanmin(pm_low))
        pm_high_series.loc[session_mask] = final_pm_high
        pm_low_series.loc[session_mask] = final_pm_low

    or_high_series = pd.Series(np.nan, index=df.index)
    or_low_series = pd.Series(np.nan, index=df.index)
    if or_mask.any():
        or_high = df.loc[or_mask, 'high'].to_numpy(dtype=np.float64)
        or_low = df.loc[or_mask, 'low'].to_numpy(dtype=np.float64)
        or_high_series.loc[or_mask] = pd.Series(or_high).cummax().values
        or_low_series.loc[or_mask] = pd.Series(or_low).cummin().values
        final_or_high = float(np.nanmax(or_high))
        final_or_low = float(np.nanmin(or_low))
        or_high_series.loc[df['time_et'] >= dt_time(9, 45)] = final_or_high
        or_low_series.loc[df['time_et'] >= dt_time(9, 45)] = final_or_low

    session_high = df['high'].where(session_mask, np.nan)
    session_low = df['low'].where(session_mask, np.nan)
    session_high_series = session_high.expanding().max()
    session_low_series = session_low.expanding().min()

    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vwap_num = (typical_price * df['volume']).where(session_mask, 0.0).cumsum()
    vwap_den = df['volume'].where(session_mask, 0.0).cumsum()
    vwap_series = vwap_num / vwap_den.replace(0, np.nan)

    df_2min = ohlcv_2min.copy()
    df_2min = df_2min.sort_values('timestamp')
    df_2min['timestamp'] = pd.to_datetime(df_2min['timestamp'], utc=True)
    sma_200_series = df_2min['close'].rolling(200).mean()
    sma_400_series = df_2min['close'].rolling(400).mean()
    sma_df = pd.DataFrame({
        'timestamp': df_2min['timestamp'],
        'sma_200': sma_200_series,
        'sma_400': sma_400_series
    })
    sma_aligned = pd.merge_asof(
        df[['timestamp']],
        sma_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    call_wall_series, put_wall_series = compute_wall_series(option_trades_df, df, date)

    return {
        'PM_HIGH': pm_high_series,
        'PM_LOW': pm_low_series,
        'OR_HIGH': or_high_series,
        'OR_LOW': or_low_series,
        'SESSION_HIGH': session_high_series,
        'SESSION_LOW': session_low_series,
        'VWAP': vwap_series,
        'SMA_200': sma_aligned['sma_200'],
        'SMA_400': sma_aligned['sma_400'],
        'CALL_WALL': call_wall_series,
        'PUT_WALL': put_wall_series
    }


def detect_dynamic_level_touches(
    ohlcv_df: pd.DataFrame,
    dynamic_levels: Dict[str, pd.Series],
    touch_tolerance: float = 0.10
) -> pd.DataFrame:
    """
    Detect touches against dynamic level series (causal).
    """
    if ohlcv_df.empty or not dynamic_levels:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    df = ohlcv_df.copy()
    timestamps = df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    lows = df['low'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)

    kind_map = {
        'PM_HIGH': 0,
        'PM_LOW': 1,
        'OR_HIGH': 2,
        'OR_LOW': 3,
        'SESSION_HIGH': 4,
        'SESSION_LOW': 5,
        'SMA_200': 6,
        'VWAP': 7,
        'CALL_WALL': 10,
        'PUT_WALL': 11,
        'SMA_400': 12
    }

    rows = []
    for kind_name, series in dynamic_levels.items():
        if kind_name not in kind_map:
            continue
        values = series.to_numpy(dtype=np.float64)
        mask = np.isfinite(values) & (lows - touch_tolerance <= values) & (values <= highs + touch_tolerance)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        level_prices = values[idx]
        direction = np.where(closes[idx] < level_prices, 'UP', 'DOWN')
        distance = np.abs(closes[idx] - level_prices)
        rows.append(pd.DataFrame({
            'ts_ns': timestamps[idx],
            'bar_idx': idx.astype(np.int64),
            'level_price': level_prices,
            'level_kind': kind_map[kind_name],
            'level_kind_name': kind_name,
            'direction': direction,
            'distance': distance,
            'spot': closes[idx]
        }))

    if not rows:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    result = pd.concat(rows, ignore_index=True)
    result = result.drop_duplicates(subset=['ts_ns', 'level_kind_name', 'level_price'])
    result = result[result['distance'] <= CONFIG.MONITOR_BAND]
    return result


def compute_confluence_features_dynamic(
    signals_df: pd.DataFrame,
    dynamic_levels: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Compute confluence metrics using per-bar dynamic levels (causal).
    """
    if signals_df.empty or not dynamic_levels:
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


def compute_attempt_features(
    signals_df: pd.DataFrame,
    time_window_minutes: Optional[int] = None,
    price_band: Optional[float] = None
) -> pd.DataFrame:
    """
    Compute touch clustering, attempt index, and deterioration trends.
    """
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
    level_prices = signals_df['level_price'].values.astype(np.float64)

    for i in range(n):
        ts = signal_ts[i]
        level_price = level_prices[i]

        sma200 = _value_at_or_before(ts, sma_ts_ns, sma_200_values)
        sma400 = _value_at_or_before(ts, sma_ts_ns, sma_400_values)

        sma_200[i] = sma200
        sma_400[i] = sma400

        if np.isfinite(sma200):
            dist_to_sma_200[i] = level_price - sma200
        if np.isfinite(sma400):
            dist_to_sma_400[i] = level_price - sma400

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
        prev_sma200 = _value_at_or_before(prev_ts, sma_ts_ns, sma_200_values)
        prev_sma400 = _value_at_or_before(prev_ts, sma_ts_ns, sma_400_values)

        if np.isfinite(prev_sma200) and np.isfinite(dist_to_sma_200[i]):
            prev_dist_200 = level_price - prev_sma200
            mean_rev_vel_200[i] = (dist_to_sma_200[i] - prev_dist_200) / vel_minutes
        if np.isfinite(prev_sma400) and np.isfinite(dist_to_sma_400[i]):
            prev_dist_400 = level_price - prev_sma400
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
    Compute signed distances to structural levels (pre-market high/low) relative to target level.
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

    level_prices = signals_df['level_price'].values.astype(np.float64)
    dist_to_pm_high = np.full(len(signals_df), np.nan, dtype=np.float64)
    dist_to_pm_low = np.full(len(signals_df), np.nan, dtype=np.float64)

    if np.isfinite(pm_high):
        dist_to_pm_high = level_prices - pm_high
    if np.isfinite(pm_low):
        dist_to_pm_low = level_prices - pm_low

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


def compute_confluence_alignment(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute confluence alignment based on SMA position/slope and approach direction.
    """
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


def compute_confluence_level_features(
    signals_df: pd.DataFrame,
    dynamic_levels: pd.DataFrame,
    hourly_cumvol: Dict[str, Dict[int, float]],
    date: str
) -> pd.DataFrame:
    """
    Compute hierarchical confluence level (1-10) based on 5 dimensions:
    1. Directional Breakout (ABOVE_ALL, BELOW_ALL, PARTIAL, INSIDE)
    2. SMA Proximity (CLOSE, FAR)
    3. Time Period (FIRST_HOUR, REST_OF_DAY)
    4. GEX Alignment (ALIGNED, OPPOSED, NEUTRAL)
    5. Relative Volume (HIGH, NORMAL, LOW)

    Returns:
        DataFrame with new columns: confluence_level, rel_vol_ratio, gex_alignment, breakout_state
    """
    from datetime import time as dt_time

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
    level_prices = result['level_price'].values.astype(np.float64)
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

    # Get dynamic levels for each signal
    if dynamic_levels.empty:
        result['confluence_level'] = confluence_level
        result['rel_vol_ratio'] = rel_vol_ratio
        result['gex_alignment'] = gex_alignment
        result['breakout_state'] = breakout_state
        return result

    # Merge dynamic levels to signals
    # Ensure timestamp has UTC timezone
    if 'timestamp' in dynamic_levels.columns:
        dynamic_levels['timestamp'] = pd.to_datetime(dynamic_levels['timestamp'], utc=True)
    dynamic_levels_idx = dynamic_levels.set_index('timestamp')

    # Get PM/OR/SMA/Wall values for each signal timestamp
    pm_high = np.full(n, np.nan, dtype=np.float64)
    pm_low = np.full(n, np.nan, dtype=np.float64)
    or_high = np.full(n, np.nan, dtype=np.float64)
    or_low = np.full(n, np.nan, dtype=np.float64)
    sma_200 = np.full(n, np.nan, dtype=np.float64)
    sma_400 = np.full(n, np.nan, dtype=np.float64)
    call_wall = np.full(n, np.nan, dtype=np.float64)
    put_wall = np.full(n, np.nan, dtype=np.float64)
    fuel_effect = np.full(n, '', dtype=object)

    # Use asof join for dynamic levels
    ts_floor = pd.to_datetime(ts_ns, unit='ns', utc=True).floor('1min')

    # Map expected column names to actual column names (uppercase in source)
    col_map = {
        'pm_high': 'PM_HIGH',
        'pm_low': 'PM_LOW',
        'or_high': 'OR_HIGH',
        'or_low': 'OR_LOW',
        'sma_200': 'SMA_200',
        'sma_400': 'SMA_400',
        'call_wall': 'CALL_WALL',
        'put_wall': 'PUT_WALL',
    }
    for target_col, source_col in col_map.items():
        if source_col in dynamic_levels_idx.columns:
            merged = pd.merge_asof(
                pd.DataFrame({'ts': ts_floor}).sort_values('ts'),
                dynamic_levels_idx[[source_col]].reset_index().sort_values('timestamp'),
                left_on='ts',
                right_on='timestamp',
                direction='backward'
            )
            if target_col == 'pm_high':
                pm_high = merged[source_col].values.astype(np.float64)
            elif target_col == 'pm_low':
                pm_low = merged[source_col].values.astype(np.float64)
            elif target_col == 'or_high':
                or_high = merged[source_col].values.astype(np.float64)
            elif target_col == 'or_low':
                or_low = merged[source_col].values.astype(np.float64)
            elif target_col == 'sma_200':
                sma_200 = merged[source_col].values.astype(np.float64)
            elif target_col == 'sma_400':
                sma_400 = merged[source_col].values.astype(np.float64)
            elif target_col == 'call_wall':
                call_wall = merged[source_col].values.astype(np.float64)
            elif target_col == 'put_wall':
                put_wall = merged[source_col].values.astype(np.float64)

    # Get fuel_effect from signals if available
    if 'fuel_effect' in result.columns:
        fuel_effect = result['fuel_effect'].values.astype(object)

    # === Compute relative volume ratio ===
    prior_dates = [d for d in hourly_cumvol.keys() if d < date]
    for i in range(n):
        hour = hour_et[i]
        if hour < 9 or hour > 15:
            continue

        # Current day cumulative volume at this hour
        if date not in hourly_cumvol or hour not in hourly_cumvol[date]:
            continue
        cumvol_now = hourly_cumvol[date][hour]

        # Average of prior days at same hour
        prior_cumvols = [
            hourly_cumvol[d][hour]
            for d in prior_dates
            if d in hourly_cumvol and hour in hourly_cumvol[d]
        ]
        if prior_cumvols:
            avg_cumvol = np.mean(prior_cumvols)
            if avg_cumvol > 0:
                rel_vol_ratio[i] = cumvol_now / avg_cumvol

    # === Compute per-signal confluence level ===
    wall_prox = CONFIG.WALL_PROXIMITY_DOLLARS
    sma_thresh = CONFIG.SMA_PROXIMITY_THRESHOLD
    vol_high = CONFIG.REL_VOL_HIGH_THRESHOLD
    vol_low = CONFIG.REL_VOL_LOW_THRESHOLD

    for i in range(n):
        spot = spot_vals[i]

        # === Dimension 1: Breakout State ===
        pm_h, pm_l = pm_high[i], pm_low[i]
        or_h, or_l = or_high[i], or_low[i]

        if np.isnan(pm_h) or np.isnan(pm_l) or np.isnan(or_h) or np.isnan(or_l):
            confluence_level[i] = 0  # UNDEFINED
            breakout_state[i] = 0
            continue

        above_all = (spot > pm_h) and (spot > or_h)
        below_all = (spot < pm_l) and (spot < or_l)
        full_breakout = above_all or below_all

        partial_breakout = (
            (spot > pm_h or spot < pm_l or spot > or_h or spot < or_l)
            and not full_breakout
        )

        if above_all:
            breakout_state[i] = 2  # ABOVE_ALL
        elif below_all:
            breakout_state[i] = 3  # BELOW_ALL
        elif partial_breakout:
            breakout_state[i] = 1  # PARTIAL
        else:
            breakout_state[i] = 0  # INSIDE

        # === Dimension 2: SMA Proximity ===
        sma_close = False
        sma_200_v, sma_400_v = sma_200[i], sma_400[i]
        if np.isfinite(sma_200_v) and np.isfinite(sma_400_v) and spot > 0:
            dist_200_pct = abs(spot - sma_200_v) / spot
            dist_400_pct = abs(spot - sma_400_v) / spot
            sma_close = (dist_200_pct < sma_thresh) and (dist_400_pct < sma_thresh)

        # === Dimension 3: Time Period ===
        first_hour = (hour_et[i] == 9) or (hour_et[i] == 10 and time_et[i].minute < 30)

        # === Dimension 4: GEX Alignment ===
        cw, pw = call_wall[i], put_wall[i]
        fe = fuel_effect[i] if fuel_effect[i] else ''
        gex_aligned = False
        gex_opposed = False

        if full_breakout and np.isfinite(cw) and np.isfinite(pw):
            if above_all:
                # Bullish breakout
                if spot > cw and fe == 'AMPLIFY':
                    gex_aligned = True  # Broken call wall, dealers chasing
                elif abs(spot - cw) < wall_prox and fe == 'DAMPEN':
                    gex_opposed = True  # Approaching resistance
            else:  # below_all
                # Bearish breakout
                if spot < pw and fe == 'AMPLIFY':
                    gex_aligned = True  # Broken put wall, dealers chasing
                elif abs(spot - pw) < wall_prox and fe == 'DAMPEN':
                    gex_opposed = True  # Approaching support

        if gex_aligned:
            gex_alignment[i] = 1
        elif gex_opposed:
            gex_alignment[i] = -1
        else:
            gex_alignment[i] = 0

        # === Dimension 5: Relative Volume ===
        rv = rel_vol_ratio[i]
        vol_is_high = np.isfinite(rv) and rv >= vol_high
        vol_is_low = np.isfinite(rv) and rv <= vol_low

        # === Determine Confluence Level (1-10) ===
        if full_breakout:
            if sma_close and first_hour:
                if gex_aligned and vol_is_high:
                    confluence_level[i] = 1   # Ultra Premium
                elif gex_aligned:
                    confluence_level[i] = 2   # Premium
                else:
                    confluence_level[i] = 3   # Strong
            elif not sma_close and first_hour:
                if gex_aligned and vol_is_high:
                    confluence_level[i] = 4   # Momentum
                else:
                    confluence_level[i] = 5   # Extended
            else:  # Rest of day
                if sma_close and vol_is_high:
                    confluence_level[i] = 6   # Late Reversion
                else:
                    confluence_level[i] = 7   # Fading
        elif partial_breakout:
            if sma_close and gex_aligned and vol_is_high:
                confluence_level[i] = 8   # Developing
            else:
                confluence_level[i] = 9   # Weak
        else:
            confluence_level[i] = 10  # Consolidation

    result['confluence_level'] = confluence_level
    result['rel_vol_ratio'] = rel_vol_ratio
    result['gex_alignment'] = gex_alignment
    result['breakout_state'] = breakout_state

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


def add_normalized_features(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ATR and spot-normalized distance features.
    """
    if signals_df.empty:
        return signals_df

    if 'atr' not in signals_df.columns:
        raise ValueError("ATR values missing; compute ATR before normalization.")

    result = signals_df.copy()
    spot = result['spot'].astype(np.float64).to_numpy()
    atr = result['atr'].astype(np.float64).to_numpy()
    eps = 1e-6

    result['distance_signed'] = result['level_price'].astype(np.float64) - spot

    distance_cols = [
        'distance',
        'distance_signed',
        'dist_to_pm_high',
        'dist_to_pm_low',
        'dist_to_sma_200',
        'dist_to_sma_400',
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


def add_sparse_feature_transforms(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicator and signed-log transforms for sparse features.
    """
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


def label_outcomes_vectorized(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    lookforward_minutes: int = None,
    outcome_threshold: float = None,
    confirmation_seconds: float = None,
    use_multi_timeframe: bool = True
) -> pd.DataFrame:
    """
    Label outcomes using vectorized operations with multi-timeframe support.

    Outcomes are anchored at confirmation time t1 and measured relative to the
    tested level price (level frame). Break vs bounce is determined by the
    first threshold hit (competing risks), with directional time-to-threshold
    columns preserved alongside legacy "either-direction" labels.

    Uses numpy searchsorted for O(log n) future price lookups.
    Threshold is $2.00 (2 strikes) for meaningful options trades.

    Multi-timeframe mode generates outcomes at 2min, 4min, 8min confirmations
    to enable training models for different trading horizons.

    Args:
        signals_df: DataFrame with signals
        ohlcv_df: OHLCV DataFrame
        lookforward_minutes: Forward window for labeling (defaults to CONFIG.LOOKFORWARD_MINUTES)
        outcome_threshold: Price move threshold for BREAK/BOUNCE (defaults to CONFIG.OUTCOME_THRESHOLD)
        use_multi_timeframe: If True, label at 2min/4min/8min; if False, use single confirmation

    Returns:
        DataFrame with outcome labels added (multi-timeframe or single)
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
    ohlcv_open = ohlcv_sorted['open'].values.astype(np.float64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    ohlcv_high = ohlcv_sorted['high'].values.astype(np.float64)
    ohlcv_low = ohlcv_sorted['low'].values.astype(np.float64)

    # Lookforward in nanoseconds
    lookforward_ns = int(lookforward_minutes * 60 * 1e9)
    
    # Multi-timeframe windows or single window
    if use_multi_timeframe:
        confirmation_windows = CONFIG.CONFIRMATION_WINDOWS_MULTI  # [120, 240, 480]
        window_labels = ['2min', '4min', '8min']
    else:
        confirmation_windows = [confirmation_seconds or CONFIG.CONFIRMATION_WINDOW_SECONDS]
        window_labels = ['']
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    directions = signals_df['direction'].values
    bar_idx = signals_df['bar_idx'].values if 'bar_idx' in signals_df.columns else None
    if 'level_price' not in signals_df.columns:
        raise ValueError("Missing level_price for outcome labeling.")
    level_prices = signals_df['level_price'].values.astype(np.float64)
    threshold_1 = CONFIG.STRENGTH_THRESHOLD_1
    threshold_2 = CONFIG.STRENGTH_THRESHOLD_2
    
    # Storage for multi-timeframe results
    results_by_window = {}
    
    # Process each confirmation window
    for window_sec, label in zip(confirmation_windows, window_labels):
        suffix = f'_{label}' if label else ''
        confirmation_ns = int(window_sec * 1e9)
        
        # Initialize arrays for this window
        outcomes = np.empty(n, dtype=object)
        future_prices = np.full(n, np.nan, dtype=np.float64)
        excursion_max = np.full(n, np.nan, dtype=np.float64)
        excursion_min = np.full(n, np.nan, dtype=np.float64)
        strength_signed = np.full(n, np.nan, dtype=np.float64)
        strength_abs = np.full(n, np.nan, dtype=np.float64)
        time_to_threshold_1 = np.full(n, np.nan, dtype=np.float64)
        time_to_threshold_2 = np.full(n, np.nan, dtype=np.float64)
        time_to_break_1 = np.full(n, np.nan, dtype=np.float64)
        time_to_break_2 = np.full(n, np.nan, dtype=np.float64)
        time_to_bounce_1 = np.full(n, np.nan, dtype=np.float64)
        time_to_bounce_2 = np.full(n, np.nan, dtype=np.float64)
        tradeable_1 = np.zeros(n, dtype=np.int8)
        tradeable_2 = np.zeros(n, dtype=np.int8)
        confirm_ts_ns = np.full(n, np.nan, dtype=np.float64)
        anchor_spot = np.full(n, np.nan, dtype=np.float64)
        
        # Vectorized: find indices for each signal's lookforward window
        for i in range(n):
            ts = signal_ts[i]
            anchor_idx = None
            if bar_idx is not None:
                anchor_idx = int(bar_idx[i])
            else:
                anchor_idx = np.searchsorted(ohlcv_ts, ts, side='right') - 1

            if anchor_idx < 0 or anchor_idx >= len(ohlcv_ts):
                outcomes[i] = 'UNDEFINED'
                continue

            anchor_time = ohlcv_ts[anchor_idx] + confirmation_ns
            confirm_ts_ns[i] = anchor_time

            confirm_idx = np.searchsorted(ohlcv_ts, anchor_time, side='left')
            if confirm_idx >= len(ohlcv_ts):
                outcomes[i] = 'UNDEFINED'
                continue

            if confirm_idx < len(ohlcv_ts) and ohlcv_ts[confirm_idx] == anchor_time:
                anchor_spot[i] = ohlcv_open[confirm_idx]
            else:
                prior_idx = confirm_idx - 1
                if prior_idx < 0:
                    outcomes[i] = 'UNDEFINED'
                    continue
                anchor_spot[i] = ohlcv_close[prior_idx]

            start_idx = confirm_idx
            end_idx = np.searchsorted(ohlcv_ts, anchor_time + lookforward_ns, side='right')

            if start_idx >= len(ohlcv_ts) or start_idx >= end_idx:
                outcomes[i] = 'UNDEFINED'
                continue

            future_close = ohlcv_close[start_idx:end_idx]
            future_high = ohlcv_high[start_idx:end_idx]
            future_low = ohlcv_low[start_idx:end_idx]

            if len(future_close) == 0:
                outcomes[i] = 'UNDEFINED'
                continue

            direction = directions[i]
            level_price = level_prices[i]
            future_prices[i] = future_close[-1]

            max_above = max(future_high.max() - level_price, 0.0)
            max_below = max(level_price - future_low.min(), 0.0)

            above_1 = np.where(future_high >= level_price + threshold_1)[0]
            above_2 = np.where(future_high >= level_price + threshold_2)[0]
            below_1 = np.where(future_low <= level_price - threshold_1)[0]
            below_2 = np.where(future_low <= level_price - threshold_2)[0]

            if direction == 'UP':
                break_1 = above_1
                break_2 = above_2
                bounce_1 = below_1
                bounce_2 = below_2
            else:
                break_1 = below_1
                break_2 = below_2
                bounce_1 = above_1
                bounce_2 = above_2

            if len(break_1) > 0:
                idx = start_idx + break_1[0]
                time_to_break_1[i] = (ohlcv_ts[idx] - anchor_time) / 1e9
            if len(bounce_1) > 0:
                idx = start_idx + bounce_1[0]
                time_to_bounce_1[i] = (ohlcv_ts[idx] - anchor_time) / 1e9

            if len(break_2) > 0:
                idx = start_idx + break_2[0]
                time_to_break_2[i] = (ohlcv_ts[idx] - anchor_time) / 1e9
            if len(bounce_2) > 0:
                idx = start_idx + bounce_2[0]
                time_to_bounce_2[i] = (ohlcv_ts[idx] - anchor_time) / 1e9

            t1_candidates = [
                v for v in (time_to_break_1[i], time_to_bounce_1[i]) if np.isfinite(v)
            ]
            if t1_candidates:
                time_to_threshold_1[i] = min(t1_candidates)
                tradeable_1[i] = 1

            t2_candidates = [
                v for v in (time_to_break_2[i], time_to_bounce_2[i]) if np.isfinite(v)
            ]
            if t2_candidates:
                time_to_threshold_2[i] = min(t2_candidates)
                tradeable_2[i] = 1

            if direction == 'UP':
                excursion_max[i] = max_above
                excursion_min[i] = max_below
                strength_signed[i] = max_above - max_below
                strength_abs[i] = max(max_above, max_below)
            else:
                excursion_max[i] = max_below
                excursion_min[i] = max_above
                strength_signed[i] = max_below - max_above
                strength_abs[i] = max(max_below, max_above)
            break_t2 = time_to_break_2[i]
            bounce_t2 = time_to_bounce_2[i]
            if np.isfinite(break_t2) and np.isfinite(bounce_t2):
                if break_t2 < bounce_t2:
                    outcomes[i] = 'BREAK'
                elif bounce_t2 < break_t2:
                    outcomes[i] = 'BOUNCE'
                else:
                    if strength_signed[i] > 0:
                        outcomes[i] = 'BREAK'
                    elif strength_signed[i] < 0:
                        outcomes[i] = 'BOUNCE'
                    else:
                        outcomes[i] = 'CHOP'
            elif np.isfinite(break_t2):
                outcomes[i] = 'BREAK'
            elif np.isfinite(bounce_t2):
                outcomes[i] = 'BOUNCE'
            else:
                outcomes[i] = 'CHOP'
        
        # Store results for this window
        results_by_window[label] = {
            'outcomes': outcomes,
            'future_prices': future_prices,
            'excursion_max': excursion_max,
            'excursion_min': excursion_min,
            'strength_signed': strength_signed,
            'strength_abs': strength_abs,
            'time_to_threshold_1': time_to_threshold_1,
            'time_to_threshold_2': time_to_threshold_2,
            'time_to_break_1': time_to_break_1,
            'time_to_break_2': time_to_break_2,
            'time_to_bounce_1': time_to_bounce_1,
            'time_to_bounce_2': time_to_bounce_2,
            'tradeable_1': tradeable_1,
            'tradeable_2': tradeable_2,
            'confirm_ts_ns': confirm_ts_ns,
            'anchor_spot': anchor_spot,
        }
    
    # Build result DataFrame
    result = signals_df.copy()
    
    # Add columns for each timeframe
    for label, data in results_by_window.items():
        suffix = f'_{label}' if label else ''
        
        result[f'outcome{suffix}'] = data['outcomes']
        result[f'excursion_max{suffix}'] = data['excursion_max']
        result[f'excursion_min{suffix}'] = data['excursion_min']
        result[f'strength_signed{suffix}'] = data['strength_signed']
        result[f'strength_abs{suffix}'] = data['strength_abs']
        result[f'time_to_threshold_1{suffix}'] = data['time_to_threshold_1']
        result[f'time_to_threshold_2{suffix}'] = data['time_to_threshold_2']
        result[f'time_to_break_1{suffix}'] = data['time_to_break_1']
        result[f'time_to_break_2{suffix}'] = data['time_to_break_2']
        result[f'time_to_bounce_1{suffix}'] = data['time_to_bounce_1']
        result[f'time_to_bounce_2{suffix}'] = data['time_to_bounce_2']
        result[f'tradeable_1{suffix}'] = data['tradeable_1']
        result[f'tradeable_2{suffix}'] = data['tradeable_2']
        result[f'confirm_ts_ns{suffix}'] = data['confirm_ts_ns']
        result[f'anchor_spot{suffix}'] = data['anchor_spot']
        result[f'future_price{suffix}'] = data['future_prices']
    
    # For backward compatibility, also add non-suffixed columns using primary window (4min)
    primary_label = '4min' if use_multi_timeframe else ''
    if use_multi_timeframe and '4min' in results_by_window:
        primary_data = results_by_window['4min']
        result['outcome'] = primary_data['outcomes']
        result['excursion_max'] = primary_data['excursion_max']
        result['excursion_min'] = primary_data['excursion_min']
        result['strength_signed'] = primary_data['strength_signed']
        result['strength_abs'] = primary_data['strength_abs']
        result['time_to_threshold_1'] = primary_data['time_to_threshold_1']
        result['time_to_threshold_2'] = primary_data['time_to_threshold_2']
        result['time_to_break_1'] = primary_data['time_to_break_1']
        result['time_to_break_2'] = primary_data['time_to_break_2']
        result['time_to_bounce_1'] = primary_data['time_to_bounce_1']
        result['time_to_bounce_2'] = primary_data['time_to_bounce_2']
        result['tradeable_1'] = primary_data['tradeable_1']
        result['tradeable_2'] = primary_data['tradeable_2']
        result['confirm_ts_ns'] = primary_data['confirm_ts_ns']
        result['anchor_spot'] = primary_data['anchor_spot']
        result['future_price_5min'] = primary_data['future_prices']

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
        self.bronze_reader = BronzeReader()

        # Engines
        self.barrier_engine = BarrierEngine()
        self.tape_engine = TapeEngine()
        self.fuel_engine = FuelEngine()

    def _get_warmup_dates(self, date: str) -> List[str]:
        warmup_days = max(0, CONFIG.SMA_WARMUP_DAYS)
        if warmup_days == 0:
            return []

        available = self.bronze_reader.get_available_dates('futures/trades', 'symbol=ES')
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
            trades_df = self.bronze_reader.read_futures_trades(symbol='ES', date=warmup_date)
            trades = _futures_trades_from_df(trades_df)
            if not trades:
                continue
            ohlcv = build_ohlcv_vectorized(trades, convert_to_spy=True, freq='2min')
            if not ohlcv.empty:
                frames.append(ohlcv)

        if not frames:
            return pd.DataFrame(), warmup_dates

        warmup_df = pd.concat(frames, ignore_index=True).sort_values('timestamp')
        return warmup_df, warmup_dates

    def _get_volume_warmup_dates(self, date: str) -> List[str]:
        """Get prior trading dates for relative volume computation."""
        warmup_days = max(0, CONFIG.VOLUME_LOOKBACK_DAYS)
        if warmup_days == 0:
            return []

        available = self.bronze_reader.get_available_dates('futures/trades', 'symbol=ES')
        weekday_dates = [
            d for d in available
            if datetime.strptime(d, '%Y-%m-%d').weekday() < 5
        ]
        if date not in weekday_dates:
            return []

        idx = weekday_dates.index(date)
        start_idx = max(0, idx - warmup_days)
        return weekday_dates[start_idx:idx]

    def _build_hourly_cumvol_table(
        self,
        date: str,
        current_ohlcv: pd.DataFrame
    ) -> Dict[str, Dict[int, float]]:
        """
        Build hourly cumulative volume lookup table for relative volume computation.

        Returns:
            Dict[date, Dict[hour, cumvol]] where hour is ET hour (9-15)
            and cumvol is cumulative volume from 9:30 to end of that hour.
        """
        from datetime import time as dt_time

        hourly_cumvol: Dict[str, Dict[int, float]] = {}

        def _compute_hourly_cumvol(ohlcv: pd.DataFrame, date_str: str) -> Dict[int, float]:
            """Compute hourly cumulative volume for a single date."""
            if ohlcv.empty or 'timestamp' not in ohlcv.columns:
                return {}

            df = ohlcv.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York')
            df['hour_et'] = df['time_et'].dt.hour

            # Filter to RTH (9:30-16:00)
            rth_mask = (df['time_et'].dt.time >= dt_time(9, 30)) & (df['time_et'].dt.time < dt_time(16, 0))
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

        # Process prior dates for lookback
        prior_dates = self._get_volume_warmup_dates(date)
        for prior_date in prior_dates:
            trades_df = self.bronze_reader.read_futures_trades(symbol='ES', date=prior_date)
            trades = _futures_trades_from_df(trades_df)
            if not trades:
                continue
            ohlcv = build_ohlcv_vectorized(trades, convert_to_spy=True, freq='1min')
            if not ohlcv.empty:
                hourly_cumvol[prior_date] = _compute_hourly_cumvol(ohlcv, prior_date)

        # Add current date
        if not current_ohlcv.empty:
            hourly_cumvol[date] = _compute_hourly_cumvol(current_ohlcv, date)

        return hourly_cumvol

    def run(self, date: str, log_level: int = None) -> pd.DataFrame:
        """
        Run complete pipeline for a date.

        Args:
            date: Date string YYYY-MM-DD

        Returns:
            DataFrame with all signals
        """
        import time
        import logging

        if log_level is None:
            log_level = logging.INFO
        logging.basicConfig(level=log_level, format='%(asctime)s - %(message)s', force=True)
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
        atr_series = compute_atr_vectorized(ohlcv_df)
        warmup_2min, warmup_dates = self._build_sma_warmup_2min(date)
        if not warmup_2min.empty:
            ohlcv_2min = pd.concat([warmup_2min, ohlcv_2min], ignore_index=True).sort_values('timestamp')
            log.info(f"  SMA warmup: {len(warmup_2min):,} 2-min bars from {len(warmup_dates)} dates")

        # Build hourly cumulative volume lookup for relative volume computation
        hourly_cumvol = self._build_hourly_cumvol_table(date, ohlcv_df)
        vol_warmup_dates = self._get_volume_warmup_dates(date)
        log.info(f"  Volume warmup: {len(vol_warmup_dates)} prior dates for relative volume")

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
        dynamic_levels = compute_dynamic_level_series(
            ohlcv_df, ohlcv_2min, option_trades_df, date
        )

        static_prices = []
        static_kinds = []
        static_names = []
        for price, kind, name in zip(level_info.prices, level_info.kinds, level_info.kind_names):
            if name in ('ROUND', 'STRIKE'):
                static_prices.append(price)
                static_kinds.append(kind)
                static_names.append(name)
        static_level_info = LevelInfo(
            prices=np.array(static_prices, dtype=np.float64),
            kinds=np.array(static_kinds, dtype=np.int8),
            kind_names=static_names
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

        touches_df = detect_touches_vectorized(ohlcv_df, static_level_info, touch_tolerance=0.10)
        dynamic_touches = detect_dynamic_level_touches(ohlcv_df, dynamic_levels, touch_tolerance=0.10)
        if not dynamic_touches.empty:
            touches_df = pd.concat([touches_df, dynamic_touches], ignore_index=True)
            touches_df = touches_df.drop_duplicates(subset=['ts_ns', 'level_kind_name', 'level_price'])

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
        signals_df['direction_sign'] = np.where(signals_df['direction'] == 'UP', 1, -1)

        # Generate event IDs
        signals_df['event_id'] = [str(uuid.uuid4()) for _ in range(len(signals_df))]

        # Attach ATR for normalization
        atr_values = atr_series.to_numpy()
        bar_idx_vals = signals_df['bar_idx'].values.astype(np.int64)
        if bar_idx_vals.max(initial=0) >= len(atr_values):
            raise ValueError("Bar index exceeds ATR series length.")
        signals_df['atr'] = atr_values[bar_idx_vals]

        # Structural distances (pre-market levels)
        signals_df = compute_structural_distances(signals_df, ohlcv_df)

        # Mean reversion features (SMA-200/400)
        signals_df = compute_mean_reversion_features(signals_df, ohlcv_df, ohlcv_2min=ohlcv_2min)

        # Confluence features (stacked key levels)
        signals_df = compute_confluence_features_dynamic(signals_df, dynamic_levels)
        signals_df = compute_confluence_alignment(signals_df)

        # Dealer mechanics velocity features
        signals_df = compute_dealer_velocity_features(signals_df, option_trades_df)

        # Fluid pressure indicators
        signals_df = compute_pressure_indicators(signals_df)
        gamma_exposure = signals_df.get('gamma_exposure')
        if gamma_exposure is not None:
            gamma_vals = gamma_exposure.values.astype(np.float64)
            gamma_bucket = np.where(np.isfinite(gamma_vals), np.where(gamma_vals < 0, "SHORT_GAMMA", "LONG_GAMMA"), "UNKNOWN")
            signals_df['gamma_bucket'] = gamma_bucket

        # ========== Stage 8: Approach Context (Backward) ==========
        log.info("Computing approach context...")
        stage_start = time.time()

        signals_df = compute_approach_context_vectorized(signals_df, ohlcv_df)

        # Sparse feature transforms + normalization
        signals_df = add_sparse_feature_transforms(signals_df)
        signals_df = add_normalized_features(signals_df)

        # Attempt clustering + deterioration trends
        signals_df = compute_attempt_features(signals_df)

        # Hierarchical confluence level (requires dynamic_levels and hourly_cumvol)
        # Convert dynamic_levels dict to DataFrame
        dynamic_levels_df = pd.DataFrame(dynamic_levels)
        dynamic_levels_df['timestamp'] = ohlcv_df['timestamp'].values
        signals_df = compute_confluence_level_features(
            signals_df, dynamic_levels_df, hourly_cumvol, date
        )

        log.info(f"  Computed approach context in {time.time()-stage_start:.2f}s")

        # ========== Stage 9: Labeling (Forward) ==========
        log.info("Labeling outcomes...")
        stage_start = time.time()

        signals_df = label_outcomes_vectorized(signals_df, ohlcv_df)

        log.info(f"  Labeled {len(signals_df):,} signals in {time.time()-stage_start:.2f}s")

        # Restrict training signals to regular session (09:30-16:00 ET) with full forward window
        session_start = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        session_end = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(hours=16)
        session_start_ns = session_start.tz_convert("UTC").value
        session_end_ns = session_end.tz_convert("UTC").value
        max_confirm = max(CONFIG.CONFIRMATION_WINDOWS_MULTI or [CONFIG.CONFIRMATION_WINDOW_SECONDS])
        max_window_ns = int((max_confirm + CONFIG.LOOKFORWARD_MINUTES * 60) * 1e9)
        latest_end_ns = signals_df["ts_ns"].astype("int64") + max_window_ns
        rth_mask = (
            (signals_df["ts_ns"] >= session_start_ns)
            & (signals_df["ts_ns"] <= session_end_ns)
            & (latest_end_ns <= session_end_ns)
        )
        before = len(signals_df)
        signals_df = signals_df.loc[rth_mask].copy()
        log.info(f"  RTH filter kept {len(signals_df):,}/{before:,} signals (full forward window)")

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
        # Load ES trades from Bronze
        trades_df = self.bronze_reader.read_futures_trades(symbol='ES', date=date)
        if trades_df.empty:
            return [], [], pd.DataFrame()

        trades = _futures_trades_from_df(trades_df)

        # Load MBP-10 downsampled to snap cadence across RTH
        session_start = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        session_end = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(hours=16)
        session_start_ns = int(session_start.tz_convert("UTC").value)
        session_end_ns = int(session_end.tz_convert("UTC").value)
        buffer_ns = int(CONFIG.W_b * 1e9)
        ts_start = session_start_ns - buffer_ns
        ts_end = session_end_ns + buffer_ns

        mbp_df = self._read_mbp10_downsampled(date=date, start_ns=ts_start, end_ns=ts_end)
        if mbp_df.empty:
            raise ValueError(f"No MBP-10 data after downsampling for {date}")

        mbp10_snapshots = _mbp10_from_df(mbp_df)

        # Load options
        option_trades_df = self.bronze_reader.read_option_trades(underlying='SPY', date=date)

        return trades, mbp10_snapshots, option_trades_df

    def _read_mbp10_downsampled(self, date: str, start_ns: int, end_ns: int) -> pd.DataFrame:
        """
        Downsample MBP-10 snapshots to the configured snap cadence within a time window.
        Keeps the latest snapshot per bucket to preserve order book state.
        """
        base = Path(self.bronze_reader.bronze_root) / "futures" / "mbp10" / f"symbol=ES" / f"date={date}"
        if not base.exists():
            return pd.DataFrame()

        bucket_ns = int(CONFIG.SNAP_INTERVAL_MS * 1e6)
        glob_pattern = str(base / "**" / "*.parquet")
        query = f"""
            SELECT * EXCLUDE(bucket, rn)
            FROM (
                SELECT *,
                    CAST((ts_event_ns - {start_ns}) / {bucket_ns} AS BIGINT) AS bucket,
                    row_number() OVER (PARTITION BY bucket ORDER BY ts_event_ns DESC) AS rn
                FROM read_parquet('{glob_pattern}', hive_partitioning=true)
                WHERE ts_event_ns BETWEEN {start_ns} AND {end_ns}
            )
            WHERE rn = 1
            ORDER BY ts_event_ns
        """
        return self.bronze_reader.duckdb.execute(query).fetchdf()

    def _initialize_market_state(
        self,
        trades: List[FuturesTrade],
        mbp10_snapshots: List[MBP10],
        option_trades_df: pd.DataFrame,
        date: str
    ) -> Tuple[MarketState, pd.DataFrame]:
        """Initialize MarketState with vectorized Greeks computation."""
        buffer_seconds = max(CONFIG.W_b, CONFIG.CONFIRMATION_WINDOW_SECONDS)
        market_state = MarketState(max_buffer_window_seconds=buffer_seconds * 2)

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
    skip_download: bool = False,
    log_level: int = None
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

    bronze_reader = BronzeReader()

    # Discover dates
    if dates is None:
        all_dates = bronze_reader.get_available_dates('futures/trades', 'symbol=ES')
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
            signals_df = pipeline.run(date, log_level=log_level)
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
    combined_df = combined_df.sort_values(["date", "ts_ns"]).reset_index(drop=True)

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
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    log_level = None
    if args.verbose:
        import logging
        log_level = logging.DEBUG

    if args.list_dates:
        bronze_reader = BronzeReader()
        dates = bronze_reader.get_available_dates('futures/trades', 'symbol=ES')
        print(f"Available dates ({len(dates)}):")
        for d in dates:
            dt = datetime.strptime(d, '%Y-%m-%d')
            print(f"  {d} ({dt.strftime('%a')})")
        return 0

    if args.benchmark:
        print("Running benchmark...")
        import time

        bronze_reader = BronzeReader()
        dates = bronze_reader.get_available_dates('futures/trades', 'symbol=ES')
        if dates:
            date = dates[-1]
            pipeline = VectorizedPipeline()

            # Warmup
            print(f"Warmup run on {date}...")
            _ = pipeline.run(date, log_level=log_level)

            # Benchmark
            print(f"\nBenchmark run on {date}...")
            start = time.time()
            signals_df = pipeline.run(date, log_level=log_level)
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
        signals_df = pipeline.run(args.date, log_level=log_level)

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
        batch_process_vectorized(dates=dates, output_path=output_path, log_level=log_level)
        return 0

    if args.all:
        batch_process_vectorized(output_path=output_path, log_level=log_level)
        return 0

    # Default: process most recent date
    bronze_reader = BronzeReader()
    dates = bronze_reader.get_available_dates('futures/trades', 'symbol=ES')
    if dates:
        weekday_dates = [d for d in dates if datetime.strptime(d, '%Y-%m-%d').weekday() < 5]
        if weekday_dates:
            pipeline = VectorizedPipeline()
            signals_df = pipeline.run(weekday_dates[-1], log_level=log_level)
            return 0

    print("No dates available. Check dbn-data/ directory.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
