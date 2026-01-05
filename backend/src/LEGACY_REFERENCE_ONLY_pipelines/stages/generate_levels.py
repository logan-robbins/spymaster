"""
Stage: Generate Levels
Type: Generation
Input: OHLCV bars (1min, 2min), Market State, Options Flow (for walls)
Output: Level Universe (LevelInfo), Dynamic Level Series

Transformation:
1. Generates structural levels based on price history and time:
   - Pre-Market High/Low (04:00-09:30 ET)
   - Opening Range High/Low (09:30-09:45 ET)
   - SMA-90 (90-period Simple Moving Average on 2min bars)
2. Filters the output to ONLY the single level specified in `ctx.level`.
   - This enforces the pipeline's "one level per run" architecture.
3. Computes dynamic time-series for the level (e.g. cumulative max for Highs, rolling mean for SMAs).

Note: This stage defines the "Target" around which all subsequent relative features are calculated.
"""
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import time as dt_time
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.core.market_state import OptionFlowAggregate
from src.common.config import CONFIG


@dataclass
class LevelInfo:
    """Level information for processing."""
    prices: np.ndarray
    kinds: np.ndarray  # integer codes for LevelKind
    kind_names: List[str]


def generate_level_universe(
    ohlcv_df: pd.DataFrame,
    option_flows: Dict[Tuple[float, str, str], OptionFlowAggregate],
    date: str,
    ohlcv_2min: Optional[pd.DataFrame] = None
) -> LevelInfo:
    """
    Generate ES system level universe (6 level kinds).
    
    Level types:
    - PM_HIGH/PM_LOW: Pre-market high/low (04:00-09:30 ET)
    - OR_HIGH/OR_LOW: Opening range (09:30-09:45 ET) high/low
    - SMA_90: 90-period Moving average on 2-min bars
    
    Returns:
        LevelInfo with arrays for processing
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

    # Ensure timestamp is datetime (can be index or column)
    df = ohlcv_df.copy()
    
    if isinstance(df.index, pd.DatetimeIndex):
        # Using DatetimeIndex
        df = df.reset_index()
        if 'timestamp' not in df.columns:
            df = df.rename(columns={'index': 'timestamp'})
    
    if 'timestamp' not in df.columns:
        raise ValueError("ohlcv_df must have DatetimeIndex or 'timestamp' column")
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Convert to ET for time-of-day logic
    df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.time

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

    # 3. SESSION HIGH/LOW (running) - REMOVED FOR V1
    # Too noisy, constantly changing - not useful for retrieval-based attribution

    # 4. SMA-90 / EMA-20 on 2-min bars
    if ohlcv_2min is None:
        df_2min = df.set_index('timestamp').resample('2min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_2min = df_2min.reset_index()
    else:
        df_2min = ohlcv_2min.copy()
        # Handle DatetimeIndex
        if isinstance(df_2min.index, pd.DatetimeIndex):
            df_2min = df_2min.reset_index()
            if 'timestamp' not in df_2min.columns:
                df_2min = df_2min.rename(columns={'index': 'timestamp'})

    if not df_2min.empty and 'timestamp' in df_2min.columns:
        df_2min = df_2min.sort_values('timestamp')

    if len(df_2min) >= 90:
        sma_90 = df_2min['close'].rolling(90).mean().iloc[-1]
        if pd.notna(sma_90):
            levels.append(sma_90)
            kinds.append(6)  # SMA_90 = re-coded to 6
            kind_names.append('SMA_90')

    # VWAP - REMOVED FOR V1
    # Lagging indicator, less useful for physics-based attribution
    
    # NOTE: ROUND and STRIKE levels removed for ES pipeline (handled via GEX features).

    # 6. CALL_WALL / PUT_WALL (max gamma concentration)
    # 6. CALL WALL / PUT WALL - REMOVED FOR V1
    # GEX treated as FEATURES (fuel field), not levels themselves
    # GEX will be computed per-event as strike-banded features around the tested level

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
    """Compute rolling call/put wall strikes using option flow data."""
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
    opt_df['dealer_flow'] = (
        -opt_df['aggressor']
        * opt_df['size']
        * opt_df['gamma']
        * CONFIG.OPTION_CONTRACT_MULTIPLIER
    )

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

        # Get timestamp from index or column
        if isinstance(ohlcv_df.index, pd.DatetimeIndex):
            ohlcv_minutes = ohlcv_df.index.floor('1min')
        else:
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
    """Build per-bar dynamic level series (causal) for structural levels."""
    df = ohlcv_df.copy()
    
    # Handle DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if 'timestamp' not in df.columns:
            df = df.rename(columns={'index': 'timestamp'})
    
    df = df.sort_values('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.time

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
    
    # Handle DatetimeIndex for ohlcv_2min
    if isinstance(df_2min.index, pd.DatetimeIndex):
        df_2min = df_2min.reset_index()
        if 'timestamp' not in df_2min.columns:
            df_2min = df_2min.rename(columns={'index': 'timestamp'})
    
    df_2min = df_2min.sort_values('timestamp')
    df_2min['timestamp'] = pd.to_datetime(df_2min['timestamp'], utc=True)
    sma_90_series = df_2min['close'].rolling(90).mean()
    sma_df = pd.DataFrame({
        'timestamp': df_2min['timestamp'],
        'sma_90': sma_90_series
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
        'SMA_90': sma_aligned['sma_90'],
        'CALL_WALL': call_wall_series,
        'PUT_WALL': put_wall_series
    }


class GenerateLevelsStage(BaseStage):
    """Generate level universe FOR THE SPECIFIED LEVEL ONLY.
    
    Level-specific pipeline: Generates ONLY the level specified by ctx.level.
    This follows the "one level at a time" principle for feature creation.

    Supported levels:
    - pm_high, pm_low: Pre-market high/low (04:00-09:30 ET)
    - or_high, or_low: Opening range high/low (09:30-09:45 ET)
    - sma_90: 90-period moving average on 2min bars
    - ema_20: 20-period exponential moving average

    Outputs:
        level_info: LevelInfo with SINGLE level
        dynamic_levels: Dict with per-bar series for THIS level only
    """

    @property
    def name(self) -> str:
        return "generate_levels"

    @property
    def required_inputs(self) -> List[str]:
        return ['ohlcv_1min', 'market_state']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        ohlcv_df = ctx.data['ohlcv_1min']
        market_state = ctx.data['market_state']
        ohlcv_2min = ctx.data.get('ohlcv_2min')
        option_trades_df = ctx.data.get('option_trades_df', pd.DataFrame())

        # Get target level from context
        target_level = ctx.level.upper() if ctx.level else None
        if not target_level:
            raise ValueError("ctx.level must be specified for level-specific pipeline")

        # Generate ALL levels first (needed for computation)
        all_level_info = generate_level_universe(
            ohlcv_df,
            market_state.option_flows,
            ctx.date,
            ohlcv_2min=ohlcv_2min
        )
        
        # Compute dynamic series for all levels
        all_dynamic_levels = compute_dynamic_level_series(
            ohlcv_df, ohlcv_2min, option_trades_df, ctx.date
        )

        # Filter to ONLY the target level
        target_idx = None
        for i, name in enumerate(all_level_info.kind_names):
            if name == target_level:
                target_idx = i
                break
        
        if target_idx is None:
            raise ValueError(f"Target level '{target_level}' not found. Available: {all_level_info.kind_names}")
        
        # Extract single level
        level_info = LevelInfo(
            prices=np.array([all_level_info.prices[target_idx]], dtype=np.float64),
            kinds=np.array([all_level_info.kinds[target_idx]], dtype=np.int8),
            kind_names=[target_level]
        )
        
        # Extract single dynamic series for target level
        dynamic_levels = {target_level: all_dynamic_levels[target_level]}

        return {
            'level_info': level_info,
            'dynamic_levels': dynamic_levels,
        }
