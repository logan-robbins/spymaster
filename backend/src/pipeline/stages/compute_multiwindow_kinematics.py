"""
Multi-window kinematic features for setup encoding.

Per user requirement: "What lookback timeframe defines the setup?"
Answer: ALL of them! Encode physics at multiple timescales.

Windows:
- 1min: Immediate entry dynamics
- 3min: Short-term momentum
- 5min: Medium-term approach
- 10min: Long-term trend
- 20min: Pre-approach context (optional)

This enables kNN to match on:
- "Fast aggressive approach" (high velocity_1min)
- "Steady approach" (consistent velocity across windows)
- "Decelerating entry" (velocity_1min < velocity_5min)
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_multiwindow_kinematics(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    windows_minutes: List[int] = None
) -> pd.DataFrame:
    """
    Compute kinematics at multiple lookback windows.
    
    Per user requirement: Encode setup across timescales UP TO 20 MINUTES.
    
    Windows:
    - 1min: Immediate entry dynamics
    - 3min: Short-term momentum
    - 5min: Medium-term approach
    - 10min: Long-term trend
    - 20min: Pre-approach context (what happened BEFORE approach started?)
    
    Args:
        signals_df: DataFrame with signals
        ohlcv_df: OHLCV DataFrame
        windows_minutes: List of lookback windows (default: [1, 3, 5, 10, 20])
    
    Returns:
        DataFrame with multi-window kinematic features
    """
    if windows_minutes is None:
        windows_minutes = [1, 3, 5, 10, 20]  # Up to 20min per user requirement
    
    if signals_df.empty or ohlcv_df.empty:
        return signals_df
    
    # Prepare OHLCV arrays
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
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    level_prices = signals_df['level_price'].values.astype(np.float64)
    directions = signals_df['direction'].values
    
    # Direction sign for level-frame coordinates
    dir_sign = np.where(directions == 'UP', 1, -1)
    
    result = signals_df.copy()
    
    # Compute for each window
    for window_min in windows_minutes:
        lookback_ns = int(window_min * 60 * 1e9)
        
        velocity = np.zeros(n, dtype=np.float64)
        acceleration = np.zeros(n, dtype=np.float64)
        jerk = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            ts = signal_ts[i]
            start_ts = ts - lookback_ns
            level = level_prices[i]
            
            # Find bars in lookback window
            start_idx = np.searchsorted(ohlcv_ts, start_ts, side='right')
            end_idx = np.searchsorted(ohlcv_ts, ts, side='right')
            
            if end_idx <= start_idx + 2:
                # Not enough bars
                continue
            
            # Extract price series in window
            window_prices = ohlcv_close[start_idx:end_idx]
            window_times = ohlcv_ts[start_idx:end_idx]
            
            if len(window_prices) < 3:
                continue
            
            # Level-frame position: p(t) = dir_sign × (price - level)
            # Increasing p = moving toward break, decreasing = toward bounce
            p_series = dir_sign[i] * (window_prices - level)
            
            # Velocity: dp/dt (first derivative)
            dt_series = np.diff(window_times) / 1e9  # seconds
            if len(dt_series) > 0 and np.all(dt_series > 0):
                dp_series = np.diff(p_series)
                v_series = dp_series / dt_series
                velocity[i] = v_series[-1]  # Latest velocity
                
                # Acceleration: dv/dt (second derivative)
                if len(v_series) > 1:
                    dv_series = np.diff(v_series)
                    dt2_series = dt_series[:-1]  # Align with dv
                    if len(dt2_series) > 0 and np.all(dt2_series > 0):
                        a_series = dv_series / dt2_series
                        acceleration[i] = a_series[-1]  # Latest acceleration
                        
                        # Jerk: da/dt (third derivative)
                        if len(a_series) > 1:
                            da_series = np.diff(a_series)
                            dt3_series = dt2_series[:-1]
                            if len(dt3_series) > 0 and np.all(dt3_series > 0):
                                j_series = da_series / dt3_series
                                jerk[i] = j_series[-1]
        
        # Add to result with window suffix
        suffix = f'_{window_min}min'
        result[f'velocity{suffix}'] = velocity
        result[f'acceleration{suffix}'] = acceleration
        result[f'jerk{suffix}'] = jerk
        
        # Derived: Momentum consistency (velocity trend)
        if window_min > 1:
            # Is velocity increasing or decreasing across the window?
            # Positive trend = accelerating, negative = decelerating
            v_start = velocity.copy()
            v_end = velocity.copy()
            # Compute velocity at start and end of window
            # (simplified: just use the computed velocity as "end")
            # For now, mark with acceleration as proxy
            result[f'momentum_trend{suffix}'] = acceleration
    
    return result


class ComputeMultiWindowKinematicsStage(BaseStage):
    """Compute kinematics at multiple lookback windows.
    
    Encodes the "setup" across timescales for kNN retrieval.
    
    Outputs:
        signals_df: Updated with multi-window kinematic features
    """
    
    @property
    def name(self) -> str:
        return "compute_multiwindow_kinematics"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']
        
        # Compute multi-window kinematics (up to 20min per user requirement)
        signals_df = compute_multiwindow_kinematics(
            signals_df=signals_df,
            ohlcv_df=ohlcv_df,
            windows_minutes=[1, 3, 5, 10, 20]
        )
        
        return {'signals_df': signals_df}
