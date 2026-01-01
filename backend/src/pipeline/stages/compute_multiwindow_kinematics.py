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
        windows_minutes = [1, 2, 3, 5, 10, 20]  # Multi-scale per RESEARCH.md
    
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
    if 'ts_ns' in ohlcv_sorted.columns:
         ohlcv_ts = ohlcv_sorted['ts_ns'].values.astype(np.int64)
    else:
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
            
            if len(window_prices) < 2:
                if i < 5 and window_min == 2:
                     print(f"DEBUG: wind={window_min}, len={len(window_prices)}, start={start_idx}, end={end_idx}, ts={ts}, start_ts={start_ts}")
                continue
            
            # Level-frame position: p(t) = dir_sign * (price - level)
            p_series = dir_sign[i] * (window_prices - level)
            t_series = (window_times - window_times[0]) / 1e9  # normalize time to start at 0 (seconds)

            # Robust Velocity: Linear regression slope of position over time
            # v_window = cov(t, p) / var(t)
            if len(p_series) >= 2:
                # Velocity (Slope of position)
                A = np.vstack([t_series, np.ones(len(t_series))]).T
                m_v, c_v = np.linalg.lstsq(A, p_series, rcond=None)[0]
                velocity[i] = m_v

                # Acceleration: Linear regression slope of velocity
                # We need local velocities to estimate acceleration trend
                # Split window into smaller chunks or use 2nd order polyfit
                # Here we use 2nd order polyfit: p(t) = 0.5*a*t^2 + v*t + x0
                # The 'a' coefficient is acceleration
                if len(p_series) >= 3:
                    coeffs = np.polyfit(t_series, p_series, 2)
                    # p(t) = c[0]*t^2 + c[1]*t + c[2]
                    # v(t) = 2*c[0]*t + c[1]
                    # a(t) = 2*c[0]
                    # Note: We overwrite the linear velocity with the instantaneous velocity 
                    # from the quadratic fit at the end of the window to allow for curvature
                    # But for "Average Window Velocity" we should stick to linear slope?
                    # RESEARCH.md implies we want "Trajectory", so quadratic fit is better
                    # as it captures the curvature (acceleration).
                    
                    # acceleration = 2 * c[0]
                    acceleration[i] = 2.0 * coeffs[0]
                    
                    # Refine velocity to be the velocity at the END of the fitted curve
                    # velocity[i] = 2 * coeffs[0] * t_series[-1] + coeffs[1] 
                    # Actually, stick to linear slope for "Velocity" feature to keep it robust
                    # and use coeffs[0] for "Acceleration".
                    
                    # Jerk: Change in acceleration
                    # Requires 3rd order fit: p(t) = (1/6)j*t^3 + 0.5*a*t^2 + v*t + x0
                    if len(p_series) >= 4:
                        coeffs_j = np.polyfit(t_series, p_series, 3)
                        # p(t) = c[0]t^3 + c[1]t^2 + c[2]t + c[3]
                        # a(t) = 6*c[0]t + 2*c[1]
                        # j(t) = 6*c[0]
                        jerk[i] = 6.0 * coeffs_j[0]

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
        return ['signals_df', 'ohlcv_10s']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_10s']
        
        # Compute multi-window kinematics (up to 20min per user requirement)
        # Using 10s bars allows high-resolution physics at short windows:
        # - 1min window = 6 bars (Valid Velocity, Accel, Jerk)
        # - 2min window = 12 bars (Robust)
        signals_df = compute_multiwindow_kinematics(
            signals_df=signals_df,
            ohlcv_df=ohlcv_df,
            windows_minutes=[1, 2, 3, 5, 10, 20]
        )
        
        return {'signals_df': signals_df}
