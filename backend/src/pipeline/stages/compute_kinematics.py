"""
Compute kinematic features in the level frame.

Per Final Call v1 spec Section 6.3: Velocity, acceleration, jerk detect phase transitions.

Level frame physics:
- x(t) = SPX_mid(t) - L(t)  (distance to level)
- p(t) = dir_sign × x(t)     (direction-aligned distance, increasing = approaching break)
- v(t) = d/dt p(t)           (velocity)
- a(t) = d/dt v(t)           (acceleration)
- j(t) = d/dt a(t)           (jerk - rate of acceleration change)

These detect:
- v > 0, a > 0: Accelerating toward break
- v > 0, a < 0: Decelerating into level (potential bounce setup)
- v < 0: Moving toward bounce (wrong direction)
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_level_frame_kinematics(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    lookback_minutes: int = 10
) -> pd.DataFrame:
    """
    Compute kinematics in the level frame (position, velocity, acceleration, jerk).
    
    Per Final Call spec: These features capture the "phase transition" dynamics
    as price approaches and interacts with a level.
    
    Args:
        signals_df: DataFrame with signals (must have level_price, direction, ts_ns)
        ohlcv_df: OHLCV DataFrame with SPX prices
        lookback_minutes: Window for kinematic computation
    
    Returns:
        DataFrame with kinematic features added
    """
    if signals_df.empty or ohlcv_df.empty:
        return signals_df
    
    # Prepare OHLCV arrays
    ohlcv_sorted = ohlcv_df.sort_values('timestamp')
    ohlcv_ts = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    
    lookback_ns = int(lookback_minutes * 60 * 1e9)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    level_prices = signals_df['level_price'].values.astype(np.float64)
    directions = signals_df['direction'].values
    
    # Initialize output arrays
    position = np.full(n, np.nan, dtype=np.float64)  # p(t) = dir_sign × (spot - L)
    velocity = np.full(n, np.nan, dtype=np.float64)  # v(t) = dp/dt
    acceleration = np.full(n, np.nan, dtype=np.float64)  # a(t) = dv/dt
    jerk = np.full(n, np.nan, dtype=np.float64)  # j(t) = da/dt
    kinetic_energy = np.full(n, np.nan, dtype=np.float64)  # KE ∝ v²
    deceleration_flag = np.zeros(n, dtype=np.int8)  # v > 0 but a < 0
    
    for i in range(n):
        ts = signal_ts[i]
        level = level_prices[i]
        direction = directions[i]
        dir_sign = 1.0 if direction == 'UP' else -1.0
        
        # Find lookback window
        start_ts = ts - lookback_ns
        start_idx = np.searchsorted(ohlcv_ts, start_ts, side='right')
        end_idx = np.searchsorted(ohlcv_ts, ts, side='right')
        
        if start_idx >= end_idx or end_idx == 0:
            continue
        
        # Extract price history
        hist_close = ohlcv_close[start_idx:end_idx]
        hist_ts = ohlcv_ts[start_idx:end_idx]
        
        if len(hist_close) < 4:  # Need at least 4 points for jerk
            continue
        
        # Compute level-frame position series: p(t) = dir_sign × (price - level)
        p = dir_sign * (hist_close - level)
        
        # Apply causal smoothing (EWMA) to reduce noise
        alpha = 0.3  # Smoothing factor
        p_smooth = pd.Series(p).ewm(alpha=alpha, adjust=False).mean().values
        
        # Compute derivatives using finite differences
        dt = np.diff(hist_ts) / 1e9  # Time steps in seconds
        
        # Velocity: dp/dt
        v = np.diff(p_smooth) / dt
        
        # Acceleration: dv/dt
        if len(v) >= 2:
            a = np.diff(v) / dt[1:]
        else:
            a = np.array([])
        
        # Jerk: da/dt
        if len(a) >= 2:
            j = np.diff(a) / dt[2:]
        else:
            j = np.array([])
        
        # Store latest values at anchor time
        position[i] = p_smooth[-1]
        
        if len(v) > 0:
            velocity[i] = v[-1]
            kinetic_energy[i] = v[-1] ** 2  # KE ∝ v²
        
        if len(a) > 0:
            acceleration[i] = a[-1]
            
            # Deceleration flag: moving in break direction but slowing down
            if velocity[i] > 0 and acceleration[i] < 0:
                deceleration_flag[i] = 1
        
        if len(j) > 0:
            jerk[i] = j[-1]
    
    result = signals_df.copy()
    result['position'] = position
    result['velocity'] = velocity
    result['acceleration'] = acceleration
    result['jerk'] = jerk
    result['kinetic_energy'] = kinetic_energy
    result['deceleration_flag'] = deceleration_flag
    
    return result


class ComputeKinematicsStage(BaseStage):
    """
    Compute level-frame kinematics (position, velocity, acceleration, jerk).
    
    Per Final Call v1 spec Section 6.3: Kinematics in level frame capture
    phase transitions as price approaches/interacts with structural levels.
    
    Outputs:
        signals_df: Updated with kinematic features
    """
    
    @property
    def name(self) -> str:
        return "compute_kinematics"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']
        
        signals_df = compute_level_frame_kinematics(
            signals_df=signals_df,
            ohlcv_df=ohlcv_df,
            lookback_minutes=CONFIG.LOOKBACK_MINUTES
        )
        
        return {'signals_df': signals_df}

