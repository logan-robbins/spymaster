"""
Stage: Compute Market Kinematics (Global Pipeline)
Type: Feature Engineering (Market-Wide)
Input: Signals DataFrame (Time Grid), OHLCV Data (10s)
Output: Signals DataFrame with Global Kinematic Features

Transformation:
1. Computes Price Motion Derivatives relative to Time:
   - Velocity: Rate of change of price ($/min).
   - Acceleration: Rate of change of velocity ($/min²).
   - Jerk: Rate of change of acceleration ($/min³).
2. Computes both Unsigned (Absolute for Volatility) and Signed (Directional for Trend) metrics.
3. Aggregates over multiple windows (1m, 2m, 5m, 10m, 20m).
   
Note: High absolute kinematics imply high volatility/instability. High signed kinematics imply strong directional trend.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.compute.kinematics import compute_kinematics_windows


class ComputeMarketKinematicsStage(BaseStage):
    """
    Compute market-wide kinematics at multiple windows.
    
    Unlike level-relative kinematics which are direction-signed,
    this computes both raw and absolute kinematics.
    
    Features:
    - velocity_1min, velocity_2min, etc.: Raw velocity (positive = up, negative = down)
    - velocity_1min_abs, etc.: Absolute velocity (magnitude only)
    - acceleration_*, jerk_*: Same pattern
    """
    
    @property
    def name(self) -> str:
        return "compute_market_kinematics"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_10s']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        ohlcv_df = ctx.data.get('ohlcv_10s', pd.DataFrame())
        
        if signals_df.empty or ohlcv_df.empty:
            return {'signals_df': signals_df}
        
        signal_ts = signals_df['ts_ns'].values.astype(np.int64)
        
        # Compute kinematics (unsigned mode)
        kinematics = compute_kinematics_windows(
            signal_ts=signal_ts,
            ohlcv_df=ohlcv_df,
            windows_minutes=[1, 2, 3, 5, 10, 20],
            directions=None,  # No direction - global mode
            signed=False,  # Get both raw and absolute
        )
        
        # Add features to DataFrame
        for name, values in kinematics.items():
            signals_df[name] = values
        
        # Add momentum (price change over windows)
        if 'velocity_1min' in signals_df.columns:
            # Momentum = velocity * time (approximate)
            signals_df['momentum_1min'] = signals_df['velocity_1min'] * 60  # points in 1 min
        if 'velocity_5min' in signals_df.columns:
            signals_df['momentum_5min'] = signals_df['velocity_5min'] * 300
        
        print(f"  Computed market kinematics for {len(signals_df)} events")
        if 'velocity_1min' in signals_df.columns:
            print(f"  Velocity_1min: min={signals_df['velocity_1min'].min():.4f}, max={signals_df['velocity_1min'].max():.4f}")
        
        return {'signals_df': signals_df}

