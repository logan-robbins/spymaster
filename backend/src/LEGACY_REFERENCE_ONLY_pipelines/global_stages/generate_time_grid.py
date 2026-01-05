"""
Stage: Generate Time Grid (Global Pipeline)
Type: Signal Generation (Time-Based)
Input: OHLCV Data (1min), Market State
Output: Signals DataFrame (Time Grid Events)

Transformation:
1. Creates a regular Time Grid (e.g., every 30 seconds) spanning the session.
2. Replaces "Price Interaction" events with "Time Sample" events.
3. Initializes Global Context:
   - Minutes since open
   - Bars since open
   - Opening Range active flag
   
Note: This is the entry point for the "Global" pipeline, which tracks market-wide state independent of specific price levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


class GenerateTimeGridStage(BaseStage):
    """
    Generate a regular time grid for global market features.
    
    Creates events at fixed intervals (default 30 seconds) throughout the session.
    Each event represents a point in time for which we compute market-wide features.
    """
    
    def __init__(self, interval_seconds: float = 30.0):
        """
        Args:
            interval_seconds: Interval between events (default 30s)
        """
        self.interval_seconds = interval_seconds
    
    @property
    def name(self) -> str:
        return "generate_time_grid"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['ohlcv_1min', 'market_state']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        ohlcv = ctx.data['ohlcv_1min']
        spot_price = ctx.data.get('spot_price', 0.0)
        
        # Determine session bounds from OHLCV data
        if ohlcv.empty:
            return {'signals_df': pd.DataFrame()}
        
        # Get timestamp column
        if 'timestamp' in ohlcv.columns:
            ts_col = ohlcv['timestamp']
        elif isinstance(ohlcv.index, pd.DatetimeIndex):
            ts_col = ohlcv.index
        else:
            return {'signals_df': pd.DataFrame()}
        
        # Session bounds (from data, will be filtered to RTH later)
        session_start = ts_col.min()
        session_end = ts_col.max()
        
        # Generate regular grid
        interval_ns = int(self.interval_seconds * 1e9)
        start_ns = session_start.value if hasattr(session_start, 'value') else int(session_start)
        end_ns = session_end.value if hasattr(session_end, 'value') else int(session_end)
        
        grid_ts = np.arange(start_ns, end_ns, interval_ns)
        n_events = len(grid_ts)
        
        if n_events == 0:
            return {'signals_df': pd.DataFrame()}
        
        # Build signals DataFrame
        signals_df = pd.DataFrame({
            'event_id': [f"global_{ctx.date}_{i:05d}" for i in range(n_events)],
            'ts_ns': grid_ts,
            'timestamp': pd.to_datetime(grid_ts, unit='ns', utc=True),
            'date': ctx.date,
            'spot': spot_price,  # Will be updated per-event in later stages
        })
        
        # Add session context
        session_open_ns = pd.Timestamp(ctx.date, tz='America/New_York').replace(
            hour=9, minute=30
        ).tz_convert('UTC').value
        
        signals_df['minutes_since_open'] = (signals_df['ts_ns'] - session_open_ns) / 1e9 / 60
        signals_df['bars_since_open'] = (signals_df['minutes_since_open'] / (self.interval_seconds / 60)).astype(int)
        
        # Opening range active flag (first 30 minutes)
        signals_df['or_active'] = (signals_df['minutes_since_open'] >= 0) & (signals_df['minutes_since_open'] <= 30)
        
        print(f"  Generated {n_events} time grid events at {self.interval_seconds}s intervals")
        print(f"  Session: {session_start} to {session_end}")
        
        return {
            'signals_df': signals_df,
        }

