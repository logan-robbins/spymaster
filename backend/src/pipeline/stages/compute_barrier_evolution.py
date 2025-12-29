"""
Barrier evolution features - how depth changes over time.

Per user requirement: Multi-window encoding of barrier dynamics.

A "thinning barrier" (depth decreasing) is bullish for BREAK.
A "thickening barrier" (depth increasing) is bullish for BOUNCE.

Windows:
- 1min: Immediate barrier change (absorbing? building?)
- 3min: Short-term barrier trend
- 5min: Medium-term barrier evolution
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10


def compute_barrier_evolution(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    windows_minutes: List[float] = None
) -> pd.DataFrame:
    """
    Compute barrier depth evolution at multiple lookback windows.
    
    Measures: barrier_depth(now) - barrier_depth(t-window)
    - Positive delta: Barrier thickening (bullish for BOUNCE)
    - Negative delta: Barrier thinning (bullish for BREAK)
    
    Args:
        signals_df: DataFrame with signals
        mbp10_snapshots: List of MBP-10 snapshots
        windows_minutes: List of lookback windows (default: [1, 3, 5])
    
    Returns:
        DataFrame with barrier evolution features
    """
    if windows_minutes is None:
        windows_minutes = [1.0, 3.0, 5.0]
    
    if signals_df.empty or not mbp10_snapshots:
        result = signals_df.copy()
        for window_min in windows_minutes:
            suffix = f'_{int(window_min)}min'
            result[f'barrier_delta{suffix}'] = 0.0
            result[f'barrier_pct_change{suffix}'] = 0.0
        return result
    
    # Build barrier depth time series
    mbp_times = np.array([mbp.ts_event_ns for mbp in mbp10_snapshots], dtype=np.int64)
    
    # Compute depth at best bid/ask (top level)
    mbp_depths = []
    for mbp in mbp10_snapshots:
        if len(mbp.levels) > 0:
            # Total depth at top 5 levels
            depth = sum(level.bid_sz + level.ask_sz for level in mbp.levels[:5])
            mbp_depths.append(depth)
        else:
            mbp_depths.append(0.0)
    
    mbp_depths = np.array(mbp_depths, dtype=np.float64)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    level_prices = signals_df['level_price'].values.astype(np.float64)
    
    result = signals_df.copy()
    
    # Compute for each window
    for window_min in windows_minutes:
        lookback_ns = int(window_min * 60 * 1e9)
        
        barrier_delta = np.zeros(n, dtype=np.float64)
        barrier_pct_change = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            ts = signal_ts[i]
            start_ts = ts - lookback_ns
            level = level_prices[i]
            
            # Find MBP snapshot at current time (closest)
            current_idx = np.searchsorted(mbp_times, ts, side='right') - 1
            if current_idx < 0 or current_idx >= len(mbp_times):
                continue
            
            # Find MBP snapshot at t-window (closest)
            past_idx = np.searchsorted(mbp_times, start_ts, side='right') - 1
            if past_idx < 0:
                continue
            
            current_depth = mbp_depths[current_idx]
            past_depth = mbp_depths[past_idx]
            
            # Delta
            barrier_delta[i] = current_depth - past_depth
            
            # Percent change
            if past_depth > 0:
                barrier_pct_change[i] = (current_depth - past_depth) / past_depth
        
        # Add to result
        suffix = f'_{int(window_min)}min'
        result[f'barrier_delta{suffix}'] = barrier_delta
        result[f'barrier_pct_change{suffix}'] = barrier_pct_change
    
    # Current snapshot depth (for reference)
    current_depths = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ts = signal_ts[i]
        current_idx = np.searchsorted(mbp_times, ts, side='right') - 1
        if 0 <= current_idx < len(mbp_times):
            current_depths[i] = mbp_depths[current_idx]
    
    result['barrier_depth_current'] = current_depths
    
    return result


class ComputeBarrierEvolutionStage(BaseStage):
    """Compute barrier evolution at multiple windows.
    
    Outputs:
        signals_df: Updated with barrier evolution features
    """
    
    @property
    def name(self) -> str:
        return "compute_barrier_evolution"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        mbp10_snapshots = ctx.data.get('mbp10', [])
        
        signals_df = compute_barrier_evolution(
            signals_df=signals_df,
            mbp10_snapshots=mbp10_snapshots,
            windows_minutes=[1.0, 3.0, 5.0]
        )
        
        return {'signals_df': signals_df}

