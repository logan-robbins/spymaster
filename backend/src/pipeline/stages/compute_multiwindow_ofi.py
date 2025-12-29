"""
Multi-window OFI features for pressure encoding across timescales.

Integrated OFI at multiple windows captures:
- 30s: Immediate pressure (last few snapshots)
- 60s: Short-term flow
- 120s: Medium-term cumulative pressure
- 300s: Long-term order flow trend
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10


def compute_multiwindow_ofi(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    windows_seconds: List[float] = None
) -> pd.DataFrame:
    """
    Compute integrated OFI at multiple lookback windows.
    
    Args:
        signals_df: DataFrame with signals
        mbp10_snapshots: List of MBP-10 snapshots
        windows_seconds: List of lookback windows (default: [30, 60, 120, 300])
    
    Returns:
        DataFrame with multi-window OFI features
    """
    if windows_seconds is None:
        windows_seconds = [30.0, 60.0, 120.0, 300.0]
    
    if signals_df.empty or not mbp10_snapshots:
        result = signals_df.copy()
        for window_sec in windows_seconds:
            suffix = f'_{int(window_sec)}s'
            result[f'ofi{suffix}'] = 0.0
            result[f'ofi_near_level{suffix}'] = 0.0
        return result
    
    # Build MBP-10 time series
    mbp_times = np.array([mbp.ts_event_ns for mbp in mbp10_snapshots], dtype=np.int64)
    
    # Pre-compute OFI for each snapshot
    # OFI = Δbid_size - Δask_size (simplified)
    # Full OFI requires tracking order book changes
    # For now, use bid-ask imbalance as proxy
    
    mbp_imbalances = []
    for mbp in mbp10_snapshots:
        bid_depth = sum(level.bid_sz for level in mbp.levels[:5])  # Top 5
        ask_depth = sum(level.ask_sz for level in mbp.levels[:5])
        imbalance = bid_depth - ask_depth
        mbp_imbalances.append(imbalance)
    
    mbp_imbalances = np.array(mbp_imbalances, dtype=np.float64)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    
    result = signals_df.copy()
    
    # Compute for each window
    for window_sec in windows_seconds:
        lookback_ns = int(window_sec * 1e9)
        
        ofi_window = np.zeros(n, dtype=np.float64)
        ofi_near_level = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            ts = signal_ts[i]
            start_ts = ts - lookback_ns
            
            # Find MBP snapshots in window
            mask = (mbp_times >= start_ts) & (mbp_times <= ts)
            
            if mask.sum() > 0:
                # Integrated OFI (cumulative imbalance in window)
                ofi_window[i] = mbp_imbalances[mask].sum()
                
                # Average imbalance (normalized by snapshot count)
                ofi_near_level[i] = mbp_imbalances[mask].mean()
        
        # Add to result with window suffix
        suffix = f'_{int(window_sec)}s'
        result[f'ofi{suffix}'] = ofi_window
        result[f'ofi_near_level{suffix}'] = ofi_near_level
    
    # Derived: OFI acceleration (is pressure building or fading?)
    # Compare short window vs long window
    if 30.0 in windows_seconds and 120.0 in windows_seconds:
        result['ofi_acceleration'] = result['ofi_30s'] / (result['ofi_120s'] + 1e-6)
        # > 1.0 = pressure accelerating (recent > historical avg)
        # < 1.0 = pressure fading
    
    return result


class ComputeMultiWindowOFIStage(BaseStage):
    """Compute OFI at multiple lookback windows.
    
    Encodes order flow pressure across timescales.
    
    Outputs:
        signals_df: Updated with multi-window OFI features
    """
    
    @property
    def name(self) -> str:
        return "compute_multiwindow_ofi"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        mbp10_snapshots = ctx.data.get('mbp10', [])
        
        # Compute multi-window OFI
        signals_df = compute_multiwindow_ofi(
            signals_df=signals_df,
            mbp10_snapshots=mbp10_snapshots,
            windows_seconds=[30.0, 60.0, 120.0, 300.0]
        )
        
        return {'signals_df': signals_df}

