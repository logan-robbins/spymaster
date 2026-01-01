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
    
    # Pre-compute OFI derived from MBP transitions
    # True OFI = Order Flow at best bid/ask
    ofi_times = []
    ofi_values = []
    
    if len(mbp10_snapshots) > 1:
        prev_mbp = mbp10_snapshots[0]
        # Sort snapshots by time to be safe
        sorted_snapshots = sorted(mbp10_snapshots, key=lambda x: x.ts_event_ns)
        
        for curr_mbp in sorted_snapshots[1:]:
            # Simplified OFI at Top of Book
            # OFI = q_b_t * I(b_t >= b_t-1) - q_b_t-1 * I(b_t <= b_t-1)
            #     - q_a_t * I(a_t <= a_t-1) + q_a_t-1 * I(a_t >= a_t-1)
            
            # Extract top level
            b_t = curr_mbp.levels[0].bid_px
            q_b_t = curr_mbp.levels[0].bid_sz
            a_t = curr_mbp.levels[0].ask_px
            q_a_t = curr_mbp.levels[0].ask_sz
            
            b_prev = prev_mbp.levels[0].bid_px
            q_b_prev = prev_mbp.levels[0].bid_sz
            a_prev = prev_mbp.levels[0].ask_px
            q_a_prev = prev_mbp.levels[0].ask_sz
            
            ofi_bid = 0.0
            if b_t > b_prev:
                ofi_bid = q_b_t
            elif b_t < b_prev:
                ofi_bid = -q_b_prev
            else: # b_t == b_prev
                ofi_bid = q_b_t - q_b_prev
                
            ofi_ask = 0.0
            if a_t < a_prev:
                ofi_ask = q_a_t
            elif a_t > a_prev:
                ofi_ask = -q_a_prev
            else: # a_t == a_prev
                ofi_ask = q_a_t - q_a_prev
                
            # Net OFI = Bid inflow - Ask inflow
            net_ofi = ofi_bid - ofi_ask
            
            ofi_times.append(curr_mbp.ts_event_ns)
            ofi_values.append(net_ofi)
            
            prev_mbp = curr_mbp
            
    ofi_times = np.array(ofi_times, dtype=np.int64)
    ofi_values = np.array(ofi_values, dtype=np.float64)
    
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
            
            # Find OFI events in window
            # Use searchsorted for speed
            start_idx = np.searchsorted(ofi_times, start_ts, side='right')
            end_idx = np.searchsorted(ofi_times, ts, side='right')
            
            if end_idx > start_idx:
                window_flows = ofi_values[start_idx:end_idx]
                
                # Integrated OFI (Flow)
                ofi_window[i] = np.sum(window_flows)
                
                # For "near level" we might want density, but sticking to 
                # original request: "Average imbalance" -> "Average Flow Rate"?
                # Actually, the original code computed "Average Imbalance" (OBI).
                # To capture "Pressure Intensity", mean flow per event is reasonable.
                ofi_near_level[i] = np.mean(window_flows)
        
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
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        
        # Compute multi-window OFI
        signals_df = compute_multiwindow_ofi(
            signals_df=signals_df,
            mbp10_snapshots=mbp10_snapshots,
            windows_seconds=[30.0, 60.0, 120.0, 300.0]
        )
        
        return {'signals_df': signals_df}

