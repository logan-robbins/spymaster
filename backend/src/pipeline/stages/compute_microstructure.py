"""
Microstructure features - Vacuum, Latency, and Velocity.

Per Microstructure Research (Phase 5):
- Vacuum Duration: Time (ms) with depth < threshold.
- Replenishment Latency: Time (ms) to refill after trade.
- Gamma Velocity: Rate of change of options delta.

Designed for 15s "Goldilocks" cadence.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10
from src.common.config import CONFIG


def compute_microstructure_features(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    vacuum_threshold: int = 5,  # Contracts
    restore_threshold: int = 20 # Contracts
) -> pd.DataFrame:
    """
    Compute microstructure features by scanning high-frequency MBP stream
    and latching max values into signal windows.
    
    Args:
        signals_df: DataFrame with signals (15s cadence)
        mbp10_snapshots: List of raw MBP-10 snapshots
        vacuum_threshold: low liquidity threshold (contracts)
        restore_threshold: replenishment threshold (contracts)
    
    Returns:
        DataFrame with added microstructure columns
    """
    if signals_df.empty or not mbp10_snapshots:
        result = signals_df.copy()
        result['vacuum_duration_ms'] = 0.0
        result['replenishment_latency_ms'] = 0.0
        return result
    
    # Sort snapshots by time
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda mbp: mbp.ts_event_ns)
    mbp_times = np.array([mbp.ts_event_ns for mbp in sorted_snapshots], dtype=np.int64)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    
    # Output arrays
    vacuum_durations = np.zeros(n, dtype=np.float64)
    replenishment_latencies = np.zeros(n, dtype=np.float64)
    
    # Scan logic: 
    # For each signal row at T, scan window [T-15s, T]
    # LATCH the MAX duration/latency found in that window.
    
    window_ns = 15 * 1_000_000_000  # 15 seconds in ns
    
    for i in range(n):
        ts = signal_ts[i]
        start_ts = ts - window_ns
        
        # Find raw events in this 15s window
        start_idx = np.searchsorted(mbp_times, start_ts, side='right')
        end_idx = np.searchsorted(mbp_times, ts, side='right')
        
        if end_idx <= start_idx:
            continue
            
        window_snapshots = sorted_snapshots[start_idx:end_idx]
        
        max_vacuum_ms = 0.0
        
        # ─── Vacuum Detection ───
        # State machine: Normal -> Vacuum -> Normal
        vacuum_start_ns = None
        
        for mbp in window_snapshots:
            # Check Best Bid/Ask Depth
            if not mbp.levels:
                continue
                
            best_bid_sz = mbp.levels[0].bid_sz
            best_ask_sz = mbp.levels[0].ask_sz
            
            # Simple Vacuum: Is EITHER side fragile?
            is_fragile = (best_bid_sz < vacuum_threshold) or (best_ask_sz < vacuum_threshold)
            is_restored = (best_bid_sz > restore_threshold) and (best_ask_sz > restore_threshold)
            
            if vacuum_start_ns is None:
                if is_fragile:
                    vacuum_start_ns = mbp.ts_event_ns
            else:
                if is_restored:
                    duration_ms = (mbp.ts_event_ns - vacuum_start_ns) / 1e6
                    if duration_ms > max_vacuum_ms:
                        max_vacuum_ms = duration_ms
                    vacuum_start_ns = None
                    
        # Handle case where vacuum persists through end of window
        if vacuum_start_ns is not None:
             duration_ms = (ts - vacuum_start_ns) / 1e6
             if duration_ms > max_vacuum_ms:
                 max_vacuum_ms = duration_ms
                 
        vacuum_durations[i] = max_vacuum_ms
        
        # ─── Replenishment Latency (Placeholder) ───
        # Require trade data stream (not just MBP snapshots) to trigger "consumption"
        # For now, we only latch Vacuum
        replenishment_latencies[i] = 0.0 

    result = signals_df.copy()
    result['vacuum_duration_ms'] = vacuum_durations
    result['replenishment_latency_ms'] = replenishment_latencies
    
    return result


class ComputeMicrostructureStage(BaseStage):
    """Compute microstructure features (Vacuum, Latency)."""
    
    @property
    def name(self) -> str:
        return "compute_microstructure"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        
        signals_df = compute_microstructure_features(
            signals_df=signals_df,
            mbp10_snapshots=mbp10_snapshots
        )
        
        return {'signals_df': signals_df}
