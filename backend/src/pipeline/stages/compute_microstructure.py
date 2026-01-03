"""
Level-Relative Microstructure Features - Vacuum and Replenishment Latency.

Computes liquidity metrics RELATIVE TO THE TARGET LEVEL, not global market depth.

For a level at price L:
- Vacuum: Time with thin liquidity at bid/ask prices near L
- Replenishment Latency Bid: Recovery time for bids defending L from below (support)
- Replenishment Latency Ask: Recovery time for asks defending L from above (resistance)

Algorithm based on Obizhaeva-Wang model:
- Liquidity shock = ≥50% depth drop in consecutive MBP snapshots
- Recovery = depth returns to 90% of pre-shock baseline
- Spatial filter: Only count depth within ±2.5pt of level (10 ES ticks)
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10


def _get_depth_near_level(mbp: MBP10, level_price: float, side: str, band: float = 2.5) -> int:
    """
    Get total depth on one side within ±band of the level price.
    
    Args:
        mbp: MBP-10 snapshot
        level_price: Target level price
        side: 'bid' or 'ask'
        band: Price band around level (default ±2.5pt = 10 ES ticks)
    
    Returns:
        Total contracts within the spatial band
    """
    if not mbp.levels:
        return 0
    
    total = 0
    for level in mbp.levels:
        if side == 'bid':
            price = level.bid_px
            size = level.bid_sz
        else:
            price = level.ask_px
            size = level.ask_sz
        
        # Only count depth within band of level
        if abs(price - level_price) <= band:
            total += size
    
    return total


def _get_best_depth_near_level(mbp: MBP10, level_price: float, band: float = 2.5) -> tuple:
    """
    Get best bid/ask depth if their prices are within band of level.
    
    Returns:
        (bid_sz, ask_sz) - 0 if price is outside band
    """
    if not mbp.levels:
        return 0, 0
    
    l0 = mbp.levels[0]
    
    # Only count if best bid/ask is near the level
    bid_sz = l0.bid_sz if abs(l0.bid_px - level_price) <= band else 0
    ask_sz = l0.ask_sz if abs(l0.ask_px - level_price) <= band else 0
    
    return bid_sz, ask_sz


def compute_microstructure_features(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    vacuum_threshold: int = 5,  # Contracts at level
    restore_threshold: int = 20,  # Contracts at level for vacuum recovery
    depletion_threshold: float = 0.50,  # 50% depth drop = liquidity shock
    recovery_threshold: float = 0.90,  # 90% recovery = replenished
    max_recovery_horizon_ns: int = 60_000_000_000,  # 60 seconds max search
    level_band: float = 2.5  # ±2.5pt around level (10 ES ticks)
) -> pd.DataFrame:
    """
    Compute level-relative microstructure features.
    
    Features measure liquidity AT THE TARGET LEVEL, not global market depth.
    
    Args:
        signals_df: DataFrame with signals (must have 'level_price' column)
        mbp10_snapshots: List of raw MBP-10 snapshots
        vacuum_threshold: low liquidity threshold (contracts near level)
        restore_threshold: vacuum recovery threshold (contracts near level)
        depletion_threshold: % depth drop to qualify as liquidity shock
        recovery_threshold: % of baseline to qualify as recovered
        max_recovery_horizon_ns: max time to search for recovery
        level_band: price band around level for spatial filtering (±2.5pt default)
    
    Returns:
        DataFrame with added columns:
        - vacuum_duration_ms: max vacuum duration near level in 15s window
        - replenishment_latency_bid_ms: max bid-side recovery time (support at level)
        - replenishment_latency_ask_ms: max ask-side recovery time (resistance at level)
    """
    if signals_df.empty or not mbp10_snapshots:
        result = signals_df.copy()
        result['vacuum_duration_ms'] = 0.0
        result['replenishment_latency_bid_ms'] = 0.0
        result['replenishment_latency_ask_ms'] = 0.0
        return result
    
    # Sort snapshots by time
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda mbp: mbp.ts_event_ns)
    mbp_times = np.array([mbp.ts_event_ns for mbp in sorted_snapshots], dtype=np.int64)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    level_prices = signals_df['level_price'].values
    
    # Output arrays
    vacuum_durations = np.zeros(n, dtype=np.float64)
    replenishment_latencies_bid = np.zeros(n, dtype=np.float64)
    replenishment_latencies_ask = np.zeros(n, dtype=np.float64)
    
    window_ns = 15 * 1_000_000_000  # 15 seconds
    
    for i in range(n):
        ts = signal_ts[i]
        level_price = level_prices[i]
        start_ts = ts - window_ns
        
        # Find MBP events in this 15s window
        start_idx = np.searchsorted(mbp_times, start_ts, side='right')
        end_idx = np.searchsorted(mbp_times, ts, side='right')
        
        if end_idx <= start_idx:
            continue
            
        window_snapshots = sorted_snapshots[start_idx:end_idx]
        
        # ─── Vacuum Detection (Level-Relative) ───
        max_vacuum_ms = 0.0
        vacuum_start_ns = None
        
        for mbp in window_snapshots:
            bid_sz, ask_sz = _get_best_depth_near_level(mbp, level_price, level_band)
            
            # Vacuum: thin liquidity on EITHER side near the level
            is_fragile = (bid_sz < vacuum_threshold) or (ask_sz < vacuum_threshold)
            is_restored = (bid_sz > restore_threshold) and (ask_sz > restore_threshold)
            
            if vacuum_start_ns is None:
                if is_fragile:
                    vacuum_start_ns = mbp.ts_event_ns
            else:
                if is_restored:
                    duration_ms = (mbp.ts_event_ns - vacuum_start_ns) / 1e6
                    max_vacuum_ms = max(max_vacuum_ms, duration_ms)
                    vacuum_start_ns = None
                    
        # Handle vacuum persisting through end of window
        if vacuum_start_ns is not None:
            duration_ms = (ts - vacuum_start_ns) / 1e6
            max_vacuum_ms = max(max_vacuum_ms, duration_ms)
                 
        vacuum_durations[i] = max_vacuum_ms
        
        # ─── Replenishment Latency (Level-Relative) ───
        max_replen_bid_ms = 0.0
        max_replen_ask_ms = 0.0
        
        if len(window_snapshots) >= 2:
            for k in range(1, len(window_snapshots)):
                prev_mbp = window_snapshots[k - 1]
                curr_mbp = window_snapshots[k]
                
                for side in ['bid', 'ask']:
                    prev_depth = _get_depth_near_level(prev_mbp, level_price, side, level_band)
                    curr_depth = _get_depth_near_level(curr_mbp, level_price, side, level_band)
                    
                    if prev_depth < 10:  # Skip if already thin
                        continue
                    
                    # Liquidity shock?
                    depletion = 1 - (curr_depth / prev_depth)
                    if depletion < depletion_threshold:
                        continue
                    
                    shock_ts = curr_mbp.ts_event_ns
                    recovery_target = prev_depth * recovery_threshold
                    recovery_limit = shock_ts + max_recovery_horizon_ns
                    
                    # Scan forward for recovery
                    shock_idx = start_idx + k
                    replen_ts = None
                    
                    for j in range(shock_idx + 1, len(sorted_snapshots)):
                        future_mbp = sorted_snapshots[j]
                        if future_mbp.ts_event_ns > recovery_limit:
                            break
                        
                        future_depth = _get_depth_near_level(future_mbp, level_price, side, level_band)
                        if future_depth >= recovery_target:
                            replen_ts = future_mbp.ts_event_ns
                            break
                    
                    if replen_ts is not None:
                        latency_ms = (replen_ts - shock_ts) / 1e6
                        if side == 'bid':
                            max_replen_bid_ms = max(max_replen_bid_ms, latency_ms)
                        else:
                            max_replen_ask_ms = max(max_replen_ask_ms, latency_ms)
        
        replenishment_latencies_bid[i] = max_replen_bid_ms
        replenishment_latencies_ask[i] = max_replen_ask_ms 

    result = signals_df.copy()
    result['vacuum_duration_ms'] = vacuum_durations
    result['replenishment_latency_bid_ms'] = replenishment_latencies_bid
    result['replenishment_latency_ask_ms'] = replenishment_latencies_ask
    
    return result


class ComputeMicrostructureStage(BaseStage):
    """Compute level-relative microstructure features (Vacuum, Replenishment Latency)."""
    
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
