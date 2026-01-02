"""
Microstructure features - Vacuum and Replenishment Latency.

Per Microstructure Research (Phase 5):
- Vacuum Duration: Time (ms) with depth < threshold at best bid/ask.
- Replenishment Latency (Bid): Time (ms) for bid-side depth to recover after shock.
  → Measures SUPPORT resilience (liquidity below price)
- Replenishment Latency (Ask): Time (ms) for ask-side depth to recover after shock.
  → Measures RESISTANCE resilience (liquidity above price)

Algorithm based on academic literature (Obizhaeva-Wang model):
- Liquidity shock = ≥30% depth drop in consecutive MBP snapshots
- Recovery = depth returns to 90% of pre-shock baseline
- Typical recovery: 5-10 seconds in liquid markets

Designed for 15s "Goldilocks" cadence.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10
from src.common.config import CONFIG


def _get_side_depth_10(mbp: MBP10, side: str) -> int:
    """Sum depth across all 10 levels for one side (bid or ask)."""
    if not mbp.levels:
        return 0
    total = 0
    for level in mbp.levels:
        if side == 'bid':
            total += level.bid_sz
        else:
            total += level.ask_sz
    return total


def _compute_pre_shock_baseline(
    sorted_snapshots: list,
    mbp_times: np.ndarray,
    trade_ts: int,
    side: str,
    pre_window_ns: int = 1_000_000_000  # 1 second
) -> float:
    """
    Compute average depth on impacted side over pre_window before trade.
    Per academic literature: use 1-5 second window.
    """
    start_ts = trade_ts - pre_window_ns
    start_idx = np.searchsorted(mbp_times, start_ts, side='right')
    end_idx = np.searchsorted(mbp_times, trade_ts, side='left')
    
    if end_idx <= start_idx:
        return 0.0
    
    depths = []
    for i in range(start_idx, end_idx):
        mbp = sorted_snapshots[i]
        depths.append(_get_side_depth_10(mbp, side))
    
    return np.mean(depths) if depths else 0.0


def compute_microstructure_features(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    trades: List = None,
    vacuum_threshold: int = 5,  # Contracts at best level
    restore_threshold: int = 20,  # Contracts at best level for vacuum
    depletion_threshold: float = 0.50,  # 50% depth drop = liquidity shock
    recovery_threshold: float = 0.90,  # 90% recovery = replenished
    max_recovery_horizon_ns: int = 60_000_000_000  # 60 seconds max search
) -> pd.DataFrame:
    """
    Compute microstructure features by scanning high-frequency MBP stream
    and latching max values into signal windows.
    
    Replenishment Latency Algorithm (per academic literature):
    1. Detect liquidity shocks from consecutive MBP snapshots (≥30% depth drop)
    2. Track BID and ASK sides separately:
       - BID replenishment = support resilience (liquidity below price)
       - ASK replenishment = resistance resilience (liquidity above price)
    3. Measure time until depth recovers to 90% of pre-shock baseline
    4. Use sum of all 10 levels (not just best bid/ask)
    
    Args:
        signals_df: DataFrame with signals (15s cadence)
        mbp10_snapshots: List of raw MBP-10 snapshots
        trades: List of trades (unused, kept for API compatibility)
        vacuum_threshold: low liquidity threshold at best level (contracts)
        restore_threshold: vacuum replenishment threshold at best level (contracts)
        depletion_threshold: % depth drop to qualify as liquidity shock (0.30 = 30%)
        recovery_threshold: % of baseline to qualify as recovered (0.90 = 90%)
        max_recovery_horizon_ns: max time to search for recovery (60s)
    
    Returns:
        DataFrame with added columns:
        - vacuum_duration_ms: max vacuum duration in 15s window
        - replenishment_latency_bid_ms: max bid-side recovery time (support)
        - replenishment_latency_ask_ms: max ask-side recovery time (resistance)
    """
    if signals_df.empty or not mbp10_snapshots:
        result = signals_df.copy()
        result['vacuum_duration_ms'] = 0.0
        result['replenishment_latency_bid_ms'] = 0.0  # Support resilience (below price)
        result['replenishment_latency_ask_ms'] = 0.0  # Resistance resilience (above price)
        return result
    
    # Sort snapshots by time
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda mbp: mbp.ts_event_ns)
    mbp_times = np.array([mbp.ts_event_ns for mbp in sorted_snapshots], dtype=np.int64)
    
    # Pre-index trades if available
    trade_times = None
    sorted_trades = None
    if trades is not None and len(trades) > 0:
        sorted_trades = sorted(trades, key=lambda t: t.ts_event_ns)
        trade_times = np.array([t.ts_event_ns for t in sorted_trades], dtype=np.int64)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    
    # Output arrays
    vacuum_durations = np.zeros(n, dtype=np.float64)
    replenishment_latencies_bid = np.zeros(n, dtype=np.float64)  # Support resilience
    replenishment_latencies_ask = np.zeros(n, dtype=np.float64)  # Resistance resilience
    
    # Scan logic: 
    # For each signal row at T, scan window [T-15s, T]
    # LATCH the MAX duration/latency found in that window.
    
    window_ns = 15 * 1_000_000_000  # 15 seconds in ns
    
    for i in range(n):
        ts = signal_ts[i]
        start_ts = ts - window_ns
        
        # Find MBP events in this 15s window
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
        
        # ─── Replenishment Latency (MBP-based Algorithm, Split by Side) ───
        # Detect liquidity shocks directly from consecutive MBP depth drops
        # Split by BID (support/below) and ASK (resistance/above)
        # Per academic literature: shock = ≥30% depth drop, recovery = 90% of pre-shock
        max_replen_bid_ms = 0.0  # Support resilience (bids = below price)
        max_replen_ask_ms = 0.0  # Resistance resilience (asks = above price)
        
        # Need at least 2 snapshots in window to detect changes
        if len(window_snapshots) >= 2:
            for k in range(1, len(window_snapshots)):
                prev_mbp = window_snapshots[k - 1]
                curr_mbp = window_snapshots[k]
                
                if not prev_mbp.levels or not curr_mbp.levels:
                    continue
                
                # Check for liquidity shock on EACH side separately
                for side in ['bid', 'ask']:
                    prev_depth = _get_side_depth_10(prev_mbp, side)
                    curr_depth = _get_side_depth_10(curr_mbp, side)
                    
                    if prev_depth < 20:  # Skip if already thin (avoid noise)
                        continue
                    
                    # Is this a liquidity shock? (depth dropped by ≥30%)
                    depletion = 1 - (curr_depth / prev_depth)
                    if depletion < 0.30:  # Threshold tuned for MBP granularity
                        continue  # Not a shock
                    
                    shock_ts = curr_mbp.ts_event_ns
                    recovery_target = prev_depth * recovery_threshold  # 90% of pre-shock
                    recovery_limit = shock_ts + max_recovery_horizon_ns
                    
                    # Find this MBP's global index to scan forward
                    shock_idx = start_idx + k
                    
                    # Scan forward for recovery
                    replen_ts = None
                    for j in range(shock_idx + 1, len(sorted_snapshots)):
                        future_mbp = sorted_snapshots[j]
                        if future_mbp.ts_event_ns > recovery_limit:
                            break  # Exceeded max horizon
                        
                        future_depth = _get_side_depth_10(future_mbp, side)
                        if future_depth >= recovery_target:
                            replen_ts = future_mbp.ts_event_ns
                            break
                    
                    # Calculate latency and track per-side
                    if replen_ts is not None:
                        latency_ms = (replen_ts - shock_ts) / 1e6
                        if side == 'bid':
                            if latency_ms > max_replen_bid_ms:
                                max_replen_bid_ms = latency_ms
                        else:  # ask
                            if latency_ms > max_replen_ask_ms:
                                max_replen_ask_ms = latency_ms
        
        replenishment_latencies_bid[i] = max_replen_bid_ms
        replenishment_latencies_ask[i] = max_replen_ask_ms 

    result = signals_df.copy()
    result['vacuum_duration_ms'] = vacuum_durations
    result['replenishment_latency_bid_ms'] = replenishment_latencies_bid  # Support resilience
    result['replenishment_latency_ask_ms'] = replenishment_latencies_ask  # Resistance resilience
    
    return result


class ComputeMicrostructureStage(BaseStage):
    """Compute microstructure features (Vacuum, Latency)."""
    
    @property
    def name(self) -> str:
        return "compute_microstructure"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots', 'trades']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        trades = ctx.data.get('trades', None)
        
        signals_df = compute_microstructure_features(
            signals_df=signals_df,
            mbp10_snapshots=mbp10_snapshots,
            trades=trades
        )
        
        return {'signals_df': signals_df}
