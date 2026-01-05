"""
Shared microstructure computation functions.

Computes spread, depth, imbalance from MBP-10 data.
Can be level-relative or market-wide.
"""

import numpy as np
from typing import List, Dict, Optional
from src.common.event_types import MBP10


def compute_market_microstructure(
    signal_ts: np.ndarray,
    mbp10_snapshots: List[MBP10],
    level_price: Optional[float] = None,
    band_pt: float = 2.5,
) -> Dict[str, np.ndarray]:
    """
    Compute microstructure features at signal timestamps.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        mbp10_snapshots: List of MBP-10 snapshots
        level_price: If set, compute depth within band of level.
                    If None, compute total market depth.
        band_pt: Band width for level-relative depth (default 2.5pt)
    
    Returns:
        Dictionary of microstructure features
    """
    n_signals = len(signal_ts)
    result = {
        'spread': np.zeros(n_signals, dtype=np.float64),
        'spread_pct': np.zeros(n_signals, dtype=np.float64),
        'bid_depth': np.zeros(n_signals, dtype=np.float64),
        'ask_depth': np.zeros(n_signals, dtype=np.float64),
        'depth_imbalance': np.zeros(n_signals, dtype=np.float64),
        'mid_price': np.zeros(n_signals, dtype=np.float64),
    }
    
    if n_signals == 0 or not mbp10_snapshots:
        return result
    
    # Sort snapshots by timestamp
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda x: x.ts_event_ns)
    mbp_ts = np.array([x.ts_event_ns for x in sorted_snapshots], dtype=np.int64)
    
    # Find nearest snapshot for each signal
    idx_lookup = np.searchsorted(mbp_ts, signal_ts, side='right') - 1
    idx_lookup = np.clip(idx_lookup, 0, len(mbp_ts) - 1)
    
    for i, idx in enumerate(idx_lookup):
        mbp = sorted_snapshots[idx]
        
        if not mbp.levels:
            continue
        
        best = mbp.levels[0]
        best_bid = best.bid_px
        best_ask = best.ask_px
        
        if best_bid <= 0 or best_ask <= 0:
            continue
        
        # Spread
        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2
        result['spread'][i] = spread
        result['spread_pct'][i] = spread / mid * 100 if mid > 0 else 0
        result['mid_price'][i] = mid
        
        # Depth
        if level_price is not None:
            # Level-relative: only count depth within band
            zone_low = level_price - band_pt
            zone_high = level_price + band_pt
            
            bid_depth = 0.0
            ask_depth = 0.0
            for level in mbp.levels:
                if zone_low <= level.bid_px <= zone_high:
                    bid_depth += level.bid_sz
                if zone_low <= level.ask_px <= zone_high:
                    ask_depth += level.ask_sz
        else:
            # Global: sum all 10 levels
            bid_depth = sum(level.bid_sz for level in mbp.levels)
            ask_depth = sum(level.ask_sz for level in mbp.levels)
        
        result['bid_depth'][i] = bid_depth
        result['ask_depth'][i] = ask_depth
        
        # Imbalance: (bid - ask) / (bid + ask)
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            result['depth_imbalance'][i] = (bid_depth - ask_depth) / total_depth
    
    return result


def compute_depth_evolution(
    signal_ts: np.ndarray,
    mbp10_snapshots: List[MBP10],
    level_price: float,
    lookback_windows: List[float] = None,
    band_pt: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Compute depth evolution (delta, pct change) over lookback windows.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        mbp10_snapshots: List of MBP-10 snapshots
        level_price: Level price for depth calculation
        lookback_windows: List of lookback windows in minutes
        band_pt: Band width for depth calculation
    
    Returns:
        Dictionary of evolution features
    """
    if lookback_windows is None:
        lookback_windows = [1.0, 2.0, 3.0, 5.0]
    
    n_signals = len(signal_ts)
    result = {}
    
    for w_min in lookback_windows:
        suffix = f'_{int(w_min)}min'
        result[f'barrier_delta{suffix}'] = np.zeros(n_signals, dtype=np.float64)
        result[f'barrier_pct_change{suffix}'] = np.zeros(n_signals, dtype=np.float64)
    
    if n_signals == 0 or not mbp10_snapshots:
        return result
    
    # Build depth time series
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda x: x.ts_event_ns)
    mbp_ts = np.array([x.ts_event_ns for x in sorted_snapshots], dtype=np.int64)
    
    zone_low = level_price - band_pt
    zone_high = level_price + band_pt
    
    depths = np.zeros(len(sorted_snapshots), dtype=np.float64)
    for j, mbp in enumerate(sorted_snapshots):
        total = 0.0
        for level in mbp.levels:
            if zone_low <= level.bid_px <= zone_high:
                total += level.bid_sz
            if zone_low <= level.ask_px <= zone_high:
                total += level.ask_sz
        depths[j] = total
    
    cum_depths = np.cumsum(depths)  # For averaging
    
    # Compute for each signal and window
    idx_now = np.searchsorted(mbp_ts, signal_ts, side='right') - 1
    idx_now = np.clip(idx_now, 0, len(mbp_ts) - 1)
    
    for w_min in lookback_windows:
        lookback_ns = int(w_min * 60 * 1e9)
        past_ts = signal_ts - lookback_ns
        idx_past = np.searchsorted(mbp_ts, past_ts, side='right') - 1
        idx_past = np.clip(idx_past, 0, len(mbp_ts) - 1)
        
        suffix = f'_{int(w_min)}min'
        
        depth_now = depths[idx_now]
        depth_past = depths[idx_past]
        
        delta = depth_now - depth_past
        pct_change = np.zeros_like(delta)
        valid = depth_past > 0
        pct_change[valid] = delta[valid] / depth_past[valid]
        
        result[f'barrier_delta{suffix}'] = delta
        result[f'barrier_pct_change{suffix}'] = pct_change
    
    return result

