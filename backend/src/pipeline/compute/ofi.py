"""
Shared OFI (Order Flow Imbalance) computation functions.

Implements true event-based OFI per Cont, Kukanov & Stoikov (2014).
Can be used with or without spatial filtering relative to a level.
"""

import numpy as np
import numba
from typing import List, Dict, Optional, Tuple
from src.common.event_types import MBP10


@numba.jit(nopython=True, cache=True)
def _compute_event_ofi_numba(
    actions: np.ndarray,   # 0=other, 1=Add, 2=Cancel, 3=Modify
    sides: np.ndarray,     # 0=None, 1=Bid, -1=Ask
    sizes: np.ndarray,     # action_size
) -> np.ndarray:
    """
    Compute event-based OFI values using Numba JIT.
    
    OFI = side * event_sign * size
    Where event_sign is +1 for Add, -1 for Cancel
    """
    n = len(actions)
    ofi = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        action = actions[i]
        side = sides[i]
        size = sizes[i]
        
        if side == 0:  # No side info
            continue
            
        if action == 1:  # Add
            ofi[i] = side * size
        elif action == 2:  # Cancel
            ofi[i] = -side * size
        # Modify and Trade contribute 0
    
    return ofi


def compute_event_ofi(mbp10_snapshots: List[MBP10]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute raw event-based OFI from MBP-10 snapshots.
    
    Returns:
        timestamps: Array of event timestamps (ns)
        ofi_values: Array of OFI values per event
        action_prices: Array of action prices for spatial filtering
    """
    if not mbp10_snapshots:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda x: x.ts_event_ns)
    
    timestamps = np.array([x.ts_event_ns for x in sorted_snapshots], dtype=np.int64)
    
    # Map actions: A=1(Add), C=2(Cancel), M=3(Modify), T/other=0
    action_map = {'A': 1, 'C': 2, 'M': 3}
    actions = np.array([action_map.get(x.action, 0) for x in sorted_snapshots], dtype=np.int32)
    
    # Map sides: B=+1(Bid), A=-1(Ask), None=0
    side_map = {'B': 1, 'A': -1}
    sides = np.array([side_map.get(x.side, 0) for x in sorted_snapshots], dtype=np.int32)
    
    sizes = np.array([x.action_size if x.action_size else 0 for x in sorted_snapshots], dtype=np.float64)
    action_prices = np.array([x.action_price if x.action_price else 0.0 for x in sorted_snapshots], dtype=np.float64)
    
    ofi_values = _compute_event_ofi_numba(actions, sides, sizes)
    
    return timestamps, ofi_values, action_prices


def compute_ofi_windows(
    signal_ts: np.ndarray,
    ofi_timestamps: np.ndarray,
    ofi_values: np.ndarray,
    action_prices: np.ndarray,
    windows_seconds: List[float],
    level_price: Optional[float] = None,
    band_total: float = 50.0,
    band_split: float = 25.0,
) -> Dict[str, np.ndarray]:
    """
    Compute OFI at multiple lookback windows.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        ofi_timestamps: Array of OFI event timestamps (ns)
        ofi_values: Array of raw OFI values
        action_prices: Array of action prices for spatial filtering
        windows_seconds: List of lookback windows
        level_price: If set, compute spatially-filtered OFI relative to level.
                    If None, compute total OFI (global mode).
        band_total: Total band for OFI sum (default 50pt)
        band_split: Split band for near/above/below (default 25pt)
    
    Returns:
        Dictionary of feature arrays
    """
    n_signals = len(signal_ts)
    result = {}
    
    # Initialize output arrays
    for w in windows_seconds:
        suffix = f'_{int(w)}s'
        result[f'ofi{suffix}'] = np.zeros(n_signals, dtype=np.float64)
        if level_price is not None:
            result[f'ofi_near_level{suffix}'] = np.zeros(n_signals, dtype=np.float64)
            result[f'ofi_above{suffix}'] = np.zeros(n_signals, dtype=np.float64)
            result[f'ofi_below{suffix}'] = np.zeros(n_signals, dtype=np.float64)
    
    if len(ofi_timestamps) == 0:
        return result
    
    # Create spatial masks
    if level_price is not None:
        # Level-relative mode: filter by distance to level
        valid_ofi = (action_prices > 0) & (ofi_values != 0)
        mask_total = valid_ofi & (np.abs(action_prices - level_price) <= band_total)
        mask_near = valid_ofi & (np.abs(action_prices - level_price) <= band_split)
        mask_above = valid_ofi & (action_prices > level_price) & (action_prices <= level_price + band_split)
        mask_below = valid_ofi & (action_prices < level_price) & (action_prices >= level_price - band_split)
        
        cum_total = np.cumsum(np.where(mask_total, ofi_values, 0.0))
        cum_near = np.cumsum(np.where(mask_near, ofi_values, 0.0))
        cum_count_near = np.cumsum(mask_near.astype(np.int64))
        cum_above = np.cumsum(np.where(mask_above, ofi_values, 0.0))
        cum_below = np.cumsum(np.where(mask_below, ofi_values, 0.0))
    else:
        # Global mode: no spatial filtering, just total OFI
        cum_total = np.cumsum(ofi_values)
    
    # Compute for all windows
    for w in windows_seconds:
        lookback_ns = int(w * 1e9)
        start_ts = signal_ts - lookback_ns
        
        idx_start = np.searchsorted(ofi_timestamps, start_ts, side='right')
        idx_end = np.searchsorted(ofi_timestamps, signal_ts, side='right')
        
        def get_diff(arr, i_start, i_end):
            v_end = np.zeros(len(i_end))
            m_e = i_end > 0
            v_end[m_e] = arr[i_end[m_e] - 1]
            
            v_start = np.zeros(len(i_start))
            m_s = i_start > 0
            v_start[m_s] = arr[i_start[m_s] - 1]
            return v_end - v_start
        
        suffix = f'_{int(w)}s'
        result[f'ofi{suffix}'] = get_diff(cum_total, idx_start, idx_end)
        
        if level_price is not None:
            sum_near = get_diff(cum_near, idx_start, idx_end)
            cnt_near = get_diff(cum_count_near, idx_start, idx_end)
            mean_near = np.zeros_like(sum_near)
            valid = cnt_near > 0
            mean_near[valid] = sum_near[valid] / cnt_near[valid]
            result[f'ofi_near_level{suffix}'] = mean_near
            result[f'ofi_above{suffix}'] = get_diff(cum_above, idx_start, idx_end)
            result[f'ofi_below{suffix}'] = get_diff(cum_below, idx_start, idx_end)
    
    return result

