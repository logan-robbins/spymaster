"""
True Event-Based Order Flow Imbalance (OFI) Computation.

Implements OFI per Cont, Kukanov & Stoikov (2014) 
"The Price Impact of Order Book Events".

Key difference from state-delta OFI:
- State-delta: compares consecutive snapshots, infers flow from size changes
- Event-based: uses actual Add/Cancel/Modify events with side (Bid/Ask)

Event-based OFI formula:
  OFI_t = Σ (e_i * s_i * size_i)
  
Where:
  e_i = +1 for Add, -1 for Cancel
  s_i = +1 for Bid side, -1 for Ask side
  size_i = order size

This gives:
  Add on Bid:    +size (buying pressure)
  Cancel on Bid: -size (reduced buying)
  Add on Ask:    -size (selling pressure)
  Cancel on Ask: +size (reduced selling)
  Trade (action='T'): excluded (execution, not order flow)
  Modify: treated as Cancel(-old) + Add(+new), but we only see final size
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import numba

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10
from src.common.config import CONFIG


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
    Modify is treated as 0 (ambiguous without old size)
    Trade is excluded (action != 1,2,3)
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
        # action == 3 (Modify) and action == 0 (Trade/other) contribute 0
    
    return ofi


def compute_multiwindow_ofi(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    windows_seconds: List[float] = None
) -> pd.DataFrame:
    """
    Compute TRUE event-based OFI at multiple lookback windows.
    
    Uses action/side fields from MBP-10 events per Cont et al. (2014).
    
    Spatial Band Ranges (consistent with Tide features):
    - Total OFI: Within ±50pt of level (CONFIG.FUEL_STRIKE_RANGE)
    - Near Level: Within ±25pt of level (CONFIG.TIDE_SPLIT_RANGE)
    - Above: (Level, Level + 25pt]
    - Below: [Level - 25pt, Level)
       
    Args:
        signals_df: DataFrame with signals
        mbp10_snapshots: List of MBP-10 events with action/side fields
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
            result[f'ofi_above_5pt{suffix}'] = 0.0
            result[f'ofi_below_5pt{suffix}'] = 0.0
        return result
    
    # 1. Build TRUE Event-Based OFI Time Series
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda x: x.ts_event_ns)
    timestamps = np.array([x.ts_event_ns for x in sorted_snapshots], dtype=np.int64)
    
    # Map action strings to integers: A=1(Add), C=2(Cancel), M=3(Modify), T/other=0
    action_map = {'A': 1, 'C': 2, 'M': 3}
    actions = np.array([action_map.get(x.action, 0) for x in sorted_snapshots], dtype=np.int32)
    
    # Map side strings to integers: B=+1(Bid), A=-1(Ask), None=0
    side_map = {'B': 1, 'A': -1}
    sides = np.array([side_map.get(x.side, 0) for x in sorted_snapshots], dtype=np.int32)
    
    # Extract action sizes
    sizes = np.array([x.action_size if x.action_size else 0 for x in sorted_snapshots], dtype=np.float64)
    
    # Extract action prices for spatial filtering
    action_prices = np.array([x.action_price if x.action_price else 0.0 for x in sorted_snapshots], dtype=np.float64)
    
    # Compute event-based OFI using Numba
    ofi_values = _compute_event_ofi_numba(actions, sides, sizes)
    
    # 2. Compute Features Grouped by Level
    # ------------------------------------
    n_signals = len(signals_df)
    result = signals_df.copy()
    
    # Initialize implementation columns
    feature_cols = {}
    for w in windows_seconds:
        suffix = f'_{int(w)}s'
        feature_cols[f'ofi{suffix}'] = np.zeros(n_signals, dtype=np.float64)
        feature_cols[f'ofi_near_level{suffix}'] = np.zeros(n_signals, dtype=np.float64)
        feature_cols[f'ofi_above_5pt{suffix}'] = np.zeros(n_signals, dtype=np.float64)
        feature_cols[f'ofi_below_5pt{suffix}'] = np.zeros(n_signals, dtype=np.float64)
    
    signal_ts = signals_df['ts_ns'].values.astype(np.int64)
    signal_levels = signals_df['level_price'].values.astype(np.float64)
    
    # Compute spatially-filtered OFI for each unique level
    unique_levels = np.unique(signal_levels)
    
    for lvl in unique_levels:
        # Find all signals at this level
        sig_mask = signal_levels == lvl
        if not np.any(sig_mask):
            continue
            
        subset_ts = signal_ts[sig_mask]
        subset_indices = np.where(sig_mask)[0] # Indices in original df
        
        # Create Spatial Masks using action_price (where the order was placed)
        band_total = CONFIG.FUEL_STRIKE_RANGE  # 50.0 pt
        band_split = CONFIG.TIDE_SPLIT_RANGE   # 25.0 pt
        
        # Only include A/C events with valid action prices (these contribute to OFI)
        # M (Modify) and T (Trade) events have valid prices but don't contribute
        valid_ofi = (action_prices > 0) & ((actions == 1) | (actions == 2))  # A or C
        
        # Total: Within ±50pt
        mask_total = valid_ofi & (np.abs(action_prices - lvl) <= band_total)
        
        # Near Level: Within ±25pt (for mean calculation)
        mask_near = valid_ofi & (np.abs(action_prices - lvl) <= band_split)
        
        # Above: (Level, Level + 25pt]
        mask_above = valid_ofi & (action_prices > lvl) & (action_prices <= lvl + band_split)
        
        # Below: [Level - 25pt, Level)
        mask_below = valid_ofi & (action_prices < lvl) & (action_prices >= lvl - band_split)
        
        # Cumulative sums for spatial filters
        cum_total = np.cumsum(np.where(mask_total, ofi_values, 0.0))
        cum_near = np.cumsum(np.where(mask_near, ofi_values, 0.0))
        cum_count_near = np.cumsum(mask_near.astype(np.int64))
        cum_above = np.cumsum(np.where(mask_above, ofi_values, 0.0))
        cum_below = np.cumsum(np.where(mask_below, ofi_values, 0.0))
        
        # Compute for all windows
        for w in windows_seconds:
            lookback_ns = int(w * 1e9)
            sub_start_ts = subset_ts - lookback_ns
            
            # Search sorted on timestamps
            idx_start = np.searchsorted(timestamps, sub_start_ts, side='right')
            idx_end = np.searchsorted(timestamps, subset_ts, side='right')
            
            # Helper to get diff
            def get_diff(arr, i_start, i_end):
                v_end = np.zeros(len(i_end))
                m_e = i_end > 0
                v_end[m_e] = arr[i_end[m_e] - 1]
                
                v_start = np.zeros(len(i_start))
                m_s = i_start > 0
                v_start[m_s] = arr[i_start[m_s] - 1]
                return v_end - v_start
            
            suffix = f'_{int(w)}s'
            
            # Total (Sum within ±50pt)
            sum_total = get_diff(cum_total, idx_start, idx_end)
            feature_cols[f'ofi{suffix}'][subset_indices] = sum_total
            
            # Near Level (Mean within ±25pt)
            sum_near = get_diff(cum_near, idx_start, idx_end)
            cnt_near = get_diff(cum_count_near, idx_start, idx_end)
            
            # Avoid divide by zero
            mean_near = np.zeros_like(sum_near)
            valid_stats = cnt_near > 0
            mean_near[valid_stats] = sum_near[valid_stats] / cnt_near[valid_stats]
            
            feature_cols[f'ofi_near_level{suffix}'][subset_indices] = mean_near
            
            # Above/Below (Sum in respective ±25pt bands)
            sum_above = get_diff(cum_above, idx_start, idx_end)
            feature_cols[f'ofi_above_5pt{suffix}'][subset_indices] = sum_above
            
            sum_below = get_diff(cum_below, idx_start, idx_end)
            feature_cols[f'ofi_below_5pt{suffix}'][subset_indices] = sum_below

    # Assign columns to result
    for k, v in feature_cols.items():
        result[k] = v
        
    # Derived: Acceleration
    if 'ofi_30s' in feature_cols and 'ofi_120s' in feature_cols:
        result['ofi_acceleration'] = result['ofi_30s'] / (result['ofi_120s'] + 1e-6)

    return result


class ComputeMultiWindowOFIStage(BaseStage):
    """Compute OFI at multiple lookback windows."""
    
    @property
    def name(self) -> str:
        return "compute_multiwindow_ofi"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'mbp10_snapshots']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        mbp10_snapshots = ctx.data.get('mbp10_snapshots', [])
        
        signals_df = compute_multiwindow_ofi(
            signals_df=signals_df,
            mbp10_snapshots=mbp10_snapshots,
            windows_seconds=[30.0, 60.0, 120.0, 300.0]
        )
        
        return {'signals_df': signals_df}
