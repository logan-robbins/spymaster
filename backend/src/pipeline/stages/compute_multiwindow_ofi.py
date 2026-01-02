import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import MBP10
from src.common.config import CONFIG


def compute_multiwindow_ofi(
    signals_df: pd.DataFrame,
    mbp10_snapshots: List[MBP10],
    windows_seconds: List[float] = None
) -> pd.DataFrame:
    """
    Compute integrated OFI at multiple lookback windows using Vectorized Prefix Sums.
    
    Optimization Strategy:
    1. Global Features (Total OFI): Calculated via global cumulative sum + searchsorted.
       Complexity: O(N_signals * log(N_ticks))
    
    2. Spatial Features (Near/Above/Below Level): Grouped by Level Price.
       Instead of filtering ticks for every signal (N_signals * Window_Size),
       we filter ticks ONCE per unique level (N_levels * N_ticks) and use cumsum.
       Complexity: O(N_levels * N_ticks + N_signals)
       
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
            result[f'ofi_above_5pt{suffix}'] = 0.0
            result[f'ofi_below_5pt{suffix}'] = 0.0
        return result
    
    # 1. Build OFI Time Series (Global)
    # ---------------------------------
    # Sort snapshots by time
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda x: x.ts_event_ns)
    
    # Extract arrays for vectorization
    n_snaps = len(sorted_snapshots)
    timestamps = np.array([x.ts_event_ns for x in sorted_snapshots], dtype=np.int64)
    
    # Extract top-of-book for OFI calc
    # Structure: [bid_px, bid_sz, ask_px, ask_sz]
    tob_data = np.zeros((n_snaps, 4), dtype=np.float64)
    for i, snap in enumerate(sorted_snapshots):
        if snap.levels:
            l = snap.levels[0]
            tob_data[i] = [l.bid_px, l.bid_sz, l.ask_px, l.ask_sz]
            
    # Vectorized OFI Calc (Shift and Compare)
    # curr: [1:]
    # prev: [:-1]
    b_t = tob_data[1:, 0]
    q_b_t = tob_data[1:, 1]
    a_t = tob_data[1:, 2]
    q_a_t = tob_data[1:, 3]
    
    b_prev = tob_data[:-1, 0]
    q_b_prev = tob_data[:-1, 1]
    a_prev = tob_data[:-1, 2]
    q_a_prev = tob_data[:-1, 3]
    
    # Bid Flow
    ofi_bid = np.zeros(n_snaps - 1, dtype=np.float64)
    # b_t > b_prev
    mask_bg = b_t > b_prev
    ofi_bid[mask_bg] = q_b_t[mask_bg]
    # b_t < b_prev
    mask_bl = b_t < b_prev
    ofi_bid[mask_bl] = -q_b_prev[mask_bl]
    # b_t == b_prev
    mask_beq = b_t == b_prev
    ofi_bid[mask_beq] = q_b_t[mask_beq] - q_b_prev[mask_beq]
    
    # Ask Flow
    ofi_ask = np.zeros(n_snaps - 1, dtype=np.float64)
    # a_t < a_prev
    mask_al = a_t < a_prev
    ofi_ask[mask_al] = q_a_t[mask_al]
    # a_t > a_prev
    mask_ag = a_t > a_prev
    ofi_ask[mask_ag] = -q_a_prev[mask_ag]
    # a_t == a_prev
    mask_aeq = a_t == a_prev
    ofi_ask[mask_aeq] = q_a_t[mask_aeq] - q_a_prev[mask_aeq]
    
    # Net OFI
    # Adjust length to match original (pad first element)
    raw_ofi = ofi_bid - ofi_ask
    ofi_values = np.insert(raw_ofi, 0, 0.0)
    
    # Mid Prices for spatial filtering
    mid_prices = (tob_data[:, 0] + tob_data[:, 2]) / 2.0
    
    # 2. Pre-compute Global Cumulative Sum
    # ------------------------------------
    # cumsum[i] = sum(0..i)
    # sum(start..end) = cumsum[end] - cumsum[start-1]
    # We use searchsorted 'right' which aligns with end index, so typically:
    # sum(t_start < t <= t_end) -> cumsum[idx_end] - cumsum[idx_start]
    
    global_cumsum = np.cumsum(ofi_values)
    
    # 3. Compute Features Grouped by Level
    # ------------------------------------
    # We'll build the result columns incrementally
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
    
    # A. Global Features (Total OFI) - Can be done for all signals at once
    # --------------------------------------------------------------------
    for w in windows_seconds:
        lookback_ns = int(w * 1e9)
        start_ts = signal_ts - lookback_ns
        
        # Find indices
        # timestamps is sorted
        idx_start = np.searchsorted(timestamps, start_ts, side='right')
        idx_end = np.searchsorted(timestamps, signal_ts, side='right')
        
        # Safe indexing (handle 0)
        # cumsum has same length as timestamps
        # val = global_cumsum[idx_end-1] - global_cumsum[idx_start-1]
        # Wait, simple differential:
        val_end = np.zeros(n_signals)
        mask_e = idx_end > 0
        val_end[mask_e] = global_cumsum[idx_end[mask_e] - 1]
        
        val_start = np.zeros(n_signals)
        mask_s = idx_start > 0
        val_start[mask_s] = global_cumsum[idx_start[mask_s] - 1]
        
        feature_cols[f'ofi_{int(w)}s'] = val_end - val_start

    # B. Spatial Features (Near/Above/Below) - Group by Level
    # -------------------------------------------------------
    unique_levels = np.unique(signal_levels)
    
    for lvl in unique_levels:
        # Find all signals at this level
        sig_mask = signal_levels == lvl
        if not np.any(sig_mask):
            continue
            
        subset_ts = signal_ts[sig_mask]
        subset_indices = np.where(sig_mask)[0] # Indices in original df
        
        # 1. Create Spatial Masks for this Level (Once per Level)
        # -----------------------------------------------------
        band = CONFIG.MONITOR_BAND
        band_5pt = 5.0
        
        # Near Level: |Mid - Level| <= Band
        # Note: mid_prices may have NaNs, use fill or safe comparison
        # isfinite check done via boolean indexing usually, here use np.where
        valid_px = np.isfinite(mid_prices)
        
        mask_near = np.zeros_like(mid_prices, dtype=bool)
        mask_near[valid_px] = np.abs(mid_prices[valid_px] - lvl) <= band
        
        mask_above = np.zeros_like(mid_prices, dtype=bool)
        mask_above[valid_px] = (mid_prices[valid_px] > lvl) & (mid_prices[valid_px] <= lvl + band_5pt)
        
        mask_below = np.zeros_like(mid_prices, dtype=bool)
        mask_below[valid_px] = (mid_prices[valid_px] < lvl) & (mid_prices[valid_px] >= lvl - band_5pt)
        
        # 2. Create Cumulative Sums for Spatial Filters
        # ---------------------------------------------
        # Values where mask is False become 0, contributing nothing to sum
        cum_near = np.cumsum(np.where(mask_near, ofi_values, 0.0))
        # For 'near level' we usually want AVERAGE flow? Or Sum?
        # Original code: "np.mean(window_flows[near_mask])"
        # Prefix sum calculates SUM. For MEAN, we need Count of valid ticks too.
        # Compute Count Cumsum
        cum_count_near = np.cumsum(mask_near.astype(int))
        
        cum_above = np.cumsum(np.where(mask_above, ofi_values, 0.0))
        cum_below = np.cumsum(np.where(mask_below, ofi_values, 0.0))
        
        # 3. Compute for all Windows for this Subset
        # ------------------------------------------
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
            
            # Near Level (Mean)
            sum_near = get_diff(cum_near, idx_start, idx_end)
            cnt_near = get_diff(cum_count_near, idx_start, idx_end)
            
            # Avoid divide by zero
            mean_near = np.zeros_like(sum_near)
            valid_stats = cnt_near > 0
            mean_near[valid_stats] = sum_near[valid_stats] / cnt_near[valid_stats]
            
            feature_cols[f'ofi_near_level{suffix}'][subset_indices] = mean_near
            
            # Above/Below (Sum)
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
