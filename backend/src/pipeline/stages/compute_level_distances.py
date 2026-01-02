import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_all_level_distances(
    signals_df: pd.DataFrame,
    dynamic_levels: Dict[str, pd.Series],
    ohlcv_df: pd.DataFrame,
    atr: pd.Series = None
) -> pd.DataFrame:
    """
    Compute signed distances to all v1 structural levels using Vectorized NumPy ops.
    
    Optimization:
    - Replaces iloc-in-loop with direct numpy array indexing: level_vals[bar_idxs]
    - Replaces nested loop for stacking with broadcasting.
    - Robustness: Computes bar_idx from timestamps if missing (critical for tick-based detection).
    
    Args:
        signals_df: DataFrame with signals (must have ts_ns)
        dynamic_levels: Dict of level series (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90, EMA_20)
        ohlcv_df: OHLCV DataFrame for alignment (must have ts_ns)
        atr: Optional ATR series for normalized distances
    
    Returns:
        DataFrame with distance features added
    """
    if signals_df.empty:
        return signals_df
    
    # v1 level types
    level_types = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']
    
    n = len(signals_df)
    
    # 1. Resolve Bar Indices (critical for looking up time-varying levels)
    # --------------------------------------------------------------------
    if 'bar_idx' in signals_df.columns and not signals_df['bar_idx'].isnull().all():
         bar_indices = signals_df['bar_idx'].fillna(-1).astype(int).values
    else:
        # Map signals to OHLCV bars via timestamp
        if ohlcv_df.empty:
             # Can't map, return empty features
             return signals_df
             
        # Extract timestamps
        if 'ts_ns' in ohlcv_df.columns:
            ohlcv_ts = ohlcv_df['ts_ns'].values.astype(np.int64)
        elif 'timestamp' in ohlcv_df.columns:
            ohlcv_ts = ohlcv_df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
        else:
            ohlcv_ts = ohlcv_df.index.values.astype('datetime64[ns]').astype(np.int64)
            
        sig_ts = signals_df['ts_ns'].values.astype(np.int64)
        
        # searchsorted 'right' - 1 gives the bar containing the timestamp (or nearest past bar)
        # Assumes ohlcv_ts is start of bar or end?
        # Standard: Timestamp is start of bar.
        # If signal is at 10:00:30, and bar is 10:00:00, index is that bar.
        # searchsorted(t, side='right') gives index where t > bar_ts.
        # index-1 gives the bar where bar_ts <= t. Correct.
        bar_indices = np.searchsorted(ohlcv_ts, sig_ts, side='right') - 1
        
        # Clip
        bar_indices = np.clip(bar_indices, 0, len(ohlcv_ts) - 1)
        
        # Store for future use?
        # signals_df['bar_idx'] = bar_indices
    
    level_prices = signals_df['level_price'].values.astype(np.float64)
    
    # Prefer actual spot/entry price when available
    if 'spot' in signals_df.columns:
        spot_prices = signals_df['spot'].astype(np.float64).to_numpy()
    elif 'entry_price' in signals_df.columns:
        spot_prices = signals_df['entry_price'].astype(np.float64).to_numpy()
    else:
        # Fallback
        spot_prices = level_prices 
    
    result = signals_df.copy()
    
    # Prepare ATR array if available
    atr_arr = None
    if atr is not None and not atr.empty:
        # Align ATR to signals
        # Handle out of bounds
        max_idx = len(atr) - 1
        valid_idx_mask = (bar_indices >= 0) & (bar_indices <= max_idx)
        
        atr_arr = np.full(n, np.nan, dtype=np.float64)
        # Use simple indexing for valid indices
        # atr.values might be pandas array, convert to numpy
        vals = atr.values
        atr_arr[valid_idx_mask] = vals[bar_indices[valid_idx_mask]]
    
    # We will collect level values for stacking calc
    # Shape: (N_signals, N_types)
    level_values_matrix = np.full((n, len(level_types)), np.nan, dtype=np.float64)
    
    # Compute distances to each level type
    for k, level_type in enumerate(level_types):
        col_name = f'dist_to_{level_type.lower()}'
        col_name_norm = f'{col_name}_atr'
        
        if level_type in dynamic_levels:
            lvl_series = dynamic_levels[level_type]
            if lvl_series.empty:
                result[col_name] = np.nan
                result[col_name_norm] = np.nan
                continue
                
            vals = lvl_series.values
            
            # Align
            max_idx = len(vals) - 1
            valid_idx_mask = (bar_indices >= 0) & (bar_indices <= max_idx)
            
            current_lvl_vals = np.full(n, np.nan, dtype=np.float64)
            current_lvl_vals[valid_idx_mask] = vals[bar_indices[valid_idx_mask]]
            
            # Store for stacking
            level_values_matrix[:, k] = current_lvl_vals
            
            # Compute Signed Distance: Spot - Level
            # Positive = Above Level
            distances = spot_prices - current_lvl_vals
            
            result[col_name] = distances
            
            # ATR Normalized
            if atr_arr is not None:
                # Avoid div by zero
                norm_distances = np.full(n, np.nan, dtype=np.float64)
                safe_atr = (atr_arr > 0) & np.isfinite(atr_arr) & np.isfinite(distances)
                norm_distances[safe_atr] = distances[safe_atr] / atr_arr[safe_atr]
                result[col_name_norm] = norm_distances
            else:
                 result[col_name_norm] = np.nan
        else:
            result[col_name] = np.nan
            result[col_name_norm] = np.nan
            
    # Additional geometric features
    result['dist_to_tested_level'] = 0.0  # By definition
    
    # Level Stacking: Count nearby levels
    # Matrix broadcasting: |Level_Matrix - Signal_Level[:, None]|
    # Signal_Level shape (N,) -> (N, 1)
    
    # Diff matrix: (N, N_types)
    diff_matrix = np.abs(level_values_matrix - level_prices[:, None])
    
    stacking_bands = [2.0, 5.0, 10.0]  # ES points
    
    for band in stacking_bands:
        col_name = f'level_stacking_{int(band)}pt'
        
        # Count where 0 < dist <= band
        # 0 < dist excludes the level itself (if it matches perfectly)
        # Note: level_price is the *targeted* level.
        # If targeted level is PM_HIGH, then dist to PM_HIGH is 0.
        # We want to count *other* levels stacking.
        # So excluding 0 distance is correct.
        
        mask = (diff_matrix > 1e-4) & (diff_matrix <= band)
        # Sum along columns (k)
        stacking_count = np.sum(mask, axis=1)
        
        result[col_name] = stacking_count
    
    return result


class ComputeLevelDistancesStage(BaseStage):
    """
    Compute signed distances to all v1 structural levels.
    
    Compute structural level distances.
    
    Outputs:
        signals_df: Updated with distance features
    """
    
    @property
    def name(self) -> str:
        return "compute_level_distances"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'dynamic_levels', 'atr', 'ohlcv_1min']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        dynamic_levels = ctx.data.get('dynamic_levels', {})
        atr = ctx.data.get('atr')
        ohlcv_df = ctx.data['ohlcv_1min']
        
        signals_df = compute_all_level_distances(
            signals_df=signals_df,
            dynamic_levels=dynamic_levels,
            ohlcv_df=ohlcv_df,
            atr=atr
        )
        
        return {'signals_df': signals_df}
