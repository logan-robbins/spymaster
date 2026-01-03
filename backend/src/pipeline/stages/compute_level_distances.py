import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_level_distances(
    signals_df: pd.DataFrame,
    dynamic_levels: Dict[str, pd.Series],
    ohlcv_df: pd.DataFrame,
    atr: pd.Series = None
) -> pd.DataFrame:
    """
    Compute distance to THE level and level stacking (confluence).
    
    Single-level pipeline: signals filtered to ONE level type.
    - dist_to_level: distance from current price to THE level
    - dist_to_level_atr: ATR-normalized distance
    - level_stacking_*: count of OTHER levels within range (confluence)
    
    Args:
        signals_df: DataFrame with signals for ONE level type
        dynamic_levels: Dict of ALL level series (for stacking calculation)
        ohlcv_df: OHLCV DataFrame for alignment
        atr: ATR series for normalized distances
    
    Returns:
        DataFrame with distance and stacking features
    """
    if signals_df.empty:
        return signals_df
    
    n = len(signals_df)
    result = signals_df.copy()
    
    # 1. Distance to THE level (simple calculation)
    level_prices = signals_df['level_price'].values.astype(np.float64)
    
    if 'entry_price' in signals_df.columns:
        spot_prices = signals_df['entry_price'].astype(np.float64).to_numpy()
    else:
        spot_prices = level_prices
    
    # Signed distance: positive = above level, negative = below level
    result['dist_to_level'] = spot_prices - level_prices
    
    # 2. ATR-normalized distance
    if atr is not None and not atr.empty and ohlcv_df is not None and not ohlcv_df.empty:
        # Resolve bar indices for ATR lookup
        if 'bar_idx' in signals_df.columns and not signals_df['bar_idx'].isnull().all():
            bar_indices = signals_df['bar_idx'].fillna(-1).astype(int).values
        else:
            # Map signals to OHLCV bars via timestamp
            if 'ts_ns' in ohlcv_df.columns:
                ohlcv_ts = ohlcv_df['ts_ns'].values.astype(np.int64)
            elif 'timestamp' in ohlcv_df.columns:
                ohlcv_ts = ohlcv_df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
            else:
                ohlcv_ts = ohlcv_df.index.values.astype('datetime64[ns]').astype(np.int64)
            
            sig_ts = signals_df['ts_ns'].values.astype(np.int64)
            bar_indices = np.searchsorted(ohlcv_ts, sig_ts, side='right') - 1
            bar_indices = np.clip(bar_indices, 0, len(ohlcv_ts) - 1)
        
        # Get ATR values
        max_idx = len(atr) - 1
        valid_idx_mask = (bar_indices >= 0) & (bar_indices <= max_idx)
        
        atr_arr = np.full(n, np.nan, dtype=np.float64)
        atr_arr[valid_idx_mask] = atr.values[bar_indices[valid_idx_mask]]
        
        # Normalize
        distances = result['dist_to_level'].values
        norm_distances = np.full(n, np.nan, dtype=np.float64)
        safe_atr = (atr_arr > 0) & np.isfinite(atr_arr) & np.isfinite(distances)
        norm_distances[safe_atr] = distances[safe_atr] / atr_arr[safe_atr]
        result['dist_to_level_atr'] = norm_distances
    else:
        result['dist_to_level_atr'] = np.nan
    
    # 3. Level Stacking (confluence): count OTHER levels near THE level
    level_types = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90']
    level_values_matrix = np.full((n, len(level_types)), np.nan, dtype=np.float64)
    
    # Build matrix of all level values at signal timestamps
    if ohlcv_df is not None and not ohlcv_df.empty:
        # Get bar indices if not already computed
        if 'bar_indices' not in locals():
            if 'bar_idx' in signals_df.columns and not signals_df['bar_idx'].isnull().all():
                bar_indices = signals_df['bar_idx'].fillna(-1).astype(int).values
            else:
                if 'ts_ns' in ohlcv_df.columns:
                    ohlcv_ts = ohlcv_df['ts_ns'].values.astype(np.int64)
                elif 'timestamp' in ohlcv_df.columns:
                    ohlcv_ts = ohlcv_df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
                else:
                    ohlcv_ts = ohlcv_df.index.values.astype('datetime64[ns]').astype(np.int64)
                
                sig_ts = signals_df['ts_ns'].values.astype(np.int64)
                bar_indices = np.searchsorted(ohlcv_ts, sig_ts, side='right') - 1
                bar_indices = np.clip(bar_indices, 0, len(ohlcv_ts) - 1)
        
        # Populate matrix
        for k, level_type in enumerate(level_types):
            if level_type in dynamic_levels:
                lvl_series = dynamic_levels[level_type]
                if not lvl_series.empty:
                    vals = lvl_series.values
                    max_idx = len(vals) - 1
                    valid_idx_mask = (bar_indices >= 0) & (bar_indices <= max_idx)
                    current_lvl_vals = np.full(n, np.nan, dtype=np.float64)
                    current_lvl_vals[valid_idx_mask] = vals[bar_indices[valid_idx_mask]]
                    level_values_matrix[:, k] = current_lvl_vals
            
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
    Compute distance to THE level and level stacking (confluence).
    
    Single-level pipeline: computes dist_to_level and level_stacking_*.
    
    Outputs:
        signals_df: Updated with distance and confluence features
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
        
        signals_df = compute_level_distances(
            signals_df=signals_df,
            dynamic_levels=dynamic_levels,
            ohlcv_df=ohlcv_df,
            atr=atr
        )
        
        return {'signals_df': signals_df}
