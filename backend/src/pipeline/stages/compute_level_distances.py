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
    Compute distance to THE level.
    
    Single-level pipeline: signals filtered to ONE level type.
    - dist_to_level: signed distance from entry price to level (positive = above)
    - dist_to_level_atr: ATR-normalized distance
    - level_stacking_*: set to 0 (disabled for single-level pipeline, compute post-pipeline if needed)
    
    Args:
        signals_df: DataFrame with signals for ONE level type
        dynamic_levels: Dict with THE level series (single entry for level-specific pipeline)
        ohlcv_df: OHLCV DataFrame for alignment
        atr: ATR series for normalized distances
    
    Returns:
        DataFrame with distance features
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
    
    # 3. Level Stacking (confluence): DISABLED for single-level pipeline
    # This feature requires all levels to be generated, which violates
    # the "one level at a time" principle. Can be computed post-pipeline
    # if needed for cross-level confluence analysis.
    
    result['level_stacking_2pt'] = 0
    result['level_stacking_5pt'] = 0
    result['level_stacking_10pt'] = 0
    
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
