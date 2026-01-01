"""
Compute distances to all structural levels.

Compute distances to structural levels:
All distance features should be SIGNED and have ATR-normalized variants.

Required distances (v1 level universe):
- dist_to_pm_high
- dist_to_pm_low
- dist_to_or_high
- dist_to_or_low
- dist_to_sma_90
- dist_to_ema_20
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_all_level_distances(
    signals_df: pd.DataFrame,
    dynamic_levels: Dict[str, pd.Series],
    atr: pd.Series = None
) -> pd.DataFrame:
    """
    Compute signed distances to all v1 structural levels.
    
    Per Claude's analysis: Missing OR_HIGH/OR_LOW distances.
    All distances are SIGNED (spot - level) and have ATR-normalized variants.
    
    Args:
        signals_df: DataFrame with signals (must have bar_idx, level_price)
        dynamic_levels: Dict of level series (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90, EMA_20)
        atr: Optional ATR series for normalized distances
    
    Returns:
        DataFrame with distance features added
    """
    if signals_df.empty:
        return signals_df
    
    # v1 level types
    level_types = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']
    
    n = len(signals_df)
    bar_idx = signals_df.get('bar_idx')
    level_prices = signals_df['level_price'].values.astype(np.float64)
    
    # Prefer actual spot/entry price when available
    if 'spot' in signals_df.columns:
        spot_prices = signals_df['spot'].astype(np.float64).to_numpy()
    elif 'entry_price' in signals_df.columns:
        spot_prices = signals_df['entry_price'].astype(np.float64).to_numpy()
    else:
        raise ValueError("signals_df missing 'spot' or 'entry_price' for distance computation")
    
    result = signals_df.copy()
    
    # Compute distances to each level type
    for level_type in level_types:
        col_name = f'dist_to_{level_type.lower()}'
        col_name_norm = f'{col_name}_atr'
        
        if level_type in dynamic_levels:
            level_series = dynamic_levels[level_type]
            distances = np.full(n, np.nan, dtype=np.float64)
            
            for i in range(n):
                if bar_idx is not None:
                    idx = int(bar_idx.iloc[i])
                    if 0 <= idx < len(level_series):
                        level_val = level_series.iloc[idx]
                        if np.isfinite(level_val) and np.isfinite(spot_prices[i]):
                            # Signed distance: positive = above level, negative = below
                            distances[i] = spot_prices[i] - level_val
            
            result[col_name] = distances
            
            # ATR-normalized version
            if atr is not None and not atr.empty:
                atr_vals = np.full(n, np.nan, dtype=np.float64)
                for i in range(n):
                    if bar_idx is not None:
                        idx = int(bar_idx.iloc[i])
                        if 0 <= idx < len(atr):
                            atr_vals[i] = atr.iloc[idx]
                
                # Normalize: distance / ATR
                norm_distances = np.where(
                    (atr_vals > 0) & np.isfinite(distances),
                    distances / atr_vals,
                    np.nan
                )
                result[col_name_norm] = norm_distances
            else:
                result[col_name_norm] = np.nan
        else:
            # Level type not available
            result[col_name] = np.nan
            result[col_name_norm] = np.nan
    
    # Additional geometric features
    
    # Distance to tested level (should be ~0 at entry, increases as moves away)
    result['dist_to_tested_level'] = 0.0  # By definition at entry
    
    # Level stacking: count how many other levels are nearby
    stacking_bands = [2.0, 5.0, 10.0]  # ES points
    
    for band in stacking_bands:
        col_name = f'level_stacking_{int(band)}pt'
        stacking_count = np.zeros(n, dtype=np.int8)
        
        for i in range(n):
            level = level_prices[i]
            count = 0
            
            for level_type in level_types:
                if level_type in dynamic_levels and bar_idx is not None:
                    idx = int(bar_idx.iloc[i])
                    if 0 <= idx < len(dynamic_levels[level_type]):
                        other_level = dynamic_levels[level_type].iloc[idx]
                        if np.isfinite(other_level):
                            dist = abs(other_level - level)
                            if 0 < dist <= band:  # Exclude self
                                count += 1
            
            stacking_count[i] = count
        
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
        return ['signals_df', 'dynamic_levels', 'atr']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        dynamic_levels = ctx.data.get('dynamic_levels', {})
        atr = ctx.data.get('atr')
        
        signals_df = compute_all_level_distances(
            signals_df=signals_df,
            dynamic_levels=dynamic_levels,
            atr=atr
        )
        
        return {'signals_df': signals_df}
