"""
Compute ES options gamma exposure (GEX) features.

Strike-banded GEX features around tested level. Aggregates gamma exposure
within ±1, ±2, ±3 strikes from the tested level (not fixed point distances).
ES 0DTE strikes are typically spaced 5 points apart at ATM.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_strike_banded_gex(
    signals_df: pd.DataFrame,
    option_trades_df: pd.DataFrame,
    date: str,
    band_points: List[float] = None
) -> pd.DataFrame:
    """
    Compute gamma exposure in strike bands around each tested level using Vectorized Prefix Sums.
    
    Optimization:
    - Pre-calculates cumulative sums of Gamma per strike.
    - Uses np.searchsorted to find band boundaries for all signals in O(N log M).
    - Computes aggregated gamma via O(1) subtraction of cumsums.
    
    Strike bands: ±1, ±2, ±3 strikes from tested level (approx ±5, ±10, ±15 pts).
    
    Args:
        signals_df: DataFrame with signals
        option_trades_df: Option trades/greeks with gamma, open_interest
        date: Session date for 0DTE filtering
        band_points: Strike spacing in points (default: [5, 10, 15])
    
    Returns:
        DataFrame with GEX features added
    """
    if band_points is None:
        base = float(CONFIG.ES_0DTE_STRIKE_SPACING)
        band_points = [base * 1, base * 2, base * 3]
    band_labels = list(range(1, len(band_points) + 1))
    
    # Initialize result columns with 0.0
    result = signals_df.copy()
    zero_fill_cols = []
    for band in band_labels:
        zero_fill_cols.extend([
            f'gex_above_{band}strike', f'gex_below_{band}strike',
            f'call_gex_above_{band}strike', f'put_gex_below_{band}strike'
        ])
    zero_fill_cols.extend(['gex_asymmetry', 'gex_ratio', 'net_gex_2strike'])
    
    for col in zero_fill_cols:
        result[col] = 0.0
        
    if signals_df.empty or option_trades_df is None or option_trades_df.empty:
        return result
    
    # Filter to 0DTE options
    opt_df = option_trades_df.copy()
    if 'exp_date' in opt_df.columns:
        # Assuming date string format matches
        opt_df = opt_df[opt_df['exp_date'].astype(str) == date]
    
    # Ensure required columns
    required = ['strike', 'right', 'gamma']
    if opt_df.empty or not all(col in opt_df.columns for col in required):
        return result
    
    # Use open_interest if available, else size
    if 'open_interest' in opt_df.columns:
        opt_df['contracts'] = opt_df['open_interest'].fillna(0)
    else:
        opt_df['contracts'] = opt_df['size'].fillna(0)
    
    # Compute GEX per strike
    opt_df['strike'] = opt_df['strike'].astype(np.float64)
    opt_df['gamma'] = opt_df['gamma'].fillna(0).astype(np.float64)
    opt_df['contracts'] = opt_df['contracts'].astype(np.float64)
    
    # Dealer GEX = -Gamma * Contracts * 100 (multiplier usually handled in gamma or implicit)
    # The existing code used -gamma * contracts. We stick to that.
    opt_df['dealer_gex'] = -opt_df['gamma'] * opt_df['contracts']
    
    # Group by Strike and Right
    # Pivot to get: Strike | TotalGEX | CallGEX | PutGEX
    gex_grouped = opt_df.groupby(['strike', 'right'])['dealer_gex'].sum().unstack('right', fill_value=0.0)
    if 'C' not in gex_grouped.columns: gex_grouped['C'] = 0.0
    if 'P' not in gex_grouped.columns: gex_grouped['P'] = 0.0
    
    gex_grouped['Total'] = gex_grouped['C'] + gex_grouped['P']
    gex_grouped = gex_grouped.sort_index()
    
    # Extract arrays for vectorization
    strikes = gex_grouped.index.values.astype(np.float64)
    total_gex = gex_grouped['Total'].values.astype(np.float64)
    call_gex = gex_grouped['C'].values.astype(np.float64)
    put_gex = gex_grouped['P'].values.astype(np.float64)
    
    if len(strikes) == 0:
        return result
        
    # Pre-compute Cumulative Sums
    # Pad with 0 at start for easier indexing logic: sum(i..j) = cum[j] - cum[i]
    # searchsorted usually gives index in original array.
    # To map to cumsum (len+1), index i corresponds to sum BEFORE element i.
    # cumsum[k] = sum(0..k-1)
    # sum(idx_start to idx_end-1) = cumsum[idx_end] - cumsum[idx_start]
    
    cum_total = np.concatenate(([0.0], np.cumsum(total_gex)))
    cum_call = np.concatenate(([0.0], np.cumsum(call_gex)))
    cum_put = np.concatenate(([0.0], np.cumsum(put_gex)))
    
    # Vectorized Lookup for Signals
    levels = signals_df['level_price'].values.astype(np.float64)
    
    # Helper to sum range [start_idx, end_idx)
    def fast_sum(cum_arr, idx_start, idx_end):
        # Result array
        return cum_arr[idx_end] - cum_arr[idx_start]

    for label, band in zip(band_labels, band_points):
        # 1. Above Region: (Level, Level + Band]
        # range > level AND <= level + band
        # idx_start = searchsorted(strikes, level, 'right') -> first element > level
        # idx_end = searchsorted(strikes, level + band, 'right') -> first element > level+band (so idx_end-1 is <=)
        
        idx_above_start = np.searchsorted(strikes, levels, side='right')
        idx_above_end = np.searchsorted(strikes, levels + band, side='right')
        
        # Clip just in case, though searchsorted does logic correctly for indices
        
        result[f'gex_above_{label}strike'] = fast_sum(cum_total, idx_above_start, idx_above_end)
        result[f'call_gex_above_{label}strike'] = fast_sum(cum_call, idx_above_start, idx_above_end)
        
        # 2. Below Region: [Level - Band, Level)
        # range >= level - band AND < level
        # idx_start = searchsorted(strikes, level - band, 'left') -> first element >= val
        # idx_end = searchsorted(strikes, level, 'left') -> first element >= level (so end-1 is < level)
        
        idx_below_start = np.searchsorted(strikes, levels - band, side='left')
        idx_below_end = np.searchsorted(strikes, levels, side='left')
        
        result[f'gex_below_{label}strike'] = fast_sum(cum_total, idx_below_start, idx_below_end)
        result[f'put_gex_below_{label}strike'] = fast_sum(cum_put, idx_below_start, idx_below_end)

    # Derived Metrics (Asymmetry, Ratio) using Max Band
    max_label = band_labels[-1]
    gex_above_max = result[f'gex_above_{max_label}strike']
    gex_below_max = result[f'gex_below_{max_label}strike']
    
    result['gex_asymmetry'] = gex_above_max - gex_below_max
    
    denom = np.abs(gex_below_max) + 1e-6
    result['gex_ratio'] = gex_above_max / denom
    
    # Net GEX within ±Max Band (Inclusive of Level)
    # Range: [Level - MaxBand, Level + MaxBand]
    # idx_start = searchsorted(strikes, level - max_band, 'left')
    # idx_end = searchsorted(strikes, level + max_band, 'right')
    
    max_band = band_points[-1]
    idx_net_start = np.searchsorted(strikes, levels - max_band, side='left')
    idx_net_end = np.searchsorted(strikes, levels + max_band, side='right')
    
    result['net_gex_2strike'] = fast_sum(cum_total, idx_net_start, idx_net_end)
    
    return result


class ComputeGEXFeaturesStage(BaseStage):
    """
    Compute ES options gamma exposure features.
    
    Point-banded GEX around tested level using all listed strikes within a points band.
    
    Outputs:
        signals_df: Updated with GEX features
    """
    
    @property
    def name(self) -> str:
        return "compute_gex_features"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'option_trades_df']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        option_trades_df = ctx.data['option_trades_df']
        
        signals_df = compute_strike_banded_gex(
            signals_df=signals_df,
            option_trades_df=option_trades_df,
            date=ctx.date
        )
        
        return {'signals_df': signals_df}
