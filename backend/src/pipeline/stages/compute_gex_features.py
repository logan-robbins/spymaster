"""
Compute SPX options gamma exposure (GEX) features.

Per Final Call v1 spec Section 6.8: Strike-banded GEX features around tested level.

Key differences for SPX vs SPY:
- SPX strikes: $5-$10 intervals (vs SPY $1)
- SPX 0DTE: Cash-settled, European exercise
- Institutional gamma exposure is MUCH larger
- Use Open Interest from greeks snapshots (if available) or trades
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_strike_banded_gex(
    signals_df: pd.DataFrame,
    option_trades_df: pd.DataFrame,
    date: str,
    strike_bands: List[int] = None
) -> pd.DataFrame:
    """
    Compute gamma exposure in strike bands around each tested level.
    
    Per Final Call spec Section 6.8 + user clarification:
    - Filter to 0DTE options (exp_date == session date)
    - Use open_interest from greeks snapshots if available, else trades
    - Compute GEX per strike, then aggregate into bands
    - Summarize relative to tested level
    
    Strike bands (ES 0DTE): ±1, ±2, ±3 strikes (at 5-point spacing)
    We model 3-strike minimum moves, so bands up to ±3 are needed.
    
    Args:
        signals_df: DataFrame with signals
        option_trades_df: Option trades/greeks with gamma, open_interest
        date: Session date for 0DTE filtering
        strike_bands: List of strike offsets (default [1, 2, 3])
    
    Returns:
        DataFrame with GEX features added
    """
    if strike_bands is None:
        strike_bands = [1, 2, 3]  # ±1, ±2, ±3 strikes (for 3-strike threshold)
    
    if signals_df.empty or option_trades_df is None or option_trades_df.empty:
        # Return with zero GEX features
        result = signals_df.copy()
        for band in strike_bands:
            result[f'gex_above_{band}strike'] = 0.0
            result[f'gex_below_{band}strike'] = 0.0
            result[f'call_gex_above_{band}strike'] = 0.0
            result[f'put_gex_below_{band}strike'] = 0.0
        result['gex_asymmetry'] = 0.0
        result['gex_ratio'] = 0.0
        result['net_gex_2strike'] = 0.0
        return result
    
    # Filter to 0DTE options
    opt_df = option_trades_df.copy()
    if 'exp_date' in opt_df.columns:
        opt_df = opt_df[opt_df['exp_date'].astype(str) == date]
    
    if opt_df.empty:
        result = signals_df.copy()
        for band in strike_bands:
            result[f'gex_above_{band}strike'] = 0.0
            result[f'gex_below_{band}strike'] = 0.0
            result[f'call_gex_above_{band}strike'] = 0.0
            result[f'put_gex_below_{band}strike'] = 0.0
        result['gex_asymmetry'] = 0.0
        result['gex_ratio'] = 0.0
        result['net_gex_2strike'] = 0.0
        return result
    
    # Ensure required columns
    required = ['strike', 'right', 'gamma']
    if not all(col in opt_df.columns for col in required):
        # Return zeros
        result = signals_df.copy()
        for band in strike_bands:
            result[f'gex_above_{band}strike'] = 0.0
            result[f'gex_below_{band}strike'] = 0.0
            result[f'call_gex_above_{band}strike'] = 0.0
            result[f'put_gex_below_{band}strike'] = 0.0
        result['gex_asymmetry'] = 0.0
        result['gex_ratio'] = 0.0
        result['net_gex_2strike'] = 0.0
        return result
    
    # Use open_interest if available, else size
    if 'open_interest' in opt_df.columns:
        opt_df['contracts'] = opt_df['open_interest'].fillna(0)
    else:
        opt_df['contracts'] = opt_df['size'].fillna(0)
    
    # Compute GEX per strike (dealer gamma = -customer gamma)
    # GEX in dollars = gamma × contracts × 100 (multiplier) × spot
    # Standardized: we'll use gamma × contracts for relative comparisons
    opt_df['strike'] = opt_df['strike'].astype(np.float64)
    opt_df['gamma'] = opt_df['gamma'].fillna(0).astype(np.float64)
    opt_df['contracts'] = opt_df['contracts'].astype(np.int64)
    opt_df['right'] = opt_df['right'].astype(str)
    
    # Aggregate by strike + right
    gex_by_strike = opt_df.groupby(['strike', 'right']).agg({
        'gamma': 'sum',
        'contracts': 'sum'
    }).reset_index()
    
    # Compute dealer GEX (negative of customer gamma)
    gex_by_strike['dealer_gex'] = -gex_by_strike['gamma'] * gex_by_strike['contracts']
    
    n = len(signals_df)
    level_prices = signals_df['level_price'].values.astype(np.float64)
    
    # Initialize result arrays (±1, ±2, ±3 strikes)
    result_dict = {}
    for band in strike_bands:
        result_dict[f'gex_above_{band}strike'] = np.zeros(n, dtype=np.float64)
        result_dict[f'gex_below_{band}strike'] = np.zeros(n, dtype=np.float64)
        result_dict[f'call_gex_above_{band}strike'] = np.zeros(n, dtype=np.float64)
        result_dict[f'put_gex_below_{band}strike'] = np.zeros(n, dtype=np.float64)
    
    gex_asymmetry = np.zeros(n, dtype=np.float64)
    gex_ratio = np.zeros(n, dtype=np.float64)
    net_gex_2strike = np.zeros(n, dtype=np.float64)
    
    # ES 0DTE strike spacing: 25 points (validated from actual data)
    # Near ATM: 25pt dominant, some 5pt very tight
    # Per user requirement: 3-strike minimum move
    strikes_available = sorted(gex_by_strike['strike'].unique())
    
    ES_STRIKE_SPACING = 25.0  # ES 0DTE ATM spacing (validated 2025-11-03 data)
    
    for i in range(n):
        level = level_prices[i]
        
        # For each band, find strikes at ±band positions
        for band in strike_bands:
            # Find strikes above and below at exact multiples
            # band=1 → ±5pts, band=2 → ±10pts, band=3 → ±15pts
            target_above = level + (band * ES_STRIKE_SPACING)
            target_below = level - (band * ES_STRIKE_SPACING)
            
            # Find nearest actual strikes
            strikes_above = [s for s in strikes_available if s >= target_above]
            strikes_below = [s for s in strikes_available if s <= target_below]
            
            if strikes_above:
                strike_above = min(strikes_above, key=lambda s: abs(s - target_above))
                gex_above = gex_by_strike[gex_by_strike['strike'] == strike_above]['dealer_gex'].sum()
                result_dict[f'gex_above_{band}strike'][i] = gex_above
                
                # Call GEX above (resistance)
                call_gex = gex_by_strike[
                    (gex_by_strike['strike'] == strike_above) &
                    (gex_by_strike['right'] == 'C')
                ]['dealer_gex'].sum()
                result_dict[f'call_gex_above_{band}strike'][i] = call_gex
            
            if strikes_below:
                strike_below = max(strikes_below, key=lambda s: abs(s - target_below))
                gex_below = gex_by_strike[gex_by_strike['strike'] == strike_below]['dealer_gex'].sum()
                result_dict[f'gex_below_{band}strike'][i] = gex_below
                
                # Put GEX below (support)
                put_gex = gex_by_strike[
                    (gex_by_strike['strike'] == strike_below) &
                    (gex_by_strike['right'] == 'P')
                ]['dealer_gex'].sum()
                result_dict[f'put_gex_below_{band}strike'][i] = put_gex
        
        # Asymmetry and ratio (using ±3 strikes for 3-strike threshold model)
        gex_above_3 = result_dict['gex_above_3strike'][i]
        gex_below_3 = result_dict['gex_below_3strike'][i]
        
        gex_asymmetry[i] = gex_above_3 - gex_below_3
        
        denom = abs(gex_below_3) + 1e-6
        gex_ratio[i] = gex_above_3 / denom
        
        # Net GEX within ±3 strikes (3 × 5pt = ±15 points)
        strikes_in_range = [s for s in strikes_available if abs(s - level) <= 3 * ES_STRIKE_SPACING]
        net_gex = sum(
            gex_by_strike[gex_by_strike['strike'] == s]['dealer_gex'].sum()
            for s in strikes_in_range
        )
        net_gex_2strike[i] = net_gex  # Keep name for compatibility, but actually 3-strike
    
    result = signals_df.copy()
    for key, values in result_dict.items():
        result[key] = values
    result['gex_asymmetry'] = gex_asymmetry
    result['gex_ratio'] = gex_ratio
    result['net_gex_2strike'] = net_gex_2strike
    
    return result


class ComputeGEXFeaturesStage(BaseStage):
    """
    Compute SPX options gamma exposure features.
    
    Per Final Call v1 spec Section 6.8: Strike-banded GEX around tested level.
    
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
            date=ctx.date,
            strike_bands=[1, 2, 3]  # ES 0DTE: ±1/±2/±3 strikes (5pt, 10pt, 15pt)
        )
        
        return {'signals_df': signals_df}

