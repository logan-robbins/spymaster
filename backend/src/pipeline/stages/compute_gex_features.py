"""
Compute ES options gamma exposure (GEX) features.

Strike-banded GEX features around tested level. Aggregates gamma exposure
within ±1, ±2, ±3 strikes from the tested level (not fixed point distances).
ES 0DTE strikes are typically spaced 5 points apart at ATM.
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
    band_points: List[float] = None
) -> pd.DataFrame:
    """
    Compute gamma exposure in strike bands around each tested level.
    
    GEX feature computation:
    - Filter to 0DTE options (exp_date == session date)
    - Use open_interest from greeks snapshots if available, else trades
    - Compute GEX per strike, then aggregate into strike bands
    - Summarize relative to tested level
    
    Strike bands: ±1, ±2, ±3 strikes from tested level.
    For ES 0DTE ATM options, strikes are typically spaced 5 points apart,
    so ±1 strike = ±5 points, ±2 strikes = ±10 points, ±3 strikes = ±15 points.
    
    Args:
        signals_df: DataFrame with signals
        option_trades_df: Option trades/greeks with gamma, open_interest
        date: Session date for 0DTE filtering
        band_points: Strike spacing in points (default: [5, 10, 15] for ±1/±2/±3 strikes)
    
    Returns:
        DataFrame with GEX features added
    """
    if band_points is None:
        base = float(CONFIG.ES_0DTE_STRIKE_SPACING)
        band_points = [base * 1, base * 2, base * 3]
    band_labels = list(range(1, len(band_points) + 1))
    
    if signals_df.empty or option_trades_df is None or option_trades_df.empty:
        # Return with zero GEX features
        result = signals_df.copy()
        for band in band_labels:
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
        for band in band_labels:
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
        for band in band_labels:
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
    
    # Compute GEX per strike (dealer gamma = -customer gamma).
    # Standardized: we use gamma × contracts for relative comparisons.
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

    gex_all = gex_by_strike.groupby('strike')['dealer_gex'].sum()
    gex_calls = gex_by_strike[gex_by_strike['right'] == 'C'].groupby('strike')['dealer_gex'].sum()
    gex_puts = gex_by_strike[gex_by_strike['right'] == 'P'].groupby('strike')['dealer_gex'].sum()

    strikes_available = gex_all.index.to_numpy(dtype=np.float64)
    gex_all_vals = gex_all.to_numpy(dtype=np.float64)
    gex_call_vals = gex_calls.reindex(strikes_available, fill_value=0.0).to_numpy(dtype=np.float64)
    gex_put_vals = gex_puts.reindex(strikes_available, fill_value=0.0).to_numpy(dtype=np.float64)

    # Initialize result arrays (band labels align with schema)
    result_dict = {}
    for band in band_labels:
        result_dict[f'gex_above_{band}strike'] = np.zeros(n, dtype=np.float64)
        result_dict[f'gex_below_{band}strike'] = np.zeros(n, dtype=np.float64)
        result_dict[f'call_gex_above_{band}strike'] = np.zeros(n, dtype=np.float64)
        result_dict[f'put_gex_below_{band}strike'] = np.zeros(n, dtype=np.float64)
    
    gex_asymmetry = np.zeros(n, dtype=np.float64)
    gex_ratio = np.zeros(n, dtype=np.float64)
    net_gex_2strike = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        level = level_prices[i]
        if strikes_available.size == 0:
            continue

        for label, band in zip(band_labels, band_points):
            above_mask = (strikes_available > level) & (strikes_available <= level + band)
            below_mask = (strikes_available < level) & (strikes_available >= level - band)

            if np.any(above_mask):
                result_dict[f'gex_above_{label}strike'][i] = float(gex_all_vals[above_mask].sum())
                result_dict[f'call_gex_above_{label}strike'][i] = float(gex_call_vals[above_mask].sum())
            if np.any(below_mask):
                result_dict[f'gex_below_{label}strike'][i] = float(gex_all_vals[below_mask].sum())
                result_dict[f'put_gex_below_{label}strike'][i] = float(gex_put_vals[below_mask].sum())

        # Asymmetry and ratio (using widest band)
        max_label = band_labels[-1]
        gex_above_max = result_dict[f'gex_above_{max_label}strike'][i]
        gex_below_max = result_dict[f'gex_below_{max_label}strike'][i]

        gex_asymmetry[i] = gex_above_max - gex_below_max

        denom = abs(gex_below_max) + 1e-6
        gex_ratio[i] = gex_above_max / denom

        # Net GEX within ±max band (include all listed strikes within points band)
        max_band = max(band_points) if band_points else 0.0
        in_range = np.abs(strikes_available - level) <= max_band
        net_gex = float(gex_all_vals[in_range].sum()) if np.any(in_range) else 0.0
        net_gex_2strike[i] = net_gex  # Keep name aligned with schema
    
    result = signals_df.copy()
    for key, values in result_dict.items():
        result[key] = values
    result['gex_asymmetry'] = gex_asymmetry
    result['gex_ratio'] = gex_ratio
    result['net_gex_2strike'] = net_gex_2strike
    
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
