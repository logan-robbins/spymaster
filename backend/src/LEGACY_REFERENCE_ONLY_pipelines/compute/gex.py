"""
Shared GEX (Gamma Exposure) computation functions.

Computes gamma exposure from options data.
Can be level-relative or market-wide.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_gex_features(
    signal_ts: np.ndarray,
    option_trades_df: pd.DataFrame,
    spot_prices: np.ndarray,
    level_price: Optional[float] = None,
    strike_range: float = 50.0,
) -> Dict[str, np.ndarray]:
    """
    Compute GEX features at signal timestamps.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        option_trades_df: DataFrame with option trades (ts_event_ns, strike, option_type, gex)
        spot_prices: Array of spot prices at each signal
        level_price: If set, compute GEX relative to level.
                    If None, compute total GEX (global mode).
        strike_range: Range for banded GEX features (default 50pt)
    
    Returns:
        Dictionary of GEX features
    """
    n_signals = len(signal_ts)
    
    # Initialize result
    if level_price is not None:
        # Level-relative features
        result = {
            'total_gex': np.zeros(n_signals, dtype=np.float64),
            'gex_above_level': np.zeros(n_signals, dtype=np.float64),
            'gex_below_level': np.zeros(n_signals, dtype=np.float64),
            'call_gex_above_level': np.zeros(n_signals, dtype=np.float64),
            'call_gex_below_level': np.zeros(n_signals, dtype=np.float64),
            'put_gex_above_level': np.zeros(n_signals, dtype=np.float64),
            'put_gex_below_level': np.zeros(n_signals, dtype=np.float64),
            'gex_asymmetry': np.zeros(n_signals, dtype=np.float64),
            'gex_ratio': np.zeros(n_signals, dtype=np.float64),
        }
    else:
        # Global features
        result = {
            'total_gex': np.zeros(n_signals, dtype=np.float64),
            'total_call_gex': np.zeros(n_signals, dtype=np.float64),
            'total_put_gex': np.zeros(n_signals, dtype=np.float64),
            'gex_above_spot': np.zeros(n_signals, dtype=np.float64),
            'gex_below_spot': np.zeros(n_signals, dtype=np.float64),
            'gex_asymmetry': np.zeros(n_signals, dtype=np.float64),
            'gex_ratio': np.zeros(n_signals, dtype=np.float64),
            'put_call_gex_ratio': np.zeros(n_signals, dtype=np.float64),
        }
    
    if n_signals == 0 or option_trades_df.empty:
        return result
    
    # Ensure required columns
    required = ['ts_event_ns', 'strike', 'option_type']
    if not all(c in option_trades_df.columns for c in required):
        return result
    
    # Compute GEX if not present
    if 'gex' not in option_trades_df.columns:
        # Simplified GEX: just use premium * sign
        # In reality, GEX = gamma * OI * spot^2 / 100
        option_trades_df = option_trades_df.copy()
        option_trades_df['gex'] = option_trades_df.get('premium', 1.0)
    
    df = option_trades_df.sort_values('ts_event_ns')
    opt_ts = df['ts_event_ns'].values.astype(np.int64)
    strikes = df['strike'].values.astype(np.float64)
    opt_types = df['option_type'].values  # 'C' or 'P'
    gex_values = df['gex'].values.astype(np.float64)
    
    # Sign GEX by option type: calls positive, puts negative
    gex_signed = np.where(opt_types == 'C', gex_values, -gex_values)
    
    # Cumulative sums for efficient windowing
    cum_total = np.cumsum(gex_signed)
    cum_call = np.cumsum(np.where(opt_types == 'C', gex_values, 0.0))
    cum_put = np.cumsum(np.where(opt_types == 'P', gex_values, 0.0))
    
    # Find index for each signal (use all options up to signal time)
    idx_lookup = np.searchsorted(opt_ts, signal_ts, side='right')
    
    for i in range(n_signals):
        idx = idx_lookup[i]
        if idx == 0:
            continue
        
        # Total GEX up to this point
        result['total_gex'][i] = cum_total[idx - 1]
        
        ref_price = level_price if level_price is not None else spot_prices[i]
        
        # Filter options by strike relative to reference price
        mask_before = np.arange(len(strikes)) < idx
        
        if level_price is not None:
            # Level-relative
            mask_above = mask_before & (strikes > ref_price)
            mask_below = mask_before & (strikes <= ref_price)
            
            result['gex_above_level'][i] = np.sum(gex_signed[mask_above])
            result['gex_below_level'][i] = np.sum(gex_signed[mask_below])
            
            mask_call = opt_types == 'C'
            mask_put = opt_types == 'P'
            
            result['call_gex_above_level'][i] = np.sum(gex_values[mask_above & mask_call])
            result['call_gex_below_level'][i] = np.sum(gex_values[mask_below & mask_call])
            result['put_gex_above_level'][i] = np.sum(gex_values[mask_above & mask_put])
            result['put_gex_below_level'][i] = np.sum(gex_values[mask_below & mask_put])
            
            # Asymmetry and ratio
            above = result['gex_above_level'][i]
            below = result['gex_below_level'][i]
        else:
            # Global
            result['total_call_gex'][i] = cum_call[idx - 1]
            result['total_put_gex'][i] = cum_put[idx - 1]
            
            mask_above = mask_before & (strikes > ref_price)
            mask_below = mask_before & (strikes <= ref_price)
            
            result['gex_above_spot'][i] = np.sum(gex_signed[mask_above])
            result['gex_below_spot'][i] = np.sum(gex_signed[mask_below])
            
            above = result['gex_above_spot'][i]
            below = result['gex_below_spot'][i]
            
            # Put/call ratio
            call_gex = result['total_call_gex'][i]
            put_gex = result['total_put_gex'][i]
            if call_gex != 0:
                result['put_call_gex_ratio'][i] = put_gex / call_gex
        
        # Common: asymmetry and ratio
        # Use absolute values for asymmetry since above/below may have opposite signs
        # (e.g., above is positive call gamma, below is negative put gamma)
        abs_above = abs(above)
        abs_below = abs(below)
        total = abs_above + abs_below
        if total > 0:
            # Asymmetry: positive means more GEX magnitude above, negative means more below
            result['gex_asymmetry'][i] = (abs_above - abs_below) / total * 100
        
        if abs_below > 0:
            # Ratio of magnitudes (always positive)
            result['gex_ratio'][i] = abs_above / abs_below
    
    return result


def compute_tide_features(
    signal_ts: np.ndarray,
    option_trades_df: pd.DataFrame,
    level_price: Optional[float] = None,
    split_range: float = 25.0,
) -> Dict[str, np.ndarray]:
    """
    Compute Market Tide (premium flow) features.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        option_trades_df: DataFrame with option trades
        level_price: If set, compute tide relative to level.
        split_range: Range for above/below split (default 25pt)
    
    Returns:
        Dictionary of tide features
    """
    n_signals = len(signal_ts)
    
    result = {
        'call_tide': np.zeros(n_signals, dtype=np.float64),
        'put_tide': np.zeros(n_signals, dtype=np.float64),
    }
    
    if level_price is not None:
        result['call_tide_above'] = np.zeros(n_signals, dtype=np.float64)
        result['call_tide_below'] = np.zeros(n_signals, dtype=np.float64)
        result['put_tide_above'] = np.zeros(n_signals, dtype=np.float64)
        result['put_tide_below'] = np.zeros(n_signals, dtype=np.float64)
    
    if n_signals == 0 or option_trades_df.empty:
        return result
    
    # Compute premium flow
    df = option_trades_df.copy()
    if 'premium' not in df.columns:
        if 'price' in df.columns and 'size' in df.columns:
            df['premium'] = df['price'] * df['size'] * 100  # Options multiplier
        else:
            return result
    
    df = df.sort_values('ts_event_ns')
    opt_ts = df['ts_event_ns'].values.astype(np.int64)
    strikes = df['strike'].values.astype(np.float64)
    opt_types = df['option_type'].values
    premiums = df['premium'].values.astype(np.float64)
    
    # Cumulative sums
    cum_call = np.cumsum(np.where(opt_types == 'C', premiums, 0.0))
    cum_put = np.cumsum(np.where(opt_types == 'P', premiums, 0.0))
    
    # Lookback window (e.g., 5 minutes)
    lookback_ns = int(300 * 1e9)
    
    for i in range(n_signals):
        ts = signal_ts[i]
        start_ts = ts - lookback_ns
        
        idx_start = np.searchsorted(opt_ts, start_ts, side='right')
        idx_end = np.searchsorted(opt_ts, ts, side='right')
        
        if idx_end == 0:
            continue
        
        # Total tide in window
        call_end = cum_call[idx_end - 1] if idx_end > 0 else 0
        call_start = cum_call[idx_start - 1] if idx_start > 0 else 0
        result['call_tide'][i] = call_end - call_start
        
        put_end = cum_put[idx_end - 1] if idx_end > 0 else 0
        put_start = cum_put[idx_start - 1] if idx_start > 0 else 0
        result['put_tide'][i] = put_end - put_start
        
        # Spatial split if level provided
        if level_price is not None:
            mask_window = (np.arange(len(opt_ts)) >= idx_start) & (np.arange(len(opt_ts)) < idx_end)
            mask_above = mask_window & (strikes > level_price)
            mask_below = mask_window & (strikes <= level_price)
            mask_call = opt_types == 'C'
            mask_put = opt_types == 'P'
            
            result['call_tide_above'][i] = np.sum(premiums[mask_above & mask_call])
            result['call_tide_below'][i] = np.sum(premiums[mask_below & mask_call])
            result['put_tide_above'][i] = np.sum(premiums[mask_above & mask_put])
            result['put_tide_below'][i] = np.sum(premiums[mask_below & mask_put])
    
    return result

