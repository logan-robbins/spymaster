"""
Shared wall detection functions for futures and options.

Walls represent significant resting liquidity that price must absorb to break through.

Futures Walls: Large resting orders in the MBP-10 order book
Options Walls: Strikes where dealers are net long gamma (hedging creates mean-reversion)

Both can be computed globally (market-wide) or relative to a specific level.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.common.event_types import MBP10


# =============================================================================
# FUTURES WALLS (from MBP-10 order book depth)
# =============================================================================

def compute_futures_walls_at_timestamps(
    signal_ts: np.ndarray,
    mbp10_snapshots: List[MBP10],
    level_price: Optional[float] = None,
    wall_threshold_pct: float = 2.0,  # Wall = size > 2x typical
) -> Dict[str, np.ndarray]:
    """
    Detect futures liquidity walls from MBP-10 order book.
    
    A "wall" is a price level with significantly larger resting size than typical.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        mbp10_snapshots: List of MBP-10 snapshots
        level_price: If set, compute walls relative to this level.
                    If None, compute global walls (largest in entire book).
        wall_threshold_pct: Multiplier to define "wall" (default 2x typical size)
    
    Returns:
        Dictionary of wall features
    """
    n_signals = len(signal_ts)
    
    if level_price is not None:
        # Level-relative features
        result = {
            'futures_wall_above_price': np.full(n_signals, np.nan, dtype=np.float64),
            'futures_wall_above_size': np.zeros(n_signals, dtype=np.float64),
            'futures_wall_below_price': np.full(n_signals, np.nan, dtype=np.float64),
            'futures_wall_below_size': np.zeros(n_signals, dtype=np.float64),
            'futures_wall_above_dist': np.full(n_signals, np.nan, dtype=np.float64),
            'futures_wall_below_dist': np.full(n_signals, np.nan, dtype=np.float64),
            'futures_wall_asymmetry': np.zeros(n_signals, dtype=np.float64),  # above - below / total
        }
    else:
        # Global features
        result = {
            'futures_bid_wall_price': np.full(n_signals, np.nan, dtype=np.float64),
            'futures_bid_wall_size': np.zeros(n_signals, dtype=np.float64),
            'futures_ask_wall_price': np.full(n_signals, np.nan, dtype=np.float64),
            'futures_ask_wall_size': np.zeros(n_signals, dtype=np.float64),
            'futures_bid_wall_dist': np.full(n_signals, np.nan, dtype=np.float64),  # From mid
            'futures_ask_wall_dist': np.full(n_signals, np.nan, dtype=np.float64),
            'futures_total_bid_depth': np.zeros(n_signals, dtype=np.float64),
            'futures_total_ask_depth': np.zeros(n_signals, dtype=np.float64),
        }
    
    if n_signals == 0 or not mbp10_snapshots:
        return result
    
    # Sort snapshots and build lookup
    sorted_snapshots = sorted(mbp10_snapshots, key=lambda x: x.ts_event_ns)
    mbp_ts = np.array([x.ts_event_ns for x in sorted_snapshots], dtype=np.int64)
    
    # Find nearest MBP snapshot for each signal
    idx_lookup = np.searchsorted(mbp_ts, signal_ts, side='right') - 1
    idx_lookup = np.clip(idx_lookup, 0, len(mbp_ts) - 1)
    
    for i, idx in enumerate(idx_lookup):
        mbp = sorted_snapshots[idx]
        
        if not mbp.levels:
            continue
        
        # Extract all bid/ask levels
        bid_prices = []
        bid_sizes = []
        ask_prices = []
        ask_sizes = []
        
        for level in mbp.levels:
            if level.bid_px > 0 and level.bid_sz > 0:
                bid_prices.append(level.bid_px)
                bid_sizes.append(level.bid_sz)
            if level.ask_px > 0 and level.ask_sz > 0:
                ask_prices.append(level.ask_px)
                ask_sizes.append(level.ask_sz)
        
        if not bid_prices or not ask_prices:
            continue
        
        bid_prices = np.array(bid_prices)
        bid_sizes = np.array(bid_sizes)
        ask_prices = np.array(ask_prices)
        ask_sizes = np.array(ask_sizes)
        
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        if level_price is not None:
            # Level-relative: find largest wall above and below level
            
            # Walls above level (asks above, or bids above if price moved down)
            above_mask_ask = ask_prices > level_price
            above_mask_bid = bid_prices > level_price
            
            # Walls below level
            below_mask_ask = ask_prices < level_price
            below_mask_bid = bid_prices < level_price
            
            # Find largest above (combine both sides)
            above_prices = np.concatenate([
                ask_prices[above_mask_ask] if above_mask_ask.any() else [],
                bid_prices[above_mask_bid] if above_mask_bid.any() else []
            ])
            above_sizes = np.concatenate([
                ask_sizes[above_mask_ask] if above_mask_ask.any() else [],
                bid_sizes[above_mask_bid] if above_mask_bid.any() else []
            ])
            
            below_prices = np.concatenate([
                ask_prices[below_mask_ask] if below_mask_ask.any() else [],
                bid_prices[below_mask_bid] if below_mask_bid.any() else []
            ])
            below_sizes = np.concatenate([
                ask_sizes[below_mask_ask] if below_mask_ask.any() else [],
                bid_sizes[below_mask_bid] if below_mask_bid.any() else []
            ])
            
            if len(above_sizes) > 0:
                max_above_idx = np.argmax(above_sizes)
                result['futures_wall_above_price'][i] = above_prices[max_above_idx]
                result['futures_wall_above_size'][i] = above_sizes[max_above_idx]
                result['futures_wall_above_dist'][i] = above_prices[max_above_idx] - level_price
            
            if len(below_sizes) > 0:
                max_below_idx = np.argmax(below_sizes)
                result['futures_wall_below_price'][i] = below_prices[max_below_idx]
                result['futures_wall_below_size'][i] = below_sizes[max_below_idx]
                result['futures_wall_below_dist'][i] = level_price - below_prices[max_below_idx]
            
            # Asymmetry: positive = more liquidity above (resistance), negative = more below (support)
            above_total = above_sizes.sum() if len(above_sizes) > 0 else 0
            below_total = below_sizes.sum() if len(below_sizes) > 0 else 0
            total = above_total + below_total
            if total > 0:
                result['futures_wall_asymmetry'][i] = (above_total - below_total) / total
        
        else:
            # Global: find largest bid wall and ask wall
            max_bid_idx = np.argmax(bid_sizes)
            max_ask_idx = np.argmax(ask_sizes)
            
            result['futures_bid_wall_price'][i] = bid_prices[max_bid_idx]
            result['futures_bid_wall_size'][i] = bid_sizes[max_bid_idx]
            result['futures_ask_wall_price'][i] = ask_prices[max_ask_idx]
            result['futures_ask_wall_size'][i] = ask_sizes[max_ask_idx]
            
            # Distance from mid
            result['futures_bid_wall_dist'][i] = mid_price - bid_prices[max_bid_idx]
            result['futures_ask_wall_dist'][i] = ask_prices[max_ask_idx] - mid_price
            
            # Total depth
            result['futures_total_bid_depth'][i] = bid_sizes.sum()
            result['futures_total_ask_depth'][i] = ask_sizes.sum()
    
    return result


# =============================================================================
# OPTIONS WALLS (from dealer gamma positioning)
# =============================================================================

def compute_options_walls_at_timestamps(
    signal_ts: np.ndarray,
    option_trades_df: pd.DataFrame,
    spot_prices: np.ndarray,
    level_price: Optional[float] = None,
    lookback_ns: int = int(300 * 1e9),  # 5 minutes
) -> Dict[str, np.ndarray]:
    """
    Detect options walls from dealer gamma positioning.
    
    Uses the same formula as generate_levels.compute_wall_series:
        dealer_flow = -aggressor * size * gamma * 100
    
    Wall = strike with most NEGATIVE dealer flow (dealer is short gamma there,
    meaning they must hedge aggressively → price tends to pin).
    
    Note: option_trades_df should come from context AFTER InitMarketStateStage,
    which adds the 'gamma' column via compute_greeks_for_dataframe.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        option_trades_df: Options trades with strike, gamma, size, aggressor, right
                         (gamma added by InitMarketStateStage)
        spot_prices: Array of spot prices at each signal
        level_price: If set, compute walls relative to this level.
                    If None, compute global walls (call wall, put wall).
        lookback_ns: Window for accumulating dealer flow (default 5 min)
    
    Returns:
        Dictionary of wall features
    """
    n_signals = len(signal_ts)
    
    if level_price is not None:
        # Level-relative features
        result = {
            'options_wall_above_price': np.full(n_signals, np.nan, dtype=np.float64),
            'options_wall_above_flow': np.zeros(n_signals, dtype=np.float64),
            'options_wall_below_price': np.full(n_signals, np.nan, dtype=np.float64),
            'options_wall_below_flow': np.zeros(n_signals, dtype=np.float64),
            'options_wall_above_dist': np.full(n_signals, np.nan, dtype=np.float64),
            'options_wall_below_dist': np.full(n_signals, np.nan, dtype=np.float64),
        }
    else:
        # Global features
        result = {
            'options_call_wall_price': np.full(n_signals, np.nan, dtype=np.float64),
            'options_call_wall_flow': np.zeros(n_signals, dtype=np.float64),
            'options_put_wall_price': np.full(n_signals, np.nan, dtype=np.float64),
            'options_put_wall_flow': np.zeros(n_signals, dtype=np.float64),
            'options_call_wall_dist': np.full(n_signals, np.nan, dtype=np.float64),
            'options_put_wall_dist': np.full(n_signals, np.nan, dtype=np.float64),
        }
    
    if n_signals == 0 or option_trades_df is None or option_trades_df.empty:
        return result
    
    # Validate required columns - gamma should exist from InitMarketStateStage
    required = ['ts_event_ns', 'strike', 'size', 'aggressor', 'right', 'gamma']
    if not all(c in option_trades_df.columns for c in required):
        # Missing gamma means InitMarketStateStage hasn't run - return empty
        return result
    
    # Use the same formula as generate_levels.compute_wall_series
    df = option_trades_df.copy()
    df['ts_event_ns'] = pd.to_numeric(df['ts_event_ns'], errors='coerce').fillna(0).astype(np.int64)
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce').fillna(0).astype(np.float64)
    df['size'] = pd.to_numeric(df['size'], errors='coerce').fillna(0).astype(np.float64)
    df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce').fillna(0).astype(np.float64)
    df['aggressor'] = pd.to_numeric(df['aggressor'], errors='coerce').fillna(0).astype(np.int8)
    
    # Dealer flow = -aggressor * size * gamma * 100
    # Negative flow = dealer is short gamma at that strike (wall/pinning effect)
    df['dealer_flow'] = -df['aggressor'] * df['size'] * df['gamma'] * 100
    
    df = df.sort_values('ts_event_ns')
    opt_ts = df['ts_event_ns'].values
    strikes = df['strike'].values
    rights = df['right'].values.astype(str)
    dealer_flow = df['dealer_flow'].values
    
    for i in range(n_signals):
        ts = signal_ts[i]
        start_ts = ts - lookback_ns
        spot = spot_prices[i] if i < len(spot_prices) else np.nan
        
        # Get trades in lookback window
        mask = (opt_ts >= start_ts) & (opt_ts <= ts)
        if not mask.any():
            continue
        
        window_strikes = strikes[mask]
        window_rights = rights[mask]
        window_flow = dealer_flow[mask]
        
        # Aggregate flow by strike
        unique_strikes = np.unique(window_strikes)
        
        if level_price is not None:
            # Level-relative: find strongest wall (most negative flow) above and below
            above_strikes = unique_strikes[unique_strikes > level_price]
            below_strikes = unique_strikes[unique_strikes <= level_price]
            
            # Wall above level: strike with most negative dealer flow
            if len(above_strikes) > 0:
                above_flow = np.array([
                    window_flow[window_strikes == s].sum() for s in above_strikes
                ])
                # Most negative = strongest wall
                min_idx = np.argmin(above_flow)
                result['options_wall_above_price'][i] = above_strikes[min_idx]
                result['options_wall_above_flow'][i] = above_flow[min_idx]
                result['options_wall_above_dist'][i] = above_strikes[min_idx] - level_price
            
            # Wall below level
            if len(below_strikes) > 0:
                below_flow = np.array([
                    window_flow[window_strikes == s].sum() for s in below_strikes
                ])
                min_idx = np.argmin(below_flow)
                result['options_wall_below_price'][i] = below_strikes[min_idx]
                result['options_wall_below_flow'][i] = below_flow[min_idx]
                result['options_wall_below_dist'][i] = level_price - below_strikes[min_idx]
        
        else:
            # Global: find call wall and put wall separately
            call_mask = window_rights == 'C'
            put_mask = window_rights == 'P'
            
            # Call wall: strike with most negative dealer flow on calls
            call_strikes_unique = np.unique(window_strikes[call_mask])
            if len(call_strikes_unique) > 0:
                call_flow = np.array([
                    window_flow[(window_strikes == s) & call_mask].sum()
                    for s in call_strikes_unique
                ])
                min_idx = np.argmin(call_flow)
                result['options_call_wall_price'][i] = call_strikes_unique[min_idx]
                result['options_call_wall_flow'][i] = call_flow[min_idx]
                if not np.isnan(spot):
                    result['options_call_wall_dist'][i] = call_strikes_unique[min_idx] - spot
            
            # Put wall: strike with most negative dealer flow on puts
            put_strikes_unique = np.unique(window_strikes[put_mask])
            if len(put_strikes_unique) > 0:
                put_flow = np.array([
                    window_flow[(window_strikes == s) & put_mask].sum()
                    for s in put_strikes_unique
                ])
                min_idx = np.argmin(put_flow)
                result['options_put_wall_price'][i] = put_strikes_unique[min_idx]
                result['options_put_wall_flow'][i] = put_flow[min_idx]
                if not np.isnan(spot):
                    result['options_put_wall_dist'][i] = spot - put_strikes_unique[min_idx]
    
    return result

