"""
Physics Engines - Apple M4 Silicon Optimized
"""

from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import jit

from src.common.config import CONFIG
from src.common.event_types import Aggressor, FuturesTrade, MBP10


# =============================================================================
# VECTORIZED DATA STRUCTURES
# =============================================================================

@dataclass
class MarketData:
    """
    Efficient struct-of-arrays market data for engine processing.
    """
    # ES Trades (Sorted by time)
    trade_ts_ns: np.ndarray          # int64, shape (N,)
    trade_prices: np.ndarray         # float64, shape (N,)
    trade_sizes: np.ndarray          # int32, shape (N,)
    trade_aggressors: np.ndarray     # int8, shape (N,)

    # ES MBP-10 Snapshots (Sorted by time)
    mbp_ts_ns: np.ndarray            # int64, shape (n,)
    mbp_bid_prices: np.ndarray       # float64, shape (n, 10)
    mbp_bid_sizes: np.ndarray        # int64, shape (n, 10)
    mbp_ask_prices: np.ndarray       # float64, shape (n, 10)
    mbp_ask_sizes: np.ndarray        # int64, shape (n, 10)

    strike_gamma: Dict[float, float]
    
    # Efficient Option Trade Arrays
    opt_ts_ns: np.ndarray            # int64
    opt_strikes: np.ndarray          # float64
    opt_is_call: np.ndarray          # bool (True=Call, False=Put)
    opt_premium: np.ndarray          # float64 (signed)
    opt_net_gamma: np.ndarray        # float64 (signed)




def build_market_data(
    trades: List[FuturesTrade],
    mbp10_snapshots: List[MBP10],
    option_flows: Dict[Tuple[float, str, str], Any],
    date: str = None,
    option_trades_df: pd.DataFrame = None
) -> MarketData:
    """
    Build efficient arrays from raw objects.
    
    Args:
        trades: List of futures trades
        mbp10_snapshots: List of MBP-10 snapshots
        option_flows: Dictionary of aggregated option flows (Backup)
        date: Expiration date string
        option_trades_df: DataFrame of option trades with greeks/aggressor
    """
    # Convert trades to numpy arrays
    n_trades = len(trades)
    trade_ts_ns = np.zeros(n_trades, dtype=np.int64)
    trade_prices = np.zeros(n_trades, dtype=np.float64)
    trade_sizes = np.zeros(n_trades, dtype=np.int32)
    trade_aggressors = np.zeros(n_trades, dtype=np.int8)

    for i, t in enumerate(trades):
        trade_ts_ns[i] = t.ts_event_ns
        trade_prices[i] = t.price
        trade_sizes[i] = t.size
        agg_val = 0
        if t.aggressor == Aggressor.BUY:
            agg_val = 1
        elif t.aggressor == Aggressor.SELL:
            agg_val = -1
        trade_aggressors[i] = agg_val

    # Sort by timestamp
    sort_idx = np.argsort(trade_ts_ns)
    trade_ts_ns = trade_ts_ns[sort_idx]
    trade_prices = trade_prices[sort_idx]
    trade_sizes = trade_sizes[sort_idx]
    trade_aggressors = trade_aggressors[sort_idx]

    # Convert MBP-10 to numpy arrays
    n_mbp = len(mbp10_snapshots)
    mbp_ts_ns = np.zeros(n_mbp, dtype=np.int64)
    mbp_bid_prices = np.zeros((n_mbp, 10), dtype=np.float64)
    mbp_bid_sizes = np.zeros((n_mbp, 10), dtype=np.int64)
    mbp_ask_prices = np.zeros((n_mbp, 10), dtype=np.float64)
    mbp_ask_sizes = np.zeros((n_mbp, 10), dtype=np.int64)

    for i, m in enumerate(mbp10_snapshots):
        mbp_ts_ns[i] = m.ts_event_ns
        for j, level in enumerate(m.levels):
            if j >= 10: break
            mbp_bid_prices[i, j] = level.bid_px
            mbp_bid_sizes[i, j] = level.bid_sz
            mbp_ask_prices[i, j] = level.ask_px
            mbp_ask_sizes[i, j] = level.ask_sz

    # Sort by timestamp
    sort_idx = np.argsort(mbp_ts_ns)
    mbp_ts_ns = mbp_ts_ns[sort_idx]
    mbp_bid_prices = mbp_bid_prices[sort_idx]
    mbp_bid_sizes = mbp_bid_sizes[sort_idx]
    mbp_ask_prices = mbp_ask_prices[sort_idx]
    mbp_ask_sizes = mbp_ask_sizes[sort_idx]

    # Pre-aggregate option flows by strike
    strike_gamma = defaultdict(float)
    
    for (strike, right, exp_date), flow in option_flows.items():
        if date is None or exp_date == date:
            strike_gamma[strike] += flow.net_gamma_flow

    # Process Option Trades DataFrame into Arrays
    if option_trades_df is not None and not option_trades_df.empty:
        # Ensure sorted by time
        if not option_trades_df['ts_event_ns'].is_monotonic_increasing:
             option_trades_df = option_trades_df.sort_values('ts_event_ns')
             
        opt_ts_ns = option_trades_df['ts_event_ns'].values.astype(np.int64)
        opt_strikes = option_trades_df['strike'].values.astype(np.float64)
        opt_is_call = (option_trades_df['right'] == 'C').values
        
        aggressors = option_trades_df['aggressor'].values.astype(float) # -1, 0, 1
        sizes = option_trades_df['size'].values.astype(float)
        prices = option_trades_df['price'].values.astype(float)
        gammas = option_trades_df['gamma'].fillna(0.0).values.astype(float)
        
        opt_premium = prices * sizes * 100.0 * aggressors
        
        opt_net_gamma = gammas * sizes * 100.0 * (-aggressors)
        
    else:
        opt_ts_ns = np.array([], dtype=np.int64)
        opt_strikes = np.array([], dtype=np.float64)
        opt_is_call = np.array([], dtype=bool)
        opt_premium = np.array([], dtype=np.float64)
        opt_net_gamma = np.array([], dtype=np.float64)

    return MarketData(
        trade_ts_ns=trade_ts_ns,
        trade_prices=trade_prices,
        trade_sizes=trade_sizes,
        trade_aggressors=trade_aggressors,
        mbp_ts_ns=mbp_ts_ns,
        mbp_bid_prices=mbp_bid_prices,
        mbp_bid_sizes=mbp_bid_sizes,
        mbp_ask_prices=mbp_ask_prices,
        mbp_ask_sizes=mbp_ask_sizes,
        strike_gamma=dict(strike_gamma),
        opt_ts_ns=opt_ts_ns,
        opt_strikes=opt_strikes,
        opt_is_call=opt_is_call,
        opt_premium=opt_premium,
        opt_net_gamma=opt_net_gamma
    )


# =============================================================================
# VECTORIZED TAPE ENGINE
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_tape_metrics_numba(
    touch_ts_ns: np.ndarray,
    level_prices_es: np.ndarray,
    trade_ts_ns: np.ndarray,
    trade_prices: np.ndarray,
    trade_sizes: np.ndarray,
    trade_aggressors: np.ndarray,
    window_ns: int,
    band_es: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated tape metrics computation.

    For each touch, compute imbalance, buy_vol, sell_vol, velocity.

    NOTE: Touch timestamps are at the START of minute bars, so we look
    FORWARD in time to analyze trades that happened during that bar.
    Window is [ts, ts + window_ns] instead of [ts - window_ns, ts].
    """
    n_touches = len(touch_ts_ns)
    n_trades = len(trade_ts_ns)

    imbalances = np.zeros(n_touches, dtype=np.float64)
    buy_vols = np.zeros(n_touches, dtype=np.int64)
    sell_vols = np.zeros(n_touches, dtype=np.int64)
    velocities = np.zeros(n_touches, dtype=np.float64)

    for i in range(n_touches):
        ts = touch_ts_ns[i]
        level = level_prices_es[i]

        # Find trades in FORWARD window [ts, ts + window_ns]
        # Touch timestamp is at bar start, trades happen during the bar
        end_ts = ts + window_ns

        # Binary search for start index (first trade >= ts)
        left = 0
        right = n_trades
        while left < right:
            mid = (left + right) // 2
            if trade_ts_ns[mid] < ts:
                left = mid + 1
            else:
                right = mid
        start_idx = left

        # Binary search for end index (first trade > end_ts)
        left = start_idx
        right = n_trades
        while left < right:
            mid = (left + right) // 2
            if trade_ts_ns[mid] <= end_ts:
                left = mid + 1
            else:
                right = mid
        end_idx = left

        if end_idx <= start_idx:
            continue

        # Filter by price band and compute volumes
        buy_vol = 0
        sell_vol = 0

        for j in range(start_idx, end_idx):
            price = trade_prices[j]
            if abs(price - level) <= band_es:
                if trade_aggressors[j] == 1:
                    buy_vol += trade_sizes[j]
                elif trade_aggressors[j] == -1:
                    sell_vol += trade_sizes[j]

        buy_vols[i] = buy_vol
        sell_vols[i] = sell_vol

        total = buy_vol + sell_vol
        if total > 0:
            imbalances[i] = (buy_vol - sell_vol) / float(total)

        # Compute velocity using trades within price band (consistent with buy/sell volumes)
        # This measures flow velocity AT THE LEVEL, not global market momentum
        # Count trades in band
        n_in_band = 0
        for j in range(start_idx, end_idx):
            if abs(trade_prices[j] - level) <= band_es:
                n_in_band += 1
        
        if n_in_band >= 2:
            # Compute means
            t_sum = 0.0
            p_sum = 0.0
            for j in range(start_idx, end_idx):
                if abs(trade_prices[j] - level) <= band_es:
                    t_sum += trade_ts_ns[j] / 1e9
                    p_sum += trade_prices[j]
            t_mean = t_sum / n_in_band
            p_mean = p_sum / n_in_band

            # Compute slope
            numerator = 0.0
            denominator = 0.0
            for j in range(start_idx, end_idx):
                if abs(trade_prices[j] - level) <= band_es:
                    dt = trade_ts_ns[j] / 1e9 - t_mean
                    dp = trade_prices[j] - p_mean
                    numerator += dt * dp
                    denominator += dt * dt

            if denominator > 1e-10:
                velocities[i] = numerator / denominator

    return imbalances, buy_vols, sell_vols, velocities


@jit(nopython=True, cache=True)
def _compute_tape_metrics_spatial_numba(
    touch_ts_ns: np.ndarray,
    level_prices_es: np.ndarray,
    trade_ts_ns: np.ndarray,
    trade_prices: np.ndarray,
    trade_sizes: np.ndarray,
    trade_aggressors: np.ndarray,
    window_ns: int,
    band_es: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatial tape metrics.
    
    Returns:
        (buy_vol_atm, sell_vol_atm, imbalance_atm,
         buy_vol_above, sell_vol_above, imbalance_above,
         buy_vol_below, sell_vol_below, imbalance_below)
         
    Above = Aggregation of L+1.0 and L+2.0
    Below = Aggregation of L-1.0 and L-2.0
    """
    n_touches = len(touch_ts_ns)
    n_trades = len(trade_ts_ns)

    # Output arrays
    # ATM
    bv_atm = np.zeros(n_touches, dtype=np.int64)
    sv_atm = np.zeros(n_touches, dtype=np.int64)
    imb_atm = np.zeros(n_touches, dtype=np.float64)
    
    # Above (L+1 + L+2)
    bv_above = np.zeros(n_touches, dtype=np.int64)
    sv_above = np.zeros(n_touches, dtype=np.int64)
    imb_above = np.zeros(n_touches, dtype=np.float64)
    
    # Below (L-1 + L-2)
    bv_below = np.zeros(n_touches, dtype=np.int64)
    sv_below = np.zeros(n_touches, dtype=np.int64)
    imb_below = np.zeros(n_touches, dtype=np.float64)

    for i in range(n_touches):
        ts = touch_ts_ns[i]
        level = level_prices_es[i]
        end_ts = ts + window_ns

        # Search window (same as before)
        start_idx = np.searchsorted(trade_ts_ns, ts)
        end_idx = np.searchsorted(trade_ts_ns, end_ts)

        if end_idx <= start_idx:
            continue
            
        # Accumulators
        b_atm, s_atm = 0, 0
        b_ab, s_ab = 0, 0
        b_be, s_be = 0, 0
        
        for j in range(start_idx, end_idx):
            price = trade_prices[j]
            size = trade_sizes[j]
            agg = trade_aggressors[j] # 1 or -1
            
            if agg == 0: continue
            
            diff = price - level
            
            # Check bands (Strict levels)
            # ATM
            if abs(diff) <= band_es:
                if agg == 1: b_atm += size
                else: s_atm += size
            
            # Above: L+1 OR L+2
            # Check L+1
            elif abs(diff - 1.0) <= band_es or abs(diff - 2.0) <= band_es:
                if agg == 1: b_ab += size
                else: s_ab += size
                
            # Below: L-1 OR L-2
            elif abs(diff + 1.0) <= band_es or abs(diff + 2.0) <= band_es:
                if agg == 1: b_be += size
                else: s_be += size

        # Store Volumes
        bv_atm[i] = b_atm
        sv_atm[i] = s_atm
        bv_above[i] = b_ab
        sv_above[i] = s_ab
        bv_below[i] = b_be
        sv_below[i] = s_be
        
        # Calc Imbalances
        tot_atm = b_atm + s_atm
        if tot_atm > 0: imb_atm[i] = (b_atm - s_atm) / tot_atm
        
        tot_ab = b_ab + s_ab
        if tot_ab > 0: imb_above[i] = (b_ab - s_ab) / tot_ab
        
        tot_be = b_be + s_be
        if tot_be > 0: imb_below[i] = (b_be - s_be) / tot_be

    return (bv_atm, sv_atm, imb_atm,
            bv_above, sv_above, imb_above,
            bv_below, sv_below, imb_below)


@jit(nopython=True, cache=True)
def _compute_fuel_metrics_numba(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,
    opt_ts_ns: np.ndarray,
    opt_strikes: np.ndarray,
    opt_premium: np.ndarray,
    opt_is_call: np.ndarray,
    strike_gamma_strikes: np.ndarray,
    strike_gamma_values: np.ndarray,
    window_ns: int,
    strike_range: float,
    split_range: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated fuel metrics computation."""
    n = len(touch_ts_ns)
    n_opts = len(opt_ts_ns)
    
    gamma_exposure = np.zeros(n, dtype=np.float64)
    fuel_effect = np.zeros(n, dtype=np.int32)  # -1=AMPLIFY, 0=NEUTRAL, 1=DAMPEN
    call_tide = np.zeros(n, dtype=np.float64)
    put_tide = np.zeros(n, dtype=np.float64)
    call_tide_above = np.zeros(n, dtype=np.float64)
    call_tide_below = np.zeros(n, dtype=np.float64)
    put_tide_above = np.zeros(n, dtype=np.float64)
    put_tide_below = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        ts = touch_ts_ns[i]
        level = level_prices[i]
        
        # Time window [ts - 30s, ts]
        start_ts = ts - window_ns
        
        # Binary search for window
        start_idx = np.searchsorted(opt_ts_ns, start_ts)
        end_idx = np.searchsorted(opt_ts_ns, ts)
        
        # Tide calculations
        for j in range(start_idx, end_idx):
            strike = opt_strikes[j]
            prem = opt_premium[j]
            is_call = opt_is_call[j]
            
            dist = abs(strike - level)
            
            # Total tide
            if dist <= strike_range:
                if is_call == 1:
                    call_tide[i] += prem
                else:
                    put_tide[i] += prem
            
            # Split tide
            if strike > level and strike <= level + split_range:
                if is_call == 1:
                    call_tide_above[i] += prem
                else:
                    put_tide_above[i] += prem
            elif strike < level and strike >= level - split_range:
                if is_call == 1:
                    call_tide_below[i] += prem
                else:
                    put_tide_below[i] += prem
        
        # Gamma exposure
        g_val = 0.0
        for j in range(len(strike_gamma_strikes)):
            if abs(strike_gamma_strikes[j] - level) <= strike_range:
                g_val += strike_gamma_values[j]
        gamma_exposure[i] = g_val
        
        # Fuel effect
        if g_val < -100000:
            fuel_effect[i] = -1  # AMPLIFY
        elif g_val > 100000:
            fuel_effect[i] = 1   # DAMPEN
        else:
            fuel_effect[i] = 0   # NEUTRAL
    
    return gamma_exposure, fuel_effect, call_tide, put_tide, call_tide_above, call_tide_below, put_tide_above, put_tide_below


@jit(nopython=True, cache=True)
def _compute_fuel_metrics_spatial_numba(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,
    opt_ts_ns: np.ndarray,
    opt_strikes: np.ndarray,
    opt_premium: np.ndarray,
    opt_is_call: np.ndarray,
    window_ns: int,
    strike_range: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated SPATIAL fuel metrics.
    
    Returns:
        (call_tide_atm, put_tide_atm,
         call_tide_above, put_tide_above,
         call_tide_below, put_tide_below)
    """
    n = len(touch_ts_ns)
    
    # ATM
    ct_atm = np.zeros(n, dtype=np.float64)
    pt_atm = np.zeros(n, dtype=np.float64)

    
    ct_above = np.zeros(n, dtype=np.float64)
    pt_above = np.zeros(n, dtype=np.float64)
    ct_below = np.zeros(n, dtype=np.float64)
    pt_below = np.zeros(n, dtype=np.float64)
    
    match_tol = 0.5 
    
    for i in range(n):
        ts = touch_ts_ns[i]
        level = level_prices[i]
        start_ts = ts - window_ns
        
        start_idx = np.searchsorted(opt_ts_ns, start_ts)
        end_idx = np.searchsorted(opt_ts_ns, ts)
        
        for j in range(start_idx, end_idx):
            strike = opt_strikes[j]
            prem = opt_premium[j]
            is_call = opt_is_call[j] # 1=True
            
            diff = strike - level
            
            # ATM
            if abs(diff) <= match_tol:
                if is_call == 1: ct_atm[i] += prem
                else: pt_atm[i] += prem
            
            # Above (L+1, L+2)
            elif abs(diff - 1.0) <= match_tol or abs(diff - 2.0) <= match_tol:
                if is_call == 1: ct_above[i] += prem
                else: pt_above[i] += prem
            
            # Below (L-1, L-2)
            elif abs(diff + 1.0) <= match_tol or abs(diff + 2.0) <= match_tol:
                if is_call == 1: ct_below[i] += prem
                else: pt_below[i] += prem
                
    return ct_atm, pt_atm, ct_above, pt_above, ct_below, pt_below


def compute_fuel_metrics(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,
    market_data: MarketData,
    strike_range: float = 100.0,
    split_range: float = 25.0
) -> Dict[str, np.ndarray]:
    """
    Compute Fuel Engine metrics (Market Tide) for all touches.
    Numba JIT compiled.
    """
    n = len(touch_ts_ns)
    
    # Extract option data
    opt_ts = market_data.opt_ts_ns
    opt_strikes = market_data.opt_strikes
    opt_premium = market_data.opt_premium
    opt_is_call = market_data.opt_is_call
    strike_gamma = market_data.strike_gamma
    
    # Handle empty data
    if len(opt_ts) == 0 or n == 0:
        return {
            'gamma_exposure': np.zeros(n, dtype=np.float64),
            'fuel_effect': np.full(n, 'NEUTRAL', dtype=object),
            'call_tide': np.zeros(n, dtype=np.float64),
            'put_tide': np.zeros(n, dtype=np.float64),
            'call_tide_above_5pt': np.zeros(n, dtype=np.float64),
            'call_tide_below_5pt': np.zeros(n, dtype=np.float64),
            'put_tide_above_5pt': np.zeros(n, dtype=np.float64),
            'put_tide_below_5pt': np.zeros(n, dtype=np.float64)
        }
    
    # Convert strike_gamma dict to arrays for numba
    sg_strikes = np.array(list(strike_gamma.keys()), dtype=np.float64)
    sg_values = np.array(list(strike_gamma.values()), dtype=np.float64)
    
    window_ns = 30_000_000_000  # 30 seconds
    
    gamma_exp, fuel_codes, call_t, put_t, call_t_above, call_t_below, put_t_above, put_t_below = _compute_fuel_metrics_numba(
        touch_ts_ns.astype(np.int64),
        level_prices.astype(np.float64),
        opt_ts,
        opt_strikes,
        opt_premium,
        opt_is_call,
        sg_strikes,
        sg_values,
        window_ns,
        strike_range,
        split_range
    )
    
    # Decode fuel effect
    effect_map = {-1: 'AMPLIFY', 0: 'NEUTRAL', 1: 'DAMPEN'}
    fuel_effect = np.array([effect_map[code] for code in fuel_codes], dtype=object)
    


    # Add Spatial Metrics
    ct_atm, pt_atm, ct_above, pt_above, ct_below, pt_below = _compute_fuel_metrics_spatial_numba(
         touch_ts_ns.astype(np.int64),
         level_prices.astype(np.float64),
         opt_ts,
         opt_strikes,
         opt_premium,
         opt_is_call,
         window_ns,
         strike_range
    )
        
        
    return {
        'gamma_exposure': gamma_exp,
        'fuel_effect': fuel_effect,
        'call_tide_atm': ct_atm, # Use ATM specific
        'put_tide_atm': pt_atm,  # Use ATM specific
        'call_tide_above': ct_above,
        'put_tide_above': pt_above,
        'call_tide_below': ct_below,
        'put_tide_below': pt_below,
    }


def compute_tape_metrics(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,  # ES prices
    market_data: MarketData,
    window_seconds: float = 5.0,
    band_dollars: float = 0.10
) -> Dict[str, np.ndarray]:
    """
    Compute tape metrics for all touches.

    Args:
        touch_ts_ns: Touch timestamps
        level_prices: Level prices (ES points)
        market_data:  market data
        window_seconds: Lookback window
        band_dollars: Price band (ES points)

    Returns:
        Dict with arrays: imbalance, buy_vol, sell_vol, velocity
    """
    # Use ES prices directly
    level_prices_es = level_prices.astype(np.float64)
    band_es = float(band_dollars)
    window_ns = int(window_seconds * 1e9)

    imbalances, buy_vols, sell_vols, velocities = _compute_tape_metrics_numba(
        touch_ts_ns.astype(np.int64),
        level_prices_es.astype(np.float64),
        market_data.trade_ts_ns,
        market_data.trade_prices,
        market_data.trade_sizes,
        market_data.trade_aggressors,
        window_ns,
        band_es
    )


    # Add Spatial Metrics
    (bv_atm, sv_atm, imb_atm,
     bv_above, sv_above, imb_above,
     bv_below, sv_below, imb_below) = _compute_tape_metrics_spatial_numba(
        touch_ts_ns.astype(np.int64),
        level_prices_es.astype(np.float64),
        market_data.trade_ts_ns,
        market_data.trade_prices,
        market_data.trade_sizes,
        market_data.trade_aggressors,
        window_ns,
        band_es
    )
        
        
    return {
        'tape_imbalance': imbalances, # Default (maybe slightly differ if band logic varied?)
        'tape_buy_vol': buy_vols,
        'tape_sell_vol': sell_vols,
        'tape_velocity': velocities,
        # Spatial
        'tape_buy_vol_above': bv_above,
        'tape_sell_vol_above': sv_above,
        'tape_imbalance_above': imb_above,
        'tape_buy_vol_below': bv_below,
        'tape_sell_vol_below': sv_below,
        'tape_imbalance_below': imb_below
    }


# =============================================================================
# VECTORIZED BARRIER ENGINE
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_depth_in_zone_numba(
    bid_prices: np.ndarray,  # shape (10,)
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    zone_low: float,
    zone_high: float,
    side: int  # 1=bid, -1=ask
) -> int:
    """Compute total depth in zone from MBP-10 snapshot."""
    total = 0
    for i in range(10):
        if side == 1:  # bid
            if zone_low <= bid_prices[i] <= zone_high:
                total += bid_sizes[i]
        else:  # ask
            if zone_low <= ask_prices[i] <= zone_high:
                total += ask_sizes[i]
    return total


@jit(nopython=True, cache=True)
def _compute_depth_profile_numba(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,
    mbp_ts_ns: np.ndarray,
    mbp_bid_prices: np.ndarray,
    mbp_bid_sizes: np.ndarray,
    mbp_ask_prices: np.ndarray,
    mbp_ask_sizes: np.ndarray,
    window_ns: int,
    band_es: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute resting liquidity (Limit Orders) at ATM, Above, Below.
    Returns: (bid_atm, ask_atm, bid_above, ask_above, bid_below, ask_below)
    """
    n = len(touch_ts_ns)
    
    # ATM
    bid_atm = np.zeros(n, dtype=np.int64)
    ask_atm = np.zeros(n, dtype=np.int64)
    # Above
    bid_above = np.zeros(n, dtype=np.int64)
    ask_above = np.zeros(n, dtype=np.int64)
    # Below
    bid_below = np.zeros(n, dtype=np.int64)
    ask_below = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        ts = touch_ts_ns[i]
        level = level_prices[i]
        end_ts = ts + window_ns
        

        idx = np.searchsorted(mbp_ts_ns, ts)
        if idx >= len(mbp_ts_ns): idx = len(mbp_ts_ns) - 1
        
        # Analyze that snapshot (single point for now, efficient)
        b_px = mbp_bid_prices[idx]
        b_sz = mbp_bid_sizes[idx]
        a_px = mbp_ask_prices[idx]
        a_sz = mbp_ask_sizes[idx]
        
        # Iterate levels (10)
        for k in range(10):
            # Bids
            diff_b = b_px[k] - level
            if abs(diff_b) <= band_es: bid_atm[i] += b_sz[k]
            elif abs(diff_b - 1.0) <= band_es or abs(diff_b - 2.0) <= band_es: bid_above[i] += b_sz[k]
            elif abs(diff_b + 1.0) <= band_es or abs(diff_b + 2.0) <= band_es: bid_below[i] += b_sz[k]
            
            # Asks
            diff_a = a_px[k] - level
            if abs(diff_a) <= band_es: ask_atm[i] += a_sz[k]
            elif abs(diff_a - 1.0) <= band_es or abs(diff_a - 2.0) <= band_es: ask_above[i] += a_sz[k]
            elif abs(diff_a + 1.0) <= band_es or abs(diff_a + 2.0) <= band_es: ask_below[i] += a_sz[k]
            
    return bid_atm, ask_atm, bid_above, ask_above, bid_below, ask_below


@jit(nopython=True, cache=True)
def _compute_barrier_metrics_numba(
    touch_ts_ns: np.ndarray,
    level_prices_es: np.ndarray,
    directions: np.ndarray,
    mbp_ts_ns: np.ndarray,
    mbp_bid_prices: np.ndarray,  # shape (N_mbp, 10)
    mbp_bid_sizes: np.ndarray,
    mbp_ask_prices: np.ndarray,
    mbp_ask_sizes: np.ndarray,
    window_ns: int,
    zone_es: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba barrier metrics computation.
    
    Returns: (delta_liqs, depth_in_zones, wall_ratios, replenishment_ratios, barrier_states)
    """
    n = len(touch_ts_ns)
    n_mbp = len(mbp_ts_ns)
    
    delta_liqs = np.zeros(n, dtype=np.float64)
    depth_in_zones = np.zeros(n, dtype=np.int64)
    wall_ratios = np.zeros(n, dtype=np.float64)
    replenishment_ratios = np.zeros(n, dtype=np.float64)
    barrier_states = np.zeros(n, dtype=np.int32)  # Encode as int: 0=NEUTRAL, 1=WEAK, etc.
    
    # Pre-compute average depth across all MBP
    avg_depth = (mbp_bid_sizes.mean() + mbp_ask_sizes.mean()) / 2.0
    
    for i in range(n):
        ts = touch_ts_ns[i]
        level_es = level_prices_es[i]
        direction = directions[i]
        
        zone_low = level_es - zone_es
        zone_high = level_es + zone_es
        side = 1 if direction == -1 else -1  # 1=bid, -1=ask
        
        # Find window indices using binary search (faster than np.where)
        end_ts = ts + window_ns
        
        # Find first index >= ts
        start_idx = np.searchsorted(mbp_ts_ns, ts, side='left')
        # Find last index <= end_ts
        end_idx = np.searchsorted(mbp_ts_ns, end_ts, side='right')
        
        if end_idx - start_idx < 2:
            barrier_states[i] = 0  # NEUTRAL
            continue
        
        first_idx = start_idx
        last_idx = end_idx - 1
        
        # Compute depth at first and last
        depth_start = _compute_depth_in_zone_numba(
            mbp_bid_prices[first_idx], mbp_bid_sizes[first_idx],
            mbp_ask_prices[first_idx], mbp_ask_sizes[first_idx],
            zone_low, zone_high, side
        )
        depth_end = _compute_depth_in_zone_numba(
            mbp_bid_prices[last_idx], mbp_bid_sizes[last_idx],
            mbp_ask_prices[last_idx], mbp_ask_sizes[last_idx],
            zone_low, zone_high, side
        )
        
        delta_liq = depth_end - depth_start
        delta_liqs[i] = delta_liq
        depth_in_zones[i] = depth_end
        wall_ratios[i] = depth_end / (avg_depth + 1e-6)
        
        # Compute replenishment ratio
        added_size = 0.0
        canceled_size = 0.0
        
        for j in range(start_idx, end_idx - 1):
            depth_curr = _compute_depth_in_zone_numba(
                mbp_bid_prices[j], mbp_bid_sizes[j],
                mbp_ask_prices[j], mbp_ask_sizes[j],
                zone_low, zone_high, side
            )
            depth_next = _compute_depth_in_zone_numba(
                mbp_bid_prices[j+1], mbp_bid_sizes[j+1],
                mbp_ask_prices[j+1], mbp_ask_sizes[j+1],
                zone_low, zone_high, side
            )
            
            delta = depth_next - depth_curr
            if delta > 0:
                added_size += delta
            elif delta < 0:
                canceled_size += abs(delta)
        
        replenishment_ratios[i] = added_size / (canceled_size + 1e-6)
        
        # Classify state
        if delta_liq < -100:
            if wall_ratios[i] < 0.3:
                barrier_states[i] = -3  # VACUUM
            else:
                barrier_states[i] = -2  # CONSUMED
        elif delta_liq > 100:
            if wall_ratios[i] > 1.5:
                barrier_states[i] = 2  # WALL
            else:
                barrier_states[i] = -1  # ABSORPTION
        elif depth_end < 50:
            barrier_states[i] = 1  # WEAK
        else:
            barrier_states[i] = 0  # NEUTRAL
    
    return delta_liqs, depth_in_zones, wall_ratios, replenishment_ratios, barrier_states


def compute_barrier_metrics(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,  # ES prices (strike-aligned)
    directions: np.ndarray,    # 'UP' or 'DOWN' as int: 1 or -1
    market_data: MarketData,
    window_seconds: float = 10.0,
    zone_es_ticks: int = 2
) -> Dict[str, np.ndarray]:
    """
    Compute barrier metrics for all touches.

    Use ES points directly for level and MBP-10 depth alignment.

    Args:
        touch_ts_ns: Touch timestamps
        level_prices: Level prices (ES, strike-aligned)
        directions: Direction array (1=UP/resistance, -1=DOWN/support)
        market_data:  market data
        window_seconds: Forward window for MBP-10 analysis
        zone_es_ticks: Zone width in ES ticks (±N ticks around level)

    Returns:
        Dict with arrays: barrier_state, delta_liq, wall_ratio, etc.
    """
    n = len(touch_ts_ns)

    # Output arrays
    barrier_states = np.empty(n, dtype=object)
    delta_liqs = np.zeros(n, dtype=np.float64)
    wall_ratios = np.zeros(n, dtype=np.float64)
    depth_in_zones = np.zeros(n, dtype=np.int64)
    replenishment_ratios = np.zeros(n, dtype=np.float64)

    # ES tick size
    ES_TICK_SIZE = 0.25
    window_ns = int(window_seconds * 1e9)

    # Use ES level prices directly
    level_prices_es = level_prices.astype(np.float64)
    zone_es = zone_es_ticks * ES_TICK_SIZE  # e.g., ±2 ticks = ±$0.50 ES

    delta_liqs, depth_in_zones, wall_ratios, replenishment_ratios, barrier_state_codes = _compute_barrier_metrics_numba(
        touch_ts_ns.astype(np.int64),
        level_prices_es,
        directions.astype(np.int32),
        market_data.mbp_ts_ns,
        market_data.mbp_bid_prices,
        market_data.mbp_bid_sizes,
        market_data.mbp_ask_prices,
        market_data.mbp_ask_sizes,
        window_ns,
        zone_es
    )
    
    # Decode states from int to string
    state_map = {-3: 'VACUUM', -2: 'CONSUMED', -1: 'ABSORPTION', 0: 'NEUTRAL', 1: 'WEAK', 2: 'WALL'}
    barrier_states = np.array([state_map[code] for code in barrier_state_codes], dtype=object)
    
    # Add Spatial Depth (Limit Liquidity Profile)
    # Use zone_es as band (strictly inside zone)
    bid_atm, ask_atm, bid_above, ask_above, bid_below, ask_below = _compute_depth_profile_numba(
        touch_ts_ns.astype(np.int64),
        level_prices_es,
        market_data.mbp_ts_ns,
        market_data.mbp_bid_prices,
        market_data.mbp_bid_sizes,
        market_data.mbp_ask_prices,
        market_data.mbp_ask_sizes,
        window_ns,
        zone_es
    )
        
    return {
        'barrier_state': barrier_states,
        'barrier_delta_liq': delta_liqs,
        'barrier_replenishment_ratio': replenishment_ratios,
        'wall_ratio': wall_ratios,
        'depth_in_zone': depth_in_zones,
        # Spatial Profile
        'limit_bid_size': bid_atm,
        'limit_ask_size': ask_atm,
        'limit_bid_size_above': bid_above,
        'limit_ask_size_above': ask_above,
        'limit_bid_size_below': bid_below,
        'limit_ask_size_below': ask_below
    }




# =============================================================================
# VECTORIZED FUEL ENGINE
# =============================================================================



# =============================================================================
# UNIFIED BATCH PHYSICS COMPUTATION
# =============================================================================

def compute_all_physics(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,  # ES prices
    directions: np.ndarray,    # 1=UP, -1=DOWN
    market_data: MarketData
) -> Dict[str, np.ndarray]:
    """
    Compute all physics metrics in a single pass.

    Combines tape, barrier, and fuel engines for maximum efficiency.

    Args:
        touch_ts_ns: Touch timestamps
        level_prices: Level prices (ES)
        directions: Direction array
        market_data: market data arrays

    Returns:
        Dict with all physics arrays
    """
    # Tape metrics
    tape_metrics = compute_tape_metrics(
        touch_ts_ns, level_prices, market_data,
        window_seconds=CONFIG.W_t,
        band_dollars=CONFIG.TAPE_BAND
    )

    # Barrier metrics
    barrier_metrics = compute_barrier_metrics(
        touch_ts_ns, level_prices, directions, market_data,
        window_seconds=CONFIG.W_b,
        zone_es_ticks=CONFIG.BARRIER_ZONE_ES_TICKS
    )

    # Fuel metrics
    fuel_metrics = compute_fuel_metrics(
        touch_ts_ns, level_prices, market_data,
        strike_range=CONFIG.FUEL_STRIKE_RANGE
    )

    # Combine all metrics
    result = {}
    result.update(tape_metrics)
    result.update(barrier_metrics)
    result.update(fuel_metrics)

    return result


# =============================================================================
# PERFORMANCE BENCHMARK
# =============================================================================

def benchmark_engines():
    """
    Benchmark physics engines performance.

    Generates synthetic data and measures throughput.
    """
    import time

    print("Physics Engines Benchmark")
    print("=" * 60)
    print(f"Numba: enabled")
    print()

    # Generate synthetic data
    n_trades = 1_000_000
    n_mbp = 100_000
    n_touches = 10_000

    print(f"Generating synthetic data...")
    print(f"  Trades: {n_trades:,}")
    print(f"  MBP-10: {n_mbp:,}")
    print(f"  Touches: {n_touches:,}")

    np.random.seed(42)

    # Synthetic trades
    base_ts = int(datetime.now().timestamp() * 1e9)
    trade_ts_ns = np.sort(base_ts + np.random.randint(0, int(6.5 * 3600 * 1e9), n_trades))
    trade_prices = 6000 + np.random.randn(n_trades).cumsum() * 0.1
    trade_sizes = np.random.randint(1, 50, n_trades).astype(np.int64)
    trade_aggressors = np.random.choice([-1, 0, 1], n_trades).astype(np.int8)

    # Synthetic MBP-10
    mbp_ts_ns = np.sort(base_ts + np.random.randint(0, int(6.5 * 3600 * 1e9), n_mbp))
    mbp_bid_prices = 6000 + np.random.randn(n_mbp, 10) * 0.5 - np.arange(10) * 0.25
    mbp_bid_sizes = np.random.randint(10, 500, (n_mbp, 10)).astype(np.int64)
    mbp_ask_prices = 6000 + np.random.randn(n_mbp, 10) * 0.5 + np.arange(10) * 0.25
    mbp_ask_sizes = np.random.randint(10, 500, (n_mbp, 10)).astype(np.int64)

    # Synthetic gamma data
    strikes = np.arange(595, 605, 1.0)
    strike_gamma = {s: np.random.randn() * 50000 for s in strikes}

    market_data = MarketData(
        trade_ts_ns=trade_ts_ns,
        trade_prices=trade_prices,
        trade_sizes=trade_sizes,
        trade_aggressors=trade_aggressors,
        mbp_ts_ns=mbp_ts_ns,
        mbp_bid_prices=mbp_bid_prices,
        mbp_bid_sizes=mbp_bid_sizes,
        mbp_ask_prices=mbp_ask_prices,
        mbp_ask_sizes=mbp_ask_sizes,
        strike_gamma=strike_gamma,
        opt_ts_ns=np.array([]),
        opt_strikes=np.array([]),
        opt_is_call=np.array([]),
        opt_premium=np.array([]),
        opt_net_gamma=np.array([])
    )

    # Synthetic touches
    touch_ts_ns = np.sort(base_ts + np.random.randint(0, int(6.5 * 3600 * 1e9), n_touches))
    level_prices = 600 + np.random.randn(n_touches) * 2
    directions = np.random.choice([1, -1], n_touches)

    print()

    # Warmup
    print("Warmup run...")
    _ = compute_all_physics(
        touch_ts_ns[:100], level_prices[:100], directions[:100], market_data
    )

    # Benchmark
    print("Benchmark run...")
    start = time.time()

    result = compute_all_physics(
        touch_ts_ns, level_prices, directions, market_data
    )

    elapsed = time.time() - start

    print()
    print("Results:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Throughput: {n_touches/elapsed:,.0f} touches/sec")
    print(f"  Per-touch: {elapsed*1000/n_touches:.3f}ms")
    print()
    print("Metrics computed:")
    for key, arr in result.items():
        print(f"  {key}: {arr.dtype}")


if __name__ == "__main__":
    benchmark_engines()