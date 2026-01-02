"""
Vectorized Physics Engines - Apple M4 Silicon Optimized

Batch processing versions of barrier, tape, and fuel engines
that operate on numpy arrays for maximum throughput.

These engines complement the existing per-signal engines by providing
batch-optimized alternatives for research pipeline processing.

Key Optimizations:
- Pre-computed lookup structures
- Numpy vectorized operations
- Numba JIT for tight loops
- Memory-efficient data structures
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from src.common.event_types import FuturesTrade, MBP10, Aggressor
from src.common.config import CONFIG


# =============================================================================
# NUMBA SUPPORT
# =============================================================================

try:
    from numba import jit, prange
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# VECTORIZED DATA STRUCTURES
# =============================================================================

@dataclass
class VectorizedMarketData:
    """
    Pre-processed market data for vectorized queries.

    Trades and depth organized for O(1) time-window lookups.
    """
    # ES Trades (sorted by timestamp)
    trade_ts_ns: np.ndarray          # int64
    trade_prices: np.ndarray         # float64 (ES prices)
    trade_sizes: np.ndarray          # int64
    trade_aggressors: np.ndarray     # int8 (1=BUY, -1=SELL, 0=MID)

    # ES MBP-10 snapshots (sorted by timestamp)
    mbp_ts_ns: np.ndarray            # int64
    mbp_bid_prices: np.ndarray       # float64, shape (n, 10)
    mbp_bid_sizes: np.ndarray        # int64, shape (n, 10)
    mbp_ask_prices: np.ndarray       # float64, shape (n, 10)
    mbp_ask_sizes: np.ndarray        # int64, shape (n, 10)

    # Option flows by strike (pre-aggregated)
    strike_gamma: Dict[float, float]
    strike_volume: Dict[float, int]
    call_gamma: Dict[float, float]
    put_gamma: Dict[float, float]
    strike_premium: Dict[float, float]  # Net premium flow by strike
    call_premium: Dict[float, float]
    put_premium: Dict[float, float]




def build_vectorized_market_data(
    trades: List[FuturesTrade],
    mbp10_snapshots: List[MBP10],
    option_flows: Dict[Tuple[float, str, str], Any],
    date: str
) -> VectorizedMarketData:
    """
    Convert raw data to vectorized format for batch processing.

    Pre-computes all lookup structures for O(1) queries.
    """
    # Convert trades to numpy arrays
    n_trades = len(trades)
    trade_ts_ns = np.empty(n_trades, dtype=np.int64)
    trade_prices = np.empty(n_trades, dtype=np.float64)
    trade_sizes = np.empty(n_trades, dtype=np.int64)
    trade_aggressors = np.empty(n_trades, dtype=np.int8)

    for i, trade in enumerate(trades):
        trade_ts_ns[i] = trade.ts_event_ns
        trade_prices[i] = trade.price
        trade_sizes[i] = trade.size
        trade_aggressors[i] = trade.aggressor.value if hasattr(trade.aggressor, 'value') else 0

    # Sort by timestamp
    sort_idx = np.argsort(trade_ts_ns)
    trade_ts_ns = trade_ts_ns[sort_idx]
    trade_prices = trade_prices[sort_idx]
    trade_sizes = trade_sizes[sort_idx]
    trade_aggressors = trade_aggressors[sort_idx]

    # Convert MBP-10 to numpy arrays
    n_mbp = len(mbp10_snapshots)
    mbp_ts_ns = np.empty(n_mbp, dtype=np.int64)
    mbp_bid_prices = np.zeros((n_mbp, 10), dtype=np.float64)
    mbp_bid_sizes = np.zeros((n_mbp, 10), dtype=np.int64)
    mbp_ask_prices = np.zeros((n_mbp, 10), dtype=np.float64)
    mbp_ask_sizes = np.zeros((n_mbp, 10), dtype=np.int64)

    for i, mbp in enumerate(mbp10_snapshots):
        mbp_ts_ns[i] = mbp.ts_event_ns
        for j, level in enumerate(mbp.levels[:10]):
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
    strike_volume = defaultdict(int)
    call_gamma = defaultdict(float)
    put_gamma = defaultdict(float)

    strike_premium = defaultdict(float)  # Store net_premium_flow by strike
    call_premium = defaultdict(float)
    put_premium = defaultdict(float)
    
    for (strike, right, exp_date), flow in option_flows.items():
        if date is None or exp_date == date:
            strike_gamma[strike] += flow.net_gamma_flow
            strike_volume[strike] += flow.cumulative_volume
            strike_premium[strike] += flow.net_premium_flow  # Aggregate premium flow
            if right == 'C':
                call_gamma[strike] += flow.net_gamma_flow
                call_premium[strike] += flow.net_premium_flow
            else:
                put_gamma[strike] += flow.net_gamma_flow
                put_premium[strike] += flow.net_premium_flow

    import logging
    logger = logging.getLogger(__name__)
    
    # Sort strikes by premium magnitude
    sorted_prem = sorted(strike_premium.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    logger.info(f"DEBUG: Top 5 Premium Strikes: {sorted_prem} (Spot ~6925?)")

    return VectorizedMarketData(
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
        strike_volume=dict(strike_volume),
        call_gamma=dict(call_gamma),
        put_gamma=dict(put_gamma),
        strike_premium=dict(strike_premium),
        call_premium=dict(call_premium),
        put_premium=dict(put_premium)
    )


# =============================================================================
# VECTORIZED TAPE ENGINE
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_tape_metrics_batch_numba(
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
    Numba-accelerated batch tape metrics computation.

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

        # Compute velocity using linear regression (all trades in window)
        n_window = end_idx - start_idx
        if n_window >= 2:
            # Compute means
            t_sum = 0.0
            p_sum = 0.0
            for j in range(start_idx, end_idx):
                t_sum += trade_ts_ns[j] / 1e9
                p_sum += trade_prices[j]
            t_mean = t_sum / n_window
            p_mean = p_sum / n_window

            # Compute slope
            numerator = 0.0
            denominator = 0.0
            for j in range(start_idx, end_idx):
                dt = trade_ts_ns[j] / 1e9 - t_mean
                dp = trade_prices[j] - p_mean
                numerator += dt * dp
                denominator += dt * dt

            if denominator > 1e-10:
                velocities[i] = numerator / denominator

    return imbalances, buy_vols, sell_vols, velocities


def compute_tape_metrics_batch(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,  # ES prices
    market_data: VectorizedMarketData,
    window_seconds: float = 5.0,
    band_dollars: float = 0.10
) -> Dict[str, np.ndarray]:
    """
    Compute tape metrics for all touches in batch.

    Args:
        touch_ts_ns: Touch timestamps
        level_prices: Level prices (ES points)
        market_data: Vectorized market data
        window_seconds: Lookback window
        band_dollars: Price band (ES points)

    Returns:
        Dict with arrays: imbalance, buy_vol, sell_vol, velocity
    """
    # Use ES prices directly
    level_prices_es = level_prices.astype(np.float64)
    band_es = float(band_dollars)
    window_ns = int(window_seconds * 1e9)

    if NUMBA_AVAILABLE:
        imbalances, buy_vols, sell_vols, velocities = _compute_tape_metrics_batch_numba(
            touch_ts_ns.astype(np.int64),
            level_prices_es.astype(np.float64),
            market_data.trade_ts_ns,
            market_data.trade_prices,
            market_data.trade_sizes,
            market_data.trade_aggressors,
            window_ns,
            band_es
        )
    else:
        # Pure numpy fallback
        n = len(touch_ts_ns)
        imbalances = np.zeros(n)
        buy_vols = np.zeros(n, dtype=np.int64)
        sell_vols = np.zeros(n, dtype=np.int64)
        velocities = np.zeros(n)

        for i in range(n):
            ts = touch_ts_ns[i]
            level_es = level_prices_es[i]

            # FORWARD time window mask [ts, ts + window_ns]
            # Touch timestamp is at bar start, trades happen during the bar
            time_mask = (market_data.trade_ts_ns >= ts) & (market_data.trade_ts_ns <= ts + window_ns)

            # Price band mask
            price_mask = np.abs(market_data.trade_prices - level_es) <= band_es

            mask = time_mask & price_mask

            # Compute volumes
            buy_mask = mask & (market_data.trade_aggressors == 1)
            sell_mask = mask & (market_data.trade_aggressors == -1)

            buy_vol = market_data.trade_sizes[buy_mask].sum()
            sell_vol = market_data.trade_sizes[sell_mask].sum()

            buy_vols[i] = buy_vol
            sell_vols[i] = sell_vol

            total = buy_vol + sell_vol
            if total > 0:
                imbalances[i] = (buy_vol - sell_vol) / total

            # Velocity
            time_trades = market_data.trade_ts_ns[time_mask]
            price_trades = market_data.trade_prices[time_mask]

            if len(time_trades) >= 2:
                times = time_trades.astype(np.float64) / 1e9
                times = times - times[0]
                if times[-1] > 0:
                    slope, _ = np.polyfit(times, price_trades, 1)
                    velocities[i] = slope

    return {
        'tape_imbalance': imbalances,
        'tape_buy_vol': buy_vols,
        'tape_sell_vol': sell_vols,
        'tape_velocity': velocities
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


def compute_barrier_metrics_batch(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,  # ES prices (strike-aligned)
    directions: np.ndarray,    # 'UP' or 'DOWN' as int: 1 or -1
    market_data: VectorizedMarketData,
    window_seconds: float = 10.0,
    zone_es_ticks: int = 2
) -> Dict[str, np.ndarray]:
    """
    Compute barrier metrics for all touches in batch.

    Use ES points directly for level and MBP-10 depth alignment.

    Args:
        touch_ts_ns: Touch timestamps
        level_prices: Level prices (ES, strike-aligned)
        directions: Direction array (1=UP/resistance, -1=DOWN/support)
        market_data: Vectorized market data
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

    for i in range(n):
        ts = touch_ts_ns[i]
        level_es = level_prices_es[i]
        direction = directions[i]

        # Zone boundaries in ES ticks around the strike level
        zone_low = level_es - zone_es
        zone_high = level_es + zone_es

        # Side: support=bid, resistance=ask
        side = 1 if direction == -1 else -1  # 1=bid, -1=ask

        # Find MBP-10 snapshots in FORWARD window [ts, ts + window_ns]
        # Touch timestamp is at bar start, MBP snapshots occur during the bar
        time_mask = (market_data.mbp_ts_ns >= ts) & (market_data.mbp_ts_ns <= ts + window_ns)
        valid_indices = np.where(time_mask)[0]

        if len(valid_indices) < 2:
            barrier_states[i] = 'NEUTRAL'
            continue

        # Get first and last snapshots in window
        first_idx = valid_indices[0]
        last_idx = valid_indices[-1]

        # Compute depth at start and end
        if side == 1:  # bid
            depth_start = _compute_depth_in_zone_numba(
                market_data.mbp_bid_prices[first_idx],
                market_data.mbp_bid_sizes[first_idx],
                market_data.mbp_ask_prices[first_idx],
                market_data.mbp_ask_sizes[first_idx],
                zone_low, zone_high, side
            ) if NUMBA_AVAILABLE else market_data.mbp_bid_sizes[first_idx].sum()

            depth_end = _compute_depth_in_zone_numba(
                market_data.mbp_bid_prices[last_idx],
                market_data.mbp_bid_sizes[last_idx],
                market_data.mbp_ask_prices[last_idx],
                market_data.mbp_ask_sizes[last_idx],
                zone_low, zone_high, side
            ) if NUMBA_AVAILABLE else market_data.mbp_bid_sizes[last_idx].sum()
        else:
            depth_start = _compute_depth_in_zone_numba(
                market_data.mbp_bid_prices[first_idx],
                market_data.mbp_bid_sizes[first_idx],
                market_data.mbp_ask_prices[first_idx],
                market_data.mbp_ask_sizes[first_idx],
                zone_low, zone_high, side
            ) if NUMBA_AVAILABLE else market_data.mbp_ask_sizes[first_idx].sum()

            depth_end = _compute_depth_in_zone_numba(
                market_data.mbp_bid_prices[last_idx],
                market_data.mbp_bid_sizes[last_idx],
                market_data.mbp_ask_prices[last_idx],
                market_data.mbp_ask_sizes[last_idx],
                zone_low, zone_high, side
            ) if NUMBA_AVAILABLE else market_data.mbp_ask_sizes[last_idx].sum()

        delta_liq = depth_end - depth_start
        delta_liqs[i] = delta_liq
        depth_in_zones[i] = depth_end

        # Compute wall ratio (defending depth vs average)
        avg_depth = (market_data.mbp_bid_sizes.mean() + market_data.mbp_ask_sizes.mean()) / 2
        wall_ratios[i] = depth_end / (avg_depth + 1e-6)
        
        # Compute replenishment ratio: added / (canceled + filled + epsilon)
        # Track depth changes across all snapshots in window
        added_size = 0.0
        canceled_size = 0.0
        
        for j in range(len(valid_indices) - 1):
            idx_curr = valid_indices[j]
            idx_next = valid_indices[j + 1]
            
            # Get depth at consecutive snapshots
            if side == 1:  # bid
                depth_curr = _compute_depth_in_zone_numba(
                    market_data.mbp_bid_prices[idx_curr],
                    market_data.mbp_bid_sizes[idx_curr],
                    market_data.mbp_ask_prices[idx_curr],
                    market_data.mbp_ask_sizes[idx_curr],
                    zone_low, zone_high, side
                ) if NUMBA_AVAILABLE else market_data.mbp_bid_sizes[idx_curr].sum()
                
                depth_next = _compute_depth_in_zone_numba(
                    market_data.mbp_bid_prices[idx_next],
                    market_data.mbp_bid_sizes[idx_next],
                    market_data.mbp_ask_prices[idx_next],
                    market_data.mbp_ask_sizes[idx_next],
                    zone_low, zone_high, side
                ) if NUMBA_AVAILABLE else market_data.mbp_bid_sizes[idx_next].sum()
            else:  # ask
                depth_curr = _compute_depth_in_zone_numba(
                    market_data.mbp_bid_prices[idx_curr],
                    market_data.mbp_bid_sizes[idx_curr],
                    market_data.mbp_ask_prices[idx_curr],
                    market_data.mbp_ask_sizes[idx_curr],
                    zone_low, zone_high, side
                ) if NUMBA_AVAILABLE else market_data.mbp_ask_sizes[idx_curr].sum()
                
                depth_next = _compute_depth_in_zone_numba(
                    market_data.mbp_bid_prices[idx_next],
                    market_data.mbp_bid_sizes[idx_next],
                    market_data.mbp_ask_prices[idx_next],
                    market_data.mbp_ask_sizes[idx_next],
                    zone_low, zone_high, side
                ) if NUMBA_AVAILABLE else market_data.mbp_ask_sizes[idx_next].sum()
            
            delta_depth = depth_next - depth_curr
            
            if delta_depth > 0:
                added_size += delta_depth
            elif delta_depth < 0:
                # Depth decreased - treat as canceled (simplified, no trade matching)
                canceled_size += abs(delta_depth)
        
        # Compute replenishment ratio with epsilon to avoid division by zero
        epsilon = 1e-6
        replenishment_ratios[i] = added_size / (canceled_size + epsilon)

        # Classify state
        if delta_liq < -100:
            if wall_ratios[i] < 0.3:
                barrier_states[i] = 'VACUUM'
            else:
                barrier_states[i] = 'CONSUMED'
        elif delta_liq > 100:
            if wall_ratios[i] > 1.5:
                barrier_states[i] = 'WALL'
            else:
                barrier_states[i] = 'ABSORPTION'
        elif depth_end < 50:
            barrier_states[i] = 'WEAK'
        else:
            barrier_states[i] = 'NEUTRAL'

    return {
        'barrier_state': barrier_states,
        'barrier_delta_liq': delta_liqs,
        'barrier_replenishment_ratio': replenishment_ratios,
        'wall_ratio': wall_ratios,
        'depth_in_zone': depth_in_zones
    }


# =============================================================================
# VECTORIZED FUEL ENGINE
# =============================================================================

def compute_fuel_metrics_batch(
    level_prices: np.ndarray,  # ES prices
    market_data: VectorizedMarketData,
    strike_range: float = 2.0
) -> Dict[str, np.ndarray]:
    """
    Compute fuel metrics for all levels in batch.

    Args:
        level_prices: Level prices (ES)
        market_data: Vectorized market data with pre-aggregated gamma
        strike_range: Strike range around level

    Returns:
        Dict with arrays: gamma_exposure, fuel_effect, call_tide, put_tide
    """
    n = len(level_prices)

    gamma_exposures = np.zeros(n, dtype=np.float64)
    fuel_effects = np.empty(n, dtype=object)
    call_tides = np.zeros(n, dtype=np.float64)
    put_tides = np.zeros(n, dtype=np.float64)

    # Pre-convert strikes to array for fast lookup
    strikes = np.array(list(market_data.strike_gamma.keys()))
    gamma_values = np.array(list(market_data.strike_gamma.values()))
    
    # Extract call/put gamma for premium flow calculation
    call_strikes = np.array(list(market_data.call_gamma.keys()))
    call_gamma_values = np.array(list(market_data.call_gamma.values()))
    put_strikes = np.array(list(market_data.put_gamma.keys()))
    put_gamma_values = np.array(list(market_data.put_gamma.values()))
    
    # Extract premium flows (separated by right)
    call_premium_strikes = np.array(list(market_data.call_premium.keys()))
    call_premium_values = np.array(list(market_data.call_premium.values()))
    put_premium_strikes = np.array(list(market_data.put_premium.keys()))
    put_premium_values = np.array(list(market_data.put_premium.values()))

    if len(strikes) == 0:
        fuel_effects[:] = 'NEUTRAL'
        return {
            'gamma_exposure': gamma_exposures,
            'fuel_effect': fuel_effects,
            'call_tide': call_tides,
            'put_tide': put_tides
        }

    for i in range(n):
        level = level_prices[i]

        # Find strikes in range
        mask = np.abs(strikes - level) <= strike_range

        if not mask.any():
            fuel_effects[i] = 'NEUTRAL'
            continue

        # Sum gamma in range
        net_gamma = gamma_values[mask].sum()
        gamma_exposures[i] = net_gamma

        # Compute call/put tide (premium flow) - use separated premiums
        if len(call_premium_strikes) > 0:
            call_mask = np.abs(call_premium_strikes - level) <= strike_range
            if call_mask.any():
                call_tides[i] = call_premium_values[call_mask].sum()
                
        if len(put_premium_strikes) > 0:
            put_mask = np.abs(put_premium_strikes - level) <= strike_range
            if put_mask.any():
                put_tides[i] = put_premium_values[put_mask].sum()

        # Classify effect
        if net_gamma < -10000:
            fuel_effects[i] = 'AMPLIFY'
        elif net_gamma > 10000:
            fuel_effects[i] = 'DAMPEN'
        else:
            fuel_effects[i] = 'NEUTRAL'

    return {
        'gamma_exposure': gamma_exposures,
        'fuel_effect': fuel_effects,
        'call_tide': call_tides,
        'put_tide': put_tides
    }


# =============================================================================
# UNIFIED BATCH PHYSICS COMPUTATION
# =============================================================================

def compute_all_physics_batch(
    touch_ts_ns: np.ndarray,
    level_prices: np.ndarray,  # ES prices
    directions: np.ndarray,    # 1=UP, -1=DOWN
    market_data: VectorizedMarketData
) -> Dict[str, np.ndarray]:
    """
    Compute all physics metrics in a single batch pass.

    Combines tape, barrier, and fuel engines for maximum efficiency.

    Args:
        touch_ts_ns: Touch timestamps
        level_prices: Level prices (ES)
        directions: Direction array
        market_data: Vectorized market data

    Returns:
        Dict with all physics arrays
    """
    # Tape metrics
    tape_metrics = compute_tape_metrics_batch(
        touch_ts_ns, level_prices, market_data,
        window_seconds=CONFIG.W_t,
        band_dollars=CONFIG.TAPE_BAND
    )

    # Barrier metrics
    barrier_metrics = compute_barrier_metrics_batch(
        touch_ts_ns, level_prices, directions, market_data,
        window_seconds=CONFIG.W_b,
        zone_es_ticks=CONFIG.BARRIER_ZONE_ES_TICKS
    )

    # Fuel metrics
    fuel_metrics = compute_fuel_metrics_batch(
        level_prices, market_data,
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

def benchmark_batch_engines():
    """
    Benchmark vectorized engines performance.

    Generates synthetic data and measures throughput.
    """
    import time

    print("Vectorized Engines Benchmark")
    print("=" * 60)
    print(f"Numba: {'enabled' if NUMBA_AVAILABLE else 'disabled'}")
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
    call_gamma = {s: np.random.randn() * 25000 for s in strikes}
    put_gamma = {s: np.random.randn() * 25000 for s in strikes}

    market_data = VectorizedMarketData(
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
        strike_volume={s: 1000 for s in strikes},
        call_gamma=call_gamma,
        put_gamma=put_gamma
    )

    # Synthetic touches
    touch_ts_ns = np.sort(base_ts + np.random.randint(0, int(6.5 * 3600 * 1e9), n_touches))
    level_prices = 600 + np.random.randn(n_touches) * 2
    directions = np.random.choice([1, -1], n_touches)

    print()

    # Warmup
    print("Warmup run...")
    _ = compute_all_physics_batch(
        touch_ts_ns[:100], level_prices[:100], directions[:100], market_data
    )

    # Benchmark
    print("Benchmark run...")
    start = time.time()

    result = compute_all_physics_batch(
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
    from datetime import datetime
    benchmark_batch_engines()
