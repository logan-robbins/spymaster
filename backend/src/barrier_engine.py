"""
Barrier Engine: SPY L1 "book physics" proxy

Measures whether liquidity at the defending side is:
- PULLED (canceled without prints)
- CONSUMED (filled with prints)
- REPLENISHED (added back)

Based on PLAN.md §5.1

Interfaces consumed:
- event_types.StockTrade, StockQuote, Aggressor from Agent A
- market_state.MarketState queries from Agent C
- config.Config from Agent A
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
from collections import deque

from .event_types import StockTrade, StockQuote, Aggressor
from .market_state import MarketState, TimestampedTrade, TimestampedQuote
from .config import CONFIG


class BarrierState(str, Enum):
    """Classification of liquidity dynamics at a level"""
    VACUUM = "VACUUM"              # Liquidity pulled without fills
    WALL = "WALL"                  # Strong replenishment
    ABSORPTION = "ABSORPTION"      # Liquidity consumed but replenished
    CONSUMED = "CONSUMED"          # Liquidity eaten faster than replenished
    WEAK = "WEAK"                  # Defending size below baseline
    NEUTRAL = "NEUTRAL"            # Normal state


class Direction(str, Enum):
    """Test direction at a level"""
    SUPPORT = "SUPPORT"    # Spot > L, approaching from above, break = DOWN
    RESISTANCE = "RESISTANCE"  # Spot < L, approaching from below, break = UP


@dataclass
class BarrierMetrics:
    """Output of barrier physics computation"""
    state: BarrierState
    delta_liq: float           # Net liquidity change
    replenishment_ratio: float # R = added / (canceled + filled + ε)
    added_size: float
    canceled_size: float
    filled_size: float
    defending_quote: dict      # {price, size}
    confidence: float          # 0-1, based on sample size and stability


class BarrierEngine:
    """
    Computes liquidity state at a level using SPY NBBO (L1) as proxy.
    
    For a given level L and direction (SUPPORT/RESISTANCE):
    - Track defending quote size changes over rolling window
    - Classify removals as FILLED (concurrent prints) vs CANCELED (no prints)
    - Compute replenishment ratio and state
    
    Interfaces consumed:
    - MarketState queries (Agent C) for windowed quotes/trades
    - Config thresholds (Agent A) for classification constants
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: Config object (defaults to global CONFIG)
        """
        self.config = config or CONFIG
        
        # Extract config values
        self.window_seconds = self.config.W_b
        self.level_proximity_threshold = 0.02  # 2% proximity to level
        self.r_vacuum = self.config.R_vac
        self.r_wall = self.config.R_wall
        self.delta_liq_thresh = self.config.F_thresh
        self.weak_percentile = self.config.WEAK_PERCENTILE
        self.baseline_window_seconds = self.config.WEAK_LOOKBACK
        
        # Ring buffers for baseline tracking (optional WEAK state)
        self.baseline_bid_sizes = deque(maxlen=1000)
        self.baseline_ask_sizes = deque(maxlen=1000)
    
    def compute_barrier_state(
        self,
        level_price: float,
        direction: Direction,
        market_state: MarketState
    ) -> BarrierMetrics:
        """
        Compute barrier state for a level using MarketState.
        
        Args:
            level_price: The critical level being tested
            direction: SUPPORT or RESISTANCE
            market_state: MarketState instance with quote/trade buffers
            
        Returns:
            BarrierMetrics with state classification and flows
        """
        # Get current time and NBBO
        ts_now_ns = market_state.get_current_ts_ns()
        
        if market_state.last_quote is None:
            # No quote data, return neutral
            return self._neutral_metrics()
        
        current_bid = market_state.last_quote.bid_px
        current_ask = market_state.last_quote.ask_px
        current_bid_size = market_state.last_quote.bid_sz
        current_ask_size = market_state.last_quote.ask_sz
        
        # Determine defending side
        if direction == Direction.SUPPORT:
            # Level is support, defending quote is BID
            defending_price = current_bid
            defending_size = current_bid_size
            quote_extractor = lambda q: (q.bid_px, q.bid_sz)
        else:
            # Level is resistance, defending quote is ASK
            defending_price = current_ask
            defending_size = current_ask_size
            quote_extractor = lambda q: (q.ask_px, q.ask_sz)
        
        # Check if defending quote is near the level
        if abs(defending_price - level_price) > level_price * self.level_proximity_threshold:
            # Quote not near level, return neutral state
            return BarrierMetrics(
                state=BarrierState.NEUTRAL,
                delta_liq=0.0,
                replenishment_ratio=0.0,
                added_size=0.0,
                canceled_size=0.0,
                filled_size=0.0,
                defending_quote={"price": defending_price, "size": defending_size},
                confidence=0.0
            )
        
        # Get windowed data from MarketState
        quotes = market_state.get_quotes_in_window(ts_now_ns, self.window_seconds)
        trades = market_state.get_trades_in_window(ts_now_ns, self.window_seconds)
        
        # Compute size changes and classify as added/canceled/filled
        added_size, canceled_size, filled_size = self._compute_liquidity_flows(
            quotes=quotes,
            trades=trades,
            level_price=level_price,
            direction=direction,
            quote_extractor=quote_extractor
        )
        
        # Compute metrics
        epsilon = 1e-6
        delta_liq = added_size - canceled_size - filled_size
        replenishment_ratio = added_size / (canceled_size + filled_size + epsilon)
        
        # Classify state
        state = self._classify_state(
            delta_liq=delta_liq,
            replenishment_ratio=replenishment_ratio,
            canceled_size=canceled_size,
            filled_size=filled_size,
            defending_size=defending_size,
            direction=direction
        )
        
        # Compute confidence based on sample size
        total_activity = added_size + canceled_size + filled_size
        confidence = min(1.0, total_activity / 10000.0)  # Scale to 1.0 at 10k shares
        
        return BarrierMetrics(
            state=state,
            delta_liq=delta_liq,
            replenishment_ratio=replenishment_ratio,
            added_size=added_size,
            canceled_size=canceled_size,
            filled_size=filled_size,
            defending_quote={"price": defending_price, "size": defending_size},
            confidence=confidence
        )
    
    def _neutral_metrics(self) -> BarrierMetrics:
        """Return neutral metrics when no data available."""
        return BarrierMetrics(
            state=BarrierState.NEUTRAL,
            delta_liq=0.0,
            replenishment_ratio=0.0,
            added_size=0.0,
            canceled_size=0.0,
            filled_size=0.0,
            defending_quote={"price": 0.0, "size": 0},
            confidence=0.0
        )
    
    def _compute_liquidity_flows(
        self,
        quotes: List[TimestampedQuote],
        trades: List[TimestampedTrade],
        level_price: float,
        direction: Direction,
        quote_extractor
    ) -> Tuple[float, float, float]:
        """
        Compute added, canceled, and filled size over the window.
        
        Returns:
            (added_size, canceled_size, filled_size)
        """
        if not quotes:
            return 0.0, 0.0, 0.0
        
        added_size = 0.0
        canceled_size = 0.0
        filled_size = 0.0
        
        # Track size changes between quotes
        prev_price = None
        prev_size = None
        
        for i, quote in enumerate(quotes):
            price, size = quote_extractor(quote)
            
            # Only consider quotes near the level
            if abs(price - level_price) > level_price * self.level_proximity_threshold:
                prev_price, prev_size = price, size
                continue
            
            if prev_price is not None and abs(prev_price - level_price) <= level_price * self.level_proximity_threshold:
                # Both quotes are near level
                size_delta = size - prev_size
                
                if size_delta > 0:
                    # Size increased = ADDED
                    added_size += size_delta
                elif size_delta < 0:
                    # Size decreased = either FILLED or CANCELED
                    size_removed = abs(size_delta)
                    
                    # Check for concurrent trades at/through the level
                    # Trades that occurred between prev_quote and current quote
                    concurrent_volume = self._get_concurrent_trade_volume(
                        trades=trades,
                        start_ts=quotes[i-1].ts_event_ns if i > 0 else quote.ts_event_ns - int(1e9),
                        end_ts=quote.ts_event_ns,
                        level_price=level_price,
                        direction=direction
                    )
                    
                    # Heuristic: attribute removal to fills up to concurrent volume
                    filled = min(size_removed, concurrent_volume)
                    canceled = max(0, size_removed - concurrent_volume)
                    
                    filled_size += filled
                    canceled_size += canceled
            
            prev_price, prev_size = price, size
        
        return added_size, canceled_size, filled_size
    
    def _get_concurrent_trade_volume(
        self,
        trades: List[TimestampedTrade],
        start_ts: int,
        end_ts: int,
        level_price: float,
        direction: Direction
    ) -> float:
        """
        Get volume of trades at/through the level in the time window.
        
        For SUPPORT: trades at or below level (selling through)
        For RESISTANCE: trades at or above level (buying through)
        """
        volume = 0.0
        
        for trade in trades:
            if start_ts <= trade.ts_event_ns <= end_ts:
                if direction == Direction.SUPPORT:
                    # Support test: count sells at/below level
                    if trade.price <= level_price:
                        volume += trade.size
                else:
                    # Resistance test: count buys at/above level
                    if trade.price >= level_price:
                        volume += trade.size
        
        return volume
    
    def _classify_state(
        self,
        delta_liq: float,
        replenishment_ratio: float,
        canceled_size: float,
        filled_size: float,
        defending_size: float,
        direction: Direction
    ) -> BarrierState:
        """
        Classify barrier state based on computed metrics.
        
        Logic from PLAN.md §5.1:
        - VACUUM: R < R_vac AND Δliq < -F_thresh
        - WALL/ABSORPTION: R > R_wall AND Δliq > +F_thresh
        - CONSUMED: Δliq << 0 AND filled > canceled
        - WEAK: defending_size below baseline percentile (optional)
        - NEUTRAL: otherwise
        """
        # VACUUM condition
        if replenishment_ratio < self.r_vacuum and delta_liq < -self.delta_liq_thresh:
            return BarrierState.VACUUM
        
        # WALL/ABSORPTION condition
        if replenishment_ratio > self.r_wall and delta_liq > self.delta_liq_thresh:
            return BarrierState.WALL
        
        # CONSUMED condition
        if delta_liq < -self.delta_liq_thresh and filled_size > canceled_size:
            return BarrierState.CONSUMED
        
        # WEAK condition (optional - requires baseline)
        if self._is_weak_size(defending_size, direction):
            return BarrierState.WEAK
        
        # NEUTRAL (contested or insufficient signal)
        if abs(delta_liq) < self.delta_liq_thresh / 2:
            return BarrierState.NEUTRAL
        
        # Mild absorption (replenishment happening but not strong enough for WALL)
        if delta_liq > 0 and replenishment_ratio > 1.0:
            return BarrierState.ABSORPTION
        
        return BarrierState.NEUTRAL
    
    def _is_weak_size(self, defending_size: float, direction: Direction) -> bool:
        """Check if defending size is below baseline (WEAK state)"""
        baseline = self.baseline_bid_sizes if direction == Direction.SUPPORT else self.baseline_ask_sizes
        
        if len(baseline) < 100:  # Not enough data
            return False
        
        sorted_baseline = sorted(baseline)
        percentile_idx = int(len(sorted_baseline) * self.weak_percentile)
        threshold = sorted_baseline[percentile_idx]
        
        return defending_size < threshold
    
    def update_baseline(self, bid_size: int, ask_size: int):
        """Update baseline distribution for WEAK detection (call periodically)"""
        self.baseline_bid_sizes.append(bid_size)
        self.baseline_ask_sizes.append(ask_size)

