"""
MarketState: central state store for market data and ring buffers.

Agent C deliverable per §12 of PLAN.md.

This module maintains:
- Last-known SPY quote and trade
- Rolling time-windowed buffers for SPY quotes and trades
- Per-strike option flow aggregates (integrates with existing flow_aggregator.py)
- Efficient window queries for engines (Barrier, Tape, Fuel)

All timestamps are Unix nanoseconds (UTC).
"""

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import time

from .event_types import StockTrade, StockQuote, OptionTrade, Aggressor
from .config import CONFIG


@dataclass
class TimestampedTrade:
    """Trade with timestamp for ring buffer storage."""
    ts_event_ns: int
    price: float
    size: int
    aggressor: Aggressor


@dataclass
class TimestampedQuote:
    """Quote with timestamp for ring buffer storage."""
    ts_event_ns: int
    bid_px: float
    ask_px: float
    bid_sz: int
    ask_sz: int


@dataclass
class OptionFlowAggregate:
    """
    Per-strike option flow metrics.
    Compatible with existing flow_aggregator.py output.
    """
    strike: float
    right: str  # 'C' or 'P'
    exp_date: str
    cumulative_volume: int = 0
    cumulative_premium: float = 0.0
    net_delta_flow: float = 0.0
    net_gamma_flow: float = 0.0
    last_price: float = 0.0
    last_timestamp_ns: int = 0
    delta: float = 0.0
    gamma: float = 0.0


class RingBuffer:
    """
    Time-windowed ring buffer with automatic cleanup.
    
    Stores timestamped events and provides efficient window queries.
    """
    
    def __init__(self, max_window_seconds: float = 60.0):
        self.buffer: deque = deque()
        self.max_window_ns = int(max_window_seconds * 1e9)
    
    def append(self, item):
        """Add item to buffer."""
        self.buffer.append(item)
    
    def cleanup(self, current_ts_ns: int):
        """Remove items older than max_window_ns."""
        cutoff_ns = current_ts_ns - self.max_window_ns
        while self.buffer and self.buffer[0].ts_event_ns < cutoff_ns:
            self.buffer.popleft()
    
    def get_window(self, current_ts_ns: int, window_seconds: float) -> List:
        """
        Get all items within the last window_seconds.
        
        Args:
            current_ts_ns: current timestamp (Unix ns)
            window_seconds: lookback window in seconds
            
        Returns:
            List of items within window, oldest first
        """
        self.cleanup(current_ts_ns)
        cutoff_ns = current_ts_ns - int(window_seconds * 1e9)
        return [item for item in self.buffer if item.ts_event_ns >= cutoff_ns]
    
    def __len__(self):
        return len(self.buffer)


class MarketState:
    """
    Central state store for market data.
    
    Maintains:
    - Last-known values for SPY spot/bid/ask
    - Rolling buffers for trades and quotes
    - Per-strike option flow aggregates
    
    Thread-safety: Not thread-safe by default. Use from single event loop.
    """
    
    def __init__(self, max_buffer_window_seconds: float = 120.0):
        # ========== Last-known values ==========
        self.last_trade: Optional[TimestampedTrade] = None
        self.last_quote: Optional[TimestampedQuote] = None
        
        # ========== Ring buffers ==========
        # Store up to max_buffer_window_seconds of history
        self.trades_buffer = RingBuffer(max_window_seconds=max_buffer_window_seconds)
        self.quotes_buffer = RingBuffer(max_window_seconds=max_buffer_window_seconds)
        
        # ========== Option flow aggregates ==========
        # Key: (strike, right, exp_date) -> OptionFlowAggregate
        self.option_flows: Dict[Tuple[float, str, str], OptionFlowAggregate] = {}
        
        # ========== Derived state (updated on demand) ==========
        self._vwap: Optional[float] = None
        self._vwap_volume: int = 0
        self._session_high: Optional[float] = None
        self._session_low: Optional[float] = None
    
    # ========== Stock updates ==========
    
    def update_stock_trade(self, trade: StockTrade, aggressor: Optional[Aggressor] = None):
        """
        Update market state with a new stock trade.
        
        Args:
            trade: StockTrade event
            aggressor: Optional aggressor classification (if not in trade)
        """
        # Infer aggressor if not provided
        if aggressor is None:
            aggressor = self._infer_aggressor_from_trade(trade)
        
        timestamped = TimestampedTrade(
            ts_event_ns=trade.ts_event_ns,
            price=trade.price,
            size=trade.size,
            aggressor=aggressor
        )
        
        self.last_trade = timestamped
        self.trades_buffer.append(timestamped)
        
        # Update session high/low
        if self._session_high is None or trade.price > self._session_high:
            self._session_high = trade.price
        if self._session_low is None or trade.price < self._session_low:
            self._session_low = trade.price
        
        # Update VWAP
        notional = trade.price * trade.size
        if self._vwap is None:
            self._vwap = trade.price
            self._vwap_volume = trade.size
        else:
            total_volume = self._vwap_volume + trade.size
            self._vwap = (self._vwap * self._vwap_volume + notional) / total_volume
            self._vwap_volume = total_volume
    
    def update_stock_quote(self, quote: StockQuote):
        """
        Update market state with a new stock quote (NBBO).
        
        Args:
            quote: StockQuote event
        """
        timestamped = TimestampedQuote(
            ts_event_ns=quote.ts_event_ns,
            bid_px=quote.bid_px,
            ask_px=quote.ask_px,
            bid_sz=quote.bid_sz,
            ask_sz=quote.ask_sz
        )
        
        self.last_quote = timestamped
        self.quotes_buffer.append(timestamped)
    
    def _infer_aggressor_from_trade(self, trade: StockTrade) -> Aggressor:
        """
        Infer aggressor side from trade price vs last known quote.
        
        Logic:
        - If trade_price >= ask: BUY (lifted ask)
        - If trade_price <= bid: SELL (hit bid)
        - Else: MID (unknown)
        """
        if self.last_quote is None:
            return Aggressor.MID
        
        if trade.price >= self.last_quote.ask_px:
            return Aggressor.BUY
        elif trade.price <= self.last_quote.bid_px:
            return Aggressor.SELL
        else:
            return Aggressor.MID
    
    # ========== Option updates ==========
    
    def update_option_trade(
        self, 
        trade: OptionTrade, 
        delta: float = 0.0, 
        gamma: float = 0.0
    ):
        """
        Update option flow aggregate with a new option trade.
        
        Args:
            trade: OptionTrade event
            delta: Greek delta (from cache or enricher)
            gamma: Greek gamma (from cache or enricher)
        """
        key = (trade.strike, trade.right, trade.exp_date)
        
        if key not in self.option_flows:
            self.option_flows[key] = OptionFlowAggregate(
                strike=trade.strike,
                right=trade.right,
                exp_date=trade.exp_date
            )
        
        agg = self.option_flows[key]
        
        # Compute flows
        premium = trade.price * trade.size * 100  # contract multiplier
        
        # Dealer gamma transfer (customer buys = dealer sells gamma)
        customer_sign = trade.aggressor.value  # +1 BUY, -1 SELL, 0 MID
        delta_notional = customer_sign * trade.size * delta * 100
        gamma_notional = customer_sign * trade.size * gamma * 100
        dealer_gamma_change = -gamma_notional  # dealer takes opposite side
        
        # Update aggregate
        agg.cumulative_volume += trade.size
        agg.cumulative_premium += premium
        agg.net_delta_flow += delta_notional
        agg.net_gamma_flow += dealer_gamma_change  # net DEALER gamma
        agg.last_price = trade.price
        agg.last_timestamp_ns = trade.ts_event_ns
        agg.delta = delta
        agg.gamma = gamma
    
    def integrate_flow_snapshot(self, flow_snapshot: Dict[str, Any]):
        """
        Integrate state from existing flow_aggregator.py snapshot.
        
        Compatible with flow_aggregator.get_snapshot() output.
        
        Args:
            flow_snapshot: Dict[ticker, Dict] from FlowAggregator
        """
        for ticker, stats in flow_snapshot.items():
            # Parse ticker to extract strike/right/exp
            # Format: O:SPY251216C00676000
            try:
                if not ticker.startswith("O:"):
                    continue
                
                # Parse suffix (last 15 chars: YYMMDD + C/P + 8-digit strike)
                suffix = ticker[-15:]
                exp_yy = suffix[0:2]
                exp_mm = suffix[2:4]
                exp_dd = suffix[4:6]
                exp_date = f"20{exp_yy}-{exp_mm}-{exp_dd}"
                right = suffix[6]
                strike = float(suffix[7:]) / 1000.0
                
                key = (strike, right, exp_date)
                
                # Create or update aggregate
                if key not in self.option_flows:
                    self.option_flows[key] = OptionFlowAggregate(
                        strike=strike,
                        right=right,
                        exp_date=exp_date
                    )
                
                agg = self.option_flows[key]
                agg.cumulative_volume = stats.get("cumulative_volume", 0)
                agg.cumulative_premium = stats.get("cumulative_premium", 0.0)
                agg.net_delta_flow = stats.get("net_delta_flow", 0.0)
                agg.net_gamma_flow = stats.get("net_gamma_flow", 0.0)
                agg.last_price = stats.get("last_price", 0.0)
                agg.last_timestamp_ns = stats.get("last_timestamp", 0) * 1_000_000  # ms -> ns
                agg.delta = stats.get("delta", 0.0)
                agg.gamma = stats.get("gamma", 0.0)
                
            except Exception as e:
                # Skip malformed tickers
                continue
    
    # ========== Window queries ==========
    
    def get_trades_in_window(
        self, 
        ts_now_ns: int, 
        window_seconds: float,
        price_band: Optional[Tuple[float, float]] = None
    ) -> List[TimestampedTrade]:
        """
        Get trades within time window and optional price band.
        
        Args:
            ts_now_ns: Current timestamp (Unix ns)
            window_seconds: Lookback window
            price_band: Optional (min_price, max_price) filter
            
        Returns:
            List of trades, oldest first
        """
        trades = self.trades_buffer.get_window(ts_now_ns, window_seconds)
        
        if price_band is not None:
            min_px, max_px = price_band
            trades = [t for t in trades if min_px <= t.price <= max_px]
        
        return trades
    
    def get_quotes_in_window(
        self, 
        ts_now_ns: int, 
        window_seconds: float
    ) -> List[TimestampedQuote]:
        """
        Get quotes within time window.
        
        Args:
            ts_now_ns: Current timestamp (Unix ns)
            window_seconds: Lookback window
            
        Returns:
            List of quotes, oldest first
        """
        return self.quotes_buffer.get_window(ts_now_ns, window_seconds)
    
    def get_trades_near_level(
        self,
        ts_now_ns: int,
        window_seconds: float,
        level_price: float,
        band_dollars: float
    ) -> List[TimestampedTrade]:
        """
        Get trades near a specific level within window.
        
        Args:
            ts_now_ns: Current timestamp
            window_seconds: Lookback window
            level_price: Level price
            band_dollars: Price band around level (e.g., ±0.10)
            
        Returns:
            List of trades within [level - band, level + band]
        """
        min_px = level_price - band_dollars
        max_px = level_price + band_dollars
        return self.get_trades_in_window(ts_now_ns, window_seconds, (min_px, max_px))
    
    def get_option_flows_near_level(
        self,
        level_price: float,
        strike_range: float,
        exp_date_filter: Optional[str] = None
    ) -> List[OptionFlowAggregate]:
        """
        Get option flow aggregates for strikes near a level.
        
        Args:
            level_price: Level price
            strike_range: Strike range around level (e.g., ±2.0)
            exp_date_filter: Optional expiration date filter (ISO format)
            
        Returns:
            List of OptionFlowAggregate objects
        """
        min_strike = level_price - strike_range
        max_strike = level_price + strike_range
        
        results = []
        for (strike, right, exp_date), agg in self.option_flows.items():
            if min_strike <= strike <= max_strike:
                if exp_date_filter is None or exp_date == exp_date_filter:
                    results.append(agg)
        
        return results
    
    # ========== Spot and derived values ==========
    
    def get_spot(self) -> Optional[float]:
        """Get current SPY spot price (from last trade)."""
        return self.last_trade.price if self.last_trade else None
    
    def get_bid_ask(self) -> Optional[Tuple[float, float]]:
        """Get current bid/ask (from last quote)."""
        if self.last_quote:
            return (self.last_quote.bid_px, self.last_quote.ask_px)
        return None
    
    def get_vwap(self) -> Optional[float]:
        """Get session VWAP."""
        return self._vwap
    
    def get_session_high(self) -> Optional[float]:
        """Get session high."""
        return self._session_high
    
    def get_session_low(self) -> Optional[float]:
        """Get session low."""
        return self._session_low
    
    def get_current_ts_ns(self) -> int:
        """Get current timestamp in Unix nanoseconds."""
        return time.time_ns()
    
    # ========== Debug and introspection ==========
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """Get buffer sizes for debugging."""
        return {
            "trades_buffer_size": len(self.trades_buffer),
            "quotes_buffer_size": len(self.quotes_buffer),
            "option_flows_count": len(self.option_flows)
        }
    
    def reset(self):
        """Clear all state (useful for testing or session reset)."""
        self.last_trade = None
        self.last_quote = None
        self.trades_buffer = RingBuffer(max_window_seconds=120.0)
        self.quotes_buffer = RingBuffer(max_window_seconds=120.0)
        self.option_flows.clear()
        self._vwap = None
        self._vwap_volume = 0
        self._session_high = None
        self._session_low = None

