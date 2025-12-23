"""
MarketState: central state store for market data and ring buffers.

Agent C deliverable per §12 of PLAN.md.

This module maintains:
- ES futures MBP-10 snapshots and trades (for barrier physics)
- SPY options flow aggregates (for fuel engine, from Polygon API)
- Derived values (VWAP, session high/low from ES)

All timestamps are Unix nanoseconds (UTC).
"""

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import time

from src.common.event_types import FuturesTrade, MBP10, OptionTrade, Aggressor
from src.common.price_converter import PriceConverter
from src.common.config import CONFIG


@dataclass
class TimestampedESTrade:
    """ES trade with timestamp for ring buffer storage."""
    ts_event_ns: int
    price: float
    size: int
    aggressor: Aggressor


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
    abs_gamma_exposure: float = 0.0  # Absolute gamma regardless of direction
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
    - ES MBP-10 snapshots (for barrier physics)
    - ES trades ring buffer
    - SPY option flow aggregates (from Polygon API)

    Thread-safety: Not thread-safe by default. Use from single event loop.
    """

    def __init__(self, max_buffer_window_seconds: float = 120.0):
        # ========== Price Converter (ES <-> SPY) ==========
        self.price_converter = PriceConverter()

        # ========== ES Futures State ==========
        # Last-known MBP-10 snapshot
        self.es_mbp10_snapshot: Optional[MBP10] = None

        # MBP-10 history for flow computation
        self.es_mbp10_buffer = RingBuffer(max_window_seconds=max_buffer_window_seconds)

        # ES trades ring buffer
        self.es_trades_buffer = RingBuffer(max_window_seconds=max_buffer_window_seconds)

        # Last ES trade (for spot price)
        self.last_es_trade: Optional[TimestampedESTrade] = None

        # ========== Option flow aggregates (SPY options from Polygon) ==========
        # Key: (strike, right, exp_date) -> OptionFlowAggregate
        self.option_flows: Dict[Tuple[float, str, str], OptionFlowAggregate] = {}

        # ========== Derived state (updated on demand) ==========
        # Note: VWAP and session high/low are stored in ES terms, converted on access
        self._vwap: Optional[float] = None
        self._vwap_volume: int = 0
        self._session_high: Optional[float] = None
        self._session_low: Optional[float] = None

    # ========== ES MBP-10 updates ==========

    def update_es_mbp10(self, mbp: MBP10):
        """
        Update market state with a new ES MBP-10 snapshot.

        Args:
            mbp: MBP10 event from DBN ingestor
        """
        self.es_mbp10_snapshot = mbp
        self.es_mbp10_buffer.append(mbp)

    def get_es_mbp10_snapshot(self) -> Optional[MBP10]:
        """Get current ES MBP-10 snapshot."""
        return self.es_mbp10_snapshot

    def get_es_mbp10_in_window(
        self,
        ts_now_ns: int,
        window_seconds: float
    ) -> List[MBP10]:
        """
        Get ES MBP-10 snapshots within time window.

        Args:
            ts_now_ns: Current timestamp (Unix ns)
            window_seconds: Lookback window

        Returns:
            List of MBP10 snapshots, oldest first
        """
        return self.es_mbp10_buffer.get_window(ts_now_ns, window_seconds)

    # ========== ES trades updates ==========

    def update_es_trade(self, trade: FuturesTrade):
        """
        Update market state with a new ES futures trade.

        Args:
            trade: FuturesTrade event from DBN ingestor
        """
        timestamped = TimestampedESTrade(
            ts_event_ns=trade.ts_event_ns,
            price=trade.price,
            size=trade.size,
            aggressor=trade.aggressor
        )

        self.last_es_trade = timestamped
        self.es_trades_buffer.append(timestamped)

        # Update price converter with ES price
        self.price_converter.update_es_price(trade.price)

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

    def get_es_trades_in_window(
        self,
        ts_now_ns: int,
        window_seconds: float,
        price_band: Optional[Tuple[float, float]] = None
    ) -> List[TimestampedESTrade]:
        """
        Get ES trades within time window and optional price band.

        Args:
            ts_now_ns: Current timestamp (Unix ns)
            window_seconds: Lookback window
            price_band: Optional (min_price, max_price) filter

        Returns:
            List of trades, oldest first
        """
        trades = self.es_trades_buffer.get_window(ts_now_ns, window_seconds)

        if price_band is not None:
            min_px, max_px = price_band
            trades = [t for t in trades if min_px <= t.price <= max_px]

        return trades

    def get_es_trades_near_level(
        self,
        ts_now_ns: int,
        window_seconds: float,
        level_price: float,
        band_dollars: float
    ) -> List[TimestampedESTrade]:
        """
        Get ES trades near a specific level within window.

        Args:
            ts_now_ns: Current timestamp
            window_seconds: Lookback window
            level_price: Level price
            band_dollars: Price band around level (e.g., ±0.50)

        Returns:
            List of trades within [level - band, level + band]
        """
        min_px = level_price - band_dollars
        max_px = level_price + band_dollars
        return self.get_es_trades_in_window(ts_now_ns, window_seconds, (min_px, max_px))

    # ========== Option updates (SPY options from Polygon) ==========

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

    # ========== Spot and derived values (SPY-equivalent) ==========

    def get_spot(self) -> Optional[float]:
        """
        Get current SPY-equivalent spot price (converted from ES trade).

        Returns:
            SPY-equivalent spot price (e.g., 687.0), or None if no ES data
        """
        if self.last_es_trade:
            return self.price_converter.es_to_spy(self.last_es_trade.price)
        return None

    def get_bid_ask(self) -> Optional[Tuple[float, float]]:
        """
        Get current SPY-equivalent bid/ask (converted from ES MBP-10).

        Returns:
            (bid, ask) tuple in SPY terms, or None if no MBP-10 data
        """
        if self.es_mbp10_snapshot and self.es_mbp10_snapshot.levels:
            best = self.es_mbp10_snapshot.levels[0]
            return (
                self.price_converter.es_to_spy(best.bid_px),
                self.price_converter.es_to_spy(best.ask_px)
            )
        return None

    def get_vwap(self) -> Optional[float]:
        """Get session VWAP (SPY-equivalent)."""
        if self._vwap:
            return self.price_converter.es_to_spy(self._vwap)
        return None

    def get_session_high(self) -> Optional[float]:
        """Get session high (SPY-equivalent)."""
        if self._session_high:
            return self.price_converter.es_to_spy(self._session_high)
        return None

    def get_session_low(self) -> Optional[float]:
        """Get session low (SPY-equivalent)."""
        if self._session_low:
            return self.price_converter.es_to_spy(self._session_low)
        return None

    # ========== Raw ES accessors (for engines that need ES prices) ==========

    def get_es_spot(self) -> Optional[float]:
        """Get raw ES spot price (not converted)."""
        return self.last_es_trade.price if self.last_es_trade else None

    def get_es_bid_ask(self) -> Optional[Tuple[float, float]]:
        """Get raw ES bid/ask (not converted)."""
        if self.es_mbp10_snapshot and self.es_mbp10_snapshot.levels:
            best = self.es_mbp10_snapshot.levels[0]
            return (best.bid_px, best.ask_px)
        return None

    def get_es_vwap(self) -> Optional[float]:
        """Get raw ES session VWAP (not converted)."""
        return self._vwap

    def get_es_session_high(self) -> Optional[float]:
        """Get raw ES session high (not converted)."""
        return self._session_high

    def get_es_session_low(self) -> Optional[float]:
        """Get raw ES session low (not converted)."""
        return self._session_low

    def get_current_ts_ns(self) -> int:
        """Get current timestamp in Unix nanoseconds."""
        return time.time_ns()

    # ========== Debug and introspection ==========

    def get_buffer_stats(self) -> Dict[str, int]:
        """Get buffer sizes for debugging."""
        return {
            "es_mbp10_buffer_size": len(self.es_mbp10_buffer),
            "es_trades_buffer_size": len(self.es_trades_buffer),
            "option_flows_count": len(self.option_flows)
        }

    def reset(self):
        """Clear all state (useful for testing or session reset)."""
        self.price_converter = PriceConverter()
        self.es_mbp10_snapshot = None
        self.last_es_trade = None
        self.es_mbp10_buffer = RingBuffer(max_window_seconds=120.0)
        self.es_trades_buffer = RingBuffer(max_window_seconds=120.0)
        self.option_flows.clear()
        self._vwap = None
        self._vwap_volume = 0
        self._session_high = None
        self._session_low = None
