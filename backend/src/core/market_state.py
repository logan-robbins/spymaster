"""
MarketState: central state store for market data and ring buffers.

Agent C deliverable per §12 of PLAN.md.

This module maintains:
- ES futures MBP-10 snapshots and trades (for barrier physics)
- ES options flow aggregates (for fuel engine)
- Derived values (VWAP, session high/low from ES)

All timestamps are Unix nanoseconds (UTC).
"""

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import time
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
import math
import numpy as np

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
    """
    strike: float
    right: str  # 'C' or 'P'
    exp_date: str
    cumulative_volume: int = 0
    cumulative_premium: float = 0.0
    net_premium_flow: float = 0.0  # Market Tide: Signed premium flow (Aggressor * Price * Size)
    net_delta_flow: float = 0.0
    net_gamma_flow: float = 0.0
    abs_gamma_exposure: float = 0.0  # Absolute gamma regardless of direction
    last_price: float = 0.0
    last_timestamp_ns: int = 0
    delta: float = 0.0
    gamma: float = 0.0


@dataclass
class MinuteBar:
    """Aggregated 1-minute bar in ES points."""
    start_ts_ns: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class SmaContext:
    """SMA context metrics computed from 2-minute closes."""
    sma_200: Optional[float]
    sma_400: Optional[float]
    sma_200_slope: Optional[float]
    sma_400_slope: Optional[float]
    sma_200_slope_5bar: Optional[float]
    sma_400_slope_5bar: Optional[float]
    sma_spread: Optional[float]


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
    - ES option flow aggregates

    Thread-safety: Not thread-safe by default. Use from single event loop.
    """

    def __init__(self, max_buffer_window_seconds: Optional[float] = None):
        if max_buffer_window_seconds is None:
            max_buffer_window_seconds = max(CONFIG.W_b, CONFIG.CONFIRMATION_WINDOW_SECONDS)
        # ========== Price Converter (ES <-> ES no-op) ==========
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

        # ========== Option flow aggregates (ES options) ==========
        # Key: (strike, right, exp_date) -> OptionFlowAggregate
        self.option_flows: Dict[Tuple[float, str, str], OptionFlowAggregate] = {}

        # ========== Derived state (updated on demand) ==========
        # Note: VWAP and session high/low are stored in ES terms, converted on access
        self._vwap: Optional[float] = None
        self._vwap_volume: int = 0
        self._session_high: Optional[float] = None
        self._session_low: Optional[float] = None

        # ========== Context + history (ES points) ==========
        # Session date (ET) used to reset day-scoped context metrics.
        self._session_date_et: Optional[date] = None

        # Structural levels (computed from trade stream, ES points)
        self._premarket_high: Optional[float] = None
        self._premarket_low: Optional[float] = None
        self._opening_range_high: Optional[float] = None
        self._opening_range_low: Optional[float] = None

        # 1-minute closes for approach context (ES points)
        self._current_minute_start_ns: Optional[int] = None
        self._current_minute_close_spy: Optional[float] = None
        self._current_minute_open_spy: Optional[float] = None
        self._current_minute_high_spy: Optional[float] = None
        self._current_minute_low_spy: Optional[float] = None
        self._minute_closes: deque[Tuple[int, float]] = deque(maxlen=240)  # 4 hours of 1-min closes
        self._minute_bars: deque[MinuteBar] = deque(maxlen=240)

        # 2-minute closes for SMA (ES points)
        self._current_2m_start_ns: Optional[int] = None
        self._current_2m_close_spy: Optional[float] = None
        self._two_minute_closes: deque[float] = deque(maxlen=400)  # Enough for SMA_400
        self._sma_200: Optional[float] = None
        self._sma_400: Optional[float] = None

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

        # ES system: price is already in ES points (no conversion)
        self._update_context_from_trade(trade.ts_event_ns, trade.price)

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

    def _update_context_from_trade(self, ts_event_ns: int, spx_price: float) -> None:
        """
        Update day-scoped context metrics from the ES trade stream.

        Notes:
        - Uses event timestamps (not wall time) so replay is deterministic.
        - Computes structural levels in ES points for the live level universe.
        """
        # --- Day boundary handling (ET) ---
        dt_et = datetime.fromtimestamp(ts_event_ns / 1e9, tz=timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
        session_date = dt_et.date()
        if self._session_date_et != session_date:
            self._session_date_et = session_date
            self._premarket_high = None
            self._premarket_low = None
            self._opening_range_high = None
            self._opening_range_low = None
            self._minute_closes.clear()
            self._minute_bars.clear()
            self._two_minute_closes.clear()
            self._current_minute_start_ns = None
            self._current_minute_close_spy = None
            self._current_minute_open_spy = None
            self._current_minute_high_spy = None
            self._current_minute_low_spy = None
            self._current_2m_start_ns = None
            self._current_2m_close_spy = None
            self._sma_200 = None
            self._sma_400 = None

        # --- Premarket + opening range ---
        minutes_since_midnight = dt_et.hour * 60 + dt_et.minute
        premarket_start = 4 * 60
        market_open = 9 * 60 + 30
        opening_range_end = 9 * 60 + 45

        if premarket_start <= minutes_since_midnight < market_open:
            if self._premarket_high is None or spx_price > self._premarket_high:
                self._premarket_high = spx_price
            if self._premarket_low is None or spx_price < self._premarket_low:
                self._premarket_low = spx_price

        if market_open <= minutes_since_midnight < opening_range_end:
            if self._opening_range_high is None or spx_price > self._opening_range_high:
                self._opening_range_high = spx_price
            if self._opening_range_low is None or spx_price < self._opening_range_low:
                self._opening_range_low = spx_price

        # --- 1-minute bars (approach context + ATR) ---
        one_min_ns = 60 * 1_000_000_000
        minute_start_ns = (ts_event_ns // one_min_ns) * one_min_ns
        if self._current_minute_start_ns is None:
            self._current_minute_start_ns = minute_start_ns
            self._current_minute_open_spy = spx_price
            self._current_minute_high_spy = spx_price
            self._current_minute_low_spy = spx_price
            self._current_minute_close_spy = spx_price
        elif minute_start_ns == self._current_minute_start_ns:
            self._current_minute_close_spy = spx_price
            if self._current_minute_high_spy is None or spx_price > self._current_minute_high_spy:
                self._current_minute_high_spy = spx_price
            if self._current_minute_low_spy is None or spx_price < self._current_minute_low_spy:
                self._current_minute_low_spy = spx_price
        else:
            # Finalize prior minute close
            if (
                self._current_minute_close_spy is not None
                and self._current_minute_open_spy is not None
                and self._current_minute_high_spy is not None
                and self._current_minute_low_spy is not None
            ):
                self._minute_closes.append((self._current_minute_start_ns, self._current_minute_close_spy))
                self._minute_bars.append(MinuteBar(
                    start_ts_ns=self._current_minute_start_ns,
                    open=self._current_minute_open_spy,
                    high=self._current_minute_high_spy,
                    low=self._current_minute_low_spy,
                    close=self._current_minute_close_spy
                ))
            self._current_minute_start_ns = minute_start_ns
            self._current_minute_open_spy = spx_price
            self._current_minute_high_spy = spx_price
            self._current_minute_low_spy = spx_price
            self._current_minute_close_spy = spx_price

        # --- 2-minute closes (SMA levels) ---
        two_min_ns = 120 * 1_000_000_000
        two_min_start_ns = (ts_event_ns // two_min_ns) * two_min_ns
        if self._current_2m_start_ns is None:
            self._current_2m_start_ns = two_min_start_ns
            self._current_2m_close_spy = spx_price
        elif two_min_start_ns == self._current_2m_start_ns:
            self._current_2m_close_spy = spx_price
        else:
            # Finalize prior 2-minute close and update SMAs
            if self._current_2m_close_spy is not None:
                self._two_minute_closes.append(self._current_2m_close_spy)
                closes = list(self._two_minute_closes)
                if len(closes) >= 200:
                    self._sma_200 = sum(closes[-200:]) / 200.0
                if len(closes) >= 400:
                    self._sma_400 = sum(closes[-400:]) / 400.0
            self._current_2m_start_ns = two_min_start_ns
            self._current_2m_close_spy = spx_price

    def get_recent_minute_closes(self, lookback_minutes: int) -> List[Tuple[int, float]]:
        """
        Return the most recent minute closes (ts_ns, price).

        The list is ordered oldest->newest and includes the current in-progress minute close.
        """
        if lookback_minutes <= 0:
            return []

        closes: List[Tuple[int, float]] = list(self._minute_closes)
        if self._current_minute_close_spy is not None and self._current_minute_start_ns is not None:
            closes.append((self._current_minute_start_ns, self._current_minute_close_spy))
        return closes[-lookback_minutes:]

    def get_recent_minute_bars(self, lookback_minutes: int) -> List[MinuteBar]:
        """
        Return the most recent minute bars (ES points), including the current in-progress bar.
        """
        if lookback_minutes <= 0:
            return []

        bars: List[MinuteBar] = list(self._minute_bars)
        if (
            self._current_minute_start_ns is not None
            and self._current_minute_open_spy is not None
            and self._current_minute_high_spy is not None
            and self._current_minute_low_spy is not None
            and self._current_minute_close_spy is not None
        ):
            bars.append(MinuteBar(
                start_ts_ns=self._current_minute_start_ns,
                open=self._current_minute_open_spy,
                high=self._current_minute_high_spy,
                low=self._current_minute_low_spy,
                close=self._current_minute_close_spy
            ))
        return bars[-lookback_minutes:]

    def get_atr(self, window_minutes: Optional[int] = None) -> Optional[float]:
        """
        Compute ATR from recent minute bars.
        """
        if window_minutes is None:
            window_minutes = CONFIG.ATR_WINDOW_MINUTES

        bars = self.get_recent_minute_bars(window_minutes)
        if not bars:
            return None

        trs: List[float] = []
        prev_close = bars[0].close
        for bar in bars:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close)
            )
            trs.append(tr)
            prev_close = bar.close

        window = min(window_minutes, len(trs))
        if window <= 0:
            return None
        return sum(trs[-window:]) / window

    def get_recent_return_std(self, window_seconds: int) -> Optional[float]:
        """
        Compute realized std of minute-to-minute returns over a short window.

        Returns:
            Std of returns in ES points per minute, or None if insufficient data.
        """
        window_minutes = max(1, int(math.ceil(window_seconds / 60)))
        closes = self.get_recent_minute_closes(window_minutes + 1)
        if len(closes) < 3:
            return None
        
        # Extract prices from (ts, price) tuples
        prices = [c[1] for c in closes]
        returns = np.diff(np.array(prices, dtype=np.float64))
        if len(returns) < 2:
            return None
        if len(returns) > window_minutes:
            returns = returns[-window_minutes:]
        return float(np.std(returns))

    def get_sma_at_offset(self, period: int, offset_bars: int) -> Optional[float]:
        closes = self._get_two_minute_closes(include_current=True)
        if not closes:
            return None
        idx = len(closes) - 1 - offset_bars
        return self._sma_at_index(closes, idx, period)

    def get_sma_context(self) -> SmaContext:
        """
        Compute SMA values and slopes from recent 2-minute closes.
        """
        closes = self._get_two_minute_closes(include_current=True)
        if not closes:
            return SmaContext(
                sma_200=None,
                sma_400=None,
                sma_200_slope=None,
                sma_400_slope=None,
                sma_200_slope_5bar=None,
                sma_400_slope_5bar=None,
                sma_spread=None
            )

        idx_now = len(closes) - 1
        sma_200 = self._sma_at_index(closes, idx_now, 200)
        if sma_200 is None:
            sma_200 = self._sma_200
        sma_400 = self._sma_at_index(closes, idx_now, 400)
        if sma_400 is None:
            sma_400 = self._sma_400

        slope_minutes = max(1, CONFIG.SMA_SLOPE_WINDOW_MINUTES)
        slope_bars = max(1, int(round(slope_minutes / 2)))
        idx_prev = idx_now - slope_bars

        sma_200_prev = self._sma_at_index(closes, idx_prev, 200)
        sma_400_prev = self._sma_at_index(closes, idx_prev, 400)

        sma_200_slope = None
        sma_400_slope = None
        if sma_200 is not None and sma_200_prev is not None:
            sma_200_slope = (sma_200 - sma_200_prev) / slope_minutes
        if sma_400 is not None and sma_400_prev is not None:
            sma_400_slope = (sma_400 - sma_400_prev) / slope_minutes

        slope_short_bars = max(1, CONFIG.SMA_SLOPE_SHORT_BARS)
        slope_short_minutes = slope_short_bars * 2
        idx_prev_short = idx_now - slope_short_bars

        sma_200_prev_short = self._sma_at_index(closes, idx_prev_short, 200)
        sma_400_prev_short = self._sma_at_index(closes, idx_prev_short, 400)

        sma_200_slope_5bar = None
        sma_400_slope_5bar = None
        if sma_200 is not None and sma_200_prev_short is not None:
            sma_200_slope_5bar = (sma_200 - sma_200_prev_short) / slope_short_minutes
        if sma_400 is not None and sma_400_prev_short is not None:
            sma_400_slope_5bar = (sma_400 - sma_400_prev_short) / slope_short_minutes

        sma_spread = None
        if sma_200 is not None and sma_400 is not None:
            sma_spread = sma_200 - sma_400

        return SmaContext(
            sma_200=sma_200,
            sma_400=sma_400,
            sma_200_slope=sma_200_slope,
            sma_400_slope=sma_400_slope,
            sma_200_slope_5bar=sma_200_slope_5bar,
            sma_400_slope_5bar=sma_400_slope_5bar,
            sma_spread=sma_spread
        )

    def _get_two_minute_closes(self, include_current: bool = True) -> List[float]:
        closes = list(self._two_minute_closes)
        if include_current and self._current_2m_close_spy is not None:
            closes.append(self._current_2m_close_spy)
        return closes

    @staticmethod
    def _sma_at_index(closes: List[float], idx: int, period: int) -> Optional[float]:
        if idx < 0 or idx + 1 < period:
            return None
        start = idx + 1 - period
        window = closes[start: idx + 1]
        if len(window) < period:
            return None
        return sum(window) / period

    def get_premarket_high(self) -> Optional[float]:
        return self._premarket_high

    def get_premarket_low(self) -> Optional[float]:
        return self._premarket_low

    def get_opening_range_high(self) -> Optional[float]:
        return self._opening_range_high

    def get_opening_range_low(self) -> Optional[float]:
        return self._opening_range_low

    def get_sma_200(self) -> Optional[float]:
        return self._sma_200

    def get_sma_400(self) -> Optional[float]:
        return self._sma_400

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

    # ========== Option updates (ES options) ==========

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
        premium = trade.price * trade.size * CONFIG.OPTION_CONTRACT_MULTIPLIER

        # Dealer gamma transfer (customer buys = dealer sells gamma)
        customer_sign = trade.aggressor.value  # +1 BUY, -1 SELL, 0 MID
        delta_notional = customer_sign * trade.size * delta * CONFIG.OPTION_CONTRACT_MULTIPLIER
        gamma_notional = customer_sign * trade.size * gamma * CONFIG.OPTION_CONTRACT_MULTIPLIER
        dealer_gamma_change = -gamma_notional  # dealer takes opposite side

        # Update aggregate
        agg.cumulative_volume += trade.size
        agg.cumulative_premium += premium
        agg.net_premium_flow += (premium * customer_sign)  # Add signed premium flow
        agg.net_delta_flow += delta_notional
        agg.net_gamma_flow += dealer_gamma_change  # net DEALER gamma
        agg.last_price = trade.price
        agg.last_timestamp_ns = trade.ts_event_ns
        agg.delta = delta
        agg.gamma = gamma

    def get_option_flow_snapshot(
        self,
        spot: Optional[float],
        strike_range: Optional[float] = None,
        exp_date_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a lightweight per-strike flow snapshot for the frontend.

        Args:
            spot: Current ES spot (used for strike filtering)
            strike_range: Optional +/- strike filter (ES points)
            exp_date_filter: Optional expiration date filter (YYYY-MM-DD)

        Returns:
            Dict keyed by synthetic ticker with flow metrics.
        """
        if spot is None:
            return {}

        snapshot: Dict[str, Any] = {}

        for (strike, right, exp_date), agg in self.option_flows.items():
            if exp_date_filter and exp_date != exp_date_filter:
                continue
            if strike_range is not None and abs(strike - spot) > strike_range:
                continue

            exp_compact = exp_date.replace("-", "")[2:]
            strike_compact = f"{int(round(strike * 1000)):08d}"
            ticker = f"O:ES{exp_compact}{right}{strike_compact}"

            snapshot[ticker] = {
                "cumulative_volume": agg.cumulative_volume,
                "cumulative_premium": agg.cumulative_premium,
                "net_premium_flow": agg.net_premium_flow,
                "last_price": agg.last_price,
                "net_delta_flow": agg.net_delta_flow,
                "net_gamma_flow": agg.net_gamma_flow,
                "delta": agg.delta,
                "gamma": agg.gamma,
                "strike_price": agg.strike,
                "type": agg.right,
                "expiration": agg.exp_date,
                "last_timestamp": agg.last_timestamp_ns // 1_000_000 if agg.last_timestamp_ns else 0
            }

        return snapshot

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
            strike_range: Strike range around level (ES points)
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

    # ========== Spot and derived values (SPX-equivalent) ==========

    def get_spot(self) -> Optional[float]:
        """
        Get current spot price (ES futures price in index points).

        Returns:
            ES spot price (e.g., 6920.0), or None if no ES data
        """
        if self.last_es_trade:
            return self.last_es_trade.price  # ES futures = ES options (no conversion)
        return None

    def get_bid_ask(self) -> Optional[Tuple[float, float]]:
        """
        Get current bid/ask from ES MBP-10.

        Returns:
            (bid, ask) tuple in ES points, or None if no MBP-10 data
        """
        if self.es_mbp10_snapshot and self.es_mbp10_snapshot.levels:
            best = self.es_mbp10_snapshot.levels[0]
            return (best.bid_px, best.ask_px)  # ES futures = ES options (no conversion)
        return None

    def get_vwap(self) -> Optional[float]:
        """Get session VWAP (ES points)."""
        return self._vwap  # ES futures = ES options (no conversion)

    def get_session_high(self) -> Optional[float]:
        """Get session high (ES points)."""
        return self._session_high  # ES futures = ES options (no conversion)

    def get_session_low(self) -> Optional[float]:
        """Get session low (ES points)."""
        return self._session_low  # ES futures = ES options (no conversion)

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
        """
        Get "now" timestamp in Unix nanoseconds.

        Canonical source is the latest event timestamp so replay stays event-time correct.
        Falls back to wall clock if no events have been seen yet.
        """
        candidates: List[int] = []
        if self.last_es_trade is not None:
            candidates.append(self.last_es_trade.ts_event_ns)
        if self.es_mbp10_snapshot is not None:
            candidates.append(self.es_mbp10_snapshot.ts_event_ns)
        if candidates:
            return max(candidates)
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

        self._session_date_et = None
        self._premarket_high = None
        self._premarket_low = None
        self._opening_range_high = None
        self._opening_range_low = None
        self._minute_closes.clear()
        self._minute_bars.clear()
        self._two_minute_closes.clear()
        self._current_minute_start_ns = None
        self._current_minute_close_spy = None
        self._current_minute_open_spy = None
        self._current_minute_high_spy = None
        self._current_minute_low_spy = None
        self._current_2m_start_ns = None
        self._current_2m_close_spy = None
        self._sma_200 = None
        self._sma_400 = None
