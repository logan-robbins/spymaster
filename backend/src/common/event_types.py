"""
Canonical event types for internal message bus and storage.

All events include:
- ts_event_ns: event time (from vendor) in Unix nanoseconds UTC
- ts_recv_ns: receive time (by our system) in Unix nanoseconds UTC
- source: where the event came from (direct_feed, replay, sim)

These dataclasses are the shared contract between ingestion, engines, and storage.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class EventSource(Enum):
    """Where did this event originate?"""
    REPLAY = "replay"
    SIM = "sim"
    DIRECT_FEED = "direct_feed"


class Aggressor(Enum):
    """Trade aggressor side"""
    BUY = 1  # lifted ask
    SELL = -1  # hit bid
    MID = 0  # unknown / mid


@dataclass
class OptionTrade:
    """
    Normalized option trade event.
    
    Schema: options.trades.v1
    """
    ts_event_ns: int
    ts_recv_ns: int
    source: EventSource
    underlying: str  # e.g., "ES"
    option_symbol: str  # vendor symbol (e.g., ES option symbol)
    exp_date: str  # ISO date YYYY-MM-DD
    strike: float
    right: str  # 'C' or 'P'
    price: float
    size: int
    opt_bid: Optional[float] = None  # option BBO if available
    opt_ask: Optional[float] = None
    aggressor: Aggressor = Aggressor.MID  # inferred from option BBO or tick rule
    conditions: Optional[List[int]] = None
    seq: Optional[int] = None


@dataclass
class FuturesTrade:
    """
    Normalized futures trade (e.g., ES).
    
    Schema: futures.trades.v1
    Optional for v1, used when ES L2 barrier physics is enabled.
    """
    ts_event_ns: int
    ts_recv_ns: int
    source: EventSource
    symbol: str  # e.g., "ES" or "ESH6"
    price: float
    size: int
    aggressor: Aggressor = Aggressor.MID
    exchange: Optional[str] = None
    conditions: Optional[List[int]] = None
    seq: Optional[int] = None


@dataclass
class BidAskLevel:
    """Single bid or ask level in MBP"""
    bid_px: float
    bid_sz: int
    ask_px: float
    ask_sz: int


@dataclass
class MBP10:
    """
    Market-by-price L2 snapshot (top 10 levels).
    
    Schema: futures.mbp10.v1
    Optional for v1, used when ES MBP-10 barrier physics is enabled.
    """
    ts_event_ns: int
    ts_recv_ns: int
    source: EventSource
    symbol: str
    levels: List[BidAskLevel]  # 10 levels
    is_snapshot: bool = False  # vs incremental update
    seq: Optional[int] = None
