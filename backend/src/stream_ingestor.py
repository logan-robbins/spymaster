import asyncio
import time
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage
from typing import Callable, List, Set
from .strike_manager import StrikeManager
from .event_types import StockTrade, StockQuote, OptionTrade, EventSource, Aggressor
import logging

class StreamIngestor:
    """
    Manages the WebSocket connection to Polygon without blocking.
    Feeds an asyncio.Queue for processing with normalized event types.
    """
    def __init__(self, api_key: str, queue: asyncio.Queue, strike_manager: StrikeManager):
        self.api_key = api_key
        self.queue = queue
        self.strike_manager = strike_manager
        self.running = False

        # Options Client (Flow)
        self.client = WebSocketClient(
            api_key=self.api_key,
            market='options',
            subscriptions=[],
            verbose=False
        )

        # Stocks Client (Price/ATM tracking + Quotes for barrier engine)
        self.stock_client = WebSocketClient(
            api_key=self.api_key,
            market='stocks',
            subscriptions=["T.SPY", "Q.SPY"],  # Trades + Quotes for SPY
            verbose=False
        )

    async def run_async(self):
        self.running = True
        print("🔌 StreamIngestor: Connecting to Polygon Options + Stocks...")
        
        # Run both clients concurrently
        await asyncio.gather(
            self.client.connect(self.handle_msg_async),
            self.stock_client.connect(self.handle_stock_msg_async)
        )

    async def handle_msg_async(self, msgs: List[WebSocketMessage]):
        """Handle Options Trades - normalize to OptionTrade events"""
        ts_recv_ns = time.time_ns()
        for m in msgs:
            # Parse option ticker: O:SPY251216C00676000
            ticker = getattr(m, "symbol", "")
            if not ticker or not ticker.startswith("O:"):
                # Forward raw message for other event types
                await self.queue.put(m)
                continue

            try:
                # Parse: O:SPY + YYMMDD + C/P + 8-digit strike
                suffix = ticker[-15:]  # YYMMDDCXXXXXXXX
                exp_yy = suffix[:2]
                exp_mm = suffix[2:4]
                exp_dd = suffix[4:6]
                right = suffix[6]
                strike_str = suffix[7:]
                strike = float(strike_str) / 1000.0
                exp_date = f"20{exp_yy}-{exp_mm}-{exp_dd}"

                # Get timestamp from message (Polygon uses milliseconds)
                ts_event_ms = getattr(m, "timestamp", 0) or getattr(m, "sip_timestamp", 0)
                ts_event_ns = ts_event_ms * 1_000_000 if ts_event_ms else ts_recv_ns

                normalized = OptionTrade(
                    ts_event_ns=ts_event_ns,
                    ts_recv_ns=ts_recv_ns,
                    source=EventSource.POLYGON_WS,
                    underlying="SPY",
                    option_symbol=ticker,
                    exp_date=exp_date,
                    strike=strike,
                    right=right,
                    price=float(getattr(m, "price", 0.0)),
                    size=int(getattr(m, "size", 0)),
                    opt_bid=getattr(m, "bid", None),
                    opt_ask=getattr(m, "ask", None),
                    aggressor=Aggressor.MID,  # Inferred later if BBO available
                    conditions=getattr(m, "conditions", None),
                    seq=getattr(m, "sequence_number", None)
                )
                await self.queue.put(normalized)
            except Exception as e:
                # Fallback: forward raw message
                logging.warning(f"Failed to normalize option trade {ticker}: {e}")
                await self.queue.put(m)

    async def handle_stock_msg_async(self, msgs: List[WebSocketMessage]):
        """Handle Stock Trades and Quotes (SPY) - normalize to StockTrade/StockQuote"""
        ts_recv_ns = time.time_ns()
        for m in msgs:
            event_type = getattr(m, "event_type", None)

            # Handle Trade messages (T.SPY)
            if hasattr(m, 'price') and m.price and event_type != 'Q':
                price = m.price

                # Check for dynamic strike update
                if self.strike_manager.should_update(price):
                    print(f"⚡️ Dynamic Strike Update triggered at ${price}")
                    add, remove = self.strike_manager.get_target_strikes(price)
                    self.update_subs(add, remove)

                # Get timestamp (Polygon uses milliseconds)
                ts_event_ms = getattr(m, "timestamp", 0) or getattr(m, "sip_timestamp", 0)
                ts_event_ns = ts_event_ms * 1_000_000 if ts_event_ms else ts_recv_ns

                normalized = StockTrade(
                    ts_event_ns=ts_event_ns,
                    ts_recv_ns=ts_recv_ns,
                    source=EventSource.POLYGON_WS,
                    symbol=getattr(m, "symbol", "SPY"),
                    price=float(price),
                    size=int(getattr(m, "size", 0)),
                    exchange=getattr(m, "exchange", None),
                    conditions=getattr(m, "conditions", None),
                    seq=getattr(m, "sequence_number", None)
                )
                await self.queue.put(normalized)

            # Handle Quote messages (Q.SPY)
            elif hasattr(m, 'bid_price') and hasattr(m, 'ask_price'):
                ts_event_ms = getattr(m, "timestamp", 0) or getattr(m, "sip_timestamp", 0)
                ts_event_ns = ts_event_ms * 1_000_000 if ts_event_ms else ts_recv_ns

                normalized = StockQuote(
                    ts_event_ns=ts_event_ns,
                    ts_recv_ns=ts_recv_ns,
                    source=EventSource.POLYGON_WS,
                    symbol=getattr(m, "symbol", "SPY"),
                    bid_px=float(getattr(m, "bid_price", 0.0)),
                    ask_px=float(getattr(m, "ask_price", 0.0)),
                    bid_sz=int(getattr(m, "bid_size", 0)),
                    ask_sz=int(getattr(m, "ask_size", 0)),
                    bid_exch=getattr(m, "bid_exchange", None),
                    ask_exch=getattr(m, "ask_exchange", None),
                    seq=getattr(m, "sequence_number", None)
                )
                await self.queue.put(normalized)

    def update_subs(self, add: List[str], remove: List[str]):
        if add:
            self.client.subscribe(*add)
            print(f"Subscribed to {len(add)} tickers.")
        if remove:
            self.client.unsubscribe(*remove)
            print(f"Unsubscribed from {len(remove)} tickers.")
