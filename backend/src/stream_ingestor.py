import asyncio
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage
from typing import Callable, List, Set
from .strike_manager import StrikeManager
import logging

class StreamIngestor:
    """
    Manages the WebSocket connection to Polygon without blocking.
    Feeds an asyncio.Queue for processing.
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
        
        # Stocks Client (Price/ATM tracking)
        self.stock_client = WebSocketClient(
            api_key=self.api_key,
            market='stocks',
            subscriptions=["T.SPY"], # Trade ticks for SPY
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
        """Handle Options Trades"""
        for m in msgs:
            await self.queue.put(m)

    async def handle_stock_msg_async(self, msgs: List[WebSocketMessage]):
        """Handle Stock Trades (SPY) for ATM updates"""
        for m in msgs:
            # We expect T.SPY messages
            # m.symbol should be 'SPY'
            # m.price should be the trade price
            if hasattr(m, 'price') and m.price:
                price = m.price
                
                # Check for dynamic strike update
                if self.strike_manager.should_update(price):
                    print(f"⚡️ Dynamic Strike Update triggered at ${price}")
                    add, remove = self.strike_manager.get_target_strikes(price)
                    self.update_subs(add, remove)
            
            # Optional: Forward SPY price to frontend via queue?
            # If we want the frontend to display EXACT underlying price.
            # But FlowAggregator expects Options tickers.
            # We can send it if FlowAggregator can handle it. 
            # For now, let's keep it clean and NOT send stock trades to FlowAggregator to avoid parsing errors.
            # The Strike update will generate new Options subs, which will generate new Options flow.

    def update_subs(self, add: List[str], remove: List[str]):
        if add:
            self.client.subscribe(*add)
            print(f"Subscribed to {len(add)} tickers.")
        if remove:
            self.client.unsubscribe(*remove)
            print(f"Unsubscribed from {len(remove)} tickers.")
