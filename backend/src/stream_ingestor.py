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
        
        self.last_spy_price = 0.0
        self.last_check_price = 0.0
        self.last_check_time = 0
        
        # Initialize client immediately 
        self.client = WebSocketClient(
            api_key=self.api_key,
            market='options', 
            subscriptions=[], 
            verbose=False
        )

    async def start(self):
        self.running = True
        # Client already initialized in __init__
        initial_subs = ["T.SPY"] # Stocks trade for SPY
        pass
    
    def handle_msg(self, msgs: List[WebSocketMessage]):
        pass

    async def run_async(self):
        # Allow running in async mode
        # self.client.connect(...) calls the async loop of the client
        await self.client.connect(self.handle_msg_async)

    async def handle_msg_async(self, msgs: List[WebSocketMessage]):
        for m in msgs:
            await self.queue.put(m)

    def update_subs(self, add: List[str], remove: List[str]):
        if add:
            self.client.subscribe(*add)
            print(f"Subscribed to {len(add)} tickers.")
        if remove:
            self.client.unsubscribe(*remove)
            print(f"Unsubscribed from {len(remove)} tickers.")
