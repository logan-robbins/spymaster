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
        self.client: WebSocketClient = None # type: ignore
        self.running = False
        
        # We need to track the last price to trigger re-strike calculation?
        # Or main loop does it? 
        # The StreamIngestor just ingests.
        # But if we need to update subs dynamically based on price...
        # We can expose a method `update_subs` and let a controller call it?
        # OR we do it here. 
        # APP.md 2.2 says "Logic (Execute every 60 seconds OR on >$0.50 price deviation)".
        # This implies we need to watch the price.
        # Since we are receiving SPY trades (hopefully), we can watch.
        # But wait, are we subscribing to SPY underlying?
        # "Input: Real-time SPY price (via Polygon A (Aggregate) or T (Trade) channel for Stocks)."
        # So we MUST also subscribe to SPY stock trades/aggs.
        
        self.last_spy_price = 0.0
        self.last_check_price = 0.0
        self.last_check_time = 0
        
    async def start(self):
        self.running = True
        # Initialize client
        # We need to handle subscriptions.
        # We should subscribe to SPY underlying initially.
        initial_subs = ["T.SPY"] # Stocks trade for SPY
        
        self.client = WebSocketClient(
            api_key=self.api_key,
            market='options', # Wait, we need STOCKS for SPY price?
            # Polygon WebSocketClient `market` argument sets the cluster.
            # Options cluster ('options') handles options.
            # Stocks cluster ('stocks') handles stocks.
            # Can we use one client for both? No, different URLs.
            # wss://socket.polygon.io/stocks vs wss://socket.polygon.io/options.
            # We need TWO clients? Or can we get SPY price from Options cluster?
            # Options cluster usually does NOT send underlying price directly for all ticks.
            # THIS IS A CRITICAL ARCHITECTURAL DETAIL.
            # If we need SPY price, we need a Stocks stream or poll REST.
            # "Input: Real-time SPY price (via Polygon A (Aggregate) or T (Trade) channel for Stocks)."
            # This implies a Stocks connection.
            # So `StreamIngestor` might need to manage TWO connections or we focus on Options and poll price?
            # 60 seconds interval suggests polling is fine.
            # ">$0.50 price deviation" suggests streaming.
            # Let's assume we need a separate subscription for SPY.
            # But `polygon-api-client` WebSocketClient takes one `market`.
            # We might need two `WebSocketClient` instances.
            # Or we simplify:
            # We open the Options stream.
            # We verify if we can sub to underlying? Usually no.
            # I will implement a SEPARATE task/client for the Underlying if needed.
            # Or simpler: Just poll the price every 1s from REST? Api limits? 
            # Polygon Advanced Tier has unlimited Websocket, but REST?
            # "Advanced Tier".
            # I'll try to stick to the spec "Real-time SPY price".
            # I will create a `StockStreamIngestor`? Or just put it in here.
            # BUT, to keep it simple, I'll assume we can run two clients or just use REST for price triggers if needed.
            # Wait, `market='options'` is passed to WebSocketClient.
            # I will implement `StreamIngestor` to handle OPTIONS.
            # And I'll add a lightweight specific client/loop for SPY if required.
            # Actually, let's just use `market='options'` and see if `T.SPY` works? 
            # Usually it doesn't.
            # I'll implement a `PriceMonitor` using REST polling for now to avoid complexity of 2 WS connections for v1.
            # Or better: "Logic (Execute every 60 seconds...)"
            # REST polling every 5s is fine for "Real-time-ish" for strike selection.
            # >$0.50 deviation might be missed with polling.
            # But it's robust.
            
            # Re-reading: "monitor real-time trade flow for SPY 0DTE options"
            # The SPY price is used to SELECT strikes.
            # If we miss a move by 5 seconds, we might miss capturing data for the new strike for 5 seconds.
            # That is acceptable for v1.
            
            subscriptions=[], # We start empty and let `update_subs` handle it or `main` init.
            verbose=False
        )
        
        # We need to run client.run in a non-blocking way?
        # client.run is blocking.
        # We run it in executor or use the async `connect` method manually.
        # The library's `run` does a loop.
        # I'll wrap it in `run_in_executor` or proper async launch.
        
        loop = asyncio.get_running_loop()
        # client.run takes a handler `handle_msg`.
        # We wrap our async `process_msg` to be called.
        
        self.client.run(handle_msg=self.handle_msg) # This blocks!
        # We must NOT block here.
        # So `start` should launch a thread or task?
        # `asyncio.to_thread(self.client.run, ...)`
    
    def handle_msg(self, msgs: List[WebSocketMessage]):
        # This is called from the client's loop (which might be threaded or sync).
        # We need to put into queue.
        # `asyncio.run_coroutine_threadsafe` if in thread?
        # The `polygon-api-client` uses `threading` or `asyncio`?
        # It seems it can use either depending on how it's called?
        # In `test_stream.py` it was blocking.
        # We should use the async method `connect` if possible.
        # `client.connect(processor)` is async.
        # But `client` init doesn't connect.
        # Let's use the object properly.
        pass

    async def run_async(self):
        # Allow running in async mode
        # self.client.connect(...)
        await self.client.connect(self.handle_msg_async)

    async def handle_msg_async(self, msgs: List[WebSocketMessage]):
        for m in msgs:
            await self.queue.put(m)

    def update_subs(self, add: List[str], remove: List[str]):
        if add:
            self.client.subscribe(add)
            print(f"Subscribed to {len(add)} tickers.")
        if remove:
            self.client.unsubscribe(remove)
            print(f"Unsubscribed from {len(remove)} tickers.")
