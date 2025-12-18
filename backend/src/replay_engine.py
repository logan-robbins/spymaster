import asyncio
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Any
from polygon import RESTClient
from zoneinfo import ZoneInfo
from .strike_manager import StrikeManager
from .historical_cache import HistoricalDataCache

class MockMessage:
    def __init__(self, data: Dict[str, Any]):
        # Data Lake Schema: ticker, price, size, timestamp
        self.symbol = data.get('ticker', '')
        self.price = data.get('price', 0.0)
        self.size = data.get('size', 0)
        self.timestamp = data.get('timestamp', 0) # ms
        self.conditions = [] 
        
        # Enriched Data Support (if present in Parquet)
        greeks = {}
        if 'delta' in data:
            greeks['delta'] = data['delta']
        if 'gamma' in data:
            greeks['gamma'] = data['gamma']
        
        # Create object-like access for greeks if populated
        if greeks:
            class GreekObj:
                pass
            g_obj = GreekObj()
            g_obj.delta = greeks.get('delta', 0.0)
            g_obj.gamma = greeks.get('gamma', 0.0)
            self.greeks = g_obj
        else:
            self.greeks = None
            
        self.ev = 'T' 

class ReplayEngine:
    """
    Unified Replay Engine.
    Reads strictly from Data Lake via HistoricalDataCache.
    """
    def __init__(self, api_key: str, queue: asyncio.Queue, strike_manager: StrikeManager):
        self.client = RESTClient(api_key=api_key)
        self.queue = queue
        self.strike_manager = strike_manager
        self.running = False
        self.cache = HistoricalDataCache() # Defaults to data/raw/flow

    async def run(self, minutes_back: int = 10, speed: float = 1.0):
        print(f"I am a Replay Engine! Preparing to replay last {minutes_back} minutes from Data Lake...")
        self.running = True
        
        # 1. Discovery
        latest_date = self.cache.get_latest_available_date()
        if not latest_date:
            print("❌ No data found in Data Lake (data/raw/flow). Run live mode first!")
            return
            
        print(f"📅 Latest Data Lake partition: {latest_date}")
        
        # 2. Timing
        # We target the end of the trading day for that date (4:00 PM ET)
        # Verify: If latest_date is TODAY and market is OPEN, we should replay up to NOW?
        # User requested replay of historical. 
        # If it's today, we might want to replay 'so far'?
        # The logic below assumes 'market_close'.
        
        market_close = datetime.combine(latest_date, time(16, 0), tzinfo=ZoneInfo("America/New_York"))
        
        # Adjust for 'Today' logic if needed
        now = datetime.now(ZoneInfo("America/New_York"))
        if latest_date == now.date() and now < market_close:
            print("⚠️ Replaying TODAY's data (ongoing session).")
            market_close = now
            
        start_time = market_close - timedelta(minutes=minutes_back)
        end_time = market_close
        
        print(f"📦 Replaying window: {start_time.time()} - {end_time.time()}")
        
        # 3. Fetch Data (All Tickers)
        # We pass None for ticker to get EVERYTHING in that window
        all_trades = self.cache.get_cached_trades(
            ticker=None, 
            trade_date=latest_date,
            start_time=start_time,
            end_time=end_time
        )
        
        if not all_trades:
            print("❌ No trades found in the requested window.")
            return

        print(f"🎬 Loaded {len(all_trades)} trades. Starting Replay at {speed}x speed...")
        
        # 4. Replay Loop
        start_ts = all_trades[0]['timestamp']
        replay_start_wall_clock = datetime.now().timestamp() * 1000 # ms
        
        for trade in all_trades:
            if not self.running:
                break
                
            trade_ts = trade['timestamp']
            offset = trade_ts - start_ts
            
            # Scaled Delay
            target_delay_ms = offset / speed
            elapsed_ms = (datetime.now().timestamp() * 1000) - replay_start_wall_clock
            
            wait_ms = target_delay_ms - elapsed_ms
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000.0)
            
            # Emit
            msg = MockMessage(trade)
            await self.queue.put(msg)
            
        print("✅ Replay Complete.")
