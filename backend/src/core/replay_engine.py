import asyncio
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Any
from polygon import RESTClient
from zoneinfo import ZoneInfo
from src.core.strike_manager import StrikeManager
from src.lake.historical_cache import HistoricalDataCache

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
        print(f"I am a Replay Engine! Preparing to replay from earliest available data...")
        self.running = True
        
        # 1. Discovery - Get earliest available date
        latest_date = self.cache.get_latest_available_date()
        if not latest_date:
            print("❌ No data found in Data Lake (data/raw/flow). Run live mode first!")
            return
            
        print(f"📅 Replaying from date: {latest_date}")
        
        # 2. Timing - Start from beginning of available data for that day
        # Query without time filters to get ALL data from that day
        start_time = None
        end_time = None
        
        print(f"📦 Replaying all data from {latest_date}")
        
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
