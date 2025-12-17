import asyncio
from datetime import datetime, timedelta, date
from typing import List, Dict, Any
from polygon import RESTClient
from zoneinfo import ZoneInfo
from .strike_manager import StrikeManager
from .historical_cache import HistoricalDataCache

# Mock WebSocketMessage to match what StreamIngestor produces
class MockMessage:
    def __init__(self, data: Dict[str, Any]):
        self.symbol = data.get('T', '') or data.get('sym', '') or data.get('ticker', '')
        self.price = data.get('p', 0.0)
        self.size = data.get('s', 0)
        self.timestamp = data.get('t', 0) # ms
        self.conditions = data.get('c', [])
        # Support pre-enriched greeks for accurate replay
        self.greeks = data.get('greeks', None) 
        # For compatibility with test_stream and others that might dump it
        self.ev = 'T' 

class ReplayEngine:
    """
    Fetches historical trades for active strikes and replays them.
    Simulates a WebSocket stream.
    Uses local cache to avoid re-fetching from Polygon.
    """
    def __init__(self, api_key: str, queue: asyncio.Queue, strike_manager: StrikeManager):
        self.client = RESTClient(api_key=api_key)
        self.queue = queue
        self.strike_manager = strike_manager
        self.running = False
        self.cache = HistoricalDataCache()

    async def run(self, minutes_back: int = 10, speed: float = 1.0):
        print(f"I am a Replay Engine! Preparing to replay last {minutes_back} minutes at {speed}x speed...")
        self.running = True
        
        # 1. Find the most recent cached trading day
        cached_dates = self.cache.get_cache_stats()['cached_dates']
        
        if not cached_dates:
            print("❌ No cached data found. Please run in live mode first to populate cache.")
            return
        
        # Use the most recent cached date
        most_recent_date_str = cached_dates[-1]
        most_recent_date = datetime.strptime(most_recent_date_str, "%Y-%m-%d").date()
        
        print(f"📅 Using most recent cached date: {most_recent_date}")
        
        # 2. Define the last N minutes of the trading day (before market close at 4:00 PM ET)
        now = datetime.now(ZoneInfo("America/New_York"))
        market_close = datetime.combine(most_recent_date, datetime.min.time()).replace(
            hour=16, minute=0, second=0, microsecond=0, tzinfo=ZoneInfo("America/New_York")
        )
        
        # Last N minutes of the day
        start_time = market_close - timedelta(minutes=minutes_back)
        end_time = market_close
        
        print(f"📦 Replaying trades from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%H:%M:%S')} (last {minutes_back} minutes)")
        
        # 3. Get reference SPY price from cached data (no API calls)
        # We'll infer the ATM strike from the cached tickers themselves
        cache_dir = self.cache.cache_dir / most_recent_date_str
        
        if not cache_dir.exists():
            print(f"❌ Cache directory not found: {cache_dir}")
            return
        
        # List cached parquet files to infer strikes
        cached_files = list(cache_dir.glob("*.parquet"))
        
        if not cached_files:
            print(f"❌ No cached trade files found in {cache_dir}")
            return
        
        print(f"📦 Found {len(cached_files)} cached trade files")
        
        # Infer SPY price from the median of cached strikes
        # Ticker format: O_SPY{date}{C/P}{strike}.parquet or O:SPY{date}{C/P}{strike}
        strikes = []
        for f in cached_files:
            # Extract strike from filename (e.g., O_SPY251216C00600000.parquet)
            fname = f.stem  # Remove .parquet
            # Strike is last 8 digits before extension, divided by 1000
            if len(fname) > 8:
                try:
                    strike_str = fname[-8:]
                    strike = float(strike_str) / 1000.0
                    strikes.append(strike)
                except ValueError:
                    continue
        
        if strikes:
            spy_price = sorted(strikes)[len(strikes) // 2]  # Median strike as proxy for SPY price
            print(f"✓ Inferred SPY Price from cached strikes: ${spy_price}")
        else:
            spy_price = 600.0
            print(f"⚠️ Could not infer price from cache, using fallback: ${spy_price}")

        # 4. Use ALL cached tickers (no need to filter by ATM ± 3 in replay mode)
        # This ensures we replay all the data we have cached
        tickers_to_fetch = []
        for f in cached_files:
            # Extract ticker from filename
            # O_SPY251216C00600000.parquet → O:SPY251216C00600000
            fname = f.stem.replace("_", ":")
            tickers_to_fetch.append(fname)
        
        print(f"📊 Replaying {len(tickers_to_fetch)} cached tickers...")
        
        # 5. Load trades from cache ONLY (no API calls)
        # Run synchronously since parquet reading is fast
        all_trades = self._load_trades_from_cache_only(tickers_to_fetch, most_recent_date, start_time, end_time)
        
        # Show cache stats
        stats = self.cache.get_cache_stats()
        print(f"📦 Cache: {stats['total_files']} files, {stats['total_size_mb']} MB")
        
        print(f"Loaded {len(all_trades)} historical trades. Starting Replay...")
        
        if not all_trades:
            print("No trades found to replay.")
            return

        # 4. Replay Loop
        start_ts = all_trades[0]['t']
        replay_start_wall_clock = datetime.now().timestamp() * 1000 # ms
        
        for trade in all_trades:
            if not self.running:
                break
                
            # Calculate delay
            trade_ts = trade['t']
            offset = trade_ts - start_ts # ms from start of sequence
            
            # Scaled delay
            target_delay_ms = offset / speed
            
            # Current elapsed time in replay
            elapsed_ms = (datetime.now().timestamp() * 1000) - replay_start_wall_clock
            
            wait_ms = target_delay_ms - elapsed_ms
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000.0)
            
            # Emit
            msg = MockMessage(trade)
            await self.queue.put(msg)
            
        print("Replay Complete.")

    def _load_trades_from_cache_only(self, tickers: List[str], trade_date: date, start: datetime, end: datetime) -> List[Dict]:
        """
        Load trades from cache ONLY - no API calls.
        Used for replay mode to ensure we never hit Polygon API.
        """
        all_trades = []
        
        for ticker in tickers:
            # Clean ticker if needed (remove T. prefix)
            clean_ticker = ticker.replace("T.", "")
            
            # Load from cache only - no API calls
            ticker_trades = self.cache.get_cached_trades(
                clean_ticker,
                trade_date,
                start,
                end
            )
            
            if ticker_trades:
                all_trades.extend(ticker_trades)
            else:
                print(f"⚠️ No cached data for {clean_ticker}")
        
        # Globally sort by time
        all_trades.sort(key=lambda x: x['t'])
        return all_trades
    
    def _fetch_all_trades_cached(self, tickers: List[str], start: datetime, end: datetime) -> List[Dict]:
        """
        Fetch trades using cache-first strategy.
        Only hits Polygon API if data not cached.
        (Kept for backward compatibility)
        """
        all_trades = []
        
        for ticker in tickers:
            # Clean ticker if needed (remove T. prefix)
            clean_ticker = ticker.replace("T.", "")
            
            # Use cache - it will fetch from Polygon only if needed
            ticker_trades = self.cache.fetch_or_get_trades(
                self.client,
                clean_ticker,
                start,
                end
            )
            
            all_trades.extend(ticker_trades)
        
        # Globally Sort by time
        all_trades.sort(key=lambda x: x['t'])
        return all_trades
