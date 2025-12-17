import asyncio
from datetime import datetime, timedelta
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
        
        # 1. Get Reference Time and Trading Hours
        now = datetime.now(ZoneInfo("America/New_York"))
        
        # Define market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Determine start time based on current time and minutes_back
        if now < market_open:
            # Before market open - use previous day's close to open
            market_open = market_open - timedelta(days=1)
            market_close = market_close - timedelta(days=1)
            start_time = market_open
            end_time = market_close
        elif now > market_close:
            # After market close - use today's market hours
            start_time = market_open
            end_time = market_close
        else:
            # During market hours - use minutes_back from now
            start_time = max(now - timedelta(minutes=minutes_back), market_open)
            end_time = now
        
        print(f"Fetching trades from {start_time} to {end_time}")
        
        # Get Reference Price (already done above)
        # Fetch last quote/trade for SPY at start_time? 
        # Or just use current price to pick strikes that are relevant NOW (simplest for dev).
        # But if we replay "history", we should pick strikes relevant THEN.
        # Let's use current price as valid proxy for "active" attention.
        
        # Get current SPY price from options chain (included in Options Advanced plan)
        spy_price = 0.0
        try:
            # Get underlying price from ANY options chain snapshot
            # This is included with Options Advanced - no separate stocks plan needed!
            from datetime import datetime as dt
            today = dt.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            
            snapshots = self.client.list_snapshot_options_chain("SPY", params={"expiration_date": today})
            snap = next(snapshots)
            
            if snap.underlying_asset and snap.underlying_asset.price:
                spy_price = snap.underlying_asset.price
                timeframe = getattr(snap.underlying_asset, 'timeframe', 'UNKNOWN')
                print(f"✓ SPY Price from Options Chain: ${spy_price} ({timeframe})")
            else:
                raise ValueError("No underlying price in snapshot")
                
        except Exception as e:
            print(f"⚠️ Options chain price failed: {e}")
            try:
                # Fallback to previous close aggregate
                aggs = self.client.get_previous_close_agg("SPY")
                if aggs:
                    spy_price = aggs[0].close
                    print(f"✓ SPY Price (Prev Close Fallback): ${spy_price}")
            except Exception as e2:
                print(f"❌ All price methods failed: {e2}")
                spy_price = 600.0
                print(f"⚠️ Using Hard Fallback: ${spy_price}")

        # 2. Identify Strikes
        # We force StrikeManager to calculate based on this price
        add, _ = self.strike_manager.get_target_strikes(spy_price)
        # We also want SPY trades
        tickers_to_fetch = ["SPY"] + [t.replace("T.", "") for t in add] # separate T. prefix if present
        # Note: StrikeManager returns "T.O:..." strings in `add`.
        # We need "O:..." for REST API.
        
        print(f"Fetching history for {len(tickers_to_fetch)} tickers...")
        
        # 3. Fetch Trades (using cache to avoid redundant API calls)
        # loop.run_in_executor for blocking REST calls
        loop = asyncio.get_running_loop()
        all_trades = await loop.run_in_executor(None, self._fetch_all_trades_cached, tickers_to_fetch, start_time, end_time)
        
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

    def _fetch_all_trades_cached(self, tickers: List[str], start: datetime, end: datetime) -> List[Dict]:
        """
        Fetch trades using cache-first strategy.
        Only hits Polygon API if data not cached.
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
