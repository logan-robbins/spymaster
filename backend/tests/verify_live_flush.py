import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from polygon import RESTClient
from src.core.strike_manager import StrikeManager
from src.lake.persistence_engine import PersistenceEngine
from src.core.flow_aggregator import FlowAggregator
from src.ingestor.stream_ingestor import StreamIngestor

# Load env including API Key
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")

class MockEnricher:
    """Pass-through enricher for testing."""
    def get_greeks(self, ticker):
        class Greeks:
            delta = 0.5
            gamma = 0.05
        return Greeks()

async def run_live_test():
    print("🚀 Starting ENHANCED Live Verification Test (Unrestricted Price Fetch)...")
    
    # 1. Get Real Price via Options Snapshot (Unrestricted on Advanced Plan)
    print("🔎 Fetching real SPY price via Options Snapshot...")
    client = RESTClient(api_key=API_KEY)
    price = 600.0
    
    try:
        # Use list_snapshot_options_chain as used in main.py
        # This returns an iterator of OptionContractSnapshot
        # We need just ONE to get the underlying price.
        today = datetime.now().strftime("%Y-%m-%d")
        
        # We limit to 1 result to save bandwidth, we just need the underlying price
        # Note: The snapshot might be huge if we don't filter? 
        # But we only read the first one.
        # Params: expiration_date is a good filter to limit scope if needed, but not strictly required for just getting underlying price from first item.
        # But let's use today's expiration to be fast.
        
        snapshots = client.list_snapshot_options_chain("SPY", params={"expiration_date": today, "limit": 1})
        
        # We only need the first one
        snap = next(snapshots, None)
        
        if snap:
            # Check structure: snap.underlying_asset.price
            if hasattr(snap, 'underlying_asset') and hasattr(snap.underlying_asset, 'price'):
                 price = snap.underlying_asset.price
                 print(f"💵 SPY Price (from Options Snapshot): ${price}")
            else:
                 print(f"⚠️ Snapshot found but missing underlying price. Object: {snap}")
        else:
             print("⚠️ No options snapshot found for today. Market might be closed or no 0DTE?")
             # Fallback: Try a longer expiration or no expiration filter if today fails?
             # But if 0DTE fails, we might have issues anyway.
             # Let's try without expiration filter if first fails?
             print("⚠️ Retrying without expiration filter...")
             snapshots_any = client.list_snapshot_options_chain("SPY", params={"limit": 1})
             snap_any = next(snapshots_any, None)
             if snap_any and hasattr(snap_any, 'underlying_asset'):
                 price = snap_any.underlying_asset.price
                 print(f"💵 SPY Price (fallback snapshot): ${price}")

    except Exception as e:
        print(f"⚠️ Error fetching price: {e}. Using 600.0")
        price = 600.0

    # 2. Setup Support Classes
    strike_manager = StrikeManager()
    persistence = PersistenceEngine()
    persistence.flush_interval = 20 # Faster flush 20s
    print("⚙️  Patched Flush Interval to 20s")
    
    enricher = MockEnricher()
    aggregator = FlowAggregator(enricher, persistence)
    
    # 3. Setup Live Stream
    queue = asyncio.Queue()
    ingestor = StreamIngestor(API_KEY, queue, strike_manager)
    
    # 4. Processing Loop (Verbose)
    async def process():
        count = 0
        print("👂 Listening for trades...")
        while True:
            try:
                msg = await queue.get()
                # Print FIRST message to confirm flow
                if count == 0:
                    print(f"🎉 FIRST MESSAGE RECEIVED: {msg}")
                
                await aggregator.process_message(msg)
                count += 1
                if count % 10 == 0:
                    print(f"⚡ Processed {count} trades...", end='\r')
                queue.task_done()
            except asyncio.CancelledError:
                break
    
    # 5. Launch
    ingestor_task = asyncio.create_task(ingestor.run_async())
    proc_task = asyncio.create_task(process())
    
    # 6. Subscribe
    # Get strikes around REAL price
    strikes, remove = strike_manager.get_target_strikes(price)
    print(f"🎯 Target Strikes: {strikes[:3]}... ({len(strikes)} total)")
    
    # Subscribe
    ingestor.update_subs(strikes, [])
    
    # 7. Wait for Flow & Flush
    # Wait 30 seconds
    print("⏳ Waiting 30s for data accumulation and flush...")
    await asyncio.sleep(30)
    
    # 8. Shutdown & Flush
    print("\n🛑 Stopping...")
    ingestor_task.cancel()
    proc_task.cancel()
    await persistence.flush()
    
    # 9. Verify
    from src.lake.historical_cache import HistoricalDataCache
    cache = HistoricalDataCache()
    
    latest = cache.get_latest_available_date()
    if latest:
        print(f"✅ FOUND PARTITION: {latest}")
        trades = cache.get_cached_trades(None, latest)
        print(f"📊 Total Trades Preserved: {len(trades)}")
        if len(trades) > 0:
            print("🚀 VERIFICATION SUCCESSFUL!")
        else:
            print("⚠️ Partition Empty. No trades processed.")
    else:
        print("❌ NO PARTITION FOUND.")

if __name__ == "__main__":
    asyncio.run(run_live_test())
