import asyncio
import pytest
from src.historical_cache import HistoricalDataCache
from src.replay_engine import ReplayEngine


@pytest.mark.asyncio
async def test_replay_discovery():
    print("🔎 Testing Unified Replay Discovery...")
    cache = HistoricalDataCache()
    latest = cache.get_latest_available_date()
    
    if latest:
        print(f"✅ Replay Engine sees partition: {latest}")
        trades = cache.get_cached_trades(None, latest)
        print(f"📊 Trades available: {len(trades)}")
    else:
        print("❌ Replay Engine sees nothing!")

if __name__ == "__main__":
    asyncio.run(test_replay_discovery())
