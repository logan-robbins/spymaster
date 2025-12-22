"""
Ingestor Service Entry Point

This service connects to Polygon WebSocket feeds and publishes
normalized events to NATS JetStream.

NATS Subjects Published:
- market.stocks.trades (StockTrade)
- market.stocks.quotes (StockQuote)
- market.options.trades (OptionTrade)

Per AGENT A tasks in NEXT.md.
"""

import asyncio
import os
import sys
from src.common.bus import NATSBus
from src.common.config import CONFIG
from src.ingestor.stream_ingestor import StreamIngestor
from src.core.strike_manager import StrikeManager


async def run_ingestor_service():
    """
    Initialize and run the Ingestor service.
    
    Connects to:
    1. NATS JetStream (for publishing events)
    2. Polygon WebSocket (for receiving market data)
    """
    print("=" * 60)
    print("🚀 INGESTOR SERVICE")
    print("=" * 60)
    
    # Get Polygon API key from environment
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("❌ ERROR: POLYGON_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize NATS Bus
    bus = NATSBus(servers=[CONFIG.NATS_URL])
    await bus.connect()
    
    # Initialize Strike Manager for dynamic option subscriptions
    strike_manager = StrikeManager(
        initial_price=600.0,  # SPY default starting point
        step=1.0,
        range_dollars=5.0
    )
    
    # Initialize Stream Ingestor
    ingestor = StreamIngestor(
        api_key=api_key,
        bus=bus,
        strike_manager=strike_manager
    )
    
    print("✅ Ingestor initialized")
    print(f"📡 Publishing to NATS at {CONFIG.NATS_URL}")
    print("🎯 Subjects: market.stocks.*, market.options.*")
    print("=" * 60)
    
    # Run the ingestor
    try:
        await ingestor.run_async()
    except KeyboardInterrupt:
        print("\n⏹ Shutting down ingestor...")
    except Exception as e:
        print(f"❌ Ingestor error: {e}")
        raise
    finally:
        await bus.close()
        print("✅ Ingestor service stopped")


def main():
    """Entry point for the service."""
    try:
        asyncio.run(run_ingestor_service())
    except KeyboardInterrupt:
        print("\n👋 Goodbye")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
