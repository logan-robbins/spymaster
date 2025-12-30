"""
Lake Service entry point (Phase 2: NATS + S3).

Subscribes to:
- market.* (Bronze writer)
- levels.signals (Gold writer)

Writes to:
- S3/MinIO or local filesystem (configured via env vars)

Per NEXT.md Agent B deliverable.
"""

import asyncio
import signal
import sys

from src.common.bus import NATSBus
from src.common.config import CONFIG
from src.io.bronze import BronzeWriter
from src.io.gold import GoldWriter


class LakeService:
    """
    Lake Service: Archives all market data and level signals to Parquet.
    
    Architecture:
    - Subscribes to NATS subjects (market.*, levels.signals)
    - Micro-batches events to Parquet files
    - Supports local filesystem or S3/MinIO storage
    """
    
    def __init__(self):
        self.bus = NATSBus(servers=[CONFIG.NATS_URL])
        self.bronze_writer: BronzeWriter = None
        self.gold_writer: GoldWriter = None
        self._shutdown = False
    
    async def start(self):
        """Start Lake service."""
        print("=" * 60)
        print("LAKE SERVICE")
        print("=" * 60)
        
        # Connect to NATS
        await self.bus.connect()
        
        # Initialize writers
        self.bronze_writer = BronzeWriter(bus=self.bus)
        self.gold_writer = GoldWriter(bus=self.bus)
        
        # Start writers (subscribes to NATS)
        await self.bronze_writer.start()
        await self.gold_writer.start()
        
        print("=" * 60)
        print("Lake service running. Press Ctrl+C to stop.")
        print("=" * 60)
        
        # Keep running
        while not self._shutdown:
            await asyncio.sleep(1)
    
    async def stop(self):
        """Stop Lake service gracefully."""
        if self._shutdown:
            return
        
        self._shutdown = True
        print("\n🛑 Shutting down Lake service...")
        
        # Stop writers (flush remaining data)
        if self.bronze_writer:
            await self.bronze_writer.stop()
        if self.gold_writer:
            await self.gold_writer.stop()
        
        # Close NATS connection
        await self.bus.close()
        
        print("✅ Lake service stopped")


async def main():
    """Main entry point."""
    service = LakeService()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        asyncio.create_task(service.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await service.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Lake service error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await service.stop()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Lake service interrupted")
        sys.exit(0)

