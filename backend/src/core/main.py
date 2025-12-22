"""
Core Service Entry Point.

Agent C deliverable per NEXT.md.

This module starts the Core Service (The Brain), which:
- Consumes market data from NATS
- Computes level signals
- Publishes signals to NATS

Usage:
    export NATS_URL=nats://localhost:4222
    uv run python -m src.core.main
"""

import asyncio
import os
import signal
import sys

from src.common.bus import NATSBus
from src.common.config import CONFIG
from src.core.service import CoreService


async def main():
    """Main entry point for Core Service."""
    print("=" * 60)
    print("🧠 SPYMASTER CORE SERVICE (The Brain)")
    print("=" * 60)
    
    # Get NATS URL from environment
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    print(f"📡 NATS URL: {nats_url}")
    
    # Initialize NATS bus
    bus = NATSBus(servers=[nats_url])
    
    try:
        # Connect to NATS
        await bus.connect()
        
        # Initialize Core Service
        service = CoreService(
            bus=bus,
            config=CONFIG,
            user_hotzones=None  # TODO: load from config/env if needed
        )
        
        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()
        
        def signal_handler(sig, frame):
            print("\n🛑 Shutdown signal received")
            stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start service (non-blocking)
        service_task = asyncio.create_task(service.start())
        
        # Wait for shutdown signal
        await stop_event.wait()
        
        # Stop service
        await service.stop()
        
        # Cancel service task
        service_task.cancel()
        try:
            await service_task
        except asyncio.CancelledError:
            pass
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Close NATS connection
        await bus.close()
        print("👋 Core Service shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)

