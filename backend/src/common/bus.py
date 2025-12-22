import json
import asyncio
from typing import Any, Callable, Dict, Optional, Awaitable
from dataclasses import asdict

import nats
from nats.js.api import StreamConfig, RetentionPolicy

from src.common.config import CONFIG

class NATSBus:
    """
    Wrapper for NATS JetStream connectivity.
    Handles connection, stream creation, and pub/sub.
    """
    def __init__(self, servers: list[str] = ["nats://localhost:4222"]):
        self.servers = servers
        self.nc = None
        self.js = None
        self._subscriptions = []

    async def connect(self):
        """Connect to NATS and initialize JetStream."""
        if self.nc and self.nc.is_connected:
            return

        print(f"🔌 Connecting to NATS at {self.servers}...")
        try:
            self.nc = await nats.connect(servers=self.servers)
            self.js = self.nc.jetstream()
            print("✅ Connected to NATS JetStream")
            
            # Initialize Streams
            await self._init_streams()
            
        except Exception as e:
            print(f"❌ NATS Connection Error: {e}")
            raise e

    async def _init_streams(self):
        """Idempotently create streams."""
        # Stream for Market Data (Trades, Quotes, MBP)
        # Retention: WorkQueue or Limits? 
        # For data lake, we want everything. For realtime, we want speed.
        # We'll use File storage (persistent) with limits.
        
        streams = [
            StreamConfig(
                name="MARKET_DATA",
                subjects=["market.*", "market.*.*"],  # e.g. market.futures.trades
                retention=RetentionPolicy.LIMITS,
                max_age=24 * 60 * 60, # 24 hours retention
                storage="file"
            ),
            StreamConfig(
                name="LEVEL_SIGNALS",
                subjects=["levels.*"],
                retention=RetentionPolicy.LIMITS,
                max_age=24 * 60 * 60,
                storage="file"
            )
        ]

        for config in streams:
            try:
                await self.js.add_stream(config)
                print(f"  - Stream '{config.name}' ready")
            except Exception as e:
                # Ignore if exists
                pass

    async def publish(self, subject: str, payload: Any):
        """
        Publish a message to NATS.
        Handles Pydantic models, Dataclasses, and Dicts.
        """
        if not self.js:
            raise ConnectionError("NATS not connected")

        if hasattr(payload, "model_dump_json"): # Pydantic
            data = payload.model_dump_json().encode()
        elif hasattr(payload, "to_json"): # Some dataclasses if mixed in
            data = payload.to_json().encode()
        elif hasattr(payload, "__dict__"): # Dataclass
            data = json.dumps(asdict(payload), default=str).encode()
        elif isinstance(payload, dict):
            data = json.dumps(payload, default=str).encode()
        else:
            data = str(payload).encode()

        ack = await self.js.publish(subject, data)
        return ack

    async def subscribe(self, subject: str, callback: Callable[[Any], Awaitable[None]], durable_name: Optional[str] = None):
        """
        Subscribe to a subject.
        """
        if not self.js:
            raise ConnectionError("NATS not connected")

        async def _msg_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                await callback(data)
                await msg.ack()
            except Exception as e:
                print(f"Error processing NATS msg on {subject}: {e}")
                # Ideally nak() or term() depending on error

        sub = await self.js.subscribe(subject, cb=_msg_handler, durable=durable_name)
        self._subscriptions.append(sub)
        print(f"🎧 Subscribed to {subject}")
        return sub

    async def close(self):
        """Close connection."""
        if self.nc:
            await self.nc.close()
            print("🔌 NATS connection closed")

# Global singleton
BUS = NATSBus()

