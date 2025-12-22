from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import asyncio
import json

from src.common.bus import NATSBus
from src.common.config import CONFIG


class SocketBroadcaster:
    """
    Gateway Service: Pure WebSocket relay that subscribes to NATS subjects
    and broadcasts to connected frontend clients.
    
    Phase 2 Transition:
    - Removed internal state computation
    - Subscribes to `levels.signals` on NATS
    - Optionally subscribes to `market.flow` (if flow view is kept)
    - Acts as a pure relay: NATS → WebSocket
    """
    def __init__(self, bus: Optional[NATSBus] = None):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self.bus = bus
        self._subscriptions = []
        self._latest_payload = {}  # Cache for new connections

    async def start(self):
        """Initialize NATS connection and subscribe to signals."""
        if not self.bus:
            self.bus = NATSBus(servers=[CONFIG.NATS_URL])
        
        await self.bus.connect()
        
        # Subscribe to level signals
        await self.bus.subscribe(
            subject="levels.signals",
            callback=self._on_level_signals,
            durable_name="gateway_levels"
        )
        
        print("✅ Gateway subscribed to NATS subjects")

    async def _on_level_signals(self, data: Dict[str, Any]):
        """
        Callback for NATS messages on `levels.signals`.
        Relay directly to WebSocket clients.
        """
        # Cache latest payload for new connections
        self._latest_payload = data
        
        # Broadcast to all connected clients
        await self.broadcast(data)

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection and send cached state if available."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        
        # Send latest cached payload to new connection (if available)
        if self._latest_payload:
            try:
                await websocket.send_text(json.dumps(self._latest_payload))
            except Exception as e:
                print(f"Failed to send cached state to new connection: {e}")

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket from active connections."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients."""
        if not self.active_connections:
            return
            
        payload = json.dumps(message)  # Serialize once
        
        to_remove = []
        async with self._lock:
            for connection in self.active_connections[:]:  # Copy list to avoid mutation during iteration
                try:
                    await connection.send_text(payload)
                except Exception as e:
                    print(f"WebSocket send failed: {e}")
                    to_remove.append(connection)
        
        # Clean up failed connections
        for c in to_remove:
            await self.disconnect(c)
    
    async def close(self):
        """Shutdown: close NATS connection."""
        if self.bus:
            await self.bus.close()
