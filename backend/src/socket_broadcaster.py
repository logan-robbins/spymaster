from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import asyncio
import json

class SocketBroadcaster:
    """
    Push updates to the Frontend via WebSocket.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        # We can't await lock in sync disconnect? 
        # WebSocketDisconnect handling usually happens in the endpoint handler.
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        pass # Lock handling should be careful. 
        # Better to access list directly if we are in async main thread always.
        # Since FastAPI runs on single loop, it's mostly safe, but cleaner to lock.

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
            
        payload = json.dumps(message) # Serialize once
        
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                to_remove.append(connection)
        
        if to_remove:
            for c in to_remove:
                self.disconnect(c)
