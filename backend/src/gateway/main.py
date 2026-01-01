"""
Gateway Service: WebSocket relay for frontend clients.

Phase 2 Transition:
- Standalone FastAPI microservice
- Subscribes to NATS `levels.signals` subject
- Relays to WebSocket clients at `/ws/stream`
- No internal state computation (that's Core Service's job)

Usage:
    uv run python -m src.gateway.main
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.gateway.socket_broadcaster import SocketBroadcaster
from src.gateway.candle_service import CandleService
from src.common.bus import NATSBus
from src.common.config import CONFIG

# Load environment
load_dotenv()

# Global instances
broadcaster: SocketBroadcaster = None
candle_service: CandleService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global broadcaster, candle_service
    
    print("🚀 Starting Gateway Service...")
    print(f"   NATS: {CONFIG.NATS_URL}")
    
    # Initialize broadcaster with NATS
    broadcaster = SocketBroadcaster()
    await broadcaster.start()
    
    # Initialize Candle Service
    candle_service = CandleService()
    
    print("✅ Gateway Service ready")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Gateway Service...")
    if broadcaster:
        await broadcaster.close()
    print("✅ Gateway shutdown complete")


# FastAPI App
app = FastAPI(
    title="Spymaster Gateway",
    description="WebSocket relay service for frontend clients",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for frontend clients.
    Receives live level signals from NATS and streams to client.
    """
    await broadcaster.connect(websocket)
    try:
        while True:
            # Keep connection alive; we don't expect client input
            # but we need to stay in the loop to detect disconnects
            data = await websocket.receive_text()
            # Optional: handle ping/pong or client commands
    except WebSocketDisconnect:
        await broadcaster.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await broadcaster.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint for orchestration."""
    return {
        "service": "gateway",
        "status": "healthy",
        "nats_url": CONFIG.NATS_URL,
        "connections": len(broadcaster.active_connections) if broadcaster else 0
    }


@app.get("/api/history/candles")
async def get_candles(
    symbol: str = Query("ES", description="Ticker symbol"),
    interval: int = Query(2, description="Interval in minutes"),
    days: int = Query(1, description="Days of history to fetch")
):
    """
    Fetch historical candles for chart initialization.
    """
    if not candle_service:
        raise HTTPException(status_code=503, detail="Candle service not initialized")
        
    try:
        return await candle_service.get_candles(symbol, interval, days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print(f"Candle fetch error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("GATEWAY_PORT", "8000"))
    
    print(f"🌐 Starting Gateway on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

