import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .strike_manager import StrikeManager
from .greek_enricher import GreekEnricher
from .persistence_engine import PersistenceEngine
from .flow_aggregator import FlowAggregator
from .stream_ingestor import StreamIngestor
from .socket_broadcaster import SocketBroadcaster

# Load environment
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("POLYGON_API_KEY not found")

# Global instances
strike_manager = StrikeManager()
greek_enricher = GreekEnricher(api_key=API_KEY)
persistence = PersistenceEngine()
aggregator = FlowAggregator(greek_enricher, persistence)
broadcaster = SocketBroadcaster()
msg_queue = asyncio.Queue()

# Mode Selection
REPLAY_MODE = os.getenv("REPLAY_MODE", "false").lower() == "true"

if REPLAY_MODE:
    from .replay_engine import ReplayEngine
    print("⚠️ STARTING IN REPLAY MODE")
    data_source = ReplayEngine(api_key=API_KEY, queue=msg_queue, strike_manager=strike_manager)
else:
    stream_ingestor = StreamIngestor(api_key=API_KEY, queue=msg_queue, strike_manager=strike_manager)
    data_source = stream_ingestor

# Background Tasks
async def processing_loop():
    """
    Consumes from queue, feeds aggregator, and broadcasts updates.
    """
    throttle_interval = 0.25 # 250ms
    last_broadcast = 0
    
    while True:
        try:
            # Batch process? Or single? Single is simpler but slower?
            # Queue.get() is one by one.
            # We can try to get multiple?
            msg = await msg_queue.get()
            
            # Feed Aggregator
            await aggregator.process_message(msg)
            
            # Broadcast limit check
            # Real-time broadcast might be too much. 
            # We broadcast SNAPSHOTS every X ms.
            # So we don't broadcast ON every trade.
            # We run a separate broadcaster loop?
            # Or we check time here.
            # If we process fast, we might trigger broadcast often.
            # Better: separate loop for broadcasting.
            
            msg_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in processing loop: {e}")

async def broadcast_loop():
    while True:
        try:
            await asyncio.sleep(0.25) # 250ms
            snapshot = aggregator.get_snapshot()
            if snapshot:
                await broadcaster.broadcast(snapshot)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in broadcast loop: {e}")

async def strike_monitor_loop():
    """
    Periodically check price and update strikes.
    """
    # For now, we need a source of PRICE.
    # We can use the aggregator's last price IF we are getting data.
    # But initially we have no data.
    # We should have a way to get SPY price.
    # Or we rely on `StreamIngestor` receiving SPY connection?
    # I mentioned I'd use `T.SPY`. 
    # Let's start by subbing to T.SPY in `StreamIngestor`.
    # Then `market='options'` might NOT accept `T.SPY`.
    # If so, we need `market='stocks'` client.
    # I'll add a simple REST poll for SPY price every 60s here for robustness.
    
    from polygon import RESTClient
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    client = RESTClient(api_key=API_KEY)
    
    while True:
        try:
            # Get SPY price from options chain (included in Options Advanced plan)
            today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            snapshots = client.list_snapshot_options_chain("SPY", params={"expiration_date": today})
            snap = next(snapshots)
            
            if snap.underlying_asset and snap.underlying_asset.price:
                price = snap.underlying_asset.price
                
                # Update Strikes
                add, remove = strike_manager.get_target_strikes(price)
                if add or remove:
                    print(f"🔄 Price ${price}: Adding {len(add)}, Removing {len(remove)}")
                    stream_ingestor.update_subs(add, remove)
                    
                    # Also trigger Greek refresh if strikes changed?
                    # GreekEnricher loop runs every 60s anyway.
                    # We can force it? 
                    # `greek_enricher._fetch_and_update()`
        except Exception as e:
            print(f"Strike monitor error: {e}")
            # Fallback for dev/unauthorized: Use 600
            # Only if no subscriptions yet?
            if not strike_manager.current_subscriptions:
                 print("⚠️ Using fallback price 600.00 due to API error")
                 add, remove = strike_manager.get_target_strikes(600.0)
                 if add:
                     data_source.update_subs(add, remove)
            
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting 0DTE Backend...")
    
    # Start Greek Enricher (background)
    enricher_task = asyncio.create_task(greek_enricher.start_snapshot_loop())
    
    # Start Ingestor (Stream or Replay)
    if REPLAY_MODE:
        # Replay run is async and can be awaited or tasked.
        # It finishes when replay is done.
        source_task = asyncio.create_task(data_source.run(minutes_back=10))
    else:
        # StreamIngestor
        source_task = asyncio.create_task(data_source.run_async())
    
    # Start Processing Loop
    proc_task = asyncio.create_task(processing_loop())
    
    # Start Broadcast Loop
    cast_task = asyncio.create_task(broadcast_loop())
    
    # Start Strike Monitor (Only if NOT replay? Or replay handles it?)
    # ReplayEngine picks strikes ONCE at start in simplified version.
    # So we don't need strike monitor in Replay v1.
    if not REPLAY_MODE:
        mon_task = asyncio.create_task(strike_monitor_loop())
    else:
        mon_task = None
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    if not REPLAY_MODE:
        data_source.running = False
    else:
        data_source.running = False # Replay engine flag
        
    greek_enricher.stop()
    enricher_task.cancel()
    source_task.cancel()
    proc_task.cancel()
    cast_task.cancel()
    if mon_task:
        mon_task.cancel()
    # persistence flushes automatically? We should force flush.
    await persistence.flush()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            # Keep alive?
            data = await websocket.receive_text()
            # We don't expect input, but maybe ping?
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
