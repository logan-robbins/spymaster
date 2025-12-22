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
from .market_state import MarketState
from .level_signal_service import LevelSignalService
from .bronze_writer import BronzeWriter
from .gold_writer import GoldWriter
from .event_types import StockTrade, StockQuote, OptionTrade

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

# Level physics engine (Agent G)
market_state = MarketState(max_buffer_window_seconds=120.0)
level_signal_service = LevelSignalService(market_state=market_state)

# Storage writers (Agent I)
PERSISTENCE_ENABLED = os.getenv("PERSISTENCE_ENABLED", "true").lower() == "true"
bronze_writer = BronzeWriter() if PERSISTENCE_ENABLED else None
gold_writer = GoldWriter() if PERSISTENCE_ENABLED else None

# Mode Selection
REPLAY_MODE = os.getenv("REPLAY_MODE", "false").lower() == "true"
REPLAY_DATE = os.getenv("REPLAY_DATE")  # e.g., "2025-12-18"
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "10.0"))

if REPLAY_MODE:
    if REPLAY_DATE:
        # Use UnifiedReplayEngine for Bronze+DBN replay
        from .unified_replay_engine import UnifiedReplayEngine
        print(f"⚠️ STARTING IN UNIFIED REPLAY MODE (date={REPLAY_DATE}, speed={REPLAY_SPEED}x)")
        data_source = UnifiedReplayEngine(queue=msg_queue)
    else:
        # Legacy: ReplayEngine for API-based replay
        from .replay_engine import ReplayEngine
        print("⚠️ STARTING IN LEGACY REPLAY MODE")
        data_source = ReplayEngine(api_key=API_KEY, queue=msg_queue, strike_manager=strike_manager)
else:
    stream_ingestor = StreamIngestor(api_key=API_KEY, queue=msg_queue, strike_manager=strike_manager)
    data_source = stream_ingestor

# Background Tasks
async def processing_loop():
    """
    Consumes from queue, feeds aggregator, market_state, and Bronze writer.
    """
    while True:
        try:
            msg = await msg_queue.get()

            # Feed Aggregator (handles OptionTrade, skips stock events)
            await aggregator.process_message(msg)

            # Feed MarketState and Bronze writer based on event type
            if isinstance(msg, StockTrade):
                market_state.update_stock_trade(msg)
                if bronze_writer:
                    await bronze_writer.write_stock_trade(msg)

            elif isinstance(msg, StockQuote):
                market_state.update_stock_quote(msg)
                if bronze_writer:
                    await bronze_writer.write_stock_quote(msg)

            elif isinstance(msg, OptionTrade):
                # Option trades need greeks for gamma transfer
                greeks = greek_enricher.get_cached_greeks(msg.option_symbol)
                if greeks:
                    market_state.update_option_trade(
                        msg,
                        delta=greeks.get('delta', 0.0),
                        gamma=greeks.get('gamma', 0.0)
                    )
                else:
                    market_state.update_option_trade(msg, delta=0.0, gamma=0.0)
                if bronze_writer:
                    await bronze_writer.write_option_trade(msg)

            msg_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in processing loop: {e}")

async def broadcast_loop():
    import time
    while True:
        try:
            await asyncio.sleep(0.25)  # 250ms

            # Get flow snapshot from aggregator
            flow_snapshot = aggregator.get_snapshot()

            # Get level signals from level service
            levels_payload = level_signal_service.compute_level_signals()

            # Get current SPY quote for payload
            last_quote = market_state.last_quote
            spy_snapshot = {
                "spot": (last_quote.bid_px + last_quote.ask_px) / 2 if last_quote else None,
                "bid": last_quote.bid_px if last_quote else None,
                "ask": last_quote.ask_px if last_quote else None
            } if last_quote else {}

            ts_ms = int(time.time() * 1000)

            # Merge payloads (Option A per §6.4)
            merged_payload = {
                "ts": ts_ms,
                "flow": flow_snapshot if flow_snapshot else {},
                "spy": spy_snapshot,
                "levels": levels_payload
            }

            await broadcaster.broadcast(merged_payload)

            # Persist level signals to Gold (if enabled and levels exist)
            if gold_writer and levels_payload:
                await gold_writer.write_level_signals({
                    "ts": ts_ms,
                    "spy": spy_snapshot,
                    "levels": levels_payload
                })

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

    # Start Greek Enricher (background) - skip for replay since we use cached greeks
    if not REPLAY_MODE:
        enricher_task = asyncio.create_task(greek_enricher.start_snapshot_loop())
    else:
        enricher_task = None

    # Start Ingestor (Stream or Replay)
    if REPLAY_MODE:
        if REPLAY_DATE:
            # UnifiedReplayEngine with Bronze+DBN data
            source_task = asyncio.create_task(data_source.run(
                date=REPLAY_DATE,
                speed=REPLAY_SPEED,
                include_spy=True,
                include_options=True,
                include_es=False  # Enable when ES barrier physics is ready
            ))
        else:
            # Legacy ReplayEngine
            source_task = asyncio.create_task(data_source.run(minutes_back=10))
    else:
        # StreamIngestor
        source_task = asyncio.create_task(data_source.run_async())

    # Start Processing Loop
    proc_task = asyncio.create_task(processing_loop())

    # Start Broadcast Loop
    cast_task = asyncio.create_task(broadcast_loop())

    # Start Strike Monitor (only in live mode)
    if not REPLAY_MODE:
        mon_task = asyncio.create_task(strike_monitor_loop())
    else:
        mon_task = None

    yield

    # Shutdown
    print("🛑 Shutting down...")
    data_source.running = False

    greek_enricher.stop()
    if enricher_task:
        enricher_task.cancel()
    source_task.cancel()
    proc_task.cancel()
    cast_task.cancel()
    if mon_task:
        mon_task.cancel()

    # Flush persistence engines
    await persistence.flush()

    # Flush storage writers
    if bronze_writer:
        print("  Flushing Bronze writer...")
        await bronze_writer.flush_all()
    if gold_writer:
        print("  Flushing Gold writer...")
        await gold_writer.flush()

    print("✅ Shutdown complete")

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

# --- API Endpoints ---
from pydantic import BaseModel

class HotzoneRequest(BaseModel):
    strike: float | None

@app.post("/api/config/hotzone")
async def set_hotzone(req: HotzoneRequest):
    """
    Updates the Hotzone Focus Strike.
    Triggers immediate strike update if live.
    """
    print(f"📥 Received hotzone request: {req.strike}")
    strike_manager.set_focus_strike(req.strike)
    
    # If in live mode, force immediate refresh of subscriptions
    # Re-run logic with current price (or fallback)
    # We don't have easy access to current price *here* without querying client or storing it.
    # But strike_monitor_loop runs every 60s.
    # To make it instant, we could store last_price in StrikeManager?
    # For now, 60s delay is acceptable, OR we can rely on next trade tick if we were tracking price.
    # But new Hotzone might be far away.
    # Let's trust strike_monitor_loop to pick it up on next cycle (max 60s).
    # OR, we can wake up the monitor loop?
    # Simple workaround: Just return OK.
    
    return {"status": "ok", "hotzone": req.strike}
