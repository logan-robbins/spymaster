from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
        self._latest_levels: Optional[Dict[str, Any]] = None
        self._latest_flow: Optional[Dict[str, Any]] = None

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

        await self.bus.subscribe(
            subject="market.flow",
            callback=self._on_flow_snapshot,
            durable_name="gateway_flow"
        )
        
        print("✅ Gateway subscribed to NATS subjects")

    async def _on_level_signals(self, data: Dict[str, Any]):
        """
        Callback for NATS messages on `levels.signals`.
        Relay directly to WebSocket clients.
        """
        # Normalize to frontend payload contract
        normalized = self._normalize_levels_payload(data)
        self._latest_levels = normalized

        # Broadcast merged payload
        await self.broadcast(self._build_payload())

    async def _on_flow_snapshot(self, data: Dict[str, Any]):
        """Callback for NATS messages on `market.flow`."""
        self._latest_flow = data
        await self.broadcast(self._build_payload())

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection and send cached state if available."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        
        # Send latest cached payload to new connection (if available)
        payload = self._build_payload()
        if payload:
            try:
                await websocket.send_text(json.dumps(payload))
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

    def _build_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self._latest_flow is not None:
            payload["flow"] = self._latest_flow
        if self._latest_levels is not None:
            payload["levels"] = self._latest_levels
        return payload

    def _normalize_levels_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ts_ms = payload.get("ts")
        if not ts_ms:
            ts_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        is_first_15m, bars_since_open = self._compute_session_context(ts_ms)

        spy = payload.get("spy")
        if not isinstance(spy, dict):
            spy = {"spot": None, "bid": None, "ask": None}

        raw_levels = payload.get("levels")
        # Some publishers may nest levels as {"levels": [...]}.
        if isinstance(raw_levels, dict) and "levels" in raw_levels:
            raw_levels = raw_levels.get("levels")
        if not isinstance(raw_levels, list):
            raw_levels = []

        normalized_levels: List[Dict[str, Any]] = []
        for level in raw_levels:
            if not isinstance(level, dict):
                continue
            normalized_levels.append(self._normalize_level_signal(level, is_first_15m, bars_since_open))

        return {
            "ts": ts_ms,
            "spy": spy,
            "levels": normalized_levels
        }

    def _normalize_level_signal(
        self,
        level: Dict[str, Any],
        is_first_15m: bool,
        bars_since_open: int
    ) -> Dict[str, Any]:
        barrier = level.get("barrier") or {}
        tape = level.get("tape") or {}
        sweep = tape.get("sweep") or {}
        fuel = level.get("fuel") or {}

        direction = level.get("direction")
        if direction == "SUPPORT":
            direction = "DOWN"
        elif direction == "RESISTANCE":
            direction = "UP"

        signal = level.get("signal")
        signal_map = {
            "REJECT": "BOUNCE",
            "CONTESTED": "CHOP",
            "NEUTRAL": "CHOP"
        }
        signal = signal_map.get(signal, signal)

        break_score_smooth = level.get("break_score_smooth")
        if break_score_smooth is None:
            break_score_smooth = level.get("break_score_raw", 0.0)

        barrier_replenishment = barrier.get("replenishment_ratio", 0.0)
        depth_in_zone = barrier.get("depth_in_zone")
        wall_ratio = level.get("wall_ratio")
        if wall_ratio is None:
            # Match historical pipeline heuristic: depth_in_zone normalized by a baseline.
            wall_ratio = (float(depth_in_zone) / 5000.0) if isinstance(depth_in_zone, (int, float)) and depth_in_zone else 0.0

        return {
            "id": level.get("id", ""),
            "level_price": level.get("price", 0.0),
            "level_kind_name": level.get("kind", "UNKNOWN"),
            "direction": direction or "UP",
            "distance": level.get("distance", 0.0),
            "is_first_15m": is_first_15m,
            "barrier_state": barrier.get("state", "NEUTRAL"),
            "barrier_delta_liq": barrier.get("delta_liq", 0.0),
            "barrier_replenishment_ratio": barrier_replenishment,
            "wall_ratio": wall_ratio,
            "tape_imbalance": tape.get("imbalance", 0.0),
            "tape_velocity": tape.get("velocity", 0.0),
            "tape_buy_vol": tape.get("buy_vol", 0),
            "tape_sell_vol": tape.get("sell_vol", 0),
            "sweep_detected": bool(sweep.get("detected", False)),
            "gamma_exposure": fuel.get("net_dealer_gamma", 0.0),
            "fuel_effect": fuel.get("effect", "NEUTRAL"),
            "approach_velocity": level.get("approach_velocity", 0.0),
            "approach_bars": level.get("approach_bars", 0),
            "approach_distance": level.get("approach_distance", 0.0),
            "prior_touches": level.get("prior_touches", 0),
            "bars_since_open": bars_since_open,
            "break_score_raw": level.get("break_score_raw", 0.0),
            "break_score_smooth": break_score_smooth,
            "signal": signal or "CHOP",
            "confidence": level.get("confidence", "LOW"),
            "note": level.get("note")
        }

    def _compute_session_context(self, ts_ms: int) -> Tuple[bool, int]:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = int((dt - market_open).total_seconds() // 60)
        if minutes_since_open < 0:
            minutes_since_open = 0
        is_first_15m = 0 <= minutes_since_open < 15
        return is_first_15m, minutes_since_open
