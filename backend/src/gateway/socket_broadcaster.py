from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np

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
        self._latest_viewport: Optional[Dict[str, Any]] = None
        self._latest_pentaview: Optional[Dict[str, Any]] = None

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
        
        # Subscribe to Pentaview streams (candles + streams + projections)
        await self.bus.subscribe(
            subject="pentaview.streams",
            callback=self._on_pentaview_stream,
            durable_name="gateway_pentaview"
        )
        
        print("✅ Gateway subscribed to NATS subjects")

    async def _on_level_signals(self, data: Dict[str, Any]):
        """
        Callback for NATS messages on `levels.signals`.
        Relay directly to WebSocket clients.
        """
        print(f"📥 Received level signal: {len(data.get('levels', []))} levels")
        
        # Normalize to frontend payload contract
        normalized = self._normalize_levels_payload(data)
        self._latest_levels = normalized
        self._latest_viewport = data.get("viewport")

        # Broadcast merged payload
        await self.broadcast(self._build_payload())
        print(f"📡 Broadcasted to {len(self.active_connections)} clients")

    async def _on_flow_snapshot(self, data: Dict[str, Any]):
        """Callback for NATS messages on `market.flow`."""
        self._latest_flow = data
        await self.broadcast(self._build_payload())
    
    async def _on_pentaview_stream(self, data: Dict[str, Any]):
        """Callback for NATS messages on `pentaview.streams`."""
        self._latest_pentaview = data
        await self.broadcast(self._build_payload())
        print(f"📡 Broadcasted Pentaview data to {len(self.active_connections)} clients")

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
        if self._latest_viewport is not None:
            payload["viewport"] = self._latest_viewport
        if self._latest_pentaview is not None:
            # Include Pentaview data (candles, streams, projections)
            payload["pentaview"] = self._latest_pentaview
        return payload

    def _normalize_levels_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ts_ms = payload.get("ts")
        if not ts_ms:
            ts_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        is_first_15m, bars_since_open = self._compute_session_context(ts_ms)

        raw_levels = payload.get("levels")
        # Some publishers may nest levels as {"levels": [...]}.
        if isinstance(raw_levels, dict) and "levels" in raw_levels:
            raw_levels = raw_levels.get("levels")
        if not isinstance(raw_levels, list):
            raw_levels = []

        # Extract viewport predictions (if present)
        viewport = payload.get("viewport")
        viewport_targets = viewport.get("targets", []) if viewport else []
        
        # Build lookup for viewport predictions by level_id
        viewport_by_id: Dict[str, Dict[str, Any]] = {}
        for target in viewport_targets:
            if isinstance(target, dict):
                level_id = target.get("level_id")
                if level_id:
                    viewport_by_id[level_id] = target

        normalized_levels: List[Dict[str, Any]] = []
        for level in raw_levels:
            if not isinstance(level, dict):
                continue
            
            # Get level ID for viewport lookup
            level_id = level.get("id", "")
            viewport_pred = viewport_by_id.get(level_id)
            
            normalized_levels.append(
                self._normalize_level_signal(level, is_first_15m, bars_since_open, viewport_pred)
            )

        return {
            "ts": ts_ms,
            "levels": normalized_levels
        }

    def _normalize_level_signal(
        self,
        level: Dict[str, Any],
        is_first_15m: bool,
        bars_since_open: int,
        viewport_pred: Optional[Dict[str, Any]] = None
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
            "CONTESTED": "NO_TRADE",
            "NEUTRAL": "NO_TRADE"
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

        # Confluence features
        confluence_level = level.get("confluence_level", 0)
        confluence_level_name = level.get("confluence_level_name", "UNDEFINED")
        
        # Base normalized level
        normalized = {
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
            "note": level.get("note"),
            "confluence_count": level.get("confluence_count", 0),
            "confluence_pressure": level.get("confluence_pressure", 0.0),
            "confluence_alignment": level.get("confluence_alignment", 0),
            "confluence_level": confluence_level,
            "confluence_level_name": confluence_level_name
        }
        
        # Add ML predictions if available from viewport
        if viewport_pred:
            normalized["ml_predictions"] = {
                "p_tradeable_2": viewport_pred.get("p_tradeable_2", 0.0),
                "p_no_trade": viewport_pred.get("p_no_trade", 0.0),
                "p_break": viewport_pred.get("p_break", 0.0),
                "p_bounce": viewport_pred.get("p_bounce", 0.0),
                "strength_signed": viewport_pred.get("strength_signed", 0.0),
                "strength_abs": viewport_pred.get("strength_abs", 0.0),
                "utility_score": viewport_pred.get("utility_score", 0.0),
                "stage": viewport_pred.get("stage", "stage_a"),
                "time_to_threshold": viewport_pred.get("time_to_threshold", {}),
                "retrieval": viewport_pred.get("retrieval", {})
            }
        
        return normalized

    def _compute_session_context(self, ts_ms: int) -> Tuple[bool, int]:
        from src.common.utils.session_time import compute_bars_since_open, is_first_15_minutes

        ts_ns = int(ts_ms) * 1_000_000
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
        date_str = dt.strftime("%Y-%m-%d")
        ts_arr = np.array([ts_ns], dtype=np.int64)
        bars_since_open = int(compute_bars_since_open(ts_arr, date_str, bar_duration_minutes=1)[0])
        is_first_15m = bool(is_first_15_minutes(ts_arr, date_str)[0])
        return is_first_15m, bars_since_open
