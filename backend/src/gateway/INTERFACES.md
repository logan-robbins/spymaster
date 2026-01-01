# Gateway Module Interfaces

**Module**: `backend/src/gateway/`
**Role**: WebSocket relay with normalization (NATS → Frontend)
**Audience**: AI Coding Agents

---

## Module Purpose

Relay service that bridges NATS JetStream and frontend WebSocket clients. Subscribes to multiple NATS subjects, normalizes payloads to frontend contract, caches latest state, and broadcasts JSON to connected clients.

---

## Input Interface

### NATS Subscriptions

**Subject 1**: `levels.signals`
**Consumer**: Durable consumer `gateway_levels`
**Schema**: `levels.signals.v1` (from Core Service)

**Subject 2**: `market.flow`
**Consumer**: Durable consumer `gateway_flow`
**Schema**: Flow snapshot (from Ingestor or Core)

**Processing**:
1. Receive message from NATS
2. Normalize payload to frontend contract (for levels.signals)
3. Update cached state (`_latest_levels`, `_latest_flow`, `_latest_viewport`)
4. Build merged payload and broadcast to all connected WebSocket clients

---

## Output Interface

### WebSocket Endpoint

**URL**: `ws://localhost:8000/ws/stream`
**Protocol**: WebSocket (JSON text frames)
**Cadence**: ~250ms (driven by Core Service snap interval)

**Connection Flow**:
1. Client connects → Gateway accepts
2. Gateway sends cached payload immediately (if available)
3. Gateway broadcasts subsequent updates as they arrive from NATS
4. Client disconnect → Gateway removes from active connections

**Message Format**: JSON object with optional keys:
```json
{
  "flow": { ... },      // From market.flow subject
  "levels": {           // Normalized levels payload
    "ts": 1703123456000,
    "spy": { "spot": 687.50, "bid": 687.49, "ask": 687.51 },
    "levels": [ ... ]   // Array of normalized level signals
  },
  "viewport": { ... }   // Viewport data from Core
}
```

---

## Normalization Pipeline

### Signal Mapping

The Gateway normalizes Core Service signals to frontend-friendly values:

**Direction**:
- `SUPPORT` → `DOWN`
- `RESISTANCE` → `UP`

**Signal**:
- `REJECT` → `BOUNCE`
- `CONTESTED` → `NO_TRADE`
- `NEUTRAL` → `NO_TRADE`

### Normalized Level Schema

Each level is transformed by `_normalize_level_signal`:

```python
{
    "id": str,                          # Level identifier
    "level_price": float,               # Price level
    "level_kind_name": str,             # STRIKE, VWAP, SESSION_HIGH, etc.
    "direction": str,                   # UP or DOWN (normalized)
    "distance": float,                  # Distance from spot
    "is_first_15m": bool,               # First 15 min of session
    "barrier_state": str,               # VACUUM, WALL, ABSORPTION, etc.
    "barrier_delta_liq": float,         # Liquidity delta
    "barrier_replenishment_ratio": float,
    "wall_ratio": float,                # Depth normalized by baseline
    "tape_imbalance": float,
    "tape_velocity": float,
    "tape_buy_vol": int,
    "tape_sell_vol": int,
    "sweep_detected": bool,
    "gamma_exposure": float,            # From fuel.net_dealer_gamma
    "fuel_effect": str,                 # AMPLIFY, DAMPEN, NEUTRAL
    "approach_velocity": float,
    "approach_bars": int,
    "approach_distance": float,
    "prior_touches": int,
    "bars_since_open": int,             # Minutes since 9:30 ET
    "break_score_raw": float,
    "break_score_smooth": float,
    "signal": str,                      # BREAK, BOUNCE, NO_TRADE (normalized)
    "confidence": str,                  # HIGH, MEDIUM, LOW
    "note": Optional[str],

    # Confluence Features
    "confluence_count": int,            # Number of nearby key levels
    "confluence_pressure": float,       # Weighted pressure (0-1)
    "confluence_alignment": int,        # -1=OPPOSED, 0=NEUTRAL, 1=ALIGNED
    "confluence_level": int,            # 0-10 hierarchical scale
    "confluence_level_name": str,       # ULTRA_PREMIUM, PREMIUM, STRONG, MODERATE, CONSOLIDATION
    
    // ML Predictions (merged from viewport)
    // NOTE: Predictions use GEOMETRY-ONLY kNN (32D DCT shape matching)
    // Physics features are provided for context but NOT used in retrieval
    // See: RESEARCH.md Phase 4 (Geometry ECE: 2.4% vs Physics ECE: 21%)
    "ml_predictions": {
        "p_tradeable_2": float,         // P(tradeable)
        "p_break": float,               // P(break | tradeable) - GEOMETRY-BASED
        "p_bounce": float,              // P(bounce | tradeable) - GEOMETRY-BASED
        "strength_signed": float,       // Predicted signed strength
        "strength_abs": float,          // Predicted absolute strength
        "utility_score": float,         // Overall utility score
        "stage": str,                   // "stage_a" or "stage_b"
        "time_to_threshold": {          // Time-to-threshold predictions
            "t1": {"60": float, "120": float},
            "t2": {"60": float, "120": float}
        },
        "retrieval": {                  // kNN retrieval predictions (GEOMETRY-ONLY)
            "p_break": float,
            "similarity": float,        // Geometric similarity (32D DCT space)
            "entropy": float,
            "n_neighbors": int,         // Sample size
            "feature_set": "geometry_only"  // Explicit: 32D trajectory basis only
        }
    }
}
```

### Session Context Computation

`_compute_session_context(ts_ms)` returns:
- `is_first_15m: bool` - Whether timestamp is within first 15 minutes of market open
- `bars_since_open: int` - Minutes since 9:30 AM Eastern

---

## HTTP Endpoints

### Health Check

**URL**: `GET /health`
**Response**:
```json
{
  "service": "gateway",
  "status": "healthy",
  "nats_url": "nats://nats:4222",
  "connections": 3
}
```

**Usage**: Docker health checks, monitoring dashboards.

---

## Class Interfaces

### SocketBroadcaster

```python
class SocketBroadcaster:
    def __init__(self, bus: Optional[NATSBus] = None)

    async def start(self) -> None
    async def connect(self, websocket: WebSocket) -> None
    async def disconnect(self, websocket: WebSocket) -> None
    async def broadcast(self, message: Dict[str, Any]) -> None
    async def close(self) -> None
```

**State**:
- `active_connections: List[WebSocket]`: Connected clients
- `_latest_levels: Optional[Dict[str, Any]]`: Cached normalized levels payload
- `_latest_flow: Optional[Dict[str, Any]]`: Cached flow snapshot
- `_latest_viewport: Optional[Dict[str, Any]]`: Cached viewport data
- `bus: NATSBus`: NATS connection
- `_lock: asyncio.Lock`: Thread-safe connection management
- `_subscriptions: List`: Active NATS subscriptions

**Methods**:

#### `start()`
1. Connect to NATS (`await bus.connect()`)
2. Subscribe to `levels.signals` with callback `_on_level_signals`, durable consumer `gateway_levels`
3. Subscribe to `market.flow` with callback `_on_flow_snapshot`, durable consumer `gateway_flow`

#### `connect(websocket)`
1. Accept WebSocket connection
2. Add to active connections list (thread-safe with lock)
3. Build and send cached payload immediately (if available)

#### `disconnect(websocket)`
1. Remove from active connections list (thread-safe)
2. Idempotent (safe to call multiple times)

#### `broadcast(message)`
1. Serialize JSON once
2. Send to all active connections
3. Remove failed connections automatically

#### `_on_level_signals(data)` (internal callback)
1. Normalize payload via `_normalize_levels_payload(data)`
2. Update `_latest_levels` and `_latest_viewport`
3. Build merged payload via `_build_payload()`
4. Call `broadcast(payload)`

#### `_on_flow_snapshot(data)` (internal callback)
1. Update `_latest_flow = data`
2. Build merged payload via `_build_payload()`
3. Call `broadcast(payload)`

#### `_build_payload()` (internal)
Merges `_latest_flow`, `_latest_levels`, `_latest_viewport` into single dict.

#### `_normalize_levels_payload(payload)` (internal)
Normalizes raw Core payload to frontend contract:
- Extracts timestamp, spy snapshot
- Extracts viewport predictions and builds lookup by level_id
- Handles nested levels structures
- Calls `_normalize_level_signal` for each level with matched viewport predictions

#### `_normalize_level_signal(level, is_first_15m, bars_since_open, viewport_pred)` (internal)
Transforms individual level to normalized schema with signal/direction mapping. Merges ML predictions from viewport if available.

#### `_compute_session_context(ts_ms)` (internal)
Computes session timing context (is_first_15m, bars_since_open).

---

## Environment Variables

- `NATS_URL` (required): NATS server address (default: `nats://nats:4222`)
- `GATEWAY_PORT` (optional): HTTP/WS port (default: 8000)

---

## Entry Point

```bash
uv run python -m src.gateway.main
```

**Startup Output**:
```
============================================================
GATEWAY SERVICE
============================================================
  NATS URL: nats://nats:4222
  WebSocket: ws://localhost:8000/ws/stream
============================================================
✅ Gateway subscribed to NATS subjects
```

---

## Client Implementation Example

### TypeScript (Angular)

```typescript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => console.log('Connected to Gateway');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.flow, data.levels, data.viewport
  if (data.levels) {
    this.processLevelSignals(data.levels);
  }
  if (data.flow) {
    this.processFlowData(data.flow);
  }
};

ws.onerror = (err) => console.error('WebSocket error', err);

ws.onclose = () => {
  console.log('Disconnected');
  // Implement reconnection logic
};
```

### Python (Test Client)

```python
import asyncio
import websockets
import json

async def test_client():
    async with websockets.connect('ws://localhost:8000/ws/stream') as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'levels' in data:
                es = data['levels'].get('es', {})
                levels = data['levels'].get('levels', [])
                print(f"ES: {es.get('spot')}, Levels: {len(levels)}")

asyncio.run(test_client())
```

---

## Docker Compose Integration

```yaml
gateway:
  build:
    context: ./backend
    dockerfile: Dockerfile
  container_name: spymaster-gateway
  ports:
    - "8000:8000"
  environment:
    NATS_URL: nats://nats:4222
    PORT: 8000
  command: ["uv", "run", "python", "-m", "src.gateway.main"]
  depends_on:
    nats:
      condition: service_healthy
```

---

## Scalability Notes

**Current Design** (v1):
- Single Gateway instance
- Handles 100+ concurrent WebSocket clients
- Bottleneck: JSON serialization (single-threaded asyncio)

**Horizontal Scaling** (future):
- Run multiple Gateway replicas behind load balancer
- Each Gateway subscribes to NATS with **unique consumer** (broadcast pattern)
- All Gateways receive all messages → consistent client experience
- No sticky sessions required

**Configuration for Broadcast**:
```python
# In socket_broadcaster.py start():
await self.bus.subscribe(
    subject="levels.signals",
    callback=self._on_level_signals,
    durable_name=None  # Ephemeral consumer
)
```

---

## Monitoring Metrics

**Key Metrics**:
- Active WebSocket connections: `len(active_connections)`
- NATS message rate: Messages/sec on `levels.signals` and `market.flow`
- WebSocket send errors: Failed broadcast count
- NATS consumer lag: Use `nats consumer info LEVEL_SIGNALS gateway_levels`

**Health Check**:
- Current: Simple `{"status":"healthy"}` response
- Future: Add `nats_connected`, `last_message_ts`, `consumer_lag`

---

## Error Handling

**NATS connection failure**:
- Service crashes at startup
- Docker Compose restarts automatically

**WebSocket send failure**:
- Log error, remove failed connection
- Does not crash service

**JSON serialization failure**:
- Should never happen (Core produces valid JSON)
- If occurs: log error, skip broadcast

---

## Critical Invariants

1. **Relay with normalization**: Normalizes signals to frontend contract
2. **Dual subscription**: Subscribes to both `levels.signals` and `market.flow`
3. **Cache latest**: New connections receive immediate state
4. **Fan-out**: Single JSON serialization per broadcast
5. **Automatic cleanup**: Failed connections removed automatically
6. **NATS durability**: Durable consumers survive Gateway restarts
7. **Signal mapping**: REJECT→BOUNCE, CONTESTED/NEUTRAL→NO_TRADE
8. **ML prediction merging**: Viewport predictions matched by level_id and merged into normalized levels
9. **Confluence preservation**: All confluence features preserved from Core output

---

## References

- Full module documentation: `backend/src/gateway/README.md`
- NATS subjects: `backend/src/common/bus.py`
- Level signals schema: `backend/src/core/INTERFACES.md`
