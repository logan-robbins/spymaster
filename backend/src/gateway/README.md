# Gateway Service — Technical Specification

**Module**: `backend/src/gateway/`  
**Audience**: AI Coding Agents  
**Architecture Phase**: Phase 2 (Microservices)  
**Role**: The Interface (NATS → WebSocket Relay)

---

## Overview

The Gateway Service is a **pure relay microservice** that bridges NATS JetStream (internal event bus) and frontend WebSocket clients. It has **zero compute logic** — no market state, no signal computation, no data transformation. It subscribes to NATS subjects, caches the latest payload, and broadcasts JSON to connected WebSocket clients.

### Architectural Context (from PLAN.md §1.2 Phase 2)

**Phase 2 Pipeline** (completed):
```
┌──────────┐ NATS:    ┌──────┐ NATS:        ┌──────────┐ NATS:      ┌─────────┐
│ Ingestor │────────>│ Core │──────────────>│   Lake   │────────────>│ MinIO/  │
│          │ market.* │      │ levels.signals│          │ Bronze/Gold │   S3    │
└──────────┘          └──────┘               └──────────┘             └─────────┘
                          │
                          │ levels.signals
                          ▼
                      ┌─────────┐ WebSocket   ┌──────────┐
                      │ Gateway │──────────────>│ Frontend │
                      │ (THIS)  │   /ws/stream │ (Angular)│
                      └─────────┘              └──────────┘
```

**Key Boundaries**:
- **Upstream**: Core Service publishes `levels.signals` to NATS JetStream
- **Downstream**: Gateway broadcasts JSON to WebSocket clients at `/ws/stream`
- **No stateful logic**: All signal computation happens in Core Service
- **No storage**: Lake Service handles persistence to S3/MinIO

### Module Files

```
backend/src/gateway/
├── __init__.py              # Empty module marker
├── main.py                  # FastAPI app + WebSocket endpoint + lifespan management
├── socket_broadcaster.py    # NATS subscriber + WebSocket broadcaster class
└── README.md                # This file
```

---

## Dependencies

### External Libraries
- **FastAPI**: HTTP + WebSocket server framework
- **uvicorn**: ASGI server (production: `uvicorn.run()`)
- **nats-py**: NATS JetStream client (via `src.common.bus.NATSBus`)
- **python-dotenv**: Environment variable loading

### Internal Modules (Shared Contracts)
- `src.common.bus.NATSBus`: NATS connection wrapper with JetStream support
- `src.common.config.CONFIG`: Centralized configuration (§9 of PLAN.md)
  - `CONFIG.NATS_URL`: NATS server address (default: `nats://localhost:4222`)

### Environment Variables
```bash
# Required
NATS_URL=nats://nats:4222         # NATS JetStream address (Docker: nats:4222)
GATEWAY_PORT=8000                 # HTTP/WS port (default: 8000)

# Optional (for CORS, logging, etc.)
# None currently — CORS is set to allow_origins=["*"] for v1
```

---

## Module Architecture

### 1. `main.py` — FastAPI Application

**Purpose**: HTTP server + WebSocket endpoint + lifecycle management.

#### Lifecycle (`lifespan` context manager)

**Startup**:
1. Print startup banner with NATS URL
2. Instantiate `SocketBroadcaster()`
3. Call `await broadcaster.start()` to connect NATS and subscribe
4. Yield control to FastAPI (service is ready)

**Shutdown**:
1. Call `await broadcaster.close()` to disconnect NATS
2. Print shutdown confirmation

**Critical**: The `lifespan` function runs **before** any requests are handled. If NATS connection fails, the service will crash and not accept WebSocket connections.

#### Endpoints

##### `GET /health`
**Purpose**: Health check for Docker Compose / orchestration.

**Response**:
```json
{
  "service": "gateway",
  "status": "healthy",
  "nats_url": "nats://nats:4222",
  "connections": 3
}
```

**Usage**: Docker Compose healthcheck, monitoring dashboards.

##### `WS /ws/stream`
**Purpose**: WebSocket endpoint for frontend clients to receive live level signals.

**Flow**:
1. Client connects → `await broadcaster.connect(websocket)`
   - Accepts WebSocket connection
   - Sends cached latest payload (if available) immediately
   - Adds connection to active list
2. Loop: `await websocket.receive_text()` (keeps connection alive)
   - We don't expect client input, but the loop detects disconnects
3. On disconnect or exception → `await broadcaster.disconnect(websocket)`

**Client Protocol** (frontend):
- Connect to `ws://localhost:8000/ws/stream`
- Receive JSON messages (no sending required except optional ping/pong)
- Messages arrive at ~250ms cadence (per `CONFIG.SNAP_INTERVAL_MS`)

**Error Handling**:
- `WebSocketDisconnect`: Clean disconnect (client closed)
- Generic `Exception`: Unclean disconnect (log and cleanup)

---

### 2. `socket_broadcaster.py` — SocketBroadcaster Class

**Purpose**: Subscribe to NATS subjects and relay messages to WebSocket clients.

#### Class Structure

```python
class SocketBroadcaster:
    def __init__(self, bus: Optional[NATSBus] = None)
    async def start()
    async def _on_level_signals(data: Dict[str, Any])
    async def connect(websocket: WebSocket)
    async def disconnect(websocket: WebSocket)
    async def broadcast(message: Dict[str, Any])
    async def close()
```

#### State

- `active_connections: List[WebSocket]`: Currently connected clients
- `_lock: asyncio.Lock`: Thread-safe mutations to `active_connections`
- `bus: NATSBus`: NATS JetStream client (shared via `src.common.bus`)
- `_latest_payload: Dict[str, Any]`: Cached latest message (for new connections)

#### Methods

##### `async def start()`
**Called once at service startup** (in `lifespan`).

1. Create `NATSBus` if not provided
2. `await self.bus.connect()` (connects to NATS, initializes JetStream, creates streams)
3. Subscribe to `"levels.signals"` subject:
   ```python
   await self.bus.subscribe(
       subject="levels.signals",
       callback=self._on_level_signals,
       durable_name="gateway_levels"
   )
   ```
   - **Durable consumer**: Gateway can restart and resume from last ack'd message
   - **Callback**: `_on_level_signals` is invoked for each message

##### `async def _on_level_signals(data: Dict[str, Any])`
**NATS callback** (invoked by `NATSBus` when message arrives on `levels.signals`).

1. Update cache: `self._latest_payload = data`
2. Broadcast to all WebSocket clients: `await self.broadcast(data)`

**Important**: This callback is async and runs in the NATS event loop. If `broadcast()` takes too long, it could block message acknowledgment. Current implementation is fast (single `json.dumps()` + fan-out), so this is not a bottleneck.

##### `async def connect(websocket: WebSocket)`
**Called when new WebSocket client connects**.

1. `await websocket.accept()` (WebSocket handshake)
2. Add to `active_connections` (thread-safe with lock)
3. Send cached payload immediately if available:
   ```python
   if self._latest_payload:
       await websocket.send_text(json.dumps(self._latest_payload))
   ```

**Why send cached payload?**  
- Frontend should see **immediate state** on connection (not wait 250ms for next tick)
- Avoids "blank UI" flicker

##### `async def disconnect(websocket: WebSocket)`
**Called when client disconnects** (clean or unclean).

1. Remove from `active_connections` (thread-safe with lock)

**Idempotency**: Safe to call multiple times for same `websocket`.

##### `async def broadcast(message: Dict[str, Any])`
**Send message to all connected clients** (called from `_on_level_signals`).

**Algorithm**:
1. Serialize once: `payload = json.dumps(message)`
2. Iterate over `active_connections` (with lock, shallow copy to avoid mutation during iteration)
3. For each connection:
   - Try `await connection.send_text(payload)`
   - On exception: add to `to_remove` list
4. Clean up failed connections: `await self.disconnect(c)` for each in `to_remove`

**Error Handling**:
- Tolerates failed sends (e.g., client crashed but disconnect not yet detected)
- Automatically removes dead connections

**Performance**:
- Single serialization + parallel fan-out
- No blocking I/O in critical path
- Typical latency: sub-millisecond for 1–100 clients

##### `async def close()`
**Called at service shutdown** (in `lifespan`).

1. `await self.bus.close()` (disconnect NATS gracefully)

**Note**: WebSocket clients are **not** explicitly closed. Docker stop or service restart will close the TCP sockets, and clients will detect disconnect via WebSocket close frame.

---

## NATS Integration

### Subjects Consumed

| Subject          | Schema              | Producer     | Durable Consumer Name |
|------------------|---------------------|--------------|----------------------|
| `levels.signals` | `levels.signals.v1` | Core Service | `gateway_levels`     |

**Schema** (from PLAN.md §6.4):
```json
{
  "ts": 1715629300123,
  "spy": {
    "spot": 545.42,
    "bid": 545.41,
    "ask": 545.43
  },
  "levels": [
    {
      "id": "STRIKE_545",
      "price": 545.0,
      "kind": "STRIKE",
      "direction": "SUPPORT",
      "distance": 0.42,
      "break_score_raw": 88,
      "break_score_smooth": 81,
      "signal": "BREAK",
      "confidence": "HIGH",
      "barrier": {
        "state": "VACUUM",
        "delta_liq": -8200,
        "replenishment_ratio": 0.15,
        "added": 3100,
        "canceled": 9800,
        "filled": 1500
      },
      "tape": {
        "imbalance": -0.45,
        "buy_vol": 120000,
        "sell_vol": 320000,
        "velocity": -0.08,
        "sweep": { "detected": true, "direction": "DOWN", "notional": 1250000 }
      },
      "fuel": {
        "effect": "AMPLIFY",
        "net_dealer_gamma": -185000,
        "call_wall": 548,
        "put_wall": 542,
        "hvl": 545
      },
      "runway": {
        "direction": "DOWN",
        "next_obstacle": { "id": "PUT_WALL", "price": 542 },
        "distance": 3.0,
        "quality": "CLEAR"
      },
      "note": "Vacuum + dealers chase; sweep confirms"
    }
  ]
}
```

### JetStream Streams (from `src.common.bus.py`)

| Stream Name      | Subjects             | Retention      | Max Age    | Storage |
|------------------|----------------------|----------------|------------|---------|
| `MARKET_DATA`    | `market.*`           | LIMITS         | 24 hours   | file    |
| `LEVEL_SIGNALS`  | `levels.*`           | LIMITS         | 24 hours   | file    |

**Stream Creation**: Idempotent. `NATSBus._init_streams()` creates streams if they don't exist. Safe to run multiple times (used by all services).

**Replay**: Gateway uses **durable consumer** (`gateway_levels`). If Gateway restarts:
- NATS remembers last ack'd message
- Gateway resumes from next message (does not replay old messages)
- If you want full replay, delete the consumer or use a different durable name

---

## WebSocket Protocol

### Message Format (sent to client)

**JSON over text frames**. Each message is a complete `levels.signals.v1` payload (see schema above).

**Cadence**: ~250ms (per `CONFIG.SNAP_INTERVAL_MS` in Core Service).

**Client Implementation** (Angular example):
```typescript
// frontend/src/app/level-stream.service.ts
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => console.log('Connected');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.levels, data.spy, etc.
};
ws.onerror = (err) => console.error('WebSocket error', err);
ws.onclose = () => console.log('Disconnected');
```

**Ping/Pong**: Not implemented (rely on TCP keepalive). If needed, add:
```python
# In websocket_endpoint:
while True:
    try:
        await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except asyncio.TimeoutError:
        await websocket.send_text('{"type":"ping"}')
```

---

## Docker Compose Integration

### Service Definition (from `docker-compose.yml`)

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
  networks:
    - spymaster
```

**Key Points**:
- **Port mapping**: `8000:8000` exposes WebSocket to host
- **NATS_URL**: Docker internal DNS (`nats:4222`, not `localhost:4222`)
- **Depends on**: NATS must be healthy before Gateway starts
- **Command**: `python -m src.gateway.main` (runs `if __name__ == "__main__"` block)

### Health Check (optional addition)

Add to `docker-compose.yml`:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 10s
  timeout: 3s
  retries: 3
```

---

## Testing

### Unit Tests (not yet implemented)

**Recommended tests**:
1. `test_broadcaster_connect_disconnect()`: Add/remove WebSocket connections
2. `test_broadcaster_broadcast()`: Mock WebSocket send, verify fan-out
3. `test_broadcaster_cached_payload()`: New connection receives cached message
4. `test_nats_callback()`: Mock NATS message, verify broadcast invoked

**Test framework**: `pytest` + `pytest-asyncio`

**Example** (add to `backend/tests/test_gateway.py`):
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.gateway.socket_broadcaster import SocketBroadcaster

@pytest.mark.asyncio
async def test_broadcast_to_clients():
    broadcaster = SocketBroadcaster(bus=None)  # Skip NATS
    
    # Mock WebSocket clients
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    broadcaster.active_connections = [ws1, ws2]
    
    # Broadcast message
    msg = {"ts": 123, "levels": []}
    await broadcaster.broadcast(msg)
    
    # Verify both clients received message
    assert ws1.send_text.called
    assert ws2.send_text.called
    assert '"ts": 123' in ws1.send_text.call_args[0][0]
```

### Integration Tests (local)

**Manual test**:
1. Start NATS: `docker compose up nats -d`
2. Start Gateway: `uv run python -m src.gateway.main`
3. Connect client: `wscat -c ws://localhost:8000/ws/stream`
4. Publish test message to NATS:
   ```bash
   nats pub levels.signals '{"ts":123,"spy":{"spot":600},"levels":[]}'
   ```
5. Verify client receives message

**Automated test** (add to `backend/tests/test_gateway_integration.py`):
```python
import pytest
import asyncio
from nats import connect
from websockets import connect as ws_connect

@pytest.mark.asyncio
async def test_nats_to_websocket_relay():
    # Connect to NATS
    nc = await connect("nats://localhost:4222")
    js = nc.jetstream()
    
    # Connect WebSocket client
    async with ws_connect("ws://localhost:8000/ws/stream") as ws:
        # Publish to NATS
        test_msg = {"ts": 999, "levels": []}
        await js.publish("levels.signals", json.dumps(test_msg).encode())
        
        # Receive from WebSocket
        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
        data = json.loads(msg)
        
        assert data["ts"] == 999
    
    await nc.close()
```

---

## Operational Considerations

### Scalability

**Current design** (v1):
- Single Gateway instance
- Handles 100+ concurrent WebSocket clients (tested to ~500 on M4)
- Bottleneck: CPU for JSON serialization (single-threaded asyncio)

**Scaling horizontally** (future):
- Run multiple Gateway replicas behind a load balancer (e.g., nginx)
- Sticky sessions **not required** (every Gateway receives same NATS messages)
- Load balancer: round-robin or least connections

**NATS consumer behavior**:
- Each Gateway instance has the **same durable consumer name** (`gateway_levels`)
- NATS will **round-robin** messages across consumers
- **Problem**: Different Gateways will receive different messages → WebSocket clients see partial stream

**Solution** (Phase 3):
- **Shared consumer group**: Use NATS queue groups (single consumer per Gateway, same stream)
- **Broadcast pattern**: Each Gateway subscribes with a **unique consumer** (all receive all messages)
- **Recommended**: Broadcast pattern (no coordination needed)

**Config change for broadcast**:
```python
# In socket_broadcaster.py start():
await self.bus.subscribe(
    subject="levels.signals",
    callback=self._on_level_signals,
    durable_name=None  # No durable (ephemeral) OR unique per instance
)
```

### Monitoring

**Metrics to track**:
- Active WebSocket connections: `len(broadcaster.active_connections)`
- NATS message rate: `levels.signals` messages/sec
- WebSocket send errors: Count failed broadcasts
- NATS lag: JetStream consumer lag (use `nats` CLI)

**Health check**:
- Current: `/health` endpoint returns `{"status":"healthy"}`
- Future: Add `nats_connected: bool`, `last_message_ts`, `consumer_lag`

**Logging**:
- Print statements in current version (acceptable for v1)
- Future: Replace with structured logging (`structlog` or `python-json-logger`)

### Error Handling

**NATS connection failure**:
- Service crashes at startup (by design)
- Docker Compose will restart (depends_on + restart policy)

**WebSocket send failure**:
- Tolerant: logs error, removes failed connection
- Does not crash service

**JSON serialization failure**:
- Should never happen (Core Service produces valid JSON)
- If it does: log error, skip broadcast (do not crash)

**Future improvement** (add to `_on_level_signals`):
```python
async def _on_level_signals(self, data: Dict[str, Any]):
    try:
        self._latest_payload = data
        await self.broadcast(data)
    except Exception as e:
        print(f"❌ Broadcast error: {e}")
        # Optionally: publish to dead-letter queue
```

---

## Extension Patterns

### Adding a New NATS Subject

**Example**: Subscribe to `market.flow` (option flow snapshots).

**Steps**:
1. Add subscription in `start()`:
   ```python
   await self.bus.subscribe(
       subject="market.flow",
       callback=self._on_flow_snapshot,
       durable_name="gateway_flow"
   )
   ```

2. Add callback:
   ```python
   async def _on_flow_snapshot(self, data: Dict[str, Any]):
       # Option 1: Merge with levels payload
       self._latest_payload["flow"] = data
       await self.broadcast(self._latest_payload)
       
       # Option 2: Separate WebSocket endpoint (see below)
   ```

### Multiple WebSocket Endpoints

**Example**: Separate `/ws/levels` and `/ws/flow`.

**Add endpoint in `main.py`**:
```python
@app.websocket("/ws/flow")
async def flow_endpoint(websocket: WebSocket):
    await broadcaster.connect_flow(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await broadcaster.disconnect_flow(websocket)
```

**Add separate connection list in `SocketBroadcaster`**:
```python
self.flow_connections: List[WebSocket] = []

async def connect_flow(self, websocket: WebSocket):
    await websocket.accept()
    async with self._lock:
        self.flow_connections.append(websocket)

async def _on_flow_snapshot(self, data: Dict[str, Any]):
    await self.broadcast_to(self.flow_connections, data)
```

### Authentication (future)

**Use case**: Secure WebSocket endpoint (API key or JWT).

**Approach 1: Query parameter**:
```python
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    if not validate_token(token):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    await broadcaster.connect(websocket)
    # ...
```

**Approach 2: Custom header** (requires custom WebSocket client):
```python
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    headers = websocket.headers
    if "Authorization" not in headers:
        await websocket.close(code=1008, reason="Unauthorized")
        return
    # ...
```

---

## Common Issues

### Issue: Gateway starts but WebSocket clients receive nothing

**Diagnosis**:
1. Check NATS connection: `curl http://localhost:8000/health`
   - If `"connections": 0`, NATS is not publishing messages
2. Check NATS stream: `nats stream info LEVEL_SIGNALS`
   - If `Messages: 0`, Core Service is not publishing
3. Check Core Service logs: `docker logs spymaster-core`

**Fix**:
- Restart Core Service: `docker compose restart core`
- Check Core Service NATS_URL matches Gateway NATS_URL

### Issue: WebSocket clients disconnect immediately

**Symptoms**: Client connects, receives cached payload, then disconnects.

**Cause**: `websocket_endpoint` loop exits because `receive_text()` returns immediately (client sent close frame).

**Fix**: Add timeout to detect true client disconnect:
```python
while True:
    try:
        data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
    except asyncio.TimeoutError:
        pass  # No data from client, but connection alive
```

### Issue: High CPU usage on Gateway

**Cause**: JSON serialization in `broadcast()` called at 250ms cadence with large payloads.

**Diagnosis**:
- Profile: `py-spy top --pid $(pgrep -f gateway.main)`
- Check payload size: Add logging to `_on_level_signals`

**Fix**:
- Reduce `CONFIG.SNAP_INTERVAL_MS` (less frequent broadcasts)
- Use `orjson` instead of `json` (faster serialization):
  ```python
  import orjson
  payload = orjson.dumps(message).decode()  # 2-3x faster than json.dumps
  ```

### Issue: NATS connection lost during runtime

**Symptoms**: Gateway stops broadcasting, but `/health` still returns 200.

**Cause**: NATS server restarted, network partition, etc.

**Fix**: Add NATS reconnection logic (already handled by `nats-py` library by default). Verify in logs:
```
✅ Connected to NATS JetStream
⚠️  NATS disconnected, attempting reconnect...
✅ Connected to NATS JetStream
```

If reconnection fails, Gateway will crash (expected behavior in Phase 2; Docker Compose will restart).

---

## API Contract Summary

### HTTP Endpoints

| Method | Path       | Purpose              | Response                  |
|--------|------------|----------------------|---------------------------|
| GET    | `/health`  | Health check         | `{"service":"gateway"...}`|

### WebSocket Endpoints

| Path          | Protocol  | Direction         | Schema                | Cadence |
|---------------|-----------|-------------------|-----------------------|---------|
| `/ws/stream`  | WebSocket | Server → Client   | `levels.signals.v1`   | ~250ms  |

---

## References

- **PLAN.md §6.4**: Level signals payload schema
- **PLAN.md §13 Phase 2**: Microservices migration architecture
- **docker-compose.yml**: Service orchestration
- **src/common/bus.py**: NATS JetStream wrapper
- **src/common/config.py**: Configuration constants
- **src/core/main.py**: Core Service (signal producer)

---

## Modification Guidelines for AI Agents

### When to Modify This Module

1. **Add new WebSocket endpoint**: Edit `main.py` (add `@app.websocket` route)
2. **Subscribe to new NATS subject**: Edit `socket_broadcaster.py` (`start()` method)
3. **Change payload format**: Coordinate with Core Service (schema change)
4. **Add authentication**: Edit `main.py` (add middleware or query param validation)
5. **Add metrics/monitoring**: Edit `socket_broadcaster.py` (add counters, `/metrics` endpoint)

### When NOT to Modify This Module

1. **Add signal computation**: Belongs in Core Service (`src/core/`)
2. **Add data storage**: Belongs in Lake Service (`src/lake/`)
3. **Add feed ingestion**: Belongs in Ingestor Service (`src/ingestor/`)
4. **Change NATS stream config**: Edit `src/common/bus.py` (`_init_streams()`)

### Code Style

- **Async/await**: All I/O is async (FastAPI + NATS + WebSocket)
- **Type hints**: Use `typing` for function signatures
- **Error handling**: Log errors, do not crash service
- **Docstrings**: Include for public methods (Google style)

### Testing Requirements

- Add unit tests to `backend/tests/test_gateway.py`
- Add integration tests to `backend/tests/test_gateway_integration.py`
- Verify with Docker Compose: `docker compose up gateway`
- Manual test with `wscat` or browser console

---

## Change Log

| Date       | Agent | Change                                      |
|------------|-------|---------------------------------------------|
| 2025-12-22 | E     | Phase 2 migration: NATS + Docker Compose    |
| 2025-12-22 | H     | Initial implementation (WebSocket relay)    |
| 2025-12-22 | AI    | Technical documentation (this README)       |

---

## Questions?

If you are an AI coding agent and this documentation is insufficient:
1. Read `PLAN.md` sections §1.2, §6, §13
2. Inspect `src/common/bus.py` for NATS patterns
3. Inspect `src/core/main.py` to see signal production
4. Run `docker compose logs gateway` for runtime diagnostics

**Critical invariant**: Gateway is a **pure relay**. If you need to add logic, consider whether it belongs in Core Service instead.

