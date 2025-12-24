# Gateway Module Interfaces

**Module**: `backend/src/gateway/`  
**Role**: WebSocket relay (NATS → Frontend)  
**Audience**: AI Coding Agents

---

## Module Purpose

Pure relay service that bridges NATS JetStream and frontend WebSocket clients. Zero compute logic—subscribes to NATS subjects, caches latest payload, broadcasts JSON.

---

## Input Interface

### NATS Subscription

**Subject**: `levels.signals`  
**Consumer**: Durable consumer `gateway_levels`  
**Schema**: `levels.signals.v1` (from Core Service)

**Payload Format**: See `backend/src/core/INTERFACES.md` for full schema.

**Processing**:
1. Receive message from NATS
2. Update cached payload
3. Broadcast to all connected WebSocket clients

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

**Message Format**: JSON-serialized level signals payload (same as NATS input).

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
    def __init__(bus: Optional[NATSBus] = None)
    
    async def start() -> None
    async def connect(websocket: WebSocket) -> None
    async def disconnect(websocket: WebSocket) -> None
    async def broadcast(message: Dict[str, Any]) -> None
    async def close() -> None
```

**State**:
- `active_connections: List[WebSocket]`: Connected clients
- `_latest_payload: Dict[str, Any]`: Cached last message
- `bus: NATSBus`: NATS connection

**Methods**:

#### `start()`
1. Connect to NATS (`await bus.connect()`)
2. Subscribe to `levels.signals` with callback `_on_level_signals`
3. Use durable consumer `gateway_levels` for restart resume

#### `connect(websocket)`
1. Accept WebSocket connection
2. Add to active connections list
3. Send cached payload immediately (if available)

#### `disconnect(websocket)`
1. Remove from active connections list
2. Idempotent (safe to call multiple times)

#### `broadcast(message)`
1. Serialize JSON once
2. Send to all active connections
3. Remove failed connections automatically

#### `_on_level_signals(data)` (internal callback)
1. Update `_latest_payload = data`
2. Call `broadcast(data)`

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
```

---

## Client Implementation Example

### TypeScript (Angular)

```typescript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => console.log('Connected to Gateway');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.spy, data.levels, data.viewport
  this.processLevelSignals(data);
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
            print(f"SPY: {data['spy']['spot']}, Levels: {len(data['levels'])}")

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
- NATS message rate: Messages/sec on `levels.signals`
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

1. **Pure relay**: No signal computation or data transformation
2. **Cache latest**: New connections receive immediate state
3. **Fan-out**: Single JSON serialization per broadcast
4. **Automatic cleanup**: Failed connections removed automatically
5. **NATS durability**: Durable consumer survives Gateway restarts

---

## References

- Full module documentation: `backend/src/gateway/README.md`
- NATS subjects: `backend/src/common/bus.py`
- Level signals schema: `backend/src/core/INTERFACES.md`

