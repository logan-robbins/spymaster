# Gateway Module

**Role**: WebSocket relay (NATS → Frontend)  
**Audience**: Backend developers working on frontend connectivity  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Pure relay microservice that bridges NATS JetStream and frontend WebSocket clients. Zero compute logic—subscribes to NATS subjects, caches latest payload, broadcasts JSON.

**Critical invariant**: Gateway is a relay. No signal computation or data transformation. All physics lives in Core Service.

---

## Architecture

```
NATS (levels.signals) → Gateway → WebSocket (frontend)
```

**Flow**:
1. Subscribe to `levels.signals` with durable consumer `gateway_levels`
2. Update cached payload
3. Broadcast to all connected WebSocket clients

---

## Components

### Main (`main.py`)
FastAPI application with WebSocket endpoint and lifecycle management.

**Endpoints**:
- `WS /ws/stream`: Level signals broadcast (~250ms cadence)
- `GET /health`: Service health check

### SocketBroadcaster (`socket_broadcaster.py`)
Core relay class that manages NATS subscription and WebSocket fan-out.

**Key methods**:
- `start()`: Connect NATS, subscribe to subjects
- `connect(websocket)`: Accept new client, send cached payload
- `broadcast(message)`: Fan-out to all connected clients
- `disconnect(websocket)`: Clean up client connection

---

## WebSocket Protocol

**URL**: `ws://localhost:8000/ws/stream`  
**Format**: JSON text frames  
**Cadence**: ~250ms (driven by Core Service snap interval)

**Connection flow**:
1. Client connects → Gateway accepts
2. Gateway sends cached payload immediately (no blank UI)
3. Gateway broadcasts subsequent updates
4. Client disconnect → Gateway cleans up

**See**: [INTERFACES.md](INTERFACES.md) for message schema.

---

## Running

### Standalone
```bash
uv run python -m src.gateway.main
```

### Docker Compose
```bash
docker-compose up gateway -d
```

**Health check**:
```bash
curl http://localhost:8000/health
```

---

## Scalability

**Current design** (v1):
- Single Gateway instance
- Handles 100+ concurrent WebSocket clients
- Bottleneck: JSON serialization (single-threaded asyncio)

**Horizontal scaling** (future):
- Run multiple replicas behind load balancer
- Each Gateway subscribes with **unique NATS consumer** (broadcast pattern)
- All Gateways receive all messages → consistent client experience
- No sticky sessions required

---

## Performance

**Latency**: <5ms (NATS receive → WebSocket send)  
**Throughput**: 100+ concurrent clients at 4 Hz  
**Memory**: ~50MB base + ~1MB per 100 clients

---

## Error Handling

**NATS connection failure**: Service crashes at startup → Docker restarts  
**WebSocket send failure**: Log error, remove failed connection, continue  
**JSON serialization failure**: Should never happen (Core produces valid JSON)

---

## Monitoring Metrics

**Key metrics to track**:
- Active WebSocket connections: `len(active_connections)`
- NATS message rate: Messages/sec on `levels.signals`
- WebSocket send errors: Failed broadcast count
- NATS consumer lag: Use `nats consumer info`

**Health check** (current): Simple `{"status":"healthy"}`  
**Health check** (future): Add `nats_connected`, `last_message_ts`, `consumer_lag`

---

## Common Issues

**Gateway starts but clients receive nothing**:
- Check Core Service is publishing: `nats sub "levels.signals"`
- Check NATS stream: `nats stream info LEVEL_SIGNALS`
- Verify Gateway NATS_URL matches Core NATS_URL

**WebSocket clients disconnect immediately**:
- Add timeout to detect true disconnect (see INTERFACES.md for code example)

**High CPU usage**:
- Profile with `py-spy top`
- Consider using `orjson` instead of `json` (2-3x faster)

**NATS connection lost during runtime**:
- `nats-py` handles auto-reconnect
- If reconnection fails persistently, service crashes → Docker restarts

---

## Extension Patterns

### Adding New NATS Subject

To relay additional subjects (e.g., `market.flow`):

1. Add subscription in `socket_broadcaster.py`
2. Define callback handler
3. Merge with existing payload or create new WebSocket endpoint

**See**: Gateway README (old version) §Extension Patterns for code example.

### Multiple WebSocket Endpoints

To create separate endpoints (e.g., `/ws/levels`, `/ws/flow`):

1. Add `@app.websocket()` route in `main.py`
2. Add separate connection list in `SocketBroadcaster`
3. Subscribe to appropriate NATS subjects

---

## Testing

Manual testing:

```bash
# Test with wscat
wscat -c ws://localhost:8000/ws/stream

# Test with Python
cd backend
uv run python -c "
import asyncio, websockets, json
async def test():
    async with websockets.connect('ws://localhost:8000/ws/stream') as ws:
        msg = await ws.recv()
        data = json.loads(msg)
        print(f'SPY: {data[\"levels\"][\"spy\"][\"spot\"]}')
asyncio.run(test())
"
```

---

## References

- **Interface contract**: [INTERFACES.md](INTERFACES.md)
- **Level signals schema**: [../core/INTERFACES.md](../core/INTERFACES.md)
- **NATS subjects**: [../common/bus.py](../common/bus.py)
- **PLAN.md**: §6.4 (WebSocket Payload)

---

**Phase**: Phase 2 (microservices)  
**Agent assignment**: Agent H (WebSocket relay)  
**Dependencies**: `common` (NATS bus)  
**Consumers**: Frontend (Angular)
