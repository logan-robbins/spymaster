# Ingestor Module

**Role**: Data ingestion and normalization layer  
**Audience**: Backend developers working on data feeds  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Ingests market data from live feeds (Polygon WebSocket) or historical files (Databento DBN), normalizes to canonical event types, and publishes to NATS JetStream.

**Key responsibility**: Be the single point of vendor integration. All downstream services consume normalized events from NATS, never directly from vendors.

---

## Architecture

```
Data Sources → Ingestor → NATS JetStream → Core/Lake/Gateway
```

**Live path**: Polygon WebSocket → StreamIngestor → NATS  
**Replay path**: Databento DBN files → DBNIngestor → ReplayPublisher → NATS

**Transparency principle**: Downstream services cannot distinguish live from replay (same NATS subjects, same schemas).

---

## Components

### StreamIngestor (`stream_ingestor.py`)
Live WebSocket adapter for Polygon feeds. Handles SPY equity (trades, quotes) and SPY options (trades).

**Dynamic strikes**: Option subscriptions automatically update as SPY price moves (managed by `StrikeManager`).

### DBNIngestor (`dbn_ingestor.py`)
Databento DBN file reader with streaming iterators. Reads ES futures trades and MBP-10 depth updates.

**Iterator pattern**: Yields events one-by-one (no full file loads) to handle large MBP-10 files (10GB+).

### ReplayPublisher (`replay_publisher.py`)
Publishes DBN data to NATS at configurable speed. Merges trades + MBP-10 into single time-ordered stream.

**Speed control**: `0.0` = fast as possible, `1.0` = realtime, `2.0` = 2x speed

---

## Normalization Contract

**Time units**: Polygon sends milliseconds → multiply by 1,000,000 to get nanoseconds  
**Aggressor**: Inferred from bid/ask for Polygon, explicit in Databento  
**Source tagging**: Every event carries `EventSource` enum (`POLYGON_WS`, `DIRECT_FEED`, `REPLAY`)

**Output schemas**: See [INTERFACES.md](INTERFACES.md) for event type specifications.

---

## NATS Subject Hierarchy

**Published subjects**:
- `market.stocks.trades`: SPY equity trades
- `market.stocks.quotes`: SPY equity quotes
- `market.options.trades`: SPY option trades
- `market.futures.trades`: ES futures trades
- `market.futures.mbp10`: ES MBP-10 depth

**Stream config**: 24-hour retention, file-backed persistence, durable consumers.

---

## Data Sources

**Polygon WebSocket**:
- Endpoint: `wss://socket.massive.com/stocks` and `/options`
- Auth: API key from `POLYGON_API_KEY` env var
- Subscriptions: `T.SPY`, `Q.SPY`, dynamic `T.O:SPY*` for options

**Databento DBN files**:
- Location: `dbn-data/trades/`, `dbn-data/MBP-10/`
- Dataset: GLBX.MDP3 (CME Globex)
- Symbology: Continuous front month (`ES.c.0`)

---

## Running

### Live Ingestion
```bash
export POLYGON_API_KEY="your_key"
uv run python -m src.ingestor.main
```

### Replay
```bash
export REPLAY_SPEED=1.0
export REPLAY_DATE=2025-12-16
uv run python -m src.ingestor.replay_publisher
```

### Docker Compose
```bash
docker-compose up ingestor -d
# Uses replay_publisher as entry point
```

---

## Performance

**Live ingestion**:
- Latency: ~5-15ms (Polygon → NATS publish)
- Throughput: 10k+ events/sec
- Memory: ~50-100MB steady state

**Replay**:
- Fast mode (0x): ~100k events/sec
- Realtime (1x): Matches wall clock
- Memory: ~200-500MB (bounded by iterator pattern)

---

## Option Ticker Parsing

**Format**: `O:SPY{YYMMDD}{C|P}{strike*1000}`  
**Example**: `O:SPY251216C00676000`
- Underlying: SPY
- Expiration: 2025-12-16
- Right: C (call)
- Strike: 676.0

**See**: [INTERFACES.md](INTERFACES.md) for parsing logic.

---

## Testing

Integration testing (requires NATS):

```bash
# Subscribe to verify publishing
nats sub "market.>"

# Check stream info
nats stream info MARKET_DATA

# Manual test client
uv run python backend/src/ingestor/test_stream.py
```

---

## Common Issues

**Polygon disconnect**: Auto-reconnects (may lose 1-2 seconds data)  
**Missing API key**: Set `POLYGON_API_KEY` environment variable  
**NATS connection failure**: Service crashes → Docker restarts automatically  
**DBN file missing**: Check `dbn-data/` directory structure  
**Out of memory**: Should not occur (iterator pattern) → check for buffer leaks

---

## Failure Modes

| Failure | Behavior | Recovery |
|---------|----------|----------|
| Polygon disconnect | Auto-reconnect | Automatic |
| NATS disconnect | Service crashes | Restart (NATS retains 24h) |
| Invalid API key | Exit with error | Set correct env var |
| Malformed vendor data | Log error, skip event | Continues processing |

---

## References

- **Interface contract**: [INTERFACES.md](INTERFACES.md)
- **Event types**: [../common/INTERFACES.md](../common/INTERFACES.md)
- **NATS subjects**: [../common/bus.py](../common/bus.py)
- **PLAN.md**: §11 (Vendor Contracts)

---

**Phase**: Phase 2 (microservices with NATS)  
**Agent assignment**: Phase 2 Agent A  
**Dependencies**: `common` (event types, NATS bus)  
**Consumers**: Core, Lake
