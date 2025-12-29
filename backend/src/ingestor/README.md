# Ingestor Module

**Role**: Data ingestion and normalization layer  
**Audience**: Backend developers working on data feeds  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Ingests market data from Databento DBN files, normalizes to canonical event types, and publishes to NATS JetStream.

**Key responsibility**: Be the single point of vendor integration. All downstream services consume normalized events from NATS, never directly from vendors.

---

## Architecture

```
Data Sources → Ingestor → NATS JetStream → Core/Lake/Gateway
```

**Replay path**: Databento DBN files → DBNIngestor → ReplayPublisher → NATS

**Transparency principle**: Downstream services cannot distinguish live from replay (same NATS subjects, same schemas).

---

## Components

### DBNIngestor (`dbn_ingestor.py`)
Databento DBN file reader with streaming iterators. Reads ES futures trades and MBP-10 depth updates.

**Iterator pattern**: Yields events one-by-one (no full file loads) to handle large MBP-10 files (10GB+).

### ReplayPublisher (`replay_publisher.py`)
Publishes DBN data to NATS at configurable speed. Merges futures trades + MBP-10 with optional Bronze ES options trades into a single time-ordered stream.

**Speed control**: `0.0` = fast as possible, `1.0` = realtime, `2.0` = 2x speed

---

## Normalization Contract

**Time units**: Databento DBN timestamps are already in nanoseconds  
**Aggressor**: Explicit in Databento (`side` field)  
**Source tagging**: Every event carries `EventSource` enum (`DIRECT_FEED`, `REPLAY`, `SIM`)

**Output schemas**: See [INTERFACES.md](INTERFACES.md) for event type specifications.

---

## NATS Subject Hierarchy

**Published subjects**:
- `market.options.trades`: ES option trades
- `market.futures.trades`: ES futures trades
- `market.futures.mbp10`: ES MBP-10 depth

**Stream config**: 24-hour retention, file-backed persistence, durable consumers.

---

## Data Sources

**Databento DBN files**:
- Location: `dbn-data/trades/`, `dbn-data/MBP-10/`
- Dataset: GLBX.MDP3 (CME Globex)
- Symbology: Continuous front month (`ES.c.0`)

---

## Running

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

**Replay**:
- Fast mode (0x): ~100k events/sec
- Realtime (1x): Matches wall clock
- Memory: ~200-500MB (bounded by iterator pattern)

---

## Testing

Integration testing (requires NATS):

```bash
# Subscribe to verify publishing
nats sub "market.>"

# Check stream info
nats stream info MARKET_DATA
```

---

## Common Issues

**NATS connection failure**: Service crashes → Docker restarts automatically  
**DBN file missing**: Check `dbn-data/` directory structure  
**Out of memory**: Should not occur (iterator pattern) → check for buffer leaks

---

## Failure Modes

| Failure | Behavior | Recovery |
|---------|----------|----------|
| NATS disconnect | Service crashes | Restart (NATS retains 24h) |
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
