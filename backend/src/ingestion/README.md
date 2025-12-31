# Ingestion Module

**Purpose**: Data acquisition from external sources  
**Status**: Production  
**Architecture**: Source contracts → Normalized events → NATS

---

## Overview

The ingestion module handles data acquisition from external vendors, normalizing data to canonical event types and publishing to NATS JetStream for downstream consumption.

**Key Principle**: Single point of vendor integration. All downstream services consume normalized events from NATS, never directly from vendors.

---

## Architecture

```
External Sources → Ingestion → NATS JetStream → Downstream Services
                                                   ├── Bronze Writer (io/)
                                                   ├── Core Service
                                                   └── Gateway
```

---

## Components

### Databento Integration (`databento/`)

#### DBNReader (`dbn_reader.py`)
Streams Databento DBN files with iterator pattern for memory efficiency.

**Key Methods**:
```python
from src.ingestion.databento.dbn_reader import DBNReader

reader = DBNReader()

# Discover available dates
dates = reader.get_available_dates('trades')

# Stream trades
for trade in reader.read_trades(date='2024-12-16', symbol_prefix='ES'):
    # Process FuturesTrade event
    pass

# Stream MBP-10 depth
for mbp in reader.read_mbp10(date='2024-12-16', symbol_prefix='ES'):
    # Process MBP10 event
    pass
```

**Features**:
- Streaming iterators (handles 10GB+ files)
- Front-month filtering (outright contracts only)
- Time-based filtering (start_ns, end_ns)
- Symbol filtering (avoid spreads/strategies)

#### ReplayPublisher (`replay.py`)
Publishes DBN data to NATS at configurable speed for testing/backtesting.

**Usage**:
```bash
cd backend
export REPLAY_SPEED=1.0      # 1x realtime (0=fast, 2=2x speed)
export REPLAY_DATE=2024-12-16
uv run python -m src.ingestion.databento.replay
```

**NATS Subjects Published**:
- `market.futures.trades` (ES futures trades)
- `market.futures.mbp10` (ES MBP-10 depth)
- `market.options.trades` (ES options trades, optional)

**Speed Control**:
- `0.0` = as fast as possible
- `1.0` = realtime (1s event time = 1s wall time)
- `2.0` = 2x speed

---

## Data Sources

**Databento GLBX.MDP3** (CME Globex):
- ES futures: trades + MBP-10 depth
- ES options: trades + NBBO

**Location**: `data/raw/trades/`, `data/raw/MBP-10/`

---

## Normalization Contract

All events are normalized to canonical types defined in `src.common.event_types`:

**Time**: Nanosecond precision (Unix nanoseconds)  
**Aggressor**: Explicit (BUY/SELL/MID enum)  
**Source**: Tagged (DIRECT_FEED, REPLAY, SIM)

---

## NATS Subject Hierarchy

**Published**:
- `market.futures.trades` → `FuturesTrade`
- `market.futures.mbp10` → `MBP10`
- `market.options.trades` → `OptionTrade`

**Stream**: `MARKET_DATA` (24h retention, file-backed)

---

## Future: Live Ingestion

Current implementation uses replay from DBN files. For live production:

1. Replace `replay.py` with live Databento streaming client
2. Use same NATS subjects (transparent to downstream)
3. Add connection management and failover

---

## Performance

**Replay Mode**:
- Fast (0x): ~100k events/sec
- Realtime (1x): Matches wall clock
- Memory: ~200-500MB (bounded by iterators)

---

## Testing

```bash
cd backend

# Check available data
uv run python -c "
from src.ingestion.databento.dbn_reader import DBNReader
reader = DBNReader()
print(reader.get_available_dates('trades'))
"

# Test replay
export REPLAY_DATE=2024-12-16
export REPLAY_SPEED=0.0
uv run python -m src.ingestion.databento.replay
```

---

## References

- **Event Types**: `backend/src/common/event_types.py`
- **NATS Bus**: `backend/src/common/bus.py`
- **Bronze Writer**: `backend/src/io/bronze.py` (consumes from NATS)
