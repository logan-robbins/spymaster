# Ingestor Module Interfaces

**Module**: `backend/src/ingestor/`
**Role**: Data ingestion and normalization
**Audience**: AI Coding Agents

---

## Module Purpose

Ingests market data from Databento DBN files, normalizes to canonical event types, and publishes to NATS JetStream.

---

## Output Interfaces

### NATS Subject Contracts

All normalized events published to NATS subjects:

| Subject | Event Type | Schema | Cadence |
|---------|-----------|--------|---------|
| `market.futures.trades` | FuturesTrade | `futures.trades.v1` | Replay or live |
| `market.futures.mbp10` | MBP10 | `futures.mbp10.v1` | Replay or live |
| `market.options.trades` | OptionTrade | `options.trades.v1` | Replay or live |

**Message Format**: JSON-serialized event dataclass from `src.common.event_types`.

---

## Normalization Contract

### Event Envelope (Required Fields)

Every event MUST include:
```python
{
    "ts_event_ns": int,      # Event time (Unix nanoseconds UTC)
    "ts_recv_ns": int,       # Receive time (Unix nanoseconds UTC)
    "source": str            # EventSource enum value
}
```

### EventSource Enum Values

```python
class EventSource(Enum):
    REPLAY = "replay"
    SIM = "sim"
    DIRECT_FEED = "direct_feed"
```

### Time Unit Conversion

**Databento DBN**: Timestamps already in nanoseconds
```python
ts_event_ns = dbn_record.ts_event  # No conversion
```

### Aggressor Classification

**Databento**:
- `side = 'B'` → `Aggressor.BUY`
- `side = 'A'` → `Aggressor.SELL`
- `side = 'N'` → `Aggressor.MID`

---

## Replay Interface (DBNIngestor + ReplayPublisher)

### DBNIngestor

**Purpose**: Read Databento DBN files with streaming iterators.

```python
class DBNIngestor:
    def __init__(self, dbn_data_root: Optional[str] = None)

    def read_trades(
        self,
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> Iterator[FuturesTrade]

    def read_mbp10(
        self,
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> Iterator[MBP10]

    def get_available_dates(self, schema: str) -> List[str]
```

**Iterator Pattern**: Yields events one-by-one (no full file load).

### ReplayPublisher

**Purpose**: Publish DBN data (and optional Bronze options) to NATS at configurable speed.

```python
@dataclass
class ReplayStats:
    events_published: int
    trades_published: int
    mbp10_published: int
    options_published: int
    start_time: Optional[float]
    first_event_ts: Optional[int]
    last_event_ts: Optional[int]

    def elapsed_wall_time(self) -> float
    def elapsed_event_time_sec(self) -> float
    def actual_speed(self) -> float

class ReplayPublisher:
    def __init__(
        self,
        bus: NATSBus,
        dbn_ingestor: DBNIngestor,
        replay_speed: float = 1.0,
        bronze_reader: Optional[BronzeReader] = None
    )

    async def replay_date(
        self,
        date: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        include_trades: bool = True,
        include_mbp10: bool = True,
        include_options: bool = False
    ) -> None

    async def replay_continuous(
        self,
        dates: Optional[List[str]] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        include_options: bool = False
    ) -> None
```

**Speed Control**:
```python
event_delta_ns = event.ts_event_ns - prev_event_ns
wall_delay_sec = (event_delta_ns / 1e9) / replay_speed

# Examples:
# replay_speed = 0.0  → no delay (fast as possible)
# replay_speed = 1.0  → wall_delay = event_delta (realtime)
# replay_speed = 2.0  → wall_delay = event_delta / 2 (2x speed)
```

**Stream-Merge Algorithm**: Events from trades, mbp10, and options iterators are merged by `ts_event_ns` to maintain proper time ordering without materializing full day in memory.

---

## Data File Locations

### DBN Files

```
project_root/dbn-data/
  trades/
    glbx-mdp3-YYYYMMDD.trades.dbn
    symbology.json
    metadata.json
  MBP-10/
    glbx-mdp3-YYYYMMDD.mbp-10.dbn
    symbology.json
```

## Environment Variables

### Replay
- `REPLAY_SPEED` (optional): Replay speed multiplier (default: 1.0)
- `REPLAY_DATE` (optional): Single date YYYY-MM-DD, or omit for all dates
- `REPLAY_INCLUDE_OPTIONS` (optional): Include Bronze option trades (default: `false`)
- `NATS_URL` (optional): NATS server URL (default: `nats://localhost:4222`)

---

## Entry Points

### Replay
```bash
export REPLAY_SPEED=1.0
export REPLAY_DATE=2025-12-16
export REPLAY_INCLUDE_OPTIONS=true
uv run python -m src.ingestor.replay_publisher
```

---

## Performance Characteristics

**Replay**:
- Fast mode (0x): ~100k events/sec
- Realtime (1x): Matches wall clock
- Memory: ~200-500MB (iterator pattern keeps bounded)
- Progress updates every 10k events

---

## Critical Invariants

1. **Time discipline**: All events carry both `ts_event_ns` and `ts_recv_ns`
2. **Source tagging**: Events tagged with correct `EventSource` enum
3. **Replay transparency**: Live and replay use same NATS subjects
4. **Iterator pattern**: No full-file loads for DBN (memory safety)
5. **Stream-merge ordering**: Events published in strict `ts_event_ns` order

---

## References

- Full module documentation: `backend/src/ingestor/README.md`
- Event types: `backend/src/common/event_types.py`
- NATS subjects: `backend/src/common/bus.py`
- Pipeline context: `backend/src/pipeline/README.md`
