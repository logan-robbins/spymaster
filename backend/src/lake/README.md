# Lake Module — Technical Specification

**Module**: `backend/src/lake`  
**Purpose**: Institutional-grade data persistence implementing Bronze/Silver/Gold lakehouse architecture per PLAN.md §2.2  
**Audience**: AI Coding Agents  
**Phase**: Phase 2 (NATS + S3/MinIO) — Complete as of Phase 2 Agent B deliverable

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Component Specifications](#component-specifications)
3. [Data Flow & Contracts](#data-flow--contracts)
4. [Storage Schemas](#storage-schemas)
5. [Usage Patterns](#usage-patterns)
6. [Extending the Lake](#extending-the-lake)
7. [Testing & Replay](#testing--replay)
8. [Migration Notes (Phase 1 → Phase 2)](#migration-notes-phase-1--phase-2)

---

## Overview & Architecture

### Design Principles (PLAN.md §1.1)

The Lake module implements a **lakehouse-ready, append-only, replayable** storage layer with:

- **Event-time first**: Every record carries `ts_event_ns` (event time UTC) and `ts_recv_ns` (ingest time UTC)
- **Schema versioning**: All schemas include version metadata for evolution tracking
- **Idempotency**: At-least-once inputs become exactly-once storage via deterministic dedup keys
- **Separation of concerns**: Bronze (raw) → Silver (clean) → Gold (derived) with explicit transform boundaries
- **ML-friendly**: Columnar Parquet, Hive partitioning, ZSTD compression, sorted by event time
- **Upgrade path**: Designed for local M4 dev → S3/MinIO → Iceberg metadata layer

### Lakehouse Tiers (PLAN.md §2.2)

```
Bronze (Raw, Immutable)
├── Purpose: Near-raw normalized events, full replay capability
├── Format: Parquet with ZSTD compression level 3
├── Partitioning: symbol/underlying + date + hour
└── Durability: NATS JetStream (Phase 2) or WAL (Phase 1)

Silver (Clean, Normalized, Deduped)
├── Purpose: Canonical typed tables, deduped via MD5(key columns)
├── Format: Parquet with ZSTD compression level 3
├── Partitioning: symbol/underlying + date + hour
└── Transforms: Dedup by event_id, sort by ts_event_ns

Gold (Derived Analytics)
├── Purpose: Computed features, ML-ready datasets
├── Format: Parquet with ZSTD compression level 3
├── Partitioning: underlying + date + hour
└── Schema: levels.signals.v1 (flattened metrics per PLAN.md §6.4)
```

### Phase 2 Architecture (Current)

```
NATS JetStream (market.* subjects)
    ↓
Lake Service (main.py)
    ├─→ BronzeWriter (NATS → Parquet)
    │   ├── Subscribes: market.stocks.trades, market.stocks.quotes
    │   ├── Subscribes: market.options.trades, market.options.greeks
    │   ├── Subscribes: market.futures.trades, market.futures.mbp10
    │   └── Writes: S3/MinIO or local filesystem
    │
    └─→ GoldWriter (NATS → Parquet)
        ├── Subscribes: levels.signals
        └── Writes: S3/MinIO or local filesystem

Silver Compactor (offline job)
    ├── Reads: Bronze partitions
    ├── Transforms: Dedup + sort
    └── Writes: Silver partitions
```

**Key Change from Phase 1**: NATS JetStream replaces in-process asyncio.Queue + WAL. WAL is deprecated but retained for reference.

---

## Component Specifications

### 1. Lake Service (`main.py`)

**Role**: Orchestrator that initializes BronzeWriter and GoldWriter and manages graceful shutdown.

**Startup Sequence**:
```python
1. Connect to NATS (CONFIG.NATS_URL)
2. Initialize BronzeWriter(bus=bus)
3. Initialize GoldWriter(bus=bus)
4. Start writers (subscribe to NATS subjects)
5. Run until SIGTERM/SIGINT
```

**Shutdown Sequence**:
```python
1. Stop writers (flush remaining buffers)
2. Close NATS connection
```

**Configuration Dependencies**:
- `CONFIG.NATS_URL` (default: `nats://localhost:4222`)
- `CONFIG.DATA_ROOT` (default: `backend/data/lake`)
- `CONFIG.USE_S3` (boolean, default: False)
- `CONFIG.S3_BUCKET`, `CONFIG.S3_ENDPOINT`, `CONFIG.S3_ACCESS_KEY`, `CONFIG.S3_SECRET_KEY`

**Running the Service**:
```bash
# From backend/
uv run python -m src.lake.main
```

---

### 2. Bronze Writer (`bronze_writer.py`)

**Role**: Consumes market data from NATS and writes append-only Parquet files to Bronze tier.

**Subscriptions (NATS subjects)**:
- `market.stocks.trades` → `stocks/trades/symbol=SPY/date=YYYY-MM-DD/hour=HH/`
- `market.stocks.quotes` → `stocks/quotes/symbol=SPY/date=YYYY-MM-DD/hour=HH/`
- `market.options.trades` → `options/trades/underlying=SPY/date=YYYY-MM-DD/hour=HH/`
- `market.options.greeks` → `options/greeks_snapshots/underlying=SPY/date=YYYY-MM-DD/`
- `market.futures.trades` → `futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/`
- `market.futures.mbp10` → `futures/mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/`

**Micro-batching Logic**:
- Buffer up to `buffer_limit` events (default: 1000)
- Flush every `flush_interval_seconds` (default: 5.0s)
- Separate buffers per schema to avoid contention

**Partition Strategy (Hive-style)**:
```python
# Stock/Futures trades/quotes
bronze/stocks/trades/symbol=SPY/date=2025-12-22/hour=14/part-143012_345678.parquet

# Options trades
bronze/options/trades/underlying=SPY/date=2025-12-22/hour=14/part-143012_345678.parquet

# Greeks snapshots (no hour partition, per PLAN.md §2.3)
bronze/options/greeks_snapshots/underlying=SPY/date=2025-12-22/part-143012_345678.parquet
```

**Schema Inference**:
Messages from NATS are inferred based on field presence:
- `bid_px`, `ask_px` → `stocks.quotes`
- `symbol`, `price`, `size` + ES symbol → `futures.trades`
- `levels` → `futures.mbp10`
- `underlying`, `option_symbol` + greeks → `options.greeks_snapshots`
- `underlying`, `option_symbol` → `options.trades`

**Write Format**:
- PyArrow Table → Parquet
- Compression: ZSTD level 3
- Sort by `ts_event_ns` within each file
- Atomic file writes (write temp → rename)

**Storage Backends**:
```python
# Local filesystem (default for M4)
self.use_s3 = False
self.bronze_root = os.path.join(CONFIG.DATA_ROOT, 'bronze')

# S3/MinIO
self.use_s3 = True
self.bronze_root = f"{CONFIG.S3_BUCKET}/bronze"
self.fs = s3fs.S3FileSystem(endpoint_url=CONFIG.S3_ENDPOINT, ...)
```

**Critical Interfaces**:
```python
class BronzeWriter:
    async def start(self, bus: NATSBus) -> None:
        """Subscribe to market.* subjects and start periodic flush."""
        
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Infer schema, buffer event, check flush triggers."""
        
    async def flush_schema(self, schema_name: str) -> None:
        """Flush specific schema buffer to Parquet."""
        
    async def stop(self) -> None:
        """Flush all buffers and cleanup."""
```

---

### 3. Gold Writer (`gold_writer.py`)

**Role**: Consumes level signals from Core Engine and writes derived analytics to Gold tier.

**Subscription (NATS subject)**:
- `levels.signals` → `gold/levels/signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/`

**Input Payload Format** (per PLAN.md §6.4):
```json
{
  "ts": 1715629300123,  // Unix ms
  "spy": {"spot": 545.42, "bid": 545.41, "ask": 545.43},
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
      "barrier": {...},
      "tape": {...},
      "fuel": {...},
      "runway": {...}
    }
  ]
}
```

**Flattening Logic**:
Each level in the payload is flattened into a single Parquet row:
- Market context: `ts_event_ns`, `underlying`, `spot`, `bid`, `ask`
- Level identity: `level_id`, `level_kind`, `level_price`, `direction`, `distance`
- Scores: `break_score_raw`, `break_score_smooth`, `signal`, `confidence`
- Metrics (flattened): `barrier_state`, `barrier_delta_liq`, `tape_imbalance`, `fuel_effect`, etc.

**Write Format**:
- PyArrow Table → Parquet
- Compression: ZSTD level 3
- Sort by `ts_event_ns`, then `level_id`
- Micro-batching: buffer up to 500 records or 10s

**Critical Interfaces**:
```python
class GoldWriter:
    async def start(self, bus: NATSBus) -> None:
        """Subscribe to levels.signals and start periodic flush."""
        
    async def write_level_signals(self, payload: Dict[str, Any]) -> None:
        """Flatten and buffer level signals."""
        
    async def flush(self) -> None:
        """Flush buffer to Parquet, grouped by partition."""
```

---

### 4. Silver Compactor (`silver_compactor.py`)

**Role**: Offline batch job that transforms Bronze → Silver (dedup + sort).

**Deduplication Strategy** (PLAN.md §2.5):
```sql
-- DuckDB query pattern
WITH bronze_data AS (
    SELECT *, md5(source || '|' || ts_event_ns || '|' || symbol || '|' || price || '|' || size || '|' || seq) as event_id
    FROM read_parquet('bronze/.../part-*.parquet', hive_partitioning=true)
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ts_recv_ns) as rn
    FROM bronze_data
)
SELECT * EXCLUDE (rn, event_id)
FROM ranked
WHERE rn = 1
ORDER BY ts_event_ns
```

**Schema Configurations**:
```python
SCHEMA_CONFIG = {
    'stocks.trades': {
        'bronze_path': 'stocks/trades',
        'silver_path': 'stocks/trades',
        'partition_key': 'symbol',
        'dedup_cols': ['source', 'ts_event_ns', 'symbol', 'price', 'size', 'exchange', 'seq'],
    },
    'futures.mbp10': {
        'bronze_path': 'futures/mbp10',
        'silver_path': 'futures/mbp10',
        'partition_key': 'symbol',
        'dedup_cols': ['source', 'ts_event_ns', 'symbol', 'seq'],  # levels are complex nested
    },
    # ... etc per PLAN.md §2.4
}
```

**Usage Patterns**:
```python
from src.lake.silver_compactor import SilverCompactor

compactor = SilverCompactor()

# Compact single schema for single date
result = compactor.compact_date('2025-12-16', schema='futures.trades', partition_value='ES')
# → {'status': 'success', 'rows_read': 12500, 'rows_written': 12480, 'duplicates_removed': 20}

# Compact all schemas for single date
results = compactor.compact_all_schemas('2025-12-16')

# Compact all dates for single schema
results = compactor.compact_all_dates('futures.trades', 'ES')
```

**Output**:
- Silver Parquet files partitioned by date + hour
- Same schema as Bronze (no enrichment in v1; enrichment planned for Phase 1+ per PLAN.md §2.2)
- Deterministic: running compaction twice on same Bronze input produces identical Silver output

---

### 5. WAL Manager (`wal_manager.py`) — **Phase 1 Only, Deprecated**

**Status**: Retained for reference; **not used in Phase 2** (NATS JetStream provides durability).

**Purpose**: Write-Ahead Log for Phase 1 durability before NATS migration.

**Design**:
- One `.arrow` file per stream (e.g., `futures_trades_ES.arrow`)
- Apache Arrow IPC Stream format for sequential writes
- Append before processing, truncate after Parquet flush
- Recovery on startup by replaying unflushed segments

**Migration Note**: If reverting to Phase 1 architecture, WAL can be re-enabled by:
1. Removing NATS bus dependency from BronzeWriter
2. Calling `wal_manager.append()` before buffering events
3. Calling `wal_manager.mark_flushed()` after Parquet write

---

### 6. Legacy Components (`persistence_engine.py`, `historical_cache.py`)

**Status**: Pre-Phase 1 prototypes, retained for backward compatibility.

- `PersistenceEngine`: High-throughput async logger (pre-Bronze/Silver/Gold design)
- `HistoricalDataCache`: Hive-partitioned data lake access layer with Polygon backfill logic

**Do Not Use** for new development. Use BronzeWriter/GoldWriter/SilverCompactor instead.

---

## Data Flow & Contracts

### Write Path (Live Ingestion)

```
Ingestor Service (stream_ingestor.py)
    ↓ publish to NATS
market.stocks.trades, market.stocks.quotes, market.options.trades, ...
    ↓ subscribe
BronzeWriter
    ↓ micro-batch (1000 events or 5s)
Parquet files (Bronze tier)
    ↓ offline job
SilverCompactor
    ↓ dedup + sort
Parquet files (Silver tier)
```

### Derived Analytics Path

```
Core Service (level_signal_service.py)
    ↓ publish to NATS
levels.signals
    ↓ subscribe
GoldWriter
    ↓ micro-batch (500 records or 10s)
Parquet files (Gold tier: levels/signals)
```

### Replay Path (Deterministic)

```
Bronze or Silver Parquet files
    ↓
UnifiedReplayEngine (src/core/unified_replay_engine.py)
    ↓ publish to NATS (or in-memory queue for testing)
Core Service (reprocesses events)
    ↓ publish to NATS
levels.signals
    ↓
GoldWriter (writes replayed results)
```

**Replay Determinism** (PLAN.md §10):
- Given same Bronze input + same config → Gold output is byte-for-byte identical (within floating-point rounding policy)
- Critical for backtesting and strategy validation

---

## Storage Schemas

### Canonical Message Envelope (PLAN.md §2.1)

All events share common metadata:

```python
{
    "schema_name": str,         # e.g., "stocks.trades.v1"
    "schema_version": int,      # e.g., 1
    "source": str,              # "massive_ws", "databento_live", "replay", etc.
    "ts_event_ns": int,         # Unix nanoseconds UTC (event time)
    "ts_recv_ns": int,          # Unix nanoseconds UTC (receive time)
    "key": str,                 # Partition key (symbol or underlying)
    "seq": int,                 # Optional: monotonic sequence per connection
    "payload": {...}            # Schema-specific fields
}
```

### Bronze Schemas (PLAN.md §2.4)

#### stocks.trades.v1
```python
ts_event_ns: int64
ts_recv_ns: int64
source: string
symbol: string
price: float64
size: int32
exchange: int16
seq: int64
```

#### stocks.quotes.v1
```python
ts_event_ns: int64
ts_recv_ns: int64
source: string
symbol: string
bid_px: float64
ask_px: float64
bid_sz: int32
ask_sz: int32
bid_exch: int16
ask_exch: int16
seq: int64
```

#### options.trades.v1
```python
ts_event_ns: int64
ts_recv_ns: int64
source: string
underlying: string
option_symbol: string
exp_date: string
strike: float64
right: string  # "C" or "P"
price: float64
size: int32
opt_bid: float64
opt_ask: float64
aggressor: int8  # +1=BUY, -1=SELL, 0=MID
seq: int64
```

#### options.greeks_snapshots.v1
```python
ts_event_ns: int64
source: string
underlying: string
option_symbol: string
delta: float64
gamma: float64
theta: float64
vega: float64
implied_volatility: float64
open_interest: int32
snapshot_id: string
```

#### futures.trades.v1 (ES time-and-sales)
```python
ts_event_ns: int64
ts_recv_ns: int64
source: string
symbol: string  # "ES" or "ES.c.0"
price: float64
size: int32
aggressor: int8
exchange: string
seq: int64
```

#### futures.mbp10.v1 (ES MBP-10)
```python
ts_event_ns: int64
ts_recv_ns: int64
source: string
symbol: string
is_snapshot: bool
seq: int64
bid_px_1..10: float64
bid_sz_1..10: int32
ask_px_1..10: float64
ask_sz_1..10: int32
```

### Gold Schema: levels.signals.v1 (Flattened)

```python
# Market context
ts_event_ns: int64
underlying: string
spot: float64
bid: float64
ask: float64

# Level identity
level_id: string
level_kind: string
level_price: float64
direction: string
distance: float64

# Scores
break_score_raw: float64
break_score_smooth: float64
signal: string
confidence: string

# Barrier metrics (flattened)
barrier_state: string
barrier_delta_liq: float64
barrier_replenishment_ratio: float64
barrier_added: int32
barrier_canceled: int32
barrier_filled: int32

# Tape metrics (flattened)
tape_imbalance: float64
tape_buy_vol: int64
tape_sell_vol: int64
tape_velocity: float64
tape_sweep_detected: bool
tape_sweep_direction: string
tape_sweep_notional: float64

# Fuel metrics (flattened)
fuel_effect: string
fuel_net_dealer_gamma: float64
fuel_call_wall: float64
fuel_put_wall: float64
fuel_hvl: float64

# Runway (flattened)
runway_direction: string
runway_next_level_id: string
runway_next_level_price: float64
runway_distance: float64
runway_quality: string

# Note
note: string
```

---

## Usage Patterns

### 1. Starting the Lake Service (Phase 2)

```bash
# From backend/
uv run python -m src.lake.main
```

**Expected Output**:
```
============================================================
LAKE SERVICE
============================================================
  Bronze writer: Local storage at /Users/.../backend/data/lake/bronze
  Gold writer: Local storage at /Users/.../backend/data/lake/gold
  Bronze writer: started
  Gold writer: started
============================================================
Lake service running. Press Ctrl+C to stop.
============================================================
```

### 2. Reading Bronze Data

```python
from src.lake.bronze_writer import BronzeReader

reader = BronzeReader()

# Read stock trades for SPY on specific date
df = reader.read_stock_trades(
    symbol='SPY',
    date='2025-12-16',
    start_ns=1734350400000000000,  # 09:30 ET
    end_ns=1734364800000000000     # 13:30 ET
)

# Read all futures MBP-10 for ES
df = reader.read_futures_mbp10(symbol='ES', date='2025-12-16')

# Get available dates
dates = reader.get_available_dates('stocks/trades', 'symbol=SPY')
```

### 3. Compacting Bronze → Silver

```python
from src.lake.silver_compactor import SilverCompactor

compactor = SilverCompactor()

# Compact specific date/schema
result = compactor.compact_date('2025-12-16', schema='futures.mbp10', partition_value='ES')
print(f"Rows: {result['rows_written']}, Duplicates: {result['duplicates_removed']}")

# Compact all schemas for date
results = compactor.compact_all_schemas('2025-12-16')
```

### 4. Reading Silver Data

```python
from src.lake.silver_compactor import SilverReader

reader = SilverReader()

# Read deduped futures trades
df = reader.read_futures_trades(symbol='ES', date='2025-12-16')

# Read enriched option trades (Silver includes greeks join in future)
df = reader.read_option_trades(underlying='SPY', date='2025-12-16')
```

### 5. Reading Gold Data (Level Signals)

```python
from src.lake.gold_writer import GoldReader

reader = GoldReader()

# Read level signals for SPY
df = reader.read_level_signals(
    underlying='SPY',
    date='2025-12-16',
    start_ns=1734350400000000000,
    end_ns=1734364800000000000
)

# Analyze specific level
level_df = df[df['level_id'] == 'STRIKE_545']
print(level_df[['ts_event_ns', 'break_score_smooth', 'signal', 'barrier_state']])
```

---

## Extending the Lake

### Adding a New Bronze Schema

**Example**: Add support for `stocks.darkpool_prints.v1`

**Step 1**: Define schema in `src/common/schemas/`

```python
# src/common/schemas/stocks_darkpool_prints.py
import pyarrow as pa

STOCKS_DARKPOOL_PRINTS_V1 = pa.schema([
    ('ts_event_ns', pa.int64()),
    ('ts_recv_ns', pa.int64()),
    ('source', pa.string()),
    ('symbol', pa.string()),
    ('price', pa.float64()),
    ('size', pa.int64()),
    ('venue', pa.string()),
    ('seq', pa.int64()),
])
```

**Step 2**: Update BronzeWriter schema mappings

```python
# bronze_writer.py
SCHEMA_PATHS = {
    ...
    'stocks.darkpool_prints': 'stocks/darkpool_prints',
}

SUBJECT_TO_SCHEMA = {
    ...
    'market.stocks.darkpool': 'stocks.darkpool_prints',
}
```

**Step 3**: Update schema inference in `_handle_message()`

```python
# bronze_writer.py
if 'venue' in message and 'darkpool' in message.get('source', ''):
    schema_name = 'stocks.darkpool_prints'
    partition_key = message['symbol']
```

**Step 4**: Update SilverCompactor config

```python
# silver_compactor.py
SCHEMA_CONFIG = {
    ...
    'stocks.darkpool_prints': {
        'bronze_path': 'stocks/darkpool_prints',
        'silver_path': 'stocks/darkpool_prints',
        'partition_key': 'symbol',
        'dedup_cols': ['source', 'ts_event_ns', 'symbol', 'price', 'size', 'venue', 'seq'],
    },
}
```

### Adding a New Gold Dataset

**Example**: Add `features.volatility_surface.v1`

**Step 1**: Define Gold schema

```python
# gold schema columns
ts_event_ns: int64
underlying: string
tenor_days: int32
strike_pct_moneyness: float64
implied_vol: float64
vega: float64
```

**Step 2**: Create writer class

```python
# src/lake/volatility_gold_writer.py
class VolatilityGoldWriter:
    async def start(self, bus: NATSBus):
        await self.bus.subscribe('features.volatility_surface', self._handle_surface)
    
    async def _handle_surface(self, payload: Dict[str, Any]):
        # Flatten volatility surface payload
        # Buffer and flush to gold/features/volatility_surface/
```

**Step 3**: Integrate into LakeService

```python
# main.py
self.volatility_writer = VolatilityGoldWriter(bus=self.bus)
await self.volatility_writer.start()
```

---

## Testing & Replay

### Unit Tests (backend/tests/)

- `test_silver_compactor.py` (12 tests): Dedup logic, partition output, schema handling
- `test_wal_manager.py` (12 tests): WAL append, rotation, recovery (Phase 1)
- `test_lake_service.py` (future): Integration test with NATS testcontainer

### Replay Determinism Verification

```python
# tests/test_replay_determinism.py
from src.lake.gold_writer import GoldReader

# Run replay twice
gold_reader = GoldReader()
run1 = gold_reader.read_level_signals(date='2025-12-16')
run2 = gold_reader.read_level_signals(date='2025-12-16')

# Compare (allowing floating-point tolerance)
pd.testing.assert_frame_equal(run1, run2, rtol=1e-9)
```

### Manual Compaction

```bash
# From backend/
uv run python -c "
from src.lake.silver_compactor import SilverCompactor
compactor = SilverCompactor()
results = compactor.compact_all_schemas('2025-12-16')
for schema, result in results.items():
    print(f'{schema}: {result[\"status\"]} ({result.get(\"rows_written\", 0)} rows)')
"
```

---

## Migration Notes (Phase 1 → Phase 2)

### Removed in Phase 2

- **WAL Manager**: Durability now provided by NATS JetStream persistence
- **In-process asyncio.Queue**: Replaced by NATS pub/sub for inter-service communication
- **Monolithic service**: Split into `ingestor`, `core`, `lake`, `gateway` services

### Added in Phase 2

- **NATS subscriptions**: BronzeWriter and GoldWriter subscribe to NATS subjects with durable consumers
- **S3/MinIO support**: Both writers support S3-compatible storage via `s3fs` library
- **Service orchestration**: Lake Service runs as independent process, managed by docker-compose

### Upgrade Path to Iceberg (Future)

The current Hive-partitioned Parquet layout is compatible with Apache Iceberg metadata layer:

```python
# Future Iceberg integration
from pyiceberg.catalog import load_catalog

catalog = load_catalog('local', warehouse='backend/data/lake')
table = catalog.create_table(
    'bronze.stocks.trades',
    schema=STOCKS_TRADES_V1,
    partition_spec=PartitionSpec(
        PartitionField(source_id=2, field_id=1000, name='symbol'),
        PartitionField(source_id=0, field_id=1001, transform=DayTransform(), name='date')
    )
)
```

### Configuration Migration

```python
# Phase 1 (deprecated)
DATA_ROOT = 'backend/data/lake'

# Phase 2 (current)
DATA_ROOT = 'backend/data/lake'  # local filesystem
USE_S3 = False  # or True for S3/MinIO
S3_BUCKET = 'spymaster-lake'
S3_ENDPOINT = 'http://localhost:9000'
S3_ACCESS_KEY = 'minioadmin'
S3_SECRET_KEY = 'minioadmin'
```

---

## Critical Path Dependencies

**Upstream** (Lake consumes from):
- `src/ingestor` (publishes to `market.*` NATS subjects)
- `src/core` (publishes to `levels.signals` NATS subject)
- `src/common/bus.py` (NATSBus wrapper)
- `src/common/config.py` (CONFIG singleton)
- `src/common/event_types.py` (Dataclass definitions)

**Downstream** (Lake produces for):
- `src/core/unified_replay_engine.py` (reads Bronze/Silver for replay)
- ML/analytics pipelines (read Silver/Gold via DuckDB queries)
- Backtesting systems (read Gold `levels.signals` for validation)

**External Dependencies**:
- NATS JetStream (docker-compose service)
- MinIO (optional, docker-compose service)
- PyArrow, pandas, s3fs, duckdb (Python packages)

---

## Key Invariants

1. **Event-time ordering**: All Parquet files sorted by `ts_event_ns` within partition
2. **Idempotency**: Running compaction twice on same Bronze input → identical Silver output (MD5 dedup)
3. **No data loss**: NATS JetStream retention ensures events are not dropped during Lake service restart
4. **Schema stability**: Bronze schemas match PLAN.md §2.4 exactly; changes require version bump
5. **Partition boundaries**: Date/hour partitions aligned to UTC, not market hours (avoids ambiguity)
6. **Compression**: ZSTD level 3 for all tiers (balance speed vs compression ratio)
7. **File sizing**: Target 256MB–1GB per Parquet file (avoids small file problem, enables efficient scans)

---

## Performance Notes

### Micro-batching Tuning

```python
# BronzeWriter (high throughput)
buffer_limit = 1000          # flush at 1000 events
flush_interval = 5.0         # or flush at 5 seconds

# GoldWriter (lower throughput, larger records)
buffer_limit = 500           # flush at 500 level snapshots
flush_interval = 10.0        # or flush at 10 seconds
```

### DuckDB Silver Compaction

- Compaction is CPU-bound (dedup hash computation + sort)
- On Apple M4: ~50–100MB/s Bronze → Silver throughput
- Parallelization: Run `compact_date()` in parallel for different schemas/dates (use `concurrent.futures`)

### S3/MinIO Write Performance

- Local filesystem: ~200–500MB/s (M4 NVMe SSD)
- MinIO (local docker): ~100–200MB/s (network overhead)
- S3 (remote): ~50–100MB/s (depends on region, bandwidth)

**Optimization**: Enable S3 multipart uploads for files >5MB (s3fs handles automatically)

---

## Debugging & Observability

### Log Patterns

```
  Bronze: 1234 rows -> backend/data/lake/bronze/stocks/trades/symbol=SPY/date=2025-12-16/hour=14/part-143012_345678.parquet
  Gold: 56 rows -> backend/data/lake/gold/levels/signals/underlying=SPY/date=2025-12-16/hour=14/part-143012_345678.parquet
  Silver: 12480 rows -> backend/data/lake/silver/futures/trades/symbol=ES/date=2025-12-16/hour=14/part-0000.parquet
  WAL truncated: backend/data/wal/futures_trades_ES.arrow  # Phase 1 only
```

### Common Issues

**Issue**: `Bronze ERROR: Cannot infer schema from message`  
**Fix**: Ensure ingestor publishes normalized events with correct field names (check `event_types.py`)

**Issue**: `Silver compaction failed: Column 'seq' not found`  
**Fix**: Some Bronze data may not have `seq` field; update dedup_cols to exclude optional fields

**Issue**: Gold writer buffer not flushing  
**Fix**: Check `levels.signals` NATS subject for messages; verify Core Service is publishing

**Issue**: S3 writes failing with 403  
**Fix**: Verify `S3_ACCESS_KEY`, `S3_SECRET_KEY`, and bucket permissions in MinIO console

---

## References

- **PLAN.md §1.2**: Phase 0–3 rollout (M4 → servers → colocation)
- **PLAN.md §2.2**: Bronze/Silver/Gold tier definitions
- **PLAN.md §2.3**: Canonical dataset partitioning
- **PLAN.md §2.4**: Storage schemas (minimum required columns)
- **PLAN.md §2.5**: Durability, backpressure, replay correctness
- **PLAN.md §6.4**: WebSocket payload format (Gold `levels.signals` input)
- **PLAN.md §10**: Acceptance criteria (replay determinism)

---

## Revision History

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0     | 2025-12-22 | Initial technical README (Phase 2 complete)  |

---

**End of Lake Module Technical Specification**

