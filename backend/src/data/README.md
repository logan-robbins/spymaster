# backend/src/data — Runtime Data Storage Layer

**Audience**: AI Coding Agent  
**Last Updated**: 2025-12-22  
**Related**: `PLAN.md` §2.2-§2.5, `backend/src/lake/` (storage code)

---

## Overview

This directory (`backend/src/data/`) serves as a **runtime storage location** for ephemeral and intermediate data structures used by the Spymaster system. It is **distinct** from the canonical data lake at `backend/data/lake/` which holds Bronze/Silver/Gold tiers.

**Purpose**:
- **Write-Ahead Log (WAL)** spillover for durability guarantees during Phase 1/local operation
- **Silver cache** for fast access to recently compacted data
- **Not** for permanent storage (use `backend/data/lake/` for that)
- **Not** for code modules (use `backend/src/lake/` for storage logic)

---

## Directory Structure

```
backend/src/data/
├── lake/
│   └── silver/              # Silver-tier cache (deduped, sorted)
│       ├── futures/
│       │   ├── trades/
│       │   └── mbp10/
│       ├── stocks/
│       │   ├── trades/
│       │   └── quotes/
│       └── options/
│           ├── trades_enriched/
│           └── greeks_snapshots/
└── wal/                     # Write-Ahead Log segments (Phase 1 durability)
    ├── futures_trades_ES.arrow
    ├── futures_mbp10_ES.arrow
    ├── options_trades_SPY.arrow
    └── stocks_trades_SPY.arrow
```

### Subdirectories

#### `lake/silver/`
- **Purpose**: Runtime cache for Silver-tier data (deduped, sorted, clean)
- **Populated by**: `backend/src/lake/silver_compactor.py`
- **Format**: Hive-style partitioned Parquet with ZSTD compression
- **Partitioning**: `symbol=X/date=YYYY-MM-DD/hour=HH/part-*.parquet`
- **Lifecycle**: Can be safely deleted and regenerated from Bronze at `backend/data/lake/bronze/`

#### `wal/`
- **Purpose**: Write-Ahead Log for durability guarantees (Phase 1 only)
- **Managed by**: `backend/src/lake/wal_manager.py`
- **Format**: Apache Arrow IPC Stream (`.arrow` files)
- **Lifecycle**:
  - **Phase 1 (local)**: Active WAL segments ensure zero data loss between ingestion and Parquet flush
  - **Phase 2+ (NATS)**: WAL is replaced by NATS JetStream persistence; this directory becomes obsolete
- **Recovery**: On startup, `WALManager` scans for unflushed segments and replays them into Bronze

---

## Storage Architecture Context

The Spymaster system uses a **lakehouse architecture** with three tiers:

### Bronze → Silver → Gold Pipeline

| Tier     | Location                       | Purpose                                      | Format    | Managed By                        |
|----------|--------------------------------|----------------------------------------------|-----------|-----------------------------------|
| **Bronze** | `backend/data/lake/bronze/`    | Raw, append-only, replayable captures        | Parquet   | `bronze_writer.py`                |
| **Silver** | `backend/data/lake/silver/`    | Clean, deduped, sorted, join-enriched        | Parquet   | `silver_compactor.py`             |
| **Gold**   | `backend/data/lake/gold/`      | Derived analytics, features, signals         | Parquet   | `gold_writer.py`                  |
| **WAL**    | `backend/src/data/wal/`        | Ephemeral durability buffer (Phase 1)        | Arrow IPC | `wal_manager.py`                  |
| **Silver Cache** | `backend/src/data/lake/silver/` | Runtime cache (symlink or copy)          | Parquet   | `silver_compactor.py` (optional) |

### Why Two Silver Locations?

1. **Canonical Silver**: `backend/data/lake/silver/` is the source of truth
2. **Runtime Cache**: `backend/src/data/lake/silver/` is a fast-access cache for hot queries during live operation
   - Populated on-demand by `SilverReader` for recent time windows
   - Can be a symlink, copy, or memory-mapped view
   - Safe to delete; will regenerate from canonical Silver

---

## Data Flow Diagram

```
┌─────────────────┐
│  Vendor Feeds   │
│ (Massive, DBN)  │
└────────┬────────┘
         │ normalize
         ▼
    ┌────────┐
    │  WAL   │  ← backend/src/data/wal/  (Phase 1 only)
    └────┬───┘
         │ flush every 1-5s
         ▼
    ┌─────────┐
    │ Bronze  │  ← backend/data/lake/bronze/
    └────┬────┘
         │ compact (dedup + sort)
         ▼
    ┌─────────┐
    │ Silver  │  ← backend/data/lake/silver/  (canonical)
    └────┬────┘     └── backend/src/data/lake/silver/  (cache)
         │ enrich + aggregate
         ▼
    ┌─────────┐
    │  Gold   │  ← backend/data/lake/gold/
    └─────────┘
```

**Phase 2+ (NATS)**: WAL is removed; NATS JetStream acts as the durable buffer before Bronze.

---

## File Formats & Conventions

### Write-Ahead Log (`.arrow`)

- **Format**: Apache Arrow IPC Stream
- **Schema**: Matches Bronze schema exactly (see `backend/src/common/schemas/`)
- **Naming**: `{schema}_{symbol}.arrow` (active segment)
  - Example: `futures_trades_ES.arrow`, `options_trades_SPY.arrow`
- **Rotation**: Segments rotate to `.001.arrow`, `.002.arrow`, etc. after Parquet flush
- **Size limit**: 256MB per segment (configurable)

### Silver Parquet

- **Format**: Apache Parquet (columnar)
- **Compression**: ZSTD level 3-6 (default)
- **Partitioning**: Hive-style
  ```
  symbol=SPY/date=2025-12-22/hour=14/part-20251222140530.parquet
  ```
- **Sorting**: Sorted by `ts_event_ns` within each file
- **Deduplication**: `event_id = MD5(source, ts_event_ns, symbol, price, size, seq)`
  - Guarantees exactly-once semantics from at-least-once inputs

---

## Schema Mappings

### Bronze → Silver Transformations

| Bronze Schema                  | Silver Schema                   | Transformations                              |
|--------------------------------|---------------------------------|----------------------------------------------|
| `stocks.trades.v1`             | `stocks.trades.v1`              | Dedup + sort                                 |
| `stocks.quotes.v1`             | `stocks.quotes.v1`              | Dedup + sort                                 |
| `futures.trades.v1`            | `futures.trades.v1`             | Dedup + sort                                 |
| `futures.mbp10.v1`             | `futures.mbp10.v1`              | Dedup + sort                                 |
| `options.trades.v1`            | `options.trades_enriched.v1`    | Dedup + sort + as-of join with greeks        |
| `options.greeks_snapshots.v1`  | `options.greeks_snapshots.v1`   | Dedup + sort                                 |

**As-of joins** (Silver enrichment, optional in Phase 1):
- Attach best-known `greeks_snapshot_id` to option trades within tolerance (e.g., 10s)
- Populate `delta`, `gamma`, `delta_notional`, `gamma_notional` columns

---

## Durability & Recovery Guarantees

### Phase 1 (Local, WAL-based)

1. **Ingestion** → Event written to WAL (`.arrow` append)
2. **Processing** → Engines consume events from in-memory queue
3. **Micro-batch** → Every 1-5s, accumulated events flushed to Bronze Parquet
4. **WAL rotation** → After successful Parquet write, WAL segment is rotated/truncated

**On crash/restart**:
- `WALManager.recover_all()` scans for unflushed segments
- Replays events into Bronze in event-time order
- Guarantees zero data loss

### Phase 2+ (NATS JetStream)

- WAL is **obsolete**; NATS JetStream acts as the durable buffer
- `backend/src/data/wal/` can be safely ignored or deleted
- Consumers replay from NATS subjects with offset tracking

---

## Configuration

### Environment Variables

```bash
# Phase 1 (local)
DATA_ROOT=backend/data/lake/           # Canonical Bronze/Silver/Gold
WAL_ROOT=backend/src/data/wal/         # WAL segments

# Phase 2+ (NATS + S3)
USE_S3=true                            # Enable S3/MinIO storage
S3_ENDPOINT=http://localhost:9000      # MinIO endpoint
S3_BUCKET=spymaster-lake               # S3 bucket name
DATA_ROOT=s3://spymaster-lake/         # S3 prefix for Bronze/Silver/Gold
```

### Python Config

```python
# backend/src/common/config.py
CONFIG = {
    "DATA_ROOT": "backend/data/lake/",
    "WAL_ROOT": "backend/src/data/wal/",
    "FLUSH_INTERVAL_SEC": 5.0,
    "WAL_SEGMENT_SIZE_MB": 256,
    "USE_S3": False,
    "S3_BUCKET": "spymaster-lake",
}
```

---

## Usage Patterns for AI Agents

### Reading Silver Data

```python
from src.lake.silver_compactor import SilverReader

reader = SilverReader()

# Read specific date + schema
df = reader.read_date('2025-12-22', schema='futures.trades')

# Read date range
df = reader.read_date_range(
    start_date='2025-12-20',
    end_date='2025-12-22',
    schema='futures.mbp10',
    symbol='ES'
)

# Filter by time window
df = df[(df['ts_event_ns'] >= start_ns) & (df['ts_event_ns'] < end_ns)]
```

### Writing to Bronze (Phase 1)

```python
from src.lake.bronze_writer import BronzeWriter

writer = BronzeWriter()

# Append events to Bronze
await writer.write_stock_trade(trade)
await writer.write_stock_quote(quote)
await writer.write_option_trade(opt_trade)

# Manual flush (normally automatic every 5s)
await writer.flush_all()
```

### WAL Recovery (Phase 1)

```python
from src.lake.wal_manager import WALManager

wal = WALManager()

# On startup: recover unflushed events
recovered = wal.recover_all()
print(f"Recovered {len(recovered)} unflushed events")

# Process recovered events
for event in recovered:
    await process_event(event)
```

### Silver Compaction

```python
from src.lake.silver_compactor import SilverCompactor

compactor = SilverCompactor()

# Compact specific date + schema
compactor.compact_date('2025-12-22', schema='futures.trades')

# Compact all schemas for a date
compactor.compact_all_schemas('2025-12-22')

# Compact all dates (backfill)
compactor.compact_all_dates(schema='futures.mbp10')
```

---

## Partitioning & Performance

### Partition Keys

| Schema                  | Partition Key    | Partition Example                          |
|-------------------------|------------------|--------------------------------------------|
| `stocks.trades`         | `symbol`         | `symbol=SPY/date=2025-12-22/hour=14/`      |
| `stocks.quotes`         | `symbol`         | `symbol=SPY/date=2025-12-22/hour=14/`      |
| `futures.trades`        | `symbol`         | `symbol=ES/date=2025-12-22/hour=14/`       |
| `futures.mbp10`         | `symbol`         | `symbol=ES/date=2025-12-22/hour=14/`       |
| `options.trades`        | `underlying`     | `underlying=SPY/date=2025-12-22/hour=14/`  |
| `options.greeks`        | `underlying`     | `underlying=SPY/date=2025-12-22/`          |

### File Sizing

- **Target**: 256MB-1GB per Parquet file
- **Row groups**: 64-256MB
- **Avoid**: Ultra-small files (<10MB) or ultra-large (>2GB) for optimal scan performance

### Query Optimization

- **Predicate pushdown**: Filter on partition keys (`symbol`, `date`, `hour`) for fast pruning
- **Column projection**: Read only required columns (Parquet is columnar)
- **Time-range queries**: Use `ts_event_ns` for precise event-time filtering
- **DuckDB**: Use for fast SQL queries over Parquet (used by `SilverCompactor`)

---

## Maintenance & Operations

### Disk Space Management

```bash
# Check Bronze size
du -sh backend/data/lake/bronze/

# Check Silver size
du -sh backend/data/lake/silver/

# Check WAL size (should be small, <1GB)
du -sh backend/src/data/wal/

# Clear runtime cache (safe)
rm -rf backend/src/data/lake/silver/*
```

### Cleanup Old Data

```bash
# Archive data older than 30 days (example)
find backend/data/lake/bronze/ -type d -name "date=*" -mtime +30 -exec mv {} archive/ \;

# Regenerate Silver from archived Bronze (if needed)
uv run python -c "
from src.lake.silver_compactor import SilverCompactor
c = SilverCompactor()
c.compact_all_dates(schema='futures.trades', start_date='2025-11-01', end_date='2025-11-30')
"
```

### Verify Data Integrity

```bash
# Run integrity checks (uses DuckDB)
uv run pytest backend/tests/test_silver_compactor.py -v

# Manual verification: count rows per tier
uv run python -c "
import duckdb
con = duckdb.connect()

# Bronze row count
bronze_count = con.execute(\"\"\"
    SELECT COUNT(*) FROM read_parquet('backend/data/lake/bronze/futures/trades/**/*.parquet')
\"\"\").fetchone()[0]

# Silver row count (should be <= Bronze after dedup)
silver_count = con.execute(\"\"\"
    SELECT COUNT(*) FROM read_parquet('backend/data/lake/silver/futures/trades/**/*.parquet')
\"\"\").fetchone()[0]

print(f'Bronze: {bronze_count:,} | Silver: {silver_count:,} | Dedup ratio: {bronze_count/silver_count:.2f}x')
"
```

---

## Troubleshooting

### Issue: WAL files accumulating

**Symptoms**: `backend/src/data/wal/*.arrow` files growing beyond 1GB  
**Cause**: Bronze writer not flushing or crashed before rotation  
**Fix**:
```python
# Force recovery + flush
from src.lake.wal_manager import WALManager
from src.lake.bronze_writer import BronzeWriter

wal = WALManager()
recovered = wal.recover_all()  # Drains WAL into Bronze

writer = BronzeWriter()
await writer.flush_all()       # Forces Parquet write
```

### Issue: Silver data missing

**Symptoms**: `SilverReader.read_date()` returns empty DataFrame  
**Cause**: Silver compaction not run for that date  
**Fix**:
```python
from src.lake.silver_compactor import SilverCompactor
c = SilverCompactor()
c.compact_date('2025-12-22', schema='futures.trades')  # Regenerate
```

### Issue: Out-of-order events in Bronze

**Symptoms**: `ts_event_ns` not monotonic within Bronze Parquet files  
**Expected**: Bronze is NOT guaranteed to be sorted (append-only)  
**Fix**: Use Silver tier (sorted by `ts_event_ns`)

### Issue: Duplicate events in Bronze

**Symptoms**: Same `event_id` appears multiple times  
**Expected**: Bronze allows duplicates (at-least-once semantics from vendor feeds)  
**Fix**: Use Silver tier (deduplicated)

---

## Testing

### Unit Tests

```bash
# WAL Manager
uv run pytest backend/tests/test_wal_manager.py -v

# Silver Compactor
uv run pytest backend/tests/test_silver_compactor.py -v

# Bronze Writer (Phase 2: NATS integration)
uv run pytest backend/tests/test_lake_service.py -v
```

### Integration Tests

```bash
# End-to-end replay correctness
uv run pytest backend/tests/test_replay_determinism.py -v

# Verify Bronze → Silver → Gold pipeline
uv run pytest backend/tests/test_e2e_replay.py -v
```

---

## Phase Migration Notes

### Phase 1 → Phase 2 (NATS + S3)

**Changes**:
1. WAL is **removed**; NATS JetStream becomes the durable buffer
2. Bronze writer subscribes to NATS subjects instead of in-memory queue
3. Storage backend can be S3/MinIO instead of local filesystem
4. `backend/src/data/wal/` becomes obsolete

**Migration steps**:
1. Ensure all WAL segments are flushed: `wal.recover_all()` + `bronze_writer.flush_all()`
2. Archive Bronze/Silver/Gold from `backend/data/lake/` to S3
3. Update `CONFIG.DATA_ROOT` to S3 prefix
4. Deploy NATS cluster + MinIO
5. Restart services with `docker-compose.yml`

### Phase 2 → Phase 3 (Colocation)

**Changes**:
1. NATS → NATS clustered or Redpanda/Kafka
2. S3 → high-performance object store with Iceberg metadata
3. Silver/Gold become Apache Iceberg tables for ACID + schema evolution

**No changes required** to Bronze/Silver partitioning scheme or file formats.

---

## Related Documentation

- **Architecture**: `PLAN.md` §1-§2 (System architecture, message envelope, dataset contracts)
- **Storage Logic**: `backend/src/lake/*.py` (Bronze/Silver/Gold writers, WAL manager)
- **Schemas**: `backend/src/common/schemas/` (Pydantic + Arrow schema definitions)
- **Testing**: `backend/tests/test_lake_*.py`, `backend/tests/test_silver_compactor.py`
- **Configuration**: `backend/src/common/config.py`

---

## Key Invariants (DO NOT VIOLATE)

1. **Bronze is append-only**: Never mutate or delete Bronze files (archive/move only)
2. **Silver is derived**: Always regeneratable from Bronze via deterministic dedup + sort
3. **WAL is ephemeral**: Safe to delete after flush (Phase 1 only)
4. **Event-time ordering**: Use `ts_event_ns` for all time-based queries, NOT `ts_recv_ns`
5. **Partition keys are immutable**: Changing partition scheme requires full backfill
6. **Schema versioning**: Use `schema_version` field; never break backward compatibility
7. **Dedup key stability**: MD5 hash of `(source, ts_event_ns, symbol, price, size, seq)` must remain consistent

---

## Performance Benchmarks (M4 MacBook, 128GB RAM)

| Operation                          | Throughput       | Latency (p99)  |
|------------------------------------|------------------|----------------|
| WAL append (Arrow IPC)             | 500K events/sec  | <1ms           |
| Bronze micro-batch flush           | 100K events/sec  | 5-50ms         |
| Silver compaction (1 day, 1 schema)| 50M events/min   | N/A (batch)    |
| SilverReader query (1 hour)        | <100ms           | <100ms         |
| DuckDB scan (1GB Parquet)          | ~2 GB/sec        | <500ms         |

**Notes**:
- Benchmarks assume NVMe SSD storage
- ZSTD compression: ~5-10x reduction (event-dependent)
- MBP-10 data is the largest schema (~50GB/day for ES)

---

## Contact & Contributions

This module is part of the **Spymaster Physics Engine** (SPY break/reject prediction system).

**Authors**: Phase 1 Agent J (Silver layer), Phase 2 Agent B (Lake service)  
**Maintainer**: Backend team  
**Last Major Refactor**: Phase 2 microservices migration (Dec 2025)

For questions or issues, consult:
1. `PLAN.md` (canonical system design)
2. `backend/tests/` (usage examples via tests)
3. Inline docstrings in `backend/src/lake/*.py`

