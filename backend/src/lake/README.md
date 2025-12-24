# Lake Module

**Role**: Bronze/Silver/Gold data persistence  
**Audience**: Backend developers working on data storage  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Implements institutional-grade lakehouse architecture with Bronze (raw) → Silver (clean) → Gold (derived) tiers. Consumes from NATS, writes to Parquet (local or S3/MinIO).

**Design principles**:
- **Event-time first**: Every record carries `ts_event_ns` and `ts_recv_ns`
- **Idempotency**: At-least-once inputs → exactly-once storage via deterministic dedup
- **Append-only**: Bronze is immutable (archive, never delete)
- **Reproducible**: Same Bronze input → identical Silver output

---

## Lakehouse Tiers

### Bronze (Raw, Immutable)
Purpose: Near-raw normalized events, full replay capability  
Format: Parquet (ZSTD level 3)  
Partitioning: `{asset_class}/{schema}/symbol={X}/date=YYYY-MM-DD/hour=HH/`  
Durability: NATS JetStream (Phase 2)

### Silver (Clean, Deduped)
Purpose: Canonical typed tables, exactly-once semantics  
Format: Parquet (ZSTD level 3)  
Partitioning: Same as Bronze  
Transforms: Dedup by MD5 event_id, sort by `ts_event_ns`

### Gold (Derived Analytics)
Purpose: Computed features, ML-ready datasets  
Format: Parquet (ZSTD level 3)  
Partitioning: `levels/signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/`  
Schema: Flattened metrics (see `levels.signals.v1`)

---

## Components

### Lake Service (`main.py`)
Orchestrator that initializes Bronze and Gold writers. Manages graceful shutdown with buffer flushing.

### BronzeWriter (`bronze_writer.py`)
Consumes `market.*` from NATS, writes append-only Parquet to Bronze tier.

**Micro-batching**: Buffer 1000 events or 5 seconds → flush to Parquet  
**Subscriptions**: All market data subjects (stocks, options, futures)

### GoldWriter (`gold_writer.py`)
Consumes `levels.signals` from NATS, writes derived analytics to Gold tier.

**Flattening**: Nested physics dicts → flat columns for Parquet efficiency  
**Micro-batching**: Buffer 500 records or 10 seconds

### SilverCompactor (`silver_compactor.py`)
Offline batch job that transforms Bronze → Silver.

**Deduplication**: MD5 hash of key columns → keep first by `ts_recv_ns`  
**DuckDB-powered**: Efficient SQL queries over Parquet

---

## Data Flow

```
NATS (market.*) → BronzeWriter → Bronze Parquet
                                      ↓
                              SilverCompactor → Silver Parquet

NATS (levels.signals) → GoldWriter → Gold Parquet
```

---

## Storage Backends

**Local filesystem** (default):
```bash
DATA_ROOT=backend/data/lake
USE_S3=false
```

**S3/MinIO**:
```bash
DATA_ROOT=s3://spymaster-lake
USE_S3=true
S3_ENDPOINT=http://localhost:9000
```

---

## Running

### Lake Service
```bash
uv run python -m src.lake.main
```

### Silver Compaction (offline)
```bash
cd backend
uv run python -c "
from src.lake.silver_compactor import SilverCompactor
compactor = SilverCompactor()
compactor.compact_all_schemas('2025-12-16')
"
```

---

## File Naming

**Bronze/Silver**:
```
stocks/trades/symbol=SPY/date=2025-12-16/hour=14/part-143012_345678.parquet
futures/mbp10/symbol=ES/date=2025-12-16/hour=14/part-143012_345678.parquet
```

**Gold**:
```
levels/signals/underlying=SPY/date=2025-12-16/hour=14/part-143012_345678.parquet
```

**Hive-style partitioning**: Enables predicate pushdown in query engines (DuckDB, Spark, Pandas).

---

## Performance

**Micro-batching**:
- Bronze: 1000 events or 5s
- Gold: 500 records or 10s

**Silver compaction**:
- ~50-100MB/s (M4 Mac)
- Parallelizable by date/schema

**Storage throughput**:
- Local NVMe: ~200-500MB/s
- MinIO (local): ~100-200MB/s
- S3 (remote): ~50-100MB/s

---

## Schema Mappings

**Bronze → Silver**:
- Dedup by event_id
- Sort by `ts_event_ns`
- No enrichment in v1 (future: join greeks with option trades)

**Bronze/NATS → Gold**:
- Flatten nested `barrier`, `tape`, `fuel`, `runway` dicts
- Example: `barrier.state` → `barrier_state`

**See**: [INTERFACES.md](INTERFACES.md) for complete schemas.

---

## Testing

```bash
cd backend
uv run pytest tests/test_silver_compactor.py -v
uv run pytest tests/test_lake_service.py -v
```

---

## Common Issues

**Bronze ERROR: Cannot infer schema**: Check event field names match schemas  
**Silver compaction failed**: Some fields may be missing/optional → adjust dedup columns  
**Gold buffer not flushing**: Verify Core Service is publishing to `levels.signals`  
**S3 writes failing 403**: Check `S3_ACCESS_KEY`, `S3_SECRET_KEY`, bucket permissions

---

## Critical Invariants

1. **Bronze is append-only**: Never mutate or delete
2. **Silver is derived**: Always regeneratable from Bronze
3. **Event-time ordering**: All files sorted by `ts_event_ns`
4. **Idempotency**: Same Bronze → same Silver (deterministic dedup)
5. **Partition boundaries**: Date/hour aligned to UTC
6. **Compression**: ZSTD level 3 for all tiers
7. **File sizing**: Target 256MB–1GB per Parquet file

---

## Phase Migration

**Phase 1 → Phase 2**:
- WAL removed (NATS JetStream provides durability)
- Added S3/MinIO support
- Microservice deployment via Docker Compose

**Phase 2 → Phase 3** (future):
- Apache Iceberg metadata layer
- ACID guarantees for Silver/Gold tables
- Multi-node object store

---

## References

- **Interface contract**: [INTERFACES.md](INTERFACES.md)
- **Storage schemas**: [../common/schemas/](../common/schemas/)
- **Configuration**: [../common/config.py](../common/config.py)
- **PLAN.md**: §2 (Bronze/Silver/Gold definitions)

---

**Phase**: Phase 2 (NATS + S3/MinIO)  
**Agent assignment**: Phase 2 Agent B  
**Dependencies**: `common` (schemas, config, NATS bus)  
**Consumers**: ML module (reads Silver/Gold), Replay engine
