# I/O Module

**Purpose**: Data lake readers and writers (Bronze/Silver/Gold)  
**Status**: Production  
**Architecture**: Medallion architecture persistence layer

---

## Overview

Implements storage components for the Medallion Architecture (Bronze → Silver → Gold).

**This module provides**:
- Streaming writers (Bronze, Gold) for NATS → Parquet
- Batch builders (Silver) for feature engineering
- Curator (Gold) for production promotion

---

## Components

### BronzeWriter / BronzeReader (`bronze.py`)

**BronzeWriter**: Consumes `market.*` from NATS, writes append-only Parquet to Bronze tier.

```python
from src.io.bronze import BronzeWriter, BronzeReader

# Streaming write (with NATS)
writer = BronzeWriter(bus=nats_bus)
await writer.start()

# Batch read
reader = BronzeReader()
trades_df = reader.read_futures_trades(
    symbol='ES',
    date='2024-12-16',
    start_ns=start_ns,
    end_ns=end_ns
)
```

**Characteristics**:
- Micro-batching: 1000 events or 5 seconds → flush
- At-least-once semantics (duplicates possible)
- Append-only (never overwrites)
- All trading hours preserved

**Output**: `bronze/futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet`

---

### SilverFeatureBuilder (`silver.py`)

Transforms Bronze data into versioned Silver feature sets via pipeline execution.

```python
from src.io.silver import SilverFeatureBuilder

builder = SilverFeatureBuilder()

# Build features for date range
stats = builder.build_feature_set(
    manifest=manifest,
    dates=['2024-12-16', '2024-12-17'],
    force=False
)

# Load features for training
df = builder.load_features('v3.1.0')

# List available versions
versions = builder.list_versions()
```

**Implementation**:
- Uses versioned pipelines (18 stages)
- Applies RTH filtering (09:30-12:30 ET)
- Writes to `silver/features/{version}/date=YYYY-MM-DD/*.parquet`
- Creates `manifest.yaml` and `validation.json` per version

---

### GoldWriter / GoldCurator (`gold.py`)

**GoldWriter**: Consumes `levels.signals` from NATS, writes real-time signals.

```python
from src.io.gold import GoldWriter

writer = GoldWriter(bus=nats_bus)
await writer.start()
```

**GoldCurator**: Promotes best Silver experiments to Gold production datasets.

```python
from src.io.gold import GoldCurator

curator = GoldCurator()

# Promote Silver → Gold
result = curator.promote_to_training(
    silver_version='v3.1.0',
    dataset_name='signals_production',
    notes='Production dataset',
    force=True
)

# Validate Gold dataset
validation = curator.validate_dataset('signals_production')

# List Gold datasets
datasets = curator.list_datasets()
```

**Implementation**:
- Reads all Silver parquet for given version
- Concatenates and validates
- Writes to `gold/training/{dataset_name}.parquet`
- Creates metadata JSON

---

### WALManager (`wal.py`)

Write-Ahead Log for crash recovery and replay.

```python
from src.io.wal import WALManager

wal = WALManager()
wal.append(event)
wal.flush()
```

---

### IOService (`service.py`)

Orchestrator for streaming Bronze and Gold writers.

```bash
cd backend
uv run python -m src.io.service
```

**Responsibilities**:
- Initialize BronzeWriter (NATS → Bronze)
- Initialize GoldWriter (NATS → Gold streaming)
- Handle graceful shutdown with buffer flushing

---

## Storage Backends

**Local filesystem** (default):
```bash
DATA_ROOT=backend/data
```

**S3/MinIO**:
```bash
DATA_ROOT=s3://spymaster-lake
USE_S3=true
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
```

---

## File Naming Conventions

**Bronze/Gold Streaming**:
```
futures/trades/symbol=ES/date=2024-12-16/hour=14/part-143012.parquet
options/trades/underlying=ES/date=2024-12-16/hour=14/part-143012.parquet
```

**Silver**:
```
features/v3.1.0/date=2024-12-16/signals.parquet
```

**Gold Training**:
```
training/signals_production.parquet
training/signals_production_metadata.json
```

**Hive-style partitioning** enables predicate pushdown in query engines (DuckDB, Spark, Pandas).

---

## Performance

**Micro-batching**:
- Bronze: 1000 events or 5s
- Gold streaming: 500 records or 10s

**Silver feature builder**:
- Single date: ~2-5 seconds (18 stages)
- 10 dates: ~30-60 seconds (M4 Mac)

**Storage throughput**:
- Local NVMe: ~200-500MB/s
- MinIO (local): ~100-200MB/s
- S3 (remote): ~50-100MB/s

---

## Testing

```bash
cd backend
uv run pytest tests/test_io_*.py -v
```

---

## References

- **Architecture**: `backend/DATA_ARCHITECTURE.md`
- **Schemas**: `backend/src/common/schemas/`
- **Pipeline**: `backend/src/pipeline/`
- **Config**: `backend/src/common/config.py`
