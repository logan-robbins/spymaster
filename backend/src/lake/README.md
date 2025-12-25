# Lake Module

**Role**: Bronze/Silver/Gold data persistence components  
**Audience**: Backend developers implementing data storage  
**Architecture**: See [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)

---

## Purpose

Implements the storage components for the Medallion Architecture (Bronze → Silver → Gold).

**This module provides**:
- Streaming writers (Bronze, Gold) for NATS → Parquet
- Batch builders (Silver) for feature versioning
- Curator (Gold) for production promotion

---

## Components

### Lake Service (`main.py`)

Orchestrator for streaming Bronze and Gold writers. Manages graceful shutdown with buffer flushing.

**Responsibilities**:
- Initialize BronzeWriter (NATS → Bronze Parquet)
- Initialize GoldWriter (NATS levels.signals → Gold streaming Parquet)
- Handle shutdown signals

**Entry Point**: `uv run python -m src.lake.main`

### BronzeWriter (`bronze_writer.py`)

Consumes `market.*` from NATS, writes append-only Parquet to Bronze tier.

**Micro-batching**: Buffer 1000 events or 5 seconds → flush to Parquet  
**Subscriptions**: `market.stocks.trades`, `market.options.trades`, `market.futures.trades`, `market.futures.mbp10`

**Key Characteristics**:
- At-least-once semantics (duplicates possible)
- Append-only (never overwrites)
- All trading hours preserved

### GoldWriter (`gold_writer.py`)

Consumes `levels.signals` from NATS, writes real-time signals to Gold streaming tier.

**Flattening**: Nested physics dicts → flat columns for Parquet efficiency  
**Micro-batching**: Buffer 500 records or 10 seconds  
**Output**: `gold/streaming/signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/*.parquet`

### SilverFeatureBuilder (`silver_feature_builder.py`)

**Purpose**: Transform Bronze data into versioned Silver feature sets for ML experimentation.

**Key Methods**:
```python
# Build new feature version
builder = SilverFeatureBuilder()
stats = builder.build_feature_set(
    manifest=manifest,  # Defines features + parameters
    dates=['2025-12-16', '2025-12-17'],
    force=False
)

# List available versions
versions = builder.list_versions()

# Load features for training
df = builder.load_features('v2.0_full_ensemble')

# Register experiment results
builder.register_experiment(
    version='v2.0_full_ensemble',
    exp_id='exp002',
    metrics={'auc': 0.72, 'precision': 0.68},
    notes='Added TA features'
)
```

**Implementation**:
- Uses `VectorizedPipeline` internally for feature computation
- Applies RTH filtering (09:30-16:00 ET)
- Writes to `silver/features/{version}/date=YYYY-MM-DD/*.parquet`
- Creates `manifest.yaml` and `validation.json` per version

### GoldCurator (`gold_curator.py`)

**Purpose**: Promote best Silver experiments to Gold production.

**Key Methods**:
```python
curator = GoldCurator()

# Promote Silver → Gold
result = curator.promote_to_training(
    silver_version='v2.0_full_ensemble',
    dataset_name='signals_production',
    notes='New production model',
    force=True
)

# Validate Gold dataset
validation = curator.validate_dataset('signals_production')

# List Gold datasets
datasets = curator.list_datasets()
```

**Implementation**:
- Reads all Silver parquet files for given version
- Concatenates and validates
- Writes to `gold/training/{dataset_name}.parquet`
- Creates metadata JSON

---

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
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
```

---

## Running Components

### Stream Bronze + Gold (Real-time)

```bash
cd backend
uv run python -m src.lake.main
```

### Build Silver Features (Batch)

See workflow in [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md#feature-engineering-workflow)

---

## File Naming

**Bronze/Gold Streaming**:
```
futures/trades/symbol=ES/date=2025-12-16/hour=14/part-143012_345678.parquet
options/trades/underlying=SPY/date=2025-12-16/hour=14/part-143012_345678.parquet
```

**Silver**:
```
features/v2.0_full_ensemble/date=2025-12-16/signals.parquet
```

**Gold Training**:
```
training/signals_production.parquet
training/signals_production_metadata.json
```

**Hive-style partitioning**: Enables predicate pushdown in query engines (DuckDB, Spark, Pandas).

---

## Performance

**Micro-batching**:
- Bronze: 1000 events or 5s
- Gold streaming: 500 records or 10s

**Silver feature builder**:
- Single date: ~2-5 seconds (VectorizedPipeline)
- 10 dates: ~30-60 seconds (M4 Mac)

**Storage throughput**:
- Local NVMe: ~200-500MB/s
- MinIO (local): ~100-200MB/s
- S3 (remote): ~50-100MB/s

---

## Testing

```bash
cd backend
uv run pytest tests/test_lake_service.py -v
```

---

## References

- **Architecture & Workflow**: [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)
- **Interface contract**: [INTERFACES.md](INTERFACES.md)
- **Storage schemas**: [../common/schemas/](../common/schemas/)
- **Configuration**: [../common/config.py](../common/config.py)
