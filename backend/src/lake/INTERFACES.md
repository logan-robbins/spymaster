# Lake Module Interfaces

**Module**: `backend/src/lake/`  
**Role**: Bronze/Silver/Gold lakehouse persistence  
**Audience**: AI Coding Agents

---

## Module Purpose

- Stream Bronze + Gold data from NATS into partitioned Parquet.
- Build Silver feature sets from Bronze using the ES pipeline (batch).
- Curate Gold training datasets from Silver.

---

## Input Interfaces

### NATS Subscriptions

**Bronze Writer** (streaming ingest):
- `market.futures.trades` -> Bronze `futures/trades/`
- `market.futures.mbp10` -> Bronze `futures/mbp10/`
- `market.options.trades` -> Bronze `options/trades/`

**Gold Writer** (streaming signals):
- `levels.signals` -> Gold `levels/signals/`

### Batch Inputs

**SilverFeatureBuilder**:
- Reads Bronze `futures/trades`, `futures/mbp10`, `options/trades`.
- Uses `es_pipeline` stages for feature generation.

**GoldCurator**:
- Reads Silver features from `silver/features/{version}/`.

---

## Output Interfaces

### Parquet File Structure

**Bronze Tier** (append-only, immutable):
```
bronze/{schema_path}/
  {partition_key}={value}/
    date=YYYY-MM-DD/
      hour=HH/
        part-{timestamp}.parquet
```

**Notes**:
- Partition key is `symbol=` for futures and `underlying=` for options.

**Silver Tier** (batch features):
```
silver/features/{version}/
  date=YYYY-MM-DD/
    features_{date}.parquet
```

**Canonical ES pipeline output**:
```
silver/features/es_pipeline/
  date=YYYY-MM-DD/
    *.parquet
```

**Gold Tier**:
- Streaming signals:
```
gold/levels/signals/
  underlying=ES/
    date=YYYY-MM-DD/
      hour=HH/
        part-{timestamp}.parquet
```
- Curated training datasets:
```
gold/training/
  {dataset_name}.parquet
  {dataset_name}_metadata.json
gold/catalog.json
```

---

## Storage Schemas

**Bronze** (PyArrow):
- `FuturesTradeV1`, `MBP10V1`, `OptionTradeV1`

**Silver** (PyArrow):
- `SilverFeaturesESPipelineV1` (182 columns)
- Reference: `backend/SILVER_SCHEMA.md`

**Gold**:
- Training: `GoldTrainingESPipelineV1` (same schema as Silver)
- Streaming signals: flattened from `levels.signals` payload (see GoldWriter)

---

## Bronze Writer Interface

### Class: BronzeWriter

```python
class BronzeWriter:
    def __init__(
        self,
        bus=None,
        data_root: Optional[str] = None,
        buffer_limit: int = 1000,
        flush_interval_seconds: float = 5.0,
        use_s3: Optional[bool] = None
    )

    async def start(self, bus=None) -> None
    async def stop(self) -> None
    async def flush_all(self) -> None
    async def flush_schema(self, schema_name: str) -> None
    def get_bronze_path(self) -> str
```

**Behavior**:
- Buffers per schema, flushes on size or time.
- Sorts by `ts_event_ns` before writing.
- Writes ZSTD-compressed Parquet.
- Supports local or S3/MinIO.

---

## Bronze Reader Interface

### Class: BronzeReader

```python
class BronzeReader:
    def __init__(self, data_root: Optional[str] = None)

    def read_futures_trades(
        self,
        symbol: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        front_month_only: bool = True,
        specific_contract: Optional[str] = None
    ) -> pd.DataFrame

    def read_futures_mbp10(
        self,
        symbol: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        front_month_only: bool = True,
        specific_contract: Optional[str] = None
    ) -> pd.DataFrame

    def read_option_trades(
        self,
        underlying: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def get_available_dates(
        self,
        schema_path: str,
        partition_key: str
    ) -> List[str]

    def get_latest_date(
        self,
        schema_path: str,
        partition_key: str
    ) -> Optional[str]
```

**Behavior**:
- Uses DuckDB `read_parquet` with hive partitioning.
- Time filtering on `ts_event_ns` when provided.
- Front-month filter uses `ContractSelector` for futures data.

---

## Silver Feature Builder Interface

### Class: SilverFeatureBuilder

```python
class SilverFeatureBuilder:
    def __init__(self, data_root: Optional[str] = None)

    def build_feature_set(
        self,
        manifest: FeatureManifest,
        dates: List[str],
        force: bool = False
    ) -> Dict[str, Any]

    def list_versions(self) -> List[str]
    def get_manifest(self, version: str) -> Optional[FeatureManifest]
    def load_features(self, version: str, dates: Optional[List[str]] = None) -> pd.DataFrame
    def register_experiment(...)
    def compare_versions(...)
```

**Behavior**:
- Runs the ES pipeline for each date, filters to schema columns, and writes versioned outputs.
- Writes `manifest.yaml` and `validation.json` under each version directory.
- Maintains `silver/features/experiments.json` registry.

---

## Gold Writer Interface

### Class: GoldWriter

```python
class GoldWriter:
    def __init__(
        self,
        bus=None,
        data_root: Optional[str] = None,
        buffer_limit: int = 500,
        flush_interval_seconds: float = 10.0,
        use_s3: Optional[bool] = None
    )

    async def start(self, bus=None) -> None
    async def stop(self) -> None
    async def write_level_signals(self, payload: Dict[str, Any]) -> None
    async def flush(self) -> None
    def get_gold_path(self) -> str
```

**Payload shape** (from `levels.signals`):
```json
{
  "ts": 1734470400123,
  "es": {"spot": 6845.25, "bid": 6845.0, "ask": 6845.5},
  "levels": [ ... ]
}
```

**Notes**:
- One Parquet row per level (flattened).

**Flattening**:
- Fields: `ts_event_ns`, `underlying`, `spot`, `bid`, `ask`, `level_*`,
  `break_score_*`, `signal`, `confidence`, `barrier_*`, `tape_*`, `fuel_*`, `runway_*`, `note`.

---

## Gold Reader Interface

### Class: GoldReader

```python
class GoldReader:
    def __init__(self, data_root: Optional[str] = None)

    def read_level_signals(
        self,
        underlying: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def get_available_dates(self, underlying: str = 'ES') -> List[str]
```

---

## Gold Curator Interface

### Class: GoldCurator

```python
class GoldCurator:
    def __init__(self, data_root: Optional[str] = None)

    def promote_to_training(
        self,
        silver_version: str,
        dataset_name: str = 'signals_production',
        dates: Optional[List[str]] = None,
        notes: str = "",
        force: bool = False
    ) -> Dict[str, Any]

    def load_training_data(self, dataset_name: str = 'signals_production') -> pd.DataFrame
    def list_training_datasets(self) -> List[Dict[str, Any]]
    def get_dataset_metadata(self, dataset_name: str) -> Optional[Dict[str, Any]]
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]
```

**Helper**:
- `promote_best_experiment(exp_id, dataset_name='signals_production')`

---

## Storage Backend Configuration

Local filesystem (default):
```
DATA_ROOT=backend/data/lake
USE_S3=false
```

S3/MinIO:
```
DATA_ROOT=s3://spymaster-lake
USE_S3=true
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_BUCKET=spymaster-lake
```

---

## Entry Points

- Lake service (Bronze + Gold streaming): `uv run python -m src.lake.main`
- Silver builder: `uv run python -m src.lake.silver_feature_builder`
- Gold curator: `uv run python -m src.lake.gold_curator`

---

## Critical Invariants

1. Sorted by `ts_event_ns` within each Parquet file
2. Bronze is append-only (never overwrite)
3. Silver and Gold are reproducible from upstream layers
4. Hive partitioning for date/hour supports predicate pushdown
5. ZSTD compression for all tiers

---

## References

- Module docs: `backend/src/lake/README.md`
- Schemas: `backend/src/common/schemas/`
- ES Silver schema: `backend/SILVER_SCHEMA.md`
- Configuration: `backend/src/common/config.py`
