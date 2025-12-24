# Lake Module Interfaces

**Module**: `backend/src/lake/`
**Role**: Bronze/Silver/Gold lakehouse persistence
**Audience**: AI Coding Agents

---

## Module Purpose

Implements institutional-grade data persistence with Bronze (raw) → Silver (clean) → Gold (derived) tiers. Consumes from NATS, writes to Parquet (local or S3/MinIO).

---

## Input Interfaces

### NATS Subscriptions

**Bronze Writer**:
- `market.stocks.trades` → Bronze `stocks/trades/`
- `market.stocks.quotes` → Bronze `stocks/quotes/`
- `market.options.trades` → Bronze `options/trades/`
- `market.options.greeks` → Bronze `options/greeks_snapshots/`
- `market.futures.trades` → Bronze `futures/trades/`
- `market.futures.mbp10` → Bronze `futures/mbp10/`

**Gold Writer**:
- `levels.signals` → Gold `levels/signals/`

---

## Output Interfaces

### Parquet File Structure

**Bronze Tier** (append-only, immutable):
```
bronze/{asset_class}/{schema}/
  {partition_key}={value}/
    date=YYYY-MM-DD/
      hour=HH/
        part-{timestamp}_{random}.parquet
```

**Example**:
```
bronze/stocks/trades/symbol=SPY/date=2025-12-16/hour=14/part-143012_345678.parquet
bronze/futures/mbp10/symbol=ES/date=2025-12-16/hour=14/part-143012_345678.parquet
bronze/options/trades/underlying=SPY/date=2025-12-16/hour=14/part-143012_345678.parquet
```

**Silver Tier** (deduped, sorted):
```
silver/{asset_class}/{schema}/
  {partition_key}={value}/
    date=YYYY-MM-DD/
      hour=HH/
        part-{sequence}.parquet
```

**Gold Tier** (derived analytics):
```
gold/levels/signals/
  underlying=SPY/
    date=YYYY-MM-DD/
      hour=HH/
        part-{timestamp}_{random}.parquet
```

---

## Storage Schemas

### Bronze: stocks.trades.v1
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

### Bronze: futures.mbp10.v1
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

### Gold: levels.signals.v1 (Flattened)
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

## Bronze Writer Interface

### Class: BronzeWriter

```python
class BronzeWriter:
    def __init__(
        self,
        bus=None,                           # NATSBus (optional, can set via start())
        data_root: Optional[str] = None,     # Root directory for data lake
        buffer_limit: int = 1000,            # Max events per buffer
        flush_interval_seconds: float = 5.0, # Max time between flushes
        use_s3: Optional[bool] = None        # Use S3 storage
    )

    async def start(self, bus=None) -> None   # bus overrides constructor value
    async def stop(self) -> None
    async def flush_all(self) -> None
    async def flush_schema(self, schema_name: str) -> None
    def get_bronze_path(self) -> str
```

**Micro-batching**:
- Buffer limit: 1000 events
- Flush interval: 5.0 seconds
- Separate buffers per schema

**Write Format**:
- PyArrow Table → Parquet
- Compression: ZSTD level 3
- Sort by `ts_event_ns` within file
- Atomic writes (temp → rename)

---

## Gold Writer Interface

### Class: GoldWriter

```python
class GoldWriter:
    def __init__(
        self,
        bus=None,                              # NATSBus (optional)
        data_root: Optional[str] = None,
        buffer_limit: int = 500,               # Max records per buffer
        flush_interval_seconds: float = 10.0,
        use_s3: Optional[bool] = None
    )

    async def start(self, bus=None) -> None
    async def stop(self) -> None
    async def write_level_signals(self, payload: Dict[str, Any]) -> None
    async def flush(self) -> None
    def get_gold_path(self) -> str
```

**Input Processing**:
1. Receive level signals payload from `levels.signals`
2. Flatten each level into single row
3. Buffer up to 500 records or 10 seconds
4. Write to Gold Parquet

**Flattening**:
- Nested `barrier`, `tape`, `fuel`, `runway` dicts → flat columns
- Example: `barrier.state` → `barrier_state`

---

## Silver Compactor Interface

### Class: SilverCompactor

```python
class SilverCompactor:
    def __init__(self, data_root: Optional[str] = None)

    def compact_date(
        self,
        date: str,                    # YYYY-MM-DD
        schema: str,                  # e.g., 'futures.trades'
        partition_value: str = 'ES',  # e.g., 'ES' or 'SPY'
        force: bool = False           # Overwrite existing
    ) -> Dict[str, Any]

    def compact_all_schemas(
        self,
        date: str,
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]    # schema -> result

    def compact_all_dates(
        self,
        schema: str,
        partition_value: str,
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]    # date -> result

    def get_available_bronze_dates(
        self,
        schema: str,
        partition_value: str
    ) -> List[str]                    # List of YYYY-MM-DD dates
```

**Deduplication Strategy**:
```python
event_id = md5(
    source + "|" +
    ts_event_ns + "|" +
    symbol + "|" +
    price + "|" +
    size + "|" +
    seq
)
# Keep first occurrence by ts_recv_ns
```

**Output**:
```python
{
    'status': 'success',  # or 'skipped', 'empty', 'error'
    'bronze_path': str,
    'silver_path': str,
    'rows_read': int,
    'rows_written': int,
    'duplicates_removed': int,
    'reason': Optional[str]  # If skipped/error
}
```

---

## Reader Interfaces

### BronzeReader

```python
class BronzeReader:
    def __init__(self, data_root: Optional[str] = None)

    def read_stock_trades(
        self,
        symbol: str = 'SPY',
        date: Optional[str] = None,          # YYYY-MM-DD
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def read_stock_quotes(
        self,
        symbol: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def read_option_trades(
        self,
        underlying: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def read_greeks_snapshots(
        self,
        underlying: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def get_available_dates(
        self,
        schema_path: str,
        partition_key: str
    ) -> List[str]                           # List of YYYY-MM-DD dates

    def get_latest_date(
        self,
        schema_path: str,
        partition_key: str
    ) -> Optional[str]                       # Most recent date or None
```

### SilverReader

```python
class SilverReader:
    def __init__(self, data_root: Optional[str] = None)

    def read_futures_trades(
        self,
        symbol: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def read_futures_mbp10(
        self,
        symbol: str = 'ES',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def read_stock_trades(
        self,
        symbol: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def read_option_trades(
        self,
        underlying: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def get_available_dates(
        self,
        schema_path: str,
        partition_key: str
    ) -> List[str]
```

### GoldReader

```python
class GoldReader:
    def __init__(self, data_root: Optional[str] = None)

    def read_level_signals(
        self,
        underlying: str = 'SPY',
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> pd.DataFrame

    def get_available_dates(
        self,
        underlying: str = 'SPY'
    ) -> List[str]
```

---

## Storage Backend Configuration

### Local Filesystem (Default)
```python
DATA_ROOT = 'backend/data/lake'
USE_S3 = False
```

### S3/MinIO
```python
DATA_ROOT = 's3://bucket-name'
USE_S3 = True
S3_BUCKET = 'spymaster-lake'
S3_ENDPOINT = 'http://localhost:9000'
S3_ACCESS_KEY = 'minioadmin'
S3_SECRET_KEY = 'minioadmin'
```

---

## Entry Point

```bash
# Start Lake Service
uv run python -m src.lake.main
```

**Output**:
```
============================================================
LAKE SERVICE
============================================================
  Bronze writer: Local storage at /path/to/bronze
  Gold writer: Local storage at /path/to/gold
  Bronze writer: started
  Gold writer: started
============================================================
```

---

## Critical Invariants

1. **Event-time ordering**: All Parquet files sorted by `ts_event_ns`
2. **Idempotency**: Same Bronze input → identical Silver output
3. **No data loss**: NATS JetStream retention ensures durability
4. **Schema stability**: Bronze schemas match `src.common.schemas`
5. **Partition boundaries**: Date/hour aligned to UTC
6. **Compression**: ZSTD level 3 for all tiers
7. **File sizing**: Target 256MB–1GB per Parquet file

---

## Performance Notes

**Micro-batching**:
- Bronze: 1000 events or 5 seconds
- Gold: 500 records or 10 seconds

**Silver Compaction**:
- ~50-100MB/s Bronze → Silver (M4 Mac)
- DuckDB-powered deduplication
- Parallelizable by date/schema

**Storage**:
- Local: ~200-500MB/s (NVMe SSD)
- MinIO: ~100-200MB/s
- S3: ~50-100MB/s

---

## References

- Full module documentation: `backend/src/lake/README.md`
- Storage schemas: `backend/src/common/schemas/`
- Configuration: `backend/src/common/config.py`

