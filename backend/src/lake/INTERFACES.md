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

# Fuel metrics (flattened)
fuel_effect: string
fuel_net_dealer_gamma: float64
fuel_call_wall: float64
fuel_put_wall: float64

# Runway (flattened)
runway_direction: string
runway_next_level_price: float64
runway_distance: float64
runway_quality: string
```

---

## Bronze Writer Interface

### Class: BronzeWriter

```python
class BronzeWriter:
    async def start(bus: NATSBus) -> None
    async def stop() -> None
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
    async def start(bus: NATSBus) -> None
    async def stop() -> None
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
    def compact_date(
        date: str,
        schema: str,
        partition_value: str
    ) -> Dict[str, Any]
    
    def compact_all_schemas(date: str) -> Dict[str, Dict]
    def compact_all_dates(schema: str, partition_value: str) -> List[Dict]
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
    'status': 'success',
    'rows_read': 12500,
    'rows_written': 12480,
    'duplicates_removed': 20
}
```

---

## Reader Interfaces

### BronzeReader

```python
class BronzeReader:
    def read_stock_trades(
        symbol: str,
        date: str,
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> pd.DataFrame
    
    def read_futures_mbp10(
        symbol: str,
        date: str
    ) -> pd.DataFrame
    
    def get_available_dates(
        schema_path: str,
        partition_filter: str
    ) -> List[str]
```

### SilverReader

```python
class SilverReader:
    def read_futures_trades(
        symbol: str,
        date: str
    ) -> pd.DataFrame
    
    def read_option_trades(
        underlying: str,
        date: str
    ) -> pd.DataFrame
```

### GoldReader

```python
class GoldReader:
    def read_level_signals(
        underlying: str,
        date: str,
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> pd.DataFrame
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

