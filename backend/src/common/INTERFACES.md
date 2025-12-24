# Common Module Interfaces

**Module**: `backend/src/common/`  
**Role**: Shared contracts, schemas, and configuration  
**Audience**: AI Coding Agents

---

## Module Purpose

Provides foundational infrastructure with zero dependencies on other backend modules. All other modules depend on `common`.

---

## Core Interfaces

### 1. Event Types (`event_types.py`)

**Purpose**: Canonical event dataclasses for runtime message passing.

**Key Types**:
- `StockTrade`: SPY equity trades
- `StockQuote`: SPY NBBO quotes
- `OptionTrade`: SPY option trades
- `FuturesTrade`: ES futures trades
- `MBP10`: ES market-by-price L2 depth (10 levels)
- `GreeksSnapshot`: Option greeks cache entries

**Contract**:
```python
# All events carry:
ts_event_ns: int      # Event time (Unix nanoseconds UTC)
ts_recv_ns: int       # Receive time (Unix nanoseconds UTC)
source: EventSource   # POLYGON_WS | DIRECT_FEED | REPLAY | etc.
```

**Import**:
```python
from src.common.event_types import (
    StockTrade, StockQuote, OptionTrade, 
    FuturesTrade, MBP10, GreeksSnapshot,
    EventSource, Aggressor
)
```

---

### 2. Configuration (`config.py`)

**Purpose**: Single source of truth for all tunable parameters.

**Access Pattern**:
```python
from src.common.config import CONFIG

# Physics windows
barrier_window = CONFIG.W_b        # 10.0 seconds
tape_window = CONFIG.W_t           # 5.0 seconds

# Bands and thresholds
monitor_band = CONFIG.MONITOR_BAND # 0.50 SPY dollars
```

**Key Sections**:
- Window sizes: `W_b`, `W_t`, `W_g`, `W_v`, `W_wall`
- Monitoring bands: `MONITOR_BAND`, `TOUCH_BAND`, `BARRIER_ZONE_SPY`
- Thresholds: `R_vac`, `R_wall`, `F_thresh`
- Score weights: `w_L`, `w_H`, `w_T`
- Smoothing: `tau_score`, `tau_velocity`, `tau_delta_liq`

---

### 3. Price Converter (`price_converter.py`)

**Purpose**: ES ↔ SPY price conversion (ES ≈ SPY × 10).

**Interface**:
```python
class PriceConverter:
    def spy_to_es(spy_price: float) -> float
    def es_to_spy(es_price: float) -> float
    def update_es_price(es_price: float)
    def update_spy_price(spy_price: float)
    
    @property
    def ratio(self) -> float  # Dynamic if both prices available
```

**Usage**:
```python
converter = PriceConverter()
es_level = converter.spy_to_es(687.0)  # → 6870.0
```

---

### 4. Storage Schemas (`schemas/`)

**Purpose**: Pydantic + PyArrow schema definitions for Bronze/Silver/Gold tiers.

**Schema Registry**:
```python
from src.common.schemas import SchemaRegistry

# List all schemas
schemas = SchemaRegistry.list_schemas()

# Get Pydantic model
model = SchemaRegistry.get('stocks.trades.v1')

# Get Arrow schema for Parquet
arrow_schema = SchemaRegistry.get_arrow_schema('stocks.trades.v1')
```

**Key Schemas**:
- Bronze: `stocks.trades.v1`, `stocks.quotes.v1`, `options.trades.v1`, `futures.trades.v1`, `futures.mbp10.v1`
- Silver: `options.trades_enriched.v1`
- Gold: `levels.signals.v1`

---

### 5. NATS Bus (`bus.py`)

**Purpose**: NATS JetStream wrapper for pub/sub messaging.

**Interface**:
```python
class NATSBus:
    async def connect() -> None
    async def publish(subject: str, data: Dict) -> None
    async def subscribe(subject: str, callback: Callable, durable_name: str) -> None
    async def close() -> None
```

**Usage**:
```python
from src.common.bus import BUS

await BUS.connect()
await BUS.publish('market.stocks.trades', trade_dict)
await BUS.subscribe('levels.signals', handle_signal, 'consumer_name')
```

---

### 6. Run Manifest Manager (`run_manifest_manager.py`)

**Purpose**: Track run metadata for reproducibility.

**Interface**:
```python
class RunManifestManager:
    def start_run() -> str  # Returns run_id
    def track_bronze_file(path: str)
    def update_event_count(schema: str, count: int)
    def complete_run(status: RunStatus)
    def mark_crashed(error: str)
```

---

## Data Flow Contract

**Event Flow**:
```
Ingestor → event_types dataclass → NATS (bus.py) → Core/Lake/Gateway
                                        ↓
                                    schemas/ (Parquet storage)
```

**Configuration Access**:
```
Any module → config.py (CONFIG singleton) → Read parameters
```

**Price Conversion**:
```
Level (SPY dollars) → price_converter → ES query → Results (convert back to SPY)
```

---

## Schema Versioning Contract

**Version Bump Triggers**:
- Add required field → bump version (backward incompatible)
- Add optional field → can stay same version
- Remove field → bump version
- Change field type → bump version

**Example**:
```python
# v1
class StockTradeV1(BaseEventModel):
    _schema_version = SchemaVersion('stocks.trades', version=1, tier='bronze')
    price: float
    size: int

# v2 (optional field added)
class StockTradeV2(BaseEventModel):
    _schema_version = SchemaVersion('stocks.trades', version=2, tier='bronze')
    price: float
    size: int
    venue_timestamp: Optional[int] = None
```

---

## Critical Invariants

1. **Time units**: All timestamps are Unix nanoseconds UTC
2. **Enum serialization**: JSON-encode enums as `.value` for NATS
3. **ES/SPY conversion**: Always use `PriceConverter`, never hardcode ratio
4. **Config immutability**: Don't modify CONFIG during runtime
5. **Schema stability**: Never break backward compatibility without version bump

---

## References

- Full module documentation: `backend/src/common/README.md`
- Configuration parameters: `backend/src/common/config.py` (inline comments)
- Schema definitions: `backend/src/common/schemas/*.py`

