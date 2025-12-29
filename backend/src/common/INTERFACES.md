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
- `FuturesTrade`: ES futures trades
- `MBP10`: ES market-by-price L2 depth (10 levels)
- `OptionTrade`: ES options trades
- `BidAskLevel`: Single bid/ask level in MBP-10

**EventSource Enum**:
```python
class EventSource(str, Enum):
    DIRECT_FEED = "direct_feed"
    REPLAY = "replay"
    SIM = "sim"
```

**Aggressor Enum**:
```python
class Aggressor(int, Enum):
    BUY = 1
    SELL = -1
    MID = 0
```

**Contract**:
```python
# All events carry:
ts_event_ns: int      # Event time (Unix nanoseconds UTC)
ts_recv_ns: int       # Receive time (Unix nanoseconds UTC)
source: EventSource   # DIRECT_FEED | REPLAY | SIM
```

**Import**:
```python
from src.common.event_types import (
    OptionTrade,
    FuturesTrade, MBP10, BidAskLevel,
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
barrier_window = CONFIG.W_b        # 240.0 seconds
tape_window = CONFIG.W_t           # 60.0 seconds
fuel_window = CONFIG.W_g           # 60.0 seconds
velocity_window = CONFIG.W_v       # 3.0 seconds
wall_window = CONFIG.W_wall        # 300.0 seconds

# Bands and thresholds
monitor_band = CONFIG.MONITOR_BAND              # 0.25 ES points
touch_band = CONFIG.TOUCH_BAND                  # 0.10 ES points
barrier_zone_ticks = CONFIG.BARRIER_ZONE_ES_TICKS  # 8 ES ticks
```

**Key Sections**:

**Window Sizes**:
- `W_b`: Barrier engine window (240.0s)
- `W_t`: Tape engine window (60.0s)
- `W_g`: Fuel engine window (60.0s)
- `W_v`: Velocity calculation window (3.0s)
- `W_wall`: Call/put wall lookback (300.0s)

**Monitoring Bands**:
- `MONITOR_BAND`: 0.25 ES points (compute signals)
- `TOUCH_BAND`: 0.10 ES points (level touching)
- `CONFLUENCE_BAND`: 0.20 ES points (nearby levels)
- `BARRIER_ZONE_ES_TICKS`: 8 ES ticks around level

**Barrier Thresholds**:
- `R_vac`: 0.3 (VACUUM replenishment ratio)
- `R_wall`: 1.5 (WALL replenishment ratio)
- `F_thresh`: 100 (delta liquidity, ES contracts)

**Tape Thresholds**:
- `TAPE_BAND`: 0.50 ES points
- `SWEEP_MIN_NOTIONAL`: 500,000.0
- `SWEEP_MAX_GAP_MS`: 100ms
- `SWEEP_MIN_VENUES`: 1

**Score Weights**:
- `w_L`: 0.45 (liquidity)
- `w_H`: 0.35 (hedge)
- `w_T`: 0.20 (tape)

**Trigger Thresholds**:
- `BREAK_SCORE_THRESHOLD`: 80.0
- `REJECT_SCORE_THRESHOLD`: 20.0
- `TRIGGER_HOLD_TIME`: 3.0 seconds

**Feasibility Priors**:
- `FEASIBILITY_LOGIT_STEP`: 1.0
- `FEASIBILITY_LOGIT_CAP`: 2.5

**Smoothing (EWMA half-lives)**:
- `tau_score`: 2.0 seconds
- `tau_velocity`: 1.5 seconds
- `tau_delta_liq`: 3.0 seconds
- `tau_dealer_gamma`: 5.0 seconds

**Outcome Labeling**:
- `CONFIRMATION_WINDOW_SECONDS`: 240.0 (Stage B confirmation window)
- `CONFIRMATION_WINDOWS_MULTI`: [120.0, 240.0, 480.0] (2/4/8 min horizons)

**Snap Cadence**:
- `SNAP_INTERVAL_MS`: 250 (publish every 250ms)

**NATS/S3 Settings**:
- `NATS_URL`: from `NATS_URL` env var (default: `nats://localhost:4222`)
- `S3_ENDPOINT`: from `S3_ENDPOINT` env var
- `S3_BUCKET`: from `S3_BUCKET` env var
- `S3_ACCESS_KEY`, `S3_SECRET_KEY`: from env vars

**Replay Settings**:
- `REPLAY_SPEED`: from `REPLAY_SPEED` env var (1.0 = realtime)

---

### 3. Storage Schemas (`schemas/`)

**Purpose**: Pydantic + PyArrow schema definitions for Bronze/Silver/Gold tiers.

**Schema Registry**:
```python
from src.common.schemas import SchemaRegistry

# List all schemas
schemas = SchemaRegistry.list_schemas()

# Get Pydantic model
model = SchemaRegistry.get('futures.trades.v1')

# Get Arrow schema for Parquet
arrow_schema = SchemaRegistry.get_arrow_schema('futures.trades.v1')
```

**Key Schemas**:
- Bronze: `options.trades.v1`, `futures.trades.v1`, `futures.mbp10.v1`
- Silver: `silver.features.es_pipeline.v1`
- Gold: `training.es_pipeline.v1`, `levels.signals.v1`

---

### 4. NATS Bus (`bus.py`)

**Purpose**: NATS JetStream wrapper for pub/sub messaging.

**Interface**:
```python
class NATSBus:
    def __init__(self, servers: list[str] = ["nats://localhost:4222"])

    async def connect(self) -> None
    async def publish(self, subject: str, payload: Any) -> Ack
    async def subscribe(
        self,
        subject: str,
        callback: Callable[[Any], Awaitable[None]],
        durable_name: Optional[str] = None
    ) -> Subscription
    async def close(self) -> None
```

**Publish Behavior**:
- Accepts Pydantic models, dataclasses, and dicts
- Automatically serializes to JSON
- Enums serialized as `.value`

**Global Singleton**:
```python
from src.common.bus import BUS

await BUS.connect()
await BUS.publish('market.futures.trades', trade_dict)
await BUS.subscribe('levels.signals', handle_signal, 'consumer_name')
```

**Streams Created**:
- `MARKET_DATA`: subjects `market.*`, `market.*.*` (24h retention)
- `LEVEL_SIGNALS`: subjects `levels.*` (24h retention)

---

### 5. Run Manifest Manager (`run_manifest_manager.py`)

**Purpose**: Track run metadata for reproducibility.

**Classes**:

```python
class RunStatus(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    CRASHED = "crashed"
    STOPPED = "stopped"

class RunMode(Enum):
    LIVE = "live"
    REPLAY = "replay"
    SIM = "sim"

@dataclass
class RunManifest:
    run_id: str
    start_time: str                    # ISO 8601 UTC
    end_time: Optional[str]            # ISO 8601 UTC
    status: str                        # RunStatus value
    mode: str                          # RunMode value
    code_commit: Optional[str]
    code_branch: Optional[str]
    code_dirty: bool
    config_hash: str
    bronze_files: List[str]
    gold_files: List[str]
    event_counts: Dict[str, int]
    error_message: Optional[str]
```

**Interface**:
```python
class RunManifestManager:
    def __init__(
        self,
        data_root: Optional[str] = None,
        mode: RunMode = RunMode.LIVE
    )

    def start_run(self) -> str                           # Returns run_id
    def track_bronze_file(self, file_path: str) -> None
    def track_gold_file(self, file_path: str) -> None
    def update_event_count(self, schema_name: str, count: int) -> None
    def complete_run(self, status: RunStatus = RunStatus.COMPLETED) -> None
    def mark_crashed(self, error_message: str) -> None
    def get_current_run_id(self) -> Optional[str]
    def load_manifest(self, run_id: str) -> Optional[RunManifest]
    def list_runs(
        self,
        mode: Optional[RunMode] = None,
        status: Optional[RunStatus] = None
    ) -> List[RunManifest]
    def get_crashed_runs(self) -> List[RunManifest]
    def get_run_directory(self, run_id: str) -> str
```

**Manifest Storage**:
```
{data_root}/_meta/runs/{run_id}/
  ├── manifest.json
  ├── config_snapshot.json
  └── schemas/versions.json
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

---

## Schema Versioning Contract

**Version Bump Triggers**:
- Add required field → bump version (breaking change)
- Add optional field → can stay same version
- Remove field → bump version
- Change field type → bump version

**Example**:
```python
# v1
class FuturesTradeV1(BaseEventModel):
    _schema_version = SchemaVersion('futures.trades', version=1, tier='bronze')
    price: float
    size: int

# v2 (optional field added)
class FuturesTradeV2(BaseEventModel):
    _schema_version = SchemaVersion('futures.trades', version=2, tier='bronze')
    price: float
    size: int
    venue_timestamp: Optional[int] = None
```

---

## Critical Invariants

1. **Time units**: All timestamps are Unix nanoseconds UTC
2. **Enum serialization**: JSON-encode enums as `.value` for NATS
3. **Config immutability**: Don't modify CONFIG during runtime
4. **Schema stability**: Version every schema change and update consumers together
5. **NATS streams**: MARKET_DATA and LEVEL_SIGNALS with 24h retention

---

## References

- Full module documentation: `backend/src/common/README.md`
- Configuration parameters: `backend/src/common/config.py` (inline comments)
- Schema definitions: `backend/src/common/schemas/*.py`
