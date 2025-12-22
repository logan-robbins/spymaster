# SPY Break/Reject Physics Engine — `common` Module

**Technical Reference for AI Coding Agents**

---

## Module Purpose

The `common` module provides **foundational infrastructure** for the SPY Break/Reject Physics Engine. It defines stable contracts that enable parallel agent development and deterministic replay across the entire system.

**Architectural principle**: This module is **dependency-free within the backend** (no imports from `core`, `gateway`, `ingestor`, or `lake`). All other backend modules depend on `common`.

**Ownership**: Agent A (per PLAN.md §12) — completed ✅

---

## Directory Structure

```
backend/src/common/
├── __init__.py              # Module initialization
├── bus.py                   # NATS JetStream message bus (Phase 2+)
├── config.py                # Single source of truth for all parameters
├── event_types.py           # Canonical event dataclasses (message envelope)
├── price_converter.py       # ES ↔ SPY price conversion (ES ≈ SPY × 10)
├── run_manifest_manager.py  # Run metadata tracking (Phase 1 institutional hygiene)
└── schemas/                 # Arrow + Pydantic schema definitions
    ├── __init__.py
    ├── base.py              # SchemaRegistry, base models, type mappings
    ├── stocks_trades.py     # stocks.trades.v1 (Bronze)
    ├── stocks_quotes.py     # stocks.quotes.v1 (Bronze)
    ├── options_trades.py    # options.trades.v1 (Bronze)
    ├── options_greeks.py    # options.greeks_snapshots.v1 (Bronze)
    ├── futures_trades.py    # futures.trades.v1 (Bronze, ES)
    ├── futures_mbp10.py     # futures.mbp10.v1 (Bronze, ES MBP-10)
    ├── options_trades_enriched.py  # options.trades_enriched.v1 (Silver)
    └── levels_signals.py    # levels.signals.v1 (Gold)
```

---

## Core Components

### 1. `event_types.py` — Canonical Event Dataclasses

**Purpose**: Defines the **normalized event envelope** for all internal message passing and storage (PLAN.md §2.1).

**Key Contracts**:
- All events carry `ts_event_ns` (event time, UTC nanoseconds) and `ts_recv_ns` (receive time, UTC nanoseconds).
- All events carry `source: EventSource` to track origin (massive_ws, replay, sim, etc.).
- Optional `seq` field for monotonic ordering diagnostics.

**Event Types**:

```python
from src.common.event_types import (
    StockTrade,      # SPY trades (normalized from Massive WS T.SPY)
    StockQuote,      # SPY NBBO (normalized from Massive WS Q.SPY)
    OptionTrade,     # SPY option trades (normalized from Massive WS T.O:SPY...)
    GreeksSnapshot,  # Greeks from REST API cache
    FuturesTrade,    # ES trades (normalized from Databento DBN)
    MBP10,           # ES MBP-10 depth (normalized from Databento DBN)
    EventSource,     # Enum: MASSIVE_WS | POLYGON_WS | REPLAY | SIM | ...
    Aggressor,       # Enum: BUY=1 | SELL=-1 | MID=0
)
```

**Critical Implementation Details**:

1. **Timestamps are Unix nanoseconds (UTC)**:
   - Vendor WS timestamps arrive in **milliseconds** → multiply by `1_000_000` to get `ts_event_ns`
   - Databento DBN timestamps are already nanoseconds → use directly
   - `ts_recv_ns = time.time_ns()` at ingestion

2. **StockQuote sizes are in shares (not round lots)**:
   - Per Massive API change 2025-11-03 (SEC MDI rules): `bid_sz` / `ask_sz` are **shares**
   - See PLAN.md §11.1

3. **Aggressor classification**:
   - `BUY` (1): trade lifted ask (bullish)
   - `SELL` (-1): trade hit bid (bearish)
   - `MID` (0): unknown or inside spread

**Usage in Agents**:

```python
# Ingestion (Agent B)
trade = StockTrade(
    ts_event_ns=msg['t'] * 1_000_000,  # Massive sends ms → convert to ns
    ts_recv_ns=time.time_ns(),
    source=EventSource.MASSIVE_WS,
    symbol=msg['sym'],
    price=msg['p'],
    size=msg['s'],
    exchange=msg.get('x'),
    conditions=msg.get('c', []),
    seq=msg.get('seq')
)

# Engine consumption (Agent C/D/E)
if isinstance(event, StockTrade):
    market_state.update_stock_trade(event)
elif isinstance(event, StockQuote):
    market_state.update_stock_quote(event)
```

---

### 2. `config.py` — Configuration Single Source of Truth

**Purpose**: Centralize all tunable parameters (PLAN.md §9). No trained calibration in v1; all constants are mechanical and physics-based.

**Design Principle**: **Single Config class** → import `CONFIG` singleton everywhere.

**Parameter Categories**:

1. **Window sizes** (`W_b`, `W_t`, `W_g`, `W_v`, `W_wall`):
   - `W_b = 10.0` seconds: Barrier engine quote/trade accounting window
   - `W_t = 5.0` seconds: Tape engine imbalance window
   - `W_g = 60.0` seconds: Fuel engine gamma aggregation window
   - `W_v = 3.0` seconds: Velocity slope calculation window
   - `W_wall = 300.0` seconds: Call/Put wall lookback (5 minutes)

2. **Monitoring bands** (SPY dollars):
   - `MONITOR_BAND = 0.50`: compute full signals when |spot - level| ≤ $0.50
   - `TOUCH_BAND = 0.05`: tight band for "touching level" triggers
   - `BARRIER_ZONE_TICKS = 2`: price zone around level for depth grouping

3. **Barrier thresholds** (ES contracts):
   - `R_vac = 0.3`: Replenishment ratio for VACUUM classification
   - `R_wall = 1.5`: Replenishment ratio for WALL/ABSORPTION
   - `F_thresh = 100`: Delta liquidity threshold (ES contracts, not shares)

4. **Tape thresholds** (SPY scale, converted to ES internally):
   - `TAPE_BAND = 0.10`: price band for imbalance calculation
   - `SWEEP_MIN_NOTIONAL = 500_000.0`: minimum $ for sweep detection (ES @ $50/point)
   - `SWEEP_MAX_GAP_MS = 100`: max gap between prints in sweep cluster

5. **Fuel thresholds**:
   - `FUEL_STRIKE_RANGE = 2.0`: monitor strikes within ±$2 of level

6. **Score weights** (GM.md composite score):
   - `w_L = 0.45`: Liquidity (Barrier) weight
   - `w_H = 0.35`: Hedge (Fuel) weight
   - `w_T = 0.20`: Tape (Momentum) weight

7. **Trigger thresholds**:
   - `BREAK_SCORE_THRESHOLD = 80.0`: score > 80 → BREAK signal
   - `REJECT_SCORE_THRESHOLD = 20.0`: score < 20 → REJECT signal
   - `TRIGGER_HOLD_TIME = 3.0`: seconds score must be sustained

8. **Smoothing parameters** (EWMA half-lives):
   - `tau_score = 2.0`: break score smoothing
   - `tau_velocity = 1.5`: tape velocity smoothing
   - `tau_delta_liq = 3.0`: barrier delta_liq smoothing
   - (See PLAN.md §5.6 for EWMA formula)

9. **Snap tick cadence**:
   - `SNAP_INTERVAL_MS = 250`: publish level signals every 250ms
   - **Important**: Book/trade ingestion remains **event-driven** (process every update); this is the *scoring/publish* cadence

10. **Level universe settings**:
    - `ROUND_LEVELS_SPACING = 1.0`: generate round levels every $1
    - `STRIKE_RANGE = 5.0`: monitor strikes within ±$5 of spot
    - `VWAP_ENABLED = True`: include VWAP as a level

11. **Storage/replay settings**:
    - `DATA_ROOT = "backend/data/lake/"`: lakehouse root
    - `MICRO_BATCH_INTERVAL_S = 5.0`: flush Bronze Parquet every 5s

**Usage**:

```python
from src.common.config import CONFIG

# In barrier_engine.py (Agent D)
window_secs = CONFIG.W_b
r_vac = CONFIG.R_vac
delta_liq_thresh = CONFIG.F_thresh

# In score_engine.py (Agent G)
break_score = (
    CONFIG.w_L * liquidity_score +
    CONFIG.w_H * hedge_score +
    CONFIG.w_T * tape_score
)
```

**Tuning Protocol**:
- Modify `CONFIG` dataclass defaults in `config.py`
- Restart system (v1); later: hot-reload or per-run overrides
- Run manifests capture exact config snapshots → reproducible experiments

---

### 3. `price_converter.py` — ES ↔ SPY Price Conversion

**Purpose**: Enable **SPY-denominated trading decisions** while using **ES futures liquidity** for barrier physics (PLAN.md §0.1).

**Why This Exists**:
- **Levels are SPY prices** (strikes, rounds) because we trade SPY 0DTE options
- **Barrier physics uses ES MBP-10** because ES has superior depth visibility vs SPY L1
- **Price conversion**: ES ≈ SPY × 10 (dynamic ratio supported)

**Design**:
```python
class PriceConverter:
    DEFAULT_RATIO = 10.0  # ES/SPY baseline
    
    def update_es_price(es_price: float)   # From ES trade
    def update_spy_price(spy_price: float) # From SPY trade or quote
    
    @property
    def ratio(self) -> float  # Dynamic if both prices available, else DEFAULT_RATIO
    
    def es_to_spy(es_price: float) -> float
    def spy_to_es(spy_price: float) -> float
    def es_ticks_to_spy_dollars(es_ticks: int, es_tick_size=0.25) -> float
    def spy_dollars_to_es_ticks(spy_dollars: float, es_tick_size=0.25) -> float
```

**Critical Usage Patterns**:

1. **MarketState (Agent C)**:
   ```python
   # Store both raw ES and SPY-equivalent
   self.converter = PriceConverter()
   
   def update_es_trade(self, trade: FuturesTrade):
       self.last_es_price = trade.price
       self.converter.update_es_price(trade.price)
   
   def get_spot(self) -> float:
       """Returns SPY-equivalent spot price."""
       return self.converter.es_to_spy(self.last_es_price)
   
   def get_es_spot(self) -> float:
       """Returns raw ES price for barrier queries."""
       return self.last_es_price
   ```

2. **BarrierEngine (Agent D)**:
   ```python
   # Input: SPY level $687.00
   # Query: ES depth at 687.00 * converter.ratio ≈ 6870.0
   def compute_barrier_state(
       self,
       spy_level: float,
       market_state: MarketState
   ) -> BarrierState:
       es_level = market_state.converter.spy_to_es(spy_level)
       
       # Query ES MBP-10 around es_level (e.g., 6870.0 ± 2 ticks)
       defending_size = market_state.get_es_depth_near(es_level, direction)
       
       # Compute physics, return result in SPY context
       return BarrierState(...)
   ```

3. **TapeEngine (Agent D)**:
   ```python
   # Convert SPY bands to ES for trade filtering
   tape_band_spy = CONFIG.TAPE_BAND  # 0.10 SPY dollars
   tape_band_es = market_state.converter.spy_to_es(tape_band_spy)
   ```

**Dynamic Ratio**:
- Ratio updates whenever both ES and SPY prices are available
- Accounts for dividend expectations, interest rate differential, fair value basis
- Typical range: 9.98–10.02 intraday

**Testing**: See `backend/tests/test_price_converter.py` (16 tests)

---

### 4. `run_manifest_manager.py` — Run Metadata Tracking

**Purpose**: Phase 1 institutional hygiene (PLAN.md §1.2, §2.5). Enables reproducibility, debugging, and ML dataset provenance.

**What It Tracks**:

```python
@dataclass
class RunManifest:
    run_id: str                    # Unique: 2025-12-22_143015_123456_live_abc123
    start_time: str                # ISO 8601 UTC
    end_time: Optional[str]        # ISO 8601 UTC
    status: str                    # STARTED | COMPLETED | CRASHED | STOPPED
    mode: str                      # LIVE | REPLAY | SIM
    
    # Code version (git)
    code_commit: Optional[str]     # SHA if available
    code_branch: Optional[str]     # Branch name
    code_dirty: bool               # Uncommitted changes?
    
    # Config hash
    config_hash: str               # MD5(CONFIG) for quick comparison
    
    # Output tracking
    bronze_files: List[str]        # Relative paths to Bronze Parquet files
    gold_files: List[str]          # Relative paths to Gold Parquet files
    event_counts: Dict[str, int]   # {'stocks.trades.v1': 1234567, ...}
    
    # Error tracking
    error_message: Optional[str]   # If crashed
```

**Manifest Storage**:
```
backend/data/lake/_meta/runs/
  2025-12-22_143015_123456_live_abc123/
    manifest.json           # RunManifest as JSON
    config_snapshot.json    # Exact CONFIG values
    schemas/
      versions.json         # Schema registry snapshot
```

**Usage in Ingestor (Agent I)**:

```python
from src.common.run_manifest_manager import RunManifestManager, RunMode, RunStatus

# Start ingestion run
manifest_mgr = RunManifestManager(mode=RunMode.LIVE)
run_id = manifest_mgr.start_run()

try:
    # During ingestion
    manifest_mgr.track_bronze_file(bronze_file_path)
    manifest_mgr.update_event_count('stocks.trades.v1', batch_size)
    
    # On graceful shutdown
    manifest_mgr.complete_run(status=RunStatus.COMPLETED)
    
except Exception as e:
    manifest_mgr.mark_crashed(str(e))
    raise
```

**Querying Historical Runs**:
```python
# List all completed live runs
runs = manifest_mgr.list_runs(mode=RunMode.LIVE, status=RunStatus.COMPLETED)

# Load specific run
manifest = manifest_mgr.load_manifest(run_id)
print(f"Events: {manifest.event_counts}")
print(f"Config hash: {manifest.config_hash}")
print(f"Git commit: {manifest.code_commit}")
```

**Use Cases**:
1. **Reproducibility**: "Re-run with exact same config/code"
2. **Debugging**: "What changed between these two runs?"
3. **ML provenance**: "Which run produced this training dataset?"
4. **Crash recovery**: "Which runs crashed? What was the error?"

**Testing**: See `backend/tests/test_run_manifest_manager.py` (24 tests)

---

### 5. `schemas/` — Bronze/Silver/Gold Schema Definitions

**Purpose**: Provide **typed, versioned schemas** for all data tiers (PLAN.md §2.2, §2.4). Dual representation: Pydantic (runtime validation) + PyArrow (Parquet storage).

**Architecture**:

```
schemas/base.py              # SchemaRegistry, type mappings, utilities
schemas/stocks_trades.py     # Bronze: stocks.trades.v1
schemas/stocks_quotes.py     # Bronze: stocks.quotes.v1
schemas/options_trades.py    # Bronze: options.trades.v1
schemas/options_greeks.py    # Bronze: options.greeks_snapshots.v1
schemas/futures_trades.py    # Bronze: futures.trades.v1
schemas/futures_mbp10.py     # Bronze: futures.mbp10.v1
schemas/options_trades_enriched.py  # Silver: options.trades_enriched.v1
schemas/levels_signals.py    # Gold: levels.signals.v1
```

#### 5.1 Schema Registry

**Purpose**: Lookup schemas by name/version, ensure backward compatibility.

```python
from src.common.schemas import SchemaRegistry

# List all registered schemas
schemas = SchemaRegistry.list_schemas()
# ['stocks.trades.v1', 'stocks.quotes.v1', ...]

# Get Pydantic model
model_class = SchemaRegistry.get('stocks.trades.v1')

# Get PyArrow schema for Parquet writing
arrow_schema = SchemaRegistry.get_arrow_schema('stocks.trades.v1')
```

#### 5.2 Bronze Tier Schemas

**Bronze = raw, normalized, immutable** (PLAN.md §2.2)

**`stocks.trades.v1`**:
```python
from src.common.schemas import StockTradeV1

trade = StockTradeV1(
    ts_event_ns=1734567890000000000,
    ts_recv_ns=1734567890001000000,
    source='massive_ws',
    symbol='SPY',
    price=687.50,
    size=100,
    exchange=4,       # Optional
    conditions=[14],  # Optional
    seq=12345         # Optional
)

# Get Arrow schema for Parquet
arrow_schema = StockTradeV1._arrow_schema
```

**`stocks.quotes.v1`** (NBBO):
```python
from src.common.schemas import StockQuoteV1

quote = StockQuoteV1(
    ts_event_ns=...,
    ts_recv_ns=...,
    source='massive_ws',
    symbol='SPY',
    bid_px=687.49,
    ask_px=687.51,
    bid_sz=500,    # SHARES (not round lots, per Massive 2025-11-03)
    ask_sz=300,    # SHARES
    bid_exch=4,    # Optional
    ask_exch=19    # Optional
)
```

**`options.trades.v1`**:
```python
from src.common.schemas import OptionTradeV1, AggressorEnum

opt_trade = OptionTradeV1(
    ts_event_ns=...,
    ts_recv_ns=...,
    source='massive_ws',
    underlying='SPY',
    option_symbol='O:SPY251216C00687000',
    exp_date='2025-12-16',
    strike=687.0,
    right='C',
    price=2.45,
    size=10,
    opt_bid=2.40,      # Optional: option BBO
    opt_ask=2.50,      # Optional
    aggressor=AggressorEnum.BUY,  # Inferred from opt_bid/opt_ask
    conditions=[],
    seq=67890
)
```

**`futures.trades.v1`** (ES):
```python
from src.common.schemas import FuturesTradeV1

es_trade = FuturesTradeV1(
    ts_event_ns=...,     # Databento: record.ts_event (already ns)
    ts_recv_ns=...,
    source='direct_feed',
    symbol='ES',
    price=6870.0,
    size=5,
    aggressor=AggressorEnum.BUY,  # Databento: record.side 'A'/'B'/'N'
    exchange='CME',
    seq=...
)
```

**`futures.mbp10.v1`** (ES MBP-10):
```python
from src.common.schemas import MBP10V1, BidAskLevelModel

mbp10 = MBP10V1(
    ts_event_ns=...,
    ts_recv_ns=...,
    source='direct_feed',
    symbol='ES',
    levels=[
        BidAskLevelModel(
            bid_px=6870.0, bid_sz=50,
            ask_px=6870.25, ask_sz=30
        ),
        # ... 9 more levels
    ],
    is_snapshot=True,  # vs incremental update
    seq=...
)
```

**`options.greeks_snapshots.v1`**:
```python
from src.common.schemas import GreeksSnapshotV1

greeks = GreeksSnapshotV1(
    ts_event_ns=...,
    source='massive_rest',
    underlying='SPY',
    option_symbol='O:SPY251216C00687000',
    delta=0.52,
    gamma=0.035,
    theta=-0.12,
    vega=0.08,
    implied_volatility=0.14,  # Optional
    open_interest=12345,      # Optional
    snapshot_id='abc123'      # MD5 hash for dedup
)
```

#### 5.3 Silver Tier Schemas

**Silver = cleaned, normalized, deduped, enriched** (PLAN.md §2.2)

**`options.trades_enriched.v1`**:
```python
from src.common.schemas import OptionTradeEnrichedV1

# Silver compactor joins option trades with greeks snapshots
enriched = OptionTradeEnrichedV1(
    # All fields from OptionTradeV1
    ts_event_ns=...,
    underlying='SPY',
    option_symbol='O:SPY251216C00687000',
    strike=687.0,
    right='C',
    price=2.45,
    size=10,
    aggressor=AggressorEnum.BUY,
    
    # Enrichment from greeks snapshot
    greeks_snapshot_id='abc123',
    delta=0.52,
    gamma=0.035,
    delta_notional=0.52 * 10 * 100,  # contracts * multiplier
    gamma_notional=0.035 * 10 * 100
)
```

#### 5.4 Gold Tier Schemas

**Gold = derived analytics, feature tables, ML-ready** (PLAN.md §2.2)

**`levels.signals.v1`** (primary output of scoring engine):
```python
from src.common.schemas import (
    LevelSignalV1,
    LevelKindEnum,
    DirectionEnum,
    BarrierStateEnum,
    FuelEffectEnum,
    SignalEnum,
    ConfidenceEnum,
    RunwayQualityEnum
)

signal = LevelSignalV1(
    ts_event_ns=...,         # Snap tick time
    underlying='SPY',
    spot=687.42,
    bid=687.41,
    ask=687.43,
    
    # Level identification
    level_id='STRIKE_687',
    level_kind=LevelKindEnum.STRIKE,
    level_price=687.0,
    direction=DirectionEnum.SUPPORT,  # approaching from above
    distance=0.42,
    
    # Scores
    break_score_raw=88.5,
    break_score_smooth=81.2,
    signal=SignalEnum.BREAK,
    confidence=ConfidenceEnum.HIGH,
    
    # Barrier metrics (flattened for Parquet)
    barrier_state=BarrierStateEnum.VACUUM,
    barrier_delta_liq=-8200.0,
    barrier_replenishment_ratio=0.15,
    barrier_added=3100,
    barrier_canceled=9800,
    barrier_filled=1500,
    
    # Tape metrics
    tape_imbalance=-0.45,
    tape_buy_vol=120000,
    tape_sell_vol=320000,
    tape_velocity=-0.08,
    tape_sweep_detected=True,
    tape_sweep_direction='DOWN',
    tape_sweep_notional=1250000.0,
    
    # Fuel metrics
    fuel_effect=FuelEffectEnum.AMPLIFY,
    fuel_net_dealer_gamma=-185000.0,
    fuel_call_wall=690.0,
    fuel_put_wall=684.0,
    fuel_hvl=687.0,
    
    # Runway metrics
    runway_direction='DOWN',
    runway_next_level_id='PUT_WALL',
    runway_next_level_price=684.0,
    runway_distance=3.0,
    runway_quality=RunwayQualityEnum.CLEAR,
    
    # Optional note
    note='Vacuum + dealers chase; sweep confirms'
)
```

**Flattening Strategy**:
- Gold schemas use **flattened fields** (no nested objects) for efficient columnar storage
- Example: `barrier.state` → `barrier_state`, `barrier.delta_liq` → `barrier_delta_liq`
- Parquet compression (ZSTD) + dictionary encoding on enums → small files

#### 5.5 Schema Evolution

**Version Bump Triggers**:
- Add required field → bump version (backward incompatible)
- Add optional field → can stay same version (forward compatible)
- Remove field → bump version
- Change field type → bump version

**Example Evolution**:
```python
# v1: stocks.trades.v1
@SchemaRegistry.register
class StockTradeV1(BaseEventModel):
    _schema_version = SchemaVersion('stocks.trades', version=1, tier='bronze')
    price: float
    size: int

# v2: stocks.trades.v2 (add venue_timestamp)
@SchemaRegistry.register
class StockTradeV2(BaseEventModel):
    _schema_version = SchemaVersion('stocks.trades', version=2, tier='bronze')
    price: float
    size: int
    venue_timestamp: Optional[int] = None  # New optional field
```

**Testing**: See `backend/tests/test_schemas.py` (31 tests)

---

### 6. `bus.py` — NATS JetStream Message Bus (Phase 2+)

**Purpose**: Replace in-process `asyncio.Queue` with persistent, distributed message bus (PLAN.md §1.2 Phase 2/3).

**Current Status**: Implemented but **not used in Phase 1**. Phase 1 uses direct in-process queues.

**Phase 2+ Architecture**:

```python
from src.common.bus import BUS

# Connect
await BUS.connect()

# Publish event
await BUS.publish('market.stocks.trades', stock_trade)
await BUS.publish('levels.signals', level_signal)

# Subscribe
async def handle_trade(data):
    trade = StockTradeV1(**data)
    # Process trade
    ...

await BUS.subscribe(
    'market.stocks.trades',
    callback=handle_trade,
    durable_name='barrier_engine_consumer'
)
```

**NATS Streams**:
```python
MARKET_DATA:
  subjects: ['market.*', 'market.*.*']
  retention: 24 hours
  examples: market.stocks.trades, market.futures.mbp10

LEVEL_SIGNALS:
  subjects: ['levels.*']
  retention: 24 hours
  examples: levels.signals
```

**Why NATS JetStream** (vs Redis/Kafka):
- Lightweight (single binary, <50MB memory footprint)
- Built-in persistence + replay by offset/time
- Excellent latency (sub-ms for colocation)
- Simpler ops than Kafka for single-machine Phase 2

**Phase 3 Alternative**: Redpanda/Kafka for multi-node colocation with schema registry.

**Testing**: Not currently tested (Phase 2+ feature).

---

## Integration Patterns

### Pattern 1: Ingestion → Normalization → Bus (Agent B)

```python
from src.common.event_types import StockTrade, EventSource
from src.common.config import CONFIG

async def handle_massive_ws_message(msg: dict, queue: asyncio.Queue):
    """Normalize Massive WS message into StockTrade."""
    if msg['ev'] == 'T' and msg['sym'] == 'SPY':
        trade = StockTrade(
            ts_event_ns=msg['t'] * 1_000_000,  # ms → ns
            ts_recv_ns=time.time_ns(),
            source=EventSource.MASSIVE_WS,
            symbol=msg['sym'],
            price=msg['p'],
            size=msg['s'],
            exchange=msg.get('x'),
            conditions=msg.get('c', []),
            seq=msg.get('seq')
        )
        await queue.put(('stock_trade', trade))
```

### Pattern 2: State Update (Agent C)

```python
from src.common.event_types import StockTrade, StockQuote
from src.common.price_converter import PriceConverter

class MarketState:
    def __init__(self):
        self.converter = PriceConverter()
        self.trades_buffer = []
        self.quotes_buffer = []
        self.last_trade = None
        self.last_quote = None
    
    def update_stock_trade(self, trade: StockTrade):
        self.last_trade = trade
        self.trades_buffer.append(trade)
        self.converter.update_spy_price(trade.price)
    
    def update_stock_quote(self, quote: StockQuote):
        self.last_quote = quote
        self.quotes_buffer.append(quote)
        self.converter.update_spy_price((quote.bid_px + quote.ask_px) / 2)
    
    def get_spot(self) -> float:
        """SPY spot price (from SPY trades or ES conversion)."""
        if self.last_trade:
            return self.last_trade.price
        # Fallback to ES conversion if available
        ...
```

### Pattern 3: Engine Queries (Agents D/E/F/G)

```python
from src.common.config import CONFIG

def compute_tape_imbalance(
    market_state: MarketState,
    level_price: float,
    ts_now_ns: int
) -> float:
    """Compute buy/sell imbalance near level."""
    window_ns = int(CONFIG.W_t * 1e9)
    band = CONFIG.TAPE_BAND
    
    # Get trades in window near level
    trades = market_state.get_trades_in_window(
        ts_now_ns,
        window_ns,
        level_price - band,
        level_price + band
    )
    
    buy_vol = sum(t.size for t in trades if t.aggressor == Aggressor.BUY)
    sell_vol = sum(t.size for t in trades if t.aggressor == Aggressor.SELL)
    
    imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)
    return imbalance
```

### Pattern 4: Bronze Writer (Agent I)

```python
from src.common.schemas import StockTradeV1, SchemaRegistry
import pyarrow as pa
import pyarrow.parquet as pq

def write_bronze_batch(
    trades: List[StockTrade],
    output_path: str
):
    """Write a batch of normalized trades to Bronze Parquet."""
    # Convert to Pydantic models
    pydantic_trades = [
        StockTradeV1(
            ts_event_ns=t.ts_event_ns,
            ts_recv_ns=t.ts_recv_ns,
            source=t.source.value,
            symbol=t.symbol,
            price=t.price,
            size=t.size,
            exchange=t.exchange,
            conditions=t.conditions,
            seq=t.seq
        )
        for t in trades
    ]
    
    # Convert to Arrow Table
    arrow_schema = SchemaRegistry.get_arrow_schema('stocks.trades.v1')
    table = pydantic_to_arrow_table(pydantic_trades, arrow_schema)
    
    # Write Parquet with compression
    pq.write_table(
        table,
        output_path,
        compression='ZSTD',
        compression_level=3
    )
```

### Pattern 5: Gold Writer (Agent I)

```python
from src.common.schemas import LevelSignalV1

def write_gold_signals(
    signals: List[LevelSignal],  # From score_engine
    output_path: str
):
    """Write level signals to Gold Parquet."""
    pydantic_signals = [
        LevelSignalV1(
            ts_event_ns=s.ts_event_ns,
            underlying=s.underlying,
            spot=s.spot,
            bid=s.bid,
            ask=s.ask,
            level_id=s.level_id,
            level_kind=s.level_kind.value,
            level_price=s.level_price,
            direction=s.direction.value,
            distance=s.distance,
            break_score_raw=s.break_score_raw,
            break_score_smooth=s.break_score_smooth,
            signal=s.signal.value,
            confidence=s.confidence.value,
            # Flatten barrier metrics
            barrier_state=s.barrier.state.value,
            barrier_delta_liq=s.barrier.delta_liq,
            # ... (continue flattening)
        )
        for s in signals
    ]
    
    arrow_schema = SchemaRegistry.get_arrow_schema('levels.signals.v1')
    table = pydantic_to_arrow_table(pydantic_signals, arrow_schema)
    
    pq.write_table(table, output_path, compression='ZSTD')
```

---

## Critical Design Decisions (Why It's Built This Way)

### 1. Why Separate `event_types.py` from `schemas/`?

**Reasoning**:
- `event_types.py`: Lightweight dataclasses for **runtime message passing** (in-memory bus)
- `schemas/`: Heavy Pydantic + PyArrow for **storage validation** and **Parquet writing**
- Separation allows fast event routing without Pydantic overhead in hot path
- Converge at storage boundary (Bronze writer converts dataclass → Pydantic → Arrow)

### 2. Why Dynamic ES/SPY Conversion Instead of Static 10.0?

**Reasoning**:
- ES/SPY ratio varies 9.98–10.02 intraday due to dividends, interest rates, fair value
- Static ratio would introduce systematic pricing error (0.1–0.2% drift)
- Dynamic ratio ensures accurate level alignment for strike-based trading

### 3. Why Flatten Gold Schema Instead of Nested JSON?

**Reasoning**:
- Parquet columnar format optimized for **flat schemas**
- Nested JSON in Parquet loses compression efficiency + requires complex readers
- Flattening enables direct Pandas/Polars loading: `df['barrier_state']` vs `df['barrier']['state']`
- ML frameworks (scikit-learn, XGBoost) expect flat feature vectors

### 4. Why EWMA Smoothing in Config Instead of Kalman Filters?

**Reasoning** (PLAN.md §5.6):
- EWMA is **parameter-free** (only τ half-life) and **deterministic**
- Kalman requires process/observation noise estimation → calibration burden
- v1 principle: "physics + math, no hindsight calibration"
- EWMA with τ=2–5s provides sufficient smoothing for 250ms snap ticks

### 5. Why Run Manifests Instead of Git Tags?

**Reasoning**:
- Git tags don't capture **runtime config** (only code version)
- Run manifests capture **exact parameter values** used for each ingestion session
- Enables "replay with identical config" even if code has diverged
- Critical for ML: "which exact config produced this training dataset?"

---

## Testing Strategy

### Unit Tests

```bash
cd backend
uv run pytest tests/test_price_converter.py    # 16 tests
uv run pytest tests/test_schemas.py            # 31 tests
uv run pytest tests/test_run_manifest_manager.py  # 24 tests
```

### Integration Tests

```python
# Test event normalization + schema validation
from src.common.event_types import StockTrade, EventSource
from src.common.schemas import StockTradeV1

def test_event_to_schema_conversion():
    # Create runtime event
    event = StockTrade(
        ts_event_ns=1734567890000000000,
        ts_recv_ns=1734567890001000000,
        source=EventSource.MASSIVE_WS,
        symbol='SPY',
        price=687.50,
        size=100
    )
    
    # Convert to Pydantic schema
    schema_model = StockTradeV1(
        ts_event_ns=event.ts_event_ns,
        ts_recv_ns=event.ts_recv_ns,
        source=event.source.value,
        symbol=event.symbol,
        price=event.price,
        size=event.size
    )
    
    # Validate
    assert schema_model.price == 687.50
    assert schema_model.size == 100
```

---

## Evolution Path (Phase 1 → Phase 3)

### Phase 1 (Current): Local M4, In-Process Bus ✅

**Status**: COMPLETE
- `asyncio.Queue` for message passing
- Local filesystem for Bronze/Silver/Gold Parquet
- Single-process deployment
- `event_types.py`, `config.py`, `price_converter.py` in active use
- `run_manifest_manager.py` tracks runs
- `schemas/` validates all storage

### Phase 2: Single-Machine Server

**Changes to `common`**:
- Activate `bus.py` (NATS JetStream)
- Add `config.py` hot-reload mechanism
- Add metrics/logging utilities to `common/`

### Phase 3: Colocation / Multi-Node

**Changes to `common`**:
- Replace NATS with Redpanda/Kafka in `bus.py`
- Add Confluent Schema Registry integration to `schemas/base.py`
- Add distributed tracing headers to event envelope
- Add object store (S3/MinIO) config to `config.py`

**Backward Compatibility**:
- `event_types.py` dataclasses remain stable (add fields, never remove)
- `schemas/` versions bump for breaking changes
- Config keys can be added (never remove or rename existing)

---

## Common Pitfalls (For AI Agents)

### Pitfall 1: Forgetting Unix Nanoseconds Conversion

**Wrong**:
```python
# Massive WS sends milliseconds
ts_event_ns = msg['t']  # ❌ This is milliseconds!
```

**Correct**:
```python
ts_event_ns = msg['t'] * 1_000_000  # ✅ Convert ms → ns
```

### Pitfall 2: Treating bid_sz/ask_sz as Round Lots

**Wrong**:
```python
# Massive WS quote
bid_contracts = quote['bs'] // 100  # ❌ bs is already in shares!
```

**Correct**:
```python
bid_shares = quote['bs']  # ✅ Already in shares (as of 2025-11-03)
```

### Pitfall 3: Not Using price_converter for ES/SPY Queries

**Wrong**:
```python
# Query ES depth at SPY level $687
es_depth = market_state.get_es_depth_at(687.0)  # ❌ Wrong scale!
```

**Correct**:
```python
es_level = market_state.converter.spy_to_es(687.0)  # ≈ 6870.0
es_depth = market_state.get_es_depth_at(es_level)  # ✅
```

### Pitfall 4: Modifying CONFIG During Runtime

**Wrong**:
```python
from src.common.config import CONFIG
CONFIG.W_b = 15.0  # ❌ Side effects! Run manifest won't match!
```

**Correct**:
```python
# Phase 1: Modify config.py and restart
# Phase 2: Use per-run config overrides with manifest tracking
```

### Pitfall 5: Creating Nested Dicts in Gold Schemas

**Wrong**:
```python
# Gold schema with nested barrier object
class LevelSignalV1(BaseEventModel):
    barrier: dict  # ❌ Inefficient for Parquet!
```

**Correct**:
```python
# Flattened fields
class LevelSignalV1(BaseEventModel):
    barrier_state: str
    barrier_delta_liq: float
    barrier_replenishment_ratio: float
    # ✅ Flat schema, efficient columnar storage
```

---

## Quick Reference

### Import Cheatsheet

```python
# Event types (runtime)
from src.common.event_types import (
    StockTrade, StockQuote, OptionTrade, FuturesTrade, MBP10,
    EventSource, Aggressor
)

# Config
from src.common.config import CONFIG

# Price conversion
from src.common.price_converter import PriceConverter

# Run manifests
from src.common.run_manifest_manager import (
    RunManifestManager, RunMode, RunStatus
)

# Schemas (storage validation)
from src.common.schemas import (
    StockTradeV1, StockQuoteV1, OptionTradeV1,
    GreeksSnapshotV1, FuturesTradeV1, MBP10V1,
    OptionTradeEnrichedV1, LevelSignalV1,
    SchemaRegistry
)
```

### Config Keys by Agent

| Agent | Key Config Keys |
|-------|----------------|
| Agent B (Ingestion) | `NATS_URL`, `REPLAY_SPEED` |
| Agent C (MarketState) | `LATENESS_BUFFER_MS` |
| Agent D (Barrier/Tape) | `W_b`, `W_t`, `W_v`, `R_vac`, `R_wall`, `F_thresh`, `TAPE_BAND`, `SWEEP_*` |
| Agent E (Fuel) | `W_g`, `W_wall`, `FUEL_STRIKE_RANGE` |
| Agent F (Levels) | `MONITOR_BAND`, `STRIKE_RANGE`, `ROUND_LEVELS_SPACING`, `VWAP_ENABLED` |
| Agent G (Scoring) | `w_L`, `w_H`, `w_T`, `BREAK_SCORE_THRESHOLD`, `REJECT_SCORE_THRESHOLD`, `tau_*` |
| Agent I (Storage) | `DATA_ROOT`, `MICRO_BATCH_INTERVAL_S`, `S3_*` |

---

## References

- **PLAN.md §2.1**: Message envelope specification
- **PLAN.md §2.2**: Bronze/Silver/Gold tier definitions
- **PLAN.md §2.4**: Dataset schemas (minimum required columns)
- **PLAN.md §9**: Configuration parameters
- **PLAN.md §11**: Vendor contracts (Massive, Databento)
- **PLAN.md §12**: Parallel agent assignments

---

## Change Log

| Date | Agent | Change |
|------|-------|--------|
| 2025-12-XX | Agent A | Initial implementation of `event_types.py`, `config.py`, `schemas/` |
| 2025-12-XX | Agent A | Added `price_converter.py` with dynamic ratio support |
| 2025-12-XX | Agent A | Added `run_manifest_manager.py` for Phase 1 hygiene |
| 2025-12-XX | Agent A | Completed 31 schema tests + 24 manifest tests |

---

**End of Technical Reference**

For questions or clarifications, refer to PLAN.md or the test suites in `backend/tests/`.

