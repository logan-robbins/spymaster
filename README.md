# Spymaster: Real-Time Market Physics Engine

**Target Audience**: AI Coding Agent  
**Asset**: SPY (equity + 0DTE options)  
**Status**: Phase 2 Complete (NATS microservices architecture)

---

## Philosophy & Core Concept

**What We Are Building**: A real-time physics-based system that watches critical price levels and predicts whether they will **BREAK** (fail) or **REJECT** (hold) as price approaches them.

### The Problem

For any critical level (strike, round number, VWAP, gamma wall), we continuously answer:
- Will price **break through** this level and run?
- Will it **reject** and reverse away?
- In **either direction** (support tests from above, resistance tests from below)?

### The Insight: Market Physics

**Price cannot move unless dealers make it move.** We trace the fundamental physics:

1. **Barrier Physics (Liquidity)**: Is displayed liquidity at the level **evaporating** (vacuum = easy break) or **replenishing** (wall = likely reject)?
   - We watch ES futures MBP-10 (top 10 depth levels) to see order flow dynamics
   - We infer FILLED vs PULLED by comparing depth changes to passive volume
   - We classify states: VACUUM, WALL, ABSORPTION, CONSUMED, WEAK, NEUTRAL

2. **Tape Physics (Momentum)**: Is tape aggression **confirming** the direction into the level?
   - We measure buy/sell imbalance, price velocity, sweep detection
   - ES time-and-sales shows institutional footprints (large aggressive prints)

3. **Fuel Physics (Dealer Hedging)**: Will dealers **amplify** or **dampen** the move based on their gamma position?
   - When customers buy options → dealers sell gamma → must chase price moves = AMPLIFY
   - When customers sell options → dealers buy gamma → fade moves = DAMPEN
   - We track SPY option flow to estimate net dealer gamma at each strike

### What Makes This Different

- **NOT High-Frequency Trading**: We operate on 100-250ms snap ticks (not microseconds)
- **NOT Pattern Recognition**: We use mechanical physics rules, not trained ML models
- **NOT Retail Indicators**: We watch institutional dealer behavior and order book dynamics
- **Get in Before Retail**: We detect market structure changes (vacuum forming, dealers chasing) before retail sees the price move

### Core Visualization Concept

We are building a **market X-ray**:
- Show the liquidity landscape around critical levels in real-time
- Visualize dealer gamma regimes (where are they forced to chase? where will they fade?)
- Track tape aggression and momentum building into levels
- Provide clear BREAK/REJECT signals with confidence levels
- Display "runway" (distance to next obstacle after a break)

---

## System Architecture

### High-Level Pipeline

```
┌─────────────┐
│  INGESTOR   │ ← Polygon WebSocket (SPY options + quotes)
│   Service   │ ← Databento DBN files (ES futures MBP-10 + trades)
└──────┬──────┘
       │ Normalize to canonical events (ts_event_ns, ts_recv_ns, source)
       ▼
┌─────────────┐
│    NATS     │ ← JetStream message bus (24h retention, file-backed)
│ JetStream   │   Subjects: market.*, levels.signals
└──────┬──────┘
       │
       ├──────────────────────────────────┐
       │                                   │
       ▼                                   ▼
┌─────────────┐                    ┌─────────────┐
│    CORE     │ ← Market physics   │    LAKE     │ ← Storage
│   Service   │   engines          │   Service   │
└──────┬──────┘                    └─────────────┘
       │                                   │
       │ Compute level signals             │ Write Bronze/Silver/Gold
       │ (break scores, signals)           │ (Parquet, Hive partitions)
       ▼                                   ▼
┌─────────────┐                    ┌─────────────┐
│   GATEWAY   │ ← WebSocket relay  │  MinIO/S3   │ ← Object storage
│   Service   │                    │   Storage   │
└──────┬──────┘                    └─────────────┘
       │
       ▼
┌─────────────┐
│  FRONTEND   │ ← Angular dashboard (real-time visualization)
│  (Angular)  │
└─────────────┘
```

### Data Flow: Live Trading Session

1. **Ingestion** (`backend/src/ingestor/`):
   - Subscribe to Polygon WebSocket: SPY trades, SPY quotes, SPY option trades
   - Read Databento DBN files: ES futures trades + MBP-10 depth
   - Normalize vendor formats → canonical event types
   - Publish to NATS subjects: `market.stocks.*`, `market.options.*`, `market.futures.*`

2. **Storage** (`backend/src/lake/`):
   - Subscribe to all `market.*` subjects
   - Micro-batch events (1000 events or 5 seconds)
   - Write Bronze Parquet files (append-only, raw, replayable)
   - Offline: compact Bronze → Silver (deduplicate, sort by event time)

3. **Core Processing** (`backend/src/core/`):
   - Subscribe to all `market.*` subjects
   - Maintain live market state: ES MBP-10 ring buffer, ES trades, SPY option flows
   - Every 250ms (snap tick):
     - Generate level universe (strikes, rounds, VWAP, walls)
     - For each level near spot price:
       * **BarrierEngine**: compute liquidity state (VACUUM/WALL/etc)
       * **TapeEngine**: compute tape imbalance, velocity, sweep detection
       * **FuelEngine**: compute net dealer gamma, walls
       * **ScoreEngine**: combine into break score (0-100), classify signal (BREAK/REJECT/etc)
       * **Smoothing**: apply EWMA for stability
       * **RoomToRun**: compute runway to next obstacle
     - Publish to NATS subject: `levels.signals`

4. **Gateway Relay** (`backend/src/gateway/`):
   - Subscribe to `levels.signals` NATS subject
   - Broadcast to WebSocket clients at `/ws/stream`
   - Frontend receives JSON payload with all level signals

5. **Gold Analytics** (`backend/src/lake/`):
   - Subscribe to `levels.signals`
   - Flatten nested structure
   - Write Gold Parquet files (derived analytics, ML-ready features)

### Why This Architecture?

- **Event-time first**: All records carry `ts_event_ns` (UTC) for deterministic replay
- **Separation of concerns**: Ingestion, computation, storage, relay are independent services
- **Institutional hygiene**: Append-only storage, schema versioning, run manifests
- **Replay capability**: Bronze Parquet can be replayed to reproduce Gold analytics exactly
- **ML-friendly**: Columnar Parquet, typed schemas, Hive partitioning

---

## Module Descriptions

### 1. Common (`backend/src/common/`)

**Role**: Foundational infrastructure for all services

**Key Components**:
- **Event Types** (`event_types.py`): Canonical dataclasses for all market events
  - `StockTrade`, `StockQuote`, `OptionTrade`, `FuturesTrade`, `MBP10`
  - All events carry `ts_event_ns` (event time) and `ts_recv_ns` (receive time) in Unix nanoseconds
- **Config** (`config.py`): Single source of truth for all tunable parameters
  - Window sizes (W_b, W_t, W_g), thresholds (R_vac, R_wall), weights (w_L, w_H, w_T)
  - No trained calibration; all constants are mechanical/tunable
- **Schemas** (`schemas/`): Pydantic + PyArrow schemas for Bronze/Silver/Gold storage
  - Schema versioning for evolution tracking
  - Dual representation: runtime validation (Pydantic) + storage (PyArrow/Parquet)
- **Price Converter** (`price_converter.py`): ES ↔ SPY price conversion
  - Levels are SPY prices (for option strikes)
  - Liquidity is ES futures (superior MBP-10 visibility)
  - Dynamic ratio support (ES ≈ SPY × 10, adjusted for dividends/basis)
- **Run Manifests** (`run_manifest_manager.py`): Track run metadata (config snapshots, git commit, file outputs)

**📖 Full Documentation**: [`backend/src/common/README.md`](backend/src/common/README.md)

---

### 2. Ingestor (`backend/src/ingestor/`)

**Role**: The Source — normalize vendor feeds into canonical events

**Key Components**:
- **StreamIngestor** (`stream_ingestor.py`): Live Polygon WebSocket adapter
  - Subscribes to SPY equity (trades + quotes) and SPY options (trades)
  - Dynamic strike management: updates option subscriptions as SPY moves
  - Normalizes vendor wire format → `StockTrade`, `StockQuote`, `OptionTrade`
- **DBNIngestor** (`dbn_ingestor.py`): Databento DBN file reader (ES futures)
  - Streaming iterators (no full file load for 10GB+ MBP-10 files)
  - Converts DBN records → `FuturesTrade`, `MBP10`
- **ReplayPublisher** (`replay_publisher.py`): Historical replay with speed control
  - Merges trades + MBP-10 into single event-time-ordered stream
  - Replay speed: 0x (fast), 1x (realtime), 2x (2x speed)
  - Publishes to same NATS subjects as live feeds (downstream services are replay-agnostic)

**Critical Contracts**:
- All events published to NATS subjects: `market.stocks.trades`, `market.stocks.quotes`, `market.options.trades`, `market.futures.trades`, `market.futures.mbp10`
- Timestamps: vendor ms → convert to Unix ns (multiply by 1,000,000)

**📖 Full Documentation**: [`backend/src/ingestor/README.md`](backend/src/ingestor/README.md)

---

### 3. Core (`backend/src/core/`)

**Role**: The Brain — market physics engines and signal computation

**Key Components**:

#### MarketState (`market_state.py`)
- Central state store for all market data
- ES MBP-10 ring buffer (60-120s window)
- ES trades ring buffer
- SPY option flow aggregates (net dealer gamma by strike)
- Price converter for ES ↔ SPY queries
- Accessors: `get_spot()`, `get_bid_ask()`, `get_vwap()`, `get_es_trades_near_level()`

#### Physics Engines

**BarrierEngine** (`barrier_engine.py`): Liquidity physics
- Tracks ES MBP-10 depth changes in zone around level (±2 ticks)
- Infers FILLED vs PULLED by comparing depth lost to passive volume
- Classifies state: VACUUM (liquidity pulled), WALL (replenishing), ABSORPTION, CONSUMED, WEAK, NEUTRAL
- **Critical**: Event-driven ingestion (process every MBP-10 update) to avoid "churn blindness"

**TapeEngine** (`tape_engine.py`): Momentum physics
- Computes buy/sell imbalance in price band around level
- Calculates price velocity (slope over time)
- Detects sweeps (clustered aggressive prints, large notional, consistent direction)
- Uses ES trades (institutional footprints, better than SPY L1)

**FuelEngine** (`fuel_engine.py`): Dealer gamma physics
- Tracks SPY option flow → estimates net dealer gamma
- Customer buys option → dealer SHORT gamma → must chase → AMPLIFY
- Customer sells option → dealer LONG gamma → fade moves → DAMPEN
- Identifies gamma walls (strikes with highest customer demand)

**ScoreEngine** (`score_engine.py`): Composite scoring
- Combines barrier, tape, fuel into break score (0-100)
- Component weights: w_L=0.45 (liquidity), w_H=0.35 (hedge), w_T=0.20 (tape)
- Trigger state machine with hysteresis (score must sustain >80 for 3s → BREAK signal)

**Smoothing** (`smoothing.py`): EWMA smoothers
- Apply exponential weighted moving average to raw scores/metrics
- Half-life parameters (tau) from CONFIG
- Prevents flicker from microstructure noise

**LevelSignalService** (`level_signal_service.py`): Orchestrator
- Generates level universe (VWAP, strikes, rounds, walls)
- Calls all engines per level
- Builds WebSocket payload (§6.4 of PLAN.md)
- Publishes to NATS subject: `levels.signals`

**📖 Full Documentation**: [`backend/src/core/README.md`](backend/src/core/README.md)

---

### 4. Lake (`backend/src/lake/`)

**Role**: Institutional-grade data persistence (Bronze/Silver/Gold lakehouse)

**Key Components**:

**BronzeWriter** (`bronze_writer.py`):
- Subscribes to all `market.*` NATS subjects
- Micro-batches events (1000 events or 5 seconds)
- Writes append-only Parquet files
- Hive partitioning: `symbol=SPY/date=YYYY-MM-DD/hour=HH/`
- ZSTD compression level 3

**SilverCompactor** (`silver_compactor.py`):
- Offline batch job: Bronze → Silver
- Deduplicates by MD5(source, ts_event_ns, symbol, price, size, seq)
- Sorts by event time
- Deterministic: same Bronze input → identical Silver output

**GoldWriter** (`gold_writer.py`):
- Subscribes to `levels.signals` NATS subject
- Flattens nested level signal payload into flat Parquet schema
- Target: ML-ready feature tables
- Stores derived analytics (break scores, barrier metrics, tape metrics, fuel metrics)

**Storage Tiers**:
- **Bronze**: Raw, append-only, full replay capability
- **Silver**: Clean, deduped, sorted, join-enriched
- **Gold**: Derived features, ML-ready, flattened schemas

**📖 Full Documentation**: [`backend/src/lake/README.md`](backend/src/lake/README.md)

---

### 5. Gateway (`backend/src/gateway/`)

**Role**: The Interface — NATS → WebSocket relay (no compute logic)

**Key Components**:
- **SocketBroadcaster** (`socket_broadcaster.py`): Pure relay microservice
  - Subscribes to `levels.signals` NATS subject
  - Caches latest payload
  - Broadcasts to all WebSocket clients at `/ws/stream`
- **FastAPI app** (`main.py`): HTTP server with lifespan management
  - WebSocket endpoint: `ws://localhost:8000/ws/stream`
  - Health check: `GET /health`

**Design Principle**: Zero business logic. All signal computation happens in Core Service.

**📖 Full Documentation**: [`backend/src/gateway/README.md`](backend/src/gateway/README.md)

---

### 6. Frontend (`frontend/`)

**Role**: Real-time visualization dashboard (Angular)

**Key Components**:
- **Level Strip** (`level-strip/`): Horizontal bar showing levels near current price
- **Strike Grid** (`strike-grid/`): Table view of option strikes with gamma metrics
- **Flow Wave** (`flow-wave/`): Time-series chart of option flow aggregates
- **Level Table** (`level-table/`): Detailed metrics per level (barrier/tape/fuel/score)

**Data Flow**:
- WebSocket connection to Gateway: `ws://localhost:8000/ws/stream`
- Receives JSON payload every 250ms (snap tick)
- Updates Angular components via RxJS observables

**Technology Stack**: Angular 19, Tailwind CSS, Chart.js

---

## Key Concepts for AI Agents

### 1. Time Discipline

**Two timestamps on every event**:
- `ts_event_ns`: Event time from vendor/exchange (Unix nanoseconds UTC)
- `ts_recv_ns`: Receive time by our system (Unix nanoseconds UTC)

**Conversion rules**:
- Polygon sends milliseconds → multiply by 1,000,000 to get nanoseconds
- Databento sends nanoseconds → use directly
- Current time: `time.time_ns()` in Python

**Why nanoseconds?** Standardize on finest granularity (Databento uses ns). Convert to ms for frontend display.

### 2. Price Conversion (ES ↔ SPY)

**Problem**: Levels are SPY prices (option strikes), but liquidity is ES futures (better MBP-10 visibility).

**Solution**: Dynamic price converter (ES ≈ SPY × 10)
- SPY level $687.00 → query ES depth at ~6870.0
- Ratio adjusts for dividends, interest rate differentials, fair value basis
- All outputs in SPY terms for consistency with option strikes

**Usage**:
```python
es_level = market_state.converter.spy_to_es(spy_level)
depth = market_state.get_es_depth_at(es_level)
defending_quote_spy = market_state.converter.es_to_spy(defending_quote_es)
```

### 3. Event-Driven vs Snap Tick

**Ingestion**: Event-driven (process every MBP-10 update, every trade immediately)
- Avoids "churn blindness" (missing depth add/cancel that net to zero)

**Scoring/Publishing**: Snap tick (100-250ms fixed cadence)
- Core Service computes level signals every 250ms
- Uses windowed queries over event buffers (e.g., "last 10 seconds of MBP-10 updates")

### 4. No Hindsight Calibration

**v1 Design Principle**: "Physics + math, no trained calibration"
- All thresholds are mechanical constants (R_vac, R_wall, F_thresh)
- Tunable via CONFIG, not learned from historical data
- Enables deterministic replay (same inputs + config → same outputs)

### 5. Deterministic Replay

**Critical for backtesting**:
- Given same Bronze Parquet input + same CONFIG → Gold output is byte-for-byte identical (within floating-point rounding policy)
- Achieved via:
  - Event-time ordering (sort by `ts_event_ns`)
  - Deterministic tie-breaking (use `ts_recv_ns`, then `seq`)
  - No random seeds, no system time in computation
  - Fixed config (do not mutate CONFIG during replay)

---

## Development Workflow

### Setup

```bash
# Clone repository
cd /Users/loganrobbins/research/qmachina/spymaster

# Install Python dependencies (backend)
cd backend
uv sync

# Install Node dependencies (frontend)
cd ../frontend
npm install
```

### Running Services (Docker Compose)

```bash
# Start all services
docker compose up

# Or start individually
docker compose up nats      # NATS JetStream
docker compose up ingestor  # Data ingestion
docker compose up core      # Physics engines
docker compose up lake      # Storage
docker compose up gateway   # WebSocket relay
docker compose up frontend  # Angular dashboard
```

### Local Development (without Docker)

```bash
# Terminal 1: NATS JetStream
docker compose up nats

# Terminal 2: Ingestor (live feeds)
export POLYGON_API_KEY="your_key"
cd backend
uv run python -m src.ingestor.main

# Terminal 3: Core Service
cd backend
uv run python -m src.core.main

# Terminal 4: Lake Service
cd backend
uv run python -m src.lake.main

# Terminal 5: Gateway
cd backend
uv run python -m src.gateway.main

# Terminal 6: Frontend
cd frontend
npm start
```

### Testing

```bash
# Backend unit tests
cd backend
uv run pytest tests/ -v

# Specific test suites
uv run pytest tests/test_barrier_engine.py -v
uv run pytest tests/test_tape_engine.py -v
uv run pytest tests/test_fuel_engine.py -v
uv run pytest tests/test_silver_compactor.py -v

# Replay determinism test
uv run pytest tests/test_replay_determinism.py -v

# E2E integration test
uv run pytest tests/test_e2e_replay.py -v
```

### Replay Historical Data

```bash
# Replay specific date at 1x realtime speed
export REPLAY_SPEED=1.0
export REPLAY_DATE=2025-12-16
cd backend
uv run python -m src.ingestor.replay_publisher

# Fast replay (no delays)
export REPLAY_SPEED=0
uv run python -m src.ingestor.replay_publisher
```

---

## Configuration

All tunable parameters live in `backend/src/common/config.py` as a `Config` dataclass.

**Key Parameters**:
- **Window sizes**: W_b (barrier, 10s), W_t (tape, 5s), W_g (fuel, 60s), W_v (velocity, 3s)
- **Thresholds**: R_vac (0.3), R_wall (1.5), F_thresh (100 ES contracts)
- **Weights**: w_L (0.45), w_H (0.35), w_T (0.20)
- **Trigger thresholds**: BREAK_SCORE_THRESHOLD (80), REJECT_SCORE_THRESHOLD (20), TRIGGER_HOLD_TIME (3s)
- **Smoothing**: tau_score (2.0s), tau_velocity (1.5s), tau_delta_liq (3.0s)
- **Snap cadence**: SNAP_INTERVAL_MS (250ms)
- **Level universe**: MONITOR_BAND (0.50 SPY dollars), STRIKE_RANGE (5.0 SPY dollars)

**Access in code**:
```python
from src.common.config import CONFIG

window_seconds = CONFIG.W_b
snap_interval = CONFIG.SNAP_INTERVAL_MS
```

---

## Data Flow Examples

### Example 1: VACUUM Forming (Easy Break)

1. **T-10s**: SPY approaching support level $687.00 from above (spot = $687.42)
2. **T-5s**: ES MBP-10 shows 1000 contracts at bid 6870.0 (defending support)
3. **T-3s**: Depth drops to 100 contracts (900 contracts pulled, not filled)
4. **BarrierEngine**: Classifies VACUUM (replenishment_ratio < 0.3, delta_liq < -100)
5. **TapeEngine**: Detects sell sweep (320k shares sold in 100ms cluster)
6. **FuelEngine**: Net dealer gamma -185k (dealers SHORT gamma → must chase down)
7. **ScoreEngine**: Combines to break_score = 88 (LIQUIDITY:100, HEDGE:100, TAPE:100)
8. **Smoothing**: Smooth score = 81 (EWMA with tau=2.0s)
9. **Trigger**: Score sustained >80 for 3s → signal = BREAK
10. **Output**: Frontend displays red "BREAK IMMINENT" with confidence HIGH

### Example 2: WALL Holding (Reject)

1. **T-10s**: SPY approaching resistance level $690.00 from below (spot = $689.58)
2. **T-5s**: ES MBP-10 shows 500 contracts at ask 6900.0
3. **T-3s**: Price hits $689.80, aggressive buying (150k shares lifted)
4. **T-1s**: Depth consumed to 100 contracts, but replenishes to 800 contracts immediately
5. **BarrierEngine**: Classifies WALL (replenishment_ratio > 1.5, delta_liq > +100)
6. **TapeEngine**: Buy imbalance +0.45, but velocity slowing (slope → 0)
7. **FuelEngine**: Net dealer gamma +120k (dealers LONG gamma → will fade the move)
8. **ScoreEngine**: break_score = 25 (LIQUIDITY:0, HEDGE:0, TAPE:50)
9. **Trigger**: Score <20 sustained while distance <0.05 → signal = REJECT
10. **Output**: Frontend displays green "REJECT" with confidence HIGH

---

## Critical Invariants

1. **Event-time ordering**: All Parquet files sorted by `ts_event_ns`
2. **Append-only Bronze**: Never mutate Bronze files (archive/move only)
3. **Silver is derived**: Always regeneratable from Bronze via deterministic dedup + sort
4. **No data loss**: NATS JetStream 24h retention ensures no events dropped during service restarts
5. **Schema stability**: Bronze schemas match PLAN.md §2.4 exactly; changes require version bump
6. **Compression**: ZSTD level 3 for all Parquet tiers
7. **Partition boundaries**: Date/hour partitions aligned to UTC (not market hours)

---

## References

- **PLAN.md**: Canonical system design document (architecture, message contracts, engine specifications)
- **Module READMEs**: Detailed technical documentation for each service
  - [`backend/src/common/README.md`](backend/src/common/README.md)
  - [`backend/src/ingestor/README.md`](backend/src/ingestor/README.md)
  - [`backend/src/core/README.md`](backend/src/core/README.md)
  - [`backend/src/lake/README.md`](backend/src/lake/README.md)
  - [`backend/src/gateway/README.md`](backend/src/gateway/README.md)
- **Tests**: Usage examples and integration patterns in `backend/tests/`

---

## Project Status

**Phase 2 Complete** (NATS microservices architecture):
- ✅ Common infrastructure (event types, schemas, config, price converter)
- ✅ Ingestor service (Polygon WebSocket + Databento DBN replay)
- ✅ Core service (barrier, tape, fuel, scoring engines)
- ✅ Lake service (Bronze/Silver/Gold Parquet storage)
- ✅ Gateway service (NATS → WebSocket relay)
- ✅ Frontend dashboard (Angular real-time visualization)
- ✅ 86 unit tests, replay determinism verified
- ✅ Docker Compose orchestration

**Next Steps** (Phase 3):
- Colocation deployment (NATS cluster, high-performance object store)
- Apache Iceberg metadata layer (ACID + schema evolution)
- Production monitoring (Prometheus + Grafana)
- Backtesting framework (strategy validation against Gold analytics)

---

## Contact

**Project**: Spymaster Physics Engine  
**Asset**: SPY (equity + 0DTE options)  
**Philosophy**: Watch the dealers, trace the physics, get in before retail  
**Not**: High-frequency trading, pattern recognition, retail indicators

For questions, consult the module READMEs or test suites for usage examples.

