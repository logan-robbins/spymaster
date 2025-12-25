# System Components Map

**Project**: Spymaster (SPY 0DTE Break/Bounce Physics Engine)  
**Audience**: Engineering Team & AI Coding Agents  
**Purpose**: Component architecture with interface contracts for parallel development

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
│  Polygon WebSocket (SPY)  │  Databento DBN (ES Futures)             │
└────────────────┬──────────┴──────────────┬──────────────────────────┘
                 │                         │
                 ▼                         ▼
         ┌────────────────────────────────────────┐
         │         INGESTOR SERVICE               │
         │  Normalization & Event Publishing      │
         └──────────────┬─────────────────────────┘
                        │ NATS JetStream (market.*)
                        │
         ┌──────────────┼─────────────────────────┐
         │              │                         │
         ▼              ▼                         ▼
    ┌────────┐    ┌─────────┐              ┌─────────┐
    │  CORE  │    │  LAKE   │              │ GATEWAY │
    │Service │    │ Service │              │ Service │
    └────┬───┘    └────┬────┘              └────┬────┘
         │             │                         │
         │             │                         ▼
         │             ▼                    WebSocket
         │        Parquet Files           (Frontend)
         │       (Bronze/Silver/Gold)
         │
         └──> NATS (levels.signals) ──> GATEWAY ──> Frontend
```

---

## Component Specifications

### 1. Common Module

**Location**: `backend/src/common/`  
**Role**: Foundational infrastructure (schemas, config, event types)  
**Interface**: [backend/src/common/INTERFACES.md](backend/src/common/INTERFACES.md)

**Key Responsibilities**:
- Event dataclass definitions (StockTrade, OptionTrade, FuturesTrade, MBP10, etc.)
- Configuration singleton (CONFIG with all tunable parameters)
- Price conversion (ES ↔ SPY)
- Storage schemas (Pydantic + PyArrow for Bronze/Silver/Gold)
- NATS bus wrapper (pub/sub messaging)

**Dependencies**: None (zero dependencies on other backend modules)

**Consumers**: All other backend modules

---

### 2. Ingestor Service

**Location**: `backend/src/ingestor/`  
**Role**: Data ingestion and normalization  
**Interface**: [backend/src/ingestor/INTERFACES.md](backend/src/ingestor/INTERFACES.md)

**Key Responsibilities**:
- Live feed ingestion (Polygon WebSocket for SPY stocks + options)
- Historical replay (Databento DBN files for ES futures)
- Event normalization to canonical types
- NATS publishing (market.* subjects)
- Dynamic strike management (option subscriptions follow SPY price)

**Inputs**:
- Polygon WebSocket: SPY trades/quotes, SPY option trades
- Databento DBN files: ES futures trades + MBP-10 depth

**Outputs**:
- NATS subjects: `market.stocks.trades`, `market.stocks.quotes`, `market.options.trades`, `market.futures.trades`, `market.futures.mbp10`

**Entry Points**:
- Live: `uv run python -m src.ingestor.main`
- Replay: `uv run python -m src.ingestor.replay_publisher`

---

### 3. Core Service

**Location**: `backend/src/core/`  
**Role**: Physics engines and signal generation  
**Interface**: [backend/src/core/INTERFACES.md](backend/src/core/INTERFACES.md)

**Key Responsibilities**:
- Maintain market state (ES MBP-10, trades, SPY option flow)
- Generate level universe (VWAP, strikes, walls, session high/low)
- Compute barrier physics (VACUUM, WALL, ABSORPTION, CONSUMED)
- Compute tape physics (imbalance, velocity, sweep detection)
- Compute fuel physics (dealer gamma, AMPLIFY vs DAMPEN)
- Composite scoring (break/bounce probability 0-100)
- Smoothing (EWMA for stable output)
- Publish level signals to NATS

**Inputs**:
- NATS subjects: `market.*` (all market data from Ingestor)

**Outputs**:
- NATS subject: `levels.signals` (level signals payload, ~250ms cadence)

**Key Engines**:
- `MarketState`: Central state store with ring buffers
- `BarrierEngine`: ES MBP-10 liquidity analysis
- `TapeEngine`: ES trade flow analysis
- `FuelEngine`: SPY option gamma analysis
- `ScoreEngine`: Composite break score (weighted sum)
- `LevelSignalService`: Orchestrator that publishes signals

**Entry Point**: Integrated into microservices architecture (runs as separate process)

---

### 4. Lake Service

**Location**: `backend/src/lake/`  
**Role**: Bronze/Silver/Gold data persistence  
**Interface**: [backend/src/lake/INTERFACES.md](backend/src/lake/INTERFACES.md)

**Key Responsibilities**:
- Bronze writer: NATS → append-only Parquet (raw, immutable)
- Gold writer: NATS levels.signals → derived analytics Parquet
- Silver compactor: Offline deduplication and sorting (Bronze → Silver)
- Storage backend: Local filesystem or S3/MinIO

**Inputs**:
- NATS subjects: `market.*` (for Bronze), `levels.signals` (for Gold)

**Outputs**:
- Parquet files:
  - Bronze: `bronze/{asset_class}/{schema}/{partition}/date=YYYY-MM-DD/hour=HH/*.parquet`
  - Silver: `silver/{asset_class}/{schema}/{partition}/date=YYYY-MM-DD/hour=HH/*.parquet`
  - Gold: `gold/levels/signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/*.parquet`

**Storage Tiers**:
- **Bronze**: Raw normalized events (at-least-once, append-only)
- **Silver**: Deduped and sorted (exactly-once via MD5 event_id)
- **Gold**: Derived features and signals (ML-ready, flattened schema)

**Entry Point**: `uv run python -m src.lake.main`

---

### 5. Gateway Service

**Location**: `backend/src/gateway/`  
**Role**: WebSocket relay (NATS → Frontend)  
**Interface**: [backend/src/gateway/INTERFACES.md](backend/src/gateway/INTERFACES.md)

**Key Responsibilities**:
- Subscribe to NATS `levels.signals`
- Cache latest payload
- Broadcast to WebSocket clients
- Health check endpoint

**Inputs**:
- NATS subject: `levels.signals`

**Outputs**:
- WebSocket: `ws://localhost:8000/ws/stream` (JSON frames, ~250ms cadence)
- HTTP: `GET /health` (health check)

**Key Characteristics**:
- Pure relay (zero compute logic)
- Immediate state delivery (cached payload sent on connect)
- Automatic client cleanup (failed connections removed)
- Durable NATS consumer (survives restarts)

**Entry Point**: `uv run python -m src.gateway.main`

---

### 6. ML Module

**Location**: `backend/src/ml/`  
**Role**: Model training and inference  
**Interface**: [backend/src/ml/INTERFACES.md](backend/src/ml/INTERFACES.md)

**Key Responsibilities**:
- Train boosted-tree models (tradeable, direction, strength, time-to-threshold)
- Build kNN retrieval index
- Train PatchTST sequence baseline
- Calibration evaluation
- Live viewport scoring (integrated into Core Service)

**Inputs**:
- Training data: Signals Parquet (from vectorized pipeline, via `features.json`)
- Live features: Engineered features from Core Service

**Outputs**:
- Model bundles: `data/ml/boosted_trees/*.joblib`
- Retrieval index: `data/ml/retrieval_index.joblib`
- PatchTST checkpoint: `patchtst_multitask.pt`
- Metadata: Train/val splits, metrics, feature names

**Key Scripts**:
- `boosted_tree_train.py`: Multi-head boosted trees
- `build_retrieval_index.py`: kNN index builder
- `sequence_dataset_builder.py`: OHLCV sequence dataset
- `patchtst_train.py`: Sequence model trainer

**Entry Points**:
- Training: `uv run python -m src.ml.boosted_tree_train --stage stage_b --ablation all`
- Retrieval: `uv run python -m src.ml.build_retrieval_index --stage stage_b`

---

### 7. Pipeline Module

**Location**: `backend/src/pipeline/`
**Role**: Offline feature engineering and signal generation for ML training
**Interface**: [backend/src/pipeline/README.md](backend/src/pipeline/README.md)

**Key Responsibilities**:
- Bronze → Gold data transformation (vectorized operations)
- Level universe generation (PM/OR/SMA/VWAP/Walls)
- Touch detection with monitor band filtering
- Physics feature computation (barrier, tape, fuel)
- Confluence level feature computation (hierarchical 1-10 scale)
- Outcome labeling (BREAK/BOUNCE with competing-risks timing)

**Inputs**:
- Bronze Parquet: ES trades, MBP-10, option trades
- Warmup data: 3 prior days (SMA), 7 prior days (relative volume)

**Outputs**:
- Gold Parquet: `gold/research/signals_vectorized.parquet`
- Schema defined in `backend/features.json`

**Key Components**:
- `VectorizedPipeline`: High-performance batch processing
- `compute_confluence_level_features`: Hierarchical setup quality scoring
- `generate_level_universe_vectorized`: Structural level detection

**Level Universe** (SPY-specific):
- PM_HIGH/PM_LOW: Pre-market high/low (04:00-09:30 ET)
- OR_HIGH/OR_LOW: Opening range (09:30-09:45 ET)
- SESSION_HIGH/SESSION_LOW: Running session extremes
- SMA_200/SMA_400: Moving averages on 2-min bars
- VWAP: Session volume-weighted average price
- CALL_WALL/PUT_WALL: Max gamma concentration strikes
- Note: ROUND/STRIKE levels disabled for SPY (duplicative with $1 strikes)

**Entry Point**: `uv run python -m src.pipeline.batch_process --start-date 2025-11-01`

---

### 8. Features Module

**Location**: `backend/src/features/`
**Role**: Context analysis and structural level identification
**Interface**: [backend/src/features/INTERFACES.md](backend/src/features/INTERFACES.md)

**Key Responsibilities**:
- Timing context detection (first 15 minutes, bars since open)
- Structural level identification (PM high/low, SMA-200/400)
- Level proximity detection

**Key Components**:
- `ContextEngine`: Caches PM/OR levels, SMA series, timing context

**Entry Point**: Used as library by Pipeline and Core modules

---

### 9. Frontend (Angular)

**Location**: `frontend/`  
**Role**: Real-time UI for SPY 0DTE break/bounce signals  
**Interface**: [frontend/INTERFACES.md](frontend/INTERFACES.md)

**Key Responsibilities**:
- Connect to Gateway WebSocket (`/ws/stream`)
- Display price ladder with level markers
- Visualize break/bounce strength meters
- Show physics attribution (barrier, tape, fuel)
- Render dealer mechanics (gamma velocity, exposure)
- Display confluence zones and signal timeline
- Options flow panel (strike grid + flow chart)
- Compute UI-specific derived metrics (strength scores, confluence detection)

**Inputs**:
- WebSocket: `ws://localhost:8000/ws/stream` (level signals payload)

**Outputs**:
- Visual UI (browser at `http://localhost:4200`)
- User interactions (level selection, hover states)

**Key Components**:
- `CommandCenterComponent`: Main layout (3-panel)
- `PriceLadderComponent`: Vertical price ladder with level markers
- `StrengthCockpitComponent`: Break/bounce meters + dealer mechanics
- `AttributionBarComponent`: Physics contribution breakdown
- `ConfluenceStackComponent`: Grouped level clusters
- `OptionsPanelComponent`: Strike grid + flow chart

**Key Services**:
- `DataStreamService`: WebSocket connection + parsing
- `LevelDerivedService`: Strength computation + confluence detection

**State Management**: RxJS BehaviorSubjects (no global state store)

**Entry Point**: `npm run start` (Angular dev server at `http://localhost:4200`)

---

## Data Flow Contracts

### Event Time Discipline

**All events carry**:
- `ts_event_ns`: Event time (from exchange) in Unix nanoseconds UTC
- `ts_recv_ns`: Receive time (by our system) in Unix nanoseconds UTC

**Conversion Rules**:
- Polygon: milliseconds → multiply by 1,000,000
- Databento: already nanoseconds → use directly

---

### NATS Subject Hierarchy

**Market Data** (Ingestor → Core/Lake):
- `market.stocks.trades`: SPY equity trades
- `market.stocks.quotes`: SPY equity quotes
- `market.options.trades`: SPY option trades
- `market.futures.trades`: ES futures trades
- `market.futures.mbp10`: ES MBP-10 depth updates

**Derived Signals** (Core → Lake/Gateway):
- `levels.signals`: Level signals payload (break/bounce physics)

**Stream Configuration**:
- Retention: 24 hours
- Storage: File-backed persistence
- Consumers: Durable (resume on restart)

---

### Price Conversion Protocol

**Levels are SPY dollars, liquidity is ES**:
1. User specifies level in SPY: `level_price = 687.0`
2. Engine converts to ES for queries: `es_level = 687.0 * 10 = 6870.0`
3. Query ES depth/trades at converted price
4. Convert results back to SPY for output

**Ratio**: ES ≈ SPY × 10 (dynamic ratio supported via `PriceConverter`)

---

## Configuration Contract

**Single Source of Truth**: `backend/src/common/config.py` (CONFIG singleton)

**Key Parameters**:
- Physics windows: `W_b=240s` (barrier/confirmation), `W_t=60s` (tape), `W_g=60s` (fuel)
- Monitoring bands: `MONITOR_BAND=0.25`, `TOUCH_BAND=0.10`
- Thresholds: `R_vac=0.3`, `R_wall=1.5`, `F_thresh=100`
- Score weights: `w_L=0.45`, `w_H=0.35`, `w_T=0.20`
- Smoothing: `tau_score=2.0s`, `tau_velocity=1.5s`
- Snap interval: `SNAP_INTERVAL_MS=250`
- Warmup: `SMA_WARMUP_DAYS=3`, `VOLUME_LOOKBACK_DAYS=7`
- Confluence: `SMA_PROXIMITY_THRESHOLD=0.005`, `WALL_PROXIMITY_DOLLARS=1.0`
- Relative volume: `REL_VOL_HIGH_THRESHOLD=1.3`, `REL_VOL_LOW_THRESHOLD=0.7`

**Access Pattern**:
```python
from src.common.config import CONFIG
barrier_window = CONFIG.W_b
monitor_band = CONFIG.MONITOR_BAND
```

---

## Deployment Architecture

### Docker Compose Services

```yaml
services:
  nats:          # Message bus (JetStream)
  minio:         # Object storage (S3-compatible)
  ingestor:      # Data ingestion (live or replay)
  core:          # Physics engines + signal generation
  lake:          # Bronze/Silver/Gold persistence
  gateway:       # WebSocket relay to frontend
```

**Startup Order**:
1. NATS + MinIO (infrastructure)
2. Ingestor → Core → Lake → Gateway (data flow)

**Health Checks**:
- NATS: `curl http://localhost:8222/healthz`
- Gateway: `curl http://localhost:8000/health`
- MinIO: `curl http://localhost:9000/minio/health/live`

---

## Development Workflow

### Running the Full Stack

```bash
# Start backend services
cd /Users/loganrobbins/research/qmachina/spymaster
docker-compose up -d

# Verify services
docker ps
curl http://localhost:8000/health

# Start frontend
cd frontend
npm run start
# Navigate to http://localhost:4200
```

### Running Individual Components (Local Dev)

```bash
# Terminal 1: Start NATS (required for all services)
docker-compose up nats -d

# Terminal 2: Start ingestor (replay mode)
cd backend
export REPLAY_SPEED=1.0
export REPLAY_DATE=2025-12-16
uv run python -m src.ingestor.replay_publisher

# Terminal 3: Start core service
uv run python -m src.core.main

# Terminal 4: Start gateway
uv run python -m src.gateway.main

# Terminal 5: Start frontend
cd ../frontend
npm run start
```

---

## Testing Strategy

### Component Testing

**Ingestor**:
- Test event normalization: `backend/tests/test_ingestor_*.py`
- Verify NATS publishing: `nats sub "market.>"`

**Core**:
- Engine unit tests: `backend/tests/test_barrier_engine.py`, etc.
- Integration tests: `backend/tests/test_core_service.py`
- Replay determinism: `backend/tests/test_replay_determinism.py`

**Lake**:
- Bronze/Silver/Gold writes: `backend/tests/test_lake_service.py`
- Silver compaction: `backend/tests/test_silver_compactor.py`

**Gateway**:
- WebSocket relay: `backend/tests/test_gateway_integration.py`
- Health check: `curl http://localhost:8000/health`

**Frontend**:
- Component tests: `frontend/src/app/**/*.spec.ts`
- E2E: Manual testing with live WebSocket connection

### End-to-End Testing

```bash
# 1. Start full stack (Docker Compose)
docker-compose up -d

# 2. Run replay for test date
docker-compose restart ingestor

# 3. Verify frontend receives data
# Navigate to http://localhost:4200
# Check browser console for WebSocket messages

# 4. Check data persistence
ls backend/data/lake/bronze/futures/trades/symbol=ES/date=2025-12-16/
ls backend/data/lake/gold/levels/signals/underlying=SPY/date=2025-12-16/
```

---

## Critical Invariants

1. **Event-time ordering**: All data sorted by `ts_event_ns`, not `ts_recv_ns`
2. **Price conversion**: Always use `PriceConverter`, never hardcode ES/SPY ratio
3. **NATS durability**: 24-hour retention ensures replay capability
4. **Schema versioning**: Never break backward compatibility without version bump
5. **Deterministic replay**: Same inputs + config → same outputs
6. **No hindsight**: Features computed from data before label anchor time (`t1`)

---

## Performance Targets

**Latency** (end-to-end):
- Polygon event → NATS publish: <15ms
- NATS → Core processing → NATS publish: <50ms
- NATS → Gateway → WebSocket: <5ms
- **Total: <70ms** (exchange → frontend)

**Throughput**:
- Ingestor: 10k+ events/sec
- Core: 4 Hz snap rate (250ms interval)
- Lake Bronze writes: 1000 events/batch (5s interval)
- Gateway: 100+ concurrent WebSocket clients

**Storage**:
- Bronze: ~5-10GB/day (SPY + ES data)
- Silver: ~4-8GB/day (after dedup)
- Gold: ~100-500MB/day (level signals)

---

## References

### Module Interfaces (Authoritative)
- Common: [backend/src/common/INTERFACES.md](backend/src/common/INTERFACES.md)
- Ingestor: [backend/src/ingestor/INTERFACES.md](backend/src/ingestor/INTERFACES.md)
- Core: [backend/src/core/INTERFACES.md](backend/src/core/INTERFACES.md)
- Lake: [backend/src/lake/INTERFACES.md](backend/src/lake/INTERFACES.md)
- Gateway: [backend/src/gateway/INTERFACES.md](backend/src/gateway/INTERFACES.md)
- ML: [backend/src/ml/INTERFACES.md](backend/src/ml/INTERFACES.md)
- Frontend: [frontend/INTERFACES.md](frontend/INTERFACES.md)

### Full Documentation
- Root: [README.md](README.md)
- Backend: [backend/README.md](backend/README.md)
- Frontend: [FRONTEND.md](FRONTEND.md)
- Feature Contract: [backend/features.json](backend/features.json)

### Module READMEs (Detailed Specs)
- Common: [backend/src/common/README.md](backend/src/common/README.md)
- Ingestor: [backend/src/ingestor/README.md](backend/src/ingestor/README.md)
- Core: [backend/src/core/README.md](backend/src/core/README.md)
- Lake: [backend/src/lake/README.md](backend/src/lake/README.md)
- Gateway: [backend/src/gateway/README.md](backend/src/gateway/README.md)
- ML: [backend/src/ml/README.md](backend/src/ml/README.md)
- Pipeline: [backend/src/pipeline/README.md](backend/src/pipeline/README.md)
- Features: [backend/src/features/README.md](backend/src/features/README.md)

---

**Version**: 1.1
**Last Updated**: 2025-12-24
**Status**: Active (Phase 3 architecture - confluence level feature)

