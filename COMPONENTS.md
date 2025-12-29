# System Components Map

**Project**: Spymaster (ES Futures + ES 0DTE Options Break/Bounce Physics Engine)  
**Audience**: Engineering Team & AI Coding Agents  
**Purpose**: Component architecture with interface contracts for parallel development

---

## Architecture (ES Futures + ES Options)

**Perfect Alignment Strategy**:
- **Spot + Liquidity**: ES futures (trades + MBP-10)
- **Gamma Exposure**: ES 0DTE options (same underlying!)
- **Venue**: CME Globex (GLBX.MDP3 dataset)
- **Conversion**: NONE - ES = ES (zero basis spread)

**Critical Advantages over SPY**:
- No equity stock trades needed
- Same underlying instrument (E-mini S&P 500)
- Same venue, same participants
- Zero latency mismatch
- Zero conversion error

**ES 0DTE Specs** (validated from real data):
- Strike spacing: 25 points (ATM dominant)
- Modeling threshold: 3 strikes = 75 points
- Time window: 09:30-13:30 ET (first 4 hours)
- Level types: 4 only (PM/OR/SMA200/SMA400)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
│  Databento GLBX.MDP3 (ES Futures) │ Databento GLBX.MDP3 (ES Options)│
│  Trades + MBP-10 (front-month)    │ Trades + NBBO (front-month)     │
└────────────────┬──────────────────┴─────────────────────────────────┘
                 │                         
                 ▼                         
         ┌────────────────────────────────────────┐
         │         INGESTOR SERVICE               │
         │  Normalization & Event Publishing      │
         │  (Front-month filtering enforced)      │
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
- Storage schemas (Pydantic + PyArrow for Bronze/Silver/Gold - **enforced**)
  - Bronze: `FuturesTradeV1`, `MBP10V1`, `OptionTradeV1` (9 schemas)
  - Silver: `SilverFeaturesESPipelineV1` (182 columns, validated in pipeline)
  - Gold: `GoldTrainingESPipelineV1`, `LevelSignalV1`
- NATS bus wrapper (pub/sub messaging)

**Dependencies**: None (zero dependencies on other backend modules)

**Consumers**: All other backend modules

---

### 2. Ingestor Service

**Location**: `backend/src/ingestor/`  
**Role**: Data ingestion and normalization  
**Interface**: [backend/src/ingestor/INTERFACES.md](backend/src/ingestor/INTERFACES.md)

**Key Responsibilities**:
- Historical replay (Databento DBN files for ES futures + ES options)
- Front-month contract filtering (ES futures AND ES options)
- Event normalization to canonical types
- NATS publishing (market.* subjects)
- 0DTE filtering for ES options (exp_date == session date)

**Inputs**:
- Databento GLBX.MDP3: ES futures trades + MBP-10 depth (front-month only)
- Databento GLBX.MDP3: ES options trades + NBBO (front-month contract only, 0DTE)

**Outputs**:
- NATS subjects: `market.futures.trades`, `market.futures.mbp10`, `market.options.trades`, `market.options.nbbo`

**Entry Points**:
- Live: `uv run python -m src.ingestor.main`
- Replay: `uv run python -m src.ingestor.replay_publisher`

---

### 3. Core Service

**Location**: `backend/src/core/`  
**Role**: Physics engines and signal generation  
**Interface**: [backend/src/core/INTERFACES.md](backend/src/core/INTERFACES.md)

**Key Responsibilities**:
- Maintain market state (ES MBP-10, trades, ES option flow)
- Generate level universe (6 level kinds: PM/OR high/low + SMA_200/400)
- Compute barrier physics (VACUUM, WALL, ABSORPTION from ES depth)
- Compute tape physics (ES trade imbalance, velocity, sweep detection)
- Compute fuel physics (ES 0DTE option dealer gamma, AMPLIFY vs DAMPEN)
- Compute kinematics (velocity, acceleration, jerk in level frame)
- Compute OFI (integrated order flow imbalance from MBP-10)
- Composite scoring (break/bounce probability 0-100)
- Smoothing (EWMA for stable output)
- Optional ML viewport scoring (tradeability, direction, strength predictions)
- Publish level signals + viewport to NATS

**Inputs**:
- NATS subjects: `market.*` (all market data from Ingestor)

**Outputs**:
- NATS subject: `levels.signals` (level signals payload, ~250ms cadence)

**Key Engines**:
- `MarketState`: Central state store with ring buffers (ES futures + ES options)
- `BarrierEngine`: ES MBP-10 liquidity analysis (VACUUM, WALL, ABSORPTION states)
- `TapeEngine`: ES trade flow analysis (imbalance, velocity, sweeps)
- `FuelEngine`: ES option gamma analysis (dealer hedging dynamics)
- `KinematicsEngine`: Level-frame velocity, acceleration, jerk (multi-window)
- `OFIEngine`: Integrated order flow imbalance (L2 pressure proxy)
- `LevelSignalService`: Orchestrator that publishes signals

**Entry Point**: Integrated into microservices architecture (runs as separate process)

---

### 4. Lake Service

**Location**: `backend/src/lake/`  
**Role**: Bronze/Silver/Gold data persistence  
**Interface**: [backend/src/lake/INTERFACES.md](backend/src/lake/INTERFACES.md)

**Key Responsibilities**:
- Bronze writer: NATS → append-only Parquet (raw, immutable)
- Gold writer: NATS levels.signals → streaming signals Parquet
- Silver feature builder: Feature engineering (Bronze → Silver using es_pipeline)
- Gold curator: Promote Silver to Gold training dataset
- Storage backend: Local filesystem or S3/MinIO

**Inputs**:
- NATS subjects: `market.*` (for Bronze), `levels.signals` (for Gold)

**Outputs**:
- Parquet files:
  - Bronze: `bronze/{asset_class}/{schema}/{partition}/date=YYYY-MM-DD/hour=HH/*.parquet`
    - `futures/trades/symbol=ES/` (front-month only)
    - `futures/mbp10/symbol=ES/` (front-month only)
    - `options/trades/underlying=ES/` (front-month contract, 0DTE)
    - `options/nbbo/underlying=ES/` (front-month contract, 0DTE)
  - Silver: `silver/features/es_pipeline/` (ES system features)
  - Gold: `gold/levels/signals/underlying=ES/date=YYYY-MM-DD/hour=HH/*.parquet`

**Storage Tiers** (Medallion Architecture):
- **Bronze**: Raw normalized events (at-least-once, append-only, immutable)
  - ES futures + ES options (front-month filtered for both)
- **Silver**: Feature engineering output (reproducible, exactly-once)
  - `silver/features/es_pipeline/*` - ES system (16 stages, ~70 features, built with current CONFIG)
- **Gold**: Production ML datasets and streaming signals
  - `gold/training/*` - Promoted from Silver
  - `gold/streaming/*` - Real-time signals from Core Service
  - `gold/evaluation/*` - Backtest and validation results

**Entry Point**: `uv run python -m src.lake.main`

---

### 5. Gateway Service

**Location**: `backend/src/gateway/`  
**Role**: WebSocket relay (NATS → Frontend)  
**Interface**: [backend/src/gateway/INTERFACES.md](backend/src/gateway/INTERFACES.md)

**Key Responsibilities**:
- Subscribe to NATS `levels.signals` and `market.flow`
- Normalize payload to frontend contract (direction/signal mapping)
- Merge ML viewport predictions into per-level schema
- Cache latest payload
- Broadcast to WebSocket clients
- Health check endpoint

**Inputs**:
- NATS subject: `levels.signals`

**Outputs**:
- WebSocket: `ws://localhost:8000/ws/stream` (JSON frames, ~250ms cadence)
- HTTP: `GET /health` (health check)

**Key Characteristics**:
- Relay with normalization (transforms Core schema to frontend contract)
- Merges ML predictions from viewport into per-level enrichment
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
- Training data: `gold/training/signals_production.parquet`
- Live features: Engineered features from Core Service

**Outputs**:
- Model: `ml/production/xgb_prod.pkl`
- kNN index: `ml/production/knn_index.faiss`
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
- Level universe generation (PM/OR/SMA - 6 level kinds)
- Touch detection with monitor band filtering
- Physics feature computation (barrier, tape, fuel)
- Outcome labeling (BREAK/BOUNCE with competing-risks timing)

**Inputs**:
- Bronze Parquet: ES trades, MBP-10, option trades
- Warmup data: 3 prior days (SMA), 7 prior days (relative volume)

**Outputs**:
- Gold Parquet: `gold/research/signals_vectorized.parquet`
- Schema defined in `backend/features.json`

**Key Components**:
- `VectorizedPipeline`: High-performance batch processing
- `generate_level_universe_vectorized`: Structural level detection (6 level kinds)

**Level Universe** (ES system):
- **PM_HIGH**: Pre-market high (04:00-09:30 ET) from ES futures
- **PM_LOW**: Pre-market low (04:00-09:30 ET) from ES futures
- **OR_HIGH**: Opening range high (09:30-09:45 ET) from ES futures
- **OR_LOW**: Opening range low (09:30-09:45 ET) from ES futures
- **SMA_200**: 200-period moving average on 2-min ES bars
- **SMA_400**: 400-period moving average on 2-min ES bars

**Total**: 6 level kinds (4 structural extremes + 2 moving averages)

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
**Role**: Real-time UI for ES 0DTE break/bounce signals (ES futures + ES options)  
**Interface**: [frontend/INTERFACES.md](frontend/INTERFACES.md)

**Key Responsibilities**:
- Connect to Gateway WebSocket (`/ws/stream`)
- Display price ladder with level markers
- Visualize break/bounce strength meters
- Show physics attribution (barrier, tape, fuel)
- Render dealer mechanics (gamma velocity, exposure)
- Options flow panel (strike grid + flow chart)
- Compute UI-specific derived metrics (strength scores)

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
- `OptionsPanelComponent`: Strike grid + flow chart

**Key Services**:
- `DataStreamService`: WebSocket connection + parsing
- `LevelDerivedService`: Strength computation

**State Management**: RxJS BehaviorSubjects (no global state store)

**Entry Point**: `npm run start` (Angular dev server at `http://localhost:4200`)

---

## Data Flow Contracts

### Event Time Discipline

**All events carry**:
- `ts_event_ns`: Event time (from exchange) in Unix nanoseconds UTC
- `ts_recv_ns`: Receive time (by our system) in Unix nanoseconds UTC

**Timestamp Conversion Rules**:
- Databento: already in nanoseconds → use directly
- All events: `ts_event_ns` (exchange time), `ts_recv_ns` (receive time)

---

### NATS Subject Hierarchy

**Market Data** (Ingestor → Core/Lake):
- `market.futures.trades`: ES futures trades (front-month only)
- `market.futures.mbp10`: ES MBP-10 depth updates (front-month only)
- `market.options.trades`: ES option trades (front-month contract, 0DTE)
- `market.options.nbbo`: ES option NBBO (front-month contract, 0DTE)

**Derived Signals** (Core → Lake/Gateway):
- `levels.signals`: Level signals payload (break/bounce physics)

**Stream Configuration**:
- Retention: 24 hours
- Storage: File-backed persistence
- Consumers: Durable (resume on restart)

---

### Price Protocol (ES System)

**ES futures = ES options = same price scale (PERFECT ALIGNMENT)**:
1. Level specified in ES index points: `level_price = 5850.0`
2. Query ES depth/trades at same price: `es_level = 5850.0` (no conversion!)
3. Query ES option strikes near level: `strikes = [5825, 5850, 5875]` (25pt spacing)
4. Output in ES index points: `level_price = 5850.0`

**Ratio**: ES futures / ES options ≈ 1.0 (NO CONVERSION! Small basis spread tracked for diagnostics)

---

## Configuration Contract

**Single Source of Truth**: `backend/src/common/config.py` (CONFIG singleton)

**Key Parameters** (ES System - Validated from Real Data):
- **Strike specs**: `ES_0DTE_STRIKE_SPACING=25.0` pts (dominant ATM), `5.0` pts (tight ATM rare)
- **Outcome**: `OUTCOME_THRESHOLD=75.0` pts (3 strikes × 25pt), `LOOKFORWARD_MINUTES=8`
- **Strength**: `STRENGTH_THRESHOLD_1=25.0` (1 strike), `STRENGTH_THRESHOLD_2=75.0` (3 strikes)
- **Physics windows**: `W_b=240s` (barrier/confirmation), `W_t=60s` (tape), `W_g=60s` (fuel)
- **Monitoring**: `MONITOR_BAND=5.0` pts (interaction zone), `TOUCH_BAND=2.0` pts (precise contact)
- **Fuel range**: `FUEL_STRIKE_RANGE=75.0` pts (±3 strikes)
- **Time window**: RTH 09:30-13:30 ET (first 4 hours only)
- **Thresholds**: `R_vac=0.3`, `R_wall=1.5`, `F_thresh=100`
- **Score weights**: `w_L=0.45`, `w_H=0.35`, `w_T=0.20`
- **Smoothing**: `tau_score=2.0s`, `tau_velocity=1.5s`
- **Snap interval**: `SNAP_INTERVAL_MS=250`
- **Warmup**: `SMA_WARMUP_DAYS=3`

**Access Pattern**: Import CONFIG singleton from `backend/src/common/config.py` to access all tunable parameters.

---

## Deployment Architecture

### Docker Compose Services

**Services**: NATS (message bus), MinIO (object storage), Ingestor (data ingestion), Core (physics engines), Lake (data persistence), Gateway (WebSocket relay)

**Configuration**: See `docker-compose.yml` for complete service definitions.

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
2. **Front-month purity**: ES futures AND ES options filtered to same contract (e.g., ESZ5)
3. **Price scale**: ES futures = ES options (same index points, no conversion)
4. **NATS durability**: 24-hour retention ensures replay capability
5. **Deterministic**: Same inputs + CONFIG → same outputs
6. **Causal**: Features use only data before label time (no hindsight)
7. **RTH window**: 09:30-13:30 ET (first 4 hours)
8. **Deterministic IDs**: Event IDs are reproducible (no UUIDs)
9. **Single Silver path**: `silver/features/es_pipeline/` (rebuild with --force when CONFIG changes)

---

## Performance Targets

**Latency** (end-to-end - replay mode):
- Databento DBN replay → NATS publish: <10ms
- NATS → Core processing → NATS publish: <50ms
- NATS → Gateway → WebSocket: <5ms
- **Total: <65ms** (replay → frontend)

**Note**: The system uses historical DBN replay (`replay_publisher.py`) for training and backtesting. The full streaming infrastructure (NATS → Core → Lake → Gateway → Frontend) is operational and tested. Live streaming is supported via the same data pipeline.

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

### Data Architecture
- Data Architecture & Workflow: [backend/DATA_ARCHITECTURE.md](backend/DATA_ARCHITECTURE.md)
- Data Directory Structure: [backend/data/README.md](backend/data/README.md)
- Feature Manifest Schema: [backend/src/common/schemas/feature_manifest.py](backend/src/common/schemas/feature_manifest.py)

### Full Documentation
- Root: [README.md](README.md)
- Backend: [backend/README.md](backend/README.md)
- Frontend: [frontend/README.md](frontend/README.md)
- Feature Contract (legacy): [backend/features.json](backend/features.json)

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

**Last Updated**: 2025-12-28  
**Status**: Production (ES futures + ES 0DTE options neuro-hybrid system)
