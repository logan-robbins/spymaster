# System Components Map

**Project**: Spymaster (ES Futures + ES 0DTE Options Break/Bounce Physics Engine)  
**Audience**: Engineering Team & AI Coding Agents  
**Purpose**: Component architecture with interface contracts for parallel development

---

## Architecture (ES Futures + ES Options)

**Perfect Alignment Strategy**:
- **Spot + Liquidity**: ES futures (trades + MBP-10)
- **Gamma Exposure**: ES 0DTE options 
- **Venue**: CME Globex (GLBX.MDP3 dataset)

**ES 0DTE Specs** (CME standard):
- Strike listing: CME schedule varies by moneyness/time-to-expiry (5/10/50/100-point intervals; dynamic 5-point additions); GEX bands aggregate across listed strikes within point bands
- Modeling barrier: volatility-scaled (point-band based, no fixed strike count)
- Time window: 09:30-13:30 ET (first 4 hours)
- Level types: 6 only (PM High/Low, OR High/Low, SMA200, SMA400)

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
- Configuration singleton (CONFIG with all tunable parameters, auto-loaded from best-config JSON at `data/ml/experiments/zone_opt_v1_best_config.json` unless `CONFIG_OVERRIDE_PATH` is set)
- Price conversion (ES ↔ ES no-op)
- Storage schemas (Pydantic + PyArrow for Bronze/Silver/Gold - **enforced**)
  - Bronze: `FuturesTradeV1`, `MBP10V1`, `OptionTradeV1` (9 schemas)
  - Silver: `SilverFeaturesESPipelineV1` (182 columns, validated in pipeline)
  - Gold: `GoldTrainingESPipelineV1`, `LevelSignalV1`
- NATS bus wrapper (pub/sub messaging)

**Dependencies**: None (zero dependencies on other backend modules)

**Consumers**: All other backend modules

---

### 2. Ingestor Service

**Location**: `backend/src/ingestion/`  
**Role**: Data ingestion and normalization  
**Interface**: [backend/src/ingestion/INTERFACES.md](backend/src/ingestion/INTERFACES.md)

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
- Live: `uv run python -m src.ingestion.main`
- Replay: `uv run python -m src.ingestion.replay_publisher`

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

**Location**: `backend/src/io/`  
**Role**: Bronze/Silver/Gold data persistence  
**Interface**: [backend/src/io/INTERFACES.md](backend/src/io/INTERFACES.md)

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
  - `silver/features/es_pipeline/*` - ES system (16 stages, 182 columns: 10 identity + 108 engineered features + 64 labels; built with CONFIG loaded from best-config JSON)
- **Gold**: Production ML datasets and streaming signals
  - `gold/training/*` - Promoted from Silver
  - `gold/streaming/*` - Real-time signals from Core Service
  - `gold/evaluation/*` - Backtest and validation results

**Entry Point**: `uv run python -m src.io.main`

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
- Outcome labeling (BREAK/BOUNCE with volatility-scaled triple-barrier timing)

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

## Pipeline Validation Progress

**Last Updated**: 2025-12-29  
**Validated Stages**: 0-15 (load_bronze → filter_rth)  
**Latest Fix**: DatetimeIndex handling in `src/pipeline/stages/label_outcomes.py`  
**Latest Result**: 53 RTH-filtered signals with silver schema pass (Stage 15)  
**Silver Schema**: `backend/SILVER_SCHEMA.md`  
**Docs Updated**: `backend/src/io/INTERFACES.md` aligned with current lake interfaces  
**Legacy Cleanup**: Removed stock/greeks schemas and legacy pipeline stage stubs
