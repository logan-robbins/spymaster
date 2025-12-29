# Data Architecture & Workflow

**Version**: 5.0  
**Last Updated**: 2025-12-28  
**Status**: Development (v1 - ES Futures + ES Options)

---

## Overview

Spymaster uses the **Medallion Architecture** (Bronze → Silver → Gold) for production ML pipelines targeting **sparse, high-precision kNN retrieval** from ES 0DTE physics.

**v1 Architecture**: ES Futures + ES 0DTE Options (Perfect Alignment)
- **Spot + Liquidity**: ES futures (trades + MBP-10)
- **Gamma Exposure**: ES 0DTE options (same underlying!)
- **Venue**: CME Globex (GLBX.MDP3 dataset)
- **Conversion**: NONE - ES = ES (zero basis spread)

**Key Principles**:
- **Immutability**: Bronze is append-only, never modified
- **Reproducibility**: Bronze + manifest → deterministic Silver output
- **Reproducibility**: Features deterministically built from Bronze + CONFIG
- **Event-time first**: All records carry `ts_event_ns` and `ts_recv_ns`
- **Continuous inference**: Features computed every 2-min candle (not just touches)
- **Multi-window lookback**: 1-20 minute windows encode "setup" across timescales

---

## Directory Structure

```
data/
├── raw/                                # Stage 0: Source data (Databento DBN)
│   ├── dbn/trades/, dbn/mbp10/        # ES futures (GLBX.MDP3)
│   └── (no polygon for v1 - ES options from Databento too!)
│
├── lake/                              # Lakehouse tiers
│   ├── bronze/                        # Stage 1: Normalized events (immutable)
│   │   ├── futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   │     └── (front-month only, filtered by ContractSelector)
│   │   ├── futures/mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   │     └── (front-month only, downsampled to 1Hz)
│   │   ├── options/trades/underlying=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   │     └── (front-month contract only, 0DTE filtered, CME GLBX.MDP3)
│   │   └── options/nbbo/underlying=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │         └── (MBP-1 NBBO for ES options, front-month contract only)
│   │
│   ├── silver/                        # Stage 2: Feature engineering output
│   │   └── features/
│   │       └── es_pipeline/           # ES system (16 stages, ~70 features)
│   │           ├── manifest.yaml      # Feature config (from CONFIG.py)
│   │           ├── validation.json    # Quality metrics
│   │           └── date=YYYY-MM-DD/*.parquet
│   │
│   └── gold/                          # Stage 3: Production ML datasets
│       ├── training/
│       │   ├── signals_v2_multiwindow.parquet  # Curated from best Silver
│       │   └── signals_v2_metadata.json
│       ├── evaluation/backtest_YYYY-MM-DD.parquet
│       └── streaming/signals/underlying=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│
└── ml/                                # Model artifacts + hyperopt studies
    ├── experiments/
    │   ├── zone_hyperopt/             # Feature engineering hyperopt
    │   └── model_hyperopt/            # Model training hyperopt
    ├── production/
    │   ├── xgb_prod.pkl               # Production XGBoost model
    │   └── knn_index.faiss            # kNN retrieval index
    └── registry.json
```

**Note**: Paths above are relative to `backend/`. From repo root, prefix with `backend/`.

---

## Data Flow

### Historical Data Ingestion (ES Options)

```
Databento GLBX.MDP3 (CME Globex)
    ├─→ ES Futures (trades + MBP-10)
    └─→ ES Options (trades + NBBO)
            ↓
    [download_es_options.py + DBNIngestor]
            ↓
    Front-Month Filtering:
    - ES futures: Volume-dominant contract selection (ContractSelector)
    - ES options: Filter to same contract (e.g., ESZ5 only)
    - 0DTE filter: exp_date == session_date
            ↓
Bronze (all hours 00:00-23:59 UTC, immutable, Parquet+ZSTD)
    ├── futures/trades/symbol=ES/
    ├── futures/mbp10/symbol=ES/
    ├── options/trades/underlying=ES/
    └── options/nbbo/underlying=ES/
```

### Batch Feature Engineering Pipeline (Continuous Inference Model)

```
Bronze (ES futures + ES options, front-month filtered)
    ↓
[ES Pipeline - 16 stages]
    ↓
Stage 1-4: Load & Sessionize
  - LoadBronzeStage (ES trades/MBP-10/options)
  - BuildSPXOHLCVStage (1min, 2min bars from ES)
  - SessionizeStage (compute minutes_since_open relative to 09:30 ET)
    ↓
Stage 5-6: Level Universe
  - GenerateLevelsStage (6 level kinds: PM/OR high/low + SMA_200/400)
  - DetectInteractionZonesStage (continuous: every 2-min candle, not just touches!)
    ↓
Stage 7-13: Physics Features (Multi-Window + Single-Window)
  - ComputePhysicsStage (barrier/tape/fuel from engines)
  - ComputeMultiWindowKinematicsStage (velocity/accel/jerk × [1,3,5,10,20]min)
  - ComputeMultiWindowOFIStage (integrated OFI × [30,60,120,300]s)
  - ComputeBarrierEvolutionStage (depth changes × [1,3,5]min)
  - ComputeLevelDistancesStage (signed distances to structural levels)
  - ComputeGEXFeaturesStage (gamma exposure ±[1,2,3] strikes)
  - ComputeForceMassStage (F=ma validation)
    ↓
Stage 14: Approach Context
  - ComputeApproachFeaturesStage (approach metrics + session timing)
    ↓
Stage 15: Label Outcomes
  - LabelOutcomesStage (triple-barrier ±75pts, 8min forward)
    ↓
Stage 16: Filter to RTH
  - FilterRTHStage (09:30-13:30 ET)
    ↓
Silver Features (RTH only 09:30-13:30 ET)
    silver/features/es_pipeline/
        ├── manifest.yaml (records CONFIG used)
        ├── validation.json (quality metrics)
        └── date=YYYY-MM-DD/*.parquet (~15-25 events/day)
    ↓
[GoldCurator] → Promote to Gold
    ↓
Gold Training (production ML dataset)
    gold/training/signals_v2_multiwindow.parquet
    ↓
    STAGE 2: Model Training Hyperopt (SEPARATE STAGE!)
    - Optimize XGBoost hyperparams, feature selection, kNN blend
    - Goal: Maximum Precision@80% on test set
    - Input: FIXED optimized features from Stage 1
    - Output: Production model
    ↓
Model Artifacts
    ml/production/
        ├── xgb_prod.pkl (XGBoost model)
        └── knn_index.faiss (kNN retrieval index)
```

### Streaming Pipeline (Real-time Databento)

**Status**: Infrastructure complete, awaiting live Databento client implementation.

**Current**: Historical replay via `replay_publisher.py` (DBN files → NATS)  
**Future**: Live Databento feed via streaming client

```
Live Databento Feed (ES futures + ES options)   [NOT YET IMPLEMENTED]
    ↓ [Ingestor with live client]
NATS (market.futures.*, market.options.*)       [WORKING - tested with replay]
    ├─→ [BronzeWriter] → Bronze (all hours, append-only)     [WORKING]
    └─→ [Core Service] → Continuous inference every 2-min candle  [WORKING]
            ├─→ Load model (xgb_prod.pkl + knn_index)
            ├─→ Compute multi-window physics features
            ├─→ kNN retrieval: Find 5 similar past events
            └─→ Predict: "4/5 past similar setups BROKE → 80% confidence"
                    ↓
            NATS (levels.signals) → Gateway → Frontend    [WORKING]
```

**What's needed for v2**: Replace `replay_publisher.py` with live Databento streaming client for ES futures + options.

---

## Lakehouse Tiers

### Bronze (Stage 1)

**Purpose**: Normalized events in canonical schema  
**Format**: Parquet with ZSTD compression  
**Schema**: Enforced via PyArrow - `FuturesTradeV1`, `MBP10V1`, `OptionTradeV1` (see `backend/src/common/schemas/`)  
**Time Coverage**: All hours (00:00-23:59 UTC, ~04:00-20:00 ET after timezone conversion)  
**Semantics**: Append-only, at-least-once delivery  
**Retention**: Permanent

**v1 Data Sources**:
- **ES Futures**: Trades + MBP-10 from Databento GLBX.MDP3
- **ES Options**: Trades + NBBO from Databento GLBX.MDP3
- **Front-month filtering**: CRITICAL quality gate
  - ES futures: Volume-dominant contract selection (60% threshold)
  - ES options: Filtered to SAME contract as ES futures (e.g., ESZ5)
  - 0DTE filter: Only options with `exp_date == session_date`

**Why all hours?** 
- Pre-market (04:00-09:30 ET) required for PM_HIGH/PM_LOW calculation
- ES futures trade 23/5 (nearly 24/7)
- Downstream layers filter to RTH (09:30-13:30 ET for v1)

**Critical Invariant**: ES futures AND ES options must use THE SAME contract to avoid roll-period contamination.

**Writer**: `scripts/download_es_options.py` (batch) + `src/lake/bronze_writer.py` (streaming)  
**Reader**: `src/lake/bronze_writer.py` (BronzeReader) and `src/pipeline/utils/duckdb_reader.py` (with `front_month_only=True`)

### Silver (Stage 2)

**Purpose**: Feature engineering output  
**Format**: Parquet with ZSTD compression  
**Schema**: `SilverFeaturesESPipelineV1` (182 columns, enforced) - see `backend/src/common/schemas/silver_features.py`  
**Time Coverage**: RTH only (09:30-13:30 ET)  
**Semantics**: Reproducible transformations (Bronze + CONFIG → deterministic Silver)  
**Retention**: Overwrite when CONFIG changes (no versioning needed)

**v1 Inference Model**: **CONTINUOUS** (not event-based!)
- Features computed **every 2-minute candle close**
- NOT only when price touches a level
- For each candle: compute features for NEAREST level in each category (PM/OR/SMA)
- **Why**: Traders need continuous assessment, not just at touches
- **Benefit**: Model gets more accurate as price approaches level

**Feature Set** (Multi-Window Lookback):
- **Kinematics**: velocity (5 windows), acceleration (4 windows), jerk (4 windows) at **1, 3, 5, 10, 20 minutes** = 13 features
- **OFI**: integrated OFI (4 windows), OFI trend (3 windows), variance, peak at **30, 60, 120, 300 seconds** = 9 features
- **Barrier**: depth evolution at **1, 3, 5 minutes** (7 features: current, 3x changes, slope, volatility, absorption rate)
- **GEX**: Strike-banded gamma at **±1, ±2, ±3 strikes** (25pt spacing) = 9 features
- **Structural**: PM/OR/SMA distance features (8 level distance features)
- **Force/Mass**: F=ma validation (4 features)
- **Session Context**: 5 features (timing, ATR, distance)
- **Tape/Fuel**: ~8 additional physics features
- **Total**: ~70 physics features (multi-window encoding)

**Key Capabilities**:
- Deterministic: Bronze + CONFIG → reproducible Silver output
- Hyperopt updates CONFIG.py, then rebuild with --force
- Manifest records exact CONFIG snapshot used for each build
- Multi-window features (1-20min lookback) for kNN retrieval

**Implementation**: `SilverFeatureBuilder` class (`src/lake/silver_feature_builder.py`) + `src/pipeline/pipelines/es_pipeline.py`

### Gold (Stage 3)

**Purpose**: Production-ready ML datasets and evaluation results  
**Format**: Parquet with ZSTD compression  
**Schema**: `GoldTrainingESPipelineV1` (182 columns, identical to Silver but curated) - see `backend/src/common/schemas/gold_training.py`  
**Semantics**: Curated, validated, production-quality  
**Retention**: Keep production datasets permanently

**Production Workflow**:

**Step 1: Build Features**
- Input: Bronze (ES futures + ES options)
- Config: Current CONFIG.py values
- Output: `silver/features/es_pipeline/` 
- Command: `uv run python -m src.lake.silver_feature_builder --pipeline es_pipeline`

**Step 2 (Optional): Optimize CONFIG**
- Run hyperopt to find better zones/windows
- Update CONFIG.py with best parameters
- Rebuild features with `--force` flag
- Metrics: kNN-5 purity, Silhouette, Precision@80%

**Step 3: Train Model**
- Input: `silver/features/es_pipeline/` (curated to Gold)
- Search: XGBoost params, feature selection, kNN blend
- Output: `ml/production/xgb_prod.pkl` + `knn_index.faiss`
- Metrics: Precision@80% > 90%

**Sources**:
- `gold/training/`: Promoted from `silver/features/es_pipeline/`
- `gold/streaming/`: Real-time signals from Core Service (continuous inference)
- `gold/evaluation/`: Backtest and validation results

**Implementation**: `GoldCurator` class (`src/lake/gold_curator.py`) + `src/ml/zone_objective.py`

---

## Pipeline Architecture

The pipeline module (`src/pipeline/`) provides modular, versioned feature engineering. Each pipeline version uses a different stage composition.

### Core Components

See `backend/src/pipeline/` directory structure for the modular pipeline architecture including stages, pipelines, and utilities.

### Pipeline Stages (v1.0 - 16 stages)

Each stage is a class extending `BaseStage` with explicit inputs/outputs:

| Stage | Class | Description | Outputs |
|-------|-------|-------------|---------|
| **Data Loading** |
| 1 | `LoadBronzeStage` | Load ES futures + options (front-month) | Raw data |
| 2 | `BuildOHLCVStage` (1min) | Build 1-min ES bars | `atr` |
| 3 | `BuildOHLCVStage` (2min) | Build 2-min ES bars + warmup | `ohlcv_2min` |
| 4 | `InitMarketStateStage` | Initialize market state | - |
| **Level Universe** |
| 5 | `GenerateLevelsStage` | Generate 6 level kinds | `level_info` |
| 6 | `DetectInteractionZonesStage` | Continuous (every 2-min candle) | `signals_df` |
| **Physics Features** |
| 7 | `ComputePhysicsStage` | Barrier/tape/fuel from engines | +10 physics |
| 8 | `ComputeMultiWindowKinematicsStage` | Velocity/accel/jerk × [1,3,5,10,20]min | +15 |
| 9 | `ComputeMultiWindowOFIStage` | Integrated OFI × [30,60,120,300]s | +9 |
| 10 | `ComputeBarrierEvolutionStage` | Barrier depth changes × [1,3,5]min | +7 |
| 11 | `ComputeLevelDistancesStage` | Distances to structural levels | +8 |
| 12 | `ComputeGEXFeaturesStage` | Gamma exposure ±[1,2,3] strikes | +9 |
| 13 | `ComputeForceMassStage` | F=ma validation | +4 |
| **Approach & Labels** |
| 14 | `ComputeApproachFeaturesStage` | Approach + session timing | +5 |
| 15 | `LabelOutcomesStage` | Triple-barrier ±75pts, 8min forward | +20 labels |
| 16 | `FilterRTHStage` | Filter to RTH 09:30-13:30 ET | Final dataset |

**Total**: 16 stages, ~77 total features (identity + level + physics + labels)

### Pipeline Versions

### Pipeline: ES Futures + ES Options Physics (16 stages)

**Architecture**: ES futures (spot + liquidity) + ES 0DTE options (gamma)  
**Inference**: Continuous (every 2-min candle)  
**Features**: ~70 physics features (multi-window 1-20min) + ~40 labels  
**Labels**: Triple-barrier ±75pts (3 strikes), 8min forward  
**RTH**: 09:30-13:30 ET (first 4 hours)

**Pipeline Flow**: LoadBronze → BuildOHLCV → InitMarketState → GenerateLevels → DetectInteractionZones → ComputePhysics → MultiWindow stages → LabelOutcomes → FilterRTH

**Implementation**: See `backend/src/pipeline/pipelines/es_pipeline.py` for complete 16-stage pipeline definition.

**Event Density**: ~15-25 events/day (sparse, high-precision)  
**Use Case**: kNN retrieval - "Find 5 similar setups, 4/5 BROKE → 80% confidence"

### Level Universe

Generated levels from ES futures (**6 level kinds** total):

| Level Kind | Description | Source |
|------------|-------------|--------|
| **PM_HIGH** | Pre-market high (04:00-09:30 ET) | ES futures |
| **PM_LOW** | Pre-market low (04:00-09:30 ET) | ES futures |
| **OR_HIGH** | Opening range high (09:30-09:45 ET) | ES futures |
| **OR_LOW** | Opening range low (09:30-09:45 ET) | ES futures |
| **SMA_200** | 200-period moving average (2-min bars) | ES futures |
| **SMA_400** | 400-period moving average (2-min bars) | ES futures |

**Total**: 6 level kinds (4 structural extremes + 2 moving averages)

**Rationale for Pruning**:
- PM/OR levels: Institutional recognition, sharp turning points
- SMA levels: Mean reversion anchors
- Removed: Lagging indicators (VWAP), constantly changing (SESSION_HIGH/LOW), now in features (walls)

### Feature Categories (v1 - Core Physics Features)

**Multi-Window Kinematics** (13 features - LEVEL FRAME):
- `velocity_{1,3,5,10,20}min` - Velocity at 5 timescales (ES pts/min) = 5 features
- `acceleration_{1,3,5,10}min` - Acceleration at 4 timescales (pts/min²) = 4 features
- `jerk_{1,3,5,10}min` - Jerk at 4 timescales (pts/min³) = 4 features
- **Purpose**: Encode "setup shape" - fast aggressive vs slow decelerating approach

**Multi-Window OFI** (9 features - ORDER FLOW):
- `integrated_ofi_{30,60,120,300}s` - Cumulative weighted OFI at 4 windows
- `ofi_trend_{60,120,300}s` - OFI slope over window
- `ofi_variance_120s` - OFI volatility
- `ofi_peak_120s` - Max OFI spike
- **Purpose**: Capture order flow pressure dynamics across timescales

**Barrier Evolution** (7 features - LIQUIDITY DYNAMICS):
- `barrier_depth_current` - Current depth at level (ES contracts)
- `barrier_depth_change_{1,3,5}min` - Depth evolution (thinning vs thickening)
- `barrier_slope_3min` - Depth trend
- `barrier_volatility_5min` - Depth stability
- `barrier_absorption_rate` - Fill rate vs replenishment
- **Purpose**: Detect "thinning barrier" (bullish BREAK) vs "thickening barrier" (bullish BOUNCE)

**GEX Features** (9 features - GAMMA EXPOSURE):
- `gex_above_{1,2,3}strike` - Gamma above level at ±1/±2/±3 strikes (25pt spacing)
- `gex_below_{1,2,3}strike` - Gamma below level
- `gex_asymmetry` - Net directional gamma pressure
- `gex_ratio_3strike` - Relative gamma concentration
- `net_gex_3strike` - Net dealer exposure ±75pts
- **Purpose**: Dealer hedging pressure (AMPLIFY vs DAMPEN)

**Level Distances** (8 features - STRUCTURAL CONTEXT):
- `dist_to_pm_high/low` - Signed distance to PM levels
- `dist_to_or_high/low` - Signed distance to OR levels  
- `dist_to_sma_200/400` - Signed distance to SMA levels
- All with `_atr` and `_pct` normalizations
- **Purpose**: Position within structural level framework

**Force-Mass** (4 features - PHYSICS VALIDATION):
- `predicted_accel` - F/m (OFI / barrier_depth)
- `actual_accel` - Observed from kinematics
- `fma_residual` - actual - predicted (hidden momentum?)
- `fma_ratio` - actual / predicted
- **Purpose**: Cross-validate force and mass proxies with observed acceleration

**Session Context** (5 features):
- `minutes_since_open` - Time since 09:30 ET (0-240 for v1)
- `bars_since_open` - Bars since open
- `is_first_15m` - Opening range volatility
- `atr` - Average True Range (volatility normalization)
- `distance` - Distance to level (ES points)

**Labels** (Triple-Barrier):
- `outcome` (BREAK, BOUNCE, CHOP) - First ±75pt (3 strike) threshold hit
- `strength_signed` - Signed excursion magnitude
- `time_to_break/bounce_{1,2}` - Time to ±25pt, ±75pt thresholds
- `tradeable_1/2` - Threshold reached flags

---


## Quick Start

### 1. Download ES Options Data

```bash
cd backend

# Set Databento API key
echo "DATABENTO_API_KEY=your_key_here" >> .env

# Download ES options (trades + NBBO) - front-month filtered
uv run python scripts/download_es_options.py \
  --start 2025-11-02 \
  --end 2025-12-28
  
# Expected: ~60 trading days, ES 0DTE options from CME
```

### 2. Build Features

```bash
cd backend

# Build features using current CONFIG.py
uv run python -m src.lake.silver_feature_builder \
  --pipeline es_pipeline \
  --start-date 2025-11-02 \
  --end-date 2025-12-28

# Output: silver/features/es_pipeline/
#   ~15-25 events/day × 60 days = ~900-1500 events
#   ~70 physics features + ~40 label columns
```

### 3. Train Model

```bash
# Promote Silver to Gold
uv run python -m src.lake.gold_curator \
  --action promote \
  --silver-path es_pipeline

# Train model
uv run python -m src.ml.boosted_tree_train \
  --features gold/training/signals_production.parquet

# Output: ml/production/xgb_prod.pkl + knn_index.faiss
```

### 4. (Optional) Optimize CONFIG

```bash
# If model performance needs improvement:

# Run hyperopt to find better zones/windows
uv run python scripts/run_zone_hyperopt.py \
  --start-date 2025-11-02 \
  --end-date 2025-11-30 \
  --n-trials 200

# Update CONFIG.py with best params from MLflow
# Edit src/common/config.py: MONITOR_BAND=3.8, etc.

# Rebuild features (overwrites)
uv run python -m src.lake.silver_feature_builder \
  --pipeline es_pipeline \
  --start-date 2025-11-02 \
  --end-date 2025-12-28 \
  --force

# Retrain model with improved features
```

### 5. Deploy

```bash
# Core Service loads: ml/production/xgb_prod.pkl + knn_index.faiss
uv run python -m src.core.service
```

---


## Configuration

Pipeline behavior controlled by `backend/src/common/config.py`:

### v1 Configuration (ES System)

**See `backend/src/common/config.py` for authoritative parameter values.**

**Key Configuration Concepts**:

| Category | Purpose |
|----------|---------|
| **Instruments** | ES futures (spot + liquidity) + ES 0DTE options (gamma) from CME GLBX.MDP3 |
| **RTH Window** | 09:30-13:30 ET (first 4 hours) - when to generate training events |
| **Premarket** | 04:00-09:30 ET - for PM_HIGH/PM_LOW calculation only |
| **Strike Spacing** | 25pt (ATM dominant), validated from real ES 0DTE data |
| **Interaction Zones** | MONITOR_BAND (event detection), TOUCH_BAND (precise contact) - **TUNABLE via hyperopt** |
| **Outcome Labels** | OUTCOME_THRESHOLD (3-strike min), LOOKFORWARD_MINUTES - **TUNABLE via hyperopt** |
| **Multi-Window Lookback** | 1-20min kinematics, 30s-5min OFI, 1-5min barrier - encode setup across timescales |
| **Base Physics Windows** | W_b/W_t/W_g - engine lookback windows - **TUNABLE via hyperopt** |
| **Level Selection** | use_pm/use_or/use_sma_200/use_sma_400 - **TUNABLE via hyperopt** |

**Inference Model**: Continuous (features computed every 2-min candle close)

**Hyperopt** (Stage 1): 29 tunable parameters including zone widths, windows, thresholds, level selection  
**Fixed**: Strike spacing (25pt from data), RTH window (09:30-13:30 ET), inference cadence (2-min)

---

## Hyperparameter Optimization

**Philosophy**: Optimize CONFIG parameters, then build features once with best config.

### How Hyperopt Works

**Stage 1: Zone/Window Optimization** (Optional - improves CONFIG):
```bash
cd backend

# Find best zones/windows using dry-run (no real data needed)
uv run python scripts/run_zone_hyperopt.py \
  --dry-run \
  --n-trials 100

# Or use real data for accurate results
uv run python scripts/run_zone_hyperopt.py \
  --start-date 2025-11-02 \
  --end-date 2025-11-30 \
  --n-trials 200
```

**Searches 29 parameters**:
- Zone widths (MONITOR_BAND, TOUCH_BAND, per-level)
- Outcome thresholds (strikes, lookforward)
- Physics windows (W_b, W_t, W_g)
- Multi-window selection (which timeframes to use)
- Level selection (use_pm, use_or, use_sma_200/400)

**Optimizes for**: kNN-5 purity (50%) + Precision@80% (30%) + Feature variance (20%)

**Output**: Best CONFIG parameters logged to MLflow
- Example: `MONITOR_BAND=3.8`, `W_t=45s`, `use_sma_200=False`

**Action**: Manually update `src/common/config.py` with best params, then rebuild

---

### Stage 2: Model Training

**Input**: Features from `silver/features/es_pipeline/` (built with current CONFIG)

```bash
# Train XGBoost + build kNN index
uv run python -m src.ml.boosted_tree_train \
  --features gold/training/signals_production.parquet

# Output: ml/production/xgb_prod.pkl + knn_index.faiss
```

**Model optimizes**: XGBoost hyperparams, feature selection, kNN blend weight

**Target**: Precision@80% > 90% (high confidence = high accuracy)

---

## Best Practices

### Immutability
- Bronze is append-only, never modified
- Silver versions are immutable once created
- Gold datasets are versioned with metadata

### Reproducibility
- Silver features are deterministic: Bronze + CONFIG → same output
- Manifest records CONFIG snapshot for each build
- Rebuild with `--force` when CONFIG changes

### Front-Month Purity (CRITICAL for ES System)
**Bronze Quality Gate**: ES futures AND ES options must be on SAME contract
- Use `ContractSelector` for volume-dominant ES futures contract
- Filter ES options to same contract (e.g., ESZ5)
- Prevents "ghost walls" during roll periods
- Implementation: `front_month_only=True` in BronzeReader

### RTH Filtering
**Critical**: Silver and Gold datasets contain ONLY RTH (09:30-13:30 ET for v1).

Why v1 uses first 4 hours only?
- Highest liquidity period
- Most institutional activity
- Clearest level respect/breaks
- Pre-market data (04:00-09:30 ET) used for PM_HIGH/PM_LOW but NOT for training labels
- Implementation: `FilterRTHStage` with `RTH_END_HOUR=13`

### Continuous Inference Model
**New in v1**: Features computed **every 2-min candle**, not just at touches
- More data: ~120 inference points/day (vs ~20 touch events)
- Early warning: Predict before price reaches level
- Distance-weighted: Model more accurate closer to level
- Use case: "Price 10pts from PM_HIGH, approaching fast → 75% BREAK confidence"

### Sparse is Better (kNN Retrieval)
**Counter-intuitive**: 10-20 high-quality events/day > 100 noisy events/day
- kNN retrieval accumulates events over time (60 days × 15 events = 900 examples)
- Precision > Recall (better to sit out than be wrong)
- Hyperopt optimizes for physics distinctiveness, not event density

---

## Continuous Inference Architecture

**Inference Model**: Features computed every 2-minute candle close

```
09:30 ── 09:32 ── 09:34 ── 09:36 ── 09:38 ── 09:40 ...
  ↓       ↓       ↓       ↓       ↓       ↓
Compute features for NEAREST level in each category (PM/OR/SMA)
```

**Benefits**:
- 120 inference points/day (240min RTH / 2min candles)
- Early warning: "Price 10pts from level, approaching fast → 75% BREAK"
- Distance-weighted: More confident closer to level
- Continuous assessment (trader mental model)

### Feature Computation Strategy

**For each 2-min candle close**:
1. Find NEAREST level in each category:
   - Nearest PM level (PM_HIGH or PM_LOW)
   - Nearest OR level (OR_HIGH or OR_LOW)
   - Nearest SMA level (SMA_200 or SMA_400)

2. Compute physics features relative to each:
   - Multi-window kinematics (1-20min lookback)
   - Multi-window OFI (30s-5min lookback)
   - Barrier evolution (1-5min lookback)
   - GEX features (±3 strikes around level)

3. Inference:
   - kNN: Find 5 most similar past events in physics space
   - Model: Combine kNN + XGBoost prediction
   - Output: BREAK/BOUNCE probability with confidence

**Data Volume**:
- RTH window: 09:30-13:30 ET = 4 hours = 240 minutes
- Candle interval: 2 minutes
- Candles per day: 240 / 2 = **120 candles/day**
- Levels per candle: 3 (nearest in each PM/OR/SMA category)
- Total rows/day: **120 × 3 = 360 rows/day**

**After filtering** (only when distance < MONITOR_BAND):
- Sparse events: ~10-20/day actually close enough to a level for prediction

---

## Performance

**Apple M4 Silicon Optimized**:
- All operations use NumPy broadcasting and vectorized pandas
- Batch processing of all candles simultaneously
- Multi-window features computed with rolling operations (efficient!)
- Memory-efficient chunked processing

**Typical Performance** (M4 Mac, 128GB RAM):
- Single date (continuous inference, 120 candles): ~8-15 seconds
- Single date (v1 pipeline, 16 stages, ~77 features): ~10-20 seconds
- 10 dates: ~2-3 minutes
- Hyperopt single trial: ~2-3 minutes (1 pipeline run + model train)
- Stage 1 hyperopt (200 trials): ~7-10 hours
- Stage 2 hyperopt (100 trials): ~2-3 hours

**Data Throughput**:
- Feature engineering: ~20-40 candles/second
- Model inference (XGBoost + kNN): ~500-1000 predictions/second
- Bronze write: ~100k events/second (NATS → Parquet)

---

## Component Reference

### SilverFeatureBuilder
**File**: `src/lake/silver_feature_builder.py`  
**Purpose**: Build versioned Silver feature sets from Bronze  
**Key Methods**: `build_feature_set()`, `list_versions()`, `load_features()`, `register_experiment()`

### GoldCurator
**File**: `src/lake/gold_curator.py`  
**Purpose**: Promote best Silver experiments to Gold production  
**Key Methods**: `promote_to_training()`, `validate_dataset()`, `list_datasets()`

### Pipeline
**File**: `src/pipeline/core/pipeline.py`  
**Purpose**: Execute stage sequences for feature computation  
**Key Methods**: `run(date)`

### Pipeline Registry
**File**: `src/pipeline/pipelines/registry.py`  
**Purpose**: Map version strings to pipeline builders  
**Key Methods**: `get_pipeline_for_version()`, `list_available_versions()`

### BronzeWriter / BronzeReader
**File**: `src/lake/bronze_writer.py`  
**Purpose**: Write/read Bronze Parquet files from NATS streams  
**Note**: BronzeWriter subscribes to NATS; BronzeReader uses DuckDB for efficient queries

### DuckDBReader
**File**: `src/pipeline/utils/duckdb_reader.py`  
**Purpose**: Efficient Bronze data reading with downsampling for MBP-10  
**Key Methods**: `read_futures_trades()`, `read_futures_mbp10_downsampled()`, `get_warmup_dates()`

---

## Critical Invariants

1. **Bronze is append-only**: Never mutate or delete
2. **Silver is derived**: Always regeneratable from Bronze
3. **Event-time ordering**: All files sorted by `ts_event_ns`
4. **Idempotency**: Same Bronze + manifest → same Silver
5. **Front-month purity**: ES futures AND ES options on SAME contract (CRITICAL!)
6. **0DTE filtering**: ES options with `exp_date == session_date` only
7. **RTH filtering**: Silver/Gold contain only 09:30-13:30 ET signals (v1 = first 4 hours)
8. **Continuous inference**: Features at every 2-min candle (not just touches)
9. **Multi-window lookback**: 1-20min for kinematics, 30s-5min for OFI/barrier
10. **Partition boundaries**: Date/hour aligned to UTC
11. **Compression**: ZSTD level 3 for all tiers
12. **No conversion**: ES futures = ES options (same index points!)

### Front-Month Purity Validation

**Why it matters**: During roll periods, TWO ES contracts trade simultaneously
- Example: Dec 2025 (ESZ5) rolling to Mar 2026 (ESH6)
- If we mix both: Creates artificial "ghost walls" in our data
- ES options on ESZ5 ≠ ES options on ESH6

**Implementation**: See `backend/src/common/utils/contract_selector.py` (ContractSelector) and `backend/src/common/utils/bronze_qa.py` (BronzeQA) for front-month selection and quality gate enforcement (60% dominance threshold).

---

## Troubleshooting

### "No Bronze data found"
```bash
cd backend
ls -la data/lake/bronze/futures/trades/symbol=ES/
# If empty, run backfill:
uv run python scripts/backfill_bronze_futures.py
```

### Features already exist
Use `--force` to rebuild:
```bash
uv run python -m src.lake.silver_feature_builder \
  --pipeline es_pipeline \
  --dates 2025-11-02:2025-12-28 \
  --force
```

### High null rates in features
Check validation report:
```bash
cat data/lake/silver/features/es_pipeline/validation.json
```

---

---

## kNN Retrieval System (Neuro-Hybrid)

### The Use Case

**Query**: "Price approaching PM_HIGH with velocity_3min=+2.5, integrated_ofi_60s=+850, barrier_depth_1min=thin"

**kNN Retrieval**: Find k=5 most similar past events in physics feature space

**Prediction**: "4/5 similar setups resulted in BREAK → 80% confidence BREAK"

### Why Multi-Window Features Are Critical

**Single-Window Problem**: A single velocity measurement (e.g., velocity_10min=+2.0) could represent fast constant approach, slow acceleration, or fast deceleration - completely different setups with different outcomes.

**Multi-Window Solution**: By encoding velocity at multiple timescales (1min, 3min, 5min, 10min, 20min), we capture the "shape" of the approach. Fast aggressive approaches (accelerating) show increasing velocity across windows, while decelerating approaches show decreasing velocity. kNN retrieval can distinguish these patterns.

### kNN Index Structure

**Built from**: `gold/training/signals_v2_multiwindow.parquet` (curated events)

**Implementation**: See `backend/src/ml/build_retrieval_index.py` for FAISS index construction using normalized physics features (~20-30 most predictive features including multi-window kinematics, OFI, barrier evolution, and GEX).

**Retrieval Process**: At inference, find k=5 nearest neighbors in physics space, compute kNN probability from neighbor outcomes, and blend with XGBoost prediction (25% kNN + 75% XGBoost).

---

## References

**Core Modules**:
- **Schemas (enforced)**: `backend/src/common/schemas/`
  - Bronze: `FuturesTradeV1`, `MBP10V1`, `OptionTradeV1` (9 schema files)
  - Silver: `SilverFeaturesESPipelineV1` (182 columns, validated in pipeline)
  - Gold: `GoldTrainingESPipelineV1`, `LevelSignalV1` (training + streaming)
- **Lake module**: `backend/src/lake/`
- **Pipeline module**: `backend/src/pipeline/`
- **Configuration**: `backend/src/common/config.py`

**Key Modules**:
- **Pipeline**: `src/pipeline/pipelines/es_pipeline.py`
- **Multi-window stages**: `src/pipeline/stages/compute_multiwindow_*.py`
- **ES contract selection**: `src/common/utils/contract_selector.py`
- **Session timing**: `src/common/utils/session_time.py`
- **Price tracker**: `src/common/price_converter.py` (ES = ES, no conversion)

**Hyperopt Framework**:
- **Stage 1 objective**: `src/ml/zone_objective.py` (29-parameter search)
- **Config override**: `src/common/utils/config_override.py`
- **Hyperopt runner**: `scripts/run_zone_hyperopt.py`
- **Grid search**: `scripts/run_zone_grid_search.py`

**Data Download**:
- **ES options downloader**: `scripts/download_es_options.py`
- **Contract calendar**: `src/common/utils/es_contract_calendar.py`

**Documentation**:
- **Hyperopt plan**: `HYPEROPT_PLAN.md` (two-stage workflow)
- **Feature schema**: `features.json` (v2.0.0 - column specs)
