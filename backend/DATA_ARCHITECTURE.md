# Data Architecture & Workflow

## Overview

Spymaster uses the **Medallion Architecture** (Bronze → Silver → Gold) for production ML pipelines targeting **sparse, high-precision kNN retrieval** from ES 0DTE physics.

**Architecture**: ES Futures + ES 0DTE Options
- **Spot + Liquidity**: ES futures (trades + MBP-10)
- **Gamma Exposure**: ES 0DTE options
- **Venue**: CME Globex (GLBX.MDP3 dataset)

**Key Principles**:
- **Immutability**: Bronze is append-only, never modified
- **Reproducibility**: Bronze + manifest → deterministic Silver output
- **Reproducibility**: Features deterministically built from Bronze + CONFIG (after best-config JSON overrides)
- **Event-time first**: All records carry `ts_event_ns` and `ts_recv_ns`
- **Adaptive inference**: Features computed on level engagement; ML updates by distance-to-level + triggers; 250ms stream holds last inference
- **Multi-window lookback**: 1-20 minute windows encode "setup" across timescales

---

## Directory Structure

```
data/
├── raw/                               # Stage 0: Source data (Databento DBN)
│   ├── dbn/trades/, dbn/mbp10/        # ES futures (GLBX.MDP3)
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
│   │       └── es_pipeline/           # ES system (16 stages, 182 columns: 10 identity + 108 engineered + 64 labels)
│   │           ├── manifest.yaml      # Feature config (from CONFIG, post override JSON)
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

### Historical Data Ingestion

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

### Batch Feature Engineering Pipeline (Event-Driven Model)

```
Bronze (ES futures + ES options, front-month filtered)
    ↓
[ES Pipeline - 16 stages]
    ↓
Stage 1-4: Load & Sessionize
  - LoadBronzeStage (ES trades/MBP-10/options)
  - BuildOHLCVStage (1min, 2min bars from ES)
  - SessionizeStage (compute minutes_since_open relative to 09:30 ET)
    ↓
Stage 5-6: Level Universe
  - GenerateLevelsStage (6 level kinds: PM/OR high/low + SMA_200/400)
  - DetectInteractionZonesStage (event-driven zone entry, not candle-gated)
    ↓
Stage 7-13: Physics Features (Multi-Window + Single-Window)
  - ComputePhysicsStage (barrier/tape/fuel from engines)
  - ComputeMultiWindowKinematicsStage (velocity/accel/jerk × [1,3,5,10,20]min)
  - ComputeMultiWindowOFIStage (integrated OFI × [30,60,120,300]s)
  - ComputeBarrierEvolutionStage (depth changes × [1,3,5]min)
  - ComputeLevelDistancesStage (signed distances to structural levels)
  - ComputeGEXFeaturesStage (gamma exposure within ±5/±10/±15 points across listed strikes per CME schedule)
  - ComputeForceMassStage (F=ma validation)
    ↓
Stage 14: Approach Context
  - ComputeApproachFeaturesStage (approach metrics + session timing)
    ↓
Stage 15: Label Outcomes
  - LabelOutcomesStage (triple-barrier with volatility-scaled barrier)
    ↓
Stage 16: Filter to RTH
  - FilterRTHStage (09:30-13:30 ET)
    ↓
Silver Features (RTH only 09:30-13:30 ET)
    silver/features/es_pipeline/
        ├── manifest.yaml (records CONFIG snapshot after JSON overrides)
        ├── validation.json (quality metrics)
        └── date=YYYY-MM-DD/*.parquet (~15-25 events/day)
    ↓
[GoldCurator] → Promote to Gold
    ↓
Gold Training (production ML dataset)
    gold/training/signals_v2_multiwindow.parquet
    ↓
    STAGE 2: Model Training Hyperopt
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

```
Live Databento Feed (ES futures + ES options)   [NOT YET IMPLEMENTED]
    ↓ [Ingestor with live client]
NATS (market.futures.*, market.options.*)       [WORKING - tested with replay]
    ├─→ [BronzeWriter] → Bronze (all hours, append-only)     [WORKING]
    └─→ [Core Service] → 250ms physics stream; ML inference adaptive to in-play distance/triggers  [WORKING]
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

**Data Sources**:
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
**Schema**: `SilverFeaturesESPipelineV1` (182 columns: 10 identity + 108 engineered features + 64 labels; enforced) - see `backend/src/common/schemas/silver_features.py`  
**Time Coverage**: RTH only (09:30-13:30 ET)  
**Semantics**: Reproducible transformations (Bronze + CONFIG-from-JSON → deterministic Silver)  
**Retention**: Overwrite when CONFIG changes (no versioning needed)

**Inference Model**: **EVENT-DRIVEN** (adaptive cadence)
- Features computed on level engagement (zone entry) for all 6 levels
- ML inference cadence adapts by distance-to-level (z) and event triggers
- Physics telemetry still updates at 250ms for UI; ML probabilities are latched between updates

**Feature Set** (108 engineered features + 64 label columns):
- **Physics (barrier/tape/fuel)**: 11 features (barrier_state, barrier_delta_liq, barrier_replenishment_ratio, wall_ratio, tape_imbalance, tape_buy_vol, tape_sell_vol, tape_velocity, sweep_detected, fuel_effect, gamma_exposure)
- **Kinematics**: 19 features (velocity/accel/jerk at 1/3/5/10/20min + momentum_trend at 3/5/10/20min)
- **OFI**: 9 features (ofi + ofi_near_level at 30/60/120/300s + ofi_acceleration)
- **Barrier Evolution**: 7 features (barrier_delta + barrier_pct_change at 1/3/5min + barrier_depth_current)
- **GEX**: 15 features (gex_above/below + call/put splits across 1/2/3 bands + asymmetry/ratio/net)
- **Structural**: 16 features (dist_to_* + *_atr for 6 levels, dist_to_tested_level, level_stacking_{2,5,10}pt)
- **Force/Mass**: 3 features (predicted_accel, accel_residual, force_mass_ratio)
- **Approach Context**: 5 features (atr, approach_velocity, approach_bars, approach_distance, prior_touches)
- **Session Timing**: 2 features (minutes_since_open, bars_since_open)
- **Sparse Transforms**: 4 features (wall_ratio_nonzero/log, barrier_delta_liq_nonzero/log)
- **Normalized Features**: 11 features (spot, distance_signed + pct/atr, dist_to_pm_high_pct, dist_to_pm_low_pct, dist_to_sma_200_pct, dist_to_sma_400_pct, approach_distance_{atr,pct}, level_price_pct)
- **Attempt Clustering**: 6 features (attempt_index, attempt_cluster_id, barrier_replenishment_trend, barrier_delta_liq_trend, tape_velocity_trend, tape_imbalance_trend)
- **Labels**: 64 columns (2/4/8min horizons + primary copies)

**Key Capabilities**:
- Deterministic: Bronze + CONFIG-from-JSON → reproducible Silver output
- Hyperopt writes best-config JSON; rebuild when the JSON changes
- Manifest records exact CONFIG snapshot used for each build (post JSON overrides)
- Multi-window features (1-20min lookback) for kNN retrieval

**Implementation**: `SilverFeatureBuilder` class (`src/lake/silver_feature_builder.py`) + `src/pipeline/pipelines/es_pipeline.py`

### Gold (Stage 3)

**Purpose**: Production-ready ML datasets and evaluation results  
**Format**: Parquet with ZSTD compression  
**Schema**: `GoldTrainingESPipelineV1` (182 columns, identical to Silver but curated) - see `backend/src/common/schemas/gold_training.py`  
**Semantics**: Curated, validated, production-quality  
**Retention**: Keep production datasets permanently

**Production Workflow**:

**Step 1: Generate Best-Config JSON (Hyperopt)**
- Required if `data/ml/experiments/zone_opt_v1_best_config.json` does not exist
- Output: Best-config JSON (set `CONFIG_OVERRIDE_PATH` if you use a non-default study name)
- Metrics: kNN-5 purity, Precision@80% (Silhouette logged only)

**Step 2: Build Features**
- Input: Bronze (ES futures + ES options)
- Config: CONFIG loaded from best-config JSON
- Output: `silver/features/es_pipeline/` 
- Command: `uv run python -m src.lake.silver_feature_builder --pipeline es_pipeline`

**Step 3: Train Model**
- Input: `silver/features/es_pipeline/` (curated to Gold)
- Search: XGBoost params, feature selection, kNN blend
- Output: `ml/production/xgb_prod.pkl` + `knn_index.faiss`
- Metrics: Precision@80% > 90%

**Sources**:
- `gold/training/`: Promoted from `silver/features/es_pipeline/`
- `gold/streaming/`: Real-time signals from Core Service (event-driven inference)
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
| 6 | `DetectInteractionZonesStage` | Event-driven zone entry | `signals_df` |
| **Physics Features** |
| 7 | `ComputePhysicsStage` | Barrier/tape/fuel from engines | +11 |
| 8 | `ComputeMultiWindowKinematicsStage` | Velocity/accel/jerk × [1,3,5,10,20]min + momentum_trend | +19 |
| 9 | `ComputeMultiWindowOFIStage` | OFI + ofi_near_level × [30,60,120,300]s + ofi_acceleration | +9 |
| 10 | `ComputeBarrierEvolutionStage` | Barrier depth changes × [1,3,5]min | +7 |
| 11 | `ComputeLevelDistancesStage` | Distances to structural levels + stacking | +16 |
| 12 | `ComputeGEXFeaturesStage` | Gamma exposure within ±5/±10/±15 points (listed strikes per CME schedule) | +15 |
| 13 | `ComputeForceMassStage` | F=ma validation | +3 |
| **Approach & Labels** |
| 14 | `ComputeApproachFeaturesStage` | Approach + timing + normalization + clustering | +28 |
| 15 | `LabelOutcomesStage` | Triple-barrier (vol-scaled barrier, multi-horizon) | +64 labels |
| 16 | `FilterRTHStage` | Filter to RTH 09:30-13:30 ET | Final dataset |

**Total**: 16 stages, 182 columns (10 identity + 108 engineered + 64 labels)

### Pipeline Versions

### Pipeline: ES Futures + ES Options Physics (16 stages)


**Pipeline Flow**: LoadBronze → BuildOHLCV → InitMarketState → GenerateLevels → DetectInteractionZones → ComputePhysics → MultiWindow stages → LabelOutcomes → FilterRTH
**Implementation**: See `backend/src/pipeline/pipelines/es_pipeline.py` for complete 16-stage pipeline definition.
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

### Feature Categories (Core Physics Features)

**Multi-Window Kinematics** (19 features - LEVEL FRAME):
- `velocity_{1,3,5,10,20}min` - Velocity at 5 timescales (ES pts/min) = 5 features
- `acceleration_{1,3,5,10,20}min` - Acceleration at 5 timescales (pts/min²) = 5 features
- `jerk_{1,3,5,10,20}min` - Jerk at 5 timescales (pts/min³) = 5 features
- `momentum_trend_{3,5,10,20}min` - Velocity trend proxy = 4 features
- **Purpose**: Encode "setup shape" - fast aggressive vs slow decelerating approach

**Multi-Window OFI** (9 features - ORDER FLOW):
- `ofi_{30,60,120,300}s` - Cumulative OFI at 4 windows
- `ofi_near_level_{30,60,120,300}s` - OFI within level band at 4 windows
- `ofi_acceleration` - Short-term OFI change rate
- **Purpose**: Capture order flow pressure dynamics across timescales

**Barrier Evolution** (7 features - LIQUIDITY DYNAMICS):
- `barrier_delta_{1,3,5}min` - Depth change (thinning vs thickening)
- `barrier_pct_change_{1,3,5}min` - Percent depth change
- `barrier_depth_current` - Current depth at level (ES contracts)
- **Purpose**: Detect "thinning barrier" (bullish BREAK) vs "thickening barrier" (bullish BOUNCE)

**GEX Features** (15 features - GAMMA EXPOSURE):
- `gex_above_{1,2,3}strike` / `gex_below_{1,2,3}strike` - Net band exposure
- `call_gex_above_{1,2,3}strike` / `put_gex_below_{1,2,3}strike` - Call/put band splits
- `gex_asymmetry`, `gex_ratio`, `net_gex_2strike` - Summary metrics
- **Strike listing**: CME schedule varies by moneyness/time-to-expiry (5/10/50/100-point intervals; dynamic 5-point additions); bands aggregate across actually listed strikes
- **Purpose**: Detect pinning/chop near strikes (NOT primary break/bounce driver)
- **Weight**: 0.3x in ML models (gamma is 0.04-0.17% of ES volume - Cboe/SpotGamma studies)

**Level Distances** (16 features - STRUCTURAL CONTEXT):
- `dist_to_{pm_high,pm_low,or_high,or_low,sma_200,sma_400}` + `_atr` variants
- `dist_to_tested_level`
- `level_stacking_{2,5,10}pt`
- **Purpose**: Position within structural level framework

**Force-Mass** (3 features - PHYSICS VALIDATION):
- `predicted_accel` - F/m (OFI / barrier_delta_liq)
- `accel_residual` - actual - predicted (hidden momentum?)
- `force_mass_ratio` - Diagnostic ratio
- **Purpose**: Cross-validate force and mass proxies with observed acceleration

**Approach + Session Context** (7 features):
- `atr` - Average True Range (volatility normalization)
- `approach_velocity`, `approach_bars`, `approach_distance`, `prior_touches`
- `minutes_since_open`, `bars_since_open`

**Labels** (Triple-Barrier):
- `outcome` (BREAK, BOUNCE, CHOP, UNDEFINED) - First volatility-scaled barrier hit after level touch
- `strength_signed` - Signed excursion magnitude
- `time_to_break/bounce_{1,2}` - Time to dynamic thresholds (fractional + full barrier)
- `tradeable_1/2` - Threshold reached flags
- 2/4/8min horizons + primary copy = 64 label columns

---

## Configuration

Pipeline behavior controlled by `backend/src/common/config.py`
Defaults to `data/ml/experiments/zone_opt_v1_best_config.json`; override with `CONFIG_OVERRIDE_PATH` if needed.

**Key Configuration Concepts**:

| Category | Purpose |
|----------|---------|
| **Instruments** | ES futures (spot + liquidity) + ES 0DTE options (gamma) from CME GLBX.MDP3 |
| **RTH Window** | 09:30-13:30 ET (first 4 hours) - when to generate training events |
| **Premarket** | 04:00-09:30 ET - for PM_HIGH/PM_LOW calculation only |
| **Strike Spacing** | CME strike listing varies by moneyness/time-to-expiry (5/10/50/100-point intervals; dynamic 5-point additions); aggregate listed strikes within point bands |
| **Interaction Zones** | MONITOR_BAND (event detection), TOUCH_BAND (precise contact) - **TUNABLE via hyperopt** |
| **Outcome Labels** | Volatility-scaled barrier (vol window + horizon) - **TUNABLE via hyperopt** |
| **Multi-Window Lookback** | 1-20min kinematics, 30s-5min OFI, 1-5min barrier - encode setup across timescales |
| **Base Physics Windows** | W_b/W_t/W_g - engine lookback windows - **TUNABLE via hyperopt** |
| **Level Selection** | use_pm/use_or/use_sma_200/use_sma_400 - **TUNABLE via hyperopt** |

**Inference Model**: Event-driven (adaptive cadence)

**Hyperopt** (Stage 1): 29 tunable parameters including zone widths, windows, thresholds, level selection  
**Fixed**: Strike listing schedule (external CME rules; intervals vary; dynamic additions), RTH window (09:30-13:30 ET)

---

## Hyperparameter Optimization

**Philosophy**: Optimize CONFIG parameters (via best-config JSON), then build features once with best config.

**Stage 1: Zone/Window Optimization** (Required if no best-config JSON exists):
```bash
cd backend

# Or use real data for accurate results
uv run python scripts/run_zone_hyperopt.py \
  --start-date 2025-11-02 \
  --end-date 2025-11-30 \
  --n-trials 1000
```

**Searches 29 parameters**:
- Zone widths (MONITOR_BAND, TOUCH_BAND, per-level)
- Outcome thresholds (vol window, barrier scale, horizon)
- Physics windows (W_b, W_t, W_g)
- Multi-window selection (which timeframes to use)
- Level selection (use_pm, use_or, use_sma_200/400)

**Optimizes for**: kNN-5 purity (50%) + Precision@80% (30%) + Feature variance (20%)

**Output**: Best CONFIG parameters logged to MLflow + JSON at `data/ml/experiments/<study_name>_best_config.json`
- Example: `MONITOR_BAND=3.8`, `W_t=45s`, `use_sma_200=False`

**Action**: Run hyperopt to write the JSON (default path), then rebuild (no manual edits). Set `CONFIG_OVERRIDE_PATH` if you use a non-default study name.

---

### Stage 2: Model Training

**Input**: Features from `silver/features/es_pipeline/` (built with CONFIG loaded from best-config JSON)

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

### Sparse is Better (kNN Retrieval)
**Counter-intuitive**: 10-20 high-quality events/day > 100 noisy events/day
- kNN retrieval accumulates events over time (60 days × 15 events = 900 examples)
- Precision > Recall (better to sit out than be wrong)
- Hyperopt optimizes for physics distinctiveness, not event density

---

## Adaptive Inference Architecture

**Inference Model**: ML updates are event-driven; physics telemetry updates every 250ms.

**Cadence tiers (per level)**:
- Far (z > 3): infer every 10s
- Approaching (1.5 < z ≤ 3): infer every 2s
- Engaged (z ≤ 1.5): infer every 250–500ms

**Hard triggers override timers**:
- Enter/exit monitor band
- Level cross
- Tape imbalance shock / sweep
- Barrier regime change (replenish → consume)
- Gamma sign flip

### Feature Computation Strategy

1. Compute LevelState for all 6 levels on each snap.
2. Evaluate distance-to-level (z) + triggers to decide ML refresh.
3. Latch ML probabilities between updates; gateway merges with physics telemetry.

**Data Volume**:
- Events only when within monitor band; typically ~10–25/day per level.
- ML updates are sparse away from levels, dense only in the decision zone.

---

## Performance

**Apple M4 Silicon Optimized**:
- All operations use NumPy broadcasting and vectorized pandas
- Batch processing of all candles simultaneously
- Multi-window features computed with rolling operations (efficient!)
- Memory-efficient chunked processing

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
8. **Adaptive inference**: Event-driven ML cadence; physics telemetry at 250ms with latched probabilities
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

## kNN Retrieval System (Neuro-Hybrid)

### The Use Case

**Query**: "Price approaching PM_HIGH with f features"

**kNN Retrieval**: Find k=5 most similar past events in physics feature space

**Prediction**: "4/5 similar setups resulted in BREAK → 80% confidence BREAK"

### Why Multi-Window Features Are Critical

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

**Hyperopt Framework**:
- **Stage 1 objective**: `src/ml/zone_objective.py` (29-parameter search)
- **Config override**: `src/common/utils/config_override.py`
- **Hyperopt runner**: `scripts/run_zone_hyperopt.py`
- **Grid search**: `scripts/run_zone_grid_search.py`
