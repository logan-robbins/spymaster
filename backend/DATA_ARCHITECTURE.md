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
- **Reproducibility**: Features deterministically built from Bronze + CONFIG
- **Event-time first**: All records carry `ts_event_ns` and `ts_recv_ns`
- **Episode-based inference**: Continuous inference at 250-500ms while within touch zone (4 points); episodes group continuous touches; retrieval votes on episode-collapsed neighbors
- **Multi-window lookback**: 1-20 minute windows encode "setup shape" before touch for kNN retrieval

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
│   │   └── options/trades/underlying=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │         └── (front-month contract only, 0DTE filtered, CME GLBX.MDP3)
│   │
│   ├── silver/                        # Stage 2: Feature engineering output
│   │   └── features/
│   │       └── es_pipeline/           # ES system (16 stages, 182 columns: 10 identity + 108 engineered + 64 labels)
│   │           ├── manifest.yaml      # Feature config snapshot from CONFIG
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
    └─→ ES Options (trades)
            ↓
    [download_es_options.py + DBNReader]
            ↓
    Front-Month Filtering:
    - ES futures: Volume-dominant contract selection (ContractSelector)
    - ES options: Filter to same contract (e.g., ESZ5 only)
    - 0DTE filter: exp_date == session_date
            ↓
Bronze (all hours 00:00-23:59 UTC, immutable, Parquet+ZSTD)
    ├── futures/trades/symbol=ES/
    ├── futures/mbp10/symbol=ES/
    └── options/trades/underlying=ES/
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
        ├── manifest.yaml (records CONFIG snapshot)
        ├── validation.json (quality metrics)
        └── date=YYYY-MM-DD/*.parquet (dense in-zone snapshots; ~20-50 episode-snapshots/day)
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
    └─→ [Core Service] → Episode-based inference while in touch zone (4 points)  [WORKING]
            ├─→ Load model (xgb_prod.pkl + knn_index)
            ├─→ When abs(price - level) <= 4.0:
            │     - Track episode_id (continuous touch episode)
            │     - Compute 20-min PRE-TOUCH state snapshot
            │     - Compute dwell/touch-intensity features
            │     - kNN retrieval: Find 50-200 similar past EPISODES (episode-collapsed)
            │     - Distance-weighted vote on outcomes
            │     - Apply EWMA smoothing
            │     - Expose similarity strength metric
            │     - Infer continuously at 250-500ms cadence
            └─→ Predict: "45/50 similar episodes BROKE → 90% confidence"
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

**Writer**: `scripts/download_es_options.py` (batch) + `src/io/bronze.py` (streaming)  
**Reader**: `src/io/bronze.py` (BronzeReader) and `src/pipeline/utils/duckdb_reader.py` (with `front_month_only=True`)

### Silver (Stage 2)

**Purpose**: Feature engineering output  
**Format**: Parquet with ZSTD compression  
**Schema**: `SilverFeaturesESPipelineV1` (~200 columns: 10 identity + ~125 engineered features + 64 labels; enforced) - see `backend/src/common/schemas/silver_features.py`  
**Time Coverage**: RTH only (09:30-13:30 ET)  
**Semantics**: Reproducible transformations (Bronze + CONFIG-from-JSON → deterministic Silver)  
**Retention**: Overwrite when CONFIG changes (no versioning needed)

**Inference Model**: **CONTINUOUS EPISODE-BASED**
- **Touch zone**: Single 4-point band (TOUCH_BAND = MONITOR_BAND = 4.0)
- **Episodes**: Continuous time spent within touch zone; grouped by `episode_id`
- **Inference cadence**: 250-500ms continuously while in-zone (engaged)
- **Direction-of-travel**: Computed from pre-touch signed distance slope (approaching from above vs below)
- **kNN retrieval**: Query on 20-minute pre-touch state snapshot; votes episode-collapsed to avoid redundancy
- **Smoothing**: EWMA on displayed probabilities; expose similarity strength for "no close analogs" detection

**Feature Set** (~125 engineered features + 64 label columns):
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
- **Episode Identity**: 2 features (episode_id, approach_direction)
- **Dwell/Persistence**: 4 features (time_in_zone_total, time_in_zone_last_60s, time_in_zone_last_300s, fraction_in_zone_last_20m)
- **Touch Intensity**: 4 features (touch_count_last_60s, touch_count_last_5m, boundary_cross_rate, min_abs_distance_in_episode)
- **Labels**: 64 columns (2/4/8min horizons + primary copies; applied at episode level)

**Key Capabilities**:
- Deterministic: Bronze + CONFIG → reproducible Silver output
- Manifest records exact CONFIG snapshot used for each build
- Multi-window features (1-20min lookback) for kNN retrieval
- Episode-based labeling prevents label leakage

**Implementation**: `SilverFeatureBuilder` class (`src/io/silver.py`) + `src/pipeline/pipelines/es_pipeline.py`

### Gold (Stage 3)

**Purpose**: Production-ready ML datasets and evaluation results  
**Format**: Parquet with ZSTD compression  
**Schema**: `GoldTrainingESPipelineV1` (~200 columns, identical to Silver but curated) - see `backend/src/common/schemas/gold_training.py`  
**Semantics**: Curated, validated, production-quality; episode-level labels applied to all in-zone snapshots  
**Retention**: Keep production datasets permanently

**Production Workflow**:

**Step 1: Build Features**
- Input: Bronze (ES futures + ES options)
- Config: Fixed CONFIG from `src/common/config.py`
- Output: `silver/features/es_pipeline/` 
- Command: `uv run python -m src.lake.silver_feature_builder --pipeline es_pipeline`

**Step 2: Train Model (with Hyperopt)**
- Input: `silver/features/es_pipeline/` (curated to Gold)
- Search: XGBoost params, feature selection, kNN blend
- Output: `ml/production/xgb_prod.pkl` + `knn_index.faiss`
- Metrics: Precision@80% > 90%

**Sources**:
- `gold/training/`: Promoted from `silver/features/es_pipeline/`
- `gold/streaming/`: Real-time signals from Core Service (event-driven inference)
- `gold/evaluation/`: Backtest and validation results

**Implementation**: `GoldCurator` class (`src/io/gold.py`)

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
| 6 | `DetectInteractionZonesStage` | Episode-based touch detection (4pt band) | `signals_df` with `episode_id` |
| **Physics Features** |
| 7 | `ComputePhysicsStage` | Barrier/tape/fuel from engines | +11 |
| 8 | `ComputeMultiWindowKinematicsStage` | Velocity/accel/jerk × [1,3,5,10,20]min + momentum_trend | +19 |
| 9 | `ComputeMultiWindowOFIStage` | OFI + ofi_near_level × [30,60,120,300]s + ofi_acceleration | +9 |
| 10 | `ComputeBarrierEvolutionStage` | Barrier depth changes × [1,3,5]min | +7 |
| 11 | `ComputeLevelDistancesStage` | Distances to structural levels + stacking | +16 |
| 12 | `ComputeGEXFeaturesStage` | Gamma exposure within ±5/±10/±15 points (listed strikes per CME schedule) | +15 |
| 13 | `ComputeForceMassStage` | F=ma validation | +3 |
| **Approach & Labels** |
| 14 | `ComputeApproachFeaturesStage` | Approach + timing + normalization + clustering + episode/dwell features | +38 |
| 15 | `LabelOutcomesStage` | Triple-barrier (vol-scaled barrier, multi-horizon) | +64 labels |
| 16 | `FilterRTHStage` | Filter to RTH 09:30-13:30 ET | Final dataset |

**Total**: 16 stages, ~200 columns (10 identity + ~125 engineered + 64 labels)

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

**Episode Identity** (2 features - EPISODE TRACKING):
- `episode_id` - Unique identifier for continuous touch episodes
- `approach_direction` - Direction of travel at episode start (from_above=-1, from_below=+1)
- **Purpose**: Group continuous touches; define BREAK/BOUNCE semantics

**Dwell/Persistence** (4 features - TIME IN ZONE):
- `time_in_zone_total` - Seconds since episode start
- `time_in_zone_last_60s` - Dwell time in last minute
- `time_in_zone_last_300s` - Dwell time in last 5 minutes
- `fraction_in_zone_last_20m` - Stickiness metric (how much of 20min was spent near level)
- **Purpose**: Capture level "holding power" - persistent vs transient touches

**Touch Intensity** (4 features - MICRO-TEST DYNAMICS):
- `touch_count_last_60s` - Number of touches in last minute
- `touch_count_last_5m` - Number of touches in last 5 minutes
- `boundary_cross_rate` - Oscillation frequency around band edge (Hz)
- `min_abs_distance_in_episode` - Deepest probe into level (ES points)
- **Purpose**: Detect "pressing" behavior - repeated tests signal potential break

**Labels** (Triple-Barrier, Episode-Level):
- `outcome` (BREAK, BOUNCE, CHOP, UNDEFINED) - Determined at episode end based on exit direction and barrier hit
- `strength_signed` - Signed excursion magnitude beyond barrier
- `time_to_break/bounce_{1,2}` - Time from episode start to dynamic thresholds (fractional + full barrier)
- `tradeable_1/2` - Threshold reached flags
- 2/4/8min horizons + primary copy = 64 label columns
- **Critical**: Same outcome label applied to ALL snapshots within an episode (prevents label leakage)

---

## Configuration

Pipeline behavior controlled by `backend/src/common/config.py` (single source of truth for all parameters).

**Key Configuration Concepts**:

| Category | Purpose |
|----------|---------|
| **Instruments** | ES futures (spot + liquidity) + ES 0DTE options (gamma) from CME GLBX.MDP3 |
| **RTH Window** | 09:30-13:30 ET (first 4 hours) - when to generate training events |
| **Premarket** | 04:00-09:30 ET - for PM_HIGH/PM_LOW calculation only |
| **Strike Spacing** | CME strike listing varies by moneyness/time-to-expiry (5/10/50/100-point intervals; dynamic 5-point additions); GEX features aggregate gamma within ±1/±2/±3 strikes (typically 5pt spacing for ES 0DTE ATM) |
| **Touch Zone** | TOUCH_BAND = MONITOR_BAND = 4.0 (single engaged zone concept) |
| **Episode Boundaries** | EXIT_HYST + EXIT_DWELL_SEC (prevent spurious episode breaks) |
| **Outcome Labels** | Volatility-scaled barrier (vol window + horizon); applied at episode level |
| **Multi-Window Lookback** | 1-20min kinematics, 30s-5min OFI, 1-5min barrier - encode "setup shape" for kNN retrieval |
| **Base Physics Windows** | W_b/W_t/W_g - engine lookback windows (see `src/common/config.py`) |
| **Level Selection** | use_pm/use_or/use_sma_200/use_sma_400 (see `src/common/config.py`) |

**Inference Model**: Continuous episode-based (250-500ms while in-zone)

**Configuration**: All parameters defined in `src/common/config.py`  
**Fixed External Constraints**: Strike listing schedule (CME rules), RTH window (09:30-13:30 ET)

---

## Model Training with Hyperparameter Optimization

**Philosophy**: Fixed CONFIG for feature engineering; optimize only model hyperparameters.

**Input**: Features from `silver/features/es_pipeline/` (built with fixed CONFIG)

```bash
# Train XGBoost + build kNN index
uv run python -m src.ml.boosted_tree_train \
  --features gold/training/signals_production.parquet

# Output: ml/production/xgb_prod.pkl + knn_index.faiss
```

**Hyperopt searches**:
- XGBoost hyperparameters (learning rate, max depth, subsample, etc.)
- Feature selection (which engineered features to use)
- kNN blend weight (how much to weight kNN vs XGBoost)
- kNN retrieval parameters (K, distance metric, episode-collapse settings)

**Optimization target**: Precision@80% > 90% (high confidence = high accuracy)

**Why this workflow**: Feature engineering is physics-based with clear semantic meaning. Tuning zone widths and windows would obscure the underlying dynamics. Instead, we fix the physics parameters and let the model learn which features are predictive.

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

### Dense Snapshots, Episode-Collapsed Retrieval
**Philosophy**: Retain dense in-zone snapshots for state evolution; retrieval votes on episode-collapsed neighbors
- **Training data**: Many snapshots per episode capture dwell behavior and touch intensity
- **kNN retrieval**: Episode-collapsed voting prevents redundancy (independent historical attempts only)
- **Data volume**: 60 days × 20 episodes/day × 6 levels = ~7,200 independent episode examples
- **Why this works**: Similarity voting on episodes (not ticks) while features capture within-episode dynamics
- Precision > Recall (better to sit out than be wrong); similarity strength metric prevents false confidence

---

## Episode-Based Inference Architecture

**Inference Model**: Continuous inference at 250-500ms while within touch zone; episode-collapsed kNN retrieval.

**Touch Zone**: Single 4-point band (TOUCH_BAND = MONITOR_BAND = 4.0)
- When `abs(price - level) <= 4.0`, we are **in-zone** (engaged)
- Continuous inference at 250-500ms cadence while engaged
- Physics telemetry updates at 250ms; ML probabilities smoothed with EWMA

**Episode Definition**:
- **Episode start**: `abs(price - level) <= 4.0` becomes true
- **Episode end**: `abs(price - level) >= 4.0 + EXIT_HYST` for at least `EXIT_DWELL_SEC`
- All touches within an episode share the same `episode_id`
- Episodes group continuous "pressing" behavior at a level

**Direction-of-Travel**:
- Computed from pre-touch signed distance slope: `x(t) = price(t) - level`
- **Approaching from above**: `x > 0` and `dx/dt < 0` (coming down)
- **Approaching from below**: `x < 0` and `dx/dt > 0` (coming up)
- Used to define BREAK vs BOUNCE outcomes

### kNN Retrieval Strategy

**Query Vector**: 20-minute pre-touch state snapshot
- Multi-window kinematics (1-20min velocity/accel/jerk)
- Multi-window OFI (30-300s)
- Barrier evolution (1-5min)
- Session timing, distances, approach context
- **Critical**: Features computed from PRE-TOUCH windows only

**Episode-Collapsed Retrieval** (prevents redundancy):
1. Retrieve `K_raw = 500` nearest neighbors
2. Group by `(session_date, level_kind, episode_id)`
3. Keep only **single closest** neighbor per episode
4. Take top `K = 50-200 episodes` (not ticks)
5. Distance-weighted vote on outcomes (BREAK/BOUNCE/CHOP)
6. Apply EWMA smoothing to displayed probabilities
7. Expose **similarity strength** metric (nearest_distance / n_close) for "no close analogs" detection

**Data Volume**:
- Dense snapshots while in-zone (250-500ms × episode duration)
- Episode-level labeling prevents label leakage
- Retrieval votes on **independent historical episodes**, not redundant ticks
- Typical: 20-50 episode-snapshots per level per day (many touches per episode)

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
**File**: `src/io/silver.py`  
**Purpose**: Build versioned Silver feature sets from Bronze  
**Key Methods**: `build_feature_set()`, `list_versions()`, `load_features()`, `register_experiment()`

### GoldCurator
**File**: `src/io/gold.py`  
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
**File**: `src/io/bronze.py`  
**Purpose**: Write/read Bronze Parquet files from NATS streams  
**Note**: BronzeWriter subscribes to NATS; BronzeReader uses DuckDB for efficient queries

### DuckDBReader
**File**: `src/pipeline/utils/duckdb_reader.py`  
**Purpose**: Efficient Bronze data reading with downsampling for MBP-10  
**Key Methods**: `read_futures_trades()`, `read_futures_mbp10_downsampled()`, `get_warmup_dates()`

---

## Episode Labeling Process

**Episode Definition**: Continuous time spent within touch zone (4 points)

**Episode Lifecycle**:
1. **Episode Start**: `abs(price - level) <= TOUCH_BAND` becomes true
   - Assign unique `episode_id`
   - Compute `approach_direction` from pre-touch signed distance slope:
     - `x(t) = price(t) - level`
     - From above: `x > 0` and `dx/dt < 0` → `approach_direction = -1`
     - From below: `x < 0` and `dx/dt > 0` → `approach_direction = +1`

2. **Episode Active**: Continuous inference at 250-500ms
   - Compute 20-minute pre-touch state snapshot on each tick
   - Compute dwell/touch-intensity features (cumulative within episode)
   - Query kNN (episode-collapsed) for similar past episodes
   - Apply EWMA smoothing to displayed probabilities
   - Latch probabilities between updates

3. **Episode End**: `abs(price - level) >= TOUCH_BAND + EXIT_HYST` for `EXIT_DWELL_SEC`
   - Determine outcome based on exit direction and barrier hits:
     - **BREAK**: Exit on opposite side from approach AND hit break barrier
     - **BOUNCE**: Exit on same side as approach AND hit bounce barrier
     - **CHOP**: Neither barrier hit within horizon
   - Apply this **same outcome** to ALL snapshots within the episode
   - This prevents label leakage (all in-episode snapshots get same label)

**Why Episode-Level Labels Matter**:
- Prevents label leakage: Can't train on tick T and predict tick T+1 within same episode
- Matches operational reality: All inference during episode uses PRE-TOUCH features
- Enables dwell/touch-intensity features without contaminating target
- kNN retrieval votes on independent episodes (not redundant ticks from same attempt)

**Example Episode**:
```
t=0s:    Price 5950.00, level PM_HIGH 5950.25 → Episode starts (from below)
t=0.5s:  Price 5950.50 → In-zone, infer, dwell features update
t=1.0s:  Price 5949.75 → Still in-zone, infer, touch_count increments
...
t=15s:   Price 5952.00 (> 5950.25 + EXIT_HYST) for EXIT_DWELL_SEC → Episode ends
Outcome: BREAK (exited opposite side + hit barrier)
Label:   All 30 snapshots (15s × 2Hz) get outcome=BREAK
```

---

## Critical Invariants

1. **Bronze is append-only**: Never mutate or delete
2. **Silver is derived**: Always regeneratable from Bronze
3. **Event-time ordering**: All files sorted by `ts_event_ns`
4. **Idempotency**: Same Bronze + manifest → same Silver
5. **Front-month purity**: ES futures AND ES options on SAME contract (CRITICAL!)
6. **0DTE filtering**: ES options with `exp_date == session_date` only
7. **RTH filtering**: Silver/Gold contain only 09:30-13:30 ET signals (v1 = first 4 hours)
8. **Episode-based inference**: Continuous 250-500ms inference while in touch zone (4 points); episode_id groups continuous touches
9. **Multi-window lookback**: 1-20min for kinematics, 30s-5min for OFI/barrier; PRE-TOUCH only for kNN retrieval
10. **Episode-collapsed kNN**: Retrieve episodes (not ticks) to avoid redundancy; distance-weighted vote on independent historical attempts
11. **Episode-level labels**: Outcome (BREAK/BOUNCE/CHOP) determined at episode end; applied to all snapshots within episode
12. **Partition boundaries**: Date/hour aligned to UTC
13. **Compression**: ZSTD level 3 for all tiers
14. **No conversion**: ES futures = ES options (same index points!)

### Front-Month Purity Validation

**Why it matters**: During roll periods, TWO ES contracts trade simultaneously
- Example: Dec 2025 (ESZ5) rolling to Mar 2026 (ESH6)
- If we mix both: Creates artificial "ghost walls" in our data
- ES options on ESZ5 ≠ ES options on ESH6

**Implementation**: See `backend/src/common/utils/contract_selector.py` (ContractSelector) and `backend/src/common/utils/bronze_qa.py` (BronzeQA) for front-month selection and quality gate enforcement (60% dominance threshold).


---

## kNN Retrieval System (Episode-Collapsed)

### The Use Case

**Query**: "Price in touch zone at PM_HIGH; 20-minute pre-touch state shows fast aggressive approach"

**kNN Retrieval**: Find k=50-200 most similar past **episodes** in physics feature space (episode-collapsed)

**Prediction**: "45/50 similar episodes resulted in BREAK → 90% confidence BREAK"

### Why Multi-Window Features Are Critical

**Multi-Window Solution**: By encoding velocity at multiple timescales (1min, 3min, 5min, 10min, 20min), we capture the "setup shape" before touch. Fast aggressive approaches (accelerating) show increasing velocity across windows, while decelerating approaches show decreasing velocity. kNN retrieval can distinguish these patterns.

**Critical**: All multi-window features computed from **PRE-TOUCH** data only (backward-looking from touch timestamp).

### kNN Index Structure

**Built from**: `gold/training/signals_v2_multiwindow.parquet` (all in-zone snapshots with episode_id)

**Implementation**: See `backend/src/ml/build_retrieval_index.py` for FAISS index construction using normalized physics features (~20-30 most predictive features including multi-window kinematics, OFI, barrier evolution, GEX, dwell/touch-intensity).

**Episode-Collapsed Retrieval Process**:
1. Query FAISS with current 20-minute state vector → retrieve `K_raw = 500` nearest neighbors
2. Group neighbors by `(session_date, level_kind, episode_id)`
3. Keep only **single closest** neighbor per episode (prevents redundancy)
4. Take top `K = 50-200 episodes` with distance weighting
5. Compute outcome distribution (BREAK/BOUNCE/CHOP) from episode representatives
6. Apply EWMA smoothing for display
7. Expose similarity strength metric: `sim_strength = nearest_distance / n_close`
8. If `sim_strength > threshold`, display "No close analogs" instead of unreliable prediction

**Blending**: kNN probability (25%) + XGBoost prediction (75%) for final confidence

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

**Model Training**:
- **Boosted tree training**: `src/ml/boosted_tree_train.py`
- **kNN index builder**: `src/ml/build_retrieval_index.py`
