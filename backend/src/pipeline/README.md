# Pipeline Module

**Purpose**: Stage-based feature engineering for ES futures + ES 0DTE options  
**Status**: Production (v4.5.0)  
**Primary Consumer**: `SilverFeatureBuilder`

---

## Data Architecture (Medallion)

### 1. Raw Layer
- `.dbn` files from Databento downloads
- Not managed by this pipeline

### 2. Bronze Layer (Normalized, Partitioned)
- **Written by**: `BronzeWriter` (streaming from NATS)
- **Location**: `data/bronze/futures/` and `data/bronze/options/`
- **Partitioning**: `symbol=ES/date=YYYY-MM-DD/hour=HH/`
- **Content**: Normalized trades and MBP-10 snapshots (all trading hours)
- **Format**: Parquet with ZSTD compression

### 3. Silver Layer (Engineered Features)
- **Written by**: This pipeline - Stage 16 (FilterRTH)
- **Location**: `data/silver/features/es_pipeline/version={version}/date=YYYY-MM-DD/`
- **Content**: ~142 engineered features (full signals_df after RTH filtering)
- **Includes**: Physics, kinematics, GEX, approach features, outcome labels
- **Format**: Parquet (signals.parquet)
- **Purpose**: Training data, experimentation, pipeline validation

### 4. Gold Layer (Production Analytics)
- **Written by**: This pipeline - Stages 17-18
- **Purpose**: Production-ready derived analytics for live trading and ML

#### Gold - Episodes (149D Source Vectors)
- **Written by**: Stage 18 (ConstructEpisodes)
- **Location**: `data/gold/episodes/es_level_episodes/version={version}/`
- **Partitions**: 
  - `vectors/date=YYYY-MM-DD/episodes.npy` (149D numpy arrays)
  - `metadata/date=YYYY-MM-DD/metadata.parquet`
  - `sequences/date=YYYY-MM-DD/sequences.npy` (40×4 raw trajectories for Transformer)
- **Purpose**: Source of truth vectors for ML training and analysis

#### Gold - FAISS Indices (32D Compressed)
- **Written by**: `build_all_indices()` from episodes
- **Location**: `data/gold/indices/es_level_indices/`
- **Partitions**: 48 partitions by `{level_kind}/{direction}/{time_bucket}/`
- **Content**: 32D geometry-only vectors (Section F of 149D)
- **Compression**: Via `compressor.pkl` (geometry_only strategy per Phase 4)
- **Purpose**: Live similarity retrieval (calibrated kNN)

#### Gold - Other
- **Streams**: `data/gold/streams/` - Aggregated 2-min bars (Pentaview)
- **Training**: `data/gold/training/` - Curated samples for specific models

**Key Point**: The 149D vector is **Gold** (episodes), not Silver. The 32D compressed vector in FAISS is also **Gold** (indices). Silver contains the ~142-column feature table from Stage 16.

---

## Overview

This pipeline transforms Bronze data (normalized trades/depth) into Silver features (engineered signals) and Gold analytics (episode vectors and FAISS indices).

---

## Architecture

```
src/pipeline/
├── core/
│   ├── stage.py                # BaseStage, StageContext
│   ├── pipeline.py             # Pipeline orchestrator
│   └── checkpoint.py           # Stage checkpoints (resume/inspect)
├── stages/
│   ├── load_bronze.py                      # Stage 0: DuckDB data loading
│   ├── build_spx_ohlcv.py                  # Stages 1-3: OHLCV (1min/10s/2min)
│   ├── init_market_state.py                # Stage 4: MarketState + Greeks
│   ├── generate_levels.py                  # Stage 5: Level universe + dynamic series
│   ├── detect_interaction_zones.py         # Stage 6: Zone entry events
│   ├── compute_physics.py                  # Stage 7: Barrier/Tape/Fuel + Market Tide
│   ├── compute_multiwindow_kinematics.py   # Stage 8: Multi-window kinematics
│   ├── compute_multiwindow_ofi.py          # Stage 9: Multi-window OFI
│   ├── compute_barrier_evolution.py        # Stage 10: Barrier evolution
│   ├── compute_level_distances.py          # Stage 11: Level distances + stacking
│   ├── compute_gex_features.py             # Stage 12: Strike-banded GEX
│   ├── compute_force_mass.py               # Stage 13: F=ma validation features
│   ├── compute_approach.py                 # Stage 14: Approach context + normalization
│   ├── label_outcomes.py                   # Stage 15: First-crossing labels
│   ├── filter_rth.py                       # Stage 16: RTH filtering + schema validation
│   ├── materialize_state_table.py          # Stage 17: 30s state table
│   └── construct_episodes.py               # Stage 18: 149D episode vectors
├── pipelines/
│   ├── bronze_to_silver.py      # Bronze → Silver (stages 0-16)
│   ├── silver_to_gold.py        # Silver → Gold (stages 0-2)
│   ├── pentaview_pipeline.py    # Pentaview streams
│   └── registry.py              # get_pipeline()
└── utils/
    └── duckdb_reader.py         # DuckDB wrapper with downsampling
```

---

## Usage

### CLI

```bash
# Step 1: Bronze → Silver (feature engineering)
uv run python -m scripts.run_pipeline \
  --pipeline bronze_to_silver \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --write-outputs

# Step 2: Silver → Gold (episode construction)
uv run python -m scripts.run_pipeline \
  --pipeline silver_to_gold \
  --date 2025-12-16 \
  --write-outputs

# Parallel mode for date ranges
uv run python -m scripts.run_pipeline \
  --pipeline bronze_to_silver \
  --start 2025-12-01 \
  --end 2025-12-31 \
  --workers 8 \
  --write-outputs
```

### Programmatic

```python
from src.pipeline.pipelines.registry import get_pipeline

# Bronze → Silver (feature engineering)
pipeline = get_pipeline("bronze_to_silver")
signals_df = pipeline.run("2025-12-16", write_outputs=True)

# Silver → Gold (episode construction)
pipeline = get_pipeline("silver_to_gold")
episodes_metadata = pipeline.run("2025-12-16", write_outputs=True)
```

---

## Pipelines (v4.5.0)

**Note**: All stage references use `stage_idx` (0-based). Checkpoints: `data/checkpoints/{pipeline_name}/YYYY-MM-DD/stage_{stage_idx:02d}/`.

### Bronze → Silver Pipeline (17 stages, 0-indexed: 0-16)

**Name**: `bronze_to_silver`  
**Purpose**: Feature engineering  
**Output**: Silver feature table (~142 columns)

0. LoadBronze (ES futures + ES 0DTE options)  
1. BuildOHLCV (1min for ATR/volatility)  
2. BuildOHLCV (10s for high-res physics validation)  
3. BuildOHLCV (2min with warmup for SMA_90/EMA_20)  
4. InitMarketState (market state + Greeks)  
5. GenerateLevels (PM/OR high/low + SMA_90/EMA_20)  
6. DetectInteractionZones (event-driven zone entry)  
7. ComputePhysics (barrier/tape/fuel + **Market Tide**: call_tide, put_tide)  
8. ComputeMultiWindowKinematics (1,2,3,5,10,20min)  
9. ComputeMultiWindowOFI (30,60,120,300s)  
10. ComputeBarrierEvolution (1,2,3,5min)  
11. ComputeLevelDistances (signed distances + stacking)  
12. ComputeGEXFeatures (strike-banded gamma)  
13. ComputeForceMass (F=ma validation)  
14. ComputeApproachFeatures (clustering + trends)  
15. LabelOutcomes (first-crossing: BREAK/REJECT/CHOP, 2/4/8min)  
16. FilterRTH (09:30-12:30 ET + write to Silver)

### Silver → Gold Pipeline (3 stages, 0-indexed: 0-2)

**Name**: `silver_to_gold`  
**Purpose**: Episode vector construction  
**Input**: Silver features from bronze_to_silver  
**Output**: Gold episodes (149D vectors + metadata + sequences)

0. LoadSilverFeatures (read from Silver layer)
1. MaterializeStateTable (30s cadence state for episode construction)
2. ConstructEpisodes (149D vectors with DCT trajectory basis)

**Episode Vectors**: See [../../EPISODE_VECTOR_SCHEMA.md](../../EPISODE_VECTOR_SCHEMA.md) for complete 149D specification.

---

## Pipeline Flow

### 1. Bronze → Silver (Feature Engineering)

**Pipeline**: `bronze_to_silver` (17 stages: 0-16)
- Input: Bronze parquet (trades, depth, options)
- Output: Silver features (signals.parquet, ~142 columns)
- Use case: Feature engineering, experiments, ML training data

### 2. Silver → Gold (Episode Construction)

**Pipeline**: `silver_to_gold` (3 stages: 0-2)
- Input: Silver features from bronze_to_silver
- Output: Gold episodes (149D vectors + metadata + sequences)
- Use case: Production retrieval, Transformer training

### 3. Gold → FAISS Indices (Index Building)

**Script**: `scripts/build_faiss_indices.py`
- Input: Gold episodes (149D)
- Output: Gold indices (32D compressed for kNN)
- Compression: geometry_only strategy per Phase 4

---

## Level Universe (ES)

Structural levels used for event generation:
- **PM_HIGH/PM_LOW**: Pre-market high/low (04:00-09:30 ET)
- **OR_HIGH/OR_LOW**: Opening range high/low (09:30-09:45 ET)
- **SMA_90/EMA_20**: Moving averages on 2-min bars (warmup required)

Dynamic series (context-only) include session highs/lows, VWAP, and call/put walls.

---

## Feature Categories

- **Barrier/Tape/Fuel Physics**: `barrier_state`, `tape_imbalance`, `fuel_effect`, `gamma_exposure`, `call_tide`, `put_tide`
- **Kinematics**: `velocity_*`, `acceleration_*`, `jerk_*`, `momentum_trend_*` (1,2,3,5,10,20min)
- **Order Flow**: `ofi_*`, `ofi_acceleration` (30,60,120,300s)
- **Barrier Evolution**: `barrier_delta_*`, `barrier_pct_change_*` (1,2,3,5min)
- **Level Distances**: `dist_to_*`, `level_stacking_*`
- **GEX**: `gex_*`, `gex_asymmetry`, `gex_ratio`, `net_gex_2strike`
- **F=ma**: `predicted_accel`, `accel_residual`, `force_mass_ratio`
- **Market Tide (Phase 4.5)**: `call_tide`, `put_tide` (Net premium flow into calls/puts)
- **Approach/Normalization**: `approach_*`, `distance_signed_*`, `*_pct`
- **Labels**: `outcome*`, `strength_*`, `excursion_*`, `time_to_*`

---

## Pipeline Outputs

The pipeline writes to multiple layers depending on configuration flags:

### Silver Layer (Stage 16 - FilterRTH)
**Flag**: `PIPELINE_WRITE_SIGNALS=True`  
**Path**: `silver/features/es_pipeline/version={version}/date=YYYY-MM-DD/signals.parquet`  
**Content**: ~142 engineered features (RTH-filtered signals_df)  
**Schema**: See [../common/schemas/silver_features.py](../common/schemas/silver_features.py)

### Gold Layer - State Table (Stage 17 - MaterializeStateTable)
**Flag**: `PIPELINE_WRITE_STATE_TABLE=True`  
**Path**: `silver/state/es_level_state/version={version}/date=YYYY-MM-DD/state.parquet`  
**Content**: 30s cadence state table for episode construction  
**Columns**: ~89 forward-filled features

### Gold Layer - Episodes (Stage 18 - ConstructEpisodes)
**Flag**: `PIPELINE_WRITE_EPISODES=True`  
**Path**: `gold/episodes/es_level_episodes/version={version}/`  
**Content**: 
- `vectors/date=YYYY-MM-DD/episodes.npy` - 149D vectors
- `metadata/date=YYYY-MM-DD/metadata.parquet` - Event metadata  
- `sequences/date=YYYY-MM-DD/sequences.npy` - Raw 40×4 trajectories  
**Schema**: See [../../EPISODE_VECTOR_SCHEMA.md](../../EPISODE_VECTOR_SCHEMA.md)

### Gold Layer - FAISS Indices (Built separately via `build_all_indices()`)
**Path**: `gold/indices/es_level_indices/{level_kind}/{direction}/{time_bucket}/`  
**Content**: 32D compressed vectors (geometry-only) for live retrieval  
**Built from**: Gold episodes (149D → 32D via compressor)

---

## Configuration

Pipeline behavior is controlled by `backend/src/common/config.py`:
- Physics windows: `W_b`, `W_t`, `W_g`
- Zone/monitor bands: `MONITOR_BAND`, `TOUCH_BAND`
- Confirmation windows: `CONFIRMATION_WINDOWS_MULTI`
- Warmup: `SMA_WARMUP_DAYS`

---

## References

- **Episode Vectors (149D)**: [../../EPISODE_VECTOR_SCHEMA.md](../../EPISODE_VECTOR_SCHEMA.md)
- **Research Context**: [../../RESEARCH.md](../../RESEARCH.md)
- **Pipelines**: [pipelines/es_pipeline.py](pipelines/es_pipeline.py)
- **Registry**: [pipelines/registry.py](pipelines/registry.py)
- **Silver Builder**: [../io/silver.py](../io/silver.py)
