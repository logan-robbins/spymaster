# Spymaster: Level Interaction Similarity Retrieval System

I am attempting to build a paltform specifically for market/dealer physics in the first 3 hours of trading (when volume is the highest). The goal is not to predict price (initially), but to retrieve similar "setups" and their labeled outcomes. I chose 6 specific levels: Pre-market high/low, 15 min opening range high/low, SMA_90 (2 min bars) and EMA_20 (2 min bars). a "setup" should be a vectorized snapshot of N lookback timeseries bars, where each bar represents the market state at that time-- RELATIVE to the specific "level" in question. The SETUP.png is a perfect example of what we are trying to model at each bar as the day progresses. The grey dotted lines are the pre-market high/low. The horizontal red and green lines are the 15 min open range high/low. The yellow/purple dotted lines are the SMA_90/EMA_20. There is clear reaction to these levels almost every single day. Sometimes the price breaks the levels, some times it does not. Sometimes it chops. We are trying to model and attribute the movements to various market physics forces. Then, we go a step further and say: at the 16 minute bar close, retrieve top 50 similar vectors and filter. What were their outcomes? the price break through the level or reject from the level. The hypothesis is that by looking at the approach over time (in granular windows computed within the large lookback window of say, 20 minutes), we can identify features that support break/reject probabilities. As a trader who follows these strict TA levels in the first 3 hours, the goal is that my platform UI begins retrieving/smoothing probabilistic signals based on the historical simliarity of the setup as the price approaches one of the levels we monitor. This can help me answer the question: if the price approaches the 15min opening range high from below, with f features, historically it has broken through with 77% probability. 



**System**: Retrieves historically similar market setups when price approaches technical levels, presenting empirical outcome distributions.

**Canonical Specification**: See **[IMPLEMENTATION_READY.md](IMPLEMENTATION_READY.md)** for complete system architecture, data contracts, and implementation details.

**Version**: 3.1.0 (December 2025)

## Quick Facts

- **Data Source**: ES futures + ES 0DTE options (Databento GLBX.MDP3)
- **Levels**: 6 kinds (PM_HIGH/LOW, OR_HIGH/LOW, SMA_90/EMA_20)
- **Outcomes**: BREAK/REJECT/CHOP (first-crossing semantics, 1.0 ATR threshold)
- **Episode Vectors**: 144 dimensions (Source) → 32 dimensions (Runtime Index/Geometry Only)
- **Zone Threshold**: 2.0 ATR for approach detection
- **Retrieval**: FAISS similarity search (60 partitions: 6 levels × 2 directions × 5 time buckets)
- **Pipeline**: 18 stages (bronze → silver → gold → indices)

---

## Quick Start

### 1. Setup Environment

```bash
cd backend

# Set Databento API key
echo "DATABENTO_API_KEY=your_key_here" >> .env

# Install dependencies (uv manages all Python packages)
uv sync
```

### 2. Download Data

```bash
cd backend

# Download ES options (trades + NBBO) → Bronze layer
uv run python scripts/download_es_options.py \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --workers 8

# Backfill ES futures from DBN files → Bronze layer
# (Requires ES futures DBN files in data/raw/)
uv run python scripts/backfill_bronze_futures.py --all
```

### 3. Run Pipeline

```bash
cd backend

# Run pipeline (18 stages) and write canonical lake outputs under a versioned hierarchy
# (rerunning the same version overwrites the date partitions by default)
uv run python -m scripts.run_pipeline --start 2024-11-01 --end 2024-12-31 --canonical-version 3.1.0

# Optional: enable incremental checkpointing / resume support
uv run python -m scripts.run_pipeline --date 2024-12-20 --checkpoint-dir data/checkpoints --canonical-version 3.1.0

# Validate pipeline output
uv run python scripts/validate_es_pipeline.py --date 2024-12-20
```

### 4. Build Retrieval System

```bash
cd backend

# Compute normalization statistics (60-day lookback, 144 features)
uv run python -c "
from pathlib import Path
from src.ml.normalization import ComputeNormalizationStage

stage = ComputeNormalizationStage(
    state_table_dir=Path('data/lake/silver/state/es_level_state/version=3.1.0'),
    output_dir=Path('data/lake/gold/normalization'),
    lookback_days=60
)
result = stage.execute()
print(f'Stats saved: {result[\"output_file\"]}')
"

# Build FAISS indices (60 partitions)
uv run python -c "
from pathlib import Path
from src.ml.index_builder import BuildIndicesStage

stage = BuildIndicesStage(
    episodes_dir=Path('data/lake/gold/episodes/es_level_episodes/version=3.1.0'),
    output_dir=Path('data/lake/gold/indices/es_level_indices/version=3.1.0')
)
result = stage.execute()
print(f'Built {result[\"n_partitions_built\"]} indices')
"
```

---

## STREAMS: Normalized Signal Streams + Forward Projections

**Pentaview** transforms the 30-second state table into continuous, interpretable **streams** that emit scalar values in `[-1, +1]` every 2-minute bar. These provide TA-style signals plus 20-minute forward projections.

### What It Does

- **5 Canonical Streams**: Momentum (Σ_M), Flow (Σ_F), Barrier (Σ_B), Dealer (Σ_D), Setup (Σ_S)
- **Merged Streams**: Pressure (Σ_P = momentum + flow), Structure (Σ_R = barrier + setup)
- **Derivatives**: Slope/curvature/jerk for acceleration-style TA
- **Projections**: 20-minute forecasts with uncertainty bands (q10/q50/q90)
- **Alerts**: 14 pattern types (exhaustion, divergence, reversal risk, etc.)
- **Exit Scoring**: Position-aware LONG/SHORT recommendations

### Quick Start

```bash
cd backend

# 1. Compute normalization statistics (60-day lookback)
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 60 --end-date 2024-12-31

# 2. Run Pentaview pipeline (compute streams for a date)
uv run python -m scripts.run_pentaview_pipeline \
  --date 2024-12-16

# 3. Validate output
uv run python -m scripts.validate_pentaview --date 2024-12-16

# 4. Build projection training dataset
uv run python -m scripts.build_projection_dataset \
  --start 2024-11-01 --end 2024-12-31 \
  --streams sigma_p,sigma_m,sigma_f,sigma_b,sigma_r

# 5. Train projection models
uv run python -m scripts.train_projection_models \
  --stream all --epochs 200

# 6. Demo projection inference
uv run python -m scripts.demo_projection

# 7. Demo state machine alerts
uv run python -m scripts.demo_state_machine
```

### Output

**Stream Bars**: `gold/streams/pentaview/version=3.1.0/date=YYYY-MM-DD/stream_bars.parquet`
- 32 columns: 5 canonical + 2 merged + 3 composites + derivatives
- One row per 2-min bar per active level
- All values bounded in [-1, +1]

**Projection Models**: `data/ml/projection_models/projection_{stream}_{version}.joblib`
- Quantile polynomial regression (q10/q50/q90)
- Forecasts 20 minutes ahead (10 bars @ 2-min)
- Smooth curves with TA interpretation (a1=slope, a2=curvature, a3=jerk)

**Alerts**: Generated on-demand via `detect_alerts(bar)`
- 14 types: exhaustion, continuation, reversal risk, divergence, barrier phases, quality gates
- Confidence scores + severity levels
- Hysteresis prevents flickering

### Testing

```bash
cd backend

# Test on single date
uv run python -m scripts.run_pentaview_pipeline --date 2024-12-16
uv run python -m scripts.validate_pentaview --date 2024-12-16

# Test projection model
uv run python -m scripts.demo_projection

# Test state machine
uv run python -m scripts.demo_state_machine
```

**See [STREAMS.md](STREAMS.md) for complete specification.**

---

## Documentation

**📘 [IMPLEMENTATION_READY.md](IMPLEMENTATION_READY.md)**: **CANONICAL SPECIFICATION** - Complete system architecture, data contracts, algorithms, and validation  
**[COMPONENTS.md](COMPONENTS.md)**: Component architecture and interface contracts  
**[backend/DATA_ARCHITECTURE.md](backend/DATA_ARCHITECTURE.md)**: Data pipeline and storage architecture  
**[backend/SILVER_SCHEMA.md](backend/SILVER_SCHEMA.md)**: Silver layer schema  
**[backend/src/common/schemas/](backend/src/common/schemas/)**: PyArrow schema definitions  

**Module Docs**: See `backend/src/{module}/README.md` for implementation details

---

## Development

### Running Services (Real-Time Mode)

```bash
# Terminal 1: NATS infrastructure
docker-compose up nats -d

# Terminal 2: Ingestion (replay mode for testing)
cd backend
export REPLAY_DATE=2024-12-16
uv run python -m src.ingestion.databento.replay

# Terminal 3: Core service (with retrieval engine)
uv run python -m src.core.main

# Terminal 4: Gateway (WebSocket API)
uv run python -m src.gateway.main

# Terminal 5: Frontend (Angular UI)
cd frontend
npm run start
```

### Validation

```bash
cd backend

# Validate full pipeline for a date
uv run python scripts/validate_es_pipeline.py --date 2024-12-20

# Validate specific stages
uv run python scripts/validate_stage_14_label_outcomes.py --date 2024-12-20
uv run python scripts/validate_stage_16_materialize_state_table.py --date 2024-12-20
uv run python scripts/validate_stage_17_construct_episodes.py --date 2024-12-20

# Batch validation
uv run python scripts/validate_es_pipeline.py --start 2024-12-01 --end 2024-12-31
```

---

## Architecture Principles

**Level-Relative Features**: All distances ATR-normalized, level-centric coordinate system  
**First-Crossing Semantics**: BREAK (1 ATR in direction), REJECT (1 ATR opposite), CHOP (neither)  
**Multi-Scale Dynamics**: 1-20min kinematics + 30-300s order flow + DCT trajectory basis  
**Trajectory Encoding**: DCT-II on 4 series (distance, OFI, barrier, tape) × 8 coefficients = 32 dims  
**Similarity Retrieval**: 144-dim vectors → FAISS ANN search → outcome distributions  
**60 Partitions**: (6 levels × 2 directions × 5 time buckets) for regime-comparable neighbors  
**Zone Threshold**: 2.0 ATR approach zone (tighter for higher-quality anchors)  
**Online Safety**: All features use only past data, labels never in feature vectors  
**Deterministic**: Event IDs reproducible, enables consistent retrieval
