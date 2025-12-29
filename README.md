# Spymaster: ES Futures + ES Options Neuro-Hybrid Attribution Engine

**System**: Break/Bounce prediction at ES futures levels using physics-based features + kNN retrieval

**Architecture**: ES futures (spot + liquidity) + ES 0DTE options (gamma) - perfect alignment, zero conversion

**Approach**: Two-stage optimization → (1) Find best zones/windows for feature extraction, (2) Train model on optimized dataset

---

## Quick Start

### 1. Start Services

```bash
# Start backend infrastructure
docker-compose up -d

# Start frontend
cd frontend
npm run start
# Navigate to http://localhost:4200
```

### 2. Download ES Options Data

```bash
cd backend

# Set Databento API key
echo "DATABENTO_API_KEY=your_key_here" >> .env

# Download ES options (trades + NBBO)
uv run python scripts/download_es_options.py \
  --start 2025-11-02 \
  --end 2025-12-28
```

### 3. Build Features & Train Model

```bash
cd backend

# Build features with current CONFIG
uv run python -m src.lake.silver_feature_builder \
  --pipeline es_pipeline \
  --start-date 2025-11-02 \
  --end-date 2025-12-28

# Train model (includes hyperopt for XGBoost params)
uv run python -m src.ml.boosted_tree_train \
  --features gold/training/signals_production.parquet

# Optional: Optimize zones first (then update CONFIG and rebuild)
uv run python scripts/run_zone_hyperopt.py \
  --start-date 2025-11-02 \
  --end-date 2025-11-30 \
  --n-trials 200 \
  --dry-run
```

---

## System Specification

**Data**: ES futures + ES 0DTE options from Databento GLBX.MDP3  
**Levels**: 6 kinds (PM_HIGH/LOW, OR_HIGH/LOW, SMA_200/400)  
**Strike Spacing**: 25 ES points (ATM dominant)  
**Outcome Threshold**: 75 ES points (3 strikes)  
**Inference**: Continuous (every 2-min candle)  
**Pipeline**: 16 stages (load → levels → physics → labels → filter)  
**Features**: ~70 physics features (multi-window) + ~40 label columns  
**Model**: XGBoost + kNN (neuro-hybrid)

**Goal**: "Price approaching PM_HIGH, find 5 similar past setups → 4/5 BROKE → 80% confidence"

---

## Documentation

**[COMPONENTS.md](COMPONENTS.md)**: Component architecture, engines, interface contracts  
**[backend/DATA_ARCHITECTURE.md](backend/DATA_ARCHITECTURE.md)**: Data pipeline, hyperopt workflow, kNN system  
**[backend/features.json](backend/features.json)**: Feature column specs (v2.0.0)  
**[backend/src/common/config.py](backend/src/common/config.py)**: All tunable parameters (CONFIG singleton)  
**[backend/HYPEROPT_PLAN.md](backend/HYPEROPT_PLAN.md)**: Two-stage optimization strategy

**Module Docs**: See `backend/src/{module}/README.md` and `backend/src/{module}/INTERFACES.md` for implementation details

---

## Development

### Running Individual Services

```bash
# Terminal 1: NATS infrastructure
docker-compose up nats -d

# Terminal 2: Ingestor (replay mode)
cd backend
export REPLAY_DATE=2025-12-16
uv run python -m src.ingestor.replay_publisher

# Terminal 3: Core service
uv run python -m src.core.main

# Terminal 4: Gateway
uv run python -m src.gateway.main

# Terminal 5: Frontend
cd frontend
npm run start
```

### Testing

```bash
# Backend tests
cd backend
uv run pytest tests/

# Specific modules
uv run pytest tests/test_barrier_engine.py -v
uv run pytest tests/test_replay_determinism.py -v

# Frontend tests
cd frontend
npm test
```

---

## Key Concepts

**Neuro-Hybrid**: Deterministic physics engines + kNN retrieval (not pure ML)  
**Continuous Inference**: Features at every 2-min candle (not just touches)  
**Multi-Window**: 1-20min lookback encodes "setup shape" for kNN matching  
**Two-Stage Hyperopt**: Optimize data generation BEFORE model training  
**Sparse > Dense**: 15 high-quality events/day better than 100 noisy events  
**Front-Month Purity**: ES futures AND ES options on same contract (prevents ghost walls)  
**Deterministic IDs**: Event IDs are reproducible (no UUIDs) for retrieval consistency

---

**Version**: 2.0  
**Last Updated**: 2025-12-28  
**Status**: Production (ES futures + ES 0DTE options system)
