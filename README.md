# Spymaster: ES Futures + ES Options Neuro-Hybrid Physics + Probabilistic Engine

**System**: Break/Bounce prediction at ES futures levels using physics-based features + kNN retrieval

**Data**: ES futures + ES 0DTE options from Databento GLBX.MDP3  
**Levels**: 6 kinds (PM_HIGH/LOW, OR_HIGH/LOW, SMA_200/400)  
**Strike Spacing**: CME listing schedule varies by moneyness/time-to-expiry (5/10/50/100-point intervals); GEX features aggregate gamma within ±1/±2/±3 strikes (typically 5pt spacing for ES 0DTE ATM)  
**Outcome Threshold**: Volatility-scaled barrier
**Inference**: Event-driven adaptive cadence (physics @ 250ms; ML updates by distance-to-level + triggers)  
**Pipeline**: 16 stages (load → levels → physics → labels → filter)  
**Features**: 182 columns (10 identity + 108 engineered features + 64 label columns; schema-enforced)  
**Model**: XGBoost + kNN (neuro-hybrid)

**Goal**: "Price approaching PM_HIGH, find 5 similar past setups → 4/5 BROKE → 80% confidence"

---

## Data Quick Start

### 1. Prepare Bronze Layer Data

```bash
cd backend

# Set Databento API key
echo "DATABENTO_API_KEY=your_key_here" >> .env

# Step 2a: Download ES options (trades + NBBO) - writes directly to Bronze
uv run python scripts/download_es_options.py \
  --start 2025-11-02 \
  --end 2025-12-28

# Step 2b: Backfill ES futures from DBN files to Bronze
# (Assumes you have ES futures DBN files in dbn-data/trades/ and dbn-data/MBP-10/)
uv run python scripts/backfill_bronze_futures.py --all
```

### 2. Build Features & Train Model

```bash
cd backend

# Build features with fixed CONFIG (src/common/config.py)
uv run python -m src.lake.silver_feature_builder \
  --pipeline es_pipeline \
  --start-date 2025-11-02 \
  --end-date 2025-12-28

# Train model (includes hyperopt for XGBoost params, feature selection, kNN blend)
uv run python -m src.ml.boosted_tree_train \
  --features gold/training/signals_production.parquet

```

---

## Documentation

**[COMPONENTS.md](COMPONENTS.md)**: Component architecture, engines, interface contracts  
**[backend/DATA_ARCHITECTURE.md](backend/DATA_ARCHITECTURE.md)**: Data pipeline, hyperopt workflow, kNN system  
**[backend/src/common/schemas/](backend/src/common/schemas/)**: Enforced PyArrow schemas (Bronze/Silver/Gold)  
**[backend/src/common/config.py](backend/src/common/config.py)**: All tunable parameters (CONFIG singleton, auto-loaded from best-config JSON)  

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
---

## Key Concepts

**Neuro-Hybrid**: Deterministic physics engines + kNN retrieval (not pure ML)  
**LevelState Engine**: For each level (PM/OR/SMA), compute kinematics/flow/liquidity/dealer mechanics vector  
**Multi-Window**: 1-20min lookback encodes "setup shape" for kNN matching  
**Two-Stage Hyperopt**: Optimize data generation BEFORE model training  
**Sparse > Dense**: 15 high-quality events/day better than 100 noisy events  
**Front-Month Purity**: ES futures AND ES options on same contract (prevents ghost walls)  
**Deterministic IDs**: Event IDs are reproducible (no UUIDs) for retrieval consistency
