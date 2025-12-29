# Spymaster Backend

Physics-based signal generation and ML feature engineering for ES 0DTE break/bounce prediction.

---

## Purpose

Predicts whether ES price will BREAK through or BOUNCE off key levels (PM/OR extremes, SMA_200/400) by modeling:
1. **Order book liquidity** (ES futures MBP-10): Dealer defense vs abandonment
2. **Trade flow** (ES futures trades): Aggressive buying/selling pressure
3. **Options flow** (ES 0DTE): Dealer gamma hedging effects

**Signal Trigger**: Price enters critical zone (±0.25 ES points of level) → compute physics → publish break/bounce prediction (0-100 score)

## Quick Start

### Running Services

```bash
# Full stack (Docker Compose)
docker-compose up -d

# Individual services (local dev)
cd backend/

# Install dependencies
uv sync

# Run ingestor (replay mode)
export REPLAY_SPEED=1.0
export REPLAY_DATE=2025-12-16
uv run python -m src.ingestor.replay_publisher

# Run core service (separate terminal)
uv run python -m src.core.main

# Run gateway (separate terminal)
uv run python -m src.gateway.main

# Run lake service (separate terminal)
uv run python -m src.lake.main
```

### Processing Historical Data

```bash
# Build baseline Silver + Gold datasets from Bronze
uv run python scripts/bootstrap_medallion.py

# Validate generated Gold dataset
uv run python -m src.lake.gold_curator --action validate --dataset-name signals_production
```

---

## Architecture

**Microservices Pipeline** (Phase 2):
```
Ingestor → NATS JetStream → Core/Lake/Gateway
                              ↓
                         Level Signals
```

**Components**:
- **Ingestor**: Normalize feeds, publish to NATS (`market.*` subjects)
- **Core**: Physics engines, signal generation, publish to NATS (`levels.signals`)
- **Lake**: Persist to Bronze/Silver/Gold Parquet (local or S3/MinIO)
- **Gateway**: WebSocket relay to frontend

**See**: [COMPONENTS.md](../COMPONENTS.md) for complete architecture map.

## Data Pipeline

**Batch Processing** (for ML training):
1. Load Bronze via DuckDB (ES trades/MBP-10 + ES options)
2. Generate level universe (PM/OR highs/lows, SMA_200/400)
3. Detect interaction zones (event-driven zone entry)
4. Filter to critical zone (|close - level| ≤ 0.25 ES points)
5. Compute physics (barrier, tape, fuel) + approach context
6. Label outcomes (BREAK/BOUNCE/CHOP using volatility-scaled barrier)
7. Output: Silver features → Gold training via GoldCurator (`data/lake/gold/training/signals_production.parquet`)

**Live Processing** (Core Service):
1. Subscribe to NATS (`market.*` subjects)
2. Update MarketState (ES MBP-10, trades, ES option flow)
3. Generate level universe every snap tick (250ms)
4. Compute physics + score for levels within MONITOR_BAND
5. Publish to NATS (`levels.signals`)

**See**: [backend/src/core/INTERFACES.md](src/core/INTERFACES.md) for signal output schema.

---

## Feature Schema

**Authoritative source**: `backend/SILVER_SCHEMA.md`

**Schema reference**: `backend/src/common/schemas/silver_features.py`

## Configuration

**Single source**: `backend/src/common/config.py` (CONFIG singleton)

**Key parameters**:
- `MONITOR_BAND = 0.25`: Compute signals if |spot - level| ≤ 0.25 ES points
- `W_b = 240.0`: Barrier window (seconds)
- `W_t = 60.0`: Tape window (seconds)
- `W_g = 60.0`: Fuel window (seconds)
- `R_vac = 0.3`, `R_wall = 1.5`: VACUUM/WALL thresholds
- `w_L = 0.45`, `w_H = 0.35`, `w_T = 0.20`: Score weights

**See**: [backend/src/common/INTERFACES.md](src/common/INTERFACES.md) for complete configuration contract.

---

## Module Structure

```
backend/src/
├── common/           # Shared contracts (event types, schemas, config)
├── ingestor/         # Data normalization (Databento)
├── core/             # Physics engines + signal generation
├── lake/             # Bronze/Silver/Gold persistence
├── gateway/          # WebSocket relay
└── ml/               # Model training + inference
```

**See module INTERFACES.md files** for detailed technical contracts.

## Data Sources

**Ingestor sources**:
- Databento DBN: ES futures (trades, MBP-10) and ES options (trades)

**Storage locations**:
- DBN files: `dbn-data/trades/`, `dbn-data/MBP-10/`
- Parquet output: `data/lake/` (Bronze/Silver/Gold)

---

## Common Tasks

### Training ML Models
```bash
# Boosted trees
uv run python -m src.ml.boosted_tree_train --stage stage_b --ablation all

# Retrieval index
uv run python -m src.ml.build_retrieval_index --stage stage_b

# PatchTST baseline
uv run python -m src.ml.sequence_dataset_builder --date 2025-12-16
uv run python -m src.ml.patchtst_train --train-files <files> --val-files <files>
```

### Processing Historical Data
```bash
# Build baseline Silver + Gold datasets from Bronze
uv run python scripts/bootstrap_medallion.py

# Validate Gold dataset
uv run python -m src.lake.gold_curator --action validate --dataset-name signals_production
```

### Running Tests
```bash
# All tests
uv run pytest tests/

# Specific engine
uv run pytest tests/test_barrier_engine.py -v

# Replay determinism
uv run pytest tests/test_replay_determinism.py -v
```

---

## Key Invariants

1. **Time windows**: Physics looks FORWARD from touch timestamp
2. **Config authority**: All parameters in `src/common/config.py`
3. **Event-time ordering**: Sort by `ts_event_ns`, not `ts_recv_ns`
4. **2.00-point threshold**: BREAK/BOUNCE require ≥2.00 ES points movement
5. **Critical zone**: Signals only when |spot - level| ≤ MONITOR_BAND
6. **Deterministic replay**: Same inputs + config → same outputs

---

## Documentation

**Interface contracts** (authoritative for AI agents):
- [common/INTERFACES.md](src/common/INTERFACES.md) - Event types, schemas, config
- [ingestor/INTERFACES.md](src/ingestor/INTERFACES.md) - Data ingestion contracts
- [core/INTERFACES.md](src/core/INTERFACES.md) - Physics engines + signal output
- [lake/INTERFACES.md](src/lake/INTERFACES.md) - Storage schemas
- [gateway/INTERFACES.md](src/gateway/INTERFACES.md) - WebSocket relay
- [ml/INTERFACES.md](src/ml/INTERFACES.md) - Model training + inference

**Implementation details**:
- [common/README.md](src/common/README.md) - Shared infrastructure
- [core/README.md](src/core/README.md) - Physics engine specifications
- [ingestor/README.md](src/ingestor/README.md) - Feed ingestion details
- [lake/README.md](src/lake/README.md) - Data persistence architecture
- [gateway/README.md](src/gateway/README.md) - WebSocket service details
- [ml/README.md](src/ml/README.md) - ML training and evaluation

---

**Version**: 2.0 (Phase 2)  
**Status**: Active development
