# Spymaster: Real-Time Market Physics Engine

Spymaster models how dealer hedging, order-book liquidity, and trade flow interact at SPY 0DTE option levels to estimate BREAK vs BOUNCE outcomes and strength.

**Scope**: SPY 0DTE options only. Every $1 level is actionable, with special focus on PM high/low, opening range, SMA-200/400, VWAP, session extremes, and option walls.

**Primary outputs**: BREAK/BOUNCE probabilities with signed strength targets, per-event context about confluence and dealer mechanics, and tradeability predictions.

---

## System Architecture

**Microservices Pipeline** (Phase 2):
```
Data Sources → Ingestor → NATS JetStream → Core/Lake/Gateway
                                              ↓
                                         Frontend (WebSocket)
```

**Key Components**:
- **Ingestor**: Normalizes Polygon (SPY) and Databento (ES futures) feeds to NATS
- **Core**: Physics engines (barrier, tape, fuel) + signal generation
- **Lake**: Bronze/Silver/Gold data persistence (Parquet + S3/MinIO)
- **Gateway**: WebSocket relay for frontend
- **ML**: Boosted-tree models + kNN retrieval for tradeability/direction/strength
- **Frontend**: Angular real-time UI with physics attribution

**See**: [COMPONENTS.md](COMPONENTS.md) for complete architecture map with interface contracts.

---

## Quick Start

### Running the Full Stack

```bash
# Start backend services (Docker Compose)
docker-compose up -d

# Verify services
docker ps
curl http://localhost:8000/health

# Start frontend
cd frontend
npm run start
# Navigate to http://localhost:4200
```

### Running Individual Services (Local Dev)

```bash
# Terminal 1: NATS (required infrastructure)
docker-compose up nats -d

# Terminal 2: Ingestor (replay mode)
cd backend
export REPLAY_SPEED=1.0
export REPLAY_DATE=2025-12-16
uv run python -m src.ingestor.replay_publisher

# Terminal 3: Core service
uv run python -m src.core.main

# Terminal 4: Gateway
uv run python -m src.gateway.main

# Terminal 5: Frontend
cd ../frontend
npm run start
```

---

## Key Capabilities

**Physics Engines**:
- Barrier: ES MBP-10 depth analysis (VACUUM, WALL, ABSORPTION states)
- Tape: ES trade flow (imbalance, velocity, sweep detection)
- Fuel: SPY option flow (dealer gamma exposure, AMPLIFY vs DAMPEN)

**Signal Generation**:
- Break score (0-100) from weighted physics: 45% liquidity, 35% hedge, 20% tape
- Smoothing via EWMA for stable output (~250ms cadence)
- Confluence detection across stacked levels
- Runway analysis (distance to next obstacle)

**ML Pipeline**:
- Multi-head boosted trees: tradeability, direction, strength, time-to-threshold
- kNN retrieval for similar historical patterns
- Walk-forward validation (no look-ahead bias)
- Live viewport scoring (optional, gated by `VIEWPORT_SCORING_ENABLED`)

**Data Persistence**:
- Bronze: Raw normalized events (append-only, immutable)
- Silver: Deduped and sorted (exactly-once semantics)
- Gold: Derived signals and features (ML-ready, flattened schema)

---

## Documentation Map

### System Overview
- **[COMPONENTS.md](COMPONENTS.md)**: Architecture map with all component interfaces
- **[FRONTEND.md](FRONTEND.md)**: Frontend implementation status and UI specifications
- **[backend/features.json](backend/features.json)**: Authoritative feature schema

### Module Interfaces (Technical Contracts)
- **[backend/src/common/INTERFACES.md](backend/src/common/INTERFACES.md)**: Event types, schemas, config
- **[backend/src/ingestor/INTERFACES.md](backend/src/ingestor/INTERFACES.md)**: Data ingestion contracts
- **[backend/src/core/INTERFACES.md](backend/src/core/INTERFACES.md)**: Physics engines + signal output
- **[backend/src/lake/INTERFACES.md](backend/src/lake/INTERFACES.md)**: Storage schemas (Bronze/Silver/Gold)
- **[backend/src/gateway/INTERFACES.md](backend/src/gateway/INTERFACES.md)**: WebSocket relay
- **[backend/src/ml/INTERFACES.md](backend/src/ml/INTERFACES.md)**: Model training + inference
- **[frontend/INTERFACES.md](frontend/INTERFACES.md)**: UI components + service contracts

### Module Implementation Details
- **[backend/src/core/README.md](backend/src/core/README.md)**: Physics engine specifications
- **[backend/src/common/README.md](backend/src/common/README.md)**: Shared infrastructure
- **[backend/src/ingestor/README.md](backend/src/ingestor/README.md)**: Feed ingestion details
- **[backend/src/lake/README.md](backend/src/lake/README.md)**: Data persistence architecture
- **[backend/src/gateway/README.md](backend/src/gateway/README.md)**: WebSocket service details
- **[backend/src/ml/README.md](backend/src/ml/README.md)**: ML training and evaluation

---

## Configuration

**Single Source of Truth**: `backend/src/common/config.py` (CONFIG singleton)

**Key Parameters**:
- Physics windows: `W_b=10s` (barrier), `W_t=5s` (tape), `W_g=60s` (fuel)
- Monitoring: `MONITOR_BAND=0.50` (compute signals if |spot - level| ≤ $0.50)
- Thresholds: `R_vac=0.3` (VACUUM), `R_wall=1.5` (WALL), `F_thresh=100` (ES contracts)
- Weights: `w_L=0.45` (liquidity), `w_H=0.35` (hedge), `w_T=0.20` (tape)
- Smoothing: `tau_score=2.0s`, `tau_velocity=1.5s`

---

## Data Sources

**Live Feeds**:
- Polygon WebSocket: SPY equity + options (trades, quotes)
- Databento: ES futures (trades + MBP-10 depth) [historical only in current setup]

**Historical Replay**:
- Databento DBN files: `dbn-data/trades/`, `dbn-data/MBP-10/`
- Configurable replay speed (0x = fast, 1x = realtime, 2x = 2x speed)

**Storage**:
- Local: `backend/data/lake/` (Bronze/Silver/Gold Parquet)
- S3/MinIO: Optional via `USE_S3=true` config

---

## Performance Targets

**Latency** (end-to-end):
- Polygon event → NATS: <15ms
- NATS → Core → NATS: <50ms
- NATS → Gateway → WebSocket: <5ms
- **Total: <70ms** (exchange → frontend)

**Throughput**:
- Ingestor: 10k+ events/sec
- Core: 4 Hz signal rate (250ms snap interval)
- Gateway: 100+ concurrent WebSocket clients

**Storage**:
- Bronze: ~5-10GB/day (SPY + ES data)
- Silver: ~4-8GB/day (after dedup)
- Gold: ~100-500MB/day (signals)

---

## Testing

```bash
# Backend unit tests
cd backend
uv run pytest tests/

# Specific engine tests
uv run pytest tests/test_barrier_engine.py -v
uv run pytest tests/test_tape_engine.py -v

# Frontend tests
cd frontend
npm test

# End-to-end replay determinism
cd backend
uv run pytest tests/test_replay_determinism.py -v
```

---

## Development Tools

**Environment**: Python 3.11+ with `uv`, Node.js 18+ with npm, Docker Compose

**Required Services**:
- NATS JetStream (message bus)
- MinIO (optional, S3-compatible storage)

**Recommended IDE Setup**:
- Python: VSCode with Pylance
- Frontend: VSCode with Angular Language Service
- Docker: Docker Desktop

---

**Version**: 2.0 (Phase 2 microservices architecture)  
**Last Updated**: 2025-12-23  
**Status**: Active development
