# Data Architecture: Medallion Pattern

**Version**: 2.0  
**Last Updated**: 2025-12-24  
**Status**: Active Implementation

---

## Directory Structure

```
backend/data/                           # Single source of truth for all data
├── raw/                                # Stage 0: Raw ingestion
│   ├── dbn/                           # Databento DBN files (ES futures)
│   │   ├── trades/
│   │   └── mbp10/
│   ├── polygon/                       # Polygon flat files (SPY options)
│   │   └── options/
│   └── manifest.json                  # Ingestion metadata
│
├── lake/                              # Lakehouse tiers
│   ├── bronze/                        # Stage 1: Normalized events (immutable)
│   │   ├── futures/
│   │   │   ├── trades/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   │   └── mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   ├── stocks/
│   │   │   └── trades/symbol=SPY/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   ├── options/
│   │   │   └── trades/underlying=SPY/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   └── manifest.json              # Bronze layer metadata
│   │
│   ├── silver/                        # Stage 2: Feature engineering experiments
│   │   ├── features/                  # Feature sets (versioned)
│   │   │   ├── v1.0_mechanics_only/
│   │   │   │   ├── manifest.yaml     # Feature definition
│   │   │   │   ├── params.yaml       # Hyperparameters
│   │   │   │   └── date=YYYY-MM-DD/*.parquet
│   │   │   ├── v1.1_ta_only/
│   │   │   │   ├── manifest.yaml
│   │   │   │   ├── params.yaml
│   │   │   │   └── date=YYYY-MM-DD/*.parquet
│   │   │   ├── v2.0_full_ensemble/
│   │   │   │   ├── manifest.yaml
│   │   │   │   ├── params.yaml
│   │   │   │   └── date=YYYY-MM-DD/*.parquet
│   │   │   └── experiments.json      # Experiment tracking
│   │   │
│   │   └── datasets/                  # Intermediate clean datasets
│   │       ├── ohlcv/                # OHLCV bars
│   │       ├── levels/               # Level universe
│   │       └── touches/              # Touch detection results
│   │
│   └── gold/                          # Stage 3: Production ML datasets
│       ├── training/
│       │   ├── signals_production.parquet
│       │   └── manifest.yaml
│       ├── evaluation/
│       │   └── backtest_YYYY-MM-DD.parquet
│       └── streaming/                 # Real-time signals (from Core)
│           └── signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/*.parquet
│
├── ml/                                # Model artifacts
│   ├── experiments/
│   │   ├── exp001_mechanics_only/
│   │   │   ├── model.joblib
│   │   │   ├── metrics.json
│   │   │   └── config.yaml
│   │   └── exp002_full_ensemble/
│   ├── production/
│   │   ├── boosted_trees/
│   │   └── retrieval_index.joblib
│   └── registry.json                  # Model registry
│
└── wal/                               # Write-ahead log (transient)
    └── *.wal

```

---

## Data Flow

### Batch Training Pipeline

```
Raw Data (dbn-data/, s3:/)
    ↓ [Ingestor]
Bronze (normalized events, immutable)
    ↓ [SilverFeatureBuilder]
Silver Features (versioned experiments)
    ↓ [GoldCurator]
Gold Training (production ML dataset)
    ↓ [ML Training]
Model Artifacts
```

### Streaming Production Pipeline

```
Live Feeds (Polygon, Databento)
    ↓ [Ingestor]
NATS (market.*)
    ↓ [Core Service]
NATS (levels.signals)
    ├─→ [Gateway] → Frontend
    └─→ [Gold Streaming Writer] → Gold/streaming/
```

---

## Layer Definitions

### Raw (Stage 0)
**Purpose**: Immutable source data  
**Format**: DBN files, flat files, API responses  
**Retention**: Permanent (archive to cold storage after 90 days)  
**Schema**: Provider-specific (Databento, Polygon)

### Bronze (Stage 1)
**Purpose**: Normalized events in canonical schema  
**Format**: Parquet with ZSTD compression  
**Schema**: `src/common/schemas/*.py` (StockTrade, OptionTrade, FuturesTrade, MBP10)  
**Semantics**: At-least-once delivery, append-only, immutable  
**Retention**: Permanent  
**Deduplication**: Deferred to Silver layer

### Silver (Stage 2)
**Purpose**: Feature engineering experimentation and clean datasets  
**Format**: Parquet with ZSTD compression  
**Schema**: Versioned feature manifests  
**Semantics**: Exactly-once (after dedup), reproducible transformations  
**Retention**: Keep active experiments, archive old versions  
**Key Capabilities**:
- Version feature sets independently
- A/B test feature engineering approaches
- Track hyperparameters per experiment
- Reproducible from Bronze + manifest

### Gold (Stage 3)
**Purpose**: Production-ready ML datasets and evaluation results  
**Format**: Parquet with ZSTD compression  
**Schema**: Final ML-ready schema (matches features.json)  
**Semantics**: Curated, validated, production-quality  
**Retention**: Keep production datasets permanently  
**Sources**: 
- `gold/training/`: Curated from best Silver experiment
- `gold/streaming/`: Real-time signals from Core Service
- `gold/evaluation/`: Backtest and validation results

---

## Feature Versioning

### Manifest Schema (silver/features/*/manifest.yaml)

```yaml
version: "1.0.0"
name: "mechanics_only"
description: "Pure physics-based features (barrier, tape, fuel) without TA"
created_at: "2025-12-24T12:00:00Z"
parent_version: null  # Or "0.9.0" for incremental changes

source:
  layer: bronze
  schemas:
    - futures/trades
    - futures/mbp10
    - options/trades

features:
  groups:
    - barrier_physics:
        columns:
          - barrier_state
          - barrier_delta_liq
          - barrier_replenishment_ratio
          - wall_ratio
    - tape_physics:
        columns:
          - tape_imbalance
          - tape_velocity
          - sweep_detected
    - fuel_physics:
        columns:
          - gamma_exposure
          - fuel_effect
          - dealer_pressure

parameters:
  W_b: 240  # Barrier window (seconds)
  W_t: 60   # Tape window (seconds)
  W_g: 60   # Fuel window (seconds)
  MONITOR_BAND: 0.25  # Monitor band ($)
  
validation:
  date_range:
    start: "2025-12-01"
    end: "2025-12-20"
  signal_count: 3324
  null_rates:
    barrier_state: 0.0
    gamma_exposure: 0.15
```

### Experiment Tracking (silver/features/experiments.json)

```json
{
  "experiments": [
    {
      "id": "exp001",
      "version": "v1.0_mechanics_only",
      "created_at": "2025-12-24T12:00:00Z",
      "status": "completed",
      "metrics": {
        "signal_count": 3324,
        "feature_count": 15,
        "date_coverage": 4,
        "null_rate_avg": 0.05
      },
      "notes": "Baseline mechanics-only model"
    },
    {
      "id": "exp002",
      "version": "v2.0_full_ensemble",
      "created_at": "2025-12-24T14:00:00Z",
      "status": "running",
      "parent": "exp001",
      "notes": "Adding TA features to mechanics baseline"
    }
  ]
}
```

---

## Migration Plan

### Phase 1: Consolidate Directories (Immediate)
1. **Remove** `backend/backend/data/` (duplicate)
2. **Rename** `backend/src/data/` → `backend/src/storage/` (to avoid confusion)
3. **Move** `/dbn-data/` → `backend/data/raw/dbn/`
4. **Keep** `backend/data/` as single source of truth

### Phase 2: Create Silver Layer (Week 1)
1. Create `backend/data/lake/silver/features/` structure
2. Implement `SilverFeatureBuilder` class
3. Create feature manifest schema
4. Refactor `VectorizedPipeline` to output to Silver

### Phase 3: Refactor Pipeline (Week 1-2)
1. Split `VectorizedPipeline` into:
   - `BronzeReader` (existing)
   - `SilverFeatureBuilder` (new)
   - `GoldCurator` (new)
2. Implement versioned feature sets
3. Add experiment tracking

### Phase 4: Update Consumers (Week 2)
1. Update ML training scripts to read from Silver
2. Update Gold layer to curate from Silver
3. Update documentation

---

## Best Practices

### Immutability
- Bronze is append-only, never modified
- Silver versions are immutable once created
- Gold datasets are versioned

### Reproducibility
- Every Silver feature set has a manifest
- Manifests include all parameters and dependencies
- Bronze + manifest → deterministic Silver output

### Versioning
- Semantic versioning: `vMAJOR.MINOR_name`
- Major: Breaking schema changes
- Minor: Feature additions/refinements

### Experimentation
- Create new Silver version for each experiment
- Track metrics in experiments.json
- Promote best experiment to Gold

---

## References

- Bronze schemas: `backend/src/common/schemas/`
- Feature definitions: `backend/features.json` (legacy, will migrate)
- Pipeline code: `backend/src/pipeline/vectorized_pipeline.py`
- Lake service: `backend/src/lake/`

