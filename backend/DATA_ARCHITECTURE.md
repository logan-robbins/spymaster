# Data Architecture & Workflow

**Version**: 3.1  
**Last Updated**: 2025-12-24  
**Status**: Production

---

## Overview

Spymaster uses the **Medallion Architecture** (Bronze → Silver → Gold) for production ML pipelines.

**Key Principles**:
- **Immutability**: Bronze is append-only, never modified
- **Reproducibility**: Bronze + manifest → deterministic Silver output
- **Versioning**: Silver feature sets use semantic versioning for A/B testing
- **Event-time first**: All records carry `ts_event_ns` and `ts_recv_ns`

---

## Directory Structure

```
backend/data/
├── raw/                                # Stage 0: Source data (DBN, Polygon)
│   ├── dbn/trades/, dbn/mbp10/
│   └── polygon/
│
├── lake/                              # Lakehouse tiers
│   ├── bronze/                        # Stage 1: Normalized events (immutable)
│   │   ├── futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   ├── futures/mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   ├── stocks/trades/symbol=SPY/date=YYYY-MM-DD/hour=HH/*.parquet
│   │   └── options/trades/underlying=SPY/date=YYYY-MM-DD/hour=HH/*.parquet
│   │
│   ├── silver/                        # Stage 2: Versioned feature experiments
│   │   └── features/
│   │       ├── v1.0_mechanics_only/
│   │       │   ├── manifest.yaml
│   │       │   ├── validation.json
│   │       │   └── date=YYYY-MM-DD/*.parquet
│   │       ├── v2.0_full_ensemble/
│   │       └── experiments.json
│   │
│   └── gold/                          # Stage 3: Production ML datasets
│       ├── training/
│       │   ├── signals_production.parquet
│       │   └── signals_production_metadata.json
│       ├── evaluation/backtest_YYYY-MM-DD.parquet
│       └── streaming/signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/*.parquet
│
└── ml/                                # Model artifacts
    ├── experiments/exp001_mechanics_only/
    ├── production/boosted_trees/
    └── registry.json
```

---

## Data Flow

### Streaming Pipeline (Real-time)

```
Live Feeds (Polygon, Databento)
    ↓ [Ingestor]
NATS (market.*)
    ├─→ [BronzeWriter] → Bronze (all hours, append-only)
    └─→ [Core Service] → NATS (levels.signals)
            └─→ [GoldWriter] → Gold/streaming/
```

### Batch Training Pipeline (Offline)

```
Bronze (all hours, immutable)
    ↓ [SilverFeatureBuilder uses versioned pipelines]
Silver Features (versioned, RTH only 9:30-16:00 ET)
    ↓ [GoldCurator]
Gold Training (production ML dataset)
    ↓ [ML Training Scripts]
Model Artifacts
```

---

## Lakehouse Tiers

### Bronze (Stage 1)

**Purpose**: Normalized events in canonical schema  
**Format**: Parquet with ZSTD compression  
**Schema**: `src/common/schemas/*.py` (StockTrade, OptionTrade, FuturesTrade, MBP10)  
**Time Coverage**: All hours (04:00 pre-market through 20:00 post-market ET)  
**Semantics**: Append-only, at-least-once delivery  
**Retention**: Permanent

**Why all hours?** Pre-market (04:00-09:30 ET) data is required for PM_HIGH/PM_LOW feature computation, but downstream layers filter to RTH.

### Silver (Stage 2)

**Purpose**: Versioned feature engineering experiments  
**Format**: Parquet with ZSTD compression  
**Schema**: Defined by versioned manifests (`manifest.yaml`)  
**Time Coverage**: RTH only (09:30-16:00 ET) for ML training  
**Semantics**: Reproducible transformations, exactly-once (after dedup)  
**Retention**: Keep active experiments, archive old versions

**Key Capabilities**:
- Version feature sets independently (v1.0, v2.0, v2.1, etc.)
- A/B test feature engineering approaches
- Track hyperparameters per experiment via manifests
- Reproducible: Bronze + manifest → deterministic Silver output

**Implementation**: `SilverFeatureBuilder` class (see `src/lake/silver_feature_builder.py`)

### Gold (Stage 3)

**Purpose**: Production-ready ML datasets and evaluation results  
**Format**: Parquet with ZSTD compression  
**Schema**: Final ML-ready schema (curated from best Silver experiment)  
**Semantics**: Curated, validated, production-quality  
**Retention**: Keep production datasets permanently

**Sources**:
- `gold/training/`: Curated from best Silver experiment version
- `gold/streaming/`: Real-time signals from Core Service
- `gold/evaluation/`: Backtest and validation results

**Implementation**: `GoldCurator` class (see `src/lake/gold_curator.py`)

---

## Feature Versioning

### Semantic Versioning

Format: `vMAJOR.MINOR_descriptive_name`

- **MAJOR**: Breaking schema changes
- **MINOR**: Feature additions/refinements
- **name**: Descriptive name (snake_case)

Examples:
- `v1.0_mechanics_only` - Initial mechanics baseline
- `v1.1_mechanics_enhanced` - Added dealer velocity features
- `v2.0_full_ensemble` - Major change: added TA features
- `v2.1_full_with_confluence` - Minor change: added confluence features

### Manifest Schema

Every Silver version has a `manifest.yaml` defining features and parameters:

```yaml
version: "v1.0.0"
name: "mechanics_only"
description: "Pure physics-based features (barrier, tape, fuel)"
created_at: "2025-12-24T12:00:00Z"
parent_version: null  # Or "v0.9.0" for incremental changes

source:
  layer: bronze
  schemas:
    - futures/trades
    - futures/mbp10
    - options/trades

features:
  groups:
    - name: barrier_physics
      columns: [barrier_state, barrier_delta_liq, ...]
    - name: tape_physics
      columns: [tape_imbalance, tape_velocity, ...]

parameters:
  W_b: 240  # Barrier window (seconds)
  W_t: 60   # Tape window (seconds)
  MONITOR_BAND: 0.25  # Monitor band ($)
```

---

## Quick Start

### 1. Bootstrap Production Pipeline

After backfilling Bronze data, run:

```bash
cd backend
uv run python scripts/bootstrap_medallion.py
```

This creates baseline Silver versions (v1.0, v2.0) and promotes best to Gold.

### 2. Train Models

```bash
cd backend
uv run python -m src.ml.boosted_tree_train --stage stage_b --ablation all
```

Models automatically read from `data/lake/gold/training/signals_production.parquet`.

---

## Feature Engineering Workflow

### Step 1: Define Feature Manifest

Create `manifests/v2.1_custom.yaml`:

```yaml
version: "v2.1_custom"
name: "custom_experiment"
description: "Testing new feature combinations"
parent_version: "v2.0_full_ensemble"
# ... (see manifest schema above)
```

### Step 2: Build Feature Set

```bash
cd backend
python -c "
from src.lake.silver_feature_builder import SilverFeatureBuilder
from src.common.schemas.feature_manifest import FeatureManifest

builder = SilverFeatureBuilder()
manifest = FeatureManifest.from_file('manifests/v2.1_custom.yaml')
stats = builder.build_feature_set(
    manifest=manifest,
    dates=['2025-12-16', '2025-12-17'],
    force=False
)
print(f'Status: {stats[\"status\"]}, Signals: {stats[\"signals_total\"]}')
"
```

### Step 3: Train and Evaluate

```python
from src.lake.silver_feature_builder import SilverFeatureBuilder

builder = SilverFeatureBuilder()

# Load features
df = builder.load_features('v2.1_custom')

# Train model (your ML code)
# ...

# Register experiment
builder.register_experiment(
    version='v2.1_custom',
    exp_id='exp003',
    status='completed',
    metrics={'auc': 0.72, 'precision': 0.68},
    notes='Improved AUC by 3% vs baseline'
)
```

### Step 4: Promote to Gold (if best)

```bash
cd backend
python -c "
from src.lake.gold_curator import GoldCurator

curator = GoldCurator()
result = curator.promote_to_training(
    silver_version='v2.1_custom',
    dataset_name='signals_production',
    notes='New production model - improved AUC',
    force=True
)
print(f'Promoted: {result[\"status\"]}')
"
```

---

## Management Commands

### List Silver Versions

```bash
cd backend
uv run python -m src.lake.silver_feature_builder --action list
```

### Compare Two Versions

```bash
cd backend
uv run python -m src.lake.silver_feature_builder --action compare \
  --version-a v1.0_mechanics_only \
  --version-b v2.0_full_ensemble
```

### Validate Gold Dataset

```bash
cd backend
uv run python -m src.lake.gold_curator --action validate \
  --dataset-name signals_production
```

---

## Best Practices

### Immutability
- Bronze is append-only, never modified
- Silver versions are immutable once created
- Gold datasets are versioned with metadata

### Reproducibility
- Every Silver feature set has a manifest
- Manifests include all parameters and dependencies
- Keep manifests in version control: `git add backend/manifests/*.yaml`

### RTH Filtering
**Critical**: Silver and Gold datasets contain ONLY RTH (09:30-16:00 ET) signals.

Why?
- Liquidity is highest during RTH
- ML models train on actionable trading hours
- Pre-market data is used for feature computation (PM_HIGH/PM_LOW) but not for training labels
- Implementation: Pipeline FilterRTHStage automatically filters output to RTH

### Experiment Tracking
Always register experiments with meaningful metadata including metrics, model path, and hypothesis notes.

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

### Pipeline (Modular)
**File**: `src/pipeline/` (modular stage-based architecture)
**Purpose**: Versioned feature computation pipelines (used by SilverFeatureBuilder)
**Key APIs**: `get_pipeline_for_version()`, `build_v1_0_pipeline()`, `build_v2_0_pipeline()`
**Note**: See `src/pipeline/README.md` for architecture details

### BronzeWriter / GoldWriter
**File**: `src/lake/bronze_writer.py`, `src/lake/gold_writer.py`  
**Purpose**: Streaming NATS → Parquet writers for real-time data  
**Note**: See `src/lake/README.md` for implementation details

---

## Critical Invariants

1. **Bronze is append-only**: Never mutate or delete
2. **Silver is derived**: Always regeneratable from Bronze
3. **Event-time ordering**: All files sorted by `ts_event_ns`
4. **Idempotency**: Same Bronze + manifest → same Silver
5. **RTH filtering**: Silver/Gold contain only 09:30-16:00 ET signals
6. **Partition boundaries**: Date/hour aligned to UTC
7. **Compression**: ZSTD level 3 for all tiers

---

## Troubleshooting

### "No Bronze data found"
```bash
cd backend
ls -la data/lake/bronze/futures/trades/symbol=ES/
# If empty, run backfill:
uv run python scripts/backfill_bronze_futures.py
```

### "Version already exists"
Use `force=True` to overwrite:
```python
builder.build_feature_set(manifest, dates=dates, force=True)
```

### High null rates in features
Check validation report:
```bash
cat backend/data/lake/silver/features/v2.0_full_ensemble/validation.json
```

---

## References

- **Bronze schemas**: `backend/src/common/schemas/`
- **Feature manifests**: `backend/src/common/schemas/feature_manifest.py`
- **Lake module**: `backend/src/lake/` (see `src/lake/README.md`)
- **Pipeline module**: `backend/src/pipeline/` (see `src/pipeline/README.md`)
- **Silver builder**: `backend/src/lake/silver_feature_builder.py`
- **Gold curator**: `backend/src/lake/gold_curator.py`
