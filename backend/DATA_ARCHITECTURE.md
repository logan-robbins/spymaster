# Data Architecture & Workflow

**Version**: 4.0  
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
    ↓ [Pipeline stages via SilverFeatureBuilder]
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

**Writer**: `src/lake/bronze_writer.py` (BronzeWriter class)  
**Reader**: `src/lake/bronze_writer.py` (BronzeReader class) and `src/pipeline/utils/duckdb_reader.py`

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

**Implementation**: `SilverFeatureBuilder` class (`src/lake/silver_feature_builder.py`)

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

**Implementation**: `GoldCurator` class (`src/lake/gold_curator.py`)

---

## Pipeline Architecture

The pipeline module (`src/pipeline/`) provides modular, versioned feature engineering. Each pipeline version uses a different stage composition.

### Core Components

```
src/pipeline/
├── core/
│   ├── stage.py         # BaseStage, StageContext abstractions
│   └── pipeline.py      # Pipeline orchestrator
├── stages/              # Individual processing stages
├── pipelines/           # Pipeline version definitions
│   ├── v1_0_mechanics_only.py
│   ├── v2_0_full_ensemble.py
│   └── registry.py      # get_pipeline_for_version()
└── utils/
    ├── duckdb_reader.py     # DuckDB wrapper with downsampling
    └── vectorized_ops.py    # Vectorized NumPy/pandas operations
```

### Pipeline Stages

Each stage is a class extending `BaseStage` with explicit inputs/outputs:

| Stage | Class | Description | Outputs |
|-------|-------|-------------|---------|
| 1 | `LoadBronzeStage` | Load Bronze data via DuckDB | `trades`, `mbp10_snapshots`, `option_trades_df` |
| 2 | `BuildOHLCVStage` (1min) | Build 1-minute OHLCV bars | `ohlcv_1min`, `atr` |
| 3 | `BuildOHLCVStage` (2min) | Build 2-minute bars with warmup | `ohlcv_2min` |
| 4 | `InitMarketStateStage` | Initialize MarketState, compute Greeks | `market_state`, `spot_price` |
| 5 | `GenerateLevelsStage` | Generate level universe | `level_info`, `static_level_info`, `dynamic_levels` |
| 6 | `DetectTouchesStage` | Detect level touches | `touches_df` |
| 7 | `ComputePhysicsStage` | Compute barrier/tape/fuel physics | `signals_df` |
| 8 | `ComputeContextFeaturesStage` | Add context features | `signals_df` (updated) |
| 9 | `ComputeSMAFeaturesStage` | SMA mean reversion (v2.0+) | `signals_df` (updated) |
| 10 | `ComputeConfluenceStage` | Confluence + pressure (v2.0+) | `signals_df` (updated) |
| 11 | `ComputeApproachFeaturesStage` | Approach context (v2.0+) | `signals_df` (updated) |
| 12 | `LabelOutcomesStage` | Label outcomes (competing risks) | `signals_df` (updated) |
| 13 | `FilterRTHStage` | Filter to RTH (09:30-16:00 ET) | `signals` (final) |

### Pipeline Versions

#### v1.0 - Mechanics Only (10 stages)

Pure physics features without TA indicators:

```
LoadBronze → BuildOHLCV(1min) → BuildOHLCV(2min) → InitMarketState
→ GenerateLevels → DetectTouches → ComputePhysics → ComputeContext
→ LabelOutcomes → FilterRTH
```

**Features**: Barrier state, tape imbalance, fuel effects, basic context

#### v2.0 - Full Ensemble (13 stages)

All features including SMA, confluence, and approach context:

```
LoadBronze → BuildOHLCV(1min) → BuildOHLCV(2min) → InitMarketState
→ GenerateLevels → DetectTouches → ComputePhysics → ComputeContext
→ ComputeSMA → ComputeConfluence → ComputeApproach
→ LabelOutcomes → FilterRTH
```

**Additional Features**: SMA distances, confluence alignment, dealer velocity, pressure indicators, approach context, normalized features

### Level Universe

Generated structural levels (SPY-specific):

| Level Type | Description |
|------------|-------------|
| PM_HIGH/PM_LOW | Pre-market high/low (04:00-09:30 ET) |
| OR_HIGH/OR_LOW | Opening range (09:30-09:45 ET) |
| SESSION_HIGH/SESSION_LOW | Running session extremes |
| SMA_200/SMA_400 | Moving averages on 2-min bars |
| VWAP | Session volume-weighted average price |
| CALL_WALL/PUT_WALL | Max gamma concentration strikes |

**Note**: ROUND and STRIKE levels are extracted but SPY uses $1 strike spacing.

### Feature Categories

**Barrier Physics** (ES MBP-10 depth):
- `barrier_state`, `barrier_delta_liq`, `barrier_replenishment_ratio`, `wall_ratio`

**Tape Physics** (ES trade flow):
- `tape_imbalance`, `tape_buy_vol`, `tape_sell_vol`, `tape_velocity`, `sweep_detected`

**Fuel Physics** (SPY option gamma):
- `gamma_exposure`, `fuel_effect`

**Context Features**:
- `is_first_15m`, `date`, `symbol`, `direction_sign`, `event_id`, `atr`
- Structural distances to key levels

**SMA Features** (v2.0+):
- Mean reversion distances and velocities

**Confluence Features** (v2.0+):
- `confluence_level`, `gex_alignment`, `rel_vol_ratio`
- Dealer velocity features, pressure indicators
- `gamma_bucket` (SHORT_GAMMA/LONG_GAMMA)

**Approach Features** (v2.0+):
- `approach_velocity`, `attempt_index`, prior touch counts
- Sparse feature transforms, normalized features

**Labels** (competing risks):
- `outcome` (BREAK, BOUNCE, STALL)
- `strength_signed` (signed magnitude)
- `t1_60`, `t1_120`, `t2_60`, `t2_120` (confirmation timestamps)
- `tradeable_1`, `tradeable_2` (tradeable flags)

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
uv run python -c "
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
uv run python -c "
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

### Run Pipeline Directly

```python
from src.pipeline import get_pipeline_for_version

# Get pipeline for version
pipeline = get_pipeline_for_version("v2.0_full_ensemble")
signals_df = pipeline.run("2025-12-16")
```

---

## Configuration

Pipeline behavior controlled by `backend/src/common/config.py`:

| Category | Parameters |
|----------|------------|
| Physics Windows | `W_b` (240s), `W_t` (60s), `W_g` (60s) |
| Touch Detection | `MONITOR_BAND` (0.25), `TOUCH_BAND` (0.10) |
| Confirmation | `CONFIRMATION_WINDOWS_MULTI` [120, 240, 480]s |
| Warmup | `SMA_WARMUP_DAYS` (3), `VOLUME_LOOKBACK_DAYS` (7) |
| Outcome | `OUTCOME_THRESHOLD` (2.0), `LOOKFORWARD_MINUTES` (8) |

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
- Implementation: `FilterRTHStage` automatically filters output to RTH

### Experiment Tracking
Always register experiments with meaningful metadata including metrics, model path, and hypothesis notes.

---

## Performance

**Apple M4 Silicon Optimized**:
- All operations use NumPy broadcasting
- Batch processing of all touches simultaneously
- Memory-efficient chunked processing

**Typical Performance** (M4 Mac, 128GB RAM):
- Single date: ~2-5 seconds
- 10 dates: ~30-60 seconds
- ~500-1000 signals/sec throughput

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

### "No pipeline for version"
Check available pipeline versions:
```python
from src.pipeline.pipelines import list_available_versions
print(list_available_versions())  # ['v1.0', 'v2.0']
```

---

## References

- **Bronze schemas**: `backend/src/common/schemas/`
- **Feature manifests**: `backend/src/common/schemas/feature_manifest.py`
- **Lake module**: `backend/src/lake/` (see `src/lake/README.md`)
- **Pipeline module**: `backend/src/pipeline/` (see `src/pipeline/README.md`)
- **Silver builder**: `backend/src/lake/silver_feature_builder.py`
- **Gold curator**: `backend/src/lake/gold_curator.py`
- **Configuration**: `backend/src/common/config.py`
