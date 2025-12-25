# Medallion Architecture Workflow

**Version**: 2.0  
**Last Updated**: 2025-12-24  
**Purpose**: Guide for using the Bronze/Silver/Gold data pipeline

---

## Quick Start

### 1. Bootstrap the Architecture

After backfilling Bronze data, run:

```bash
cd backend
uv run python scripts/bootstrap_medallion.py
```

This will:
- Create baseline feature sets (`v1.0_mechanics_only`, `v2.0_full_ensemble`)
- Promote the best experiment to Gold
- Validate the production dataset

### 2. Train Models

```bash
cd backend
uv run python -m src.ml.boosted_tree_train --stage stage_b --ablation all
```

### 3. Create New Experiments

See "Feature Engineering Workflow" below.

---

## Architecture Overview

```
Bronze (raw, immutable)
  ↓ [VectorizedPipeline + SilverFeatureBuilder]
Silver Features (versioned experiments)
  ├─ v1.0_mechanics_only/
  ├─ v2.0_full_ensemble/
  └─ v2.1_custom_experiment/
  ↓ [GoldCurator]
Gold Training (production ML dataset)
  └─ signals_production.parquet
  ↓ [ML Training]
Model Artifacts
```

---

## Feature Engineering Workflow

### Step 1: Define Feature Manifest

Create a manifest defining your feature set:

```python
from src.common.schemas.feature_manifest import (
    FeatureManifest,
    FeatureGroup,
    Parameters,
    SourceConfig
)
from datetime import datetime

# Define custom feature groups
custom_groups = [
    FeatureGroup(
        name="custom_mechanics",
        description="Custom mechanics features",
        columns=[
            "barrier_state",
            "tape_imbalance",
            "gamma_exposure",
            # ... your features
        ]
    )
]

# Create manifest
manifest = FeatureManifest(
    version="v2.1_custom_experiment",
    name="custom_experiment",
    description="Testing new feature combinations",
    created_at=datetime.utcnow().isoformat() + "Z",
    source=SourceConfig(
        layer="bronze",
        schemas=["futures/trades", "futures/mbp10", "options/trades"]
    ),
    feature_groups=custom_groups,
    parameters=Parameters(
        W_b=240,  # Custom barrier window
        W_t=60,
        # ... custom parameters
    ),
    parent_version="v2.0_full_ensemble",  # Optional: parent for comparison
    tags=["custom", "experiment"],
    notes="Testing hypothesis X"
)

# Save manifest
manifest.to_file("manifests/v2.1_custom_experiment.yaml")
```

### Step 2: Build Feature Set

```python
from src.lake.silver_feature_builder import SilverFeatureBuilder

builder = SilverFeatureBuilder()

# Load manifest
manifest = FeatureManifest.from_file("manifests/v2.1_custom_experiment.yaml")

# Build features for available dates
stats = builder.build_feature_set(
    manifest=manifest,
    dates=['2025-12-16', '2025-12-17', '2025-12-18', '2025-12-19'],
    force=False
)

print(f"Status: {stats['status']}")
print(f"Signals: {stats['signals_total']}")
```

**CLI**:
```bash
cd backend
python -c "
from src.lake.silver_feature_builder import SilverFeatureBuilder
from src.common.schemas.feature_manifest import FeatureManifest

builder = SilverFeatureBuilder()
manifest = FeatureManifest.from_file('manifests/v2.1_custom_experiment.yaml')
stats = builder.build_feature_set(manifest, dates=['2025-12-16'])
print(stats)
"
```

### Step 3: Train and Evaluate

```python
from src.lake.silver_feature_builder import SilverFeatureBuilder

builder = SilverFeatureBuilder()

# Load features
df = builder.load_features('v2.1_custom_experiment')

# Train model (your ML code)
# ...

# Register experiment
builder.register_experiment(
    version='v2.1_custom_experiment',
    exp_id='exp003',
    status='completed',
    metrics={
        'auc': 0.72,
        'precision': 0.68,
        'recall': 0.65
    },
    model_path='data/ml/experiments/exp003/model.joblib',
    notes='Improved AUC by 3% vs baseline'
)
```

### Step 4: Promote to Gold (if best)

```python
from src.lake.gold_curator import GoldCurator

curator = GoldCurator()

# Promote to production
result = curator.promote_to_training(
    silver_version='v2.1_custom_experiment',
    dataset_name='signals_production',
    notes='New production model (exp003) - improved AUC',
    force=True  # Overwrite existing
)

print(f"Promoted: {result['status']}")
```

**CLI**:
```bash
cd backend
uv run python -m src.lake.gold_curator --action promote \
  --silver-version v2.1_custom_experiment \
  --dataset-name signals_production \
  --force
```

---

## Managing Experiments

### List All Silver Versions

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

Output:
```json
{
  "version_a": "v1.0_mechanics_only",
  "version_b": "v2.0_full_ensemble",
  "features_added": ["sma_200", "confluence_level", ...],
  "features_removed": [],
  "features_common": ["barrier_state", "tape_imbalance", ...],
  "feature_count_a": 15,
  "feature_count_b": 35,
  "parameters_a": {...},
  "parameters_b": {...}
}
```

### List Gold Datasets

```bash
cd backend
uv run python -m src.lake.gold_curator --action list
```

### Validate Gold Dataset

```bash
cd backend
uv run python -m src.lake.gold_curator --action validate \
  --dataset-name signals_production
```

---

## Best Practices

### Versioning

Use semantic versioning for feature sets:

- **vMAJOR.MINOR_name**
  - `vMAJOR`: Increment for breaking schema changes
  - `MINOR`: Increment for feature additions/refinements
  - `name`: Descriptive name (snake_case)

Examples:
- `v1.0_mechanics_only` - Initial mechanics baseline
- `v1.1_mechanics_enhanced` - Added dealer velocity features
- `v2.0_full_ensemble` - Major change: added TA features
- `v2.1_full_with_confluence` - Minor change: added confluence features

### Experiment Tracking

Always register experiments with meaningful metadata:

```python
builder.register_experiment(
    version='v2.1_custom',
    exp_id='exp003',
    metrics={
        'auc': 0.72,
        'precision': 0.68,
        'recall': 0.65,
        'f1': 0.66
    },
    notes='Hypothesis: Adding X improves Y. Result: +3% AUC'
)
```

### Feature Selection

Start with baseline, iterate incrementally:

1. `v1.0_mechanics_only` - Pure physics (barrier, tape, fuel)
2. `v1.1_mechanics_enhanced` - Add dealer velocity
3. `v2.0_full_ensemble` - Add TA features
4. `v2.1_full_optimized` - Feature selection on v2.0

### Reproducibility

Each Silver version is fully reproducible:

```
Bronze + manifest.yaml → deterministic Silver output
```

Keep manifests in version control:
```bash
git add backend/manifests/*.yaml
git commit -m "feat: new feature set v2.1"
```

---

## Common Workflows

### A/B Testing Features

```python
# Build two versions
builder = SilverFeatureBuilder()

# Version A: Mechanics only
manifest_a = create_mechanics_only_manifest('v1.0_mechanics_only')
builder.build_feature_set(manifest_a, dates=all_dates)

# Version B: With TA
manifest_b = create_full_ensemble_manifest('v2.0_full_ensemble')
builder.build_feature_set(manifest_b, dates=all_dates)

# Train both
df_a = builder.load_features('v1.0_mechanics_only')
df_b = builder.load_features('v2.0_full_ensemble')

# Compare models trained on each
# ...
```

### Incremental Feature Engineering

```python
# Start from existing version
base_manifest = builder.get_manifest('v2.0_full_ensemble')

# Add new features
new_groups = base_manifest.feature_groups + [
    FeatureGroup(
        name="new_features",
        description="Experimental new features",
        columns=["new_feature_1", "new_feature_2"]
    )
]

# Create new version
new_manifest = FeatureManifest(
    version="v2.1_with_new_features",
    name="with_new_features",
    description="Added experimental features X and Y",
    created_at=datetime.utcnow().isoformat() + "Z",
    source=base_manifest.source,
    feature_groups=new_groups,
    parameters=base_manifest.parameters,
    parent_version="v2.0_full_ensemble",  # Track lineage
    tags=["experimental"],
    notes="Testing if new features improve performance"
)

builder.build_feature_set(new_manifest, dates=all_dates)
```

### Production Deployment

```bash
# 1. Train and validate best model
uv run python -m src.ml.boosted_tree_train --silver-version v2.1_best

# 2. Promote to Gold
uv run python -m src.lake.gold_curator --action promote \
  --silver-version v2.1_best \
  --dataset-name signals_production \
  --force

# 3. Validate Gold dataset
uv run python -m src.lake.gold_curator --action validate \
  --dataset-name signals_production

# 4. Deploy model (use Gold data for inference)
# Model will read from: backend/data/lake/gold/training/signals_production.parquet
```

---

## Troubleshooting

### "No Bronze data found"

Ensure Bronze layer is populated:
```bash
cd backend
ls -la data/lake/bronze/futures/trades/symbol=ES/
```

If empty, run backfill:
```bash
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

Investigate null sources in VectorizedPipeline.

---

## File Structure

```
backend/data/
├── lake/
│   ├── bronze/              # Raw normalized events
│   ├── silver/
│   │   └── features/
│   │       ├── v1.0_mechanics_only/
│   │       │   ├── manifest.yaml       # Feature definition
│   │       │   ├── validation.json     # Quality metrics
│   │       │   └── date=YYYY-MM-DD/*.parquet
│   │       ├── v2.0_full_ensemble/
│   │       └── experiments.json        # Experiment registry
│   └── gold/
│       └── training/
│           ├── signals_production.parquet
│           └── signals_production_metadata.json
└── ml/
    └── experiments/
        └── exp001_mechanics_only/
            ├── model.joblib
            └── metrics.json
```

---

## References

- **Architecture**: `backend/DATA_ARCHITECTURE.md`
- **Feature Manifests**: `backend/src/common/schemas/feature_manifest.py`
- **Silver Builder**: `backend/src/lake/silver_feature_builder.py`
- **Gold Curator**: `backend/src/lake/gold_curator.py`
- **Pipeline**: `backend/src/pipeline/vectorized_pipeline.py`

