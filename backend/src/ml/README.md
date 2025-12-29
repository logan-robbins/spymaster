# ML Module

**Role**: Model training and inference  
**Audience**: Data scientists and backend developers  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Trains boosted-tree multi-head models for tradeability, direction, strength, and time-to-threshold predictions. Provides kNN retrieval for similar historical patterns. PatchTST sequence model serves as baseline.

**Scope**: ES 0DTE options only (consistent with system scope).

---

## Model Architecture

### Primary: Boosted Trees
Multi-head HistGradientBoosting models with walk-forward validation:

**Heads**:
1. `tradeable_2`: Binary classifier (will move ≥$2.00 in either direction?)
2. `direction`: Binary classifier on tradeable samples (break vs bounce)
3. `strength_signed`: Regressor for movement magnitude
4. `t1_{horizon}s`: Reach probability for threshold 1 (either direction)
5. `t2_{horizon}s`: Reach probability for threshold 2 (either direction)
6. `t1_break_{horizon}s`: Reach probability for threshold 1 in break direction
7. `t1_bounce_{horizon}s`: Reach probability for threshold 1 in bounce direction
8. `t2_break_{horizon}s`: Reach probability for threshold 2 in break direction
9. `t2_bounce_{horizon}s`: Reach probability for threshold 2 in bounce direction

**Feature sets**:
- **Stage A**: Core physics only (barrier, tape, fuel)
- **Stage B**: Stage A + technical analysis (SMA, confluence, approach)

### Secondary: kNN Retrieval
Normalized feature space search for similar historical patterns. Provides ensemble predictions with tree models.

### Baseline: PatchTST
Sequence classification/regression over OHLCV context. Useful for comparing engineered features vs raw sequences.

---

## Training Pipeline

**Input**: Silver feature datasets (versioned experiments from Bronze)

**Process**:
1. Load Silver features for specific version (e.g., `v2.0_full_ensemble`)
2. Walk-forward split by date (no random shuffles)
3. Train boosted-tree heads independently
4. Build kNN retrieval index from normalized features
5. Evaluate calibration (reliability curves, Brier scores)
6. Log experiments to MLflow with metrics
7. Save model bundles + metadata

**Output**: Model bundles in `data/ml/experiments/{exp_id}/`, tracked in MLflow

**Workflow**:
```
Silver Features (versioned, e.g., v2.0_full_ensemble)
    ↓ [MLflow Experiments]
    └─ Train multiple models with different hyperparameters
    └─ Evaluate on validation set
    └─ Compare experiments in MLflow
    ↓ [Promote Best]
Gold Training Dataset (curated from best Silver version)
    ↓ [Final Production Training]
Model Artifacts (production-ready, final hyperparameters)
```

---

## Running

### Boosted Trees
```bash
cd backend
# Train using features.json output_path by default
uv run python -m src.ml.boosted_tree_train \
  --stage stage_b \
  --ablation all

# Override dataset path (Gold or Silver parquet)
uv run python -m src.ml.boosted_tree_train \
  --data-path data/lake/gold/training/signals_production.parquet \
  --stage stage_b \
  --ablation full
```

### Retrieval Index
```bash
uv run python -m src.ml.build_retrieval_index \
  --stage stage_b \
  --ablation full
```

### PatchTST Baseline
```bash
# Build sequence dataset
uv run python -m src.ml.sequence_dataset_builder --date 2025-12-16

# Train model
uv run python -m src.ml.patchtst_train \
  --train-files sequence_dataset_*.npz \
  --val-files sequence_dataset_*.npz \
  --epochs 100
```

### Calibration Evaluation
```bash
uv run python -m src.ml.calibration_eval \
  --stage stage_b \
  --ablation full
```

---

## Live Inference (Viewport Scoring)

**Integration**: Live scoring runs inside Core Service when enabled.

**Prerequisites**:
- Boosted-tree models: `data/ml/boosted_trees/`
- Retrieval index: `data/ml/retrieval_index.joblib`
- Environment variables:
  ```bash
  VIEWPORT_SCORING_ENABLED=true
  VIEWPORT_MODEL_DIR=data/ml/boosted_trees
  VIEWPORT_RETRIEVAL_INDEX=data/ml/retrieval_index.joblib
  VIEWPORT_TIMEFRAME=4min  # optional: use 2min/4min/8min heads for tradeable/direction/strength
  ```

**Output**: Published to `viewport.targets` in level signals payload (optional field).

**See**: [INTERFACES.md](INTERFACES.md) for viewport target schema.

---

## Feature Engineering

**Stage A** (core physics):
- Barrier: `barrier_delta_liq`, `barrier_replenishment_ratio`, `wall_ratio`
- Tape: `tape_imbalance`, `tape_velocity`, `sweep_detected`
- Fuel: `gamma_exposure`, `fuel_effect`

**Stage B** (+ technical analysis):
- SMA: `sma_200_distance`, `sma_400_distance`, `sma_slope_short`
- Confluence: `confluence_count`, `confluence_score`, `confluence_min_distance`
- Approach: `approach_velocity`, `approach_bars`, `prior_touches`

**Feature sets**: Defined in `feature_sets.py` with ablation support.

---

## Walk-Forward Validation

**Critical**: Models MUST use walk-forward splits (no random shuffles).

**Rationale**: Prevents look-ahead bias, matches live deployment conditions.

**Implementation**: Sort dates chronologically, split by cutoff date.

---

## Model Bundles

**Format**: Joblib serialized sklearn pipelines containing:
- Trained HistGradientBoosting models with preprocessing
- Feature names (for consistency check)
- Stage, ablation, head metadata
- Train/val/test date ranges
- Evaluation metrics
- Timestamp

**Location**: `data/ml/boosted_trees/{head}_{stage}_{ablation}.joblib`

---

## Experiment Tracking

**MLflow** (Primary for Silver Experiments):
- Experiments track Silver feature versions: `spymaster_v2.0_full_ensemble`, `spymaster_v1.0_mechanics_only`, etc.
- Each run logs:
  - Hyperparameters (stage, ablation, val/test sizes, HistGradientBoosting params)
  - Metrics (AUC, precision, recall, Brier scores per head)
  - Artifacts (model bundles, feature importance, calibration curves)
  - Dataset metadata (Silver version, manifest hash, signal count)
- Compare experiments across Silver versions to select best for Gold promotion

**W&B** (Weights & Biases):
- Project: `spymaster`
- Logs: per-epoch metrics, model artifact (primarily for PatchTST)
- Requires: `WANDB_API_KEY` (or `wandb.txt`) or `WANDB_MODE=offline`
- Note: No tracking URL required for W&B; use `WANDB_ENTITY` only if logging to a team namespace.

**Experiment Workflow**:
1. Create Silver feature version (e.g., `v2.1_custom`)
2. Run MLflow experiments on that Silver version
3. Compare metrics across runs in MLflow UI
4. Identify best model/hyperparameters
5. Promote winning Silver version to Gold (`GoldCurator.promote_to_training()`)
6. Optional: Run final production training on Gold with winning hyperparameters

**Status**:
- PatchTST training logs to MLflow + W&B (`src/ml/patchtst_train.py`)
- Boosted trees, retrieval index, and calibration eval will log to MLflow (implementation in progress)

---

## Common Issues

**No Silver data**: Run `SilverFeatureBuilder.build_feature_set()` to create versioned features from Bronze  
**Not enough dates**: Ensure dataset has ≥ (val_size + test_size + 1) unique dates  
**Walk-forward violation**: Ensure no random shuffles in train/val split  
**Model bundle missing**: Run training before attempting live inference  
**W&B auth failure**: Set `WANDB_API_KEY` or use offline mode  
**Wrong Gold dataset**: Ensure correct Silver version was promoted to Gold

---

## Testing

```bash
cd backend
uv run pytest tests/test_research_lab.py -v
# Tests feature engineering and model training workflows
```

---

## Adding New Features

To add new features to models:

1. **Add stage to pipeline** (in `src/pipeline/stages/`)
2. **Create new Silver version** with updated feature manifest
3. **Update `feature_sets.py`** (Stage A or B) if needed
4. **Run MLflow experiments** on new Silver version
5. **Compare with baseline** in MLflow UI
6. **Promote to Gold** if better performance
7. **Update INTERFACES.md** to document new feature

---

## Critical Invariants

1. **Walk-forward only**: No random train/val splits
2. **Feature stability**: Same features for training and inference
3. **Label anchoring**: All labels anchored at `t1` (confirmation time) and measured in the level frame
4. **No leakage**: Features computed from data before `t1`
5. **Deterministic splits**: Same dates → same train/val split
6. **Model versioning**: Bundle includes timestamp and data hash

---

## References

- **Interface contract**: [INTERFACES.md](INTERFACES.md)
- **Data Architecture**: [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md) (Bronze/Silver/Gold)
- **Feature manifests**: [../common/schemas/feature_manifest.py](../common/schemas/feature_manifest.py)
- **Feature sets**: [feature_sets.py](feature_sets.py)
- **Silver builder**: [../lake/silver_feature_builder.py](../lake/silver_feature_builder.py)
- **Gold curator**: [../lake/gold_curator.py](../lake/gold_curator.py)
- **Core interface**: [../core/INTERFACES.md](../core/INTERFACES.md) (viewport output)

---

**Scope**: ES 0DTE only  
**Dependencies**: `common` (schemas), `lake` (Silver/Gold datasets), `pipeline` (versioned pipelines)  
**Integration**: Core Service (live viewport scoring)
