# ML Module

**Role**: Model training and inference  
**Audience**: Data scientists and backend developers  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Trains boosted-tree multi-head models for tradeability, direction, strength, and time-to-threshold predictions. Provides kNN retrieval for similar historical patterns. PatchTST sequence model serves as baseline.

**Scope**: SPY 0DTE options only (consistent with system scope).

---

## Model Architecture

### Primary: Boosted Trees
Multi-head XGBoost models with walk-forward validation:

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

**Input**: Signals Parquet from vectorized pipeline (via `backend/features.json`)

**Process**:
1. Load signals with features + labels
2. Walk-forward split by date (no random shuffles)
3. Train boosted-tree heads independently
4. Build kNN retrieval index from normalized features
5. Evaluate calibration (reliability curves, Brier scores)
6. Save model bundles + metadata

**Output**: Joblib models in `data/ml/boosted_trees/`, retrieval index in `data/ml/`

---

## Running

### Boosted Trees
```bash
cd backend
uv run python -m src.ml.boosted_tree_train \
  --stage stage_b \
  --ablation all \
  --train-dates 2025-12-14 2025-12-15 \
  --val-dates 2025-12-16
```

### Retrieval Index
```bash
uv run python -m src.ml.build_retrieval_index \
  --stage stage_b \
  --ablation full \
  --k-neighbors 50
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
  --ablation full \
  --test-dates 2025-12-17 2025-12-18
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

**Format**: Joblib serialized dictionaries containing:
- Trained model (XGBClassifier/XGBRegressor)
- Feature names (for consistency check)
- Stage, ablation, head metadata
- Train/val dates
- Evaluation metrics
- Timestamp

**Location**: `data/ml/boosted_trees/{head}_{stage}_{ablation}.joblib`

---

## Experiment Tracking

**MLflow**:
- Experiment: `spymaster_patchtst`
- Logs: params, metrics, model checkpoint, metadata

**W&B** (Weights & Biases):
- Project: `spymaster`
- Logs: per-epoch metrics, model artifact
- Requires: `WANDB_API_KEY` (or `wandb.txt`) or `WANDB_MODE=offline`
- Note: No tracking URL required for W&B; use `WANDB_ENTITY` only if logging to a team namespace.

**Status**:
- PatchTST training already logs to MLflow + W&B (`src/ml/patchtst_train.py`).
- Boosted trees, retrieval index, and calibration eval do not yet emit runs.

**Implementation Plan (MLflow + W&B)**:
- [ ] Add a shared tracking helper (run naming, tags, dataset hash, git SHA).
- [ ] Boosted trees: log hyperparams, stage/ablation, train/val dates, metrics, feature list, model bundle artifact.
- [ ] Retrieval index: log k, normalization settings, feature list, index artifact, retrieval metrics (if available).
- [ ] Calibration eval: log reliability metrics, Brier scores, and calibration curve artifacts.
- [ ] Standardize run IDs so W&B and MLflow share a common `run_name` and metadata tags.
- [ ] Add environment config to `.env`/docs for MLflow (`MLFLOW_TRACKING_URI` optional, `MLFLOW_EXPERIMENT_NAME`) and W&B (`WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`, `WANDB_API_KEY` or `wandb.txt`).
- [ ] Add optional sweep wrappers (MLflow or W&B) for tuning `TOUCH_BAND`, `LOOKFORWARD_MINUTES`, and physics windows.

---

## Common Issues

**Missing features.json**: Pipeline must run first to generate signals  
**No matching dates**: Check train/val date ranges exist in signals Parquet  
**Walk-forward violation**: Ensure no random shuffles in train/val split  
**Model bundle missing**: Run training before attempting live inference  
**W&B auth failure**: Set `WANDB_API_KEY` or use offline mode

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

1. **Update vectorized pipeline** to compute feature
2. **Add to `features.json`** schema
3. **Update `feature_sets.py`** (Stage A or B)
4. **Retrain models** with new feature set
5. **Update INTERFACES.md** to document new feature

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
- **Feature schema**: [../../features.json](../../features.json)
- **Feature sets**: [feature_sets.py](feature_sets.py)
- **Core interface**: [../core/INTERFACES.md](../core/INTERFACES.md) (viewport output)

---

**Scope**: SPY 0DTE only  
**Dependencies**: `common` (schemas), vectorized pipeline (signals Parquet)  
**Integration**: Core Service (live viewport scoring)
