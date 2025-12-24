# ML Module Interfaces

**Module**: `backend/src/ml/`  
**Role**: Model training and inference  
**Audience**: AI Coding Agents

---

## Module Purpose

Trains boosted-tree multi-head models for tradeability, direction, strength, and time-to-threshold predictions. Provides kNN retrieval and PatchTST sequence baseline.

---

## Input Interfaces

### Training Data

**Source**: `backend/features.json` → `output_path` (signals Parquet)  
**Schema**: Vectorized pipeline output with:
- Event identity: `event_id`, `ts_ns`, `date`, `symbol`
- Level context: `level_price`, `level_kind_name`, `direction`, `distance`
- Features: Barrier, tape, fuel, approach, confluence metrics
- Labels: `outcome`, `strength_signed`, `tradeable_1`, `tradeable_2`, time-to-threshold

**Required Columns** (from `features.json`):
- Identity: `event_id`, `ts_ns`, `date`
- Context: `level_kind_name`, `direction`, `distance`
- Physics: All numeric barrier/tape/fuel features
- Labels: `tradeable_2`, `strength_signed`, `t1_60`, `t1_120`, `t2_60`, `t2_120`

---

## Model Interfaces

### Boosted Trees (Primary)

**Training Script**: `boosted_tree_train.py`

**CLI**:
```bash
uv run python -m src.ml.boosted_tree_train \
  --stage stage_b \
  --ablation all \
  --train-dates 2025-12-14 2025-12-15 \
  --val-dates 2025-12-16
```

**Feature Sets** (from `feature_sets.py`):
- **Stage A**: Core physics only (barrier, tape, fuel)
- **Stage B**: Stage A + technical analysis (SMA, confluence, approach)
- **Ablations**: `ta_only`, `mechanics_only`, `full`

**Model Heads**:
1. `tradeable_2`: Binary classifier (tradeable vs not)
2. `direction`: Binary classifier on tradeable samples (break vs bounce)
3. `strength_signed`: Regressor for movement magnitude
4. `t1_{horizon}s`: Reach probability for threshold 1
5. `t2_{horizon}s`: Reach probability for threshold 2

**Output Location**: `data/ml/boosted_trees/`
- Models: `{head}_{stage}_{ablation}.joblib`
- Metadata: `metadata_{stage}_{ablation}.json`

---

### Retrieval Engine (kNN)

**Index Builder**: `build_retrieval_index.py`

**CLI**:
```bash
uv run python -m src.ml.build_retrieval_index \
  --stage stage_b \
  --ablation full \
  --k-neighbors 50
```

**Index Format**: Joblib serialized
- Feature matrix (normalized)
- Labels (outcomes, strength)
- Metadata (event_ids, timestamps)

**Output**: `data/ml/retrieval_index_{stage}_{ablation}.joblib`

**Query Interface**:
```python
from src.ml.retrieval_engine import RetrievalEngine

engine = RetrievalEngine.load('data/ml/retrieval_index_stage_b_full.joblib')

neighbors = engine.query(
    features=current_features,  # numpy array
    k=50
)
# Returns: neighbor indices, distances, labels
```

---

### PatchTST (Sequence Baseline)

**Dataset Builder**: `sequence_dataset_builder.py`

**CLI**:
```bash
uv run python -m src.ml.sequence_dataset_builder \
  --date 2025-12-16 \
  --seq-len 60 \
  --output sequence_dataset_2025-12-16.npz
```

**Output Schema**:
```python
{
    'X': (n_samples, seq_len, n_features),  # OHLCV sequences
    'mask': (n_samples, seq_len),           # Padding mask
    'static': (n_samples, n_static),        # Event context
    'y_break': (n_samples,),                # 1=BREAK, 0=BOUNCE, -1=other
    'y_strength': (n_samples,),             # Signed strength
    'event_id': (n_samples,),
    'ts_ns': (n_samples,),
    'seq_feature_names': List[str],
    'static_feature_names': List[str]
}
```

**Training Script**: `patchtst_train.py`

**CLI**:
```bash
uv run python -m src.ml.patchtst_train \
  --train-files sequence_dataset_2025-12-14.npz sequence_dataset_2025-12-15.npz \
  --val-files sequence_dataset_2025-12-16.npz \
  --epochs 100 \
  --batch-size 64
```

**Output**: `patchtst_multitask.pt` (PyTorch checkpoint)

---

## Inference Interfaces

### Live Viewport Scoring

**Prerequisites**:
- Boosted-tree models: `data/ml/boosted_trees/`
- Retrieval index: `data/ml/retrieval_index.joblib`
- Environment variables:
  - `VIEWPORT_SCORING_ENABLED=true`
  - `VIEWPORT_MODEL_DIR` (default: `data/ml/boosted_trees`)
  - `VIEWPORT_RETRIEVAL_INDEX` (default: `data/ml/retrieval_index.joblib`)

**Inference Flow** (in Core Service):
1. Compute engineered features for target level
2. Load Stage A model → get baseline predictions
3. Load Stage B model → get enhanced predictions
4. Query retrieval engine → get kNN neighbors
5. Ensemble tree + retrieval predictions
6. Apply feasibility gate (deterministic rules)
7. Publish to `viewport.targets` in level signals payload

**Output Fields** (per target):
```python
{
    'level_id': str,
    'level_price': float,
    'direction': str,
    'distance': float,
    'distance_signed': float,
    
    # Tree predictions
    'p_tradeable_2': float,
    'p_break': float,
    'p_bounce': float,
    'strength_signed': float,
    'strength_abs': float,
    'time_to_threshold': {
        't1': {'60': float, '120': float},
        't2': {'60': float, '120': float}
    },
    
    # Retrieval predictions
    'retrieval': {
        'p_break': float,
        'p_bounce': float,
        'p_tradeable_2': float,
        'strength_signed_mean': float,
        'strength_abs_mean': float,
        'time_to_threshold_1_mean': float,
        'time_to_threshold_2_mean': float,
        'neighbors': []  # Optional: neighbor metadata
    },
    
    # Scoring metadata
    'utility_score': float,
    'viewport_state': str,  # 'IN_MONITOR_BAND' | 'OUTSIDE_BAND'
    'stage': str,           # 'stage_a' | 'stage_b'
    'pinned': bool,
    'relevance': float
}
```

---

## Calibration Interface

**Script**: `calibration_eval.py`

**CLI**:
```bash
uv run python -m src.ml.calibration_eval \
  --stage stage_b \
  --ablation full \
  --test-dates 2025-12-17 2025-12-18
```

**Output**:
- Reliability curves (predicted vs observed probabilities)
- Brier scores per head
- Calibration metrics (ECE, MCE)

---

## Model Bundle Format

### Boosted Tree Bundle (Joblib)

```python
{
    'model': XGBClassifier | XGBRegressor,
    'feature_names': List[str],
    'stage': str,           # 'stage_a' | 'stage_b'
    'ablation': str,        # 'full' | 'ta_only' | 'mechanics_only'
    'head': str,            # 'tradeable_2' | 'direction' | ...
    'train_dates': List[str],
    'val_dates': List[str],
    'metrics': Dict[str, float],
    'timestamp': str
}
```

### Retrieval Index (Joblib)

```python
{
    'feature_matrix': np.ndarray,  # (n_samples, n_features)
    'labels': {
        'outcome': np.ndarray,
        'strength_signed': np.ndarray,
        'tradeable_2': np.ndarray,
        't1_60': np.ndarray,
        # ... other labels
    },
    'metadata': {
        'event_ids': List[str],
        'timestamps': List[int],
        'dates': List[str]
    },
    'feature_names': List[str],
    'stage': str,
    'ablation': str,
    'scaler': StandardScaler,  # For normalization
    'k_neighbors': int
}
```

---

## Walk-Forward Splits Contract

**Requirement**: Models MUST use walk-forward validation (no random shuffles).

**Split Strategy**:
```python
# Sort dates chronologically
dates = sorted(train_dates + val_dates)

# Split by date cutoff
train_mask = df['date'].isin(train_dates)
val_mask = df['date'].isin(val_dates)

X_train = X[train_mask]
X_val = X[val_mask]
```

**Rationale**: Prevents look-ahead bias, matches live deployment conditions.

---

## Feature Engineering Contract

**Stage A Features** (core physics):
- Barrier: `barrier_delta_liq`, `barrier_replenishment_ratio`, `wall_ratio`
- Tape: `tape_imbalance`, `tape_velocity`, `sweep_detected`
- Fuel: `gamma_exposure`, `fuel_effect`

**Stage B Features** (+ technical analysis):
- Stage A features +
- SMA: `sma_200_distance`, `sma_400_distance`, `sma_slope_short`, `sma_slope_long`
- Confluence: `confluence_count`, `confluence_score`, `confluence_min_distance`
- Approach: `approach_velocity`, `approach_bars`, `prior_touches`

**Missing Value Handling**:
- SMA features: NaN for first `SMA_WARMUP_DAYS` sessions → use warmup
- Confluence features: 0 when no other levels nearby
- Approach features: 0 for first touch at level

---

## Critical Invariants

1. **Walk-forward only**: No random train/val splits
2. **Feature stability**: Same features for training and inference
3. **Label anchoring**: All labels anchored at `t1` (confirmation time)
4. **No leakage**: Features computed from data before `t1`
5. **Deterministic splits**: Same dates → same train/val split
6. **Model versioning**: Bundle includes timestamp and data hash

---

## References

- Full module documentation: `backend/src/ml/README.md`
- Feature contract: `backend/features.json`
- Feature sets: `backend/src/ml/feature_sets.py`
- Training scripts: `backend/src/ml/boosted_tree_train.py`, `backend/src/ml/patchtst_train.py`

