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

**Source**: Silver feature datasets (versioned experiments from Bronze)

**Location**: `data/lake/silver/features/{version}/date=YYYY-MM-DD/*.parquet`

**Access**:
```python
from src.lake.silver_feature_builder import SilverFeatureBuilder

builder = SilverFeatureBuilder()
df = builder.load_features('v2.0_full_ensemble')
```

**Schema**: Versioned feature set with:
- Event identity: `event_id`, `ts_ns`, `date`, `symbol`
- Level context: `level_price`, `level_kind_name`, `direction`, `distance`
- Features: Barrier, tape, fuel, approach, confluence metrics (per manifest)
- Labels: `outcome`, `strength_signed`, `tradeable_1`, `tradeable_2`, time-to-threshold

**Required Columns**:
- Identity: `event_id`, `ts_ns`, `date`
- Context: `level_kind_name`, `direction`, `distance`
- Physics: All numeric barrier/tape/fuel features
- Labels: `tradeable_2`, `strength_signed`, `t1_60`, `t1_120`, `t2_60`, `t2_120`, `t1_break_60`, `t1_bounce_60`, `t2_break_60`, `t2_bounce_60`

**Gold Production Data** (optional, after promotion):
- Location: `data/lake/gold/training/signals_production.parquet`
- Same schema as Silver, curated from best experiment

---

## Feature Sets Interface

**Module**: `feature_sets.py`

```python
@dataclass
class FeatureSet:
    numeric: List[str]
    categorical: List[str]

def select_features(
    df: pd.DataFrame,
    stage: str,           # 'stage_a' or 'stage_b'
    ablation: str = 'full'  # 'ta', 'mechanics', or 'full'
) -> FeatureSet
```

**Ablation Types**:
- `ta`: Technical analysis features only (approach, distance, SMA, confluence)
- `mechanics`: Market mechanics features only (barrier, tape, fuel, gamma)
- `full`: All features combined

**Feature Prefix Categories**:

**Stage B Only Prefixes** (excluded from Stage A):
- `barrier_`, `tape_`, `wall_ratio`, `liquidity_pressure`
- `tape_pressure`, `net_break_pressure`
- Trend features: `*_trend`

**TA Prefixes**:
- `approach_`, `dist_`, `distance`, `sma_`, `mean_reversion_`
- `confluence_`, `bars_since_open`, `is_first_15m`, `atr`
- `level_price_pct`, `direction_sign`, `attempt_`, `prior_touches`

**Mechanics Prefixes**:
- `barrier_`, `tape_`, `fuel_`, `gamma_`, `wall_ratio`
- `liquidity_pressure`, `tape_pressure`, `gamma_pressure`
- `dealer_pressure`, `net_break_pressure`, `reversion_pressure`

---

## Model Interfaces

### Boosted Trees (Primary)

**Training Script**: `boosted_tree_train.py`

**CLI**:
```bash
# Train using features.json output_path by default
uv run python -m src.ml.boosted_tree_train \
  --stage stage_b \
  --ablation full

# Override dataset path (Gold or Silver parquet)
uv run python -m src.ml.boosted_tree_train \
  --data-path data/lake/gold/training/signals_production.parquet \
  --stage stage_b \
  --ablation full
```

**Ablation Options**: `ta`, `mechanics`, `full`

**Model Heads**:
1. `tradeable_2`: Binary classifier (tradeable vs not)
2. `direction`: Binary classifier on tradeable samples (break vs bounce)
3. `strength`: Regressor for movement magnitude
4. `t1_{horizon}s`: Reach probability for threshold 1 (either direction)
5. `t2_{horizon}s`: Reach probability for threshold 2 (either direction)
6. `t1_break_{horizon}s`: Reach probability for threshold 1 in break direction
7. `t1_bounce_{horizon}s`: Reach probability for threshold 1 in bounce direction
8. `t2_break_{horizon}s`: Reach probability for threshold 2 in break direction
9. `t2_bounce_{horizon}s`: Reach probability for threshold 2 in bounce direction

**Output Location**: 
- Experiments: `data/ml/experiments/{exp_id}/model.joblib`
- Production (after promotion): `data/ml/production/boosted_trees/`
- Models: `{head}_{stage}_{ablation}.joblib`
- Metadata: `metadata_{stage}_{ablation}.json`
- MLflow tracking: `mlruns/`

---

### Tree Inference Interface

**Module**: `tree_inference.py`

```python
@dataclass
class TreePredictions:
    tradeable_2: np.ndarray       # P(tradeable)
    p_break: np.ndarray           # P(break | tradeable)
    strength_signed: np.ndarray   # Predicted signed strength
    t1_probs: Dict[int, np.ndarray]  # {horizon: P(reach t1)}
    t2_probs: Dict[int, np.ndarray]  # {horizon: P(reach t2)}
    t1_break_probs: Dict[int, np.ndarray]  # {horizon: P(reach t1 break)}
    t1_bounce_probs: Dict[int, np.ndarray]  # {horizon: P(reach t1 bounce)}
    t2_break_probs: Dict[int, np.ndarray]  # {horizon: P(reach t2 break)}
    t2_bounce_probs: Dict[int, np.ndarray]  # {horizon: P(reach t2 bounce)}

class TreeModelBundle:
    def __init__(
        self,
        model_dir: Path,
        stage: str,               # 'stage_a' or 'stage_b'
        ablation: str,            # 'ta', 'mechanics', 'full'
        horizons: List[int],      # e.g., [60, 120]
        timeframe: Optional[str] = None  # e.g., '2min', '4min', '8min'
    )

    def predict(self, df: pd.DataFrame) -> TreePredictions
```

**Model Loading**:
- Expects models at: `{model_dir}/{head}_{stage}_{ablation}.joblib`
- If `timeframe` is set, uses `{head}_{timeframe}_{stage}_{ablation}.joblib` for tradeable/direction/strength.
- Heads loaded: `tradeable_2`, `direction`, `strength`, `t1_{h}s`, `t2_{h}s`, `t1_break_{h}s`, `t1_bounce_{h}s`, `t2_break_{h}s`, `t2_bounce_{h}s`

---

### Retrieval Engine (kNN)

**Module**: `retrieval_engine.py`

**Index Builder**: `build_retrieval_index.py`

**CLI**:
```bash
uv run python -m src.ml.build_retrieval_index \
  --stage stage_b \
  --ablation full \
  --k-neighbors 50
```

**Output**: `data/ml/retrieval_index_{stage}_{ablation}.joblib`

```python
@dataclass
class RetrievalSummary:
    p_break: float
    p_bounce: float
    p_tradeable_2: float
    strength_signed_mean: float
    strength_abs_mean: float
    time_to_threshold_1_mean: float
    time_to_threshold_2_mean: float
    time_to_break_1_mean: float
    time_to_bounce_1_mean: float
    time_to_break_2_mean: float
    time_to_bounce_2_mean: float
    similarity: float              # Mean 1/(1+distance)
    entropy: float                 # Outcome distribution entropy
    neighbors: pd.DataFrame        # Neighbor metadata

class RetrievalIndex:
    def __init__(
        self,
        feature_cols: List[str],
        metadata_cols: Optional[List[str]] = None  # Default: ["level_kind_name", "direction"]
    )

    def fit(self, df: pd.DataFrame) -> None
    def query(
        self,
        feature_vector: np.ndarray,
        filters: Optional[Dict[str, str]] = None,
        k: int = 20
    ) -> RetrievalSummary
```

**Query Flow**:
1. Normalize query vector using fitted StandardScaler
2. Apply optional metadata filters (level_kind_name, direction)
3. Compute Euclidean distances to candidates
4. Select top-k neighbors
5. Weight by inverse distance
6. Compute weighted outcome probabilities

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
        't2': {'60': float, '120': float},
        't1_break': {'60': float, '120': float},
        't1_bounce': {'60': float, '120': float},
        't2_break': {'60': float, '120': float},
        't2_bounce': {'60': float, '120': float}
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
        'time_to_break_1_mean': float,
        'time_to_bounce_1_mean': float,
        'time_to_break_2_mean': float,
        'time_to_bounce_2_mean': float,
        'similarity': float,
        'entropy': float,
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
    'ablation': str,        # 'full' | 'ta' | 'mechanics'
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
3. **Label anchoring**: All labels anchored at `t1` (confirmation time) and measured in the level frame
4. **No leakage**: Features computed from data before `t1`
5. **Deterministic splits**: Same dates → same train/val split
6. **Model versioning**: Bundle includes timestamp and data hash
7. **Ablation names**: Use `ta`, `mechanics`, `full` (NOT `ta_only`, `mechanics_only`)

---

## References

- Full module documentation: `backend/src/ml/README.md`
- Data Architecture: `backend/DATA_ARCHITECTURE.md` (Bronze/Silver/Gold)
- Feature manifests: `backend/src/common/schemas/feature_manifest.py`
- Feature sets: `backend/src/ml/feature_sets.py`
- Training scripts: `backend/src/ml/boosted_tree_train.py`, `backend/src/ml/patchtst_train.py`
- Inference: `backend/src/ml/tree_inference.py`, `backend/src/ml/retrieval_engine.py`
- Silver builder: `backend/src/lake/silver_feature_builder.py`
- Gold curator: `backend/src/lake/gold_curator.py`
