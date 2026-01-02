# Pentaview Overview

Pentaview is a **stream-based signal processing system** that transforms the raw 30-second state table (from the main pipeline's Stage 16) into continuous, normalized scalar streams that emit values in `[-1, +1]` every 2 minutes. It's designed to provide TA-style (Technical Analysis) interpretable signals for discretionary trading decisions.

---

### Architecture

The system has a clean pipeline structure:

**Pipeline Flow:**
1. **Input**: 30-second state table from `es_pipeline` Stage 16
2. **Aggregation**: 30s samples → 2-minute bars (via `ComputeStreamsStage`)
3. **Normalization**: Robust statistics (median/MAD) with stratification
4. **Stream Computation**: 5 canonical streams + derivatives
5. **Output**: Parquet files with 32 columns of normalized signals

---

### Core Components

#### 1. **Normalization System** (`stream_normalization.py`)

Before computing streams, raw features must be normalized to `[-1, +1]`:

**Two Methods:**
- **Robust** (for heavy-tailed distributions like OFI, tape, GEX):
  ```
  robust_z = (x - median) / (1.4826 * MAD)
  norm = tanh(clip(robust_z, -6, +6) / 2.0)
  ```
  
- **Z-score** (for symmetric distributions like kinematics):
  ```
  z = (x - mean) / std
  norm = tanh(clip(z, -6, +6) / 2.0)
  ```

**Stratification**: Stats are computed separately for different time buckets (T0-15, T15-30, etc.) to handle regime changes during the trading session.

#### 2. **Five Canonical Streams** (`stream_builder.py`)

Each stream captures a specific market dynamic:

**Σ_M (MOMENTUM)** - Directional price dynamics:
- Formula: `0.40*velocity + 0.30*acceleration + 0.15*jerk + 0.15*momentum_trend`
- Inputs: Multi-scale kinematics (1-20 min windows)
- Interpretation: `> 0` = upward momentum, `< 0` = downward

**Σ_F (FLOW)** - Order aggression:
- Formula: `0.25*ofi_core + 0.20*ofi_level + 0.25*tape_imb + 0.15*ofi_acc + 0.10*alignment + 0.05*dct_shape`
- Inputs: OFI (Order Flow Imbalance), tape statistics, DCT trajectory
- Interpretation: `> 0` = net buying pressure, `< 0` = net selling

**Σ_B (BARRIER)** - Liquidity dynamics:
- Formula: `dir_sign * tanh(0.50*consume + 0.25*rate + 0.15*state + 0.10*repl)`
- **Critical**: Applies `dir_sign` multiplier (+1 for UP approaches, -1 for DOWN)
- Interpretation: `> 0` = barrier favors break up, `< 0` = favors break down

**Σ_D (DEALER/GAMMA)** - Non-directional amplification:
- Formula: `0.45*fuel_effect - 0.25*gamma_exposure - 0.15*gex_ratio - 0.15*abs(local_gex)`
- Interpretation: `> 0` = amplification (fuel), `< 0` = dampening (pin)

**Σ_S (SETUP)** - Quality/confidence scalar:
- Formula: Weighted combination of proximity, freshness, confluence, approach speed, trajectory cleanness
- Maps `[0,1] → [-1,+1]`
- Interpretation: `> +0.5` = high quality setup, `< -0.5` = degraded

#### 3. **Merged Streams**

**Σ_P (PRESSURE)** = Momentum + Flow:
```python
sigma_p = tanh(0.55*sigma_m + 0.45*sigma_f)
```
Primary directional signal.

**Σ_R (STRUCTURE)** = Barrier + Setup:
```python
sigma_r = tanh(0.70*sigma_b + 0.30*sigma_s)
```
Measures structural support for moves.

#### 4. **Derivatives** (TA-style acceleration indicators)

For each stream, computes via EMA smoothing:
- **Slope** (1st difference): Rate of change
- **Curvature** (2nd difference): Acceleration
- **Jerk** (3rd difference): Rate of acceleration change

These enable detection of exhaustion, continuation, and reversal patterns.

#### 5. **DCT Trajectory Encoding**

Uses **Discrete Cosine Transform** on 40-bar (20-minute) lookback windows for 4 key series:
- Distance from level (`d_atr`)
- Order flow (`ofi_60s`)
- Barrier liquidity (`barrier_delta_liq_log`)
- Tape imbalance

Extracts first 8 DCT coefficients to capture:
- `c1`: Trend component
- `c2`: Curvature
- `c3-c7`: Higher-frequency patterns

These feed into flow and setup streams as "shape" features.

---

### 6. **Forward Projections** (`stream_projector.py`)

**Goal**: Forecast stream values 20 minutes ahead with uncertainty bands.

**Method**: Quantile polynomial regression
- Predicts **polynomial coefficients** `{a1, a2, a3}` for each quantile (q10, q50, q90)
- Coefficients represent: slope, curvature, jerk
- Generates smooth curves (no jagged forecasts)

**Curve Formula:**
```
ŷ(h) = σ[t] + a1*h + 0.5*a2*h² + (1/6)*a3*h³
```

**Training:**
- Uses 20-bar stream history + derivatives + cross-stream context
- Trains separate HistGradientBoostingRegressor for each quantile
- MultiOutputRegressor predicts 3 coefficients simultaneously

**Output:** 11-point curves (current + 10 future bars) for q10/q50/q90

---

### 7. **State Machine & Alerts** (`stream_state_machine.py`)

**Purpose**: Rule-based interpretation layer for discretionary trading.

**14 Alert Types:**

**Exhaustion/Continuation:**
- `CONTINUATION_UP`: P > +0.35 and slope > +0.05
- `EXHAUSTION_UP`: P > +0.35 but slope < 0 (buying fading)
- `REVERSAL_RISK_UP`: P > +0.35, slope < 0, curvature < 0, |jerk| > thresh

**Divergence:**
- `FLOW_DIVERGENCE`: Flow and momentum disagree (squeeze/trap detector)
- `FLOW_CONFIRMATION`: Flow and momentum aligned

**Barrier Phases:**
- `BARRIER_BREAK_SUPPORT`: Barrier favors break through
- `BARRIER_OPPOSES_PRESSURE`: Barrier resisting pressure
- `BARRIER_WEAKENING`: Barrier losing strength

**Quality Gates:**
- `HIGH_QUALITY_SETUP`: sigma_s > +0.25
- `LOW_QUALITY_SETUP`: sigma_s < -0.25
- `FUEL_REGIME`: sigma_d > +0.25 (amplification)
- `PIN_REGIME`: sigma_d < -0.25 (dampening)

**Exit Scoring:**
Position-aware recommendations (`HOLD`, `REDUCE`, `EXIT`) based on:
- Pressure direction vs position
- Flow confirmation
- Barrier opposition
- Setup quality

**Hysteresis:** Prevents alert flickering by requiring sustained conditions (>5 bars).

---

### Data Flow Example

```
Bronze (ES futures trades)
    ↓
Silver (OHLCV bars, kinematics, OFI, barriers, GEX)
    ↓
Stage 16: State Table (30s cadence)
    ↓
Pentaview Pipeline:
    1. Aggregate 30s → 2min bars
    2. Load normalization stats (median/MAD per stratum)
    3. For each level:
        - Compute 5 canonical streams (M,F,B,D,S)
        - Compute merged streams (P,R)
        - Compute derivatives (slope, curvature, jerk)
        - Compute DCT coefficients on 20-min history
    ↓
Gold Layer: stream_bars.parquet (32 columns)
    ↓
Optional:
    - Projection models → 20-min forecasts with uncertainty
    - State machine → Alerts for UI
```

---

### Key Design Principles

1. **Bounded Outputs**: All streams in `[-1, +1]` via tanh squashing
2. **Multi-Scale**: Combines 30s-20min dynamics
3. **Robust**: Median/MAD normalization handles outliers
4. **Stratified**: Different stats per time-of-day regime
5. **Interpretable**: Each stream has clear market semantics
6. **Smooth**: Derivatives via EMA, projections via polynomials
7. **Online-Safe**: All features use only past data

---

### Output Schema

**Stream Bars** (`gold/streams/pentaview/version=3.1.0/date=YYYY-MM-DD/stream_bars.parquet`):

32 columns per 2-min bar:
- Metadata: `timestamp`, `level_kind`, `direction`, `spot`, `atr`, `level_price`
- Canonical: `sigma_m`, `sigma_f`, `sigma_b`, `sigma_d`, `sigma_s`
- Merged: `sigma_p`, `sigma_r`
- Composites: `alignment`, `divergence`, `alignment_adj`
- Derivatives (×4 streams): `sigma_p_smooth`, `sigma_p_slope`, `sigma_p_curvature`, `sigma_p_jerk`, ...

---

### Testing & Validation

**Scripts:**
- `compute_stream_normalization.py` - Compute robust stats from 60-day lookback
- `run_pentaview_pipeline.py` - Run for single date or batch
- `validate_pentaview.py` - Validate outputs
- `demo_projection.py` - Test projection models
- `demo_state_machine.py` - Test alert detection

---

**Summary:**

Pentaview transforms raw market physics features into TA-interpretable scalar signals, enabling the UI to present probabilistic guidance like:  
*"At 9:56 AM, approaching OR_HIGH from below: Pressure = +0.67 (strong), Flow = +0.55 (buying), Barrier = -0.42 (opposing), Setup = +0.71 (high quality). Historical similar setups broke through 58% of the time. 20-min projection shows momentum peaking then fading."*
- Pentaview is a 7-component stream processing system: normalization, 5 canonical streams, projections, and state machine
- Converts 30s state samples → 2min bars → bounded [-1,+1] signals with TA semantics
- Provides 14 alert types, 20-min forecasts, and position-aware exit scoring

## Pentaview Training & Evaluation

### Overview

Pentaview has **two distinct training components**:
1. **Normalization Statistics** (pre-training step)
2. **Projection Models** (stream forecasting models)

---

## 1. Compute Normalization Statistics

**Purpose**: Calculate robust median/MAD statistics for feature normalization

**Script**: `backend/scripts/compute_stream_normalization.py`

**What it does**:
- Loads 30s state table from Stage 16 output
- Computes stratified statistics (by time bucket)
- Saves to `data/gold/normalization/current.json`

**Command**:
```bash
cd backend

# Compute stats from 60-day lookback
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 60 \
  --end-date 2024-12-31

# Custom output path
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 60 \
  --end-date 2024-12-31 \
  --output-path data/gold/normalization/stats_v002.json
```

**Output**:
```json
{
  "version": "1.0",
  "created_at": "2024-12-30T...",
  "n_samples": 125000,
  "stratify_by": ["time_bucket"],
  "global_stats": {
    "velocity_1min": {"method": "zscore", "mean": 0.12, "std": 0.45},
    "ofi_60s": {"method": "robust", "median": 125.0, "mad": 450.0},
    ...
  },
  "stratified_stats": {
    "T0_15": {...},
    "T15_30": {...},
    ...
  }
}
```

---

## 2. Build Projection Training Dataset

**Purpose**: Extract training samples (stream history + future targets) for projection models

**Script**: `backend/scripts/build_projection_dataset.py`

**What it does**:
- Reads stream bars from Pentaview pipeline output
- For each bar with sufficient history:
  - Extracts L=20 bars of history (stream values, slopes, cross-streams)
  - Extracts H=10 bars of future target
  - Computes setup quality weight
  - Saves as compressed `.npz` files

**Command**:
```bash
cd backend

# Build dataset for specific streams
uv run python -m scripts.build_projection_dataset \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --streams sigma_p,sigma_m,sigma_f,sigma_b,sigma_r \
  --output-dir data/gold/training/projection_samples

# Build for single stream
uv run python -m scripts.build_projection_dataset \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --streams sigma_p
```

**Output Files**:
```
data/gold/training/projection_samples/
├── projection_samples_sigma_p_v1.npz      # Pressure stream
├── projection_samples_sigma_m_v1.npz      # Momentum stream
├── projection_samples_sigma_f_v1.npz      # Flow stream
├── projection_samples_sigma_b_v1.npz      # Barrier stream
└── projection_samples_sigma_r_v1.npz      # Structure stream
```

**Sample Structure** (inside .npz):
```python
{
    'stream_hist': np.ndarray,          # [N, L=20] history
    'slope_hist': np.ndarray,           # [N, L=20] slope history
    'current_value': np.ndarray,        # [N] current stream value
    'future_target': np.ndarray,        # [N, H=10] future values
    'setup_weight': np.ndarray,         # [N] quality weights
    'cross_streams': np.ndarray,        # [N, n_streams, 5] cross context
    'static_features': np.ndarray,      # [N, n_static] level/time features
    'cross_stream_names': List[str],    # ['sigma_m', 'sigma_f', ...]
    'static_feature_names': List[str]   # ['level_kind', 'direction', ...]
}
```

---

## 3. Train Projection Models

**Purpose**: Train quantile polynomial regression models for 20-min forecasts

**Script**: `backend/scripts/train_projection_models.py`

**Architecture**:
- **Model**: HistGradientBoostingRegressor with MultiOutputRegressor
- **Output**: 3 coefficients {a1, a2, a3} per quantile {q10, q50, q90}
- **Tracking**: Logs to MLFlow experiment `stream_projection` + W&B

**Command**:
```bash
cd backend

# Train single stream
uv run python -m scripts.train_projection_models \
  --stream sigma_p \
  --epochs 200 \
  --learning-rate 0.05 \
  --max-depth 6 \
  --val-ratio 0.2

# Train all streams at once
uv run python -m scripts.train_projection_models \
  --stream all \
  --epochs 200 \
  --learning-rate 0.05 \
  --max-depth 6

# Custom paths and experiment name
uv run python -m scripts.train_projection_models \
  --stream all \
  --data-path data/gold/training/projection_samples \
  --output-dir data/ml/projection_models \
  --experiment pentaview_forecast_v2 \
  --version v2
```

**Training Process**:

```python
# From train_projection_models.py

with tracking_run(
    run_name=f"projection_{stream_name}_{version}",
    experiment="stream_projection",
    params={...},
    tags={'model_type': 'stream_projection', 'stream': stream_name},
    wandb_tags=['stream_projection', stream_name],
    project='spymaster',
    repo_root=repo_root
) as tracking:
    # 1. Load training data
    samples = load_training_data(data_path, stream_name, version)
    
    # 2. Split train/val (80/20)
    train_samples, val_samples = split_train_val(samples, val_ratio=0.2)
    
    # 3. Initialize projector
    projector = StreamProjector(stream_name=stream_name, config=config)
    
    # 4. Train (fit polynomial coefficients to future trajectories)
    train_metrics = projector.fit(
        training_samples=train_samples,
        max_iter=epochs,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    # Logs: q10_r2_a1, q10_r2_a2, q10_r2_a3, q50_r2_*, q90_r2_*
    
    # 5. Evaluate on validation set
    val_metrics = evaluate_projector(projector, val_samples)
    # Computes:
    # - val_path_mae_q50: Mean absolute error over 10-bar path
    # - val_endpoint_mae_q50: Error at horizon H=10
    # - val_path_r2_q50: R² for path prediction
    # - val_coverage_80pct: % targets within [q10, q90] band
    # - val_band_width: Average uncertainty band width
    
    # 6. Log to MLFlow + W&B
    log_metrics(all_metrics, tracking.wandb_run)
    
    # 7. Save model
    projector.save(output_dir / f'projection_{stream_name}_{version}.joblib')
    log_artifacts([model_path], name=f'projection_model_{stream_name}',
                  artifact_type='model', wandb_run=tracking.wandb_run)
```

**Output**:
```
data/ml/projection_models/
├── projection_sigma_p_v1.joblib
├── projection_sigma_m_v1.joblib
├── projection_sigma_f_v1.joblib
├── projection_sigma_b_v1.joblib
└── projection_sigma_r_v1.joblib
```

---

## 4. MLFlow UI for Pentaview

### Starting the UI

```bash
cd backend
mlflow ui
# Open http://localhost:5000
```

### Finding Pentaview Experiments

**Experiment Name**: `stream_projection` (default)

**Runs**: One per stream (sigma_p, sigma_m, sigma_f, sigma_b, sigma_r)

**Run Naming**: `projection_{stream_display_name}_{version}`
- Example: `projection_pressure_v1`, `projection_flow_v1`

### Key Metrics to Monitor

**Training Metrics** (per quantile):
```
q10_r2_a1: R² for slope coefficient (q10)
q10_r2_a2: R² for curvature coefficient (q10)
q10_r2_a3: R² for jerk coefficient (q10)
q10_r2_mean: Average R² across coefficients (q10)

q50_r2_a1, q50_r2_a2, q50_r2_a3, q50_r2_mean  # Median quantile
q90_r2_a1, q90_r2_a2, q90_r2_a3, q90_r2_mean  # Upper quantile
```

**Validation Metrics**:
```
val_path_mae_q50: Path MAE (key metric, target: <0.15)
val_endpoint_mae_q50: 20-min endpoint error (target: <0.20)
val_path_r2_q50: Path R² (target: >0.30)
val_mae_h1_q50: 2-min ahead MAE
val_mae_h5_q50: 10-min ahead MAE
val_mae_h10_q50: 20-min ahead MAE
val_coverage_80pct: Uncertainty calibration (target: ~0.80)
val_band_width: Average [q10, q90] spread
```

**Target Performance**:
- `val_path_r2_q50 > 0.30`: Reasonable forecast skill
- `val_coverage_80pct ≈ 0.80`: Well-calibrated uncertainty
- `val_path_mae_q50 < 0.15`: Accurate path prediction (in [-1,+1] scale)

### Visualization in MLFlow UI

**Navigate to Experiment**:
1. Open `http://localhost:5000`
2. Click `stream_projection` in experiments list
3. See table of runs

**Compare Streams**:
1. Select all 5 runs (sigma_p, sigma_m, sigma_f, sigma_b, sigma_r)
2. Click "Compare" button
3. In comparison view:
   - **Metrics Table**: See all metrics side-by-side
   - **Parallel Coordinates**: Visualize hyperparameter-metric relationships
   - **Scatter Plots**: val_path_r2_q50 vs val_coverage_80pct

**View Training Curves**:
- Training happens in one shot (not iterative like neural nets)
- No per-epoch curves (single-pass boosting)
- Metrics are final values after training

**Download Model**:
1. Click run name (e.g., `projection_pressure_v1`)
2. Scroll to "Artifacts" section
3. Click `projection_model_sigma_p/`
4. Download `.joblib` file

---

## 5. Evaluation & Validation

### Demo Scripts

**Demo Projection Inference**:
```bash
cd backend

# Test projection models on sample data
uv run python -m scripts.demo_projection

# Output example:
# ✓ Loaded projection model: sigma_p
# 
# Current bar (t=0):
#   timestamp: 2024-12-16 10:32:00
#   sigma_p: +0.67
#   spot: 6875.25
# 
# 20-min Projection:
#   q10 curve: [0.67, 0.65, 0.62, ..., 0.45]  # Conservative
#   q50 curve: [0.67, 0.69, 0.71, ..., 0.58]  # Median forecast
#   q90 curve: [0.67, 0.73, 0.79, ..., 0.72]  # Optimistic
# 
# Interpretation:
#   a1 (slope): +0.048   → Positive momentum
#   a2 (curvature): -0.012 → Decelerating
#   a3 (jerk): -0.003   → Rate of deceleration slowing
```

**Demo State Machine**:
```bash
cd backend

# Test alert detection system
uv run python -m scripts.demo_state_machine

# Output example:
# Scenario: Exhaustion Up
#   P = +0.72, P1 = -0.08, P2 = -0.05
# 
# Alerts:
#   ⚠ EXHAUSTION_UP (conf: 0.85)
#      "Buying pressure fading (P=+0.72, slope=-0.08)"
# 
#   ⚠ REVERSAL_RISK_UP (conf: 0.67)
#      "Reversal risk from up (P=+0.72, jerk=-0.05)"
```

### Full Validation Pipeline

**1. Compute Streams for Test Date**:
```bash
cd backend

# Run Pentaview pipeline
uv run python -m scripts.run_pentaview_pipeline --date 2024-12-16

# Validates normalization, aggregation, stream computation
```

**2. Validate Stream Output**:
```bash
cd backend

# Comprehensive validation
uv run python -m scripts.validate_pentaview --date 2024-12-16

# Checks:
# ✓ Stream values in [-1, +1]
# ✓ Derivatives consistent
# ✓ No NaN/Inf values
# ✓ Alignment/divergence computed correctly
# ✓ Output schema matches spec
```

**3. Batch Evaluation** (for multiple dates):
```bash
cd backend

# Process date range
for date in $(seq -f "2024-12-%02g" 1 31); do
    uv run python -m scripts.run_pentaview_pipeline --date $date
    uv run python -m scripts.validate_pentaview --date $date
done
```

---

## 6. Model Performance Diagnostics

### Interpreting Metrics

**Good Performance**:
```yaml
val_path_r2_q50: 0.42        # Strong predictive power
val_endpoint_mae_q50: 0.12   # Low horizon error
val_coverage_80pct: 0.79     # Well-calibrated (near 0.80)
val_band_width: 0.35         # Reasonable uncertainty
q50_r2_mean: 0.65            # Coefficients fit well
```

**Poor Performance**:
```yaml
val_path_r2_q50: 0.08        # Weak predictive power
val_endpoint_mae_q50: 0.28   # High horizon error
val_coverage_80pct: 0.52     # Under-confident (too narrow)
val_band_width: 0.15         # Uncertainty bands too tight
q50_r2_mean: 0.25            # Coefficients don't fit
```

**Diagnostic Steps**:

**If R² is low**:
- Check training data quality (enough samples?)
- Increase lookback window (L > 20)
- Add more features (cross-streams, static context)
- Increase model complexity (max_depth, max_iter)

**If coverage ≠ 0.80**:
- Too high (>0.85): Bands too wide, over-conservative
- Too low (<0.75): Bands too narrow, under-calibrated
- Solution: Adjust quantile loss weighting or post-calibration

**If endpoint MAE is high but path R² is OK**:
- Model captures short-term dynamics but loses accuracy at horizon
- Solution: Use ensemble forecasting or reduce horizon

---

## 7. W&B Dashboard for Pentaview

**Access**: `https://wandb.ai/{entity}/spymaster/runs?tag=stream_projection`

**Visualizations**:
- **Metrics Summary Table**: Compare all streams at once
- **Custom Charts**: Create plots like:
  - `val_path_r2_q50` vs `val_coverage_80pct` (scatter)
  - `val_mae_h{1,5,10}_q50` (line chart showing error growth)
  - R² distribution across coefficients (box plot)

**Hyperparameter Tuning** (future):
```bash
# Sweep configuration
uv run python -m scripts.train_projection_models \
  --stream sigma_p \
  --epochs 200 \
  --learning-rate 0.03,0.05,0.10 \
  --max-depth 4,6,8

# W&B will log all combinations
# Use parallel coordinates plot to find optimal config
```

---

## 8. Typical Workflow

### Initial Training

```bash
cd backend

# 1. Compute normalization stats (once, or when data changes)
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 60 \
  --end-date 2024-12-31

# 2. Run Pentaview pipeline for date range (generates stream bars)
uv run python -m scripts.run_pentaview_pipeline \
  --start 2024-11-01 \
  --end 2024-12-31

# 3. Build projection training dataset
uv run python -m scripts.build_projection_dataset \
  --start 2024-11-01 \
  --end 2024-12-31 \
  --streams sigma_p,sigma_m,sigma_f,sigma_b,sigma_r

# 4. Train projection models (logs to MLFlow)
uv run python -m scripts.train_projection_models \
  --stream all \
  --epochs 200

# 5. Open MLFlow UI
mlflow ui
# Navigate to: http://localhost:5000
# Click: "stream_projection" experiment
# Review: val_path_r2_q50, val_coverage_80pct metrics
# Download best models from Artifacts tab

# 6. Test inference
uv run python -m scripts.demo_projection
uv run python -m scripts.demo_state_machine
```

### Retraining / Iteration

```bash
# Add more training data
uv run python -m scripts.run_pentaview_pipeline \
  --start 2025-01-01 \
  --end 2025-01-31

# Rebuild dataset with extended date range
uv run python -m scripts.build_projection_dataset \
  --start 2024-11-01 \
  --end 2025-01-31 \
  --streams all \
  --version v2

# Train with new data
uv run python -m scripts.train_projection_models \
  --stream all \
  --version v2 \
  --epochs 250 \
  --learning-rate 0.05

# Compare v1 vs v2 in MLFlow UI
# Filter by tag: version=v1 vs version=v2
```

---

## Summary

**Pentaview Training Pipeline**:
1. `compute_stream_normalization.py` → normalization stats
2. `run_pentaview_pipeline.py` → generate stream bars
3. `build_projection_dataset.py` → extract training samples
4. `train_projection_models.py` → train forecasters (logs to MLFlow)
5. `mlflow ui` → visualize metrics, download models
6. `demo_projection.py` + `demo_state_machine.py` → test inference

**MLFlow Experiment**: `stream_projection` (http://localhost:5000)  
**Key Metrics**: `val_path_r2_q50`, `val_coverage_80pct`, `val_endpoint_mae_q50`  
**Model Location**: `data/ml/projection_models/*.joblib`