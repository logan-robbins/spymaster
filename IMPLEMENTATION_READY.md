# IMPLEMENTATION_READY.md
## Level Interaction Similarity Retrieval System — Canonical Specification

**Version**: 3.1  
**Purpose**: Complete specification for level-interaction similarity retrieval  
**Scope**: First 3 hours of RTH (09:30-12:30 ET), ES Futures  
**Architecture**: 144-dimensional episode vectors with DCT trajectory basis  
**Last Updated**: December 29, 2025

---

## System Overview

This specification defines a similarity retrieval system for level-interaction trading in ES futures during the first 3 hours of RTH. The system constructs 144-dimensional episode vectors that capture market state, approach dynamics, and 20-minute trajectory shape using frequency-domain encoding, then retrieves historically similar setups to present empirical outcome distributions.

### Architecture Summary

| Component | Specification |
|-----------|---------------|
| **Episode Vectors** | 144 dimensions (6 sections) |
| **Vector Sections** | Context (25) + Dynamics (37) + Micro-History (35) + Physics (11) + Trends (4) + DCT Basis (32) |
| **Trajectory Encoding** | DCT-II on 4 series × 8 coefficients = 32 dims |
| **Log Transforms** | barrier_delta_liq_log, wall_ratio_log in micro-history |
| **Zone Threshold** | 2.0 ATR for approach detection |
| **Time Buckets** | 5 buckets (T0_15, T15_30, T30_60, T60_120, T120_180) |
| **Total Partitions** | 60 indices (6 levels × 2 directions × 5 time buckets) |
| **Similarity Metric** | Cosine similarity (L2-normalized vectors, inner product) |
| **Index Type** | Auto-select: Flat / IVF / IVFPQ based on corpus size |
| **Outcomes** | BREAK/REJECT/CHOP (first-crossing semantics, 1.0 ATR threshold) |
| **Horizons** | 2min/4min/8min (primary: 4min) |
| **Retrieval** | 500 candidates → dedup (max 2/day, 1/episode) → top 50 neighbors |

### Key Modules

**Core ML**:
- `backend/src/ml/constants.py` - System constants and thresholds
- `backend/src/ml/normalization.py` - Normalization statistics computation
- `backend/src/ml/episode_vector.py` - 144-dimensional vector construction with DCT
- `backend/src/ml/index_builder.py` - FAISS index building (60 partitions)
- `backend/src/ml/retrieval_engine.py` - Real-time query engine
- `backend/src/ml/outcome_aggregation.py` - Outcome distributions
- `backend/src/ml/attribution.py` - Explainability system
- `backend/src/ml/validation.py` - Quality monitoring

**Pipeline Stages**:
- `backend/src/pipeline/stages/label_outcomes.py` - First-crossing semantics
- `backend/src/pipeline/stages/materialize_state_table.py` - Stage 16 (30s state)
- `backend/src/pipeline/stages/construct_episodes.py` - Stage 17 (144D vectors)

**Pipeline Integration**:
- `backend/src/pipeline/pipelines/es_pipeline.py` - 18-stage pipeline

### Validation Scripts

**Main Pipeline Validator**: `backend/scripts/validate_es_pipeline.py`
- Validates full pipeline with 6 QA gates
- Checks for REJECT outcomes, ATR-normalized excursions
- Usage: `uv run python backend/scripts/validate_es_pipeline.py --date 2024-12-20`

**Stage Validators** (0-indexed stage indices, consistent with `es_pipeline.py`):
- **Stage 14** (`validate_stage_14_label_outcomes.py`) - First-crossing labels
- **Stage 16** (`validate_stage_16_materialize_state_table.py`) - 30s state table
- **Stage 17** (`validate_stage_17_construct_episodes.py`) - 144-dim vectors, 5 time buckets

**How to Run**:
```bash
# Stage 14 (Label Outcomes)
uv run python backend/scripts/validate_stage_14_label_outcomes.py --date 2024-12-20

# Stage 16 (State Table)
uv run python backend/scripts/validate_stage_16_materialize_state_table.py --date 2024-12-20

# Stage 17 (Episode Vectors, 144D)
uv run python backend/scripts/validate_stage_17_construct_episodes.py --date 2024-12-20
```

All validators create JSON output in `backend/logs/` and return exit code 0 (pass) or 1 (fail).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Contracts](#2-data-contracts)
3. [Outcome Label Contract](#3-outcome-label-contract)
4. [Level-Relative State Table](#4-level-relative-state-table)
5. [Anchor Point Specification](#5-anchor-point-specification)
6. [Episode Vector Architecture](#6-episode-vector-architecture)
7. [Normalization Specification](#7-normalization-specification)
8. [Index Architecture](#8-index-architecture)
9. [Retrieval Pipeline](#9-retrieval-pipeline)
10. [Outcome Aggregation](#10-outcome-aggregation)
11. [Attribution System](#11-attribution-system)
12. [Validation Framework](#12-validation-framework)
13. [Pipeline Integration](#13-pipeline-integration)
14. [Appendix A: Complete Feature Specification](#appendix-a-complete-feature-specification)
15. [Appendix B: Schema Reference](#appendix-b-schema-reference)
16. [Appendix C: Constants and Thresholds](#appendix-c-constants-and-thresholds)

---

## 1. System Overview

### 1.1 Core Function

This system retrieves historically similar market setups when price approaches key technical levels, then presents the empirical outcome distribution of those historical analogs. It does **not** predict price. It answers:

> "Given the current market state as price approaches [LEVEL], what happened historically in the K most similar situations?"

### 1.2 Monitored Levels

| Level Kind | Code | Description | Establishment Time |
|------------|------|-------------|-------------------|
| Pre-Market High | `PM_HIGH` | Highest price during pre-market session | 09:30:00 ET |
| Pre-Market Low | `PM_LOW` | Lowest price during pre-market session | 09:30:00 ET |
| Opening Range High | `OR_HIGH` | Highest price in first 15 minutes | 09:45:00 ET |
| Opening Range Low | `OR_LOW` | Lowest price in first 15 minutes | 09:45:00 ET |
| 200-period SMA | `SMA_200` | 200-bar SMA on 2-minute bars | 09:30:00 ET (dynamic) |
| 400-period SMA | `SMA_400` | 400-bar SMA on 2-minute bars | 09:30:00 ET (dynamic) |

### 1.3 System Outputs

For each query, the system returns:

```
QueryResult:
  - outcome_probabilities: {BREAK: float, REJECT: float, CHOP: float} per horizon
  - confidence_intervals: {outcome: {mean, ci_low, ci_high}} per horizon
  - expected_excursions: {favorable: float, adverse: float} in ATR units
  - conditional_excursions: {given BREAK: {...}, given REJECT: {...}}
  - attribution: {similarity_drivers: [...], outcome_drivers: [...]}
  - reliability: {n_retrieved, effective_n, avg_similarity, min_similarity}
  - neighbors: [{event_id, date, similarity, outcome, excursions, ...}]
```

### 1.4 Coordinate System Convention

**All distance-based features use the following sign convention:**

```
distance_signed = (spot - level_price) / atr

Positive: spot is ABOVE level
Negative: spot is BELOW level
```

**Direction convention:**

```
direction = 'UP'   → approaching level from BELOW (spot < level, expecting to cross up)
direction = 'DOWN' → approaching level from ABOVE (spot > level, expecting to cross down)
```

---

## 2. Data Contracts

### 2.1 Input: Silver Layer Event Table

**Location**: `silver/features/es_pipeline/date=YYYY-MM-DD/*.parquet`

This table contains event records triggered at level interactions. Each row represents one interaction event.

**Required columns** (from existing schema):

```
Identifiers:
  - event_id: string (unique)
  - date: date
  - timestamp: datetime64[ns]
  - ts_ns: int64

Level Context:
  - level_kind: string {PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_200, SMA_400}
  - level_price: float64
  - direction: string {UP, DOWN}
  - zone_width: float64

Price State:
  - spot: float64
  - atr: float64
  - distance_signed_atr: float64

Outcome Fields (used for labeling only, never in features):
  - outcome_2min, outcome_4min, outcome_8min: string
  - time_to_break_1_2min, time_to_break_1_4min, time_to_break_1_8min: float64 (seconds, nullable)
  - time_to_bounce_1_2min, time_to_bounce_1_4min, time_to_bounce_1_8min: float64 (seconds, nullable)
  - excursion_max, excursion_min: float64
  - strength_signed, strength_abs: float64

[All other feature columns per Appendix A]
```

### 2.2 Output: Gold Layer Episode Corpus

**Location**: `gold/episodes/es_level_episodes/`

```
gold/episodes/es_level_episodes/
├── vectors/
│   └── date=YYYY-MM-DD/
│       └── episodes.npy          # [N_episodes × 111] float32
├── metadata/
│   └── date=YYYY-MM-DD/
│       └── metadata.parquet      # Episode metadata
└── corpus/
    ├── all_vectors.npy           # Memory-mapped, all dates
    └── all_metadata.parquet      # All metadata concatenated
```

**Metadata schema**:

```
event_id: string
date: date
timestamp: datetime64[ns]
ts_ns: int64
level_kind: string
level_price: float64
direction: string
spot: float64
atr: float64
minutes_since_open: float64
time_bucket: string {T0_30, T30_60, T60_120, T120_180}

Labels (per horizon):
  outcome_2min: string {BREAK, REJECT, CHOP}
  outcome_4min: string {BREAK, REJECT, CHOP}
  outcome_8min: string {BREAK, REJECT, CHOP}
  
Continuous Outcomes:
  excursion_favorable: float64  # ATR-normalized
  excursion_adverse: float64    # ATR-normalized
  strength_signed: float64
  strength_abs: float64
  time_to_resolution: float64   # seconds

Quality:
  emission_weight: float64      # [0, 1]

Attempt Context:
  prior_touches: int
  attempt_index: int
```

### 2.3 Output: FAISS Index Structure

**Location**: `gold/indices/es_level_indices/`

```
gold/indices/es_level_indices/
├── PM_HIGH/
│   ├── UP/
│   │   ├── T0_30/
│   │   │   ├── index.faiss
│   │   │   ├── vectors.npy
│   │   │   └── metadata.parquet
│   │   ├── T30_60/
│   │   ├── T60_120/
│   │   └── T120_180/
│   └── DOWN/
│       └── [same structure]
├── PM_LOW/
│   └── [same structure]
├── OR_HIGH/
│   └── [same structure]
├── OR_LOW/
│   └── [same structure]
├── SMA_200/
│   └── [same structure]
├── SMA_400/
│   └── [same structure]
└── config.json
```

---

## 3. Outcome Label Contract

**Implementation**: `backend/src/pipeline/stages/label_outcomes.py`

The outcome label is determined using first-crossing semantics with a fixed 1.0 ATR threshold. Labels are computed independently for three time horizons (2min, 4min, 8min).

### 3.1 Label Function

**Implementation**: `backend/src/pipeline/stages/label_outcomes.py`

The outcome label is determined by **first-crossing semantics** using existing schema fields `time_to_break_1_{H}` and `time_to_bounce_1_{H}`.

**Logic**:
- Treat null timestamps as "never hit"
- If neither threshold crossed within horizon → CHOP
- If break crossed first → BREAK
- If bounce crossed first → REJECT
- Tie (same timestamp) → CHOP

**Threshold**: 1.0 ATR in both directions (fixed)

### 3.2 Label Semantics by Direction

| Direction | BREAK means | REJECT means |
|-----------|-------------|--------------|
| UP | Price crossed above level and held | Price failed to cross or reversed down |
| DOWN | Price crossed below level and held | Price failed to cross or reversed up |

### 3.3 Multi-Horizon Labels

Labels are computed independently for three time horizons: 2min (120s), 4min (240s), 8min (480s).

Primary horizon for retrieval: **4min**

### 3.4 Continuous Outcome Variables

In addition to discrete labels, the system stores continuous outcome measures in episode metadata:

- `excursion_favorable`: Movement in break direction (ATR-normalized)
- `excursion_adverse`: Movement in reject direction (ATR-normalized)
- `time_to_resolution`: Time to first threshold hit (seconds)
- `strength_signed`, `strength_abs`: Outcome strength metrics

**Direction mapping**:
- UP: favorable = excursion_max, adverse = |excursion_min|
- DOWN: favorable = |excursion_min|, adverse = excursion_max

### 3.5 Online Safety

**CRITICAL**: Outcome fields are used ONLY for labeling historical episodes. They must NEVER appear in the feature vector.

**Excluded from features**:
- `outcome_*`
- `time_to_break_*`, `time_to_bounce_*`
- `excursion_*`
- `strength_*`
- `tradeable_*`
- `future_price_*`
- Any field computed using data after the anchor timestamp

---

## 4. Level-Relative State Table

**Implementation**: `backend/src/pipeline/stages/materialize_state_table.py` (Stage 16)

The state table provides time-sampled market state at fixed 30-second cadence in the coordinate frame of each level. This enables window extraction for episode construction and visualization of approach dynamics.

### 4.1 Purpose

The state table provides time-sampled market state at fixed cadence, enabling:
1. Window extraction for any anchor point
2. Consistent feature computation across time
3. Visualization of approach dynamics

### 4.2 Specification

**Location**: `silver/state/es_level_state/date=YYYY-MM-DD/*.parquet`

**Cadence**: 30 seconds

**Time Range**: 09:30:00 ET to 12:30:00 ET (360 samples per level per day)

**Partitioning**: By date

**Row Definition**: One row per (timestamp, level_kind) pair

### 4.3 Schema

```
Identifiers:
  timestamp: datetime64[ns]
  ts_ns: int64
  date: date
  minutes_since_open: float64
  bars_since_open: int  # 2-minute bars

Level Context:
  level_kind: string
  level_price: float64
  level_active: bool  # False before level is established (e.g., OR before 09:45)

Price State:
  spot: float64
  atr: float64
  distance_signed_atr: float64  # (spot - level_price) / atr

Distances to All Levels (ATR-normalized):
  dist_to_pm_high_atr: float64
  dist_to_pm_low_atr: float64
  dist_to_or_high_atr: float64 | null  # null before 09:45
  dist_to_or_low_atr: float64 | null
  dist_to_sma_200_atr: float64
  dist_to_sma_400_atr: float64

Level Stacking:
  level_stacking_2pt: int
  level_stacking_5pt: int
  level_stacking_10pt: int

Kinematics (at this timestamp):
  velocity_1min: float64
  velocity_3min: float64
  velocity_5min: float64
  velocity_10min: float64
  velocity_20min: float64
  acceleration_1min: float64
  acceleration_3min: float64
  acceleration_5min: float64
  acceleration_10min: float64
  acceleration_20min: float64
  jerk_1min: float64
  jerk_3min: float64
  jerk_5min: float64
  jerk_10min: float64
  jerk_20min: float64
  momentum_trend_3min: float64
  momentum_trend_5min: float64
  momentum_trend_10min: float64
  momentum_trend_20min: float64

Approach Dynamics:
  approach_velocity: float64
  approach_bars: int
  approach_distance_atr: float64

Order Flow:
  ofi_30s: float64
  ofi_60s: float64
  ofi_120s: float64
  ofi_300s: float64
  ofi_near_level_30s: float64
  ofi_near_level_60s: float64
  ofi_near_level_120s: float64
  ofi_near_level_300s: float64
  ofi_acceleration: float64

Tape:
  tape_imbalance: float64
  tape_velocity: float64
  tape_buy_vol: float64
  tape_sell_vol: float64
  sweep_detected: bool

Barrier/Liquidity:
  barrier_state: string
  barrier_depth_current: float64
  barrier_delta_liq: float64
  barrier_replenishment_ratio: float64
  wall_ratio: float64
  barrier_delta_1min: float64
  barrier_delta_3min: float64
  barrier_delta_5min: float64
  barrier_pct_change_1min: float64
  barrier_pct_change_3min: float64
  barrier_pct_change_5min: float64

Dealer/Options:
  gamma_exposure: float64
  fuel_effect: string {AMPLIFY, NEUTRAL, DAMPEN}
  gex_asymmetry: float64
  gex_ratio: float64
  net_gex_2strike: float64
  gex_above_1strike: float64
  gex_below_1strike: float64
  call_gex_above_2strike: float64
  put_gex_below_2strike: float64

Physics Proxies:
  predicted_accel: float64
  accel_residual: float64
  force_mass_ratio: float64

Touch/Attempt (for this level_kind):
  prior_touches: int
  attempt_index: int
  time_since_last_touch: float64 | null

Cluster Trends:
  barrier_replenishment_trend: float64
  barrier_delta_liq_trend: float64
  tape_velocity_trend: float64
  tape_imbalance_trend: float64
```

### 4.4 Computation Notes

1. **SMA levels**: `level_price` for SMA_200/SMA_400 is computed at each timestamp (they are moving targets)

2. **OR levels before 09:45**: Set `level_active = false`, `level_price = null`, distances to OR = null

3. **Features are online-safe**: Every feature at timestamp T uses only data from T and before

4. **Barrier features**: Computed relative to `level_price` even when spot is far from level

---

## 5. Anchor Point Specification

### 5.1 Touch Anchor (Primary)

A touch anchor is created when price first enters the interaction zone for a level (within 2.0 ATR).

**Source**: Existing event table rows (each `event_id` is a touch anchor)

**Anchor timestamp**: `confirm_ts_ns` from event table (or `ts_ns` if confirm not available)

**Properties**:
- High signal-to-noise (price is at/near level)
- Clear decision point
- Well-defined outcome measurement window

### 5.2 Time Bucket Assignment

**Implementation**: `backend/src/ml/episode_vector.py` (assign_time_bucket)

Assign each anchor to a time bucket based on `minutes_since_open`. Five buckets provide finer temporal resolution, especially in the first 30 minutes when OR is being established:

| Minutes | Bucket | Description |
|---------|--------|-------------|
| 0-15 | T0_15 | OR formation period |
| 15-30 | T15_30 | Post-OR early |
| 30-60 | T30_60 | Mid-session |
| 60-120 | T60_120 | Late-session |
| 120-180 | T120_180 | Final hour |

### 5.3 Emission Weight

**Implementation**: `backend/src/ml/episode_vector.py` (compute_emission_weight)

Each anchor receives a quality weight in [0, 1] based on:

**Proximity weight**: Exponential decay from level, `exp(-distance_atr / 1.5)`
- 1.0 at level, 0.5 at 1 ATR, 0.1 at 3.5 ATR

**Velocity weight**: Faster approach = more decisive, clipped to [0.2, 1.0]
- `clip(|approach_velocity| / 2.0, 0.2, 1.0)`

**OFI alignment**: Flow direction matches approach direction
- 1.0 if aligned or neutral, 0.6 if opposing

**Combined**: `w_emission = proximity_w × velocity_w × ofi_w`

---

## 6. Episode Vector Architecture

**Implementation**: `backend/src/ml/episode_vector.py`, `backend/src/pipeline/stages/construct_episodes.py`  

The episode vector is a **144-dimensional float32 array** optimized for similarity search:

### 6.1 Design Principles

1. **Level-relative**: All price-based features expressed relative to tested level
2. **ATR-normalized**: Scale-invariant across price regimes
3. **Trajectory encoding**: Full 20-minute approach shape captured via DCT coefficients
4. **Hybrid architecture**: Combines slow context, multi-scale dynamics, fast micro-history, and frequency-domain trajectory
5. **Log-transformed heavy-tailed features**: Barrier and wall features use log transforms for better normalization
6. **Optimal dimensionality**: 144 dimensions (within 100-200 optimal range for ANN search)

### 6.2 Construction Process

For each anchor:
- Extract 5-bar micro-history (last 2.5 minutes @ 30s cadence)
- Extract 40-bar trajectory window (last 20 minutes @ 30s cadence)
- Compute DCT coefficients for 4 key time series
- Assemble 144-dimensional raw vector
- Apply feature-specific normalization
- Compute labels and emission weight

### 6.3 Vector Sections

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         EPISODE VECTOR (144 DIMENSIONS)                          │
│                                                                                  │
│  ┌──────────────────┐ ┌──────────────────────┐ ┌──────────────────────────────┐ │
│  │  SECTION A:      │ │  SECTION B:          │ │  SECTION C:                  │ │
│  │  CONTEXT +       │ │  MULTI-SCALE         │ │  MICRO-HISTORY               │ │
│  │  REGIME          │ │  DYNAMICS            │ │  (T-4 to T=0, 5 bars)        │ │
│  │  (T=0 snapshot)  │ │  (T=0, encodes       │ │                              │ │
│  │                  │ │   temporal dynamics) │ │  35 dims                     │ │
│  │  25 dims         │ │                      │ │  (7 features × 5 bars)       │ │
│  │                  │ │  37 dims             │ │                              │ │
│  │  • Session pos   │ │                      │ │  • d_atr                     │ │
│  │  • OR active     │ │  • Velocity scales   │ │  • tape_imbalance            │ │
│  │  • GEX struct    │ │  • Accel scales      │ │  • tape_velocity             │ │
│  │  • Stacking      │ │  • Jerk scales       │ │  • ofi_60s                   │ │
│  │  • Ref distances │ │  • Momentum trends   │ │  • barrier_delta_liq_log (†) │ │
│  │  • Touch memory  │ │  • OFI scales        │ │  • wall_ratio_log (†)        │ │
│  │                  │ │  • Barrier evolution │ │  • gamma_exposure            │ │
│  │                  │ │  • Approach dynamics │ │                              │ │
│  └──────────────────┘ └──────────────────────┘ └──────────────────────────────┘ │
│                                                                                  │
│  ┌──────────────────┐ ┌──────────────────────┐ ┌──────────────────────────────┐ │
│  │  SECTION D:      │ │  SECTION E:          │ │  SECTION F:                  │ │
│  │  DERIVED PHYSICS │ │  ONLINE TRENDS       │ │  TRAJECTORY BASIS            │ │
│  │                  │ │                      │ │                              │ │
│  │  11 dims         │ │  4 dims              │ │  32 dims                     │ │
│  │                  │ │                      │ │  (4 series × 8 DCT coeffs)   │ │
│  │  • Force model   │ │  • Replenish trend   │ │                              │ │
│  │  • Mass proxy    │ │  • Delta liq trend   │ │  • DCT(d_atr)                │ │
│  │  • Force proxy   │ │  • Tape trends       │ │  • DCT(ofi_60s)              │ │
│  │  • Flow align    │ │                      │ │  • DCT(barrier_log)          │ │
│  │  • Barrier state │ │                      │ │  • DCT(tape_imbal)           │ │
│  └──────────────────┘ └──────────────────────┘ └──────────────────────────────┘ │
│                                                                                  │
│  (†) LOG-TRANSFORMED: Heavy-tailed features use log transforms for stability    │
│  TOTAL: 25 + 37 + 35 + 11 + 4 + 32 = 144 dimensions                             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Section A: Context + Regime (25 dimensions)

Single snapshot at T=0. These features define the environment and change slowly.

**Note**: `level_kind` and `direction` are **not** encoded in the vector since they are partition keys (each index is specific to a level_kind × direction combination).

```
Index   Feature                      Encoding / Notes
─────   ───────                      ────────────────
0       minutes_since_open           MinMax normalized [0, 180]
1       bars_since_open              MinMax normalized [0, 90]
2       atr                          Z-score normalized
3       or_active                    0.0 if minutes_since_open < 15, else 1.0
4       level_stacking_2pt           MinMax normalized [0, 6]
5       level_stacking_5pt           MinMax normalized [0, 6]
6       level_stacking_10pt          MinMax normalized [0, 6]
7       dist_to_pm_high_atr          Z-score normalized
8       dist_to_pm_low_atr           Z-score normalized
9       dist_to_or_high_atr          Z-score normalized (0 if OR not active)
10      dist_to_or_low_atr           Z-score normalized (0 if OR not active)
11      dist_to_sma_200_atr          Z-score normalized
12      dist_to_sma_400_atr          Z-score normalized
13      prior_touches                MinMax normalized [0, 10]
14      attempt_index                MinMax normalized [0, 10]
15      time_since_last_touch_sec    MinMax normalized (seconds since last touch)
16      gamma_exposure               Robust normalized
17      fuel_effect                  Encoded: AMPLIFY=1, NEUTRAL=0, DAMPEN=-1
18      gex_ratio                    Robust normalized
19      gex_asymmetry                Robust normalized
20      net_gex_2strike              Robust normalized
21      gex_above_1strike            Robust normalized
22      gex_below_1strike            Robust normalized
23      call_gex_above_2strike       Robust normalized
24      put_gex_below_2strike        Robust normalized
```

### 6.5 Section B: Multi-Scale Dynamics (37 dimensions)

Single snapshot at T=0. These features already encode temporal dynamics via multi-scale computation.

```
Index   Feature                      
─────   ───────                      
26-30   velocity_{1min,3min,5min,10min,20min}         (5)
31-35   acceleration_{1min,3min,5min,10min,20min}     (5)
36-40   jerk_{1min,3min,5min,10min,20min}             (5)
41-44   momentum_trend_{3min,5min,10min,20min}        (4)
45-48   ofi_{30s,60s,120s,300s}                       (4)
49-52   ofi_near_level_{30s,60s,120s,300s}            (4)
53      ofi_acceleration                              (1)
54-56   barrier_delta_{1min,3min,5min}                (3)
57-59   barrier_pct_change_{1min,3min,5min}           (3)
60      approach_velocity                             (1)
61      approach_bars                                 (1)
62      approach_distance_atr                         (1)
```

### 6.6 Section C: Micro-History (35 dimensions)

5-bar history (T-4 to T=0) for 7 fast-changing features. These capture the immediate approach dynamics.

**Log Transforms**: Barrier and wall features use log transforms to handle heavy-tailed distributions.

```
Index   Feature × Time                   Transform
─────   ──────────────                   ─────────
62-66   d_atr[t-4..t0]                  None (signed distance)
67-71   tape_imbalance[t-4..t0]         None
72-76   tape_velocity[t-4..t0]          None
77-81   ofi_60s[t-4..t0]                None
82-86   barrier_delta_liq_log[t-4..t0]  Signed log: log1p(|x|) * sign(x)
87-91   wall_ratio_log[t-4..t0]         Log: log(max(x, 1e-6))
92-96   gamma_exposure[t-4..t0]         None
```

**Bar cadence**: 30 seconds (state table cadence), so 5 bars = 2.5 minutes of micro-history.

### 6.7 Section D: Derived Physics (11 dimensions)

Physics-aligned state descriptors providing force/mass framing and flow alignment.

```
Index   Feature                      Computation
─────   ───────                      ───────────
97      predicted_accel              From F=ma model
98      accel_residual               Actual - predicted
99      force_mass_ratio             From F=ma model
100     mass_proxy                   log1p(barrier_depth_current)
101     force_proxy                  ofi_60s / (mass_proxy + ε)
102     barrier_state_encoded        STRONG_SUPPORT=2, WEAK_SUPPORT=1, NEUTRAL=0, WEAK_RESISTANCE=-1, STRONG_RESISTANCE=-2
103     barrier_replenishment_ratio  Barrier replenishment metric
104     sweep_detected               0 or 1
105     tape_log_ratio               log((tape_buy_vol+1) / (tape_sell_vol+1))
106     tape_log_total               log(tape_buy_vol + tape_sell_vol + 1)
107     flow_alignment               ofi_60s * (-sign(d_atr)); positive = flow aligned with approach
```

### 6.8 Section E: Online Trends (4 dimensions)

Rolling trends computed incrementally (no lookahead).

```
Index   Feature
─────   ───────
108     barrier_replenishment_trend
109     barrier_delta_liq_trend
110     tape_velocity_trend
111     tape_imbalance_trend
```

### 6.9 Section F: Trajectory Basis (32 dimensions)

DCT-II coefficients encoding the full 20-minute approach shape in frequency domain. Provides compact representation of trajectory dynamics.

**Window**: 40 samples @ 30s cadence = 20 minutes  
**Method**: DCT-II (Discrete Cosine Transform, Type 2)  
**Coefficients**: First 8 per series (c0..c7)

```
Index    Series                        Description
──────   ─────                        ───────────
112-119  DCT(d_atr)                   Distance trajectory (approach path geometry)
120-127  DCT(ofi_60s)                 Order flow trajectory (buying/selling pressure)
128-135  DCT(barrier_delta_liq_log)   Liquidity trajectory (barrier depth changes)
136-143  DCT(tape_imbalance)          Aggression trajectory (tape buy/sell imbalance)
```

**Purpose**: Explicitly encodes "approach shape over time" - gradual vs sudden approaches, oscillating vs monotonic paths, etc. This captures patterns that summary statistics (velocities/accelerations) may miss.

### 6.10 Vector Section Indices

**Implementation**: See `backend/src/ml/constants.py` for `VECTOR_SECTIONS` and `VECTOR_DIMENSION`

| Section | Start Index | End Index | Dimensions |
|---------|-------------|-----------|------------|
| context_regime | 0 | 25 | 25 |
| multiscale_dynamics | 25 | 62 | 37 |
| micro_history | 62 | 97 | 35 |
| derived_physics | 97 | 108 | 11 |
| online_trends | 108 | 112 | 4 |
| trajectory_basis | 112 | 144 | 32 |
| **TOTAL** | | | **144** |

### 6.11 Vector Construction Procedure


### 6.11 Vector Construction Algorithm

**Implementation**: `backend/src/ml/episode_vector.py` (construct_episode_vector)

**Inputs**:
- current_bar: Feature dictionary at T=0
- history_buffer: Last 5 bars for micro-history (2.5 min)
- trajectory_window: Last 40 bars for DCT (20 min)
- level_price: Level being tested

**Assembly Order**:
1. Section A: 25 context features (time, stacking, distances, GEX, touch memory)
2. Section B: 37 multi-scale features (kinematics, OFI, barrier evolution, approach)
3. Section C: 35 micro-history features (7 features × 5 bars with log transforms)
4. Section D: 11 physics features (force model, mass/force proxies, flow alignment)
5. Section E: 4 trend features (barrier, tape)
6. Section F: 32 DCT coefficients (4 time series × 8 coefficients each)

**Key Transforms**:
- Categorical encodings: fuel_effect (-1/0/+1), barrier_state (-2 to +2), or_active (0/1)
- Log transforms: barrier_delta_liq_log, wall_ratio_log (signed and unsigned)
- DCT: Type-II orthonormal on 40-sample windows (scipy.fft.dct)
- Derived features: mass_proxy = log1p(barrier_depth), force_proxy = ofi/mass, flow_alignment = ofi × (-sign(d_atr))

---

## 7. Normalization Specification

**Implementation**: `backend/src/ml/normalization.py`

Normalization ensures features are on comparable scales for similarity search. Statistics are computed from 60 days of historical state table data and applied to all 144 vector dimensions.

### 7.1 Normalization Strategy

Different features require different normalization methods based on their distributions.

| Category | Method | Parameters | Features |
|----------|--------|------------|----------|
| **Robust** | (x - median) / IQR | clip ±4σ | Tape, OFI, barrier deltas, wall_ratio, force_mass_ratio, accel_residual |
| **Z-Score** | (x - mean) / std | clip ±4σ | Velocity, acceleration, jerk, momentum_trend, distance_signed_atr, predicted_accel |
| **MinMax** | (x - min) / (max - min) | [0, 1] | minutes_since_open, bars_since_open, level_stacking_*, prior_touches, attempt_index, approach_bars |
| **Passthrough** | No transformation | — | Encoded categoricals (level_kind, direction, fuel_effect, barrier_state, sweep_detected) |

### 7.2 Normalization Statistics Computation

**Implementation**: `backend/src/ml/normalization.py` (compute_normalization_stats)

**Process**:
1. Load 60 days of state table data
2. For each of 144 features:
   - Determine normalization method via `classify_feature_method()`
   - Compute statistics (median/IQR for robust, mean/std for zscore, min/max for minmax)
   - Store in statistics dictionary
3. Save to `gold/normalization/stats_v{version}.json`

**Pattern Matching**:
- Features ending in `_t0` through `_t4`: Micro-history, classify by base feature
- Features starting with `dct_`: DCT coefficients, use zscore
- Features containing `_log`: Log-transformed, use robust
- Fallback: Check feature name against predefined sets

### 7.3 Normalization Application

**Implementation**: `backend/src/ml/normalization.py` (normalize_value, normalize_vector)

For each feature value:
- **Passthrough**: Return as-is (categoricals, binary flags)
- **Robust**: `(x - median) / IQR`, clip to [-4, +4]
- **Z-Score**: `(x - mean) / std`, clip to [-4, +4]  
- **MinMax**: `(x - min) / (max - min)`, clip to [0, 1]

### 7.4 Statistics Storage Format

**Location**: `gold/normalization/stats_v{version}.json`

Contains version metadata, computation date, lookback period, sample count, and per-feature statistics dictionaries with method type and parameters (center/scale for robust/zscore, min/max for minmax).

---

## 8. Index Architecture

**Implementation**: `backend/src/ml/index_builder.py`

The retrieval system uses partitioned FAISS indices to ensure regime-comparable neighbors. Each partition contains episodes from a specific level type, approach direction, and session phase.

### 8.1 Partitioning Strategy

Indices are partitioned by three dimensions:

1. **level_kind**: {PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_200, SMA_400} (6 values)
2. **direction**: {UP, DOWN} (2 values)
3. **time_bucket**: {T0_15, T15_30, T30_60, T60_120, T120_180} (5 values)

**Total partitions**: 6 × 2 × 5 = **60 indices**

**Rationale for 5 time buckets**: The first 30 minutes is split into two buckets (T0_15 and T15_30) to separate the OR formation period (0-15 min) from immediate post-OR behavior (15-30 min). This provides finer temporal resolution when market dynamics change rapidly.

### 8.2 Index Type Selection

| Corpus Size (per partition) | Index Type | Parameters |
|----------------------------|------------|------------|
| < 10,000 episodes | IndexFlatIP | Exact search |
| 10,000 - 100,000 | IndexIVFFlat | nlist = N/100, nprobe = 64 |
| > 100,000 | IndexIVFPQ | nlist = 4096, m = 8, nprobe = 64 |

### 8.3 Index Construction

**Implementation**: `backend/src/ml/index_builder.py` (build_index)

**Process**:
1. L2-normalize vectors for cosine similarity via inner product
2. Select index type based on corpus size (Flat/IVF/IVFPQ)
3. Create FAISS index with appropriate parameters
4. Train index (if needed for IVF/IVFPQ)
5. Add vectors to index
6. Set search parameters (nprobe for IVF variants)

**Index Types**:
- **Flat**: Exact search, no training, best for <10K
- **IVF**: Inverted file with flat quantizer, `nlist = N/100`, `nprobe = 64`
- **IVFPQ**: Inverted file with product quantization, `nlist = 4096`, `m = 12` (divides 144), `nprobe = 64`

**Similarity Metric**: Inner product on L2-normalized vectors (equivalent to cosine similarity)

### 8.4 Index Building Process

**Implementation**: `backend/src/ml/index_builder.py` (BuildIndicesStage)

For each of 60 partitions (level_kind × direction × time_bucket):
1. Filter corpus to partition using metadata masks
2. Skip if partition has fewer than `MIN_PARTITION_SIZE` (100) vectors
3. Select index type based on partition corpus size
4. Build FAISS index with L2-normalized vectors
5. Save three files per partition:
   - `index.faiss` - FAISS index structure
   - `vectors.npy` - Raw vectors for attribution
   - `metadata.parquet` - Episode metadata with labels

**Output**: 60 partition directories under `gold/indices/es_level_indices/{level}/{dir}/{bucket}/` plus `config.json` with build metadata.

### 8.5 Index Manager

**Implementation**: `backend/src/ml/retrieval_engine.py` (IndexManager class)

Manages lazy loading and caching of FAISS indices across 60 partitions.

**Key Methods**:
- `load_partition(level_kind, direction, time_bucket)`: Loads FAISS index, vectors, and metadata for a partition
- `query(level_kind, direction, time_bucket, query_vector, k)`: Executes similarity search

**Query Process**:
1. Lazy-load partition if not cached
2. L2-normalize query vector
3. Search FAISS index for k nearest neighbors
4. Filter invalid indices (artifacts from approximate search)
5. Retrieve metadata and optionally vectors for attribution
6. Return similarities, indices, metadata, and vectors

---

## 9. Retrieval Pipeline

**Implementation**: `backend/src/ml/retrieval_engine.py`

The retrieval pipeline transforms real-time market state into query vectors, searches FAISS indices for similar historical episodes, and returns outcome distributions with attribution.

**Components**:
- **IndexManager**: Lazy-loads and caches FAISS indices per partition
- **LiveEpisodeBuilder**: Builds query vectors from real-time state with 5-bar and 40-bar buffers
- **SimilarityQueryEngine**: Executes queries and returns results
- **RealTimeQueryService**: Main service with caching (30s TTL) and quality filtering

### 9.1 Live Episode Builder

**Implementation**: `backend/src/ml/retrieval_engine.py` (LiveEpisodeBuilder class)

Maintains 5-bar and 40-bar buffers per level and emits query vectors when approach conditions are met.

**Initialization**:
- Loads normalization statistics
- Initializes empty buffers for each (level_kind, level_price) pair
- Sets buffer sizes: 5 bars (micro-history), 40 bars (trajectory)

**on_state_update(state_row)**: Called every 30 seconds with state table row
        """
**Process**:
1. Append state_row to appropriate buffer (5-bar and 40-bar)
2. Check emission conditions:
   - `|distance_atr| < Z_APPROACH_ATR` (2.0)
   - `approach_bars >= MIN_APPROACH_BARS` (2)
   - `approach_velocity >= MIN_APPROACH_V_ATR_PER_MIN` (0.10)
3. If conditions met and buffers full:
   - Construct 144D raw vector
   - Normalize using precomputed statistics
   - Determine direction (UP if spot < level, else DOWN)
   - Assign time bucket based on minutes_since_open
   - Compute emission_weight
   - Return EpisodeQuery ready for retrieval

**Emission Weight**: `w = proximity_w × velocity_w × ofi_alignment_w` (Section 5.3)

### 9.2 Query Engine

**Implementation**: `backend/src/ml/retrieval_engine.py` (SimilarityQueryEngine class)

**Configuration**:
- `M_CANDIDATES = 500`: Over-fetch from FAISS
- `K_NEIGHBORS = 50`: Final neighbors after deduplication
- `MAX_PER_DAY = 2`: Maximum neighbors from same date
- `MAX_PER_EPISODE = 1`: Maximum neighbors from same episode

**query(episode_query, filters) → QueryResult**:

**Process**:
1. Retrieve M_CANDIDATES (500) from appropriate partition via IndexManager
2. Apply deduplication constraints (max 2/day, 1/episode)
3. Apply optional filters if provided
4. Take top K_NEIGHBORS (50) by similarity
5. Compute outcome distributions (Section 10)
6. Compute attribution (Section 11)
7. Compute reliability metrics
8. Return QueryResult with probabilities, excursions, attribution, and neighbors

### 9.3 Real-Time Service

**Implementation**: `backend/src/ml/retrieval_engine.py` (RealTimeQueryService class)

Main service coordinating live episode building, querying, and caching.

**Components**:
- LiveEpisodeBuilder: Maintains state buffers, emits queries
- SimilarityQueryEngine: Executes retrieval
- Result cache: 30-second TTL per level

**process_state_update(state_row) → list[QueryResult]**:
1. Pass state_row to LiveEpisodeBuilder
2. Receive list of EpisodeQuery objects (if emission conditions met)
3. For each query:
   - Check cache (skip if fresh result exists)
   - Execute query via SimilarityQueryEngine
   - Apply quality filters (min similarity 0.70, min samples 30)
   - Cache result if quality threshold met
4. Return list of QueryResult objects

**Quality Filtering**: Only return results where `avg_similarity >= 0.70` and `n_retrieved >= 30`

---

## 10. Outcome Aggregation

**Implementation**: `backend/src/ml/outcome_aggregation.py`

Outcome aggregation converts retrieved neighbors into probabilistic outcome distributions and reliability metrics using neighbor weighting and Bayesian smoothing.

**Neighbor Weighting**:
- **Power Transform**: `similarity^4` (emphasizes high-quality matches)
- **Recency Decay**: `exp(-age_days / 60)` (60-day halflife)
- **Combined**: `weight = (similarity^4) × exp(-age_days / 60)`, normalized to sum to 1

**Functions**:
- `compute_neighbor_weights()`: Power transform + recency decay
- `compute_outcome_distribution()`: Weighted probabilities with Dirichlet posterior
- `compute_expected_excursions()`: Expected favorable/adverse excursions
- `compute_conditional_excursions()`: Excursions conditional on outcome
- `compute_multi_horizon_distribution()`: Outcomes for 2min/4min/8min
- `compute_bootstrap_ci()`: Bootstrap confidence intervals (1000 samples)
- `compute_reliability()`: Effective N, entropy, similarity stats

### 10.1 Weighted Distribution with Dirichlet Posterior

**Implementation**: `backend/src/ml/outcome_aggregation.py` (compute_outcome_distribution)

**Algorithm**:
1. Apply power transform: `weight = similarity^4`
2. Apply recency decay: `weight *= exp(-age_days / 60)`
3. Normalize weights to sum to 1
4. Apply Dirichlet prior for Bayesian smoothing:
   - Priors: `{BREAK: 1.0, REJECT: 1.0, CHOP: 0.5}`
   - Posterior: `P(outcome) = (weighted_count + prior) / (total_weight + sum_priors)`
5. Return probability distribution with sample count and average similarity

**Dirichlet Smoothing**: Prevents extreme probabilities (0% or 100%) when sample size is small. The symmetric priors for BREAK/REJECT reflect no directional bias, while the lower CHOP prior reflects its rarity as a decisive outcome.

**Result**: `{probabilities: {BREAK: p1, REJECT: p2, CHOP: p3}, n_samples: N, avg_similarity: s}`

### 10.2 Expected Excursions

**Implementation**: `backend/src/ml/outcome_aggregation.py` (compute_expected_excursions)

Computes weighted expected values of favorable and adverse excursions (in ATR units) using power transform and recency decay.

**Result**: `{expected_excursion_favorable, expected_excursion_adverse, excursion_ratio}`

### 10.3 Conditional Excursions

**Implementation**: `backend/src/ml/outcome_aggregation.py` (compute_conditional_excursions)

For each outcome (BREAK, REJECT), compute weighted excursion statistics within that outcome subset using power transform and recency decay.

**Result**: `{BREAK: {expected_favorable, expected_adverse, mean_strength, n_samples}, REJECT: {...}}`

### 10.4 Multi-Horizon Distribution

**Implementation**: `backend/src/ml/outcome_aggregation.py` (compute_multi_horizon_distribution)

Computes outcome distributions independently for 2min, 4min, and 8min horizons using power transform, recency decay, and Dirichlet smoothing.

**Result**: `{2min: {probabilities: {...}, n_valid}, 4min: {...}, 8min: {...}}`

### 10.5 Bootstrap Confidence Intervals

**Implementation**: `backend/src/ml/outcome_aggregation.py` (compute_bootstrap_ci)

Uses weighted bootstrap resampling (n=1000) with power transform and recency decay to estimate uncertainty in outcome probabilities.

**Process**:
1. Compute weights: `(similarity^4) × exp(-age_days / 60)`
2. Resample outcomes with replacement using computed weights
3. Compute proportions for each bootstrap sample
4. Calculate 95% confidence intervals from bootstrap distribution

**Result**: `{BREAK: {mean, ci_low, ci_high, std}, REJECT: {...}, CHOP: {...}}`

### 10.6 Reliability Metrics

**Implementation**: `backend/src/ml/outcome_aggregation.py` (compute_reliability)

Quantifies retrieval quality and sample diversity.

**Metrics**:
- `n_retrieved`: Number of neighbors
- `effective_n`: Effective sample size accounting for weight concentration
- `avg_similarity`, `min_similarity`, `max_similarity`: Similarity statistics
- `similarity_std`: Spread of similarity scores
- `entropy`: Shannon entropy of normalized weights (higher = more diverse)

---

## 11. Attribution System

**Implementation**: `backend/src/ml/attribution.py`

Attribution explains both why neighbors were selected and what drives different outcomes within the neighborhood.

**Methods**:
- **Similarity Attribution**: Which features drove neighbor selection (inner product contributions)
- **Outcome Attribution**: Which features differentiate BREAK vs REJECT (weighted logistic regression)
- **Section Attribution**: Which vector sections lean toward each outcome (centroid distances)
- **Physics Attribution**: Aggregate coefficients into physics buckets (kinematics, order flow, liquidity, gamma, context)

### 11.1 Similarity Attribution

**Implementation**: `backend/src/ml/attribution.py` (compute_similarity_attribution)

Explains which features drove neighbor selection by decomposing inner product similarity.

**Method**: For L2-normalized vectors, `similarity = Σ(q_i × r_i)`. Each feature's contribution is `q_i × r_i`.

**Process**:
1. Normalize query and retrieved vectors
2. Compute element-wise products (per-feature contributions)
3. Aggregate across neighbors weighted by similarity
4. Sort by absolute contribution magnitude

**Result**: `{top_matching_features: [(feature, contribution), ...], all_attributions: {...}}`

### 11.2 Outcome Attribution (Local Surrogate Model)

**Implementation**: `backend/src/ml/attribution.py` (compute_outcome_attribution)

Explains which features differentiate BREAK vs REJECT within the retrieved neighborhood using weighted logistic regression as a local surrogate model.

**Process**:
1. Filter neighbors to BREAK and REJECT only (drop CHOP)
2. Fit weighted logistic regression (target: 1=BREAK, 0=REJECT, weights=similarities)
3. Extract coefficients
4. Positive coefficients → BREAK drivers
5. Negative coefficients → REJECT drivers

**Requirements**: Minimum 10 BREAK+REJECT neighbors

**Result**: `{top_break_drivers: [(feature, coef), ...], top_reject_drivers: [...], model_accuracy, all_coefficients}`

### 11.3 Section-Level Attribution

**Implementation**: `backend/src/ml/attribution.py` (compute_section_attribution)

Identifies which of the 6 vector sections (context/dynamics/micro-history/physics/trends/trajectory) differentiate BREAK vs REJECT.

**Method**:
1. Compute centroids for BREAK and REJECT subsets within each section
2. Measure query distance to each centroid
3. Determine which outcome the query section leans toward
4. Calculate confidence as relative distance difference

**Result**: Per-section analysis with `{dist_to_break, dist_to_reject, lean, confidence}`

### 11.4 Physics-Bucket Attribution

**Implementation**: `backend/src/ml/attribution.py` (compute_physics_attribution)

Aggregates feature-level attributions into interpretable physics categories.

**Physics Buckets**:
- **Kinematics**: Velocities, accelerations, jerks, momentum trends, DCT(d_atr)
- **Order Flow**: OFI series, tape metrics, DCT(ofi), DCT(tape)
- **Liquidity/Barrier**: Barrier deltas, depth, wall ratio, DCT(barrier)
- **Dealer Gamma**: GEX features, fuel_effect
- **Context**: Time-of-day, stacking, distances to other levels, touch memory

**Process**:
1. Match each feature to its physics bucket via pattern matching
2. Sum absolute coefficients within each bucket
3. Normalize bucket scores to sum to 1

**Result**: `{kinematics: score, order_flow: score, liquidity_barrier: score, dealer_gamma: score, context: score}`

---

## 12. Validation Framework

**Implementation**: `backend/src/ml/validation.py`

The validation framework ensures retrieval quality through temporal cross-validation, calibration monitoring, and drift detection.

**Methods**:
- **Temporal CV**: Expanding window splits, no leakage
- **Retrieval Metrics**: AUC, Brier score, log loss
- **Calibration Curve**: Reliability diagram + ECE
- **Lift Analysis**: Break rate vs threshold
- **Sanity Check**: Same-outcome neighbors closer than different-outcome
- **Drift Detection**: Wasserstein distance + mean shift per feature

### 12.1 Temporal Cross-Validation

**Implementation**: `backend/src/ml/validation.py` (temporal_cv_split)

Generates time-forward train/test splits to prevent leakage.

**Strategy**: Expanding window
- Minimum 60 days for training
- Test periods always follow training periods
- Typically 5 splits

**Constraint**: Training data strictly precedes test data (no overlap)

### 12.2 Retrieval Quality Metrics

**Implementation**: `backend/src/ml/validation.py` (evaluate_retrieval_system)

Comprehensive evaluation using temporal CV test set.

**Process**:
1. For each test episode, query retrieval system
2. Filter to quality retrievals (n_retrieved >= 10)
3. Extract predicted P(BREAK) and actual outcome
4. Compute standard probabilistic forecasting metrics

**Metrics**:
- AUC: Area under ROC curve (BREAK vs REJECT+CHOP)
- Brier score: Mean squared error of probabilities
- Log loss: Negative log-likelihood
- Calibration curve: Reliability diagram
- Lift analysis: Break rate at various confidence thresholds

### 12.3 Calibration Curve

**Implementation**: `backend/src/ml/validation.py` (compute_calibration_curve)

Reliability diagram showing predicted vs observed frequencies in probability bins.

**Process**:
1. Bin predictions into 10 deciles
2. Compute mean predicted probability and observed frequency per bin
3. Calculate Expected Calibration Error (ECE): weighted mean absolute deviation

**Result**: `{bin_means, bin_true_freqs, bin_counts, expected_calibration_error}`

**Interpretation**: Well-calibrated model has bin_means ≈ bin_true_freqs

### 12.4 Lift Analysis

**Implementation**: `backend/src/ml/validation.py` (compute_lift_analysis)

Measures how much better the system performs when filtering to high-confidence predictions.

**Thresholds**: [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

For each threshold, compute:
- Number of cases with P(BREAK) >= threshold
- Break rate among those cases
- Lift: `high_conf_rate / base_rate`

**Result**: `{base_rate, lift_by_threshold: [{threshold, n_samples, break_rate, lift}, ...]}`

### 12.5 Similarity Sanity Check

**Implementation**: `backend/src/ml/validation.py` (sanity_check_similarity)

Verifies that the vector space has meaningful structure: episodes with the same outcome should be closer together than episodes with different outcomes.

**Method**:
1. Sample 500 random episodes
2. For each, find its nearest neighbor (cosine distance)
3. Track whether nearest neighbor has same or different outcome
4. Compute mean distance for same-outcome vs different-outcome pairs

**Expected**: `same_outcome_mean_dist < diff_outcome_mean_dist`

**Result**: `{same_outcome_mean_dist, diff_outcome_mean_dist, separation_ratio, interpretation}`

### 12.6 Feature Drift Detection

**Implementation**: `backend/src/ml/validation.py` (detect_feature_drift)

Monitors feature distribution changes over time to detect regime shifts or data quality issues.

**Method**:
- Compare historical baseline (60 days ago) vs recent period (last 5 days)
- Compute Wasserstein distance (distribution shift) per feature
- Compute mean shift in standard deviation units per feature

**Thresholds**:
- Warning: Wasserstein distance > 0.5
- Alert: Mean shift > 2.0 std

**Result**: DataFrame sorted by drift magnitude with columns `{feature, wasserstein_distance, mean_shift_std, hist_mean, recent_mean}`

---

## 13. Pipeline Integration

**Implementation**: `backend/src/pipeline/pipelines/es_pipeline.py` (v3.1)

The ES pipeline consists of 18 stages that transform raw Databento data through bronze, silver, and gold layers, culminating in episode vectors ready for retrieval.

**Pipeline Stages** (18 stages, 0-indexed):

| Index | Stage Name | Description |
|-------|------------|-------------|
| 0 | LoadBronze | Load ES futures + options from Bronze layer |
| 1 | BuildOHLCV (1min) | 1-minute OHLCV for ATR computation |
| 2 | BuildOHLCV (2min) | 2-minute OHLCV for SMA calculation |
| 3 | InitMarketState | Market state + Greeks initialization |
| 4 | GenerateLevels | 6 level kinds (PM/OR high/low, SMA_200/400) |
| 5 | DetectInteractionZones | Event detection at zone entry |
| 6 | ComputePhysics | Barrier/Tape/Fuel computation |
| 7 | ComputeMultiWindowKinematics | Velocity/Accel/Jerk at 5 scales |
| 8 | ComputeMultiWindowOFI | OFI at 4 windows (30/60/120/300s) |
| 9 | ComputeBarrierEvolution | Barrier depth changes (1/3/5min) |
| 10 | ComputeLevelDistances | Signed distance to all levels |
| 11 | ComputeGEXFeatures | Gamma exposure features |
| 12 | ComputeForceMass | F=ma validation features |
| 13 | ComputeApproach | Approach dynamics and clustering |
| 14 | LabelOutcomes | First-crossing labels (BREAK/REJECT/CHOP) |
| 15 | FilterRTH | Filter to RTH 09:30-12:30 |
| 16 | MaterializeStateTable | 30s cadence state table |
| 17 | ConstructEpisodes | 144-dim episode vectors |

**Offline Stages** (run separately):
- Normalization Stats Computation (from 60 days of state data)
- Index Building (60 FAISS partitions, daily or incremental)
- Validation (weekly quality monitoring)

### 13.1 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

BRONZE LAYER (Raw Databento, 250ms)
    │
    ▼
SILVER LAYER (Existing es_pipeline, 15 stages)
    │
    ├──► silver/features/es_pipeline/*.parquet  [Event records]
    │
    ▼
STAGE 16: State Table Materialization
    │
    └──► silver/state/es_level_state/*.parquet  [30s cadence state]
    
    │
    ▼
OFFLINE: Normalization Statistics (Daily, expanding window)
    │
    └──► gold/normalization/stats_v{N}.json
    
    │
    ▼
STAGE 17: Episode Vector Construction
    │
    └──► gold/episodes/es_level_episodes/vectors/*.npy
    └──► gold/episodes/es_level_episodes/metadata/*.parquet
    
    │
    ▼
OFFLINE: Index Building (Daily, after RTH close)
    │
    └──► gold/indices/es_level_indices/{level}/{dir}/{bucket}/*
    
    │
    ▼
OFFLINE: Validation (Weekly)
    │
    └──► gold/validation/calibration_*.json
    └──► gold/validation/drift_*.json
```

### 13.2 Stage Specifications

**Stage 16: State Table Materialization**

```
Input:  silver/features/es_pipeline/date=YYYY-MM-DD/*.parquet
        + raw price data at 30s intervals
Output: silver/state/es_level_state/date=YYYY-MM-DD/*.parquet

Logic:
1. Generate timestamps at 30s intervals from 09:30 to 12:30 ET
2. For each timestamp and each level_kind:
   a. Compute level_price (static for PM/OR, dynamic for SMA)
   b. Interpolate/forward-fill features from event table
   c. Compute distance_signed_atr
   d. Write row
3. Partition by date

Constraints:
- All features must be online-safe (no future data)
- Handle OR levels being undefined before 09:45
```

**Normalization Statistics Computation**

```
Input:  silver/state/es_level_state/ (last 60 days)
Output: gold/normalization/stats_v{N}.json

Logic:
1. Load 60 days of state table
2. For each of 144 features, compute statistics per Section 7
3. Save JSON with version number

Schedule: Daily at 05:00 ET (before market open)
```

**Stage 17: Episode Vector Construction**

```
Input:  silver/features/es_pipeline/date=YYYY-MM-DD/*.parquet (event anchors)
        silver/state/es_level_state/date=YYYY-MM-DD/*.parquet (state for windows)
        gold/normalization/stats_v{N}.json
Output: gold/episodes/es_level_episodes/vectors/date=YYYY-MM-DD/episodes.npy [N × 144]
        gold/episodes/es_level_episodes/metadata/date=YYYY-MM-DD/metadata.parquet

Logic:
1. For each event (anchor) in event table:
   a. Get anchor timestamp t0
   b. Extract 5-bar micro-history from state table (t0-4*30s to t0)
   c. Extract 40-bar trajectory window (t0-39*30s to t0, 20 minutes)
   d. Compute DCT coefficients for 4 time series
   e. Construct raw 144-dim vector (Section 6.11)
   f. Normalize vector (Section 7)
   g. Compute labels (Section 3)
   h. Compute emission_weight (Section 5.3)
   i. Store vector and metadata
2. Partition by date

Schedule: Daily, after RTH close (16:15 ET)
```

**Index Building (Offline)**

```
Input:  gold/episodes/es_level_episodes/ (all dates)
Output: gold/indices/es_level_indices/{level}/{dir}/{bucket}/

Logic:
1. Load all episode vectors and metadata
2. For each of 60 partitions (level_kind, direction, time_bucket):
   a. Filter to partition
   b. Select index type based on corpus size (Flat/IVF/IVFPQ)
   c. Build FAISS index with L2-normalized vectors
   d. Save index.faiss, vectors.npy, metadata.parquet
3. Update config.json with build timestamp

Schedule: Daily at 17:00 ET (or incremental with weekly full rebuild)
```

**Validation (Offline)**

```
Input:  gold/episodes/ (all)
        gold/indices/ (current)
Output: gold/validation/calibration_YYYY-MM-DD.json
        gold/validation/drift_YYYY-MM-DD.json
        gold/validation/sanity_YYYY-MM-DD.json

Logic:
1. Run temporal CV evaluation (Section 12.2)
2. Run calibration analysis (Section 12.3)
3. Run drift detection (Section 12.6)
4. Run sanity checks (Section 12.5)
5. Alert if metrics degrade

Schedule: Weekly (Saturday)
```

### 13.3 Storage Layout

```
data/
├── silver/
│   ├── features/
│   │   └── es_pipeline/
│   │       └── date=YYYY-MM-DD/*.parquet
│   └── state/
│       └── es_level_state/
│           └── date=YYYY-MM-DD/*.parquet
│
└── gold/
    ├── normalization/
    │   ├── stats_v001.json
    │   ├── stats_v002.json
    │   └── current -> stats_v002.json
    │
    ├── episodes/
    │   └── es_level_episodes/
    │       ├── vectors/
    │       │   └── date=YYYY-MM-DD/episodes.npy
    │       ├── metadata/
    │       │   └── date=YYYY-MM-DD/metadata.parquet
    │       └── corpus/
    │           ├── all_vectors.npy          # Memory-mapped
    │           └── all_metadata.parquet
    │
    ├── indices/
    │   └── es_level_indices/
│       ├── PM_HIGH/UP/T0_15/
│       ├── PM_HIGH/UP/T15_30/
│       ├── PM_HIGH/UP/T30_60/
    │       ├── ... (60 partitions total)
    │       └── config.json
    │
    └── validation/
        ├── calibration_YYYY-MM-DD.json
        ├── drift_YYYY-MM-DD.json
        └── sanity_YYYY-MM-DD.json
```

---

## Appendix A: Complete Feature Specification

### A.1 Section A: Context + Regime (25 features)

| Index | Feature | Source | Normalization |
|-------|---------|--------|---------------|
| 0 | minutes_since_open | minutes_since_open | MinMax [0, 180] |
| 1 | bars_since_open | bars_since_open | MinMax [0, 90] |
| 2 | atr | atr | Z-Score |
| 3 | or_active | computed | Passthrough (0/1) |
| 4 | level_stacking_2pt | level_stacking_2pt | MinMax [0, 6] |
| 5 | level_stacking_5pt | level_stacking_5pt | MinMax [0, 6] |
| 6 | level_stacking_10pt | level_stacking_10pt | MinMax [0, 6] |
| 7 | dist_to_pm_high_atr | dist_to_pm_high_atr | Z-Score |
| 8 | dist_to_pm_low_atr | dist_to_pm_low_atr | Z-Score |
| 9 | dist_to_or_high_atr | dist_to_or_high_atr | Z-Score |
| 10 | dist_to_or_low_atr | dist_to_or_low_atr | Z-Score |
| 11 | dist_to_sma_200_atr | dist_to_sma_200_atr | Z-Score |
| 12 | dist_to_sma_400_atr | dist_to_sma_400_atr | Z-Score |
| 13 | prior_touches | prior_touches | MinMax [0, 10] |
| 14 | attempt_index | attempt_index | MinMax [0, 10] |
| 15 | time_since_last_touch_sec | time_since_last_touch | MinMax |
| 16 | gamma_exposure | gamma_exposure | Robust |
| 17 | fuel_effect_encoded | fuel_effect | Passthrough |
| 18 | gex_ratio | gex_ratio | Robust |
| 19 | gex_asymmetry | gex_asymmetry | Robust |
| 20 | net_gex_2strike | net_gex_2strike | Robust |
| 21 | gex_above_1strike | gex_above_1strike | Robust |
| 22 | gex_below_1strike | gex_below_1strike | Robust |
| 23 | call_gex_above_2strike | call_gex_above_2strike | Robust |
| 24 | put_gex_below_2strike | put_gex_below_2strike | Robust |

**Note**: `level_kind` and `direction` are partition keys and are NOT included in the vector.

### A.2 Section B: Multi-Scale Dynamics (37 features)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 25-29 | velocity_{1,3,5,10,20}min | Z-Score |
| 30-34 | acceleration_{1,3,5,10,20}min | Z-Score |
| 35-39 | jerk_{1,3,5,10,20}min | Z-Score |
| 40-43 | momentum_trend_{3,5,10,20}min | Z-Score |
| 44-47 | ofi_{30s,60s,120s,300s} | Robust |
| 48-51 | ofi_near_level_{30s,60s,120s,300s} | Robust |
| 52 | ofi_acceleration | Robust |
| 53-55 | barrier_delta_{1,3,5}min | Robust |
| 56-58 | barrier_pct_change_{1,3,5}min | Robust |
| 59 | approach_velocity | Z-Score |
| 60 | approach_bars | MinMax [0, 40] |
| 61 | approach_distance_atr | Z-Score |

### A.3 Section C: Micro-History (35 features)

**Log Transforms**: Barrier and wall features use log transforms for stability.

| Index | Feature × Time | Transform | Normalization |
|-------|----------------|-----------|---------------|
| 62-66 | d_atr[t-4..t0] | None | Z-Score |
| 67-71 | tape_imbalance[t-4..t0] | None | Robust |
| 72-76 | tape_velocity[t-4..t0] | None | Robust |
| 77-81 | ofi_60s[t-4..t0] | None | Robust |
| 82-86 | barrier_delta_liq_log[t-4..t0] | Signed log | Robust |
| 87-91 | wall_ratio_log[t-4..t0] | Log | Robust |
| 92-96 | gamma_exposure[t-4..t0] | None | Robust |

### A.4 Section D: Derived Physics (11 features)

| Index | Feature | Computation | Normalization |
|-------|---------|-------------|---------------|
| 97 | predicted_accel | F=ma model | Z-Score |
| 98 | accel_residual | Actual - predicted | Robust |
| 99 | force_mass_ratio | F=ma model | Robust |
| 100 | mass_proxy | log1p(barrier_depth) | Robust |
| 101 | force_proxy | ofi_60s / (mass_proxy + ε) | Robust |
| 102 | barrier_state_encoded | Encoded [-2, +2] | Passthrough |
| 103 | barrier_replenishment_ratio | Barrier metric | Robust |
| 104 | sweep_detected | Binary 0/1 | Passthrough |
| 105 | tape_log_ratio | log((buy+1)/(sell+1)) | Robust |
| 106 | tape_log_total | log(buy+sell+1) | Robust |
| 107 | flow_alignment | ofi_60s * (-sign(d_atr)) | Robust |

### A.5 Section E: Online Trends (4 features)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 108 | barrier_replenishment_trend | Robust |
| 109 | barrier_delta_liq_trend | Robust |
| 110 | tape_velocity_trend | Robust |
| 111 | tape_imbalance_trend | Robust |

### A.6 Section F: Trajectory Basis (32 features)

DCT-II coefficients encoding 20-minute approach trajectory.

| Index | Series | Window | Normalization |
|-------|--------|--------|---------------|
| 112-119 | DCT(d_atr) | 40 samples @ 30s | Z-Score |
| 120-127 | DCT(ofi_60s) | 40 samples @ 30s | Z-Score |
| 128-135 | DCT(barrier_delta_liq_log) | 40 samples @ 30s | Z-Score |
| 136-143 | DCT(tape_imbalance) | 40 samples @ 30s | Z-Score |

Each series produces 8 DCT coefficients (c0..c7) for a total of 32 dimensions.

---

## Appendix B: Schema Reference

### B.1 QueryResult Schema

```
QueryResult:
    outcome_probabilities:
        probabilities:
            BREAK: float [0, 1]
            REJECT: float [0, 1]
            CHOP: float [0, 1]
        n_samples: int
        avg_similarity: float
    
    confidence_intervals:
        BREAK: {mean: float, ci_low: float, ci_high: float, std: float}
        REJECT: {mean: float, ci_low: float, ci_high: float, std: float}
        CHOP: {mean: float, ci_low: float, ci_high: float, std: float}
    
    multi_horizon:
        2min: {probabilities: {...}, n_valid: int}
        4min: {probabilities: {...}, n_valid: int}
        8min: {probabilities: {...}, n_valid: int}
    
    expected_excursions:
        expected_excursion_favorable: float (ATR units)
        expected_excursion_adverse: float (ATR units)
        excursion_ratio: float
    
    conditional_excursions:
        BREAK: {expected_favorable: float, expected_adverse: float, mean_strength: float, n_samples: int}
        REJECT: {expected_favorable: float, expected_adverse: float, mean_strength: float, n_samples: int}
    
    attribution:
        similarity:
            top_matching_features: list[(feature_name, contribution)]
        outcome:
            top_break_drivers: list[(feature_name, coefficient)]
            top_reject_drivers: list[(feature_name, coefficient)]
            model_accuracy: float
        section:
            {section_name: {dist_to_break, dist_to_reject, lean, confidence}}
        physics:
            {bucket_name: normalized_score}
    
    reliability:
        n_retrieved: int
        effective_n: float
        avg_similarity: float
        min_similarity: float
        max_similarity: float
        entropy: float
    
    neighbors: list[
        {
            event_id: string
            date: date
            similarity: float
            outcome_2min: string
            outcome_4min: string
            outcome_8min: string
            excursion_favorable: float
            excursion_adverse: float
            strength_abs: float
        }
    ]
    
    query_metadata:
        level_kind: string
        direction: string
        time_bucket: string
        timestamp: datetime
        emission_weight: float
```

---

## Appendix C: Constants and Thresholds

**File**: `backend/src/ml/constants.py`

### C.1 Zone and Trigger Parameters

```python
Z_APPROACH_ATR = 2.0              # Episode triggers within this distance
Z_CONTACT_ATR = 0.20              # "At level" threshold
Z_EXIT_ATR = 2.5                  # Episode exit threshold
MAX_EXIT_GAP_BARS = 2             # Allow brief excursions

MIN_APPROACH_BARS = 2             # Minimum bars approaching level
MIN_APPROACH_V_ATR_PER_MIN = 0.10 # Minimum velocity (ATR/min)
```

### C.2 Time Buckets

```python
TIME_BUCKETS = {
    'T0_15':    (0, 15),      # OR formation period
    'T15_30':   (15, 30),     # Post-OR early
    'T30_60':   (30, 60),     # Minutes 30-60
    'T60_120':  (60, 120),    # Minutes 60-120
    'T120_180': (120, 180)    # Minutes 120-180
}
```

### C.3 Retrieval Parameters

```python
M_CANDIDATES = 500                # Over-fetch from FAISS
K_NEIGHBORS = 50                  # Final neighbors after dedup
MAX_PER_DAY = 2                   # Max neighbors from same date
MAX_PER_EPISODE = 1               # Max neighbors from same episode_id

SIM_POWER = 4.0                   # Power transform on similarity
RECENCY_HALFLIFE_DAYS = 60        # Exponential decay halflife

MIN_SIMILARITY_THRESHOLD = 0.70   # Minimum similarity for quality result
MIN_SAMPLES_THRESHOLD = 30        # Minimum neighbors for reliable estimate
N_EFF_MIN = 15                    # Minimum effective sample size

CACHE_TTL_SECONDS = 30            # Query result cache TTL
```

### C.4 Normalization Parameters

```
LOOKBACK_DAYS = 60                # Days of history for normalization stats
CLIP_SIGMA = 4.0                  # Clip normalized values at ±4σ
```

### C.5 Validation Thresholds

```python
MIN_PARTITION_SIZE = 100          # Don't create index for smaller partitions
DRIFT_WARNING_WASSERSTEIN = 0.5   # Feature drift warning threshold
DRIFT_ALERT_MEAN_SHIFT = 2.0      # Mean shift alert threshold (in std)
CALIBRATION_WARNING_ECE = 0.10    # Expected calibration error warning
```

### C.6 Level Kinds

```python
LEVEL_KINDS = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_200', 'SMA_400']
```

### C.8 Vector Parameters

```python
VECTOR_DIMENSION = 144            # Episode vector dimensionality
STATE_CADENCE_SEC = 30            # State table cadence
LOOKBACK_WINDOW_MIN = 20          # For trajectory basis DCT
```

### C.7 Horizons

```
HORIZONS = {
    '2min': 120,    # seconds
    '4min': 240,
    '8min': 480
}
PRIMARY_HORIZON = '4min'
```

---

## End of Specification

**Key Design Decisions:**

1. **144 dimensions**: Optimal for ANN search while capturing full approach dynamics
2. **DCT trajectory basis**: Frequency-domain encoding of 20-minute approach shape
3. **Log transforms**: Stabilize heavy-tailed barrier/liquidity features (barrier_delta_liq_log, wall_ratio_log)
4. **5 time buckets**: Finer resolution in first 30 min when OR is forming (T0_15, T15_30, T30_60, T60_120, T120_180)
5. **60 partitions**: Ensures regime-comparable neighbors (6 levels × 2 directions × 5 time buckets)
6. **2.0 ATR zone**: Tighter threshold for higher-quality anchors
7. **Deduplication**: Retrieve 500 candidates, apply constraints (max 2/day, 1/episode), return top 50
8. **Neighbor weighting**: Power transform (similarity^4) + recency decay (exp(-age/60)) for quality emphasis
9. **Dirichlet posterior**: Bayesian smoothing prevents extreme probabilities with small samples

**Critical Invariants:**

- All features in vectors are online-safe (no future data)
- All price-based features are ATR-normalized
- All distance features use consistent sign convention: `(spot - level) / ATR`
- Temporal CV prevents any data leakage
- Index partitioning ensures regime-comparable neighbors
- DCT coefficients computed only from historical trajectory window
- Neighbor weights combine similarity quality and temporal relevance
