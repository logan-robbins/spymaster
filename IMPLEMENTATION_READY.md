# IMPLEMENTATION_READY.md
## Level Interaction Similarity Retrieval System — Canonical Specification

**Version**: 3.0  
**Purpose**: Complete specification for AI coding agent implementation  
**Scope**: First 3 hours of RTH (09:30-12:30 ET), ES Futures  

---

## 🎉 IMPLEMENTATION STATUS: COMPLETE ✅

**Implementation Date**: December 29, 2025  
**Completion**: 10/10 sections implemented (100%)

### Implementation Summary

All core components have been implemented per specification:

✅ **Section 3**: Outcome label contract (first-crossing, BREAK/REJECT/CHOP)  
✅ **Section 4**: State table materialization (30s cadence, Stage 16)  
✅ **Section 7**: Normalization statistics (robust/zscore/minmax, Stage 17)  
✅ **Section 6**: Episode vector construction (111 dims, Stage 18)  
✅ **Section 8**: Index building (FAISS, 48 partitions, Stage 19)  
✅ **Section 9**: Retrieval pipeline (IndexManager, LiveBuilder, QueryEngine)  
✅ **Section 10**: Outcome aggregation (weighted distributions, CIs, excursions)  
✅ **Section 11**: Attribution system (similarity, outcome, section, physics)  
✅ **Section 12**: Validation framework (temporal CV, calibration, drift, sanity)  
✅ **Section 13**: Pipeline integration (updated es_pipeline to v3.0.0)

### Key Files Created

**Core Modules**:
- `backend/src/ml/normalization.py` - Normalization statistics computation
- `backend/src/ml/episode_vector.py` - 111-dimensional vector construction
- `backend/src/ml/index_builder.py` - FAISS index building
- `backend/src/ml/retrieval_engine.py` - Real-time query engine
- `backend/src/ml/outcome_aggregation.py` - Outcome distributions
- `backend/src/ml/attribution.py` - Explainability system
- `backend/src/ml/validation.py` - Quality monitoring

**Pipeline Stages**:
- `backend/src/pipeline/stages/label_outcomes.py` - Updated to first-crossing
- `backend/src/pipeline/stages/materialize_state_table.py` - Stage 16
- `backend/src/pipeline/stages/construct_episodes.py` - Stage 18

**Pipeline Integration**:
- `backend/src/pipeline/pipelines/es_pipeline.py` - Updated to v3.0.0

### Next Steps (Operational)

1. **Compute Normalization Stats**: Run Stage 17 on 60 days of state table data
2. **Build Indices**: Run Stage 19 to build FAISS indices from episode corpus
3. **Deploy Query Service**: Initialize RealTimeQueryService with normalization stats and IndexManager
4. **Run Validation**: Execute ValidationRunner weekly to monitor quality
5. **Monitor Drift**: Check feature drift detection output regularly

The system is ready for backtesting and deployment.

### Validation Scripts (Updated for v3.0.0)

**Main Pipeline Validator**: `backend/scripts/validate_es_pipeline.py`
- Updated to v3.0.0 with 6 QA gates
- Now checks for REJECT (not BOUNCE) in outcomes
- Validates ATR-normalized excursion fields
- Usage: `uv run python backend/scripts/validate_es_pipeline.py --date 2024-12-20`

**Stage Validators Created/Updated** (filenames use 1-based, code uses 0-based indices):
- ✅ **Stage 14** (`validate_stage_14_label_outcomes.py`, index=14) - Updated for first-crossing, REJECT, new excursion fields
- ✅ **Stage 16** (`validate_stage_16_materialize_state_table.py`, index=16) - NEW: Validates 30s state table, OR inactive check
- ✅ **Stage 17** (`validate_stage_18_construct_episodes.py`, index=17) - NEW: Validates 111-dim vectors, metadata schema
  
**⚠️ Stage Indexing Note**: Pipeline uses 0-based indices (0-17). Validators named with 1-based for readability but use correct 0-based indices internally.

**Schema Updates**:
- ✅ **Silver Schema** (`backend/SILVER_SCHEMA.md`) - Added v3.0.0 notes on REJECT and new fields
- ✅ **README.md** - Updated to lean operational guide, points to IMPLEMENTATION_READY.md

**How to Run Stage Validators**:
```bash
# Stage 14 (Label Outcomes)
uv run python backend/scripts/validate_stage_14_label_outcomes.py --date 2024-12-20

# Stage 16 (State Table, index=16)
uv run python backend/scripts/validate_stage_16_materialize_state_table.py --date 2024-12-20

# Stage 17 (Episode Vectors, index=17, filename says "18" for 1-based docs)
uv run python backend/scripts/validate_stage_18_construct_episodes.py --date 2024-12-20
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

## 3. Outcome Label Contract ✅ COMPLETE

**Implementation**: `backend/src/pipeline/stages/label_outcomes.py`  
**Status**: Updated to use first-crossing semantics with REJECT terminology  
**Changes**: 
- Replaced BOUNCE with REJECT
- Simplified to use 1.0 ATR threshold (fixed)
- Removed dynamic barrier computation
- Added excursion_favorable and excursion_adverse (ATR-normalized)
- Multi-horizon labels: outcome_2min, outcome_4min, outcome_8min

### 3.1 Label Function

The outcome label is determined by **first-crossing semantics** using existing schema fields.

```
FUNCTION compute_outcome_label(
    direction: string,
    time_to_break: float | null,    # time_to_break_1_{H} field
    time_to_bounce: float | null,   # time_to_bounce_1_{H} field
) -> string:

    # Treat null as "never hit" (infinity)
    t_break = time_to_break if time_to_break is not null else +∞
    t_bounce = time_to_bounce if time_to_bounce is not null else +∞
    
    # Neither threshold crossed within horizon
    IF t_break == +∞ AND t_bounce == +∞:
        RETURN 'CHOP'
    
    # First crossing determines outcome
    IF t_break < t_bounce:
        RETURN 'BREAK'
    ELSE IF t_bounce < t_break:
        RETURN 'REJECT'
    ELSE:
        # Exact tie (same timestamp) - rare, treat as CHOP
        RETURN 'CHOP'
```

### 3.2 Label Semantics by Direction

| Direction | BREAK means | REJECT means |
|-----------|-------------|--------------|
| UP | Price crossed above level and held | Price failed to cross or reversed down |
| DOWN | Price crossed below level and held | Price failed to cross or reversed up |

### 3.3 Multi-Horizon Labels

Compute labels independently for each horizon:

```
outcome_2min = compute_outcome_label(direction, time_to_break_1_2min, time_to_bounce_1_2min)
outcome_4min = compute_outcome_label(direction, time_to_break_1_4min, time_to_bounce_1_4min)
outcome_8min = compute_outcome_label(direction, time_to_break_1_8min, time_to_bounce_1_8min)
```

### 3.4 Continuous Outcome Variables

In addition to discrete labels, store continuous outcome measures:

```
# Favorable excursion: movement in the "break" direction
IF direction == 'UP':
    excursion_favorable = excursion_max / atr
    excursion_adverse = abs(excursion_min) / atr
ELSE:  # DOWN
    excursion_favorable = abs(excursion_min) / atr
    excursion_adverse = excursion_max / atr

# Time to resolution (first threshold hit)
time_to_resolution = min(
    time_to_break if time_to_break is not null else +∞,
    time_to_bounce if time_to_bounce is not null else +∞
)
```

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

## 4. Level-Relative State Table ✅ COMPLETE

**Implementation**: `backend/src/pipeline/stages/materialize_state_table.py`  
**Status**: New Stage 16 created  
**Features**:
- Samples every 30 seconds from 09:30-12:30 ET (360 samples/level/day)
- One row per (timestamp, level_kind) pair
- Forward-fills features from event table (all online-safe)
- Handles OR levels being inactive before 09:45
- Computes dynamic SMA levels from 2min OHLCV

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

A touch anchor is created when price first enters the interaction zone for a level.

**Source**: Existing event table rows (each `event_id` is a touch anchor)

**Anchor timestamp**: `confirm_ts_ns` from event table (or `ts_ns` if confirm not available)

**Properties**:
- High signal-to-noise (price is at/near level)
- Clear decision point
- Well-defined outcome measurement window

### 5.2 Time Bucket Assignment

Assign each anchor to a time bucket based on `minutes_since_open`:

```
FUNCTION assign_time_bucket(minutes_since_open: float) -> string:
    IF minutes_since_open < 30:
        RETURN 'T0_30'
    ELSE IF minutes_since_open < 60:
        RETURN 'T30_60'
    ELSE IF minutes_since_open < 120:
        RETURN 'T60_120'
    ELSE:
        RETURN 'T120_180'
```

### 5.3 Emission Weight

Each anchor receives a quality weight used in retrieval:

```
FUNCTION compute_emission_weight(
    spot: float,
    level_price: float,
    atr: float,
    approach_velocity: float,
    ofi_60s: float
) -> float:

    distance_atr = abs(spot - level_price) / atr
    
    # Proximity weight: closer to level = higher weight
    # 1.0 at level, 0.5 at 1 ATR, 0.1 at 3.5 ATR
    proximity_w = exp(-distance_atr / 1.5)
    
    # Velocity weight: faster approach = more decisive setup
    # Clip to [0.2, 1.0]
    velocity_w = clip(abs(approach_velocity) / 2.0, 0.2, 1.0)
    
    # OFI alignment: flow in direction of approach = cleaner signal
    ofi_sign = sign(ofi_60s)
    approach_sign = sign(level_price - spot)  # positive if approaching from below
    ofi_aligned = (ofi_sign == approach_sign) OR (ofi_sign == 0)
    ofi_w = 1.0 if ofi_aligned else 0.6
    
    RETURN proximity_w * velocity_w * ofi_w
```

---

## 6. Episode Vector Architecture ✅ COMPLETE

**Implementation**: `backend/src/ml/episode_vector.py`, `backend/src/pipeline/stages/construct_episodes.py`  
**Status**: Stage 18 created  
**Features**:
- Constructs 111-dimensional vectors from events + state table
- 5 sections: Context (26), Trajectory (37), Micro-History (35), Physics (9), Trends (4)
- Extracts 5-bar (2.5min) history windows at 30s cadence
- Applies normalization using precomputed stats
- Computes labels (outcome_2min/4min/8min) and emission weights
- Outputs: vectors.npy + metadata.parquet (date-partitioned)

### 6.1 Design Principles

The episode vector is a **1-dimensional array** optimized for similarity search:

1. **Level-relative**: All price-based features expressed relative to tested level
2. **ATR-normalized**: Scale-invariant across price regimes
3. **Hybrid architecture**: Combines slow context, multi-scale trajectory, and fast micro-history
4. **Optimal dimensionality**: 111 dimensions (within 100-200 optimal range for ANN search)

### 6.2 Vector Sections

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EPISODE VECTOR (111 DIMENSIONS)                         │
│                                                                                 │
│  ┌──────────────────┐ ┌──────────────────────┐ ┌─────────────────────────────┐ │
│  │  SECTION A:      │ │  SECTION B:          │ │  SECTION C:                 │ │
│  │  CONTEXT STATE   │ │  MULTI-SCALE         │ │  MICRO-HISTORY              │ │
│  │  (T=0 snapshot)  │ │  TRAJECTORY          │ │  (T-4 to T=0, 5 bars)       │ │
│  │                  │ │  (T=0, encodes       │ │                             │ │
│  │  26 dims         │ │   temporal dynamics) │ │  35 dims                    │ │
│  │                  │ │                      │ │  (7 features × 5 bars)      │ │
│  │  • Level ID      │ │  37 dims             │ │                             │ │
│  │  • Session pos   │ │                      │ │  • distance_signed_atr      │ │
│  │  • GEX struct    │ │  • Velocity scales   │ │  • tape_imbalance           │ │
│  │  • Stacking      │ │  • Accel scales      │ │  • tape_velocity            │ │
│  │  • Ref distances │ │  • Jerk scales       │ │  • ofi_60s                  │ │
│  │  • Touch memory  │ │  • Momentum trends   │ │  • barrier_delta_liq        │ │
│  │                  │ │  • OFI scales        │ │  • wall_ratio               │ │
│  │                  │ │  • Barrier evolution │ │  • gamma_exposure           │ │
│  │                  │ │  • Approach dynamics │ │                             │ │
│  └──────────────────┘ └──────────────────────┘ └─────────────────────────────┘ │
│                                                                                 │
│  ┌──────────────────┐ ┌──────────────────────┐                                 │
│  │  SECTION D:      │ │  SECTION E:          │                                 │
│  │  DERIVED PHYSICS │ │  CLUSTER TRENDS      │                                 │
│  │                  │ │                      │                                 │
│  │  9 dims          │ │  4 dims              │                                 │
│  │                  │ │                      │                                 │
│  │  • Force model   │ │  • Replenish trend   │                                 │
│  │  • Barrier state │ │  • Delta liq trend   │                                 │
│  │  • Current flow  │ │  • Tape trends       │                                 │
│  └──────────────────┘ └──────────────────────┘                                 │
│                                                                                 │
│  TOTAL: 26 + 37 + 35 + 9 + 4 = 111 dimensions                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Section A: Context State (26 dimensions)

Single snapshot at T=0. These features define the environment and change slowly.

```
Index   Feature                      Encoding
─────   ───────                      ────────
0       level_kind                   Ordinal: PM_HIGH=0, PM_LOW=1, OR_HIGH=2, OR_LOW=3, SMA_200=4, SMA_400=5
1       direction                    UP=1, DOWN=-1
2       minutes_since_open           Raw (will be MinMax normalized)
3       bars_since_open              Raw (will be MinMax normalized)
4       atr                          Raw (will be robust normalized)
5       gex_asymmetry                Raw
6       gex_ratio                    Raw
7       net_gex_2strike              Raw
8       gamma_exposure               Raw
9       gex_above_1strike            Raw
10      gex_below_1strike            Raw
11      call_gex_above_2strike       Raw
12      put_gex_below_2strike        Raw
13      fuel_effect                  AMPLIFY=1, NEUTRAL=0, DAMPEN=-1
14      level_stacking_2pt           Raw
15      level_stacking_5pt           Raw
16      level_stacking_10pt          Raw
17      dist_to_pm_high_atr          Raw
18      dist_to_pm_low_atr           Raw
19      dist_to_or_high_atr          Raw (0 if OR not yet established)
20      dist_to_or_low_atr           Raw (0 if OR not yet established)
21      dist_to_sma_200_atr          Raw
22      dist_to_sma_400_atr          Raw
23      prior_touches                Raw
24      attempt_index                Raw
25      attempt_cluster_id_mod       attempt_cluster_id % 1000 (bounded)
```

### 6.4 Section B: Multi-Scale Trajectory (37 dimensions)

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

### 6.5 Section C: Micro-History (35 dimensions)

5-bar history (T-4 to T=0) for 7 fast-changing features. These capture the immediate approach dynamics.

```
Index   Feature × Time
─────   ──────────────
63-67   distance_signed_atr[t-4], distance_signed_atr[t-3], ..., distance_signed_atr[t0]
68-72   tape_imbalance[t-4..t0]
73-77   tape_velocity[t-4..t0]
78-82   ofi_60s[t-4..t0]
83-87   barrier_delta_liq[t-4..t0]
88-92   wall_ratio[t-4..t0]
93-97   gamma_exposure[t-4..t0]
```

**Bar cadence for micro-history**: Use the state table cadence (30 seconds), so 5 bars = 2.5 minutes of micro-history.

### 6.6 Section D: Derived Physics (9 dimensions)

```
Index   Feature
─────   ───────
98      predicted_accel
99      accel_residual
100     force_mass_ratio
101     barrier_state_encoded        STRONG_SUPPORT=2, WEAK_SUPPORT=1, NEUTRAL=0, WEAK_RESISTANCE=-1, STRONG_RESISTANCE=-2
102     barrier_depth_current
103     barrier_replenishment_ratio
104     sweep_detected               0 or 1
105     tape_log_ratio               log(tape_buy_vol / tape_sell_vol)
106     tape_log_total               log(tape_buy_vol + tape_sell_vol)
```

### 6.7 Section E: Cluster Trends (4 dimensions)

```
Index   Feature
─────   ───────
107     barrier_replenishment_trend
108     barrier_delta_liq_trend
109     tape_velocity_trend
110     tape_imbalance_trend
```

### 6.8 Vector Section Indices

```
VECTOR_SECTIONS = {
    'context_state':        (0, 26),
    'multiscale_trajectory': (26, 63),
    'micro_history':        (63, 98),
    'derived_physics':      (98, 107),
    'cluster_trends':       (107, 111),
}

VECTOR_DIMENSION = 111
```

### 6.9 Vector Construction Procedure

```
FUNCTION construct_episode_vector(
    current_bar: dict,           # All features at T=0
    history_buffer: list[dict],  # Last 5 bars (T-4 to T=0), from state table
    level_price: float
) -> array[111]:

    vector = zeros(111)
    idx = 0
    
    # ─── SECTION A: Context State (26 dims) ───
    vector[idx] = encode_level_kind(current_bar['level_kind'])
    idx += 1
    vector[idx] = encode_direction(current_bar['direction'])
    idx += 1
    vector[idx] = current_bar['minutes_since_open']
    idx += 1
    vector[idx] = current_bar['bars_since_open']
    idx += 1
    vector[idx] = current_bar['atr']
    idx += 1
    
    FOR f IN ['gex_asymmetry', 'gex_ratio', 'net_gex_2strike', 'gamma_exposure',
              'gex_above_1strike', 'gex_below_1strike', 'call_gex_above_2strike',
              'put_gex_below_2strike']:
        vector[idx] = current_bar[f]
        idx += 1
    
    vector[idx] = encode_fuel_effect(current_bar['fuel_effect'])
    idx += 1
    
    FOR f IN ['level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt']:
        vector[idx] = current_bar[f]
        idx += 1
    
    FOR f IN ['dist_to_pm_high_atr', 'dist_to_pm_low_atr', 'dist_to_or_high_atr',
              'dist_to_or_low_atr', 'dist_to_sma_200_atr', 'dist_to_sma_400_atr']:
        vector[idx] = current_bar[f] if current_bar[f] is not null else 0.0
        idx += 1
    
    vector[idx] = current_bar['prior_touches']
    idx += 1
    vector[idx] = current_bar['attempt_index']
    idx += 1
    vector[idx] = current_bar.get('attempt_cluster_id', 0) % 1000
    idx += 1
    
    # ─── SECTION B: Multi-Scale Trajectory (37 dims) ───
    FOR scale IN ['1min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar[f'velocity_{scale}']
        idx += 1
    
    FOR scale IN ['1min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar[f'acceleration_{scale}']
        idx += 1
    
    FOR scale IN ['1min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar[f'jerk_{scale}']
        idx += 1
    
    FOR scale IN ['3min', '5min', '10min', '20min']:
        vector[idx] = current_bar[f'momentum_trend_{scale}']
        idx += 1
    
    FOR scale IN ['30s', '60s', '120s', '300s']:
        vector[idx] = current_bar[f'ofi_{scale}']
        idx += 1
    
    FOR scale IN ['30s', '60s', '120s', '300s']:
        vector[idx] = current_bar[f'ofi_near_level_{scale}']
        idx += 1
    
    vector[idx] = current_bar['ofi_acceleration']
    idx += 1
    
    FOR scale IN ['1min', '3min', '5min']:
        vector[idx] = current_bar[f'barrier_delta_{scale}']
        idx += 1
    
    FOR scale IN ['1min', '3min', '5min']:
        vector[idx] = current_bar[f'barrier_pct_change_{scale}']
        idx += 1
    
    vector[idx] = current_bar['approach_velocity']
    idx += 1
    vector[idx] = current_bar['approach_bars']
    idx += 1
    vector[idx] = current_bar['approach_distance_atr']
    idx += 1
    
    # ─── SECTION C: Micro-History (35 dims) ───
    # Pad history if less than 5 bars
    WHILE len(history_buffer) < 5:
        history_buffer.insert(0, history_buffer[0] if len(history_buffer) > 0 else current_bar)
    
    history = history_buffer[-5:]  # Last 5 bars, oldest first
    
    MICRO_FEATURES = ['distance_signed_atr', 'tape_imbalance', 'tape_velocity',
                      'ofi_60s', 'barrier_delta_liq', 'wall_ratio', 'gamma_exposure']
    
    FOR feature IN MICRO_FEATURES:
        FOR bar IN history:
            vector[idx] = bar[feature]
            idx += 1
    
    # ─── SECTION D: Derived Physics (9 dims) ───
    vector[idx] = current_bar['predicted_accel']
    idx += 1
    vector[idx] = current_bar['accel_residual']
    idx += 1
    vector[idx] = current_bar['force_mass_ratio']
    idx += 1
    vector[idx] = encode_barrier_state(current_bar['barrier_state'])
    idx += 1
    vector[idx] = current_bar['barrier_depth_current']
    idx += 1
    vector[idx] = current_bar['barrier_replenishment_ratio']
    idx += 1
    vector[idx] = 1.0 if current_bar['sweep_detected'] else 0.0
    idx += 1
    
    tape_buy = current_bar['tape_buy_vol'] + 1
    tape_sell = current_bar['tape_sell_vol'] + 1
    vector[idx] = log(tape_buy / tape_sell)
    idx += 1
    vector[idx] = log(tape_buy + tape_sell)
    idx += 1
    
    # ─── SECTION E: Cluster Trends (4 dims) ───
    vector[idx] = current_bar['barrier_replenishment_trend']
    idx += 1
    vector[idx] = current_bar['barrier_delta_liq_trend']
    idx += 1
    vector[idx] = current_bar['tape_velocity_trend']
    idx += 1
    vector[idx] = current_bar['tape_imbalance_trend']
    idx += 1
    
    ASSERT idx == 111
    
    RETURN vector
```

### 6.10 Encoding Functions

```
FUNCTION encode_level_kind(level_kind: string) -> float:
    MAPPING = {
        'PM_HIGH': 0.0, 'PM_LOW': 1.0,
        'OR_HIGH': 2.0, 'OR_LOW': 3.0,
        'SMA_200': 4.0, 'SMA_400': 5.0
    }
    RETURN MAPPING.get(level_kind, -1.0)

FUNCTION encode_direction(direction: string) -> float:
    RETURN 1.0 if direction == 'UP' else -1.0

FUNCTION encode_fuel_effect(fuel_effect: string) -> float:
    MAPPING = {'AMPLIFY': 1.0, 'NEUTRAL': 0.0, 'DAMPEN': -1.0}
    RETURN MAPPING.get(fuel_effect, 0.0)

FUNCTION encode_barrier_state(barrier_state: string) -> float:
    MAPPING = {
        'STRONG_SUPPORT': 2.0, 'WEAK_SUPPORT': 1.0,
        'NEUTRAL': 0.0,
        'WEAK_RESISTANCE': -1.0, 'STRONG_RESISTANCE': -2.0
    }
    RETURN MAPPING.get(barrier_state, 0.0)
```

---

## 7. Normalization Specification ✅ COMPLETE

**Implementation**: `backend/src/ml/normalization.py`  
**Status**: Stage 17 module created  
**Features**:
- Computes robust/zscore/minmax statistics from 60 days of state data
- Classifies 111 features into appropriate normalization methods
- Saves versioned JSON (stats_v{N}.json) with symlink to current
- Provides normalize_value() and normalize_vector() functions
- Clips robust/zscore at ±4σ, minmax at [0,1]

### 7.1 Normalization Strategy

Different features require different normalization methods based on their distributions.

| Category | Method | Parameters | Features |
|----------|--------|------------|----------|
| **Robust** | (x - median) / IQR | clip ±4σ | Tape, OFI, barrier deltas, wall_ratio, force_mass_ratio, accel_residual |
| **Z-Score** | (x - mean) / std | clip ±4σ | Velocity, acceleration, jerk, momentum_trend, distance_signed_atr, predicted_accel |
| **MinMax** | (x - min) / (max - min) | [0, 1] | minutes_since_open, bars_since_open, level_stacking_*, prior_touches, attempt_index, approach_bars |
| **Passthrough** | No transformation | — | Encoded categoricals (level_kind, direction, fuel_effect, barrier_state, sweep_detected) |

### 7.2 Normalization Statistics Computation

```
FUNCTION compute_normalization_stats(
    historical_data: DataFrame,  # 60+ days of state table data
    feature_list: list[string]
) -> dict:

    stats = {}
    
    FOR feature IN feature_list:
        values = historical_data[feature].dropna()
        
        IF feature IN PASSTHROUGH_FEATURES:
            stats[feature] = {'method': 'passthrough'}
            CONTINUE
        
        IF feature IN ROBUST_FEATURES:
            median = values.median()
            q75 = values.quantile(0.75)
            q25 = values.quantile(0.25)
            iqr = q75 - q25
            stats[feature] = {
                'method': 'robust',
                'center': median,
                'scale': iqr if iqr > 1e-6 else 1.0
            }
        
        ELSE IF feature IN ZSCORE_FEATURES:
            stats[feature] = {
                'method': 'zscore',
                'center': values.mean(),
                'scale': values.std() if values.std() > 1e-6 else 1.0
            }
        
        ELSE IF feature IN MINMAX_FEATURES:
            stats[feature] = {
                'method': 'minmax',
                'min': values.min(),
                'max': values.max() if values.max() > values.min() else values.min() + 1.0
            }
        
        ELSE:
            # Default to robust
            median = values.median()
            iqr = values.quantile(0.75) - values.quantile(0.25)
            stats[feature] = {
                'method': 'robust',
                'center': median,
                'scale': iqr if iqr > 1e-6 else 1.0
            }
    
    RETURN stats
```

### 7.3 Feature Classification

```
PASSTHROUGH_FEATURES = {
    'level_kind', 'direction', 'fuel_effect', 'barrier_state', 'sweep_detected'
}

ROBUST_FEATURES = {
    'tape_velocity', 'tape_imbalance', 'tape_buy_vol', 'tape_sell_vol',
    'barrier_delta_liq', 'barrier_delta_1min', 'barrier_delta_3min', 'barrier_delta_5min',
    'ofi_30s', 'ofi_60s', 'ofi_120s', 'ofi_300s',
    'ofi_near_level_30s', 'ofi_near_level_60s', 'ofi_near_level_120s', 'ofi_near_level_300s',
    'wall_ratio', 'accel_residual', 'force_mass_ratio',
    'barrier_pct_change_1min', 'barrier_pct_change_3min', 'barrier_pct_change_5min',
    'tape_log_ratio', 'tape_log_total',
    'gex_asymmetry', 'gex_ratio', 'net_gex_2strike', 'gamma_exposure',
    'gex_above_1strike', 'gex_below_1strike', 'call_gex_above_2strike', 'put_gex_below_2strike',
    'barrier_depth_current', 'barrier_replenishment_ratio',
    'barrier_replenishment_trend', 'barrier_delta_liq_trend', 'tape_velocity_trend', 'tape_imbalance_trend'
}

ZSCORE_FEATURES = {
    'velocity_1min', 'velocity_3min', 'velocity_5min', 'velocity_10min', 'velocity_20min',
    'acceleration_1min', 'acceleration_3min', 'acceleration_5min', 'acceleration_10min', 'acceleration_20min',
    'jerk_1min', 'jerk_3min', 'jerk_5min', 'jerk_10min', 'jerk_20min',
    'momentum_trend_3min', 'momentum_trend_5min', 'momentum_trend_10min', 'momentum_trend_20min',
    'distance_signed_atr', 'approach_velocity', 'approach_distance_atr',
    'predicted_accel', 'atr',
    'dist_to_pm_high_atr', 'dist_to_pm_low_atr', 'dist_to_or_high_atr', 'dist_to_or_low_atr',
    'dist_to_sma_200_atr', 'dist_to_sma_400_atr'
}

MINMAX_FEATURES = {
    'minutes_since_open', 'bars_since_open',
    'level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt',
    'prior_touches', 'attempt_index', 'approach_bars', 'attempt_cluster_id_mod'
}
```

### 7.4 Apply Normalization

```
FUNCTION normalize_value(value: float, feature: string, stats: dict) -> float:
    stat = stats[feature]
    
    IF stat['method'] == 'passthrough':
        RETURN value
    
    IF stat['method'] == 'robust':
        normalized = (value - stat['center']) / stat['scale']
        RETURN clip(normalized, -4.0, 4.0)
    
    IF stat['method'] == 'zscore':
        normalized = (value - stat['center']) / stat['scale']
        RETURN clip(normalized, -4.0, 4.0)
    
    IF stat['method'] == 'minmax':
        normalized = (value - stat['min']) / (stat['max'] - stat['min'])
        RETURN clip(normalized, 0.0, 1.0)
```

### 7.5 Normalization Statistics Storage

```
Location: gold/normalization/stats_v{version}.json

Schema:
{
    "version": int,
    "computed_date": "YYYY-MM-DD",
    "lookback_days": 60,
    "n_samples": int,
    "features": {
        "velocity_1min": {
            "method": "zscore",
            "center": float,
            "scale": float
        },
        ...
    }
}
```

---

## 8. Index Architecture ✅ COMPLETE

**Implementation**: `backend/src/ml/index_builder.py`  
**Status**: Stage 19 created  
**Features**:
- Builds 48 partitions (6 levels × 2 directions × 4 time buckets)
- Auto-selects index type: Flat (<10K), IVF (10-100K), IVFPQ (>100K)
- L2-normalizes vectors for cosine similarity via inner product
- Saves index.faiss, vectors.npy, metadata.parquet per partition
- Skips partitions with < 100 vectors
- Outputs config.json with build statistics

### 8.1 Partitioning Strategy

Indices are partitioned by:
1. **level_kind**: {PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_200, SMA_400}
2. **direction**: {UP, DOWN}
3. **time_bucket**: {T0_30, T30_60, T60_120, T120_180}

Total partitions: 6 × 2 × 4 = 48 indices

### 8.2 Index Type Selection

| Corpus Size (per partition) | Index Type | Parameters |
|----------------------------|------------|------------|
| < 10,000 episodes | IndexFlatIP | Exact search |
| 10,000 - 100,000 | IndexIVFFlat | nlist = N/100, nprobe = 64 |
| > 100,000 | IndexIVFPQ | nlist = 4096, m = 8, nprobe = 64 |

### 8.3 Index Construction

```
FUNCTION build_index(
    vectors: array[N, 111],
    index_type: string
) -> FAISSIndex:

    N, D = vectors.shape
    
    # L2-normalize for cosine similarity via inner product
    vectors_normalized = vectors.copy()
    faiss.normalize_L2(vectors_normalized)
    
    IF index_type == 'Flat':
        index = faiss.IndexFlatIP(D)
        index.add(vectors_normalized)
    
    ELSE IF index_type == 'IVF':
        nlist = min(4096, max(16, N // 100))
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors_normalized)
        index.add(vectors_normalized)
        index.nprobe = min(64, nlist // 4)
    
    ELSE IF index_type == 'IVFPQ':
        nlist = min(4096, max(16, N // 100))
        m = 8  # subquantizers (D=111 is not divisible by 8, pad to 112 or use m=3)
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, m, 8)
        index.train(vectors_normalized)
        index.add(vectors_normalized)
        index.nprobe = 64
    
    RETURN index
```

### 8.4 Index File Structure

```
FOR EACH level_kind IN LEVEL_KINDS:
    FOR EACH direction IN ['UP', 'DOWN']:
        FOR EACH time_bucket IN TIME_BUCKETS:
            
            partition_path = f"gold/indices/es_level_indices/{level_kind}/{direction}/{time_bucket}/"
            
            # Filter corpus to this partition
            mask = (metadata['level_kind'] == level_kind) AND
                   (metadata['direction'] == direction) AND
                   (metadata['time_bucket'] == time_bucket)
            
            partition_vectors = corpus_vectors[mask]
            partition_metadata = corpus_metadata[mask]
            
            IF len(partition_vectors) < MIN_PARTITION_SIZE:
                SKIP  # Don't create index for tiny partitions
            
            # Build and save
            index = build_index(partition_vectors, select_index_type(len(partition_vectors)))
            
            faiss.write_index(index, partition_path + "index.faiss")
            save_npy(partition_vectors, partition_path + "vectors.npy")
            save_parquet(partition_metadata, partition_path + "metadata.parquet")
```

### 8.5 Index Manager

```
CLASS IndexManager:
    
    FUNCTION __init__(self, index_dir: string):
        self.index_dir = index_dir
        self.indices = {}      # {partition_key: FAISSIndex}
        self.metadata = {}     # {partition_key: DataFrame}
        self.vectors = {}      # {partition_key: ndarray}
    
    FUNCTION load_partition(self, level_kind, direction, time_bucket):
        key = f"{level_kind}/{direction}/{time_bucket}"
        path = f"{self.index_dir}/{key}/"
        
        IF path exists:
            self.indices[key] = faiss.read_index(path + "index.faiss")
            self.metadata[key] = read_parquet(path + "metadata.parquet")
            self.vectors[key] = load_npy(path + "vectors.npy")
    
    FUNCTION query(self, level_kind, direction, time_bucket, query_vector, k=50) -> dict:
        key = f"{level_kind}/{direction}/{time_bucket}"
        
        IF key NOT IN self.indices:
            self.load_partition(level_kind, direction, time_bucket)
        
        IF key NOT IN self.indices:
            RETURN {'similarities': [], 'indices': [], 'metadata': empty_dataframe()}
        
        # Normalize query vector
        query = query_vector.copy().reshape(1, -1).astype(float32)
        faiss.normalize_L2(query)
        
        # Search
        similarities, indices = self.indices[key].search(query, k)
        similarities = similarities[0]
        indices = indices[0]
        
        # Filter invalid
        valid_mask = indices >= 0
        similarities = similarities[valid_mask]
        indices = indices[valid_mask]
        
        # Get metadata
        retrieved_metadata = self.metadata[key].iloc[indices].copy()
        retrieved_metadata['similarity'] = similarities
        
        # Get vectors for attribution
        retrieved_vectors = self.vectors[key][indices] if key in self.vectors else None
        
        RETURN {
            'similarities': similarities,
            'indices': indices,
            'metadata': retrieved_metadata,
            'vectors': retrieved_vectors
        }
```

---

## 9. Retrieval Pipeline ✅ COMPLETE

**Implementation**: `backend/src/ml/retrieval_engine.py`  
**Status**: Complete with 4 main classes  
**Components**:
- **IndexManager**: Lazy-loads and caches FAISS indices per partition
- **LiveEpisodeBuilder**: Builds query vectors from real-time state (5-bar buffers)
- **SimilarityQueryEngine**: Executes queries and returns results
- **RealTimeQueryService**: Main service with caching (30s TTL) and quality filtering

### 9.1 Live Episode Builder

```
CLASS LiveEpisodeBuilder:
    
    FUNCTION __init__(self, normalizer_stats: dict, state_table_cadence_seconds=30):
        self.normalizer_stats = normalizer_stats
        self.cadence = state_table_cadence_seconds
        self.buffers = {}  # {(level_kind, level_price): deque of bars}
        self.buffer_size = 5
    
    FUNCTION on_state_update(self, state_row: dict) -> list[EpisodeQuery]:
        """
        Called every cadence interval with new state table row.
        Returns list of episode queries ready for retrieval.
        """
        queries = []
        
        level_kind = state_row['level_kind']
        level_price = state_row['level_price']
        level_key = (level_kind, level_price)
        
        # Initialize buffer if needed
        IF level_key NOT IN self.buffers:
            self.buffers[level_key] = deque(maxlen=self.buffer_size)
        
        # Add to buffer
        self.buffers[level_key].append(state_row)
        
        # Check if in approach zone
        distance_atr = abs(state_row['distance_signed_atr'])
        in_zone = distance_atr < ZONE_THRESHOLD_ATR  # 3.0
        
        # Check approach velocity
        approach_velocity = abs(state_row.get('approach_velocity', 0))
        has_velocity = approach_velocity > MIN_APPROACH_VELOCITY  # 0.5
        
        # Emit query if conditions met
        IF len(self.buffers[level_key]) >= self.buffer_size AND in_zone AND has_velocity:
            
            # Build vector
            vector = construct_episode_vector(
                current_bar=state_row,
                history_buffer=list(self.buffers[level_key]),
                level_price=level_price
            )
            
            # Normalize vector
            normalized_vector = self.normalize_vector(vector)
            
            # Determine direction
            direction = 'UP' if state_row['spot'] < level_price else 'DOWN'
            
            # Assign time bucket
            time_bucket = assign_time_bucket(state_row['minutes_since_open'])
            
            # Compute emission weight
            emission_weight = compute_emission_weight(
                spot=state_row['spot'],
                level_price=level_price,
                atr=state_row['atr'],
                approach_velocity=state_row['approach_velocity'],
                ofi_60s=state_row['ofi_60s']
            )
            
            queries.append(EpisodeQuery(
                level_kind=level_kind,
                level_price=level_price,
                direction=direction,
                time_bucket=time_bucket,
                vector=normalized_vector,
                emission_weight=emission_weight,
                timestamp=state_row['timestamp'],
                metadata={
                    'spot': state_row['spot'],
                    'atr': state_row['atr'],
                    'minutes_since_open': state_row['minutes_since_open']
                }
            ))
        
        RETURN queries
    
    FUNCTION normalize_vector(self, raw_vector: array) -> array:
        normalized = zeros(VECTOR_DIMENSION)
        
        FOR idx, feature_name IN enumerate(FEATURE_NAMES_ORDERED):
            normalized[idx] = normalize_value(
                raw_vector[idx], 
                feature_name, 
                self.normalizer_stats
            )
        
        RETURN normalized
```

### 9.2 Query Engine

```
CLASS SimilarityQueryEngine:
    
    FUNCTION __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self.k_retrieve = 100  # Over-fetch for filtering
        self.k_return = 50     # Final neighbors
    
    FUNCTION query(self, episode_query: EpisodeQuery, filters: dict = None) -> QueryResult:
        
        # Retrieve from appropriate partition
        result = self.index_manager.query(
            level_kind=episode_query.level_kind,
            direction=episode_query.direction,
            time_bucket=episode_query.time_bucket,
            query_vector=episode_query.vector,
            k=self.k_retrieve
        )
        
        retrieved_metadata = result['metadata']
        retrieved_vectors = result['vectors']
        
        IF len(retrieved_metadata) == 0:
            RETURN self.empty_result(episode_query)
        
        # Apply filters
        IF filters:
            mask = ones(len(retrieved_metadata), dtype=bool)
            FOR key, value IN filters.items():
                IF key IN retrieved_metadata.columns:
                    mask &= (retrieved_metadata[key] == value)
            retrieved_metadata = retrieved_metadata[mask]
            IF retrieved_vectors is not None:
                retrieved_vectors = retrieved_vectors[mask]
        
        # Take top k_return
        retrieved_metadata = retrieved_metadata.head(self.k_return)
        IF retrieved_vectors is not None:
            retrieved_vectors = retrieved_vectors[:self.k_return]
        
        IF len(retrieved_metadata) == 0:
            RETURN self.empty_result(episode_query)
        
        # Compute outcome distributions
        outcome_dist = compute_outcome_distribution(retrieved_metadata)
        confidence_intervals = compute_bootstrap_ci(retrieved_metadata)
        multi_horizon = compute_multi_horizon_distribution(retrieved_metadata)
        
        # Compute attribution
        IF retrieved_vectors is not None:
            attribution = compute_attribution(
                query_vector=episode_query.vector,
                retrieved_vectors=retrieved_vectors,
                outcomes=retrieved_metadata['outcome_4min'].values
            )
        ELSE:
            attribution = {}
        
        # Compute reliability metrics
        reliability = compute_reliability(retrieved_metadata)
        
        RETURN QueryResult(
            outcome_probabilities=outcome_dist,
            confidence_intervals=confidence_intervals,
            multi_horizon=multi_horizon,
            attribution=attribution,
            reliability=reliability,
            neighbors=retrieved_metadata.to_dict('records'),
            query_metadata={
                'level_kind': episode_query.level_kind,
                'direction': episode_query.direction,
                'time_bucket': episode_query.time_bucket,
                'timestamp': episode_query.timestamp,
                'emission_weight': episode_query.emission_weight
            }
        )
```

### 9.3 Real-Time Service

```
CLASS RealTimeQueryService:
    
    FUNCTION __init__(self, normalizer_stats, index_manager):
        self.episode_builder = LiveEpisodeBuilder(normalizer_stats)
        self.query_engine = SimilarityQueryEngine(index_manager)
        self.result_cache = {}  # {level_key: (timestamp, result)}
        self.cache_ttl_seconds = 30
        self.min_similarity_threshold = 0.70
        self.min_samples_threshold = 30
    
    FUNCTION process_state_update(self, state_row: dict) -> list[QueryResult]:
        """
        Main entry point: process new state, return any query results.
        """
        results = []
        
        # Build episode queries
        queries = self.episode_builder.on_state_update(state_row)
        
        FOR query IN queries:
            level_key = (query.level_kind, query.level_price)
            
            # Check cache
            IF self.is_cached(level_key):
                CONTINUE
            
            # Execute query
            result = self.query_engine.query(query)
            
            # Filter low-quality results
            IF self.is_quality_result(result):
                results.append(result)
                self.cache_result(level_key, result)
        
        RETURN results
    
    FUNCTION is_quality_result(self, result: QueryResult) -> bool:
        RETURN (
            result.reliability['avg_similarity'] >= self.min_similarity_threshold AND
            result.reliability['n_retrieved'] >= self.min_samples_threshold
        )
```

---

## 10. Outcome Aggregation ✅ COMPLETE

**Implementation**: `backend/src/ml/outcome_aggregation.py`  
**Status**: Complete with 6 functions + integrated into retrieval engine  
**Functions**:
- `compute_outcome_distribution()`: Similarity-weighted probabilities
- `compute_expected_excursions()`: Expected favorable/adverse excursions
- `compute_conditional_excursions()`: Excursions conditional on outcome
- `compute_multi_horizon_distribution()`: Outcomes for 2min/4min/8min
- `compute_bootstrap_ci()`: Bootstrap confidence intervals (1000 samples)
- `compute_reliability()`: Effective N, entropy, similarity stats

### 10.1 Similarity-Weighted Distribution

```
FUNCTION compute_outcome_distribution(retrieved_metadata: DataFrame) -> dict:
    """
    Compute outcome probabilities weighted by similarity.
    """
    IF len(retrieved_metadata) == 0:
        RETURN {'BREAK': 0, 'REJECT': 0, 'CHOP': 0, 'n_samples': 0}
    
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    probs = {}
    FOR outcome IN ['BREAK', 'REJECT', 'CHOP']:
        mask = retrieved_metadata['outcome_4min'] == outcome  # Primary horizon
        probs[outcome] = weights[mask].sum()
    
    RETURN {
        'probabilities': probs,
        'n_samples': len(retrieved_metadata),
        'avg_similarity': retrieved_metadata['similarity'].mean()
    }
```

### 10.2 Expected Excursions

```
FUNCTION compute_expected_excursions(retrieved_metadata: DataFrame) -> dict:
    """
    Compute expected favorable and adverse excursions.
    """
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()
    
    expected_favorable = (weights * retrieved_metadata['excursion_favorable']).sum()
    expected_adverse = (weights * retrieved_metadata['excursion_adverse']).sum()
    
    RETURN {
        'expected_excursion_favorable': expected_favorable,
        'expected_excursion_adverse': expected_adverse,
        'excursion_ratio': expected_favorable / (expected_adverse + 1e-6)
    }
```

### 10.3 Conditional Excursions

```
FUNCTION compute_conditional_excursions(retrieved_metadata: DataFrame) -> dict:
    """
    Compute expected excursions conditional on outcome.
    """
    weights = retrieved_metadata['similarity'].values
    
    conditional = {}
    FOR outcome IN ['BREAK', 'REJECT']:
        mask = retrieved_metadata['outcome_4min'] == outcome
        IF mask.sum() > 0:
            subset = retrieved_metadata[mask]
            subset_weights = weights[mask]
            subset_weights = subset_weights / subset_weights.sum()
            
            conditional[outcome] = {
                'expected_favorable': (subset_weights * subset['excursion_favorable']).sum(),
                'expected_adverse': (subset_weights * subset['excursion_adverse']).sum(),
                'mean_strength': (subset_weights * subset['strength_abs']).sum(),
                'n_samples': mask.sum()
            }
    
    RETURN conditional
```

### 10.4 Multi-Horizon Distribution

```
FUNCTION compute_multi_horizon_distribution(retrieved_metadata: DataFrame) -> dict:
    """
    Compute outcome distributions for all horizons.
    """
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()
    
    horizons = {
        '2min': 'outcome_2min',
        '4min': 'outcome_4min',
        '8min': 'outcome_8min'
    }
    
    results = {}
    FOR horizon_name, col IN horizons.items():
        IF col NOT IN retrieved_metadata.columns:
            CONTINUE
        
        probs = {}
        FOR outcome IN ['BREAK', 'REJECT', 'CHOP']:
            mask = retrieved_metadata[col] == outcome
            probs[outcome] = weights[mask].sum()
        
        results[horizon_name] = {
            'probabilities': probs,
            'n_valid': retrieved_metadata[col].notna().sum()
        }
    
    RETURN results
```

### 10.5 Bootstrap Confidence Intervals

```
FUNCTION compute_bootstrap_ci(
    retrieved_metadata: DataFrame,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> dict:
    """
    Compute bootstrap confidence intervals for outcome probabilities.
    """
    IF len(retrieved_metadata) < 5:
        RETURN {outcome: {'mean': 0, 'ci_low': 0, 'ci_high': 1} 
                FOR outcome IN ['BREAK', 'REJECT', 'CHOP']}
    
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()
    outcomes = retrieved_metadata['outcome_4min'].values
    
    boot_probs = {'BREAK': [], 'REJECT': [], 'CHOP': []}
    
    FOR _ IN range(n_bootstrap):
        # Weighted bootstrap sample
        sample_idx = random.choice(
            len(outcomes),
            size=len(outcomes),
            replace=True,
            p=weights
        )
        sample_outcomes = outcomes[sample_idx]
        
        # Compute proportions
        FOR outcome IN boot_probs:
            prop = (sample_outcomes == outcome).mean()
            boot_probs[outcome].append(prop)
    
    ci = {}
    FOR outcome, probs IN boot_probs.items():
        probs = array(probs)
        ci[outcome] = {
            'mean': probs.mean(),
            'ci_low': percentile(probs, 100 * alpha / 2),
            'ci_high': percentile(probs, 100 * (1 - alpha / 2)),
            'std': probs.std()
        }
    
    RETURN ci
```

### 10.6 Reliability Metrics

```
FUNCTION compute_reliability(retrieved_metadata: DataFrame) -> dict:
    """
    Compute reliability metrics for the retrieval.
    """
    similarities = retrieved_metadata['similarity'].values
    weights = similarities / similarities.sum()
    
    # Effective sample size
    effective_n = (weights.sum() ** 2) / (weights ** 2).sum()
    
    RETURN {
        'n_retrieved': len(retrieved_metadata),
        'effective_n': effective_n,
        'avg_similarity': similarities.mean(),
        'min_similarity': similarities.min(),
        'max_similarity': similarities.max(),
        'similarity_std': similarities.std(),
        'entropy': -sum(weights * log(weights + 1e-10))
    }
```

---

## 11. Attribution System ✅ COMPLETE

**Implementation**: `backend/src/ml/attribution.py`  
**Status**: Complete with 4 attribution methods + integrated into retrieval engine  
**Methods**:
- **Similarity Attribution**: Which features drove neighbor selection (inner product contributions)
- **Outcome Attribution**: Which features differentiate BREAK vs REJECT (weighted logistic regression)
- **Section Attribution**: Which vector sections lean toward each outcome (centroid distances)
- **Physics Attribution**: Aggregate coefficients into physics buckets (kinematics, order flow, liquidity, gamma, context)

### 11.1 Similarity Attribution

```
FUNCTION compute_similarity_attribution(
    query_vector: array,
    retrieved_vectors: array,
    similarities: array,
    feature_names: list[string]
) -> dict:
    """
    Explain which features drove similarity for each neighbor.
    Returns aggregate attribution across all neighbors.
    """
    n_neighbors = len(retrieved_vectors)
    n_features = len(feature_names)
    
    # Per-feature contribution to similarity
    # For L2-normalized vectors using inner product:
    # similarity = sum(q_i * r_i)
    # contribution of feature i = q_i * r_i
    
    query_norm = query_vector / (norm(query_vector) + 1e-10)
    
    feature_contributions = zeros(n_features)
    
    FOR i IN range(n_neighbors):
        neighbor_norm = retrieved_vectors[i] / (norm(retrieved_vectors[i]) + 1e-10)
        
        # Element-wise contribution
        contributions = query_norm * neighbor_norm
        
        # Weight by similarity
        feature_contributions += similarities[i] * contributions
    
    # Normalize
    feature_contributions = feature_contributions / similarities.sum()
    
    # Create sorted list
    attribution_list = list(zip(feature_names, feature_contributions))
    attribution_list.sort(key=lambda x: abs(x[1]), reverse=True)
    
    RETURN {
        'top_matching_features': attribution_list[:10],
        'all_attributions': dict(attribution_list)
    }
```

### 11.2 Outcome Attribution (Local Surrogate Model)

```
FUNCTION compute_outcome_attribution(
    query_vector: array,
    retrieved_vectors: array,
    outcomes: array,
    similarities: array,
    feature_names: list[string]
) -> dict:
    """
    Explain which features differentiate BREAK vs REJECT in the neighborhood.
    Uses weighted logistic regression as local surrogate.
    """
    # Filter to BREAK and REJECT only
    mask = (outcomes == 'BREAK') | (outcomes == 'REJECT')
    IF mask.sum() < 10:
        RETURN {'top_break_drivers': [], 'top_reject_drivers': []}
    
    X = retrieved_vectors[mask]
    y = (outcomes[mask] == 'BREAK').astype(int)
    weights = similarities[mask]
    
    # Fit weighted logistic regression
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    TRY:
        model.fit(X, y, sample_weight=weights)
    EXCEPT:
        RETURN {'top_break_drivers': [], 'top_reject_drivers': []}
    
    # Extract coefficients
    coefficients = model.coef_[0]
    
    # Pair with feature names
    feature_importance = list(zip(feature_names, coefficients))
    
    # Positive coefficients favor BREAK
    break_drivers = [(f, c) FOR f, c IN feature_importance IF c > 0]
    break_drivers.sort(key=lambda x: x[1], reverse=True)
    
    # Negative coefficients favor REJECT
    reject_drivers = [(f, abs(c)) FOR f, c IN feature_importance IF c < 0]
    reject_drivers.sort(key=lambda x: x[1], reverse=True)
    
    RETURN {
        'top_break_drivers': break_drivers[:10],
        'top_reject_drivers': reject_drivers[:10],
        'model_accuracy': model.score(X, y, sample_weight=weights),
        'all_coefficients': dict(feature_importance)
    }
```

### 11.3 Section-Level Attribution

```
FUNCTION compute_section_attribution(
    query_vector: array,
    retrieved_vectors: array,
    outcomes: array
) -> dict:
    """
    Identify which vector sections differentiate BREAK vs REJECT.
    """
    break_vectors = retrieved_vectors[outcomes == 'BREAK']
    reject_vectors = retrieved_vectors[outcomes == 'REJECT']
    
    IF len(break_vectors) == 0 OR len(reject_vectors) == 0:
        RETURN {}
    
    section_analysis = {}
    
    FOR section_name, (start, end) IN VECTOR_SECTIONS.items():
        # Compute centroids
        break_centroid = break_vectors[:, start:end].mean(axis=0)
        reject_centroid = reject_vectors[:, start:end].mean(axis=0)
        
        # Distance from query to each centroid
        query_section = query_vector[start:end]
        dist_to_break = norm(query_section - break_centroid)
        dist_to_reject = norm(query_section - reject_centroid)
        
        section_analysis[section_name] = {
            'dist_to_break': dist_to_break,
            'dist_to_reject': dist_to_reject,
            'lean': 'BREAK' if dist_to_break < dist_to_reject else 'REJECT',
            'confidence': abs(dist_to_reject - dist_to_break) / (dist_to_break + dist_to_reject + 1e-6)
        }
    
    RETURN section_analysis
```

### 11.4 Physics-Bucket Attribution

```
PHYSICS_BUCKETS = {
    'kinematics': [
        'velocity_*', 'acceleration_*', 'jerk_*', 'momentum_trend_*',
        'approach_velocity', 'approach_bars', 'approach_distance_atr',
        'predicted_accel', 'accel_residual', 'force_mass_ratio'
    ],
    'order_flow': [
        'ofi_*', 'tape_imbalance', 'tape_velocity', 'sweep_detected',
        'tape_log_ratio', 'tape_log_total'
    ],
    'liquidity_barrier': [
        'barrier_*', 'wall_ratio'
    ],
    'dealer_gamma': [
        'gamma_exposure', 'fuel_effect', 'gex_*', 'net_gex_2strike'
    ],
    'context': [
        'level_kind', 'direction', 'minutes_since_open', 'level_stacking_*',
        'dist_to_*', 'prior_touches', 'attempt_index', 'atr'
    ]
}

FUNCTION compute_physics_attribution(all_coefficients: dict) -> dict:
    """
    Aggregate feature attributions into physics buckets.
    """
    bucket_scores = {bucket: 0.0 FOR bucket IN PHYSICS_BUCKETS}
    
    FOR feature, coef IN all_coefficients.items():
        FOR bucket, patterns IN PHYSICS_BUCKETS.items():
            IF feature_matches_any_pattern(feature, patterns):
                bucket_scores[bucket] += abs(coef)
                BREAK
    
    # Normalize to sum to 1
    total = sum(bucket_scores.values())
    IF total > 0:
        bucket_scores = {k: v/total FOR k, v IN bucket_scores.items()}
    
    RETURN bucket_scores
```

---

## 12. Validation Framework ✅ COMPLETE

**Implementation**: `backend/src/ml/validation.py`  
**Status**: Complete with 6 validation methods + ValidationRunner class  
**Methods**:
- **Temporal CV**: Expanding window splits, no leakage
- **Retrieval Metrics**: AUC, Brier score, log loss
- **Calibration Curve**: Reliability diagram + ECE
- **Lift Analysis**: Break rate vs threshold
- **Sanity Check**: Same-outcome neighbors closer than different-outcome
- **Drift Detection**: Wasserstein distance + mean shift per feature

### 12.1 Temporal Cross-Validation

```
FUNCTION temporal_cv_split(
    dates: list[date],
    n_splits: int = 5,
    min_train_days: int = 60
) -> list[tuple[list[date], list[date]]]:
    """
    Generate temporal train/test splits.
    Training always precedes test to prevent leakage.
    """
    dates = sorted(dates)
    n_dates = len(dates)
    
    test_size = (n_dates - min_train_days) // n_splits
    
    splits = []
    FOR i IN range(n_splits):
        train_end = min_train_days + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]
        
        splits.append((train_dates, test_dates))
    
    RETURN splits
```

### 12.2 Retrieval Quality Metrics

```
FUNCTION evaluate_retrieval_system(
    query_engine: SimilarityQueryEngine,
    test_episodes: array,
    test_metadata: DataFrame
) -> dict:
    """
    Comprehensive evaluation of retrieval system.
    """
    predictions = []
    actuals = []
    
    FOR idx IN range(len(test_episodes)):
        episode_vector = test_episodes[idx]
        meta = test_metadata.iloc[idx]
        
        result = query_engine.query(EpisodeQuery(
            level_kind=meta['level_kind'],
            direction=meta['direction'],
            time_bucket=meta['time_bucket'],
            vector=episode_vector,
            ...
        ))
        
        IF result.reliability['n_retrieved'] < 10:
            CONTINUE
        
        pred_break = result.outcome_probabilities['probabilities']['BREAK']
        predictions.append(pred_break)
        actuals.append(1 if meta['outcome_4min'] == 'BREAK' else 0)
    
    predictions = array(predictions)
    actuals = array(actuals)
    
    metrics = {
        'n_evaluated': len(predictions),
        'base_rate_break': actuals.mean(),
        'mean_prediction': predictions.mean(),
        'auc': roc_auc_score(actuals, predictions) if len(unique(actuals)) > 1 else None,
        'brier_score': brier_score_loss(actuals, predictions),
        'log_loss': log_loss(actuals, clip(predictions, 0.01, 0.99))
    }
    
    # Calibration curve
    metrics['calibration'] = compute_calibration_curve(actuals, predictions, n_bins=10)
    
    # Lift analysis
    metrics['lift'] = compute_lift_analysis(actuals, predictions)
    
    RETURN metrics
```

### 12.3 Calibration Curve

```
FUNCTION compute_calibration_curve(
    actuals: array,
    predictions: array,
    n_bins: int = 10
) -> dict:
    """
    Compute reliability diagram data.
    """
    bin_edges = linspace(0, 1, n_bins + 1)
    
    bin_means = []
    bin_true_freqs = []
    bin_counts = []
    
    FOR i IN range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        IF mask.sum() > 0:
            bin_means.append(predictions[mask].mean())
            bin_true_freqs.append(actuals[mask].mean())
            bin_counts.append(mask.sum())
    
    # Calibration error
    ece = sum(
        (count / len(predictions)) * abs(mean - freq)
        FOR mean, freq, count IN zip(bin_means, bin_true_freqs, bin_counts)
    )
    
    RETURN {
        'bin_means': bin_means,
        'bin_true_freqs': bin_true_freqs,
        'bin_counts': bin_counts,
        'expected_calibration_error': ece
    }
```

### 12.4 Lift Analysis

```
FUNCTION compute_lift_analysis(
    actuals: array,
    predictions: array
) -> dict:
    """
    Compute lift at various confidence thresholds.
    """
    base_rate = actuals.mean()
    
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    lift_results = []
    
    FOR threshold IN thresholds:
        high_conf_mask = predictions >= threshold
        IF high_conf_mask.sum() > 0:
            high_conf_rate = actuals[high_conf_mask].mean()
            lift = high_conf_rate / base_rate
            lift_results.append({
                'threshold': threshold,
                'n_samples': high_conf_mask.sum(),
                'break_rate': high_conf_rate,
                'lift': lift
            })
    
    RETURN {
        'base_rate': base_rate,
        'lift_by_threshold': lift_results
    }
```

### 12.5 Similarity Sanity Check

```
FUNCTION sanity_check_similarity(
    episode_vectors: array,
    metadata: DataFrame,
    n_samples: int = 500
) -> dict:
    """
    Verify that similar episodes have similar outcomes.
    Same-outcome neighbors should be closer than different-outcome neighbors.
    """
    sample_idx = random.choice(len(episode_vectors), min(n_samples, len(episode_vectors)), replace=False)
    
    same_outcome_distances = []
    diff_outcome_distances = []
    
    # Normalize for cosine distance
    vectors_norm = episode_vectors / (norm(episode_vectors, axis=1, keepdims=True) + 1e-6)
    
    FOR i IN sample_idx:
        # Cosine distance to all others
        similarities = dot(vectors_norm, vectors_norm[i])
        distances = 1 - similarities
        distances[i] = inf  # Exclude self
        
        # Find nearest neighbor
        nearest = argmin(distances)
        
        IF metadata.iloc[i]['outcome_4min'] == metadata.iloc[nearest]['outcome_4min']:
            same_outcome_distances.append(distances[nearest])
        ELSE:
            diff_outcome_distances.append(distances[nearest])
    
    same_mean = mean(same_outcome_distances) if same_outcome_distances else 0
    diff_mean = mean(diff_outcome_distances) if diff_outcome_distances else 0
    
    RETURN {
        'same_outcome_mean_dist': same_mean,
        'diff_outcome_mean_dist': diff_mean,
        'separation_ratio': diff_mean / (same_mean + 1e-6),
        'interpretation': 'GOOD' if diff_mean > same_mean else 'POOR'
    }
```

### 12.6 Feature Drift Detection

```
FUNCTION detect_feature_drift(
    corpus_vectors: array,
    corpus_dates: array,
    lookback_days: int = 60,
    recent_days: int = 5
) -> DataFrame:
    """
    Detect drift in feature distributions.
    """
    cutoff = today() - timedelta(days=recent_days)
    historical_end = cutoff - timedelta(days=lookback_days)
    
    historical_mask = corpus_dates < historical_end
    recent_mask = corpus_dates >= cutoff
    
    drift_metrics = []
    
    FOR feat_idx, feat_name IN enumerate(FEATURE_NAMES_ORDERED):
        hist_vals = corpus_vectors[historical_mask, feat_idx]
        recent_vals = corpus_vectors[recent_mask, feat_idx]
        
        IF len(hist_vals) == 0 OR len(recent_vals) == 0:
            CONTINUE
        
        # Wasserstein distance
        w_dist = wasserstein_distance(hist_vals, recent_vals)
        
        # Mean shift in std units
        mean_shift = (recent_vals.mean() - hist_vals.mean()) / (hist_vals.std() + 1e-6)
        
        drift_metrics.append({
            'feature': feat_name,
            'wasserstein_distance': w_dist,
            'mean_shift_std': mean_shift,
            'hist_mean': hist_vals.mean(),
            'recent_mean': recent_vals.mean()
        })
    
    RETURN DataFrame(drift_metrics).sort_values('wasserstein_distance', ascending=False)
```

---

## 13. Pipeline Integration ✅ COMPLETE

**Implementation**: Updated `backend/src/pipeline/pipelines/es_pipeline.py`  
**Status**: Pipeline updated to v3.0.0 with new stages  
**Changes**:
- Added Stage 16: MaterializeStateTableStage (30s cadence state)
- Added Stage 18: ConstructEpisodesStage (111-dim episode vectors)
- Updated to use first-crossing outcome labels (BREAK/REJECT/CHOP)
- Pipeline now generates episode corpus ready for index building

**Full Pipeline** (18 stages, 0-indexed):

| Index | Stage Name | Status | Description |
|-------|------------|--------|-------------|
| 0 | LoadBronze | Existing | Load ES futures + options from Bronze |
| 1 | BuildOHLCV (1min) | Existing | 1-minute OHLCV for ATR |
| 2 | BuildOHLCV (2min) | Existing | 2-minute OHLCV for SMA |
| 3 | InitMarketState | Existing | Market state + Greeks |
| 4 | GenerateLevels | Existing | 6 level kinds |
| 5 | DetectInteractionZones | Existing | Event detection |
| 6 | ComputePhysics | Existing | Barrier/Tape/Fuel |
| 7 | ComputeMultiWindowKinematics | Existing | Velocity/Accel/Jerk |
| 8 | ComputeMultiWindowOFI | Existing | Multi-window OFI |
| 9 | ComputeBarrierEvolution | Existing | Barrier depth changes |
| 10 | ComputeLevelDistances | Existing | Distance to all levels |
| 11 | ComputeGEXFeatures | Existing | Gamma exposure features |
| 12 | ComputeForceMass | Existing | F=ma validation |
| 13 | ComputeApproach | Existing | Approach dynamics |
| 14 | LabelOutcomes | **Updated** | **First-crossing (BREAK/REJECT/CHOP)** |
| 15 | FilterRTH | Existing | RTH 09:30-12:30 filter |
| 16 | MaterializeStateTable | **NEW** | **30s cadence state** |
| 17 | ConstructEpisodes | **NEW** | **111-dim episode vectors** |

**Offline Stages** (not in main pipeline):
- Stage 17 (offline): Normalization Stats Computation
- Stage 19 (offline): Index Building (48 FAISS partitions)
- Stage 20 (offline): Validation (weekly)

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
STAGE 17: Normalization Statistics (Daily, expanding window)
    │
    └──► gold/normalization/stats_v{N}.json
    
    │
    ▼
STAGE 18: Episode Vector Construction
    │
    └──► gold/episodes/es_level_episodes/vectors/*.npy
    └──► gold/episodes/es_level_episodes/metadata/*.parquet
    
    │
    ▼
STAGE 19: Index Building (Daily, after RTH close)
    │
    └──► gold/indices/es_level_indices/{level}/{dir}/{bucket}/*
    
    │
    ▼
STAGE 20: Validation (Weekly)
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

**Stage 17: Normalization Statistics**

```
Input:  silver/state/es_level_state/ (last 60 days)
Output: gold/normalization/stats_v{N}.json

Logic:
1. Load 60 days of state table
2. For each feature, compute statistics per Section 7
3. Save JSON with version number
4. Archive previous version

Schedule: Daily at 05:00 ET (before market open)
```

**Stage 18: Episode Vector Construction**

```
Input:  silver/features/es_pipeline/date=YYYY-MM-DD/*.parquet (event anchors)
        silver/state/es_level_state/date=YYYY-MM-DD/*.parquet (state for windows)
        gold/normalization/stats_v{N}.json
Output: gold/episodes/es_level_episodes/vectors/date=YYYY-MM-DD/episodes.npy
        gold/episodes/es_level_episodes/metadata/date=YYYY-MM-DD/metadata.parquet

Logic:
1. For each event (anchor) in event table:
   a. Get anchor timestamp t0
   b. Extract 5-bar history from state table (t0-4*30s to t0)
   c. Construct raw vector (Section 6.9)
   d. Normalize vector (Section 7)
   e. Compute labels (Section 3)
   f. Compute emission_weight (Section 5.3)
   g. Store vector and metadata
2. Partition by date

Schedule: Daily, after RTH close (16:15 ET)
```

**Stage 19: Index Building**

```
Input:  gold/episodes/es_level_episodes/ (all dates)
Output: gold/indices/es_level_indices/{level}/{dir}/{bucket}/

Logic:
1. Load all episode vectors and metadata
2. For each partition (level_kind, direction, time_bucket):
   a. Filter to partition
   b. Select index type based on corpus size
   c. Build FAISS index
   d. Save index, vectors, metadata
3. Update config.json with build timestamp

Schedule: Daily at 17:00 ET

Incremental Option:
- Append new vectors to existing index
- Full rebuild weekly (Sunday)
```

**Stage 20: Validation**

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
    │       ├── PM_HIGH/UP/T0_30/
    │       ├── PM_HIGH/UP/T30_60/
    │       ├── ... (48 partitions total)
    │       └── config.json
    │
    └── validation/
        ├── calibration_YYYY-MM-DD.json
        ├── drift_YYYY-MM-DD.json
        └── sanity_YYYY-MM-DD.json
```

---

## Appendix A: Complete Feature Specification

### A.1 Section A: Context State (26 features)

| Index | Feature | Source | Normalization |
|-------|---------|--------|---------------|
| 0 | level_kind_encoded | level_kind | Passthrough |
| 1 | direction_encoded | direction | Passthrough |
| 2 | minutes_since_open | minutes_since_open | MinMax [0, 180] |
| 3 | bars_since_open | bars_since_open | MinMax [0, 90] |
| 4 | atr | atr | Z-Score |
| 5 | gex_asymmetry | gex_asymmetry | Robust |
| 6 | gex_ratio | gex_ratio | Robust |
| 7 | net_gex_2strike | net_gex_2strike | Robust |
| 8 | gamma_exposure | gamma_exposure | Robust |
| 9 | gex_above_1strike | gex_above_1strike | Robust |
| 10 | gex_below_1strike | gex_below_1strike | Robust |
| 11 | call_gex_above_2strike | call_gex_above_2strike | Robust |
| 12 | put_gex_below_2strike | put_gex_below_2strike | Robust |
| 13 | fuel_effect_encoded | fuel_effect | Passthrough |
| 14 | level_stacking_2pt | level_stacking_2pt | MinMax [0, 6] |
| 15 | level_stacking_5pt | level_stacking_5pt | MinMax [0, 6] |
| 16 | level_stacking_10pt | level_stacking_10pt | MinMax [0, 6] |
| 17 | dist_to_pm_high_atr | dist_to_pm_high_atr | Z-Score |
| 18 | dist_to_pm_low_atr | dist_to_pm_low_atr | Z-Score |
| 19 | dist_to_or_high_atr | dist_to_or_high_atr | Z-Score |
| 20 | dist_to_or_low_atr | dist_to_or_low_atr | Z-Score |
| 21 | dist_to_sma_200_atr | dist_to_sma_200_atr | Z-Score |
| 22 | dist_to_sma_400_atr | dist_to_sma_400_atr | Z-Score |
| 23 | prior_touches | prior_touches | MinMax [0, 10] |
| 24 | attempt_index | attempt_index | MinMax [0, 10] |
| 25 | attempt_cluster_id_mod | attempt_cluster_id % 1000 | MinMax [0, 1000] |

### A.2 Section B: Multi-Scale Trajectory (37 features)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 26-30 | velocity_{1,3,5,10,20}min | Z-Score |
| 31-35 | acceleration_{1,3,5,10,20}min | Z-Score |
| 36-40 | jerk_{1,3,5,10,20}min | Z-Score |
| 41-44 | momentum_trend_{3,5,10,20}min | Z-Score |
| 45-48 | ofi_{30s,60s,120s,300s} | Robust |
| 49-52 | ofi_near_level_{30s,60s,120s,300s} | Robust |
| 53 | ofi_acceleration | Robust |
| 54-56 | barrier_delta_{1,3,5}min | Robust |
| 57-59 | barrier_pct_change_{1,3,5}min | Robust |
| 60 | approach_velocity | Z-Score |
| 61 | approach_bars | MinMax [0, 40] |
| 62 | approach_distance_atr | Z-Score |

### A.3 Section C: Micro-History (35 features)

| Index | Feature × Time | Normalization |
|-------|----------------|---------------|
| 63-67 | distance_signed_atr[t-4..t0] | Z-Score |
| 68-72 | tape_imbalance[t-4..t0] | Robust |
| 73-77 | tape_velocity[t-4..t0] | Robust |
| 78-82 | ofi_60s[t-4..t0] | Robust |
| 83-87 | barrier_delta_liq[t-4..t0] | Robust |
| 88-92 | wall_ratio[t-4..t0] | Robust |
| 93-97 | gamma_exposure[t-4..t0] | Robust |

### A.4 Section D: Derived Physics (9 features)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 98 | predicted_accel | Z-Score |
| 99 | accel_residual | Robust |
| 100 | force_mass_ratio | Robust |
| 101 | barrier_state_encoded | Passthrough |
| 102 | barrier_depth_current | Robust |
| 103 | barrier_replenishment_ratio | Robust |
| 104 | sweep_detected | Passthrough |
| 105 | tape_log_ratio | Robust |
| 106 | tape_log_total | Robust |

### A.5 Section E: Cluster Trends (4 features)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 107 | barrier_replenishment_trend | Robust |
| 108 | barrier_delta_liq_trend | Robust |
| 109 | tape_velocity_trend | Robust |
| 110 | tape_imbalance_trend | Robust |

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

### C.1 Zone and Trigger Parameters

```
ZONE_THRESHOLD_ATR = 3.0          # Episode triggers within this distance
MIN_APPROACH_VELOCITY = 0.5       # Minimum velocity to trigger query (points/min)
CONTACT_THRESHOLD_ATR = 0.2       # "At level" threshold
```

### C.2 Time Buckets

```
TIME_BUCKETS = {
    'T0_30':   (0, 30),     # Minutes 0-30 since open
    'T30_60':  (30, 60),    # Minutes 30-60
    'T60_120': (60, 120),   # Minutes 60-120
    'T120_180': (120, 180)  # Minutes 120-180
}
```

### C.3 Retrieval Parameters

```
K_RETRIEVE = 100                  # Over-fetch from FAISS
K_RETURN = 50                     # Final neighbors returned
MIN_SIMILARITY_THRESHOLD = 0.70   # Minimum similarity for quality result
MIN_SAMPLES_THRESHOLD = 30        # Minimum neighbors for reliable estimate
CACHE_TTL_SECONDS = 30            # Query result cache TTL
```

### C.4 Normalization Parameters

```
LOOKBACK_DAYS = 60                # Days of history for normalization stats
CLIP_SIGMA = 4.0                  # Clip normalized values at ±4σ
```

### C.5 Validation Thresholds

```
MIN_PARTITION_SIZE = 100          # Don't create index for smaller partitions
DRIFT_WARNING_WASSERSTEIN = 0.5   # Feature drift warning threshold
DRIFT_ALERT_MEAN_SHIFT = 2.0      # Mean shift alert threshold (in std)
CALIBRATION_WARNING_ECE = 0.10    # Expected calibration error warning
```

### C.6 Level Kinds

```
LEVEL_KINDS = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_200', 'SMA_400']
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

**Implementation Order:**

1. Outcome Label Contract (Section 3)
2. State Table Materialization (Section 4)
3. Episode Vector Construction (Section 6)
4. Normalization (Section 7)
5. Index Building (Section 8)
6. Retrieval Pipeline (Section 9)
7. Outcome Aggregation (Section 10)
8. Attribution (Section 11)
9. Validation (Section 12)
10. Pipeline Integration (Section 13)

**Critical Invariants:**

- All features in vectors are online-safe (no future data)
- All price-based features are ATR-normalized
- All distance features use consistent sign convention
- Temporal CV prevents any data leakage
- Index partitioning ensures regime-comparable neighbors
