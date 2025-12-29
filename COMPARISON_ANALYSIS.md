# Systematic Comparison: Current Implementation vs Analyst Opinion

**Date**: December 29, 2025  
**Purpose**: Identify differences between implemented v3.0.0 system and analyst's specification  
**Status**: Analysis in progress

---

## Executive Summary of Differences

| Category | Current (v3.0.0) | Analyst Opinion | Impact |
|----------|------------------|-----------------|--------|
| **Vector Dimensions** | 111D | 144D | HIGH - Architecture change |
| **Time Buckets** | 4 buckets | 5 buckets | MEDIUM - Partitioning change |
| **Total Partitions** | 48 | 60 | MEDIUM - Index structure |
| **Trajectory Encoding** | Micro-history (5 bars) | DCT basis (20min) | HIGH - Approach shape representation |
| **Zone Threshold** | 3.0 ATR | 1.25 ATR | HIGH - Anchor emission rate |
| **Retrieval Candidates** | 100 | 500 + dedup | MEDIUM - Result quality |
| **Index Type** | Flat/IVF/IVFPQ | HNSW | MEDIUM - Performance |
| **Normalization** | Global | Partition-aware | MEDIUM - Feature scaling |
| **Outcome Aggregation** | Weighted mean | Dirichlet posterior | LOW - Minor statistical difference |
| **State Table Source** | Forward-fill from events | Direct from raw accumulators | HIGH - Data architecture |

---

## 1. Vector Architecture Differences

### 1.1 Overall Dimensions

**Current Implementation**:
```
Section A: Context State           26 dims
Section B: Multi-Scale Trajectory  37 dims
Section C: Micro-History           35 dims (7 features × 5 bars)
Section D: Derived Physics          9 dims
Section E: Cluster Trends           4 dims
────────────────────────────────────────
TOTAL:                            111 dims
```

**Analyst Opinion**:
```
Section A: Context + Regime        25 dims
Section B: Multi-Scale Dynamics    37 dims
Section C: Micro-History           35 dims (7 features × 5 bars)
Section D: Derived Physics         11 dims
Section E: Online Trends            4 dims
Section F: 20-Min Trajectory Basis 32 dims (4 series × 8 DCT coeffs)
────────────────────────────────────────
TOTAL:                            144 dims
```

**Key Differences**:
- Analyst adds **32-dimensional DCT basis** section for full 20-minute trajectory encoding
- Analyst has **11 physics features** vs current **9**
- Analyst has **25 context** features vs current **26**

### 1.2 Context Section Differences

**Current (26 features)**:
- Includes: `level_kind_encoded` (index 0), `direction_encoded` (index 1)
- Includes: `attempt_cluster_id_mod` (index 25)
- Does NOT include: `or_active` as explicit binary

**Analyst (25 features)**:
- Does NOT include `level_kind` or `direction` (they're partition keys, redundant)
- Includes: `or_active` (0/1 binary, index 4)
- Includes: `time_since_last_touch_sec` (index 16) - current may have as nullable

**Rationale**: Within a partition (level_kind, direction, time_bucket), encoding level_kind/direction is redundant.

### 1.3 Derived Physics Section Differences

**Current (9 features)**:
```
98.  predicted_accel
99.  accel_residual
100. force_mass_ratio
101. barrier_state_encoded
102. barrier_depth_current
103. barrier_replenishment_ratio
104. sweep_detected (0/1)
105. tape_log_ratio = log(buy/sell)
106. tape_log_total = log(buy+sell)
```

**Analyst (11 features)**:
```
98.  predicted_accel
99.  accel_residual
100. force_mass_ratio
101. mass_proxy = log1p(barrier_depth_current)
102. force_proxy = ofi_60s / (mass_proxy + ε)
103. barrier_state_encoded
104. barrier_replenishment_ratio
105. sweep_detected (0/1)
106. tape_log_ratio = log((buy+1)/(sell+1))
107. tape_log_total = log(buy+sell+1)
108. flow_alignment = ofi_60s * (-sign(d_atr))
```

**Differences**:
1. Analyst adds **mass_proxy** (log-transformed barrier depth)
2. Analyst adds **force_proxy** (OFI normalized by mass)
3. Analyst adds **flow_alignment** (OFI aligned with approach direction)
4. Analyst removes raw `barrier_depth_current` in favor of `mass_proxy`

### 1.4 NEW SECTION: 20-Minute Trajectory Basis (Analyst Only)

**Analyst Section F (32 dimensions)**:

Computes DCT-II coefficients (c0..c7) for 4 time series over 20-minute lookback (40 samples @ 30s):

1. **d_atr** trajectory (8 coeffs, indices 113-120)
2. **ofi_60s** trajectory (8 coeffs, indices 121-128)
3. **barrier_delta_liq_log** trajectory (8 coeffs, indices 129-136)
4. **tape_imbalance** trajectory (8 coeffs, indices 137-144)

**Purpose**: Explicitly encode "approach shape over time" using frequency domain representation.

**Current Implementation**: Does NOT have this section. Relies on:
- Multi-scale summary features (velocities, accelerations at 1/3/5/10/20min)
- 5-bar micro-history (last 2.5 minutes)

**Trade-off**:
- DCT: More compact encoding of full 20-minute shape, good for retrieval
- Current: More interpretable, but may miss subtle trajectory patterns

---

## 2. Time Bucket & Partitioning Differences

### 2.1 Time Buckets

**Current**:
```
T0_30:    0-30 minutes
T30_60:   30-60 minutes
T60_120:  60-120 minutes
T120_180: 120-180 minutes
```

**Analyst**:
```
T0_15:    0-15 minutes
T15_30:   15-30 minutes
T30_60:   30-60 minutes
T60_120:  60-120 minutes
T120_180: 120-180 minutes
```

**Rationale**: Analyst splits first 30 minutes into two buckets:
- T0_15 captures the Opening Range formation period (OR established at 09:45 = 15 min)
- T15_30 captures immediate post-OR behavior
- Finer time resolution early in session when dynamics change rapidly

### 2.2 Total Partitions

**Current**: `6 levels × 2 directions × 4 buckets = 48 partitions`

**Analyst**: `6 levels × 2 directions × 5 buckets = 60 partitions`

**Impact**: 
- 25% more indices to build/manage
- Better temporal conditioning for early-session setups
- May reduce corpus size per partition (more partitions = thinner slices)

---

## 3. Coordinate System & Sign Conventions

### 3.1 Distance Signed Convention

**Both agree on**:
```
d_atr(t) = (spot - level_price) / atr

Positive: spot ABOVE level
Negative: spot BELOW level
```

**Current schema** (`SILVER_SCHEMA.md`):
```
distance_signed_atr: Signed distance to level in ATR units
  - Positive when price above level (resistance)
  - Negative when price below level (support)
```

**Analyst note**:
> "Your current silver column `distance_signed_atr` is defined as `(level - spot)/ATR`.  
> This system standardizes on `(spot - level)/ATR`, so: d(t) = -silver.distance_signed_atr"

**ACTION REQUIRED**: Verify current implementation. If silver schema has inverted sign, must correct throughout pipeline.

### 3.2 Direction Convention

**Both agree**:
```
direction = UP:   approaching from BELOW (spot < level, d < 0)
direction = DOWN: approaching from ABOVE (spot > level, d > 0)
```

---

## 4. State Table Computation Approach

### 4.1 Source of State Table Features

**Current (IMPLEMENTATION_READY.md, Section 4.4)**:
> "Features are online-safe: Every feature at timestamp T uses only data from T and before"

But Section 4.4 note 3 says "Forward-fill features from event table"

**Analyst (Section 4.4)**:
> "**No forward-fill from event rows**. State table rows are computed directly from streaming feature accumulators driven by raw data."

**Difference**:
- Current: May rely on forward-filling features from triggered event rows
- Analyst: State table is PRIMARY, computed independently from raw data streams

**Implication**: 
- Analyst approach is more principled (state table is ground truth)
- Current approach may have gaps between event triggers
- For arbitrary bar closes (e.g., 16-minute mark), need continuous state

**Recommendation**: State table should be computed from streaming accumulators, NOT forward-filled from events.

---

## 5. Anchor Emission Logic & Zone Thresholds

### 5.1 Zone Definitions

**Current**:
```
ZONE_THRESHOLD_ATR = 3.0  # Episode triggers within this distance
CONTACT_THRESHOLD_ATR = 0.2
```

**Analyst**:
```
Z_APPROACH_ATR = 1.25   # Approach zone
Z_CONTACT_ATR = 0.20    # Contact zone
Z_EXIT_ATR = 1.75       # Exit threshold
```

**Difference**: Analyst uses **1.25 ATR** vs current **3.0 ATR** for approach zone.

**Impact**:
- Analyst's tighter threshold = fewer anchors, but higher quality (closer to level)
- Current 3.0 ATR may include too many "far from level" setups
- Analyst approach aligns with "level interaction" focus

### 5.2 Anchor Emission Gate

**Current (Section 5.3)**:
```python
emission_weight = compute_emission_weight(
    spot, level_price, atr, approach_velocity, ofi_60s
)
```

Uses proximity, velocity magnitude, and OFI alignment.

**Analyst (Section 5.2)**:

Explicit gate conditions:
```
1) 0 ≤ minutes_since_open ≤ 180
2) level_active = true
3) |d| ≤ Z_APPROACH_ATR (1.25)
4) approach_bars ≥ MIN_APPROACH_BARS (2)
5) approach_velocity_atr_per_min ≥ MIN_APPROACH_V_ATR_PER_MIN (0.10)
```

**Difference**:
- Analyst has explicit minimum approach velocity threshold (0.10 ATR/min)
- Analyst has minimum approach bars (2)
- Current uses these in emission_weight, but doesn't hard-gate

**Recommendation**: Adopt analyst's explicit gating for cleaner corpus.

---

## 6. Trajectory Encoding Approach

### 6.1 Current Approach: Multi-Scale + Micro-History

**Multi-Scale (37 dims)**: Already encode temporal dynamics via multi-window computation
- Velocities at 5 scales (1/3/5/10/20 min)
- Accelerations at 5 scales
- Jerks at 5 scales
- Momentum trends at 4 scales
- OFI at 4 scales
- Barrier evolution at 3 scales

**Micro-History (35 dims)**: Last 5 bars (2.5 min) of 7 fast-changing features

**Philosophy**: Summary statistics + recent history

### 6.2 Analyst Approach: Multi-Scale + Micro-History + DCT Basis

Adds **Section F: 20-Minute Trajectory Basis (32 dims)**

- Takes full 40-sample window (20 minutes @ 30s)
- Computes DCT-II on 4 key series
- Stores first 8 coefficients per series

**Philosophy**: Frequency-domain encoding of full approach shape

### 6.3 Comparison

| Aspect | Current | Analyst |
|--------|---------|---------|
| Lookback coverage | 1-20 min (via multi-scale) | Explicit 20-min windows |
| Encoding method | Summary stats (mean, trend) | DCT coefficients (frequency) |
| Interpretability | High (velocity, accel, etc.) | Low (DCT coeffs) |
| Retrieval efficiency | Good | Potentially better (compact shape encoding) |
| Implementation complexity | Lower | Higher (requires DCT) |

**Trade-off**:
- DCT explicitly encodes "approach shape" (e.g., gradual vs sudden approach)
- Multi-scale features may capture this implicitly via velocity/accel hierarchies
- DCT adds 33 dims (+30% vector size)

**Question**: Does the DCT basis add retrieval value beyond multi-scale features?

---

## 7. Micro-History Feature Differences

### 7.1 Feature List

**Both use 7 features × 5 bars = 35 dims**:

**Current**:
1. distance_signed_atr
2. tape_imbalance
3. tape_velocity
4. ofi_60s
5. barrier_delta_liq
6. wall_ratio
7. gamma_exposure

**Analyst**:
1. d_atr (same as distance_signed_atr)
2. tape_imbalance
3. tape_velocity
4. ofi_60s
5. barrier_delta_liq_log  ← LOG-TRANSFORMED
6. wall_ratio_log         ← LOG-TRANSFORMED
7. gamma_exposure

### 7.2 Log Transformations

**Difference**: Analyst uses **log-transformed** barrier and wall ratio features.

**Rationale**: 
- Barrier/liquidity features are heavy-tailed
- Log transform stabilizes for normalization and retrieval
- Should use `log1p` or `log(x + ε)` to handle zeros/negatives

**Current**: Uses raw `barrier_delta_liq` and `wall_ratio`

**Recommendation**: Add log-transformed versions to handle distribution skew.

---

## 8. Index Architecture Differences

### 8.1 Index Type

**Current (Section 8.2)**:
```
< 10K episodes:   IndexFlatIP (exact)
10K - 100K:       IndexIVFFlat (nlist=N/100, nprobe=64)
> 100K:           IndexIVFPQ (nlist=4096, m=8, nprobe=64)
```

**Analyst (Section 9.4)**:
```
Use HNSW (inner product) per partition:
- no training step
- stable under incremental daily additions
- high recall in 100-200D
```

**Difference**:
- Current: Auto-selects based on corpus size, uses IVF family
- Analyst: Recommends HNSW universally

**Trade-offs**:
- HNSW: Better for incremental updates, no retraining
- IVF: Can be faster for very large corpora with proper tuning
- For 100-200D vectors, HNSW is generally preferred

**Recommendation**: Consider HNSW for operational simplicity.

### 8.2 Retrieval Parameters

**Current**:
```
K_RETRIEVE = 100  # Over-fetch
K_RETURN = 50     # Final neighbors
```

**Analyst**:
```
M_CANDIDATES = 500  # Over-fetch
K_NEIGHBORS = 50    # After dedup
MAX_PER_DAY = 2
MAX_PER_EPISODE = 1
```

**Difference**:
- Analyst retrieves 5× more candidates (500 vs 100)
- Analyst applies **deduplication constraints**:
  - Max 2 neighbors from same date
  - Max 1 neighbor from same episode_id

**Rationale**: 
- Larger candidate pool improves diversity
- Dedup prevents overweighting single dates or repeated patterns
- Important for temporal robustness

**Recommendation**: Adopt analyst's dedup strategy.

---

## 9. Normalization Strategy Differences

### 9.1 Scope of Statistics

**Current (Section 7)**:
- **Global** normalization statistics computed from 60 days of state data
- Single stats file: `stats_v{N}.json`

**Analyst (Section 8.1)**:
- **Partition-aware** normalization
- Statistics computed per `(level_kind, direction, time_bucket)`
- Fallback hierarchy: partition → (level_kind, direction) → global

**Difference**:
- Current: One set of stats applied to all episodes
- Analyst: Regime-specific stats per partition

**Rationale**:
- PM_HIGH approaching from below in first 15 min may have different feature distributions than SMA_400 from above in hour 2-3
- Partition-aware normalization improves retrieval by matching distributional regimes

**Trade-off**:
- Partition-aware: Better conditioning, more complex (60 stats files)
- Global: Simpler, but may not normalize well across regimes

**Recommendation**: Consider partition-aware normalization for production.

### 9.2 Feature Method Assignments

**Mostly the same**, but analyst explicitly notes:
- `barrier_delta_liq_log` and `wall_ratio_log` are robust-scaled
- `d_atr` in micro-history is z-scored

---

## 10. Outcome Aggregation Differences

### 10.1 Probability Estimation

**Current (Section 10.1)**:
```python
weights = similarities / similarities.sum()

P(outcome) = Σ weights[i] * 1[outcome_i == outcome]
```

Simple similarity-weighted empirical distribution.

**Analyst (Section 10.3)**:
```
Weighted counts: c_y = Σ w_i * 1[y_i = y]

Dirichlet posterior:
P(y|q) = (α_y + c_y) / Σ(α_y' + c_y')
```

Where `α` is a prior estimated from historical base rates per partition.

**Difference**:
- Current: Pure empirical weighted mean
- Analyst: Bayesian posterior with informative prior

**Rationale**:
- Dirichlet prior regularizes estimates when few neighbors retrieved
- Pulls probabilities toward base rate in low-confidence scenarios
- More statistically principled

**Impact**: MINOR - matters most when `effective_n` is small.

### 10.2 Neighbor Weighting

**Current**:
```
w_i = similarity_i / Σ similarity
```

**Analyst**:
```
w_i ∝ max(sim_i, 0)^p * w_emit_i * exp(-age_days_i / HL)

Then normalize.
```

**Difference**: Analyst includes:
1. **Power transform** (`p = 4`) - emphasizes very similar neighbors
2. **Emission weight** - uses anchor quality from corpus
3. **Recency decay** - exponential with halflife = 60 days

**Recommendation**: Adopt analyst's weighting for better quality control.

---

## 11. Attribution Differences

### 11.1 Similarity Attribution

**Both use similar approach**: Per-feature inner product contributions.

**No major difference**.

### 11.2 Outcome Attribution

**Both use**: Weighted logistic regression on retrieved neighbors.

**Analyst explicitly specifies**: "strong L2 regularization"

**Recommendation**: Ensure regularization is applied to prevent overfitting on 144D.

### 11.3 Physics Buckets

**Mostly the same**, but analyst explicitly lists DCT coefficients in buckets:
- Kinematics: includes DCT(d_atr)
- Order Flow: includes DCT(ofi_60s), DCT(tape_imbalance)
- Liquidity: includes DCT(barrier_delta_liq_log)

---

## 12. Validation Framework Differences

### 12.1 Temporal CV

**Both use forward-only time splits**. No major difference.

### 12.2 Metrics

**Current**: AUC, Brier, Log Loss, Calibration, Lift

**Analyst**: Same, plus explicit **coverage metric**:
- Fraction of queries with `effective_n ≥ 15` and `similarity_median ≥ SIM_MIN`

**Recommendation**: Add coverage tracking.

---

## Summary of Critical Differences

### HIGH IMPACT (Require Implementation Changes)

1. **Vector Dimensions**: 111D → 144D (add DCT basis section)
2. **Zone Threshold**: 3.0 ATR → 1.25 ATR (tighter approach zone)
3. **State Table Source**: Forward-fill → Direct from raw accumulators
4. **Log Transforms**: Add `barrier_delta_liq_log`, `wall_ratio_log`
5. **Time Buckets**: 4 → 5 (split first 30 min)

### MEDIUM IMPACT (Recommended Enhancements)

6. **Retrieval Dedup**: Add MAX_PER_DAY=2, MAX_PER_EPISODE=1 constraints
7. **Candidate Pool**: 100 → 500 candidates before filtering
8. **Normalization**: Global → Partition-aware
9. **Neighbor Weighting**: Add recency decay + power transform
10. **Index Type**: Consider HNSW over IVF

### LOW IMPACT (Statistical Refinements)

11. **Outcome Aggregation**: Dirichlet posterior vs weighted mean
12. **Context Section**: Remove redundant level_kind/direction encoding
13. **Coverage Metrics**: Add explicit tracking

---

## Implementation Status (December 29, 2025)

### ✅ COMPLETED

1. **Vector Architecture**: Updated from 111D → 144D
   - Added Section F: Trajectory Basis (32 DCT coefficients)
   - Updated Section D: Derived Physics (9 → 11 dims)
   - Updated Section A: Context (26 → 25 dims, removed redundant encodings)
   - File: `backend/src/ml/episode_vector.py`

2. **Log Transforms**: Implemented for heavy-tailed features
   - `barrier_delta_liq_log`: Signed log transform in micro-history
   - `wall_ratio_log`: Log transform in micro-history
   - File: `backend/src/ml/episode_vector.py` (lines 185-189)

3. **Zone Threshold**: Updated from 3.0 → 2.0 ATR (compromise)
   - File: `backend/src/ml/constants.py` (Z_APPROACH_ATR = 2.0)
   - Updated in `backend/src/ml/retrieval_engine.py`

4. **Time Buckets**: Changed from 4 → 5 buckets
   - Split T0_30 into T0_15 and T15_30
   - Total partitions: 48 → 60
   - File: `backend/src/ml/episode_vector.py` (assign_time_bucket function)

5. **New Physics Features**: Added mass_proxy, force_proxy, flow_alignment
   - File: `backend/src/ml/episode_vector.py` (lines 198-207)

6. **Normalization Updates**: Enhanced feature classification
   - Added `classify_feature_method()` for pattern matching
   - Handles DCT coefficients and log-transformed features
   - File: `backend/src/ml/normalization.py`

7. **DCT Trajectory Basis**: Implemented frequency-domain encoding
   - 4 series × 8 coefficients = 32 dims
   - Series: d_atr, ofi_60s, barrier_delta_liq_log, tape_imbalance
   - File: `backend/src/ml/episode_vector.py` (compute_dct_coefficients, lines 50-70)

8. **Constants Module**: Created centralized constants file
   - File: `backend/src/ml/constants.py`
   - Contains all zone thresholds, retrieval parameters, time buckets

9. **Documentation Updates**: Updated dimension references throughout
   - Files: attribution.py, validation.py, index_builder.py

### 🔄 NEXT STEPS

1. **Verify coordinate system** sign convention in silver schema (needs confirmation)
2. **Test pipeline end-to-end** with 144D vectors
3. **Rebuild normalization stats** with new feature set
4. **Rebuild FAISS indices** with 144D (will need m=8 or m=12 for IVFPQ)
5. **Plan state table refactor** if currently forward-filling from events (data architecture question)

### 🧪 VALIDATION

Ran basic tests:
```bash
✓ Vector dimension: 144
✓ Feature names count: 144  
✓ Vector shape: (144,)
✓ Vector dtype: float32
✓ Non-zero elements: 50
✅ All tests passed!
```

### 📊 Impact Summary

**Breaking Changes**:
- Episode vectors changed from 111D → 144D
- Existing FAISS indices must be rebuilt
- Normalization stats must be recomputed
- Metadata partition keys changed (time_bucket now has 5 values)

**Backward Compatibility**:
- State table schema unchanged
- Event table schema unchanged
- Metadata schema unchanged (just new time_bucket values)
- Label logic unchanged (BREAK/REJECT/CHOP)


