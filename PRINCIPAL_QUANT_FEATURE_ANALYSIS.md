# Principal Quant Feature Analysis Methodology

## Execution Plan (live)

1. [x] Load silver approach data for futures (2025-06-04 to 2025-06-13) and build the raw feature matrix.
2. [x] Fit normalization on train dates and save params.
3. [x] Run Level 1–3 analysis (variance, univariate, redundancy) on normalized features.
4. [x] Run Level 4–7 analysis (forward selection, backward elimination, interaction, stability).
5. [x] Run category analysis and compute final tiers.
6. [x] Record outputs for each step in machine-readable files and summarize here.

## Latest Run Summary

- Symbol: ESU5
- Dates: 2025-06-04 to 2025-06-13 (skipped 2025-06-07, 2025-06-08)
- Rows after trigger + RTH filter: 238
- Numeric feature matrix: 448 features (string + label columns excluded)
- Cluster representatives: 253
- Train dates: 2025-06-04, 2025-06-05, 2025-06-06, 2025-06-09, 2025-06-10, 2025-06-11
- Test dates: 2025-06-12, 2025-06-13
- Outputs: `backend/ablation_results/principal_quant_feature_analysis/` with
  - `feature_names.json`
  - `normalization_params.json`
  - `variance_analysis.json`
  - `univariate_analysis.csv`
  - `redundancy_analysis.json`
  - `cluster_representatives.json`
  - `cluster_representatives_used.json`
  - `forward_selection.json`
  - `backward_elimination.json`
  - `interaction_analysis.csv`
  - `stability_analysis.json`
  - `category_analysis.csv`
  - `final_recommendations.json`
  - `run_summary.json`

### Step Outputs (counts)

- Variance (train-normalized): 12 zero-variance, 0 low-variance, 436 normal (`variance_analysis.json`)
- Univariate: 448 rows (`univariate_analysis.csv`)
- Redundancy: 253 clusters, 253 representatives (`redundancy_analysis.json`, `cluster_representatives.json`)
- Forward selection: 3 features selected (`forward_selection.json`)
- Backward elimination: 3 features retained (`backward_elimination.json`)
- Interaction: `interaction_analysis.csv`
- Stability: 4 folds, 0 stable features at >=60% frequency (`stability_analysis.json`)
- Category: `category_analysis.csv`
- Final tiers: GOLD 0, SILVER 0, BRONZE 253, DISCARD 195 (`final_recommendations.json`)

**Cardinal rule:** Never discard features without empirical evidence. Let the data decide.

---

Raw Features
    ↓
[NORMALIZE HERE] ←────────────────────────────────────
    ↓                                                 │
Level 1: Variance Filter (on normalized data)         │
    ↓                                                 │
Level 2: Univariate Screen                            │  ALL ANALYSIS
    ↓                                                 │  HAPPENS ON
Level 3: Redundancy Clustering                        │  NORMALIZED
    ↓                                                 │  FEATURES
Level 4: Forward Selection                            │
    ↓                                                 │
Level 5: Backward Elimination                         │
    ↓                                                 │
Level 6-7: Interaction & Stability ───────────────────
    ↓
Final Feature Set
    ↓
[SAVE NORMALIZATION PARAMS] ← Fitted on training data only
    ↓
Production: Apply saved params to new data

---

## Section 1 — Feature Analysis Framework

### 1.1 The Hierarchy of Evidence

| Level | Method | What It Tells You |
|-------|--------|-------------------|
| 1 | Variance analysis | Is the feature constant? (zero information) |
| 2 | Univariate correlation | Does feature alone predict outcome? |
| 3 | Redundancy analysis | Is feature duplicate of another? |
| 4 | Ablation (remove) | Does removing feature hurt performance? |
| 5 | Ablation (add) | Does adding feature help performance? |
| 6 | Interaction analysis | Does feature help in combination? |
| 7 | Stability analysis | Do results hold out-of-sample? |

**You should not discard a feature until it fails at Level 4 or 5.**

---

## Section 2 — Level 1: Variance Analysis


### Variance Analysis Output EXAMPLE

```
ZERO VARIANCE (REMOVE):
  - level_polarity: var=0.0 (constant value)
  - recent_flow_away: var=0.0
  - recent_aggbuy_vol: var=0.0
  ... (your 6 zero-variance features)

LOW VARIANCE (FLAG):
  - is_standard_approach: var=0.002 (nearly binary)
  - approach_alignment_eob: var=0.003
  ...

NORMAL VARIANCE (KEEP):
  - 438 features
```

### 2.3 Decision Rule

| Variance | Action |
|----------|--------|
| = 0 | Remove immediately |
| < 0.01 | Flag, but include in downstream analysis |
| >= 0.01 | Keep |

---

## Section 3 — Level 2: Univariate Analysis

### 3.2 Correlation with Outcome Output Format EXAMPLE

```
UNIVARIATE PREDICTIVE POWER:

| Rank | Feature                        | Pearson r | Spearman r | F-stat | MI    |
|------|--------------------------------|-----------|------------|--------|-------|
| 1    | setup_velocity_trend           | 0.182     | 0.175      | 45.2   | 0.08  |
| 2    | deriv_dist_d1_w12              | 0.156     | 0.148      | 38.1   | 0.07  |
| 3    | rvol_bid_ask_net_asymmetry     | 0.134     | 0.129      | 31.4   | 0.06  |  ← YOU DISCARDED THIS
| 4    | setup_approach_ratio           | 0.128     | 0.122      | 28.9   | 0.05  |
| ...  | ...                            | ...       | ...        | ...    | ...   |
| 400  | bar5s_shape_bid_sz_l07_eob     | 0.003     | 0.002      | 0.8    | 0.001 |
```

### 3.3 Decision Rule

| Univariate Signal | Action |
|-------------------|--------|
| \|r\| > 0.10 | Strong candidate — prioritize in forward selection |
| 0.05 < \|r\| < 0.10 | Moderate candidate — include in analysis |
| 0.02 < \|r\| < 0.05 | Weak candidate — may help in combination |
| \|r\| < 0.02 | Likely noise — but test before discarding |

**CRITICAL:** Do NOT discard features based solely on low univariate correlation. Features can have low marginal value but high conditional value (interactions).

---

## Section 4 — Level 3: Redundancy Analysis

### 4.3 Output Format EXAMPLE

```
REDUNDANCY CLUSTERS:

Cluster 1 (8 members, r > 0.85):
  Representative: deriv_dist_d1_w12 (|r|=0.156)
  Dropped:
    - deriv_dist_d1_w36 (r=0.94 with representative)
    - deriv_dist_d1_w3 (r=0.91)
    - deriv_dist_d1_w72 (r=0.89)
    - setup_late_velocity (r=0.87)
    ...

Cluster 2 (5 members):
  Representative: state_obi0_eob (|r|=0.089)
  Dropped:
    - state_obi10_eob (r=0.92)
    - setup_obi0_mean (r=0.88)
    ...

...

SUMMARY:
  - 454 features → 87 clusters
  - 87 representative features retained
  - 367 redundant features dropped
```

---

## Section 5 — Level 4: Forward Selection


### 5.2 Greedy Forward Selection Output Format EXAMPLE

```
FORWARD SELECTION LOG:

Step 1:  +setup_velocity_trend          → 0.312 (+0.112)  # Biggest single feature
Step 2:  +deriv_dist_d1_w12             → 0.378 (+0.066)
Step 3:  +rvol_bid_ask_net_asymmetry    → 0.412 (+0.034)  ← RVOL FEATURE SELECTED
Step 4:  +setup_approach_ratio          → 0.435 (+0.023)
Step 5:  +state_obi0_eob                → 0.449 (+0.014)
Step 6:  +rvol_cumul_trade_vol_dev_pct  → 0.458 (+0.009)  ← ANOTHER RVOL FEATURE
Step 7:  +cumul_flow_imbal              → 0.464 (+0.006)
...
Step 23: +bar5s_shape_bid_sz_l00_eob    → 0.483 (+0.002)  ← SHAPE FEATURE HELPED
Step 24: +setup_bid_wall_max_z          → 0.484 (+0.001)
Step 25: +deriv_obi0_d1_w12             → 0.484 (+0.000)  ← No improvement

STOPPING: 5 rounds without improvement > 0.1%

FINAL SELECTED SET: 24 features
FINAL ACCURACY: 48.4%
```

---

## Section 6 — Level 5: Backward Elimination


### 6.2 Feature is Redundancy Output Format EXAMPLE

```
BACKWARD ELIMINATION:

Testing removal of 24 forward-selected features...

  - setup_bid_wall_max_z: drop=0.001 → REMOVE (marginal)
  - deriv_obi0_d1_w12: drop=-0.002 → REMOVE (actually helps to remove!)
  - bar5s_shape_bid_sz_l00_eob: drop=0.003 → REMOVE (marginal)
  - setup_velocity_trend: drop=0.045 → KEEP (significant drop)
  - rvol_bid_ask_net_asymmetry: drop=0.028 → KEEP
  ...

FINAL SET AFTER ELIMINATION: 19 features
FINAL ACCURACY: 48.6% (+0.2% from removing noise)
```

---

## Section 7 — Level 6: Interaction Analysis

### 7.1 Test Feature Combinations

Some features are only useful in combination. Test this


### 7.2 Output Format EXAMPLE

```
INTERACTION ANALYSIS:

Features that failed forward selection but might help in combination:

| Candidate                    | Single | +BaseSet | Improve | Best Pair With           |
|------------------------------|--------|----------|---------|--------------------------|
| bar5s_shape_ask_ct_l00_eob   | 22.1%  | 49.2%    | +0.6%   | state_obi0_eob (34.2%)   |
| rvol_lookback_elevated_bars  | 23.4%  | 49.0%    | +0.4%   | setup_velocity_trend     |
| bar5s_depth_below_p0_1_eob   | 21.8%  | 48.9%    | +0.3%   | approach_dist_to_level   |

CONCLUSION: 3 additional features show interaction effects. Consider adding.
```

---

## Section 8 — Level 7: Stability Validation

### 8.1 Time-Based Cross-Validation


### 8.2 Output Format EXAMPLE

```
STABILITY ANALYSIS (5-fold time-series CV):

| Fold | Train End  | Test Period        | Features | Test Acc |
|------|------------|--------------------|----------|----------|
| 0    | 2024-07-15 | Jul 15 - Aug 01    | 21       | 46.2%    |
| 1    | 2024-08-01 | Aug 01 - Aug 15    | 19       | 48.1%    |
| 2    | 2024-08-15 | Aug 15 - Sep 01    | 22       | 47.8%    |
| 3    | 2024-09-01 | Sep 01 - Sep 15    | 20       | 49.2%    |
| 4    | 2024-09-15 | Sep 15 - Sep 30    | 18       | 51.3%    |

STABLE FEATURES (selected in ALL folds):
  1. setup_velocity_trend
  2. deriv_dist_d1_w12
  3. rvol_bid_ask_net_asymmetry
  4. setup_approach_ratio
  5. cumul_flow_imbal
  ... (14 total)

UNSTABLE FEATURES (selected in SOME folds):
  - bar5s_shape_bid_sz_l00_eob (3/5 folds)
  - rvol_lookback_elevated_bars (2/5 folds)
  - setup_bid_wall_max_z (2/5 folds)
  ... (8 total)

STABILITY RATIO: 14 / 22 = 63.6%

MEAN TEST ACCURACY: 48.5% ± 1.8%

CONCLUSION: 
  - 14 features are consistently selected (HIGH CONFIDENCE)
  - 8 features are regime-dependent (MEDIUM CONFIDENCE)
  - Consider keeping stable features only for production
```

---

## Section 9 — Category-Level Analysis

Before discarding entire categories (like "ALL rvol features"), run category-level tests.

### Output Format EXAMPLE

```
CATEGORY-LEVEL ANALYSIS:

| Category        | N Feats | Alone  | Without | Contribution | Verdict       |
|-----------------|---------|--------|---------|--------------|---------------|
| profile_traj    | 12      | 44.1%  | 39.2%   | +4.9%        | CRITICAL      |
| position        | 5       | 38.9%  | 43.8%   | +0.3%        | HELPFUL       |
| rvol_*          | 34      | 36.2%  | 42.1%   | +2.0%        | HELPFUL ← !!! |
| flow_snapshot   | 10      | 35.8%  | 43.1%   | +1.0%        | HELPFUL       |
| deriv_*         | 48      | 33.2%  | 43.9%   | +0.2%        | MARGINAL      |
| shape_*         | 100     | 28.1%  | 44.2%   | -0.1%        | REMOVE        |
| walls           | 8       | 25.4%  | 45.8%   | -1.7%        | HARMFUL       |

CONCLUSIONS:
  - rvol features contribute +2.0% — DO NOT DISCARD
  - shape features are neutral — safe to discard
  - walls are harmful — should discard
```

---

## Section 10 — Final Recommendations Process

### 10.1 Decision Matrix

For each feature, compute:

| Criterion | Weight | Measurement |
|-----------|--------|-------------|
| Variance | Pass/Fail | var > 1e-6 |
| Univariate signal | 20% | \|r\| with outcome |
| Non-redundancy | 20% | Cluster representative? |
| Forward-selected | 30% | Made the cut? |
| Survives backward elimination | 15% | Not eliminated? |
| Stable across folds | 15% | Selected in >3/5 folds? |

### 10.2 Feature Tiers

| Tier | Criteria | Action |
|------|----------|--------|
| **GOLD** | Forward-selected + stable + survives elimination | Always include |
| **SILVER** | Forward-selected OR high univariate + stable | Include if dimension budget allows |
| **BRONZE** | Shows interaction effects OR selected in some folds | Test in production A/B |
| **DISCARD** | Zero variance OR redundant OR harmful in ablation | Remove |

### 10.3 Final Output EXAMPLE

```
PRINCIPAL QUANT FEATURE ANALYSIS — FINAL RECOMMENDATIONS

DATA: 454 features, 2,248 episodes, June-September 2024

PROCESS:
  1. Variance filter: 454 → 448 (removed 6 zero-variance)
  2. Redundancy clustering: 448 → 94 clusters
  3. Forward selection: 94 → 26 features
  4. Backward elimination: 26 → 22 features
  5. Stability validation: 22 → 18 stable features

FINAL FEATURE SET (18 features):

GOLD TIER (always include):
  1. setup_velocity_trend         (profile_traj)
  2. deriv_dist_d1_w12            (deriv)
  3. setup_approach_ratio         (profile_traj)
  4. rvol_bid_ask_net_asymmetry   (rvol)         ← RVOL MADE THE CUT
  5. cumul_flow_imbal             (flow)
  6. setup_start_dist_pts         (profile_traj)
  7. state_obi0_eob               (book_state)
  8. approach_dist_to_level_pts   (position)
  9. setup_dist_range_pts         (profile_traj)
  10. rvol_cumul_trade_vol_dev_pct (rvol)        ← ANOTHER RVOL
  11. setup_early_velocity        (profile_traj)
  12. approach_side_of_level      (position)

SILVER TIER (include if budget allows):
  13. lvl_depth_imbal_eob         (level)
  14. rvol_lookback_trade_vol_trend (rvol)
  15. setup_flow_toward_away_ratio (flow)
  16. deriv_cdi01_d1_w12          (deriv)

BRONZE TIER (A/B test):
  17. bar5s_shape_bid_sz_l00_eob  (shape)        ← SOME SHAPE HELPS
  18. wall_ask_nearest_strong_dist (walls)

DISCARDED (with evidence):
  - 100 shape features: category contribution = -0.1%
  - 6 wall features: category contribution = -1.7%
  - 20 rvol features: redundant with 14 kept
  - 40 deriv features: redundant with 8 kept
  ...

PERFORMANCE:
  - Final accuracy: 49.2% (vs 43.2% baseline with all features)
  - Final score correlation: 0.38 (vs 0.15 baseline)
  - Cross-validation stability: ±1.8%
```

---

## Section 11 — What You Should Do Now

### Immediate Actions

1. **DO NOT discard rvol features** — run the category analysis first
2. **Run variance analysis** — confirm which features are actually zero-variance
3. **Run redundancy clustering** — see true cluster structure
4. **Run forward selection** — let the data pick features

### Code to Start

```python
# Step 1: Load your 454 features
vectors = np.load('all_454_features.npy')
feature_names = load_feature_names()  # All 454 names

# Step 2: Variance filter
variance_results = variance_analysis(vectors, feature_names)
print(f"Zero variance: {len(variance_results['zero_variance'])}")
print(f"Keeping: {len(variance_results['normal_variance'])}")

# Step 3: Univariate analysis
univariate_df = univariate_analysis(vectors, outcomes, outcome_class, feature_names)
print(univariate_df.head(30))  # Top 30 predictive features

# Step 4: Redundancy clustering
redundancy = redundancy_analysis(vectors, feature_names, threshold=0.85)
print(f"Clusters: {redundancy['n_clusters']}")
print(f"Redundant pairs (r>0.95): {len(redundancy['redundant_pairs'])}")

# Step 5: Category analysis (before discarding any category!)
categories = {
    'shape': shape_indices,
    'rvol': rvol_indices,
    'deriv': deriv_indices,
    ...
}
category_df = category_analysis(vectors, metadata, categories)
print(category_df)

# Step 6: Forward selection on cluster representatives
representatives = select_cluster_representatives(redundancy['clusters'], univariate_df)
rep_indices = [r['representative_idx'] for r in representatives]

forward_results = forward_selection(vectors, metadata, feature_names, rep_indices)
print(f"Selected: {len(forward_results['selected_names'])} features")
print(f"Accuracy: {forward_results['final_accuracy']:.1%}")
```

---

**Bottom line:** You spent effort building 454 features. Don't throw them away based on intuition. Run this analysis, let the data tell you what works, and you'll end up with a rigorous, defensible feature set.



