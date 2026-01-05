# Feature Analysis & Ablation Studies

## Overview

This document specifies experiments to understand which features drive predictive performance, identify redundancy, and optimize the 256-dimensional setup vector.

**Current Baseline:**
- 151 raw features → 256 dimensions (padded)
- Backtest accuracy: 44.5%
- Score correlation: 0.195
- Total episodes: 2,248

**Goals:**
1. Identify which feature groups contribute most to predictive power
2. Find and remove redundant or noisy features
3. Determine optimal dimensionality via PCA
4. Test alternative feature groupings for retrieval
5. Produce recommendations for v2 feature set

---

## Section 1 — Feature Group Definitions

### 1.1 Current Feature Groups

| Group | Features | Dimensions | Description |
|-------|----------|------------|-------------|
| `position` | 5 | 5 | Distance to level, side, alignment |
| `book_state` | 16 | 16 | OBI, CDI, spread, depth imbalances |
| `walls` | 8 | 8 | Wall z-scores, distances, indices |
| `flow_snapshot` | 10 | 10 | Cumulative and bar-level flows at trigger |
| `deriv_dist` | 8 | 8 | Distance velocity/acceleration |
| `deriv_imbal` | 16 | 16 | Imbalance derivatives |
| `deriv_depth` | 8 | 8 | Depth derivatives |
| `deriv_wall` | 8 | 8 | Wall z-score derivatives |
| `profile_traj` | 12 | 12 | Lookback trajectory summary |
| `profile_book` | 20 | 20 | Lookback book pressure summary |
| `profile_flow` | 16 | 16 | Lookback flow summary |
| `profile_wall` | 12 | 12 | Lookback wall summary |
| `recent` | 12 | 12 | Last-minute momentum |
| `padding` | 0 | 105 | Zero-padding to 256 |
| **Total** | **151** | **256** | |

### 1.2 Semantic Groupings (Alternative)

| Semantic Group | Included Feature Groups | Dimensions |
|----------------|------------------------|------------|
| `WHERE` | position | 5 |
| `TRAJECTORY` | deriv_dist, profile_traj, recent (dist) | 24 |
| `BOOK_PHYSICS` | book_state, deriv_imbal, deriv_depth, profile_book | 60 |
| `FLOW_PHYSICS` | flow_snapshot, profile_flow, recent (flow) | 34 |
| `WALLS` | walls, deriv_wall, profile_wall | 28 |

### 1.3 Feature Index Mapping

```python
FEATURE_GROUPS = {
    'position': list(range(0, 5)),
    'book_state': list(range(5, 21)),
    'walls': list(range(21, 29)),
    'flow_snapshot': list(range(29, 39)),
    'deriv_dist': list(range(39, 47)),
    'deriv_imbal': list(range(47, 63)),
    'deriv_depth': list(range(63, 71)),
    'deriv_wall': list(range(71, 79)),
    'profile_traj': list(range(79, 91)),
    'profile_book': list(range(91, 111)),
    'profile_flow': list(range(111, 127)),
    'profile_wall': list(range(127, 139)),
    'recent': list(range(139, 151)),
    'padding': list(range(151, 256)),
}

SEMANTIC_GROUPS = {
    'WHERE': FEATURE_GROUPS['position'],
    'TRAJECTORY': FEATURE_GROUPS['deriv_dist'] + FEATURE_GROUPS['profile_traj'] + [139, 140, 141, 142],
    'BOOK_PHYSICS': FEATURE_GROUPS['book_state'] + FEATURE_GROUPS['deriv_imbal'] + FEATURE_GROUPS['deriv_depth'] + FEATURE_GROUPS['profile_book'],
    'FLOW_PHYSICS': FEATURE_GROUPS['flow_snapshot'] + FEATURE_GROUPS['profile_flow'] + [143, 144, 145, 146, 147, 148],
    'WALLS': FEATURE_GROUPS['walls'] + FEATURE_GROUPS['deriv_wall'] + FEATURE_GROUPS['profile_wall'],
}
```

---

## Section 2 — Ablation Study Design

### 2.1 Leave-One-Group-Out Ablation

**Objective:** Measure importance of each feature group by removing it and measuring performance drop.

**Method:**
```
For each feature_group G in FEATURE_GROUPS:
    1. Create ablated vectors: zero out dimensions in G
    2. Rebuild FAISS index on train set
    3. Run temporal backtest
    4. Record: accuracy, score_correlation, top2_accuracy
    5. Compute: Δaccuracy = baseline_accuracy - ablated_accuracy
```

**Expected Output:**
| Group Removed | Accuracy | Δ Accuracy | Score Corr | Δ Score Corr | Importance Rank |
|---------------|----------|------------|------------|--------------|-----------------|
| position | ? | ? | ? | ? | ? |
| book_state | ? | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |

**Interpretation:**
- Large Δ = Important group (removing hurts performance)
- Δ ≈ 0 = Redundant or uninformative group
- Δ < 0 = Noisy group (removing helps!)

### 2.2 Leave-One-Group-In Ablation

**Objective:** Measure sufficiency of each feature group alone.

**Method:**
```
For each feature_group G in FEATURE_GROUPS:
    1. Create isolated vectors: zero out ALL dimensions EXCEPT G
    2. Rebuild FAISS index on train set
    3. Run temporal backtest
    4. Record metrics
```

**Expected Output:**
| Group Only | Accuracy | Score Corr | % of Baseline |
|------------|----------|------------|---------------|
| position only | ? | ? | ?% |
| book_state only | ? | ? | ?% |
| ... | ... | ... | ... |

**Interpretation:**
- High % of baseline with single group = Powerful standalone signal
- Low % = Group needs other features to be useful

### 2.3 Semantic Group Ablation

**Objective:** Test higher-level groupings.

**Experiments:**
```
1. WHERE only (5 dims)
2. TRAJECTORY only (24 dims)
3. BOOK_PHYSICS only (60 dims)
4. FLOW_PHYSICS only (34 dims)
5. WALLS only (28 dims)
6. WHERE + TRAJECTORY (29 dims)
7. WHERE + TRAJECTORY + BOOK_PHYSICS (89 dims)
8. WHERE + TRAJECTORY + FLOW_PHYSICS (63 dims)
9. All except WALLS (123 dims)
10. All except BOOK_PHYSICS (91 dims)
```

**Expected Output:**
| Configuration | Dims | Accuracy | Score Corr | Efficiency (Acc/Dim) |
|---------------|------|----------|------------|----------------------|
| Full baseline | 151 | 44.5% | 0.195 | 0.29 |
| WHERE only | 5 | ? | ? | ? |
| ... | ... | ... | ... | ... |

### 2.4 Progressive Feature Addition

**Objective:** Find minimal feature set that achieves ~90% of baseline performance.

**Method:**
```
1. Start with empty feature set
2. Greedily add the group that improves accuracy most
3. Repeat until accuracy plateaus or all groups added
4. Record accuracy at each step
```

**Expected Output:**
| Step | Groups Included | Dims | Accuracy | Marginal Gain |
|------|-----------------|------|----------|---------------|
| 0 | (none) | 0 | 20% | - |
| 1 | +??? | ? | ?% | +?% |
| 2 | +??? | ? | ?% | +?% |
| ... | ... | ... | ... | ... |

---

## Section 3 — PCA Analysis

### 3.1 Variance Explained Analysis

**Objective:** Determine intrinsic dimensionality of setup vectors.

**Method:**
```python
from sklearn.decomposition import PCA

# Fit PCA on training data (excluding padding)
pca = PCA(n_components=151)  # Max non-padded dims
pca.fit(train_vectors[:, :151])

# Compute cumulative variance explained
cumvar = np.cumsum(pca.explained_variance_ratio_)
```

**Expected Output:**
| Components | Cumulative Variance | 
|------------|---------------------|
| 10 | ?% |
| 20 | ?% |
| 30 | ?% |
| 50 | ?% |
| 75 | ?% |
| 100 | ?% |
| 151 | 100% |

**Key Questions:**
- How many components for 90% variance?
- How many components for 95% variance?
- Is there a clear "elbow" in the scree plot?

### 3.2 PCA Retrieval Performance

**Objective:** Test if PCA-reduced vectors maintain retrieval quality.

**Method:**
```
For n_components in [10, 20, 30, 50, 75, 100, 151]:
    1. Fit PCA on train vectors
    2. Transform train and test vectors
    3. Build FAISS index on transformed train vectors
    4. Run backtest
    5. Record accuracy, score_correlation
```

**Expected Output:**
| Components | Variance | Accuracy | Score Corr | vs Baseline |
|------------|----------|----------|------------|-------------|
| 10 | ?% | ?% | ? | ?% |
| 20 | ?% | ?% | ? | ?% |
| 30 | ?% | ?% | ? | ?% |
| 50 | ?% | ?% | ? | ?% |
| 75 | ?% | ?% | ? | ?% |
| 100 | ?% | ?% | ? | ?% |
| 151 | 100% | ?% | ? | ?% |

**Key Questions:**
- What's the minimum components for 95% of baseline accuracy?
- Does PCA improve or hurt performance? (regularization effect)

### 3.3 PCA Component Interpretation

**Objective:** Understand what principal components represent.

**Method:**
```python
# For top 10 components, examine loadings
for i in range(10):
    loadings = pca.components_[i]
    top_features = np.argsort(np.abs(loadings))[-10:]
    print(f"PC{i+1}: {[FEATURE_NAMES[j] for j in top_features]}")
```

**Expected Output:**
```
PC1 (23% var): deriv_dist_d1_w12, deriv_dist_d1_w36, setup_velocity_trend, ...
PC2 (15% var): state_obi0_eob, state_obi10_eob, setup_obi0_delta, ...
PC3 (11% var): wall_ask_maxz_eob, wall_bid_maxz_eob, setup_wall_imbal, ...
...
```

**Interpretation:**
- PC1 likely captures "approach velocity"
- PC2 likely captures "book imbalance"
- Etc.

### 3.4 PCA vs Original: Neighbor Comparison

**Objective:** Do PCA-reduced vectors find the same neighbors?

**Method:**
```
For 100 random queries:
    1. Find top-10 neighbors in original space
    2. Find top-10 neighbors in PCA-reduced space (50 dims)
    3. Compute overlap
```

**Metric:**
- Neighbor overlap @ k=10: What % of neighbors are the same?

**Expected:**
- High overlap (>70%) = PCA preserves similarity structure
- Low overlap (<50%) = PCA changes what "similar" means

---

## Section 4 — Individual Feature Analysis

### 4.1 Per-Feature Importance (Distance Contribution)

**Objective:** Which features contribute most to L2 distance between vectors?

**Method:**
```python
# For 1000 random pairs, compute per-dimension distance contribution
contributions = np.zeros(151)
for i in range(1000):
    v1, v2 = random_pair()
    squared_diff = (v1[:151] - v2[:151]) ** 2
    contributions += squared_diff

contributions /= contributions.sum()  # Normalize to %
```

**Expected Output:**
| Feature | % Distance Contribution | Rank |
|---------|------------------------|------|
| deriv_dist_d1_w12 | ?% | ? |
| state_obi0_eob | ?% | ? |
| ... | ... | ... |

**Interpretation:**
- High % = Feature strongly influences similarity
- Low % = Feature has minimal impact on retrieval

### 4.2 Per-Feature Predictive Correlation

**Objective:** Which individual features correlate with outcome_score?

**Method:**
```python
correlations = {}
for i, name in enumerate(FEATURE_NAMES):
    r = np.corrcoef(vectors[:, i], metadata['outcome_score'])[0, 1]
    correlations[name] = r
```

**Expected Output:**
| Feature | Correlation with Outcome | |r| Rank |
|---------|-------------------------|----------|
| deriv_dist_d1_w12 | ? | ? |
| setup_obi0_delta | ? | ? |
| ... | ... | ... |

**Interpretation:**
- |r| > 0.1 = Meaningful predictive feature
- |r| < 0.02 = Likely noise

### 4.3 Feature Redundancy Analysis

**Objective:** Identify highly correlated feature pairs.

**Method:**
```python
corr_matrix = np.corrcoef(vectors[:, :151].T)
redundant_pairs = []
for i in range(151):
    for j in range(i+1, 151):
        if abs(corr_matrix[i, j]) > 0.9:
            redundant_pairs.append((FEATURE_NAMES[i], FEATURE_NAMES[j], corr_matrix[i, j]))
```

**Expected Output:**
| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| deriv_dist_d1_w12 | deriv_dist_d1_w36 | 0.95 |
| setup_obi0_mean | setup_obi0_end | 0.92 |
| ... | ... | ... |

**Action:** Consider removing one of each highly correlated pair.

### 4.4 Feature Variance Analysis

**Objective:** Identify low-variance (near-constant) features.

**Method:**
```python
variances = np.var(vectors[:, :151], axis=0)
low_var_features = [FEATURE_NAMES[i] for i in np.where(variances < 0.01)[0]]
```

**Action:** Low-variance features add minimal information, consider removing.

---

## Section 5 — Alternative Distance Metrics

### 5.1 Cosine vs L2 Distance

**Objective:** Test if cosine similarity outperforms L2.

**Method:**
```python
# Normalize vectors to unit length
normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Build cosine index (equivalent to L2 on normalized vectors)
index_cosine = faiss.IndexFlatIP(256)  # Inner product
index_cosine.add(normalized)

# Run backtest with cosine similarity
```

**Expected Output:**
| Metric | L2 Distance | Cosine Similarity |
|--------|-------------|-------------------|
| Accuracy | 44.5% | ?% |
| Score Corr | 0.195 | ? |
| Top-2 Acc | ?% | ?% |

### 5.2 Weighted Distance

**Objective:** Test if weighting features by importance improves retrieval.

**Method:**
```python
# Weight by predictive correlation (from 4.2)
weights = np.abs(feature_correlations)
weights = weights / weights.sum() * 151  # Normalize

# Apply weights
weighted_vectors = vectors[:, :151] * np.sqrt(weights)

# Rebuild index and test
```

**Expected Output:**
| Metric | Unweighted | Correlation-Weighted |
|--------|------------|---------------------|
| Accuracy | 44.5% | ?% |
| Score Corr | 0.195 | ? |

### 5.3 Group-Weighted Distance

**Objective:** Test weighting by feature group importance (from ablation).

**Method:**
```python
# Weight groups by importance from ablation study
group_weights = {
    'position': ablation_importance['position'],
    'book_state': ablation_importance['book_state'],
    # ... etc
}

# Apply group weights to vectors
for group, indices in FEATURE_GROUPS.items():
    weighted_vectors[:, indices] *= np.sqrt(group_weights[group])
```

---

## Section 6 — Feature Selection Experiments

### 6.1 Correlation-Based Selection

**Objective:** Remove redundant features.

**Method:**
```
1. Start with all 151 features
2. For each pair with |r| > 0.9:
   - Keep the feature with higher |correlation with outcome|
   - Remove the other
3. Test reduced feature set
```

**Expected:** 10-30 features removed, minimal accuracy loss.

### 6.2 Variance Threshold Selection

**Objective:** Remove near-constant features.

**Method:**
```
1. Compute variance of each feature
2. Remove features with variance < 0.01 (after normalization)
3. Test reduced feature set
```

### 6.3 Univariate Selection (SelectKBest)

**Objective:** Keep only features with strongest individual predictive power.

**Method:**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Convert outcome to numeric
outcome_numeric = metadata['outcome'].map({
    'STRONG_BOUNCE': 0, 'WEAK_BOUNCE': 1, 'CHOP': 2, 'WEAK_BREAK': 3, 'STRONG_BREAK': 4
})

for k in [20, 40, 60, 80, 100]:
    selector = SelectKBest(f_classif, k=k)
    selected = selector.fit_transform(vectors[:, :151], outcome_numeric)
    # Test selected feature set
```

**Expected Output:**
| K Features | Accuracy | Score Corr | vs Baseline |
|------------|----------|------------|-------------|
| 20 | ?% | ? | ?% |
| 40 | ?% | ? | ?% |
| 60 | ?% | ? | ?% |
| 80 | ?% | ? | ?% |
| 100 | ?% | ? | ?% |

### 6.4 Recursive Feature Elimination

**Objective:** Iteratively remove least important features.

**Method:**
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Use logistic regression to rank features
model = LogisticRegression(max_iter=1000, multi_class='multinomial')

for n_features in [20, 40, 60, 80, 100]:
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(train_vectors[:, :151], train_outcomes)
    
    # Get selected feature indices
    selected_indices = np.where(rfe.support_)[0]
    
    # Test with retrieval
```

### 6.5 LASSO-Based Selection

**Objective:** Use L1 regularization to identify important features.

**Method:**
```python
from sklearn.linear_model import LassoCV

# Predict outcome_score (regression)
lasso = LassoCV(cv=5)
lasso.fit(train_vectors[:, :151], train_metadata['outcome_score'])

# Non-zero coefficients indicate selected features
selected = np.where(np.abs(lasso.coef_) > 1e-5)[0]
print(f"LASSO selected {len(selected)} features")
```

---

## Section 7 — Experiment Execution Plan

### 7.1 Execution Order

| Phase | Experiments | Priority | Compute Time |
|-------|-------------|----------|--------------|
| 1 | PCA variance analysis (3.1) | P0 | 5 min |
| 1 | Feature correlation matrix (4.3) | P0 | 5 min |
| 1 | Feature variance analysis (4.4) | P0 | 2 min |
| 2 | Leave-one-group-out ablation (2.1) | P0 | 2 hours |
| 2 | Per-feature predictive correlation (4.2) | P1 | 10 min |
| 3 | PCA retrieval performance (3.2) | P1 | 1 hour |
| 3 | Leave-one-group-in ablation (2.2) | P1 | 2 hours |
| 4 | Semantic group ablation (2.3) | P1 | 2 hours |
| 4 | Progressive feature addition (2.4) | P2 | 3 hours |
| 5 | Distance metric comparison (5.1, 5.2) | P2 | 1 hour |
| 5 | Feature selection experiments (6.x) | P2 | 2 hours |

### 7.2 Compute Requirements

- All experiments can run on CPU
- Memory: ~8GB for full vector set
- Parallelizable: Ablation experiments can run independently

### 7.3 Output Artifacts

```
ablation_results/
  leave_one_out/
    results.csv              # Group, Accuracy, ScoreCorr, Delta
    importance_ranking.json  # Ordered list by importance
  leave_one_in/
    results.csv
  semantic_groups/
    results.csv
  progressive_addition/
    results.csv
    addition_order.json

pca_analysis/
  scree_plot.png
  cumulative_variance.csv
  component_loadings.csv
  retrieval_by_components.csv
  pca_model.pkl

feature_analysis/
  correlation_matrix.png
  correlation_matrix.csv
  redundant_pairs.csv
  low_variance_features.json
  predictive_correlations.csv
  distance_contributions.csv

feature_selection/
  correlation_based/
    selected_features.json
    results.csv
  selectkbest/
    results_by_k.csv
  rfe/
    results_by_n.csv
  lasso/
    selected_features.json
    coefficients.csv

summary/
  recommendations.md         # Final recommendations
  optimal_feature_set.json   # Recommended features for v2
  comparison_table.csv       # All experiments side-by-side
```

---

## Section 8 — Analysis Templates

### 8.1 Ablation Results Template

```python
def run_ablation_leave_one_out(vectors, metadata, feature_groups):
    """Run leave-one-group-out ablation."""
    
    results = []
    baseline = run_backtest(vectors, metadata)
    
    for group_name, indices in feature_groups.items():
        # Zero out this group
        ablated = vectors.copy()
        ablated[:, indices] = 0
        
        # Run backtest
        metrics = run_backtest(ablated, metadata)
        
        results.append({
            'group': group_name,
            'dims_removed': len(indices),
            'accuracy': metrics['accuracy'],
            'delta_accuracy': baseline['accuracy'] - metrics['accuracy'],
            'score_corr': metrics['score_correlation'],
            'delta_score_corr': baseline['score_correlation'] - metrics['score_correlation'],
            'importance': baseline['accuracy'] - metrics['accuracy'],  # Higher = more important
        })
    
    return pd.DataFrame(results).sort_values('importance', ascending=False)
```

### 8.2 PCA Analysis Template

```python
def run_pca_analysis(vectors, metadata):
    """Run comprehensive PCA analysis."""
    
    from sklearn.decomposition import PCA
    
    # Remove padding
    X = vectors[:, :151]
    
    # Full PCA
    pca = PCA(n_components=151)
    pca.fit(X)
    
    # Variance analysis
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Find elbow points
    n_90 = np.argmax(cumvar >= 0.90) + 1
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1
    
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    print(f"Components for 99% variance: {n_99}")
    
    # Test retrieval at different component counts
    retrieval_results = []
    for n_comp in [10, 20, 30, 50, 75, 100, 151]:
        pca_n = PCA(n_components=n_comp)
        X_reduced = pca_n.fit_transform(X)
        
        metrics = run_backtest_with_vectors(X_reduced, metadata)
        retrieval_results.append({
            'n_components': n_comp,
            'variance_explained': cumvar[n_comp-1],
            'accuracy': metrics['accuracy'],
            'score_correlation': metrics['score_correlation'],
        })
    
    return {
        'pca_model': pca,
        'cumulative_variance': cumvar,
        'n_for_90': n_90,
        'n_for_95': n_95,
        'retrieval_results': pd.DataFrame(retrieval_results),
    }
```

### 8.3 Feature Importance Template

```python
def compute_feature_importance(vectors, metadata):
    """Compute multiple importance metrics per feature."""
    
    X = vectors[:, :151]
    y_score = metadata['outcome_score'].values
    y_class = metadata['outcome'].values
    
    importance = []
    
    for i in range(151):
        # Correlation with outcome score
        corr = np.corrcoef(X[:, i], y_score)[0, 1]
        
        # Variance
        var = np.var(X[:, i])
        
        # Distance contribution (sample)
        sample_idx = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
        dist_contrib = 0
        for j in range(0, len(sample_idx), 2):
            if j+1 < len(sample_idx):
                dist_contrib += (X[sample_idx[j], i] - X[sample_idx[j+1], i]) ** 2
        
        importance.append({
            'feature_idx': i,
            'feature_name': FEATURE_NAMES[i],
            'outcome_correlation': corr,
            'abs_correlation': abs(corr),
            'variance': var,
            'distance_contribution': dist_contrib,
        })
    
    return pd.DataFrame(importance)
```

---

## Section 9 — Interpretation Guidelines

### 9.1 Ablation Results Interpretation

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| Removing group drops accuracy > 5pp | Critical feature group | Keep, possibly expand |
| Removing group drops accuracy 2-5pp | Important feature group | Keep |
| Removing group drops accuracy 0-2pp | Marginal contribution | Consider simplifying |
| Removing group improves accuracy | Noisy feature group | Remove in v2 |

### 9.2 PCA Results Interpretation

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| 30 components achieve 95% baseline | High redundancy | Use PCA(30) in production |
| 100 components needed for 95% baseline | Features are diverse | Keep more features |
| PCA improves over raw features | Regularization helps | Use PCA |
| PCA hurts performance | Original features better | Don't use PCA |

### 9.3 Feature Selection Interpretation

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| 40 features achieve 90% baseline | Heavy redundancy | Simplify to 40 |
| LASSO selects <50 features | Most features are noise | Use LASSO selection |
| All methods converge on similar set | Robust signal | High confidence in selected features |

---

## Section 10 — Recommendations Template

After all experiments, produce a final recommendations document:

```markdown
# Feature Analysis Recommendations

## Executive Summary
- Baseline: 151 features, 44.5% accuracy, 0.195 score correlation
- Recommended: [N] features, [X]% accuracy, [Y] score correlation
- Dimensionality reduction: [X]% with [Z]% of baseline performance

## Key Findings

### Most Important Feature Groups (Keep)
1. [Group 1]: [Δ accuracy when removed]
2. [Group 2]: [Δ accuracy when removed]
3. [Group 3]: [Δ accuracy when removed]

### Least Important Feature Groups (Consider Removing)
1. [Group 1]: [Δ accuracy when removed]
2. [Group 2]: [Δ accuracy when removed]

### Redundant Features (Remove One of Each Pair)
1. [Feature A] ↔ [Feature B] (r = 0.95) → Keep [A]
2. [Feature C] ↔ [Feature D] (r = 0.92) → Keep [C]

### Low-Variance Features (Remove)
1. [Feature]: variance = 0.001
2. [Feature]: variance = 0.003

## Recommended v2 Feature Set

### Option A: Minimal (N features)
- [List of features]
- Expected accuracy: X%
- Expected score correlation: Y

### Option B: Balanced (N features)
- [List of features]
- Expected accuracy: X%
- Expected score correlation: Y

### Option C: PCA Reduced (N components)
- Expected accuracy: X%
- Expected score correlation: Y
- Variance explained: Z%

## Next Steps
1. Implement v2 feature set
2. Retrain normalization parameters
3. Rebuild indices
4. Validate on held-out data
```

---

## Appendix A — Feature Names Reference

```python
FEATURE_NAMES = [
    # Position (0-4)
    'approach_dist_to_level_pts_eob',
    'approach_side_of_level_eob',
    'approach_alignment_eob',
    'level_polarity',
    'is_standard_approach',
    
    # Book State (5-20)
    'state_obi0_eob',
    'state_obi10_eob',
    'state_spread_pts_eob',
    'state_cdi_p0_1_eob',
    'state_cdi_p1_2_eob',
    'state_cdi_p2_3_eob',
    'state_cdi_p3_5_eob',
    'state_cdi_p5_10_eob',
    'lvl_depth_imbal_eob',
    'lvl_cdi_p0_1_eob',
    'lvl_cdi_p1_2_eob',
    'lvl_cdi_p2_5_eob',
    'depth_bid10_qty_eob',
    'depth_ask10_qty_eob',
    'lvl_depth_above_qty_eob',
    'lvl_depth_below_qty_eob',
    
    # Walls (21-28)
    'wall_bid_maxz_eob',
    'wall_ask_maxz_eob',
    'wall_bid_maxz_levelidx_eob',
    'wall_ask_maxz_levelidx_eob',
    'wall_bid_nearest_strong_dist_pts_eob',
    'wall_ask_nearest_strong_dist_pts_eob',
    'wall_bid_nearest_strong_levelidx_eob',
    'wall_ask_nearest_strong_levelidx_eob',
    
    # Flow Snapshot (29-38)
    'cumul_signed_trade_vol',
    'cumul_flow_imbal',
    'cumul_flow_net_bid',
    'cumul_flow_net_ask',
    'lvl_flow_toward_net_sum',
    'lvl_flow_away_net_sum',
    'lvl_flow_toward_away_imbal_sum',
    'trade_signed_vol_sum',
    'trade_aggbuy_vol_sum',
    'trade_aggsell_vol_sum',
    
    # Derivative - Distance (39-46)
    'deriv_dist_d1_w3',
    'deriv_dist_d1_w12',
    'deriv_dist_d1_w36',
    'deriv_dist_d1_w72',
    'deriv_dist_d2_w3',
    'deriv_dist_d2_w12',
    'deriv_dist_d2_w36',
    'deriv_dist_d2_w72',
    
    # Derivative - Imbalance (47-62)
    'deriv_obi0_d1_w12',
    'deriv_obi0_d1_w36',
    'deriv_obi10_d1_w12',
    'deriv_obi10_d1_w36',
    'deriv_cdi01_d1_w12',
    'deriv_cdi01_d1_w36',
    'deriv_cdi12_d1_w12',
    'deriv_cdi12_d1_w36',
    'deriv_obi0_d2_w12',
    'deriv_obi0_d2_w36',
    'deriv_obi10_d2_w12',
    'deriv_obi10_d2_w36',
    'deriv_cdi01_d2_w12',
    'deriv_cdi01_d2_w36',
    'deriv_cdi12_d2_w12',
    'deriv_cdi12_d2_w36',
    
    # Derivative - Depth (63-70)
    'deriv_dbid10_d1_w12',
    'deriv_dbid10_d1_w36',
    'deriv_dask10_d1_w12',
    'deriv_dask10_d1_w36',
    'deriv_dbelow01_d1_w12',
    'deriv_dbelow01_d1_w36',
    'deriv_dabove01_d1_w12',
    'deriv_dabove01_d1_w36',
    
    # Derivative - Wall (71-78)
    'deriv_wbidz_d1_w12',
    'deriv_wbidz_d1_w36',
    'deriv_waskz_d1_w12',
    'deriv_waskz_d1_w36',
    'deriv_wbidz_d2_w12',
    'deriv_wbidz_d2_w36',
    'deriv_waskz_d2_w12',
    'deriv_waskz_d2_w36',
    
    # Profile - Trajectory (79-90)
    'setup_start_dist_pts',
    'setup_min_dist_pts',
    'setup_max_dist_pts',
    'setup_dist_range_pts',
    'setup_approach_bars',
    'setup_retreat_bars',
    'setup_approach_ratio',
    'setup_early_velocity',
    'setup_mid_velocity',
    'setup_late_velocity',
    'setup_velocity_trend',
    'setup_velocity_std',
    
    # Profile - Book (91-110)
    'setup_obi0_start',
    'setup_obi0_end',
    'setup_obi0_delta',
    'setup_obi0_min',
    'setup_obi0_max',
    'setup_obi0_mean',
    'setup_obi0_std',
    'setup_obi10_start',
    'setup_obi10_end',
    'setup_obi10_delta',
    'setup_obi10_min',
    'setup_obi10_max',
    'setup_obi10_mean',
    'setup_obi10_std',
    'setup_cdi01_mean',
    'setup_cdi01_std',
    'setup_lvl_depth_imbal_mean',
    'setup_lvl_depth_imbal_std',
    'setup_lvl_depth_imbal_trend',
    'setup_spread_mean',
    
    # Profile - Flow (111-126)
    'setup_total_trade_vol',
    'setup_total_signed_vol',
    'setup_trade_imbal_pct',
    'setup_flow_imbal_total',
    'setup_flow_toward_total',
    'setup_flow_away_total',
    'setup_flow_toward_away_ratio',
    'setup_flow_net_bid_total',
    'setup_flow_net_ask_total',
    'setup_trade_vol_early',
    'setup_trade_vol_mid',
    'setup_trade_vol_late',
    'setup_trade_vol_trend',
    'setup_signed_vol_early',
    'setup_signed_vol_mid',
    'setup_signed_vol_late',
    
    # Profile - Wall (127-138)
    'setup_bid_wall_max_z',
    'setup_ask_wall_max_z',
    'setup_bid_wall_bars',
    'setup_ask_wall_bars',
    'setup_wall_imbal',
    'setup_bid_wall_mean_z',
    'setup_ask_wall_mean_z',
    'setup_bid_wall_closest_dist_min',
    'setup_ask_wall_closest_dist_min',
    'setup_wall_appeared_bid',
    'setup_wall_appeared_ask',
    'setup_wall_disappeared_bid',
    
    # Recent (139-150)
    'recent_dist_delta',
    'recent_obi0_delta',
    'recent_obi10_delta',
    'recent_cdi01_delta',
    'recent_trade_vol',
    'recent_signed_vol',
    'recent_flow_toward',
    'recent_flow_away',
    'recent_aggbuy_vol',
    'recent_aggsell_vol',
    'recent_bid_depth_delta',
    'recent_ask_depth_delta',
]

assert len(FEATURE_NAMES) == 151
```
