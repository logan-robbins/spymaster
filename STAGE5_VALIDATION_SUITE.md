# Stage 5 Validation Suite

## Overview

This document specifies comprehensive tests to validate the Gold layer output from Stage 5 (Setup Vectorization & Retrieval). The goal is to ensure data integrity, statistical sanity, index functionality, and retrieval quality before production deployment.

**Data Under Test:**
- Date range: June 4 – September 30, 2024
- Expected volume: ~50 episodes/day × 4 levels × ~85 trading days ≈ 17,000 episodes
- Vector dimension: 256

**Test Categories:**
1. Data Integrity
2. Statistical Sanity
3. Index Functionality
4. Retrieval Quality
5. Predictive Signal
6. Edge Cases & Data Leakage

---

## Section 1 — Data Integrity Tests

### 1.1 Completeness

```
TEST: All episodes from Silver have corresponding Gold vectors

STEPS:
1. Load all episode_ids from Stage 3 Silver output
2. Load all episode_ids from Gold metadata store
3. Compare sets

ASSERTIONS:
- silver_episode_ids == gold_episode_ids
- No missing episodes
- No extra episodes

FAILURE ACTION: List missing/extra episode_ids, investigate Silver→Gold pipeline
```

### 1.2 Vector Shape

```
TEST: All vectors have correct dimensionality

STEPS:
1. Load all vectors from setup_vectors/
2. Check shape

ASSERTIONS:
- vectors.shape[1] == 256
- vectors.shape[0] == len(metadata)
- vectors.dtype == float32 or float64

FAILURE ACTION: Identify malformed vectors, check extraction function
```

### 1.3 Vector-Metadata Alignment

```
TEST: Vector indices match metadata vector_ids

STEPS:
1. For each vector_id in metadata:
   - Extract vector at that index
   - Verify it corresponds to correct episode

SAMPLING: Test 100 random episodes
METHOD: 
   - Load episode from Silver
   - Re-extract setup vector
   - Compare to stored Gold vector

ASSERTIONS:
- np.allclose(recomputed_vector, stored_vector, rtol=1e-5)

FAILURE ACTION: Indicates index misalignment or extraction inconsistency
```

### 1.4 No NaN or Inf Values

```
TEST: Vectors contain no invalid values

STEPS:
1. Check all vectors for NaN
2. Check all vectors for Inf

ASSERTIONS:
- np.isnan(vectors).sum() == 0
- np.isinf(vectors).sum() == 0

FAILURE ACTION: Identify which features produce NaN/Inf, fix normalization or edge case handling
```

### 1.5 Metadata Completeness

```
TEST: All metadata fields are populated

REQUIRED FIELDS:
- vector_id (int, unique)
- episode_id (string, unique)
- date (date)
- symbol (string)
- level_type (string, one of 4 values)
- level_price (float, > 0)
- trigger_bar_ts (int, valid timestamp)
- approach_direction (int, -1 or +1)
- outcome (string, one of 5 values)
- outcome_score (float)

ASSERTIONS:
- No NULL values in required fields
- vector_id is sequential 0 to N-1
- episode_id is unique
- level_type ∈ {'PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW'}
- outcome ∈ {'STRONG_BREAK', 'WEAK_BREAK', 'CHOP', 'WEAK_BOUNCE', 'STRONG_BOUNCE'}
- approach_direction ∈ {-1, +1}

FAILURE ACTION: Identify records with missing/invalid fields
```

---

## Section 2 — Statistical Sanity Tests

### 2.1 Feature Distribution Analysis

```
TEST: Each vector dimension has reasonable distribution

STEPS:
For each dimension i in [0, 255]:
1. Extract column: values = vectors[:, i]
2. Compute statistics:
   - mean, std, min, max
   - skewness, kurtosis
   - % zeros
   - % unique values

ASSERTIONS:
- std > 1e-6 (not constant)
- |mean| < 100 (reasonable scale after normalization)
- |skewness| < 10 (not extremely skewed)
- % zeros < 95% (not degenerate)

OUTPUT: 
- Table of per-dimension statistics
- Flag dimensions that fail assertions
- Histograms for flagged dimensions

FAILURE ACTION: Investigate feature extraction for flagged dimensions
```

### 2.2 Normalization Verification

```
TEST: Normalized features are approximately standard normal

STEPS:
For features that should be z-scored:
1. Compute mean and std across all vectors
2. Compare to expected (0, 1)

ASSERTIONS:
- |mean| < 0.1 for z-scored features
- 0.8 < std < 1.2 for z-scored features

EXCEPTIONS:
- Binary features (expected mean ≠ 0)
- Bounded features like OBI (already in [-1, 1])
- Padded dimensions (expected = 0)

FAILURE ACTION: Recompute normalization parameters, check for data drift
```

### 2.3 Feature Correlation Matrix

```
TEST: Feature correlations are sensible

STEPS:
1. Compute correlation matrix (256 × 256)
2. Identify highly correlated pairs (|r| > 0.95)
3. Identify uncorrelated-but-should-be-correlated pairs

EXPECTED HIGH CORRELATIONS:
- deriv_dist_d1_w12 ↔ deriv_dist_d1_w36 (similar windows)
- setup_obi0_start ↔ setup_obi0_mean (related aggregations)
- cumul_flow_net_bid ↔ cumul_flow_imbal (component)

EXPECTED LOW CORRELATIONS:
- approach_direction ↔ level_polarity (independent)
- wall features ↔ trade features (different signals)

OUTPUT:
- Heatmap of correlation matrix
- List of top 20 correlated pairs
- List of unexpected correlations

FAILURE ACTION: Investigate feature extraction bugs if correlations don't match expectations
```

### 2.4 Outcome Distribution

```
TEST: Outcome labels have reasonable distribution

STEPS:
1. Count outcomes overall
2. Count outcomes by level_type
3. Count outcomes by approach_direction

EXPECTED DISTRIBUTION (overall):
- STRONG_BREAK: 10-20%
- WEAK_BREAK: 15-25%
- CHOP: 30-40%
- WEAK_BOUNCE: 15-25%
- STRONG_BOUNCE: 10-20%

ASSERTIONS:
- No single outcome > 60%
- No single outcome < 5%
- Distribution roughly similar across level_types

OUTPUT:
- Bar chart of outcome distribution
- Breakdown table by level_type
- Chi-square test for uniformity

FAILURE ACTION: If heavily skewed, review outcome labeling logic in Stage 2
```

### 2.5 Temporal Distribution

```
TEST: Episodes are distributed across the date range

STEPS:
1. Count episodes per date
2. Count episodes per level_type per date

ASSERTIONS:
- No date has 0 episodes (unless market closed)
- No date has > 200 episodes (unrealistic)
- Reasonable variation (std/mean between 0.3 and 1.5)

OUTPUT:
- Time series plot of daily episode counts
- Heatmap: date × level_type

FAILURE ACTION: Investigate dates with anomalous counts
```

---

## Section 3 — Index Functionality Tests

### 3.1 Index Load Test

```
TEST: FAISS indices load correctly

STEPS:
1. Load each index file:
   - pm_high_setups.index
   - pm_low_setups.index
   - or_high_setups.index
   - or_low_setups.index
2. Verify basic properties

ASSERTIONS:
- Index loads without error
- index.ntotal == expected count for that level_type
- index.d == 256

FAILURE ACTION: Rebuild corrupted index
```

### 3.2 Self-Query Test

```
TEST: Querying a vector returns itself as top match

STEPS:
For 100 random vectors:
1. Query the index with the vector
2. Check if top result (k=1) is the same vector

ASSERTIONS:
- Top result index == query vector's index
- Distance to self ≈ 0 (< 1e-6)

FAILURE ACTION: Index corruption or vector mismatch
```

### 3.3 K-Nearest Neighbors Sanity

```
TEST: KNN queries return sensible results

STEPS:
For 100 random queries with k=20:
1. Execute query
2. Verify results

ASSERTIONS:
- Exactly k results returned (or fewer if index is smaller)
- Distances are non-negative
- Distances are monotonically non-decreasing
- No duplicate indices in results
- All returned indices < index.ntotal

FAILURE ACTION: Index configuration issue
```

### 3.4 Cross-Index Isolation

```
TEST: Level-type indices are properly separated

STEPS:
1. Pick a PM_HIGH vector
2. Query PM_HIGH index → get results A
3. Query PM_LOW index → get results B
4. Verify A ≠ B (different episodes)

ASSERTIONS:
- Results from PM_HIGH index are all PM_HIGH episodes
- No cross-contamination between indices

FAILURE ACTION: Index build process mixed level types
```

### 3.5 Query Performance

```
TEST: Query latency is acceptable

STEPS:
Run 1000 queries, measure latency:
1. Single query latency
2. Batch query latency (100 vectors at once)

ASSERTIONS:
- Single query p50 < 10ms
- Single query p99 < 50ms
- Batch query p50 < 100ms

OUTPUT:
- Latency histogram
- p50, p95, p99 statistics

FAILURE ACTION: Consider index optimization (more centroids, GPU, etc.)
```

---

## Section 4 — Retrieval Quality Tests

### 4.1 Nearest Neighbor Similarity

```
TEST: Nearest neighbors have similar characteristics

STEPS:
For 100 random queries:
1. Get top-10 nearest neighbors
2. Compare metadata characteristics

METRICS:
- Same level_type: Should be 100% (we query per-level index)
- Same approach_direction: Expected > 60%
- Similar outcome: Expected > 30% (better than 20% random)
- Date spread: Should span multiple weeks (not just adjacent days)

OUTPUT:
- Distribution of approach_direction agreement
- Distribution of outcome agreement
- Histogram of date spans in result sets

FAILURE ACTION: If neighbors are too random, vectors aren't capturing meaningful signal
```

### 4.2 Distance Distribution

```
TEST: Distance distributions are informative

STEPS:
1. Compute all pairwise distances (or sample)
2. Compare to distances for top-k queries

ASSERTIONS:
- Top-k distances << random pair distances
- Clear separation between "similar" and "random"

OUTPUT:
- Histogram of all pairwise distances
- Histogram of top-10 query distances
- Overlap analysis

FAILURE ACTION: If distributions overlap heavily, vector space isn't meaningful
```

### 4.3 Cluster Analysis

```
TEST: Vectors form meaningful clusters

STEPS:
1. Run K-means with k=20 on all vectors
2. Analyze cluster composition

METRICS:
- Cluster purity by outcome
- Cluster purity by level_type
- Cluster purity by approach_direction
- Silhouette score

EXPECTED:
- Silhouette score > 0.1 (some structure)
- At least some clusters have outcome purity > 50%

OUTPUT:
- Cluster composition table
- t-SNE or UMAP visualization colored by outcome
- Silhouette score

FAILURE ACTION: If no cluster structure, features may not be discriminative
```

### 4.4 Feature Importance for Similarity

```
TEST: Identify which features drive similarity

STEPS:
1. For 100 query-result pairs:
   - Compute per-dimension contribution to L2 distance
2. Aggregate across pairs

OUTPUT:
- Ranked list of dimensions by contribution to distance
- Identify if padded dimensions (should be 0) contribute

ASSERTIONS:
- Padded dimensions contribute 0 to distance
- Top contributors are interpretable features

FAILURE ACTION: Unexpected features dominating may indicate normalization issues
```

---

## Section 5 — Predictive Signal Tests

### 5.1 Temporal Hold-Out Backtest

```
TEST: Retrieval system predicts outcomes better than random

SETUP:
- Train set: June 4 – August 31
- Test set: September 1 – September 30
- Ensure NO test data in index during test queries

STEPS:
For each test episode:
1. Query index (train data only) with k=20
2. Compute outcome distribution from neighbors
3. Record predicted vs actual outcome

METRICS:
- Accuracy: predicted_outcome == actual_outcome
- Top-2 Accuracy: actual in top 2 predicted
- Expected outcome score vs actual outcome score correlation
- Brier score for probability calibration

BASELINES:
- Random guess: 20% accuracy (5 classes)
- Prior distribution: predict most common outcome

ASSERTIONS:
- Accuracy > 25% (better than random)
- Top-2 Accuracy > 45%
- Correlation > 0.1 between expected and actual score

OUTPUT:
- Confusion matrix
- Accuracy by level_type
- Accuracy by outcome (which outcomes are easier to predict?)
- Calibration plot

FAILURE ACTION: If no better than random, features don't capture predictive signal
```

### 5.2 Similarity-Weighted vs Unweighted

```
TEST: Similarity weighting improves predictions

STEPS:
For test episodes:
1. Compute outcome distribution (unweighted)
2. Compute outcome distribution (similarity-weighted)
3. Compare prediction quality

ASSERTIONS:
- Weighted accuracy >= Unweighted accuracy
- Weighted Brier score <= Unweighted Brier score

FAILURE ACTION: If weighting hurts, distance metric may not reflect true similarity
```

### 5.3 K Sensitivity Analysis

```
TEST: Determine optimal k for retrieval

STEPS:
For k in [5, 10, 20, 50, 100]:
1. Run backtest with that k
2. Record accuracy metrics

OUTPUT:
- Accuracy vs k curve
- Optimal k value
- Diminishing returns analysis

EXPECTED:
- Accuracy increases then plateaus
- Optimal k typically 10-30

FAILURE ACTION: Informs production configuration
```

### 5.4 Outcome Score Regression

```
TEST: Predicted outcome_score correlates with actual

STEPS:
1. For test episodes, compute expected_score from neighbors
2. Compare to actual outcome_score

METRICS:
- Pearson correlation
- Spearman correlation
- MAE (Mean Absolute Error)
- Directional accuracy (sign match)

ASSERTIONS:
- Pearson r > 0.1
- Directional accuracy > 55%

OUTPUT:
- Scatter plot: expected vs actual
- Residual analysis

FAILURE ACTION: If no correlation, continuous scoring isn't working
```

### 5.5 By-Level-Type Performance

```
TEST: Performance is consistent across level types

STEPS:
Run backtest separately for each level_type

OUTPUT:
- Accuracy table by level_type
- Identify best/worst performing levels

EXPECTED:
- No level_type dramatically worse than others
- Some variation is acceptable (OR levels may have less data)

FAILURE ACTION: Investigate underperforming level types
```

---

## Section 6 — Edge Cases & Data Leakage Tests

### 6.1 Temporal Leakage Check

```
TEST: Nearest neighbors aren't just temporally adjacent episodes

STEPS:
For 100 queries:
1. Get top-10 neighbors
2. Check temporal distance to query

ASSERTIONS:
- Median days between query and neighbor > 5
- < 20% of neighbors are same-day
- < 5% of neighbors are same-week same-level

FAILURE ACTION: If neighbors are temporally clustered, may indicate:
- Data leakage (future data in index)
- Autocorrelation not captured (features too similar day-to-day)
```

### 6.2 Truncated Lookback Episodes

```
TEST: Truncated lookback episodes are handled correctly

STEPS:
1. Identify episodes where is_truncated_lookback == True
2. Verify their vectors are reasonable

ASSERTIONS:
- Truncated episodes have vectors (not excluded)
- Vector values are not NaN
- Truncated episodes cluster separately OR blend with full episodes

OUTPUT:
- Count of truncated episodes
- Comparison of truncated vs full episode distributions

FAILURE ACTION: May need special handling or exclusion of truncated episodes
```

### 6.3 Extended Forward Episodes

```
TEST: Extended episodes don't bias results

STEPS:
1. Identify episodes where is_extended_forward == True
2. Compare outcome distribution to non-extended

ASSERTIONS:
- Extended episodes don't have dramatically different outcome distribution
- Extension count doesn't correlate strongly with outcome

OUTPUT:
- Outcome distribution: extended vs not
- Correlation: extension_count ↔ outcome_score

FAILURE ACTION: If extended episodes skew results, may need normalization
```

### 6.4 Level Price Variation

```
TEST: Vectors are price-invariant

STEPS:
1. Group episodes by level_price ranges (e.g., 5800-5900, 5900-6000, ...)
2. Check if price range affects vector distributions

ASSERTIONS:
- Mean vectors are similar across price ranges
- Nearest neighbors span multiple price ranges

FAILURE ACTION: If price leaks into vectors, features aren't properly relative
```

### 6.5 Same-Day Cross-Level Test

```
TEST: Same-day episodes for different levels are distinguishable

STEPS:
1. Find days with episodes for all 4 levels
2. For each such day, check distances between level vectors

ASSERTIONS:
- Same-level episodes are closer than cross-level episodes
- PM_HIGH episode closer to other PM_HIGH than to PM_LOW

FAILURE ACTION: If same-day episodes are too similar regardless of level, feature extraction may have bugs
```

---

## Section 7 — Test Execution Plan

### 7.1 Test Priority

| Priority | Category | Run Frequency |
|----------|----------|---------------|
| P0 (Blocking) | Data Integrity | Every pipeline run |
| P0 (Blocking) | Index Functionality | Every pipeline run |
| P1 (Critical) | Statistical Sanity | Daily |
| P1 (Critical) | Predictive Signal | Weekly |
| P2 (Important) | Retrieval Quality | Weekly |
| P2 (Important) | Edge Cases | On schema change |

### 7.2 Test Artifacts

```
test_results/
  {date}/
    data_integrity/
      completeness_report.json
      vector_validation.json
    statistical_sanity/
      feature_distributions.parquet
      correlation_matrix.png
      outcome_distribution.png
    index_functionality/
      self_query_results.json
      query_latency.json
    retrieval_quality/
      cluster_analysis.png
      tsne_visualization.png
    predictive_signal/
      backtest_results.parquet
      confusion_matrix.png
      calibration_plot.png
    edge_cases/
      temporal_leakage_report.json
      truncated_episodes_analysis.json
    summary_report.html
```

### 7.3 Pass/Fail Criteria

**Pipeline BLOCKED if:**
- Any P0 test fails
- > 1% of vectors have NaN/Inf
- Index fails to load
- Self-query accuracy < 99%

**Warning issued if:**
- Any P1 test fails
- Backtest accuracy < 25%
- Outcome distribution severely skewed (any class > 50%)
- Temporal leakage detected (> 30% same-week neighbors)

---

## Section 8 — Test Implementation Notes

### 8.1 Data Loading

```python
# Load vectors
vectors = np.load('setup_vectors/vectors.npy')

# Load metadata
import sqlite3
conn = sqlite3.connect('metadata/setup_metadata.db')
metadata = pd.read_sql('SELECT * FROM setup_metadata', conn)

# Load indices
import faiss
indices = {
    'PM_HIGH': faiss.read_index('indices/pm_high_setups.index'),
    'PM_LOW': faiss.read_index('indices/pm_low_setups.index'),
    'OR_HIGH': faiss.read_index('indices/or_high_setups.index'),
    'OR_LOW': faiss.read_index('indices/or_low_setups.index'),
}
```

### 8.2 Backtest Implementation

```python
def temporal_backtest(vectors, metadata, train_end_date, k=20):
    """
    Run temporal hold-out backtest.
    """
    train_mask = metadata['date'] <= train_end_date
    test_mask = metadata['date'] > train_end_date
    
    train_vectors = vectors[train_mask]
    test_vectors = vectors[test_mask]
    train_metadata = metadata[train_mask]
    test_metadata = metadata[test_mask]
    
    # Build index on train only
    index = faiss.IndexFlatL2(256)
    index.add(train_vectors)
    
    results = []
    
    for i, (vec, row) in enumerate(zip(test_vectors, test_metadata.itertuples())):
        # Query
        distances, indices = index.search(vec.reshape(1, -1), k)
        
        # Get neighbor outcomes
        neighbor_outcomes = train_metadata.iloc[indices[0]]['outcome'].values
        neighbor_scores = train_metadata.iloc[indices[0]]['outcome_score'].values
        neighbor_distances = distances[0]
        
        # Compute predictions
        similarities = 1 / (1 + neighbor_distances)
        
        # Weighted outcome distribution
        outcome_weights = defaultdict(float)
        for outcome, sim in zip(neighbor_outcomes, similarities):
            outcome_weights[outcome] += sim
        total_weight = sum(outcome_weights.values())
        outcome_probs = {k: v/total_weight for k, v in outcome_weights.items()}
        
        predicted_outcome = max(outcome_probs, key=outcome_probs.get)
        expected_score = np.average(neighbor_scores, weights=similarities)
        
        results.append({
            'episode_id': row.episode_id,
            'actual_outcome': row.outcome,
            'predicted_outcome': predicted_outcome,
            'prediction_prob': outcome_probs.get(predicted_outcome, 0),
            'actual_score': row.outcome_score,
            'expected_score': expected_score,
            'level_type': row.level_type,
        })
    
    return pd.DataFrame(results)
```

### 8.3 Visualization Helpers

```python
def plot_confusion_matrix(results):
    outcomes = ['STRONG_BREAK', 'WEAK_BREAK', 'CHOP', 'WEAK_BOUNCE', 'STRONG_BOUNCE']
    cm = confusion_matrix(results['actual_outcome'], results['predicted_outcome'], labels=outcomes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=outcomes, yticklabels=outcomes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Outcome Prediction Confusion Matrix')
    plt.savefig('confusion_matrix.png')


def plot_tsne(vectors, metadata, color_by='outcome'):
    from sklearn.manifold import TSNE
    
    # Sample for speed
    sample_idx = np.random.choice(len(vectors), min(5000, len(vectors)), replace=False)
    sample_vectors = vectors[sample_idx]
    sample_metadata = metadata.iloc[sample_idx]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(sample_vectors)
    
    plt.figure(figsize=(12, 10))
    for outcome in sample_metadata[color_by].unique():
        mask = sample_metadata[color_by] == outcome
        plt.scatter(coords[mask, 0], coords[mask, 1], label=outcome, alpha=0.5, s=10)
    plt.legend()
    plt.title(f't-SNE Visualization (colored by {color_by})')
    plt.savefig(f'tsne_{color_by}.png')
```

---

## Appendix A — Expected Test Results (Benchmarks)

Based on similar trading ML systems, expected ranges for a healthy pipeline:

| Metric | Poor | Acceptable | Good |
|--------|------|------------|------|
| Backtest Accuracy | < 22% | 22-30% | > 30% |
| Top-2 Accuracy | < 40% | 40-50% | > 50% |
| Score Correlation | < 0.05 | 0.05-0.15 | > 0.15 |
| Silhouette Score | < 0.05 | 0.05-0.15 | > 0.15 |
| Query Latency p99 | > 100ms | 50-100ms | < 50ms |

---

## Appendix B — Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All vectors identical | Normalization collapsed features | Check for division by zero in z-score |
| NaN in vectors | Missing data in lookback | Improve NaN handling in Stage 3 |
| Self-query fails | Index/metadata misalignment | Rebuild index with verified ordering |
| No predictive signal | Features don't capture setup | Review feature selection, add new features |
| Temporal leakage | Same-day episodes too similar | Add time-decay or exclude same-week |
| One outcome dominates | Threshold miscalibration | Adjust outcome thresholds in Stage 2 |
| High query latency | Index not optimized | Use IVF index, tune nprobe |
