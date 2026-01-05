# Stage 5 — Setup Vectorization & Similarity Retrieval

**STATUS: COMPLETE**

## Overview

This stage transforms multi-bar episodes into **fixed-length setup vectors** suitable for similarity search. Given a live setup approaching a key level, the system retrieves historically similar setups and their outcomes.

**Input:** Stage 3 output — episodes with ~400 features per 5-second bar

**Output:**
1. **Setup vectors** — One fixed-length vector per episode capturing the setup signature
2. **FAISS index** — Indexed vectors for fast approximate nearest neighbor search
3. **Metadata store** — Episode IDs, outcomes, and context for retrieved results

**Core Question Answered:**
> "This setup approaching PM_HIGH from below looks like setups X, Y, Z from the past. Those resulted in: 60% STRONG_BREAK, 25% WEAK_BREAK, 15% CHOP. Average outcome_score: +1.8 points."

---

## Section 0 — Design Principles

### 0.1 What Makes Two Setups "Similar"?

Two setups are similar if they share:

1. **Approach geometry** — Same direction, similar velocity/acceleration profile
2. **Book pressure** — Similar bid/ask imbalance patterns relative to the level
3. **Flow dynamics** — Similar liquidity addition/removal patterns during approach
4. **Wall structure** — Similar large resting order positions relative to the level
5. **Trade pressure** — Similar aggressive buying/selling patterns

They do NOT need to share:
- Absolute price (6800 vs 5400 — irrelevant)
- Absolute volume (varies by market regime — normalize)
- Calendar time (January vs July — irrelevant)
- Level type (PM_HIGH vs OR_LOW — can be filtered, but physics are comparable)

### 0.2 Vector Composition Strategy

The setup vector is computed at **trigger time** (bar where price enters ±1pt of level) and captures:

| Component | What It Captures | Source |
|-----------|-----------------|--------|
| Snapshot features | Book state at trigger moment | Trigger bar EOB features |
| Trajectory features | How price approached | Stage 3 derivatives at trigger |
| Cumulative features | What accumulated during approach | Stage 3 cumulatives at trigger |
| Profile features | Shape of approach over time | Aggregations over lookback window |

### 0.3 Dimensionality Target

- Raw features per episode: ~400 features × ~280 bars = ~112,000 values
- Target vector dimension: **256-512 dimensions**
- Compression ratio: ~200-400x

This is achieved through:
1. Selecting trigger-bar features only (not all bars)
2. Computing window aggregations (not per-bar values)
3. Optional: PCA or autoencoder for final compression

---

## Section 1 — Feature Selection for Setup Vector

### 1.1 Trigger-Bar Snapshot (Direct Inclusion)

These features from the **trigger bar only** go directly into the vector:

#### Position Features (5 dims)
```
approach_dist_to_level_pts_eob      # Should be near 0 at trigger
approach_side_of_level_eob          # +1 or -1
approach_alignment_eob              # +1 (standard) or -1 (retest)
level_polarity                      # +1 (high) or -1 (low)
is_standard_approach                # 1 or 0
```

#### Book State Features (16 dims)
```
state_obi0_eob                      # Top-of-book imbalance
state_obi10_eob                     # Full book imbalance
state_spread_pts_eob                # Spread width
state_cdi_p0_1_eob                  # Near-level cross-depth imbalance
state_cdi_p1_2_eob
state_cdi_p2_3_eob
state_cdi_p3_5_eob
state_cdi_p5_10_eob
lvl_depth_imbal_eob                 # Level-relative depth imbalance
lvl_cdi_p0_1_eob
lvl_cdi_p1_2_eob
lvl_cdi_p2_5_eob
depth_bid10_qty_eob                 # Total depths (normalized later)
depth_ask10_qty_eob
lvl_depth_above_qty_eob
lvl_depth_below_qty_eob
```

#### Wall Features (8 dims)
```
wall_bid_maxz_eob
wall_ask_maxz_eob
wall_bid_maxz_levelidx_eob
wall_ask_maxz_levelidx_eob
wall_bid_nearest_strong_dist_pts_eob
wall_ask_nearest_strong_dist_pts_eob
wall_bid_nearest_strong_levelidx_eob
wall_ask_nearest_strong_levelidx_eob
```

#### Flow Summary at Trigger (10 dims)
```
cumul_signed_trade_vol              # At trigger bar
cumul_flow_imbal
cumul_flow_net_bid
cumul_flow_net_ask
lvl_flow_toward_net_sum             # Bar's flow toward level
lvl_flow_away_net_sum
lvl_flow_toward_away_imbal_sum
trade_signed_vol_sum                # Trigger bar's trades
trade_aggbuy_vol_sum
trade_aggsell_vol_sum
```

**Trigger-bar snapshot subtotal: ~39 dimensions**

### 1.2 Trajectory Features at Trigger (Derivatives)

Stage 3 derivatives computed at the trigger bar:

#### Distance Derivatives (8 dims)
```
deriv_dist_d1_w3                    # Velocity toward level
deriv_dist_d1_w12
deriv_dist_d1_w36
deriv_dist_d1_w72
deriv_dist_d2_w3                    # Acceleration
deriv_dist_d2_w12
deriv_dist_d2_w36
deriv_dist_d2_w72
```

#### Imbalance Derivatives (16 dims)
```
deriv_obi0_d1_w12
deriv_obi0_d1_w36
deriv_obi10_d1_w12
deriv_obi10_d1_w36
deriv_cdi01_d1_w12
deriv_cdi01_d1_w36
deriv_cdi12_d1_w12
deriv_cdi12_d1_w36
deriv_obi0_d2_w12
deriv_obi0_d2_w36
deriv_obi10_d2_w12
deriv_obi10_d2_w36
deriv_cdi01_d2_w12
deriv_cdi01_d2_w36
deriv_cdi12_d2_w12
deriv_cdi12_d2_w36
```

#### Depth Derivatives (8 dims)
```
deriv_dbid10_d1_w12
deriv_dbid10_d1_w36
deriv_dask10_d1_w12
deriv_dask10_d1_w36
deriv_dbelow01_d1_w12
deriv_dbelow01_d1_w36
deriv_dabove01_d1_w12
deriv_dabove01_d1_w36
```

#### Wall Derivatives (8 dims)
```
deriv_wbidz_d1_w12
deriv_wbidz_d1_w36
deriv_waskz_d1_w12
deriv_waskz_d1_w36
deriv_wbidz_d2_w12
deriv_wbidz_d2_w36
deriv_waskz_d2_w12
deriv_waskz_d2_w36
```

**Derivative subtotal: ~40 dimensions**

### 1.3 Lookback Profile Features (Window Aggregations)

Computed over the full 15-minute lookback window:

#### Trajectory Profile (12 dims)
```
setup_start_dist_pts                # Where approach started
setup_min_dist_pts                  # Closest approach before trigger
setup_max_dist_pts                  # Furthest during lookback
setup_dist_range_pts                # Total price range
setup_approach_bars                 # Bars moving toward level
setup_retreat_bars                  # Bars moving away
setup_approach_ratio                # approach / total
setup_early_velocity                # First third velocity
setup_mid_velocity                  # Middle third
setup_late_velocity                 # Final third
setup_velocity_trend                # late - early
setup_velocity_std                  # Consistency of approach (NEW)
```

#### Book Pressure Profile (20 dims)
```
setup_obi0_start
setup_obi0_end
setup_obi0_delta
setup_obi0_min
setup_obi0_max
setup_obi0_mean                     # NEW
setup_obi0_std                      # NEW
setup_obi10_start
setup_obi10_end
setup_obi10_delta
setup_obi10_min
setup_obi10_max
setup_obi10_mean
setup_obi10_std
setup_cdi01_mean                    # NEW: avg near-level imbalance
setup_cdi01_std
setup_lvl_depth_imbal_mean          # NEW: avg level-relative imbalance
setup_lvl_depth_imbal_std
setup_lvl_depth_imbal_trend         # NEW: end - start
setup_spread_mean                   # NEW: avg spread during approach
```

#### Flow Profile (16 dims)
```
setup_total_trade_vol
setup_total_signed_vol
setup_trade_imbal_pct
setup_flow_imbal_total
setup_flow_toward_total             # NEW: total flow toward level
setup_flow_away_total               # NEW: total flow away from level
setup_flow_toward_away_ratio        # NEW
setup_flow_net_bid_total
setup_flow_net_ask_total
setup_trade_vol_early               # NEW: trade vol in first third
setup_trade_vol_mid
setup_trade_vol_late
setup_trade_vol_trend               # NEW: late - early (accelerating?)
setup_signed_vol_early
setup_signed_vol_mid
setup_signed_vol_late
```

#### Wall Profile (12 dims)
```
setup_bid_wall_max_z
setup_ask_wall_max_z
setup_bid_wall_bars
setup_ask_wall_bars
setup_wall_imbal                    # ask_bars - bid_bars
setup_bid_wall_mean_z               # NEW: avg bid wall intensity
setup_ask_wall_mean_z
setup_bid_wall_closest_dist_min     # NEW: closest bid wall got
setup_ask_wall_closest_dist_min
setup_wall_появился_bid             # NEW: did bid wall appear? (bool)
setup_wall_appeared_ask
setup_wall_disappeared_bid          # NEW: did bid wall disappear?
```

**Lookback profile subtotal: ~60 dimensions**

### 1.4 Recent Momentum Features (Last 1-Minute Before Trigger)

Captures the "final approach" character:

```
recent_dist_delta                   # Distance change in last 12 bars
recent_obi0_delta                   # OBI0 change
recent_obi10_delta
recent_cdi01_delta
recent_trade_vol                    # Trade volume in last minute
recent_signed_vol                   # Signed trade vol
recent_flow_toward                  # Flow toward level
recent_flow_away
recent_aggbuy_vol
recent_aggsell_vol
recent_bid_depth_delta              # Bid depth change
recent_ask_depth_delta
```

**Recent momentum subtotal: 12 dimensions**

### 1.5 Total Raw Vector Dimensions

| Component | Dimensions |
|-----------|------------|
| Trigger-bar snapshot | 39 |
| Derivatives at trigger | 40 |
| Lookback profile | 60 |
| Recent momentum | 12 |
| **Total** | **151** |

After normalization and optional expansion: target **256 dimensions** for FAISS index.

---

## Section 2 — Normalization Strategy

### 2.1 Why Normalization Matters

Similarity search uses distance metrics (L2, cosine). Features must be on comparable scales or high-variance features dominate.

### 2.2 Normalization Approaches by Feature Type

| Feature Type | Normalization | Rationale |
|--------------|---------------|-----------|
| Imbalances (OBI, CDI) | None (already [-1, +1]) | Naturally bounded |
| Distances (pts) | Z-score by level type | Different levels have different typical ranges |
| Volumes/Counts | Log-transform then Z-score | Heavy-tailed distributions |
| Velocities | Z-score | Varying market activity levels |
| Binary/Categorical | Leave as-is (0/1) | Discrete |
| Level indices (0-9) | Divide by 9 → [0, 1] | Bounded ordinal |

### 2.3 Z-Score Computation

Compute normalization statistics **per level type** from the training set:

```python
for level_type in ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW']:
    subset = train_episodes[train_episodes.level_type == level_type]
    for feature in continuous_features:
        mean = subset[feature].mean()
        std = subset[feature].std()
        norm_params[level_type][feature] = (mean, std)
```

At inference, normalize using the params for the relevant level type.

### 2.4 Log Transform for Volume Features

```python
volume_features = [
    'cumul_signed_trade_vol', 'cumul_trade_vol', 
    'setup_total_trade_vol', 'trade_vol_sum',
    'depth_bid10_qty_eob', 'depth_ask10_qty_eob',
    ...
]

for f in volume_features:
    df[f'{f}_log'] = np.sign(df[f]) * np.log1p(np.abs(df[f]))
```

### 2.5 Handling NaN and Inf

| Condition | Handling |
|-----------|----------|
| NaN from insufficient lookback | Fill with 0 (neutral) |
| NaN from no wall found | Fill with max_distance (10 pts) for distance, -1 for level_idx |
| Inf from division | Clamp to ±10 std |
| All-zero denominator | Set ratio to 0 |

---

## Section 3 — Vector Assembly

### 3.1 Extraction Point

The setup vector is extracted at the **trigger bar** — the first bar where price enters the ±1pt band of the level.

```python
trigger_bar = episode[episode['is_trigger_bar'] == True].iloc[0]
setup_vector = extract_setup_vector(trigger_bar, episode_lookback)
```

### 3.2 Assembly Function

```python
def extract_setup_vector(trigger_bar: pd.Series, lookback_bars: pd.DataFrame) -> np.ndarray:
    """
    Extract fixed-length setup vector from episode data.
    
    Args:
        trigger_bar: Single row (Series) at trigger time with all Stage 3 features
        lookback_bars: DataFrame of bars preceding trigger (for profile computation)
    
    Returns:
        np.ndarray of shape (256,)
    """
    
    components = []
    
    # 1. Trigger-bar snapshot features (39 dims)
    snapshot_features = [
        'approach_dist_to_level_pts_eob',
        'approach_side_of_level_eob',
        # ... (full list from Section 1.1)
    ]
    components.append(trigger_bar[snapshot_features].values)
    
    # 2. Derivative features at trigger (40 dims)
    deriv_features = [
        'deriv_dist_d1_w3',
        'deriv_dist_d1_w12',
        # ... (full list from Section 1.2)
    ]
    components.append(trigger_bar[deriv_features].values)
    
    # 3. Lookback profile features (60 dims)
    profile = compute_lookback_profile(lookback_bars)
    components.append(profile)
    
    # 4. Recent momentum features (12 dims)
    recent = compute_recent_momentum(lookback_bars.tail(12))
    components.append(recent)
    
    # Concatenate
    raw_vector = np.concatenate(components)  # 151 dims
    
    # Normalize
    normalized = normalize_vector(raw_vector, level_type=trigger_bar['level_type'])
    
    # Pad or project to target dimension
    final_vector = project_to_dimension(normalized, target_dim=256)
    
    return final_vector
```

### 3.3 Dimension Projection

To reach exactly 256 dimensions from 151:

**Option A: Zero-padding** (simple, preserves interpretability)
```python
def project_to_dimension(vec, target_dim=256):
    if len(vec) < target_dim:
        return np.pad(vec, (0, target_dim - len(vec)), mode='constant')
    return vec[:target_dim]
```

**Option B: Learned projection** (better similarity, less interpretable)
```python
# Train a projection matrix on historical data
projection_matrix = train_projection(train_vectors, target_dim=256)

def project_to_dimension(vec, target_dim=256):
    return projection_matrix @ vec
```

**Option C: PCA** (captures variance, moderate interpretability)
```python
# Fit PCA on training data
pca = PCA(n_components=256).fit(train_vectors)

def project_to_dimension(vec, target_dim=256):
    return pca.transform(vec.reshape(1, -1))[0]
```

**Recommendation:** Start with Option A (padding) for initial deployment, transition to Option C (PCA) after gathering sufficient episodes.

---

## Section 4 — FAISS Index Configuration

### 4.1 Index Type Selection

| Index Type | Use Case | Trade-off |
|------------|----------|-----------|
| `IndexFlatL2` | < 100K vectors | Exact, slow |
| `IndexIVFFlat` | 100K - 10M vectors | Approximate, fast |
| `IndexIVFPQ` | > 10M vectors | Compressed, fastest |
| `IndexHNSW` | Any size, high recall needed | Graph-based, high memory |

**Recommendation for initial deployment:**
- Expected volume: ~50 episodes/day × 4 levels × 252 trading days = ~50K episodes/year
- Use `IndexIVFFlat` with `nlist=100` for first year
- Transition to `IndexIVFPQ` after 2+ years of data

### 4.2 Index Construction

```python
import faiss

def build_index(vectors: np.ndarray, index_type: str = 'ivf_flat') -> faiss.Index:
    """
    Build FAISS index from setup vectors.
    
    Args:
        vectors: np.ndarray of shape (n_episodes, 256)
        index_type: 'flat', 'ivf_flat', or 'ivf_pq'
    
    Returns:
        Trained FAISS index
    """
    d = vectors.shape[1]  # 256
    n = vectors.shape[0]
    
    if index_type == 'flat':
        index = faiss.IndexFlatL2(d)
        
    elif index_type == 'ivf_flat':
        nlist = min(100, n // 10)  # ~10 vectors per cell minimum
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(vectors)
        
    elif index_type == 'ivf_pq':
        nlist = min(1000, n // 100)
        m = 32  # Number of subquantizers (d must be divisible by m)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        index.train(vectors)
    
    index.add(vectors)
    return index
```

### 4.3 Index Persistence

```python
# Save
faiss.write_index(index, 'setup_vectors.index')

# Load
index = faiss.read_index('setup_vectors.index')
```

### 4.4 Separate Indices by Level Type (Optional)

For faster, more relevant retrieval, maintain separate indices:

```
indices/
  pm_high_setups.index
  pm_low_setups.index
  or_high_setups.index
  or_low_setups.index
```

Query the appropriate index based on which level is being tested.

---

## Section 5 — Metadata Store

### 5.1 Purpose

FAISS returns vector indices, not episode data. The metadata store maps indices to episode information.

### 5.2 Schema

```sql
CREATE TABLE setup_metadata (
    vector_id       INTEGER PRIMARY KEY,  -- Position in FAISS index
    episode_id      TEXT NOT NULL,        -- Unique episode identifier
    date            DATE NOT NULL,
    symbol          TEXT NOT NULL,
    level_type      TEXT NOT NULL,        -- PM_HIGH, PM_LOW, OR_HIGH, OR_LOW
    level_price     REAL NOT NULL,
    trigger_bar_ts  INTEGER NOT NULL,     -- Nanosecond timestamp
    approach_direction INTEGER NOT NULL,  -- +1 or -1
    outcome         TEXT NOT NULL,        -- STRONG_BREAK, WEAK_BREAK, CHOP, etc.
    outcome_score   REAL NOT NULL,        -- Continuous score
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Optional: store key setup characteristics for filtering without loading full episode
    velocity_at_trigger     REAL,
    obi0_at_trigger         REAL,
    wall_imbal_at_trigger   REAL
);

CREATE INDEX idx_level_type ON setup_metadata(level_type);
CREATE INDEX idx_outcome ON setup_metadata(outcome);
CREATE INDEX idx_date ON setup_metadata(date);
```

### 5.3 Implementation Options

| Option | Pros | Cons |
|--------|------|------|
| SQLite | Simple, embedded | Single-writer |
| PostgreSQL | Concurrent, scalable | External dependency |
| Parquet + pandas | File-based, fast reads | Append-heavy workloads |

**Recommendation:** SQLite for initial deployment, PostgreSQL for production.

---

## Section 6 — Query Interface

### 6.1 Core Query Function

```python
def find_similar_setups(
    query_vector: np.ndarray,
    level_type: str,
    k: int = 20,
    filters: dict = None
) -> List[SimilarSetup]:
    """
    Find k most similar historical setups.
    
    Args:
        query_vector: Setup vector from live episode (256 dims)
        level_type: Which level is being tested
        k: Number of similar setups to return
        filters: Optional filters (e.g., date_range, min_outcome_confidence)
    
    Returns:
        List of SimilarSetup objects with metadata and distances
    """
    
    # Select appropriate index
    index = indices[level_type]
    
    # Search (returns distances and indices)
    distances, indices = index.search(query_vector.reshape(1, -1), k * 2)  # Over-fetch for filtering
    
    # Fetch metadata
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        metadata = get_metadata(idx)
        
        # Apply filters
        if filters:
            if filters.get('min_date') and metadata.date < filters['min_date']:
                continue
            if filters.get('exclude_chop') and metadata.outcome == 'CHOP':
                continue
        
        results.append(SimilarSetup(
            episode_id=metadata.episode_id,
            distance=dist,
            similarity=1 / (1 + dist),  # Convert distance to similarity
            outcome=metadata.outcome,
            outcome_score=metadata.outcome_score,
            date=metadata.date,
            level_price=metadata.level_price,
            approach_direction=metadata.approach_direction
        ))
        
        if len(results) >= k:
            break
    
    return results
```

### 6.2 Outcome Distribution Computation

```python
def compute_outcome_distribution(similar_setups: List[SimilarSetup]) -> OutcomeDistribution:
    """
    Compute outcome distribution from similar setups.
    
    Returns probability distribution over outcomes, weighted by similarity.
    """
    
    outcomes = ['STRONG_BREAK', 'WEAK_BREAK', 'CHOP', 'WEAK_BOUNCE', 'STRONG_BOUNCE']
    
    # Unweighted counts
    counts = Counter(s.outcome for s in similar_setups)
    total = len(similar_setups)
    
    unweighted = {o: counts.get(o, 0) / total for o in outcomes}
    
    # Similarity-weighted
    weighted_counts = defaultdict(float)
    total_weight = 0
    for s in similar_setups:
        weighted_counts[s.outcome] += s.similarity
        total_weight += s.similarity
    
    weighted = {o: weighted_counts.get(o, 0) / total_weight for o in outcomes}
    
    # Expected outcome score
    expected_score = sum(s.outcome_score * s.similarity for s in similar_setups) / total_weight
    
    return OutcomeDistribution(
        unweighted=unweighted,
        weighted=weighted,
        expected_score=expected_score,
        n_samples=total,
        avg_similarity=total_weight / total
    )
```

### 6.3 Response Format

```python
@dataclass
class RetrievalResponse:
    query_level_type: str
    query_level_price: float
    query_approach_direction: int
    
    similar_setups: List[SimilarSetup]
    outcome_distribution: OutcomeDistribution
    
    # Summary statistics
    n_retrieved: int
    avg_distance: float
    avg_similarity: float
    
    # Confidence indicators
    retrieval_confidence: float  # Based on how close matches are
    outcome_confidence: float    # Based on outcome consistency
    
    def to_dict(self) -> dict:
        return {
            'level': f"{self.query_level_type} @ {self.query_level_price}",
            'approach': 'from_below' if self.query_approach_direction == 1 else 'from_above',
            'n_similar': self.n_retrieved,
            'outcome_probs': self.outcome_distribution.weighted,
            'expected_move': f"{self.outcome_distribution.expected_score:+.2f} pts",
            'confidence': f"{self.retrieval_confidence:.0%}",
            'top_matches': [
                {
                    'date': s.date,
                    'outcome': s.outcome,
                    'similarity': f"{s.similarity:.0%}"
                }
                for s in self.similar_setups[:5]
            ]
        }
```

---

## Section 7 — Live Query Pipeline

### 7.1 Real-Time Setup Detection

```python
class LiveSetupDetector:
    """Monitors incoming bars and detects setups approaching levels."""
    
    def __init__(self, levels: Dict[str, float], lookback_buffer_size: int = 200):
        self.levels = levels  # {'PM_HIGH': 6800.0, 'PM_LOW': 6750.0, ...}
        self.buffer = deque(maxlen=lookback_buffer_size)
        self.last_query_time = {}  # Cooldown tracking per level
        
    def process_bar(self, bar: pd.Series) -> Optional[LiveSetup]:
        """
        Process incoming bar, return LiveSetup if trigger detected.
        """
        self.buffer.append(bar)
        
        if len(self.buffer) < 180:  # Need full lookback
            return None
        
        microprice = compute_microprice(bar)
        
        for level_type, level_price in self.levels.items():
            # Check cooldown
            if self._in_cooldown(level_type):
                continue
            
            # Check trigger condition
            dist = abs(microprice - level_price)
            if dist <= 1.0:  # Within band
                # Compute setup vector
                lookback = pd.DataFrame(list(self.buffer))
                trigger_bar = self._add_stage3_features(bar, lookback)
                vector = extract_setup_vector(trigger_bar, lookback)
                
                # Mark cooldown
                self.last_query_time[level_type] = bar['bar_ts']
                
                return LiveSetup(
                    level_type=level_type,
                    level_price=level_price,
                    trigger_bar=trigger_bar,
                    setup_vector=vector,
                    timestamp=bar['bar_ts']
                )
        
        return None
```

### 7.2 Full Query Flow

```
[Live Bar Stream]
       ↓
[LiveSetupDetector] → Detects trigger
       ↓
[Stage 3 Feature Computation] → Adds derivatives, cumulatives
       ↓
[Vector Extraction] → 256-dim vector
       ↓
[Normalization] → Using stored params
       ↓
[FAISS Query] → Top-k similar
       ↓
[Metadata Fetch] → Get outcomes
       ↓
[Outcome Distribution] → Probability breakdown
       ↓
[Response Formatting] → User-facing output
```

---

## Section 8 — Index Maintenance

### 8.1 Daily Update Process

```python
def daily_index_update(new_episodes: List[Episode]):
    """
    Add new episodes to index after market close.
    """
    
    for episode in new_episodes:
        # Extract setup vector
        trigger_bar = episode.get_trigger_bar()
        lookback = episode.get_lookback_bars()
        vector = extract_setup_vector(trigger_bar, lookback)
        
        # Add to appropriate index
        level_type = episode.level_type
        vector_id = indices[level_type].ntotal  # Next available ID
        indices[level_type].add(vector.reshape(1, -1))
        
        # Add metadata
        add_metadata(
            vector_id=vector_id,
            episode_id=episode.episode_id,
            level_type=level_type,
            outcome=episode.outcome,
            outcome_score=episode.outcome_score,
            ...
        )
    
    # Persist updated indices
    for level_type in indices:
        faiss.write_index(indices[level_type], f'{level_type}_setups.index')
```

### 8.2 Retraining Schedule

| Trigger | Action |
|---------|--------|
| Weekly | Recompute normalization params |
| Monthly | Retrain IVF centroids (if using IVF) |
| Quarterly | Full PCA refit (if using PCA projection) |
| Data quality issue | Full rebuild |

### 8.3 Version Control

```
indices/
  v1/
    pm_high_setups.index
    norm_params.json
    pca_model.pkl
  v2/
    pm_high_setups.index
    norm_params.json
    pca_model.pkl
  current -> v2  (symlink)
```

---

## Section 9 — Evaluation Metrics

### 9.1 Retrieval Quality

| Metric | Definition | Target |
|--------|------------|--------|
| Recall@k | % of true similar setups in top-k | > 80% |
| Outcome Accuracy | Does predicted outcome match actual? | > 50% (above random 20%) |
| Calibration | Are probability estimates accurate? | Brier score < 0.2 |

### 9.2 Backtesting Framework

```python
def backtest_retrieval(test_episodes: List[Episode], train_cutoff_date: date):
    """
    Backtest retrieval system using temporal split.
    """
    
    results = []
    
    for episode in test_episodes:
        if episode.date <= train_cutoff_date:
            continue  # Skip training data
        
        # Query using only data available at that time
        similar = find_similar_setups(
            query_vector=episode.setup_vector,
            level_type=episode.level_type,
            k=20,
            filters={'max_date': episode.date - timedelta(days=1)}
        )
        
        # Get predicted distribution
        dist = compute_outcome_distribution(similar)
        predicted_outcome = max(dist.weighted, key=dist.weighted.get)
        
        results.append({
            'episode_id': episode.episode_id,
            'actual_outcome': episode.outcome,
            'predicted_outcome': predicted_outcome,
            'prediction_prob': dist.weighted[predicted_outcome],
            'expected_score': dist.expected_score,
            'actual_score': episode.outcome_score,
            'n_similar': len(similar),
            'avg_similarity': np.mean([s.similarity for s in similar])
        })
    
    return pd.DataFrame(results)
```

---

## Section 10 — Output Artifacts

### 10.1 Per-Episode Output

```
setup_vectors/
  date=2024-01-15/
    vectors.npy              # Shape: (n_episodes, 256)
    episode_ids.json         # Ordered list of episode IDs
```

### 10.2 Index Artifacts

```
indices/
  pm_high_setups.index       # FAISS index file
  pm_low_setups.index
  or_high_setups.index
  or_low_setups.index
```

### 10.3 Normalization Artifacts

```
normalization/
  norm_params.json           # Per-level-type means and stds
  pca_model.pkl              # Optional PCA projection matrix
  feature_order.json         # Ordered list of features in vector
```

### 10.4 Metadata Store

```
metadata/
  setup_metadata.db          # SQLite database
```

---

## Appendix A — Complete Feature List for Setup Vector

```python
SETUP_VECTOR_FEATURES = {
    # Section 1.1: Trigger-bar snapshot (39 features)
    'snapshot': [
        'approach_dist_to_level_pts_eob',
        'approach_side_of_level_eob',
        'approach_alignment_eob',
        'level_polarity',
        'is_standard_approach',
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
        'wall_bid_maxz_eob',
        'wall_ask_maxz_eob',
        'wall_bid_maxz_levelidx_eob',
        'wall_ask_maxz_levelidx_eob',
        'wall_bid_nearest_strong_dist_pts_eob',
        'wall_ask_nearest_strong_dist_pts_eob',
        'wall_bid_nearest_strong_levelidx_eob',
        'wall_ask_nearest_strong_levelidx_eob',
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
    ],
    
    # Section 1.2: Derivatives at trigger (40 features)
    'derivatives': [
        'deriv_dist_d1_w3',
        'deriv_dist_d1_w12',
        'deriv_dist_d1_w36',
        'deriv_dist_d1_w72',
        'deriv_dist_d2_w3',
        'deriv_dist_d2_w12',
        'deriv_dist_d2_w36',
        'deriv_dist_d2_w72',
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
        'deriv_dbid10_d1_w12',
        'deriv_dbid10_d1_w36',
        'deriv_dask10_d1_w12',
        'deriv_dask10_d1_w36',
        'deriv_dbelow01_d1_w12',
        'deriv_dbelow01_d1_w36',
        'deriv_dabove01_d1_w12',
        'deriv_dabove01_d1_w36',
        'deriv_wbidz_d1_w12',
        'deriv_wbidz_d1_w36',
        'deriv_waskz_d1_w12',
        'deriv_waskz_d1_w36',
        'deriv_wbidz_d2_w12',
        'deriv_wbidz_d2_w36',
        'deriv_waskz_d2_w12',
        'deriv_waskz_d2_w36',
    ],
    
    # Section 1.3: Lookback profile (60 features)
    'profile': [
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
    ],
    
    # Section 1.4: Recent momentum (12 features)
    'recent': [
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
    ],
}

# Total: 39 + 40 + 60 + 12 = 151 raw features
# Padded to 256 for FAISS
```

---

## Appendix B — Normalization Parameters Schema

```json
{
  "version": "1.0",
  "created_at": "2024-01-15T00:00:00Z",
  "n_episodes_train": 10000,
  "level_types": {
    "PM_HIGH": {
      "features": {
        "deriv_dist_d1_w12": {"mean": -0.05, "std": 0.15},
        "setup_total_trade_vol": {"mean": 8.2, "std": 1.5, "log_transform": true},
        ...
      }
    },
    "PM_LOW": {...},
    "OR_HIGH": {...},
    "OR_LOW": {...}
  },
  "global": {
    "features": {
      "state_obi0_eob": {"min": -1, "max": 1, "normalize": false},
      ...
    }
  }
}
```

---

## Appendix C — Example Query Response

```json
{
  "query": {
    "level": "PM_HIGH @ 6800.00",
    "approach": "from_below",
    "trigger_time": "2024-01-15T10:32:15Z"
  },
  "retrieval": {
    "n_similar": 20,
    "avg_similarity": 0.72,
    "confidence": "HIGH"
  },
  "outcome_distribution": {
    "STRONG_BREAK": 0.35,
    "WEAK_BREAK": 0.25,
    "CHOP": 0.20,
    "WEAK_BOUNCE": 0.12,
    "STRONG_BOUNCE": 0.08
  },
  "expected_move": "+1.24 pts",
  "top_matches": [
    {"date": "2024-01-10", "outcome": "STRONG_BREAK", "similarity": "89%"},
    {"date": "2023-12-18", "outcome": "WEAK_BREAK", "similarity": "85%"},
    {"date": "2024-01-08", "outcome": "STRONG_BREAK", "similarity": "82%"},
    {"date": "2023-11-22", "outcome": "CHOP", "similarity": "79%"},
    {"date": "2024-01-03", "outcome": "WEAK_BOUNCE", "similarity": "77%"}
  ],
  "interpretation": "Setup strongly resembles historical breakout patterns. 60% probability of breaking through level within 4-6 minutes."
}
```
