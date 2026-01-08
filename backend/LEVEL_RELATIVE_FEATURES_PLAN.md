## Level-Relative Feature Conversion (112 Market-Wide → Level-Relative)

### Goal
Convert 112 market-wide features (currently computed relative to MID) to level-relative variants (computed relative to LEVEL price: PM_HIGH, PM_LOW, OR_HIGH, OR_LOW).

### Feature Breakdown

| Category | Count | Current Reference | New Reference |
|----------|-------|-------------------|---------------|
| DEPTH | 40 | MID price | LEVEL price |
| FLOW | 70 | MID price | LEVEL price |
| WALL | 2 | MID price | LEVEL price |
| **TOTAL** | **112** | | |

### DEPTH Features (40)
Banded depth quantities and fractions above/below reference price.

**Current**: `bar5s_depth_{above,below}_{band}_qty_{twa,eob}` and `bar5s_depth_{above,below}_{band}_frac_{twa,eob}`
**New**: `bar5s_lvldepth_{above,below}_{band}_qty_eob` and `bar5s_lvldepth_{above,below}_{band}_frac_eob`

Bands: p0_1, p1_2, p2_3, p3_5, p5_10
- 5 bands × 2 directions × 2 metrics (qty, frac) × 2 snapshots (twa, eob) = 40 features

### FLOW Features (70)
Order flow volumes and counts in price bands around reference price.

**Current**: `bar5s_flow_{metric}_{side}_{band}_sum`
**New**: `bar5s_lvlflow_{metric}_{side}_{band}_sum`

Metrics: add_vol, rem_vol, net_vol, cnt_add, cnt_cancel, cnt_modify, net_volnorm
Sides: bid, ask
Bands: p0_1, p1_2, p2_3, p3_5, p5_10
- 7 metrics × 2 sides × 5 bands = 70 features

### WALL Features (2)
Distance from reference price to nearest strong wall.

**Current**: `bar5s_wall_{bid,ask}_nearest_strong_dist_pts_eob`
**New**: `bar5s_lvlwall_{bid,ask}_nearest_strong_dist_pts_eob`

### Implementation Approach

1. **Create `level_relative_features.py`** in `src/data_eng/stages/silver/future/mbp10_bar5s/`
   - Reuse existing `compute_banded_quantities`, `compute_wall_features` functions
   - Accept `level_price` parameter instead of computing microprice
   - Return dict with `bar5s_lvl*` prefixed keys

2. **Schema Contract Update**
   - Add 112 new fields to `level_relative_features.avsc`
   - Naming: `bar5s_lvldepth_*`, `bar5s_lvlflow_*`, `bar5s_lvlwall_*`

3. **Integration Point**
   - These features computed AFTER bar5s aggregation
   - Requires joining bar5s data with level prices from `market_by_price_10_with_levels`
   - Level price available per row as `pm_high`, `pm_low`, `or_high`, `or_low`

### Status
1. [x] Create plan document
2. [x] Implement level_relative_features.py
3. [x] Add 112 features to schema (357 total features in schema now)
4. [x] Verify with December 18 data

### Verification Results (Dec 18, 2025 - ESH6)
- PM_HIGH = 6840.75
- Sample bar at level (microprice 6840.772, 0.022 pts from level):
  - MID-relative depth: below=720, above=936
  - LVL-relative depth: below=711, above=945
  - Wall distance: MID=1.98pts, LVL=2.00pts from ask wall

### Files Created/Modified
- `src/data_eng/stages/silver/future/mbp10_bar5s/level_relative.py` (NEW)
- `src/data_eng/contracts/silver/future/level_relative_features.avsc` (UPDATED)
- `src/data_eng/analysis/v2/test_level_relative.py` (NEW - test script)
- `src/data_eng/analysis/v2/test_level_relative_at_touch.py` (NEW - verification)
- `src/data_eng/stages/gold/future/episode_embeddings.py` (NEW - PCA embeddings)

---

## Episode Embedding Pipeline (Gold Layer)

### Architecture
```
Silver Episodes (per bar)     Gold Embeddings (per episode)
─────────────────────────────────────────────────────────────
(N bars × 207 features)  →  Flatten  →  PCA  →  (D,) vector
      ↓
(181, 207) per episode   →  (37,467)  →  PCA  →  (100,) embedding
```

### PCA Variance Analysis (1,335 episodes, Oct-Dec 2025)

| Dimensions | Variance Explained |
|------------|-------------------|
| 10 | 85.2% |
| 25 | 89.2% |
| 50 | 92.3% |
| 90 | 95.0% |
| 100 | 95.5% |

**Recommendation**: Use 100-200 dimensions for vector search (95%+ variance)

### Outcome Distribution
- CHOP: 40.4%
- STRONG_BOUNCE: 25.2%
- STRONG_BREAK: 13.3%
- WEAK_BOUNCE: 13.2%
- WEAK_BREAK: 7.9%

### Next Steps for PatchTST
1. Use (181, 207) tensors as input
2. Patch size: 10-20 bars
3. Train with contrastive loss on outcome labels
4. Encoder output → fixed embedding for retrieval
