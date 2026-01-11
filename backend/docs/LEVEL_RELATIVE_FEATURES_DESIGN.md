# Level-Relative Features Design Document

## Product Context

Spymaster is a **level-interaction similarity retrieval system** for ES futures trading. The core value proposition:

> When price approaches a key level (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW), retrieve historically similar approach patterns to show empirical outcome distributions (BREAK/REJECT/CHOP).

The key insight: **dealer behavior during the approach** predicts the outcome. We need features that capture how market microstructure evolves as price moves toward a level - not just a snapshot at the moment of touch.

---

## Critical Constraint: Scale Invariance

For similarity retrieval to work across different:
- Price levels (PM_HIGH at 5400 vs OR_LOW at 5380)
- Contracts (ESU5 vs ESZ5 vs ESH6)
- Market regimes (high vs low volatility days)
- Liquidity environments (morning vs afternoon)

**All features must be normalized and level-relative.** Raw quantities like "500 contracts added" are meaningless for comparison. We need ratios like "bid adds dominated ask adds 3:1 in the 0-1 point band."

---

## Current Feature Inventory

### Level-Relative Depth (lvldepth) - 48 features

| Feature Type | Count | Normalized? | Valid for Retrieval? |
|-------------|-------|-------------|---------------------|
| `_qty_eob` / `_qty_twa` | 22 | No - raw quantities | No |
| `_frac_eob` / `_frac_twa` | 20 | Yes - fraction of total | Yes |
| `_cdi_{band}_eob` | 5 | Yes - [-1, 1] imbalance | Yes |
| `_imbal_eob` | 1 | Yes - [-1, 1] imbalance | Yes |

**Usable: 26 features** (frac + cdi + imbal)

### Level-Relative Flow (lvlflow) - 70 features

| Feature Type | Count | Normalized? | Valid for Retrieval? |
|-------------|-------|-------------|---------------------|
| `add_vol_{side}_{band}_sum` | 10 | No - raw volume | No |
| `rem_vol_{side}_{band}_sum` | 10 | No - raw volume | No |
| `net_vol_{side}_{band}_sum` | 10 | No - raw volume | No |
| `cnt_add_{side}_{band}_sum` | 10 | No - raw count | No |
| `cnt_cancel_{side}_{band}_sum` | 10 | No - raw count | No |
| `cnt_modify_{side}_{band}_sum` | 10 | No - raw count | No |
| `net_volnorm_{side}_{band}_sum` | 10 | Yes - normalized by TWA | Yes |

**Usable: 10 features** (only net_volnorm)

### Level-Relative Wall (lvlwall) - 8 features

| Feature Type | Count | Normalized? | Valid for Retrieval? |
|-------------|-------|-------------|---------------------|
| `_maxz_eob` | 2 | Yes - z-score | Yes |
| `_maxz_levelidx_eob` | 2 | Yes - ordinal position | Yes |
| `_nearest_strong_dist_pts_eob` | 2 | Partially - distance in points | Marginal |
| `_nearest_strong_levelidx_eob` | 2 | Yes - ordinal position | Yes |

**Usable: 6-8 features**

### Other Level Features (lvl_, approach_) - 43 features

Need audit - likely mix of normalized and raw.

---

## Current Problem Summary

**Total "level-relative" features: 169**
**Actually usable for retrieval: ~44** (26 + 10 + 8)

The flow features - which capture the most important dealer behavior dynamics - are 86% unusable because they're raw sums.

---

## Required Normalized Flow Features

### 1. Flow Imbalance Per Band (bid vs ask competition)

For each band (p0_1, p1_2, p2_3, p3_5, p5_10):

```
add_imbal_{band} = (add_vol_bid - add_vol_ask) / (add_vol_bid + add_vol_ask + ε)
cancel_imbal_{band} = (cancel_vol_bid - cancel_vol_ask) / (cancel_vol_bid + cancel_vol_ask + ε)
net_imbal_{band} = (net_vol_bid - net_vol_ask) / (|net_vol_bid| + |net_vol_ask| + ε)
```

**Interpretation:**
- `add_imbal > 0`: Dealers adding more bids than asks at this distance from level
- `cancel_imbal > 0`: Dealers pulling asks faster than bids (bullish)
- `net_imbal > 0`: Net buying pressure at this distance

**Count: 15 features** (3 metrics × 5 bands)

### 2. Count-Based Imbalance (order frequency)

```
cnt_add_imbal_{band} = (cnt_add_bid - cnt_add_ask) / (cnt_add_bid + cnt_add_ask + ε)
cnt_cancel_imbal_{band} = (cnt_cancel_bid - cnt_cancel_ask) / (cnt_cancel_bid + cnt_cancel_ask + ε)
```

**Interpretation:**
- Separates "many small orders" from "few large orders"
- High volume imbalance + low count imbalance = large institutional orders
- Low volume imbalance + high count imbalance = retail/algo activity

**Count: 10 features** (2 metrics × 5 bands)

### 3. Activity Concentration (where is the action?)

```
add_intensity_p0_1 = add_vol_p0_1 / total_add_vol
add_intensity_p0_2 = add_vol_p0_2 / total_add_vol  # cumulative first 2 bands
cancel_intensity_p0_1 = cancel_vol_p0_1 / total_cancel_vol

activity_centroid = Σ(band_midpoint × band_volume) / total_volume
activity_dispersion = std of volume across bands
```

**Interpretation:**
- `intensity_p0_1` high: Activity tightening toward level (breakout setup?)
- `intensity_p0_1` low: Activity dispersed (level may not matter)
- `centroid` decreasing over approach: Flow converging on level

**Count: 8-10 features**

### 4. Above/Below Level Flow Asymmetry

Current code tracks by bid/ask side, but we also need by direction relative to level:

```
add_above_below_imbal = (add_above_level - add_below_level) / (total_add + ε)
cancel_above_below_imbal = (cancel_above - cancel_below) / (total_cancel + ε)
net_above_below_imbal = (net_above - net_below) / (|total_net| + ε)
```

**Interpretation:**
- For PM_HIGH approach from below:
  - `add_above_below_imbal > 0`: Sellers building wall above level
  - `add_above_below_imbal < 0`: Buyers building support below
- Combined with approach direction: predicts break vs reject

**Count: 3-6 features**

### 5. Cross-Band Ratios

```
near_far_add_ratio = add_vol_p0_1 / (add_vol_p5_10 + ε)
near_far_cancel_ratio = cancel_vol_p0_1 / (cancel_vol_p5_10 + ε)
```

**Interpretation:**
- High near/far ratio: Battle concentrated at level (decisive move coming)
- Low near/far ratio: Level not being contested

**Count: 4 features**

### 6. Flow Momentum (rate of change)

These are computed as temporal derivatives on the normalized features above:

```
add_imbal_d1_{band} = d/dt(add_imbal_{band})  # velocity
add_imbal_d2_{band} = d²/dt²(add_imbal_{band})  # acceleration
intensity_d1_p0_1 = d/dt(intensity_p0_1)
```

**Interpretation:**
- `add_imbal_d1 > 0`: Buying pressure increasing
- `add_imbal_d2 > 0`: Buying pressure accelerating (strong conviction)

**Count: Computed at bar5s stage, ~30-40 additional features**

---

## Total New Features

| Category | New Features |
|----------|-------------|
| Flow Imbalance | 15 |
| Count Imbalance | 10 |
| Activity Concentration | 10 |
| Above/Below Asymmetry | 6 |
| Cross-Band Ratios | 4 |
| **Subtotal (base)** | **45** |
| Temporal Derivatives (d1, d2) | ~90 |
| **Total** | **~135** |

Combined with existing usable features (~44), this gives **~180 normalized, level-relative features** for similarity retrieval.

---

## Implementation Location

### `level_relative.py` Changes

1. **Modify `extract_level_relative_flow_features()`** to compute imbalances alongside raw sums
2. **Add `compute_flow_concentration_features()`** for intensity metrics
3. **Track flow by above/below direction** in `process_lvl_flow_ticks()` using existing `compute_lvl_flow_direction()`

### `compute_bar5s_features.py` Changes

1. Compute d1/d2 derivatives on the new normalized flow features
2. Ensure derivatives are computed on normalized features, not raw

### Contract Updates

1. Add new features to `market_by_price_10_level_episodes.avsc`
2. Propagate to `market_by_price_10_level_approach.avsc`

### `episode_embeddings.py` Changes

1. Update `TRUE_LEVEL_RELATIVE_PREFIXES` to include new feature patterns
2. Verify all raw `_sum` features are excluded

---

## Validation Criteria

After implementation, verify:

1. **Scale invariance**: Same approach pattern on different days produces similar feature vectors
2. **Discriminative power**: BREAK vs REJECT episodes have statistically different feature distributions
3. **Temporal signal**: Features show meaningful evolution during approach (not flat)
4. **No data leakage**: Features only use information available at each bar timestamp

---

## Pipeline Reprocessing Required

After feature changes:

```bash
# Clear silver episode/approach data
rm -rf lake/silver/product_type=future/symbol=*/table=market_by_price_10_*_episodes/
rm -rf lake/silver/product_type=future/symbol=*/table=market_by_price_10_*_approach/

# Reprocess silver
uv run python -m src.data_eng.runner --product-type future --layer silver --symbol ESH6 --dates 2025-12-01:2025-12-31 --workers 8
```

---

## Open Questions

1. **Band granularity**: Current bands (0-1, 1-2, 2-3, 3-5, 5-10 pts) - is this optimal for ES tick size?
2. **TWA vs EOB**: Should imbalances be TWA (time-weighted) or EOB (end-of-bar)?
3. **Aggregation window**: 5-second bars may be too granular - consider 15s or 30s for flow?
4. **Normalization method**: Simple ratio vs z-score vs percentile rank?
