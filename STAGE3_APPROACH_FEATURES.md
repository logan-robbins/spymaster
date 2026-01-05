# Stage 3 — Approach & Setup Feature Engineering

## Context

**Input:** Stage 2 output — datasets containing 15-minute lookback + 8-minute forward windows around level touches.

**Each dataset contains:**
- Continuous 5-second bars (233 base features per bar)
- A `level_type` column ∈ {PM_HIGH, PM_LOW, OR_HIGH, OR_LOW}
- A `level_price` column (the actual price of the level being tested)
- A `touch_id` to group bars belonging to the same touch event

**Goal:** Compute features that characterize:
1. **Where** — Position relative to the level (signed distance)
2. **How** — Direction and speed of approach (derivatives)
3. **What's building** — Evolving book physics during approach (cumulative flows, depth shifts)
4. **The setup signature** — Compressed representation of the approach pattern

---

## Section 0 — Naming Convention

Stage 3 features extend the base naming:

```
bar5s_<family>_<detail>_<agg>_<derivative>
```

New families:
- `approach` — Price trajectory relative to level
- `deriv` — Temporal derivatives of base features
- `cumul` — Cumulative sums over lookback
- `setup` — Aggregated setup characterization

Derivative suffixes:
- `d1_<window>` — First derivative over N bars
- `d2_<window>` — Second derivative over N bars
- `cuml` — Cumulative from start of lookback

Window notation:
- `w3` = 3 bars = 15 seconds
- `w12` = 12 bars = 1 minute
- `w36` = 36 bars = 3 minutes
- `w180` = 180 bars = 15 minutes (full lookback)

---

## Section 1 — Level-Relative Position Features

These define WHERE price is relative to the level being tested.

### 1.1 Signed Distance to Level

```
dist_to_level(t) = (microprice(t) - level_price) / POINT
```

- **Positive** = price is ABOVE the level
- **Negative** = price is BELOW the level
- **Zero** = price is AT the level

Microprice is computed same as base features:
```
microprice = (ask_px_00 * bid_sz_00 + bid_px_00 * ask_sz_00) / (bid_sz_00 + ask_sz_00)
```

| Feature | Description |
|---------|-------------|
| `bar5s_approach_dist_to_level_pts_eob` | Signed distance at bar end (points) |
| `bar5s_approach_dist_to_level_pts_twa` | TWA signed distance over bar |
| `bar5s_approach_abs_dist_to_level_pts_eob` | Absolute distance (for "closeness" regardless of side) |

### 1.2 Side of Level (Binary)

```
side_of_level = 1 if microprice > level_price else -1
```

| Feature | Description |
|---------|-------------|
| `bar5s_approach_side_of_level_eob` | +1 = above level, -1 = below level |

### 1.3 Level Type Encoding

One-hot encode the level type for models that need it:

| Feature | Description |
|---------|-------------|
| `bar5s_approach_is_pm_high` | 1 if testing PM_HIGH, else 0 |
| `bar5s_approach_is_pm_low` | 1 if testing PM_LOW, else 0 |
| `bar5s_approach_is_or_high` | 1 if testing OR_HIGH, else 0 |
| `bar5s_approach_is_or_low` | 1 if testing OR_LOW, else 0 |

### 1.4 Level Polarity

For directional interpretation — is this a "ceiling" or "floor" test?

```
level_polarity = +1 if level_type in {PM_HIGH, OR_HIGH} else -1
```

| Feature | Description |
|---------|-------------|
| `bar5s_approach_level_polarity` | +1 = high (resistance), -1 = low (support) |

### 1.5 Approach Direction Alignment

Is price approaching FROM the expected direction?
- Testing a HIGH from below = "normal" approach (+1)
- Testing a HIGH from above = "retest" approach (-1)
- Testing a LOW from above = "normal" approach (+1)
- Testing a LOW from below = "retest" approach (-1)

```
approach_alignment = -1 * side_of_level * level_polarity
```

| Feature | Description |
|---------|-------------|
| `bar5s_approach_alignment_eob` | +1 = normal approach, -1 = retest |

---

## Section 2 — Temporal Derivatives

Compute derivatives of key base features to capture momentum and acceleration.

### 2.1 Derivative Windows

| Window | Bars | Duration | Purpose |
|--------|------|----------|---------|
| `w3` | 3 | 15 sec | Micro momentum |
| `w12` | 12 | 1 min | Short-term trend |
| `w36` | 36 | 3 min | Medium-term trend |
| `w72` | 72 | 6 min | Longer setup context |

### 2.2 First Derivative (Velocity)

```
d1_<feature>_<window> = (feature[t] - feature[t - window]) / window
```

This is the average rate of change per bar over the window.

### 2.3 Second Derivative (Acceleration)

```
d2_<feature>_<window> = (d1[t] - d1[t - window]) / window
```

Or equivalently, the change in velocity.

### 2.4 Features to Differentiate

**Priority 1 — Approach trajectory:**
| Base Feature | Rationale |
|--------------|-----------|
| `approach_dist_to_level_pts_eob` | Velocity toward level, acceleration toward level |

**Priority 2 — Book imbalance dynamics:**
| Base Feature | Rationale |
|--------------|-----------|
| `state_obi0_eob` | Is top-of-book pressure shifting? |
| `state_obi10_eob` | Is full-book pressure shifting? |
| `state_cdi_p0_1_eob` | Is near-level imbalance shifting? |
| `state_cdi_p1_2_eob` | Is 1-2 point band imbalance shifting? |

**Priority 3 — Depth buildup:**
| Base Feature | Rationale |
|--------------|-----------|
| `depth_bid10_qty_eob` | Is bid depth growing or shrinking? |
| `depth_ask10_qty_eob` | Is ask depth growing or shrinking? |
| `depth_below_p0_1_qty_eob` | Is near-bid depth building? |
| `depth_above_p0_1_qty_eob` | Is near-ask depth building? |

**Priority 4 — Wall dynamics:**
| Base Feature | Rationale |
|--------------|-----------|
| `wall_bid_maxz_eob` | Are bid walls intensifying? |
| `wall_ask_maxz_eob` | Are ask walls intensifying? |

### 2.5 Derivative Feature Names

For each (base_feature, window) combination:

```
bar5s_deriv_<base_feature_short>_d1_<window>
bar5s_deriv_<base_feature_short>_d2_<window>
```

**Shortened base feature names for derivatives:**

| Full Name | Short Name |
|-----------|------------|
| `approach_dist_to_level_pts_eob` | `dist` |
| `state_obi0_eob` | `obi0` |
| `state_obi10_eob` | `obi10` |
| `state_cdi_p0_1_eob` | `cdi01` |
| `state_cdi_p1_2_eob` | `cdi12` |
| `depth_bid10_qty_eob` | `dbid10` |
| `depth_ask10_qty_eob` | `dask10` |
| `depth_below_p0_1_qty_eob` | `dbelow01` |
| `depth_above_p0_1_qty_eob` | `dabove01` |
| `wall_bid_maxz_eob` | `wbidz` |
| `wall_ask_maxz_eob` | `waskz` |

**Example feature names:**
```
bar5s_deriv_dist_d1_w3      # Velocity toward level (15s window)
bar5s_deriv_dist_d1_w12     # Velocity toward level (1min window)
bar5s_deriv_dist_d2_w12     # Acceleration toward level (1min window)
bar5s_deriv_obi0_d1_w12     # Rate of OBI0 change (1min)
bar5s_deriv_cdi01_d1_w36    # Rate of near-level imbalance change (3min)
```

### 2.6 Derivative Feature Count

- 11 base features
- 4 windows for d1
- 4 windows for d2
- Total: 11 × (4 + 4) = **88 derivative features**

---

## Section 3 — Cumulative Features

Track what has accumulated during the approach.

### 3.1 Cumulative Trade Imbalance

```
cumul_signed_trade_vol(t) = sum(trade_signed_vol_sum[0:t])
```

From the start of the lookback window to current bar.

| Feature | Description |
|---------|-------------|
| `bar5s_cumul_signed_trade_vol` | Net aggressive buying minus selling |
| `bar5s_cumul_trade_vol` | Total trade volume |
| `bar5s_cumul_aggbuy_vol` | Cumulative aggressive buy volume |
| `bar5s_cumul_aggsell_vol` | Cumulative aggressive sell volume |

### 3.2 Cumulative Flow Imbalance

Net additions minus removals, cumulated over lookback.

**By side:**
```
cumul_flow_net_bid(t) = sum over bands: sum(flow_net_vol_bid_<band>_sum[0:t])
cumul_flow_net_ask(t) = sum over bands: sum(flow_net_vol_ask_<band>_sum[0:t])
```

| Feature | Description |
|---------|-------------|
| `bar5s_cumul_flow_net_bid` | Cumulative net bid-side flow |
| `bar5s_cumul_flow_net_ask` | Cumulative net ask-side flow |
| `bar5s_cumul_flow_imbal` | cumul_flow_net_bid - cumul_flow_net_ask |

**By band (for near vs far dynamics):**

| Feature | Description |
|---------|-------------|
| `bar5s_cumul_flow_net_bid_p0_1` | Cumulative net flow, bid side, 0-1 pt band |
| `bar5s_cumul_flow_net_ask_p0_1` | Cumulative net flow, ask side, 0-1 pt band |
| `bar5s_cumul_flow_net_bid_p1_2` | ... |
| `bar5s_cumul_flow_net_ask_p1_2` | ... |

(Continue for all 5 bands × 2 sides = 10 features)

### 3.3 Cumulative Event Counts

| Feature | Description |
|---------|-------------|
| `bar5s_cumul_msg_cnt` | Total messages processed |
| `bar5s_cumul_trade_cnt` | Total trades |
| `bar5s_cumul_add_cnt` | Total adds |
| `bar5s_cumul_cancel_cnt` | Total cancels |

### 3.4 Normalized Cumulative Features

Normalize cumulatives by time elapsed (bars since lookback start):

```
bars_elapsed = current_bar_index - lookback_start_index + 1
cumul_signed_trade_vol_per_bar = cumul_signed_trade_vol / bars_elapsed
```

| Feature | Description |
|---------|-------------|
| `bar5s_cumul_signed_trade_vol_rate` | Avg signed trade vol per bar |
| `bar5s_cumul_flow_imbal_rate` | Avg flow imbalance per bar |

---

## Section 4 — Level-Relative Book Reframing

The base features compute depth "above/below microprice." For level tests, we also need "above/below THE LEVEL."

### 4.1 Concept

Partition the book relative to `level_price` instead of `microprice`:

```
depth_above_level = sum of ask sizes where ask_px > level_price
depth_below_level = sum of bid sizes where bid_px < level_price
depth_at_level = sum of sizes where |px - level_price| <= 0.5 * POINT
```

### 4.2 Level-Relative Depth Features

| Feature | Description |
|---------|-------------|
| `bar5s_lvl_depth_above_qty_eob` | Total depth above the level |
| `bar5s_lvl_depth_below_qty_eob` | Total depth below the level |
| `bar5s_lvl_depth_at_qty_eob` | Depth right at the level (±0.5 pt) |
| `bar5s_lvl_depth_imbal_eob` | (below - above) / (below + above + ε) |

### 4.3 Level-Relative Banded Depth

Distance bands relative to level_price, not microprice:

| Band | Description |
|------|-------------|
| `lvl_p0_1` | Within 1 point of level |
| `lvl_p1_2` | 1-2 points from level |
| `lvl_p2_5` | 2-5 points from level |

For each band, compute:
- `bar5s_lvl_depth_above_<band>_qty_eob` — Ask depth in this band above level
- `bar5s_lvl_depth_below_<band>_qty_eob` — Bid depth in this band below level
- `bar5s_lvl_cdi_<band>_eob` — Cross-depth imbalance for this band

### 4.4 Level-Relative Flow

Same reframing for flows — where is liquidity being added/removed relative to the level?

```
flow toward level = adds on the side between price and level
flow away from level = adds on the side away from level
```

This requires knowing `side_of_level`:

If price is BELOW level (approaching from below):
- "Toward" = ask-side flow (liquidity price will have to buy through)
- "Away" = bid-side flow (support building behind)

If price is ABOVE level (approaching from above):
- "Toward" = bid-side flow (liquidity price will have to sell through)
- "Away" = ask-side flow (resistance building behind)

| Feature | Description |
|---------|-------------|
| `bar5s_lvl_flow_toward_net_sum` | Net flow on the side between price and level |
| `bar5s_lvl_flow_away_net_sum` | Net flow on the side away from level |
| `bar5s_lvl_flow_toward_away_imbal_sum` | toward - away |

---

## Section 5 — Setup Signature Features

Compressed characterization of the full approach pattern.

### 5.1 Approach Trajectory Summary

Over the full lookback window:

| Feature | Description |
|---------|-------------|
| `bar5s_setup_start_dist_pts` | Distance to level at lookback start |
| `bar5s_setup_min_dist_pts` | Closest approach during lookback |
| `bar5s_setup_max_dist_pts` | Furthest distance during lookback |
| `bar5s_setup_dist_range_pts` | max - min (total price range traveled) |
| `bar5s_setup_approach_bars` | Bars spent moving toward level |
| `bar5s_setup_retreat_bars` | Bars spent moving away from level |
| `bar5s_setup_approach_ratio` | approach_bars / total_bars |

### 5.2 Approach Velocity Profile

Characterize whether approach is accelerating, decelerating, or steady.

Divide lookback into thirds (early, mid, late):
```
early_velocity = mean(d1_dist) for bars 0 to N/3
mid_velocity = mean(d1_dist) for bars N/3 to 2N/3  
late_velocity = mean(d1_dist) for bars 2N/3 to N
```

| Feature | Description |
|---------|-------------|
| `bar5s_setup_early_velocity` | Approach speed in first third |
| `bar5s_setup_mid_velocity` | Approach speed in middle third |
| `bar5s_setup_late_velocity` | Approach speed in final third |
| `bar5s_setup_velocity_trend` | late - early (+ = accelerating) |

### 5.3 Book Pressure Evolution

How did book imbalance evolve during approach?

| Feature | Description |
|---------|-------------|
| `bar5s_setup_obi0_start` | OBI0 at lookback start |
| `bar5s_setup_obi0_end` | OBI0 at current bar |
| `bar5s_setup_obi0_delta` | end - start |
| `bar5s_setup_obi0_min` | Minimum OBI0 during lookback |
| `bar5s_setup_obi0_max` | Maximum OBI0 during lookback |

Same pattern for:
- `obi10` (full book imbalance)
- `cdi_p0_1` (near-price imbalance)
- `lvl_depth_imbal` (level-relative imbalance)

### 5.4 Flow Accumulation Summary

| Feature | Description |
|---------|-------------|
| `bar5s_setup_total_trade_vol` | Sum of all trade volume in lookback |
| `bar5s_setup_total_signed_vol` | Net signed trade volume |
| `bar5s_setup_trade_imbal_pct` | signed / total (% net buying) |
| `bar5s_setup_flow_imbal_total` | Total flow_net_bid - flow_net_ask |

### 5.5 Wall Presence Summary

| Feature | Description |
|---------|-------------|
| `bar5s_setup_bid_wall_max_z` | Max bid wall z-score seen in lookback |
| `bar5s_setup_ask_wall_max_z` | Max ask wall z-score seen in lookback |
| `bar5s_setup_bid_wall_bars` | Bars where bid wall z > 2 |
| `bar5s_setup_ask_wall_bars` | Bars where ask wall z > 2 |
| `bar5s_setup_wall_imbal` | ask_wall_bars - bid_wall_bars |

---

## Section 6 — Feature Groupings for Retrieval

For similarity search, features should be grouped by what they measure.

### Group A: Position (7 features)
Where is price relative to level?
```
approach_dist_to_level_pts_eob
approach_abs_dist_to_level_pts_eob
approach_side_of_level_eob
approach_alignment_eob
approach_is_pm_high, is_pm_low, is_or_high, is_or_low
approach_level_polarity
```

### Group B: Trajectory (44 features)
How is price moving toward/away from level?
```
deriv_dist_d1_w3, d1_w12, d1_w36, d1_w72
deriv_dist_d2_w3, d2_w12, d2_w36, d2_w72
setup_start_dist, min_dist, max_dist, dist_range
setup_approach_bars, retreat_bars, approach_ratio
setup_early/mid/late_velocity, velocity_trend
```

### Group C: Book Pressure (44 features)
What does liquidity look like around price and level?
```
state_obi0, obi10, cdi_* (from base)
lvl_depth_above/below/at, lvl_depth_imbal
lvl_cdi_* 
deriv_obi0_*, deriv_obi10_*, deriv_cdi01_*
```

### Group D: Flow Dynamics (36 features)
Where is liquidity being added/removed?
```
cumul_flow_net_bid/ask by band
cumul_flow_imbal, flow_imbal_rate
lvl_flow_toward/away, toward_away_imbal
deriv for depth derivatives
```

### Group E: Trade Pressure (12 features)
What are aggressive traders doing?
```
cumul_signed_trade_vol, cumul_trade_vol
cumul_aggbuy/aggsell_vol
cumul_signed_trade_vol_rate
setup_total_trade_vol, signed_vol, trade_imbal_pct
```

### Group F: Walls & Obstacles (16 features)
Where are large resting orders?
```
wall_bid/ask_maxz, levelidx
wall_bid/ask_nearest_strong_dist
deriv_wbidz_*, deriv_waskz_*
setup_bid/ask_wall_max_z, wall_bars, wall_imbal
```

---

## Section 7 — Computation Notes

### 7.1 Lookback Handling

- Features with `_d1_w36` require at least 36 prior bars
- For bars early in the lookback window, either:
  - Output NaN (preferred for ML models that handle it)
  - Use available bars with adjusted denominator

### 7.2 Edge Cases

| Scenario | Handling |
|----------|----------|
| Division by zero in imbalances | Use EPSILON = 1e-9 in denominator |
| No bars in a lookback third | Set velocity = 0 for that segment |
| No wall found (z < threshold) | wall_bars = 0, max_z = actual max |
| Price crosses level during lookback | approach_bars includes bars moving toward regardless of side |

### 7.3 Temporal Alignment

All derivatives and cumulatives are computed WITHIN each touch_id group. Do not compute derivatives across different touch events.

```python
for touch_id in dataset.touch_id.unique():
    touch_data = dataset[dataset.touch_id == touch_id]
    # compute derivatives/cumulatives only within this group
```

### 7.4 Output Schema

Each row remains a 5-second bar. New columns are added:

```
bar_ts                          # Timestamp (from Stage 1)
symbol                          # Instrument
touch_id                        # Groups bars in same touch event
level_type                      # PM_HIGH, PM_LOW, OR_HIGH, OR_LOW
level_price                     # Actual price of the level
bar_index_in_touch              # 0-indexed position within touch
is_pre_touch                    # True if in 15-min lookback
is_post_touch                   # True if in 8-min forward

# ... 233 base features ...
# ... ~100 approach/derivative features ...
# ... ~30 cumulative features ...
# ... ~20 level-relative features ...
# ... ~30 setup summary features ...
```

---

## Section 8 — Feature Count Summary

| Category | Count |
|----------|-------|
| Base features (Stage 1) | 233 |
| Position features | 9 |
| Derivative features | 88 |
| Cumulative features | 24 |
| Level-relative features | 18 |
| Setup summary features | 32 |
| **Total** | **~404** |

---

## Section 9 — Validation Checks

### 9.1 Position Features
- `dist_to_level_pts` should be in range [-1, +1] by construction (±1 pt filter)
- `side_of_level` ∈ {-1, +1}
- One-hot level features sum to 1

### 9.2 Derivatives
- `d1_dist` with negative value = approaching level
- `d2_dist` with same sign as d1 = accelerating
- Derivatives should be continuous (no jumps across touch boundaries)

### 9.3 Cumulatives
- Monotonically increasing for volume/count features
- `cumul[t] = cumul[t-1] + bar_value[t]`

### 9.4 Setup Features
- Only valid for bars where sufficient lookback exists
- `approach_bars + retreat_bars ≤ total_bars`
- `early + mid + late velocity` segments should cover full lookback

---

## Appendix A — Full Derivative Feature List

```
# Distance derivatives (8 features)
bar5s_deriv_dist_d1_w3
bar5s_deriv_dist_d1_w12
bar5s_deriv_dist_d1_w36
bar5s_deriv_dist_d1_w72
bar5s_deriv_dist_d2_w3
bar5s_deriv_dist_d2_w12
bar5s_deriv_dist_d2_w36
bar5s_deriv_dist_d2_w72

# OBI0 derivatives (8 features)
bar5s_deriv_obi0_d1_w3
bar5s_deriv_obi0_d1_w12
bar5s_deriv_obi0_d1_w36
bar5s_deriv_obi0_d1_w72
bar5s_deriv_obi0_d2_w3
bar5s_deriv_obi0_d2_w12
bar5s_deriv_obi0_d2_w36
bar5s_deriv_obi0_d2_w72

# OBI10 derivatives (8 features)
bar5s_deriv_obi10_d1_w3
bar5s_deriv_obi10_d1_w12
bar5s_deriv_obi10_d1_w36
bar5s_deriv_obi10_d1_w72
bar5s_deriv_obi10_d2_w3
bar5s_deriv_obi10_d2_w12
bar5s_deriv_obi10_d2_w36
bar5s_deriv_obi10_d2_w72

# CDI p0_1 derivatives (8 features)
bar5s_deriv_cdi01_d1_w3
bar5s_deriv_cdi01_d1_w12
bar5s_deriv_cdi01_d1_w36
bar5s_deriv_cdi01_d1_w72
bar5s_deriv_cdi01_d2_w3
bar5s_deriv_cdi01_d2_w12
bar5s_deriv_cdi01_d2_w36
bar5s_deriv_cdi01_d2_w72

# CDI p1_2 derivatives (8 features)
bar5s_deriv_cdi12_d1_w3
bar5s_deriv_cdi12_d1_w12
bar5s_deriv_cdi12_d1_w36
bar5s_deriv_cdi12_d1_w72
bar5s_deriv_cdi12_d2_w3
bar5s_deriv_cdi12_d2_w12
bar5s_deriv_cdi12_d2_w36
bar5s_deriv_cdi12_d2_w72

# Bid depth derivatives (8 features)
bar5s_deriv_dbid10_d1_w3
bar5s_deriv_dbid10_d1_w12
bar5s_deriv_dbid10_d1_w36
bar5s_deriv_dbid10_d1_w72
bar5s_deriv_dbid10_d2_w3
bar5s_deriv_dbid10_d2_w12
bar5s_deriv_dbid10_d2_w36
bar5s_deriv_dbid10_d2_w72

# Ask depth derivatives (8 features)
bar5s_deriv_dask10_d1_w3
bar5s_deriv_dask10_d1_w12
bar5s_deriv_dask10_d1_w36
bar5s_deriv_dask10_d1_w72
bar5s_deriv_dask10_d2_w3
bar5s_deriv_dask10_d2_w12
bar5s_deriv_dask10_d2_w36
bar5s_deriv_dask10_d2_w72

# Near-bid depth derivatives (8 features)
bar5s_deriv_dbelow01_d1_w3
bar5s_deriv_dbelow01_d1_w12
bar5s_deriv_dbelow01_d1_w36
bar5s_deriv_dbelow01_d1_w72
bar5s_deriv_dbelow01_d2_w3
bar5s_deriv_dbelow01_d2_w12
bar5s_deriv_dbelow01_d2_w36
bar5s_deriv_dbelow01_d2_w72

# Near-ask depth derivatives (8 features)
bar5s_deriv_dabove01_d1_w3
bar5s_deriv_dabove01_d1_w12
bar5s_deriv_dabove01_d1_w36
bar5s_deriv_dabove01_d1_w72
bar5s_deriv_dabove01_d2_w3
bar5s_deriv_dabove01_d2_w12
bar5s_deriv_dabove01_d2_w36
bar5s_deriv_dabove01_d2_w72

# Wall z-score derivatives (16 features)
bar5s_deriv_wbidz_d1_w3
bar5s_deriv_wbidz_d1_w12
bar5s_deriv_wbidz_d1_w36
bar5s_deriv_wbidz_d1_w72
bar5s_deriv_wbidz_d2_w3
bar5s_deriv_wbidz_d2_w12
bar5s_deriv_wbidz_d2_w36
bar5s_deriv_wbidz_d2_w72
bar5s_deriv_waskz_d1_w3
bar5s_deriv_waskz_d1_w12
bar5s_deriv_waskz_d1_w36
bar5s_deriv_waskz_d1_w72
bar5s_deriv_waskz_d2_w3
bar5s_deriv_waskz_d2_w12
bar5s_deriv_waskz_d2_w36
bar5s_deriv_waskz_d2_w72
```

---

## Appendix B — Cumulative Feature List

```
# Trade cumulatives (5 features)
bar5s_cumul_trade_vol
bar5s_cumul_signed_trade_vol
bar5s_cumul_aggbuy_vol
bar5s_cumul_aggsell_vol
bar5s_cumul_signed_trade_vol_rate

# Flow cumulatives - totals (4 features)
bar5s_cumul_flow_net_bid
bar5s_cumul_flow_net_ask
bar5s_cumul_flow_imbal
bar5s_cumul_flow_imbal_rate

# Flow cumulatives - by band (10 features)
bar5s_cumul_flow_net_bid_p0_1
bar5s_cumul_flow_net_bid_p1_2
bar5s_cumul_flow_net_bid_p2_3
bar5s_cumul_flow_net_bid_p3_5
bar5s_cumul_flow_net_bid_p5_10
bar5s_cumul_flow_net_ask_p0_1
bar5s_cumul_flow_net_ask_p1_2
bar5s_cumul_flow_net_ask_p2_3
bar5s_cumul_flow_net_ask_p3_5
bar5s_cumul_flow_net_ask_p5_10

# Event count cumulatives (4 features)
bar5s_cumul_msg_cnt
bar5s_cumul_trade_cnt
bar5s_cumul_add_cnt
bar5s_cumul_cancel_cnt
```

---

## Appendix C — Setup Summary Feature List

```
# Trajectory summary (11 features)
bar5s_setup_start_dist_pts
bar5s_setup_min_dist_pts
bar5s_setup_max_dist_pts
bar5s_setup_dist_range_pts
bar5s_setup_approach_bars
bar5s_setup_retreat_bars
bar5s_setup_approach_ratio
bar5s_setup_early_velocity
bar5s_setup_mid_velocity
bar5s_setup_late_velocity
bar5s_setup_velocity_trend

# Book pressure summary (10 features)
bar5s_setup_obi0_start
bar5s_setup_obi0_end
bar5s_setup_obi0_delta
bar5s_setup_obi0_min
bar5s_setup_obi0_max
bar5s_setup_obi10_start
bar5s_setup_obi10_end
bar5s_setup_obi10_delta
bar5s_setup_obi10_min
bar5s_setup_obi10_max

# Flow summary (4 features)
bar5s_setup_total_trade_vol
bar5s_setup_total_signed_vol
bar5s_setup_trade_imbal_pct
bar5s_setup_flow_imbal_total

# Wall summary (5 features)
bar5s_setup_bid_wall_max_z
bar5s_setup_ask_wall_max_z
bar5s_setup_bid_wall_bars
bar5s_setup_ask_wall_bars
bar5s_setup_wall_imbal
```
