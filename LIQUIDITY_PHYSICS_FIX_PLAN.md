# Liquidity Physics Implementation Plan

## Problem Statement
Ablation studies show liquidity physics features underperforming. Root cause: features fail to capture dealer positioning at target levels during approach.

## Core Issues Identified

1. **Spatial Blindness**: System cannot see liquidity AT the target level until price touches it (`dist <= 0.5 pts`)
2. **Moving Reference Frame**: Bands use microprice (moves continuously), causing liquidity to shift between bands without actual changes
3. **Missing Level-Specific Physics**: No tracking of size changes AT the specific target price
4. **No Temporal Weighting**: Early approach activity weighted equally to late approach activity

## Implementation Steps

### 1. Add Price Levels to bar5s Output
**Status**: COMPLETE
**File**: `backend/src/data_eng/stages/silver/future/mbp10_bar5s/compute.py`
**Change**: Output `bar5s_shape_bid_px_l{i:02d}_eob` and `bar5s_shape_ask_px_l{i:02d}_eob` alongside sizes
**Reason**: Required for spatial lookahead to match level_price against actual order book rows
**Note**: Contract already had price fields defined

### 2. Implement Spatial Lookahead Features
**Status**: COMPLETE
**File**: `backend/src/data_eng/stages/silver/future/compute_approach_features.py`
**New Features**:
- `bar5s_lvl_bid_size_at_level_eob`: Size at exact level price on bid side (0 if not visible)
- `bar5s_lvl_ask_size_at_level_eob`: Size at exact level price on ask side (0 if not visible)
- `bar5s_lvl_total_size_at_level_eob`: Combined size at level
- `bar5s_lvl_level_is_visible`: Boolean flag if level is within MBP-10 range
- `bar5s_lvl_level_book_index_bid/ask`: Which ladder index contains the level (-1 if not visible)
- `bar5s_lvl_size_at_level_imbal_eob`: Imbalance at level
- `bar5s_lvl_wall_at_level`: Boolean if strong wall detected AT the target level

### 3. Implement Level-Centric Bands
**Status**: COMPLETE
**File**: `backend/src/data_eng/stages/silver/future/compute_approach_features.py`
**New Features**:
- `bar5s_lvl_depth_band_0to1_qty_eob`: Sum of sizes 0.0-1.0 pts from level
- `bar5s_lvl_depth_band_1to2_qty_eob`: Sum of sizes 1.0-2.0 pts from level
- `bar5s_lvl_depth_band_beyond2_qty_eob`: Sum beyond 2.0 pts
- `bar5s_lvl_depth_band_{}_frac_eob`: Fractions for each band
**Logic**: Use `abs(price - level_price) / POINT` as fixed distance metric (level-centric, not microprice-centric)

### 4. Implement Liquidity Velocity at Level
**Status**: COMPLETE
**File**: `backend/src/data_eng/stages/silver/future/compute_approach_features.py`
**New Features**:
- `bar5s_lvl_size_at_level_d1_w3`: First derivative of size at level (3-bar window)
- `bar5s_lvl_size_at_level_d1_w12`: First derivative (12-bar window)
- `bar5s_lvl_size_at_level_d2_w12`: Second derivative (12-bar window)
**Calculation**: Group by touch_id, diff the `bar5s_lvl_total_size_at_level_eob` column

### 5. Implement Time-Weighted Physics
**Status**: COMPLETE
**File**: `backend/src/data_eng/stages/silver/future/compute_approach_features.py`
**New Features**:
- `bar5s_setup_size_at_level_start/end/delta/max`: Episode-level size tracking
- `bar5s_setup_size_at_level_recent12_sum`: Recent 12-bar sum
- `bar5s_setup_size_at_level_early_late_ratio`: Early third vs late third ratio
- `bar5s_setup_flow_toward_recent12_sum`: Recent flow toward level
- `bar5s_setup_flow_toward_early_late_ratio`: Early vs late flow comparison

### 6. Update Contracts
**Status**: COMPLETE
**Files**: 
- `backend/src/data_eng/contracts/silver/future/market_by_price_10_bar5s.avsc` (prices already present)
- `backend/src/data_eng/contracts/silver/future/market_by_price_10_level_approach.avsc`
**Change**: Added all new spatial lookahead, level-centric band, velocity, and time-weighted fields

### 7. Verify Changes
**Status**: COMPLETE
**Tests Run**: Logic verification script
**Results**: 
- ✓ Spatial Lookahead: Correctly identifies size at level (1000 contracts at 6802 while at 6800)
- ✓ Level-Centric Bands: Stable reference frame (150 contracts in 0-1pt band regardless of microprice movement)
- ✓ Liquidity Velocity: Derivatives capture acceleration (183/bar vs 83/bar as wall builds)

## Success Criteria
- [x] Prices available in bar5s output
- [x] Size at level visible when within 2.5 pts
- [x] Level-centric bands stable as microprice moves
- [x] Derivatives of level size computed per episode
- [x] Recent activity features capture late-stage positioning
- [ ] Single date runs successfully through pipeline (requires reprocessing data)

## Next Steps
To fully deploy these changes:
1. Reprocess historical data through the updated pipeline stages
2. Retrain models with the new physics features
3. Run ablation study to compare feature importance before/after

