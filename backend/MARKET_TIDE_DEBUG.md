# Market Tide Debugging Session - Complete Fix

**Date**: 2026-01-01  
**Session Duration**: ~3 hours  
**Objective**: Resolve Market Tide similarity inversion and ensure research-quality data

## Problems Discovered

### 1. Call/Put Premium Conflation (Bronze Layer)
**File**: `src/core/batch_engines.py`  
**Issue**: `call_tide` and `put_tide` were both using the combined `strike_premium` dictionary instead of separate premium flows.  
**Impact**: Both features were identical (correlation = 1.0), losing directional information.  
**Fix**: Created separate `call_premium` and `put_premium` dictionaries, populated them independently during option flow aggregation.

### 2. Market Tide Missing from Normalization
**File**: `src/ml/normalization.py`  
**Issue**: `call_tide` and `put_tide` were not classified in any normalization category (ROBUST, ZSCORE, MINMAX, PASSTHROUGH).  
**Impact**: Features were inserted into episode vectors **raw** (values in millions), completely dominating cosine similarity.  
**Fix**: Added to `ROBUST_FEATURES` for robust scaling (median/IQR), appropriate for heavy-tailed premium distributions.

### 3. Market Tide Missing from State Table
**File**: `src/pipeline/stages/materialize_state_table.py`  
**Issue**: Features existed in `signals.parquet` but weren't copied to `state.parquet`. Normalization script reads from state table.  
**Impact**: Normalization computed stats on all zeros (IQR=0), resulting in constant normalized values (4.0).  
**Fix**: Added `call_tide` and `put_tide` to the `feature_cols` list for forward-filling from signals to state.

## Data Audit Results

**Before Fixes** (version 4.0.0 - buggy):
- Call Tide = Put Tide (identical, correlation 1.0)
- Similarity inversion: -80.5%

**After Premium Separation** (version 4.5.0 - partial fix):
- Call Tide: Mean 4.99M ≠ Put Tide: Mean 24.7M ✅
- But still inverted: -81.1%
- Root cause: Not normalized, dominating vectors

**After Full Fix** (pending regeneration):
- Expected: Robust normalization will scale to comparable range
- Expected: Inversion eliminated, positive similarity scaling

## Files Modified

1. `src/core/batch_engines.py`
   - Added `call_premium`, `put_premium` to `VectorizedMarketData`
   - Separated premium aggregation in `build_vectorized_market_data`
   - Updated `compute_fuel_metrics_batch` to use separate premium arrays

2. `src/ml/normalization.py`
   - Added `call_tide`, `put_tide` to `ROBUST_FEATURES`

3. `src/pipeline/stages/materialize_state_table.py`
   - Added `call_tide`, `put_tide` to state table `feature_cols` list

## Regeneration Plan

**Step 1**: Bronze→Silver (Jun 5 - Sept 30, 82 days)
- Rebuild signals with separated call/put premiums
- Rebuild state tables with Market Tide included
- Command: `run_pipeline --pipeline bronze_to_silver --start 2025-06-05 --end 2025-09-30 --workers 8`

**Step 2**: Recompute Normalization Stats
- Read from state tables (now includes Market Tide)
- Compute robust stats (median, IQR) for proper scaling
- Command: `compute_normalization_stats.py`

**Step 3**: Silver→Gold (Episode Vectors)
- Rebuild vectors with properly normalized Market Tide
- Command: `run_pipeline --pipeline silver_to_gold --start 2025-06-05 --end 2025-09-30 --workers 8`

**Step 4**: Rerun Ablation Study
- Test Physics/Geometry/Market Tide/Combined on Sept 15-30 validation
- Command: `run_physics_ablation.py --start-date 2025-09-15 --end-date 2025-09-30 --version 4.0.0`

## Expected Outcome

Market Tide should exhibit:
- **Distinct values** from Call vs Put (already achieved in 4.5.0)
- **Proper normalization** (scaled to ~same range as OFI, velocity)
- **Positive or near-zero similarity scaling** (not -80% inversion)
- **Meaningful contribution** to combined vector accuracy

If inversion persists after proper normalization, it suggests Market Tide is fundamentally unsuitable for vector similarity and should be used as a categorical filter/regime classifier instead.
