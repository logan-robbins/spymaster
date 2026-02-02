# GRUNT_LOG

## 2026-02-01 Orphaned legacy data cleanup
- Deleted `backend/src/data_eng/analysis/mbo_preview_validation.py` (broken imports: `BronzeIngestMboPreview`, `compute_level_vacuum_5s.py` modules no longer exist)
- Verified orphaned silver tables (`book_wall_1s`, `gex_5s`, `gex_surface_1s`, `gex_flow_surface_1s`, `book_flow_1s`) already cleaned up - `lake/silver/product_type=future_option_mbo/` does not exist
- Updated README.md and GRUNT_LOG.md to remove orphaned table references

## 2026-02-01 silver.future_mbo FuturesBookEngine._fill_order fix
- **Bug**: `depth_qty_rest > depth_qty_end` for ~1.1% of rows
- **Root cause**: Databento MBO fill events for aggressor orders report trade size instead of order's fill amount. When fill_qty > order.qty, depth was over-reduced.
- **Fix**: Cap fill quantity at order's remaining quantity: `actual_fill = min(fill_qty, order.qty)`
- **File**: `backend/src/data_eng/stages/silver/future_mbo/book_engine.py`
- **Additional fixes**:
  - Added F_LAST flag handling to exit snapshot mode after snapshot sequence completes
  - Added spot_ref fallback to compute at window end if it was 0 at window start
  - Added HUD_MAX_TICKS alias for backward compatibility with tests
- **Validation**: 0% violations across 2026-01-05, 2026-01-06, 2026-01-07

## 2026-02-01 silver.future_option_mbo OptionsBookEngine.depth_start tracking fix
- **Bug**: 0.08% of rows had negative `depth_qty_start` when calculated via formula
- **Root cause**: Formula `depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty` breaks when aggregating per-instrument flows to $5 strike buckets. Multiple instruments at same bucket can have add/cancel flows that sum to negative start.
- **Fix**: Track `depth_qty_start` explicitly in `options_book_engine.py`:
  - Added `win_depth_start` dict to capture depth BEFORE first event in each window
  - Capture `depth_start` when key first accessed in window (before ADD/FILL/CANCEL/MODIFY)
  - Output `depth_start` directly instead of deriving from formula
  - Aggregate `depth_start` sums in `compute_book_states_1s.py`
- **Files**:
  - `backend/src/data_eng/stages/silver/future_option_mbo/options_book_engine.py`
  - `backend/src/data_eng/stages/silver/future_option_mbo/compute_book_states_1s.py`
- **Validation**: 0% negative depth_qty_start across 2026-01-06, 2026-01-07
- **Note**: Accounting identity mismatch ~32% is expected (identity holds per-instrument, not after bucketing)

## 2026-02-02 bronze.equity_mbo backfill and validation
- **Backfill**: Ran bronze pipeline for all 18 trading days (2026-01-05 to 2026-01-29)
- **Previously**: Only 4 dates existed (2026-01-08, 2026-01-09, 2026-01-16, 2026-01-27)
- **New validation script**: `backend/scripts/validate_bronze_equity_mbo.py`
- **Validation results (3 random dates: 2026-01-13, 2026-01-22, 2026-01-28)**:
  - Schema: PASS (15 fields match avsc contract)
  - Timestamp ordering: strict monotonic
  - Action codes: A, C, T, F (valid MBO actions)
  - Side codes: A, B, N (valid)
  - 0 null values in critical fields
  - Price range for trades/fills: $620-$634 (expected QQQ range)
  - Stub quotes detected at $1000-$199999 (expected market maker behavior)
  - Total: 304.8M rows, 3.4 GB parquet
- **Coverage**: 18/18 trading days (full)

## 2026-02-01 silver.equity_option_cmbp_1 Cmbp1BookEngine._reset_accumulators fix
- **Bug**: `AttributeError: 'Cmbp1BookEngine' object has no attribute '_reset_accumulators'`
- **Root cause**: Method `_reset_accumulators()` was called in `_start_window()` and `_emit_window()` but never defined
- **Fix**: Added `_reset_accumulators()` method to clear `acc_add` and `acc_pull` dictionaries between windows
- **File**: `backend/src/data_eng/stages/silver/equity_option_cmbp_1/cmbp1_book_engine.py`
- **Tests added**: `backend/tests/test_cmbp1_book_engine.py` (14 tests, all passing)
- **Validation**: 0% null values, 0 crossed books, accounting identity holds across 2026-01-07, 2026-01-15, 2026-01-27
