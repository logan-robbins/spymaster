# Features Module Analysis

**Date**: 2025-12-23  
**Status**: ✅ Operational, Partially Integrated  
**Test Coverage**: 29/29 tests passing

---

## Overview

The `features/` module contains two engines originally designed as "Agent A" (Physics) and "Agent B" (Context) for level analysis:

1. **`context_engine.py`**: ✅ **ACTIVELY USED** - Structural level identification + timing context
2. **`physics_engine.py`**: ⚠️ **LEGACY** - Prototype replaced by production engines in `core/`

---

## Module Status by Component

### 1. ContextEngine (Agent B) - ✅ ACTIVELY USED

**Purpose**: Macro context - structural levels and market timing

**Integration Points**:
- ✅ Used in `src/pipeline/vectorized_pipeline.py`
- ✅ Used in `src/pipeline/batch_process.py`
- ✅ Referenced in pipeline tests

**Key Functionality**:
| Method | Purpose | Status |
|--------|---------|--------|
| `is_first_15m()` | Detects opening volatility period (09:30-09:45 ET) | ✅ Production |
| `get_premarket_high/low()` | Cached PM levels from 04:00-09:30 ET | ✅ Production |
| `get_sma_200_at_time()` | SMA-200 on 2-min bars with forward fill | ✅ Production |
| `get_sma_400_at_time()` | SMA-400 on 2-min bars with forward fill | ✅ Production |
| `get_active_levels()` | Returns levels within $0.10 tolerance | ✅ Production |

**Data Processing**:
- Converts 1-min OHLCV to 2-min bars for SMA calculation
- Handles timezone conversion (UTC → ET)
- Caches expensive computations (PM high/low, SMAs)
- Forward-fills SMA values for intrabar queries

**Test Coverage**: 19 tests, 100% pass rate

**Dependencies**:
- `src.common.schemas.levels_signals.LevelKind`
- `pandas` for time series operations
- `pytz` for timezone handling

---

### 2. PhysicsEngine (Agent A) - ⚠️ LEGACY

**Purpose**: Microstructure metrics (wall ratio, replenishment, tape velocity)

**Current Status**: **Prototype replaced by production engines**

**Replacement Mapping**:
| PhysicsEngine Method | Replaced By | Location |
|---------------------|-------------|----------|
| `calculate_wall_ratio()` | `BarrierEngine.compute_barrier_state()` | `core/barrier_engine.py` |
| `detect_replenishment()` | `BarrierEngine.compute_barrier_state()` | `core/barrier_engine.py` |
| `calculate_tape_velocity()` | `TapeEngine.compute_tape_state()` | `core/tape_engine.py` |

**Why Replaced**:
1. ✅ **Production engines** in `core/` have:
   - Full MarketState integration
   - Vectorized batch processing
   - Proper configuration via CONFIG
   - EWMA smoothing
   - More sophisticated state classification (VACUUM, WALL, ABSORPTION, etc.)

2. ❌ **PhysicsEngine** limitations:
   - Hardcoded constants (DEFAULT_AVG_VOLUME = 5000)
   - No integration with MarketState
   - Simple logic vs production state machines
   - No smoothing/filtering

**Integration Points**:
- ⚠️ **NOT USED** in production pipeline
- ⚠️ Only used in legacy tests (`test_physics_engine.py`, `test_agent_a_b_integration.py`)

**Test Coverage**: 10 tests, 100% pass rate (but testing legacy code)

**Dependencies**:
- `src.common.event_types` (MBP10, FuturesTrade, etc.)
- Uses actual schemas (good design)

---

## Architecture Relationship

### Current Production Architecture

```
Pipeline Flow (from vectorized_pipeline.py):
1. Load ES futures data (DBN) → MarketState
2. Load SPY option data → MarketState
3. Build OHLCV from ES trades
4. Initialize ContextEngine(ohlcv_df)           ← features/context_engine.py ✅
5. Initialize BarrierEngine(config)             ← core/barrier_engine.py ✅
6. Initialize TapeEngine(config)                ← core/tape_engine.py ✅
7. Initialize FuelEngine(config)                ← core/fuel_engine.py ✅
8. Detect touches via ContextEngine             ← features/context_engine.py ✅
9. Calculate physics via production engines     ← core/* ✅
10. Label outcomes → research signals
```

### Legacy vs Production

| Feature | PhysicsEngine (Legacy) | Production Engines (core/) |
|---------|------------------------|---------------------------|
| Wall detection | Simple ratio | VACUUM/WALL/ABSORPTION states |
| Replenishment | Boolean + latency | Continuous ratio + trend |
| Tape metrics | Velocity only | Velocity + imbalance + sweeps |
| Configuration | Hardcoded | CONFIG singleton |
| Smoothing | None | EWMA on scores/velocities |
| Integration | Standalone | MarketState + ring buffers |
| Batch processing | No | Vectorized via `vectorized_engines.py` |

---

## Test Coverage Analysis

### ContextEngine Tests (19 tests) - ✅ Excellent Coverage

**Time Context (6 tests)**:
- ✅ `test_is_first_15m_true` - Opening period detection
- ✅ `test_is_first_15m_false_before` - Before market open
- ✅ `test_is_first_15m_false_after` - After opening period
- ✅ `test_is_first_15m_exactly_0930` - Boundary: market open
- ✅ `test_is_first_15m_exactly_0945` - Boundary: 15 min mark
- ✅ `test_full_day_scenario` - End-to-end day simulation

**Level Detection (8 tests)**:
- ✅ `test_premarket_high_low_calculation` - PM level computation
- ✅ `test_get_active_levels_pm_high` - PM high within tolerance
- ✅ `test_get_active_levels_pm_low` - PM low within tolerance
- ✅ `test_get_active_levels_no_levels_far_from_price` - Outside tolerance
- ✅ `test_get_active_levels_sma_200` - SMA level detection
- ✅ `test_level_tolerance` - Tolerance boundary testing
- ✅ `test_multiple_levels_detected` - Confluence scenarios
- ✅ `test_level_detection_workflow` - Integration test

**SMA Calculation (2 tests)**:
- ✅ `test_sma_200_calculation` - 2-min bar SMA
- ✅ `test_sma_200_insufficient_data` - Warmup period handling

**Edge Cases (3 tests)**:
- ✅ `test_empty_ohlcv_dataframe` - Empty data
- ✅ `test_no_dataframe_initialization` - No data init
- ✅ `test_generate_mock_ohlcv` - Mock data generator

### PhysicsEngine Tests (10 tests) - ✅ Good Coverage (Legacy Code)

**Wall Ratio (3 tests)**:
- ✅ `test_calculate_wall_ratio_with_mbp10` - Standard calculation
- ✅ `test_calculate_wall_ratio_empty_mbp10` - Empty book
- ✅ `test_wall_ratio_with_price_tolerance` - Price matching

**Tape Velocity (2 tests)**:
- ✅ `test_calculate_tape_velocity_with_futures_trades` - Standard calc
- ✅ `test_calculate_tape_velocity_no_trades` - Empty tape

**Replenishment (2 tests)**:
- ✅ `test_detect_replenishment_success` - Successful reload
- ✅ `test_detect_replenishment_no_sweep` - No sweep event

**Mock Data (2 tests)**:
- ✅ `test_mock_mbp10_generator` - MBP-10 mock
- ✅ `test_mock_trades_generator` - Trade tape mock

**Integration (1 test)**:
- ✅ `test_integration_complete_workflow` - End-to-end

---

## Critical Findings

### ✅ Strengths

1. **ContextEngine is production-ready**:
   - Well-tested (19 tests)
   - Actively used in pipeline
   - Clean API design
   - Proper timezone handling
   - Efficient caching

2. **PhysicsEngine demonstrates good design**:
   - Uses actual event schemas
   - Mock data generators for testing
   - Clear documentation
   - Comprehensive test coverage

3. **Clear separation of concerns**:
   - Context (macro) vs Physics (micro)
   - Standalone modules with minimal dependencies

### ⚠️ Issues & Recommendations

#### 1. PhysicsEngine is Orphaned Legacy Code

**Problem**: PhysicsEngine is not used in production pipeline but is still maintained with tests.

**Recommendation**:
```
Option A: Delete PhysicsEngine
- Remove src/features/physics_engine.py
- Remove tests/test_physics_engine.py
- Remove tests/test_agent_a_b_integration.py
- Update documentation to clarify ContextEngine only

Option B: Keep as Educational Reference
- Move to src/research/prototypes/
- Add README explaining it's a prototype
- Keep tests for reference but mark as legacy
```

**Preferred**: Option A (delete) unless there's historical/educational value.

#### 2. Naming Confusion: "Agent A" vs "Agent B"

**Problem**: "Agent" terminology is not used elsewhere in codebase.

**Recommendation**:
- Remove "Agent A"/"Agent B" comments
- Use descriptive names: "Context Analysis" vs "Microstructure Physics"

#### 3. Missing from Main Documentation

**Problem**: `features/` module not documented in:
- `backend/README.md`
- `COMPONENTS.md`
- `backend/src/features/` has no README.md

**Recommendation**:
```bash
# Create features module documentation
backend/src/features/README.md

# Update COMPONENTS.md to mention ContextEngine usage
# Update backend/README.md to list features/ in module overview
```

#### 4. ContextEngine Could Support More Levels

**Current**: PM High/Low, SMA-200, SMA-400  
**Missing**: Opening Range High/Low, Session High/Low, VWAP

**Recommendation**: Extend `get_active_levels()` to include:
- Opening Range (first 30 min high/low)
- Session High/Low (current session extremes)
- VWAP calculation on OHLCV data

These are mentioned in `features.json` and `LevelKind` enum but not implemented in ContextEngine.

#### 5. No INTERFACES.md

**Problem**: Features module has no interface documentation.

**Recommendation**: Create `backend/src/features/INTERFACES.md` documenting:
- ContextEngine API contract
- Input/output schemas
- Integration points with pipeline
- Configuration parameters

---

## Integration Status

### Where ContextEngine is Used

**Primary**:
1. `src/pipeline/vectorized_pipeline.py`:
   ```python
   from src.features.context_engine import ContextEngine
   context_engine = ContextEngine(ohlcv_df=ohlcv)
   is_first_15m = context_engine.is_first_15m(ts_ns)
   sma_200 = context_engine.get_sma_200_at_time(ts_ns)
   ```

2. `src/pipeline/`:
   - Uses modular stage-based pipelines for batch processing multiple dates

**Tests**:
3. `tests/test_context_engine.py` - Unit tests
4. `tests/test_agent_a_b_integration.py` - Integration with PhysicsEngine (legacy)

### Where ContextEngine Should Be Used (But Isn't)

**Missing Integration**:
1. ❌ **Core Service** (`src/core/service.py`, `src/core/level_signal_service.py`):
   - Currently uses `LevelUniverse` for level detection
   - Could leverage ContextEngine for PM high/low, SMA levels
   - Would unify level detection logic

2. ❌ **Vectorized Pipeline** (`src/pipeline/vectorized_pipeline.py`):
   - Has its own `detect_touches_vectorized()` function
   - Could use ContextEngine for level identification
   - Potential duplication of logic

---

## Performance Considerations

### ContextEngine Efficiency

**Caching Strategy**:
- ✅ PM high/low computed once on initialization
- ✅ SMA values computed once and indexed by timestamp
- ✅ Forward-fill lookup is O(log n) via binary search

**Potential Optimizations**:
1. **Pre-compute level grids**:
   - Instead of checking each level on each query
   - Build time-indexed level grid once

2. **Vectorize tolerance checks**:
   - Use numpy for batch level distance calculations
   - Especially useful for `detect_touches` in pipeline

3. **Memoize active level queries**:
   - Cache results for repeated queries at same timestamp
   - Use LRU cache decorator

---

## Recommendations Summary

### High Priority

1. **✅ CLARIFY PHYSICS ENGINE STATUS**:
   - Either delete or move to prototypes
   - Update documentation

2. **📝 CREATE FEATURES/ DOCUMENTATION**:
   - Add `README.md` explaining module purpose
   - Add `INTERFACES.md` with API contracts
   - Update `COMPONENTS.md` to reference ContextEngine

3. **🔧 EXTEND CONTEXT ENGINE**:
   - Add Opening Range High/Low
   - Add Session High/Low tracking
   - Add VWAP calculation
   - Match completeness of `LevelKind` enum

### Medium Priority

4. **🔄 UNIFY LEVEL DETECTION**:
   - Investigate using ContextEngine in Core Service
   - Reduce duplication with LevelUniverse
   - Centralize level detection logic

5. **⚡ OPTIMIZE PERFORMANCE**:
   - Pre-compute level grids
   - Vectorize distance calculations
   - Add memoization for repeated queries

### Low Priority

6. **🧪 EXPAND TEST COVERAGE**:
   - Add integration test with actual DBN data
   - Test SMA-400 extensively (currently basic coverage)
   - Test edge cases: DST transitions, market holidays

---

## Action Items

### Immediate (This Session)

- [x] Analyze features module structure
- [x] Run existing tests (29/29 pass)
- [x] Document findings in ANALYSIS.md
- [ ] Create README.md for features/
- [ ] Create INTERFACES.md for features/
- [ ] Update COMPONENTS.md

### Next Session

- [ ] Decide PhysicsEngine fate (delete vs archive)
- [ ] Extend ContextEngine with OR/Session High/Low
- [ ] Add VWAP calculation
- [ ] Integrate ContextEngine into Core Service

---

## Conclusion

**Overall Assessment**: ✅ **Healthy but Incomplete**

**What Works**:
- ContextEngine is production-ready and actively used
- Excellent test coverage (29/29 passing)
- Clean API design with proper separation of concerns

**What Needs Attention**:
- PhysicsEngine orphaned legacy code
- Missing documentation (README, INTERFACES)
- Incomplete level support (OR high/low, session extremes, VWAP)
- Potential duplication with LevelUniverse logic

**Priority**: Medium (not blocking, but would benefit from cleanup and expansion)

---

**Test Summary**:
- ContextEngine: 19/19 tests ✅
- PhysicsEngine: 10/10 tests ✅ (legacy)
- Total: 29/29 tests passing
- Coverage: Excellent for current functionality
- Gaps: Missing tests for extended levels (OR, session, VWAP)

