# Pipeline Module

**Purpose**: VectorizedPipeline - core feature computation engine  
**Status**: Production  
**Primary Consumer**: SilverFeatureBuilder  
**Architecture**: See [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)

---

## Overview

Provides the VectorizedPipeline class - a high-performance feature computation engine used internally by SilverFeatureBuilder to transform Bronze data into Silver features.

**Key Principle**: This is a utility class, not a standalone tool. Used by SilverFeatureBuilder in the Medallion pipeline.

---

## VectorizedPipeline

**File**: `vectorized_pipeline.py`  
**Purpose**: High-performance batch processing using vectorized numpy operations optimized for Apple M4 Silicon

**Key Stages**:
1. **Data Loading**: Read Bronze parquet (ES trades, MBP-10, option trades)
2. **OHLCV Building**: 1-min and 2-min bars from ES trades (converted to SPY scale)
3. **Warmup**: Load 3 prior days for SMA-200/400, 7 prior days for relative volume
4. **Level Universe**: Generate structural levels (PM/OR/SMA/VWAP/Walls)
5. **Touch Detection**: Numpy broadcasting for monitor band filtering
6. **Physics Computation**: Batch metrics from barrier/tape/fuel engines
7. **Feature Transforms**: Normalization, sparse encoding, confluence features
8. **Labeling**: Forward-looking outcome determination (competing risks)
9. **RTH Filtering**: Output filtered to 09:30-16:00 ET with full forward window

**Usage** (via SilverFeatureBuilder):
```python
from src.lake.silver_feature_builder import SilverFeatureBuilder

builder = SilverFeatureBuilder()
builder.build_feature_set(
    manifest=manifest,
    dates=['2025-12-16', '2025-12-17']
)
# Internally calls VectorizedPipeline.run() for each date
```

**Direct Usage** (development/testing only):
```python
from src.pipeline.vectorized_pipeline import VectorizedPipeline

pipeline = VectorizedPipeline()
signals_df = pipeline.run(date='2025-12-16')
# Returns DataFrame with ~800-3000 signals (RTH only)
```

---

## Level Universe (SPY-Specific)

VectorizedPipeline generates structural levels for SPY:

1. **PM_HIGH/PM_LOW**: Pre-market high/low (04:00-09:30 ET)
   - Computed from Bronze pre-market data
   - Used as features but pre-market signals are filtered out

2. **OR_HIGH/OR_LOW**: Opening range (09:30-09:45 ET)

3. **SESSION_HIGH/SESSION_LOW**: Running session extremes

4. **SMA_200/SMA_400**: Moving averages on 2-min bars
   - Requires 3 prior days warmup

5. **VWAP**: Session volume-weighted average price

6. **CALL_WALL/PUT_WALL**: Max gamma concentration strikes

**Note**: ROUND ($5, $10) and STRIKE (generic) levels are disabled for SPY due to duplicative $1 strike spacing.

---

## Feature Categories

**Barrier Physics** (ES MBP-10 depth):
- `barrier_state`: VACUUM/WALL/ABSORPTION/NEUTRAL
- `barrier_delta_liq`: Net liquidity change
- `barrier_replenishment_ratio`: Add/remove ratio
- `wall_ratio`: Concentration vs baseline

**Tape Physics** (ES trade flow):
- `tape_imbalance`: Buy/sell ratio [-1, 1]
- `tape_velocity`: Trade arrival rate
- `sweep_detected`: Aggressive multi-level execution

**Fuel Physics** (SPY option gamma):
- `gamma_exposure`: Net dealer gamma at level
- `fuel_effect`: AMPLIFY/DAMPEN/NEUTRAL
- `gamma_flow_velocity`: Flow per minute
- `dealer_pressure`: Normalized pressure score

**Approach Context**:
- `distance_atr`: ATR-normalized distance to level
- `approach_velocity`: $/min toward level
- `attempt_index`: Touch count in cluster
- `prior_touches`: Historical touches at level

**Confluence Features**:
- `confluence_level`: Hierarchical setup quality (1-10 scale)
- `breakout_state`: Multi-timeframe trend alignment
- `gex_alignment`: Gamma exposure alignment with direction
- `rel_vol_ratio`: Current vs 7-day average volume

**Labels** (competing risks):
- `outcome`: BREAK_1/BOUNCE_1/BREAK_2/BOUNCE_2/CHOP
- `strength_signed`: Signed distance moved
- `t1_60`, `t1_120`: Time to first threshold (seconds)
- `t2_60`, `t2_120`: Time to second threshold
- `t1_break_60`, `t1_bounce_60`: Directional timing
- `tradeable_1`, `tradeable_2`: Binary tradeable flags

---

## RTH Filtering

**Critical**: Silver and Gold datasets contain ONLY RTH (09:30-16:00 ET) signals.

**Implementation** (lines 2814-2829 of vectorized_pipeline.py):
- Filters output to 09:30-16:00 ET
- Ensures full forward window available (no partial labels at 4pm)
- Pre-market data still used for feature computation (PM_HIGH/PM_LOW)

---

## Performance Optimizations

**Apple M4 Silicon Optimized**:
- All operations use numpy broadcasting (no Python loops)
- Batch processing of all touches simultaneously
- Memory-efficient chunked processing
- Optional Numba JIT compilation for hot paths

**Performance Targets**:
- Process 1M+ trades in <10 seconds
- Generate 10K+ signals per day
- Memory usage <16GB for full day

**Actual Performance** (M4 Mac, 128GB RAM):
- Single date: ~2-5 seconds
- 10 dates: ~30-60 seconds
- ~500-1000 signals/sec throughput

---

## Integration with SilverFeatureBuilder

VectorizedPipeline is called internally by SilverFeatureBuilder. For each date, it:
1. Loads Bronze data (ES trades + MBP-10 + options)
2. Builds OHLCV (1min, 2min bars)
3. Loads warmup data (SMA, relative volume)
4. Generates level universe
5. Detects touches (monitor band)
6. Computes physics features (barrier/tape/fuel)
7. Computes approach context
8. Labels outcomes (competing risks)
9. Filters to RTH (9:30-16:00)
10. Returns in-memory DataFrame to SilverFeatureBuilder

---

## Configuration

Pipeline behavior is controlled by `backend/src/common/config.py`:
- Physics windows: `W_b`, `W_t`, `W_g` (barrier, tape, fuel)
- Touch detection: `MONITOR_BAND`, `TOUCH_BAND`
- Confirmation windows: `CONFIRMATION_WINDOWS_MULTI`
- Warmup: `SMA_WARMUP_DAYS`, `VOLUME_LOOKBACK_DAYS`
- RTH session: Hardcoded 09:30-16:00 ET

---

## Testing

```bash
cd backend

# Test VectorizedPipeline (via unit tests)
uv run pytest tests/test_vectorized_pipeline.py -v

# Test end-to-end determinism
uv run pytest tests/test_replay_determinism.py -v
```

---

## Common Issues

**"No signals generated"**:
- Check Bronze data exists for date: `ls data/lake/bronze/futures/trades/symbol=ES/date=2025-12-16/`
- Verify warmup data available (3 prior days)

**"High null rates in features"**:
- Option trades may be sparse: `gamma_exposure` can have 10-15% nulls
- MBP-10 gaps: `barrier_state` should be <1% null

**"Performance slow"**:
- Check if Numba is installed: `pip list | grep numba`
- Verify sufficient RAM (16GB+ recommended)
- Consider reducing date range if memory-constrained

---

## References

- **Architecture & Workflow**: [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)
- **Feature Manifests**: [../common/schemas/feature_manifest.py](../common/schemas/feature_manifest.py)
- **Physics Engines**: [../core/](../core/) (barrier, tape, fuel)
- **Silver Builder**: [../lake/silver_feature_builder.py](../lake/silver_feature_builder.py)
- **Configuration**: [../common/config.py](../common/config.py)
