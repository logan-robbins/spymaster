# Pipeline Checkpointing System

## Validation Status

**Current Progress**: Stages 0-15 validated on 2025-12-16  
**Strategy**: Validate each stage individually on single date before batch processing  
**Next**: Run full date range (Nov 3 - Dec 19, 2025)

| Stage | Name | Status | Issues Found | Fixed |
|-------|------|--------|--------------|-------|
| 0 | load_bronze | ✅ PASS | None | N/A |
| 1 | build_ohlcv_1min | ✅ PASS | DatetimeIndex lost on reset_index | Yes |
| 1 | build_ohlcv_1min | ✅ PASS | Checkpoint serialization with index=False | Yes |
| 1 | build_ohlcv_1min | ✅ PASS | Incorrect rth_only=True (needs premarket) | Yes |
| 2 | build_ohlcv_2min | ✅ PASS | Warmup concatenation used ignore_index | Yes |
| 3 | init_market_state | ✅ PASS | None | N/A |
| 4 | generate_levels | ✅ PASS | DatetimeIndex not handled in level generation | Yes |
| 5 | detect_interaction_zones | ✅ PASS | DatetimeIndex not handled in zone detection | Yes |
| 6 | compute_physics | ✅ PASS | None | N/A |
| 7 | compute_multiwindow_kinematics | ✅ PASS | DatetimeIndex not handled in OHLCV for kinematics | Yes |
| 8 | compute_multiwindow_ofi | ✅ PASS | None | N/A |
| 9 | compute_barrier_evolution | ✅ PASS | None | N/A |
| 10 | compute_level_distances | ✅ PASS | Missing bar_idx in interaction events | Yes |
| 11 | compute_gex_features | ✅ PASS | None | N/A |
| 12 | compute_force_mass | ✅ PASS | Acceleration column mismatch | Yes |
| 13 | compute_approach_features | ✅ PASS | DatetimeIndex not handled in approach context | Yes |
| 14 | label_outcomes | ✅ PASS | DatetimeIndex not handled in labeling | Yes |
| 15 | filter_rth | ✅ PASS | None | N/A |

## Overview

Incremental pipeline execution with stage-by-stage checkpointing for:
- **Debugging**: Inspect intermediate outputs
- **Recovery**: Resume from failures without re-running all stages
- **Validation**: Verify each stage's correctness before proceeding
- **Development**: Iterate on later stages without re-running earlier ones

## Architecture

```
data/checkpoints/
└── es_pipeline/
    └── 2025-12-16/
        ├── stage_00/
        │   ├── metadata.json          # Stage name, timing, config hash
        │   ├── trades.parquet          # DataFrame outputs
        │   ├── mbp10_snapshots.pkl     # Complex object outputs
        │   └── option_trades_df.parquet
        ├── stage_01/
        │   ├── metadata.json
        │   ├── trades.parquet
        │   ├── ohlcv_1min.parquet
        │   └── ...
        └── ...
```

## Usage

### Basic Checkpointing

```bash
cd backend

# Run with checkpoints enabled
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints
```

### Incremental Execution

```bash
# Run first 3 stages only (0-2)
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --stop-at-stage 2

# Inspect stage 2 outputs
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --inspect 2

# Continue from stage 3 (loads stage 2 checkpoint)
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --resume-from-stage 3
```

### Stage-by-Stage Validation

```bash
# Validate each stage incrementally
for i in {0..15}; do
  echo "Running through stage $i..."
  uv run python -m scripts.run_pipeline_incremental \
    --date 2025-12-16 \
    --checkpoint-dir data/checkpoints \
    --stop-at-stage $i
  
  echo "Inspecting stage $i output..."
  uv run python -m scripts.run_pipeline_incremental \
    --date 2025-12-16 \
    --checkpoint-dir data/checkpoints \
    --inspect $i
  
  read -p "Continue? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    break
  fi
done
```

### Checkpoint Management

```bash
# List checkpoints
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --list

# Clear checkpoints
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --clear
```

## Stage Index Reference

| Idx | Stage Name | Outputs |
|-----|------------|---------|
| 0 | load_bronze | trades, mbp10_snapshots, option_trades_df |
| 1 | build_ohlcv (1min) | ohlcv_1min |
| 2 | build_ohlcv (2min) | ohlcv_2min |
| 3 | init_market_state | market_state |
| 4 | generate_levels | static_level_info, dynamic_levels |
| 5 | detect_interaction_zones | touches_df |
| 6 | compute_physics | signals_df (with physics) |
| 7 | compute_multiwindow_kinematics | signals_df (+ kinematics) |
| 8 | compute_multiwindow_ofi | signals_df (+ OFI) |
| 9 | compute_barrier_evolution | signals_df (+ barrier evolution) |
| 10 | compute_level_distances | signals_df (+ distances) |
| 11 | compute_gex_features | signals_df (+ GEX) |
| 12 | compute_force_mass | signals_df (+ F=ma) |
| 13 | compute_approach | signals_df (+ approach) |
| 14 | label_outcomes | signals_df (+ labels) |
| 15 | filter_rth | signals_df (RTH filtered) |

## Programmatic Usage

```python
from src.pipeline.pipelines.es_pipeline import build_es_pipeline
from src.pipeline.core.checkpoint import CheckpointManager

# Build pipeline
pipeline = build_es_pipeline()

# Run with checkpointing
signals_df = pipeline.run(
    date="2025-12-16",
    checkpoint_dir="data/checkpoints",
    stop_at_stage=5  # Run through stage 5
)

# Load checkpoint
manager = CheckpointManager("data/checkpoints")
ctx = manager.load_checkpoint(
    pipeline_name="es_pipeline",
    date="2025-12-16",
    stage_idx=5
)

# Inspect outputs
print(f"Available outputs: {list(ctx.data.keys())}")
print(f"Signals shape: {ctx.data['signals_df'].shape}")
```

## Current Validation Workflow (In Progress)

### Stage-by-Stage Validation Strategy

**Goal**: Validate all 16 stages on single date (2025-12-16) before batch processing 42 trading days

**For Each Stage**:
1. Create dedicated validation script: `scripts/validate_stage_NN_*.py`
2. Run validation in nohup (non-blocking)
3. Review results JSON and logs
4. Fix any issues found
5. Re-validate to confirm fixes
6. Move to next stage

### Commands Being Used

**Stage 0 (LoadBronze)**:
```bash
cd backend
nohup uv run python -m scripts.validate_stage_00_load_bronze \
  --date 2025-12-16 > logs/validate_stage_00.out 2>&1 &

# Check results
cat logs/validate_stage_00_2025-12-16_results.json
```

**Stage 1 (BuildOHLCV 1min)**:
```bash
nohup uv run python -m scripts.validate_stage_01_build_ohlcv_1min \
  --date 2025-12-16 > logs/validate_stage_01.out 2>&1 &
```

**Stage 2 (BuildOHLCV 2min)**:
```bash
nohup uv run python -m scripts.validate_stage_02_build_ohlcv_2min \
  --date 2025-12-16 > logs/validate_stage_02.out 2>&1 &
```

**Stage 3 (InitMarketState)**:
```bash
nohup uv run python -m scripts.validate_stage_03_init_market_state \
  --date 2025-12-16 > logs/validate_stage_03.out 2>&1 &
```

**Stage 4 (GenerateLevels)**:
```bash
nohup uv run python -m scripts.validate_stage_04_generate_levels \
  --date 2025-12-16 > logs/validate_stage_04.out 2>&1 &
```

### Inspection Commands

**View stage outputs**:
```bash
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --inspect N
```

**Clear and re-run after fixes**:
```bash
rm -rf data/checkpoints/es_pipeline/2025-12-16
nohup uv run python -m scripts.validate_stage_0N_* --date 2025-12-16 > logs/validate.out 2>&1 &
```

### Issues Found and Fixed

**Stage 1 (BuildOHLCV 1min)**:
- **Issue 1**: `reset_index()` converted DatetimeIndex → RangeIndex
  - **Fix**: Removed `reset_index()`, kept DatetimeIndex throughout
  - **File**: `src/pipeline/stages/build_spx_ohlcv.py:83`
  
- **Issue 2**: Checkpoint saved with `index=False`, lost DatetimeIndex
  - **Fix**: Changed to `index=True` in checkpoint serialization
  - **File**: `src/pipeline/core/checkpoint.py:114`
  
- **Issue 3**: Incorrectly added `rth_only=True` (breaks PM_HIGH/PM_LOW)
  - **Fix**: Reverted to `rth_only=False` (RTH filter is Stage 16, not Stage 1)
  - **File**: `src/pipeline/pipelines/es_pipeline.py:62`

**Stage 2 (BuildOHLCV 2min)**:
- **Issue**: Warmup concatenation used `ignore_index=True`
  - **Fix**: Changed to `axis=0` concatenation preserving DatetimeIndex
  - **File**: `src/pipeline/stages/build_spx_ohlcv.py:241`

**Stage 4 (GenerateLevels)**:
- **Issue**: Expected 'timestamp' column but received DatetimeIndex
  - **Fix**: Added DatetimeIndex handling with reset_index/rename
  - **Files**: `src/pipeline/stages/generate_levels.py:51-59, 237, 296, 211`

**Stage 5 (DetectInteractionZones)**:
- **Issue**: Expected 'timestamp' column but received DatetimeIndex
  - **Fix**: Added DatetimeIndex handling with reset_index/rename
  - **File**: `src/pipeline/stages/detect_interaction_zones.py:88-99`

**Stage 7 (ComputeMultiWindowKinematics)**:
- **Issue**: Expected 'timestamp' column but received DatetimeIndex
  - **Fix**: Added DatetimeIndex handling with reset_index/rename
  - **File**: `src/pipeline/stages/compute_multiwindow_kinematics.py:59-67`

**Stage 10 (ComputeLevelDistances)**:
- **Issue**: Interaction events missing bar_idx (distance features all NaN)
  - **Fix**: Added bar_idx/spot to interaction events
  - **File**: `src/pipeline/stages/detect_interaction_zones.py:163-175`

**Stage 12 (ComputeForceMass)**:
- **Issue**: Stage expected `acceleration` but pipeline provides `acceleration_1min`
  - **Fix**: Accept `acceleration_1min` as the kinematics source
  - **File**: `src/pipeline/stages/compute_force_mass.py:52-58`

**Stage 13 (ComputeApproachFeatures)**:
- **Issue**: Expected 'timestamp' column but received DatetimeIndex
  - **Fix**: Added DatetimeIndex handling with reset_index/rename
  - **File**: `src/pipeline/stages/compute_approach.py:43-53`

**Stage 14 (LabelOutcomes)**:
- **Issue**: Expected 'timestamp' column but received DatetimeIndex
  - **Fix**: Added DatetimeIndex handling with reset_index/rename
  - **File**: `src/pipeline/stages/label_outcomes.py:60-68`

### Next Steps (For Continuation)

1. **Run Full Batch** (after all stages validated):
   ```bash
   # Generate all dates
   dates=$(python -c "from datetime import datetime, timedelta; start=datetime(2025,11,3); end=datetime(2025,12,19); print(','.join([(start+timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end-start).days+1) if (start+timedelta(days=i)).weekday()<5]))")
   
   # Process all dates
   uv run python -m scripts.run_pipeline_incremental \
     --dates $dates \
     --checkpoint-dir data/checkpoints
   ```

## Validation Workflow (Generic)

## Checkpoint Invalidation

Checkpoints are **not automatically invalidated** when CONFIG changes. To ensure consistency:

```bash
# Clear checkpoints after CONFIG changes
uv run python -m scripts.run_pipeline_incremental \
  --date 2025-12-16 \
  --checkpoint-dir data/checkpoints \
  --clear
```

Or compare `config_hash` in `metadata.json` with current CONFIG hash.

## Performance Notes

- **Checkpoint overhead**: ~2-5s per stage for serialization
- **Resume speed**: ~1-2s to load checkpoint
- **Disk usage**: ~500MB-2GB per date (depends on data volume)
- **Optimizations**:
  - DataFrames saved as ZSTD-compressed Parquet
  - Complex objects use pickle protocol 5 (fastest)
  - Only saves `ctx.data`, not intermediate computation state

## Troubleshooting

### Checkpoint not found
```
ValueError: Checkpoint not found for stage 5
```
**Solution**: Run `--list` to see available checkpoints, or run without `--resume-from-stage`

### Config mismatch
Checkpoint was saved with different CONFIG values.
**Solution**: Clear checkpoints and re-run with current CONFIG

### Serialization failure
```
Failed to serialize 'some_key': ...
```
**Solution**: Check that all `ctx.data` values are serializable (DataFrame, list, dict, or pickle-serializable)

## Best Practices

1. **Clear checkpoints after CONFIG changes** to avoid stale data
2. **Use `--stop-at-stage` for debugging** to catch issues early
3. **Inspect outputs at key stages** (after load_bronze, after physics, after labels)
4. **Resume from last good checkpoint** when fixing stage failures
5. **Delete checkpoints for old dates** to manage disk usage

## Integration with Production

For production Silver feature generation:

```python
# In src.lake.silver_feature_builder.py
builder = SilverFeatureBuilder()
pipeline = get_pipeline("es_pipeline")

# Add checkpoint support
signals_df = pipeline.run(
    date=date,
    checkpoint_dir=str(builder.features_root / "checkpoints")  # Optional
)
```

Checkpointing is **optional** - production can run without it for performance.

---

## Continuation Instructions for AI Agent

**Current State**: Stages 0-4 validated and passing on 2025-12-16

**To Continue Validation**:

1. **Create validation script for Stage 5** (DetectInteractionZones):
   - Template: Use `scripts/validate_stage_04_generate_levels.py` as reference
   - Expected outputs: `touches_df` (interaction zone entries)
   - Key checks: Zone detection, deterministic event IDs, correct level associations
   
2. **Run in nohup**:
   ```bash
   cd backend
   nohup uv run python -m scripts.validate_stage_05_* --date 2025-12-16 > logs/validate_stage_05.out 2>&1 &
   ```

3. **Review results**:
   ```bash
   cat logs/validate_stage_05_2025-12-16_results.json
   tail -100 logs/validate_stage_05.out
   ```

4. **Fix any issues** following the pattern:
   - DatetimeIndex issues: Check for `.reset_index()` or column access on index
   - Schema issues: Verify stage doesn't assume column structure
   - Re-run after fixes: `rm -rf data/checkpoints/es_pipeline/2025-12-16/stage_0N`

5. **Repeat for Stages 6-15**

**After All Stages Pass**:

Generate Silver features for full date range (42 trading days):
```bash
cd backend

# Create date list (Nov 3 - Dec 19, 2025, weekdays only)
dates="2025-11-03,2025-11-04,2025-11-05,2025-11-06,2025-11-07,2025-11-10,2025-11-11,2025-11-12,2025-11-13,2025-11-14,2025-11-17,2025-11-18,2025-11-19,2025-11-20,2025-11-21,2025-11-24,2025-11-25,2025-11-26,2025-11-27,2025-11-28,2025-12-01,2025-12-02,2025-12-03,2025-12-04,2025-12-05,2025-12-08,2025-12-09,2025-12-10,2025-12-11,2025-12-12,2025-12-15,2025-12-16,2025-12-17,2025-12-18,2025-12-19"

# Run full pipeline for all dates (this will take hours)
nohup uv run python -c "
from src.pipeline.pipelines.es_pipeline import build_es_pipeline
import pandas as pd
from pathlib import Path

pipeline = build_es_pipeline()
dates = '$dates'.split(',')

for date in dates:
    print(f'Processing {date}...')
    signals_df = pipeline.run(date=date)
    
    # Save to Silver
    output_dir = Path('data/lake/silver/features/es_pipeline')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'signals_{date}.parquet'
    signals_df.to_parquet(output_file, compression='zstd', index=True)
    print(f'  Saved {len(signals_df):,} signals to {output_file}')
" > logs/batch_pipeline_run.log 2>&1 &

# Monitor progress
tail -f logs/batch_pipeline_run.log
```

**Validation Scripts Created**:
- `scripts/validate_stage_00_load_bronze.py` ✅
- `scripts/validate_stage_01_build_ohlcv_1min.py` ✅
- `scripts/validate_stage_02_build_ohlcv_2min.py` ✅
- `scripts/validate_stage_03_init_market_state.py` ✅
- `scripts/validate_stage_04_generate_levels.py` ✅
- `scripts/validate_stage_05_detect_interaction_zones.py` ✅
- `scripts/validate_stage_06_compute_physics.py` ✅
- `scripts/validate_stage_07_compute_multiwindow_kinematics.py` ✅
- `scripts/validate_stage_08_compute_multiwindow_ofi.py` ✅
- `scripts/validate_stage_09_compute_barrier_evolution.py` ✅
- `scripts/validate_stage_10_compute_level_distances.py` ✅
- `scripts/validate_stage_11_compute_gex_features.py` ✅
- `scripts/validate_stage_12_compute_force_mass.py` ✅
- `scripts/validate_stage_13_compute_approach.py` ✅
- `scripts/validate_stage_14_label_outcomes.py` ✅
- `scripts/validate_stage_15_filter_rth.py` ✅

**Key Files Modified**:
- `src/pipeline/core/checkpoint.py` - Added checkpointing manager
- `src/pipeline/core/pipeline.py` - Added resume/stop_at_stage support
- `src/pipeline/stages/build_spx_ohlcv.py` - Fixed DatetimeIndex preservation
- `src/pipeline/stages/generate_levels.py` - Added DatetimeIndex handling
- `src/pipeline/stages/detect_interaction_zones.py` - Added DatetimeIndex handling + bar_idx/spot
- `src/pipeline/stages/compute_multiwindow_kinematics.py` - Added DatetimeIndex handling
- `src/pipeline/stages/compute_approach.py` - Added DatetimeIndex handling
- `src/pipeline/stages/label_outcomes.py` - Added DatetimeIndex handling
- `src/pipeline/pipelines/es_pipeline.py` - Corrected rth_only flags

## Validation Results (2025-12-16)

### Stage 0: LoadBronze ✅
**Execution**: 3.59s  
**Data Loaded**:
- ES Futures Trades: 612,748 records
- MBP-10 Snapshots: 95,409 records (downsampled)
- ES Options: 973 records (0DTE, front-month ESH6)

**Key Findings**:
- Single ES contract (100% front-month purity)
- Full 24-hour coverage (00:00-23:59 UTC)
- Price range: 6817.50 - 6892.00 ES points
- All schemas valid, no data quality issues

### Stage 1: BuildOHLCV (1min) ✅
**Execution**: 0.12s  
**Bars Generated**: 1,380 (23-hour trading session)

**Coverage**:
- Premarket: 330 bars (04:00-09:29 ET) - needed for PM_HIGH/PM_LOW
- RTH: 240 bars (09:30-13:29 ET) - where signals are generated
- Extended: 810 bars (13:30-03:59 ET)

**ATR Computation**:
- 1,380 values computed (full session)
- Mean: 2.35 points, Range: 0.70-7.70 points
- All positive, no NaN values

**Issues Fixed**:
- DatetimeIndex preservation
- Checkpoint serialization with index=True
- Removed incorrect rth_only=True filter

### Stage 2: BuildOHLCV (2min with warmup) ✅
**Execution**: 3.46s  
**Bars Generated**: 2,730 (multi-day with warmup)

**Warmup Data**:
- 2025-12-11: 690 bars
- 2025-12-12: 660 bars
- 2025-12-15: 690 bars
- Current (2025-12-16): 690 bars
- **Total**: 2,730 bars (sufficient for SMA_400)

**Key Findings**:
- 2min bars = 1min bars / 2 (correct aggregation)
- Multi-day timestamps properly ordered
- Volume: 5.94M contracts across 4 days

**Issues Fixed**:
- Warmup concatenation preserving DatetimeIndex

### Stage 3: InitMarketState ✅
**Execution**: 1.29s  
**Outputs**: MarketState, enriched options, spot_price

**MarketState Population**:
- ES trades buffer: 612,748
- MBP-10 buffer: 95,409
- Option flows: 102 unique strikes
- Buffer window: 480s (2x confirmation window)

**Greeks (Black-76)**:
- Delta range: -0.998 to 0.9996 (valid)
- Call delta mean: 0.0424 (positive ✓)
- Put delta mean: -0.0589 (negative ✓)
- Gamma: 0.000000 to 0.013509 (all positive)

**Spot Price**: 6845.25 ES points

### Stage 4: GenerateLevels ✅
**Execution**: 0.03s  
**Levels Generated**: 6 structural levels

**Level Values**:
- PM_HIGH: 6892.00 (from 04:00-09:30 ET premarket)
- PM_LOW: 6846.75 (spread: 45.25 points)
- OR_HIGH: 6874.50 (from 09:30-09:45 ET opening range)
- OR_LOW: 6850.75 (spread: 23.75 points)
- SMA_200: 6845.70 (400 min lookback with warmup)
- SMA_400: 6855.21 (800 min lookback with warmup)

**Dynamic Level Series**: 11 time series (1,380 values each)
- PM_HIGH/LOW, OR_HIGH/LOW, SESSION_HIGH/LOW, VWAP, SMA_200/400, CALL_WALL, PUT_WALL

**Issues Fixed**:
- DatetimeIndex handling in generate_level_universe
- DatetimeIndex handling in compute_dynamic_level_series

### Stage 5: DetectInteractionZones ✅
**Execution**: 0.01s  
**Events Detected**: 166 interaction entries

**Distributions**:
- Levels: SMA_200 (41), OR_LOW (39), PM_LOW (36), SMA_400 (30), OR_HIGH (19), PM_HIGH (1)
- Direction: UP (97), DOWN (69)

**Key Findings**:
- Deterministic event IDs (unique, reproducible)
- Zone width respects MONITOR_BAND floor
- Level kind/name mapping consistent with LevelInfo

### Stage 6: ComputePhysics ✅
**Execution**: 2.13s  
**Signals Computed**: 166 (matches interaction events)

**Distributions**:
- Barrier states: NEUTRAL (93), WEAK (46), WALL (27)
- Fuel effects: NEUTRAL (166)

**Key Findings**:
- Physics columns populated for all events
- No NaNs in barrier/tape/fuel metrics

### Stage 7: ComputeMultiWindowKinematics ✅
**Execution**: 0.01s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- Multi-window velocity/acceleration/jerk features populated
- Momentum trend columns added for 3/5/10/20 minute windows

### Stage 8: ComputeMultiWindowOFI ✅
**Execution**: 0.11s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- OFI windows populated for 30s/60s/120s/300s
- OFI acceleration computed (mean ~0.04)

### Stage 9: ComputeBarrierEvolution ✅
**Execution**: 0.06s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- Barrier delta/pct-change features populated for 1/3/5 minute windows
- Current barrier depth computed (mean ~265)

### Stage 10: ComputeLevelDistances ✅
**Execution**: 0.02s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- Distance/ATR-normalized features populated for all structural levels
- Partial NaNs on PM/OR distances where dynamic levels are undefined

### Stage 11: ComputeGEXFeatures ✅
**Execution**: 0.01s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- Strike-banded GEX features populated for ±1/±2/±3 strikes
- GEX asymmetry/ratio features available for retrieval

### Stage 12: ComputeForceMass ✅
**Execution**: 0.00s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- F=ma consistency features populated (predicted_accel, residual, ratio)

### Stage 13: ComputeApproachFeatures ✅
**Execution**: 0.05s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- Approach context + normalization features populated
- Attempt clustering and deterioration trends available

### Stage 14: LabelOutcomes ✅
**Execution**: 0.01s  
**Signals Computed**: 166 (matches interaction events)

**Key Findings**:
- Outcome distribution: CHOP 81, UNDEFINED 66, BOUNCE 16, BREAK 3
- Tradeable signals: 19 (11.4%)

### Stage 15: FilterRTH ✅
**Execution**: 0.01s  
**Signals Computed**: 53 (RTH-filtered)

**Key Findings**:
- All signals within 09:30-13:30 ET window
- Silver schema validation passed (182 columns)

## Overview

Incremental pipeline execution with stage-by-stage checkpointing for:
- **Debugging**: Inspect intermediate outputs
- **Recovery**: Resume from failures without re-running all stages
- **Validation**: Verify each stage's correctness before proceeding
- **Development**: Iterate on later stages without re-running earlier ones
