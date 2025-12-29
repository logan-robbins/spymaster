# Pipeline Module

**Purpose**: Modular stage-based feature engineering pipeline
**Status**: Production
**Primary Consumer**: SilverFeatureBuilder
**Architecture**: See [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)

---

## Overview

Provides a modular, versioned pipeline architecture for transforming Bronze data into Silver features. Different pipeline versions use different stage compositions, allowing feature sets to be tailored to specific experiments.

**Key Principle**: Pipelines are composed of independent stages. Each stage has explicit inputs/outputs and uses vectorized NumPy/pandas operations optimized for Apple M4 Silicon.

---

## Architecture

```
src/pipeline/
├── core/
│   ├── stage.py         # BaseStage, StageContext abstractions
│   └── pipeline.py      # Pipeline orchestrator
├── stages/
│   ├── load_bronze.py           # Stage 1: DuckDB data loading
│   ├── build_ohlcv.py           # Stage 2-3: OHLCV bar construction
│   ├── init_market_state.py     # Stage 4: MarketState + Greeks
│   ├── generate_levels.py       # Stage 5: Level universe
│   ├── detect_touches.py        # Stage 6: Touch detection
│   ├── compute_physics.py       # Stage 7: Barrier/Tape/Fuel
│   ├── compute_context.py       # Stage 8: Context features
│   ├── compute_sma.py           # Stage 9: SMA features (v2.0+)
│   ├── compute_confluence.py    # Stage 10: Confluence (v2.0+)
│   ├── compute_approach.py      # Stage 11: Approach features (v2.0+)
│   ├── label_outcomes.py        # Stage 12: Outcome labeling
│   └── filter_rth.py            # Stage 13: RTH filtering
├── pipelines/
│   ├── v1_0_mechanics_only.py   # 10 stages: pure physics
│   ├── v2_0_full_ensemble.py    # 13 stages: physics + TA
│   └── registry.py              # get_pipeline_for_version()
└── utils/
    ├── duckdb_reader.py         # DuckDB wrapper with downsampling
    └── vectorized_ops.py        # Vectorized operation exports
```

---

## Usage

**Via SilverFeatureBuilder** (recommended):
```python
from src.lake.silver_feature_builder import SilverFeatureBuilder

builder = SilverFeatureBuilder()
builder.build_feature_set(
    manifest=manifest,  # manifest.version determines pipeline
    dates=['2025-12-16', '2025-12-17']
)
# Internally calls get_pipeline_for_version(manifest.version)
```

**Direct Usage**:
```python
from src.pipeline import get_pipeline_for_version

# Get pipeline for version
pipeline = get_pipeline_for_version("v1.0_mechanics_only")
signals_df = pipeline.run("2025-12-16")

# Or use specific builders
from src.pipeline.pipelines import build_v1_0_pipeline, build_v2_0_pipeline

pipeline = build_v2_0_pipeline()
signals_df = pipeline.run("2025-12-16")
```

---

## Pipeline Versions

### v1.0 - Mechanics Only (10 stages)
Pure physics features without TA indicators:
1. LoadBronze → 2. BuildOHLCV (1min) → 3. BuildOHLCV (2min)
4. InitMarketState → 5. GenerateLevels → 6. DetectTouches
7. ComputePhysics → 8. ComputeContext → 9. LabelOutcomes → 10. FilterRTH

### v2.0 - Full Ensemble (13 stages)
All features including SMA, confluence, and approach context:
1-8. Same as v1.0
9. ComputeSMAFeatures → 10. ComputeConfluence → 11. ComputeApproachFeatures
12. LabelOutcomes → 13. FilterRTH

---

## Stage Dependencies

Each stage declares required inputs via `required_inputs` property:

| Stage | Required Inputs |
|-------|-----------------|
| LoadBronze | (none) |
| BuildOHLCV | trades |
| InitMarketState | trades, mbp10_snapshots, option_trades_df |
| GenerateLevels | ohlcv_1min, market_state |
| DetectTouches | ohlcv_1min, static_level_info, dynamic_levels |
| ComputePhysics | touches_df, market_state, trades, mbp10_snapshots |
| ComputeContext | signals_df, atr, ohlcv_1min |
| ComputeSMA | signals_df, ohlcv_1min |
| ComputeConfluence | signals_df, dynamic_levels, option_trades_df, ohlcv_1min |
| ComputeApproach | signals_df, ohlcv_1min |
| LabelOutcomes | signals_df, ohlcv_1min |
| FilterRTH | signals_df |

---

## Level Universe (SPY-Specific)

Generated structural levels:
- **PM_HIGH/PM_LOW**: Pre-market high/low (04:00-09:30 ET)
- **OR_HIGH/OR_LOW**: Opening range (09:30-09:45 ET)
- **SESSION_HIGH/SESSION_LOW**: Running session extremes
- **SMA_200/SMA_400**: Moving averages on 2-min bars (requires warmup)
- **VWAP**: Session volume-weighted average price
- **CALL_WALL/PUT_WALL**: Max gamma concentration strikes

**Note**: ROUND and STRIKE levels are disabled for SPY due to $1 strike spacing.

---

## Feature Categories

**Barrier Physics** (ES MBP-10 depth):
- `barrier_state`, `barrier_delta_liq`, `wall_ratio`

**Tape Physics** (ES trade flow):
- `tape_imbalance`, `tape_velocity`, `sweep_detected`

**Fuel Physics** (SPY option gamma):
- `gamma_exposure`, `fuel_effect`, `dealer_pressure`

**Approach Context** (v2.0+):
- `approach_velocity`, `attempt_index`, `prior_touches`

**Confluence Features** (v2.0+):
- `confluence_level`, `gex_alignment`, `rel_vol_ratio`

**Labels** (competing risks):
- `outcome`, `strength_signed`, `t1_60`, `t2_60`, `tradeable_1`, `tradeable_2`

---

## Performance

**Apple M4 Silicon Optimized**:
- All operations use numpy broadcasting
- Batch processing of all touches simultaneously
- Memory-efficient chunked processing
- Optional Numba JIT compilation

**Typical Performance** (M4 Mac, 128GB RAM):
- Single date: ~2-5 seconds
- 10 dates: ~30-60 seconds
- ~500-1000 signals/sec throughput

---

## Configuration

Pipeline behavior controlled by `backend/src/common/config.py`, which loads best-config JSON by default (`data/ml/experiments/zone_opt_v1_best_config.json` or `CONFIG_OVERRIDE_PATH`):
- Physics windows: `W_b`, `W_t`, `W_g`
- Touch detection: `MONITOR_BAND`, `TOUCH_BAND`
- Confirmation windows: `CONFIRMATION_WINDOWS_MULTI`
- Warmup: `SMA_WARMUP_DAYS`, `VOLUME_LOOKBACK_DAYS`

---

## Creating New Pipeline Versions

1. Create new file in `pipelines/` (e.g., `v2_1_custom.py`)
2. Define stage sequence using existing or new stages
3. Register in `pipelines/registry.py`

```python
# pipelines/v2_1_custom.py
def build_v2_1_pipeline() -> Pipeline:
    return Pipeline(
        stages=[
            LoadBronzeStage(),
            BuildOHLCVStage(freq='1min'),
            # ... custom stage composition
        ],
        name="custom",
        version="v2.1"
    )

# pipelines/registry.py
_PIPELINES = {
    'v1.0': build_v1_0_pipeline,
    'v2.0': build_v2_0_pipeline,
    'v2.1': build_v2_1_pipeline,  # Add new version
}
```

---

## References

- **Architecture**: [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)
- **Feature Manifests**: [../common/schemas/feature_manifest.py](../common/schemas/feature_manifest.py)
- **Physics Engines**: [../core/](../core/)
- **Silver Builder**: [../lake/silver_feature_builder.py](../lake/silver_feature_builder.py)
