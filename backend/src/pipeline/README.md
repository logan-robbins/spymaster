# Pipeline Module

**Purpose**: Stage-based feature engineering for ES futures + ES 0DTE options  
**Status**: Production  
**Primary Consumer**: `SilverFeatureBuilder`  
**Architecture**: See [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)

---

## Overview

This module defines the ES pipeline used to transform Bronze data into Silver features.
The pipeline is composed of explicit stages with declared inputs/outputs and validated
against the Silver schema after RTH filtering.

---

## Architecture

```
src/pipeline/
├── core/
│   ├── stage.py                # BaseStage, StageContext
│   ├── pipeline.py             # Pipeline orchestrator
│   └── checkpoint.py           # Stage checkpoints (resume/inspect)
├── stages/
│   ├── load_bronze.py                      # Stage 0: DuckDB data loading
│   ├── build_spx_ohlcv.py                  # Stage 1-2: OHLCV (1min/2min)
│   ├── init_market_state.py                # Stage 3: MarketState + Greeks
│   ├── generate_levels.py                  # Stage 4: Level universe + dynamic series
│   ├── detect_interaction_zones.py         # Stage 5: Zone entry events
│   ├── compute_physics.py                  # Stage 6: Barrier/Tape/Fuel physics
│   ├── compute_multiwindow_kinematics.py   # Stage 7: Multi-window kinematics
│   ├── compute_multiwindow_ofi.py          # Stage 8: Multi-window OFI
│   ├── compute_barrier_evolution.py        # Stage 9: Barrier evolution
│   ├── compute_level_distances.py          # Stage 10: Level distances + stacking
│   ├── compute_gex_features.py             # Stage 11: Strike-banded GEX
│   ├── compute_force_mass.py               # Stage 12: F=ma validation features
│   ├── compute_approach.py                 # Stage 13: Approach context + normalization
│   ├── label_outcomes.py                   # Stage 14: First-crossing labels (v3.0.0)
│   ├── filter_rth.py                       # Stage 15: RTH filtering + schema validation
│   ├── materialize_state_table.py          # Stage 16: 30s state table (v3.0.0)
│   └── construct_episodes.py               # Stage 17: Episode vectors (v3.0.0)
├── pipelines/
│   ├── es_pipeline.py           # build_es_pipeline()
│   └── registry.py              # get_pipeline()
└── utils/
    └── duckdb_reader.py         # DuckDB wrapper with downsampling
```

---

## Usage

**Via SilverFeatureBuilder** (recommended):
```python
from src.io.silver import SilverFeatureBuilder

builder = SilverFeatureBuilder()
builder.build_feature_set(
    manifest=manifest,  # manifest.version maps to es_pipeline
    dates=['2025-12-16']
)
```

**Direct Usage**:
```python
from src.pipeline.pipelines.es_pipeline import build_es_pipeline
from src.pipeline.pipelines.registry import get_pipeline

pipeline = build_es_pipeline()
signals_df = pipeline.run("2025-12-16")

# Or via registry
pipeline = get_pipeline("es_pipeline")
signals_df = pipeline.run("2025-12-16")
```

---

## ES Pipeline v3.0.0 (18 stages, 0-indexed: 0-17)

**Note**: All pipeline stage references are `stage_idx` (0-based). Checkpoints are stored under `data/checkpoints/.../stage_{stage_idx:02d}/`.

0. LoadBronze  
1. BuildOHLCV (1min)  
2. BuildOHLCV (2min, warmup)  
3. InitMarketState  
4. GenerateLevels  
5. DetectInteractionZones  
6. ComputePhysics  
7. ComputeMultiWindowKinematics  
8. ComputeMultiWindowOFI  
9. ComputeBarrierEvolution  
10. ComputeLevelDistances  
11. ComputeGEXFeatures  
12. ComputeForceMass  
13. ComputeApproachFeatures  
14. LabelOutcomes (v3.0.0: first-crossing, BREAK/REJECT/CHOP)  
15. FilterRTH  
16. MaterializeStateTable (v3.0.0: NEW - 30s state cadence)  
17. ConstructEpisodes (v3.0.0: NEW - 144-dim episode vectors)

---

## Stage Dependencies

Each stage declares required inputs via `required_inputs`:

| Stage | Required Inputs |
|-------|-----------------|
| LoadBronze | (none) |
| BuildOHLCV | trades |
| InitMarketState | trades, mbp10_snapshots, option_trades_df |
| GenerateLevels | ohlcv_1min, market_state |
| DetectInteractionZones | ohlcv_1min, level_info, atr |
| ComputePhysics | touches_df, market_state, trades, mbp10_snapshots |
| ComputeMultiWindowKinematics | signals_df, ohlcv_1min |
| ComputeMultiWindowOFI | signals_df, mbp10_snapshots |
| ComputeBarrierEvolution | signals_df, mbp10_snapshots |
| ComputeLevelDistances | signals_df, dynamic_levels, atr |
| ComputeGEXFeatures | signals_df, option_trades_df |
| ComputeForceMass | signals_df |
| ComputeApproachFeatures | signals_df, ohlcv_1min, atr |
| LabelOutcomes | signals_df, ohlcv_1min |
| FilterRTH | signals_df |

---

## Level Universe (ES)

Structural levels used for event generation:
- **PM_HIGH/PM_LOW**: Pre-market high/low (04:00-09:30 ET)
- **OR_HIGH/OR_LOW**: Opening range high/low (09:30-09:45 ET)
- **SMA_200/SMA_400**: Moving averages on 2-min bars (warmup required)

Dynamic series (context-only) include session highs/lows, VWAP, and call/put walls.

---

## Feature Categories

- **Barrier/Tape/Fuel Physics**: `barrier_state`, `tape_imbalance`, `fuel_effect`, `gamma_exposure`
- **Kinematics**: `velocity_*`, `acceleration_*`, `jerk_*`, `momentum_trend_*`
- **Order Flow**: `ofi_*`, `ofi_acceleration`
- **Barrier Evolution**: `barrier_delta_*`, `barrier_pct_change_*`
- **Level Distances**: `dist_to_*`, `level_stacking_*`
- **GEX**: `gex_*`, `gex_asymmetry`, `gex_ratio`, `net_gex_2strike`
- **F=ma**: `predicted_accel`, `accel_residual`, `force_mass_ratio`
- **Approach/Normalization**: `approach_*`, `distance_signed_*`, `*_pct`
- **Labels**: `outcome*`, `strength_*`, `tradeable_*`

---

## Output

Canonical pipeline runs write to:
`silver/features/es_pipeline/version={canonical_version}/date=YYYY-MM-DD/*.parquet`

Silver feature experiments (via `SilverFeatureBuilder`) write to:
`silver/features/{manifest.version}/date=YYYY-MM-DD/*.parquet`

Schema reference:
- [../../SILVER_SCHEMA.md](../../SILVER_SCHEMA.md)
- [../common/schemas/silver_features.py](../common/schemas/silver_features.py)

---

## Configuration

Pipeline behavior is controlled by `backend/src/common/config.py`:
- Physics windows: `W_b`, `W_t`, `W_g`
- Zone/monitor bands: `MONITOR_BAND`, `TOUCH_BAND`
- Confirmation windows: `CONFIRMATION_WINDOWS_MULTI`
- Warmup: `SMA_WARMUP_DAYS`

---

## References

- **Architecture**: [../../DATA_ARCHITECTURE.md](../../DATA_ARCHITECTURE.md)
- **Pipelines**: [pipelines/es_pipeline.py](pipelines/es_pipeline.py)
- **Registry**: [pipelines/registry.py](pipelines/registry.py)
- **Silver Builder**: [../lake/silver_feature_builder.py](../lake/silver_feature_builder.py)
