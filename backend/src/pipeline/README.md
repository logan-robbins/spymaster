# Pipeline Module

**Purpose**: Offline feature engineering and signal generation for SPY 0DTE ML training
**Status**: Production
**Primary Output**: Gold-layer parquet files with labeled signals

---

## Overview

The pipeline module orchestrates the complete Bronze → Gold data transformation:
- **Data Loading**: Read ES futures trades, MBP-10 quotes, and option trades from Bronze layer
- **Feature Engineering**: Compute physics-based features (barrier, tape, fuel, approach)
- **Level Detection**: Generate structural level universe and detect touches
- **Labeling**: Create supervised learning targets (BREAK/BOUNCE/CHOP)
- **Export**: Write ML-ready parquet files to Gold layer

---

## Components

### VectorizedPipeline (Primary)

**File**: `vectorized_pipeline.py`
**Purpose**: High-performance batch processing using numpy/pandas vectorized operations

**Key Stages**:
1. **Data Loading**: Parallel file reads from Bronze parquet
2. **OHLCV Building**: 1-min and 2-min bars from ES trades (converted to SPY scale)
3. **SMA Warmup**: Load 3 prior days for SMA-200/400 computation
4. **Volume Warmup**: Load 7 prior days for relative volume computation
5. **Level Universe**: Generate structural levels (PM/OR/SMA/VWAP/Walls)
6. **Touch Detection**: Numpy broadcasting for monitor band filtering
7. **Physics Computation**: Batch metrics from barrier/tape/fuel engines
8. **Feature Transforms**: Normalization, sparse encoding, confluence features
9. **Labeling**: Forward-looking outcome determination (competing risks)

**Usage**:
```python
from src.pipeline.vectorized_pipeline import VectorizedPipeline

pipeline = VectorizedPipeline()
signals_df = pipeline.run(date="2025-12-16")

# Output: DataFrame with ~800-3000 signals per day
print(f"Generated {len(signals_df)} signals")
print(signals_df.columns.tolist())
```

---

### BatchProcess

**File**: `batch_process.py`
**Purpose**: Multi-date batch orchestration using VectorizedPipeline

**Usage**:
```bash
cd backend

# Process all available dates (with sufficient warmup)
uv run python -m src.pipeline.batch_process

# Process specific date range
uv run python -m src.pipeline.batch_process --start-date 2025-12-10 --end-date 2025-12-19

# Process specific dates only
uv run python -m src.pipeline.batch_process --dates 2025-12-18,2025-12-19

# Dry run (show what would be processed)
uv run python -m src.pipeline.batch_process --dry-run
```

---

## Level Universe

### Active Level Types (SPY-specific)

| Code | Name | Description |
|------|------|-------------|
| 0 | PM_HIGH | Pre-market high (04:00-09:30 ET) |
| 1 | PM_LOW | Pre-market low (04:00-09:30 ET) |
| 2 | OR_HIGH | Opening range high (09:30-09:45 ET) |
| 3 | OR_LOW | Opening range low (09:30-09:45 ET) |
| 4 | SESSION_HIGH | Running session high |
| 5 | SESSION_LOW | Running session low |
| 6 | SMA_200 | 200-period SMA on 2-min bars |
| 7 | VWAP | Session volume-weighted average price |
| 10 | CALL_WALL | Max call gamma concentration strike |
| 11 | PUT_WALL | Max put gamma concentration strike |
| 12 | SMA_400 | 400-period SMA on 2-min bars |

**Disabled for SPY**:
- Code 8 (ROUND): Duplicative with $1 strike spacing
- Code 9 (STRIKE): Duplicative with $1 strike spacing

---

## Confluence Level Feature

### Overview

The `confluence_level` feature provides a hierarchical quality score (1-10) for each signal based on 5 dimensions:

| Dimension | States | Description |
|-----------|--------|-------------|
| Breakout State | INSIDE, PARTIAL, ABOVE_ALL, BELOW_ALL | Relationship to PM/OR ranges |
| SMA Proximity | CLOSE, FAR | Distance to SMA-200 AND SMA-400 |
| Time Period | FIRST_HOUR, REST_OF_DAY | 09:30-10:30 ET vs after |
| GEX Alignment | ALIGNED, OPPOSED, NEUTRAL | Gamma supports/resists move |
| Relative Volume | HIGH, NORMAL, LOW | Participation vs 7-day average |

### Confluence Levels

| Level | Name | Conditions |
|-------|------|------------|
| 1 | ULTRA_PREMIUM | Full breakout + SMA close + first hour + GEX aligned + high volume |
| 2 | PREMIUM | Full breakout + SMA close + first hour + GEX aligned |
| 3 | STRONG | Full breakout + SMA close + first hour |
| 4 | MOMENTUM | Full breakout + SMA far + first hour + GEX aligned + high volume |
| 5 | EXTENDED | Full breakout + SMA far + first hour |
| 6 | LATE_REVERSION | Full breakout + SMA close + rest of day + high volume |
| 7 | FADING | Full breakout + rest of day |
| 8 | DEVELOPING | Partial breakout + SMA close + GEX aligned + high volume |
| 9 | WEAK | Partial breakout |
| 10 | CONSOLIDATION | Inside both PM and OR ranges |
| 0 | UNDEFINED | Missing PM/OR/SMA/volume data |

### Related Features

| Column | Type | Description |
|--------|------|-------------|
| `confluence_level` | int8 | Hierarchical quality score (0-10) |
| `breakout_state` | int8 | 0=INSIDE, 1=PARTIAL, 2=ABOVE_ALL, 3=BELOW_ALL |
| `gex_alignment` | int8 | -1=OPPOSED, 0=NEUTRAL, 1=ALIGNED |
| `rel_vol_ratio` | float64 | Current hour cumvol / 7-day avg at same hour |

---

## Configuration

### Key Constants (from `src/common/config.py`)

```python
# Monitoring bands
MONITOR_BAND = 0.25       # $0.25 - compute signals within this distance
TOUCH_BAND = 0.10         # $0.10 - tight band for "touching level"

# Warmup periods
SMA_WARMUP_DAYS = 3       # Prior days for SMA-200/400
VOLUME_LOOKBACK_DAYS = 7  # Prior days for relative volume

# Confluence thresholds
SMA_PROXIMITY_THRESHOLD = 0.005   # 0.5% of spot for "close to SMA"
WALL_PROXIMITY_DOLLARS = 1.0      # $1 for GEX wall proximity
REL_VOL_HIGH_THRESHOLD = 1.3      # 30% above average = HIGH
REL_VOL_LOW_THRESHOLD = 0.7       # 30% below average = LOW

# Confirmation windows
CONFIRMATION_WINDOWS_MULTI = [120, 240, 480]  # 2min, 4min, 8min
```

---

## Output Schema

The pipeline outputs a parquet file with columns defined in `backend/features.json`.

**Key Column Groups**:
- `identity`: event_id, ts_ns, date, symbol
- `level`: level_price, level_kind, direction, distance
- `confluence`: confluence_level, breakout_state, gex_alignment, rel_vol_ratio
- `barrier_physics`: barrier_state, barrier_delta_liq, wall_ratio
- `tape_physics`: tape_imbalance, tape_velocity, sweep_detected
- `fuel_physics`: gamma_exposure, fuel_effect, dealer_flow_*
- `outcomes`: outcome (BREAK/BOUNCE), time_to_break_*, time_to_bounce_*

---

## Performance

### Typical Runtime (M4 Silicon, 128GB RAM)

| Date Characteristics | Runtime | Signals |
|---------------------|---------|---------|
| Normal day (~500K trades) | 15-25s | 800-1500 |
| High volume day (~1M trades) | 30-45s | 2000-3500 |
| Low volume day (~200K trades) | 8-12s | 400-800 |

### Memory Usage

- Peak: ~4-8GB during physics computation
- Steady state: ~1-2GB for data structures
- Output parquet: ~5-15MB per day

---

## Testing

```bash
cd backend
uv run pytest tests/test_ml_module.py tests/test_research_lab.py -v
```

---

## References

- **Feature Schema**: `backend/features.json`
- **Configuration**: `src/common/config.py`
- **Bronze Reader**: `src/lake/bronze_reader.py`
- **Physics Engines**: `src/core/barrier_engine.py`, `src/core/tape_engine.py`, `src/core/fuel_engine.py`

---

**Version**: 1.0
**Last Updated**: 2025-12-24
**Status**: Production
