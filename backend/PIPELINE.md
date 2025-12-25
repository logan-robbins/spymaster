# Spymaster Feature Engineering Pipeline

## Overview

This document defines the complete feature engineering pipeline for generating ML training data. The pipeline is designed to:

1. **Run once per date** - Avoid redundant regeneration
2. **Compute all features** - Full physics + context features
3. **Generate all level types** - Not just STRIKE, but PM_HIGH, SMA200, etc.
4. **Output to Gold tier** - ML-ready Parquet files

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA SOURCES                               │
├─────────────────────────────────────────────────────────────────────────┤
│  DBN Files                      │  Bronze Parquet                        │
│  ├── ES trades                  │  └── SPY 0DTE options                  │
│  └── ES MBP-10                  │      (with tick rule aggressor)        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: DATA LOADING                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ load_es_trades  │  │ load_es_mbp10   │  │ load_options    │          │
│  │ → FuturesTrade[]│  │ → MBP10[]       │  │ → DataFrame     │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: OHLCV BUILD                             │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ build_ohlcv_from_trades()                                    │        │
│  │ → 1-minute OHLCV bars (SPY prices, converted from ES)        │        │
│  │ → 2-minute OHLCV bars (for SMA-200, SMA-400)                 │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: LEVEL UNIVERSE                             │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ Generate ALL level types:                                    │        │
│  │                                                              │        │
│  │ STRUCTURAL LEVELS:                                           │        │
│  │   PM_HIGH    - Pre-market high (04:00-09:30 ET)             │        │
│  │   PM_LOW     - Pre-market low                                │        │
│  │   OR_HIGH    - Opening range high (09:30-09:45 ET)          │        │
│  │   OR_LOW     - Opening range low                             │        │
│  │   SESSION_HIGH - Current session high                        │        │
│  │   SESSION_LOW  - Current session low                         │        │
│  │                                                              │        │
│  │ TECHNICAL LEVELS:                                            │        │
│  │   SMA_200    - 200-period SMA (2-min bars)                  │        │
│  │   SMA_400    - 400-period SMA (2-min bars)                  │        │
│  │   VWAP       - Volume-weighted average price                 │        │
│  │                                                              │        │
│  │ OPTIONS LEVELS:                                              │        │
│  │   STRIKE     - Option strikes with volume                    │        │
│  │   CALL_WALL  - Strike with max call gamma                    │        │
│  │   PUT_WALL   - Strike with max put gamma                     │        │
│  │   GAMMA_FLIP - Gamma flip level (HVL)                        │        │
│  │                                                              │        │
│  │ PRICE LEVELS:                                                │        │
│  │   ROUND      - Round numbers ($1 increments)                 │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 4: MARKET STATE                               │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ Initialize MarketState with:                                 │        │
│  │   - ES trades buffer                                         │        │
│  │   - ES MBP-10 buffer                                         │        │
│  │   - Option flow aggregates (with Black-Scholes greeks)       │        │
│  │   - Price converter (ES ↔ SPY)                               │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: TOUCH DETECTION                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ For each minute bar, check all levels:                       │        │
│  │   - Did price approach within $0.30?                         │        │
│  │   - What direction (UP/DOWN)?                                │        │
│  │   - Is it first 15 minutes?                                  │        │
│  │                                                              │        │
│  │ Output: List[(timestamp, level, direction, is_first_15m)]    │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 6: PHYSICS COMPUTATION                        │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ For each touch event, compute ALL features:                  │        │
│  │                                                              │        │
│  │ BARRIER ENGINE:                                              │        │
│  │   barrier_state, barrier_delta_liq, barrier_replenishment,   │        │
│  │   barrier_added, barrier_canceled, barrier_filled,           │        │
│  │   wall_ratio, depth_in_zone, churn                           │        │
│  │                                                              │        │
│  │ TAPE ENGINE:                                                 │        │
│  │   tape_imbalance, tape_buy_vol, tape_sell_vol,              │        │
│  │   tape_velocity, tape_sweep_detected, tape_sweep_direction,  │        │
│  │   tape_sweep_notional                                        │        │
│  │                                                              │        │
│  │ FUEL ENGINE:                                                 │        │
│  │   gamma_exposure, fuel_effect, fuel_net_dealer_gamma,        │        │
│  │   fuel_call_wall, fuel_put_wall, fuel_hvl                    │        │
│  │                                                              │        │
│  │ SCORE ENGINE:                                                │        │
│  │   break_score_raw, break_score_smooth,                       │        │
│  │   signal, confidence,                                        │        │
│  │   liquidity_score, hedge_score, tape_score                   │        │
│  │                                                              │        │
│  │ RUNWAY ENGINE:                                               │        │
│  │   runway_direction, runway_next_level_price,                 │        │
│  │   runway_distance, runway_quality                            │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 7: LABELING                                   │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ For each touch, look forward 5 minutes:                      │        │
│  │   - BREAK: moved >$0.20 through level in approach direction  │        │
│  │   - BOUNCE: moved >$0.20 away from level                     │        │
│  │   - CHOP: stayed within $0.20                                │        │
│  │                                                              │        │
│  │ Store future_price_5min for P&L calculations                 │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 8: OUTPUT                                     │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ Write to Gold tier:                                          │        │
│  │   data/lake/gold/research/signals_{date}.parquet             │        │
│  │                                                              │        │
│  │ Aggregated multi-day:                                        │        │
│  │   data/lake/gold/research/signals_multi_day.parquet          │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Feature Set

### Identity Features
| Feature | Type | Description |
|---------|------|-------------|
| `event_id` | string | Unique signal identifier |
| `ts_event_ns` | int64 | Event timestamp (Unix nanoseconds UTC) |
| `date` | string | Trading date (YYYY-MM-DD) |
| `symbol` | string | Underlying symbol (SPY) |

### Level Features
| Feature | Type | Description |
|---------|------|-------------|
| `level_price` | float | Price of level being tested |
| `level_kind` | enum | PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_200, SMA_400, STRIKE, VWAP, ROUND, CALL_WALL, PUT_WALL, GAMMA_FLIP, SESSION_HIGH, SESSION_LOW |
| `level_id` | string | Stable level identifier |
| `direction` | enum | UP (resistance test) or DOWN (support test) |
| `distance` | float | Distance from spot to level ($) |

### Context Features
| Feature | Type | Description |
|---------|------|-------------|
| `spot` | float | Current SPY price |
| `is_first_15m` | bool | True if 09:30-09:45 ET |
| `is_opening_range` | bool | True if within opening range period |
| `dist_to_sma_200` | float | Distance to SMA-200 ($) |
| `dist_to_sma_400` | float | Distance to SMA-400 ($) |
| `dist_to_vwap` | float | Distance to VWAP ($) |
| `session_position` | float | Position within session range (0-1) |

### Barrier Physics Features (ES MBP-10)
| Feature | Type | Description |
|---------|------|-------------|
| `barrier_state` | enum | VACUUM, WALL, ABSORPTION, CONSUMED, WEAK, NEUTRAL |
| `barrier_delta_liq` | float | Net liquidity change (contracts) |
| `barrier_replenishment_ratio` | float | added / (canceled + filled) |
| `barrier_added` | int | Size added to defending quote |
| `barrier_canceled` | int | Size canceled from defending quote |
| `barrier_filled` | int | Size filled at defending quote |
| `wall_ratio` | float | Defending size / average depth |
| `depth_in_zone` | int | Total depth in monitoring zone |
| `churn` | float | Gross activity (added + removed) |

### Tape Physics Features (ES Trades)
| Feature | Type | Description |
|---------|------|-------------|
| `tape_imbalance` | float | Buy/sell imbalance [-1, 1] |
| `tape_buy_vol` | int | Buy volume near level |
| `tape_sell_vol` | int | Sell volume near level |
| `tape_velocity` | float | Trades per second |
| `tape_sweep_detected` | bool | Sweep detected |
| `tape_sweep_direction` | string | UP, DOWN, or NONE |
| `tape_sweep_notional` | float | Sweep notional value |

### Fuel Physics Features (SPY 0DTE Options)
| Feature | Type | Description |
|---------|------|-------------|
| `gamma_exposure` | float | Net dealer gamma at level |
| `fuel_effect` | enum | AMPLIFY, DAMPEN, NEUTRAL |
| `fuel_net_dealer_gamma` | float | Same as gamma_exposure (alias) |
| `fuel_call_wall` | float | Call wall strike (if identified) |
| `fuel_put_wall` | float | Put wall strike (if identified) |
| `fuel_hvl` | float | Gamma flip level |
| `fuel_call_gamma` | float | Total call gamma near level |
| `fuel_put_gamma` | float | Total put gamma near level |

### Score Features
| Feature | Type | Description |
|---------|------|-------------|
| `break_score_raw` | float | Raw composite score (0-100) |
| `break_score_smooth` | float | EWMA smoothed score (0-100) |
| `liquidity_score` | float | S_L component (0-100) |
| `hedge_score` | float | S_H component (0-100) |
| `tape_score` | float | S_T component (0-100) |
| `signal` | enum | BREAK, REJECT, CONTESTED, NEUTRAL |
| `confidence` | enum | HIGH, MEDIUM, LOW |

### Runway Features
| Feature | Type | Description |
|---------|------|-------------|
| `runway_direction` | string | Expected move direction |
| `runway_next_level_price` | float | Next obstacle level |
| `runway_distance` | float | Distance to next obstacle ($) |
| `runway_quality` | enum | CLEAR, OBSTRUCTED |

### Outcome Features (Labels)
| Feature | Type | Description |
|---------|------|-------------|
| `outcome` | enum | BREAK, BOUNCE, CHOP |
| `future_price_5min` | float | Price 5 minutes after touch |
| `excursion_max` | float | Maximum favorable excursion |
| `excursion_min` | float | Maximum adverse excursion |

---

## Level Generation Logic

### PM_HIGH / PM_LOW
```python
# Pre-market: 04:00-09:30 ET
premarket = ohlcv[(time >= 04:00) & (time < 09:30)]
pm_high = premarket['high'].max()
pm_low = premarket['low'].min()
```

### OR_HIGH / OR_LOW (Opening Range)
```python
# Opening range: 09:30-09:45 ET (first 15 minutes)
opening_range = ohlcv[(time >= 09:30) & (time < 09:45)]
or_high = opening_range['high'].max()
or_low = opening_range['low'].min()
```

### SMA_200 / SMA_400 (2-Minute Bars)
```python
# Resample to 2-minute bars
ohlcv_2m = ohlcv.resample('2T').agg({
    'open': 'first', 'high': 'max',
    'low': 'min', 'close': 'last', 'volume': 'sum'
})

# Calculate SMAs
sma_200 = ohlcv_2m['close'].rolling(200).mean()
sma_400 = ohlcv_2m['close'].rolling(400).mean()

# Current SMA values (last computed)
current_sma_200 = sma_200.iloc[-1]
current_sma_400 = sma_400.iloc[-1]
```

### VWAP
```python
# Cumulative VWAP from session start (09:30 ET)
session = ohlcv[time >= 09:30]
vwap = (session['close'] * session['volume']).cumsum() / session['volume'].cumsum()
```

### STRIKE Levels
```python
# Option strikes with volume from option flow data
active_strikes = market_state.get_active_strikes(min_volume=100)
```

### CALL_WALL / PUT_WALL
```python
# Strikes with maximum gamma concentration
gamma_by_strike = market_state.get_gamma_by_strike()
call_wall = max(gamma_by_strike, key=lambda s: gamma_by_strike[s]['call_gamma'])
put_wall = max(gamma_by_strike, key=lambda s: gamma_by_strike[s]['put_gamma'])
```

---

## Pipeline Execution

### Single Date (Python API)
```python
from src.pipeline.vectorized_pipeline import VectorizedPipeline

pipeline = VectorizedPipeline()
signals_df = pipeline.run(date="2025-12-17")
```

### Batch Processing (CLI)
```bash
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

## Caching Strategy

To avoid redundant computation:

1. **Bronze tier** - Downloaded once per date, never regenerated
2. **OHLCV cache** - Computed once per date, stored in memory during pipeline run
3. **Level universe** - Computed once per date at start of pipeline
4. **Gold output** - One Parquet file per date, skip if exists

```
data/lake/gold/research/
├── signals_2025-12-16.parquet  # Per-date file
├── signals_2025-12-17.parquet
├── signals_2025-12-18.parquet
├── signals_2025-12-19.parquet
└── signals_multi_day.parquet   # Aggregated (regenerate on demand)
```

---

## Implementation Checklist

### Stage 1: Data Loading ✅
- [x] Load ES trades from DBN
- [x] Load ES MBP-10 from DBN
- [x] Load SPY options from Bronze (with tick rule aggressor)
- [x] Black-Scholes greeks computation (vectorized numpy)

### Stage 2: OHLCV ✅
- [x] Build 1-minute OHLCV from ES trades (vectorized pandas)
- [x] Convert ES → SPY prices
- [x] Build 2-minute OHLCV for SMA calculations

### Stage 3: Level Universe ✅
- [x] STRIKE levels from option flow
- [x] PM_HIGH / PM_LOW from pre-market (04:00-09:30 ET)
- [x] OR_HIGH / OR_LOW from opening range (09:30-09:45 ET)
- [x] SMA_200 (200-period on 2-min bars)
- [x] VWAP calculation (cumulative session VWAP)
- [x] CALL_WALL / PUT_WALL from gamma concentration
- [x] ROUND numbers (every $1)
- [x] SESSION_HIGH / SESSION_LOW (running extremes)

### Stage 4: Market State ✅
- [x] ES trades buffer (RingBuffer with time-window queries)
- [x] ES MBP-10 buffer (RingBuffer for depth snapshots)
- [x] Option flow aggregates (per-strike gamma/delta/volume)
- [x] Price converter (SPY ↔ ES 10:1 ratio)

### Stage 5: Touch Detection ✅
- [x] Vectorized touch detection using numpy broadcasting
- [x] Touch detection for ALL level types
- [x] is_first_15m flag (09:30-09:45 ET detection)
- [x] Direction (UP/DOWN) inference from close vs level

### Stage 6: Physics Computation ✅
- [x] FuelEngine (gamma_exposure, fuel_effect, call_wall, put_wall)
- [x] BarrierEngine (barrier_state, delta_liq, wall_ratio, depth_in_zone)
- [x] TapeEngine (imbalance, buy_vol, sell_vol, velocity, sweep_detected)
- [x] Batch processing with Numba JIT acceleration
- [ ] ScoreEngine (composite scores) - pending
- [ ] RoomToRun (runway metrics) - pending

### Stage 7: Labeling ✅
- [x] BREAK/BOUNCE/CHOP classification (vectorized)
- [x] future_price_5min (vectorized searchsorted)
- [x] excursion_max / excursion_min

### Stage 8: Output ✅
- [x] Per-date Parquet files (ZSTD compressed)
- [x] Multi-day aggregation (signals_vectorized.parquet)

---

## Completed Optimizations (Apple M4 Silicon)

The pipeline has been fully vectorized for optimal performance on Apple M4 Silicon:

### Performance Metrics (128GB RAM, M4 chip)
- **Data loading**: 1-2s per day (394K-819K trades)
- **OHLCV building**: 0.15-0.31s (vectorized pandas)
- **Touch detection**: ~0s (numpy broadcasting)
- **Physics computation**: 0.15-0.16s for 5K signals (Numba JIT)
- **Labeling**: 0.02s (vectorized searchsorted)
- **Total per day**: 20-27s

### Vectorization Techniques Used
1. **Numpy broadcasting** for touch detection
2. **Numba JIT compilation** for tape/barrier metrics
3. **Vectorized Black-Scholes** greeks (~100x faster)
4. **Searchsorted** for O(log n) future price lookups
5. **Pre-computed lookup structures** for market data

## Next Implementation Steps

1. **Integrate ScoreEngine** for composite break score calculation
2. **Add RoomToRun** for runway metrics (distance to next obstacle)
3. **Optimize MarketState initialization** (currently bottleneck at 18-24s)
4. **Add multiprocessing** for parallel date processing
5. **Implement incremental updates** for real-time streaming

---

## Expected Output Schema

After full implementation, each signal will have ~50 features:

```python
{
    # Identity (4)
    'event_id', 'ts_event_ns', 'date', 'symbol',

    # Level (5)
    'level_price', 'level_kind', 'level_id', 'direction', 'distance',

    # Context (6)
    'spot', 'is_first_15m', 'is_opening_range',
    'dist_to_sma_200', 'dist_to_sma_400', 'dist_to_vwap',

    # Barrier (9)
    'barrier_state', 'barrier_delta_liq', 'barrier_replenishment_ratio',
    'barrier_added', 'barrier_canceled', 'barrier_filled',
    'wall_ratio', 'depth_in_zone', 'churn',

    # Tape (7)
    'tape_imbalance', 'tape_buy_vol', 'tape_sell_vol', 'tape_velocity',
    'tape_sweep_detected', 'tape_sweep_direction', 'tape_sweep_notional',

    # Fuel (7)
    'gamma_exposure', 'fuel_effect', 'fuel_net_dealer_gamma',
    'fuel_call_wall', 'fuel_put_wall', 'fuel_hvl',
    'fuel_call_gamma', 'fuel_put_gamma',

    # Scores (7)
    'break_score_raw', 'break_score_smooth',
    'liquidity_score', 'hedge_score', 'tape_score',
    'signal', 'confidence',

    # Runway (4)
    'runway_direction', 'runway_next_level_price',
    'runway_distance', 'runway_quality',

    # Outcome (4)
    'outcome', 'future_price_5min', 'excursion_max', 'excursion_min'
}
```

Total: ~53 features per signal
