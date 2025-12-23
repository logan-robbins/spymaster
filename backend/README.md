# Spymaster Backend

ML feature engineering pipeline for SPY 0DTE options trading signals.

## Core Concept

**Goal:** Predict whether SPY price will BREAK through or BOUNCE off key price levels (strikes, VWAP, session highs/lows).

**Trading Context:** SPY options are traded at $1 strike intervals. When price approaches a strike level (e.g., $685), dealers must hedge their gamma exposure. This creates observable patterns in:
1. **Order book liquidity** - Are dealers defending or abandoning the level?
2. **Trade flow** - Is aggressive buying/selling pushing through or getting absorbed?
3. **Options flow** - Will dealer hedging amplify or dampen the move?

**Signal Generation:** When price enters the "critical zone" (±$0.25 of a level), we compute physics features and label the outcome based on whether price moved ≥$2.00 (2 strikes) in either direction within 5 minutes.

## Critical: ES/SPY Price Conversion

**This platform uses ES futures as a liquidity proxy for SPY.**

```
ES_price = SPY_price × 10
```

| SPY | ES | Notes |
|-----|-----|-------|
| $685.00 | $6850.00 | Price conversion |
| $1 strike interval | $10 / 40 ticks | Strike spacing |
| ±$0.25 critical zone | ±$2.50 / 10 ticks | MONITOR_BAND |
| ±$0.20 barrier zone | ±$2.00 / 8 ticks | BARRIER_ZONE_ES_TICKS |

**Why ES?** SPY equity has no public order book. ES futures provide:
- Deep MBP-10 order book data
- Trade-by-trade flow with aggressor flags
- Direct hedging instrument for SPY option dealers

**ALL barrier and tape computations operate on ES data, then convert back to SPY scale for output.**

## Quick Start

```bash
cd backend/

# Install dependencies
uv sync

# Process all available dates
uv run python -m src.pipeline.vectorized_pipeline --all

# Validate generated data
uv run python -m scripts.validate_data --verbose
```

## Pipeline Stages

```
Stage 1: Load Data
    ├── ES trades (Databento DBN) - price, size, aggressor
    ├── ES MBP-10 (Databento DBN) - 10 levels of bid/ask depth
    └── SPY 0DTE options (Polygon) - trades for gamma calculation

Stage 2: Build OHLCV
    └── 1-minute bars from ES trades (converted to SPY scale)

Stage 3: Generate Level Universe
    ├── STRIKE levels ($1 intervals near spot)
    ├── ROUND levels ($1 intervals)
    ├── PM_HIGH, PM_LOW (pre-market extremes)
    ├── OR_HIGH, OR_LOW (opening range)
    ├── SESSION_HIGH, SESSION_LOW
    ├── VWAP, SMA_200
    └── CALL_WALL, PUT_WALL (from options flow)

Stage 4: Detect Touches
    └── When OHLCV high/low crosses a level price

Stage 5: Filter to Critical Zone
    └── Keep only signals where |close - level| ≤ $0.25

Stage 6: Compute Physics (FORWARD-looking from touch)
    ├── Barrier metrics (60s window) - order book dynamics
    ├── Tape metrics (60s window) - trade flow imbalance
    └── Fuel metrics - dealer gamma exposure

Stage 7: Compute Approach Context (BACKWARD-looking)
    └── How price approached the level (velocity, bars, distance)

Stage 8: Label Outcomes (FORWARD 5 minutes)
    ├── BREAK: moved ≥$2.00 through level
    ├── BOUNCE: moved ≥$2.00 away from level
    └── CHOP: didn't move $2.00 either direction

Output: data/lake/gold/research/signals_vectorized.parquet
```

## Time Window Directions

**CRITICAL FOR AI AGENTS:** Touch timestamps are at the START of 1-minute bars.

| Computation | Direction | Window | Why |
|-------------|-----------|--------|-----|
| Barrier physics | FORWARD | 60s | Capture trades/quotes during the bar |
| Tape physics | FORWARD | 60s | Capture trade flow during the bar |
| Approach context | BACKWARD | 10 min | How price got to this level |
| Outcome labeling | FORWARD | 5 min | What happened after |

**Wrong:** Looking backward from touch timestamp misses all activity.
**Right:** Looking forward from touch timestamp captures the bar's activity.

## Feature Schema

### Identity
- `event_id` - UUID
- `ts_ns` - Unix nanoseconds UTC
- `date` - Trading date
- `symbol` - Always "SPY"

### Level Context
- `level_price` - The price level being tested
- `level_kind` / `level_kind_name` - Type (STRIKE, VWAP, PM_HIGH, etc.)
- `direction` - UP (approaching resistance) or DOWN (approaching support)
- `distance` - |spot - level| in dollars (filtered to ≤$0.25)
- `spot` - Current SPY price

### Barrier Physics (ES Order Book)
Measures liquidity dynamics at the level. Source: ES MBP-10.

- `barrier_state` - Classification:
  - `VACUUM` - Liquidity pulled without fills → easy break
  - `WALL` - Liquidity replenishing → likely reject
  - `ABSORPTION` - Large orders absorbing flow
  - `CONSUMED` - Defending liquidity was filled
  - `WEAK` - Minimal liquidity present
  - `NEUTRAL` - No significant signal
- `barrier_delta_liq` - Net change in defending liquidity (ES contracts)
- `barrier_replenishment_ratio` - added / (canceled + filled)
- `wall_ratio` - Defending liquidity vs baseline

### Tape Physics (ES Trade Flow)
Measures buy/sell pressure near the level. Source: ES trades.

- `tape_imbalance` - (buy_vol - sell_vol) / total, range [-1, 1]
- `tape_buy_vol` - Buy volume in price band (ES contracts)
- `tape_sell_vol` - Sell volume in price band (ES contracts)
- `tape_velocity` - Trades per second
- `sweep_detected` - Rapid multi-level execution detected

### Fuel Physics (Dealer Gamma)
Measures how dealer hedging will affect price movement. Source: SPY 0DTE options.

- `gamma_exposure` - Net dealer gamma at level
  - Negative = dealers short gamma → will chase moves (AMPLIFY)
  - Positive = dealers long gamma → will fade moves (DAMPEN)
- `fuel_effect` - `AMPLIFY`, `DAMPEN`, or `NEUTRAL`

### Approach Context (Backward-Looking)
How price approached the level - critical for ML.

- `approach_velocity` - $/minute toward level (positive = moving toward)
- `approach_bars` - Consecutive 1-min bars moving toward level
- `approach_distance` - Total price distance traveled in lookback
- `prior_touches` - Previous touches at this level today
- `bars_since_open` - Session timing context

### Outcome Labels
Ground truth for supervised learning.

- `outcome` - `BREAK`, `BOUNCE`, or `CHOP`
- `future_price_5min` - SPY price 5 minutes after signal
- `excursion_max` - Max favorable move in lookforward window
- `excursion_min` - Max adverse move in lookforward window

**Threshold:** $2.00 (2 strikes) required for BREAK/BOUNCE. This ensures outcomes are meaningful for options trading.

## Key Configuration (src/common/config.py)

```python
# Critical zone - where bounce/break decision happens
MONITOR_BAND: float = 0.25      # $0.25 SPY = 10 ES ticks

# Barrier zone around strike-aligned levels
BARRIER_ZONE_ES_TICKS: int = 8  # ±8 ticks = ±$2 ES = ±$0.20 SPY

# Physics computation windows (FORWARD from touch)
W_b: float = 60.0               # Barrier engine window (seconds)
W_t: float = 60.0               # Tape engine window (seconds)

# Outcome labeling
OUTCOME_THRESHOLD: float = 2.0  # $2.00 = 2 strikes for BREAK/BOUNCE
LOOKFORWARD_MINUTES: int = 5    # Forward window for outcome
LOOKBACK_MINUTES: int = 10      # Backward window for approach context
```

## File Structure

```
backend/
├── src/
│   ├── pipeline/
│   │   └── vectorized_pipeline.py    # Main entry point
│   ├── core/
│   │   ├── vectorized_engines.py     # Batch physics (Numba JIT)
│   │   ├── barrier_engine.py         # Order book physics
│   │   ├── tape_engine.py            # Trade flow physics
│   │   ├── fuel_engine.py            # Gamma exposure physics
│   │   ├── market_state.py           # Ring buffers for streaming
│   │   └── black_scholes.py          # Vectorized Greeks
│   ├── ingestor/
│   │   └── dbn_ingestor.py           # Databento DBN reader
│   ├── lake/
│   │   └── bronze_writer.py          # Options data reader
│   └── common/
│       ├── config.py                 # All tunable parameters
│       └── event_types.py            # Data structures
├── scripts/
│   └── validate_data.py              # Data quality checks
├── data/
│   └── lake/gold/research/           # Output parquet files
├── dbn-data/                         # Raw Databento DBN files
│   ├── trades/                       # ES trades
│   └── MBP-10/                       # ES order book snapshots
├── features.json                     # Complete feature schema
└── README.md
```

## Data Sources

| Source | Provider | Format | Content |
|--------|----------|--------|---------|
| ES Trades | Databento | DBN | price, size, aggressor flag |
| ES MBP-10 | Databento | DBN | 10 levels bid/ask depth |
| SPY Options | Polygon | CSV.gz | 0DTE option trades |

## Common Tasks

### Process New Data
```bash
# Single date
uv run python -m src.pipeline.vectorized_pipeline --date 2025-12-20

# All available dates
uv run python -m src.pipeline.vectorized_pipeline --all
```

### Validate Output
```bash
uv run python -m scripts.validate_data --verbose
```

### Run Tests
```bash
uv run pytest tests/
```

## AI Agent Notes

1. **ES/SPY conversion is everywhere.** When debugging barrier/tape issues, always check if prices are being converted correctly (×10 for SPY→ES, ÷10 for ES→SPY).

2. **Time windows look FORWARD.** Touch timestamps are at bar START. Physics engines must look forward `[ts, ts + window_ns]` not backward.

3. **Config is the source of truth.** All thresholds, windows, and bands are in `src/common/config.py`. Don't hardcode values.

4. **Vectorized only.** There is no non-vectorized pipeline. All computation uses numpy/numba for performance.

5. **$2.00 outcome threshold.** BREAK/BOUNCE require ≥$2.00 moves (2 strikes). Smaller moves are CHOP.

6. **Barrier zone = ±8 ES ticks.** This is ±$0.20 SPY around strike-aligned levels. Wider zones dilute the signal.

7. **Critical zone = ±$0.25 SPY.** Only signals where close is within $0.25 of level are kept. This is where the decision happens.
