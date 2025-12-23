# Spymaster: Real-Time Market Physics Engine

**Asset**: SPY (equity + 0DTE options)
**Core Prediction**: Will price **BREAK** through or **BOUNCE** off critical levels?
**Method**: Trace dealer hedging flows, order book dynamics, and tape momentum

---

## The Physics Model

Price movement at critical levels is governed by three mechanical forces:

### 1. Barrier Physics (Liquidity)
**Source**: ES futures MBP-10 (top 10 depth levels)

Liquidity at a price level determines resistance to movement:
- **VACUUM**: Defending orders pulled → price slides through easily
- **WALL**: Orders replenishing faster than consumption → price rejected
- **ABSORPTION**: Large hidden buyer/seller absorbing flow

```
barrier_state = f(delta_liquidity, replenishment_rate, passive_volume)
```

### 2. Tape Physics (Momentum)
**Source**: ES futures time & sales

Aggressive order flow reveals institutional intent:
- **tape_imbalance**: Buy vs sell pressure in band around level
- **tape_velocity**: Trade arrival rate (institutional urgency)
- **sweep_detection**: Clustered aggressive prints lifting/hitting

```
momentum_signal = f(imbalance, velocity, sweep_count)
```

### 3. Fuel Physics (Dealer Gamma)
**Source**: SPY 0DTE option trades

Dealers must delta-hedge option positions, creating mechanical price pressure:

| Customer Action | Dealer Position | Hedging Behavior | Effect |
|-----------------|-----------------|------------------|--------|
| Buys calls/puts | Short gamma | Chase price moves | **AMPLIFY** |
| Sells calls/puts | Long gamma | Fade price moves | **DAMPEN** |

```python
gamma_exposure = sum(customer_sign * size * gamma * 100)  # per strike
fuel_effect = AMPLIFY if gamma_exposure < -threshold else DAMPEN if > threshold else NEUTRAL
```

**Critical Insight**: When dealers are short gamma near a strike, they must buy into rallies and sell into selloffs—accelerating breakouts. When long gamma, they do the opposite—supporting mean reversion.

---

## Data Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────┤
│  Databento DBN Files          │  Polygon S3 Flat Files          │
│  ├── ES futures trades        │  ├── SPY 0DTE option trades     │
│  └── ES MBP-10 depth          │  └── ~800k trades/day           │
│      (10 levels bid/ask)      │      (50MB compressed/day)      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BRONZE TIER                              │
│  Append-only Parquet, Hive partitioned by date                  │
│  ├── futures/trades/symbol=ES/date=YYYY-MM-DD/                  │
│  ├── futures/mbp10/symbol=ES/date=YYYY-MM-DD/                   │
│  └── options/trades/underlying=SPY/date=YYYY-MM-DD/             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PHYSICS ENGINES                             │
│  ├── BarrierEngine  → barrier_state, delta_liq, wall_ratio     │
│  ├── TapeEngine     → tape_imbalance, tape_velocity            │
│  └── FuelEngine     → gamma_exposure, fuel_effect              │
│                                                                  │
│  BlackScholesCalculator: Real delta/gamma (vectorized numpy)    │
│  TickRuleInference: BUY/SELL from price changes                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GOLD TIER                                │
│  ML-ready labeled signals: signals_multi_day.parquet            │
│  Schema: backend/features.json                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Schema

Full specification: [`backend/features.json`](backend/features.json)

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `spot` | float | Current SPY price |
| `level_price` | float | Strike/VWAP/round being tested |
| `level_kind` | enum | STRIKE, VWAP, ROUND, GAMMA_WALL |
| `direction` | enum | UP (approaching resistance) or DOWN (approaching support) |
| `distance` | float | Dollars from spot to level |
| `is_first_15m` | bool | High-volatility opening period |
| `barrier_state` | enum | VACUUM, WALL, ABSORPTION, CONSUMED, WEAK, NEUTRAL |
| `barrier_delta_liq` | float | Net liquidity change (ES contracts) |
| `wall_ratio` | float | Defending liquidity vs average depth |
| `tape_imbalance` | float | Buy/sell pressure [-1, 1] |
| `tape_velocity` | float | Trades per second |
| `gamma_exposure` | float | Net dealer gamma at level |
| `fuel_effect` | enum | AMPLIFY, DAMPEN, NEUTRAL |

### Labels

| Label | Definition |
|-------|------------|
| `BREAK` | Price moved through level by >$0.20 in approach direction |
| `BOUNCE` | Price reversed away from level by >$0.20 |
| `CHOP` | Indeterminate (stayed within $0.20) |

### Current Dataset Statistics

```
Signals: 800 (4 trading days)
Label Distribution: BREAK 60%, BOUNCE 38%, CHOP 2%
Gamma Coverage: 85% of signals have non-zero gamma_exposure

Fuel Effect → Outcome:
  AMPLIFY: 68% BREAK, 32% BOUNCE
  DAMPEN:  59% BREAK, 38% BOUNCE
  NEUTRAL: 62% BREAK, 38% BOUNCE
```

---

## Core Modules

### `src/core/`

| Module | Purpose |
|--------|---------|
| `market_state.py` | Central state store: ES depth buffer, trades buffer, option flow aggregates |
| `barrier_engine.py` | Classify liquidity state from MBP-10 depth changes |
| `tape_engine.py` | Compute momentum from ES trades |
| `fuel_engine.py` | Compute dealer gamma from option flows |
| `black_scholes.py` | Vectorized Greeks calculation (delta, gamma, theta, vega) |
| `level_signal_service.py` | Orchestrator: generate signals for all levels near spot |

### `src/ingestor/`

| Module | Purpose |
|--------|---------|
| `dbn_ingestor.py` | Stream Databento DBN files (ES trades + MBP-10) |
| `polygon_flatfiles.py` | Download SPY options from S3 flat files |

### `src/lake/`

| Module | Purpose |
|--------|---------|
| `bronze_writer.py` | Write raw events to Parquet (append-only) |
| `bronze_writer.BronzeReader` | Read Bronze tier with DuckDB |

### `src/pipeline/`

| Module | Purpose |
|--------|---------|
| `run_pipeline.py` | Single-day: load data → compute physics → label outcomes |
| `batch_process.py` | Multi-day: generate ML training dataset |

### `src/research/`

| Module | Purpose |
|--------|---------|
| `labeler.py` | Label level touches as BREAK/BOUNCE/CHOP |
| `experiment_runner.py` | Statistical analysis and backtesting |

---

## Key Algorithms

### Black-Scholes Greeks (Vectorized)

```python
# src/core/black_scholes.py
def compute_greeks_vectorized(S, K, T, r, sigma, is_call):
    """
    Compute delta and gamma for arrays of options.

    Args:
        S: Spot prices (numpy array)
        K: Strike prices (numpy array)
        T: Time to expiry in years (numpy array)
        r: Risk-free rate (scalar)
        sigma: Volatility (scalar or array)
        is_call: Boolean array (True for calls)

    Returns:
        (delta_array, gamma_array)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1)

    return delta, gamma
```

### Tick Rule Aggressor Inference

```python
# src/ingestor/polygon_flatfiles.py
# Infer trade direction from price changes (Lee-Ready fallback)
df['prev_price'] = df.groupby('option_symbol')['price'].shift(1)
df['aggressor'] = 0  # MID
df.loc[df['price'] > df['prev_price'], 'aggressor'] = 1   # BUY (uptick)
df.loc[df['price'] < df['prev_price'], 'aggressor'] = -1  # SELL (downtick)
```

### Gamma Flow Accumulation

```python
# src/core/market_state.py
def update_option_trade(self, trade, delta, gamma):
    customer_sign = trade.aggressor.value  # +1 BUY, -1 SELL, 0 MID
    gamma_notional = customer_sign * trade.size * gamma * 100
    dealer_gamma_change = -gamma_notional  # dealer takes opposite side

    self.option_flows[key].net_gamma_flow += dealer_gamma_change
```

---

## Usage

### Generate ML Training Data

```bash
cd backend

# Download SPY options from Polygon S3 (requires credentials in .env)
uv run python -m src.ingestor.polygon_flatfiles --download-all

# Run batch processing for all available dates
uv run python -m src.pipeline.batch_process

# Output: data/lake/gold/research/signals_multi_day.parquet
```

### Environment Variables

```bash
# .env
POLYGON_API_KEY=your_api_key
POLYGON_S3_ACCESS_KEY=your_s3_access_key
POLYGON_S3_SECRET_KEY=your_s3_secret_key
```

### Single-Day Pipeline

```bash
uv run python -m src.pipeline.run_pipeline --date 2025-12-18
```

---

## Extension Points

### Add New Level Types

```python
# src/core/level_universe.py
class LevelKind(Enum):
    STRIKE = "STRIKE"
    VWAP = "VWAP"
    ROUND = "ROUND"
    GAMMA_WALL = "GAMMA_WALL"
    # Add new types here
```

### Add New Physics Features

1. Add computation to appropriate engine (`barrier_engine.py`, `tape_engine.py`, `fuel_engine.py`)
2. Add field to `LevelSignalV1` schema in `src/common/schemas/levels_signals.py`
3. Update `features.json` documentation
4. Regenerate training data with `batch_process.py`

### Build ML Classifier

```python
# Example: src/research/ml_classifier.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet('data/lake/gold/research/signals_multi_day.parquet')

# Encode categoricals
df['fuel_effect_encoded'] = df['fuel_effect'].map({'AMPLIFY': -1, 'NEUTRAL': 0, 'DAMPEN': 1})
df['direction_encoded'] = df['direction'].map({'DOWN': -1, 'UP': 1})

# Features
X = df[['distance', 'gamma_exposure', 'fuel_effect_encoded', 'direction_encoded',
        'wall_ratio', 'tape_imbalance', 'is_first_15m']]

# Binary classification (drop CHOP)
df_binary = df[df['outcome'] != 'CHOP']
y = (df_binary['outcome'] == 'BREAK').astype(int)

# Train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Feature importance
for name, imp in zip(X.columns, clf.feature_importances_):
    print(f"{name}: {imp:.3f}")
```

---

## Potential Applications

### Real-Time Signal Generation
- Subscribe to live ES MBP-10 + SPY options feeds
- Compute physics features every 250ms
- Emit BREAK/BOUNCE predictions with confidence scores
- WebSocket broadcast to trading frontend

### Gamma Exposure Dashboard
- Visualize dealer gamma by strike in real-time
- Identify gamma walls (high open interest strikes)
- Track gamma flip level (where dealers switch from long to short)
- Alert when approaching high-gamma zones

### Backtesting Framework
- Replay historical data through physics engines
- Simulate P&L for break/bounce predictions
- Optimize entry/exit thresholds
- Time-of-day and volatility regime analysis

### Risk Management
- Quantify probability of level break before entry
- Adjust position size based on gamma regime
- Identify dangerous setups (VACUUM + AMPLIFY near support)

---

## File Structure

```
backend/
├── features.json              # ML feature schema documentation
├── src/
│   ├── common/
│   │   ├── config.py          # Tunable parameters
│   │   ├── event_types.py     # Canonical dataclasses
│   │   └── schemas/           # Pydantic + PyArrow schemas
│   ├── core/
│   │   ├── market_state.py    # Central state store
│   │   ├── barrier_engine.py  # Liquidity physics
│   │   ├── tape_engine.py     # Momentum physics
│   │   ├── fuel_engine.py     # Gamma physics
│   │   └── black_scholes.py   # Greeks calculator
│   ├── ingestor/
│   │   ├── dbn_ingestor.py    # Databento DBN reader
│   │   └── polygon_flatfiles.py # S3 options downloader
│   ├── lake/
│   │   └── bronze_writer.py   # Parquet read/write
│   ├── pipeline/
│   │   ├── run_pipeline.py    # Single-day processing
│   │   └── batch_process.py   # Multi-day batch
│   └── research/
│       ├── labeler.py         # Outcome labeling
│       └── experiment_runner.py # Statistical analysis
└── data/
    └── lake/
        ├── bronze/            # Raw append-only data
        └── gold/research/     # ML training datasets
```

---

## Technical Invariants

1. **Timestamps**: All events carry `ts_event_ns` (Unix nanoseconds UTC)
2. **0DTE Only**: Option processing filters to same-day expiration
3. **No Estimation**: Greeks computed via Black-Scholes, never hardcoded
4. **Tick Rule**: Aggressor inferred from price changes when not provided
5. **ES→SPY Conversion**: ES price / 10 ≈ SPY price
6. **Append-Only Bronze**: Never mutate raw data files
7. **Deterministic Replay**: Same inputs + config → identical outputs

---

## Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "pandas",
    "numpy",
    "scipy",
    "pyarrow",
    "duckdb",
    "boto3",
    "databento",
    "python-dotenv",
]
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Download options | `uv run python -m src.ingestor.polygon_flatfiles --download-all` |
| Generate training data | `uv run python -m src.pipeline.batch_process` |
| Single-day pipeline | `uv run python -m src.pipeline.run_pipeline --date YYYY-MM-DD` |
| Read training data | `pd.read_parquet('data/lake/gold/research/signals_multi_day.parquet')` |

---

**Core Hypothesis**: Dealer gamma hedging creates predictable mechanical pressure at option strikes. By quantifying this pressure alongside order book dynamics and tape momentum, we can predict whether price will break through or bounce off critical levels with edge over participants who only see price.
