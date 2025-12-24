# Features Module

**Purpose**: Context analysis and structural level identification for SPY 0DTE trading  
**Status**: Production (ContextEngine), Legacy (PhysicsEngine)  
**Primary User**: Research pipeline (`src/pipeline/`)

---

## Overview

The features module provides macro-level market context analysis:
- **Timing context**: Is it the opening 15 minutes (high volatility)?
- **Structural levels**: Where are PM high/low, SMA-200/400?
- **Level proximity**: Which levels are within actionable distance?

This complements the microstructure physics computed by `core/` engines (BarrierEngine, TapeEngine, FuelEngine).

---

## Components

### ✅ ContextEngine (Production)

**Purpose**: Identify structural price levels and market timing context

**Key Methods**:
```python
from src.features.context_engine import ContextEngine

# Initialize with OHLCV data
engine = ContextEngine(ohlcv_df=ohlcv)

# Check timing context
is_opening = engine.is_first_15m(ts_ns)  # True if 09:30-09:45 ET

# Get cached structural levels
pm_high = engine.get_premarket_high()    # Pre-market high (04:00-09:30 ET)
pm_low = engine.get_premarket_low()      # Pre-market low
sma_200 = engine.get_sma_200_at_time(ts_ns)  # SMA-200 on 2-min bars
sma_400 = engine.get_sma_400_at_time(ts_ns)  # SMA-400 on 2-min bars

# Find active levels near current price
levels = engine.get_active_levels(
    current_price=687.50,
    current_time=ts_ns
)
# Returns levels within $0.10 tolerance
```

**Data Requirements**:
- 1-minute OHLCV bars (timestamp, open, high, low, close, volume)
- Timestamps in Unix nanoseconds (UTC) or datetime with timezone

**Integration**:
- Used in `src/pipeline/run_pipeline.py` for level detection
- Used in `src/pipeline/batch_process.py` for batch processing

**Test Coverage**: 19 tests in `tests/test_context_engine.py`

---

### ⚠️ PhysicsEngine (Legacy)

**Status**: **Prototype replaced by production engines**

This was an early implementation of microstructure metrics:
- Wall ratio calculation
- Replenishment detection
- Tape velocity

**Replaced By**:
- `src/core/barrier_engine.py` - Order book liquidity physics
- `src/core/tape_engine.py` - Trade flow physics
- `src/core/fuel_engine.py` - Option gamma physics

**Test Coverage**: 10 tests in `tests/test_physics_engine.py` (legacy reference)

---

## Usage Example

### Research Pipeline Integration

```python
from src.features.context_engine import ContextEngine
from src.core.barrier_engine import BarrierEngine
from src.core.tape_engine import TapeEngine
from src.core.fuel_engine import FuelEngine

# Step 1: Build OHLCV from ES futures
ohlcv = build_ohlcv_from_es_trades(es_trades_df)

# Step 2: Initialize ContextEngine for level detection
context_engine = ContextEngine(ohlcv_df=ohlcv)

# Step 3: Initialize production physics engines
barrier_engine = BarrierEngine()
tape_engine = TapeEngine()
fuel_engine = FuelEngine()

# Step 4: Detect level touches
for ts_ns, price in price_stream:
    # Check if in opening volatility period
    is_first_15m = context_engine.is_first_15m(ts_ns)
    
    # Get nearby structural levels
    active_levels = context_engine.get_active_levels(price, ts_ns)
    
    for level in active_levels:
        # Compute microstructure physics at level
        barrier_state = barrier_engine.compute_barrier_state(
            level_price=level['level_price'],
            direction=direction,
            market_state=market_state
        )
        
        # ... rest of signal generation
```

---

## Level Detection Logic

### Pre-Market High/Low

**Calculation Window**: 04:00 - 09:30 ET  
**Formula**:
```python
pm_high = max(ohlcv['high']) where time in [04:00, 09:30) ET
pm_low = min(ohlcv['low']) where time in [04:00, 09:30) ET
```

**Use Case**: Key reference levels for intraday range

---

### SMA-200 and SMA-400

**Timeframe**: 2-minute bars (resampled from 1-minute OHLCV)  
**Formula**:
```python
ohlcv_2min = ohlcv.resample('2min').agg({
    'open': 'first',
    'high': 'max', 
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

sma_200 = ohlcv_2min['close'].rolling(window=200).mean()
sma_400 = ohlcv_2min['close'].rolling(window=400).mean()
```

**Lookup**: Forward-fill for intrabar queries (use most recent 2-min bar value)

**Use Case**: Mean reversion levels, trend bias indicators

---

### First 15 Minutes Detection

**Window**: 09:30:00 - 09:44:59 ET (inclusive start, exclusive end)  
**Use Case**: Opening volatility period - higher risk, wider stops needed

---

## Performance Characteristics

### Initialization Cost

- **Pre-market calculation**: O(n) where n = number of pre-market bars (~330 bars)
- **SMA calculation**: O(n) where n = number of 2-min bars (~960 bars for 8 hours)
- **Total init time**: ~50-100ms for typical trading day

### Query Cost

- **is_first_15m()**: O(1) - simple time check
- **get_active_levels()**: O(k) where k = number of level types (typically 2-4)
- **get_sma_at_time()**: O(log n) - binary search on pre-computed series

### Memory Footprint

- **OHLCV storage**: ~0.5 MB for full trading day (1-min bars)
- **Cached values**: ~10 KB (PM levels + SMA series)
- **Total**: <1 MB per trading day

---

## Configuration

### Constants (in ContextEngine)

```python
# Market hours (ET)
PREMARKET_START = time(4, 0, 0)     # 04:00 ET
MARKET_OPEN = time(9, 30, 0)         # 09:30 ET  
FIRST_15M_END = time(9, 45, 0)       # 09:45 ET

# Level detection
LEVEL_TOLERANCE_USD = 0.10  # $0.10 - levels within this distance are "active"

# SMA parameters
SMA_PERIOD = 200
SMA_400_PERIOD = 400
SMA_TIMEFRAME_MINUTES = 2  # Use 2-min bars for SMA
```

### Customization

To adjust level detection sensitivity:
```python
# Tighter tolerance (more conservative)
engine.LEVEL_TOLERANCE_USD = 0.05  # Only $0.05 away

# Wider tolerance (catch more levels)
engine.LEVEL_TOLERANCE_USD = 0.25  # Up to $0.25 away
```

---

## Testing

### Run ContextEngine Tests
```bash
cd backend
uv run pytest tests/test_context_engine.py -v
```

### Test Coverage
- ✅ Time context detection (6 tests)
- ✅ Level detection (8 tests)  
- ✅ SMA calculation (2 tests)
- ✅ Edge cases (3 tests)
- ✅ Integration scenarios (2 tests)

**Total**: 19/19 tests passing

---

## Future Enhancements

### Planned Features

1. **Opening Range High/Low**:
   - Calculate first 30 minutes (09:30-10:00 ET) high/low
   - Common intraday reference levels

2. **Session High/Low**:
   - Track session extremes dynamically
   - Update as new highs/lows are made

3. **VWAP Calculation**:
   - Volume-weighted average price
   - Standard institutional reference

4. **Volume Profile**:
   - Identify high-volume nodes
   - Point of control (POC) detection

### Integration Improvements

1. **Unify with LevelUniverse**:
   - Consolidate level detection logic
   - Reduce code duplication

2. **Vectorize Level Detection**:
   - Batch process multiple timestamps
   - Improve pipeline throughput

3. **Add to Core Service**:
   - Make structural levels available in live system
   - Not just research pipeline

---

## References

- **Interface Documentation**: [INTERFACES.md](INTERFACES.md)
- **Module Analysis**: [ANALYSIS.md](ANALYSIS.md)
- **Pipeline Integration**: [../pipeline/README.md](../pipeline/README.md)
- **Level Schema**: [../common/schemas/levels_signals.py](../common/schemas/levels_signals.py)

---

**Version**: 1.0  
**Last Updated**: 2025-12-23  
**Status**: Production (ContextEngine), Legacy (PhysicsEngine)

