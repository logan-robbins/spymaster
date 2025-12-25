# Features Module Interfaces

**Module**: `backend/src/features/`  
**Role**: Context analysis and structural level identification  
**Audience**: AI Coding Agents & Developers

---

## Module Purpose

Provides macro-level market context:
- Timing context (opening volatility period)
- Structural levels (PM high/low, SMA-200/400)
- Level proximity detection

Complements microstructure physics from `core/` engines.

---

## ContextEngine API

### Initialization

```python
from src.features.context_engine import ContextEngine
import pandas as pd

engine = ContextEngine(ohlcv_df: Optional[pd.DataFrame] = None)
```

**Parameters**:
- `ohlcv_df`: DataFrame with columns:
  - `timestamp`: Unix nanoseconds (int64) or datetime with UTC timezone
  - `open`: float
  - `high`: float
  - `low`: float
  - `close`: float
  - `volume`: int

**Returns**: ContextEngine instance

**Preprocessing**:
- Calculates pre-market high/low from 04:00-09:30 ET data
- Resamples to 2-minute bars for SMA calculation
- Computes SMA-200 and SMA-400 on close prices
- Caches all values for fast queries

---

### Method: is_first_15m

```python
def is_first_15m(self, ts: int) -> bool
```

**Purpose**: Check if timestamp is in opening volatility period (09:30-09:45 ET)

**Parameters**:
- `ts`: Unix timestamp in nanoseconds (UTC)

**Returns**: `bool`
- `True` if timestamp is between 09:30:00 and 09:44:59 ET (inclusive)
- `False` otherwise

**Example**:
```python
ts_ns = 1703167800000000000  # Some timestamp
is_opening = engine.is_first_15m(ts_ns)
print(f"Opening period: {is_opening}")
```

**Use Case**: Flag high-volatility opening period for risk management

---

### Method: get_premarket_high

```python
def get_premarket_high(self) -> Optional[float]
```

**Purpose**: Get cached pre-market high (04:00-09:30 ET)

**Returns**: `Optional[float]`
- Pre-market high price if data available
- `None` if no pre-market data

**Calculation**:
```python
pm_high = max(ohlcv['high']) where time in [04:00, 09:30) ET
```

**Example**:
```python
pm_high = engine.get_premarket_high()
if pm_high:
    print(f"PM High: ${pm_high:.2f}")
```

---

### Method: get_premarket_low

```python
def get_premarket_low(self) -> Optional[float]
```

**Purpose**: Get cached pre-market low (04:00-09:30 ET)

**Returns**: `Optional[float]`
- Pre-market low price if data available
- `None` if no pre-market data

**Calculation**:
```python
pm_low = min(ohlcv['low']) where time in [04:00, 09:30) ET
```

---

### Method: get_sma_200_at_time

```python
def get_sma_200_at_time(self, ts: int) -> Optional[float]
```

**Purpose**: Get SMA-200 value at specific timestamp

**Parameters**:
- `ts`: Unix timestamp in nanoseconds (UTC)

**Returns**: `Optional[float]`
- SMA-200 value at timestamp
- `None` if insufficient data (need 200 bars warmup)

**Calculation**:
- Uses 2-minute bars resampled from 1-minute OHLCV
- Rolling mean over 200 periods (= 400 minutes = 6.67 hours)
- Forward-fills intrabar queries

**Warmup**: Requires at least 200 × 2min = 400 minutes of data

**Example**:
```python
sma = engine.get_sma_200_at_time(ts_ns)
if sma:
    distance = current_price - sma
    print(f"Distance to SMA-200: ${distance:.2f}")
```

---

### Method: get_sma_400_at_time

```python
def get_sma_400_at_time(self, ts: int) -> Optional[float]
```

**Purpose**: Get SMA-400 value at specific timestamp

**Parameters**:
- `ts`: Unix timestamp in nanoseconds (UTC)

**Returns**: `Optional[float]`
- SMA-400 value at timestamp
- `None` if insufficient data (need 400 bars warmup)

**Calculation**:
- Uses 2-minute bars
- Rolling mean over 400 periods (= 800 minutes = 13.33 hours)
- Forward-fills intrabar queries

**Warmup**: Requires at least 400 × 2min = 800 minutes of data

---

### Method: get_active_levels

```python
def get_active_levels(
    self, 
    current_price: float,
    current_time: int
) -> List[Dict[str, Any]]
```

**Purpose**: Find all structural levels within tolerance of current price

**Parameters**:
- `current_price`: Current price in dollars (e.g., 687.50)
- `current_time`: Current timestamp in nanoseconds (UTC)

**Returns**: `List[Dict[str, Any]]` where each dict contains:
- `level_kind`: `LevelKind` enum value (PM_HIGH, PM_LOW, SMA_200, SMA_400)
- `level_price`: `float` - Price of the level
- `distance`: `float` - Absolute distance from current price

**Tolerance**: Levels within `$0.10` (configurable via `LEVEL_TOLERANCE_USD`)

**Example**:
```python
levels = engine.get_active_levels(
    current_price=687.55,
    current_time=ts_ns
)

for level in levels:
    print(f"{level['level_kind'].name}: ${level['level_price']:.2f} "
          f"(distance: ${level['distance']:.3f})")
```

**Output**:
```
SMA_200: $687.50 (distance: $0.050)
PM_HIGH: $687.60 (distance: $0.050)
```

**Use Case**: Detect level confluence, identify support/resistance zones

---

## Data Schemas

### Input: OHLCV DataFrame

```python
pd.DataFrame({
    'timestamp': pd.Series(dtype='datetime64[ns, UTC]'),  # Or int64 Unix nanoseconds
    'open': pd.Series(dtype='float64'),
    'high': pd.Series(dtype='float64'),
    'low': pd.Series(dtype='float64'),
    'close': pd.Series(dtype='float64'),
    'volume': pd.Series(dtype='int64')
})
```

**Requirements**:
- Timestamps must be chronologically sorted
- No gaps >1 minute (for accurate SMA calculation)
- At least 400 minutes of data for SMA-200 (200 × 2-min bars)
- At least 800 minutes of data for SMA-400 (400 × 2-min bars)

**Example**:
```python
ohlcv = pd.DataFrame({
    'timestamp': pd.date_range('2025-12-16 04:00:00', periods=480, freq='1min', tz='UTC'),
    'open': [687.0] * 480,
    'high': [687.5] * 480,
    'low': [686.5] * 480,
    'close': [687.0] * 480,
    'volume': [10000] * 480
})
```

---

### Output: Active Levels

```python
[
    {
        'level_kind': LevelKind.PM_HIGH,  # Enum from src.common.schemas.levels_signals
        'level_price': 687.60,            # float
        'distance': 0.05                  # float (absolute distance)
    },
    {
        'level_kind': LevelKind.SMA_200,
        'level_price': 687.50,
        'distance': 0.05
    }
]
```

**Level Kinds Supported**:
- `LevelKind.PM_HIGH`: Pre-market high
- `LevelKind.PM_LOW`: Pre-market low
- `LevelKind.SMA_200`: SMA-200 on 2-min bars
- `LevelKind.SMA_400`: SMA-400 on 2-min bars

**Not Yet Implemented**:
- `LevelKind.OR_HIGH`: Opening range high (first 30 min)
- `LevelKind.OR_LOW`: Opening range low
- `LevelKind.SESSION_HIGH`: Session high (intraday)
- `LevelKind.SESSION_LOW`: Session low
- `LevelKind.VWAP`: Volume-weighted average price

---

## Configuration Constants

```python
class ContextEngine:
    # Market hours (ET)
    PREMARKET_START = time(4, 0, 0)     # 04:00 ET
    MARKET_OPEN = time(9, 30, 0)         # 09:30 ET
    FIRST_15M_END = time(9, 45, 0)       # 09:45 ET
    
    # Level detection
    LEVEL_TOLERANCE_USD = 0.10  # $0.10
    
    # SMA parameters
    SMA_PERIOD = 200
    SMA_400_PERIOD = 400
    SMA_TIMEFRAME_MINUTES = 2
```

**Customization**:
```python
engine = ContextEngine(ohlcv_df)
engine.LEVEL_TOLERANCE_USD = 0.05  # Tighter tolerance
```

---

## Error Handling

### Invalid Input

```python
# Empty DataFrame
engine = ContextEngine(pd.DataFrame())
# Result: No error, but get_premarket_high() returns None

# Missing 'timestamp' column
engine = ContextEngine(pd.DataFrame({'close': [1, 2, 3]}))
# Result: Raises ValueError("OHLCV DataFrame must have 'timestamp' column")
```

### Insufficient Data

```python
# Not enough data for SMA-200
ohlcv_short = generate_ohlcv(num_minutes=100)  # Only 100 minutes
engine = ContextEngine(ohlcv_short)

sma = engine.get_sma_200_at_time(ts_ns)
# Result: sma = None (need 400 minutes for 200 × 2-min bars)
```

### Timezone Handling

```python
# Timestamps without timezone
ohlcv_no_tz = pd.DataFrame({
    'timestamp': pd.date_range('2025-12-16', periods=10, freq='1min'),  # No tz
    ...
})
engine = ContextEngine(ohlcv_no_tz)
# Result: Automatically treats as UTC and converts to ET internally
```

---

## Performance Contracts

### Initialization

| Operation | Time Complexity | Typical Time |
|-----------|----------------|--------------|
| Load OHLCV | O(n) | ~10ms for 480 bars |
| Calculate PM high/low | O(n) | ~5ms |
| Resample to 2-min | O(n) | ~15ms |
| Calculate SMA-200 | O(n) | ~20ms |
| Calculate SMA-400 | O(n) | ~20ms |
| **Total Init** | **O(n)** | **~70ms** |

Where n = number of 1-minute bars (~480 for 8-hour day)

### Query

| Method | Time Complexity | Typical Time |
|--------|----------------|--------------|
| `is_first_15m()` | O(1) | <1μs |
| `get_premarket_high()` | O(1) | <1μs |
| `get_sma_200_at_time()` | O(log n) | <10μs |
| `get_active_levels()` | O(k) | <50μs |

Where k = number of level types (typically 2-4)

### Memory

| Component | Size |
|-----------|------|
| OHLCV (480 bars) | ~40 KB |
| SMA series | ~8 KB |
| Cached values | ~1 KB |
| **Total** | **~50 KB** |

---

## Integration Patterns

### Pattern 1: Research Pipeline

```python
from src.features.context_engine import ContextEngine
from src.core.market_state import MarketState
from src.core.barrier_engine import BarrierEngine

# Initialize with historical data
ohlcv = load_ohlcv_from_es_trades(date="2025-12-16")
context_engine = ContextEngine(ohlcv_df=ohlcv)

# Initialize market state + physics engines
market_state = MarketState()
barrier_engine = BarrierEngine()

# Process each price update
for ts_ns, price in price_stream:
    # Step 1: Get context
    is_opening = context_engine.is_first_15m(ts_ns)
    active_levels = context_engine.get_active_levels(price, ts_ns)
    
    # Step 2: For each level, compute microstructure physics
    for level in active_levels:
        barrier_state = barrier_engine.compute_barrier_state(
            level_price=level['level_price'],
            direction=direction,
            market_state=market_state
        )
        
        # Combine context + physics → signal
        signal = create_level_signal(
            level_kind=level['level_kind'],
            is_first_15m=is_opening,
            barrier_state=barrier_state,
            ...
        )
```

### Pattern 2: Level Touch Detection

```python
def detect_level_touches(ohlcv_df, context_engine, tolerance=0.05):
    """
    Detect when price crosses structural levels.
    """
    touches = []
    
    for i, row in ohlcv_df.iterrows():
        ts_ns = int(row['timestamp'].value)
        
        # Check high cross
        active_up = context_engine.get_active_levels(row['high'], ts_ns)
        for level in active_up:
            touches.append({
                'ts_ns': ts_ns,
                'price': row['high'],
                'level_kind': level['level_kind'],
                'direction': 'UP'
            })
        
        # Check low cross
        active_down = context_engine.get_active_levels(row['low'], ts_ns)
        for level in active_down:
            touches.append({
                'ts_ns': ts_ns,
                'price': row['low'],
                'level_kind': level['level_kind'],
                'direction': 'DOWN'
            })
    
    return pd.DataFrame(touches)
```

---

## Dependencies

### Required

```python
import pandas as pd
import numpy as np  # Implicit via pandas
from datetime import datetime, time, timezone
from typing import List, Optional, Dict, Any
```

### Internal

```python
from src.common.schemas.levels_signals import LevelKind
```

**Note**: ContextEngine has minimal dependencies and can run standalone.

---

## Testing

### Unit Test Coverage

**File**: `tests/test_context_engine.py`

**Coverage**:
- ✅ Time context detection (6 tests)
- ✅ Level detection (8 tests)
- ✅ SMA calculation (2 tests)
- ✅ Edge cases (3 tests)

**Run Tests**:
```bash
cd backend
uv run pytest tests/test_context_engine.py -v
```

---

## Known Limitations

1. **No VWAP**: Not implemented yet (planned)
2. **No Opening Range**: OR high/low not implemented (planned)
3. **No Session Extremes**: Dynamic high/low tracking not implemented (planned)
4. **Fixed tolerance**: Level tolerance is global, not per-level-type
5. **No level priorities**: All levels treated equally (no weighting)
6. **DST transitions**: Uses fixed ET offset, doesn't handle DST automatically
7. **No level caching**: `get_active_levels()` recomputes each call

---

## Future API Extensions

### Planned Methods

```python
# Opening Range (first 30 minutes)
def get_opening_range_high(self) -> Optional[float]
def get_opening_range_low(self) -> Optional[float]

# Session Extremes (dynamic)
def get_session_high(self, up_to_time: int) -> Optional[float]
def get_session_low(self, up_to_time: int) -> Optional[float]

# VWAP
def get_vwap_at_time(self, ts: int) -> Optional[float]

# Volume Profile
def get_point_of_control(self) -> Optional[float]
def get_high_volume_nodes(self, num_nodes: int = 3) -> List[float]
```

---

## References

- **Implementation**: [context_engine.py](context_engine.py)
- **Tests**: [../../tests/test_context_engine.py](../../tests/test_context_engine.py)
- **Module README**: [README.md](README.md)
- **Level Schema**: [../common/schemas/levels_signals.py](../common/schemas/levels_signals.py)
- **Pipeline Integration**: [../pipeline/README.md](../pipeline/README.md)

---

**Version**: 1.0  
**Last Updated**: 2025-12-23  
**Status**: Production
