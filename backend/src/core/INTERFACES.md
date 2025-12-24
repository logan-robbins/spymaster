# Core Module Interfaces

**Module**: `backend/src/core/`  
**Role**: Physics engines and signal generation  
**Audience**: AI Coding Agents

---

## Module Purpose

Implements break/bounce physics classification using ES futures liquidity data and SPY option flow. Publishes level signals to NATS for Gateway broadcast and Gold storage.

---

## Input Interfaces

### Market State Updates (from Ingestor via NATS)

**Consumed NATS Subjects**:
- `market.stocks.trades` → SPY trades
- `market.stocks.quotes` → SPY quotes
- `market.options.trades` → SPY option trades (for fuel engine)
- `market.futures.trades` → ES futures trades
- `market.futures.mbp10` → ES MBP-10 depth updates

**Processing**: All events buffered in `MarketState` ring buffers (event-driven ingestion).

---

## Output Interfaces

### Level Signals Payload

**Published to**: `levels.signals` NATS subject  
**Format**: JSON  
**Cadence**: Every `CONFIG.SNAP_INTERVAL_MS` (default: 250ms)

**Schema**:
```json
{
  "ts": 1715629300123,
  "spy": {
    "spot": 687.42,
    "bid": 687.41,
    "ask": 687.43
  },
  "levels": [
    {
      "id": "STRIKE_687",
      "price": 687.0,
      "kind": "STRIKE",
      "direction": "SUPPORT",
      "distance": 0.42,
      "break_score_raw": 88.5,
      "break_score_smooth": 81.2,
      "signal": "BREAK",
      "confidence": "HIGH",
      "barrier": {
        "state": "VACUUM",
        "delta_liq": -8200.0,
        "replenishment_ratio": 0.15,
        "added": 3100,
        "canceled": 9800,
        "filled": 1500
      },
      "tape": {
        "imbalance": -0.45,
        "buy_vol": 120000,
        "sell_vol": 320000,
        "velocity": -0.08,
        "sweep": {
          "detected": true,
          "direction": "DOWN",
          "notional": 1250000.0
        }
      },
      "fuel": {
        "effect": "AMPLIFY",
        "net_dealer_gamma": -185000.0,
        "call_wall": 690.0,
        "put_wall": 684.0,
        "hvl": 687.0
      },
      "runway": {
        "direction": "DOWN",
        "next_level_id": "PUT_WALL",
        "next_level_price": 684.0,
        "distance": 3.0,
        "quality": "CLEAR"
      },
      "note": "Vacuum + dealers chase; sweep confirms"
    }
  ]
}
```

---

## Engine Interfaces

### 1. MarketState (`market_state.py`)

**Purpose**: Central state store for all market data.

**Update Methods** (called by Ingestor):
```python
def update_es_mbp10(mbp: MBP10) -> None
def update_es_trade(trade: FuturesTrade) -> None
def update_option_trade(trade: OptionTrade, delta: float, gamma: float) -> None
```

**Query Methods** (called by Engines):
```python
def get_spot() -> float  # SPY-equivalent
def get_bid_ask() -> (float, float)
def get_es_mbp10_snapshot() -> MBP10
def get_es_trades_in_window(ts_ns: int, window_sec: float) -> List[FuturesTrade]
def get_option_flows_near_level(level_price: float, strike_range: float) -> List[OptionFlowAggregate]
```

---

### 2. BarrierEngine (`barrier_engine.py`)

**Purpose**: Compute liquidity state from ES MBP-10 depth changes.

**Interface**:
```python
def compute_barrier_state(
    level_price: float,      # SPY level (e.g., 687.0)
    direction: Direction,     # SUPPORT or RESISTANCE
    market_state: MarketState
) -> BarrierMetrics
```

**Output**:
```python
@dataclass
class BarrierMetrics:
    state: BarrierState  # VACUUM | WALL | ABSORPTION | CONSUMED | WEAK | NEUTRAL
    delta_liq: float
    replenishment_ratio: float
    added_size: float
    canceled_size: float
    filled_size: float
    confidence: float
```

---

### 3. TapeEngine (`tape_engine.py`)

**Purpose**: Compute tape momentum from ES trades.

**Interface**:
```python
def compute_tape_state(
    level_price: float,
    market_state: MarketState
) -> TapeMetrics
```

**Output**:
```python
@dataclass
class TapeMetrics:
    imbalance: float  # -1 to +1
    buy_vol: int
    sell_vol: int
    velocity: float   # $/sec
    sweep: SweepDetection
    confidence: float
```

---

### 4. FuelEngine (`fuel_engine.py`)

**Purpose**: Estimate dealer gamma effect from SPY option flow.

**Interface**:
```python
def compute_fuel_state(
    level_price: float,
    market_state: MarketState,
    exp_date_filter: Optional[str] = None
) -> FuelMetrics
```

**Output**:
```python
@dataclass
class FuelMetrics:
    effect: FuelEffect  # AMPLIFY | DAMPEN | NEUTRAL
    net_dealer_gamma: float
    call_wall: Optional[float]
    put_wall: Optional[float]
    hvl: Optional[float]
    confidence: float
```

---

### 5. ScoreEngine (`score_engine.py`)

**Purpose**: Combine physics into composite break score.

**Interface**:
```python
def compute_score(
    barrier_metrics: BarrierMetrics,
    tape_metrics: TapeMetrics,
    fuel_metrics: FuelMetrics,
    break_direction: str,
    ts_ns: int,
    distance_to_level: float
) -> CompositeScore
```

**Output**:
```python
@dataclass
class CompositeScore:
    raw_score: float              # 0-100
    component_scores: dict        # S_L, S_H, S_T
    signal: Signal                # BREAK | REJECT | CONTESTED | NEUTRAL
    confidence: Confidence        # HIGH | MEDIUM | LOW
```

**Formula**:
```
S = w_L * S_L + w_H * S_H + w_T * S_T
  = 0.45 * Liquidity + 0.35 * Hedge + 0.20 * Tape
```

---

### 6. LevelSignalService (`level_signal_service.py`)

**Purpose**: Orchestrate all engines and produce WebSocket payload.

**Interface**:
```python
def compute_level_signals() -> Dict[str, Any]
```

**Flow**:
1. Get spot, bid/ask from MarketState
2. Generate level universe
3. Filter to levels within MONITOR_BAND
4. For each level:
   - Compute BarrierMetrics, TapeMetrics, FuelMetrics
   - Compute CompositeScore
   - Apply smoothing
   - Compute Runway
5. Build payload dict

---

## Price Conversion Contract

**Critical**: Levels are expressed in SPY terms. Engines convert to ES internally.

```python
# User specifies level in SPY
level_price_spy = 687.0

# Engine converts for ES query
es_level = market_state.converter.spy_to_es(687.0)  # → 6870.0

# Query ES depth at converted price
depth = market_state.get_es_depth_at(es_level)

# Convert results back to SPY for display
output_spy = market_state.converter.es_to_spy(result_es)
```

---

## Direction Context

| Condition | Level Type | Break Direction | Reject Direction |
|-----------|-----------|-----------------|------------------|
| spot > L  | SUPPORT   | DOWN            | UP               |
| spot < L  | RESISTANCE| UP              | DOWN             |

---

## Critical Parameters (from CONFIG)

**Windows**:
- `W_b = 10.0` seconds (barrier)
- `W_t = 5.0` seconds (tape)
- `W_g = 60.0` seconds (fuel)
- `W_v = 3.0` seconds (velocity)

**Thresholds**:
- `MONITOR_BAND = 0.50` (compute signals if |spot - level| ≤ $0.50)
- `TOUCH_BAND = 0.05` (tight band for "touching")
- `R_vac = 0.3` (VACUUM threshold)
- `R_wall = 1.5` (WALL threshold)
- `F_thresh = 100` (delta liquidity threshold, ES contracts)

**Weights**:
- `w_L = 0.45` (liquidity)
- `w_H = 0.35` (hedge)
- `w_T = 0.20` (tape)

---

## Determinism Contract

**Guarantee**: Same inputs + config → same outputs (within FP rounding).

**Requirements**:
1. Event-time ordering (sort by `ts_event_ns`)
2. Deterministic tie-breaking (use `ts_recv_ns`, then `seq`)
3. No random seeds or system time in computation
4. Fixed config (don't mutate during replay)

---

## References

- Full module documentation: `backend/src/core/README.md`
- Engine specifications: See README §6 (Engine Specifications)
- Configuration: `backend/src/common/config.py`

