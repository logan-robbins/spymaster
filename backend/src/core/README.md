# Core Module: Break/Reject Physics Engine

**Target Audience**: AI Coding Agents  
**Asset**: SPY only (underlying + 0DTE options)  
**Data Source**: ES futures (MBP-10 + trades) for barrier/tape physics, SPY options for fuel  
**Reference**: See `/PLAN.md` §5 (Core Engines) and §12 (Agent Assignments)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Purpose & Physics Model](#module-purpose--physics-model)
3. [Data Flow & Dependencies](#data-flow--dependencies)
4. [Core Components](#core-components)
5. [Interfaces & Contracts](#interfaces--contracts)
6. [Engine Specifications](#engine-specifications)
7. [Price Conversion (ES ↔ SPY)](#price-conversion-es--spy)
8. [Configuration](#configuration)
9. [Adding New Engines](#adding-new-engines)
10. [Testing Strategy](#testing-strategy)
11. [Critical Implementation Notes](#critical-implementation-notes)

---

## Architecture Overview

The `core` module implements a real-time physics-based decision engine that classifies whether critical price levels will **BREAK** (fail) or **REJECT** (hold) as price approaches them. The system operates on a fixed snap cadence (100–250ms) while ingesting events continuously.

### Key Design Principles

1. **Event-driven ingestion, time-bucketed scoring**: Ingest every MBP-10 update and trade immediately to avoid "churn blindness," but compute and publish scores on a fixed snap tick.
2. **No hindsight calibration**: All thresholds are mechanical/tunable constants (no ML training in v1).
3. **Deterministic replay**: Same inputs + config → same outputs (within floating-point rounding policy).
4. **Separation of concerns**: Each engine focuses on one dimension of physics (liquidity, momentum, hedging).
5. **Price conversion abstraction**: Levels are expressed in SPY terms, but barrier/tape engines query ES data internally.

### Module Structure

```
backend/src/core/
├── market_state.py          # Central state store (ES MBP-10, trades, option flows)
├── level_universe.py        # Generate critical levels (VWAP, strikes, walls, etc.)
├── barrier_engine.py        # Liquidity physics (VACUUM, WALL, CONSUMED, etc.)
├── tape_engine.py           # Tape momentum (imbalance, velocity, sweep detection)
├── fuel_engine.py           # Dealer gamma impulse (AMPLIFY vs DAMPEN)
├── score_engine.py          # Composite score + trigger state machine
├── smoothing.py             # EWMA smoothers for stable output
├── room_to_run.py           # Runway analysis (distance to next obstacle)
├── level_signal_service.py  # Main orchestrator (produces WS payload)
├── flow_aggregator.py       # (Legacy) Option flow aggregation
├── greek_enricher.py        # (Legacy) Greeks cache
├── strike_manager.py        # (Legacy) Strike selection
└── service.py               # (Legacy) Service wrapper
```

**Note**: `flow_aggregator.py`, `greek_enricher.py`, `strike_manager.py`, and `service.py` are legacy components from the previous iteration. The new v1 architecture uses `market_state.py` as the unified state backbone.

---

## Module Purpose & Physics Model

### Problem Statement

For each critical level \(L\) (e.g., SPY $687.00), continuously decide:

- **BREAK THROUGH**: Level fails; price crosses and runs.
- **REJECT / BOUNCE**: Level holds; price reverses away.
- **Direction-agnostic**: Support tests (approach from above) and resistance tests (approach from below).

### Physics Framework (GM.md + CD.md Synthesis)

The model decomposes the decision into three independent dimensions:

1. **Barrier / Liquidity (CAN it move?)**  
   - Is displayed liquidity **evaporating** (VACUUM) or **replenishing** (WALL/ABSORPTION)?  
   - Computed from ES MBP-10 depth changes and trades.

2. **Tape / Momentum (IS it moving?)**  
   - Is tape aggression **confirming** the direction into the level?  
   - Computed from ES trades (imbalance, velocity, sweep detection).

3. **Fuel / Hedging (WILL it move?)**  
   - Are dealers likely to **amplify** or **dampen** the move (gamma regime at/near level)?  
   - Computed from SPY option trades + greeks (net dealer gamma).

These three dimensions combine into a **Break Score** (0–100) via weighted sum:

```
S = w_L * S_L + w_H * S_H + w_T * S_T
```

Default weights: `w_L=0.45`, `w_H=0.35`, `w_T=0.20` (see `CONFIG`).

---

## Data Flow & Dependencies

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Ingest Layer (backend/src/ingestor/)                           │
│  - DBN Ingestor: ES MBP-10 + trades from Databento files       │
│  - Stream Ingestor: SPY options from Polygon/Massive WS         │
│  - Replay Engine: Merged Bronze Parquet + DBN replay           │
└─────────────────────┬───────────────────────────────────────────┘
                      │ (event stream)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  MarketState (market_state.py)                                  │
│  - ES MBP-10 ring buffer (60-120s window)                      │
│  - ES trades ring buffer                                        │
│  - SPY option flow aggregates (net dealer gamma by strike)     │
│  - Price converter (ES ↔ SPY)                                  │
│  - Accessors: get_spot(), get_bid_ask(), get_vwap(), etc.     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Level Signal Service (level_signal_service.py)                 │
│  - Generate level universe                                      │
│  - For each active level:                                       │
│    ├─ BarrierEngine → BarrierMetrics                           │
│    ├─ TapeEngine    → TapeMetrics                              │
│    ├─ FuelEngine    → FuelMetrics                              │
│    ├─ ScoreEngine   → CompositeScore + Signal                  │
│    ├─ Smoothing     → Smooth scores/metrics                    │
│    └─ RoomToRun     → Runway (distance to next obstacle)       │
│  - Build WS payload (§6.4 of PLAN.md)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Gateway (backend/src/gateway/)                                 │
│  - WebSocket broadcaster (250ms cadence)                        │
│  - Publishes level signals to frontend                          │
└─────────────────────────────────────────────────────────────────┘
```

### Dependency Graph (Modules)

```
MarketState (C)
    ↓ (consumed by)
    ├─ LevelUniverse (F)
    ├─ BarrierEngine (D)
    ├─ TapeEngine (D)
    ├─ FuelEngine (E)
    └─ ScoreEngine (G)

Config (A) ← consumed by all engines

LevelSignalService (G) orchestrates:
    ├─ LevelUniverse
    ├─ BarrierEngine
    ├─ TapeEngine
    ├─ FuelEngine
    ├─ ScoreEngine
    ├─ Smoothing
    └─ RoomToRun (F)
```

**Agent Assignments (from PLAN.md §12)**:
- **Agent A**: Shared contracts + config ✅
- **Agent C**: MarketState + ring buffers ✅
- **Agent D**: Barrier + Tape engines ✅
- **Agent E**: Fuel engine ✅
- **Agent F**: Level universe + room-to-run ✅
- **Agent G**: Scoring + smoothing + orchestrator ✅

---

## Core Components

### 1. MarketState (`market_state.py`)

**Purpose**: Central state store for all market data.

**Key Responsibilities**:
- Store latest ES MBP-10 snapshot and ring buffer
- Store ES trades ring buffer (timestamped, with aggressor)
- Store SPY option flow aggregates (net dealer gamma by strike/right/exp)
- Price conversion (ES ↔ SPY)
- Provide window queries: `get_es_trades_in_window()`, `get_es_trades_near_level()`, etc.

**Critical Accessors**:
```python
# SPY-equivalent (converted from ES)
get_spot() -> float                    # Current SPY spot from ES trade
get_bid_ask() -> (float, float)        # (bid, ask) in SPY terms
get_vwap() -> float                    # Session VWAP in SPY terms

# Raw ES accessors (for engines)
get_es_spot() -> float
get_es_bid_ask() -> (float, float)
get_es_mbp10_snapshot() -> MBP10
get_es_trades_in_window(ts_now_ns, window_seconds) -> List[TimestampedESTrade]
get_es_trades_near_level(ts_now_ns, window_seconds, level_price, band_dollars) -> List[...]

# Option flows
get_option_flows_near_level(level_price, strike_range, exp_date_filter) -> List[OptionFlowAggregate]
```

**Thread Safety**: Not thread-safe by default. Designed for single event loop usage.

**Implementation Note**: `MarketState` uses `RingBuffer` class for efficient time-windowed storage with automatic cleanup.

---

### 2. LevelUniverse (`level_universe.py`)

**Purpose**: Generate the active set of critical levels to monitor.

**Level Sources (v1)**:
- **VWAP**: Session VWAP (from ES, converted to SPY)
- **Round numbers**: Every $1 near spot (configurable spacing)
- **Option strikes**: Active strikes from option flows (within `STRIKE_RANGE`)
- **Flow-derived walls**: Call wall / Put wall (strikes with max |dealer gamma|)
- **Session high/low**: From ES trades
- **User hotzones**: Manual override levels

**Key Method**:
```python
get_levels(market_state: MarketState) -> List[Level]
```

**Level Object**:
```python
@dataclass
class Level:
    id: str                  # Stable identifier (e.g., "VWAP", "STRIKE_687", "ROUND_690")
    price: float             # Level price in SPY dollars
    kind: LevelKind          # VWAP | STRIKE | ROUND | CALL_WALL | PUT_WALL | ...
    metadata: Optional[dict] # Additional info (e.g., net_dealer_gamma for walls)
```

**Direction Context**: Per §3.2 of PLAN.md:
- If `spot > L`: Level is **SUPPORT** (approach from above) → break direction = **DOWN**
- If `spot < L`: Level is **RESISTANCE** (approach from below) → break direction = **UP**

---

### 3. BarrierEngine (`barrier_engine.py`)

**Purpose**: Compute liquidity state at a level using ES MBP-10 + trades.

**Physics Implementation**: Per §5.1.1 of PLAN.md (L2/MBP-10 + Trades inference):

1. **Track depth changes** across MBP-10 levels in a zone around the level (±N ticks).
2. **Infer FILLED vs PULLED** by comparing depth lost to passive volume (trades that hit the defending side).
3. **Classify state** based on replenishment ratio and delta liquidity.

**State Enumeration**:
```python
class BarrierState(Enum):
    VACUUM      # Liquidity pulled without fills (easy break)
    WALL        # Strong replenishment (reject likely)
    ABSORPTION  # Liquidity consumed but replenished (contested)
    CONSUMED    # Liquidity eaten faster than replenished (contested → break)
    WEAK        # Defending size below baseline (vulnerable)
    NEUTRAL     # Normal state
```

**Key Method**:
```python
compute_barrier_state(
    level_price: float,          # SPY level (e.g., 687.0)
    direction: Direction,         # SUPPORT or RESISTANCE
    market_state: MarketState
) -> BarrierMetrics
```

**Output**:
```python
@dataclass
class BarrierMetrics:
    state: BarrierState
    delta_liq: float           # Net liquidity change (added - canceled - filled)
    replenishment_ratio: float # R = added / (canceled + filled + ε)
    added_size: float
    canceled_size: float
    filled_size: float
    defending_quote: dict      # {price, size} - best defending level
    confidence: float          # 0-1, based on sample size
    churn: float               # gross_added + gross_removed (activity measure)
    depth_in_zone: int         # Total depth in monitoring zone
```

**Key Algorithms**:
- **Depth flow computation**: Iterate through MBP-10 history, compute `Δdepth = depth(t1) - depth(t0)`.
- **Passive volume**: Count ES trades where aggressor matches defending side (BID → SELL aggressor, ASK → BUY aggressor).
- **Inference**: If `Δdepth < 0`, `filled = min(depth_lost, V_passive)`, `pulled = max(0, depth_lost - V_passive)`.

**Price Conversion**: Input `level_price` is in SPY terms. Engine converts to ES internally for queries.

**Tunable Parameters** (from `CONFIG`):
- `W_b`: Window size (default: 10s)
- `BARRIER_ZONE_TICKS`: Zone size around level (default: ±2 ES ticks = ±$0.50)
- `R_vac`: VACUUM threshold (default: 0.3)
- `R_wall`: WALL threshold (default: 1.5)
- `F_thresh`: Delta liquidity threshold (default: 100 ES contracts)

---

### 4. TapeEngine (`tape_engine.py`)

**Purpose**: Compute tape momentum and aggression near a level using ES trades.

**Metrics Computed**:

1. **Imbalance**: Buy vs sell aggression in price band around level.
   ```
   I = (buy_vol - sell_vol) / (buy_vol + sell_vol + ε)
   ```
   Range: [-1, +1]

2. **Velocity**: Price slope over time (linear regression).
   ```
   v = slope(price(t))  # in $/sec
   ```

3. **Sweep Detection**: Clustered aggressive prints (rapid sequence, large notional, consistent direction).

**Key Method**:
```python
compute_tape_state(
    level_price: float,          # SPY level
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
    velocity: float   # $/sec (positive = rising, negative = falling)
    sweep: SweepDetection
    confidence: float # 0-1
```

**Aggressor Classification** (from `event_types.py`):
- **BUY** (+1): Trade at/above ask (lifted ask)
- **SELL** (-1): Trade at/below bid (hit bid)
- **MID** (0): Trade between bid/ask (ignored for imbalance)

**Sweep Detection Logic**:
1. Group trades into clusters (max gap ≤ `SWEEP_MAX_GAP_MS`, default: 100ms).
2. Check cluster for:
   - Total notional ≥ `SWEEP_MIN_NOTIONAL` (default: $500k)
   - Consistent direction (all BUY or all SELL)
3. Return largest sweep if multiple found.

**Tunable Parameters** (from `CONFIG`):
- `W_t`: Imbalance window (default: 5s)
- `W_v`: Velocity window (default: 3s)
- `TAPE_BAND`: Price band around level (default: $0.10 SPY → ~$1.00 ES)
- `SWEEP_MIN_NOTIONAL`: Sweep threshold (default: $500k)
- `SWEEP_MAX_GAP_MS`: Max gap in cluster (default: 100ms)

---

### 5. FuelEngine (`fuel_engine.py`)

**Purpose**: Estimate whether dealers will **amplify** or **dampen** a move near a level based on net dealer gamma.

**Key Insight**: Per §5.3 of PLAN.md:
- **Customer buys option** → dealer sells gamma → `net_gamma_flow` is **NEGATIVE** → dealers **SHORT gamma** → must hedge by chasing → **AMPLIFY**.
- **Customer sells option** → dealer buys gamma → `net_gamma_flow` is **POSITIVE** → dealers **LONG gamma** → fade moves → **DAMPEN**.

**Gamma Transfer Computation** (per trade):
```python
customer_sign = trade.aggressor.value  # +1 BUY, -1 SELL, 0 MID
gamma_notional = customer_sign * trade.size * gamma * 100  # contract multiplier
dealer_gamma_change = -gamma_notional  # dealer takes opposite side
```

**Key Method**:
```python
compute_fuel_state(
    level_price: float,
    market_state: MarketState,
    exp_date_filter: Optional[str] = None  # e.g., "2025-12-16" for 0DTE
) -> FuelMetrics
```

**Output**:
```python
@dataclass
class FuelMetrics:
    effect: FuelEffect             # AMPLIFY | DAMPEN | NEUTRAL
    net_dealer_gamma: float        # Negative = dealers short gamma
    call_wall: Optional[GammaWall]
    put_wall: Optional[GammaWall]
    hvl: Optional[float]           # High Volatility Line (gamma flip level)
    confidence: float
    gamma_by_strike: dict          # {strike: net_gamma} for debugging
```

**Wall Identification** (flow-based, per §5.3):
- **Call Wall**: Strike with most negative dealer gamma for calls (highest customer demand).
- **Put Wall**: Strike with most negative dealer gamma for puts.

**Tunable Parameters** (from `CONFIG`):
- `W_g`: Option flow window (default: 60s)
- `W_wall`: Wall lookback window (default: 300s = 5 min)
- `FUEL_STRIKE_RANGE`: Strike range around level (default: ±$2.0)
- `gamma_threshold`: Min |gamma| for non-neutral (default: 10,000)
- `wall_strength_threshold`: Min gamma for strong wall (default: 50,000)

---

### 6. ScoreEngine (`score_engine.py`)

**Purpose**: Combine Barrier, Tape, and Fuel states into composite break score + discrete signal triggers.

**Component Scores** (per §5.4.1 of PLAN.md):

1. **Liquidity Score (S_L)**:
   ```
   VACUUM:         100
   WEAK:            75
   CONSUMED:        60 (if delta_liq << 0), else 50
   NEUTRAL:         50
   WALL/ABSORPTION:  0
   ```

2. **Hedge Score (S_H)**:
   ```
   AMPLIFY (in break direction): 100
   NEUTRAL:                       50
   DAMPEN:                         0
   ```

3. **Tape Score (S_T)**:
   ```
   Sweep detected (in break direction): 100
   Otherwise: 0-50 based on velocity + imbalance alignment
   ```

**Composite Score**:
```python
S = w_L * S_L + w_H * S_H + w_T * S_T
```
Clamped to [0, 100].

**Signal Triggers** (with hysteresis, per §5.4.3):
- **BREAK_IMMINENT**: Score > 80 sustained for `TRIGGER_HOLD_TIME` (default: 3s)
- **REJECT**: Score < 20 while price within `TOUCH_BAND` of level (default: $0.05)
- **CONTESTED**: Mid scores (30–70) with CONSUMED state and high tape activity
- **NEUTRAL**: Default state

**Key Method**:
```python
compute_score(
    barrier_metrics: BarrierMetrics,
    tape_metrics: TapeMetrics,
    fuel_metrics: FuelMetrics,
    break_direction: str,  # 'UP' or 'DOWN'
    ts_ns: int,
    distance_to_level: float
) -> CompositeScore
```

**Output**:
```python
@dataclass
class CompositeScore:
    raw_score: float
    component_scores: ComponentScores  # S_L, S_H, S_T
    signal: Signal                     # BREAK | REJECT | CONTESTED | NEUTRAL
    confidence: Confidence             # HIGH | MEDIUM | LOW
```

**Trigger State Machine**: Internal `TriggerStateMachine` class maintains state for hysteresis. Call `reset()` to clear state (e.g., on session boundary).

**Tunable Parameters** (from `CONFIG`):
- `w_L`, `w_H`, `w_T`: Component weights (default: 0.45, 0.35, 0.20)
- `BREAK_SCORE_THRESHOLD`: Break trigger (default: 80)
- `REJECT_SCORE_THRESHOLD`: Reject trigger (default: 20)
- `TRIGGER_HOLD_TIME`: Sustained time required (default: 3s)

---

### 7. Smoothing (`smoothing.py`)

**Purpose**: Apply EWMA and optional robust smoothing for stable signal outputs.

**Why Smoothing?** Raw scores/metrics can flicker due to micro-structure noise. Per §5.6 of PLAN.md, we smooth:
- `break_score`
- `delta_liq` (barrier)
- `replenishment_ratio` (barrier)
- `velocity` (tape)
- `net_dealer_gamma` (fuel)

**EWMA Formula**:
```
x_smooth(t) = α * x(t) + (1 - α) * x_smooth(t-1)

where α = 1 - exp(-Δt / τ)
```

`τ` is the half-life parameter (time constant for exponential decay).

**Key Classes**:

1. **`EWMA`**: Basic exponential weighted moving average.
   ```python
   smoother = EWMA(tau=2.0)
   smoothed_value = smoother.update(value, ts_ns)
   ```

2. **`RobustRollingMedian`**: Median filter for outlier rejection.
   ```python
   smoother = RobustRollingMedian(window_size=10)
   smoothed_value = smoother.update(value)
   ```

3. **`HybridSmoother`**: EWMA with optional robust pre-filtering.
   ```python
   smoother = HybridSmoother(tau=2.0, use_robust=True, robust_window=5)
   smoothed_value = smoother.update(value, ts_ns)
   ```

4. **`SmootherSet`**: Collection of smoothers for all level signals.
   ```python
   smoothers = SmootherSet(config=CONFIG)
   score_smooth = smoothers.update_score(raw_score, ts_ns)
   delta_liq_smooth = smoothers.update_delta_liq(delta_liq, ts_ns)
   # ... etc.
   ```

**Tunable Parameters** (from `CONFIG`):
- `tau_score`: Break score smoothing (default: 2.0s)
- `tau_velocity`: Tape velocity smoothing (default: 1.5s)
- `tau_delta_liq`: Barrier delta_liq smoothing (default: 3.0s)
- `tau_replenish`: Replenishment ratio smoothing (default: 3.0s)
- `tau_dealer_gamma`: Net dealer gamma smoothing (default: 5.0s)

**Usage Pattern**: `LevelSignalService` maintains one `SmootherSet` per level (keyed by `level.id`).

---

### 8. RoomToRun (`room_to_run.py`)

**Purpose**: Compute runway (distance to next obstacle) after a break/reject signal.

**Key Method**:
```python
compute_runway(
    current_level: Level,
    direction: Direction,      # UP or DOWN
    all_levels: List[Level],
    spot: float
) -> Runway
```

**Output**:
```python
@dataclass
class Runway:
    direction: Direction           # UP or DOWN
    distance: float                # Dollars to next obstacle
    next_obstacle: Optional[Level]
    quality: RunwayQuality         # CLEAR or OBSTRUCTED
    intermediate_levels: List[Level]
```

**Runway Quality** (per §5.5 of PLAN.md):
- **CLEAR**: No strong obstacles (walls, session high/low, etc.) between current and next.
- **OBSTRUCTED**: At least one strong obstacle in path.

**Strong Obstacle Kinds**:
```python
STRONG_OBSTACLE_KINDS = {
    LevelKind.CALL_WALL,
    LevelKind.PUT_WALL,
    LevelKind.SESSION_HIGH,
    LevelKind.SESSION_LOW,
    LevelKind.VWAP,
    LevelKind.GAMMA_FLIP
}
```

**Helper Functions**:
```python
get_break_direction(level_price, spot) -> Direction
get_reject_direction(level_price, spot) -> Direction
```

---

### 9. LevelSignalService (`level_signal_service.py`)

**Purpose**: Main orchestrator that integrates all engines and produces the WebSocket payload.

**Initialization**:
```python
service = LevelSignalService(
    market_state=market_state,
    user_hotzones=[687.0, 690.0],  # optional manual levels
    config=CONFIG
)
```

**Main Method**:
```python
compute_level_signals() -> Dict[str, Any]
```

Returns payload per §6.4 of PLAN.md:
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
      "break_score_raw": 88,
      "break_score_smooth": 81,
      "signal": "BREAK",
      "confidence": "HIGH",
      "barrier": { ... },
      "tape": { ... },
      "fuel": { ... },
      "runway": { ... },
      "note": "Vacuum; dealers chase; sweep confirms"
    }
  ]
}
```

**Execution Flow** (per snap tick):
1. Get spot, bid/ask from `MarketState`.
2. Generate level universe via `LevelUniverse.get_levels()`.
3. Filter to levels within `MONITOR_BAND` (default: $0.50).
4. For each active level:
   - Compute `BarrierMetrics`, `TapeMetrics`, `FuelMetrics`.
   - Compute `CompositeScore` + signal.
   - Smooth all metrics via `SmootherSet`.
   - Compute `Runway`.
   - Build `LevelSignal` output.
5. Sort levels by distance (nearest first).
6. Build payload dict.

**Snap Cadence**: Called by gateway broadcaster every `SNAP_INTERVAL_MS` (default: 250ms).

---

## Interfaces & Contracts

### Event Types (from `src/common/event_types.py`)

All events carry:
- `ts_event_ns`: Event time (from vendor) in Unix nanoseconds UTC
- `ts_recv_ns`: Receive time (by our system) in Unix nanoseconds UTC
- `source`: EventSource enum (MASSIVE_WS, POLYGON_WS, REPLAY, etc.)

**Critical Types for Core Module**:

1. **`FuturesTrade`**: ES time-and-sales
   ```python
   @dataclass
   class FuturesTrade:
       ts_event_ns: int
       ts_recv_ns: int
       source: EventSource
       symbol: str          # "ES"
       price: float
       size: int
       aggressor: Aggressor # BUY, SELL, MID
   ```

2. **`MBP10`**: ES Market-by-Price L2 (top 10 levels)
   ```python
   @dataclass
   class MBP10:
       ts_event_ns: int
       ts_recv_ns: int
       source: EventSource
       symbol: str
       levels: List[BidAskLevel]  # 10 levels
       is_snapshot: bool
   ```

3. **`OptionTrade`**: SPY option trades (for fuel engine)
   ```python
   @dataclass
   class OptionTrade:
       ts_event_ns: int
       ts_recv_ns: int
       source: EventSource
       underlying: str      # "SPY"
       option_symbol: str
       exp_date: str
       strike: float
       right: str           # 'C' or 'P'
       price: float
       size: int
       aggressor: Aggressor
   ```

### MarketState Update Methods

Engines **DO NOT** call these directly. Only the ingest layer calls these:

```python
# ES updates
market_state.update_es_mbp10(mbp: MBP10)
market_state.update_es_trade(trade: FuturesTrade)

# SPY option updates
market_state.update_option_trade(trade: OptionTrade, delta: float, gamma: float)

# Legacy flow aggregator integration
market_state.integrate_flow_snapshot(flow_snapshot: Dict[str, Any])
```

### Engine Query Methods

Engines call these to retrieve data:

```python
# ES MBP-10
market_state.get_es_mbp10_snapshot() -> Optional[MBP10]
market_state.get_es_mbp10_in_window(ts_now_ns, window_seconds) -> List[MBP10]

# ES trades
market_state.get_es_trades_in_window(ts_now_ns, window_seconds, price_band=None) -> List[TimestampedESTrade]
market_state.get_es_trades_near_level(ts_now_ns, window_seconds, level_price, band_dollars) -> List[TimestampedESTrade]

# Option flows
market_state.get_option_flows_near_level(level_price, strike_range, exp_date_filter=None) -> List[OptionFlowAggregate]

# Accessors
market_state.get_spot() -> Optional[float]  # SPY-equivalent
market_state.get_bid_ask() -> Optional[Tuple[float, float]]  # SPY-equivalent
market_state.get_vwap() -> Optional[float]  # SPY-equivalent
market_state.get_current_ts_ns() -> int
```

---

## Engine Specifications

### Barrier Engine: State Classification Logic

**Inputs**:
- `level_price` (SPY)
- `direction` (SUPPORT or RESISTANCE)
- `market_state`

**Conversion**:
```python
es_level_price = market_state.price_converter.spy_to_es(level_price)
```

**Zone Definition** (ES terms):
```python
zone_low = es_level_price - (BARRIER_ZONE_TICKS * ES_TICK_SIZE)
zone_high = es_level_price + (BARRIER_ZONE_TICKS * ES_TICK_SIZE)
```

**Defending Side**:
- SUPPORT → BID side
- RESISTANCE → ASK side

**Flow Computation** (pseudo-code):
```python
for each (prev_mbp, curr_mbp) in mbp_history:
    prev_depth = sum_zone_depth(prev_mbp, zone, side)
    curr_depth = sum_zone_depth(curr_mbp, zone, side)
    Δdepth = curr_depth - prev_depth
    
    if Δdepth > 0:
        added_size += Δdepth
        gross_added += Δdepth
    elif Δdepth < 0:
        depth_lost = abs(Δdepth)
        gross_removed += depth_lost
        
        # Get passive volume in [prev_ts, curr_ts]
        V_passive = sum(trade.size for trade in trades 
                        if trade in time_window and trade in zone and trade.aggressor matches defending_side)
        
        filled = min(depth_lost, V_passive + ε_fill)
        pulled = max(0, depth_lost - V_passive)
        
        filled_size += filled
        canceled_size += pulled

churn = gross_added + gross_removed
delta_liq = added_size - canceled_size - filled_size
replenishment_ratio = added_size / (canceled_size + filled_size + ε)
```

**Classification** (thresholds from `CONFIG`):
```python
if replenishment_ratio < R_vac and delta_liq < -F_thresh:
    state = VACUUM
elif replenishment_ratio > R_wall and delta_liq > F_thresh:
    state = WALL
elif delta_liq < -F_thresh and filled_size > canceled_size:
    state = CONSUMED
elif delta_liq > 0 and replenishment_ratio > 1.0:
    state = ABSORPTION
elif depth_in_zone < 50:
    state = WEAK
else:
    state = NEUTRAL
```

### Tape Engine: Imbalance & Velocity

**Imbalance** (within `TAPE_BAND` around level, window `W_t`):
```python
buy_vol = sum(trade.size for trade in trades if trade.aggressor == BUY)
sell_vol = sum(trade.size for trade in trades if trade.aggressor == SELL)
imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + ε)
```

**Velocity** (all trades, window `W_v`):
```python
times = [trade.ts_event_ns / 1e9 for trade in trades]
prices = [trade.price for trade in trades]
slope, _ = np.polyfit(times - times[0], prices, 1)
velocity = slope  # $/sec
```

**Sweep Detection**:
1. Cluster trades by time gaps (≤ `SWEEP_MAX_GAP_MS`).
2. For each cluster:
   - Check total notional ≥ `SWEEP_MIN_NOTIONAL`.
   - Check consistent direction (all BUY or all SELL).
3. Return best (largest notional) sweep.

### Fuel Engine: Net Dealer Gamma

**Per-Trade Gamma Transfer**:
```python
customer_sign = trade.aggressor.value  # +1 BUY, -1 SELL
gamma_notional = customer_sign * trade.size * gamma * 100
dealer_gamma_change = -gamma_notional  # dealer opposite side
```

**Aggregate Near Level**:
```python
option_flows = market_state.get_option_flows_near_level(level_price, FUEL_STRIKE_RANGE)
net_dealer_gamma = sum(flow.net_gamma_flow for flow in option_flows)
```

**Effect Classification**:
```python
if net_dealer_gamma < -gamma_threshold:
    effect = AMPLIFY
elif net_dealer_gamma > gamma_threshold:
    effect = DAMPEN
else:
    effect = NEUTRAL
```

**Wall Identification** (flow-based):
- **Call Wall**: `argmin(net_gamma_flow)` for calls above spot (most negative = highest customer demand).
- **Put Wall**: `argmin(net_gamma_flow)` for puts below spot.

### Score Engine: Composite Formula

**Component Score Mapping**:

| Barrier State | S_L | Tape Condition | S_T | Fuel Effect | S_H |
|--------------|-----|----------------|-----|-------------|-----|
| VACUUM       | 100 | Sweep (aligned)| 100 | AMPLIFY     | 100 |
| WEAK         | 75  | Velocity+Imb   | 0-50| NEUTRAL     | 50  |
| CONSUMED     | 60* | No sweep       | 0-25| DAMPEN      | 0   |
| NEUTRAL      | 50  |                |     |             |     |
| ABSORPTION   | 0   |                |     |             |     |
| WALL         | 0   |                |     |             |     |

*60 if `delta_liq < -F_thresh`, else 50.

**Composite**:
```python
S = w_L * S_L + w_H * S_H + w_T * S_T
S = clamp(S, 0, 100)
```

**Trigger State Machine** (hysteresis):

```python
# BREAK IMMINENT trigger
if score > BREAK_SCORE_THRESHOLD:
    if high_score_sustained >= TRIGGER_HOLD_TIME:
        signal = BREAK_IMMINENT
else:
    reset high_score_timer

# REJECT trigger
if score < REJECT_SCORE_THRESHOLD and distance <= TOUCH_BAND:
    if low_score_sustained >= TRIGGER_HOLD_TIME:
        signal = REJECT
else:
    reset low_score_timer

# CONTESTED trigger (no hysteresis)
if 30 <= score <= 70 and barrier_state == CONSUMED and tape_activity > 50000:
    signal = CONTESTED

# Default
signal = NEUTRAL
```

---

## Price Conversion (ES ↔ SPY)

### Rationale

**Levels are SPY prices** (because we trade SPY options).  
**Liquidity source is ES futures** (superior depth visibility via MBP-10).

### PriceConverter (`src/common/price_converter.py`)

```python
class PriceConverter:
    def __init__(self, initial_ratio: float = 10.0):
        self.ratio = initial_ratio  # ES ≈ SPY × 10
    
    def spy_to_es(self, spy_price: float) -> float:
        return spy_price * self.ratio
    
    def es_to_spy(self, es_price: float) -> float:
        return es_price / self.ratio
    
    def update_es_price(self, es_price: float):
        # Can update ratio dynamically if needed
        pass
```

**Usage Pattern**:
1. User/system specifies level in SPY terms: `level_price = 687.0`
2. Engine converts to ES for queries: `es_level = converter.spy_to_es(687.0)  # 6870.0`
3. Query ES depth/trades at `es_level ± zone`
4. Convert output prices back to SPY for display: `defending_quote_spy = converter.es_to_spy(defending_quote_es)`

**Output Guarantees**: All outputs (defending quotes, spot, bid/ask) are in SPY terms for consistency with frontend and option strikes.

---

## Configuration

All tunable parameters live in `src/common/config.py` as a `Config` dataclass.

**Critical Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `W_b` | float | 10.0 | Barrier engine window (seconds) |
| `W_t` | float | 5.0 | Tape imbalance window (seconds) |
| `W_g` | float | 60.0 | Fuel option flow window (seconds) |
| `W_v` | float | 3.0 | Velocity window (seconds) |
| `W_wall` | float | 300.0 | Wall lookback (seconds) |
| `MONITOR_BAND` | float | 0.50 | Compute signals if \|spot - L\| ≤ $0.50 |
| `TOUCH_BAND` | float | 0.05 | Tight band for "touching level" |
| `BARRIER_ZONE_TICKS` | int | 2 | ES ticks around level for zone depth |
| `R_vac` | float | 0.3 | VACUUM threshold |
| `R_wall` | float | 1.5 | WALL threshold |
| `F_thresh` | int | 100 | Delta liquidity threshold (ES contracts) |
| `TAPE_BAND` | float | 0.10 | Price band for tape imbalance (SPY $) |
| `SWEEP_MIN_NOTIONAL` | float | 500,000 | Sweep notional threshold |
| `FUEL_STRIKE_RANGE` | float | 2.0 | Strike range around level (SPY $) |
| `w_L`, `w_H`, `w_T` | float | 0.45, 0.35, 0.20 | Composite score weights |
| `BREAK_SCORE_THRESHOLD` | float | 80.0 | Break trigger score |
| `REJECT_SCORE_THRESHOLD` | float | 20.0 | Reject trigger score |
| `TRIGGER_HOLD_TIME` | float | 3.0 | Sustained time for trigger (seconds) |
| `tau_score` | float | 2.0 | Break score EWMA half-life (seconds) |
| `tau_velocity` | float | 1.5 | Velocity EWMA half-life (seconds) |
| `tau_delta_liq` | float | 3.0 | Delta liq EWMA half-life (seconds) |
| `tau_replenish` | float | 3.0 | Replenishment EWMA half-life (seconds) |
| `tau_dealer_gamma` | float | 5.0 | Dealer gamma EWMA half-life (seconds) |
| `SNAP_INTERVAL_MS` | int | 250 | Snap tick cadence (ms) |
| `ROUND_LEVELS_SPACING` | float | 1.0 | Round level spacing (SPY $) |
| `STRIKE_RANGE` | float | 5.0 | Strike monitoring range (SPY $) |

**Accessing Config**:
```python
from src.common.config import CONFIG

barrier_window = CONFIG.W_b
snap_interval = CONFIG.SNAP_INTERVAL_MS
```

**Overriding Config** (for testing):
```python
test_config = Config(
    W_b=5.0,
    MONITOR_BAND=1.0,
    w_L=0.5,
    w_H=0.3,
    w_T=0.2
)
engine = BarrierEngine(config=test_config)
```

---

## Adding New Engines

To add a new physics dimension (e.g., "Spread Engine" for bid-ask width):

### Step 1: Define Output Dataclass

```python
# backend/src/core/spread_engine.py
from dataclasses import dataclass

@dataclass
class SpreadMetrics:
    spread_bps: float       # Bid-ask spread in basis points
    spread_percentile: float # vs recent distribution
    state: str              # "TIGHT" | "WIDE" | "NORMAL"
    confidence: float
```

### Step 2: Implement Engine Class

```python
class SpreadEngine:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.window_seconds = self.config.W_spread  # add to Config
    
    def compute_spread_state(
        self,
        level_price: float,
        market_state: MarketState
    ) -> SpreadMetrics:
        # Query MBP-10 for bid-ask width
        mbp = market_state.get_es_mbp10_snapshot()
        if mbp is None or not mbp.levels:
            return self._neutral_metrics()
        
        best = mbp.levels[0]
        spread = best.ask_px - best.bid_px
        spread_bps = (spread / best.bid_px) * 10000
        
        # Compute percentile vs rolling history
        # ... (implement windowed spread distribution)
        
        return SpreadMetrics(
            spread_bps=spread_bps,
            spread_percentile=0.5,  # placeholder
            state="NORMAL",
            confidence=1.0
        )
    
    def _neutral_metrics(self) -> SpreadMetrics:
        return SpreadMetrics(
            spread_bps=0.0,
            spread_percentile=0.5,
            state="NORMAL",
            confidence=0.0
        )
```

### Step 3: Integrate into ScoreEngine

```python
# backend/src/core/score_engine.py

# Add component score method
def _compute_spread_score(self, spread: SpreadMetrics) -> float:
    if spread.state == "TIGHT":
        return 75.0  # Easier to move
    elif spread.state == "WIDE":
        return 25.0  # Harder to move
    else:
        return 50.0

# Update composite score
def compute_score(
    self,
    barrier_metrics: BarrierMetrics,
    tape_metrics: TapeMetrics,
    fuel_metrics: FuelMetrics,
    spread_metrics: SpreadMetrics,  # NEW
    ...
) -> CompositeScore:
    S_L = self._compute_liquidity_score(barrier_metrics)
    S_H = self._compute_hedge_score(fuel_metrics, break_direction)
    S_T = self._compute_tape_score(tape_metrics, break_direction)
    S_S = self._compute_spread_score(spread_metrics)  # NEW
    
    # Add weight to config
    raw_score = (
        self.w_L * S_L + 
        self.w_H * S_H + 
        self.w_T * S_T + 
        self.w_S * S_S  # NEW
    )
    ...
```

### Step 4: Update LevelSignalService

```python
# backend/src/core/level_signal_service.py

# Initialize in __init__
self.spread_engine = SpreadEngine(config=self.config)

# Compute in _compute_level_signal
spread_metrics = self.spread_engine.compute_spread_state(
    level_price=level_price,
    market_state=self.market_state
)

composite_score = self.score_engine.compute_score(
    barrier_metrics=barrier_metrics,
    tape_metrics=tape_metrics,
    fuel_metrics=fuel_metrics,
    spread_metrics=spread_metrics,  # NEW
    ...
)

# Add to payload
level_signal = LevelSignal(
    ...,
    spread={  # NEW
        "spread_bps": spread_metrics.spread_bps,
        "state": spread_metrics.state,
        "confidence": spread_metrics.confidence
    }
)
```

### Step 5: Add Tests

```python
# backend/tests/test_spread_engine.py
from src.core.spread_engine import SpreadEngine, SpreadMetrics
from src.core.market_state import MarketState

def test_spread_engine_tight():
    market_state = MarketState()
    # ... populate with tight spread MBP-10
    
    engine = SpreadEngine()
    metrics = engine.compute_spread_state(687.0, market_state)
    
    assert metrics.state == "TIGHT"
    assert metrics.confidence > 0.5
```

### Step 6: Update Config

```python
# backend/src/common/config.py
@dataclass
class Config:
    ...
    # Spread engine
    W_spread: float = 10.0
    w_S: float = 0.05  # Add spread weight
    
    # Adjust existing weights to sum to 1.0
    w_L: float = 0.43  # was 0.45
    w_H: float = 0.33  # was 0.35
    w_T: float = 0.19  # was 0.20
```

---

## Testing Strategy

### Unit Tests (per engine)

**Location**: `backend/tests/test_*.py`

**Pattern**: Synthetic data to force each state/classification.

**Example** (`test_barrier_engine.py`):
```python
def test_barrier_vacuum():
    """Test VACUUM state: depth pulled without fills."""
    market_state = MarketState()
    
    # Create MBP-10 history with depth drop
    mbp_t0 = create_mbp10(bid_depth=1000, ask_depth=1000)
    mbp_t1 = create_mbp10(bid_depth=100, ask_depth=1000)  # bid depth dropped
    
    # No trades concurrent with depth drop → PULLED
    market_state.update_es_mbp10(mbp_t0)
    market_state.update_es_mbp10(mbp_t1)
    
    engine = BarrierEngine()
    metrics = engine.compute_barrier_state(
        level_price=687.0,
        direction=BarrierDirection.SUPPORT,
        market_state=market_state
    )
    
    assert metrics.state == BarrierState.VACUUM
    assert metrics.canceled_size > metrics.filled_size
    assert metrics.replenishment_ratio < CONFIG.R_vac
```

**Coverage Targets**:
- Barrier: VACUUM, WALL, ABSORPTION, CONSUMED, WEAK, NEUTRAL
- Tape: Positive/negative imbalance, velocity, sweep detection, no sweep
- Fuel: AMPLIFY, DAMPEN, NEUTRAL, call/put wall identification
- Score: Component scores, composite formula, trigger hysteresis
- Smoothing: EWMA convergence, robust median filtering

### Integration Tests

**Location**: `backend/tests/test_*_integration.py`

**Pattern**: End-to-end flow with realistic data.

**Example** (`test_level_signal_integration.py`):
```python
def test_break_signal_integration():
    """Test full pipeline: VACUUM + AMPLIFY + Sweep → BREAK signal."""
    market_state = MarketState()
    service = LevelSignalService(market_state)
    
    # Populate market_state with:
    # - ES MBP-10 showing vacuum (depth pulled)
    # - ES trades showing sweep down
    # - SPY options showing dealers short gamma
    
    payload = service.compute_level_signals()
    
    assert len(payload["levels"]) > 0
    level_687 = [l for l in payload["levels"] if l["id"] == "STRIKE_687"][0]
    assert level_687["signal"] == "BREAK"
    assert level_687["break_score_raw"] > 80
```

### Replay Tests

**Location**: `backend/tests/test_replay_determinism.py`

**Goal**: Ensure deterministic output given same inputs.

```python
def test_replay_determinism():
    """Replay same Bronze data twice → identical level signals."""
    from src.core.unified_replay_engine import UnifiedReplayEngine
    
    # Replay once
    signals_1 = run_replay(date="2025-12-16", time_range="09:30-10:00")
    
    # Replay again
    signals_2 = run_replay(date="2025-12-16", time_range="09:30-10:00")
    
    # Compare outputs
    assert signals_1 == signals_2  # Exact match
```

### Test Fixtures

**Reusable Factories** (`tests/fixtures.py`):
```python
def create_mbp10(
    ts_ns: int = None,
    symbol: str = "ES",
    bid_depth: int = 1000,
    ask_depth: int = 1000,
    bid_px: float = 6870.0,
    ask_px: float = 6870.25
) -> MBP10:
    """Factory for MBP10 events."""
    if ts_ns is None:
        ts_ns = time.time_ns()
    
    levels = [
        BidAskLevel(
            bid_px=bid_px,
            bid_sz=bid_depth,
            ask_px=ask_px,
            ask_sz=ask_depth
        )
    ] + [
        BidAskLevel(0, 0, 0, 0) for _ in range(9)  # Fill remaining levels
    ]
    
    return MBP10(
        ts_event_ns=ts_ns,
        ts_recv_ns=ts_ns,
        source=EventSource.REPLAY,
        symbol=symbol,
        levels=levels,
        is_snapshot=True
    )

def create_futures_trade(
    price: float,
    size: int,
    aggressor: Aggressor = Aggressor.BUY,
    ts_ns: int = None
) -> FuturesTrade:
    """Factory for FuturesTrade events."""
    if ts_ns is None:
        ts_ns = time.time_ns()
    
    return FuturesTrade(
        ts_event_ns=ts_ns,
        ts_recv_ns=ts_ns,
        source=EventSource.REPLAY,
        symbol="ES",
        price=price,
        size=size,
        aggressor=aggressor
    )
```

### Running Tests

```bash
cd backend

# Run all tests
uv run pytest tests/

# Run specific module
uv run pytest tests/test_barrier_engine.py

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src/core --cov-report=term-missing
```

**Current Test Coverage** (from PLAN.md §10):
- **86 tests** total
- Barrier, tape, fuel, score, market_state, price_converter, silver_compactor, wal_manager, run_manifest_manager

---

## Critical Implementation Notes

### 1. Time Units (§0 of PLAN.md)

**Vendor WS timestamps**: Unix milliseconds  
**Internal time**: Unix nanoseconds (UTC)

**Conversion**:
```python
ts_event_ns = vendor_ts_ms * 1_000_000
```

**Timestamp Fields**:
- `ts_event_ns`: Event time (from exchange/vendor)
- `ts_recv_ns`: Receive time (by our system)

**Why nanoseconds?** Databento DBN uses nanosecond timestamps. Standardize on ns internally, convert to ms for WS broadcast.

### 2. ES MBP-10 Event Stream vs Sampling

**Critical**: Per §5.1.1 of PLAN.md, MBP-10 must be treated as an **event stream** (process every update), not coarse sampling.

**Why?** Coarse sampling (e.g., every 250ms) suffers from "churn blindness":
- If cancels and adds occur between samples and net depth is unchanged, the churn is invisible.
- Result: Underestimate activity, misclassify walls as neutral.

**Implementation**:
- Ingest every MBP-10 update into `MarketState.es_mbp10_buffer`.
- Barrier engine iterates through all MBP-10 updates in window to compute gross flows.

### 3. Passive Volume for Fill Inference

**Critical**: Only count **passive fills** on the defending side.

**Logic**:
- **BID defending** (support test): Count SELL-aggressor trades (hit bid).
- **ASK defending** (resistance test): Count BUY-aggressor trades (lift ask).

**Why?** Aggressor-initiated trades are the ones that consume passive liquidity. Crossing the spread is the key signal.

### 4. Price Conversion: ES → SPY

**Critical**: All level prices are in SPY terms. Engines convert internally for ES queries.

**Conversion Factor**: ES ≈ SPY × 10 (dynamic ratio supported via `PriceConverter`).

**Example**:
```python
# User specifies level in SPY
level_price_spy = 687.0

# Engine converts for ES query
level_price_es = market_state.price_converter.spy_to_es(687.0)  # 6870.0

# Query ES depth
zone_low = level_price_es - (2 * 0.25)  # 6869.5
zone_high = level_price_es + (2 * 0.25)  # 6870.5
depth = get_zone_depth(mbp, zone_low, zone_high, 'bid')

# Convert output for display
defending_quote_spy = market_state.price_converter.es_to_spy(defending_quote_es)
```

### 5. Smoothing: Per-Level State

**Critical**: Each level maintains its own `SmootherSet` instance.

**Why?** Levels have independent dynamics. Smoothing state must not leak between levels.

**Implementation**:
```python
# In LevelSignalService.__init__
self.smoothers: Dict[str, SmootherSet] = {}

# In _compute_level_signal
if level.id not in self.smoothers:
    self.smoothers[level.id] = SmootherSet(config=self.config)

smoother = self.smoothers[level.id]
score_smooth = smoother.update_score(raw_score, ts_now_ns)
```

### 6. Trigger Hysteresis

**Critical**: Signals require sustained conditions to avoid flickering.

**Example**: BREAK signal requires `score > 80` sustained for 3 seconds (default `TRIGGER_HOLD_TIME`).

**Implementation**: `TriggerStateMachine` class in `score_engine.py`.

**State Machine Vars**:
- `high_score_since_ns`: Timestamp when score first exceeded 80.
- `low_score_since_ns`: Timestamp when score first dropped below 20.
- `current_signal`: Last emitted signal.

**Reset Policy**: Call `score_engine.reset()` on session boundaries or when market state is cleared.

### 7. Ring Buffer Cleanup

**Critical**: `RingBuffer` must call `cleanup()` before queries to purge old data.

**Implementation**:
```python
def get_window(self, current_ts_ns: int, window_seconds: float) -> List:
    self.cleanup(current_ts_ns)  # Remove items older than max_window_ns
    cutoff_ns = current_ts_ns - int(window_seconds * 1e9)
    return [item for item in self.buffer if item.ts_event_ns >= cutoff_ns]
```

**Why?** Memory-efficient sliding window without manual expiration logic.

### 8. Option Flow: Net Dealer Gamma Convention

**Critical**: `MarketState.option_flows` stores **net DEALER gamma**, not customer gamma.

**Sign Convention**:
- Customer buys option → `net_gamma_flow` is **NEGATIVE** (dealer sold gamma).
- Customer sells option → `net_gamma_flow` is **POSITIVE** (dealer bought gamma).

**Fuel Engine Classification**:
```python
if net_dealer_gamma < -threshold:
    effect = AMPLIFY  # Dealers short gamma → chase moves
elif net_dealer_gamma > threshold:
    effect = DAMPEN   # Dealers long gamma → fade moves
```

### 9. Direction Context (§3.2 of PLAN.md)

**Critical**: Level direction depends on spot position.

| Condition | Level Type | Break Direction | Reject Direction |
|-----------|-----------|-----------------|------------------|
| spot > L  | SUPPORT   | DOWN            | UP               |
| spot < L  | RESISTANCE| UP              | DOWN             |

**Implementation**:
```python
if spot > level_price:
    direction_str = "SUPPORT"
    barrier_direction = BarrierDirection.SUPPORT
    break_dir = "DOWN"
else:
    direction_str = "RESISTANCE"
    barrier_direction = BarrierDirection.RESISTANCE
    break_dir = "UP"
```

### 10. Monitoring Band (§3.3 of PLAN.md)

**Critical**: Only compute full signals if `abs(spot - L) <= MONITOR_BAND`.

**Why?** Far-away levels are not relevant. Avoid wasted computation.

**Default**: `MONITOR_BAND = 0.50` (SPY dollars).

**Implementation**:
```python
def _filter_active_levels(self, levels: List[Level], spot: float) -> List[Level]:
    return [
        level for level in levels
        if abs(spot - level.price) <= self.config.MONITOR_BAND
    ]
```

### 11. ES Tick Size

**ES futures tick size**: $0.25 per tick.

**Used in**:
- Barrier zone calculations: `zone_ticks * 0.25`
- Tick-based distance metrics

### 12. Thread Safety

**MarketState is NOT thread-safe**. Designed for single event loop (asyncio).

**Multi-threading Warning**: If you need to run engines in parallel, clone `MarketState` or use locks.

### 13. Replay Determinism

**Goal**: Same inputs + config → same outputs (within floating-point rounding).

**Requirements**:
1. Event-time ordering (sort by `ts_event_ns`).
2. Deterministic tie-breaking (use `ts_recv_ns`, then `seq`).
3. No random seeds, no system time in computation (use `ts_event_ns` for windows).
4. Fixed config (don't mutate CONFIG during replay).

**Testing**: Use `test_replay_determinism.py` to verify.

---

## Quick Start (for AI Agents)

### Minimal Example: Compute Level Signals

```python
from src.core.market_state import MarketState
from src.core.level_signal_service import LevelSignalService
from src.common.event_types import FuturesTrade, MBP10, Aggressor, EventSource, BidAskLevel
from src.common.config import CONFIG
import time

# Initialize
market_state = MarketState()
service = LevelSignalService(market_state, user_hotzones=[687.0])

# Populate with synthetic data
ts_now_ns = time.time_ns()

# Add ES MBP-10 snapshot
mbp = MBP10(
    ts_event_ns=ts_now_ns,
    ts_recv_ns=ts_now_ns,
    source=EventSource.REPLAY,
    symbol="ES",
    levels=[
        BidAskLevel(bid_px=6870.0, bid_sz=1000, ask_px=6870.25, ask_sz=1000)
    ] + [BidAskLevel(0, 0, 0, 0)] * 9,
    is_snapshot=True
)
market_state.update_es_mbp10(mbp)

# Add ES trade
trade = FuturesTrade(
    ts_event_ns=ts_now_ns,
    ts_recv_ns=ts_now_ns,
    source=EventSource.REPLAY,
    symbol="ES",
    price=6870.0,
    size=10,
    aggressor=Aggressor.BUY
)
market_state.update_es_trade(trade)

# Compute level signals
payload = service.compute_level_signals()

# Inspect
print(payload["spy"])
print(f"Found {len(payload['levels'])} active levels")
for level in payload["levels"]:
    print(f"  {level['id']}: score={level['break_score_raw']:.1f}, signal={level['signal']}")
```

### Minimal Example: Single Engine

```python
from src.core.barrier_engine import BarrierEngine, Direction as BarrierDirection
from src.core.market_state import MarketState
from src.common.event_types import MBP10, FuturesTrade, Aggressor, EventSource, BidAskLevel
import time

# Initialize
market_state = MarketState()
engine = BarrierEngine()

# Populate with ES data
ts = time.time_ns()

# Initial snapshot
mbp_t0 = MBP10(
    ts_event_ns=ts,
    ts_recv_ns=ts,
    source=EventSource.REPLAY,
    symbol="ES",
    levels=[BidAskLevel(6870.0, 1000, 6870.25, 1000)] + [BidAskLevel(0, 0, 0, 0)] * 9,
    is_snapshot=True
)
market_state.update_es_mbp10(mbp_t0)

# Depth drop (no fills)
mbp_t1 = MBP10(
    ts_event_ns=ts + int(1e9),  # +1 second
    ts_recv_ns=ts + int(1e9),
    source=EventSource.REPLAY,
    symbol="ES",
    levels=[BidAskLevel(6870.0, 100, 6870.25, 1000)] + [BidAskLevel(0, 0, 0, 0)] * 9,
    is_snapshot=True
)
market_state.update_es_mbp10(mbp_t1)

# Compute barrier state
metrics = engine.compute_barrier_state(
    level_price=687.0,  # SPY level
    direction=BarrierDirection.SUPPORT,
    market_state=market_state
)

print(f"State: {metrics.state}")
print(f"Delta liq: {metrics.delta_liq}")
print(f"Replenishment ratio: {metrics.replenishment_ratio:.2f}")
print(f"Canceled: {metrics.canceled_size}, Filled: {metrics.filled_size}")
```

---

## References

- **PLAN.md §5**: Core Engines (physics specifications)
- **PLAN.md §12**: Parallel Agent Assignments (ownership map)
- **PLAN.md §2**: Canonical Message Envelope & Dataset Contracts
- **PLAN.md §9**: Configuration (single source of truth)
- **CD.md**: State machine and decision logic (VACUUM, WALL, CONSUMED)
- **GM.md**: 3-module decomposition + composite score

---

## Changelog

| Version | Date | Agent | Changes |
|---------|------|-------|---------|
| 1.0 | 2025-12-22 | AI Agent | Initial README for core module |

---

**End of Core Module README**

For questions or clarifications, consult the PLAN.md in the project root or examine the test suite in `backend/tests/`.

