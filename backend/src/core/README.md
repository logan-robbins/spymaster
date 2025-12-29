# Core Module

**Role**: Break/bounce physics engines and signal generation  
**Audience**: Backend developers working on signal logic  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Implements real-time physics-based classification of whether price levels will BREAK (fail) or REJECT (hold). Operates on fixed snap cadence (~250ms) while ingesting events continuously.

**Physics model**: Decomposes decision into three independent dimensions:
1. **Barrier** (Liquidity): Is displayed liquidity evaporating or replenishing?
2. **Tape** (Momentum): Is tape aggression confirming the direction?
3. **Fuel** (Hedging): Will dealers amplify or dampen the move?

**Output**: Level signals payload published to NATS (`levels.signals`) for Gateway broadcast and Gold storage.

---

## Architecture Principles

1. **Event-driven ingestion, time-bucketed scoring**: Ingest every update immediately (avoid "churn blindness"), compute scores on fixed snap tick
2. **No hindsight calibration**: All thresholds are mechanical constants (no ML training in signal generation)
3. **Deterministic replay**: Same inputs + config → same outputs (within FP rounding)
4. **Separation of concerns**: Each engine focuses on one physics dimension
5. **Price conversion abstraction**: Levels in SPY terms, barrier/tape query ES internally

---

## Key Engines

### MarketState (`market_state.py`)
Central state store with ring buffers for ES MBP-10, trades, and ES 0DTE option flow. Provides query interface for engines.

**Update methods**: Called by Ingestor (event-driven)  
**Query methods**: Called by engines (snap-driven)

### BarrierEngine (`barrier_engine.py`)
Computes liquidity state from ES MBP-10 depth changes. Classifies as VACUUM, WALL, ABSORPTION, CONSUMED, WEAK, or NEUTRAL.

**Key insight**: Track depth flow + passive volume to infer FILLED vs PULLED liquidity.

### TapeEngine (`tape_engine.py`)
Computes tape momentum from ES trades. Measures imbalance, velocity, and sweep detection.

**Key insight**: Aggressor classification reveals buy/sell pressure directionality.

### FuelEngine (`fuel_engine.py`)
Estimates dealer gamma effect from ES 0DTE option flow. Classifies as AMPLIFY (dealers chase) or DAMPEN (dealers fade).

**Key insight**: Customer buys option → dealer sells gamma → SHORT gamma → must chase moves.

### ScoreEngine (`score_engine.py`)
Combines physics into composite break score (0-100) via weighted sum:
```
S = 0.45*Liquidity + 0.35*Hedge + 0.20*Tape
```

Includes trigger state machine with hysteresis to avoid flickering signals.

### LevelSignalService (`level_signal_service.py`)
Orchestrator that integrates all engines and produces WebSocket payload. Runs on snap tick cadence.

---

## Data Flow

```
NATS (market.*) → MarketState (ring buffers)
                      ↓
              Level Universe Generation
                      ↓
         For each active level (within MONITOR_BAND):
           ├─ BarrierEngine → BarrierMetrics
           ├─ TapeEngine    → TapeMetrics
           ├─ FuelEngine    → FuelMetrics
           ├─ ScoreEngine   → CompositeScore + Signal
           ├─ Smoothing     → Smooth scores
           └─ RoomToRun     → Runway analysis
                      ↓
         Build WebSocket payload
                      ↓
         Publish to NATS (levels.signals)
```

---

## Price Conversion Protocol

**Critical**: ES futures = ES options (perfect alignment, same underlying).

**Workflow**:
1. Level specified in ES index points: `level_price = 6870.0` (ES)
2. Query ES depth/trades at same price: `es_level = 6870.0` (no conversion!)
3. Query ES option strikes near level: `strikes = [6865, 6870, 6875]` (5pt spacing ATM)
4. Output in ES index points: `level_price = 6870.0`

**No conversion needed**: ES futures and ES options use identical price scale.

---

## Configuration

All parameters sourced from `src/common/config.py`:

**Windows**: `W_b=240s`, `W_t=60s`, `W_g=60s`, `W_v=3s`  
**Bands**: `MONITOR_BAND=0.25`, `TOUCH_BAND=0.10`  
**Thresholds**: `R_vac=0.3`, `R_wall=1.5`, `F_thresh=100`  
**Weights**: `w_L=0.45`, `w_H=0.35`, `w_T=0.20`  
**Smoothing**: `tau_score=2.0s`, `tau_velocity=1.5s`

---

## Direction Context

Level direction depends on spot position:

| Condition | Level Type | Break Direction |
|-----------|-----------|-----------------|
| spot > L  | SUPPORT   | DOWN            |
| spot < L  | RESISTANCE| UP              |

---

## Testing

```bash
cd backend

# Engine unit tests
uv run pytest tests/test_barrier_engine.py -v
uv run pytest tests/test_tape_engine.py -v
uv run pytest tests/test_fuel_engine.py -v
uv run pytest tests/test_score_engine.py -v

# Integration tests
uv run pytest tests/test_core_service.py -v

# Replay determinism
uv run pytest tests/test_replay_determinism.py -v
```

---

## Adding New Engines

To add a new physics dimension (e.g., "Spread Engine"):

1. **Define output dataclass** with metrics + confidence
2. **Implement engine class** with `compute_state()` method
3. **Integrate into ScoreEngine** (add component score method)
4. **Update LevelSignalService** (call new engine in orchestrator)
5. **Add to CONFIG** (new parameters like `W_spread`, `w_S`)
6. **Write tests** (unit + integration)

**See**: Core README (old version) §9 for detailed example.

---

## Critical Invariants

1. **ES MBP-10 event stream**: Process every update (no coarse sampling)
2. **Passive volume only**: Count trades on defending side for fill inference
3. **Price conversion**: Always use `PriceConverter`, never hardcode ratio
4. **Per-level smoothing**: Each level maintains independent `SmootherSet`
5. **Trigger hysteresis**: Signals require sustained conditions (3s default)
6. **Ring buffer cleanup**: Call `cleanup()` before queries to purge old data

---

## Common Issues

**No levels showing**: Check MONITOR_BAND allows levels (spot ± 0.25)  
**Weird spot values**: Data parsing issue (ES/SPY conversion error)  
**Flickering signals**: Increase `TRIGGER_HOLD_TIME` or smoothing `tau`  
**High CPU usage**: Reduce snap interval or optimize query patterns

---

## References

- **Interface contract**: [INTERFACES.md](INTERFACES.md)
- **PLAN.md**: §5 (Core Engines), §12 (Agent Assignments)
- **Common module**: [../common/INTERFACES.md](../common/INTERFACES.md)
- **Tests**: `backend/tests/test_*.py`

---

**Agent assignments**:
- Agent C: MarketState ✅
- Agent D: Barrier + Tape engines ✅
- Agent E: Fuel engine ✅
- Agent F: Level universe + room-to-run ✅
- Agent G: Scoring + smoothing + orchestrator ✅

**Dependencies**: `common` (event types, config, price converter)  
**Consumers**: Gateway (WebSocket), Lake (Gold storage)
