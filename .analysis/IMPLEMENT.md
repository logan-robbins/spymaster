# IMPLEMENT: Event-Driven Dense-Grid Vacuum/Pressure System

## Objective
Build the vacuum-pressure product as a single event-driven engine where:
1. every order book event updates pressure-variant math,
2. live and replay run identical formulas,
3. output grid is always dense (no empty cells),
4. legacy `5s/15s/60s` lookback concept is removed.

Reference spec:
- `.analysis/vp-event-driven-target-spec-iteration-4.md`

Context/rationale:
- `.analysis/vp-derivative-line-by-line-audit-iteration-1.md`
- `.analysis/vp-rollup-math-and-rebuild-plan-iteration-2.md`
- `.analysis/vp-implementation-milestones-iteration-3.md`

## Required behavioral guarantees
These are hard acceptance constraints:
1. `G1`: For each event `e_n`, engine state advances once and recomputes pressure variant.
2. `G2`: Emitted grid contains all buckets `k ∈ [-K, +K]` every time.
3. `G3`: No bucket value is null/NaN/Inf.
4. `G4`: Untouched buckets persist prior values (after spot-frame remap).
5. `G5`: Replay and live produce identical outputs for identical event stream.

## Scope
Modify only vacuum-pressure product components:
- backend: `backend/src/vacuum_pressure/*`, `backend/scripts/run_vacuum_pressure.py`, `backend/tests/test_vacuum_pressure_*`
- frontend: `frontend/src/vacuum-pressure.ts`, `frontend/vacuum-pressure.html`

If existing VP code blocks requirements, replace it.

## Build plan

### Step 1: Create canonical event engine
Implement a single engine class as source of truth for VP math.

Requirements:
1. Input: one MBO event at a time (`ts`, `action`, `side`, `price`, `size`, `order_id`, `flags`).
2. Internal state:
- dense bucket table for `[-K,+K]`
- mechanics and derivative substate per bucket
- event id/version per bucket
3. Output: full dense grid snapshot (or pointer to state) after each event.

Suggested file actions:
- add/replace engine in `backend/src/vacuum_pressure/incremental.py`
- split helper state objects into new module(s) under `backend/src/vacuum_pressure/` if needed

### Step 2: Unify replay/live source adapters
Replay and live must call the exact same `engine.update(event)` function.

Requirements:
1. Replay path iterates historical events and feeds canonical engine.
2. Live path iterates real-time events and feeds canonical engine.
3. No separate formula branches by mode.

Primary files:
- `backend/src/vacuum_pressure/stream_pipeline.py`
- `backend/src/vacuum_pressure/server.py`
- `backend/src/vacuum_pressure/engine.py` (if retained for orchestration only)

### Step 3: Implement spot-relative dense-grid remap
When spot changes:
1. shift bucket field to preserve relative semantics,
2. map existing bucket state into new indices,
3. initialize new edge buckets with finite defaults.

Rules:
- never drop to missing bucket
- never output sparse buckets

### Step 4: Implement derivative-defined pressure variant
Pressure variant must be derivative-led.

Requirements:
1. maintain derivative chains for mechanics in event-time (`dt` aware),
2. compute `pressure_variant_k` primarily from derivative terms,
3. optionally compute companion `vacuum_variant_k`/`resistance_variant_k` for context.

Note:
- raw levels may condition or normalize but should not be the primary force signal.

### Step 5: Guarantee no-empty-cell serialization
Define stream schema to always carry full grid at each emission.

Requirements:
1. include all bucket indices every message,
2. include finite force values for every bucket,
3. include per-cell `last_event_id` (or equivalent provenance).

Primary files:
- `backend/src/vacuum_pressure/server.py`
- `frontend/src/vacuum-pressure.ts`

### Step 6: Frontend persistent shading
Renderer must reflect state persistence:
1. if bucket untouched this event, shade persists,
2. no visual empty holes in grid,
3. force-layer rendering uses pressure-variant field, not net-flow-only fallback.

Primary files:
- `frontend/src/vacuum-pressure.ts`
- `frontend/vacuum-pressure.html`

## Proof plan (must be implemented as tests)

### T1: Event update invariant
For every processed event:
- at least one bucket’s `last_event_id` equals current event id.

### T2: Dense-grid invariant
After every event:
- exactly `2K+1` buckets emitted,
- all expected `k` indices exist once.

### T3: Numeric invariant
All force values are finite for all buckets/all events.

### T4: Persistence invariant
If bucket not touched and not remapped out-of-range by spot shift:
- value remains unchanged from prior event.

### T5: Replay/live parity
Feed same event list to replay and live adapters:
- compare emitted grid states, must match.

### T6: Spot-shift stress
Rapid spot changes:
- grid remains dense, remap stable, no empty cells.

Primary test files:
- `backend/tests/test_vacuum_pressure_incremental.py`
- `backend/tests/test_vacuum_pressure_incremental_events.py`
- add new tests as needed for parity and dense-grid invariants

## Suggested execution checklist
1. Replace dual math paths with one canonical event engine.
2. Wire both modes to same update function.
3. Expand schema to dense per-bucket force payload.
4. Update frontend rendering for force field + persistence.
5. Add invariants/parity tests.
6. Run:
- `cd backend`
- `uv run pytest tests/test_vacuum_pressure_incremental.py -v`
- `uv run pytest tests/test_vacuum_pressure_incremental_events.py -v`
- run any new parity/dense-grid tests

## Done definition
This effort is complete only if:
1. no lookback-lane dependency remains as product-defining math,
2. pressure variant updates at every event,
3. no empty grid cells are emitted/rendered,
4. replay/live parity is proven by tests.
