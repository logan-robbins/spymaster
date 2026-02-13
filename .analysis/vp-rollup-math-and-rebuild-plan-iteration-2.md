# Vacuum/Pressure Continuous Derivative Model (Iteration 2, Updated)

## Decision framing
Latest requirement is not "better 1s rollups."  
It is a **continuous derivative-force model**:
- pressure/vacuum definitions come from derivatives of order-flow mechanics
- live and replay run exactly the same math
- per-bucket force shade persists until that bucket receives a new update

Therefore the recommended canonical architecture is event-time in-memory state, with optional sampled outputs.

## 1) Core model shift

### 1.1 From current to target
- Current bias: windowed aggregation -> scalar signal -> derivatives.
- Target: per-bucket mechanics -> derivatives -> local forces -> market-level synthesis.

### 1.2 Per-bucket mechanics state (`k=-40..+40`)
At each relative tick bucket around spot, maintain continuously:
- additive flows: `add`, `pull`, `fill`, `pull_rest`
- level terms: `depth_end`, `depth_rest`
- derived local mechanics: depletion/build rates, wall erosion/support rates

### 1.3 Derivative-first force definition
Define local pressure/vacuum from derivatives of mechanics, not raw values alone:
- `d1` captures local force velocity (how fast a condition is building)
- `d2` captures acceleration/deceleration of that force
- `d3` captures jerk/regime transition pressure

Resistance remains both:
- boundary condition (resting depth structure)
- dynamic derivative term (resistance erosion/build rate)

## 2) Timing model and horizons

### 2.1 Canonical clock
- Compute on every event/tick in memory.
- Replay uses event timestamps from file; live uses incoming stream timestamps.
- Same state machine and formulas for both modes.

### 2.2 Output cadence
- UI stream cadence can be throttled (for example 100-250ms).
- Optional 1s snapshots for persistence/backtesting/storage only.
- 1s must not define physics; it is a sampled representation.

### 2.3 Horizon lanes
`5s/15s/60s` should be interpreted as force context/projection lanes over the same continuous engine state, not distinct window-aggregation code paths.

## 3) Relative-spot bucket mechanics

### 3.1 Bucket frame
- Force field index is relative bucket `k` around current spot.
- When spot moves, shift/reindex bucket field so semantics remain "k above/below current spot."

### 3.2 Sparse updates and persistence
- Only buckets touched by events are recalculated at that event.
- Untouched buckets keep prior local state and rendered shade.
- This persistence rule is required behavior, not fallback.

## 4) "Perfect rollup" reinterpretation under continuous model

If sampled horizons are still needed for reporting:
- never roll derivatives
- recompute derivative lanes from sampled local-force process
- aggregate primitives with operator classes only when producing sampled summaries

Operator classes:
- additive: `SUM` over interval
- boundary state: `LAST` (optionally `MEAN` for occupancy diagnostics)
- nonlinear/ratio/force metrics: recompute from rolled primitives

## 5) Required parity guarantee

Live and replay must differ only by source adapter.

- same engine class
- same update function
- same configs/spans/weights
- same schema fields
- same tests and deterministic replay checks

No dual math stacks.

## 6) Current gaps relative to target

1. Replay/live math divergence remains.
2. Force definitions are still partially level-first rather than derivative-first per bucket.
3. `5s/15s/60s` labels currently map to fixed EMA lanes, not explicit continuous-lane semantics.
4. Frontend rendering is not yet force-field-native.
5. Output contract does not yet expose complete per-bucket derivative-force surfaces.

## 7) Conclusion

For the intended aerodynamic/Bernoulli style product, the right design is:
- continuous event-time bucket-state engine
- derivative-defined local pressure/vacuum/resistance forces
- sampled projections across horizon lanes
- persistent per-bucket shading between updates
- identical live/replay math path

Implementation handoff is specified in `.analysis/IMPLEMENT.md`.
