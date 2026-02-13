# Vacuum/Pressure Event-Driven Target Spec (Iteration 4)

## Hard requirements (from latest direction)
1. `5s/15s/60s` lookbacks are removed from the product definition.
2. The system is event-driven: every order book event updates pressure-state math.
3. Live and replay use the same math path (only source adapter differs).
4. Grid is always dense: no empty cells at any emitted state.
5. Legacy code/path compatibility is not a requirement.

## Product definition (new canonical)
The product is a continuous, event-time force-field engine over relative price buckets around spot.

- Domain: futures MBO events
- Grid: `k ∈ [-K, +K]` relative ticks around current spot (`K` configurable; default 40)
- At each event, engine emits a full dense grid with force variables per bucket
- Primary force family: derivative-defined pressure variant (and companion vacuum/resistance variants if needed for downstream inference)

## Core state model per bucket
For each `k`, maintain:
1. mechanics state:
- `add_mass_k`
- `pull_mass_k`
- `fill_mass_k`
- `rest_depth_k`
- `rest_decay_k`
2. derivative state:
- `v_*` (first derivative in event-time with dt normalization)
- `a_*` (second derivative)
- `j_*` (third derivative)
3. force state:
- `pressure_variant_k`
- `vacuum_variant_k` (optional but recommended companion)
- `resistance_variant_k`
4. metadata:
- `last_event_id_k`
- `cell_valid=true` (always true once initialized)

All cells are initialized at startup so no cell is ever missing.

## Event update algorithm
For each incoming MBO event `e`:

1. Resolve current spot reference and relative index frame.
2. Reindex/shift grid if spot moved:
- preserve previous state by bucket shift
- fill newly introduced edge cells with boundary defaults (finite values, never null)
3. Apply event delta to touched bucket(s):
- map event price -> `k_touched`
- update mechanics mass terms
4. Update derivative chains using `dt` from event timestamps.
5. Recompute force variables:
- derivative-defined pressure variant per bucket
- companion variants as configured
6. Guarantee dense output:
- emit every `k` from `-K..+K`
- if untouched, carry previous force value
- no null/NaN/missing buckets
7. Mark provenance:
- touched buckets get `last_event_id_k = e.id`
- untouched buckets retain prior `last_event_id_k`

## Pressure-variant definition style
Pressure variant must be derivative-led. Example structure (implementation may tune weights):

`pressure_variant_k = c1*v_add_k + c2*v_fill_k - c3*v_pull_k + c4*max(-a_rest_depth_k, 0) + c5*j_flow_k`

Guidelines:
- use derivatives of mechanics as primary signal terms
- raw levels can appear only as conditioning terms, not primary drivers
- all terms finite and bounded/normalized for stable rendering

## No-empty-cell guarantee (formal invariants)
At emitted state `S_n` for event `n`:

1. Cardinality invariant:
- `|S_n.grid| == 2K + 1`
2. Completeness invariant:
- every `k ∈ [-K, +K]` exists exactly once
3. Numeric invariant:
- each force variable is finite (`isfinite == true`)
4. Continuity invariant:
- for untouched cell `k`, `S_n[k].value == S_{n-1}[k].value` unless shift-reindex remaps it
5. Event-update invariant:
- at least one bucket has `last_event_id_k == n`

## Architecture decision
Canonical runtime:
- in-memory event engine as single source of truth
- replay feeds historical events into same engine
- live feeds real-time events into same engine

Sampling/serialization:
- emit per event (or throttle transport rate while preserving internal per-event updates)
- optional downstream snapshots/parquet are derived artifacts, not computation authority

## Pipeline impact
Allowed/expected:
1. Keep bronze raw events as immutable source.
2. Bypass silver/gold for vacuum-pressure runtime math if they impose 1s-window assumptions.
3. If persisted artifacts are needed, create dedicated VP event-state outputs (full grid snapshots).

## Proof strategy
To prove requirements:

1. Deterministic replay/live parity:
- same event list -> byte-equivalent grid outputs
2. Property tests on random event streams:
- no missing buckets
- no NaN/Inf
- event-update invariant always true
3. Spot-shift stress tests:
- grid remains dense and stable through rapid spot changes
4. Throughput test:
- every event updates pressure variant without dropped updates in engine state

## Implementation handoff link
Execution plan: `.analysis/IMPLEMENT.md`
