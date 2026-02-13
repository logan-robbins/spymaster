# Vacuum/Pressure Implementation Milestones (Iteration 3, Updated)

## Phase 0: Lock target semantics
1. Freeze product semantics:
- derivative-defined pressure/vacuum/resistance at bucket level
- live and replay are source adapters only
- bucket shade persists if no bucket update occurs
2. Freeze bucket frame:
- relative ticks around current spot are canonical index
- explicit reindex/shift policy when spot changes
3. Freeze horizon interpretation:
- lanes are context/projection views over one continuous engine, not separate math stacks

Acceptance:
- approved semantic spec with no open ambiguity

## Phase 1: Build one canonical event-time engine
1. Implement a single in-memory bucket-state engine updated per event.
2. Move replay and live to call the same engine update path.
3. Remove replay composite-only fallback behavior for required force fields.
4. Keep optional sampled snapshot emission separate from core update loop.

Acceptance:
- same inputs produce same outputs in replay and live harnesses
- no branch where formulas differ by mode

## Phase 2: Derivative-force model per bucket
1. Add derivative chains (`d1/d2/d3`) for local mechanics per bucket.
2. Define local pressure/vacuum/resistance force equations from those derivatives.
3. Build lane summaries (`5/15/60`) from the same canonical local-force process.
4. Keep weighted forward projection logic on lane outputs.

Acceptance:
- derivative-first force outputs are stable and bounded
- no "roll derivative" shortcuts

## Phase 3: Stream contract and rendering persistence
1. Expand stream payload for bucket-force surfaces and lane summaries.
2. Update frontend rendering to consume force surfaces directly.
3. Enforce persistence rule: no update at bucket => carry previous shade/state.
4. Preserve optional debug overlays (legacy net-flow view).

Acceptance:
- visual statement "pressure at +4, vacuum above" is directly observable
- unchanged buckets remain visually stable until touched

## Phase 4: Verification and hardening
1. Add deterministic parity tests for replay/live.
2. Add bucket persistence tests for no-update intervals.
3. Add spot-shift/reindex correctness tests.
4. Add throughput and latency profiling for continuous updates.

Acceptance:
- parity tests pass
- persistence behavior passes
- performance target met at expected event rates
