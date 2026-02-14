1. [completed] Read `README.md` fully and inventory all vacuum-pressure derivative code paths.
2. [completed] Trace derivative calculations line-by-line in `backend/src/vacuum_pressure/formulas.py`.
3. [completed] Trace derivative calculations line-by-line in `backend/src/vacuum_pressure/incremental.py` (live path).
4. [completed] Validate how `5s/15s/60s` are currently produced (true rollup vs EMA-on-1s stream) and identify mathematical implications.
5. [completed] Define exact rollup math for each metric class (additive, ratio/non-additive, derivative state).
6. [completed] Compare architecture options: derive from rolled-up windows vs compute all windows in real time from event stream.
7. [completed] Write iterative analysis docs in `.analysis/` with recommended canonical approach and migration/validation plan.
8. [completed] Update analysis after derivative-defined-force clarification and create `.analysis/IMPLEMENT.md` handoff plan for implementation LLM.
9. [completed] Pivot to strict event-driven dense-grid design (remove `5s/15s/60s` requirement) and rewrite implementation handoff accordingly.
10. [completed] Read `README.md` in full and audit runtime/mode/version language against current single-pipeline implementation.
11. [completed] Rewrite `README.md` to enforce one canonical PRE-PROD mode (`.dbn` ingest adapter now, Databento socket adapter later) with unchanged math/pipeline semantics.
12. [completed] Verify documented commands locally (`uv run ... --help`, frontend type-check command surface) and finalize task status annotations.
13. [completed] Audit backend/frontend runtime labels for residual `live/replay` wording that can imply multiple behavioral modes.
14. [completed] Update non-functional naming in backend/frontend to canonical PRE-PROD terminology while preserving runtime behavior and parameters.
15. [completed] Verify patched command surfaces and frontend type-check after terminology updates.
16. [completed] Remove remaining misleading `live` status labels from runtime-facing messages where they implied a separate behavioral mode.
