# Codebase Analysis (Current)

Updated on February 20, 2026 after validating all prior findings against the moved codepaths.

This document now contains only active findings that remain open after cleanup and fixes.

---

## Open Findings

### 1. Rolling robust z-score still uses Python-window loops

`backend/src/vp_shared/zscore.py` `robust_zscore_rolling_1d()` still iterates per index and recomputes median/MAD on each sliding window.

Impact:
- This remains a cost center for offline signals that call `robust_zscore()` repeatedly.

Status:
- `rolling_ols_slope()` was optimized to O(n) in `eval_engine.py`.
- Robust z-score path is still the remaining heavy component.

---

### 2. Streaming hot path still builds per-cell Python dict payloads

`backend/src/vacuum_pressure/stream_pipeline.py` still constructs ~101 dict rows per emitted bin and performs many per-field `float()` casts.

Impact:
- Avoidable allocation and conversion pressure in producer path.

Status:
- One extra pass was removed by returning `state5_series` directly from `_annotate_permutation_labels()`.
- Main dict-heavy serialization path is still open.

---

### 3. Frontend MAD calculation still sorts on each push

`frontend/src/experiment-math.ts` `RollingRobustZScore.push()` still builds `absDevs` and sorts every update to compute MAD median.

Impact:
- O(w log w) per update in client-side dev scoring.

---

### 4. Frontend/browser contract still only supports derivative launch URLs

Experiment-browser launch URL construction in `backend/src/vacuum_pressure/server.py` intentionally only maps derivative signal params.

Impact:
- Non-derivative experiment runs still cannot be launched directly into equivalent live visualization mode from the browser.

Note:
- This remains a product/design decision, not a backend correctness bug.

---

### 5. FE/model boundary remains mixed in immutable dataset artifacts

Immutable datasets still include model-derived fields (for example, flow scoring columns in grid rows), not only raw/engineered features.

Impact:
- Feature artifacts are still coupled to modeling assumptions.

---

### 6. Full model one-version architecture is still not complete

Derivative math is now shared via `src/vp_shared`, but there are still separate adapters and additional cross-language model logic (frontend ADS/PFP/SVac vs backend harness versions).

Impact:
- Drift risk remains for duplicated model families.

---

## Completed In This Pass

The following previously reported findings are now fixed and removed from active findings:

- CLI dataset generation wrote wrong `bins.parquet` schema.
- Manifest parsing mismatch that broke experiment-browser stream URL construction.
- `cooldown_bins` drop between `ExperimentSpec` and runner schema.
- Incorrect `promote` WebSocket URL output.
- Uncentered global-std fallback in robust z-score fallback paths.
- Derivative-weight constraint mismatch between scoring and derivative runtime validation.
- `EvalEngine.load_dataset()` row-by-row Python pivot loop.
- Duplicate `query_runs()` read in experiment-runs endpoint.
- `score_dataset()` per-group sort + `.loc` write pattern.
- Dead `frontend/src/experiment-erd.ts` file.
- Dead `products_yaml_path` parameter across runtime config resolution and callsites.
- Missing frontend fail-fast grid coverage validation for hardcoded ADS/PFP k-ranges.
- Runner partial-success behavior (now fail-fast on spec failure).
- Async producer error swallowing in stream wrapper (now propagates as runtime error).
- Silent unknown-field schema behavior in experiment config bridge (now `extra="forbid"` on relevant config models).
- Parallel and eval config default/field divergence (`cooldown_bins`, `max_workers`, `timeout_seconds`) between bridge schemas.

---

## Priority (Remaining)

1. Optimize robust rolling z-score implementation (`vp_shared/zscore.py`).
2. Reduce streaming producer allocation/serialization overhead (`stream_pipeline.py`).
3. Optimize frontend MAD computation in `RollingRobustZScore`.
4. Decide and enforce final FE/model artifact boundary policy.
5. Decide whether to keep or remove frontend/back-end duplicate model families.
