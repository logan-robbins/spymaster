# Codebase Analysis

Rigorous review of math correctness, application flow, latency, code clarity, and frontend-backend contract integrity.

---

## CRITICAL — Data Correctness

### 1. CLI `generate` writes wrong schema to `bins.parquet`

`backend/src/experiment_harness/cli.py` lines 120–151 write per-bucket (bin_seq × k) rows to `bins.parquet`:

```python
for bucket in bin_grid["buckets"]:
    bucket["bin_seq"] = bin_seq
    all_bucket_rows.append(bucket)
df = pd.DataFrame(all_bucket_rows)
df.to_parquet(bins_path, index=False)
```

`EvalEngine.load_dataset` (`eval_engine.py` lines 170–175) expects one row per time bin with `mid_price` and `ts_ns` columns:

```python
bins_df = pd.read_parquet(paths.bins_parquet)
n_bins = len(bins_df)
mid_price = bins_df["mid_price"].values.astype(np.float64)
```

`DatasetPaths` docstring confirms: `bins_parquet: Path to the bins.parquet file (time bins with mid_price, ts_ns)`.

`GridGenerator._run_pipeline` (`grid_generator.py` lines 293–361) does this correctly — it builds `bins_records` (one row per bin with mid_price, ts_ns) separately from `grid_rows` (one row per bucket). The CLI `_generate_dataset` does not.

**Impact**: Any dataset generated via `cli generate` or `cli run` (which calls `_generate_dataset`) will crash `EvalEngine.load_dataset` with a `KeyError` on `mid_price`. Only datasets generated via `GridGenerator` (grid variant sweeps) have the correct schema.

**Fix**: Refactor `_generate_dataset` to match `GridGenerator._run_pipeline` — collect bin-level metadata (bin_seq, ts_ns, mid_price, bin_start_ns, bin_end_ns, event_id, bin_event_count, book_valid, best_bid_price_int, best_ask_price_int, spot_ref_price_int) into a separate DataFrame and write that as `bins.parquet`. Write the per-bucket grid data to `grid_clean.parquet` only.

---

### 2. Manifest schema mismatch breaks experiment browser streaming URLs

`cli.py` `_generate_dataset` writes a flat manifest (lines 163–173):

```python
manifest = {
    "pipeline_spec_name": pipeline_spec.name,
    "dataset_id": dataset_id,
    "symbol": pipeline_spec.capture.symbol,
    "dt": pipeline_spec.capture.dt,
    "start_time": pipeline_spec.capture.start_time,
    ...
}
```

`server.py` `_build_streaming_url` reads a nested structure (lines 219–225):

```python
src = manifest.get("source_manifest", {})
product_type = src.get("product_type", "future_mbo")
symbol = src.get("symbol", "")
dt = src.get("dt", "")
start_et = src.get("capture_start_et", src.get("stream_start_time_hhmm", ""))
```

Since `source_manifest` doesn't exist in CLI-generated manifests, `symbol` resolves to `""`, `dt` to `""`, and `start_et` to `""`. The "Launch Stream" button in the experiment browser constructs a broken URL.

The `GridGenerator._build_manifest` (grid_generator.py lines 363–398) also writes a different structure (`spec`, `grid_dependent_params`) with no `source_manifest` key.

**Fix**: Standardize on one manifest schema. Either: (a) the server's reader should fall back to top-level keys (`manifest.get("symbol", "")`), or (b) all generators should write a `source_manifest` sub-dict. Option (a) is simpler and backward-compatible.

---

### 3. `cooldown_bins` silently dropped from ExperimentSpec → runner

`ExperimentSpec.to_harness_config()` (`experiment_config.py` line 256) puts `cooldown_bins` inside the `eval` dict:

```python
"eval": {
    ...
    "cooldown_bins": self.eval.cooldown_bins,
}
```

But `config_schema.EvalConfig` has no `cooldown_bins` field. Pydantic v2 silently ignores unknown fields by default. The value is dropped.

The runner's `_expand_param_grid` (`runner.py` lines 457–461) reads cooldown from `config.sweep.universal.get("cooldown_bins", [30])`, which defaults to `[30]`.

**Impact**: If a user sets `eval.cooldown_bins: 10` in their ExperimentSpec YAML, it is silently ignored. All experiments use cooldown=30 unless the user also sets it in the sweep config.

**Fix**: Either add `cooldown_bins` to `config_schema.EvalConfig` and wire it through `_expand_param_grid`, or map `self.eval.cooldown_bins` into the `sweep.universal` dict inside `to_harness_config()`.

---

### 4. `promote` CLI prints wrong WebSocket URL

`cli.py` line 513:

```python
f"ws://localhost:8000/ws/vp?serving={promoted.name}"
```

Correct URL per the server and README: `ws://localhost:8002/v1/vacuum-pressure/stream?serving={name}`.

Wrong port (8000 vs 8002) and wrong path (`/ws/vp` vs `/v1/vacuum-pressure/stream`).

---

## MATH / STATISTICAL

### 5. Uncentered z-score fallback — runtime_model.py and derivative.py

`runtime_model.py` `_robust_or_global_z_from_history` lines 147–150:

```python
std = float(np.std(arr))
if std <= 1e-12:
    return 0.0
return float(arr[-1] / std)
```

When MAD is zero but std > 0, this computes `x / std` without subtracting the mean. The MAD branch correctly centers: `(x - median) / (1.4826 * MAD)`. The fallback does not center, producing a biased z-score shifted by `mean / std`.

Same pattern in `derivative.py` `_robust_or_global_z` lines 43–48:

```python
scale = float(np.std(x))
if scale <= 1e-12:
    return z
return x / scale
```

Divides the entire signal array by global std without centering. This is a normalization, not a z-score.

**When this triggers**: MAD=0 occurs when >50% of values in the window are identical (common during low-activity periods or early warmup with many zeros). The bias equals `mean / std`, which can be large for non-zero-mean distributions.

**Fix**: Both fallbacks should center: `(value - mean) / std`.

---

### 6. SpectrumScorer rejects configs that DerivativeRuntime accepts

`scoring.py` line 100: `if np.any(raw_w <= 0.0): raise ValueError("derivative_weights values must be > 0")`

The SpectrumScorer (per-cell flow scoring) requires ALL three derivative weights to be strictly positive.

`runtime_model.py` line 80: `if abs(d1_weight) + abs(d2_weight) + abs(d3_weight) <= 0.0: raise ValueError(...)` — allows individual zero weights.

The canonical runtime model defaults are `d1_weight=1.0, d2_weight=0.0, d3_weight=0.0` (`config.py` lines 49–51). These defaults would be rejected by SpectrumScorer.

This isn't a live bug because SpectrumScorer and DerivativeRuntime operate at different layers (per-cell vs cross-cell), but it creates confusion when the same parameter names (`derivative_weights`, `d1/d2/d3_weight`) appear in both contexts with incompatible constraints. A user tuning `derivative_weights` via ServingSpec could trigger SpectrumScorer's validation while the harness DerivativeSignal runs fine with those same values.

---

### 7. `eval_engine.py` rolling functions are pure-Python O(n·w) loops

`rolling_ols_slope` (lines 30–63): Recomputes `sum_y` and `sum_xy` from scratch every iteration. The `x` vector is fixed (`[0, 1, ..., w-1]`), so `sum_xy` and `sum_y` can be maintained incrementally with O(1) subtract-old/add-new updates.

`robust_zscore` (lines 66–102): Allocates a new array slice and computes two `np.median()` calls per bin. For a dataset with 3600 bins and window=300, this is 3600 × 300 = 1M array elements touched per signal evaluation.

These functions are called inside `DerivativeSignal.compute()` for z1, z2, z3 — meaning 3× the cost.

**Fix**: Use `pandas.Series.rolling` with custom apply, or port to the incremental ring-buffer pattern already used in `SpectrumScorer._robust_z`. The frontend's `experiment-math.ts` already has incremental implementations (`IncrementalOLSSlope`, `RollingRobustZScore`).

---

## LATENCY / PERFORMANCE

### 8. `EvalEngine.load_dataset` grid pivot is a row-by-row Python loop

`eval_engine.py` lines 199–209:

```python
for row_i in range(len(grid_df)):
    b_idx = bin_seq_to_idx.get(bin_seqs[row_i])
    k_idx = int(k_vals[row_i]) - K_MIN
    arr[b_idx, k_idx] = col_vals[row_i]
```

For a dataset with 3600 bins × 101 ticks × N columns, this is 363K iterations per column in pure Python. Each iteration does a dict lookup and two array index operations.

**Fix**: Use `pd.pivot_table` or vectorized numpy fancy indexing:

```python
b_idxs = np.array([bin_seq_to_idx[s] for s in bin_seqs])
k_idxs = k_vals - K_MIN
arr[b_idxs, k_idxs] = col_vals
```

---

### 9. `ResultsDB.append_run` is O(n²) over experiment history

`results_db.py` lines 60–73: Every `append_run()` reads the full parquet file, concatenates one row, and rewrites. For an experiment with 500 runs (common in sweep experiments), this means 500 full file reads and writes, with the file growing each time.

**Fix**: Buffer appends in memory and flush periodically, or use a transactional store (e.g., DuckDB, SQLite). At minimum, collect results in a list during `ExperimentRunner.run()` and write once at the end.

---

### 10. `server.py` experiment_runs endpoint queries ResultsDB twice

`server.py` lines 261–262 and 313:

```python
meta_df = db.query_runs()          # First full read
...
all_meta = db.query_runs()          # Second full read (identical)
```

Each `query_runs()` reads the full `runs_meta.parquet`. The second call is redundant.

**Fix**: Reuse `meta_df` for both the params lookup and the filter enumeration.

---

### 11. `score_dataset` groupby iteration pattern

`scoring.py` lines 308–326: Iterates over pandas `groupby` groups, calls `sort_values` per group, then writes back via `df.loc[idx, col]`. Pandas `.loc` indexing with a column name triggers label-based lookup per assignment.

For 3600 bins × 101 rows/bin, this is 3600 sort + 3600 × 2 `.loc` write operations.

**Fix**: Pre-sort the DataFrame once by (bin_col, k_col), then iterate in chunks of `n_cells`. Use `.values` indexing instead of `.loc` for writes.

---

### 12. `stream_pipeline._build_bin_grid` — hot-path allocation pressure

`stream_pipeline.py` lines 571–651: Per bin (~10 Hz), creates ~101 Python dicts with ~30 `float()` casts each = ~3000 `float()` calls plus ~101 dict allocations. This runs on the producer thread and feeds the WebSocket queue.

The `_annotate_permutation_labels` function (lines 130–161) then iterates over these dicts AGAIN to add state5_code and microstate labels.

The state_model input (lines 843–847) iterates over the dicts a THIRD time to extract state5_code into a numpy array:

```python
state5_series = np.asarray(
    [int(row["state5_code"]) for row in grid["buckets"]],
    dtype=np.int8,
)
```

Three full Python-loop passes over 101 dicts per bin on the hot streaming path.

**Fix**: Build the grid output from numpy arrays directly rather than through Python dicts. Maintain a structured numpy array or use Arrow record batches natively. Extract state5_code from the engine arrays before dict construction.

---

### 13. Frontend `RollingRobustZScore` — O(n) sort per push for MAD

`experiment-math.ts` lines 126–131:

```typescript
const absDevs: number[] = new Array(n);
for (let i = 0; i < n; i++) {
    absDevs[i] = Math.abs(this.sorted[i] - med);
}
absDevs.sort((a, b) => a - b);
```

Every `push()` allocates a new array and sorts it (O(w log w)) to find the median of absolute deviations. Since the sorted values array is already maintained, a more efficient approach would maintain a second sorted array of deviations, or use a selection algorithm for the median.

For ADS with 3 OLS slope windows × 3 z-score trackers running at 10 Hz, this is 30 array sorts per second.

---

## DEAD CODE / STALE REFERENCES

### 14. `experiment-erd.ts` — ERD removed from composite but file persists

`experiment-engine.ts` line 18: `ERD removed — weakest signal (36.7% TP, negative PnL)`. The engine imports only ADS, PFP, SVac. But `experiment-erd.ts` (104 lines) still exists and ships with the frontend build.

**Fix**: Delete `frontend/src/experiment-erd.ts` or move to a disabled/archive directory.

---

### 15. `config.py` `resolve_config` — dead `products_yaml_path` parameter

`config.py` lines 573–579:

```python
def resolve_config(
    product_type: str,
    symbol: str,
    products_yaml_path: Path,
) -> VPRuntimeConfig:
    del products_yaml_path
```

The parameter is accepted and immediately deleted. Call sites still pass it. This is a migration artifact that adds confusion.

**Fix**: Remove the parameter and update all call sites.

---

### 16. Python 3.12 typing imports — `Dict`, `Sequence`, `Tuple`, `List` from `typing`

Multiple files import `Dict`, `Sequence`, `Tuple`, `List`, `Any` from `typing` when Python 3.12 supports `dict`, `list`, `tuple`, `Sequence` natively. Examples:

- `config.py`: `from typing import Any, Dict, Iterable, Mapping, Tuple`
- `stream_pipeline.py`: `from typing import Any, AsyncGenerator, Dict, Generator`
- `server.py`: `from typing import Any, Dict, List`
- `spectrum.py`: `from typing import Dict, Sequence`

Inconsistent: some files already use lowercase (`dict[str, Any]`) while importing uppercase aliases.

---

## FRONTEND ↔ BACKEND CONTRACT

### 17. Frontend experiment signals use hardcoded k-ranges, no adaptation to grid_radius_ticks

ADS bands span k = ±23 (`experiment-ads.ts` lines 23–38). PFP zones span k = ±12 (`experiment-pfp.ts` lines 13–16). These are hardcoded.

Backend `grid_radius_ticks` defaults to 50 but is configurable. If `grid_radius_ticks < 23`, ADS silently computes with partial bands (the `kMean` function skips missing k values). Signal quality degrades without warning.

No validation exists in `ExperimentEngine` to check whether the received grid covers the required k-ranges for all sub-signals.

---

### 18. Experiment browser → streaming launch does not set `dev_scoring=true`

`server.py` `_build_streaming_url` (lines 202–240) constructs the streaming URL from signal params but never includes `dev_scoring=true`. The frontend `ExperimentEngine` (client-side ADS/PFP/SVac composite) is only instantiated when `dev_scoring=true` (`vacuum-pressure.ts` line 1916).

This means launched streams always use backend-only scoring, which is correct for promoted derivative signals. However, there is no way from the experiment browser to launch a stream with the client-side composite engine for non-derivative signals (ADS, PFP, etc.). The experiment browser marks all non-derivative signals as `can_stream: false`.

This is a design limitation, not a bug — but it means the experiment browser can only visualize derivative signal results in real-time. Other statistical signals that performed well in offline evaluation have no path to live visualization.

---

### 19. `ExperimentSpec.to_harness_config` sweep config bridging is fragile

The ExperimentSpec uses `ExperimentSweepConfig` with a `scoring` dict for scoring parameter sweeps. The harness runner uses `SweepConfig` with a `universal` dict. The bridge in `_build_universal_sweep` (`experiment_config.py` lines 277–299) copies `sweep.scoring` keys into the `universal` dict.

But `ExperimentSweepConfig` has `scoring` + `per_signal`, while `SweepConfig` has `universal` + `per_signal`. The field names differ (`scoring` vs `universal`) without a clear reason. The `_build_universal_sweep` method only maps `scoring` → `universal`, meaning any `cooldown_bins` values in `sweep.scoring` also get mapped to universal (where the runner looks for them).

This creates a non-obvious coupling: to set cooldown sweep values through ExperimentSpec, a user must put `cooldown_bins` in `sweep.scoring` (not `eval.cooldown_bins`), because the eval path is broken (see finding #3).

---

## CONFIG SCHEMA DUPLICATION

### 20. Two `ParallelConfig` classes with different defaults

- `config_schema.ParallelConfig`: `max_workers: int = 4`, `timeout_seconds: int = 3600`
- `experiment_config.ExperimentParallelConfig`: `max_workers: int = 3`, `timeout_seconds: int = 7200`

The ExperimentSpec path uses 3 workers with 7200s timeout. The direct ExperimentConfig path (if used without the ExperimentSpec bridge) uses 4 workers with 3600s timeout.

---

### 21. Two `EvalConfig` classes with subtly different field sets

`config_schema.EvalConfig` (used by the runner): `tp_ticks`, `sl_ticks`, `max_hold_bins`, `warmup_bins`, `tick_size`, `min_signals` — NO `cooldown_bins`.

`experiment_config.ExperimentEvalConfig` (user-facing): same fields PLUS `cooldown_bins: int | list[int] = 20`.

The `cooldown_bins` field exists only in the user-facing schema and is silently dropped during the bridge (finding #3).

---

## SUGGESTED FIX PRIORITY

| # | Severity | Finding | Effort |
|---|----------|---------|--------|
| 1 | **CRITICAL** | CLI generate bins.parquet schema wrong | Medium — refactor to match GridGenerator pattern |
| 2 | **CRITICAL** | Manifest schema mismatch for streaming URLs | Low — normalize server-side reader |
| 3 | **HIGH** | cooldown_bins silently dropped | Low — wire through to_harness_config or add to EvalConfig |
| 4 | **HIGH** | Promote CLI wrong URL | Trivial — fix string literal |
| 5 | **HIGH** | Uncentered z-score fallback | Low — add centering to both fallbacks |
| 7 | **MEDIUM** | eval_engine O(n·w) Python loops | Medium — port to incremental or vectorized |
| 8 | **MEDIUM** | Grid pivot Python loop | Low — use numpy fancy indexing |
| 9 | **MEDIUM** | ResultsDB O(n²) append | Medium — buffer appends, flush once |
| 10 | **LOW** | Double ResultsDB query | Trivial — reuse variable |
| 11 | **MEDIUM** | score_dataset groupby perf | Low — pre-sort + chunk |
| 12 | **MEDIUM** | stream_pipeline hot-path dict allocation | High — requires structural refactor |
| 13 | **LOW** | Frontend MAD sort per push | Medium — maintain sorted deviation array |
| 14 | **LOW** | Dead experiment-erd.ts | Trivial — delete file |
| 15 | **LOW** | Dead products_yaml_path param | Low — remove param + update call sites |
| 16 | **LOW** | typing imports cleanup | Low — search and replace |
| 6 | **LOW** | SpectrumScorer vs DerivativeRuntime weight constraints | Low — document or harmonize |
| 17 | **LOW** | Hardcoded k-ranges in frontend signals | Low — add range validation |
| 18 | **INFO** | Non-derivative signals can't be live-visualized | Design decision |
| 19 | **LOW** | Fragile sweep config bridge | Low — rename for clarity |
| 20 | **LOW** | Duplicate ParallelConfig defaults | Trivial — unify |
| 21 | **LOW** | Duplicate EvalConfig schemas | Folds into fix #3 |

---

## PASS 2 — Fail-Fast, One-Version, Strict SoC

This pass re-prioritizes findings for your stated target:
- immediate catastrophic failure on contract breaks
- one canonical implementation per concept
- strict Data Engineering -> Feature Engineering -> Modeling separation
- minimal code, no soft compatibility paths

### A) Highest-Risk Architectural Violations

### A1. The runtime model is duplicated across offline and online paths (not one-version)

There are two independent implementations of the same derivative signal family:
- Online/live: `backend/src/vacuum_pressure/runtime_model.py` (`DerivativeRuntime`)
- Offline/harness: `backend/src/experiment_harness/signals/statistical/derivative.py` (`DerivativeSignal`)

They are mathematically similar but not identical:
- Different z-score fallback behavior (rolling history scalar in runtime vs full-array fallback in harness)
- Different fallback centering behavior (both currently wrong in different ways)
- Different implementation shape (incremental deque vs batch vectorized)

This is train/serve skew risk for promoted derivative runs.

**Fail-fast one-version fix**: keep one canonical derivative implementation with two adapters:
1) incremental adapter for streaming
2) batch replay adapter for harness
Both must call shared internal math functions.

---

### A2. Frontend and backend both implement ADS/PFP/SVac logic (model duplication)

Duplicate implementations exist in:
- Frontend: `frontend/src/experiment-ads.ts`, `experiment-pfp.ts`, `experiment-svac.ts`
- Backend harness: `backend/src/experiment_harness/signals/statistical/ads_pfp_svac.py`

This duplicates model logic across languages and guarantees drift over time.

**Fail-fast one-version fix**:
- Either remove frontend dev scoring entirely and consume backend-provided model outputs only,
- Or make frontend load a generated model artifact from backend (same coefficients and transforms), not a second hand-coded implementation.

---

### A3. Claimed scoring invariant is partially stale in practice

The docs state `scoring.py` is the single source of truth for server + harness. But `score_dataset()` is only referenced in docs, not actually called in harness execution paths.

Result: the invariant is true for one module by design, but not true for the full modeling stack users care about (especially derivative runtime promotion flow).

**Fix**: enforce invariant by integration tests that compare online-vs-offline outputs on the same bins for each promoted model family.

---

### B) Separation of Concerns Violations (Data -> Features -> Model)

### B1. `stream_pipeline.py` mixes all layers in one hot path

`backend/src/vacuum_pressure/stream_pipeline.py` currently performs:
1) raw event ingest and book-state updates
2) feature generation (`v_*`, `a_*`, `j_*`, vacuum/pressure)
3) model scoring (`SpectrumScorer`, state model)
4) label synthesis (`state5_code`, `microstate_id`)
5) serving payload formatting (`buckets` dicts)

This couples data and modeling concerns and makes strict validation/repro difficult.

**Fix**:
- Stage 1 module: raw -> canonical bin snapshots
- Stage 2 module: snapshots -> deterministic feature frame
- Stage 3 module: feature frame -> model outputs
- Stage 4 module: transport serialization only

---

### B2. Pipeline datasets include modeled fields, not just features

`cli generate` and `grid_generator` persist `flow_score` and `flow_state_code` in dataset rows. These are model outputs, not pure engineered features.

That breaks a clean FE/Model boundary and can lock downstream experiments to stale model assumptions baked at dataset generation time.

**Fix**:
- Immutable dataset should contain only raw-derived and deterministic engineered features.
- Model outputs should be generated in experiment runs (or stored in separate model-output artifacts keyed by model hash).

---

### B3. There are two dataset generation implementations with diverging behavior

- `backend/src/experiment_harness/cli.py::_generate_dataset`
- `backend/src/experiment_harness/grid_generator.py::_run_pipeline`

They currently diverge in schema correctness (`bins.parquet` bug), manifest shape, and metadata richness.

**Fix**: one canonical dataset builder function, called by both paths.

---

### C) Catastrophic-Failure Contract Gaps

### C1. Runner suppresses run failures and continues (non-catastrophic)

`backend/src/experiment_harness/runner.py` catches broad `Exception` inside both sequential and parallel execution and continues:
- `_run_sequential` catches and logs per spec
- `_run_parallel` catches and logs per future

This allows partial-success experiment outputs with hidden failed specs.

**Fix**:
- Default mode: fail-fast (first spec failure aborts entire experiment with non-zero exit)
- Optional explicit `--allow-partial` mode if needed, but not default

---

### C2. Async producer swallows errors and ends stream quietly

`async_stream_events` producer catches exceptions, logs, then sends sentinel. Client only sees stream end.

**Fix**:
- propagate producer exception to consumer and WebSocket layer,
- emit explicit error frame and hard-close with non-success code.

---

### C3. Config bridge drops fields silently (`cooldown_bins`)

`ExperimentSpec -> ExperimentConfig` bridge silently loses fields due schema mismatch.

**Fix**:
- `extra = "forbid"` for all user-facing and bridge schemas,
- hard fail on unknown/dropped keys.

---

### C4. Streaming URL param mapping drops unknown signal params silently

`server.py` `_SIGNAL_PARAM_TO_WS` maps known derivative keys and ignores all others. If run metadata contains unrecognized params, launch URL is silently incomplete.

**Fix**:
- fail URL build when derivative params contain unmapped keys,
- return explicit API error in run detail to prevent false-confidence launch.

---

### D) PhD Robust Standardization (chosen direction)

Decision for implementation: move from ad-hoc fallback rules to a single robust
standardization operator used by both online and offline paths.

Canonical model:
- robust center: `m_t = median(window_t)`
- robust scale: `s_t = 1.4826 * MAD(window_t)`
- uncertainty-aware shrinkage scale:
  - `s_t^2_shrunk = (1 - lambda_t) * s_t^2 + lambda_t * s_prior^2`
  - `lambda_t` increases when information quality is poor (flat windows, low effective variation)
- score:
  - `z_t = (x_t - m_t) / max(s_t_shrunk, eps)`

This removes the non-centered `x/std` bias while avoiding hard discontinuities
from a binary degenerate-scale branch.

Production constraints for SMTS-grade efficiency:
- one implementation in Python for this operator, imported by both:
  - `backend/src/vacuum_pressure/runtime_model.py`
  - `backend/src/experiment_harness/signals/statistical/derivative.py`
- no Python per-element loops in hot paths; vectorized arrays only
- bounded memory and fixed-size ring buffers
- explicit `nan/inf` sanitation and deterministic clipping in one place

Fail-fast requirements:
- hard error if windows, min_periods, or prior-scale config are invalid
- hard error if online and offline parity checks exceed tolerance
- no silent alternate fallback modes

Latency target:
- <= 0.05 ms/bin additional overhead in state-model runtime path
- <= 5% overhead in offline harness derivative evaluation vs baseline
- parity metric tracked continuously: online vs batch absolute delta p99

---

### E) Canonical Minimal Target Architecture

One strict pipeline only:
1. **Data Engineering (`de`)**
   - Input: provider `.dbn`
   - Output: canonical bin-aligned raw tensors + strict schema checks
   - No model math allowed
2. **Feature Engineering (`fe`)**
   - Input: canonical raw tensors
   - Output: deterministic features only (`v_*`, `a_*`, `j_*`, etc.)
   - No thresholds, no scores, no labels tied to model choices
3. **Modeling (`model`)**
   - Input: feature tensors
   - Output: scores, states, decisions
   - Single implementation reused for batch and incremental adapters
4. **Serving (`serve`)**
   - Input: model output
   - Output: transport frames (JSON/Arrow), no math

Hard requirements:
- one function for dataset generation
- one function for each model family
- schema version/hash must include all upstream contracts
- all bridge schemas set to forbid unknown fields

---

### F) Updated Critical Priority Order

1. Unify dataset generation path and fix `bins.parquet` schema.
2. Enforce fail-fast in runner and async stream producer.
3. Enforce strict schema (`extra=forbid`) across spec bridges.
4. Implement single robust standardization operator (PhD variant) and reuse across online/offline derivative paths.
5. Remove model outputs from immutable feature datasets (or split artifacts).
6. Resolve manifest/URL contract and reject unmapped params.
7. Add parity + latency acceptance gates to CI before rollout.

---

### G) Implementation Plan (PhD + SMTS Efficiency)

1. **Design lock (one version)**
   - Define one `robust_standardize()` API with strict config schema.
   - Forbid local re-implementations in signal modules.

2. **Core implementation**
   - Add shared module under `backend/src/vacuum_pressure/` for:
     - rolling median/MAD over ring buffers
     - shrinkage-scale computation
     - deterministic sanitize/clip
   - Keep vectorized batch and incremental adapters backed by same internals.

3. **Replace duplicate paths**
   - Refactor `runtime_model.py` derivative z calculations to call shared operator.
   - Refactor harness `derivative.py` to call the same operator.

4. **Verification and gates**
   - Build fixture replay test: identical bins through online and offline adapters.
   - Add acceptance thresholds:
     - numerical parity (tight tolerance)
     - runtime latency budget (state model path)
   - Fail CI on either breach.

5. **Rollout**
   - Run shadow comparison on historical sessions.
   - Promote only after parity and latency pass on representative regimes.
