# Spymaster

Vacuum-pressure signal research system. Replays Databento MBO .dbn files through an order-book physics engine, emits fixed-bin dense grids over WebSocket, and exposes an offline experiment harness for signal tuning and promotion.

---

## ASCII System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIVE SERVING PATH                                 │
│                                                                             │
│  instrument.yaml ──► VPRuntimeConfig                                        │
│                            │                                                │
│  .dbn files ──► iter_mbo_events ──► AbsoluteTickEngine (8192 ticks, O(1))   │
│                                          │  two-force model (pressure/vacuum)│
│                                          │  EMA derivative chain (v, a, j)  │
│                                          ▼                                  │
│  [warmup phase] ── book cache pkl ──► reanchor_to_bbo                       │
│                                          │                                  │
│                                          ▼ (every cell_width_ms)            │
│                                   grid_snapshot_arrays()                    │
│                                          │                                  │
│                                          ▼                                  │
│                              IndependentCellSpectrum                        │
│                         (flow windows, rollup, tanh-z-score)               │
│                                          │                                  │
│                                          ▼                                  │
│                               DerivativeRuntime                             │
│                          (state5_code → z-score → tanh blend)              │
│                                          │                                  │
│                                          ▼                                  │
│                              ┌────────────────────┐                        │
│                              │  FastAPI WebSocket  │ :8002                  │
│                              │ /v1/vacuum-pressure │                        │
│                              │      /stream        │                        │
│                              └────────┬───────────┘                        │
│                                       │ runtime_config JSON (once)          │
│                                       │ grid_update JSON (per bin)          │
│                                       │ Arrow IPC binary (per bin)          │
│                                       ▼                                     │
│                              ┌────────────────────┐                        │
│                              │   Vite Frontend     │ :5174                  │
│                              │  vacuum-pressure    │ heatmap + gauges        │
│                              │  experiments        │ result browser          │
│                              └────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT HARNESS PATH                              │
│                                                                             │
│  ExperimentSpec YAML ──► ServingSpec YAML ──► PipelineSpec YAML             │
│  (sweep/eval/TP-SL)       (scoring/signal)     (grid/features)              │
│         │                       │                     │                     │
│         └───────────────────────┴─────────────────────┘                    │
│                                 │                                           │
│                         resolve_runtime_config()                            │
│                                 │                                           │
│                          stream_events() ──► bins.parquet                   │
│                                              grid_clean.parquet             │
│                                              manifest.json                  │
│                                              checksums.json                 │
│                                 │                                           │
│                          ExperimentRunner                                   │
│                          (sweep × signal × cooldown × TP/SL)               │
│                                 │                                           │
│                         ResultsDB (parquet)                                 │
│                         MLflow tracking (optional)                          │
│                                 │                                           │
│           promote winner ──► PublishedServingSpec + alias registry update    │
│                                 │                                           │
│              serve: ws://localhost:8002/.../stream?serving=<alias_or_id>     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Instrument Lock

One environment = one instrument. Lock defaults live in `backend/src/vacuum_pressure/instrument.yaml`.

Override path via env: `VP_INSTRUMENT_CONFIG_PATH=/path/to/override.yaml`

Effective lock/runtime values (`product_type`, `symbol`, bin sizing, grid radius, warmup, model params) are config-driven. This README intentionally does not define a canonical value set.

`resolve_config(product_type, symbol)` in `config.py` enforces the lock — mismatches fail fast.

---

## Environment

```bash
cd backend && uv sync
cd frontend && npm ci
```

Backend: Python 3.12, `uv` exclusively. Frontend: TypeScript/Vite, port 5174.

---

## Check Data Before Starting

```bash
# Raw .dbn replay files
find backend/lake/raw/source=databento/product_type=<product_type> -maxdepth 3 -type d -name 'table=market_by_order_dbn'

# Immutable experiment datasets
find backend/lake/research/vp_immutable -maxdepth 2 -type f -name bins.parquet
```

---

## Start Backend

```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
cd backend
nohup uv run scripts/run_vacuum_pressure.py --port 8002 > /tmp/vp_backend.log 2>&1 &
```

Health: `curl -s http://localhost:8002/health`

All flags: `cd backend && uv run scripts/run_vacuum_pressure.py --help`

Runtime contract: stream clients must pass `serving=<alias_or_id>`. Instrument/session/runtime params are resolved from immutable published serving specs via registry; ad-hoc stream overrides are rejected.

Performance telemetry: `--perf-latency-jsonl /tmp/latency.jsonl --perf-window-start-et 09:30 --perf-window-end-et 10:30`

---

## Start Frontend

```bash
kill $(lsof -t -iTCP:5174) 2>/dev/null
cd frontend
nohup npm run dev > /tmp/vp_frontend.log 2>&1 &
```

Pages:
- `http://localhost:5174/vacuum-pressure.html` — live heatmap visualization
- `http://localhost:5174/experiments.html` — experiment result browser (requires backend on :8002)

Frontend URL params: `serving=<alias_or_id>` only. Any runtime-changing params are rejected fail-fast.

---

## WebSocket Protocol

```
ws://localhost:8002/v1/vacuum-pressure/stream?serving=<serving_alias_or_id>
```

Strict parity mode: any extra query params are rejected (`1008`).

**Message sequence per connection:**

1. **text** `runtime_config` JSON — full `VPRuntimeConfig` fields + `state_model` block + `serving` block + `effective_config_hash`. Frontend uses this to configure all rendering constants (no hardcoded instrument assumptions).

2. Per bin (repeated until EOF):
   - **text** `grid_update` JSON — `ts_ns`, `bin_seq`, `bin_start_ns`, `bin_end_ns`, `bin_event_count`, `event_id`, `mid_price`, `spot_ref_price_int`, `best_bid/ask_price_int`, `book_valid`, `state_model_*` fields (score, ready, d1/d2/d3, z1/z2/z3, bull/bear/mixed intensity, dominant_state5_code)
   - **binary** Arrow IPC — 2×`grid_radius_ticks`+1 rows, schema defined in `stream_contract.py:_BASE_GRID_FIELDS`

**Arrow IPC schema fields** (per row = one price tick, `k` = relative offset from spot):
`k`, `pressure_variant`, `vacuum_variant`, `add_mass`, `pull_mass`, `fill_mass`, `rest_depth`, `bid_depth`, `ask_depth`, `v_add`, `v_pull`, `v_fill`, `v_rest_depth`, `v_bid_depth`, `v_ask_depth`, `a_add`, `a_pull`, `a_fill`, `a_rest_depth`, `a_bid_depth`, `a_ask_depth`, `j_add`, `j_pull`, `j_fill`, `j_rest_depth`, `j_bid_depth`, `j_ask_depth`, `composite`, `composite_d1`, `composite_d2`, `composite_d3`, `flow_score`, `flow_state_code`, `best_ask_move_ticks`, `best_bid_move_ticks`, `ask_reprice_sign`, `bid_reprice_sign`, `microstate_id`, `state5_code`, `chase_up_flag`, `chase_down_flag`, `last_event_id`

---

## Experiment Workflow

All commands from `backend/`.

### 1. Generate dataset (from PipelineSpec)

```bash
uv run python -m src.experiment_harness.cli generate <pipeline_spec.yaml>
```

Idempotent: skips if hash-addressed output already exists under `lake/research/vp_immutable/`.
Output per dataset: `bins.parquet` (one row/bin), `grid_clean.parquet` (per-bin per-k, model cols excluded), `manifest.json`, `checksums.json`.

### 2. Run experiment (from ExperimentSpec)

```bash
uv run python -m src.experiment_harness.cli run <experiment_spec.yaml>
```

Resolves ExperimentSpec → ServingSpec → PipelineSpec. Auto-generates dataset if missing. Expands sweep axes × signal params × cooldown × TP/SL. Persists to `lake/research/vp_harness/results/`. Fail-fast on any spec error.

### 3. Compare results

```bash
uv run python -m src.experiment_harness.cli compare --min-signals 5
uv run python -m src.experiment_harness.cli compare --signal derivative --sort tp_rate
```

### 4. Promote winner

```bash
uv run python -m src.experiment_harness.cli promote <experiment_spec.yaml> --run-id <winner_run_id> --alias <serving_alias>
```

Writes an immutable `PublishedServingSpec` YAML to `configs/serving_versions/` and updates `serving_registry.sqlite` alias mapping.

### 5. Stream with promoted config

```
http://localhost:5174/vacuum-pressure.html?serving=<serving_alias_or_id>
```

### Other CLI commands

```bash
uv run python -m src.experiment_harness.cli list-signals
uv run python -m src.experiment_harness.cli list-datasets
uv run python -m src.experiment_harness.cli online-sim --signal derivative --dataset-id <id> --bin-budget-ms 100
```

Harness internals: `backend/src/experiment_harness/README.md`

---

## Config Architecture

Three-layer pydantic v2 hierarchy for experiment workflow. Each layer references the one below by name.

```
ExperimentSpec  →  ServingSpec  →  PipelineSpec
(sweep/eval)       (scoring/signal)   (grid/features)
```

YAML locations (all relative to `backend/lake/research/vp_harness/configs/`):
- `pipelines/` — PipelineSpec YAMLs: capture window + engine param overrides
- `serving/` — ServingSpec YAMLs: experiment-time scoring + signal + projection params
- `serving_versions/` — immutable PublishedServingSpec YAMLs used by live stream runtime
- `experiments/` — ExperimentSpec YAMLs: sweep axes + TP/SL eval + tracking
- `gold_campaigns/` — batch dataset generation campaigns

Config source files (stable):
- `backend/src/vacuum_pressure/pipeline_config.py` — `PipelineSpec`, `CaptureConfig`, `PipelineOverrides`
- `backend/src/vacuum_pressure/serving_config.py` — `ServingSpec`, `PublishedServingSpec`, and runtime snapshot mapping helpers
- `backend/src/vacuum_pressure/serving_registry.py` — SQLite alias/version registry + serving resolution
- `backend/src/vacuum_pressure/experiment_config.py` — `ExperimentSpec`
- `backend/src/vacuum_pressure/config.py` — `VPRuntimeConfig` (frozen dataclass), `resolve_config()`, `build_config_from_mapping()`, `build_config_with_overrides()`

---

## Experiment Browser REST API

Served by VP server on :8002:
- `GET /v1/experiments/runs?signal=&dataset_id=&sort=tp_rate&min_signals=5&top_n=50`
- `GET /v1/experiments/runs/{run_id}/detail`

Streaming URL is built from serving alias registry mappings (`run_id -> serving_alias`). Runs without promoted serving alias return `can_stream=false`.

---

## Source Map

### Stable: never changes structure

| File | Owns |
|------|------|
| `backend/src/vacuum_pressure/instrument.yaml` | Single-instrument lock: all runtime defaults |
| `backend/src/vacuum_pressure/config.py` | `VPRuntimeConfig` frozen dataclass, `resolve_config()`, `build_config_from_mapping()`, `build_config_with_overrides()` |
| `backend/src/vacuum_pressure/event_engine.py` | `AbsoluteTickEngine`: 8192-tick pre-allocated arrays, O(1) BBO, two-force model, EMA chain, book import/export |
| `backend/src/vacuum_pressure/stream_pipeline.py` | `stream_events()` / `async_stream_events()`: DBN → fixed-bin grids; book cache logic; permutation label annotation; state model invocation |
| `backend/src/vacuum_pressure/spectrum.py` | `IndependentCellSpectrum`: multi-window flow rollup, tanh-z-score, forward projection |
| `backend/src/vacuum_pressure/scoring.py` | `SpectrumScorer`: single implementation used by server AND harness (zero train/serve skew) |
| `backend/src/vacuum_pressure/runtime_model.py` | `DerivativeRuntime`: incremental per-bin state5_code scorer; `DerivativeRuntimeParams`; `DerivativeRuntimeOutput` |
| `backend/src/vacuum_pressure/app.py` | FastAPI composition root: middleware + health + route registration |
| `backend/src/vacuum_pressure/api_experiments.py` | REST experiment browser endpoints and run launch URL mapping |
| `backend/src/vacuum_pressure/api_stream.py` | WebSocket route registration and serving-only query contract enforcement |
| `backend/src/vacuum_pressure/stream_session.py` | Stream session setup + runtime config resolution + async stream loop |
| `backend/src/vacuum_pressure/stream_contract.py` | Wire contract builders and Arrow IPC schema/serialization |
| `backend/src/vacuum_pressure/serving_registry.py` | Serving alias/version registry (SQLite) and immutable spec resolution |
| `backend/src/vacuum_pressure/replay_source.py` | `iter_mbo_events()`: DBN file iterator; `_resolve_dbn_path()` |
| `backend/src/vp_shared/hashing.py` | `stable_short_hash()`: deterministic dataset IDs |
| `backend/src/vp_shared/zscore.py` | `robust_or_global_z_latest()`, `weighted_tanh_blend()`, validation helpers |
| `backend/src/vp_shared/derivative_core.py` | `compute_state5_intensities()`, `derivative_base_from_intensities()`, `normalized_spatial_weights()`, `STATE5_CODES` |
| `backend/src/vp_shared/yaml_io.py` | `load_yaml_mapping()` |
| `backend/src/experiment_harness/cli.py` | Click CLI: `generate`, `run`, `compare`, `promote`, `list-signals`, `list-datasets`, `online-sim` |
| `backend/src/experiment_harness/runner.py` | `ExperimentRunner`: sweep expansion, signal evaluation, result persistence |
| `backend/src/experiment_harness/results_db.py` | Parquet-backed run storage: `query_best()`, `query_runs()` |
| `backend/src/experiment_harness/eval_engine.py` | TP/SL backtesting: walk-forward evaluation per signal, cooldown, timeout |
| `backend/src/experiment_harness/grid_generator.py` | Cartesian parameter grid expansion for sweep axes |
| `backend/src/experiment_harness/online_simulator.py` | Per-bin latency profiling for signal implementations |
| `backend/src/experiment_harness/dataset_registry.py` | Discovers available datasets in `vp_immutable/` |
| `backend/src/experiment_harness/signals/statistical/derivative.py` | Canonical offline derivative signal (shares math via `vp_shared`) |
| `backend/src/experiment_harness/tracking.py` | MLflow integration |
| `backend/src/data_eng/` | Legacy data engineering (Bronze→Silver→Gold pipelines) — not used by VP runtime |
| `backend/scripts/run_vacuum_pressure.py` | CLI entry: argparse → `create_app()` → uvicorn |
| `backend/scripts/analyze_vp_signals.py` | Offline regime analysis of VP signal outputs |
| `backend/scripts/cache_vp_output.py` | Pre-warm server caches for a date range |
| `backend/scripts/warm_cache.py` | Pre-build book state pkl cache files |
| `backend/scripts/benchmark_vp_core.py` | Throughput benchmark for event engine + stream pipeline |
| `backend/scripts/build_gold_dataset_campaign.py` | Batch dataset generation from gold campaign YAMLs |
| `backend/scripts/publish_vp_research_dataset.py` | Dataset publication/export utilities |
| `backend/scripts/batch_download_futures.py` | Databento batch download for futures .dbn files |
| `frontend/src/vacuum-pressure.ts` | Streaming visualization: Arrow IPC parsing, heatmap, depth profile, signal gauges, projection rendering |
| `frontend/src/experiments.ts` | Experiment browser: table, filters, detail panel, run launch |

### Config YAMLs (changes frequently)

```
backend/lake/research/vp_harness/configs/
├── pipelines/        # PipelineSpec YAMLs (dataset definitions)
├── serving/          # ServingSpec YAMLs (experiment defaults)
├── serving_versions/ # Immutable PublishedServingSpec versions (live runtime)
├── experiments/      # ExperimentSpec YAMLs (sweep definitions)
└── gold_campaigns/   # Batch dataset campaigns
```

---

## Data Locations

| Path | Contents |
|------|----------|
| `backend/lake/raw/source=databento/product_type=<product_type>/symbol=<root>/table=market_by_order_dbn/` | Raw .dbn replay files (immutable) |
| `backend/lake/cache/vp_book/` | Book state pkl checkpoints (SHA256-keyed, auto-built) |
| `backend/lake/research/vp_immutable/<dataset_id>/` | Immutable datasets: `bins.parquet`, `grid_clean.parquet`, `manifest.json`, `checksums.json` |
| `backend/lake/research/vp_harness/results/` | `runs_meta.parquet` + `runs.parquet` (experiment results) |
| `backend/lake/research/vp_harness/configs/` | Experiment config YAMLs (pipelines / serving / experiments) |
| `backend/lake/research/vp_harness/configs/serving_versions/` | Immutable serving version specs consumed by live stream runtime |
| `backend/lake/research/vp_harness/serving_registry.sqlite` | Alias -> serving_id mapping, immutable version metadata, promotion audit |

`grid_clean.parquet` excludes `flow_score` and `flow_state_code` (model-agnostic). Harness derives these from `composite_d1/d2/d3` using manifest scoring config when legacy signals request them.

---

## Runtime/Signal Source Of Truth

Config-dependent values are intentionally not hardcoded in this README.

- Physics constants, tau/force coefficients, warmup, state-model, and projection parameters come from immutable PublishedServingSpec runtime snapshots + stream `runtime_config` payload (effective hash included).
- Two-force/scoring/state logic definitions are authoritative in code: `backend/src/vacuum_pressure/event_engine.py`, `backend/src/vacuum_pressure/scoring.py`, and `backend/src/vp_shared/derivative_core.py`.
- Active signal names and parameters are config-driven; use serving/experiment YAMLs and `uv run python -m src.experiment_harness.cli list-signals` to inspect what is available.
- Per-run provenance is in MLflow tracking metadata and harness artifacts under `backend/lake/research/vp_harness/results/`.

---

## Verification

```bash
# All backend tests (from backend/)
cd backend
uv run pytest tests/

# Parity-critical tests
uv run pytest tests/test_derivative_train_serve_parity.py tests/test_scoring_equivalence.py

# Frontend typecheck
cd frontend && npx tsc --noEmit
```

---

## Archived Code

Inactive files preserved under `dep/` with original path structure:
- `dep/backend/scripts/batch_download_equities.py`
- `dep/backend/src/experiment_harness/comparison.py`
- `dep/backend/lake/research/vp_experiments/`

---

## Architecture Notes

- Train/serve parity is enforced by `vp_shared/` — `scoring.py` and `derivative.py` share identical math, no divergence possible.
- Book cache (`lake/cache/vp_book/`) is keyed by `sha256(product_type:symbol:dt:warmup_start_ns:mtime_ns:size)[:16]`. First run builds it; subsequent runs load it, reanchor to BBO, and sync rest depth.
- `reanchor_to_bbo()` MUST be called after book import — overnight anchor can be $300+ off from RTH price, causing all price events to map to None.
- Snapshot/clear cycles at 09:30 ET temporarily zero BBO. Skip `mid=0` bins in evaluation.
- `grid_clean.parquet` uses SHA256 checksums for integrity. `manifest.json` carries both legacy top-level and nested `source_manifest`/`spec` layouts for backward compat.
- Stream sessions are built directly from immutable published `runtime_snapshot` via `build_config_from_mapping()`, then validated against instrument lock by `resolve_config(product_type, symbol)`.
- `build_config_with_overrides()` is used by config authoring/harness layers to derive new configs from a base runtime config; it recomputes `config_version` and `projection_horizons_ms` and fails fast on unknown keys.
