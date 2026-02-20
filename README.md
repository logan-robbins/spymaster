# Spymaster

Futures-only vacuum-pressure system: live streaming server, experiment harness, and frontend visualization.

## Instrument Lock

Single instrument defined in `backend/src/vacuum_pressure/instrument.yaml` (currently MNQH6). All runtime defaults live here.

## Config Architecture

Three-layer pydantic v2 config hierarchy. Each layer references the one below it by name.

```
ExperimentSpec  ->  ServingSpec  ->  PipelineSpec
(sweep/eval)        (scoring/signal)   (grid/features)
```

Config models:
- `backend/src/vacuum_pressure/pipeline_config.py` — PipelineSpec: raw .dbn replay to grid of engineered features (no scoring)
- `backend/src/vacuum_pressure/serving_config.py` — ServingSpec: references a PipelineSpec + scoring + signal + projection params
- `backend/src/vacuum_pressure/experiment_config.py` — ExperimentSpec: references a ServingSpec + sweep axes + TP/SL eval + tracking
- `backend/src/vacuum_pressure/scoring.py` — SpectrumScorer: single Python implementation used by both server and harness (zero train/serve skew)
- `backend/src/vp_shared/` — canonical shared core for deterministic hashing, YAML model loading, robust z-score primitives, and derivative state-model math used by both `experiment_harness` and `vacuum_pressure`

Runtime config loading and validation: `backend/src/vacuum_pressure/config.py`

Config YAMLs on disk:
- `backend/lake/research/vp_harness/configs/pipelines/` — PipelineSpec YAMLs (capture window + engine params)
- `backend/lake/research/vp_harness/configs/serving/` — ServingSpec YAMLs (scoring + signal + projection params)
- `backend/lake/research/vp_harness/configs/experiments/` — ExperimentSpec YAMLs (sweep axes + TP/SL eval + tracking)
- `backend/lake/research/vp_harness/configs/gold_campaigns/` — batch dataset generation campaigns

## Environment

```bash
# Install (once)
cd backend && uv sync
cd frontend && npm ci
```

Backend: Python 3.12, `uv` exclusively. Frontend: Node + npm, Vite dev server on port 5174.

## Start Backend

```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
cd backend
nohup uv run scripts/run_vacuum_pressure.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --port 8002 \
  --start-time 09:25 > /tmp/vp_backend.log 2>&1 &
```

Health check: `curl -s http://localhost:8002/health`

State model overrides (pass to the run script):
- `--state-model-zscore-window-bins`, `--state-model-zscore-min-periods`
- `--state-model-d1-weight`, `--state-model-d2-weight`, `--state-model-d3-weight`

Full flag list: `cd backend && uv run scripts/run_vacuum_pressure.py --help`

## Start Frontend

```bash
kill $(lsof -t -iTCP:5174) 2>/dev/null
cd frontend
nohup npm run dev > /tmp/vp_frontend.log 2>&1 &
```

Pages (Vite multi-page, port 5174):
- `http://localhost:5174/vacuum-pressure.html` — live streaming visualization
- `http://localhost:5174/experiments.html` — experiment browser (requires backend on :8002)

## WebSocket Streaming

Endpoint:

```
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25
```

With a promoted serving config:

```
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25&serving=<serving_name>
```

`?serving=name` loads ServingSpec YAML from `configs/serving/`, applies scoring/signal/projection params, streams server-computed `flow_score`/`flow_state_code`. Explicit state-model URL params override serving config values.

Message frames:
- text: `runtime_config` JSON (includes `serving` block when active)
- text per bin: `grid_update` JSON
- binary per bin: Arrow IPC dense grid rows

Frontend URL query params:
- `product_type`, `symbol`, `dt`, `start_time` — required instrument params
- `serving=<name>` — load ServingSpec for server-side scoring
- state-model override params (`state_model_enabled`, `state_model_d1_weight`, etc.) — forwarded to WebSocket
- `projection_source=backend|frontend` — projection band source (default: backend)
- `dev_scoring=true` — enable client-side ADS/PFP/SVac composite (dev only, disabled by default). Requires grid coverage at least `k=-23..+23` (backend `grid_radius_ticks >= 23`).

## Experiment Workflow

All commands run from `backend/`.

### 1. Generate dataset (from PipelineSpec)

```bash
uv run python -m src.experiment_harness.cli generate <pipeline_spec.yaml>
```

Idempotent: skips if hash-addressed output exists under `lake/research/vp_immutable/`.
Output contract:
- `bins.parquet` = one row per emitted bin (`bin_seq`, `ts_ns`, `mid_price`, bin boundaries, and book metadata).
- `grid_clean.parquet` = per-bin per-k rows.

### 2. Run experiment (from ExperimentSpec)

```bash
uv run python -m src.experiment_harness.cli run <experiment_spec.yaml>
```

Auto-generates the referenced pipeline's dataset if missing. Resolves ExperimentSpec -> ServingSpec -> PipelineSpec. Expands sweep axes, evaluates TP/SL, persists results.
Runner behavior is fail-fast: any spec error aborts the experiment run.

### 3. Compare results

```bash
uv run python -m src.experiment_harness.cli compare --min-signals 5
```

### 4. Promote winner to ServingSpec

```bash
uv run python -m src.experiment_harness.cli promote <experiment_spec.yaml> --run-id <winner_run_id>
```

Writes a new ServingSpec YAML to `configs/serving/`. Prints runtime overrides and serving URL (`ws://localhost:8002/v1/vacuum-pressure/stream?serving=<name>`).

### 5. Stream with promoted config

```
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25&serving=<promoted_name>
```

### Other CLI commands

```bash
uv run python -m src.experiment_harness.cli list-signals
uv run python -m src.experiment_harness.cli list-datasets
uv run python -m src.experiment_harness.cli online-sim --signal <name> --dataset-id <id> --bin-budget-ms 100
```

Harness internals: `backend/src/experiment_harness/README.md`

## Experiment Browser

REST API endpoints (served by VP server on :8002):
- `GET /v1/experiments/runs?signal=&dataset_id=&sort=tp_rate&min_signals=5&top_n=50`
- `GET /v1/experiments/runs/{run_id}/detail`

Streaming URL is constructed from `manifest.json` instrument metadata + signal params mapped to state-model WebSocket query params.
Manifest parsing accepts both legacy top-level instrument fields and nested `source_manifest`/`spec` layouts.
Derivative launch URL building is strict: unknown derivative params are rejected (`can_stream=false`) to avoid silent partial mappings.

## Data Locations

Raw replay input:
- `backend/lake/raw/source=databento/product_type=future_mbo/symbol=<root>/table=market_by_order_dbn/` — .dbn files per symbol

Immutable experiment datasets:
- `backend/lake/research/vp_immutable/<dataset_id>/` — per dataset: `bins.parquet`, `grid_clean.parquet`, `manifest.json`, `checksums.json`

Harness results:
- `backend/lake/research/vp_harness/results/` — `runs_meta.parquet` and `runs.parquet`

## Archived Python Paths

Inactive and legacy Python files are archived under `dep/` with original path structure preserved.

Current archived examples:
- `dep/backend/scripts/batch_download_equities.py`
- `dep/backend/src/experiment_harness/comparison.py`
- `dep/backend/lake/research/vp_experiments/mnqh6_20260206_0925_1025/`

## System Map

Live serving:
1. `backend/scripts/run_vacuum_pressure.py` — CLI entry, starts FastAPI
2. `backend/src/vacuum_pressure/server.py` — WebSocket + REST endpoints, Arrow IPC framing
3. `backend/src/vacuum_pressure/stream_pipeline.py` — per-bin event processing, grid emission
4. `backend/src/vacuum_pressure/event_engine.py` — tick-level order book engine
5. `backend/src/vacuum_pressure/spectrum.py` — multi-window flow computation (module name retained)
6. `backend/src/vacuum_pressure/scoring.py` — z-score + tanh blend + state classification
7. `backend/src/vacuum_pressure/runtime_model.py` — incremental state-model scoring on `state5_code`
8. `backend/src/vp_shared/` — shared train/serve primitives consumed by items 3, 6, 7 and by offline harness signals/config loaders

Frontend:
1. `frontend/src/vacuum-pressure.ts` — streaming visualization, Arrow IPC parsing, projection rendering
2. `frontend/src/experiments.ts` — experiment browser table, filters, detail panel, launch logic
3. `frontend/src/experiment-engine.ts` — client-side ADS/PFP/SVac composite (dev mode only)

Offline experiment:
1. `backend/src/experiment_harness/cli.py` — Click CLI (generate, run, compare, promote, list-signals, list-datasets, online-sim)
2. `backend/src/experiment_harness/runner.py` — sweep expansion, signal evaluation, result persistence
3. `backend/src/experiment_harness/config_schema.py` — internal runner schema (not user-facing, produced by ExperimentSpec.to_harness_config())
4. `backend/src/experiment_harness/signals/` — signal implementations (`statistical/` and `ml/` subdirs), with `statistical/derivative.py` sharing canonical runtime math via `src/vp_shared`
5. `backend/src/experiment_harness/results_db.py` — parquet-backed run storage and query
6. `backend/src/experiment_harness/tracking.py` — MLflow tracking integration

## Verification

```bash
# Backend tests (from backend/)
cd backend
uv run pytest tests/

# Focused train/serve parity checks
uv run pytest \
  tests/test_derivative_train_serve_parity.py \
  tests/test_scoring_equivalence.py

# Frontend typecheck (from frontend/)
cd frontend
npx tsc --noEmit
```

## Architecture Review Notes

- Current deep-dive findings and the PhD-grade robust-standardization migration plan are documented in `ANALYSIS.md`.
- The target direction is a single canonical online/offline implementation for robust standardization with strict fail-fast parity and latency gates.

## Check Data Before Starting

```bash
# Raw .dbn replay files
find backend/lake/raw/source=databento/product_type=future_mbo -maxdepth 3 -type d -name 'table=market_by_order_dbn'

# Immutable datasets
find backend/lake/research/vp_immutable -maxdepth 2 -type f -name bins.parquet
```
