# Spymaster

qMachina research platform. Replays Databento MBO .dbn files through an order-book physics engine, emits fixed-bin dense grids over WebSocket, and exposes an offline experiment harness for signal tuning and promotion.

---

## System Diagrams

```
+-----------------------------------------------------------------------------+
|                           LIVE SERVING PATH                                 |
|                                                                             |
|  instrument.yaml --> RuntimeConfig                                          |
|                            |                                                |
|  .dbn files --> iter_mbo_events --> AbsoluteTickEngine (n_absolute_ticks)   |
|                                          |  two-force model (pressure/vacuum)|
|                                          |  EMA derivative chain (v, a, j)  |
|                                          v                                  |
|  [warmup phase] -- book cache pkl --> reanchor_to_bbo                       |
|                                          |                                  |
|                                          v (every cell_width_ms)            |
|                                   grid_snapshot_arrays()                    |
|                                          |                                  |
|                                          v                                  |
|                              IndependentCellSpectrum                        |
|                         (flow windows, rollup, tanh-z-score)               |
|                                          |                                  |
|                                          v                                  |
|                               DerivativeRuntime                             |
|                          (state5_code -> z-score -> tanh blend)             |
|                                          |                                  |
|                                          v                                  |
|                              +--------------------+                         |
|                              |  FastAPI WebSocket  |                        |
|                              |    /v1/stream       |                        |
|                              +--------+-----------+                         |
|                                       | runtime_config JSON (once)          |
|                                       | grid_update JSON (per bin)          |
|                                       | Arrow IPC binary (per bin)          |
|                                       v                                     |
|                              +--------------------+                         |
|                              |   Vite Frontend     |                        |
|                              |  vp-stream          | heatmap + gauges      |
|                              |  experiments        | result browser         |
|                              +--------------------+                         |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
|                        EXPERIMENT HARNESS PATH                              |
|                                                                             |
|  ExperimentSpec YAML --> ServingSpec YAML --> PipelineSpec YAML              |
|  (sweep/eval/TP-SL)       (scoring/signal)     (grid/features)              |
|         |                       |                     |                     |
|         +-------------------------------------------+                       |
|                                 |                                           |
|                         resolve_runtime_config()                            |
|                                 |                                           |
|                          stream_events() --> bins.parquet                   |
|                                              grid_clean.parquet             |
|                                              manifest.json                  |
|                                              checksums.json                 |
|                                 |                                           |
|                          ExperimentRunner                                   |
|                          (sweep x signal x cooldown x TP/SL)               |
|                                 |                                           |
|                         ResultsDB (parquet)                                 |
|                         MLflow tracking (optional)                          |
|                                 |                                           |
|           promote winner --> PublishedServingSpec + alias registry update    |
|                                 |                                           |
|              serve: ws://localhost:<port>/v1/stream?serving=<alias_or_id>   |
+-----------------------------------------------------------------------------+
```

---

## Setup

```bash
cd backend && uv sync
cd frontend && npm ci
```

---

## Operations

### Check data availability

```bash
# Raw .dbn replay files
find backend/lake/raw/source=databento/product_type=<product_type> -maxdepth 3 -type d -name 'table=market_by_order_dbn'

# Immutable experiment datasets
find backend/lake/research/datasets -maxdepth 2 -type f -name bins.parquet
```

### Start backend

```bash
cd backend
kill $(lsof -t -iTCP:8002) 2>/dev/null
nohup uv run scripts/run_server.py --port 8002 > /tmp/qm_backend.log 2>&1 &
```

Health check: `curl -s http://localhost:8002/health`

All flags: `uv run scripts/run_server.py --help`

### Start frontend

```bash
cd frontend
kill $(lsof -t -iTCP:5174) 2>/dev/null
nohup npm run dev > /tmp/qm_frontend.log 2>&1 &
```

- Live heatmap: `http://localhost:5174/vp-stream.html?serving=<alias_or_id>`
- Experiment browser: `http://localhost:5174/experiments.html`

### Generate dataset

```bash
cd backend
uv run python -m src.experiment_harness.cli generate <pipeline_spec.yaml>
```

Idempotent -- skips if hash-addressed output exists. Output under `lake/research/datasets/<dataset_id>/`: `bins.parquet`, `grid_clean.parquet`, `manifest.json`, `checksums.json`.

### Run experiment

```bash
cd backend
uv run python -m src.experiment_harness.cli run <experiment_spec.yaml>
```

Resolves ExperimentSpec -> ServingSpec -> PipelineSpec. Auto-generates dataset if missing. Expands sweep axes x signal params x cooldown x TP/SL. Results to `lake/research/harness/results/`.

### Compare results

```bash
cd backend
uv run python -m src.experiment_harness.cli compare --min-signals 5
uv run python -m src.experiment_harness.cli compare --signal derivative --sort tp_rate
```

### Promote winner

```bash
cd backend
uv run python -m src.experiment_harness.cli promote <experiment_spec.yaml> --run-id <winner_run_id> --alias <serving_alias>
```

Writes immutable PublishedServingSpec to `configs/serving_versions/` and updates `serving_registry.sqlite`.

### Stream with promoted config

```
http://localhost:5174/vp-stream.html?serving=<serving_alias_or_id>
```

Stream clients must pass `serving=<alias_or_id>`. Ad-hoc overrides rejected.

### List signals and datasets

```bash
cd backend
uv run python -m src.experiment_harness.cli list-signals
uv run python -m src.experiment_harness.cli list-datasets
```

### Online simulation (latency profiling)

```bash
cd backend
uv run python -m src.experiment_harness.cli online-sim --signal <name> --dataset-id <id> --bin-budget-ms 100
```

### Migrate lake paths

After updating code, run the lake migration script to rename directories:

```bash
cd backend
bash scripts/migrate_lake_paths.sh
```

### Run tests

```bash
cd backend && uv run pytest tests/
cd frontend && npx tsc --noEmit
```

---

## Instrument Lock

One environment = one instrument. Defaults: `backend/src/qmachina/instrument.yaml`.

Override: `QMACHINA_INSTRUMENT_CONFIG_PATH=/path/to/override.yaml` (legacy `VP_INSTRUMENT_CONFIG_PATH` still supported with deprecation warning).

`resolve_config()` in `config.py` enforces the lock -- mismatches fail fast.

---

## WebSocket Protocol

```
ws://localhost:8002/v1/stream?serving=<alias_or_id>
```

Extra query params rejected (close code 1008).

Message sequence: (1) text `runtime_config` JSON once, (2) per bin: text `grid_update` JSON + binary Arrow IPC.

Wire contract defined in `backend/src/qmachina/stream_contract.py` -- `build_runtime_config_payload()`, `build_grid_update_payload()`, `grid_schema()`.

---

## Config Chain

```
ExperimentSpec  ->  ServingSpec  ->  PipelineSpec
(sweep/eval)       (scoring/signal)   (grid/features)
```

Each layer references the one below by name. Config models:
- `backend/src/qmachina/pipeline_config.py` -- PipelineSpec
- `backend/src/qmachina/serving_config.py` -- ServingSpec, PublishedServingSpec, StreamFieldSpec
- `backend/src/qmachina/experiment_config.py` -- ExperimentSpec
- `backend/src/qmachina/config.py` -- RuntimeConfig, `resolve_config()`, `build_config_from_mapping()`, `build_config_with_overrides()`

Config YAMLs (mutable, change frequently):

```
backend/lake/research/harness/configs/
  pipelines/        # PipelineSpec
  serving/          # ServingSpec
  serving_versions/ # Immutable PublishedServingSpec (live runtime)
  experiments/      # ExperimentSpec
  gold_campaigns/   # Batch dataset campaigns
```

---

## Experiment Browser REST API

- `GET /v1/experiments/runs?signal=&dataset_id=&sort=tp_rate&min_signals=5&top_n=50`
- `GET /v1/experiments/runs/{run_id}/detail`

Runs without promoted serving alias return `can_stream=false`.

---

## Source Map

Entry points that compose the system. Each imports branch/leaf modules -- read these to trace any subsystem.

| Entry Point | Role |
|-------------|------|
| `backend/scripts/run_server.py` | Server entry: argparse -> `create_app()` -> uvicorn |
| `backend/src/qmachina/app.py` | FastAPI composition root: middleware + health + routes |
| `backend/src/experiment_harness/cli.py` | Click CLI: generate, run, compare, promote, list-signals, list-datasets, online-sim |
| `backend/src/experiment_harness/runner.py` | ExperimentRunner: sweep expansion, signal evaluation, result persistence |
| `backend/scripts/analyze_signals.py` | Offline regime analysis |
| `backend/scripts/warm_cache.py` | Pre-build book state pkl cache |
| `backend/scripts/cache_output.py` | Pre-warm server caches for a date range |
| `backend/scripts/benchmark_core.py` | Throughput benchmark |
| `backend/scripts/build_gold_dataset_campaign.py` | Batch dataset generation |
| `backend/scripts/publish_dataset.py` | Dataset publication |
| `backend/scripts/batch_download_futures.py` | Databento batch download daemon |
| `frontend/src/vp-stream.ts` | Live heatmap + gauges |
| `frontend/src/experiments.ts` | Experiment browser UI |

---

## Package Structure

```
backend/src/
  shared/                     # Reusable utilities (hashing, yaml_io, zscore, derivative_core)
  models/
    vacuum_pressure/          # VP model: event engine, core pipeline, spectrum, scoring, runtime model, stream pipeline
  qmachina/                   # Platform infrastructure: config, serving, stream contract, API routes, app
  experiment_harness/         # Offline experiment CLI, runner, eval engine, signals, dataset registry
  data_eng/                   # Bronze/Silver/Gold data engineering pipelines
```

---

## Data Locations

| Path | Contents |
|------|----------|
| `backend/lake/raw/source=databento/product_type=<product_type>/symbol=<root>/table=market_by_order_dbn/` | Raw .dbn replay files (immutable) |
| `backend/lake/cache/book_engine/` | Book state pkl checkpoints (SHA256-keyed, auto-built) |
| `backend/lake/research/datasets/<dataset_id>/` | Immutable datasets: `bins.parquet`, `grid_clean.parquet`, `manifest.json`, `checksums.json` |
| `backend/lake/research/harness/results/` | `runs_meta.parquet` + `runs.parquet` (experiment results) |
| `backend/lake/research/harness/configs/` | Experiment config YAMLs (pipelines / serving / experiments) |
| `backend/lake/research/harness/configs/serving_versions/` | Immutable serving version specs consumed by live stream runtime |
| `backend/lake/research/harness/serving_registry.sqlite` | Alias -> serving_id mapping, immutable version metadata, promotion audit |

---

## Gotchas

- `reanchor_to_bbo()` MUST be called after book import -- overnight anchor can be $300+ off from RTH price, causing all events to map to None.
- Snapshot/clear cycles at 09:30 ET temporarily zero BBO. Skip `mid=0` bins in evaluation.
- Book cache keyed by `sha256(product_type:symbol:dt:warmup_start_ns:mtime_ns:size)[:16]`. First run builds; subsequent runs load + reanchor.
- Train/serve parity enforced by `shared/` -- `scoring.py` and `derivative_core.py` share identical math.
- Stream sessions built from immutable `runtime_snapshot` via `build_config_from_mapping()`, validated against instrument lock.

---

## Archived Code

Inactive files under `dep/` with original path structure.
