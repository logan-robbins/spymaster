# Spymaster — qMachina Research Platform

Replays Databento MBO `.dbn` files through an order-book physics engine, emits fixed-bin dense grids over WebSocket, and exposes a full platform for signal authoring, experiment orchestration, model promotion, and serving lifecycle management.

---

## System Diagram

```
+------------------------------------------------------------------------------------+
|                            LIVE SERVING PATH                                       |
|                                                                                    |
|  instrument.yaml ──► RuntimeConfig                                                 |
|                              │                                                     |
|  .dbn files ──► iter_mbo_events ──► AbsoluteTickEngine                             |
|                                          │  two-force model (pressure/vacuum)      |
|                                          │  EMA derivative chain (v, a, j)         |
|                                          ▼                                         |
|  book cache pkl ──► reanchor_to_bbo ──► SILVER STREAM (Arrow IPC)                 |
|                         (every cell_width_ms)                                      |
|                                          │                                         |
|                                          ▼                                         |
|                              FastAPI  /v1/stream (WebSocket)                       |
|                              ┌─────────────────────────────┐                      |
|                              │ 1. runtime_config JSON (once)│                      |
|                              │ 2. grid_update JSON + Arrow  │                      |
|                              │    IPC binary (per bin)      │                      |
|                              └──────────────┬──────────────┘                      |
|                                             ▼                                      |
|                                  Vite Frontend                                     |
|                                  GoldFeatureRuntime (browser gold)                 |
|                                  heatmap · gauges · overlays                       |
+------------------------------------------------------------------------------------+

+------------------------------------------------------------------------------------+
|                         EXPERIMENT + PROMOTION PATH                                |
|                                                                                    |
|  ExperimentSpec YAML ──► ServingSpec YAML ──► PipelineSpec YAML                   |
|  (sweep/eval/TP-SL)       (scoring/signal)     (grid/features)                    |
|         │                                                                          |
|         ▼                                                                          |
|  [CLI] cli.py generate ──► stream_events() ──► lake/research/datasets/<id>/       |
|                                                 bins.parquet · grid_clean.parquet  |
|                                                 manifest.json · checksums.json     |
|         │                                                                          |
|         ▼                                                                          |
|  [CLI] cli.py run ──► ExperimentRunner                                             |
|  [API] /v1/jobs/experiments ──► JobQueue ──► experiment_job_runner.py             |
|                                  (Redis Streams / asyncio fallback)                |
|                                          │                                         |
|                                   ResultsDB (parquet)                              |
|                                   SSE events ──► frontend/jobs.html               |
|                                          │                                         |
|         ┌──────────────────────────────────────────────────────┐                  |
|         │              PROMOTION (two paths)                    │                  |
|         │  [CLI]  cli.py promote                                │                  |
|         │  [API]  /v1/modeling/sessions/{id}/promote           │                  |
|         │         (7-step gated wizard: dataset → gold →       │                  |
|         │          signal → eval → experiment → review →       │                  |
|         │          promotion)                                   │                  |
|         └──────────────────────┬───────────────────────────────┘                  |
|                                ▼                                                   |
|  PublishedServingSpec ──► serving_registry.sqlite (alias → serving_id)            |
|                           serving_versions/<serving_id>.yaml (immutable)           |
|                                ▼                                                   |
|  ws://.../v1/stream?serving=<alias_or_id>  ──► live stream (above)                |
+------------------------------------------------------------------------------------+

+------------------------------------------------------------------------------------+
|                         PLATFORM CONTROL PLANE                                     |
|                                                                                    |
|  Postgres (async SQLAlchemy 2.x)                                                   |
|  ┌─────────────────────────────────────────────────────────┐                      |
|  │ workspace · workspace_member · experiment_job           │                      |
|  │ job_event · job_artifact · modeling_session             │                      |
|  │ modeling_step_state · ingestion_live_session            │                      |
|  │ serving_activation · audit_event                        │                      |
|  └─────────────────────────────────────────────────────────┘                      |
|    ▲                                                                               |
|    │  FastAPI control-plane APIs (all under /v1/...)                               |
|    │  api_jobs.py · api_modeling.py · api_serving.py · api_gold_dsl.py            |
|    │                                                                               |
|  Gold DSL ──► GoldDslSpec (DAG) ──► validate_dsl() ──► execute_dsl_preview()     |
|              compat.py: GoldFeatureConfig ◄──► GoldDslSpec (bidirectional)        |
|                                                                                    |
|  Serving Lifecycle                                                                 |
|  serving_diff.py ──► /v1/serving/diff (snapshot compare)                          |
|  /v1/serving/aliases/{alias}/activate ──► alias repoint (rollback)                |
+------------------------------------------------------------------------------------+
```

---

## Config Chain

Each layer references the one below by name:

```
ExperimentSpec  ──►  ServingSpec  ──►  PipelineSpec
(sweep/eval)         (scoring/signal)   (grid/features)
```

Models:
- `backend/src/qmachina/experiment_config.py` — ExperimentSpec
- `backend/src/qmachina/serving_config.py` — ServingSpec (`display_name`, `model_id`, `visualization`), OverlaySpec (`label`, `type`, `params`), VisualizationConfig, EmaConfig (overlay param validation only)
- `backend/src/qmachina/pipeline_config.py` — PipelineSpec
- `backend/src/qmachina/config.py` — RuntimeConfig, `resolve_config()`, `build_config_with_overrides()`

Config YAMLs live under `backend/lake/research/harness/configs/` (mutable, frequently changing):
- `pipelines/` — PipelineSpec
- `serving/` — ServingSpec (mutable authoring surface)
- `serving_versions/` — PublishedServingSpec (immutable, read at runtime)
- `experiments/` — ExperimentSpec
- `gold_campaigns/` — batch dataset campaigns

---

## Stage Boundary (Silver → Gold)

**Silver** (wire, lowest latency): mass fields, EMA v/a/j chains, BBO permutation labels.
Allow-list: `SILVER_COLS` in `backend/src/qmachina/stage_schema.py`. Sent over WebSocket as Arrow IPC.

**Gold** (computed): VP force block (`pressure_variant`, `vacuum_variant`, `composite`, `composite_d{1,2,3}`, `state5_code`), spectrum scoring. Computed in browser by `GoldFeatureRuntime` or offline via `generate-gold` CLI.

Adding a field to the stream requires updating both `SILVER_COLS` and the TypeScript `GridBucketRow` interface.

---

## Package Structure

```
backend/src/
  shared/                         # hashing · yaml_io · zscore · derivative_core
  models/
    registry.py                   # get_async_stream_events(model_id), get_build_model_config(model_id)
    vacuum_pressure/              # VP: event_engine · core_pipeline · spectrum · scoring · stream_pipeline
    ema_ensemble/                 # EMA: single-row bin emission, BBO tracking, no LOB grid
  qmachina/                       # Platform infrastructure
    app.py                        # FastAPI composition root — all routes registered here
    config.py                     # RuntimeConfig, instrument lock enforcement
    serving_config.py             # ServingSpec, PublishedServingSpec, StreamFieldSpec, StreamFieldRole enum
    serving_registry.py           # SQLite alias → serving_id map + immutable version store
    serving_diff.py               # diff_runtime_snapshots() — structured snapshot comparison
    stream_contract.py            # build_runtime_config_payload(), build_grid_update_payload(), grid_schema()
    stage_schema.py               # SILVER_COLS allow-list
    gold_config.py                # GoldFeatureConfig (c1..c7 + flow params)
    validate_configs.py           # Config validation CLI + validate_serving_spec() (used by promotion preflight)
    modeling_session_service.py   # 7-step gated ModelingSession state machine
    engine_factory.py             # create_absolute_tick_engine(config) — shared by all models
    book_cache.py                 # ensure_book_cache(...) — shared book state pkl cache
    stream_time_utils.py          # compute_time_boundaries, resolve_tick_int
    async_stream_wrapper.py       # make_async_stream_events(sync_fn), ProducerLatencyConfig
    api_stream.py                 # WebSocket /v1/stream
    api_experiments.py            # GET /v1/experiments/runs
    api_jobs.py                   # /v1/jobs/experiments (submit · list · cancel · SSE events · artifacts)
    api_modeling.py               # /v1/modeling/sessions (wizard CRUD · step commit · promote · preview)
    api_serving.py                # /v1/serving/versions · /v1/serving/aliases (lifecycle + rollback)
    api_gold_dsl.py               # /v1/gold/validate · preview · compare · from_legacy
    db/
      engine.py                   # create_async_engine, get_db_session() context manager
      base.py                     # DeclarativeBase
      models.py                   # 10 ORM tables (workspace through audit_event + job_artifact)
      repositories.py             # WorkspaceRepository · ExperimentJobRepository · JobArtifactRepository
                                  # ModelingSessionRepository · IngestionSessionRepository · AuditRepository
  gold_dsl/
    schema.py                     # DslNode union (SilverRef · TemporalWindow · SpatialNeighborhood · ArithmeticExpr · NormExpr · OutputNode), GoldDslSpec
    validate.py                   # validate_dsl() — Kahn cycle detection + ref integrity + SILVER_COLS check
    preview.py                    # execute_dsl_preview() — topological execution against sample parquets
    compat.py                     # gold_config_to_dsl() / dsl_to_gold_config() — bidirectional mapper
  jobs/
    queue.py                      # JobQueue — Redis Streams producer/consumer, asyncio.Queue fallback
    experiment_job_runner.py      # Async wrapper: asyncio.to_thread, cancel flags, artifact persistence
    worker.py                     # Background worker loop — polls queue, dispatches jobs
  experiment_harness/
    cli.py                        # Click CLI: generate · generate-gold · run · compare · promote · online-sim
    runner.py                     # ExperimentRunner — sweep expansion, signal eval, result persistence
    eval_engine.py                # TP/SL evaluation with cooldown
    gold_builder.py               # Offline gold feature computation
    dataset_registry.py           # Dataset metadata + path resolution
    results_db.py                 # Append-only parquet results store
    signals/                      # Signal implementations (statistical + ML)
    feature_store/                # Feast offline feature store (opt-in, enabled per spec)
  data_eng/                       # Bronze/Silver/Gold pipelines (futures + equities)
```

---

## Data Locations

| Path | Contents | Mutability |
|------|----------|------------|
| `backend/lake/raw/source=databento/product_type=<t>/symbol=<root>/table=market_by_order_dbn/` | Raw `.dbn` files | Immutable |
| `backend/lake/cache/book_engine/` | SHA256-keyed book state pkl checkpoints | Regenerable |
| `backend/lake/research/datasets/<dataset_id>/` | `bins.parquet` · `grid_clean.parquet` · `manifest.json` · `checksums.json` · `gold_grid.parquet` (optional) | Immutable once written |
| `backend/lake/research/harness/results/` | `runs_meta.parquet` · `runs.parquet` | Append-only |
| `backend/lake/research/harness/configs/` | YAML config tree | Mutable |
| `backend/lake/research/harness/configs/serving_versions/` | Immutable PublishedServingSpec files | Immutable |
| `backend/lake/research/harness/serving_registry.sqlite` | Alias → serving_id mapping + promotion audit | Append-only |
| `backend/lake/research/feature_store/` | Feast offline store root (auto-created on first sync) | Regenerable |

---

## Source Map — Entry Points

| File | Role |
|------|------|
| `backend/scripts/run_server.py` | Server entry → `create_app()` → uvicorn |
| `backend/src/qmachina/app.py` | FastAPI composition root: all routers registered here |
| `backend/src/experiment_harness/cli.py` | Click CLI (generate / run / promote / compare) |
| `backend/src/experiment_harness/runner.py` | ExperimentRunner — compute core, never modified by orchestration layer |
| `backend/scripts/register_serving.py` | Register ServingSpec directly (no experiment run) |
| `backend/scripts/validate_e2e.py` | Full offline E2E pipeline validator (config → dataset → gold → experiment → serving) |
| `backend/scripts/analyze_signals.py` | Offline regime analysis |
| `backend/scripts/warm_cache.py` | Pre-build book state pkl cache |
| `backend/scripts/batch_download_futures.py` | Databento 3-phase batch download daemon |
| `frontend/src/stream.ts` | Live stream viewer — heatmap (VP/grid models) + candle (EMA/indicator models) |
| `frontend/src/experiments.ts` | Experiment browser |
| `frontend/src/jobs.ts` | Job queue monitor (SSE progress) |
| `frontend/src/model_studio.ts` | Model Studio wizard (7-step) |
| `frontend/src/serving_registry.ts` | Serving version management + rollback |

---

## Architectural Invariants

- **Instrument lock**: one process = one instrument. `instrument.yaml` is the single source of truth. `resolve_config()` enforces it at startup — mismatches fail fast. Override via `QMACHINA_INSTRUMENT_CONFIG_PATH`.
- **`reanchor_to_bbo()` is mandatory** after every book cache import. Overnight anchor diverges from RTH price, mapping all events to `None` without it.
- **Silver is wire-only**: gold features are never transmitted over WebSocket. Browser computes gold from silver fields via `GoldFeatureRuntime`.
- **Serving versions are immutable**: `PublishedServingSpec` files and their `runtime_snapshot` are never modified after writing. Rollback = alias repoint, not record mutation.
- **Promotion preflight is bypass-proof**: `validate_serving_spec_preflight()` is called inside `ServingRegistry.promote()` before any DB writes. It cannot be bypassed via direct API calls.
- **ExperimentRunner is never modified by orchestration**: `experiment_job_runner.py` wraps it via `asyncio.to_thread`. The runner code path is identical for CLI and API-submitted jobs.
- **Train/serve parity**: `backend/src/shared/derivative_core.py` and `backend/src/shared/zscore.py` are shared between offline signal computation and live scoring. Same math, same path.
- **Config CI gate**: `.github/workflows/validate-configs.yml` runs `python -m src.qmachina.validate_configs` on every push/PR. Invalid YAMLs block merge.
- **Snap/clear at 09:30 ET**: temporarily zeros BBO. Skip `mid=0` bins in evaluation.
- **Book cache key**: `sha256(product_type:symbol:dt:warmup_start_ns:mtime_ns:size)[:16]`. Deterministic — first run builds, subsequent runs load + reanchor.
- **DSL acyclicity**: `validate_dsl()` uses Kahn's algorithm. Any cycle is a hard error — no exceptions.
- **Postgres is optional**: app starts without it. `_probe_control_plane_db()` in `app.py` logs a warning and continues. SQLite serving registry and WebSocket streaming are unaffected.
- **Redis is optional**: `JobQueue` falls back to `asyncio.Queue` when `REDIS_URL` is unset or unreachable. Jobs are lost on restart in fallback mode.

---

## Gotchas

- `SILVER_INT_COL_DTYPES` in `stream_pipeline.py` includes `"k"` — the initialization loop explicitly skips `"k"` to avoid overwriting the pre-set tick range array.
- `depth_qty_rest` must be clamped to `min(rest, end)` — enforced in all 4 data engineering pipelines.
- `pull_qty_rest` is always 0 for `future_option_mbo` and `equity_option_cmbp_1`.
- `fill_qty` is always 0 for `equity_option_cmbp_1` (CMBP-1 has no trade data).
- Feast requires `uvicorn>=0.34` — project constraint relaxed to accommodate. Feature store is opt-in per spec (`feature_store: {enabled: true}`).
- pyarrow was downgraded 23→22 by feast; iterate schema via `for field in schema` (no `.num_fields`).
- `uv` environments limited to `sys_platform == 'darwin'` in `pyproject.toml` (feast/linux conflict).
- EMA `alpha` is required when `agg="ewm"` in Gold DSL `TemporalWindow` nodes. Missing alpha is a validation error.
- `ScoringConfig` fields in EMA YAML must use valid (non-zero, non-one) values or be omitted entirely to use defaults.
