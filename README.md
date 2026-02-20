# Spymaster

Canonical runbook for the live MNQ vacuum-pressure system and its experiment harness.

This document is intentionally operational:
- how to check data
- how to start backend/frontend
- how serving works
- where immutable datasets and results live
- how to promote a harness configuration into the frontend prediction algorithm

## Scope

Product focus is futures only.

Primary runtime instrument is locked by:
- `backend/src/vacuum_pressure/instrument.yaml`

Runtime config loading and validation:
- `backend/src/vacuum_pressure/config.py`

## System Map

Live serving path:
1. `backend/scripts/run_vacuum_pressure.py`
2. `backend/src/vacuum_pressure/server.py`
3. `backend/src/vacuum_pressure/stream_pipeline.py`
4. `backend/src/vacuum_pressure/event_engine.py`
5. `backend/src/vacuum_pressure/runtime_model.py`

Frontend path:
1. `frontend/src/vacuum-pressure.ts`
2. `frontend/vacuum-pressure.html`

Offline experiment path:
1. `backend/src/experiment_harness/cli.py`
2. `backend/src/experiment_harness/runner.py`
3. `backend/src/experiment_harness/signals/`

## Data Locations

Raw replay input (`.dbn`):
- `backend/lake/raw/source=databento/product_type=future_mbo/symbol=<root>/table=market_by_order_dbn/`

Immutable experiment datasets:
- `backend/lake/research/vp_immutable/<dataset_id>/`
- Files per dataset: `bins.parquet`, `grid_clean.parquet`, `manifest.json`, `checksums.json`

Harness results store:
- `backend/lake/research/vp_harness/results/`
- `runs_meta.parquet` and `runs.parquet`

Harness configs:
- `backend/lake/research/vp_harness/configs/`

Gold dataset campaigns:
- `backend/lake/research/vp_harness/configs/gold_campaigns/`
- `backend/scripts/build_gold_dataset_campaign.py`

## Environment

Backend:
- Python 3.12
- `uv`

Frontend:
- Node + npm

Install once:

```bash
cd backend && uv sync
cd frontend && npm ci
```

## Check Inputs Before Starting

Confirm raw futures replay files exist:

```bash
find backend/lake/raw/source=databento/product_type=future_mbo -maxdepth 3 -type d -name 'table=market_by_order_dbn'
```

Confirm at least one immutable dataset exists:

```bash
find backend/lake/research/vp_immutable -maxdepth 2 -type f -name bins.parquet
```

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

Health check:

```bash
curl -s http://localhost:8002/health
```

## Start Frontend

```bash
kill $(lsof -t -iTCP:5174) 2>/dev/null
cd frontend
nohup npm run dev > /tmp/vp_frontend.log 2>&1 &
```

Open:

```text
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25
```

## Serving Contract

WebSocket endpoint:

```text
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25
```

Runtime schema authority:
- Config object and validation: `backend/src/vacuum_pressure/config.py`
- Arrow grid schema and websocket frames: `backend/src/vacuum_pressure/server.py`
- Frontend parser/contract enforcement: `frontend/src/vacuum-pressure.ts`

Message shape:
- text frame: `runtime_config`
- text frame per bin: `grid_update`
- binary frame per bin: Arrow IPC for dense grid rows

## Runtime Model Used For Frontend Predictions

Current frontend projection bands use backend runtime model output by default.

Implementation and wiring:
- Model math: `backend/src/vacuum_pressure/runtime_model.py`
- Per-bin model execution: `backend/src/vacuum_pressure/stream_pipeline.py`
- Stream emission (`runtime_model` + `runtime_model_*`): `backend/src/vacuum_pressure/server.py`
- Frontend consumption and rendering: `frontend/src/vacuum-pressure.ts`

Frontend projection source switch:
- backend (default): `projection_source=backend`
- local shadow path: `projection_source=frontend`

Example:

```text
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25&projection_source=backend
```

## Tune Runtime Model At Launch Time

Use CLI overrides for quick tests:

```bash
cd backend
uv run scripts/run_vacuum_pressure.py --help
```

Most-used runtime model overrides:
- `--perm-zscore-window-bins`
- `--perm-zscore-min-periods`
- `--perm-d1-weight`
- `--perm-d2-weight`
- `--perm-d3-weight`

Permanent defaults should be kept in:
- `backend/src/vacuum_pressure/instrument.yaml`

## Generate / Refresh Gold Datasets

Campaign-driven generation:

```bash
cd backend
uv run scripts/build_gold_dataset_campaign.py --config lake/research/vp_harness/configs/gold_campaigns/<campaign>.yaml
```

Published immutable datasets are written under:
- `backend/lake/research/vp_immutable/`

## Run Experiment Harness

List available assets:

```bash
cd backend
uv run python -m src.experiment_harness.cli list-signals
uv run python -m src.experiment_harness.cli list-datasets
```

Run one config:

```bash
cd backend
uv run python -m src.experiment_harness.cli run lake/research/vp_harness/configs/smoke_perm_derivative.yaml
```

Compare best runs:

```bash
cd backend
uv run python -m src.experiment_harness.cli compare --min-signals 5
```

Harness operating guide:
- `backend/src/experiment_harness/README.md`

## Promote Harness Configuration To Frontend Prediction Algorithm

### Parameter promotion (no algorithm code change)

Use this when the selected harness winner is the same runtime model family already served.

1. Select winning run and params from harness output / MLflow.
2. Update runtime defaults in `backend/src/vacuum_pressure/instrument.yaml`.
3. Restart backend and frontend.
4. Verify frontend is using backend projection source and model is live.

### Algorithm promotion (new model family)

Use this when the selected harness winner is a different signal family.

1. Implement incremental runtime scorer in `backend/src/vacuum_pressure/runtime_model.py`.
2. Wire scorer execution and per-bin attachment in `backend/src/vacuum_pressure/stream_pipeline.py`.
3. Expose runtime config + per-bin outputs in `backend/src/vacuum_pressure/server.py`.
4. Update frontend ingest/render mapping in `frontend/src/vacuum-pressure.ts`.
5. Set defaults in `backend/src/vacuum_pressure/instrument.yaml`.
6. Re-run backend tests + frontend typecheck.

## MLflow

Harness tracking is configured in YAML (`tracking` block).

Key implementation:
- `backend/src/experiment_harness/tracking.py`

## Verification Commands

Backend targeted tests:

```bash
cd backend
uv run pytest tests/test_runtime_perm_model.py tests/test_stream_pipeline_perf.py tests/test_runtime_config_overrides.py
```

Frontend typecheck:

```bash
cd frontend
npx tsc --noEmit
```

## Canonical Files To Read First

- `backend/src/vacuum_pressure/instrument.yaml`
- `backend/src/vacuum_pressure/config.py`
- `backend/src/vacuum_pressure/runtime_model.py`
- `backend/src/vacuum_pressure/stream_pipeline.py`
- `backend/src/vacuum_pressure/server.py`
- `frontend/src/vacuum-pressure.ts`
- `backend/src/experiment_harness/README.md`
- `backend/lake/research/vp_harness/configs/`
