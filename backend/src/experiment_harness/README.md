# Experiment Harness

Canonical runbook for offline evaluation and tuning of VP signals.

This is the source of truth for:
- running experiments from YAML
- comparing outcomes
- tracking with MLflow
- promoting a winning configuration into the live frontend prediction path

## Purpose

The harness evaluates signals on immutable datasets without changing live serving.

Entrypoint:
- `backend/src/experiment_harness/cli.py`

Core runner:
- `backend/src/experiment_harness/runner.py`

Dataset resolution:
- `backend/src/experiment_harness/dataset_registry.py`

## Required Inputs

Datasets must exist in one of:
- `backend/lake/research/vp_immutable/<dataset_id>/`
- `backend/lake/research/vp_harness/generated_grids/<dataset_id>/`

Each dataset directory must contain:
- `bins.parquet`
- `grid_clean.parquet`

Configs live in:
- `backend/lake/research/vp_harness/configs/`

## Core Commands

Run from `backend/`.

List signals and datasets:

```bash
uv run python -m src.experiment_harness.cli list-signals
uv run python -m src.experiment_harness.cli list-datasets
```

Run one experiment config:

```bash
uv run python -m src.experiment_harness.cli run lake/research/vp_harness/configs/smoke_perm_derivative.yaml
```

Compare best runs:

```bash
uv run python -m src.experiment_harness.cli compare --min-signals 5
```

Optional online simulation:

```bash
uv run python -m src.experiment_harness.cli online-sim --signal perm_derivative --dataset-id <dataset_id> --bin-budget-ms 100
```

## Current Production-Relevant Configs

- `lake/research/vp_harness/configs/smoke_perm_derivative.yaml`
- `lake/research/vp_harness/configs/tune_perm_derivative.yaml`
- `lake/research/vp_harness/configs/tune_ads_pfp_svac_runtime.yaml`
- `lake/research/vp_harness/configs/tune_ads_pfp_svac_rr_20_8.yaml`

## Results and Tracking

Local results database:
- `backend/lake/research/vp_harness/results/runs_meta.parquet`
- `backend/lake/research/vp_harness/results/runs.parquet`

Tracking implementation:
- `backend/src/experiment_harness/tracking.py`

MLflow configuration is controlled per YAML config (`tracking` block).

## Add or Tune a Signal

1. Choose or create a YAML in `lake/research/vp_harness/configs/`.
2. Set `datasets`, `signals`, and `eval`.
3. Add sweep axes under `sweep.universal` and `sweep.per_signal.<signal_name>`.
4. Set `tracking.experiment_name` and tags.
5. Run via CLI.
6. Rank with `compare`.

Fail-fast contracts:
- unknown sweep params are rejected
- missing datasets/configs are rejected

## Promote Harness Winner To Live Frontend Prediction Algorithm

### A) Promote parameter set for current runtime model (no algorithm code change)

Use when the winner is already represented by the live runtime model family.

1. Take winning parameter values from harness run output.
2. Update runtime defaults in `backend/src/vacuum_pressure/instrument.yaml`.
3. Restart backend (`scripts/run_vacuum_pressure.py`) and frontend.
4. Verify live stream/visual output.

Files that enforce this runtime path:
- `backend/src/vacuum_pressure/runtime_model.py`
- `backend/src/vacuum_pressure/stream_pipeline.py`
- `backend/src/vacuum_pressure/server.py`
- `frontend/src/vacuum-pressure.ts`

### B) Promote a different signal family as frontend prediction algorithm

Use when the winner is not yet implemented in live serving.

1. Implement incremental runtime scorer in `backend/src/vacuum_pressure/runtime_model.py`.
2. Wire per-bin execution in `backend/src/vacuum_pressure/stream_pipeline.py`.
3. Expose runtime config/update payloads in `backend/src/vacuum_pressure/server.py`.
4. Update frontend ingestion/render mapping in `frontend/src/vacuum-pressure.ts`.
5. Set defaults in `backend/src/vacuum_pressure/instrument.yaml`.
6. Re-run backend tests and frontend typecheck before launch.

## Verification

Backend targeted checks:

```bash
cd backend
uv run pytest tests/test_experiment_harness/test_perm_derivative_signal.py tests/test_experiment_harness/test_ads_pfp_svac_signal.py tests/test_experiment_harness/test_runner_core.py
```

Live-runtime integration checks:

```bash
cd backend
uv run pytest tests/test_runtime_perm_model.py tests/test_stream_pipeline_perf.py tests/test_runtime_config_overrides.py
```

Frontend typecheck:

```bash
cd frontend
npx tsc --noEmit
```
