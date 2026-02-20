# Experiment Harness

Offline evaluation and tuning of VP signals on immutable datasets.

## Entry Point

```bash
cd backend
uv run python -m src.experiment_harness.cli --help
```

CLI commands: `generate`, `run`, `compare`, `promote`, `list-signals`, `list-datasets`, `online-sim`.

## Config Architecture

Three-layer YAML config chain. Each layer references the one below by name.

```
ExperimentSpec  ->  ServingSpec  ->  PipelineSpec
```

- ExperimentSpec: sweep axes, TP/SL eval params, cooldown, tracking. References a ServingSpec by name.
- ServingSpec: scoring params (z-score window, tanh scale, derivative weights), signal name + params, projection config. References a PipelineSpec by name.
- PipelineSpec: capture window (symbol, date, start/end time), engine params (cell_width_ms, flow windows, tau constants). Deterministic `dataset_id()` from content hash.

Config YAML locations:
- `lake/research/vp_harness/configs/pipelines/` — PipelineSpec YAMLs
- `lake/research/vp_harness/configs/serving/` — ServingSpec YAMLs
- `lake/research/vp_harness/configs/serving_versions/` — immutable PublishedServingSpec YAMLs for runtime
- `lake/research/vp_harness/configs/experiments/` — ExperimentSpec YAMLs

Config models:
- `src/vacuum_pressure/pipeline_config.py` — PipelineSpec
- `src/vacuum_pressure/serving_config.py` — ServingSpec, PublishedServingSpec
- `src/vacuum_pressure/serving_registry.py` — alias/version registry
- `src/vacuum_pressure/experiment_config.py` — ExperimentSpec
- `src/vacuum_pressure/scoring.py` — SpectrumScorer (single implementation for server + harness)

## Commands

All run from `backend/`.

### Generate dataset

```bash
uv run python -m src.experiment_harness.cli generate <pipeline_spec.yaml>
```

Streams .dbn events through the VP engine, writes to `lake/research/vp_immutable/<dataset_id>/`. Idempotent: skips if output exists.

Output files per dataset: `bins.parquet`, `grid_clean.parquet`, `manifest.json`, `checksums.json`.

### Run experiment

```bash
uv run python -m src.experiment_harness.cli run <experiment_spec.yaml>
```

Resolves ExperimentSpec -> ServingSpec -> PipelineSpec. Auto-generates dataset if missing. Expands sweep axes (scoring params x signal params x cooldown values). Evaluates each combination with TP/SL. Persists results to `lake/research/vp_harness/results/`.

### Compare results

```bash
uv run python -m src.experiment_harness.cli compare --min-signals 5
uv run python -m src.experiment_harness.cli compare --signal <name> --sort tp_rate
```

Options: `--signal`, `--dataset-id`, `--sort {tp_rate,mean_pnl_ticks,events_per_hour}`, `--min-signals`.

### Promote winner

```bash
uv run python -m src.experiment_harness.cli promote <experiment_spec.yaml> --run-id <winner_run_id> --alias <serving_alias>
```

Extracts winning params from ResultsDB, builds a runtime snapshot, writes immutable PublishedServingSpec YAML to `configs/serving_versions/`, and updates serving alias mapping in `serving_registry.sqlite`.

### List signals and datasets

```bash
uv run python -m src.experiment_harness.cli list-signals
uv run python -m src.experiment_harness.cli list-datasets
```

### Online simulation

```bash
uv run python -m src.experiment_harness.cli online-sim \
  --signal <name> --dataset-id <id> --bin-budget-ms 100
```

Measures per-bin latency of a signal against a budget constraint.

## Signals

Signal implementations live in `src/experiment_harness/signals/`:
- `signals/statistical/` — statistical signals
- `signals/ml/` — ML signals
- `signals/base.py` — `StatisticalSignal` and `MLSignal` base classes

Use `list-signals` to see the current registered set.

## Scoring Invariant

`src/vacuum_pressure/scoring.py` contains `SpectrumScorer` — the single scoring implementation used by both the live server and the harness. Zero train/serve skew.

- Incremental API: `update(d1, d2, d3) -> (score, state_code)` per cell (`state_code` is persisted as `flow_state_code`)
- Batch API: `score_dataset(grid_df, scoring_config) -> DataFrame with flow_score/flow_state_code`

Both APIs produce identical outputs for identical inputs.

## Key Modules

- `src/experiment_harness/cli.py` — Click CLI entry point
- `src/experiment_harness/runner.py` — ExperimentRunner: sweep expansion, signal evaluation, result persistence
- `src/experiment_harness/config_schema.py` — internal `ExperimentConfig` flat schema (not user-facing, produced by `ExperimentSpec.to_harness_config()`)
- `src/experiment_harness/results_db.py` — `ResultsDB`: parquet-backed run storage, `query_runs()`, `query_best()`
- `src/experiment_harness/dataset_registry.py` — `DatasetRegistry`: discovers datasets under `vp_immutable/`
- `src/experiment_harness/eval_engine.py` — TP/SL evaluation engine
- `src/experiment_harness/tracking.py` — MLflow tracking integration
- `src/experiment_harness/online_simulator.py` — per-bin latency simulation

## Results Store

- `lake/research/vp_harness/results/` — `runs_meta.parquet` (run-level metadata) and `runs.parquet` (per-threshold results: tp_rate, n_signals, mean_pnl_ticks, events_per_hour)

## Promotion Flow

1. `compare` to find best run_id
2. `promote` writes immutable PublishedServingSpec to `configs/serving_versions/`
3. `promote` updates alias mapping in `serving_registry.sqlite`
4. Stream using `?serving=<alias_or_id>` (no ad-hoc runtime query overrides)

## Tracking

MLflow tracking configured per ExperimentSpec YAML (`tracking` block).

## Verification

```bash
cd backend

# Harness tests
uv run pytest tests/test_experiment_harness/

# All backend tests
uv run pytest tests/
```
