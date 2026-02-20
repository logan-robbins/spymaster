# Spymaster Backend

Futures-only vacuum-pressure backend (live server + experiment harness + offline dataset pipeline).

## Environment

- Python: `>=3.12,<3.13`
- Package/tool runner: `uv` only
- Virtual environment: `backend/.venv`

## Setup

```bash
cd backend
uv sync
```

## Primary Entry Points

```bash
# Live VP server
uv run scripts/run_vacuum_pressure.py --help

# Experiment harness CLI
uv run python -m src.experiment_harness.cli --help

# Databento futures downloader
uv run scripts/batch_download_futures.py --help
```

## Offline Data Pipeline Commands

```bash
# Capture cached stream output from the VP pipeline
uv run scripts/cache_vp_output.py --help

# Build batch gold datasets from campaign configs
uv run scripts/build_gold_dataset_campaign.py --help

# Publish cache output into immutable + experiment dataset layout
uv run scripts/publish_vp_research_dataset.py --help
```

## VP Diagnostics

```bash
# Warm replay cache for faster startup
uv run scripts/warm_cache.py --help

# Core throughput benchmark
uv run scripts/benchmark_vp_core.py --help

# Signal analytics diagnostics
uv run scripts/analyze_vp_signals.py --help
```

## Data Layout

Lake root: `backend/lake/`

- Raw replay `.dbn`: `lake/raw/source=databento/product_type=future_mbo/symbol=<root>/table=market_by_order_dbn/`
- Immutable harness datasets: `lake/research/vp_immutable/<dataset_id>/`
- Harness configs: `lake/research/vp_harness/configs/{pipelines,serving,experiments,gold_campaigns}/`
- Harness results: `lake/research/vp_harness/results/`

## Archived Python Paths

Legacy and non-primary Python paths are archived under `dep/` with original structure preserved.

- Equities downloader: `dep/backend/scripts/batch_download_equities.py`
- Deprecated harness helper: `dep/backend/src/experiment_harness/comparison.py`
- Archived experiment workspace Python: `dep/backend/lake/research/vp_experiments/mnqh6_20260206_0925_1025/`

## Verification

```bash
cd backend

# Focused checks
uv run scripts/run_vacuum_pressure.py --help
uv run python -m src.experiment_harness.cli --help
uv run scripts/batch_download_futures.py --help

# Backend tests
uv run pytest tests/
```
