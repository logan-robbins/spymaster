# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Post-Compression Recovery

**AFTER EVERY CONTEXT COMPRESSION OR AUTO-COMPACT, YOUR FIRST ACTION MUST BE:**

```
Read DEV.md in its entirety before doing anything else.
```

DEV.md contains the current task state, recent progress, and critical context that does not survive compression. Failure to re-read DEV.md will result in lost context and repeated work. This is non-negotiable.

---

## System Overview

Spymaster is a level-interaction similarity retrieval system for ES futures trading. It retrieves historically similar market setups when price approaches technical levels (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW) and presents empirical outcome distributions (BREAK/REJECT/CHOP).

**Core value proposition**: When price approaches a key level, retrieve similar historical setups to show what typically happened next.

## Build and Development Commands

All Python work happens in `backend/`. Use `uv` exclusively for all Python operations:

```bash
cd backend

# Run any script
uv run python script.py

# Run tests
uv run pytest                           # All tests
uv run pytest tests/test_pipeline.py -v # Single file
uv run pytest -k "test_name" -v         # Single test

# Add packages
uv add package_name
```

## Data Pipeline Commands

The pipeline transforms raw Databento DBN files through Bronze → Silver → Gold layers:

```bash
cd backend

# Run silver pipeline (features)
uv run python -m src.data_eng.runner --product-type future --layer silver --symbol ESU5 --dt 2025-06-05

# Run gold pipeline (setup vectors)
uv run python -m src.data_eng.runner --product-type future --layer gold --symbol ESU5 --dt 2025-06-05

# Date range with parallel workers
uv run python -m src.data_eng.runner --product-type future --layer silver --symbol ESU5 --dates 2025-06-05:2025-06-10 --workers 8

# Run retrieval evaluation
uv run python src/data_eng/analysis/v2/test_retrieval_evaluation.py
```

Contract symbols: ESU5 (Jun-Sep), ESZ5 (Sep-Dec), ESH6 (Dec-Mar). Bronze uses root symbol ES.

## Architecture

### Data Lake (backend/lake/)
```
Bronze  → Raw DBN converted to Parquet (MBP-10 book data)
Silver  → Feature engineering (5s bars, levels, volume profiles, episodes)
Gold    → Setup vectors ready for FAISS similarity search
```

### Pipeline System (backend/src/data_eng/)
- `runner.py` - CLI entry point
- `pipeline.py` - Builds ordered stage lists: `build_pipeline("future", "silver")`
- `stages/base.py` - Stage base class with idempotency via `_SUCCESS` marker
- `stages/{bronze,silver,gold}/future/` - Stage implementations
- `config/datasets.yaml` - Dataset paths and contracts
- `contracts/` - Avro schemas defining field contracts

Stages are atomic and idempotent. Each checks for `_SUCCESS` marker before running. To reprocess: `rm -rf lake/{layer}/.../dt=YYYY-MM-DD/`

### Silver Pipeline Stages
1. `SilverConvertUtcToEst` - Timezone conversion
2. `SilverAddSessionLevels` - PM_HIGH, PM_LOW, OR_HIGH, OR_LOW calculation
3. `SilverComputeBar5sFeatures` - 5-second bar feature engineering (order flow, book shape, microstructure)
4. `SilverBuildVolumeProfiles` - 7-day lookback volume profiles
5. `SilverExtractLevelEpisodes` - Extract approach episodes per level type
6. `SilverComputeApproachFeatures` - Level-relative features

### Gold Pipeline Stages
1. `GoldFilterFirst3Hours` - Keep 09:30-12:30 ET only
2. `GoldFilterBandRange` - Filter to 5-point band around levels
3. `GoldExtractSetupVectors` - Final vectors with outcomes

### ML Module (backend/src/ml/)
- `index_builder.py` - FAISS index construction
- `retrieval_engine.py` - Similarity search
- `boosted_tree_train.py` - HistGradientBoosting models
- Feature stages A (core physics) and B (+ technical analysis)

## Feature Naming Conventions

Pattern: `bar5s_<family>_<detail>_<suffix>`

Families: `shape`, `flow`, `depth`, `state`, `wall`, `trade`, `ladder`, `microprice`

Suffixes: `_eob` (end-of-bar), `_twa` (time-weighted average), `_sum`, `_d1_wN` (derivative)

Metadata fields (excluded from vectors): `symbol`, `bar_ts`, `episode_id`, `level_type`, `outcome`

## Key Constants

- `BAR_DURATION_NS = 5_000_000_000` (5-second bars)
- `LOOKBACK_DAYS = 7` (volume profile history)
- `N_BUCKETS = 48` (5-min buckets for 09:30-13:30)
- `EPSILON = 1e-9` (division safety)
- Level types: `PM_HIGH`, `PM_LOW`, `OR_HIGH`, `OR_LOW`
- Outcomes: `BREAK`, `REJECT`, `CHOP`

## Code Principles

- Search codebase for existing patterns before creating new files
- Single canonical implementation, no fallbacks or optional code paths
- Fail fast with clear errors when prerequisites missing
- All changes are breaking - no versioning unless explicitly requested
- Refer to pipeline definitions for final stage output schemas
- Level features must be prefixed with level name to prevent duplication

## Current State

Pipeline processes ES futures data from 2025-06-04 to 2025-09-30 across three contracts (ESU5, ESZ5, ESH6). Retrieval evaluation shows ~37% direction P@10 with 1.05-1.11x lift over baseline.
