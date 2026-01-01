# ENTERPRISE.md (AI-AGENT IMPLEMENTATION SPEC) — Spymaster vNext

This document is **for AI coding agents only**. It is a **concrete, breaking/atomic implementation spec** for refactoring this repo into an enterprise-grade, multi-instrument, multi-pipeline-version **lakehouse + feature-store + MLOps** architecture.

**Primary outcome**: after completion, this repo supports:
- multiple **instruments** (e.g. `es`, `tsla`)
- multiple **pipelines** (feature engineering, episodes, indices, streams, training)
- multiple **pipeline versions** per instrument/pipeline
- consistent **batch + real-time** feature definitions (no divergence)
- ML training, inference, and model registry paths that are deterministic and Azure-migratable

## Non‑negotiables (GLOBAL RULES)

- **NO compatibility**: do not add dual-read paths, fallbacks, shims, or legacy loaders. This is a **breaking/atomic** refactor.
- **One canonical architecture**: after refactor, there is exactly one supported directory layout, one set of path builders, and one pipeline config model.
- **Fail fast**: missing prerequisites must raise clear exceptions; do not silently continue.
- **Instrument-first everywhere**: `instrument` is the first directory under **every** layer (`raw/`, `bronze/`, `silver/`, `gold/`, `_meta/`).
- **Product sub-domains**: under each instrument, use product buckets:
  - `futures/` (for futures data)
  - `options/` (for options data)
  - `equities/` (for cash equities)
  - (optional later) `fx/`, `crypto/` etc — but do not implement unless needed now
- **Do not implement cloud resources**: only design the layout/abstractions so Azure Fabric/ADF migration is straightforward.
- **Pseudo-code only**: do not paste full code into this doc; implement in repo code.

## Canonical Terms (must be used consistently)

- **instrument**: top-level market domain symbol (lowercase), e.g. `es`, `tsla`
- **product**: sub-domain under instrument: `futures`, `options`, `equities`
- **dataset**: logical data product under a product (examples below)
- **schema_version**: dataset contract version (optional now; recommended; if used, encode as `schema=vX`)
- **pipeline_id**: workflow family name (e.g. `feature_engineering`, `episode_construction`, `pentaview`)
- **pipeline_version**: semantic version of transformation logic (immutable)
- **run_id**: unique execution instance id (already exists conceptually via `RunManifestManager`; extend for pipeline runs)

## Canonical Root (MUST CHANGE)

Set the ONLY canonical data root to:
- `CONFIG.DATA_ROOT == <repo>/backend/data`

Required:
- delete all code paths that compute or assume a data root outside of `CONFIG.DATA_ROOT`
- after refactor, **no module** computes its own root by walking directories; it always uses `CONFIG.DATA_ROOT` (or `StageContext.config["DATA_ROOT"]`).

## Canonical Lake Layout (FINAL)

Root is `CONFIG.DATA_ROOT` (single source of truth).

### Raw (source-native; not normalized)

`raw/{instrument}/{product}/{source}/...`

Examples:
- `raw/es/futures/databento/dbn/...`
- `raw/tsla/equities/polygon/...`

### Bronze (append-only, normalized, partitioned; minimal transformations)

`bronze/{instrument}/{product}/{dataset}/schema=v1/{partition_keys...}/date=YYYY-MM-DD/hour=HH/...`

Required: Hive partitions on `date` and (if applicable) `hour`.

Examples:
- `bronze/es/futures/trades/schema=v1/contract=ESZ5/date=2025-12-16/hour=14/part-*.parquet`
- `bronze/es/futures/mbp10/schema=v1/contract=ESZ5/date=2025-12-16/hour=14/part-*.parquet`
- `bronze/es/options/trades/schema=v1/underlying=ES/date=2025-12-16/hour=14/part-*.parquet`
- `bronze/tsla/equities/trades/schema=v1/symbol=TSLA/date=2025-12-16/hour=14/part-*.parquet`
- `bronze/tsla/options/trades/schema=v1/underlying=TSLA/date=2025-12-16/hour=14/part-*.parquet`

Partition keys (inside path) — enterprise constraint:
- futures: `contract=...` (moderate cardinality; REQUIRED)
- options: `underlying=...` (REQUIRED). Optional: `expiry=YYYY-MM-DD` (recommended if you have it)
- equities: `symbol=...` (REQUIRED unless you guarantee one-symbol-per-instrument forever)

Do NOT partition by extremely high-cardinality keys (e.g. `option_symbol=...`), to avoid partition explosion.

**Rule**: the partition key must be inside the dataset path, e.g.:
`bronze/tsla/equities/trades/schema=v1/symbol=TSLA/date=.../hour=...`

### Silver (offline feature store; versioned by pipeline_version)

Silver is organized by pipeline, then dataset:

`silver/{instrument}/{pipeline_id}/version={pipeline_version}/{dataset}/date=YYYY-MM-DD/...`

Examples:
- `silver/es/feature_engineering/version=4.5.0/signals/date=2025-12-16/signals.parquet`
- `silver/es/episode_construction/version=4.5.0/state_table/date=2025-12-16/state.parquet`
- `silver/tsla/feature_engineering/version=1.0.0/signals/date=2025-12-16/signals.parquet`

### Gold (curated production data products; versioned)

Gold is organized by pipeline, then dataset:

`gold/{instrument}/{pipeline_id}/version={pipeline_version}/{dataset}/...`

Examples:
- `gold/es/episode_construction/version=4.5.0/episodes/vectors/date=2025-12-16/episodes.npy`
- `gold/es/episode_construction/version=4.5.0/episodes/metadata/date=2025-12-16/metadata.parquet`
- `gold/es/index_building/version=4.5.0/indices/PM_HIGH/UP/T0_15/index.faiss`
- (optional addon) `gold/es/pentaview/version=3.1.0/stream_bars/date=2025-12-16/stream_bars.parquet`

### Meta (run manifests, lineage, schema registry)

`_meta/{instrument}/runs/{run_id}/...`
`_meta/pipelines/{instrument}/{pipeline_id}/version={pipeline_version}/manifest.json`
`_meta/schemas/{dataset}/schema=vX.json`

No legacy meta locations.

### Models (artifacts + registry pointers; NOT data tables)

Models are stored under the same root, instrument-first:

- `models/{instrument}/{model_id}/version={model_version}/...`
- `_meta/{instrument}/models/{model_id}/...` (registry metadata; pointers to MLflow/W&B runs)

No `data/ml/...` paths remain after refactor.

## Migration Matrix (OLD → NEW) — must be exhaustive in codebase

These are mandatory breaking replacements. After refactor, OLD paths must not appear anywhere.

### Bronze
- OLD: `data/bronze/futures/trades/...` → NEW: `bronze/{instrument}/futures/trades/schema=v1/contract=.../date=.../hour=.../`
- OLD: `data/bronze/options/trades/...` → NEW: `bronze/{instrument}/options/trades/schema=v1/underlying=.../date=.../hour=.../`

### Silver
- OLD: `silver/features/es_pipeline/version=.../date=.../signals.parquet`
  → NEW: `silver/{instrument}/feature_engineering/version=.../signals/date=.../signals.parquet`
- OLD: `silver/state/es_level_state/version=.../date=.../state.parquet`
  → NEW: `silver/{instrument}/episode_construction/version=.../state_table/date=.../state.parquet`

### Gold
- OLD: `gold/episodes/es_level_episodes/version=.../...`
  → NEW: `gold/{instrument}/episode_construction/version=.../episodes/...`
- OLD: `gold/indices/es_level_indices/...`
  → NEW: `gold/{instrument}/index_building/version=.../indices/...`
- (optional addon) OLD: `gold/streams/pentaview/version=.../...`
  → NEW: `gold/{instrument}/pentaview/version=.../stream_bars/...`

### ML / Models
- OLD: `data/ml/...` (joblib, projection models, etc.)
  → NEW: `models/{instrument}/{model_id}/version={model_version}/...` (+ MLflow/W&B tracking)

## Implementation: REQUIRED Code Refactors (Atomic)

### 1) Introduce a single Path Resolver (replace ad-hoc path joins)

Replace `backend/src/common/lake_paths.py` function set with a **single canonical resolver** that is instrument-aware.

Pseudo-code (shape only):

```
class LakePathResolver:
  init(data_root: Path)
  # Bronze (ingestion / append-only)
  bronze_path(instrument, product, dataset, *, schema_version="v1", partitions: dict) -> Path

  # Silver/Gold (pipeline outputs)
  silver_path(instrument, pipeline_id, pipeline_version, dataset, *, partitions: dict) -> Path
  gold_path(instrument, pipeline_id, pipeline_version, dataset, *, partitions: dict) -> Path

  # Meta + models
  run_dir(instrument, run_id) -> Path
  pipeline_manifest_path(instrument, pipeline_id, pipeline_version) -> Path
  models_dir(instrument, model_id, model_version) -> Path
```

Rules:
- `instrument` is required for every call.
- `pipeline_version` is required for Silver/Gold.
- `product` is required for Bronze.
- `schema_version` is fixed at `v1` for Bronze in this refactor (do not support multiple yet).
- Partitions must be hive-style: `key=value` segments.

Mandatory design addition:
- implement an `InstrumentResolver` that maps raw symbols to canonical instrument ids.
  - futures contract `ESZ5` → `instrument="es"`
  - options underlying `ES` → `instrument="es"`
  - equities symbol `TSLA` → `instrument="tsla"`

### 2) Make `instrument` a first-class runtime parameter (StageContext)

Extend `StageContext.config` and pipeline runner CLI so every pipeline run includes:
- `INSTRUMENT`
- `PIPELINE_ID`
- `PIPELINE_VERSION`

Stages must not assume `ES` or `TSLA`.

Pseudo-code:

```
ctx.config["INSTRUMENT"] = "es"
ctx.config["PIPELINE_ID"] = "feature_engineering"
ctx.config["PIPELINE_VERSION"] = "4.5.0"
```

### 3) Rewrite BronzeWriter/BronzeReader to new layout

Files to modify (non-exhaustive):
- `backend/src/io/bronze.py`
- `backend/src/pipeline/utils/duckdb_reader.py`
- Any scripts under `backend/scripts/` that write/read bronze

Rules:
- Bronze paths must be written to `bronze/{instrument}/{product}/{dataset}/...`
- Partition keys must be correct per product (see above).
- Delete any assumptions about bronze root pathing. Use **only** `CONFIG.DATA_ROOT/bronze`.

Concrete required changes:
- `BronzeWriter` must:
  - infer `instrument` from each message (or be configured per instrument and validate message belongs)
  - map schema names (`futures.trades`, `futures.mbp10`, `options.trades`, …) → `(product, dataset)`
  - write under `bronze/{instrument}/{product}/{dataset}/schema=v1/...`
- `BronzeReader` / `DuckDBReader` must:
  - accept `instrument` and `product`
  - query the new glob patterns rooted at `CONFIG.DATA_ROOT/bronze/{instrument}/...`

### 4) Rewrite Silver/Gold stage outputs to new layout (no dataset hardcoding)

Immediate offenders (must be rewritten):
- `backend/src/pipeline/stages/filter_rth.py` (writes `silver/...`)
- `backend/src/pipeline/stages/materialize_state_table.py` (writes `silver/...`)
- `backend/src/pipeline/stages/construct_episodes.py` (writes `gold/...`)
- `backend/src/pipeline/pipelines/silver_to_gold.py` (reads Silver via hardcoded `es_pipeline`)

Rules:
- Stage code must compute output paths via `LakePathResolver`.
- Dataset names must be canonical and stable (see “Dataset Catalog” below).
- No strings like `dataset="es_pipeline"` or `dataset="es_level_state"` remain anywhere.
- All reads use the new layout; if data not found, raise.

Concrete required changes (by file):
- `backend/src/pipeline/stages/filter_rth.py`
  - replace `canonical_signals_dir(... dataset="es_pipeline" ...)` with resolver:
    - `silver/{instrument}/feature_engineering/version={pipeline_version}/signals/date=.../signals.parquet`
  - remove any implicit ES assumptions in filtering logic where possible; keep time window if product spec requires.
- `backend/src/pipeline/pipelines/silver_to_gold.py`
  - remove hardcoded read:
    - `Path(data_root) / "silver" / "features" / "es_pipeline" / ...`
  - instead, read from resolver silver signals location above.
- `backend/src/pipeline/stages/materialize_state_table.py`
  - write to:
    - `silver/{instrument}/episode_construction/version={pipeline_version}/state_table/date=.../state.parquet`
- `backend/src/pipeline/stages/construct_episodes.py`
  - write to:
    - `gold/{instrument}/episode_construction/version={pipeline_version}/episodes/...`

### 5) Replace pipeline builders with config-defined pipelines (no registry-by-import)

Delete the concept of `get_pipeline(name)` returning hardcoded builders.

New model:
- `PipelineManifest` is a JSON/YAML artifact stored under `_meta/pipelines/...`.
- Runtime loads manifest and builds a pipeline stage list.

Manifest schema (minimum viable):
- `instrument`
- `pipeline_id`
- `pipeline_version`
- `inputs`: list of dataset refs (optional)
- `outputs`: list of dataset refs (optional)
- `stages`: ordered list of `{stage_id, params}`

Pseudo-code:

```
manifest = load_manifest(instrument, pipeline_id, pipeline_version)
stages = [stage_registry.instantiate(stage_id, params) for stage in manifest.stages]
pipeline = Pipeline(stages=stages, name=pipeline_id, version=pipeline_version)
```

Concrete requirements:
- store the manifest at:
  - `_meta/pipelines/{instrument}/{pipeline_id}/version={pipeline_version}/manifest.json`
- a pipeline run must copy the manifest into:
  - `_meta/{instrument}/runs/{run_id}/pipelines/{pipeline_id}/manifest.json`

### 6) Stage registry with explicit override precedence (no fallbacks for old behavior)

Implement a stage registry that resolves stage implementations with this precedence:

1. `instrument + pipeline_id + pipeline_version + stage_id`
2. `instrument + pipeline_id + stage_id`
3. `stage_id` (shared/default)

This is not “compatibility”; it is the single supported override mechanism.

### 7) Unify Silver “feature set” versioning (remove split-brain)

Current repo has:
- canonical pipeline silver outputs (written by `FilterRTHStage`)
- experiment feature sets (written by `SilverFeatureBuilder`)

Target:
- **Silver is the offline feature store**.
- A feature set is addressable by `(instrument, dataset, pipeline_version)`.
- `FeatureManifest` becomes the **pipeline manifest** (or is directly referenced by it).

Required actions:
- Delete/replace `silver/features/{manifest.version}/...` layout.
- Replace with: `silver/{instrument}/{pipeline_id}/version={pipeline_version}/{dataset}/...`
- If you keep a manifest file, store it alongside that version:
  - `silver/{instrument}/{pipeline_id}/version={pipeline_version}/manifest.yaml`

### 8) Fix episode vector schema/versioning as a first-class dataset contract

Enforce that `backend/EPISODE_VECTOR_SCHEMA.md` is aligned with actual vector dimensionality.

Required:
- One canonical “episodes” dataset contract per `pipeline_version`.
- Vectors + metadata + sequences are written under:
  - `gold/{instrument}/episode_construction/version={pipeline_version}/episodes/...`
- Index builder reads from that location and writes to:
  - `gold/{instrument}/index_building/version={pipeline_version}/indices/...`

No mixed paths like `gold/episodes/es_level_episodes/...`.

## Pipeline Runtime (Concrete Requirements)

### CLI contract (must implement)

Update `backend/scripts/run_pipeline.py` to require:
- `--instrument` (e.g. `es`)
- `--pipeline-id` (e.g. `feature_engineering`)
- `--pipeline-version` (e.g. `4.5.0`)

Do not accept the old `--pipeline bronze_to_silver` API.

Pseudo-code:

```
run_pipeline(
  instrument="es",
  pipeline_id="feature_engineering",
  pipeline_version="4.5.0",
  date="YYYY-MM-DD",
  write_outputs=true,
)
```

### StageContext contract (must implement)

Every stage must be able to read:
- `ctx.config["DATA_ROOT"]`
- `ctx.config["INSTRUMENT"]`
- `ctx.config["PIPELINE_ID"]`
- `ctx.config["PIPELINE_VERSION"]`

No stage may read global `CONFIG` for pathing.

### Pipeline/run metadata (must implement)

Each run writes:
- `_meta/{instrument}/runs/{run_id}/run.json` containing:
  - instrument, pipeline_id, pipeline_version, date_range, git_sha, config hash, input dataset fingerprints, output dataset locations

## Pipeline Manifests (MUST EXIST) — concrete bootstrap set

The repo must ship with a minimal set of pipeline manifests (checked into repo) so a fresh clone can run end-to-end locally with no “out-of-band” setup.

Store these manifests under:
- `backend/src/pipeline/manifests/{instrument}/{pipeline_id}/version={pipeline_version}/manifest.json`

At runtime, copy them into the lake meta location:
- `_meta/pipelines/{instrument}/{pipeline_id}/version={pipeline_version}/manifest.json`

### Required pipelines for `instrument=es` (initial product scope)

1) `pipeline_id=feature_engineering`
- Inputs: Bronze futures + options
- Outputs: Silver `signals`
- Stage graph: **exactly** the current Bronze→Silver stages, but instrumentized and manifest-driven:
  - `load_bronze`
  - `build_ohlcv` (1min)
  - `build_ohlcv` (10s)
  - `build_ohlcv` (2min + warmup)
  - `init_market_state`
  - `generate_levels`
  - `detect_interaction_zones`
  - `compute_physics`
  - `compute_multiwindow_kinematics`
  - `compute_multiwindow_ofi`
  - `compute_barrier_evolution`
  - `compute_level_distances`
  - `compute_gex_features`
  - `compute_force_mass`
  - `compute_approach_features`
  - `label_outcomes`
  - `filter_rth_and_write_silver_signals` (write to the new silver location)

2) `pipeline_id=episode_construction`
- Inputs: Silver `signals` (from `feature_engineering`)
- Outputs:
  - Silver `state_table`
  - Gold `episodes` (vectors + metadata + sequences)
- Stages:
  - `load_silver_signals`
  - `materialize_state_table` (write silver)
  - `construct_episodes` (write gold)

3) `pipeline_id=index_building`
- Inputs: Gold `episodes`
- Outputs: Gold `indices`
- Stages:
  - `build_indices` (wrap existing `src/ml/index_builder.py` logic behind a stage)

### Optional addon pipelines (NOT core product; do not gate core acceptance)

These are explicitly **derived/addon** pipelines. They must be cleanly pluggable, but are not required for “core product done”.

- `pipeline_id=pentaview` (addon)
  - Inputs: choose ONE canonical upstream and enforce it. Recommended:
    - `silver/{instrument}/episode_construction/version={pipeline_version}/state_table/...`
    - and/or `gold/{instrument}/episode_construction/version={pipeline_version}/episodes/metadata/...`
  - Outputs: `gold/{instrument}/pentaview/version={pipeline_version}/stream_bars/...` and projections
  - Stages: `compute_streams`
  - Hard constraint: must not be a dependency of core pipelines (`feature_engineering`, `episode_construction`, `index_building`)

### Required pipelines for `instrument=tsla` (initial structure only)

Only require manifests to exist (they may be “not implemented” if stages are absent), but the layout + wiring must support them from day one:
- `feature_engineering`
- `episode_construction`
- `index_building`

Optional addon:
- `pentaview`

## Stage ID Catalog (MUST IMPLEMENT EXACTLY)

Stage IDs are stable identifiers used in manifests. Map each `stage_id` → concrete `BaseStage` class.

Agents must implement a `StageFactory` with this registry (ES v1):
- `load_bronze` → `backend/src/pipeline/stages/load_bronze.py:LoadBronzeStage` (instrumentized)
- `build_ohlcv` → `backend/src/pipeline/stages/build_es_ohlcv.py:BuildOHLCVStage`
- `init_market_state` → `backend/src/pipeline/stages/init_market_state.py:InitMarketStateStage`
- `generate_levels` → `backend/src/pipeline/stages/generate_levels.py:GenerateLevelsStage`
- `detect_interaction_zones` → `backend/src/pipeline/stages/detect_interaction_zones.py:DetectInteractionZonesStage`
- `compute_physics` → `backend/src/pipeline/stages/compute_physics.py:ComputePhysicsStage`
- `compute_multiwindow_kinematics` → `backend/src/pipeline/stages/compute_multiwindow_kinematics.py:ComputeMultiWindowKinematicsStage`
- `compute_multiwindow_ofi` → `backend/src/pipeline/stages/compute_multiwindow_ofi.py:ComputeMultiWindowOFIStage`
- `compute_barrier_evolution` → `backend/src/pipeline/stages/compute_barrier_evolution.py:ComputeBarrierEvolutionStage`
- `compute_level_distances` → `backend/src/pipeline/stages/compute_level_distances.py:ComputeLevelDistancesStage`
- `compute_gex_features` → `backend/src/pipeline/stages/compute_gex_features.py:ComputeGEXFeaturesStage`
- `compute_force_mass` → `backend/src/pipeline/stages/compute_force_mass.py:ComputeForceMassStage`
- `compute_approach_features` → `backend/src/pipeline/stages/compute_approach.py:ComputeApproachFeaturesStage`
- `label_outcomes` → `backend/src/pipeline/stages/label_outcomes.py:LabelOutcomesStage`
- `filter_rth_and_write_silver_signals` → `backend/src/pipeline/stages/filter_rth.py:FilterRTHStage` (but rewritten to write new silver layout)
- `load_silver_signals` → NEW stage (must be implemented) that reads the new silver location
- `materialize_state_table` → `backend/src/pipeline/stages/materialize_state_table.py:MaterializeStateTableStage`
- `construct_episodes` → `backend/src/pipeline/stages/construct_episodes.py:ConstructEpisodesStage`
- `build_indices` → NEW stage wrapper around `backend/src/ml/index_builder.py`

Optional addon stage IDs (only required if addon pipelines are implemented):
- `compute_streams` → `backend/src/pipeline/stages/compute_streams.py:ComputeStreamsStage`

No implicit stage naming. No dynamic imports by file name. Stage IDs are the contract.

## Dataset Fingerprints (MUST IMPLEMENT)

Every run must compute and record a deterministic fingerprint for each input dataset it consumes.

Pseudo-definition:
- `dataset_fingerprint = sha256( manifest.json + schema_versions + date_partitions + file_listing )`

Minimum viable fingerprint inputs:
- manifest content hash
- list of date partitions included
- list of files matched by the dataset reader glob (paths + sizes)

Store fingerprints in:
- `_meta/{instrument}/runs/{run_id}/run.json`

## ML / Feature Store / Model Registry (Concrete Requirements)

### Offline Feature Store = Silver

Silver datasets are the offline store. Training must read from Silver/Gold using resolver + explicit `(instrument, pipeline_id, pipeline_version)`.

### Training scripts MUST be re-parameterized

Modify training entrypoints to remove implicit `features.json` coupling and accept:
- `--instrument`
- `--pipeline-version` (for the dataset they train on)
- `--dataset` (explicit: `signals`, `episodes`, etc.)

Required file changes:
- `backend/src/ml/boosted_tree_train.py`
  - remove reliance on repo-root `features.json` `output_path`
  - construct `data_path` via resolver from `instrument + pipeline_id + pipeline_version + dataset`
  - default MLflow experiment name must include instrument + pipeline_version
- `backend/src/ml/patchtst_train.py` and any other trainers:
  - same: no `data/ml` paths; take instrument + pipeline version; resolve data via resolver; write models via resolver
- `backend/src/ml/tracking.py`
  - enforce MLflow experiment naming:
    - `spymaster/{instrument}/{model_id}/{pipeline_version}`
  - always tag runs with:
    - `instrument`, `pipeline_version`, `dataset_fingerprint`, `git_sha`

Model artifact output (joblib, json metadata) must go to:
- `models/{instrument}/{model_id}/version={model_version}/...`

Where `model_version` is a deterministic identifier:
- recommended: `{pipeline_version}_{gitsha[:7]}_{YYYYMMDDHHMMSS}`

### Inference MUST be strict (no “robust to missing columns”)

Remove any “fill missing columns” patterns in inference. Missing columns/features are an error.

Required file changes:
- `backend/src/ml/tree_inference.py`
  - remove logic that injects missing columns with NaNs (raise instead)
  - model loading path must be `models/{instrument}/{model_id}/...`
- `backend/src/ml/feature_sets.py`
  - feature selection must be deterministic and versioned (no hidden defaults)
- `backend/src/ml/retrieval_engine.py`
  - `IndexManager(index_dir=...)` must take resolver-computed path:
    - `gold/{instrument}/index_building/version={pipeline_version}/indices`
- `backend/src/core/inference_engine.py`
  - inference engine must be initialized with:
    - instrument, pipeline versions for models + indices (explicit)
  - remove any TODO/fallback logic that fabricates vectors or time_bucket; enforce real inputs exist

### Model Registry (local, deterministic)

Implement a minimal registry under `_meta/{instrument}/models/{model_id}/`:
- `registry.json` containing:
  - list of model versions, their artifact path under `models/...`, MLflow run id, dataset fingerprints

No “production fallback.” Promotion, if implemented, is a pointer update.

## Real-time / Streaming Alignment (Concrete Requirements)

Core product streaming/inference alignment:
- Core services MUST load models + indices using resolver + explicit `instrument` + version parameters.
- No service may hardcode `ES` or hardcode a `data/...` path.

Optional addon streaming alignment (only if addon is enabled):
- `backend/src/gateway/pentaview_streamer.py`
  - remove hardcoded `gold/streams/pentaview/version=3.1.0`
  - read from: `gold/{instrument}/pentaview/version={pipeline_version}/stream_bars/...`

Update any service that loads models/indices:
- `backend/src/core/main.py` (or wherever the core service wires inference)
  - must accept instrument + pipeline_version for:
    - tree models
    - indices (retrieval)
  - must error if artifacts not present

## Codebase Search/Replace Targets (agents must eliminate)

Agents must ripgrep and remove all occurrences of these substrings (they indicate legacy layout):
- `/data/gold/` (if it bypasses resolver)
- `/data/silver/` (if it bypasses resolver)
- `es_pipeline`
- `es_level_state`
- `es_level_episodes`
- `es_level_indices`
- `gold/streams/pentaview` (must become instrument-first)
- `data/ml/`

## Dataset Catalog (Canonical Names)

These are the stable dataset identifiers used by `LakePathResolver` and manifests.

Bronze datasets:
- futures: `trades`, `mbp10`
- options: `trades` (later: `nbbo`, `greeks`, etc.)
- equities: `trades`, `quotes` (later: `bars`)

Silver datasets:
- `signals` (engineered event table)
- `state_table` (30s cadence state)

Gold datasets:
- `episodes` (vectors + metadata + sequences)
- `indices` (FAISS partitions + compressor)
- (optional addon) `stream_bars` (Pentaview output)
- `training_sets` (optional)

## Performance/Scale Guidance (do implement)

- Avoid small file explosions:
  - Micro-batching is fine in streaming Bronze, but Silver/Gold batch outputs must be consolidated (target large files).
  - If/when using Delta: use compaction/optimization patterns appropriate to the engine (Fabric/Delta has guidance; implement local compaction logic only if required now).
- Partitioning rules:
  - `date=...` is mandatory for all large tables.
  - `hour=...` is mandatory for high-frequency Bronze.
  - Do not over-partition (e.g., do not partition by `minute`).

## Migration to Azure (conceptual alignment only)

- The target folder/table naming must map to:
  - Microsoft Fabric medallion (Bronze/Silver/Gold) and Delta-first for Silver/Gold
  - Azure Data Factory metadata-driven orchestration: iterate `instrument × pipeline_version × date`
- Do not implement Fabric/ADF code; only ensure architecture is compatible.

## Azure Cloud-Native Alignment (MANDATORY: stage/pipeline code organization)

Design constraint: local pipelines must map 1:1 onto Azure orchestration without rewriting stage logic.

### 0) Direct answer: “ADF or Azure ML Pipelines?” (MANDATORY SPLIT)

This repo must reflect Microsoft’s recommended split:
- **Data orchestration (Data → Data)**: Azure Data Factory / Fabric Data Factory
- **Model orchestration (Data → Model)**: Azure Machine Learning Pipelines

Source (Microsoft Learn): [“Which Azure pipeline technology should I use?”](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2#which-azure-pipeline-technology-should-i-use)

Hard boundaries (must enforce in code):
- ADF/Fabric DF is responsible for:
  - ingestion + medallion materialization: Bronze → Silver → Gold
  - building indices (Gold indices) as a **data product**
  - writing `_meta/{instrument}/runs/{run_id}/run.json` + dataset fingerprints
- Azure ML Pipelines is responsible for:
  - training/evaluation/registration of models
  - batch scoring and model promotion (registry pointer updates)
  - writing model artifacts to `models/{instrument}/...` + updating `_meta/{instrument}/models/{model_id}/registry.json`

Integration contract (the ONLY supported hand-off):
- ADF produces: `run_id` + `dataset_fingerprint` + canonical dataset locations (resolver-derived)
- Azure ML Pipelines consumes: `run_id` (and optionally fingerprint) to locate the training dataset deterministically

Do NOT implement training inside ADF stage tasks. Do NOT implement lakehouse materialization inside Azure ML pipeline steps beyond the minimal joins needed to assemble training data from Silver/Gold.

### 1) Code layout (must implement)

Organize pipeline runtime code so it can run:
- locally (developer machine)
- as a cloud task (ADF custom activity / Fabric notebook job) using the same entrypoints

Required module layout:

```
backend/src/pipeline/
  runtime/
    resolver.py              # LakePathResolver + InstrumentResolver
    manifest_loader.py       # loads/validates PipelineManifest
    stage_registry.py        # stage_id -> implementation with override precedence
    dataset_fingerprint.py   # deterministic dataset fingerprints
    run_metadata.py          # writes _meta/{instrument}/runs/{run_id}/run.json
    orchestrator.py          # local orchestrator (reads manifest, executes stages)
    cli/
      run_pipeline.py        # CLI entrypoint: run whole pipeline
      run_stage.py           # CLI entrypoint: run ONE stage (cloud activity unit)
  stages/                    # shared stages (pure compute + thin IO wrapper)
  instruments/               # optional instrument-specific overrides
    es/stages/
    tsla/stages/
  manifests/                 # shipped-with-repo pipeline manifests
    {instrument}/{pipeline_id}/version={pipeline_version}/manifest.json
```

### 2) “Stage task” execution contract (must implement)

Every stage must be invokable as an isolated unit (this is what Azure orchestrators execute).

Pseudo-CLI:

```
python -m src.pipeline.runtime.cli.run_stage \
  --instrument es \
  --pipeline-id feature_engineering \
  --pipeline-version 4.5.0 \
  --date 2025-12-16 \
  --stage-id compute_physics \
  --run-id <generated_if_missing>
```

Hard requirements:
- `run_stage` reads its required inputs from lake paths derived via `LakePathResolver`
- `run_stage` writes its outputs to lake paths derived via `LakePathResolver`
- `run_stage` writes stage metadata under `_meta/{instrument}/runs/{run_id}/stages/{stage_id}/...`
- no stage depends on in-memory outputs from a prior stage when run via `run_stage`

### 3) Manifest schema must be orchestrator-ready (must implement)

PipelineManifest must be directly translatable into a metadata-driven Azure Data Factory pipeline:
- pipeline parameters: `instrument`, `pipeline_id`, `pipeline_version`, `date` (or date range), `run_id`
- each `stages[]` item must support:
  - `stage_id`
  - `params`
  - `depends_on` (list of stage_ids) OR strictly ordered list

### 4) Mapping rules (what Azure will do later)

- **ADF / Fabric Data Factory**:
  - reads a control table of `(instrument, pipeline_id, pipeline_version, date_range, enabled)`
  - for each row, triggers `run_pipeline` or runs `run_stage` activities with dependencies
- **Fabric / OneLake**:
  - points `DATA_ROOT` at OneLake/ADLS-compatible storage and executes the same stage tasks


**Concrete mismatches you should resolve (docs + implementation)**
These aren’t “philosophical”—they’re the specific deltas between the current repo’s documented usage and the enterprise target.
README path examples conflict with ENTERPRISE.md
README.md still uses paths like data/silver/state/es_level_state/... and gold/episodes/es_level_episodes/..., and data/ml/... for projection models.
Decision: after the refactor, update README to only reference the resolver-backed instrument-first layout (and remove data/ml references).
Versioning language needs one consistent rule
README emphasizes “Canonical Version 3.1.0”; ENTERPRISE.md uses pipeline_version examples like 4.5.0.
Decision: treat pipeline_version as the only version that controls Silver/Gold outputs. If you keep a “product version,” it must be explicitly mapped to pipeline versions (otherwise it will cause drift/confusion).
Silver/Gold table format
ENTERPRISE.md is Parquet-centric but mentions Delta as optional.
Decision (Azure-friendly): keep the layout and dataset boundaries stable now; you can adopt Delta for Silver/Gold later without changing the resolver contract. If you do adopt Delta, do it only for Silver/Gold where ACID/merges matter (Bronze can remain Parquet).

## Execution Checklist (AI Agent)

1. **Decide canonical `DATA_ROOT`** (single root) and update all modules to use it consistently.
2. Implement `LakePathResolver` and delete legacy path helpers.
3. Update BronzeWriter/BronzeReader + DuckDB readers to new `bronze/{instrument}/{product}/...` layout.
4. Update every pipeline stage read/write to use resolver + new dataset names.
5. Rewrite pipeline runner to require `--instrument --pipeline-id --pipeline-version`.
6. Replace pipeline registry/builders with manifest-driven construction + stage registry.
7. Remove old Silver experiment layout; unify Silver as offline feature store.
8. Update index building + retrieval components to read/write the new Gold locations.
9. Update ML training + inference modules to be instrument/version aware and to write models to `models/{instrument}/...`.
10. Update streaming services (gateway/core) to load from new gold/silver paths.
11. Update tests to the new paths and delete obsolete tests.
12. Run `uv run pytest` from `backend/` until green.


## Acceptance Criteria (must all be true)

- No code path can read or write the old lake layout.
- All outputs are under `{layer}/{instrument}/...` exactly.
- `instrument` is required to run any pipeline; attempting to run without it errors.
- `pipeline_version` is required and controls Silver/Gold versioned outputs.
- ES pipelines run end-to-end and write:
  - Bronze: `bronze/es/...`
  - Silver: `silver/es/feature_engineering/version=.../signals/...` and `silver/es/episode_construction/version=.../state_table/...`
  - Gold: `gold/es/episode_construction/version=.../episodes/...` and `gold/es/index_building/version=.../indices/...`
- Training runs produce:
  - MLflow runs tagged with instrument + pipeline_version + dataset fingerprint
  - model artifacts at `models/es/...`
- Inference services load:
  - models from `models/{instrument}/...`
  - indices from `gold/{instrument}/index_building/...`
No code references old roots/strings (data/ml/, es_pipeline, es_level_*).
All reads/writes go through LakePathResolver.
Every pipeline run is parameterized by instrument, pipeline_id, pipeline_version, and date/date-range.
Every stage is runnable in isolation and is deterministic/idempotent for a given (instrument, pipeline_version, date).
Every run writes _meta with dataset fingerprints.
Training/inference read datasets via resolver and write models via resolver, with strict feature/schema enforcement.

**IMPORTANT**
Pentaview is explicitly NOT required for core acceptance.


