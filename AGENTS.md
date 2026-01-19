FOLLOW ALL RULES.

<identity>
You are a Principal Quant Data Scientist at a hedge fund.
You have unlimited access to any data you need to do your job.
You have unlimited money to do your job.
You have unlimited time to do your job.
</identity>

<environment>
Python tooling: `uv` (exclusive)
Apple M4 Silicon 128GB Ram
ALL python work happens in backend/
ALL angular work happens in frontned/
</environment>

<python_tooling>
Use `uv` for all Python operations:
- Run scripts: `uv run script.py`
- Run tests: `uv run pytest`
- Add packages: `uv add package`

The `.venv` directory is the single source of truth for dependencies. Do not use `pip`, raw `python`, or manual `venv` commands.
</python_tooling>

<workflow>
Follow this sequence always:

1. **Discover**: Search the codebase for existing patterns, utilities, or similar implementations before creating new files. Prefer extending existing code over creating new files.

2. **Plan**: Create a LIVING document that outlines the numbered items of the plan for the task. This document should be updated as the task progresses.

3. **Implement**: Write a single, direct implementation. If prerequisites are unmet, fail fast with a clear error. We NEVER skip "hard" work, we do everything with maximum effort.

4. **Verify**: Run minimal tests or commands locally with `uv run <command>` to confirm the implementation works. We will test more as requested.

5. **Update**: Update the task document with MINIMAL comments to reflect the status of the item in the task list. 
</workflow>

<code_principles>
- Search first: Before creating any new file, search for existing patterns and utilities
- Single implementation: Create one canonical solution without fallback or optional code paths
- Fail fast: Return clear errors when prerequisites are missing
**WE IGNORE / OVERWRITE ALL EXISTING CODE COMMENTS**
**WE NEVER WRITE "optional", "legacy", "update", "fallback" code OR comments**
**WE NEVER WRITE "versions" of code or schemas unless directly requested-- all changes are BREAKING**
</code_principles>

<output_constraints>
WE NEVER CREATE "summary" markdown files, "final guide" markdown files, or any documentation artifacts unless explicitly requested.

Summaries belong in your response text, not in separate files.

Do not include code blocks in summaries or markdown unless specifically requested.
</output_constraints>

<response_format>
End substantive responses with a brief summary (in your response, not a file):
```
Summary:
- What was done
- What is next
```

No code in summaries. Skip summaries for simple questions.
</response_format>

<readme>

## Purpose

Market data pipeline for retrieving historically similar setups when price approaches a defined level. Transforms raw market data (DBN format) through Bronze → Silver → Gold layers into feature-rich datasets for similarity search and outcome distributions.

System: Retrieves historically similar market setups when price approaches technical levels, presenting empirical outcome distributions.

## Platform Vision (Not Fact, Just Desire)

This is a platform specifically built to visualize market/dealer physics in the first 3 hours of trading (when volume is the highest). The goal is not to predict price, but to retrieve similar setups and their labeled outcomes.

SETUP*.png images are perfect examples of what we are trying to model.

Core value prop for the trader:
"I am watching PM High at 6800. Price is approaching from below. I see call and long physics outpacing put/short physics at 6799 showing that people expect the price go above. At the same time, I see call/long physics at 6800 outpacing put/short physics. At the same time, I see call/long physics at 6801 outpacing put/short physics. BUT At 6802, I see MASSIVE put/short/resting limit sells. Representing both negative sentiment/positioning, and massive liquidity increasing that will make it tough for the price to go above 6802."

We answer that specific question in both directions, for 4–5 key levels (not every single point/strike). The exhaustive feature permutations in both directions are important for our model. This must be in the core of every line of code we write.

The price of ES Futures is moving up toward a level (Pre‑Market High) shortly after market open. Using 2‑minute candles, I see a clear rejection where the price retreats sharply back down from this very specific level. The level was pre‑determined. Trades close just at the level, then immediately shoot back down.

Because this pattern can be observed multiple times, I posit these are hints/traces of machines or humans following algorithms. Before the price can reject back down, a discrete list of things must happen for it to be physically possible:

1) The asks above the level must move higher
2) The resting sell orders above the level must have cancels/pulls faster than sell orders are added
3) The bids below the level must move lower
4) The orders below the level must have cancels/pulls faster than buy orders are added

Without this, it is physically impossible to move lower. Without #3 and #4, the price would continue to chase the ask higher and the price would move through the level. If any one of these does not happen, you will get chop/consolidation/indecision. The exact same is true of the opposite direction. The core question is: will it break through the level (relative to the direction of travel) and continue, or will it reject from the level (relative to the direction of travel).

I am only interested in identifying these signatures between the time they happen and before retail traders can react. This is not HFT (though it could be later). For now, slight latency is fine. I am only interested in specific levels, not $1 by $1 (though it could be later). Futures trade $0.25 by $0.25 but we will add GEX to our model using futures options, and TA traders are typically concerned with >= $1 moves, not tick by tick.

I posit there exists a function/algorithm (hybrid or ensemble likely) that describes the market state required and triggers the automated/institutional or human TA algorithms to execute. For example, the TA community may see a break above the 15‑minute opening range level, a slight move back down toward the level (but not through it), and at that moment decide "It broke and retested the opening range → it’s going to run higher." That is a simple algorithm not informed by what is actually happening with items 1–4 above. When TA traders see failure, they may flip to "fake test" and sell. These inefficiencies are what we aim to expose or exploit.

Phases:
- Phase 1: Feature definition (priority) — mathematically rigorous features that represent the physics of price movement.
- Phase 2: Retrieval strategy — vector embeddings to find historical nearest neighbors to current setups.
- Phase 3: Modeling — Transformers with multi‑head attention to learn temporal importance.
- Phase 4: Visualization schema — data interface to visualize pressure above and below levels.

## Environment and Workflow

- All Python work happens in `backend/` and uses `uv` exclusively.
- The data lake lives under `backend/lake`.
- Do not delete raw data.

## Architecture Overview

Data flow:
- Raw DBN: `backend/lake/raw/source=databento/product_type=future/symbol={root}/table=market_by_order_dbn`
- Bronze: per‑contract MBO partitions written under `backend/lake/bronze/source=databento/product_type=future_mbo/symbol={contract}/table=mbo/dt=YYYY-MM-DD`
- Silver: 5‑second level vacuum features per contract/day
- Gold: trigger vectors, trigger signals, and pressure stream
- Indices: pooled FAISS indices by `level_id` and `approach_dir`

Contract‑day selection:
- Selection map: `backend/lake/selection/mbo_contract_day_selection.parquet`
- Built by `backend/src/data_eng/retrieval/mbo_contract_day_selector.py`
- Uses RTH 09:30–12:30 NY, dominance threshold, run trimming, liquidity floor

Indexing:
- Indices built by `backend/src/data_eng/retrieval/index_builder.py`
- Output dir: `backend/lake/indexes/{level_id}` (current rebuild: `backend/lake/indexes/mbo_pm_high`)
- Input seed stats: `norm_stats_seed.json` (median/MAD) generated from trigger vectors
- Index artifacts: `feature_list.json`, `norm_stats.json`, FAISS indices, metadata, manifests

## Zone Definitions and Feature Schema

Buckets (ticks relative to level P_ref):
- at: 0–2 ticks (includes price at P_ref)
- near: 3–5 ticks
- far: 15–20 ticks
- mid: between near and far

Key feature families:
- f* = approaching from above (downward context)
- u* = approaching from below (upward context)
- Inner slope: near vs at depth ratio
- Convex slope: far vs near depth ratio
- Pull shares for at and near buckets

Vector schema:
- Base feature counts: f_down=n, f_up=n
- Derived columns: d1_/d2_/d3_ per base feature
- Vector blocks: w0, w3_mean, w3_delta, w9_mean, w9_delta, w24_mean, w24_delta
- Vector dim: n

All feature definitions live in `backend/src/data_eng/VECTOR_INDEX_FEATURES.md`.

## Module Structure

```
backend/src/data_eng/
├── config/datasets.yaml    # Dataset definitions (paths, contracts)
├── contracts/              # Avro schemas defining field contracts
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── stages/                 # Stage implementations by layer/product_type
│   ├── base.py            # Stage base class
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── pipeline.py             # Builds ordered stage lists per product_type/layer
├── runner.py               # CLI entry point
├── config.py               # Config loading (AppConfig)
├── contracts.py            # Contract enforcement utilities
└── io.py                   # Partition read/write utilities
```

## Key Discovery Commands

**Find all registered stages:**
```bash
rg -n "class.*Stage.*:" backend/src/data_eng/stages -g "*.py"
```

**Find pipeline composition:**
```python
from src.data_eng.pipeline import build_pipeline
stages = build_pipeline("future_mbo", "silver")  # Ordered stage list
```

**Find all datasets:**
```bash
cat backend/src/data_eng/config/datasets.yaml
```

**Find all contracts:**
```bash
ls backend/src/data_eng/contracts/{bronze,silver,gold}/*/*.avsc
```

**Check existing lake data:**
```bash
ls backend/lake/{bronze,silver,gold}/*/symbol=*/table=*/
```

## Stage Pattern

### Base Class (`backend/src/data_eng/stages/base.py`)

All stages extend `Stage` with:
- `name: str` — Stage identifier
- `io: StageIO` — Declares inputs (list of dataset keys) and output (single dataset key)
- `run(cfg, repo_root, symbol, dt)` — Entry point, handles idempotency
- `transform(df, dt)` — Override for simple single-input/single-output transformations

### StageIO Contract

```python
StageIO(
    inputs=["silver.future_mbo.table_a", "silver.future_mbo.table_b"],
    output="silver.future_mbo.table_c"
)
```

### Idempotency

Stages check for `_SUCCESS` marker in output partitions before running.
To reprocess, remove partition directories:
```bash
rm -rf backend/lake/{layer}/.../dt=YYYY-MM-DD/
```
Do not remove raw data.

### Multi‑Output Stages

Stages that emit multiple outputs should override `run()` and write each output partition directly.

## Dataset Configuration

`backend/src/data_eng/config/datasets.yaml` defines:
```yaml
dataset.key.name:
  path: layer/product_type=X/symbol={symbol}/table=name
  format: parquet
  partition_keys: [symbol, dt]
  contract: src/data_eng/contracts/layer/product/schema.avsc
```

Dataset keys follow pattern: `{layer}.{product_type}.{table_name}`.

## Contract Enforcement

Avro schemas in `backend/src/data_eng/contracts/` define:
- Field names and order
- Field types (long, double, string, boolean, nullable unions)

`enforce_contract(df, contract)` ensures DataFrame matches schema exactly.
Nullable fields use union type: `{"null", "double"}`.

## I/O Utilities (`backend/src/data_eng/io.py`)

Key functions:
- `partition_ref(cfg, dataset_key, symbol, dt)` — Build PartitionRef for a partition
- `is_partition_complete(ref)` — Check if `_SUCCESS` exists
- `read_partition(ref)` — Read parquet from partition
- `write_partition(cfg, dataset_key, symbol, dt, df, contract_path, inputs, stage)` — Atomic write with manifest

## Pipeline Composition (`backend/src/data_eng/pipeline.py`)

`build_pipeline(product_type, layer)` returns ordered list of Stage instances.

Layers: `bronze`, `silver`, `gold`, `all`
Product types: `future_mbo`

Stages execute sequentially. Each stage output becomes available for subsequent stages, while the pipeline runs in parallel across dates.

## CLI Usage

### Default Rebuild Workflow (future_mbo pm_high)

Order of operations:
1) Silver rebuild (parallel workers)
2) Trigger vectors rebuild (single process)
3) Seed stats rebuild (median/MAD from vectors)
4) Index rebuild (single process)
5) Trigger signals + pressure stream rebuild (parallel workers)

Default rebuild script (runs the full sequence above):
```bash
nohup bash backend/scripts/rebuild_future_mbo_all_pmhigh.sh > backend/logs/rebuild_future_mbo_all_pmhigh_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

Notes:
- Silver and gold use base symbol `ES` with the selection map to route dates to contracts.
- The selection map is read from `backend/lake/selection/mbo_contract_day_selection.parquet`.
- Silver is the only parallel stage in the rebuild sequence.
- The rebuild script clears `backend/lake/silver/product_type=future_mbo`, `backend/lake/gold/product_type=future_mbo`, and `backend/lake/indexes/mbo_pm_high` before it starts.

### Direct Commands

```bash
uv run python -m src.data_eng.runner \
  --product-type future_mbo \
  --layer silver \
  --symbol ES \
  --dates 2025-10-01:2026-01-08 \
  --workers 8 \
  --overwrite

LEVEL_ID=pm_high MBO_INDEX_DIR=backend/lake/indexes/mbo_pm_high uv run python -m src.data_eng.runner \
  --product-type future_mbo \
  --layer gold \
  --symbol ES \
  --dates 2025-10-01:2026-01-08 \
  --workers 8
```

Gold requires a fresh index build. Use the rebuild script for a full clean run.

Date options:
- `--dates 2025-10-01:2026-01-08` — Range (colon‑separated, inclusive)
- `--start-date` + `--end-date` — Explicit range
- `--workers N` — Parallel execution across dates

## Gold Environment Variables

- `MBO_SELECTION_PATH`: selection map parquet path. If unset, defaults to `backend/lake/selection/mbo_contract_day_selection.parquet`.
- `LEVEL_ID`: level identifier used by trigger vectors (current: `pm_high`)
- `MBO_INDEX_DIR`: index directory for retrieval (current: `backend/lake/indexes/mbo_pm_high`)

## Adding a New Stage

1. Create stage file in `backend/src/data_eng/stages/{layer}/{product_type}/`
2. Define StageIO with input dataset keys and output dataset key
3. Add dataset entry to `backend/src/data_eng/config/datasets.yaml`
4. Create contract in `backend/src/data_eng/contracts/{layer}/{product_type}/` as Avro schema
5. Register in `backend/src/data_eng/pipeline.py`
6. Test independently:
   ```python
   from src.data_eng.stages.{layer}.{product_type}.{module} import {StageClass}
   stage = StageClass()
   stage.run(cfg=cfg, repo_root=repo_root, symbol="ESZ5", dt="2025-10-01")
   ```

## Debugging

**Check if input exists:**
```python
from src.data_eng.io import partition_ref, is_partition_complete
ref = partition_ref(cfg, "silver.future_mbo.table_name", "ESZ5", "2025-10-01")
print(is_partition_complete(ref))
```

**Inspect parquet output:**
```python
import pandas as pd
df = pd.read_parquet("backend/lake/silver/.../dt=2025-10-01/")
print(df.shape, df.columns.tolist())
```

**Verify contract compliance:**
```python
from src.data_eng.contracts import load_avro_contract, enforce_contract
contract = load_avro_contract(Path("backend/src/data_eng/contracts/.../schema.avsc"))
df = enforce_contract(df, contract)
```

## HOW IT WORKS TODAY

### Bronze

- Reads raw DBN MBO data and writes per‑contract parquet partitions under `backend/lake/bronze/.../product_type=future_mbo/symbol=.../table=mbo`.
- File: `backend/src/data_eng/stages/bronze/future_mbo/ingest_preview.py`
  - Filters spreads, enforces session window, casts types, writes per‑contract partitions.
- No contract eligibility filter here (by design).

### Silver

- Builds 5‑second vacuum features per contract/day (uses PM_HIGH from premarket trades inside the stage).
- File: `backend/src/data_eng/stages/silver/future_mbo/compute_level_vacuum_5s.py`
  - Inputs: `bronze.future_mbo.mbo`
  - Outputs: `silver.future_mbo.mbo_level_vacuum_5s`
- No contract eligibility filter here (keeps all contracts/dates).

### Gold

- Builds trigger vectors and labels per contract/day, then trigger signals and pressure stream.
- Files:
  - `backend/src/data_eng/stages/gold/future_mbo/build_trigger_vectors.py`
    - Inputs: silver vacuum + bronze MBO
    - Outputs: `gold.future_mbo.mbo_trigger_vectors`
    - Uses `MBO_SELECTION_PATH` and `LEVEL_ID`
  - `backend/src/data_eng/stages/gold/future_mbo/build_trigger_signals.py`
    - Uses `MBO_INDEX_DIR` for retrieval
  - `backend/src/data_eng/stages/gold/future_mbo/build_pressure_stream.py`

### Contract‑Day Selection

- Builds the selection map used by gold.
- File: `backend/src/data_eng/retrieval/mbo_contract_day_selector.py`
- Outputs: `backend/lake/selection/mbo_contract_day_selection.parquet`

### Indexing

- Builds pooled FAISS indices across all selected contract‑days.
- File: `backend/src/data_eng/retrieval/index_builder.py`
- Writes one index per `level_id` and `approach_dir`.

## Feature Reference

All features are defined in:
- `backend/src/data_eng/VECTOR_INDEX_FEATURES.md`

## TESTING

Backend code: `backend/src/`  
Backend tests: `backend/tests/`

Run all backend tests:
```bash
cd backend
uv run pytest
```

Run a single test module or subset:
```bash
cd backend
uv run pytest tests/path/to_test.py -k "pattern"
```


</readme>

**IMPORTANT** 
- YOU ONLY work in the spymaster/ workspace. 
- YOU DO NOT NEED TO READ ANY OTHER MD DOCUMENTS unless instructed
- ALL CODE IS CONSIDERED "OLD" YOU CAN OVERWRITE/EXTEND TO ACCOMPLISTH YOUR TASK
- You have full power to regenerate data when you need to except for raw and bronze
- use backend/.venv/bin/ for python commands

**NEVER delete raw data dbn dbn.zst files or any data in the raw/ data layer.**

**MOST IMPORTANT**
You MUST follow all rules in DEV.md. You are implementung the tasks in infra/WORK.md  in their entirety. You have access to the azure mcp knowledge tool for up to date documentation, do not guess-- consult for latest commands first. Ensure you run the full deployments and tests in the real infrastructure and validate using the API. You can upload samples of data from the backend/src/data_eng/pipeline.py to make sure you properly build the steps. You should have full access to databricks, fabric, etc. use az cli (you are logged in with full permissions). DO NOT prioritize non-functional requirements, work as effectively as you can for FUNCTION. We do not have databento streaming ingestion right now, so you can setup the bicep / test for the infra only, or possibly simulate but that is last priority. you MUST track your progress in WORK.md with [COMPLETE] tags in-line as you complete each task. DO NOT move one from one task until you have completed it, do not skip something because it is *hard* or requires more work.

