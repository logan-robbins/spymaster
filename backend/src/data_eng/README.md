# data_eng Module — AI Agent Reference

## Purpose

Market data pipeline for retrieving historically similar setups when price approaches technical levels. Transforms raw market data (DBN format) through Bronze → Silver → Gold layers into feature-rich datasets for similarity search.

## Domain Context

Two pipeline families:
1. **LEVEL pipeline** — Features relative to specific price levels (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW). Answers: "Will price bounce or break through this level?"

Level features MUST be prefixed with level name to prevent duplication when vectors are combined.

## Last Schema P

- ALWAYS refer to the pipeline definition to find the final Stage output Schema per layer. 

## Module Structure

```
src/data_eng/
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
grep -r "class.*Stage.*:" stages/ --include="*.py"
```

**Find pipeline composition:**
```python
from src.data_eng.pipeline import build_pipeline
stages = build_pipeline("future", "silver")  # Returns ordered stage list
```

**Find all datasets:**
```bash
cat config/datasets.yaml
```

**Find all contracts:**
```bash
ls contracts/{bronze,silver,gold}/*/*.avsc
```

**Check existing lake data:**
```bash
ls lake/{bronze,silver,gold}/*/symbol=*/table=*/
```

## Stage Pattern

### Base Class (`stages/base.py`)

All stages extend `Stage` with:
- `name: str` — Stage identifier
- `io: StageIO` — Declares inputs (list of dataset keys) and output (single dataset key)
- `run(cfg, repo_root, symbol, dt)` — Entry point, handles idempotency
- `transform(df, dt)` — Override for simple single-input/single-output transformations

### StageIO Contract

```python
StageIO(
    inputs=["silver.future.table_a", "silver.future.table_b"],  # Input dataset keys
    output="silver.future.table_c"  # Output dataset key
)
```

### Idempotency

Stages check for `_SUCCESS` marker in output partition before running. To reprocess:
```bash
rm -rf lake/{layer}/.../dt=YYYY-MM-DD/
```

### Multi-Output Stages

For stages producing multiple outputs (e.g., one per level type), override `run()` directly instead of `transform()`. See `extract_level_episodes.py` and `compute_approach_features.py` as examples.

## Dataset Configuration

`config/datasets.yaml` defines:
```yaml
dataset.key.name:
  path: layer/product_type=X/symbol={symbol}/table=name  # Lake path pattern
  format: parquet
  partition_keys: [symbol, dt]
  contract: src/data_eng/contracts/layer/product/schema.avsc
```

Dataset keys follow pattern: `{layer}.{product_type}.{table_name}`

## Contract Enforcement

Avro schemas in `contracts/` define:
- Field names and order
- Field types (long, double, string, boolean, nullable unions)
- Documentation per field

`enforce_contract(df, contract)` ensures DataFrame matches schema exactly.

Nullable fields use union type: `["null", "double"]`

## I/O Utilities (`io.py`)

Key functions:
- `partition_ref(cfg, dataset_key, symbol, dt)` — Build PartitionRef for a partition
- `is_partition_complete(ref)` — Check if `_SUCCESS` exists
- `read_partition(ref)` — Read parquet from partition
- `write_partition(cfg, dataset_key, symbol, dt, df, contract_path, inputs, stage)` — Atomic write with manifest

## Pipeline Composition (`pipeline.py`)

`build_pipeline(product_type, layer)` returns ordered list of Stage instances.

Layers: `bronze`, `silver`, `gold`, `all`
Product types: `future`, `future_option`

Stages execute sequentially. Each stage's output becomes available for subsequent stages but the PIPLINE RUNS IN PARALLEL.

## CLI Usage

```bash

uv run python -m src.data_eng.runner \
  --product-type future \
  --layer silver \
  --symbol ESU5 \
  --dates 2025-06-4:2025-09-30 \
  --workers 8

uv run python -m src.data_eng.runner \
  --product-type future \
  --layer silver \
  --symbol ESZ5 \
  --dates 2025-09-01:2025-09-30 \
  --workers 8

# Single date
uv run python -m src.data_eng.runner \
  --product-type future \
  --layer silver \
  --symbol ESU5 \
  --dt 2025-06-05

# Date range (inclusive)
uv run python -m src.data_eng.runner \
  --product-type future \
  --layer gold \
  --symbol ESU5 \
  --dates 2025-06-11:2025-08-30 \
  --workers 8

uv run python -m src.data_eng.runner \
  --product-type future \
  --layer gold \
  --symbol ESZ5 \
  --dates 2025-09-01:2025-09-30 \
  --workers 8

```

Date options:
- `--dates 2025-06-05:2025-06-10` — Range (colon-separated, inclusive)
- `--start-date` + `--end-date` — Explicit range
- `--workers N` — Parallel execution across dates

Bronze uses root symbol (ES), Silver/Gold use specific contract (ESU5).

## Feature Naming Convention

**Features are dynamically defined by Avro contracts in `contracts/` directory.**


Pattern: `bar5s_<family>_<detail>_<suffix>`

Features are defined in `contracts/silver/future/market_by_price_10_bar5s.avsc`
- Load contract at runtime to see available families, details, and suffixes
- Families include state, depth, ladder, shape, flow, trade, wall, etc.
- Suffixes include `_eob` (end-of-bar), `_twa` (time-weighted), `_sum` (aggregated)

### Metadata Fields (not features)

These are context/label fields excluded from feature vectors:
- `symbol`, `bar_ts`, `episode_id`, `touch_id`
- `level_type`, `level_price`, `approach_direction`
- `dist_to_level_pts`, `signed_dist_pts`
- `bar_index_in_episode`, `bar_index_in_touch`, `bars_to_trigger`
- `is_*` flags (trigger states)
- `outcome`, `outcome_score` (labels)

## Adding a New Stage

1. **Create stage file** in appropriate `stages/{layer}/{product_type}/` directory
2. **Define StageIO** with input dataset keys and output dataset key
3. **Add dataset entry** to `config/datasets.yaml` with path and contract reference
4. **Create contract** in `contracts/{layer}/{product_type}/` as Avro schema
5. **Register in pipeline.py** — Import and add to appropriate layer list
6. **Test independently:**
   ```python
   from src.data_eng.stages.{layer}.{product_type}.{module} import {StageClass}
   stage = StageClass()
   stage.run(cfg=cfg, repo_root=repo_root, symbol="ESU5", dt="2025-06-05")
   ```

## Debugging

**Check if input exists:**
```python
from src.data_eng.io import partition_ref, is_partition_complete
ref = partition_ref(cfg, "silver.future.table_name", "ESU5", "2025-06-05")
print(is_partition_complete(ref))  # True if _SUCCESS exists
```

**Inspect parquet output:**
```python
import pandas as pd
df = pd.read_parquet("lake/silver/.../dt=2025-06-05/")
print(df.shape, df.columns.tolist())
```

**Verify contract compliance:**
```python
from src.data_eng.contracts import load_avro_contract, enforce_contract
contract = load_avro_contract(Path("src/data_eng/contracts/.../schema.avsc"))
df = enforce_contract(df, contract)  # Raises if mismatch
```

## Constants

Common constants are defined in stage files or dedicated constants modules:
- `EPSILON = 1e-9` — Prevent division by zero
- `POINT = 1`
- `BAR_DURATION_NS = 5_000_000_000` — 5-second bar duration
- `LOOKBACK_DAYS = 3` — Days of history for volume profiles
- `MIN_DAYS = 3` — Minimum days required for valid profile

## Performance Notes

When adding many columns iteratively, use `df = df.copy()` at the start and consider batch assignment to avoid DataFrame fragmentation warnings. For large-scale column additions, pre-allocate columns or use `pd.concat(axis=1)`.