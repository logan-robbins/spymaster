# Spymaster - Streaming-First Radar Architecture (Technical Reference)

**System Role**: Expert-level guide for AI Coding Agents.
**Context**: Replay-based streaming of Databento MBO data for futures and options.
**Date**: `2026-01-06` (Single-day full scope).

## Purpose

Transforms raw market data (DBN format) through Bronze → Silver → Gold layers into feature-rich datasets and vectors. We aim to visualize market/dealer physics in near-real-time. ETUP*.png images are perfect examples of what we are trying to model.

Core value prop for the trader:
"I am watching PM High at 6800. Price is approaching from below. I see call/long physics outpacing put/short physics at 6799 showing that people expect the price go above. At the same time, I see call/long physics at 6800 outpacing put/short physics. At the same time, I see call/long physics at 6801 outpacing put/short physics. BUT At 6802, I see MASSIVE put/short/resting limit sells. Representing both negative sentiment/positioning, and massive liquidity increasing that will make it tough for the price to go above 6802."

The price of ES Futures is moving up toward a level (Pre‑Market High) shortly after market open. Using 2‑minute candles, I see a clear rejection where the price retreats sharply back down from this very specific level. The level was pre‑determined. Trades close just at the level, then immediately shoot back down. Because this pattern can be observed multiple times, I posit there is a way to predict it.  

Several conditions must happen (within milliseconds/seconds maxiumum):

1) The asks above the level must move higher
2) The resting sell orders above the level must have cancels/pulls faster than sell orders are added
3) The bids below the level must move lower
4) The orders below the level must have cancels/pulls faster than buy orders are added

Without this, it is physically impossible to move lower. Without #3 and #4, the price would continue to chase the ask higher and the price would move through the level. If any one of these does not happen, you will get chop/consolidation/indecision. The exact same is true of the opposite direction. The core question is: catalog and visualize the mechanics/state and change charactistics (deriviatives) of a market in the seconds before a key movement, and is there a model that can be trained to predict it- with the special note that we are trying harder to find proof of algorithmic trading than to predict the price. "The physics/presure/fluid mechanics in the last n seconds have typically preceeded a movement" (this is where the vector retrieval comes in).

**CRITICAL**
- We attempt to make ALL features for prediction, retrieval, and even visulization, relative to the SPOT price at a given time. This way, models can be applied to any regime or market. 
- This means we try to make every feature's final form a scaled derivative or ratio ONLY (when it makes sense)
- Mental Model: A wave/water drop model may be _somewhat_ applicable two ponds with similar composition. The same bernoulli formulas apply to multiple plane wings considering x,y,z imputs.

## Environment and Workflow

- All Python work happens in `backend/` `backend/.venv/` and uses `uv` exclusively.
- The data lake lives under `backend/lake`.

---

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

## Stage Pattern

### Base Class (`backend/src/data_eng/stages/base.py`)

All stages extend `Stage` with:
- `name: str` — Stage identifier
- `io: StageIO` — Declares inputs (list of dataset keys) and output (single dataset key)
- `run(cfg, repo_root, symbol, dt)` — Entry point, handles idempotency
- `transform(df, dt)` — Override for simple single-input/single-output transformations

### Adding a New Stage

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

## 1. Core Workflows

### Run Main Pipeline (2026-01-06)
Executes all stages effectively in order: Bronze (Ingest) -> Silver (Compute Surfaces) -> Gold (Calibration).

```bash
# Recompute all silver/gold outputs for the replay date
nohup uv run python scripts/run_pipeline_2026_01_06.py --dt 2026-01-06 > /tmp/pipeline_2026-01-06.log 2>&1 &
tail -f /tmp/pipeline_2026-01-06.log
```

### Current Workflow (Spot-Anchored Architecture)

**Silver layer (futures MBO)**:
```bash
uv run python -m src.data_eng.runner \
  --product-type future_mbo \
  --layer silver \
  --symbol ES \
  --dates 2025-10-01:2026-01-08 \
  --workers 8 \
  --overwrite
```

**Silver layer (options MBO)**:
```bash
uv run python -m src.data_eng.runner \
  --product-type future_option_mbo \
  --layer silver \
  --symbol ES \
  --dates 2025-10-01:2026-01-08 \
  --workers 8 \
  --overwrite
```

**Gold layer (HUD normalization)**:
```bash
uv run python -m src.data_eng.runner \
  --product-type hud \
  --layer gold \
  --symbol ES \
  --dates 2025-10-01:2026-01-08 \
  --workers 1
```

Date options:
- `--dates 2025-10-01:2026-01-08` — Range (colon‑separated, inclusive)
- `--start-date` + `--end-date` — Explicit range
- `--workers N` — Parallel execution across dates

Data flow:
- Raw DBN: `backend/lake/raw/source=databento/product_type=future/symbol={root}/table=market_by_order_dbn`
- Bronze: per‑contract MBO partitions written under `backend/lake/bronze/source=databento/product_type=future_mbo/symbol={contract}/table=mbo/dt=YYYY-MM-DD`
- Silver: spot-anchored surfaces (book_snapshot_1s, wall_surface_1s, vacuum_surface_1s, radar_vacuum_1s, physics_bands_1s)
- Gold: HUD normalization calibration (physics_norm_calibration)

Contract‑day selection:
- Selection map: `backend/lake/selection/mbo_contract_day_selection.parquet`
- Built by `backend/src/data_eng/retrieval/mbo_contract_day_selector.py`
- Uses RTH 09:30–12:30 NY, dominance threshold, run trimming, liquidity floor

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

### Idempotency

Stages check for `_SUCCESS` marker in output partitions before running.

To reprocess, remove partition directories:
```bash
rm -rf backend/lake/{layer}/.../dt=YYYY-MM-DD/
```
**IMPORTANT**
- DO NOT remove anything from backend/lake/raw/*

### Dataset Configuration

`backend/src/data_eng/config/datasets.yaml` defines:
```yaml
dataset.key.name:
  path: layer/product_type=X/symbol={symbol}/table=name
  format: parquet
  partition_keys: [symbol, dt]
  contract: src/data_eng/contracts/layer/product/schema.avsc
```

Dataset keys follow pattern: `{layer}.{product_type}.{table_name}`.

### Contract Enforcement

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

### Verify System Integrity
These scripts **MUST** pass after any pipeline changes.
```bash
# 1. Futures Order Book & Radar Integrity (Validates: Snapshot valid, Best Bid < Ask, Radar alignment)
uv run python scripts/test_data_integrity.py --dt 2026-01-06

# 2. Physics & Vacuum Integrity (Validates: 0..1 normalization, Band consistency, Calibration application)
uv run python scripts/test_physics_integrity.py --dt 2026-01-06

# 3. Streaming Service Architecture (Validates: Ring buffer, Batch iteration, Arrow IPC, GEX alignment)
uv run python scripts/verify_hud_service.py
```

---

## 2. Architecture & Implementation Details

### Futures Book Engine (Canonical source)
- **Path**: `src/data_eng/stages/silver/future_mbo/book_engine.py`
- **Logic**: Applies MBO messages linearly. State persists across windows.
- **Snapshot Handling**: Handled via `F_SNAPSHOT` transition detection (Implicit Start/End).
- **Valid Flag**: `book_valid=True` ONLY when a full snapshot has been processed and we are in incremental mode.
- **Outputs**: `book_snapshot_1s`, `wall_surface_1s`, `radar_vacuum_1s` (Share exact same engine state).

### Vacuum & Physics
- **Vacuum Surface**: Computed from `wall_surface_1s` + `physics_norm_calibration`.
- **Physics Bands**: Aggregates Wall + Vacuum into AT/NEAR/MID/FAR bands. 0..1 Scores.
- **Calibration**: Gold layer built from full-day stats. Required for normalization.

### Options GEX
- **Path**: `src/data_eng/stages/silver/future_option_mbo/compute_gex_surface_1s.py`
- **Logic**: Vectorized Numba processing of Option MBO.
- **Alignment**: Joins with Futures Spot Ref explicitly.

### Streaming Service
- **Path**: `src/serving/hud_streaming.py`
- **Mechanism**: Pre-computes surfaces from Silver lake. Loads into memory cache. Simulates 1s streaming via `iter_batches`.
- **Protocol**: Arrow IPC over WebSocket / HTTP.

---

## 3. Data Lake Structure
**Root**: `lake/`
- **Bronze**: Raw MBO, Instrument Defs.
- **Silver**: Computed Surfaces (Snapshot, Wall, Radar, Vacuum, Bands, GEX).
- **Gold**: Calibration models.

---

## 4. Debugging & Maintenance
- **Debug MBO Flags**: `uv run python scripts/debug_flags.py` (Check for F_SNAPSHOT/F_LAST presence).
- **Run Radar Only**: `uv run python scripts/run_radar_only.py` (Fast iteration on radar stage).
- **Clear & Re-run**: `uv run python scripts/run_updates.py` (Targeted re-run of Snapshot+Radar stages).

**NOTE: The pipeline has transitioned from level-anchored (hardcoded P_ref like PM_HIGH) to spot-anchored (continuous stream relative to spot price for live streaming/real-time cacluation/visualization.) REMOVE/Discard any function that does not properly reference the SPOT.**