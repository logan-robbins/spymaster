# Spymaster - LLM Ops Reference

**CRITICAL INSTRUCTIONS** 
- ONLY the README.md or *_data.json files is considered current, do not read any other MD unless specifically requested by the user. (the *_data.json files are living documents possibly being edited by multiple engineers.)
- ALL CODE IS CONSIDERED "OLD" YOU CAN OVERWRITE/DELETE/EXTEND TO ACCOMPLISTH YOUR TASK
- You have full power to regenerate data when you need to, except for raw data. Do not modify or delete or change raw data.
- YOU MUST use nohup and VERBOSE logging for long running commands and remember to check in increments of 15 seconds so you can exit it something is not working. 
- We do not create versions of functions, classes, or files or allude to updates-- we make changes directly in line and delete old comments/outdated functions and files
- We are ONLY working on 2026-01-06 (we have full MBO data for that date) 
- We are ONLY working the first hour of RTH (0930AM EST - 1030AM EST) so limit ALL data loads and data engineering to that for speed/efficiency. 
- Remember we are simulating/planning for REAL TIME MBO ingestion -> pipeline -> visualization
- Always follow the workflow backward from the entry point to find the most current implementation.
- If ANY changes are made to the features or data pipeline, **MUST** update the avro contracts, datasets.yaml, futures_data.json to match current state.
- When you are done, you MUST update README.md to reflect the CURRENT state *note that these documents are not meant to be human readable, they are for AI / LLMS to know specific commands and key information to be able to launch, run, and debug the system.*

## Constraints
- product_types: future_mbo, future_option_mbo, equity_mbo, equity_option_mbo
- dt: 2026-01-06
- session window: 09:30-09:40 ET (dev); config: `backend/src/data_eng/stages/silver/future_mbo/mbo_batches.py` (first_hour_window_ns)
- symbol: ESH6
- tick size: $0.25 (TICK_INT = 250_000_000)
- grid: ±200 ticks from spot_ref_price_int
- spot_ref_price_int: window-start on-book anchor (last trade if on-book, else nearest best bid/ask)
- rel_ticks: spot-anchored
- rel_ticks_side: side-anchored (best bid/ask at window start)
- option strike grid: $5 buckets, ±$50 from spot_ref_price_int (rel_ticks multiples of 20)
- equity bucket size: $0.50 (BUCKET_INT = 500_000_000)
- equity grid: ±$50 from spot_ref_price_int (100 buckets)
- equity option strike grid: $1 buckets, ±$25 from spot_ref_price_int (rel_ticks multiples of 2)

## Commands

### Raw data download (Databento batch API)
Two canonical scripts download ALL raw data needed for pipelines. Both use batch job lifecycle: submit -> poll -> download.

**Futures + Futures Options (GLBX.MDP3):**
```bash
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
    --start 2026-01-06 --end 2026-01-06 \
    --symbols ES \
    --include-futures \
    --options-schemas definition,mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &
```
- Downloads: futures MBO (`ES.FUT`), options definitions, 0DTE options MBO, 0DTE options statistics
- Raw output: `lake/raw/source=databento/product_type=future_mbo/`, `lake/raw/source=databento/product_type=future_option_mbo/`, `lake/raw/source=databento/dataset=definition/`
- Job tracker: `logs/futures_jobs.json`

**Equities + Equity Options (XNAS.ITCH + OPRA.PILLAR):**
```bash
cd backend
nohup uv run python scripts/batch_download_equities.py daemon \
    --start 2026-01-06 --end 2026-01-06 \
    --symbols SPY,QQQ \
    --equity-schemas mbo \
    --options-schemas definition,cmbp-1,statistics \
    --poll-interval 60 \
    --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
```
- Downloads: equity MBO (`XNAS.ITCH`), options definitions (`OPRA.PILLAR`), 0DTE options CMBP-1, 0DTE options statistics
- Raw output: `lake/raw/source=databento/product_type=equity_mbo/`, `lake/raw/source=databento/product_type=equity_option_cmbp_1/`, `lake/raw/source=databento/dataset=definition/venue=opra/`
- Job tracker: `logs/equity_options_jobs.json`

**Monitor daemon progress:**
```bash
tail -f logs/futures.log
tail -f logs/equities.log
cat logs/futures_jobs.json | jq '.jobs | to_entries | map({key: .key, state: .value.state}) | group_by(.state) | map({state: .[0].state, count: length})'
```

**Poll/download only (if daemon stopped):**
```bash
cd backend
uv run python scripts/batch_download_futures.py poll --log-file logs/futures.log
uv run python scripts/batch_download_equities.py poll --log-file logs/equities.log
```

### Data pipeline
- `cd backend`
- `uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol ESH6 --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol ESH6 --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type future_option_mbo --layer silver --symbol ES --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type future_option_mbo --layer gold --symbol ES --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type equity_mbo --layer silver --symbol SPY --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type equity_mbo --layer gold --symbol SPY --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type equity_option_mbo --layer silver --symbol SPY --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type equity_option_mbo --layer gold --symbol SPY --dt 2026-01-06 --workers 1`
- add `--overwrite` for silver/gold rebuilds

### Unified stream server (futures + options)
- `cd backend`
- `uv run python -m src.serving.velocity_main`
- `ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=2026-01-06`
- params: `speed` (playback multiplier), `skip_minutes` (skip N minutes at start)

### Frontend2 (primary)
- `cd frontend2`
- `npm install && npm run dev`
- `http://localhost:5174`

### Process checks
- `lsof -iTCP:8001 -sTCP:LISTEN`
- `lsof -iTCP:5174 -sTCP:LISTEN`

## Pipeline (future_mbo)
- BronzeIngestFutureMbo -> `bronze.future_mbo.mbo`
- SilverComputeBookStates1s -> `silver.future_mbo.book_snapshot_1s`, `silver.future_mbo.depth_and_flow_1s`
- GoldComputePhysicsSurface1s -> `gold.future_mbo.physics_surface_1s`

## Pipeline (future_option_mbo)
- BronzeIngestFutureOptionMbo -> `bronze.future_option_mbo.mbo`
- SilverComputeOptionBookStates1s -> `silver.future_option_mbo.book_snapshot_1s`, `silver.future_option_mbo.depth_and_flow_1s`
- GoldComputeOptionPhysicsSurface1s -> `gold.future_option_mbo.physics_surface_1s`

## Pipeline (equity_mbo)
- BronzeIngestEquityMbo -> `bronze.equity_mbo.mbo`
- SilverComputeEquityBookStates1s -> `silver.equity_mbo.book_snapshot_1s`, `silver.equity_mbo.depth_and_flow_1s`
- GoldComputeEquityPhysicsSurface1s -> `gold.equity_mbo.physics_surface_1s`

## Pipeline (equity_option_mbo)
- BronzeIngestEquityOptionMbo -> `bronze.equity_option_mbo.cmbp_1` (CMBP-1)
- SilverComputeEquityOptionBookStates1s -> `silver.equity_option_mbo.book_snapshot_1s`, `silver.equity_option_mbo.depth_and_flow_1s`
- GoldComputeEquityOptionPhysicsSurface1s -> `gold.equity_option_mbo.physics_surface_1s`

## Data products
- `bronze.future_mbo.mbo` -> `backend/src/data_eng/contracts/bronze/future_mbo/mbo.avsc`
- `silver.future_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/future_mbo/book_snapshot_1s.avsc`
- `silver.future_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/future_mbo/depth_and_flow_1s.avsc` (rel_ticks, rel_ticks_side)
- `gold.future_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/future_mbo/physics_surface_1s.avsc` (rel_ticks, rel_ticks_side, liquidity_velocity)
- `bronze.future_option_mbo.mbo` -> `backend/src/data_eng/contracts/bronze/future_option_mbo/mbo.avsc`
- `silver.future_option_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/future_option_mbo/book_snapshot_1s.avsc`
- `silver.future_option_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/future_option_mbo/depth_and_flow_1s.avsc`
- `gold.future_option_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/future_option_mbo/physics_surface_1s.avsc`
- `bronze.equity_mbo.mbo` -> `backend/src/data_eng/contracts/bronze/equity_mbo/mbo.avsc`
- `silver.equity_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/equity_mbo/book_snapshot_1s.avsc`
- `silver.equity_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/equity_mbo/depth_and_flow_1s.avsc` (rel_ticks, rel_ticks_side)
- `gold.equity_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/equity_mbo/physics_surface_1s.avsc`
- `bronze.equity_option_mbo.cmbp_1` -> `backend/src/data_eng/contracts/bronze/equity_option_mbo/cmbp_1.avsc`
- `silver.equity_option_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/equity_option_mbo/book_snapshot_1s.avsc`
- `silver.equity_option_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/equity_option_mbo/depth_and_flow_1s.avsc`
- `gold.equity_option_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/equity_option_mbo/physics_surface_1s.avsc`

## Streaming protocol (unified)
Per 1-second window, server sends:
1. JSON `{"type": "batch_start", "window_end_ts_ns": "...", "surfaces": ["snap", "velocity", "options", "forecast"]}`
2. JSON `{"type": "surface_header", "surface": "snap"}` + Arrow IPC (futures spot reference)
3. JSON `{"type": "surface_header", "surface": "velocity"}` + Arrow IPC (futures liquidity_velocity at $0.25 + physics fields)
4. JSON `{"type": "surface_header", "surface": "options"}` + Arrow IPC (aggregated options liquidity_velocity at $5 + physics fields)
5. JSON `{"type": "surface_header", "surface": "forecast"}` + Arrow IPC (causal lookahead 1-30s + diagnostics)

Options aggregation: C+P+A+B summed per (window_end_ts_ns, spot_ref_price_int, rel_ticks) — NET velocity per strike level

## Visualization (frontend2)
- Futures grid (VelocityGrid): green=building, red=eroding, $0.25 resolution
- Options grid (OptionsGrid): cyan=building, magenta=eroding, horizontal bars at $5 increments
- Options bar positioning: strikes above spot → bars render above strike line; below spot → below
- Spot line: turquoise line tracking price history

## Key files
- `backend/scripts/batch_download_futures.py` (raw data download: futures + futures options from GLBX.MDP3)
- `backend/scripts/batch_download_equities.py` (raw data download: equities + equity options from XNAS.ITCH + OPRA.PILLAR)
- `backend/src/data_eng/pipeline.py`
- `backend/src/data_eng/stages/silver/future_mbo/book_engine.py`
- `backend/src/data_eng/stages/gold/future_mbo/compute_physics_surface_1s.py`
- `backend/src/data_eng/stages/silver/future_option_mbo/options_book_engine.py`
- `backend/src/data_eng/stages/silver/future_option_mbo/compute_book_states_1s.py`
- `backend/src/data_eng/stages/gold/future_option_mbo/compute_physics_surface_1s.py`
- `backend/src/data_eng/stages/silver/equity_mbo/book_engine.py`
- `backend/src/data_eng/stages/silver/equity_mbo/compute_book_states_1s.py`
- `backend/src/data_eng/stages/gold/equity_mbo/compute_physics_surface_1s.py`
- `backend/src/data_eng/stages/silver/equity_option_mbo/cmbp1_book_engine.py`
- `backend/src/data_eng/stages/silver/equity_option_mbo/compute_book_states_1s.py`
- `backend/src/data_eng/stages/gold/equity_option_mbo/compute_physics_surface_1s.py`
- `backend/src/data_eng/config/datasets.yaml`
- `futures_data.json`
- `equities_data.json`
- `backend/src/serving/velocity_streaming.py`
- `backend/src/serving/velocity_main.py`
- `backend/src/serving/routers/velocity.py`
- `frontend2/src/main.ts`
- `frontend2/src/ws-client.ts`
- `frontend2/src/velocity-grid.ts`
- `frontend2/src/options-grid.ts`
- `frontend2/src/spot-line.ts`
- `frontend2/src/price-axis.ts`


## Required updates when pipeline/features change
- `backend/src/data_eng/contracts/`
- `backend/src/data_eng/config/datasets.yaml`
- `futures_data.json`
- `equities_data.json`
- `README.md`
