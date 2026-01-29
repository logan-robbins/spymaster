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
- product_types: future_mbo, future_option_mbo
- dt: 2026-01-06
- session window: 09:30-09:40 ET (dev); config: `backend/src/data_eng/stages/silver/future_mbo/mbo_batches.py` (first_hour_window_ns)
- symbol: ESH6
- tick size: $0.25 (TICK_INT = 250_000_000)
- grid: ±200 ticks from spot_ref_price_int
- spot_ref_price_int: window-start on-book anchor (last trade if on-book, else nearest best bid/ask)
- rel_ticks: spot-anchored
- rel_ticks_side: side-anchored (best bid/ask at window start)
- option strike grid: $5 buckets, ±$50 from spot_ref_price_int (rel_ticks multiples of 20)

## Commands
### Data pipeline
- `cd backend`
- `uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol ESH6 --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol ESH6 --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type future_option_mbo --layer silver --symbol ES --dt 2026-01-06 --workers 1`
- `uv run python -m src.data_eng.runner --product-type future_option_mbo --layer gold --symbol ES --dt 2026-01-06 --workers 1`
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

## Data products
- `bronze.future_mbo.mbo` -> `backend/src/data_eng/contracts/bronze/future_mbo/mbo.avsc`
- `silver.future_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/future_mbo/book_snapshot_1s.avsc`
- `silver.future_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/future_mbo/depth_and_flow_1s.avsc` (rel_ticks, rel_ticks_side)
- `gold.future_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/future_mbo/physics_surface_1s.avsc` (rel_ticks, rel_ticks_side, liquidity_velocity)
- `bronze.future_option_mbo.mbo` -> `backend/src/data_eng/contracts/bronze/future_option_mbo/mbo.avsc`
- `silver.future_option_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/future_option_mbo/book_snapshot_1s.avsc`
- `silver.future_option_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/future_option_mbo/depth_and_flow_1s.avsc`
- `gold.future_option_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/future_option_mbo/physics_surface_1s.avsc`

## Streaming protocol (unified)
Per 1-second window, server sends:
1. JSON `{"type": "batch_start", "window_end_ts_ns": "...", "surfaces": ["snap", "velocity", "options"]}`
2. JSON `{"type": "surface_header", "surface": "snap"}` + Arrow IPC (futures spot reference)
3. JSON `{"type": "surface_header", "surface": "velocity"}` + Arrow IPC (futures liquidity_velocity at $0.25)
4. JSON `{"type": "surface_header", "surface": "options"}` + Arrow IPC (aggregated options liquidity_velocity at $5)

Options aggregation: C+P+A+B summed per (window_end_ts_ns, spot_ref_price_int, rel_ticks) — NET velocity per strike level

## Visualization (frontend2)
- Futures grid (VelocityGrid): green=building, red=eroding, $0.25 resolution
- Options grid (OptionsGrid): cyan=building, magenta=eroding, horizontal bars at $5 increments
- Options bar positioning: strikes above spot → bars render above strike line; below spot → below
- Spot line: turquoise line tracking price history

## Key files
- `backend/src/data_eng/pipeline.py`
- `backend/src/data_eng/stages/silver/future_mbo/book_engine.py`
- `backend/src/data_eng/stages/gold/future_mbo/compute_physics_surface_1s.py`
- `backend/src/data_eng/stages/silver/future_option_mbo/options_book_engine.py`
- `backend/src/data_eng/stages/silver/future_option_mbo/compute_book_states_1s.py`
- `backend/src/data_eng/stages/gold/future_option_mbo/compute_physics_surface_1s.py`
- `backend/src/data_eng/config/datasets.yaml`
- `futures_data.json`
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
- `README.md`
