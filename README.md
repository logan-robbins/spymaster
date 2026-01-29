# Spymaster - LLM Ops Reference

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

### Velocity stream server
- `cd backend`
- `uv run python -m src.serving.velocity_main`
- `ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=2026-01-06`

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
- `frontend2/src/main.ts`
- `frontend2/src/ws-client.ts`
- `frontend2/src/velocity-grid.ts`
- `frontend2/src/spot-line.ts`
- `frontend2/src/price-axis.ts`

## Deprecations
- `frontend/` (removed)
- `DOCS_FRONTEND.md` (legacy)

## Required updates when pipeline/features change
- `backend/src/data_eng/contracts/`
- `backend/src/data_eng/config/datasets.yaml`
- `futures_data.json`
- `README.md`
