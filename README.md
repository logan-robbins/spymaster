# Spymaster - LLM Ops Reference

**CRITICAL INSTRUCTIONS** 
- ONLY the README.md or *_data.json files is considered current, do not read any other MD unless specifically requested by the user. (the *_data.json files are living documents possibly being edited by multiple engineers.)
- ALL CODE IS CONSIDERED "OLD" YOU CAN OVERWRITE/DELETE/EXTEND TO ACCOMPLISH YOUR TASK
- You have full power to regenerate data when you need to, except for raw data. Do not modify or delete or change raw data.
- YOU MUST use nohup and VERBOSE logging for long running commands and remember to check in increments of 15 seconds so you can exit if something is not working. 
- We do not create versions of functions, classes, or files or allude to updates-- we make changes directly in line and delete old comments/outdated functions and files
- We are ONLY working on 2026-01-06 (we have full MBO data for that date) 
- We are ONLY working the first hour of RTH (0930AM EST - 1030AM EST) so limit ALL data loads and data engineering to that for speed/efficiency. 
- Remember we are simulating/planning for REAL TIME MBO ingestion -> pipeline -> visualization
- Always follow the workflow backward from the entry point to find the most current implementation.
- If ANY changes are made to the features or data pipeline, **MUST** update the avro contracts, datasets.yaml, futures_data.json to match current state.
- When you are done, you MUST update README.md to reflect the CURRENT state *note that these documents are not meant to be human readable, they are for AI / LLMS to know specific commands and key information to be able to launch, run, and debug the system.*

## Quick Start (LLM Reference)

### 1. Check/Start Backend Server
```bash
# Check if backend is running
lsof -iTCP:8001 -sTCP:LISTEN

# If not running, start it:
cd /Users/loganrobbins/research/qmachina/spymaster/backend
nohup uv run python -m src.serving.velocity_main > /tmp/backend.log 2>&1 &
# Wait 3s, verify:
sleep 3 && lsof -iTCP:8001 -sTCP:LISTEN
```

### 2. Check/Start Frontend
```bash
# Check if frontend dev server is running
lsof -iTCP:5174 -sTCP:LISTEN

# If not running:
cd /Users/loganrobbins/research/qmachina/spymaster/frontend2
npm run dev &
# Or for production build:
npm run build && npm run preview &
```

### 3. View Application
- Browser: http://localhost:5174
- WebSocket: ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=2026-01-06

## Constraints
- product_types: future_mbo, future_option_mbo, equity_mbo, equity_option_mbo
- dt: 2026-01-06
- session window: 09:30-09:40 ET (dev); config: `backend/src/data_eng/stages/silver/future_mbo/mbo_batches.py` (first_hour_window_ns)
- symbol: ESH6 (use ES for bronze, ESH6 for silver/gold)
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

### MBO Contract Selection (prerequisite for silver/gold with symbol=ES)
```bash
cd backend
uv run python -m src.data_eng.retrieval.mbo_contract_day_selector --dates 2026-01-06 --output-path lake/selection/mbo_contract_day_selection.parquet
```
- Creates: `lake/selection/mbo_contract_day_selection.parquet` (maps session_date -> front-month contract)
- Required before running silver/gold with --symbol ES

### Data Pipeline
```bash
cd backend

# Bronze (raw -> normalized)
uv run python -m src.data_eng.runner --product-type future_mbo --layer bronze --symbol ES --dt 2026-01-06 --workers 1
uv run python -m src.data_eng.runner --product-type future_option_mbo --layer bronze --symbol ES --dt 2026-01-06 --workers 1

# Silver (book reconstruction)
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol ESH6 --dt 2026-01-06 --workers 1
uv run python -m src.data_eng.runner --product-type future_option_mbo --layer silver --symbol ES --dt 2026-01-06 --workers 1

# Gold (physics fields)
uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol ESH6 --dt 2026-01-06 --workers 1
uv run python -m src.data_eng.runner --product-type future_option_mbo --layer gold --symbol ES --dt 2026-01-06 --workers 1

# Add --overwrite for silver/gold rebuilds
```

### Unified Stream Server
```bash
cd backend
uv run python -m src.serving.velocity_main
```
- Serves: ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=2026-01-06
- Params: `speed` (playback multiplier), `skip_minutes` (skip N minutes at start)

### Frontend
```bash
cd frontend2
npm install
npm run dev       # Development (hot reload)
npm run build     # Production build
```
- URL: http://localhost:5174

### Process Checks
```bash
lsof -iTCP:8001 -sTCP:LISTEN   # Backend
lsof -iTCP:5174 -sTCP:LISTEN   # Frontend dev server
```

### Kill Processes
```bash
# Kill by port
kill $(lsof -t -iTCP:8001)
kill $(lsof -t -iTCP:5174)
```

## Pipeline Architecture

### future_mbo
- BronzeIngestFutureMbo -> `bronze.future_mbo.mbo`
- SilverComputeBookStates1s -> `silver.future_mbo.book_snapshot_1s`, `silver.future_mbo.depth_and_flow_1s`
- GoldComputePhysicsSurface1s -> `gold.future_mbo.physics_surface_1s`

### future_option_mbo
- BronzeIngestFutureOptionMbo -> `bronze.future_option_mbo.mbo`
- SilverComputeOptionBookStates1s -> `silver.future_option_mbo.book_snapshot_1s`, `silver.future_option_mbo.depth_and_flow_1s`
- GoldComputeOptionPhysicsSurface1s -> `gold.future_option_mbo.physics_surface_1s`

### equity_mbo
- BronzeIngestEquityMbo -> `bronze.equity_mbo.mbo`
- SilverComputeEquityBookStates1s -> `silver.equity_mbo.book_snapshot_1s`, `silver.equity_mbo.depth_and_flow_1s`
- GoldComputeEquityPhysicsSurface1s -> `gold.equity_mbo.physics_surface_1s`

### equity_option_mbo
- BronzeIngestEquityOptionMbo -> `bronze.equity_option_mbo.cmbp_1` (CMBP-1)
- SilverComputeEquityOptionBookStates1s -> `silver.equity_option_mbo.book_snapshot_1s`, `silver.equity_option_mbo.depth_and_flow_1s`
- GoldComputeEquityOptionPhysicsSurface1s -> `gold.equity_option_mbo.physics_surface_1s`

## Data Products (Contracts)
- `bronze.future_mbo.mbo` -> `backend/src/data_eng/contracts/bronze/future_mbo/mbo.avsc`
- `silver.future_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/future_mbo/book_snapshot_1s.avsc`
- `silver.future_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/future_mbo/depth_and_flow_1s.avsc`
- `gold.future_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/future_mbo/physics_surface_1s.avsc`
- `bronze.future_option_mbo.mbo` -> `backend/src/data_eng/contracts/bronze/future_option_mbo/mbo.avsc`
- `silver.future_option_mbo.book_snapshot_1s` -> `backend/src/data_eng/contracts/silver/future_option_mbo/book_snapshot_1s.avsc`
- `silver.future_option_mbo.depth_and_flow_1s` -> `backend/src/data_eng/contracts/silver/future_option_mbo/depth_and_flow_1s.avsc`
- `gold.future_option_mbo.physics_surface_1s` -> `backend/src/data_eng/contracts/gold/future_option_mbo/physics_surface_1s.avsc`

## Streaming Protocol
Per 1-second window, server sends:
1. JSON `{"type": "batch_start", "window_end_ts_ns": "...", "surfaces": ["snap", "velocity", "options", "forecast"]}`
2. JSON `{"type": "surface_header", "surface": "snap"}` + Arrow IPC (futures spot reference)
3. JSON `{"type": "surface_header", "surface": "velocity"}` + Arrow IPC (futures physics fields)
4. JSON `{"type": "surface_header", "surface": "options"}` + Arrow IPC (aggregated options velocity)
5. JSON `{"type": "surface_header", "surface": "forecast"}` + Arrow IPC (30s prediction + diagnostics)

**Options aggregation:** C+P+A+B summed per (window_end_ts_ns, spot_ref_price_int, rel_ticks) — NET velocity per strike level

**Forecast fields:** horizon_s, predicted_tick_delta, confidence, RunScore_up, RunScore_down, D_up, D_down

## Visualization (frontend2)

### Unified Pressure vs Obstacles View
Single composite shader showing liquidity dynamics:
- **Amber/orange** = Upward pressure (bid support building)
- **Teal/cyan** = Downward pressure (ask resistance building)
- **Dark maroon** = Eroding liquidity (vacuum zones)
- **Ice-white highlights** = Strong walls (high omega + building velocity)

### Forecast Line
Projects from current spot into 30% right margin:
- **Amber** = Upward prediction (positive delta)
- **Teal** = Downward prediction (negative delta)
- Direction cone at endpoint shows predicted direction
- X = horizon_s (seconds forward), Y = predicted tick delta

### Options Grid
Horizontal bars at $5 increments showing aggregate options liquidity velocity

### Spot Line
Turquoise line tracking price history through the spatiotemporal grid

### Diagnostic HUD
- RunScore ↑/↓: Average pressure in free path toward wall
- Wall distances (D_up, D_down): Ticks to nearest obstacle (Omega > 3.0)
- Confidence: Forecast confidence level

## Key Files

### Pipeline
- `backend/src/data_eng/runner.py` - Pipeline runner entry point
- `backend/src/data_eng/pipeline.py` - Stage registry
- `backend/src/data_eng/config/datasets.yaml` - Dataset definitions
- `backend/src/data_eng/stages/gold/future_mbo/compute_physics_surface_1s.py` - Futures gold stage
- `backend/src/data_eng/stages/gold/future_option_mbo/compute_physics_surface_1s.py` - Options gold stage
- `backend/src/data_eng/stages/silver/future_mbo/book_engine.py` - Futures book reconstruction
- `backend/src/data_eng/stages/silver/future_option_mbo/options_book_engine.py` - Options book reconstruction

### Streaming
- `backend/src/serving/velocity_main.py` - FastAPI server entry point
- `backend/src/serving/velocity_streaming.py` - Stream service + ForecastEngine
- `backend/src/serving/routers/velocity.py` - WebSocket endpoint

### Frontend
- `frontend2/src/main.ts` - Main entry point, callbacks, render loop
- `frontend2/src/velocity-grid.ts` - VelocityGrid class, composite shader
- `frontend2/src/options-grid.ts` - OptionsGrid class
- `frontend2/src/forecast-overlay.ts` - ForecastOverlay class
- `frontend2/src/ws-client.ts` - WebSocket client, Arrow parsing
- `frontend2/src/spot-line.ts` - SpotLine class
- `frontend2/index.html` - UI layout, HUD, legend

### Data
- `backend/scripts/batch_download_futures.py` - Raw data download (futures + options)
- `backend/scripts/batch_download_equities.py` - Raw data download (equities + options)
- `backend/src/data_eng/retrieval/mbo_contract_day_selector.py` - Contract selection
- `backend/lake/selection/mbo_contract_day_selection.parquet` - Contract selection map

### Configuration
- `futures_data.json` - Feature/transformation documentation (LIVING DOCUMENT)
- `backend/src/data_eng/config/datasets.yaml` - Dataset registry

## Debugging

### Backend Not Starting
```bash
# Check logs
tail -20 /tmp/backend.log

# Check if port in use
lsof -iTCP:8001

# Kill and restart
kill $(lsof -t -iTCP:8001)
cd backend && uv run python -m src.serving.velocity_main
```

### Pipeline Fails
```bash
# Check specific stage
cd backend
uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol ESH6 --dt 2026-01-06 --workers 1 --overwrite

# Common issues:
# - MBO selection not generated: run mbo_contract_day_selector first
# - Silver data stale: add --overwrite to rebuild
# - Missing columns: check avro contract matches code
```

### Frontend Issues
```bash
# Rebuild
cd frontend2
npm run build

# Check console in browser for errors
# Common: WebSocket connection failed -> check backend is running
```

### Data Issues
```bash
# Verify parquet exists
ls -la backend/lake/gold/future_mbo/physics_surface_1s/symbol=ESH6/dt=2026-01-06/

# Read parquet columns
cd backend
uv run python -c "import pandas as pd; df = pd.read_parquet('lake/gold/future_mbo/physics_surface_1s/symbol=ESH6/dt=2026-01-06/'); print(df.columns.tolist())"
```

## Physics Fields Reference (Gold Layer)

### Temporal (per cell: rel_ticks, side)
- `u_ema_{2,8,32,128}`: Multi-scale EMAs of liquidity_velocity
- `u_band_{fast,mid,slow}`: Differences between adjacent EMAs
- `u_wave_energy`: sqrt(sum of squared bands)
- `du_dt`, `d2u_dt2`: Temporal derivatives of u_ema_2

### Spatial (per frame: window_end_ts_ns, side)
- `u_near`: Gaussian smoothed u_ema_8 (sigma=6, win=16 ticks)
- `u_far`: Gaussian smoothed u_ema_32 (sigma=24, win=64 ticks)
- `u_prom`: u_near - u_far (prominence)
- `du_dx`, `d2u_dx2`: Spatial derivatives of u_near

### Obstacles
- `rho`: log(1 + depth_end) - density
- `phi_rest`: depth_rest / (depth_end + 1) - persistence
- `u_p`: phi_rest * u_ema_8 - persistence-weighted mid velocity
- `u_p_slow`: phi_rest * u_ema_32 - persistence-weighted slow velocity
- `Omega`: rho * (0.5 + 0.5*phi_rest) * (1 + max(0, u_p_slow)) - obstacle strength
- `Omega_near`, `Omega_far`, `Omega_prom`: Spatially smoothed obstacle fields
- `nu`: 1 + Omega_near + 2*max(0, Omega_prom) - effective viscosity
- `kappa`: 1/nu - permeability

### Pressure
- `pressure_grad`: +u_p (Bid), -u_p (Ask) - directional pressure force
