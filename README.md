# Spymaster - LLM Ops Reference

## Critical Rules
- Do not modify raw data unless explicitly instructed
- Use nohup + verbose logging for long-running commands, check every 15s
- If pipeline/features change: update avro contracts, datasets.yaml, {product}_data.json

## Product Types
- future_mbo, future_option_mbo, equity_mbo, equity_option_cmbp_1

## Symbol Convention
- **Bronze (futures)**: Parent symbol (MNQ, SI) → downloader resolves front-month via FREE `symbology.resolve` API → bronze partitioned by contract (MNQH6, SIH6)
- **Bronze (equities)**: Ticker symbol (QQQ)
- **Silver/Gold**: Always pass the resolved contract (MNQH6) or ticker (QQQ) directly

---

## 1. DOWNLOADING DATA

### Raw Data Location
`backend/lake/raw/source=databento/product_type={product_type}/`

Job trackers: `backend/logs/*jobs.json`

### Current Raw Data (as of 2026-02-09)
All raw data is `.dbn` format (Databento native). Date: **2026-02-06 only**.

| Symbol | Product Types | Size |
|--------|--------------|------|
| SI | future_mbo (283 MB), future_option_mbo (1.0 GB + 5.3 MB stats), definition JSON | ~1.3 GB |
| MNQ | future_mbo (2.7 GB), future_option_mbo (1.7 GB + 6.8 MB stats), definition (820 KB) | ~4.4 GB |
| QQQ | equity_mbo (2.0 GB), equity_option_cmbp_1 (6.4 GB), statistics (1.8 MB), definition (3.7 MB) | ~8.4 GB |

Decompress: `find . -name "*.dbn.zst" -exec zstd -d --rm {} \;`

### Download Scripts
Location: `backend/scripts/batch_download_*.py`

**Futures + Futures Options:**
```bash
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols ES \
    --include-futures \
    --options-schemas mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &
```

**Equities + Equity Options:**
```bash
cd backend
nohup uv run python scripts/batch_download_equities.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols QQQ \
    --equity-schemas mbo \
    --options-schemas cmbp-1,statistics \
    --poll-interval 60 \
    --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
```

### LLM Request Routing (Instrument -> Pipeline)
- Normalize date text to `YYYY-MM-DD`. Example: `Feb 06 2026` -> `2026-02-06`.
- `ES`, `NQ`, `SI`, `GC`, `CL`, `6E`, `MNQ` → `scripts/batch_download_futures.py`
- `QQQ`, `AAPL`, `SPY` (any OPRA ticker) → `scripts/batch_download_equities.py`
- Both → run both scripts.
- Single date: `--start` and `--end` same. Multi-date: range.


---

## 2. DATA PIPELINE

### Entry Point
`backend/src/data_eng/runner.py`

### Layer Flow
Bronze (normalize) → Silver (book reconstruction) → Gold (feature engineering)

### Key Files
- Stage registry: `backend/src/data_eng/pipeline.py`
- Dataset definitions: `backend/src/data_eng/config/datasets.yaml`
- Product config: `backend/src/data_eng/config/products.yaml`
- Bronze session windows: `backend/src/data_eng/utils.py` → `session_window_ns()`
- Avro contracts: `backend/src/data_eng/contracts/`
- Stage implementations: `backend/src/data_eng/stages/{bronze,silver,gold}/{product_type}/`
- Book engines: `backend/src/data_eng/stages/silver/{equity,future}_mbo/book_engine.py`
- Filters: `backend/src/data_eng/filters/`

### Per-Product Configuration
`backend/src/data_eng/config/products.yaml`

| Root | tick_size | grid_max_ticks | strike_step | strike_ticks | multiplier |
|------|-----------|----------------|-------------|--------------|------------|
| ES   | 0.25      | 200            | $5          | 20           | 50.0       |
| MES  | 0.25      | 200            | $5          | 20           | 5.0        |
| NQ   | 0.25      | 400            | $5          | 20           | 20.0       |
| MNQ  | 0.25      | 400            | $5          | 20           | 2.0        |
| GC   | 0.10      | 200            | $5          | 50           | 100.0      |
| SI   | 0.005     | 200            | $0.25       | 50           | 5000.0     |
| CL   | 0.01      | 200            | $0.50       | 50           | 1000.0     |
| 6E   | 0.00005   | 200            | $0.005      | 100          | 125000.0   |

Runner extracts root from symbol (e.g., MNQH6 → MNQ) and passes `ProductConfig` to all stages.

### Commands
```bash
cd backend

# Bronze: parent symbol (no --overwrite; delete partition dir to rebuild)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer bronze --symbol ES --dt YYYY-MM-DD --workers 4

# Silver: full contract (--overwrite supported)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer silver --symbol ESH6 --dt YYYY-MM-DD --workers 4

# Gold: full contract (--overwrite supported)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer gold --symbol ESH6 --dt YYYY-MM-DD --workers 4
```

### Output Path Pattern
`lake/{bronze,silver,gold}/product_type={PRODUCT_TYPE}/symbol={SYMBOL}/table={TABLE}/dt={DATE}/`

### Bronze Ingestion Windows

Controlled by `session_window_ns(session_date, product_type)` in `backend/src/data_eng/utils.py`. Snapshot (`F_SNAPSHOT=32`) and Clear (`action=R`) records are exempt from the time filter (their `ts_event` preserves original order placement time, which predates the session window).

| Product Type | Bronze Window | Session Start |
|---|---|---|
| `equity_mbo` | 02:00–16:00 ET | XNAS Clear at ~03:05 ET |
| `equity_option_cmbp_1` | 02:00–16:00 ET | Same as equities |
| `future_mbo` | 00:00–24:00 UTC | GLBX snapshot at 00:00 UTC (1 Clear + ~6K snapshot Adds) |
| `future_option_mbo` | 00:00–24:00 UTC | Same as futures |

### Silver Warmup

Book engines warm up from bronze start before the output window. Must reach back past session start for zero orphan orders.

| Product Type | Warmup | Output Start | Reaches Back To |
|---|---|---|---|
| `equity_mbo` | 8 hours | 09:30 ET | 01:30 ET (before 02:00 ET bronze) |
| `future_mbo` | 15 hours | 09:30 ET (14:30 UTC) | 23:30 UTC prior day (before 00:00 UTC bronze) |

### Databento MBO Flags (`u8` bitmask)

Canonical source: `databento-python/databento/common/enums.py`

| Flag | Value | Bit | Meaning |
|---|---|---|---|
| `F_LAST` | 128 | 7 | Last record in event for instrument_id |
| `F_TOB` | 64 | 6 | Top-of-book message |
| `F_SNAPSHOT` | 32 | 5 | Sourced from replay/snapshot server |
| `F_MBP` | 16 | 4 | Aggregated price level |
| `F_BAD_TS_RECV` | 8 | 3 | ts_recv is inaccurate |
| `F_MAYBE_BAD_BOOK` | 4 | 2 | Unrecoverable gap detected |

### Current Coverage (as of 2026-02-09)
- Raw: SI, MNQ, QQQ for 2026-02-06 only (ES metadata only)
- Bronze: MNQH6 future_mbo (51.4M rows incl. 6,811 snapshot), QQQ equity_mbo (38.1M rows, 0 orphans) + equity_option_cmbp_1
- Silver: MNQH6 future_mbo (10,801 snap + 8.6M flow), QQQ equity_mbo (601 snap + 121K flow) + equity_option_cmbp_1
- Gold: Empty (needs pipeline run)

### Dependencies
- `equity_option_cmbp_1` silver requires `equity_mbo` silver (for spot reference)

---

## 3. ML

### Calibration
```bash
cd backend
uv run python -m scripts.fit_lookahead_beta_gamma
```
Output: `backend/data/physics/physics_beta_gamma.json`

```bash
uv run python -m scripts.eval_lookahead_beta_gamma
```
Output: `backend/data/physics/physics_beta_gamma_eval.json`

---

## 4. SERVING

### Backend Server
- Entry: `backend/src/serving/velocity_main.py`
- Stream service: `backend/src/serving/velocity_streaming.py`
- Endpoints: `backend/src/serving/routers/`

```bash
cd backend
nohup uv run python -m src.serving.velocity_main > /tmp/backend.log 2>&1 &
```

**WebSocket:** `ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=YYYY-MM-DD`

Query params: `speed`, `skip_minutes`

`batch_start` messages include product metadata: `tick_size`, `tick_int`, `strike_ticks`, `grid_max_ticks`.

Requires: `backend/data/physics/physics_beta_gamma.json`

---

## 5. UI

### Frontend
Location: `frontend2/`

- Entry: `src/main.ts`
- WebSocket client: `src/ws-client.ts`
- Layout: `index.html`

```bash
cd frontend2
npm install
npm run dev        # http://localhost:5174
npm run build
npm run preview
```

### Particle Wave Tester
URL: http://localhost:5175/particle-wave.html
Files: `frontend2/particle-wave.html`, `frontend2/src/particle-wave.ts`

---

## 6. FEATURE DOCUMENTATION

Canonical definitions (LIVING DOCUMENTS):
- `futures_data.json` - Futures/futures options
- `equities_data.json` - Equities/equity options

---

## 7. VALIDATION

```bash
cd backend
uv run python scripts/validate_silver_future_mbo.py
uv run python scripts/validate_silver_equity_mbo.py
uv run python scripts/validate_silver_future_option_mbo.py --dt YYYY-MM-DD
uv run python scripts/validate_silver_equity_option_cmbp_1.py
uv run python scripts/test_integrity_v2.py --symbol MNQH6 --dt YYYY-MM-DD
```

Tests: `uv run pytest tests/streaming/ -v`

---

## 8. PROCESS MANAGEMENT

```bash
lsof -iTCP:8001 -sTCP:LISTEN   # Backend (velocity server)
lsof -iTCP:8002 -sTCP:LISTEN   # Backend (vacuum pressure server)
lsof -iTCP:5174 -sTCP:LISTEN   # Frontend
kill $(lsof -t -iTCP:8001)
kill $(lsof -t -iTCP:5174)
tail -20 /tmp/backend.log
```

---

## 9. VACUUM PRESSURE DETECTOR

- Config resolver: `backend/src/vacuum_pressure/config.py`
- Formulas: `backend/src/vacuum_pressure/formulas.py`
- Engine (batch/silver): `backend/src/vacuum_pressure/engine.py`
- Incremental engine (live): `backend/src/vacuum_pressure/incremental.py`
- DBN replay source: `backend/src/vacuum_pressure/replay_source.py`
- Stream pipeline: `backend/src/vacuum_pressure/stream_pipeline.py`
- Server: `backend/src/vacuum_pressure/server.py`
- CLI: `backend/scripts/run_vacuum_pressure.py`
- FIRE sidecar (background accuracy tracker): `backend/scripts/vp_fire_sidecar.py`
- Threshold evaluator: `backend/scripts/eval_vacuum_pressure_thresholds.py`
- FIRE experiment harness: `backend/scripts/eval_vacuum_pressure_fire.py`
- Evaluation core: `backend/src/vacuum_pressure/evaluation.py`
- Frontend: `frontend2/vacuum-pressure.html`, `frontend2/src/vacuum-pressure.ts`
- Tests: `backend/tests/test_vacuum_pressure_config.py`, `backend/tests/test_vacuum_pressure_incremental.py`, `backend/tests/test_vacuum_pressure_incremental_events.py`, `backend/tests/test_vacuum_pressure_evaluation.py`, `backend/tests/test_vacuum_pressure_fire_sidecar.py`

### Canonical Runbook (No Ambiguity)

Use this exact sequence for live vacuum-pressure streaming:

1. Ensure no stale listeners:
   - `kill $(lsof -t -iTCP:8002) 2>/dev/null`
   - `kill $(lsof -t -iTCP:5174) 2>/dev/null`
2. Start backend with nohup:
   - `cd backend`
   - `nohup uv run python scripts/run_vacuum_pressure.py --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --port 8002 --mode live --start-time 09:30 --log-level INFO > /tmp/vp_live.log 2>&1 &`
3. Start frontend with nohup:
   - `cd frontend2`
   - `nohup npm run dev > /tmp/frontend2_vp.log 2>&1 &`
4. Poll both logs every 15 seconds until healthy:
   - `tail -n 60 /tmp/vp_live.log`
   - `tail -n 60 /tmp/frontend2_vp.log`
5. Start FIRE sidecar in background (tracks `FIRE -> 8 ticks`):
   - `cd backend`
   - `nohup uv run python scripts/vp_fire_sidecar.py --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --mode live --start-time 09:30 --speed 10 --tick-target 8 --max-horizon-s 15 --print-interval-s 15 --output logs/vp_fire_sidecar_mnqh6_live.jsonl --log-level INFO > /tmp/vp_fire_sidecar.log 2>&1 &`
6. Poll sidecar log every 15 seconds:
   - `tail -n 80 /tmp/vp_fire_sidecar.log`
7. Open:
   - `http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&mode=live&speed=10&start_time=09:30`

Health checks:

- Backend: `curl -s http://localhost:8002/health`
- Listeners:
  - `lsof -iTCP:8002 -sTCP:LISTEN`
  - `lsof -iTCP:5174 -sTCP:LISTEN`
- Sidecar output:
  - `tail -n 40 backend/logs/vp_fire_sidecar_mnqh6_live.jsonl`

### Streaming Modes

Two streaming modes, same WebSocket protocol and frontend:

| Mode | Source | Compute | Use Case |
|---|---|---|---|
| `replay` (default) | Silver parquet (precomputed) | Batch `run_full_pipeline` | Requires bronze->silver pipeline run |
| `live` | Raw `.dbn` file (event-by-event) | Incremental per-window | No pipeline needed, just raw data |

**Live mode architecture:**
1. `replay_source.py` reads raw `.dbn` files record-by-record via `databento.DBNStore`. Snapshot timestamps are normalized to prevent gap-fill explosion (Databento snapshots carry original order placement times spanning days).
2. `stream_pipeline.py` feeds events to the existing book engine (`FuturesBookEngine`/`EquityBookEngine`), captures window emissions via `StreamingBookAdapter`. Large gaps (>3600s) are fast-forwarded.
3. `incremental.py` computes Bernoulli lift signals per-window using stateful EMA accumulators at 3 timescales (5s/15s/60s).
4. Background thread produces windows; async consumer paces and sends over WebSocket.
5. Frontend receives identical protocol.

**`--start-time` parameter (live mode only):**
- `--start-time 09:30` means "start emitting windows at 9:30 AM EST"
- Automatically processes 30 minutes of warmup (book state + EMA warmup) before emitting
- Snapshot records are always processed regardless (they seed the initial book)
- Startup time: ~30 seconds from connect to first window at market open
- Without `--start-time`, replay starts from midnight UTC (beginning of .dbn file)
- The `start_time` param must also be in the browser URL: `&start_time=09:30`

**`speed` parameter:**
- `speed=1` = real-time (1 wall-clock second per 1-second window)
- `speed=10` = 10x (1 wall-clock second = 10 windows)
- `speed=0` = fire-hose (as fast as producer thread can generate)

### Runtime Configuration

`--product-type` is required. Config resolver sources futures from `products.yaml`, equity defaults are built-in.

| Product Type | Example Symbol | bucket_size_dollars | tick_size | qty_unit |
|---|---|---|---|---|
| equity_mbo | QQQ, SPY | $0.50 | $0.01 | shares |
| future_mbo | ESH6, MNQH6 | = tick_size | varies by root | contracts |

WebSocket sends `runtime_config` control message with full instrument config before first data batch.
Cache keys: `product_type:symbol:dt:config_version`.

### Signal Model: Bernoulli Lift (live mode)

The live-mode incremental engine (`incremental.py`) uses a fluid-dynamics-inspired model. Price moves when there is **asymmetric lift**: pressure pushing from one side into vacuum on the other, with low resistance.

**Per-window fields computed from flow_df:**
- `vacuum_above` / `vacuum_below` — liquidity drainage rate (pull > add), proximity-weighted
- `pressure_above` / `pressure_below` — active force near spot (fills + adds), proximity-weighted
- `resistance_above` / `resistance_below` — resting depth walls, proximity-weighted

**Bernoulli lift:**
- `lift_up = pressure_below * vacuum_above / (resistance_above + eps)`
- `lift_down = pressure_above * vacuum_below / (resistance_below + eps)`
- `net_lift = lift_up - lift_down` (core signal, replaces old composite)

**Multi-timescale derivatives (3 chains fed with net_lift):**
- Fast (5s): pre-smooth=3, d1_span=3, d2_span=5, projection=2s
- Medium (15s): pre-smooth=8, d1_span=8, d2_span=15, projection=10s
- Slow (60s): pre-smooth=30, d1_span=20, d2_span=40, projection=30s

**Cross-timescale confidence:** `min(magnitude) / max(magnitude)` when all 3 timescales agree in sign; 0.0 when they disagree.

**Regime classification:**
- `LIFT` — net_lift > 0.5, all timescales agree (green)
- `DRAG` — net_lift < -0.5, all timescales agree (red)
- `NEUTRAL` — balanced forces (gray)
- `CHOP` — timescales disagree (amber)

**Slope change alerts (bitmask on medium timescale):**
- `1` = INFLECTION (d1 crossed zero)
- `2` = DECELERATION (d2 opposes d1)
- `4` = REGIME_SHIFT (d2 crossed zero while d1 sustained)

### Deterministic Eventing + Feasibility

Live mode now emits explicit directional readiness/event fields (no ML fitting):

- `feasibility_up`, `feasibility_down` — bounded `[0,1]` directional feasibility from Bernoulli lift asymmetry
- `directional_bias` — bounded `[-1,1]` directional bias
- `projection_coherence` — projection sign/magnitude coherence across 5s/15s/60s
- `event_state` — `WATCH | ARMED | FIRE | COOLDOWN`
- `event_direction` — `UP | DOWN | NONE`
- `event_strength`, `event_confidence` — bounded `[0,1]` event quality scores

State machine behavior:

- `WATCH -> ARMED` after persistence on aligned `net_lift`, `d1_15s`, confidence, and projection coherence
- `ARMED -> FIRE` on stronger sustained directional thresholds
- `FIRE -> COOLDOWN` on hold failure, then refractory windows before `WATCH`

This path is deterministic/hysteretic to reduce high-frequency flicker.

Frontend event markers are directional:

- `ARMED` marker: outlined triangle pointing event direction (up/down)
- `FIRE` marker: filled triangle pointing event direction (up/down)

### Commands

```bash
cd backend

# ── LIVE MODE (from raw .dbn, no pipeline needed) ──

# Start live-mode server at market open (futures, ~30s startup)
uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 \
  --port 8002 --mode live --start-time 09:30

# Start live-mode server from beginning of session (no skip)
uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 \
  --port 8002 --mode live

# Start frontend (separate terminal)
cd ../frontend2 && npm run dev

# Start FIRE sidecar (background accuracy tracking: FIRE -> 8 ticks)
cd ../backend
nohup uv run python scripts/vp_fire_sidecar.py \
  --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 \
  --mode live --start-time 09:30 --speed 10 \
  --tick-target 8 --max-horizon-s 15 --print-interval-s 15 \
  --output logs/vp_fire_sidecar_mnqh6_live.jsonl > /tmp/vp_fire_sidecar.log 2>&1 &

# Open in browser -- start_time MUST match server's --start-time:
# http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&mode=live&speed=10&start_time=09:30
# Without start_time (from beginning):
# http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&mode=live&speed=10

# ── REPLAY MODE (from silver parquet, default) ──

# Equity: requires silver equity_mbo
uv run python -m src.data_eng.runner --product-type equity_mbo --layer silver --symbol QQQ --dt 2026-02-06 --workers 4

# Start replay server (equity, default mode)
uv run python scripts/run_vacuum_pressure.py --product-type equity_mbo --symbol QQQ --dt 2026-02-06 --port 8002

# Start replay server (futures)
uv run python scripts/run_vacuum_pressure.py --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --port 8002

# Start server with tuned smoothing/projection knobs
uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --port 8002 \
  --pre-smooth-span 15 --d1-span 10 --d2-span 20 --d3-span 40 \
  --w-d1 0.50 --w-d2 0.35 --w-d3 0.15 \
  --projection-horizon-s 10.0 --fast-projection-horizon-s 0.5 \
  --smooth-zscore-window 120

# Open equity: http://localhost:5174/vacuum-pressure.html?product_type=equity_mbo&symbol=QQQ&dt=2026-02-06
# Open futures: http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06
# Optional tuning params can be appended to URL, e.g.:
# ...&pre_smooth_span=15&d1_span=10&d2_span=20&d3_span=40&w_d1=0.50&w_d2=0.35&w_d3=0.15&projection_horizon_s=10&fast_projection_horizon_s=0.5&smooth_zscore_window=120

# ── COMPUTE-ONLY ──

# Compute-only (save to parquet)
uv run python scripts/run_vacuum_pressure.py --product-type equity_mbo --symbol QQQ --dt 2026-02-06 --compute-only

# Deterministic threshold evaluation (no model fitting)
uv run python scripts/eval_vacuum_pressure_thresholds.py \
  --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 \
  --source live --start-time 09:30 --max-windows 5000 \
  --output-json data/physics/vacuum_pressure_threshold_eval_mnqh6_open.json --pretty

# Deterministic FIRE harness (first-touch FIRE -> target ticks)
uv run python scripts/eval_vacuum_pressure_fire.py \
  --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 \
  --source replay --max-windows 5000 --horizons 15 --target-ticks 8 --top-k 3 \
  --output-json data/physics/vacuum_pressure_fire_eval_mnqh6.json --pretty

# Run tests
uv run pytest tests/test_vacuum_pressure_config.py -v
uv run pytest tests/test_vacuum_pressure_incremental.py tests/test_vacuum_pressure_incremental_events.py tests/test_vacuum_pressure_evaluation.py tests/test_vacuum_pressure_fire_sidecar.py -v
```

---

## Quick Start

### Velocity Streaming (futures, gold, physics)

1. Download data (section 1) — futures
2. Run pipeline: bronze → silver → gold (section 2) — `future_mbo`
3. Fit calibration (section 3)
4. Start backend (section 4): `nohup uv run python -m src.serving.velocity_main > /tmp/backend.log 2>&1 &`
5. Start frontend (section 5): `cd frontend2 && npm run dev`
6. Open: http://localhost:5174

### Vacuum Pressure -- Live Mode (raw .dbn, no pipeline)

1. Download raw data (section 1) -- futures or equities
2. Kill any existing server: `kill $(lsof -t -iTCP:8002) 2>/dev/null`
3. Start live server at market open (section 9):
   ```bash
   cd backend
   uv run python scripts/run_vacuum_pressure.py \
     --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 \
     --port 8002 --mode live --start-time 09:30
   ```
4. Start frontend (separate terminal): `cd frontend2 && npm run dev`
5. Start sidecar (separate terminal): `cd backend && nohup uv run python scripts/vp_fire_sidecar.py --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --mode live --start-time 09:30 --speed 10 --tick-target 8 --max-horizon-s 15 --print-interval-s 15 --output logs/vp_fire_sidecar_mnqh6_live.jsonl > /tmp/vp_fire_sidecar.log 2>&1 &`
6. Open: http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&mode=live&speed=10&start_time=09:30
7. First window appears ~30s after page load (warmup processing). `speed=1` for real-time, `speed=10` for 10x.

### Vacuum Pressure -- Replay Mode (silver parquet)

1. Download data (section 1) -- equities or futures
2. Run pipeline: bronze -> silver (section 2) -- `equity_mbo` or `future_mbo`
3. Start replay server (section 9): `uv run python scripts/run_vacuum_pressure.py --product-type equity_mbo --symbol QQQ --dt YYYY-MM-DD --port 8002`
4. Start frontend (section 5): `cd frontend2 && npm run dev`
5. Open: http://localhost:5174/vacuum-pressure.html?product_type=equity_mbo&symbol=QQQ&dt=YYYY-MM-DD
