# Spymaster

## Rules
- Never modify raw data unless explicitly instructed
- Use `nohup` + logging for long-running commands, poll every 15s
- All commands from `backend/` use `uv run`. No pip, no raw python.
- Frontend uses `npm` from `frontend2/`

## Symbol Routing
- `ES`, `NQ`, `SI`, `GC`, `CL`, `6E`, `MES`, `MNQ` → futures → `scripts/batch_download_futures.py`
- `QQQ`, `AAPL`, `SPY` (OPRA tickers) → equities → `scripts/batch_download_equities.py`
- Bronze futures: parent symbol (MNQ) → resolves to contract (MNQH6)
- Bronze equities: ticker symbol (QQQ)
- Silver/Gold: always resolved contract (MNQH6) or ticker (QQQ)
- Dates: always `YYYY-MM-DD`

## Product Config
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

Product types: `future_mbo`, `future_option_mbo`, `equity_mbo`, `equity_option_cmbp_1`

---

## 1. Download Data

Raw data location: `backend/lake/raw/source=databento/product_type={product_type}/`

```bash
cd backend

# Futures + futures options
nohup uv run python scripts/batch_download_futures.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols MNQ \
    --include-futures \
    --options-schemas mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &

# Equities + equity options
nohup uv run python scripts/batch_download_equities.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols QQQ \
    --equity-schemas mbo \
    --options-schemas cmbp-1,statistics \
    --poll-interval 60 \
    --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
```

Decompress: `find backend/lake -name "*.dbn.zst" -exec zstd -d --rm {} \;`

---

## 2. Pipeline

Entry: `backend/src/data_eng/runner.py`. Flow: Bronze → Silver → Gold.

Output: `lake/{layer}/product_type={PT}/symbol={SYM}/table={TABLE}/dt={DATE}/`

```bash
cd backend

# Bronze (parent symbol, no --overwrite; delete partition dir to rebuild)
uv run python -m src.data_eng.runner --product-type future_mbo --layer bronze --symbol MNQ --dt YYYY-MM-DD --workers 4

# Silver (resolved contract, --overwrite supported)
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol MNQH6 --dt YYYY-MM-DD --workers 4

# Gold (resolved contract, --overwrite supported)
uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol MNQH6 --dt YYYY-MM-DD --workers 4
```

Dependency: `equity_option_cmbp_1` silver requires `equity_mbo` silver.

---

## 3. Calibration

```bash
cd backend
uv run python -m scripts.fit_lookahead_beta_gamma
uv run python -m scripts.eval_lookahead_beta_gamma
```

Output: `backend/data/physics/physics_beta_gamma.json`

---

## 4. Velocity Server

Requires: pipeline through gold + calibration.

```bash
cd backend
nohup uv run python -m src.serving.velocity_main > /tmp/backend.log 2>&1 &
```

WebSocket: `ws://localhost:8001/v1/velocity/stream?symbol=MNQH6&dt=YYYY-MM-DD&speed=10`

---

## 5. Vacuum Pressure Server

Three modes, same endpoint: `ws://localhost:8002/v1/vacuum-pressure/stream`

| Mode | Requires | Source |
|---|---|---|
| `event` (canonical) | Raw .dbn only | Per-MBO-event `EventDrivenVPEngine`, dense grid |
| `live` (legacy) | Raw .dbn only | Per-window `IncrementalSignalEngine`, Bernoulli lift |
| `replay` (default) | Bronze → Silver | Silver parquet |

```bash
cd backend

# Event mode (canonical, from raw .dbn)
uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo --symbol MNQH6 --dt YYYY-MM-DD \
  --port 8002 --mode event --start-time 09:30

# Live mode (legacy, from raw .dbn)
uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo --symbol MNQH6 --dt YYYY-MM-DD \
  --port 8002 --mode live --start-time 09:30

# Replay mode (from silver parquet)
uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo --symbol MNQH6 --dt YYYY-MM-DD \
  --port 8002

# Equity replay
uv run python scripts/run_vacuum_pressure.py \
  --product-type equity_mbo --symbol QQQ --dt YYYY-MM-DD \
  --port 8002

# Compute-only (save to parquet, no server)
uv run python scripts/run_vacuum_pressure.py \
  --product-type equity_mbo --symbol QQQ --dt YYYY-MM-DD --compute-only
```

Parameters: `--speed` (0=firehose, 1=realtime, 10=10x), `--start-time HH:MM` (ET, triggers warmup).

FIRE sidecar (background accuracy tracker):
```bash
cd backend
nohup uv run python scripts/vp_fire_sidecar.py \
  --product-type future_mbo --symbol MNQH6 --dt YYYY-MM-DD \
  --mode live --start-time 09:30 --speed 10 \
  --tick-target 8 --max-horizon-s 15 --print-interval-s 15 \
  --output logs/vp_fire_sidecar.jsonl > /tmp/vp_fire_sidecar.log 2>&1 &
```

---

## 6. Frontend

```bash
cd frontend2
npm install
npm run dev    # http://localhost:5174
```

Velocity: `http://localhost:5174/`

Vacuum Pressure URLs (mode/start_time must match server):
```
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=YYYY-MM-DD&mode=event&speed=10&start_time=09:30
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=YYYY-MM-DD&mode=live&speed=10&start_time=09:30
http://localhost:5174/vacuum-pressure.html?product_type=equity_mbo&symbol=QQQ&dt=YYYY-MM-DD
```

Live/replay mode tuning params (append to URL): `pre_smooth_span`, `d1_span`, `d2_span`, `d3_span`, `w_d1`, `w_d2`, `w_d3`, `projection_horizon_s`, `fast_projection_horizon_s`, `smooth_zscore_window`.

---

## 7. Tests

```bash
cd backend

# Pipeline tests (169 tests)
uv run pytest tests/test_future_mbo_gold_math.py tests/test_equity_mbo_math.py \
  tests/test_equity_option_cmbp1_pipeline.py tests/test_gold_option_physics_math.py \
  tests/test_cmbp1_book_engine.py tests/streaming/test_options_book_engine.py -v

# Event engine (32 unit + 7 invariant = 39 tests)
uv run pytest tests/test_event_engine.py tests/test_event_engine_invariants.py -v

# Vacuum pressure (incremental/config/eval/sidecar)
uv run pytest tests/test_vacuum_pressure_config.py \
  tests/test_vacuum_pressure_incremental.py \
  tests/test_vacuum_pressure_incremental_events.py \
  tests/test_vacuum_pressure_evaluation.py \
  tests/test_vacuum_pressure_fire_sidecar.py -v

# Download tests
uv run pytest tests/test_batch_download_filters.py tests/test_batch_download_futures_flow.py -v
```

Validation scripts:
```bash
cd backend
uv run python scripts/validate_event_engine.py
uv run python scripts/validate_stream_pipeline.py
uv run python scripts/validate_silver_future_mbo.py
uv run python scripts/validate_silver_equity_mbo.py
uv run python scripts/validate_silver_future_option_mbo.py --dt YYYY-MM-DD
uv run python scripts/validate_silver_equity_option_cmbp_1.py
```

Frontend type check:
```bash
cd frontend2
npx tsc --noEmit
```

---

## 8. Process Management

```bash
lsof -iTCP:8001 -sTCP:LISTEN   # Velocity server
lsof -iTCP:8002 -sTCP:LISTEN   # Vacuum pressure server
lsof -iTCP:5174 -sTCP:LISTEN   # Frontend
kill $(lsof -t -iTCP:8001) 2>/dev/null
kill $(lsof -t -iTCP:8002) 2>/dev/null
kill $(lsof -t -iTCP:5174) 2>/dev/null
curl -s http://localhost:8002/health
```

---

## 9. Key Files

| Component | Path |
|---|---|
| Pipeline runner | `backend/src/data_eng/runner.py` |
| Pipeline registry | `backend/src/data_eng/pipeline.py` |
| Product config | `backend/src/data_eng/config/products.yaml` |
| Dataset definitions | `backend/src/data_eng/config/datasets.yaml` |
| Avro contracts | `backend/src/data_eng/contracts/` |
| Stage implementations | `backend/src/data_eng/stages/{bronze,silver,gold}/{product_type}/` |
| Book engines | `backend/src/data_eng/stages/silver/{equity,future}_mbo/book_engine.py` |
| Feature docs | `futures_data.json`, `equities_data.json` |
| VP event engine | `backend/src/vacuum_pressure/event_engine.py` |
| VP incremental engine | `backend/src/vacuum_pressure/incremental.py` |
| VP batch engine | `backend/src/vacuum_pressure/engine.py` |
| VP stream pipeline | `backend/src/vacuum_pressure/stream_pipeline.py` |
| VP server | `backend/src/vacuum_pressure/server.py` |
| VP config | `backend/src/vacuum_pressure/config.py` |
| VP replay source | `backend/src/vacuum_pressure/replay_source.py` |
| VP formulas | `backend/src/vacuum_pressure/formulas.py` |
| VP CLI | `backend/scripts/run_vacuum_pressure.py` |
| VP evaluation | `backend/src/vacuum_pressure/evaluation.py` |
| VP FIRE sidecar | `backend/scripts/vp_fire_sidecar.py` |
| VP threshold eval | `backend/scripts/eval_vacuum_pressure_thresholds.py` |
| VP FIRE eval | `backend/scripts/eval_vacuum_pressure_fire.py` |
| Velocity server | `backend/src/serving/velocity_main.py` |
| Frontend entry | `frontend2/src/main.ts` |
| Frontend WS client | `frontend2/src/ws-client.ts` |
| VP frontend | `frontend2/src/vacuum-pressure.ts` |
| Download futures | `backend/scripts/batch_download_futures.py` |
| Download equities | `backend/scripts/batch_download_equities.py` |

---

## Quick Start: Vacuum Pressure Event Mode (end-to-end)

```bash
# 1. Download raw data
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
    --start 2026-02-06 --end 2026-02-06 --symbols MNQ \
    --include-futures --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &

# 2. Wait for download, decompress
find backend/lake -name "*.dbn.zst" -exec zstd -d --rm {} \;

# 3. Kill stale processes
kill $(lsof -t -iTCP:8002) 2>/dev/null
kill $(lsof -t -iTCP:5174) 2>/dev/null

# 4. Start VP server (event mode)
cd backend
nohup uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 \
  --port 8002 --mode event --start-time 09:30 > /tmp/vp_event.log 2>&1 &

# 5. Start frontend
cd frontend2
nohup npm run dev > /tmp/frontend2_vp.log 2>&1 &

# 6. Open browser
# http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&mode=event&speed=10&start_time=09:30
```

## Quick Start: Velocity (end-to-end)

```bash
# 1. Download raw futures data (see above)

# 2. Run full pipeline
cd backend
uv run python -m src.data_eng.runner --product-type future_mbo --layer bronze --symbol MNQ --dt 2026-02-06 --workers 4
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol MNQH6 --dt 2026-02-06 --workers 4
uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol MNQH6 --dt 2026-02-06 --workers 4

# 3. Calibrate
uv run python -m scripts.fit_lookahead_beta_gamma

# 4. Start velocity server
nohup uv run python -m src.serving.velocity_main > /tmp/backend.log 2>&1 &

# 5. Start frontend
cd frontend2
nohup npm run dev > /tmp/frontend2.log 2>&1 &

# 6. Open browser
# http://localhost:5174/?symbol=MNQH6&dt=2026-02-06&speed=10
```
