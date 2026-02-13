# Spymaster

## Runtime Contract (LLM)
- Backend commands: run from `backend/` with `uv run ...` only.
- Long-running processes: use `nohup ... > /tmp/<name>.log 2>&1 &`.
- Raw data is immutable. Do not modify raw `.dbn` files.
- Vacuum Pressure (VP) is live-only now: one in-memory event pipeline, dense grid fixed at `K=50` (101 rows), no replay/silver mode.

## Data Download Notes
Raw path:
- `backend/lake/raw/source=databento/product_type={product_type}/...`

Symbol routing:
- Futures roots (`ES`, `NQ`, `MES`, `MNQ`, `GC`, `SI`, `CL`, `6E`) use `scripts/batch_download_futures.py`
- Equities (`QQQ`, `SPY`, `AAPL`) use `scripts/batch_download_equities.py`

Download futures raw data:
```bash
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
  --start YYYY-MM-DD --end YYYY-MM-DD \
  --symbols MNQ \
  --include-futures \
  --options-schemas mbo,statistics \
  --poll-interval 60 \
  --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &
```

Download equities raw data:
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

Decompress:
```bash
find backend/lake -name "*.dbn.zst" -exec zstd -d --rm {} \;
```

## Bring Up VP (Current Path)
1. Kill stale VP/frontend processes:
```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
kill $(lsof -t -iTCP:5174) 2>/dev/null
```

2. Start VP live dense-grid server:
```bash
cd backend
nohup uv run python scripts/run_vacuum_pressure.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt YYYY-MM-DD \
  --port 8002 \
  --start-time 09:30 \
  --speed 1 \
  --throttle-ms 25 > /tmp/vp_live.log 2>&1 &
```

3. Start frontend:
```bash
cd frontend2
nohup npm run dev > /tmp/frontend2_vp.log 2>&1 &
```

4. Open VP UI:
```text
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=YYYY-MM-DD&speed=1&start_time=09:30&throttle_ms=25
```

VP websocket:
```text
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=YYYY-MM-DD&speed=1&start_time=09:30&throttle_ms=25
```

## Bring Up Velocity (Optional)
Requires pipeline + calibration.

Pipeline:
```bash
cd backend
uv run python -m src.data_eng.runner --product-type future_mbo --layer bronze --symbol MNQ --dt YYYY-MM-DD --workers 4
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol MNQH6 --dt YYYY-MM-DD --workers 4
uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol MNQH6 --dt YYYY-MM-DD --workers 4
```

Calibration + server:
```bash
cd backend
uv run python -m scripts.fit_lookahead_beta_gamma
nohup uv run python -m src.serving.velocity_main > /tmp/velocity.log 2>&1 &
```

Velocity websocket:
```text
ws://localhost:8001/v1/velocity/stream?symbol=MNQH6&dt=YYYY-MM-DD&speed=10
```

## Health / Debug
```bash
lsof -iTCP:8001 -sTCP:LISTEN
lsof -iTCP:8002 -sTCP:LISTEN
lsof -iTCP:5174 -sTCP:LISTEN
curl -s http://localhost:8002/health
tail -f /tmp/vp_live.log
tail -f /tmp/frontend2_vp.log
```

## Tests
VP live dense-grid engine:
```bash
cd backend
uv run pytest tests/test_event_engine.py tests/test_event_engine_invariants.py -v
```

Frontend type check:
```bash
cd frontend2
npx tsc --noEmit
```

## Key Files
- VP CLI: `backend/scripts/run_vacuum_pressure.py`
- VP server: `backend/src/vacuum_pressure/server.py`
- VP stream pipeline: `backend/src/vacuum_pressure/stream_pipeline.py`
- VP event engine: `backend/src/vacuum_pressure/event_engine.py`
- VP ingest adapter (.dbn): `backend/src/vacuum_pressure/replay_source.py`
- VP config resolver: `backend/src/vacuum_pressure/config.py`
- VP frontend: `frontend2/src/vacuum-pressure.ts`
- Product config: `backend/src/data_eng/config/products.yaml`
