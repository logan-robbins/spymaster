# Spymaster

## Runtime Contract (LLM)
- Backend commands: run from `backend/` with `uv run ...` only.
- Long-running processes: use `nohup ... > /tmp/<name>.log 2>&1 &`.
- Raw data is immutable. Do not modify raw `.dbn` files.
- Vacuum Pressure (VP) is live-only now: one in-memory event pipeline, dense grid fixed at `K=50` (101 rows), no replay/silver mode.
- Timezones: all ET boundaries use `America/New_York` (handles EST/EDT automatically). Do not use `Etc/GMT+5`.
- VP pipeline uses 3-phase processing: book-only fast-forward → VP warmup → live emit. Pre-warmup events build correct book state via lightweight `apply_book_event()` without VP grid computation.

## Data Download Notes
Raw path:
- `backend/lake/raw/source=databento/product_type={product_type}/...`

Symbol routing:
- Futures roots (`ES`, `NQ`, `MES`, `MNQ`, `GC`, `SI`, `CL`, `6E`) use `scripts/batch_download_futures.py`
- Equities (`QQQ`, `SPY`, `AAPL`) use `scripts/batch_download_equities.py`

Download futures raw data:
```bash
cd backend
nohup uv run scripts/batch_download_futures.py daemon \
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
nohup uv run scripts/batch_download_equities.py daemon \
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
nohup uv run scripts/run_vacuum_pressure.py \
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
cd frontend
npm ci
nohup npm run dev > /tmp/frontend_vp.log 2>&1 &
```

4. Open VP UI:
```text
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=YYYY-MM-DD&speed=1&start_time=09:30&throttle_ms=25
```

VP websocket:
```text
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=YYYY-MM-DD&speed=1&start_time=09:30&throttle_ms=25
```

## Health / Debug
```bash
lsof -iTCP:8002 -sTCP:LISTEN
lsof -iTCP:5174 -sTCP:LISTEN
curl -s http://localhost:8002/health
tail -f /tmp/vp_live.log
tail -f /tmp/frontend_vp.log
```

## Tests
VP backend sanity checks (trimmed backend):
```bash
cd backend
uv run scripts/run_vacuum_pressure.py --help
uv run scripts/batch_download_futures.py --help
uv run scripts/batch_download_equities.py --help
```

Frontend type check:
```bash
cd frontend
npm ci
npx tsc --noEmit
```

## Key Files
- VP CLI: `backend/scripts/run_vacuum_pressure.py`
- VP server: `backend/src/vacuum_pressure/server.py`
- VP stream pipeline: `backend/src/vacuum_pressure/stream_pipeline.py`
- VP event engine: `backend/src/vacuum_pressure/event_engine.py`
- VP ingest adapter (.dbn): `backend/src/vacuum_pressure/replay_source.py`
- VP config resolver: `backend/src/vacuum_pressure/config.py`
- VP frontend: `frontend/src/vacuum-pressure.ts`
- VP frontend page: `frontend/vacuum-pressure.html`
- Product config: `backend/src/data_eng/config/products.yaml`
