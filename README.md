# Spymaster

## Runtime Contract (LLM)
- Backend commands: run from `backend/` with `uv run ...` only.
- Long-running processes: use `nohup ... > /tmp/<name>.log 2>&1 &`.
- Raw data is immutable. Do not modify raw `.dbn` files.
- Vacuum Pressure (VP) is live-only: one in-memory event pipeline, dense grid fixed at `K=50` (101 rows), no replay/silver mode.
- Timezones: all ET boundaries use `America/New_York` (handles EST/EDT automatically). Do not use `Etc/GMT+5`.

## VP Two-Force Model

The engine computes two non-negative force variants per price bucket:

**Pressure** (depth building — liquidity arriving):
```
pressure = 1.0·v_add + 0.5·max(v_rest_depth, 0) + 0.3·max(a_add, 0)
```

**Vacuum** (depth draining — liquidity removed/consumed):
```
vacuum = 1.0·v_pull + 1.5·v_fill + 0.5·max(-v_rest_depth, 0) + 0.3·max(a_pull, 0)
```

Three market states from spatial distribution:
1. Vacuum above spot + Pressure below → price goes UP
2. Pressure above spot + Vacuum below → price goes DOWN
3. Weak/balanced → CHOP

There is no separate `resistance_variant`. Pressure above spot IS resistance. Pressure below spot IS support.

Derivative chain: velocity (τ=2s), acceleration (τ=5s), jerk (τ=10s) via continuous-time EMA (`α = 1 - exp(-dt/τ)`). Mechanics (add/pull/fill mass) use delta-based derivatives to separate passive decay from market activity. rest_depth uses value-change derivatives.

## VP Pipeline (3-Phase Processing)

When `--start-time` is provided (e.g. `09:30`):

1. **Book-only fast-forward** — All events from session start (midnight UTC for futures) through warmup boundary are processed via lightweight `apply_book_event()` (~40k evt/s). Builds correct order book without grid/derivative computation. **Cached** after first run at `lake/cache/vp_book/{symbol}_{dt}_{hash}.pkl`.
2. **VP warmup** (30 min before start_time) — Full engine processes events to populate derivative chains. Grids not emitted.
3. **Live emit** — Full engine + grid emission to WebSocket consumers.

Book state cache:
- Auto-invalidates when raw `.dbn` data is re-downloaded (key includes file mtime + size).
- Formula changes do NOT require cache regeneration (book state is formula-independent).
- Force regeneration: `rm -rf backend/lake/cache/vp_book/`

First run with cache miss: ~6 min book build + ~9 min file scan/warmup for MNQ.
Subsequent runs with cache hit: book loads in <0.01s, then ~9 min file scan/warmup.

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

## Bring Up VP

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
  --dt 2026-02-06 \
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
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&speed=1&start_time=09:30&throttle_ms=25
```

VP websocket:
```text
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&speed=1&start_time=09:30&throttle_ms=25
```

CLI parameters:
- `--product-type`: `future_mbo` or `equity_mbo`
- `--symbol`: contract symbol (e.g. `MNQH6`, `QQQ`)
- `--dt`: session date `YYYY-MM-DD`
- `--port`: server port (default 8002)
- `--start-time`: emit start `HH:MM` in ET (warmup 30min before)
- `--speed`: replay pacing multiplier (1=realtime, 0=firehose)
- `--throttle-ms`: min event-time spacing between emitted grids (default 25)

## Health / Debug
```bash
lsof -iTCP:8002 -sTCP:LISTEN
lsof -iTCP:5174 -sTCP:LISTEN
curl -s http://localhost:8002/health
tail -f /tmp/vp_live.log
tail -f /tmp/frontend_vp.log
```

Monitor book cache:
```bash
ls -lah backend/lake/cache/vp_book/
```

Clear book cache (force rebuild on next run):
```bash
rm -rf backend/lake/cache/vp_book/
```

## Tests
VP backend sanity checks:
```bash
cd backend
uv run scripts/run_vacuum_pressure.py --help
uv run python -c "from src.vacuum_pressure.event_engine import EventDrivenVPEngine; e = EventDrivenVPEngine(K=50, tick_int=250000000); print('OK:', len(e._grid), 'buckets')"
```

Frontend type check:
```bash
cd frontend
npm ci
npx tsc --noEmit
```

## Supported Products
Futures (from `products.yaml`): `ES`, `MES`, `MNQ`, `NQ`, `SI`, `GC`, `CL`, `6E`
Equities (inline defaults): `QQQ`, `SPY` (bucket_size_dollars=0.50, tick_size=0.01)

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
- Data engineering utils: `backend/src/data_eng/utils.py`
- Book state cache: `backend/lake/cache/vp_book/*.pkl`
