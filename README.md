# Spymaster

## Single Runtime Model (PRE-PROD)
Spymaster Vacuum Pressure (VP) runs one canonical runtime model.

Current state is `PRE-PROD` only because the event source adapter reads local Databento `.dbn` files.
Everything else is production-equivalent and unchanged:
- same VP math
- same in-memory event engine
- same 3-phase pipeline behavior
- same websocket payload contract
- same frontend rendering and explainability logic

When a live Databento subscription is available, the only planned change is source adapter input (`.dbn` file reader -> live Databento socket). There are no alternate math modes.

## Runtime Contract (Operator + LLM)
- Run backend commands from `backend/` with `uv run ...` only.
- Do not use `pip`, raw `python`, or ad-hoc virtualenvs.
- Run long-lived processes with `nohup ... > /tmp/<name>.log 2>&1 &`.
- For long-lived jobs, poll logs every 15 seconds and stop on stall/error.
- Raw `.dbn` data is immutable. Never edit or delete raw files unless explicitly instructed.
- ET boundaries must use `America/New_York` (handles EST/EDT correctly).

## Canonical VP Pipeline
With `--start-time HH:MM` in ET, pipeline behavior is deterministic and always identical:

1. Book bootstrap phase
- Process events from session start through warmup boundary using `apply_book_event()`.
- Builds book/spot state without running full force derivatives.
- Saves checkpoint to `backend/lake/cache/vp_book/{symbol}_{dt}_{hash}.pkl`.

2. VP warmup phase
- Process the next 30 minutes of events through full VP engine.
- No grid emission yet; used to initialize derivative state.

3. Emit phase
- Full VP engine updates and dense-grid websocket emission.

If `--start-time` is omitted, all events run through full engine from the first event and no pre-emit book bootstrap cache is used.

## Cache Behavior (Fast Startup)
Book cache key includes:
- `product_type`
- `symbol`
- `dt`
- warmup boundary (`start_time - 30m`)
- raw `.dbn` file `mtime` and `size`

Implications:
- Re-downloaded raw data auto-invalidates cache.
- Formula/math changes do not require cache rebuild (cache stores book state only).
- Changing `--start-time` creates a different cache key.
- Force rebuild: `rm -rf backend/lake/cache/vp_book/`

To launch UI quickly at `09:25` ET without rescanning from open each time:
- Keep `--start-time 09:25` consistent for that session/date/symbol.
- First run builds cache.
- Subsequent runs with same inputs reuse cache and skip pre-warmup book reconstruction.

Pre-warm cache without starting the server:
```bash
cd backend
uv run scripts/warm_cache.py --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --start-time 09:25
# Batch mode:
uv run scripts/warm_cache.py --product-type future_mbo --symbols MNQH6 ESH6 --dt 2026-02-06 --start-time 09:25
```

## VP Two-Force Math Contract
Per bucket:

Pressure (depth building / liquidity arriving):
```text
pressure = 1.0*v_add + 0.5*max(v_rest_depth, 0) + 0.3*max(a_add, 0)
```

Vacuum (depth draining / liquidity removed or consumed):
```text
vacuum = 1.0*v_pull + 1.5*v_fill + 0.5*max(-v_rest_depth, 0) + 0.3*max(a_pull, 0)
```

State interpretation:
1. vacuum above + pressure below => up bias
2. pressure above + vacuum below => down bias
3. weak/balanced => chop

No separate resistance variant exists. Pressure above spot is resistance; pressure below spot is support.

Derivative chain:
- velocity tau = 2s
- acceleration tau = 5s
- jerk tau = 10s
- continuous-time EMA alpha = `1 - exp(-dt/tau)`

## Frontend Explainability Contract
The right panel is explainability-only and derived from dense-grid pressure/vacuum aggregates.

Hard constraints:
- no legacy `signals` surface consumption
- no `5s/15s/60s` UI fields
- no projection/event-marker overlays

Panel sections:
- Net Edge: `bull_edge - bear_edge`
- Force Balance: `vac_above`, `press_above`, `press_below`, `vac_below`
- State: `UP BIAS`, `DOWN BIAS`, `UP LEAN`, `DOWN LEAN`, `CHOP`
- Trade Guidance: posture + reason + risk flag
- Depth Context: rest-depth tilt/total and force context (`NOT ENABLED` predictive model)

## End-to-End Runbook (Exact Order)

1. Set runtime variables (example)
```bash
export VP_PRODUCT_TYPE=future_mbo
export VP_SYMBOL=MNQH6
export VP_DT=2026-02-06
export VP_START_TIME=09:25
export VP_THROTTLE_MS=25
export VP_BACKEND_PORT=8002
export VP_FRONTEND_PORT=5174
```

2. Install/sync dependencies
```bash
cd backend
uv sync
cd ../frontend
npm ci
cd ..
```

3. Ensure raw `.dbn` data exists (download if missing)

Futures download daemon:
```bash
cd backend
nohup uv run scripts/batch_download_futures.py daemon \
  --start ${VP_DT} --end ${VP_DT} \
  --symbols MNQ \
  --include-futures \
  --options-schemas mbo,statistics \
  --poll-interval 60 \
  --log-file logs/futures.log > /tmp/futures_daemon.log 2>&1 &
```

Equities download daemon:
```bash
cd backend
nohup uv run scripts/batch_download_equities.py daemon \
  --start ${VP_DT} --end ${VP_DT} \
  --symbols QQQ \
  --equity-schemas mbo \
  --options-schemas cmbp-1,statistics \
  --poll-interval 60 \
  --log-file logs/equities.log > /tmp/equities_daemon.log 2>&1 &
```

Poll downloader logs every 15 seconds:
```bash
while true; do date; tail -n 40 /tmp/futures_daemon.log; sleep 15; done
```

4. Decompress any downloaded `.dbn.zst` files
```bash
find backend/lake -name "*.dbn.zst" -exec zstd -d --rm {} \;
```

5. Stop stale VP/frontend processes
```bash
kill $(lsof -t -iTCP:${VP_BACKEND_PORT}) 2>/dev/null
kill $(lsof -t -iTCP:${VP_FRONTEND_PORT}) 2>/dev/null
```

6. Start VP backend server (single PRE-PROD runtime)
```bash
cd backend
nohup uv run scripts/run_vacuum_pressure.py \
  --product-type ${VP_PRODUCT_TYPE} \
  --symbol ${VP_SYMBOL} \
  --dt ${VP_DT} \
  --port ${VP_BACKEND_PORT} \
  --start-time ${VP_START_TIME} \
  --throttle-ms ${VP_THROTTLE_MS} > /tmp/vp_preprod.log 2>&1 &
```

7. Start frontend
```bash
cd frontend
nohup npm run dev > /tmp/frontend_vp.log 2>&1 &
```

8. Poll startup logs every 15 seconds until healthy
```bash
while true; do date; tail -n 40 /tmp/vp_preprod.log; sleep 15; done
```

9. Open UI
```text
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25&throttle_ms=25
```

10. Health checks
```bash
lsof -iTCP:${VP_BACKEND_PORT} -sTCP:LISTEN
lsof -iTCP:${VP_FRONTEND_PORT} -sTCP:LISTEN
curl -s http://localhost:${VP_BACKEND_PORT}/health
ls -lah backend/lake/cache/vp_book/
```

11. Websocket endpoint
```text
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25&throttle_ms=25
```

## Debug Playbook
- Missing data file: run downloader and decompression, then restart VP.
- Product/symbol resolution error: verify `backend/src/data_eng/config/products.yaml` (futures roots) and symbol spelling.
- Cache/schema mismatch error: clear `backend/lake/cache/vp_book/` and restart.
- No grid updates yet: backend is likely still in bootstrap/warmup before `start_time` boundary.
- Backend unhealthy: check `/tmp/vp_preprod.log` first, then rerun with `--log-level DEBUG`.

## Command Surface Verification
Backend:
```bash
cd backend
uv run scripts/run_vacuum_pressure.py --help
uv run scripts/batch_download_futures.py daemon --help
uv run scripts/batch_download_equities.py daemon --help
```

Frontend:
```bash
cd frontend
npx tsc --noEmit
```

## VP Signal Analysis
Offline analysis of VP derivative signal predictive power for directional mid-price prediction:
```bash
cd backend
uv run scripts/analyze_vp_signals.py
uv run scripts/analyze_vp_signals.py --start-time 09:30 --eval-start 09:50 --eval-minutes 5
uv run scripts/analyze_vp_signals.py --help
```

Features evaluated: PV net edge (full/near/mid/far/k-weighted), depth tilt (full/near/mid/far), velocity tilts (add/pull/fill/depth), acceleration tilts, jerk magnitude/tilts. Each feature is z-scored at 4 EWM lookback windows (5/15/50/150 snapshots) and tested against 3 forward return horizons (25/100/500 snapshots) via Spearman rank IC, hit rate, and t-stat. A composite signal is built from the top-5 features by IC on the training window and evaluated on the held-out window with regime conditioning (CHOP vs DIRECTIONAL split by median jerk magnitude).

## Key Files
- VP signal analysis: `backend/scripts/analyze_vp_signals.py`
- Cache warming: `backend/scripts/warm_cache.py`
- VP entrypoint: `backend/scripts/run_vacuum_pressure.py`
- VP websocket app: `backend/src/vacuum_pressure/server.py`
- VP pipeline + cache: `backend/src/vacuum_pressure/stream_pipeline.py`
- VP event engine math: `backend/src/vacuum_pressure/event_engine.py`
- PRE-PROD source adapter (`.dbn`): `backend/src/vacuum_pressure/replay_source.py`
- Runtime config resolver: `backend/src/vacuum_pressure/config.py`
- Frontend logic: `frontend/src/vacuum-pressure.ts`
- Frontend page: `frontend/vacuum-pressure.html`
- Futures product config: `backend/src/data_eng/config/products.yaml`
- Book cache directory: `backend/lake/cache/vp_book/`
