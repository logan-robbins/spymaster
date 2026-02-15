# Spymaster

## System

One canonical VP runtime. PRE-PROD only means source adapter reads `.dbn` files instead of live Databento socket. All math, engine, pipeline, protocol, and frontend are production-equivalent.

## Constraints

- All backend commands: `cd backend && uv run ...`. No `pip`, no raw `python`, no ad-hoc venvs.
- Long-lived processes: `nohup ... > /tmp/<name>.log 2>&1 &`. Poll every 15s, kill on stall/error.
- Raw `.dbn` files are immutable. Never edit or delete unless explicitly instructed.
- ET time boundaries: always `America/New_York` (handles EST/EDT).
- Runtime is single-instrument only, locked by `backend/src/vacuum_pressure/instrument.yaml` (override path with `VP_INSTRUMENT_CONFIG_PATH`).

## Commands

### 1. Install Dependencies

```bash
cd backend && uv sync
cd frontend && npm ci
```

### 2. Download Raw Data

Futures (substitute symbol root and date):
```bash
cd backend
nohup uv run scripts/batch_download_futures.py daemon \
  --start 2026-02-06 --end 2026-02-06 \
  --symbols MNQ \
  --include-futures \
  --options-schemas mbo,statistics \
  --poll-interval 60 \
  --log-file logs/futures.log > /tmp/futures_daemon.log 2>&1 &
```

After download completes, decompress:
```bash
find backend/lake -name "*.dbn.zst" -exec zstd -d --rm {} \;
```

### 3. Warm Book Cache (Optional, Saves Minutes on First Server Start)

Single symbol:
```bash
cd backend
uv run scripts/warm_cache.py --product-type future_mbo --symbol MNQH6 --dt 2026-02-06 --start-time 09:00
```

### 4. Start VP Server

```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
cd backend
nohup uv run scripts/run_vacuum_pressure.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --port 8002 \
  --start-time 09:25 \
  --throttle-ms 25 > /tmp/vp_preprod.log 2>&1 &
```

Parameters:
| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--product-type` | yes | — | `future_mbo` or `equity_mbo` |
| `--symbol` | no | `MNQH6` | Must match locked runtime config symbol |
| `--dt` | no | `2026-02-06` | Date `YYYY-MM-DD` |
| `--port` | no | `8002` | Server port |
| `--host` | no | `0.0.0.0` | Bind host |
| `--start-time` | no | None | Emit start `HH:MM` in ET. Enables 3-phase pipeline with book cache. If omitted: all events go through full engine from first event, no cache. |
| `--throttle-ms` | no | `25` | Minimum event-time ms between emitted grid updates |
| `--log-level` | no | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### 5. Start Frontend

```bash
kill $(lsof -t -iTCP:5174) 2>/dev/null
cd frontend
nohup npm run dev > /tmp/frontend_vp.log 2>&1 &
```

### 6. Open UI

```
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25&throttle_ms=25
```

### 7. Health Checks

```bash
curl -s http://localhost:8002/health
lsof -iTCP:8002 -sTCP:LISTEN
lsof -iTCP:5174 -sTCP:LISTEN
ls -lah backend/lake/cache/vp_book/
```

### 8. WebSocket Endpoint

```
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:25&throttle_ms=25
```

### 9. Regime Analysis (Offline, Derivative-Only)

```bash
cd backend
uv run scripts/analyze_vp_signals.py \
  --mode regime \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00 \
  --eval-start 09:00 \
  --eval-end 12:00 \
  --directional-bands 4,8,16 \
  --micro-windows 25,50,100,200 \
  --tp-ticks 8 \
  --sl-ticks 4 \
  --max-hold-snapshots 1200 \
  --projection-horizons-ms 250,500,1000,2500 \
  --projection-rollup-windows 8,32,96 \
  --projection-buckets -8,8
```

Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `regime` | Analysis mode (canonical path) |
| `--product-type` | `future_mbo` | Product type |
| `--symbol` | `MNQH6` | Must match locked runtime config symbol |
| `--dt` | `2026-02-06` | Date |
| `--start-time` | `09:00` | Capture start `HH:MM` ET |
| `--eval-start` | `09:00` | Evaluation window start `HH:MM` ET |
| `--eval-end` | `12:00` | Evaluation window end `HH:MM` ET |
| `--eval-minutes` | `None` | Optional fallback duration when `--eval-end` is omitted |
| `--throttle-ms` | `25` | Grid throttle ms |
| `--normalization-window` | `300` | Trailing snapshots for robust z-score (median/MAD) |
| `--normalization-min-periods` | `75` | Minimum trailing snapshots before z-score is valid |
| `--directional-bands` | `4,8,16` | Symmetric side bands for directional layer rollups |
| `--micro-windows` | `25,50,100,200` | Trailing windows for derivative slope rollups |
| `--spectrum-threshold` | `0.15` | Threshold for `pressure/transition/vacuum` state mapping |
| `--directional-edge-threshold` | `0.20` | Threshold for `up/down/flat` directional edge state |
| `--signal-cooldown` | `8` | Minimum snapshots between directional switch events |
| `--tp-ticks` / `--sl-ticks` | `8` / `4` | Trade-style continuation target and adverse stop |
| `--max-hold-snapshots` | `1200` | Max hold horizon before timeout |
| `--stability-max-drift` | `0.35` | Maximum relative drift across hourly buckets |
| `--stability-min-signals-per-hour` | `5` | Minimum directional events required per hour |
| `--projection-horizons-ms` | `250,500,1000,2500` | Projection horizons in milliseconds |
| `--projection-rollup-windows` | `8,32,96` | Trailing derivative rollup windows (snapshots) |
| `--projection-buckets` | `-8,8` | Relative buckets printed in projection summary |
| `--json-output` | `None` | Optional JSON metrics output path |

Outputs directional spectrum states (`pressure -> transition -> vacuum`), directional switch events, trade-style TP/SL/timeout metrics, hourly stability gate diagnostics, and per-bucket forward projection composites.

Uses `stream_events()` directly (no WebSocket, no real-time pacing). Runtime is warmup/cache dominated; see `ANALYSIS.md` for methodology and reproducibility.

To persist machine-readable metrics for replay-golden workflows:

```bash
cd backend
uv run scripts/analyze_vp_signals.py \
  --mode regime \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00 \
  --eval-start 09:00 \
  --eval-end 12:00 \
  --directional-bands 4,8,16 \
  --micro-windows 25,50,100,200 \
  --tp-ticks 8 \
  --sl-ticks 4 \
  --max-hold-snapshots 1200 \
  --projection-horizons-ms 250,500,1000,2500 \
  --projection-rollup-windows 8,32,96 \
  --projection-buckets -8,8 \
  --json-output tests/golden_mnq_20260206_0900_1200.json
```

## Pipeline Phases

When `--start-time` is provided, the pipeline has 3 deterministic phases:

1. **Book bootstrap** — `apply_book_event()` from session start to `start_time - 30min`. Builds order book state without VP math. **Cached to disk via `ensure_book_cache()`.**
2. **VP warmup** — Full VP engine from `start_time - 30min` to `start_time`. Populates derivative chains. No grid emission.
3. **Emit** — Full VP engine + dense-grid WebSocket emission at real-time pacing.

When `--start-time` is omitted: all events go through full VP engine from first event. No book cache is created or used.

## Book Cache Behavior

Function: `ensure_book_cache()` in `stream_pipeline.py`. Called by both `stream_events()` and `warm_cache.py`.

Cache path: `backend/lake/cache/vp_book/{symbol}_{dt}_{hash}.pkl`

Cache key (determines `{hash}`):
- `product_type`
- `symbol`
- `dt`
- `warmup_start_ns` (derived from `start_time - 30min`)
- `.dbn` file `st_mtime_ns`
- `.dbn` file `st_size`

**Decision tree:**

```
Is --start-time provided?
├── NO  → No cache. All events through full engine. warmup_start_ns=0.
└── YES → warmup_start_ns = start_time - 30min
          Does cache .pkl file exist at computed path?
          ├── YES → Load from disk (instant, <0.01s). Skip Phase 1 entirely.
          └── NO  → Build from scratch (process all pre-warmup events via
                     apply_book_event). Save .pkl to disk. ~6 min for MNQH6.
```

**Cache invalidation rules:**
- Re-downloading raw `.dbn` data → new mtime/size → new hash → auto-invalidates. Old cache file becomes orphaned (harmless).
- Changing `--start-time` → different `warmup_start_ns` → different hash → separate cache file per start-time.
- VP math/formula changes → NO invalidation needed. Cache stores order book state only, not VP derivatives.
- Explicit invalidation: `rm -rf backend/lake/cache/vp_book/`

**Cache is always reused** when all 6 key components match. There is no TTL, no expiry, no conditional logic beyond file existence. Once built, it persists until deleted or the raw `.dbn` file changes.

## VP Math

Per bucket (k = -50..+50 around spot):

```
pressure = 1.0*v_add + 0.5*max(v_rest_depth, 0) + 0.3*max(a_add, 0)
vacuum   = 1.0*v_pull + 1.5*v_fill + 0.5*max(-v_rest_depth, 0) + 0.3*max(a_pull, 0)
```

Both pressure and vacuum are >= 0. State interpretation:
- vacuum_above + pressure_below → UP BIAS
- pressure_above + vacuum_below → DOWN BIAS
- weak/balanced → CHOP

Derivative chain (continuous-time EMA, alpha = `1 - exp(-dt/tau)`):
- velocity: tau=2s
- acceleration: tau=5s
- jerk: tau=10s

## Frontend

Explainability-only right panel derived from dense-grid pressure/vacuum aggregates:
- Net Edge: `bull_edge - bear_edge`
- Force Balance: `vac_above`, `press_above`, `press_below`, `vac_below`
- State: `UP BIAS`, `DOWN BIAS`, `UP LEAN`, `DOWN LEAN`, `CHOP`
- Trade Guidance: posture + reason + risk flag
- Depth Context: rest-depth tilt/total

No `signals` surface, no `5s/15s/60s` fields, no projection overlays.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: Raw DBN directory not found` | Missing `.dbn` data | Run download daemon (step 2), decompress, retry |
| `Cannot extract product root from futures symbol` | Symbol not in `products.yaml` | Check `backend/src/data_eng/config/products.yaml` for known roots |
| `grid_max_ticks mismatch` | Config/frontend disagreement | Verify product config. Frontend expects K=50 |
| No grid updates after startup | Backend in warmup phase | Wait. Check `/tmp/vp_preprod.log` for warmup progress |
| Stale cache after data re-download | Should not happen | Cache auto-invalidates on mtime/size change. If persists: `rm -rf backend/lake/cache/vp_book/` |
| Backend unhealthy | Process crashed | Check `/tmp/vp_preprod.log`, rerun with `--log-level DEBUG` |

## Verification Commands

```bash
cd backend && uv run scripts/run_vacuum_pressure.py --help
cd backend && uv run scripts/warm_cache.py --help
cd backend && uv run scripts/analyze_vp_signals.py --help
cd backend && uv run pytest -q
cd backend && VP_ENABLE_REAL_REPLAY_TESTS=1 uv run pytest -q
cd backend && uv run scripts/batch_download_futures.py daemon --help
cd frontend && npx tsc --noEmit
```

## File Map

| File | Purpose |
|------|---------|
| `backend/scripts/run_vacuum_pressure.py` | VP server entrypoint |
| `backend/scripts/warm_cache.py` | Standalone book cache builder |
| `backend/scripts/analyze_vp_signals.py` | Offline derivative-only directional micro-regime analysis |
| `backend/tests/test_analyze_vp_signals_regime.py` | Real-data replay-golden and invariants for 09:00-12:00 MNQ analysis |
| `backend/scripts/batch_download_futures.py` | Futures `.dbn` download daemon |
| `backend/src/vacuum_pressure/server.py` | FastAPI WebSocket app |
| `backend/src/vacuum_pressure/stream_pipeline.py` | 3-phase pipeline, `stream_events()`, `ensure_book_cache()` |
| `backend/src/vacuum_pressure/event_engine.py` | VP engine with derivative chain math |
| `backend/src/vacuum_pressure/replay_source.py` | `.dbn` file reader, MBO event iterator |
| `backend/src/vacuum_pressure/config.py` | Runtime config resolver |
| `backend/src/vacuum_pressure/instrument.yaml` | Locked single-instrument runtime config |
| `frontend/src/vacuum-pressure.ts` | Frontend visualization + explainability |
| `frontend/vacuum-pressure.html` | Frontend page |
| `backend/lake/cache/vp_book/` | Book state cache directory |
| `ANALYSIS.md` | Derivative-only regime detection methodology and evaluation protocol |
