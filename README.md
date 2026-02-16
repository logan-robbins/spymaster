# Spymaster

## System

One canonical runtime:

- Source adapter: Databento `.dbn` replay (PRE-PROD)
- Engine: `AbsoluteTickEngine` — pre-allocated NumPy arrays indexed by absolute price tick
- Output lattice: fixed-width time bins x absolute price ticks
- Phase-1 pressure core path: full-grid columnar snapshots (`n_absolute_ticks` rows, no radius filtering)
- Serve-time window: `±grid_radius_ticks` around spot (default ±50 = 101 visible ticks)
- Canonical state per cell: `pressure / neutral / vacuum`
- Spot is a read-only overlay computed from BBO, not coupled to engine state

There are no legacy/fallback contracts in this runtime.

## Runtime Contract

The locked instrument config (`backend/src/vacuum_pressure/instrument.yaml`) is the single source of truth.

Key fields:

- `n_absolute_ticks`: total pre-allocated absolute tick slots (default `8192`, covers ±1024 points of MNQ, 2048 total points)
- `grid_radius_ticks`: serve-time window radius in ticks around spot (default `50`, yields 101 visible ticks)
- `cell_width_ms`: fixed bin width in wall-clock milliseconds (default `100`)
- `spectrum_windows`: per-cell trailing windows (bin counts)
- `spectrum_rollup_weights`: rollup weights for `spectrum_windows`
- `spectrum_derivative_weights`: d1/d2/d3 hybrid weights
- `spectrum_tanh_scale`: tanh normalization scale
- `spectrum_threshold_neutral`: deadband threshold for state mapping
- `zscore_window_bins`, `zscore_min_periods`: robust z-score controls
- `projection_horizons_ms`: per-cell forward projection horizons

Core-grid phase-1 notes:

- `stream_core_events` in `backend/src/vacuum_pressure/core_pipeline.py` emits full-grid snapshots and skips serve-time radius extraction.
- Core mode decouples anchor initialization from BBO by seeding from the first priced event.
- Core mode defaults to fail-fast on out-of-range prices (`fail_on_out_of_range=True`).

## Constraints

- Backend commands: `cd backend && uv run ...`
- No `pip`, no raw `python`, no ad-hoc venvs
- Raw `.dbn` files are immutable
- ET time boundaries: `America/New_York`
- Single-instrument lock enforced by `VP_INSTRUMENT_CONFIG_PATH` (optional override)

## Install

```bash
cd backend && uv sync
cd frontend && npm ci
```

## Data Download

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

After download:

```bash
find backend/lake -name "*.dbn.zst" -exec zstd -d --rm {} \;
```

## Optional Cache Warmup

```bash
cd backend
uv run scripts/warm_cache.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00
```

## Start Backend

```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
cd backend
nohup uv run scripts/run_vacuum_pressure.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --port 8002 \
  --start-time 09:00 > /tmp/vp_preprod.log 2>&1 &
```

## Start Frontend

```bash
kill $(lsof -t -iTCP:5174) 2>/dev/null
cd frontend
nohup npm run dev > /tmp/frontend_vp.log 2>&1 &
```

Open:

```text
http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:00
```

## WebSocket Endpoint

```text
ws://localhost:8002/v1/vacuum-pressure/stream?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:00
```

## Stream Payload

Control frame (`runtime_config`) includes canonical config fields and schema metadata.

Each `grid_update` frame includes:

- `bin_seq`
- `bin_start_ns`
- `bin_end_ns`
- `bin_event_count`
- `event_id`
- `mid_price`, BBO, and book validity

Arrow rows include per cell:

- VP mechanics/derivatives (`add_mass`, `v_*`, `a_*`, `j_*`, `pressure_variant`, `vacuum_variant`)
- `spectrum_score` in `[-1, +1]`
- `spectrum_state_code` in `{-1,0,+1}` (`vacuum`, `neutral`, `pressure`)
- `proj_score_h{ms}` for each configured projection horizon

## Analysis Script

Canonical analysis script:

```bash
cd backend
uv run scripts/analyze_vp_signals.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00 \
  --eval-start 09:00 \
  --eval-end 12:00
```

Regime mode (directional signal + TP/SL trade evaluation):

```bash
cd backend
uv run scripts/analyze_vp_signals.py \
  --mode regime \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00 \
  --eval-start 09:00 \
  --eval-end 10:00 \
  --directional-bands 4,8,16 \
  --micro-windows 25,50,100,200 \
  --tp-ticks 8 \
  --sl-ticks 4 \
  --max-hold-snapshots 1200 \
  --json-output /tmp/regime_results.json
```

## Pressure Core Benchmark

Math-first replay benchmark (full grid, no radius filtering):

```bash
cd backend
uv run scripts/benchmark_vp_core.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00 \
  --max-bins 200
```

## Verification

```bash
cd backend && uv run scripts/run_vacuum_pressure.py --help
cd backend && uv run scripts/warm_cache.py --help
cd backend && uv run scripts/analyze_vp_signals.py --help
cd backend && uv run scripts/benchmark_vp_core.py --help
cd backend && uv run pytest -q
cd frontend && npx tsc --noEmit
```

## File Map

- `backend/src/vacuum_pressure/config.py`: locked runtime schema and validation
- `backend/src/vacuum_pressure/instrument.yaml`: canonical runtime parameters
- `backend/src/vacuum_pressure/event_engine.py`: absolute-tick VP engine (order_id map + NumPy depth/BBO arrays, supports BBO auto-anchor or explicit core anchor, fail-fast out-of-range mode)
- `backend/src/vacuum_pressure/core_pipeline.py`: phase-1 full-grid pressure-core fixed-bin stream (no radius filtering)
- `backend/src/vacuum_pressure/spectrum.py`: vectorized independent per-cell spectrum kernel
- `backend/src/vacuum_pressure/spectrum.py`: vectorized independent per-cell spectrum kernel with pre-allocated ring-history buffers (no deque/stack hot-path)
- `backend/src/vacuum_pressure/stream_pipeline.py`: fixed-bin stream builder
- `backend/src/vacuum_pressure/server.py`: websocket server + Arrow contract
- `backend/scripts/run_vacuum_pressure.py`: backend entrypoint
- `backend/scripts/benchmark_vp_core.py`: pressure-core throughput benchmark runner
- `backend/scripts/warm_cache.py`: book cache warmup
- `backend/scripts/analyze_vp_signals.py`: canonical fixed-bin analysis
- `backend/tests/test_vp_math_validation.py`: 22 math validation tests (derivative chain, composite, force model, decay, book stress, fills, modifies)
- `backend/tests/test_analyze_vp_signals_regime.py`: 11 integration tests (engine lifecycle, spectrum, pipeline)
- `frontend/src/vacuum-pressure.ts`: fixed-bin UI consumer and renderer
