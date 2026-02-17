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
- Frontend grid anchoring uses `spot_ref_price_int` (tick-rounded BBO mid); `mid_price` is display-only

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
- `projection_horizons_bins`: per-cell forward projection horizons in bin counts (derived ms = `projection_horizons_bins * cell_width_ms`)

Core-grid phase-1 notes:

- `stream_core_events` in `backend/src/vacuum_pressure/core_pipeline.py` emits full-grid snapshots and skips serve-time radius extraction.
- Core mode decouples anchor initialization from BBO by seeding from the first priced event.
- Core mode defaults to tolerant out-of-range handling (`fail_on_out_of_range=False`): out-of-range prices are skipped from depth mapping and logged with `WARNING`.
- Core mode performs a one-time soft re-anchor to raw order-book BBO at the first valid BBO after `09:30 ET` or after `10,000` processed events (whichever happens first), then keeps the grid fixed.
- Before each emitted bin snapshot (serve-time and core), the engine applies a vectorized passive time-advance to `bin_end_ns` across all active ticks (zero-delta decay + force recompute). This prevents stale per-cell state through empty bins without applying future event deltas.

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

## Offline Compute Cache (Replay Dataset)

Canonical way to persist full computed stream output for data science/replay:

```bash
cd backend
nohup uv run scripts/cache_vp_output.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --capture-start-et 09:25:00 \
  --capture-end-et 10:25:00 \
  --output-dir /tmp/vp_cache_mnqh6_20260206_0925_1025 > /tmp/vp_cache_mnqh6_20260206_0925_1025.log 2>&1 &
```

Capture output files:

- `bins.parquet`: one row per emitted fixed-width bin
- `buckets.parquet`: flattened per-bin x per-k rows (includes all VP/spectrum/projection columns)
- `manifest.json`: run parameters, row counts, config version, and output paths

Notes:

- Capture window is ET and end-exclusive (`[capture_start_et, capture_end_et)`).
- `capture_start_et` must be aligned to minute boundary (`HH:MM:00`) because warmup/start in stream pipeline is minute-based.
- Capture includes the complete emitted serve-time grid (`2*grid_radius_ticks + 1` rows per bin, default 101 around spot).
- Script suppresses high-volume out-of-range warning spam from `event_engine` so logs remain usable during long captures.

## Publish Immutable Base + Agent Workspaces

Split cached output into immutable clean grid vs projection-only experiment data:

```bash
cd backend
uv run scripts/publish_vp_research_dataset.py publish \
  --source-dir /tmp/vp_cache_mnqh6_20260206_0925_1025 \
  --dataset-id mnqh6_20260206_0925_1025 \
  --agents eda,projection,regime
```

Published layout:

- Immutable base (read-only): `backend/lake/research/vp_immutable/<dataset_id>/`
- Experiment store (writable): `backend/lake/research/vp_experiments/<dataset_id>/`

Immutable base files:

- `bins.parquet`
- `grid_clean.parquet` (all non-projection bucket columns)
- `manifest.json`
- `checksums.json` (SHA256 for immutable parquet files)

Experiment files:

- `projection_seed.parquet` (keys + projection columns only)
- Per-agent workspaces under `agents/<agent>/`:
- `data/base_immutable` symlink to immutable base
- `data/projection_experiment.parquet` writable projection copy
- `outputs/` writable output directory

Add more agents later:

```bash
cd backend
uv run scripts/publish_vp_research_dataset.py add-agents \
  --dataset-id mnqh6_20260206_0925_1025 \
  --agents alpha,beta
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

Projection experiment runtime overrides (cubic + damping):

```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
cd backend
nohup uv run scripts/run_vacuum_pressure.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --port 8002 \
  --start-time 09:25 \
  --projection-use-cubic \
  --projection-cubic-scale 0.1666666667 \
  --projection-damping-lambda 0.001 > /tmp/vp_preprod_cubic.log 2>&1 &
```

Producer-latency capture (optional, disabled by default):

```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
cd backend
nohup uv run scripts/run_vacuum_pressure.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --port 8002 \
  --start-time 09:00 \
  --perf-latency-jsonl /tmp/vp_latency_20260206_0925_0940.jsonl \
  --perf-window-start-et 09:25 \
  --perf-window-end-et 09:40 \
  --perf-summary-every-bins 200 > /tmp/vp_preprod_perf.log 2>&1 &
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

Control frame (`runtime_config`) includes canonical config fields, `projection_model` runtime overrides, and schema metadata.

Each `grid_update` frame includes:

- `bin_seq`
- `bin_start_ns`
- `bin_end_ns`
- `bin_event_count`
- `event_id`
- `spot_ref_price_int` (canonical grid anchor), `mid_price` (display), BBO, and book validity

Arrow rows include per cell:

- VP mechanics/derivatives (`add_mass`, `v_*`, `a_*`, `j_*`, `pressure_variant`, `vacuum_variant`)
- `spectrum_score` in `[-1, +1]`
- `spectrum_state_code` in `{-1,0,+1}` (`vacuum`, `neutral`, `pressure`)
- `proj_score_h{ms}` for each configured projection horizon

## Projection Bands

The frontend computes three experiment signals in-browser from the Arrow grid data:

- **ADS** (Asymmetric Derivative Slope): Multi-scale OLS slope of bid/ask velocity asymmetry
- **PFP** (Pressure Front Propagation): Inner/outer velocity lead-lag detection
- **SVac** (Spatial Vacuum Asymmetry): 1/|k| distance-weighted vacuum imbalance above vs below spot

Signals are blended (ADS=0.40, PFP=0.30, SVac=0.30) into a composite directional signal rendered as purple Gaussian bands in the right 15% of the heatmap. Horizon list comes from runtime `projection_horizons_bins` (default `[1, 2, 3, 4]`) and is converted to milliseconds via `cell_width_ms`.
Example: `cell_width_ms=300` with `projection_horizons_bins=[1,2,3,4]` yields projection columns `proj_score_h300`, `proj_score_h600`, `proj_score_h900`, `proj_score_h1200`.

Band interpretation:

- Band skews above spot = bullish prediction
- Band skews below spot = bearish prediction
- Brightness = signal strength x horizon confidence (fades with longer horizon)

Warmup is time-based (not fixed-bin): SVac after first sample, PFP at 500ms, ADS at 20s, converted to bin counts using runtime `cell_width_ms`. Historical band alignment uses fractional-bin interpolation (`horizon_ms / cell_width_ms`) and no integer rounding policy.

Source: `frontend/src/experiment-engine.ts`, `experiment-pfp.ts`, `experiment-ads.ts`, `experiment-svac.ts`, `experiment-math.ts`

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

Projection experiment sweep mode (09:25 ET to 09:35 ET):

```bash
cd backend
uv run scripts/analyze_vp_signals.py \
  --mode projection_experiment \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:25 \
  --eval-start 09:25 \
  --eval-end 09:35 \
  --experiment-use-cubic-values false,true \
  --experiment-cubic-scale-values 0.10,0.1666666667,0.22 \
  --experiment-damping-lambda-values 0.0,0.0005,0.001,0.002 \
  --experiment-regime-horizon-bins 4 \
  --experiment-slope-window-bins 30 \
  --experiment-shift-z-threshold 2.0 \
  --json-output /tmp/projection_experiment_0925_0935.json
```

## Pressure Core Benchmark

Math-first replay benchmark (full grid, no radius filtering):

```bash
cd backend
uv run scripts/benchmark_vp_core.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00
```

Optional strict mode (fail fast on out-of-range prices):

```bash
cd backend
uv run scripts/benchmark_vp_core.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00 \
  --fail-on-out-of-range
```

## Producer Latency Analysis

Replay-source producer latency is measured on the canonical websocket path:

- Start: event ingress into the fixed-bin pipeline loop (simulated from `.dbn`)
- End: handoff complete to async streaming queue (`q.put` done)

Telemetry output:

- Sidecar JSONL path from `--perf-latency-jsonl`
- One record per emitted bin in optional ET window (`--perf-window-start-et`, `--perf-window-end-et`)
- Rolling p50/p95/p99 logs every `--perf-summary-every-bins`

Key JSONL fields:

- `bin_seq`, `bin_start_ns`, `bin_end_ns`, `bin_event_count`, `event_id`
- `first_ingest_to_grid_ready_us`
- `last_ingest_to_grid_ready_us`
- `grid_ready_to_queue_put_done_us`
- `first_ingest_to_queue_put_done_us`
- `last_ingest_to_queue_put_done_us`
- `queue_block_us`

## Performance Notes

- Spectrum stage keeps robust-zscore work in pre-allocated NumPy buffers to avoid repeated large temporary allocations in the hot path.
- Serving stage Arrow encoding uses `pyarrow.RecordBatch.from_pylist(...)` + IPC stream writer, replacing per-field Python list construction loops.

## Verification

```bash
cd backend && uv run scripts/run_vacuum_pressure.py --help
cd backend && uv run scripts/warm_cache.py --help
cd backend && uv run scripts/cache_vp_output.py --help
cd backend && uv run scripts/publish_vp_research_dataset.py --help
cd backend && uv run scripts/analyze_vp_signals.py --help
cd backend && uv run scripts/benchmark_vp_core.py --help
cd backend && uv run pytest -q
cd frontend && npx tsc --noEmit
```

## File Map

- `FEATURES.md`: canonical feature definitions and formulas (engine, spectrum, serve-time grid, UI signal panel, telemetry)
- `backend/src/vacuum_pressure/config.py`: locked runtime schema and validation
- `backend/src/vacuum_pressure/instrument.yaml`: canonical runtime parameters
- `backend/src/vacuum_pressure/event_engine.py`: absolute-tick VP engine (order_id map + NumPy depth/BBO arrays, supports BBO auto-anchor or explicit core anchor, tolerant skipped-price warnings, one-time soft re-anchor helper for core mode)
- `backend/src/vacuum_pressure/core_pipeline.py`: phase-1 full-grid pressure-core fixed-bin stream (no radius filtering)
- `backend/src/vacuum_pressure/spectrum.py`: vectorized independent per-cell spectrum kernel with pre-allocated ring-history and zscore work buffers (no deque/stack hot-path)
- `backend/src/vacuum_pressure/stream_pipeline.py`: fixed-bin stream builder + optional producer-latency JSONL telemetry (ingest -> queue handoff)
- `backend/src/vacuum_pressure/server.py`: websocket server + Arrow contract (RecordBatch-based IPC serialization path)
- `backend/scripts/run_vacuum_pressure.py`: backend entrypoint (+ optional `--perf-*` latency telemetry flags and projection overrides)
- `backend/scripts/benchmark_vp_core.py`: pressure-core throughput benchmark runner
- `backend/scripts/warm_cache.py`: book cache warmup
- `backend/scripts/cache_vp_output.py`: bounded-window compute capture to parquet (`bins`, flattened `buckets`, `manifest`)
- `backend/scripts/publish_vp_research_dataset.py`: split cache into immutable clean grid store + projection experiment workspaces
- `backend/scripts/analyze_vp_signals.py`: canonical fixed-bin analysis (+ `projection_experiment` sweep mode)
- `backend/tests/test_vp_math_validation.py`: 22 math validation tests (derivative chain, composite, force model, decay, book stress, fills, modifies)
- `backend/tests/test_analyze_vp_signals_regime.py`: 11 integration tests (engine lifecycle, spectrum, pipeline)
- `backend/tests/test_stream_pipeline_perf.py`: producer-latency telemetry tests (metadata capture, JSONL output, window filtering)
- `backend/tests/test_server_arrow_serialization.py`: Arrow IPC serialization round-trip test for websocket bucket payloads
- `backend/tests/test_projection_experiments.py`: projection math + derivative-slope regime diagnostics tests
- `backend/tests/test_cache_vp_output.py`: compute-capture serialization + window filtering tests
- `backend/tests/test_publish_vp_research_dataset.py`: immutable publication + agent workspace tests
- `frontend/src/vacuum-pressure.ts`: fixed-bin UI consumer and renderer
- `frontend/src/experiment-engine.ts`: composite experiment signal blender (ADS/PFP/SVac)
- `frontend/src/experiment-ads.ts`: Asymmetric Derivative Slope signal (multi-scale OLS)
- `frontend/src/experiment-pfp.ts`: Pressure Front Propagation signal (inner/outer lead-lag)
- `frontend/src/experiment-svac.ts`: Spatial Vacuum Asymmetry signal (1/|k| distance-weighted)
- `frontend/src/experiment-math.ts`: incremental OLS slope + rolling robust z-score utilities
