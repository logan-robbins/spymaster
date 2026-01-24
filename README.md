# Spymaster - Streaming-First Radar Architecture (Technical Reference)

**System Role**: Expert-level guide for AI Coding Agents.
**Context**: Replay-based streaming of Databento MBO data for futures and options.
**Date**: `2026-01-06` (Single-day full scope).

## Purpose

Transforms raw market data (DBN format) through Bronze → Silver → Gold layers into feature-rich datasets and vectors. We aim to visualize market/dealer physics in near-real-time. ETUP*.png images are perfect examples of what we are trying to model.

## Environment and Workflow

- All Python work happens in `backend/` `backend/.venv/` and uses `uv` exclusively.
- The data lake lives under `backend/lake`.
- The Frontend is a React/WebGL app in `frontend/`.

---

## Module Structure

```
backend/src/data_eng/
├── config/datasets.yaml    # Dataset definitions (paths, contracts)
├── contracts/              # Avro schemas defining field contracts
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── stages/                 # Stage implementations by layer/product_type
│   ├── base.py            # Stage base class
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── pipeline.py             # Builds ordered stage lists per product_type/layer
├── runner.py               # CLI entry point
```

```
backend/src/serving/
├── hud_streaming.py        # Core Service: Reads Silver Parquet -> Simulates Stream
└── routers/
    └── hud.py              # WebSocket Endpoint (/v1/hud/stream)
```

```
frontend/src/
├── hud/
│   ├── data-loader.ts      # WebSocket Consumer (Arrow IPC)
│   ├── renderer.ts         # WebGL Visualization
│   └── state.ts            # State Management
└── main.ts                 # Entry Point & Connection Logic
```

## 1. Core Workflows

### Run Main Pipeline (2026-01-06)
Executes all stages effectively in order: Bronze (Ingest) -> Silver (Compute Surfaces) -> Gold (Calibration).

**Silver layer (Futures MBO -> Radar/Vacuum)**:
```bash
uv run python -m src.data_eng.runner \
  --product-type future_mbo \
  --layer silver \
  --symbol ES \
  --dates 2026-01-06 \
  --workers 1 \
  --overwrite
```

**Silver layer (Options MBO -> GEX)**:
```bash
# First Run Bronze (Ingest)
uv run python -m src.data_eng.runner \
  --product-type future_option_mbo \
  --layer bronze \
  --symbol ES \
  --dates 2026-01-06 \
  --workers 1

# Then Run Silver (Compute GEX)
uv run python -m src.data_eng.runner \
  --product-type future_option_mbo \
  --layer silver \
  --symbol ES \
  --dates 2026-01-06 \
  --workers 1 \
  --overwrite
```

### Key Verified Datasets (Silver)
These are the input for the Serving Layer. All are partitioned by `symbol` and `dt`.
- `silver.future_mbo.book_snapshot_1s`: Spot price anchor.
- `silver.future_mbo.wall_surface_1s`: Liquidity depth relative to spot.
- `silver.future_mbo.radar_vacuum_1s`: Aggregated physics forces (pull/add).
- `silver.future_mbo.vacuum_surface_1s`: Normalized vacuum scores (0..1).
- `silver.future_option_mbo.gex_surface_1s`: Gamma Exposure levels.

## 2. Serving Layer (Real-Time Simulation)

### Architecture
The backend serves pre-computed Silver data as if it were a live stream.
- **Source**: `HudStreamService` loads Parquet files directly from `lake/silver/...`.
- **Simulation**: `simulate_stream()` yields frames at precise 1.0s intervals using `asyncio.sleep` based on timestamp deltas.
- **Protocol**: WebSocket at `/v1/hud/stream`.
    - **Header**: JSON (Metadata: surface name, timestamp).
    - **Payload**: Binary (Arrow IPC stream).

### WebSocket Endpoint
`ws://localhost:8000/v1/hud/stream?symbol=ESH6&dt=2026-01-06`

Messages typically arrive in batches:
1. `{"type": "batch_start", "window_end_ts_ns": "..."}`
2. `{"type": "surface_header", "surface": "wall", ...}` + [Binary Arrow]
3. `{"type": "surface_header", "surface": "radar", ...}` + [Binary Arrow]
...

## 3. Frontend Integration

### Visualization Stack
- **Loader**: `frontend/src/hud/data-loader.ts` connects to WebSocket, parses disjoint JSON/Binary messages, and reassembles batches.
- **State**: `frontend/src/hud/state.ts` holds the latest "Frame" for all surfaces.
- **Renderer**: `frontend/src/hud/renderer.ts` draws heatmaps relative to the **Spot Price**.

### Running the Stack
1. **Backend**:
   ```bash
   uv run python src/main.py
   ```
2. **Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

## 4. Verification & Testing

### Integrity Scripts
These scripts **MUST** pass after any pipeline changes.
```bash
# 1. Futures Order Book & Radar Integrity (Validates: Snapshot valid, Best Bid < Ask, Radar alignment)
uv run python scripts/test_data_integrity.py --dt 2026-01-06

# 2. Physics & Vacuum Integrity (Validates: 0..1 normalization, Band consistency, Calibration application)
uv run python scripts/test_physics_integrity.py --dt 2026-01-06

# 3. GEX Statistical Inspection (Validates: Non-zero values, Strike range)
uv run python scripts/inspect_gex_values.py
```

### Simulation Verification
Verify the streaming cadence (1.0s ticks) and data payload without a browser.
```bash
# Connects as a client, logs frame deltas and row counts
uv run python scripts/simulate_frontend.py
```
**Success Criteria**:
- Output shows `Delta: ~1.00s`.
- Row counts for `Vacuum`, `Radar`, and `Wall` are non-zero.

---

**CRITICAL NOTE**: The entire architecture is **Spot-Anchored**. All spatial visualizations (Wall, Vacuum, GEX) are relative to the `spot_ref_price` in the `book_snapshot_1s` frame. Do not use absolute price levels for visualization logic unless re-anchoring.