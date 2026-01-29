# Spymaster - AI Agent Reference

## Constraints
- **Date**: 2026-01-06 (only date with full MBO data)
- **Session**: 09:30-09:40 ET (10 min for dev), 09:30-10:30 ET (prod)
- **Symbol**: ESH6 (ES March 2026 futures)
- **Session config**: `backend/src/data_eng/stages/silver/future_mbo/mbo_batches.py:first_hour_window_ns()`

## Launch Commands

### Backend WebSocket Server
```bash
cd backend
uv run python -m src.serving.main
# WebSocket: ws://localhost:8000/v1/hud/stream?symbol=ESH6&dt=2026-01-06
```

### Frontend (Full HUD)
```bash
cd frontend
npm run dev
# http://localhost:5173
```

### Frontend2 (Velocity Grid)
```bash
cd frontend2
npm install && npm run dev
# http://localhost:5174
```

### Velocity Stream Server (for frontend2)
```bash
cd backend
uv run python -m src.serving.velocity_main
# WebSocket: ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=2026-01-06
```

### Pipeline Runner
```bash
cd backend
# future_mbo pipeline (Bronze → Silver → Gold)
uv run python -m src.data_eng.runner --product-type future_mbo --layer all --symbol ESH6 --dt 2026-01-06 --workers 1

# Single layer
uv run python -m src.data_eng.runner --product-type future_mbo --layer bronze --symbol ESH6 --dt 2026-01-06
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol ESH6 --dt 2026-01-06
uv run python -m src.data_eng.runner --product-type future_mbo --layer gold --symbol ESH6 --dt 2026-01-06
```

## Pipeline Structure (future_mbo)

```
Bronze: BronzeIngestFutureMbo
   ↓ bronze.future_mbo.mbo
Silver: SilverComputeBookStates1s
   ↓ silver.future_mbo.book_snapshot_1s (time series)
   ↓ silver.future_mbo.depth_and_flow_1s (panel: time × price_level)
Gold: GoldComputePhysicsSurface1s
   ↓ gold.future_mbo.physics_surface_1s (panel: time × rel_ticks)
Gold: GoldComputePhysicsBands1s [PENDING_REDESIGN]
   ↓ gold.future_mbo.physics_bands_1s
```

## Key Files

| Purpose | File |
|---------|------|
| Pipeline definition | `backend/src/data_eng/pipeline.py` |
| Stage implementations | `backend/src/data_eng/stages/{bronze,silver,gold}/` |
| Avro contracts | `backend/src/data_eng/contracts/` |
| Book engine (stateful) | `backend/src/data_eng/stages/silver/future_mbo/book_engine.py` |
| Feature lineage (future_mbo) | `futures_data.json` |
| Backend schema reference | `backend_data.json` |
| Frontend schema reference | `frontend_data.json` |
| Frontend docs | `DOCS_FRONTEND.md` |
| Velocity streaming | `backend/src/serving/velocity_streaming.py` |
| Velocity router | `backend/src/serving/routers/velocity.py` |
| Frontend2 velocity grid | `frontend2/src/velocity-grid.ts` |

## Grid Coordinates

- **X-axis**: time (1s windows)
- **Y-axis**: rel_ticks (price levels relative to spot_ref_price_int)
- **Spot anchor**: 20-tick ($5.00) grid alignment
- **Tick size**: $0.25 (TICK_INT = 250,000,000 in scaled int)
- **Range**: ±200 ticks ($50) from spot

## Process Management
```bash
# Check if main backend running (port 8000)
lsof -iTCP:8000 -sTCP:LISTEN
pgrep -fl "src.serving.main"

# Check if velocity server running (port 8001)
lsof -iTCP:8001 -sTCP:LISTEN
pgrep -fl "src.serving.velocity_main"

# Check if frontend2 running (port 5174)
lsof -iTCP:5174 -sTCP:LISTEN
```

## Frontend2 Architecture

Frontend2 is a TradingView-style visualization streaming from `gold.future_mbo.physics_surface_1s`.

**Display:**
- Turquoise line chart showing spot price over time
- Semi-transparent velocity heatmap overlay (green=building, red=eroding)
- Camera follows spot price vertically
- 30% right margin reserved for predictions
- Price tick labels on Y-axis ($5.00 intervals)
- Mouse wheel zoom (0.25x to 4x)

**Data Sources:**
- `silver.future_mbo.book_snapshot_1s` → snap (mid_price, spot_ref_price_int)
- `gold.future_mbo.physics_surface_1s` → velocity (rel_ticks, liquidity_velocity)

**Key Files:**
- `frontend2/src/main.ts` - Scene setup, WebSocket handlers, camera follow, zoom
- `frontend2/src/spot-line.ts` - Turquoise price line chart
- `frontend2/src/velocity-grid.ts` - Velocity heatmap with rectification shader
- `frontend2/src/price-axis.ts` - Price tick labels (HTML overlay)
- `frontend2/src/ws-client.ts` - WebSocket client with Arrow IPC parsing

**Grid:**
- Width: 1800 columns (30 min @ 1s/col)
- Height: 801 rows (±400 ticks from spot)
- Default view: 300 seconds visible, ±50 ticks around spot
- Color: Green (velocity > 0) / Red (velocity < 0)
- Alpha: |liquidity_velocity| normalized via tanh(velocity * 2) * 0.8

**Controls:**
- Mouse wheel: Zoom in/out (0.25x to 4x)

**WebSocket Protocol:**
```
batch_start (JSON) → surface_header (JSON) → Arrow IPC (binary)
```

**Query Parameters:**
- `skip_minutes` (default: 5) - Skip initial minutes of stream for faster testing
