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

### Frontend
```bash
cd frontend
npm run dev
# http://localhost:5173
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

## Grid Coordinates

- **X-axis**: time (1s windows)
- **Y-axis**: rel_ticks (price levels relative to spot_ref_price_int)
- **Spot anchor**: 20-tick ($5.00) grid alignment
- **Tick size**: $0.25 (TICK_INT = 250,000,000 in scaled int)
- **Range**: ±200 ticks ($50) from spot

## Process Management
```bash
# Check if backend running
lsof -iTCP:8000 -sTCP:LISTEN
pgrep -fl "src.serving.main"
```
