# Spymaster - AI Agent Reference

## Launch Commands

### Backend
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

### Pipeline (Silver Layer)
```bash
cd backend
# Futures
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol ES --dates 2026-01-06 --workers 1

# Options
uv run python -m src.data_eng.runner --product-type future_option_mbo --layer silver --symbol ES --dates 2026-01-06 --workers 1
```

## Data Flow

```
Bronze (DBN ingest) → Silver (surfaces) → WebSocket → Frontend (WebGL)
```

### Silver Datasets
| Dataset | Producer | Visualization |
|---------|----------|---------------|
| `book_snapshot_1s` | SilverComputeSnapshotAndWall1s | Spot line (cyan) |
| `wall_surface_1s` | SilverComputeSnapshotAndWall1s | Liquidity heatmap (blue/red) |
| `vacuum_surface_1s` | SilverComputeVacuumSurface1s | Erosion overlay (black) |
| `physics_surface_1s` | SilverComputePhysicsSurface1s | Per-tick directional ease (green/red) |
| `radar_vacuum_1s` | SilverComputeSnapshotAndWall1s | **NOT VISUALIZED** (ML inference) |
| `gex_surface_1s` | SilverComputeGexSurface1s | GEX heatmap (green/red) |

## Grid Structure

- **1 column = 1 second** (window cadence)
- **1 row = 1 tick** ($0.25)
- **Center = spot price** (rel_ticks = 0)
- **GEX**: $5 strike intervals = 20 ticks apart

## Layer Stack (Z-order back→front)

| Z | Layer | Color | Data Field |
|---|-------|-------|------------|
| -0.02 | Physics | green above / red below | `physics_score_signed` |
| -0.01 | Vacuum | black | `vacuum_score` |
| 0.00 | Wall | blue asks / red bids | `depth_qty_rest`, `side` |
| 0.01 | GEX | green calls / red puts | `gex_abs`, `gex_imbalance_ratio` |
| 1.00 | Spot Line | cyan | `mid_price` |

## File References

| Purpose | File |
|---------|------|
| Backend schema (LLM) | `backend_data.json` |
| Frontend schema (LLM) | `frontend_data.json` |
| Avro contracts | `backend/src/data_eng/contracts/silver/` |
| Pipeline stages | `backend/src/data_eng/stages/silver/` |
| Grid renderer | `frontend/src/hud/renderer.ts` |
| Grid layer | `frontend/src/hud/grid-layer.ts` |
| WebSocket loader | `frontend/src/hud/data-loader.ts` |

## Verification

```bash
cd backend
uv run python scripts/test_integrity_v2.py --dt 2026-01-06 # Canonical Grid/Physics Check
uv run python scripts/simulate_frontend.py
```
