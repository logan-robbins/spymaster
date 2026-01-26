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
# Futures (requires 6-hour warmup for state hydration)
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol ESH6 --dt 2026-01-06 --workers 1

# Options
uv run python -m src.data_eng.runner --product-type future_option_mbo --layer silver --symbol ESH6 --dates 2026-01-06 --workers 1
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
| `bucket_radar_surface_1s` | SilverComputeBucketRadar1s | Bucket Radar (2-tick) |

## Grid Structure

- **1 column = 1 second** (window cadence)
- **1 row = 1 tick** ($0.25)
- **Center = spot price** (rel_ticks = 0)
- **GEX**: $5 strike intervals = 20 ticks apart

## Layer Stack (Z-order back→front)

| Z | Layer | Color | Data Field |
|---|-------|-------|------------|
| -0.02 | Physics | Cyan above / Blue below | `physics_score_signed` |
| -0.01 | Vacuum | black | `vacuum_score` |
| 0.00 | Wall | blue asks / red bids | `depth_qty_rest`, `side` |
| 0.01 | GEX | Magenta calls / Green puts | `gex_abs`, `gex_imbalance_ratio` |
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

## Market Wind Tunnel (Unreal Engine Integration)

**Project**: `/Users/loganrobbins/Documents/Unreal Projects/MarketWindTunnel`
**Map**: `MWT_Main` (only map in project)

### Current Status
- Phase 2 (UE Ingestion): **COMPLETE**
- Phase 3 (Niagara): **PENDING** (requires manual Editor work)
- Data flow verified: Backend → Bridge → UDP → UE → Debug Visualization

### 1. Launch Backend
```bash
cd backend
uv run python -m src.serving.main
```

### 2. Launch Bridge (WebSocket -> UDP)
```bash
cd backend
uv run python -m src.bridge.main --symbol ESH6 --dt 2026-01-06
```

### 3. Unreal Engine Receiver
- **Class**: `AMwtUdpReceiver` (C++ in `unreal/MarketWindTunnel/Source/`)
- **Renderer**: `MwtHeatmapRenderer` component (Debug Draw visualization)
- **Blueprint**: `BP_MwtReceiver` (placed in MWT_Main level)
- **Port**: 7777 (UDP)

**Visualization Layers:**
- Blue boxes = Ask walls (above spot)
- Red boxes = Bid walls (below spot)
- Green/Red gradient = Physics directional ease
- Black boxes = Vacuum overlay
- Cyan line = Spot price
- Gray lines = $5 price grid

### 4. Remote Control API (UE 5.7)
```bash
# CLI Examples
cd backend
uv run python scripts/remote_control_cli.py info
uv run python scripts/remote_control_cli.py actors
uv run python scripts/remote_control_cli.py open-map --asset-path /Game/MarketWindTunnel/Maps/MWT_Main
```

### 5. What Needs Manual Editor Work
- NS_MarketWindTunnel Niagara system configuration
- Grid 2D Gas emitter setup
- User parameter array connections
- See TASK_UNREAL.md for detailed steps
- Default HTTP port: `30010` (Project Settings → Web Remote Control).
- CLI (backend): `uv run python scripts/remote_control_cli.py info`, `uv run python scripts/remote_control_cli.py actors`, `uv run python scripts/remote_control_cli.py describe --object-path <path>`, `uv run python scripts/remote_control_cli.py set --object-path <path> --property WallGain --value 1.0`.
- Use `remote/object/call` to invoke `AMwtUdpReceiver` functions for layer control.

### 5. Unreal Project Cleanup
- Cleanup script (default target: `/Users/loganrobbins/Documents/Unreal Projects/MarketWindTunnel`): `uv run python scripts/clean_unreal_project.py`.
- Removes Unreal build/cache artifacts (`Binaries`, `Intermediate`, `Saved`, `DerivedDataCache`, `.vs`, `.idea`, `.vscode`, `.xcodeproj`, `.xcworkspace`, `.sln`, `.suo`, `.VC.db`, `.VC.opendb`).

### 6. Map/Level Single-Entry Workflow
- List maps: `uv run python scripts/remote_control_cli.py maps`
- Open the single map you want: `uv run python scripts/remote_control_cli.py open-map --asset-path /Game/Maps/<YourMap>`
- Delete all other maps (keeps one): `uv run python scripts/remote_control_cli.py prune-maps --keep /Game/Maps/<YourMap>`
- Set Unreal Project Settings → `Editor Startup Map` + `Game Default Map` to the same map so there is one entry point.
- `EditorStartupMap` can be updated via Remote Control; `GameDefaultMap` may require manual Project Settings verification.

### Verification
```bash
cd backend
uv run python scripts/test_integrity_v2.py --dt 2026-01-06 
uv run python scripts/verify_websocket_stream_v1.py # Verify WS
uv run python scripts/test_udp_receiver.py # Verify Bridge UDP Output
```
