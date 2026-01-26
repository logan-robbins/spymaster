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

## Market Wind Tunnel (Unreal Engine Integration)

### 1. Launch Backend
```bash
cd backend
uv run python -m src.serving.main
```

### 2. Launch Bridge (WebSocket -> UDP)
Connects to 1Hz WebSocket stream, decodes Arrow IPC, transforms data, and emits sparse UDP packets to Unreal Engine.
```bash
cd backend
uv run python -m src.bridge.main --udp-ip 127.0.0.1 --udp-port 7777
```

### 3. Unreal Engine Receiver
- Source Code: `unreal/MarketWindTunnel/Source/MarketWindTunnel/`
- **Class**: `AMwtUdpReceiver`
- **Setup**:
    1. Create a Blueprint inheriting from `AMwtUdpReceiver`.
    2. Add a Niagara Component with System `NS_MarketWindTunnel`.
    3. The Receiver updates User Parameters: `User.WallAsk`, `User.WallBid`, `User.WallErosion`, `User.Vacuum`, `User.PhysicsSigned`, `User.GexAbs`, `User.GexImbalance`, `User.SpotHistory`, `User.SpotHead`, `User.SpotRefTick`, `User.MidPrice`, `User.BookValid`, `User.SpotLineEnabled`, `User.HistorySeconds`.
    4. Remote control hooks: `ResetSurfaces`, `SetLayerGains`, `SetLayerEnabled` (BlueprintCallable).

### 4. Remote Control API (UE 5.7)
- Enable plugins: `Remote Control API`, `Remote Control Web Interface`, `Editor Scripting Utilities`.
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
