# Market Wind Tunnel - Unreal Implementation Task Plan

## Status: C++ VISUALIZATION COMPLETE - Ready to Compile

### What Changed
The Niagara-based approach failed because Array Data Interfaces weren't configured.

**New Architecture**: `MwtHeatmapRenderer` component renders the visualization using **Debug Draw primitives** (immediate mode rendering). This matches the frontend's tick-native architecture and works without any Editor configuration.

### Architecture (Matches frontend_data.json / DOCS_FRONTEND.md)

**Coordinate Systems:**
- Data Space: Integer ticks, `spot_ref_price_int` anchor
- Texture Space: Y=0..801 (layer height), X=0..HistorySeconds (time)
- World Space: 1 tick = `TickWorldScale` units, 1 second = `TimeWorldScale` units

**Layer Stack (Z-order back→front):**
| Layer | World Y | Color |
|-------|---------|-------|
| Physics | -5 | Green (up ease) / Red (down ease) |
| Wall | 0 | Blue (asks) / Red (bids) |
| Vacuum | +2 | Black overlay |
| GEX | +3 | Green (calls) / Red (puts) |
| Spot Line | +5 | Cyan |
| Grid Lines | -10 | Gray (dark) |

**Dissipation Model:**
- Physics: τ=5s (decays ~18% per second)
- Vacuum: τ=5s
- Wall: τ=0 (instant clear per window)
- GEX: preserved

### Files Updated

**UE Project** (`/Users/loganrobbins/Documents/Unreal Projects/MarketWindTunnel/Source/`):
- `MwtUdpReceiver.h/cpp` - UDP receiver, data processing
- `MwtHeatmapRenderer.h/cpp` - Visualization renderer (NEW)
- `MarketWindTunnel.Build.cs` - Added ProceduralMeshComponent module

**Spymaster Repo** (`unreal/MarketWindTunnel/Source/`):
- Synced from UE project

### To Compile and Run

**Step 1: Close Unreal Editor**

**Step 2: Force Recompile (delete old binaries)**
```bash
cd "/Users/loganrobbins/Documents/Unreal Projects/MarketWindTunnel"
rm -rf Binaries Intermediate
```

**Step 3: Open Project in Unreal**
- Open `MarketWindTunnel.uproject`
- Wait for shader compilation
- It will prompt to recompile - click Yes

**Step 4: Delete Old Blueprint**
- In Content Browser: Delete `BP_MwtReceiver`
- The old BP references NiagaraComponent which is no longer used

**Step 5: Create New Blueprint**
- Right-click in Content Browser → Blueprint Class
- Search for `MwtUdpReceiver` → Select it
- Name it `BP_MwtReceiver`
- Open BP, verify `HeatmapRenderer` component exists

**Step 6: Place in Level**
- Open `MWT_Main` level
- Drag `BP_MwtReceiver` into level
- Position at origin (0, 0, 0)
- Save level

**Step 7: Start Backend + Bridge**
```bash
# Terminal 1: Backend
cd backend && uv run python -m src.serving.main

# Terminal 2: Bridge
cd backend && uv run python -m src.bridge.main --symbol ESH6 --dt 2026-01-06
```

**Step 8: Press Play in UE**
- You should see colored boxes appearing:
  - Blue/Cyan boxes = Ask walls (above spot)
  - Red boxes = Bid walls (below spot)
  - Green/Red gradient = Physics ease
  - Black overlay = Vacuum
  - Cyan line = Spot price
  - Gray grid lines = $5 price levels

### Configurable Properties (in BP_MwtReceiver Details)

**MwtHeatmapRenderer:**
- `TickWorldScale`: World units per tick (default 0.5)
- `TimeWorldScale`: World units per second (default 2.0)
- `HistorySeconds`: Visible time history (default 300 = 5 min)
- `WallIntensityMult`: Wall brightness (default 2.0)
- `bShowWall/Vacuum/Physics/Gex/SpotLine/GridLines`: Toggle layers

**MwtUdpReceiver:**
- `Port`: UDP port (default 7777)
- `bLogPackets`: Debug logging
- `bLogSurfaceUpdates`: Surface update logging

### Verification

**Check Output Log for:**
```
MWT UDP Receiver: Listening on port 7777
MWT: New window ts=..., spot=... ticks
```

**If No Visualization:**
1. Check bridge is running (`ps aux | grep bridge`)
2. Check UDP port not blocked (try `netstat -an | grep 7777`)
3. Check `bShow*` properties are true in BP
4. Camera may need repositioning to see the visualization

### Data Flow
```
Backend HUD (ws://localhost:8000/v1/hud/stream)
    ↓ (Arrow IPC)
Bridge (Python - src/bridge/main.py)
    ↓ (MWT-UDP v1 protocol)
MwtUdpReceiver (C++ - port 7777)
    ↓ (TArray<float> arrays)
MwtHeatmapRenderer (C++ - Debug Draw)
    ↓
Viewport (colored boxes)
```
