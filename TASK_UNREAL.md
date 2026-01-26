# Market Wind Tunnel - Unreal Implementation Task Plan

## Status: VERIFIED WORKING - Phase 3 Needs Editor Work

### System Status (Verified 2026-01-26)
- Remote Control API: **CONNECTED** (port 30010)
- MWT_Main level: **LOADED**
- BP_MwtReceiver: **PRESENT** (port 7777)
- Backend HUD stream: **RUNNING** (port 8000)
- Bridge (WS→UDP): **RUNNING** 
- Data flow: **VERIFIED** (snap/wall/vacuum/physics/gex surfaces streaming)

### Debug Visualization
The C++ receiver includes **built-in debug visualization** using DrawDebugSolidBox.
To see it: **Press Play in UE Editor while bridge is running.**
- Blue boxes = Ask walls
- Red boxes = Bid walls
- Position = relative to receiver actor location
- Height = price tick offset from spot

### Remote Control API Capabilities
**CAN DO via Remote Control:**
- Load/switch maps: `client.load_level('/Game/MarketWindTunnel/Maps/MWT_Main')`
- Read properties: `client.get_properties(object_path, property_name)`
- Write properties: `client.set_property(object_path, prop, value)`
- Call functions: `client.call_function(object_path, function_name, params)`
- Spawn actors, search assets, list actors

**CANNOT DO via Remote Control:**
- Create/configure Niagara systems (requires Editor)
- Add emitters, modules, or GPU simulation code
- Edit Blueprint graphs
- Create materials or visual assets

### Phase 2 — UE Ingestion (COMPLETE)
- [x] UDP receiver binds port 7777
- [x] Packet validation (MWT1 magic, version check)
- [x] Surface processing: SNAP, WALL, VACUUM, PHYSICS, GEX
- [x] Arrays updated (801 ticks = ±400 around spot)
- [x] Decay applied (wall instant clear, vacuum/physics τ=5s)
- [x] Debug visualization renders walls in-game

### Phase 3 — Niagara (REQUIRES EDITOR)
The NS_MarketWindTunnel asset exists but needs manual configuration:
- [ ] Add Grid 2D Gas emitter from Niagara Fluids templates
- [ ] Create User parameter arrays (Float Array Data Interface)
- [ ] Wire arrays into grid simulation (viscosity, pressure sources)
- [ ] Add particle rendering (tracer, density visualization)

### Test Commands

```bash
# Backend should already be running, but if not:
cd backend && uv run python -m src.serving.main

# Start bridge (streams data to UE on UDP 7777):
cd backend && uv run python -m src.bridge.main --symbol ESH6 --dt 2026-01-06

# Remote Control CLI examples:
cd backend && uv run python scripts/remote_control_cli.py info
cd backend && uv run python scripts/remote_control_cli.py actors
cd backend && uv run python scripts/remote_control_cli.py maps --recursive-paths
```

### What Happens When You Press Play
1. BP_MwtReceiver::BeginPlay() binds UDP socket on port 7777
2. Bridge sends MWT1 packets at ~1Hz per surface
3. AMwtUdpReceiver::OnDataReceived() processes packets
4. Arrays (WallAsk, WallBid, Vacuum, etc.) are populated
5. AMwtUdpReceiver::Tick() calls UpdateNiagara() and renders debug boxes
6. Debug visualization shows colored boxes representing liquidity walls

### Niagara Configuration (Manual Steps in Editor)
1. Open NS_MarketWindTunnel in Niagara Editor
2. Add emitter: Right-click → Add Emitter → From Template → Grid 2D Gas
3. Add User Parameters: System Overview → User Parameters:
   - User.WallAsk (Float Array)
   - User.WallBid (Float Array)
   - User.Vacuum (Float Array)
   - User.PhysicsSigned (Float Array)
   - User.GexAbs (Float Array)
   - User.GexImbalance (Float Array)
4. In Grid simulation module: Connect arrays to viscosity/velocity fields
5. Compile and save
