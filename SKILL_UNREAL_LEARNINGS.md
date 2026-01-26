# Session Learnings: Unreal Engine Market Wind Tunnel

## Context

This document captures detailed learnings from implementing a real-time market data visualization in Unreal Engine 5.7, streaming data from a Python backend via UDP.

---

## 1. The Problem We Solved

**Goal**: Replicate a WebGL frontend visualization (tick-native heatmap with multiple layers) in Unreal Engine.

**Frontend Reference** (from DOCS_FRONTEND.md):
- Cyan spot line at current price
- Green gradient above spot (asks/upward ease)
- Red gradient below spot (bids/downward ease)
- Multiple stacked layers: Wall, Physics, Vacuum, GEX
- Temporal dissipation (older data fades)
- Ring buffer time history (scrolls left)

**Challenges**:
1. No direct way to configure Niagara User Parameters via Remote Control
2. C++ code sync issues between workspace and UE project
3. Scale/camera positioning for visibility
4. Efficient data streaming over UDP

---

## 2. Architecture Decisions

### Why UDP Over WebSocket to UE?
- UE5 doesn't have native WebSocket client for Blueprint/C++
- UDP is simpler to implement, lower latency
- Packet loss acceptable for visualization (next frame corrects)
- Binary protocol more efficient than JSON

### Why Debug Draw Over Niagara (V1)?
- Niagara requires manual Editor configuration for Data Interfaces
- Remote Control cannot create User Parameters
- Debug Draw works immediately without asset setup
- Easier to debug (can log every draw call)
- Trade-off: less performant, but acceptable for prototype

### Future: Niagara with Pre-configured System
For production, create the Niagara System manually once:
1. Add User Parameters: `WallAsk`, `WallBid`, `Vacuum`, `Physics`, `GEX` (Float Array type)
2. Add Grid2D Gas emitter with GPU Compute Sim
3. Configure material to read from arrays
4. C++ code then just pushes data, no setup needed

---

## 3. Data Transformation Pipeline

### Backend → Bridge

```python
# WebSocket receives Arrow IPC tables with fields:
# - snap: mid_price (float)
# - wall: tick_offset (int16), ask_intensity (float), bid_intensity (float)
# - vacuum: scores (float array, per tick)
# - physics: signed_scores (float array, per tick, negative=down ease)
# - gex: abs_gamma (float array), imbalance (float array)
```

### Bridge Transformations

```python
# Wall intensity: log scale for visualization
def transform_wall_intensity(raw: float) -> int:
    # Input: raw order book depth
    # Output: 0-65535 for UDP packet
    return int(min(65535, np.log1p(raw) * 8192))

# Physics: signed float, -1 to +1 range
# No transform needed, just pack as float

# Vacuum: 0-1 float score
# No transform needed
```

### UDP Packet Structure

```
Header (32 bytes):
  [0:4]   Magic "MWT1"
  [4]     Version (1)
  [5]     SurfaceId (0=snap, 1=wall, 2=vacuum, 3=physics, 4=gex)
  [6]     PayloadType
  [7]     Reserved
  [8:12]  NumEntries (uint32)
  [12:20] WindowEndTsNs (uint64)
  [20:24] SpotRefPriceInt (int32, ticks)
  [24:32] Padding

Payload (variable):
  Surface 0 (snap): 8 bytes - mid_price as double
  Surface 1 (wall): N * 8 bytes - FWallEntry structs
  Surface 2-4: N * 4 bytes - float arrays
```

### UE Receiver Processing

```cpp
// On packet receive:
// 1. Validate header magic + version
// 2. Check WindowEndTsNs - if new window:
//    a. Clear wall arrays (τ=0)
//    b. Apply decay to physics/vacuum (τ=5s)
//    c. Advance ring buffer head
// 3. Parse payload into internal arrays
// 4. Set bDataDirty = true

// On Tick:
// 1. If bDataDirty, push to renderer
// 2. Renderer draws using Debug Draw
```

---

## 4. Coordinate System Details

### Frontend (WebGL)
- Y-axis: price (1 pixel = 1 tick = $0.25)
- X-axis: time (1 pixel = 1 second)
- Origin: bottom-left of canvas
- Spot line: horizontal at current price Y

### Unreal Engine
- Z-axis: price (up = higher price)
- X-axis: time (negative = past)
- Y-axis: layer depth (negative = further back)
- Origin: actor location (spot price at Z=0)

### Mapping

```cpp
// Frontend Y → UE Z
float ZPos = (TickOffset) * TickWorldScale;
// TickOffset = row - CENTER_IDX (so spot = 0)

// Frontend X → UE X  
float XPos = -ColumnAge * TimeWorldScale;
// ColumnAge = how many seconds ago (0 = now, 60 = 1 min ago)

// Layer stacking (Y in UE)
// Physics: Y = -10 (back)
// Wall: Y = 0 (middle)
// Vacuum: Y = +5 (overlay)
// GEX: Y = +10 (front)
// Spot line: Y = +15 (frontmost)
```

### Scale Choices

```cpp
// V1 (too small, invisible):
float TickWorldScale = 0.5f;
float TimeWorldScale = 2.0f;
// Result: 801 ticks * 0.5 = 400 units total height

// V2 (visible):
float TickWorldScale = 5.0f;
float TimeWorldScale = 20.0f;
int32 VisibleTickRange = 200; // Only ±100 ticks shown
// Result: 200 ticks * 5 = 1000 units height
//         90 seconds * 20 = 1800 units width
```

---

## 5. Color Mapping

### Wall Layer
```cpp
// Input: Ask intensity (0-255), Bid intensity (0-255)
// Above spot: cyan/blue (asks)
// Below spot: red (bids)

if (C.B > C.R) {
  // Ask dominant - cyan
  DrawColor = FColor(0, C.B, C.B * 1.2f, 255);
} else {
  // Bid dominant - red
  DrawColor = FColor(C.R * 1.2f, 0, 0, 255);
}
```

### Physics Layer
```cpp
// Input: Signed score (-1 to +1)
// Positive (up ease): green
// Negative (down ease): red

if (SignedScore > 0) {
  uint8 G = (uint8)(SignedScore * 255);
  DrawColor = FColor(0, G, 0, 255);
} else {
  uint8 R = (uint8)(-SignedScore * 255);
  DrawColor = FColor(R, 0, 0, 255);
}
```

### Vacuum Layer
```cpp
// Input: Score 0-1 (higher = more vacuum)
// Output: Dark overlay

uint8 Darkness = (uint8)(Score * 200);
DrawColor = FColor(10, 10, 10, Darkness);
```

---

## 6. Ring Buffer Implementation

```cpp
// State
int32 HeadColumn = 0;  // Current write position
int32 HistorySeconds = 120;  // Total columns

// Advance on new time window
void AdvanceTime() {
  HeadColumn = (HeadColumn + 1) % HistorySeconds;
  // Clear wall data for new column (τ=0)
  for (int32 Row = 0; Row < LAYER_HEIGHT; Row++) {
    int32 Idx = Row * HistorySeconds + HeadColumn;
    WallData[Idx] = FColor(0, 0, 0, 0);
  }
}

// Read column N seconds ago
int32 GetColumn(int32 SecondsAgo) {
  return (HeadColumn - SecondsAgo + HistorySeconds) % HistorySeconds;
}

// 2D index (row-major storage)
int32 GetIndex(int32 Row, int32 Col) {
  return Row * HistorySeconds + Col;
}
```

---

## 7. Dissipation Model

```cpp
// Called each time window (1 second)
void ApplyDecay() {
  // τ = 5 seconds decay constant
  float DecayFactor = FMath::Exp(-1.0f / 5.0f);  // ≈ 0.8187
  
  for (int32 i = 0; i < PhysicsData.Num(); i++) {
    FColor& C = PhysicsData[i];
    C.R = (uint8)(C.R * DecayFactor);
    C.G = (uint8)(C.G * DecayFactor);
    C.A = (uint8)(C.A * DecayFactor);
  }
  
  // Same for VacuumData
  for (int32 i = 0; i < VacuumData.Num(); i++) {
    FColor& C = VacuumData[i];
    C.A = (uint8)(C.A * DecayFactor);
  }
  
  // Wall has τ=0, cleared on AdvanceTime()
}
```

---

## 8. Critical Bugs Encountered

### Bug 1: Properties Not Visible via Remote Control

**Symptom**: `get_properties` returns 400 for known UPROPERTY

**Root Cause**: The `.h` file in workspace was newer than the one actually compiled in UE project. UE was running an older version of the class.

**Lesson**: Always verify which source files are in the UE project's `Source/` folder. When in doubt, copy FROM UE project TO workspace to establish ground truth.

### Bug 2: FObjectFinder Crash

**Symptom**: Fatal crash on Play: "FObjectFinders can't be used outside of constructors"

**Root Cause**: `ConstructorHelpers::FObjectFinder` was called in `BeginPlay()` 

**Lesson**: `FObjectFinder` is a compile-time asset reference helper that ONLY works in constructors. For runtime loading, use `LoadObject<T>()` or `TSoftObjectPtr<T>`.

### Bug 3: Stale Binaries

**Symptom**: Code changes don't take effect, old crashes repeat

**Root Cause**: UE's hot reload doesn't always work. Old compiled binaries persist.

**Fix**: 
```bash
rm -rf Binaries Intermediate DerivedDataCache
# Then reopen Editor for full rebuild
```

### Bug 4: Visualization Too Small

**Symptom**: Data streaming correctly but nothing visible in viewport

**Root Cause**: Default scales (0.5 units/tick) too small relative to camera distance

**Fix**: Increase scales 10x and limit visible range to focus on data

---

## 9. Remote Control API Gotchas

### Object Paths are Precise
```python
# Wrong (will 404):
"/Game/Maps/MWT_Main"

# Correct:
"/Game/Maps/MWT_Main.MWT_Main:PersistentLevel.ActorName_0"
```

### Blueprint Instances Have _C Suffix
```python
# C++ class instance:
"MyActor_0"

# Blueprint class instance:
"BP_MyActor_C_0"  # Note the _C
```

### Some Functions Need WorldContextObject
```python
# This won't work:
client.call_function("/Script/Engine.Default__GameplayStatics", "GetAllActorsOfClass", {})

# The function signature requires WorldContextObject which can't be passed via Remote Control
# Use alternative approaches (search_assets, level inspection)
```

### Editor vs Runtime
Some Remote Control functionality only works in Editor mode, not during Play. Always test both scenarios.

---

## 10. Performance Considerations

### Debug Draw Limits
- Each `DrawDebugSolidBox` is a separate draw call
- 200 ticks × 90 columns × 4 layers = 72,000 boxes per frame
- Acceptable for prototype, but GPU-based approach needed for production

### Optimization Strategies
1. **Reduce visible range**: Only draw ±100 ticks instead of ±400
2. **Skip low-alpha**: Don't draw tiles with alpha < 5
3. **Batch similar colors**: Could group into instanced meshes
4. **Move to Niagara**: GPU particles handle this natively

### UDP Considerations
- 65KB max datagram size
- Wall data is sparse (only active levels), fits easily
- Dense arrays (physics/vacuum) may need chunking for 801 ticks
- Currently chunking at 128 entries per packet

---

## 11. Future Improvements

### Short Term
1. Position camera automatically via Remote Control
2. Add keyboard controls for zoom/pan
3. HUD overlay showing current price, timestamp

### Medium Term
1. Pre-configure Niagara System with User Parameters
2. Replace Debug Draw with Niagara GPU particles
3. Add mouse interaction (hover for details)

### Long Term
1. VR support (room-scale price visualization)
2. Multi-symbol support (side-by-side)
3. Audio feedback (price changes → sound)

---

## Summary

Key takeaways:
1. **Verify compiled code** matches workspace code
2. **Debug Draw** is fastest path to visible prototype
3. **Scale matters** - start large, optimize later
4. **Ring buffers** simplify temporal data
5. **Remote Control** is powerful but has limits
6. **Clean rebuilds** solve mysterious issues
