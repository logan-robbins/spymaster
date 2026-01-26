# Frontend - AI Agent Reference (Tick-Native Architecture)

## Core Philosophy
**1. The Spot Line is the Foundation.**
Everything starts with the "Spot Line" (the current price). It must always work perfectly and serves as the immutable anchor (Y=0) for the entire visualization. If the spot line is wrong, everything else is wrong.

**2. Feature Layers are Stacked Overlays.**
All other datasets—GEX, Wall, Vacuum, Physics—are "Features" that are stacked on top of the base layer.
- They are **strictly relative** to the spot line.
- They scale specifically to the chart's coordinate system (Tick Space).
- They are composable: you can turn them on/off, but they always snap to the same underlying grid defined by the spot anchor.

## Launch Instructions
**Backend**:
```bash
uv run python -m src.serving.main
```
**Frontend**:
```bash
cd frontend
npm run dev
# Dashboard available at: http://localhost:5173
```

## Architecture (Tick-Native)
The renderer uses a **Tick Space** coordinate system to eliminate floating-point drift and alignment artifacts.
- **X-axis (Time)**: Discrete Columns. `1 pixel = 1 second`. No continuous UV scrolling / interpolation.
- **Y-axis (Price)**: `1 unit = 4 ticks` ($1.00). Internally, all logic uses **Integer Ticks**.
- **Data Anchor**: `spot_ref_price_int` (Integer, tick-aligned to $5.00 grid).

### Coordinate Systems
| Space | Unit | Description |
|-------|------|-------------|
| **Data Space** | Integer Ticks | `spot_ref_price_int` (e.g., 23800 ticks). `rel_ticks` offsets (e.g., +20). |
| **Texture Space** | Texels | Y-axis: 0..801 (Layer Height). For Bucket Radar: Height/Resolution (e.g. 401). X-axis: 0..1800 (History). |
| **World Space** | GL Units | `1 tick = 0.25 units`. `1 second = 0.1 units` (configurable). |
| **Screen Space** | Pixels | Camera projects World Space → Screen. |

## Critical Logic Rules

### 1. Discrete Column Addressing (Time Alignment)
**NEVER** use continuous UV x-coordinates for time history. This causes bilinear filtering bleed between seconds.
**ALWAYS** use the discrete addressing pattern in shaders:
```glsl
float colIndex = floor((vUv.x + uHeadOffset) * uWidth);
float x = (colIndex + 0.5) / uWidth;
```
- `uWidth`: Texture width (History Seconds).
- Encodes strict "1 second = 1 column" logic.

### 2. Tick-Space Rectification (Price Alignment)
The mesh is static. The shader rectifies textures based on the difference between "Current Spot" and "Historical Spot".
**Formula**:
```glsl
// Both must be in Tick Index Space (Float representing Integer)
float currentTickIndex = uSpotRef + (vUv.y - 0.5) * uHeight;
float historicalTickIndex = texture2D(uSpotHistory, vec2(x, 0.5)).r; // Stored as Tick Index

float relTicks = currentTickIndex - historicalTickIndex;
float textureRow = (uHeight * 0.5) + relTicks;
```
- `uSpotRef`: Current `spot_ref_price_int / TICK_INT`.
- `uSpotHistory`: Ring buffer storing `spot_ref_price_int / TICK_INT` for each historical second.

### 3. Data Integrity
- **Spot Anchor**: `spot_ref_price_int` MUST be used for geometry anchoring. `mid_price` is cosmetic only.
- **GEX Alignment**: `rel_ticks` provided by backend are authoritative. They are guaranteed to be multiples of 20 (for ES $5 strikes).
- **Rounding**: `Math.round` is BANNED for grid math. Use `floor` or exact integer arithmetic.
- **Control Frames**: WebSocket `batch_start`/`surface_header` JSON uses `window_end_ts_ns` as a string; parse to `BigInt` before comparisons.

## Streams & Fields

| Stream | Update Method | Key Fields (Tick-Native) |
|--------|---------------|--------------------------|
| `snap` | `state.setSpotData` | `window_end_ts_ns`, `mid_price`, `spot_ref_price_int`, `book_valid` |
| `wall` | `renderer.updateWall` | `window_end_ts_ns`, `rel_ticks`, `side`, `depth_qty_rest` |
| `vacuum` | `renderer.updateVacuum` | `window_end_ts_ns`, `rel_ticks`, `vacuum_score` |
| `physics`| `renderer.updatePhysics`| `window_end_ts_ns`, `rel_ticks`, `physics_score`, `physics_score_signed` |
| `gex` | `renderer.updateGex` | `window_end_ts_ns`, `strike_points`, `spot_ref_price_int`, `rel_ticks` (multiples of 20), `underlying_spot_ref`, `gex_abs`, `gex`, `gex_imbalance_ratio` |
| `bucket_radar`| `renderer.updateBucketRadar`| `window_end_ts_ns`, `bucket_rel`, `blocked_level`, `cavitation`, `gex_stiffness` |

HUD stream columns are exactly those listed above. Optional wall/vacuum fields (`d1_depth_qty`, `d2_depth_qty`, `d2_pull_add_log`, `wall_erosion`) are not present in the stream and are treated as 0 by the renderer.

## Render Layers

| Layer | Z-Depth | Type | Filter | Note |
|-------|---------|------|--------|------|
| **Physics** | -0.02 | `GridLayer` | Linear | Per-tick directional ease (Cyan/Blue) |
| **Wall** | 0.00 | `GridLayer` | Linear | Liquidity heatmap |
| **Bucket Radar** | 0.005 | `GridLayer` | Nearest | **2-Tick Bucket Native** |
| **GEX** | 0.01 | `GridLayer` | Nearest | **Magenta (Calls) / Green (Puts)** |
| **Vacuum** | 0.015 | `GridLayer` | Linear | Dark erosion overlay |
| **Grid Lines** | 0.02 | `THREE.Line` | N/A | Drawn at every 1.00 (4 ticks) relative to spot |

## Configuration
- `TICK_SIZE`: `0.25`
- `TICK_INT`: `250,000,000` (ns)
- `LAYER_HEIGHT_TICKS`: `801` (±400 ticks around spot)
- `PRICE_SCALE`: `1e-9`

## Dissipation Model
To visualize pressure "hanging" in the air, layers use a temporal decay model:
- **Physics**: `τ=5s`. Decays ~18% per second.
- **Bucket Radar**: `τ=0`. (Stateful evolution handled by backend or instant).
- **Vacuum**: `τ=5s`.
- **Wall**: `τ=0` (Instant clear).
- **Logic**: `new_cell = old_cell * exp(-Δt/τ)`. Writes are applied on top.

## Visual Diagnostics
A debug overlay (top-left) displays real-time alignment data:
- `SpotRefTick`: Canonical integer spot anchor.
- `Head`: Current ring buffer column.
- `Latest TS`: Timestamp of last rendered window.
- **Grid Lines**: Drawn at every 4 ticks ($1) and 20 ticks ($5) to verify bucket alignment.

## Debugging Checklist
1. **Vertical Jitter**: Check if `uSpotRef` is float-price instead of tick-index.
2. **Horizontal Smear**: Check if Shader uses `floor(x * width)`.
3. **GEX Misalignment**: Verify backend is sending `rel_ticks % 20 == 0`.
4. **Zero-Price Snap**: Ensure `state.ts` filters `spot_ref_price_int > 0`.
5. **Layers Not Rendering**: Check that `rel_ticks` is converted to `Number()` (Arrow returns BigInt for int32 fields).
6. **Data Race**: Ensure all surfaces are received before advancing ring buffer (see `allSurfacesReceived()` in main.ts).

## Known Fixes Applied
- **Shader Variable Redefinition**: Removed duplicate `float x` declaration in `grid-layer.ts` rectifyLogic.
- **Arrow BigInt Conversion**: All `rel_ticks` fields must use `Number(row.rel_ticks)` since Apache Arrow JS returns int32 as BigInt.
- **Surface Synchronization**: `main.ts` now waits for all surfaces (snap, wall, vacuum, physics, gex) before advancing the ring buffer and writing data.
