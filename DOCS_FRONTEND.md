# Frontend - AI Agent Reference (Tick-Native Architecture)

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
| **Texture Space** | Texels | Y-axis: 0..801 (Layer Height). X-axis: 0..1800 (History). |
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

## Streams & Fields

| Stream | Update Method | Key Fields (Tick-Native) |
|--------|---------------|--------------------------|
| `snap` | `state.setSpotData` | `spot_ref_price_int` (Master Anchor), `mid_price` (Cosmetic) |
| `gex` | `renderer.updateGex` | `spot_ref_price_int`, `rel_ticks` (Integer, +/- 20, 40...) |
| `wall` | `renderer.updateWall` | `spot_ref_price_int`, `rel_ticks` (Integer) |
| `vacuum` | `renderer.updateVacuum` | `rel_ticks` (Integer) |
| `physics`| `renderer.updatePhysics`| `spot_ref_price_int` (for fallback anchor) |

## Render Layers

| Layer | Z-Depth | Type | Filter | Note |
|-------|---------|------|--------|------|
| **Physics** | -0.02 | `GridLayer` | Linear | Background gradient |
| **Wall** | 0.00 | `GridLayer` | Linear | Liquidity heatmap |
| **GEX** | 0.01 | `GridLayer` | Nearest | **Must align to 20-tick grid exactly** |
| **Vacuum** | 0.015 | `GridLayer` | Linear | Dark erosion overlay |
| **Grid Lines** | 0.02 | `THREE.Line` | N/A | Drawn at every 1.00 (4 ticks) relative to spot |

## Configuration
- `TICK_SIZE`: `0.25`
- `TICK_INT`: `250,000,000` (ns)
- `LAYER_HEIGHT_TICKS`: `801` (±400 ticks around spot)
- `PRICE_SCALE`: `1e-9`

## Debugging Checklist
1. **Vertical Jitter**: Check if `uSpotRef` is float-price instead of tick-index.
2. **Horizontal Smear**: Check if Shader uses `floor(x * width)`.
3. **GEX Misalignment**: Verify backend is sending `rel_ticks % 20 == 0`.
4. **Zero-Price Snap**: Ensure `state.ts` filters `spot_ref_price_int > 0`.
