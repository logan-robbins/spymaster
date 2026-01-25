# Frontend - AI Agent Reference

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

## Architecture (World Space First)
The renderer uses a **World Space** coordinate system where:
- **X-axis (Time)**: 0 = "Now" (Latest Snapshot). Scale: `0.1 units/sec`.
- **Y-axis (Price)**: 0 = "Spot Price at t=Now". Scale: `1.0 units/$`.

The View uses an Orthographic Camera that pans/zooms over this world space.
- **Layers** (Physics, Wall, GEX) are static meshes positioned at `Z` depths.
- **Grid/Axis** are drawn based on **Absolute Price Levels** (e.g., `.00`, `.25`) projected into this relative world space.

```
WebSocket (Arrow IPC) → data-loader.ts → state.ts → renderer.ts (WebGL)
```

## Files & Responsibilities

| File | Primary Responsibility |
|------|------------------------|
| `src/main.ts` | WebSocket connection, batching, HMR disposal logic. |
| `src/hud/data-loader.ts` | Arrow IPC decoding, schema parsing. |
| `src/hud/state.ts` | Data buffer management. **Robustness**: Filters invalid spot prices (<=100). |
| `src/hud/renderer.ts` | WebGL scene, camera logic, **Absolute Axis Alignment**, Smoothing. |
| `src/hud/grid-layer.ts` | Ring-buffer textures. **Linear Filtering** enabled for smoothness. |

## Streams Received

| Stream | Rows/Window | Key Fields | Renderer Method |
|--------|-------------|------------|-----------------|
| `snap` | 1 | `mid_price` | `state.updateSpot` |
| `physics`| 1 | `above_score`, `below_score` | `renderer.updatePhysics` (Applies Convolution) |
| `wall` | ~40-80 | `rel_ticks`, `side`, `qty` | `renderer.updateWall` |
| `vacuum` | ~40-80 | `vacuum_score`, `erosion` | `renderer.updateVacuum` |
| `gex` | ~25 | `strike`, `gex_abs` | `renderer.updateGex` |

## Rendering Logic

### 1. Robustness (Zero-Snapping Fix)
- `state.ts` rejects any spot price <= 100.
- `renderer.ts` maintains `lastValidSpot`. If the data stream sends an empty/invalid spot, the renderer holds the last known goodY position to prevent the view from snapping to 0.

### 2. Alignment (Absolute Axis)
- **Grid Lines**: Drawn at `Math.floor(price / step) * step`.
- **Labels**: Iterated in **Absolute Price Space** (e.g. `6950.00`, `6950.25`).
- **Spot Line**: Floats freely between grid lines (e.g. `6950.38`).

### 3. Layer Z-Order
```
-0.02  physicsLayer   (Gradient + Convolution)
 0.00  wallLayer      (Liquidity)
 0.01  gexLayer       (Gamma Exposure)
 0.015 vacuumLayer    (Erosion Overlay)
 0.02  gridGroup      (Absolute Grid Lines)
 2.00  priceLineGroup (Spot Line & Marker)
```

## Configuration Constants
- `DEFAULT_PRICE_RANGE`: **20** (Total vertical units visible).
- `SEC_PER_UNIT_X`: **0.1** (10 units = 100 seconds).
- `HISTORY_SECONDS`: **1800** (30 minute buffer).

## Debugging

| Symptom | Check |
|---------|-------|
| "Connecting..." forever | Check backend terminal. Check `data-loader.ts` WebSocket URL. |
| View snaps to $0 | Check `lastValidSpot` logic in `renderer.ts`. Check stream for 0-price packets. |
| Grid misalignment | Verify `updateOverlay` loops over `price += step` (Absolute), not `y += step`. |
| Pixelated Physics | Verify `LinearFilter` is enabled in `grid-layer.ts`. |
