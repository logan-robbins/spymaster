# Frontend - AI Agent Reference

## Launch
```bash
cd frontend
npm run dev
# Requires backend running: uv run python -m src.serving.main
```

## Architecture

```
WebSocket (Arrow IPC) → data-loader.ts → state.ts → renderer.ts (WebGL)
```

## Files

| File | Modify When |
|------|-------------|
| `src/main.ts` | Connection logic, batch handling, HMR disposal |
| `src/hud/data-loader.ts` | WebSocket parsing, Arrow IPC decoding |
| `src/hud/state.ts` | Data history, spot price extraction |
| `src/hud/renderer.ts` | WebGL layers, GEX density, spot line |
| `src/hud/grid-layer.ts` | Scrolling texture for wall/vacuum/physics/GEX |

## Streams Received

| Stream | Rows/Window | Key Fields | Renderer Method |
|--------|-------------|------------|-----------------|
| `snap` | 1 | `mid_price`, `book_valid` | state.setSpotData |
| `wall` | ~40-80 | `rel_ticks`, `side`, `depth_qty_rest` | renderer.updateWall |
| `vacuum` | ~40-80 | `rel_ticks`, `vacuum_score` | renderer.updateVacuum |
| `physics` | 1 | `mid_price`, `above_score`, `below_score` | renderer.updatePhysics |
| `gex` | 25 | `strike_points`, `gex_abs`, `gex_imbalance_ratio` | renderer.updateGex |
| `radar` | 1 | ~200 ML features | **IGNORED** (future ML inference) |

## Grid Layer

```
GridLayer(width, height)
- width = seconds of history (1800 = 30 min)
- height = ticks from center (800 = ±400 ticks = ±$100)
- write(relTicks, [r,g,b,a]) → paint cell at current time column
- advance() → scroll time forward 1 second
```

## Renderer Methods

```typescript
updateWall(data: any[])     // Blue asks (side='A'), red bids (side='B')
updateVacuum(data: any[])   // Black erosion overlay (vacuum + wall_erosion)
updatePhysics(data: any[])  // Green above spot (above_score), red below (below_score)
updateGex(data: any[])      // Tick-aligned density bands (overlap thickens)
createPriceLine(spots, numWindows)  // Cyan spot line + marker
```

## Layer Z-Order

```
-0.02  physicsLayer   (directional gradient)
 0.00  wallLayer      (liquidity)
 0.01  gexLayer       (tick-aligned density)
 0.015 vacuumLayer    (erosion overlay)
 0.02  gridGroup      (price grid lines)
 1.00  priceLineGroup (spot line)
```

## Debugging

| Symptom | Check |
|---------|-------|
| "Connecting..." forever | data-loader.ts onmessage, backend running |
| Black screen | state.ts ranges, WebGL context lost |
| No spot line | state.ts append vs replace, need ≥2 points |
| No physics gradient | main.ts calling renderer.updatePhysics |

## HMR Requirement

```typescript
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    cancelAnimationFrame(animationId);
    renderer.dispose();
  });
}
```
