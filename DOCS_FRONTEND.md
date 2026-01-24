# Spymaster Frontend: AI Expert Guide

This document is for AI agents to instantly understand how to run, modify, and debug the Spymaster frontend.

## 1. Quick Start

### Launch Terminals
**Terminal 1 (Backend - Stream Source)**
```bash
# Must be running for frontend data
cd backend
uv run python -m src.serving.main
```

**Terminal 2 (Frontend - Visualization)**
```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

## 2. Architecture & Key Files

The frontend is a **Vite + Three.js** application using **WebSocket** for real-time Arrow IPC data.

### Critical Files
| File | Purpose | Key Learnings/Gotchas |
|------|---------|-----------------------|
| `src/main.ts` | Entry point, HMR, Loop | **MUST** implement HMR disposal (`import.meta.hot.dispose`) to prevent WebGL context loss. Handles `physics` and `gex` data ingestion. |
| `src/hud/data-loader.ts` | WebSocket & Arrow Parsing | **CRITICAL**: Uses a **Message Queue** to prevent race conditions between JSON headers and Binary blobs. Do not remove the `await` queue logic. |
| `src/hud/state.ts` | State & History | Uses `MAX_HISTORY` ring buffer. **MUST** append data (`[...old, ...new]`), not replace, to allow drawing lines via `CatmullRomCurve3`. Handles `physics` vs `gex` vs `snap` spot price sources. |
| `src/hud/renderer.ts` | Three.js Visuals | Draws Spot Line (Cyan), Heatmap (GEX/Wall), and Axes. **MUST** have `dispose()` method to clean up geometries/materials on reload. |

## 3. Data Flow (The "Stream")

1.  **Transport**: WebSocket `ws://localhost:8000/v1/hud/stream`
2.  **Protocol**: Mixed JSON/Binary.
    *   `{"type": "surface_header", "surface": "..."}` -> Sets context.
    *   `[Binary Blob]` -> Arrow IPC table.
3.  **Surfaces**:
    *   `physics`: **Primary Spot Source**. Contains `spot_price`.
    *   `gex`: Gamma Exposure. Optional for spot, critical for heatmap.
    *   `snap`: Fallback for spot.
    *   `wall`, `vacuum`, `radar`: High-density surfaces.

## 4. Debugging Playbook

### Symptom: "Connecting..." Forever
*   **Cause**: WebSocket connected but no data parsing.
*   **Fix**: Check `data-loader.ts`. Ensure `onmessage` isn't blocked. Use console logs to confirm `Parsed [surface]`.

### Symptom: Black Screen / No Chart
*   **Cause 1**: data is empty/null.
*   **Fix**: Check backend stream. `src/hud/state.ts` might have 0 ranges.
*   **Cause 2**: WebGL Crash.
*   **Fix**: Check console for "Context lost". Verify `renderer.dispose()` is called in `main.ts`.

### Symptom: No Spot Line ("Worm")
*   **Cause**: `HUDState` overwriting data instead of appending. Line needs >= 2 points.
*   **Fix**: Ensure `state.ts` uses `this.spotData = [...this.spotData, ...newData]`.

### Symptom: Missing Axis Labels
*   **Cause**: Zoom level too high/low or DOM overlay hidden.
*   **Fix**: Check `renderer.ts` `updateOverlay()`. Labels are HTML `<div>` elements in `#price-axis` / `#time-axis`.

## 5. Implementation Rules
1.  **Partial Batches**: Emit data via `onBatch` immediately after parsing a surface. Do not wait for "full batch".
2.  **Relaxed Types**: Use `any` for incoming row types if schemas shift (e.g. `gex` vs `physics` fields).
3.  **Strict Context**: Always check `!this.renderer` or context loss before render.
