/**
 * SpyMaster HUD - WebGL Visualization
 * 
 * Renders physics surfaces (Wall, Vacuum, GEX) as heatmaps
 * Uses Three.js for WebGL rendering
 * Consumes Arrow IPC data from the backend API
 */

import { HUDRenderer } from './hud/renderer';
import { DataLoader } from './hud/data-loader';
import { HUDState } from './hud/state';

// Configuration
const API_BASE = 'http://localhost:8000';
const SYMBOL = 'ESH6';
const DT = '2026-01-06';

// Initialize state
const state = new HUDState();

// Initialize renderer
const canvas = document.getElementById('hud-canvas') as HTMLCanvasElement;
const renderer = new HUDRenderer(canvas, state);

// Initialize data loader
const loader = new DataLoader(API_BASE);

// UI Elements
const statusEl = document.getElementById('connection-status')!;
const spotEl = document.getElementById('metric-spot')!;
const windowsEl = document.getElementById('metric-windows')!;

// Stream Connection
const connect = () => {
  statusEl.textContent = 'Connecting...';
  statusEl.className = 'status';

  loader.connectStream(SYMBOL, DT, (batch) => {
    // Process batch
    // batch is Record<surface, rows[]>

    // Update GEX state if available
    if (batch.gex && batch.gex.length > 0) {
      state.setGexData(batch.gex);

      const latest = batch.gex[batch.gex.length - 1];
      const spot = latest.underlying_spot_ref || latest.spot_ref_price;
      // Note: radar uses spot_ref_price, gex uses underlying_spot_ref.
      // We need to handle schema diffs.

      if (spot) {
        spotEl.textContent = Number(spot).toFixed(2);
      }
    }

    // We should also update Radar/Vacuum state in state.ts if implemented.
    // For now, we just triggering render.

    renderer.render();

    statusEl.textContent = 'Streaming';
    statusEl.className = 'status connected';
  });
};

document.getElementById('btn-load')?.addEventListener('click', connect);

// Auto-connect for dev convenience?
// connect();
// Center button handler
document.getElementById('btn-center')?.addEventListener('click', () => {
  renderer.centerView();
});

// Zoom button handlers
let zoomLevel = 1.0;
document.getElementById('btn-zoom-in')?.addEventListener('click', () => {
  zoomLevel = Math.min(10, zoomLevel * 1.5);
  renderer.setZoom(zoomLevel);
});
document.getElementById('btn-zoom-out')?.addEventListener('click', () => {
  zoomLevel = Math.max(0.1, zoomLevel / 1.5);
  renderer.setZoom(zoomLevel);
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  renderer.render();
}

animate();

console.log('SpyMaster HUD initialized');
