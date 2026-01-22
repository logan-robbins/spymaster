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

// Load button handler
document.getElementById('btn-load')?.addEventListener('click', async () => {
  statusEl.textContent = 'Loading...';
  statusEl.className = 'status';

  try {
    // Load GEX surface data
    const gexData = await loader.loadSurface(SYMBOL, DT, 'gex');
    console.log('GEX data loaded:', gexData);

    if (gexData && gexData.length > 0) {
      state.setGexData(gexData);

      // Update metrics
      const latestWindow = gexData[gexData.length - 1];
      const currentSpot = latestWindow.underlying_spot_ref;
      spotEl.textContent = currentSpot?.toFixed(2) ?? '--';
      windowsEl.textContent = String(new Set(gexData.map((r: { window_end_ts_ns: bigint }) => r.window_end_ts_ns)).size);

      // Update price axis
      const spots = gexData.map((r: { underlying_spot_ref: number }) => Number(r.underlying_spot_ref));
      const minSpot = Math.min(...spots);
      const maxSpot = Math.max(...spots);

      document.getElementById('price-high')!.textContent = maxSpot.toFixed(2);
      document.getElementById('price-current')!.textContent = currentSpot.toFixed(2);
      document.getElementById('price-low')!.textContent = minSpot.toFixed(2);

      statusEl.textContent = 'Connected';
      statusEl.className = 'status connected';
    }

    // Request render
    renderer.render();

  } catch (err) {
    console.error('Failed to load data:', err);
    statusEl.textContent = 'Error';
    statusEl.className = 'status';
  }
});

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
