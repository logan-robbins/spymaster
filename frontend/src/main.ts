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
      spotEl.textContent = latestWindow.underlying_spot_ref?.toFixed(2) ?? '--';
      windowsEl.textContent = String(new Set(gexData.map(r => r.window_end_ts_ns)).size);

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

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  renderer.render();
}

animate();

console.log('SpyMaster HUD initialized');
