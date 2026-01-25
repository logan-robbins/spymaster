/**
 * SpyMaster HUD - WebGL Visualization
 * 
 * Renders physics surfaces (Wall, Vacuum, GEX) as heatmaps
 * Uses Three.js for WebGL rendering
 * Consumes Arrow IPC data from the backend API
 */

import { HUDRenderer } from './hud/renderer';
import { DataLoader, type GexRow, type SnapshotRow } from './hud/data-loader';
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

let pendingTickTs: bigint | null = null;
let advancedTickTs: bigint | null = null;

const maybeAdvanceForTick = () => {
  if (pendingTickTs === null || advancedTickTs === pendingTickTs) return false;
  const spotsByTime = state.getSpotsByTime();
  if (!spotsByTime.has(pendingTickTs)) return false;
  renderer.advanceLayers();
  advancedTickTs = pendingTickTs;
  return true;
};

const isReadyForTick = () => pendingTickTs !== null && advancedTickTs === pendingTickTs;

// Stream Connection
const connect = () => {
  statusEl.textContent = 'Connecting...';
  statusEl.className = 'status';

  loader.connectStream(
    SYMBOL,
    DT,
    // onTick
    (ts) => {
      pendingTickTs = ts;
    },
    // onBatch
    (batch) => {
      // console.log(`[Main] Batch received. Keys: ${Object.keys(batch).join(', ')}`);

      // Update Snapshot/Spot state if available
      if (batch.snap && batch.snap.length > 0) {
        state.setSpotData(batch.snap as SnapshotRow[]);
        const latest = batch.snap[batch.snap.length - 1];
        if (latest && latest.mid_price) {
          spotEl.textContent = Number(latest.mid_price).toFixed(2);
        }
      }

      // Update Physics state and visualization if available
      if (batch.physics && batch.physics.length > 0) {
        state.setPhysicsData(batch.physics);
      }

      // Update GEX state if available
      if (batch.gex && batch.gex.length > 0) {
        state.setGexData(batch.gex as GexRow[]);
      }

      maybeAdvanceForTick();

      if (isReadyForTick()) {
        if (batch.physics && batch.physics.length > 0) {
          (renderer as any).updatePhysics(batch.physics);
        }

        // Update Wall Surface
        if (batch.wall && batch.wall.length > 0) {
          (renderer as any).updateWall(batch.wall);
        }

        // Update Vacuum Surface
        if (batch.vacuum && batch.vacuum.length > 0) {
          (renderer as any).updateVacuum(batch.vacuum);
        }

        // Update GEX Surface
        if (batch.gex && batch.gex.length > 0) {
          (renderer as any).updateGex(batch.gex);
        }
      }

      // Fallback if snap missing but GEX has spot ref
      const gexData = state.getGexData();
      if ((!batch.snap || batch.snap.length === 0) && gexData.length > 0) {
        const latest = gexData[gexData.length - 1];
        const spot = latest.underlying_spot_ref || latest.spot_ref_price;
        if (spot) {
          spotEl.textContent = Number(spot).toFixed(2);
        }
      }

      renderer.render();

      statusEl.textContent = 'Streaming';
      statusEl.className = 'status connected';
    });
};


document.getElementById('btn-load')?.addEventListener('click', connect);

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
let animationId: number;
function animate() {
  animationId = requestAnimationFrame(animate);
  renderer.render();
}

animate();

console.log('SpyMaster HUD initialized');

// HMR Handling
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    console.log('[Main] Disposing HMR...');
    cancelAnimationFrame(animationId);
    renderer.dispose();
  });
}
