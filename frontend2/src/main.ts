import * as THREE from 'three';
import { connectStream, type SnapRow, type VelocityRow } from './ws-client';
import { VelocityGrid, spotToTickIndex } from './velocity-grid';
import { SpotLine } from './spot-line';
import { PriceAxis } from './price-axis';

// Configuration
const SYMBOL = 'ESH6';
const DATE = '2026-01-06';
const SURFACES = 'snap,velocity';

const GRID_WIDTH = 1800;   // 30 minutes @ 1s/col
const GRID_HEIGHT = 801;   // ±400 ticks from spot
const VIEW_SECONDS = 300;  // Show 5 minutes of history
const VIEW_TICKS = 100;    // Show ±50 ticks around spot
const PREDICTION_MARGIN = 0.30; // 30% right margin for predictions

// DOM elements
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const spotLabel = document.getElementById('spot-label')!;
const tsLabel = document.getElementById('ts-label')!;
const frameLabel = document.getElementById('frame-label')!;
const zoomLabel = document.getElementById('zoom-label')!

// Three.js setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

// Orthographic camera
// X: time in seconds (negative = past, 0 = now, positive = predictions area)
// Y: price in ticks (centered on current spot)
const PREDICTION_SECONDS = VIEW_SECONDS * PREDICTION_MARGIN / (1 - PREDICTION_MARGIN); // ~75s for 20% margin
const camera = new THREE.OrthographicCamera(
  -VIEW_SECONDS, PREDICTION_SECONDS,  // X: -300s to +75s (predictions on right)
  VIEW_TICKS / 2, -VIEW_TICKS / 2,    // Y: ±50 ticks from center
  0.1, 100
);
camera.position.z = 10;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setSize(canvas.clientWidth, canvas.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);

// Velocity grid overlay (behind spot line)
const velocityGrid = new VelocityGrid(GRID_WIDTH, GRID_HEIGHT);
const gridMesh = velocityGrid.getMesh();
gridMesh.renderOrder = 0;
scene.add(gridMesh);

// Spot price line (turquoise, in front)
const spotLine = new SpotLine();
spotLine.getLine().renderOrder = 1;
scene.add(spotLine.getLine());

// Price axis labels
const priceAxis = new PriceAxis('price-axis', 20, VIEW_TICKS);

// Zoom state
let zoomLevel = 1.0;
const MIN_ZOOM = 0.25;  // Zoom out 4x
const MAX_ZOOM = 4.0;   // Zoom in 4x
const ZOOM_SPEED = 0.1;

// State
let currentSpotTickIndex = 0;
let lastTs: bigint = 0n;
let frameCount = 0;
let pendingVelocityRows: VelocityRow[] = [];
let batchReady = false;

// WebSocket callbacks
function onTick(ts: bigint, _surfaces: string[]): void {
  lastTs = ts;
  batchReady = false;
  pendingVelocityRows = [];
}

function onSnap(row: SnapRow): void {
  if (row.spot_ref_price_int <= 0n) return;

  currentSpotTickIndex = spotToTickIndex(row.spot_ref_price_int);

  // Add to spot line history
  spotLine.addPrice(currentSpotTickIndex);

  // Update spot label
  const midPrice = row.mid_price;
  spotLabel.textContent = `Spot: $${midPrice.toFixed(2)} (tick: ${currentSpotTickIndex.toFixed(0)})`;
}

function onVelocity(rows: VelocityRow[]): void {
  pendingVelocityRows = rows;
  batchReady = true;
  processBatch();
}

function processBatch(): void {
  if (!batchReady || currentSpotTickIndex === 0) return;

  // Advance velocity grid ring buffer
  velocityGrid.advance(currentSpotTickIndex);

  // Write velocity data
  let writeCount = 0;
  for (const row of pendingVelocityRows) {
    const relTicks = row.rel_ticks;
    const velocity = row.liquidity_velocity;
    if (velocity !== 0) {
      velocityGrid.write(relTicks, velocity);
      writeCount++;
    }
  }
  velocityGrid.flush();

  // Update spot line geometry
  // X: newest point at x=0, older points at negative x (1 unit = 1 second)
  spotLine.updateGeometry(1, 0);

  // Position velocity grid mesh to align with spot line:
  // The mesh is a unit square that gets scaled/positioned
  // Shader maps vUv.x (0..1) to columns in ring buffer
  // Mesh right edge should be at x=0 (newest data)
  const count = velocityGrid.getCount();
  if (count > 1) {
    const timeSpan = count - 1; // Seconds from oldest to newest
    gridMesh.scale.set(timeSpan, GRID_HEIGHT, 1);
    gridMesh.position.x = -timeSpan / 2; // Center mesh (left edge at -timeSpan, right at 0)
    gridMesh.position.y = currentSpotTickIndex; // Center on current spot
    gridMesh.position.z = 0; // Same plane as spot line
  }

  // Camera follows spot price vertically (Y tracks spot)
  camera.position.y = currentSpotTickIndex;

  // Update price axis labels
  priceAxis.update(currentSpotTickIndex, camera);

  // Update labels
  frameCount++;
  const tsMs = Number(lastTs / 1_000_000n);
  const date = new Date(tsMs);
  tsLabel.textContent = `TS: ${date.toISOString().slice(11, 19)}`;
  frameLabel.textContent = `Frame: ${frameCount}`;

  if (frameCount <= 5) {
    console.log(`Frame ${frameCount}: wrote ${writeCount} velocity values, spot=${currentSpotTickIndex.toFixed(0)}, count=${count}`);
  }

  batchReady = false;
  pendingVelocityRows = [];
}

// Connect WebSocket
connectStream(SYMBOL, DATE, SURFACES, {
  onTick,
  onSnap,
  onVelocity
});

// Update camera based on zoom level
function updateCamera(): void {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const aspect = width / height;

  // Base view dimensions adjusted by zoom
  const viewTicks = VIEW_TICKS / zoomLevel;
  // viewSeconds could be used for X-axis zoom (future enhancement)

  // Calculate camera bounds maintaining prediction margin
  const totalViewWidth = viewTicks * aspect;
  const dataWidth = totalViewWidth * (1 - PREDICTION_MARGIN);
  const predictionWidth = totalViewWidth * PREDICTION_MARGIN;

  camera.left = -dataWidth;
  camera.right = predictionWidth;
  camera.top = viewTicks / 2;
  camera.bottom = -viewTicks / 2;
  camera.updateProjectionMatrix();

  // Update price axis with new visible range
  priceAxis.update(currentSpotTickIndex, camera);
}

// Handle resize
function onResize(): void {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setSize(width, height);
  updateCamera();
}
window.addEventListener('resize', onResize);

// Handle zoom with mouse wheel
function onWheel(event: WheelEvent): void {
  event.preventDefault();

  // Zoom in on scroll up, out on scroll down
  const delta = event.deltaY > 0 ? -ZOOM_SPEED : ZOOM_SPEED;
  zoomLevel = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoomLevel * (1 + delta)));

  updateCamera();
  zoomLabel.textContent = `Zoom: ${zoomLevel.toFixed(1)}x`;
}
canvas.addEventListener('wheel', onWheel, { passive: false });

// Animation loop
function animate(): void {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

console.log('Frontend2 initialized');
console.log(`Connecting to ws://localhost:8001/v1/velocity/stream?symbol=${SYMBOL}&dt=${DATE}`);
