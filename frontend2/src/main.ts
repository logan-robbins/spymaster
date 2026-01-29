import * as THREE from 'three';
import { connectStream, type SnapRow, type VelocityRow, type OptionsRow, type ForecastRow } from './ws-client';
import { VelocityGrid, spotToTickIndex } from './velocity-grid';
import { OptionsGrid } from './options-grid';
import { SpotLine } from './spot-line';
import { PriceAxis } from './price-axis';
import { ForecastOverlay } from './forecast-overlay';

// Configuration
const SYMBOL = 'ESH6';
const DATE = '2026-01-06';

const GRID_WIDTH = 1800;   // 30 minutes @ 1s/col
const GRID_HEIGHT = 801;   // ±400 ticks from spot
const VIEW_SECONDS = 300;  // Show 5 minutes of history
const VIEW_TICKS = 100;    // Show ±50 ticks around center
const PREDICTION_MARGIN = 0.30; // 30% right margin for predictions

// DOM elements
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const spotLabel = document.getElementById('spot-label')!;
const tsLabel = document.getElementById('ts-label')!;
const frameLabel = document.getElementById('frame-label')!;
const zoomLabel = document.getElementById('zoom-label')!;
const fieldSelect = document.getElementById('field-select') as HTMLSelectElement;

// Three.js setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f); // Dark blue/black background

// Camera state - FIXED position, only moves with zoom/pan
// Y position set once we receive first spot price
let cameraYCenter = 0;  // Absolute tick index for camera center
let cameraInitialized = false;

const PREDICTION_SECONDS = VIEW_SECONDS * PREDICTION_MARGIN / (1 - PREDICTION_MARGIN);
const camera = new THREE.OrthographicCamera(
  -VIEW_SECONDS, PREDICTION_SECONDS,
  VIEW_TICKS / 2, -VIEW_TICKS / 2,
  0.1, 100
);
camera.position.z = 10;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setSize(canvas.clientWidth, canvas.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);

// Futures velocity grid (green/red, fine-grained at $0.25)
const velocityGrid = new VelocityGrid(GRID_WIDTH, GRID_HEIGHT);
const velocityMesh = velocityGrid.getMesh();
velocityMesh.renderOrder = 0;
scene.add(velocityMesh);

// Options velocity grid (cyan/magenta, horizontal bars at FIXED $5 increments)
const optionsGrid = new OptionsGrid(GRID_WIDTH);
const optionsMesh = optionsGrid.getMesh();
optionsMesh.renderOrder = 0.5;
scene.add(optionsMesh);

// Forecast Overlay
const forecastOverlay = new ForecastOverlay(scene);

// Spot price line (turquoise with glow, in front)
const spotLine = new SpotLine();
const spotLineGroup = spotLine.getLine();
spotLineGroup.renderOrder = 1;
scene.add(spotLineGroup);

// Price axis labels and grid lines (dynamic based on zoom)
const priceAxis = new PriceAxis('price-axis', 20, VIEW_TICKS);
scene.add(priceAxis.getGridGroup());

// Zoom state
let zoomLevel = 1.0;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4.0;
const ZOOM_SPEED = 0.1;

// State
let currentSpotTickIndex = 0;
let lastTs: bigint = 0n;
let frameCount = 0;
let pendingVelocityRows: VelocityRow[] = [];
let pendingOptionsRows: OptionsRow[] = [];
let velocityReady = false;
let optionsReady = false;

// WebSocket callbacks
function onTick(ts: bigint, _surfaces: string[]): void {
  lastTs = ts;
  velocityReady = false;
  optionsReady = false;
  pendingVelocityRows = [];
  pendingOptionsRows = [];
}

function onSnap(row: SnapRow): void {
  if (row.spot_ref_price_int <= 0n) return;

  currentSpotTickIndex = spotToTickIndex(row.spot_ref_price_int);

  // Initialize camera center on first spot (and anchor options grid)
  if (!cameraInitialized) {
    cameraYCenter = currentSpotTickIndex;
    camera.position.y = cameraYCenter;
    optionsGrid.setReferenceStrike(currentSpotTickIndex);
    cameraInitialized = true;
    console.log(`Camera initialized at tick ${cameraYCenter.toFixed(0)}`);
  }

  // Add to spot line history
  spotLine.addPrice(currentSpotTickIndex);

  // Update spot label
  const midPrice = row.mid_price;
  spotLabel.textContent = `Spot: $${midPrice.toFixed(2)} (tick: ${currentSpotTickIndex.toFixed(0)})`;
}

function onVelocity(rows: VelocityRow[]): void {
  pendingVelocityRows = rows;
  velocityReady = true;
  tryProcessBatch();
}

function onOptions(rows: OptionsRow[]): void {
  pendingOptionsRows = rows;
  optionsReady = true;
  tryProcessBatch();
}

function onForecast(rows: ForecastRow[]): void {
  // Forecast usually arrives last or is separate.
  // We can update overlay immediately if we have currentSpot from snap
  if (cameraInitialized) {
    forecastOverlay.update(rows, currentSpotTickIndex);
  }
}

function tryProcessBatch(): void {
  // Wait for both surfaces before processing
  if (!velocityReady || !optionsReady || currentSpotTickIndex === 0) return;

  // Advance both grids
  velocityGrid.advance(currentSpotTickIndex);
  optionsGrid.advance();

  // Write futures velocity data (including new physics fields)
  let velocityWriteCount = 0;
  for (const row of pendingVelocityRows) {
    const relTicks = row.rel_ticks;
    const velocity = row.liquidity_velocity;

    // Always write even if velocity is 0, because other fields might be non-zero?
    // Wait, optimization: if ALL fields are zero, skip?
    // Fields default to 0.
    // If we only iterate sparse rows, we write sparse.
    // But backend sends dense grid (-200..200).
    // So we iterate 401 rows.

    if (velocity !== 0 || row.u_wave_energy !== 0 || row.pressure_grad !== 0 || row.nu !== 0 || row.Omega !== 0) {
      velocityGrid.write(
        relTicks,
        velocity,
        row.u_wave_energy,
        row.pressure_grad,
        row.nu,
        row.Omega
      );
      velocityWriteCount++;
    }
  }
  velocityGrid.flush();

  // Write options velocity data (converted to absolute positions)
  let optionsWriteCount = 0;
  for (const row of pendingOptionsRows) {
    const relTicks = row.rel_ticks;
    const velocity = row.liquidity_velocity;
    if (velocity !== 0) {
      optionsGrid.write(currentSpotTickIndex, relTicks, velocity);
      optionsWriteCount++;
    }
  }
  optionsGrid.flush();

  // Update spot line geometry
  spotLine.updateGeometry(1, 0);

  // Position grids - camera is FIXED, grids are in absolute coordinates
  const count = velocityGrid.getCount();
  if (count > 1) {
    const timeSpan = count - 1;

    // Futures velocity grid - centered on CAMERA position (which is fixed)
    velocityGrid.setMeshCenter(cameraYCenter);  // Tell shader where mesh is centered
    velocityMesh.scale.set(timeSpan, GRID_HEIGHT, 1);
    velocityMesh.position.x = -timeSpan / 2;
    velocityMesh.position.y = cameraYCenter;  // Use camera center, not current spot
    velocityMesh.position.z = 0;

    // Options grid - covers the visible range at absolute tick positions
    const viewTicks = VIEW_TICKS / zoomLevel;
    const meshBottom = cameraYCenter - viewTicks / 2 - 100;  // Extra buffer
    const meshHeight = viewTicks + 200;
    optionsGrid.updateMesh(timeSpan, meshBottom, meshHeight);
  }

  // NOTE: Camera position.y does NOT change - it stays at cameraYCenter
  // The spot line moves UP/DOWN relative to the fixed camera view

  // Update price axis labels (fixed at absolute $5 prices)
  priceAxis.update(cameraYCenter, camera);

  // Update labels
  frameCount++;
  const tsMs = Number(lastTs / 1_000_000n);
  const date = new Date(tsMs);
  tsLabel.textContent = `TS: ${date.toISOString().slice(11, 19)}`;
  frameLabel.textContent = `Frame: ${frameCount}`;

  if (frameCount <= 5) {
    console.log(
      `Frame ${frameCount}: futures=${velocityWriteCount}, options=${optionsWriteCount}, ` +
      `spot=${currentSpotTickIndex.toFixed(0)}, camera=${cameraYCenter.toFixed(0)}`
    );
  }

  velocityReady = false;
  optionsReady = false;
  pendingVelocityRows = [];
  pendingOptionsRows = [];
}

// Connect unified WebSocket stream
connectStream(SYMBOL, DATE, {
  onTick,
  onSnap,
  onVelocity,
  onOptions,
  onForecast
});

// Field Selector Logic
fieldSelect.addEventListener('change', (e) => {
  const mode = (e.target as HTMLSelectElement).value as any;
  velocityGrid.setDisplayMode(mode);
});

// Update camera based on zoom level
function updateCamera(): void {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const aspect = width / height;

  const viewTicks = VIEW_TICKS / zoomLevel;

  const totalViewWidth = viewTicks * aspect;
  const dataWidth = totalViewWidth * (1 - PREDICTION_MARGIN);
  const predictionWidth = totalViewWidth * PREDICTION_MARGIN;

  // Camera bounds are RELATIVE to camera.position.y (which is cameraYCenter)
  camera.left = -dataWidth;
  camera.right = predictionWidth;
  camera.top = viewTicks / 2;
  camera.bottom = -viewTicks / 2;
  camera.updateProjectionMatrix();

  // Update price axis with fixed center
  if (cameraInitialized) {
    priceAxis.update(cameraYCenter, camera);
  }
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

console.log('Frontend2 initialized (unified futures + options stream)');
console.log(`Connecting to ws://localhost:8001/v1/velocity/stream?symbol=${SYMBOL}&dt=${DATE}`);
