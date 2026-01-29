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

// Diagnostic HUD elements
const runScoreUpLabel = document.getElementById('run-score-up')!;
const runScoreDownLabel = document.getElementById('run-score-down')!;
const wallDistLabel = document.getElementById('wall-dist')!;
const confidenceLabel = document.getElementById('confidence-label')!;

// Three.js setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f); // Dark blue/black background

// Camera state - FIXED position, only moves with zoom/pan
let cameraYCenter = 0;
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

// Futures velocity grid - UNIFIED composite view (pressure vs obstacles)
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

// Price axis labels and grid lines
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

// Diagnostic state
let latestRunScoreUp = 0;
let latestRunScoreDown = 0;
let latestDUp = 0;
let latestDDown = 0;
let latestConfidence = 0;

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

  // Initialize camera center on first spot
  if (!cameraInitialized) {
    cameraYCenter = currentSpotTickIndex;
    camera.position.y = cameraYCenter;
    optionsGrid.setReferenceStrike(currentSpotTickIndex);
    forecastOverlay.setCameraCenter(cameraYCenter);
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
  if (!cameraInitialized) return;
  forecastOverlay.update(rows, currentSpotTickIndex);
  
  // Extract diagnostics from forecast rows
  for (const row of rows) {
    if (row.horizon_s === 0) {
      // Diagnostic row
      latestRunScoreUp = row.run_score_up ?? 0;
      latestRunScoreDown = row.run_score_down ?? 0;
      latestDUp = row.d_up ?? 0;
      latestDDown = row.d_down ?? 0;
    } else if (row.horizon_s === 30) {
      latestConfidence = row.confidence ?? 0;
    }
  }
  
  // Update HUD
  updateDiagnosticHUD();
}

function updateDiagnosticHUD(): void {
  // Run scores: positive = vacuum corridor open, negative = wall reinforcing
  const upColor = latestRunScoreUp > 0.1 ? '#ffaa00' : (latestRunScoreUp < -0.1 ? '#00aaff' : '#666');
  const downColor = latestRunScoreDown > 0.1 ? '#00aaff' : (latestRunScoreDown < -0.1 ? '#ffaa00' : '#666');
  
  runScoreUpLabel.style.color = upColor;
  runScoreUpLabel.textContent = `↑ Run: ${latestRunScoreUp.toFixed(2)}`;
  
  runScoreDownLabel.style.color = downColor;
  runScoreDownLabel.textContent = `↓ Run: ${latestRunScoreDown.toFixed(2)}`;
  
  // Wall distances
  wallDistLabel.textContent = `Walls: ↑${latestDUp.toFixed(0)} ↓${latestDDown.toFixed(0)} ticks`;
  
  // Confidence
  const confColor = latestConfidence > 0.5 ? '#00ff88' : (latestConfidence > 0.2 ? '#ffaa00' : '#666');
  confidenceLabel.style.color = confColor;
  confidenceLabel.textContent = `Conf: ${(latestConfidence * 100).toFixed(0)}%`;
}

function tryProcessBatch(): void {
  if (!velocityReady || !optionsReady || currentSpotTickIndex === 0) return;

  // Advance both grids
  velocityGrid.advance(currentSpotTickIndex);
  optionsGrid.advance();

  // Write futures velocity data
  let velocityWriteCount = 0;
  for (const row of pendingVelocityRows) {
    const relTicks = row.rel_ticks;
    const velocity = row.liquidity_velocity;

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

  // Write options velocity data
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

  // Position grids
  const count = velocityGrid.getCount();
  if (count > 1) {
    const timeSpan = count - 1;

    velocityGrid.setMeshCenter(cameraYCenter);
    velocityMesh.scale.set(timeSpan, GRID_HEIGHT, 1);
    velocityMesh.position.x = -timeSpan / 2;
    velocityMesh.position.y = cameraYCenter;
    velocityMesh.position.z = 0;

    const viewTicks = VIEW_TICKS / zoomLevel;
    const meshBottom = cameraYCenter - viewTicks / 2 - 100;
    const meshHeight = viewTicks + 200;
    optionsGrid.updateMesh(timeSpan, meshBottom, meshHeight);
  }

  // Update price axis labels
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

// Update camera based on zoom level
function updateCamera(): void {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const aspect = width / height;

  const viewTicks = VIEW_TICKS / zoomLevel;

  const totalViewWidth = viewTicks * aspect;
  const dataWidth = totalViewWidth * (1 - PREDICTION_MARGIN);
  const predictionWidth = totalViewWidth * PREDICTION_MARGIN;

  camera.left = -dataWidth;
  camera.right = predictionWidth;
  camera.top = viewTicks / 2;
  camera.bottom = -viewTicks / 2;
  camera.updateProjectionMatrix();

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

console.log('Frontend2 initialized - UNIFIED pressure vs obstacles view');
console.log(`Connecting to ws://localhost:8001/v1/velocity/stream?symbol=${SYMBOL}&dt=${DATE}`);
