/**
 * Vacuum & Pressure Detector -- frontend visualisation.
 *
 * Price-anchored heatmap: the Y-axis represents fixed dollar price levels.
 * Spot moves within the grid as a visible polyline trail.
 *
 * Connects to ws://localhost:8002/v1/vacuum-pressure/stream
 * Renders:
 *   - Left panel:  current depth profile (horizontal bars per level)
 *   - Centre panel: scrolling heatmap   (time x price, colour = flow state)
 *   - Right axis:   dollar price labels aligned to grid rows
 *   - Right panel:  signal gauges       (DOM-based, updated per tick)
 *   - Bottom bar:   composite direction + derivative readings
 *
 * Colour encoding for heatmap cells:
 *   Building (add > pull) -> cyan-green   (pressure zone)
 *   Draining (pull > add) -> red-magenta  (vacuum / active drain)
 *   No liquidity          -> near-black   (void)
 *
 * Zoom & Pan:
 *   Mouse wheel on heatmap zooms (Shift=horiz, plain=vert, Ctrl/Cmd=both).
 *   Click-drag pans.  Double-click resets to auto-follow.
 *
 * Runtime config:
 *   All instrument-specific constants (bucket size, tick size, decimals,
 *   multiplier) are received from the server via a JSON control message
 *   with type "config" before the first data batch.  No hardcoded
 *   instrument assumptions remain in this module.
 */

import { tableFromIPC } from 'apache-arrow';

// ------------------------------------------------------------------ Types

interface RuntimeConfig {
  product_type: string;
  symbol: string;
  symbol_root: string;
  price_scale: number;
  tick_size: number;
  bucket_size_dollars: number;
  rel_tick_size: number;
  grid_max_ticks: number;
  contract_multiplier: number;
  qty_unit: string;
  price_decimals: number;
  config_version: string;
}

interface SnapData {
  window_end_ts_ns: bigint;
  mid_price: number;
  spot_ref_price_int: bigint;
  best_bid_price_int: bigint;
  best_ask_price_int: bigint;
  book_valid: boolean;
}

interface FlowRow {
  rel_ticks: number;
  side: string;
  depth_qty_end: number;
  add_qty: number;
  pull_qty: number;
  fill_qty: number;
  depth_qty_rest: number;
  pull_qty_rest: number;
  net_flow: number;
  vacuum_intensity: number;
  pressure_intensity: number;
  rest_fraction: number;
}

interface SignalsData {
  window_end_ts_ns: bigint;
  vacuum_above: number;
  vacuum_below: number;
  resting_drain_ask: number;
  resting_drain_bid: number;
  flow_imbalance: number;
  fill_imbalance: number;
  depth_imbalance: number;
  rest_depth_imbalance: number;
  bid_migration_com: number;
  ask_migration_com: number;
  composite: number;
  d1_composite: number;
  d2_composite: number;
  d3_composite: number;
  confidence: number;
  strength: number;
}

/** Parsed and validated URL query parameters. */
interface StreamParams {
  product_type: string;
  symbol: string;
  dt: string;
  speed: string;
  skip: string;
}

// --------------------------------------------------------- Layout constants

const WS_PORT = 8002;
const MAX_REL_TICKS = 40;                   // +/-40 buckets from anchor
const HMAP_LEVELS = MAX_REL_TICKS * 2 + 1;  // 81 rows
const HMAP_HISTORY = 360;                    // 6 min of 1-second columns
const FLOW_NORM_SCALE = 500;                 // characteristic shares for tanh norm
const DEPTH_NORM_PERCENTILE_DECAY = 0.995;
const SCROLL_MARGIN = 10;                    // rows from edge before auto-scroll

// Legacy equity fallback (used only when server config is absent)
const LEGACY_BUCKET_DOLLARS = 0.50;
const LEGACY_PRICE_DECIMALS = 2;

// --------------------------------------------------- Runtime config state

let runtimeConfig: RuntimeConfig | null = null;
let configReceived = false;

/** Active bucket size in dollars -- resolved from server config or legacy fallback. */
function bucketDollars(): number {
  return runtimeConfig?.bucket_size_dollars ?? LEGACY_BUCKET_DOLLARS;
}

/** Active price decimal precision -- resolved from server config or legacy fallback. */
function priceDecimals(): number {
  return runtimeConfig?.price_decimals ?? LEGACY_PRICE_DECIMALS;
}

// ----------------------------------------------------------- Stream state

let snap: SnapData | null = null;
let currentFlow: FlowRow[] = [];
let currentSignals: SignalsData | null = null;
let windowCount = 0;

// Price-anchored grid
let anchorPriceDollars = 0;
let anchorInitialized = false;
let currentSpotDollars = 0;

// Spot trail: fractional row position per heatmap column (null = no data)
const spotTrail: (number | null)[] = new Array(HMAP_HISTORY).fill(null);

// Heatmap pixel buffer (RGBA, HMAP_HISTORY x HMAP_LEVELS)
const hmapData = new Uint8ClampedArray(HMAP_HISTORY * HMAP_LEVELS * 4);
// Initialise to dark background
for (let i = 0; i < hmapData.length; i += 4) {
  hmapData[i] = 10;
  hmapData[i + 1] = 10;
  hmapData[i + 2] = 15;
  hmapData[i + 3] = 255;
}

let runningMaxDepth = 100; // adaptive normalisation

// --------------------------------------------------------- Viewport / Zoom

let zoomX = 1.0;
let zoomY = 1.0;
let vpX = 0;
let vpY = 0;
let userPanned = false;

const MIN_ZOOM = 1.0;
const MAX_ZOOM_X = HMAP_HISTORY / 30;
const MAX_ZOOM_Y = HMAP_LEVELS / 10;
const ZOOM_STEP = 1.08;

// Pan drag state
let isPanning = false;
let panPointerId = -1;
let panStartX = 0;
let panStartY = 0;
let panStartVpX = 0;
let panStartVpY = 0;

// ---------------------------------------------------------------- DOM refs

const $spotVal     = document.getElementById('spot-val')!;
const $tsVal       = document.getElementById('ts-val')!;
const $winVal      = document.getElementById('win-val')!;
const $compLabel   = document.getElementById('composite-label')!;
const $compMarker  = document.getElementById('composite-marker')!;
const $spotLine    = document.getElementById('spot-line-label')!;

// Metadata display elements
const $metaProduct  = document.getElementById('meta-product')!;
const $metaSymbol   = document.getElementById('meta-symbol')!;
const $metaTick     = document.getElementById('meta-tick')!;
const $metaBucket   = document.getElementById('meta-bucket')!;
const $metaMult     = document.getElementById('meta-mult')!;

// Warning banner
const $warningBanner = document.getElementById('warning-banner')!;

function $(id: string) { return document.getElementById(id)!; }

// -------------------------------------------------------- Price-grid helpers

/** Map dollar price to fractional grid row (row 0 = top = highest price). */
function priceToRow(priceDollars: number): number {
  return MAX_REL_TICKS - (priceDollars - anchorPriceDollars) / bucketDollars();
}

/** Map grid row to dollar price. */
function rowToPrice(row: number): number {
  return anchorPriceDollars + (MAX_REL_TICKS - row) * bucketDollars();
}

// --------------------------------------------------------- Viewport helpers

function visibleCols(): number { return HMAP_HISTORY / zoomX; }
function visibleRows(): number { return HMAP_LEVELS / zoomY; }

function clampViewport(): void {
  const maxVpX = Math.max(0, HMAP_HISTORY - visibleCols());
  const maxVpY = Math.max(0, HMAP_LEVELS - visibleRows());
  vpX = Math.max(0, Math.min(vpX, maxVpX));
  vpY = Math.max(0, Math.min(vpY, maxVpY));
}

/** Reset viewport to rightmost data, vertically centred on spot. */
function resetViewport(): void {
  vpX = Math.max(0, HMAP_HISTORY - visibleCols());
  const centerRow = (currentSpotDollars > 0 && anchorInitialized)
    ? priceToRow(currentSpotDollars)
    : HMAP_LEVELS / 2;
  vpY = centerRow - visibleRows() / 2;
  clampViewport();
}

/** Apply cursor-centred zoom and clamp. */
function applyZoom(
  fx: number, fy: number,
  mx: number, my: number,
  cw: number, ch: number,
): void {
  const oldW = visibleCols();
  const oldH = visibleRows();
  const dataX = vpX + (mx / cw) * oldW;
  const dataY = vpY + (my / ch) * oldH;

  zoomX = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM_X, zoomX * fx));
  zoomY = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM_Y, zoomY * fy));

  vpX = dataX - (mx / cw) * visibleCols();
  vpY = dataY - (my / ch) * visibleRows();
  clampViewport();
}

/**
 * Pick a "nice" price-label interval for the visible dollar range.
 * The candidate list is built dynamically from the bucket size so
 * that gridlines always land on bucket-aligned prices.
 */
function nicePriceInterval(visDollars: number, target: number): number {
  const raw = visDollars / target;
  const bucket = bucketDollars();
  // Build nice intervals: bucket, 1, 2, 5, 10, 20, 50, 100, ...
  // Always include the bucket size as the minimum interval.
  const nice: number[] = [];
  if (bucket < 1) nice.push(bucket);
  for (const base of [1, 2, 5, 10, 20, 50, 100, 200, 500]) {
    if (base >= bucket) nice.push(base);
  }
  for (const n of nice) {
    if (n >= raw) return n;
  }
  return nice[nice.length - 1];
}

// ---------------------------------------------------------- General helpers

/** Format nanosecond timestamp as HH:MM:SS ET. */
function formatTs(ns: bigint): string {
  const ms = Number(ns / 1_000_000n);
  const d = new Date(ms);
  // Offset to US/Eastern (rough: UTC-5)
  const et = new Date(d.getTime() - 5 * 3600_000);
  const hh = String(et.getUTCHours()).padStart(2, '0');
  const mm = String(et.getUTCMinutes()).padStart(2, '0');
  const ss = String(et.getUTCSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

/** Colour a value: positive -> green, negative -> red, zero -> grey. */
function signColour(v: number, intensity: number = 1): string {
  const t = Math.min(1, Math.abs(v) * intensity);
  if (v > 0) return `rgb(${30 + 30 * t}, ${160 + 95 * t}, ${100 + 55 * t})`;
  if (v < 0) return `rgb(${160 + 95 * t}, ${30 + 30 * t}, ${80 + 40 * t})`;
  return '#888';
}

function fmt(v: number, dp: number = 1): string {
  if (v == null || isNaN(v)) return '0';
  const s = v.toFixed(dp);
  return v > 0 ? `+${s}` : s;
}

/** Heatmap cell colour from depth + net flow.
 *  Uses log-scale normalization so large outlier levels don't crush
 *  the brightness of smaller (but still meaningful) depth.
 */
function heatmapRGB(
  depth: number, netFlow: number, maxDepth: number,
): [number, number, number] {
  const depthN = Math.min(1.0, Math.log1p(depth) / Math.log1p(maxDepth));
  const flowN = Math.tanh(netFlow / FLOW_NORM_SCALE);
  const lum = 0.04 + depthN * 0.56;

  if (flowN > 0.03) {
    const t = flowN;
    return [
      Math.round((0.04 + t * 0.06) * 255 * lum),
      Math.round((0.45 + t * 0.55) * 255 * lum),
      Math.round((0.35 + t * 0.35) * 255 * lum),
    ];
  } else if (flowN < -0.03) {
    const t = -flowN;
    return [
      Math.round((0.55 + t * 0.45) * 255 * lum),
      Math.round(0.04 * 255 * lum),
      Math.round((0.20 + t * 0.35) * 255 * lum),
    ];
  } else {
    return [
      Math.round(0.12 * 255 * lum),
      Math.round(0.12 * 255 * lum),
      Math.round(0.22 * 255 * lum),
    ];
  }
}

// --------------------------------------------------------------- Grid scroll

/**
 * Shift the heatmap pixel buffer and spot trail vertically when the
 * anchor price changes.  shiftRows > 0 means anchor moved up (price
 * increased), so pixel data slides down.
 */
function shiftGrid(shiftRows: number): void {
  const w = HMAP_HISTORY;
  const h = HMAP_LEVELS;
  const d = hmapData;
  const rowBytes = w * 4;

  if (Math.abs(shiftRows) >= h) {
    for (let i = 0; i < d.length; i += 4) {
      d[i] = 10; d[i + 1] = 10; d[i + 2] = 15; d[i + 3] = 255;
    }
    for (let i = 0; i < spotTrail.length; i++) spotTrail[i] = null;
    vpY += shiftRows;
    clampViewport();
    return;
  }

  if (shiftRows > 0) {
    for (let y = h - 1; y >= shiftRows; y--) {
      d.copyWithin(y * rowBytes, (y - shiftRows) * rowBytes, (y - shiftRows + 1) * rowBytes);
    }
    for (let y = 0; y < shiftRows; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        d[i] = 10; d[i + 1] = 10; d[i + 2] = 15; d[i + 3] = 255;
      }
    }
  } else {
    const absShift = -shiftRows;
    for (let y = 0; y < h - absShift; y++) {
      d.copyWithin(y * rowBytes, (y + absShift) * rowBytes, (y + absShift + 1) * rowBytes);
    }
    for (let y = h - absShift; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        d[i] = 10; d[i + 1] = 10; d[i + 2] = 15; d[i + 3] = 255;
      }
    }
  }

  for (let i = 0; i < spotTrail.length; i++) {
    if (spotTrail[i] !== null) {
      spotTrail[i] = spotTrail[i]! + shiftRows;
    }
  }

  vpY += shiftRows;
  clampViewport();
}

// -------------------------------------------------------- Heatmap buffer ops

/**
 * Push one column of flow data into the heatmap pixel buffer.
 * Maps rel_ticks to absolute price levels on the anchored grid.
 */
function pushHeatmapColumn(flow: FlowRow[], spotDollars: number): void {
  if (!anchorInitialized) {
    anchorPriceDollars = spotDollars;
    anchorInitialized = true;
  }

  currentSpotDollars = spotDollars;
  const spotRow = priceToRow(spotDollars);
  const bucket = bucketDollars();

  if (!userPanned && (spotRow < SCROLL_MARGIN || spotRow > HMAP_LEVELS - 1 - SCROLL_MARGIN)) {
    const newAnchor = spotDollars;
    const shiftRows = Math.round((newAnchor - anchorPriceDollars) / bucket);
    if (shiftRows !== 0) {
      shiftGrid(shiftRows);
      anchorPriceDollars = newAnchor;
    }
  }

  const w = HMAP_HISTORY;
  const h = HMAP_LEVELS;
  const d = hmapData;

  for (let y = 0; y < h; y++) {
    const rowOff = y * w * 4;
    d.copyWithin(rowOff, rowOff + 4, rowOff + w * 4);
  }

  for (let y = 0; y < h; y++) {
    const idx = (y * w + (w - 1)) * 4;
    d[idx] = 10; d[idx + 1] = 10; d[idx + 2] = 15; d[idx + 3] = 255;
  }

  spotTrail.shift();
  spotTrail.push(priceToRow(spotDollars));

  const byRow = new Map<number, { depth: number; net: number }>();
  for (const r of flow) {
    const absPrice = spotDollars + r.rel_ticks * bucket;
    const row = Math.round(priceToRow(absPrice));
    if (row < 0 || row >= h) continue;
    const existing = byRow.get(row);
    if (existing) {
      existing.depth += r.depth_qty_end;
      existing.net += r.net_flow;
    } else {
      byRow.set(row, { depth: r.depth_qty_end, net: r.net_flow });
    }
  }

  let maxD = 0;
  for (const v of byRow.values()) {
    if (v.depth > maxD) maxD = v.depth;
  }
  runningMaxDepth = Math.max(
    runningMaxDepth * DEPTH_NORM_PERCENTILE_DECAY,
    maxD,
  );

  for (const [row, v] of byRow) {
    const [r, g, b] = heatmapRGB(v.depth, v.net, runningMaxDepth);
    const idx = (row * w + (w - 1)) * 4;
    d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255;
  }
}

// ---------------------------------------------------------------- Rendering

let hmapOffscreen: HTMLCanvasElement | null = null;

function renderHeatmap(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  if (!userPanned) {
    resetViewport();
  }

  const srcW = visibleCols();
  const srcH = visibleRows();

  if (!hmapOffscreen) {
    hmapOffscreen = document.createElement('canvas');
    hmapOffscreen.width = HMAP_HISTORY;
    hmapOffscreen.height = HMAP_LEVELS;
  }
  const offCtx = hmapOffscreen.getContext('2d')!;
  const imgData = new ImageData(hmapData, HMAP_HISTORY, HMAP_LEVELS);
  offCtx.putImageData(imgData, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(hmapOffscreen, vpX, vpY, srcW, srcH, 0, 0, cw, ch);

  if (!anchorInitialized) return;

  const rowToY = (row: number): number => ((row - vpY) / srcH) * ch;
  const colToX = (col: number): number => ((col - vpX) / srcW) * cw;

  // Gridlines
  const topPrice = rowToPrice(vpY);
  const botPrice = rowToPrice(vpY + srcH);
  const visDollars = (topPrice - botPrice);
  const gridInterval = nicePriceInterval(visDollars, 40);

  ctx.strokeStyle = 'rgba(60, 60, 90, 0.2)';
  ctx.lineWidth = 0.5;
  const firstGrid = Math.ceil(botPrice / gridInterval) * gridInterval;
  for (let p = firstGrid; p <= topPrice; p += gridInterval) {
    const y = rowToY(priceToRow(p));
    if (y < -1 || y > ch + 1) continue;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(cw, y);
    ctx.stroke();
  }

  // Spot trail polyline
  ctx.strokeStyle = 'rgba(0, 255, 170, 0.85)';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < HMAP_HISTORY; i++) {
    const row = spotTrail[i];
    if (row === null || row < -2 || row > HMAP_LEVELS + 2) {
      started = false;
      continue;
    }
    const x = colToX(i + 0.5);
    const y = rowToY(row);
    if (x < -50 || x > cw + 50) { started = false; continue; }
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // Spot dot at current position (rightmost)
  const lastSpot = spotTrail[HMAP_HISTORY - 1];
  if (lastSpot !== null) {
    const x = colToX(HMAP_HISTORY - 0.5);
    const y = rowToY(lastSpot);

    if (x >= -20 && x <= cw + 20 && y >= -20 && y <= ch + 20) {
      ctx.fillStyle = 'rgba(0, 255, 170, 0.3)';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#00ffaa';
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = 'rgba(0, 255, 170, 0.25)';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(cw, y);
      ctx.stroke();
      ctx.setLineDash([]);

      if (y >= 0 && y <= ch) {
        $spotLine.style.top = `${y - 10}px`;
        $spotLine.style.display = '';
      } else {
        $spotLine.style.display = 'none';
      }
    } else {
      $spotLine.style.display = 'none';
    }
  }

  // Zoom indicator (only when zoomed)
  if (zoomX > 1.01 || zoomY > 1.01) {
    ctx.save();
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(0, 255, 170, 0.5)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${zoomX.toFixed(1)}x \u00d7 ${zoomY.toFixed(1)}x`, 6, ch - 6);
    ctx.font = '9px monospace';
    ctx.fillStyle = 'rgba(100, 100, 150, 0.4)';
    ctx.fillText('dblclick to reset', 6, ch - 18);
    ctx.restore();
  }
}

function renderPriceAxis(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (!ctx || !anchorInitialized) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, cw, ch);

  const srcH = visibleRows();
  const topPrice = rowToPrice(vpY);
  const botPrice = rowToPrice(vpY + srcH);
  const visDollars = topPrice - botPrice;
  const labelInterval = nicePriceInterval(visDollars, 20);
  const labelDp = labelInterval < 1 ? priceDecimals() : Math.min(priceDecimals(), 0);

  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';

  const firstLabel = Math.ceil(botPrice / labelInterval) * labelInterval;
  for (let p = firstLabel; p <= topPrice; p += labelInterval) {
    const row = priceToRow(p);
    const y = ((row - vpY) / srcH) * ch;
    if (y < 2 || y > ch - 2) continue;

    ctx.strokeStyle = 'rgba(80, 80, 120, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(5, y);
    ctx.stroke();

    ctx.fillStyle = '#666';
    ctx.fillText(p.toFixed(labelDp), 7, y);
  }

  // Spot price highlight
  if (currentSpotDollars > 0) {
    const spotRow = priceToRow(currentSpotDollars);
    const spotY = ((spotRow - vpY) / srcH) * ch;

    if (spotY >= -10 && spotY <= ch + 10) {
      ctx.fillStyle = 'rgba(0, 255, 170, 0.12)';
      ctx.fillRect(0, spotY - 7, cw, 14);

      ctx.fillStyle = '#00ffaa';
      ctx.font = 'bold 10px monospace';
      ctx.fillText(currentSpotDollars.toFixed(priceDecimals()), 7, spotY);

      ctx.strokeStyle = '#00ffaa';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, spotY);
      ctx.lineTo(5, spotY);
      ctx.stroke();
    }
  }
}

function renderProfile(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, cw, ch);

  if (!anchorInitialized) return;

  const srcH = visibleRows();
  const rowH = ch / srcH;
  const barAreaW = cw - 40;
  const barLeft = 40;
  const midX = barLeft + barAreaW / 2;
  const barMax = barAreaW / 2 - 2;
  const maxD = runningMaxDepth || 1;
  const bucket = bucketDollars();

  const topPrice = rowToPrice(vpY);
  const botPrice = rowToPrice(vpY + srcH);
  const visDollars = topPrice - botPrice;
  const labelInterval = nicePriceInterval(visDollars, 20);
  const labelDp = labelInterval < 1 ? priceDecimals() : Math.min(priceDecimals(), 0);

  ctx.font = '8px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  const firstLabel = Math.ceil(botPrice / labelInterval) * labelInterval;
  for (let p = firstLabel; p <= topPrice; p += labelInterval) {
    const row = priceToRow(p);
    const y = (row - vpY) * rowH + rowH / 2;
    if (y < 2 || y > ch - 2) continue;
    ctx.fillStyle = '#555';
    ctx.fillText(p.toFixed(labelDp), 36, y);
  }

  // Depth bars
  if (currentFlow.length > 0 && currentSpotDollars > 0) {
    for (const r of currentFlow) {
      const absPrice = currentSpotDollars + r.rel_ticks * bucket;
      const row = Math.round(priceToRow(absPrice));
      if (row < vpY - 1 || row > vpY + srcH + 1) continue;
      const y = (row - vpY) * rowH;
      const barW = Math.min(barMax, (Math.log1p(r.depth_qty_end) / Math.log1p(maxD)) * barMax);
      const flowT = Math.tanh(r.net_flow / FLOW_NORM_SCALE);
      const alpha = 0.4 + r.rest_fraction * 0.5;

      if (flowT >= 0) {
        ctx.fillStyle = `rgba(30, ${140 + 80 * flowT}, ${120 + 50 * flowT}, ${alpha})`;
      } else {
        ctx.fillStyle = `rgba(${140 + 80 * (-flowT)}, 30, ${60 + 40 * (-flowT)}, ${alpha})`;
      }

      if (r.side === 'B') {
        ctx.fillRect(midX - barW, y, barW, Math.max(1, rowH - 0.5));
      } else {
        ctx.fillRect(midX, y, barW, Math.max(1, rowH - 0.5));
      }
    }
  }

  ctx.strokeStyle = 'rgba(100, 100, 150, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(midX, 0);
  ctx.lineTo(midX, ch);
  ctx.stroke();

  if (currentSpotDollars > 0) {
    const spotRow = priceToRow(currentSpotDollars);
    const spotY = (spotRow - vpY) * rowH;
    if (spotY >= 0 && spotY <= ch) {
      ctx.strokeStyle = 'rgba(0, 255, 170, 0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(barLeft, spotY);
      ctx.lineTo(cw, spotY);
      ctx.stroke();
    }
  }

  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(100, 180, 200, 0.6)';
  ctx.fillText('BID', barLeft + 2, 12);
  ctx.fillStyle = 'rgba(200, 140, 100, 0.6)';
  ctx.fillText('ASK', cw - 24, 12);
}

function updateSignalPanel(): void {
  const s = currentSignals;
  if (!s) return;

  $('sig-composite').textContent = fmt(s.composite, 1);
  $('sig-composite').style.color = signColour(s.composite, 0.01);

  $('sig-confidence').textContent = `${(s.confidence * 100).toFixed(0)}%`;
  $('sig-strength').textContent = `${(s.strength * 100).toFixed(0)}%`;

  const norm = Math.tanh(s.composite / 2000) * 0.5 + 0.5;
  $compMarker.style.left = `${norm * 100}%`;

  $('sig-d1').textContent = fmt(s.d1_composite, 2);
  $('sig-d1').style.color = signColour(s.d1_composite, 0.05);
  $('sig-d2').textContent = fmt(s.d2_composite, 2);
  $('sig-d2').style.color = signColour(s.d2_composite, 0.1);
  $('sig-d3').textContent = fmt(s.d3_composite, 3);
  $('sig-d3').style.color = signColour(s.d3_composite, 0.2);

  fillGauge('d1-fill', s.d1_composite, 500);
  fillGauge('d2-fill', s.d2_composite, 200);
  fillGauge('d3-fill', s.d3_composite, 100);

  $('bot-d1').textContent = fmt(s.d1_composite, 1);
  $('bot-d2').textContent = fmt(s.d2_composite, 1);
  $('bot-d3').textContent = fmt(s.d3_composite, 2);

  $('sig-vac-above').textContent = fmt(s.vacuum_above, 0);
  $('sig-vac-above').style.color = signColour(s.vacuum_above, 0.005);
  $('sig-vac-below').textContent = fmt(s.vacuum_below, 0);
  $('sig-vac-below').style.color = signColour(-s.vacuum_below, 0.005);
  $('sig-flow').textContent = fmt(s.flow_imbalance, 0);
  $('sig-flow').style.color = signColour(s.flow_imbalance, 0.005);
  $('sig-fill').textContent = fmt(s.fill_imbalance, 0);
  $('sig-fill').style.color = signColour(s.fill_imbalance, 0.01);
  $('sig-depth').textContent = fmt(s.depth_imbalance, 3);
  $('sig-depth').style.color = signColour(s.depth_imbalance, 2);
  $('sig-rest-depth').textContent = fmt(s.rest_depth_imbalance, 3);
  $('sig-rest-depth').style.color = signColour(s.rest_depth_imbalance, 2);
  $('sig-drain-ask').textContent = fmt(s.resting_drain_ask, 0);
  $('sig-drain-bid').textContent = fmt(s.resting_drain_bid, 0);

  const comp = s.composite;
  if (comp > 50) {
    $compLabel.textContent = `BULLISH ${fmt(comp, 0)}`;
    $compLabel.style.color = '#22cc66';
  } else if (comp < -50) {
    $compLabel.textContent = `BEARISH ${fmt(comp, 0)}`;
    $compLabel.style.color = '#cc2255';
  } else {
    $compLabel.textContent = `NEUTRAL ${fmt(comp, 0)}`;
    $compLabel.style.color = '#888';
  }
}

function fillGauge(id: string, value: number, scale: number): void {
  const el = document.getElementById(id);
  if (!el) return;
  const norm = Math.tanh(value / scale);
  const pct = Math.abs(norm) * 50;
  if (norm >= 0) {
    el.style.left = '50%';
    el.style.width = `${pct}%`;
    el.style.background = '#22cc66';
  } else {
    el.style.left = `${50 - pct}%`;
    el.style.width = `${pct}%`;
    el.style.background = '#cc2255';
  }
}

// -------------------------------------------- Runtime config application

/**
 * Apply a runtime config received from the server.
 * Updates the metadata display and hides the legacy fallback warning.
 */
function applyRuntimeConfig(cfg: RuntimeConfig): void {
  runtimeConfig = cfg;
  configReceived = true;

  // Update metadata display
  $metaProduct.textContent = cfg.product_type;
  $metaSymbol.textContent = cfg.symbol;
  $metaTick.textContent = `$${cfg.tick_size}`;
  $metaBucket.textContent = `$${cfg.rel_tick_size}`;
  $metaMult.textContent = String(cfg.contract_multiplier);

  // Hide legacy fallback warning
  $warningBanner.style.display = 'none';

  console.log(
    `[VP] Runtime config applied: product_type=${cfg.product_type} ` +
    `symbol=${cfg.symbol} bucket=$${cfg.bucket_size_dollars} ` +
    `tick=$${cfg.tick_size} decimals=${cfg.price_decimals} ` +
    `multiplier=${cfg.contract_multiplier} version=${cfg.config_version}`
  );
}

/**
 * Activate the legacy fallback warning banner.
 * Called when the first data batch arrives without a prior config message.
 */
function activateLegacyFallback(): void {
  console.warn(
    '[VP] DEPRECATION: No runtime config received from server. ' +
    'Falling back to legacy equity defaults (bucket=$0.50, decimals=2). ' +
    'Update the backend to send a config control message before data.'
  );
  $warningBanner.style.display = '';
}

// -------------------------------------------------------- URL contract

/** Parse and validate URL query parameters.  Fails fast if product_type is missing. */
function parseStreamParams(): StreamParams {
  const params = new URLSearchParams(window.location.search);
  const product_type = params.get('product_type');
  if (!product_type) {
    const msg =
      'Missing required query parameter: product_type. ' +
      'Example: ?product_type=equity_mbo&symbol=QQQ&dt=2026-02-06&speed=1';
    console.error(`[VP] ${msg}`);
    // Show error in the UI
    $warningBanner.textContent = msg;
    $warningBanner.style.display = '';
    $warningBanner.style.background = '#660000';
  }

  return {
    product_type: product_type || 'equity_mbo',
    symbol: params.get('symbol') || 'QQQ',
    dt: params.get('dt') || '2026-02-06',
    speed: params.get('speed') || '1',
    skip: params.get('skip') || '5',
  };
}

// ---------------------------------------------------------------- WebSocket

/** Cached stream params so reconnect preserves original params. */
let streamParams: StreamParams | null = null;

function connectWS(): void {
  if (!streamParams) {
    streamParams = parseStreamParams();
  }
  const { product_type, symbol, dt, speed, skip } = streamParams;

  const url =
    `ws://localhost:${WS_PORT}/v1/vacuum-pressure/stream` +
    `?product_type=${encodeURIComponent(product_type)}` +
    `&symbol=${encodeURIComponent(symbol)}` +
    `&dt=${encodeURIComponent(dt)}` +
    `&speed=${encodeURIComponent(speed)}` +
    `&skip_minutes=${encodeURIComponent(skip)}`;

  console.log(`[VP] Connecting to: ${url}`);
  const ws = new WebSocket(url);

  // Message queue pattern (aligned with velocity ws-client.ts)
  let pendingSurface: string | null = null;
  const messageQueue: MessageEvent[] = [];
  let isProcessing = false;

  const processQueue = async (): Promise<void> => {
    if (isProcessing) return;
    isProcessing = true;

    while (messageQueue.length > 0) {
      const event = messageQueue.shift();
      if (!event) continue;

      try {
        if (typeof event.data === 'string') {
          // ── Control frame (JSON) ──
          const msg = JSON.parse(event.data) as Record<string, unknown>;

          if (msg.type === 'runtime_config') {
            // Runtime config -- must arrive before first data batch
            applyRuntimeConfig({
              product_type: msg.product_type as string,
              symbol: msg.symbol as string,
              symbol_root: (msg.symbol_root ?? '') as string,
              price_scale: msg.price_scale as number,
              tick_size: msg.tick_size as number,
              bucket_size_dollars: msg.bucket_size_dollars as number,
              rel_tick_size: msg.rel_tick_size as number,
              grid_max_ticks: msg.grid_max_ticks as number,
              contract_multiplier: msg.contract_multiplier as number,
              qty_unit: (msg.qty_unit ?? 'shares') as string,
              price_decimals: msg.price_decimals as number,
              config_version: (msg.config_version ?? '') as string,
            });
          } else if (msg.type === 'batch_start') {
            // If config was never received, activate legacy fallback once
            if (!configReceived) {
              activateLegacyFallback();
              // Prevent re-triggering on subsequent batches
              configReceived = true;
            }
            windowCount++;
            $winVal.textContent = String(windowCount);
          } else if (msg.type === 'surface_header') {
            pendingSurface = msg.surface as string;
          }
        } else if (event.data instanceof Blob) {
          // ── Binary frame (Arrow IPC) ──
          const surface = pendingSurface;
          if (!surface) continue;
          pendingSurface = null;

          const buffer = await event.data.arrayBuffer();
          const table = tableFromIPC(buffer);

          if (surface === 'snap' && table.numRows > 0) {
            const row = table.get(0);
            if (row) {
              const j = row.toJSON() as Record<string, unknown>;
              snap = {
                window_end_ts_ns: BigInt(j.window_end_ts_ns as string),
                mid_price: j.mid_price as number,
                spot_ref_price_int: BigInt(j.spot_ref_price_int as string),
                best_bid_price_int: BigInt(j.best_bid_price_int as string),
                best_ask_price_int: BigInt(j.best_ask_price_int as string),
                book_valid: j.book_valid as boolean,
              };
              currentSpotDollars = snap.mid_price;
              $spotVal.textContent = `$${snap.mid_price.toFixed(priceDecimals())}`;
              $tsVal.textContent = formatTs(snap.window_end_ts_ns);
            }
          } else if (surface === 'flow') {
            const rows: FlowRow[] = [];
            for (let i = 0; i < table.numRows; i++) {
              const row = table.get(i);
              if (!row) continue;
              const j = row.toJSON() as Record<string, unknown>;
              rows.push({
                rel_ticks: Number(j.rel_ticks),
                side: j.side as string,
                depth_qty_end: j.depth_qty_end as number,
                add_qty: j.add_qty as number,
                pull_qty: j.pull_qty as number,
                fill_qty: j.fill_qty as number,
                depth_qty_rest: j.depth_qty_rest as number,
                pull_qty_rest: j.pull_qty_rest as number,
                net_flow: j.net_flow as number,
                vacuum_intensity: j.vacuum_intensity as number,
                pressure_intensity: j.pressure_intensity as number,
                rest_fraction: j.rest_fraction as number,
              });
            }
            currentFlow = rows;
            pushHeatmapColumn(rows, currentSpotDollars);
          } else if (surface === 'signals' && table.numRows > 0) {
            const row = table.get(0);
            if (row) {
              const j = row.toJSON() as Record<string, unknown>;
              currentSignals = {
                window_end_ts_ns: BigInt(j.window_end_ts_ns as string),
                vacuum_above: (j.vacuum_above ?? 0) as number,
                vacuum_below: (j.vacuum_below ?? 0) as number,
                resting_drain_ask: (j.resting_drain_ask ?? 0) as number,
                resting_drain_bid: (j.resting_drain_bid ?? 0) as number,
                flow_imbalance: (j.flow_imbalance ?? 0) as number,
                fill_imbalance: (j.fill_imbalance ?? 0) as number,
                depth_imbalance: (j.depth_imbalance ?? 0) as number,
                rest_depth_imbalance: (j.rest_depth_imbalance ?? 0) as number,
                bid_migration_com: (j.bid_migration_com ?? 0) as number,
                ask_migration_com: (j.ask_migration_com ?? 0) as number,
                composite: (j.composite ?? 0) as number,
                d1_composite: (j.d1_composite ?? 0) as number,
                d2_composite: (j.d2_composite ?? 0) as number,
                d3_composite: (j.d3_composite ?? 0) as number,
                confidence: (j.confidence ?? 0) as number,
                strength: (j.strength ?? 0) as number,
              };
              updateSignalPanel();
            }
          }
        }
      } catch (e) {
        console.error('[VP] Message processing error:', e);
      }
    }

    isProcessing = false;
  };

  ws.onopen = () => console.log('[VP] WebSocket connected');
  ws.onmessage = (event: MessageEvent) => {
    messageQueue.push(event);
    processQueue();
  };
  ws.onerror = (err) => console.error('[VP] WebSocket error:', err);
  ws.onclose = () => {
    console.log('[VP] WebSocket closed, reconnecting in 3s...');
    setTimeout(connectWS, 3000);
  };
}

// ------------------------------------------------------------ Render loop

function startRenderLoop(): void {
  const hmapCanvas = document.getElementById('heatmap-canvas') as HTMLCanvasElement;
  const profCanvas = document.getElementById('profile-canvas') as HTMLCanvasElement;
  const axisCanvas = document.getElementById('price-axis-canvas') as HTMLCanvasElement;

  // Zoom (mouse wheel on heatmap)
  hmapCanvas.addEventListener('wheel', (e: WheelEvent) => {
    e.preventDefault();
    const rect = hmapCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let magnitude: number;
    if (e.deltaMode === 1) {
      magnitude = Math.abs(e.deltaY) * 40;
    } else if (e.deltaMode === 2) {
      magnitude = Math.abs(e.deltaY) * 400;
    } else {
      magnitude = Math.abs(e.deltaY);
    }

    const steps = Math.min(magnitude / 100, 3);
    const factor = e.deltaY > 0
      ? Math.pow(1 / ZOOM_STEP, steps)
      : Math.pow(ZOOM_STEP, steps);

    if (e.ctrlKey || e.metaKey) {
      applyZoom(factor, factor, mx, my, rect.width, rect.height);
    } else if (e.shiftKey) {
      applyZoom(factor, 1, mx, my, rect.width, rect.height);
    } else {
      applyZoom(1, factor, mx, my, rect.width, rect.height);
    }
  }, { passive: false });

  // Pan (pointer drag on heatmap)
  hmapCanvas.addEventListener('pointerdown', (e: PointerEvent) => {
    if (e.button !== 0) return;
    isPanning = true;
    panPointerId = e.pointerId;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panStartVpX = vpX;
    panStartVpY = vpY;
    hmapCanvas.setPointerCapture(e.pointerId);
    hmapCanvas.style.cursor = 'grabbing';
  });

  hmapCanvas.addEventListener('pointermove', (e: PointerEvent) => {
    if (!isPanning || e.pointerId !== panPointerId) return;
    const rect = hmapCanvas.getBoundingClientRect();
    const dxPx = e.clientX - panStartX;
    const dyPx = e.clientY - panStartY;
    const srcW = visibleCols();
    const srcH = visibleRows();

    vpX = panStartVpX - (dxPx / rect.width) * srcW;
    vpY = panStartVpY - (dyPx / rect.height) * srcH;
    clampViewport();

    if (Math.abs(dxPx) > 3 || Math.abs(dyPx) > 3) {
      userPanned = true;
    }
  });

  hmapCanvas.addEventListener('pointerup', (e: PointerEvent) => {
    if (e.pointerId !== panPointerId) return;
    isPanning = false;
    panPointerId = -1;
    hmapCanvas.releasePointerCapture(e.pointerId);
    hmapCanvas.style.cursor = 'grab';
  });

  hmapCanvas.addEventListener('pointercancel', (e: PointerEvent) => {
    if (e.pointerId !== panPointerId) return;
    isPanning = false;
    panPointerId = -1;
    hmapCanvas.style.cursor = 'grab';
  });

  // Double-click to reset to auto-follow
  hmapCanvas.addEventListener('dblclick', () => {
    userPanned = false;
    zoomX = 1;
    zoomY = 1;
    resetViewport();
  });

  hmapCanvas.style.cursor = 'grab';

  function frame() {
    renderHeatmap(hmapCanvas);
    renderProfile(profCanvas);
    renderPriceAxis(axisCanvas);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// -------------------------------------------------------------------- Init

connectWS();
startRenderLoop();
