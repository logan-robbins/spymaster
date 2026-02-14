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

/** Dense grid bucket row from event-driven mode (mode=event).
 *  Two-force model: pressure (depth building, always >= 0) and
 *  vacuum (depth draining, always >= 0).  resistance_variant removed. */
interface GridBucketRow {
  k: number;
  pressure_variant: number;
  vacuum_variant: number;
  add_mass: number;
  pull_mass: number;
  fill_mass: number;
  rest_depth: number;
  v_add: number;
  v_pull: number;
  v_fill: number;
  v_rest_depth: number;
  a_add: number;
  a_pull: number;
  a_fill: number;
  a_rest_depth: number;
  j_add: number;
  j_pull: number;
  j_fill: number;
  j_rest_depth: number;
  last_event_id: number;
}

/** Parsed and validated URL query parameters. */
interface StreamParams {
  product_type: string;
  symbol: string;
  dt: string;
  speed: string;
  start_time?: string;
  throttle_ms?: string;
}

// --------------------------------------------------------- Layout constants

const WS_PORT = 8002;
const MAX_REL_TICKS = 50;                   // +/-50 buckets from anchor
const HMAP_LEVELS = MAX_REL_TICKS * 2 + 1;  // 101 rows
const HMAP_HISTORY = 360;                    // 6 min of 1-second columns
const FLOW_NORM_SCALE = 500;                 // characteristic shares for tanh norm
const DEPTH_NORM_PERCENTILE_DECAY = 0.995;
const SCROLL_MARGIN = 10;                    // rows from edge before auto-scroll

// --------------------------------------------------- Runtime config state

let runtimeConfig: RuntimeConfig | null = null;
let configReceived = false;

/** Active bucket size in dollars from runtime config. */
function bucketDollars(): number {
  if (!runtimeConfig) {
    streamContractError('runtime_config', 'runtime config missing before grid data');
  }
  return runtimeConfig.bucket_size_dollars;
}

/** Active price decimal precision from runtime config. */
function priceDecimals(): number {
  if (!runtimeConfig) {
    streamContractError('runtime_config', 'runtime config missing before price render');
  }
  return runtimeConfig.price_decimals;
}

// ----------------------------------------------------------- Stream state

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

const streamContractErrors = new Set<string>();

// --------------------------------------------------- Event-mode state
let isEventMode = false;
let currentGrid: Map<number, GridBucketRow> = new Map();
/** Per-bucket last_event_id tracker for persistence (keyed by heatmap row). */
const lastRenderedEventIdByRow: Map<number, number> = new Map();
/** Running max for |pressure_variant| adaptive normalization. */
let runningMaxPressure = 10;

// --------------------------------------------------------- Viewport / Zoom

let zoomX = 1.0;
let zoomY = 1.0;
let vpX = 0;
let vpY = 0;
let userPanned = false;

const MIN_ZOOM_X = 1.0;       // no horizontal zoom-out (buffer only holds 360 cols)
const MIN_ZOOM_Y = 0.15;      // vertical zoom-out: see ~540 price levels compressed
const MAX_ZOOM_X = HMAP_HISTORY / 30;  // 12× zoom in
const MAX_ZOOM_Y = HMAP_LEVELS / 10;   // ~8× zoom in
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
const $winIdVal    = document.getElementById('win-id-val')!;
const $spotLine    = document.getElementById('spot-line-label')!;

// Metadata display elements
const $metaProduct  = document.getElementById('meta-product')!;
const $metaSymbol   = document.getElementById('meta-symbol')!;
const $metaTick     = document.getElementById('meta-tick')!;
const $metaBucket   = document.getElementById('meta-bucket')!;
const $metaMult     = document.getElementById('meta-mult')!;

// Warning banner
const $warningBanner = document.getElementById('warning-banner')!;
const $ctrlPause = document.getElementById('ctrl-pause') as HTMLButtonElement;
const $ctrlPlay = document.getElementById('ctrl-play') as HTMLButtonElement;
const $ctrlRestart = document.getElementById('ctrl-restart') as HTMLButtonElement;
const $streamState = document.getElementById('stream-state')!;

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
  const vCols = visibleCols();
  const vRows = visibleRows();

  if (vCols >= HMAP_HISTORY) {
    // Zoomed out horizontally: pin right edge (latest data visible)
    vpX = HMAP_HISTORY - vCols;
  } else {
    vpX = Math.max(0, Math.min(vpX, HMAP_HISTORY - vCols));
  }

  if (vRows >= HMAP_LEVELS) {
    // Zoomed out vertically: centre the data band, allow limited panning
    const slack = (vRows - HMAP_LEVELS) * 0.25;
    const centre = (HMAP_LEVELS - vRows) / 2;
    vpY = Math.max(centre - slack, Math.min(vpY, centre + slack));
  } else {
    vpY = Math.max(0, Math.min(vpY, HMAP_LEVELS - vRows));
  }
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

  zoomX = Math.max(MIN_ZOOM_X, Math.min(MAX_ZOOM_X, zoomX * fx));
  zoomY = Math.max(MIN_ZOOM_Y, Math.min(MAX_ZOOM_Y, zoomY * fy));

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

/** Format nanosecond timestamp as "h:MM:SS AM/PM EST". */
function formatTs(ns: bigint): string {
  const ms = Number(ns / 1_000_000n);
  const d = new Date(ms);
  // Fixed EST offset (UTC-5). Covers regular + extended trading hours.
  const et = new Date(d.getTime() - 5 * 3600_000);
  let h = et.getUTCHours();
  const ampm = h >= 12 ? 'PM' : 'AM';
  h = h % 12 || 12;
  const mm = String(et.getUTCMinutes()).padStart(2, '0');
  const ss = String(et.getUTCSeconds()).padStart(2, '0');
  return `${h}:${mm}:${ss} ${ampm} EST`;
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

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function projectionDataWidth(totalWidth: number): number {
  return Math.max(32, totalWidth);
}

function streamContractError(surface: string, detail: string): never {
  const key = `${surface}:${detail}`;
  if (!streamContractErrors.has(key)) {
    streamContractErrors.add(key);
    const msg = `[VP] Stream contract error (${surface}): ${detail}`;
    console.error(msg);
    $warningBanner.textContent = msg;
    $warningBanner.style.display = '';
    $warningBanner.style.background = '#660000';
    $warningBanner.style.color = '#ffbbbb';
  }
  throw new Error(`[VP] stream contract violation in ${surface}: ${detail}`);
}

function requireField(
  surface: string,
  obj: Record<string, unknown>,
  field: string,
): unknown {
  const value = obj[field];
  if (value === undefined || value === null) {
    streamContractError(surface, `missing required field "${field}"`);
  }
  return value;
}

function requireNumberField(
  surface: string,
  obj: Record<string, unknown>,
  field: string,
): number {
  const raw = requireField(surface, obj, field);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    streamContractError(surface, `field "${field}" is not a finite number`);
  }
  return parsed;
}

function requireStringField(
  surface: string,
  obj: Record<string, unknown>,
  field: string,
): string {
  const raw = requireField(surface, obj, field);
  if (typeof raw !== 'string') {
    streamContractError(surface, `field "${field}" is not a string`);
  }
  return raw;
}

function requireBooleanField(
  surface: string,
  obj: Record<string, unknown>,
  field: string,
): boolean {
  const raw = requireField(surface, obj, field);
  if (typeof raw !== 'boolean') {
    streamContractError(surface, `field "${field}" is not a boolean`);
  }
  return raw;
}

function requireBigIntField(
  surface: string,
  obj: Record<string, unknown>,
  field: string,
): bigint {
  const raw = requireField(surface, obj, field);
  try {
    return typeof raw === 'bigint' ? raw : BigInt(String(raw));
  } catch {
    streamContractError(surface, `field "${field}" is not bigint-coercible`);
  }
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
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

// ------------------------------------------------ Event-mode heatmap column

/**
 * Push one column of dense grid data into the heatmap pixel buffer.
 * Analogous to pushHeatmapColumn() but for event-driven mode.
 *
 * Uses pressure_variant as the primary color signal (replaces net_flow)
 * and rest_depth as the depth signal (replaces depth_qty_end).
 *
 * Persistence: only updates heatmap pixels for buckets whose last_event_id
 * differs from the previously rendered event_id at that row position.
 * Untouched buckets carry forward their existing pixel color.
 */
function pushHeatmapColumnFromGrid(
  grid: Map<number, GridBucketRow>,
  spotDollars: number,
): void {
  if (!anchorInitialized) {
    anchorPriceDollars = spotDollars;
    anchorInitialized = true;
  }

  currentSpotDollars = spotDollars;
  const spotRow = priceToRow(spotDollars);
  const bucket = bucketDollars();

  // Auto-scroll when spot nears edge
  if (!userPanned && (spotRow < SCROLL_MARGIN || spotRow > HMAP_LEVELS - 1 - SCROLL_MARGIN)) {
    const newAnchor = spotDollars;
    const shiftRows = Math.round((newAnchor - anchorPriceDollars) / bucket);
    if (shiftRows !== 0) {
      shiftGrid(shiftRows);
      anchorPriceDollars = newAnchor;
      // Shift the lastRenderedEventIdByRow keys accordingly
      const shifted = new Map<number, number>();
      for (const [row, eid] of lastRenderedEventIdByRow) {
        const newRow = row + shiftRows;
        if (newRow >= 0 && newRow < HMAP_LEVELS) {
          shifted.set(newRow, eid);
        }
      }
      lastRenderedEventIdByRow.clear();
      for (const [row, eid] of shifted) {
        lastRenderedEventIdByRow.set(row, eid);
      }
    }
  }

  const w = HMAP_HISTORY;
  const h = HMAP_LEVELS;
  const d = hmapData;

  // Scroll heatmap left by one column
  for (let y = 0; y < h; y++) {
    const rowOff = y * w * 4;
    d.copyWithin(rowOff, rowOff + 4, rowOff + w * 4);
  }

  // Clear rightmost column to dark background
  for (let y = 0; y < h; y++) {
    const idx = (y * w + (w - 1)) * 4;
    d[idx] = 10; d[idx + 1] = 10; d[idx + 2] = 15; d[idx + 3] = 255;
  }

  // Advance spot trail
  spotTrail.shift();
  spotTrail.push(priceToRow(spotDollars));

  // Map grid buckets to heatmap rows and track adaptive normalization.
  // Two-force model: both pressure_variant and vacuum_variant are >= 0.
  // Net flow = pressure - vacuum: positive = building, negative = draining.
  let maxNetForce = 0;
  let maxRestD = 0;
  for (const bucket_row of grid.values()) {
    const netForce = Math.abs(bucket_row.pressure_variant - bucket_row.vacuum_variant);
    if (netForce > maxNetForce) maxNetForce = netForce;
    if (bucket_row.rest_depth > maxRestD) maxRestD = bucket_row.rest_depth;
  }
  runningMaxPressure = Math.max(
    runningMaxPressure * DEPTH_NORM_PERCENTILE_DECAY,
    maxNetForce,
  );
  runningMaxDepth = Math.max(
    runningMaxDepth * DEPTH_NORM_PERCENTILE_DECAY,
    maxRestD,
  );

  // Write pixels for each grid bucket
  for (const [k, bucketData] of grid) {
    const absPrice = spotDollars + k * bucket;
    const row = Math.round(priceToRow(absPrice));
    if (row < 0 || row >= h) continue;

    // Persistence check: skip if this bucket's last_event_id hasn't changed
    const prevEventId = lastRenderedEventIdByRow.get(row);
    if (prevEventId !== undefined && prevEventId === bucketData.last_event_id) {
      continue;
    }
    lastRenderedEventIdByRow.set(row, bucketData.last_event_id);

    // Heatmap colour: net force = pressure (building) - vacuum (draining).
    //   Positive (building) → cyan-green (pressure zone)
    //   Negative (draining) → red-magenta (vacuum zone)
    const netForce = bucketData.pressure_variant - bucketData.vacuum_variant;
    const scaledForce = netForce * (FLOW_NORM_SCALE / Math.max(1, runningMaxPressure));
    const [r, g, b] = heatmapRGB(bucketData.rest_depth, scaledForce, runningMaxDepth);
    const idx = (row * w + (w - 1)) * 4;
    d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255;
  }
}

/**
 * Compute aggregate pressure/vacuum from the current grid (two-force model).
 *
 * Both pressure_variant and vacuum_variant are always >= 0:
 *   pressure = rate of depth BUILDING at a level
 *   vacuum   = rate of depth DRAINING at a level
 *
 * Net signal (positive = bullish, negative = bearish):
 *   pressureBelow (support forming)     → +bullish
 *   pressureAbove (resistance forming)  → −bearish
 *   vacuumAbove   (path clearing up)    → +bullish
 *   vacuumBelow   (support crumbling)   → −bearish
 */
function computeGridAggregates(grid: Map<number, GridBucketRow>): {
  pressureAbove: number;
  pressureBelow: number;
  vacuumAbove: number;
  vacuumBelow: number;
  netPressure: number;
  totalRestDepthAbove: number;
  totalRestDepthBelow: number;
} {
  let pressureAbove = 0;
  let pressureBelow = 0;
  let vacuumAbove = 0;
  let vacuumBelow = 0;
  let totalRestDepthAbove = 0;
  let totalRestDepthBelow = 0;

  for (const [k, b] of grid) {
    if (k > 0) {
      // Above spot: pressure = resistance forming, vacuum = path clearing
      pressureAbove += b.pressure_variant;
      vacuumAbove += b.vacuum_variant;
      totalRestDepthAbove += b.rest_depth;
    } else if (k < 0) {
      // Below spot: pressure = support forming, vacuum = support crumbling
      pressureBelow += b.pressure_variant;
      vacuumBelow += b.vacuum_variant;
      totalRestDepthBelow += b.rest_depth;
    }
  }

  const netPressure = (pressureBelow - pressureAbove) + (vacuumAbove - vacuumBelow);

  return {
    pressureAbove, pressureBelow,
    vacuumAbove, vacuumBelow,
    netPressure,
    totalRestDepthAbove, totalRestDepthBelow,
  };
}

/**
 * Update the signal panel from event-mode grid data (two-force model).
 * Produces explainable, in-trade guidance from pressure/vacuum balance.
 */
function updateSignalPanelFromGrid(grid: Map<number, GridBucketRow>): void {
  const agg = computeGridAggregates(grid);
  const bullEdge = agg.pressureBelow + agg.vacuumAbove;
  const bearEdge = agg.pressureAbove + agg.vacuumBelow;
  const netEdge = bullEdge - bearEdge;
  const forceTotal = bullEdge + bearEdge;
  const conviction = forceTotal > 0 ? clamp(Math.abs(netEdge) / forceTotal, 0, 1) : 0;
  const restDepthTilt = agg.totalRestDepthBelow - agg.totalRestDepthAbove;
  const restDepthTotal = agg.totalRestDepthAbove + agg.totalRestDepthBelow;

  const upAligned = agg.vacuumAbove > agg.pressureAbove && agg.pressureBelow > agg.vacuumBelow;
  const downAligned = agg.pressureAbove > agg.vacuumAbove && agg.vacuumBelow > agg.pressureBelow;

  let state = 'CHOP';
  let stateColor = '#2f3138';
  let stateTextColor = '#ccaa22';
  let stateNote = 'Directional forces are mixed. Reduce conviction and wait for imbalance.';
  if (netEdge > 5 && upAligned) {
    state = 'UP BIAS';
    stateColor = '#163124';
    stateTextColor = '#22cc66';
    stateNote = 'Vacuum above and pressure below align for upside continuation.';
  } else if (netEdge < -5 && downAligned) {
    state = 'DOWN BIAS';
    stateColor = '#341722';
    stateTextColor = '#cc2255';
    stateNote = 'Pressure above and vacuum below align for downside continuation.';
  } else if (netEdge > 2) {
    state = 'UP LEAN';
    stateColor = '#1b2c24';
    stateTextColor = '#66d692';
    stateNote = 'Bull edge leads, but structure is not fully aligned yet.';
  } else if (netEdge < -2) {
    state = 'DOWN LEAN';
    stateColor = '#2f1a22';
    stateTextColor = '#dd667f';
    stateNote = 'Bear edge leads, but structure is not fully aligned yet.';
  }

  let longAction = 'WAIT';
  let longColor = '#888';
  let longBg = '#2f3138';
  let longNote = 'No clear continuation edge for longs.';
  if (netEdge > 5 && upAligned) {
    longAction = 'HOLD';
    longColor = '#22cc66';
    longBg = '#163124';
    longNote = 'Support is building below while liquidity is clearing above.';
  } else if (netEdge > -2) {
    longAction = 'TIGHTEN';
    longColor = '#ccaa22';
    longBg = '#332a1a';
    longNote = 'Long edge is weakening. Tighten stop or reduce size.';
  } else {
    longAction = 'EXIT';
    longColor = '#cc2255';
    longBg = '#341722';
    longNote = 'Bear pressure dominates. Protection for longs is deteriorating.';
  }

  let shortAction = 'WAIT';
  let shortColor = '#888';
  let shortBg = '#2f3138';
  let shortNote = 'No clear continuation edge for shorts.';
  if (netEdge < -5 && downAligned) {
    shortAction = 'HOLD';
    shortColor = '#cc2255';
    shortBg = '#341722';
    shortNote = 'Resistance is building above while support is draining below.';
  } else if (netEdge < 2) {
    shortAction = 'TIGHTEN';
    shortColor = '#ccaa22';
    shortBg = '#332a1a';
    shortNote = 'Short edge is weakening. Tighten stop or reduce size.';
  } else {
    shortAction = 'EXIT';
    shortColor = '#22cc66';
    shortBg = '#163124';
    shortNote = 'Bull pressure dominates. Shorts risk squeeze and reversal.';
  }

  let riskFlag = 'BALANCED';
  let riskColor = '#888';
  if (agg.vacuumBelow > Math.max(agg.pressureBelow * 1.15, 2)) {
    riskFlag = 'SUPPORT FAILING';
    riskColor = '#cc2255';
  } else if (agg.vacuumAbove > Math.max(agg.pressureAbove * 1.15, 2)) {
    riskFlag = 'RESISTANCE THINNING';
    riskColor = '#22cc66';
  } else if (Math.abs(netEdge) >= 2) {
    riskFlag = netEdge > 0 ? 'UP ADVANTAGE' : 'DOWN ADVANTAGE';
    riskColor = netEdge > 0 ? '#22cc66' : '#cc2255';
  }

  $('sig-net-edge').textContent = fmt(netEdge, 1);
  $('sig-net-edge').style.color = signColour(netEdge, 0.01);
  $('sig-bull-edge').textContent = fmt(bullEdge, 1);
  $('sig-bull-edge').style.color = '#22cc66';
  $('sig-bear-edge').textContent = fmt(bearEdge, 1);
  $('sig-bear-edge').style.color = '#cc2255';

  const edgeNorm = Math.tanh(netEdge / 200) * 0.5 + 0.5;
  $('lift-marker').style.left = `${edgeNorm * 100}%`;

  $('sig-vac-above').textContent = fmt(agg.vacuumAbove, 1);
  $('sig-vac-above').style.color = signColour(agg.vacuumAbove, 0.01);
  $('sig-vac-below').textContent = fmt(agg.vacuumBelow, 1);
  $('sig-vac-below').style.color = signColour(-agg.vacuumBelow, 0.01);
  $('sig-press-above').textContent = fmt(agg.pressureAbove, 1);
  $('sig-press-above').style.color = signColour(-agg.pressureAbove, 0.01);
  $('sig-press-below').textContent = fmt(agg.pressureBelow, 1);
  $('sig-press-below').style.color = signColour(agg.pressureBelow, 0.01);

  const stateEl = $('sig-vp-state');
  stateEl.textContent = state;
  stateEl.style.color = stateTextColor;
  stateEl.style.background = stateColor;
  $('sig-conviction').textContent = `${(conviction * 100).toFixed(0)}%`;
  $('sig-conviction').style.color = conviction >= 0.6 ? '#22cc66' : conviction >= 0.35 ? '#ccaa22' : '#888';
  $('sig-state-note').textContent = stateNote;

  const longEl = $('sig-long-action');
  longEl.textContent = longAction;
  longEl.style.color = longColor;
  longEl.style.background = longBg;
  $('sig-long-note').textContent = longNote;

  const shortEl = $('sig-short-action');
  shortEl.textContent = shortAction;
  shortEl.style.color = shortColor;
  shortEl.style.background = shortBg;
  $('sig-short-note').textContent = shortNote;

  $('sig-risk-flag').textContent = riskFlag;
  $('sig-risk-flag').style.color = riskColor;

  $('sig-depth').textContent = fmt(restDepthTilt, 1);
  $('sig-depth').style.color = signColour(restDepthTilt, 0.005);
  $('sig-rest-depth').textContent = fmt(restDepthTotal, 1);
  $('sig-rest-depth').style.color = '#aaa';
  $('sig-force-total').textContent = fmt(forceTotal, 1);
  $('sig-force-total').style.color = '#ddd';
  $('sig-model-status').textContent = 'NOT ENABLED';
  $('sig-model-status').style.color = '#888';

  $('regime-label').textContent = state;
  $('regime-label').style.color = stateTextColor;
  $('regime-lift-val').textContent = fmt(netEdge, 1);
  $('regime-lift-val').style.color = signColour(netEdge, 0.01);
  $('bot-bull-edge').textContent = fmt(bullEdge, 1);
  $('bot-bear-edge').textContent = fmt(bearEdge, 1);
  $('bot-conviction').textContent = `${(conviction * 100).toFixed(0)}%`;
  $('bot-risk-flag').textContent = riskFlag;
  $('bot-risk-flag').style.color = riskColor;
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
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, cw, ch);

  if (!userPanned) {
    resetViewport();
  }

  const srcW = visibleCols();
  const srcH = visibleRows();
  const dataWidth = projectionDataWidth(cw);

  if (!hmapOffscreen) {
    hmapOffscreen = document.createElement('canvas');
    hmapOffscreen.width = HMAP_HISTORY;
    hmapOffscreen.height = HMAP_LEVELS;
  }
  const offCtx = hmapOffscreen.getContext('2d')!;
  const imgData = new ImageData(hmapData, HMAP_HISTORY, HMAP_LEVELS);
  offCtx.putImageData(imgData, 0, 0);

  ctx.imageSmoothingEnabled = false;

  // When zoomed out (vpX < 0 or vpY < 0), the source rect extends
  // beyond the pixel buffer. drawImage clips negative source coords,
  // so we compute the visible overlap and position the dest accordingly.
  const sx = Math.max(0, vpX);
  const sy = Math.max(0, vpY);
  const sx2 = Math.min(HMAP_HISTORY, vpX + srcW);
  const sy2 = Math.min(HMAP_LEVELS, vpY + srcH);
  const sw = Math.max(0, sx2 - sx);
  const sh = Math.max(0, sy2 - sy);

  if (sw > 0 && sh > 0) {
    // Map source pixel range to destination canvas range
    const dx = ((sx - vpX) / srcW) * dataWidth;
    const dy = ((sy - vpY) / srcH) * ch;
    const dw = (sw / srcW) * dataWidth;
    const dh = (sh / srcH) * ch;
    ctx.drawImage(hmapOffscreen, sx, sy, sw, sh, dx, dy, dw, dh);
  }

  if (!anchorInitialized) return;

  const rowToY = (row: number): number => ((row - vpY) / srcH) * ch;
  const colToX = (col: number): number => ((col - vpX) / srcW) * dataWidth;

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
    ctx.lineTo(dataWidth, y);
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
    if (row === null) {
      started = false;
      continue;
    }
    const x = colToX(i + 0.5);
    const y = rowToY(row);
    // Clip to canvas bounds (not buffer bounds -- zoom-out may extend past)
    if (x < -50 || x > dataWidth + 50 || y < -50 || y > ch + 50) { started = false; continue; }
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
  if (Math.abs(zoomX - 1.0) > 0.01 || Math.abs(zoomY - 1.0) > 0.01) {
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

  // Depth bars from event-mode dense grid
  if (isEventMode && currentGrid.size > 0 && currentSpotDollars > 0) {
    // Event mode: render rest_depth bars colored by net force (pressure - vacuum)
    const maxP = runningMaxPressure || 1;
    for (const [k, b] of currentGrid) {
      const absPrice = currentSpotDollars + k * bucket;
      const row = Math.round(priceToRow(absPrice));
      if (row < vpY - 1 || row > vpY + srcH + 1) continue;
      const y = (row - vpY) * rowH;
      const barW = Math.min(barMax, (Math.log1p(b.rest_depth) / Math.log1p(maxD)) * barMax);
      const netForce = b.pressure_variant - b.vacuum_variant;
      const forceT = Math.tanh(netForce / maxP);

      if (forceT >= 0) {
        // Building (pressure > vacuum) → cyan-green
        ctx.fillStyle = `rgba(30, ${140 + Math.round(80 * forceT)}, ${120 + Math.round(50 * forceT)}, 0.7)`;
      } else {
        // Draining (vacuum > pressure) → red-magenta
        ctx.fillStyle = `rgba(${140 + Math.round(80 * (-forceT))}, 30, ${60 + Math.round(40 * (-forceT))}, 0.7)`;
      }

      // k > 0 = above spot (ask side), k < 0 = below spot (bid side)
      if (k < 0) {
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

// -------------------------------------------- Runtime config application

/**
 * Apply a runtime config received from the server.
 * Updates the metadata display and hides the legacy fallback warning.
 */
function applyRuntimeConfig(cfg: RuntimeConfig): void {
  if (cfg.grid_max_ticks !== MAX_REL_TICKS) {
    streamContractError(
      'runtime_config',
      `grid_max_ticks mismatch: frontend supports ${MAX_REL_TICKS}, backend sent ${cfg.grid_max_ticks}`,
    );
  }

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
    start_time: params.get('start_time') || undefined,
    throttle_ms: params.get('throttle_ms') || undefined,
  };
}

// ---------------------------------------------------------------- WebSocket

/** Cached stream params so reconnect preserves original params. */
let streamParams: StreamParams | null = null;
let wsClient: WebSocket | null = null;
let reconnectTimerId: number | null = null;
let reconnectEnabled = true;
let streamPaused = false;

function updateStreamControlUi(): void {
  $streamState.textContent = streamPaused ? 'PAUSED' : 'RUNNING';
  $streamState.style.color = streamPaused ? '#ccaa22' : '#00ffaa';
  $ctrlPause.style.opacity = streamPaused ? '0.55' : '1';
  $ctrlPlay.style.opacity = streamPaused ? '1' : '0.55';
}

function clearReconnectTimer(): void {
  if (reconnectTimerId !== null) {
    window.clearTimeout(reconnectTimerId);
    reconnectTimerId = null;
  }
}

function scheduleReconnect(): void {
  if (!reconnectEnabled || streamPaused) return;
  clearReconnectTimer();
  reconnectTimerId = window.setTimeout(() => {
    reconnectTimerId = null;
    connectWS();
  }, 3000);
}

function resetStreamState(): void {
  windowCount = 0;
  anchorPriceDollars = 0;
  anchorInitialized = false;
  currentSpotDollars = 0;
  runningMaxDepth = 100;
  configReceived = false;
  isEventMode = false;
  currentGrid = new Map();
  lastRenderedEventIdByRow.clear();
  runningMaxPressure = 10;

  for (let i = 0; i < hmapData.length; i += 4) {
    hmapData[i] = 10;
    hmapData[i + 1] = 10;
    hmapData[i + 2] = 15;
    hmapData[i + 3] = 255;
  }
  for (let i = 0; i < spotTrail.length; i++) spotTrail[i] = null;

  $spotVal.textContent = '--';
  $tsVal.textContent = '--:--:--';
  $winVal.textContent = '0';
  $winIdVal.textContent = '--';
  $('regime-label').textContent = 'CHOP';
  $('regime-label').style.color = '#ccaa22';
  $('regime-lift-val').textContent = '0.0';
  $('regime-lift-val').style.color = '#888';
  $('sig-vp-state').textContent = 'CHOP';
  $('sig-vp-state').style.color = '#ccaa22';
  $('sig-vp-state').style.background = '#2f3138';
  $('sig-state-note').textContent = 'Waiting for directional imbalance.';
}

function closeSocketForControl(reason: string): void {
  clearReconnectTimer();
  if (
    wsClient &&
    (wsClient.readyState === WebSocket.OPEN || wsClient.readyState === WebSocket.CONNECTING)
  ) {
    wsClient.close(1000, reason);
  }
}

function pauseStream(): void {
  if (streamPaused) return;
  streamPaused = true;
  reconnectEnabled = false;
  closeSocketForControl('user-pause');
  updateStreamControlUi();
}

function playStream(): void {
  if (!streamPaused && wsClient && wsClient.readyState === WebSocket.OPEN) return;
  streamPaused = false;
  reconnectEnabled = true;
  updateStreamControlUi();
  connectWS();
}

function restartStream(): void {
  streamPaused = false;
  reconnectEnabled = false;
  resetStreamState();
  closeSocketForControl('user-restart');
  wsClient = null;
  reconnectEnabled = true;
  updateStreamControlUi();
  connectWS();
}

function setupStreamControls(): void {
  $ctrlPause.addEventListener('click', () => pauseStream());
  $ctrlPlay.addEventListener('click', () => playStream());
  $ctrlRestart.addEventListener('click', () => restartStream());
  updateStreamControlUi();
}

function connectWS(): void {
  if (
    wsClient &&
    (wsClient.readyState === WebSocket.OPEN || wsClient.readyState === WebSocket.CONNECTING)
  ) {
    return;
  }
  if (streamPaused) {
    return;
  }
  if (!streamParams) {
    streamParams = parseStreamParams();
  }
  const { product_type, symbol, dt, speed, start_time, throttle_ms } = streamParams;

  const urlBase =
    `ws://localhost:${WS_PORT}/v1/vacuum-pressure/stream` +
    `?product_type=${encodeURIComponent(product_type)}` +
    `&symbol=${encodeURIComponent(symbol)}` +
    `&dt=${encodeURIComponent(dt)}` +
    `&speed=${encodeURIComponent(speed)}`;
  const wsParams = new URLSearchParams();
  if (start_time) wsParams.set('start_time', start_time);
  if (throttle_ms) wsParams.set('throttle_ms', throttle_ms);
  const url = wsParams.toString() ? `${urlBase}&${wsParams.toString()}` : urlBase;

  console.log(`[VP] Connecting to: ${url}`);
  const ws = new WebSocket(url);
  wsClient = ws;

  // Message queue pattern
  let expectingGridBinary = false;
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
            applyRuntimeConfig({
              product_type: requireStringField('runtime_config', msg, 'product_type'),
              symbol: requireStringField('runtime_config', msg, 'symbol'),
              symbol_root: optionalString(msg.symbol_root) ?? '',
              price_scale: requireNumberField('runtime_config', msg, 'price_scale'),
              tick_size: requireNumberField('runtime_config', msg, 'tick_size'),
              bucket_size_dollars: requireNumberField('runtime_config', msg, 'bucket_size_dollars'),
              rel_tick_size: requireNumberField('runtime_config', msg, 'rel_tick_size'),
              grid_max_ticks: requireNumberField('runtime_config', msg, 'grid_max_ticks'),
              contract_multiplier: requireNumberField('runtime_config', msg, 'contract_multiplier'),
              qty_unit: optionalString(msg.qty_unit) ?? 'shares',
              price_decimals: requireNumberField('runtime_config', msg, 'price_decimals'),
              config_version: optionalString(msg.config_version) ?? '',
            });
            const cfgMode = optionalString(msg.mode);
            const cfgStage = optionalString(msg.deployment_stage);
            const cfgFormat = optionalString(msg.stream_format);
            if (
              cfgMode === 'event' ||
              cfgMode === 'live' ||
              cfgMode === 'pre_prod' ||
              cfgFormat === 'dense_grid'
            ) {
              isEventMode = true;
              console.log(
                `[VP] Dense-grid stream detected, stage=${String(cfgStage ?? cfgMode ?? 'unknown')}, ` +
                `grid_rows=${msg.grid_rows}, ` +
                `grid_schema_fields=${JSON.stringify(msg.grid_schema_fields)}`
              );
            } else {
              streamContractError(
                'runtime_config',
                `unexpected stream format mode=${String(cfgMode)} format=${String(cfgFormat)}`
              );
            }
          } else if (msg.type === 'grid_update') {
            if (!configReceived) {
              streamContractError('grid_update', 'received before runtime_config');
            }
            isEventMode = true;
            const tsNs = requireBigIntField('grid_update', msg, 'ts_ns');
            const eventId = requireNumberField('grid_update', msg, 'event_id');
            const midPrice = requireNumberField('grid_update', msg, 'mid_price');
            requireBigIntField('grid_update', msg, 'spot_ref_price_int');
            requireBigIntField('grid_update', msg, 'best_bid_price_int');
            requireBigIntField('grid_update', msg, 'best_ask_price_int');
            requireBooleanField('grid_update', msg, 'book_valid');

            currentSpotDollars = midPrice;
            windowCount++;

            $spotVal.textContent = `$${midPrice.toFixed(priceDecimals())}`;
            $tsVal.textContent = formatTs(tsNs);
            $winVal.textContent = String(windowCount);
            $winIdVal.textContent = String(eventId);

            expectingGridBinary = true;
          } else if (msg.type === 'error') {
            const errorMessage = optionalString(msg.message) ?? 'unknown stream error';
            streamContractError('server', errorMessage);
          }
        } else if (event.data instanceof Blob) {
          if (!expectingGridBinary) {
            streamContractError('grid', 'binary frame arrived without grid_update header');
          }
          expectingGridBinary = false;

          const buffer = await event.data.arrayBuffer();
          const table = tableFromIPC(buffer);
          const gridMap = new Map<number, GridBucketRow>();
          for (let i = 0; i < table.numRows; i++) {
            const row = table.get(i);
            if (!row) continue;
            const j = row.toJSON() as Record<string, unknown>;
            const k = requireNumberField('grid', j, 'k');
            gridMap.set(k, {
              k,
              pressure_variant: requireNumberField('grid', j, 'pressure_variant'),
              vacuum_variant: requireNumberField('grid', j, 'vacuum_variant'),
              add_mass: requireNumberField('grid', j, 'add_mass'),
              pull_mass: requireNumberField('grid', j, 'pull_mass'),
              fill_mass: requireNumberField('grid', j, 'fill_mass'),
              rest_depth: requireNumberField('grid', j, 'rest_depth'),
              v_add: requireNumberField('grid', j, 'v_add'),
              v_pull: requireNumberField('grid', j, 'v_pull'),
              v_fill: requireNumberField('grid', j, 'v_fill'),
              v_rest_depth: requireNumberField('grid', j, 'v_rest_depth'),
              a_add: requireNumberField('grid', j, 'a_add'),
              a_pull: requireNumberField('grid', j, 'a_pull'),
              a_fill: requireNumberField('grid', j, 'a_fill'),
              a_rest_depth: requireNumberField('grid', j, 'a_rest_depth'),
              j_add: requireNumberField('grid', j, 'j_add'),
              j_pull: requireNumberField('grid', j, 'j_pull'),
              j_fill: requireNumberField('grid', j, 'j_fill'),
              j_rest_depth: requireNumberField('grid', j, 'j_rest_depth'),
              last_event_id: requireNumberField('grid', j, 'last_event_id'),
            });
          }
          currentGrid = gridMap;
          pushHeatmapColumnFromGrid(gridMap, currentSpotDollars);
          updateSignalPanelFromGrid(gridMap);
        }
      } catch (e) {
        console.error('[VP] Message processing error:', e);
      }
    }

    isProcessing = false;
  };

  ws.onopen = () => {
    console.log('[VP] WebSocket connected');
    updateStreamControlUi();
  };
  ws.onmessage = (event: MessageEvent) => {
    messageQueue.push(event);
    processQueue();
  };
  ws.onerror = (err) => console.error('[VP] WebSocket error:', err);
  ws.onclose = () => {
    wsClient = null;
    if (reconnectEnabled && !streamPaused) {
      console.log('[VP] WebSocket closed, reconnecting in 3s...');
      scheduleReconnect();
    } else {
      console.log('[VP] WebSocket closed');
    }
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
    const drawWidth = projectionDataWidth(rect.width);
    const mx = clamp(e.clientX - rect.left, 0, drawWidth);
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
      applyZoom(factor, factor, mx, my, drawWidth, rect.height);
    } else if (e.shiftKey) {
      applyZoom(factor, 1, mx, my, drawWidth, rect.height);
    } else {
      applyZoom(1, factor, mx, my, drawWidth, rect.height);
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
    const drawWidth = projectionDataWidth(rect.width);
    const dxPx = e.clientX - panStartX;
    const dyPx = e.clientY - panStartY;
    const srcW = visibleCols();
    const srcH = visibleRows();

    vpX = panStartVpX - (dxPx / drawWidth) * srcW;
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

setupStreamControls();
connectWS();
startRenderLoop();
