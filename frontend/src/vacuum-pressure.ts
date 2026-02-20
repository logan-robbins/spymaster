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
 *   flow_score > 0 -> cyan-green   (pressure side)
 *   flow_score < 0 -> red-magenta  (vacuum side)
 *   flow_score ~ 0 -> near-black   (neutral side)
 *
 * Zoom & Pan:
 *   Mouse wheel on heatmap zooms (Shift=horiz, plain=vert, Ctrl/Cmd=both).
 *   Click-drag pans.  Double-click resets to auto-follow.
 *
 * Runtime config:
 *   All instrument-specific constants (bucket size, tick size, decimals,
 *   multiplier) are received from the server via a JSON control message
 *   with type "runtime_config" before the first data batch.  No hardcoded
 *   instrument assumptions remain in this module.
 */

import { tableFromIPC } from 'apache-arrow';
import { ExperimentEngine } from './experiment-engine';
import type { CompositeSignal } from './experiment-engine';

// ------------------------------------------------------------------ Types

interface RuntimeConfig {
  product_type: string;
  symbol: string;
  symbol_root: string;
  price_scale: number;
  tick_size: number;
  bucket_size_dollars: number;
  rel_tick_size: number;
  grid_radius_ticks: number;
  cell_width_ms: number;
  flow_windows: number[];
  flow_rollup_weights: number[];
  flow_derivative_weights: number[];
  flow_tanh_scale: number;
  flow_neutral_threshold: number;
  flow_zscore_window_bins: number;
  flow_zscore_min_periods: number;
  projection_horizons_bins: number[];
  projection_horizons_ms: number[];
  contract_multiplier: number;
  qty_unit: string;
  price_decimals: number;
  config_version: string;
}

interface RuntimeModelConfig {
  name: string;
  enabled: boolean;
}

interface RuntimeModelUpdate {
  name: string;
  score: number;
  ready: boolean;
  sampleCount: number;
  base: number;
  d1: number;
  d2: number;
  d3: number;
  z1: number;
  z2: number;
  z3: number;
  bullIntensity: number;
  bearIntensity: number;
  mixedIntensity: number;
  dominantState5Code: number;
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
  composite: number;
  composite_d1: number;
  composite_d2: number;
  composite_d3: number;
  flow_score: number;
  flow_state_code: number;
  last_event_id: number;
}

/** Parsed and validated URL query parameters. */
interface StreamParams {
  product_type: string;
  symbol: string;
  dt: string;
  start_time?: string;
  serving?: string;
  projection_horizons_bins?: string;
  projection_source?: 'backend' | 'frontend';
  dev_scoring?: boolean;
  // Runtime model overrides (forwarded to WebSocket query params)
  state_model_enabled?: string;
  state_model_center_exclusion_radius?: string;
  state_model_spatial_decay_power?: string;
  state_model_zscore_window_bins?: string;
  state_model_zscore_min_periods?: string;
  state_model_tanh_scale?: string;
  state_model_d1_weight?: string;
  state_model_d2_weight?: string;
  state_model_d3_weight?: string;
  state_model_bull_pressure_weight?: string;
  state_model_bull_vacuum_weight?: string;
  state_model_bear_pressure_weight?: string;
  state_model_bear_vacuum_weight?: string;
  state_model_mixed_weight?: string;
  state_model_enable_weighted_blend?: string;
}

// --------------------------------------------------------- Layout constants

const WS_PORT = 8002;
let GRID_RADIUS_TICKS = 40;
let HMAP_LEVELS = GRID_RADIUS_TICKS * 2 + 1;
const HMAP_HISTORY = 360;                    // 6 min of 1-second columns
const DEPTH_NORM_PERCENTILE_DECAY = 0.995;
const SCROLL_MARGIN = 10;                    // rows from edge before auto-scroll

// --------------------------------------------------- Runtime config state

let runtimeConfig: RuntimeConfig | null = null;
let configReceived = false;
let runtimeModelConfig: RuntimeModelConfig | null = null;
let latestRuntimeModel: RuntimeModelUpdate | null = null;
let projectionSource: 'backend' | 'frontend' = 'backend';
let devScoringEnabled = false;

// BBO state for overlay rendering
let bestBidDollars = 0;
let bestAskDollars = 0;
let currentBinEndNs: bigint = 0n;

// BBO trail: bid/ask row positions per heatmap column
let bidTrail: (number | null)[] = [];
let askTrail: (number | null)[] = [];

// Time axis: timestamp (ns) for each heatmap column
let columnTimestamps: (bigint | null)[] = [];

// Experiment engine for projection bands
let experimentEngine: ExperimentEngine | null = null;
let currentCompositeSignal: CompositeSignal = {
  composite: 0, pfp: 0, ads: 0, svac: 0, warmupFraction: 0,
};
// Composite signal trail: stores composite value per heatmap column for persistent bands
let compositeTrail: (number | null)[] = [];
// Warmup fraction trail: stores warmup state per column
let warmupTrail: (number | null)[] = [];

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
let spotTrail: (number | null)[] = [];

// Heatmap pixel buffer (RGBA, HMAP_HISTORY x HMAP_LEVELS)
let hmapData = new Uint8ClampedArray(0);

let runningMaxDepth = 100; // adaptive normalisation

const streamContractErrors = new Set<string>();

// --------------------------------------------------- Event-mode state
let isEventMode = false;
let currentGrid: Map<number, GridBucketRow> = new Map();
/** Per-bucket last_event_id tracker for persistence (keyed by heatmap row). */
const lastRenderedEventIdByRow: Map<number, number> = new Map();
/** Running max for |flow_score| adaptive normalization. */
let runningMaxSpectrum = 10;

function resetHeatmapBuffers(gridRadiusTicks: number): void {
  GRID_RADIUS_TICKS = gridRadiusTicks;
  HMAP_LEVELS = GRID_RADIUS_TICKS * 2 + 1;
  MAX_ZOOM_Y = HMAP_LEVELS / 10;
  anchorPriceDollars = 0;
  anchorInitialized = false;
  currentSpotDollars = 0;
  windowCount = 0;
  spotTrail = new Array(HMAP_HISTORY).fill(null);
  bidTrail = new Array(HMAP_HISTORY).fill(null);
  askTrail = new Array(HMAP_HISTORY).fill(null);
  columnTimestamps = new Array(HMAP_HISTORY).fill(null);
  compositeTrail = new Array(HMAP_HISTORY).fill(null);
  warmupTrail = new Array(HMAP_HISTORY).fill(null);
  hmapData = new Uint8ClampedArray(HMAP_HISTORY * HMAP_LEVELS * 4);
  for (let i = 0; i < hmapData.length; i += 4) {
    hmapData[i] = 10;
    hmapData[i + 1] = 10;
    hmapData[i + 2] = 15;
    hmapData[i + 3] = 255;
  }
  lastRenderedEventIdByRow.clear();
}

// --------------------------------------------------------- Viewport / Zoom

let zoomX = 1.0;
let zoomY = 1.0;
let vpX = 0;
let vpY = 0;
let userPanned = false;

const MIN_ZOOM_X = 1.0;       // no horizontal zoom-out (buffer only holds 360 cols)
const MIN_ZOOM_Y = 0.15;      // vertical zoom-out: see ~540 price levels compressed
const MAX_ZOOM_X = HMAP_HISTORY / 30;  // 12× zoom in
let MAX_ZOOM_Y = HMAP_LEVELS / 10;   // ~8× zoom in
const ZOOM_STEP = 1.08;

resetHeatmapBuffers(GRID_RADIUS_TICKS);

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
  return GRID_RADIUS_TICKS - (priceDollars - anchorPriceDollars) / bucketDollars();
}

/** Map grid row to dollar price. */
function rowToPrice(row: number): number {
  return anchorPriceDollars + (GRID_RADIUS_TICKS - row) * bucketDollars();
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

/** Fraction of heatmap canvas width used for historical data (0..1).
 *  The remaining right portion is reserved for future projection overlays. */
const PROJECTION_ZONE_FRACTION = 0.15;

function projectionDataWidth(totalWidth: number): number {
  return Math.max(32, Math.round(totalWidth * (1 - PROJECTION_ZONE_FRACTION)));
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

function requireNumberArrayField(
  surface: string,
  obj: Record<string, unknown>,
  field: string,
): number[] {
  const raw = requireField(surface, obj, field);
  if (!Array.isArray(raw)) {
    streamContractError(surface, `field "${field}" is not an array`);
  }
  const out: number[] = [];
  for (const item of raw) {
    const parsed = Number(item);
    if (!Number.isFinite(parsed)) {
      streamContractError(surface, `field "${field}" contains non-numeric value`);
    }
    out.push(parsed);
  }
  return out;
}

function requirePositiveIntArrayField(
  surface: string,
  obj: Record<string, unknown>,
  field: string,
): number[] {
  const out = requireNumberArrayField(surface, obj, field);
  if (out.length === 0) {
    streamContractError(surface, `field "${field}" must contain at least one value`);
  }
  const ints: number[] = [];
  for (const value of out) {
    const rounded = Math.round(value);
    if (!Number.isFinite(value) || rounded !== value || rounded <= 0) {
      streamContractError(surface, `field "${field}" must contain positive integers`);
    }
    ints.push(rounded);
  }
  return ints;
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

function optionalNumber(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function optionalBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

/** Heatmap cell colour: single-channel green pressure gradient.
 *
 *  Encodes ONE signal: how much pressure (liquidity building) exists
 *  at this bucket right now.
 *
 *  Pressure (flow_score > 0): dark emerald → bright green
 *  Vacuum/neutral (score ≤ 0):   black (absence of pressure IS the signal)
 *  rest_depth gates a faint grey floor so resting-but-neutral depth is
 *  distinguishable from truly empty levels.
 */
function heatmapRGB(
  depth: number, spectrumScore: number, maxDepth: number,
): [number, number, number] {
  const depthN = Math.min(1.0, Math.log1p(depth) / Math.log1p(maxDepth));
  const scoreN = Math.max(0, spectrumScore);
  const pressureT = Math.pow(scoreN, 0.7);

  if (pressureT > 0.05) {
    // Pressure path: green intensity driven primarily by flow_score.
    // Rest depth adds only a subtle base glow so cells with strong
    // flow_score but low depth aren't invisible.
    const lum = 0.10 + depthN * 0.15 + pressureT * 0.75;
    return [
      Math.round((0.02 + pressureT * 0.06) * 255 * lum),
      Math.round((0.08 + pressureT * 0.92) * 255 * lum),
      Math.round((0.03 + pressureT * 0.12) * 255 * lum),
    ];
  } else {
    // Neutral / vacuum path: faint blue-grey proportional to depth.
    const baseLum = depthN * 0.08;
    return [
      Math.round(0.12 * baseLum * 255),
      Math.round(0.12 * baseLum * 255),
      Math.round(0.18 * baseLum * 255),
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
    for (let i = 0; i < bidTrail.length; i++) bidTrail[i] = null;
    for (let i = 0; i < askTrail.length; i++) askTrail[i] = null;
    for (let i = 0; i < columnTimestamps.length; i++) columnTimestamps[i] = null;
    for (let i = 0; i < compositeTrail.length; i++) compositeTrail[i] = null;
    for (let i = 0; i < warmupTrail.length; i++) warmupTrail[i] = null;
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
    if (bidTrail[i] !== null) {
      bidTrail[i] = bidTrail[i]! + shiftRows;
    }
    if (askTrail[i] !== null) {
      askTrail[i] = askTrail[i]! + shiftRows;
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

  // Advance BBO trails
  bidTrail.shift();
  askTrail.shift();
  if (bestBidDollars > 0 && anchorInitialized) {
    bidTrail.push(priceToRow(bestBidDollars));
  } else {
    bidTrail.push(null);
  }
  if (bestAskDollars > 0 && anchorInitialized) {
    askTrail.push(priceToRow(bestAskDollars));
  } else {
    askTrail.push(null);
  }

  // Advance time axis timestamps
  columnTimestamps.shift();
  columnTimestamps.push(currentBinEndNs > 0n ? currentBinEndNs : null);

  // Advance composite signal trail (populated after engine.update below)
  compositeTrail.shift();
  compositeTrail.push(null); // placeholder; filled after engine update
  warmupTrail.shift();
  warmupTrail.push(null);

  // Map grid buckets to heatmap rows and track adaptive normalization.
  let maxSpectrumAbs = 0;
  let maxRestD = 0;
  for (const bucket_row of grid.values()) {
    const spectrumAbs = Math.abs(bucket_row.flow_score);
    if (spectrumAbs > maxSpectrumAbs) maxSpectrumAbs = spectrumAbs;
    if (bucket_row.rest_depth > maxRestD) maxRestD = bucket_row.rest_depth;
  }
  runningMaxSpectrum = Math.max(
    runningMaxSpectrum * DEPTH_NORM_PERCENTILE_DECAY,
    maxSpectrumAbs,
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

    const [r, g, b] = heatmapRGB(bucketData.rest_depth, bucketData.flow_score, runningMaxDepth);
    const idx = (row * w + (w - 1)) * 4;
    d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255;
  }

  // Update experiment engine for projection bands
  let localComposite: CompositeSignal | null = null;
  if (experimentEngine) {
    localComposite = experimentEngine.update(grid);
    currentCompositeSignal = localComposite;
  }

  if (projectionSource === 'backend' && latestRuntimeModel !== null) {
    compositeTrail[HMAP_HISTORY - 1] = latestRuntimeModel.score;
    warmupTrail[HMAP_HISTORY - 1] = latestRuntimeModel.ready ? 1 : 0;
  } else if (localComposite !== null) {
    compositeTrail[HMAP_HISTORY - 1] = localComposite.composite;
    warmupTrail[HMAP_HISTORY - 1] = localComposite.warmupFraction;
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

function updateRuntimeModelPanel(): void {
  const statusEl = $('sig-model-status');
  const scoreEl = $('sig-model-score');
  const detailEl = $('sig-model-detail');

  if (!runtimeModelConfig || !runtimeModelConfig.enabled) {
    statusEl.textContent = 'DISABLED';
    statusEl.style.color = '#888';
    scoreEl.textContent = '0.0';
    scoreEl.style.color = '#888';
    detailEl.textContent = 'runtime model disabled by config';
    detailEl.style.color = '#777';
    return;
  }

  if (!latestRuntimeModel) {
    statusEl.textContent = `${runtimeModelConfig.name.toUpperCase()} WAIT`;
    statusEl.style.color = '#ccaa22';
    scoreEl.textContent = '0.0';
    scoreEl.style.color = '#888';
    detailEl.textContent = 'awaiting runtime model ticks';
    detailEl.style.color = '#777';
    return;
  }

  statusEl.textContent = latestRuntimeModel.ready
    ? `${latestRuntimeModel.name.toUpperCase()} LIVE`
    : `${latestRuntimeModel.name.toUpperCase()} WARM`;
  statusEl.style.color = latestRuntimeModel.ready ? '#22cc66' : '#ccaa22';

  scoreEl.textContent = fmt(latestRuntimeModel.score, 3);
  scoreEl.style.color = signColour(latestRuntimeModel.score, 3.0);

  const drift = currentCompositeSignal.composite - latestRuntimeModel.score;
  detailEl.textContent =
    `bull=${latestRuntimeModel.bullIntensity.toFixed(3)} ` +
    `bear=${latestRuntimeModel.bearIntensity.toFixed(3)} ` +
    `d1/z1=${latestRuntimeModel.d1.toFixed(3)}/${latestRuntimeModel.z1.toFixed(3)} ` +
    `drift(frontend-backend)=${drift.toFixed(3)}`;
  detailEl.style.color = '#888';
}

// ---------------------------------------------------------------- Rendering

interface ProjectionHorizon {
  bins: number;
  ms: number;
  label: string;
  spreadTicks: number;
  alpha: number;
}

const DEFAULT_PROJECTION_HORIZON_BINS = [1, 2, 3, 4];
const SPREAD_ANCHORS = [
  { bins: 1, value: 6 },
  { bins: 2, value: 10 },
  { bins: 4, value: 16 },
  { bins: 8, value: 24 },
];
const ALPHA_ANCHORS = [
  { bins: 1, value: 0.50 },
  { bins: 2, value: 0.40 },
  { bins: 4, value: 0.30 },
  { bins: 8, value: 0.20 },
];

let projectionHorizons: ProjectionHorizon[] = buildProjectionHorizons(
  DEFAULT_PROJECTION_HORIZON_BINS,
  100,
);

/** Signal amplification: composite signal is typically [-0.3, +0.3],
 *  scale up to make directional shifts visually obvious. */
const COMPOSITE_SCALE = 8.0;

/** Band half-width in rows for the projected envelope. */
const BAND_HALF_WIDTH_ROWS = 4;

function horizonLabel(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms % 1000 === 0) return `${ms / 1000}s`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function interpolateAnchor(
  x: number,
  anchors: { bins: number; value: number }[],
): number {
  if (anchors.length === 0) return 0;
  if (x <= anchors[0].bins) return anchors[0].value;
  if (x >= anchors[anchors.length - 1].bins) return anchors[anchors.length - 1].value;

  for (let i = 1; i < anchors.length; i++) {
    const left = anchors[i - 1];
    const right = anchors[i];
    if (x <= right.bins) {
      const t = (x - left.bins) / (right.bins - left.bins);
      return left.value + (right.value - left.value) * t;
    }
  }
  return anchors[anchors.length - 1].value;
}

function buildProjectionHorizons(
  horizonBins: number[],
  cellWidthMs: number,
): ProjectionHorizon[] {
  const sorted = Array.from(new Set(horizonBins))
    .map(v => Math.round(v))
    .filter(v => v > 0)
    .sort((a, b) => a - b);

  return sorted.map(bins => ({
    bins,
    ms: bins * cellWidthMs,
    label: horizonLabel(bins * cellWidthMs),
    spreadTicks: Math.max(2, Math.round(interpolateAnchor(bins, SPREAD_ANCHORS))),
    alpha: clamp(interpolateAnchor(bins, ALPHA_ANCHORS), 0.10, 0.60),
  }));
}

function deriveProjectionHorizonMs(
  horizonBins: number[],
  cellWidthMs: number,
): number[] {
  return horizonBins.map(binCount => binCount * cellWidthMs);
}

function validateProjectionHorizonContract(cfg: RuntimeConfig): void {
  const expected = deriveProjectionHorizonMs(cfg.projection_horizons_bins, cfg.cell_width_ms);
  if (cfg.projection_horizons_ms.length !== expected.length) {
    streamContractError(
      'runtime_config',
      'projection_horizons_ms length must match projection_horizons_bins length',
    );
  }
  for (let i = 0; i < expected.length; i++) {
    if (Math.round(cfg.projection_horizons_ms[i]) !== expected[i]) {
      streamContractError(
        'runtime_config',
        `projection_horizons_ms[${i}] must equal projection_horizons_bins[${i}] * cell_width_ms`,
      );
    }
  }
}

function normalizeRuntimeProjectionHorizonMs(cfg: RuntimeConfig): number[] {
  validateProjectionHorizonContract(cfg);
  return deriveProjectionHorizonMs(cfg.projection_horizons_bins, cfg.cell_width_ms);
}

function sampleTrailInterpolated(
  trail: (number | null)[],
  colPosition: number,
): number | null {
  if (colPosition < 0 || colPosition > trail.length - 1) return null;
  const left = Math.floor(colPosition);
  const right = Math.ceil(colPosition);

  const lv = trail[left];
  if (lv === null) return null;
  if (left === right) return lv;

  const rv = trail[right];
  if (rv === null) return null;

  const t = colPosition - left;
  return lv + (rv - lv) * t;
}

/**
 * Render projection bands as persistent scrolling overlays.
 *
 * Two rendering modes:
 * 1. HISTORICAL: In the main heatmap area, draw faded polylines showing where
 *    past predictions pointed (band center trails at each horizon timescale).
 *    This lets the trader see if predictions were accurate.
 * 2. FORWARD: In the projection zone (right 15%), draw the current prediction
 *    as expanding bands at 4 horizon positions.
 */
function renderProjectionBands(
  ctx: CanvasRenderingContext2D,
  _signal: CompositeSignal,
  _spotRow: number,
  ch: number,
  dataWidth: number,
  cw: number,
): void {
  if (!anchorInitialized) return;

  const srcW = visibleCols();
  const srcH = visibleRows();
  const colToX = (col: number): number => ((col - vpX) / srcW) * dataWidth;
  const rowToY = (row: number): number => ((row - vpY) / srcH) * ch;

  // === PART 1: PROJECTION ZONE — forward-looking bands ===
  const zoneWidth = cw - dataWidth;
  if (zoneWidth > 0) {
    // Dark tint background for projection zone
    ctx.fillStyle = 'rgba(8, 8, 12, 0.7)';
    ctx.fillRect(dataWidth, 0, zoneWidth, ch);

    const lastSpot = spotTrail[HMAP_HISTORY - 1];
    const lastComposite = compositeTrail[HMAP_HISTORY - 1];
    const lastWarmup = warmupTrail[HMAP_HISTORY - 1];

    if (lastSpot !== null && lastComposite !== null && lastWarmup !== null && lastWarmup > 0) {
      const scaledSignal = lastComposite * COMPOSITE_SCALE;
      const nHorizons = projectionHorizons.length;
      if (nHorizons === 0) return;
      const colWidth = zoneWidth / nHorizons;

      for (let hi = 0; hi < nHorizons; hi++) {
        const horizon = projectionHorizons[hi];
        // Band center: spot shifted by amplified composite × spreadTicks
        // Bullish (positive) → lower row number (higher price)
        const bandCenterRow = lastSpot - scaledSignal * horizon.spreadTicks;
        const bandTopRow = bandCenterRow - BAND_HALF_WIDTH_ROWS;
        const bandBotRow = bandCenterRow + BAND_HALF_WIDTH_ROWS;

        const xLeft = dataWidth + colWidth * hi;
        const xRight = dataWidth + colWidth * (hi + 1);
        const yCenter = rowToY(bandCenterRow);
        const yTop = rowToY(bandTopRow);
        const yBot = rowToY(bandBotRow);

        // Filled band
        const alpha = horizon.alpha * lastWarmup;
        ctx.fillStyle = `rgba(120, 40, 180, ${alpha * 0.4})`;
        ctx.fillRect(xLeft + 1, yTop, xRight - xLeft - 2, yBot - yTop);

        // Band edges
        ctx.strokeStyle = `rgba(140, 60, 200, ${alpha * 0.7})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xLeft + 1, yTop);
        ctx.lineTo(xRight - 1, yTop);
        ctx.moveTo(xLeft + 1, yBot);
        ctx.lineTo(xRight - 1, yBot);
        ctx.stroke();

        // Center line
        ctx.strokeStyle = `rgba(180, 100, 255, ${alpha * 0.9})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(xLeft + 1, yCenter);
        ctx.lineTo(xRight - 1, yCenter);
        ctx.stroke();
      }

      // Connect spot to first horizon band center
      const firstBandCenter = lastSpot - scaledSignal * projectionHorizons[0].spreadTicks;
      const spotY = rowToY(lastSpot);
      const firstY = rowToY(firstBandCenter);
      ctx.strokeStyle = 'rgba(180, 100, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(dataWidth, spotY);
      ctx.lineTo(dataWidth + zoneWidth / projectionHorizons.length * 0.5, firstY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // === PART 2: HISTORICAL — filled band envelopes overlaid on heatmap ===
  // Draw filled bands showing where past predictions pointed at each horizon.
  // These persist and scroll with the heatmap, letting the trader compare
  // predicted direction envelope vs actual spot path (cyan line).

  const cellWidthMs = runtimeConfig ? runtimeConfig.cell_width_ms : 100;
  for (let hi = projectionHorizons.length - 1; hi >= 0; hi--) {
    const horizon = projectionHorizons[hi];
    const horizonCols = horizon.ms / cellWidthMs;

    // Build arrays of valid (x, topY, botY) points for this horizon's band
    const points: { x: number; topY: number; botY: number }[] = [];

    for (let col = 0; col < HMAP_HISTORY; col++) {
      const srcColPos = col - horizonCols;
      const pastSpot = sampleTrailInterpolated(spotTrail, srcColPos);
      const pastComposite = sampleTrailInterpolated(compositeTrail, srcColPos);
      const pastWarmup = sampleTrailInterpolated(warmupTrail, srcColPos);

      if (pastSpot === null || pastComposite === null || pastWarmup === null || pastWarmup === 0) {
        continue;
      }

      const scaledSignal = pastComposite * COMPOSITE_SCALE;
      const centerRow = pastSpot - scaledSignal * horizon.spreadTicks;
      const topRow = centerRow - BAND_HALF_WIDTH_ROWS;
      const botRow = centerRow + BAND_HALF_WIDTH_ROWS;
      const x = colToX(col + 0.5);

      if (x < -50 || x > dataWidth + 50) continue;

      points.push({ x, topY: rowToY(topRow), botY: rowToY(botRow) });
    }

    if (points.length < 2) continue;

    // Filled band envelope (forward top edge, backward bottom edge)
    const fillAlpha = horizon.alpha * 0.18;
    ctx.fillStyle = `rgba(130, 50, 200, ${fillAlpha})`;
    ctx.beginPath();
    // Top edge forward
    ctx.moveTo(points[0].x, points[0].topY);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].topY);
    }
    // Bottom edge backward
    for (let i = points.length - 1; i >= 0; i--) {
      ctx.lineTo(points[i].x, points[i].botY);
    }
    ctx.closePath();
    ctx.fill();

    // Edge lines for definition
    const edgeAlpha = horizon.alpha * 0.45;
    ctx.strokeStyle = `rgba(160, 80, 220, ${edgeAlpha})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].topY);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].topY);
    }
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].botY);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].botY);
    }
    ctx.stroke();
  }
}

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

  // Projection zone: render experiment-driven directional bands
  if (dataWidth < cw) {
    const lastSpotRow = spotTrail[HMAP_HISTORY - 1];
    renderProjectionBands(
      ctx, currentCompositeSignal,
      lastSpotRow !== null ? lastSpotRow : GRID_RADIUS_TICKS,
      ch, dataWidth, cw,
    );
  }

  ctx.strokeStyle = 'rgba(60, 60, 90, 0.2)';
  ctx.lineWidth = 0.5;
  const firstGrid = Math.ceil(botPrice / gridInterval) * gridInterval;
  for (let p = firstGrid; p <= topPrice; p += gridInterval) {
    const y = rowToY(priceToRow(p));
    if (y < -1 || y > ch + 1) continue;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(cw, y);    // gridlines span full width including projection zone
    ctx.stroke();
  }

  // "Now" separator line at the data/projection boundary
  if (dataWidth < cw) {
    ctx.strokeStyle = 'rgba(100, 100, 150, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(dataWidth, 0);
    ctx.lineTo(dataWidth, ch);
    ctx.stroke();
    ctx.setLineDash([]);
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

  // BBO bid trail (green dashed line)
  ctx.strokeStyle = 'rgba(34, 204, 102, 0.45)';
  ctx.lineWidth = 1.0;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  started = false;
  for (let i = 0; i < HMAP_HISTORY; i++) {
    const row = bidTrail[i];
    if (row === null) { started = false; continue; }
    const x = colToX(i + 0.5);
    const y = rowToY(row);
    if (x < -50 || x > dataWidth + 50 || y < -50 || y > ch + 50) { started = false; continue; }
    if (!started) { ctx.moveTo(x, y); started = true; }
    else { ctx.lineTo(x, y); }
  }
  ctx.stroke();

  // BBO ask trail (red dashed line)
  ctx.strokeStyle = 'rgba(204, 34, 85, 0.45)';
  ctx.beginPath();
  started = false;
  for (let i = 0; i < HMAP_HISTORY; i++) {
    const row = askTrail[i];
    if (row === null) { started = false; continue; }
    const x = colToX(i + 0.5);
    const y = rowToY(row);
    if (x < -50 || x > dataWidth + 50 || y < -50 || y > ch + 50) { started = false; continue; }
    if (!started) { ctx.moveTo(x, y); started = true; }
    else { ctx.lineTo(x, y); }
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Spread fill: semi-transparent region between bid and ask
  const spreadXs: number[] = [];
  const spreadBidYs: number[] = [];
  const spreadAskYs: number[] = [];
  for (let i = 0; i < HMAP_HISTORY; i++) {
    const bRow = bidTrail[i];
    const aRow = askTrail[i];
    if (bRow === null || aRow === null) continue;
    const x = colToX(i + 0.5);
    if (x < -50 || x > dataWidth + 50) continue;
    spreadXs.push(x);
    spreadBidYs.push(rowToY(bRow));
    spreadAskYs.push(rowToY(aRow));
  }
  if (spreadXs.length > 1) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.beginPath();
    ctx.moveTo(spreadXs[0], spreadAskYs[0]);
    for (let i = 1; i < spreadXs.length; i++) ctx.lineTo(spreadXs[i], spreadAskYs[i]);
    for (let i = spreadXs.length - 1; i >= 0; i--) ctx.lineTo(spreadXs[i], spreadBidYs[i]);
    ctx.closePath();
    ctx.fill();
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

  // Bid price label
  if (bestBidDollars > 0) {
    const bidRow = priceToRow(bestBidDollars);
    const bidY = ((bidRow - vpY) / srcH) * ch;
    if (bidY >= -10 && bidY <= ch + 10) {
      ctx.fillStyle = '#22cc66';
      ctx.font = '8px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`B ${bestBidDollars.toFixed(priceDecimals())}`, 7, bidY);
    }
  }

  // Ask price label
  if (bestAskDollars > 0) {
    const askRow = priceToRow(bestAskDollars);
    const askY = ((askRow - vpY) / srcH) * ch;
    if (askY >= -10 && askY <= ch + 10) {
      ctx.fillStyle = '#cc2255';
      ctx.font = '8px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`A ${bestAskDollars.toFixed(priceDecimals())}`, 7, askY);
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
    // Event mode: render rest_depth bars colored by per-cell spectrum score.
    for (const [k, b] of currentGrid) {
      const absPrice = currentSpotDollars + k * bucket;
      const row = Math.round(priceToRow(absPrice));
      if (row < vpY - 1 || row > vpY + srcH + 1) continue;
      const y = (row - vpY) * rowH;
      const barW = Math.min(barMax, (Math.log1p(b.rest_depth) / Math.log1p(maxD)) * barMax);
      const pressureT = Math.max(0, Math.min(1, b.flow_score));

      if (pressureT > 0.01) {
        // Pressure: green intensity proportional to score
        const gVal = 100 + Math.round(155 * pressureT);
        ctx.fillStyle = `rgba(20, ${gVal}, ${60 + Math.round(40 * pressureT)}, 0.7)`;
      } else {
        // Neutral/vacuum: dim grey
        ctx.fillStyle = 'rgba(40, 40, 55, 0.35)';
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

// ---------------------------------------------------------- Time axis

function renderTimeAxis(canvas: HTMLCanvasElement): void {
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

  if (!anchorInitialized || windowCount === 0) return;

  const srcW = visibleCols();
  const dataWidth = projectionDataWidth(cw);
  const colToX = (col: number): number => ((col - vpX) / srcW) * dataWidth;

  ctx.font = '8px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  // Determine label interval based on zoom-dependent pixel spacing
  const cellWidthMs = runtimeConfig ? runtimeConfig.cell_width_ms : 100;
  const pixelsPerCol = dataWidth / srcW;
  const minLabelSpacingPx = 70;
  const colsPerLabel = Math.max(1, Math.ceil(minLabelSpacingPx / pixelsPerCol));
  const colsPerSecond = 1000 / cellWidthMs;
  const rawIntervalSec = colsPerLabel / colsPerSecond;

  const niceSeconds = [1, 2, 5, 10, 15, 30, 60];
  let intervalSec = niceSeconds[niceSeconds.length - 1];
  for (const ns of niceSeconds) {
    if (ns >= rawIntervalSec) { intervalSec = ns; break; }
  }
  const intervalCols = Math.round(intervalSec * colsPerSecond);

  for (let col = 0; col < HMAP_HISTORY; col++) {
    if (intervalCols > 0 && col % intervalCols !== 0) continue;

    const ts = columnTimestamps[col];
    if (ts === null || ts === 0n) continue;

    const x = colToX(col + 0.5);
    if (x < -20 || x > cw + 20) continue;

    // Format as HH:MM:SS ET (fixed UTC-5)
    const ms = Number(ts / 1_000_000n);
    const d = new Date(ms);
    const et = new Date(d.getTime() - 5 * 3600_000);
    const hh = String(et.getUTCHours()).padStart(2, '0');
    const mm = String(et.getUTCMinutes()).padStart(2, '0');
    const ss = String(et.getUTCSeconds()).padStart(2, '0');

    // Tick mark
    ctx.strokeStyle = 'rgba(100, 100, 150, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 4);
    ctx.stroke();

    // Label
    ctx.fillStyle = '#666';
    ctx.fillText(`${hh}:${mm}:${ss}`, x, 5);
  }

  // Projection zone: "NOW" separator + horizon labels
  if (dataWidth < cw) {
    // Background tint
    ctx.fillStyle = 'rgba(15, 15, 25, 0.5)';
    ctx.fillRect(dataWidth, 0, cw - dataWidth, ch);

    // "NOW" separator
    ctx.strokeStyle = 'rgba(100, 100, 150, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(dataWidth, 0);
    ctx.lineTo(dataWidth, ch);
    ctx.stroke();
    ctx.setLineDash([]);

    // "NOW" label
    ctx.fillStyle = 'rgba(100, 100, 150, 0.4)';
    ctx.font = '7px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('NOW', dataWidth + 3, 2);

    // Horizon labels at column centers
    const zoneWidth = cw - dataWidth;
    const nHorizons = projectionHorizons.length;
    if (nHorizons === 0) return;
    const colWidth = zoneWidth / nHorizons;
    ctx.fillStyle = 'rgba(140, 100, 180, 0.6)';
    ctx.font = '7px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i < nHorizons; i++) {
      const x = dataWidth + colWidth * (i + 0.5);
      ctx.fillText(projectionHorizons[i].label, x, 14);
    }
  }
}

// -------------------------------------------- Runtime config application

/**
 * Apply a runtime config received from the server.
 * Updates the metadata display and hides the warning banner.
 */
function applyRuntimeConfig(cfg: RuntimeConfig, modelCfg: RuntimeModelConfig | null): void {
  if (cfg.grid_radius_ticks < 1) {
    streamContractError(
      'runtime_config',
      `grid_radius_ticks must be >= 1, got ${cfg.grid_radius_ticks}`,
    );
  }
  const normalizedHorizonMs = normalizeRuntimeProjectionHorizonMs(cfg);
  const normalizedCfg: RuntimeConfig = {
    ...cfg,
    projection_horizons_ms: normalizedHorizonMs,
  };
  resetHeatmapBuffers(cfg.grid_radius_ticks);
  projectionHorizons = buildProjectionHorizons(
    normalizedCfg.projection_horizons_bins,
    normalizedCfg.cell_width_ms,
  );
  if (devScoringEnabled) {
    experimentEngine = new ExperimentEngine({
      cellWidthMs: normalizedCfg.cell_width_ms,
    });
    console.warn('[VP] DEV: client-side scoring enabled -- may differ from server');
  } else {
    experimentEngine = null;
  }
  runtimeModelConfig = modelCfg;
  latestRuntimeModel = null;

  runtimeConfig = normalizedCfg;
  configReceived = true;

  // Update metadata display
  $metaProduct.textContent = normalizedCfg.product_type;
  $metaSymbol.textContent = normalizedCfg.symbol;
  $metaTick.textContent = `$${normalizedCfg.tick_size}`;
  $metaBucket.textContent = `$${normalizedCfg.rel_tick_size}`;
  $metaMult.textContent = String(normalizedCfg.contract_multiplier);

  // Hide warning banner
  $warningBanner.style.display = 'none';

  console.log(
    `[VP] Runtime config applied: product_type=${normalizedCfg.product_type} ` +
    `symbol=${normalizedCfg.symbol} bucket=$${normalizedCfg.bucket_size_dollars} ` +
    `tick=$${normalizedCfg.tick_size} decimals=${normalizedCfg.price_decimals} ` +
    `multiplier=${normalizedCfg.contract_multiplier} version=${normalizedCfg.config_version}`
  );
  updateRuntimeModelPanel();
}

// -------------------------------------------------------- URL contract

/** Parse and validate URL query parameters.  Fails fast if product_type is missing. */
function parseStreamParams(): StreamParams {
  const params = new URLSearchParams(window.location.search);
  const product_type = params.get('product_type');
  if (!product_type) {
    const msg =
      'Missing required query parameter: product_type. ' +
      'Example: ?product_type=equity_mbo&symbol=QQQ&dt=2026-02-06';
    console.error(`[VP] ${msg}`);
    // Show error in the UI
    $warningBanner.textContent = msg;
    $warningBanner.style.display = '';
    $warningBanner.style.background = '#660000';
  }

  const runtimeKeys = [
    'state_model_enabled', 'state_model_center_exclusion_radius', 'state_model_spatial_decay_power',
    'state_model_zscore_window_bins', 'state_model_zscore_min_periods', 'state_model_tanh_scale',
    'state_model_d1_weight', 'state_model_d2_weight', 'state_model_d3_weight',
    'state_model_bull_pressure_weight', 'state_model_bull_vacuum_weight',
    'state_model_bear_pressure_weight', 'state_model_bear_vacuum_weight',
    'state_model_mixed_weight', 'state_model_enable_weighted_blend',
  ] as const;

  const result: StreamParams = {
    product_type: product_type || 'equity_mbo',
    symbol: params.get('symbol') || 'QQQ',
    dt: params.get('dt') || '2026-02-06',
    start_time: params.get('start_time') || undefined,
    serving: params.get('serving') || undefined,
    projection_horizons_bins: params.get('projection_horizons_bins') || undefined,
    projection_source: params.get('projection_source') === 'frontend' ? 'frontend' : 'backend',
    dev_scoring: params.get('dev_scoring') === 'true',
  };

  for (const key of runtimeKeys) {
    const val = params.get(key);
    if (val !== null) (result as unknown as Record<string, unknown>)[key] = val;
  }

  return result;
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
  runtimeModelConfig = null;
  latestRuntimeModel = null;
  isEventMode = false;
  currentGrid = new Map();
  lastRenderedEventIdByRow.clear();
  runningMaxSpectrum = 10;
  bestBidDollars = 0;
  bestAskDollars = 0;
  currentBinEndNs = 0n;

  if (experimentEngine) experimentEngine.reset();
  currentCompositeSignal = {
    composite: 0, pfp: 0, ads: 0, svac: 0, warmupFraction: 0,
  };

  for (let i = 0; i < compositeTrail.length; i++) compositeTrail[i] = null;
  for (let i = 0; i < warmupTrail.length; i++) warmupTrail[i] = null;

  for (let i = 0; i < bidTrail.length; i++) bidTrail[i] = null;
  for (let i = 0; i < askTrail.length; i++) askTrail[i] = null;
  for (let i = 0; i < columnTimestamps.length; i++) columnTimestamps[i] = null;

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
  $('sig-model-status').textContent = 'DISABLED';
  $('sig-model-status').style.color = '#888';
  $('sig-model-score').textContent = '0.0';
  $('sig-model-score').style.color = '#888';
  $('sig-model-detail').textContent = 'runtime model not initialized';
  $('sig-model-detail').style.color = '#777';
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
  const { product_type, symbol, dt, start_time, projection_horizons_bins, projection_source, dev_scoring } = streamParams;
  projectionSource = projection_source ?? 'backend';
  devScoringEnabled = dev_scoring ?? false;

  const urlBase =
    `ws://localhost:${WS_PORT}/v1/vacuum-pressure/stream` +
    `?product_type=${encodeURIComponent(product_type)}` +
    `&symbol=${encodeURIComponent(symbol)}` +
    `&dt=${encodeURIComponent(dt)}`;
  const wsParams = new URLSearchParams();
  if (start_time) wsParams.set('start_time', start_time);
  if (streamParams.serving) wsParams.set('serving', streamParams.serving);
  if (projection_horizons_bins) wsParams.set('projection_horizons_bins', projection_horizons_bins);

  // Forward runtime-model overrides from URL to WebSocket
  const runtimeKeys = [
    'state_model_enabled', 'state_model_center_exclusion_radius', 'state_model_spatial_decay_power',
    'state_model_zscore_window_bins', 'state_model_zscore_min_periods', 'state_model_tanh_scale',
    'state_model_d1_weight', 'state_model_d2_weight', 'state_model_d3_weight',
    'state_model_bull_pressure_weight', 'state_model_bull_vacuum_weight',
    'state_model_bear_pressure_weight', 'state_model_bear_vacuum_weight',
    'state_model_mixed_weight', 'state_model_enable_weighted_blend',
  ] as const;
  for (const key of runtimeKeys) {
    const val = (streamParams as unknown as Record<string, unknown>)[key];
    if (val !== undefined && val !== null) wsParams.set(key, String(val));
  }

  const url = wsParams.toString() ? `${urlBase}&${wsParams.toString()}` : urlBase;

  console.log(`[VP] Connecting to: ${url} (projection_source=${projectionSource})`);
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
            const cellWidthMs = requireNumberField('runtime_config', msg, 'cell_width_ms');
            const projectionHorizonBins = requirePositiveIntArrayField(
              'runtime_config',
              msg,
              'projection_horizons_bins',
            );
            const projectionHorizonMs = requirePositiveIntArrayField(
              'runtime_config',
              msg,
              'projection_horizons_ms',
            );
            let modelCfg: RuntimeModelConfig | null = null;
            const runtimeModelRaw = msg.state_model;
            if (
              runtimeModelRaw !== undefined &&
              runtimeModelRaw !== null &&
              typeof runtimeModelRaw === 'object' &&
              !Array.isArray(runtimeModelRaw)
            ) {
              const runtimeModelObj = runtimeModelRaw as Record<string, unknown>;
              const modelName = optionalString(runtimeModelObj.name);
              const modelEnabled = optionalBoolean(runtimeModelObj.enabled);
              if (modelName && modelEnabled !== undefined) {
                modelCfg = { name: modelName, enabled: modelEnabled };
              }
            }

            applyRuntimeConfig({
              product_type: requireStringField('runtime_config', msg, 'product_type'),
              symbol: requireStringField('runtime_config', msg, 'symbol'),
              symbol_root: optionalString(msg.symbol_root) ?? '',
              price_scale: requireNumberField('runtime_config', msg, 'price_scale'),
              tick_size: requireNumberField('runtime_config', msg, 'tick_size'),
              bucket_size_dollars: requireNumberField('runtime_config', msg, 'bucket_size_dollars'),
              rel_tick_size: requireNumberField('runtime_config', msg, 'rel_tick_size'),
              grid_radius_ticks: requireNumberField('runtime_config', msg, 'grid_radius_ticks'),
              cell_width_ms: cellWidthMs,
              flow_windows: requireNumberArrayField('runtime_config', msg, 'flow_windows'),
              flow_rollup_weights: requireNumberArrayField('runtime_config', msg, 'flow_rollup_weights'),
              flow_derivative_weights: requireNumberArrayField('runtime_config', msg, 'flow_derivative_weights'),
              flow_tanh_scale: requireNumberField('runtime_config', msg, 'flow_tanh_scale'),
              flow_neutral_threshold: requireNumberField('runtime_config', msg, 'flow_neutral_threshold'),
              flow_zscore_window_bins: requireNumberField('runtime_config', msg, 'flow_zscore_window_bins'),
              flow_zscore_min_periods: requireNumberField('runtime_config', msg, 'flow_zscore_min_periods'),
              projection_horizons_bins: projectionHorizonBins,
              projection_horizons_ms: projectionHorizonMs,
              contract_multiplier: requireNumberField('runtime_config', msg, 'contract_multiplier'),
              qty_unit: optionalString(msg.qty_unit) ?? 'shares',
              price_decimals: requireNumberField('runtime_config', msg, 'price_decimals'),
              config_version: optionalString(msg.config_version) ?? '',
            }, modelCfg);
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
            const binSeq = requireNumberField('grid_update', msg, 'bin_seq');
            requireBigIntField('grid_update', msg, 'bin_start_ns');
            const binEndNs = requireBigIntField('grid_update', msg, 'bin_end_ns');
            requireNumberField('grid_update', msg, 'bin_event_count');
            const eventId = requireNumberField('grid_update', msg, 'event_id');
            const midPrice = requireNumberField('grid_update', msg, 'mid_price');
            const spotRefPriceInt = requireBigIntField('grid_update', msg, 'spot_ref_price_int');
            const bestBidPriceInt = requireBigIntField('grid_update', msg, 'best_bid_price_int');
            const bestAskPriceInt = requireBigIntField('grid_update', msg, 'best_ask_price_int');
            requireBooleanField('grid_update', msg, 'book_valid');
            const runtimeModelName = optionalString(msg.state_model_name);
            const runtimeModelScore = optionalNumber(msg.state_model_score);
            const runtimeModelReady = optionalBoolean(msg.state_model_ready);
            const runtimeModelSampleCount = optionalNumber(msg.state_model_sample_count);
            const runtimeModelBase = optionalNumber(msg.state_model_base);
            const runtimeModelD1 = optionalNumber(msg.state_model_d1);
            const runtimeModelD2 = optionalNumber(msg.state_model_d2);
            const runtimeModelD3 = optionalNumber(msg.state_model_d3);
            const runtimeModelZ1 = optionalNumber(msg.state_model_z1);
            const runtimeModelZ2 = optionalNumber(msg.state_model_z2);
            const runtimeModelZ3 = optionalNumber(msg.state_model_z3);
            const runtimeModelBullIntensity = optionalNumber(msg.state_model_bull_intensity);
            const runtimeModelBearIntensity = optionalNumber(msg.state_model_bear_intensity);
            const runtimeModelMixedIntensity = optionalNumber(msg.state_model_mixed_intensity);
            const runtimeModelDominantState5 = optionalNumber(msg.state_model_dominant_state5_code);
            if (
              runtimeModelName &&
              runtimeModelScore !== undefined &&
              runtimeModelReady !== undefined &&
              runtimeModelSampleCount !== undefined &&
              runtimeModelBase !== undefined &&
              runtimeModelD1 !== undefined &&
              runtimeModelD2 !== undefined &&
              runtimeModelD3 !== undefined &&
              runtimeModelZ1 !== undefined &&
              runtimeModelZ2 !== undefined &&
              runtimeModelZ3 !== undefined &&
              runtimeModelBullIntensity !== undefined &&
              runtimeModelBearIntensity !== undefined &&
              runtimeModelMixedIntensity !== undefined &&
              runtimeModelDominantState5 !== undefined
            ) {
              latestRuntimeModel = {
                name: runtimeModelName,
                score: runtimeModelScore,
                ready: runtimeModelReady,
                sampleCount: Math.round(runtimeModelSampleCount),
                base: runtimeModelBase,
                d1: runtimeModelD1,
                d2: runtimeModelD2,
                d3: runtimeModelD3,
                z1: runtimeModelZ1,
                z2: runtimeModelZ2,
                z3: runtimeModelZ3,
                bullIntensity: runtimeModelBullIntensity,
                bearIntensity: runtimeModelBearIntensity,
                mixedIntensity: runtimeModelMixedIntensity,
                dominantState5Code: Math.round(runtimeModelDominantState5),
              };
            } else {
              latestRuntimeModel = null;
            }

            // Convert integer prices to dollars using runtime price scale.
            // spot_ref_price_int is the canonical anchor for k-to-price mapping.
            const priceScale = runtimeConfig!.price_scale;
            currentSpotDollars = Number(spotRefPriceInt) * priceScale;
            bestBidDollars = Number(bestBidPriceInt) * priceScale;
            bestAskDollars = Number(bestAskPriceInt) * priceScale;
            currentBinEndNs = binEndNs;
            windowCount = Number(binSeq) + 1;

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
            const parsed: GridBucketRow = {
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
              composite: requireNumberField('grid', j, 'composite'),
              composite_d1: requireNumberField('grid', j, 'composite_d1'),
              composite_d2: requireNumberField('grid', j, 'composite_d2'),
              composite_d3: requireNumberField('grid', j, 'composite_d3'),
              flow_score: requireNumberField('grid', j, 'flow_score'),
              flow_state_code: requireNumberField('grid', j, 'flow_state_code'),
              last_event_id: requireNumberField('grid', j, 'last_event_id'),
            };
            gridMap.set(k, parsed);
          }
          currentGrid = gridMap;
          pushHeatmapColumnFromGrid(gridMap, currentSpotDollars);
          updateSignalPanelFromGrid(gridMap);
          updateRuntimeModelPanel();
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
  const timeCanvas = document.getElementById('time-axis-canvas') as HTMLCanvasElement;

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
    renderTimeAxis(timeCanvas);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// -------------------------------------------------------------------- Init

setupStreamControls();
connectWS();
startRenderLoop();
