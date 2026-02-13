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
  composite_smooth?: number;
  d1_composite: number;
  d2_composite: number;
  d3_composite: number;
  d1_smooth?: number;
  d2_smooth?: number;
  d3_smooth?: number;
  wtd_slope?: number;
  wtd_projection?: number;
  wtd_projection_500ms?: number;
  wtd_deriv_conf?: number;
  z_composite_raw?: number;
  z_composite_smooth?: number;
  confidence: number;
  strength: number;
  strength_smooth?: number;
  // Pressure and resistance
  pressure_above?: number;
  pressure_below?: number;
  resistance_above?: number;
  resistance_below?: number;
  // Bernoulli lift model
  lift_up?: number;
  lift_down?: number;
  net_lift?: number;
  // Multi-timescale (fast ~5s)
  lift_5s?: number;
  d1_5s?: number;
  d2_5s?: number;
  proj_5s?: number;
  dir_5s?: number;
  // Multi-timescale (medium ~15s)
  lift_15s?: number;
  d1_15s?: number;
  d2_15s?: number;
  proj_15s?: number;
  dir_15s?: number;
  // Multi-timescale (slow ~60s)
  lift_60s?: number;
  d1_60s?: number;
  d2_60s?: number;
  proj_60s?: number;
  dir_60s?: number;
  // Cross-timescale
  cross_confidence?: number;
  projection_coherence?: number;
  alert_flags?: number;
  regime?: string;
  // Optional backend-provided event metadata
  event_state?: string;
  event_direction?: string;
  event_strength?: number;
  event_confidence?: number;
  event_transition?: string;
  feasibility_up?: number;
  feasibility_down?: number;
  directional_bias?: number;
  directional_feasibility?: number;
  directional_feasible?: boolean;
}

type EventState = 'WATCH' | 'ARMED' | 'FIRE' | 'COOLDOWN';
type EventMarkerPhase = 'ARMED' | 'FIRE';

interface SpotEventMarker {
  col: number;
  row: number;
  phase: EventMarkerPhase;
  direction: number;
  confidence: number;
  source: 'derived' | 'backend';
}

/** Parsed and validated URL query parameters. */
interface StreamParams {
  product_type: string;
  symbol: string;
  dt: string;
  speed: string;
  skip: string;
  mode: string;
  start_time?: string;
  pre_smooth_span?: string;
  d1_span?: string;
  d2_span?: string;
  d3_span?: string;
  w_d1?: string;
  w_d2?: string;
  w_d3?: string;
  projection_horizon_s?: string;
  fast_projection_horizon_s?: string;
  smooth_zscore_window?: string;
}

// --------------------------------------------------------- Layout constants

const WS_PORT = 8002;
const MAX_REL_TICKS = 40;                   // +/-40 buckets from anchor
const HMAP_LEVELS = MAX_REL_TICKS * 2 + 1;  // 81 rows
const HMAP_HISTORY = 360;                    // 6 min of 1-second columns
const FLOW_NORM_SCALE = 500;                 // characteristic shares for tanh norm
const DEPTH_NORM_PERCENTILE_DECAY = 0.995;
const SCROLL_MARGIN = 10;                    // rows from edge before auto-scroll
const PROJECTION_MARGIN = 0.18;              // reserve right margin for future projection
const PROJECTION_MAX_ROW_DELTA = 12;
const PROJECTION_IMPULSE_SCALE = 40;
const EVENT_MAX_MARKERS = 80;
const EVENT_SLOPE_EPS = 0.08;
const EVENT_FIRE_CONF_MIN = 0.58;
const EVENT_FIRE_IMPULSE_MIN = 0.22;

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
let currentWindowId: bigint | null = null;

// Price-anchored grid
let anchorPriceDollars = 0;
let anchorInitialized = false;
let currentSpotDollars = 0;

// Spot trail: fractional row position per heatmap column (null = no data)
const spotTrail: (number | null)[] = new Array(HMAP_HISTORY).fill(null);
const spotEventMarkers: SpotEventMarker[] = [];

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
let derivedEventPhase: EventState = 'WATCH';
let derivedEventTransition = '---';
let derivedEventDirection = 0;
let lastSlopeSign = 0;
let lastBackendEventState: string | null = null;

const streamContractErrors = new Set<string>();

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
const $sigProjDir = document.getElementById('sig-proj-dir')!;
const $sigProjConf = document.getElementById('sig-proj-conf')!;
const $sigEventState = document.getElementById('sig-event-state')!;
const $sigEventTransition = document.getElementById('sig-event-transition')!;
const $sigFeasibility = document.getElementById('sig-feasibility')!;
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
  return Math.max(32, totalWidth * (1 - PROJECTION_MARGIN));
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

function optionalNumber(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function optionalBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

function slopeSign(value: number): number {
  if (value > EVENT_SLOPE_EPS) return 1;
  if (value < -EVENT_SLOPE_EPS) return -1;
  return 0;
}

function projectionConfidence(s: SignalsData): number {
  return clamp(
    s.event_confidence
      ?? s.wtd_deriv_conf
      ?? s.projection_coherence
      ?? s.cross_confidence
      ?? s.confidence
      ?? 0,
    0,
    1,
  );
}

function projectionImpulse(s: SignalsData): number {
  const base = s.composite_smooth ?? s.composite ?? 0;
  const projected = s.wtd_projection ?? s.proj_15s ?? base;
  const impulse = projected - base;
  if (Math.abs(impulse) > 1e-6) return impulse;
  return s.wtd_slope ?? s.d1_15s ?? s.d1_composite ?? 0;
}

function projectionRowDelta(s: SignalsData): number {
  const raw = Math.tanh(projectionImpulse(s) / PROJECTION_IMPULSE_SCALE) * PROJECTION_MAX_ROW_DELTA;
  return clamp(-raw, -PROJECTION_MAX_ROW_DELTA, PROJECTION_MAX_ROW_DELTA);
}

function derivedFeasibilityScore(s: SignalsData): number {
  const conf = projectionConfidence(s);
  const impulse = Math.abs(projectionImpulse(s));
  const impulseScore = clamp(Math.tanh(impulse / 40), 0, 1);
  const dirBySlope = slopeSign(s.wtd_slope ?? s.d1_15s ?? s.d1_composite ?? 0);
  const dirByProjection = Math.sign(projectionImpulse(s));
  const alignment = dirByProjection !== 0 && dirByProjection === dirBySlope ? 1 : 0;
  return clamp(conf * 0.55 + impulseScore * 0.3 + alignment * 0.15, 0, 1);
}

function feasibilityLabel(score: number): string {
  if (score >= 0.7) return 'FEASIBLE';
  if (score >= 0.45) return 'WATCH';
  return 'BLOCKED';
}

function pushEventMarker(
  phase: EventMarkerPhase,
  direction: number,
  confidence: number,
  source: 'derived' | 'backend',
): void {
  const row = spotTrail[HMAP_HISTORY - 1] ?? priceToRow(currentSpotDollars);
  if (!Number.isFinite(row)) return;
  spotEventMarkers.push({
    col: HMAP_HISTORY - 1,
    row,
    phase,
    direction,
    confidence: clamp(confidence, 0, 1),
    source,
  });
  while (spotEventMarkers.length > EVENT_MAX_MARKERS) {
    spotEventMarkers.shift();
  }
}

function directionFromLabel(label: string | undefined): number {
  const normalized = (label ?? '').toUpperCase();
  if (normalized === 'UP') return 1;
  if (normalized === 'DOWN') return -1;
  return 0;
}

function fallbackDirectionFromSignals(s: SignalsData): number {
  const biasDirection = Math.sign(optionalNumber(s.directional_bias) ?? 0);
  if (biasDirection !== 0) return biasDirection;
  const liftDirection = Math.sign(s.net_lift ?? s.composite ?? 0);
  if (liftDirection !== 0) return liftDirection;
  return Math.sign(projectionImpulse(s));
}

function updateDerivedEventState(s: SignalsData): void {
  const backendState = optionalString(s.event_state)?.toUpperCase();
  const backendDirection = directionFromLabel(optionalString(s.event_direction));
  const backendTransition = optionalString(s.event_transition)?.toUpperCase();

  if (backendState) {
    if (
      backendState === 'WATCH'
      || backendState === 'ARMED'
      || backendState === 'FIRE'
      || backendState === 'COOLDOWN'
    ) {
      derivedEventPhase = backendState;
    } else if (backendState === 'IDLE') {
      derivedEventPhase = 'WATCH';
    }

    const detectedTransition =
      (lastBackendEventState && lastBackendEventState !== derivedEventPhase)
        ? `${lastBackendEventState}->${derivedEventPhase}`
        : '---';
    derivedEventTransition = backendTransition ?? detectedTransition;
    if (backendDirection !== 0) {
      derivedEventDirection = backendDirection;
    }

    if (
      detectedTransition !== '---'
      && (derivedEventPhase === 'ARMED' || derivedEventPhase === 'FIRE')
    ) {
      // Backend transitions must carry explicit direction; avoid stale fallback.
      if (backendDirection !== 0) {
        pushEventMarker(
          derivedEventPhase as EventMarkerPhase,
          backendDirection,
          projectionConfidence(s),
          'backend',
        );
      }
    }
    lastBackendEventState = derivedEventPhase;
    return;
  }
  lastBackendEventState = null;

  const currentSlopeSign = slopeSign(s.wtd_slope ?? s.d1_15s ?? s.d1_composite ?? 0);
  const confidence = projectionConfidence(s);
  const impulse = Math.abs(projectionImpulse(s));
  const hasInflectionFlag = ((s.alert_flags ?? 0) & 1) !== 0;
  derivedEventTransition = '---';

  if (lastSlopeSign !== 0 && currentSlopeSign !== 0 && currentSlopeSign !== lastSlopeSign) {
    derivedEventPhase = 'ARMED';
    derivedEventDirection = currentSlopeSign;
    derivedEventTransition = 'SLOPE_FLIP';
    pushEventMarker('ARMED', currentSlopeSign, confidence, 'derived');
  } else if (hasInflectionFlag && currentSlopeSign !== 0) {
    derivedEventPhase = 'ARMED';
    derivedEventDirection = currentSlopeSign;
    derivedEventTransition = 'INFLECTION';
    pushEventMarker('ARMED', currentSlopeSign, confidence, 'derived');
  }

  if (derivedEventPhase === 'ARMED') {
    if (
      currentSlopeSign !== 0 &&
      currentSlopeSign === derivedEventDirection &&
      confidence >= EVENT_FIRE_CONF_MIN &&
      impulse >= EVENT_FIRE_IMPULSE_MIN
    ) {
      derivedEventPhase = 'FIRE';
      derivedEventTransition = 'ARMED->FIRE';
      pushEventMarker('FIRE', derivedEventDirection, confidence, 'derived');
    } else if (currentSlopeSign === 0 && confidence < 0.25) {
      derivedEventPhase = 'WATCH';
      derivedEventDirection = 0;
      derivedEventTransition = 'ARMED->WATCH';
    }
  } else if (derivedEventPhase === 'FIRE' && currentSlopeSign === 0 && confidence < 0.2) {
    derivedEventPhase = 'WATCH';
    derivedEventDirection = 0;
    derivedEventTransition = 'FIRE->WATCH';
  }

  if (currentSlopeSign !== 0) {
    lastSlopeSign = currentSlopeSign;
  }
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
  for (let i = spotEventMarkers.length - 1; i >= 0; i--) {
    const marker = spotEventMarkers[i];
    marker.row += shiftRows;
    if (marker.row < 0 || marker.row >= HMAP_LEVELS) {
      spotEventMarkers.splice(i, 1);
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
  for (let i = spotEventMarkers.length - 1; i >= 0; i--) {
    const marker = spotEventMarkers[i];
    marker.col -= 1;
    if (marker.col < 0) {
      spotEventMarkers.splice(i, 1);
    }
  }

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
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, cw, ch);

  if (!userPanned) {
    resetViewport();
  }

  const srcW = visibleCols();
  const srcH = visibleRows();
  const dataWidth = projectionDataWidth(cw);
  const projectionStartX = dataWidth;

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

  // Projection margin with separator
  ctx.fillStyle = 'rgba(12, 14, 24, 0.85)';
  ctx.fillRect(projectionStartX, 0, cw - projectionStartX, ch);
  ctx.strokeStyle = 'rgba(120, 150, 180, 0.35)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(projectionStartX + 0.5, 0);
  ctx.lineTo(projectionStartX + 0.5, ch);
  ctx.stroke();

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

  // Event markers on spot trail
  for (const marker of spotEventMarkers) {
    const x = colToX(marker.col + 0.5);
    const y = rowToY(marker.row);
    if (x < -16 || x > dataWidth + 16 || y < -16 || y > ch + 16) continue;

    const dir = marker.direction > 0 ? 1 : marker.direction < 0 ? -1 : 0;
    const directionColor = dir >= 0 ? '#22cc66' : '#cc2255';
    const color = marker.phase === 'FIRE' ? directionColor : '#ccaa22';
    const radius = marker.phase === 'FIRE' ? 4.8 : 3.8;
    const confAlpha = 0.35 + marker.confidence * 0.55;

    ctx.fillStyle = `rgba(12, 12, 20, ${confAlpha})`;
    ctx.beginPath();
    ctx.arc(x, y, radius + 2.0, 0, Math.PI * 2);
    ctx.fill();

    // Directional marker: triangle points UP for +1, DOWN for -1.
    const tipY = dir >= 0 ? y - radius : y + radius;
    const baseY = dir >= 0 ? y + radius * 0.8 : y - radius * 0.8;
    const halfW = radius * 0.9;

    if (marker.phase === 'FIRE') {
      ctx.fillStyle = color;
    } else {
      ctx.fillStyle = 'rgba(0,0,0,0)';
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = marker.phase === 'FIRE' ? 1.9 : 1.4;
    ctx.beginPath();
    ctx.moveTo(x, tipY);
    ctx.lineTo(x + halfW, baseY);
    ctx.lineTo(x - halfW, baseY);
    ctx.closePath();
    if (marker.phase === 'FIRE') {
      ctx.fill();
    }
    ctx.stroke();

    // ARMED gets an inner notch to visually differ from FIRE.
    if (marker.phase === 'ARMED') {
      const notchY = dir >= 0 ? y - radius * 0.25 : y + radius * 0.25;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(x - halfW * 0.45, notchY);
      ctx.lineTo(x + halfW * 0.45, notchY);
      ctx.stroke();
    }
  }

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

      // Forward projection cone and direction line (anchored at spot)
      if (currentSignals) {
        const conf = projectionConfidence(currentSignals);
        const rowDelta = projectionRowDelta(currentSignals);
        const impulse = projectionImpulse(currentSignals);
        const direction = Math.sign(impulse) || Math.sign(-rowDelta);
        const yEnd = rowToY(lastSpot + rowDelta);
        const coneRows = 1 + (1 - conf) * 8;
        const yUpper = rowToY(lastSpot + rowDelta - coneRows);
        const yLower = rowToY(lastSpot + rowDelta + coneRows);
        const xEnd = cw - 8;
        const color = direction > 0 ? '#22cc66' : (direction < 0 ? '#cc2255' : '#888');

        ctx.fillStyle = direction > 0
          ? `rgba(34, 204, 102, ${0.1 + conf * 0.22})`
          : direction < 0
            ? `rgba(204, 34, 85, ${0.1 + conf * 0.22})`
            : `rgba(120, 120, 120, ${0.08 + conf * 0.12})`;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(xEnd, yUpper);
        ctx.lineTo(xEnd, yLower);
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(xEnd, yEnd);
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(xEnd, yEnd, 3.2, 0, Math.PI * 2);
        ctx.fill();

        const dirText = direction > 0 ? 'UP' : direction < 0 ? 'DOWN' : 'FLAT';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(200, 210, 230, 0.85)';
        ctx.fillText(`${dirText} ${(conf * 100).toFixed(0)}%`, projectionStartX + 6, 6);
      }

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
  updateDerivedEventState(s);

  // ── Section 1: Lift Gauge ──
  const netLift = s.net_lift ?? 0;
  $('sig-net-lift').textContent = fmt(netLift, 1);
  $('sig-net-lift').style.color = signColour(netLift, 0.01);

  const liftNorm = Math.tanh(netLift / 200) * 0.5 + 0.5;
  $('lift-marker').style.left = `${liftNorm * 100}%`;

  // Condition indicators: V=vacuum, P=pressure, R=low resistance
  const vacPresent = (s.vacuum_above > 1) || (s.vacuum_below < -1);
  const pressPresent = ((s.pressure_below ?? 0) > 5) || ((s.pressure_above ?? 0) > 5);
  const moveSideResist = netLift >= 0
    ? (s.resistance_above ?? 999)
    : (s.resistance_below ?? 999);
  setInd('cond-v', vacPresent, '#22cc66');
  setInd('cond-p', pressPresent, '#22cc66');
  setInd('cond-r', moveSideResist < 50, '#22cc66');

  $('sig-cross-conf').textContent = `${((s.cross_confidence ?? 0) * 100).toFixed(0)}%`;
  $('sig-lift-str').textContent = `${(Math.min(1, Math.abs(netLift) / 100) * 100).toFixed(0)}%`;

  // ── Section 2: Multi-Timescale ──
  const timescales = [
    { key: '5s', lift: s.lift_5s ?? 0, d1: s.d1_5s ?? 0, dir: Math.sign(s.dir_5s ?? 0) },
    { key: '15s', lift: s.lift_15s ?? 0, d1: s.d1_15s ?? 0, dir: Math.sign(s.dir_15s ?? 0) },
    { key: '60s', lift: s.lift_60s ?? 0, d1: s.d1_60s ?? 0, dir: Math.sign(s.dir_60s ?? 0) },
  ];

  for (const ts of timescales) {
    $(`sig-lift-${ts.key}`).textContent = fmt(ts.lift, 1);
    $(`sig-lift-${ts.key}`).style.color = signColour(ts.lift, 0.01);
    $(`sig-d1-${ts.key}`).textContent = fmt(ts.d1, 1);
    $(`sig-d1-${ts.key}`).style.color = signColour(ts.d1, 0.05);
    fillGauge(`d1-${ts.key}-fill`, ts.d1, 50);

    const arrowEl = $(`sig-arrow-${ts.key}`);
    if (ts.dir > 0) {
      arrowEl.textContent = '\u25B2';
      arrowEl.style.color = '#22cc66';
    } else if (ts.dir < 0) {
      arrowEl.textContent = '\u25BC';
      arrowEl.style.color = '#cc2255';
    } else {
      arrowEl.textContent = '\u2014';
      arrowEl.style.color = '#555';
    }
  }

  // Alignment: all timescales agree on sign?
  const signs = timescales.map(t => Math.sign(t.lift));
  const nonZero = signs.filter(v => v !== 0);
  const alignEl = $('sig-alignment');
  if (nonZero.length === 0) {
    alignEl.textContent = '---';
    alignEl.style.color = '#555';
  } else if (nonZero.every(v => v === nonZero[0])) {
    alignEl.textContent = 'ALIGNED';
    alignEl.style.color = '#22cc66';
  } else {
    alignEl.textContent = 'DIVERGENT';
    alignEl.style.color = '#ccaa22';
  }

  // ── Section 3: Alerts ──
  const flags = s.alert_flags ?? 0;
  const hasInf = (flags & 1) !== 0;
  const hasDec = (flags & 2) !== 0;
  const hasReg = (flags & 4) !== 0;
  setInd('alert-inf', hasInf, '#ccaa22');
  setInd('alert-dec', hasDec, '#cc7722');
  setInd('alert-reg', hasReg, '#cc2255');
  $('alert-none').style.display = (hasInf || hasDec || hasReg) ? 'none' : '';

  // ── Section 4: Projection/Event badges ──
  const impulse = projectionImpulse(s);
  const direction = Math.sign(impulse);
  const backendDirection = directionFromLabel(optionalString(s.event_direction));
  const conf = projectionConfidence(s);
  const directionText = direction > 0 ? 'UP' : direction < 0 ? 'DOWN' : 'FLAT';
  const directionColor = direction > 0 ? '#22cc66' : direction < 0 ? '#cc2255' : '#888';
  const eventDirection = backendDirection !== 0
    ? backendDirection
    : (derivedEventDirection !== 0 ? derivedEventDirection : fallbackDirectionFromSignals(s));
  $sigProjDir.textContent = directionText;
  $sigProjDir.style.color = directionColor;
  $sigProjDir.style.background = direction === 0 ? '#2f3138' : 'rgba(20, 20, 28, 0.9)';
  $sigProjConf.textContent = `${(conf * 100).toFixed(0)}%`;
  $sigProjConf.style.color = conf >= 0.6 ? '#22cc66' : conf >= 0.35 ? '#ccaa22' : '#888';

  const eventState = optionalString(s.event_state)?.toUpperCase() ?? derivedEventPhase;
  const eventTransition = optionalString(s.event_transition)?.toUpperCase() ?? derivedEventTransition;
  $sigEventState.textContent = eventState;
  if (eventState === 'FIRE') {
    $sigEventState.style.color = '#ffffff';
    $sigEventState.style.background = eventDirection > 0 ? '#22cc66' : '#cc2255';
  } else if (eventState === 'ARMED') {
    $sigEventState.style.color = '#111';
    $sigEventState.style.background = '#ccaa22';
  } else {
    $sigEventState.style.color = '#888';
    $sigEventState.style.background = '#2f3138';
  }
  $sigEventTransition.textContent = eventTransition;
  $sigEventTransition.style.color = eventTransition === '---' ? '#666' : '#ddd';

  const backendFeasibilityUp = optionalNumber(s.feasibility_up);
  const backendFeasibilityDown = optionalNumber(s.feasibility_down);
  const backendDirectionalFeasibility = optionalNumber(s.directional_feasibility);
  const backendDirectionalFlag = optionalBoolean(s.directional_feasible);
  const biasDirection = Math.sign(optionalNumber(s.directional_bias) ?? 0);

  let feasibilityScore: number | undefined;
  if (eventDirection > 0 && backendFeasibilityUp !== undefined) {
    feasibilityScore = backendFeasibilityUp;
  } else if (eventDirection < 0 && backendFeasibilityDown !== undefined) {
    feasibilityScore = backendFeasibilityDown;
  } else if (backendFeasibilityUp !== undefined || backendFeasibilityDown !== undefined) {
    feasibilityScore = Math.max(backendFeasibilityUp ?? 0, backendFeasibilityDown ?? 0);
  } else if (backendDirectionalFeasibility !== undefined) {
    feasibilityScore = backendDirectionalFeasibility;
  } else if (backendDirectionalFlag !== undefined) {
    feasibilityScore = backendDirectionalFlag ? 1 : 0;
  } else {
    feasibilityScore = derivedFeasibilityScore(s);
  }
  feasibilityScore = clamp(feasibilityScore, 0, 1);

  if (eventDirection === 0 && biasDirection !== 0) {
    $sigProjDir.textContent = biasDirection > 0 ? 'UP' : 'DOWN';
    $sigProjDir.style.color = biasDirection > 0 ? '#22cc66' : '#cc2255';
  }

  const feasibility = feasibilityLabel(feasibilityScore);
  $sigFeasibility.textContent = feasibility;
  $sigFeasibility.style.color = feasibilityScore >= 0.7 ? '#22cc66' : feasibilityScore >= 0.45 ? '#ccaa22' : '#cc2255';

  // ── Section 5: Components ──
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
  $('sig-press-above').textContent = fmt(s.pressure_above ?? 0, 1);
  $('sig-press-above').style.color = signColour(s.pressure_above ?? 0, 0.02);
  $('sig-press-below').textContent = fmt(s.pressure_below ?? 0, 1);
  $('sig-press-below').style.color = signColour(s.pressure_below ?? 0, 0.02);
  $('sig-resist-above').textContent = fmt(s.resistance_above ?? 0, 1);
  $('sig-resist-above').style.color = signColour(s.resistance_above ?? 0, 0.01);
  $('sig-resist-below').textContent = fmt(s.resistance_below ?? 0, 1);
  $('sig-resist-below').style.color = signColour(s.resistance_below ?? 0, 0.01);

  // ── Bottom Bar ──
  const regime = s.regime || 'NEUTRAL';
  const regimeColors: Record<string, string> = {
    LIFT: '#22cc66', DRAG: '#cc2255', NEUTRAL: '#888', CHOP: '#ccaa22',
  };
  $('regime-label').textContent = regime;
  $('regime-label').style.color = regimeColors[regime] || '#888';
  $('regime-lift-val').textContent = fmt(netLift, 1);
  $('regime-lift-val').style.color = signColour(netLift, 0.01);

  $('bot-d1-5s').textContent = fmt(s.d1_5s ?? 0, 1);
  $('bot-d1-15s').textContent = fmt(s.d1_15s ?? 0, 1);
  $('bot-d1-60s').textContent = fmt(s.d1_60s ?? 0, 1);
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

/** Toggle an indicator badge between dim and lit states. */
function setInd(id: string, active: boolean, color: string): void {
  const el = document.getElementById(id);
  if (!el) return;
  if (active) {
    el.style.background = color;
    el.style.color = '#fff';
  } else {
    el.style.background = '#333';
    el.style.color = '#555';
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
    mode: params.get('mode') || 'replay',
    start_time: params.get('start_time') || undefined,
    pre_smooth_span: params.get('pre_smooth_span') || undefined,
    d1_span: params.get('d1_span') || undefined,
    d2_span: params.get('d2_span') || undefined,
    d3_span: params.get('d3_span') || undefined,
    w_d1: params.get('w_d1') || undefined,
    w_d2: params.get('w_d2') || undefined,
    w_d3: params.get('w_d3') || undefined,
    projection_horizon_s: params.get('projection_horizon_s') || undefined,
    fast_projection_horizon_s: params.get('fast_projection_horizon_s') || undefined,
    smooth_zscore_window: params.get('smooth_zscore_window') || undefined,
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
  $streamState.textContent = streamPaused ? 'PAUSED' : 'LIVE';
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
  snap = null;
  currentFlow = [];
  currentSignals = null;
  windowCount = 0;
  currentWindowId = null;
  anchorPriceDollars = 0;
  anchorInitialized = false;
  currentSpotDollars = 0;
  runningMaxDepth = 100;
  derivedEventPhase = 'WATCH';
  derivedEventTransition = '---';
  derivedEventDirection = 0;
  lastSlopeSign = 0;
  lastBackendEventState = null;
  configReceived = false;

  for (let i = 0; i < hmapData.length; i += 4) {
    hmapData[i] = 10;
    hmapData[i + 1] = 10;
    hmapData[i + 2] = 15;
    hmapData[i + 3] = 255;
  }
  for (let i = 0; i < spotTrail.length; i++) spotTrail[i] = null;
  spotEventMarkers.length = 0;

  $spotVal.textContent = '--';
  $tsVal.textContent = '--:--:--';
  $winVal.textContent = '0';
  $winIdVal.textContent = '--';
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
  const {
    product_type, symbol, dt, speed, skip, mode, start_time,
    pre_smooth_span, d1_span, d2_span, d3_span,
    w_d1, w_d2, w_d3, projection_horizon_s,
    fast_projection_horizon_s, smooth_zscore_window,
  } = streamParams;

  const tuningParams = new URLSearchParams();
  if (mode && mode !== 'replay') tuningParams.set('mode', mode);
  if (start_time) tuningParams.set('start_time', start_time);
  if (pre_smooth_span) tuningParams.set('pre_smooth_span', pre_smooth_span);
  if (d1_span) tuningParams.set('d1_span', d1_span);
  if (d2_span) tuningParams.set('d2_span', d2_span);
  if (d3_span) tuningParams.set('d3_span', d3_span);
  if (w_d1) tuningParams.set('w_d1', w_d1);
  if (w_d2) tuningParams.set('w_d2', w_d2);
  if (w_d3) tuningParams.set('w_d3', w_d3);
  if (projection_horizon_s) tuningParams.set('projection_horizon_s', projection_horizon_s);
  if (fast_projection_horizon_s) tuningParams.set('fast_projection_horizon_s', fast_projection_horizon_s);
  if (smooth_zscore_window) tuningParams.set('smooth_zscore_window', smooth_zscore_window);

  const urlBase =
    `ws://localhost:${WS_PORT}/v1/vacuum-pressure/stream` +
    `?product_type=${encodeURIComponent(product_type)}` +
    `&symbol=${encodeURIComponent(symbol)}` +
    `&dt=${encodeURIComponent(dt)}` +
    `&speed=${encodeURIComponent(speed)}` +
    `&skip_minutes=${encodeURIComponent(skip)}`;
  const url = tuningParams.toString()
    ? `${urlBase}&${tuningParams.toString()}`
    : urlBase;

  console.log(`[VP] Connecting to: ${url}`);
  const ws = new WebSocket(url);
  wsClient = ws;

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
          } else if (msg.type === 'batch_start') {
            // If config was never received, activate legacy fallback once
            if (!configReceived) {
              activateLegacyFallback();
              // Prevent re-triggering on subsequent batches
              configReceived = true;
            }
            windowCount++;
            $winVal.textContent = String(windowCount);
            const rawWindowId = msg.window_end_ts_ns;
            if (rawWindowId !== undefined && rawWindowId !== null) {
              try {
                currentWindowId = BigInt(String(rawWindowId));
                $winIdVal.textContent = currentWindowId.toString();
              } catch {
                $winIdVal.textContent = String(rawWindowId);
              }
            }
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
                window_end_ts_ns: requireBigIntField('snap', j, 'window_end_ts_ns'),
                mid_price: requireNumberField('snap', j, 'mid_price'),
                spot_ref_price_int: requireBigIntField('snap', j, 'spot_ref_price_int'),
                best_bid_price_int: requireBigIntField('snap', j, 'best_bid_price_int'),
                best_ask_price_int: requireBigIntField('snap', j, 'best_ask_price_int'),
                book_valid: requireBooleanField('snap', j, 'book_valid'),
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
                rel_ticks: requireNumberField('flow', j, 'rel_ticks'),
                side: requireStringField('flow', j, 'side'),
                depth_qty_end: requireNumberField('flow', j, 'depth_qty_end'),
                add_qty: requireNumberField('flow', j, 'add_qty'),
                pull_qty: requireNumberField('flow', j, 'pull_qty'),
                fill_qty: requireNumberField('flow', j, 'fill_qty'),
                depth_qty_rest: requireNumberField('flow', j, 'depth_qty_rest'),
                pull_qty_rest: requireNumberField('flow', j, 'pull_qty_rest'),
                net_flow: requireNumberField('flow', j, 'net_flow'),
                vacuum_intensity: requireNumberField('flow', j, 'vacuum_intensity'),
                pressure_intensity: requireNumberField('flow', j, 'pressure_intensity'),
                rest_fraction: requireNumberField('flow', j, 'rest_fraction'),
              });
            }
            currentFlow = rows;
            pushHeatmapColumn(rows, currentSpotDollars);
          } else if (surface === 'signals' && table.numRows > 0) {
            const row = table.get(0);
            if (row) {
              const j = row.toJSON() as Record<string, unknown>;
              currentSignals = {
                window_end_ts_ns: requireBigIntField('signals', j, 'window_end_ts_ns'),
                vacuum_above: requireNumberField('signals', j, 'vacuum_above'),
                vacuum_below: requireNumberField('signals', j, 'vacuum_below'),
                resting_drain_ask: requireNumberField('signals', j, 'resting_drain_ask'),
                resting_drain_bid: requireNumberField('signals', j, 'resting_drain_bid'),
                flow_imbalance: requireNumberField('signals', j, 'flow_imbalance'),
                fill_imbalance: requireNumberField('signals', j, 'fill_imbalance'),
                depth_imbalance: requireNumberField('signals', j, 'depth_imbalance'),
                rest_depth_imbalance: requireNumberField('signals', j, 'rest_depth_imbalance'),
                bid_migration_com: optionalNumber(j.bid_migration_com) ?? 0,
                ask_migration_com: optionalNumber(j.ask_migration_com) ?? 0,
                composite: requireNumberField('signals', j, 'composite'),
                composite_smooth: optionalNumber(j.composite_smooth),
                d1_composite: requireNumberField('signals', j, 'd1_composite'),
                d2_composite: requireNumberField('signals', j, 'd2_composite'),
                d3_composite: requireNumberField('signals', j, 'd3_composite'),
                d1_smooth: optionalNumber(j.d1_smooth),
                d2_smooth: optionalNumber(j.d2_smooth),
                d3_smooth: optionalNumber(j.d3_smooth),
                wtd_slope: optionalNumber(j.wtd_slope),
                wtd_projection: optionalNumber(j.wtd_projection),
                wtd_projection_500ms: optionalNumber(j.wtd_projection_500ms),
                wtd_deriv_conf: optionalNumber(j.wtd_deriv_conf),
                z_composite_raw: optionalNumber(j.z_composite_raw),
                z_composite_smooth: optionalNumber(j.z_composite_smooth),
                confidence: requireNumberField('signals', j, 'confidence'),
                strength: requireNumberField('signals', j, 'strength'),
                strength_smooth: optionalNumber(j.strength_smooth),
                // Bernoulli lift model
                pressure_above: optionalNumber(j.pressure_above) ?? 0,
                pressure_below: optionalNumber(j.pressure_below) ?? 0,
                resistance_above: optionalNumber(j.resistance_above) ?? 0,
                resistance_below: optionalNumber(j.resistance_below) ?? 0,
                lift_up: optionalNumber(j.lift_up) ?? 0,
                lift_down: optionalNumber(j.lift_down) ?? 0,
                net_lift: optionalNumber(j.net_lift) ?? 0,
                // Multi-timescale
                lift_5s: optionalNumber(j.lift_5s) ?? 0,
                d1_5s: optionalNumber(j.d1_5s) ?? 0,
                d2_5s: optionalNumber(j.d2_5s) ?? 0,
                proj_5s: optionalNumber(j.proj_5s) ?? 0,
                dir_5s: Math.trunc(optionalNumber(j.dir_5s) ?? 0),
                lift_15s: optionalNumber(j.lift_15s) ?? 0,
                d1_15s: optionalNumber(j.d1_15s) ?? 0,
                d2_15s: optionalNumber(j.d2_15s) ?? 0,
                proj_15s: optionalNumber(j.proj_15s) ?? 0,
                dir_15s: Math.trunc(optionalNumber(j.dir_15s) ?? 0),
                lift_60s: optionalNumber(j.lift_60s) ?? 0,
                d1_60s: optionalNumber(j.d1_60s) ?? 0,
                d2_60s: optionalNumber(j.d2_60s) ?? 0,
                proj_60s: optionalNumber(j.proj_60s) ?? 0,
                dir_60s: Math.trunc(optionalNumber(j.dir_60s) ?? 0),
                // Cross-timescale
                cross_confidence: optionalNumber(j.cross_confidence) ?? 0,
                projection_coherence: optionalNumber(j.projection_coherence),
                alert_flags: Math.trunc(optionalNumber(j.alert_flags) ?? 0),
                regime: optionalString(j.regime) ?? 'NEUTRAL',
                event_state: optionalString(j.event_state),
                event_direction: optionalString(j.event_direction),
                event_strength: optionalNumber(j.event_strength),
                event_confidence: optionalNumber(j.event_confidence),
                event_transition: optionalString(j.event_transition),
                feasibility_up: optionalNumber(j.feasibility_up),
                feasibility_down: optionalNumber(j.feasibility_down),
                directional_bias: optionalNumber(j.directional_bias),
                directional_feasibility: optionalNumber(j.directional_feasibility),
                directional_feasible: optionalBoolean(j.directional_feasible),
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
