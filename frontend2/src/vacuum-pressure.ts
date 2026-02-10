/**
 * Vacuum & Pressure Detector – frontend visualisation.
 *
 * Connects to ws://localhost:8002/v1/vacuum-pressure/stream
 * Renders:
 *   - Left panel:  current depth profile (horizontal bars per level)
 *   - Centre panel: scrolling heatmap   (time × price, colour = flow state)
 *   - Right panel:  signal gauges        (DOM-based, updated per tick)
 *   - Bottom bar:   composite direction + derivative readings
 *
 * Colour encoding for heatmap cells:
 *   Building (add > pull) → cyan-green   (pressure zone)
 *   Draining (pull > add) → red-magenta  (vacuum / active drain)
 *   No liquidity          → near-black   (void)
 */

import { tableFromIPC } from 'apache-arrow';

// ────────────────────────────── Types ──────────────────────────────

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

// ────────────────────────────── Config ─────────────────────────────

const WS_PORT = 8002;
const MAX_REL_TICKS = 40;              // ±40 ticks = ±$20
const HMAP_LEVELS = MAX_REL_TICKS * 2 + 1; // 81 rows
const HMAP_HISTORY = 360;             // 6 min of 1-second columns
const PRICE_SCALE = 1e-9;
const FLOW_NORM_SCALE = 500;          // characteristic shares for tanh norm
const DEPTH_NORM_PERCENTILE_DECAY = 0.995; // EMA decay for running max depth

// ────────────────────────────── State ──────────────────────────────

let snap: SnapData | null = null;
let currentFlow: FlowRow[] = [];
let currentSignals: SignalsData | null = null;
let windowCount = 0;

// Heatmap pixel buffer (RGBA, HMAP_HISTORY × HMAP_LEVELS)
const hmapData = new Uint8ClampedArray(HMAP_HISTORY * HMAP_LEVELS * 4);
// Initialise to dark background
for (let i = 0; i < hmapData.length; i += 4) {
  hmapData[i] = 10;
  hmapData[i + 1] = 10;
  hmapData[i + 2] = 15;
  hmapData[i + 3] = 255;
}

let runningMaxDepth = 100; // adaptive normalisation

// ────────────────────────────── DOM refs ───────────────────────────

const $spotVal     = document.getElementById('spot-val')!;
const $tsVal       = document.getElementById('ts-val')!;
const $winVal      = document.getElementById('win-val')!;
const $compLabel   = document.getElementById('composite-label')!;
const $compMarker  = document.getElementById('composite-marker')!;
const $spotLine    = document.getElementById('spot-line-label')!;

function $(id: string) { return document.getElementById(id)!; }

// ────────────────────────────── Helpers ────────────────────────────

/** rel_ticks → heatmap row (row 0 = top = highest price). */
function tickToRow(rt: number): number {
  return MAX_REL_TICKS - rt; // +40 → row 0, 0 → row 40, -40 → row 80
}

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

/** Colour a value: positive → green, negative → red, zero → grey. */
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

/** Heatmap cell colour from depth + net flow. */
function heatmapRGB(
  depth: number, netFlow: number, maxDepth: number,
): [number, number, number] {
  const depthN = Math.min(1.0, depth / (maxDepth + 1));
  const flowN = Math.tanh(netFlow / FLOW_NORM_SCALE);
  const lum = 0.04 + depthN * 0.56;

  if (flowN > 0.03) {
    // Building: cyan-green
    const t = flowN;
    return [
      Math.round((0.04 + t * 0.06) * 255 * lum),
      Math.round((0.45 + t * 0.55) * 255 * lum),
      Math.round((0.35 + t * 0.35) * 255 * lum),
    ];
  } else if (flowN < -0.03) {
    // Draining: red-magenta
    const t = -flowN;
    return [
      Math.round((0.55 + t * 0.45) * 255 * lum),
      Math.round(0.04 * 255 * lum),
      Math.round((0.20 + t * 0.35) * 255 * lum),
    ];
  } else {
    // Neutral: dark blue-grey
    return [
      Math.round(0.12 * 255 * lum),
      Math.round(0.12 * 255 * lum),
      Math.round(0.22 * 255 * lum),
    ];
  }
}

// ────────────────────────────── Heatmap buffer ops ─────────────────

/** Shift heatmap left by 1 column, fill rightmost with new flow data. */
function pushHeatmapColumn(flow: FlowRow[]): void {
  const w = HMAP_HISTORY;
  const h = HMAP_LEVELS;
  const d = hmapData;

  // Shift each row left by 1 pixel (4 bytes)
  for (let y = 0; y < h; y++) {
    const rowOff = y * w * 4;
    d.copyWithin(rowOff, rowOff + 4, rowOff + w * 4);
  }

  // Clear rightmost column to background
  for (let y = 0; y < h; y++) {
    const idx = (y * w + (w - 1)) * 4;
    d[idx] = 10; d[idx + 1] = 10; d[idx + 2] = 15; d[idx + 3] = 255;
  }

  // Aggregate flow: combine bid + ask at same rel_ticks
  const byLevel = new Map<number, { depth: number; net: number }>();
  for (const r of flow) {
    if (r.rel_ticks < -MAX_REL_TICKS || r.rel_ticks > MAX_REL_TICKS) continue;
    const existing = byLevel.get(r.rel_ticks);
    if (existing) {
      existing.depth += r.depth_qty_end;
      existing.net += r.net_flow;
    } else {
      byLevel.set(r.rel_ticks, { depth: r.depth_qty_end, net: r.net_flow });
    }
  }

  // Update running max depth (EMA)
  let maxD = 0;
  for (const v of byLevel.values()) {
    if (v.depth > maxD) maxD = v.depth;
  }
  runningMaxDepth = Math.max(
    runningMaxDepth * DEPTH_NORM_PERCENTILE_DECAY,
    maxD,
  );

  // Fill rightmost column
  for (const [rt, v] of byLevel) {
    const row = tickToRow(rt);
    if (row < 0 || row >= h) continue;
    const [r, g, b] = heatmapRGB(v.depth, v.net, runningMaxDepth);
    const idx = (row * w + (w - 1)) * 4;
    d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255;
  }

  // Spot line: draw bright row at rel_ticks = 0
  const spotRow = tickToRow(0);
  if (spotRow >= 0 && spotRow < h) {
    const idx = (spotRow * w + (w - 1)) * 4;
    d[idx] = 0; d[idx + 1] = 255; d[idx + 2] = 170; d[idx + 3] = 255;
  }
}

// ────────────────────────────── Rendering ──────────────────────────

// Cached offscreen canvas for heatmap ImageData
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

  // Create offscreen canvas at heatmap logical resolution
  if (!hmapOffscreen) {
    hmapOffscreen = document.createElement('canvas');
    hmapOffscreen.width = HMAP_HISTORY;
    hmapOffscreen.height = HMAP_LEVELS;
  }
  const offCtx = hmapOffscreen.getContext('2d')!;
  const imgData = new ImageData(hmapData, HMAP_HISTORY, HMAP_LEVELS);
  offCtx.putImageData(imgData, 0, 0);

  // Draw scaled to display
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(hmapOffscreen, 0, 0, cw, ch);

  // Spot line overlay (horizontal dashed line)
  const spotY = (tickToRow(0) / HMAP_LEVELS) * ch;
  ctx.strokeStyle = 'rgba(0, 255, 170, 0.35)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, spotY);
  ctx.lineTo(cw, spotY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Position the SPOT label
  $spotLine.style.top = `${spotY - 10}px`;
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

  if (currentFlow.length === 0) return;

  const midX = cw / 2;
  const rowH = ch / HMAP_LEVELS;
  const maxD = runningMaxDepth || 1;
  const barMax = midX - 4;

  for (const r of currentFlow) {
    if (r.rel_ticks < -MAX_REL_TICKS || r.rel_ticks > MAX_REL_TICKS) continue;
    const row = tickToRow(r.rel_ticks);
    const y = row * rowH;
    const barW = Math.min(barMax, (r.depth_qty_end / maxD) * barMax);

    if (r.side === 'B') {
      // Bid: draw left from midX
      const flowT = Math.tanh(r.net_flow / FLOW_NORM_SCALE);
      const alpha = 0.4 + r.rest_fraction * 0.5;
      if (flowT >= 0) {
        ctx.fillStyle = `rgba(30, ${140 + 80 * flowT}, ${120 + 50 * flowT}, ${alpha})`;
      } else {
        ctx.fillStyle = `rgba(${140 + 80 * (-flowT)}, 30, ${60 + 40 * (-flowT)}, ${alpha})`;
      }
      ctx.fillRect(midX - barW, y, barW, Math.max(1, rowH - 0.5));
    } else {
      // Ask: draw right from midX
      const flowT = Math.tanh(r.net_flow / FLOW_NORM_SCALE);
      const alpha = 0.4 + r.rest_fraction * 0.5;
      if (flowT >= 0) {
        ctx.fillStyle = `rgba(30, ${140 + 80 * flowT}, ${120 + 50 * flowT}, ${alpha})`;
      } else {
        ctx.fillStyle = `rgba(${140 + 80 * (-flowT)}, 30, ${60 + 40 * (-flowT)}, ${alpha})`;
      }
      ctx.fillRect(midX, y, barW, Math.max(1, rowH - 0.5));
    }
  }

  // Center line
  ctx.strokeStyle = 'rgba(100, 100, 150, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(midX, 0);
  ctx.lineTo(midX, ch);
  ctx.stroke();

  // Spot line
  const spotY = tickToRow(0) * rowH;
  ctx.strokeStyle = 'rgba(0, 255, 170, 0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, spotY);
  ctx.lineTo(cw, spotY);
  ctx.stroke();

  // Labels
  ctx.font = '9px monospace';
  ctx.fillStyle = 'rgba(100, 180, 200, 0.6)';
  ctx.fillText('BID', 4, 12);
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

  // Composite bar marker: map composite to [0%, 100%]
  const norm = Math.tanh(s.composite / 2000) * 0.5 + 0.5;
  $compMarker.style.left = `${norm * 100}%`;

  // Derivatives
  $('sig-d1').textContent = fmt(s.d1_composite, 2);
  $('sig-d1').style.color = signColour(s.d1_composite, 0.05);
  $('sig-d2').textContent = fmt(s.d2_composite, 2);
  $('sig-d2').style.color = signColour(s.d2_composite, 0.1);
  $('sig-d3').textContent = fmt(s.d3_composite, 3);
  $('sig-d3').style.color = signColour(s.d3_composite, 0.2);

  // Derivative gauge fills
  fillGauge('d1-fill', s.d1_composite, 500);
  fillGauge('d2-fill', s.d2_composite, 200);
  fillGauge('d3-fill', s.d3_composite, 100);

  // Bottom bar
  $('bot-d1').textContent = fmt(s.d1_composite, 1);
  $('bot-d2').textContent = fmt(s.d2_composite, 1);
  $('bot-d3').textContent = fmt(s.d3_composite, 2);

  // Components
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

  // Composite label
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

// ────────────────────────────── WebSocket ──────────────────────────

function connectWS(): void {
  const params = new URLSearchParams(window.location.search);
  const symbol = params.get('symbol') || 'QQQ';
  const dt = params.get('dt') || '2026-02-06';
  const speed = params.get('speed') || '10';
  const skip = params.get('skip') || '5';

  const url =
    `ws://localhost:${WS_PORT}/v1/vacuum-pressure/stream` +
    `?symbol=${symbol}&dt=${dt}&speed=${speed}&skip_minutes=${skip}`;

  console.log(`Connecting to: ${url}`);
  const ws = new WebSocket(url);

  let pendingSurface: string | null = null;

  ws.onopen = () => console.log('VP WebSocket connected');

  ws.onmessage = async (event) => {
    try {
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        if (msg.type === 'batch_start') {
          windowCount++;
          $winVal.textContent = String(windowCount);
        } else if (msg.type === 'surface_header') {
          pendingSurface = msg.surface;
        }
      } else if (event.data instanceof Blob) {
        const surface = pendingSurface;
        if (!surface) return;
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
            $spotVal.textContent = `$${snap.mid_price.toFixed(2)}`;
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
          pushHeatmapColumn(rows);
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
      console.error('VP message error:', e);
    }
  };

  ws.onerror = (err) => console.error('VP WebSocket error:', err);
  ws.onclose = () => {
    console.log('VP WebSocket closed, reconnecting in 3s…');
    setTimeout(connectWS, 3000);
  };
}

// ────────────────────────────── Render loop ────────────────────────

function startRenderLoop(): void {
  const hmapCanvas = document.getElementById('heatmap-canvas') as HTMLCanvasElement;
  const profCanvas = document.getElementById('profile-canvas') as HTMLCanvasElement;

  function frame() {
    renderHeatmap(hmapCanvas);
    renderProfile(profCanvas);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ────────────────────────────── Init ──────────────────────────────

connectWS();
startRenderLoop();
