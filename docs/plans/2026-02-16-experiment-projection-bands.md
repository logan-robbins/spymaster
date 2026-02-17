# Experiment-Driven Projection Bands Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render directional pressure bands in the frontend projection zone driven by PFP/ADS/ERD experiment signals computed in-browser from Arrow grid data.

**Architecture:** Pure frontend computation — three experiment signals (PFP, ADS, ERD) computed incrementally per 100ms bin from `GridBucketRow` data already arriving via WebSocket. Blended into one composite directional signal. Rendered as Gaussian-falloff purple bands at 4 horizon positions in the 15% projection zone. Zero backend changes.

**Tech Stack:** TypeScript, Canvas 2D, `Uint8ClampedArray` pixel buffers, TypedArrays for rolling state.

**Design doc:** `docs/plans/2026-02-16-experiment-projection-bands-design.md`

---

### Task 1: Utility Functions — Rolling OLS + Robust Z-Score

These are shared building blocks used by ADS and ERD. Port them from Python (`eval_harness.py:318-369`) to TypeScript as standalone classes with incremental (O(1) per bin) interfaces.

**Files:**
- Create: `frontend/src/experiment-math.ts`
- Test: `frontend/src/experiment-math.test.ts` (manual verification via console, not jest — frontend has no test runner)

**Step 1: Create the math utilities file**

```typescript
// frontend/src/experiment-math.ts

/**
 * Incremental rolling OLS slope.
 *
 * Uses: slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
 * where x = [0, 1, ..., w-1] are fixed positions.
 *
 * Ring buffer stores y values. On each push, the oldest y exits
 * and the new y enters. Sums are updated in O(1).
 *
 * Reference: eval_harness.py:318-347
 */
export class IncrementalOLSSlope {
  private readonly w: number;
  private readonly ring: Float64Array;
  private cursor: number = 0;
  private count: number = 0;

  // Precomputed x constants
  private readonly sum_x: number;
  private readonly sum_x2: number;
  private readonly denom: number; // n*sum_x2 - sum_x^2

  // Running sums
  private sum_y: number = 0;
  private sum_xy: number = 0;

  constructor(window: number) {
    this.w = window;
    this.ring = new Float64Array(window);

    // x = [0, 1, ..., w-1]
    let sx = 0, sx2 = 0;
    for (let i = 0; i < window; i++) {
      sx += i;
      sx2 += i * i;
    }
    this.sum_x = sx;
    this.sum_x2 = sx2;
    this.denom = window * sx2 - sx * sx;
  }

  /** Push a new y value. Returns current slope (NaN if window not full). */
  push(y: number): number {
    if (this.count < this.w) {
      // Filling phase: accumulate
      this.ring[this.count] = y;
      this.sum_y += y;
      this.sum_xy += this.count * y;
      this.count++;
      if (this.count < this.w) return NaN;
      // Window just became full
      this.cursor = 0;
      return (this.w * this.sum_xy - this.sum_x * this.sum_y) / this.denom;
    }

    // Steady-state: remove oldest, add newest
    const oldest = this.ring[this.cursor];
    this.ring[this.cursor] = y;
    this.cursor = (this.cursor + 1) % this.w;

    // When we remove the oldest value (which was at position 0) and add new at position w-1,
    // all existing values shift left by 1 position.
    // sum_y update: remove oldest, add newest
    this.sum_y = this.sum_y - oldest + y;

    // sum_xy update: each existing y's x-position decreases by 1,
    // so sum_xy decreases by sum_y_remaining. New y enters at position w-1.
    // sum_xy_new = (sum_xy_old - oldest*0) - (sum_y_old - oldest) + y*(w-1)
    // Simplified: when oldest exits position 0 and all shift left by 1:
    //   sum_xy -= sum_y (shift all x down by 1)
    //   sum_xy += y * 0 (oldest was at 0, contributes 0)
    //   but we already removed oldest from sum_y above
    // Correct derivation:
    //   old sum_xy = 0*y0 + 1*y1 + ... + (w-1)*y_{w-1}
    //   after shift: new sum_xy = 0*y1 + 1*y2 + ... + (w-2)*y_{w-1} + (w-1)*y_new
    //   = (sum_xy - 0*y0) - (y1 + y2 + ... + y_{w-1}) + (w-1)*y_new
    //   = sum_xy - (sum_y - y0) + (w-1)*y_new
    //   But sum_y is already updated (sum_y_new = sum_y_old - y0 + y_new)
    //   = sum_xy_old - (sum_y_old - oldest) + (this.w - 1) * y
    this.sum_xy = this.sum_xy - (this.sum_y - y) + (this.w - 1) * y;
    // Wait, sum_y is already the new value. Let's be precise:
    // Let S_old = sum_y before update = sum_y_new + oldest - y
    // sum_xy_new = sum_xy_old - (S_old - oldest) + (w-1)*y
    //            = sum_xy_old - sum_y_new - oldest + y + oldest + (w-1)*y
    //            = sum_xy_old - sum_y_new + y + (w-1)*y
    //            = sum_xy_old - sum_y_new + w*y
    // Actually let me just recompute to be safe for the first version:
    this.sum_xy = 0;
    for (let i = 0; i < this.w; i++) {
      const idx = (this.cursor + i) % this.w;
      this.sum_xy += i * this.ring[idx];
    }

    return (this.w * this.sum_xy - this.sum_x * this.sum_y) / this.denom;
  }

  reset(): void {
    this.ring.fill(0);
    this.cursor = 0;
    this.count = 0;
    this.sum_y = 0;
    this.sum_xy = 0;
  }
}


/**
 * Rolling robust z-score using median and MAD.
 *
 * z = (x - median) / (1.4826 * MAD)
 * where MAD = median(|x_i - median|)
 *
 * Maintains a sorted array for O(w) median lookup.
 * Reference: eval_harness.py:350-369
 */
export class RollingRobustZScore {
  private readonly w: number;
  private readonly minPeriods: number;
  private readonly ring: Float64Array;
  private readonly sorted: number[];
  private cursor: number = 0;
  private count: number = 0;

  constructor(window: number, minPeriods: number = 30) {
    this.w = window;
    this.minPeriods = minPeriods;
    this.ring = new Float64Array(window);
    this.sorted = [];
  }

  /** Push value, return z-score (0 if insufficient data or zero MAD). */
  push(x: number): number {
    if (this.count < this.w) {
      // Filling phase
      this.ring[this.count] = x;
      this._sortedInsert(x);
      this.count++;
    } else {
      // Remove oldest, add new
      const oldest = this.ring[this.cursor];
      this.ring[this.cursor] = x;
      this.cursor = (this.cursor + 1) % this.w;
      this._sortedRemove(oldest);
      this._sortedInsert(x);
    }

    if (this.count < this.minPeriods) return 0;

    const n = this.sorted.length;
    const med = n % 2 === 1
      ? this.sorted[(n - 1) >> 1]
      : (this.sorted[(n >> 1) - 1] + this.sorted[n >> 1]) / 2;

    // Compute MAD
    const absDevs: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      absDevs[i] = Math.abs(this.sorted[i] - med);
    }
    absDevs.sort((a, b) => a - b);
    const mad = n % 2 === 1
      ? absDevs[(n - 1) >> 1]
      : (absDevs[(n >> 1) - 1] + absDevs[n >> 1]) / 2;

    const scale = 1.4826 * mad;
    if (scale < 1e-12) return 0;

    return (x - med) / scale;
  }

  private _sortedInsert(val: number): void {
    // Binary search for insertion point
    let lo = 0, hi = this.sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.sorted[mid] < val) lo = mid + 1;
      else hi = mid;
    }
    this.sorted.splice(lo, 0, val);
  }

  private _sortedRemove(val: number): void {
    let lo = 0, hi = this.sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.sorted[mid] < val) lo = mid + 1;
      else hi = mid;
    }
    if (lo < this.sorted.length && this.sorted[lo] === val) {
      this.sorted.splice(lo, 1);
    }
  }

  reset(): void {
    this.ring.fill(0);
    this.sorted.length = 0;
    this.cursor = 0;
    this.count = 0;
  }
}
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation (no errors).

**Step 3: Commit**

```bash
git add frontend/src/experiment-math.ts
git commit -m "feat: add incremental OLS slope + robust z-score utilities for experiment engine"
```

---

### Task 2: PFP Signal — Incremental Computation

Port the PFP experiment from `agents/pfp/run.py` to an incremental TypeScript class that processes one `GridBucketRow` map per bin.

**Files:**
- Create: `frontend/src/experiment-pfp.ts`

**Step 1: Create the PFP signal class**

```typescript
// frontend/src/experiment-pfp.ts

/**
 * Pressure Front Propagation (PFP) — incremental per-bin computation.
 *
 * Detects when inner-tick velocity leads outer-tick velocity,
 * signaling aggressive directional intent propagating from BBO outward.
 *
 * Reference: agents/pfp/run.py
 */

interface GridBucketLike {
  k: number;
  v_add: number;
  v_fill: number;
  v_pull: number;
}

// Zone definitions: k values (NOT column indices — frontend uses k directly)
const INNER_BID_K = [-3, -2, -1];
const INNER_ASK_K = [1, 2, 3];
const OUTER_BID_K = [-12, -11, -10, -9, -8, -7, -6, -5];
const OUTER_ASK_K = [5, 6, 7, 8, 9, 10, 11, 12];

const LAG_BINS = 5;
const EMA_ALPHA = 0.1;
const EPS = 1e-12;
const ADD_WEIGHT = 0.6;
const PULL_WEIGHT = 0.4;

export class PFPSignal {
  private binCount = 0;

  // Lag buffers for outer zone intensities (ring buffer, 4 channels)
  // [add_outer_bid, add_outer_ask, pull_outer_bid, pull_outer_ask]
  private lagRing: Float64Array;
  private lagCursor = 0;

  // EMA accumulators: [lagged_product, unlagged_product] × 4 channels
  // Channels: add_bid, add_ask, pull_bid, pull_ask
  private emaLagged = new Float64Array(4);
  private emaUnlagged = new Float64Array(4);

  constructor() {
    this.lagRing = new Float64Array(LAG_BINS * 4);
  }

  /** Returns true when warmup is complete (>= LAG_BINS bins processed). */
  get warm(): boolean {
    return this.binCount >= LAG_BINS;
  }

  /**
   * Process one bin's grid data. Returns the PFP signal (positive = bullish).
   * Returns 0 during warmup.
   */
  update(grid: Map<number, GridBucketLike>): number {
    // Compute zone intensities
    const iInnerBid = this._zoneMean(grid, INNER_BID_K, 'add_fill');
    const iInnerAsk = this._zoneMean(grid, INNER_ASK_K, 'add_fill');
    const iOuterBid = this._zoneMean(grid, OUTER_BID_K, 'add_fill');
    const iOuterAsk = this._zoneMean(grid, OUTER_ASK_K, 'add_fill');

    const pInnerBid = this._zoneMean(grid, INNER_BID_K, 'pull');
    const pInnerAsk = this._zoneMean(grid, INNER_ASK_K, 'pull');
    const pOuterBid = this._zoneMean(grid, OUTER_BID_K, 'pull');
    const pOuterAsk = this._zoneMean(grid, OUTER_ASK_K, 'pull');

    // Get lagged outer values (from LAG_BINS ago)
    const lagIdx = this.lagCursor * 4;
    const laggedOuterBidAdd = this.binCount >= LAG_BINS ? this.lagRing[lagIdx + 0] : 0;
    const laggedOuterAskAdd = this.binCount >= LAG_BINS ? this.lagRing[lagIdx + 1] : 0;
    const laggedOuterBidPull = this.binCount >= LAG_BINS ? this.lagRing[lagIdx + 2] : 0;
    const laggedOuterAskPull = this.binCount >= LAG_BINS ? this.lagRing[lagIdx + 3] : 0;

    // Store current outer intensities in lag ring
    this.lagRing[lagIdx + 0] = iOuterBid;
    this.lagRing[lagIdx + 1] = iOuterAsk;
    this.lagRing[lagIdx + 2] = pOuterBid;
    this.lagRing[lagIdx + 3] = pOuterAsk;
    this.lagCursor = (this.lagCursor + 1) % LAG_BINS;

    // Compute cross-products
    const prodLagged = [
      iInnerBid * laggedOuterBidAdd,   // add bid lagged
      iInnerAsk * laggedOuterAskAdd,   // add ask lagged
      pInnerBid * laggedOuterBidPull,  // pull bid lagged
      pInnerAsk * laggedOuterAskPull,  // pull ask lagged
    ];
    const prodUnlagged = [
      iInnerBid * iOuterBid,
      iInnerAsk * iOuterAsk,
      pInnerBid * pOuterBid,
      pInnerAsk * pOuterAsk,
    ];

    // Update EMAs
    const a = EMA_ALPHA;
    const b = 1 - a;
    for (let ch = 0; ch < 4; ch++) {
      if (this.binCount === 0) {
        this.emaLagged[ch] = prodLagged[ch];
        this.emaUnlagged[ch] = prodUnlagged[ch];
      } else {
        this.emaLagged[ch] = a * prodLagged[ch] + b * this.emaLagged[ch];
        this.emaUnlagged[ch] = a * prodUnlagged[ch] + b * this.emaUnlagged[ch];
      }
    }

    this.binCount++;

    if (!this.warm) return 0;

    // Lead-lag ratios
    const leadBidAdd = this.emaLagged[0] / (this.emaUnlagged[0] + EPS);
    const leadAskAdd = this.emaLagged[1] / (this.emaUnlagged[1] + EPS);
    const leadBidPull = this.emaLagged[2] / (this.emaUnlagged[2] + EPS);
    const leadAskPull = this.emaLagged[3] / (this.emaUnlagged[3] + EPS);

    // Directional signals
    const addSignal = leadBidAdd - leadAskAdd;
    const pullSignal = leadAskPull - leadBidPull;

    return ADD_WEIGHT * addSignal + PULL_WEIGHT * pullSignal;
  }

  private _zoneMean(
    grid: Map<number, GridBucketLike>,
    kValues: number[],
    channel: 'add_fill' | 'pull',
  ): number {
    let sum = 0;
    let count = 0;
    for (const k of kValues) {
      const row = grid.get(k);
      if (row) {
        sum += channel === 'add_fill'
          ? (row.v_add + row.v_fill)
          : row.v_pull;
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  }

  reset(): void {
    this.binCount = 0;
    this.lagRing.fill(0);
    this.lagCursor = 0;
    this.emaLagged.fill(0);
    this.emaUnlagged.fill(0);
  }
}
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 3: Commit**

```bash
git add frontend/src/experiment-pfp.ts
git commit -m "feat: add incremental PFP signal computation for projection bands"
```

---

### Task 3: ADS Signal — Incremental Computation

Port the ADS experiment from `agents/ads/run.py` to incremental TypeScript.

**Files:**
- Create: `frontend/src/experiment-ads.ts`

**Step 1: Create the ADS signal class**

```typescript
// frontend/src/experiment-ads.ts

/**
 * Asymmetric Derivative Slope (ADS) — incremental per-bin computation.
 *
 * Multi-scale OLS slope of bid/ask velocity asymmetry, robust z-scored,
 * tanh-compressed, and blended across 3 window scales.
 *
 * Reference: agents/ads/run.py
 */

import { IncrementalOLSSlope, RollingRobustZScore } from './experiment-math';

interface GridBucketLike {
  k: number;
  v_add: number;
  v_pull: number;
}

// Band definitions: k values
const BANDS = [
  {
    name: 'inner',
    bidK: [-3, -2, -1],
    askK: [1, 2, 3],
    width: 3,
  },
  {
    name: 'mid',
    bidK: [-11, -10, -9, -8, -7, -6, -5, -4],
    askK: [4, 5, 6, 7, 8, 9, 10, 11],
    width: 8,
  },
  {
    name: 'outer',
    bidK: [-23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12],
    askK: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    width: 12,
  },
];

// Precomputed bandwidth weights: 1/sqrt(width), normalized
const RAW_WEIGHTS = BANDS.map(b => 1 / Math.sqrt(b.width));
const WEIGHT_SUM = RAW_WEIGHTS.reduce((a, b) => a + b, 0);
const BAND_WEIGHTS = RAW_WEIGHTS.map(w => w / WEIGHT_SUM);

const SLOPE_WINDOWS = [10, 25, 50];
const ZSCORE_WINDOW = 200;
const BLEND_WEIGHTS = [0.40, 0.35, 0.25];
const BLEND_SCALE = 3.0;
const WARMUP_BINS = 200; // deepest window (zscore) needs 200 bins

export class ADSSignal {
  private binCount = 0;

  // 3 OLS slope trackers (one per slope window)
  private readonly olsSlopes: IncrementalOLSSlope[];

  // 3 robust z-score trackers (one per slope window)
  private readonly zscores: RollingRobustZScore[];

  constructor() {
    this.olsSlopes = SLOPE_WINDOWS.map(w => new IncrementalOLSSlope(w));
    this.zscores = SLOPE_WINDOWS.map(() => new RollingRobustZScore(ZSCORE_WINDOW));
  }

  get warm(): boolean {
    return this.binCount >= WARMUP_BINS;
  }

  /**
   * Process one bin. Returns ADS signal in ~[-1, +1] (positive = bullish).
   * Returns 0 during warmup.
   */
  update(grid: Map<number, GridBucketLike>): number {
    // Step 1: Compute combined asymmetry
    let combined = 0;
    for (let b = 0; b < BANDS.length; b++) {
      const band = BANDS[b];

      const addBid = this._kMean(grid, band.bidK, 'v_add');
      const addAsk = this._kMean(grid, band.askK, 'v_add');
      const pullBid = this._kMean(grid, band.bidK, 'v_pull');
      const pullAsk = this._kMean(grid, band.askK, 'v_pull');

      const addAsym = addBid - addAsk;   // positive = more adding on bid = bullish
      const pullAsym = pullAsk - pullBid; // positive = more pulling on ask = bullish

      combined += BAND_WEIGHTS[b] * (addAsym + pullAsym);
    }

    // Step 2: Push through OLS slopes → z-scores → tanh blend
    let signal = 0;
    for (let i = 0; i < SLOPE_WINDOWS.length; i++) {
      const slope = this.olsSlopes[i].push(combined);
      const z = isNaN(slope) ? 0 : this.zscores[i].push(slope);
      signal += BLEND_WEIGHTS[i] * Math.tanh(z / BLEND_SCALE);
    }

    this.binCount++;
    return this.warm ? signal : 0;
  }

  private _kMean(
    grid: Map<number, GridBucketLike>,
    kValues: number[],
    field: 'v_add' | 'v_pull',
  ): number {
    let sum = 0;
    let count = 0;
    for (const k of kValues) {
      const row = grid.get(k);
      if (row) {
        sum += row[field];
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  }

  reset(): void {
    this.binCount = 0;
    for (const ols of this.olsSlopes) ols.reset();
    for (const z of this.zscores) z.reset();
  }
}
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 3: Commit**

```bash
git add frontend/src/experiment-ads.ts
git commit -m "feat: add incremental ADS signal computation for projection bands"
```

---

### Task 4: ERD Signal — Incremental Computation

Port the ERD experiment from `agents/erd/run.py` to incremental TypeScript.

**Files:**
- Create: `frontend/src/experiment-erd.ts`

**Step 1: Create the ERD signal class**

```typescript
// frontend/src/experiment-erd.ts

/**
 * Entropy Regime Detector (ERD) — incremental per-bin computation.
 *
 * Detects entropy spikes in the spectrum state field. When disorder
 * increases above baseline, the directional asymmetry of entropy
 * (above vs below spot) provides directional bias.
 *
 * Uses Variant B: signal = entropy_asym * max(0, z_H - 0.5)
 *
 * Reference: agents/erd/run.py
 */

import { RollingRobustZScore } from './experiment-math';

interface GridBucketLike {
  k: number;
  spectrum_state_code: number;
  spectrum_score: number;
}

const ZSCORE_WINDOW = 100;
const SPIKE_FLOOR = 0.5;
const WARMUP_BINS = 100;

export class ERDSignal {
  private binCount = 0;
  private readonly zscore: RollingRobustZScore;

  constructor() {
    this.zscore = new RollingRobustZScore(ZSCORE_WINDOW);
  }

  get warm(): boolean {
    return this.binCount >= WARMUP_BINS;
  }

  /**
   * Process one bin. Returns ERD signal (positive = bullish).
   * Returns 0 during warmup.
   */
  update(grid: Map<number, GridBucketLike>): number {
    // Count states across regions
    let nPFull = 0, nVFull = 0, nNFull = 0;
    let nPAbove = 0, nVAbove = 0, nNAbove = 0;
    let nPBelow = 0, nVBelow = 0, nNBelow = 0;
    let totalFull = 0, totalAbove = 0, totalBelow = 0;

    for (const [k, row] of grid) {
      const sc = row.spectrum_state_code;
      totalFull++;
      if (sc === 1) nPFull++;
      else if (sc === -1) nVFull++;
      else nNFull++;

      if (k > 0) {
        totalAbove++;
        if (sc === 1) nPAbove++;
        else if (sc === -1) nVAbove++;
        else nNAbove++;
      } else if (k < 0) {
        totalBelow++;
        if (sc === 1) nPBelow++;
        else if (sc === -1) nVBelow++;
        else nNBelow++;
      }
      // k === 0 (spot) only counts in full
    }

    // Shannon entropy
    const hFull = this._entropy3(nPFull, nVFull, nNFull, totalFull);
    const hAbove = this._entropy3(nPAbove, nVAbove, nNAbove, totalAbove);
    const hBelow = this._entropy3(nPBelow, nVBelow, nNBelow, totalBelow);

    // Entropy asymmetry: positive = more disorder above spot
    const entropyAsym = hAbove - hBelow;

    // Robust z-score of full entropy
    const zH = this.zscore.push(hFull);

    // Spike gate
    const spikeGate = Math.max(0, zH - SPIKE_FLOOR);

    this.binCount++;

    // Variant B signal
    return this.warm ? entropyAsym * spikeGate : 0;
  }

  private _entropy3(nP: number, nV: number, nN: number, total: number): number {
    if (total === 0) return 0;
    let h = 0;
    for (const count of [nP, nV, nN]) {
      if (count > 0) {
        const p = count / total;
        h -= p * Math.log2(p + 1e-12);
      }
    }
    return h;
  }

  reset(): void {
    this.binCount = 0;
    this.zscore.reset();
  }
}
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 3: Commit**

```bash
git add frontend/src/experiment-erd.ts
git commit -m "feat: add incremental ERD signal computation for projection bands"
```

---

### Task 5: ExperimentEngine — Composite Signal Blender

Orchestrator class that owns PFP + ADS + ERD instances, handles dynamic warmup re-weighting, and produces the composite directional signal.

**Files:**
- Create: `frontend/src/experiment-engine.ts`

**Step 1: Create the engine**

```typescript
// frontend/src/experiment-engine.ts

/**
 * Composite experiment engine: blends PFP, ADS, ERD into one
 * directional signal for projection band rendering.
 *
 * Dynamic warmup: only warm signals contribute. Weights renormalize
 * among warm signals so composite stays in a consistent range.
 *
 * PFP comes online at 500ms, ERD at 10s, ADS at 20s.
 */

import { PFPSignal } from './experiment-pfp';
import { ADSSignal } from './experiment-ads';
import { ERDSignal } from './experiment-erd';

/** Raw blend weights (sum to 1.0 when all warm). */
const W_PFP = 0.40;
const W_ADS = 0.35;
const W_ERD = 0.25;

export interface CompositeSignal {
  /** Blended directional signal (positive = bullish, negative = bearish). */
  composite: number;
  /** Individual experiment signals. */
  pfp: number;
  ads: number;
  erd: number;
  /** Fraction of experiments warm (0..1). 0.33 = only PFP, 1.0 = all 3. */
  warmupFraction: number;
}

export interface GridBucketMinimal {
  k: number;
  v_add: number;
  v_fill: number;
  v_pull: number;
  spectrum_state_code: number;
  spectrum_score: number;
}

export class ExperimentEngine {
  private readonly pfp = new PFPSignal();
  private readonly ads = new ADSSignal();
  private readonly erd = new ERDSignal();

  /**
   * Process one bin's grid data. Returns composite directional signal.
   */
  update(grid: Map<number, GridBucketMinimal>): CompositeSignal {
    const pfpVal = this.pfp.update(grid);
    const adsVal = this.ads.update(grid);
    const erdVal = this.erd.update(grid);

    // Dynamic warmup weighting
    let totalWeight = 0;
    let composite = 0;
    let warmCount = 0;

    if (this.pfp.warm) {
      composite += W_PFP * pfpVal;
      totalWeight += W_PFP;
      warmCount++;
    }
    if (this.ads.warm) {
      composite += W_ADS * adsVal;
      totalWeight += W_ADS;
      warmCount++;
    }
    if (this.erd.warm) {
      composite += W_ERD * erdVal;
      totalWeight += W_ERD;
      warmCount++;
    }

    // Renormalize so composite has consistent scale
    if (totalWeight > 0) {
      composite /= totalWeight;
    }

    return {
      composite,
      pfp: pfpVal,
      ads: adsVal,
      erd: erdVal,
      warmupFraction: warmCount / 3,
    };
  }

  reset(): void {
    this.pfp.reset();
    this.ads.reset();
    this.erd.reset();
  }
}
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 3: Commit**

```bash
git add frontend/src/experiment-engine.ts
git commit -m "feat: add ExperimentEngine composite signal blender"
```

---

### Task 6: Wire ExperimentEngine Into Data Pipeline

Connect the engine to the per-bin data flow in `vacuum-pressure.ts`.

**Files:**
- Modify: `frontend/src/vacuum-pressure.ts` (lines 110, 577-684, 1424-1452, 1511-1550)

**Step 1: Add import and module-level state**

At top of `vacuum-pressure.ts` (after line 31, the `apache-arrow` import), add:

```typescript
import { ExperimentEngine, CompositeSignal } from './experiment-engine';
```

After line 120 (after `columnTimestamps` declaration), add:

```typescript
// Experiment engine for projection bands
let experimentEngine: ExperimentEngine | null = null;
let currentCompositeSignal: CompositeSignal = {
  composite: 0, pfp: 0, ads: 0, erd: 0, warmupFraction: 0,
};
```

**Step 2: Instantiate engine in applyRuntimeConfig**

In `applyRuntimeConfig()` (line 1431, after `resetHeatmapBuffers(cfg.grid_radius_ticks)`), add:

```typescript
  experimentEngine = new ExperimentEngine();
```

**Step 3: Update engine in pushHeatmapColumnFromGrid**

In `pushHeatmapColumnFromGrid()`, after the pixel-writing loop (after line 683, before the closing `}`), add:

```typescript
  // Update experiment engine for projection bands
  if (experimentEngine) {
    currentCompositeSignal = experimentEngine.update(grid);
  }
```

**Step 4: Reset engine in resetStreamState**

In `resetStreamState()` (after line 1524, the `currentBinEndNs = 0n` line), add:

```typescript
  if (experimentEngine) experimentEngine.reset();
  currentCompositeSignal = {
    composite: 0, pfp: 0, ads: 0, erd: 0, warmupFraction: 0,
  };
```

**Step 5: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 6: Commit**

```bash
git add frontend/src/vacuum-pressure.ts
git commit -m "feat: wire ExperimentEngine into per-bin data pipeline"
```

---

### Task 7: Projection Band Rendering

Add the purple Gaussian band rendering in the projection zone, replacing the blank tint.

**Files:**
- Modify: `frontend/src/vacuum-pressure.ts` (renderHeatmap function, ~line 968-996)

**Step 1: Add the renderProjectionBands function**

Add this function before `renderHeatmap` (before line 901):

```typescript
/** Projection band horizons and their visual properties. */
const PROJECTION_HORIZONS = [
  { ms: 250,  label: '250ms', spreadTicks: 2, alpha: 1.0 },
  { ms: 500,  label: '500ms', spreadTicks: 4, alpha: 0.8 },
  { ms: 1000, label: '1s',    spreadTicks: 6, alpha: 0.6 },
  { ms: 2500, label: '2.5s',  spreadTicks: 8, alpha: 0.4 },
];
const BAND_HALF_WIDTH = 6;   // ticks from center
const BAND_SIGMA = 2.5;      // Gaussian sigma in ticks
const BAND_SIGMA2_INV = 1 / (2 * BAND_SIGMA * BAND_SIGMA);

/** Projection band pixel buffer: 4 columns × HMAP_LEVELS rows × RGBA. */
let projBandData: Uint8ClampedArray | null = null;
let projBandOffscreen: HTMLCanvasElement | null = null;

/**
 * Render purple directional pressure bands in the projection zone.
 *
 * Each of 4 horizon columns shows a Gaussian band centered at
 * spot + composite * spreadTicks, with confidence-dependent alpha fade.
 */
function renderProjectionBands(
  ctx: CanvasRenderingContext2D,
  signal: CompositeSignal,
  spotRow: number,
  ch: number,
  dataWidth: number,
  cw: number,
): void {
  const h = HMAP_LEVELS;
  const nHorizons = PROJECTION_HORIZONS.length;

  // Lazy-init projection pixel buffer
  if (!projBandData || projBandData.length !== nHorizons * h * 4) {
    projBandData = new Uint8ClampedArray(nHorizons * h * 4);
    projBandOffscreen = document.createElement('canvas');
    projBandOffscreen.width = nHorizons;
    projBandOffscreen.height = h;
  }

  // Clear to transparent dark
  for (let i = 0; i < projBandData.length; i += 4) {
    projBandData[i] = 8;
    projBandData[i + 1] = 8;
    projBandData[i + 2] = 12;
    projBandData[i + 3] = 255;
  }

  // Only render if we have some signal
  if (signal.warmupFraction > 0 && spotRow >= 0 && spotRow < h) {
    const composite = signal.composite;

    for (let hi = 0; hi < nHorizons; hi++) {
      const horizon = PROJECTION_HORIZONS[hi];
      const bandCenter = spotRow - composite * horizon.spreadTicks;
      const confAlpha = horizon.alpha * signal.warmupFraction;

      for (let row = 0; row < h; row++) {
        const d = row - bandCenter;
        if (Math.abs(d) > BAND_HALF_WIDTH) continue;

        const gaussianI = Math.exp(-d * d * BAND_SIGMA2_INV);
        const intensity = gaussianI * confAlpha;

        if (intensity < 0.01) continue;

        const idx = (row * nHorizons + hi) * 4;
        // Purple: R=0.35, G=0.08, B=0.55 scaled by intensity
        projBandData[idx]     = Math.round(0.35 * intensity * 255);
        projBandData[idx + 1] = Math.round(0.08 * intensity * 255);
        projBandData[idx + 2] = Math.round(0.55 * intensity * 255);
        projBandData[idx + 3] = 255;
      }
    }
  }

  // Blit to canvas
  if (!projBandOffscreen) return;
  const offCtx = projBandOffscreen.getContext('2d')!;
  const imgData = new ImageData(projBandData, nHorizons, h);
  offCtx.putImageData(imgData, 0, 0);

  // Scale the 4-pixel-wide buffer into the projection zone
  const zoneWidth = cw - dataWidth;
  if (zoneWidth <= 0) return;

  ctx.imageSmoothingEnabled = false;

  // Map projection buffer rows to visible viewport rows
  const srcH = visibleRows();
  const dy = ((0 - vpY) / srcH) * ch;
  const dh = (h / srcH) * ch;

  ctx.drawImage(
    projBandOffscreen,
    0, Math.max(0, vpY), nHorizons, Math.min(h, vpY + srcH) - Math.max(0, vpY),
    dataWidth, Math.max(0, dy), zoneWidth, Math.min(ch, dh),
  );
}
```

**Step 2: Replace the blank projection zone tint in renderHeatmap**

In `renderHeatmap()`, replace lines 968-972 (the projection zone tint block):

```typescript
  // Projection zone: subtle background tint to distinguish from data area
  if (dataWidth < cw) {
    ctx.fillStyle = 'rgba(15, 15, 25, 0.5)';
    ctx.fillRect(dataWidth, 0, cw - dataWidth, ch);
  }
```

With:

```typescript
  // Projection zone: render experiment-driven directional bands
  if (dataWidth < cw) {
    const lastSpotRow = spotTrail[HMAP_HISTORY - 1];
    renderProjectionBands(
      ctx, currentCompositeSignal,
      lastSpotRow !== null ? lastSpotRow : GRID_RADIUS_TICKS,
      ch, dataWidth, cw,
    );
  }
```

**Step 3: Extend spot line into projection zone**

After the spot dashed horizontal line (around line 1047, after `ctx.setLineDash([])` in the spot rendering block), add:

```typescript
      // Extend spot into projection zone as dashed line
      if (dataWidth < cw) {
        ctx.strokeStyle = 'rgba(0, 255, 170, 0.3)';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([4, 6]);
        ctx.beginPath();
        ctx.moveTo(dataWidth, y);
        ctx.lineTo(cw, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
```

**Step 4: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 5: Commit**

```bash
git add frontend/src/vacuum-pressure.ts
git commit -m "feat: render purple projection bands with Gaussian falloff in projection zone"
```

---

### Task 8: Projection Time Axis Labels

Add horizon labels ("250ms", "500ms", "1s", "2.5s") to the time axis canvas in the projection zone area.

**Files:**
- Modify: `frontend/src/vacuum-pressure.ts` (renderTimeAxis function, ~line 1396-1415)

**Step 1: Replace the projection zone time axis block**

In `renderTimeAxis()`, replace lines 1396-1415 (the projection zone tint + "NOW" block):

```typescript
  // Projection zone tint + "NOW" marker on time axis
  if (dataWidth < cw) {
    ctx.fillStyle = 'rgba(15, 15, 25, 0.5)';
    ctx.fillRect(dataWidth, 0, cw - dataWidth, ch);

    ctx.strokeStyle = 'rgba(100, 100, 150, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(dataWidth, 0);
    ctx.lineTo(dataWidth, ch);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(100, 100, 150, 0.4)';
    ctx.font = '7px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('NOW', dataWidth + 3, 5);
  }
```

With:

```typescript
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
    const nHorizons = 4;
    const colWidth = zoneWidth / nHorizons;
    const labels = ['250ms', '500ms', '1s', '2.5s'];

    ctx.fillStyle = 'rgba(140, 100, 180, 0.6)'; // purple-tinted text
    ctx.font = '7px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i < nHorizons; i++) {
      const x = dataWidth + colWidth * (i + 0.5);
      ctx.fillText(labels[i], x, 14);
    }
  }
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 3: Commit**

```bash
git add frontend/src/vacuum-pressure.ts
git commit -m "feat: add horizon labels to projection zone time axis"
```

---

### Task 9: Legend Update + HTML Polish

Update the legend in `vacuum-pressure.html` to include projection band description.

**Files:**
- Modify: `frontend/vacuum-pressure.html`

**Step 1: Add projection band legend entry**

In the legend section of `vacuum-pressure.html`, find the existing legend items (green "Pressure", grey "Neutral", black "Vacuum") and add after the last one:

```html
<span style="display:inline-block;width:10px;height:10px;background:#59148d;border-radius:2px;vertical-align:middle;margin-right:3px"></span>
<span style="font-size:10px;color:#888;margin-right:10px">Projected direction</span>
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/logan.robbins/research/spymaster/frontend && npx tsc --noEmit`
Expected: Clean compilation.

**Step 3: Commit**

```bash
git add frontend/vacuum-pressure.html
git commit -m "feat: add projection band legend entry"
```

---

### Task 10: Visual Verification

Launch the application and visually verify the projection bands render correctly.

**Step 1: Ensure backend is running**

```bash
lsof -iTCP:8002 -sTCP:LISTEN -P 2>/dev/null
```

If not running:
```bash
kill $(lsof -t -iTCP:8002) 2>/dev/null
cd /Users/logan.robbins/research/spymaster/backend
nohup uv run scripts/run_vacuum_pressure.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --port 8002 \
  --start-time 09:00 > /tmp/vp_preprod.log 2>&1 &
```

**Step 2: Ensure frontend dev server is running**

```bash
lsof -iTCP:5174 -sTCP:LISTEN -P 2>/dev/null
```

If not running:
```bash
kill $(lsof -t -iTCP:5174) 2>/dev/null
cd /Users/logan.robbins/research/spymaster/frontend
nohup npm run dev > /tmp/frontend_vp.log 2>&1 &
```

**Step 3: Open in browser and verify**

URL: `http://localhost:5174/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&start_time=09:00`

Verify:
- [ ] Projection zone shows purple bands (not blank tint)
- [ ] Bands appear after ~500ms (PFP warmup)
- [ ] Bands shift directionally (above/below spot) based on signal
- [ ] Bands fade across horizons (left-to-right in zone = dimmer)
- [ ] Horizon labels visible on time axis ("250ms", "500ms", "1s", "2.5s")
- [ ] "NOW" separator visible at data/projection boundary
- [ ] Spot dashed line extends into projection zone
- [ ] Green pressure heatmap unaffected (no visual regression)
- [ ] Signal panel (right column) at 256px width
- [ ] No console errors

**Step 4: Take screenshot for user review**

---

### Task 11: README Update

Update README.md to document the projection band feature.

**Files:**
- Modify: `README.md`

**Step 1: Add projection bands section**

After the "Stream Payload" section, add:

```markdown
## Projection Bands

The frontend computes three experiment signals in-browser from the Arrow grid data:

- **PFP** (Pressure Front Propagation): Inner/outer velocity lead-lag detection
- **ADS** (Asymmetric Derivative Slope): Multi-scale OLS slope of bid/ask velocity asymmetry
- **ERD** (Entropy Regime Detector): Shannon entropy spike detection with directional gating

Signals are blended (PFP=0.40, ADS=0.35, ERD=0.25) into a composite directional signal rendered as purple Gaussian bands at 4 horizons (250ms, 500ms, 1s, 2.5s) in the right 15% of the heatmap.

Band interpretation:
- Band skews **above** spot = bullish prediction
- Band skews **below** spot = bearish prediction
- Brightness = signal strength × horizon confidence (fades with longer horizon)

Warmup: PFP at 500ms, ERD at 10s, ADS at 20s. Bands appear progressively as each signal comes online.

Source: `frontend/src/experiment-engine.ts`, `experiment-pfp.ts`, `experiment-ads.ts`, `experiment-erd.ts`, `experiment-math.ts`
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add projection bands section to README"
```
