/**
 * Pressure Front Propagation (PFP) — incremental per-bin computation.
 *
 * Detects when inner-tick velocity leads outer-tick velocity,
 * signaling aggressive directional intent propagating from BBO outward.
 *
 * Signal: 0.6 * (leadBidAdd - leadAskAdd) + 0.4 * (leadAskPull - leadBidPull)
 *
 * Reference: agents/pfp/run.py
 */

// Zone definitions by k value (NOT column index — frontend uses k directly)
const INNER_BID_K = [-3, -2, -1];
const INNER_ASK_K = [1, 2, 3];
const OUTER_BID_K = [-12, -11, -10, -9, -8, -7, -6, -5];
const OUTER_ASK_K = [5, 6, 7, 8, 9, 10, 11, 12];

const LAG_MS = 500;
const EMA_ALPHA = 0.1;
const EPS = 1e-12;
const ADD_WEIGHT = 0.6;
const PULL_WEIGHT = 0.4;

export interface PFPBucketLike {
  k: number;
  v_add: number;
  v_fill: number;
  v_pull: number;
}

export class PFPSignal {
  private binCount = 0;
  private readonly lagBins: number;

  // Lag ring: lagBins × 4 channels
  // Channels: [add_outer_bid, add_outer_ask, pull_outer_bid, pull_outer_ask]
  private readonly lagRing: Float64Array;
  private lagCursor = 0;

  // EMA accumulators per channel
  // Channels: [add_bid, add_ask, pull_bid, pull_ask]
  private readonly emaLagged = new Float64Array(4);
  private readonly emaUnlagged = new Float64Array(4);

  constructor(cellWidthMs: number) {
    if (!Number.isFinite(cellWidthMs) || cellWidthMs <= 0) {
      throw new Error(`PFPSignal requires positive cellWidthMs, got ${cellWidthMs}`);
    }
    this.lagBins = Math.max(1, Math.ceil(LAG_MS / cellWidthMs));
    this.lagRing = new Float64Array(this.lagBins * 4);
  }

  get warm(): boolean {
    return this.binCount >= this.lagBins;
  }

  /**
   * Process one bin's grid data. Returns PFP signal (positive = bullish).
   * Returns 0 during warmup.
   */
  update(grid: Map<number, PFPBucketLike>): number {
    // Zone intensities: add+fill channel
    const iInnerBid = zoneMean(grid, INNER_BID_K, addFillGetter);
    const iInnerAsk = zoneMean(grid, INNER_ASK_K, addFillGetter);
    const iOuterBid = zoneMean(grid, OUTER_BID_K, addFillGetter);
    const iOuterAsk = zoneMean(grid, OUTER_ASK_K, addFillGetter);

    // Zone intensities: pull channel
    const pInnerBid = zoneMean(grid, INNER_BID_K, pullGetter);
    const pInnerAsk = zoneMean(grid, INNER_ASK_K, pullGetter);
    const pOuterBid = zoneMean(grid, OUTER_BID_K, pullGetter);
    const pOuterAsk = zoneMean(grid, OUTER_ASK_K, pullGetter);

    // Get lagged outer values (from lagBins ago)
    const lagIdx = this.lagCursor * 4;
    const lagOuterBidAdd = this.binCount >= this.lagBins ? this.lagRing[lagIdx + 0] : 0;
    const lagOuterAskAdd = this.binCount >= this.lagBins ? this.lagRing[lagIdx + 1] : 0;
    const lagOuterBidPull = this.binCount >= this.lagBins ? this.lagRing[lagIdx + 2] : 0;
    const lagOuterAskPull = this.binCount >= this.lagBins ? this.lagRing[lagIdx + 3] : 0;

    // Store current outer intensities in lag ring (overwrites oldest)
    this.lagRing[lagIdx + 0] = iOuterBid;
    this.lagRing[lagIdx + 1] = iOuterAsk;
    this.lagRing[lagIdx + 2] = pOuterBid;
    this.lagRing[lagIdx + 3] = pOuterAsk;
    this.lagCursor = (this.lagCursor + 1) % this.lagBins;

    // Cross-products: lagged and unlagged
    const prodLagged = [
      iInnerBid * lagOuterBidAdd,
      iInnerAsk * lagOuterAskAdd,
      pInnerBid * lagOuterBidPull,
      pInnerAsk * lagOuterAskPull,
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

    const addSignal = leadBidAdd - leadAskAdd;
    const pullSignal = leadAskPull - leadBidPull;

    return ADD_WEIGHT * addSignal + PULL_WEIGHT * pullSignal;
  }

  reset(): void {
    this.binCount = 0;
    this.lagRing.fill(0);
    this.lagCursor = 0;
    this.emaLagged.fill(0);
    this.emaUnlagged.fill(0);
  }
}

// Helpers
function addFillGetter(row: PFPBucketLike): number {
  return row.v_add + row.v_fill;
}

function pullGetter(row: PFPBucketLike): number {
  return row.v_pull;
}

function zoneMean(
  grid: Map<number, PFPBucketLike>,
  kValues: number[],
  getter: (row: PFPBucketLike) => number,
): number {
  let sum = 0;
  let count = 0;
  for (const k of kValues) {
    const row = grid.get(k);
    if (row) {
      sum += getter(row);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}
