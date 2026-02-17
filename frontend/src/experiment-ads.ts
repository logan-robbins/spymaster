/**
 * Asymmetric Derivative Slope (ADS) — incremental per-bin computation.
 *
 * Multi-scale OLS slope of bid/ask velocity asymmetry across three spatial
 * bands (inner/mid/outer), robust z-scored, tanh-compressed, and blended.
 *
 * signal = 0.40 * tanh(z10/3) + 0.35 * tanh(z25/3) + 0.25 * tanh(z50/3)
 *
 * Reference: agents/ads/run.py
 */

import { IncrementalOLSSlope, RollingRobustZScore } from './experiment-math';

export interface ADSBucketLike {
  k: number;
  v_add: number;
  v_pull: number;
}

// Band definitions by k values
const BANDS = [
  {
    bidK: [-3, -2, -1],
    askK: [1, 2, 3],
    width: 3,
  },
  {
    bidK: [-11, -10, -9, -8, -7, -6, -5, -4],
    askK: [4, 5, 6, 7, 8, 9, 10, 11],
    width: 8,
  },
  {
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
const WARMUP_BINS = 200;

export class ADSSignal {
  private binCount = 0;

  // One OLS slope tracker + z-score tracker per slope window
  private readonly olsSlopes: IncrementalOLSSlope[];
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
  update(grid: Map<number, ADSBucketLike>): number {
    // Step 1: Compute combined asymmetry
    let combined = 0;
    for (let b = 0; b < BANDS.length; b++) {
      const band = BANDS[b];

      const addBid = kMean(grid, band.bidK, 'v_add');
      const addAsk = kMean(grid, band.askK, 'v_add');
      const pullBid = kMean(grid, band.bidK, 'v_pull');
      const pullAsk = kMean(grid, band.askK, 'v_pull');

      const addAsym = addBid - addAsk;    // positive = more adding on bid = bullish
      const pullAsym = pullAsk - pullBid;  // positive = more pulling on ask = bullish

      combined += BAND_WEIGHTS[b] * (addAsym + pullAsym);
    }

    // Step 2: OLS slopes → z-scores → tanh blend
    let signal = 0;
    for (let i = 0; i < SLOPE_WINDOWS.length; i++) {
      const slope = this.olsSlopes[i].push(combined);
      const z = isNaN(slope) ? 0 : this.zscores[i].push(slope);
      signal += BLEND_WEIGHTS[i] * Math.tanh(z / BLEND_SCALE);
    }

    this.binCount++;
    return this.warm ? signal : 0;
  }

  reset(): void {
    this.binCount = 0;
    for (const ols of this.olsSlopes) ols.reset();
    for (const z of this.zscores) z.reset();
  }
}

function kMean(
  grid: Map<number, ADSBucketLike>,
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
