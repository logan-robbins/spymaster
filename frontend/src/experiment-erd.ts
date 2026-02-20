/**
 * Entropy Regime Detector (ERD) — incremental per-bin computation.
 *
 * Detects entropy spikes in the spectrum state field as precursors to
 * regime transitions. Uses Variant B: signal = entropy_asym * spike_gate.
 *
 * Reference: agents/erd/run.py
 */

import { RollingRobustZScore } from './experiment-math';

export interface ERDBucketLike {
  k: number;
  flow_state_code: number;
  flow_score: number;
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
  update(grid: Map<number, ERDBucketLike>): number {
    // Count states across regions
    let nPFull = 0, nVFull = 0, nNFull = 0;
    let nPAbove = 0, nVAbove = 0, nNAbove = 0;
    let nPBelow = 0, nVBelow = 0, nNBelow = 0;
    let totalFull = 0, totalAbove = 0, totalBelow = 0;

    for (const [k, row] of grid) {
      const sc = row.flow_state_code;
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

    // Shannon entropy per region
    const hFull = entropy3(nPFull, nVFull, nNFull, totalFull);
    const hAbove = entropy3(nPAbove, nVAbove, nNAbove, totalAbove);
    const hBelow = entropy3(nPBelow, nVBelow, nNBelow, totalBelow);

    // Entropy asymmetry: positive = more disorder above spot
    const entropyAsym = hAbove - hBelow;

    // Robust z-score of full entropy
    const zH = this.zscore.push(hFull);

    // Spike gate: only active when entropy z-score exceeds floor
    const spikeGate = Math.max(0, zH - SPIKE_FLOOR);

    this.binCount++;

    // Variant B: entropy asymmetry × spike gate
    return this.warm ? entropyAsym * spikeGate : 0;
  }

  reset(): void {
    this.binCount = 0;
    this.zscore.reset();
  }
}

/** Shannon entropy of 3-state distribution. */
function entropy3(nP: number, nV: number, nN: number, total: number): number {
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
