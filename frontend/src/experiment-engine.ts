/**
 * Composite experiment engine: blends ADS, PFP, SVac into one
 * directional signal for projection band rendering.
 *
 * Dynamic warmup: only warm signals contribute. Weights renormalize
 * among warm signals so composite stays in a consistent range.
 *
 * SVac online at 100ms (1 bin), PFP at 500ms, ADS at 20s.
 *
 * Weight rationale (MSD Round 3 experiment):
 *   ADS  0.40 — highest TP% (43.4%), strongest PnL/trade
 *   PFP  0.30 — strong balanced performer (40.6% TP, 731 signals)
 *   SVac 0.30 — spatial vacuum asymmetry (41.4% TP, 85.8% large-move selectivity)
 *   ERD  removed — weakest signal (36.7% TP, negative PnL)
 */

import { PFPSignal } from './experiment-pfp';
import { ADSSignal } from './experiment-ads';
import { SVacSignal } from './experiment-svac';

/** Raw blend weights (sum to 1.0 when all warm). */
const W_ADS = 0.40;
const W_PFP = 0.30;
const W_SVAC = 0.30;

export interface CompositeSignal {
  /** Blended directional signal (positive = bullish, negative = bearish). */
  composite: number;
  /** Individual experiment signals. */
  pfp: number;
  ads: number;
  svac: number;
  /** Fraction of experiments warm (0..1). 0.33 = one warm, 1.0 = all 3. */
  warmupFraction: number;
}

/** Minimal bucket interface — GridBucketRow satisfies this. */
export interface ExperimentBucketRow {
  k: number;
  v_add: number;
  v_fill: number;
  v_pull: number;
  spectrum_state_code: number;
  spectrum_score: number;
  vacuum_variant: number;
}

export class ExperimentEngine {
  private readonly ads = new ADSSignal();
  private readonly pfp = new PFPSignal();
  private readonly svac = new SVacSignal();

  /**
   * Process one bin's grid data. Returns composite directional signal.
   */
  update(grid: Map<number, ExperimentBucketRow>): CompositeSignal {
    const adsVal = this.ads.update(grid);
    const pfpVal = this.pfp.update(grid);
    const svacVal = this.svac.update(grid);

    // Dynamic warmup weighting
    let totalWeight = 0;
    let composite = 0;
    let warmCount = 0;

    if (this.ads.warm) {
      composite += W_ADS * adsVal;
      totalWeight += W_ADS;
      warmCount++;
    }
    if (this.pfp.warm) {
      composite += W_PFP * pfpVal;
      totalWeight += W_PFP;
      warmCount++;
    }
    if (this.svac.warm) {
      composite += W_SVAC * svacVal;
      totalWeight += W_SVAC;
      warmCount++;
    }

    // Renormalize
    if (totalWeight > 0) {
      composite /= totalWeight;
    }

    return {
      composite,
      pfp: pfpVal,
      ads: adsVal,
      svac: svacVal,
      warmupFraction: warmCount / 3,
    };
  }

  reset(): void {
    this.ads.reset();
    this.pfp.reset();
    this.svac.reset();
  }
}
