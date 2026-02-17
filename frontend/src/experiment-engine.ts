/**
 * Composite experiment engine: blends PFP, ADS, ERD into one
 * directional signal for projection band rendering.
 *
 * Dynamic warmup: only warm signals contribute. Weights renormalize
 * among warm signals so composite stays in a consistent range.
 *
 * PFP online at 500ms, ERD at 10s, ADS at 20s.
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
  /** Fraction of experiments warm (0..1). 0.33 = PFP only, 1.0 = all 3. */
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
}

export class ExperimentEngine {
  private readonly pfp = new PFPSignal();
  private readonly ads = new ADSSignal();
  private readonly erd = new ERDSignal();

  /**
   * Process one bin's grid data. Returns composite directional signal.
   */
  update(grid: Map<number, ExperimentBucketRow>): CompositeSignal {
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

    // Renormalize
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
