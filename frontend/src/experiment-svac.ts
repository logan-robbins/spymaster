/**
 * Spatial Vacuum Asymmetry (SVac) — incremental per-bin computation.
 *
 * Measures vacuum asymmetry above vs below spot using 1/|k| distance
 * weighting (variant C from MSD experiment). Near-spot vacuum gets
 * 50x higher weight than distant vacuum.
 *
 * Signal: weighted_above - weighted_below (positive = bullish)
 * Normalized by running EMA of |signal| to keep output in ~[-1, 1].
 *
 * Reference: agents/msd/run.py (compute_spatial_vacuum, variant C)
 */

export interface SVacBucketLike {
  k: number;
  vacuum_variant: number;
}

/** EMA smoothing factor for running magnitude normalization. */
const NORM_ALPHA = 0.05;
/** Minimum normalization denominator to avoid division instability. */
const NORM_FLOOR = 1e-6;

export class SVacSignal {
  private binCount = 0;
  /** Running EMA of |raw signal| for adaptive normalization. */
  private runningMag = 0;

  get warm(): boolean {
    // Purely spatial — no rolling windows, warm after first bin.
    return this.binCount >= 1;
  }

  /**
   * Process one bin. Returns normalized SVac signal (positive = bullish).
   * Returns 0 on the very first bin (need one sample for normalization).
   */
  update(grid: Map<number, SVacBucketLike>): number {
    let weightedBelow = 0;
    let weightedAbove = 0;

    for (const [k, row] of grid) {
      if (k === 0) continue;

      const absK = Math.abs(k);
      const w = 1.0 / absK;
      const vac = row.vacuum_variant;

      if (k < 0) {
        weightedBelow += vac * w;
      } else {
        weightedAbove += vac * w;
      }
    }

    // Positive = more weighted vacuum above spot = bullish
    const raw = weightedAbove - weightedBelow;

    // Update running magnitude EMA
    const abr = Math.abs(raw);
    if (this.binCount === 0) {
      this.runningMag = abr;
    } else {
      this.runningMag = NORM_ALPHA * abr + (1 - NORM_ALPHA) * this.runningMag;
    }

    this.binCount++;

    // Normalize to ~[-1, 1] range using running magnitude
    const denom = Math.max(this.runningMag, NORM_FLOOR);
    return raw / denom;
  }

  reset(): void {
    this.binCount = 0;
    this.runningMag = 0;
  }
}
