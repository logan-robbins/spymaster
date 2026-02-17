/**
 * Incremental math utilities for experiment signal computation.
 *
 * These are streaming (O(1) per push) implementations of:
 * - Rolling OLS slope (IncrementalOLSSlope)
 * - Rolling robust z-score using median/MAD (RollingRobustZScore)
 *
 * Reference: eval_harness.py:318-369
 */

/**
 * Incremental rolling OLS slope.
 *
 * slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
 * where x = [0, 1, ..., w-1] are fixed positions.
 *
 * Ring buffer stores y values. On each push after warmup,
 * the oldest y exits and the new y enters.
 */
export class IncrementalOLSSlope {
  private readonly w: number;
  private readonly ring: Float64Array;
  private cursor: number = 0;
  private count: number = 0;

  private readonly sum_x: number;
  private readonly denom: number;

  constructor(window: number) {
    this.w = window;
    this.ring = new Float64Array(window);

    let sx = 0, sx2 = 0;
    for (let i = 0; i < window; i++) {
      sx += i;
      sx2 += i * i;
    }
    this.sum_x = sx;
    this.denom = window * sx2 - sx * sx;
  }

  /** Push a new y value. Returns current slope (NaN if window not full). */
  push(y: number): number {
    if (this.count < this.w) {
      this.ring[this.count] = y;
      this.count++;
      if (this.count < this.w) return NaN;
      this.cursor = 0;
      return this._computeSlope();
    }

    // Steady-state: replace oldest
    this.ring[this.cursor] = y;
    this.cursor = (this.cursor + 1) % this.w;
    return this._computeSlope();
  }

  /** Recompute slope from full ring buffer. O(w) but w is small. */
  private _computeSlope(): number {
    let sum_y = 0;
    let sum_xy = 0;
    for (let i = 0; i < this.w; i++) {
      const idx = (this.cursor + i) % this.w;
      const val = this.ring[idx];
      sum_y += val;
      sum_xy += i * val;
    }
    return (this.w * sum_xy - this.sum_x * sum_y) / this.denom;
  }

  reset(): void {
    this.ring.fill(0);
    this.cursor = 0;
    this.count = 0;
  }
}


/**
 * Rolling robust z-score using median and MAD.
 *
 * z = (x - median) / (1.4826 * MAD)
 * where MAD = median(|x_i - median|)
 *
 * Maintains a sorted array for O(log w) insert/remove via binary search.
 * Median is O(1) from sorted middle.
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
      this.ring[this.count] = x;
      this._sortedInsert(x);
      this.count++;
    } else {
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

    // Compute MAD: median of absolute deviations
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
