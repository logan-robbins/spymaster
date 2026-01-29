import * as THREE from 'three';

const MAX_POINTS = 1800; // 30 minutes at 1s/point

/**
 * Turquoise line chart showing spot price history over time.
 * X-axis: time (0 = oldest, MAX_POINTS-1 = newest)
 * Y-axis: price in ticks
 */
export class SpotLine {
  private positions: Float32Array;
  private geometry: THREE.BufferGeometry;
  private material: THREE.LineBasicMaterial;
  private line: THREE.Line;

  private head: number = 0;
  private count: number = 0;
  private priceHistory: Float32Array;

  constructor() {
    // Store price history (in tick units)
    this.priceHistory = new Float32Array(MAX_POINTS);

    // Line geometry - positions updated each frame
    this.positions = new Float32Array(MAX_POINTS * 3);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setDrawRange(0, 0);

    // Turquoise line
    this.material = new THREE.LineBasicMaterial({
      color: 0x00d4aa,
      linewidth: 2,
    });

    this.line = new THREE.Line(this.geometry, this.material);
    this.line.frustumCulled = false;
  }

  getLine(): THREE.Line {
    return this.line;
  }

  /**
   * Add a new price point to the history
   * @param tickIndex - Price in tick units (spot_ref_price_int / TICK_INT)
   */
  addPrice(tickIndex: number): void {
    this.priceHistory[this.head] = tickIndex;
    this.head = (this.head + 1) % MAX_POINTS;
    if (this.count < MAX_POINTS) {
      this.count++;
    }
  }

  /**
   * Update line geometry to render current history.
   * Call this after addPrice() and before render.
   * @param xScale - Units per second (default 1)
   * @param xOffset - X offset for newest point (right edge)
   */
  updateGeometry(xScale: number = 1, xOffset: number = 0): void {
    if (this.count === 0) return;

    const posAttr = this.geometry.getAttribute('position') as THREE.BufferAttribute;

    // Draw from oldest to newest
    // Oldest is at (head - count + MAX_POINTS) % MAX_POINTS
    // Newest is at (head - 1 + MAX_POINTS) % MAX_POINTS

    for (let i = 0; i < this.count; i++) {
      const histIdx = (this.head - this.count + i + MAX_POINTS) % MAX_POINTS;
      const price = this.priceHistory[histIdx];

      // X: time position (newest on right)
      // i=0 is oldest, i=count-1 is newest
      const x = (i - this.count + 1) * xScale + xOffset;
      const y = price;
      const z = 0.1; // Slightly in front of grid

      this.positions[i * 3] = x;
      this.positions[i * 3 + 1] = y;
      this.positions[i * 3 + 2] = z;
    }

    posAttr.needsUpdate = true;
    this.geometry.setDrawRange(0, this.count);
  }

  getLatestPrice(): number {
    if (this.count === 0) return 0;
    const latestIdx = (this.head - 1 + MAX_POINTS) % MAX_POINTS;
    return this.priceHistory[latestIdx];
  }

  getCount(): number {
    return this.count;
  }

  /**
   * Get price at a specific column (for velocity grid rectification)
   * @param colsBack - How many columns back from newest (0 = newest)
   */
  getPriceAt(colsBack: number): number {
    if (colsBack >= this.count || colsBack < 0) return 0;
    const idx = (this.head - 1 - colsBack + MAX_POINTS) % MAX_POINTS;
    return this.priceHistory[idx];
  }
}
