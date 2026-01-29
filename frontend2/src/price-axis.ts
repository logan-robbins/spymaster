import * as THREE from 'three';

const TICK_SIZE = 0.25; // $0.25 per tick

/**
 * Price axis labels rendered as HTML overlay
 */
export class PriceAxis {
  private container: HTMLElement;
  private labels: HTMLElement[] = [];
  private tickSpacing: number;
  private visibleTicks: number;

  constructor(containerId: string, tickSpacing: number = 20, visibleTicks: number = 100) {
    this.container = document.getElementById(containerId)!;
    this.tickSpacing = tickSpacing; // Label every N ticks (20 = $5.00)
    this.visibleTicks = visibleTicks;
    this.createLabels();
  }

  private createLabels(): void {
    // Create enough labels to cover visible range plus buffer
    const numLabels = Math.ceil(this.visibleTicks / this.tickSpacing) + 4;

    for (let i = 0; i < numLabels; i++) {
      const label = document.createElement('div');
      label.className = 'price-label';
      label.style.cssText = `
        position: absolute;
        right: 8px;
        color: #888;
        font-family: monospace;
        font-size: 11px;
        pointer-events: none;
        transform: translateY(-50%);
        text-align: right;
        min-width: 60px;
      `;
      this.container.appendChild(label);
      this.labels.push(label);
    }
  }

  /**
   * Update label positions based on current spot and camera
   * @param spotTickIndex Current spot price in ticks
   * @param camera Orthographic camera
   */
  update(spotTickIndex: number, camera: THREE.OrthographicCamera): void {
    const containerHeight = this.container.clientHeight;
    const viewHeight = camera.top - camera.bottom; // Total ticks visible

    // Find the base tick (rounded to tickSpacing)
    const baseTick = Math.floor(spotTickIndex / this.tickSpacing) * this.tickSpacing;

    // Calculate how many labels above and below
    const labelsAbove = Math.ceil((camera.top) / this.tickSpacing) + 1;
    const labelsBelow = Math.ceil((-camera.bottom) / this.tickSpacing) + 1;

    let labelIdx = 0;

    for (let i = -labelsBelow; i <= labelsAbove && labelIdx < this.labels.length; i++) {
      const tickValue = baseTick + (i * this.tickSpacing);
      const priceValue = tickValue * TICK_SIZE;

      // Convert tick to screen Y position
      // camera.position.y is at center, camera.top/bottom are relative offsets
      const tickOffset = tickValue - spotTickIndex;
      const normalizedY = (tickOffset - camera.bottom) / viewHeight;
      const screenY = containerHeight * (1 - normalizedY);

      const label = this.labels[labelIdx];

      // Only show if within visible area
      if (screenY >= -20 && screenY <= containerHeight + 20) {
        label.style.display = 'block';
        label.style.top = `${screenY}px`;
        label.textContent = `$${priceValue.toFixed(2)}`;
      } else {
        label.style.display = 'none';
      }

      labelIdx++;
    }

    // Hide remaining labels
    for (; labelIdx < this.labels.length; labelIdx++) {
      this.labels[labelIdx].style.display = 'none';
    }
  }
}
