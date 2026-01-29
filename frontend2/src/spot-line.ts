import * as THREE from 'three';

const MAX_POINTS = 1800; // 30 minutes at 1s/point

/**
 * Glowing blue spot price line.
 * Creates a soft glow effect using multiple semi-transparent layers.
 */
export class SpotLine {
  private positions: Float32Array;
  private geometry: THREE.BufferGeometry;
  private group: THREE.Group;

  private head: number = 0;
  private count: number = 0;
  private priceHistory: Float32Array;

  constructor() {
    this.group = new THREE.Group();

    // Store price history (in tick units)
    this.priceHistory = new Float32Array(MAX_POINTS);

    // Line geometry - positions updated each frame
    this.positions = new Float32Array(MAX_POINTS * 3);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setDrawRange(0, 0);

    // Outer glow layers (soft blue haze)
    const glowLayers = [
      { yOffset: 1.0, opacity: 0.08, color: 0x0044aa },
      { yOffset: 0.7, opacity: 0.12, color: 0x0066cc },
      { yOffset: 0.5, opacity: 0.18, color: 0x0088dd },
      { yOffset: 0.3, opacity: 0.25, color: 0x00aaee },
      { yOffset: 0.15, opacity: 0.4, color: 0x00ccff },
    ];

    for (const layer of glowLayers) {
      // Upper glow
      this.addGlowLine(layer.yOffset, layer.opacity, layer.color);
      // Lower glow
      this.addGlowLine(-layer.yOffset, layer.opacity, layer.color);
    }

    // Core line (bright white-blue, multiple for thickness)
    const coreOffsets = [-0.08, -0.04, 0, 0.04, 0.08];
    for (const offset of coreOffsets) {
      const coreMaterial = new THREE.ShaderMaterial({
        uniforms: {
          color: { value: new THREE.Color(0x44ddff) },
          yOffset: { value: offset },
        },
        vertexShader: `
          uniform float yOffset;
          void main() {
            vec3 pos = position;
            pos.y += yOffset;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
          }
        `,
        fragmentShader: `
          uniform vec3 color;
          void main() {
            gl_FragColor = vec4(color, 1.0);
          }
        `,
      });
      const coreLine = new THREE.Line(this.geometry, coreMaterial);
      coreLine.frustumCulled = false;
      this.group.add(coreLine);
    }

  }

  private addGlowLine(yOffset: number, opacity: number, color: number): void {
    const material = new THREE.ShaderMaterial({
      uniforms: {
        color: { value: new THREE.Color(color) },
        opacity: { value: opacity },
        yOffset: { value: yOffset },
      },
      vertexShader: `
        uniform float yOffset;
        void main() {
          vec3 pos = position;
          pos.y += yOffset;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 color;
        uniform float opacity;
        void main() {
          gl_FragColor = vec4(color, opacity);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const line = new THREE.Line(this.geometry, material);
    line.frustumCulled = false;
    this.group.add(line);
  }

  getLine(): THREE.Group {
    return this.group;
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
    for (let i = 0; i < this.count; i++) {
      const histIdx = (this.head - this.count + i + MAX_POINTS) % MAX_POINTS;
      const price = this.priceHistory[histIdx];

      const x = (i - this.count + 1) * xScale + xOffset;
      const y = price;
      const z = 0.5; // In front of grids

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
