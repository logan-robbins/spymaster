import * as THREE from 'three';

const TICK_INT = 250_000_000;

export class VelocityGrid {
  private width: number;      // columns = seconds of history
  private height: number;     // rows = ticks (±400 from spot)
  private head: number = 0;
  private count: number = 0;  // How many columns have data

  private texture: THREE.DataTexture;
  private data: Float32Array;

  // Spot history for rectification - stores tick index per column
  private spotHistoryTexture: THREE.DataTexture;
  private spotHistoryData: Float32Array;

  private mesh: THREE.Mesh;
  private material: THREE.ShaderMaterial;

  constructor(width: number = 1800, height: number = 801) {
    this.width = width;
    this.height = height;

    // Main data texture: RGBA Float32
    // R = velocity (signed), G/B unused, A = computed alpha
    this.data = new Float32Array(width * height * 4);
    this.texture = new THREE.DataTexture(this.data, width, height);
    this.texture.format = THREE.RGBAFormat;
    this.texture.type = THREE.FloatType;
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.wrapS = THREE.RepeatWrapping;
    this.texture.wrapT = THREE.ClampToEdgeWrapping;
    this.texture.needsUpdate = true;

    // Spot history texture: 1D ring buffer of spot prices (tick index)
    this.spotHistoryData = new Float32Array(width);
    this.spotHistoryTexture = new THREE.DataTexture(this.spotHistoryData, width, 1);
    this.spotHistoryTexture.format = THREE.RedFormat;
    this.spotHistoryTexture.type = THREE.FloatType;
    this.spotHistoryTexture.minFilter = THREE.NearestFilter;
    this.spotHistoryTexture.magFilter = THREE.NearestFilter;
    this.spotHistoryTexture.wrapS = THREE.RepeatWrapping;
    this.spotHistoryTexture.wrapT = THREE.ClampToEdgeWrapping;
    this.spotHistoryTexture.needsUpdate = true;

    const vertexShader = `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    // Shader renders velocity as colored overlay
    // World coordinates: X = seconds back from now (0 = now, negative = past)
    //                    Y = absolute tick price
    // The mesh is positioned/scaled so vUv maps to the visible time range
    const fragmentShader = `
      uniform sampler2D uData;
      uniform sampler2D uSpotHistory;
      uniform float uWidth;
      uniform float uHeight;
      uniform float uHead;        // Current head column index
      uniform float uCount;       // Number of valid columns
      uniform float uCurrentSpot; // Current spot tick index
      varying vec2 vUv;

      void main() {
        // vUv.x: 0 = left edge (oldest visible), 1 = right edge (newest)
        // Map to column index in ring buffer
        // Newest is at head, oldest visible is at (head - count + 1)

        float colsBack = (1.0 - vUv.x) * (uCount - 1.0);
        float colIndex = mod(uHead - colsBack + uWidth, uWidth);
        float texX = (floor(colIndex) + 0.5) / uWidth;

        // Get historical spot for this column
        float historicalSpot = texture2D(uSpotHistory, vec2(texX, 0.5)).r;

        // Skip if no historical data yet
        if (historicalSpot == 0.0) {
          discard;
        }

        // vUv.y maps to tick range around current spot
        // 0 = bottom (currentSpot - height/2), 1 = top (currentSpot + height/2)
        float worldY = uCurrentSpot + (vUv.y - 0.5) * uHeight;

        // Convert to texture row: what rel_ticks was this at the historical spot?
        float historicalRelTicks = worldY - historicalSpot;

        // Texture Y: center row (height/2) is rel_ticks=0
        float texY = (historicalRelTicks + uHeight * 0.5) / uHeight;

        // Out of bounds check
        if (texY < 0.0 || texY > 1.0) {
          discard;
        }

        vec4 data = texture2D(uData, vec2(texX, texY));
        float velocity = data.r;

        // Skip near-zero velocity
        if (abs(velocity) < 0.001) {
          discard;
        }

        // Color based on sign
        vec3 color;
        if (velocity > 0.0) {
          color = vec3(0.0, 1.0, 0.6); // Green - building
        } else {
          color = vec3(1.0, 0.27, 0.13); // Red - eroding
        }

        // Alpha from velocity magnitude (tanh normalization)
        float alpha = tanh(abs(velocity) * 2.0) * 0.8;

        gl_FragColor = vec4(color, alpha);
      }
    `;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uData: { value: this.texture },
        uSpotHistory: { value: this.spotHistoryTexture },
        uWidth: { value: this.width },
        uHeight: { value: this.height },
        uHead: { value: 0.0 },
        uCount: { value: 0.0 },
        uCurrentSpot: { value: 0.0 },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    const geometry = new THREE.PlaneGeometry(1, 1);
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
  }

  getMesh(): THREE.Mesh {
    return this.mesh;
  }

  /**
   * Advance ring buffer by one column and record current spot
   */
  advance(currentSpotTickIndex: number): void {
    this.head = (this.head + 1) % this.width;
    if (this.count < this.width) {
      this.count++;
    }
    this.clearColumn(this.head);

    // Store spot for this column (for rectification)
    this.spotHistoryData[this.head] = currentSpotTickIndex;
    this.spotHistoryTexture.needsUpdate = true;

    // Update uniforms
    this.material.uniforms.uHead.value = this.head;
    this.material.uniforms.uCount.value = this.count;
    this.material.uniforms.uCurrentSpot.value = currentSpotTickIndex;
  }

  /**
   * Set current spot reference for view centering (no-op, kept for API compatibility)
   */
  setSpotRef(_spotTickIndex: number): void {
    // No longer needed - handled in advance()
  }

  /**
   * Write velocity data to current column
   * @param relTicks - Relative tick offset from spot
   * @param velocity - Liquidity velocity value (can be negative)
   */
  write(relTicks: number, velocity: number): void {
    const centerY = Math.floor(this.height / 2);
    const y = centerY + relTicks;

    if (y < 0 || y >= this.height) return;

    const idx = (y * this.width + this.head) * 4;
    this.data[idx] = velocity;     // R = velocity
    this.data[idx + 1] = 0;        // G unused
    this.data[idx + 2] = 0;        // B unused
    this.data[idx + 3] = 1;        // A = 1 (computed in shader)
  }

  flush(): void {
    // Force texture re-upload
    this.texture.needsUpdate = true;
    this.texture.version++;
  }

  private clearColumn(colIdx: number): void {
    const stride = this.width * 4;
    for (let y = 0; y < this.height; y++) {
      const idx = y * stride + colIdx * 4;
      this.data[idx] = 0;
      this.data[idx + 1] = 0;
      this.data[idx + 2] = 0;
      this.data[idx + 3] = 0;
    }
    this.texture.needsUpdate = true;
  }

  getHead(): number {
    return this.head;
  }

  getCount(): number {
    return this.count;
  }

  getWidth(): number {
    return this.width;
  }

  getHeight(): number {
    return this.height;
  }
}

/**
 * Convert spot_ref_price_int to tick index
 */
export function spotToTickIndex(spotRefPriceInt: bigint): number {
  return Number(spotRefPriceInt) / TICK_INT;
}
