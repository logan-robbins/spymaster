import * as THREE from 'three';

const TICK_INT = 250_000_000;

export class VelocityGrid {
  private width: number;      // columns = seconds of history
  private height: number;     // rows = ticks (±400 from spot)
  private head: number = 0;
  private count: number = 0;  // How many columns have data

  private texture: THREE.DataTexture;
  private data: Float32Array;

  // Additional fields
  private energyTexture: THREE.DataTexture;
  private energyData: Float32Array;

  private pressureTexture: THREE.DataTexture;
  private pressureData: Float32Array;

  private nuTexture: THREE.DataTexture;
  private nuData: Float32Array;

  private omegaTexture: THREE.DataTexture;
  private omegaData: Float32Array;

  // Spot history for rectification - stores tick index per column
  private spotHistoryTexture: THREE.DataTexture;
  private spotHistoryData: Float32Array;

  private mesh: THREE.Mesh;
  private material: THREE.ShaderMaterial;

  constructor(width: number = 1800, height: number = 801) {
    this.width = width;
    this.height = height;

    const size = width * height * 4;

    // Helper to create texture
    const createTex = () => {
      const d = new Float32Array(size);
      const t = new THREE.DataTexture(d, width, height);
      t.format = THREE.RGBAFormat;
      t.type = THREE.FloatType;
      t.minFilter = THREE.NearestFilter;
      t.magFilter = THREE.NearestFilter;
      t.wrapS = THREE.RepeatWrapping;
      t.wrapT = THREE.ClampToEdgeWrapping;
      t.needsUpdate = true;
      return { data: d, texture: t };
    };

    // Velocity (Main)
    const vel = createTex();
    this.data = vel.data;
    this.texture = vel.texture;

    // Energy
    const en = createTex();
    this.energyData = en.data;
    this.energyTexture = en.texture;

    // Pressure
    const pr = createTex();
    this.pressureData = pr.data;
    this.pressureTexture = pr.texture;

    // Viscosity (nu)
    const nu = createTex();
    this.nuData = nu.data;
    this.nuTexture = nu.texture;

    // Obstacle (Omega)
    const om = createTex();
    this.omegaData = om.data;
    this.omegaTexture = om.texture;

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
      uniform int uMode;          // 0=vel, 1=energy, 2=pressure, 3=viscosity, 4=obstacle
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
        float value = data.r;

        vec3 color = vec3(0.0);
        float alpha = 0.0;
        
        if (uMode == 0) { // Velocity
            if (abs(value) < 0.001) discard;
            if (value > 0.0) color = vec3(0.0, 1.0, 0.6); // Green
            else color = vec3(1.0, 0.27, 0.13); // Red
            alpha = tanh(abs(value) * 2.0) * 0.8;
            
        } else if (uMode == 1) { // Energy
            if (value < 0.001) discard;
            // Hot heatmap (Black -> Red -> Yellow -> White)
            color = vec3(min(value * 5.0, 1.0), min(max(value * 5.0 - 1.0, 0.0), 1.0), min(max(value * 5.0 - 2.0, 0.0), 1.0));
            alpha = min(value * 5.0, 0.8);
            
        } else if (uMode == 2) { // Pressure
             if (abs(value) < 0.001) discard;
             // Blue (neg) to Red (pos)
             if (value > 0.0) color = vec3(1.0, 0.2, 0.2);
             else color = vec3(0.2, 0.2, 1.0);
             alpha = tanh(abs(value) * 10.0) * 0.8;
             
        } else if (uMode == 3) { // Viscosity
             if (value <= 1.001) discard; // nu starts at 1
             // Purple haze
             float intensity = (value - 1.0) * 0.5;
             color = vec3(0.6, 0.0, 0.8);
             alpha = tanh(intensity) * 0.6;
             
        } else if (uMode == 4) { // Obstacle (Omega)
             if (value < 0.1) discard;
             // White/Grey walls
             color = vec3(0.8, 0.8, 0.9);
             alpha = min(value * 0.3, 0.9);
        }

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
        uMode: { value: 0 },
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

  setDisplayMode(mode: 'velocity' | 'energy' | 'pressure' | 'viscosity' | 'obstacle') {
    let tex = this.texture;
    let m = 0;
    switch (mode) {
      case 'velocity': tex = this.texture; m = 0; break;
      case 'energy': tex = this.energyTexture; m = 1; break;
      case 'pressure': tex = this.pressureTexture; m = 2; break;
      case 'viscosity': tex = this.nuTexture; m = 3; break;
      case 'obstacle': tex = this.omegaTexture; m = 4; break;
    }
    this.material.uniforms.uData.value = tex;
    this.material.uniforms.uMode.value = m;
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
  }

  /**
   * Set the mesh center for rendering (should match camera Y position)
   * This is separate from current spot - the mesh is centered on the camera view.
   */
  setMeshCenter(meshCenterTick: number): void {
    this.material.uniforms.uCurrentSpot.value = meshCenterTick;
  }

  /**
   * Write velocity data to current column
   */
  write(relTicks: number, velocity: number, energy: number, pressure: number, nu: number, omega: number): void {
    const centerY = Math.floor(this.height / 2);
    const y = centerY + relTicks;

    if (y < 0 || y >= this.height) return;

    const idx = (y * this.width + this.head) * 4;

    // Write to all arrays
    this.data[idx] = velocity;
    this.energyData[idx] = energy;
    this.pressureData[idx] = pressure;
    this.nuData[idx] = nu;
    this.omegaData[idx] = omega;
  }

  flush(): void {
    // Force texture re-upload
    this.texture.needsUpdate = true;
    this.energyTexture.needsUpdate = true;
    this.pressureTexture.needsUpdate = true;
    this.nuTexture.needsUpdate = true;
    this.omegaTexture.needsUpdate = true;
  }

  private clearColumn(colIdx: number): void {
    const stride = this.width * 4;
    for (let y = 0; y < this.height; y++) {
      const idx = y * stride + colIdx * 4;
      this.data[idx] = 0;
      this.energyData[idx] = 0;
      this.pressureData[idx] = 0;
      this.nuData[idx] = 0;
      this.omegaData[idx] = 0;
    }
    // We only need to flag update if we are looking at it potentially?
    // Safer to flag all.
    this.texture.needsUpdate = true;
    this.energyTexture.needsUpdate = true;
    this.pressureTexture.needsUpdate = true;
    this.nuTexture.needsUpdate = true;
    this.omegaTexture.needsUpdate = true;
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
