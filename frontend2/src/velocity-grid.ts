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

    // UNIFIED COMPOSITE SHADER: Pressure vs Obstacles
    // Key insight: Only show activity, not smoothed background fields
    // - Velocity: The primary signal of liquidity change
    // - Pressure: Directional force (persistence-weighted velocity)
    // - Obstacles: Only where velocity shows building/depth (not smoothed omega)
    const fragmentShader = `
      uniform sampler2D uVelocity;
      uniform sampler2D uPressure;
      uniform sampler2D uOmega;
      uniform sampler2D uNu;
      uniform sampler2D uSpotHistory;
      uniform float uWidth;
      uniform float uHeight;
      uniform float uHead;
      uniform float uCount;
      uniform float uCurrentSpot;
      varying vec2 vUv;

      // Color palette (red/green directional)
      const vec3 BUILD_UP = vec3(0.2, 0.9, 0.45);        // Green - building bid (upward support)
      const vec3 BUILD_DOWN = vec3(0.9, 0.2, 0.2);       // Red - building ask (downward resistance)
      const vec3 ERODE_COLOR = vec3(0.25, 0.05, 0.06);   // Dark maroon - eroding (vacuum)
      const vec3 WALL_COLOR = vec3(0.85, 0.9, 0.95);     // Ice white - strong wall
      
      void main() {
        // Map UV to ring buffer coordinates
        float colsBack = (1.0 - vUv.x) * (uCount - 1.0);
        float colIndex = mod(uHead - colsBack + uWidth, uWidth);
        float texX = (floor(colIndex) + 0.5) / uWidth;

        float historicalSpot = texture2D(uSpotHistory, vec2(texX, 0.5)).r;
        if (historicalSpot == 0.0) discard;

        float worldY = uCurrentSpot + (vUv.y - 0.5) * uHeight;
        float historicalRelTicks = worldY - historicalSpot;
        float texY = (historicalRelTicks + uHeight * 0.5) / uHeight;
        if (texY < 0.0 || texY > 1.0) discard;

        vec2 texCoord = vec2(texX, texY);

        // Sample fields
        float velocity = texture2D(uVelocity, texCoord).r;
        float pressure = texture2D(uPressure, texCoord).r;
        float omega = texture2D(uOmega, texCoord).r;

        // Only render where there's actual activity
        // velocity = add_intensity - pull_intensity - fill_intensity
        // This is sparse - only non-zero where order flow happened
        if (abs(velocity) < 0.005 && abs(pressure) < 0.005) {
          discard;
        }

        vec3 finalColor = vec3(0.0);
        float finalAlpha = 0.0;

        // ===========================================
        // PRIMARY: Velocity determines the story
        // ===========================================
        // Positive velocity = liquidity building (support/resistance forming)
        // Negative velocity = liquidity eroding (vacuum opening)
        
        if (velocity > 0.01) {
          // BUILDING liquidity - color by NET pressure direction
          // After bid/ask aggregation: positive pressure = net bullish, negative = net bearish
          float intensity = tanh(velocity * 2.0);  // Slower saturation = more gradation
          
          if (pressure > 0.0) {
            // Net bid pressure = bullish = green
            finalColor = mix(vec3(0.1, 0.5, 0.25), BUILD_UP, intensity);
          } else {
            // Net ask pressure = bearish = red  
            finalColor = mix(vec3(0.5, 0.1, 0.1), BUILD_DOWN, intensity);
          }
          finalAlpha = 0.5 + intensity * 0.45;  // Range: 0.5 to 0.95 (more vibrant)
          
          // Strong walls get highlighted (high omega + building = solid wall)
          if (omega > 2.0 && velocity > 0.1) {
            float wallBoost = min((omega - 2.0) / 3.0, 0.6);
            finalColor = mix(finalColor, WALL_COLOR, wallBoost);
            finalAlpha = min(finalAlpha + wallBoost * 0.2, 0.98);
          }
          
        } else if (velocity < -0.01) {
          // ERODING liquidity - vacuum/thin zones
          float intensity = tanh(abs(velocity) * 2.0);
          
          // Dark maroon for erosion - shows where liquidity is leaving
          finalColor = mix(vec3(0.15, 0.05, 0.08), ERODE_COLOR, intensity);
          finalAlpha = 0.3 + intensity * 0.5;  // Range: 0.3 to 0.8
        }

        // ===========================================  
        // SECONDARY: Pure pressure with no velocity
        // ===========================================
        // Shows persistent force even without immediate flow
        if (finalAlpha < 0.1 && abs(pressure) > 0.01) {
          float pMag = tanh(abs(pressure) * 4.0);
          if (pressure > 0.0) {
            finalColor = vec3(0.15, 0.55, 0.3) * pMag;  // Dim green
          } else {
            finalColor = vec3(0.55, 0.15, 0.15) * pMag;  // Dim red
          }
          finalAlpha = 0.2 + pMag * 0.35;  // Range: 0.2 to 0.55
        }

        if (finalAlpha < 0.02) discard;

        gl_FragColor = vec4(finalColor, finalAlpha);
      }
    `;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uVelocity: { value: this.texture },
        uPressure: { value: this.pressureTexture },
        uOmega: { value: this.omegaTexture },
        uNu: { value: this.nuTexture },
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
  }

  /**
   * Set the mesh center for rendering (should match camera Y position)
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

  clear(): void {
    this.head = 0;
    this.count = 0;
    this.data.fill(0);
    this.energyData.fill(0);
    this.pressureData.fill(0);
    this.nuData.fill(0);
    this.omegaData.fill(0);
    this.spotHistoryData.fill(0);
    this.texture.needsUpdate = true;
    this.energyTexture.needsUpdate = true;
    this.pressureTexture.needsUpdate = true;
    this.nuTexture.needsUpdate = true;
    this.omegaTexture.needsUpdate = true;
    this.spotHistoryTexture.needsUpdate = true;
  }
}

/**
 * Convert spot_ref_price_int to tick index
 */
export function spotToTickIndex(spotRefPriceInt: bigint): number {
  return Number(spotRefPriceInt) / TICK_INT;
}
