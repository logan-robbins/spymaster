import * as THREE from 'three';

const TICK_INT = 250_000_000;
const STRIKE_TICKS = 20;  // $5 / $0.25 = 20 ticks per strike

/**
 * Options physics overlay rendered as horizontal bars at FIXED $5 strike levels.
 *
 * Key design: Options bars are at ABSOLUTE price levels (e.g., $6945, $6950, $6955).
 * They do NOT move when spot moves - only their COLOR changes based on composite fields.
 *
 * Composite view mirrors futures (pressure vs obstacles):
 * - Velocity drives build/erode intensity
 * - Pressure sign sets direction (green up / red down)
 * - Omega highlights strong walls
 */
export class OptionsGrid {
  private width: number;      // columns = seconds of history
  private numStrikes: number = 41;  // ±20 strikes around reference = 41 total
  private head: number = 0;
  private count: number = 0;

  // Reference strike (tick index of center $5 strike, set on first data)
  private referenceStrikeTick: number = 0;

  // Composite data: ring buffer [width][numStrikes]
  private velocityData: Float32Array;
  private texture: THREE.DataTexture;

  private mesh: THREE.Mesh;
  private material: THREE.ShaderMaterial;

  // Bar rendering parameters
  private barHeightTicks: number = 3;  // Height of each bar in ticks (~$0.75, thin accent)

  constructor(width: number = 1800) {
    this.width = width;

    // Composite texture: columns = time, rows = strikes
    this.velocityData = new Float32Array(width * this.numStrikes * 4);
    this.texture = new THREE.DataTexture(this.velocityData, width, this.numStrikes);
    this.texture.format = THREE.RGBAFormat;
    this.texture.type = THREE.FloatType;
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.wrapS = THREE.RepeatWrapping;
    this.texture.wrapT = THREE.ClampToEdgeWrapping;
    this.texture.needsUpdate = true;

    const vertexShader = `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    // Shader renders horizontal bars at FIXED $5 strike levels
    const fragmentShader = `
      uniform sampler2D uData;
      uniform float uWidth;
      uniform float uNumStrikes;
      uniform float uHead;
      uniform float uCount;
      uniform float uReferenceStrike;  // Tick index of center strike
      uniform float uStrikeTicks;      // 20 ticks = $5
      uniform float uBarHeight;        // Bar height in ticks
      uniform float uMeshBottom;       // Bottom Y of mesh in world coords
      uniform float uMeshHeight;       // Height of mesh in world coords
      varying vec2 vUv;

      void main() {
        // Map vUv.x to column in ring buffer
        float colsBack = (1.0 - vUv.x) * (uCount - 1.0);
        float colIndex = mod(uHead - colsBack + uWidth, uWidth);
        float texX = (floor(colIndex) + 0.5) / uWidth;

        // World Y position (absolute tick index)
        float worldY = uMeshBottom + vUv.y * uMeshHeight;

        // Find which $5 strike this Y is near
        // Strikes are at referenceStrike + n * STRIKE_TICKS
        float relToRef = worldY - uReferenceStrike;
        float strikeIndex = floor(relToRef / uStrikeTicks + 0.5);
        float strikeTick = uReferenceStrike + strikeIndex * uStrikeTicks;

        // Distance from strike center
        float distFromStrike = abs(worldY - strikeTick);

        // Only render within bar height
        if (distFromStrike > uBarHeight * 0.5) {
          discard;
        }

        // Map strikeIndex to texture row
        float centerStrikeIdx = (uNumStrikes - 1.0) * 0.5;
        float texRow = centerStrikeIdx + strikeIndex;
        if (texRow < 0.0 || texRow >= uNumStrikes) {
          discard;
        }
        float texY = (texRow + 0.5) / uNumStrikes;

        vec4 data = texture2D(uData, vec2(texX, texY));
        float velocity = data.r;
        float pressure = data.g;
        float omega = data.b;

        // Skip near-zero activity
        if (abs(velocity) < 0.005 && abs(pressure) < 0.005) {
          discard;
        }

        // Color palette (matches futures composite, red/green directional)
        const vec3 BUILD_UP = vec3(0.2, 0.9, 0.45);
        const vec3 BUILD_DOWN = vec3(0.9, 0.2, 0.2);
        const vec3 ERODE_COLOR = vec3(0.25, 0.05, 0.06);
        const vec3 WALL_COLOR = vec3(0.9, 0.95, 1.0);

        vec3 color = vec3(0.0);
        float alpha = 0.0;

        float dir = (abs(pressure) > 0.001) ? pressure : velocity;

        if (velocity > 0.01) {
          float intensity = tanh(velocity * 3.0);
          if (dir >= 0.0) {
            color = mix(vec3(0.1, 0.45, 0.2), BUILD_UP, intensity);
          } else {
            color = mix(vec3(0.45, 0.1, 0.1), BUILD_DOWN, intensity);
          }
          alpha = intensity * 0.50;  // Lower alpha so futures show through

          if (omega > 2.0 && velocity > 0.1) {
            float wallBoost = min((omega - 2.0) / 3.0, 0.5);
            color = mix(color, WALL_COLOR, wallBoost);
            alpha = min(alpha + wallBoost * 0.2, 0.65);
          }
        } else if (velocity < -0.01) {
          float intensity = tanh(abs(velocity) * 3.0);
          color = mix(vec3(0.12, 0.03, 0.04), ERODE_COLOR, intensity);
          alpha = intensity * 0.45;
        }

        // Edge fade for cleaner bars
        float edgeFade = 1.0 - smoothstep(uBarHeight * 0.35, uBarHeight * 0.5, distFromStrike);
        alpha *= edgeFade;

        gl_FragColor = vec4(color, alpha);
      }
    `;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uData: { value: this.texture },
        uWidth: { value: this.width },
        uNumStrikes: { value: this.numStrikes },
        uHead: { value: 0.0 },
        uCount: { value: 0.0 },
        uReferenceStrike: { value: 0.0 },
        uStrikeTicks: { value: STRIKE_TICKS },
        uBarHeight: { value: this.barHeightTicks },
        uMeshBottom: { value: 0.0 },
        uMeshHeight: { value: 1.0 },
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
   * Set the reference strike tick (center of the $5 grid).
   * Call once when first receiving data to anchor the strike grid.
   */
  setReferenceStrike(spotTickIndex: number): void {
    if (this.referenceStrikeTick === 0) {
      // Round spot to nearest $5 strike
      this.referenceStrikeTick = Math.round(spotTickIndex / STRIKE_TICKS) * STRIKE_TICKS;
      this.material.uniforms.uReferenceStrike.value = this.referenceStrikeTick;
      console.log(`OptionsGrid: reference strike set to tick ${this.referenceStrikeTick}`);
    }
  }

  /**
   * Advance ring buffer by one column
   */
  advance(): void {
    this.head = (this.head + 1) % this.width;
    if (this.count < this.width) {
      this.count++;
    }
    this.clearColumn(this.head);

    this.material.uniforms.uHead.value = this.head;
    this.material.uniforms.uCount.value = this.count;
  }

  /**
   * Write options composite data to current column.
   * @param absoluteStrikeTick - Absolute tick index of the strike
   * @param velocity - Aggregated liquidity velocity
   * @param pressure - Aggregated pressure gradient
   * @param omega - Aggregated obstacle strength
   */
  writeAbsolute(absoluteStrikeTick: number, velocity: number, pressure: number, omega: number): void {
    if (this.referenceStrikeTick === 0) return;

    // Compute strike index relative to reference
    const relTicks = absoluteStrikeTick - this.referenceStrikeTick;
    const strikeIndex = Math.round(relTicks / STRIKE_TICKS);

    // Map to texture row (center is at numStrikes/2)
    const centerIdx = Math.floor(this.numStrikes / 2);
    const row = centerIdx + strikeIndex;

    if (row < 0 || row >= this.numStrikes) return;

    const idx = (row * this.width + this.head) * 4;
    this.velocityData[idx] = velocity;
    this.velocityData[idx + 1] = pressure;
    this.velocityData[idx + 2] = omega;
    this.velocityData[idx + 3] = 1;
  }

  /**
   * Write options composite data using spot-relative ticks (converts to absolute internally)
   * @param spotTickIndex - Current spot tick index
   * @param relTicks - Relative ticks from spot (must be multiple of 20)
   * @param velocity - Aggregated liquidity velocity
   * @param pressure - Aggregated pressure gradient
   * @param omega - Aggregated obstacle strength
   */
  write(spotTickIndex: number, relTicks: number, velocity: number, pressure: number, omega: number): void {
    const absoluteStrikeTick = spotTickIndex + relTicks;
    // Round to nearest $5 strike
    const roundedStrike = Math.round(absoluteStrikeTick / STRIKE_TICKS) * STRIKE_TICKS;
    
    // DEBUG: Log first few writes
    if (this.count < 3 && Math.abs(relTicks) <= 40) {
      const relToRef = roundedStrike - this.referenceStrikeTick;
      const strikeIndex = Math.round(relToRef / STRIKE_TICKS);
      const centerIdx = Math.floor(this.numStrikes / 2);
      const row = centerIdx + strikeIndex;
      console.log(`write: spotTick=${spotTickIndex.toFixed(0)}, relTicks=${relTicks}, absStrike=${roundedStrike}, refStrike=${this.referenceStrikeTick}, strikeIdx=${strikeIndex}, row=${row}`);
    }
    
    this.writeAbsolute(roundedStrike, velocity, pressure, omega);
  }

  flush(): void {
    this.texture.needsUpdate = true;
    this.texture.version++;
  }

  /**
   * Update mesh positioning for rendering
   */
  updateMesh(timeSpan: number, meshBottom: number, meshHeight: number): void {
    this.mesh.scale.set(timeSpan, meshHeight, 1);
    this.mesh.position.x = -timeSpan / 2;
    this.mesh.position.y = meshBottom + meshHeight / 2;
    this.mesh.position.z = -0.01;  // Behind futures

    this.material.uniforms.uMeshBottom.value = meshBottom;
    this.material.uniforms.uMeshHeight.value = meshHeight;
  }

  private clearColumn(colIdx: number): void {
    const stride = this.width * 4;
    for (let row = 0; row < this.numStrikes; row++) {
      const idx = row * stride + colIdx * 4;
      this.velocityData[idx] = 0;
      this.velocityData[idx + 1] = 0;
      this.velocityData[idx + 2] = 0;
      this.velocityData[idx + 3] = 0;
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

  getReferenceStrike(): number {
    return this.referenceStrikeTick;
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
    this.referenceStrikeTick = 0;
    this.referenceSet = false;
    this.velocityData.fill(0);
    this.texture.needsUpdate = true;
  }
}

/**
 * Convert spot_ref_price_int to tick index
 */
export function spotToTickIndex(spotRefPriceInt: bigint): number {
  return Number(spotRefPriceInt) / TICK_INT;
}
