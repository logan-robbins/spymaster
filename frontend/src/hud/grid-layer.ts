import * as THREE from 'three';

export class GridLayer {
    private width: number;   // columns = seconds of history
    private height: number;  // rows = ticks (logical)
    private texHeight: number; // rows = texels (actual)
    private resolution: number; // ticks per texel
    private head: number = 0;

    // Main Data (Relative Physics)
    private texture: THREE.DataTexture;
    private data: Float32Array;

    // Spot History (For Rectification)
    private spotCheckTexture: THREE.DataTexture;
    private spotHistoryData: Float32Array; // 1D Ring Buffer of Spot Prices

    private mesh: THREE.Mesh;
    private material: THREE.ShaderMaterial;

    private decayFactor: number = 0.0; // 0.0 = clear, 1.0 = persist
    private blendMode: 'replace' | 'max' = 'replace';

    constructor(width: number, height: number, type: 'wall' | 'vacuum' | 'physics' | 'gex' | 'bucket_radar', resolution: number = 1.0) {
        this.width = width;
        this.height = height;
        this.resolution = resolution;
        this.texHeight = Math.ceil(height / resolution);

        // 1. Physics Data Texture (RGBA Float32)
        this.data = new Float32Array(width * this.texHeight * 4);
        this.texture = new THREE.DataTexture(this.data, width, this.texHeight);
        this.texture.format = THREE.RGBAFormat;
        this.texture.type = THREE.FloatType;
        // Task 9: Force NearestFilter for bucketed look
        // Task 9: Force NearestFilter for bucketed look
        this.texture.minFilter = THREE.NearestFilter;
        this.texture.magFilter = THREE.NearestFilter;
        this.texture.wrapS = THREE.RepeatWrapping;
        this.texture.wrapT = THREE.ClampToEdgeWrapping;
        this.texture.needsUpdate = true;

        // 2. Spot History Texture (R Float32)
        // Stores the Spot Price (Int) for each column in the ring buffer
        this.spotHistoryData = new Float32Array(width);
        this.spotCheckTexture = new THREE.DataTexture(this.spotHistoryData, width, 1);
        this.spotCheckTexture.format = THREE.RedFormat;
        this.spotCheckTexture.type = THREE.FloatType;
        this.spotCheckTexture.minFilter = THREE.NearestFilter;
        this.spotCheckTexture.magFilter = THREE.NearestFilter;
        this.spotCheckTexture.wrapS = THREE.RepeatWrapping;
        this.spotCheckTexture.wrapT = THREE.ClampToEdgeWrapping;
        this.spotCheckTexture.needsUpdate = true;

        const vertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        // Shader Factory
        let fragmentShader = '';

        const commonUniforms = `
            uniform sampler2D map;
            uniform sampler2D uSpotHistory;
            uniform float uHeadOffset;
            uniform float uTime;
            uniform float uSpotRef; // Current Spot Price (Center of View)
            uniform float uWidth;   // Texture Width (History Seconds)
            uniform float uHeight;  // Logical Height (Ticks)
            uniform float uTexHeight; // Texture Height (Texels)
            uniform float uResolution; // Ticks per Texel
            varying vec2 vUv;
        `;

        // RECTIFICATION FUNCTION
        // Maps Screen UV (Absolute Price) -> Texture UV (Relative Tick)
        const rectifyLogic = `
            vec2 getRectifiedUV() {
                // 1. Column Addressing (Discrete 1s columns)
                float rawX = vUv.x + uHeadOffset;
                
                // Snap to nearest column center to prevent horizontal interpolation bleeding
                float colIndex = floor(rawX * uWidth);
                float x = (colIndex + 0.5) / uWidth;

                // 2. Vertical Rectification (Tick Space)
                // uSpotRef is now passed as (spot_ref_price_int / TICK_INT).
                // It is a Float Index of the spot price in "Tick Space".
                
                // Historical Spot: The texture stores the integer tick index of spot for that column.
                float historicalSpotTick = texture2D(uSpotHistory, vec2(x, 0.5)).r;
                
                if (historicalSpotTick < 1.0) return vec2(-1.0); // Invalid data
                
                // Current View Center (Tick Index) = uSpotRef
                // The Quad covers Height (uHeight ticks).
                // vUv.y = 0.5 corresponds to uSpotRef.
                // pixelTickIndex = uSpotRef + (vUv.y - 0.5) * uHeight
                
                float currentTickIndex = uSpotRef + (vUv.y - 0.5) * uHeight;
                
                // Relative Tick = PixelTick - HistoricalSpotTick
                // We want to map this to texture Y [0, 1]
                // Texture Range: 0..Height corresponds to relative ticks centered?
                // center = Height/2.
                
                
                float relTicks = currentTickIndex - historicalSpotTick;
                float relTexels = floor(relTicks / uResolution);
                
                // +0.5 to sample center of texel
                float textureRow = (uTexHeight * 0.5) + relTexels + 0.5;
                
                // Normalize to [0, 1]
                float v = textureRow / uTexHeight;
                
                return vec2(x, v);
            }
        `;

        if (type === 'wall') {
            fragmentShader = `
                ${commonUniforms}
                ${rectifyLogic}

                vec3 askColor = vec3(0.08, 0.2, 1.0);
                vec3 bidColor = vec3(1.0, 0.2, 0.08);

                void main() {
                    vec2 uv = getRectifiedUV();
                    if (uv.y < 0.0 || uv.y > 1.0) discard; // Out of data range

                    vec4 data = texture2D(map, uv);
                    
                    float strength = data.r;
                    float velocity = data.g;
                    float accel = data.b;
                    float side = data.a;

                    if (strength <= 0.001) discard;

                    vec3 color = (side > 0.0) ? askColor : bidColor;
                    float intensity = clamp(strength / 12.0, 0.0, 1.0);
                    
                    // Doppler
                    float doppler = clamp(velocity * 0.5, -0.5, 0.5);
                    
                    // Iceberg Flash (White)
                    vec3 flash = vec3(0.0);
                    if (velocity > 2.0) flash = vec3(0.4);

                    gl_FragColor = vec4(color * (intensity + doppler) + flash, intensity * 0.95);
                }
            `;
        } else if (type === 'vacuum') {
            fragmentShader = `
                ${commonUniforms}
                ${rectifyLogic}
                
                // Pseudo-noise
                float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

                void main() {
                    vec2 uv = getRectifiedUV();
                    if (uv.y < 0.0 || uv.y > 1.0) discard;

                    vec4 data = texture2D(map, uv);
                    float vacuum = data.r;
                    float turbulence = data.g;
                    float erosion = data.b;

                    float darkness = clamp(vacuum * 0.9 + erosion * 0.6, 0.0, 1.0);
                    if (darkness < 0.1) discard;
                    
                    // Crack Noise
                    float n = hash(uv * 50.0 + vec2(uTime * 0.2, 0.0));
                    
                    // Turbulence widens cracks
                    float threshold = 0.6 - (turbulence * 0.2) + (erosion * 0.2);
                    if (n > threshold) discard; 

                    gl_FragColor = vec4(0.0, 0.0, 0.0, darkness);
                }
            `;
        } else if (type === 'bucket_radar') {
            fragmentShader = `
                ${commonUniforms}
                ${rectifyLogic}

                void main() {
                    vec2 uv = getRectifiedUV();
                    if (uv.y < 0.0 || uv.y > 1.0) discard;

                    vec4 data = texture2D(map, uv);
                    float blocked = data.r;      // 0..5
                    float cavitation = data.g;   // 0..1
                    float gex = data.b;          // 0..1
                    float mobility = data.a;     // 0..1

                    // Blockedness (Red/Orange)
                    float blockAlpha = clamp(blocked / 5.0, 0.0, 1.0);
                    vec3 blockColor = vec3(1.0, 0.3 * (1.0 - blockAlpha), 0.0) * blockAlpha;

                    // Cavitation (Blue/Cyan Neon)
                    float cavAlpha = clamp(cavitation, 0.0, 1.0);
                    // Blue glow
                    vec3 cavColor = vec3(0.0, 0.8, 1.0) * cavAlpha;

                    // Gex Stiffness (Magenta/Purple "Force Field" overlay)
                    // Matches "Pressure" metaphor (High energy/stiffness = Warm-ish)
                    float gexAlpha = clamp(gex, 0.0, 1.0);
                    vec3 gexColor = vec3(0.9, 0.0, 0.9) * (gexAlpha * 0.4);

                    // Combine (Additive for glow, or Alpha blend?)
                    // Start with blocked wall
                    vec3 color = blockColor;
                    
                    // Add Cavitation (glowing inside/over wall)
                    color += cavColor;
                    
                    // Add GEX
                    color += gexColor;
                    
                    // Alpha
                    float alpha = clamp(blockAlpha + cavAlpha + (gexAlpha * 0.3), 0.0, 0.95);
                    
                    if (alpha < 0.05) discard;

                    gl_FragColor = vec4(color, alpha);
                }
            `;
        } else {
            // Physics / GEX (pass-through)
            fragmentShader = `
                ${commonUniforms}
                // Physics layer is aggregate, so no rectification needed actually?
				// But we want it aligned.
                ${rectifyLogic}

                void main() {
                   // Physics/GEX data is tick-relative with rectification applied.
                   
                   vec2 uv = getRectifiedUV();
                   if (uv.y < 0.0 || uv.y > 1.0) discard;

                   vec4 data = texture2D(map, uv);
                   gl_FragColor = data; // Task 10: Colors are 0..1 
                }
            `;
        }

        this.material = new THREE.ShaderMaterial({
            uniforms: {
                map: { value: this.texture },
                uSpotHistory: { value: this.spotCheckTexture },
                uHeadOffset: { value: 0.0 },
                uTime: { value: 0.0 },
                uSpotRef: { value: 6000.0 }, // Updated by renderer
                uWidth: { value: this.width },
                uHeight: { value: this.height },
                uTexHeight: { value: this.texHeight },
                uResolution: { value: this.resolution }
            },
            vertexShader,
            fragmentShader,
            transparent: true,
            side: THREE.DoubleSide,
            blending: type === 'gex' ? THREE.AdditiveBlending : THREE.NormalBlending,
            depthWrite: type !== 'gex' && type !== 'bucket_radar'
        });

        const geometry = new THREE.PlaneGeometry(1, 1);
        this.mesh = new THREE.Mesh(geometry, this.material);
        this.mesh.frustumCulled = false;
    }

    getMesh(): THREE.Mesh {
        return this.mesh;
    }

    advance(time: number, newSpot: number): void {
        this.head = (this.head + 1) % this.width;

        // Write or Decay
        const prevHead = (this.head - 1 + this.width) % this.width;
        if (this.decayFactor > 0.001) {
            this.decayColumn(this.head, prevHead);
        } else {
            this.clearColumn(this.head);
        }

        // Write Spot History
        this.spotHistoryData[this.head] = newSpot;
        this.spotCheckTexture.needsUpdate = true;

        // Update Uniforms
        const offset = (this.head + 1) / this.width;
        this.material.uniforms.uHeadOffset.value = offset;
        this.material.uniforms.uTime.value = time;
    }

    setSpotRef(price: number): void {
        this.material.uniforms.uSpotRef.value = price;
    }

    clearColumn(colIdx: number): void {
        const start = colIdx * 4;
        const stride = this.width * 4;
        for (let y = 0; y < this.texHeight; y++) {
            const idx = y * stride + start;
            this.data[idx] = 0;
            this.data[idx + 1] = 0;
            this.data[idx + 2] = 0;
            this.data[idx + 3] = 0;
        }
        this.texture.needsUpdate = true;
    }

    private decayColumn(destCol: number, srcCol: number): void {
        const stride = this.width * 4;
        const decay = this.decayFactor;

        for (let y = 0; y < this.texHeight; y++) {
            const srcIdx = y * stride + srcCol * 4;
            const destIdx = y * stride + destCol * 4;

            // Decay all channels? Typically alpha is key, but fading color is safe too.
            this.data[destIdx] = this.data[srcIdx] * decay;
            this.data[destIdx + 1] = this.data[srcIdx + 1] * decay;
            this.data[destIdx + 2] = this.data[srcIdx + 2] * decay;
            this.data[destIdx + 3] = this.data[srcIdx + 3] * decay;
        }
        this.texture.needsUpdate = true;
    }

    setDecay(tau: number): void {
        if (tau <= 0) this.decayFactor = 0;
        else this.decayFactor = Math.exp(-1.0 / tau);
    }

    setBlendMode(mode: 'replace' | 'max'): void {
        this.blendMode = mode;
    }

    write(relTicks: number, vector: [number, number, number, number]): void {
        this.writeAt(this.head, relTicks, vector);
    }

    writeAt(colIdx: number, relTicks: number, vector: [number, number, number, number]): void {
        const centerY = Math.floor(this.texHeight / 2);
        const relTexels = Math.floor(relTicks / this.resolution);
        const y = centerY + relTexels;

        if (y < 0 || y >= this.texHeight) return;

        const idx = (y * this.width + colIdx) * 4;

        if (this.blendMode === 'max') {
            // Max blend for persisting peaks
            this.data[idx] = Math.max(this.data[idx], vector[0]);
            this.data[idx + 1] = Math.max(this.data[idx + 1], vector[1]);
            this.data[idx + 2] = Math.max(this.data[idx + 2], vector[2]);
            this.data[idx + 3] = Math.max(this.data[idx + 3], vector[3]);
        } else {
            // Replace
            this.data[idx] = vector[0];
            this.data[idx + 1] = vector[1];
            this.data[idx + 2] = vector[2];
            this.data[idx + 3] = vector[3];
        }
        this.texture.needsUpdate = true;
    }

    getHead(): number {
        return this.head;
    }

    getWidth(): number {
        return this.width;
    }
}


