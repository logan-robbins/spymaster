import * as THREE from 'three';

export class GridLayer {
    private width: number;   // columns = seconds of history
    private height: number;  // rows = ticks
    private head: number = 0;

    // Main Data (Relative Physics)
    private texture: THREE.DataTexture;
    private data: Float32Array;

    // Spot History (For Rectification)
    private spotCheckTexture: THREE.DataTexture;
    private spotHistoryData: Float32Array; // 1D Ring Buffer of Spot Prices

    private mesh: THREE.Mesh;
    private material: THREE.ShaderMaterial;

    constructor(width: number, height: number, type: 'wall' | 'vacuum' | 'physics' | 'gex') {
        this.width = width;
        this.height = height;

        // 1. Physics Data Texture (RGBA Float32)
        this.data = new Float32Array(width * height * 4);
        this.texture = new THREE.DataTexture(this.data, width, height);
        this.texture.format = THREE.RGBAFormat;
        this.texture.type = THREE.FloatType;
        const useLinearFilter = type === 'physics' || type === 'wall' || type === 'gex';
        this.texture.minFilter = useLinearFilter ? THREE.LinearFilter : THREE.NearestFilter;
        this.texture.magFilter = useLinearFilter ? THREE.LinearFilter : THREE.NearestFilter;
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
            uniform float uHeight;  // Texture Height (Tick Range)
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
                float x = (colIndex + 0.5) / uWidth; // continuous [0, 1] covering history window.
                // We want to snap to the exact texture column corresponding to the second.
                // Texture width = uHistorySeconds.
                
                float totalCols = 1800.0; // Fixed history? Or uniform uWidth?
                // Let's assume vUv.x maps 0..1 to 0..Width.
                // headOffset shifts the ring buffer.
                // x_ring = mod(floor(vUv.x * totalCols) + uHeadOffset * totalCols, totalCols) / totalCols
                
                // Simplified: The texture wraps.
                // We just need to snap vUv.x to nearest column center?
                // No, NearestFilter handles fractional UVs by picking nearest texel.
                // But specifically for TIME alignment, we want 1 pixel = 1 second.
                // If we rely on UV, linear interpolation might bleed neighbors.
                // NearestFilter is set in constructor. So vUv.x is safe if we don't offset by sub-pixels.
                
                float x = vUv.x + uHeadOffset;

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
                // No, texture is written as: center + relTicks.
                // center = Height/2.
                
                float relTicks = currentTickIndex - historicalSpotTick;
                float textureRow = (uHeight * 0.5) + relTicks;
                
                // Normalize to [0, 1]
                float v = textureRow / uHeight;
                
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
                   gl_FragColor = data / 255.0; 
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
                uWidth: { value: width },
                uHeight: { value: height }
            },
            vertexShader,
            fragmentShader,
            transparent: true,
            side: THREE.DoubleSide,
            blending: type === 'gex' ? THREE.AdditiveBlending : THREE.NormalBlending,
            depthWrite: type !== 'gex'
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

        // Clear Column Data
        this.clearColumn(this.head);

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
        for (let y = 0; y < this.height; y++) {
            const idx = y * stride + start;
            this.data[idx] = 0;
            this.data[idx + 1] = 0;
            this.data[idx + 2] = 0;
            this.data[idx + 3] = 0;
        }
        this.texture.needsUpdate = true;
    }

    write(relTicks: number, vector: [number, number, number, number]): void {
        const centerY = Math.floor(this.height / 2);
        const y = centerY + relTicks;
        if (y < 0 || y >= this.height) return;

        const idx = (y * this.width + this.head) * 4;
        this.data[idx] = vector[0];
        this.data[idx + 1] = vector[1];
        this.data[idx + 2] = vector[2];
        this.data[idx + 3] = vector[3];
        this.texture.needsUpdate = true;
    }
}


