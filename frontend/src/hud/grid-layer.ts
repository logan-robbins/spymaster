import * as THREE from 'three';

export class GridLayer {
    private width: number;   // columns = seconds of history
    private height: number;  // rows = ticks (1 tick = $0.25)
    private head: number = 0;
    private texture: THREE.DataTexture;
    private data: Uint8Array;
    private mesh: THREE.Mesh;
    private material: THREE.ShaderMaterial;

    constructor(width: number, height: number) {
        this.width = width;   // 1 column = 1 second
        this.height = height; // 1 row = 1 tick ($0.25)

        // Initialize Ref Count / ID Buffer? No, simple accumulation.
        // RGBA: B=Intensity, G=Alpha/Type? 
        // Let's stick to simple visually: 
        // - We write RGB for color, A for strength?
        // - Or simple intensity map and shader creates color?

        // Let's use 4 channels: R, G, B, A
        this.data = new Uint8Array(width * height * 4);
        this.texture = new THREE.DataTexture(this.data, width, height);
        this.texture.format = THREE.RGBAFormat;
        this.texture.type = THREE.UnsignedByteType;
        this.texture.minFilter = THREE.NearestFilter;
        this.texture.magFilter = THREE.NearestFilter;
        this.texture.wrapS = THREE.RepeatWrapping; // Important for scrolling
        this.texture.wrapT = THREE.ClampToEdgeWrapping;
        this.texture.needsUpdate = true;

        // Custom Shader for Scrolling
        const vertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            uniform sampler2D map;
            uniform float uHeadOffset; // (head + 1) / width
            uniform float uOpacity;
            varying vec2 vUv;

            void main() {
                // Scroll: x=0 maps to Oldest (uHead + 1)
                // x=1 maps to Newest (uHead)
                float x = vUv.x + uHeadOffset;
                
                // Read from texture
                vec4 texColor = texture2D(map, vec2(x, vUv.y));
                
                // Add fade/scanline?
                // For now, raw color
                
                if (texColor.a == 0.0) discard;
                
                gl_FragColor = texColor * uOpacity;
            }
        `;

        this.material = new THREE.ShaderMaterial({
            uniforms: {
                map: { value: this.texture },
                uHeadOffset: { value: 0.0 },
                uOpacity: { value: 0.9 }
            },
            vertexShader,
            fragmentShader,
            transparent: true,
            side: THREE.DoubleSide
        });

        // Initialize Plane
        const geometry = new THREE.PlaneGeometry(1, 1); // Unit square, scaled later
        this.mesh = new THREE.Mesh(geometry, this.material);
        this.mesh.frustumCulled = false;
    }

    getMesh(): THREE.Mesh {
        return this.mesh;
    }

    advance(): void {
        this.head = (this.head + 1) % this.width;
        this.clearColumn(this.head);

        // Update Uniform
        // Oldest is head + 1
        const offset = (this.head + 1) / this.width;
        this.material.uniforms.uHeadOffset.value = offset;
    }

    clearColumn(colIdx: number): void {
        const start = colIdx * 4;
        const stride = this.width * 4;

        // Clear entire column
        for (let y = 0; y < this.height; y++) {
            const idx = y * stride + start;
            this.data[idx] = 0;
            this.data[idx + 1] = 0;
            this.data[idx + 2] = 0;
            this.data[idx + 3] = 0;
        }
        this.texture.needsUpdate = true;
    }

    /**
     * Write a cell at the current time column
     * @param relTicks - Relative ticks from spot (0 is center, +N above, -N below)
     * @param color - [r, g, b, a] 0-255
     */
    write(relTicks: number, color: [number, number, number, number]): void {
        // Map relTicks to Y index
        // Center of texture is height/2 (spot price line)
        const centerY = Math.floor(this.height / 2);
        const y = centerY + relTicks;

        if (y < 0 || y >= this.height) return;

        const idx = (y * this.width + this.head) * 4;
        this.data[idx] = color[0];
        this.data[idx + 1] = color[1];
        this.data[idx + 2] = color[2];
        this.data[idx + 3] = color[3];

        this.texture.needsUpdate = true;
    }
}
