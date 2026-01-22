/**
 * HUD Renderer - WebGL visualization using Three.js
 * 
 * Renders:
 * - GEX Surface as a heatmap (time × strike)
 * - Spot price line
 * - Grid with labels
 */

import * as THREE from 'three';
import { HUDState } from './state';

// Constants removed - now using actual data from API

export class HUDRenderer {
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene;
    private camera: THREE.OrthographicCamera;
    private state: HUDState;

    // Visual elements
    private gexMesh: THREE.Mesh | null = null;
    private spotLine: THREE.Line | null = null;
    private gridGroup: THREE.Group;

    constructor(canvas: HTMLCanvasElement, state: HUDState) {
        this.state = state;

        // Initialize renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x0a0a0f, 1);

        // Initialize scene
        this.scene = new THREE.Scene();

        // Initialize orthographic camera for 2D view
        const aspect = canvas.clientWidth / canvas.clientHeight;
        const viewHeight = 200; // ES points
        this.camera = new THREE.OrthographicCamera(
            -viewHeight * aspect / 2,
            viewHeight * aspect / 2,
            viewHeight / 2,
            -viewHeight / 2,
            0.1,
            1000
        );
        this.camera.position.z = 100;

        // Grid group
        this.gridGroup = new THREE.Group();
        this.scene.add(this.gridGroup);

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
        this.onResize();

        // Create initial grid
        this.createGrid();
    }

    private onResize(): void {
        const container = this.renderer.domElement.parentElement;
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        this.renderer.setSize(width, height);

        const aspect = width / height;
        const viewHeight = 200;
        this.camera.left = -viewHeight * aspect / 2;
        this.camera.right = viewHeight * aspect / 2;
        this.camera.top = viewHeight / 2;
        this.camera.bottom = -viewHeight / 2;
        this.camera.updateProjectionMatrix();
    }

    private createGrid(): void {
        // Clear existing grid
        while (this.gridGroup.children.length > 0) {
            this.gridGroup.remove(this.gridGroup.children[0]);
        }

        const gridMaterial = new THREE.LineBasicMaterial({
            color: 0x1a1a2e,
            transparent: true,
            opacity: 0.5
        });

        // Horizontal lines (price levels)
        for (let i = -20; i <= 20; i++) {
            const y = i * 5; // Every 5 points
            const points = [
                new THREE.Vector3(-1000, y, 0),
                new THREE.Vector3(1000, y, 0)
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, gridMaterial);
            this.gridGroup.add(line);
        }

        // Vertical lines (time)
        for (let i = -100; i <= 100; i++) {
            const x = i * 10;
            const points = [
                new THREE.Vector3(x, -1000, 0),
                new THREE.Vector3(x, 1000, 0)
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, gridMaterial);
            this.gridGroup.add(line);
        }
    }

    private updateGexVisualization(): void {
        // Remove old mesh
        if (this.gexMesh) {
            this.scene.remove(this.gexMesh);
            this.gexMesh = null;
        }

        const gexData = this.state.getGexData();
        if (gexData.length === 0) {
            console.log('No GEX data');
            return;
        }

        console.log('GEX data count:', gexData.length);

        // Get unique strikes from the actual data
        const uniqueStrikes = [...new Set(gexData.map(r => Number(r.strike_points)))].sort((a, b) => a - b);
        const allWindows = this.state.getTimeWindows();

        // Limit to last 5 minutes of data (60 windows at 5s, or fewer at larger intervals)
        // Sample data is at 60s intervals, so 5 windows = 5 minutes
        const MAX_WINDOWS = 60; // At most 60 time slices
        const windows = allWindows.slice(-MAX_WINDOWS);

        console.log('Unique strikes:', uniqueStrikes.length, 'Windows:', windows.length);
        console.log('Strike range:', uniqueStrikes[0], '-', uniqueStrikes[uniqueStrikes.length - 1]);

        const numStrikes = uniqueStrikes.length;
        const numWindows = windows.length;

        if (numWindows === 0 || numStrikes === 0) {
            console.log('No windows or strikes');
            return;
        }

        // Create texture data
        const textureWidth = numWindows;
        const textureHeight = numStrikes;
        const data = new Uint8Array(textureWidth * textureHeight * 4);

        // Build a strike-to-index map
        const strikeToIdx = new Map<number, number>();
        uniqueStrikes.forEach((strike, idx) => strikeToIdx.set(strike, idx));

        // Find max GEX for normalization
        const maxGexAbs = Math.max(...gexData.map(r => Number(r.gex_abs)), 1);
        console.log('Max GEX abs:', maxGexAbs);

        let filledPixels = 0;

        // Build texture
        for (let t = 0; t < numWindows; t++) {
            const windowTs = windows[t]; // Already sliced to last N windows

            for (const row of gexData) {
                if (row.window_end_ts_ns !== windowTs) continue;

                const strike = Number(row.strike_points);
                const s = strikeToIdx.get(strike);
                if (s === undefined) continue;

                const idx = (s * textureWidth + t) * 4;
                const intensity = Math.min(Number(row.gex_abs) / maxGexAbs, 1);
                const imbalance = Number(row.gex_imbalance_ratio);

                // Boost intensity for visibility
                const boostedIntensity = Math.pow(intensity, 0.5) * 0.8 + 0.2;

                // Color: green for call-dominated, red for put-dominated
                if (imbalance > 0) {
                    data[idx] = Math.floor(boostedIntensity * 50);
                    data[idx + 1] = Math.floor(boostedIntensity * 255);
                    data[idx + 2] = Math.floor(boostedIntensity * 150);
                } else {
                    data[idx] = Math.floor(boostedIntensity * 255);
                    data[idx + 1] = Math.floor(boostedIntensity * 50);
                    data[idx + 2] = Math.floor(boostedIntensity * 150);
                }
                data[idx + 3] = Math.floor(boostedIntensity * 200 + 55);
                filledPixels++;
            }
        }

        console.log('Filled pixels:', filledPixels, 'of', textureWidth * textureHeight);

        // Create texture
        const texture = new THREE.DataTexture(
            data,
            textureWidth,
            textureHeight,
            THREE.RGBAFormat
        );
        texture.needsUpdate = true;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;

        // Create plane geometry - scale to fill view
        const planeWidth = 300; // Fill most of the horizontal view
        const planeHeight = 150; // Fill vertical view

        const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            side: THREE.DoubleSide,
        });

        this.gexMesh = new THREE.Mesh(geometry, material);
        this.gexMesh.position.x = 0; // Center
        this.gexMesh.position.y = 0; // Center
        this.gexMesh.position.z = 0; // In front of grid
        this.scene.add(this.gexMesh);

        console.log('GEX mesh added at center');

        // Update spot line
        this.updateSpotLine();
    }

    private updateSpotLine(): void {
        if (this.spotLine) {
            this.scene.remove(this.spotLine);
        }

        const gexData = this.state.getGexData();
        if (gexData.length === 0) return;

        // Get spot prices per time window
        const allWindows = this.state.getTimeWindows();
        const MAX_WINDOWS = 60;
        const windows = allWindows.slice(-MAX_WINDOWS);

        // Build map of window -> spot price
        const spotByWindow = new Map<bigint, number>();
        for (const row of gexData) {
            if (!spotByWindow.has(row.window_end_ts_ns)) {
                spotByWindow.set(row.window_end_ts_ns, Number(row.underlying_spot_ref));
            }
        }

        // Get spot values for our windows
        const spots: number[] = [];
        for (const w of windows) {
            const spot = spotByWindow.get(w);
            if (spot) spots.push(spot);
        }

        if (spots.length < 2) return;

        // Calculate price range for Y mapping
        const minSpot = Math.min(...spots);
        const maxSpot = Math.max(...spots);
        const spotRange = maxSpot - minSpot || 1;
        const midSpot = (maxSpot + minSpot) / 2;

        // Map to screen coordinates
        // X: from -140 to +130 (offset from right edge by 20 units)
        // Y: centered, scaled to fit view
        const xStart = -140;
        const xEnd = 130; // Leave room on right side
        const xRange = xEnd - xStart;
        const yScale = 100; // How much Y range to use for spot line

        const points: THREE.Vector3[] = [];
        for (let i = 0; i < spots.length; i++) {
            const x = xStart + (i / (spots.length - 1)) * xRange;
            const y = ((spots[i] - midSpot) / spotRange) * yScale;
            points.push(new THREE.Vector3(x, y, 2)); // z=2 to be in front
        }

        const material = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2 });
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        this.spotLine = new THREE.Line(geometry, material);
        this.scene.add(this.spotLine);

        // Add current price marker at the end
        const lastSpot = spots[spots.length - 1];
        const lastY = ((lastSpot - midSpot) / spotRange) * yScale;

        // Create a small sphere at current price
        const markerGeom = new THREE.CircleGeometry(3, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        const marker = new THREE.Mesh(markerGeom, markerMat);
        marker.position.set(xEnd, lastY, 2);
        this.scene.add(marker);

        console.log('Spot line: range', minSpot.toFixed(2), '-', maxSpot.toFixed(2), 'current:', lastSpot.toFixed(2));
    }

    centerView(): void {
        this.camera.position.x = 0;
        this.camera.position.y = 0;
        this.camera.updateProjectionMatrix();
    }

    render(): void {
        // Update visualization if data changed
        if (this.state.getGexData().length > 0 && !this.gexMesh) {
            this.updateGexVisualization();
        }

        this.renderer.render(this.scene, this.camera);
    }
}
