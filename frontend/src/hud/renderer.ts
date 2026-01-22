/**
 * HUD Renderer - Professional WebGL visualization using Three.js
 * 
 * TradingView/Bookmap-style features:
 * - Price grid with $0.25 ticks (thin) and $1.00 ticks (bold)
 * - Dynamic spot line following actual price
 * - 15% future void on right side for predictions
 * - Zoom/pan support
 * - Heatmap aligned to actual price levels
 */

import * as THREE from 'three';
import { HUDState } from './state';

// Chart constants
const TICK_SIZE = 0.25;           // ES tick size
const DOLLAR_TICK = 1.0;          // Bold grid every $1
const FUTURE_VOID_PERCENT = 0.15; // 15% of right side for predictions
const DEFAULT_PRICE_RANGE = 20;   // Default visible price range in points

export class HUDRenderer {
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene;
    private camera: THREE.OrthographicCamera;
    private state: HUDState;

    // Visual element groups
    private gridGroup: THREE.Group;
    private heatmapGroup: THREE.Group;
    private priceLineGroup: THREE.Group;
    private overlayGroup: THREE.Group;

    // View state
    private viewCenter: { x: number; y: number } = { x: 0, y: 0 };
    private zoomLevel: number = 1.0;
    private priceRange: { min: number; max: number } = { min: 0, max: 0 };

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

        // Initialize orthographic camera
        const aspect = canvas.clientWidth / canvas.clientHeight;
        const viewHeight = DEFAULT_PRICE_RANGE;
        this.camera = new THREE.OrthographicCamera(
            -viewHeight * aspect / 2,
            viewHeight * aspect / 2,
            viewHeight / 2,
            -viewHeight / 2,
            0.1,
            1000
        );
        this.camera.position.z = 100;

        // Create groups for layered rendering
        this.gridGroup = new THREE.Group();
        this.heatmapGroup = new THREE.Group();
        this.priceLineGroup = new THREE.Group();
        this.overlayGroup = new THREE.Group();

        // Add groups in z-order
        this.scene.add(this.gridGroup);
        this.scene.add(this.heatmapGroup);
        this.scene.add(this.priceLineGroup);
        this.scene.add(this.overlayGroup);

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
        this.onResize();

        // Add mouse wheel zoom
        canvas.addEventListener('wheel', (e) => this.onWheel(e));
    }

    private onResize(): void {
        const container = this.renderer.domElement.parentElement;
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        this.renderer.setSize(width, height);
        this.updateCamera();
    }

    private onWheel(e: WheelEvent): void {
        e.preventDefault();
        const zoomDelta = e.deltaY > 0 ? 1.1 : 0.9;
        this.zoomLevel = Math.max(0.1, Math.min(10, this.zoomLevel * zoomDelta));
        this.updateCamera();
        this.updateVisualization();
    }

    private updateCamera(): void {
        const container = this.renderer.domElement.parentElement;
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;
        const aspect = width / height;

        const viewHeight = DEFAULT_PRICE_RANGE / this.zoomLevel;
        const viewWidth = viewHeight * aspect;

        this.camera.left = -viewWidth / 2 + this.viewCenter.x;
        this.camera.right = viewWidth / 2 + this.viewCenter.x;
        this.camera.top = viewHeight / 2 + this.viewCenter.y;
        this.camera.bottom = -viewHeight / 2 + this.viewCenter.y;
        this.camera.updateProjectionMatrix();
    }

    setZoom(level: number): void {
        this.zoomLevel = Math.max(0.1, Math.min(10, level));
        this.updateCamera();
        this.updateVisualization();
    }

    private clearGroup(group: THREE.Group): void {
        while (group.children.length > 0) {
            const child = group.children[0];
            group.remove(child);
            if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
        }
    }

    private createPriceGrid(): void {
        this.clearGroup(this.gridGroup);

        const { min: minPrice, max: maxPrice } = this.priceRange;
        if (minPrice === maxPrice) return;

        const viewWidth = (this.camera.right - this.camera.left);
        const chartWidth = viewWidth * (1 - FUTURE_VOID_PERCENT);
        const leftEdge = this.camera.left;
        const rightEdge = leftEdge + chartWidth;

        // $0.25 tick lines (thin, subtle)
        const tickMaterial = new THREE.LineBasicMaterial({
            color: 0x1a1a2e,
            transparent: true,
            opacity: 0.3
        });

        // $1.00 tick lines (bolder)
        const dollarMaterial = new THREE.LineBasicMaterial({
            color: 0x2a2a4e,
            transparent: true,
            opacity: 0.6
        });

        // Draw price grid lines
        const startPrice = Math.floor(minPrice / TICK_SIZE) * TICK_SIZE;
        const endPrice = Math.ceil(maxPrice / TICK_SIZE) * TICK_SIZE;

        for (let price = startPrice; price <= endPrice; price += TICK_SIZE) {
            const y = this.priceToY(price);
            const isDollarLine = Math.abs(price % DOLLAR_TICK) < 0.001;
            const material = isDollarLine ? dollarMaterial : tickMaterial;

            const points = [
                new THREE.Vector3(leftEdge, y, -1),
                new THREE.Vector3(rightEdge, y, -1)
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, material);
            this.gridGroup.add(line);
        }

        // Draw future void separator
        const voidMaterial = new THREE.LineBasicMaterial({
            color: 0x00ccff,
            transparent: true,
            opacity: 0.3
        });
        const voidPoints = [
            new THREE.Vector3(rightEdge, minPrice - 10, 0),
            new THREE.Vector3(rightEdge, maxPrice + 10, 0)
        ];
        const voidGeometry = new THREE.BufferGeometry().setFromPoints(voidPoints);
        const voidLine = new THREE.Line(voidGeometry, voidMaterial);
        this.gridGroup.add(voidLine);
    }

    private priceToY(price: number): number {
        // Map price to Y coordinate (price IS the Y coordinate in our system)
        const midPrice = (this.priceRange.min + this.priceRange.max) / 2;
        return price - midPrice;
    }

    private timeToX(timeIdx: number, totalWindows: number): number {
        // Map time index to X coordinate
        // Leave 15% on the right for future predictions
        const viewWidth = (this.camera.right - this.camera.left);
        const chartWidth = viewWidth * (1 - FUTURE_VOID_PERCENT);
        const leftEdge = this.camera.left;

        return leftEdge + (timeIdx / Math.max(1, totalWindows - 1)) * chartWidth;
    }

    updateVisualization(): void {
        const gexData = this.state.getGexData();
        if (gexData.length === 0) return;

        // Calculate ranges from data
        const allWindows = this.state.getTimeWindows();
        const MAX_WINDOWS = 60;
        const windows = allWindows.slice(-MAX_WINDOWS);

        // Get spot prices and calculate price range
        const spotByWindow = new Map<bigint, number>();
        for (const row of gexData) {
            if (!spotByWindow.has(row.window_end_ts_ns)) {
                spotByWindow.set(row.window_end_ts_ns, Number(row.underlying_spot_ref));
            }
        }

        const spots: number[] = [];
        for (const w of windows) {
            const spot = spotByWindow.get(w);
            if (spot) spots.push(spot);
        }

        if (spots.length === 0) return;

        const minSpot = Math.min(...spots);
        const maxSpot = Math.max(...spots);
        const spotPadding = Math.max((maxSpot - minSpot) * 0.2, 5); // At least 5 points padding

        this.priceRange = {
            min: minSpot - spotPadding,
            max: maxSpot + spotPadding
        };

        // Center view on current price
        this.viewCenter.y = 0; // Keep centered on mid-price

        // Rebuild visualizations
        this.createPriceGrid();
        this.createHeatmap(gexData, windows);
        this.createPriceLine(spots, windows.length);
    }

    private createHeatmap(gexData: { window_end_ts_ns: bigint; strike_points: number; gex_abs: number; gex_imbalance_ratio: number }[], windows: bigint[]): void {
        this.clearGroup(this.heatmapGroup);

        const uniqueStrikes = [...new Set(gexData.map(r => Number(r.strike_points)))].sort((a, b) => a - b);
        const numStrikes = uniqueStrikes.length;
        const numWindows = windows.length;

        if (numWindows === 0 || numStrikes === 0) return;

        // Build strike lookup
        const strikeToIdx = new Map<number, number>();
        uniqueStrikes.forEach((strike, idx) => strikeToIdx.set(strike, idx));

        // Create texture
        const textureWidth = numWindows;
        const textureHeight = numStrikes;
        const data = new Uint8Array(textureWidth * textureHeight * 4);

        const maxGexAbs = Math.max(...gexData.map(r => Number(r.gex_abs)), 1);

        for (let t = 0; t < numWindows; t++) {
            const windowTs = windows[t];

            for (const row of gexData) {
                if (row.window_end_ts_ns !== windowTs) continue;

                const strike = Number(row.strike_points);
                const s = strikeToIdx.get(strike);
                if (s === undefined) continue;

                const idx = (s * textureWidth + t) * 4;
                const intensity = Math.min(Number(row.gex_abs) / maxGexAbs, 1);
                const imbalance = Number(row.gex_imbalance_ratio);

                // Gamma-corrected intensity for better visibility
                const boosted = Math.pow(intensity, 0.4);

                if (imbalance > 0) {
                    // Call dominated - cyan/green
                    data[idx] = Math.floor(boosted * 30);
                    data[idx + 1] = Math.floor(boosted * 255);
                    data[idx + 2] = Math.floor(boosted * 180);
                } else {
                    // Put dominated - magenta/red
                    data[idx] = Math.floor(boosted * 255);
                    data[idx + 1] = Math.floor(boosted * 30);
                    data[idx + 2] = Math.floor(boosted * 150);
                }
                data[idx + 3] = Math.floor(boosted * 180 + 40);
            }
        }

        const texture = new THREE.DataTexture(data, textureWidth, textureHeight, THREE.RGBAFormat);
        texture.needsUpdate = true;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;

        // Map heatmap to chart area
        const viewWidth = (this.camera.right - this.camera.left);
        const chartWidth = viewWidth * (1 - FUTURE_VOID_PERCENT);
        const leftEdge = this.camera.left;

        const minStrike = uniqueStrikes[0];
        const maxStrike = uniqueStrikes[numStrikes - 1];
        const strikeRange = maxStrike - minStrike || 1;

        const planeWidth = chartWidth;
        const planeHeight = strikeRange;

        const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            side: THREE.DoubleSide,
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.x = leftEdge + chartWidth / 2;
        mesh.position.y = this.priceToY((minStrike + maxStrike) / 2);
        mesh.position.z = 0;
        this.heatmapGroup.add(mesh);
    }

    private createPriceLine(spots: number[], numWindows: number): void {
        this.clearGroup(this.priceLineGroup);

        if (spots.length < 2) return;

        // Create the dynamic price line
        const points: THREE.Vector3[] = [];
        for (let i = 0; i < spots.length; i++) {
            const x = this.timeToX(i, numWindows);
            const y = this.priceToY(spots[i]);
            points.push(new THREE.Vector3(x, y, 1));
        }

        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            linewidth: 2
        });
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const priceLine = new THREE.Line(lineGeometry, lineMaterial);
        this.priceLineGroup.add(priceLine);

        // Add current price marker
        const lastX = this.timeToX(spots.length - 1, numWindows);
        const lastY = this.priceToY(spots[spots.length - 1]);

        const markerGeom = new THREE.CircleGeometry(0.3 / this.zoomLevel, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        const marker = new THREE.Mesh(markerGeom, markerMat);
        marker.position.set(lastX, lastY, 2);
        this.priceLineGroup.add(marker);

        // Add horizontal price level line at current price
        const priceLineMaterial = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.5
        });
        const priceLevelPoints = [
            new THREE.Vector3(this.camera.left, lastY, 1),
            new THREE.Vector3(this.camera.right, lastY, 1)
        ];
        const priceLevelGeometry = new THREE.BufferGeometry().setFromPoints(priceLevelPoints);
        const priceLevelLine = new THREE.Line(priceLevelGeometry, priceLineMaterial);
        this.priceLineGroup.add(priceLevelLine);
    }

    centerView(): void {
        this.viewCenter = { x: 0, y: 0 };
        this.zoomLevel = 1.0;
        this.updateCamera();
        this.updateVisualization();
    }

    render(): void {
        if (this.state.getGexData().length > 0 && this.heatmapGroup.children.length === 0) {
            this.updateVisualization();
        }
        this.renderer.render(this.scene, this.camera);
    }
}
