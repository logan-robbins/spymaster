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

    private updateOverlay(): void {
        const priceAxis = document.getElementById('price-axis');
        const timeAxis = document.getElementById('time-axis');
        if (!priceAxis || !timeAxis) return;

        // Clear existing labels
        priceAxis.innerHTML = '';
        timeAxis.innerHTML = '';

        const { min: minPrice, max: maxPrice } = this.priceRange;
        const viewHeight = this.camera.top - this.camera.bottom;

        // Determine tick interval based on zoom/visible range
        let tickInterval = 5.0;
        if (viewHeight <= 10) tickInterval = 0.25;
        else if (viewHeight <= 30) tickInterval = 1.0;

        // --- Price Labels ---
        const startPrice = Math.floor(minPrice / tickInterval) * tickInterval;
        const endPrice = Math.ceil(maxPrice / tickInterval) * tickInterval;

        for (let price = startPrice; price <= endPrice; price += tickInterval) {
            // Only show if within view
            if (price < minPrice || price > maxPrice) continue;

            const y = this.priceToY(price);
            // Convert Y world coord to CSS % (relative to container height)
            // Camera top is +viewHeight/2, bottom is -viewHeight/2 (if center y=0)
            // But center y moves with priceToY logic? 
            // this.priceToY maps price -> world Y relative to viewCenter.y logic?
            // Actually priceToY: return price - midPrice; 
            // Camera top = viewHeight/2 + center.y
            // Camera bottom = -viewHeight/2 + center.y

            const normalizedY = (y - (this.camera.bottom)) / (this.camera.top - this.camera.bottom);
            const topPct = (1 - normalizedY) * 100; // CSS top is from top

            if (topPct < -5 || topPct > 105) continue;

            const el = document.createElement('div');
            el.className = 'price-label';
            // Highlight current price (approximate)
            if (Math.abs(price - this.state.getSpotRef()) < tickInterval / 2) {
                el.className += ' current';
            }
            el.textContent = price.toFixed(2);
            el.style.top = `${topPct}%`;
            el.style.position = 'absolute';
            // el.style.transform = 'translateY(-50%)'; // handled by CSS usually or add here
            el.style.transform = 'translateY(-50%)';
            priceAxis.appendChild(el);
        }

        // --- Time Labels ---
        const windows = this.state.getTimeWindows();
        const numWindows = windows.length;
        if (numWindows < 2) return;

        // Show ~5-8 time labels
        const timeStep = Math.max(1, Math.floor(numWindows / 6));

        for (let i = 0; i < numWindows; i += timeStep) {
            const x = this.timeToX(i, numWindows);
            const normalizedX = (x - this.camera.left) / (this.camera.right - this.camera.left);
            const leftPct = normalizedX * 100;

            if (leftPct < -5 || leftPct > 105) continue;

            const ts = windows[i];
            const date = new Date(Number(ts) / 1e6); // ns -> ms
            // EST 12-hour format
            const timeStr = date.toLocaleTimeString('en-US', {
                timeZone: 'America/New_York',
                hour: 'numeric',
                minute: '2-digit',
                second: '2-digit', // optional
                hour12: true
            });

            const el = document.createElement('div');
            el.className = 'time-label';
            el.textContent = timeStr;
            el.style.left = `${leftPct}%`;
            timeAxis.appendChild(el);
        }
    }

    private createPriceGrid(): void {
        this.clearGroup(this.gridGroup);

        const { min: minPrice, max: maxPrice } = this.priceRange;
        if (minPrice === maxPrice) return;

        const viewHeight = this.camera.top - this.camera.bottom;

        // Determine tick interval
        let tickInterval = 5.0;
        if (viewHeight <= 10) tickInterval = 0.25;
        else if (viewHeight <= 30) tickInterval = 1.0;

        const viewWidth = (this.camera.right - this.camera.left);
        const chartWidth = viewWidth * (1 - FUTURE_VOID_PERCENT);
        const leftEdge = this.camera.left;
        const rightEdge = leftEdge + chartWidth;

        // Grid lines
        const materials = {
            0.25: new THREE.LineBasicMaterial({ color: 0x1a1a2e, transparent: true, opacity: 0.3 }),
            1.0: new THREE.LineBasicMaterial({ color: 0x2a2a4e, transparent: true, opacity: 0.5 }),
            5.0: new THREE.LineBasicMaterial({ color: 0x3a3a5e, transparent: true, opacity: 0.7 })
        };

        // Optimization: Only draw lines that match current zoom?
        // User asked for scalable increments.
        // Let's draw current interval lines (e.g. at $1 zoom, only draw $1 lines?)
        // Or draw finer lines faintly?
        // Let's draw the chosen interval.

        const drawStart = Math.floor(minPrice / tickInterval) * tickInterval;
        const drawEnd = Math.ceil(maxPrice / tickInterval) * tickInterval;

        const mat = materials[tickInterval as keyof typeof materials] || materials[5.0];

        for (let price = drawStart; price <= drawEnd; price += tickInterval) {
            const y = this.priceToY(price);
            const points = [
                new THREE.Vector3(leftEdge, y, -1),
                new THREE.Vector3(rightEdge, y, -1)
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, mat);
            this.gridGroup.add(line);
        }

        // Draw future void separator
        const voidMaterial = new THREE.LineBasicMaterial({
            color: 0x00ccff,
            transparent: true,
            opacity: 0.3
        });
        const voidPoints = [
            new THREE.Vector3(rightEdge, this.camera.bottom, 0),
            new THREE.Vector3(rightEdge, this.camera.top, 0)
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
        const spotData = this.state.getSpotData();

        if (gexData.length === 0 && spotData.length === 0) return;

        // Calculate ranges from data
        const allWindows = this.state.getTimeWindows();
        const MAX_WINDOWS = 60;
        const windows = allWindows.slice(-MAX_WINDOWS);

        // Get spot prices
        const spotByWindow = this.state.getSpotsByTime();
        const spots: number[] = [];
        for (const w of windows) {
            const spot = spotByWindow.get(w);
            if (spot !== undefined) spots.push(spot);
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
        if (gexData.length > 0) {
            this.createHeatmap(gexData, windows);
        } else {
            this.clearGroup(this.heatmapGroup);
        }
        this.createPriceLine(spots, windows.length);
        this.updateOverlay();
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

        // Create smoothed path using CatmullRomCurve3
        const points: THREE.Vector3[] = [];
        for (let i = 0; i < spots.length; i++) {
            const x = this.timeToX(i, numWindows);
            const y = this.priceToY(spots[i]);
            points.push(new THREE.Vector3(x, y, 1));
        }

        const curve = new THREE.CatmullRomCurve3(points);
        const curvePoints = curve.getPoints(spots.length * 5); // 5x resolution for smoothness
        const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);

        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            linewidth: 2
        });

        const priceLine = new THREE.Line(geometry, lineMaterial);
        this.priceLineGroup.add(priceLine);

        // Add current price marker (Worm head)
        const lastPt = points[points.length - 1];

        // 3-4px dot. In ortho view, size is constant in world units? 
        // No, in ortho, world units size = screen size * zoom?
        // Actually MeshBasicMaterial size is in world units.
        // We need it to be 3-4px screen size.
        // viewHeight corresponds to canvas height.
        // worldUnitPerPixel = viewHeight / canvasHeight.
        const canvasHeight = this.renderer.domElement.clientHeight;
        const pixelSize = (this.camera.top - this.camera.bottom) / canvasHeight;
        const dotWorldSize = 4 * pixelSize; // 4 pixels

        const markerGeom = new THREE.CircleGeometry(dotWorldSize, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        const marker = new THREE.Mesh(markerGeom, markerMat);
        marker.position.copy(lastPt);
        marker.position.z = 2;
        this.priceLineGroup.add(marker);

        // Add horizontal price level line at current price
        const priceLineMaterial = new THREE.LineDashedMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.5,
            dashSize: 0.5,
            gapSize: 0.5
        });
        const priceLevelPoints = [
            new THREE.Vector3(this.camera.left, lastPt.y, 1),
            new THREE.Vector3(this.camera.right, lastPt.y, 1)
        ];
        const priceLevelGeometry = new THREE.BufferGeometry().setFromPoints(priceLevelPoints);
        const priceLevelLine = new THREE.Line(priceLevelGeometry, priceLineMaterial);
        priceLevelLine.computeLineDistances();
        this.priceLineGroup.add(priceLevelLine);
    }

    centerView(): void {
        this.viewCenter = { x: 0, y: 0 };
        this.zoomLevel = 1.0;
        this.updateCamera();
        this.updateVisualization();
    }

    render(): void {
        // Continually update for smooth animations if we add them, 
        // but efficiently only if needed. 
        // For now, overlay update needs to happen on render to track camera changes/zoom
        if (this.state.getGexData().length > 0 || this.state.getSpotData().length > 0) {
            this.updateOverlay();
        }

        this.renderer.render(this.scene, this.camera);
    }
    dispose(): void {
        this.renderer.dispose();
        this.gridGroup.clear();
        this.heatmapGroup.clear();
        this.priceLineGroup.clear();
        this.overlayGroup.clear();
    }
}
