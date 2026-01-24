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
import { GridLayer } from './grid-layer';

const FUTURE_VOID_PERCENT = 0.15; // 15% of right side for predictions
const DEFAULT_PRICE_RANGE = 6;    // +/- 12 ticks = +/- 3 points -> Total 6
const HISTORY_SECONDS = 1800;     // 30 minutes

export class HUDRenderer {
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene;
    private camera: THREE.OrthographicCamera;
    private state: HUDState;

    // Layers
    private wallLayer: GridLayer;
    private vacuumLayer: GridLayer;

    // Groups
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

        // Initialize camera
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

        // Initialize Layers
        const LAYER_HEIGHT = 800;
        this.wallLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT);
        this.vacuumLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT);

        // Groups
        this.gridGroup = new THREE.Group();
        this.heatmapGroup = new THREE.Group();
        this.priceLineGroup = new THREE.Group();
        this.overlayGroup = new THREE.Group();

        // Add to scene (Order handled by Z-position)
        this.scene.add(this.gridGroup);

        // Add GridLayers
        const vacuumMesh = this.vacuumLayer.getMesh();
        this.scene.add(vacuumMesh);

        const wallMesh = this.wallLayer.getMesh();
        this.scene.add(wallMesh);

        this.scene.add(this.heatmapGroup);
        this.scene.add(this.priceLineGroup);
        this.scene.add(this.overlayGroup);

        // Resize
        window.addEventListener('resize', () => this.onResize());
        this.onResize();

        // Events
        canvas.addEventListener('wheel', (e) => this.onWheel(e));
    }

    advanceLayers(): void {
        this.wallLayer.advance();
        this.vacuumLayer.advance();
    }

    updateWall(data: any[]): void {
        for (const row of data) {
            if (typeof row.rel_ticks !== 'number') continue;

            let r = 0, g = 0, b = 0, a = 255;
            const strength = Math.log1p(Number(row.depth_qty_rest || 0));
            const maxLog = 12;

            const intensity = Math.min(strength / maxLog, 1.0);

            if (row.side === 'A' || row.side === 'ask') {
                // Blue
                r = 20; g = 50; b = 255;
                a = Math.floor(intensity * 180);
            } else {
                // Red
                r = 255; g = 50; b = 20;
                a = Math.floor(intensity * 180);
            }

            this.wallLayer.write(row.rel_ticks, [r, g, b, a]);
        }
    }

    updateVacuum(data: any[]): void {
        for (const row of data) {
            if (typeof row.rel_ticks !== 'number') continue;

            const score = Number(row.vacuum_score || 0);
            if (score < 0.05) continue;

            const alpha = Math.floor(score * 128);
            this.vacuumLayer.write(row.rel_ticks, [0, 0, 0, alpha]);
        }
    }

    private onResize(): void {
        const container = this.renderer.domElement.parentElement;
        if (!container) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        this.renderer.setSize(width, height);
        this.updateCamera();
        this.updateLayerTransforms();
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

        const aspect = container.clientWidth / container.clientHeight;
        const viewHeight = DEFAULT_PRICE_RANGE / this.zoomLevel;
        const viewWidth = viewHeight * aspect;

        this.camera.left = -viewWidth / 2 + this.viewCenter.x;
        this.camera.right = viewWidth / 2 + this.viewCenter.x;
        this.camera.top = viewHeight / 2 + this.viewCenter.y;
        this.camera.bottom = -viewHeight / 2 + this.viewCenter.y;
        this.camera.updateProjectionMatrix();

        this.updateLayerTransforms();
    }

    private updateLayerTransforms(): void {
        const SEC_PER_UNIT_X = 0.1;
        const width = HISTORY_SECONDS * SEC_PER_UNIT_X;
        const height = 200;

        const meshW = width;
        const meshH = height;

        // Update Vacuum
        const vMesh = this.vacuumLayer.getMesh();
        vMesh.scale.set(meshW, meshH, 1);
        vMesh.position.x = -meshW / 2;
        vMesh.position.y = 0;
        vMesh.position.z = -0.01; // Bottom

        // Update Wall
        const wMesh = this.wallLayer.getMesh();
        wMesh.scale.set(meshW, meshH, 1);
        wMesh.position.x = -meshW / 2;
        wMesh.position.y = 0;
        wMesh.position.z = 0; // Middle
    }

    setZoom(level: number): void {
        this.zoomLevel = level;
        this.updateCamera();
        this.updateVisualization();
    }

    centerView(): void {
        this.viewCenter = { x: 0, y: 0 };
        this.zoomLevel = 1.0;
        this.updateCamera();
        this.updateVisualization();
    }

    render(): void {
        if (this.state.getGexData().length > 0 || this.state.getSpotData().length > 0) {
            this.updateVisualization();
        }
        this.renderer.render(this.scene, this.camera);
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

    private priceToNormalizedY(price: number): number {
        const worldY = this.priceToY(price);
        const viewHeight = this.camera.top - this.camera.bottom;
        const cameraBottomWorld = this.camera.bottom;
        return (worldY - cameraBottomWorld) / viewHeight;
    }

    private updateOverlay(): void {
        const priceAxis = document.getElementById('price-axis');
        const timeAxis = document.getElementById('time-axis');
        if (!priceAxis || !timeAxis) return;

        priceAxis.innerHTML = '';
        timeAxis.innerHTML = '';

        const viewHeight = this.camera.top - this.camera.bottom;
        const canvasHeight = this.renderer.domElement.clientHeight;

        const targetPixelsPerTick = 50;
        const priceRangePerPixel = viewHeight / canvasHeight;
        const targetPriceStep = targetPixelsPerTick * priceRangePerPixel;
        const step = this.getNiceStep(targetPriceStep);

        const { min: minPrice, max: maxPrice } = this.priceRange;
        const startPrice = Math.floor((minPrice - step) / step) * step;
        const endPrice = Math.ceil((maxPrice + step) / step) * step;

        for (let price = startPrice; price <= endPrice; price += step) {
            const normY = this.priceToNormalizedY(price);
            const bottomPct = normY * 100;
            if (bottomPct < -5 || bottomPct > 105) continue;

            const el = document.createElement('div');
            el.className = 'price-label';
            if (Math.abs(price - this.state.getSpotRef()) < step / 2) {
                el.className += ' current';
            }
            el.textContent = price.toFixed(2);
            el.style.bottom = `${bottomPct}%`;
            el.style.position = 'absolute';
            el.style.transform = 'translateY(50%)';
            priceAxis.appendChild(el);
        }

        const windows = this.state.getTimeWindows();
        const numWindows = windows.length;
        if (numWindows < 2) return;

        const viewWidth = this.camera.right - this.camera.left;
        const chartWidth = viewWidth * (1 - FUTURE_VOID_PERCENT);
        const leftEdge = this.camera.left;
        const getX = (i: number) => leftEdge + (i / Math.max(1, numWindows - 1)) * chartWidth;

        const canvasWidth = this.renderer.domElement.clientWidth;
        const pixelsPerWindow = (canvasWidth * (1 - FUTURE_VOID_PERCENT)) / numWindows;
        const targetLabelWidth = 100;
        const windowStep = Math.ceil(targetLabelWidth / pixelsPerWindow);

        for (let i = 0; i < numWindows; i += windowStep) {
            const x = getX(i);
            const normX = (x - this.camera.left) / viewWidth;
            const leftPct = normX * 100;
            if (leftPct < -5 || leftPct > 100) continue;

            const ts = windows[i];
            const date = new Date(Number(ts) / 1e6);
            const timeStr = date.toLocaleTimeString('en-US', {
                timeZone: 'America/New_York',
                hour: 'numeric',
                minute: '2-digit',
                second: '2-digit',
                hour12: true
            });

            const el = document.createElement('div');
            el.className = 'time-label';
            el.textContent = timeStr;
            el.style.left = `${leftPct}%`;
            timeAxis.appendChild(el);
        }
    }

    private getNiceStep(target: number): number {
        const base = Math.pow(10, Math.floor(Math.log10(target)));
        const fraction = target / base;
        let niceFraction;
        if (fraction <= 1.0) niceFraction = 1.0;
        else if (fraction <= 2.5) niceFraction = 2.5;
        else if (fraction <= 5.0) niceFraction = 5.0;
        else niceFraction = 10.0;
        let step = niceFraction * base;
        if (step < 0.25) return 0.25;
        if (step === 0.25 || step === 0.5 || step >= 1.0) return step;
        return step;
    }

    private createPriceGrid(): void {
        this.clearGroup(this.gridGroup);
        const { min: minPrice, max: maxPrice } = this.priceRange;
        if (minPrice === maxPrice) return;

        const viewHeight = this.camera.top - this.camera.bottom;
        const canvasHeight = this.renderer.domElement.clientHeight;
        const targetPixelsPerTick = 50;
        const priceRangePerPixel = viewHeight / canvasHeight;
        const targetPriceStep = targetPixelsPerTick * priceRangePerPixel;
        const step = this.getNiceStep(targetPriceStep);

        const viewWidth = (this.camera.right - this.camera.left);
        const chartWidth = viewWidth * (1 - FUTURE_VOID_PERCENT);
        const leftEdge = this.camera.left;
        const rightEdge = leftEdge + chartWidth;
        // Forward Projection: Grid extends to camera.right
        const farRight = this.camera.right;

        const gridMat = new THREE.LineBasicMaterial({
            color: 0x3a3a5e,
            transparent: true,
            opacity: 0.2
        });

        const startPrice = Math.floor(minPrice / step) * step;
        const endPrice = Math.ceil(maxPrice / step) * step;
        const points: THREE.Vector3[] = [];
        for (let price = startPrice; price <= endPrice; price += step) {
            const y = this.priceToY(price);
            points.push(new THREE.Vector3(leftEdge, y, -1));
            // Extend to far right
            points.push(new THREE.Vector3(farRight, y, -1));
        }

        if (points.length > 0) {
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const lines = new THREE.LineSegments(geometry, gridMat);
            lines.position.z = 0.02; // Above everything
            this.gridGroup.add(lines);
        }

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
        const midPrice = (this.priceRange.min + this.priceRange.max) / 2;
        return price - midPrice;
    }

    private timeToX(timeIdx: number, totalWindows: number): number {
        const viewWidth = (this.camera.right - this.camera.left);
        const chartWidth = viewWidth * (1 - FUTURE_VOID_PERCENT);
        const leftEdge = this.camera.left;
        return leftEdge + (timeIdx / Math.max(1, totalWindows - 1)) * chartWidth;
    }

    updateVisualization(): void {
        const gexData = this.state.getGexData();

        const allWindows = this.state.getTimeWindows();
        const MAX_WINDOWS = 60;
        const windows = allWindows.slice(-MAX_WINDOWS);

        const spotByWindow = this.state.getSpotsByTime();
        const spots: number[] = [];
        for (const w of windows) {
            const spot = spotByWindow.get(w);
            if (spot !== undefined) spots.push(spot);
        }

        if (spots.length === 0 && gexData.length === 0) return;

        if (spots.length > 0) {
            const minSpot = Math.min(...spots);
            const maxSpot = Math.max(...spots);
            const spotPadding = Math.max((maxSpot - minSpot) * 0.2, 5);

            this.priceRange = {
                min: minSpot - spotPadding,
                max: maxSpot + spotPadding
            };
            // Center view on current price
            this.viewCenter.y = 0;
        }

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

        const strikeToIdx = new Map<number, number>();
        uniqueStrikes.forEach((strike, idx) => strikeToIdx.set(strike, idx));

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
                const boosted = Math.pow(intensity, 0.4);

                if (imbalance > 0) {
                    data[idx] = Math.floor(boosted * 20); // Low R
                    data[idx + 1] = Math.floor(boosted * 200); // High G
                    data[idx + 2] = Math.floor(boosted * 150); // Med B
                } else {
                    data[idx] = Math.floor(boosted * 200); // High R
                    data[idx + 1] = Math.floor(boosted * 20); // Low G
                    data[idx + 2] = Math.floor(boosted * 100); // Low B
                }
                // Drastically reduce alpha for additive blending
                data[idx + 3] = Math.floor(boosted * 60);
            }
        }
        const texture = new THREE.DataTexture(data, textureWidth, textureHeight, THREE.RGBAFormat);
        texture.needsUpdate = true;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;

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
            blending: THREE.AdditiveBlending, // Key change for visibility
            depthWrite: false
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.x = leftEdge + chartWidth / 2;
        mesh.position.y = this.priceToY((minStrike + maxStrike) / 2);
        mesh.position.z = 0.01; // Top layer
        this.heatmapGroup.add(mesh);
    }

    private createPriceLine(spots: number[], numWindows: number): void {
        this.clearGroup(this.priceLineGroup);
        if (spots.length < 2) return;
        const points: THREE.Vector3[] = [];
        for (let i = 0; i < spots.length; i++) {
            const x = this.timeToX(i, numWindows);
            const y = this.priceToY(spots[i]);
            points.push(new THREE.Vector3(x, y, 1));
        }
        const curve = new THREE.CatmullRomCurve3(points);
        const curvePoints = curve.getPoints(spots.length * 5);
        const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            linewidth: 2
        });
        const priceLine = new THREE.Line(geometry, lineMaterial);
        this.priceLineGroup.add(priceLine);

        const lastPt = points[points.length - 1];
        const canvasHeight = this.renderer.domElement.clientHeight;
        const pixelSize = (this.camera.top - this.camera.bottom) / canvasHeight;
        const dotWorldSize = 4 * pixelSize;
        const markerGeom = new THREE.CircleGeometry(dotWorldSize, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        const marker = new THREE.Mesh(markerGeom, markerMat);
        marker.position.copy(lastPt);
        marker.position.z = 2;
        this.priceLineGroup.add(marker);

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

    dispose(): void {
        this.renderer.dispose();
    }
}
