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
const TICK_SIZE = 0.25;

export class HUDRenderer {
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene;
    private camera: THREE.OrthographicCamera;
    private state: HUDState;

    // Layers
    private wallLayer: GridLayer;
    private vacuumLayer: GridLayer;
    private physicsLayer: GridLayer;
    private gexLayer: GridLayer;

    // Groups
    private gridGroup: THREE.Group;
    private priceLineGroup: THREE.Group;
    private overlayGroup: THREE.Group;

    // View state
    private viewCenter: { x: number; y: number } = { x: 0, y: 0 };
    private zoomLevel: number = 1.0;
    private priceRange: { min: number; max: number } = { min: 0, max: 0 };

    // World Scale Constants
    private readonly SEC_PER_UNIT_X = 0.1;
    private readonly UNIT_PER_DOLLAR = 1.0;

    // Robustness
    private lastValidSpot: number = 6000;

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
        const LAYER_HEIGHT = 801;
        this.wallLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT, 'wall');
        this.vacuumLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT, 'vacuum');
        this.physicsLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT, 'physics');
        this.gexLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT, 'gex');

        // Groups
        this.gridGroup = new THREE.Group();
        this.priceLineGroup = new THREE.Group();
        this.overlayGroup = new THREE.Group();

        // Add to scene (Order handled by Z-position)
        this.scene.add(this.gridGroup);

        // Add GridLayers (back to front: physics -> vacuum -> wall)
        // These are static meshes in World Space, we just position them once relative to "Now" (X=0)
        this.scene.add(this.physicsLayer.getMesh());
        this.scene.add(this.vacuumLayer.getMesh());
        this.scene.add(this.wallLayer.getMesh());
        this.scene.add(this.gexLayer.getMesh());

        this.scene.add(this.priceLineGroup);
        this.scene.add(this.overlayGroup);

        this.updateLayerTransforms();

        // Resize
        window.addEventListener('resize', () => this.onResize());
        this.onResize();

        // Events
        canvas.addEventListener('wheel', (e) => this.onWheel(e));
    }

    advanceLayers(): void {
        const t = performance.now() / 1000.0;
        const currentSpot = this.state.getSpotRef() || 6000;

        this.wallLayer.advance(t, currentSpot);
        this.vacuumLayer.advance(t, currentSpot);
        this.physicsLayer.advance(t, currentSpot);
        this.gexLayer.advance(t, currentSpot);
    }

    updateWall(data: any[]): void {
        for (const row of data) {
            if (typeof row.rel_ticks !== 'number') continue;
            const strength = Math.log1p(Number(row.depth_qty_rest || 0));
            const velocity = Number(row.d1_depth_qty || 0);
            const accel = Number(row.d2_depth_qty || 0);
            const sideCode = (row.side === 'A' || row.side === 'ask') ? 1.0 : -1.0;
            this.wallLayer.write(row.rel_ticks, [strength, velocity, accel, sideCode]);
        }
    }

    updateVacuum(data: any[]): void {
        for (const row of data) {
            if (typeof row.rel_ticks !== 'number') continue;
            const score = Number(row.vacuum_score || 0);
            if (score < 0.01) continue;
            const turbulence = Number(row.d2_pull_add_log || 0);
            const erosion = Number(row.wall_erosion || 0);
            this.vacuumLayer.write(row.rel_ticks, [score, turbulence, erosion, 1.0]);
        }
    }

    updateGex(data: any[]): void {
        if (!data || data.length === 0) return;
        const maxAbs = Math.max(...data.map(row => Math.abs(Number(row.gex_abs || 0))), 1);
        const fallbackSpot = this.state.getSpotRef();

        for (const row of data) {
            const strike = Number(row.strike_points);
            if (!Number.isFinite(strike)) continue;
            const spot = Number(row.underlying_spot_ref || row.spot_ref_price || fallbackSpot);
            if (!Number.isFinite(spot)) continue;

            const relTicks = Math.round((strike - spot) / TICK_SIZE);
            const gexAbs = Math.abs(Number(row.gex_abs || 0));
            const imbalance = Number(row.gex_imbalance_ratio || 0);
            const norm = Math.min(gexAbs / maxAbs, 1);
            if (norm <= 0) continue;

            const density = Math.pow(norm, 0.5);
            const band = Math.max(1, Math.round(density * 3));
            const alphaBase = Math.min(255, density * 220);

            let r = 180, g = 180, b = 180;
            if (imbalance >= 0) { r = 130; g = 200; b = 190; }
            else { r = 210; g = 150; b = 120; }

            const colorScale = 0.4 + (0.6 * density);
            r = Math.min(255, r * colorScale);
            g = Math.min(255, g * colorScale);
            b = Math.min(255, b * colorScale);

            for (let offset = -band; offset <= band; offset++) {
                const falloff = 1 - Math.abs(offset) / (band + 1);
                const alpha = alphaBase * falloff;
                this.gexLayer.write(relTicks + offset, [r * falloff, g * falloff, b * falloff, alpha]);
            }
        }
    }

    updatePhysics(data: any[]): void {
        if (!data || data.length === 0) return;
        const latest = data[data.length - 1];
        const aboveScore = Number(latest.above_score || 0);
        const belowScore = Number(latest.below_score || 0);
        const maxTicks = 100;

        for (let tick = 1; tick <= maxTicks; tick++) {
            const distanceFade = Math.max(0, 1 - tick / maxTicks);
            const intensity = aboveScore * distanceFade;
            if (intensity > 0.02) {
                const alpha = intensity * 60;
                this.physicsLayer.write(tick, [20, 180, 80, alpha]);
            }
        }
        for (let tick = 1; tick <= maxTicks; tick++) {
            const distanceFade = Math.max(0, 1 - tick / maxTicks);
            const intensity = belowScore * distanceFade;
            if (intensity > 0.02) {
                const alpha = intensity * 60;
                this.physicsLayer.write(-tick, [180, 40, 40, alpha]);
            }
        }
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

        // Horizontal Pan when shift is held or just wheel?
        // Let's allow X panning if we want history
        if (e.shiftKey) {
            const panDelta = (e.deltaY / 100) * (30 * this.SEC_PER_UNIT_X); // 30 seconds scan
            this.viewCenter.x += panDelta;
            // Clamp? allow looking back
            this.viewCenter.x = Math.min(0, Math.max(-HISTORY_SECONDS * this.SEC_PER_UNIT_X, this.viewCenter.x));
        }

        this.updateCamera();
        this.updateVisualization();
    }

    private updateCamera(): void {
        const container = this.renderer.domElement.parentElement;
        if (!container) return;

        const aspect = container.clientWidth / container.clientHeight;
        const viewHeight = DEFAULT_PRICE_RANGE / this.zoomLevel;
        const viewWidth = viewHeight * aspect;

        this.camera.left = -viewWidth * (1 - FUTURE_VOID_PERCENT) + this.viewCenter.x;
        this.camera.right = viewWidth * FUTURE_VOID_PERCENT + this.viewCenter.x;
        this.camera.top = viewHeight / 2 + this.viewCenter.y;
        this.camera.bottom = -viewHeight / 2 + this.viewCenter.y;

        this.camera.updateProjectionMatrix();
    }

    private updateLayerTransforms(): void {
        const width = HISTORY_SECONDS * this.SEC_PER_UNIT_X;
        const height = 200; // Physical vertical size (covers +/- $100 range roughly if Y scale is 1)

        // Position meshes so X=0 is the Right Edge (Latest)
        // Mesh center at -width/2
        const xPos = -width / 2;

        [this.physicsLayer, this.vacuumLayer, this.wallLayer, this.gexLayer].forEach((layer, idx) => {
            const mesh = layer.getMesh();
            mesh.scale.set(width, height, 1);
            mesh.position.set(xPos, 0, idx * 0.01);
        });

        // Set specific Z depths
        this.physicsLayer.getMesh().position.z = -0.02;
        this.vacuumLayer.getMesh().position.z = 0.015;
        this.wallLayer.getMesh().position.z = 0.0;
        this.gexLayer.getMesh().position.z = 0.01;
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

    private updateOverlay(currentSpot: number): void {
        const priceAxis = document.getElementById('price-axis');
        const timeAxis = document.getElementById('time-axis');
        if (!priceAxis || !timeAxis) return;

        priceAxis.innerHTML = '';
        timeAxis.innerHTML = '';

        const viewHeight = this.camera.top - this.camera.bottom;
        const canvasHeight = this.renderer.domElement.clientHeight;

        const targetPixelsPerTick = 80;
        const priceRangePerPixel = viewHeight / canvasHeight;
        const step = this.getNiceStep(targetPixelsPerTick * priceRangePerPixel);

        // Use 0.5 or 1.0 steps if possible when zoomed out
        let displayStep = step;
        if (this.zoomLevel < 1.0 && step < 0.5) displayStep = 0.5;

        // Absolute Price Bounds
        const minPrice = currentSpot + this.camera.bottom;
        const maxPrice = currentSpot + this.camera.top;
        const startPrice = Math.floor(minPrice / displayStep) * displayStep;

        for (let price = startPrice; price <= maxPrice; price += displayStep) {
            // World Y relative to Spot
            const y = price - currentSpot;

            // Normalized Screen Coordinates
            const normY = (y - this.camera.bottom) / viewHeight;
            const bottomPct = normY * 100;
            if (bottomPct < -5 || bottomPct > 105) continue;

            const el = document.createElement('div');
            el.className = 'price-label';
            if (Math.abs(price - currentSpot) < displayStep / 2) {
                el.className += ' current';
            }
            el.textContent = price.toFixed(2);
            el.style.bottom = `${bottomPct}%`;
            el.style.position = 'absolute';
            el.style.transform = 'translateY(50%)';
            priceAxis.appendChild(el);
        }

        // Time Axis
        const viewWidth = this.camera.right - this.camera.left;
        const worldMinX = this.camera.left;
        const worldMaxX = this.camera.right;

        // Determine step size (seconds)
        const canvasWidth = this.renderer.domElement.clientWidth;
        const secRange = viewWidth / this.SEC_PER_UNIT_X;
        const targetLabelPixels = 100;
        const targetNumLabels = canvasWidth / targetLabelPixels;
        const targetSecStep = secRange / targetNumLabels;

        // nice time steps: 1s, 5s, 10s, 30s, 60s
        let timeStep = 1;
        if (targetSecStep > 60) timeStep = 60;
        else if (targetSecStep > 30) timeStep = 30;
        else if (targetSecStep > 10) timeStep = 10;
        else if (targetSecStep > 5) timeStep = 5;
        else timeStep = Math.ceil(targetSecStep);

        // Current Timestamp (X=0)
        // We need an absolute reference. The state has latest window timestamp.
        const windows = this.state.getTimeWindows();
        if (windows.length === 0) return;
        const latestTs = Number(windows[windows.length - 1]);

        // Calculate "Now" in X is 0.
        // X = (t - latest) * scale
        // t = (X / scale) + latest

        const startX = worldMinX; // e.g. -10 (100 seconds ago)
        const endX = worldMaxX;   // e.g. +2 (20 seconds future)

        // Iterate X in TimeStep increments
        // Align to timeStep (e.g. 10s marks)
        // t_aligned = floor(t / step) * step

        const startT_ns = latestTs + (startX / this.SEC_PER_UNIT_X) * 1e9;
        const endT_ns = latestTs + (endX / this.SEC_PER_UNIT_X) * 1e9;

        const stepNs = timeStep * 1e9;
        const startT_aligned = Math.ceil(startT_ns / stepNs) * stepNs;

        for (let t = startT_aligned; t <= endT_ns; t += stepNs) {
            const offsetNs = t - latestTs;
            const x = (offsetNs / 1e9) * this.SEC_PER_UNIT_X;

            const normX = (x - worldMinX) / viewWidth;
            const leftPct = normX * 100;
            if (leftPct < -5 || leftPct > 100) continue;

            const date = new Date(t / 1e6);
            const timeStr = date.toLocaleTimeString('en-US', {
                timeZone: 'America/New_York',
                hour12: true,
                hour: 'numeric',
                minute: '2-digit',
                second: '2-digit'
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
        // Enforce quarter ticks
        if (step < 0.25) return 0.25;
        if (step === 0.25 || step === 0.5 || step >= 1.0) return step;
        return step;
    }

    private createPriceGrid(currentSpot: number): void {
        this.clearGroup(this.gridGroup);

        const viewHeight = this.camera.top - this.camera.bottom;
        const canvasHeight = this.renderer.domElement.clientHeight;
        const targetPixelsPerTick = 50;
        const priceRangePerPixel = viewHeight / canvasHeight;
        const step = this.getNiceStep(targetPixelsPerTick * priceRangePerPixel);

        const gridMat = new THREE.LineBasicMaterial({ color: 0x3a3a5e, transparent: true, opacity: 0.2 });

        // Absolute Price Bounds
        const minPrice = currentSpot + this.camera.bottom;
        const maxPrice = currentSpot + this.camera.top;
        const startPrice = Math.floor(minPrice / step) * step;

        const points: THREE.Vector3[] = [];
        const left = this.camera.left;
        const right = this.camera.right;

        for (let price = startPrice; price <= maxPrice; price += step) {
            const y = price - currentSpot;
            points.push(new THREE.Vector3(left, y, -1));
            points.push(new THREE.Vector3(right, y, -1));
        }

        if (points.length > 0) {
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const lines = new THREE.LineSegments(geometry, gridMat);
            lines.position.z = 0.02;
            this.gridGroup.add(lines);
        }

        // Draw Zero Line (Spot)
        const zeroPoints = [
            new THREE.Vector3(left, 0, 0),
            new THREE.Vector3(right, 0, 0)
        ];
        const zeroMat = new THREE.LineBasicMaterial({ color: 0x00ccff, transparent: true, opacity: 0.3 });
        const zeroLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(zeroPoints), zeroMat);
        this.gridGroup.add(zeroLine);
    }

    updateVisualization(): void {
        const allWindows = this.state.getTimeWindows();
        const spotByWindow = this.state.getSpotsByTime();
        if (allWindows.length === 0) return;

        // Current Spot (Anchor)
        // This is strictly the latest known spot price.
        // It defines Y=0.
        const snapData = this.state.getSpotData();
        const gexData = this.state.getGexData();

        let candidateSpot = 0;
        if (snapData.length > 0) candidateSpot = Number(snapData[snapData.length - 1].mid_price);
        else if (gexData.length > 0) candidateSpot = Number(gexData[gexData.length - 1].underlying_spot_ref || 0);

        // Update robust spot
        if (candidateSpot > 100) {
            this.lastValidSpot = candidateSpot;
        }

        const currentSpot = this.lastValidSpot;

        // Update Shaders with this Anchor
        this.wallLayer.setSpotRef(currentSpot);
        this.vacuumLayer.setSpotRef(currentSpot);
        this.physicsLayer.setSpotRef(currentSpot);
        this.gexLayer.setSpotRef(currentSpot);

        this.createPriceGrid(currentSpot);
        this.createPriceLine(allWindows, spotByWindow, currentSpot);
        this.updateOverlay(currentSpot);
    }

    private createPriceLine(windows: bigint[], spotMap: Map<bigint, number>, currentSpot: number): void {
        this.clearGroup(this.priceLineGroup);
        if (windows.length < 2) return;

        const latestTs = Number(windows[windows.length - 1]);
        const points: THREE.Vector3[] = [];

        // We can draw all history available in state (up to 1800s)
        // Optimization: Standardize stride or use camera bounds to clip?
        // For now, draw all valid points.

        const startIdx = Math.max(0, windows.length - 3600); // Max 1 hour points if dense?

        for (let i = startIdx; i < windows.length; i++) {
            const ts = windows[i];
            const price = spotMap.get(ts);
            if (price === undefined) continue;

            // X = Time Delta * Scale
            const deltaSec = (Number(ts) - latestTs) / 1e9;
            const x = deltaSec * this.SEC_PER_UNIT_X;

            // Y = Price Delta * Scale
            const y = (price - currentSpot) * this.UNIT_PER_DOLLAR;

            points.push(new THREE.Vector3(x, y, 1));
        }

        if (points.length < 2) return;

        // Price Line
        const curve = new THREE.CatmullRomCurve3(points);
        // Resample for smoothness? actually points are dense (1s). Line is fine.
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2 });
        const priceLine = new THREE.Line(geometry, lineMaterial);
        this.priceLineGroup.add(priceLine);

        // Current Price Marker
        const lastPt = points[points.length - 1]; // Should be (0, 0) if data is synced
        const canvasHeight = this.renderer.domElement.clientHeight;
        const pixelSize = (this.camera.top - this.camera.bottom) / canvasHeight;
        const dotWorldSize = 4 * pixelSize;

        const marker = new THREE.Mesh(
            new THREE.CircleGeometry(dotWorldSize, 16),
            new THREE.MeshBasicMaterial({ color: 0x00ffff })
        );
        marker.position.copy(lastPt);
        marker.position.z = 2;
        this.priceLineGroup.add(marker);

        // Price Level Line (Dashed)
        const priceLevelLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(this.camera.left, lastPt.y, 1),
                new THREE.Vector3(this.camera.right, lastPt.y, 1)
            ]),
            new THREE.LineDashedMaterial({ color: 0x00ffff, dashSize: 0.5, gapSize: 0.5, transparent: true, opacity: 0.5 })
        );
        priceLevelLine.computeLineDistances();
        this.priceLineGroup.add(priceLevelLine);
    }

    dispose(): void {
        this.renderer.dispose();
    }
}
