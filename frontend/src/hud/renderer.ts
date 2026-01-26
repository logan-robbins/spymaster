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
const PRICE_SCALE = 1e-9;
const TICK_INT = Math.round(TICK_SIZE / PRICE_SCALE);
const LAYER_HEIGHT_TICKS = 801; // +/- 400 ticks around spot

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
    private bucketRadarLayer: GridLayer;

    // Groups
    private gridGroup: THREE.Group;
    private priceLineGroup: THREE.Group;
    private overlayGroup: THREE.Group;

    // View state
    private viewCenter: { x: number; y: number } = { x: 0, y: 0 };
    private zoomLevel: number = 1.0;
    private autoCenter: boolean = true;
    private isUserPanning: boolean = false;

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
        // Height is now strictly in TICKS
        this.wallLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT_TICKS, 'wall');
        this.vacuumLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT_TICKS, 'vacuum');
        this.physicsLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT_TICKS, 'physics');
        this.gexLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT_TICKS, 'gex');
        this.bucketRadarLayer = new GridLayer(HISTORY_SECONDS, LAYER_HEIGHT_TICKS, 'bucket_radar', 2.0); // 2-tick resolution

        // Task 16: Dissipation Model
        this.physicsLayer.setDecay(5.0); // 5-second half-life
        this.vacuumLayer.setDecay(5.0);

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
        this.scene.add(this.bucketRadarLayer.getMesh());

        this.scene.add(this.priceLineGroup);
        this.scene.add(this.overlayGroup);

        this.updateLayerTransforms();

        // Resize
        window.addEventListener('resize', () => this.onResize());
        this.onResize();

        // Events
        // Events
        canvas.addEventListener('wheel', (e) => this.onWheel(e));

        // Task: Mouse Drag Panning
        canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        window.addEventListener('mousemove', (e) => this.onMouseMove(e));
        window.addEventListener('mouseup', () => this.onMouseUp());
    }

    private isDragging: boolean = false;
    private dragStart: { x: number; y: number } = { x: 0, y: 0 };

    private onMouseDown(e: MouseEvent): void {
        this.isDragging = true;
        this.dragStart = { x: e.clientX, y: e.clientY };
        this.isUserPanning = true;
        this.autoCenter = false;
    }

    private onMouseMove(e: MouseEvent): void {
        if (!this.isDragging) return;

        const deltaX = e.clientX - this.dragStart.x;
        const deltaY = e.clientY - this.dragStart.y;

        // Convert pixels to World Units
        // Y Axis:
        const container = this.renderer.domElement.parentElement;
        if (!container) return;

        const viewHeight = this.camera.top - this.camera.bottom;
        const pixelsPerUnitY = container.clientHeight / viewHeight;

        const viewWidth = this.camera.right - this.camera.left;
        const pixelsPerUnitX = container.clientWidth / viewWidth;

        this.viewCenter.x -= deltaX / pixelsPerUnitX;
        this.viewCenter.y += deltaY / pixelsPerUnitY;

        this.dragStart = { x: e.clientX, y: e.clientY };
        this.updateCamera();
        this.updateVisualization();
    }

    private onMouseUp(): void {
        this.isDragging = false;
    }

    advanceLayers(): void {
        const t = performance.now() / 1000.0;
        // Pass Spot Tick Index (Float space if sub-tick, but here we snap to grid)
        // Store int-based spot in texture
        const currentSpotInt = this.state.getSpotRefInt();
        const spotTick = Number(currentSpotInt) / TICK_INT;

        this.wallLayer.advance(t, spotTick);
        this.vacuumLayer.advance(t, spotTick);
        this.physicsLayer.advance(t, spotTick);
        this.gexLayer.advance(t, spotTick);
        this.bucketRadarLayer.advance(t, spotTick);
    }

    updateWall(data: any[], currentTs: bigint): void {
        const layer = this.wallLayer;

        for (const row of data) {
            const relTicks = Number(row.rel_ticks);
            if (isNaN(relTicks)) continue;

            const rowTs = BigInt(row.window_end_ts_ns || 0);
            if (rowTs !== currentTs) continue;

            const strength = Math.log1p(Number(row.depth_qty_rest || 0));
            const velocity = Number(row.d1_depth_qty || 0);
            const accel = Number(row.d2_depth_qty || 0);
            const sideCode = (row.side === 'A' || row.side === 'ask') ? 1.0 : -1.0;

            layer.write(relTicks, [strength, velocity, accel, sideCode]);
        }
    }

    updateVacuum(data: any[], currentTs: bigint): void {
        const layer = this.vacuumLayer;

        for (const row of data) {
            const relTicks = Number(row.rel_ticks);
            if (isNaN(relTicks)) continue;

            const rowTs = BigInt(row.window_end_ts_ns || 0);
            if (rowTs !== currentTs) continue;

            const score = Number(row.vacuum_score || 0);
            if (score < 0.01) continue;
            const turbulence = Number(row.d2_pull_add_log || 0);
            const erosion = Number(row.wall_erosion || 0);

            layer.write(relTicks, [score, turbulence, erosion, 1.0]);
        }
    }

    updateGex(data: any[], currentTs: bigint): void {
        if (!data || data.length === 0) return;
        const maxAbs = Math.max(...data.map(row => Math.abs(Number(row.gex_abs || 0))), 1);

        const layer = this.gexLayer;

        for (const row of data) {
            const relTicks = Number(row.rel_ticks);
            if (isNaN(relTicks)) continue;

            const rowTs = BigInt(row.window_end_ts_ns || 0);
            if (rowTs !== currentTs) continue;

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

            const rNorm = r / 255.0;
            const gNorm = g / 255.0;
            const bNorm = b / 255.0;
            const alphaNorm = alphaBase / 255.0;

            for (let offset = -band; offset <= band; offset++) {
                const falloff = 1 - Math.abs(offset) / (band + 1);
                const a = alphaNorm * falloff;
                layer.write(relTicks + offset, [rNorm * falloff, gNorm * falloff, bNorm * falloff, a]);
            }
        }
    }

    updatePhysics(data: any[], currentTs: bigint): void {
        const layer = this.physicsLayer;

        for (const row of data) {
            const rowTs = BigInt(row.window_end_ts_ns || 0);
            if (rowTs !== currentTs) continue;

            const relTicks = Number(row.rel_ticks);
            if (isNaN(relTicks)) continue;

            const signedScore = Number(row.physics_score_signed || 0);
            const intensity = Math.abs(signedScore);

            // Filter noise
            if (intensity < 0.05) continue;

            const alpha = Math.min(intensity * 1.5, 1.0);

            // Color Coding: Cyan (Up/Ease) vs Blue (Down/Ease)
            // Consistent with "Cool = Low Pressure / Vacuum"
            let r, g, b;
            if (signedScore > 0) {
                // Up Ease -> Cyan
                r = 0; g = 200; b = 255;
            } else {
                // Down Ease -> Blue/Indigo
                r = 60; g = 80; b = 255;
            }

            layer.write(relTicks, [r / 255.0, g / 255.0, b / 255.0, alpha]);
        }
    }

    updateBucketRadar(data: any[], currentTs: bigint): void {
        const layer = this.bucketRadarLayer;

        for (const row of data) {
            const rowTs = BigInt(row.window_end_ts_ns || 0);
            if (rowTs !== currentTs) continue;

            const bucketRel = Number(row.bucket_rel);
            if (isNaN(bucketRel)) continue;

            // Expand 2-tick bucket to relTicks
            // Bucket 0 -> Ticks 0, 1
            // Bucket -1 -> Ticks -2, -1
            const tickStart = bucketRel * 2;

            const blocked = Number(row.blocked_level || 0);
            const cavitation = Number(row.cavitation || 0);
            const gex = Number(row.gex_stiffness || 0);
            const mobility = Number(row.mobility || 0);

            // Write once (Layer handles resolution mapping)
            layer.write(tickStart, [blocked, cavitation, gex, mobility]);
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
            // Horizontal Pan (History)
            const panDelta = (e.deltaY / 100) * (30 * this.SEC_PER_UNIT_X);
            this.viewCenter.x += panDelta;
            this.viewCenter.x = Math.min(0, Math.max(-HISTORY_SECONDS * this.SEC_PER_UNIT_X, this.viewCenter.x));
            this.isUserPanning = true;
        } else {
            // Vertical Pan or Zoom? 
            // Ideally wheel is zoom, drag is pan.
            // For now let's assume wheel is zoom.
            // Reset auto-center if user manually pans Y (need logic for interaction)
            // Let's stick to Tracking by default unless we add mouse drag.
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
        // Height is now defined by Tick Height in World Units.
        // If 1.0 World Unit = $1.00 = 4 Ticks.
        // And Texture covers LAYER_HEIGHT_TICKS (801).
        // Total Height in World Units = (801 / 4) * 1.0? 
        // No, SEC_PER_UNIT_X = 0.1 means 10 sec = 1 unit.
        // UNIT_PER_DOLLAR = 1.0.
        // TICK_SIZE = 0.25 (dollars).
        // So 1 Tick = 0.25 World Units.
        // Texture Height in World Units = 801 * 0.25 = 200.25.

        const tickHeightWorld = TICK_SIZE * this.UNIT_PER_DOLLAR;
        const height = LAYER_HEIGHT_TICKS * tickHeightWorld;

        // Position meshes so X=0 is the Right Edge (Latest)
        // Mesh center at -width/2
        const xPos = -width / 2;

        [this.physicsLayer, this.vacuumLayer, this.wallLayer, this.gexLayer, this.bucketRadarLayer].forEach((layer, idx) => {
            const mesh = layer.getMesh();
            mesh.scale.set(width, height, 1);
            mesh.position.set(xPos, 0, idx * 0.01);
        });

        // Set specific Z depths
        // Set specific Z depths (Task 11: GEX must be on top of Vacuum)
        this.physicsLayer.getMesh().position.z = -0.02;
        this.wallLayer.getMesh().position.z = 0.0;
        this.bucketRadarLayer.getMesh().position.z = 0.005; // Primary Layer
        this.vacuumLayer.getMesh().position.z = 0.015;
        this.gexLayer.getMesh().position.z = 0.02; // Moved in front of vacuum (was 0.01)
    }

    setZoom(level: number): void {
        this.zoomLevel = level;
        this.updateCamera();
        this.updateVisualization();
    }

    centerView(): void {
        this.viewCenter = { x: 0, y: 0 };
        this.zoomLevel = 1.0;
        this.autoCenter = true; // Re-enable tracking
        this.isUserPanning = false;
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
            }
        }
    }

    private updateOverlay(currentSpot: number, midPrice: number): void {
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

        // Align to step
        const startPrice = Math.floor(minPrice / displayStep) * displayStep;

        // Draw Grid Labels
        for (let price = startPrice; price <= maxPrice; price += displayStep) {
            // World Y relative to Spot
            const y = price - currentSpot;

            // Normalized Screen Coordinates
            const normY = (y - this.camera.bottom) / viewHeight;
            const bottomPct = normY * 100;
            if (bottomPct < -5 || bottomPct > 105) continue;

            const el = document.createElement('div');
            el.className = 'price-label';
            el.textContent = price.toFixed(2);
            el.style.bottom = `${bottomPct}%`;
            el.style.position = 'absolute';
            el.style.transform = 'translateY(50%)';
            priceAxis.appendChild(el);
        }

        // Draw Current Price Label (Mid)
        const midY = midPrice - currentSpot;
        const midNormY = (midY - this.camera.bottom) / viewHeight;
        const midPct = midNormY * 100;

        if (midPct >= 0 && midPct <= 100) {
            const el = document.createElement('div');
            el.className = 'price-label current';
            el.textContent = midPrice.toFixed(2);
            el.style.bottom = `${midPct}%`;
            el.style.position = 'absolute';
            el.style.transform = 'translateY(50%)';
            // Add prominent styling via inline or ensure css covers .current
            el.style.backgroundColor = '#00ccff';
            el.style.color = '#000';
            el.style.padding = '2px 4px';
            el.style.borderRadius = '2px';
            el.style.zIndex = '10';
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

    private createPriceGrid(): void {
        this.clearGroup(this.gridGroup);

        const gridMat1 = new THREE.LineBasicMaterial({ color: 0x3a3a5e, transparent: true, opacity: 0.2 });
        const gridMat5 = new THREE.LineBasicMaterial({ color: 0x5a5a8e, transparent: true, opacity: 0.4 });

        // World Bounds (Y=0 is Spot)
        // 1 unit = $1.00 (4 ticks)
        const topY = this.camera.top;
        const bottomY = this.camera.bottom;

        // Convert to Ticks (Approximation: 1 unit = 4 ticks)
        // Y = (Price - Spot) * 1.0
        // PriceDelta = Y
        // Ticks = PriceDelta / 0.25 = Y * 4

        const points1: THREE.Vector3[] = [];
        const points5: THREE.Vector3[] = [];
        const left = this.camera.left;
        const right = this.camera.right;

        // Fix: Draw grid lines at ABSOLUTE price levels, adjusted for current spot

        const currentSpot = this.lastValidSpot;
        const absMinPrice = currentSpot + bottomY;
        const absMaxPrice = currentSpot + topY;

        const minAbsTick = Math.floor(absMinPrice / 0.25);
        const maxAbsTick = Math.ceil(absMaxPrice / 0.25);

        for (let absTick = minAbsTick; absTick <= maxAbsTick; absTick++) {
            // Task 19: Grid Lines
            // Every 20 ticks ($5) -> Strong
            // Every 4 ticks ($1) -> Weak
            // Use absTick to determine modulus

            const isStrike = (absTick % 20 === 0);
            const isPoint = (absTick % 4 === 0);

            if (!isPoint) continue; // Only draw integer points

            // Y relative to center (Spot)
            // Y = Price - Spot
            // Price = absTick * 0.25
            const y = (absTick * 0.25) - currentSpot;

            if (isStrike) {
                points5.push(new THREE.Vector3(left, y, -1));
                points5.push(new THREE.Vector3(right, y, -1));
            } else {
                points1.push(new THREE.Vector3(left, y, -1));
                points1.push(new THREE.Vector3(right, y, -1));
            }
        }

        if (points1.length > 0) {
            const lines = new THREE.LineSegments(
                new THREE.BufferGeometry().setFromPoints(points1),
                gridMat1
            );
            lines.position.z = 0.02;
            this.gridGroup.add(lines);
        }

        if (points5.length > 0) {
            const lines = new THREE.LineSegments(
                new THREE.BufferGeometry().setFromPoints(points5),
                gridMat5
            );
            lines.position.z = 0.02;
            this.gridGroup.add(lines);
        }

        // Draw Zero Line (Spot)
        const zeroPoints = [
            new THREE.Vector3(left, 0, 0),
            new THREE.Vector3(right, 0, 0)
        ];
        const zeroMat = new THREE.LineBasicMaterial({ color: 0x00ccff, transparent: true, opacity: 0.5 });
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
        // const gexData = this.state.getGexData();

        // Task 20: Canonical Spot Anchoring
        // We must use spot_ref_price_int (Tick Aligned) for the Grid/Camera anchor.
        // MidPrice is cosmetic only (drawn as line).

        const spotRefInt = this.state.getSpotRefInt();
        let anchorPrice = 6000.0;

        if (spotRefInt > 0n) {
            anchorPrice = Number(spotRefInt) * PRICE_SCALE;
        } else {
            // Fallback if no parsed int yet
            if (snapData.length > 0) anchorPrice = Number(snapData[snapData.length - 1].mid_price);
        }

        // Update robust spot
        if (anchorPrice > 100) {
            this.lastValidSpot = anchorPrice;
        }

        const currentSpot = this.lastValidSpot; // This is now SNAP-ALIGNED PRICE

        // Mid Price for Tracking
        let midPrice = currentSpot;
        // snapData is already declared above (line ~564)
        if (snapData.length > 0) {
            const rawMid = Number(snapData[snapData.length - 1].mid_price);
            if (!isNaN(rawMid) && rawMid > 100) {
                midPrice = rawMid;
            }
        } else {
            // Fallback to Gex?
            const gex = this.state.getGexData();
            if (gex.length > 0) {
                const best = gex[gex.length - 1];
                const rawGexSpot = Number(best.underlying_spot_ref || best.spot_ref_price_int / 250000000n * 100n / 100n);
                if (!isNaN(rawGexSpot) && rawGexSpot > 100) {
                    midPrice = rawGexSpot;
                }
            }
        }

        // Task: Camera Tracking
        // We want viewCenter.y to place midPrice in center.
        // Y = Price - currentSpot (Anchor)
        // targetY = midPrice - currentSpot

        if (this.autoCenter && !this.isUserPanning) {
            const targetY = (midPrice - currentSpot) * this.UNIT_PER_DOLLAR;
            // Smooth lerp?
            // this.viewCenter.y += (targetY - this.viewCenter.y) * 0.1;
            // For crispness, instant for now.
            this.viewCenter.y = targetY;
            this.updateCamera();
        }

        // Update Shaders with this Anchor
        // Convert to Tick Index
        // FIX: uSpotRef represents the "Center of View" tick index for rectification.
        // We must pass the Real Spot (midPrice), not the Anchor, so the shader
        // calculates the correct relative offset from the Anchor-aligned texture.
        const currentSpotTick = midPrice / TICK_SIZE;

        this.wallLayer.setSpotRef(currentSpotTick);
        this.vacuumLayer.setSpotRef(currentSpotTick);
        this.physicsLayer.setSpotRef(currentSpotTick);
        this.gexLayer.setSpotRef(currentSpotTick);
        this.bucketRadarLayer.setSpotRef(currentSpotTick);

        this.createPriceGrid();
        this.createPriceLine(allWindows, spotByWindow, currentSpot);

        // Passed computed midPrice to overlay (calculated above)
        this.updateOverlay(currentSpot, midPrice);
        this.updateDebugDiagnostics(spotRefInt, allWindows);
    }

    private updateDebugDiagnostics(spotRef: bigint, windows: bigint[]): void {
        let debugEl = document.getElementById('debug-overlay');
        if (!debugEl) {
            debugEl = document.createElement('div');
            debugEl.id = 'debug-overlay';
            debugEl.style.position = 'absolute';
            debugEl.style.top = '10px';
            debugEl.style.left = '10px';
            debugEl.style.color = '#00ff00';
            debugEl.style.fontFamily = 'monospace';
            debugEl.style.fontSize = '12px';
            debugEl.style.backgroundColor = 'rgba(0,0,0,0.7)';
            debugEl.style.padding = '5px';
            debugEl.style.pointerEvents = 'none';
            document.body.appendChild(debugEl);
        }

        const head = this.physicsLayer.getHead();
        const latestTs = windows.length > 0 ? windows[windows.length - 1] : 0n;
        const dateStr = new Date(Number(latestTs) / 1e6).toISOString().split('T')[1].replace('Z', '');

        debugEl.innerHTML = `
            <div>SpotRefTick: ${spotRef / 250000000n}</div>
            <div>Head Col: ${head}</div>
            <div>Latest TS: ${latestTs} (${dateStr})</div>
            <div>Physics: ${this.physicsLayer.getWidth()}x${801}</div>
        `;
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
