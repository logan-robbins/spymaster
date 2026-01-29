import * as THREE from 'three';
import { type ForecastRow } from './ws-client';

export class ForecastOverlay {
    private scene: THREE.Scene;
    private pathMesh: THREE.Line;
    private coneMesh: THREE.Mesh;
    
    private pathMaterial: THREE.LineBasicMaterial;
    private pathGeometry: THREE.BufferGeometry;
    private coneGeometry: THREE.ConeGeometry;
    private coneMaterial: THREE.MeshBasicMaterial;

    // Diagnostic data
    private runScoreUp: number = 0;
    private runScoreDown: number = 0;
    private dUp: number = 0;
    private dDown: number = 0;
    private confidence: number = 0;

    // Track camera center for positioning
    private currentCameraY: number = 0;

    constructor(scene: THREE.Scene) {
        this.scene = scene;

        // Path line - thicker, more visible
        this.pathMaterial = new THREE.LineBasicMaterial({ 
            color: 0xffcc00,
            linewidth: 3,
        });
        this.pathGeometry = new THREE.BufferGeometry();
        
        // Pre-allocate buffer for full trajectory (31 points: now + 30 horizons)
        const positions = new Float32Array(31 * 3);
        this.pathGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        this.pathMesh = new THREE.Line(this.pathGeometry, this.pathMaterial);
        this.pathMesh.frustumCulled = false;
        this.pathMesh.renderOrder = 10;
        this.scene.add(this.pathMesh);

        // Direction cone at the end of the forecast
        this.coneGeometry = new THREE.ConeGeometry(2, 5, 8);
        this.coneMaterial = new THREE.MeshBasicMaterial({ 
            color: 0xffcc00,
            transparent: true,
            opacity: 0.85,
        });
        this.coneMesh = new THREE.Mesh(this.coneGeometry, this.coneMaterial);
        this.coneMesh.frustumCulled = false;
        this.coneMesh.renderOrder = 11;
        this.scene.add(this.coneMesh);
    }

    // Set the camera Y center for proper positioning
    setCameraCenter(cameraYCenter: number): void {
        this.currentCameraY = cameraYCenter;
        // Position the mesh at camera center so relative Y coordinates work
        this.pathMesh.position.y = cameraYCenter;
        this.coneMesh.position.y = cameraYCenter;
    }

    update(rows: ForecastRow[], currentSpotTick: number) {
        if (rows.length === 0) return;

        // Extract diagnostic row (horizon_s = 0)
        const diagnosticRow = rows.find(r => r.horizon_s === 0);
        if (diagnosticRow) {
            this.runScoreUp = diagnosticRow.run_score_up ?? 0;
            this.runScoreDown = diagnosticRow.run_score_down ?? 0;
            this.dUp = diagnosticRow.d_up ?? 0;
            this.dDown = diagnosticRow.d_down ?? 0;
        }

        // Sort forecast rows by horizon (excluding diagnostic)
        const forecastRows = rows
            .filter(r => r.horizon_s > 0)
            .sort((a, b) => a.horizon_s - b.horizon_s);

        if (forecastRows.length === 0) return;

        const positions = this.pathGeometry.attributes.position.array as Float32Array;

        // Calculate tick offset from camera center to current spot
        // This makes the line start where the spot currently is
        const spotOffset = currentSpotTick - this.currentCameraY;

        // Point 0: Current position (now) - at X=0 (right edge of data), Y = spot offset
        positions[0] = 0;
        positions[1] = spotOffset;  // RELATIVE to camera center
        positions[2] = 5;

        // Fill in forecast trajectory
        let maxHorizon = 0;
        let finalDelta = 0;
        let maxConfidence = 0;

        for (let i = 0; i < Math.min(forecastRows.length, 30); i++) {
            const row = forecastRows[i];
            const h = row.horizon_s;
            // Raw delta can be huge due to backend bug - clamp to reasonable range
            const rawDelta = Number(row.predicted_tick_delta);
            const delta = Math.max(-100, Math.min(100, rawDelta));
            const conf = row.confidence ?? 0;

            const idx = (i + 1) * 3;
            positions[idx] = h;  // X = seconds into future (projects into 30% margin)
            positions[idx + 1] = spotOffset + delta;  // Y = RELATIVE (offset + predicted change)
            positions[idx + 2] = 5;

            if (h > maxHorizon) {
                maxHorizon = h;
                finalDelta = Math.max(-100, Math.min(100, rawDelta));  // Clamp for display
            }
            if (conf > maxConfidence) {
                maxConfidence = conf;
            }
        }

        this.confidence = maxConfidence;

        // Update geometry draw range
        this.pathGeometry.setDrawRange(0, forecastRows.length + 1);
        this.pathGeometry.attributes.position.needsUpdate = true;

        // Color the path based on direction and confidence
        let pathColor: THREE.Color;
        
        if (maxConfidence > 0.3) {
            if (finalDelta > 0) {
                // Upward with confidence - bright amber/orange
                pathColor = new THREE.Color(1.0, 0.65 + maxConfidence * 0.25, 0.15);
            } else if (finalDelta < 0) {
                // Downward with confidence - bright teal/cyan
                pathColor = new THREE.Color(0.15, 0.65 + maxConfidence * 0.25, 0.9);
            } else {
                // Neutral
                pathColor = new THREE.Color(0.6, 0.6, 0.55);
            }
        } else {
            // Low confidence - dim yellow/gray
            pathColor = new THREE.Color(0.45, 0.45, 0.38);
        }

        this.pathMaterial.color = pathColor;
        this.coneMaterial.color = pathColor;
        this.coneMaterial.opacity = 0.4 + maxConfidence * 0.5;

        // Position the direction cone at the end of the forecast path
        // Cone position is in LOCAL coordinates (relative to mesh position)
        const finalY = spotOffset + finalDelta;
        
        // Remove prior position.y offset since we're using local coords
        this.coneMesh.position.x = maxHorizon + 3;
        this.coneMesh.position.z = 5;
        // Note: coneMesh.position.y is set by setCameraCenter, we just adjust locally
        this.coneMesh.position.y = this.currentCameraY + finalY;
        
        // Rotate cone to point in direction of movement
        if (finalDelta > 0) {
            this.coneMesh.rotation.z = 0; // Point up
        } else if (finalDelta < 0) {
            this.coneMesh.rotation.z = Math.PI; // Point down
        } else {
            this.coneMesh.rotation.z = -Math.PI / 2; // Point right (neutral)
        }
    }

    // Getters for HUD display
    getRunScoreUp(): number { return this.runScoreUp; }
    getRunScoreDown(): number { return this.runScoreDown; }
    getDUp(): number { return this.dUp; }
    getDDown(): number { return this.dDown; }
    getConfidence(): number { return this.confidence; }
}
