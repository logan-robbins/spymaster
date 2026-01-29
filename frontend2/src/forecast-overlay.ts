import * as THREE from 'three';
import { type ForecastRow } from './ws-client';

export class ForecastOverlay {
    private scene: THREE.Scene;
    private pathMesh: THREE.Line;

    private material: THREE.LineBasicMaterial;
    private geometry: THREE.BufferGeometry;

    constructor(scene: THREE.Scene) {
        this.scene = scene;

        this.material = new THREE.LineBasicMaterial({ color: 0xffff00 }); // Yellow path
        this.geometry = new THREE.BufferGeometry();
        // Pre-allocate buffer for trajectory
        // 2 points for now
        const positions = new Float32Array(2 * 3);
        this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        this.pathMesh = new THREE.Line(this.geometry, this.material);
        this.pathMesh.frustumCulled = false;
        this.scene.add(this.pathMesh);
    }

    update(rows: ForecastRow[], currentSpotTick: number) {
        // Find row with max horizon, or specific horizon
        // Assuming one row per frame for now (horizon=30)
        if (rows.length === 0) return;

        const forecast = rows[0];
        const delta = Number(forecast.predicted_tick_delta);
        const horizon = forecast.horizon_s;

        // Points:
        // P0: (0, currentSpotTick)
        // P1: (horizon, currentSpotTick + delta)

        // Coordinates in world space:
        // X = seconds from now (positive)
        // Y = tick index

        // However, our main scene has velocity mesh at X < 0.
        // Forecast overlays at X > 0.

        const positions = this.geometry.attributes.position.array as Float32Array;

        // P0
        positions[0] = 0;
        positions[1] = currentSpotTick;
        positions[2] = 10; // Z-index above grid

        // P1
        positions[3] = horizon;
        positions[4] = currentSpotTick + delta;
        positions[5] = 10;

        this.geometry.attributes.position.needsUpdate = true;

        // Update color based on RunScore?
        // RunScore is "pressure".
        // If delta > 0, maybe Green? If < 0 Red?
        // Default yellow for "path".
    }
}
