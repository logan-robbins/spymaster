import * as THREE from 'three';

const TICK_SIZE = 0.25; // $0.25 per tick

// Tick spacing levels (in ticks)
const SPACING_5_DOLLAR = 20;   // $5.00 = 20 ticks
const SPACING_1_DOLLAR = 4;    // $1.00 = 4 ticks  
const SPACING_25_CENT = 1;     // $0.25 = 1 tick

/**
 * Price axis labels and grid lines with dynamic tick spacing based on zoom.
 * Shows $5 increments at normal zoom, $1 when zoomed in, $0.25 when very zoomed.
 */
export class PriceAxis {
  private container: HTMLElement;
  private labels: HTMLElement[] = [];
  private maxLabels: number = 100;

  // Grid lines rendered in Three.js
  private gridGroup: THREE.Group;
  private gridMaterial5: THREE.LineBasicMaterial;
  private gridMaterial1: THREE.LineBasicMaterial;
  private gridMaterial025: THREE.LineBasicMaterial;

  constructor(containerId: string, _tickSpacing: number = 20, _visibleTicks: number = 100) {
    this.container = document.getElementById(containerId)!;
    this.createLabels();

    // Create grid line materials
    this.gridMaterial5 = new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.6 });
    this.gridMaterial1 = new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.4 });
    this.gridMaterial025 = new THREE.LineBasicMaterial({ color: 0x1a1a1a, transparent: true, opacity: 0.25 });

    this.gridGroup = new THREE.Group();
  }

  getGridGroup(): THREE.Group {
    return this.gridGroup;
  }

  private createLabels(): void {
    for (let i = 0; i < this.maxLabels; i++) {
      const label = document.createElement('div');
      label.className = 'price-label';
      label.style.cssText = `
        position: absolute;
        right: 8px;
        color: #888;
        font-family: monospace;
        font-size: 11px;
        pointer-events: none;
        transform: translateY(-50%);
        text-align: right;
        min-width: 60px;
      `;
      this.container.appendChild(label);
      this.labels.push(label);
    }
  }

  /**
   * Determine tick spacing based on visible ticks (zoom level)
   */
  private getTickSpacing(viewHeight: number): { major: number; minor: number | null; micro: number | null } {
    // viewHeight is total ticks visible
    if (viewHeight > 80) {
      // Zoomed out: show only $5 increments
      return { major: SPACING_5_DOLLAR, minor: null, micro: null };
    } else if (viewHeight > 30) {
      // Medium zoom: show $5 major, $1 minor
      return { major: SPACING_5_DOLLAR, minor: SPACING_1_DOLLAR, micro: null };
    } else {
      // Zoomed in: show $5 major, $1 minor, $0.25 micro
      return { major: SPACING_5_DOLLAR, minor: SPACING_1_DOLLAR, micro: SPACING_25_CENT };
    }
  }

  /**
   * Update label positions and grid lines based on camera
   */
  update(centerTickIndex: number, camera: THREE.OrthographicCamera): void {
    const containerHeight = this.container.clientHeight;
    const viewHeight = camera.top - camera.bottom;
    const spacing = this.getTickSpacing(viewHeight);

    // Calculate visible tick range
    const minTick = Math.floor(centerTickIndex + camera.bottom);
    const maxTick = Math.ceil(centerTickIndex + camera.top);

    // Clear old grid lines
    while (this.gridGroup.children.length > 0) {
      const child = this.gridGroup.children[0];
      this.gridGroup.remove(child);
      if (child instanceof THREE.Line) {
        child.geometry.dispose();
      }
    }

    let labelIdx = 0;

    // Render micro ticks ($0.25) if visible
    if (spacing.micro !== null) {
      for (let tick = Math.floor(minTick); tick <= maxTick; tick += spacing.micro) {
        // Skip if this is a minor or major tick
        if (tick % SPACING_1_DOLLAR === 0) continue;
        this.addGridLine(tick, camera, this.gridMaterial025);
      }
    }

    // Render minor ticks ($1) if visible
    if (spacing.minor !== null) {
      for (let tick = Math.floor(minTick / spacing.minor) * spacing.minor; tick <= maxTick; tick += spacing.minor) {
        // Skip if this is a major tick
        if (tick % SPACING_5_DOLLAR === 0) continue;
        this.addGridLine(tick, camera, this.gridMaterial1);
        
        // Add label for $1 ticks when zoomed in enough
        if (viewHeight <= 50 && labelIdx < this.labels.length) {
          const priceValue = tick * TICK_SIZE;
          const screenY = this.tickToScreenY(tick, centerTickIndex, camera, containerHeight);
          
          if (screenY >= -20 && screenY <= containerHeight + 20) {
            const label = this.labels[labelIdx];
            label.style.display = 'block';
            label.style.top = `${screenY}px`;
            label.style.color = '#555';
            label.style.fontSize = '9px';
            label.textContent = `$${priceValue.toFixed(2)}`;
            labelIdx++;
          }
        }
      }
    }

    // Render major ticks ($5) - always visible
    for (let tick = Math.floor(minTick / SPACING_5_DOLLAR) * SPACING_5_DOLLAR; tick <= maxTick; tick += SPACING_5_DOLLAR) {
      this.addGridLine(tick, camera, this.gridMaterial5);
      
      if (labelIdx < this.labels.length) {
        const priceValue = tick * TICK_SIZE;
        const screenY = this.tickToScreenY(tick, centerTickIndex, camera, containerHeight);
        
        if (screenY >= -20 && screenY <= containerHeight + 20) {
          const label = this.labels[labelIdx];
          label.style.display = 'block';
          label.style.top = `${screenY}px`;
          label.style.color = '#888';
          label.style.fontSize = '11px';
          label.textContent = `$${priceValue.toFixed(2)}`;
          labelIdx++;
        }
      }
    }

    // Hide remaining labels
    for (; labelIdx < this.labels.length; labelIdx++) {
      this.labels[labelIdx].style.display = 'none';
    }
  }

  private tickToScreenY(tick: number, centerTick: number, camera: THREE.OrthographicCamera, containerHeight: number): number {
    const viewHeight = camera.top - camera.bottom;
    const tickOffset = tick - centerTick;
    const normalizedY = (tickOffset - camera.bottom) / viewHeight;
    return containerHeight * (1 - normalizedY);
  }

  private addGridLine(tick: number, camera: THREE.OrthographicCamera, material: THREE.LineBasicMaterial): void {
    const points = [
      new THREE.Vector3(camera.left - 100, tick, -0.1),
      new THREE.Vector3(camera.right + 100, tick, -0.1),
    ];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(geometry, material);
    this.gridGroup.add(line);
  }
}
