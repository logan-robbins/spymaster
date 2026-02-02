/**
 * 2D Particle Wave Function Tester
 * 
 * Simulates a particle wave moving in the x+ direction with configurable physics parameters.
 * The wave is stored as an nxnxm tensor where m represents different physics inputs:
 * - viscosity: medium viscosity affecting wave propagation
 * - decay: amplitude decay over distance
 * - distance_from_object: distance to nearest obstacle in y direction
 * - permeability: medium permeability
 * - object_density: density of obstacles affecting wave reflection/transmission
 */

// ============================================
// API Interface - Tunable Parameters
// ============================================

interface WaveParameters {
  // Wave properties
  wavelength: number;      // λ: spatial period (units)
  frequency: number;       // f: temporal frequency (Hz)
  speed: number;          // v: wave propagation speed (units/s)
  amplitude: number;      // A: wave amplitude
  
  // Medium properties (physics tensor channels)
  viscosity: number;      // μ: medium viscosity (0-1)
  decay: number;          // α: amplitude decay rate
  permeability: number;   // ε: medium permeability
  
  // Object properties
  objectDensity: number;  // ρ: obstacle density (0-2)
  objectX: number;        // Object center X position
  objectY: number;        // Object center Y position
  objectWidth: number;    // Object width
  objectHeight: number;   // Object height
}

interface PhysicsTensor {
  // Shape: [width, height, channels]
  // channels: [viscosity, decay, distance_from_object, permeability, object_density]
  data: Float32Array;
  width: number;
  height: number;
  channels: number;
}

// ============================================
// Default Configuration
// ============================================

const DEFAULT_PARAMS: WaveParameters = {
  wavelength: 10.0,
  frequency: 1.0,
  speed: 10.0,
  amplitude: 1.0,
  viscosity: 0.0,
  decay: 0.0,
  permeability: 1.0,
  objectDensity: 0.5,
  objectX: 10.0,
  objectY: 5.0,
  objectWidth: 2.0,
  objectHeight: 3.0,
};

// Grid dimensions (20 units long, 10 units high)
const GRID_WIDTH_UNITS = 20;
const GRID_HEIGHT_UNITS = 10;
const PIXELS_PER_UNIT = 40; // Resolution
const GRID_WIDTH = GRID_WIDTH_UNITS * PIXELS_PER_UNIT;
const GRID_HEIGHT = GRID_HEIGHT_UNITS * PIXELS_PER_UNIT;

// ============================================
// Physics Tensor Generator
// ============================================

class PhysicsTensorGenerator {
  private width: number;
  private height: number;
  private channels: number;
  
  constructor(width: number, height: number, channels: number = 5) {
    this.width = width;
    this.height = height;
    this.channels = channels;
  }
  
  /**
   * Generate physics tensor with for loops as specified
   * Returns tensor of shape [width, height, channels]
   */
  generate(params: WaveParameters): PhysicsTensor {
    const size = this.width * this.height * this.channels;
    const data = new Float32Array(size);
    
    // Channel indices
    const CH_VISCOSITY = 0;
    const CH_DECAY = 1;
    const CH_DIST_TO_OBJ = 2;
    const CH_PERMEABILITY = 3;
    const CH_OBJ_DENSITY = 4;
    
    // Generate tensor using for loops
    for (let x = 0; x < this.width; x++) {
      for (let y = 0; y < this.height; y++) {
        const idx = (x * this.height + y) * this.channels;
        
        // Convert pixel coordinates to world coordinates
        const worldX = x / PIXELS_PER_UNIT;
        const worldY = y / PIXELS_PER_UNIT;
        
        // Channel 0: Viscosity (uniform across medium, higher near objects)
        const distToObj = this.distanceToObject(worldX, worldY, params);
        const viscosity = params.viscosity * (1 + 0.5 * Math.exp(-distToObj * 0.5));
        data[idx + CH_VISCOSITY] = Math.min(viscosity, 1.0);
        
        // Channel 1: Decay rate (spatially varying)
        const decay = params.decay * (1 + 0.3 * params.objectDensity * Math.exp(-distToObj * 0.3));
        data[idx + CH_DECAY] = decay;
        
        // Channel 2: Distance to nearest object in Y direction
        data[idx + CH_DIST_TO_OBJ] = distToObj;
        
        // Channel 3: Permeability (affected by object density)
        const inObject = this.isInsideObject(worldX, worldY, params);
        const permeability = inObject 
          ? params.permeability / (1 + params.objectDensity)
          : params.permeability;
        data[idx + CH_PERMEABILITY] = permeability;
        
        // Channel 4: Object density at this point
        data[idx + CH_OBJ_DENSITY] = inObject ? params.objectDensity : 0.0;
      }
    }
    
    return {
      data,
      width: this.width,
      height: this.height,
      channels: this.channels,
    };
  }
  
  private distanceToObject(x: number, y: number, params: WaveParameters): number {
    const halfWidth = params.objectWidth / 2;
    const halfHeight = params.objectHeight / 2;
    
    // Distance to rectangle in Y direction only (as specified)
    const dy = Math.max(0, Math.abs(y - params.objectY) - halfHeight);
    
    // Also consider X distance if within object's x-range
    const inXRange = x >= params.objectX - halfWidth && x <= params.objectX + halfWidth;
    
    if (inXRange) {
      return dy;
    } else {
      // Distance to nearest corner in x direction
      const dx = Math.min(
        Math.abs(x - (params.objectX - halfWidth)),
        Math.abs(x - (params.objectX + halfWidth))
      );
      return Math.sqrt(dx * dx + dy * dy);
    }
  }
  
  private isInsideObject(x: number, y: number, params: WaveParameters): boolean {
    const halfWidth = params.objectWidth / 2;
    const halfHeight = params.objectHeight / 2;
    
    return x >= params.objectX - halfWidth && 
           x <= params.objectX + halfWidth &&
           y >= params.objectY - halfHeight && 
           y <= params.objectY + halfHeight;
  }
}

// ============================================
// Wave Generator
// ============================================

class WaveGenerator {
  private width: number;
  private height: number;
  
  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
  }
  
  /**
   * Generate 2D wave array using sine function
   * Wave moves in x+ direction
   */
  generate(time: number, params: WaveParameters, tensor: PhysicsTensor): Float32Array {
    const wave = new Float32Array(this.width * this.height);
    
    const k = 2 * Math.PI / params.wavelength; // Wave number
    const omega = 2 * Math.PI * params.frequency; // Angular frequency
    
    // Generate wave using for loops as specified
    for (let x = 0; x < this.width; x++) {
      for (let y = 0; y < this.height; y++) {
        const idx = x * this.height + y;
        const tensorIdx = idx * tensor.channels;
        
        // World coordinates
        const worldX = x / PIXELS_PER_UNIT;
        // worldY available for future use in y-dependent physics
        // const worldY = y / PIXELS_PER_UNIT;
        
        // Get physics properties from tensor
        const viscosity = tensor.data[tensorIdx + 0];
        const decay = tensor.data[tensorIdx + 1];
        const distToObj = tensor.data[tensorIdx + 2];
        const permeability = tensor.data[tensorIdx + 3];
        const objDensity = tensor.data[tensorIdx + 4];
        
        // Base wave: A * sin(kx - ωt + φ)
        // Wave moving in +x direction: phase = kx - ωt
        const phase = k * worldX - omega * time;
        let amplitude = params.amplitude;
        
        // Apply decay based on distance traveled
        amplitude *= Math.exp(-decay * worldX);
        
        // Apply viscosity damping
        amplitude *= (1 - viscosity * 0.5);
        
        // Apply permeability effect
        amplitude *= Math.sqrt(permeability);
        
        // Object interaction: reflection/transmission
        if (objDensity > 0) {
          // Inside object: attenuated wave
          amplitude *= Math.exp(-objDensity * 2);
        } else {
          // Near object: interference pattern
          const interference = 0.2 * objDensity * Math.exp(-distToObj) * Math.cos(phase * 2);
          amplitude += interference;
        }
        
        // Sine wave generation
        wave[idx] = amplitude * Math.sin(phase);
      }
    }
    
    return wave;
  }
}

// ============================================
// Renderer
// ============================================

class WaveRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private imageData: ImageData;
  
  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.canvas.width = GRID_WIDTH;
    this.canvas.height = GRID_HEIGHT;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    this.ctx = ctx;
    
    this.imageData = ctx.createImageData(GRID_WIDTH, GRID_HEIGHT);
  }
  
  render(wave: Float32Array, tensor: PhysicsTensor, params: WaveParameters, showTensor: boolean): void {
    const data = this.imageData.data;
    
    for (let x = 0; x < GRID_WIDTH; x++) {
      for (let y = 0; y < GRID_HEIGHT; y++) {
        const idx = x * GRID_HEIGHT + y;
        const pixelIdx = (y * GRID_WIDTH + x) * 4;
        
        if (showTensor) {
          // Visualize physics tensor channels
          const tensorIdx = idx * tensor.channels;
          const viscosity = tensor.data[tensorIdx + 0];
          // decay = tensor.data[tensorIdx + 1] (used in visualization logic)
          const distToObj = tensor.data[tensorIdx + 2];
          const permeability = tensor.data[tensorIdx + 3];
          const objDensity = tensor.data[tensorIdx + 4];
          
          // Color mapping for tensor visualization
          data[pixelIdx] = Math.floor(255 * (viscosity + objDensity * 0.5));     // R
          data[pixelIdx + 1] = Math.floor(255 * (permeability * 0.5));            // G
          data[pixelIdx + 2] = Math.floor(255 * (1 - distToObj / 10));            // B
          data[pixelIdx + 3] = 255;
        } else {
          // Visualize wave amplitude
          const value = wave[idx];
          
          // Normalize to 0-255 with center at 128
          const intensity = Math.floor(128 + value * 127);
          const clamped = Math.max(0, Math.min(255, intensity));
          
          // Color based on amplitude: cyan for positive, magenta for negative
          if (value > 0) {
            data[pixelIdx] = 0;           // R
            data[pixelIdx + 1] = clamped; // G
            data[pixelIdx + 2] = 255;     // B
          } else {
            data[pixelIdx] = 255;         // R
            data[pixelIdx + 1] = 0;       // G
            data[pixelIdx + 2] = clamped; // B
          }
          data[pixelIdx + 3] = 255;
        }
      }
    }
    
    this.ctx.putImageData(this.imageData, 0, 0);
    
    // Draw object outline
    this.drawObject(params);
    
    // Draw grid overlay
    this.drawGrid();
  }
  
  private drawObject(params: WaveParameters): void {
    const halfWidth = params.objectWidth / 2;
    const halfHeight = params.objectHeight / 2;
    
    const x = (params.objectX - halfWidth) * PIXELS_PER_UNIT;
    const y = (params.objectY - halfHeight) * PIXELS_PER_UNIT;
    const w = params.objectWidth * PIXELS_PER_UNIT;
    const h = params.objectHeight * PIXELS_PER_UNIT;
    
    this.ctx.strokeStyle = '#ff5555';
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(x, y, w, h);
    
    // Fill with semi-transparent red based on density
    this.ctx.fillStyle = `rgba(255, 85, 85, ${params.objectDensity * 0.3})`;
    this.ctx.fillRect(x, y, w, h);
    
    // Label
    this.ctx.fillStyle = '#ff5555';
    this.ctx.font = '12px monospace';
    this.ctx.fillText(`ρ=${params.objectDensity.toFixed(2)}`, x + 4, y - 4);
  }
  
  private drawGrid(): void {
    this.ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)';
    this.ctx.lineWidth = 1;
    
    // Vertical grid lines every 5 units
    for (let x = 0; x <= GRID_WIDTH_UNITS; x += 5) {
      const px = x * PIXELS_PER_UNIT;
      this.ctx.beginPath();
      this.ctx.moveTo(px, 0);
      this.ctx.lineTo(px, GRID_HEIGHT);
      this.ctx.stroke();
      
      // Label
      this.ctx.fillStyle = '#666';
      this.ctx.font = '10px monospace';
      this.ctx.fillText(`${x}`, px + 2, GRID_HEIGHT - 4);
    }
    
    // Horizontal grid lines every 2 units
    for (let y = 0; y <= GRID_HEIGHT_UNITS; y += 2) {
      const py = y * PIXELS_PER_UNIT;
      this.ctx.beginPath();
      this.ctx.moveTo(0, py);
      this.ctx.lineTo(GRID_WIDTH, py);
      this.ctx.stroke();
      
      // Label
      this.ctx.fillStyle = '#666';
      this.ctx.font = '10px monospace';
      this.ctx.fillText(`${y}`, 4, py - 2);
    }
  }
  
  resize(): void {
    // Canvas maintains fixed resolution, CSS handles display size
  }
}

// ============================================
// UI Controller
// ============================================

class UIController {
  private params: WaveParameters;
  private onChange: (params: WaveParameters) => void;
  private onToggleTensor: (show: boolean) => void;
  private onPause: (paused: boolean) => void;
  
  constructor(
    params: WaveParameters,
    onChange: (params: WaveParameters) => void,
    onToggleTensor: (show: boolean) => void,
    onPause: (paused: boolean) => void
  ) {
    this.params = { ...params };
    this.onChange = onChange;
    this.onToggleTensor = onToggleTensor;
    this.onPause = onPause;
    
    this.bindControls();
  }
  
  private bindControls(): void {
    // Wave parameters
    this.bindSlider('wavelength', 'wavelength-value', (v) => this.params.wavelength = v, ' units');
    this.bindSlider('frequency', 'frequency-value', (v) => this.params.frequency = v, ' Hz');
    this.bindSlider('speed', 'speed-value', (v) => this.params.speed = v, ' units/s');
    this.bindSlider('amplitude', 'amplitude-value', (v) => this.params.amplitude = v, '');
    
    // Medium properties
    this.bindSlider('viscosity', 'viscosity-value', (v) => this.params.viscosity = v, '');
    this.bindSlider('decay', 'decay-value', (v) => this.params.decay = v, '');
    this.bindSlider('permeability', 'permeability-value', (v) => this.params.permeability = v, '');
    
    // Object properties
    this.bindSlider('density', 'density-value', (v) => this.params.objectDensity = v, '');
    this.bindSlider('obj-y', 'obj-y-value', (v) => this.params.objectY = v, ' units');
    this.bindSlider('obj-x', 'obj-x-value', (v) => this.params.objectX = v, ' units');
    this.bindSlider('obj-width', 'obj-width-value', (v) => this.params.objectWidth = v, ' units');
    
    // Toggle buttons
    const tensorBtn = document.getElementById('show-tensor') as HTMLButtonElement;
    let showTensor = false;
    tensorBtn.addEventListener('click', () => {
      showTensor = !showTensor;
      tensorBtn.classList.toggle('active', showTensor);
      tensorBtn.textContent = showTensor ? 'Hide Physics Tensor' : 'Show Physics Tensor';
      this.onToggleTensor(showTensor);
    });
    
    const pauseBtn = document.getElementById('pause-btn') as HTMLButtonElement;
    let paused = false;
    pauseBtn.addEventListener('click', () => {
      paused = !paused;
      pauseBtn.classList.toggle('active', paused);
      pauseBtn.textContent = paused ? 'Resume' : 'Pause';
      this.onPause(paused);
    });
  }
  
  private bindSlider(
    id: string, 
    valueId: string, 
    setter: (v: number) => void,
    suffix: string
  ): void {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valueDisplay = document.getElementById(valueId) as HTMLDivElement;
    
    slider.addEventListener('input', () => {
      const value = parseFloat(slider.value);
      setter(value);
      valueDisplay.textContent = value.toFixed(2) + suffix;
      this.onChange(this.params);
    });
  }
}

// ============================================
// Main Application
// ============================================

class ParticleWaveApp {
  private params: WaveParameters;
  private tensorGenerator: PhysicsTensorGenerator;
  private waveGenerator: WaveGenerator;
  private renderer: WaveRenderer;
  
  private tensor: PhysicsTensor;
  private wave: Float32Array;
  private time: number = 0;
  private paused: boolean = false;
  private showTensor: boolean = false;
  
  private lastFrameTime: number = 0;
  private animationId: number | null = null;
  
  constructor() {
    // Initialize parameters
    this.params = { ...DEFAULT_PARAMS };
    
    // Initialize generators
    this.tensorGenerator = new PhysicsTensorGenerator(GRID_WIDTH, GRID_HEIGHT);
    this.waveGenerator = new WaveGenerator(GRID_WIDTH, GRID_HEIGHT);
    
    // Initialize renderer
    const canvas = document.getElementById('wave-canvas') as HTMLCanvasElement;
    this.renderer = new WaveRenderer(canvas);
    
    // Generate initial tensor and wave
    this.tensor = this.tensorGenerator.generate(this.params);
    this.wave = this.waveGenerator.generate(0, this.params, this.tensor);
    
    // Initialize UI (stored to prevent garbage collection)
    const _ui = new UIController(
      this.params,
      (p) => this.onParamsChange(p),
      (show) => this.onToggleTensor(show),
      (paused) => this.onPause(paused)
    );
    // Keep reference to UI controller
    (window as unknown as { _particleWaveUI: UIController })._particleWaveUI = _ui;
    
    // Start animation loop
    this.start();
  }
  
  private onParamsChange(params: WaveParameters): void {
    this.params = params;
    // Regenerate tensor when parameters change
    this.tensor = this.tensorGenerator.generate(this.params);
  }
  
  private onToggleTensor(show: boolean): void {
    this.showTensor = show;
  }
  
  private onPause(paused: boolean): void {
    this.paused = paused;
  }
  
  private start(): void {
    this.lastFrameTime = performance.now();
    this.animate();
  }
  
  private animate = (): void => {
    const now = performance.now();
    const dt = (now - this.lastFrameTime) / 1000;
    this.lastFrameTime = now;
    
    if (!this.paused) {
      // Update time
      this.time += dt;
      
      // Regenerate wave
      this.wave = this.waveGenerator.generate(this.time, this.params, this.tensor);
      
      // Update stats
      this.updateStats();
    }
    
    // Render
    this.renderer.render(this.wave, this.tensor, this.params, this.showTensor);
    
    this.animationId = requestAnimationFrame(this.animate);
  };
  
  private updateStats(): void {
    // Calculate max amplitude
    let maxAmp = 0;
    let energy = 0;
    for (let i = 0; i < this.wave.length; i++) {
      const v = Math.abs(this.wave[i]);
      maxAmp = Math.max(maxAmp, v);
      energy += v * v;
    }
    energy = Math.sqrt(energy / this.wave.length);
    
    // Update display
    const timeDisplay = document.getElementById('time-display');
    const ampDisplay = document.getElementById('amplitude-display');
    const energyDisplay = document.getElementById('energy-display');
    
    if (timeDisplay) timeDisplay.textContent = `t: ${this.time.toFixed(2)}s`;
    if (ampDisplay) ampDisplay.textContent = `Max Amp: ${maxAmp.toFixed(3)}`;
    if (energyDisplay) energyDisplay.textContent = `Energy: ${energy.toFixed(3)}`;
  }
  
  destroy(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }
  }
}

// ============================================
// Export API for external use
// ============================================

export type {
  WaveParameters,
  PhysicsTensor,
};

export {
  PhysicsTensorGenerator,
  WaveGenerator,
  WaveRenderer,
  DEFAULT_PARAMS,
  GRID_WIDTH,
  GRID_HEIGHT,
  GRID_WIDTH_UNITS,
  GRID_HEIGHT_UNITS,
};

// ============================================
// Initialize on DOM ready
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  new ParticleWaveApp();
});
