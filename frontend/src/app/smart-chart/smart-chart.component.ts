import { Component, ElementRef, OnInit, AfterViewInit, ViewChild, effect, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataStreamService } from '../data-stream.service';
import { LevelDerivedService, DerivedLevel } from '../level-derived.service';
import { FlowAnalyticsService } from '../flow-analytics.service';
import { ViewportSelectionService } from '../viewport-selection.service';
import { ChartSettingsService, ChartTimeframe } from '../chart-settings.service';

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ChartCandle extends Candle {
  x: number;
  yOpen: number;
  yHigh: number;
  yLow: number;
  yClose: number;
}

@Component({
    selector: 'app-smart-chart',
    standalone: true,
    imports: [CommonModule],
    template: `
    <div class="chart-wrapper">
      <div class="chart-header">
        <div class="chart-title">Price Chart</div>
        <div class="scale-control">
          <span class="scale-label">Scale</span>
          <input 
            type="range" 
            min="0.5" 
            max="2.0" 
            step="0.1" 
            [value]="chartScale()" 
            (input)="onScaleChange($any($event.target).value)"
            class="scale-slider"
          />
          <span class="scale-value">{{ chartScale() | number:'1.1-1' }}x</span>
        </div>
      </div>
      <div class="chart-canvas-container">
        <canvas #chartCanvas class="chart-canvas"></canvas>
        <div class="y-axis-right" #yAxisRight></div>
        <div class="y-axis-left" #yAxisLeft></div>
      </div>
    </div>
  `,
    styles: [`
    :host {
      display: block;
      width: 100%;
      height: 100%;
      position: relative;
    }
    .chart-wrapper {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      background: #0f172a;
      border-radius: 8px;
      overflow: hidden;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
      background: rgba(15, 23, 42, 0.6);
    }
    .chart-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.85rem;
      font-weight: 600;
      color: #e2e8f0;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    .scale-control {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .scale-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    .scale-slider {
      width: 100px;
      height: 6px;
      accent-color: #38bdf8;
      cursor: pointer;
    }
    .scale-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #cbd5e1;
      font-weight: 600;
      min-width: 40px;
    }
    .chart-canvas-container {
      flex: 1;
      position: relative;
      min-height: 0;
    }
    .chart-canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .y-axis-right {
      position: absolute;
      right: 0;
      top: 0;
      bottom: 0;
      width: 60px;
      background: rgba(15, 23, 42, 0.95);
      border-left: 1px solid rgba(148, 163, 184, 0.2);
      pointer-events: none;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      color: #94a3b8;
    }
    .y-axis-left {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 50px;
      background: rgba(15, 23, 42, 0.95);
      border-right: 1px solid rgba(148, 163, 184, 0.2);
      pointer-events: none;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px;
      color: #94a3b8;
    }
  `]
})
export class SmartChartComponent implements OnInit, AfterViewInit {
  private stream = inject(DataStreamService);
  private derived = inject(LevelDerivedService);
  private analytics = inject(FlowAnalyticsService);
  private viewportService = inject(ViewportSelectionService);
  private chartSettings = inject(ChartSettingsService);

  @ViewChild('chartCanvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('yAxisRight', { static: true }) yAxisRef!: ElementRef<HTMLDivElement>;
  @ViewChild('yAxisLeft', { static: true }) yAxisLeft!: ElementRef<HTMLDivElement>;

  private ctx!: CanvasRenderingContext2D;
  private candles: Candle[] = [];
  private chartCandles: ChartCandle[] = [];
  private lastCandleTime: number = 0;
  
  private candleWidth = 12;
  private candleSpacing = 3;
  private padding = { top: 20, right: 80, bottom: 40, left: 70 }; // Increased left padding for GEX
  
  private priceRange = signal<{ min: number; max: number } | null>(null);
  private hoveredLevelId: string | null = null;
  private levelHitBoxes: Map<string, { x: number; y: number; width: number; height: number; level: DerivedLevel }> = new Map();
  private chartScaleSignal = signal(1.0);
  
  public chartScale = this.chartScaleSignal.asReadonly();
  
  spy = computed(() => this.derived.spy());
  levels = computed(() => this.derived.levels());

    constructor() {
        // Reset candle aggregation when timeframe changes
        effect(() => {
          const _tf = this.chartSettings.timeframe();
          this.resetCandles();
        });

        // Effect to handle price updates (event-time bucketing)
        effect(() => {
          const payload = this.stream.levelsData();
          const spot = payload?.spy?.spot ?? null;
          const ts = payload?.ts ?? null; // event-time ms
          if (typeof spot === 'number' && Number.isFinite(spot) && typeof ts === 'number' && Number.isFinite(ts)) {
            this.updateRealtimeCandle(spot, ts);
          }
        });

    // Effect to redraw when levels change
        effect(() => {
      const _ = this.levels();
      this.draw();
        });
    }

    ngOnInit() {
    // Initialize with empty state - candles will build in real-time
    this.initializeEmptyState();
  }

  ngAfterViewInit() {
    this.initCanvas();
    
    // Handle resize
    new ResizeObserver(() => {
      this.initCanvas();
      this.draw();
    }).observe(this.canvasRef.nativeElement);
    
    // Add click handler for level selection
    this.canvasRef.nativeElement.addEventListener('click', (e) => {
      this.handleCanvasClick(e);
    });
    
    // Add hover handler for level highlighting
    this.canvasRef.nativeElement.addEventListener('mousemove', (e) => {
      this.handleCanvasHover(e);
    });
  }

  private initCanvas() {
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    
    this.ctx = ctx;
  }

  private initializeEmptyState() {
    // Start with empty candle array - real-time updates will build candles
    this.candles = [];
    
    // Initialize with a default price range that will adjust as data arrives
    const defaultSpot = 685.0; // Reasonable default for SPY
    this.priceRange.set({
      min: defaultSpot - 2,
      max: defaultSpot + 2
    });
    
    this.draw();
  }

  private computePriceRange() {
    const payload = this.stream.levelsData();
    const spot = payload?.spy?.spot ?? null;
    const levels = this.getRenderableLevels();
    
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    
    // Include candle prices
    if (this.candles.length > 0) {
      for (const c of this.candles) {
        if (Number.isFinite(c.high) && c.high > max) max = c.high;
        if (Number.isFinite(c.low) && c.low < min) min = c.low;
      }
    }
    
    // Include current spot
    if (typeof spot === 'number' && Number.isFinite(spot)) {
      if (spot > max) max = spot;
      if (spot < min) min = spot;
    }
    
    // Include renderable levels in range (but ignore absurd outliers)
    if (typeof spot === 'number' && Number.isFinite(spot)) {
      const maxAbsDist = Math.max(50, spot * 0.3); // guardrail against bad data (e.g. 5.83)
      for (const level of levels) {
        if (!Number.isFinite(level.price)) continue;
        if (Math.abs(level.price - spot) > maxAbsDist) continue;
        if (level.price > max) max = level.price;
        if (level.price < min) min = level.price;
      }
    } else {
      for (const level of levels) {
        if (!Number.isFinite(level.price)) continue;
        if (level.price > max) max = level.price;
        if (level.price < min) min = level.price;
      }
    }
    
    // Sanity check
    if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max) {
      // Fallback
      const fallbackSpot = spot?.spot || 685.0;
      this.priceRange.set({ min: fallbackSpot - 3, max: fallbackSpot + 3 });
      return;
    }
    
    const center = (typeof spot === 'number' && Number.isFinite(spot)) ? spot : (min + max) / 2;
    const maxDist = Math.max(Math.abs(max - center), Math.abs(center - min), 0.5);
    
    // Scale behaves like a TradingView vertical zoom: higher = zoom in (smaller range)
    const scale = this.chartScaleSignal();
    const baseRange = maxDist * 2;
    const scaledRange = baseRange / Math.max(0.5, scale);
    const pad = Math.max(0.5, scaledRange * 0.05);
    const finalRange = scaledRange + pad * 2;
    
    this.priceRange.set({
      min: center - finalRange / 2,
      max: center + finalRange / 2
    });
  }

  public onScaleChange(value: string) {
    const scale = Number(value);
    if (Number.isFinite(scale)) {
      this.chartScaleSignal.set(scale);
      this.computePriceRange();
      this.draw();
    }
  }

  private updateRealtimeCandle(price: number, tsMs: number) {
    const timeframe = this.chartSettings.timeframe();
    const intervalSec = this.timeframeToSeconds(timeframe);
    const tsSec = Math.floor(tsMs / 1000);
    const candleTime = Math.floor(tsSec / intervalSec) * intervalSec;

    // Ignore out-of-order timestamps
    if (this.lastCandleTime && candleTime < this.lastCandleTime) {
      return;
    }
    
    // If we have no candles yet, start the first one
    if (this.candles.length === 0) {
      const newCandle: Candle = {
        time: candleTime,
        open: price,
        high: price,
        low: price,
        close: price,
        volume: 0
      };
      this.candles.push(newCandle);
      this.lastCandleTime = candleTime;
      this.computePriceRange();
      this.draw();
      return;
    }
    
    // Check if we need to start a new candle
    if (candleTime > this.lastCandleTime) {
      // Start new candle
      const newCandle: Candle = {
        time: candleTime,
        open: price,
        high: price,
        low: price,
        close: price,
        volume: 0
      };
      this.candles.push(newCandle);
      this.lastCandleTime = candleTime;
      
      // Keep a decent window so it behaves more like a chart
      if (this.candles.length > 500) {
        this.candles = this.candles.slice(-500);
      }
    } else {
      // Update current candle (last one in array)
      const currentCandle = this.candles[this.candles.length - 1];
      currentCandle.close = price;
      currentCandle.high = Math.max(currentCandle.high, price);
      currentCandle.low = Math.min(currentCandle.low, price);
    }
    
    this.computePriceRange();
    this.draw();
  }

  private draw() {
    if (!this.ctx) return;
    
    const canvas = this.canvasRef.nativeElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    
    // Clear
    this.ctx.clearRect(0, 0, width, height);
    
    // Background
    this.ctx.fillStyle = '#0f172a';
    this.ctx.fillRect(0, 0, width, height);
    
    const priceRange = this.priceRange();
    if (!priceRange) return;
    
    const chartWidth = width - this.padding.left - this.padding.right;
    const chartHeight = height - this.padding.top - this.padding.bottom;
    
    // Draw grid
    this.drawGrid(chartWidth, chartHeight, priceRange);
    
    // Draw GEX bars FIRST (behind candles and levels)
    if (this.chartSettings.showGEX()) {
      this.drawGEXBars(priceRange, chartHeight, chartWidth);
    }
    
    // Calculate visible candles
    const totalCandleWidth = this.candleWidth + this.candleSpacing;
    const visibleCount = Math.floor(chartWidth / totalCandleWidth);
    const startIndex = Math.max(0, this.candles.length - visibleCount);
    const visibleCandles = this.candles.slice(startIndex);
    
    // Map to chart coordinates
    this.chartCandles = visibleCandles.map((c, i) => {
      const x = this.padding.left + i * totalCandleWidth + this.candleWidth / 2;
      return {
        ...c,
        x,
        yOpen: this.priceToY(c.open, priceRange, chartHeight),
        yHigh: this.priceToY(c.high, priceRange, chartHeight),
        yLow: this.priceToY(c.low, priceRange, chartHeight),
        yClose: this.priceToY(c.close, priceRange, chartHeight)
      };
    });
    
    // Draw candles
    for (const c of this.chartCandles) {
      this.drawCandle(c);
    }
    
    // Draw level overlays ON TOP
    this.drawLevelOverlays(priceRange, chartHeight, chartWidth);
    
    // Update Y-axes
    this.updateYAxis(priceRange);
  }

  private drawGrid(chartWidth: number, chartHeight: number, priceRange: { min: number; max: number }) {
    this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
    this.ctx.lineWidth = 1;
    
    // Horizontal grid lines at $0.25 increments
    const minRounded = Math.floor(priceRange.min * 4) / 4;
    const maxRounded = Math.ceil(priceRange.max * 4) / 4;
    
    for (let price = minRounded; price <= maxRounded; price += 0.25) {
      if (price < priceRange.min || price > priceRange.max) continue;
      
      const y = this.priceToY(price, priceRange, chartHeight);
      
      // Make whole dollar lines more prominent
      const isWholeDollar = Math.abs(price - Math.round(price)) < 0.01;
      this.ctx.strokeStyle = isWholeDollar 
        ? 'rgba(148, 163, 184, 0.4)' 
        : 'rgba(51, 65, 85, 0.25)';
      
      this.ctx.beginPath();
      this.ctx.moveTo(this.padding.left, y);
      this.ctx.lineTo(this.padding.left + chartWidth, y);
      this.ctx.stroke();
    }
  }

  private drawCandle(c: ChartCandle) {
    const isUp = c.close >= c.open;
    const color = isUp ? '#22c55e' : '#ef4444';
    
    this.ctx.strokeStyle = color;
    this.ctx.fillStyle = color;
    this.ctx.lineWidth = 1;
    
    // Draw wick
    this.ctx.beginPath();
    this.ctx.moveTo(c.x, c.yHigh);
    this.ctx.lineTo(c.x, c.yLow);
    this.ctx.stroke();
    
    // Draw body
    const bodyTop = Math.min(c.yOpen, c.yClose);
    const bodyHeight = Math.abs(c.yClose - c.yOpen);
    const bodyX = c.x - this.candleWidth / 2;
    
    if (bodyHeight < 2) {
      // Doji - draw a line
      this.ctx.fillRect(bodyX, bodyTop, this.candleWidth, 2);
    } else {
      this.ctx.fillRect(bodyX, bodyTop, this.candleWidth, bodyHeight);
    }
  }

  private drawLevelOverlays(priceRange: { min: number; max: number }, chartHeight: number, chartWidth: number) {
    const levels = this.getRenderableLevels();
    this.levelHitBoxes.clear();
    
    if (!levels || levels.length === 0) return;
    
    for (const level of levels) {
      if (level.price < priceRange.min || level.price > priceRange.max) continue;
      
      const y = this.priceToY(level.price, priceRange, chartHeight);
      const isUp = level.direction === 'UP';
      const isHovered = this.hoveredLevelId === level.id;
      
      // Color based on direction and hover state
      const baseColor = isUp ? '#ef4444' : '#22c55e';
      const lineAlpha = isHovered ? '0.9' : '0.6';
      const color = `${baseColor}${Math.floor(parseFloat(lineAlpha) * 255).toString(16).padStart(2, '0')}`;
      
      // Draw level line (dashed)
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = isHovered ? 3 : 2;
      this.ctx.setLineDash([8, 4]);
      
      this.ctx.beginPath();
      this.ctx.moveTo(this.padding.left, y);
      this.ctx.lineTo(this.padding.left + chartWidth, y);
      this.ctx.stroke();
      
      this.ctx.setLineDash([]);
      
      // Draw label box (clickable, floating on right side)
      const labelText = `${level.kind} ${level.price.toFixed(2)}`;
      this.ctx.font = 'bold 11px "IBM Plex Mono", monospace';
      const textWidth = this.ctx.measureText(labelText).width;
      const boxPadding = 8;
      const boxWidth = textWidth + boxPadding * 2;
      const boxHeight = 24;
      const boxX = this.padding.left + chartWidth - boxWidth - 15;
      const boxY = y - boxHeight / 2;
      
      // Draw box background with stronger opacity
      this.ctx.fillStyle = isHovered ? 'rgba(15, 23, 42, 0.98)' : 'rgba(15, 23, 42, 0.92)';
      this.ctx.fillRect(boxX, boxY, boxWidth, boxHeight);
      
      // Draw box border
      this.ctx.strokeStyle = baseColor;
      this.ctx.lineWidth = isHovered ? 2 : 1.5;
      this.ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
      
      // Draw text
      this.ctx.fillStyle = baseColor;
      this.ctx.fillText(labelText, boxX + boxPadding, boxY + boxHeight / 2 + 4);
      
      // Store hit box for click/hover detection (expand hit area)
      this.levelHitBoxes.set(level.id, {
        x: boxX - 5,
        y: boxY - 5,
        width: boxWidth + 10,
        height: boxHeight + 10,
        level
      });
    }
  }

  private drawGEXBars(priceRange: { min: number; max: number }, chartHeight: number, chartWidth: number) {
    const gammaByStrike = this.analytics.netGammaByStrike();
    
    if (!gammaByStrike || gammaByStrike.size === 0) {
      return;
    }
    
    // Find max gamma for scaling (only in visible range)
    let maxGamma = 0;
    const visibleGamma = new Map<number, number>();
    
    for (const [strike, gamma] of gammaByStrike.entries()) {
      if (strike < priceRange.min || strike > priceRange.max) continue;
      visibleGamma.set(strike, gamma);
      maxGamma = Math.max(maxGamma, Math.abs(gamma));
    }
    
    if (maxGamma === 0 || visibleGamma.size === 0) return;
    
    // Max bar width is ~15% of chart width, extending from left edge
    const maxBarWidth = chartWidth * 0.15;
    
    for (const [strike, gamma] of visibleGamma.entries()) {
      const y = this.priceToY(strike, priceRange, chartHeight);
      const barWidth = (Math.abs(gamma) / maxGamma) * maxBarWidth;
      const isPositive = gamma > 0;
      
      // More vibrant colors for visibility
      const color = isPositive 
        ? 'rgba(34, 197, 94, 0.5)' 
        : 'rgba(239, 68, 68, 0.5)';
      
      // Draw horizontal bar extending from left edge
      this.ctx.fillStyle = color;
      const barHeight = 8;
      this.ctx.fillRect(this.padding.left, y - barHeight / 2, barWidth, barHeight);
      
      // Draw border for definition
      this.ctx.strokeStyle = isPositive ? '#22c55e' : '#ef4444';
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(this.padding.left, y - barHeight / 2, barWidth, barHeight);
    }
  }

  private updateYAxis(priceRange: { min: number; max: number }) {
    const yAxis = this.yAxisRef.nativeElement;
    yAxis.innerHTML = '';
    
    // Calculate $0.25 increments
    const minRounded = Math.floor(priceRange.min * 4) / 4; // Round down to nearest 0.25
    const maxRounded = Math.ceil(priceRange.max * 4) / 4;   // Round up to nearest 0.25
    
    const chartHeight = yAxis.clientHeight - this.padding.top - this.padding.bottom;
    
    // Draw labels for each $0.25 increment
    for (let price = minRounded; price <= maxRounded; price += 0.25) {
      if (price < priceRange.min || price > priceRange.max) continue;
      
      const y = this.priceToY(price, priceRange, chartHeight);
      
      const label = document.createElement('div');
      label.style.position = 'absolute';
      label.style.top = `${y + this.padding.top}px`;
      label.style.right = '5px';
      label.style.transform = 'translateY(-50%)';
      label.style.fontSize = '10px';
      label.textContent = price.toFixed(2);
      
      yAxis.appendChild(label);
    }
    
    // Update left Y-axis (GEX scale) if needed
    const yAxisLeft = this.yAxisLeft.nativeElement;
    yAxisLeft.innerHTML = '';
    
    // GEX axis label
    const gexLabel = document.createElement('div');
    gexLabel.style.position = 'absolute';
    gexLabel.style.top = '10px';
    gexLabel.style.left = '5px';
    gexLabel.style.fontSize = '9px';
    gexLabel.style.color = '#64748b';
    gexLabel.textContent = 'GEX';
    yAxisLeft.appendChild(gexLabel);
  }

  private priceToY(price: number, priceRange: { min: number; max: number }, chartHeight: number): number {
    const ratio = (price - priceRange.min) / (priceRange.max - priceRange.min);
    return this.padding.top + chartHeight - (ratio * chartHeight);
  }

  private handleCanvasClick(event: MouseEvent) {
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Check if click is on any level label box
    for (const hitBox of this.levelHitBoxes.values()) {
      if (
        x >= hitBox.x && x <= hitBox.x + hitBox.width &&
        y >= hitBox.y && y <= hitBox.y + hitBox.height
      ) {
        // Level clicked - emit selection event
        this.selectLevel(hitBox.level);
        return;
      }
    }
  }

  private handleCanvasHover(event: MouseEvent) {
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    let foundHover = false;
    
    // Check if hover is on any level label box
    for (const hitBox of this.levelHitBoxes.values()) {
      if (
        x >= hitBox.x && x <= hitBox.x + hitBox.width &&
        y >= hitBox.y && y <= hitBox.y + hitBox.height
      ) {
        if (this.hoveredLevelId !== hitBox.level.id) {
          this.hoveredLevelId = hitBox.level.id;
          canvas.style.cursor = 'pointer';
          this.draw();
        }
        foundHover = true;
        break;
      }
    }
    
    if (!foundHover && this.hoveredLevelId !== null) {
      this.hoveredLevelId = null;
      canvas.style.cursor = 'default';
      this.draw();
    }
  }

  private selectLevel(level: DerivedLevel) {
    // Always allow selection; ML may or may not be available.
    this.viewportService.selectLevel(level.id, false);
  }

  private resetCandles() {
    this.candles = [];
    this.lastCandleTime = 0;
    this.computePriceRange();
    this.draw();
  }

  private timeframeToSeconds(tf: ChartTimeframe): number {
    if (tf === '5m') return 5 * 60;
    if (tf === '15m') return 15 * 60;
    return 2 * 60;
  }

  private getRenderableLevels(): DerivedLevel[] {
    const levels = this.levels();
    if (!levels || levels.length === 0) return [];

    const showVWAP = this.chartSettings.showVWAP();
    const showSMAs = this.chartSettings.showSMAs();
    const showWalls = this.chartSettings.showWalls();
    const showStructural = this.chartSettings.showStructural();

    return levels.filter((l) => {
      switch (l.kind) {
        case 'VWAP':
          return showVWAP;
        case 'SMA_200':
        case 'SMA_400':
          return showSMAs;
        case 'PM_HIGH':
        case 'PM_LOW':
        case 'OR_HIGH':
        case 'OR_LOW':
          return showStructural;
        case 'CALL_WALL':
        case 'PUT_WALL':
        case 'GAMMA_FLIP':
        case 'STRIKE':
        case 'ROUND':
          return showWalls;
        default:
          // Keep other “physics” levels behind the Walls toggle for now
          return showWalls;
      }
    });
    }
}
