import { Component, OnInit, OnDestroy, ElementRef, ViewChild, inject, effect, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { createChart, IChartApi, ISeriesApi, LineData } from 'lightweight-charts';
import { DataStreamService } from '../data-stream.service';
import { FlowAnalyticsService, FlowMetric, DerivativeOrder, BarIntervalMs, BucketKey } from '../flow-analytics.service';

@Component({
  selector: 'app-flow-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-container">
      <div class="chart-header">
        <h3>{{ chartTitle() }}</h3>
        <div class="controls">
          <div class="control-group">
            <label>Metric:</label>
            <button 
              [class.active]="selectedMetric() === 'delta'" 
              (click)="setMetric('delta')">DELTA</button>
            <button 
              [class.active]="selectedMetric() === 'gamma'" 
              (click)="setMetric('gamma')">GAMMA</button>
          </div>
          <div class="control-group">
            <label>Derivative:</label>
            <button 
              [class.active]="selectedOrder() === 1" 
              (click)="setOrder(1)">d1</button>
            <button 
              [class.active]="selectedOrder() === 2" 
              (click)="setOrder(2)">d2</button>
            <button 
              [class.active]="selectedOrder() === 3" 
              (click)="setOrder(3)">d3</button>
          </div>
          <div class="control-group">
            <label>Bars:</label>
            <button 
              [class.active]="selectedBarInterval() === 5000" 
              (click)="setBarInterval(5000)">5s</button>
            <button 
              [class.active]="selectedBarInterval() === 30000" 
              (click)="setBarInterval(30000)">30s</button>
            <button 
              [class.active]="selectedBarInterval() === 60000" 
              (click)="setBarInterval(60000)">1m</button>
          </div>
        </div>
      </div>
      <div class="chart-stats">
        <div class="stat call-above">
          <span class="label">🟢 Bullish (> Anchor):</span>
          <span class="value">{{ latestValues().call_above | number:'1.1-1' }} {{ units() }}</span>
        </div>
        <div class="stat call-below">
          <span class="label">🟢 Bullish (< Anchor):</span>
          <span class="value">{{ latestValues().call_below | number:'1.1-1' }} {{ units() }}</span>
        </div>
        <div class="stat put-above">
          <span class="label">🔴 Bearish (> Anchor):</span>
          <span class="value">{{ latestValues().put_above | number:'1.1-1' }} {{ units() }}</span>
        </div>
        <div class="stat put-below">
          <span class="label">🔴 Bearish (< Anchor):</span>
          <span class="value">{{ latestValues().put_below | number:'1.1-1' }} {{ units() }}</span>
        </div>
        <div class="stat net">
          <span class="label">NET {{ selectedOrder() === 1 ? 'VELOCITY' : selectedOrder() === 2 ? 'ACCELERATION' : 'JERK' }}:</span>
          <span class="value" [class.positive]="latestValues().net >= 0" [class.negative]="latestValues().net < 0">
            {{ latestValues().net | number:'1.1-1' }} {{ units() }}
          </span>
        </div>
      </div>
      <div #chartContainer class="chart"></div>
    </div>
  `,
  styles: [`
    .chart-container {
      background: #1a1f2e;
      border: 1px solid #2d3748;
      border-radius: 8px;
      padding: 1rem;
      height: 700px;
      display: flex;
      flex-direction: column;
    }

    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid #2d3748;
    }

    .chart-header h3 {
      margin: 0;
      font-size: 1.125rem;
      font-weight: 600;
      color: #e2e8f0;
    }

    .controls {
      display: flex;
      gap: 1.5rem;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .control-group label {
      font-size: 0.75rem;
      color: #a0aec0;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .control-group button {
      background: #0f1419;
      color: #a0aec0;
      border: 1px solid #2d3748;
      border-radius: 4px;
      padding: 0.25rem 0.75rem;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .control-group button:hover {
      background: #2d3748;
      color: #e2e8f0;
    }

    .control-group button.active {
      background: #3182ce;
      color: #ffffff;
      border-color: #3182ce;
    }

    .chart-stats {
      display: flex;
      gap: 1rem;
      padding: 0.75rem 0;
      margin-bottom: 0.5rem;
      border-bottom: 1px solid #2d3748;
      flex-wrap: wrap;
    }

    .stat {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      flex: 1;
      min-width: 140px;
    }

    .stat .label {
      font-size: 0.625rem;
      color: #a0aec0;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .stat .value {
      font-family: 'Monaco', 'Menlo', monospace;
      font-size: 0.875rem;
      font-weight: 700;
      color: #e2e8f0;
    }

    .stat.call-above .value {
      color: #48bb78;
    }

    .stat.call-below .value {
      color: #68d391;
    }

    .stat.put-above .value {
      color: #fc8181;
    }

    .stat.put-below .value {
      color: #f56565;
    }

    .stat.net .value.positive {
      color: #48bb78;
    }

    .stat.net .value.negative {
      color: #f56565;
    }

    .chart {
      flex: 1;
      position: relative;
    }
  `]
})
export class FlowChartComponent implements OnInit, OnDestroy {
  @ViewChild('chartContainer', { static: true }) chartContainer!: ElementRef;

  private dataService = inject(DataStreamService);
  private analytics = inject(FlowAnalyticsService);

  private chart: IChartApi | null = null;
  private series: Map<BucketKey, ISeriesApi<'Line'>> = new Map();

  // Expose analytics selectors as signals
  public selectedMetric = this.analytics.selectedMetric;
  public selectedOrder = this.analytics.selectedOrder;
  public selectedBarInterval = this.analytics.selectedBarInterval;
  public atmStrike = this.analytics.atmStrike;

  // Computed signal for latest values
  public latestValues = computed(() => this.analytics.getLatestByBucket());

  // Computed signal for units label
  public units = computed(() => {
    const metric = this.selectedMetric();
    const order = this.selectedOrder();
    const interval = this.selectedBarInterval() / 1000; // Convert to seconds
    const metricSymbol = metric === 'delta' ? 'Δ' : metric === 'gamma' ? 'Γ' : '$';

    if (order === 1) return `${metricSymbol}/s`; // Velocity: flow rate per second
    if (order === 2) return `${metricSymbol}/s² per ${interval}s bar`; // Acceleration per bar
    return `${metricSymbol}/s³`; // Jerk
  });

  // Computed signal for chart title
  public chartTitle = computed(() => {
    const metric = this.selectedMetric();
    const order = this.selectedOrder();
    const interval = this.selectedBarInterval() / 1000;
    const metricName = metric === 'delta' ? 'Delta Flow' : 'Gamma Flow';
    const derivativeName = order === 1 ? 'Velocity' : order === 2 ? 'Acceleration' : 'Jerk';
    return `${metricName} ${derivativeName} [${interval}s Bars] (ATM ${this.atmStrike() || '...'})`;
  });

  constructor() {
    // Watch for analytics tick to update chart
    effect(() => {
      // Trigger on any analytics update
      this.analytics.atmStrike();
      this.updateChart();
    });
  }

  public setMetric(metric: FlowMetric) {
    this.analytics.selectedMetric.set(metric);
  }

  public setOrder(order: DerivativeOrder) {
    this.analytics.selectedOrder.set(order);
  }

  public setBarInterval(interval: BarIntervalMs) {
    this.analytics.selectedBarInterval.set(interval);
  }

  ngOnInit() {
    this.initializeChart();
  }

  ngOnDestroy() {
    if (this.chart) {
      this.chart.remove();
    }
  }

  private initializeChart() {
    const container = this.chartContainer.nativeElement;

    this.chart = createChart(container, {
      layout: {
        background: { color: '#0f1419' },
        textColor: '#a0aec0',
      },
      grid: {
        vertLines: { color: '#1a1f2e' },
        horzLines: { color: '#1a1f2e' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#2d3748',
      },
      timeScale: {
        borderColor: '#2d3748',
        timeVisible: true,
        secondsVisible: false,
      },
      localization: {
        timeFormatter: (time: number) => {
          // TradingView passes Unix timestamp in seconds
          // Convert to Eastern time in 12-hour format
          const date = new Date(time * 1000);
          const formatter = new Intl.DateTimeFormat('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            timeZone: 'America/New_York',
            hour12: true
          });
          const formatted = formatter.format(date);
          // Convert "3:52 PM" to "3:52p" (trader format)
          return formatted.replace(' PM', 'p').replace(' AM', 'a');
        },
      },
      width: container.clientWidth,
      height: container.clientHeight,
    });

    // Create 4 line series for each bucket with distinct styles
    const bucketConfig: Array<{ bucket: BucketKey; color: string; title: string; style: number }> = [
      { bucket: 'call_above', color: '#48bb78', title: '🟢 Bullish (> Anchor)', style: 0 }, // Solid
      { bucket: 'call_below', color: '#48bb78', title: '🟢 Bullish (< Anchor)', style: 2 }, // Dashed
      { bucket: 'put_above', color: '#f56565', title: '🔴 Bearish (> Anchor)', style: 0 }, // Solid
      { bucket: 'put_below', color: '#f56565', title: '🔴 Bearish (< Anchor)', style: 2 }, // Dashed
    ];

    for (const config of bucketConfig) {
      const series = this.chart.addLineSeries({
        color: config.color,
        lineWidth: 2,
        lineStyle: config.style,
        title: config.title,
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => price.toFixed(1),
        },
      });

      // Add zero reference line to first series for visual reference
      if (config.bucket === 'call_above') {
        series.createPriceLine({
          price: 0,
          color: '#718096',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: 'Zero',
        });
      }

      this.series.set(config.bucket, series);
    }

    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {
      if (this.chart && entries.length > 0) {
        const { width, height } = entries[0].contentRect;
        this.chart.applyOptions({ width, height });
      }
    });

    resizeObserver.observe(container);
  }

  private updateChart() {
    if (!this.chart || this.series.size === 0) return;

    const buckets: BucketKey[] = ['call_above', 'call_below', 'put_above', 'put_below'];

    // Update each bucket series with bar data from analytics service
    for (const bucket of buckets) {
      const chartSeries = this.series.get(bucket);
      if (!chartSeries) continue;

      // Get bar data from analytics (already has time and value)
      const barData = this.analytics.getSeries(bucket);

      // Convert to TradingView LineData format
      const lineData: LineData[] = barData.map(point => ({
        time: point.time as any,
        value: point.value
      }));

      chartSeries.setData(lineData);
    }

    // Auto-scale to fit data
    this.chart.timeScale().fitContent();
  }
}

