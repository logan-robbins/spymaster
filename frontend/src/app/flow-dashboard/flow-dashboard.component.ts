import { Component, computed, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import {
    FlowAnalyticsService,
    FlowMetric,
    BAR_INTERVALS_MS,
    BAR_INTERVAL_LABEL,
    BarIntervalMs,
    DerivativeOrder
} from '../flow-analytics.service';
import { StrikeGridComponent } from '../strike-grid/strike-grid.component';
import { FlowChartComponent } from '../flow-chart/flow-chart.component';

@Component({
    selector: 'app-flow-dashboard',
    standalone: true,
    imports: [CommonModule, DecimalPipe, StrikeGridComponent, FlowChartComponent],
    template: `
    <div class="dashboard-container">
      <div class="dashboard-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
          <div style="display: flex; gap: 1rem;">
            <div class="stat-card">
              <div class="stat-label">ATM Strike (derived)</div>
              <div class="stat-value">{{ analytics.atmStrike() ?? '—' }}</div>
            </div>

            <div class="stat-card">
              <div class="stat-label">Net (selected)</div>
              <div class="stat-value" [class.green-text]="latest().net >= 0" [class.red-text]="latest().net < 0">
                {{ latest().net | number:'1.0-2' }}
              </div>
            </div>
          </div>

          <div class="control-panel">
            <!-- Metric -->
            <div class="btn-group">
              @for (m of metrics; track m) {
                <button
                  class="btn"
                  [class.active]="analytics.selectedMetric() === m"
                  (click)="analytics.selectedMetric.set(m)"
                >
                  {{ m.toUpperCase() }}
                </button>
              }
            </div>

            <!-- Order -->
            <div class="btn-group">
              @for (o of orders; track o) {
                <button
                  class="btn"
                  [class.active]="analytics.selectedOrder() === o"
                  (click)="analytics.selectedOrder.set(o)"
                >
                  d{{ o }}
                </button>
              }
            </div>

            <!-- Bar Interval -->
            <div class="btn-group">
              @for (interval of barIntervals; track interval) {
                <button
                  class="btn"
                  [class.active]="analytics.selectedBarInterval() === interval"
                  (click)="analytics.selectedBarInterval.set(interval)"
                >
                  {{ barIntervalLabel[interval] }}
                </button>
              }
            </div>
          </div>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 400px; gap: 1rem; height: calc(100vh - 180px);">
        <!-- Left: Chart -->
        <div>
          <app-flow-chart></app-flow-chart>
        </div>

        <!-- Right: Strike ladder -->
        <div style="overflow-y: auto;">
          <app-strike-grid></app-strike-grid>
        </div>
      </div>
    </div>
  `
})
export class FlowDashboardComponent {
    public analytics = inject(FlowAnalyticsService);

    public metrics: readonly FlowMetric[] = ['premium', 'delta', 'gamma'] as const;
    public orders: readonly DerivativeOrder[] = [1, 2, 3] as const;
    public barIntervals: readonly BarIntervalMs[] = BAR_INTERVALS_MS;
    public barIntervalLabel = BAR_INTERVAL_LABEL;

    public latest = computed(() => this.analytics.getLatestByBucket());
}


