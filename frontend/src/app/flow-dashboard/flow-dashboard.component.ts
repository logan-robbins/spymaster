import { Component, computed, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import {
    FlowAnalyticsService,
    FlowMetric,
    TIMESCALES_MS,
    TIMESCALE_LABEL,
    TimescaleMs,
    DerivativeOrder
} from '../flow-analytics.service';
import { StrikeGridComponent } from '../strike-grid/strike-grid.component';
import { FlowWaveComponent } from '../flow-wave/flow-wave.component';

@Component({
    selector: 'app-flow-dashboard',
    standalone: true,
    imports: [CommonModule, DecimalPipe, StrikeGridComponent, FlowWaveComponent],
    template: `
    <div class="p-4">
      <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3">
          <div class="rounded-md border border-gray-800 bg-gray-900/60 px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-gray-400">ATM strike (derived)</div>
            <div class="font-mono text-sm text-gray-100">{{ analytics.atmStrike() ?? '—' }}</div>
          </div>

          <div class="rounded-md border border-gray-800 bg-gray-900/60 px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-gray-400">Net (selected)</div>
            <div class="font-mono text-sm" [class.text-green-400]="latest().net >= 0" [class.text-red-400]="latest().net < 0">
              {{ latest().net | number:'1.0-2' }}
            </div>
          </div>
        </div>

        <div class="flex flex-wrap items-center gap-2">
          <!-- Metric -->
          <div class="flex overflow-hidden rounded-md border border-gray-800">
            @for (m of metrics; track m) {
              <button
                class="px-3 py-2 text-xs font-semibold"
                [class.bg-gray-800]="analytics.selectedMetric() === m"
                [class.text-white]="analytics.selectedMetric() === m"
                [class.text-gray-300]="analytics.selectedMetric() !== m"
                (click)="analytics.selectedMetric.set(m)"
              >
                {{ m.toUpperCase() }}
              </button>
            }
          </div>

          <!-- Order -->
          <div class="flex overflow-hidden rounded-md border border-gray-800">
            @for (o of orders; track o) {
              <button
                class="px-3 py-2 text-xs font-semibold"
                [class.bg-gray-800]="analytics.selectedOrder() === o"
                [class.text-white]="analytics.selectedOrder() === o"
                [class.text-gray-300]="analytics.selectedOrder() !== o"
                (click)="analytics.selectedOrder.set(o)"
              >
                d{{ o }}
              </button>
            }
          </div>

          <!-- Timescale -->
          <div class="flex overflow-hidden rounded-md border border-gray-800">
            @for (ts of timescales; track ts) {
              <button
                class="px-3 py-2 text-xs font-semibold"
                [class.bg-gray-800]="analytics.selectedTimescaleMs() === ts"
                [class.text-white]="analytics.selectedTimescaleMs() === ts"
                [class.text-gray-300]="analytics.selectedTimescaleMs() !== ts"
                (click)="analytics.selectedTimescaleMs.set(ts)"
              >
                {{ timescaleLabel[ts] }}
              </button>
            }
          </div>
        </div>
      </div>

      <div class="grid grid-cols-12 gap-4">
        <!-- Left: Strike ladder -->
        <div class="col-span-12 xl:col-span-7">
          <app-strike-grid></app-strike-grid>
        </div>

        <!-- Right: Pressure + wave -->
        <div class="col-span-12 xl:col-span-5 space-y-4">
          <div class="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div class="mb-3 flex items-center justify-between">
              <div class="text-xs font-semibold text-gray-200">Above/Below Pressure (selected)</div>
              <div class="text-[10px] uppercase tracking-wider text-gray-500 font-mono">
                {{ analytics.selectedMetric() }} · d{{ analytics.selectedOrder() }} · {{ timescaleLabel[analytics.selectedTimescaleMs()] }}
              </div>
            </div>

            <div class="space-y-2">
              @for (row of bucketRows(); track row.key) {
                <div class="grid grid-cols-12 items-center gap-2">
                  <div class="col-span-4 text-xs text-gray-300">{{ row.label }}</div>
                  <div class="col-span-6 h-2 overflow-hidden rounded bg-gray-800">
                    <div
                      class="h-2 rounded"
                      [style.width.%]="row.pct"
                      [style.background]="row.color"
                      [style.opacity]="0.85"
                    ></div>
                  </div>
                  <div class="col-span-2 text-right font-mono text-xs" [style.color]="row.color">
                    {{ row.value | number:'1.0-2' }}
                  </div>
                </div>
              }
            </div>
          </div>

          <app-flow-wave
            [label]="'Calls (>ATM) wave'"
            [series]="seriesCallAbove()"
            [stroke]="'rgba(34,197,94,0.9)'"
            nowMode="center"
          ></app-flow-wave>

          <app-flow-wave
            [label]="'Puts (<ATM) wave'"
            [series]="seriesPutBelow()"
            [stroke]="'rgba(244,63,94,0.9)'"
            nowMode="center"
          ></app-flow-wave>
        </div>
      </div>
    </div>
  `
})
export class FlowDashboardComponent {
    public analytics = inject(FlowAnalyticsService);

    public metrics: readonly FlowMetric[] = ['premium', 'delta', 'gamma'] as const;
    public orders: readonly DerivativeOrder[] = [1, 2, 3] as const;
    public timescales: readonly TimescaleMs[] = TIMESCALES_MS;
    public timescaleLabel = TIMESCALE_LABEL;

    public latest = computed(() => this.analytics.getLatestByBucket());

    public seriesCallAbove = computed(() => this.analytics.getSeries('call_above'));
    public seriesPutBelow = computed(() => this.analytics.getSeries('put_below'));

    public bucketRows = computed(() => {
        const latest = this.latest();
        const maxAbs = this.analytics.getLatestAbsMax();
        const toPct = (v: number) => clamp((Math.abs(v) / maxAbs) * 100, 0, 100);
        return [
            {
                key: 'call_above',
                label: 'Calls > ATM',
                value: latest.call_above,
                pct: toPct(latest.call_above),
                color: 'rgb(34,197,94)'
            },
            {
                key: 'call_below',
                label: 'Calls < ATM',
                value: latest.call_below,
                pct: toPct(latest.call_below),
                color: 'rgb(16,185,129)'
            },
            {
                key: 'put_above',
                label: 'Puts > ATM',
                value: latest.put_above,
                pct: toPct(latest.put_above),
                color: 'rgb(248,113,113)'
            },
            {
                key: 'put_below',
                label: 'Puts < ATM',
                value: latest.put_below,
                pct: toPct(latest.put_below),
                color: 'rgb(244,63,94)'
            }
        ];
    });
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}


