import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataStreamService, FlowMetrics } from '../data-stream.service';
import { DecimalPipe } from '@angular/common';
import { FlowAnalyticsService } from '../flow-analytics.service';

@Component({
    selector: 'app-strike-grid',
    standalone: true,
    imports: [CommonModule, DecimalPipe],
    templateUrl: './strike-grid.component.html',
    styleUrl: './strike-grid.component.css' // Angular 17 uses styleUrl
})
export class StrikeGridComponent {
    private dataService = inject(DataStreamService);
    private analytics = inject(FlowAnalyticsService);

    // Signal of all data
    flowData = this.dataService.flowData;
    perStrikeVel = this.analytics.perStrikeVel;

    heatMaxPremium = computed(() => {
        const vel = this.perStrikeVel();
        let max = 1;
        for (const row of Object.values(vel)) {
            if (row.call?.premium_vel != null) max = Math.max(max, Math.abs(row.call.premium_vel));
            if (row.put?.premium_vel != null) max = Math.max(max, Math.abs(row.put.premium_vel));
        }
        return max;
    });

    // Computed: Get Sorted Strikes
    // We need to group by Strike Price.
    // The backend sends Flat Tickers.
    // We transform:
    /*
       [
         { strike: 572, call: {...}, put: {...} },
         ...
       ]
    */

    rows = computed((): Array<{ strike: number; call?: FlowMetrics; put?: FlowMetrics }> => {
        const data = this.flowData();
        const map = new Map<number, { strike: number; call?: FlowMetrics; put?: FlowMetrics }>();

        // Group
        for (const key in data) {
            const item = data[key];
            const strike = item.strike_price;

            if (!map.has(strike)) {
                map.set(strike, { strike });
            }

            const row = map.get(strike)!;
            if (item.type === 'C') {
                row.call = item;
            } else {
                row.put = item;
            }
        }

        // Sort High to Low
        return Array.from(map.values()).sort((a, b) => b.strike - a.strike);
    });

    // Helper for Net Delta Color
    getDeltaColor(val: number | null | undefined): string {
        if (val == null || !Number.isFinite(val) || val === 0) return 'text-gray-500';
        return val > 0 ? 'text-green-400' : 'text-red-400';
    }

    getPremiumVelocity(strike: number, side: 'call' | 'put'): number {
        const row = this.perStrikeVel()[strike];
        if (!row) return 0;
        return side === 'call' ? row.call?.premium_vel ?? 0 : row.put?.premium_vel ?? 0;
    }

    getPremiumHeatBg(strike: number, side: 'call' | 'put'): string {
        const v = this.getPremiumVelocity(strike, side);
        const max = this.heatMaxPremium();
        const intensity = Math.min(1, Math.abs(v) / max);
        const alpha = 0.08 + intensity * 0.35;
        // Calls = green, Puts = red
        return side === 'call' ? `rgba(34,197,94,${alpha})` : `rgba(244,63,94,${alpha})`;
    }
}
