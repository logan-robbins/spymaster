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

    rows = computed((): Array<{ strike: number; call?: FlowMetrics; put?: FlowMetrics; netGamma: number; netDelta: number }> => {
        const data = this.flowData();
        const map = new Map<number, { strike: number; call?: FlowMetrics; put?: FlowMetrics; netGamma: number; netDelta: number }>();

        // Group
        for (const key in data) {
            const item = data[key];
            const strike = item.strike_price;

            if (!map.has(strike)) {
                map.set(strike, { strike, netGamma: 0, netDelta: 0 });
            }

            const row = map.get(strike)!;
            if (item.type === 'C') {
                row.call = item;
            } else {
                row.put = item;
            }
        }

        // Aggregate net gamma / delta at the strike level
        for (const row of map.values()) {
            const callGamma = row.call?.net_gamma_flow ?? 0;
            const putGamma = row.put?.net_gamma_flow ?? 0;
            const callDelta = row.call?.net_delta_flow ?? 0;
            const putDelta = row.put?.net_delta_flow ?? 0;
            row.netGamma = callGamma + putGamma;
            row.netDelta = callDelta + putDelta;
        }

        // Sort High to Low
        return Array.from(map.values()).sort((a, b) => b.strike - a.strike);
    });

    maxAbsNetGamma = computed(() => {
        let max = 1;
        for (const row of this.rows()) {
            max = Math.max(max, Math.abs(row.netGamma));
        }
        return max;
    });

    atmStrike = computed(() => this.analytics.atmStrike());

    // Helper for Net Delta Color
    getDeltaColor(val: number | null | undefined): string {
        if (val == null || !Number.isFinite(val) || val === 0) return 'gray-text';
        return val > 0 ? 'green-text' : 'red-text';
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
        const alpha = 0.15 + intensity * 0.55; // Increased visibility
        // Calls = green, Puts = red
        return side === 'call' ? `rgba(34,197,94,${alpha})` : `rgba(244,63,94,${alpha})`;
    }

    getGammaBarWidth(netGamma: number): number {
        const max = this.maxAbsNetGamma();
        if (!Number.isFinite(netGamma) || !Number.isFinite(max) || max <= 0) return 0;
        return Math.min(100, (Math.abs(netGamma) / max) * 100);
    }

    getGammaBarColor(netGamma: number): string {
        if (!Number.isFinite(netGamma) || netGamma === 0) return 'rgba(148, 163, 184, 0.18)';
        return netGamma > 0 ? 'rgba(34, 197, 94, 0.35)' : 'rgba(248, 113, 113, 0.35)';
    }

    formatK(value: number | null | undefined): string {
        if (value == null || !Number.isFinite(value)) return '0';
        const sign = value > 0 ? '+' : value < 0 ? '-' : '';
        const abs = Math.abs(value);
        if (abs >= 1_000_000) return `${sign}${(abs / 1_000_000).toFixed(1)}M`;
        if (abs >= 1_000) return `${sign}${(abs / 1_000).toFixed(1)}K`;
        return `${sign}${abs.toFixed(0)}`;
    }
}
