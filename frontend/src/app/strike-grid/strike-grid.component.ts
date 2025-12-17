import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataStreamService, FlowMap } from '../data-stream.service';
import { DecimalPipe } from '@angular/common';

@Component({
    selector: 'app-strike-grid',
    standalone: true,
    imports: [CommonModule, DecimalPipe],
    templateUrl: './strike-grid.component.html',
    styleUrl: './strike-grid.component.css' // Angular 17 uses styleUrl
})
export class StrikeGridComponent {
    private dataService = inject(DataStreamService);

    // Signal of all data
    flowData = this.dataService.flowData;

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

    rows = computed(() => {
        const data = this.flowData();
        const map = new Map<number, { strike: number, call?: any, put?: any }>();

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
    getDeltaColor(val: number): string {
        if (!val) return 'text-gray-500';
        return val > 0 ? 'text-green-400' : 'text-red-400';
    }
}
