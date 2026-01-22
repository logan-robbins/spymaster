/**
 * HUD State - Manages visualization state
 */

import type { GexRow } from './data-loader';

export class HUDState {
    private gexData: GexRow[] = [];
    private spotRef: number = 6000;
    private timeRange: { start: bigint; end: bigint } = { start: 0n, end: 0n };
    private priceRange: { min: number; max: number } = { min: 5900, max: 6100 };

    setGexData(data: GexRow[]): void {
        this.gexData = data;

        if (data.length > 0) {
            // Extract time range (timestamps are BigInt from Arrow)
            const timestamps = data.map(r => r.window_end_ts_ns);
            let minTs = timestamps[0];
            let maxTs = timestamps[0];
            for (const ts of timestamps) {
                if (ts < minTs) minTs = ts;
                if (ts > maxTs) maxTs = ts;
            }
            this.timeRange = { start: minTs, end: maxTs };

            // Extract price/strike range (these are numbers)
            const strikes = data.map(r => Number(r.strike_points));
            const spots = data.map(r => Number(r.underlying_spot_ref));

            this.spotRef = spots[spots.length - 1];

            let minPrice = strikes[0];
            let maxPrice = strikes[0];
            for (const s of strikes) {
                if (s < minPrice) minPrice = s;
                if (s > maxPrice) maxPrice = s;
            }
            for (const s of spots) {
                if (s < minPrice) minPrice = s;
                if (s > maxPrice) maxPrice = s;
            }

            this.priceRange = {
                min: minPrice - 10,
                max: maxPrice + 10,
            };
        }
    }

    getGexData(): GexRow[] {
        return this.gexData;
    }

    getSpotRef(): number {
        return this.spotRef;
    }

    getTimeRange(): { start: bigint; end: bigint } {
        return this.timeRange;
    }

    getPriceRange(): { min: number; max: number } {
        return this.priceRange;
    }

    // Get GEX values for a specific time window, mapped to price axis
    getGexAtTime(windowEndTs: bigint): Map<number, { gex: number; gexAbs: number; imbalance: number }> {
        const result = new Map();

        for (const row of this.gexData) {
            if (row.window_end_ts_ns === windowEndTs) {
                result.set(row.strike_points, {
                    gex: row.gex,
                    gexAbs: row.gex_abs,
                    imbalance: row.gex_imbalance_ratio,
                });
            }
        }

        return result;
    }

    // Get unique time windows
    getTimeWindows(): bigint[] {
        const set = new Set(this.gexData.map(r => r.window_end_ts_ns));
        return Array.from(set).sort();
    }
}
