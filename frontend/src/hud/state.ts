/**
 * HUD State - Manages visualization state
 */

import type { GexRow, SnapshotRow } from './data-loader';

export class HUDState {
    private gexData: GexRow[] = [];
    private spotData: SnapshotRow[] = [];
    private physicsData: any[] = [];
    private spotRef: number = 6000;
    private timeRange: { start: bigint; end: bigint } = { start: 0n, end: 0n };
    private priceRange: { min: number; max: number } = { min: 5900, max: 6100 };
    private readonly MAX_HISTORY = 2000; // Keep ~30 mins of history

    setGexData(data: GexRow[]): void {
        this.gexData = [...this.gexData, ...data];
        if (this.gexData.length > this.MAX_HISTORY) {
            this.gexData = this.gexData.slice(this.gexData.length - this.MAX_HISTORY);
        }
        this.updateRanges();
    }

    setSpotData(data: SnapshotRow[]): void {
        this.spotData = [...this.spotData, ...data];
        if (this.spotData.length > this.MAX_HISTORY) {
            this.spotData = this.spotData.slice(this.spotData.length - this.MAX_HISTORY);
        }
        this.updateRanges();
    }

    setPhysicsData(data: any[]): void {
        this.physicsData = [...this.physicsData, ...data];
        if (this.physicsData.length > this.MAX_HISTORY) {
            this.physicsData = this.physicsData.slice(this.physicsData.length - this.MAX_HISTORY);
        }
        this.updateRanges();
    }

    private updateRanges(): void {
        const timestamps: bigint[] = [];
        const spots: number[] = [];
        const strikes: number[] = [];

        // Collect from GEX
        for (const r of this.gexData) {
            timestamps.push(r.window_end_ts_ns);
            if (r.strike_points) strikes.push(Number(r.strike_points));
            if (r.underlying_spot_ref) spots.push(Number(r.underlying_spot_ref));
        }

        // Collect from Spot
        for (const r of this.spotData) {
            timestamps.push(r.window_end_ts_ns);
            if (r.mid_price) spots.push(Number(r.mid_price));
        }

        // Collect from Physics
        for (const r of this.physicsData) {
            timestamps.push(r.window_end_ts_ns);
            if (r.mid_price) spots.push(Number(r.mid_price));
        }

        if (timestamps.length > 0) {
            let minTs = timestamps[0];
            let maxTs = timestamps[0];
            for (const ts of timestamps) {
                if (ts < minTs) minTs = ts;
                if (ts > maxTs) maxTs = ts;
            }
            this.timeRange = { start: minTs, end: maxTs };
        }

        if (spots.length > 0) {
            this.spotRef = spots[spots.length - 1];
        }

        let minPrice = this.priceRange.min;
        let maxPrice = this.priceRange.max;
        const allPrices = [...strikes, ...spots];

        if (allPrices.length > 0) {
            minPrice = allPrices[0];
            maxPrice = allPrices[0];
            for (const p of allPrices) {
                if (p < minPrice) minPrice = p;
                if (p > maxPrice) maxPrice = p;
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

    getSpotData(): SnapshotRow[] {
        return this.spotData;
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
        const set = new Set<bigint>();
        this.gexData.forEach(r => set.add(r.window_end_ts_ns));
        this.spotData.forEach(r => set.add(r.window_end_ts_ns));
        // Sort requires conversion to something subtractable or custom sort
        // BigInt subtraction works
        return Array.from(set).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    }

    // Get Spots by Time
    getSpotsByTime(): Map<bigint, number> {
        const map = new Map<bigint, number>();

        // 1. Physics
        for (const r of this.physicsData) {
            const p = Number(r.mid_price);
            if (!isNaN(p) && p > 100) map.set(r.window_end_ts_ns, p);
        }

        // 2. GEX
        for (const r of this.gexData) {
            if (r.underlying_spot_ref && Number(r.underlying_spot_ref) > 100) {
                map.set(r.window_end_ts_ns, Number(r.underlying_spot_ref));
            }
        }

        // 3. Snap (authoritative)
        for (const r of this.spotData) {
            const p = Number(r.mid_price);
            if (!isNaN(p) && p > 100) map.set(r.window_end_ts_ns, p);
        }
        return map;
    }
}
