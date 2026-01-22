/**
 * Data Loader - Fetches Arrow IPC data from the HUD API
 */

import { tableFromIPC, Table } from 'apache-arrow';

export interface GexRow {
    window_start_ts_ns: bigint;
    window_end_ts_ns: bigint;
    underlying: string;
    strike_price_int: bigint;
    underlying_spot_ref: number;
    strike_points: number;
    gex_call_abs: number;
    gex_put_abs: number;
    gex_abs: number;
    gex: number;
    gex_imbalance_ratio: number;
}

export class DataLoader {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    async loadSurface(symbol: string, dt: string, surface: string): Promise<GexRow[]> {
        const url = `${this.baseUrl}/v1/hud/surfaces?symbol=${symbol}&dt=${dt}&surface=${surface}`;

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load ${surface}: ${response.status}`);
        }

        const buffer = await response.arrayBuffer();
        const table = tableFromIPC(buffer);

        return this.tableToRows(table);
    }

    private tableToRows(table: Table): GexRow[] {
        const rows: GexRow[] = [];

        for (let i = 0; i < table.numRows; i++) {
            const row = table.get(i);
            if (row) {
                rows.push({
                    window_start_ts_ns: row.window_start_ts_ns,
                    window_end_ts_ns: row.window_end_ts_ns,
                    underlying: row.underlying,
                    strike_price_int: row.strike_price_int,
                    underlying_spot_ref: row.underlying_spot_ref,
                    strike_points: row.strike_points,
                    gex_call_abs: row.gex_call_abs,
                    gex_put_abs: row.gex_put_abs,
                    gex_abs: row.gex_abs,
                    gex: row.gex,
                    gex_imbalance_ratio: row.gex_imbalance_ratio,
                });
            }
        }

        return rows;
    }
}
