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
    [key: string]: any; // Allow other fields from Arrow
}

export interface SnapshotRow {
    window_end_ts_ns: bigint;
    price: number;
    [key: string]: any;
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

    connectStream(
        symbol: string,
        dt: string,
        onBatch: (surfaces: Record<string, any[]>) => void
    ): void {
        const wsUrl = `ws://localhost:8000/v1/hud/stream?symbol=${symbol}&dt=${dt}`;
        console.log(`Connecting to stream: ${wsUrl}`);
        const ws = new WebSocket(wsUrl);

        let currentBatch: Record<string, GexRow[]> = {};
        const messageQueue: MessageEvent[] = [];
        let isProcessing = false;

        const processQueue = async () => {
            if (isProcessing) return;
            isProcessing = true;

            while (messageQueue.length > 0) {
                const event = messageQueue.shift();
                if (!event) continue;

                try {
                    if (typeof event.data === 'string') {
                        // Parse JSON message
                        const msg = JSON.parse(event.data);

                        if (msg.type === 'batch_start') {
                            currentBatch = {};
                        } else if (msg.type === 'surface_header') {
                            // Store surface context for next binary
                            (this as any)._pendingSurface = msg.surface;
                        }
                    } else {
                        // Binary message
                        const surface = (this as any)._pendingSurface;
                        if (surface && event.data instanceof Blob) {
                            const buffer = await event.data.arrayBuffer();
                            const table = tableFromIPC(buffer);
                            const rows = this.tableToRows(table);
                            console.log(`[DataLoader] Parsed ${surface}: ${rows.length} rows`);

                            currentBatch[surface] = rows;
                            (this as any)._pendingSurface = null;

                            // Emit immediately
                            onBatch(currentBatch);
                        }
                    }
                } catch (e) {
                    console.error("Error processing message", e);
                }
            }

            isProcessing = false;
        };

        ws.onopen = () => console.log("WebSocket connected");

        ws.onmessage = (event) => {
            messageQueue.push(event);
            processQueue();
        };

        ws.onerror = (err) => console.error("WebSocket error:", err);
        ws.onclose = () => console.log("WebSocket closed");
    }

    private tableToRows(table: Table): GexRow[] {
        const rows: GexRow[] = [];

        for (let i = 0; i < table.numRows; i++) {
            const row = table.get(i);
            if (row) {
                // We should genericize this or ensure GexRow covers all needed fields
                // For now, mapping blindly or sticking to known GEX structure.
                // The renderer expects specific fields.
                // Let's assume the row object is compatible or spread it.
                // NOTE: Arrow JS objects are proxies. Spreading ...row might not work fully as expected without toJSON() or strict mapping.
                // But previously tableToRows strictly mapped GEX fields. 
                // We need to support ALL surfaces now.
                // Let's return the row struct as-is (Proxy) or a simple dict.
                // To support "radar" and "vacuum" fields, we should probably allow `any`.

                // Let's use a more flexible mapping for this demo since we have many surfaces.
                rows.push(row.toJSON() as unknown as GexRow);
            }
        }

        return rows;
    }
}
