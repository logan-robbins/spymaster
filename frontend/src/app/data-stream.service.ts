import { Injectable, signal, WritableSignal } from '@angular/core';

export interface GreekMetrics {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
}

export interface TradeRecord {
    timestamp: string;
    ticker: string;
    price: number;
    size: number;
    premium: number;
    aggressor_side: number;
    delta: number;
    gamma: number;
    net_delta_impact: number;
}

// Map Ticker -> Record
// Actually FlowAggregator broadcasts "State Store" snapshot.
// The snapshot is Dict[Ticker, ContractMetrics]
// Schema from FlowAggregator:
/*
{
    "cumulative_volume": 0,
    "cumulative_premium": 0.0,
    "last_price": 0.0,
    "net_delta_flow": 0.0,
    "net_gamma_flow": 0.0,
    "delta": 0.0,
    "gamma": 0.0,
    "strike_price": 0.0,
    "type": "",  # 'C' or 'P'
    "expiration": ""
}
*/
export interface FlowMetrics {
    cumulative_volume: number;
    cumulative_premium: number;
    last_price: number;
    net_delta_flow: number;
    net_gamma_flow: number;
    delta: number;
    gamma: number;
    strike_price: number;
    type: string;
    expiration: string;
    last_timestamp: number; // ms Unix epoch
}

export interface FlowMap {
    [ticker: string]: FlowMetrics;
}

// Level signals payload (matches backend §6.4)
export interface SpySnapshot {
    spot: number | null;
    bid: number | null;
    ask: number | null;
}

export interface LevelSignal {
    id: string;
    level_price: number;
    level_kind_name: string;
    direction: 'UP' | 'DOWN';
    distance: number;
    is_first_15m: boolean;
    barrier_state: string;
    barrier_delta_liq: number;
    barrier_replenishment_ratio: number;
    wall_ratio: number;
    tape_imbalance: number;
    tape_velocity: number;
    tape_buy_vol: number;
    tape_sell_vol: number;
    sweep_detected: boolean;
    gamma_exposure: number;
    fuel_effect: string;
    approach_velocity: number;
    approach_bars: number;
    approach_distance: number;
    prior_touches: number;
    bars_since_open: number;
    break_score_raw: number;
    break_score_smooth: number;
    signal: 'BREAK' | 'BOUNCE' | 'CHOP';
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
    note?: string;
}

export interface LevelsPayload {
    ts: number;
    spy: SpySnapshot;
    levels: LevelSignal[];
}

// Merged payload from backend (Option A per §6.4)
export interface MergedPayload {
    flow: FlowMap;
    levels: LevelsPayload;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isLevelsPayload(value: unknown): value is LevelsPayload {
    if (!isRecord(value)) return false;
    const v: any = value;
    return typeof v.ts === 'number' && isRecord(v.spy) && Array.isArray(v.levels);
}

@Injectable({
    providedIn: 'root'
})
export class DataStreamService {
    public flowData: WritableSignal<FlowMap> = signal({});
    public levelsData: WritableSignal<LevelsPayload | null> = signal(null);
    public connectionStatus: WritableSignal<'connecting' | 'connected' | 'disconnected'> = signal('disconnected');
    public dataStatus: WritableSignal<'ok' | 'unavailable'> = signal('unavailable');
    public lastError: WritableSignal<string | null> = signal(null);
    
    private socket!: WebSocket;
    private readonly URL = 'ws://localhost:8000/ws/stream';
    private reconnectTimeout?: number;
    private reconnectDelay = 3000;

    constructor() {
        this.connect();
    }

    private connect() {
        this.connectionStatus.set('connecting');
        console.log('🔌 Connecting to 0DTE Stream...');
        this.socket = new WebSocket(this.URL);

        this.socket.onopen = () => {
            console.log('✅ Connected to Stream');
            this.connectionStatus.set('connected');
            this.reconnectDelay = 3000; // Reset backoff
        };

        this.socket.onmessage = (event) => {
            try {
                const parsed: unknown = JSON.parse(event.data);
                if (!isRecord(parsed)) {
                    return;
                }
                const data = parsed as Record<string, unknown>;
                
                // Update flow data (existing functionality)
                const flow = data['flow'];
                if (isRecord(flow)) {
                    this.flowData.set(flow as FlowMap);
                }
                
                // Update levels data (new functionality)
                const levels = data['levels'];
                if (isLevelsPayload(levels)) {
                    this.levelsData.set(levels);
                    this.dataStatus.set('ok');
                    this.lastError.set(null);
                } else if ('levels' in data) {
                    // Fail-soft: keep UI alive and surface a data-unavailable state.
                    this.levelsData.set(null);
                    this.dataStatus.set('unavailable');
                    this.lastError.set('Invalid levels payload');
                }
            } catch (err) {
                console.error('Error parsing stream data:', err);
                this.dataStatus.set('unavailable');
                this.lastError.set('Stream parse error');
            }
        };

        this.socket.onclose = () => {
            console.warn('⚠️ Disconnected. Reconnecting in 3s...');
            this.connectionStatus.set('disconnected');
            this.scheduleReconnect();
        };

        this.socket.onerror = (err) => {
            console.error('WebSocket Error:', err);
            this.connectionStatus.set('disconnected');
        };
    }

    private scheduleReconnect() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
        }
        
        this.reconnectTimeout = window.setTimeout(() => {
            this.connect();
            // Exponential backoff, max 30s
            this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 30000);
        }, this.reconnectDelay);
    }

    public disconnect() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
        }
        if (this.socket) {
            this.socket.close();
        }
    }
}
