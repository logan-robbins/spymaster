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
    price: number;
    kind: string;
    direction: string;
    distance: number;
    break_score_raw: number;
    break_score_smooth: number | null;
    signal: string;
    confidence: string;
    barrier: any;
    tape: any;
    fuel: any;
    runway: any;
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

@Injectable({
    providedIn: 'root'
})
export class DataStreamService {
    public flowData: WritableSignal<FlowMap> = signal({});
    public levelsData: WritableSignal<LevelsPayload | null> = signal(null);
    public connectionStatus: WritableSignal<'connecting' | 'connected' | 'disconnected'> = signal('disconnected');
    
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
                const data: MergedPayload = JSON.parse(event.data);
                
                // Update flow data (existing functionality)
                if (data.flow) {
                    this.flowData.set(data.flow);
                }
                
                // Update levels data (new functionality)
                if (data.levels) {
                    this.levelsData.set(data.levels);
                }
            } catch (err) {
                console.error('Error parsing stream data:', err);
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
