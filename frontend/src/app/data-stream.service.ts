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
}

export interface FlowMap {
    [ticker: string]: FlowMetrics;
}

@Injectable({
    providedIn: 'root'
})
export class DataStreamService {
    public flowData: WritableSignal<FlowMap> = signal({});
    private socket!: WebSocket;
    private readonly URL = 'ws://localhost:8000/ws/stream';

    constructor() {
        this.connect();
    }

    private connect() {
        console.log('🔌 Connecting to 0DTE Stream...');
        this.socket = new WebSocket(this.URL);

        this.socket.onopen = () => {
            console.log('✅ Connected to Stream');
        };

        this.socket.onmessage = (event) => {
            try {
                const data: FlowMap = JSON.parse(event.data);
                // Direct signal update (Signals are performant)
                this.flowData.set(data);
            } catch (err) {
                console.error('Error parsing stream data:', err);
            }
        };

        this.socket.onclose = () => {
            console.warn('⚠️ Disconnected. Reconnecting in 3s...');
            setTimeout(() => this.connect(), 3000);
        };

        this.socket.onerror = (err) => {
            console.error('WebSocket Error:', err);
        };
    }
}
