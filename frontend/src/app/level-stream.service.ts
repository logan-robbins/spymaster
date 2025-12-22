import { Injectable, signal, WritableSignal } from '@angular/core';

/**
 * Level signal payload structure per PLAN.md §6.4
 */

export interface SpySnapshot {
    spot: number;
    bid: number;
    ask: number;
}

export interface BarrierMetrics {
    state: 'VACUUM' | 'WALL' | 'ABSORPTION' | 'CONSUMED' | 'WEAK' | 'NEUTRAL';
    delta_liq: number;
    replenishment_ratio: number;
    added: number;
    canceled: number;
    filled: number;
}

export interface TapeMetrics {
    imbalance: number;  // [-1, +1]
    buy_vol: number;
    sell_vol: number;
    velocity: number;  // $/sec
    sweep: {
        detected: boolean;
        direction?: 'UP' | 'DOWN';
        notional?: number;
        window_ms?: number;
        venues?: number;
    };
}

export interface FuelMetrics {
    effect: 'AMPLIFY' | 'DAMPEN' | 'NEUTRAL';
    net_dealer_gamma: number;
    call_wall?: number;
    put_wall?: number;
    hvl?: number;
}

export interface RunwayMetrics {
    direction: 'UP' | 'DOWN';
    next_obstacle: {
        id: string;
        price: number;
    } | null;
    distance: number;
    quality: 'CLEAR' | 'OBSTRUCTED';
}

export interface LevelSignal {
    id: string;
    price: number;
    kind: 'VWAP' | 'STRIKE' | 'ROUND' | 'SESSION_HIGH' | 'SESSION_LOW' | 'GAMMA_WALL' | 'USER';
    direction: 'SUPPORT' | 'RESISTANCE';
    distance: number;
    break_score_raw: number;
    break_score_smooth: number;
    signal: 'BREAK' | 'REJECT' | 'CONTESTED' | 'NEUTRAL';
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
    barrier: BarrierMetrics;
    tape: TapeMetrics;
    fuel: FuelMetrics;
    runway: RunwayMetrics;
    note?: string;
}

export interface LevelsPayload {
    ts: number;  // Unix ms
    spy: SpySnapshot;
    levels: LevelSignal[];
}

@Injectable({
    providedIn: 'root'
})
export class LevelStreamService {
    public levelsData: WritableSignal<LevelsPayload | null> = signal(null);
    public connectionStatus: WritableSignal<'connecting' | 'connected' | 'disconnected'> = signal('disconnected');
    
    private socket?: WebSocket;
    private readonly URL = 'ws://localhost:8000/ws/levels';
    private reconnectTimeout?: number;
    private reconnectDelay = 3000;

    constructor() {
        this.connect();
    }

    private connect() {
        this.connectionStatus.set('connecting');
        console.log('🔌 Connecting to Levels Stream...');
        
        try {
            this.socket = new WebSocket(this.URL);

            this.socket.onopen = () => {
                console.log('✅ Connected to Levels Stream');
                this.connectionStatus.set('connected');
                this.reconnectDelay = 3000; // Reset backoff
            };

            this.socket.onmessage = (event) => {
                try {
                    const data: LevelsPayload = JSON.parse(event.data);
                    this.levelsData.set(data);
                } catch (err) {
                    console.error('Error parsing levels data:', err);
                }
            };

            this.socket.onclose = () => {
                console.warn('⚠️ Levels Stream Disconnected. Reconnecting...');
                this.connectionStatus.set('disconnected');
                this.scheduleReconnect();
            };

            this.socket.onerror = (err) => {
                console.error('Levels WebSocket Error:', err);
                this.connectionStatus.set('disconnected');
            };
        } catch (err) {
            console.error('Failed to create WebSocket:', err);
            this.connectionStatus.set('disconnected');
            this.scheduleReconnect();
        }
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

