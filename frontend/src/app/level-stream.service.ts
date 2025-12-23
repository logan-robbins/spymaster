import { Injectable, signal, WritableSignal } from '@angular/core';

/**
 * Level signal payload structure per PLAN.md §6.4
 */

export interface SpySnapshot {
    spot: number;
    bid: number;
    ask: number;
}

export interface LevelSignal {
    id: string;
    level_price: number;
    level_kind_name: 'PM_HIGH' | 'PM_LOW' | 'OR_HIGH' | 'OR_LOW' | 'SESSION_HIGH' | 'SESSION_LOW' | 'SMA_200' | 'SMA_400' | 'VWAP' | 'ROUND' | 'STRIKE' | 'CALL_WALL' | 'PUT_WALL';
    direction: 'UP' | 'DOWN';
    distance: number;
    is_first_15m: boolean;
    barrier_state: 'VACUUM' | 'WALL' | 'ABSORPTION' | 'CONSUMED' | 'WEAK' | 'NEUTRAL';
    barrier_delta_liq: number;
    barrier_replenishment_ratio: number;
    wall_ratio: number;
    tape_imbalance: number;
    tape_velocity: number;
    tape_buy_vol: number;
    tape_sell_vol: number;
    sweep_detected: boolean;
    gamma_exposure: number;
    fuel_effect: 'AMPLIFY' | 'DAMPEN' | 'NEUTRAL';
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
