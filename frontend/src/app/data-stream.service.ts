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

// ML Viewport Predictions (from backend ML models)
export interface TimeToThreshold {
    t1: Record<string, number>;  // {60: prob, 120: prob}
    t2: Record<string, number>;
}

export interface RetrievalSummary {
    p_break: number;
    p_bounce: number;
    p_tradeable_2: number;
    strength_signed_mean: number;
    strength_abs_mean: number;
    time_to_threshold_1_mean: number;
    time_to_threshold_2_mean: number;
    similarity: number;
    entropy: number;
    neighbors: any[];
}

export interface ViewportTarget {
    level_id: string;
    level_kind_name: string;
    level_price: number;
    direction: 'UP' | 'DOWN';
    distance: number;
    distance_signed: number;
    
    // Tree model predictions
    p_tradeable_2: number;
    p_break: number;
    p_bounce: number;
    strength_signed: number;
    strength_abs: number;
    time_to_threshold: TimeToThreshold;
    
    // Retrieval predictions
    retrieval: RetrievalSummary;
    
    // Scoring metadata
    utility_score: number;
    viewport_state: string;
    stage: string;
    pinned: boolean;
    relevance: number;
}

export interface ViewportPayload {
    ts: number;
    targets: ViewportTarget[];
}

// Merged payload from backend (Option A per §6.4)
export interface MergedPayload {
    flow: FlowMap;
    levels: LevelsPayload;
    viewport?: ViewportPayload;  // Optional ML predictions
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isLevelsPayload(value: unknown): value is LevelsPayload {
    if (!isRecord(value)) return false;
    const v: any = value;
    return typeof v.ts === 'number' && isRecord(v.spy) && Array.isArray(v.levels);
}

function isViewportPayload(value: unknown): value is ViewportPayload {
    if (!isRecord(value)) return false;
    const v: any = value;
    return typeof v.ts === 'number' && Array.isArray(v.targets);
}

@Injectable({
    providedIn: 'root'
})
export class DataStreamService {
    public flowData: WritableSignal<FlowMap> = signal({});
    public levelsData: WritableSignal<LevelsPayload | null> = signal(null);
    public viewportData: WritableSignal<ViewportPayload | null> = signal(null);
    public connectionStatus: WritableSignal<'connecting' | 'connected' | 'disconnected'> = signal('disconnected');
    public dataStatus: WritableSignal<'ok' | 'unavailable'> = signal('unavailable');
    public lastError: WritableSignal<string | null> = signal(null);
    
    private socket!: WebSocket;
    private readonly URL = 'ws://localhost:8000/ws/stream';
    private reconnectTimeout?: number;
    private reconnectDelay = 3000;
    private mockInterval?: number;

    constructor() {
        // Check for mock mode via query param
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('mock')) {
            this.startMockStream();
        } else {
            this.connect();
        }
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
                
                // Update viewport data (ML predictions)
                const viewport = data['viewport'];
                if (isViewportPayload(viewport)) {
                    this.viewportData.set(viewport);
                } else if (viewport !== undefined) {
                    console.warn('Invalid viewport payload structure');
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
        if (this.mockInterval) {
            clearInterval(this.mockInterval);
        }
        if (this.socket) {
            this.socket.close();
        }
    }

    // --- MOCK MODE IMPLEMENTATION ---
    private startMockStream() {
        console.log('🎭 MOCK MODE ACTIVATED');
        this.connectionStatus.set('connected');
        this.dataStatus.set('ok');
        
        // Initial Base Price
        let spot = 585.12;
        let vwap = 584.80;
        let tick = 0;

        // Helper to format ticker: SPY + YYMMDD + C/P + 00000.000 padded
        // e.g. SPY251223C00585000
        const formatTicker = (date: string, type: 'C'|'P', strike: number) => {
            const strikeStr = (strike * 1000).toString().padStart(8, '0');
            return `SPY${date}${type}${strikeStr}`;
        };

        this.mockInterval = window.setInterval(() => {
            tick++;
            // Wander spot price slightly
            const drift = (Math.random() - 0.5) * 0.05;
            spot += drift;

            // Drift VWAP slowly (Dynamic Level)
            vwap += (Math.random() - 0.5) * 0.015;

            // Generate Mock Levels
            const levels: LevelSignal[] = [
                {
                    id: 'PM_HIGH',
                    level_price: 585.50,
                    level_kind_name: 'PM_HIGH',
                    direction: 'UP',
                    distance: 585.50 - spot,
                    is_first_15m: false,
                    barrier_state: 'WALL',
                    barrier_delta_liq: 250.0 + Math.random() * 50,
                    barrier_replenishment_ratio: 2.1,
                    wall_ratio: 1.8,
                    tape_imbalance: 0.4,
                    tape_velocity: 35.0,
                    tape_buy_vol: 800,
                    tape_sell_vol: 500,
                    sweep_detected: false,
                    gamma_exposure: -15000,
                    fuel_effect: 'DAMPEN',
                    approach_velocity: 0.1,
                    approach_bars: 5,
                    approach_distance: 0.38,
                    prior_touches: 2,
                    bars_since_open: 45,
                    break_score_raw: 35,
                    break_score_smooth: 32,
                    signal: 'BOUNCE',
                    confidence: 'HIGH'
                },
                {
                    id: 'VWAP',
                    level_price: vwap,
                    level_kind_name: 'VWAP',
                    direction: 'DOWN',
                    distance: spot - vwap,
                    is_first_15m: false,
                    barrier_state: 'VACUUM',
                    barrier_delta_liq: -120.0,
                    barrier_replenishment_ratio: 0.5,
                    wall_ratio: 0.4,
                    tape_imbalance: -0.8, // Heavy selling
                    tape_velocity: -85.0,  // Fast selling
                    tape_buy_vol: 200,
                    tape_sell_vol: 1500,
                    sweep_detected: true, // Sweep!
                    gamma_exposure: 5000,
                    fuel_effect: 'AMPLIFY',
                    approach_velocity: -0.25,
                    approach_bars: 2,
                    approach_distance: 0.32,
                    prior_touches: 0,
                    bars_since_open: 45,
                    break_score_raw: 78,
                    break_score_smooth: 75,
                    signal: 'BREAK',
                    confidence: 'MEDIUM'
                },
                {
                   id: 'CALL_WALL',
                   level_price: 586.00,
                   level_kind_name: 'CALL_WALL',
                   direction: 'UP',
                   distance: 586.00 - spot,
                   is_first_15m: false,
                   barrier_state: 'ABSORPTION',
                   barrier_delta_liq: 100.0,
                   barrier_replenishment_ratio: 1.2,
                   wall_ratio: 1.5,
                   tape_imbalance: 0.2,
                   tape_velocity: 30.0,
                   tape_buy_vol: 1200,
                   tape_sell_vol: 900,
                   sweep_detected: false,
                   gamma_exposure: -50000, // Big negative gamma
                   fuel_effect: 'DAMPEN',
                   approach_velocity: 0.05,
                   approach_bars: 10,
                   approach_distance: 0.88,
                   prior_touches: 1,
                   bars_since_open: 45,
                   break_score_raw: 45,
                   break_score_smooth: 42,
                   signal: 'CHOP',
                   confidence: 'LOW'
               }
            ];

            const payload: LevelsPayload = {
                ts: Date.now(),
                spy: {
                    spot: spot,
                    bid: spot - 0.01,
                    ask: spot + 0.01
                },
                levels: levels
            };

            this.levelsData.set(payload);

            // Mock Flow Data for 11 strikes (Center +/- 5)
            const centerStrike = Math.round(spot);
            const flow: FlowMap = {};
            
            for (let i = -5; i <= 5; i++) {
                const strike = centerStrike + i;
                const distFromAtm = Math.abs(i);
                
                // Volume decays as we move away from ATM
                const volumeBase = Math.max(100, 5000 - distFromAtm * 800);
                const premBase = volumeBase * (2.0 - distFromAtm * 0.2); // Premium approx
                const gammaShape = Math.max(0.02, 0.1 - (distFromAtm * 0.01)); // Peak near ATM, taper OTM

                // Mock "Net Gamma" profile (proxy for GEX shape). Crosses near call wall (i=+1).
                const netGammaStrike = (-(i - 1)) * 100000 * gammaShape + (Math.random() - 0.5) * 1500;
                const callGammaFlow = netGammaStrike * 0.7;
                const putGammaFlow = netGammaStrike - callGammaFlow;
                
                // Calls
                const callTicker = formatTicker('251223', 'C', strike);
                flow[callTicker] = {
                    cumulative_volume: volumeBase + Math.floor(Math.random() * 50) + tick * 5,
                    cumulative_premium: premBase * 100 + tick * 200,
                    last_price: Math.max(0.01, 2.0 - i * 0.3) + Math.random() * 0.05,
                    net_delta_flow: (volumeBase * 0.4) + tick * 10,
                    net_gamma_flow: callGammaFlow,
                    delta: 0.5 + (i * -0.05), // Delta drops as strike increases
                    gamma: gammaShape,
                    strike_price: strike,
                    type: 'C',
                    expiration: '2025-12-23',
                    last_timestamp: Date.now()
                };

                // Puts
                const putTicker = formatTicker('251223', 'P', strike);
                flow[putTicker] = {
                    cumulative_volume: (volumeBase * 0.8) + Math.floor(Math.random() * 50) + tick * 4,
                    cumulative_premium: (premBase * 0.8) * 100 + tick * 150,
                    last_price: Math.max(0.01, 2.0 + i * 0.3) + Math.random() * 0.05,
                    net_delta_flow: -(volumeBase * 0.3) - tick * 10,
                    net_gamma_flow: putGammaFlow,
                    delta: -0.5 + (i * 0.05), // Delta increases (less negative) as strike increases
                    gamma: gammaShape,
                    strike_price: strike,
                    type: 'P',
                    expiration: '2025-12-23',
                    last_timestamp: Date.now()
                };
            }

            this.flowData.set(flow);

            // Mock Viewport Data (ML Predictions)
            const viewportTargets: ViewportTarget[] = levels.map((level, idx) => {
                // Generate realistic ML predictions based on physics
                const physics_break_bias = level.break_score_smooth > 50 ? 0.15 : -0.15;
                const base_tradeable = 0.45 + (Math.random() * 0.3);
                const base_break = 0.50 + physics_break_bias + (Math.random() * 0.2 - 0.1);
                
                return {
                    level_id: level.id,
                    level_kind_name: level.level_kind_name,
                    level_price: level.level_price,
                    direction: level.direction,
                    distance: level.distance,
                    distance_signed: level.direction === 'UP' ? level.distance : -level.distance,
                    
                    // Tree model predictions
                    p_tradeable_2: base_tradeable + (idx === 0 ? 0.15 : 0),  // Boost nearest level
                    p_break: Math.max(0, Math.min(1, base_break)),
                    p_bounce: Math.max(0, Math.min(1, 1 - base_break)),
                    strength_signed: (base_break - 0.5) * 2.5,  // -1.25 to +1.25
                    strength_abs: 0.8 + Math.random() * 0.8,
                    time_to_threshold: {
                        t1: {
                            '60': 0.25 + Math.random() * 0.4,   // 25-65% chance in 60s
                            '120': 0.40 + Math.random() * 0.35  // 40-75% chance in 120s
                        },
                        t2: {
                            '60': 0.10 + Math.random() * 0.25,  // 10-35% chance in 60s
                            '120': 0.20 + Math.random() * 0.35  // 20-55% chance in 120s
                        }
                    },
                    
                    // Retrieval predictions (kNN)
                    retrieval: {
                        p_break: Math.max(0, Math.min(1, base_break + (Math.random() * 0.1 - 0.05))),
                        p_bounce: Math.max(0, Math.min(1, 1 - base_break + (Math.random() * 0.1 - 0.05))),
                        p_tradeable_2: base_tradeable + (Math.random() * 0.1 - 0.05),
                        strength_signed_mean: (base_break - 0.5) * 2.0,
                        strength_abs_mean: 0.9 + Math.random() * 0.6,
                        time_to_threshold_1_mean: 80 + Math.random() * 60,  // 80-140s
                        time_to_threshold_2_mean: 150 + Math.random() * 100, // 150-250s
                        similarity: 0.65 + Math.random() * 0.25,  // 65-90% similar
                        entropy: 0.3 + Math.random() * 0.4,
                        neighbors: []
                    },
                    
                    // Scoring metadata
                    utility_score: base_tradeable * (Math.abs(base_break - 0.5) * 2),
                    viewport_state: level.distance <= 0.25 ? 'IN_MONITOR_BAND' : 'OUTSIDE_BAND',
                    stage: idx === 0 ? 'stage_b' : 'stage_a',  // Nearest level gets stage_b
                    pinned: false,
                    relevance: 1.0 - (idx * 0.15)  // Decay with distance
                };
            });

            const viewport: ViewportPayload = {
                ts: Date.now(),
                targets: viewportTargets
            };

            this.viewportData.set(viewport);

        }, 250); // 4Hz update like real backend
    }
}
