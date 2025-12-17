import { Injectable, effect, inject, signal } from '@angular/core';
import { DataStreamService, FlowMap, FlowMetrics } from './data-stream.service';

export type FlowMetric = 'premium' | 'delta' | 'gamma';
export type DerivativeOrder = 1 | 2 | 3;
export type TimescaleMs = 1000 | 5000 | 30000;
export type BucketKey = 'call_above' | 'call_below' | 'put_above' | 'put_below';

export const TIMESCALES_MS: readonly TimescaleMs[] = [1000, 5000, 30000] as const;
export const TIMESCALE_LABEL: Record<TimescaleMs, string> = {
    1000: '1s',
    5000: '5s',
    30000: '30s'
};

export interface BucketTriple {
    premium: number;
    delta: number;
    gamma: number;
}

export interface LatestByBucket {
    call_above: number;
    call_below: number;
    put_above: number;
    put_below: number;
    net: number;
}

export interface StrikeSideVelocity {
    premium_vel: number;
    delta_vel: number;
    gamma_vel: number;
}

export interface StrikeVelocityRow {
    strike: number;
    call?: StrikeSideVelocity;
    put?: StrikeSideVelocity;
}

export interface StrikeVelocityMap {
    [strike: number]: StrikeVelocityRow;
}

type DerivativeKey = `${BucketKey}|${FlowMetric}|${DerivativeOrder}|${TimescaleMs}`;

interface AggregatedCum {
    t: number; // ms
    atmStrike: number | null;
    cum: Record<BucketKey, BucketTriple>;
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

function median(values: number[]): number | null {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 1) return sorted[mid];
    return (sorted[mid - 1] + sorted[mid]) / 2;
}

function safeDelta(now: number, prev: number): number {
    const d = now - prev;
    // If the backend restarted, cumulative can jump backwards.
    if (!Number.isFinite(d) || d < 0) return 0;
    return d;
}

function getBucket(type: string, strike: number, atmStrike: number): BucketKey | null {
    // Treat ATM as neutral (excluded from above/below buckets).
    if (strike === atmStrike) return null;
    if (type === 'C') return strike > atmStrike ? 'call_above' : 'call_below';
    if (type === 'P') return strike > atmStrike ? 'put_above' : 'put_below';
    return null;
}

@Injectable({ providedIn: 'root' })
export class FlowAnalyticsService {
    private stream = inject(DataStreamService);

    // UI selectors
    public selectedMetric = signal<FlowMetric>('delta');
    public selectedOrder = signal<DerivativeOrder>(2);
    public selectedTimescaleMs = signal<TimescaleMs>(5000);

    private _tick = signal(0);
    private _atmStrike = signal<number | null>(null);
    public atmStrike = this._atmStrike.asReadonly();

    private _perStrikeVel = signal<StrikeVelocityMap>({});
    public perStrikeVel = this._perStrikeVel.asReadonly();

    private prevSnapshot: FlowMap | null = null;
    private prevTs = 0;

    private history: AggregatedCum[] = [];
    private readonly maxHistoryMs = 120_000; // 2 minutes
    private readonly maxSeriesPoints = 260; // ~65s @ 250ms

    private series = new Map<DerivativeKey, number[]>();
    private latest = new Map<DerivativeKey, number>();

    // Store previous (per-update) derived values to compute accel/jerk
    private prevVel = new Map<DerivativeKey, number>();
    private prevAccel = new Map<DerivativeKey, number>();

    constructor() {
        effect(() => {
            const snapshot = this.stream.flowData();
            this.ingestSnapshot(snapshot);
        });
    }

    public getSeries(bucket: BucketKey): number[] {
        // Depend on tick + selectors so Angular updates consumers.
        this._tick();
        const key = this.makeKey(bucket, this.selectedMetric(), this.selectedOrder(), this.selectedTimescaleMs());
        return [...(this.series.get(key) ?? [])];
    }

    public getLatestByBucket(): LatestByBucket {
        this._tick();
        const metric = this.selectedMetric();
        const order = this.selectedOrder();
        const ts = this.selectedTimescaleMs();
        const callAbove = this.latest.get(this.makeKey('call_above', metric, order, ts)) ?? 0;
        const callBelow = this.latest.get(this.makeKey('call_below', metric, order, ts)) ?? 0;
        const putAbove = this.latest.get(this.makeKey('put_above', metric, order, ts)) ?? 0;
        const putBelow = this.latest.get(this.makeKey('put_below', metric, order, ts)) ?? 0;
        return {
            call_above: callAbove,
            call_below: callBelow,
            put_above: putAbove,
            put_below: putBelow,
            net: callAbove + callBelow + putAbove + putBelow
        };
    }

    public getLatestAbsMax(): number {
        const latest = this.getLatestByBucket();
        return Math.max(
            1,
            Math.abs(latest.call_above),
            Math.abs(latest.call_below),
            Math.abs(latest.put_above),
            Math.abs(latest.put_below),
            Math.abs(latest.net)
        );
    }

    private makeKey(bucket: BucketKey, metric: FlowMetric, order: DerivativeOrder, timescaleMs: TimescaleMs): DerivativeKey {
        return `${bucket}|${metric}|${order}|${timescaleMs}`;
    }

    private ingestSnapshot(snapshot: FlowMap) {
        const now = Date.now();

        // Establish ATM strike from the current strike set (median strike).
        const strikes = Object.values(snapshot)
            .map((m) => m.strike_price)
            .filter((s) => Number.isFinite(s) && s > 0);
        const atmStrike = median(Array.from(new Set(strikes)));
        this._atmStrike.set(atmStrike);

        // Build per-strike instantaneous velocity map (dt based on websocket cadence).
        if (this.prevSnapshot && this.prevTs > 0) {
            const dt = (now - this.prevTs) / 1000;
            if (dt > 0 && dt < 10) {
                this._perStrikeVel.set(this.computePerStrikeVelocity(snapshot, this.prevSnapshot, dt));
            }
        }

        // Aggregated cumulative state for multi-timescale slopes.
        if (atmStrike != null) {
            const aggregated = this.aggregateCumulative(snapshot, atmStrike, now);
            this.history.push(aggregated);
            this.pruneHistory(now);
            this.updateDerivedSeries(now);
        }

        this.prevSnapshot = snapshot;
        this.prevTs = now;
        this._tick.update((v) => v + 1);
    }

    private computePerStrikeVelocity(nowSnap: FlowMap, prevSnap: FlowMap, dtSeconds: number): StrikeVelocityMap {
        const map: StrikeVelocityMap = {};

        for (const ticker of Object.keys(nowSnap)) {
            const nowM = nowSnap[ticker];
            const prevM = prevSnap[ticker];
            if (!nowM || !prevM) continue;
            const strike = nowM.strike_price;
            if (!Number.isFinite(strike) || strike <= 0) continue;

            const dPremium = safeDelta(nowM.cumulative_premium, prevM.cumulative_premium);
            const dDelta = safeDelta(nowM.net_delta_flow, prevM.net_delta_flow);
            const dGamma = safeDelta(nowM.net_gamma_flow, prevM.net_gamma_flow);

            const vel: StrikeSideVelocity = {
                premium_vel: dPremium / dtSeconds,
                delta_vel: dDelta / dtSeconds,
                gamma_vel: dGamma / dtSeconds
            };

            const key = strike;
            if (!map[key]) {
                map[key] = { strike: key };
            }

            if (nowM.type === 'C') map[key].call = vel;
            if (nowM.type === 'P') map[key].put = vel;
        }

        return map;
    }

    private aggregateCumulative(snapshot: FlowMap, atmStrike: number, now: number): AggregatedCum {
        const zero: BucketTriple = { premium: 0, delta: 0, gamma: 0 };
        const cum: Record<BucketKey, BucketTriple> = {
            call_above: { ...zero },
            call_below: { ...zero },
            put_above: { ...zero },
            put_below: { ...zero }
        };

        for (const m of Object.values(snapshot)) {
            if (!m) continue;
            const strike = m.strike_price;
            if (!Number.isFinite(strike) || strike <= 0) continue;

            const bucket = getBucket(m.type, strike, atmStrike);
            if (!bucket) continue;

            cum[bucket].premium += m.cumulative_premium;
            cum[bucket].delta += m.net_delta_flow;
            cum[bucket].gamma += m.net_gamma_flow;
        }

        return { t: now, atmStrike, cum };
    }

    private pruneHistory(now: number) {
        const minT = now - this.maxHistoryMs;
        while (this.history.length && this.history[0].t < minT) {
            this.history.shift();
        }
    }

    private findLookbackSample(now: number, timescaleMs: TimescaleMs): AggregatedCum | null {
        const target = now - timescaleMs;
        // Find the first sample with t <= target (search backwards).
        for (let i = this.history.length - 1; i >= 0; i--) {
            const s = this.history[i];
            if (s.t <= target) return s;
        }
        return null;
    }

    private updateDerivedSeries(now: number) {
        if (this.history.length < 2) return;
        const latest = this.history[this.history.length - 1];

        const dtUpdate = (latest.t - this.history[this.history.length - 2].t) / 1000;
        if (!(dtUpdate > 0 && dtUpdate < 10)) return;

        for (const ts of TIMESCALES_MS) {
            const lookback = this.findLookbackSample(now, ts);
            if (!lookback) continue;

            const dtWindow = (latest.t - lookback.t) / 1000;
            if (!(dtWindow > 0.05)) continue;

            for (const bucket of Object.keys(latest.cum) as BucketKey[]) {
                const nowCum = latest.cum[bucket];
                const oldCum = lookback.cum[bucket];

                this.appendDerived(bucket, 'premium', ts, (nowCum.premium - oldCum.premium) / dtWindow, dtUpdate);
                this.appendDerived(bucket, 'delta', ts, (nowCum.delta - oldCum.delta) / dtWindow, dtUpdate);
                this.appendDerived(bucket, 'gamma', ts, (nowCum.gamma - oldCum.gamma) / dtWindow, dtUpdate);
            }
        }
    }

    private appendDerived(bucket: BucketKey, metric: FlowMetric, ts: TimescaleMs, velocity: number, dtUpdate: number) {
        // Order 1: velocity
        const keyVel = this.makeKey(bucket, metric, 1, ts);
        const prevVel = this.prevVel.get(keyVel) ?? 0;
        this.prevVel.set(keyVel, velocity);

        // Order 2: acceleration (Δvelocity / Δt)
        const accel = (velocity - prevVel) / dtUpdate;
        const keyAccel = this.makeKey(bucket, metric, 2, ts);
        const prevAccel = this.prevAccel.get(keyAccel) ?? 0;
        this.prevAccel.set(keyAccel, accel);

        // Order 3: jerk (Δaccel / Δt)
        const jerk = (accel - prevAccel) / dtUpdate;
        const keyJerk = this.makeKey(bucket, metric, 3, ts);

        this.pushSeries(keyVel, velocity);
        this.pushSeries(keyAccel, accel);
        this.pushSeries(keyJerk, jerk);

        this.latest.set(keyVel, velocity);
        this.latest.set(keyAccel, accel);
        this.latest.set(keyJerk, jerk);
    }

    private pushSeries(key: DerivativeKey, value: number) {
        const existing = this.series.get(key) ?? [];
        existing.push(value);
        if (existing.length > this.maxSeriesPoints) {
            existing.splice(0, existing.length - this.maxSeriesPoints);
        }
        this.series.set(key, existing);
    }
}


