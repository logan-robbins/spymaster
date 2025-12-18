import { Injectable, effect, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { DataStreamService, FlowMap, FlowMetrics } from './data-stream.service';

export type FlowMetric = 'premium' | 'delta' | 'gamma';
export type DerivativeOrder = 1 | 2 | 3;
export type BarIntervalMs = 5000 | 30000 | 60000;
export type BucketKey = 'call_above' | 'call_below' | 'put_above' | 'put_below';

export const BAR_INTERVALS_MS: readonly BarIntervalMs[] = [5000, 30000, 60000] as const;
export const BAR_INTERVAL_LABEL: Record<BarIntervalMs, string> = {
    5000: '5s',
    30000: '30s',
    60000: '60s'
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

type DerivativeKey = `${BucketKey}|${FlowMetric}|${DerivativeOrder}`;

interface BarData {
    barStartTime: number; // ms - start of bar interval
    barEndTime: number;   // ms - end of bar interval
    atmStrike: number | null;
    flow: Record<BucketKey, BucketTriple>; // Net flow during this bar (not cumulative)
}

interface ActiveBar {
    startTime: number;
    startCum: Record<BucketKey, BucketTriple>; // Cumulative values at bar open
    atmStrike: number | null;
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
    private http = inject(HttpClient);

    // UI selectors (defaults: DELTA, d2 acceleration, 5s bars)
    public selectedMetric = signal<FlowMetric>('delta');
    public selectedOrder = signal<DerivativeOrder>(2);
    public selectedBarInterval = signal<BarIntervalMs>(5000);

    private _tick = signal(0);
    private _atmStrike = signal<number | null>(null);
    public atmStrike = this._atmStrike.asReadonly();

    // New: Hotzone Anchor (User defined). If null, falls back to ATM.
    public anchorStrike = signal<number | null>(null);

    private _perStrikeVel = signal<StrikeVelocityMap>({});
    public perStrikeVel = this._perStrikeVel.asReadonly();

    private prevSnapshot: FlowMap | null = null;

    // Bar-based storage (one set of bars per interval)
    private bars = new Map<BarIntervalMs, BarData[]>();
    private activeBars = new Map<BarIntervalMs, ActiveBar | null>();

    private readonly maxBars = 50; // Keep last 50 bars per interval

    // Current derivative values for display
    private latest = new Map<DerivativeKey, number>();

    // Per-bucket bar series for the chart (for selected interval only)
    private barSeries = new Map<BucketKey, Array<{ time: number, value: number }>>();

    constructor() {
        // Initialize bar storage
        for (const interval of BAR_INTERVALS_MS) {
            this.bars.set(interval, []);
            this.activeBars.set(interval, null);
        }

        effect(() => {
            const snapshot = this.stream.flowData();
            this.ingestSnapshot(snapshot);
        });
    }

    public setAnchor(strike: number | null) {
        this.anchorStrike.set(strike);
        // Sync with backend
        this.http.post('/api/config/hotzone', { strike }).subscribe({
            next: (res) => console.log('Hotzone synced:', res),
            error: (err) => console.error('Hotzone sync failed:', err)
        });
    }

    public getSeries(bucket: BucketKey): Array<{ time: number, value: number }> {
        // Return bar series for this bucket
        this._tick();
        return [...(this.barSeries.get(bucket) ?? [])];
    }

    public getLatestByBucket(): LatestByBucket {
        this._tick();
        const metric = this.selectedMetric();
        const order = this.selectedOrder();
        const callAbove = this.latest.get(this.makeKey('call_above', metric, order)) ?? 0;
        const callBelow = this.latest.get(this.makeKey('call_below', metric, order)) ?? 0;
        const putAbove = this.latest.get(this.makeKey('put_above', metric, order)) ?? 0;
        const putBelow = this.latest.get(this.makeKey('put_below', metric, order)) ?? 0;
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

    private makeKey(bucket: BucketKey, metric: FlowMetric, order: DerivativeOrder): DerivativeKey {
        return `${bucket}|${metric}|${order}`;
    }

    private ingestSnapshot(snapshot: FlowMap) {
        // Use actual data timestamp from trades, not browser time (critical for replay)
        // Get timestamp from first ticker in snapshot, or fall back to Date.now()
        let dataTimestamp = Date.now();
        const firstTicker = Object.values(snapshot)[0];
        if (firstTicker && firstTicker.last_timestamp && firstTicker.last_timestamp > 0) {
            dataTimestamp = firstTicker.last_timestamp;
        }

        // Establish ATM strike from the current strike set (median strike)
        const strikes = Object.values(snapshot)
            .map((m) => m.strike_price)
            .filter((s) => Number.isFinite(s) && s > 0);
        const atmStrike = median(Array.from(new Set(strikes)));
        this._atmStrike.set(atmStrike);

        // Determine Pivot Strike: Anchor (Hotzone) takes precedence over ATM
        const pivotStrike = this.anchorStrike() ?? atmStrike;

        if (pivotStrike == null) return;

        // Get current cumulative values aggregated by bucket
        const currentCum = this.aggregateCumulative(snapshot, pivotStrike);

        // Process each bar interval using actual data timestamp
        for (const interval of BAR_INTERVALS_MS) {
            this.processBarInterval(interval, dataTimestamp, pivotStrike, currentCum);
        }

        // Update chart series for selected interval
        this.updateChartSeries();

        this.prevSnapshot = snapshot;
        this._tick.update((v) => v + 1);
    }

    private processBarInterval(interval: BarIntervalMs, dataTimestamp: number, pivotStrike: number, currentCum: Record<BucketKey, BucketTriple>) {
        let activeBar = this.activeBars.get(interval);
        const barList = this.bars.get(interval)!;

        // Determine bar boundaries based on actual data timestamp
        const barStartTime = Math.floor(dataTimestamp / interval) * interval;
        const barEndTime = barStartTime + interval;

        // console.log(`Bar check [${interval}ms]: dataTimestamp=${new Date(dataTimestamp).toLocaleTimeString('en-US', {timeZone: 'America/New_York'})}, barStart=${new Date(barStartTime).toLocaleTimeString('en-US', {timeZone: 'America/New_York'})}`);

        // Check if we need to start a new bar
        // Condition: Time boundary crossed OR Pivot changed (to avoid mixing buckets)
        const pivotChanged = activeBar && activeBar.atmStrike !== pivotStrike;

        if (!activeBar || activeBar.startTime !== barStartTime || pivotChanged) {
            // Close previous bar if it exists AND pivot hasn't changed
            // If pivot changed, we cannot validly close the bar because 'endCum' is based on new pivot
            // while 'startCum' was based on old pivot. The delta would be meaningless.
            if (activeBar && this.prevSnapshot && !pivotChanged) {
                const closedBar = this.closeBar(activeBar, currentCum, dataTimestamp, pivotStrike);
                // console.log(`Closed bar [${interval}ms]: ${new Date(closedBar.barStartTime).toLocaleTimeString('en-US', {timeZone: 'America/New_York'})} - ${new Date(closedBar.barEndTime).toLocaleTimeString('en-US', {timeZone: 'America/New_York'})}`);
                barList.push(closedBar);

                // Prune old bars
                if (barList.length > this.maxBars) {
                    barList.shift();
                }
            } else if (pivotChanged) {
                console.log(`Pivot changed from ${activeBar?.atmStrike} to ${pivotStrike}. Resetting active bar.`);
            }

            // Start new bar
            activeBar = {
                startTime: barStartTime,
                startCum: { ...currentCum },
                atmStrike: pivotStrike // Storing pivot as 'atmStrike' property in ActiveBar interface
            };
            this.activeBars.set(interval, activeBar);
        }
    }

    private closeBar(activeBar: ActiveBar, endCum: Record<BucketKey, BucketTriple>, endTime: number, atmStrike: number): BarData {
        const zero: BucketTriple = { premium: 0, delta: 0, gamma: 0 };
        const netFlow: Record<BucketKey, BucketTriple> = {
            call_above: { ...zero },
            call_below: { ...zero },
            put_above: { ...zero },
            put_below: { ...zero }
        };

        // Compute net flow during this bar
        for (const bucket of Object.keys(netFlow) as BucketKey[]) {
            netFlow[bucket].premium = safeDelta(endCum[bucket].premium, activeBar.startCum[bucket].premium);
            netFlow[bucket].delta = safeDelta(endCum[bucket].delta, activeBar.startCum[bucket].delta);
            netFlow[bucket].gamma = safeDelta(endCum[bucket].gamma, activeBar.startCum[bucket].gamma);
        }

        return {
            barStartTime: activeBar.startTime,
            barEndTime: endTime,
            atmStrike,
            flow: netFlow
        };
    }

    private aggregateCumulative(snapshot: FlowMap, atmStrike: number): Record<BucketKey, BucketTriple> {
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

        return cum;
    }

    private updateChartSeries() {
        const interval = this.selectedBarInterval();
        const metric = this.selectedMetric();
        const order = this.selectedOrder();
        const barList = this.bars.get(interval);

        if (!barList || barList.length < 2) return;

        // Clear existing series
        this.barSeries.clear();

        const buckets: BucketKey[] = ['call_above', 'call_below', 'put_above', 'put_below'];

        for (const bucket of buckets) {
            const points: Array<{ time: number, value: number }> = [];

            // Compute derivatives bar-to-bar
            for (let i = 1; i < barList.length; i++) {
                const currentBar = barList[i];
                const prevBar = barList[i - 1];

                const current = this.getMetricValue(currentBar.flow[bucket], metric);
                const prev = this.getMetricValue(prevBar.flow[bucket], metric);
                const dt = (currentBar.barEndTime - prevBar.barEndTime) / 1000;

                let value = 0;

                if (order === 1) {
                    // Velocity: flow per second during this bar
                    const barDuration = (currentBar.barEndTime - currentBar.barStartTime) / 1000;
                    value = current / barDuration;
                } else if (order === 2) {
                    // Acceleration: change in velocity bar-to-bar
                    const currentVel = current / ((currentBar.barEndTime - currentBar.barStartTime) / 1000);
                    const prevVel = prev / ((prevBar.barEndTime - prevBar.barStartTime) / 1000);
                    value = (currentVel - prevVel) / dt;
                } else if (order === 3) {
                    // Jerk: would need 3+ bars, skip for now
                    value = 0;
                }

                const timeSeconds = currentBar.barEndTime / 1000;
                console.log(`Adding chart point: time=${new Date(currentBar.barEndTime).toLocaleTimeString('en-US', { timeZone: 'America/New_York' })}, value=${value.toFixed(1)}`);
                points.push({
                    time: timeSeconds, // Convert to seconds for TradingView
                    value
                });
            }

            this.barSeries.set(bucket, points);

            // Update latest value
            if (points.length > 0) {
                const key = this.makeKey(bucket, metric, order);
                this.latest.set(key, points[points.length - 1].value);
            }
        }
    }

    private getMetricValue(triple: BucketTriple, metric: FlowMetric): number {
        if (metric === 'premium') return triple.premium;
        if (metric === 'delta') return triple.delta;
        if (metric === 'gamma') return triple.gamma;
        return 0;
    }
}


