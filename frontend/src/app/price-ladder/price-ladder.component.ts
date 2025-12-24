import { Component, Input, computed, inject, signal, effect } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { LevelDerivedService, DerivedLevel } from '../level-derived.service';
import { FlowAnalyticsService } from '../flow-analytics.service';

interface LadderTick {
  price: number;
  top: number;
}

interface LadderMarker extends DerivedLevel {
  top: number;
}

interface GammaLadderMarker {
  id: string;
  kind: 'MAGNET' | 'CLIFF';
  price: number;
  top: number;
  netGamma: number | null;
}

@Component({
  selector: 'app-price-ladder',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  template: `
    <div class="ladder">
      <div class="ladder-header">
        <div class="ladder-title">Price Ladder</div>
        <div class="ladder-header-right">
          @if (spot() !== null) {
            <div class="ladder-spot">SPY {{ spot() | number:'1.2-2' }}</div>
          }
          <div class="scale-controls" title="Adjust vertical spacing between strikes">
            <div class="scale-label">Scale</div>
            <input
              class="scale-slider"
              type="range"
              min="0.8"
              max="1.6"
              step="0.1"
              [value]="scale()"
              (input)="onScaleInput($any($event.target).value)"
              aria-label="Price ladder scale"
            />
            <div class="scale-readout">{{ scale() | number:'1.1-1' }}×</div>
          </div>
        </div>
      </div>
      <div class="ladder-body">
        <div class="ladder-content" [style.height.px]="contentHeight()">
          <div class="ladder-rail"></div>

          @for (tick of ticks(); track tick.price) {
            <div class="ladder-tick" [style.top.px]="tick.top">
              <span>{{ tick.price | number:'1.0-0' }}</span>
            </div>
          }

          @if (spotMarker() !== null) {
            <div class="spot-marker" [style.top.px]="spotMarker()!">
              <div class="spot-line"></div>
              <div class="spot-chip">SPOT</div>
            </div>
          }

          @for (g of gammaMarkers(); track g.id) {
            <div class="gamma-marker"
                 [class.magnet]="g.kind === 'MAGNET'"
                 [class.cliff]="g.kind === 'CLIFF'"
                 [style.top.px]="g.top">
              <div class="gamma-dot"></div>
              <div class="gamma-chip">{{ g.kind }}</div>
            </div>
          }

          @for (level of markers(); track level.id) {
            <div
              class="level-marker"
              [class.compact]="!isExpanded(level.id)"
              [class.expanded]="isExpanded(level.id)"
              [style.top.px]="level.top"
              [ngClass]="[
                'bias-' + level.bias,
                'barrier-' + level.barrier.state,
                'fuel-' + level.fuel.effect,
                level.confluenceId ? 'is-confluence' : ''
              ]"
              (mouseenter)="setExpanded(level.id)"
              (mouseleave)="clearExpanded()"
            >
              <!-- Barrier Physics Visualizer (Left Border) -->
              <div class="barrier-indicator"></div>

              <div class="marker-content">
                <!-- Compact Header Row -->
                <div class="marker-header">
                  <div class="header-main">
                    <span class="marker-kind">{{ level.kind }}</span>
                    <span class="marker-price">{{ level.price | number:'1.2-2' }}</span>
                  </div>

                  <!-- Compact Strength Visual (Only in compact mode) -->
                  @if (!isExpanded(level.id)) {
                    <div class="compact-meter">
                      <div class="mini-bar" 
                           [style.width.%]="level.bias === 'BREAK' ? level.breakStrength : level.bounceStrength"
                           [class.break]="level.bias === 'BREAK'"
                           [class.bounce]="level.bias === 'BOUNCE'">
                      </div>
                    </div>
                  }

                  <!-- Physics Badges -->
                  <div class="physics-badges">
                    @if (level.approach.priorTouches > 0) {
                      <div class="badge touch-badge" title="Prior touches">
                        T{{ level.approach.priorTouches }}
                      </div>
                    }
                    @if (isUnderPressure(level)) {
                      <div class="badge pressure-badge" title="High Tape Velocity">
                        ⚡
                      </div>
                    }
                  </div>
                </div>

                <!-- Expanded Details -->
                @if (isExpanded(level.id)) {
                  <div class="marker-details">
                    <div class="marker-sub-row">
                      <div class="marker-direction">{{ level.direction }}</div>
                      @if (level.barrier.state !== 'NEUTRAL') {
                        <div class="state-label">{{ level.barrier.state }}</div>
                      }
                    </div>

                    <div class="marker-strength">
                      <div class="strength-row">
                        <div class="strength-label">BREAK</div>
                        <div class="strength-bar">
                          <div class="strength-fill break" [style.width.%]="level.breakStrength"></div>
                        </div>
                        <div class="strength-value">{{ level.breakStrength }}%</div>
                      </div>
                      <div class="strength-row">
                        <div class="strength-label">BOUNCE</div>
                        <div class="strength-bar">
                          <div class="strength-fill bounce" [style.width.%]="level.bounceStrength"></div>
                        </div>
                        <div class="strength-value">{{ level.bounceStrength }}%</div>
                      </div>
                    </div>
                  </div>
                }
              </div>

              <div class="marker-fluid"></div>
            </div>
          }
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
    }

    .ladder {
      background: #0f172a;
      border: 1px solid #233047;
      border-radius: 16px;
      padding: 1rem;
      height: 100%;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      position: relative;
      overflow: hidden;
    }

    .ladder-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    .ladder-header-right {
      display: flex;
      align-items: center;
      gap: 0.9rem;
    }

    .ladder-title {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 600;
      font-size: 1rem;
      color: #e2e8f0;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .ladder-spot {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.9rem;
      color: #38bdf8;
    }

    .scale-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-family: 'IBM Plex Mono', monospace;
    }

    .scale-label {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: rgba(148, 163, 184, 0.9);
      white-space: nowrap;
    }

    .scale-slider {
      width: 120px;
      height: 6px;
      accent-color: #38bdf8;
      opacity: 0.9;
    }

    .scale-readout {
      font-size: 0.7rem;
      color: rgba(226, 232, 240, 0.92);
      min-width: 3.2rem;
      text-align: right;
      white-space: nowrap;
    }

    .ladder-body {
      position: relative;
      flex: 1;
      border-radius: 12px;
      background: radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.08), transparent 45%),
        linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.6));
      overflow-x: hidden;
      overflow-y: auto;
      border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .ladder-content {
      position: relative;
      min-height: 100%;
    }

    .ladder-rail {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 2rem;
      width: 2px;
      background: linear-gradient(180deg, rgba(148, 163, 184, 0.1), rgba(148, 163, 184, 0.5), rgba(148, 163, 184, 0.1));
    }

    .ladder-tick {
      position: absolute;
      left: 0.25rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transform: translateY(-50%);
      color: #94a3b8;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
    }

    .spot-marker {
      position: absolute;
      left: 2rem;
      right: 0.75rem;
      transform: translateY(-50%);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      z-index: 10;
      pointer-events: none;
    }

    .spot-line {
      height: 2px;
      flex: 1;
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(56, 189, 248, 0.8));
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }

    .spot-chip {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      letter-spacing: 0.2em;
      color: #0f172a;
      background: #38bdf8;
      padding: 0.2rem 0.4rem;
      border-radius: 999px;
      text-transform: uppercase;
      font-weight: 700;
    }

    /* Gamma profile markers (MAGNET / CLIFF) */
    .gamma-marker {
      position: absolute;
      left: 1.65rem;
      transform: translateY(-50%);
      display: flex;
      align-items: center;
      gap: 0.35rem;
      z-index: 7;
      pointer-events: none;
    }

    .gamma-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      border: 2px solid rgba(148, 163, 184, 0.55);
      background: rgba(15, 23, 42, 0.95);
      box-shadow: 0 0 8px rgba(148, 163, 184, 0.12);
    }

    .gamma-chip {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.55rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      padding: 0.12rem 0.35rem;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      background: rgba(15, 23, 42, 0.6);
      color: rgba(226, 232, 240, 0.92);
      line-height: 1;
      white-space: nowrap;
    }

    .gamma-marker.magnet .gamma-dot {
      border-color: rgba(34, 197, 94, 0.75);
      box-shadow: 0 0 10px rgba(34, 197, 94, 0.25);
    }

    .gamma-marker.magnet .gamma-chip {
      border-color: rgba(34, 197, 94, 0.45);
      background: rgba(34, 197, 94, 0.12);
      color: #86efac;
    }

    .gamma-marker.cliff .gamma-dot {
      border-color: rgba(248, 113, 113, 0.75);
      box-shadow: 0 0 10px rgba(248, 113, 113, 0.25);
    }

    .gamma-marker.cliff .gamma-chip {
      border-color: rgba(248, 113, 113, 0.45);
      background: rgba(248, 113, 113, 0.10);
      color: #fca5a5;
    }

    .level-marker {
      position: absolute;
      right: 0.75rem;
      left: 3rem;
      transform: translateY(-50%);
      display: flex;
      gap: 0.75rem;
      padding: 0.4rem 0.6rem;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 4px 8px 8px 4px;
      backdrop-filter: blur(8px);
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.4);
      overflow: hidden;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      cursor: pointer;
      z-index: 5;
      min-height: 48px;
    }
    
    .level-marker.compact {
      max-height: 48px;
    }

    .level-marker.expanded {
      z-index: 100;
      background: rgba(15, 23, 42, 0.95);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
      min-height: auto;
      border-color: rgba(56, 189, 248, 0.4);
      transform: translateY(-50%) scale(1.02);
    }

    .marker-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      position: relative;
      z-index: 2;
      justify-content: center;
    }

    .marker-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.8rem;
      width: 100%;
    }

    .header-main {
      display: flex;
      align-items: baseline;
      gap: 0.6rem;
    }

    .marker-kind {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #e2e8f0;
      font-weight: 600;
      white-space: nowrap;
    }

    .marker-price {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.1rem;
      color: #f8fafc;
      font-weight: 700;
    }

    .compact-meter {
      flex: 1;
      height: 4px;
      background: rgba(148, 163, 184, 0.2);
      border-radius: 999px;
      overflow: hidden;
      max-width: 60px;
    }

    .mini-bar {
      height: 100%;
      border-radius: 999px;
    }

    .mini-bar.break { background: #f87171; }
    .mini-bar.bounce { background: #22c55e; }

    /* Expanded Details */
    .marker-details {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      animation: slide-down 0.2s ease-out;
    }

    .marker-sub-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      border-top: 1px solid rgba(148, 163, 184, 0.1);
      padding-top: 0.5rem;
    }

    .marker-strength {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      font-family: 'IBM Plex Mono', monospace;
    }

    /* Barrier Visuals */
    .barrier-indicator {
      width: 4px;
      background: #475569;
      border-radius: 2px;
      height: auto;
      min-height: 100%;
    }

    .barrier-WALL .barrier-indicator {
      background: #f8fafc;
      box-shadow: 0 0 8px rgba(248, 250, 252, 0.5);
      width: 5px;
    }

    .barrier-VACUUM .barrier-indicator {
      background: transparent;
      border: 1px dashed #94a3b8;
      width: 4px;
      opacity: 0.6;
    }

    .barrier-ABSORPTION .barrier-indicator {
      background: repeating-linear-gradient(
        45deg,
        #38bdf8,
        #38bdf8 4px,
        transparent 4px,
        transparent 8px
      );
      width: 5px;
    }

    .level-marker.is-confluence {
      border-color: rgba(56, 189, 248, 0.7);
    }

    .physics-badges {
      display: flex;
      gap: 0.25rem;
      margin-left: auto;
    }

    .badge {
      font-size: 0.6rem;
      font-weight: 700;
      padding: 0.1rem 0.3rem;
      border-radius: 3px;
      font-family: 'IBM Plex Mono', monospace;
      line-height: 1;
    }

    .touch-badge {
      background: rgba(148, 163, 184, 0.2);
      color: #e2e8f0;
      border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .pressure-badge {
      background: rgba(248, 113, 113, 0.2);
      color: #f87171;
      border: 1px solid rgba(248, 113, 113, 0.3);
      animation: pulse-red 1.5s infinite;
    }

    .marker-direction {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: #94a3b8;
    }
    
    .state-label {
      font-size: 0.55rem;
      text-transform: uppercase;
      padding: 0.1rem 0.3rem;
      border-radius: 3px;
      background: rgba(15, 23, 42, 0.5);
      color: #cbd5e1;
      border: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* Strength bars (Expanded) */
    .strength-row {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 0.4rem;
      align-items: center;
      font-size: 0.65rem;
      color: #cbd5f5;
    }

    .strength-label { letter-spacing: 0.12em; font-weight: 600; }
    .strength-bar { height: 5px; border-radius: 999px; background: rgba(148, 163, 184, 0.2); overflow: hidden; }
    .strength-fill { height: 100%; border-radius: 999px; }
    .strength-fill.break { background: linear-gradient(90deg, rgba(248, 113, 113, 0.4), rgba(248, 113, 113, 0.9)); }
    .strength-fill.bounce { background: linear-gradient(90deg, rgba(34, 197, 94, 0.4), rgba(34, 197, 94, 0.9)); }
    .strength-value { font-weight: 700; color: #f8fafc; }

    .marker-fluid {
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(56, 189, 248, 0.1), transparent 60%);
      opacity: 0.4;
      pointer-events: none;
      z-index: 1;
    }

    /* Bias-based fluid colors */
    .bias-BREAK .marker-fluid { background: linear-gradient(120deg, rgba(248, 113, 113, 0.15), transparent 65%); }
    .bias-BOUNCE .marker-fluid { background: linear-gradient(120deg, rgba(34, 197, 94, 0.15), transparent 65%); }

    @keyframes slide-down {
      from { opacity: 0; transform: translateY(-5px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse-red {
      0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.4); }
      70% { box-shadow: 0 0 0 4px rgba(248, 113, 113, 0); }
      100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
    }
  `]
})
export class PriceLadderComponent {
  private derived = inject(LevelDerivedService);
  private analytics = inject(FlowAnalyticsService);
  protected Math = Math;

  private rangeSignal = signal(3);
  public expandedId = signal<string | null>(null);
  private anchorPrice = signal<number | null>(null);
  private scaleSignal = signal(1.0);

  public scale = this.scaleSignal.asReadonly();

  private rowHeightPx = computed(() => Math.round(80 * this.scaleSignal()));

  private ladderViewport = computed(() => {
    const anchor = this.anchorPrice();
    if (anchor === null) return null;

    const range = this.rangeSignal();
    const min = Math.floor(anchor - range);
    const max = Math.ceil(anchor + range);
    const span = max - min;
    if (span <= 0) return null;

    const row = this.rowHeightPx();
    const half = row / 2;
    const height = (span + 1) * row;

    return { anchor, min, max, span, row, half, height };
  });

  public contentHeight = computed(() => this.ladderViewport()?.height ?? 0);

  constructor() {
    effect(() => {
      const s = this.spot();
      if (s === null) return;
      
      const anchor = this.anchorPrice();
      // Initialize anchor or update if spot moves too far (hysteresis)
      if (anchor === null || Math.abs(s - anchor) > 1.8) {
        this.anchorPrice.set(Math.round(s));
      }
    }, { allowSignalWrites: true });
  }

  @Input()
  set range(value: number) {
    if (Number.isFinite(value)) {
      this.rangeSignal.set(Math.max(2, value));
    }
  }

  public spot = computed(() => {
    const spy = this.derived.spy();
    return spy ? spy.spot : null;
  });

  public ticks = computed(() => {
    const vp = this.ladderViewport();
    if (!vp) return [] as LadderTick[];

    const ticks: LadderTick[] = [];
    for (let price = vp.max; price >= vp.min; price -= 1) {
      const top = (vp.max - price) * vp.row + vp.half;
      ticks.push({ price, top });
    }
    return ticks;
  });

  public spotMarker = computed(() => {
    const vp = this.ladderViewport();
    const spot = this.spot();
    if (!vp || spot === null) return null;
    return (vp.max - spot) * vp.row + vp.half;
  });

  public gammaMarkers = computed(() => {
    const vp = this.ladderViewport();
    if (!vp) return [] as GammaLadderMarker[];

    const gammaByStrike = this.analytics.netGammaByStrike();
    const strikesAsc = Array.from(gammaByStrike.keys())
      .filter((s) => s >= vp.min && s <= vp.max)
      .sort((a, b) => a - b);

    if (strikesAsc.length < 1) return [] as GammaLadderMarker[];

    // "Local" MAGNET/CLIFF relative to the current viewport. This keeps markers visible
    // even when the global profile extrema are slightly off-screen.
    let magnet: number | null = null;
    let bestGamma = 0;
    for (const s of strikesAsc) {
      const g = gammaByStrike.get(s) ?? 0;
      if (!Number.isFinite(g)) continue;
      if (g > bestGamma) {
        bestGamma = g;
        magnet = s;
      }
    }

    let cliff: number | null = null;
    let bestDist = Number.POSITIVE_INFINITY;
    for (let i = 1; i < strikesAsc.length; i++) {
      const s0 = strikesAsc[i - 1];
      const s1 = strikesAsc[i];
      const a = gammaByStrike.get(s0) ?? 0;
      const b = gammaByStrike.get(s1) ?? 0;
      if (a === 0 || b === 0 || (a > 0) !== (b > 0)) {
        const candidate = Math.abs(a) < Math.abs(b) ? s0 : s1;
        const dist = Math.abs(candidate - vp.anchor);
        if (dist < bestDist) {
          bestDist = dist;
          cliff = candidate;
        }
      }
    }

    const out: GammaLadderMarker[] = [];

    if (magnet != null && bestGamma > 0) {
      out.push({
        id: 'gamma-magnet',
        kind: 'MAGNET',
        price: magnet,
        top: (vp.max - magnet) * vp.row + vp.half,
        netGamma: gammaByStrike.get(magnet) ?? null
      });
    }

    if (cliff != null) {
      out.push({
        id: 'gamma-cliff',
        kind: 'CLIFF',
        price: cliff,
        top: (vp.max - cliff) * vp.row + vp.half,
        netGamma: gammaByStrike.get(cliff) ?? null
      });
    }

    return out;
  });

  public markers = computed(() => {
    const vp = this.ladderViewport();
    if (!vp) return [] as LadderMarker[];
    return this.derived.levels()
      .filter((level) => level.price >= vp.min && level.price <= vp.max)
      .map((level) => ({
        ...level,
        top: (vp.max - level.price) * vp.row + vp.half
      }));
  });

  public onScaleInput(value: string) {
    const n = Number(value);
    if (!Number.isFinite(n)) return;
    this.scaleSignal.set(Math.min(1.6, Math.max(0.8, n)));
  }

  public isUnderPressure(level: DerivedLevel): boolean {
     const vel = level.tape.velocity;
     const isHighVelocity = Math.abs(vel) > 20; 
     if (!isHighVelocity) return false;
     if (level.direction === 'UP' && vel > 0) return true;
     if (level.direction === 'DOWN' && vel < 0) return true;
     return false;
  }

  public setExpanded(id: string) {
    this.expandedId.set(id);
  }

  public clearExpanded() {
    this.expandedId.set(null);
  }

  public isExpanded(id: string): boolean {
    return this.expandedId() === id;
  }
}
