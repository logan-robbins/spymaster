import { Component, Input, computed, inject, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { LevelDerivedService, DerivedLevel } from '../level-derived.service';

interface LadderTick {
  price: number;
  position: number;
}

interface LadderMarker extends DerivedLevel {
  position: number;
}

@Component({
  selector: 'app-price-ladder',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  template: `
    <div class="ladder">
      <div class="ladder-header">
        <div class="ladder-title">Price Ladder</div>
        @if (spot() !== null) {
          <div class="ladder-spot">SPY {{ spot() | number:'1.2-2' }}</div>
        }
      </div>
      <div class="ladder-body">
        <div class="ladder-rail"></div>

        @for (tick of ticks(); track tick.price) {
          <div class="ladder-tick" [style.top.%]="tick.position">
            <span>{{ tick.price | number:'1.0-0' }}</span>
          </div>
        }

        @if (spotMarker() !== null) {
          <div class="spot-marker" [style.top.%]="spotMarker()!">
            <div class="spot-line"></div>
            <div class="spot-chip">SPOT</div>
          </div>
        }

        @for (level of markers(); track level.id) {
          <div
            class="level-marker"
            [style.top.%]="level.position"
            [ngClass]="[
              'bias-' + level.bias,
              'barrier-' + level.barrier.state,
              'fuel-' + level.fuel.effect,
              level.confluenceId ? 'is-confluence' : ''
            ]"
          >
            <div class="marker-core">
              <div class="marker-kind">{{ level.kind }}</div>
              <div class="marker-price">{{ level.price | number:'1.2-2' }}</div>
              <div class="marker-direction">{{ level.direction }}</div>
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
            <div class="marker-fluid"></div>
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
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

    .ladder-body {
      position: relative;
      flex: 1;
      border-radius: 12px;
      background: radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.08), transparent 45%),
        linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.6));
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.15);
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

    .level-marker {
      position: absolute;
      right: 0.75rem;
      left: 3rem;
      transform: translateY(-50%);
      display: grid;
      grid-template-columns: 1fr 1.4fr;
      gap: 0.75rem;
      padding: 0.6rem 0.7rem;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 12px;
      backdrop-filter: blur(8px);
      box-shadow: 0 0 18px rgba(15, 23, 42, 0.4);
      overflow: hidden;
    }

    .level-marker.is-confluence {
      border-color: rgba(56, 189, 248, 0.7);
      box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
    }

    .marker-core {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      position: relative;
      z-index: 2;
    }

    .marker-kind {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #e2e8f0;
      font-weight: 600;
    }

    .marker-price {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1rem;
      color: #f8fafc;
      font-weight: 700;
    }

    .marker-direction {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: #94a3b8;
    }

    .marker-strength {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      font-family: 'IBM Plex Mono', monospace;
      position: relative;
      z-index: 2;
    }

    .strength-row {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 0.4rem;
      align-items: center;
      font-size: 0.65rem;
      color: #cbd5f5;
    }

    .strength-label {
      letter-spacing: 0.12em;
      font-weight: 600;
    }

    .strength-bar {
      height: 6px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.2);
      overflow: hidden;
    }

    .strength-fill {
      height: 100%;
      border-radius: 999px;
    }

    .strength-fill.break {
      background: linear-gradient(90deg, rgba(248, 113, 113, 0.4), rgba(248, 113, 113, 0.9));
    }

    .strength-fill.bounce {
      background: linear-gradient(90deg, rgba(34, 197, 94, 0.4), rgba(34, 197, 94, 0.9));
    }

    .strength-value {
      font-weight: 700;
      color: #f8fafc;
    }

    .marker-fluid {
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(56, 189, 248, 0.15), transparent 60%);
      opacity: 0.6;
      animation: drift 6s linear infinite;
      z-index: 1;
    }

    .bias-BREAK .marker-fluid {
      background: linear-gradient(120deg, rgba(248, 113, 113, 0.25), transparent 65%);
    }

    .bias-BOUNCE .marker-fluid {
      background: linear-gradient(120deg, rgba(34, 197, 94, 0.25), transparent 65%);
    }

    .barrier-VACUUM .marker-fluid {
      animation-duration: 3.5s;
    }

    .barrier-WALL .marker-fluid {
      animation-duration: 7.5s;
      opacity: 0.4;
    }

    .fuel-AMPLIFY .marker-fluid {
      filter: saturate(1.4);
    }

    .fuel-DAMPEN .marker-fluid {
      filter: saturate(0.6);
    }

    @keyframes drift {
      0% { transform: translateX(-10%); }
      50% { transform: translateX(10%); }
      100% { transform: translateX(-10%); }
    }
  `]
})
export class PriceLadderComponent {
  private derived = inject(LevelDerivedService);

  private rangeSignal = signal(6);

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
    const spot = this.spot();
    if (spot === null) return [] as LadderTick[];
    const range = this.rangeSignal();
    const min = Math.floor(spot - range);
    const max = Math.ceil(spot + range);
    const span = max - min;
    if (span <= 0) return [] as LadderTick[];

    const ticks: LadderTick[] = [];
    for (let price = max; price >= min; price -= 1) {
      const position = ((max - price) / span) * 100;
      ticks.push({ price, position });
    }
    return ticks;
  });

  public spotMarker = computed(() => {
    const spot = this.spot();
    if (spot === null) return null;
    const range = this.rangeSignal();
    const min = spot - range;
    const max = spot + range;
    if (max <= min) return null;
    return ((max - spot) / (max - min)) * 100;
  });

  public markers = computed(() => {
    const spot = this.spot();
    if (spot === null) return [] as LadderMarker[];
    const range = this.rangeSignal();
    const min = spot - range;
    const max = spot + range;
    const span = max - min;
    if (span <= 0) return [] as LadderMarker[];

    return this.derived.levels()
      .filter((level) => level.price >= min && level.price <= max)
      .map((level) => ({
        ...level,
        position: ((max - level.price) / span) * 100
      }));
  });
}
