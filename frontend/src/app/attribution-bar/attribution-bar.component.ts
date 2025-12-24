import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LevelDerivedService } from '../level-derived.service';

interface AttributionSlice {
  key: string;
  label: string;
  value: number;
  color: string;
}

@Component({
  selector: 'app-attribution-bar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="attribution">
      <div class="header">
        <div class="title">Attribution</div>
        @if (level()) {
          <div class="meta">
            Bias: {{ level()!.bias }}
            <span class="edge" [ngClass]="edgeClass()">{{ edgeText() }}</span>
          </div>
        }
      </div>

      <div class="tug-wrap">
        <div class="tug-label break">
          BREAK <span class="tug-total">{{ breakTotal() }}%</span>
        </div>

        <div class="tug">
          <div class="half break">
            @for (slice of breakSlices(); track slice.key) {
              <div class="seg" [style.width.%]="slice.value" [style.background]="slice.color"></div>
            }
          </div>

          <div class="half bounce">
            @for (slice of bounceSlices(); track slice.key) {
              <div class="seg" [style.width.%]="slice.value" [style.background]="slice.color"></div>
            }
          </div>

          <div class="axis"></div>
          <div class="knot" [style.left.%]="knotPos()"></div>
        </div>

        <div class="tug-label bounce">
          <span class="tug-total">{{ bounceTotal() }}%</span> BOUNCE
        </div>
      </div>

      <div class="rows">
        @for (row of rows(); track row.key) {
          <div class="row">
            <div class="row-label">
              <span class="dot" [style.background]="row.color"></span>
              <span class="lbl">{{ row.label }}</span>
            </div>
            <div class="row-bars">
              <div class="row-axis"></div>
              <div class="row-fill break" [style.width.%]="row.breakValue / 2" [style.background]="row.color"></div>
              <div class="row-fill bounce" [style.width.%]="row.bounceValue / 2" [style.background]="row.color"></div>
            </div>
            <div class="row-values">
              <span class="v break">{{ row.breakValue | number:'1.0-0' }}</span>
              <span class="v bounce">{{ row.bounceValue | number:'1.0-0' }}</span>
            </div>
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
    .attribution {
      background: #0f172a;
      border: 1px solid #233047;
      border-radius: 14px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
    }

    .title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.85rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #e2e8f0;
    }

    .meta {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: #94a3b8;
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
    }

    .edge {
      padding: 0.1rem 0.35rem;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.25);
      background: rgba(15, 23, 42, 0.6);
      color: #cbd5f5;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .edge.edge-break {
      border-color: rgba(248, 113, 113, 0.35);
      color: #fca5a5;
    }

    .edge.edge-bounce {
      border-color: rgba(34, 197, 94, 0.35);
      color: #86efac;
    }

    .tug-wrap {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 0.8rem;
      align-items: center;
    }

    .tug-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #94a3b8;
      white-space: nowrap;
      display: inline-flex;
      gap: 0.45rem;
      align-items: center;
    }

    .tug-label.break {
      color: #fca5a5;
    }

    .tug-label.bounce {
      color: #86efac;
    }

    .tug-total {
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: 0;
    }

    .tug {
      position: relative;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(148, 163, 184, 0.18);
      display: flex;
    }

    .half {
      width: 50%;
      height: 100%;
      display: flex;
    }

    .half.break {
      flex-direction: row-reverse;
      justify-content: flex-start;
    }

    .half.bounce {
      justify-content: flex-start;
    }

    .seg {
      height: 100%;
      opacity: 0.9;
    }

    .axis {
      position: absolute;
      left: 50%;
      top: -3px;
      bottom: -3px;
      width: 2px;
      background: rgba(248, 250, 252, 0.4);
      transform: translateX(-50%);
    }

    .knot {
      position: absolute;
      top: 50%;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 2px solid rgba(248, 250, 252, 0.7);
      background: rgba(15, 23, 42, 0.95);
      transform: translate(-50%, -50%);
      box-shadow: 0 0 14px rgba(56, 189, 248, 0.22);
    }

    .rows {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      margin-top: 0.15rem;
    }

    .row {
      display: grid;
      grid-template-columns: 120px 1fr 64px;
      gap: 0.6rem;
      align-items: center;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #cbd5f5;
    }

    .row-label {
      display: flex;
      align-items: center;
      gap: 0.45rem;
      min-width: 0;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex: 0 0 auto;
    }

    .lbl {
      text-transform: uppercase;
      letter-spacing: 0.1em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .row-bars {
      position: relative;
      height: 8px;
      background: rgba(148, 163, 184, 0.16);
      border-radius: 999px;
      overflow: hidden;
    }

    .row-axis {
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      width: 1px;
      background: rgba(248, 250, 252, 0.25);
      transform: translateX(-50%);
    }

    .row-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      opacity: 0.9;
    }

    .row-fill.break {
      right: 50%;
      border-radius: 999px 0 0 999px;
    }

    .row-fill.bounce {
      left: 50%;
      border-radius: 0 999px 999px 0;
    }

    .row-values {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.35rem;
      justify-items: end;
      color: #94a3b8;
    }

    .v.break { color: #fca5a5; font-weight: 700; }
    .v.bounce { color: #86efac; font-weight: 700; }
  `]
})
export class AttributionBarComponent {
  private derived = inject(LevelDerivedService);

  public level = this.derived.primaryLevel;

  private palette = {
    barrier: { label: 'Barrier', color: '#38bdf8' },
    tape: { label: 'Tape', color: '#f97316' },
    fuel: { label: 'Fuel', color: '#f43f5e' },
    approach: { label: 'Approach', color: '#a855f7' },
    confluence: { label: 'Confluence', color: '#22c55e' }
  } as const;

  public breakTotal = computed(() => this.level()?.forces.break.total ?? 0);
  public bounceTotal = computed(() => this.level()?.forces.bounce.total ?? 0);
  public net = computed(() => this.level()?.forces.net ?? 0);

  public knotPos = computed(() => {
    // Net is [-100, 100]. We keep knot within [25, 75] to avoid extremes.
    const net = Math.max(-100, Math.min(100, this.net()));
    const shift = Math.max(-25, Math.min(25, net * 0.25)); // 100 -> 25
    return 50 - shift;
  });

  public edgeText = computed(() => {
    const net = this.net();
    const sign = net > 0 ? '+' : '';
    return `EDGE ${sign}${net}`;
  });

  public edgeClass = computed(() => {
    const net = this.net();
    if (net > 5) return 'edge-break';
    if (net < -5) return 'edge-bounce';
    return '';
  });

  public breakSlices = computed<AttributionSlice[]>(() => {
    const forces = this.level()?.forces;
    if (!forces) {
      return [
        { key: 'barrier', label: this.palette.barrier.label, value: 0, color: this.palette.barrier.color },
        { key: 'tape', label: this.palette.tape.label, value: 0, color: this.palette.tape.color },
        { key: 'fuel', label: this.palette.fuel.label, value: 0, color: this.palette.fuel.color },
        { key: 'approach', label: this.palette.approach.label, value: 0, color: this.palette.approach.color },
        { key: 'confluence', label: this.palette.confluence.label, value: 0, color: this.palette.confluence.color }
      ];
    }

    return [
      { key: 'barrier', label: this.palette.barrier.label, value: forces.break.barrier, color: this.palette.barrier.color },
      { key: 'tape', label: this.palette.tape.label, value: forces.break.tape, color: this.palette.tape.color },
      { key: 'fuel', label: this.palette.fuel.label, value: forces.break.fuel, color: this.palette.fuel.color },
      { key: 'approach', label: this.palette.approach.label, value: forces.break.approach, color: this.palette.approach.color },
      { key: 'confluence', label: this.palette.confluence.label, value: forces.break.confluence, color: this.palette.confluence.color }
    ];
  });

  public bounceSlices = computed<AttributionSlice[]>(() => {
    const forces = this.level()?.forces;
    if (!forces) {
      return [
        { key: 'barrier', label: this.palette.barrier.label, value: 0, color: this.palette.barrier.color },
        { key: 'tape', label: this.palette.tape.label, value: 0, color: this.palette.tape.color },
        { key: 'fuel', label: this.palette.fuel.label, value: 0, color: this.palette.fuel.color },
        { key: 'approach', label: this.palette.approach.label, value: 0, color: this.palette.approach.color },
        { key: 'confluence', label: this.palette.confluence.label, value: 0, color: this.palette.confluence.color }
      ];
    }

    return [
      { key: 'barrier', label: this.palette.barrier.label, value: forces.bounce.barrier, color: this.palette.barrier.color },
      { key: 'tape', label: this.palette.tape.label, value: forces.bounce.tape, color: this.palette.tape.color },
      { key: 'fuel', label: this.palette.fuel.label, value: forces.bounce.fuel, color: this.palette.fuel.color },
      { key: 'approach', label: this.palette.approach.label, value: forces.bounce.approach, color: this.palette.approach.color },
      { key: 'confluence', label: this.palette.confluence.label, value: forces.bounce.confluence, color: this.palette.confluence.color }
    ];
  });

  public rows = computed(() => {
    const forces = this.level()?.forces;
    const safe = forces ?? {
      break: { barrier: 0, tape: 0, fuel: 0, approach: 0, confluence: 0, total: 0 },
      bounce: { barrier: 0, tape: 0, fuel: 0, approach: 0, confluence: 0, total: 0 },
      net: 0
    };

    return [
      { key: 'barrier', label: this.palette.barrier.label, color: this.palette.barrier.color, breakValue: safe.break.barrier, bounceValue: safe.bounce.barrier },
      { key: 'tape', label: this.palette.tape.label, color: this.palette.tape.color, breakValue: safe.break.tape, bounceValue: safe.bounce.tape },
      { key: 'fuel', label: this.palette.fuel.label, color: this.palette.fuel.color, breakValue: safe.break.fuel, bounceValue: safe.bounce.fuel },
      { key: 'approach', label: this.palette.approach.label, color: this.palette.approach.color, breakValue: safe.break.approach, bounceValue: safe.bounce.approach },
      { key: 'confluence', label: this.palette.confluence.label, color: this.palette.confluence.color, breakValue: safe.break.confluence, bounceValue: safe.bounce.confluence }
    ];
  });
}
