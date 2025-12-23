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
          <div class="meta">Bias: {{ level()!.bias }}</div>
        }
      </div>

      <div class="bar">
        @for (slice of slices(); track slice.key) {
          <div class="slice" [style.width.%]="slice.value" [style.background]="slice.color"></div>
        }
      </div>

      <div class="legend">
        @for (slice of slices(); track slice.key) {
          <div class="legend-item">
            <span class="legend-dot" [style.background]="slice.color"></span>
            <span class="legend-label">{{ slice.label }}</span>
            <span class="legend-value">{{ slice.value }}%</span>
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
    }

    .bar {
      display: flex;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(148, 163, 184, 0.2);
    }

    .slice {
      height: 100%;
      transition: width 0.3s ease;
    }

    .legend {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.5rem;
    }

    .legend-item {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 0.4rem;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #cbd5f5;
    }

    .legend-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .legend-label {
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .legend-value {
      font-weight: 700;
    }
  `]
})
export class AttributionBarComponent {
  private derived = inject(LevelDerivedService);

  public level = this.derived.primaryLevel;

  public slices = computed(() => {
    const level = this.level();
    if (!level) {
      return [
        { key: 'barrier', label: 'Barrier', value: 0, color: '#38bdf8' },
        { key: 'tape', label: 'Tape', value: 0, color: '#f97316' },
        { key: 'fuel', label: 'Fuel', value: 0, color: '#f43f5e' },
        { key: 'approach', label: 'Approach', value: 0, color: '#a855f7' },
        { key: 'confluence', label: 'Confluence', value: 0, color: '#22c55e' }
      ];
    }

    return [
      { key: 'barrier', label: 'Barrier', value: level.contributions.barrier, color: '#38bdf8' },
      { key: 'tape', label: 'Tape', value: level.contributions.tape, color: '#f97316' },
      { key: 'fuel', label: 'Fuel', value: level.contributions.fuel, color: '#f43f5e' },
      { key: 'approach', label: 'Approach', value: level.contributions.approach, color: '#a855f7' },
      { key: 'confluence', label: 'Confluence', value: level.contributions.confluence, color: '#22c55e' }
    ];
  });
}
