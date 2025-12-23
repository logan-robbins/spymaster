import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LevelDerivedService } from '../level-derived.service';

@Component({
  selector: 'app-strength-cockpit',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="cockpit">
      <div class="cockpit-header">
        <div class="title">Strength Cockpit</div>
        @if (level()) {
          <div class="meta">
            Nearest: {{ level()!.kind }} {{ level()!.price | number:'1.2-2' }} ({{ level()!.direction }})
          </div>
        } @else {
          <div class="meta">Waiting for level stream...</div>
        }
      </div>

      @if (level()) {
        <div class="strength-grid">
          <div class="strength-card break">
            <div class="card-label">Break Strength</div>
            <div class="card-value">{{ level()!.breakStrength }}%</div>
            <div class="meter">
              <div class="meter-fill break" [style.width.%]="level()!.breakStrength"></div>
            </div>
          </div>

          <div class="strength-card bounce">
            <div class="card-label">Bounce Strength</div>
            <div class="card-value">{{ level()!.bounceStrength }}%</div>
            <div class="meter">
              <div class="meter-fill bounce" [style.width.%]="level()!.bounceStrength"></div>
            </div>
          </div>
        </div>

        <div class="trade-grid">
          <div class="trade-card call">
            <div class="trade-label">Call Success</div>
            <div class="trade-value">{{ callSuccess() }}%</div>
            <div class="trade-hint">{{ callHint() }}</div>
          </div>
          <div class="trade-card put">
            <div class="trade-label">Put Success</div>
            <div class="trade-value">{{ putSuccess() }}%</div>
            <div class="trade-hint">{{ putHint() }}</div>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .cockpit {
      background: #0f172a;
      border: 1px solid #233047;
      border-radius: 16px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .cockpit-header {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: baseline;
    }

    .title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #e2e8f0;
    }

    .meta {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      color: #94a3b8;
    }

    .strength-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
    }

    .strength-card {
      border-radius: 12px;
      padding: 0.9rem;
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.8));
      border: 1px solid rgba(148, 163, 184, 0.2);
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .strength-card.break {
      border-color: rgba(248, 113, 113, 0.4);
    }

    .strength-card.bounce {
      border-color: rgba(34, 197, 94, 0.4);
    }

    .card-label {
      font-size: 0.7rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: #cbd5f5;
    }

    .card-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.6rem;
      font-weight: 700;
      color: #f8fafc;
    }

    .meter {
      height: 8px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.2);
      overflow: hidden;
    }

    .meter-fill {
      height: 100%;
      border-radius: 999px;
    }

    .meter-fill.break {
      background: linear-gradient(90deg, rgba(248, 113, 113, 0.4), rgba(248, 113, 113, 0.95));
    }

    .meter-fill.bounce {
      background: linear-gradient(90deg, rgba(34, 197, 94, 0.4), rgba(34, 197, 94, 0.95));
    }

    .trade-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.75rem;
    }

    .trade-card {
      border-radius: 12px;
      padding: 0.8rem 0.9rem;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .trade-card.call {
      border-color: rgba(56, 189, 248, 0.4);
    }

    .trade-card.put {
      border-color: rgba(248, 113, 113, 0.4);
    }

    .trade-label {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: #94a3b8;
    }

    .trade-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.2rem;
      font-weight: 700;
      color: #f8fafc;
    }

    .trade-hint {
      font-size: 0.7rem;
      color: #cbd5f5;
    }
  `]
})
export class StrengthCockpitComponent {
  private derived = inject(LevelDerivedService);

  public level = this.derived.primaryLevel;

  public callSuccess = computed(() => {
    const level = this.level();
    if (!level) return 0;
    return level.direction === 'UP' ? level.breakStrength : level.bounceStrength;
  });

  public putSuccess = computed(() => {
    const level = this.level();
    if (!level) return 0;
    return level.direction === 'UP' ? level.bounceStrength : level.breakStrength;
  });

  public callHint = computed(() => {
    const level = this.level();
    if (!level) return '';
    return level.direction === 'UP' ? 'Break-through bias' : 'Mean reversion bias';
  });

  public putHint = computed(() => {
    const level = this.level();
    if (!level) return '';
    return level.direction === 'UP' ? 'Rejection bias' : 'Breakdown bias';
  });
}
