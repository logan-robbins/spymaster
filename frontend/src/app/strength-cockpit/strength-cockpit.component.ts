import { Component, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LevelDerivedService } from '../level-derived.service';
import { DataStreamService } from '../data-stream.service';

type TrafficState = 'go' | 'wait' | 'no-go' | 'offline';

@Component({
  selector: 'app-strength-cockpit',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="cockpit">
      <div class="cockpit-header">
        <div class="header-top">
          <div class="title">Strength Cockpit</div>
          <div class="traffic-pill" [ngClass]="'traffic-' + tradeStatus().state">
            <div class="traffic-label">{{ tradeStatus().label }}</div>
            <div class="traffic-detail">{{ tradeStatus().detail }}</div>
          </div>
        </div>
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

        <!-- Velocity & Gamma Section -->
        <div class="mechanics-section">
          <div class="section-label">🚀 Mechanics</div>
          <div class="mechanics-grid">
            <!-- Tape Velocity -->
            <div class="mechanic-item">
              <div class="mechanic-label">Tape Velocity</div>
              <div class="mechanic-value" [ngClass]="tapeVelocityClass()">
                {{ tapeVelocity() | number:'1.1-1' }}
              </div>
              <div class="mechanic-bar">
                <div class="bar-fill tape" [style.width.%]="tapeVelocityPercent()"></div>
              </div>
            </div>

            <!-- Approach Velocity -->
            <div class="mechanic-item">
              <div class="mechanic-label">Approach Speed</div>
              <div class="mechanic-value" [ngClass]="approachVelocityClass()">
                {{ approachVelocity() | number:'1.2-2' }}
              </div>
              <div class="mechanic-bar">
                <div class="bar-fill approach" [style.width.%]="approachVelocityPercent()"></div>
              </div>
            </div>

            <!-- Gamma Exposure -->
            <div class="mechanic-item gamma-exposure">
              <div class="mechanic-label">Dealer Gamma</div>
              <div class="mechanic-value" [ngClass]="gammaExposureClass()">
                {{ formatGamma(gammaExposure()) }}
              </div>
              <div class="gamma-indicator">
                <div class="gamma-bar" [style.left.%]="50" [style.width.%]="Math.abs(gammaExposurePercent())"></div>
                <div class="gamma-center"></div>
                <div class="gamma-label left">SHORT</div>
                <div class="gamma-label right">LONG</div>
              </div>
            </div>

            <!-- Gamma Velocity -->
            <div class="mechanic-item">
              <div class="mechanic-label">Gamma Velocity</div>
              <div class="mechanic-value" [ngClass]="gammaVelocityClass()">
                {{ formatGamma(gammaVelocity()) }}/s
              </div>
              <div class="mechanic-hint">{{ gammaVelocityHint() }}</div>
            </div>

            <!-- Gamma Regime -->
            <div class="mechanic-item">
              <div class="mechanic-label">Gamma Regime</div>
              <div class="mechanic-value" [ngClass]="gammaRegimeClass()">
                {{ gammaRegimeLabel() }}
              </div>
              <div class="mechanic-hint">{{ gammaRegimeHint() }}</div>
            </div>
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
      flex-direction: column;
      gap: 1rem;
      align-items: baseline;
    }

    .header-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      gap: 1rem;
    }

    .title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #e2e8f0;
    }

    .traffic-pill {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 0.15rem;
      padding: 0.35rem 0.55rem;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      background: rgba(15, 23, 42, 0.6);
      font-family: 'IBM Plex Mono', monospace;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      min-width: 92px;
    }

    .traffic-label {
      font-size: 0.7rem;
      font-weight: 800;
      color: #f8fafc;
    }

    .traffic-detail {
      font-size: 0.55rem;
      letter-spacing: 0.12em;
      color: #94a3b8;
      font-weight: 600;
    }

    .traffic-go {
      border-color: rgba(34, 197, 94, 0.35);
      background: rgba(34, 197, 94, 0.10);
    }
    .traffic-go .traffic-label { color: #86efac; }

    .traffic-wait {
      border-color: rgba(251, 191, 36, 0.35);
      background: rgba(251, 191, 36, 0.10);
    }
    .traffic-wait .traffic-label { color: #fbbf24; }

    .traffic-no-go {
      border-color: rgba(248, 113, 113, 0.35);
      background: rgba(248, 113, 113, 0.10);
    }
    .traffic-no-go .traffic-label { color: #fca5a5; }

    .traffic-offline {
      border-color: rgba(148, 163, 184, 0.25);
      background: rgba(148, 163, 184, 0.08);
    }
    .traffic-offline .traffic-label { color: #cbd5f5; }

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

    .mechanics-section {
      margin-top: 0.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .section-label {
      font-size: 0.75rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #38bdf8;
      font-weight: 600;
    }

    .mechanics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }

    .mechanic-item {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 10px;
      padding: 0.7rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .mechanic-item.gamma-exposure {
      grid-column: span 2;
    }

    .mechanic-label {
      font-size: 0.65rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #94a3b8;
    }

    .mechanic-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.1rem;
      font-weight: 700;
      color: #f8fafc;
    }

    .mechanic-value.positive { color: #22c55e; }
    .mechanic-value.negative { color: #f87171; }
    .mechanic-value.neutral { color: #94a3b8; }

    .mechanic-bar {
      height: 6px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.2);
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.3s ease;
    }

    .bar-fill.tape {
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.4), rgba(56, 189, 248, 0.95));
    }

    .bar-fill.approach {
      background: linear-gradient(90deg, rgba(251, 191, 36, 0.4), rgba(251, 191, 36, 0.95));
    }

    .gamma-indicator {
      position: relative;
      height: 24px;
      background: linear-gradient(90deg, rgba(248, 113, 113, 0.3), rgba(148, 163, 184, 0.2) 50%, rgba(34, 197, 94, 0.3));
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .gamma-bar {
      position: absolute;
      top: 0;
      bottom: 0;
      background: rgba(56, 189, 248, 0.6);
      border-left: 2px solid #38bdf8;
      transition: left 0.3s ease, width 0.3s ease;
    }

    .gamma-center {
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      width: 2px;
      background: rgba(248, 250, 252, 0.5);
      transform: translateX(-50%);
    }

    .gamma-label {
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      font-size: 0.6rem;
      letter-spacing: 0.1em;
      color: #94a3b8;
      font-weight: 600;
    }

    .gamma-label.left { left: 6px; }
    .gamma-label.right { right: 6px; }

    .mechanic-hint {
      font-size: 0.65rem;
      color: #64748b;
      font-style: italic;
    }
  `]
})
export class StrengthCockpitComponent {
  private derived = inject(LevelDerivedService);
  private stream = inject(DataStreamService);
  public Math = Math;  // Expose Math to template

  public level = this.derived.primaryLevel;

  public tradeStatus = computed((): { state: TrafficState; label: string; detail: string } => {
    const connection = this.stream.connectionStatus();
    const dataStatus = this.stream.dataStatus();
    if (connection !== 'connected') {
      return { state: 'offline', label: 'OFFLINE', detail: 'NO STREAM' };
    }
    if (dataStatus !== 'ok') {
      return { state: 'offline', label: 'OFFLINE', detail: 'NO DATA' };
    }

    const level = this.level();
    if (!level) {
      return { state: 'offline', label: 'WAIT', detail: 'NO LEVEL' };
    }

    const edge = level.breakStrength - level.bounceStrength;
    const edgeAbs = Math.abs(edge);
    const edgeText = `${edge >= 0 ? '+' : ''}${edge}`;
    const conf = level.confidence;

    // Simplified tradeability heuristic using existing schema.
    if (conf === 'LOW' || level.signal === 'CHOP') {
      return { state: 'no-go', label: 'NO-GO', detail: `${conf} · EDGE ${edgeText}` };
    }
    if (conf === 'HIGH' && edgeAbs >= 15) {
      return { state: 'go', label: 'GO', detail: `${conf} · EDGE ${edgeText}` };
    }
    return { state: 'wait', label: 'WAIT', detail: `${conf} · EDGE ${edgeText}` };
  });

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

  // Velocity & Mechanics
  public tapeVelocity = computed(() => {
    const level = this.level();
    return level ? level.tape.velocity : 0;
  });

  public tapeVelocityPercent = computed(() => {
    const vel = Math.abs(this.tapeVelocity());
    return Math.min((vel / 100) * 100, 100);
  });

  public tapeVelocityClass = computed(() => {
    const vel = this.tapeVelocity();
    if (vel > 5) return 'positive';
    if (vel < -5) return 'negative';
    return 'neutral';
  });

  public approachVelocity = computed(() => {
    const level = this.level();
    return level ? level.approach.velocity : 0;
  });

  public approachVelocityPercent = computed(() => {
    const vel = Math.abs(this.approachVelocity());
    return Math.min((vel / 1.0) * 100, 100);
  });

  public approachVelocityClass = computed(() => {
    const vel = this.approachVelocity();
    if (vel > 0.1) return 'positive';
    if (vel < -0.1) return 'negative';
    return 'neutral';
  });

  public gammaExposure = computed(() => {
    const level = this.level();
    return level ? level.fuel.gammaExposure : 0;
  });

  public gammaExposurePercent = computed(() => {
    const gamma = this.gammaExposure();
    const normalized = Math.max(-50, Math.min(50, gamma / 1000));
    return normalized;
  });

  public gammaExposureClass = computed(() => {
    const gamma = this.gammaExposure();
    if (gamma > 5000) return 'positive';
    if (gamma < -5000) return 'negative';
    return 'neutral';
  });

  public gammaVelocity = computed(() => {
    const level = this.level();
    return level ? level.fuel.gammaVelocity : 0;
  });

  public gammaVelocityClass = computed(() => {
    const vel = this.gammaVelocity();
    if (vel > 100) return 'positive';
    if (vel < -100) return 'negative';
    return 'neutral';
  });

  public gammaVelocityHint = computed(() => {
    const vel = this.gammaVelocity();
    if (vel > 500) return 'Dealers accumulating FAST';
    if (vel > 100) return 'Dealers building position';
    if (vel < -500) return 'Dealers exiting FAST';
    if (vel < -100) return 'Dealers reducing exposure';
    return 'Stable positioning';
  });

  // Gamma Regime (Sticky vs Slippery)
  public gammaRegimeLabel = computed(() => {
    const level = this.level();
    if (!level) return 'UNKNOWN';
    const eff = level.fuel.effect;
    if (eff === 'AMPLIFY') return 'SLIPPERY';
    if (eff === 'DAMPEN') return 'STICKY';
    return 'NEUTRAL';
  });

  public gammaRegimeClass = computed(() => {
    const label = this.gammaRegimeLabel();
    if (label === 'STICKY') return 'positive';
    if (label === 'SLIPPERY') return 'negative';
    return 'neutral';
  });

  public gammaRegimeHint = computed(() => {
    const level = this.level();
    if (!level) return 'No level selected';
    const eff = level.fuel.effect;
    if (eff === 'AMPLIFY') return 'Vol expands (dealers chase)';
    if (eff === 'DAMPEN') return 'Pinning / mean reversion';
    return 'Mixed regime';
  });

  public formatGamma(value: number): string {
    const abs = Math.abs(value);
    if (abs >= 1000) {
      return (value / 1000).toFixed(1) + 'K';
    }
    return value.toFixed(0);
  }
}
